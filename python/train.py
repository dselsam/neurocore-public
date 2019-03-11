import os
import time
import tempfile
from queue import Queue
import argparse
import util
import math
import traceback
import uuid
import numpy as np
import tensorflow as tf
import random
import os
import threading
import json
import tfutil
from neurosat import *
from tftd import example_to_tftd
import subprocess
import collections

Stats    = collections.namedtuple('Stats', ['dp_id', 'cv_loss', 'cc_loss', 'l2_loss'])
StepKeys = Stats._fields + ('train_id', 'n_secs')

class TrainPostThread(threading.Thread):
    def __init__(self, cfg, q):
        threading.Thread.__init__(self)
        self.cfg = cfg
        self.q   = q

    def run(self):
        util.log(author='train-post-thread', msg='starting...')
        history = []
        while True:
            history.append(self.q.get())
            if len(history) == self.cfg['n_steps_per_log']:
                if random.random() < self.cfg['p_log']: util.db_insert_many(table='train_steps', ks=StepKeys, vs=history)
                del history[:]

def main(opts, cluster_spec, cfg):
    util.log(author='%s:%d' % (opts.job_name, opts.task_index), msg='starting @ %s' % util.get_hostname(expensive=True))
    cluster = tf.train.ClusterSpec(cluster_spec)
    server  = tf.train.Server(cluster, job_name=opts.job_name, task_index=opts.task_index)

    if opts.job_name == "ps":
        util.log(author='%s:%d' % (opts.job_name, opts.task_index), msg='joining server')
        server.join()
        raise Exception("Expecting server.join() to block forever")

    assert(opts.job_name == "worker")
    is_chief = (opts.task_index == 0)

    outqueue = Queue()
    train_post_thread = TrainPostThread(cfg, outqueue)
    train_post_thread.start()

    with tf.device("/job:ps/task:0"):
        params = NeuroSATParams(cfg=cfg)

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % opts.task_index, cluster=cluster)):
        filenames      = [os.path.join(util.gd_tfr_dir(gd_id=cfg['gd_id']), x) for x in os.listdir(util.gd_tfr_dir(gd_id=cfg['gd_id']))]
        dataset        = tf.data.TFRecordDataset(filenames=filenames, compression_type="GZIP", num_parallel_reads=cfg['n_parallel_reads'])
        # TODO(dselsam): don't hardcode the number of shards
        # idea: extend cluster_spec to map (job, task) -> (n_shards, shard_idx)
        dataset        = dataset.shard(num_shards=4, index=opts.task_index % 4)
        dataset        = dataset.map(example_to_tftd, num_parallel_calls=cfg['n_parallel_calls'])
        dataset        = dataset.filter(lambda tftd: 2 * tftd.n_vars + tftd.n_clauses < cfg['max_n_nodes'])
        dataset        = dataset.repeat()
        dataset        = dataset.prefetch(cfg['n_prefetch'])

        tftd           = dataset.make_one_shot_iterator().get_next()

        args           = NeuroSATArgs(n_vars=tftd.n_vars, n_clauses=tftd.n_clauses, CL_idxs=tftd.CL_idxs)
        guesses        = apply_neurosat(cfg=cfg, params=params, args=args)

        pi_v_targets   = tf.cast(tftd.core_var_mask, tf.float32)
        pi_v_targets   = pi_v_targets / tf.reduce_sum(pi_v_targets)

        pi_c_targets   = tf.cast(tftd.core_clause_mask, tf.float32)
        pi_c_targets   = pi_c_targets / tf.reduce_sum(pi_c_targets)

        cv_loss        = cfg['cv_loss_scale'] * tfutil.kldiv(logits=guesses.pi_core_var_logits, labels=pi_v_targets)
        cc_loss        = cfg['cc_loss_scale'] * tfutil.kldiv(logits=guesses.pi_core_clause_logits, labels=pi_c_targets)
        l2_loss        = cfg['l2_loss_scale'] * tfutil.build_l2_loss()
        loss           = cv_loss + cc_loss + l2_loss

        stats          = Stats(dp_id=tftd.dp_id, cv_loss=cv_loss, cc_loss=cc_loss, l2_loss=l2_loss)

        global_step    = tf.train.get_or_create_global_step()
        learning_rate  = tfutil.build_learning_rate(cfg, global_step)

        apply_grads    = tf.cond(tftd.is_train,
                                 lambda: tfutil.build_apply_gradients(cfg, loss, learning_rate, global_step),
                                 lambda: True)

    util.log(author='%s:%d' % (opts.job_name, opts.task_index), msg='creating session (train_id=%d)...' % cfg['train_id'])
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=util.checkpoint_dir(train_id=cfg['train_id'])) as mon_sess:
        util.log(author='%s:%d' % (opts.job_name, opts.task_index), msg='starting session loop')
        step = 0
        while True:
            try:
                (_, stats_v), n_secs = util.timeit(mon_sess.run, [apply_grads, stats])
                outqueue.put(tuple(map(util.de_numpify, stats_v)) + (cfg['train_id'], n_secs))
                step += 1
            except tf.errors.ResourceExhaustedError as e:
                tb = traceback.format_exc()
                util.log(kind='error', author='train', msg="RESOURCE_EXHAUSTED\n%s\n%s" % (str(e), tb))
                util.db_insert(table='tune_ooms', train_id=cfg['train_id'])
            except tf.errors.OpError as e:
                tb = traceback.format_exc()
                util.log(kind='error', author='train', msg="OP_ERROR\n%s\n%s" % (str(e), tb))
            except Exception as e:
                tb = traceback.format_exc()
                util.log(kind='error', author='train', msg="EXCEPTION\n%s\n%s" % (str(e), tb))
            except:
                tb = traceback.format_exc()
                util.log(kind='error', author='train', msg="UNKNOWN\n%s" % (tb))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name", action="store", type=str)
    parser.add_argument("task_index", action="store", type=int)
    parser.add_argument("--cfg_path", action="store", dest='cfg_path', type=str, default='/home/dselsam/neurocore/configs/train/train.json')
    parser.add_argument("--cluster_spec_path", action="store", dest='cluster_spec_path', type=str, default='/home/dselsam/neurocore/configs/train/cluster_spec.json')
    opts = parser.parse_args()
    with open(opts.cluster_spec_path, 'r') as f: cluster_spec = json.load(f)
    with open(opts.cfg_path, 'r') as f: cfg = json.load(f)
    assert(os.path.exists(util.gd_tfr_dir(gd_id=cfg['gd_id'])))

    if opts.job_name == "ps" and opts.task_index == 0:
        cfg['train_id'] = util.db_insert(table='train_runs', **cfg, git_commit=util.get_commit())
    else:
        time.sleep(5)
        cfg['train_id'] = util.db_query_one('select max(train_id) as train_id from train_runs')['train_id']

    try:
        main(opts, cluster_spec=cluster_spec, cfg=cfg)
    except tf.errors.OpError as e:
        tb = traceback.format_exc()
        util.log(kind='error', author='train', msg="OP_ERROR\n%s\n%s" % (str(e), tb))
    except Exception as e:
        tb = traceback.format_exc()
        util.log(kind='error', author='train', msg="EXCEPTION\n%s\n%s" % (str(e), tb))
    except:
        tb = traceback.format_exc()
        util.log(kind='error', author='train', msg="UNKNOWN\n%s" % (tb))
