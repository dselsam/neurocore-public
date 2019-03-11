import numpy as np
import tensorflow as tf
import os
import subprocess
import threading
import uuid
import random
import queue
import traceback
import util
import time
import tempfile
import Pyro4
import dill as pickle
import cloudpickle
import collections
from neurosat import apply_neurosat, NeuroSATParams, NeuroSATArgs, NeuroSATGuesses

@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class PyroQueryServer:
    def __init__(self, train_id, checkpoint):
        cfg             = util.db_lookup_one(table='train_runs', kvs={'train_id':train_id})
        self.n_vars     = tf.placeholder(tf.int64, shape=[])
        self.n_clauses  = tf.placeholder(tf.int64, shape=[])
        self.CL_idxs    = tf.placeholder(tf.int32, shape=[None, 2])

        params          = NeuroSATParams(cfg=cfg)
        self.guesses    = apply_neurosat(cfg=cfg, params=params,
                                         args=NeuroSATArgs(self.n_vars, self.n_clauses, self.CL_idxs))

        self.sess       = tf.Session()
        saver           = tf.train.Saver()

        saver.restore(self.sess, os.path.join(util.checkpoint_dir(train_id=train_id), checkpoint))
        util.log(kind='info', author='pyro-server', msg="restored from checkpoint")

        self.dummy_query()

    def dummy_query(self):
        print("Dummy query: ", self.sess.run(self.guesses, feed_dict={
            self.n_vars:3,
            self.n_clauses:2,
            self.CL_idxs:np.array([[0, 0], [0, 1], [0,2], [1, 3], [1, 4], [1, 5]])
        }))

    def query(self, args):
        try:
            guesses, n_secs_gpu = util.timeit(self.sess.run, self.guesses, feed_dict={
                self.n_vars:args.n_vars,
                self.n_clauses:args.n_clauses,
                self.CL_idxs:args.CL_idxs
            })
            return { "n_secs_gpu" : n_secs_gpu, "pi_core_var_logits" : guesses.pi_core_var_logits }
        except tf.errors.OpError as e:
            util.log(kind='error', author='pyro-server', msg=str(e))
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_gpus', action='store', type=int)

    parser.add_argument("--train_id", action="store", dest='train_id', type=int, default=21)
    parser.add_argument("--checkpoint", action="store", dest="checkpoint", type=str, default="model.ckpt-661581")

    parser.add_argument('--host', action='store', dest='host', type=str, default='0.0.0.0')
    parser.add_argument('--threadpool_size', action='store', dest='threadpool_size', type=int, default=64)
    parser.add_argument('--threadpool_size_min', action='store', dest='threadpool_size_min', type=int, default=32)
    opts = parser.parse_args()

    host_uuid = str(uuid.uuid4())

    def launch_server(gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        util.set_pyro_config(threadpool_size=opts.threadpool_size, threadpool_size_min=opts.threadpool_size_min)
        Pyro4.config.SERVERTYPE = "multiplex"
        try:
            util.log(kind='info', author='pyro-server-main', msg="about to serve")
            Pyro4.Daemon.serveSimple({ PyroQueryServer(train_id=opts.train_id, checkpoint=opts.checkpoint): "pyro_query_server" },
                                     host=opts.host, port=9092 + gpu_id, ns=False)
        except Exception as e:
            tb = traceback.format_exc()
            util.log(kind='error', author='pyro-server-main', msg="%s\n%s" % (str(e), tb))

    import multiprocessing
    actors = [multiprocessing.Process(target=launch_server, args=(gpu_id,)) for gpu_id in range(opts.n_gpus)]
    [actor.start() for actor in actors]
    [actor.join() for actor in actors]

if __name__ == "__main__":
    main()
