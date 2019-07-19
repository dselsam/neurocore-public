import ray
import os
import subprocess
import cloudpickle
import itertools
import scipy
import tempfile
import uuid
import traceback
import dill as pickle
import random
import time
import auth
import util
from queue import Queue
import collections
import json
import numpy as np
import tensorflow as tf
from neurosat import apply_neurosat, NeuroSATParams, NeuroSATArgs
from azure.storage.blob import BlockBlobService

Task = collections.namedtuple('Task', ['eval_id', 'problem_id', 'solver_id', 'timeout_s'])

def call_solver(server, sinfo, dimacs, timeout_s, outfilename):
    assert(sinfo['mode'] in ['NONE', 'RAND', 'NEURO'])
    assert(sinfo['solver'] in ['GLUCOSE'])

    def build_glucose_cmd(server, sinfo, dimacs, timeout_s, outfilename):
        cmd = ['/home/dselsam/neurocore/hybrids/glucose/build/glucose', dimacs]
        cmd.append('-mode=%s' % sinfo['mode'].upper())
        cmd.append('-neuro-outfile=%s' % outfilename)
        cmd.append('-verb=0')
        cmd.append('-timeout-s=%d' % timeout_s)

        if sinfo['n_secs_pause'] is not None: cmd.append('-n-secs-pause=%d' % round(sinfo['n_secs_pause']))
        if sinfo['n_secs_pause_inc'] is not None: cmd.append('-n-secs-pause-inc=%d' % round(100 * sinfo['n_secs_pause_inc']))
        if sinfo['max_lclause_size'] is not None: cmd.append('-max-lclause-size=%d' % sinfo['max_lclause_size'])
        if sinfo['max_n_nodes_cells'] is not None: cmd.append('-max-n-nodes-cells=%d' % sinfo['max_n_nodes_cells'])
        if sinfo['itau'] is not None: cmd.append('-itau=%d' % round(sinfo['itau']))
        if sinfo['scale'] is not None: cmd.append('-scale=%d' % round(sinfo['scale']))

        if sinfo['call_if_too_big'] is not None and sinfo['call_if_too_big']: cmd.append('-call-if-too-big')
        elif sinfo['call_if_too_big'] is not None and not sinfo['call_if_too_big']: cmd.append('-no-call-if-too-big')

        if sinfo['mode'] == 'NEURO': cmd.append('-server=%s' % server)
        return cmd

    if   sinfo['solver'] == 'GLUCOSE':   cmd = build_glucose_cmd(server, sinfo, dimacs, timeout_s, outfilename)
    else:                                raise Exception("UNEXPECTED SOLVER")

    try:
        subprocess.run(cmd)
        with open(outfilename, 'r') as f:
            status, n_secs_user, n_calls, n_fails, n_secs_gpu = f.read().split(" ")
        return status, float(n_secs_user), int(n_calls), int(n_fails), float(n_secs_gpu)
    except subprocess.CalledProcessError as e:
        util.log(kind='error', author='eval-worker', msg="SUBPROCESS_ERROR:\n%s" % str(e))
        raise e
    except Exception as e:
        util.log(kind='error', author='eval-worker', msg="EXCEPTION:\n%s" % str(e))
        raise e

@ray.remote(num_cpus=1, max_calls=1)
def work(server, task):
    sinfo = util.db_lookup_one(table='eval_solvers', kvs={"solver_id" : task.solver_id})
    pinfo = util.db_lookup_one(table='sat_problems', kvs={"problem_id" : task.problem_id})
    bbs   = BlockBlobService(account_name=auth.store_name(), account_key=auth.store_key())
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, "%s.dimacs" % str(uuid.uuid4()))
        bbs.get_blob_to_path(pinfo['bcname'], pinfo['bname'], tmpfilename)
        outfilename = os.path.join(tmpdir, "results.out")
        status, n_secs_user, n_calls, n_fails, n_secs_gpu = call_solver(server=server, sinfo=sinfo, dimacs=tmpfilename,
                                                                        timeout_s=task.timeout_s,
                                                                        outfilename=outfilename)

    assert(status in ["UNSAT", "SAT", "UNKNOWN"])

    util.db_insert(table="eval_problems",
                   eval_id=task.eval_id,
                   problem_id=task.problem_id,
                   solver_id=task.solver_id,
                   timeout_s=task.timeout_s,
                   n_secs_user=n_secs_user,
                   n_calls=n_calls,
                   n_fails=n_fails,
                   n_secs_gpu=n_secs_gpu,
                   status=status)
    return None

def evaluate_cdcl(opts):
    eval_id = util.db_insert(table='eval_runs', train_id=opts.train_id, checkpoint=opts.checkpoint,
                             git_commit=util.get_commit(), n_gpus=opts.n_gpus, n_workers=opts.n_workers)
    tasks   = [Task(eval_id=eval_id, problem_id=task['problem_id'], solver_id=task['solver_id'], timeout_s=opts.timeout_s)
               for task in util.db_lookup_many(query=opts.task_query)]

    util.log(kind='info', author='eval-head', msg='found %d tasks' % len(tasks), expensive=True)

    q = Queue()
    [q.put(task) for task in tasks]

    servers           = ["%s:%s" % p
                         for p in itertools.product(["10.1.1.9%d" % i for i in range(2, 7)],
                                                    ["5005%d" % i for i in range(1, 5)])]

    util.log(kind='info', author='eval-head', msg="servers:\n%s" % str(servers))
    server_to_n_jobs  = { server : 0 for server in servers }
    job_to_server     = {}
    job_to_task       = {}
    jobs              = []

    while jobs or not q.empty():
        try:
            while len(jobs) < opts.n_workers and not q.empty():
                task                       = q.get()
                server                     = min(servers, key=(lambda server: server_to_n_jobs[server]))
                job                        = work.remote(server=server, task=task)
                server_to_n_jobs[server]  += 1
                job_to_server[job]         = server
                job_to_task[job]           = task
                jobs.append(job)

            if jobs:
                job  = ray.wait(jobs, num_returns=1)[0][0]
                task = job_to_task[job]
                del job_to_task[job]
                server_to_n_jobs[job_to_server[job]] -= 1
                del job_to_server[job]
                jobs.remove(job)

                try:
                    _ = ray.get(job)
                except Exception as e:
                    util.log(kind='error', author='eval-head', msg="RE-ENQUEUING TASK: %s" % str(task))
                    q.put(task)
                    tb = traceback.format_exc()
                    util.log(kind='error', author='eval-head', msg="ERROR: %s" % str(e))
                    util.log(kind='error', author='eval-head', msg="TRACEBACK: %s" % tb)
        except Exception as e:
            util.log(kind='error', author='eval-head', msg="OUTER-EXCEPTION: %s" % str(e))

DEFAULT_TASK_QUERY="""
select    problem_id, solver_id
from      sat_problems, eval_solvers
where     problem_id in (select satcomp_info.problem_id from satcomp_info where year = 2018)
and       solver_id = %d
"""

if __name__ == "__main__":
    print("Evaluate-CDCL!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('redis_address', action='store', type=str)
    parser.add_argument('--n_gpus', action='store', dest='n_gpus', type=int, default=20)
    parser.add_argument('--n_workers', action='store', dest='n_workers', type=int, default=510)
    parser.add_argument('--timeout_s', action='store', dest='timeout_s', type=int, default=5000)
    parser.add_argument('--task_query', action='store', dest='task_query', type=str, default=None)
    parser.add_argument("--train_id", action="store", dest='train_id', type=int, default=21)
    parser.add_argument("--checkpoint", action="store", dest="checkpoint", type=str, default="model.ckpt-661581")
    opts = parser.parse_args()

    print("Joining %s..." % opts.redis_address)
    ray.init(redis_address=opts.redis_address)

    if opts.task_query is None: setattr(opts, "task_query", DEFAULT_TASK_QUERY % opts.solver_id)

    print("task query: " , opts.task_query)

    # This outer try/catch is only designed to catch infrastructure problems,
    # e.g. errors loading from checkpoints.
    # Query errors will be caught inside query, and all worker errors will be caught
    # and continued from.
    try:
        print("Calling evaluate_cdcl...")
        evaluate_cdcl(opts=opts)
    except tf.errors.OpError as e:
        tb = traceback.format_exc()
        util.log(kind='error', author='eval-head', msg="FAILING\n%s\n%s" % (str(e), tb))
        print("OpError: ", e)
    except Exception as e:
        tb = traceback.format_exc()
        util.log(kind='error', author='eval-head', msg="FAILING\n%s\n%s" % (str(e), tb))
        print("Exception: ", e)
