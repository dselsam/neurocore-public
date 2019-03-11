import ray
import os
import cloudpickle
import itertools
import scipy
import tempfile
import uuid
import traceback
import Pyro4
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
import neurominisat
from azure.storage.blob import BlockBlobService

Task = collections.namedtuple('Task', ['eval_id', 'problem_id', 'solver_id', 'timeout_s'])

def build_neurosat_config(sinfo):
    def parse_mode(mode):
        if mode == "NONE":        return neurominisat.NeuroSolverMode.NONE
        elif mode == "NEURO":     return neurominisat.NeuroSolverMode.NEURO
        elif mode == "RAND_SAME": return neurominisat.NeuroSolverMode.RAND_SAME
        elif mode == "RAND_DIFF": return neurominisat.NeuroSolverMode.RAND_DIFF
        
    return neurominisat.NeuroSATConfig(
        mode=parse_mode(sinfo['mode']),
        n_secs_pause=(sinfo['n_secs_pause'] if sinfo['n_secs_pause'] is not None else 0.0),
        max_lclause_size=(sinfo['max_lclause_size'] if sinfo['max_lclause_size'] is not None else 0),
        max_n_nodes_cells=(sinfo['max_n_nodes_cells'] if sinfo['max_n_nodes_cells'] is not None else 0),
        itau=(sinfo['itau'] if sinfo['itau'] is not None else 0.0),
        scale=(sinfo['scale'] if sinfo['scale'] is not None else 0.0)
    )
     
@ray.remote(num_cpus=1, max_calls=1)
def work(server, task):
    util.log(kind='info', author='eval-work', msg='starting on %d:%d:%s' % (task.problem_id, task.solver_id, server))
    util.set_pyro_config()
    proxy = Pyro4.Proxy(server)
    util.log(kind='info', author='eval-work', msg='connected to %s' % server)

    def query(nm_args):
        try:
            args    = NeuroSATArgs(n_vars=nm_args.n_vars, n_clauses=nm_args.n_clauses, CL_idxs=nm_args.CL_idxs)
            guesses = proxy.query(args)
            if guesses is None:
                return neurominisat.neurosat_failed_to_guess()
            else:
                return neurominisat.NeuroSATGuesses(n_secs_gpu=guesses['n_secs_gpu'],
                                                    pi_core_var_logits=guesses['pi_core_var_logits'])
        except Exception as e:
            tb = traceback.format_exc()
            util.log(kind='error', author='query', msg="TASK: %s\n%s\n%s" % (str(task), str(e), tb))
            return neurominisat.neurosat_failed_to_guess()

    sinfo = util.db_lookup_one(table='eval_solvers', kvs={"solver_id" : task.solver_id})
    s     = neurominisat.NeuroSolver(func=query, cfg=build_neurosat_config(sinfo))
    pinfo = util.db_lookup_one(table='sat_problems', kvs={"problem_id" : task.problem_id})
    bbs   = BlockBlobService(account_name=auth.store_name(), account_key=auth.store_key())
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, "%s.dimacs" % str(uuid.uuid4()))
        bbs.get_blob_to_path(pinfo['bcname'], pinfo['bname'], tmpfilename)
        s.from_file(filename=tmpfilename)

    results  = s.check_with_timeout_s(timeout_s=task.timeout_s)
    util.db_insert(table="eval_problems",
                   eval_id=task.eval_id,
                   problem_id=task.problem_id,
                   solver_id=task.solver_id,
                   timeout_s=task.timeout_s,
                   n_secs_user=results.n_secs_user,
                   n_secs_call=results.n_secs_call,
                   n_secs_gpu=results.n_secs_gpu,                                      
                   status=str(results.status).split(".")[1])
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
                         for p in itertools.product(["PYRO:pyro_query_server@10.1.1.8%d" % i for i in range(1, 6)],
                                                    ["909%d" % i for i in range(2, 6)])]

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
    parser.add_argument('solver_id', action='store', type=int)    
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
