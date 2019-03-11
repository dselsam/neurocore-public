import ray
import auth
import util
import solver
import time
import threading
import collections
import sutil
from queue import PriorityQueue, Queue
import traceback
import tempfile
import os
import uuid
import random
import numpy as np
import tensorflow as tf
from tftd import TFTD, tftd_to_example, tfd_to_tftd
from sutil import TFData
from azure.storage.blob import BlockBlobService
from azure.storage.queue import QueueService

TaskID     = collections.namedtuple('TaskID',     ['problem_id', 'node_id', 'node_depth', 'is_train'])
Task       = collections.namedtuple('Task',       ['id', 'bcnf'])
TaskResult = collections.namedtuple('TaskResult', ['btfds', 'new_bcnfs'])

class TaskIDCounter:
    def __init__(self):
        self.counters = {}

    def fresh_id(self, problem_id, is_train):
        assert(problem_id not in self.counters)
        self.counters[problem_id] = 1
        return TaskID(problem_id=problem_id, node_id=0, node_depth=0, is_train=is_train)

    def next_child_id(self, id):
        assert(id.problem_id in self.counters)
        node_id = self.counters[id.problem_id]
        self.counters[id.problem_id] += 1
        assert(id.node_id < node_id)
        return TaskID(problem_id=id.problem_id, node_id=node_id, node_depth=id.node_depth+1, is_train=id.is_train)

class TFTDWriter:
    def __init__(self, opts):
        self.opts           = opts
        self.n_files        = 0
        self.tmpdir         = tempfile.TemporaryDirectory()
        self._next_file()

    def _next_file(self):
        self.n_writes = 0
        self.outfile  = "file%d_%s.tfr" % (self.n_files, str(uuid.uuid4()))
        self.n_files += 1
        tfropts       = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)
        self.writer   = tf.io.TFRecordWriter(os.path.join(self.tmpdir.name, self.outfile), options=tfropts)

    def _move_file(self):
        self.writer.flush()
        self.writer.close()
        bbs = BlockBlobService(account_name=auth.store_name(), account_key=auth.store_key())
        bbs.create_blob_from_path(util.gd_tfr_bcname(gd_id=self.opts.gd_id),
                                  self.outfile, os.path.join(self.tmpdir.name, self.outfile))

    def finalize(self):
        util.log(kind='info', author='tfwriter', msg="finalize: %s" % str(self.n_writes))
        if self.n_writes > 0:
            util.log(kind='info', author='tfwriter', msg="moving last file #%d (%s)" % (self.n_files, self.outfile))
            self._move_file()

    def write_tftd(self, tftd):
        self.writer.write(tftd_to_example(tftd).SerializeToString())
        self.n_writes += 1
        if self.n_writes == self.opts.n_tfrs_per_file:
            util.log(kind='info', author='tfwriter', msg="file #%d ready (%s)" % (self.n_files, self.outfile))
            self._move_file()
            self._next_file()

def to_blob(opts, bbs, x, prefix="blob"):
    return util.to_blob(bbs, util.gd_scratch_bcname(gd_id=opts.gd_id), prefix=prefix, x=x)

def from_blob(opts, bbs, blob_name, delete):
    return util.from_blob(bbs, util.gd_scratch_bcname(gd_id=opts.gd_id), blob_name=blob_name, delete=delete)

def delete_blob(opts, bbs, x):
    return bbs.delete_blob(util.gd_scratch_bcname(gd_id=opts.gd_id), x)

@ray.remote(num_cpus=1, max_calls=1)
def gen_data_for(opts, task):
    bbs      = BlockBlobService(account_name=auth.store_name(), account_key=auth.store_key())
    sdimacs   = from_blob(opts, bbs, task.bcnf, delete=False)
    ctx       = solver.Context()
    s         = solver.deserialize(ctx, sutil.mk_opts(opts), sdimacs)

    btfds     = []
    new_bcnfs = []

    def push_tfd(tfd):
        btfds.append(to_blob(opts, bbs, sutil.tfd_to_py(tfd)))

    propagate_status = s.propagate()
    if propagate_status == solver.Status.UNKNOWN:
        s_pre_check = s.clone(ctx)
        assert(s_pre_check.propagate() == solver.Status.UNKNOWN)

        check_status, check_time = util.timeit(s.check)

        if check_status == solver.Status.SAT:
            pass
        elif check_status == solver.Status.UNSAT:
            push_tfd(s_pre_check.to_tf_data_with_core())
            [push_tfd(tfd) for tfd in s_pre_check.get_more_cores(ctx=ctx, max_tries=opts.find_max_tries, percent_to_keep=opts.find_percent_to_keep)]
        else:
            assert(check_status == solver.Status.UNKNOWN)
            s_pre_cube = s.clone(ctx)
            assert(s_pre_cube.propagate() == solver.Status.UNKNOWN)
            (cube_status, cubes), cube_time = util.timeit(s.cube)

            if cube_status == solver.Status.UNKNOWN:
                assert(len(cubes) in [1, 2])
                random.shuffle(cubes)
                for cube in cubes:
                    s_child = s.clone(ctx)
                    s_child.add(cube)
                    new_bcnfs.append(to_blob(opts, bbs, s_child.serialize()))

    return TaskResult(btfds=btfds, new_bcnfs=new_bcnfs)

def mk_query(opts, is_train):
    if is_train:
        return "SELECT problem_id, bcname, bname FROM sat_problems WHERE 2 * n_vars + n_clauses < %d AND bname NOT LIKE 'randkcnf%%' AND problem_id NOT IN (select problem_id from satcomp_info where year = 2018) ORDER BY n_clauses ASC LIMIT %d" % (opts.max_n_nodes_train, opts.limit)
    else:
        return "SELECT problem_id, bcname, bname FROM sat_problems WHERE  2 * n_vars + n_clauses > %d AND 2 * n_vars + n_clauses < %d AND bname NOT LIKE 'randkcnf%%' AND problem_id NOT IN (select problem_id from satcomp_info where year = 2018)) ORDER BY n_clauses ASC LIMIT %d" % (opts.max_n_nodes_train, opts.max_n_nodes_test, opts.limit)

def gen_all_data(opts):
    tftdw           = TFTDWriter(opts)
    tc              = TaskIDCounter()
    bbs             = BlockBlobService(account_name=auth.store_name(), account_key=auth.store_key())
    task_pq         = PriorityQueue()
    jobs            = []
    job_to_task     = {}

    setattr(opts, 'gd_id', util.db_insert(table='gd_runs', git_commit=util.get_commit(), wait_n_secs=opts.wait_n_secs,
                                          n_jobs_at_once=opts.n_jobs_at_once, n_tfrs_per_file=opts.n_tfrs_per_file,
                                          max_n_nodes_train=opts.max_n_nodes_train, max_n_nodes_test=opts.max_n_nodes_test,
                                          find_max_tries=opts.find_max_tries, find_percent_to_keep=opts.find_percent_to_keep,
                                          query_limit=opts.limit, timeout_ms=opts.timeout_ms))

    assert(not bbs.exists(util.gd_scratch_bcname(gd_id=opts.gd_id)))
    assert(not bbs.exists(util.gd_tfr_bcname(gd_id=opts.gd_id)))

    bbs.create_container(util.gd_scratch_bcname(gd_id=opts.gd_id))
    bbs.create_container(util.gd_tfr_bcname(gd_id=opts.gd_id))

    def launch_task(task):
        job = gen_data_for.remote(opts, task)
        jobs.append(job)
        job_to_task[job] = task

    def push_task(task, prio=None):
        if prio is None: prio = task.id.node_id
        task_pq.put_nowait((prio, task))

    def reload_jobs():
        while not task_pq.empty() and len(jobs) < opts.n_jobs_at_once:
            launch_task(task_pq.get_nowait()[1])

    def push_problems():
        util.log(author='push_problems', msg='starting')
        problem_infos = []
        for is_train in [True, False]:
            conn = util._connect()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(mk_query(opts=opts, is_train=is_train))
                    problem_infos.extend([(is_train, result) for result in list(cursor.fetchall_unbuffered())])
            finally:
                conn.close()
        util.log(author='push_problems', msg='found %d problems' % len(problem_infos))

        for is_train, info in problem_infos:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfilename = os.path.join(tmpdir, "%s.dimacs" % str(uuid.uuid4()))
                bbs.get_blob_to_path(info['bcname'], info['bname'], tmpfilename)
                s = solver.Solver(solver.Context(), solver.Options())
                s.from_file(tmpfilename)
                os.system('rm %s' % tmpfilename)
                task = Task(id=tc.fresh_id(info['problem_id'], is_train=is_train), bcnf=to_blob(opts, bbs, s.serialize()))
                assert(task.id.problem_id == info['problem_id'])
                push_task(task)

        util.log(author='push_problems', msg='pushed all problems')

    push_problems_thread = threading.Thread(target=push_problems, args=())
    push_problems_thread.start()

    def get_ready_job():
        while True:
            reload_jobs()
            if jobs:
                ready_jobs, _ = ray.wait(jobs, num_returns=1, timeout=opts.wait_n_secs)
                if ready_jobs:
                    job = ready_jobs[0]
                    jobs.remove(job)
                    assert(job in job_to_task)
                    task = job_to_task[job]
                    del job_to_task[job]
                    return job, task
            time.sleep(1)

    task_result_q = Queue()

    def process_task_result():
        while True:
            task, task_result = task_result_q.get()
            delete_blob(opts, bbs, task.bcnf)

            for btfd in task_result.btfds:
                tfd = from_blob(opts, bbs, btfd, delete=True)
                assert(tfd.n_vars > 0)
                assert(tfd.n_clauses > 0)
                dp_id = util.db_insert(table='gd_dps',
                                       gd_id=opts.gd_id, problem_id=task.id.problem_id, node_id=task.id.node_id, node_depth=task.id.node_depth, is_train=task.id.is_train,
                                       n_vars=tfd.n_vars, n_clauses=tfd.n_clauses, n_cells=np.shape(tfd.CL_idxs)[0],
                                       percent_vars_in_core=float(np.mean(tfd.core_var_mask.astype(np.float32))),
                                       percent_clauses_in_core=float(np.mean(tfd.core_clause_mask.astype(np.float32))))
                tftdw.write_tftd(tftd=tfd_to_tftd(dp_id=dp_id, is_train=task.id.is_train, tfd=tfd))

    process_results_thread = threading.Thread(target=process_task_result, args=())
    process_results_thread.start()

    try:
        while True:
            job, task = get_ready_job()
            try:
                task_result = ray.get(job)
            except Exception as e:
                tb = traceback.format_exc()
                util.log(kind='error', author='remote-worker', msg="TASK-ID: %s\n%s\n%s" % (str(task.id), str(e), tb))
                push_task(task, prio=1000000)
                continue

            if task_result.new_bcnfs:
                child_ids = [tc.next_child_id(task.id) for _ in task_result.new_bcnfs]
                for child_id, child_bcnf in zip(child_ids, task_result.new_bcnfs):
                    push_task(Task(id=child_id, bcnf=child_bcnf))

            task_result_q.put((task, task_result))

    except Exception as e:
        tb = traceback.format_exc()
        util.log(kind='error', author='master', msg="FAILING\n%s\n%s" % (str(e), tb))
        print("Exception: ", e)
        print("Failing...")
    finally:
        print("Finally...")
        util.log(kind='info', author='master', msg="finalizing")
        tftdw.finalize()
        util.log(kind='info', author='master', msg="deleting scratch blob container")
        bbs.delete_container(util.gd_scratch_bcname(gd_id=opts.gd_id))
        util.log(kind='info', author='master', msg="finished")
        print("All done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait_n_secs', action='store', dest='wait_n_secs', type=int, default=0.05)
    parser.add_argument('--n_jobs_at_once', action='store', dest='n_jobs_at_once', type=int, default=510)
    parser.add_argument('--n_tfrs_per_file', action='store', dest='n_tfrs_per_file', type=int, default=5000)
    parser.add_argument('--max_n_nodes_train', action='store', dest='max_n_nodes_train', type=int, default=300000)
    parser.add_argument('--max_n_nodes_test', action='store', dest='max_n_nodes_test', type=int, default=300000)
    parser.add_argument('--limit', action='store', dest='limit', type=int, default=1000000)
    parser.add_argument('--timeout_ms', action='store', dest='timeout_ms', type=int, default=60000)
    parser.add_argument('--find_max_tries', action='store', dest='find_max_tries', type=int, default=10)
    parser.add_argument('--find_percent_to_keep', action='store', dest='find_percent_to_keep', type=int, default=0.99995)
    parser.add_argument('--redis_address', action='store', dest='redis_address', type=str, default=None)
    opts = parser.parse_args()

    util.log(kind='error', author='master', msg="joining ray @ %s" % opts.redis_address)
    ray.init(redis_address=opts.redis_address)

    gen_all_data(opts=opts)
