import subprocess
import threading
import collections
import inspect
import random
import tempfile
import numpy as np
import Pyro4
import os
import uuid
import auth
import pymysql
import time
import cloudpickle
import dill as pickle
import datetime

## misc
def flip(p=0.5): return random.random() < p

def is_small_enough(max_n_nodes, n_vars, n_clauses):
    return (2 * n_vars + n_clauses) < max_n_nodes

## system
def get_caller_linenum(offset=1):
    frame    = inspect.stack()[1 + offset]
    return os.path.basename(frame[0].f_code.co_filename), frame.lineno

def get_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')

def get_hostname(expensive=False):
    # Warning: memory usage spike from the fork, may not be worth constantly re-calling
    if expensive:
        return subprocess.check_output(['hostname']).strip().decode('ascii')
    else:
        return "<unknown>"

def timeit(f, *args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# mariadb
def _connect():
    while True:
        try:
            return pymysql.connect(host=auth.get_db_host(),
                                   user=auth.get_db_user(),
                                   password=auth.get_db_password(),
                                   db=auth.get_db_name(),
                                   cursorclass=pymysql.cursors.SSDictCursor)
        except:
            time.sleep(random.random())
            continue

def db_execute(cmd, values):
    conn = _connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute(cmd, values)
        conn.commit()
    finally:
        conn.close()

def de_numpify(v):
    if type(v) in [np.float32, np.float64]:
        return float(v)
    elif type(v) in [np.int32, np.int64]:
        return int(v)
    else:
        return v

def db_insert(table, **kvs):
    conn = _connect()
    keys         = ", ".join(["`%s`" % k for k in kvs])
    placeholders = ", ".join(["%s" for k in kvs])
    values       = tuple([kvs[k] for k in kvs])
    cmd          = "INSERT INTO `%s` (%s) VALUES (%s)" % (table, keys, placeholders)
    try:
        with conn.cursor() as cursor:
            cursor.execute(cmd, values)
            ident = cursor.lastrowid
        conn.commit()
        return ident
    finally:
        conn.close()

def db_update(table, where_kvs, set_kvs):
    conn = _connect()
    sets         = ",".join(["`%s` = %s" % (set_key, "%s") for set_key in set_kvs])
    wheres       = " AND ".join(["`%s` = %s" % (where_key, "%s") for where_key in where_kvs])
    values       = [set_kvs[set_key] for set_key in set_kvs] + [where_kvs[where_key] for where_key in where_kvs]
    cmd          = "UPDATE `%s` SET %s WHERE %s" % (table, sets, wheres)

    try:
        with conn.cursor() as cursor:
            cursor.execute(cmd, values)
        conn.commit()
    finally:
        conn.close()

def db_query_one(query):
    conn = _connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone()
    finally:
        conn.close()

def db_lookup_one(table, kvs):
    conn         = _connect()
    where_string = " AND ".join([("`%s`" % str(k)) + " = %s" for k in kvs])
    values       = tuple([kvs[k] for k in kvs])
    cmd          = "SELECT * FROM `%s` WHERE %s;" % (table, where_string)
    try:
        with conn.cursor() as cursor:
            cursor.execute(cmd, values)
            return cursor.fetchone()
    finally:
        conn.close()

def db_lookup_many(query):
    conn         = _connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return list(cursor.fetchall_unbuffered())
    finally:
        conn.close()

def db_insert_many(table, ks, vs, max_per_call=200):
    keys         = ", ".join(["`%s`" % k for k in ks])
    placeholders = ", ".join(["%s" for k in ks])

    # Not sure if this is necessary, but suspected that a very big insert caused a previous crash
    i = 0
    while i < len(vs):
        next_vs = vs[i:i+max_per_call]
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.executemany("INSERT INTO `%s` (%s) VALUES (%s)" % (table, keys, placeholders), next_vs)
            conn.commit()
        finally:
            conn.close()
        i += max_per_call

def log(msg, author=None, n_secs=None, kind='info', p=1.0, expensive=False):
    if random.random() > p: return None
    caller, linenum = get_caller_linenum()
    db_insert(table="events",
              hostname=get_hostname(expensive=expensive),
              caller=caller,
              linenum=linenum,
              pid=os.getpid(),
              tid=threading.current_thread().name,
              kind=kind,
              author=author,
              n_secs=n_secs,
              msg=msg)

#pyro4
def set_pyro_config(threadpool_size=None, threadpool_size_min=None):
    Pyro4.config.SERIALIZERS_ACCEPTED  = ["marshal", "json", "serpent", "pickle", "dill"]
    Pyro4.config.SERIALIZER            = "dill"
    # TODO(dselsam): it is WAY slower with compression
    Pyro4.config.COMPRESSION           = False
    Pyro4.config.DETAILED_TRACEBACK    = True
    # TODO(dselsam): these numbers need to be adjusted depending on the experiment
    if threadpool_size is not None:     Pyro4.config.THREADPOOL_SIZE     = threadpool_size
    if threadpool_size_min is not None: Pyro4.config.THREADPOOL_SIZE_MIN = threadpool_size_min

def pyro_default_ns_host():
    # Note: `persist` machine is running the name server
    return "10.1.1.4"

def pyro_locate(name, ns_host=None):
    if ns_host is None: ns_host = pyro_default_ns_host()
    with Pyro4.locateNS(host=ns_host) as ns:
        return ns.lookup(name)


# numpy
def np_top_k_idxs(arr, k):
    assert(k <= np.size(arr))
    return arr.argsort()[-k:][::-1]

def sample_unnormalized(qs):
    if np.isnan(qs).any():
        raise Exception("sample_unnormalized found nans: %s" % str(qs))
    qs  = np.array(qs)
    ps  = qs / np.sum(qs)
    idx = np.random.choice(len(ps), size=1, p=ps)[0]
    return idx, ps[idx]

def sample_unnormalized_dict(kvs):
    qs   = np.zeros(len(kvs))
    keys = list(kvs.keys())
    for i, key in enumerate(keys):
        qs[i] = kvs[key]

    idx, p = sample_unnormalized(qs)
    return keys[idx], p

def np_placeholder():
    return np.zeros(0)

###################### azure stuff
def enqueue(aqs, qname, elem):
    aqs.put_message(qname, elem)

def dequeue(aqs, qname, visibility_n_secs, wait_n_secs, logger=None):
    while True:
        messages = aqs.get_messages(qname, visibility_timeout=visibility_n_secs)
        if messages:
            assert(len(messages) == 1)
            return messages[0]
        else:
            time.sleep(wait_n_secs)

def to_blob(bbs, bcname, prefix, x):
    blob_name = "%s_%s" % (prefix, uuid.uuid4())
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, blob_name)
        with open(local_path, 'wb') as f:
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
        bbs.create_blob_from_path(bcname, blob_name, local_path)
    return blob_name

def from_blob(bbs, bcname, blob_name, delete):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, blob_name)
        bbs.get_blob_to_path(bcname, blob_name, local_path)
        if delete: bbs.delete_blob(bcname, blob_name)
        with open(local_path, 'rb') as f:
            return pickle.load(f)

def gd_scratch_bcname(gd_id): return "v2gdscratch%d" % gd_id
def gd_tfr_bcname(gd_id):     return "v2gdtfrs%d" % gd_id
def gd_tfr_dir(gd_id):        return "/home/dselsam/v2data/gd/id%d/tfrs" % gd_id

def checkpoint_bcname():           return "v2checkpoints"
def checkpoint_root_dir():         return "/home/dselsam/v2data/train/checkpoints"
def checkpoint_base_dir(train_id): return "train_id%d" % train_id
def checkpoint_dir(train_id):      return os.path.join(checkpoint_root_dir(), checkpoint_base_dir(train_id=train_id))

def load_checkpoint(bbs, sess, saver, train_id):
    checkpoint_info = db_lookup_one(table="train_checkpoints", kvs={'train_id', train_id})
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, checkpoint_info['bname'])
        bbs.get_blob_to_path(checkpoint_bcname(), checkpoint_info['bname'], local_path)
        subprocess.run(['tar', '-xzvf', local_path])
        saver.restore(sess, os.path.join(tmpdir, checkpoint_base_dir(train_id=train_id), "checkpoint"))
