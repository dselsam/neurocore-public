import collections
import numpy as np
import util
import tensorflow as tf

def _tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _tf_float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

TFTD = collections.namedtuple('TFTD', ['dp_id', 'is_train',
                                       'n_vars', 'n_clauses', 'CL_idxs',
                                       'core_var_mask', 'core_clause_mask'])

def tfd_to_tftd(dp_id, is_train, tfd):
    assert(0 < tfd.n_vars)
    assert(0 < tfd.n_clauses)
    assert(np.size(tfd.core_var_mask) == tfd.n_vars)
    assert(np.any(tfd.core_var_mask))
    assert(np.size(tfd.core_clause_mask) == tfd.n_clauses)
    assert(np.any(tfd.core_clause_mask))
    return TFTD(dp_id=dp_id,
                is_train=is_train,
                n_vars=tfd.n_vars,
                n_clauses=tfd.n_clauses,
                CL_idxs=tfd.CL_idxs,
                core_var_mask=tfd.core_var_mask,
                core_clause_mask=tfd.core_clause_mask)

def tftd_to_example(tftd):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'dp_id'            : _tf_int64_feature(tftd.dp_id),
                'is_train'         : _tf_int64_feature(np.int64(tftd.is_train)),
                'n_vars'           : _tf_int64_feature(tftd.n_vars),
                'n_clauses'        : _tf_int64_feature(tftd.n_clauses),
                'n_cells'          : _tf_int64_feature(np.shape(tftd.CL_idxs)[0]),
                'CL_idxs'          : _tf_bytes_feature(tftd.CL_idxs.astype(np.int32).tostring()),

                'core_var_mask'    : _tf_bytes_feature(tftd.core_var_mask.astype(np.int32).tostring()),
                'core_clause_mask' : _tf_bytes_feature(tftd.core_clause_mask.astype(np.int32).tostring())
            }))

def example_to_tftd(example):
    features = tf.parse_single_example(
        example,
        features={
            'dp_id'            : tf.io.FixedLenFeature([], dtype=tf.int64),
            'is_train'         : tf.io.FixedLenFeature([], dtype=tf.int64),
            'n_vars'           : tf.io.FixedLenFeature([], dtype=tf.int64),
            'n_clauses'        : tf.io.FixedLenFeature([], dtype=tf.int64),
            'n_cells'          : tf.io.FixedLenFeature([], dtype=tf.int64),
            'CL_idxs'          : tf.io.FixedLenFeature([], dtype=tf.string),

            'core_var_mask'    : tf.io.FixedLenFeature([], dtype=tf.string),
            'core_clause_mask' : tf.io.FixedLenFeature([], dtype=tf.string)
        })

    dp_id            = features['dp_id']
    is_train         = features['is_train']
    n_vars           = features['n_vars']
    n_clauses        = features['n_clauses']
    CL_idxs          = tf.reshape(tf.decode_raw(features['CL_idxs'], tf.int32), [features['n_cells'], 2])
    core_var_mask    = tf.cast(tf.reshape(tf.decode_raw(features['core_var_mask'], tf.int32), [features['n_vars']]), tf.bool)
    core_clause_mask = tf.cast(tf.reshape(tf.decode_raw(features['core_clause_mask'], tf.int32), [features['n_clauses']]), tf.bool)

    asserts          = [
        tf.Assert(0 < n_vars, [n_vars], name="ASSERT_n_vars_pos"),
        tf.Assert(0 < n_clauses, [n_clauses], name="ASSERT_n_clauses_pos"),
        tf.Assert(tf.equal(tf.size(core_var_mask), tf.cast(n_vars, tf.int32)), [core_var_mask], name="CORE_VARS_N_VARS"),
        tf.Assert(tf.reduce_any(core_var_mask), [core_var_mask], name="CORE_VARS_EXIST"),
        tf.Assert(tf.equal(tf.size(core_clause_mask), tf.cast(n_clauses, tf.int32)), [core_clause_mask], name="CORE_CLAUSES_N_CLAUSES"),
        tf.Assert(tf.reduce_any(core_clause_mask), [core_clause_mask], name="CORE_CLAUSES_EXIST")
    ]

    with tf.control_dependencies(asserts):
        return TFTD(dp_id=dp_id,
                    is_train=tf.cast(is_train, tf.bool),
                    n_vars=n_vars,
                    n_clauses=n_clauses,
                    CL_idxs=CL_idxs,
                    core_var_mask=core_var_mask,
                    core_clause_mask=core_clause_mask)


####################
def test_example_to_tftd():
    from nose.tools import assert_equals, assert_true
    import tempfile

    tfropts  = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)
    out_file = tempfile.NamedTemporaryFile()
    writer   = tf.io.TFRecordWriter(out_file.name, options=tfropts)

    N_EXAMPLES = 50

    def sample_tftd():
        n_vars    = np.random.randint(10000)
        n_clauses = np.random.randint(100000)
        n_cells   = np.random.randint(1000000)
        return TFTD(dp_id=np.random.randint(10000),
                    is_train=util.flip(0.5),
                    n_vars=n_vars,
                    n_clauses=n_clauses,
                    CL_idxs=np.random.randint(n_clauses, size=(n_cells, 2), dtype=np.int32),
                    core_var_mask=(np.random.randint(2, size=(n_vars), dtype=np.int32) < 1),
                    core_clause_mask=(np.random.randint(2, size=(n_clauses), dtype=np.int32) < 1))


    tftds = [sample_tftd() for _ in range(N_EXAMPLES)]
    for tftd in tftds:
        writer.write(tftd_to_example(tftd).SerializeToString())

    writer.close()

    dataset   = tf.data.TFRecordDataset([out_file.name], compression_type="GZIP")
    dataset   = dataset.map(example_to_tftd)
    next_tftd = dataset.make_one_shot_iterator().get_next()

    sess      = tf.Session()
    for i in range(N_EXAMPLES):
        tftd1 = tftds[i]
        tftd2 = sess.run(next_tftd)
        assert_equals(tftd1.dp_id, tftd2.dp_id)
        assert_equals(tftd1.is_train, tftd2.is_train)
        assert_equals(tftd1.n_vars, tftd2.n_vars)
        assert_equals(tftd1.n_clauses, tftd2.n_clauses)
        assert_true((tftd1.CL_idxs == tftd2.CL_idxs).all())
        assert_true((tftd1.core_var_mask == tftd2.core_var_mask).all())
        assert_true((tftd1.core_clause_mask == tftd2.core_clause_mask).all())
