import numpy as np
import tensorflow as tf
from collections import namedtuple

def decode_transfer_fn(transfer_fn):
    if transfer_fn == "relu": return tf.nn.relu
    elif transfer_fn == "relu6": return tf.nn.relu6
    elif transfer_fn == "tanh": return tf.nn.tanh
    elif transfer_fn == "sig": return tf.nn.sigmoid
    elif transfer_fn == "elu": return tf.nn.elu
    else:
        raise Exception("Unsupported transfer function %s" % transfer_fn)

def repeat_end(val, n, k):
    return [val for i in range(n)] + [k]

def build_l2_loss():
    l2_loss = tf.zeros([])
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)
    return l2_loss

def build_learning_rate(cfg, global_step):
    lr = cfg['learning_rate']
    if type(lr) is float:
        return tf.constant(lr)
    elif lr['kind'] == "poly":
        return tf.train.polynomial_decay(learning_rate=lr['start'],
                                         global_step=global_step,
                                         end_learning_rate=lr['end'],
                                         decay_steps=lr['decay_steps'],
                                         power=lr['power'])
    elif lr['kind'] == "exp":
        return tf.train.exponential_decay(learning_rate=lr['start'],
                                          global_step=global_step,
                                          decay_steps=lr['decay_steps'],
                                          decay_rate=lr['decay_rate'],
                                          staircase=False)
    else:
        raise Exception("lr_decay_type must be 'none', 'poly' or 'exp'")

def build_apply_gradients(cfg, loss, learning_rate, global_step):
    optimizer      = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gs, vs         = zip(*optimizer.compute_gradients(loss))
    gs             = [tf.clip_by_value(g, clip_value_min=-cfg['clip_val_val'], clip_value_max=cfg['clip_val_val']) for g in gs]
    gs, _          = tf.clip_by_global_norm(gs, cfg['clip_norm_val'])

    apply_grads    = optimizer.apply_gradients(zip(gs, vs), name='apply_gradients', global_step=global_step)

    return apply_grads

def normalize(x, axis, eps):
    mean, variance = tf.nn.moments(x, axes=[axis], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=eps)

class MLP(object):
    def __init__(self, cfg, d_in, d_outs, name, nl_at_end):
        self.cfg = cfg
        self.name = name
        self.transfer_fn = decode_transfer_fn(cfg['mlp_transfer_fn'])
        self.nl_at_end = nl_at_end
        self._init_weights(d_in, d_outs)

    def _init_weights(self, d_in, d_outs):
        self.ws = []
        self.bs = []

        d = d_in

        with tf.variable_scope(self.name) as scope:
            for i, d_out in enumerate(d_outs):
                with tf.variable_scope('%d' % i) as scope:
                    if self.cfg['weight_reparam']:
                        w = tf.get_variable(name="w", shape=[d, d_out], initializer=tf.contrib.layers.xavier_initializer())
                        g = tf.get_variable(name="g", shape=[1, d_out], initializer=tf.ones_initializer())
                        self.ws.append(tf.nn.l2_normalize(w, axis=0) * tf.tile(g, [d, 1]))
                    else:
                        self.ws.append(tf.get_variable(name="w", shape=[d, d_out], initializer=tf.contrib.layers.xavier_initializer()))

                    self.bs.append(tf.get_variable(name="b", shape=[d_out], initializer=tf.zeros_initializer()))
                d = d_out

    def forward(self, z):
        x = z
        for i in range(len(self.ws)):
            x = tf.matmul(x, self.ws[i]) + self.bs[i]
            if self.nl_at_end or i + 1 < len(self.ws):
                x = self.transfer_fn(x)
        return x

def kldiv(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels) \
        + tf.reduce_sum(labels * tf.math.log(labels + 1e-8))
