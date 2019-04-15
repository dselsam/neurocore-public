# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np
import math
import random
from tfutil import repeat_end, decode_transfer_fn, MLP
import tfutil
from collections import namedtuple

NeuroSATArgs    = namedtuple('NeuroSATArgs',    ['n_vars', 'n_clauses', 'CL_idxs'])
NeuroSATGuesses = namedtuple('NeuroSATGuesses', ['pi_core_var_logits'])

class NeuroSATParams:
    def __init__(self, cfg):
        if cfg['repeat_layers']:
            self.L_updates = [MLP(cfg, 2 * cfg['d'] + cfg['d'], repeat_end(cfg['d'], cfg['n_update_layers'], cfg['d']),
                                  name="L_u", nl_at_end=cfg['mlp_update_nl_at_end'])] * cfg['n_rounds']
            self.C_updates = [MLP(cfg, cfg['d'] + cfg['d'], repeat_end(cfg['d'], cfg['n_update_layers'], cfg['d']),
                                  name="C_u", nl_at_end=cfg['mlp_update_nl_at_end'])] * cfg['n_rounds']
        else:
            self.L_updates = [MLP(cfg, 2 * cfg['d'] + cfg['d'], repeat_end(cfg['d'], cfg['n_update_layers'], cfg['d']),
                                  name=("L_u_%d" % t), nl_at_end=cfg['mlp_update_nl_at_end']) for t in range(cfg['n_rounds'])]
            self.C_updates = [MLP(cfg, cfg['d'] + cfg['d'], repeat_end(cfg['d'], cfg['n_update_layers'], cfg['d']),
                                  name=("C_update_%d" % t), nl_at_end=cfg['mlp_update_nl_at_end']) for t in range(cfg['n_rounds'])]

        self.L_init_scale = tf.get_variable(name="L_init_scale", shape=[], initializer=tf.constant_initializer(1.0 / math.sqrt(cfg['d'])))
        self.C_init_scale = tf.get_variable(name="C_init_scale", shape=[], initializer=tf.constant_initializer(1.0 / math.sqrt(cfg['d'])))

        self.LC_scale = tf.get_variable(name="LC_scale", shape=[], initializer=tf.constant_initializer(cfg['LC_scale']))
        self.CL_scale = tf.get_variable(name="CL_scale", shape=[], initializer=tf.constant_initializer(cfg['CL_scale']))

        self.V_score  = MLP(cfg, 2 * cfg['d'], repeat_end(cfg['d'], cfg['n_score_layers'], 1), name=("V_score"), nl_at_end=False)

def apply_neurosat(cfg, params, args):
    n_vars, n_lits, n_clauses = args.n_vars, 2 * args.n_vars, args.n_clauses

    CL = tf.sparse_reorder(tf.SparseTensor(indices=tf.cast(args.CL_idxs, tf.int64),
                                           values=tf.ones(tf.shape(args.CL_idxs)[0]),
                                           dense_shape=[tf.cast(n_clauses, tf.int64), tf.cast(n_lits, tf.int64)]))

    L  = tf.ones(shape=[2 * n_vars, cfg['d']], dtype=tf.float32) * params.L_init_scale
    C  = tf.ones(shape=[n_clauses, cfg['d']], dtype=tf.float32) * params.C_init_scale

    LC = tf.sparse_transpose(CL)

    def flip(lits): return tf.concat([lits[n_vars:, :], lits[0:n_vars, :]], axis=0)

    for t in range(cfg['n_rounds']):
        C_old, L_old = C, L

        LC_msgs = tf.sparse_tensor_dense_matmul(CL, L, adjoint_a=False) * params.LC_scale
        C       = params.C_updates[t].forward(tf.concat([C, LC_msgs], axis=-1))
        C       = tf.check_numerics(C, message="C after update")
        C       = tfutil.normalize(C, axis=cfg['norm_axis'], eps=cfg['norm_eps'])
        C       = tf.check_numerics(C, message="C after norm")
        if cfg['res_layers']: C = C + C_old

        CL_msgs = tf.sparse_tensor_dense_matmul(LC, C, adjoint_a=False) * params.CL_scale
        L       = params.L_updates[t].forward(tf.concat([L, CL_msgs, flip(L)], axis=-1))
        L       = tf.check_numerics(L, message="L after update")
        L       = tfutil.normalize(L, axis=cfg['norm_axis'], eps=cfg['norm_eps'])
        L       = tf.check_numerics(L, message="L after norm")
        if cfg['res_layers']: L = L + L_old

    V         = tf.concat([L[0:n_vars, :], L[n_vars:, :]], axis=1)
    V_scores  = params.V_score.forward(V) # (n_vars, 1)

    return NeuroSATGuesses(pi_core_var_logits=tf.squeeze(V_scores))
