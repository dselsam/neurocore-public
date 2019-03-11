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
from solver import *
from nose.tools import assert_equals, assert_not_equal, assert_true, assert_raises
import tempfile
import numpy as np
import os
import random
import time

problems = {
    "unsat1" : """p cnf 1 2
1 0
-1 0
""",

    "sat1" : """p cnf 3 2
1 2 3 0
-1 -2 -3 0
""",

    "red1" : """p cnf 5 4
-4 1 0
5 2 3 4 0
-5 -2 -3 4 0
-1 0
""",

    # rivest with extras
    # (the 5-clauses are extras)
    "core1" : """p cnf 5 10
1 2 -3 0
1 2 3 4 5 0
2 3 -4 0
3 4 1 0
4 -1 2 0
-1 -2 3 0
-2 -3 4 0
-3 -4 -1 0
-1 -2 3 -4 -5 0
-4 1 -2 0
""",

    # rivest with extras after propagate
    "core2" : """p cnf 6 11
1 2 -3 6 0
1 2 3 4 5 0
2 3 -4 6 0
3 4 1 6 0
4 -1 2 6 0
-1 -2 3 6 0
-2 -3 4 6 0
-3 -4 -1 6 0
-1 -2 3 -4 -5 -6 0
-4 1 -2 6 0
-6 0
""",

    "2cores" : """p cnf 8 16
1 2 -3 0
2 3 -4 0
3 4 1 0
4 -1 2 0
-1 -2 3 0
-2 -3 4 0
-3 -4 -1 0
-4 1 -2 0
5 6 -7 0
6 7 -8 0
7 8 5 0
8 -5 6 0
-5 -6 7 0
-6 -7 8 0
-7 -8 -5 0
-8 5 -6 0
"""
}

def new_solver(problem, timeout_ms=1000000):
    opts = Options()
    opts.timeout_ms = timeout_ms
    ctx = Context()
    s = Solver(ctx, opts)
    s.from_string(problems[problem])
    return s, ctx

def test_Solver_basic_unsat():
    s = new_solver("unsat1")[0]
    assert_equals(s.check(), Status.UNSAT)

def test_Solver_basic_sat():
    s = new_solver("sat1")[0]
    assert_equals(s.check(), Status.SAT)

def check_to_tf_data(problem, expected):
    s, ctx = new_solver(problem)
    st = ", ".join(s.dimacs().split('\n'))
    s.propagate()
    tfq = s.to_tf_data()
    assert_equals(tfq.n_vars, expected['n_vars'])
    assert_equals(tfq.n_clauses, expected['n_clauses'])
    assert_equals(np.size(tfq.CL_idxs), np.size(expected['CL_idxs']))
    print(tfq.CL_idxs)
    print(expected['CL_idxs'])
    # not guaranteed to hold in general
    assert_true((tfq.CL_idxs == expected['CL_idxs']).all())

def test_to_tf_data():
    tests = [
        ("sat1", {
            'n_vars':3,
            'n_clauses':2,
            'CL_idxs': np.array([
                [0, 0], [0, 1], [0, 2],
                [1, 3], [1, 4], [1, 5]
            ])
        }),

        ("red1", {
            'n_vars':3,
            'n_clauses':2,
            'CL_idxs': np.array([
                [0, 0], [0, 1], [0, 2],
                [1, 3], [1, 4], [1, 5]
            ])
        })
    ]

    for problem, expected in tests:
        yield check_to_tf_data, problem, expected

def check_unsat_core(problem, expected_core_var_mask):
    s, ctx = new_solver(problem)
    s_clone = s.clone(ctx)
    assert_equals(s_clone.check(), Status.UNSAT)
    assert_equals(s.propagate(), Status.UNKNOWN)
    tfd = s.to_tf_data_with_core()
    assert_equals(tfd.core_var_mask.size, tfd.n_vars)
    assert_equals(np.sum(tfd.core_var_mask.astype(float)), np.sum(expected_core_var_mask.astype(float)))
    # neither need actually hold, because of permutations
    assert_true((tfd.core_var_mask == expected_core_var_mask).all())

def test_unsat_core():
    tests = [
        ("core1",
         np.array([True, True, True, True, False])),
        ("core2",
         np.array([True, True, True, True, False]))
    ]

    for problem, expected_core_var_mask in tests:
        yield check_unsat_core, problem, expected_core_var_mask

def test_clone():
    s0, ctx = new_solver('core2')
    s0.propagate()
    tfq0 = s0.to_tf_data()

    s1 = s0.clone(ctx)
    s1.check()
    tfq1 = s0.to_tf_data()

    assert_equals(tfq0.n_vars, tfq1.n_vars)
    assert_equals(tfq0.n_clauses, tfq1.n_clauses)
    assert_true((tfq0.CL_idxs == tfq1.CL_idxs).all())

def test_flip():
    s, ctx = new_solver("core1")
    v  = s.get_free_var(v_idx=0)
    assert_equals(v.ilit(), v.flip().flip().ilit())
    assert_equals(v.flip().ilit(), v.flip().flip().flip().ilit())
