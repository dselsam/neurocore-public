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
from solver import Var, Lit
from nose.tools import assert_equals, assert_not_equal, assert_true, assert_raises
import gzip
import pickle

def test_var_idx():
    v = Var(3)
    assert_equals(3, v.idx())

def test_var_eq():
    v1 = Var(3)
    v2 = Var(3)
    assert_equals(v1, v2)

def test_var_neq():
    v1 = Var(3)
    v2 = Var(4)
    assert_not_equal(v1, v2)

def test_var_hash_eq():
    v1 = Var(3)
    v2 = Var(3)
    assert_equals(hash(v1), hash(v2))

def test_var_hash_neq():
    v1 = Var(3)
    v2 = Var(4)
    assert_not_equal(hash(v1), hash(v2))

def test_var_pickle():
    v1a = Var(3)
    v1b = Var(7)
    with gzip.GzipFile('.tmp', 'w') as f:
        pickle.dump((v1a, v1b), f)

    with gzip.GzipFile('.tmp', 'r') as f:
        v2a, v2b = pickle.load(f)
    assert_equals(v1a, v2a)
    assert_equals(v1b, v2b)

def test_lit_var_neg():
    v = Var(3)
    b = False
    l = Lit(v, b)
    assert_equals(l.var(), v)
    assert_equals(l.neg(), b)

def test_lit_flip():
    v = Var(3)
    b = False
    l = Lit(v, b).flip()
    assert_equals(l.var(), v)
    assert_equals(l.neg(), not b)

def test_lit_vidx():
    l = Lit(Var(3), False)
    assert_equals(l.vidx(100), 3)
    assert_equals(l.flip().vidx(100), 103)

def test_lit_pickle():
    l1a = Lit(Var(3), False)
    l1b = Lit(Var(5), True)
    with gzip.GzipFile('.tmp', 'w') as f:
        pickle.dump((l1a, l1b), f)

    with gzip.GzipFile('.tmp', 'r') as f:
        (l2a, l2b) = pickle.load(f)
    assert_equals(l1a, l2a)
    assert_equals(l1b, l2b)
