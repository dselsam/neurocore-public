# NeuroCore

We train a simplified NeuroSAT architecture to directly predict the
unsatisfiable cores of real problems, and modify MiniSat to
periodically replace its variable activity scores with NeuroSAT's
prediction of how likely they are to appear in an unsatisfiable
core. Our modified MiniSat solves 10% more problems on SAT-COMP 2018
than the original does. Although MiniSat is no longer considered a
state-of-the-art solver, our results nonetheless demonstrate the
potential for NeuroSAT (and in particular, NeuroCore) to provide
useful guidance to CDCL solvers on real problems.

More information can be found in the paper [https://arxiv.org/abs/1903.04671](https://arxiv.org/abs/1903.04671).

## Overview of repository

1. `solver/`: pybind11 module wrapping Z3, for generating training data.
2. `python/gen_data.py`: ray script to generate training data.
3. `python/neurosat.py`: simplified reimplementation of NeuroSAT.
4. `python/train.py`: distributed tensorflow script to train NeuroCore.
5. `hybrids/glucose/`: code to extend glucose with NeuroCore.
6. `python/evaluate_cdcl.py`: ray script for evaluating solvers on benchmarks.
7. `weights/`: the trained weights used for the experiments in the paper.

## Dependencies

There are several dependencies. For now, we list the ones that cannot be easily installed with `apt-get` or `pip3`:

1. z3 (we used commit 773c61369480b6f031eb8fa98a7eb24bd52c7070)
2. drat-trim
3. eigen
4. pybind11
5. minisat
6. grpc

The code also uses a MariaDB database and an Azure storage account. Replace the fake credentials in `auth.json`.

## Team

* [Daniel Selsam](https://dselsam.github.io), Microsoft Research, Stanford University
* [Nikolaj Bj&#248;rner](https://www.microsoft.com/en-us/research/people/nbjorner/), Microsoft Research
