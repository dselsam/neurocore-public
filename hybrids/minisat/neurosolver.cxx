#include <errno.h>
#include "neurosolver.h"
#include "minisat/core/Dimacs.h"
#include "minisat/utils/System.h"
#include "minisat/mtl/Sort.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>

using std::cout;
using std::endl;
using Minisat::Var;
using Minisat::Lit;
using Minisat::l_False;
using Minisat::l_True;
using Minisat::l_Undef;

static Minisat::Solver* _static_solver;
static void SIGINT_interrupt(int) { _static_solver->interrupt(); }

static float lse(Eigen::ArrayXf const & x) {
  float x_star = x.maxCoeff();
  return x_star + log((x - x_star).exp().sum());
}

NeuroSolver::NeuroSolver(NeuroSATFunc const & nfunc, NeuroSATConfig const & ncfg):
  _nfunc(nfunc), _ncfg(ncfg) {
  _t_start = clock();
}

NeuroSATGuesses neurosat_failed_to_guess() {
  return NeuroSATGuesses();
}

void NeuroSolver::from_file(string const & filename) {
  gzFile in = gzopen(filename.c_str(), "rb");
  Minisat::parse_DIMACS(in, *this, true);
  gzclose(in);
}

struct ClauseSize_lt {
  Minisat::ClauseAllocator& ca;
  ClauseSize_lt(Minisat::ClauseAllocator& ca_) : ca(ca_) {}
  bool operator () (Minisat::CRef x, Minisat::CRef y) {
    return ca[x].size() < ca[y].size(); }
};

pair<NeuroSATArgs, NeuroSATTranslateInfo> NeuroSolver::_prepare_for_neurosat() {
  NeuroSATArgs          args;
  NeuroSATTranslateInfo tinfo;
  tinfo.v_to_nv.reserve(nVars());
  tinfo.nv_to_v.reserve(nVars());
  tinfo.c_to_nc.reserve(clauses.size() + learnts.size());
  tinfo.nc_to_c.reserve(clauses.size() + learnts.size());

  vector<bool> assigned(nVars(), false);
  int trail_limit = (trail_lim.size() == 0 ? trail.size() : trail_lim[0]);
  for (int i = 0; i < trail_limit; ++i) {
    assigned[Minisat::var(trail[i])] = true;
  }

  for (int v = 0; v < nVars(); ++v) {
    if (assigned[v] || isEliminated(v)) {
      tinfo.v_to_nv.push_back(-1);
    } else {
      tinfo.v_to_nv.push_back(tinfo.nv_to_v.size());
      tinfo.nv_to_v.push_back(tinfo.v_to_nv.size() - 1);
    }
  }

  args.n_vars = tinfo.nv_to_v.size();
  if (args.n_vars != (unsigned) nFreeVars()) {
    throw SolverException(mstream() << "args.n_vars: " << args.n_vars << " vs " << nFreeVars());
  }

  unsigned n_cells = 0;

  auto traverse_clause = [&](Minisat::Clause const & clause) {
    if ((2 * args.n_vars + tinfo.nc_to_c.size() + n_cells > _ncfg.max_n_nodes_cells)
	|| (clause.learnt() && (unsigned) clause.size() > _ncfg.max_lclause_size)) {
      tinfo.c_to_nc.push_back(-1);
    } else {
      for (int arg_idx = 0; arg_idx < clause.size(); ++arg_idx) {
	if (tinfo.v_to_nv[Minisat::var(clause[arg_idx])] != -1) {
	  n_cells++;
	}
      }
      tinfo.c_to_nc.push_back(tinfo.nc_to_c.size());
      tinfo.nc_to_c.push_back(tinfo.c_to_nc.size() - 1);
    }
  };

  for (int c_idx = 0; c_idx < clauses.size(); ++c_idx) { traverse_clause(ca[clauses[c_idx]]); }
  if (2 * args.n_vars + tinfo.nc_to_c.size() + n_cells < _ncfg.max_n_nodes_cells) {
    // only sort if it might make a difference
    Minisat::sort(learnts, ClauseSize_lt(ca));
  } else {
    _original_too_big=true;
  }

  for (int lc_idx = 0; lc_idx < learnts.size(); ++lc_idx) { traverse_clause(ca[learnts[lc_idx]]); }

  args.n_clauses = tinfo.nc_to_c.size();

  // populate CL_idxs
  args.CL_idxs = Eigen::MatrixXi(n_cells, 2);
  unsigned cell_idx = 0;

  for (unsigned nc_idx = 0; nc_idx < tinfo.nc_to_c.size(); ++nc_idx) {
    unsigned c_idx = tinfo.nc_to_c[nc_idx];
    Minisat::Clause const & clause = (c_idx < (unsigned) clauses.size()) ? ca[clauses[c_idx]] : ca[learnts[c_idx - clauses.size()]];
    for (int arg_idx = 0; arg_idx < clause.size(); ++arg_idx) {
      Lit lit  = clause[arg_idx];
      Var v    = Minisat::var(lit);
      int nv   = tinfo.v_to_nv[v];
      if (nv != -1) {
	args.CL_idxs(cell_idx, 0) = nc_idx;
	args.CL_idxs(cell_idx, 1) = Minisat::sign(lit) ? (nv + args.n_vars) : nv;
	cell_idx++;
      }
    }
  }
  if (n_cells != cell_idx) { throw SolverException("incorrect cell_idx, must be a bug"); }

  return { args, tinfo };
}

void NeuroSolver::_neuro_set_var_scores(NeuroSATTranslateInfo const & tinfo, NeuroSATGuesses const & guesses) {
  Eigen::ArrayXf pi_core_var_logits = _ncfg.itau * guesses.pi_core_var_logits;
  Eigen::ArrayXf pi_core_var_ps     = (pi_core_var_logits - lse(pi_core_var_logits)).exp();

  var_inc = 1.0;
  for (unsigned nv = 0; nv < tinfo.nv_to_v.size(); ++nv) {
    activity[tinfo.nv_to_v[nv]] = _ncfg.scale * tinfo.nv_to_v.size() * pi_core_var_ps(nv);
  }
  rebuildOrderHeap();
}

void NeuroSolver::_set_var_scores(Eigen::ArrayXf const & unscaled_logits) {
  Eigen::ArrayXf logits = unscaled_logits * _ncfg.itau;
  Eigen::ArrayXf ps = (logits - lse(logits)).exp();

  var_inc = 1.0;
  for (int v = 0; v < nVars(); ++v) {
    activity[v] = _ncfg.scale * nVars() * ps(v);
  }
  rebuildOrderHeap();
}

void NeuroSolver::_do_neuro_stuff() {
  if (_original_too_big) { return; }
  if (_has_called && ((double) (clock() - _t_last_neuro) / CLOCKS_PER_SEC) < _ncfg.n_secs_pause) { return; }

  switch (_ncfg.mode) {
  case NeuroSolverMode::NONE:
    return;
  case NeuroSolverMode::RAND_DIFF:
    _set_var_scores(Eigen::ArrayXf::Random(nVars()));
    return;
  case NeuroSolverMode::RAND_SAME:
    if (nVars() != _rand_same_vscores.size()) { _rand_same_vscores = Eigen::ArrayXf::Random(nVars()); }
    _set_var_scores(_rand_same_vscores);
    return;
  case NeuroSolverMode::NEURO:
    break;
  }
  
  auto args_info = _prepare_for_neurosat();

  if (_original_too_big) { return; }
  
  NeuroSATArgs const & args = args_info.first;
  NeuroSATTranslateInfo const & tinfo = args_info.second;

  clock_t t_neuro_start = clock();
  NeuroSATGuesses guesses;

  {
    pybind11::gil_scoped_acquire _acquire;
    guesses = _nfunc(args);
  }

  _has_called   = true;
  _t_last_neuro = clock();
  
  _n_secs_call += (double) (_t_last_neuro - t_neuro_start) / CLOCKS_PER_SEC;

  if (!guesses.success) { return; }

  _n_secs_gpu += guesses.n_secs_gpu;
  _neuro_set_var_scores(tinfo, guesses);
}

Minisat::Lit NeuroSolver::pickBranchLit() {
  _do_neuro_stuff();
  return Solver::pickBranchLit();
}

static Status lbool_to_status(lbool const & ret) {
  if (ret == l_False)     return Status::UNSAT;
  else if (ret == l_True) return Status::SAT;
  else                    return Status::UNKNOWN;
}

NeuroSolverResults NeuroSolver::check_with_timeout_s(unsigned n_secs) {
  pybind11::gil_scoped_release _release;
  
  Minisat::limitTime(n_secs);
  _static_solver = this;
  Minisat::sigTerm(SIGINT_interrupt);

  _do_neuro_stuff();

  Minisat::vec<Minisat::Lit> dummy;
  lbool ret = solveLimited(dummy);

  float n_secs_user = (double) (clock() - _t_start) / CLOCKS_PER_SEC;
  return NeuroSolverResults(lbool_to_status(ret), n_secs_user, _n_secs_call, _n_secs_gpu);
}
