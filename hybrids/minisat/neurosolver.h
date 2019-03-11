#pragma once

#include "minisat/core/Solver.h"
#include "minisat/simp/SimpSolver.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>

using Minisat::lbool;
using std::vector;
using std::string;
using std::pair;

class mstream {
  std::ostringstream m_strm;
 public:
  std::string str() const { return m_strm.str(); }
  template<typename T> mstream & operator<<(T const & t) { m_strm << t; return *this; }
};

class SolverException : public std::exception {
 private:
  std::string _msg;
 public:
  explicit SolverException(const std::string & msg): _msg(msg) {}
  explicit SolverException(const mstream & msg): _msg(msg.str()) {}
  virtual const char* what() const throw() override { return _msg.c_str(); }
};

enum class Status { UNKNOWN, UNSAT, SAT };

enum class NeuroSolverMode { NONE, NEURO, RAND_SAME, RAND_DIFF };

struct NeuroSATConfig {
  NeuroSolverMode mode;  
  float    n_secs_pause;
  unsigned max_lclause_size;
  unsigned max_n_nodes_cells;
  float    itau;
  float    scale;

NeuroSATConfig(NeuroSolverMode const & mode, float n_secs_pause, unsigned max_lclause_size,
	       unsigned max_n_nodes_cells, float itau, float scale):
  mode(mode),
    n_secs_pause(n_secs_pause),
    max_lclause_size(max_lclause_size),
    max_n_nodes_cells(max_n_nodes_cells),
    itau(itau),
    scale(scale) {}
};

struct NeuroSATArgs {
  unsigned         n_vars;
  unsigned         n_clauses;
  Eigen::MatrixXi  CL_idxs;
};

struct NeuroSATTranslateInfo {
  vector<int>      v_to_nv;
  vector<unsigned> nv_to_v;

  vector<int>      c_to_nc;
  vector<unsigned> nc_to_c;
};

struct NeuroSATGuesses {
  bool            success;
  float           n_secs_gpu;
  Eigen::ArrayXf  pi_core_var_logits;

NeuroSATGuesses(): success(false) {}

NeuroSATGuesses(float n_secs_gpu, Eigen::ArrayXf const & pi_core_var_logits):
  success(true), n_secs_gpu(n_secs_gpu), pi_core_var_logits(pi_core_var_logits) {}
};

NeuroSATGuesses neurosat_failed_to_guess();

typedef std::function<NeuroSATGuesses(NeuroSATArgs const &)> NeuroSATFunc;

struct NeuroSolverResults {
  Status status;
  float  n_secs_user, n_secs_call, n_secs_gpu;
NeuroSolverResults(Status status, float n_secs_user, float n_secs_call, float n_secs_gpu):
  status(status), n_secs_user(n_secs_user), n_secs_call(n_secs_call), n_secs_gpu(n_secs_gpu) {}
};

class NeuroSolver : public Minisat::SimpSolver {
 private:
  NeuroSATFunc                 _nfunc;
  NeuroSATConfig               _ncfg;
  float                        _n_secs_gpu{0.0};
  float                        _n_secs_call{0.0};

  clock_t                      _t_start;
  clock_t                      _t_last_neuro;
  
  bool                         _has_called{false};
  bool                         _original_too_big{false};

  Eigen::ArrayXf               _rand_same_vscores;
  
  pair<NeuroSATArgs, NeuroSATTranslateInfo> _prepare_for_neurosat();
  void _do_neuro_stuff();
  void _set_var_scores(Eigen::ArrayXf const & unscaled_logits);
  void _neuro_set_var_scores(NeuroSATTranslateInfo const & tinfo, NeuroSATGuesses const & guesses);

 protected:
  Minisat::Lit pickBranchLit() override;

 public:
  NeuroSolver(NeuroSATFunc const & nfunc, NeuroSATConfig const & ncfg);
  void from_file(string const & filename);
  NeuroSolverResults check_with_timeout_s(unsigned n_secs);
};
