/*
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
*/
#include <string>
#include <exception>
#include <Eigen/Dense>
#include "z3++.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <set>
#include <vector>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <sstream>

using std::vector;
using std::pair;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::set;

#define MAX_UINT 4294967295

struct z3_expr_hash { unsigned operator()(z3::expr const & e) const { return e.hash(); } };
struct z3_expr_eq { bool operator()(z3::expr const & e1, z3::expr const & e2) const { return z3::eq(e1, e2); } };

template<typename T>
using z3_expr_map = typename std::unordered_map<z3::expr, T, z3_expr_hash, z3_expr_eq>;
using z3_expr_set = typename std::unordered_set<z3::expr, z3_expr_hash, z3_expr_eq>;

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

struct Var {
  unsigned _idx;
  Var(unsigned idx): _idx(idx) {}
  unsigned idx() const { return _idx; }

  string repr() const { return string("Var(") + std::to_string(idx()) + ")"; }
  bool operator==(const Var & other) const { return idx() == other.idx(); }
  bool operator<(const Var & other) const { return idx() < other.idx(); }
  unsigned hash() const { return idx(); }
};

namespace std {
  template <> struct hash<Var> { std::size_t operator()(const Var& var) const { return var.hash(); }};
}

struct Lit {
  Var      _var;
  bool     _neg;
  Lit(Var var, bool neg): _var(var), _neg(neg) {}

  Var var() const { return _var; }
  bool neg() const { return _neg; }

  unsigned vidx(unsigned n_vars) const { return neg() ? var().idx() + n_vars : var().idx(); }
  Lit flip() const { return Lit(var(), !neg()); }

  string repr() const { return string("Lit(") + var().repr() + ", " + (neg() ? "1" : "0") + ")"; }
  bool operator==(const Lit & other) const { return var() == other.var() && neg() == other.neg(); }
  bool operator<(const Lit & other) const { return var() < other.var() || (var() == other.var() && neg() < other.neg()); }
  unsigned hash() const { return var().hash() + (neg() ? 100003 : 0); } // smallest 6-digit prime
};

namespace std {
  template <> struct hash<Lit> { std::size_t operator()(const Lit& lit) const { return lit.hash(); }};
}

enum class Assignment { POS, NEG, SKIP };

enum class Status { UNKNOWN, UNSAT, SAT };

static Status check_result_to_Status(z3::check_result const & cr) {
  switch (cr) {
  case z3::unknown: return Status::UNKNOWN;
  case z3::unsat: return Status::UNSAT;
  case z3::sat: return Status::SAT;
  }
  throw SolverException("check_result_to_Status: unexpected status from z3");
}

struct Options {
  unsigned timeout_ms{MAX_UINT};
  unsigned max_conflicts{MAX_UINT};
  unsigned restart_max{MAX_UINT};
  unsigned variable_decay{110};

  bool     override_incremental{false};
  bool     lookahead_simplify{false};
  bool     acce{false};

  unsigned z3_replay_timeout_scale{10};
  unsigned drat_trim_timeout_scale{200};
  string   logfilename{"/tmp/solver.log"};
};

class Context {
  z3::context _zctx;
public:
  Context() {}
  z3::context & z() { return _zctx; }
};

z3::expr mk_not(z3::expr const & lit) {
  if (lit.is_not()) {
    if (!lit.arg(0).is_const()) { throw SolverException(mstream() << "mk_not called on non-literal: " << lit); }
    return lit.arg(0);
  } else {
    if (!lit.is_const()) { throw SolverException(mstream() << "mk_not called on non-literal: " << lit); }
    return !lit;
  }
}

class Expr {
  z3::expr _zexpr;
public:
  Expr(z3::expr const & zexpr): _zexpr(zexpr) {}
  z3::expr const & z() const { return _zexpr; }

  Expr flip() const { return Expr(mk_not(z())); }
  Expr var() const { return z().is_not() ? flip() : Expr(z()); }
  bool is_neg() const { return z().is_not(); }

  int ilit() const {
    z3::expr var = z().is_not() ? z().arg(0) : z();
    if (!var.is_const()) { throw SolverException(mstream() << "ilit() called on non-literal: " << z()); }
    int ivar = var.decl().name().to_int();
    return z().is_not() ? -ivar : ivar;
  }

};

typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;


struct TFData {
  unsigned         n_vars;
  unsigned         n_clauses;
  Eigen::MatrixXi  CL_idxs;

  TFData(unsigned n_vars, unsigned n_clauses, Eigen::MatrixXi const & CL_idxs):
    n_vars(n_vars), n_clauses(n_clauses), CL_idxs(CL_idxs) {
    if (n_vars == 0) {
      throw SolverException("Creating TFData with no variables");
    } else if (n_clauses == 0) {
      throw SolverException("Creating TFData with no clauses");
    }
  }

  ArrayXb          core_var_mask;
  ArrayXb          core_clause_mask;
};

static void check_exists_and_nonempty(string const & filename) {
  std::ifstream f(filename);
  if (!f.good()) {
    throw SolverException(mstream() << "file '" << filename << "' does not exist");
  } else if (f.peek() == std::ifstream::traits_type::eof()) {
    throw SolverException(mstream() << "file '" << filename << "' is empty");
  }
}

static void system_throw(string const & cmd, string const & finally) {
  int ret = system(cmd.c_str());
  if (WEXITSTATUS(ret) != 0) {
    system(finally.c_str());
    throw SolverException(mstream() << "Executing '" << cmd << "' failed.");
  }
}

class Solver;

struct TFDataManager {
  z3::expr_vector   non_units;
  vector<z3::expr>  pruned_clauses;
  z3_expr_map<Var>  _non_unit_to_var;
  unsigned          n_cells{0};

  TFDataManager(Solver const & s);

  Var non_unit_to_var(z3::expr const & e) const {
    if (_non_unit_to_var.count(e)) {
      return _non_unit_to_var.at(e);
    } else {
      throw SolverException(mstream() << "expr not found in non_unit_to_var: " << e);
    }
  }

  Lit free_expr_to_lit(z3::expr const & e) const {
    if (e.is_not()) {
      return Lit(non_unit_to_var(e.arg(0)), true);
    } else if (!e.is_const()) {
      throw SolverException(mstream() << "zexpr_to_lit called on non-literal: " << e);
    } else {
      return Lit(non_unit_to_var(e), false);
    }
  }

  Eigen::MatrixXi to_CL_idxs() const {
    Eigen::MatrixXi CL_idxs = Eigen::MatrixXi(n_cells, 2);
    unsigned cell_idx = 0;
    for (unsigned c_idx = 0; c_idx < pruned_clauses.size(); ++c_idx) {
      for (unsigned arg_idx = 0; arg_idx < pruned_clauses[c_idx].num_args(); ++arg_idx) {
	CL_idxs(cell_idx, 0) = c_idx;
	CL_idxs(cell_idx, 1) = free_expr_to_lit(pruned_clauses[c_idx].arg(arg_idx)).vidx(non_units.size());
	cell_idx++;
      }
    }
    return CL_idxs;
  }
};

class Solver {
public:

  Options          _opts;
  z3::solver       _zsolver;
  unsigned         _orig_n_vars; // wrt original problem
  unsigned         _total_n_vars;

  void set_options() {
    z().set(":lookahead.reward", "march_cu");
    z().set(":lookahead.cube.cutoff", "depth");
    z().set(":lookahead.cube.depth", (unsigned)1);

    z().set(":max_conflicts", _opts.max_conflicts);
    z().set(":sat.restart.max", _opts.restart_max);
    z().set(":sat.variable_decay", _opts.variable_decay);

    z().set(":override_incremental", _opts.override_incremental);
    z().set(":lookahead_simplify", _opts.lookahead_simplify);
    z().set(":acce", _opts.acce);

    z().set(":sat.force_cleanup", true);
  }

  void init_post_from() {
    _total_n_vars = z().units().size() + z().non_units().size();
    _orig_n_vars = _total_n_vars;
  }

    void init_post_clone() {
      _total_n_vars = z().units().size() + z().non_units().size();
    }

  z3::solver & z() { return _zsolver; }
  z3::solver const & z() const { return _zsolver; }

public:
  Solver(Context & ctx, Options const & opts):
    _opts(opts), _zsolver(ctx.z(), "QF_FD")
  { set_options(); }

  Solver(Context & ctx, Options const & opts, Solver const & s):
    _opts(opts), _zsolver(ctx.z(), s.z(), z3::solver::translate()), _orig_n_vars(s._orig_n_vars)
  { set_options(); init_post_clone(); }

  Solver clone(Context & ctx) const { return Solver(ctx, _opts, *this); }
  void from_string(string const & s) {
    try {
      z().from_string(s.c_str());
    } catch (...) {
      throw SolverException("error in Solver::from_string(...)");
    }
    init_post_from();
  }

  void from_file(string const & filename) {
    try {
      z().from_file(filename.c_str());
    } catch (...) {
      throw SolverException("error in Solver::from_file(...)");
    }
    init_post_from();
  }

  string serialize() { return z().dimacs(); }
  string dimacs() { return z().dimacs(); }

  unsigned total_n_vars() const {
    return _total_n_vars;
  }

  void add(Expr const & expr) {
    z().add(expr.z());
  }

  Status propagate() {
    z().set(":max_conflicts", (unsigned) 0);
    z3::check_result cr = z().check();
    z().set(":max_conflicts", _opts.max_conflicts);
    return check_result_to_Status(cr);
  }

  Status check() {
    return check_with_timeout_ms(_opts.timeout_ms);
  }

  Status check_with_timeout_ms(unsigned timeout_ms) {
    z().set(":timeout", timeout_ms);
    z3::check_result result = z().check();
    z().set(":timeout", (unsigned) MAX_UINT);

    if (result == z3::unknown) {
      return propagate();
    } else {
      return check_result_to_Status(result);
    }
  }

private:
  void validate_cube(z3::expr_vector const & cube) {
    if (cube.size() == 0) {
      throw SolverException("z3::cube() returned empty cube");
    } else if (cube.size() > 1) {
      throw SolverException("unexpected cube of size > 1");
    }
  }

public:
  pair<Status, vector<Expr>> cube() {
    z3::solver::cube_generator cg    = z().cubes();
    z3::solver::cube_iterator  start = cg.begin();
    z3::solver::cube_iterator  end   = cg.end();

    if (start == end) { return { Status::UNSAT, {} }; }

    z3::expr_vector cube1 = *start;
    validate_cube(cube1);

    assert(cube1.size() == 1);
    if (cube1[0].is_true()) { return { Status::SAT, {} }; }

    ++start;
    if (start == end) { /* failed lit */ return { Status::UNKNOWN, { Expr(cube1[0]) } }; }

    z3::expr_vector cube2 = *start;
    validate_cube(cube2);

    if (cube1 == cube2) {
      throw SolverException("Both cubes are the same: did you forget to increment the iterator?");
    }

    ++start;
    if (start != end) {
      throw SolverException("z3::cube() returned more than two cubes");
    }
    return { Status::UNKNOWN, { Expr(cube1[0]), Expr(cube2[0]) } };
  }

  Expr get_free_var(unsigned v_idx) {
    // Warning: slow
    z3::expr_vector non_units = z().non_units();
    if (v_idx >= non_units.size()) { throw SolverException("v_idx too big!"); }
    z3::expr e = non_units[v_idx];
    return Expr(e);
  }

  TFData to_tf_data() const {
    //if (propagate() != Status::UNKNOWN) { throw SolverException("to_tf_data(): propagate does not return UNKNOWN"); }
    TFDataManager dm(*this);
    if (dm.non_units.size() == 0) { throw SolverException("to_tf_data() but no free variables"); }
    if (dm.pruned_clauses.size() == 0) { throw SolverException("to_tf_data() but no pruned clauses"); }
    return TFData(dm.non_units.size(), dm.pruned_clauses.size(), dm.to_CL_idxs());
  }

  TFData to_tf_data_with_core() const {
    TFDataManager dm(*this);
    TFData tfd(dm.non_units.size(), dm.pruned_clauses.size(), dm.to_CL_idxs());
    tfd.core_clause_mask = compute_core_clause_mask(dm.pruned_clauses);
    if ((unsigned) tfd.core_clause_mask.size() != dm.pruned_clauses.size()) {
      throw SolverException(mstream() << "error while computing core_clause_mask: wrong number of clauses: " << tfd.core_clause_mask.size() << " vs " << tfd.n_clauses);
    }

    tfd.core_var_mask = ArrayXb::Constant(tfd.n_vars, false);
    for (unsigned c_idx = 0; c_idx < dm.pruned_clauses.size(); ++c_idx) {
      if (tfd.core_clause_mask[c_idx]) {
	z3::expr const & clause = dm.pruned_clauses[c_idx];
	for (unsigned arg_idx = 0; arg_idx < clause.num_args(); ++arg_idx) {
	  tfd.core_var_mask[dm.free_expr_to_lit(clause.arg(arg_idx)).var().idx()] = true;
	}
      }
    }
    return tfd;
  }

  vector<z3::expr> get_pruned_clauses() const {
    z3::expr_vector zclauses = z().assertions();
    vector<z3::expr> clauses;
    for (unsigned c_idx = 0; c_idx < zclauses.size(); ++c_idx) {
      if (zclauses[c_idx].is_or()) { clauses.push_back(zclauses[c_idx]); }
    }
    return clauses;
  }

  Solver clauses_to_solver(Context & ctx, vector<z3::expr> const & clauses, unsigned n_takes) const {
    if (n_takes > clauses.size()) { throw SolverException("n_takes too high"); }
    Solver s(ctx, _opts);
    for (unsigned i = 0; i < n_takes; ++i) {
      s.add(Expr(clauses[i]));
    }
    return s;
  }

  vector<TFData> get_more_cores(Context & ctx, unsigned max_tries, float percent_to_keep) const {
    // Current solver must be in UNKNOWN state, but must have been determined to be UNSAT
    vector<z3::expr> clauses = get_pruned_clauses();

    std::srand(unsigned(std::time(0)));
    std::random_shuffle(clauses.begin(), clauses.end());

    vector<TFData> results;
    unsigned n_take = clauses.size();

    for (unsigned core_idx = 0; core_idx < max_tries; ++core_idx) {
      n_take               = floor(n_take * percent_to_keep);
      Solver s             = clauses_to_solver(ctx, clauses, n_take);
      Status status        = s.check();
      if (status == Status::UNSAT) {
	Solver s_unsat = clauses_to_solver(ctx, clauses, n_take);
	if (s_unsat.propagate() == Status::UNKNOWN) {
	  results.push_back(s_unsat.to_tf_data_with_core());
	}
      } else {
	break;
      }
    }
    return results;
  }

private:
  static unordered_map<unsigned, unsigned> build_name_map_from_dimacs(string const & filename) {
    unordered_map<unsigned, unsigned> m;
    std::ifstream file(filename);
    string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      string result;
      if (std::getline(iss, result, ' ')) {
	if (result != "c") continue;

	std::getline(iss, result, ' ');
	unsigned dimacs_var = stoi(result);

	std::getline(iss, result, '\n');
	string z3_var_name = result;

	if (z3_var_name.substr(0, 2) != "k!") { throw SolverException(mstream() << "found variable without k!<int> name: " << z3_var_name); }
	m.insert({stoi(z3_var_name.substr(2)), dimacs_var});
      }
    }
    return m;
  }

  vector<vector<unsigned>> pruned_clauses_rename_to_dimacs_vars(vector<z3::expr> const & pruned_clauses,
								unordered_map<unsigned, unsigned> const & name_map) const {
    vector<vector<unsigned>> vs;

    for (z3::expr const & clause : pruned_clauses) {
      if (clause.num_args() < 2) { throw SolverException(mstream() << "or-clause with 1 argument:" << clause); }
      vs.emplace_back();

      for (unsigned arg_idx = 0; arg_idx < clause.num_args(); ++arg_idx) {
	  z3::expr e   = clause.arg(arg_idx);
	  bool     neg = e.is_not();
	  e            = neg ? e.arg(0) : e;
	  if (!e.is_const()) { throw SolverException(mstream() << "expected constant: " << e); }
	  unsigned var = name_map.at(e.decl().name().to_int());
	  unsigned lit = neg ? (var + total_n_vars()) : var;
	  vs.back().push_back(lit);
	}
	std::sort(vs.back().begin(), vs.back().end());
    }
    return vs;
  }

  vector<vector<unsigned>> clauses_prune_to_dimacs_vars(z3::expr_vector const & clauses) const {
    vector<vector<unsigned>> vs;

    for (unsigned c_idx = 0; c_idx < clauses.size(); ++c_idx) {
      z3::expr clause = clauses[c_idx];
      if (clause.is_or()) {
	if (clause.num_args() < 2) { throw SolverException(mstream() << "or-clause with 1 argument: " << clause); }
	vs.emplace_back(); // prune the clauses
	for (unsigned arg_idx = 0; arg_idx < clause.num_args(); ++arg_idx) {
	  z3::expr e   = clause.arg(arg_idx);
	  bool     neg = e.is_not();
	  e            = neg ? e.arg(0) : e;
	  if (!e.is_const()) { throw SolverException(mstream() << "expected constant: " << e); }
	  unsigned var = e.decl().name().to_int();
	  unsigned lit = neg ? (var + total_n_vars()) : var;
	  vs.back().push_back(lit);
	}
	std::sort(vs.back().begin(), vs.back().end());
      }
    }
    return vs;
  }

  string build_z3_drat_cmd(string const & drat_name, string const & dimacs_name) const {
    int buffer_size = 1000;
    char buffer[buffer_size];
    int cx = snprintf(buffer,
		      buffer_size,
		      "z3 -T:%u sat.max_conflicts=%u sat.restart.max=%u sat.variable_decay=%u sat.override_incremental=%s sat.acce=%s sat.lookahead_simplify=%s sat.drat.file=%s %s >> %s",
		      (_opts.z3_replay_timeout_scale * _opts.timeout_ms) / 1000,
		      _opts.max_conflicts,
		      _opts.restart_max,
		      _opts.variable_decay,
		      _opts.override_incremental ? "true" : "false",
		      _opts.acce ? "true" : "false",
		      _opts.lookahead_simplify ? "true" : "false",
		      drat_name.c_str(),
		      dimacs_name.c_str(),
		      _opts.logfilename.c_str());
    if (cx >= 0 && cx < buffer_size) {
      return string(buffer);
    } else {
      throw SolverException("z3 command too big for buffer!");
    }
  }

  string build_drat_trim_cmd(string const & dimacs_name, string const & drat_name, string const & core_name) const {
    int buffer_size = 1000;
    char buffer[buffer_size];
    int cx = snprintf(buffer,
		      buffer_size,
		      "drat-trim %s %s -c %s -t %u >> %s",
		      dimacs_name.c_str(),
		      drat_name.c_str(),
		      core_name.c_str(),
		      (_opts.drat_trim_timeout_scale * _opts.timeout_ms) / 1000,
		      _opts.logfilename.c_str());
    if (cx >= 0 && cx < buffer_size) {
      return string(buffer);
    } else {
      throw SolverException("drat-trim command too big for buffer!");
    }
  }

public:
ArrayXb compute_core_clause_mask(vector<z3::expr> const & pruned_clauses) const {

    string prefix      = std::tmpnam(nullptr);
    string dimacs_name = prefix + ".dimacs";
    string drat_name   = prefix + ".drat";
    string core_name   = prefix + ".core.dimacs";

    std::ofstream dimacs_out(dimacs_name, std::ios_base::out);
    dimacs_out << z().dimacs();
    dimacs_out.close();

    string debug_rm_cmd = string("rm -f ") + drat_name + " " + core_name;
    string rm_cmd = string("rm -f ") + drat_name + " " + core_name + " " + dimacs_name;
    string z3_cmd = build_z3_drat_cmd(drat_name, dimacs_name);

    std::ofstream flog(_opts.logfilename, std::ios_base::app);
    flog << "Executing '" << z3_cmd << "'..." << std::endl;

    system_throw(z3_cmd, debug_rm_cmd);
    check_exists_and_nonempty(drat_name);

    string drat_trim_cmd = build_drat_trim_cmd(dimacs_name, drat_name, core_name);
    flog << "Executing '" << drat_trim_cmd << "'..." << std::endl;
    flog.close();

    system_throw(drat_trim_cmd, debug_rm_cmd);
    check_exists_and_nonempty(core_name);

    z3::solver s_core(z().ctx(), "QF_FD");
    s_core.from_file(core_name.c_str());

    z3::expr_vector clauses_core = s_core.assertions();

    unordered_map<unsigned, unsigned> name_map = build_name_map_from_dimacs(dimacs_name);
    vector<vector<unsigned>>          oclauses = pruned_clauses_rename_to_dimacs_vars(pruned_clauses, name_map); // old
    vector<vector<unsigned>>          nclauses = clauses_prune_to_dimacs_vars(clauses_core); // new

    // oclauses_permutation[k] is the index in the original oclauses of sorted oclauses[k].
    vector<unsigned> oclauses_permutation;
    for (unsigned i = 0 ; i < oclauses.size(); ++i) { oclauses_permutation.push_back(i); }
    std::sort(oclauses_permutation.begin(),
	      oclauses_permutation.end(),
	      [&](const unsigned & i1, const unsigned & i2) { return (oclauses[i1] < oclauses[i2]); });

    std::sort(oclauses.begin(), oclauses.end());
    std::sort(nclauses.begin(), nclauses.end());

    // construct core_clause_mask
    ArrayXb core_clause_mask = ArrayXb::Constant(oclauses.size(), false);

    unsigned o_idx = 0; // old_idx
    for (unsigned n_idx = 0; n_idx < nclauses.size(); ++n_idx) {
      while (oclauses[o_idx] != nclauses[n_idx]) {
	o_idx++;
	if (o_idx >= oclauses.size()) { throw SolverException(mstream() << "core clause not present in original clauses"); }
      }
      assert(oclauses[o_idx] == nclauses[n_idx]);
      core_clause_mask[oclauses_permutation[o_idx]] = true;
    }

    system_throw(rm_cmd, "echo 'no finally'");
    return core_clause_mask;
  }

private:
  vector<z3::expr> collect_pruned_clauses() const {
    // Warning: slow
    z3::expr_vector clauses = z().assertions();
    vector<z3::expr> pruned_clauses;

    for (unsigned c_idx = 0; c_idx < clauses.size(); ++c_idx) {
      z3::expr const & clause = clauses[c_idx];
      if (clause.is_or()) {
	if (clause.num_args() < 2) { throw SolverException(mstream() << "cleaned clause has singleton or: " << clause); }
	pruned_clauses.push_back(clause);
      }
    }
    return pruned_clauses;
  }

public:
  void set_activity_scores(Eigen::ArrayXf const & pi, unsigned pi_scale) {
    z3::expr_vector non_units = z().non_units();
    for (unsigned i = 0; i < non_units.size(); ++i) {
      double act = pi_scale * pi[i];
      z().set_activity(non_units[i], act);
    }
  }

};

Solver deserialize(Context & ctx, Options const & opts, string const & serial) {
  Solver s(ctx, opts);
  s.from_string(serial);
  return s;
}

TFDataManager::TFDataManager(Solver const & s): non_units(s.z().non_units()) {
  for (unsigned i = 0; i < non_units.size(); ++i) {
    _non_unit_to_var.insert({non_units[i], Var(i)});
  }
  pruned_clauses = s.get_pruned_clauses();
  for (z3::expr const & clause : pruned_clauses) {
    n_cells += clause.num_args();
  }
}

// Pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(solver, m) {
  py::register_exception<SolverException>(m, "SolverException");

  py::class_<Var>(m, "Var")
    .def(py::init<unsigned>(), py::arg("idx"))
    .def("idx", &Var::idx)
    .def("__str__", &Var::repr)
    .def("__repr__", &Var::repr)
    .def("__eq__", &Var::operator==)
    .def("__lt__", &Var::operator<)
    .def("__hash__", &Var::hash)
    .def(py::pickle([](const Var & var) { return var.idx(); },
		    [](unsigned idx) { return Var(idx); }));

  py::class_<Lit>(m, "Lit")
    .def(py::init<Var, bool>(), py::arg("var"), py::arg("sign"))
    .def("var", &Lit::var)
    .def("neg", &Lit::neg)
    .def("vidx", &Lit::vidx)
    .def("flip", &Lit::flip)
    .def("__str__", &Lit::repr)
    .def("__repr__", &Lit::repr)
    .def("__eq__", &Lit::operator==)
    .def("__lt__", &Lit::operator<)
    .def("__hash__", &Lit::hash)
    .def(py::pickle([](const Lit & lit) { return py::make_tuple(lit.var().idx(), lit.neg()); },
		    [](py::tuple t) { return Lit(Var(t[0].cast<unsigned>()), t[1].cast<bool>()); }));

  py::class_<Options>(m, "Options")
    .def(py::init<>())
    .def_readwrite("timeout_ms", &Options::timeout_ms)
    .def_readwrite("max_conflicts", &Options::max_conflicts)
    .def_readwrite("restart_max", &Options::restart_max)
    .def_readwrite("variable_decay", &Options::variable_decay)
    .def_readwrite("override_incremental", &Options::override_incremental)
    .def_readwrite("lookahead_simplify", &Options::lookahead_simplify)
    .def_readwrite("acce", &Options::acce)
    .def_readwrite("z3_replay_timeout_scale", &Options::z3_replay_timeout_scale)
    .def_readwrite("drat_trim_timeout_scale", &Options::drat_trim_timeout_scale)
    .def_readwrite("logfilename", &Options::logfilename);

  py::class_<Context>(m, "Context")
    .def(py::init<>());

  py::class_<Expr>(m, "Expr")
    .def("flip", &Expr::flip)
    .def("var", &Expr::var)
    .def("is_neg", &Expr::is_neg)
    .def("ilit", &Expr::ilit);

  // TODO(dselsam): make all caps to be consistent
  py::enum_<Status>(m, "Status")
    .value("UNKNOWN", Status::UNKNOWN)
    .value("UNSAT", Status::UNSAT)
    .value("SAT", Status::SAT);

  py::enum_<Assignment>(m, "Assignment")
    .value("POS", Assignment::POS)
    .value("NEG", Assignment::NEG)
    .value("SKIP", Assignment::SKIP);

  py::class_<TFData>(m, "TFData")
    .def_readonly("n_vars", &TFData::n_vars)
    .def_readonly("n_clauses", &TFData::n_clauses)
    .def_readonly("CL_idxs", &TFData::CL_idxs)
    .def_readonly("core_var_mask", &TFData::core_var_mask)
    .def_readonly("core_clause_mask", &TFData::core_clause_mask);

  py::class_<Solver>(m, "Solver")
    .def(py::init<Context &, Options const & >(),
	 py::arg("ctx"), py::arg("opts"),
	 py::keep_alive<1, 2>(),
	 py::call_guard<py::gil_scoped_release>())

    .def("clone", &Solver::clone, py::arg("ctx"), py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>())
    .def("from_string", &Solver::from_string, py::arg("s"), py::call_guard<py::gil_scoped_release>())
    .def("from_file", &Solver::from_file, py::arg("filename"), py::call_guard<py::gil_scoped_release>())
    .def("serialize", &Solver::serialize, py::call_guard<py::gil_scoped_release>())
    .def("dimacs", &Solver::dimacs, py::call_guard<py::gil_scoped_release>())
    .def("add", &Solver::add, py::arg("expr"))
    .def("get_free_var", &Solver::get_free_var, py::arg("v_idx"))
    .def("propagate", &Solver::propagate, py::call_guard<py::gil_scoped_release>())
    .def("check", &Solver::check, py::call_guard<py::gil_scoped_release>())
    .def("check_with_timeout_ms", &Solver::check_with_timeout_ms, py::arg("timeout_ms"), py::call_guard<py::gil_scoped_release>())
    .def("set_activity_scores", &Solver::set_activity_scores, py::arg("pi"), py::arg("pi_scale"), py::call_guard<py::gil_scoped_release>())
    .def("cube", &Solver::cube, py::call_guard<py::gil_scoped_release>())

    .def("to_tf_data", &Solver::to_tf_data, py::call_guard<py::gil_scoped_release>())
    .def("to_tf_data_with_core", &Solver::to_tf_data_with_core, py::call_guard<py::gil_scoped_release>())
    .def("get_more_cores", &Solver::get_more_cores, py::arg("ctx"), py::arg("max_tries"), py::arg("percent_to_keep"), py::call_guard<py::gil_scoped_release>());

  m.def("deserialize", &deserialize, py::arg("ctx"), py::arg("opts"), py::arg("serial"), py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>());
}
