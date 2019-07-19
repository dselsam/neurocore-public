#include "simp/SimpSolver.h"
#include "simp/Neuro.h"
#include "mtl/Sort.h"
#include <vector>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

using std::vector;

namespace Glucose {

static float lse(Eigen::ArrayXf const & x) {
  float x_star = x.maxCoeff();
  return x_star + log((x - x_star).exp().sum());
}


NeuroSolverMode neuro_parse_solver_mode(std::string const & mode) {
  if (mode == "NONE") { return NeuroSolverMode::NONE; }
  else if (mode == "RAND") { return NeuroSolverMode::RAND; }
  else if (mode == "NEURO") { return NeuroSolverMode::NEURO; }
  else { throw std::runtime_error("invalid solver mode"); }
}

void SimpSolver::neuro_begin_solve() {
    printf("--NEURO_BEGIN_SOLVE--\n");
    if (ncfg.mode == NeuroSolverMode::NEURO) {
	auto channel_args = grpc::ChannelArguments();
	channel_args.SetMaxReceiveMessageSize(pow(2, 30));
	channel_args.SetMaxSendMessageSize(pow(2, 30));
	channel_args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
	auto channel = grpc::CreateCustomChannel(ncfg.server, grpc::InsecureChannelCredentials(), channel_args);
	nstate.stub = std::unique_ptr<neurosat::NeuroSATServer::Stub>(neurosat::NeuroSATServer::NewStub(channel));
    }
    nstate.t_start           = clock();
    nstate.t_next            = nstate.t_start;
    nstate.n_secs_next_pause = ncfg.n_secs_pause;
}

bool SimpSolver::neuro_should_stop() const {
    return nstate.n_secs_user() > ncfg.timeout_s;
}

struct ClauseSize_lt {
    ClauseAllocator& ca;
    ClauseSize_lt(ClauseAllocator& ca_) : ca(ca_) {}
    bool operator () (CRef x, CRef y) {
	return ca[x].size() < ca[y].size(); }
};

void SimpSolver::neuro_begin_pick_branch_lit() {
    if (ncfg.mode == NeuroSolverMode::NONE) { return; }
    if (nstate.n_secs_user() > ncfg.timeout_s) { interrupt(); return; }
    if (!ncfg.call_if_too_big && nstate.original_too_big) { return; }
    if (nstate.n_calls > 0 && clock() < nstate.t_next) { return; }

    vector<int>      v_to_nv;
    vector<unsigned> nv_to_v;

    vector<bool> assigned(nVars(), false);
    int trail_limit = (trail_lim.size() == 0 ? trail.size() : trail_lim[0]);
    for (int i = 0; i < trail_limit; ++i) {
	assigned[var(trail[i])] = true;
    }

    for (int v = 0; v < nVars(); ++v) {
	if (assigned[v] || isEliminated(v)) {
	    v_to_nv.push_back(-1);
	} else {
	    v_to_nv.push_back(nv_to_v.size());
	    nv_to_v.push_back(v_to_nv.size() - 1);
	}
    }

    neurosat::NeuroSATArgs args;
    args.set_n_vars(nv_to_v.size());
    if (!(args.n_vars() > 0)) { return; } // throw std::runtime_error("no vars!"); }

    unsigned n_clauses = 0;

    auto traverse_clause = [&](Clause const & clause) {
	if (clause.reloced()) {
	    return;
	} else if (2 * args.n_vars() + n_clauses + args.c_idxs_size() > ncfg.max_n_nodes_cells) {
	    return;
	} else if (clause.learnt() && (unsigned) clause.size() > ncfg.max_lclause_size) {
	    return;
	} else {
	    for (int arg_idx = 0; arg_idx < clause.size(); ++arg_idx) {
		Lit lit = clause[arg_idx];
		Var v   = var(lit);
		if (v >= nVars()) { throw std::runtime_error("var too big!"); }
		int nv  = v_to_nv[v];
		if (nv != -1) {
		    args.add_c_idxs(n_clauses);
		    args.add_l_idxs(sign(lit) ? (nv + args.n_vars()) : nv);
		}
	    }
	    n_clauses++;
	    return;
	}
    };

    for (int c_idx = 0; c_idx < clauses.size(); ++c_idx) { traverse_clause(ca[clauses[c_idx]]); }
    for (int c_idx = 0; c_idx < permanentLearnts.size(); ++c_idx) { traverse_clause(ca[permanentLearnts[c_idx]]); }
    if (2 * args.n_vars() + n_clauses + args.c_idxs_size() > ncfg.max_n_nodes_cells) {
	// unlike in minisat, here we query the first time no matter what (but rely on the cutoff to avoid OOM)
	nstate.original_too_big = true;
    }

    sort(learnts, ClauseSize_lt(ca));
    for (int c_idx = 0; c_idx < learnts.size(); ++c_idx) { traverse_clause(ca[learnts[c_idx]]); }

    args.set_n_clauses(n_clauses);
    if (!(args.n_clauses() > 0)) { throw std::runtime_error("no clauses!"); }
    if (!(args.c_idxs_size() > 0)) { throw std::runtime_error("no cells!"); }
    args.set_itau(ncfg.itau);

    neurosat::NeuroSATGuesses guesses;

    if (ncfg.mode == NeuroSolverMode::NEURO) {
      grpc::ClientContext context;
      context.set_compression_algorithm(GRPC_COMPRESS_NONE);
      std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(120000);
      context.set_deadline(deadline);
      grpc::Status status = nstate.stub->query_neurosat(&context, args, &guesses);
      if (!status.ok()) {
	string msg = "GRPC status not OK";
	msg += " (" + status.error_message() + ")";
	throw std::runtime_error(msg);
      }
    } else {
      // RAND
      guesses.set_success(true);
      guesses.set_msg("rand");
      guesses.set_n_secs_gpu(0.0);

      Eigen::ArrayXf random_logits = ncfg.itau * Eigen::ArrayXf::Random(args.n_vars());
      Eigen::ArrayXf random_ps     = (random_logits - lse(random_logits)).exp();
      for (unsigned i = 0; i < args.n_vars(); ++i) {
	guesses.add_pi_core_var_ps(random_ps[i]);
      }
    }

    if (!guesses.success()) {
      nstate.n_fails++;
      return;
    }

    var_inc = 1.0;
    for (unsigned nv = 0; nv < nv_to_v.size(); ++nv) {
	activity[nv_to_v[nv]] = ncfg.scale * args.n_vars() * guesses.pi_core_var_ps(nv);
    }
    rebuildOrderHeap();

    nstate.n_calls++;
    nstate.n_secs_gpu        += guesses.n_secs_gpu();
    nstate.t_next             = clock() + nstate.n_secs_next_pause * CLOCKS_PER_SEC;
    nstate.n_secs_next_pause *= ncfg.n_secs_pause_inc;
}

void SimpSolver::neuro_end_solve(lbool status) {
    printf("--NEURO_END_SOLVE--\n");
    FILE * res = fopen(ncfg.neuro_outfile.c_str(), "w");
    if (status == l_True) { fprintf(res, "SAT"); }
    else if (status == l_False) { fprintf(res, "UNSAT"); }
    else { fprintf(res, "UNKNOWN"); }

    fprintf(res, " %f %d %d %f", nstate.n_secs_user(), nstate.n_calls, nstate.n_fails, nstate.n_secs_gpu);
    fflush(res);
    fclose(res);
}
}
