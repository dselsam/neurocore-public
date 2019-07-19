#pragma once
#include <ctime>
#include <memory>
#include "simp/neurosat.pb.h"
#include "simp/neurosat.grpc.pb.h"

using std::string;

namespace Glucose {

enum class NeuroSolverMode { NONE, NEURO, RAND };

NeuroSolverMode neuro_parse_solver_mode(std::string const & mode);

struct NeuroSATConfig {
    NeuroSolverMode mode{NeuroSolverMode::NONE};
    string          server{"NULL"};
    unsigned        timeout_s{UINT32_MAX};
    float           n_secs_pause{0.0};
    float           n_secs_pause_inc{0.0};
    unsigned        max_lclause_size{0};
    unsigned        max_n_nodes_cells{0};
    float           itau{0.0};
    float           scale{0.0};
    bool            call_if_too_big{false};
    string          neuro_outfile{"out"};
};

struct NeuroSATState {
    std::unique_ptr<neurosat::NeuroSATServer::Stub> stub;
    unsigned n_calls{0};
    unsigned n_fails{0};
    float    n_secs_gpu{0.0};
    clock_t  t_start{};
    clock_t  t_next{};
    float    n_secs_next_pause{0.0};
    bool     has_called{false};
    bool     original_too_big{false};

    float n_secs_user() const { return (double) (clock() - t_start) / CLOCKS_PER_SEC; }
};

}
