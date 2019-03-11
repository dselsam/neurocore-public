#include "neurosolver.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;

PYBIND11_MODULE(neurominisat, m) {
  py::register_exception<SolverException>(m, "SolverException");

  py::enum_<Status>(m, "Status")
    .value("UNKNOWN", Status::UNKNOWN)
    .value("UNSAT",   Status::UNSAT)
    .value("SAT",     Status::SAT);

  py::enum_<NeuroSolverMode>(m, "NeuroSolverMode")
    .value("NONE",      NeuroSolverMode::NONE)
    .value("NEURO",     NeuroSolverMode::NEURO)
    .value("RAND_SAME", NeuroSolverMode::RAND_SAME)
    .value("RAND_DIFF", NeuroSolverMode::RAND_DIFF);  
  
  py::class_<NeuroSATConfig>(m, "NeuroSATConfig")
    .def(py::init<NeuroSolverMode, float, unsigned, unsigned, float, float>(),
	 py::arg("mode"),
	 py::arg("n_secs_pause"),
	 py::arg("max_lclause_size"),
	 py::arg("max_n_nodes_cells"),
	 py::arg("itau"),
	 py::arg("scale"));

  py::class_<NeuroSATArgs>(m, "NeuroSATArgs")
    .def_readonly("n_vars",    &NeuroSATArgs::n_vars)
    .def_readonly("n_clauses", &NeuroSATArgs::n_clauses)
    .def_readonly("CL_idxs",   &NeuroSATArgs::CL_idxs);

  py::class_<NeuroSATGuesses>(m, "NeuroSATGuesses")
    .def(py::init<float, Eigen::ArrayXf const &>(),
	 py::arg("n_secs_gpu"),
	 py::arg("pi_core_var_logits"));

  m.def("neurosat_failed_to_guess", &neurosat_failed_to_guess);

  py::class_<NeuroSolverResults>(m, "NeuroSolverResults")
    .def_readonly("status",    &NeuroSolverResults::status)
    .def_readonly("n_secs_user", &NeuroSolverResults::n_secs_user)
    .def_readonly("n_secs_call", &NeuroSolverResults::n_secs_call)    
    .def_readonly("n_secs_gpu",   &NeuroSolverResults::n_secs_gpu);

  py::class_<NeuroSolver>(m, "NeuroSolver")
    .def(py::init<NeuroSATFunc const &, NeuroSATConfig const &>(), py::arg("func"), py::arg("cfg"))
    .def("from_file",               &NeuroSolver::from_file, py::arg("filename"))
    .def("check_with_timeout_s",    &NeuroSolver::check_with_timeout_s, py::arg("timeout_s"));
}
