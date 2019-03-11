import solver
import argparse
import collections

def status_to_str(status):
    if status == solver.Status.UNSAT: return "unsat"
    elif status == solver.Status.SAT: return "sat"
    elif status == solver.Status.UNKNOWN: return "unknown"
    else: raise Exception("unknown status: %s" % str(status))

def mk_opts(cfg):
    if type(cfg) is argparse.Namespace: cfg = vars(cfg)
    opts = solver.Options()
    keys = ['timeout_ms', 'max_conflicts', 'acce', 'override_incremental', 'lookahead_simplify',
            'z3_replay_timeout_scale', 'drat_trim_timeout_scale', 'solver_logfilename']
    for key in [key for key in keys if key in cfg]:
        setattr(opts, key, cfg[key])
    return opts

# tftds
TFData = collections.namedtuple('TFData', ['n_vars', 'n_clauses', 'CL_idxs', 'core_var_mask', 'core_clause_mask'])
def tfd_to_py(tfd):
    assert(type(tfd) is solver.TFData)
    assert(tfd.n_vars > 0)
    assert(tfd.n_clauses > 0)
    return TFData(n_vars=tfd.n_vars,
                  n_clauses=tfd.n_clauses,
                  CL_idxs=tfd.CL_idxs,
                  core_var_mask=tfd.core_var_mask,
                  core_clause_mask=tfd.core_clause_mask)
