#!/usr/bin/env python3
from pathlib import Path

p = Path('native_online_baselines_v4.py')
s = p.read_text()

# CP-SAT: replace nondeterministic wall-clock budget by OR-Tools deterministic time.
old = "solver.parameters.max_time_in_seconds = 0.12\n    solver.parameters.num_search_workers = 1"
new = "solver.parameters.max_deterministic_time = 0.12\n    solver.parameters.num_search_workers = 1"
assert old in s
s = s.replace(old, new, 1)

# CBC: OR-Tools MPSolver's CBC adapter exposes wall-clock stopping but not a clean
# deterministic node budget in this Python path. Keep CBC itself, but use the
# open-source Python-MIP CBC binding and a fixed B&B node budget instead.
anchor = "from ortools.linear_solver import pywraplp\n"
assert anchor in s
s = s.replace(anchor, anchor + "from mip import Model as MipModel, xsum as mip_xsum, BINARY as MIP_BINARY, MAXIMIZE as MIP_MAXIMIZE, CBC as MIP_CBC, OptimizationStatus as MIPStatus\n", 1)

start = s.index("def choose_cbc_saa(")
end = s.index("\n\nclass SoftCoverageMCTSState", start)
replacement = '''def choose_cbc_saa(P_full, rems, residual_cols, seed, step):
    P = np.asarray(P_full[np.ix_(rems, residual_cols)], dtype=float)
    if P.shape[1] == 0:
        return rems[0]
    worlds = saa_world(P, seed, step)
    model = MipModel(sense=MIP_MAXIMIZE, solver_name=MIP_CBC)
    model.verbose = 0
    model.threads = 1
    model.seed = int((BASE_SEED + seed * 1000003 + step * 9176) % 2147483647)
    x = [model.add_var(var_type=MIP_BINARY, name=f'x{i}') for i in range(len(rems))]
    y = [[model.add_var(var_type=MIP_BINARY, name=f'y{s}_{t}') for t in range(P.shape[1])] for s in range(SAA_SCENARIOS)]
    model += mip_xsum(x) <= min(HORIZON, len(rems))
    for ss in range(SAA_SCENARIOS):
        for t in range(P.shape[1]):
            inds = np.flatnonzero(worlds[ss, :, t]).tolist()
            if inds:
                model += y[ss][t] <= mip_xsum(x[i] for i in inds)
            else:
                model += y[ss][t] == 0
    model.objective = mip_xsum(y[ss][t] for ss in range(SAA_SCENARIOS) for t in range(P.shape[1]))
    status = model.optimize(max_nodes=1000)
    if status not in (MIPStatus.OPTIMAL, MIPStatus.FEASIBLE):
        return rems[int(np.argmax(P.sum(axis=1)))]
    plan = [i for i in range(len(rems)) if x[i].x is not None and x[i].x > 0.5]
    if not plan:
        return rems[int(np.argmax(P.sum(axis=1)))]
    first_local = max(plan, key=lambda i: (plan_unique_contribution(P, plan, i), -i))
    return rems[first_local]
'''
s = s[:start] + replacement + s[end:]

# Record the corrected planner budgets in the result manifest.
s = s.replace("'saa_scenarios': SAA_SCENARIOS,", "'saa_scenarios': SAA_SCENARIOS,\n        'cpsat_max_deterministic_time': 0.12,\n        'cbc_max_nodes': 1000,\n        'cbc_binding': 'python-mip-1.17.6/CBC',", 1)

p.write_text(s)
print('DETERMINISTIC_SOLVER_BUDGET_FIX_APPLIED')
