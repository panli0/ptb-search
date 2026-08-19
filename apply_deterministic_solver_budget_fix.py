#!/usr/bin/env python3
from pathlib import Path

p = Path('native_online_baselines_v4.py')
s = p.read_text()

# CP-SAT: replace nondeterministic wall-clock budget by OR-Tools deterministic time.
old = "solver.parameters.max_time_in_seconds = 0.12\n    solver.parameters.num_search_workers = 1"
new = "solver.parameters.max_deterministic_time = 0.12\n    solver.parameters.num_search_workers = 1"
assert old in s
s = s.replace(old, new, 1)

# CBC: keep the CBC branch-and-bound solver, but invoke it through the stable
# open-source PuLP command interface, which exposes a fixed maxNodes budget.
anchor = "from ortools.linear_solver import pywraplp\n"
assert anchor in s
s = s.replace(anchor, anchor + "import pulp as pl\n", 1)

start = s.index("def choose_cbc_saa(")
end = s.index("\n\nclass SoftCoverageMCTSState", start)
replacement = '''def choose_cbc_saa(P_full, rems, residual_cols, seed, step):
    P = np.asarray(P_full[np.ix_(rems, residual_cols)], dtype=float)
    if P.shape[1] == 0:
        return rems[0]
    worlds = saa_world(P, seed, step)
    prob = pl.LpProblem('cbc_saa', pl.LpMaximize)
    x = [pl.LpVariable(f'x{i}', cat='Binary') for i in range(len(rems))]
    y = [[pl.LpVariable(f'y{ss}_{t}', cat='Binary') for t in range(P.shape[1])] for ss in range(SAA_SCENARIOS)]
    prob += pl.lpSum(x) <= min(HORIZON, len(rems))
    for ss in range(SAA_SCENARIOS):
        for t in range(P.shape[1]):
            inds = np.flatnonzero(worlds[ss, :, t]).tolist()
            if inds:
                prob += y[ss][t] <= pl.lpSum(x[i] for i in inds)
            else:
                prob += y[ss][t] == 0
    prob += pl.lpSum(y[ss][t] for ss in range(SAA_SCENARIOS) for t in range(P.shape[1]))
    cbc_seed = int((BASE_SEED + seed * 1000003 + step * 9176) % 2147483647)
    cbc = pl.PULP_CBC_CMD(msg=False, threads=1, maxNodes=1000,
                          options=[f'randomSeed {cbc_seed}'])
    status = prob.solve(cbc)
    if pl.LpStatus.get(status) not in ('Optimal', 'Not Solved'):
        return rems[int(np.argmax(P.sum(axis=1)))]
    plan = [i for i in range(len(rems)) if x[i].value() is not None and x[i].value() > 0.5]
    if not plan:
        return rems[int(np.argmax(P.sum(axis=1)))]
    first_local = max(plan, key=lambda i: (plan_unique_contribution(P, plan, i), -i))
    return rems[first_local]
'''
s = s[:start] + replacement + s[end:]

# Record corrected planner budgets/bindings in the result manifest.
s = s.replace("'saa_scenarios': SAA_SCENARIOS,", "'saa_scenarios': SAA_SCENARIOS,\n        'cpsat_max_deterministic_time': 0.12,\n        'cbc_max_nodes': 1000,\n        'cbc_binding': 'PuLP-3.3.2/PULP_CBC_CMD',", 1)

p.write_text(s)
print('DETERMINISTIC_SOLVER_BUDGET_FIX_APPLIED')
