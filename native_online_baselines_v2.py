#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from simpleai.search import SearchProblem, astar, greedy
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from submodlib import ProbabilisticSetCoverFunction
from mcts import mcts

ROOT = Path(sys.argv[1])
PI = Path(sys.argv[2])
OUT = Path(sys.argv[3])
METHOD = sys.argv[4]

TH = (0.5, 0.7, 0.9, 0.95, 0.99, 1.0)
HORIZON = 4
FUTURE_BRANCH = 12  # only future tree-search branches are pruned; every real first action remains eligible
SAA_SCENARIOS = 6
RIDGE_ALPHA = 3.0
FEATURES = 48
BASE_SEED = 20260818


def atomr(s):
    m = re.fullmatch(r'\s*ring_deduced\("(has|lacks)",\s*(\d+),\s*(\d+)\)\s*', str(s or ''))
    return (int(m[3]), int(m[2]), m[1] == 'has') if m else None


def atomm(s):
    m = re.fullmatch(r'\s*module_deduced\("(has|lacks)",\s*(\d+)\)\s*', str(s or ''))
    return (int(m[2]), m[1] == 'has') if m else None


def pi_data():
    d = json.loads(PI.read_text())
    S = {x['uid']: x for x in d['spaces']}
    T = {x['uid']: x for x in d['theorems']}
    truth = {s: {} for s in S}
    for x in d['traits']:
        if x.get('space') in S and isinstance(x.get('value'), bool):
            truth[x['space']][x['property']] = x['value']

    def ev(f, tr):
        if not isinstance(f, dict):
            return None
        if f.get('kind') == 'atom':
            p = f.get('property')
            return tr.get(p) == f.get('value') if p in tr else None
        z = [ev(x, tr) for x in f.get('subs', [])]
        if f.get('kind') == 'and':
            return False if False in z else (True if z and all(x is True for x in z) else None)
        if f.get('kind') == 'or':
            return True if True in z else (False if z and all(x is False for x in z) else None)
        return None

    cov = {s: set() for s in S}
    for s, tr in truth.items():
        for tid, t in T.items():
            if ev(t.get('then'), tr) is True and ev(t.get('when'), tr) is False:
                cov[s].add(tid)
    U = set().union(*cov.values()) if cov else set()
    txt = {s: ' '.join(str(S[s].get(k, '') or '') for k in ('name', 'description', 'aliases')) for s in S}
    return sorted(S), txt, cov, U


def dart(dom):
    ring = dom == 'ring'
    b = ROOT / ('db/ringapp' if ring else 'db/moduleapp')
    root = b / ('ring' if ring else 'module')
    parse = atomr if ring else atomm
    trs, txt = {}, {}
    for od in sorted(x for x in root.iterdir() if x.is_dir()):
        pp, dp = od / 'properties.yaml', od / 'data.yaml'
        if not pp.exists():
            continue
        try:
            p = yaml.safe_load(pp.read_text()) or {}
            meta = yaml.safe_load(dp.read_text()) or {}
        except Exception:
            continue
        q = {}
        if ring:
            for k, v in p.items():
                try:
                    i = int(k.split('_')[-1])
                except Exception:
                    continue
                l, r = v.get('has_on_left'), v.get('has_on_right')
                if isinstance(l, bool) and isinstance(r, bool):
                    q[(i, 0)] = l
                    q[(i, 1)] = r
                    q[(i, 2)] = l if l == r else None
            txt[od.name] = ' '.join([
                str(meta.get('name', '')), str(meta.get('description', '')),
                ' '.join(map(str, meta.get('keywords') or [])), 'comm=' + str(meta.get('is_commutative'))
            ])
        else:
            for k, v in p.items():
                try:
                    i = int(k.split('_')[-1])
                except Exception:
                    continue
                if isinstance(v.get('has'), bool):
                    q[i] = v['has']
            txt[od.name] = ' '.join([
                str(meta.get('name', '')), str(meta.get('description', '')),
                'ring=' + str(meta.get('ring', '')), 'opp=' + str(meta.get('opposite_ring', ''))
            ])
        trs[od.name] = q

    rules = []
    for f in sorted((b / 'logic').glob('*.yaml')):
        try:
            d = yaml.safe_load(f.read_text()) or {}
        except Exception:
            continue
        if d.get('active') is False:
            continue
        a, c = parse(d.get('hyps')), parse(d.get('concs'))
        if a and c:
            rules.append((f.stem, a, c))

    cov = {o: set() for o in trs}
    for o, q in trs.items():
        for rid, a, c in rules:
            if ring:
                pa, sa, va = a
                pc, sc, vc = c
                xa, xc = q.get((pa, sa)), q.get((pc, sc))
                xa = q.get((pa, 2)) if xa is None else xa
                xc = q.get((pc, 2)) if xc is None else xc
            else:
                pa, va = a
                pc, vc = c
                xa, xc = q.get(pa), q.get(pc)
            if isinstance(xa, bool) and isinstance(xc, bool) and xc == vc and xa != va:
                cov[o].add(rid)
    U = set().union(*cov.values()) if cov else set()
    return sorted(trs), txt, cov, U


@dataclass
class OracleEnv:
    ids: list
    _cov: dict
    targets: list

    def __post_init__(self):
        self.ti = {t: i for i, t in enumerate(self.targets)}

    def reveal_after_commit(self, j: int):
        """The sole interface to target truth. Caller must commit j before calling."""
        o = self.ids[j]
        row = np.zeros(len(self.targets), dtype=float)
        for t in self._cov[o]:
            row[self.ti[t]] = 1.0
        return o, row


def warm_count(n):
    return min(8, max(3, n // 20))


def expected_union(P, selected, weights=None):
    if P.shape[1] == 0 or len(selected) == 0:
        return 0.0
    w = np.ones(P.shape[1], dtype=float) if weights is None else np.asarray(weights, dtype=float)
    q = np.prod(1.0 - P[list(selected)], axis=0)
    return float(np.dot(w, 1.0 - q))


def marginal(P, selected, action, weights=None):
    if P.shape[1] == 0:
        return 0.0
    w = np.ones(P.shape[1], dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if selected:
        q = np.prod(1.0 - P[list(selected)], axis=0)
    else:
        q = np.ones(P.shape[1], dtype=float)
    return float(np.dot(w, q * P[action]))


def plan_unique_contribution(P, plan, action):
    if P.shape[1] == 0:
        return 0.0
    rest = [x for x in plan if x != action]
    q = np.prod(1.0 - P[rest], axis=0) if rest else np.ones(P.shape[1], dtype=float)
    return float(np.sum(q * P[action]))


class FixedHorizonCoverageProblem(SearchProblem):
    """Full-soft expected-coverage search; only future branches are pruned.

    Every candidate is available as the first real action. After a first action is
    hypothetically chosen inside the plan, only the top FUTURE_BRANCH current
    marginal continuations are exposed to keep repeated online planning tractable.
    """
    def __init__(self, P, mode):
        self.P = np.asarray(P, dtype=float)
        self.n = self.P.shape[0]
        self.h = min(HORIZON, self.n)
        self.C = float(max(1, self.P.shape[1]))
        self.mode = mode
        super().__init__(initial_state=())

    @lru_cache(maxsize=100000)
    def value(self, state):
        return expected_union(self.P, state)

    def actions(self, state):
        if len(state) >= self.h:
            return []
        used = set(state)
        rem = [i for i in range(self.n) if i not in used]
        if not state:
            return rem  # critical: never pre-filter the real next action
        scored = sorted(rem, key=lambda i: (marginal(self.P, state, i), -i), reverse=True)
        return scored[:min(FUTURE_BRANCH, len(scored))]

    def result(self, state, action):
        return tuple(sorted(state + (int(action),)))

    def is_goal(self, state):
        return len(state) == self.h

    def cost(self, state, action, state2):
        # For fixed h, sum(C - marginal) = h*C - expected_union(final_set).
        return self.C - marginal(self.P, state, int(action))

    def heuristic(self, state):
        if self.mode == 'bestfirst':
            # Greedy best-first: prefer the state with largest current expected coverage.
            return -self.value(state)
        steps = self.h - len(state)
        if steps <= 0:
            return 0.0
        used = set(state)
        rem = [i for i in range(self.n) if i not in used]
        best = max((marginal(self.P, state, i) for i in rem), default=0.0)
        # Submodularity implies future marginals cannot exceed current best marginal.
        return steps * max(0.0, self.C - best)


def choose_tree(P_full, rems, residual_cols, method):
    P = np.asarray(P_full[np.ix_(rems, residual_cols)], dtype=float)
    if P.shape[1] == 0:
        return rems[0]
    prob = FixedHorizonCoverageProblem(P, 'bestfirst' if method == 'bestfirst' else 'astar')
    sol = greedy(prob, graph_search=True) if method == 'bestfirst' else astar(prob, graph_search=True)
    if sol is None:
        return rems[int(np.argmax(P.sum(axis=1)))]
    plan = list(sol.state)
    if not plan:
        return rems[int(np.argmax(P.sum(axis=1)))]
    # The fixed-horizon solution is an unordered set; execute the member most
    # structurally indispensable to that predicted plan, then replan after reveal.
    first_local = max(plan, key=lambda i: (plan_unique_contribution(P, plan, i), -i))
    return rems[first_local]


def choose_psc(P_full, rems, residual_cols, weights=None):
    P = np.asarray(P_full[np.ix_(rems, residual_cols)], dtype=float)
    if P.shape[1] == 0:
        return rems[0]
    w = [1.0] * P.shape[1] if weights is None else list(map(float, weights))
    f = ProbabilisticSetCoverFunction(
        n=len(rems), probs=P.tolist(), num_concepts=P.shape[1], concept_weights=w
    )
    budget = min(HORIZON, len(rems))
    try:
        ans = f.maximize(budget=budget, optimizer='LazyGreedy', show_progress=False,
                         stopIfZeroGain=False, stopIfNegativeGain=False)
    except Exception:
        ans = f.maximize(budget=budget, optimizer='NaiveGreedy', show_progress=False,
                         stopIfZeroGain=False, stopIfNegativeGain=False)
    if ans:
        x = ans[0][0] if isinstance(ans[0], (tuple, list)) else ans[0]
        return rems[int(x)]
    score = P @ np.asarray(w, dtype=float)
    return rems[int(np.argmax(score))]


def saa_world(P, seed, step):
    rng = np.random.default_rng(BASE_SEED + seed * 1000003 + step * 9176)
    return rng.random((SAA_SCENARIOS, P.shape[0], P.shape[1])) < P[None, :, :]


def choose_cpsat_saa(P_full, rems, residual_cols, seed, step):
    P = np.asarray(P_full[np.ix_(rems, residual_cols)], dtype=float)
    if P.shape[1] == 0:
        return rems[0]
    worlds = saa_world(P, seed, step)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f'x{i}') for i in range(len(rems))]
    y = [[model.new_bool_var(f'y{s}_{t}') for t in range(P.shape[1])] for s in range(SAA_SCENARIOS)]
    model.add(sum(x) <= min(HORIZON, len(rems)))
    for s in range(SAA_SCENARIOS):
        for t in range(P.shape[1]):
            inds = np.flatnonzero(worlds[s, :, t]).tolist()
            if inds:
                model.add(y[s][t] <= sum(x[i] for i in inds))
            else:
                model.add(y[s][t] == 0)
    model.maximize(sum(y[s][t] for s in range(SAA_SCENARIOS) for t in range(P.shape[1])))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.12
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return rems[int(np.argmax(P.sum(axis=1)))]
    plan = [i for i in range(len(rems)) if solver.value(x[i])]
    if not plan:
        return rems[int(np.argmax(P.sum(axis=1)))]
    first_local = max(plan, key=lambda i: (plan_unique_contribution(P, plan, i), -i))
    return rems[first_local]


def choose_cbc_saa(P_full, rems, residual_cols, seed, step):
    P = np.asarray(P_full[np.ix_(rems, residual_cols)], dtype=float)
    if P.shape[1] == 0:
        return rems[0]
    worlds = saa_world(P, seed, step)
    solver = pywraplp.Solver.CreateSolver('CBC')
    solver.SetTimeLimit(120)
    x = [solver.BoolVar(f'x{i}') for i in range(len(rems))]
    y = [[solver.BoolVar(f'y{s}_{t}') for t in range(P.shape[1])] for s in range(SAA_SCENARIOS)]
    solver.Add(sum(x) <= min(HORIZON, len(rems)))
    for s in range(SAA_SCENARIOS):
        for t in range(P.shape[1]):
            inds = np.flatnonzero(worlds[s, :, t]).tolist()
            if inds:
                solver.Add(y[s][t] <= sum(x[i] for i in inds))
            else:
                solver.Add(y[s][t] == 0)
    solver.Maximize(sum(y[s][t] for s in range(SAA_SCENARIOS) for t in range(P.shape[1])))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return rems[int(np.argmax(P.sum(axis=1)))]
    plan = [i for i in range(len(rems)) if x[i].solution_value() > 0.5]
    if not plan:
        return rems[int(np.argmax(P.sum(axis=1)))]
    first_local = max(plan, key=lambda i: (plan_unique_contribution(P, plan, i), -i))
    return rems[first_local]


class SoftCoverageMCTSState:
    def __init__(self, P, rem, q=None, depth=0, action_order=None):
        self.P = P
        self.rem = tuple(rem)
        self.q = np.ones(P.shape[1], dtype=float) if q is None else q
        self.depth = depth
        self.action_order = tuple(self.rem if action_order is None else action_order)
        self.start_mass = float(P.shape[1])

    def getCurrentPlayer(self):
        return 1

    def getPossibleActions(self):
        return list(self.action_order)

    def takeAction(self, action):
        q2 = self.q * (1.0 - self.P[action])
        rem2 = tuple(i for i in self.rem if i != action)
        # Deterministic ordering; no predictor top-k filter.
        order2 = tuple(sorted(rem2))
        return SoftCoverageMCTSState(self.P, rem2, q=q2, depth=self.depth + 1, action_order=order2)

    def isTerminal(self):
        return self.depth >= min(HORIZON, self.P.shape[0]) or not self.rem or self.P.shape[1] == 0

    def getReward(self):
        return self.start_mass - float(np.sum(self.q))


def choose_mcts(P_full, rems, residual_cols, seed, step):
    P = np.asarray(P_full[np.ix_(rems, residual_cols)], dtype=float)
    if P.shape[1] == 0:
        return rems[0]
    # Randomize root expansion order reproducibly so the upstream implementation
    # does not systematically favor low-index actions before full expansion.
    rr = random.Random(BASE_SEED + seed * 1000003 + step * 9176)
    root_order = list(range(len(rems)))
    rr.shuffle(root_order)
    state = SoftCoverageMCTSState(P, tuple(range(len(rems))), action_order=tuple(root_order))
    iterations = max(180, len(rems) + 160)  # full root expansion + deeper search budget
    random.seed(BASE_SEED + seed * 99991 + step * 313)
    action_local = mcts(iterationLimit=iterations).search(initialState=state)
    return rems[int(action_local)]


def target_weights(Yobs, obs, residual_cols):
    if not residual_cols:
        return []
    floor = 1.0 / (len(obs) + 1.0)
    vals = []
    for t in residual_cols:
        prev = float(np.mean(Yobs[obs, t])) if obs else 0.0
        vals.append(1.0 / math.sqrt(max(prev, floor)))
    m = float(np.mean(vals)) if vals else 1.0
    return [v / m for v in vals]


def run_seed(ids, txt, cov, U, method, seed):
    targets = sorted(U)
    env = OracleEnv(ids, cov, targets)
    n = len(ids)
    V = HashingVectorizer(n_features=FEATURES, alternate_sign=False, norm='l2', ngram_range=(1, 2))
    X = V.transform([txt[i] for i in ids]).toarray()
    Yobs = np.zeros((n, len(targets)), dtype=float)
    rem = set(range(n))
    obs, order, fractions = [], [], []
    seen = np.zeros(len(targets), dtype=bool)
    rng = random.Random(seed)

    def commit_and_reveal(j):
        # j is irrevocably selected before this call; chooser functions never receive env/cov.
        o, row = env.reveal_after_commit(j)
        Yobs[j] = row
        obs.append(j)
        order.append(o)
        rem.remove(j)
        seen[:] = np.logical_or(seen, row > 0.5)
        fractions.append(float(np.mean(seen)) if len(seen) else 1.0)

    for j in rng.sample(sorted(rem), min(warm_count(n), len(rem))):
        commit_and_reveal(j)
        if seen.all():
            break

    step = len(order)
    while rem and not seen.all():
        model = Ridge(alpha=RIDGE_ALPHA).fit(X[obs], Yobs[obs])
        Pfull = np.clip(model.predict(X), 0.0, 1.0)
        rems = sorted(rem)
        residual_cols = np.flatnonzero(~seen).tolist()

        if method == 'psc':
            j = choose_psc(Pfull, rems, residual_cols)
        elif method == 'wpsc':
            j = choose_psc(Pfull, rems, residual_cols, target_weights(Yobs, obs, residual_cols))
        elif method in ('bestfirst', 'astar'):
            j = choose_tree(Pfull, rems, residual_cols, method)
        elif method == 'cpsat':
            j = choose_cpsat_saa(Pfull, rems, residual_cols, seed, step)
        elif method == 'cbc':
            j = choose_cbc_saa(Pfull, rems, residual_cols, seed, step)
        elif method == 'mcts':
            j = choose_mcts(Pfull, rems, residual_cols, seed, step)
        else:
            raise ValueError(method)

        assert j in rem
        commit_and_reveal(j)
        step += 1

    # Thresholds are based only on actually committed queries. AUC is padded with
    # terminal coverage to the full object budget for comparability with old curves.
    hits = {}
    for q in TH:
        hits[str(q)] = next((i + 1 for i, v in enumerate(fractions) if v + 1e-12 >= q), None)
    terminal = fractions[-1] if fractions else 0.0
    padded = fractions + [terminal] * max(0, n - len(fractions))
    auc = float(np.mean(padded)) if padded else 0.0
    seq_hash = hashlib.sha256('\n'.join(map(str, order)).encode()).hexdigest()
    return {'seed': seed, 'hits': hits, 'auc': auc, 'queries_run': len(order), 'order_sha256': seq_hash, 'order': order}


def run_domain(name, data, method):
    ids, txt, cov, U = data
    seeds = [run_seed(ids, txt, cov, U, method, s) for s in range(5)]
    means = {}
    for q in TH:
        vals = [x['hits'][str(q)] for x in seeds if x['hits'][str(q)] is not None]
        means[str(q)] = float(np.mean(vals)) if vals else None
    return {
        'domain': name,
        'objects': len(ids),
        'targets': len(U),
        'method': method,
        'adaptive': True,
        'uses_unqueried_truth': False,
        'strict_reveal_boundary': True,
        'full_residual_target_set': True,
        'real_first_action_prefiltered': False,
        'horizon': HORIZON,
        'hits_mean': means,
        'auc_mean': float(np.mean([x['auc'] for x in seeds])),
        'seeds': seeds,
    }


UPSTREAMS = {
    'psc': 'decile-team/submodlib ProbabilisticSetCoverFunction + LazyGreedy',
    'wpsc': 'decile-team/submodlib ProbabilisticSetCoverFunction + LazyGreedy with revealed-prevalence weights',
    'bestfirst': 'simpleai-team/simpleai greedy best-first search',
    'astar': 'simpleai-team/simpleai A* search',
    'cpsat': 'Google OR-Tools CP-SAT on posterior-sampled stochastic maximum coverage',
    'cbc': 'Google OR-Tools CBC branch-and-bound/branch-and-cut on the same posterior-sampled model',
    'mcts': 'pbsinclair42/MCTS with full real action space and soft expected-coverage transitions',
}

res = {
    'protocol': (
        'Native online redo after invalidating the shared 10-target/16-candidate binary adapter. '
        'All methods share only the legal information state, Ridge predictor, warm-start, horizon=4, and post-commit reveal rule. '
        'All residual targets remain continuous probabilities. Every remaining candidate is eligible as the real next action. '
        'Method-specific search/optimization semantics are preserved; tree-search methods prune only hypothetical future branches.'
    ),
    'method': METHOD,
    'upstream': UPSTREAMS[METHOD],
    'frozen': {
        'ridge_alpha': RIDGE_ALPHA,
        'hash_features': FEATURES,
        'horizon': HORIZON,
        'tree_future_branch': FUTURE_BRANCH,
        'saa_scenarios': SAA_SCENARIOS,
        'seeds': [0, 1, 2, 3, 4],
    },
    'domains': [
        run_domain('pi-base', pi_data(), METHOD),
        run_domain('ring', dart('ring'), METHOD),
        run_domain('module', dart('module'), METHOD),
    ],
}
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
