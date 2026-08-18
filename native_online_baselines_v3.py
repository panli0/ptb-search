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
from sklearn.linear_model import SGDClassifier
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
PAIR_ALPHA = 0.0005
FEATURES = 48
CROSS_FEATURES = 64
BASE_SEED = 20260818


def atomr(s):
    m = re.fullmatch(r'\s*ring_deduced\("(has|lacks)",\s*(\d+),\s*(\d+)\)\s*', str(s or ''))
    return (int(m[3]), int(m[2]), m[1] == 'has') if m else None


def atomm(s):
    m = re.fullmatch(r'\s*module_deduced\("(has|lacks)",\s*(\d+)\)\s*', str(s or ''))
    return (int(m[2]), m[1] == 'has') if m else None


def _formula_properties(f):
    if not isinstance(f, dict):
        return set()
    if f.get('kind') == 'atom' and f.get('property'):
        return {f['property']}
    out = set()
    for x in f.get('subs', []) or []:
        out.update(_formula_properties(x))
    return out


def pi_data():
    d = json.loads(PI.read_text())
    S = {x['uid']: x for x in d['spaces']}
    T = {x['uid']: x for x in d['theorems']}
    Pmeta = {x['uid']: x for x in d.get('properties', [])}
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
    target_txt = {}
    for tid in U:
        t = T[tid]
        ps = sorted(_formula_properties(t.get('when')) | _formula_properties(t.get('then')))
        public = {
            'theorem': t,
            'properties': [Pmeta[p] for p in ps if p in Pmeta],
        }
        target_txt[tid] = json.dumps(public, sort_keys=True, ensure_ascii=False)
    return sorted(S), txt, cov, U, target_txt


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

    prop_txt = {}
    p_root = b / 'property'
    if p_root.exists():
        for pd in sorted(x for x in p_root.iterdir() if x.is_dir()):
            try:
                pid = int(pd.name.split('_')[-1])
                meta = yaml.safe_load((pd / 'data.yaml').read_text()) or {}
            except Exception:
                continue
            prop_txt[pid] = ' '.join([
                str(meta.get('name', '')), str(meta.get('definition', '')),
                str(meta.get('description', '')),
            ])

    rules = []
    target_txt = {}
    for f in sorted((b / 'logic').glob('*.yaml')):
        try:
            d = yaml.safe_load(f.read_text()) or {}
        except Exception:
            continue
        if d.get('active') is False:
            continue
        a, c = parse(d.get('hyps')), parse(d.get('concs'))
        if a and c:
            rid = f.stem
            rules.append((rid, a, c))
            if ring:
                pa, sa, va = a
                pc, sc, vc = c
                sem = f'antecedent={prop_txt.get(pa, pa)} side={sa} value={va} consequent={prop_txt.get(pc, pc)} side={sc} value={vc}'
            else:
                pa, va = a
                pc, vc = c
                sem = f'antecedent={prop_txt.get(pa, pa)} value={va} consequent={prop_txt.get(pc, pc)} value={vc}'
            target_txt[rid] = sem + ' ' + json.dumps(d, sort_keys=True, ensure_ascii=False)

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
    target_txt = {t: target_txt.get(t, str(t)) for t in U}
    return sorted(trs), txt, cov, U, target_txt


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
    if len(rems) == 1:
        return rems[0]
    w = [1.0] * P.shape[1] if weights is None else list(map(float, weights))
    f = ProbabilisticSetCoverFunction(
        n=len(rems), probs=P.tolist(), num_concepts=P.shape[1], concept_weights=w
    )
    # Submodlib requires budget to be strictly smaller than effective ground size.
    budget = min(HORIZON, len(rems) - 1)
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


def predicted_rarity_weights(Pfull, rems, residual_cols):
    if not residual_cols:
        return []
    P = np.asarray(Pfull[np.ix_(rems, residual_cols)], dtype=float)
    if P.size == 0:
        return [1.0] * len(residual_cols)
    prevalence = np.mean(P, axis=0)
    floor = 1.0 / max(2.0, float(len(rems)))
    vals = 1.0 / np.sqrt(np.maximum(prevalence, floor))
    vals = np.minimum(vals, 4.0 * np.median(vals) if len(vals) else 1.0)
    m = float(np.mean(vals)) if len(vals) else 1.0
    return (vals / max(m, 1e-12)).tolist()


class PairPosterior:
    """Leakage-free shared object×target posterior.

    It learns from every revealed object-target label but shares parameters across
    targets through public target text. This avoids the degenerate multi-output
    regression failure where an as-yet-uncovered target has an all-zero label column.
    """
    def __init__(self, object_texts, target_texts, seed):
        vo = HashingVectorizer(n_features=FEATURES, alternate_sign=True, norm='l2', ngram_range=(1, 2))
        vt = HashingVectorizer(n_features=FEATURES, alternate_sign=True, norm='l2', ngram_range=(1, 2))
        self.O = vo.transform(object_texts).toarray().astype(np.float32)
        self.T = vt.transform(target_texts).toarray().astype(np.float32)
        rr = np.random.default_rng(BASE_SEED)
        scale = 1.0 / math.sqrt(FEATURES)
        self.Ro = rr.normal(0.0, scale, size=(FEATURES, CROSS_FEATURES)).astype(np.float32)
        self.Rt = rr.normal(0.0, scale, size=(FEATURES, CROSS_FEATURES)).astype(np.float32)
        self.Op = self.O @ self.Ro
        self.Tp = self.T @ self.Rt
        self.clf = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=PAIR_ALPHA,
            learning_rate='optimal', fit_intercept=True,
            random_state=BASE_SEED + seed, average=False,
        )
        self.fitted = False

    def features(self, object_idx, target_idx):
        oi = np.asarray(object_idx, dtype=int)
        ti = np.asarray(target_idx, dtype=int)
        no, nt = len(oi), len(ti)
        if no == 0 or nt == 0:
            return np.zeros((0, FEATURES * 2 + CROSS_FEATURES), dtype=np.float32)
        O = np.repeat(self.O[oi], nt, axis=0)
        T = np.tile(self.T[ti], (no, 1))
        C = np.repeat(self.Op[oi], nt, axis=0) * np.tile(self.Tp[ti], (no, 1))
        return np.concatenate([O, T, C], axis=1).astype(np.float32, copy=False)

    def update(self, object_idx, row):
        ti = np.arange(len(row), dtype=int)
        X = self.features([object_idx], ti)
        y = (np.asarray(row) > 0.5).astype(int)
        pos = int(y.sum())
        neg = len(y) - pos
        sw = np.ones(len(y), dtype=float)
        if pos > 0 and neg > 0:
            sw[y == 1] = min(12.0, max(1.0, neg / pos))
        if not self.fitted:
            self.clf.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=sw)
            self.fitted = True
        else:
            self.clf.partial_fit(X, y, sample_weight=sw)

    def predict_matrix(self, object_idx, target_idx):
        oi = list(object_idx)
        ti = list(target_idx)
        if not oi or not ti:
            return np.zeros((len(oi), len(ti)), dtype=float)
        X = self.features(oi, ti)
        if not self.fitted:
            p = np.full(X.shape[0], 0.5, dtype=float)
        else:
            p = self.clf.predict_proba(X)[:, 1]
        return np.clip(p.reshape(len(oi), len(ti)), 1e-6, 1.0 - 1e-6)


def run_seed(ids, txt, cov, U, target_txt, method, seed):
    targets = sorted(U)
    env = OracleEnv(ids, cov, targets)
    n = len(ids)
    Yobs = np.zeros((n, len(targets)), dtype=float)
    posterior = PairPosterior([txt[i] for i in ids], [target_txt[t] for t in targets], seed)
    rem = set(range(n))
    obs, order, fractions = [], [], []
    seen = np.zeros(len(targets), dtype=bool)
    rng = random.Random(seed)
    first_prediction_audit = None

    def commit_and_reveal(j):
        # j is irrevocably selected before this call; chooser functions never receive env/cov.
        o, row = env.reveal_after_commit(j)
        Yobs[j] = row
        posterior.update(j, row)
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
        rems = sorted(rem)
        residual_cols = np.flatnonzero(~seen).tolist()
        Pred = posterior.predict_matrix(rems, residual_cols)
        Pfull = np.zeros((n, len(targets)), dtype=float)
        Pfull[np.ix_(rems, residual_cols)] = Pred

        if first_prediction_audit is None:
            col_std = np.std(Pred, axis=0) if Pred.size else np.array([])
            first_prediction_audit = {
                'remaining_objects': len(rems),
                'residual_targets': len(residual_cols),
                'mean_probability': float(np.mean(Pred)) if Pred.size else 0.0,
                'global_std': float(np.std(Pred)) if Pred.size else 0.0,
                'mean_target_across_object_std': float(np.mean(col_std)) if len(col_std) else 0.0,
                'nonconstant_target_columns': int(np.sum(col_std > 1e-8)) if len(col_std) else 0,
            }

        if method == 'psc':
            j = choose_psc(Pfull, rems, residual_cols)
        elif method == 'wpsc':
            j = choose_psc(Pfull, rems, residual_cols, predicted_rarity_weights(Pfull, rems, residual_cols))
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

    hits = {}
    for q in TH:
        hits[str(q)] = next((i + 1 for i, v in enumerate(fractions) if v + 1e-12 >= q), None)
    terminal = fractions[-1] if fractions else 0.0
    padded = fractions + [terminal] * max(0, n - len(fractions))
    auc = float(np.mean(padded)) if padded else 0.0
    seq_hash = hashlib.sha256('\n'.join(map(str, order)).encode()).hexdigest()
    return {
        'seed': seed, 'hits': hits, 'auc': auc, 'queries_run': len(order),
        'order_sha256': seq_hash, 'order': order,
        'first_adaptive_prediction_audit': first_prediction_audit,
    }


def run_domain(name, data, method):
    ids, txt, cov, U, target_txt = data
    seeds = [run_seed(ids, txt, cov, U, target_txt, method, s) for s in range(5)]
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
        'uncovered_target_zero_column_degeneracy_fixed': True,
        'public_target_text_only': True,
        'horizon': HORIZON,
        'hits_mean': means,
        'auc_mean': float(np.mean([x['auc'] for x in seeds])),
        'seeds': seeds,
    }


UPSTREAMS = {
    'psc': 'decile-team/submodlib ProbabilisticSetCoverFunction + LazyGreedy',
    'wpsc': 'decile-team/submodlib ProbabilisticSetCoverFunction + LazyGreedy with posterior-predicted rarity weights',
    'bestfirst': 'simpleai-team/simpleai greedy best-first search',
    'astar': 'simpleai-team/simpleai A* search',
    'cpsat': 'Google OR-Tools CP-SAT on posterior-sampled stochastic maximum coverage',
    'cbc': 'Google OR-Tools CBC branch-and-bound/branch-and-cut on the same posterior-sampled model',
    'mcts': 'pbsinclair42/MCTS with full real action space and soft expected-coverage transitions',
}

res = {
    'protocol': (
        'Native online redo after invalidating the shared 10-target/16-candidate binary adapter. '
        'All methods share only the legal information state, a pairwise online logistic posterior over public object/target text, warm-start, horizon=4, and post-commit reveal rule. '
        'All residual targets remain continuous probabilities. Every remaining candidate is eligible as the real next action. '
        'Method-specific search/optimization semantics are preserved; tree-search methods prune only hypothetical future branches.'
    ),
    'method': METHOD,
    'upstream': UPSTREAMS[METHOD],
    'frozen': {
        'pair_alpha': PAIR_ALPHA,
        'hash_features_per_side': FEATURES,
        'cross_features': CROSS_FEATURES,
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
