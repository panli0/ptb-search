#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(sys.argv[1])
PI = Path(sys.argv[2])
MAS = Path(sys.argv[3])
OUT = Path(sys.argv[4])
MODE = sys.argv[5]
DOMAIN_ARG = sys.argv[6] if len(sys.argv) > 6 else 'all'
TH = (0.5, 0.7, 0.9, 0.95, 0.99, 1.0)
BASE_SEED = 20260818

if MAS.exists():
    sys.path.insert(0, str(MAS))


def atomr(s):
    m = re.fullmatch(r'\s*ring_deduced\("(has|lacks)",\s*(\d+),\s*(\d+)\)\s*', str(s or ''))
    return (int(m[3]), int(m[2]), m[1] == 'has') if m else None


def atomm(s):
    m = re.fullmatch(r'\s*module_deduced\("(has|lacks)",\s*(\d+)\)\s*', str(s or ''))
    return (int(m[2]), m[1] == 'has') if m else None


def atoms_pi(f):
    if not isinstance(f, dict):
        return []
    if f.get('kind') == 'atom':
        return [f.get('property')] if f.get('property') is not None else []
    out = []
    for z in f.get('subs', []) or []:
        out += atoms_pi(z)
    return out


@dataclass
class StrictEnv:
    ids: list
    _cov: dict

    def reveal_after_commit(self, object_id):
        # Sole truth interface. Policies choose object_id before this call.
        return set(self._cov[object_id])


def pi_data():
    d = json.loads(PI.read_text())
    S = {x['uid']: x for x in d['spaces']}
    T = {x['uid']: x for x in d['theorems']}
    P = {x['uid']: x for x in d['properties']}
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
    eval_U = set().union(*cov.values()) if cov else set()
    policy_targets = set(T)  # public official universe, not outcome-defined subset
    txt = {s: ' '.join(str(S[s].get(k, '') or '') for k in ('name', 'description', 'aliases')) for s in S}
    tn = {}
    edges = []
    for tid, t in T.items():
        a, b = atoms_pi(t.get('when')), atoms_pi(t.get('then'))
        tn[tid] = set(a + b)
        for x in a:
            for y in b:
                if x in P and y in P:
                    edges.append((x, y))
    nodes = sorted(P)
    return sorted(S), txt, cov, eval_U, policy_targets, tn, nodes, edges


def dart(dom):
    ring = dom == 'ring'
    b = ROOT / ('db/ringapp' if ring else 'db/moduleapp')
    r = b / ('ring' if ring else 'module')
    parse = atomr if ring else atomm
    tr, txt = {}, {}
    for od in sorted(x for x in r.iterdir() if x.is_dir()):
        pp, dp = od / 'properties.yaml', od / 'data.yaml'
        if not pp.exists():
            continue
        try:
            p = yaml.safe_load(pp.read_text()) or {}
            m = yaml.safe_load(dp.read_text()) or {}
        except Exception:
            continue
        q = {}
        if ring:
            for k, v in p.items():
                try:
                    i = int(k.split('_')[-1])
                except Exception:
                    continue
                l, rr = v.get('has_on_left'), v.get('has_on_right')
                if isinstance(l, bool) and isinstance(rr, bool):
                    q[(i, 0)] = l
                    q[(i, 1)] = rr
                    q[(i, 2)] = l if l == rr else None
            txt[od.name] = ' '.join([str(m.get('name', '')), str(m.get('description', '')),
                                      ' '.join(map(str, m.get('keywords') or [])),
                                      'comm=' + str(m.get('is_commutative'))])
        else:
            for k, v in p.items():
                try:
                    i = int(k.split('_')[-1])
                except Exception:
                    continue
                if isinstance(v.get('has'), bool):
                    q[i] = v['has']
            txt[od.name] = ' '.join([od.name, str(m.get('name', '')), str(m.get('description', '')),
                                      'ring=' + str(m.get('ring', '')),
                                      'opp=' + str(m.get('opposite_ring', ''))])
        tr[od.name] = q

    rules, tn, edges = [], {}, []
    for f in sorted((b / 'logic').glob('*.yaml')):
        try:
            d = yaml.safe_load(f.read_text()) or {}
        except Exception:
            continue
        if d.get('active') is False:
            continue
        a, c = parse(d.get('hyps')), parse(d.get('concs'))
        if not (a and c):
            continue
        rules.append((f.stem, a, c))
        if ring:
            na, nc = f'{a[0]}:{a[1]}', f'{c[0]}:{c[1]}'
        else:
            na, nc = str(a[0]), str(c[0])
        tn[f.stem] = {na, nc}
        edges.append((na, nc))

    cov = {o: set() for o in tr}
    for o, q in tr.items():
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
    eval_U = set().union(*cov.values()) if cov else set()
    policy_targets = {rid for rid, _, _ in rules}
    nodes = sorted({x for e in edges for x in e})
    return sorted(tr), txt, cov, eval_U, policy_targets, tn, nodes, edges


def evaluate(order, cov, eval_U):
    seen = set()
    vals = []
    for o in order:
        seen |= cov[o]
        vals.append(len(seen & eval_U) / max(1, len(eval_U)))
    hits = {str(q): next((i + 1 for i, v in enumerate(vals) if v + 1e-12 >= q), None) for q in TH}
    return hits, float(np.mean(vals)) if vals else 0.0


def target_policy(ids, txt, env, policy_targets, mode, seed):
    rng = random.Random(seed)
    targets = sorted(policy_targets)
    ti = {t: i for i, t in enumerate(targets)}
    V = HashingVectorizer(n_features=48, alternate_sign=False, norm='l2', ngram_range=(1, 2))
    X = V.transform([txt[i] for i in ids]).toarray()
    ix = {o: i for i, o in enumerate(ids)}
    d = 50
    Ai = np.eye(d) / 2.0
    B = np.zeros((d, len(targets)))
    rem = set(ids)
    seen = set()
    order = []

    def update(o, result):
        nonlocal Ai, B
        cur = len(seen) / max(1, len(targets))
        z = np.r_[X[ix[o]], cur, 1.0]
        az = Ai @ z
        Ai = Ai - np.outer(az, az) / (1 + z @ az)
        y = np.zeros(len(targets))
        for t in result:
            if t in ti:
                y[ti[t]] = 1.0
        B += np.outer(z, y)

    warm = min(8, max(3, len(ids) // 20))
    for o in rng.sample(ids, warm):
        result = env.reveal_after_commit(o)
        update(o, result)
        seen |= result
        order.append(o)
        rem.remove(o)

    while rem:
        theta = Ai @ B
        uncovered = np.array([t not in seen for t in targets])
        cur = len(seen) / max(1, len(targets))
        C = []
        for o in rem:
            z = np.r_[X[ix[o]], cur, 1.0]
            p = np.clip(z @ theta, 0, 1)
            mu = float(p[uncovered].sum())
            unc = math.sqrt(max(0.0, float(z @ Ai @ z))) * math.sqrt(max(1, int(uncovered.sum())))
            score = mu + 0.6 * unc if mode == 'target_ucb' else (
                rng.random() * len(targets) if mode == 'target_eps' and rng.random() < 0.08 else mu)
            C.append((score, rng.random(), o))
        b = max(C)[2]
        result = env.reveal_after_commit(b)
        update(b, result)
        seen |= result
        order.append(b)
        rem.remove(b)
    return order


def scalar_policy(ids, txt, env, policy_target_count, kind, seed):
    rng = random.Random(seed)
    V = HashingVectorizer(n_features=32, alternate_sign=False, norm='l2', ngram_range=(1, 2))
    X = V.transform([txt[i] for i in ids]).toarray()
    ix = {o: i for i, o in enumerate(ids)}
    rem = set(ids)
    seen = set()
    order = []
    hist = []
    warm = min(6, max(2, len(ids) // 20))
    d = 34
    Ai = np.eye(d) / 3.0
    bv = np.zeros(d)

    def upd(z, y):
        nonlocal Ai, bv
        az = Ai @ z
        Ai = Ai - np.outer(az, az) / (1 + z @ az)
        bv += z * y

    for o in rng.sample(ids, warm):
        result = env.reveal_after_commit(o)
        y = len(result - seen)  # raw revealed marginal count; no outcome-defined denominator
        z = np.r_[X[ix[o]], len(seen) / max(1, policy_target_count), 1.0]
        upd(z, y)
        hist.append((X[ix[o]], y))
        seen |= result
        order.append(o)
        rem.remove(o)

    while rem:
        cur = len(seen) / max(1, policy_target_count)
        theta = Ai @ bv
        C = []
        if kind == 'knn':
            H = np.array([h[0] for h in hist])
            yy = np.array([h[1] for h in hist])
            for o in rem:
                sm = H @ X[ix[o]]
                top = np.argsort(sm)[-min(5, len(sm)):]
                w = np.maximum(sm[top], 0) + 1e-6
                C.append((float(w @ yy[top] / w.sum()), rng.random(), o))
        else:
            for o in rem:
                z = np.r_[X[ix[o]], cur, 1.0]
                mu = float(theta @ z)
                unc = math.sqrt(max(0.0, float(z @ Ai @ z)))
                score = mu + 1.5 * unc if kind == 'linucb' else (
                    rng.random() * 1000 if kind == 'epsridge' and rng.random() < 0.12 else mu)
                C.append((score, rng.random(), o))
        b = max(C)[2]
        result = env.reveal_after_commit(b)
        y = len(result - seen)
        z = np.r_[X[ix[b]], cur, 1.0]
        upd(z, y)
        hist.append((X[ix[b]], y))
        seen |= result
        order.append(b)
        rem.remove(b)
    return order


def pi_family(meta):
    z = ' '.join(str(meta.get(k, '') or '') for k in ('name', 'description', 'aliases')).lower()
    for k in ['discrete', 'indiscrete', 'sierpi', 'ordinal', 'metric', 'product', 'sum', 'cofinite',
              'cocountable', 'fort', 'compact', 'connected', 'line', 'plane', 'circle', 'sequence',
              'real', 'rational', 'integer', 'finite']:
        if k in z:
            return k
    return 'other'


def tpe_policy(ids, public_meta, env, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    rng = random.Random(seed)
    fs = defaultdict(list)
    for s in ids:
        fs[pi_family(public_meta[s])].append(s)
    for f in fs:
        fs[f].sort()
    rem = {f: list(v) for f, v in fs.items()}
    names = sorted(fs)
    seen = set()
    order = []
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=max(5, len(names))))
    while len(order) < len(ids):
        av = [f for f in names if rem[f]]
        if not av:
            break
        tr = study.ask()
        f = tr.suggest_categorical('family', names)
        if not rem[f]:
            f = rng.choice(av)
        q = tr.suggest_float('within_family_quantile', 0, 1)
        j = min(len(rem[f]) - 1, int(q * len(rem[f])))
        s = rem[f].pop(j)
        result = env.reveal_after_commit(s)
        gain = len(result - seen)  # raw count; removes hidden U scaling
        seen |= result
        order.append(s)
        study.tell(tr, gain)
    return order


def topology(nodes, edges):
    deps = defaultdict(list)
    for a, b in edges:
        deps[b].append(a)
    return [{'name': n, 'dependencies': sorted(set(deps[n]))} for n in nodes]


def maspob_policy(ids, txt, env, policy_targets, tn, nodes, edges, seed):
    import torch
    from scripts.gnn_model import WorkflowGAT, initialize_fisher, update_fisher, compute_prediction_and_uncertainty

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    n, N = len(ids), len(nodes)
    ni = {x: i for i, x in enumerate(nodes)}
    indeg = Counter(b for a, b in edges)
    outdeg = Counter(a for a, b in edges)
    mx = max(1, max([*indeg.values(), *outdeg.values(), 1]))
    V = HashingVectorizer(n_features=3, alternate_sign=False, norm='l2', ngram_range=(1, 2))
    TX = V.transform([txt[o] for o in ids]).toarray().astype('float32')
    oi = {o: i for i, o in enumerate(ids)}
    target_inc = {t: [ni[x] for x in tn.get(t, set()) if x in ni] for t in policy_targets}
    topo = topology(nodes, edges)
    device = torch.device('cpu')
    model = WorkflowGAT(embedding_dim=6, num_operators=N, hidden_dim=4, num_gnn_layers=1,
                        num_heads=1, topology=topo, dropout=0.0, bidirectional=True,
                        use_sigmoid=True).to(device)
    frozen = copy.deepcopy(model).to(device)
    frozen.eval()
    fisher = initialize_fisher('neural', frozen_model=frozen, lambda_reg=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossfn = torch.nn.MSELoss()
    hist_x, hist_y = [], []
    seen = set()
    rem = set(ids)
    order = []
    policy_n = max(1, len(policy_targets))

    def emb(o, seen_now):
        z = np.zeros((N, 6), dtype='float32')
        base = TX[oi[o]]
        unc = set(policy_targets) - seen_now
        res = np.zeros(N, dtype='float32')
        for t in unc:
            for j in target_inc.get(t, []):
                res[j] += 1
        if res.max() > 0:
            res /= res.max()
        for j, node in enumerate(nodes):
            z[j, :3] = base
            z[j, 3] = indeg[node] / mx
            z[j, 4] = outdeg[node] / mx
            z[j, 5] = res[j]
        return torch.tensor(z.reshape(-1), dtype=torch.float32, device=device)

    warm = min(8, max(3, n // 20))
    for o in rng.sample(sorted(rem), warm):
        x = emb(o, seen)
        result = env.reveal_after_commit(o)
        gain = len(result - seen) / policy_n
        hist_x.append(x.detach())
        hist_y.append(gain)
        seen |= result
        order.append(o)
        rem.remove(o)
        g = compute_prediction_and_uncertainty('neural', model, x, frozen_model=frozen, fisher_matrix=fisher)[2]
        fisher = update_fisher('neural', fisher, g, fisher_coef=10)

    while rem:
        model.train()
        X = torch.stack(hist_x)
        Y = torch.tensor(hist_y, dtype=torch.float32, device=device)
        for _ in range(8):
            opt.zero_grad()
            pred = model(X).reshape(-1)
            loss = lossfn(pred, Y)
            loss.backward()
            opt.step()
        model.eval()
        best, bestinfo = None, None
        for o in rem:
            x = emb(o, seen)
            pred, unc, g = compute_prediction_and_uncertainty('neural', model, x, frozen_model=frozen, fisher_matrix=fisher)
            score = pred + 0.2 * unc
            key = (score, rng.random(), o)
            if best is None or key > best:
                best, bestinfo = key, (o, x, g)
        o, x, g = bestinfo
        result = env.reveal_after_commit(o)
        gain = len(result - seen) / policy_n
        hist_x.append(x.detach())
        hist_y.append(gain)
        fisher = update_fisher('neural', fisher, g, fisher_coef=10)
        seen |= result
        order.append(o)
        rem.remove(o)
    return order


def summarize_method(name, data, orders):
    ids, txt, cov, eval_U, policy_targets, tn, nodes, edges = data
    vv = [evaluate(order, cov, eval_U) for order in orders]
    return {
        'method': name,
        'uses_unqueried_truth': False,
        'uses_outcome_defined_target_universe_for_policy': False,
        'policy_target_count': len(policy_targets),
        'evaluation_coverable_target_count': len(eval_U),
        'hits_mean': {str(q): float(np.mean([x[0][str(q)] for x in vv if x[0][str(q)] is not None])) for q in TH},
        'auc_mean': float(np.mean([x[1] for x in vv])),
    }


def run_domain(name, data, family):
    ids, txt, cov, eval_U, policy_targets, tn, nodes, edges = data
    env = StrictEnv(ids, cov)
    if family == 'target':
        methods = {}
        for mode in ('target_mean', 'target_ucb', 'target_eps'):
            orders = [target_policy(ids, txt, env, policy_targets, mode, s) for s in range(10)]
            methods[mode] = summarize_method(mode, data, orders)
        return {'domain': name, 'methods': methods}
    if family == 'scalar':
        methods = {}
        for mode in ('ridge', 'epsridge', 'linucb', 'knn'):
            orders = [scalar_policy(ids, txt, env, len(policy_targets), mode, s) for s in range(5)]
            methods[mode] = summarize_method(mode, data, orders)
        return {'domain': name, 'methods': methods}
    if family == 'maspob':
        orders = [maspob_policy(ids, txt, env, policy_targets, tn, nodes, edges, s) for s in range(3)]
        return {'domain': name, 'methods': {'maspob_gnn_ucb': summarize_method('maspob_gnn_ucb', data, orders)}}
    raise ValueError(family)


def main():
    if MODE == 'tpe':
        data = pi_data()
        ids, txt, cov, eval_U, policy_targets, tn, nodes, edges = data
        raw = json.loads(PI.read_text())
        meta = {x['uid']: x for x in raw['spaces']}
        env = StrictEnv(ids, cov)
        orders = [tpe_policy(ids, meta, env, s) for s in range(5)]
        res = {'protocol': 'Optuna TPE redo with raw revealed marginal gain; policy never receives hindsight coverable target universe.',
               'domains': [{'domain': 'pi-base', 'methods': {'tpe': summarize_method('tpe', data, orders)}}]}
    else:
        domains = [DOMAIN_ARG] if DOMAIN_ARG != 'all' else (['ring', 'module'] if MODE == 'scalar' else ['pi-base', 'ring', 'module'])
        rows = []
        for dom in domains:
            data = pi_data() if dom == 'pi-base' else dart(dom)
            rows.append(run_domain(dom, data, MODE))
        res = {
            'protocol': ('Outcome-universe-safe legacy baseline redo. Policy target space is all public official theorems/unary rules; '
                         'hindsight coverable subset is used only after trajectory for evaluation. Query result is revealed only after action commit.'),
            'family': MODE,
            'domain_arg': DOMAIN_ARG,
            'domains': rows,
        }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
