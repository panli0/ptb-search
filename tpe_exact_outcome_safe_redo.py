#!/usr/bin/env python3
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import optuna

PI = Path(sys.argv[1])
OUT = Path(sys.argv[2])
TH = (0.5, 0.7, 0.9, 0.95, 0.99, 1.0)

d = json.loads(PI.read_text())
props = d.get('properties', [])
theorems = d.get('theorems', [])
spaces = d.get('spaces', [])
traits = d.get('traits', [])
uid = lambda x: x.get('uid') or x.get('id')
P = {uid(x): x for x in props}
S = {uid(x): x for x in spaces}
T = {uid(x): x for x in theorems}
truth = defaultdict(dict)
for x in traits:
    s, p, v = x.get('space'), x.get('property'), x.get('value')
    if s in S and p in P and isinstance(v, bool):
        truth[s][p] = v


def feval(f, tr):
    if not isinstance(f, dict):
        return None
    k = f.get('kind')
    if k == 'atom':
        p, want = f.get('property'), f.get('value')
        if p not in tr or not isinstance(want, bool):
            return None
        return tr[p] == want
    vals = [feval(x, tr) for x in f.get('subs', [])]
    if k == 'and':
        if any(v is False for v in vals):
            return False
        if all(v is True for v in vals):
            return True
        return None
    if k == 'or':
        if any(v is True for v in vals):
            return True
        if all(v is False for v in vals):
            return False
        return None
    return None


covers = {s: set() for s in S}
for sid, tr in truth.items():
    for tid, t in T.items():
        if feval(t.get('then'), tr) is True and feval(t.get('when'), tr) is False:
            covers[sid].add(tid)
eval_universe = set().union(*covers.values()) if covers else set()
ids = sorted(S)


def txt(s):
    x = S[s]
    return ' '.join(str(x.get(k, '') or '') for k in ('name', 'description', 'aliases')).lower()


def card(s):
    z = txt(s)
    for pat, n in [(r'one[- ]point|singleton', 1),
                   (r'two[- ]point|two[- ]element|two elements|\\?\{0\s*,\s*1\\?\}', 2)]:
        if re.search(pat, z):
            return n
    m = re.search(r'\b(\d{1,3})[- ](?:point|element)', z)
    return int(m.group(1)) if m else 10**9


def family(s):
    z = txt(s)
    keys = ['discrete', 'indiscrete', 'sierpi', 'ordinal', 'metric', 'product', 'sum', 'cofinite',
            'cocountable', 'fort', 'compact', 'connected', 'line', 'plane', 'circle', 'sequence',
            'real', 'rational', 'integer', 'finite']
    for k in keys:
        if k in z:
            return k
    c = card(s)
    if c <= 2:
        return 'finite2'
    if c <= 5:
        return 'finite3_5'
    if c < 10**9:
        return 'finite6plus'
    return 'other'


def curve(order):
    seen = set()
    vals = []
    for s in order:
        seen |= covers[s]
        vals.append(len(seen & eval_universe) / max(1, len(eval_universe)))
    hits = {str(q): next((i + 1 for i, v in enumerate(vals) if v + 1e-12 >= q), None) for q in TH}
    return hits, sum(vals) / max(1, len(vals))


def tpe_policy(seed=0):
    # Exact old public search space/order. The sole scientific change is reward:
    # raw revealed marginal count instead of dividing by hindsight |eval_universe|.
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    fs = defaultdict(list)
    for s in ids:
        fs[family(s)].append(s)
    for f in fs:
        fs[f].sort(key=lambda s: (card(s), s))
    rem = {f: list(v) for f, v in fs.items()}
    names = sorted(fs)
    rng = random.Random(seed)
    seen = set()
    order = []
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(5, len(names))),
    )
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
        # Commit s first; only then reveal this queried object's result.
        result = covers[s]
        gain = len(result - seen)
        seen |= result
        order.append(s)
        study.tell(tr, gain)
    return order


# Preserve old formal protocol: one seed, seed=0.
order = tpe_policy(seed=0)
hits, auc = curve(order)
res = {
    'protocol': ('Exact old Optuna TPE search space, family/cardinality rules, ordering and seed=0. '
                 'Only change: reward is raw post-query marginal coverage count, removing hindsight coverable-universe scaling.'),
    'method': 'tpe',
    'seed': 0,
    'uses_unqueried_truth': False,
    'uses_outcome_defined_target_universe_for_policy': False,
    'policy_target_count': len(T),
    'evaluation_coverable_target_count': len(eval_universe),
    'hits': hits,
    'auc': auc,
    'order_sha256_note': 'trajectory retained in order for exact audit',
    'order': order,
}
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
