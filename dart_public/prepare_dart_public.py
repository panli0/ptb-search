#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import zlib
from collections import Counter
from pathlib import Path

import networkx as nx
import yaml

FACT_RE = re.compile(r'^ring_deduced\(\s*"(has|lacks)"\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$')
ID_RE = re.compile(r'(\d+)$')
SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]


def yload(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def oid(tag):
    m = ID_RE.search(str(tag))
    if not m:
        raise ValueError(f'cannot parse id from {tag!r}')
    return int(m.group(1))


def parse_fact(text):
    m = FACT_RE.fullmatch((text or '').strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def both_state(left, right):
    if left is False or right is False:
        return -1
    if left is True and right is True:
        return 1
    return 0


def load_properties(root: Path):
    props = {}
    for d in sorted((root / 'db/ringapp/property').glob('PROP_*')):
        p = d / 'data.yaml'
        if not p.exists():
            continue
        data = yload(p) or {}
        pid = oid(d.name)
        props[pid] = {
            'name': str(data.get('name', '')),
            'symmetric': bool(data.get('symmetric', False)),
        }
    if not props:
        raise RuntimeError('no properties loaded')
    return props


def is_both_has(fact, props):
    if not fact or fact[0] != 'has':
        return False
    _, side, prop = fact
    if prop not in props:
        return False
    if side == 1:
        return True
    if side == 0 and props[prop]['symmetric']:
        return True
    return False


def load_edges(root: Path, props):
    rows = []
    excluded = Counter()
    for p in sorted((root / 'db/ringapp/logic').glob('LOG_*.yaml')):
        d = yload(p) or {}
        if not d.get('active', True):
            excluded['inactive'] += 1
            continue
        if d.get('symmetric') is not True:
            excluded['logic_not_symmetric'] += 1
            continue
        h = parse_fact(d.get('hyps'))
        c = parse_fact(d.get('concs'))
        if not h or not c:
            excluded['non_atomic_or_multi_fact'] += 1
            continue
        if not is_both_has(h, props) or not is_both_has(c, props):
            excluded['not_positive_both_sides'] += 1
            continue
        u, v = h[2], c[2]
        variety = int(d.get('variety', 0) or 0)
        lid = oid(p.stem)
        rows.append((u, v, lid, 'forward'))
        if variety == 1:
            rows.append((v, u, lid, 'equivalence_reverse'))
        elif variety != 0:
            excluded['unknown_variety'] += 1
    pairs = sorted(set((u, v) for u, v, _, _ in rows if u != v))
    if not pairs:
        raise RuntimeError(f'no accepted edges; excluded={dict(excluded)}')
    return rows, pairs, dict(excluded)


def build_scc(props, pairs):
    G = nx.DiGraph()
    G.add_nodes_from(sorted(props))
    G.add_edges_from(pairs)
    comps = list(nx.strongly_connected_components(G))
    comps.sort(key=lambda s: min(s))
    p2c = {p: i for i, comp in enumerate(comps) for p in comp}
    D = nx.DiGraph()
    D.add_nodes_from(range(len(comps)))
    for u, v in pairs:
        a, b = p2c[u], p2c[v]
        if a != b:
            D.add_edge(a, b)
    if not nx.is_directed_acyclic_graph(D):
        raise RuntimeError('SCC quotient is not DAG')
    succ = []
    pred = []
    for c in range(len(comps)):
        sm = 1 << c
        pm = 1 << c
        for d in nx.descendants(D, c):
            sm |= 1 << d
        for a in nx.ancestors(D, c):
            pm |= 1 << a
        succ.append(sm)
        pred.append(pm)
    return comps, p2c, D, succ, pred


def load_rings(root: Path, p2c, succ, pred):
    rings = {}
    contradictions = []
    missing_properties_files = 0
    for d in sorted((root / 'db/ringapp/ring').glob('RING_*')):
        rid = oid(d.name)
        data = yload(d / 'data.yaml') or {}
        pp = d / 'properties.yaml'
        pdata = yload(pp) if pp.exists() else None
        if not pp.exists():
            missing_properties_files += 1
        pdata = pdata or {}
        tm = 0
        fm = 0
        for key, rec in pdata.items():
            try:
                pid = oid(key)
            except Exception:
                continue
            if pid not in p2c:
                continue
            rec = rec or {}
            st = both_state(rec.get('has_on_left'), rec.get('has_on_right'))
            c = p2c[pid]
            if st == 1:
                tm |= 1 << c
            elif st == -1:
                fm |= 1 << c
        if tm & fm:
            contradictions.append((rid, 'initial_scc', (tm & fm).bit_count()))
            continue
        t0, f0 = tm, fm
        m = t0
        while m:
            z = m & -m
            c = z.bit_length() - 1
            tm |= succ[c]
            m -= z
        m = f0
        while m:
            z = m & -m
            c = z.bit_length() - 1
            fm |= pred[c]
            m -= z
        if tm & fm:
            contradictions.append((rid, 'closure', (tm & fm).bit_count()))
            continue
        rings[rid] = {
            'name': str(data.get('name', '')),
            't': tm,
            'f': fm,
        }
    if contradictions:
        raise RuntimeError(f'official truth/logic contradiction examples={contradictions[:20]} total={len(contradictions)}')
    if not rings:
        raise RuntimeError('no rings loaded')
    return rings, missing_properties_files


def families_from_rings(rings):
    key_to_f = {}
    families = []
    ring_family = {}
    for rid, r in sorted(rings.items()):
        key = (r['t'], r['f'])
        if key not in key_to_f:
            key_to_f[key] = len(families)
            families.append({'t': r['t'], 'f': r['f'], 'rings': []})
        fid = key_to_f[key]
        families[fid]['rings'].append(rid)
        ring_family[rid] = fid
    return families, ring_family


def witness_map(rings, succ):
    witnesses = {}
    for rid, r in sorted(rings.items()):
        mt = r['t']
        while mt:
            z = mt & -mt
            a = z.bit_length() - 1
            mt -= z
            mf = r['f']
            while mf:
                y = mf & -mf
                b = y.bit_length() - 1
                mf -= y
                if a == b:
                    continue
                if (succ[a] >> b) & 1:
                    raise RuntimeError(f'closure contradiction survived: ring={rid}, {a}->{b}')
                witnesses.setdefault((a, b), []).append(rid)
    return witnesses


def actions_for_backend(witnesses, ring_family, backend):
    out = []
    for (a, b), ws in sorted(witnesses.items()):
        ws = sorted(ws)
        rid = ws[0] if backend == 'lowest' else ws[-1]
        out.append((a, b, ring_family[rid]))
    return out


def cap_by_source(actions, source_ids, cap=60):
    wanted = set(source_ids)
    by = {s: [] for s in wanted}
    for a, b, f in actions:
        if a in wanted:
            by[a].append((a, b, f))
    out = []
    for s in sorted(by):
        out.extend(sorted(by[s], key=lambda x: (x[1], x[2]))[:cap])
    return out


def compact_split(actions, seed, family_count):
    source_counts = Counter(a for a, _, _ in actions)
    eligible = sorted(s for s, n in source_counts.items() if n >= 10)
    if len(eligible) < 6:
        raise RuntimeError(f'too few eligible sources: {len(eligible)}')
    ntest = min(20, max(3, len(eligible) // 5))
    import random
    test_sources = sorted(random.Random(seed).sample(eligible, ntest))
    test_set = set(test_sources)
    train_sources = [s for s in eligible if s not in test_set]
    train = cap_by_source(actions, train_sources, 60)
    test = cap_by_source(actions, test_sources, 60)
    counts = Counter(f for _, _, f in train)
    sparse_counts = [[int(f), int(n)] for f, n in sorted(counts.items())]
    return {
        'seed': seed,
        'eligible_sources': len(eligible),
        'train_sources': len(train_sources),
        'test_sources': len(test_sources),
        'train_actions': len(train),
        'test_actions': len(test),
        'train_counts': sparse_counts,
        'test': [[int(a), int(b), int(f)] for a, b, f in test],
        'train_family_count': len(counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--commit', required=True)
    args = ap.parse_args()
    root = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    props = load_properties(root)
    edge_rows, pairs, excluded = load_edges(root, props)
    comps, p2c, D, succ, pred = build_scc(props, pairs)
    rings, missing_props = load_rings(root, p2c, succ, pred)
    families, ring_family = families_from_rings(rings)
    witnesses = witness_map(rings, succ)

    payload = {
        'meta': {
            'upstream': 'rschwiebert/dart_data',
            'commit': args.commit,
            'raw_properties': len(props),
            'accepted_logic_rows': len(edge_rows),
            'accepted_direct_property_pairs': len(pairs),
            'scc_nodes': len(comps),
            'direct_interclass_edges': D.number_of_edges(),
            'official_rings': len(rings),
            'truth_signature_families': len(families),
            'witnessable_pairs': len(witnesses),
            'missing_ring_properties_files': missing_props,
            'excluded_logic': excluded,
            'semantics': 'official atomic positive both-sided implication/equivalence only; SCC compression; forward true and contrapositive false closure; unknown remains unknown',
        },
        'dag_edges': [[int(a), int(b)] for a, b in sorted(D.edges())],
        'families': [[format(f['t'], 'x'), format(f['f'], 'x')] for f in families],
        'backends': {},
    }

    for backend in ['lowest', 'highest']:
        actions = actions_for_backend(witnesses, ring_family, backend)
        payload['backends'][backend] = {
            'actions_total': len(actions),
            'splits': [compact_split(actions, s, len(families)) for s in SEEDS],
        }

    raw = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode('utf-8')
    packed = base64.b85encode(zlib.compress(raw, 9)).decode('ascii')
    (out / 'DART_COMPACT_INPUT.b85').write_text(packed + '\n', encoding='ascii')
    (out / 'SUMMARY.json').write_text(json.dumps(payload['meta'] | {
        'lowest_actions_total': payload['backends']['lowest']['actions_total'],
        'highest_actions_total': payload['backends']['highest']['actions_total'],
        'compact_json_bytes': len(raw),
        'compact_b85_bytes': len(packed),
        'splits_lowest': [{k: s[k] for k in ['seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count']} for s in payload['backends']['lowest']['splits']],
        'splits_highest': [{k: s[k] for k in ['seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count']} for s in payload['backends']['highest']['splits']],
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print((out / 'SUMMARY.json').read_text())


if __name__ == '__main__':
    main()
