#!/usr/bin/env python3
import argparse, base64, json, random, re, zlib
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import yaml

FRONT = re.compile(r'^---\s*\n(.*?)\n---\s*(?:\n|$)', re.S)

def frontmatter(path):
    text = Path(path).read_text(encoding='utf-8')
    m = FRONT.match(text)
    if not m:
        return {}
    x = yaml.safe_load(m.group(1))
    return x if isinstance(x, dict) else {}

def atom_true(x):
    if not isinstance(x, dict) or len(x) != 1:
        return None
    k, v = next(iter(x.items()))
    if isinstance(k, str) and k.startswith('P') and v is True:
        return k
    return None

def setmask(items):
    m = 0
    for i in items:
        m |= 1 << i
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--commit', required=True)
    args = ap.parse_args()
    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    properties = sorted(p.stem for p in (src/'properties').glob('P*.md'))
    prop_set = set(properties)

    G = nx.DiGraph()
    G.add_nodes_from(properties)
    accepted = []
    raw_theorems = 0
    for p in sorted((src/'theorems').glob('T*.md')):
        raw_theorems += 1
        fm = frontmatter(p)
        a = atom_true(fm.get('if'))
        b = atom_true(fm.get('then'))
        if a in prop_set and b in prop_set:
            G.add_edge(a, b)
            accepted.append((a,b,p.stem))

    comps = [sorted(c) for c in nx.strongly_connected_components(G)]
    comps.sort(key=lambda c: c[0])
    node_of = {}
    for i,c in enumerate(comps):
        for p in c: node_of[p]=i

    H = nx.DiGraph()
    H.add_nodes_from(range(len(comps)))
    direct_inter = set()
    for a,b,_ in accepted:
        u,v=node_of[a],node_of[b]
        if u != v:
            H.add_edge(u,v)
            direct_inter.add((u,v))
    assert nx.is_directed_acyclic_graph(H)

    descendants = {u: set(nx.descendants(H,u))|{u} for u in H.nodes()}
    ancestors = {u: set(nx.ancestors(H,u))|{u} for u in H.nodes()}

    spaces = []
    explicit_traits = 0
    contradictions = []
    for spdir in sorted((src/'spaces').glob('S*')):
        if not spdir.is_dir(): continue
        uid = spdir.name
        meta = frontmatter(spdir/'README.md') if (spdir/'README.md').exists() else {'uid':uid}
        true0, false0 = set(), set()
        raw_traits = []
        for p in sorted((spdir/'properties').glob('P*.md')):
            fm=frontmatter(p)
            pr=fm.get('property')
            val=fm.get('value')
            if pr not in prop_set or not isinstance(val,bool): continue
            explicit_traits += 1
            u=node_of[pr]
            raw_traits.append((pr,val))
            (true0 if val else false0).add(u)
        true=set()
        for u in true0: true |= descendants[u]
        false=set()
        for u in false0: false |= ancestors[u]
        overlap=true&false
        if overlap:
            contradictions.append([uid, sorted(overlap)])
            continue
        cid=meta.get('counterexamples_id')
        try: cid_i=int(cid)
        except Exception: cid_i=None
        spaces.append({
            'uid': uid, 'counterexamples_id': cid_i,
            'true': setmask(true), 'false': setmask(false),
            'explicit_count': len(raw_traits)
        })

    fam_index={}
    families=[]
    for s in spaces:
        key=(s['true'],s['false'])
        if key not in fam_index:
            fam_index[key]=len(families); families.append(key)
        s['family']=fam_index[key]

    def witness_key(s):
        return (0, s['counterexamples_id'], s['uid']) if s['counterexamples_id'] is not None else (1, 10**9, s['uid'])
    ordered=sorted(spaces,key=witness_key)
    by_pair=defaultdict(list)
    for s in ordered:
        t=s['true']; f=s['false']
        ti=[i for i in range(len(comps)) if (t>>i)&1]
        fi=[i for i in range(len(comps)) if (f>>i)&1]
        for a in ti:
            for b in fi:
                if a != b and not nx.has_path(H,a,b):
                    by_pair[(a,b)].append(s)
    witnessed=sorted(by_pair)
    outcome={(a,b): by_pair[(a,b)][0]['family'] for a,b in witnessed}

    by_source=defaultdict(list)
    for a,b in witnessed: by_source[a].append(b)
    eligible=sorted(a for a,bs in by_source.items() if len(bs)>=60)
    capped={a: sorted(by_source[a])[:60] for a in eligible}

    seeds=list(range(20260816,20260821))
    splits=[]
    for seed in seeds:
        rng=random.Random(seed)
        test_sources=sorted(rng.sample(eligible,20))
        testset=set(test_sources)
        train_sources=[a for a in eligible if a not in testset]
        train=[(a,b,outcome[(a,b)]) for a in train_sources for b in capped[a]]
        test=[(a,b,outcome[(a,b)]) for a in test_sources for b in capped[a]]
        counts=Counter(fid for _,_,fid in train)
        zero=0
        supports=[]
        for a,b,_ in test:
            sup=0
            for fid,n in counts.items():
                t,f=families[fid]
                if ((t>>a)&1) and ((f>>b)&1): sup+=n
            supports.append(sup)
            zero += (sup==0)
        splits.append({
            'seed':seed,'eligible_sources':len(eligible),
            'train_sources':len(train_sources),'test_sources':len(test_sources),
            'train_actions':len(train),'test_actions':len(test),
            'train_family_count':len(counts),
            'train_counts':sorted([[int(k),int(v)] for k,v in counts.items()]),
            'test':[list(x) for x in test],
            'zero_support_test':int(zero),
            'support_min':min(supports),'support_max':max(supports)
        })

    payload={
        'meta':{
            'upstream':'pi-base/data','commit':args.commit,
            'properties':len(properties),'scc_nodes':len(comps),
            'accepted_binary_positive_theorems':len(accepted),
            'direct_interclass_edges':len(direct_inter),
            'official_spaces':len(spaces),'explicit_traits':explicit_traits,
            'truth_signature_families':len(families),
            'witnessable_pairs':len(witnessed),
            'eligible_sources':len(eligible),
            'contradictory_spaces':len(contradictions),
            'semantics':'official atom(true)->atom(true) theorems only; SCC compression; forward truth and contrapositive false closure; unknown remains unknown',
            'witness_order':'counterexamples_id when present, then official space UID',
            'cap_per_source':60
        },
        'dag_edges':[list(e) for e in sorted(direct_inter)],
        'families':[[format(t,'x'),format(f,'x')] for t,f in families],
        'backends':{'official':{'splits':splits}}
    }
    raw=json.dumps(payload,separators=(',',':'),sort_keys=True).encode()
    enc=base64.b85encode(zlib.compress(raw,9))
    (out/'PI_BASE_COMPACT.b85').write_bytes(enc+b'\n')
    summary=dict(payload['meta'])
    summary['splits']=[{k:s[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count','zero_support_test','support_min','support_max')} for s in splits]
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'CONTRADICTIONS.json').write_text(json.dumps(contradictions,indent=2)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':
    main()
