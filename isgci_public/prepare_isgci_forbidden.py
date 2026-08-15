#!/usr/bin/env python3
from __future__ import annotations

import argparse, base64, hashlib, json, random, zlib
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx

SEEDS=[20260816,20260817,20260818,20260819,20260820]


def load_smallgraphs(path):
    order=[]; codes={}; graphs={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        name,code=line.split('\t',1)
        if name in codes: continue
        order.append(name); codes[name]=code
        graphs[name]=nx.from_graph6_bytes(code.encode('ascii'))
    return order,codes,graphs


def parse_xml(path):
    classes={}; inclusions=[]
    root=ET.parse(path).getroot()
    for e in root.iter():
        tag=e.tag.split('}')[-1]
        if tag=='GraphClass':
            cid=e.attrib['id']; typ=e.attrib['type']
            name=''
            small=[]
            for c in e:
                ct=c.tag.split('}')[-1]
                if ct=='name' and c.text: name=c.text.strip()
                elif ct=='smallgraph' and c.text: small.append(c.text.strip())
            classes[cid]={'type':typ,'name':name,'smallgraphs':small}
        elif tag=='incl':
            inclusions.append((e.attrib['sub'],e.attrib['super'], 'proper' in e.attrib, e.attrib.get('confidence')))
    return classes,inclusions


def induced_contains(G,H):
    if H.number_of_nodes()>G.number_of_nodes(): return False
    # GraphMatcher.subgraph_is_isomorphic is induced; monomorphism is the non-induced variant.
    return nx.algorithms.isomorphism.GraphMatcher(G,H).subgraph_is_isomorphic()


def rect_mask(t,f,C):
    out=0; mt=t
    while mt:
        z=mt&-mt; a=z.bit_length()-1; mt-=z
        mf=f; base=a*C
        while mf:
            y=mf&-mf; b=y.bit_length()-1; mf-=y
            out |= 1<<(base+b)
    return out


def cap_by_source(actions,sources,cap=60):
    wanted=set(sources); by={s:[] for s in wanted}
    for x in actions:
        if x[0] in wanted: by[x[0]].append(x)
    out=[]
    for s in sorted(by): out.extend(sorted(by[s],key=lambda x:(x[1],x[2]))[:cap])
    return out


def make_split(actions,seed):
    cnt=Counter(a for a,_,_ in actions)
    elig=sorted(a for a,n in cnt.items() if n>=10)
    if len(elig)<10: raise RuntimeError(f'too few eligible sources: {len(elig)}')
    ntest=min(20,max(3,len(elig)//5))
    testsrc=sorted(random.Random(seed).sample(elig,ntest)); testset=set(testsrc)
    trainsrc=[s for s in elig if s not in testset]
    train=cap_by_source(actions,trainsrc); test=cap_by_source(actions,testsrc)
    tc=Counter(f for _,_,f in train)
    return {'seed':seed,'eligible_sources':len(elig),'train_sources':len(trainsrc),'test_sources':len(testsrc),
            'train_actions':len(train),'test_actions':len(test),'train_family_count':len(tc),
            'train_counts':[[f,n] for f,n in sorted(tc.items())],
            'test':[[a,b,f] for a,b,f in test]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--xml',required=True); ap.add_argument('--smallgraphs',required=True); ap.add_argument('--out',required=True); ap.add_argument('--bundle-sha',required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    sg_order,codes,graphs=load_smallgraphs(a.smallgraphs)
    classes,incls=parse_xml(a.xml)

    G=nx.DiGraph(); G.add_nodes_from(classes)
    accepted_incls=[]
    for sub,sup,proper,confidence in incls:
        if confidence=='unpublished': continue
        if sub in classes and sup in classes:
            G.add_edge(sub,sup); accepted_incls.append((sub,sup,proper))
    comps=list(nx.strongly_connected_components(G)); comps.sort(key=lambda s:min(s))
    c_of={x:i for i,s in enumerate(comps) for x in s}
    Q=nx.DiGraph(); Q.add_nodes_from(range(len(comps)))
    for u,v in G.edges():
        a0,b0=c_of[u],c_of[v]
        if a0!=b0: Q.add_edge(a0,b0)
    assert nx.is_directed_acyclic_graph(Q)

    evaluable={}
    missing=Counter()
    for cid,d in classes.items():
        if d['type']!='forbidden' or not d['smallgraphs']: continue
        miss=[x for x in d['smallgraphs'] if x not in graphs]
        if miss:
            missing.update(miss); continue
        evaluable[cid]=tuple(d['smallgraphs'])

    # Selected relation nodes are full official inclusion SCCs with at least one exact-evaluable forbidden class.
    scc_reps={}
    for qid,comp in enumerate(comps):
        ev=sorted(x for x in comp if x in evaluable)
        if ev: scc_reps[qid]=ev
    selected_q=sorted(scc_reps)
    q2n={q:i for i,q in enumerate(selected_q)}
    C=len(selected_q)

    # Cache exact induced-subgraph containment for official smallgraphs.
    needed_patterns=sorted({h for cid in evaluable for h in evaluable[cid]})
    contain={}
    for gi,gname in enumerate(sg_order):
        GG=graphs[gname]
        for hname in needed_patterns:
            HH=graphs[hname]
            contain[(gname,hname)]=induced_contains(GG,HH)

    # Verify all evaluable definitions inside an official SCC agree on every concrete smallgraph.
    scc_patterns={}
    for qid,evs in scc_reps.items():
        signatures=[]
        for cid in evs:
            pat=evaluable[cid]
            sig=tuple(not any(contain[(g,h)] for h in pat) for g in sg_order)
            signatures.append((cid,sig))
        ref=signatures[0][1]
        bad=[cid for cid,sig in signatures[1:] if sig!=ref]
        if bad: raise RuntimeError(f'equivalent forbidden definitions disagree in SCC {qid}: {bad[:5]}')
        scc_patterns[qid]=evaluable[signatures[0][0]]

    # Official relation among selected nodes = reachability through the full official quotient; then transitive reduction.
    S=nx.DiGraph(); S.add_nodes_from(range(C))
    for i,qid in enumerate(selected_q):
        for qj in nx.descendants(Q,qid):
            if qj in q2n: S.add_edge(i,q2n[qj])
    assert nx.is_directed_acyclic_graph(S)
    R=nx.transitive_reduction(S)

    objects=[]
    key2f={}; families=[]; obj_f=[]
    for gname in sg_order:
        tm=0; fm=0
        for n,qid in enumerate(selected_q):
            pat=scc_patterns[qid]
            truth=not any(contain[(gname,h)] for h in pat)
            if truth: tm|=1<<n
            else: fm|=1<<n
        # complete exact truth over selected forbidden nodes
        key=(tm,fm)
        if key not in key2f:
            key2f[key]=len(families); families.append({'t':tm,'f':fm,'rect':rect_mask(tm,fm,C),'objects':[]})
        fid=key2f[key]; families[fid]['objects'].append(gname); obj_f.append(fid); objects.append(gname)

    # Inclusion consistency: no official implication may be falsified by any exact smallgraph evaluation.
    for u,v in S.edges():
        for fid,F in enumerate(families):
            if ((F['t']>>u)&1) and ((F['f']>>v)&1):
                raise RuntimeError(f'truth contradicts official inclusion {u}->{v}, family {fid}')

    witnesses=defaultdict(list)
    for oi,fid in enumerate(obj_f):
        F=families[fid]; mt=F['t']
        while mt:
            z=mt&-mt; src=z.bit_length()-1; mt-=z
            mf=F['f']
            while mf:
                y=mf&-mf; tgt=y.bit_length()-1; mf-=y
                if src!=tgt and not nx.has_path(S,src,tgt): witnesses[(src,tgt)].append(oi)

    payload={'meta':{
        'source':'ISGCI official data.zip','bundle_sha256':a.bundle_sha,'xml_date':'2026-05-10',
        'graphclasses':len(classes),'direct_inclusions_accepted':len(accepted_incls),'full_sccs':len(comps),
        'official_smallgraphs':len(sg_order),'forbidden_classes_exact_evaluable':len(evaluable),
        'selected_scc_nodes':C,'selected_transitive_reduction_edges':R.number_of_edges(),
        'truth_signature_families':len(families),'witnessable_pairs':len(witnesses),
        'missing_forbidden_smallgraph_names':len(missing),
        'semantics':'official published inclusion edges; full SCC/closure; selected nodes have exact forbidden-smallgraph definitions; official graph6 smallgraphs evaluated by induced-subgraph isomorphism; no unknown->false'},
        'dag_edges':[[u,v] for u,v in sorted(R.edges())],
        'families':[[format(F['t'],'x'),format(F['f'],'x')] for F in families],
        'object_names':objects,
        'backends':{}}
    for backend in ('first','last'):
        actions=[]
        for (src,tgt),ois in sorted(witnesses.items()):
            oi=min(ois) if backend=='first' else max(ois)
            actions.append((src,tgt,obj_f[oi]))
        payload['backends'][backend]={'actions_total':len(actions),'splits':[make_split(actions,s) for s in SEEDS],
            'global_family_ids_b85':base64.b85encode(zlib.compress(bytes(f for _,_,f in actions),9)).decode('ascii')}
    raw=json.dumps(payload,separators=(',',':'),sort_keys=True).encode()
    packed=base64.b85encode(zlib.compress(raw,9)).decode('ascii')
    (out/'ISGCI_FORBIDDEN_COMPACT.b85').write_text(packed+'\n',encoding='ascii')
    (out/'SUMMARY.json').write_text(json.dumps(payload['meta']|{
        'first_actions_total':payload['backends']['first']['actions_total'],'last_actions_total':payload['backends']['last']['actions_total'],
        'compact_json_bytes':len(raw),'compact_b85_bytes':len(packed),
        'first_splits':[{k:s[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count')} for s in payload['backends']['first']['splits']],
        'last_splits':[{k:s[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count')} for s in payload['backends']['last']['splits']]},indent=2,sort_keys=True)+'\n')
    chunks=out/'chunks'; chunks.mkdir(exist_ok=True)
    for p in chunks.glob('part_*.txt'): p.unlink()
    for i in range(0,len(packed),5000): (chunks/f'part_{i//5000:02d}.txt').write_text(packed[i:i+5000]+'\n')
    print((out/'SUMMARY.json').read_text())

if __name__=='__main__': main()
