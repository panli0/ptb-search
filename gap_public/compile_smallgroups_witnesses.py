#!/usr/bin/env python3
import csv,json,random
from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
base=Path('gap_public/smallgroups_out')
GJ=json.loads((base/'ABSTRACT_GRAPH.json').read_text())
nodes=GJ['nodes']; idx={n:i for i,n in enumerate(nodes)}; C=len(nodes)
D=nx.DiGraph();D.add_nodes_from(range(C))
for e in GJ['edges']:D.add_edge(idx[e['source']],idx[e['target']])
comps=list(nx.strongly_connected_components(D));comps.sort(key=lambda s:min(s));co={u:i for i,s in enumerate(comps) for u in s}
Q=nx.DiGraph();Q.add_nodes_from(range(len(comps)))
for u,v in D.edges():
    if co[u]!=co[v]:Q.add_edge(co[u],co[v])
assert nx.is_directed_acyclic_graph(Q)
R=nx.transitive_reduction(Q)
with open(base/'TRUTH.csv') as f:rows=list(csv.DictReader(f))
if len(rows)<10:raise SystemExit(f'truth table unexpectedly empty: {len(rows)} rows')
# Hard consistency check against every author edge.
bad=[]
for r in rows:
    vals=[int(r[n]) for n in nodes]
    for u,v in D.edges():
        if vals[u] and not vals[v]:bad.append((r['order'],r['id'],nodes[u],nodes[v]))
if bad:raise SystemExit('AUTHOR EDGE CONTRADICTIONS '+repr(bad[:20]))
for comp in comps:
    if len(comp)>1:
        for r in rows:
            vv={int(r[nodes[u]]) for u in comp}
            if len(vv)>1:raise SystemExit('SCC TRUTH DISAGREEMENT '+repr((sorted(comp),r['order'],r['id'])))
fam_map={};fams=[];objfam=[]
for r in rows:
    t=f=0
    for q,comp in enumerate(comps):
        v=int(r[nodes[min(comp)]])
        if v:t|=1<<q
        else:f|=1<<q
    key=(t,f)
    if key not in fam_map:fam_map[key]=len(fams);fams.append({'true':t,'false':f,'objects':[]})
    fi=fam_map[key];fams[fi]['objects'].append([int(r['order']),int(r['id'])]);objfam.append(fi)
witnesses=defaultdict(list)
for oi,fi in enumerate(objfam):
    F=fams[fi];t=F['true'];f=F['false']
    while t:
        z=t&-t;u=z.bit_length()-1;t-=z;m=f
        while m:
            y=m&-m;v=y.bit_length()-1;m-=y
            if u!=v and not nx.has_path(Q,u,v):witnesses[(u,v)].append(oi)
def cap(A,S):
    d=defaultdict(list)
    for x in A:
        if x[0] in S:d[x[0]].append(x)
    return [x for s in sorted(S) for x in sorted(d[s],key=lambda q:(q[1],q[2]))[:60]]
def split(A,seed):
    cnt=Counter(x[0] for x in A);elig=sorted(k for k,v in cnt.items() if v>=3)
    if len(elig)<3:raise SystemExit(f'too few eligible sources: {len(elig)}')
    nt=min(20,max(2,len(elig)//5));te=set(random.Random(seed).sample(elig,nt));tr=set(elig)-te
    train=cap(A,tr);test=cap(A,te);tc=Counter(x[2] for x in train)
    return {'seed':seed,'eligible_sources':len(elig),'train_sources':len(tr),'test_sources':len(te),'train_actions':len(train),'test_actions':len(test),'train_family_count':len(tc),'train_counts':sorted([list(x) for x in tc.items()]),'test':[list(x) for x in test]}
payload={'meta':{'gap_source_commit':GJ['gap_commit'],'runtime_gap_version':'4.12.1','smallgroups_orders':'1..96','raw_property_nodes':C,'scc_nodes':len(comps),'raw_author_edges':D.number_of_edges(),'transitive_reduction_edges':R.number_of_edges(),'objects':len(rows),'truth_families':len(fams),'witnessable_pairs':len(witnesses),'semantics':'GAP author-declared InstallTrueMethod implications under finite-group ambient assumptions; exact SmallGroups property values; no edges inferred from object truth'},'scc_members':[[nodes[u] for u in sorted(c)] for c in comps],'dag_edges':[list(e) for e in sorted(R.edges())],'families':[[format(F['true'],'x'),format(F['false'],'x')] for F in fams],'backends':{}}
for b in ('first','last'):
    A=[]
    for (u,v),ois in sorted(witnesses.items()):
        oi=ois[0] if b=='first' else ois[-1];A.append((u,v,objfam[oi]))
    payload['backends'][b]={'actions_total':len(A),'splits':[split(A,s) for s in range(20260816,20260821)]}
(base/'COMPACT.json').write_text(json.dumps(payload,separators=(',',':'),sort_keys=True)+'\n')
summary=payload['meta']|{'first_actions_total':payload['backends']['first']['actions_total'],'last_actions_total':payload['backends']['last']['actions_total'],'first_splits':[{k:s[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count')} for s in payload['backends']['first']['splits']],'last_splits':[{k:s[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count')} for s in payload['backends']['last']['splits']]}
(base/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
