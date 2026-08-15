import csv,json,random
from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
B=Path('semigroup_public/out');J=json.loads((B/'PROPERTY_GRAPH.json').read_text());ns=J['nodes']; ix={n:i for i,n in enumerate(ns)}
D=nx.DiGraph();D.add_nodes_from(range(len(ns)));D.add_edges_from((ix[e['source']],ix[e['target']]) for e in J['edges'])
cs=list(nx.strongly_connected_components(D));cs.sort(key=min);co={u:i for i,c in enumerate(cs) for u in c};Q=nx.DiGraph();Q.add_nodes_from(range(len(cs)))
for u,v in D.edges():
 if co[u]!=co[v]:Q.add_edge(co[u],co[v])
R=nx.transitive_reduction(Q); rows=list(csv.DictReader((B/'TRUTH.csv').open())); assert len(rows)>1000
for r in rows:
 vals=[int(r[n]) for n in ns]
 for u,v in D.edges():
  if vals[u] and not vals[v]:raise SystemExit(f'implication contradiction {r["size"]},{r["id"]}: {ns[u]}->{ns[v]}')
for c in cs:
 if len(c)>1:
  for r in rows:
   if len({int(r[ns[u]]) for u in c})>1:raise SystemExit('SCC truth mismatch')
fm={};fs=[];of=[]
for r in rows:
 t=f=0
 for q,c in enumerate(cs):
  if int(r[ns[min(c)]]):t|=1<<q
  else:f|=1<<q
 k=(t,f)
 if k not in fm:fm[k]=len(fs);fs.append(k)
 of.append(fm[k])
W=defaultdict(list)
for oi,fi in enumerate(of):
 t,f=fs[fi]
 for u in range(len(cs)):
  if (t>>u)&1:
   for v in range(len(cs)):
    if u!=v and ((f>>v)&1) and not nx.has_path(Q,u,v):W[(u,v)].append(oi)
cnt=Counter(u for u,v in W); std=[u for u,n in cnt.items() if n>=10]; thr=10 if len(std)>=10 else 3; elig=[u for u,n in cnt.items() if n>=thr]; assert len(elig)>=5

def cap(A,S):
 d=defaultdict(list)
 for x in A:
  if x[0] in S:d[x[0]].append(x)
 return [x for s in sorted(S) for x in sorted(d[s])[:60]]
def spl(A,seed):
 E=sorted(u for u,n in Counter(x[0] for x in A).items() if n>=thr); nt=min(20,max(3,len(E)//5)) if len(E)>=10 else min(2,max(1,len(E)//5)); te=set(random.Random(seed).sample(E,nt));tr=set(E)-te;T=cap(A,tr);V=cap(A,te);tc=Counter(x[2] for x in T)
 return {'seed':seed,'eligible_sources':len(E),'train_sources':len(tr),'test_sources':len(te),'train_actions':len(T),'test_actions':len(V),'train_family_count':len(tc),'train_counts':sorted([list(x) for x in tc.items()]),'test':[list(x) for x in V]}
meta={'gap_commit':J['gap_commit'],'semigroups_commit':J['semigroups_commit'],'smallsemi_commit':(B/'SMALLSEMI_COMMIT.txt').read_text().strip(),'gap_runtime_version':(B/'GAP_VERSION.txt').read_text().strip(),'smallsemi_version':(B/'SMALLSEMI_VERSION.txt').read_text().strip(),'objects':len(rows),'orders':'1..6','raw_property_nodes':len(ns),'scc_nodes':len(cs),'raw_author_edges':D.number_of_edges(),'transitive_reduction_edges':R.number_of_edges(),'truth_families':len(fs),'witnessable_pairs':len(W),'standard_eligible_sources_ge10':len(std),'eligibility_actions_per_source':thr,'fallback_protocol':thr!=10,'semantics':J['semantics']+'; exact Smallsemi property truth; no edge inferred from objects'}
P={'meta':meta,'scc_members':[[ns[u] for u in sorted(c)] for c in cs],'dag_edges':[list(e) for e in sorted(R.edges())],'families':[[format(t,'x'),format(f,'x')] for t,f in fs],'source_action_counts':sorted([[u,n] for u,n in cnt.items()]),'backends':{}}
for b in ('first','last'):
 A=[]
 for (u,v),oo in sorted(W.items()):
  oi=oo[0] if b=='first' else oo[-1];A.append((u,v,of[oi]))
 P['backends'][b]={'actions_total':len(A),'splits':[spl(A,s) for s in range(20260816,20260821)]}
(B/'COMPACT.json').write_text(json.dumps(P,separators=(',',':'))+'\n'); S=meta|{'source_action_counts':P['source_action_counts'],'first_splits':[{k:x[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count')} for x in P['backends']['first']['splits']]};(B/'SUMMARY.json').write_text(json.dumps(S,indent=2)+'\n');print(json.dumps(S,indent=2))
