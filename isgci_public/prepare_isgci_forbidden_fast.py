#!/usr/bin/env python3
import argparse,base64,itertools,json,random,zlib,xml.etree.ElementTree as ET
from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
SEEDS=range(20260816,20260821)
def loadsg(p):
 o=[]; g={}
 for s in Path(p).read_text().splitlines():
  if not s: continue
  n,c=s.split('\t',1)
  if n not in g:o.append(n);g[n]=nx.from_graph6_bytes(c.encode())
 return o,g
def parse(p):
 C={};E=[]
 for _,e in ET.iterparse(p,events=('end',)):
  t=e.tag.split('}')[-1]
  if t=='GraphClass':
   sm=[x.text.strip() for x in e if x.tag.split('}')[-1]=='smallgraph' and x.text]
   C[e.attrib['id']]=(e.attrib['type'],sm)
  elif t=='incl':E.append((e.attrib['sub'],e.attrib['super'],e.attrib.get('confidence')))
  e.clear()
 return C,E
def inv(g):return(g.number_of_nodes(),g.number_of_edges(),tuple(sorted(dict(g.degree()).values())))
def exact_contains(order,graphs,patterns):
 groups=defaultdict(list)
 for h in patterns:groups[inv(graphs[h])].append(h)
 sizes=sorted({graphs[h].number_of_nodes() for h in patterns})
 out={}
 for i,nm in enumerate(order):
  G=graphs[nm];found=set();nodes=list(G)
  for k in sizes:
   if k>len(nodes):break
   for vs in itertools.combinations(nodes,k):
    H=G.subgraph(vs);cand=groups.get(inv(H),())
    for h in cand:
     if h not in found and nx.is_isomorphic(H,graphs[h]):found.add(h)
  out[nm]=found
  if (i+1)%25==0:print('smallgraphs',i+1,'/',len(order),flush=True)
 return out
def rect(t,f,C):
 r=0
 while t:
  z=t&-t;a=z.bit_length()-1;t-=z;m=f
  while m:y=m&-m;b=y.bit_length()-1;m-=y;r|=1<<(a*C+b)
 return r
def cap(A,S):
 d=defaultdict(list)
 for x in A:
  if x[0] in S:d[x[0]].append(x)
 return [x for s in sorted(S) for x in sorted(d[s],key=lambda q:(q[1],q[2]))[:60]]
def split(A,seed):
 ec=Counter(x[0] for x in A);elig=sorted(k for k,v in ec.items() if v>=10)
 if len(elig)<10:raise RuntimeError('too few sources '+str(len(elig)))
 nt=min(20,max(3,len(elig)//5));teS=set(random.Random(seed).sample(elig,nt));trS=set(elig)-teS
 tr=cap(A,trS);te=cap(A,teS);cnt=Counter(x[2] for x in tr)
 return {'seed':seed,'eligible_sources':len(elig),'train_sources':len(trS),'test_sources':len(teS),'train_actions':len(tr),'test_actions':len(te),'train_family_count':len(cnt),'train_counts':sorted([list(x) for x in cnt.items()]),'test':[list(x) for x in te]}
def main():
 a=argparse.ArgumentParser();a.add_argument('--xml');a.add_argument('--smallgraphs');a.add_argument('--out');a.add_argument('--bundle-sha');x=a.parse_args();O=Path(x.out);O.mkdir(parents=True,exist_ok=True)
 names,graphs=loadsg(x.smallgraphs);C,E=parse(x.xml)
 G=nx.DiGraph();G.add_nodes_from(C);acc=[]
 for u,v,conf in E:
  if conf!='unpublished' and u in C and v in C:G.add_edge(u,v);acc.append((u,v))
 scc=list(nx.strongly_connected_components(G));scc.sort(key=min);co={u:i for i,s in enumerate(scc) for u in s};Q=nx.DiGraph();Q.add_nodes_from(range(len(scc)))
 for u,v in G.edges():
  if co[u]!=co[v]:Q.add_edge(co[u],co[v])
 ev={cid:tuple(sm) for cid,(typ,sm) in C.items() if typ=='forbidden' and sm and all(h in graphs for h in sm)}
 missing=sum(1 for cid,(typ,sm) in C.items() if typ=='forbidden' and sm and not all(h in graphs for h in sm))
 reps={q:sorted(u for u in s if u in ev) for q,s in enumerate(scc)};reps={q:v for q,v in reps.items() if v};sel=sorted(reps);q2n={q:i for i,q in enumerate(sel)};N=len(sel)
 pats=sorted({h for v in ev.values() for h in v});contains=exact_contains(names,graphs,pats)
 sigs={}
 for q,ids in reps.items():
  vals=[]
  for cid in ids:vals.append((cid,tuple(not any(h in contains[g] for h in ev[cid]) for g in names)))
  if any(s!=vals[0][1] for _,s in vals[1:]):raise RuntimeError('SCC definition disagreement '+str(q))
  sigs[q]=vals[0][1]
 S=nx.DiGraph();S.add_nodes_from(range(N))
 for q in sel:
  for z in nx.descendants(Q,q):
   if z in q2n:S.add_edge(q2n[q],q2n[z])
 R=nx.transitive_reduction(S)
 fam=[];k2f={};of=[]
 for gi,g in enumerate(names):
  t=f=0
  for n,q in enumerate(sel):
   if sigs[q][gi]:t|=1<<n
   else:f|=1<<n
  k=(t,f)
  if k not in k2f:k2f[k]=len(fam);fam.append((t,f,rect(t,f,N)))
  of.append(k2f[k])
 for u,v in S.edges():
  if any(((t>>u)&1) and ((f>>v)&1) for t,f,_ in fam):raise RuntimeError('official inclusion contradicted')
 W=defaultdict(list)
 for oi,fi in enumerate(of):
  t,f,_=fam[fi]
  while t:
   z=t&-t;u=z.bit_length()-1;t-=z;m=f
   while m:y=m&-m;v=y.bit_length()-1;m-=y
   if u!=v and not nx.has_path(S,u,v):W[(u,v)].append(oi)
 P={'meta':{'source':'ISGCI official data.zip','bundle_sha256':x.bundle_sha,'graphclasses':len(C),'direct_inclusions_accepted':len(acc),'full_sccs':len(scc),'official_smallgraphs':len(names),'forbidden_classes_exact_evaluable':len(ev),'forbidden_classes_missing_pattern':missing,'selected_scc_nodes':N,'selected_transitive_reduction_edges':R.number_of_edges(),'truth_signature_families':len(fam),'witnessable_pairs':len(W),'semantics':'official published inclusion edges; SCC/closure; exact official forbidden-smallgraph definitions evaluated on official graph6 by induced-subgraph isomorphism; no unknown->false'},'dag_edges':sorted([list(e) for e in R.edges()]),'families':[[format(t,'x'),format(f,'x')] for t,f,_ in fam],'backends':{}}
 for b in ('first','last'):
  A=[]
  for (u,v),ois in sorted(W.items()):
   oi=min(ois) if b=='first' else max(ois);A.append((u,v,of[oi]))
  P['backends'][b]={'actions_total':len(A),'splits':[split(A,s) for s in SEEDS],'global_family_ids_b85':base64.b85encode(zlib.compress(bytes(q[2] for q in A),9)).decode()}
 raw=json.dumps(P,separators=(',',':'),sort_keys=True).encode();pack=base64.b85encode(zlib.compress(raw,9)).decode();(O/'ISGCI_FORBIDDEN_COMPACT.b85').write_text(pack+'\n')
 sm=P['meta']|{'first_actions_total':P['backends']['first']['actions_total'],'last_actions_total':P['backends']['last']['actions_total'],'compact_b85_bytes':len(pack),'first_splits':[{k:s[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count')} for s in P['backends']['first']['splits']],'last_splits':[{k:s[k] for k in ('seed','eligible_sources','train_sources','test_sources','train_actions','test_actions','train_family_count')} for s in P['backends']['last']['splits']]};(O/'SUMMARY.json').write_text(json.dumps(sm,indent=2,sort_keys=True)+'\n')
 ch=O/'chunks';ch.mkdir(exist_ok=True)
 for p in ch.glob('part_*.txt'):p.unlink()
 for i in range(0,len(pack),5000):(ch/f'part_{i//5000:02d}.txt').write_text(pack[i:i+5000]+'\n')
 print((O/'SUMMARY.json').read_text())
if __name__=='__main__':main()
