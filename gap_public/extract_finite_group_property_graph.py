#!/usr/bin/env python3
from pathlib import Path
import json,re,subprocess
ROOT=Path('_upstream/gap')
commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
# Properties explicitly declared as GAP properties in group-related declarations.
declared=set()
for p in list((ROOT/'lib').glob('grp*.gd'))+list((ROOT/'lib').glob('magma.gd')):
    s=p.read_text(errors='replace')
    declared.update(re.findall(r'DeclareProperty\s*\(\s*["\']([A-Za-z0-9_]+)["\']',s))
# Group-applicable mathematical properties from ambient magma layer.
declared.update({'IsCommutative','IsAssociative','IsTrivial','IsNonTrivial','IsFinite'})
exclude_prefix=('Can','Has')
exclude_exact={
    'IsGroup','IsPcGroup','IsPermGroup','IsFinitelyGeneratedGroup','IsFinitelyGeneratedMagma',
    'IsMagma','IsMagmaWithOne','IsMagmaWithInverses','IsAssociative','IsFinite',
    'IsFiniteOrderElementCollection','IsSubsetLocallyFiniteGroup','IsInternalRep',
}
def scientific(n):
    return n in declared and n not in exclude_exact and not n.startswith(exclude_prefix)
calls=[]
for p in list((ROOT/'lib').glob('*.gd'))+list((ROOT/'lib').glob('*.gi')):
    s=p.read_text(errors='replace')
    for m in re.finditer(r'InstallTrueMethod\s*\(',s):
        start=m.start(); i=m.end(); depth=1
        while i<len(s) and depth:
            if s[i]=='(': depth+=1
            elif s[i]==')': depth-=1
            i+=1
        call=s[start:i]
        inner=call[call.find('(')+1:-1]
        depth=0; cut=None
        for j,ch in enumerate(inner):
            if ch=='(': depth+=1
            elif ch==')': depth-=1
            elif ch==',' and depth==0:
                cut=j;break
        if cut is None: continue
        lhs=inner[:cut].strip(); rhs=inner[cut+1:].strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',lhs): continue
        toks=re.findall(r'\bIs[A-Za-z0-9_]+\b',rhs)
        substantive=[t for t in toks if t not in {'IsGroup','IsFinite'}]
        if len(substantive)!=1: continue
        src=substantive[0]; dst=lhs
        if src==dst or not scientific(src) or not scientific(dst): continue
        line=s.count('\n',0,start)+1
        calls.append({'source':src,'target':dst,'file':str(p.relative_to(ROOT)),'line':line,
                      'call':re.sub(r'\s+',' ',call)})
by={}
for c in calls: by.setdefault((c['source'],c['target']),[]).append(c)
edges=[{'source':u,'target':v,'proofs':proofs} for (u,v),proofs in sorted(by.items())]
nodes=sorted({x for e in edges for x in (e['source'],e['target'])})
out={'gap_commit':commit,'domain':'finite groups','nodes':nodes,'edges':edges,'node_count':len(nodes),'edge_count':len(edges)}
Path('gap_public/FINITE_GROUP_PROPERTY_GRAPH.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'gap_commit':commit,'node_count':len(nodes),'edge_count':len(edges),'nodes':nodes,
                  'edges':[[e['source'],e['target']] for e in edges]},indent=2))
