#!/usr/bin/env python3
from pathlib import Path
import argparse,json,re,subprocess

def true_calls(root, files):
    out=[]
    for p in files:
        s=p.read_text(errors='replace')
        for m in re.finditer(r'InstallTrueMethod\s*\(',s):
            start=m.start();i=m.end();depth=1
            while i<len(s) and depth:
                if s[i]=='(':depth+=1
                elif s[i]==')':depth-=1
                i+=1
            call=s[start:i]
            inner=call[call.find('(')+1:-1]
            d=0;cut=None
            for j,ch in enumerate(inner):
                if ch=='(':d+=1
                elif ch==')':d-=1
                elif ch==',' and d==0:cut=j;break
            if cut is None:continue
            lhs=inner[:cut].strip();rhs=inner[cut+1:].strip()
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',lhs):continue
            toks=re.findall(r'\bIs[A-Za-z0-9_]+\b',rhs)
            substantive=[t for t in toks if t not in {'IsSemigroup','IsFinite','IsSmallSemigroup'}]
            if len(substantive)!=1:continue
            src=substantive[0]
            if src==lhs:continue
            out.append({'source':src,'target':lhs,'file':str(p.relative_to(root)),'line':s.count('\n',0,start)+1,'call':re.sub(r'\s+',' ',call)})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gap',required=True);ap.add_argument('--semigroups',required=True);ap.add_argument('--whitelist',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();gap=Path(a.gap);semi=Path(a.semigroups);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    white={x.strip() for x in Path(a.whitelist).read_text().splitlines() if x.strip()}
    gfiles=list((gap/'lib').glob('*.gd'))+list((gap/'lib').glob('*.gi'))
    sfiles=list((semi/'gap').rglob('*.gd'))+list((semi/'gap').rglob('*.gi'))
    raw=[]
    for x in true_calls(gap,gfiles):x['source_repo']='gap';raw.append(x)
    for x in true_calls(semi,sfiles):x['source_repo']='semigroups';raw.append(x)
    kept=[x for x in raw if x['source'] in white and x['target'] in white]
    by={}
    for x in kept:by.setdefault((x['source'],x['target']),[]).append(x)
    edges=[{'source':u,'target':v,'proofs':ps} for (u,v),ps in sorted(by.items())]
    nodes=sorted({z for e in edges for z in (e['source'],e['target'])})
    obj={'gap_commit':subprocess.check_output(['git','-C',str(gap),'rev-parse','HEAD'],text=True).strip(),
         'semigroups_commit':subprocess.check_output(['git','-C',str(semi),'rev-parse','HEAD'],text=True).strip(),
         'smallsemi_evaluable_whitelist_count':len(white),'node_count':len(nodes),'edge_count':len(edges),'nodes':nodes,'edges':edges,
         'semantics':'author-declared InstallTrueMethod binary implications; finite-semigroup ambient assumptions IsSemigroup/IsFinite/IsSmallSemigroup removed; nodes restricted to Smallsemi-evaluable properties'}
    out.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'node_count':len(nodes),'edge_count':len(edges),'nodes':nodes,'edges':[[e['source'],e['target']] for e in edges]},indent=2))
if __name__=='__main__':main()
