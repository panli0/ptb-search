#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys,types
from pathlib import Path
import numpy as np

K=5; FRACTION=.75; INSTANCE_BASE_SEED=202608190000
EXPECTED={
 'pibase':{'mcts':184.8,'cpsat':188.0},
 'ring':{'mcts':91.8,'cpsat':90.6},
 'module':{'mcts':15.8,'cpsat':16.6},
}

def load_nv(path:Path, root:Path, pi:Path):
    text=path.read_text(); marker='\nres = {\n'; assert marker in text
    m=types.ModuleType('p1_expensive_nv'); sys.modules[m.__name__]=m
    old=sys.argv[:]; sys.argv=[str(path),str(root),str(pi),'/tmp/unused.json','mcts']
    try: exec(compile(text.split(marker,1)[0],str(path),'exec'),m.__dict__)
    finally: sys.argv=old
    return m

def recompute_U(ids,cov):
    u=set()
    for o in ids:u |= set(cov[o])
    return u

def subset(d, keep):
    ids,txt,cov,_,pt,tt=d; ids=sorted(keep); return ids,txt,cov,recompute_U(ids,cov),pt,tt

def mean90(nv,d,method):
    ids,txt,cov,U,pt,tt=d
    vals=[float(nv.run_seed(ids,txt,cov,U,pt,tt,method,s)['hits']['0.9']) for s in range(5)]
    return float(np.mean(vals)),vals

def random90(d):
    ids,_,cov,U,_,_=d; vals=[]
    for s in range(20):
        o=list(ids);random.Random(s).shuffle(o);seen=set();hit=None
        for i,x in enumerate(o,1):
            seen |= cov[x]
            if len(seen & U) + 1e-12 >= .9*len(U): hit=i;break
        vals.append(float(hit if hit is not None else len(ids)+1))
    return float(np.mean(vals)),vals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=['pibase','ring','module'],required=True);ap.add_argument('--dart-root',type=Path,required=True);ap.add_argument('--pi',type=Path,required=True);ap.add_argument('--nv',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);A=ap.parse_args();A.outdir.mkdir(parents=True,exist_ok=True)
    nv=load_nv(A.nv,A.dart_root,A.pi);d=nv.pi_data() if A.domain=='pibase' else nv.dart(A.domain)
    full={};fullseeds={}
    for m in ('mcts','cpsat'):full[m],fullseeds[m]=mean90(nv,d,m)
    gate={m:abs(full[m]-EXPECTED[A.domain][m])<1e-9 for m in full}
    if not all(gate.values()): raise RuntimeError('GOLD_REPRO_GATE_FAIL '+json.dumps({'observed':full,'expected':EXPECTED[A.domain],'seeds':fullseeds,'gate':gate}))
    ids=d[0];nkeep=math.ceil(FRACTION*len(ids));rows=[]
    for i in range(K):
        keep=sorted(random.Random(INSTANCE_BASE_SEED+i).sample(ids,nkeep));d2=subset(d,keep);rnd,rseeds=random90(d2);scores={'random':rnd};seedvals={'random':rseeds}
        for m in ('mcts','cpsat'):scores[m],seedvals[m]=mean90(nv,d2,m)
        rows.append({'instance':i,'seed':INSTANCE_BASE_SEED+i,'objects':len(keep),'reachable_targets':len(d2[3]),'scores':scores,'seed_values':seedvals})
        print(A.domain,'instance',i,'done',scores,flush=True)
    result={'version':1,'status':'PASS','domain':A.domain,'protocol':{'K_prime':K,'action_pool_fraction':FRACTION,'instance_base_seed':INSTANCE_BASE_SEED,'same_instance_pool_for_all_methods':True,'random_seeds':20,'planner_seeds':5,'commit_before_reveal':True,'policy_target_universe':'all official target converses; unchanged by subsampling','evaluation_universe':'union of coverage rows reachable by sampled object pool','note':'expensive methods are supplementary and do not enter primary K=20 Kendall tau'},'gold_reproduction':{'observed':full,'expected':EXPECTED[A.domain],'seed_values':fullseeds,'all_exact':True},'instances':rows,'mean_diff_vs_random':{m:float(np.mean([r['scores'][m]-r['scores']['random'] for r in rows])) for m in ('mcts','cpsat')}}
    jp=A.outdir/f'P1_{A.domain.upper()}_EXPENSIVE_K5_RESULTS.json';jp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');cp=A.outdir/f'P1_{A.domain.upper()}_EXPENSIVE_K5.csv'
    with cp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['instance','seed','objects','reachable_targets','random','mcts','cpsat']);
        for r in rows:w.writerow([r['instance'],r['seed'],r['objects'],r['reachable_targets'],r['scores']['random'],r['scores']['mcts'],r['scores']['cpsat']])
    md=A.outdir/f'P1_{A.domain.upper()}_EXPENSIVE_K5_READOUT.md';md.write_text('\n'.join([f'# P1 {A.domain} expensive K′=5','', '**Status: PASS. Full Gold MCTS and deterministic CP-SAT SAA reproduced exactly before subsampling.**','',f"- MCTS mean calls difference vs random: {result['mean_diff_vs_random']['mcts']:+.3f}",f"- CP-SAT SAA mean calls difference vs random: {result['mean_diff_vs_random']['cpsat']:+.3f}",'','Negative difference means fewer calls than random. These K′=5 results are supplementary only.'])+'\n');hp=A.outdir/f'P1_{A.domain.upper()}_EXPENSIVE_K5_SHA256SUMS.txt';hp.write_text('\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}' for p in [jp,cp,md])+'\n');print(json.dumps({'domain':A.domain,'gold':full,'mean_diff_vs_random':result['mean_diff_vs_random']},indent=2))
if __name__=='__main__':main()
