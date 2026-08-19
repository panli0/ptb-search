#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, random, sys, types
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

K=20
FRACTION=0.75
INSTANCE_BASE_SEED=202608190000
BOOT_SEED=202608190777
BOOT_REPS=10000
CHEAP_METHODS=[
    'random','target_mean','target_ucb','target_eps',
    'ridge','epsridge','linucb','knn','psc','wpsc','bestfirst'
]
EXPECTED={
 'ring': {'random':87.25,'target_mean':88.7,'target_ucb':93.5,'target_eps':85.2,
          'ridge':91.0,'epsridge':83.4,'linucb':90.2,'knn':89.0,
          'psc':92.0,'wpsc':93.4,'bestfirst':94.6},
 'module': {'random':16.75,'target_mean':17.5,'target_ucb':15.9,'target_eps':16.3,
          'ridge':18.2,'epsridge':15.4,'linucb':18.0,'knn':16.4,
          'psc':16.6,'wpsc':17.0,'bestfirst':17.0},
}

def load_ou(path:Path, root:Path):
    old=sys.argv[:]
    sys.argv=[str(path),str(root),'/tmp/unused-pi.json','/tmp/unused-maspob','/tmp/unused.json','scalar','all']
    try:
        spec=importlib.util.spec_from_file_location('p1_ou',path)
        m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m)
        return m
    finally: sys.argv=old

def load_nv_defs(path:Path, root:Path):
    # Exact Gold source definitions, stopping before its top-level full-domain execution block.
    text=path.read_text()
    marker='\nres = {\n'
    if marker not in text: raise RuntimeError('native runner execution marker missing')
    lib=text.split(marker,1)[0]
    m=types.ModuleType('p1_nv')
    old=sys.argv[:]
    sys.argv=[str(path),str(root),'/tmp/unused-pi.json','/tmp/unused.json','psc']
    try: exec(compile(lib,str(path),'exec'),m.__dict__)
    finally: sys.argv=old
    return m

def recompute_U(ids,cov):
    u=set()
    for o in ids: u |= set(cov[o])
    return u

def subset_ou(data, keep):
    ids,txt,cov,_,policy_targets,tn,nodes,edges=data
    ids=sorted(keep)
    return (ids,txt,cov,recompute_U(ids,cov),policy_targets,tn,nodes,edges)

def subset_nv(data, keep):
    ids,txt,cov,_,policy_targets,target_txt=data
    ids=sorted(keep)
    return (ids,txt,cov,recompute_U(ids,cov),policy_targets,target_txt)

def hit90_ou(ou, order, data):
    h,_=ou.evaluate(order,data[2],data[3]); return float(h['0.9'])

def score_random(ou,data):
    ids=data[0]; vals=[]
    for s in range(20):
        order=list(ids); random.Random(s).shuffle(order); vals.append(hit90_ou(ou,order,data))
    return float(np.mean(vals))

def score_target(ou,data,mode):
    ids,txt,cov,eval_U,policy_targets,*_=data; env=ou.StrictEnv(ids,cov)
    return float(np.mean([hit90_ou(ou,ou.target_policy(ids,txt,env,policy_targets,mode,s),data) for s in range(10)]))

def score_scalar(ou,data,mode):
    ids,txt,cov,eval_U,policy_targets,*_=data; env=ou.StrictEnv(ids,cov)
    return float(np.mean([hit90_ou(ou,ou.scalar_policy(ids,txt,env,len(policy_targets),mode,s),data) for s in range(5)]))

def score_nv(nv,data,method):
    ids,txt,cov,eval_U,policy_targets,target_txt=data
    vals=[]
    for s in range(5):
        z=nv.run_seed(ids,txt,cov,eval_U,policy_targets,target_txt,method,s)
        vals.append(float(z['hits']['0.9']))
    return float(np.mean(vals))

def all_scores(ou,nv,odu,ndu):
    out={'random':score_random(ou,odu)}
    for m in ('target_mean','target_ucb','target_eps'): out[m]=score_target(ou,odu,m)
    for m in ('ridge','epsridge','linucb','knn'): out[m]=score_scalar(ou,odu,m)
    for m in ('psc','wpsc','bestfirst'): out[m]=score_nv(nv,ndu,m)
    return out

def rank_tau(a,b):
    x=[a[m] for m in CHEAP_METHODS]; y=[b[m] for m in CHEAP_METHODS]
    t=kendalltau(x,y,variant='b').statistic
    return float(t) if np.isfinite(t) else None

def qci(xs):
    a=np.asarray(xs,float); return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]

def bootstrap_diff(rows, method):
    rng=np.random.default_rng(BOOT_SEED + CHEAP_METHODS.index(method))
    d=np.asarray([r['scores'][method]-r['scores']['random'] for r in rows],float)
    means=np.empty(BOOT_REPS)
    for i in range(BOOT_REPS): means[i]=d[rng.integers(0,len(d),len(d))].mean()
    return {'mean_calls_diff_vs_random':float(d.mean()),'bootstrap95':qci(means),
            'instances_better_than_random':int(np.sum(d<0)),'instances_equal_random':int(np.sum(d==0)),
            'n':len(d)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--domain',choices=['ring','module'],required=True)
    ap.add_argument('--dart-root',type=Path,required=True); ap.add_argument('--ou',type=Path,required=True)
    ap.add_argument('--nv',type=Path,required=True); ap.add_argument('--outdir',type=Path,required=True); A=ap.parse_args()
    A.outdir.mkdir(parents=True,exist_ok=True)
    ou=load_ou(A.ou,A.dart_root); nv=load_nv_defs(A.nv,A.dart_root)
    od=ou.dart(A.domain); nd=nv.dart(A.domain)
    assert od[0]==nd[0], 'object-id mismatch between Gold runners'
    assert od[2]==nd[2], 'coverage-row mismatch between Gold runners'
    assert od[3]==nd[3], 'evaluation universe mismatch between Gold runners'
    full=all_scores(ou,nv,od,nd)
    gate={m:abs(full[m]-EXPECTED[A.domain][m])<1e-9 for m in CHEAP_METHODS}
    if not all(gate.values()):
        raise RuntimeError('GOLD_REPRO_GATE_FAIL '+json.dumps({'observed':full,'expected':EXPECTED[A.domain],'gate':gate}))
    ids=od[0]; nkeep=math.ceil(FRACTION*len(ids)); rows=[]
    for i in range(K):
        seed=INSTANCE_BASE_SEED+i
        keep=sorted(random.Random(seed).sample(ids,nkeep))
        o2=subset_ou(od,keep); n2=subset_nv(nd,keep)
        scores=all_scores(ou,nv,o2,n2)
        rows.append({'instance':i,'seed':seed,'objects':len(keep),'reachable_targets':len(o2[3]),
                     'scores':scores,'tau_vs_full':rank_tau(scores,full)})
        print(A.domain,'instance',i,'done',scores,flush=True)
    pair_tau=[]
    for i in range(K):
        for j in range(i+1,K):
            t=rank_tau(rows[i]['scores'],rows[j]['scores'])
            if t is not None: pair_tau.append(t)
    tau_full=[r['tau_vs_full'] for r in rows if r['tau_vs_full'] is not None]
    stats={m:bootstrap_diff(rows,m) for m in CHEAP_METHODS if m!='random'}
    primary=float(np.median(pair_tau))
    if primary<0.4: verdict='TAU_LT_0_4_RANKING_UNSTABLE'
    elif primary>=0.6: verdict='TAU_GE_0_6_RANKING_STABLE'
    else: verdict='TAU_0_4_TO_0_6_AMBIGUOUS'
    result={'version':1,'status':'PASS','domain':A.domain,'protocol':{
        'K':K,'action_pool_fraction':FRACTION,'subsample_mode':'uniform fixed-seed object/action-pool subsampling without replacement',
        'policy_target_universe':'all official unary-rule converse targets; unchanged by subsampling',
        'evaluation_universe':'union of revealed coverage rows reachable by the sampled object pool',
        'commit_before_reveal':True,'same_instance_pool_for_all_methods':True,'instance_base_seed':INSTANCE_BASE_SEED,
        'primary_tau':'median pairwise Kendall tau-b between all K instance method-ranking vectors over the 11 cheap methods including random',
        'bootstrap_reps':BOOT_REPS,'bootstrap_seed':BOOT_SEED},
      'gold_reproduction':{'observed':full,'expected':EXPECTED[A.domain],'all_exact':True},
      'instances':rows,'ranking_stability':{'pair_count':len(pair_tau),'pairwise_tau_median':primary,
        'pairwise_tau_mean':float(np.mean(pair_tau)),'pairwise_tau_p05_p95':[float(np.quantile(pair_tau,.05)),float(np.quantile(pair_tau,.95))],
        'tau_vs_full_median':float(np.median(tau_full)),'tau_vs_full_mean':float(np.mean(tau_full)),'verdict':verdict},
      'paired_bootstrap_vs_random':stats}
    jp=A.outdir/f'P1_{A.domain.upper()}_CHEAP_RESULTS.json'; jp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    cp=A.outdir/f'P1_{A.domain.upper()}_INSTANCES.csv'
    with cp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['instance','seed','objects','reachable_targets','tau_vs_full']+CHEAP_METHODS)
        for r in rows: w.writerow([r['instance'],r['seed'],r['objects'],r['reachable_targets'],r['tau_vs_full']]+[r['scores'][m] for m in CHEAP_METHODS])
    md=A.outdir/f'P1_{A.domain.upper()}_CHEAP_READOUT.md'
    lines=[f'# P1 {A.domain} — 75% action-pool × 20 stability audit','',
      '**Status: PASS. Gold reproduction gate passed exactly before subsampling.**','',
      f'- Primary median pairwise Kendall tau-b: **{primary:.3f}**',
      f'- Mean pairwise tau-b: {np.mean(pair_tau):.3f}',f'- Median tau vs full Gold ranking: {np.median(tau_full):.3f}',f'- Verdict: **{verdict}**','',
      '| Method | mean calls diff vs random | paired bootstrap 95% CI | better instances / 20 |','|---|---:|---:|---:|']
    for m in CHEAP_METHODS:
        if m=='random': continue
        z=stats[m]; lines.append(f"| {m} | {z['mean_calls_diff_vs_random']:.3f} | [{z['bootstrap95'][0]:.3f}, {z['bootstrap95'][1]:.3f}] | {z['instances_better_than_random']} |")
    lines += ['','Negative call difference means fewer calls than random. P1 primary tau was fixed in code before results were observed.']
    md.write_text('\n'.join(lines)+'\n')
    hp=A.outdir/f'P1_{A.domain.upper()}_SHA256SUMS.txt'
    hp.write_text('\n'.join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in [jp,cp,md])+'\n')
    print(json.dumps({'domain':A.domain,'primary_tau':primary,'verdict':verdict,'bootstrap':stats},indent=2))
if __name__=='__main__': main()
