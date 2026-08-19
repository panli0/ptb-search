#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,random,sys,types
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
K=20; FRACTION=0.75; INSTANCE_BASE_SEED=202608190000; BOOT_SEED=202608190777; BOOT_REPS=10000
METHODS=['random','small_first','tpe','target_mean','target_ucb','target_eps','psc','wpsc','bestfirst']
EXPECTED={'random':182.35,'small_first':175.0,'tpe':177.0,'target_mean':189.8,'target_ucb':197.3,'target_eps':183.9,'psc':185.2,'wpsc':185.4,'bestfirst':185.2}

def load_module_defs(path,name,argv,marker=None):
    text=path.read_text()
    if marker is not None:
        assert marker in text, f'marker missing in {path}'
        text=text.split(marker,1)[0]
    m=types.ModuleType(name); sys.modules[name]=m
    old=sys.argv[:]; sys.argv=argv
    try: exec(compile(text,str(path),'exec'),m.__dict__)
    finally: sys.argv=old
    return m

def load_ou(path,pi):
    return load_module_defs(path,'p1_ou',[str(path),'/tmp/unused-dart',str(pi),'/tmp/unused-maspob','/tmp/unused.json','scalar','all'],'\ndef main():\n')
def load_nv(path,pi):
    return load_module_defs(path,'p1_nv',[str(path),'/tmp/unused-dart',str(pi),'/tmp/unused.json','psc'],'\nres = {\n')
def load_tpe(path,pi):
    return load_module_defs(path,'p1_tpe',[str(path),str(pi),'/tmp/unused-tpe.json'],'\n# Preserve old formal protocol: one seed, seed=0.\n')

def recompute_U(ids,cov):
    u=set()
    for o in ids:u|=set(cov[o])
    return u

def subset_ou(d,keep):
    ids,txt,cov,_,pt,tn,nodes,edges=d; ids=sorted(keep); return ids,txt,cov,recompute_U(ids,cov),pt,tn,nodes,edges
def subset_nv(d,keep):
    ids,txt,cov,_,pt,tt=d; ids=sorted(keep); return ids,txt,cov,recompute_U(ids,cov),pt,tt

def hit90(ou,order,d): return float(ou.evaluate(order,d[2],d[3])[0]['0.9'])
def score_random(ou,d):
    vals=[]
    for s in range(20):
        o=list(d[0]);random.Random(s).shuffle(o);vals.append(hit90(ou,o,d))
    return float(np.mean(vals))
def score_target(ou,d,mode):
    ids,txt,cov,_,pt,*_=d;env=ou.StrictEnv(ids,cov)
    return float(np.mean([hit90(ou,ou.target_policy(ids,txt,env,pt,mode,s),d) for s in range(10)]))
def score_nv(nv,d,meth):
    ids,txt,cov,U,pt,tt=d
    return float(np.mean([float(nv.run_seed(ids,txt,cov,U,pt,tt,meth,s)['hits']['0.9']) for s in range(5)]))
def score_small(tpe,ou,d): return hit90(ou,sorted(d[0],key=lambda s:(tpe.card(s),s)),d)
def score_tpe(tpe,ou,d):
    old_ids,old_U=tpe.ids,tpe.eval_universe
    try:
        tpe.ids=list(d[0]);tpe.eval_universe=set(d[3]);o=tpe.tpe_policy(seed=0);return hit90(ou,o,d)
    finally:tpe.ids, tpe.eval_universe=old_ids,old_U

def all_scores(ou,nv,tpe,od,nd):
    z={'random':score_random(ou,od),'small_first':score_small(tpe,ou,od),'tpe':score_tpe(tpe,ou,od)}
    for m in ('target_mean','target_ucb','target_eps'):z[m]=score_target(ou,od,m)
    for m in ('psc','wpsc','bestfirst'):z[m]=score_nv(nv,nd,m)
    return z

def tau(a,b):
    t=kendalltau([a[m] for m in METHODS],[b[m] for m in METHODS],variant='b').statistic
    return None if not np.isfinite(t) else float(t)
def bstat(rows,m):
    d=np.asarray([r['scores'][m]-r['scores']['random'] for r in rows]);rng=np.random.default_rng(BOOT_SEED+METHODS.index(m));means=[]
    for _ in range(BOOT_REPS):means.append(float(d[rng.integers(0,len(d),len(d))].mean()))
    return {'mean_calls_diff_vs_random':float(d.mean()),'bootstrap95':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],
            'instances_better_than_random':int((d<0).sum()),'instances_equal_random':int((d==0).sum()),'n':len(d)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pi',type=Path,required=True);ap.add_argument('--ou',type=Path,required=True);ap.add_argument('--nv',type=Path,required=True);ap.add_argument('--tpe',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);A=ap.parse_args();A.outdir.mkdir(parents=True,exist_ok=True)
    ou=load_ou(A.ou,A.pi);nv=load_nv(A.nv,A.pi);tp=load_tpe(A.tpe,A.pi);od=ou.pi_data();nd=nv.pi_data()
    assert od[0]==nd[0] and od[2]==nd[2] and od[3]==nd[3]
    full=all_scores(ou,nv,tp,od,nd);gate={m:abs(full[m]-EXPECTED[m])<1e-9 for m in METHODS}
    if not all(gate.values()):raise RuntimeError('GOLD_REPRO_GATE_FAIL '+json.dumps({'observed':full,'expected':EXPECTED,'gate':gate}))
    ids=od[0];nkeep=math.ceil(FRACTION*len(ids));rows=[]
    for i in range(K):
        keep=sorted(random.Random(INSTANCE_BASE_SEED+i).sample(ids,nkeep));o2=subset_ou(od,keep);n2=subset_nv(nd,keep);s=all_scores(ou,nv,tp,o2,n2)
        rows.append({'instance':i,'seed':INSTANCE_BASE_SEED+i,'objects':len(keep),'reachable_targets':len(o2[3]),'scores':s,'tau_vs_full':tau(s,full)});print('pi-base instance',i,'done',s,flush=True)
    pair=[]
    for i in range(K):
        for j in range(i+1,K):
            t=tau(rows[i]['scores'],rows[j]['scores']);
            if t is not None:pair.append(t)
    tv=[r['tau_vs_full'] for r in rows if r['tau_vs_full'] is not None];primary=float(np.median(pair));verdict='TAU_LT_0_4_RANKING_UNSTABLE' if primary<.4 else ('TAU_GE_0_6_RANKING_STABLE' if primary>=.6 else 'TAU_0_4_TO_0_6_AMBIGUOUS');stats={m:bstat(rows,m) for m in METHODS if m!='random'}
    res={'version':1,'status':'PASS','domain':'pi-base','protocol':{'K':K,'action_pool_fraction':FRACTION,'same_instance_pool_for_all_methods':True,'instance_base_seed':INSTANCE_BASE_SEED,'policy_target_universe':'all official theorem converses; unchanged by subsampling','evaluation_universe':'union of coverage rows reachable by sampled spaces','commit_before_reveal':True,'gold_stochastic_seed_counts':{'random':20,'tpe':1,'target':10,'predicted_planners':5},'ridge_knn_note':'not defined in the pi-Base Gold ledger; not invented for P1','primary_tau':'median pairwise Kendall tau-b across K instance ranking vectors','bootstrap_reps':BOOT_REPS,'bootstrap_seed':BOOT_SEED},'gold_reproduction':{'observed':full,'expected':EXPECTED,'all_exact':True},'instances':rows,'ranking_stability':{'pair_count':len(pair),'pairwise_tau_median':primary,'pairwise_tau_mean':float(np.mean(pair)),'pairwise_tau_p05_p95':[float(np.quantile(pair,.05)),float(np.quantile(pair,.95))],'tau_vs_full_median':float(np.median(tv)),'tau_vs_full_mean':float(np.mean(tv)),'verdict':verdict},'paired_bootstrap_vs_random':stats}
    jp=A.outdir/'P1_PIBASE_CHEAP_RESULTS.json';jp.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');cp=A.outdir/'P1_PIBASE_INSTANCES.csv'
    with cp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['instance','seed','objects','reachable_targets','tau_vs_full']+METHODS)
        for r in rows:w.writerow([r['instance'],r['seed'],r['objects'],r['reachable_targets'],r['tau_vs_full']]+[r['scores'][m] for m in METHODS])
    md=A.outdir/'P1_PIBASE_CHEAP_READOUT.md';lines=['# P1 pi-Base — 75% action-pool × 20 stability audit','','**Status: PASS. Gold reproduction gate passed exactly before subsampling.**','',f'- Primary median pairwise Kendall tau-b: **{primary:.3f}**',f'- Mean pairwise tau-b: {np.mean(pair):.3f}',f'- Median tau vs full Gold ranking: {np.median(tv):.3f}',f'- Verdict: **{verdict}**','','| Method | mean calls diff vs random | paired bootstrap 95% CI | better instances / 20 |','|---|---:|---:|---:|']
    for m in METHODS:
        if m=='random':continue
        z=stats[m];lines.append(f"| {m} | {z['mean_calls_diff_vs_random']:.3f} | [{z['bootstrap95'][0]:.3f}, {z['bootstrap95'][1]:.3f}] | {z['instances_better_than_random']} |")
    lines+=['','Negative call difference means fewer calls than random. Ridge/LinUCB/kNN are N/A in the frozen pi-Base Gold ledger and were not newly invented for this audit.'];md.write_text('\n'.join(lines)+'\n');hp=A.outdir/'P1_PIBASE_SHA256SUMS.txt';hp.write_text('\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}' for p in [jp,cp,md])+'\n');print(json.dumps({'primary_tau':primary,'verdict':verdict,'bootstrap':stats},indent=2))
if __name__=='__main__':main()
