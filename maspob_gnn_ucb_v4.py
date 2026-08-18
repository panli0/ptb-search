#!/usr/bin/env python3
import copy, json, random, re, sys, yaml, numpy as np, torch
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import HashingVectorizer

ROOT=Path(sys.argv[1]); PI=Path(sys.argv[2]); MAS=Path(sys.argv[3]); OUT=Path(sys.argv[4]); TH=(.5,.7,.9,.95,.99,1.0)
sys.path.insert(0,str(MAS))
from scripts.gnn_model import WorkflowGAT, initialize_fisher, update_fisher, compute_prediction_and_uncertainty


def curve(order,cov,eval_U):
    seen=set(); vals=[]
    for o in order:
        seen |= (cov[o] & eval_U)
        vals.append(len(seen)/max(1,len(eval_U)))
    return {str(q):next((i+1 for i,x in enumerate(vals) if x+1e-12>=q),None) for q in TH}, float(np.mean(vals)) if vals else 0.0


def atomr(s):
    m=re.fullmatch(r'\s*ring_deduced\("(has|lacks)",\s*(\d+),\s*(\d+)\)\s*',str(s or'')); return (int(m[3]),int(m[2]),m[1]=='has') if m else None

def atomm(s):
    m=re.fullmatch(r'\s*module_deduced\("(has|lacks)",\s*(\d+)\)\s*',str(s or'')); return (int(m[2]),m[1]=='has') if m else None

def atoms_pi(f):
    if not isinstance(f,dict): return []
    if f.get('kind')=='atom': return [f.get('property')]
    out=[]
    for z in f.get('subs',[]): out += atoms_pi(z)
    return [x for x in out if x is not None]


def pi_data():
    d=json.loads(PI.read_text()); S={x['uid']:x for x in d['spaces']}; T={x['uid']:x for x in d['theorems']}; truth={s:{} for s in S}
    props={p['uid'] for p in d['properties']}
    for x in d['traits']:
        if x.get('space') in S and isinstance(x.get('value'),bool): truth[x['space']][x['property']]=x['value']
    def ev(f,tr):
        if not isinstance(f,dict): return None
        if f.get('kind')=='atom': return tr.get(f.get('property'))==f.get('value') if f.get('property') in tr else None
        z=[ev(x,tr) for x in f.get('subs',[])]
        if f.get('kind')=='and': return False if False in z else (True if z and all(x is True for x in z) else None)
        if f.get('kind')=='or': return True if True in z else (False if z and all(x is False for x in z) else None)
        return None
    cov={s:set() for s in S}; tn={}; edges=[]
    for tid,t in T.items():
        a=atoms_pi(t.get('when')); b=atoms_pi(t.get('then')); tn[tid]=set(a+b)
        for x in a:
            for y in b:
                if x in props and y in props: edges.append((x,y))
    for s,tr in truth.items():
        for tid,t in T.items():
            if ev(t.get('then'),tr) is True and ev(t.get('when'),tr) is False: cov[s].add(tid)
    eval_U=set().union(*cov.values()) if cov else set(); policy_U=set(T)
    txt={s:' '.join(str(S[s].get(k,'') or '') for k in ('name','description','aliases')) for s in S}
    nodes=sorted({x for e in edges for x in e} | set().union(*tn.values()))
    return sorted(S),txt,cov,eval_U,policy_U,tn,nodes,edges


def dart(dom):
    ring=dom=='ring'; b=ROOT/('db/ringapp' if ring else 'db/moduleapp'); r=b/('ring' if ring else 'module'); parse=atomr if ring else atomm; tr={}; txt={}
    for od in sorted(x for x in r.iterdir() if x.is_dir()):
        if not (od/'properties.yaml').exists(): continue
        try: p=yaml.safe_load((od/'properties.yaml').read_text()) or {}; m=yaml.safe_load((od/'data.yaml').read_text()) or {}
        except Exception: continue
        q={}
        if ring:
            for k,v in p.items():
                try: i=int(k.split('_')[-1])
                except Exception: continue
                l=v.get('has_on_left'); rr=v.get('has_on_right')
                if isinstance(l,bool) and isinstance(rr,bool): q[(i,0)]=l; q[(i,1)]=rr; q[(i,2)]=l if l==rr else None
            txt[od.name]=' '.join([str(m.get('name','')),str(m.get('description','')),' '.join(map(str,m.get('keywords') or [])),'comm='+str(m.get('is_commutative'))])
        else:
            for k,v in p.items():
                try: i=int(k.split('_')[-1])
                except Exception: continue
                if isinstance(v.get('has'),bool): q[i]=v['has']
            txt[od.name]=' '.join([str(m.get('name','')),str(m.get('description','')),'ring='+str(m.get('ring','')),'opp='+str(m.get('opposite_ring',''))])
        tr[od.name]=q
    rules=[]; tn={}; edges=[]
    for f in sorted((b/'logic').glob('*.yaml')):
        try: d=yaml.safe_load(f.read_text()) or {}
        except Exception: continue
        if d.get('active') is False: continue
        a=parse(d.get('hyps')); c=parse(d.get('concs'))
        if a and c:
            rules.append((f.stem,a,c))
            if ring: na=f'{a[0]}:{a[1]}'; nc=f'{c[0]}:{c[1]}'
            else: na=str(a[0]); nc=str(c[0])
            tn[f.stem]={na,nc}; edges.append((na,nc))
    cov={o:set() for o in tr}
    for o,q in tr.items():
        for rid,a,c in rules:
            if ring:
                pa,sa,va=a; pc,sc,vc=c; xa=q.get((pa,sa)); xc=q.get((pc,sc)); xa=q.get((pa,2)) if xa is None else xa; xc=q.get((pc,2)) if xc is None else xc
            else:
                pa,va=a; pc,vc=c; xa=q.get(pa); xc=q.get(pc)
            if isinstance(xa,bool) and isinstance(xc,bool) and xc==vc and xa!=va: cov[o].add(rid)
    eval_U=set().union(*cov.values()) if cov else set(); policy_U={rid for rid,_,_ in rules}; nodes=sorted({x for e in edges for x in e})
    return sorted(tr),txt,cov,eval_U,policy_U,tn,nodes,edges


def topology(nodes,edges):
    deps=defaultdict(list)
    for a,b in edges: deps[b].append(a)
    return [{'name':n,'dependencies':sorted(set(deps[n]))} for n in nodes]


def run_policy(ids,txt,cov,policy_U,tn,nodes,edges,seed):
    torch.manual_seed(seed); np.random.seed(seed); rng=random.Random(seed); n=len(ids); N=len(nodes); ni={x:i for i,x in enumerate(nodes)}
    indeg=Counter(b for a,b in edges); outdeg=Counter(a for a,b in edges); mx=max(1,max([*indeg.values(),*outdeg.values(),1])); V=HashingVectorizer(n_features=3,alternate_sign=False,norm='l2',ngram_range=(1,2)); TX=V.transform([txt[o] for o in ids]).toarray().astype('float32'); oi={o:i for i,o in enumerate(ids)}
    target_inc={t:[ni[x] for x in tn.get(t,set()) if x in ni] for t in policy_U}
    topo=topology(nodes,edges); device=torch.device('cpu'); model=WorkflowGAT(embedding_dim=6,num_operators=N,hidden_dim=4,num_gnn_layers=1,num_heads=1,topology=topo,dropout=0.0,bidirectional=True,use_sigmoid=True).to(device); frozen=copy.deepcopy(model).to(device); frozen.eval(); fisher=initialize_fisher('neural',frozen_model=frozen,lambda_reg=1.0); opt=torch.optim.Adam(model.parameters(),lr=.01); lossfn=torch.nn.MSELoss()
    hist_x=[]; hist_y=[]; seen=set(); rem=set(ids); order=[]
    def emb(o,seen_now):
        z=np.zeros((N,6),dtype='float32'); base=TX[oi[o]]; unc=set(policy_U)-seen_now; res=np.zeros(N,dtype='float32')
        for t in unc:
            for j in target_inc.get(t,[]): res[j]+=1
        if res.max()>0: res/=res.max()
        for j,node in enumerate(nodes): z[j,:3]=base; z[j,3]=indeg[node]/mx; z[j,4]=outdeg[node]/mx; z[j,5]=res[j]
        return torch.tensor(z.reshape(-1),dtype=torch.float32,device=device)
    warm=min(8,max(3,n//20))
    for o in rng.sample(sorted(rem),warm):
        x=emb(o,seen); gain=len(cov[o]-seen)/max(1,len(policy_U)); hist_x.append(x.detach()); hist_y.append(gain); seen|=cov[o]; order.append(o); rem.remove(o)
        g=compute_prediction_and_uncertainty('neural',model,x,frozen_model=frozen,fisher_matrix=fisher)[2]; fisher=update_fisher('neural',fisher,g,fisher_coef=10)
    while rem:
        model.train(); X=torch.stack(hist_x); Y=torch.tensor(hist_y,dtype=torch.float32,device=device)
        for _ in range(8):
            opt.zero_grad(); pred=model(X).reshape(-1); loss=lossfn(pred,Y); loss.backward(); opt.step()
        model.eval(); best=None; bestinfo=None
        for o in rem:
            x=emb(o,seen); pred,unc,g=compute_prediction_and_uncertainty('neural',model,x,frozen_model=frozen,fisher_matrix=fisher); score=pred+0.2*unc; key=(score,rng.random(),o)
            if best is None or key>best: best=key; bestinfo=(o,x,g)
        o,x,g=bestinfo; gain=len(cov[o]-seen)/max(1,len(policy_U)); hist_x.append(x.detach()); hist_y.append(gain); fisher=update_fisher('neural',fisher,g,fisher_coef=10); seen|=cov[o]; order.append(o); rem.remove(o)
    return order


def run(name,data):
    ids,txt,cov,eval_U,policy_U,tn,nodes,edges=data; z={'domain':name,'objects':len(ids),'policy_targets':len(policy_U),'evaluation_coverable_targets':len(eval_U),'graph_nodes':len(nodes),'graph_edges':len(edges),'uses_outcome_defined_target_universe_for_policy':False}
    vv=[]; hashes=[]
    for s in range(3):
        order=run_policy(ids,txt,cov,policy_U,tn,nodes,edges,s); vv.append(curve(order,cov,eval_U)); hashes.append(__import__('hashlib').sha256('\n'.join(map(str,order)).encode()).hexdigest())
    z['maspob_gnn_neural_ucb']={'adaptive':True,'uses_unqueried_truth':False,'upstream':'HZ1008/MASPOB scripts/gnn_model.py','ucb_alpha':0.2,'warm_queries':'min(8,max(3,n//20))','hits_mean':{str(q):float(np.mean([x[0][str(q)] for x in vv if x[0][str(q)] is not None])) for q in TH},'auc_mean':float(np.mean([x[1] for x in vv])),'order_sha256':hashes}
    return z

res={'protocol':'Outcome-universe-safe adapter around official MASPOB WorkflowGAT + neural-UCB utilities. Policy residual state contains ALL official theorem/rule converse targets, including targets with no positive in the benchmark. The hindsight coverable subset is used only after the trajectory for evaluation. Candidate features use only public text, public implication topology, and history from committed queries. Same model/UCB hyperparameters across domains.','domains':[run('pi-base',pi_data()),run('ring',dart('ring')),run('module',dart('module'))]}
OUT.write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
