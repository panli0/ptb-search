from __future__ import annotations
import gzip, io, json, re, tarfile, zipfile
from pathlib import Path
import gdown, requests

OUT=Path('lpt_compact'); OUT.mkdir(exist_ok=True)
R={'provenance':{}}

# ---------- DiscoveryWorld: official Hypothesizer / Normal / GPT-4o public logs ----------
DW_ID='1-RonnUu-j9jtfAY2PBrC1udOeGMh4DQr'
dwzip=OUT/'hypothesizer-normal.zip'
gdown.download(id=DW_ID, output=str(dwzip), quiet=False)
R['provenance']['discoveryworld']={'drive_id':DW_ID,'bytes':dwzip.stat().st_size}
dw_runs=[]
with zipfile.ZipFile(dwzip) as z:
    names=[n for n in z.namelist() if 'output_allhistory.' in n and n.endswith('.json') and not n.startswith('__MACOSX')]
    for n in names:
        try:
            rows=json.loads(z.read(n))
        except Exception as e:
            dw_runs.append({'file':n,'error':repr(e)}); continue
        m=re.search(r'output_allhistory\.(.*?)-Normal-s(\d+)-',n)
        theme=m.group(1) if m else None; seed=int(m.group(2)) if m else None
        rec={'file':n,'theme':theme,'seed':seed,'n_raw_rows':len(rows),'steps':[]}
        env_i=0
        for raw_i,r in enumerate(rows):
            if not isinstance(r,dict): continue
            if 'consolidated_scientific_knowledge' in r and 'nextAction' not in r: continue
            step={'env_i':env_i,'raw_i':raw_i,'actionSuccess':r.get('actionSuccess'),'nextAction':r.get('nextAction'),'oracle_scorecard':r.get('oracle_scorecard')}
            rec['steps'].append(step); env_i+=1
        dw_runs.append(rec)
R['discoveryworld_runs']=dw_runs

# ---------- AgentBoard: official released result files + task subgoals ----------
AB_URL='https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz'
abtar=OUT/'agentboard-data.tar.gz'
with requests.get(AB_URL,stream=True,timeout=120) as rr:
    rr.raise_for_status()
    with abtar.open('wb') as f:
        for c in rr.iter_content(1024*1024):
            if c: f.write(c)
R['provenance']['agentboard']={'url':AB_URL,'bytes':abtar.stat().st_size}
ab={'tasks':{},'baseline_text':{}}
with tarfile.open(abtar,'r:gz') as tf:
    for m in tf:
        if not m.isfile(): continue
        name=m.name
        lname=name.lower()
        want_task=any(name.endswith(f'data/{env}/test.jsonl') or f'/data/{env}/test.jsonl' in name for env in ['alfworld','pddl','scienceworld'])
        want_base=('baseline_results/' in name and any(name.endswith('/'+env+'.txt') for env in ['alfworld','pddl','scienceworld']))
        if not (want_task or want_base): continue
        b=tf.extractfile(m).read()
        txt=b.decode('utf-8','replace')
        if want_task:
            env=next(env for env in ['alfworld','pddl','scienceworld'] if f'/{env}/test.jsonl' in '/'+name or name.endswith(f'data/{env}/test.jsonl'))
            rows=[]
            for line in txt.splitlines():
                try: rows.append(json.loads(line))
                except: pass
            ab['tasks'][env]=rows
        else:
            mm=re.search(r'baseline_results/([^/]+)/([^/]+)\.txt$',name)
            if mm: ab['baseline_text'][mm.group(1)+'/'+mm.group(2)]=txt
R['agentboard']=ab

# ---------- AppWorld: official bundled GPT-4o experiment outputs ----------
AW_URL='https://s3.us-west-2.amazonaws.com/appworld.dev/experiment-outputs-0.1.3.bundle'
awpath=OUT/'appworld.bundle'
with requests.get(AW_URL,stream=True,timeout=120) as rr:
    rr.raise_for_status()
    with awpath.open('wb') as f:
        for c in rr.iter_content(1024*1024):
            if c: f.write(c)
R['provenance']['appworld']={'url':AW_URL,'bytes':awpath.stat().st_size}
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
enc=awpath.read_bytes(); password='WEquKLy##9M@qu'; salt=b'Nvx#rYcYQ2%btf'
kdf=PBKDF2HMAC(algorithm=hashes.SHA256(),length=32,salt=salt,iterations=100000,backend=default_backend()); key=kdf.derive(password.encode())
iv=enc[:16]; cipher=Cipher(algorithms.AES(key),modes.CFB(iv),backend=default_backend()); dec=cipher.decryptor(); raw=dec.update(enc[16:])+dec.finalize()
z=zipfile.ZipFile(io.BytesIO(raw))
# Prefer legacy GPT-4o function-calling baseline because it is the paper-era public baseline and has full logs.
prefix='legacy_function_calling_agent/openai/gpt-4o-2024-05-13/'
aw={}
for n in z.namelist():
    if not n.startswith(prefix): continue
    rel=n[len(prefix):]
    m=re.match(r'(test_normal|test_challenge)/tasks/([^/]+)/(logs/environment_io\.md|logs/api_calls\.jsonl|evaluation/report\.md)$',rel)
    if not m: continue
    split,tid,kind=m.groups(); task=aw.setdefault(split,{}).setdefault(tid,{})
    task[kind]=z.read(n).decode('utf-8','replace')
R['appworld']=aw

# Compact gzip JSON. Raw downloaded archives are intentionally not uploaded.
with gzip.open(OUT/'public_compact.json.gz','wt',encoding='utf-8',compresslevel=6) as f:
    json.dump(R,f,ensure_ascii=False,separators=(',',':'),default=str)
manifest={
 'discoveryworld_runs':len([x for x in dw_runs if 'steps' in x]),
 'discoveryworld_steps':sum(len(x.get('steps',[])) for x in dw_runs),
 'agentboard_task_counts':{k:len(v) for k,v in ab['tasks'].items()},
 'agentboard_baseline_files':len(ab['baseline_text']),
 'appworld_task_counts':{k:len(v) for k,v in aw.items()},
 'compact_bytes':(OUT/'public_compact.json.gz').stat().st_size,
}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
