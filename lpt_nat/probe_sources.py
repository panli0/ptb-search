from __future__ import annotations
import json, os, subprocess, sys, tarfile, re
from pathlib import Path
R={}; ROOT=Path('lpt_probe'); ROOT.mkdir(exist_ok=True)
def run(cmd,timeout=3600):
    print('$',cmd,flush=True); p=subprocess.run(cmd,shell=True,text=True,capture_output=True,timeout=timeout); print(p.stdout[-30000:]); print(p.stderr[-10000:],file=sys.stderr); return p
# AgentBoard: inspect complete archive inventory, not just baseline_results
arc=ROOT/'agentboard.tar.gz'
p=run(f"wget -q --show-progress -O '{arc}' 'https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz'")
R['ab_download_rc']=p.returncode
if arc.exists():
    with tarfile.open(arc,'r:gz') as tf:
        ms=tf.getmembers()
        cand=[m for m in ms if m.isfile() and (m.name.lower().endswith(('.jsonl','.json','.pkl','.pickle','.gz','.zip')) or any(x in m.name.lower() for x in ['trajectory','log','history','baseline_result']))]
        R['ab_candidate_files']=[{'name':m.name,'size':m.size} for m in cand]
        # extract tiny candidate files under 5 MB for schema samples
        sam=[]
        for m in cand:
            if m.size <= 5_000_000:
                try:
                    f=tf.extractfile(m)
                    if f:
                        b=f.read(20000); sam.append({'name':m.name,'sample':b.decode('utf-8','replace')})
                except Exception as e: sam.append({'name':m.name,'error':repr(e)})
                if len(sam)>=80: break
        R['ab_candidate_samples']=sam
# DiscoveryWorld enumerate drive folder
try:
 import gdown
 url='https://drive.google.com/drive/folders/14FucVzVCm1HZ0EfPEKoPwdRsFnZi769k'
 try:
  x=gdown.download_folder(url=url,output=str(ROOT/'dw_listing'),quiet=False,use_cookies=False,skip_download=True)
  R['dw_listing_type']=str(type(x)); R['dw_listing_repr']=repr(x)[:300000]
 except Exception as e: R['dw_listing_error']=repr(e)
except Exception as e: R['dw_outer_error']=repr(e)
# AppWorld direct official S3 bundle; inspect header and download if <= 3 GB
import requests
url='https://s3.us-west-2.amazonaws.com/appworld.dev/experiment-outputs-0.1.3.bundle'
try:
 h=requests.head(url,allow_redirects=True,timeout=60); R['aw_head']={'status':h.status_code,'headers':dict(h.headers)}
 size=int(h.headers.get('content-length') or 0)
 if 0 < size <= 3_000_000_000:
  dest=ROOT/'experiment-outputs-0.1.3.bundle'; q=run(f"wget -q --show-progress -O '{dest}' '{url}'")
  R['aw_download_rc']=q.returncode; R['aw_size']=dest.stat().st_size if dest.exists() else None
  if dest.exists(): R['aw_header_hex']=dest.read_bytes()[:128].hex()
except Exception as e: R['aw_error']=repr(e)
Path('lpt_probe.json').write_text(json.dumps(R,ensure_ascii=False,indent=2,default=str))
print('===PROBE==='); print(Path('lpt_probe.json').read_text()[:500000])
