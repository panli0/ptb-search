from __future__ import annotations
import io,json,os,re,subprocess,sys,zipfile,glob
from pathlib import Path
ROOT=Path('lpt_extract'); ROOT.mkdir(exist_ok=True); R={}
def run(cmd,timeout=3600):
 print('$',cmd,flush=True); p=subprocess.run(cmd,shell=True,text=True,capture_output=True,timeout=timeout); print(p.stdout[-10000:]);print(p.stderr[-10000:],file=sys.stderr);return p
# DiscoveryWorld Hypothesizer Normal + human summary
try:
 import gdown
 files={
  'hypothesizer_normal':'1-RonnUu-j9jtfAY2PBrC1udOeGMh4DQr',
  'human_summary':'1eqzM-LatSGWgUl6ZzuP90359F2l8N1Wj'
 }
 for key,fid in files.items():
  p=ROOT/f'{key}.zip'; gdown.download(id=fid,output=str(p),quiet=False)
  R[key+'_bytes']=p.stat().st_size
  with zipfile.ZipFile(p) as z:
   ns=z.namelist(); R[key+'_names']=ns[:10000]
   meta=[]
   for n in ns:
    if n.endswith('/') or n.startswith('__MACOSX'): continue
    info=z.getinfo(n); rec={'name':n,'size':info.file_size}
    if n.lower().endswith('.json') and info.file_size<100_000_000:
     try:
      obj=json.loads(z.read(n)); rec['json_type']=type(obj).__name__
      if isinstance(obj,list):
       rec['len']=len(obj)
       if obj and isinstance(obj[0],dict): rec['item_keys']=list(obj[0].keys())
      elif isinstance(obj,dict): rec['keys']=list(obj.keys())
     except Exception as e: rec['json_error']=repr(e)
    meta.append(rec)
   R[key+'_meta']=meta[:20000]
except Exception as e:R['dw_error']=repr(e)
# AppWorld direct bundle decrypt+unzip, only structural schema output
try:
 import requests
 from cryptography.hazmat.backends import default_backend
 from cryptography.hazmat.primitives import hashes
 from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
 from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
 url='https://s3.us-west-2.amazonaws.com/appworld.dev/experiment-outputs-0.1.3.bundle'; bpath=ROOT/'appworld.bundle'
 with requests.get(url,stream=True,timeout=120) as r:
  r.raise_for_status();
  with open(bpath,'wb') as f:
   for c in r.iter_content(1024*1024):
    if c:f.write(c)
 R['aw_bundle_bytes']=bpath.stat().st_size
 enc=bpath.read_bytes(); password='WEquKLy##9M@qu'; salt=b'Nvx#rYcYQ2%btf'
 kdf=PBKDF2HMAC(algorithm=hashes.SHA256(),length=32,salt=salt,iterations=100000,backend=default_backend()); key=kdf.derive(password.encode())
 iv=enc[:16]; cipher=Cipher(algorithms.AES(key),modes.CFB(iv),backend=default_backend()); dec=cipher.decryptor(); raw=dec.update(enc[16:])+dec.finalize()
 z=zipfile.ZipFile(io.BytesIO(raw)); ns=z.namelist(); R['aw_names']=ns[:30000]
 meta=[]
 for n in ns:
  if n.endswith('/'):continue
  info=z.getinfo(n); rec={'name':n,'size':info.file_size}
  if n.lower().endswith(('.json','.jsonl','.md','.txt')) and info.file_size<20_000_000:
   try:
    bb=z.read(n)
    if n.lower().endswith('.json'):
     obj=json.loads(bb);rec['json_type']=type(obj).__name__
     if isinstance(obj,dict):rec['keys']=list(obj.keys())
     elif isinstance(obj,list):
      rec['len']=len(obj)
      if obj and isinstance(obj[0],dict):rec['item_keys']=list(obj[0].keys())
    elif n.lower().endswith('.jsonl'):
     line=bb.splitlines()[0] if bb.splitlines() else b''; obj=json.loads(line) if line else None
     rec['first_line_type']=type(obj).__name__
     if isinstance(obj,dict):rec['first_line_keys']=list(obj.keys())
    elif n.lower().endswith('.md'):
     txt=bb.decode('utf8','replace');rec['heading_lines']=[x for x in txt.splitlines() if x.startswith('#')][:20]
   except Exception as e:rec['parse_error']=repr(e)
  meta.append(rec)
 R['aw_meta']=meta[:50000]
except Exception as e:R['aw_error']=repr(e)
Path('lpt_extract_schema.json').write_text(json.dumps(R,ensure_ascii=False,indent=2,default=str))
print('DONE',len(json.dumps(R)))
