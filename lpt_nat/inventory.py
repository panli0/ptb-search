from __future__ import annotations
import json, os, re, subprocess, sys, tarfile, shutil
from pathlib import Path

ROOT=Path('lpt_work'); ROOT.mkdir(exist_ok=True)
R={}

def run(cmd, timeout=1800, env=None):
    print('$',cmd,flush=True)
    p=subprocess.run(cmd,shell=True,text=True,capture_output=True,timeout=timeout,env=env)
    print(p.stdout[-20000:]); print(p.stderr[-10000:],file=sys.stderr)
    return p

# ---------------- AgentBoard ----------------
ab=ROOT/'agentboard'; ab.mkdir(exist_ok=True)
arc=ab/'data.tar.gz'
if not arc.exists():
    p=run(f"wget -q --show-progress -O '{arc}' 'https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz'",timeout=3600)
    R['agentboard_download_rc']=p.returncode
if arc.exists():
    R['agentboard_archive_bytes']=arc.stat().st_size
    with tarfile.open(arc,'r:gz') as tf:
        ms=tf.getmembers(); names=[m.name for m in ms]
        R['agentboard_entries']=len(ms)
        bas=[m for m in ms if 'baseline_results' in m.name]
        R['agentboard_baseline_entries']=[{'name':m.name,'size':m.size} for m in bas]
        # extract all baseline_results only
        for m in bas:
            try: tf.extract(m, ab/'x', filter='data')
            except TypeError: tf.extract(m, ab/'x')
    files=[p for p in (ab/'x').rglob('*') if p.is_file()]
    R['agentboard_baseline_files']=[{'path':str(p),'size':p.stat().st_size} for p in files]
    samples={}
    for p in files:
        if p.suffix.lower() in ['.json','.jsonl','.txt','.csv'] and p.stat().st_size<30_000_000:
            try:
                samples[str(p)]=p.read_text(errors='replace')[:12000]
            except Exception as e: samples[str(p)]='ERR '+repr(e)
            if len(samples)>=30: break
    R['agentboard_samples']=samples

# ---------------- DiscoveryWorld ----------------
dw=ROOT/'discoveryworld'; dw.mkdir(exist_ok=True)
try:
    import gdown
    url='https://drive.google.com/drive/folders/14FucVzVCm1HZ0EfPEKoPwdRsFnZi769k'
    # gdown internals can enumerate file urls without downloading in recent versions
    try:
        out=gdown.download_folder(url=url, output=str(dw/'listing'), quiet=False, use_cookies=False, remaining_ok=True, skip_download=True)
        R['dw_skip_download']=str(out)
        # stringify any returned objects deeply enough
        R['dw_skip_download_repr']=repr(out)[:100000]
    except Exception as e:
        R['dw_skip_download_error']=repr(e)
    # download folder structure if total is manageable; gdown will show sizes only after transfer, so use Drive HTML helper via gdown parse_url if available
    # Try direct folder download with a 10GB runner budget; archives are expected to be separate and can be inspected afterward.
    outdir=dw/'full'
    try:
        out=gdown.download_folder(url=url, output=str(outdir), quiet=False, use_cookies=False, remaining_ok=True)
        R['dw_download_result']=str(out)
    except Exception as e:
        R['dw_download_error']=repr(e)
    files=[p for p in outdir.rglob('*') if p.is_file()] if outdir.exists() else []
    R['dw_files']=[{'path':str(p),'size':p.stat().st_size} for p in files]
except Exception as e:
    R['dw_gdown_outer_error']=repr(e)

# ---------------- AppWorld ----------------
aw=ROOT/'appworld'; aw.mkdir(exist_ok=True)
env=os.environ.copy(); env['APPWORLD_ROOT']=str(aw.resolve())
try:
    p=run('appworld download experiment-outputs',timeout=3600,env=env)
    R['appworld_download_rc']=p.returncode; R['appworld_download_stdout']=p.stdout[-15000:]; R['appworld_download_stderr']=p.stderr[-15000:]
    files=[p for p in aw.rglob('*') if p.is_file()]
    R['appworld_files']=[{'path':str(p),'size':p.stat().st_size} for p in files[:5000]]
    samples={}
    for p in files:
        if p.suffix.lower() in ['.json','.jsonl','.txt','.md','.csv'] and p.stat().st_size<30_000_000:
            try: samples[str(p)]=p.read_text(errors='replace')[:12000]
            except Exception as e: samples[str(p)]='ERR '+repr(e)
            if len(samples)>=40: break
    R['appworld_samples']=samples
except Exception as e:
    R['appworld_outer_error']=repr(e)

Path('lpt_inventory.json').write_text(json.dumps(R,ensure_ascii=False,indent=2,default=str))
print('=== INVENTORY ===')
print(Path('lpt_inventory.json').read_text()[:200000])
