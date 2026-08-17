from __future__ import annotations
import json, zipfile
from pathlib import Path
import gdown
FID='1-RonnUu-j9jtfAY2PBrC1udOeGMh4DQr'
outdir=Path('lpt_dw_sample'); outdir.mkdir(exist_ok=True)
zpath=outdir/'normal.zip'
gdown.download(id=FID,output=str(zpath),quiet=False)
with zipfile.ZipFile(zpath) as z:
    names=[n for n in z.namelist() if 'output_allhistory.' in n and n.endswith('.json')]
    infos=sorted([(z.getinfo(n).file_size,n) for n in names])
    chosen=[n for _,n in infos[:6]]
    allout=[]
    for n in chosen:
        rows=json.loads(z.read(n))
        rec={'file':n,'file_size':z.getinfo(n).file_size,'n_rows':len(rows),'rows':[]}
        for i,r in enumerate(rows[:120]):
            if not isinstance(r,dict): continue
            na=r.get('nextAction')
            sc=r.get('oracle_scorecard')
            # keep just compact structures; no giant prompt/observation/memory
            rr={'i':i,'actionSuccess':r.get('actionSuccess'),'nextAction':na}
            if isinstance(sc,dict):
                rr['scorecard_keys']=list(sc.keys())
                # common compact task score fields only
                for k in ['score','maxScore','completed','completedSuccessfully','taskName','taskDescription','scoreCard','scorecard']:
                    if k in sc: rr['scorecard_'+k]=sc[k]
                if not any(k.startswith('scorecard_') for k in rr):
                    rr['scorecard_preview']=str(sc)[:8000]
            else: rr['scorecard_preview']=str(sc)[:8000]
            rec['rows'].append(rr)
        allout.append(rec)
(outdir/'sample.json').write_text(json.dumps(allout,ensure_ascii=False,indent=2,default=str))
print(json.dumps(allout,ensure_ascii=False,indent=2,default=str)[:100000])
