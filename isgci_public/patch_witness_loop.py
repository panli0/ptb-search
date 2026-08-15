#!/usr/bin/env python3
from pathlib import Path
p=Path('isgci_public/prepare_isgci_forbidden_fast.py')
s=p.read_text(encoding='utf-8')
s=s.replace(' W=defaultdict(list)\n',' W={}\n',1)
old="""   while m:y=m&-m;v=y.bit_length()-1;m-=y\n   if u!=v and not nx.has_path(S,u,v):W[(u,v)].append(oi)\n"""
new="""   while m:\n    y=m&-m;v=y.bit_length()-1;m-=y\n    if u!=v:\n     key=(u,v)\n     if key not in W: W[key]=[oi,oi]\n     else: W[key][1]=oi\n"""
if old not in s:
    raise SystemExit('expected buggy witness loop not found')
s=s.replace(old,new,1)
old2="oi=min(ois) if b=='first' else max(ois);A.append((u,v,of[oi]))"
new2="oi=ois[0] if b=='first' else ois[1];A.append((u,v,of[oi]))"
if old2 not in s:
    raise SystemExit('expected backend witness selection not found')
s=s.replace(old2,new2,1)
p.write_text(s,encoding='utf-8')
print('patched witness enumeration: every false target; exact first/last witness only; redundant path search removed after full inclusion-consistency check')
