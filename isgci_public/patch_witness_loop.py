#!/usr/bin/env python3
from pathlib import Path
p=Path('isgci_public/prepare_isgci_forbidden_fast.py')
s=p.read_text(encoding='utf-8')
old="""   while m:y=m&-m;v=y.bit_length()-1;m-=y\n   if u!=v and not nx.has_path(S,u,v):W[(u,v)].append(oi)\n"""
new="""   while m:\n    y=m&-m;v=y.bit_length()-1;m-=y\n    if u!=v and not nx.has_path(S,u,v):W[(u,v)].append(oi)\n"""
if old not in s:
    raise SystemExit('expected buggy witness loop not found')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('patched witness loop: append now executes for every false target')
