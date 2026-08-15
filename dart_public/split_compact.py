#!/usr/bin/env python3
from pathlib import Path

src = Path('dart_public/out/DART_COMPACT_INPUT.b85')
s = src.read_text(encoding='ascii').strip()
out = Path('dart_public/out/chunks')
out.mkdir(parents=True, exist_ok=True)
for p in out.glob('part_*.txt'):
    p.unlink()
chunk = 5000
for i in range(0, len(s), chunk):
    (out / f'part_{i//chunk:02d}.txt').write_text(s[i:i+chunk] + '\n', encoding='ascii')
(out / 'MANIFEST.txt').write_text(f'length={len(s)}\nchunks={(len(s)+chunk-1)//chunk}\nchunk_size={chunk}\n', encoding='ascii')
print((out / 'MANIFEST.txt').read_text())
