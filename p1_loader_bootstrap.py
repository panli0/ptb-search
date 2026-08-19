#!/usr/bin/env python3
"""Infrastructure-only loader fix for P1.

The frozen Gold runners use dataclasses. Python 3.11 requires dynamically
loaded modules to be registered in sys.modules before their class decorators
execute. This wrapper makes only those two registration edits in-memory, then
executes the otherwise byte-for-byte P1 audit script.
"""
from pathlib import Path

src = Path(__file__).with_name('p1_ring_module_stability.py')
text = src.read_text()
old1 = "m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m)"
new1 = "m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; assert spec.loader; spec.loader.exec_module(m)"
old2 = "m=types.ModuleType('p1_nv')"
new2 = "m=types.ModuleType('p1_nv'); sys.modules[m.__name__]=m"
assert text.count(old1) == 1, text.count(old1)
assert text.count(old2) == 1, text.count(old2)
text = text.replace(old1, new1).replace(old2, new2)
ns = {'__name__': '__main__', '__file__': str(src)}
exec(compile(text, str(src), 'exec'), ns)
