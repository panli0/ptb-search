#!/usr/bin/env python3
import json
from pathlib import Path
src=Path('gap_public/FINITE_GROUP_PROPERTY_GRAPH.json')
outdir=Path('gap_public/smallgroups_out'); outdir.mkdir(parents=True,exist_ok=True)
g=json.loads(src.read_text())
keep={
 'IsCommutative','IsCyclic','IsElementaryAbelian','IsMonomialGroup','IsNilpotentGroup','IsPGroup',
 'IsPerfectGroup','IsPolycyclicGroup','IsPowerfulPGroup','IsQuasisimpleGroup','IsRegularPGroup',
 'IsSimpleGroup','IsSolvableGroup','IsSupersolvableGroup','IsTrivial','IsFittingFree','IsFrattiniFree'
}
edges=[e for e in g['edges'] if e['source'] in keep and e['target'] in keep]
nodes=sorted({x for e in edges for x in (e['source'],e['target'])})
if len(nodes)<5 or len(edges)<4: raise SystemExit('abstract graph unexpectedly tiny')
out={'gap_commit':g['gap_commit'],'domain':'finite abstract groups','nodes':nodes,'edges':edges,'node_count':len(nodes),'edge_count':len(edges)}
(outdir/'ABSTRACT_GRAPH.json').write_text(json.dumps(out,indent=2)+'\n')
props=', '.join(nodes); names=', '.join('"'+n+'"' for n in nodes)
script='SizeScreen([1000000,1000000]);;\n'
script+='if LoadPackage("smallgrp")=fail then Error("smallgrp unavailable"); fi;\n'
script+=f'names := [{names}];; props := [{props}];;\n'
script+='Print("order,id");; for n in names do Print(",",n); od; Print("\\n");;\n'
script+='for ord in [1..96] do if SmallGroupsAvailable(ord) then for id in [1..NumberSmallGroups(ord)] do G:=SmallGroup(ord,id);; Print(ord,",",id);; for p in props do v:=p(G);; if v=true then Print(",1"); else Print(",0"); fi; od; Print("\\n"); od; fi; od;\nQUIT;\n'
Path('/tmp/eval_smallgroups.g').write_text(script)
print(json.dumps({'gap_commit':g['gap_commit'],'nodes':nodes,'edges':[[e['source'],e['target']] for e in edges]},indent=2))
