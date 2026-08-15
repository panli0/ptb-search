#!/usr/bin/env python3
"""Export only public DaRT backend outcomes; contains no search-policy code."""
from __future__ import annotations

import argparse, base64, hashlib, json, sys, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dart_public import (
    load_properties, load_edges, build_scc, load_rings,
    families_from_rings, witness_map, actions_for_backend,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    root=Path(args.source); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    props=load_properties(root)
    _,pairs,_=load_edges(root,props)
    comps,p2c,D,succ,pred=build_scc(props,pairs)
    rings,_=load_rings(root,p2c,succ,pred)
    families,ring_family=families_from_rings(rings)
    witnesses=witness_map(rings,succ)

    pair_order=sorted(witnesses)
    pair_bytes=b''.join(bytes((a,b)) for a,b in pair_order)
    payload={
        'pair_count':len(pair_order),
        'pair_bytes_sha256':hashlib.sha256(pair_bytes).hexdigest(),
        'family_count':len(families),
        'backends':{},
    }
    for backend in ('lowest','highest'):
        actions=actions_for_backend(witnesses,ring_family,backend)
        assert [(a,b) for a,b,_ in actions]==pair_order
        raw=bytes(f for _,_,f in actions)
        payload['backends'][backend]={
            'raw_sha256':hashlib.sha256(raw).hexdigest(),
            'family_ids_b85':base64.b85encode(zlib.compress(raw,9)).decode('ascii'),
        }
    text=json.dumps(payload,separators=(',',':'),sort_keys=True)
    (out/'DART_BACKEND_OUTCOMES.json').write_text(text+'\n',encoding='utf-8')
    print(json.dumps({
        'pair_count':payload['pair_count'],
        'pair_bytes_sha256':payload['pair_bytes_sha256'],
        'file_bytes':len(text),
        'lowest_encoded':len(payload['backends']['lowest']['family_ids_b85']),
        'highest_encoded':len(payload['backends']['highest']['family_ids_b85']),
    },indent=2))

if __name__=='__main__': main()
