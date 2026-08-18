#!/usr/bin/env python
"""Merge per-rank solutions.rank*.jsonl into a single solutions.jsonl for a run dir."""
import json, os, sys, glob

def merge(outdir):
    paths=sorted(glob.glob(os.path.join(outdir,'solutions.rank*.jsonl')))
    rows=[]
    for p in paths:
        with open(p) as f:
            for ln,l in enumerate(f,1):
                l=l.strip()
                if not l: continue
                try:
                    rows.append(json.loads(l))
                except json.JSONDecodeError as e:
                    print(f"  WARN {p}:{ln} bad json ({e}): {l[:80]!r}", file=sys.stderr)
    seen=set(); uniq=[]
    for r in rows:
        if r['task_id'] in seen: continue
        seen.add(r['task_id']); uniq.append(r)
    outp=os.path.join(outdir,'solutions.jsonl')
    with open(outp,'w') as f:
        for r in uniq: f.write(json.dumps(r)+'\n')
    print(f"{outdir}: merged {len(uniq)} rows from {len(paths)} shards -> {outp}")

if __name__=='__main__':
    for d in sys.argv[1:]:
        merge(d)
