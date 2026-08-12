import glob,json
Z="/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft"
for cell,exp,cap in [("he_c8",164,2060),("he_c32",164,2084),("he_c128",164,2180),
                     ("mbpp_c8",378,2060),("mbpp_c32",378,2084)]:
    rows=[]
    for f in sorted(glob.glob(f"{Z}/runs/a05_k1/{cell}/metrics.rank*.jsonl")):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    assert len(rows)==exp
    nfe=[r["process"]["nfe"] for r in rows]
    at_cap=sum(1 for x in nfe if x>=cap)
    tot=sum(nfe)
    cap_mass=sum(x for x in nfe if x>=cap)
    # count items whose nfe == exactly the canvas (i.e. one pass per mask, no expansion)
    canvas=int(cell.split("_c")[1])
    exact=sum(1 for x in nfe if x==canvas)
    print(f"{cell}: n={exp} at_cap({cap})={at_cap} ({100*at_cap/exp:.1f}%)  "
          f"cap_share_of_total_NFE={100*cap_mass/tot:.1f}%  nfe==canvas on {exact}/{exp} items")
