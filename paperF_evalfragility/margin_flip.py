import json,os,sys
CORE6=["hellaswag","arc_challenge","arc_easy","piqa","winogrande","openbookqa"]
R=os.environ.get("RES","olmo2_downstream_results")
def load(d,t):
    p=os.path.join(R,d,f"per_example_{t}.jsonl")
    if not os.path.exists(p): return None
    o={}
    for line in open(p):
        r=json.loads(line); o[r["item_id"]]=r
    return o
def marg(r):
    s=sorted(r["option_scores"].values(),reverse=True)
    return s[0]-s[1] if len(s)>1 else 1e9
BUCK=[(0,0.01),(0.01,0.05),(0.05,0.1),(0.1,0.25),(0.25,0.5),(0.5,1.0),(1.0,1e9)]
def bidx(m):
    for i,(a,b) in enumerate(BUCK):
        if a<=m<b: return i
    return len(BUCK)-1
def analyze(dA,dB,label):
    tot=[0]*len(BUCK); fl=[0]*len(BUCK); nf=0; n=0
    for t in CORE6:
        A=load(dA,t); B=load(dB,t)
        if A is None or B is None: print(f"  MISSING {t} in {dA} or {dB}"); return None
        for k in A:
            if k not in B: continue
            m=min(marg(A[k]),marg(B[k])); i=bidx(m)
            tot[i]+=1; n+=1
            if A[k]["pred_letter"]!=B[k]["pred_letter"]: fl[i]+=1; nf+=1
    print(f"\n### {label}   n={n}  total pred-flips={nf}  ({nf/n*100:.3f}%)")
    print(f"  {'margin bucket':<16}{'n_items':>9}{'flips':>7}{'P(flip|bucket)':>16}")
    for i,(a,b) in enumerate(BUCK):
        lab=f"[{a},{b})" if b<1e9 else f"[{a},inf)"
        if tot[i]: print(f"  {lab:<16}{tot[i]:>9}{fl[i]:>7}{fl[i]/tot[i]*100:>15.3f}%")
    return tot,fl,n,nf
if __name__=="__main__":
    pairs=json.loads(sys.argv[1])
    res={}
    for lab,a,b in pairs:
        r=analyze(a,b,lab)
        if r: res[lab]=r
    # decomposition: apply a common P(flip|bucket) to each rung's margin distribution
    print("\n=== DECOMPOSITION: predicted flips using a SHARED conditional flip curve ===")
    if len(res)>1:
        tots=[0]*len(BUCK); fls=[0]*len(BUCK)
        for lab,(tot,fl,n,nf) in res.items():
            for i in range(len(BUCK)): tots[i]+=tot[i]; fls[i]+=fl[i]
        pooled=[fls[i]/tots[i] if tots[i] else 0 for i in range(len(BUCK))]
        print("  pooled P(flip|bucket) =",[f"{p*100:.3f}%" for p in pooled])
        for lab,(tot,fl,n,nf) in res.items():
            pred=sum(tot[i]*pooled[i] for i in range(len(BUCK)))
            print(f"  {lab:<22} observed={nf:>5}  predicted_from_margins={pred:>8.1f}  ratio={nf/pred if pred else 0:.3f}")
