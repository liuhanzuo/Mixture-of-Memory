import json,os,math
CORE6=["hellaswag","arc_challenge","arc_easy","piqa","winogrande","openbookqa"]
PAIRS=[
 ("base_full32","olmo2_downstream_results/7B_full32_base_wzc1_v2","/tmp/z6/7B_base_full_bs8",0.703649),
 ("ShortGPT-16","olmo2_downstream_results/7B_shortgpt16_step200000_wzc1","/tmp/z6/7B_shortgpt16_step200000_v2",0.622473),
 ("keep14@200k","olmo2_downstream_results/7B_keep14_step200000_wzc1_v2","/tmp/z6/7B_keep14_step200000_v2",0.595324),
 ("keep10@83.5k","olmo2_downstream_results/7B_keep10_step83500_wzc1","/tmp/z6/7B_keep10_step83500_v2",0.529988),
 ("keep8@121k","olmo2_downstream_results/7B_keep8_step121000_wzc1","/tmp/z6/7B_keep8_step121000_v2",0.523284),
]
def load(d,t):
    p=os.path.join(d,f"per_example_{t}.jsonl")
    if not os.path.exists(p): return None
    return {json.loads(l)["item_id"]:json.loads(l) for l in open(p)}
def pct(v,q):
    v=sorted(v); 
    if not v: return 0
    i=min(len(v)-1,int(q*len(v)))
    return v[i]
print("PERTURBATION MAGNITUDE of the *margin* (top1-top2 gap) under the L20A->H20 swap")
print("delta = |margin_A - margin_B|, in nats. This is the numerical noise the argmax competes against.")
print(f"\n{'rung':<14}{'core6':>9}{'n':>7}{'med|dm|':>11}{'p90|dm|':>11}{'p99|dm|':>11}{'mean|dm|':>11}")
rows={}
for lab,dA,dB,c6 in PAIRS:
    dm=[]; ok=True
    for t in CORE6:
        A=load(dA,t); B=load(dB,t)
        if A is None or B is None: ok=False; break
        for k in A:
            if k not in B: continue
            sa=sorted(A[k]["option_scores"].values(),reverse=True)
            sb=sorted(B[k]["option_scores"].values(),reverse=True)
            if len(sa)<2 or len(sb)<2: continue
            dm.append(abs((sa[0]-sa[1])-(sb[0]-sb[1])))
    if not ok: continue
    rows[lab]=(c6,dm)
    print(f"{lab:<14}{c6:>9.5f}{len(dm):>7}{pct(dm,.5):>11.5f}{pct(dm,.9):>11.5f}{pct(dm,.99):>11.5f}{sum(dm)/len(dm):>11.5f}")
print("\nSame, restricted to items whose margin is ALREADY large (>1 nat) -> pure noise-magnitude probe,")
print("free of any 'damaged models have more near-ties' confound:")
print(f"{'rung':<14}{'n(>1nat)':>10}{'med|dm|':>11}{'p90|dm|':>11}{'p99|dm|':>11}")
big={}
for lab,dA,dB,c6 in PAIRS:
    dm=[]
    for t in CORE6:
        A=load(dA,t); B=load(dB,t)
        if A is None or B is None: break
        for k in A:
            if k not in B: continue
            sa=sorted(A[k]["option_scores"].values(),reverse=True)
            sb=sorted(B[k]["option_scores"].values(),reverse=True)
            if len(sa)<2: continue
            if min(sa[0]-sa[1],sb[0]-sb[1])>1.0:
                dm.append(abs((sa[0]-sa[1])-(sb[0]-sb[1])))
    if dm:
        big[lab]=(c6,pct(dm,.5))
        print(f"{lab:<14}{len(dm):>10}{pct(dm,.5):>11.5f}{pct(dm,.9):>11.5f}{pct(dm,.99):>11.5f}")
def spear(x,y):
    def rk(v):
        s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for j,i in enumerate(s): r[i]=j+1
        return r
    rx,ry=rk(x),rk(y); nn=len(x); mx=sum(rx)/nn; my=sum(ry)/nn
    return sum((a-mx)*(b-my) for a,b in zip(rx,ry))/math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
c=[big[l][0] for l in big]; m=[big[l][1] for l in big]
print(f"\nSpearman(core6_acc, median |delta margin| on >1nat items) = {spear(c,m):.4f}")
print("  -> negative means MORE damaged = LARGER numerical perturbation (second, independent mechanism)")
