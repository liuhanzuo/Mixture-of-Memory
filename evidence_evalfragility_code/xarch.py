import json,os,math
CORE6=["hellaswag","arc_challenge","arc_easy","piqa","winogrande","openbookqa"]
# (label, wzc1 dir [L20A, local], zwfy6 dir [H20, /tmp/z6])
PAIRS=[
 ("base_full32","olmo2_downstream_results/7B_full32_base_wzc1_v2","/tmp/z6/7B_base_full_bs8",0.703649),
 ("ShortGPT-16","olmo2_downstream_results/7B_shortgpt16_step200000_wzc1","/tmp/z6/7B_shortgpt16_step200000_v2",0.622473),
 ("keep14@200k","olmo2_downstream_results/7B_keep14_step200000_wzc1_v2","/tmp/z6/7B_keep14_step200000_v2",0.595324),
 ("keep10@83.5k","olmo2_downstream_results/7B_keep10_step83500_wzc1","/tmp/z6/7B_keep10_step83500_v2",0.529988),
 ("keep8@121k","olmo2_downstream_results/7B_keep8_step121000_wzc1","/tmp/z6/7B_keep8_step121000_v2",0.523284),
]
BUCK=[(0,0.01),(0.01,0.05),(0.05,0.1),(0.1,0.25),(0.25,0.5),(0.5,1.0),(1.0,1e9)]
def bi(m):
    for i,(a,b) in enumerate(BUCK):
        if a<=m<b: return i
    return len(BUCK)-1
def load(d,t):
    p=os.path.join(d,f"per_example_{t}.jsonl")
    if not os.path.exists(p): return None
    o={}
    for l in open(p):
        r=json.loads(l); o[r["item_id"]]=r
    return o
def marg(r):
    s=sorted(r["option_scores"].values(),reverse=True)
    return s[0]-s[1] if len(s)>1 else 1e9
res={}
print(f"{'rung':<14}{'core6':>9}{'n':>7}{'flips':>7}{'flip%':>8}{'nt<0.1':>8}{'nt%':>8}")
for lab,dA,dB,c6 in PAIRS:
    tot=[0]*len(BUCK); fl=[0]*len(BUCK); n=0; nf=0; nt=0
    ok=True
    for t in CORE6:
        A=load(dA,t); B=load(dB,t)
        if A is None or B is None: print(f"  !! MISSING {t}: {dA if A is None else dB}"); ok=False; break
        for k in A:
            if k not in B: continue
            m=min(marg(A[k]),marg(B[k])); i=bi(m); tot[i]+=1; n+=1
            if m<0.1: nt+=1
            if A[k]["pred_letter"]!=B[k]["pred_letter"]: fl[i]+=1; nf+=1
    if not ok: continue
    res[lab]=(tot,fl,n,nf,c6,nt)
    print(f"{lab:<14}{c6:>9.5f}{n:>7}{nf:>7}{nf/n*100:>7.3f}%{nt:>8}{nt/n*100:>7.3f}%")
print("\n=== per-bucket P(flip | margin bucket), cross-arch L20A vs H20 ===")
hdr=f"{'rung':<14}"+"".join(f"{('['+str(a)+','+(str(b) if b<1e9 else 'inf')+')')[:11]:>12}" for a,b in BUCK)
print(hdr)
for lab,(tot,fl,n,nf,c6,nt) in res.items():
    print(f"{lab:<14}"+"".join(f"{(fl[i]/tot[i]*100 if tot[i] else 0):>11.2f}%" for i in range(len(BUCK))))
print(f"{'n_items:':<14}"+"".join("" for _ in BUCK))
for lab,(tot,fl,n,nf,c6,nt) in res.items():
    print(f"  n[{lab:<12}]"+"".join(f"{tot[i]:>12}" for i in range(len(BUCK))))
print("\n=== MEDIATION: does one SHARED conditional flip curve reproduce the damage ordering? ===")
tots=[0]*len(BUCK); fls=[0]*len(BUCK)
for lab,(tot,fl,n,nf,c6,nt) in res.items():
    for i in range(len(BUCK)): tots[i]+=tot[i]; fls[i]+=fl[i]
pooled=[fls[i]/tots[i] if tots[i] else 0 for i in range(len(BUCK))]
print("  pooled P(flip|bucket) =",[f"{p*100:.2f}%" for p in pooled])
print(f"\n  {'rung':<14}{'observed':>10}{'pred_margins':>14}{'ratio':>8}")
obs=[];pred=[]
for lab,(tot,fl,n,nf,c6,nt) in res.items():
    p=sum(tot[i]*pooled[i] for i in range(len(BUCK)))
    obs.append(nf); pred.append(p)
    print(f"  {lab:<14}{nf:>10}{p:>14.1f}{nf/p:>8.3f}")
def spear(x,y):
    def rk(v):
        s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for j,i in enumerate(s): r[i]=j+1
        return r
    rx,ry=rk(x),rk(y); nn=len(x); mx=sum(rx)/nn; my=sum(ry)/nn
    return sum((a-mx)*(b-my) for a,b in zip(rx,ry))/math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
print(f"\n  Spearman(observed_flips, predicted_from_margins) = {spear(obs,pred):.4f}")
c6s=[res[l][4] for l in res]
print(f"  Spearman(core6_acc, observed_flips)              = {spear(c6s,obs):.4f}")
print(f"  Spearman(core6_acc, predicted_from_margins)      = {spear(c6s,pred):.4f}")
