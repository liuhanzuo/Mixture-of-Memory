import sys, json, collections, numpy as np
sys.path.insert(0,'/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/icc_track_c')
import icc_lib
TAG=sys.argv[1]
E,traj,env,step,model=icc_lib.load(TAG)
N=len(E)
rng=np.random.default_rng(0)
res=dict(tag=TAG,model=model,N=int(N),dim=int(E.shape[1]))

# ---------- RDS+ selection score ----------
# target set: held-out queries spread over all envs (RDS+ = cosine to target, max over queries)
NT=1000
tid=rng.choice(N,NT,replace=False)
pm=np.ones(N,bool); pm[tid]=False
P=np.where(pm)[0]
# chunked matmul (100K x 1000 is fine)
S=E[P]@E[tid].T
score=S.max(1).astype(np.float64)
K=int(round(0.05*N))
res['K']=K; res['n_target']=NT

# ---------- ICC on the SELECTION SCORE (the pin's stated quantity) ----------
tp=traj[P]; ep=env[P]
cnt=collections.Counter(tp.tolist())
keep=np.array([cnt[t]>=2 for t in tp])       # ICC needs >=2 per cluster
r1=icc_lib.icc_oneway(score[keep],tp[keep])
res['icc_score_oneway']=r1
nest=icc_lib.icc_nested(score[keep],ep[keep],tp[keep])
res['icc_score_nested']=nest
# ---------- ICC on embedding coords (what the prior script measured) ----------
res['icc_embed_multivar']=icc_lib.icc_multivar(E[P][keep],tp[keep])

# ---------- concentration at 5% budget ----------
def conc(sel,name):
    c=collections.Counter(traj[sel].tolist()); ec=collections.Counter(env[sel].tolist())
    top3=sum(v for _,v in ec.most_common(3))/len(sel)*100
    d=icc_lib.design_effect(c.values(), r1['icc'])
    out=dict(name=name,K=len(sel),distinct=len(c),distinct_pct=len(c)/len(sel)*100,
             max_from_one=int(max(c.values())),envs=len(ec),top3=top3,
             deff=d['deff_exact'],neff=d['neff_exact'],m_kish=d['m_kish'])
    print(f"{name:22s} K={len(sel)} distinct={len(c):5d} ({out['distinct_pct']:5.1f}%) "
          f"maxONE={out['max_from_one']:3d} envs={len(ec):2d}/19 top3={top3:5.1f}% "
          f"DEFF={d['deff_exact']:.2f} n_eff={d['neff_exact']:.0f}")
    return out
arms=[]
ordr=P[np.argsort(-score)]
arms.append(conc(ordr[:K],'RDS+ global top-k'))
# TRUE RDS+ round-robin over target queries (per arXiv:2503.01807)
rank=np.argsort(-S,axis=0)
ptr=np.zeros(NT,int); taken=set(); o=[]; q=0
while len(o)<K:
    c=q%NT
    while ptr[c]<len(P) and rank[ptr[c],c] in taken: ptr[c]+=1
    if ptr[c]<len(P):
        li=rank[ptr[c],c]; taken.add(li); o.append(P[li]); ptr[c]+=1
    q+=1
arms.append(conc(np.array(o),'RDS+ round-robin'))
arms.append(conc(rng.choice(P,K,replace=False),'random'))
byenv=collections.defaultdict(list)
for i in P: byenv[env[i]].append(i)
cE=int(np.ceil(K/len(byenv))); st=[]
for e in sorted(byenv): st+=list(rng.choice(byenv[e],min(len(byenv[e]),cE),replace=False))
st=np.array(st); 
if len(st)<K:
    rest=np.setdiff1d(P,st); st=np.concatenate([st,rng.choice(rest,K-len(st),replace=False)])
arms.append(conc(st[:K],'stratified-random'))
res['arms']=arms
print(f"\nICC(score,one-way traj) = {r1['icc']:.4f}  [k={r1['k']} n={r1['N']} m0={r1['m0']:.2f}]")
print(f"ICC(score,nested): env={nest['icc_env']:.4f} traj|env={nest['icc_traj_within']:.4f} sum={nest['icc_traj_oneway_like']:.4f}")
print(f"ICC(embed,multivar one-way) = {res['icc_embed_multivar']:.4f}")
json.dump(res,open(f'res_{TAG}.json','w'),indent=1,default=float)
