import json, collections, numpy as np
rows=[json.loads(l) for l in open('pool.jsonl')]
N=len(rows)
traj=np.array([r['traj'] for r in rows]); env=np.array([r['env'] for r in rows])
cnt=collections.Counter(traj.tolist()); m=np.array(sorted(cnt.values()))
K=int(round(0.05*N))
print(f'POOL N={N} trajs={len(cnt)} envs={len(set(env))} K(5%)={K}')
print(f'mean m={m.mean():.4f}  median={np.median(m)}  max={m.max()}')
print(f'm* = sum m^2/sum m = {(m**2).sum()/m.sum():.4f}   <-- variance-weighted cluster size')
# --- EXACT expected distinct trajectories under SRS without replacement, K of N
# E[distinct] = sum_g (1 - C(N-m_g,K)/C(N,K)) ; use log-gamma for stability
from math import lgamma
def logC(n,k):
    if k<0 or k>n: return -np.inf
    return lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)
base=logC(N,K)
mv=np.array(sorted(cnt.values()))
uniq,uc=np.unique(mv,return_counts=True)
Edist=0.0
for mg,c in zip(uniq,uc):
    Edist += c*(1-np.exp(logC(N-mg,K)-base))
print(f'\nRANDOM  E[distinct traj] (exact hypergeometric) = {Edist:.1f}  ({Edist/K*100:.1f}% of K)')
print(f'   pin says 3764 (75.3%)   -> delta {Edist-3764:+.1f}')
# expected max from one traj under random: max_g Hypergeom(m_g); approx via simulation
rng=np.random.default_rng(0)
mx=[]; d3=[]; dst=[]
envs_sorted=sorted(set(env.tolist()))
for s in range(20):
    r=rng.choice(N,K,replace=False)
    c=collections.Counter(traj[r].tolist()); mx.append(max(c.values())); dst.append(len(c))
    ec=collections.Counter(env[r].tolist()); d3.append(sum(v for _,v in ec.most_common(3))/K*100)
print(f'RANDOM  simulated distinct={np.mean(dst):.1f}+-{np.std(dst):.1f} maxONEtraj={np.mean(mx):.2f}+-{np.std(mx):.2f} (pin 9) top3share={np.mean(d3):.2f}%+-{np.std(d3):.2f} (pin 45.0)')
# --- stratified random: equal per-env cap
byenv=collections.defaultdict(list)
for i,e in enumerate(env): byenv[e].append(i)
cE=int(np.ceil(K/len(byenv)))
st_d=[];st_mx=[];st_3=[];st_n=[]
for s in range(20):
    rng2=np.random.default_rng(100+s); sel=[]
    for e in envs_sorted:
        idx=byenv[e]; sel+=list(rng2.choice(idx,min(len(idx),cE),replace=False))
    # if short of K (small envs), top up randomly from remainder
    sel=set(sel)
    if len(sel)<K:
        rest=[i for i in range(N) if i not in sel]
        sel|=set(rng2.choice(rest,K-len(sel),replace=False).tolist())
    sel=np.array(sorted(sel))[:K]
    c=collections.Counter(traj[sel].tolist()); st_d.append(len(c)); st_mx.append(max(c.values()))
    ec=collections.Counter(env[sel].tolist()); st_3.append(sum(v for _,v in ec.most_common(3))/K*100); st_n.append(len(ec))
print(f'STRATIFIED distinct={np.mean(st_d):.1f}+-{np.std(st_d):.1f} ({np.mean(st_d)/K*100:.1f}%, pin 4018/80.4%) '
      f'maxONE={np.mean(st_mx):.2f} (pin 11) top3={np.mean(st_3):.2f}% (pin 17.0) envs={np.mean(st_n):.1f}')
print(f'   per-env cap cE=ceil({K}/{len(byenv)})={cE}')
ec_pool=collections.Counter(env.tolist())
print(f'\nPOOL top-3 env share = {sum(v for _,v in ec_pool.most_common(3))/N*100:.2f}%  (random selector should match this)')
print('   top5 pool envs:',[(k,round(v/N*100,1)) for k,v in ec_pool.most_common(5)])
