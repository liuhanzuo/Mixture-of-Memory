"""Multi-seed: is ICC(score) / concentration stable across target-set draws?"""
import sys, json, collections, numpy as np
sys.path.insert(0,'/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/icc_track_c')
import icc_lib
TAG=sys.argv[1]; NSEED=int(sys.argv[2]) if len(sys.argv)>2 else 5
E,traj,env,step,model=icc_lib.load(TAG); N=len(E); K=int(round(0.05*N))
out=collections.defaultdict(list)
for seed in range(NSEED):
    rng=np.random.default_rng(seed)
    tid=rng.choice(N,1000,replace=False)
    pm=np.ones(N,bool); pm[tid]=False; P=np.where(pm)[0]
    score=(E[P]@E[tid].T).max(1).astype(np.float64)
    tp=traj[P]; ep=env[P]
    cnt=collections.Counter(tp.tolist()); keep=np.array([cnt[t]>=2 for t in tp])
    r=icc_lib.icc_oneway(score[keep],tp[keep]); out['icc_score'].append(r['icc'])
    nest=icc_lib.icc_nested(score[keep],ep[keep],tp[keep])
    out['icc_env'].append(nest['icc_env']); out['icc_traj_within'].append(nest['icc_traj_within'])
    out['icc_embed'].append(icc_lib.icc_multivar(E[P][keep],tp[keep]))
    sel=P[np.argsort(-score)][:K]
    c=collections.Counter(traj[sel].tolist()); ec=collections.Counter(env[sel].tolist())
    out['distinct'].append(len(c)); out['maxone'].append(max(c.values()))
    out['envs'].append(len(ec)); out['top3'].append(sum(v for _,v in ec.most_common(3))/K*100)
    d=icc_lib.design_effect(c.values(), r['icc']); out['deff'].append(d['deff_exact']); out['neff'].append(d['neff_exact'])
    h=icc_lib.deff_hierarchical(env[sel],traj[sel],nest['icc_env'],nest['icc_traj_within'])
    out['deff_h'].append(h['deff']); out['neff_h'].append(h['neff'])
    out['Mkish'].append(h['M_kish']); out['mkish'].append(h['m_kish'])
print(f'--- {TAG} ({model.split("/")[-1]}) {NSEED} target-set seeds, K={K} ---')
for k,v in out.items():
    v=np.array(v,dtype=float); print(f'  {k:16s} {v.mean():10.4f} +- {v.std():.4f}   {np.round(v,3).tolist()}')
json.dump({k:list(map(float,v)) for k,v in out.items()}|{'tag':TAG,'model':model,'K':K},
          open(f'seeds_{TAG}.json','w'),indent=1)
