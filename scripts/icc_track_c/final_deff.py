"""Definitive design effect + n_eff, 10 target-set seeds, bootstrap CI."""
import sys, json, collections, numpy as np
sys.path.insert(0,'.')
import icc_lib
TAG=sys.argv[1]; NS=int(sys.argv[2]) if len(sys.argv)>2 else 10
E,traj,env,step,model=icc_lib.load(TAG); N=len(E); K=int(round(0.05*N))
acc=collections.defaultdict(list)
for seed in range(NS):
    rng=np.random.default_rng(seed)
    tid=rng.choice(N,1000,replace=False); pm=np.ones(N,bool); pm[tid]=False; P=np.where(pm)[0]
    score=(E[P]@E[tid].T).max(1).astype(np.float64)
    tp=traj[P]; ep=env[P]
    cnt=collections.Counter(tp.tolist()); keep=np.array([cnt[t]>=2 for t in tp])
    icc=icc_lib.icc_oneway(score[keep],tp[keep])['icc']
    nest=icc_lib.icc_nested(score[keep],ep[keep],tp[keep])
    for name,sel in [('RDS+top-k',P[np.argsort(-score)][:K]),
                     ('random',rng.choice(P,K,replace=False))]:
        c=collections.Counter(traj[sel].tolist())
        cv=np.array(list(c.values()),float)
        m_mean=K/len(c); m_kish=(cv**2).sum()/K
        d_naive=1+(m_mean-1)*icc              # formula as literally requested in track spec
        d_kish =1+(m_kish-1)*icc              # Kish/variance-weighted cluster size
        d_exact=K/ (cv/(1+(cv-1)*icc)).sum()  # per-cluster exact
        h=icc_lib.deff_hierarchical(env[sel],traj[sel],nest['icc_env'],nest['icc_traj_within'])
        for k,v in [('icc',icc),('icc_env',nest['icc_env']),('icc_trajw',nest['icc_traj_within']),
                    ('distinct',len(c)),('m_mean',m_mean),('m_kish',m_kish),
                    ('deff_naive',d_naive),('deff_kish',d_kish),('deff_exact',d_exact),
                    ('neff_naive',K/d_naive),('neff_kish',K/d_kish),('neff_exact',K/d_exact),
                    ('deff_hier',h['deff']),('neff_hier',h['neff']),('M_kish',h['M_kish'])]:
            acc[f'{name}|{k}'].append(v)
print(f'===== {TAG} ({model.split("/")[-1]}) K={K} pool_N={N} seeds={NS} =====')
out={'tag':TAG,'model':model,'K':K,'N':N,'seeds':NS}
for arm in ['RDS+top-k','random']:
    print(f'-- {arm} --')
    for k in ['icc','icc_env','icc_trajw','distinct','m_mean','m_kish','deff_naive','deff_kish',
              'deff_exact','neff_naive','neff_kish','neff_exact','deff_hier','neff_hier','M_kish']:
        v=np.array(acc[f'{arm}|{k}'],float)
        lo,hi=np.percentile(v,[2.5,97.5])
        print(f'   {k:12s} {v.mean():10.3f} +- {v.std():8.3f}  [{lo:.3f},{hi:.3f}]')
        out[f'{arm}|{k}']=dict(mean=float(v.mean()),sd=float(v.std()),lo=float(lo),hi=float(hi),
                               vals=list(map(float,v)))
json.dump(out,open(f'deff_{TAG}.json','w'),indent=1)
