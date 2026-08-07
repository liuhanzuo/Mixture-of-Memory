"""Provenance: reproduce the PIN protocol exactly -- 12K prefix subsample (as E17.npy),
500-query target, and both ICC definitions -- to identify which quantity the pin reported."""
import sys, json, collections, numpy as np
sys.path.insert(0,'.')
import icc_lib
for TAG in ['i17','i40','b80','b17','i80']:
    E,traj,env,step,model=icc_lib.load(TAG)
    for LIMIT in [12000, len(E)]:
        Es=E[:LIMIT]; ts=traj[:LIMIT]; es=env[:LIMIT]
        cnt=collections.Counter(ts.tolist()); keep=np.array([cnt[t]>=2 for t in ts])
        emb=icc_lib.icc_multivar(Es[keep],ts[keep])
        # score version with 500-query target drawn from within the subsample
        rng=np.random.default_rng(0)
        tid=rng.choice(LIMIT,500,replace=False)
        pm=np.ones(LIMIT,bool); pm[tid]=False; P=np.where(pm)[0]
        sc=(Es[P]@Es[tid].T).max(1).astype(np.float64)
        c2=collections.Counter(ts[P].tolist()); k2=np.array([c2[t]>=2 for t in ts[P]])
        s1=icc_lib.icc_oneway(sc[k2],ts[P][k2])['icc']
        print(f'{TAG:4s} N={LIMIT:6d}  ICC_embed={emb:.4f}  ICC_score={s1:.4f}')
    print()
