"""ICC + design-effect library. Two ICCs are NOT the same thing:
 (a) ICC on the SELECTION SCORE (scalar) -- what the pin table claims, and what
     actually governs the design effect of a score-ranked selection.
 (b) ICC on the EMBEDDING coordinates (multivariate) -- what the prior script computed.
Also: a one-way (traj-only) ANOVA absorbs BENCHMARK-level clustering into the
between-traj term, inflating ICC. We report a nested env/traj decomposition too."""
import numpy as np, collections, json, glob

def load(tag, root='.'):
    Es=[]; metas=[]
    for g in range(8):
        f=f'{root}/emb/{tag}_shard{g}of8.npy'
        Es.append(np.load(f)); metas.append(json.load(open(f+'.meta.json')))
    # shards are strided [g::8]; reassemble original pool order
    N=sum(len(e) for e in Es)
    dim=Es[0].shape[1]
    E=np.zeros((N,dim),dtype=np.float32)
    traj=np.empty(N,dtype=object); env=np.empty(N,dtype=object); step=np.zeros(N,int)
    for g in range(8):
        idx=np.arange(g,N,8)
        assert len(idx)==len(Es[g]), (g,len(idx),len(Es[g]))
        E[idx]=Es[g]
        for j,r in enumerate(metas[g]['rows']):
            traj[idx[j]]=r['traj']; env[idx[j]]=r['env']; step[idx[j]]=r['step']
    return E, traj, env, step, metas[0]['model']

def icc_oneway(x, g):
    """One-way random-effects ICC(1) on a SCALAR x with group labels g.
    ICC = (MSB-MSW)/(MSB+(m0-1)MSW), m0 = (N - sum n_i^2/N)/(k-1)."""
    x=np.asarray(x,dtype=np.float64)
    uniq,inv=np.unique(g,return_inverse=True)
    k=len(uniq); N=len(x)
    gs=np.bincount(inv,minlength=k).astype(np.float64)
    gsum=np.bincount(inv,weights=x,minlength=k)
    gmean=gsum/gs
    gm=x.mean()
    SSB=(gs*(gmean-gm)**2).sum()
    SSW=((x-gmean[inv])**2).sum()
    MSB=SSB/(k-1); MSW=SSW/(N-k)
    m0=(N-(gs**2).sum()/N)/(k-1)
    icc=(MSB-MSW)/(MSB+(m0-1)*MSW)
    return dict(icc=float(icc), k=int(k), N=int(N), MSB=float(MSB), MSW=float(MSW),
                m0=float(m0), mean_m=float(N/k))

def icc_multivar(E, g):
    """Multivariate one-way ICC: variance components summed over dims (what prior code did)."""
    uniq,inv=np.unique(g,return_inverse=True)
    k=len(uniq); N,dim=E.shape
    gs=np.bincount(inv,minlength=k).astype(np.float64)
    gsum=np.zeros((k,dim)); np.add.at(gsum,inv,E.astype(np.float64))
    gmean=gsum/gs[:,None]
    gm=E.astype(np.float64).mean(0)
    SSB=(gs[:,None]*(gmean-gm)**2).sum()
    SSW=((E-gmean[inv])**2).sum()
    MSB=SSB/(k-1); MSW=SSW/(N-k)
    m0=(N-(gs**2).sum()/N)/(k-1)
    return float((MSB-MSW)/(MSB+(m0-1)*MSW))

def icc_nested(x, env, traj):
    """Nested random effects: x = mu + a_env + b_traj(env) + e.
    Returns ICC_env (share of var from benchmark) and ICC_traj_within_env
    (share from trajectory AFTER removing benchmark). Method-of-moments on
    unbalanced data via sequential sums of squares."""
    x=np.asarray(x,np.float64); N=len(x)
    ue,ie=np.unique(env,return_inverse=True); a=len(ue)
    ut,it=np.unique(traj,return_inverse=True); b=len(ut)
    gm=x.mean()
    ne=np.bincount(ie,minlength=a).astype(float); me=np.bincount(ie,weights=x,minlength=a)/ne
    nt=np.bincount(it,minlength=b).astype(float); mt=np.bincount(it,weights=x,minlength=b)/nt
    # traj -> its env
    t2e=np.zeros(b,int); t2e[it]=ie
    SS_env=(ne*(me-gm)**2).sum()
    SS_traj=(nt*(mt-me[t2e])**2).sum()
    SS_err=((x-mt[it])**2).sum()
    df_e=a-1; df_t=b-a; df_r=N-b
    MS_e=SS_env/df_e; MS_t=SS_traj/df_t; MS_r=SS_err/df_r
    # coefficients for unbalanced nested EMS
    # E[MS_r]=s2r ; E[MS_t]=s2r + c_t*s2t ; E[MS_e]=s2r + c_te*s2t + c_e*s2e
    sum_nt2_by_e=np.bincount(t2e,weights=nt**2,minlength=a)
    c_t=(N-(sum_nt2_by_e/ne).sum())/df_t
    c_te=((sum_nt2_by_e/ne).sum()-(nt**2).sum()/N)/df_e
    c_e=(N-(ne**2).sum()/N)/df_e
    s2r=MS_r
    s2t=max(0.0,(MS_t-s2r)/c_t)
    s2e=max(0.0,(MS_e-s2r-c_te*s2t)/c_e)
    tot=s2e+s2t+s2r
    return dict(icc_env=s2e/tot, icc_traj_within=s2t/tot,
                icc_traj_oneway_like=(s2e+s2t)/tot,
                s2_env=s2e, s2_traj=s2t, s2_resid=s2r)

def design_effect(counts, icc):
    """counts: per-cluster selected counts. DEFF = 1+(m-1)*icc with m = variance-weighted
    (Kish) mean cluster size sum(n^2)/sum(n); n_eff = sum n/DEFF. Also exact per-cluster
    n_eff = sum_g n_g/(1+(n_g-1)icc)."""
    c=np.asarray(list(counts),dtype=np.float64)
    n=c.sum(); m_kish=(c**2).sum()/n
    deff_kish=1+(m_kish-1)*icc
    neff_exact=float((c/(1+(c-1)*icc)).sum())
    return dict(n=int(n), n_clusters=len(c), m_mean=float(n/len(c)), m_kish=float(m_kish),
                deff_kish=float(deff_kish), neff_kish=float(n/deff_kish),
                neff_exact=neff_exact, deff_exact=float(n/neff_exact), max_from_one=int(c.max()))

def deff_hierarchical(env_sel, traj_sel, icc_env, icc_traj, ):
    """Exact design effect for a NESTED (env > traj > sample) selection.
    x_ijk = mu + a_i + b_ij + e_ijk.  For the sample mean over the selection:
      Var = s2a*sum M_i^2/n^2 + s2b*sum m_ij^2/n^2 + s2e/n
      Var_SRS = (s2a+s2b+s2e)/n
      => DEFF = icc_env*M_kish + icc_traj*m_kish + icc_resid
    where M_kish = sum M_i^2/n (env Kish size), m_kish = sum m_ij^2/n (traj Kish size).
    Reduces to 1+(m-1)icc when there is a single level."""
    import collections as _c, numpy as _np
    n=len(env_sel)
    Me=_np.array(list(_c.Counter(list(env_sel)).values()),dtype=float)
    mt=_np.array(list(_c.Counter(list(traj_sel)).values()),dtype=float)
    M_kish=(Me**2).sum()/n; m_kish=(mt**2).sum()/n
    icc_resid=1.0-icc_env-icc_traj
    deff=icc_env*M_kish + icc_traj*m_kish + icc_resid
    return dict(n=int(n), M_kish=float(M_kish), m_kish=float(m_kish),
                deff=float(deff), neff=float(n/deff),
                contrib_env=float(icc_env*M_kish), contrib_traj=float(icc_traj*m_kish),
                contrib_resid=float(icc_resid))
