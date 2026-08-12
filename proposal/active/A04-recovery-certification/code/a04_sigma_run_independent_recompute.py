"""INDEPENDENT recompute of A04 Stage-B sigma_run from raw per-example shards.
Does NOT read any committed verdict JSON. Enforces shard-index set + exact counts.
CPU only."""
import json, sys, glob, math
from pathlib import Path
import numpy as np

ROOT = Path("/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory")
CB, MM = ROOT/"olmo2_closedbook_results", ROOT/"olmo2_mmlu_content_results"
EXPECT = {"triviaqa":17944, "popqa":14267, "nq_open":3610, "mmlu_content":14042}

def load_cb_task(tag, task):
    d = CB/(tag+"_nq") if task=="nq_open" else CB/tag
    fn = "nq_open" if task=="nq_open" else task
    files = sorted(glob.glob(str(d/f"per_example_{fn}_shard*of8.jsonl")))
    idx = sorted(int(f.split("shard")[-1].split("of")[0]) for f in files)
    assert idx == list(range(8)), f"{tag}/{task}: shard index set {idx} != 0..7"
    rec = {}
    for f in files:
        for line in open(f):
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            iid=r.get("item_id", r.get("idx"))
            assert iid not in rec, f"{tag}/{task}: duplicate item_id {iid}"
            v=r.get("em")
            assert v is not None and not (isinstance(v,float) and math.isnan(v)), f"nan in {tag}/{task}"
            rec[iid]=float(v)
    assert len(rec)==EXPECT[task], f"{tag}/{task}: n={len(rec)} != {EXPECT[task]}"
    return 100.0*np.mean(list(rec.values()))

def load_mmlu(tag):
    files = sorted(glob.glob(str(MM/tag/"per_example_mmlu_shard*of8.jsonl")))
    idx = sorted(int(f.split("shard")[-1].split("of")[0]) for f in files)
    assert idx == list(range(8)), f"{tag}/mmlu: shard index set {idx} != 0..7"
    rec={}
    for f in files:
        for line in open(f):
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            iid=r.get("item_id", r.get("idx"))
            assert iid not in rec, f"dup {iid}"
            assert r.get("nan") is not True, f"{tag}: nan:true row"
            cn=r.get("content_norm")
            assert isinstance(cn,dict), f"{tag}: content_norm not nested dict"
            v=cn.get("correct")
            assert isinstance(v,bool), f"{tag}: content_norm.correct not bool: {v!r}"
            assert v is not None, f"{tag}: content_norm.correct missing (PRE-FIX loader regime?)"
            rec[iid]=float(v)
    assert len(rec)==EXPECT["mmlu_content"], f"{tag}/mmlu n={len(rec)}"
    return 100.0*np.mean(list(rec.values()))

seeds=[101,102,103]
tags={s:f"A04_1B_stageB_keep12_seed{s}_step5000" for s in seeds}
DELTA={"triviaqa":4.043134195274186,"popqa":1.3205298941613512,
       "mmlu_content":1.0238926078906136,"nq_open":0.9695290858725762}
DECISION={"triviaqa","popqa","mmlu_content"}
class chi2:  # df=2 closed form: CDF=1-exp(-x/2) => ppf(p)=-2ln(1-p). scipy absent on .73.
    @staticmethod
    def ppf(p, df):
        assert df == 2, "closed form only valid at df=2"
        return -2.0 * math.log(1.0 - p)
t_05_df2 = 2.9199855803537124   # one-sided t_.95, df=2

out={"scope":"INDEPENDENT recompute of Stage-B sigma_run from raw shards (no verdict JSON read)",
     "seeds":seeds,"S":3,"df":2,"t_0.05_df2":t_05_df2,"per_axis":{}}
print(f"{'axis':<14}{'s101':>9}{'s102':>9}{'s103':>9}{'sd_run':>9}{'bound3':>9}{'Delta':>9}  fire?")
nfire=0; nfire_hi=0
for axis in ["triviaqa","popqa","mmlu_content","nq_open"]:
    if axis=="mmlu_content":
        m=[load_mmlu(tags[s]) for s in seeds]
    else:
        m=[load_cb_task(tags[s],axis) for s in seeds]
    sd=float(np.std(m,ddof=1)); df=2
    bound=t_05_df2*sd/math.sqrt(3)
    lo=sd*math.sqrt(df/chi2.ppf(0.975,df)); hi=sd*math.sqrt(df/chi2.ppf(0.025,df))
    bhi=t_05_df2*hi/math.sqrt(3)
    fire = bound>DELTA[axis]
    if axis in DECISION:
        nfire+=fire; nfire_hi+= (bhi>DELTA[axis])
    print(f"{axis:<14}{m[0]:>9.4f}{m[1]:>9.4f}{m[2]:>9.4f}{sd:>9.4f}{bound:>9.4f}{DELTA[axis]:>9.4f}  {fire}")
    out["per_axis"][axis]={"means_pct":m,"sd_run_pp":sd,"bound_S3_pp":bound,
        "delta_pp":DELTA[axis],"sigma_chi2_95ci_pp":[lo,hi],
        "bound_at_sigma_ci_hi_pp":bhi,"exceeds_delta":bool(fire),
        "would_exceed_at_sigma_ci_hi":bool(bhi>DELTA[axis]),
        "decision_weight":axis in DECISION}
out["n_decision_axes_exceeding"]=int(nfire)
out["n_decision_axes_exceeding_at_sigma_ci_hi"]=int(nfire_hi)
out["verdict"]="K2_DOES_NOT_FIRE" if nfire<2 else "K2_FIRES"
print(f"\ndecision axes exceeding Delta: {nfire}/3  -> {out['verdict']}")
print(f"at chi2 upper of sigma: {nfire_hi}/3 would exceed (rule needs >=2)")
json.dump(out,open(sys.argv[1],"w"),indent=2)
print("wrote",sys.argv[1])
