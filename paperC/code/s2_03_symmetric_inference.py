"""S2-03: give the CHANCE side of the flip claim the same inference the FLOOR side has.

Defect: the floor comparison carries ci95, boot_p, and exact McNemar; the chance
comparison is a bare point comparison. So "above chance but not above the floor"
pairs a TESTED claim with an UNTESTED one, and the paper's own headline rests on it.

chance = mean_i(1/n_opt_i) is, like the floor, a deterministic function of the item
set, so the matching test is the paired item bootstrap: resample items, recompute
BOTH acc and mean(1/n_opt) inside each resample, report CI and p on the difference.
The floor side is recomputed the same way here as a consistency check against the
already-published ci95/boot_p.
"""
import glob, json, os
import numpy as np

BASE = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
N_BOOT, SEED = 10000, 7
LET = "ABCDEFGHIJ"

CELLS = json.load(open("/tmp/heal_cells.json"))

def load(root, arm):
    sh = sorted(glob.glob(os.path.join(BASE, root, arm, "per_example_mmlu_pro_shard*of8.jsonl")))
    idx = {int(os.path.basename(s).split("_shard")[1].split("of")[0]) for s in sh}
    assert idx == set(range(8)), f"{arm}: MISSING shards {sorted(set(range(8))-idx)}"
    recs = []
    for s in sh:
        with open(s) as f:
            for ln in f:
                ln = ln.strip()
                if ln: recs.append(json.loads(ln))
    ids = [r["item_id"] for r in recs]
    assert len(set(ids)) == len(ids), f"{arm}: duplicate item_id"
    assert len(recs) == 12032, f"{arm}: n_scored={len(recs)} != 12032"
    assert sum(1 for r in recs if r.get("nan")) == 0, f"{arm}: nan present"
    recs.sort(key=lambda r: r["item_id"])
    return recs

out = []
for c in CELLS:
    recs = load(c["results_root"], c["arm_dir"])
    corr  = np.array([1.0 if r["letter"]["correct"] else 0.0 for r in recs])
    chance= np.array([1.0 / r["n_opt"] for r in recs])
    # floor = best-constant letter, a dataset property (global argmax, always-A)
    floor_vec = np.array([1.0 if r["gold_letter"] == "A" else 0.0 for r in recs])
    n = len(recs)
    acc, ch, fl = corr.mean(), chance.mean(), floor_vec.mean()
    rng = np.random.default_rng(SEED)
    d_ch = np.empty(N_BOOT); d_fl = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        d_ch[b] = corr[i].mean() - chance[i].mean()   # BOTH recomputed inside
        d_fl[b] = corr[i].mean() - floor_vec[i].mean()
    def summ(d, obs):
        lo, hi = np.percentile(d, [2.5, 97.5])
        # two-sided bootstrap p by the sign-symmetry convention used elsewhere
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        p = max(p, 1.0 / N_BOOT)
        return dict(delta_pp=100*obs, ci95_lo_pp=100*lo, ci95_hi_pp=100*hi,
                    half_width_pp=100*(hi-lo)/2, boot_p=float(p),
                    ci_excludes_zero=bool(lo > 0 or hi < 0))
    r_ch = summ(d_ch, acc - ch); r_fl = summ(d_fl, acc - fl)
    out.append(dict(label=c["label"], results_root=c["results_root"], arm_dir=c["arm_dir"],
                    n=n, acc=float(acc), chance_mean_1_over_nopt=float(ch), floor_always_A=float(fl),
                    vs_chance=r_ch, vs_floor=r_fl,
                    published_v1_delta_pp=c.get("v1_delta_vs_floor_pp"),
                    published_v1_boot_p=c.get("v1_boot_p")))
    print(f"{c['label'][:32]:32} acc={acc:.6f} | vs_chance {100*(acc-ch):+7.4f} pp "
          f"[{r_ch['ci95_lo_pp']:+6.3f},{r_ch['ci95_hi_pp']:+6.3f}] p={r_ch['boot_p']:.4f} "
          f"{'SIG' if r_ch['ci_excludes_zero'] else 'ns ':3} | vs_floor {100*(acc-fl):+7.4f} pp "
          f"p={r_fl['boot_p']:.4f} {'SIG' if r_fl['ci_excludes_zero'] else 'ns '}")

json.dump(out, open("/tmp/s2_03_result.json", "w"), indent=1)
print()
# the headline recount
sig_ch = sum(1 for r in out if r["vs_chance"]["ci_excludes_zero"] and r["vs_chance"]["delta_pp"] > 0)
pt_ch  = sum(1 for r in out if r["vs_chance"]["delta_pp"] > 0)
sig_fl = sum(1 for r in out if r["vs_floor"]["ci_excludes_zero"] and r["vs_floor"]["delta_pp"] > 0)
pt_fl  = sum(1 for r in out if r["vs_floor"]["delta_pp"] > 0)
print(f"n cells = {len(out)}")
print(f"  above chance, POINT estimate only : {pt_ch}/{len(out)}   <- what the paper counts")
print(f"  above chance, CI excludes 0       : {sig_ch}/{len(out)}   <- what the floor side is held to")
print(f"  above floor,  POINT estimate only : {pt_fl}/{len(out)}")
print(f"  above floor,  CI excludes 0       : {sig_fl}/{len(out)}")
