"""S2-02: is the paper's `v2 null always <= v1 floor` claim safe on MMLU-Pro?

The proof in 03_method.tex:62 is  sum_L p_L m_L <= max_L m_L, i.e. UNSTRATIFIED.
The estimator is stratified within n_opt. Stratified, the bound one actually gets is
  acc_hat <= sum_s w_s max_L m_{s,L}                        (per-stratum argmax)
which is >= max_L sum_s w_s m_{s,L} = f_const               (global argmax).
The paper's ordering is recovered IFF argmax_L m_{s,L} is the SAME letter in every
stratum. Pure dataset property -> one arm's per-item file settles it.
"""
import glob, json, os
from collections import Counter, defaultdict

LET = "ABCDEFGHIJ"
ROOT = "<REPO_ROOT>/mmlu_pro_letter_content_results"
ARM = "7B_base"          # dataset property: any arm gives the identical answer
shards = sorted(glob.glob(os.path.join(ROOT, ARM, "per_example_mmlu_pro_shard*of8.jsonl")))
idx = {int(os.path.basename(s).split("_shard")[1].split("of")[0]) for s in shards}
assert idx == set(range(8)), f"MISSING shards {sorted(set(range(8))-idx)}"
recs = []
for s in shards:
    with open(s) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
ids = [r["item_id"] for r in recs]
assert len(set(ids)) == len(ids), "duplicate item_id"
assert len(recs) == 12032, f"n={len(recs)} != 12032"
assert sum(1 for r in recs if r.get("nan")) == 0, "nan present"

strata = defaultdict(Counter)
for r in recs:
    strata[r["n_opt"]][r["gold_letter"]] += 1
n = len(recs)
w = {k: sum(v.values()) / n for k, v in strata.items()}

# global argmax (v1) -- tie-break exactly as best_constant_letter() does
gold = Counter(r["gold_letter"] for r in recs)
gbest = max(gold.items(), key=lambda kv: (kv[1], -LET.index(kv[0])))[0]

print(f"n={n}  global best-constant letter = {gbest}  ({gold[gbest]/n:.6f})")
print()
print(" n_opt   n_s   w_s     argmax_L  m_s,argmax   m_s,{gbest}   differs")
rows = []
for k in sorted(strata):
    c = strata[k]; ns = sum(c.values())
    L = max(c.items(), key=lambda kv: (kv[1], -LET.index(kv[0])))[0]
    rows.append((k, ns, L, c[L]/ns, c.get(gbest, 0)/ns))
    print(f"  {k:2}   {ns:5}  {w[k]:.4f}     {L}      {c[L]/ns:.4f}      "
          f"{c.get(gbest,0)/ns:.4f}     {'YES' if L != gbest else '-'}")

f_const = max(sum(w[k] * strata[k].get(L, 0) / sum(strata[k].values())
                  for k in strata) for L in LET)
loose = sum(w[k] * max(strata[k].values()) / sum(strata[k].values()) for k in strata)
same = all(r[2] == gbest for r in rows)
print()
print(f"argmax_L identical across ALL {len(rows)} strata? {same}")
print(f"f_const (v1, global argmax)             = {f_const:.6f}")
print(f"sum_s w_s max_L m_(s,L) (stratified UB) = {loose:.6f}")
print(f"SLACK = {100*(loose - f_const):.4f} pp   <- how far the proof's conclusion could be violated")
print()
print("VERDICT: " + ("bound holds by coincidence of argmaxes (must be stated as an "
                     "empirical property, not a theorem)" if same else
                     "argmaxes DIFFER across strata -> the 'always' is FALSE as written; "
                     "the true stratified upper bound is the loose one above"))
json.dump({"n": n, "global_best_letter": gbest, "n_strata": len(rows),
           "per_stratum": [{"n_opt": r[0], "n_s": r[1], "argmax_L": r[2],
                            "m_argmax": r[3], "m_global_best": r[4]} for r in rows],
           "argmax_identical_across_strata": same,
           "f_const_v1": f_const, "stratified_upper_bound": loose,
           "slack_pp": 100*(loose - f_const)},
          open("/tmp/s2_02_strata_result.json", "w"), indent=1)
