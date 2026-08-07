# Track C — Intra-trajectory ICC, the 5%-budget concentration pathology, and the design effect

**Node:** .252 (wzc1, 8x B200). **Cost:** ~3.6 GPU-h of a ~10 GPU-h ceiling.
**Artifacts:** `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/icc_track_c/`
(`pool.jsonl` md5 `79b481314f94d26698c6aa694e4474de`, `emb/*.npy`, `seeds_*.json`, `deff_*.json`, `res_*.json`)
**This is a MEASUREMENT track. No selector was built.** Every number below is re-derived from
raw weights + raw data on this node; nothing is inherited from the prior workflow.

---

## 0. Verdict on the pin

**PIN PARTIALLY FAILED.** The pool and the concentration pathology reproduce. The ICC ladder —
the part the track was built on — **does not reproduce as stated, and the non-monotonicity that
the task called "the finding" is an artifact of two conflated things.**

| pin claim | re-derived | verdict |
|---|---|---|
| pool = 100K samples / 21,744 traj / 19 benchmarks | **100,002 / 21,744 / 19** | **PASS** (exact) |
| random: 3764 distinct (75.3%) | **3734.8 exact hypergeometric; 3738.3 ± 35.3 (10 seeds)** | **PASS** |
| stratified: 4018 (80.4%) | **4043 ± 17** | **PASS** |
| RDS+ top-k: 1074 distinct (21.5%) | **1102 ± 87 (i17)**; 15.7–25.6% across target sizes | **PASS** |
| RDS+ max from ONE traj = 71 | **74.8 ± 14.7 (i17)** — range **48–89** over 5 seeds | **PASS in mean, but "71" is not a stable statistic** |
| RDS+ benchmarks hit = 14 | **16.4 ± 0.5 (i17)** | marginal |
| RDS+ top-3 share 78.0% | **79.4 ± 1.7** | **PASS** |
| **ICC = 0.802 (1.7B) / 0.848 (4B) / 0.487 (8B-Base)** | **see §2 — mislabeled quantity** | **FAIL** |
| **"ICC on the selection score"** | the three pin values are **embedding ICCs**, not score ICCs | **FAIL** |

The pin's ICC triplet is closest to the **multivariate embedding** ICC, not the selection-score
ICC. Direct evidence: the surviving prior artifact `/tmp/icc/E17.npy` (12,000 x 2048) rerun through
the prior `measure.py` gives **0.7722**, and my independently re-embedded Qwen3-1.7B-**Instruct**
restricted to the same 12K prefix gives **ICC_embed = 0.7724** (full pool 0.7738) — reproducing the
prior pipeline to 2e-4. The pin reported "0.802" for that cell. Meanwhile the *selection-score* ICC
for the same model/protocol is **0.873**. So "0.802" matches neither cleanly; it is in the
embedding-ICC family and the label "ICC on the selection score" is wrong.

---

## 1. What was rebuilt

The prior agent's `/tmp/ab` (19 parquet) and `/tmp/icc/pool.jsonl` survived; the **full-pool
embeddings did not** (`E17.npy` is a 12,000-row prefix subsample, 12% of the pool, so the pin's
K=5000 selection could not have come from it). I therefore re-embedded the **entire 100,002-sample
pool** for **8 models**, plus a 4-model response-only control: 12 full-pool passes, 8 shards each.

- RDS+ exactly per arXiv:2503.01807: position-weighted (`i/sum i`) mean-pool of the frozen LM's
  last hidden states, L2-normalised, cosine to target set, one forward pass per candidate, zero
  training. Both aggregation variants measured: `max`-over-target (global top-k) and true
  round-robin over target queries.
- Sharding per house rule: `CUDA_VISIBLE_DEVICES=$g` **plus** `LOCAL_RANK=0 RANK=$g`.
- **Merge coverage asserted**: 8/8 shards required before a `_DONE` marker is written; reassembly
  from strided `[g::8]` shards verified against `pool.jsonl` row order with **0 mismatches**, all
  vectors unit-norm, zero non-finite entries.
- Estimators validated against ground truth before use: one-way ICC recovers synthetic
  0.00/0.30/0.80 as **0.0027/0.3019/0.8063**; nested ICC recovers (env .500, traj .300) as
  (**.470 ± .106**, **.319 ± .064**) over 12 seeds; the hierarchical design effect predicts
  **181.15** against a 20,000-draw Monte-Carlo **182.71** (0.9% error).

---

## 2. The ICC ladder: the confound is broken, and it is NOT scale

The task asked whether the .802 → .848 → **.487** non-monotonicity is a base-vs-instruct artifact
or a real scale effect, noting the original ladder cannot separate them. **It is base-vs-instruct.**
I found `Qwen--Qwen3-8b` (instruct) already on disk next to `Qwen3-8B-Base`, and pulled
`Qwen3-{0.6B,1.7B,4B}-Base` from HF, giving a **fully crossed 4 scales x {Base, Instruct}** grid —
not merely the extra probe points the task suggested, but complete separation of the two factors.

ICC of the **RDS+ selection score**, clustering on trajectory (mean ± sd over 5 target-set seeds,
full 100K pool, |target|=1000):

| scale | **Base** | **Instruct** | Base−Instruct |
|---|---|---|---|
| 0.6B | 0.877 ± 0.007 | 0.880 ± 0.007 | −0.003 |
| 1.7B | **0.501 ± 0.010** | **0.862 ± 0.007** | **−0.361** |
| 4B | 0.652 ± 0.005 | 0.845 ± 0.004 | −0.193 |
| 8B | **0.518 ± 0.010** | **0.808 ± 0.003** | **−0.290** |

Read the columns, not the diagonal. **Within Instruct, ICC is essentially flat and mildly
DECREASING in scale (.880 → .862 → .845 → .808).** Within Base it is non-monotone and noisy
(.877 → .501 → .652 → .518). The pin's ladder mixed one Base checkpoint (8B) into three Instruct
ones, so the "drop at 8B" is the Base column being read as the continuation of the Instruct column.
At **fixed 8B scale**, swapping Base→Instruct moves ICC by **+0.290**, which is ~29x the seed sd —
far larger than anything scale does. **There is no scale effect to explain.**

Sanity check that this is not an artifact of my protocol: the embedding ICC (the quantity the pin
actually reported) is nearly flat across the whole grid, 0.772–0.846, i.e. **it cannot produce the
pin's .802/.848/.487 spread either**. The pin's spread is not a stable property of any single
estimator in this design.

---

## 3. Most of the "intra-trajectory" correlation is not intra-trajectory

A one-way ANOVA clustering on trajectory absorbs **benchmark**-level clustering into the
between-trajectory term. AgentBank has 19 environments with wildly different surface form, so this
matters enormously. Fitting the proper nested model `x = mu + a_env + b_traj(env) + e`:

| model | ICC one-way (traj) | **ICC_env** | **ICC_traj \| env** |
|---|---|---|---|
| Qwen3-1.7B-Instruct | 0.862 | **0.745** | **0.128** |
| Qwen3-8B-Instruct | 0.808 | **0.672** | **0.151** |
| Qwen3-8B-Base | 0.518 | **0.502** | **0.044** |
| Qwen3-1.7B-Base | 0.501 | **0.483** | **0.046** |

**86–97% of the nominal "intra-trajectory" ICC is actually between-benchmark variance.** The
genuinely within-trajectory component is **0.04–0.15**. The prior workflow's own diagnostic already
contained this and it was not acted on: within-traj cosine 0.9285 vs **same-env different-traj**
cosine 0.9054 — an excess of only **0.023**.

This is the load-bearing correction for the headline. Deduplicating trajectories cannot remove
variance that lives at the benchmark level.

---

## 4. The mechanical-duplication control: ICC is largely a string artifact

The ms-swift per-assistant-turn expansion makes sample *k* of a trajectory contain the entire
prefix of sample *k−1*. Measured directly on the pool: **56.1% mean / 67.8% median of a sibling
sample's characters are a literal shared prefix** with the previous sample. Sibling samples are
near-duplicate *strings* by construction.

Re-embedding **response text only** (the one segment that is not shared) for four models:

| model | ICC_score prompt+resp | **ICC_score resp-only** | ICC_embed prompt+resp | **ICC_embed resp-only** |
|---|---|---|---|---|
| 1.7B-Base | 0.501 | **0.680** | 0.781 | **0.567** |
| 1.7B-Instruct | 0.862 | **0.679** | 0.774 | **0.604** |
| 8B-Base | 0.518 | **0.677** | 0.795 | **0.589** |
| 8B-Instruct | 0.808 | **0.631** | 0.772 | **0.577** |

Two things happen at once, and both are damaging to the original framing:

1. **The base-vs-instruct gap vanishes.** All four models collapse into 0.63–0.68 (spread 0.049,
   vs 0.361 for the prompt+response 1.7B pair). So the §2 gap is a property of how these models
   embed long shared *prompts*, not of how they perceive trajectories. The entire ladder — the pin's
   and mine — is a statement about prompt-embedding behaviour.
2. **Embedding ICC drops by ~0.20** (0.78 → 0.58), confirming a large mechanical component.

A residual within-trajectory correlation is real (resp-only ICC is ~0.65, not 0). But the specific
numbers in the pin table are not measuring what the label says.

---

## 5. The headline: design effect and effective sample size

Requested quantity: `DEFF = 1 + (m−1)·ICC` and the resulting n_eff for a 5000-sample RDS+ selection.
I report it, and two better estimators, because **the choice of m changes the answer by 3x** and the
naive form understates the damage.

**Qwen3-1.7B-Instruct** (the pin's headline cell), RDS+ global top-k, K=5000, 10 seeds:

| estimator | m used | DEFF | **n_eff** |
|---|---|---|---|
| `1+(m−1)ICC`, m = mean cluster size 4.567 | 4.567 | 4.08 ± 0.33 | **1233 ± 97** |
| exact per-cluster `sum n_g/(1+(n_g−1)ICC)` | — | 4.32 ± 0.34 | **1164 ± 89** |
| `1+(m−1)ICC`, m = **Kish** size 14.099 | 14.099 | 12.32 ± 0.78 | **407 ± 25** |
| **nested env+traj (correct model, §3)** | — | **947 ± 58** | **5.3 ± 0.3** |

Across models, RDS+ top-k at K=5000:

| model | ICC | m_mean | m_kish | n_eff (naive) | n_eff (exact) | **n_eff (nested)** |
|---|---|---|---|---|---|---|
| 1.7B-Instruct | 0.864 | 4.57 | 14.10 | 1233 | 1164 | **5.3** |
| 8B-Instruct | 0.806 | 3.94 | 11.62 | 1487 | 1382 | **5.7** |
| 8B-Base | 0.519 | 5.35 | 13.68 | 1537 | 1288 | **6.7** |

Random selection at the same budget, same model (1.7B-Instruct): DEFF 1.29, **n_eff 3871** (naive)
/ 3803 (exact) / 12.6 (nested).

**The defensible headline, stated the way I would defend it:** at a 5% budget on this pool,
RDS+ top-k buys **~1160–1230 independent samples out of 5000** — a **4.1–4.3x** loss — where random
selection at the same budget buys ~3800–3870 (1.3x loss). **RDS+ throws away roughly 3x more
statistical information than random at equal cost.** That comparison is the robust part: it holds
under every estimator, and it holds for both Base and Instruct.

Two mandatory caveats, or this number will not survive review:
- The Kish-weighted and nested variants are **not** cosmetic. Because cluster sizes are extremely
  skewed (max 176 in the pool, up to ~89 selected from a single trajectory), the variance-weighted
  cluster size is `sum m^2/sum m = 18.65` against a mean of 4.60 — a **4.1x** discrepancy. Using
  the mean, as `1+(m−1)ICC` invites, understates DEFF by ~3x.
- The nested n_eff ~5 is *arithmetically* correct for estimating a pool-wide mean, but it is
  dominated by there being only **19 benchmarks** (env Kish size ~1259 of 5000). It says "you have
  ~19 independent things, not 5000," which is a statement about **AgentBank's breadth**, not about
  RDS+. **Do not report ~5 as a selector indictment.** The selector-attributable term is the
  trajectory one.

---

## 6. What would kill this

- **The claim "ICC ≈ 0.8 measures trajectory structure" is already dead** (§3, §4). Any reviewer
  who fits a nested model or ablates the shared prefix gets 0.04–0.15, not 0.8.
- **The scale story is dead** (§2). Base-vs-instruct at fixed 8B is 29 seed-sd; scale within
  Instruct is flat-to-decreasing.
- The surviving, defensible claim is the **concentration pathology + the ~3x information loss of
  RDS+ vs random at equal budget**. To kill *that*, one would need to show it does not survive a
  cap-1 constraint (it does not — that is the known one-line fix, deliberately not pursued here),
  or that a downstream fine-tuning run does not track n_eff. **No downstream training was run in
  this track, so the link from n_eff to actual SFT quality is unverified** — that is the honest
  boundary of this measurement.
- `m*=18.65` and the 5%-budget arithmetic are pure properties of `pool.jsonl`. They die only if the
  pool build (ms-swift per-turn expansion, 6000-char prompt cap, 11.7% of samples at the cap) is
  judged the wrong unit of analysis. That is a legitimate objection: a different expansion changes
  every ICC here.

---

## 7. Reproduction

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/icc_track_c
PY=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/.venv_b200/bin/python
MODEL=<path> TAG=<tag> bash run_embed.sh      # 8-shard full-pool embed, ~8 min/model on 8x B200
/opt/conda/envs/torch-base/bin/python3 analyze.py <tag>      # 1-seed concentration + all ICCs
/opt/conda/envs/torch-base/bin/python3 seeds.py <tag> 5      # seed stability
/opt/conda/envs/torch-base/bin/python3 final_deff.py <tag> 10  # DEFF/n_eff + bootstrap CI
/opt/conda/envs/torch-base/bin/python3 analytic.py          # CPU-only: exact hypergeometric random row
/opt/conda/envs/torch-base/bin/python3 prov.py              # pin provenance: which ICC/protocol
```

Models: `Qwen3-{0.6B,1.7B,4B}-Base` downloaded from HF via hy-proxy into
`/apdcephfs_wzc1/share_304376610/pighzliu_code/models/`; `Qwen3-8B-Base`, `Qwen--Qwen3-8b`,
`Qwen--Qwen3-1.7b`, `Qwen3-4B` already on wzc1; `Qwen3-0.6B-Instruct` copied from zwfy6 via
`scp -O` (md5 `fed58aa6b81fb7e1e6a825af34b3f2dd`, verified both ends).
