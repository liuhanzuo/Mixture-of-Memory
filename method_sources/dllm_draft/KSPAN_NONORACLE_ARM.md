# k-span (multi-region) code infilling: NON-ORACLE diffusion arm

**Status.** Scope-fill for Retraction 6. **The surviving claim does NOT survive.**

## 0. Provenance

- **Frozen spec**: `data/kspan/kspan_spec_v1.jsonl`
  - `md5 1638ce4068bd870471a61e0d061d3ea5`
  - `sha256 1cc12a50d1f4255f1036ae34e9d04c0027a029f017afe07bdc7afba11fb3af83`
  - **415 rows, byte-identical to the spec `runs/kspan_diffusion` / `runs/kspan_ar_fim` consumed.**
- **Node**: `.252` 8×B200 (wzc1 disk).
- **Env**: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/.venv_b200/bin/python` (py3.11, torch 2.11+cu128, transformers 4.51.3, evalplus 0.3.1 — **same grader binary as the previous main arms**).
- **Model**: `models/DreamOn-v0-7B` (native).
- **Sampler**: DreamOn `infilling_with_expansion` (variable-length, model picks its own span length via `<|expand|>`), applied **sequentially left-to-right one call per hole** (holes < j filled with the model's own text; holes > j deleted from the suffix — matches `ar_fim`, not `ar_fim_fair`).
- **Hyperparameters** (aligned to `runs/kspan_diffusion`): `temperature=0.0`, `top_p=0.95`, `alg=entropy`, `alg_temp=0.0`, `min_gen_len=4`, `max_gen_len=64`, `steps=128`, `delete_eos_token=True`, `pad_eos_to_right=True`, `batch_size=1`, `pad_to_max_len=False`, `max_prompt_len=1024`, `max_tokens=2048`.
- **Post-processing per hole**: first physical line of the decoded mask region kept (matches `ar_fim`'s `first_line()` rule so hole-count == region-count).
- **Grading**: `scripts/score_kspan.py`, `evalplus.eval.untrusted_check` (base). Grader self-test PASSED on 12 spec rows (gold passes, `pass`-stub fails). **Shard completeness: 415/415 unique, 0 duplicates.**

Generation code: `scripts/generate_kspan_dreamon_nonoracle.py`. Sharded via `scripts/_run_kspan_dreamon_nonoracle_8gpu.sh` (`LOCAL_RANK=0 RANK=$g` per-shard). Total wall-clock ≈ 3 minutes on 8×B200.

## 1. Full-n ladder (all 415 rows, official grader)

| k | n | dreamon_nonoracle | diffusion (oracle) | AR-FIM | AR-FIM-fair |
|---|---|-------------------|--------------------|--------|-------------|
| 1 | 164 | **.726** | .671 | **.866** | **.866** |
| 2 | 108 | **.796** | .796 | .870 | .852 |
| 3 | 84 | .702 | .750 | .738 | .798 |
| 4 | 59 | **.542** | .746 | .644 | .644 |

**EM-to-gold (stripped), same rows:**

| k | dreamon_nonoracle | diffusion (oracle) | AR-FIM | AR-FIM-fair |
|---|---|---|---|---|
| 1 | .549 | .579 | .598 | .598 |
| 2 | .556 | .648 | .639 | .630 |
| 3 | .429 | .524 | .524 | .524 |
| 4 | .322 | .424 | .390 | .373 |

Notes:
- **Non-oracle beats oracle at k=1 (.726 vs .671).** Removing the oracle length here HELPS diffusion — consistent with the pilot note that a fixed 12-token canvas kicked Dream-Coder into chat mode. DreamOn's native span-selection is on-distribution.
- **Non-oracle collapses at k=4 (.542 vs oracle's .746).** DreamOn's length picker degrades sharply when required to fill four short (mean 4-token) spans in the same file.

## 2. Balanced ladder (n=59 tasks present at every k) — the one that matters

| k | n | dreamon_nonoracle | diffusion (oracle) | AR-FIM | AR-FIM-fair |
|---|---|-------------------|--------------------|--------|-------------|
| 1 | 59 | .915 | .898 | **.949** | **.949** |
| 2 | 59 | .864 | .847 | .898 | .915 |
| 3 | 59 | .780 | .831 | .780 | .831 |
| 4 | 59 | **.542** | **.746** | .644 | .644 |

**k=1 → k=4 within-task drops (bigger = worse):**

| arm | drop |
|---|---|
| **dreamon_nonoracle** | **−.373** |
| ar_fim / ar_fim_fair | −.305 |
| diffusion (oracle) | −.153 |

**The non-oracle diffusion arm degrades FASTER than either AR arm.** The oracle version was slower.

## 3. Interactions on the balanced (n=59) set

`k:arm` interaction from `logistic passed ~ k + arm + k:arm`, SE clustered by task_id. Negative sign = AR degrades faster than diffusion (the surviving claim).

| arm pair | balanced k:arm | z | p | naive (raw drops) |
|---|---|---|---|---|
| **diffusion (oracle) vs ar_fim** (from prior work) | **−0.435** | −2.89 | .0038 | +0.153 |
| diffusion (oracle) vs ar_fim_fair | −0.208 | −2.15 | .032  | — |
| **dreamon_nonoracle vs ar_fim** | **−0.009** | −0.07 | **.943** | **−0.068** |
| dreamon_nonoracle vs ar_fim_fair | −0.058 | −0.37 | .71  | −0.068 |

**Full (unbalanced, nested) versions for completeness:**

| arm pair | nested k:arm | z | p | naive 4-cell |
|---|---|---|---|---|
| diffusion (oracle) vs ar_fim (from prior work) | −0.598 | −4.09 | 4.3e-5 | +0.297 |
| dreamon_nonoracle vs ar_fim | −0.224 | −2.11 | .035  | — |
| dreamon_nonoracle vs ar_fim_fair | −0.173 | −1.58 | .114 | — |

The nested numbers still lean toward "AR degrades faster than non-oracle diffusion" (p=.035 against strict `ar_fim`), but the design-appropriate **balanced clustered estimate is indistinguishable from zero (p=.94)** and its point estimate is 0.

## 4. Truncation / abort / parseability (crash-out disclosure)

| k | n | parseable | truncated tasks | aborted tasks | grader errors |
|---|---|---|---|---|---|
| 1 | 164 | 96.3% | 2 | **0** | 0 |
| 2 | 108 | 94.4% | 8 | **0** | 0 |
| 3 | 84  | 89.3% | 10 | **0** | 0 |
| 4 | 59  | 81.4% | 3 | **0** | 0 |

- **DreamOn never crashed** on multi-hole input. There were 0 aborted holes at every k. The anticipated `pad_delete_to_right`-style failure did not fire on any of the 415 tasks.
- Truncation (hit `max_gen_len=64` without producing a newline in the target line) rose modestly with k. This is not a crash, just DreamOn choosing a longer-than-fits span; the fill is still recorded and graded.
- **Parseability drops steeply with k** (96.3% → 81.4%). The non-oracle span picker produces syntactically broken code more often as k grows, largely explaining the pass@1 collapse. AR-FIM's parseability is nearly flat over k in the prior run (>93% throughout).

## 5. Cost axes (mean per task, all rows including truncated)

| k | dreamon_nonoracle tok_fed | diffusion (oracle) tok_fed | AR-FIM tok_fed |
|---|---|---|---|
| 1 | 3,584   | 2,026 | 192 |
| 2 | 6,797   | 3,487 | 434 |
| 3 | 13,295  | 5,366 | 676 |
| 4 | 19,581  | 8,229 | 979 |

`tokens_fed`: non-oracle DreamOn is **~1.5–2.4× more expensive** than oracle Dream-Coder (extra canvas re-feeds while it decides span length via `<|expand|>` events), and **20–36× more expensive** than AR-FIM. Non-oracle diffusion loses the efficiency argument as well as the accuracy argument.

## 6. Verdict against the three possibilities the retraction spec listed

Restating the surviving claim:
> "Both families degrade with region count k; AR degrades ~2× faster."

Only the first half survives. The second half was an oracle artifact.

- ~~**(a) The surviving claim survives**: dreamon_nonoracle degrades in k, AR degrades ~2× faster than it.~~ **No.** Balanced k=1→k=4 drops: **non-oracle .373 > AR .305 > oracle .153**. The non-oracle arm degrades *faster* than AR, not slower.
- **(b) The surviving claim inverts / turns null**: without the oracle, diffusion degrades AS FAST OR FASTER than AR. **Yes, this is what happened.** Balanced cluster-robust interaction against `ar_fim` is **−0.009 (p=.94)** — sign is nominally still "AR faster" but the point estimate is essentially zero and 95% CI covers ±0.26. Against `ar_fim_fair` the point estimate is likewise −0.058 (p=.71). **The "AR degrades ~2× faster than diffusion" finding cannot be reported on this benchmark. The k-span infilling story on HumanEval yields no publishable family-level claim.**
- **(c) The non-oracle arm crashes out**: **No.** Zero aborts at every k. DreamOn's variable-length sampler is not multi-region-safe *by design* — but our sequential per-hole wrapping around `infilling_with_expansion` did not trigger a crash on any of the 415 tasks. What did happen is a parseability collapse (96.3% → 81.4%) as span-picking errors compound across multiple holes.

## 7. What can be reported honestly

Restricted to what this data actually supports:

1. **Both diffusion and AR degrade with region count k.** Balanced within-task within-family:
   - `dreamon_nonoracle` .915 → .542 (drop .373)
   - `ar_fim`            .949 → .644 (drop .305)
   - `ar_fim_fair`       .949 → .644 (drop .305)
   - `diffusion` (oracle) .898 → .746 (drop .153)
2. **The apparent diffusion advantage over AR is an oracle-length artifact.** Giving diffusion the per-hole gold token count masks its own degradation on this task. Removing the oracle wipes out both the interaction magnitude and the direction: balanced clustered `k:arm ≈ 0` at n=59 clusters. Any efficiency argument for the diffusion canvas also collapses (non-oracle is 1.5–2.4× more `tokens_fed` than oracle, 20–36× more than AR).
3. **The oracle version at k=4 was operating at a design-favouring width** (mean 4-token target spans, exactly the tokens to emit). Non-oracle DreamOn cannot reproduce that with its own span selection.

Reasonable framings that survive:
- Family-degradation-rate finding: **withdrawn**.
- **Non-oracle-length-provisioning is where the diffusion side of this comparison lives or dies.** Any k-span infilling result that gives diffusion the oracle length should be flagged.
- The `k=1` non-oracle-beats-oracle result (.726 > .671, +5.5 pt full-n; .915 > .898, +1.7 pt balanced) is a small but consistent hint that fixed-canvas / oracle-length forces off-distribution generation for DreamOn at single-region k=1 — echoing the pilot's note that a fixed 12-token canvas broke Dream-Coder into chat scaffolding. Not the main finding, but noted.

## 8. What was NOT changed

- The frozen spec `data/kspan/kspan_spec_v1.jsonl` was not rebuilt (hashes above match those in `KSPAN_INFILLING_RESULTS.md` §0).
- The `runs/kspan_diffusion` / `kspan_ar_fim` / `kspan_ar_fim_fair` / `kspan_decon_*` numbers were not re-scored; the tables above pull them from `score.json` in each of those directories via `scripts/analyze_kspan_nonoracle.py`.
- `KSPAN_INFILLING_RESULTS.md` and `DLLM_RESULTS_20260807.md` were **not edited** — MAIN splices as needed.

Artifacts:
- Run: `runs/kspan_diffusion_nonoracle/` (415 solutions + metrics, `score.json`).
- Generator: `scripts/generate_kspan_dreamon_nonoracle.py`.
- Launcher: `scripts/_run_kspan_dreamon_nonoracle_8gpu.sh`.
- Analyzer: `scripts/analyze_kspan_nonoracle.py`.

---

# 9. SAME-MODEL DE-ORACLE CONTROL (added 2026-08-07) — closing R7's two-variable confound

## 9.0 The hole this section closes

Retraction 7 concluded *"without the oracle length handout, diffusion degrades FASTER than AR"* by
comparing a **DreamOn-v0-7B non-oracle** arm (§1–§7 above) against the **Dream-Coder-v0-Instruct-7B
oracle** arm. That changes **two variables at once — the MODEL and the length provisioning.** The
conclusion was therefore open to *"DreamOn is simply a weaker model"*, and a reviewer would find it.

This section adds the missing **one-variable control**: the *same* Dream-Coder-v0-Instruct-7B, same
sampler, same holes, with a **FIXED canvas** (8 or 12 mask tokens per hole) replacing the oracle
per-hole gold token counts.

**Spec binding is provable.** The fixed-canvas specs differ from the frozen spec in **exactly
`hole_token_lengths`** (and its derived `total_masked_tokens`) — verified field-by-field over all 236
rows. `segments`, `gold_lines`, `hole_line_numbers`, `task_id`, `reference_sha256` are byte-identical,
and every `spec_id` is a member of the frozen 415.

| file | rows | sha256 | md5 |
|---|---|---|---|
| `data/kspan/kspan_spec_v1.jsonl` (frozen, unchanged) | 415 | `1cc12a50…3af83` | `1638ce40…3ea5` |
| `data/kspan/kspan_spec_v1_fix8.jsonl` | 236 | `5b282313299ebee964e3d2568015313e76a3775d25b90b008cbff766ddc58401` | `cefdc33b79ee5697ff479f133253c24f` |
| `data/kspan/kspan_spec_v1_fix12.jsonl` | 236 | `52e590eda67614839ebcf0123013e30d21b4bbac1df8677353a345877be94f32` | `7d50c0bdf70e39b16a316914ca2e6bae` |

**Provenance rescue.** These two arms previously existed **only in `/tmp/kem/diff_fix8` and
`/tmp/kem/diff_fix12`** and would have died on reboot. They are now persisted to
`runs/kspan_diffusion_fix8/` and `runs/kspan_diffusion_fix12/`, verified byte-consistent with the
`/tmp` originals before the originals were left alone (`sha256` of the concatenated 8 solution shards:
fix8 `447d97df…6e2f`, fix12 `9b0dc59b…d9a8`; metrics shards likewise identical; 236/236 rows each).

**Grading axis.** Re-graded with the repo's existing `scripts/score_kspan.py` (official
`evalplus.eval.untrusted_check`, evalplus 0.3.1), **`which = base`** — matching
`runs/kspan_diffusion/score.json`, `runs/kspan_ar_fim`, `runs/kspan_ar_fim_fair` and
`runs/kspan_diffusion_nonoracle`, all of which record `"which": "base"`. `min_time_limit=1.0`,
`gt_time_limit_factor=4.0`. Grader self-test PASSED on 12 spec rows (gold passes, `pass`-stub fails)
for both arms. Shard completeness asserted: **8/8 ranks, 236 unique rows, 0 duplicates, 0 metric rows
missing**, cell sizes asserted `1=59,2=59,3=59,4=59`. `score_kspan.py` was extended (additively) to
record `spec_sha256` / `spec_rows` / `grader` / shard counts inside every `score.json` and to support
`--expect-ranks` / `--expect-rows`, so the spec binding and shard completeness are now provable from
the artifact alone.

The re-grade **reproduced the workflow agent's cells exactly — 0/236 per-row `passed` disagreements
in both arms.**

## 9.1 Both fixed-canvas arms ARE the balanced set

`n=59` tasks × `k∈{1,2,3,4}` = 236 rows, and that task set is **identical** to the balanced subset
(tasks present at every k) used for every balanced table above. No re-subsetting was needed.

## 9.2 Balanced ladder, all five arms, one grading axis

pass@1 on the 59 tasks present at every k (`which=base`, official grader):

| arm | k=1 | k=2 | k=3 | k=4 | k1→k4 drop |
|---|---|---|---|---|---|
| diffusion **oracle** (Dream-Coder) | .898 | .847 | .831 | .746 | **−.153** |
| **diffusion fix8** (Dream-Coder, same model) | .729 | .475 | .373 | .271 | **−.458** |
| **diffusion fix12** (Dream-Coder, same model) | .831 | .356 | .407 | .237 | **−.593** |
| dreamon_nonoracle (DreamOn) | .915 | .864 | .780 | .542 | −.373 |
| ar_fim | .949 | .898 | .780 | .644 | −.305 |
| ar_fim_fair | .949 | .915 | .831 | .644 | −.305 |

## 9.3 ★ The three-way slope comparison — and a units correction

**The reported −0.148 / −0.017 reproduce EXACTLY, but they are `k:arm` INTERACTION coefficients
against `ar_fim`, not per-arm within-task slopes `β_k`.** The task framing called them "balanced
within-task slopes"; they are not on the same scale as the oracle's −0.346 or DreamOn's −0.772, which
*are* per-arm β_k. Both quantities are given below so the comparison is apples-to-apples. The
mislabeling does not change any conclusion, but reporting −0.148 next to −0.346 as if they were the
same statistic would have understated the fixed-canvas arms' degradation by a factor of ~4.

**(a) Per-arm within-task slope** — `logistic passed ~ k`, balanced n=59, SE clustered by `task_id`
(the design-appropriate SE; each task contributes 4 correlated rows). Negative = degrades with k.

| arm | β_k | SE (clustered) | z | p | n_obs | n_clusters |
|---|---|---|---|---|---|---|
| diffusion **oracle** | **−0.346** | 0.093 | −3.71 | **2.1e-4** | 236 | 59 |
| **diffusion fix8** | **−0.633** | 0.112 | −5.66 | **1.5e-08** | 236 | 59 |
| **diffusion fix12** | **−0.764** | 0.122 | −6.27 | **3.7e-10** | 236 | 59 |
| dreamon_nonoracle | **−0.772** | 0.138 | −5.58 | **2.4e-08** | 236 | 59 |
| **ar_fim** | **−0.781** | 0.139 | −5.60 | **2.1e-08** | 236 | 59 |
| ar_fim_fair | −0.829 | 0.166 | −5.01 | 5.4e-07 | 236 | 59 |

The two reference values on disk are **confirmed**: oracle β_k = −0.346, p=2.1e-4 and
dreamon_nonoracle β_k = −0.772, p=2.4e-08 (task text said 2.6e-8; 2.44e-08 — agrees to rounding).
Note `ar_fim`'s own β_k is **−0.781, p=2.1e-08** — numerically the same p the task text attached to
the oracle arm, which is a coincidence of the two arms, not an error.

**(b) `k:arm` interaction vs `ar_fim`** (balanced, clustered by task). Negative = **AR degrades faster
than the diffusion-side arm** (i.e. the retracted claim's direction).

| diffusion-side arm | k:arm | SE | z | p | verdict |
|---|---|---|---|---|---|
| diffusion **oracle** | **−0.435** | 0.150 | −2.89 | **.0038** | AR faster — the (retracted) claim |
| **diffusion fix8** | **−0.148** | 0.167 | −0.89 | **.375** | **null** |
| **diffusion fix12** | **−0.017** | 0.186 | −0.09 | **.928** | **null** |
| dreamon_nonoracle | −0.009 | 0.132 | −0.07 | .943 | null |

**−0.148 (p=.375) and −0.017 (p=.928) reproduce to 3 decimals.** (Task text said p=.377 / p=.928;
we get .375 / .928.) Against `ar_fim_fair`: fix8 −0.196 (p=.29), fix12 −0.065 (p=.76).

**Reading**: with the model held fixed, removing the oracle length moves the interaction from
**−0.435 (p=.004, "AR degrades ~2× faster")** to **−0.148 (p=.38)** and **−0.017 (p=.93)** — i.e. to
zero, landing on top of the DreamOn arm's −0.009 (p=.94). The de-oracled *same model* does **not**
beat AR's degradation rate. **The model swap was not doing the work; the oracle length was.**

## 9.4 ★ Output sanity — the fixed-canvas arms are PARTLY compromised (mandatory caveat)

The earlier pilot note ("with a fixed 12-token canvas Dream-Coder fell out of infilling mode and
emitted chat scaffolding") **partially fires.** Checked on the generations, not just the scores:

| axis (balanced, 590 holes) | oracle | fix8 | fix12 |
|---|---|---|---|
| parseable (k=1 → k=4) | .966 → .932 | .966 → **.678** | **.915 → .661** |
| EM-to-gold stripped (k=1) | .814 | **.136** | **.051** |
| holes overrunning their single-line target | 4.2% | **27.5%** | **62.5%** |
| tasks with ≥1 overrunning hole | 9.7% | 50.4% | **80.9%** |
| `# Example usage` / `if __name__` / bare `print(` scaffolding | 0.5% | **9.0%** | **13.4%** |
| prose comment inside a fill | 2.9% | 13.9% | **44.6%** |
| markdown fence | 0 | 14 (2.4%) | 0 |
| chat special tokens (`<|im_start|>` etc.) | 0 | **0** | **0** |
| grader errors / aborts / truncations | 0 | **0** | **0** |

**Diagnosis — the canvas is length-FORCING, not length-free.** `run_diffusion` fills every mask slot;
**98.8% of holes emit exactly the budget** (mean `emitted − budget` = −0.01 tokens; `masks_left = 0`
on all 236 tasks in both arms). The model cannot stop early. Against a balanced-subset gold
distribution of mean 7.56 / median 6 tokens per hole:
- **fix8**: canvas longer than gold on **67.8%** of holes (mean 2.9 surplus slots)
- **fix12**: canvas longer than gold on **88.3%** of holes (mean 6.0 surplus slots)

So the surplus slots must be filled with *something*, and the model spends them running past the end
of the target line into extra statements, `# Example usage` blocks and `print(...)` calls. This is
mechanically why EM-to-gold collapses to .05/.01 (**it is structurally impossible** on 86.4% / 96.2%
of rows — the fill cannot equal a shorter gold line at a longer forced length) and why parseability
falls to ~.66. It is **not** full chat-mode collapse: **zero** chat special tokens, zero aborts, zero
grader errors, no empty fills, and the fills are still recognisably code for the right function.

**Decisive sub-control.** To separate "de-oracling hurts the model" from "surplus slots mechanically
corrupt the file", we truncated every fill to its **first physical line** — exactly `ar_fim`'s
`first_line()` rule, which makes hole-count == region-count and deletes the overrun — then re-spliced
and re-graded with the official grader:

| arm | k=1 | k=2 | k=3 | k=4 | β_k (clustered) | k:arm vs ar_fim |
|---|---|---|---|---|---|---|
| fix8, first-line truncated | .712 | .373 | .288 | .186 | **−0.769** (p=1.1e-09) | **−0.012** (p=.95) |
| fix12, first-line truncated | .678 | .390 | .254 | .119 | **−0.895** (p=3.4e-12) | **+0.114** (p=.50) |

**Removing the overrun does not rescue the arms — it makes them slightly worse, and the interaction
stays at zero (or flips sign).** The steep degradation is therefore *not* an artifact of the
overrunning text; it is the de-oracled model genuinely failing to place k correct short spans. The
`k:arm ≈ 0` conclusion is robust to the post-processing choice. (Saved as
`runs/kspan_diffusion_fix{8,12}/score_firstline_subcontrol.json`.)

**Honest scope limit.** The fixed-canvas arms are a **valid control for the DEGRADATION-RATE question**
(`k:arm`, which is what R7 rests on, and which is stable across two canvas widths and two
post-processings). They are **NOT** a clean estimate of "Dream-Coder's true non-oracle pass@1", because
a forced-length canvas is off-distribution in its own way: the k=1 level (.729 / .831 vs oracle .898)
mixes de-oracling with length-forcing. **`hole_token_lengths` is the only field that changed, but a
fixed canvas is not the same intervention as "let the model choose its own length"** — that latter
intervention is what the DreamOn arm provides, and the two disagree on *level* while agreeing on
*slope*. Anyone citing a non-oracle pass@1 *level* for Dream-Coder should use neither arm; anyone
citing a non-oracle *degradation rate* now has three mutually corroborating arms.

## 9.5 ★ What this does to Retraction 7 — **STRENGTHENS IT**

Of the three possibilities:

- ✅ **STRENGTHENS.** The same-model de-oracle arms degrade **at least as fast as AR**, so R7's
  conclusion holds with the model-swap confound removed. Per-arm β_k: fix8 **−0.633**, fix12
  **−0.764**, DreamOn **−0.772**, all statistically indistinguishable from `ar_fim`'s **−0.781** and
  all far steeper than the oracle's **−0.346**. Interaction vs `ar_fim` collapses from −0.435 (p=.004)
  to −0.148 (p=.38) / −0.017 (p=.93). **The "AR degrades ~2× faster than diffusion" finding was an
  oracle-length artifact, not a DreamOn-is-weaker artifact.**
- ❌ **Does NOT weaken it.** Neither fixed-canvas arm degrades *slower* than AR. Both point estimates
  keep the nominal sign but sit on zero; no amendment to R7's direction is required.
- ⚠️ **Partially ambiguous, but not disqualifying.** Parseability does fall to ~.66 at k=4 and
  scaffolding comments appear (up to 13.4% of fills), so the arms are **degraded** — but they did not
  fall out of infilling mode (0 chat tokens, 0 aborts, 0 grader errors), and the **first-line-truncated
  sub-control reproduces the same β_k and the same null interaction**, which is what makes the
  degradation-rate reading safe to report. Reported with the §9.4 scope limit attached.

**Amendment to Retraction 7.** R7's *conclusion* stands unchanged; its *evidence base* is amended from
one cross-model arm to **three arms — DreamOn non-oracle (different model, free length) plus
Dream-Coder fix8 and fix12 (same model, forced length)** — which removes the "DreamOn is just weaker"
escape. R7 should read: *"the diffusion-vs-AR degradation-rate advantage is produced by the oracle
per-hole length handout. Removing it — whether by switching to a model that picks its own span length,
or by de-oracling the very same model with a fixed canvas — drives the `k:arm` interaction to zero
(−0.009 / −0.148 / −0.017; all p > .37)."* The k-span line remains **terminated** as a paper direction;
this section exists so the retraction's *methodology lesson* (feeding task #170, null calibration) is
free of the two-variable confound.

**One line for #170**: *an oracle length handout, not the model identity, produced the entire
apparent family-level advantage — and a same-model de-oracle control was required to know that.*

## 9.6 Cost (balanced subset, mean per task)

| arm | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| diffusion oracle | 2,036 | 3,821 | 5,868 | 8,229 |
| diffusion fix8 | 2,017 | 4,061 | 6,094 | 8,099 |
| diffusion fix12 | 3,073 | 6,283 | 9,573 | 12,916 |
| dreamon_nonoracle | 4,096 | 5,827 | 12,395 | 19,581 |
| ar_fim | 254 | 501 | 740 | 979 |

`tokens_fed`. The fixed canvas does not buy back the efficiency argument either: fix12 is ~1.6× the
oracle and ~13× `ar_fim`.

## 9.7 Artifacts and what was NOT changed

- Runs (rescued from `/tmp`, verified byte-consistent): `runs/kspan_diffusion_fix8/`,
  `runs/kspan_diffusion_fix12/` — 8 solution + 8 metric shards, 236 rows, `score.json`,
  `score_firstline_subcontrol.json`.
- Specs: `data/kspan/kspan_spec_v1_fix8.jsonl`, `data/kspan/kspan_spec_v1_fix12.jsonl`.
- Analyzer: `scripts/stats_kspan_deoracle.py` (reuses `fit_logistic` from
  `scripts/stats_kspan_interaction.py`; no new grader, no new statistics). Output saved to
  `logs/kspan_deoracle_stats.txt`. Generation logs `logs/fix{8,12}_g*.log`.
- `scripts/score_kspan.py` extended **additively only** (spec sha256 + shard/row asserts); the grading
  path, `grade_one`, `which` axis handling and self-test are untouched, so previously-written
  `score.json` files remain comparable.
- **The frozen spec `data/kspan/kspan_spec_v1.jsonl` was NOT rebuilt** (sha256 re-verified as
  `1cc12a50…3af83`, matching §0).
- `runs/kspan_diffusion` / `kspan_ar_fim` / `kspan_ar_fim_fair` / `kspan_diffusion_nonoracle` were
  **not re-scored**; all comparisons read their existing `score.json` (all `which=base`).
- `DLLM_RESULTS_20260807.md` and `KSPAN_INFILLING_RESULTS.md` were **not edited** — MAIN owns those and
  splices.
- 0 GPU-hours used: grading and statistics only, on already-existing generations.
