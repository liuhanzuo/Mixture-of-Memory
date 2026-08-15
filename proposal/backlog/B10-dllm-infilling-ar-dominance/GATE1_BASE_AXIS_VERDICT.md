# B10 Gate 1 — base-axis re-score: **VERDICT = KILL**

**Date**: 2026-08-15
**GPU cost**: **0** (CPU-only; the 8 H20s on `.73` stayed 99–100 % occupied by the
pre-existing `train_olmo2_arch_probe2.py` DDP job throughout — verified before and
after, see §6)
**Adjudicated by**: base-axis re-score of the six existing arms. No generation was
re-run; `solutions.jsonl` was read as-is.

---

## 1. The pre-registered judgement criterion, verbatim

From `PROPOSAL.md` §5 (lines 199–219) and `STATUS.json` `kill_gate.gate_1`:

> ### Gate 1 — base-axis re-score (0 GPU, CPU only, hours)
>
> Re-score all six existing arms with `score_infilling.py --which base`. Solutions
> are already on disk; nothing is regenerated.
>
> **KILL if:** on the base axis (gold ceiling 0.9894, so ≥98 % of items feasible),
> the `qwen_fim` vs `dreamon_oracle` paired contrast is **not** significant at
> α=0.05 **and** |Δ| < 0.02 — i.e. AR and the best diffusion arm are
> indistinguishable on the axis the benchmark was designed for.
>
> **PROCEED only if** the base axis produces a significant, ceiling-robust, and
> *directionally stable* AR advantage over the strongest diffusion arm. Given the
> evidence, treat proceeding as the unlikely branch.

`STATUS.json` `kill_gate.gate_1.if_killed`, verbatim:

> rewrite as a protocol note or archive. Do NOT re-frame to hunt a different
> ranking from the same six arms -- that is the nested-ladder error already
> retracted twice in this line of work (Retractions 6 and 7).

Δ is reported throughout as **`qwen_fim` − `dreamon_oracle`** = AR minus
best-diffusion. **Positive Δ = AR advantage.**

---

## 2. Data-integrity assertions — run FIRST, all PASS

Nothing was computed until these passed. Raw output:
`evidence/gate1_base/gate1_integrity.json` (md5 `cbd8c0ea70a7975cdf8f8b4d4657342a`).

| assertion | result |
|---|---|
| 6 arms × rows | **1033 each** (`dream_fim`, `dreamon_fim`, `dreamon_oracle`, `dream_prefix`, `qwen_fim`, `qwen_prefix`) |
| duplicate `task_id` within any arm | **0** |
| arm item-id sets identical to each other | **True** |
| arm item-id set == benchmark split id set | **True** (1033 unique) |
| rows with missing/non-`str` `solution` | **0** |
| rows with `middle is None` | **0** |
| rows with empty `solution` | **0** |
| grader self-test per arm (`canonical_pass`, `stub_fail`) | **10/10 and 10/10, `trustworthy: true`, all 6 arms** |

The grader self-test matters here specifically: `score_infilling.py`'s docstring
records that a previous hand-rolled runner gave an *empty* program a full pass and
caused a retraction. All six re-scores re-verified on every invocation that the
canonical middle passes and a `pass` stub fails.

**Grading determinism** (checked because the primary Δ is worth ~1 item): both
primary arms were independently re-scored a second time on the base axis.
`qwen_fim` 966→966, `dreamon_oracle` 965→965, **0 per-item flips** in either.
Files under `evidence/gate1_base/replicate/` are byte-identical (same md5) to the
first pass. The verdict is not resting on grader jitter.

---

## 3. Measured base-axis numbers

`pass@1` on `--which base`, n=1033 per arm (plus axis reproduced from the
pre-existing `score.json` for reference only — it was not recomputed):

| arm | **base pass@1** | base n_pass | plus pass@1 (reference) |
|---|---|---|---|
| **qwen_fim** (AR, FIM) | **0.9351403679** | **966** | 0.7637947725 |
| **dreamon_oracle** (diffusion, oracle length) | **0.9341723136** | **965** | 0.7589545015 |
| dream_fim | 0.8799612778 | 909 | 0.7115198451 |
| dreamon_fim | 0.8664085189 | 895 | 0.7018393030 |
| qwen_prefix | 0.6602129719 | 682 | 0.5324298161 |
| dream_prefix | 0.5024201355 | 519 | 0.4123910939 |

`dreamon_oracle` is confirmed to be the **strongest diffusion arm on the base
axis** (0.9342 > 0.8800 `dream_fim` > 0.8664 `dreamon_fim` > 0.5024
`dream_prefix`), so the pre-registered contrast is against the right arm and no
arm substitution is needed.

### Gold ceiling — measured, with a discrepancy declared

The kill condition quotes **0.9894**. That value comes from wzc1
`dllm_draft/runs/spanlen/gold_ceiling_SingleLine.json`
(`gold_ceiling_base = 0.989351403678606`, 11 of 1033 items base-infeasible).

Gate 1 ran on zwfy6, so the ceiling was **re-measured there** with the same
splice (`prompt + canonical_solution + suffix`) and the same official EvalPlus
sandbox:

| source | `gold_ceiling_base` | base-infeasible items | `gold_ceiling_plus` |
|---|---|---|---|
| wzc1 record (quoted in the gate) | 0.989351404 | 11 | 0.802516941 |
| **zwfy6 re-measurement (this gate)** | **1.0000000** | **0** | 0.812197483 |

**The 11-item difference is entirely `HumanEval/32`** (`find_zero`, all 11 of its
`L0`–`L10` rows). `find_zero` is one of EvalPlus's *special-oracle* tasks
(`_special_oracle._poly`, `atol=1e-4`, 100 base inputs each wall-clock-limited),
i.e. exactly the kind of item whose pass/fail can move with host load. The inputs
were ruled out as the cause: the split file (md5 `30129634e180…`),
`HumanEvalPlus-v0.1.10.jsonl` (md5 `fe585eb4df8c…`) and the vendored
`evalplus/eval/*.py` (md5 `bcd21dfd…`, `8d95f931…`, `e9ff521c…`, commit `26d6d00`)
are **byte-identical across both disks**. Both ceilings are ≥98 % feasible, so
the gate's stated precondition ("gold ceiling 0.9894, so ≥98 % of items feasible")
holds under either reading. **To keep the adjudication independent of which
ceiling is authoritative, the contrast is reported on both feasible sets** (§4).

---

## 4. The primary paired contrast

Test used: **exact McNemar** (two-sided exact binomial on the discordant pairs —
no χ², no continuity correction) as the primary, **plus a paired bootstrap**
(10 000 resamples of *items*, so pairing is preserved; **seed 20260815**,
numpy 1.26.4). Both are reported. No scipy exists on either disk, so the
exact-test code is validated in-run against textbook Clopper-Pearson values and
against the definitional tail invariants `P(X≥k|lo)=P(X≤k|hi)=0.025`; the
self-check is embedded in `gate1_base_stats.json`
(`exact_stat_implementation_selfcheck.all_ok = true`).

### 4.1 All 1033 items (PRIMARY — zwfy6 ceiling is 1.0, so this *is* the feasible set)

| quantity | value |
|---|---|
| n (paired items) | **1033** |
| `qwen_fim` pass@1 | 0.9351403679 (966/1033) |
| `dreamon_oracle` pass@1 | 0.9341723136 (965/1033) |
| **Δ = qwen_fim − dreamon_oracle** | **+0.0009680542** (+0.097 pp) |
| discordant pairs | **77** (b = 39 qwen-only, c = 38 dreamon-only) |
| concordant | 927 both pass, 29 both fail |
| **exact McNemar p (two-sided)** | **1.0000** |
| Δ 95 % CI (Clopper-Pearson on the discordant split) | **[−0.01639, +0.01825]** |
| paired bootstrap Δ | +0.0009680542 |
| paired bootstrap 95 % CI | **[−0.01646, +0.01742]** |
| paired bootstrap p (two-sided) | **0.9498** |

The Δ is **one item** out of 1033. 39 vs 38 discordant is as close to a coin flip
as this design can produce.

### 4.2 Robustness on the wzc1 feasible subset (11 `HumanEval/32` items removed)

| quantity | value |
|---|---|
| n | **1022** |
| `qwen_fim` pass@1 | 0.9354207436 |
| `dreamon_oracle` pass@1 | 0.9363992172 |
| **Δ** | **−0.0009784736** ← **sign flips** |
| discordant | 75 (b = 37, c = 38) |
| exact McNemar p | **1.0000** |
| Δ 95 % CI (CP) | [−0.01822, +0.01634] |
| bootstrap Δ 95 % CI / p | [−0.01761, +0.01566] / **0.9528** |

Both ceiling readings give p = 1.0000 and |Δ| ≈ 0.001. **The direction is not even
stable between them** — Δ is +0.00097 on all 1033 items and −0.00098 on the wzc1
feasible subset. Under the gate's own PROCEED wording ("*directionally stable*"),
this alone forecloses PROCEED.

### 4.3 Full pairwise base-axis matrix (record only, NOT used to adjudicate)

Recorded so nobody has to re-run it, and explicitly **not** a menu to pick a new
headline from (see §7). `qwen_fim`–`dreamon_oracle` is the *only* pair in the
matrix that is not significant.

| pair | Δ | exact McNemar p |
|---|---|---|
| dreamon_oracle–qwen_fim | −0.000968 | **1.0** |
| dream_fim–dreamon_fim | +0.013553 | 0.2649 |
| dream_fim–dreamon_oracle | −0.054211 | 2.28e−08 |
| dream_fim–qwen_fim | −0.055179 | 8.25e−07 |
| dreamon_fim–dreamon_oracle | −0.067764 | 5.14e−16 |
| dreamon_fim–qwen_fim | −0.068732 | 1.18e−10 |
| dream_fim–dream_prefix | +0.377541 | 5.02e−95 |
| dream_fim–qwen_prefix | +0.219748 | 1.06e−35 |
| dreamon_fim–dream_prefix | +0.363988 | 9.21e−88 |
| dreamon_fim–qwen_prefix | +0.206196 | 2.72e−34 |
| dreamon_oracle–dream_prefix | +0.431752 | 8.28e−123 |
| dreamon_oracle–qwen_prefix | +0.273959 | 1.79e−65 |
| dream_prefix–qwen_fim | −0.432720 | 5.95e−117 |
| dream_prefix–qwen_prefix | −0.157793 | 9.92e−32 |
| qwen_fim–qwen_prefix | +0.274927 | 4.89e−72 |

---

## 5. Adjudication — the two KILL conditions, each checked

| # | pre-registered condition | measured | holds? |
|---|---|---|---|
| 1 | contrast **not** significant at α=0.05 | exact McNemar **p = 1.0000** (b=39, c=38, discordant=77); paired bootstrap p = 0.9498. Both ≫ 0.05 | **YES** |
| 2 | **\|Δ\| < 0.02** | **\|Δ\| = 0.00097** (0.097 pp); 95 % CI [−0.0164, +0.0183] — the CI itself lies inside ±0.02 | **YES** |

Both conditions hold → **KILL**.

Cross-checking against the PROCEED wording, all three requirements fail:
- *significant* — no (p = 1.0000);
- *ceiling-robust* — no (p = 1.0000 on both ceiling readings);
- *directionally stable* — no (sign flips, +0.00097 → −0.00098).

The gate's own §5 prediction is confirmed: plus axis Δ = +0.0048 (p = 0.635),
plus-axis feasible subset Δ = −0.0012 (p = 1.000), and now **base axis
Δ = +0.00097 (p = 1.000)**. All three axes agree. Per PROPOSAL.md §5:
**"AR vs diffusion" has no measurable answer on this surface at n=1033**, and the
conclusion-reversal framing is dead for good.

## VERDICT: **KILL**

---

## 6. GPU budget compliance

The gate is CPU-only by construction (`solutions.jsonl` read from disk; the only
compute is EvalPlus's `untrusted_check` sandbox, which is plain CPU subprocesses).
Every step ran with `CUDA_VISIBLE_DEVICES=""` forced (hard-coded in
`run_gate1_base_rescore.sh` and passed explicitly to the ad-hoc invocations).

`.73` GPU state at launch (08:38) and after completion (08:57): **all 8 cards
96421 MiB / 99–100 % util**, owned by the pre-existing
`scripts/train_olmo2_arch_probe2.py --resume_from …` DDP ranks (started Aug 14,
PIDs 3914162/3914171/3914203/3914205/3914208…). That job was still alive and at
full utilisation after Gate 1 finished. **Zero GPU-seconds consumed, zero
interference.**

---

## 7. What this forecloses (binding)

Per `PROPOSAL.md` §5 "Standing rule" and `STATUS.json`
`kill_gate.gate_1.if_killed`:

- B10 must be **rewritten as a protocol note or archived**.
- **Gate 2 (lineage repair, 2–4 GPU-h) and Gate 3 (matched suffix-gain, 1–2 GPU-h)
  are NOT authorised** — both were conditioned on Gate 1 saying PROCEED.
- ⛔ **It is forbidden to re-frame B10 to hunt a different ranking from the same
  six arms.** That is the nested-ladder error this repo already retracted twice
  (Retractions 6 and 7). The pairwise matrix in §4.3 is provenance, **not** a menu.
- The `MUST NOT CLAIM` list in `PROPOSAL.md` §4.4 stands unchanged and is now
  additionally supported on the base axis: "AR beats masked diffusion on
  diffusion's home turf" is **not** measurable here on any of the three axes.

What survives (already recorded in `STATUS.json.robust_findings_that_survive`, not
re-derived here): the two protocol observations (the benchmark's own feasibility
ceiling on the axis it grades; the cost-unit sensitivity) and the suffix-gain
result. Gate 4's standing rule also still binds: **no absolute pass@1 from this
surface may be reported as a capability measurement without a decontaminated
companion**, so none of the §3 numbers may be quoted as capability claims.

One new, narrow protocol observation was *produced* by this gate and is worth
carrying into any protocol note: **the base-axis gold ceiling is not reproducible
across hosts on the special-oracle task `HumanEval/32`** — 0.9894 on wzc1 vs
1.0000 on zwfy6, from byte-identical inputs, grader and dataset. That is a
statement about `find_zero`'s wall-clock-limited special oracle, not about any
model, and it is measured (§3), not inferred.

---

## 8. Provenance

**Ran on**: `.73` = `28.85.35.73` (8×H20, zwfy6 disk), hostname `TENCENT64.site`.
Adjudication statistics also re-verified on wzc1 LOCAL for the self-check.

| item | value |
|---|---|
| scorer | `/apdcephfs_zwfy6/…/dllm_draft_104/scripts/score_infilling.py`, md5 `41a5dd1816a7ef8a51e66f43d33ef730` (unmodified; `--which base` is an existing flag) |
| grader | `evalplus.eval.untrusted_check`, vendored `dllm_draft/vendor/evalplus` @ commit `26d6d00`, `evalplus.__version__ 0.1.0.dev1` |
| split file | `data/humaneval_infilling/HumanEval-SingleLineInfilling.jsonl`, 1033 rows, md5 `30129634e180d80c19d6ddcd4cf43f9c` (**identical on both disks**) |
| EvalPlus dataset | `data/evalplus/HumanEvalPlus-v0.1.10.jsonl`, md5 `fe585eb4df8c88d844eeb463ea4d0302` (**identical on both disks**) |
| python | `/apdcephfs_zwfy6/…/dllm_draft_104/.venv_dream/bin/python` = **Python 3.11.6**, numpy 1.26.4, **no scipy** (hence the validated hand-rolled exact tests) |
| repo commit (`dllm_draft_104`) | `3555dc79` |
| bootstrap | 10 000 resamples, **seed 20260815**, paired at the item level |
| `--jobs` | 48 CPU workers (host has 384 cores) |

Solutions consumed (all 1033 rows, unmodified, `outputs/infilling_single_line/<arm>/solutions.jsonl`):

| arm | md5 |
|---|---|
| dream_fim | `9e84f127ffee9fc1ff16cea4650d39d7` |
| dreamon_fim | `2759299d89a8f98580e87c2cb8d6f926` |
| dreamon_oracle | `fb5a988b6421cba39e237c91a8455ff1` |
| dream_prefix | `6f7b58d4fc9e78a94a38c2731275f8e9` |
| qwen_fim | `491d4968afab61ac0a8e3be7945a083a` |
| qwen_prefix | `c9342563db731020c2cc346ff9474016` |

### Artefacts, copied zwfy6 → wzc1 with `scp -O` and md5-verified

All under `proposal/backlog/B10-dllm-infilling-ar-dominance/evidence/gate1_base/`:

| file | md5 (identical on both disks) |
|---|---|
| `gate1_base_stats.json` (full adjudication) | `804056f7f9dbb015c4c05dc483d03fa6` |
| `gate1_integrity.json` | `cbd8c0ea70a7975cdf8f8b4d4657342a` |
| `gate1_gold_ceiling_zwfy6.json` | `f770e2893bfb2fefa117e604375a9f55` |
| `score_base/dream_fim_score_base.json` | `2a29a03333845836083d68d510f55c08` |
| `score_base/dreamon_fim_score_base.json` | `6c03dd1a11a82ac0179decbae936c32f` |
| `score_base/dreamon_oracle_score_base.json` | `1f653ffb60a053b5fea041a5e9e834ac` |
| `score_base/dream_prefix_score_base.json` | `d6e69cbcc403448a686092fe26040496` |
| `score_base/qwen_fim_score_base.json` | `55c11d014c13218d278d4826f8f6688b` |
| `score_base/qwen_prefix_score_base.json` | `e14b6392e5b8a83634588e021c2bf43e` |
| `replicate/qwen_fim_score_base_rep2.json` | `55c11d014c13218d278d4826f8f6688b` (= rep1 ⇒ 0 flips) |
| `replicate/dreamon_oracle_score_base_rep2.json` | `1f653ffb60a053b5fea041a5e9e834ac` (= rep1 ⇒ 0 flips) |
| `gold_ceiling_SingleLine_wzc1_reference.json` | `21b4766e824e9210adf0d6ce08240eea` |

Scripts written for this gate (also copied to `dllm_draft_104/scripts/` on zwfy6
so it is re-runnable there): `gate1_integrity_assert.py`,
`run_gate1_base_rescore.sh`, `gate1_gold_ceiling.py`, `gate1_stats.py`.
Per-arm scorer logs (with the six grader self-tests) in `scorer_logs/` — the
directory is named `scorer_logs/` and the files carry a `.txt` suffix because the
repo `.gitignore` drops both `logs/` (line 74) and `*.log` (line 73), which would
have left the self-test provenance untracked.

**Note on non-destructiveness**: base-axis output was written to `score_base.json`,
leaving the pre-existing plus-axis `score.json` — the provenance for the numbers
already in `STATUS.json` — untouched.

### Reproduce

```bash
# on .73, zero GPU
D=/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104
CUDA_VISIBLE_DEVICES="" $D/.venv_dream/bin/python $D/scripts/gate1_integrity_assert.py \
    $D/outputs/infilling_single_line \
    $D/data/humaneval_infilling/HumanEval-SingleLineInfilling.jsonl 1033
bash $D/scripts/run_gate1_base_rescore.sh          # 6 arms, ~45 s each
CUDA_VISIBLE_DEVICES="" $D/.venv_dream/bin/python $D/scripts/gate1_stats.py \
    $D/outputs/infilling_single_line \
    $D/outputs/infilling_single_line/gate1_gold_ceiling_zwfy6.json \
    $D/outputs/infilling_single_line/gate1_base_stats.json \
    $D/outputs/infilling_single_line/gold_ceiling_SingleLine_wzc1.json
```
