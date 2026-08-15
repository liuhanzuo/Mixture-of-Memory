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

Both axes moved, so this is not one flaky item. **The root cause of each axis has
since been isolated to a specific mechanism** — see
§ *Ceiling discrepancy vs `NUMBER_AUDIT.md:284`* below, and
`evidence/gate1_base/ceiling_discrepancy_rootcause.json`. Both readings are
≥98 % feasible, so the gate's stated precondition ("gold ceiling 0.9894, so
≥98 % of items feasible") holds under either. **To keep the adjudication
independent of which ceiling is authoritative, the contrast is reported on both
feasible sets** (§4).

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

## Ceiling discrepancy vs `NUMBER_AUDIT.md:284`

`NUMBER_AUDIT.md:284` records, for the **same split and the same `n_rows = 1033`**:

```
n_rows 1033   gold_ceiling_base = 0.9894   gold_ceiling_plus = 0.8025
```

`PROPOSAL.md:204` and `STATUS.json.kill_gate.gate_1.kill_if` both quote the
`0.9894` figure verbatim. Gate 1's zwfy6 re-measurement gives `1.0` / `0.8122`.
**Both axes moved**, so this was treated as a possible grader/data/protocol
change and root-caused before any p-value was accepted. Full evidence:
`evidence/gate1_base/ceiling_discrepancy_rootcause.{py,json}` (**0 GPU**).

### Old vs new, and the exact items

| axis | wzc1 (`NUMBER_AUDIT.md:284`) | zwfy6 (this gate) | Δ items | which items |
|---|---|---|---|---|
| base | 0.989351404 (1022/1033) | **1.0** (1033/1033) | **11** | every line `L0`–`L10` of **`HumanEval/32`** (`find_zero`) |
| plus | 0.802516941 (829/1033) | **0.812197483** (839/1033) | **10** | 9 lines of **`HumanEval/130`** (`tri`) + **`HumanEval/15/L0`** (`string_sequence`) |

Closure is asserted, not eyeballed: `1033 − 1022 = 11` and `839 − 829 = 10`
exactly match the per-item diff sets, and the `task_id` sets are identical
(1033 = 1033, no extras on either side). In every one of the 21 discordant items
the direction is the same: **wzc1 FAIL → zwfy6 PASS.**

### Root cause — base axis: an evalplus **version** difference (not data)

The wzc1 ceiling was produced by `dllm_draft/scripts/spanlen_gold_ceiling.py`
under `.venv_b200`, where `import evalplus` resolves to **PyPI evalplus 0.3.1**
in `site-packages`. Gate 1 on zwfy6 resolved it to the repo's **vendored**
evalplus (upstream commit `26d6d00`). In 0.3.1, `eval/__init__.py:unsafe_execute`
has:

```python
if dataset == "humaneval":
    if "find_zero" == entry_point:
        assert abs(_poly(*inp, out)) <= atol
        continue                      # <-- returns WITHOUT details[i] = True
```

It `continue`s **before** `details[i] = True` / `progress.value += 1`. So
`progress.value` stays 0, `untrusted_check` returns `details = []`, and its own
guard `if len(details) != len(inputs): stat = FAIL` rewrites a genuine PASS into
FAIL. The vendored copy carries the two missing lines before its `continue`.
Measured on **one host, same data files, same grade wrapper, only `PYTHONPATH`
differing**:

| grader | status | `n_details` | `n_inputs` | `n_pass` |
|---|---|---|---|---|
| PyPI evalplus **0.3.1** | `fail` | **0** | 100 | 0 |
| vendored (`26d6d00`) | `pass` | **100** | 100 | 100 |

`HumanEval/32` is the **only** `find_zero` task in the split, which is exactly why
the base-axis discrepancy is exactly its 11 rows and nothing else. This is
deterministic across 3 repeats per version — it is **not** host load or
wall-clock flakiness.

> This corrects an earlier reading in this same document, which noted that the
> *vendored* `evalplus/eval/*.py` are byte-identical across both disks (md5
> `bcd21dfd…`, true) and inferred from that the grader could not be the cause.
> The vendored copies do match; the wzc1 ceiling run simply **never imported the
> vendored copy**. Byte-identity of a file on disk is not evidence about which
> file the interpreter loaded.

### Root cause — plus axis: the sandbox's **4 GiB `RLIMIT_AS`** (host-dependent)

The plus-axis 10 are a *different* mechanism, and it is **not** a version
difference: with the **same vendored evalplus**, LOCAL/wzc1 still fails these and
zwfy6/`.73` passes them. `query_maximum_memory_bytes()` defaults to 4 GiB and
`reliability_guard()` applies it as `RLIMIT_AS`/`RLIMIT_DATA` inside the grading
subprocess. `HumanEval/130` and `HumanEval/15` have `plus_input` entries with
n ≈ 10⁶ whose reference outputs are ~10⁶-element lists; materialising them
exceeds 4 GiB of address space once the interpreter's own footprint counts. The
allocation raises a bare `MemoryError`, which `unsafe_execute`'s
`except BaseException` silently books as a **wrong answer** rather than an error.
Measured on LOCAL, vendored evalplus, numpy 2.4.6,
`SingleLineInfilling/HumanEval/130/L0`:

| `RLIMIT_AS` | result | exception |
|---|---|---|
| 4 GiB (evalplus default) | 7 / 125 inputs fail | `MemoryError` at `n=999999, 999997, …` |
| unlimited (`EVALPLUS_MAX_MEMORY_BYTES=-1`) | **0 / 125 fail, status `pass`** | — |

Because the trip point depends on the host's baseline footprint, this is a
**cross-host reproducibility defect of the harness**, not a property of the
benchmark or of any arm. (The wzc1 record additionally marks 7 rows of
`HumanEval/63` `timeout` on the plus axis where zwfy6 marks them `fail`; both
readings agree those 7 are infeasible, so they do not move the ceiling.)

### Ruled out by measurement (not by assumption)

| candidate | wzc1 | zwfy6 | verdict |
|---|---|---|---|
| split file md5 | `30129634e180d80c19d6ddcd4cf43f9c` | same | **identical** |
| `HumanEvalPlus-v0.1.10.jsonl` md5 | `fe585eb4df8c88d844eeb463ea4d0302` | same | **identical** |
| `get_human_eval_plus_hash()` | `fe585eb4df8c…` | same | **identical** |
| expected outputs, `md5(repr(gt[bid][axis]))` for `HumanEval/{32,130,15,63}` × {base, plus} | — | — | **identical, all 8** |
| vendored `evalplus/eval/__init__.py` md5 | `bcd21dfd412e10b6825fab093428d579` | same | identical — but **not the file wzc1 loaded** |
| grade wrapper semantics | `spanlen_gold_ceiling._grade` | `score_infilling.grade_one` | same inputs/expected/ref_time assembly, same `min_time_limit=1.0`, `gt_time_limit_factor=4.0`, same acceptance rule |
| numpy | 2.4.6 | 1.26.4 | **not the cause** — the failing comparisons raise `MemoryError` before any `allclose`; lifting `RLIMIT_AS` on the *same* numpy 2.4.6 makes them pass |
| wall-clock flakiness | — | — | **not the cause** — deterministic across 3 repeats per (host, version) |

One genuine difference that is *not* causal: the on-disk groundtruth pickle md5
differs across disks (`7f1bfa50…` vs `ded78f78…`) despite identical byte size.
The **decoded** expected values are identical (row above), so this is pickle
container nondeterminism, not a data difference.

### Which ceiling is authoritative

**zwfy6 / Gate 1** (base `1.0`, plus `0.8122`). Both discrepancies are wzc1-side
defects with identified mechanisms — a grader bug fixed upstream, and a sandbox
address-space cap that silently converts `MemoryError` into a wrong answer.
Neither is a property of the benchmark. `NUMBER_AUDIT.md:284` is therefore
**superseded**; per `LIFECYCLE_SCHEMA.md` §0 the original line is left
byte-intact and a dated note is appended below it.

### Does this change Gate 1's decidability? **No.**

- The pre-registered `kill_if` is a function of the `qwen_fim` vs
  `dreamon_oracle` paired contrast **only** (significant at α=0.05 **and**
  |Δ| < 0.02). The ceiling enters the clause solely as the parenthetical
  precondition *"gold ceiling 0.9894, so ≥98 % of items feasible"*.
- Under the measured base ceiling of **1.0**, that precondition is **100 %
  feasible** — *strictly more permissive* than the pre-registered ≥98 %, i.e.
  satisfied a fortiori. The ceiling change is **favourable** to Gate 1's
  decidability, not adverse.
- **α=0.05 and |Δ|<0.02 are retained verbatim.** They were *not* rewritten
  because the ceiling moved; doing so would be exactly the kind of
  post-hoc threshold edit this proposal's own retraction history forbids.
- The contrast is adjudicated on **both** feasible sets (§4.1 all 1033 items
  under the zwfy6 ceiling; §4.2 the 1022-item wzc1 feasible subset). Both give
  exact-McNemar **p = 1.0000** and |Δ| < 0.001. **The verdict is identical under
  either ceiling**, so nothing about the KILL depends on which one is
  authoritative.

### Bonus finding, recorded for the protocol note

This is a **cross-host gold-ceiling irreproducibility on HumanEval-Infilling**
with two independent, separately demonstrated mechanisms — an upstream
special-oracle bug that turns a passing program into a failure, and a 4 GiB
sandbox cap that turns an out-of-memory event into a wrong answer. Both are
silent: neither surfaces as an error to the caller, and both move a *ceiling*,
i.e. the denominator every arm is normalised against. It belongs in the
survivors list alongside the existing ceiling/cost-unit observations.

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

---

## 9. MAIN independent verification (2026-08-15, after agent delivery)

MAIN did not accept the two claims above on the agent's word. Both were re-derived
from primary sources, because a **pre-registered constant** (`gold_ceiling_base`)
and the **KILL arithmetic** are the two things a wrong agent report would most
easily corrupt.

### 9.1 The paired contrast, recomputed from the raw per-item files

Recomputed directly from the six `evidence/gate1_base/score_base/*_score_base.json`,
reading `per_task[].pass` and ignoring every summary field the agent wrote:

| arm | `n_pass` | recomputed pass@1 | `pass_at_1` in file | agree |
|---|---|---|---|---|
| `qwen_fim` | 966 | 0.935140 | 0.935140 | ✅ |
| `dreamon_oracle` | 965 | 0.934172 | 0.934172 | ✅ |
| `dream_fim` | 909 | 0.879961 | 0.879961 | ✅ |
| `dreamon_fim` | 895 | 0.866409 | 0.866409 | ✅ |
| `qwen_prefix` | 682 | 0.660213 | 0.660213 | ✅ |
| `dream_prefix` | 519 | 0.502420 | 0.502420 | ✅ |

All six arms carry an **identical 1033-element `task_id` set** (asserted, not assumed),
and each file's `n` is 1033.

MAIN's own exact two-sided McNemar, hand-computed as
`2 * sum(C(n,i) for i in 0..min(b,c)) / 2**n` on the discordant pairs:

```
b (qwen-only) = 39   c (dreamon-only) = 38   discordant = 77
exact two-sided McNemar p = 1.000000
Delta = +0.00096805   |Delta| = 0.00096805 < 0.02  -> True
```

**Both KILL conditions reproduce on MAIN's independent arithmetic.** The verdict
does not rest on the agent's statistics code.

### 9.2 The evalplus version defect, verified at source level

The claim "PyPI 0.3.1 `continue`s before recording the pass" is a source-level
assertion, so MAIN read both files rather than trusting the quoted snippet.

`/opt/.../dllm_draft/vendor/evalplus/evalplus/eval/__init__.py` (zwfy6, **vendored**):

```
187:  if "find_zero" == entry_point:
188:      assert abs(_poly(*inp, out)) <= atol
189:      details[i] = True
190:      progress.value += 1
191:      continue
```

`dllm_draft/.venv_b200/lib/python3.11/site-packages/evalplus/eval/__init__.py`
(wzc1, **PyPI**, `evalplus-0.3.1.dist-info` → `Version: 0.3.1`):

```
187:  if "find_zero" == entry_point:
188:      assert abs(_poly(*inp, out)) <= atol
189:      continue
```

The two lines are **absent** in 0.3.1 at exactly the stated position. MAIN further
confirmed the import actually resolves that way, rather than inferring it:

```
$ dllm_draft/.venv_b200/bin/python -c "import evalplus.eval as E; print(E.__file__)"
.../dllm_draft/.venv_b200/lib/python3.11/site-packages/evalplus/eval/__init__.py
```

and that **no** `site-packages/evalplus` exists on `.73` at all, so the zwfy6 run
could only have loaded the vendored copy. Confirmed: `zwfy6` is authoritative,
and §8's earlier md5-based exculpation of the grader was the wrong inference.

### 9.3 What MAIN did NOT re-verify

- The plus-axis `RLIMIT_AS` mechanism (agent-measured; not load-bearing for Gate 1,
  which is adjudicated on the **base** axis).
- The 10 000-resample bootstrap (the **exact** McNemar is the pre-registered primary
  and it reproduces; the bootstrap is corroborative).

### 9.4 Standing consequence

`gold_ceiling_base = 1.0` is **more permissive** than the pre-registered
"≥98 % of items feasible", so the base axis remains the correct axis to judge on and
the KILL stands *a fortiori*. The thresholds α=0.05 and |Δ|<0.02 were **not** altered.
`NUMBER_AUDIT.md:284` stays byte-intact with its dated append-only note.
