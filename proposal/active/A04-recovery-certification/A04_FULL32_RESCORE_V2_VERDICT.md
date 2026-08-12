# A04 — Does the `RATIO`-vs-`NI` disagreement survive an independent re-scoring?

**Date**: 2026-08-12. **Node**: `.73` only (8×H20 sm_90), verified 0% idle before launch.
**Measured wall time**: **962 s** (16:04 min) for the re-scoring; **286 s** for the batch-size
sensitivity probe. **GPU-h: 2.78** (8 cards × 1248 s). Nothing else was touched: `.104` (paperC
Qwen3 heal), `.82` (A02 j=0), `LOCAL`/`.21` (SparseForge #246) all left alone.

**Answer in one line: the disagreement SURVIVES, and it is stronger than the prior document
allowed — the re-scoring is not merely "within jitter", it is BIT-IDENTICAL.**

---

## 0. What was RAN vs what was READ

| | |
|---|---|
| **RAN** | A second, protocol-identical scoring of `full32_dolmino@step25000` on **triviaqa, popqa, nq_open** into NEW dirs `full32_step25000_v2` / `full32_step25000_nqopen_v2` (the archived dirs were **not** overwritten). |
| **RAN** | A **control** re-execution of the committed `a04_shallow_rung_ni_7b.py` with **no overrides**, to prove the analysis path itself has not drifted since 2026-08-12 16:20. |
| **RAN** | The same committed script again with only the full32 arm's input dirs repointed to `_v2`. `NI` and `RATIO` were **not** reimplemented. |
| **RAN** | A `batch_size=48` **sensitivity probe** on the two flip-critical axes (popqa, nq_open). |
| **READ** | `A04_SHALLOW_RUNG_NI_DISCRIMINATION_VERDICT.md`, `evidence/a04_shallow_rung_ni_7b.json`, `evidence/a04_shallow_rung_remeasurement_sensitivity.json`, `A04_GATE_DESIGN.md` §2, and the archive's own scheduler logs. |

---

## 1. The recovered protocol, and how it was established

**Not** inferred from a current driver's defaults — recovered from **the archive's own scheduler
logs**, which echo the full parameter set at launch:

```
logs/cb_full32_step25000_sched.out        (zwfy6, written 2026-08-03 01:55:13)
logs/cb_full32_step25000_nqopen_sched.out (zwfy6, written 2026-08-03 02:07:50)

[2026-08-03 01:55:13] START full32_step25000  base=../models/OLMo-2-1124-7B
  extra=[--ckpt outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt
         --keep_front_layers 32 --n_fresh_layers 0]  bs=32  tasks=popqa,triviaqa
```

| parameter | recovered value | source |
|---|---|---|
| driver | `scripts/_run_closedbook_8shard.sh` | `bs=/tasks=` echo format is unique to it |
| `--batch_size` | **32** | `bs=32` in **both** sched logs |
| `--add_bos` | **0** | `"add_bos": false` in `summary.json` + all 8 `shardNof8.json` |
| `--num_shards` | 8 | `n_shards: 8`, 8 shard files present |
| `--max_new_tokens` | 32 | recorded in `meta` |
| `--max_ctx_len` | 512 | scorer default, never overridden by the driver |
| base LM protocol | `mode=pruned`, greedy (`do_sample=False`, `num_beams=1`), **no chat template** | scorer has no chat path; `meta.mode` |
| task call structure | `popqa,triviaqa` in one invocation, then `nq_open` in a second | two separate sched logs |

**Why this is a *same-code* comparison and not a code-version diff** — the trap that ruined the
prior document's jitter estimate:

* `scripts/eval_olmo2_closedbook_qa.py` md5 = `2ed41993241226c795a3ca38375933f7`, **identical on
  both disks**, and identical to the blob at commit `9fabb88` (2026-08-02) — i.e. **committed one
  day BEFORE the archive was produced and unmodified since.**
* Stack identical on both nodes: python 3.14.6, torch 2.13.0, transformers 5.5.4, CUDA 13.2,
  installed 2026-07-10 (before the archive).
* Archive was produced on **`.104`**; re-scoring ran on **`.73`**. Both are **H20 sm_90**, same
  disk (zwfy6), same conda env path. So the only uncontrolled variable is *which physical H20*.

I did **not** assume the archive ran on `.73` — `status/TRAINER_ACTIVITY.jsonl` task #126 records
it on `.104`, and I verified `.104`'s stack read-only rather than trusting that they match.

---

## 2. The result: bit-identical

`load_cb` from `proposal/shared/code/canonical_eval_loaders.py` (8/8 shards, exact n, no duplicate
`item_id`, no `nan` — merge **not** hand-rolled):

| axis | n | archived EM | re-scored EM | Δ (pp) | EM flips | pred-string flips |
|---|---:|---:|---:|---:|---:|---:|
| triviaqa | 17,944 | `0.5715002229157379` | `0.5715002229157379` | **+0.000000000000** | **0** | **0** |
| popqa | 14,267 | `0.18420130370785728` | `0.18420130370785728` | **+0.000000000000** | **0** | **0** |
| nq_open | 3,610 | `0.15817174515235458` | `0.15817174515235458` | **+0.000000000000** | **0** | **0** |

Stronger than equal means: **24/24 per-example shard files are byte-identical by sha256.** Not
"agrees to 4 dp" — the same bytes. `em_hits` 10255 / 2628 / 571 unchanged; `meta` dicts compare
equal; `shard_fail=0` on both merges.

### Integrity assertions (all enforced by the canonical loader, none hand-rolled)

`8/8` shards per axis · exact counts 17,944 / 14,267 / 3,610 · **0** duplicate `item_id` · **0**
`nan` rows · `item_id` sequences aligned across all arms.

---

## 3. `RATIO` and `NI` recomputed by RE-RUNNING the committed script

The committed `a04_shallow_rung_ni_7b.py` was re-run twice. The only code change is **two new
optional flags** (`--full32_cb`, `--full32_nq`) **whose defaults are the archived dir names**;
`ni_rule`, `ratio_rule`, the anchor (guard **G2**), `Δ`, `ρ` and every threshold are untouched.

**The control run is load-bearing**: re-running with *no* overrides reproduces
`evidence/a04_shallow_rung_ni_7b.json` **structurally identically** (every accuracy, residual,
margin and verdict; only timestamps/paths differ). So "the v2 numbers equal the archived numbers"
is demonstrated against a *re-execution*, not against a transcribed table.

### `RATIO(ρ = 0.85)`

| axis | archived ratio | re-scored ratio |
|---|---|---|
| triviaqa | `0.8993247391037447` | `0.8993247391037447` |
| popqa | `0.7442650807136788` | `0.7442650807136788` |
| mmlu_content | `0.9907687651331719` | `0.9907687651331719` |
| nq_open | `0.7716216216216216` | `0.7716216216216216` |

`mean_ratio` = **`0.8514950516430542`** → unchanged. Margin over ρ = **`+0.0014950516430541905`**
→ unchanged. **`RATIO` STILL ACCEPTS.**

(These per-axis ratios match the dispatch's independently-derived values to the last digit, so the
arithmetic is now confirmed by three separate computations.)

### `NI(Δ)`, `split`, full32 arm

| axis | decision axis | margin archived | margin v2 | `NI` |
|---|:--:|---:|---:|:--:|
| triviaqa | yes | −0.603544 | −0.603544 | REJECT (both) |
| popqa | yes | −4.539146 | −4.539146 | REJECT (both) |
| mmlu_content | yes | **+1.049530** | **+1.049530** | **ACCEPT (both)** |
| *nq_open (demoted)* | no | −3.657895 | −3.657895 | REJECT (both) |

Verdict object identical: `{"n_decision_axes_surviving_guard": 3, "n_decision_axes_accepting": 1,
"axes_accepting": ["mmlu_content"], "NI_OBSERVED_TO_ACCEPT": false}`.

**Checked across all four tie conventions (`split`/`first`/`last`/`credit`) and all three arms:
every ratio object and every cell is identical. Full-JSON comparison excluding only
timestamps/paths: identical.**

### Does the disagreement survive?

| | archived | re-scored |
|---|:--|:--|
| `RATIO(0.85)` accepts full32 | yes (0.8514950516430542) | **yes (identical)** |
| `NI` rejects on ≥2 of 3 decision axes | yes (2/3: triviaqa, popqa) | **yes (2/3, identical)** |
| **DISAGREEMENT** | **yes** | ### **YES — SURVIVES** |

---

## 4. My correction to the fragility framing — the premise was wrong in BOTH directions

The dispatch corrected the prior document's "0.1353 pp of measured harness jitter" (a
base-vs-`_v2` pair = code-version drift) down to a same-code bound of **0.0892 pp**, noting that
this sits *below* the 0.1226 pp flip threshold. **I checked that 0.0892 pp figure too, and it is
also not runtime jitter.** It comes from `shortgpt16_step200k` triviaqa:

| measurement | dir | triviaqa EM | driver `--batch_size` |
|---|---|---:|---:|
| STAGE (original) | `a04_shallow_stage/shortgpt16_step200k` | 33.00824788230049% | **32** (`_run_closedbook_8shard.sh`) |
| `_v2` | `olmo2_closedbook_results/7B_shortgpt16_step200000_v2` | 33.097414177440925% | **8** (`_run_olmo2_p24_eval_ladder_prev2_73.sh:189`) |
| `_v3` | `olmo2_closedbook_results/7B_shortgpt16_step200000_v3` | 33.097414177440925% | **8** (`_run_olmo2_within_disk_floor_v3.sh:183`) |

The 0.0892 pp spread is **STAGE(bs=32) vs v2/v3(bs=8)** — a **batch-size** difference, exactly the
pad-width numerics effect the dispatch warned me about. Meanwhile **`_v2` vs `_v3`** — two
independent runs at the *same* batch size on *different nodes* (`.73` vs `.104`) — agree to
**`0.33097414177440926` exactly, 0.0000 pp, 0 EM flips**.

So the corrected picture is: **there is no measured runtime-jitter floor on this harness at all.**
Every documented "jitter" number in A04's sensitivity file is either a code-version diff or a
batch-size diff. `memory/same-harness-runs-bit-identical` is confirmed and now extends to the
closed-book axes and across nodes.

> Small honest footnote: `_v2` vs `_v3` differ in **2 of 17,944 prediction strings** (item 9697,
> 17733 — both degenerate repetition loops on the same "drama queen" prompt, `em=0` under both), so
> 2 shard files differ byte-wise while the metric is unchanged. Non-determinism exists at the
> token level; it did not reach any metric. **On `full32` even that did not occur: 0 pred-string
> flips.**

### The sensitivity that actually matters (measured, not argued)

A bit-identical re-run bounds runtime jitter at 0.0 pp but says **nothing** about robustness to a
perturbation that *does* move items. So I ran the known item-moving perturbation — batch size —
as an explicitly **inadmissible probe** (the frozen protocol is bs=32; these numbers never enter a
verdict):

| axis | bs=32 (protocol) | bs=48 (probe) | Δ (pp) | EM flips |
|---|---:|---:|---:|---:|
| popqa | 0.18420130370785728 | 0.18406112006728814 | **−0.014018** | 12 / 14,267 |
| nq_open | 0.15817174515235458 | 0.15872576177285320 | **+0.055402** | 10 / 3,610 |

Recomputed through the imported `ratio_rule`:

| | `mean_ratio` | margin over ρ | accept |
|---|---:|---:|:--:|
| bs=32 (protocol) | 0.8514950516430542 | +0.0014950516 | yes |
| bs=48 (probe) | 0.8520291243733886 | **+0.0020291244** | yes |

**Under a real 22-item perturbation the margin WIDENS by +0.000534, it does not approach the flip
point.** The mechanism is worth recording: the two flip-critical axes moved in **opposite
directions**, so the 4-axis mean is materially more stable than any single axis. The prior
document's fragility table (§6.2) computed each axis's flip threshold **one axis at a time**, which
implicitly assumes the worst case that only the fragile axis moves. That is a legitimate worst case
but it is not what perturbations here actually do.

**What is still true**: `RATIO`'s margin is genuinely thin in absolute terms (+0.0015, i.e. ~3.3
nq_open items). A perturbation large enough, or adversarially aligned across axes, would flip it.
What is now false is the specific claim that *"a re-scoring of full32 on NQ-open or PopQA, within
measured harness noise, would flip `RATIO` from ACCEPT to REJECT"* — that re-scoring has been done
and it changed nothing whatsoever.

---

## 5. Consequences for A04, without dressing up

**What improved.** §4.3's disagreement — the one arm where "recovered" is defensible (97.7% MMLU
recovery), escaping the "disagreement is automatic because the arm is simply bad" defect that made
keep7/keep12 uninformative — is now **verified by independent re-scoring** and is no longer
"one re-scoring away from vanishing". §6.2 of
`A04_SHALLOW_RUNG_NI_DISCRIMINATION_VERDICT.md` should be read as **superseded**: its concern was
legitimate and correctly flagged, and the test it asked for has now been run and came back clean.

**What did NOT change, and must keep travelling with the claim.**

1. **`full32` is a ZERO-DAMAGE control.** The disagreement is about **continued-pretraining
   drift**, not about recovery from structural injury. It shows `RATIO` is too permissive
   (an 85% retention headline concealing 6.40 pp / 6.33 pp absolute deficits on 2 of 3 axes); it
   is **not** evidence about healing damaged models, which is A04's actual claim.
2. **Every *damaged* rung is still constant-REJECT** at 16.5–72.4 bootstrap SE, across two scales,
   50–87.5% depth kept, 5k–220k heal steps. Re-scoring did not touch that and cannot.
3. **The margin is thin in absolute terms** even though it did not move. Report `+0.0015`, not
   "robust".
4. **No 7B `sd_run` exists** (one seed per rung, seeds unrecorded). The cross-scale caveat of §5
   stands unchanged.

**So: A04 now has one verified, non-trivial rule disagreement — on a zero-damage control.** That
clears the specific fragility blocker, and does **not** clear "the rule discriminates among
damaged rungs", which remains the promotion question.

---

## 6. Trajectory scan (deliverable 5) — NOT run, and why, with the blocker measured

The dispatch asked for the intermediate `full32` checkpoints to be scored to locate where `NI`
transitions reject→accept. **I did not start it, because the assets are not where the dispatch
assumed, and I verified this rather than reporting it as done.**

```
zwfy6 (.73/.82/.104):  outputs/olmo2_probe2_7B_full32_dolmino/  ->  step25000.pt ONLY
wzc1  (LOCAL/.21):     step5000 step10000 step15000 step20000 step25000  (5 files, 81.6 GiB each)
```

`find outputs -name 'step*.pt' -path '*full32*'` on zwfy6 returns exactly one file. So the four
intermediate checkpoints are **wzc1-only** — and per the standing GPU budget, `LOCAL` and `.21`
are running SparseForge #246 until 08-13 ~11:10/~15:30 and are not mine to use.

Measured, not quoted from the handbook: I streamed 2 GiB and 3 GiB samples wzc1→`.73` via
`dd | ssh`. The 3 GiB sample took **183 s = 16 MiB/s**, so one 81.6 GiB checkpoint is **≈89 min**
and the four intermediates are **≈5 h of pure transfer** (326 GiB) **before** any GPU work — for
what is, per §5, a scan of a **zero-damage control's** trajectory. (16 MiB/s sits inside the
12–37 MB/s band this repository has measured before, so the handbook figure is confirmed rather
than assumed.)

**The recommendation is therefore to run it on `LOCAL`/`.21` when SparseForge frees them (the
checkpoints are already there — zero transfer), not to stage 326 GiB across disks.** Cost estimate
from *this* run's measurement: 4 ckpts × 3 axes at the same protocol ≈ **4 × 1248 s ≈ 1.4 h wall,
11 GPU-h**, plus MMLU-content which needs its own harness invocation.

**One thing the scan should be designed to expect**, from the numbers already in hand: `full32`
sits at 97.7% MMLU recovery with triviaqa only **1.86 SE** from accepting, and it is the *endpoint*
of the trajectory. Earlier checkpoints are *less* converged, so they should reject *harder* — the
accept boundary is most likely **beyond** step 25000 (or off this trajectory entirely), not
between 5000 and 25000. A scan of steps 5000–20000 will most likely map a monotone approach to the
boundary without crossing it. That is still worth having — it is the "does the rule discriminate"
curve — but it should not be sold in advance as likely to *find* an accept.

---

## 7. Provenance

| item | path |
|---|---|
| comparison code (new) | `proposal/active/A04-recovery-certification/code/a04_full32_rescore_v2_compare.py` |
| analysis code (2 inert optional flags added) | `proposal/active/A04-recovery-certification/code/a04_shallow_rung_ni_7b.py` |
| **this run's results** | `proposal/active/A04-recovery-certification/evidence/a04_full32_rescore_v2.json` |
| full NI re-run with `_v2` substituted | `proposal/active/A04-recovery-certification/evidence/a04_shallow_rung_ni_7b_full32v2.json` |
| re-scored shards (CB) | `zwfy6:olmo2_closedbook_results/full32_step25000_v2/` (8/8 × popqa, triviaqa) |
| re-scored shards (NQ) | `zwfy6:olmo2_closedbook_results/full32_step25000_nqopen_v2/` (8/8) |
| bs=48 probe shards | `zwfy6:olmo2_closedbook_results/full32_step25000{_nqopen,}_bs48/` |
| run logs | `zwfy6:logs/a04_v2_rescore_MAIN.out`, `logs/cb_full32_step25000{,_nqopen}_v2_sched.out`, `logs/a04_bs48_probe_MAIN.out` |
| driver | `zwfy6:a04_v2_rescore.sh`, `a04_bs48_probe.sh` |
| archived comparanda | `zwfy6:olmo2_closedbook_results/full32_step25000{,_nqopen}/` (**not** modified) |

**Reproduce** (on `.73`, `PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`):

```bash
R=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
# (1) the re-scoring, protocol-identical to the archive
bash a04_v2_rescore.sh
# (2) control: NO overrides -> must reproduce evidence/a04_shallow_rung_ni_7b.json
$PY proposal/active/A04-recovery-certification/code/a04_shallow_rung_ni_7b.py \
  --raw_root $R \
  --shortgpt_cb $R/a04_shallow_stage/shortgpt16_step200k \
  --shortgpt_nq $R/a04_shallow_stage/shortgpt16_step200k_nqopen \
  --d5_cb $R/a04_shallow_stage/D5_intact_wzc1_cb \
  --d5_mm $R/a04_shallow_stage/D5_intact_wzc1_mm \
  --out_json /tmp/a04v2/control_reproduce.json
# (3) same script, only the full32 arm's dirs repointed
$PY proposal/active/A04-recovery-certification/code/a04_shallow_rung_ni_7b.py \
  --raw_root $R \
  --shortgpt_cb $R/a04_shallow_stage/shortgpt16_step200k \
  --shortgpt_nq $R/a04_shallow_stage/shortgpt16_step200k_nqopen \
  --d5_cb $R/a04_shallow_stage/D5_intact_wzc1_cb \
  --d5_mm $R/a04_shallow_stage/D5_intact_wzc1_mm \
  --full32_cb full32_step25000_v2 --full32_nq full32_step25000_nqopen_v2 \
  --out_json /tmp/a04v2/ni_with_v2.json
# (4) the diff + evidence json  (runs anywhere, CPU only)
python3 proposal/active/A04-recovery-certification/code/a04_full32_rescore_v2_compare.py \
  --archived_json proposal/active/A04-recovery-certification/evidence/a04_shallow_rung_ni_7b.json \
  --control_json /tmp/a04v2/control_reproduce.json \
  --v2_json /tmp/a04v2/ni_with_v2.json \
  --bs48_probe_json /tmp/a04v2/bs48_probe.json \
  --ratio_bs48_json /tmp/a04v2/ratio_bs48_sensitivity.json \
  --out_json proposal/active/A04-recovery-certification/evidence/a04_full32_rescore_v2.json
```

**Nothing was tuned to save the finding.** The protocol was fixed from the archive's logs before
any number was produced, the analysis script's defaults still point at the archived dirs, and the
one perturbation I *chose* (batch size) was the one most likely to break the result, not least
likely. The outcome happens to be favourable; the bs=48 probe was run and reported even though a
bit-identical re-run already technically discharged the dispatch.
