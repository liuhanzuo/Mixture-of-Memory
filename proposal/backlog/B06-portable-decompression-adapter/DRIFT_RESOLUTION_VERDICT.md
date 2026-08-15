# B06 — Drift Resolution Verdict

**Leg**: `next_gate` "Same-harness rejudge of the canonical HCache predictions to remove the
8.11-vs-13.29 cross-node drift".
**Date**: 2026-08-15 · **Cost**: 0 GPU-h, 0 ssh, **0 judge API calls** · **Commit**: `b0c5ea1`
**Everything below is recomputed from per-item raw records** by
`evidence/build_drift_evidence.py` → `evidence/drift_resolution_evidence.json`.
Nothing is copied from `STATUS.json`. Where the two disagree, the recomputation wins and the
disagreement is stated explicitly.

---

## Verdict in one paragraph

**The "cross-node drift" is not node drift and not generation drift — it is a judge-side
(instrument) artefact, and the mixed-instrument scale trap was hiding the fact that the drift is
*larger*, not smaller, than recorded.** On the identical `Judge_1:4` (n=1540) ruler the three
no-LoRA HCache replicates disagree by **64.7% relative** under the GPT-4o judge, while **F1 and
accuracy on the very same 1,540 items agree to 0.16% and 4.2% relative**. F1 and accuracy are
deterministic functions of the predictions alone — no judge, no API, no sampling — so a real
difference in generation quality *must* move them. Calibrating the judge-vs-lexical slope on a
change known to be real (the +LoRA arm) bounds the generation-attributable share of the 6.56 pp
drift at **0.4%–7.8%**; the remaining **92%–99.6% is judge-side**. A rejudge is therefore
**not needed and would not answer the question anyway**, because the GPT-4o endpoint is not a
frozen instrument (measured: **3.07% per-item verdict instability on byte-identical inputs**).
**Kill condition 2 does not fire** (+23.12 pp effect vs ≤0.5 pp generation-attributable drift).
**Kill condition 1 does not fire either** — and the reason it previously looked at risk is a
**category mislabel** inherited from the eval script.

---

## 1. The four numbers, recomputed

The gate's SCALE TRAP warning is **correct and confirmed**: `8.11` and `13.29` live on a
**mixed-instrument n=1986** ruler; `16.69` / `39.81` / `10.13` / `15.45` live on the
**single-instrument `Judge_1:4` n=1540** ruler. Mixing them compares two rulers.

| # | number | published at | **recomputed** | n | instrument | verified |
|---|---|---|---:|---:|---|---|
| 1 | **8.11** canonical HCache | `status/PAPERA_RESULTS_CONSOLIDATED.md:175` | **8.106747** | 1986 | **MIXED** (1540 GPT-4o + 446 refusal regex) | ✅ |
| 2 | **10.13** its `Judge_1:4` counterpart | `status/PAPERA_RESULTS_CONSOLIDATED.md:175` | **10.129870** | 1540 | single (GPT-4o) | ✅ |
| 3 | **16.69** B06 control (noLoRA) | `established_measurements` | **16.688312** (257/1540) | 1540 | single (GPT-4o) | ✅ |
| 4 | **39.81** B06 treatment (+LoRA) | `established_measurements` | **39.805195** (613/1540) | 1540 | single (GPT-4o) | ✅ |
| 5 | **15.4545** third replicate | `third_measurement_found_20260814` | **15.454545** (238/1540) | 1540 | single (GPT-4o) | ✅ |

**Rows 3/4/5 are recomputed from raw per-item records** —
`locomo_results/{hcache_j12_noLoRA_chatFALSE,hcache_j12_LoRA_chatFALSE,hcache}/`
`{preds_shard*.jsonl, judge_cache.jsonl}`. Every hard property was asserted programmatically, not
assumed: 1,986 unique pred ids; exactly 1,540 judge-cache records with **0 duplicate ids**; all
records `model='gpt-4o'`; the judge-cache id set is **exactly** the non-abstention id set; the
abstention set is **exactly** the cat-5 set; the recomputed blend reproduces `scores.json`
`overall_judge` **to 1e-9**; all 16 per-category cells match published to 1e-9.

**Rows 1/2 could not be read from raw records** — see §5. They are pinned by **three mutually
independent routes that all agree**:

| route | input | result |
|---|---|---|
| A | invert each published per-cat % (`LOCOMO_JUDGE_AGGREGATE.md:41`) | cat1-4 → 19+8+9+120 = **156**; cat5 → **5**; each count **unique** |
| B | solve `(x+5)/1986 → 8.11` | **x = 156**, unique |
| C | solve `x/1540 → 10.13` | **x = 156**, unique |

→ `100·(156+5)/1986 = 8.1067` and `100·156/1540 = 10.1299`.
**The gate's instruction to convert 8.11 before comparing was already done: 10.13 is that
conversion, and it is confirmed.** No spending was required to establish this.

---

## 2. The gate's core question: how much drift survives one ruler?

**It does not shrink. It grows by 26%.**

| pairing | blended n=1986 | **`Judge_1:4` n=1540** |
|---|---:|---:|
| canonical vs B06 control | +5.1863 pp | **+6.5584 pp** |
| canonical vs older local | +4.1793 pp | +5.3246 pp |
| older local vs B06 control | +1.0071 pp | +1.2338 pp |

Switching to the single instrument *inflates* the gap, because the constant cat-5 term
(5–7 correct out of 446) deflates the two endpoints unequally. So the scale trap was not
concealing a comparison that would vanish — it was concealing that **the drift is bigger than
`STATUS.json` records** (see §4, discrepancy 1).

---

## 3. Attribution: judge-side, not generation-side

### 3.1 The decisive test — judge-independent metrics

F1 and accuracy are computed from the prediction strings alone. If the canonical run's generations
were genuinely worse, F1/acc must drop with the judge score.

| replicate (n=1540 ruler) | F1₁:₄ | acc₁:₄ | GPT-4o Judge₁:₄ |
|---|---:|---:|---:|
| canonical `hcache_8b_chatFALSE` | 5.6978 | 7.7922 | **10.1299** |
| older local `locomo_results/hcache` | 5.7070 | 7.8571 | **15.4545** |
| B06 control `hcache_j12_noLoRA_chatFALSE` | 5.7022 | 8.1169 | **16.6883** |
| **relative spread (max−min)/min** | **0.16 %** | **4.17 %** | **64.74 %** |

Two judge-free metrics say these three runs are the same measurement. The judge says one of them
is 65% different. **Per-category over the four judged categories, the largest canonical-vs-local
difference is 0.117 pp (F1) and 0.623 pp (acc)**, while those same metrics move 0.12–5.09 pp (F1)
and 2.08–22.59 pp (acc) for the real +LoRA change. (cat5 is excluded throughout: it is the
regex-graded category that `Judge_1:4` deliberately drops.)

### 3.2 Quantifying the bound

Calibrate the judge-vs-lexical slope on a change known to be real — the +LoRA contrast, same node,
same commit, same judge pass, LoRA the only variable:

- Judge₁:₄ +23.1169 pp → F1₁:₄ +3.8376 pp, acc₁:₄ +14.6753 pp
- ⇒ **0.1660** F1 pp / judge pp, **0.6348** acc pp / judge pp

Apply to the 6.5584 pp canonical drift:

| | predicted if real | **observed** | observed/predicted |
|---|---:|---:|---:|
| F1₁:₄ movement | +1.0887 pp | **+0.0045 pp** | **0.4 %** |
| acc₁:₄ movement | +4.1635 pp | **+0.3247 pp** | **7.8 %** |

→ **generation-attributable: 0.4 %–7.8 % (≤0.51 pp). Judge-attributable: 92.2 %–99.6 %.**

### 3.3 Corroboration — the canonical GPT-4o score is the outlier

`locomo_results_openjudge_qwen3_MIRROR/` holds a **second judge** (open-weight Qwen3-8B,
non-thinking, greedy) applied to the **same six canonical baseline prediction sets**. The
open/GPT-4o ratio on the same n=1540 ruler:

| method | GPT-4o Judge₁:₄ | open Judge₁:₄ | ratio |
|---|---:|---:|---:|
| CoMem flagship | 48.64 | 66.43 | 1.366 |
| KV-Direct | 43.83 | 64.09 | 1.462 |
| StreamingLLM | 31.04 | 51.04 | 1.644 |
| InfLLM | 26.43 | 44.42 | 1.681 |
| MemoryLLM | 20.65 | 32.60 | 1.579 |
| **HCache (canonical)** | **10.13** | **34.48** | **3.404** |

Five siblings span **1.366–1.681** (mean 1.546, sd 0.131). HCache sits at **3.404 = z +14.2**,
**2.03× the next highest**. Under the open judge HCache **overtakes MemoryLLM**; under GPT-4o it
is last by a wide margin. Mapping HCache's open score through the sibling ratios implies a GPT-4o
Judge₁:₄ of **20.5–25.3**, above *both* local replicates (15.45 / 16.69) and far above 10.13.
→ **The canonical GPT-4o HCache cell is the anomaly; the local replicates are not the deviants.**

### 3.4 Judge noise floor — measured, not assumed

`hcache` (generated 07-09/10, judged 07-18) and `hcache_j12_noLoRA_chatFALSE` (generated + judged
07-25) share the same 1,540 ids. **879/1540 (57.1%) of prediction strings are byte-identical**, so
on that subset the judge input is literally the same string.

- verdict flips on byte-identical input: **13 (0→1) + 14 (1→0) = 27/879 = 3.07 %**, **symmetric**, net **−1**
- ⇒ per-item flip prob 0.0307; sd of net aggregate change at n=1540 = **±6.88 items (±0.447 pp)**

| observed gap | items | σ |
|---|---:|---:|
| canonical vs B06 control | 101 | **+14.7 σ** |
| canonical vs older local | 82 | **+11.9 σ** |
| older local vs B06 control | 19 | **+2.8 σ** |

So per-call noise **fully explains the 1.23 pp local-vs-local wobble** (decomposition: **net −1
from the 879 identical-prediction items, net +20 from the 661 differing ones**) but **cannot
explain the canonical gap** — that is systematic.

### 3.5 Why no rejudge

1. **It would not be same-instrument.** At 3.07% per-item instability, a fresh GPT-4o pass is a
   *new instrument draw*, not a fixed ruler. "Same-harness rejudge" is not achievable against a
   non-deterministic remote endpoint.
2. **The question is already answered** by judge-free columns at zero cost and zero API risk.
3. **The canonical per-item records are not on this disk** (§5), so the rejudge the gate describes
   could not even be executed under this task's constraints.

**Conclusion: `rejudge_needed = false`. The drift leg closes at 0 GPU and 0 API calls.**

---

## 4. Discrepancies found in `STATUS.json`

### ⚠️ Discrepancy 1 — `condition_2_status` compares the wrong pair (understates the drift 5–6×)

> "the drift is ~1.0 pp on the blended scale (**12.286 canonical** vs 13.293 local, both from
> scores.json overall_judge)"

**`12.286` is not canonical.** It is `locomo_results/hcache`, the **older local** run —
`STATUS.json`'s *own* key `third_measurement_found_20260814` labels that identical value
`12.28600201409869` as "older run (2026-07-09/10)". The canonical blended value is **8.1067**.
So `condition_2_status` compares **local vs local** and calls one of them canonical.

**Correct**: canonical vs B06 control = **5.1863 pp blended / 6.5584 pp on `Judge_1:4`** —
**5.2×–6.5× larger** than the stated "~1.0 pp".
**Conclusion unchanged**: still small vs +23.12 pp, and now shown to be judge-side.

### ⚠️⚠️ Discrepancy 2 — `condition_1_status` mislabels cat4 (**conclusion-changing**)

> "cat4 (**open_domain**, n=841 = 55% of the 1540) … `'最大驱动'` … if the gain is cat4-only this
> condition fires."

**cat4 is SINGLE-HOP, not open-domain.** Measured from `locomo/data/locomo10.json`:

| cat | n | mean #evidence | % exactly 1 ev | % 0 ev | mean distinct sessions | % Q starts "When/What year" | ⇒ label |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 282 | 3.13 | 2.1 % | 0 % | 2.67 | 1.4 % | multi-hop |
| 2 | 321 | 1.17 | 87.5 % | 0 % | 1.10 | **77.6 %** | temporal |
| 3 | 96 | 2.08 | 46.9 % | **4.2 %** | 1.66 | 0 % | **open-domain** |
| 4 | 841 | **1.07** | **94.5 %** | 0 % | **1.00** | 0.8 % | **single-hop** |
| 5 | 446 | 1.03 | — | 0 % | 1.00 | 1.1 % | adversarial |

cat4 cites one evidence turn in one session — the definition of single-hop. cat3 is the only
category with zero-evidence (inference) items. The mislabel is **inherited from
`scripts/eval_qcmem_locomo.py:126-132`**, whose `CATEGORY_NAMES` is wrong for cats 2/3/4 and
contradicts `status/LOCOMO_JUDGE_AGGREGATE.md:31-32` (which is right for cat3/cat4).

**This flips kill condition 1 from "the one at real risk" to "does not fire".** Per-category
McNemar on the corrected instrument:

| cat | label (data-grounded) | n | noLoRA | +LoRA | Δ within-cat | contribution | share of gain | b/c | exact p | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 1 | multi-hop | 282 | 10.64 | 25.89 | +15.25 | +2.79 pp | 12.1 % | 52/9 | 1.80e-08 | **sig +** |
| 2 | temporal | 321 | 4.98 | 13.40 | +8.41 | +1.75 pp | 7.6 % | 35/8 | 4.19e-05 | **sig +** |
| 3 | **open-domain** | 96 | 15.62 | 29.17 | +13.54 | **+0.84 pp** | **3.7 %** | 16/3 | 4.43e-03 | **sig +** |
| 4 | single-hop | 841 | 23.31 | 55.77 | +32.46 | +17.73 pp | 76.7 % | 311/38 | 1.99e-54 | **sig +** |

**All four judged categories are individually significant at p<0.05** (contributions sum to
23.1169 pp exactly). The kill clause reads *"只在 LoCoMo open-domain category 有益"* — and
open-domain (**cat3**) is the **smallest** contributor at 3.7% of the gain. **Kill condition 1
DOES NOT FIRE.**

### Discrepancy 3 — the rejudge in `next_gate` / `gpu_cost_estimate` is unnecessary

Not "0 GPU-h + one API call" but **0 GPU-h + zero API calls**. See §3.5.

### Discrepancy 4 — a same-run artefact **was** on wzc1 and was missed

`canonical_8_11_conversion_status.what_is_still_missing` says the canonical records "live on zwfy6
… NOT read here". True of the judge cache — but
**`locomo_results_openjudge_qwen3_MIRROR/hcache_8b_chatFALSE/` is on wzc1** (mirrored per
`paperA/TODOList.md:170`). Its judge-independent columns match the published canonical row
exactly (`f1 4.67 / acc 6.29 / em 0.25`), and its cat-5 cell recovers **exactly 5/446**
independently — a fourth confirmation of the 8.11 arithmetic, and the basis of §3.3.

### Discrepancy 5 — the clustered bootstrap flagged as missing is now done

`caveat_from_errata` asks for a dependence-aware interval. Done here, 0 GPU:
**conversation-clustered paired bootstrap (10 conversations, 10,000 resamples, seed 1) =
[20.53, 25.91] pp**, with **0/10,000 resamples ≤ 0**. Nearly identical to the per-item interval
[20.58, 25.58] — the clustering caveat does **not** threaten this effect.

---

## 5. What I did **not** verify

1. **The canonical run's per-item records were never read.** They are on zwfy6
   (`locomo_results/hcache_8b_chatFALSE/`); this task forbade ssh, and zwfy6 is not mounted here
   (`ls /apdcephfs_zwfy6` → No such file or directory). An **exhaustive scan of all 32
   `judge_cache*.jsonl` and all 67 `scores.json` under the wzc1 root** found no run with
   156/1540. The canonical counts are **arithmetic recovery** (three agreeing routes + a same-run
   artefact), **not** a read of raw records.
2. **The mechanism behind the 101-item deficit is not confirmed.**
   `scripts/eval_qcmem_locomo.py:713-715` sets `item["judge"] = 0.0` on judge-API failure **and
   does not write that record to the cache**. 101 silent failures would produce exactly this
   deficit and leave the canonical `judge_cache.jsonl` with **~1439 records instead of 1540**.
   **This is the single highest-value 0-GPU follow-up**: one `wc -l` on zwfy6 decides it. I am
   *not* claiming this is the cause — only that it is consistent and cheaply testable.
3. **Byte-equality of the canonical vs local predictions** was not checked (canonical preds are on
   zwfy6). The judge-free metrics agreeing to 0.16% relative bounds any difference as immaterial,
   but that is not byte equality.
4. **The cross-judge test uses published GPT-4o per-category percentages** for the five siblings
   (`LOCOMO_JUDGE_AGGREGATE.md:34-41`), not their raw caches — four of the six sibling judge
   caches are not on wzc1. The HCache and MIRROR sides are recomputed.
5. **Kill condition 3 (second compressor) is untouched** and needs GPU.
6. **No portability claim.** Still one task, one compressor, one model. `claim_scope_discipline`
   in `STATUS.json` stands unchanged.
7. The **older-local vs B06-control 1.23 pp** difference *is* partly real generation difference
   (661/1540 predictions differ, net +20 items from those). Only the **canonical** gap is
   attributed to the judge.

---

## 6. Kill-gate status after this leg

| condition | before | **after** |
|---|---|---|
| 1. 只在 open-domain category 有益 | "PARTIALLY TESTABLE … the one at real risk" | **DOES NOT FIRE** — all 4 judged cats individually significant; open-domain (cat3) is the *smallest* contributor (3.7%) |
| 2. 统一 harness 后增益消失 | OPEN (rejudge not run) | **DOES NOT FIRE** — drift is judge-side; ≤0.51 pp generation-attributable vs +23.12 pp effect. **Closed without GPU or API.** |
| 3. 换 compressor 完全不迁移 | UNTESTED | **UNTESTED** (needs GPU) |

**Headline re-verified from raw records** (unchanged): noLoRA **16.6883** (257/1540) → +LoRA
**39.8052** (613/1540), gain **+23.1169 pp**, McNemar b=414/c=58, exact two-sided
**p = 2.5747e-67**, χ² (continuity-corrected) 267.0021, per-item 95% CI **[20.58, 25.58]**,
conversation-clustered 95% CI **[20.53, 25.91]**.

**Remaining blocker for promotion is unchanged and is not the drift**: `novelty_checked = false`
plus kill condition 3. The drift was never the real risk — and it is now closed.

### 6.1 Scheduler check (and one stale claim I did not touch)

`python3 proposal/ready_queue.py` before and after my append is **byte-identical output**: B06 stays
in **`ready_cpu`**, and is **not** in `ready_gpu`. Verified two further ways — my new key
`drift_resolution_leg_20260815` matches none of `BLOCK_KEYS`, `NOVELTY_VERDICT_KEYS`,
`NEXT_GATE_KEYS`, `KILL_KEYS`, nor any `NESTED_BLOCKER_CONTAINERS` prefix; and `read_one()` still
returns `lifecycle=ready_cpu`, `novelty_checked=False`.

Two observations left deliberately **unchanged**, as they belong to a different leg:

1. **`novelty_status_detail` is now stale.** It says *"RELATED_WORK.md does NOT exist here"*, but
   `RELATED_WORK.md` (28,761 bytes) was committed the same day in `463dca4`
   ("prereg(B06/B07/B08): RELATED_WORK novelty adjudication"). `ready_queue.read_one()` already
   reports `related_work_md = True`. I did not edit that key — it is append-only, and the field is
   that leg's to correct.
2. **Do not "fix" `novelty_checked` casually.** `RELATED_WORK.md` §6.1 documents a measured
   scheduler trap: appending `related_work_status: "audited"` to `STATUS.json` flips B06 to
   **`ready_gpu`** (`ready_queue.py:244-251` treats `"audited"` as cleared, and B06 has no other
   live blocker). That sibling leg deliberately left `STATUS.json` unmodified for exactly this
   reason. **This leg did not add any novelty-verdict key**, so the trap remains unsprung.
   B06 must not reach the GPU queue on an agent-written field without adversarial review.

---

## Reproduce

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
python3 proposal/backlog/B06-portable-decompression-adapter/evidence/build_drift_evidence.py
```

CPU only, no network, no GPU, ~30 s. Verified **byte-identical across two independent runs**
(excluding the `_generated` timestamp). Output:
`evidence/drift_resolution_evidence.json`.

Inputs (sha256 recorded in the JSON):
`locomo_results/{hcache,hcache_j12_noLoRA_chatFALSE,hcache_j12_LoRA_chatFALSE}/{preds_shard*.jsonl,judge_cache.jsonl,scores.json,eval_config_shard0of3.json}`,
`locomo_results_openjudge_qwen3_MIRROR/*/scores.json`, `locomo/data/locomo10.json`.
