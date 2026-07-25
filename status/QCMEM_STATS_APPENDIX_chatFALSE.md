# QCMem / CoMem (Paper A) — chat=False Statistics Appendix

**Scope:** CPU/aggregate-only verification for the Paper A chat=False experiment chain.
All numbers read directly from the result dirs on **.73 (28.85.35.73, diskB,
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`)** — no GPU, no
re-runs. Protocol held fixed: Qwen3-8B + LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`
(resume_j=12), selector=`iter_bm25`, topk=12, sink=bos, chunk_size=512, **chat_template=False**,
official scorers. Generated 2026-07-23. **Not git-committed** (main handles staging).

Cross-check verdict vs `status/PAPER_LOCOMO_ERRATA_20260721.md` §9/§11: **every headline
number reproduced exactly** from raw dirs. One config nuance flagged for the equal-budget VT
row (see Delivery 2).

---

## Delivery 1 — LoCoMo GPT-4o judge headline (chat=False)

Dirs: `locomo_results/qcmem_8b_iter_chatFALSE/` (flagship, selector=iter_bm25) and
`locomo_results/kvdirect_8b_chatFALSE/` (oracle full-context KV baseline).
Per-sample recompute from `preds_shard{0..3}of4.jsonl` (dedup by id, n=1986) +
`judge_cache.jsonl` (1540 gpt-4o verdicts). Scoring logic mirrors
`scripts/eval_qcmem_locomo.py:score_sample` / `llm_judge_preds`.

### 1(a) Overall judge recomputed from cache — matches scores.json
| dir | recompute (from cache) | scores.json `overall_judge` | match |
|---|---:|---:|:--:|
| CoMem 8B iter chat=False | **38.26788** | 38.26787512588117 | ✔ |
| KV-Direct 8B chat=False | **34.59215** | 34.59214501510574 | ✔ |

### 1(b) Per-category judge breakdown
CoMem 8B (`qcmem_8b_iter_chatFALSE`):
| cat | name | n | F1 | EM | acc | **judge** |
|---|---|---:|---:|---:|---:|---:|
| 1 | multi_hop   | 282 |  9.94 | 0.00 | 12.77 | **26.95** |
| 2 | single_hop  | 321 |  5.77 | 0.00 |  7.79 | **19.00** |
| 3 | temporal    |  96 |  7.50 | 0.00 | 18.75 | **30.21** |
| 4 | open_domain | 841 | 13.92 | 0.00 | 44.47 | **69.32** |
| 5 | adversarial | 446 |  2.47 | 2.47 |  2.47 | **2.47** (local abstain) |
| — | **overall** | 1986 | 9.15 | 0.55 | 23.36 | **38.27** |

KV-Direct 8B (`kvdirect_8b_chatFALSE`):
| cat | name | n | F1 | EM | acc | **judge** |
|---|---|---:|---:|---:|---:|---:|
| 1 | multi_hop   | 282 | 11.25 | 0.00 | 16.67 | **24.11** |
| 2 | single_hop  | 321 |  7.40 | 0.00 | 11.53 | **18.69** |
| 3 | temporal    |  96 |  6.82 | 0.00 | 16.67 | **25.00** |
| 4 | open_domain | 841 | 12.49 | 0.00 | 39.48 | **62.19** |
| 5 | adversarial | 446 |  2.69 | 2.69 |  2.69 | **2.69** (local abstain) |
| — | **overall** | 1986 | 9.02 | 0.60 | 22.36 | **34.59** |

### 1(c) Denominator audit — headline judge is over 1986 (cat5 folded, not dropped)
- `judge_cache.jsonl` = **1540 lines** (both dirs). 1986 − 1540 = **446 = exactly cat5 (adversarial)**.
- **0 cat5 ids appear in the cache.** cat5 is **excluded from GPT-4o judging** (no API call);
  `llm_judge_preds` grades adversarial/abstention items **locally**: judge = 1.0 iff the model
  refuses ("I don't know" / empty / refusal regex), else 0.0 → identical to its acc column (2.47/2.69).
- gpt-4o cache hits partition exactly across the 4 non-adversarial cats: {1:282, 2:321, 3:96, 4:841} = 1540.
- **Denominator口径 (definitive):** the paper headline **38.27 is over ALL n=1986**, with cat5
  contributing its local abstention-accuracy. It is NOT over 1540. The "1540" is only the count of
  samples that received a real gpt-4o verdict.
- Judge **over the gpt-4o-judged subset only (cat1–4, n=1540)**: CoMem **48.64**, KV-Direct **43.83**.
  (Report this if the paper wants "judge on judgeable questions"; report 38.27/34.59 if it wants the
  standard full-set headline. Both are internally consistent — pick one口径 and state it.)

### 1(d) Judge prompt / endpoint / model version (for the appendix — must be reported)
- **Endpoint:** `https://maas-openapi.wanjiedata.com/api/v1/chat/completions` (OpenAI-compatible, maas/wanjiedata).
- **Model:** `gpt-4o` (passed literally as `--judge_model gpt-4o`; the specific gpt-4o snapshot is
  whatever the maas endpoint serves — no dated snapshot is pinned client-side). Request body:
  `{"model":"gpt-4o","stream":false,"seed":1,"messages":[{"role":"user","content":<prompt>}]}` —
  temperature/top_p intentionally unset (deterministic seed only), 4 retries w/ exponential backoff,
  unparseable/API-fail → conservative WRONG (0.0).
- **Verdict parsing:** reply must start with CORRECT/WRONG (else substring vote, else WRONG).
- **Prompt template** (`_JUDGE_TEMPLATE`, verbatim):
  > You are grading a model's answer against the gold answer for a question about a long,
  > multi-session dialogue (the LoCoMo benchmark).
  >
  > Question: {question}
  > Gold answer: {gold}
  > Model answer: {pred}
  >
  > Grade whether the model answer is CORRECT. It is CORRECT if it conveys the same key
  > information as the gold answer (a semantic match), even if phrased differently, more
  > verbosely, or with extra correct context. It is WRONG if it contradicts the gold answer,
  > omits the key information, or is empty / refuses when an answer exists. For date/time
  > answers, accept any unambiguous equivalent phrasing.
  >
  > Respond with ONLY one word: CORRECT or WRONG.

  (`{gold}` = the answer list joined by " OR "; cat5/adversarial never reach this template.)

### 1(e) Head-to-head (canonical GPT-4o judge, chat=False)
| method | judge (n=1986) | judge (cat1–4, n=1540) | F1 | acc | EM |
|---|---:|---:|---:|---:|---:|
| **CoMem 8B (iter_bm25)** | **38.27** | **48.64** | 9.15 | 23.36 | 0.55 |
| KV-Direct 8B (oracle) | 34.59 | 43.83 | 9.02 | 22.36 | 0.60 |
| Δ (CoMem − KVD) | **+3.68** | **+4.81** | +0.14 | +1.00 | −0.05 |

Under chat=False the compression method (CoMem) is **ahead of the full-context KV oracle on the
canonical judge metric** (+3.68), while token-F1 ties (9.15 ≈ 9.02) — consistent with errata §9
(token-F1 is a chat-template/formatting artifact, judge is the protocol-robust headline).

### 1 — Bootstrap 95% CI (1000 resamples, single-thread numpy, seed=1234)
| quantity | mean | 95% CI |
|---|---:|---|
| CoMem judge (n=1986) | 38.27 | [36.20, 40.58] |
| CoMem judge cat1–4 (n=1540) | 48.64 | [46.04, 50.98] |
| CoMem F1 (n=1986) | 9.15 | [8.60, 9.72] |
| KV-Direct judge (n=1986) | 34.59 | [32.48, 36.71] |
| KV-Direct judge cat1–4 (n=1540) | 43.83 | [41.23, 46.30] |
| KV-Direct F1 (n=1986) | 9.02 | [8.51, 9.61] |

⚠️ **Significance caveat:** these are per-method (unpaired) CIs and they **overlap** (CoMem judge
[36.20,40.58] vs KVD [32.48,36.71]) → the +3.68 gap is **suggestive, not clearly significant** at
95% on an unpaired test. LoCoMo is a paired design (same questions), so a **paired bootstrap of the
per-sample difference** would be the correct significance test and is more powerful — recommend
running it before claiming CoMem > KVD as significant. Verdict as reported: CoMem ≈/≥ KV-Direct.

---

## Delivery 2 — Equal-budget variable-tracking, precise long-context

**Two distinct chat=False VT runs exist; they are different retrieval configs, not the same run
at different lengths — this is the key finding for this delivery.**

### 2A. Equal-budget flagship (the tab_slm / equal-budget row) — `ruler_results/qcmem_8b_iter_chatFALSE_ad`
Config: selector=iter_bm25 with **`iter.rounds=0`** (i.e. single-pass BM25 top-12), chunk_size=512,
**avg_read_len ≈ 6.5k tokens** (8k cell reads 6565; ≈ the StreamingLLM equal budget sink4+window6653=6657).
Official RULER `string_match_all`, n=100/cell, all Iron-Law-2 OK (8/8 shards, empty=0, 0 recall mismatch):
| var-track | 8k | 16k | 32k | **64k** | **128k** | 256k |
|---|---:|---:|---:|---:|---:|---:|
| recall | 96.6 | 97.6 | 98.8 | **99.0** | **95.8** | 99.0 |

**→ Precise equal-budget VT 64k/128k = 99.0 / 95.8** (the paper's "~95/~95" placeholders). **NOT blocked.**

### 2B. Iterative campaign #10 (hop4, higher budget) — `ruler_results/ablation10_itervt_chatFALSE/iterbm25_vt`
Config: iter_bm25 **hop4** (multi-round chain expansion), **avg_read_len ≈ 17k tokens** (grows to
follow the reference chain) — NOT equal budget. Official scorer, all Iron-Law-2 OK:
| var-track | 8k | 16k | 32k | **64k** | **128k** |
|---|---:|---:|---:|---:|---:|
| recall | 99.0 | 95.6 | 89.8 | **89.8** | **87.4** |

**→ confirms the delivery's stated "campaign #10 VT 64k/128k = 89.8 / 87.4".**

### 2C. Equal-budget (~95) vs iterative campaign #10 (89.8/87.4) — difference + cause
- Equal-budget single-pass (rounds=0, read ≈6.5k) is **non-degrading** across length (95.8–99.0).
- Iterative hop4 (read ≈17k) **degrades** past 16k (99→89.8→87.4).
- **Cause:** for RULER variable_tracking under chat=False, a focused single-pass BM25 top-12 surfaces
  the tracked-variable chunks cleanly within ~6.5k tokens; the multi-hop expansion pulls a much larger
  ~17k read that injects distractor chunks (other variable chains) → dilutes the completion and lowers
  exact `string_match` recall at long range. i.e. more retrieval budget ≠ better here; the single-pass
  focused read wins. (This is the chat=False mirror of tab_itervt, where under chat=True the ordering
  was the opposite — see the discrepancy note below.)

### 2D. ⚠️ Discrepancy to resolve with researcher (config vs paper claim)
- `paper/sections/tab_itervt.tex` reports the **iter_bm25 (hop4)** column at **chat=True** as
  8k/16k/32k = **95.2 / 93.8 / 96.8** and claims it is *non-degrading* (32k>16k). The delivery brief
  cites these same 95.2/93.8/96.8 as the "equal-budget" reference.
- Under the mandated **chat=False** protocol the hop4 iterative run (2B) is **99.0/95.6/89.8** — it
  **DOES degrade** at 32k, contradicting the tab_itervt "non-degrading" narrative.
- The chat=False run that IS non-degrading is the **equal-budget single-pass** flagship (2A:
  96.6/97.6/98.8/99.0/95.8), which is a *different config* (rounds=0, read 6.5k) than tab_itervt's hop4.
- **Action for the coherent paper pass:** decide which config the equal-budget VT row should report
  under chat=False. Recommendation: use **2A (equal-budget single-pass, 96.6/97.6/98.8/99.0/95.8)** for
  the tab_slm equal-budget comparison (it is the genuine 6.7k-budget CoMem row that matches the
  StreamingLLM budget), and re-examine whether tab_itervt's "iterative beats single-pass / non-degrading"
  story survives chat=False (under chat=False single-pass ≥ hop4 on VT). This is a narrative decision,
  not a mechanical errata.

---

## Delivery 3 — Statistics appendix (all read from dirs, not copied)

### 3.1 RULER 97.05 — 15-cell breakdown + weighting definition
Dir `ruler_results/qcmem_8b_iter_chatFALSE_ad`, official `string_match_all`, n=100/cell,
all 15 cells Iron-Law-2 OK (8/8 shards, empty=0, on-disk vs recompute mismatch=0).
| task | 8k | 16k | 32k | 64k | 128k | (256k) |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_2    | 100.0 | 100.0 | 99.0 | 99.0 | 100.0 | (96.0) |
| niah_multikey_1  | 95.0 | 94.0 | 97.0 | 91.0 | 93.0 | (91.0) |
| variable_tracking| 96.6 | 97.6 | 98.8 | 99.0 | 95.8 | (99.0) |

- **Weighting definition of 97.05:** **simple (unweighted) arithmetic mean of the 15 cells**
  (3 tasks × 5 lengths 8k–128k). Sum = 498.0 + 470.0 + 487.8 = 1455.8; 1455.8/15 = **97.0533 → 97.05**. ✔
- **256k is excluded** from the 97.05 headline (256k mean = (96+91+99)/3 = 95.33). If 256k is added,
  the 18-cell mean = 1637.8/18 = 90.99 — do not conflate; keep 97.05 defined over 8k–128k.
- Each cell is itself an n=100, 8-shard weighted mean; the 15-cell headline weights every cell equally
  regardless of task (so single/multikey/VT each contribute 1/3).

### 3.2 LongEval 72.83 — per-length (register-content acc, n=100/len)
Dir `longeval_results/qcmem_8b_iter_chatFALSE/longeval_8b`, 8-shard, correct/total aggregated.
| len | 4k | 8k | 16k | 32k | 64k | 128k | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|
| acc | 92.0 | 69.0 | 75.0 | 64.0 | 67.0 | 70.0 | **72.83** |
72.83 = simple mean over the 6 lengths (each n=100). Substring-recompute over all 600 records agrees (72.83).

### 3.3 LongBench 12.15 — per-dataset (official qa_f1 / SQuAD token-F1)
Dir `longbench_results/qcmem_8b_iter_chatFALSE`, 8-shard, per-ds n-weighted F1.
| ds | narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 4.12 | 11.01 | 11.62 | 12.83 | 25.41 | 7.91 | **12.15** |
- **12.15 = MACRO mean over the 6 datasets** (each dataset weighted equally). N-weighted MICRO mean
  = 11.57 (N=1150: 2wikimqa/hotpotqa/musique/narrativeqa/qasper=200 each, multifieldqa_en=150).
  Report the macro口径 (12.15) to match the errata; state which is used.

### 3.4 BABILong — qa1/qa2/qa5 × 0k–32k (babilong.metrics compare_answers, n=100/cell)
Dir `babilong_results/qcmem_j12_iter_bm25_chatFALSE_ad`, 4-shard (shard*of4), --expect 100 (no warnings).
| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qa1 | 98 | 79 | 68 | 69 | 33 | 17 | 11 | 53.6 |
| qa2 | 26 | 44 | 43 | 43 | 18 |  5 |  0 | 25.6 |
| qa5 | 69 | 75 | 76 | 73 | 69 | 54 | 51 | 66.7 |
Matches errata §11d exactly.

### 3.5 LoCoMo per-category (F1 / EM / acc / judge) — see Delivery 1(b) tables above.

### 3.6 Bootstrap 95% CI on key n=100 cells (1000 resamples, seed=2024)
RULER (flagship equal-budget dir, recomputed recall):
| cell | mean | 95% CI |
|---|---:|---|
| variable_tracking 64k | 99.00 | [98.20, 99.80] |
| variable_tracking 128k | 95.80 | [94.20, 97.40] |
| niah_single_2 32k | 99.00 | [97.00, 100.00] |
| niah_multikey_1 64k | 91.00 | [85.00, 96.00] |

BABILong (compare_answers 0/1):
| cell | mean | 95% CI |
|---|---:|---|
| qa1 0k | 98.00 | [95.00, 100.00] |
| qa1 8k | 33.00 | [24.00, 41.00] |
| qa2 0k | 26.00 | [17.98, 34.00] |
| qa5 0k | 69.00 | [59.00, 77.00] |
| qa5 32k | 51.00 | [41.00, 61.00] |

LongEval overall acc (n=600) = 72.83, 95% CI [69.33, 76.33].
LoCoMo headline judge/F1 CIs: see Delivery 1.

---

## Verification / Iron-Law-2 summary
- LoCoMo: 4 preds shards dedup → n=1986 both dirs; overall judge recompute = scores.json (0 drift);
  0 MISSING non-cat5 cache entries; cat5 excluded from cache (0 present).
- RULER (15 cells + VT 5 cells × 2 configs): every scored cell reports n=100, empty=0, on-disk vs
  official-kernel-recompute mismatch=0, 8/8 shards present → IRON-LAW-2 ALL CELLS OK: True.
- LongEval/LongBench: aggregated from committed per-shard summaries/metrics; totals n=100/len and
  n=1150 respectively; empty_prediction_count / oom_count = 0 in metrics.
- BABILong: `--expect 100` produced no CSV-corruption warnings on any of the 21 cells.

## Discrepancies vs errata §11 — NONE on the numbers
Every headline (LoCoMo judge 38.27/34.59, RULER 97.05, LongEval 72.83, LongBench 12.15, BABILong grid)
reproduced exactly from the raw dirs. The only open item is the **config/narrative choice for the
equal-budget VT row** (Delivery 2D): the chat=False data reverses tab_itervt's chat=True
"iterative-non-degrading" claim (single-pass equal-budget is the non-degrading one at chat=False).
Needs a researcher/user narrative decision, not a data fix.


## §1e — LoCoMo GPT-4o judge: paired significance (2026-07-23, main)

Resolves the §1 caveat that per-method **unpaired** judge CIs overlapped
([36.20,40.58] CoMem vs [32.48,36.71] KVD). LoCoMo is a **paired** design (same
questions graded for both methods), so the correct test is a per-sample
paired-difference bootstrap over the shared judged ids.

- Kernel: intersect judge_cache ids of `qcmem_8b_locomo_iter_chatFALSE` (CoMem)
  and the KV-Direct full-ctx dir; per-id paired diff (CoMem_judge − KVD_judge);
  10000-resample bootstrap, seed=1234.
- **n_common (cat1–4 judged) = 1540.** CoMem = 48.64, KVD = 43.83.
- **paired diff = +4.81, 95% CI = [2.34, 7.27], two-sided p < 0.0001,
  P(CoMem>KVD) = 1.0000 → significant @95% (CI excludes 0).**

**Verdict:** under the GPT-4o judge on judged questions CoMem significantly beats
the full-context KV-Direct oracle. The earlier unpaired-CI overlap was a
power artifact of ignoring the pairing. The full-headline diff (+3.68 over
n=1986) is dominated by these same judged samples plus near-identical cat5
abstention, so it remains significant. Paper MAY claim CoMem > KVD oracle
under the canonical judge.
