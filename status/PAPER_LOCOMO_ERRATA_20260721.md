# Paper LoCoMo table — verified numbers + errata analysis (2026-07-21)

**Status: EVIDENCE GATHERED, NO PAPER EDIT MADE.** This is a narrative-flipping change,
not a mechanical errata — needs a deliberate researcher/user decision (see §4).

All numbers below are **official `run_scoring` output** (token-level SQuAD-F1, n=1986,
selector=iter_bm25, chat_template=ON + no-think) read directly from
`locomo_results/*/scores.json` on .73 (28.85.35.73, diskB). 铁律2 satisfied for F1/acc/EM.

## 1. CoMem 8B iter_bm25 (flagship) — VERIFIED
`locomo_results/qcmem_8b_iter_chatnothink/scores.json`, n=1986
- **overall: F1 19.51 / acc 28.65 / EM 5.99**  ← confirms errata target 19.51/28.65
- by-category (F1 / EM / acc):
  - cat1 (n=282): 19.59 / 3.90 / 21.63
  - cat2 (n=321): 20.14 / 3.12 / 18.38
  - cat3 (n=96):  11.59 / 3.13 / 18.75
  - cat4 (n=841): 29.77 / 10.58 / 50.54
  - cat5 (n=446):  1.35 /  1.35 /  1.35
- **⚠️ NO judge file in this dir** → the paper's `Judge=39.5` column CANNOT be re-verified
  against this result. Would need a fresh `--use_llm_judge` (gpt-4o) run on this exact dir.

## 2. Baselines — VERIFIED (LoCoMo, iter/chatnothink, official run_scoring)
| model | F1 | acc | EM | dir |
|---|---|---|---|---|
| KV-Direct 8B | **40.06** | **43.05** | 19.59 | kvdirect_8b_chatnothink |
| InfLLM 8B | 25.76 | 26.38 | 11.33 | infllm_8b |
| **CoMem 8B** | 19.51 | 28.65 | 5.99 | qcmem_8b_iter_chatnothink |
| StreamingLLM 8B | 12.73 | 17.57 | 5.24 | streamingllm_8b_chatnothink |
| MemoryLLM | 9.93 | 9.72 | 0.96 | memoryllm_chatnothink |
| HCache 8B | 7.82 | 8.06 | 0.05 | hcache_8b_chatnothink |
| LLoCO | (not present — dir missing) | | | |

## 3. Discrepancy vs current paper `tab_locomo.tex`
Paper currently shows (stale, pre-iter_bm25 protocol):
- CoMem 9.05 / 24.1 / Judge 39.5
- KV-Direct 8.80 / 20.0 / 31.5
- HCache 4.73 / 6.4 / 12.3
The verified iter_bm25 numbers are on a **completely different scale** — the paper's
LoCoMo table was computed under a DIFFERENT protocol (old selector/scoring). Swapping in
one row only (CoMem→19.51) would leave the table mixing two protocols → internally
inconsistent. Must swap the WHOLE table coherently or none of it.

## 4. Why this is NOT a heartbeat edit (narrative flip)
Under verified iter_bm25: **KV-Direct (F1 40.06) massively BEATS CoMem (19.51)** on LoCoMo.
The paper's current claim (05_experiments.tex:171-178) is "CoMem ≈ KV-Direct" (9.05 vs 8.80).
Correcting to real numbers **inverts the headline comparison** on this benchmark. That is a
scientific/narrative decision (how to frame an oracle-KV baseline beating the compression
method on a real-QA task), NOT a mechanical errata. Requires deliberate researcher/user pass:
- decide framing (KV-Direct = uncompressed oracle upper bound; expected to win — but must be
  stated honestly, not buried).
- re-run gpt-4o judge on the iter_bm25 dir so the Judge column is grounded, OR drop it.
- audit whether BABILong/LongBench/overview tables use old-protocol or iter_bm25 numbers, so
  the whole results section is one coherent protocol.

## 5. Recommendation for task #10
Do the paper integration as ONE coordinated, protocol-coherent pass (dispatch a coder/
researcher with THIS verified number set), covering: tab_locomo (all rows + per-cat + judge
decision), tab_overview LoCoMo row, and the 05_experiments prose. Do NOT edit piecemeal.
Baseline rows to ADD (verified above): InfLLM, StreamingLLM, MemoryLLM (+ HCache/KV-Direct
already present but need protocol-consistent updates).

## 6. Read-only protocol audit (2026-07-21 06:26 heartbeat, GPU-free de-risk)
Purpose: scope how far the "stale LoCoMo protocol" contamination reaches, so §5's pass
knows the full edit surface. Paper was restructured — sections are now
02_related/03_motivation/04_methodology/05_experiments/06_conclusion/07_limitations
(old 02_method..07_hunyuan_moe deleted). LoCoMo appears in exactly:
`sections/tab_locomo.tex`, `sections/tab_overview.tex`, `sections/05_experiments.tex`,
`sections/07_limitations.tex`.

**Finding A — TWO tables carry stale LoCoMo, not one.** Both must swap coherently:
- `tab_locomo.tex`  rows: CoMem 9.05/24.1/Judge39.5, KV-Direct 8.80/20.0/31.5,
  HCache 4.73/6.4/12.3, + a full per-category(a)/(b) block (cat1-5 F1/acc/Judge).
- `tab_overview.tex` LoCoMo row (line 17): CoMem **9.05/24.1**, KV-Direct **8.72/20.3**,
  HCache **4.73/6.4**. (note KV-Direct here=8.72, vs tab_locomo 8.80 — already inconsistent.)
  Verified iter_bm25 replacements for this row: CoMem 19.51/28.65, KV-Direct 40.06/43.05,
  HCache 7.82/8.06.

**Finding B — OPEN, higher-stakes than LoCoMo alone.** `tab_overview.tex` also holds
CoMem rows for RULER (niah_single/multikey), LongEval, BABILong (qa1/qa5), LongBench
(AVG 9.58) under caption "per-task optimal top-k, official scorers". Per the 2026-07-17
user directive (memory: qcmem-eval-selector-iterbm25 — "所有 QCMem eval 统一 iter_bm25,
旧 bm25 结果作废需重跑"), iter_bm25 is a *retrieval selector* → it changes which chunks
CoMem reads on EVERY benchmark, not just LoCoMo. It is UNVERIFIED whether these
RULER/LongEval/BABILong/LongBench numbers were re-run with iter_bm25 or are pre-directive
bm25/top-k numbers. **The coherent pass MUST first confirm each non-LoCoMo row's selector
against its result dir before trusting the overview table** — if any are stale bm25, the
fix is a re-run (GPU), not a text edit. This is the true blocker that keeps #10 off the
heartbeat auto-lane.

**Not chased this tick (correctly bounded):** actually cross-referencing each benchmark's
result dir → selector is the coherent pass's job (needs dir-by-dir audit + possibly GPU
re-runs); flagged here, not resolved, to avoid half-baked piecemeal edits.

## 7. Finding B RESOLVED — dir-by-dir selector audit (2026-07-21 17:56 heartbeat, GPU-free)
Cross-referenced each `tab_overview.tex` CoMem row against the result dir that fed its
number (source: `status/QCMEM_PAPER_DRAFT.md §2.0`, which cites the exact dirs). Read the
`selector` field straight out of each dir's own `eval_config.json` — not draft prose:
- **LongEval** `longeval_results/longeval_qcmem_tk8_32k/eval_config.json` → **`"selector": "bm25"`**
- **RULER** `ruler_results/qcmem_128k/{niah_single,niah_multikey}_*.json` → **`"selector": "bm25"`**
- **LongBench** `longbench_results/qcmem_j12/eval_config_0.json` → **`"selector": "bm25"`**
- **BABILong** `babilong_results/qcmem_trained/*` (self-distilled, draft §2.5 "最优 topk") →
  bm25/top-k family, no iter_bm25 config present.

**Verdict:** ALL FOUR non-LoCoMo overview CoMem numbers are OLD `selector=bm25` (per the
draft's own READ-path definition, §2.0 line 20: "bm25 检索 topk chunk"). Under the
2026-07-17 user directive (iter_bm25 mandatory, 旧 bm25 作废需重跑) these are **STALE →
require GPU re-runs, NOT text edits.**

**iter_bm25 flagship availability (checked):** LongEval=**NONE**, BABILong(non-LM2)=**NONE**,
LongBench=**NONE**, RULER = only exploratory dirs (`qcmem_iter` = iterL maxsim/meanpool
variants on select cells; `qcmem_iterbm25_vt` = vt task only; `qcmem_1.7b_iter` = 1.7B not
8B) — NONE covers the 8B flagship overview cells. So iter_bm25 8B numbers exist ONLY for
LoCoMo (§1: 19.51/28.65/5.99).

**→ TRUE BLOCKER for #10, now quantified:** the paper integration needs **4 benchmark ×
CoMem-8B iter_bm25 re-runs** (RULER niah_single/multikey, LongEval, BABILong qa1/qa5,
LongBench 4-task) before tab_overview can be made protocol-coherent. These are GPU jobs.
Currently 32/32 GPU busy with Paper B training → cannot schedule now; queue behind the
first Paper B arm to finish/early-stop. LoCoMo table (§1-5) is the ONLY part editable
text-only today (its iter_bm25 numbers are in hand), but even that flips the "CoMem ≈
KV-Direct" headline → still a deliberate researcher/user narrative decision, not mechanical.

## 8. ⚠️ §7 CORRECTED — iter_bm25 8B data ALREADY EXISTS (2026-07-22 ~10:40 heartbeat)
§7's conclusion ("iter_bm25 8B exists ONLY for LoCoMo → need 4 GPU re-runs") is **WRONG**.
Re-audited the actual `*_results/` dirs on .73 (28.85.35.73, diskB) — §7 checked the wrong
dir names (the dirs the paper draft *cites*), missing the newer iter_bm25 re-run dirs that
supersede them. Verified live:

**(a) The paper's currently-cited old bm25 dirs are all GONE (superseded/deleted):**
- `longeval_results/longeval_qcmem_tk8_32k` → GONE
- `ruler_results/qcmem_128k` → GONE
- `longbench_results/qcmem_j12` → GONE
→ the paper table cites dead dirs; they were replaced by iter_bm25 runs.

**(b) CoMem-8B iter_bm25 results EXIST and are scored for ALL 4 benchmarks:**
- **LongBench** `longbench_results/qcmem_8b_iter_chatnothink/` — `selector=iter_bm25`, has
  `scores.json` + per-ds `*_metrics.json` (also `qcmem_8b_zs_iter_chatnothink` zero-shot).
- **LongEval** `longeval_results/qcmem_8b_iter_chatnothink/longeval_8b/` — summary present:
  4k=0.95 8k=0.73 16k=0.76 32k=0.79 64k=0.72 128k=0.7x (also `_zs_` zero-shot w/ _summary_t27).
- **RULER** `ruler_results/qcmem_8b_iter_bm25_chatnothink_ad/` + `qcmem_8b_zs_iter_bm25_chatnothink/`
  — niah_single/niah_multikey/variable_tracking × 8k/16k/32k (also `_iterbm25_ext` for 64k/128k,
  `_iterbm25_vt`, `_iterbm25_baseline`).
- **BABILong** `babilong_results/qcmem_j12_iter_bm25_chatnothink_ad/` (adapter) +
  `qcmem_j9_iter_bm25_chatnothink_zs/` (zero-shot).
- Plus a FULL iter_bm25 scale sweep (0.6b/1.7b/4b/8b/14b/32b, zs+adapter) across LongEval/
  RULER/LongBench/BABILong from T25/T26/T27.

**(c) KV-Direct selector mismatch (NEW, real issue):** `locomo_results/kvdirect_8b_chatnothink/`
uses `selector=bm25` (top-12), while CoMem flagship uses `selector=iter_bm25` (top-12). So the
§2 head-to-head (KV-Direct 40.06 vs CoMem 19.51) compares MISMATCHED selectors. For a clean
same-protocol table, baselines must either all use iter_bm25 too, or the table must state
"each method uses its own best selector" explicitly.

**→ REVISED task#10 scope (much lighter than §7 claimed):** NOT 4 GPU re-runs. It is a
CPU/scoring + table-repointing job: (1) re-point tab_overview/tab_locomo from the dead bm25
dirs to the existing iter_bm25 dirs; (2) read official-scorer numbers (BABILong compare_answers,
RULER string_match, LongBench/LoCoMo run_scoring) off those dirs; (3) reconcile baseline
selectors (KV-Direct etc.); (4) backfill only genuinely-missing cells (verify RULER 64k/128k
coverage in `_iterbm25_ext`; confirm BABILong qa1/qa5 lengths). Does NOT need a dedicated
training node → the offered B200 should go to Paper B acceleration, not task#10.

## 9. Task#44 COMPLETE — chat_template T/F × {token-F1, GPT-4o judge} ablation (2026-07-22 21:4x)
User spec (locked): selector=iter_bm25 fixed; chat_template∈{True,False}; metrics = token-F1 +
GPT-4o LLM-as-judge (+ acc/EM as scientific extras). n=1986 each, official `run_scoring`,
gpt-4o judge (maas endpoint, cached judge_cache.jsonl). Verified live off scores.json (.82).

| protocol | model | token-F1 | acc | EM | **GPT-4o judge** |
|---|---|---|---|---|---|
| **chat=True**  | CoMem (iter_bm25) | 19.51 | 28.65 | 5.99 | **37.76** |
| **chat=True**  | KV-Direct (oracle) | **40.06** | 43.05 | 19.59 | **38.22** |
| **chat=False** | CoMem (iter_bm25) | 9.15 | 23.36 | 0.55 | **38.27** |
| **chat=False** | KV-Direct (oracle) | 9.02 | 22.36 | 0.60 | **34.59** |

dirs: qcmem_8b_iter_chatnothink / kvdirect_8b_chatnothink (chat=True);
qcmem_8b_iter_chatFALSE / kvdirect_8b_chatFALSE (chat=False), all on .82/.73 diskB.

**FINDINGS (this is the errata resolution for the "KV-Direct beats CoMem" flip):**
1. **Under the CANONICAL metric (GPT-4o judge), CoMem ≈ KV-Direct under BOTH protocols** —
   chat=True 37.76 vs 38.22 (tie, KVD +0.46); chat=False **38.27 vs 34.59 (CoMem +3.68, AHEAD)**.
2. **token-F1 is protocol-dependent & biased toward the extractive oracle:** chat=True F1
   shows KVD 40.06 >> CoMem 19.51 (+20.55 gap) — but that gap **VANISHES** under native
   chat=False (9.15 ≈ 9.02). The gap is a **chat-template artifact**, not a capability gap.
3. **CoMem's judge is protocol-STABLE (37.76→38.27); KV-Direct's judge DROPS (38.22→34.59)**
   when chat template removed → CoMem (a BASE-trained compression method) is MORE robust to
   its native no-chat regime; the oracle relies more on chat formatting for clean answers.
4. token-F1 collapses True→False for BOTH methods (verbose/less-formatted base output) while
   judge stays ~stable → token-F1 penalizes FORMATTING, not correctness. Classic bag-of-tokens
   bias against generative/compression methods.
5. **RETRACTION:** the earlier "KV-Direct massively beats CoMem (F1 40.06 vs 19.51)" headline
   (§4) is retracted as a metric+protocol artifact. **Corrected claim: on LoCoMo, under the
   canonical GPT-4o judge, CoMem matches or beats the full-context KV oracle at both chat
   states; under the native chat=False token-F1 they tie.** Consistent with old-protocol
   bm25+chatF reference (CoMem judge 39.48 > KVD 31.54).

**→ Paper implication for task#10:** the LoCoMo table should report the **GPT-4o judge** as the
headline metric (it is LoCoMo's canonical metric AND protocol-robust). If token-F1 is shown, it
MUST be paired with the chat state + a note that it is formatting-sensitive; do NOT report the
chat=True F1 alone (it manufactures a spurious KVD win). Recommended headline row (judge):
CoMem 37.76–38.27 ≈ KV-Direct 34.59–38.22 across protocols.

**HCache supporting evidence (artifact is method-wide, 2026-07-22 22:00):** HCache token-F1
also collapses chat True→False: **7.82 → 4.67** (dir hcache_8b_chatnothink → hcache_8b_chatFALSE,
n=1986, official run_scoring). Both < CoMem/KVD (HCache is the weakest compression baseline at
either chat state). judge for hcache_8b_chatFALSE running (CPU). Confirms: the chat-template
sensitivity of token-F1 is universal across compression/oracle methods, not specific to the
CoMem-vs-KVD pair → strengthens the "report GPT-4o judge as headline" recommendation.

### 9b. Baseline three chat=False COMPLETE (InfLLM / StreamingLLM / MemoryLLM) — 2026-07-23
Completes the chat T/F artifact matrix for the three non-oracle baselines. token-F1/acc/EM only
(no LLM judge run this pass). All n=1986, official `run_scoring`, 8-GPU sharded on .82 diskB.
chat=False = flip chat_template OFF, ALL other hparams identical to the chat=True dirs (Qwen3-8B
backbone; InfLLM block=128/n_init=128/n_local=4096/topk=16/chunk=8192; StreamingLLM
sink=4/window=6653; MemoryLLM chunk=1024/attn=sdpa; max_new_tokens=48; locomo10.json).

| baseline | dirs (chatTrue / chatFALSE) | chat=True F1/acc/EM | chat=False F1/acc/EM | dF1 |
|---|---|---|---|---|
| InfLLM        | infllm_8b / infllm_8b_chatFALSE                          | 25.76 / 26.38 / 11.33 | 7.39 / 13.34 / 1.71 | -18.37 |
| StreamingLLM  | streamingllm_8b_chatnothink / streamingllm_8b_chatFALSE  | 12.73 / 17.57 / 5.24  | 7.67 / 13.75 / 1.56 |  -5.06 |
| MemoryLLM     | memoryllm_chatnothink / memoryllm_chatFALSE              |  9.93 /  9.72 / 0.96  | 5.91 /  9.52 / 0.10 |  -4.02 |

**Finding:** token-F1 collapses chat True->False for ALL three baselines too (InfLLM -18.4,
StreamingLLM -5.1, MemoryLLM -4.0) — the same chat-template artifact seen for CoMem/KVD (§9) and
HCache (§9-HCache). The collapse is universal across token-space eviction (InfLLM/StreamingLLM),
fixed-pool memory (MemoryLLM), compression (HCache), retrieval (CoMem) and oracle-KV (KVD) ->
strongly supports the paper-wide chat=False policy (§10) and reporting GPT-4o judge as headline.

**Ops note:** MemoryLLM chat=False initially ran at ~309s/sample @2% GPU util under naive 8-way
launch — chat=False generates the full 48 tokens (no early `<|eot_id|>` stop) => ~10x more
per-token GPU->CPU syncs; 8 unpinned procs thrashed the 384-core CPU (loadavg 769). Fix = pin
OMP/MKL/RAYON threads per proc (`scripts/_run_memoryllm_locomo_chatFALSE_8gpu.sh`) -> 12-14s/sample,
GPU util ~94%, loadavg ~118, ~57 min/shard. InfLLM/StreamingLLM chat=False were unaffected (they
honor eos + evict long ctx, few tokens/step) — ran via `scripts/_run_baselines_locomo_chatFALSE_8gpu.sh`.

## 10. ★ PAPER-WIDE POLICY: all results chat_template=False (user directive 2026-07-22)
User locked a paper-level protocol (memory: [[paper-eval-chat-false-mandatory]]): **every result
in the paper (Paper A QCMem + Paper B OLMo-2) reports `chat_template=False`**, and the paper must
**explicitly state our models have no SFT/RL (continue-trained BASE LMs) → chat template is unfair/OOD**.
Task#44 (§9) is the empirical justification: chat=True token-F1 manufactures a spurious KVD>>CoMem
gap that vanishes at chat=False; base models never saw `<|im_start|>` tokens.

**Re-run scope under this policy (chat=False, selector=iter_bm25 held fixed):**
- **LoCoMo — largely DONE (chat=False):** CoMem iter F1 9.15 / judge 38.27 (`qcmem_8b_iter_chatFALSE`);
  KV-Direct 9.02 / judge 34.59 (`kvdirect_8b_chatFALSE`); HCache 4.67 (`hcache_8b_chatFALSE`);
  CoMem-bm25 8.76 (`qcmem_8b_adapter`). Baseline three (InfLLM/StreamingLLM/MemoryLLM) chat=False
  running via coder abaed0fc on .82.
- **RULER / LongEval / LongBench / BABILong — ALL chat=True only (0 chat=False dirs; 17/16/17/19
  `chatnothink` dirs respectively, audited 2026-07-22 on .73).** The §8 iter_bm25 8B data exists but
  is `chatnothink` (=chat=True) → **needs chat=False re-runs** for CoMem 8B flagship + baselines
  (+ scale sweep 0.6B–32B if it stays in the paper). These are GPU jobs.
- **GPU reality:** 4 nodes locked in Paper B training (to step200000, no plateau→no early-stop);
  .82 running LoCoMo baseline chat=False. → the 4-benchmark chat=False campaign **queues behind Paper B**;
  prioritize CoMem-8B flagship + core baselines over the full scale sweep. Earliest free node = whichever
  Paper B arm finishes/early-stops, or .82 after its LoCoMo baseline job.
- **Paper text (task#10):** repoint tab_overview/tab_locomo/per-benchmark tables to chat=False dirs;
  headline LoCoMo metric = GPT-4o judge; add methodology sentence "baselines + CoMem share one base
  backbone + one chat=False protocol; no SFT/RL → chat template unfair."

## 11. Phase 1 COMPLETE — CoMem 8B flagship chat=False on all 4 benchmarks (2026-07-23 00:2x)
Executed the §10 chat=False re-run policy for the **CoMem 8B flagship** (the 4 non-LoCoMo
benchmarks; LoCoMo already done in §9). Method: replicate each existing `*_chatnothink` (chat=True
+ no-think) 8B CoMem run's config EXACTLY, drop only `--use_chat_template`, write to a parallel
`*_chatFALSE` dir → guarantees a clean chat-True-vs-False A/B. Config held fixed:
`models/Qwen3-8b-local` + LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final` (resume_j=12),
selector=iter_bm25, topk=12, sink=bos, chunk_size=512, n=100/cell (or full ds), official scorers.
Run on .73 (28.85.35.73, diskB), torch-base python. **RULER extended to 64k/128k** (canonical
chatnothink j12 only had 8k/16k/32k). Dirs: `ruler_results/qcmem_8b_iter_chatFALSE_ad`,
`longeval_results/qcmem_8b_iter_chatFALSE`, `longbench_results/qcmem_8b_iter_chatFALSE`,
`babilong_results/qcmem_j12_iter_bm25_chatFALSE_ad`.

### 11a. RULER (official string_match, n=100/cell) — chat=False DRAMATICALLY HELPS
| task | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| niah_multikey_1  T | 85.0 | 82.0 | 80.0 | — | — |
| niah_multikey_1  F | 95.0 | 94.0 | 97.0 | 91.0 | 93.0 |
| niah_single_2    T | 100.0 | 100.0 | 99.0 | — | — |
| niah_single_2    F | 100.0 | 100.0 | 99.0 | 99.0 | 100.0 |
| variable_tracking T | **1.8** | **1.2** | **0.6** | — | — |
| variable_tracking F | **96.6** | **97.6** | **98.8** | 99.0 | 95.8 |

**mean: chat=True 61.07 (9 cells 8-32k) → chat=False 97.05 (15 cells); on the SAME 9 cells
chat=False = 97.56.** The gain is driven ENTIRELY by **variable_tracking**: chat=True ≈1%
(essentially broken) → chat=False ≈97%. The chat template wraps VT completions in preamble/
formatting that breaks exact `string_match`; the base completion emits the tracked variables
verbatim. niah_multikey also up ~13pt (same mechanism). Dirs: chat=True
`ruler_results/qcmem_8b_iter_bm25_chatnothink_ad` vs chat=False `ruler_results/qcmem_8b_iter_chatFALSE_ad`.

### 11b. LongBench (official qa_f1 / SQuAD token-F1) — chat=False HURTS ~3x
| ds | narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **mean** |
|---|---|---|---|---|---|---|---|
| chat=True  | 21.09 | 37.11 | 49.54 | 37.67 | 45.50 | 23.70 | **35.77** |
| chat=False |  4.12 | 11.01 | 11.62 | 12.83 | 25.41 |  7.91 | **12.15** |

Same token-F1 collapse as LoCoMo §9: without the chat template the base LM produces verbose/
non-extractive continuations → token-F1 (which rewards concise extractive spans) drops ~3x.
**Confirms the artifact is method-wide:** KV-Direct hotpotqa collapses identically 57.83→12.68
(`longbench_results/kvdirect_8b_chatnothink` → `kvdirect_8b_chatFALSE`), and at chat=False the
oracle's F1 advantage over CoMem VANISHES (KVD hotpotqa 12.68 ≈ CoMem 11.62). Dirs: chat=True
`longbench_results/qcmem_8b_iter_chatnothink` vs chat=False `longbench_results/qcmem_8b_iter_chatFALSE`.

### 11c. LongEval (register-content acc, n=100/len) — chat=False small uniform drop
| len | 4k | 8k | 16k | 32k | 64k | 128k | **mean** |
|---|---|---|---|---|---|---|---|
| chat=True  | 95.0 | 73.0 | 76.0 | 79.0 | 72.0 | 76.0 | **78.50** |
| chat=False | 92.0 | 69.0 | 75.0 | 64.0 | 67.0 | 70.0 | **72.83** |

~5.7pt mean drop, uniform across lengths — mild format effect. Dirs: chat=True
`longeval_results/qcmem_8b_iter_chatnothink` vs chat=False `longeval_results/qcmem_8b_iter_chatFALSE`.

### 11d. BABILong (babilong.metrics compare_answers, n=100/cell) — mixed
| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean |
|---|---|---|---|---|---|---|---|---|
| qa1 T | 100 | 82 | 68 | 63 | 50 | 23 | 27 | 59.0 |
| qa1 F |  98 | 79 | 68 | 69 | 33 | 17 | 11 | 53.6 |
| qa2 T |  57 | 58 | 53 | 51 | 31 | 20 |  6 | 39.4 |
| qa2 F |  26 | 44 | 43 | 43 | 18 |  5 |  0 | 25.6 |
| qa5 T |  83 | 83 | 78 | 76 | 70 | 59 | 61 | 72.9 |
| qa5 F |  69 | 75 | 76 | 73 | 69 | 54 | 51 | 66.7 |

qa1/qa2 drop (qa2 hardest: 0k 57→26); qa5 roughly flat (72.9→66.7). compare_answers is a lenient
substring match → smaller effect than LongBench F1. Dirs: chat=True
`babilong_results/qcmem_j12_iter_bm25_chatnothink_ad` vs chat=False `..._chatFALSE_ad`.

### 11e. HEADLINE FINDING — the chat-template effect is METRIC-DEPENDENT, not uniform
Across all 4 benchmarks the DIRECTION of the chat=True→False change is governed by how much the
metric tolerates verbose output, NOT by any capability change:
- **Exact-match / completion tasks (RULER, esp. variable_tracking): chat=False HELPS strongly**
  (61→97; VT 1%→97%). Base completion matches the expected literal format; chat preamble breaks it.
- **Extractive token-F1 QA (LongBench): chat=False HURTS ~3x** (35.77→12.15). Base LM rambles.
- **LongEval / BABILong: small drop.**
This is the direct 4-benchmark generalization of §9's LoCoMo thesis (chat=True token-F1 penalizes
FORMATTING, not correctness; base LMs never saw `<|im_start|>`). **Reviewer-facing implication:**
under the mandated chat=False protocol some CoMem numbers RISE (RULER) and some FALL (LongBench) —
the paper must state this is a single formatting phenomenon, and should prefer format-robust metrics
(exact-match, GPT-4o judge) over token-F1 where possible; where token-F1 is reported, pair it with
the chat state + the formatting caveat.

**Status:** Phase 1 (CoMem 8B flagship, 4 benchmarks) DONE + scored. Phase 2 (baselines KV-Direct/
HCache/… chat=False) is being run comprehensively by the **.104 driver a93d4e558** (LongBench kvd/
hcache chatFALSE already in progress on shared diskB) → this session STANDS DOWN on baselines to
avoid shared-FS concurrent-write races (a duplicate .73 baseline launch was superseded). Phase 3
(CoMem 8B zero-shot chat=False) NOT run (optional; .73 released for main to resume training).
New launchers + scorer committed on diskB main @ eff036f (5× `_run_8b_*_chatFALSE.sh` +
`scripts/_score_chatFALSE.py`), no push.

---

## 12. LoCoMo GPT-4o judge backfill — 3 chat=False baselines (2026-07-24, task #64) ✅ DONE

The 3 remaining chat=False baselines (InfLLM/StreamingLLM/MemoryLLM) previously had only
F1/EM/acc; their GPT-4o judge scores were backfilled on .73 (28.85.35.73) via the funded
`maas-openapi` JWT key (cat1–4 → API, cat5 adversarial = local abstention). n=1986 each,
`judge_model=gpt-4o`. Scores in `locomo_results/{infllm_8b,streamingllm_8b,memoryllm}_chatFALSE/scores.json`.

| method (chat=False) | overall_judge | cat1 | cat2 | cat3 | cat4 | cat5 (adv) | overall_f1 | overall_acc |
|---------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| StreamingLLM        | **25.63** | 22.70 | 11.21 | 23.96 | 42.21 | 6.95 | 7.67 | 13.75 |
| InfLLM              | **22.21** | 18.44 | 14.33 | 25.00 | 33.89 | 7.62 | 7.39 | 13.34 |
| MemoryLLM           | **16.11** | 15.60 |  8.72 | 15.63 | 27.47 | 0.45 | 5.91 |  9.52 |

Notes for the paper LoCoMo table:
- **All three fixed-budget streaming/compression baselines sit far below CoMem/KV-Direct** on the
  judge metric — StreamingLLM 25.6, InfLLM 22.2, MemoryLLM 16.1. **CORRECTION (2026-07-24):** the full
  chat=False judge ranking is **CoMem 38.27 > KVD 34.59 > StreamingLLM 25.63 > InfLLM 22.21 > MemoryLLM
  16.11 > HCache 8.11** (see `status/LOCOMO_JUDGE_AGGREGATE.md`, single source of truth). The earlier
  "KV-Direct judge 40.06 >> CoMem-oneshot 19.51" phrasing was WRONG — 40.06/19.51 are token-**F1**
  (chat=True), NOT judge; on the judge metric CoMem and KVD are near-tied at chat=True (37.76 vs 38.22)
  and CoMem *leads* at chat=False (38.27 vs 34.59).
- cat4 (single-hop) is every method's best category; cat2 (multi-hop temporal) the hardest.
- cat5 (adversarial / "no answer") is scored locally as abstention, so its judge==f1==em==acc by
  construction — MemoryLLM's 0.45 there means it almost never abstained correctly.
- **Baseline judge coverage is now COMPLETE** for the chat=False LoCoMo table. Remaining LoCoMo cell:
  CoMem adapter-free (#65) judge pass (later `--score_only --use_llm_judge`, same key).
