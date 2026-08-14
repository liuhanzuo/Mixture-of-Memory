# lane4_cheapest_killer — scouting report

Scope: sweep whole `proposal/` tree for the cheapest gate that can KILL something,
runnable tonight on `.21` (wzc1, 8xL20A 183GB, 8/8 idle) or `.82` (zwfy6, 8xH20 97.8GB, 8/8 idle).
Read-only. No launches. Every "on disk" claim below cites a path + observed size.

**TOP PICK: B04 cross-family Qwen replication on `.21`.** All 6 ckpts + base model + harness
+ analysis script are on **wzc1**, so `.21` needs zero cross-disk transfer. It is a genuine
kill test for B04's generality (which is B04's own named "next kill test") and B04 is the only
direction whose within-family evidence is already at max significance. ~4-6 GPU-hours.
Needs one 3-line copy-edit of an existing script (paths + harness name), no new code.

---

## Triage table (all 13 proposals)

`id | status | cheapest gate | can it kill? | assets on which disk?`

| id | status | cheapest gate | can it kill? | assets disk |
|---|---|---|---|---|
| A01 | active, p1 | gate-4 C4 aggregation pre-reg (CPU) | NO (removes a reviewer attack, decorates) | wzc1 |
| A01 | active, p1 | gate-2 non-MMLU MC interface case (GPU) | PARTIAL — kill needs BOTH gate-1 and gate-2 to fail | wzc1 (harness+18 per-ex jsonl) |
| A01 | active, p1 | gate-3 fp32-vs-bf16 causal tie test (GPU) | NO (mechanism, not existence) | wzc1 (`code/a01_gate3_fp32_vs_bf16.py` 28267 B) |
| A02 | active, p2 | phase-1 5-config x 5-benchmark, zero training | YES (decisive: paired CI<0 → retire "CoMem>RAG") | **zwfy6** Write-LoRA; Read-LoRA on wzc1 |
| A03 | active, p3 | floor gate on 3 remaining knowledge axes | YES in principle — **but needs new harness code** | zwfy6 ckpts (12G each) |
| A04 | active_design, p4 | 24x 1B trainings, or CPU protocol doc | NO cheaply (most expensive proposal) | needs new training |
| B01 | backlog | persist bottleneck latent + long-memory quality | YES but store not implemented | wzc1 |
| B02 | backlog | **BROKEN PREMISE (doc bug, confirmed)** | YES once a fresh fixed-sample run exists | zwfy6 |
| B03 | hold_gate_only | 2x3 1B pretraining (regime x reset count) | YES but = 6 pretraining runs | needs new training |
| B04 | **SURVIVING** | **Qwen cross-family replication** ★ | **YES — generality kill, B04's own next gate** | **wzc1 (all of it)** |
| B05 | backlog | j x readout x task phase diagram | weak (self-demoted: "fold into A/B appendix") | mixed |
| B06 | backlog_confirmed_seed | unified-harness rejudge | YES (kill cond: gain vanishes) — but **already effectively answered, see below** | wzc1 |
| B07 | backlog systems | concurrency/edit serving bench | NO (systems measurement, no scientific kill) | zwfy6 |
| B08 | backlog portfolio | notes faithfulness | NO (portfolio of 3, no single gate) | mixed |
| B09 | backlog | 5K-of-100K selection | NO (method development) | TBD |

---

## The four specifics asked for

### 1. B04 is DONE and SURVIVING — do not re-run Direction A

`proposal/backlog/B04-eval-fragility-incubator/DIRECTION_A_VERDICT.md` (5132 B):
Spearman(core6, median_margin) = **+1.0000**, exact p = **0.0028**;
Spearman(core6, frac<0.005) = **-1.0000**, exact p = **0.0028** — both at the n=6
exact-permutation lower bound 1/360. 6/6 rungs, N=17,195 per-item margins per rung.
`STATUS.json` = `SURVIVING`, `promotion_pending: novelty_check_only`.

**Subsumption check (asked explicitly): NO other proposal's gate is subsumed by B04's result.**
B04 measures the *per-item acc_norm margin distribution* on `core6` vs structural damage.
- A01's MC-interface case is a different construct (letter-vs-content *interface validity*
  against input-blind nulls) and B04's own verdict excludes MMLU/knowledge tasks: "MMLU/knowledge
  tasks excluded from this analysis; they have separate margin distributions and separate
  confounds (interface effect studied in Paper E)".
- A04's gate is matched-PPL capability across seeds — different quantity, needs new training.
- A03's gate is null-floor certification on knowledge axes — different metric family.
So B04 buys MAIN nothing to skip. It does however *supply the ladder* that makes the cheapest
new kill test possible (see §top pick).

### 2. A03 — alive, and its next floor gate is NOT cheap tonight

`status/A03_1B_FLOOR_VERDICT.md` records `1B_PILOT_VIABLE`; `STATUS.json.next_gate[2]` says:
> run the same floor gate on A03's three remaining knowledge axes (new injected facts,
> updated/conflicting knowledge, multi-evidence) before building the 6 arms — only the
> old-parametric-knowledge axis has been floor-certified

**Assessment: NOT runnable tonight.** The reason is code, not compute:

- The certified axis worked because two *existing* harnesses cover it verbatim:
  `scripts/eval_olmo2_mmlu_content.py` (35124 B) and `scripts/eval_olmo2_closedbook_qa.py`
  (20791 B), both driven by `scripts/_run_a03_1b_floor_82.sh` (5010 B), all present on
  **both** disks (verified on `.82`: same three files, plus
  `proposal/active/A03-.../code/analyze_1b_knowledge_floor.py` 24225 B).
- The three remaining axes have **no such harness**. What exists is:
  - `src/tasks/synthetic_update_task.py` (12952 B) — a *Chinese-language synthetic dialogue*
    generator (`FACT_TEMPLATES` values are `北京/上海/...`, queries `我现在住在哪个城市？`).
    It is a MoM-agent memory-update fixture, not an OLMo-2-1B base-LM knowledge probe, and
    running a BASE English LM on Chinese chat-style prompts would violate the project's
    own `chat_template=False` / base-protocol conventions.
  - `src/eval/update_eval.py` (7474 B) — keyword-coverage scorer for the above, again
    agent-oriented (`expected_keywords` / `forbidden_keywords`).
  - `scripts/eval_qcmem_longmemeval.py` (51654 B) — LongMemEval, but its model args are
    `--model_path` / `--bottleneck_ckpt` / `--baseline`; **it has no
    `--keep_front_layers` / `--n_fresh_layers` / `load_pruned_model` path** (verified by grep:
    `load_pruned_model` appears in 15 files, and this is not one of them). So it cannot even
    load the pruned+healed 1B arm.
- Additionally the 1B ckpts are **zwfy6-only**: `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`
  = **12G** and `outputs/olmo2_probe2_1B_keep7fresh2/step500.pt` = **12G** on `.82`;
  `ls` on wzc1 returns *No such file or directory* for both parent dirs. So A03 is `.82`-only,
  which is fine — but the missing piece is a harness, and writing + validating a new
  knowledge-injection/conflict eval with construct-appropriate nulls is a coder task, not a
  tonight launch.

Its two design findings are confirmed in the verdict and should be carried forward verbatim:
MMLU-**letter** is banned at 1B (pruned arm 0.2512 vs best-constant always-D 0.2689, i.e.
*below* its own floor at p=3.4e-3, and indistinguishable from its own modal-C constant p=0.28);
generative `contains` needs a length-matched null (naive 56.4% → 17.0% residual fraction once
matched, because healing inflates mean prediction length 13.4 → 80.8 chars).

### 3. B02 — doc bug CONFIRMED, in TWO files (report only, do not edit)

`proposal/backlog/B02-adaptive-depth-and-read-budget/PROPOSAL.md` still reads, under `## Stage 0`:

> 使用现有 sweep 计算 per-example oracle action 和 regret：
> - 若 oracle 相对最佳 fixed config 的收益不足，方向关闭；

and `proposal/backlog/B02-adaptive-depth-and-read-budget/STATUS.json` still reads:

> `"next_gate": "measure per-example oracle headroom from existing sweeps"`

Both retain the premise MAIN already falsified in `proposal/MINIMAL_VALIDATION_PLAN.md` §3:
the 8 T21 configs each used a *different* 50 samples, question-md5 intersection **0/50** for
all 7 pairs against j3 (target column likewise 0/50). Oracle and regret are therefore
uncomputable from existing data, and MAIN's own §3 conclusion was that B02's compute class
must change from `recompute_existing_data` to `needs_new_gpu_run`. **Neither file carries that
correction**, so the next agent reading only the proposal will repeat the dead end. Two-file fix.

### 4. B03's literature verdict, one sentence

**ALIVE-but-wounded, not dead:** `proposal/backlog/B03-cyclic-layer-reset-boundary/literature/KILLCHECK_forward_citations.md`
§0 returns **`SURVIVES`** after 434 forward citations across 6 seed papers (AC/DC 2106.12379 = 82,
Active Forgetting 2307.01163 = 48, 2109.00267 = 24, LLF 2202.00155 = 50, SEAL 2304.04858 = 11,
DSD 1607.04381 = 219) plus 10 LLM tech-report full-text greps (all CLEAN) plus OpenReview
ICLR25/26 + NeurIPS25 + ICML26 — **no published paper hits all five criteria** (decoder-only +
layer-granularity + cyclic + during-pretraining + size-invariant); the two nearest misses are
**arXiv:2410.16168** (decoder-only OLMo/GPT-2 active forgetting but it resets *token embeddings*,
not decoder blocks) and **arXiv:2602.04536 (IFA)** (layer-granularity cyclic reinit but on
CIFAR-10 / MIT-Indoors / Stanford Dogs federated image classification, not an LLM), with
**LLF (ICLR 2022)** + **2109.00267** owning the *operator*, **arXiv:2602.08040 FIRE (ICLR 2026,
OpenReview `CfZLxT3zIZ`)** doing weight-matrix-level Newton-Schulz reinit, and
**arXiv:2508.06412 (LoRR)** already reporting that `full_layers` reset is *detrimental* — but
in **post-training**, not pretraining.

**Is the verdict over-strict? Partly the opposite, and where it is over-strict it is a novelty
judgement rather than an experimental one.** The audit correctly refuses to upgrade
WEAKENED→REFUTED and names the exact physical difference for each near-miss (embedding-vs-block,
image-vs-LLM, matrix-vs-block, post-training-vs-pretraining) — that is precisely the
"overlap is not preemption" standard. What *is* over-strict is its self-imposed narrowing to
"the only surviving framing is a negative result / boundary characterization on 7B decoder-only".
Under the project rule (2-3 month gap = concurrent; only "completely identical" counts), B03
remains recoverable as a **follow-up that fixes the prior work's defects**: LLF and 2109.00267
never separated PPL recovery from parametric-knowledge loss, and LoRR's "full_layers is
detrimental" is a post-training claim with no pretraining control — which is exactly B03's
2x3 design. **B03 is not literature-killed. Its real blocker is cost (6 x 1B pretraining runs),
not novelty.** MAIN should not spend a GPU-hour on it tonight, but should also not archive it.

---

## Top 3 by decisiveness-per-GPU-hour

### #1 (LAUNCH THIS) — B04 cross-family Qwen replication, on `.21`

Why it is the best ratio: it is the *only* candidate where (a) the gate is a real kill test
named by the proposal itself, (b) every asset is on the same disk as an idle 8-GPU node, and
(c) the harness already exists and was already validated against a reproduction gate.

**What it decides.** B04's verdict explicitly limits itself: "**NOT** established beyond
OLMo-2-7B. Cross-family replication (Qwen prune-heal ladder) is the next kill test."
`STATUS.json.next_gates[1]` repeats it. So: if Spearman(core6_qwen, frac<0.005_qwen) comes out
near -1 at n>=5, B04's generality claim survives and B04 becomes promotable to `paper<X>`
(only the CPU novelty check would remain). If the sign *flips or goes flat* on Qwen, the
"damage compresses decision margins" story is revealed as OLMo-2-specific and B04's headline
must be narrowed to a single-model observation — that is a genuine kill of the general claim.
Either outcome changes what MAIN writes. Contrast with A01 gate-4 (CPU, decorative) and
B06 (see #3, likely already settled).

**Assets — all on wzc1, all verified by me just now:**

| what | path | size / evidence |
|---|---|---|
| base full-36 | `/apdcephfs_wzc1/.../models/Qwen--Qwen3-8b` | 5 safetensors shards, `model-0000{1..5}-of-00005.safetensors` (3996250744 + 3993160032 + 3959604768 + 3187841392 + 1244659840 B). Note `models/Qwen3-8b-local` is a **512-B symlink** resolving to this dir (`readlink -f` confirmed) — the same weights the shells were carved from |
| rung: f12k2 @ **step 2000** | `outputs/qwen3_minarch_armB_f12k2/final.pt` | **15G**, `step=2000`, arch_meta `keep_front_layers=12 n_fresh_layers=2 num_hidden_layers=14 from_scratch=False` |
| rung: f12k2 @ **step 20000** | `outputs/qwen3_minarch_armB_f12k2_20k/final.pt` | **15G**, `step=20000` |
| rung: f12k2 @ **step 200000** | `outputs/qwen3_minarch_armB_f12k2_200k/final.pt` | **45G** (includes `optimizer_state`; loader reads only `model_state`), `step=200000` |
| rung: f12k4 @ step 2000 | `outputs/qwen3_minarch_armB_f12k4/final.pt` | **17G**, `num_hidden_layers=16`, `n_fresh_layers=4` |
| rung: scratch14L @ step 2000 | `outputs/qwen3_minarch_scratch_f12k2/final.pt` | **15G**, `from_scratch=True` |
| harness | `scripts/eval_qwen3_probe2_downstream.py` | 32068 B; `ALL_TASKS` = exactly B04's core6; supports `--ckpt --keep_front_layers --n_fresh_layers --num_shards --shard_index --batch_size --add_bos --save_per_example --merge`; `results_root` default `qwen3_probe2_downstream_results` |
| runner template | `scripts/_run_paperF_bs16_ladder_73.sh` | 5237 B; has the 8/8-shard assert + per-task `n_scored` assert + skip-if-`summary.json` idempotency guard |
| analysis | `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_5rung.py` | 5775 B; margin = `norm_scores[gold] - max_{d!=gold} norm_scores[d]`, exact-permutation p |
| reproduction gate reference | `qwen3_mmlu_content_results/qwen3_base/summary.json` letter_acc **0.7293832787352229**; `.../qwen3_f12k2_inherit_s200k/summary.json` letter_acc **0.2514599059962968** | both n=14042; these are the paperB-documented sanity targets (~.7297 / ~.2495) so the loader path is already proven correct on these exact ckpts |
| datasets | `data/hf_datasets_cache/` on wzc1 | `Rowan___hellaswag`, `allenai___ai2_arc`, `ybisk___piqa`, `allenai___winogrande`, `allenai___openbookqa` all present |

**The one small fix required (why this is READY_AFTER_SMALL_FIX, not READY_NOW).** Three
mechanical edits to a copy of `_run_paperF_bs16_ladder_73.sh`: `ROOT` → the wzc1 path,
`BASE`/`PY`/`CONFIGS` → Qwen equivalents, and `eval_olmo2_probe2_downstream.py` →
`eval_qwen3_probe2_downstream.py` (+ `--results_root qwen3_probe2_downstream_results`).
No library code changes.

**One real gotcha I verified, and its fix.** The Qwen harness's `save_per_example` block
(lines 458-469) writes `option_scores` / `acc_norm_score` but **does NOT write
`norm_lens` / `norm_scores`** — that 2026-08-08 addition landed only in
`scripts/eval_olmo2_probe2_downstream.py` (lines 470-475). B04's analyzer prefers `norm_scores`
and silently falls back to raw `option_scores`, which would compute the **wrong margin**
(un-length-normalized) for Qwen and make the cross-family comparison invalid.
Fix without touching the harness: run `scripts/enrich_per_example_normscores.py` afterwards.
I verified it is safe to reuse across families — I extracted `load_task_examples` from both
harnesses and from the enrich script and they are **byte-identical (8536 chars each,
`IDENTICAL`)**, and the `item_id = shard_index + ei * num_shards` convention is identical in
all three (olmo line 410, qwen line 419, enrich line 151). So `norm_lens` depends only on the
dataset, never on the model family. The enrich script is idempotent (skips files that already
have `norm_lens`) and self-verifies that argmax(`norm_scores`) reproduces the harness's stored
`acc_norm_score`.

**Cost.** Measured from `logs/olmo2_downstream_7B_keep14_step200000_wzc1_v2_shard0.log`:
a full core6 8-shard pass on a 7B pruned OLMo on wzc1 ran **09:01:27 → 09:02:17 = ~50 s per
shard** (hellaswag 17.4 s, piqa 1.1 s, winogrande 0.6 s, openbookqa 0.8 s). Qwen-8B/14-16L is
comparable per-token; the dominant cost is **ckpt load** (15-45 GB fp32 state from cephfs,
one per rung). Budget ~20-40 min/rung wall including load, i.e. **~3-4 h for 5 rungs**, ~4-6 h
with headroom. `.21` has 2013 GB RAM and 256 cores (verified), so the 45 GB `_200k` ckpt load
is not a memory risk; 183 GB/card is far above what a 14L fp32-master/bf16-autocast forward needs.

**Design caveats MAIN must know before interpreting.** The Qwen ladder's damage axis is
**mostly heal-steps at fixed depth** (2000 / 20000 / 200000 at keep12+fresh2) plus one
depth/width variant (f12k4 = 16L) and one from-scratch control — it is *not* the OLMo-style
keep{8,10,12,14} depth ladder. That is still a valid damage ladder (B04's OLMo ladder also
mixed prune depth with heal steps, and shortgpt16 with keepN), and n=5 gives exact-permutation
resolution p >= 1/60 = 0.0167 two-sided, enough to detect rho = ±1 but **not** enough to reach
0.0028. Report it as an n=5 cross-family replication, not as a second p=0.0028 result. Adding
the base full-36 arm as a 6th rung is free (no ckpt) and restores n=6 → p can reach 0.0028.
**I recommend including base full-36 as rung 0**, which the launch plan below does.

### #2 (runner-up) — A02 phase-1 zero-training natural-task transfer gate

Most decisive gate in the tree: A02 states "若 paired quality CI 仍显著低于 0，则停止
'CoMem 优于 RAG' 的叙事" — that retires a whole narrative. And phase 1 claims **零新增训练**.

But it is **not** tonight-ready, and the blocker is cross-disk + a missing artifact:
- Write-LoRA is **zwfy6-only**: `outputs/qcmem_writepath_distill_qwen_j12_r32/` = **556 MB**
  per MAIN's §7 (I confirmed the wzc1 side is absent: `ls outputs/qcmem_writepath_distill_qwen_j12_r32/`
  → *No such file or directory*). 556 MB at ~37 MB/s 4-stream is only ~15 min, so this is a
  *minor* transfer, not blocking.
- The Read-LoRA side is the real gap. MAIN flagged `outputs/lora_best_ref/` as a suspicious
  512 B. On wzc1 that path **does not exist at all** (`ls` → No such file or directory). The
  flagship read adapter that *does* exist on wzc1 is
  `outputs/qcmem_distill_qwen_j12_r32_4k/final/adapter_model.safetensors` = **232829168 B**
  (~222 MB) + `adapter_config.json` (1341 B) + `README.md` (5288 B). Whether that is the
  paper's flagship Read-LoRA or a different distill run is **unsure** — I did not resolve it,
  and A02's 5-config design needs the identity pinned before the comparison means anything.
- So A02 = `BLOCKED_MISSING_ASSET` until someone identifies the canonical Read-LoRA. Cheap to
  resolve (a provenance grep, no GPU), and then A02 becomes the best *next* launch.

### #3 (runner-up, likely already settled — worth telling MAIN before it spends anything)

B06's kill condition is: "统一 harness 后增益消失" — the +17.88 lift must survive rejudging
the canonical predictions under one harness, and its own next-step #1 is "消除 8.11 vs 13.29 drift".

**I think this is already effectively answered on wzc1 and needs no GPU.** All three relevant
score files are present with judge caches:
- `locomo_results/hcache_j12_noLoRA_chatFALSE/scores.json` → `overall_judge` **13.293051359516618**, n=1986, `judge_model` gpt-4o
- `locomo_results/hcache_j12_LoRA_chatFALSE/scores.json` → `overall_judge` **31.16817724068479**, n=1986, gpt-4o
- `locomo_results/hcache/scores.json` (the older/canonical run) → `overall_judge` **12.28600201409869**, n=1986, gpt-4o

So the drift B06 worried about is **12.286 vs 13.293 = ~1.0 pp**, while the claimed effect is
**+17.88 pp** — an order of magnitude larger than the harness drift. And the two chatFALSE arms
were generated by `scripts/_p1_hcache_lora_toggle.sh` (2139 B) as a genuine single-variable
toggle (same `--baseline hcache --resume_j 12`, same model, same 3 shards; the only difference
is `--lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final --force_lora_with_baseline`).
Predictions (`preds_shard{0,1,2}of3.jsonl`, ~340 KB each) and `judge_cache.jsonl` (~109 KB)
are on disk for all three, so a same-harness rescore is **CPU + judge-API only**. MAIN should
do this as a CPU task, not a GPU launch. B06's remaining GPU work (BABILong/RULER/LongEval
replication, second compressor) is real but is *generalization*, not the kill gate.

---

## Explicitly NOT recommended tonight, with reasons

- **A01 gate-4** (C4 aggregation pre-registration): genuinely ready and genuinely cheap, but
  it is **CPU-only and cannot kill anything** — A01's own text says its purpose is to stop
  selective reporting of the 10x figure. Correct to do, wrong to spend an idle 8-GPU node on.
  Evidence in place: `evidence/null_calibration_p1_nperm2000.json` (24935 B),
  `evidence/null_calibration_obs4_nperm2000.json` (63143 B),
  `code/build_null_calibration_table.py` (53411 B).
- **A01 gate-2** (second MC benchmark): plausible GPU job, and 18 `per_example_mmlu.jsonl`
  files exist under `olmo2_mmlu_content_results/`. But A01's kill condition requires
  "第三家族和第二 benchmark **均**不复现 interface failure" — one leg alone cannot kill, and
  the harness's dual letter/content interface is MMLU-specific (`eval_olmo2_mmlu_content.py`
  is hardwired to the 14042-item MMLU set; there is no arc/obqa dual-interface variant). So
  gate-2 is `BLOCKED_NEEDS_CODE` for a new benchmark's letter interface.
- **A04**: 24 x 1B trainings. Most expensive thing in the tree; its own STATUS carries
  `"warning": "Do not use the historical depth ladder as a controlled scaling law."`
- **B03**: alive (see §4) but 6 x 1B pretraining runs.
- **B07 / B08 / B09**: no gate that can kill a scientific claim; B07 is systems measurement,
  B08 is a 3-item portfolio, B09 is method development (though B09 is the most *complete*
  proposal in the tree — 18025 B PROPOSAL + 16669 B NOVELTY + 6037 B SOURCES).

## Environment notes worth recording

- `.21` verified: 8x `NVIDIA L20A`, 0 MiB / 0% on all 8; `/opt/conda/envs/torch-base/bin/python`
  = torch **2.13.0**, `cuda.device_count()` **8**, transformers **4.57.6**; 2013 GB RAM, 256 cores.
- `.21` PID 25999 = `tools/prepare_dolmino_llama2.py --stage download`, ELAPSED 01:01:46,
  CPU+network only, **no GPU processes** (`nvidia-smi --query-compute-apps` returns empty).
  A GPU eval will not conflict; do not start large downloads alongside it.
- `.82` verified: 8x H20, 0 MiB / 0% on all 8; git HEAD `2d98c5a` (behind wzc1, as documented).
- Because `.21` is wzc1, every file check above was a plain local `ls`/`du` — no ssh needed,
  and no cross-disk cost for the recommended job.
