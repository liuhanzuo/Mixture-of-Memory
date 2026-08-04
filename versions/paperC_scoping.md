# Paper C Scoping — "Shallow Regrown Cap on a Frozen Trunk" (FINETUNE paper)

> Planning-only doc (2026-07-26). No GPU touched, no run launched. All GPU-hour
> numbers are estimates with stated assumptions; the freeze-front recipe already
> exists in-repo (task #59), so infra risk is LOW and cost estimates lean on the
> **measured** `train_olmo2_arch_probe2.py` throughput (1.32 s/step, 8×B200).

## 0. Positioning vs Paper A / Paper B

| | object | training regime | data |
|---|---|---|---|
| Paper A (CoMem/QCMem) | depth-split + retrieval memory | distill adapter (LoRA) | long-ctx |
| Paper B (OLMo-2 prune-heal) | keep front-N, DROP top, graft K fresh, continue **PRETRAIN** all | continue-pretrain (heal base LM) | Dolmino DCLM |
| **Paper C (this doc)** | keep front-j (**FREEZE**), DROP top L-j, graft K fresh, **FINETUNE only the K** | supervised finetune, front frozen | **downstream / instruction** |

**Core object of Paper C:** a *shallow regrown cap* — freeze a pretrained trunk's
front `j` layers, discard the top `L-j`, graft `K` fresh trainable transformer
layers (K small), and finetune **only** those K (+ final norm + lm_head) on a
downstream task. Distinct from Paper B (which continue-*pretrains* to heal the base
LM and trains everything) — Paper C freezes the trunk and finetunes a tiny cap for
task adaptation.

## 1. Reusable project assets (already on disk — this is why Paper C is cheap)

- **The freeze-front + fresh-graft training path ALREADY EXISTS.**
  `scripts/train_olmo2_arch_probe2.py` implements exactly the recipe:
  - `--keep_front_layers j` transplants front-j layers + embed/norm/lm_head from base.
  - `--n_fresh_layers K` grafts K freshly Olmo2-initialised layers on top (asserted
    fresh via `_assert_fresh_init`).
  - `--freeze_front` freezes the inherited front layers, trains **fresh + norm +
    lm_head only** (verbatim Paper C construction; log confirms `frozen=2833.5M
    trainable=1226.9M`).
  - `--from_scratch` random-inits ALL layers = the depth-matched (j+K) control.
  - differential LR (`--lr` fresh / `--lr_inherited` inherited), grad-ckpt, fp32 master.
  - Launcher `scripts/run_olmo2_7B_keepN.sh` — task **#59** runs it TODAY as
    `FREEZE_FRONT=1 KEEP=14 N_FRESH=2` on LOCAL 8×B200 (step 54280/200000, healthy).
  - **The ONLY change for Paper C** = swap `--data_path` from the Dolmino npy to a
    downstream/instruction tokenised set, and cut `--max_steps` from 200k → ~1–3k.
- **P2 probe infra** (`scripts/probe_linguistic_layerwise.py`): edge-probing
  (POS/DEPREL/CoLA/WiC/SST2/RTE) + logit-lens + `--task knowledge_logit_lens`,
  forward-only. Results already computed: `results/knowledge_logit_lens_OLMo-2-1124-7B.json`
  (knowledge onset layer 18 = 0.562L, sat95 = 0.594L), `..._Qwen3-8b-local.json`
  (sat95 0.694L), `results/probe_linguistic_{qwen3,llama3}_8b.json`.
- **Downstream eval harness** (`scripts/eval_olmo2_probe2_downstream.py`): zero-shot
  MC on hellaswag/arc/piqa/winogrande/openbookqa + knowledge (mmlu/lambada/boolq/
  csqa/siqa), pulls via HF `load_dataset` (proxy) — reuse verbatim for eval arms.
- **LoRA infra**: `scripts/train_qcmem_distill.py` already uses `peft.LoraConfig`
  (`--lora_rank`, `--lora_alpha`, `target_modules`) — a param-matched LoRA baseline
  is a small fork, not new code.
- **Base models on disk**: `models/OLMo-2-1124-7B` (32L, hidden 4096) — primary,
  because all of the above already target it; `models/Qwen3-8B-Base` (36L, hidden
  4096) — secondary for generality; also OLMo-2-0425-1B, Qwen3-4B for cheap sweeps.
- **Finetune data on disk**: `data/squad_train.jsonl`/`squad_val.jsonl` (QA),
  `data/slimpajama_chunks_2048_*` / `data/pg19_*` / `data/dolmino_*` (pretrain-style).
  **NOT on disk**: GLUE, Alpaca/Tulu instruction mixes — pullable via `load_dataset`
  over hy-proxy (same path the eval harness already uses), then tokenised to a
  `.npy` shard by adapting `scripts/tokenize_dolmino_olmo2.py`.
- **Existing checkpoints usable for FREE preliminaries**: Paper B `outputs/
  olmo2_probe2_7B_keep14fresh2_freezefront/step*.pt`, `..._keep14fresh2` (healed),
  `..._fromscratch` — seed the P-C1 ΔW/CKA measurement at zero training cost.

### Throughput anchors (measured / estimated)
- **Measured**: OLMo-2-7B graft (16L, keep14+fresh2), seq2048, eff_bs128, fp32-master,
  grad-ckpt, 8×B200/L20A = **1.32 s/step**, peak 87.8 GB/card.
- **Estimated**: full-FT of original 32L ≈ 2.6 s/step (2× depth); 8×H20 ≈ 2.3× the
  B200 wall (bf16 throttle + fp32 master, per Paper A #67 finding).
- Finetune step budget assumed **2000 steps** (≈512M tok) default; instruction-mix
  up to 3000; GLUE-style ~1000. Derived per-run cost:

| run type (2000 steps) | 8×B200 GPU-h | 8×H20 GPU-h |
|---|---|---|
| graft 16L (freeze-front / from-scratch) | ~6 | ~14 |
| full-FT original 32L | ~12 | ~27 |
| LoRA on 32L (fwd full, light bwd) | ~11 | ~25 |

---

## P-C1 "Finetuning lives in the top layers" (mechanism + construction)

**Claim.** (measurement) During standard FT, per-layer ‖ΔW‖ and representation drift
(CKA) concentrate in top layers; front layers barely move. (construction) Therefore
freeze front-j, regrow K fresh top layers, FT only those → matches full-FT, beats
param-matched LoRA (esp. under distribution shift), beats depth-matched from-scratch
(j+K) at equal budget.

### Experiment list

| # | run | model | arm | data | steps | notes |
|---|---|---|---|---|---|---|
| M1 | ΔW/CKA analysis (FREE) | OLMo-2-7B | base vs existing Paper-B ckpts | — | 0 | forward-only; per-layer ‖ΔW‖ + CKA drift; new ~120-line script |
| M2 | ΔW/CKA on true full-FT | OLMo-2-7B | full-FT | T1 | — | derived from A1 ckpt, no extra train |
| A1 | full-FT | OLMo-2-7B 32L | full-FT | T1 (instruct) | 2–3k | baseline |
| A2 | LoRA param-matched | OLMo-2-7B 32L | LoRA r≈256–512 (match K layers) | T1 | 2–3k | + reference r=16/64 |
| A3 | from-scratch (j+K) | 16L random | from_scratch | T1 | 2–3k | depth/budget-matched control |
| A4 | **freeze-graft (HERO)** | keep14+fresh2 | freeze_front | T1 | 2–3k | the proposed method |
| B1–B4 | repeat A1–A4 | OLMo-2-7B | 4 arms | T2 (domain QA / GLUE-MNLI) | 2k | generality |
| C1–C4 | repeat A1–A4 | OLMo-2-7B | 4 arms | T3 (2nd distribution-shift, e.g. code/math) | 2k | distribution-shift is where hero should beat LoRA |
| D1–D5 | j×K depth sweep | keep{14,18,22}+fresh{2,4} | freeze_front | T1 | 2k each | locate optimal (j,K); D=6−1 hero |
| E* | eval | all ckpts | — | test split | — | reuse `eval_olmo2_probe2_downstream.py` + task metric |

Datasets: **T1 = instruction-mix SFT** (Tulu-3/Alpaca slice, load via proxy →
tokenise; strongest distribution shift → best case for "top moves most"). **T2 =
downstream QA** (`data/squad_train.jsonl` already on disk) or GLUE-MNLI (proxy).
**T3 = 2nd shift** (a code or GSM8k slice via proxy) to make the "beats LoRA under
shift" claim on ≥2 domains.

### P-C1 resource

| item | count | GPU-h each (B200) | subtotal |
|---|---|---|---|
| full-FT (A1,B1,C1) | 3 | 12 | 36 |
| LoRA (A2,B2,C2) | 3 | 11 | 33 |
| from-scratch (A3,B3,C3) | 3 | 6 | 18 |
| freeze-graft (A4,B4,C4) | 3 | 6 | 18 |
| depth sweep (D1–D5) | 5 | 6 | 30 |
| eval runs (~15) | 15 | ~1 | 15 |
| ΔW/CKA (M1,M2) | 2 | ~2 | 4 |
| **P-C1 TOTAL** | **20 train + 15 eval + 2 analysis** | | **≈154 GPU-h** |

- Wall on ONE 8×B200 node (serial): ~19 h ≈ **~1 day**. On 8×H20: ~2.2 days.

**Novelty gut-check.** ⚠️ The construction half is **partially occupied**: Zhang et
al. 2021 ("Revisiting Few-sample BERT FT") re-init top-K encoder layers before FT;
Surgical Fine-Tuning (Lee et al., ICLR'23) tunes a *subset of existing* blocks;
gradual unfreezing (ULMFiT). Differentiators: (a) DISCARD top L-j + regrow K where
K<L-j → a **net-shallower** finetuned net (compute win), not mere re-init; (b)
frozen trunk (Surgical FT still updates chosen blocks); (c) decoder LLM @ 7B, not
BERT. **P-C1 alone is at novelty risk** — it needs P-C2's predictive hook to
differentiate.

---

## P-C2 "Adaptation depth is predictable" (the differentiated hook)

**Claim.** The cut j and regrowth K needed to adapt to a task are predicted by a
cheap probe of the *base* model — an "adaptation-onset depth" (where representations
start to move under FT) reusable from the existing logit-lens / linguistic-probe
infra. Multiple tasks of varying "computational depth" → per-task optimal-K curve →
correlation between probe-predicted depth and empirical optimal K.

### Experiment list

| # | run | purpose | steps | notes |
|---|---|---|---|---|
| P1 | logit-lens + edge-probe (FREE) | base "onset/sat" depth per model | 0 | already have OLMo (onset L18/0.562L) + Qwen3 |
| P2 | **adaptation-onset probe** (NEW) | per-layer CKA drift base↔FT-ckpt | 0 | forward-only, reuses A1/B1/C1 full-FT ckpts (~2 GPU-h ea) |
| S1 | optimal-K sweep, task τ1 | j×K grid {3 j × 2 K} = 6 graft runs | 2k | τ1 = shallow-adapt task |
| S2 | optimal-K sweep, task τ2 | 6 graft runs | 2k | τ2 |
| S3 | optimal-K sweep, task τ3 | 6 graft runs | 2k | τ3 = deep-adapt (reasoning/code) |
| S4 | optimal-K sweep, task τ4 | 6 graft runs | 2k | τ4 |
| R1 | correlation analysis (CPU) | probe-depth vs empirical optimal-K | 0 | the paper's key scatter/curve |

### P-C2 resource

| item | count | GPU-h each (B200) | subtotal |
|---|---|---|---|
| optimal-K sweeps (S1–S4) | 24 graft runs | 6 | 144 |
| adaptation-onset probes (P2) | ~4 | 2 | 8 |
| eval runs | ~24 | 1 | 24 |
| logit-lens/edge-probe (P1) | 0 (done) | — | 0 |
| **P-C2 TOTAL** | **24 train + probes/eval** | | **≈176 GPU-h** |

- Wall on ONE 8×B200 node: ~22 h ≈ **~1 day** (spreadable across the 4 tasks/nodes
  → ~6 h if 4 nodes). **Heavily overlaps P-C1's depth sweep** (S1 ≈ D1–D5) → when
  done after P-C1, marginal cost is only the 3 extra tasks' sweeps (~18 runs ≈ 108 GPU-h).

**Novelty gut-check.** ✅ **This is the genuinely novel hook** — "predict the cut/
regrowth depth from a cheap forward-only probe of the base model, before any FT."
No close prior work does a *predictive* depth selection from an intrinsic base-model
probe; Surgical FT selects blocks by post-hoc gradient/Fisher (needs FT signal),
not an a-priori probe. Ties directly to the project's own logit-lens results. **Lead
the paper with P-C2.**

---

## P-C3 "Modular shallow caps on a frozen trunk" (systems / multi-task)

**Claim.** One frozen shared front-j trunk + N small task-specific top-caps =
memory-efficient multi-task serving; swap caps, share trunk. Compare vs N full models
/ N LoRAs on serving memory & throughput.

### Experiment list

| # | run | purpose | steps | notes |
|---|---|---|---|---|
| G1–G6 | N=6 task caps | freeze_front K=2 per task | 2k | tiny cap per task |
| H1–H6 | N full FT models | baseline | 2k | worst memory (N×7B) |
| L1–L6 | N LoRAs | baseline | 2k | reuse train_qcmem_distill LoRA |
| SV1 | serving bench (NEW harness) | trunk-shared vs N-model vs N-LoRA mem+tput | 0 | ~150-line multi-cap serving script (coder) |
| E* | eval per task | correctness parity | — | reuse downstream harness |

### P-C3 resource

| item | count | GPU-h each (B200) | subtotal |
|---|---|---|---|
| task caps (G1–G6) | 6 | 6 | 36 |
| full-FT baselines (H1–H6) | 6 | 12 | 72 |
| LoRA baselines (L1–L6) | 6 | 11 | 66 |
| serving bench (SV1) | 1 | ~8 | 8 |
| eval runs | ~18 | 1 | 18 |
| **P-C3 TOTAL** | **18 train + bench/eval** | | **≈200 GPU-h** |

- Wall on ONE 8×B200 node: ~25 h ≈ **~1.2 days** (+ new serving harness dev). If G/L
  arms reuse P-C1's full-FT & LoRA runs, marginal training cost drops to the 6 caps
  (~36 GPU-h) + serving bench.

**Novelty gut-check.** ⚠️ **Most occupied / incremental.** Shared-backbone + swappable
task modules is exactly the LoRA/adapter multi-task-serving thesis (S-LoRA, AdapterHub,
"one backbone many adapters"). Frozen-trunk + fresh-transformer-cap at LLM scale is a
mild reframing; the memory story (share a 7B trunk, N tiny caps) is real but reads as
a systems footnote. **Recommend DROP as a standalone proposition; keep as a 1-figure
systems appendix if P-C1/P-C2 land.**

---

## SYNTHESIS

### Minimal viable Paper C
Story = **P-C2 hook (predictable adaptation depth)** carried by **P-C1 construction**
on ONE base model (OLMo-2-7B, all infra + baselines + probe already exist).

- Base: OLMo-2-7B only.
- Tasks: **T1 = instruction-mix SFT** (distribution-shift showcase) + **T2 = one more
  task** (SQuAD on-disk or GLUE-MNLI) for generality.
- T1 arms: full-FT / LoRA(param-matched) / from-scratch(j+K) / freeze-graft(hero) = 4.
- T1 depth sweep to locate optimal (j,K) **and validate the P-C2 prediction**: 3 j ×
  2 K = 6, minus hero = 5.
- T2 generality: full-FT / LoRA / freeze-graft = 3.
- P-C2 probe: existing logit-lens (FREE) + 1 adaptation-onset CKA analysis (reuse T1
  ckpts, ~2 GPU-h).

| item | runs | GPU-h |
|---|---|---|
| T1 4-arm (12+11+6+6) | 4 | 35 |
| T1 depth sweep | 5 | 30 |
| T2 (12+11+6) | 3 | 29 |
| eval (~12) | 12 | 12 |
| ΔW/CKA + adaptation-onset probe | 2 | ~4 |
| **MINIMAL VIABLE TOTAL** | **12 train + evals** | **≈110 GPU-h** |

- **Wall-clock on ONE 8×B200 node: ~14 h serial ≈ ~0.6 day**; with eval serialisation
  and setup, budget **~1 day**. On 8×H20: ~1.3 days.

### Recommended proposition
**P-C1 + P-C2 combined, led by P-C2.** P-C1 alone is novelty-thin (Zhang'21 re-init /
Surgical FT); P-C2's a-priori probe-predicted depth is the defensible differentiator.
**Drop P-C3** to (at most) a systems appendix — it is the most occupied by the LoRA
multi-task-serving literature and needs a new serving harness for marginal payoff.

### Dependency / parallelism (do NOT double-count)
- **One grafted checkpoint → many evals.** Each freeze-graft ckpt is evaluated across
  its task's metrics with `eval_olmo2_probe2_downstream.py` at ~1 GPU-h — evals are
  cheap and never re-train.
- **P-C1 depth sweep ⊇ P-C2 τ1 sweep.** D1–D5 == S1. If P-C1 runs first, P-C2's
  marginal cost is only the 3 extra tasks' sweeps (~108 GPU-h) not 176.
- **P-C1 full-FT & LoRA arms ⊇ P-C3 baselines.** H/L arms reuse A1/A2-style runs; P-C3
  marginal = 6 caps + serving bench (~44 GPU-h) not 200.
- **M1/M2 ΔW-CKA reuse existing Paper-B checkpoints** for a free preliminary; the
  true full-FT ΔW (M2) is derived from the A1 checkpoint — no dedicated training.
- **P-C2 probe P1 is already done** (logit-lens JSONs on disk) — zero cost.
- Combined P-C1+P-C2 (dedup) ≈ 154 + 108 ≈ **~260 GPU-h**; +P-C3 appendix (dedup) ≈
  +44 → **~300 GPU-h** for the full three-legged version.

### When can each start, and the cheapest FIRST
Available nodes (from GPU_STATUS 2026-07-26): **LOCAL 8×B200** busy on freeze_front
#59 — ~53 h remaining (≈2.2 days) → frees ~**2026-07-28**; **.82 8×H20** busy on P3.2
YaRN, frees ~**tomorrow AM (2026-07-27)**. (User nodes .73/.104/.252 off-limits;
dllm 29.162.226.120 never touch.)

- **CHEAPEST FIRST (zero training GPU, gates the whole paper): M1 — the P-C1
  measurement-half ΔW-norm + CKA analysis on EXISTING OLMo-2 checkpoints** (base vs
  Paper-B healed keep14 / freeze_front / from_scratch). Forward-only, ~2–4 GPU-h,
  needs only a ~120-line analysis script (coder). Confirms "top layers move most"
  before spending any training budget — runnable the moment **.82 frees (tomorrow AM)**,
  or opportunistically on any transient idle card.
- **Next on .82 (H20, ~tomorrow AM): T1 4-arm single-task P-C1** (the make-or-break
  "does freeze-graft match full-FT & beat LoRA under shift?"). ~35 GPU-h B200 /
  ~80 GPU-h H20 → ~1 day on the H20 node.
- **On LOCAL when #59 finishes (~2026-07-28): the T1 depth sweep + T2 generality +
  P-C2 adaptation-onset probe** (B200 is faster; sweep is 16L-graft = cheap).

**Bottom line:** kick off with the free M1 measurement analysis (script it now, run
it the instant a card frees), then the T1 4-arm on .82 tomorrow, then the sweep on
LOCAL after #59. Minimal viable Paper C ≈ **~110 GPU-h ≈ ~1 day** on one 8-GPU node.

---

## P-C1 RESULTS — SQuAD 4-arm slice (2026-08-03, task #92)

> First empirical P-C1 slice: the **T2/SQuAD** 4-arm (not T1 instruction-mix). Trained
> on `data/squad_sft_olmo2_2048_train.npy` (770 chunks / 1.58M tok packed), eff_bs=128,
> seq2048, **max_steps=1000** (⚠️ ≈166 epochs → over-training regime, capacity-bound).
> Eval: SQuAD dev n=2000, base protocol (chat_template=False, add_bos=0, greedy,
> first-line completion, SQuAD-normalized EM + token-F1) — **口径 verified identical
> across all 4 arms** (researcher a64858a, all numbers checked against ground-truth
> `paperC_squad_results/*/summary.json`). Run on .104 8×H20; orchestrator
> `scripts/paperC_pc1_orchestrate.sh`.

| arm | construction | EM | F1 | trainable |
|-----|-------------|-----|-----|-----------|
| **A2_lora_r160** | full 32L + LoRA r=160 (param-matched) | **0.6590** | **0.7139** | 399.8M |
| BASE_ref | raw OLMo-2 32L, **NO SFT** (ceiling ref, confounded) | 0.3385 | 0.3999 | — |
| **A4_hero (freeze-graft)** | keep14 frozen + fresh2, FT fresh+norm+lm_head | 0.2930 | 0.2970 | 1226.9M |
| A3_fromscratch | 16L random-init, all trainable | 0.2605 | 0.2612 | ~4.06B |

**(A1 full-FT-32L NOT run — H20 OOM (7B fp32-AdamW); deferred to B200 as true ceiling.)**

### Findings (researcher a64858a, verified)
1. **A4 > A3 (+3.25pp EM / +3.6pp F1) is statistically significant** — paired McNemar
   χ²_cc=11.41, **p=7.3e-4**; paired bootstrap 95% CI on EM diff = **[+1.4pp, +5.2pp]**
   (excludes 0). **NOT noise.** This is the clean controlled contrast (matched 16L
   depth / data / steps / eval; only front-block inherit+freeze vs random-init differs)
   → **genuine positive evidence that freeze-graft > from-scratch at matched depth.**
2. **BASE_ref is CONFOUNDED, not a fair peer**: differs from A4 on TWO axes (32L vs 16L
   AND no-SFT vs SFT). A4 < BASE_ref does **NOT** mean freeze-graft hurt — it means "a
   zero-effort intact 32L base out-QAs a pruned-to-16L-then-SFT'd model." BASE_ref =
   intact-model capability ceiling reference only.
3. **Mechanism**: SFT mainly teaches the terse SQuAD answer format (A2 base→LoRA jumps
   +32pp EM, p≈1e-111). F1−EM gap ≈0 on A4/A3 (0.4pp / 0.07pp) vs 6.1pp on BASE_ref →
   format-SFT **succeeded** on the pruned arms; their low score is a **genuine
   depth/capacity deficit**, not a formatting failure. 32→16L pruning cost > 1000-step
   SFT recovery. At ~166 epochs the 16L arms are capacity-bound → **more steps won't help.**

### Framing decision (MAIN, adopting researcher rec #1, confidence very_high)
- **Defensible headline**: *"conditional on aggressive pruning to 16L, freeze-graft
  finetuning significantly beats from-scratch (EM 0.293 vs 0.261, p=7e-4)."* A4-vs-A3
  is the controlled comparison; A2 (and future A1) are ceilings; BASE_ref is the
  intact-model reference reported honestly with the 2-axis caveat.
- **Do NOT** claim freeze-graft yields a competitive *absolute* QA model here (intact
  base beats it). The paper must frame the axis as "which init recovers best under
  aggressive pruning," not "freeze-graft is a general FT method." **A single
  format-dominated benchmark (SQuAD EM) is a thin base** → needs a capability-sensitive
  second task + a depth-sweep to be robust (staged below).

### P-C1 follow-ups queued (researcher recs, MAIN-scheduled)
- **[task #132] second-task capability eval** (rec #5, high): MMLU-MC + closed-book QA
  on A4/A3/A2/BASE via existing `eval_olmo2_probe2_downstream.py` + `eval_olmo2_closedbook_qa.py`
  — tests knowledge/reasoning retention (depth-sensitive, format-SFT can't paper over).
  Eval-only → H20 when a slot opens (queues behind Paper A on .104 per H20=PaperA-first).
- **[task #133] depth-sweep** (rec #3, high, highest-value): freeze-graft vs from-scratch
  at keep∈{20,24,28}+fresh2, same recipe → turns single point into a curve; if A4>A3 gap
  persists/widens with depth it's a robust method claim. Training → B200 when P0.5 frees.
- **[task #134] A1 full-FT-32L** (rec #4, medium): true ceiling; B200 (fp32-AdamW fits
  183GB), 1000 steps ~15min. Low narrative-criticality (A2 LoRA already a strong ceiling).
- **Do NOT just add steps** (rec #2, high) — over-trained; larger/diverse SFT mix (rec #6,
  medium) is the lever if absolute numbers matter.
- raw: `paperC_squad_results/{A2_lora_r160,A3_fromscratch,A4_hero,BASE_ref}/{summary.json,per_example_shard0of1.jsonl}` (on .104 diskB, gitignored).

---

## P-C1 SECOND-TASK RESULTS — capability battery (2026-08-04, task #132)

> The follow-up the researcher ranked #1: does the SQuAD A4>A3 edge survive on
> **capability-sensitive** benchmarks that format-SFT cannot paper over? Run on
> .73 8×H20, ~31 min wall, 64 units (4 arms × 2 eval types × 8 shards), **0 FAIL /
> 0 SKIP**, all 8 cells with 8/8 shards + merged per-example dumps.
> Protocol identical across arms: **chat_template=False, add_bos=0**, MC =
> likelihood argmax over teacher-forced continuation logprob (acc + char-normed
> acc_norm, no generation), closed-book QA = greedy, zero-shot, no retrieval.
> Arch verified: A4/A3 = 16L (`mode=pruned`, strict load, 179 tensors), A2/BASE =
> 32L; DUP_CHECK clean (no two arms identical).

| benchmark | n | **A4_hero** 16L | **A3_scratch** 16L | A2_LoRA 32L | BASE 32L |
|---|---|---|---|---|---|
| **mmlu** (acc) | 14042 | **0.2596** | 0.2474 | 0.5935 | 0.6056 |
| hellaswag (acc/accn) | 10042 | 0.2686/0.2907 | 0.2566/0.2593 | 0.5128/0.6668 | 0.6043/0.8052 |
| arc_challenge | 1172 | 0.2142 | 0.2014 | 0.4172 | 0.5392 |
| arc_easy | 2376 | 0.2824 | 0.2567 | 0.6721 | 0.8232 |
| piqa | 1838 | 0.5163 | 0.5239 | 0.7040 | 0.8090 |
| winogrande | 1267 | 0.5178 | 0.5083 | 0.5785 | 0.7443 |
| openbookqa | 500 | 0.1400 | 0.1700 | 0.3140 | 0.3720 |
| lambada (greedy) | 5153 | 0.0008 | 0.0000 | 0.6326 | 0.7314 |
| boolq | 3270 | 0.3896 | 0.4052 | 0.7612 | 0.8153 |
| commonsense_qa | 1221 | 0.1761 | 0.1777 | 0.5225 | 0.6634 |
| social_iqa | 1954 | 0.3270 | 0.3270 | 0.4222 | 0.5010 |
| **popqa** (EM) | 14267 | 0.0000 | 0.0000 | 0.1483 | 0.2474 |
| **triviaqa** (EM) | 17944 | 0.0006 | 0.0001 | 0.5411 | 0.6356 |
| **nq_open** (EM) | 3610 | 0.0000 | 0.0000 | 0.1360 | 0.2047 |

### Answer: A4>A3 survives **directionally** but is **NOT a capability win**

Paired McNemar + paired bootstrap, same 口径 as the SQuAD slice
(`logs/paperC_132_paired_stats.log`):
- **Pooled over all 14 benchmarks (n=78,656): A4>A3 +0.39pp, χ²_cc=10.45,
  p=1.2e-3, 95% CI [+0.15,+0.63]pp** — significant, sign agrees with SQuAD.
- But the effect is **~8× smaller than SQuAD's +3.25pp** and non-uniform: A4 wins
  mmlu (+1.22, p=0.015), hellaswag (+1.19, p=3.5e-4), arc_easy (+2.57, p=0.015);
  A4 **loses** boolq (−1.56, p=0.010); **9 of 14 cells null**.

### ★★ The decisive caveat: both 16L arms are at/below chance almost everywhere

Binomial z vs chance (`logs/paperC_132_interpret.log`): A4 MMLU 0.2596 is only
z=+2.6 above the 0.25 guess floor; A3 0.2474 is z=−0.7 (**at chance**). Both
A4/A3 are AT_CHANCE or BELOW on piqa, winogrande, social_iqa, commonsense_qa,
arc_challenge, openbookqa, boolq. Closed-book QA **collapses entirely**: EM≈0
with output degeneration — A4 emits the identical Chinese refusal string
"根据提供的信息无法回答这个问题" on **52.4% of PopQA / 53.2% of NQ-open**
(A3 52.3%/34.2%) vs 2.2%/1.4% for BASE. ⇒ the pruned arms retain essentially
**no parametric knowledge** and their decoder has partially collapsed.

### Framing consequence (MAIN, binding)

**The SQuAD A4>A3 advantage is NOT knowledge/reasoning recovery.** Freeze-graft's
inherited front block gives a small but real edge *in the near-chance regime*
(pooled p=1.2e-3; A4 is the only 16L arm measurably above chance on MMLU), but at
keep14 **both arms are so far below the capability floor that the contrast is
between two broken models**. This *supports* the existing narrow framing ("which
init recovers best under aggressive pruning") and argues **against any claim of
general capability retention**. Do not upgrade this to a capability claim.

⇒ **This makes task #133 (depth sweep keep∈{20,24,28}) the critical experiment**:
only at shallower pruning can we test whether the A4>A3 gap widens into a genuine
capability effect rather than a near-chance artefact.

Ceilings behave sanely and bracket everything. Note **A2 < BASE on every
capability benchmark** (MMLU 0.5935 vs 0.6056, triviaqa EM 0.541 vs 0.636) →
SQuAD-format LoRA SFT **cost** general capability, consistent with
format-specialization. **A1 (full-FT 32L) still absent** (only `arch_meta.json`,
no `final.pt` — never trained, H20 OOM); correctly detected + dropped, NOT
substituted. Needs B200 → task #134.

**Ops notes (important)**: (1) on **.73** `/apdcephfs_wzc1` is a **symlink to
zwfy6** — the wzc1 path string resolves there but is a *different physical disk*
from LOCAL/.252; PROJECT_ROOT for .73 is the zwfy6 path. (2) The pre-existing
`scripts/_run_paperC_secondtask_8gpu.sh` (cd0f527) would have failed / caused an
8-way download race: it overrides `HF_HOME` to a project cache lacking
mmlu/hellaswag/arc/popqa/triviaqa/nq_open. The new runner
`scripts/_run_paperC_132_secondtask.sh` (commit 69131ad) leaves the default cache
alone and pre-warms datasets on CPU.

- raw: `paperC_secondtask_results/{A4,A3,A2,BASE}_{downstream,closedbook}/summary.json`
  + `shard{0..7}of8.json` + `per_example_*.jsonl` (~200 MB, **zwfy6**, gitignored);
  logs `logs/paperC_132_{secondtask,secondtask_summary,paired_stats,interpret}.log`.

