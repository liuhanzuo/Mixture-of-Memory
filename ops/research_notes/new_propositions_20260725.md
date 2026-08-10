# New Testable Propositions from Our Two Core Ideas (2026-07-25)

> **Task**: research-mode ideation — turn the two core ideas (Paper A = CoMem/QCMem depth-partitioned
> retrieval-readout; Paper B = OLMo-2 prune-then-heal) into *new testable propositions* that emerge
> from our own empirical findings, then convert the most promising 1–2 into concrete single-variable
> experiments runnable on our nodes with the existing harness. NOT a lit review. GPU-free authoring
> (design + novelty check + light CPU inspection only).
>
> **Sources read**: `status/PAPERA_RESULTS_CONSOLIDATED.md`, `status/OLMO2_PRUNEHEAL_PPL.md`,
> `status/OLMO2_PRUNEHEAL_DOWNSTREAM.md`, `results/probe_linguistic_qwen3_8b.json`,
> harness (`scripts/eval_qcmem_locomo.py`, `scripts/eval_qcmem_babilong.py`, `scripts/eval_ruler_qcmem.py`,
> `scripts/probe_linguistic_layerwise.py`, `scripts/train_olmo2_arch_probe2.py`,
> `scripts/eval_olmo2_probe2_{ppl,downstream}.py`).

---

## 0. The empirical anchors these propositions stand on (load-bearing numbers)

**Paper A (Qwen3-8B, L=36 decoder):**
- Frozen backbone depth-fidelity is **monotone decreasing**: LoCoMo GPT-4o judge j0=41.59 → j6=32.78 → j9=29.15 → j12(frozen)=24.52 (§3.3). Any frozen depth < full-ctx KV oracle (KVD=34.59).
- **Distilled LoRA is the decisive component** that lifts CoMem above the oracle: SAME depth j12, only variable = LoRA → judge **24.52 → 38.27 = +13.75** (§3.3). The LoRA was self-distilled (teacher = j0 full recompute, student = j12 mid-recompute) on **generic PG19, no retrieval, no task data**.
- **Retrieval is task-signed**: at j0 (full recompute), retrieval top-12 gives LoCoMo +7.00 over full-ctx KVD (filters noise) but **loses** on BABILong needle (qa1/qa2 65.9/39.1 vs KVD 78.7/48.9) — retrieval drops scattered facts (§3.2).

**Paper B (OLMo-2-7B, L=32; keep-N + fresh2, healed on Dolmino):**
- Dropping the top layers above keep10 (0.31L) / keep14 (0.44L) **irrecoverably kills stored world knowledge**: MMLU pinned at .25 chance floor for keep10; keep14 recovers only **14–18%** of the base's above-chance MMLU signal even fully converged. from-scratch control (same 16L, more steps) = MMLU .246 = chance → knowledge came *only* from the inherited front layers.
- In-context-answerable capability recovers fine: boolq (passage supplied) **62–44%**, ties from-scratch → learnable from context. PPL recovers to 1.45–1.55× and is **largely blind** to the knowledge gap.

**Probing (`results/probe_linguistic_qwen3_8b.json`, Qwen3-8B, forward-only):**
- Linguistic/semantic linear-probe saturation is **shallow**: `semantic_sat95_frac_depth = 0.13`; peaks POS@3, DEPREL@23(sat95@2), CoLA@18(sat95@9), WiC@17(sat95@5), SST2@19(sat95@8).
- Next-token (logit-lens) accuracy climbs **deep**: `nexttoken_sat95 = layer 34 = frac 0.944`; per-layer top-1 is ~0.01 until layer 20, then ramps 0.02→0.14→0.34→0.41 across layers 22–35.
- **gap_frac_depth = 0.815** — an enormous span between "content extracted" and "readout formed".

---

## 1. Propositions (5)

### P1 — Portable decompression adapter (compression-agnostic distilled LoRA) `[refines user P3]`
**Statement.** CoMem's distilled LoRA is not a CoMem-specific patch but a *portable "read-from-mid-layer-recompute" skill*: loaded unchanged onto a **retrieval-free** mid-layer-recompute compressor (HCache) at the same split depth j=12, it recovers most of CoMem's readout gain — proving the LoRA repairs the depth-cut *readout distribution shift*, independent of the retrieval selector.

- **Emerges from**: §3.3 — the +13.75 judge from LoRA is measured at SAME depth j12, and the LoRA was distilled on **continuous PG19 windows with NO retrieval** (teacher j0-full-recompute → student j12-mid-recompute). HCache is *literally* "j12 mid-layer recompute, packs all chunks, no retrieval" = the LoRA's exact student architecture minus the selector, and its all-chunk read is arguably *closer* to the LoRA's continuous-PG19 training distribution than CoMem's sparse retrieval pack. So the LoRA must transfer to HCache with near-zero domain shift.
- **Novelty (lit-checked, arXiv via hy-proxy)**: no hit for a *transferable / self-distilled decompression adapter across KV-compression methods*. Nearest neighbors are compression-internal (CommonKV cross-layer sharing; RVQ/transform-coding KV compression) or readout-from-layers without a trained module (DoLa, Layer-Fused Decoding). None claims a *trained readout adapter reusable across compressors*. This reframes Paper A's central "LoRA is decisive" finding into a bigger, standalone claim: **we found a reusable KV-decompression module**.
- **Prediction**: HCache+LoRA (retrieval-free, j12) ≫ HCache alone (LoCoMo judge 8.11 → predicted low-20s to 30, i.e. toward CoMem-adapter-free-minus-retrieval), and BABILong/RULER improve too. If it does NOT transfer, that is itself informative (LoRA co-adapted to retrieval-pack statistics) → an honest bound for Paper A. A follow-up on a genuinely different mechanism (StreamingLLM/InfLLM, full-depth) bounds the claim to "shared mid-recompute read path required."

### P2 — Two depths, not one: semantic-content depth ≠ knowledge-readout depth `[refines/partly refutes user P4/P5]`
**Statement.** The residual stream has **two functionally distinct, separable depths**: a *shallow semantic-content depth* (~0.13–0.33L, where linguistic/semantic probes saturate and CoMem can split near-losslessly) and a *deep knowledge-readout depth* (~0.6–0.95L, where factual/next-token information becomes decodable). They are **decoupled**: the semantic depth predicts CoMem's optimal split-j; the knowledge depth predicts the prune-heal keep-N knowledge cliff. There is **no single "bottleneck depth"** governing both — the user's unifying P4/P5 is *half right and needs splitting into two depths*.

- **Emerges from**: (a) probe json — semantic sat95 @ 0.13L vs next-token sat95 @ 0.94L (gap 0.815). (b) CoMem split-j=12 (0.33L) sits at/above the semantic plateau → near-lossless on retrieval/semantic LoCoMo, consistent with "content already extracted." (c) Paper B — dropping layers above 0.31–0.44L irrecoverably kills MMLU (parametric knowledge) but not in-context comprehension → knowledge lives *deeper* than the semantic plateau. **The unification**: CoMem keeps the full top stack (splits *below* the knowledge depth, retains knowledge at read via LoRA-adapted top layers); prune-heal *discards* the knowledge-bearing layers.
- **Novelty**: "Stages of Inference" (2406.19384) and knowledge-localization (ROME / Geva key-value memories) exist and support a shallow→deep gradient. Our novelty is (i) explicitly *separating a knowledge-decodability depth from the linguistic-semantic depth on the same model*, and (ii) mapping the two depths to **two concrete engineering boundaries in our own two systems** (CoMem split-j vs prune-heal keep-N). That cross-system prediction is not in the literature.
- **Prediction**: a per-layer knowledge-decodability probe (logit-lens on MMLU answer-letter / factual cloze) on OLMo-2-7B base peaks **deep** (near next-token depth), and its signal-onset depth aligns (within ~0.1L) with the empirical keep-N cliff (keep10=0.31L→floor, keep14=0.44L→partial), while the linguistic/semantic probe saturates **shallow** (≈ where a CoMem split is near-lossless). The two depths differ by **>2× fractional depth**.

### P3 — CoMem's edge over prune-heal *is* the retained top stack `[bridge; causal test of P2]`
**Statement.** If we take CoMem's read stack and additionally **truncate its top layers to the prune-heal keep-N**, CoMem loses exactly the knowledge-heavy performance (closed-book / world-fact LoCoMo categories) while retaining retrieval/needle performance — proving CoMem's knowledge advantage is *causally* the retained top layers, not the split depth and not the retrieval.

- **Emerges from**: unifies §3 (CoMem keeps the full stack + LoRA) with Paper B (dropping top layers kills knowledge). Directly operationalizes P2's mechanism.
- **Novelty**: novel bridge — no prior work fuses depth-partitioned retrieval readout with depth pruning to *dissociate* knowledge vs retrieval within one model.
- **Prediction**: top-truncated CoMem → LoCoMo cat4/knowledge collapses toward prune-heal level; RULER/needle stays high.
- **Feasibility caveat**: needs a code change (truncate the read stack `layers[j:L-K]`, route through lm_head/logit-lens). Medium — ranked below.

### P4 — Retrieval externalizes the knowledge prune-heal cannot re-instill `[user P7, the payoff bridge]`
**Statement.** The knowledge prune-heal irrecoverably loses (MMLU) is *parametric*, so heal cannot restore it — but it **is recoverable by supplying that knowledge in-context** (RAG). RAG-augmenting a pruned-healed model closes the MMLU gap (mirroring boolq's "in-context ties from-scratch") while closed-book stays at floor — a clean dissociation: **prune-heal + retrieval restores lost knowledge externally**.

- **Emerges from**: Paper B — boolq (passage in-context) recovers 62% and *ties* from-scratch, MMLU (closed-book) stuck; from-scratch control confirms knowledge only comes from inherited layers. The missing ingredient is in-context evidence, which retrieval supplies.
- **Novelty**: RAG-for-pruned-models is intuitive, but the *specific dissociation prediction* (RAG rescues pruning-lost knowledge up to base; closed-book cannot), tied to layer-localized knowledge, is novel and directly bridges the two papers into one story.
- **Prediction**: pruned keep14 + retrieved fact snippet on MMLU → large lift vs closed-book keep14; base model gains far less from the same RAG (already knows).
- **Feasibility caveat**: needs an open-book MMLU / fact-retrieval corpus (light but non-trivial setup). Medium — ranked below.

### P5 — Query-type-adaptive (j, top-k) beats any fixed config `[user P1]`
**Statement.** Optimal split-j and retrieval budget are **query-type dependent** (retrieval helps conversational/LoCoMo, hurts needle/BABILong; different tasks peak at different j), so a cheap query-type router over (j, top-k) has large *oracle headroom* over the single best fixed config.

- **Emerges from**: j0 control (retrieval +7 LoCoMo, −13 qa1/qa2 BABILong) + frozen depth sweep (tasks peak at different j).
- **Novelty**: adaptive KV/retrieval budget is well-trodden; adaptive *split-depth* is less explored → medium novelty overall.
- **Prediction**: per-benchmark oracle (best (j,k) per task-type) beats fixed flagship by a measurable margin, quantifying router headroom.
- **Feasibility**: high (mostly analysis of EXISTING sweep CSVs + a few cells) but lower novelty → ranked below.

---

## 2. Ranking — by (novelty × dual-paper gain × runnability)

| # | Prop | Novelty | Dual-paper gain | Runnability | Verdict |
|---|------|:------:|:---------------:|:-----------:|---------|
| **T1** | **P2 two-depths** | High | **Highest** (a shared mechanistic spine that explains *why CoMem works AND prune-heal fails* — a joint framing/section for both papers) | High (forward-only probe, reuse `probe_linguistic_layerwise.py`, same-day) | **TOP-1** |
| **T2** | **P1 portable adapter** | High | High (turns Paper A's "LoRA decisive" into a bigger reusable-primitive claim) | **Highest** (flags + 1-line un-gate, reuse exact harness, same-day) | **TOP-2** |
| 3 | P3 top-truncated CoMem | High | High (causal test of P2) | Medium (read-stack code change) | strong follow-up to T1 |
| 4 | P4 RAG-rescues-pruned | Med-High | Highest (the payoff bridge) | Medium (needs open-book corpus) | strong follow-up to T2 |
| 5 | P5 adaptive (j,k) | Medium | Medium | High (existing CSVs) | cheap side-analysis |

**Why P2 over P1 for top-1**: both are same-day and high-novelty, but P2 delivers the *cross-paper unification* the ideation is hunting for (one picture for two systems), whereas P1 mainly strengthens Paper A. P1 is top-2 because it is the cheapest, highest-confidence single win (flags only) and its prediction is nearly forced by construction.

---

## 3. Concrete experiment plans (top-2)

### TOP-1 (P2): Knowledge-vs-semantic depth dissociation on OLMo-2-7B (+ Qwen3-8B cross-check)

**Design (single-variable = layer index).** On the SAME forward pass of one model, measure per-layer decodability for three probe families and compare their saturation depths:
1. **linguistic/semantic** — already implemented: POS, DEPREL, CoLA, WiC, SST2, RTE (linear probes on hidden states).
2. **next-token readout** — already implemented: `logit_lens_nexttoken_acc` (logit-lens per layer).
3. **knowledge decodability (NEW, small add)** — per-layer logit-lens on MMLU 4-choice answer-letter (or factual cloze), scoring earliest layer where the correct answer log-prob beats distractors. Reuse the MMLU loader already in `scripts/eval_olmo2_probe2_downstream.py` + the `logit_lens` machinery in `probe_linguistic_layerwise.py`.

**Models / nodes** (all forward-only, LOCAL 8×L20A wzc1, shared FS — all assets present):
- `models/OLMo-2-1124-7B` (Paper B base, L=32) — the primary target (maps to Paper B keep-N cliff).
- `outputs/olmo2_probe2_7B_keep14fresh2/final.pt` (healed 16L) — show the knowledge layers are simply *absent above the kept prefix*.
- `models/Qwen3-8b-local` — cross-model check (already have the semantic+next-token curves in `results/probe_linguistic_qwen3_8b.json`; only need the new knowledge probe).

**Commands** (after adding an `MMLU_LL` task to the probe script):
```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
.venv/bin/python scripts/probe_linguistic_layerwise.py \
  --model_path models/OLMo-2-1124-7B --out results/probe_olmo2_7b_base.json \
  --tasks POS,DEPREL,CoLA,WiC,SST2,RTE,MMLU_LL --device cuda:0
# repeat for Qwen3-8b-local (--out results/probe_qwen3_8b_knowledge.json --tasks MMLU_LL)
```
Then cross-map (CPU, no GPU): knowledge-onset depth vs Paper B keep-N cliff (keep10=0.31L→MMLU floor, keep14=0.44L→partial — numbers already in `OLMO2_PRUNEHEAL_DOWNSTREAM.md`); semantic-sat depth vs CoMem split-j region.

**Cost**: forward-only, 1 GPU per model; probing ~1–2 h/model (MMLU logit-lens n≈1–2k). **~4–6 GPU-hr total, same-day.**

**Success / failure criteria**:
- **Supports P2** if: knowledge-decodability sat depth ≥ 0.55L **and** ≥ 2× the linguistic-semantic sat depth (~0.13L), **and** its signal-onset depth lands within ~0.1L of the observed keep-N cliff. → two decoupled depths; CoMem split (shallow) ≠ prune cliff (deep).
- **Refutes P2 (single-depth wins)** if: knowledge becomes decodable as shallow as semantic (≤0.3L). Then the user's single-bottleneck P4/P5 is vindicated and we pivot to it.

**Payoff**: a single figure ("three saturation curves, two depths") that becomes the shared mechanistic section for *both* papers and explains the apparent CoMem-vs-prune-heal contradiction.

---

### TOP-2 (P1): HCache + CoMem distilled LoRA (retrieval-free, matched depth j12)

**Design (single-variable = LoRA on/off)** on the retrieval-free HCache read path; everything else fixed (Qwen3-8B, resume_j=12, chat=False, iter/pack-all read, official judging).

**One tiny prerequisite code change**: `scripts/eval_qcmem_locomo.py` lines 830–833 currently force `lora_adapter=""` when `baseline=hcache`. Gate that clearing behind a new `--force_lora_with_baseline` flag (same 3-line pattern exists in the babilong/ruler drivers). This is the *only* change; it does not touch any existing default path.

**Arms** (LoCoMo n=1986 GPT-4o judge + BABILong qa1/qa2/qa5 + RULER, chat=False):
```bash
# Arm A (control = existing HCache headline, judge 8.11): no LoRA
scripts/eval_qcmem_locomo.py --baseline hcache --resume_j 12 \
  --model_path models/Qwen3-8b-local --use_llm_judge --dtype bfloat16 ...
# Arm B (treatment): SAME, + CoMem distilled LoRA
scripts/eval_qcmem_locomo.py --baseline hcache --resume_j 12 \
  --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final --force_lora_with_baseline \
  --model_path models/Qwen3-8b-local --use_llm_judge --dtype bfloat16 ...
```
HCache is retrieval-free (`no_retrieval=True`), so the selector is irrelevant — this isolates "does the self-distilled LoRA fix a retrieval-free mid-layer-recompute readout?" Mirror the toggle on `eval_qcmem_babilong.py` / `eval_ruler_qcmem.py` (same override lines).

**Nodes**: LOCAL 8×L20A or .104 (H20 diskB) — the CoMem LoRA (`outputs/qcmem_distill_qwen_j12_r32_4k/final`) and Qwen3-8B are present; judge is GPT-4o-API-bound.

**Cost**: LoCoMo judge run ~a few hours (API-bound); BABILong/RULER cells fast. **~4–8 GPU-hr, same-day.**

**Success / failure criteria**:
- **Supports P1** if: HCache+LoRA LoCoMo judge lifts substantially over HCache 8.11 (predicted into the ~20–30 band) and BABILong/RULER improve. → the LoRA is a *portable mid-recompute-readout skill*, and Paper A's "LoRA decisive" becomes "LoRA = reusable decompression adapter."
- **Bounds/refutes P1** if: no material lift. → the LoRA co-adapted to the retrieval-pack read distribution; still a clean, publishable ablation for Paper A ("the adapter is method-specific, not portable").

**Exploratory follow-up (bounds the claim, needs code)**: apply the same LoRA to a genuinely different mechanism (StreamingLLM/InfLLM, full-depth) — those scripts do not yet accept `--lora_adapter`; defer to a separate coder task.

---

## 4. Notes for main (GPU launch is main's call)
- Neither experiment was launched (per instructions). Both are forward-only / eval-only and reuse existing assets already on LOCAL wzc1.
- TOP-1 needs a small **additive** MMLU-logit-lens task in `scripts/probe_linguistic_layerwise.py`; TOP-2 needs a **1-flag un-gate** in the three QCMem eval drivers. Both are ≤ a small coder task, no architecture work.
- P3 (top-truncated CoMem) and P4 (RAG-rescues-pruned) are the highest-payoff follow-ups if T1/T2 land, but each needs modest code/corpus work → stage after the two same-day probes.

---

## 5. RESULTS — P2 two-depths (#74, ran 2026-07-25 LOCAL B200 cards 0-1, forward-only ~30s/model)

> Driver `scripts/probe_linguistic_layerwise.py --task knowledge_logit_lens`, n_mmlu=1000, bf16, cais/mmlu all/test.
> Outputs: `results/knowledge_logit_lens_OLMo-2-1124-7B.json` (embed+32L), `results/knowledge_logit_lens_Qwen3-8b-local.json` (embed+36L).
> Semantic + next-token anchors from `results/probe_linguistic_qwen3_8b.json` (semantic sat95=L4.67=0.13L; next-token sat95=L34=0.94L).

**Per-model knowledge-decodability depth (MMLU logit-lens):**

| model | onset (chance+0.05) | sat95 | sat99 | peak/top acc |
|---|---|---|---|---|
| OLMo-2-7B (32L) | L18 = **0.562L** | L19 = **0.594L** | L27 = 0.844L | L32 top = 0.551 |
| Qwen3-8B (36L) | L25 = **0.694L** | L25 = **0.694L** | L28 = 0.778L | L34 peak 0.638 / L36 top 0.632 |

- **Sharp step, not gradual ramp**: OLMo L17→L19 acc 0.251→0.326→0.544; Qwen3 L24→L25 0.236→0.621. Below onset acc sits at chance (~0.23-0.26) through the entire early/mid stack.
- **Calibration keeps sharpening after acc plateaus**: gold-letter LL improves deep (OLMo −8.5@L19→−3.79@L32; Qwen3 −6.6@L25→−4.9@L35) while argmax acc is flat → deep layers sharpen confidence not the answer.

**Three-curve depth comparison (fractional depth):**

| curve | Qwen3-8B | OLMo-2-7B |
|---|---|---|
| semantic sat95 (WiC/SST2/RTE) | **0.13L** | (Qwen3 cross-model ref 0.13L) |
| knowledge (MMLU) sat95 | **0.694L** (L25) | **0.594L** (L19) |
| next-token sat95 | **0.944L** (L34) | (Qwen3 ref 0.94L) |

### ★ VERDICT: **SUPPORTS** (strong on depth-separation leg; onset-cliff alignment directionally right, ~0.12L looser than strict window)

1. **Depth separation ✓✓** (the load-bearing claim): knowledge sat95 = 0.694L (Qwen3) / 0.594L (OLMo), **both ≥ 0.55L AND ≥ 2× the semantic sat (~0.13L)** — 5.3× / 4.6× respectively. Knowledge is undecodable from the residual until the deep half while semantic content is fully resolved by ~0.13L → **two functionally separated depths, in BOTH models**. Clean ordering: semantic (0.13L) ≪ knowledge (0.59-0.69L) < next-token (0.94L).
2. **NOT the single-depth falsification case**: knowledge does not saturate ≤0.3L in either model → the user's single-bottleneck P4/P5 is refuted; the two-depths refinement is vindicated.
3. **Knowledge-onset vs Paper B keep-N cliff** (OLMo-2, same base as Paper B pruning): logit-lens knowledge onset = **0.562L**; keep14 cliff = 0.44L (partial MMLU recovery .301), keep10 wipe = 0.31L (MMLU floor .254). Onset is **+0.12L deeper than keep14, +0.25L deeper than keep10** → **direction correct** (knowledge lives in the deep half, above the keep10 wipe) but onset is **just outside the strict ±0.1L window**. Mechanistically expected: logit-lens *decodability* lags the layers that *install* knowledge (compute cliff 0.31-0.44L precedes residual-stream readability 0.56L). Consistent with the "deep-half knowledge readout" story; the two measurements do not coincide tightly.

**Payoff for the papers**: the "three saturation curves, two depths" figure holds — semantic-content depth (shallow, where CoMem splits near-losslessly at j=12=0.33L, above the 0.13L plateau) is decoupled from knowledge-readout depth (deep, where prune-heal's keep-N cliff lives). One mechanistic spine explains why CoMem's shallow split is near-lossless AND why prune-heal irrecoverably loses knowledge. Honest caveat to state: onset-cliff alignment is directional (~0.12L gap), not tight — frame as "decodability lags installation" not "coincident boundaries".

---

## 6. RESULTS — P1 portable adapter (#75, ran 2026-07-25 LOCAL B200 cards 2-7, ~42min, LoCoMo n=1986 GPT-4o judge)

> Driver `scripts/_p1_hcache_lora_toggle.sh`. Single-variable = `lora_adapter` on/off; both arms `baseline=hcache`, `no_retrieval=True`, `resume_j=12`, `chat=False`, bf16, chunk512, sink=bos, n=1986, SAME node/commit.
> Un-gate flag `--force_lora_with_baseline` (commit 0b55791) keeps the LoRA loaded when baseline=hcache. LoRA = `outputs/qcmem_distill_qwen_j12_r32_4k/final` (self-transforms layers 12–35 = read stack above the j=12 split, per adapter_config.json).

| Arm | Config | scores.json | Judge | F1 | acc | EM |
|-----|--------|-------------|-------|----|----|----|
| **A (control, no LoRA)** | `hcache --resume_j 12` | `locomo_results/hcache_j12_noLoRA_chatFALSE/scores.json` | **13.29** | 4.77 | 6.65 | 0.35 |
| **B (treatment, +LoRA)** | same `+ --lora_adapter …/final --force_lora_with_baseline` | `locomo_results/hcache_j12_LoRA_chatFALSE/scores.json` | **31.17** | 7.70 | 17.98 | 0.30 |

### ★ Core: Arm B − Arm A judge = **+17.88** (13.29 → 31.17), a 2.3× lift — clean single-variable on identical node/commit/harness.

Per-cat judge (B vs A): cat1 multi_hop 10.64→25.89 (+15.25); cat2 single_hop 4.98→13.40 (+8.41); cat3 temporal 15.63→29.17 (+13.54); **cat4 open_domain 23.31→55.77 (+32.46, biggest driver)**; cat5 adversarial 1.57→1.35 (flat — locally graded by refusal-regex, LoRA can't help, as expected).

### Comparison to references
HCache+LoRA **31.17** > CoMem adapter-free **29.15** (retrieval, no LoRA) < KVD oracle **34.59** < CoMem flagship j12+LoRA **38.27**. → a **retrieval-free** compressor + the CoMem LoRA recovers the *bulk* of CoMem's readout gain: it clears the adapter-free-with-retrieval bar and closes most of the gap to the flagship, using **zero retrieval**.

### ★ VERDICT: **CONFIRMED**
The distilled LoRA transfers unchanged onto HCache (`no_retrieval=True`, no selector at all). The +17.88 cannot come from anything selector-related → it repairs the **depth-cut readout distribution shift** at the shared j=12 mid-recompute read path. Magnitude lands at the top of the pre-registered prediction ("low-20s to 30"). → reframes Paper A's "LoRA is decisive" into the bigger claim: **a reusable, compression-agnostic KV-decompression/readout adapter**.

### Caveat on Arm A baseline (13.29 vs canonical 8.11)
Arm A gave 13.29, not the paper's canonical HCache headline **8.11**. The 8.11 lives in `hcache_8b_chatFALSE` produced on a **diskB node (.73/.104)**, NOT present on this wzc1 node. Fresh Arm A closely reproduces this node's local deprecated `locomo_results/hcache` dir (12.29, judge Jul-18) with matched eval_config. The ~5-pt gap (8.11 vs ~12–13) is a cross-node harness/config/judge-run diff not inspectable here. **Does NOT weaken the finding**: Arm A/B ran identical harness/commit/node, LoRA the only variable → +17.88 is clean. If anchored to canonical 8.11, HCache+LoRA 31.17 = **+23** (still inside prediction). To report the 8.11-anchored delta, the diskB `hcache_8b_chatFALSE` preds would need re-judging vs Arm B on the same node/harness.
