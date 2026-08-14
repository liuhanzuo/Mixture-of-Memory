# B05 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-14. 0 GPU. Adjudication only — this file does not run anything.**

This closes the blocker that `proposal/ready_queue.py` actually trips on
(`RELATED_WORK.md absent`, `novelty_checked: false`), and answers the six collision
families named in `proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:95`
(rating: **不足但自限** — insufficient but self-limiting; prescribed framing:
`j_content/j_native/j_adapt` phase diagram, **not** a universal semantic cut).

## Standing rule this adjudication obeys

Per `memory/prior-work-differentiate-dont-abandon` (user 2026-08-07: 「别因1-2篇类似
工作就放弃方向」): the bar for preemption is **完全相同 / 抄袭**, not overlap. Work
within 2–3 months is **concurrent** and does not preempt. A direction dies from its own
kill gate (`PHASE_SEPARATION_PREREG.md` §3.2), never from a literature hit.

## Verification discipline

Venues verified per family, not by one API:
`memory/venue-verify-must-use-openreview-2026` (OpenReview `venueid` for ICLR/NeurIPS/ICML)
and `memory/venue-verify-acl-family-needs-anthology` (Anthology + DBLP for ACL/Findings).
**Semantic Scholar returned HTTP 429 for most of this session**, so anything below marked
`arXiv-only` means *I could not verify a peer-reviewed venue from this node*, not that one
does not exist. That is recorded as an honest gap, not smoothed over.

---

## 1. The six collision families

### 1.1 Logit lens / tuned lens

- *Eliciting Latent Predictions from Transformers with the Tuned Lens* — DBLP: **CoRR 2023**
  (`journals/corr/abs-2303-08112`), `arXiv:2303.08112`. **No peer-reviewed venue found via
  DBLP.** Downstream: *Do Transformer Interpretability Methods Transfer to RNNs?* —
  **AAAI 2025** (`conf/aaai/PauloMB25`, DOI `10.1609/aaai.v39i26.34969`).
- *TriLens: Per-Layer Logit-Lens Entropy for White-Box Hallucination Detection* — DBLP:
  **CoRR 2026**, arXiv-only.

**Overlap:** both read intermediate layers with a trained affine map. **Not identity, and
this repo has already paid to learn why:** `proposal/archive/paperC-v1-frozen-cap/POSTMORTEM.md`
§4.1 measured four forward-probe families on one model (OLMo-2-7B, L=32) whose answers span
the *entire* depth domain — linguistic edge-probe sat95 0.000–0.156L, knowledge logit-lens
onset 0.562L, next-token logit-lens sat95 1.000L, CKA-drift 50 % mass 0.938L — and found the
knowledge logit-lens regresses on the measured recovery curve at only **r = +0.7347**, and is
systematically too shallow. **Lens methods answer "what is linearly decodable here"; B05's
primary axis is "does the frozen upper stack still function when handed h_j", which the lens
provably does not predict.** B05's `必须避免` item 3 forbids using a forward probe as the
predictor. This is the strongest available differentiation and it is *our own measurement*,
not an argument.

### 1.2 Layerwise emergence

- *Phase Diagram of Vision Large Language Models Inference: A Perspective from Interaction
  across Image and Instruction* — `arXiv:2411.00646` (2024-11), arXiv-only from this node.

**Overlap:** the phrase "phase diagram" over layers. **Not identity:** it is a
**vision–language interaction** analysis, descriptive of attention flow, with no split-and-resume
execution and no readout-capacity axis. B05's cells are causal (the upper stack is actually run
from a cached h_j) rather than observational.

### 1.3 Causal tracing / activation patching

Family acknowledged. **Differentiation is structural, not rhetorical:** causal tracing patches a
*single* activation and measures an effect on one output. B05 hands over the **entire prefix
state at depth j** and measures **task accuracy of the intact upper stack** at n=100 per cell
with paired `input_ids_sha256` joins. This repo owns that leg
(`scripts/eval_ruler_qcmem.py --resume_j`), which is why the grid is 3 GPU-h rather than a new
codebase. Note task **#128 "Paper B P2.2 activation patching harness"** is `in_progress` in this
repo — a B05 claim must not be phrased so it collides with our own patching work.

### 1.4 Early exit / adaptive depth

- *Robust and Efficient Early Exit for Large Language Models: Mitigating KV Cache Loss…* —
  DBLP **ISNN 2025**.
- *You Need Multiple Exiting: Dynamic Early Exiting…* — DBLP **CVPR 2023** (+ CoRR 2022).
- *A training-inference consistent framework for early exiting…* — DBLP
  **Int. J. Mach. Learn. Cybern. 2026**.
- *Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates…* —
  `arXiv:2607.20519` (2026-07), arXiv-only.

**Overlap:** both cut the stack at depth j. **Direction is opposite, and that is the whole
difference:** early exit **skips the upper layers** and decodes from h_j (goal: latency).
B05 **keeps every upper layer and runs all of them**; j only marks where the *cache boundary*
is. In B05 the upper stack is the consumer, in early exit it is the thing removed. No early-exit
paper measures "does the retained upper stack read a cached h_j".

### 1.5 Split computing / split inference

- *Salted Inference…* — DBLP **HotMobile 2024** (+ CoRR 2023).
- *Accelerating CNN Inference in Split Computing* — DBLP **ICOIN 2024**.
- *SplitTracr* — DBLP **ICPE 2025**.
- *Privacy-Aware Split Inference with Speculative Decoding for LLMs* — `arXiv:2602.16760`
  (2026-02), arXiv-only.

**Overlap:** an intermediate tensor crosses a boundary. **Not identity:** split computing's
objective is **bandwidth/privacy under a device–server partition**; the model is unchanged and
the question "at which depth is the state still readable *by a frozen stack, per task family*"
is never asked. B05 has no device, no network, no privacy claim.

### 1.6 Model stitching / readout adapter

- *Revisiting Model Stitching to Compare Neural Representations* — DBLP
  **NeurIPS 2021** (`conf/nips/...`), peer-reviewed. ✅
- *Model Stitching: Looking For Functional Similarity Between Representations* — DBLP
  **CoRR 2023**, arXiv-only.
- *MoSECroT: Model Stitching with Static Word Embeddings for Crosslingual Zero-shot Transfer* —
  `arXiv:2401.04821`, arXiv-only from this node.

**This is the closest family and must be discussed head-on.** Stitching trains a map between
**two different networks** to test *representational similarity*. B05 stitches a model to
**itself across a cache boundary** and varies **readout capacity {none, LoRA}** at fixed depth,
asking whether depth and readout capacity separate into **named phases per task family**. The
`A6` capacity-matched control already on disk (r40@j12 vs r32@j6, **exactly** 72,744,960 params
both) shows the effect survives exact capacity matching — a control stitching literature does not
run because it is not testing depth.

---

## 2. The one genuinely close hit, and what it means

**SeDeM — *Selective Decompression of Hidden-State Memories for Long-Context Question
Answering***, `arXiv:2608.00311` (2026-08), arXiv-only. Abstract fetched and read this session.
Already flagged in `memory/paperc-pc1-scooped-eval-invalid` as a collision for A02.

SeDeM: extract hidden states from **a chosen intermediate layer**, compress to memory blocks,
query-conditioned selector picks blocks, decompressor expands them into states compatible with
**an intermediate decoder layer**. 1B/3B backbones, four long-context QA benchmarks.

**Assessment: overlapping, concurrent, NOT preempting.**

1. **Concurrent.** 2026-08 vs this prereg 2026-08-14 — same month. Per the standing rule this
   cannot constitute preemption.
2. **Different object.** SeDeM proposes a *method* and optimises QA score. B05 is a
   *measurement*: it holds the pipeline fixed and varies **readout capacity** to test whether
   depths partition into phases. SeDeM **fixes** its layer choice; B05's dependent variable
   *is* the layer choice.
3. **The axis SeDeM does not have.** SeDeM always uses a trained compressor+decompressor.
   B05's decisive column is the **zero-parameter native-suffix readout at j > 0** — no trained
   module at all. Without that arm there is no `j_native`, hence no Phase I/II boundary, hence
   no phase diagram. This is exactly what B05's 16 new cells buy.
4. **Their benchmark family is the one we already demoted.** SeDeM evaluates on long-context
   **QA**; `A02_DEPTH_VS_RETRIEVAL_VERDICT.md` measured recall@12 = 22.9–63.2 % on our QA cells
   and showed **54.9–78.6 %** of the movement there is *retrieval*, not read. B05 restricts its
   primary read-out to **retrieval-CLOSED** RULER cells (recall@12 = 99–100 %). That is a
   correction to how this family is measured, i.e. a **follow-up fixing a defect**, which is what
   the standing rule asks for instead of abandonment.

**Required of B05's write-up:** cite SeDeM as concurrent, state that B05 measures a phase
structure rather than proposing a compression method, and never claim priority on
intermediate-layer hidden-state memory.

---

## 3. Verdict

```
verdict: hold_in_backlog -- novelty gate CLEARED for GPU, PROMOTION still requires the gate to pass
```

- **No candidate is 完全相同/抄袭.** Every family above differs on at least one load-bearing
  axis (direction of the cut, trained-vs-zero-parameter readout, method-vs-measurement, or
  retrieval-closed-vs-confounded read-out). `already_dead_should_archive` is **not** warranted.
- **Narrowing is mandatory and is evidence-forced**, per the audit's "自限" rating and our own
  prior measurements: primary read-out is the **retrieval-closed RULER slice only**; semantic QA
  is contrast-only; format is excluded (scorer confound); parametric knowledge is a descriptive
  band and never a predictor.
- **Not promotable yet.** Promotion needs `PHASE_SEPARATION_PREREG.md` §3.2 to run and not fire.

## 4. Honest gaps in this adjudication

1. **S2 rate-limited (HTTP 429) for most queries**, so several entries are `arXiv-only` =
   *venue unverified from this node*, not *no venue*. Before any B05 submission, re-run venue
   verification per family (OpenReview `venueid` for ICLR/NeurIPS/ICML; Anthology+DBLP for ACL).
2. **DBLP returned HTTP 500** on two queries (`hidden state cache long context`,
   `tuned lens eliciting latent predictions`); those were covered via arXiv instead.
3. **No `.bib` entries are emitted here.** Per `memory/tcodex-exec-no-dash-c-flag`, entries must
   not enter the bibliography until venue-verified by family.
4. **Zero cross-disk verification.** `/apdcephfs_zwfy6` is not mounted on LOCAL, so every zwfy6
   path cited in the prereg is recorded-only from here and must be `ls`-confirmed on
   `.73`/`.82`/`.104` before dispatch (`memory/two-disk-rule-applies-to-main-too`).
