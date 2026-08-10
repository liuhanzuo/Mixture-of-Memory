# Semantic Handoff Interface — synthesis + what we can build-on / refute (2026-07-27)

> Trigger: user shared a layer-wise-interpretation essay (external "consensus" on
> decoder-only depth roles) and asked what we can **build on** vs **refute**.
> This note maps that framework onto Paper A/B/C and lists concrete, runnable
> experiments where WE have leverage others don't (causal evidence, not just probing).

## 0. The reframe (accept, with one sharpening)

External framing (accept): decoder-only LLMs are NOT "front-j = encoder, back = NTP".
They emergently develop a soft **semantic handoff interface**:
`token→entity (front) → reusable semantic/task state (front-mid) → query-conditioned
composition (mid-back) → verbalization/calibration (last few)`.

The essay's single best contribution is the **three-depth decomposition**:
- **j_content**  — depth where the info is first *present* (probe-recoverable).
- **j_native**   — deepest split where the *original* upper stack still reads h_j zero-shot.
- **j_adapt**    — deepest split where a *small* new decoder / LoRA / OPD recovers full quality.
- Conjecture (essay, "需实验确定"): `j_content < j_native ≤ j_adapt`.

★ Our position: we already have all three instruments AND a causal control Paper B —
so we can *measure* this ordering across scale instead of asserting it. That's the paper.

## 1. What our EXISTING data already settles (build-on, low/zero new GPU)

| Essay claim (stated as hypothesis / consensus) | Our evidence | Status |
|---|---|---|
| content-depth ≠ readable-depth | Paper A double-j table: readout-safe j (native) vs content-j ~0.45L (probe peak), 6 sizes | **CONFIRMED, we own it** |
| mid-layer state is a better general representation than final layer | zero-shot mid-j CoMem works; final-layer NTP-specialized | consistent |
| "knowledge lives in the reusable front/mid state" | **Paper B causal**: from_scratch MMLU=.246 (chance) vs healed .30–.31; keep8 cliff (MMLU chance even after 4.4× heal) → knowledge is inherited in the trunk, NOT re-learnable by the tail | **CAUSAL — stronger than the probing the essay cites** |
| back layers are NOT just NTP; they do query-conditioned composition | **CoMem block-diagonal ablation**: killing upper cross-chunk attention hurts multi-key/multi-fact | **REFUTES "back=NTP", we own it** |
| gap between native and adaptable shrinks with scale | our T27 note: content-j adapter value monotone-decreasing with scale ("0.6B huge gain, 32B ~none") | **CONFIRMED trend, publishable as a scale law** |

## 2. Where we can REFUTE / SHARPEN the essay

1. **Ordering can INVERT for exact-copy tasks.** Essay assumes monotone
   `j_content < j_native`. Our T27 data: content-j (deep) adapter *hurts* small-model
   literal exact-match (LongEval 0.6B 37→1, RULER-vt 0.6B 81→1). → for copy/verbatim
   tasks the *readable* signal is SHALLOWER than the semantic-content peak, i.e.
   `j_native < j_content` — opposite of QA. The handoff depth is **task-dependent**,
   not a single scalar. This is a genuine correction, backed by data.

2. **"j_adapt is what random-2-layer measures, not j_native"** (essay §9.3) — agreed,
   and Paper C's freeze-graft is *literally* the j_adapt instrument. So Paper C should be
   reframed FROM "shallow cap beats LoRA" TO "**measuring j_adapt and its predictability**".

## 3. Reframed measurement proposition

This superseded the historical P-C2 framing now archived at
`proposal/archive/paperC-v1-frozen-cap/scoping/SCOPING_AND_POSTMORTEM.md`.

> **Claim**: The depth at which a frozen pretrained trunk becomes *adaptably decodable*
> (j_adapt) is (a) deeper than the natively-readable depth j_native, (b) predictable from
> the model's intrinsic semantic/knowledge depth structure, and (c) the j_adapt−j_native
> gap shrinks with model scale. A K-layer fresh cap (+ frozen trunk) recovers full-FT
> quality up to j_adapt, beating same-budget LoRA under distribution shift.

This unifies:
- CoMem = exploits **cacheability** of h_j (Paper A)
- prune-heal = shows **knowledge is in the trunk, tail is replaceable** (Paper B)
- freeze-graft cap = measures **j_adapt / shallow decodability** (Paper C)
- (external: OPD = behavioral adaptability, DoLa = layerwise maturation) — related work.

## 4. THE killer experiment (essay §10, scoped to our infra) — "handoff phase diagram"

2D sweep: **split depth j × readout method/decoder-depth r**, measured on
**task families that separate representation from composition**.

- j ∈ {shallow, content-peak, deep} per model (reuse our readout-safe/content-j points)
- readout r ∈ {native suffix, logit-lens, tuned-lens(1 affine layer), 2-layer random+distill, LoRA}
- task axis (the novel cut): **retrieval (NIAH single) vs composition (RULER-vt multi-hop, BABILong qa2/qa3) vs format-control**
- metrics: teacher-logit KL, token-agreement, task score.

**Predicted phase diagram** (falsifiable):
- retrieval recovers at shallow r (info present + readable) 
- composition needs deeper r at the SAME j → proves upper layers do composition, not NTP
- content-good / native-bad region → the "handoff exists but original stack can't read it" cell
- format-control recovers with tiny r (last-layer verbalizer)

We already have: native-j (zero-shot per size), adapter-j (LoRA), block-diag ablation,
knowledge probes. **Missing pieces to run**: tuned-lens per-layer affine, and the
K-layer-random-decoder × task-family recovery grid. ~modest GPU (forward + small train).

## 5. Runnable plan (queued behind current jobs; Paper C node frees after LoCoMo)

- **N1 (zero-GPU, now-ish)**: assemble j_content (probes) + j_native (zero-shot-j) +
  j_adapt (existing 8B adapter) into the three-depth table across 6 sizes from
  data ALREADY on disk → first empirical `j_content < j_native ≤ j_adapt` (+ the
  copy-task inversion). This is a research-note/figure, no training.
- **N2 (Paper C P-C1, queued #92)**: freeze-graft cap depth sweep = the j_adapt measurement.
- **N3 (phase diagram, §4)**: split-j × decoder-depth-r × {retrieval, composition, format}
  on Qwen3-8B first. Needs 1 node ~1 day. The unifying mechanism figure for all 3 papers.

## 6. One-line thesis to defend
> Decoder-only LLMs expose a **locatable, cacheable, distillable, re-decodable semantic
> handoff interface** that partially decouples reusable content representation from
> query-conditioned generation — and the interface depth is task-dependent and
> scale-dependent, which we measure causally (prune-heal) and constructively (CoMem cache,
> freeze-graft cap), not just by probing.
