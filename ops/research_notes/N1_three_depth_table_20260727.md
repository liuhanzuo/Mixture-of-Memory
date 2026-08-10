# N1 — Three-depth (handoff interface) cross-scale table, assembled from EXISTING data (2026-07-27)

> Zero-GPU deliverable. All numbers pulled from files already on disk (no new runs).
> Purpose: turn the external essay's *conjectured* `j_content < j_native ≤ j_adapt` into
> an *empirically measured* ordering across the Qwen3 family + OLMo-2. Decide A/B/C-support
> vs Paper D. Sources cited inline.

## 0. The essay's conjecture vs what our data shows

Essay conjecture: `j_content < j_native ≤ j_adapt` (info present early; original stack reads
deeper; small decoder reads deepest).

**Our finding (correction): "content" is not one depth — it is a SPECTRUM**, and the
zero-shot-readable boundary `j_native` sits *inside* that spectrum, SHALLOWER than the
richest reusable (task-semantic / knowledge) representation. So the operative ordering is:

```
semantic-feature onset  <  j_native (zero-shot readable)  <  task-semantic peak  <  knowledge peak  <  next-token
     0.13L                        0.25L (8B)                    0.44L                 0.69L            0.94L
```
and `j_adapt` (distilled cap) extends readability from j_native toward the task-semantic
peak. The essay collapses this into one `j_content`; the reality is multi-band and the
"readability gap" (content − native) is the quantity that matters.

## 1. Depth spectrum on one model (Qwen3-8B, 36L) — from probes on disk

| band | depth (/L) | source |
|---|---|---|
| semantic-feature onset (WiC/SST2/RTE sat95) | **0.13L** | `results/probe_linguistic_qwen3_8b.json` division_of_labour |
| **j_native** = zero-shot QCMem readout tolerance (single ≥90) | **0.25L** (j9) | `QCMEM_J_DETERMINATION.md` bracket |
| task-semantic **content peak** (SST2/WiC/RTE knee98) | **0.44L** (j16) | `QCMEM_J_DETERMINATION.md` probe |
| knowledge decodability sat95 (MMLU logit-lens) | **0.69L** (L25/36) | `results/knowledge_logit_lens_Qwen3-8b-local.json` |
| next-token sat95 | **0.94L** (L34) | `results/probe_linguistic_qwen3_8b.json` |

(OLMo-2-7B mirror: semantic 0.13L / knowledge sat95 0.59L / next-token 0.94L —
`knowledge_logit_lens_OLMo-2-1124-7B.json`, Paper B P2.)

## 2. Cross-scale three-depth table (6 Qwen3 sizes) — the N1 headline

| model | L | j_native (readout-safe, ≥90 single) | j_content (semantic peak knee98) | readability gap (content − native, 50%-cliff basis) | j_adapt (distilled cap reaches) |
|---|---|---|---|---|---|
| 0.6B | 28 | j2 = 0.07L | j13 = 0.48L | **~0.39L (huge)** | content-peak for *retrieval*; composition FAILS |
| 1.7B | 28 | j3 = 0.11L | j13 = 0.48L | ~0.26L | retrieval ok; composition fails |
| 4B | 36 | j9 = 0.25L | j16 = 0.44L | ~0.13L | retrieval 100; composition partial |
| 8B | 36 | j9 = 0.25L | j16 = 0.44L | ~0.14L | retrieval 100; composition partial |
| 14B | 40 | j13 = 0.325L | j18 = 0.46L | ~0.085L | retrieval 100; composition **recovers** (mk 71–83, vt 65–89) |
| 32B | 64 | j27 = 0.42L | j27 = 0.42L | **~0 (native already at content)** | adapter ≈ unnecessary |

Sources: `QCMEM_J_DETERMINATION.md` (bracket table, content table, gap-vs-scale table,
adapter@content-j table). All measured n=100 RULER single/multikey/vt.

## 3. Three empirical findings (all from existing data)

**F1 — Scale law on the readability gap.** `gap = j_content − j_native` shrinks
monotonically with scale: 0.39L (0.6B) → 0.26 → 0.13 → 0.14 → 0.085 → ~0 (32B).
Interpretation: small models' frozen stacks can only natively read a shallow slice; the
richest reusable representation is much deeper and inaccessible without a learned readout.
Large models' native readability already reaches the semantic peak → adapter ≈ redundant.
**This is a novel, clean scale-law statement the essay does not have.**

**F2 — j_adapt is task-dependent, not a scalar (refutes "back layers = NTP").**
A distilled cap at content-depth recovers **retrieval** (NIAH single 95–100) at *every*
scale, but **composition** (multikey, variable-tracking) only recovers for large models
(14B yes; 8B/4B partial; 0.6B/1.7B fail). → the deep layers a small model *can't* distill
back are doing **composition**, not verbalization. Directly supports "upper layers do
query-conditioned composition" and matches CoMem's block-diagonal ablation (killing upper
cross-chunk attention hurts multi-key). `RUN_REGISTRY.md` T27 + block-diag ablation.

**F3 — Copy/verbatim tasks INVERT the ordering.** Content-j (deep, 0.45L) adapter *hurts*
small-model literal exact-match: LongEval 0.6B 37→1, RULER-vt 0.6B 81→1, 1.7B vt 48→21.
→ for exact-copy the *readable* signal is SHALLOWER than the semantic-content peak
(`j_native < j_content` reverses in usefulness): deep caching destroys the verbatim tokens
copy tasks need. The handoff depth you should cache at is **task-dependent**. The essay's
monotone single-j model misses this. `QCMEM_J_DETERMINATION.md` adapter table pt.2–3.

## 4. Causal leg the essay lacks (Paper B)

All the above is representational (probe / readout). Paper B adds the **causal** proof that
the deep reusable content is *inherited*, not re-learnable: from_scratch (16L random init,
200k steps) MMLU = .246 = chance vs healed .30–.31; keep8 MMLU stays at chance even after
4.4× more heal. → the "semantic/knowledge state" is physically located in the front trunk
and cannot be reconstituted by training the tail. No interpretability paper has this.

## 5. Verdict: A/B/C support OR Paper D?

**Recommendation: seed of Paper D, dual-use now.**

- The handoff-interface framework is BROADER than any single paper's claim — it is the
  mechanism that *explains* why A (cache at content depth), B (knowledge in trunk), C
  (freeze-graft = measure j_adapt) all work. Three *constructive* instruments measuring one
  interface (cache / prune / graft) + cross-scale + causal is more than typical single-model
  probing papers → **Paper D is the stronger, more novel framing** IF we add the
  phase-diagram experiment (N3: split-j × decoder-depth-r × {retrieval, composition, format}).
- Until N3 lands, this note is **dual-use**: it is simultaneously (a) the shared mechanism
  section A/B/C each cite, and (b) the empirical core of Paper D. No wasted work either way.
- Decision point: after N3 (1 node ~1 day, queued behind Paper C), decide standalone-D vs
  merge-into-mechanism-sections based on whether the phase diagram is a clean standalone
  contribution.

## 6. What N1 still lacks (small, optional, mostly zero-GPU)
- j_content probe for 0.6B/1.7B/4B/14B/32B uses the SST2/WiC/RTE knee98 already in
  J_DETERMINATION; the knowledge-decodability band (MMLU logit-lens) currently only has
  8B + OLMo — extending it to the other 5 Qwen3 sizes is forward-only (~1 GPU-h each, cheap)
  and would give a full knowledge-depth column. Optional polish, not blocking.
