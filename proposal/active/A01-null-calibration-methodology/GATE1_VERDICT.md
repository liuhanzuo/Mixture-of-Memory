---
gate: A01 gate-1 (third model family MC interface) -- INTACT-ONLY LEG
date: 2026-08-09
node: .21 (8x L20A 183GB, wzc1)
verdict: SUPERSEDED -- this file measured INTACT bases, which cannot test A01's kill clause
superseded_by: GATE1_DAMAGED_VERDICT.md (6/6 damaged non-OLMo arms below floor -> A01_GENERAL_CLAIM_CONFIRMED)
status: the NUMBERS below are valid and still cited; the VERDICT section (§1) is RETRACTED
---

> **★ RETRACTION NOTICE (2026-08-09, same day).** §1 of this file concluded
> `KILL_CONDITION_CLAUSE_2_TRIGGERED` and "narrow A01 to OLMo-only". **That conclusion is
> withdrawn.** A01's kill clause is about **structurally damaged** models; this run measured
> **intact** bases, which were never expected to show the pathology and therefore cannot
> trigger or clear the clause. The damaged-arm run (`GATE1_DAMAGED_VERDICT.md`) found
> **6/6 non-OLMo damaged arms at/below their own best-constant floor**, i.e. the general
> claim is **CONFIRMED**, not narrowed.
>
> **What survives from this file:** the intact-base measurements themselves (§2 onward),
> which are the source of a *different*, second-order finding — on three of four **healthy**
> strong models the letter interface is **+13 to +23pp better** than the label-free content
> interface. So "content is the fair interface" is a statement about damaged models, not a
> general property of MC interfaces.

# A01 gate-1 — third-model-family verdict

## 1. Verdict in one line ⚠️ RETRACTED — see notice above

**0 of 3 non-OLMo families replicate the MMLU letter-interface failure.** A01's
Kill condition clause 2 ("第三家族和第二 benchmark 均不复现 interface failure") is
**triggered on the third-family half**. A01's claim must narrow from *"the letter
MC interface is an unreliable instrument"* to *"the letter interface degenerates
in structurally damaged OLMo-2 arms."* The **protocol** contribution
(construct-appropriate nulls before any cross-arm comparison) is untouched and
in fact is what made this narrowing detectable.

*(Paragraph above retained verbatim as the record of what was claimed and withdrawn.
Do not cite it. The reason it was wrong: it tested intact, not damaged, models.)*

## 2. The measurement

Full MMLU (n=14,042, 0 nan, 0 truncated), `chat_template=False`, `--add_bos 0`,
no system prompt, no few-shot, fp32 weights / bf16-autocast forward, 8 shards
across 8 GPUs with 8/8-shard + exact-n assertion before every merge.

**The only thing that differs from the archived OLMo-2 runs is the model class.**
`--any_family` routes to `load_base_model_any_family`, which swaps
`Olmo2ForCausalLM` for `AutoModelForCausalLM`. Tokenisation (already
`AutoTokenizer`), prompt construction, truncation, sharding, per-option scoring,
length normalisation, tie detection and aggregation are the identical code path.
The OLMo path was not edited, so every archived OLMo number still reproduces.

Null: best-constant letter recomputed from the MMLU gold distribution
(A 3222 / B 3462 / C 3582 / **D 3776**) → **always-D = 0.2689**. Never `.25`.

| arm | n | letter | resid vs floor | modal% | **tie%** | content_norm | resid | letter−content | exact McNemar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama-2-7B** | 14042 | 0.4100 | **+14.11 pp** | 41.4% | **15.79%** | 0.4135 | +14.46 | −0.35 pp | 5.1e-01 |
| **Llama-3-8B** | 14042 | 0.6220 | **+35.31 pp** | 29.2% | 4.43% | 0.4624 | +19.35 | **+15.96 pp** | 1.2e-196 |
| **Qwen3-8B-Base** | 14042 | 0.7464 | **+47.75 pp** | 30.6% | 4.90% | 0.5173 | +24.84 | **+22.91 pp** | ~0 |
| OLMo-2-7B base | 14042 | 0.6054 | +33.65 pp | 28.7% | **0.13%** | 0.4706 | +20.17 | +13.48 pp | 3.8e-143 |
| OLMo-2-7B keep14@200k | 14042 | 0.3184 | +4.95 pp | 39.5% | **24.36%** | 0.3832 | +11.43 | −6.48 pp | 6.2e-33 |
| OLMo-2-7B keep8@121k | 14042 | 0.2550 | **−1.39 pp** | 48.8% | **30.64%** | 0.3423 | +7.34 | −8.72 pp | 1.9e-56 |

Provenance: `evidence/a01_gate1_third_family.json`; analysis
`code/a01_gate1_verdict.py`; driver `scripts/_a01_gate1_driver_21.sh`;
raw per-item `olmo2_mmlu_content_results/gate1_{llama2_7b,llama3_8b,qwen3_8b_base}/`.

## 3. ★ The finding that matters more than the kill

**Llama-2-7B carries 15.79% exact ties among its four letter-option scores and is
nonetheless completely healthy**: +14.11 pp above its own best-constant floor,
and statistically indistinguishable from the content interface (letter − content
= −0.35 pp, exact McNemar p = 0.51). Meanwhile OLMo-2 **base** has a 0.13% tie
rate and is also healthy.

So the tie rate does **not** order the pathology across families:

```
tie rate    0.13%   4.43%   4.90%   15.79%   24.36%   30.64%
arm         OLMo    L3-8B   Q3-8B   L2-7B    keep14   keep8
healthy?    yes     yes     yes     YES      no       NO
```

**This falsifies "exact ties are the mechanism of interface failure" as a
family-general causal claim.** A high tie rate is compatible with a perfectly
usable letter interface. Within OLMo-2 the tie rate does track damage
monotonically (0.13 → 24.36 → 30.64%), so ties remain a *correlate of damage in
that family* — but they are not sufficient for failure.

**Direct consequence for gate-3** (running on `.73` right now, fp32-vs-bf16):
gate-3 was framed as `H_artifact` (ties are a bf16 representation artifact) vs
`H_real` (the damaged model genuinely puts equal mass on the four letters). This
result adds a third possibility that gate-3 must now distinguish: ties may be
*real and yet harmless*, in which case what breaks the damaged OLMo arms is not
the tie-breaking at all but the collapse of the letter-position → answer mapping
(consistent with keep8's 48.8% modal share). Gate-3's fp32 arm should therefore
be read against Llama-2 as a positive control for "ties without failure."

## 4. Second-order observation: the letter−content sign flip

The sign of (letter − content) separates healthy from damaged arms perfectly on
these six arms:

| | letter − content | arms |
|---|---:|---|
| letter **better** | +13 to +23 pp | OLMo-2 base, Llama-3-8B, Qwen3-8B-Base |
| indistinguishable | −0.35 pp (p=.51) | Llama-2-7B |
| content **better** | −6 to −9 pp | OLMo-2 keep14, keep8 |

This inverts a narrative A01 leans on. On three of four *healthy* models the
letter interface is **substantially better** than content — so "content is the
valid interface, letter is the broken one" is not a general statement about MC
interfaces; it is a statement about damaged models. For an intact strong model,
the label-free content interface actually *loses* 16–23 pp of measurable
capability. That is a real methodological point worth reporting on its own: the
choice of MC interface is not a free axis, and the correct choice depends on
whether the model under test can still use letter positions at all.

## 5. What survives, what does not

**Survives:**
1. Report every construct against a *construct-appropriate* null (best-constant,
   not chance) before comparing arms — this run is itself the demonstration: at
   `.25` chance line, keep8's 0.2550 looks like "barely above chance"; against
   0.2689 it is **below floor**.
2. The self-falsification narrative (A01 using its own protocol to retract its
   own headline).
3. The letter-interface degeneration result **scoped to structurally damaged
   OLMo-2**, now with a clean cross-family contrast that makes the scoping
   evidence-based rather than a caveat.

**Does not survive:**
4. Any claim that the letter MC interface is *generally* an unreliable
   instrument. Three families, spanning three tokenizers (32k SP, 128k BPE,
   Qwen BPE) and three pretraining corpora, all show it healthy and mostly
   *better* than content.
5. "Exact ties cause interface failure" as a family-general mechanism (see §3).

## 6. Remaining gate-1/gate-2 status

* gate-1 (third family): **DONE, kill clause 2 triggered on this half.**
* gate-2 (non-MMLU MC benchmark): still open, but it is a **CPU-only recompute** —
  wzc1 already holds per-item `option_scores` for arc_challenge / arc_easy / piqa /
  openbookqa / winogrande across 20+ arms under `olmo2_downstream_results/*/`.
  No GPU needed. If gate-2 also fails to reproduce the pathology, the Kill
  condition's clause 2 is fully satisfied and A01 must be rewritten as a
  protocol paper with an OLMo-2 case study, not a general interface-validity paper.

## 7. Cost

5 minutes wall on 8 L20A (three 7-8B models x 14,042 items). One line of new
model-loading code plus a driver. The `datasets` package on `.21` had to be
upgraded 2.21.0 → 5.0.0 to read the parquet cache written by 5.0.0 on the other
nodes; `transformers` left at 4.57.6.
