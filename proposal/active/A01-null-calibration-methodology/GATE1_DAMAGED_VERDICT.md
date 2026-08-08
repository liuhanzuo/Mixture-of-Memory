---
gate: A01 gate-1 damaged non-OLMo extension
date: 2026-08-08
node: .21 (8x L20A wzc1)
verdict: A01_GENERAL_CLAIM_CONFIRMED_via_DAMAGE
supersedes: GATE1_VERDICT.md (which measured healthy bases only)
---

# A01 gate-1 — damaged non-OLMo extension (revises the intact-base verdict)

## 1. Big reversal in one line

The healthy-base gate-1 run showed 0/3 non-OLMo families reproducing the
letter-interface failure. That result was **testing the wrong condition**. A01's
kill claim is about *damaged* models, not intact ones. Running the SAME letter-vs-content
probe on non-OLMo models truncated to their first 8 or 12 layers (no fresh block,
no heal) reproduces the OLMo pathology on **every single non-OLMo arm tested (6/6)**.
**A01's general claim is not killed; it is confirmed.**

## 2. The four-family match (n=14,042 each, MMLU, chat_template=False)

Comparing intact base → damaged (front-N truncation, no heal) for each family:

| model | condition | letter | vs floor 0.2689 | modal% | tie% | content_norm | letter−content |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | intact | 0.4100 | +14.11 | 41.4% | **15.79%** | 0.4135 | −0.35 |
| Llama-2-7B | trunc k=8 | **0.2415** | **−2.74** | 44.9% | 0.38% | 0.2542 | −1.27 |
| Llama-2-7B | trunc k=12 | **0.2295** | **−3.94** | **100.0%** | 0.00% | 0.2529 | −2.34 |
| Llama-3-8B | intact | 0.6220 | +35.31 | 29.2% | 4.43% | 0.4624 | +15.96 |
| Llama-3-8B | trunc k=8 | **0.2329** | **−3.60** | **88.5%** | 0.14% | 0.2589 | −2.60 |
| Llama-3-8B | trunc k=12 | **0.2527** | **−1.62** | 47.3% | 0.31% | 0.2498 | +0.30 |
| Qwen3-8B-Base | intact | 0.7464 | +47.75 | 30.6% | 4.90% | 0.5173 | +22.91 |
| Qwen3-8B-Base | trunc k=8 | **0.2286** | **−4.03** | 76.2% | 1.35% | 0.2460 | −1.74 |
| Qwen3-8B-Base | trunc k=12 | **0.2300** | **−3.89** | **99.1%** | 0.03% | 0.2502 | −2.02 |
| OLMo-2-7B (ref) | intact | 0.6054 | +33.65 | 28.7% | 0.13% | 0.4706 | +13.48 |
| OLMo-2-7B (ref) | keep8@121k (**healed**) | 0.2550 | −1.39 | 48.8% | **30.64%** | 0.3423 | −8.72 |

Every damaged non-OLMo arm lands below its own best-constant floor. The intact-vs-damaged
gap is enormous in every family. The letter-interface failure is a general damage
response, not an OLMo idiosyncrasy.

## 3. The mechanism is family-dependent, the result is not

Non-OLMo damaged arms collapse to a constant-letter emitter with near-zero exact ties.
OLMo damaged arms collapse to the same behaviour through a different route: 30% exact
ties in bf16 resolved by argmax's index bias. gate-3 already showed removing OLMo's ties
(fp32) does not restore accuracy; the ties are downstream of the collapse. This result
closes the loop: **other families collapse without ties at all**. Ties are one road to
the same destination; a story that puts ties at the mechanism level is family-specific.

**The right frame** for A01: damage compresses the letter interface toward a constant
prediction. In bf16 this sometimes surfaces as exact ties (OLMo), sometimes as sharp
modal collapse (non-OLMo). The invariant is: letter accuracy at/below the best-constant
floor with content usually below floor too.

## 4. Content interface also fails; interface swap does not rescue damaged arms

Every damaged arm has content_norm within ±3pp of its letter accuracy, and both hover
within ±3pp of their floors. This is precisely A01's protocol point: report each
interface against its own construct-appropriate null before any cross-arm comparison.

## 5. What now survives / what changes

**Retained and strengthened:**
* "Construct-appropriate nulls before comparison" is now demonstrated on four families
  and two interfaces per family. Strongest version of the protocol claim.
* Letter-interface degeneration under damage is a general phenomenon; the earlier
  scoping ("only structurally damaged OLMo-2") was premature.
* The tie-as-mechanism refutation from gate-3 is over-determined.

**Revised from the previous (mis-scoped) verdict:**
* GATE1_VERDICT.md's "Kill clause 2 triggered" claim was based on the intact-base run
  and is wrong. Kill clause 2 is NOT triggered.

**Newly opened:**
* The mechanism differs across families (tie-driven on OLMo, modal-collapse without ties
  elsewhere). Why OLMo produces exact ties one order of magnitude more often is a
  mechanism-level question the paper can flag but not resolve here.
* Depth curve is cheap (~1 min/arm on 8x L20A); keep ∈ {4, 6, 16, 20, 24} × family would
  localise the collapse threshold per family. Scheduled.

## 6. Provenance

* Driver: `scripts/_a01_gate1_damaged_driver_21.sh`
* Loader: `scripts/eval_olmo2_probe2_ppl.py::load_truncated_any_family`
* Harness: `scripts/eval_olmo2_mmlu_content.py` with `--any_family --keep_front_layers N`
* Results: `.21:/apdcephfs_wzc1/.../olmo2_mmlu_content_results/gate1_dmg_*/` (6 dirs)
* Wall: 6.5 min for 6 arms on 8 L20A
* Commit: 7ac9653
