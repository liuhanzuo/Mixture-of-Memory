# Six-review synthesis

## Scores

| Group | Soundness | Excitement | Overall | Confidence | Reproducibility |
|---|---:|---:|---:|---:|---:|
| Strict | 3.00 [3.0, 3.0] | 2.50 [2.5, 2.5] | 2.83 [2.5, 3.0] | 4.33 [4.0, 4.5] | 2.00 [2.0, 2.0] |
| Normal | 3.50 [3.5, 3.5] | 2.83 [2.5, 3.0] | 3.00 [3.0, 3.0] | 4.17 [4.0, 4.5] | 2.33 [2.0, 2.5] |
| All six | 3.25 [3.0, 3.5] | 2.67 [2.5, 3.0] | 2.92 [2.5, 3.0] | 4.25 [4.0, 4.5] | 2.17 [2.0, 2.5] |

## Individual reviews

- `v5_strict_1_GPT56.md` (strict): S=3.0, E=2.5, O=2.5, C=4.0, R=2.0
- `v5_strict_2_GPT56.md` (strict): S=3.0, E=2.5, O=3.0, C=4.5, R=2.0
- `v5_strict_3_GPT56.md` (strict): S=3.0, E=2.5, O=3.0, C=4.5, R=2.0
- `v5_normal_1_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.5, R=2.5
- `v5_normal_2_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=2.5
- `v5_normal_3_GPT56.md` (normal): S=3.5, E=2.5, O=3.0, C=4.0, R=2.0

## Critique extraction note

Consensus critique must be synthesized manually from the six evidence-anchored reports. Treat issues raised by >=3 reviewers as consensus; retain one-review concerns as outliers rather than deleting them.

## Consensus critique (manual synthesis)

### Consensus major issues

1. **No run-level replication for the load-bearing path (6/6).** All reviewers
   accept the literal keep14 trace but reject seed-stable dynamics or general
   recovery claims. Item-level intervals condition on one historical model.
2. **No matched 200k intact continuation (6/6).** full32@25k only bounds early
   corpus shift; it cannot identify whether the 200k keep14 deficit is
   intervention-specific relative to a same-horizon intact model.
3. **Exact archival reproduction remains blocked (6/6).** Missing historical
   seeds and within-epoch loader offset, incomplete GPU-hour/FLOP accounting,
   local-only evaluator commits, and incomplete aligned prediction bundles
   continue to drive reproducibility near 2.0 despite much better documentation.
4. **The paper does not operationalize a prospective PPL certificate (3/6).**
   Several reviewers found the title/certificate framing broader than the actual
   test: keep14 PPL improves but remains 1.428x the base, and no predeclared
   threshold/stopping/calibration rule is evaluated. The safest claim is “PPL
   improvement alone does not imply target recovery along this path.”

### Consensus secondary issues

5. **External validity and novelty remain modest (4/6).** One principal OLMo
   recipe/family and known loss--task dissociation make this a careful control
   synthesis rather than a new phenomenon or method.
6. **Strongest alternative construction lacks closed-book QA (4/6).** ShortGPT
   is the strongest 16-layer comparator but is missing the generation checks used
   to exclude an answer-letter artifact.
7. **Scope of the proxy is narrow (4/6).** PPL is same-source and in-domain,
   with no out-of-domain likelihood or contamination audit; “knowledge-sensitive”
   spans heterogeneous constructs.
8. **Controls remain operating points, not causal isolation (3/6).** ShortGPT,
   random, and frozen expose useful boundaries but cannot identify which
   structural/adaptation factor causes the endpoint gap.

### Local presentation/audit issues

- “Continuation-length normalized” refers to character length in one task table
  and token length in another; rename these explicitly.
- `core6` in the ShortGPT table is undefined in one reviewer's audit.
- One reviewer considered “near the base” too strong for some full32 closed-book
  values.
- Figure 1 is conceptually much improved, but one strict reviewer flagged small
  text under ACL readability guidance.
- The nearest-work matrix should define subjective binary criteria.

### Strength consensus

- v5's proxy-validity framing, Figure 1, and main operating-point table are much
  clearer and more scientifically honest than v4.
- Confounds are exposed at first use rather than hidden in Limitations.
- The random content-score floor, paired interface analysis, closed-book checks,
  and explicit non-claims make the measurement contribution useful.
- Documentation, sample counts, hashes, and evaluator provenance improved even
  though historical reproducibility cannot be recovered.

## Score trend and decision

v5 all-six Overall is **2.92**, improving from v4 **2.75**. Strict mean rises
from **2.50 to 2.83**, and the range tightens to **2.5–3.0**; normal reviewers
remain unanimous at **3.0**. Thus the rewrite clearly improved the paper into a
more stable Findings-level measurement study. It is still not main-conference
ready because the limiting factors are now experimental and archival, not prose.

## Next iteration target

Without new training, make one final writing/analysis pass:

1. narrow “certificate” to “improvement alone does not imply recovery,” or add a
   clearly post-hoc threshold sensitivity analysis from existing checkpoints;
2. fix metric-normalization terminology, undefined labels, and any overbroad
   “near base” wording;
3. make Figure 1 text comfortably readable at the final two-column size;
4. package all currently available evaluator code/predictions/manifests anonymously.

If new experiments are authorized, prioritize: (a) 200k full32, (b) matched
keep14 seeds, and (c) ShortGPT closed-book QA. Further unmatched benchmark/model
expansion is lower value.
