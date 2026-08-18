# Six-review synthesis

## Scores

| Group | Soundness | Excitement | Overall | Confidence | Reproducibility |
|---|---:|---:|---:|---:|---:|
| Strict | 2.83 [2.5, 3.0] | 2.50 [2.5, 2.5] | 2.83 [2.5, 3.0] | 4.00 [4.0, 4.0] | 1.83 [1.5, 2.0] |
| Normal | 3.33 [3.0, 3.5] | 2.83 [2.5, 3.0] | 3.00 [3.0, 3.0] | 4.00 [4.0, 4.0] | 2.33 [2.0, 2.5] |
| All six | 3.08 [2.5, 3.5] | 2.67 [2.5, 3.0] | 2.92 [2.5, 3.0] | 4.00 [4.0, 4.0] | 2.08 [1.5, 2.5] |

## Individual reviews

- `v6_strict_1_GPT56.md` (strict): S=3.0, E=2.5, O=3.0, C=4.0, R=1.5
- `v6_strict_2_GPT56.md` (strict): S=2.5, E=2.5, O=2.5, C=4.0, R=2.0
- `v6_strict_3_GPT56.md` (strict): S=3.0, E=2.5, O=3.0, C=4.0, R=2.0
- `v6_normal_1_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=2.5
- `v6_normal_2_GPT56.md` (normal): S=3.0, E=2.5, O=3.0, C=4.0, R=2.0
- `v6_normal_3_GPT56.md` (normal): S=3.5, E=3.0, O=3.0, C=4.0, R=2.5

## Critique extraction note

Consensus critique must be synthesized manually from the six evidence-anchored reports. Treat issues raised by >=3 reviewers as consensus; retain one-review concerns as outliers rather than deleting them.

## Consensus critique (manual synthesis)

### Consensus major issues

1. **No training-run replication (6/6).** The improvement-only claim is valid
   for the literal trace, but its magnitude/stability and reporting
   recommendation cannot be generalized over optimization randomness.
2. **No 200k intact continuation (6/6).** full32@25k cannot adjudicate the
   long-horizon counterfactual.
3. **Artifact/reproduction remains incomplete (6/6).** v6 created a useful
   anonymous artifact locally, but the frozen reviewer source produced by the
   freeze script did not include `anonymous_artifact/`; multiple reviewers
   therefore could not inspect the promised files. Even with it attached,
   historical seeds/loader offset and closed-book per-item generations remain
   unavailable.
4. **The improvement-only premise is narrow/non-operational (3/6).** The rewrite
   correctly disclaims calibrated thresholds, but several reviewers still find
   the question practically weak because final PPL remains 1.428x base and no
   prospective stopping/certification rule is evaluated.

### Consensus secondary issues

- ShortGPT still lacks closed-book QA and aligned per-item closed-book outputs.
- Same-source PPL lacks OOD likelihood/contamination audit.
- Figure 1 is more readable but remains dense for some strict reviewers.
- Interface scoring is multi-factor and cannot isolate answer-symbol effects.

### What v6 fixed

- The broad “certificate” overclaim was replaced by the precise statement that
  **improvement alone does not imply target recovery on the observed path**.
- Metric-normalization terminology, `core6`, full32 wording, prior-work criteria,
  bibliography rendering, and Figure 1 minimum text size were improved.
- An anonymous artifact snapshot was assembled with substantial per-item
  score-level evidence and evaluator source snapshots.

## Score trend and stopping assessment

v6 all-six Overall is **2.92**, exactly the same as v5 **2.92**. The manuscript
has converged to a stable Findings-level score. v6's local improvements were
partly offset by reviewers discovering that the frozen review-source snapshot
excluded the newly assembled artifact; this is a packaging bug that should be
fixed immediately, but it does not recover the historically missing training
state.

No further prose-only iteration is likely to reach ACL main. The final
non-experimental tasks are:

1. update the freeze/submission packaging so the promised anonymous artifact is
   actually included and verify it from a clean extracted archive;
2. ensure raw paths/credentials/benchmark text are absent;
3. preserve the narrow improvement-only claim and Figure 1 readability.

The research blockers are unchanged: matched seeds, full32@200k, ShortGPT
closed-book evaluation, and OOD/contamination evidence. Paper B is ready as a
carefully bounded Findings submission after the artifact packaging fix, not as
a stable main-conference submission.
