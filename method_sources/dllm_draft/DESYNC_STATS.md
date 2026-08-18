# Per-Top-Level Desynchronization Diagnostics

Implementation:

- one sampled global `t`;
- independent `δ_f ~ U(-σ_d, σ_d)` for each top-level line/subtree;
- each subtree rendered at `clip(t + δ_f, 0, 1)`;
- prompt included once;
- generated subtree canvases concatenated in original module order;
- per-subtree loss mass averaged so multi-function files do not receive
  automatically larger sample weight.

## Corpus applicability

On the 1,000-example normalized eval split:

```text
204 / 1,000 = 20.4%
```

contain more than one top-level line and can exhibit cross-subtree
desynchronization. Most remaining examples are single-function benchmark-style
programs and reduce to the synchronized sampler.

## Mixed-rung frequency

Five states were sampled per eligible example:

| σ_d | Mixed rung fraction among multi-top-level states | Approx. fraction over all eval states |
|---:|---:|---:|
| 0.10 | 36.57% | 7.46% |
| 0.20 | 42.45% | 8.66% |
| 0.30 | 47.35% | 9.66% |

At `σ_d=0.20`, observed combinations include:

- body-plan + root-plan;
- body-plan + leaf-infill;
- leaf-infill + root-plan;
- multiple independently expanded leaf subtrees.

Recommendation:

- strict/local-body pilot: `σ_d=0`;
- first soft-decoding checkpoint: `σ_d=0.20`;
- ablate 0.10 and 0.30 if soft decoding is beneficial.

## Current limitation

Offsets currently apply to top-level siblings only. This covers independent
functions and module statements while preserving their shared bidirectional
context. Nested sibling loops/branches still use one local clock. Extending
offsets recursively requires a precedence-preserving subtree clock sampler and
remains part of `COLLATOR-003`.

