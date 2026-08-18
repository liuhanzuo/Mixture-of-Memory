# Sampled Rung/Edit Collator Diagnostics

Validation:

- normalized eval rows: 1,000
- stochastic samples per row: 5
- total sampled training states: 5,000
- artifact: `ops/sampled_state_stats.json`

## Rung mixture

Configured probabilities:

- root plan: 0.20
- local body plan: 0.30
- leaf infill: 0.50

Observed:

| Rung | Samples | Fraction |
|---|---:|---:|
| root plan | 930 | 18.6% |
| body plan | 1,535 | 30.7% |
| leaf infill | 2,535 | 50.7% |

The sampler is deterministic for a fixed sample seed.

## Final edit-augmentation defaults

- token merge base probability: 0.5
- 50/50 static vs. dynamic-inverse scheduler
- dynamic-inverse merge probability: `0.5 * (1 - local_u)`
- line merge probability: 0.5
- maximum synthetic token deletes per hole: 1
- maximum synthetic line deletes per body: 1

An initial direct translation using up to eight delete targets **per hole**
produced 97,103 delete targets versus 63,597 lexical targets in only 5,000
states. DreamOn applies its large delete range to one FIM middle, so that range
does not transfer directly to a program containing many holes. Reducing the
multi-hole default to one delete per hole produced:

| Target | Count |
|---|---:|
| lexical | 63,277 |
| `[expand]` | 13,665 |
| `[delete]` | 13,106 |
| `[STMT]` | 1,572 |
| `[FUNC]` | 853 |
| `[IF]` | 272 |
| `[FOR]` | 141 |
| `[WHILE]` | 39 |

Role counts:

| Role | Count |
|---|---:|
| token statement | 54,902 |
| token header | 33,074 |
| body line | 3,360 |
| module line | 1,589 |

Rare structural targets remain sparse in raw token counts. The collator
therefore normalizes token losses within each sample and gives the whole sample
the local schedule weight. This prevents a leaf state with dozens of masks from
automatically outweighing a root state with one structural mask.

## State sizes

- mean canvas length: 112.5 tokens
- p90: 191
- p99: 349
- maximum: 458
- mean supervised masks: 18.6
- median supervised masks: 5
- p99 supervised masks: 123
- maximum supervised masks: 268

All sampled states passed role/target-law checks and had finite non-negative
weights.

## Weighting

Current experimental mode:

```text
base sample weight = min(20, 1 / local_u)
```

Within a sample:

- synthetic delete targets share one ordinary target’s aggregate weight;
- all positive weights are rescaled so their sum equals the base sample weight.

This is an implementation hypothesis, not yet a claimed global MDLM NELBO.
Released-DreamOn `1-t` weighting remains a baseline mode.

## Remaining work

The current sampler is a validated mixture of reachable deterministic rungs. It
does not yet implement:

- one global continuous `t` with per-depth bands;
- construct-collapse hazards as a function of region length;
- sibling/subtree desynchronization;
- per-depth target statistics;
- exact released-DreamOn attention-mask merge representation.

Those items remain `COLLATOR-002`.

