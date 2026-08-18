# Global-t / Per-Depth Band Sampler Validation

Artifact: `ops/global_band_state_stats.json`

Validation:

- 1,000 normalized eval rows;
- five independently sampled global timesteps per row;
- 5,000 total states;
- uniform global `t ∈ [0,1]`;
- all states passed mask, role, target, and finite-weight checks.

## Implemented schedule

Content phase:

```text
t ∈ [0.00, 0.45)
```

For a depth-3 example, generated content bands exactly match the design example:

```text
depth 3: [0.00, 0.30]
depth 2: [0.05, 0.35]
depth 1: [0.10, 0.40]
depth 0: [0.15, 0.45]
```

Structural phase:

```text
t ∈ [0.45, 0.95)
```

- equal-width deepest-to-shallowest body bands;
- adjacent structural overlap = 15% of base band width;
- partial line masking uses the selected body’s local clock;
- committed sibling labels remain visible;
- ancestors on the selected path are expanded with committed headers.

Final root phase:

```text
t ∈ [0.95, 1.00]
```

The sampled state is a reachable local transition under strict/local-body
decoding. It does not yet represent multiple desynchronized sibling subtrees in
one example.

## Observed phase counts

| Rung | Count |
|---|---:|
| leaf infill | 2,289 |
| body plan | 1,696 |
| root plan | 1,015 |

The apparent root count exceeds the final 5% band because a module-depth body
transition in the structural phase is also represented as a root-plan state.

Global-t bins are approximately uniform. Observed phase boundaries:

- 0.0–0.4: leaf only;
- 0.4–0.5: content/structural transition;
- 0.5–0.7: body planning;
- 0.7–0.9: body/root structural transition;
- 0.9–1.0: root planning.

## Selected structural/content depths

| Depth | Count |
|---:|---:|
| 0 | 2,529 |
| 1 | 1,078 |
| 2 | 739 |
| 3 | 511 |
| 4 | 129 |
| 5 | 9 |
| 6 | 5 |

This follows the corpus distribution: most educational programs have depth at
most four.

## Targets

| Target | Count |
|---|---:|
| lexical | 56,309 |
| `[delete]` | 12,167 |
| `[expand]` | 9,867 |
| `[STMT]` | 1,595 |
| `[FUNC]` | 925 |
| `[IF]` | 327 |
| `[FOR]` | 163 |
| `[WHILE]` | 43 |

## Canvas and weighting

- mean canvas length: 110.8
- p90 canvas length: 190
- p99 canvas length: 331
- maximum canvas length: 500
- mean supervised masks: 16.3
- median supervised masks: 2
- p99 supervised masks: 130

Local schedule weight is:

```text
min(maximum_weight, u'(t) / u(t))
```

with `maximum_weight=20`, followed by within-sample normalization. The clip is
active frequently:

- mean sample weight: 10.93
- median: 9.22
- p90 and above: 20

This makes clipping a real hyperparameter rather than a numerical footnote. The
first pilot should log unclipped and clipped weights; likely ablations are 10
versus 20.

## Remaining extension

Subsequent implementation added:

- literal all-masked and coupled region-level collapse modes;
- per-top-level-subtree clock offsets;
- simultaneous mixed states for multi-function files;
- exact per-position local band weights resolved from node/body depth.

See `COLLAPSE_STATS.md` and `DESYNC_STATS.md`.
