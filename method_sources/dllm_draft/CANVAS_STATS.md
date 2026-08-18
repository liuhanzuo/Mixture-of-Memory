# Deterministic Canvas-Rung Validation

Validation set: 1,000 normalized educational_instruct examples  
Body-plan states generated: 4,295  
Artifact: `ops/canvas_state_validation.json`

Every state passed:

- source/role length consistency;
- supervised-position mask invariant;
- immediate-rung target-law validation;
- role legality;
- special-token atomicity;
- clean segmented-canvas decode equality with canonical Python.

## Canvas lengths

| State | Mean | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| clean expanded code | 91.4 | 79 | 166 | 207 | 315 | 439 |
| root line plan | 2.64 | 2 | 4 | 6 | 8 | 24 |
| full template skeleton | 38.1 | 35 | 66 | 82 | 121 | 172 |
| local body-plan context | 33.2 | 32 | 66 | 77 | 98 | 135 |
| chat prompt + root plan | 57.3 | 50 | 86 | 103 | 137 | 226 |

Relative to fully expanded code:

- root-plan canvas is approximately **97.1% shorter**;
- full template skeleton is approximately **58.3% shorter**;
- local body-plan context is approximately **63.7% shorter**.

These numbers empirically support lazy typed holes as a canvas-saving mechanism,
not only a conceptual planning signal.

## Supervised masks

| State | Mean | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| root line plan | 1.32 | 1 | 2 | 3 | 4 | 12 |
| 50% leaf corruption | 33.0 | 28 | 62 | 78 | 117 | 173 |
| local body plan | 2.02 | 1 | 4 | 5 | 7 | 16 |

The root task is extremely compressed and structurally sparse. Structural
targets will require explicit sampling/loss balancing so that one or two
line-label targets are not drowned out by dozens of leaf targets.

## Implemented deterministic rungs

1. hidden module root opened into line-level masks;
2. full template skeleton with `[HDR]`, `[STMT]`, and `[CLAUSES]`;
3. leaf-token infilling with rule text never masked;
4. target-body local planning state with committed ancestor headers, collapsed
   sibling labels, and line-level masks in the selected body;
5. chat-prompt composition with clean prompt and response-only supervision.

Continuous bands, desynchronization, and DreamOn merge/delete augmentation are
not included in this validation and remain the next collator milestone.

