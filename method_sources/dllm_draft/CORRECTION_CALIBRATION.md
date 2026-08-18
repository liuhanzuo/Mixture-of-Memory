# Held-Out Correction Calibration

This protocol tunes inference-only correction policies without using
HumanEval, MBPP, or their Plus tests as a development set.

## Calibration set

`scripts/build_correction_calibration_set.py` deterministically samples 32
rows from the fixed 1,000-row `educational_instruct` evaluation split:

| Canonical compound depth | Rows |
|---|---:|
| 0–1 | 12 |
| 2 | 12 |
| 3+ | 8 |

The sampling seed is `20260724`. Each task stores the original instruction,
required function header, test assertions, canonical code, depth, and token
lengths. The builder executes every selected canonical solution under the same
resource-limited evaluator and replaces any row whose oracle does not pass.
The manifest records input/output SHA256 values, task IDs, quotas, and rejected
oracle rows.

The required function header is seeded into the Scaffold runtime for every
arm. This isolates correction behavior from benchmark-signature transcription
and gives every threshold the same initial tree.

## Execution harness

`scripts/evaluate_correction_calibration.py` combines generated code with the
held-out assertions and runs each sample in a fresh isolated Python process
with:

- deterministic Python `random.seed(0)`;
- a wall-clock timeout (default five seconds);
- CPU, address-space, file-size, file-descriptor, process-count, and core-dump
  resource limits;
- no stdin and isolated-mode Python;
- bounded stdout/stderr files.

Outcomes distinguish generation error, empty output, syntax error, timeout,
assertion failure, runtime error, and pass. Reports include pass/parse rate,
depth slices, NFE, cumulative model tokens, and structural-backtrack count.
The 32 canonical solutions pass 32/32 under this harness.

## C2 sweep

`scripts/run_c2_calibration_sweep_8gpu.sh` runs the existing Scaffold Stage-1
checkpoint on all eight GPUs with one committed token per model call and a
256-NFE hard limit:

| Arm | Mean-content confidence threshold | Max backtracks |
|---|---:|---:|
| `c0` | disabled | 0 |
| `c2_t005` | 0.05 | 1 |
| `c2_t010` | 0.10 | 1 |
| `c2_t020` | 0.20 | 1 |
| `c2_t030` | 0.30 | 1 |

All C2 arms use minimum age one call and at most one backtrack per structural
anchor. Generation is resumable per rank.

## Selection rule

`scripts/select_correction_calibration.py` selects a C2 arm only when it:

1. strictly improves held-out execution pass rate over C0;
2. does not reduce parse rate;
3. stays within 1.25× C0 mean NFE;
4. stays within 1.35× C0 mean cumulative model tokens.

Among eligible arms, pass rate is primary, parse rate secondary, and lower
token cost breaks ties. Otherwise C0 remains selected. This is a routing rule
for a later separately reported correction experiment, not a significance
claim from 32 samples.

The unattended sweep emits both machine-readable
`ops/artifacts/c2_calibration_selection.json` and a human-readable
`ops/artifacts/C2_CALIBRATION_RESULTS.md`.

## C1 and C3 sweeps

After C2 completes, `scripts/run_c1c3_calibration_sweep_8gpu.sh` reuses the
same immutable 32-task set and C0 baseline:

- C1 completion remasking: thresholds 0.05/0.10/0.20, lowest 5% of eligible
  leaves, at most eight total remasks and one per token;
- C3 structural deferral: thresholds 0.05/0.10/0.20 and at most one defer call
  per structural mask.

C1 and C3 are selected independently under the same pass/parse/NFE/token gates
as C2. They emit `c1_calibration_selection.json`,
`C1_CALIBRATION_RESULTS.md`, `c3_calibration_selection.json`, and
`C3_CALIBRATION_RESULTS.md`.

After all three policies finish,
`scripts/summarize_correction_policies.py` chooses the strongest independently
eligible arm and writes `correction_policy_selection.json` plus
`CORRECTION_POLICY_SELECTION.md`. This is still only a held-out routing
decision; the chosen policy requires a separately reported evaluation.

## Disjoint validation

`scripts/run_selected_correction_validation_8gpu.sh` samples a second,
oracle-clean 64-task set (24/24/16 by depth, seed `20260725`) after excluding
every calibration task. It regenerates C0 and the selected correction arm,
then reports paired pass delta, bootstrap interval, exact McNemar, parse delta,
NFE ratio, token ratio, and correction action counts. The outputs are
`correction_validation.json` and `CORRECTION_VALIDATION.md`.

## Observed result

On the 32-task calibration set, C0 achieved 18.75% execution pass and 81.25%
parseability. Every tested C1, C2, and C3 threshold produced exactly the same
outputs and zero correction actions:

```text
mean C1 leaf remasks          0
mean C2 structural backtracks 0
mean C3 structural deferrals  0
```

Unified routing therefore selected C0. On the disjoint 64-task validation set,
C0 achieved 31.25% pass and 84.38% parseability; selected-versus-baseline
delta was exactly zero. This means the tested threshold grid was inactive. It
does not establish that correction is intrinsically ineffective; an adaptive
grid must first measure the checkpoint's actual confidence distribution.
