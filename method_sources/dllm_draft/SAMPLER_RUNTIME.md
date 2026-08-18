# Scaffold-Coder Reverse Runtime and Model Sampler

## Implemented v0 lifecycle

The mutable runtime begins with a hidden module body containing line-level
masks. It supports:

- line targets: `[STMT]`, `[FUNC]`, `[FOR]`, `[WHILE]`, `[IF]`;
- line-level `[expand]` and `[delete]`;
- token-level header/statement masks;
- token-level `[expand]` and `[delete]`;
- deterministic construct templates;
- local-body planning barriers;
- header-before-body gates;
- rule-emitted indentation, colons, and newlines;
- `if -> elif* -> else?`;
- `for/while -> else?`;
- deterministic `pass` for a required body whose slots all delete;
- stable mask IDs across length-changing rewrites;
- tree depth, line count, token count, canvas, expansion, and model-call limits.

The model never predicts a multi-token object. Every commit is one token, after
which the runtime applies rules to a fixed point.

## Local-body barrier

If one line label commits while sibling line masks remain, it stays as a visible
pending label. Its header/body/statement masks do not exist yet. Once every slot
in that body resolves, all pending labels materialize and their local holes open.

This behavior is explicitly unit tested.

## Vocabulary constraints

At line positions, only the line label/edit set is legal. Clause positions use a
family-specific set. Token holes:

- ban all scaffold structural tokens except expand/delete;
- ban mask/BOS/EOS/pad and other special IDs;
- ban vocabulary entries whose decoded piece contains a physical newline;
- ban tokenizer IDs beyond the actual extended tokenizer length.

The sampler uses entropy normalized by `log(|V_allowed|)` by default, avoiding
direct comparison of raw probabilities from radically different support sizes.
A top-1/top-2 margin mode is also available.

## Neural sampling loop

`ScaffoldModelSampler` executes:

```text
rule fixed point
render response canvas
prepend clean chat prompt and append EOS
full-sequence bidirectional model call
Dream shift-op on logits
role-specific vocabulary support
normalized confidence ranking
commit top-k predictions
repeat
```

The production path currently starts with one committed token per call, matching
the quality-oriented Dream/DreamOn setting.

## CPU evidence

A scripted prediction provider drove the complete sampler loop to:

```python
def f(x):
    return x
```

The test exercised:

- root `[FUNC]` prediction;
- function template expansion;
- header token shrink/fill;
- body line deletion and `[STMT]`;
- statement token shrink/fill;
- fixed-point completion and pure-Python rendering.

Additional tests cover pending-label barriers, line expansion, empty-body pass,
illegal target rejection, expansion budgets, clause deletion, and
`elif/else` chains.

The integrated suite now has 114 passing tests locally; the new C2-focused
tests also pass in the remote research environment.

The sampler also records:

- per-call prompt+canvas lengths;
- cumulative processed model tokens;
- minimum/maximum canvas length;
- NFE;
- expansion count;
- C1 leaf-remask count and correction rounds;
- C3 structural deferral count;
- C2 structural-backtrack count;
- parseability of a placeholder-completed program after every step.

Unresolved line slots render as `pass`; unresolved conditions use `True` or
`False`; closed required bodies render as `pass`. This makes intermediate
`ast.parse` a well-defined process metric rather than attempting to parse raw
mask tokens.

## Opt-in correction policies

The default remains monotone C0. Three bounded, inference-only policies are now
implemented and use the same checkpoint.

### C1 leaf remasking

Each model-committed lexical token stores its stable runtime cell ID,
normalized confidence, model-call index, and prior remask count. At a
configured interval and/or once the tree reaches a provisional fixed point,
the sampler may select the lowest-confidence eligible fraction and turn those
committed cells back into ordinary token masks.

Rule-emitted tokens, including deterministic empty-suite `pass`, have no model
provenance and are never eligible. Selection is bounded by:

- a global remask budget;
- a per-token remask budget;
- minimum token age;
- an optional confidence threshold.

The re-masked state is a normal partially masked token-hole state. Correction
cost is included in NFE and cumulative model tokens.

### C3 structural confidence gate

Before a predicted construct or clause label is committed and expanded, an
optional confidence threshold can defer it for a bounded number of model
calls. The conservative v0 implementation keeps the position masked while it
waits, rather than exposing an unsupervised provisional-token state. Other
independent masks may commit and change the context; the label can change on a
later call. Once the bounded defer count is exhausted, the best current label
is allowed through so the gate cannot deadlock generation.

`[STMT]`, `[expand]`, and `[delete]` are not gated. C3 is prevention rather
than post-expansion structural correction.

### C2 structural subtree backtracking

Every model-committed construct line and clause anchor retains its stable
runtime ID, confidence, model-call index, and prior backtrack count. Once the
expanded subtree is complete, the runtime can calculate the mean confidence of
all model-produced lexical tokens inside it. A bounded C2 policy may delete the
expanded template and descendants and restore one line-level or clause-level
mask at the same tree location.

This is the exact reverse-runtime counterpart of the legal forward collapse:
the replacement target is again one ordinary mask, not a multi-token model
prediction. Nested eligible subtrees are repaired deepest-first so the sampler
discards the smallest region before considering an ancestor. Selection is
bounded by:

- a global structural-backtrack budget;
- a per-anchor backtrack budget;
- minimum age since the final content commit;
- a mean-content-confidence threshold.

Backtracking clears stale C3 deferrals and elastic-cycle fingerprints. Every
backtrack adds to correction rounds, NFE remains the actual model-call count,
and cumulative model tokens include all regenerated canvases.

The EvalPlus and neural-smoke CLIs expose C1/C2/C3 settings. C0-compatible
defaults keep every correction budget at zero and set no structural threshold.
Thresholds must be calibrated on a held-out slice, and comparisons are made at
matched NFE and cumulative model-token budgets.

CPU scripted tests verify:

- a low-confidence lexical token is re-masked and replaced at completion;
- a low-confidence `[FUNC]` proposal is deferred and can change to `[STMT]`
  before any function template is expanded;
- a completed low-confidence `[FUNC]` subtree is collapsed and regenerated as
  a different line type;
- completed `else`/`elif` clause subtrees can be restored to their clause mask;
- completion on exactly the final allowed model call succeeds;
- rule-emitted tokens cannot enter the C1 candidate set.

## Remaining GPU and optimization work

- run the neural sampler against a trained Scaffold checkpoint;
- add length-bucketed multi-sample evaluation;
- record cumulative canvas tokens and wall time;
- calibrate and evaluate C1/C2/C3 at matched NFE;
- add optional class/try/with runtime templates after v0.
