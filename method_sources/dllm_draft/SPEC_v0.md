# Scaffold-Coder Normative Specification v0

Status: implementation specification  
Date: 2026-07-22  
Normative precedence: this file supersedes conflicting statements in the
earlier plan, mechanism-design, and Q&A drafts.

## 1. Objective and claim boundary

Scaffold-Coder is a full-sequence masked diffusion language model augmented with
a deterministic, tree-aware rewrite runtime. The model predicts one token at
each masked position. The runtime performs every multi-token or variable-length
operation:

- expand a predicted construct label into a Python template;
- open typed holes;
- split `[expand]` into two masks of the same granularity;
- remove `[delete]`;
- emit indentation, colons, clause layout, and newlines;
- maintain ownership of every generated line by a program-tree node.

The primary guaranteed property is:

> Compound-statement block topology, indentation, template punctuation, and
> clause attachment/order are valid by construction.

The v0 method does **not** guarantee that arbitrary header or simple-statement
content is valid Python. Full-program parseability is conditional on each leaf
hole producing a valid member of its Python syntactic category. Papers and
documentation must use “structural/block validity by construction,” not an
unqualified “Python syntax is guaranteed.”

## 2. Authoritative representation

### 2.1 Program IR

The runtime owns a typed program tree rather than inferring structure from a
flat generated string. At minimum, every node records:

- stable node ID;
- node kind;
- parent node ID and child-list role;
- indentation depth;
- source role and hole subtype;
- ordered child lines;
- current lifecycle state;
- committed token IDs or unresolved masks;
- generation confidence metadata;
- expansion/delete budget.

The module is a hidden `MODULE_BODY` root. It is a list of top-level line slots
and does not require a model-visible `[MODULE]` token.

### 2.2 Canonical normalization

Accepted training code is parsed and converted to the custom IR, then rendered
to a canonical target. The target of training is the canonical renderer output,
not the original formatting.

Required normalization:

- four-space indentation;
- one simple statement per physical line;
- no semicolon-separated statements;
- no statement-internal physical newline in v0;
- standalone comments removed;
- a deterministic blank-line policy;
- unsupported syntax filtered or normalized away;
- final newline policy fixed and tested.

For accepted examples:

```text
render(parse(normalize(code))) == normalize(code)
```

Text equality is preferred. AST equivalence is a secondary diagnostic, not a
replacement for deterministic canonical rendering.

### 2.3 Token-canvas policy

The tree is authoritative, but the model consumes a flat token-ID canvas.
Canvas construction uses a deterministic **segmented tokenizer**:

1. rule-emitted fragments, committed leaf fragments, masks, and special tokens
   are separate typed segments;
2. special tokens are atomic tokenizer entries;
3. ordinary text segments are encoded with the base tokenizer using a fixed
   boundary convention;
4. segment token-ID sequences are concatenated without retokenizing already
   committed segments;
5. decoding the concatenated token IDs must reproduce the rendered text exactly.

The required invariant is:

```text
decode(segment_tokenize(render_segments(tree))) == render_text(tree)
```

It is **not** required that the segmented tokenization equal
`encode(render_text(tree))` token-for-token. Training and inference must use the
same segmented-tokenization policy. Canonical whole-text tokenization mismatch
rate is measured during the tokenizer pilot.

This policy avoids changing previously committed token boundaries after a
mid-canvas template expansion.

## 3. Vocabulary and token categories

Implementation token allocation reserves the first free Dream-Coder ID for
`<|expand|>`, matching DreamOn. Paper notation such as `[FUNC]` maps to rare
atomic strings such as `<|sc_func|>`. On the pinned checkpoint, the tokenizer is
shorter than the configured 152,064-row embedding/LM-head matrices, so v0 uses
reserved rows and must not shrink the model with
`resize_token_embeddings(len(tokenizer))`.

### 3.1 Predicted line labels

These tokens may be targets of a line-level mask:

- `[STMT]`
- construct labels enabled by the grammar, for example:
  `[FUNC]`, `[CLASS]`, `[FOR]`, `[WHILE]`, `[IF]`, `[TRY]`, `[WITH]`
- clause labels, but only inside an appropriate clause hole:
  `[ELIF]`, `[ELSE]`, `[EXCEPT]`, `[FINALLY]`
- `[expand]`
- `[delete]`

`[STMT]` is a predicted leaf line label. It is not a rule-only template hole.

### 3.2 Rule-only typed holes

These tokens are emitted by template rules and are never prediction targets:

- `[HDR]`
- `[DOC]`
- `[BODY]`
- `[CLAUSES]`

They may appear as visible conditioning context while closed. They are consumed
by deterministic opening or construct-collapse rules. No training example may
contain a mask whose target is one of these four tokens.

The runtime may attach an invisible subtype to a visible `[HDR]`, for example:

- function signature;
- `for` target-and-iterator;
- Boolean condition;
- class header;
- `with` items;
- `except` type/alias.

One visible token can therefore retain a compact vocabulary while allowing
different opening lengths, constraints, and validation.

### 3.3 Edit labels

The semantic runtime interface shares one edit pair across granularities:

- `[expand]`: replace this position with two masks of the same runtime role;
- `[delete]`: remove this position and its enclosing slot wrapper, if any.

`[delete]` is an abstract action name. The released DreamOn checkpoint maps this
action to EOS rather than a dedicated added token. Backends must declare one of:

- `dreamon-eos-delete`, used for exact upstream reproduction;
- `dedicated-delete`, used when a distinct atomic token is trained.

The runtime never infers granularity from token text. Every position has an
explicit role in the source map. Separate line/token edit labels remain the
`E-lex` ablation.

### 3.4 Mask token

The base model’s ordinary mask token is reused at both granularities. Its role
is carried by the runtime source map:

- `LINE_BODY`
- `LINE_MODULE`
- `LINE_CLAUSE`
- `TOKEN_STMT`
- `TOKEN_HDR`
- `TOKEN_DOC`

The model sees role-indicating textual context; the runtime additionally applies
role-specific vocabulary constraints.

## 4. Single-token target law

Every model-supervised masked position has exactly one immediate-rung target.
No model prediction creates multiple tokens.

| Mask role | Legal immediate targets | Deterministic action after commit |
|---|---|---|
| `LINE_MODULE` / `LINE_BODY` | `[STMT]`, legal construct label, `[expand]`, `[delete]` | open leaf, expand construct, split slot, or remove slot |
| `LINE_CLAUSE` | legal clause label for current state, `[delete]` where legal | expand clause/state transition or omit clause |
| `TOKEN_STMT` | ordinary vocabulary, `[expand]`, `[delete]` | retain token, split mask, or remove token position |
| `TOKEN_HDR` | allowed ordinary vocabulary, `[expand]`, `[delete]` if subtype permits empty | retain token, split mask, or remove token position |
| `TOKEN_DOC` | allowed ordinary vocabulary, `[expand]`, `[delete]` | retain token, split mask, or omit/shrink docstring |

Forbidden targets at every model-supervised mask include `[HDR]`, `[DOC]`,
`[BODY]`, and `[CLAUSES]`.

Training-time assertions must reject examples that violate this table.

## 5. Runtime rewrite rules

### 5.1 Construct expansion

A newly committed construct label is replaced in place by a typed template.
Examples:

```text
[FUNC]
  -> "def " + [HDR:function] + ":" + newline
     + [DOC]? + [BODY]

[FOR]
  -> indent + "for " + [HDR:for] + ":" + newline
     + [BODY] + [CLAUSES:for]

[IF]
  -> indent + "if " + [HDR:condition] + ":" + newline
     + [BODY] + [CLAUSES:if]

[WHILE]
  -> indent + "while " + [HDR:condition] + ":" + newline
     + [BODY] + [CLAUSES:while]
```

Indentation and punctuation fragments are rule-emitted and never masked.
Expansion creates closed holes, not raw content masks.

### 5.2 Header opening

Opening `[HDR]` replaces it with an initial token-level mask run. Initial length
depends on the invisible header subtype. The run grows or shrinks through
token-level `[expand]` and `[delete]`.

V0 constraints include:

- no token whose decoded text contains a physical newline;
- no structural special token except `[expand]`/`[delete]`;
- subtype-specific minimum and maximum token budgets;
- non-empty headers where Python requires content.

Full incremental Python grammar constraints are outside v0.

### 5.3 Statement opening

A line-level mask that commits `[STMT]` initially remains a one-token line label.
When its gate opens, `[STMT]` is replaced by a token-level mask run. The
statement content grows or shrinks with token-level edit labels.

`[STMT]` denotes one canonical physical line. It cannot emit a physical newline.

### 5.4 Body opening

Opening `[BODY]` replaces it with a list of line slots. Each slot contains:

```text
rule-indent + line-level mask + rule-newline
```

It does **not** contain a pre-committed `[STMT]`.

A line-level `[expand]` replaces one line slot with two unresolved line slots.
A line-level `[delete]` removes the entire slot, including rule indentation and
newline wrappers.

Every Python suite has minimum cardinality one. The runtime must choose exactly
one of these policies and use it in training and inference:

1. disallow deletion of the final slot; or
2. deterministically render an otherwise empty suite as `pass`.

V0 adopts policy 2 because it guarantees termination and a valid suite even
under pathological predictions.

### 5.5 Module-root opening

Generation from scratch begins by opening the hidden `MODULE_BODY` root into a
small list of top-level line slots, not an undifferentiated adjacent token-mask
run. V0 uses one or two initial slots; line-level expansion supplies additional
top-level definitions/statements.

The training collator must explicitly generate root-open states matching this
initialization.

### 5.6 Clause holes

`[CLAUSES]` is a rule-only specialized line hole at the owning construct’s
indentation.

For `if`:

```text
[CLAUSES:if] -> [delete] | [ELIF] | [ELSE]
[ELIF]       -> "elif " + [HDR:condition] + ":" + [BODY] + [CLAUSES:if]
[ELSE]       -> "else:" + [BODY]
```

For `try`, a finite-state controller enforces:

```text
except+ -> else? -> finally?
```

with the Python requirement that a `try` have at least one `except` or
`finally`. For `for` and `while`, the only optional trailing clause is `else`.

Illegal clause labels are removed from the role-specific softmax support.

Absent optional nodes are trained with synthetic `[delete]` supervision. The
forward IR therefore includes epsilon-capable optional components even when no
source-code span exists.

### 5.7 Docstrings

Docstrings are not required for the first model pilot. When enabled:

- the renderer emits delimiters by rule;
- v0 doc content is single-line;
- physical newlines and delimiter-breaking sequences are forbidden;
- an absent docstring is represented by `[delete]`;
- a docstring, when present, is structurally first in the suite.

The first runtime implementation may strip docstrings and add them after core
round-trip tests pass.

### 5.8 Rule fixed point

After each commit step, all newly enabled deterministic rewrites execute to a
fixed point before the next model call:

1. expand committed construct/clause labels whose gate permits expansion;
2. open newly enabled holes;
3. apply committed edit labels;
4. insert `pass` for a resolved empty required suite;
5. rebuild the flat canvas and role/source map.

Rule moves do not consume NFE.

## 6. Gating and decode partial order

### 6.1 Mandatory local dependencies

The following dependencies are invariant:

- a construct must be predicted before its template exists;
- a required header must finish before its body opens;
- a line slot must resolve to `[STMT]` or a construct before leaf content exists;
- descendants cannot exist before the owning ancestor template is expanded.

### 6.2 V0 local-body barrier

The first trainable implementation uses a local-body barrier:

1. open a body into line-level slots;
2. resolve all line-level masks and line-level edit operations in that body;
3. only then open its `[STMT]` leaves and expand its predicted child constructs;
4. different already-enabled bodies may progress in parallel.

This is less globally synchronous than strict decoding but preserves a visible
plan for each body before filling its details.

### 6.3 Ablation modes

- `strict-global`: all masks at a global frontier resolve before any finer
  frontier opens;
- `local-body` (implementation v0);
- `soft-immediate`: a child opens as soon as its own local parent dependency is
  satisfied;
- `none`: meaningful only for eager-materialization controls.

The final paper primary may use `soft-immediate` if desynchronized training
demonstrably supports it. It is not required for the first working checkpoint.

## 7. Forward corruption

### 7.1 Interpretation of global time

Global `t ∈ [0,1]` is a monotonic hierarchy/corruption progress coordinate. With
class-specific schedules and sequence collapse, it is not generally equal to
the global expected mask fraction.

Each role/level has a local clock:

```text
u_l(t) = clip((t - a_l) / (b_l - a_l), 0, 1)
```

Adjacent bands may overlap. Desynchronization offsets are disabled in the first
pilot and added with soft decoding.

### 7.2 Local stochastic operation and deterministic collapse

Within an open token region, ordinary target tokens are independently masked
according to its active local clock. Within an open body, line labels are
independently line-masked according to the body’s structural clock.

Deterministic collapse runs after stochastic masking:

1. fully masked simple-statement content collapses to `[STMT]`;
2. fully masked header content collapses to rule-only `[HDR]`;
3. fully masked doc content collapses to rule-only `[DOC]`;
4. a body whose line entries are fully line-masked collapses to rule-only
   `[BODY]`;
5. a fully collapsed construct template collapses to its predicted construct
   label;
6. that construct label can be line-masked only when its parent body is noised.

Thus a construct label’s own template-collapse depth and its later masking depth
are distinct events. For example, an expanded `if` collapses to `[IF]` at the
`if` node, but `[IF]` becomes a line-level mask only while corrupting the
parent body containing that line.

### 7.3 Collapse-frequency control

Independent masking makes early all-masked probability depend on region length
as `u^n`. The collator must record collapse frequency by region length.

V0 may force collapse when a local clock reaches one. If long regions receive
insufficient collapsed-state coverage, replace accidental all-mask triggering
with an explicit region-level collapse hazard. This change must be recorded as
a corruption-distribution revision, not hidden as an implementation detail.

### 7.4 DreamOn augmentation

Inside open token regions:

- consecutive token masks may be represented through token-level `[expand]`
  supervision;
- synthetic trailing token positions target `[delete]`;
- delete losses use the reproduced DreamOn downweighting.

Inside open body/root line lists:

- consecutive line masks may be represented through line-level `[expand]`
  supervision;
- synthetic surplus line slots target `[delete]`.

Rule tokens and rule-only holes are never merged.

The exact upstream DreamOn collator behavior must first be reproduced in an
isolated test before applying it to line-level roles.

### 7.5 Target map

The collator returns at least:

- input token IDs;
- immediate-rung target token ID per supervised mask;
- loss mask;
- mask role;
- node ID and tree path;
- local clock and band ID;
- allowed vocabulary class;
- DreamOn edit/delete weights.

Every emitted batch is checked against the single-token target law.

## 8. Loss

For local absorbing corruption with `alpha_l(t) = 1 - u_l(t)`, the MDLM-style
weight magnitude is:

```text
w_l(t) = u'_l(t) / u_l(t)
```

For a linear band:

```text
w_l(t) = 1 / ((b_l - a_l) * u_l(t))
```

within the active interval, with numerical clipping near zero. Outside the
active interval, `u'_l(t)=0`.

Because the hierarchy changes representation and sequence length, v0 describes
the total objective as a **hierarchical masked-denoising objective with local
MDLM-style weighting**. It must not be called an exact global MDLM NELBO unless
a separate derivation establishes that result.

Structural target frequency is monitored separately from leaf-token frequency.
If structural targets are drowned out, use an explicit sampling distribution or
loss normalization and report it. A global `1/u` factor does not by itself
balance rare top-level labels.

Plain-format auxiliary samples require an explicit mode indicator. Mixing direct
code targets and scaffold targets under an indistinguishable prompt/mask state
is forbidden.

## 9. Reverse decoding algorithm

```text
tree = prompt adapter + hidden root/designated output hole
open initial root/body line slots

while unresolved positions or closed enabled holes remain:
    apply deterministic rules to fixed point
    render segmented token canvas and role/source map
    logits = model(full bidirectional canvas)
    apply role-specific vocabulary support
    score eligible masks with calibrated confidence
    commit selected mask predictions

apply deterministic rules to fixed point
validate budgets and absence of special tokens
render pure Python
```

### 9.1 Vocabulary support

Training and inference use the same role-specific target support where possible.
At minimum:

- line roles cannot emit ordinary lexical tokens;
- token roles cannot emit construct labels or rule-only holes;
- clause roles follow the clause finite-state machine;
- newline-containing vocabulary entries are banned in single-line holes;
- `[expand]` is disabled when a hole or global budget is exhausted.

### 9.2 Confidence calibration

Raw maximum probability is not directly comparable between a line-level
softmax with a small legal support and a token-level softmax with a large legal
support.

V0 decoding uses one of:

- entropy normalized by `log(|V_allowed|)`;
- legal-support top-1/top-2 margin;
- separate line/token commit quotas.

The chosen score is logged. Raw confidence across heterogeneous support sizes is
not used for global ranking without calibration.

### 9.3 Correction

The primary v0 result uses monotone `C0`: committed model predictions do not
change.

Three opt-in inference policies are implemented against the same checkpoint:

- **C1 leaf remasking:** model-committed lexical cells retain confidence and
  model-call provenance. A bounded policy may re-mask the lowest-confidence
  eligible cells periodically or at provisional completion. Rule-emitted text
  is never eligible.
- **C3 confidence-gated structural expansion:** a construct/clause proposal
  below a threshold remains masked for a bounded number of calls before it may
  commit. This conservative implementation avoids introducing a visible
  provisional-token state that was absent from training, and has a forced
  release bound to prevent deadlock.
- **C2 structural subtree backtracking:** after a model-created construct or
  clause subtree completes, a bounded policy may use its mean lexical-token
  confidence to collapse the entire expansion back to the single line/clause
  mask at the same tree anchor. Stable anchor IDs retain a per-anchor
  backtrack count. Nested eligible subtrees are selected deepest-first.

Forward reachability alone does not prove that every corrected mixed context has
high training density. Correction variants are sampler experiments and report
matched NFE, cumulative model tokens, leaf-remask counts, correction rounds,
structural deferrals, and structural backtracks.

## 10. Termination and resource limits

Every decode has explicit limits:

- maximum total canvas tokens;
- maximum generated text tokens;
- maximum tree depth;
- maximum lines per body/module;
- maximum tokens per header/statement/doc;
- maximum `[expand]` applications per hole;
- maximum model calls;
- maximum rule moves between model calls;
- optional correction/backtrack budget.

At a limit:

- `[expand]` is removed from legal support;
- unresolved optional positions may delete;
- unresolved required suites become `pass`;
- an unresolved required header or leaf causes a recorded decode failure rather
  than an infinite loop.

Successful final output contains:

- no masks;
- no predicted meta labels;
- no rule-only holes;
- no edit labels.

## 11. Prompt adapters

The runtime supports two initialization modes:

1. **full-code generation**: prompt is immutable natural-language/context input;
   output begins at hidden `MODULE_BODY`;
2. **completion/infilling**: an immutable code prefix/suffix is parsed into
   locked IR fragments and generation begins at a designated body/statement
   hole with the correct indentation and ownership.

Benchmark protocols must declare which adapter is used. HumanEval-style prompts
containing an existing function signature cannot be treated as identical to
generating a fresh full module.

## 12. V0 supported grammar

The first symbolic/model pilot enables:

- module body;
- function definitions without decorators or async;
- `if` / `elif` / `else`;
- `for` / optional `else`;
- `while` / optional `else`;
- canonical single-line simple statements.

Initially disabled or normalized away:

- docstrings, until delimiter tests pass;
- class definitions;
- `try` / `except` / `else` / `finally`;
- `with`;
- `match` / `case`;
- decorators;
- async variants;
- multiline string/statement content;
- standalone comments.

The full vocabulary may reserve future labels, but disabled labels are absent
from legal support and training targets.

## 13. Required property tests

Before neural-model integration:

1. canonical parse/render round trip on every accepted corpus example;
2. segmented-token decode equals canonical rendered text;
3. every supervised mask has exactly one legal target;
4. rule-only holes are never targets;
5. body opening creates line masks, never pre-committed `[STMT]`;
6. line edit operations preserve slot ownership and indentation;
7. empty required suites become `pass`;
8. clause finite-state machines reject every illegal ordering;
9. forward state plus oracle immediate-rung predictor reaches the canonical
   clean program;
10. random rule sequences respect all budgets and terminate;
11. final oracle output contains no special tokens and passes `ast.parse`;
12. source-map/tree ownership remains correct after every length-changing move.

## 14. Initial experiments and diagnostics

Required early diagnostics:

- target counts by role, construct, depth, and local-clock bin;
- collapse frequency versus region length;
- line-label accuracy and calibration;
- oracle-scaffold upper bound;
- predicted-scaffold versus oracle-scaffold gap;
- final `ast.parse` rate;
- deep-nesting structural error rate;
- peak and cumulative canvas tokens;
- NFE, wall-clock latency, and approximate cumulative attention cost;
- number of model-predicted versus rule-emitted tokens.

For intermediate parseability, unresolved fragments are deterministically
completed before `ast.parse`, for example:

- condition/expression -> `True` or `None`;
- statement/body -> `pass`.

`plan-edit-distance` is not a primary C0 metric because monotone structure makes
it nearly zero by construction.

## 15. Distributed-training policy

Hardware target: one node with 8 × H20 96GB and full NVLink connectivity.

Topology is selected by measured useful throughput:

1. benchmark DDP when optimizer/model memory fits;
2. benchmark FSDP or ZeRO-style sharding when memory headroom or batch size
   benefits;
3. add tensor parallelism only when required by memory or demonstrated kernel
   efficiency;
4. use all eight GPUs for the final 7B runs, but do not equate allocation with
   efficiency;
5. record tokens/s, step time, memory, communication overhead, and MFU proxy.

Existing unrelated GPU processes are never terminated by the Scaffold-Coder
watchdog.

## 16. Implementation sequence

1. pure IR, canonical renderer, source maps, and rule engine;
2. oracle reverse decoder and property tests;
3. tokenizer surgery and segmented-tokenizer tests;
4. line/token corruption collator without DreamOn augmentation;
5. strict/local-body model smoke test;
6. exact DreamOn token-level reproduction;
7. line-level edit augmentation;
8. 1-GPU model smoke, then 8-GPU throughput topology benchmark;
9. soft gating and desynchronized training;
10. additional grammar and correction policies.
