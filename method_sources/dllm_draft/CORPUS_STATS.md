# `educational_instruct` Week-1 Structure Statistics

Dataset:

- repository: `OpenCoder-LLM/opc-sft-stage2`
- pinned revision: `7d28f40d579edd7c24402d17d0c7639f991e6f8d`
- split/file:
  `educational_instruct/train-00000-of-00001.parquet`
- rows: 118,278
- raw parquet size: approximately 53.6 MB

Analysis script: `scripts/analyze_corpus.py`  
Full machine-readable output: `ops/educational_instruct_ast_stats.json`

## Parse and v0 coverage

- `ast.parse` success: 118,278 / 118,278 = **100%**
- accepted by the initial Scaffold-Coder grammar:
  115,369 / 118,278 = **97.54%**

Primary rejection reason, counted once per rejected sample:

| Reason | Samples | Fraction |
|---|---:|---:|
| class definitions | 2,325 | 1.966% |
| `try` | 408 | 0.345% |
| decorators | 175 | 0.148% |
| `with` | 1 | 0.0008% |

No async function, async loop, `match`, or `try*` nodes were found.

This validates the reduced v0 grammar. Function + simple statement + `if` +
`for` + `while` covers nearly the entire controlled dataset. Class/try support
can be added after the core mechanism works rather than blocking the first run.

## Construct counts

Counts are AST-node occurrences, not sample counts:

| Construct | Count |
|---|---:|
| `if` | 155,637 |
| function | 134,685 |
| `for` | 101,947 |
| `while` | 27,877 |
| class | 2,552 |
| `try` | 438 |
| `with` | 2 |

The large counts for `if` and loops mean line-level structural labels will have
substantial training support. Rare construct labels need either later
introduction or targeted balancing.

## Nesting depth

The observed compound-depth histogram has a maximum of 13, but most examples
are shallow:

- depth ≤3: 76.20%
- depth ≤4: 94.12%
- depth ≤5: 98.50%
- depth ≤6: 99.60%
- depth ≤7: 99.93%

Implications:

- fixed bands should cover at least depths 0–6 directly;
- deeper examples can share an overflow band or use relative-depth scheduling;
- deep-nesting slices remain useful for evaluation despite being rare.

## Docstrings, decorators, and main guards

- samples containing at least one docstring: 3,092 = **2.614%**
- total docstring nodes: 3,209
- samples containing `if __name__ == "__main__"`: 616 = **0.521%**
- samples containing decorated definitions: 196 = **0.166%**

Decisions:

- docstrings remain disabled/stripped for the first model pilot;
- a dedicated `[MAIN]` token is not justified by the proposed ≥1% threshold;
- main guards are handled as ordinary `[IF]` or normalized away;
- decorators are filtered initially.

## Program size

Raw physical lines:

- mean: 11.09
- median: 9
- p90: 21
- p95: 26
- p99: 41
- maximum: 227

Accepted canonicalized programs:

- mean: 10.01 lines
- median: 9
- p90: 18
- p95: 22
- p99: 31
- maximum: 108

Raw characters:

- mean: 307
- median: 250
- p99: 1,159
- maximum: 7,340

Accepted canonicalized characters:

- mean: 285
- median: 241
- p99: 975
- maximum: 5,737

## Dream-tokenizer lengths

Measured with the pinned Dream-Coder tokenizer and the model chat template on
the 115,369 v0-accepted rows:

| Quantity | Mean | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| raw code tokens, all rows | 87.2 | 70 | 167 | 213 | 328 | 5,447 |
| normalized code tokens | 82.7 | 69 | 155 | 192 | 290 | 5,416 |
| chat-formatted prompt tokens | 53.6 | 45 | 83 | 100 | 148 | 872 |
| prompt + normalized code + EOS | 137.3 | 122 | 224 | 269 | 380 | 5,478 |
| total recursive IR line nodes | 9.16 | 8 | 16 | 20 | 28 | 86 |
| top-level line nodes | 1.31 | 1 | 2 | 3 | 5 | 40 |

Prompt + normalized-code context exceedance:

| Threshold | Samples | Fraction of v0 rows |
|---:|---:|---:|
| 512 | 206 | 0.1786% |
| 1,024 | 6 | 0.0052% |
| 2,048 | 1 | 0.00087% |
| 4,000 | 1 | 0.00087% |
| 8,192 | 0 | 0% |

Consequences:

- Stage-1 SFT does not need a RoPE/context extension.
- A 1,024-token training length reproduces DreamOn while retaining essentially
  all normalized educational samples except a handful of outliers.
- Root initialization with one or two line slots is well matched: 90% of
  accepted samples have at most two top-level lines.
- Context pressure remains relevant for LiveCodeBench prompts, but it is not a
  meaningful risk for the controlled `educational_instruct` training stage.

The dataset is short enough that the structural runtime and state distribution,
not long-context capacity, are the immediate bottlenecks.

## Parser bug found during the sweep

An early parser guard classified any rendered statement beginning with
`"match "` as a compound `match` statement. This falsely rejected assignments
such as:

```python
match = re.fullmatch(pattern, string)
```

The check was removed in favor of AST-node-type dispatch, increasing v0 coverage
from 97.48% to 97.54%. A regression test was added.
