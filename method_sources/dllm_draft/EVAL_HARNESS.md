# EvalPlus Harness

Pinned source:

```text
evalplus/evalplus
commit 26d6d00bb1fd0fa37f39c99d5290da67891d1c5e
```

Pinned datasets:

| Dataset | Version | Rows | MD5 |
|---|---|---:|---|
| HumanEval+ | v0.1.10 | 164 | `fe585eb4df8c88d844eeb463ea4d0302` |
| MBPP+ | v0.2.0 | 378 | `ee43ecabebf20deef4bb776a405ac5b1` |

The files are stored locally and supplied through:

```text
HUMANEVAL_OVERRIDE_PATH
MBPP_OVERRIDE_PATH
```

so evaluation does not depend on outbound network access from the GPU server.

## Oracle plumbing validation

Full-solution JSONL schema:

```json
{"task_id": "HumanEval/0", "solution": "...full Python program..."}
```

Results:

| Dataset | Base pass@1 | Plus pass@1 |
|---|---:|---:|
| HumanEval | 1.000 | 1.000 |
| MBPP | 1.000 | 0.99735 |

The one MBPP+ oracle failure is `Mbpp/255`; its dataset canonical solution passes
the original tests but fails one added plus input. This is evidence of a dataset
oracle edge case, not harness failure.

Artifacts:

```text
ops/artifacts/evalplus_oracle/humaneval_oracle_results.json
ops/artifacts/evalplus_oracle/mbpp_oracle_results.json
```

## Scaffold evaluation protocol

Generated outputs use full-solution `solution` records, not prompt-relative
completions. For HumanEval, the function-prefix prompt is embedded in a user
instruction asking for a complete Python solution. MBPP prompts are likewise
converted to full-code instructions.

Every generated record should additionally retain process metrics in a separate
sidecar:

- NFE;
- cumulative model tokens;
- min/max canvas length;
- expansions;
- final parseability;
- intermediate placeholder parse rate;
- latency and peak memory.

EvalPlus consumes only `task_id` and `solution`.

