# Sources

## The metric under test (read these first — the claim is about these exact lines)

- `../../../third_party/babilong-pkg/babilong/metrics.py`
  - `preprocess_output` (line 24): `output.split('.')[0]` — **the first-period
    truncation that is the subject of the claim**
  - `compare_answers` (line 35): requires the target to be the **only** surviving
    task label (`target in labels_in_output and len(labels_in_output) == 1`)
  - `TASK_LABELS` (line 1): the 6-room / 7-entity label vocabularies that make
    chance inflation possible for multiple-choice outputs

## Evidence produced by A02 (both disks, md5-identical)

- `../A02-comem-write-read-repair/evidence/babilong_misorder/a02_babilong_misorder.json`
  — per-cell McNemar exact + paired bootstrap + ladder Spearman + Holm + the
  retrieval-conditional contrast that refuted retrieval domination
- `../A02-comem-write-read-repair/evidence/babilong_misorder/a02_babilong_format_diagnosis.json`
  — list-format / truncation-kill / no-label / multi-label rates per (cell, arm)
- `../A02-comem-write-read-repair/evidence/babilong_misorder/a02_format_mechanism.json`
  — per-item paired statistics on the format asymmetry + the cell-level dissociation
- `../A02-comem-write-read-repair/evidence/babilong_misorder/a02_truncation_ablation.json`
  — **the load-bearing artefact**: the one-operation ablation, all 100 items per cell,
    no conditioning, hence no collider risk

## Analyzers (import the canonical scorer, never reimplement it)

- `../A02-comem-write-read-repair/code/analyze_a02_babilong_misorder.py`
- `../A02-comem-write-read-repair/code/diagnose_a02_babilong_format.py`
- `../A02-comem-write-read-repair/code/analyze_a02_format_mechanism.py`
- `../A02-comem-write-read-repair/code/analyze_a02_truncation_ablation.py`

## Context documents

- `../A02-comem-write-read-repair/A02_BABILONG_MISORDER_PREREG.md`
  — the pre-registration, including the falsification conditions F1–F4 and the
    ownership criteria this proposal was created under
- `../A02-comem-write-read-repair/A02_BABILONG_MISORDER_VERDICT.md` §1
  — the adjudication; §1.5 is the ownership argument
- `../A02-comem-write-read-repair/A02_READ_TAX_RULER_VERDICT.md` §4
  — where the observation was first flagged (and stated more strongly than the
    statistics support; narrowed by the verdict above)
- `../A02-comem-write-read-repair/A02_DEPTH_VS_RETRIEVAL_VERDICT.md` §4/§5
  — the recall@12 measurements and the HIT/MISS labels used for the conditional test

## Boundary with B04 (do not merge — checked, not assumed)

- `../B04-eval-fragility-incubator/PROPOSAL.md`, `STATUS.json`, `NOVELTY_CHECK.md`
  — B04 = per-item `acc_norm` margin compression under damage, likelihood-ranking,
    `NARROWED_TO_OLMO_2_ONLY`. Disjoint construct and mechanism from B11.
- `../../../.claude/projects/-apdcephfs-wzc1-share-304376610-pighzliu-code-Mixture-of-Memory/memory/direction-a-eval-fragility-established.md`
  — records B04's narrowing; consulted to confirm B11 is **not** a cross-family
    extension of B04's claim

## Literature to check for K1 (novelty gate — NOT yet done, blocks all GPU)

Search targets, not citations: `lm-evaluation-harness` answer-extraction and
`exact_match` filter implementations; MCQ answer-parsing robustness; the
"LLM-as-judge vs string match" comparisons; format-sensitivity papers
(e.g. delimiter / prompt-format fragility) — note the last of these is *adjacent*
(prompt-side) whereas B11 is *scorer-side*, and that distinction is the whole
novelty question.
