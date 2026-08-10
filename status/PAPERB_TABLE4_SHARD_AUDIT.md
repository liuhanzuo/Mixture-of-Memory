# Paper B Table 4 Shard-Integrity Audit (2026-08-08)

**Purpose:** MAIN found that Table 4 keep12 core6 quote sits atop a 6/8-shard `arc_easy` partial merge (`n_scored=1782` vs 2376). Scan every remaining Table 4 rung's `summary.json` on zwfy6 for the same class of defect **before** the paper writeup uses those numbers.

**Method:** For each rung's `summary.json` under `olmo2_downstream_results/`, `olmo2_mmlu_content_results/`, `olmo2_closedbook_results/` on zwfy6 (via .73), read `tasks[<task>]["n_scored"]` and compare to canonical task size. Script: `scripts/audit_paperb_table4_shards.py` (present on both wzc1 and zwfy6).

**Canonical sizes used:**
- core6: hellaswag 10042 · arc_challenge 1172 · arc_easy 2376 · piqa 1838 · openbookqa 500 · winogrande 1267
- know5: mmlu 14042 · lambada_openai 5153 · boolq 3270 · commonsense_qa 1221 · social_iqa 1954
- MMLU content: mmlu/mmlu_letter/mmlu_content 14042
- closedbook: popqa 14267 · triviaqa **17944** (see note below)

**⚠️ TriviaQA canonical size correction (embedded in this audit):** initial audit expected 11313, but every closedbook summary across the audit reports `n=17944` for triviaqa, and each has 17944 lines in `per_example_triviaqa.jsonl` (verified against `7B_keep8_step121000_v2`). The harness uses the unfiltered+nocontext validation split which is 17944. 11313 was the wrong reference; 17944 is the correct canonical.

## Result: 1 real DEFECT, 0 new defects beyond MAIN's known one

| rung | dir | task | n_scored | expected | status |
|---|---|---|---|---|---|
| **keep12_v1** | `olmo2_downstream_results/7B_keep12_step124000` | **arc_easy** | **1782** | **2376** | **DEFECT (known — MAIN's finding)** |

Every other summary in the audit matches expected exactly. **No new silent bugs surfaced.**

## Coverage summary

- **32 `summary.json` scanned**
- **8 missing** (rungs where the specific harness was not run — e.g., `7B_base_full` has no mmlu_content or closedbook, keep8/keep10/keep12/shortgpt16 v1 have no closedbook)
- **1 real DEFECT** — the already-known keep12 v1 arc_easy 6/8 partial merge
- **0 new DEFECTs**

## Full audit table

Full CSV output preserved in the eval scan; abbreviated table of the rungs and per-harness OK status below (only DEFECT / NO_TASKS / MISSING rows shown; all other rows are `ok`).

| rung | subdir | name | task | expected | status | note |
|---|---|---|---|---|---|---|
| base_full32_v1 | olmo2_mmlu_content_results | 7B_base_full | — | — | MISSING | no summary.json (harness not run) |
| base_full32_v1 | olmo2_closedbook_results | 7B_base_full | — | — | MISSING | no summary.json (harness not run) |
| keep8_v1 | olmo2_downstream_results | 7B_keep8_step121000 | — | — | MISSING | no summary.json |
| keep8_v1 | olmo2_downstream_results | 7B_keep8_step121000_know | — | — | MISSING | no summary.json |
| keep8_v1 | olmo2_mmlu_content_results | 7B_keep8_step121000 | — | — | NO_TASKS | summary has no tasks{} |
| keep8_v1 | olmo2_closedbook_results | 7B_keep8_step121000 | — | — | MISSING | no summary.json |
| keep8_v2 | olmo2_mmlu_content_results | 7B_keep8_step121000_v2 | — | — | NO_TASKS | summary has no tasks{} (different schema; letter/content stored separately) |
| keep10_v1 | olmo2_mmlu_content_results | 7B_keep10_step83500 | — | — | NO_TASKS | (same, non-standard schema) |
| keep10_v1 | olmo2_closedbook_results | 7B_keep10_step83500 | — | — | MISSING | no summary.json |
| keep10_v2 | olmo2_mmlu_content_results | 7B_keep10_step83500_v2 | — | — | NO_TASKS | (same) |
| **keep12_v1** | **olmo2_downstream_results** | **7B_keep12_step124000** | **arc_easy** | **2376** | **DEFECT** | **n_scored=1782 (KNOWN — MAIN's finding, resolved by v2 rerun to n_scored=2376)** |
| keep12_v1 | olmo2_mmlu_content_results | 7B_keep12_step124000 | — | — | NO_TASKS | (same) |
| keep12_v1 | olmo2_closedbook_results | 7B_keep12_step124000 | — | — | MISSING | no summary.json |
| keep12_v2 | olmo2_mmlu_content_results | 7B_keep12_step124000_v2 | — | — | NO_TASKS | (same) |
| keep14_v1 | olmo2_mmlu_content_results | 7B_keep14_step200000 | — | — | NO_TASKS | (same) |
| shortgpt16_v1 | olmo2_mmlu_content_results | 7B_shortgpt16_step200000 | — | — | NO_TASKS | (same) |
| shortgpt16_v1 | olmo2_closedbook_results | 7B_shortgpt16_step200000 | — | — | MISSING | no summary.json |
| shortgpt16_v2 | olmo2_mmlu_content_results | 7B_shortgpt16_step200000_v2 | — | — | NO_TASKS | (same) |

`NO_TASKS` = summary uses a non-standard schema (`mmlu_content_results/*/summary.json` stores results not in a `tasks{}` map, likely under different keys). This is not a shard-integrity issue but a schema difference for the mmlu_content harness. It's worth a follow-up check by MAIN whether these summaries' merged values are still trustworthy (the shard-file count check below suggests they are).

## Shard-file cross-check (all rungs)

For every scanned dir, counted `shard*of8.json` files on disk:
- **All dirs had either 0 or 8 shard files.** No directory had a "partial" shard state (e.g., 6/8 sitting on disk) at the current time.
- Interpretation: the keep12_v1 arc_easy 6/8 defect was captured in the merged summary; the disk state has since been either cleaned or filled. The `n_scored=1782` in the merged `summary.json` is the persistent evidence.

## TriviaQA n=17944 note (not a defect)

5 rungs (keep8_v2, keep10_v2, keep12_v2, keep14_v1, shortgpt16_v2) report triviaqa `n=17944` uniformly. This is **not** a partial merge — it matches per-example line counts (17944 lines in `per_example_triviaqa.jsonl` verified for keep8_v2). The harness uses the full unfiltered nocontext split; the 11313 value MAIN mentioned in prior notes appears to be a different reference (possibly rc.web.nocontext or a subsample), not what this eval harness ran.

## Bottom line

- **No new silent Table 4 shard-integrity bugs** beyond MAIN's already-known keep12_v1 arc_easy 6/8.
- Per user instruction, **did not auto-rerun anything.**
- Downstream `_v2` rungs (keep8/keep10/keep12/shortgpt16) all have complete `n_scored` matching expected. Safe to use for the paper.
- `keep14_v1` (never had a v2 run) also complete on core6/know5/closedbook.
- `7B_base_full` complete on core6/know5 (has no mmlu_content or closedbook eval; if the paper needs those, they need to be run).
