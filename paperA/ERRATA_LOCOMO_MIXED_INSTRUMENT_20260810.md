# ERRATA — the HCache±LoRA "LoCoMo Judge" column blended two instruments

- **Date**: 2026-08-10
- **Severity**: published-artifact numeric correction (two tables in `paperA/sections/`)
- **Affects**: `paperA/sections/tab_hcache_lora.tex`, `paperA/sections/tab_adapter_hcache_combined.tex`
- **Does NOT affect**: `paperA/sections/tab_locomo.tex`, `paperA/sections/08_statistics_appendix.tex`
  (Table~\ref{tab:locomo} already reports Judge and Judge$_{1:4}$ as **two separate
  columns** and the appendix already discloses the denominator split — those are correct
  as written and are unchanged.)

## 1. What the tables said, and what they now say

| Table | Row | OLD (retracted) | NEW (correct) |
|---|---|---:|---:|
| `tab_hcache_lora.tex` / `tab_adapter_hcache_combined.tex` | HCache $j{=}12$, LoRA **no** | **13.29** | **16.69** |
| same | HCache $j{=}12$, LoRA **yes** | **31.17** | **39.81** |
| same | difference (the ablation's claim) | +17.88 | **+23.12** |

The column header also changed from `LoCoMo Judge` / `LoCoMo` to
`LoCoMo Judge$_{1:4}$`, because the header was itself part of the defect: it
named one instrument while the cell held two.

## 2. The defect

LoCoMo scoring in `scripts/eval_qcmem_locomo.py` uses **two different graders**,
selected per item by category:

- **Categories 1–4 (1,540 items, "answerable")** are graded by the **GPT-4o LLM
  judge** (`llm_judge_preds`, an API call whose verdict is cached in
  `judge_cache.jsonl`).
- **Category 5 (446 items, "adversarial")** is **never sent to the judge**. It is
  graded locally by a regular expression: `eval_qcmem_locomo.py:687-690` short-circuits
  `is_abstention` items and sets `item["judge"] = 1.0 if refused else 0.0`, where
  `refused` is `bool(_REFUSAL_RE.search(pred)) or pred.strip() == ""`.

`scores.json`'s `overall_judge` then averages **both** populations into one
number (`run_scoring`, lines 434-435 append to a single `overall["judge"]` list).
So `overall_judge` is a *weighted blend of an LLM judge and a refusal regex*, with
the regex carrying 446/1986 = 22.5% of the weight.

Published as a single column labelled "LoCoMo Judge" in a table whose entire
purpose is a **single-variable** LoRA on/off ablation, this silently changes the
measuring instrument between the header and the data, and it dilutes the effect
being measured by a constant that has nothing to do with the adapter.

## 3. Verification, read off disk in this session

Both arms' `judge_cache.jsonl` contain **exactly 1,540 records, 0 duplicates**,
and joining against `preds_shard*.jsonl` shows the cache covers categories
1/2/3/4 at **282/321/96/841** and **category 5 at 0 records** — direct confirmation
that cat-5 never reached the judge. All 1,540 records in both arms carry
`"model": "gpt-4o"`. The two caches share **all 1,540 ids** (perfect pairing).

Judge-correct counts from the caches:

```
noLoRA  257 / 1540 = 16.6883 %   -> 16.69
LoRA    613 / 1540 = 39.8052 %   -> 39.81
```

Reconstructing the published numbers from the two instruments confirms the blend
exactly, to all printed digits:

```
noLoRA  (257 judge-correct + 7 regex-correct) / 1986 = 13.29305...  == scores.json overall_judge 13.293051359516618
LoRA    (613 judge-correct + 6 regex-correct) / 1986 = 31.16817...  == scores.json overall_judge 31.168177240684795
```

The 7 and 6 are the cat-5 locally-scored correct items, recovered from
`scores.json` `by_category["5"]` (`n`=446, judge=1.5695067264573992% and
1.345291479820628% respectively). Note that the cat-5 contribution moves the
*wrong way* for the treatment arm (7 → 6), so the blend does not merely shrink the
effect, it adds a term with the opposite sign.

## 4. Why the new number is the right single-instrument choice

Two single-instrument options existed. We take **Judge$_{1:4}$ on n=1,540**:

- It is the instrument the table's own header claimed ("Judge").
- It is **already a published, named column in this paper** —
  `tab_locomo.tex` prints `Judge$_{1:4}$` for all six methods and the statistics
  appendix already defines it as "GPT-4o accuracy on the 1,540 answerable items".
  So the corrected cells are directly comparable to an existing paperA column
  rather than introducing a new convention.
- Both arms are graded from the **same 1,540 ids**, so the ablation becomes
  exactly paired, which is what a single-variable control wants.

The rejected option was "report cat-5 refusal rate as its own column." It is not
wrong, but the adapter is not a refusal intervention and the cat-5 cell moves by
1 item (7 vs 6) — reporting it would add a column with no inferential content.
Anyone who wants the full-set blended figure can still read it from
`scores.json`'s `overall_judge`; we simply no longer print a blend under a
single-instrument header.

**Disclosure of when this choice was made**: the Judge$_{1:4}$-on-1,540 convention
was **pre-existing** in this paper (it is already in `tab_locomo.tex` and the
statistics appendix, both written before this errata). The *decision to apply it
to these two tables* was made **after** seeing that the blend understated the
effect (+17.88 → +23.12). We disclose that direction explicitly: the correction
moves the number in the direction that flatters the paper's claim. That is a
reason for a reader to scrutinise it, and it is why the mechanism above is written
out arithmetically and reproducibly rather than asserted.

## 5. Paired statistics on the corrected instrument

Because the caches are id-identical, the contrast admits exact paired inference,
which the blended column could not support:

- discordant pairs: **414** items LoRA-correct/noLoRA-wrong, **58** the reverse
- McNemar, continuity-corrected $\chi^2 = 267.00$; exact two-sided
  $p = 2.6\times10^{-67}$
- paired per-item bootstrap of the difference (10,000 resamples, seed 1):
  **+23.12**, 95% CI **[20.58, 25.58]**

Caveat retained from the paper's own LoCoMo protocol discussion: LoCoMo items are
nested in only 10 conversations, so the per-item interval is not a
dependence-aware interval. It is reported here for consistency with
`08_statistics_appendix.tex`, which makes the same caveat for the
CoMem-vs-KV-Direct comparison.

## 6. Provenance

- `locomo_results/hcache_j12_noLoRA_chatFALSE/{judge_cache.jsonl,scores.json,preds_shard{0,1,2}of3.jsonl,eval_config_shard0of3.json}`
- `locomo_results/hcache_j12_LoRA_chatFALSE/{judge_cache.jsonl,scores.json,preds_shard{0,1,2}of3.jsonl,eval_config_shard0of3.json}`
- Grading code: `scripts/eval_qcmem_locomo.py` — cat-5 short-circuit at lines
  687-690; single-list blend at lines 434-435 and 453-455.
- The two arms differ only in `lora_adapter` (`""` vs
  `outputs/qcmem_distill_qwen_j12_r32_4k/final`) and
  `force_lora_with_baseline` (`false` vs `true`); all other eval_config fields are
  identical, including `use_chat_template: false`, `selector: bm25`, `topk: 12`,
  `no_retrieval: true`, `chunk_size: 512`, `max_new_tokens: 48`. Verified field by
  field this session.

## 7. Not fixed here, deliberately

- **Frozen snapshots keep the old numbers on purpose.**
  `paperA/review_history/v*_source/`, `paperA/venue_versions/*/source/`,
  `paperA/submission_source/`, and `paperA/submission_source_v6/` are records of
  what was actually submitted at each date. Rewriting them would destroy the
  audit trail. They still contain 13.29/31.17 and that is correct *as history*;
  this errata is the pointer that supersedes them. `paperA/main.pdf` /
  `final.pdf` likewise still render the old numbers until recompiled.
- **`status/PAPERA_RESULTS_CONSOLIDATED.md:383-388`** and
  **`proposal/backlog/B06-portable-decompression-adapter/PROPOSAL.md:11-12`**
  quote 13.29/31.17. They are downstream consumers, not the published table; a
  banner has been added to each pointing here.
- **The separately-known 8.11-vs-13.29 cross-node drift** on the noLoRA arm
  (recorded in `status/PAPERA_RESULTS_CONSOLIDATED.md` and B06's next-step list)
  is a *different* defect — same-instrument disagreement between the local
  measurement and the canonical zwfy6 HCache headline. It is untouched by this
  errata and still open. Note that it was measured against the blended 13.29, so
  it needs re-expressing on the Judge$_{1:4}$ scale before it can be compared.

## 8. What stayed unchanged

No prediction was re-generated, no model was re-run, and no judge call was
re-issued. Both `judge_cache.jsonl` files, both `scores.json` files, and all six
`preds_shard*.jsonl` files are byte-untouched. Only the two `.tex` cells, their
column headers, and their captions changed. Zero GPU was used.
