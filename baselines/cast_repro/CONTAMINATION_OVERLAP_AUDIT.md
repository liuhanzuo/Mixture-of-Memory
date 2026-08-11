---
scope: CPU-only text-overlap / contamination audit of the SparseForge-5B training corpus (`qa_format_sft_llama`) against the CAST-7 evaluation test sets
date: 2026-08-11
status: FINDING — contamination does NOT explain the openbookqa/arc gains; the MC-QA-format-adaptation reading survives
supersedes: `overlap_audit.py` (v1) and `_audit_work/overlap_results.json`, which contained two verified metric bugs (see §2)
follows: `AST_VS_SPARSEFORGE_DATA_CONFOUND.md` (commit `e44c742`) §4 action item 4
---

# Verdict up front

**Contamination does not explain the +5.20 / +4.35 / +3.03 pattern on openbookqa /
arc_challenge / arc_easy.** After exhaustive manual adjudication of every flagged
item, the answer-bearing contamination rate is **1/500 = 0.20 %** on openbookqa,
**9/2376 = 0.379 %** on arc_easy, and **0/1172 = 0.00 %** on arc_challenge. Even
under the maximally uncharitable assumption that *every* contaminated item flips
from wrong to right and contributes its full weight, the arithmetic ceiling on the
contamination contribution is **0.20 pp of +5.20 (3.8 %)**, **0.379 pp of +3.03
(12.5 %)**, and **0.00 pp of +4.35 (0 %)**. The gains are ~87–100 % unexplained by
overlap.

The method is calibrated: it **flags the known-contaminated positive control
(`race`) at 9.57 % answer-bearing, 43.2 % exact-question-match, 48.3 % 8-gram** —
25–113× the arc/obqa rates — and returns the three negative controls
(`hellaswag`, `piqa`, `winogrande`) at **exactly 0.000 % answer-bearing**. So the
instrument both fires when it should and stays silent when it should.

**Consequence: the MC-QA-format-adaptation reading of the per-task pattern is the
surviving explanation**, which is what `AST_VS_SPARSEFORGE_DATA_CONFOUND.md` §2
argued on structural grounds. That is a stronger and cleaner claim than
"contaminated", and it is the one the data supports.

---

# 1. Method

## 1.1 Corpus provenance — route (a), raw text, fidelity-gated

I did **not** detokenize `train.bin`. I reconstructed the raw text by re-running the
actual builder's own formatter functions, imported at runtime (not copied, so they
cannot drift from the real builder) from

    /apdcephfs_wzc1/share_304376610/pighzliu_code/scripts/prepare_benchmark_sft.py

over the same 8 HF train splits from the same local offline cache
(`data/hf_datasets`). That script is the one that wrote `metadata.json`: its
`argparse` defaults are `--repeat 3`, `--val_ratio 0.005`, `--seed 42`, matching the
manifest exactly.

* builder: `baselines/cast_repro/build_train_corpus_for_audit.py`
* output: `_audit_work/train_corpus.jsonl`, **182,208 unique docs**, one JSON line
  per doc `{bench, doc, q, a}`
* 182,208 == the exact sum of `metadata.json` `per_benchmark_stats.original`
  (9741 + 33410 + 25262 + 11679 + 25421 + 62445 + 8134 + 6116)

**Fidelity gate** (`_audit_work/verify_fidelity.py` → `_audit_work/fidelity_check.json`).
Replaying the builder's exact tokenization (`add_special_tokens=False`, truncate ids
to 2048, append EOS, concatenate) with the same tokenizer
(`models/Llama--Llama2-7b`) over those 182,208 docs gives

| quantity | value |
|---|---:|
| reconstructed unique-corpus tokens | 43,464,735 |
| on-disk `train.bin` + `val.bin` | 130,394,205 |
| on-disk / 3 | 43,464,735.0 |
| **delta** | **+0.0** |
| **relative error** | **0.000e+00** |

The reconstruction is the **byte-exact preimage** of the tokenized corpus. This also
proves `repeat: 3` merely triplicates byte-identical strings, so it is irrelevant to
overlap detection and the unique corpus is indexed once. (The `~144.7×` epoch wrap is
likewise irrelevant to *whether* text overlaps — it only affects how strongly
whatever overlap exists is memorized.)

## 1.2 Normalization

`norm(s)`: lowercase → every character not in `[a-z0-9]` becomes a space → collapse
whitespace runs → strip. All matching is on the resulting whole-word sequence.

Word-boundary containment is implemented by storing each doc space-padded
(`" " + text + " "`) and testing `" " + needle + " "` as a substring — a C-level
search that **cannot** match a word-internal fragment. This is the fix for v1's
bug 1 (§2).

## 1.3 n-gram size

**13-gram is the primary threshold**, following the GPT-3 / PaLM decontamination
convention (an eval item is "dirty" if it shares ≥1 13-gram with training). **8-gram
is reported as a deliberately more sensitive, higher-false-positive lower bound.**
PaLM's "≥70 % of the item's 8-grams collide" rule is reported as `frac8_ge70`.

Items with fewer than *n* words **cannot** have an *n*-gram; they are counted in
`n_too_short_{8,13}` and **every n-gram rate is reported over the eligible subset**.
This matters a great deal here and is a trap: MC-QA question stems are short, so for
openbookqa only **123/500** items are even 13-gram-eligible (377 too short) and for
piqa only **126/1838**. Quoting a bare "0 % 13-gram overlap" without the denominator
would be misleading, so both are always given.

## 1.4 Query definition — what lm_eval 0.4.8 actually conditions on

Read off the vendored task yamls at
`dllm_draft/vendor/Dream-Coder/base/lm_eval/tasks/{arc,race,piqa,hellaswag,winogrande}/`:

| task | query | source |
|---|---|---|
| openbookqa | `question_stem` | `allenai/openbookqa` main/test |
| arc_easy / arc_challenge | `question` | `allenai/ai2_arc` ARC-Easy/ARC-Challenge test |
| race | `last_problem(doc)["question"]` | `EleutherAI/race` **name=high**, test |
| piqa | `goal` (`doc_to_decontamination_query: goal`) | `baber/piqa` validation |
| hellaswag | the `query` `utils.process_docs` builds: `preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())` | `Rowan/hellaswag` validation |
| winogrande | `sentence` (`doc_to_decontamination_query: sentence`) | `allenai/winogrande` winogrande_xl validation |

Notes on two non-obvious choices:
* The arc yamls' `doc_to_decontamination_query` is `"Question: {{question}}\nAnswer:"`.
  That scaffold is **constant across every item**, so including it would manufacture a
  guaranteed collision with every `Question:`/`Answer:`-formatted training doc. It is
  dropped; only the question is used.
* `preprocess_race.py` scores **only the last problem** of each article. The article
  itself is audited **separately** as `race_article`, because the article is what the
  model conditions on and is the plausible locus of passage-level reuse.

Gold answer = the choice text at the gold index per each task's `doc_to_choice`.

**All test sets were obtained offline** from the existing local cache
`/apdcephfs_wzc1/share_304376610/pighzliu_code/data/hf_datasets` — no network was
needed. Every item count is **asserted in code** against the `n-samples` block of a
real lm_eval run
(`outputs/cast_eval_spec_union9/ast_official/lm_eval_out/.../results_2026-08-11T12-03-48.712100.json`,
`git_hash b86c479`), and the assertion passes: 500 / 2376 / 1172 / 1045 / 1838 /
10042 / 1267. So the audited populations are exactly the scored populations.

## 1.5 Metrics

| id | metric | definition |
|---|---|---|
| M1 | `exact_field` | `norm(test query)` equals `norm(training doc's question field)` for some doc |
| M1b | `verbatim_sub` | `norm(test query)` occurs as a contiguous **whole-word** run inside some normalized training doc |
| M2 | `any13` / `any8` | ≥1 shared 13-gram / 8-gram (rates over the eligible subset) |
| | `frac8_ge70` | ≥70 % of the item's 8-grams collide (PaLM rule) |
| | `maxrun` | per-item longest contiguous shared run, bucketed {0, ≥5, ≥8, ≥13} |
| **M3** | **`ans_bear_strict`** | **exists a training doc D sharing ≥1 8-gram with the question AND containing the gold answer at word boundaries.** The decision-relevant form: question verbatim-recognizable *and* answer present. |
| | `ans_bear_loose` | same but D need only share a 5-gram (topical co-occurrence; upper bound, many false positives) |
| M4 | `cov_ge90` | contiguity-free paraphrase probe: some candidate doc contains ≥90 % of the query's distinct content-word types (stopword-filtered, len>2) |

**Locality window for M3.** "Question and answer near each other in training" is
operationalized as **same training doc**, which is exactly right here: every training
doc *is* one QA item (longest doc = **1,221** normalized words, ≤2048 tokens by
construction). There is no cross-doc window to worry about.

**Candidate generation** (bounded and deterministic). Phase A builds postings lists
for exactly those 5/8/13-grams occurring in some eval query, via a doc-major parallel
scan. Phase B per item unions the postings of the item's 8-grams (`cand8`, used when
the item has ≥8 words) or 5-grams (`cand5`, fallback / loose variant), each truncated
to `CAND_CAP=20000` docs by sorted doc id. **`cap_bound = 0` for every task**, so the
cap never actually bound and no result depends on it.

## 1.6 Reproduce

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/baselines/cast_repro
PY=/opt/conda/envs/torch-base/bin/python
$PY build_train_corpus_for_audit.py      # -> _audit_work/train_corpus.jsonl (182,208 docs)
$PY _audit_work/verify_fidelity.py       # fidelity gate: delta must be 0
$PY overlap_audit_v4.py                  # -> _audit_work/overlap_results_v4.json  (~13 s, 48 proc)
$PY adjudicate_strict_hits.py            # -> _audit_work/strict_hits_adjudication.json
$PY verify_and_bound.py                  # -> _audit_work/contribution_bound.json
```

Runtime is 13 s wall for the main audit on 48 forked workers; the whole audit is
CPU-only and used no GPU. One implementation gotcha worth recording: **Python 3.14
changed the default multiprocessing start method to `forkserver`**, under which
module globals are *not* inherited and the corpus arrives empty in workers
(`IndexError`). `overlap_audit_v4.py` therefore pins `mp.get_context('fork')` so the
31 M-word corpus is shared copy-on-write instead of being pickled 48× (which is what
made an earlier draft appear to hang at 1 % CPU per worker).

---

# 2. Two bugs in the previous pass, both verified

An earlier pass left `overlap_audit.py` (v1) and `_audit_work/overlap_results.json`
on disk with no write-up. Its numbers should **not** be used; two defects were found
by inspection and then confirmed against v1's own emitted output.

**Bug 1 — false positives in `answer_bearing` (raw substring instead of word
boundary).** v1 tested `norm(answer) in norm(doc)` as a raw substring. Verified on
v1's own reported arc_challenge example: v1 claimed the gold answer **`"ice"`** was
present in a `race_middle` doc, but that doc contains **no word "ice"** — it contains
**`"icebergs"`**:

```
raw substring "ice" present:            True
word-boundary " ice " present:          False
words containing substring "ice":       ['icebergs']
```

Equally, the answer `"6"` would match inside `"1968"`. v1's `answer_bearing` was
therefore inflated. v4 requires whole-word alignment.

**Bug 2 — order-dependent truncation.** v1 did `if len(cand) > 4000: break` while
iterating a Python `set`, so *which* 4000 docs got checked depended on set iteration
order and the run was not reproducible. v4 truncates deterministically (sorted doc
id) and reports `cap_bound`, which is 0 everywhere.

A third, softer issue: v1's `hellaswag` query used the raw `ctx` field, whereas
lm_eval conditions on `activity_label + ": " + ctx_a + " " + ctx_b.capitalize()` run
through `utils.preprocess`. v4 uses the real `query`.

---

# 3. Results

`_audit_work/overlap_results_v4.json`. Positive control first, negative controls
last; the three suspicious tasks in the middle. Rates are % of `n_test` except the
n-gram columns, which are % of the **eligible** subset (denominator shown).

| task | n_test | exact Q match | verbatim sub | ≥1 13-gram (of elig) | ≥1 8-gram (of elig) | PaLM 70 %/8 | **answer-bearing (strict)** | ans-bear (loose) | cov≥90 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`race` — POSITIVE CTRL** | 1045 | **452 (43.25 %)** | **456 (43.64 %)** | **24/132 (18.18 %)** | **358/742 (48.25 %)** | **285** | **100 (9.569 %)** | 145 (13.88 %) | **545 (52.15 %)** |
| `race_article` (passage) | 1045 | 0 | 4 (0.38 %) | **277/1045 (26.51 %)** | 368/1045 (35.22 %) | 135 | 0 | 0 | 152 (14.55 %) |
| `openbookqa` — suspicious | 500 | 4 (0.80 %) | 3 (0.60 %) | **0/123 (0.00 %)** | 3/295 (1.02 %) | 1 | **1 (0.200 %)** | 4 (0.80 %) | 6 (1.20 %) |
| `arc_easy` — suspicious | 2376 | 3 (0.13 %) | 6 (0.25 %) | **0/1564 (0.00 %)** | 48/2206 (2.18 %) | 3 | **11 (0.463 %)** | 150 (6.31 %) | 38 (1.60 %) |
| `arc_challenge` — suspicious | 1172 | 1 (0.09 %) | 1 (0.09 %) | **0/868 (0.00 %)** | 27/1108 (2.44 %) | 1 | **1 (0.085 %)** | 33 (2.82 %) | 12 (1.02 %) |
| `hellaswag` — NEG CTRL | 10042 | 0 | 0 | 0/9808 (0.00 %) | 52/10042 (0.52 %) | 0 | **0 (0.000 %)** | 0 | 0 |
| `piqa` — NEG CTRL | 1838 | 2 (0.11 %) | 2 (0.11 %) | 0/126 (0.00 %) | 0/803 (0.00 %) | 0 | **0 (0.000 %)** | 0 | 4 (0.22 %) |
| `winogrande` — NEG CTRL | 1267 | 0 | 0 | 0/1267 (0.00 %) | 1/1267 (0.08 %) | 0 | **0 (0.000 %)** | 25 (1.97 %) | 0 |

Corpus index stats (`meta`): 182,208 docs, 31,447,161 normalized words, 121,420
unique normalized question fields; query n-gram sets 5=741,776 / 8=716,848 /
13=642,931; of these, 77,338 / 50,302 / 41,127 had ≥1 corpus hit. `cap_bound = 0`
for every task.

## 3.1 The positive control fires — the method works

`race` is contaminated by construction (`race_middle` + `race_high` train splits are
in the corpus, ×3 repeat, ×~144.7 epochs) and the method flags it hard, on every
metric, at **25–113× the arc/obqa rates**:

* `exact_field` **43.25 %** vs 0.09–0.80 %
* `any8` **48.25 %** of eligible vs 1.02–2.44 %
* `frac8_ge70` **285** items vs 1–3
* `ans_bear_strict` **9.569 %** vs 0.085–0.463 %
* `cov_ge90` **52.15 %** vs 1.02–1.60 %

Attribution is correct too: race's 100 strict hits resolve to
`{race_high: 93, race_middle: 7}` — i.e. to the RACE train splits, not to noise.
**The instrument is calibrated. Had it not flagged race, nothing else here would be
reportable.**

Two important qualifications on *what kind* of contamination race has, because they
change the recommended remedy:

* **Most of race's `exact_field` is stem templating, not item leakage.** Of the 452
  matches, only **278** are distinct stems, and the top ones are RACE boilerplate
  shared by hundreds of training docs: `"what is the passage mainly about"`
  (259 training docs), `"which of the following is true according to the passage"`
  (583), `"what would be the best title for the passage"` (266). Only **75** stems
  match exactly one training doc, i.e. are candidate genuine item duplicates.
* **Passage-level reuse across the RACE train/test split is small but real:**
  **9/1045** race test articles have their first 300 normalized characters occurring
  verbatim in a RACE *train* doc (`verify_and_bound.py` §3). So RACE's own splits are
  not perfectly disjoint at the passage level, independent of anything SparseForge
  did.
* `race_article` shows **26.51 %** 13-gram overlap, by far the highest 13-gram rate of
  any row — consistent with long English passages sharing stock phrasing plus those 9
  true duplicates.

Either way, race's answer-bearing rate (9.57 %) is ~21× arc_easy's and ~112×
arc_challenge's, and its `exact_field` is 54–507× the arc/obqa rates. The
contamination signal is unmistakable where contamination is known to exist.

## 3.2 The negative controls come back clean

`hellaswag`, `piqa`, `winogrande` — the three flat/negative-delta tasks — return
**exactly 0.000 % answer-bearing** and **0.00 % 13-gram**. `hellaswag`'s 52 8-gram
hits out of 10,042 (0.52 %) and `winogrande`'s 1/1267 are floor-level noise, and
`winogrande`'s 25 loose answer-bearing hits are pure artifact (winogrande answers are
single common nouns like "table", trivially present in a 31 M-word corpus). This is
the expected clean result and it means the metric is not simply firing on everything.

## 3.3 The suspicious tasks pattern with the negative controls, not the positive one

This is the core observation. On every metric, obqa / arc_easy / arc_challenge sit
next to the clean negative controls and **orders of magnitude** away from race:

| metric | race (dirty) | obqa | arc_e | arc_c | hellaswag / piqa / wino (clean) |
|---|---:|---:|---:|---:|---:|
| exact Q match | 43.25 % | 0.80 % | 0.13 % | 0.09 % | 0 / 0.11 % / 0 |
| ≥1 13-gram | 18.18 % | **0.00 %** | **0.00 %** | **0.00 %** | 0 / 0 / 0 |
| answer-bearing | 9.569 % | 0.200 % | 0.463 % | 0.085 % | 0 / 0 / 0 |
| cov ≥90 % | 52.15 % | 1.20 % | 1.60 % | 1.02 % | 0 / 0.22 % / 0 |

**Not one single item** in openbookqa, arc_easy or arc_challenge shares even one
13-gram with the training corpus (0/123, 0/1564, 0/868 eligible). Under the standard
GPT-3/PaLM criterion these three test sets are **clean**.

## 3.4 Exhaustive manual adjudication of every strict hit

The strict counts are small enough (13 items total) to adjudicate by hand rather than
trust the automated metric — which matters, because the automated metric still
over-counts. Full dumps with the shared n-gram: `adjudicate_strict_hits.py` →
`_audit_work/strict_hits_adjudication.json`; verdicts recorded in
`_audit_work/adjudication_verdicts.json`.

Classification: **REAL** = test item's specific content duplicated and gold answer
present; **TOPICAL** = same topic/related fact, test item not reproduced; **BOILER**
= shared n-gram is question-format boilerplate and the answer word is coincidental.

| task | strict hits | REAL | TOPICAL | BOILER (false positive) |
|---|---:|---:|---:|---:|
| openbookqa | 1 | 1 | 0 | 0 |
| arc_easy | 10 | 8 | 1 | 1 |
| arc_challenge | 1 | 0 | 0 | **1** |

Two false positives confirmed by direct inspection:

* **arc_challenge's only hit is spurious.** Q: *"How many valence electrons does
  selenium have?"*, gold `6`, matched a `sciq` doc — but that doc is about
  lithium/beryllium/**boron** and asks *"how many valence electrons does boron have"*.
  I verified the doc **does not contain the word "selenium"** (only 13 docs in the
  whole corpus mention selenium; this is not one of them). The single-character gold
  answer `6` matched an unrelated numeral. **arc_challenge's true contamination count
  is 0 — and its delta is the second largest at +4.35 pp.**
* **arc_easy hit [1]** — Q: *"Which of the following is an example of a compound?"*,
  gold `water` — matched via the pure-format 8-gram
  `"which of the following is an example of"` against a `race_high` passage about
  **story settings** that merely contains the common word "water".

The genuine cases are real and worth stating plainly. They are science-QA facts that
`sciq`/`qasc` restate almost verbatim, e.g. arc_easy *"Matter is defined as anything
that takes up space and has ___"* → gold `mass`, and the `sciq` training doc is
literally `question what is defined as anything that takes up space and has mass
answer matter`. Similarly *"When a neutral atom gains or loses electrons, it becomes
___"* → `an ion`, and the sciq doc contains that sentence verbatim. Attribution of
arc_easy's 11 strict hits: `{sciq: 8, qasc: 2, race_high: 1}` — i.e. exactly the
science-QA neighbours flagged as a risk in `AST_VS_SPARSEFORGE_DATA_CONFOUND.md` §4.
**The risk was real; its magnitude is ~0.4 %.**

---

# 4. Bounding the contribution

Upper bound, deliberately maximally uncharitable: assume every contaminated item was
answered **wrong** by AST and **right** by SparseForge, i.e. each contributes its
full `1/n` weight to the delta. Counting REAL + TOPICAL (excluding only the verified
BOILER false positives):

| task | n_test | contaminated | max pp | observed Δ | **max share of Δ** |
|---|---:|---:|---:|---:|---:|
| openbookqa | 500 | 1 | 0.200 | **+5.20** | **3.8 %** |
| arc_easy | 2376 | 9 | 0.379 | **+3.03** | **12.5 %** |
| arc_challenge | 1172 | 0 | 0.000 | **+4.35** | **0.0 %** |
| winogrande | 1267 | 0 | 0.000 | +1.66 | 0.0 % |
| hellaswag / piqa | — | 0 | 0.000 | −0.06 / −0.65 | n/a |

Including the BOILER false positives too (i.e. using the raw automated
`ans_bear_strict`) only moves these to 3.85 % / 15.28 % / 1.96 %
(`_audit_work/contribution_bound.json`).

So **87.5 % of arc_easy's gain, 96.2 % of openbookqa's, and 100 % of
arc_challenge's is not attributable to text overlap.** And the bound is loose in the
conservative direction: a 7B model at ~50 % sparsity will already answer many of
these easy science facts correctly *without* memorizing them, so the true
contamination-attributable share is lower still.

Note the anti-correlation that kills the contamination hypothesis outright:
contamination ranks `arc_easy (0.379 %) > obqa (0.200 %) > arc_c (0.000 %)`, but the
gains rank `obqa (+5.20) > arc_c (+4.35) > arc_easy (+3.03)`. **The task with the
largest measured contamination has the smallest gain, and the task with zero
contamination has the second largest.** If overlap were driving the effect, these
orderings would agree. They are essentially inverted.

Separately, and consistent with `AST_VS_SPARSEFORGE_DATA_CONFOUND.md` §1: race's
contamination *is* severe (9.57 % answer-bearing, 43.25 % exact stem match), and
race's `+2.78 pp` is the one cell that should not be reported as an algorithmic
result. But race supplies only 17.0 % of the CAST-7 margin, and dropping it still
leaves +2.2544 pp.

---

# 5. What this means

The four largest gains are the four MC-QA tasks; the three flat/negative cells are
the continuation/cloze tasks. This audit removes contamination as the explanation for
three of those four. Combined with the PPL direction already on record (`SPEC.md`,
commit `501dafb`: at matched seqlen 4096 SparseForge's WikiText-2 PPL **6.2179** is
*worse* than AST-official's **5.9125**), the surviving reading is the one
`AST_VS_SPARSEFORGE_DATA_CONFOUND.md` §2 argued structurally:

> a ~144-epoch finetune on eight MC-QA datasets in exactly the answer format the
> benchmark scores, beating a web-text-trained arm on MC-QA while *degrading*
> held-out LM perplexity, is the signature of **task/format adaptation**, not of a
> better 2:4 mask.

That is a data-vs-data confound, not a leakage scandal — and it is still fatal to
attributing the CAST-7 margin to the mask-learning machinery. The remedy is unchanged
and is now better supported: **AST+SLoRB must be trained data-matched on
`qa_format_sft_llama`** (same 17,000+3,000 iters, `global_batch_size 256`,
`block_size 4096`) so that data, budget and SLoRB are held fixed and only
mask-learning differs. Only the margin that survives *that* is an algorithmic result.

Actions this audit closes and opens:
1. ✅ Action 4 of `AST_VS_SPARSEFORGE_DATA_CONFOUND.md` §6 (run the overlap audit) is
   **done**; result is a clean negative for arc/obqa.
2. `metadata.json`'s `"no benchmark overlap"` description is still false for CAST-7
   (race) and should be corrected — now with a measured rate (9.57 % answer-bearing)
   rather than just the structural argument.
3. Report CAST-6 alongside CAST-7 with race's contamination stated quantitatively.
4. The arc/obqa cells **can** be used as evidence of MC-QA format adaptation. They
   **cannot** be dismissed as contamination.
5. `overlap_audit.py` (v1) and `_audit_work/overlap_results.json` should be treated as
   superseded; their `answer_bearing` numbers are inflated by the word-boundary bug.

---

# 6. What I could NOT determine

* **Whether the ~0.4 % genuinely-overlapping arc_easy items were actually answered
  correctly by SparseForge and incorrectly by AST.** That needs per-item
  (`doc_id`-level) predictions from both arms. lm_eval was run without
  `--log_samples`, so only aggregate accuracies exist on disk. Everything in §4 is
  therefore an *upper bound*, not a measured contribution. Re-running both arms with
  `--log_samples` would convert the bound into an exact number; it needs GPU and was
  out of scope for this CPU-only audit.
* **Semantic (non-lexical) contamination.** M4 (`cov_ge90`) catches reworded
  duplication that shares content words, but a fact restated with entirely different
  vocabulary would evade every metric here. Detecting that needs embedding-space
  nearest-neighbour retrieval, which is a different (GPU) audit. Note M4 already
  agrees with the lexical metrics (obqa/arc 1.0–1.6 % vs race 52.15 %), so there is no
  evidence of a large hidden paraphrase population — but absence of evidence at this
  sensitivity is not proof.
* **Pretraining-corpus contamination.** This audit compares the *finetune* corpus to
  the test sets. LLaMA-2-7B's own pretraining data is not public and may contain ARC
  or OpenBookQA. That confound applies **equally to both arms** (both start from the
  same LLaMA-2-7B), so it cannot explain a *difference* between them, which is why it
  does not affect the verdict.
* **Whether the 9 duplicated RACE test articles inflate race's score specifically.**
  Same per-item-prediction limitation as above.
* **The other five training benchmarks' relationship to non-CAST tasks.**
  `commonsenseqa`, `social_iqa`, `cosmosqa`, `dream` were indexed and are included in
  all counts, but I did not audit them against boolq/rte (the AST-7 tasks CAST-7
  replaces), since those are not where the suspicious gains are.
