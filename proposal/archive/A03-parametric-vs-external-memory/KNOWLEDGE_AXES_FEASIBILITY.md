---
task: A03 knowledge-axes data acquisition + harness-integration feasibility
date: 2026-08-10
compute: CPU + network only on LOCAL (wzc1). No GPU touched (all 8 LOCAL GPUs were at 100% on keep14; .21/.73/.104 training and .82's Arm6 watcher untouched).
verdict: >
  All four blocked axes' data ARE obtainable — STATUS.json's "NONE on either disk" was true as a
  statement about the disk and false as a reason to stay blocked. But acquiring the data does NOT
  unblock the axes: three of the four fail on null-floor / construct grounds, and the ONE that
  survives (CounterFact conflicting-knowledge as a 2AFC) needs a floor of 0.54, not 0.50.
recommendation: >
  Do ONE axis (CounterFact 2AFC over the pooled 21919 items, floor 0.5364 held-out).
  Do NOT do MQuAKE (floor 0.177-0.235, and it measures multi-hop composition not the target axis).
  Do NOT do HotpotQA-stripped (10.2% of answers are copyable from the question; it is a
  reading-comprehension benchmark with the comprehension removed).
  zsRE is a strictly worse CounterFact — keep it only as a robustness replicate.
supersedes: STATUS.json:remaining_axes_status (the "NO_DATA" / "WRONG_FORMAT" reasons, not the reframing conclusions)
---

# A03 — the three "blocked" knowledge axes: data acquisition and feasibility

## 0. Executive summary

`STATUS.json:remaining_axes_status` blocks three axes with two reasons. This audit tested both.

| STATUS.json's reason | verdict after test |
|---|---|
| "CounterFact, MQuAKE, zsRE, KnowEdit — NONE on either disk" | **factually correct about the disk, but not a blocker.** All four downloaded in ~7 min through `hy-proxy`. 197 MB total. |
| "all require context injection (edit-then-query), not native to closed-book harness" | **wrong as stated.** CounterFact/zsRE ship `(subject, relation, target_true, target_new)` as *data*; the edit-then-query machinery is the ROME/MEMIT *harness*, which we do not need. A closed-book 2AFC or fill-in-the-blank read is native. |
| "HotpotQA/2Wiki/MuSiQue live in LongBench form (open-book)" | **wrong about availability** (the raw HF HotpotQA with a separable `context` field downloads fine, and `data/longbench_raw/data/hotpotqa.jsonl` was already on disk), **but right in spirit for a different reason than stated** — see §4. |

**So the blockers were misdiagnosed, and the axes are still mostly not worth doing — for
reasons that only became visible after measuring the data.** The load-bearing finding is §3:
a CounterFact true-vs-false 2AFC does **not** have a 50 % null.

---

## 1. What was downloaded (all numbers read off the files)

Full manifest with sha256 for every file: `data/knowledge_axes/MANIFEST.md`. Summary:

| axis | dataset | rows (measured) | on-disk | usable? |
|---|---|---:|---:|---|
| conflicting knowledge | `counterfact` test + train parquet | **2191 + 19728 = 21919** | 15 MB | **YES** (best of the four) |
| conflicting knowledge | `counterfact` KnowEdit `test_cf.json` | 839 | (incl.) | too small (MDE 4.8pp) |
| conflicting knowledge | `zsre_mend_eval.json` | **19086** | 76 MB (+train 163196) | yes, but strictly worse than CounterFact (§5) |
| conflicting knowledge | `KnowEdit ZsRE-test-all.json` | 1301 | (incl.) | too small |
| multi-evidence | `MQuAKE-CF-3k.json` (+ `-v2`, `-CF` 9218, `-T` 1868) | **3000** cases / 9000 questions | 82 MB | **NO** — floor 0.177 (§4) |
| multi-evidence | `hotpot_qa` distractor validation | **7405** | 27 MB | **NO** — 10.2 % answer-in-question (§4) |
| new injected facts | — | — | — | still blocked *by definition*, correctly (§6) |

**zsRE was found.** STATUS.json implies its canonical URLs are dead — the two I was told 404 do
404, but `https://rome.baulab.info/data/dsets/zsre_mend_{eval,train}.json` returns 200 and is the
ROME/MEND canonical mirror. There is also a de-duplicated 1301-item variant in `zjunlp/KnowEdit`
(HF, **not** gated — 200 without auth; only the *MQuAKE* HF mirror is gated, GitHub raw is open).

Nothing here pre-existed: `find` over wzc1 `data/`+`.cache/` and over zwfy6 `data/`+`.cache/`
(run from `.82`) returns only LongBench HotpotQA logs and `data/longbench_raw/data/hotpotqa.jsonl`.

---

## 2. The existing harness: what it expects, and what it cannot do

`scripts/eval_olmo2_closedbook_qa.py` (510 lines).

**Input contract.** One function, `load_task_examples(task)` (line 134), returns
`list[{"question": str, "answers": list[str]}]`. Everything downstream is task-agnostic:
`build_prompt` (line 212) hardcodes `f"Question: {question}\nAnswer:"`; sharding is
`examples_all[shard_index::num_shards]` (line 439); `item_id = shard_index + li*num_shards`
(line 469). Datasets come from `datasets.load_dataset` (HF cache), **not** from a file path —
there is no `--data_file` argument.

**Scoring** (`score_prediction`, line 113) — max over gold aliases of three metrics:
`em` (normalised string equality), `contains` (normalised gold is a substring of normalised
pred), `f1` (token-level, `Counter` intersection). `normalize_answer` (line 91) = lowercase →
strip `string.punctuation` → drop `\b(a|an|the)\b` → collapse whitespace.
Prediction = greedy `model.generate`, **first line only** (line 257).

**Baselines it computes itself**: `majority_em` — the *a-priori* most frequent normalised
first-alias gold (line 434), scored per item. Note this is weaker than the
best-constant floor the A03 analyzer computes; `analyze_1b_knowledge_floor.py` maximises over
candidate constants and explicitly documents that it "dominates the harness's own `majority_em`".

**Multi-task switch**: `--tasks popqa,triviaqa` → `tasks = [t.strip() for t in args.tasks.split(",")]`
(line 376) → a plain `for task in tasks` loop (line 422). There is **no TASK registry dict** —
task names are string-compared inside `load_task_examples` via a chain of `if task == ...`
blocks. So "register in the TASK dict" is not the shape of the change; it is "add an
`if task == "x":` branch".

### ★ The harness is generation-only. It cannot do 2AFC.

This is the single most important integration fact and it is not in STATUS.json.
`eval_olmo2_closedbook_qa.py` has no log-likelihood path — it only calls `model.generate`.
A true-vs-false discrimination needs teacher-forced continuation scoring, which lives in a
**different** harness: `scripts/eval_olmo2_mmlu_content.py`, whose `score_examples` (line 155)
sums fp32 `log_softmax` over continuation tokens and reports both raw and
length-normalised (`content_norm`) argmax.

Consequence for cost estimates: **which harness you extend depends on the interface you pick**,
and the two answers differ by an order of magnitude.

| target interface | host harness | change needed | honest size |
|---|---|---|---|
| fill-in-the-blank generative (CounterFact/zsRE/MQuAKE) | `eval_olmo2_closedbook_qa.py` | one `if task == "counterfact":` branch in `load_task_examples` reading the local parquet + **one change to `build_prompt`** (see below) | ~40 lines, genuinely small |
| **true-vs-false 2AFC** (the interface CounterFact is actually for) | `eval_olmo2_mmlu_content.py` | a loader emitting `letter_cands`/`content_cands`-shaped 2-candidate lists, **plus** `n_opt=2` handling, **plus** decoupling `_LETTERS`-keyed score dicts, **plus** a new results_root and a new null | ~150-250 lines + a new null-floor branch in `analyze_1b_knowledge_floor.py`. **Not a gate-sized change.** |

**The `build_prompt` blocker for CounterFact specifically.** CounterFact's `prompt` field is a
**cloze template with a `{}` placeholder** (`"{} is located in"`, 218 distinct templates, all
21919 items contain `{}`, 17575 are subject-initial), *not* a question. Rendering it into
`f"Question: {Angola is located in}\nAnswer:"` is malformed. A correct integration needs either
a per-task prompt-builder hook or a pre-conversion of clozes into questions — the latter changes
what is being measured and would not be comparable to the ROME literature.

---

## 3. ★ CounterFact null floor: it is **not 50 %**

The task prompt asked for a power estimate "in a 2-choice setting where the null floor is 50 %
(random guessing)". **Measured on the actual data, the input-blind floor is 54–64 %.** A03's own
iron law ("every interface must clear its own *construct-appropriate* null") makes 0.50 the wrong
number, exactly as A01 documents for MMLU-letter (where 0.25 is wrong and best-constant-D is right).

Three input-blind heuristics, all of which see **only the two candidate strings**, never the fact:

| heuristic (knowledge-free) | CF test (n=2191) | CF train (n=19728) |
|---|---:|---:|
| always pick the shorter string (split-tie) | 0.5164 | 0.5068 |
| pick the string with higher **marginal** frequency across both roles | 0.5454 | 0.5515 |
| pick the string more often seen in the **`target_true` role** (in-sample) | **0.6426** | **0.5707** |

The third is the binding one and it is not an overfitting artifact — it survives held-out fitting:

| prior fit on | applied to | 2AFC accuracy |
|---|---|---:|
| train (19728) | **test** (2191) | **0.5429** |
| test (2191) | **train** (19728) | **0.5364** |
| relation-conditional, fit train | test | 0.5203 |

**So the construct-appropriate 2AFC floor is ≈0.5364–0.5429** (use the held-out value; the
in-sample 0.6426 is the number a careless analysis would report as "chance", and using 0.50
instead would credit an arm with a spurious **+7.1 pp (train) to +14.3 pp (test)** of headroom).

*Mechanism*: CounterFact's counterfactual targets are sampled from other items' true targets, but
**not uniformly** — some strings (`Antarctica`, `French`) are heavily over-represented as
`target_new` relative to `target_true`, so candidate-set identity alone leaks the answer.

### Power, against the right floor

MDE = minimum detectable effect at α=.05 two-sided, 80 % power, one-sample binomial vs p₀.

| set | n | p₀ (measured) | MDE | min significant (1.96·SE) |
|---|---:|---:|---:|---:|
| CF test only | 2191 | 0.5429 | **2.98 pp** | 2.09 pp |
| CF train only | 19728 | 0.5364 | **0.99 pp** | 0.70 pp |
| **CF pooled test+train** | **21919** | ≈0.5396 | **0.94 pp** | 0.66 pp |
| KnowEdit `test_cf` | 839 | ~0.54 | 4.83 pp | 3.38 pp |

**Calibration against effects A03 actually sees.** Pruned+healed 1B clears its floor by
+9.33 pp on TriviaQA EM, +2.30 pp on NQ-open EM, +1.65 pp on PopQA EM; the Arm-3/Arm-6 CPT
trajectory headline is **+0.48 pp** at n=17944.

* At n=21919 the MDE is **0.94 pp**, so the *pruned-vs-floor* question (effects of 2–9 pp) is
  comfortably powered. **This axis can answer "is the pruned+healed model above floor".**
* The *CPT-trajectory* question (effects ~0.5 pp) is **NOT** powered at n=21919 as a
  one-sample-vs-floor test. It would be powered as a **paired** intact-vs-pruned McNemar over the
  same items (the analyzer already does paired bootstrap + exact McNemar), which is the test A03
  actually runs for its trajectory arms. Use the paired test, not the vs-floor test, for Arm-3-style
  deltas — and **do not** pool test+train and then quote a vs-floor p-value as if it settled a
  0.5 pp trajectory question.
* Using CF **test only** (n=2191, MDE 2.98 pp) would make even the pruned-vs-floor call marginal.
  **If this axis is done, it must be done on all 21919 items.** The test/train split is a
  model-editing convention with no meaning for a closed-book read (`case_id` sets are disjoint;
  245 subjects recur across the split, so pooling is safe for a *closed-book* read but the
  subject recurrence must be disclosed).

### Alternative interfaces for CounterFact, and their nulls

| interface | null (measured) | n | MDE | assessment |
|---|---:|---:|---:|---|
| **2AFC**, LL(true) vs LL(false), length-normalised | **0.5364** held-out role-prior | 21919 | 0.94 pp | **best**. All targets are single-word (measured: 21919/21919 have word-count 1, mean 6.7 chars), so raw vs length-normalised LL barely differ and the tie-convention swamp that plagues MMLU-content is mostly absent. |
| fill-in-the-blank generative + EM | best-constant `'french'` = **0.04886** | 21919 | 0.41 pp | viable, and it reuses the *existing* generative harness. But: no alias lists anywhere in CounterFact (single gold string), so a correct-but-differently-phrased answer scores 0 — this is why the model-editing literature scores CounterFact by LL comparison, not by EM. Expect a heavily deflated absolute number. |
| closed-set MC over the 826 distinct target strings | best-constant ≈0.049; or per-relation sets (5–232 candidates) | 21919 | ~0.4 pp | most sensitive but **most expensive** (826 continuations/item = 18 M forward passes) and no longer comparable to any published protocol. |

**Recommended**: 2AFC as headline (it is the construct the dataset was built for), with generative
EM disclosed alongside as the cheap cross-check — exactly the letter/content dual-interface
pattern A03 already uses for MMLU.

### The honest caveat about 1B

The task prompt asked me to say plainly if the 1B model will sit at floor. My read: **it will
clear the 2AFC floor, but the margin will be modest and the axis will be less informative than
TriviaQA.** Reasoning from measured numbers, not vibes:

* CounterFact facts are Wikidata head-entity relations (top relations P30 `continent`,
  P27 `citizenship`, P413 `position played`, P1412 `language spoken`) — this is *more* head-heavy
  than PopQA's long-tail entities, where the pruned+healed 1B still cleared floor by +1.65 pp.
* 2AFC is a far easier interface than free generation: the intact 1B gets 40.7 % EM on TriviaQA
  *generatively*, so a 2-way discrimination on head facts should land well above 0.54.
* **But** the ceiling is 1.0 and the floor is 0.54, so the *available headroom is only 0.46* —
  versus TriviaQA's 0.997 headroom (floor 0.0026). A03's residual-fraction statistic
  (**corrected 2026-08-10**: it is `(reported − null)/reported`, *not* `(reported − null)/(1 − null)`
  as this bullet originally stated — see the erratum at `GATE_FOURAXES_VERDICT.md` §2 and the
  `residual_fraction_of_{reported,headroom}` keys now in
  `evidence/a03_1b_floor_nulls_4axes.json`) will therefore look inflated on this axis for the same
  absolute gain. The direction of this warning is unchanged and if anything stronger under the real
  formula: dividing by `reported` inflates *low-scoring* cells, so a 0.54-floor 2AFC axis whose
  reported score is near the floor can post a large fraction off a trivial absolute gain.
  **Residual fractions must not be compared across axes with such different headroom**, and
  the report should say so or it will invite exactly the error A01 documents.
* The genuine risk is not "at floor", it is **"above floor but measuring the candidate-set prior
  rather than the fact"**. Mandatory control: run the 2AFC with the *subject masked out* of the
  cloze (or with a random-subject cloze). If accuracy barely drops, the arm is reading the
  candidate prior and the cell is void. This control does not exist in any current script.

---

## 4. MQuAKE and HotpotQA: measured reasons not to do them

### MQuAKE — killed by its own answer distribution

| unit | n | best-constant EM floor | constant | MDE |
|---|---:|---:|---|---:|
| multi-hop questions (3 paraphrases × 3000 cases) | 9000 | **0.17733** | `washington dc` | 1.14 pp |
| single-hop sub-facts | 9000 (3473 unique clozes) | **0.12744** | `united states of america` | 0.99 pp |
| multi-hop, `contains` metric | 9000 | **0.20833** | `american english` | — |

A floor of 0.177–0.235 on a generative task is pathological: it means a model that emits
"Washington, D.C." for every question beats a real 17.7 % of items. Compare the certified axes
(TriviaQA 0.0026, PopQA 0.0229, NQ-open 0.0055). The floor is high because MQuAKE composes
Wikidata relations whose ranges are tiny (`head of state of country of citizenship of X` → a few
hundred people, dominated by US/UK).

Two further disqualifiers:
1. **Only 3473 unique clozes among 9000 single-hops** — the 3 paraphrases per case and repeated
   sub-facts make items non-independent. Bootstrapping over 9000 rows would overstate precision;
   the honest n is ~3000 cases (MDE 1.98 pp).
2. **It measures the wrong thing.** MQuAKE's difficulty is *multi-hop composition*, which a
   9-layer pruned model will fail for reasons having nothing to do with where knowledge is stored.
   A03 would not be able to attribute a drop to knowledge loss vs. lost compositional depth.
   This is the real reason to drop it — the floor just makes it un-measurable as well.

**Note**: STATUS.json calls this axis "multi_evidence" and blocks it as "WRONG_FORMAT". The format
is fine (MQuAKE is closed-book-shaped: no context field at all). The correct reason is
**floor + construct confound**. Update the reason so a future agent doesn't "unblock" it on
discovering the format is fine.

### HotpotQA-stripped — killed by answer leakage

Stripping `context` is trivial (the parquet has `question` / `answer` as separate top-level
columns; ~10 lines). The problem is what remains:

| property (measured, n=7405) | value |
|---|---:|
| answers appearing **verbatim inside the question** | **756 / 7405 = 10.21 %** |
| — among `comparison` items (n=1487) | **643 = 43.24 %** |
| — among `bridge` items (n=5918) | 113 = 1.91 % |
| yes/no items | 458 = 6.19 % (best-constant within them: 0.5087) |
| best-constant EM floor, full set | 0.03147 (`'no'`) |
| best-constant EM floor, after dropping yes/no **and** leaked | 0.00387 (n=6208) |
| full 10-doc context length | mean 1264 OLMo-2 tokens, p95 1856, max 3049 |

43 % of comparison items are answerable by copying a span from the question
("Which band, **Letters to Cleo** or Screaming Trees, had more members?" → `Letters to Cleo`).
A shallow pruned model that has learned "echo a capitalised span" scores on these **without any
parametric knowledge** — the exact failure mode A03's key control forbids ("不把'给答案原文'误称为
恢复参数知识"). This is a *span-copy* channel, not a floor problem, so a best-constant null does
**not** protect against it; you would need a "copy-a-question-span" heuristic baseline.

It is fixable — drop yes/no and leaked items → n=6208, floor 0.00387, MDE 0.24 pp, which on paper
is the most sensitive axis of all. But then it is a hand-filtered 6208-item subset of a
reading-comprehension benchmark with the reading removed, defensible to no reviewer as a
"multi-evidence parametric knowledge" axis. **Recommend: do not do it.** If multi-hop closed-book
is ever wanted, `context`-stripped 2WikiMultihopQA or MuSiQue answerable-subsets are cleaner —
but that is a separate paper, which is what STATUS.json already concluded.

### Redundancy check with existing axes (measured)

Only **1 of 34** CounterFact relations (`P106` occupation) overlaps PopQA's 16 relations, and
subject overlap is **539 items (2.64 % of CounterFact, 5.18 % of PopQA)**. So CounterFact is
genuinely a *new* slice of Wikidata, not a re-cut of PopQA. That is a point in its favour and
worth stating explicitly if the axis is added.

---

## 5. zsRE: a strictly worse CounterFact

`zsre_mend_eval.json` is well-formed (19086 items, 19085 unique questions, 10720 subjects) and
its `src` field **is** a natural question, so it drops into `build_prompt` with no template
problem — better than CounterFact on that one axis. But:

| property | zsRE | CounterFact |
|---|---:|---:|
| generative best-constant EM floor | 0.01865 (`'french'`) | 0.04886 |
| **2AFC role-frequency prior (in-sample)** | **0.8565** | 0.6426 |
| 2AFC marginal-frequency prior | 0.5902 | 0.5454 |
| multi-word true answers | **53.0 %** | 0.0 % |
| alias lists | partial (`answers` is a list, but 18629/19086 have length 1) | none |

The 2AFC prior of **0.8565** is disqualifying for the discrimination interface: zsRE's `alt`
counterfactuals are drawn from a much narrower pool than its true answers, so candidate identity
alone predicts 86 % of items. And 53 % multi-word answers make length-normalised LL comparison
sensitive to the tokenizer in a way CounterFact's uniformly single-word targets are not.

Also a trap for a future agent: zsRE's fields are `answers` (TRUE), `alt` (counterfactual) and
**`pred` (a stale artifact of whatever model MEND was calibrated on)**. `pred` is neither gold nor
the counterfactual. Using it as gold would silently produce garbage.

**Verdict**: keep on disk as a robustness replicate for a *generative* CounterFact result
(floor 0.0187, n=19086, MDE 0.28 pp — genuinely well-powered). Do **not** use it for 2AFC.

---

## 6. "New injected facts": STATUS.json is right, and downloading data cannot help

This axis needs a CPT phase that injects held-out facts, then measures recall against a probe
set. No dataset download addresses it — the blocker is that A03 has no injection arm and no
injection design. **Leave BLOCKED_BY_DEFINITION.** (One cheap note: CounterFact's `target_new`
strings are *deliberately false* facts, so CounterFact could double as the injection payload for
such an arm later — the data now on disk would be reusable. Not a reason to build the arm now.)

---

## 7. Contamination: a reusable script exists, and the cost is known

**Reusable as-is**: `scripts/audit_olmo2_dolmino_contamination.py` — long token-n-gram containment
against the *exact tokenized* Dolmino stream `data/dolmino_now15b.npy`
(measured `(7570911, 2048) uint32` = 15.505 B tokens), inverted single streaming pass,
64-bit Horner hash, `n ∈ {13, 8}`, verdicts at `contam_high=0.80` / `contam_low=0.10`.
Companion rescorer: `scripts/recompute_closedbook_clean_subset.py`.

**Measured cost of the previous run** (`logs/dolmino_contam_audit.log`, 2026-08-04):
dataset load 12:26:07→12:26:17; eval sketch built by 12:26:33 (571,193 unique n13 hashes from
772,463); train scan **12:26:33 → 12:31:17 = 284 s at 54.6 M tok/s with 64 workers**; artifacts
written 12:31:19. **Total ~5 min 12 s, CPU only.**

**Cost to add the new datasets: ~5 minutes, unchanged.** The wall time is dominated by the
15.5 B-token scan, not the sketch. Measured gram counts for the new sets:

| new eval set | items | n-grams @ n=8 | @ n=13 |
|---|---:|---:|---:|
| CounterFact fact-unit (cloze + `target_true`) | 21919 | 43,867 | 2,558 |
| zsRE `src` | 19086 | 73,786 | 10,114 |
| MQuAKE (all 3 questions/case) | 9000 | 102,378 | 57,997 |
| HotpotQA `question` | 7405 | 99,493 | 62,957 |
| **total added** | | **319,524** | **133,626** |

vs. the existing sketch's 952,098 (n8) / 772,463 (n13) total grams — i.e. a ~34 % / ~17 %
larger sketch, which changes only the `searchsorted` membership cost, not the scan.
**Estimate: one run covering all four new sets ≈ 5–6 min on a 64-core CPU. Essentially free.**

### ★ But the audit as written is nearly blind on CounterFact

| eval set | fraction of items **< 13 tokens** (undecidable at n=13) | < 8 tokens |
|---|---:|---:|
| **CounterFact rendered cloze** | **97.2 %** | 48.6 % |
| CounterFact cloze + target (fact unit) | 94.2 % | 28.5 % |
| MQuAKE single-hop cloze | 94.4 % | 32.8 % |
| zsRE `src` | 78.4 % | 5.6 % |
| MQuAKE multi-hop question | 8.0 % | 0.3 % |
| HotpotQA question | 7.8 % | 0.0 % |

The script treats `len < n` as **SHORT / undecidable**, not clean (correctly — see its docstring).
So an n=13 audit of CounterFact decides only ~2.8 % of items and would report a meaningless
`contam_rate ≈ 0`. This is the same artifact already visible in the existing summary for PopQA:
`n_decidable=750` of 14267 at n13 (5.2 %), which is why PopQA's `contam_rate=0.0` there must not
be read as "PopQA is clean".

**Required protocol change if CounterFact is audited**: run at **n ∈ {8, 5}** and treat the
*fact* (`cloze + " " + target_true`) as the unit rather than the question — a contaminated
CounterFact item is one where Dolmino states the fact, and 21919 short clozes at n=5 give
105,936 decidable grams (only 0.4 % undecidable). Expect a *high* raw hit rate at n=5 (a
5-gram like "Angola is located in Africa" is a common true statement) — which is the honest
finding: **CounterFact's true targets are head facts and are almost certainly in Dolmino by
construction.** For A03 that is fine (we *want* the model to know them; we are not claiming
they are held out) — but it means the contamination audit cannot be used to build a
"clean subset" for this axis, only to document the overlap. Say that up front rather than
producing a clean-subset number that means nothing.

One non-issue, checked so nobody re-checks it: the existing audit's tokenizer is the **7B**
OLMo-2 (`models/OLMo-2-1124-7B`) while A03's arms are 1B (`models/OLMo-2-0425-1B`).
**Verified identical** — both report `vocab_size = 100278` and produce byte-identical ids
(`"Angola is located in Africa"` → `[10976, 8083, 374, 7559, 304, 10384]` under both). So the
existing audit's rates and the token-length table above are valid for the 1B arms as-is.

---

## 8. Bottom line

**Do one thing**: CounterFact as a **conflicting-knowledge 2AFC** axis, pooled n=21919,
null floor **0.5364** (held-out role-frequency prior, NOT 0.50), with the subject-masked control
and generative EM (floor 0.04886) disclosed alongside.

But be clear-eyed about the cost/benefit: this requires extending `eval_olmo2_mmlu_content.py`
(~150–250 lines + an analyzer null branch), which is **a coder project, not a gate** — the same
verdict `GATE_FOURAXES_VERDICT.md` §6 reached for Arms 4–6. Given that A03's live question right
now is whether Arm 6 replicates Arm 3's +0.48 pp — a question this axis is **underpowered to
answer as a vs-floor test** — the honest sequencing is:

1. finish Arm 6 (running on `.73`, watcher on `.82`);
2. only then decide whether A03 needs a fourth conflicting-knowledge axis at all.

**Drop permanently**: MQuAKE (floor 0.177–0.235 + multi-hop-depth confound),
HotpotQA-stripped (43 % answer-copy channel among comparison items).
**Keep as replicate only**: zsRE (2AFC prior 0.857 kills its discrimination interface;
generative floor 0.0187 at n=19086 is genuinely good).
**Leave blocked**: new-injected-facts (no download helps).

### Corrections to make in `STATUS.json:remaining_axes_status`

* `conflicting_knowledge`: reason is **not** "NO_DATA" (data now on disk, 15 MB) and **not**
  "requires context injection" (a closed-book 2AFC is native to the construct). Real reason:
  needs an LL-scoring harness extension + a 0.5364 non-obvious null. Status →
  `AVAILABLE_BUT_NEEDS_HARNESS_WORK`.
* `multi_evidence`: reason is **not** "WRONG_FORMAT" (MQuAKE has no context field at all; HotpotQA's
  is a separable column). Real reasons: MQuAKE floor 0.177; HotpotQA 10.2 % answer-in-question.
  Status → `REJECTED_ON_FLOOR_AND_CONSTRUCT`.
* `new_injected_facts`: unchanged, correctly blocked.

## 9. Provenance

* Data + sha256 manifest: `data/knowledge_axes/MANIFEST.md` (197 MB, wzc1 only, not in git)
* Harnesses read: `scripts/eval_olmo2_closedbook_qa.py`, `scripts/eval_olmo2_mmlu_content.py`
* Floor conventions: `proposal/archive/A03-parametric-vs-external-memory/code/analyze_1b_knowledge_floor.py`,
  `GATE_FOURAXES_VERDICT.md`
* Contamination: `scripts/audit_olmo2_dolmino_contamination.py`,
  `logs/dolmino_contam_audit.log`, `bench_results/olmo2_dolmino_contamination/`
* All floors/MDEs/overlaps in this report were computed from the downloaded files on
  2026-08-10, CPU-only on LOCAL. Tokenizer stats via `models/OLMo-2-1124-7B`.
* MDE convention: α=.05 two-sided, 80 % power, one-sample binomial vs p₀,
  solved by fixed-point iteration on `p₁ = p₀ + z_.975·√(p₀(1−p₀)/n) + z_.80·√(p₁(1−p₁)/n)`.

---

## MAIN independent verification (2026-08-10 12:0x GMT+8)

I re-derived the load-bearing number myself from the parquet files rather than
accepting the report, because the whole recommendation rests on it.

**Reproduced exactly.** Candidate-only heuristic — score each candidate string by
`count(appears as target_true) − count(appears as target_new)` in the *other*
split, pick the higher, never look at the subject or the fact:

| direction | n | accuracy | leak vs naive 0.50 |
|---|---:|---:|---:|
| train → test | 2191 | **0.5429** | +4.29 pp |
| test → train | 19728 | **0.5364** | +3.64 pp |
| in-sample (pooled) | 21919 | 0.5697 | +6.97 pp |

Held-out values match the report's 0.5364 / 0.5429 to 4 decimal places.

**One correction.** The agent's chat summary said the in-sample figure was
**64.3%**; with the heuristic as described I measure **57.0%**. The committed
report does not contain 64.3 anywhere, so nothing here depends on it — but
**64.3% must not be quoted**; it is not reproducible from the stated procedure
(it may have come from a relation-stratified variant that was not written down).
The conclusion is unaffected: the floor is empirically ≈0.54, not 0.50.

**The leak mechanism, made concrete.** First test item is
`prompt="{} is located in", subject="Angola", target_true="Africa",
target_new="Antarctica"`. `Africa` is a common true target across the dataset;
`Antarctica` is a common *counterfactual* target. So "prefer whichever string
more often plays the true-answer role" wins without any knowledge of Angola.
CounterFact's counterfactuals were not sampled uniformly over the candidate
vocabulary, and that asymmetry is worth 3.6–4.3 pp.

**Consequence for A03.** Had this axis been run against a 0.50 floor, a pruned
arm scoring 0.55 would have been reported as "+5 pp above floor / knowledge
retained" when the honest reading is "+1 pp, indistinguishable from a
fact-blind string-frequency heuristic." That is the same class of error A01
exists to catch, on a new dataset.

Verified by MAIN directly from
`data/knowledge_axes/counterfact/{test,train}-*.parquet`
(fields live under `requested_rewrite.{target_true,target_new}.str`, not at top
level — a detail worth recording since the obvious top-level column scan finds
nothing).
