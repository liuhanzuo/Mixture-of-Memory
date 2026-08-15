# B08 leg-1 — IMPLEMENTATION PLAN for the `A-notes-only` arm, and the `mom_notes` asset verdict

**Written 2026-08-15. 0 GPU, 0 ssh, PRE-DATA. No source file was modified by this document.**

Closes CPU blockers 2 and 4 of `STATUS.json.remaining_blockers_all_CPU` at the *decision* level:
this file says **exactly what to change, where, and what else it touches**. The task that produced
it explicitly forbade editing `longmemeval/run_baseline.py` this round, because it is a
git-tracked shared harness file. **The edit is specified here and not applied.**

Every code claim below is `file:line` against the working tree at commit `46ea84d`.

---

## 0. Headline verdicts, up front

| question asked | verdict |
|---|---|
| Is the `A-notes-only` arm a few lines, or does it ripple? | **A few lines, and it does NOT ripple.** `run_baseline.py` has **zero importers** — nothing in the repo imports it or shells out to it. The change is +1 CLI flag and a 3-branch replacement of one line (`:152-162`). |
| Does the 3-arm design survive the missing `mom_notes` asset? | **YES — the third arm was never supposed to be `mom_notes`.** The gate already specifies **self-notes** (`STATUS.json.next_gate.notes_generator_must_be_the_reader_itself`, prereg §4.1). `mom_notes` being unrunnable removes an *option*, not an *arm*. **No narrowing to two arms is required.** |
| So is any arm asset-less? | **No.** All three arms run on `models/Meta-Llama-3-8B` (15G, wzc1-verified) which is *already* the reader. The unbuilt part is ~80 lines of harness glue, not a missing checkpoint. |
| Did anything NEW turn up? | **Yes — 3 further blockers the prior audit did not have (§5), one of which silently changes the cost model and one of which makes the gate unrunnable as specified.** |

---

## 1. Blocker 2 — the `A-notes-only` arm

### 1.1 The defect, verbatim

`longmemeval/run_baseline.py:152-162`:

```python
        reader_evidence = evidence
        notes = compressor.compress(ex.question, ex.question_date, evidence)
        if notes:
            notes_block = Evidence(
                round_id=f"{ex.question_id}_mom_notes",
                session_id="MoM-NOTES",
                session_date=ex.question_date,
                text=f"MoM NOTES: {notes}",
                score=float("inf"),
            )
            reader_evidence = [notes_block] + list(evidence)      # <-- line 162
```

Line 162 is the only assignment of `reader_evidence` in the notes branch, and it is
unconditional. There is **no** path that hands the reader the notes block without the raw
evidence, so `A-notes-only` is not expressible. Confirmed by exhaustion: `--compressor` has
exactly two choices (`:329-334`, `none|mom_notes`) and neither changes composition.

### 1.2 Blast radius — MEASURED, and it is empty

```
grep -rn "longmemeval.run_baseline|from .run_baseline|from longmemeval.run_baseline" \
     --include=*.py --include=*.sh .        # excluding .git/.claude/cc_state/__pycache__
```

→ **4 hits, all inside `run_baseline.py` itself** (its own docstring examples at `:11,15,23` and
its own `prog=` string at `:264`). Corroborating evidence that it is a leaf:

- `longmemeval/__init__.py:26-38` re-exports `data` / `backends` / `reader` / `scoring` and
  **deliberately not** `run_baseline` — so `import longmemeval` does not pull it in.
- `_apply_token_budget` (`:59`) is referenced only at `:142`, i.e. its single internal call site.
- `tests/` has **19** test files, **0** of which mention `longmemeval`
  (`grep -rln longmemeval tests/` → empty).
- The only other longmemeval consumer in the tree, `scripts/eval_qcmem_longmemeval.py:130`,
  imports `longmemeval.data.load_longmemeval` — **not** `run_baseline` — so it is unaffected.

**Conclusion: `run_baseline.py` is a CLI leaf.** Changing its internals cannot break another
experiment. (It IS git-tracked — `git ls-files longmemeval/` lists all 8 files — so the edit still
needs a normal commit, just not a coordination round.)

### 1.3 Minimal change — exact diff plan

**One new flag.** Insert after `run_baseline.py:334` (end of the `--compressor` argument, inside
the `# -- MoM notes compressor` block):

```python
    p.add_argument("--reader_evidence_mode", type=str, default="notes_plus_evidence",
                   choices=["notes_plus_evidence", "notes_only", "evidence_only"],
                   help="What occupies the reader's context when the compressor "
                        "returns a non-empty notes string. 'notes_plus_evidence' "
                        "(default) = notes block PREPENDED to raw evidence "
                        "(the pre-2026-08-15 behaviour, bit-identical); "
                        "'notes_only' = notes block ONLY, raw WITHHELD; "
                        "'evidence_only' = ignore the notes (equivalent to "
                        "--compressor none but keeps the notes-generation cost, "
                        "so it is a cost control, not a quality arm).")
```

**Replace exactly one line.** `run_baseline.py:162`:

```python
-            reader_evidence = [notes_block] + list(evidence)
+            mode = getattr(args, "reader_evidence_mode", "notes_plus_evidence")
+            if mode == "notes_only":
+                reader_evidence = [notes_block]
+            elif mode == "evidence_only":
+                reader_evidence = list(evidence)
+            else:  # notes_plus_evidence -- unchanged default
+                reader_evidence = [notes_block] + list(evidence)
```

**One report field**, so the arm is recoverable from the artifact rather than from shell history.
Add to the `report` dict (`run_baseline.py:177-193`, alongside the existing `"compressor"` key at
`:186`):

```python
        "reader_evidence_mode": args.reader_evidence_mode,
```

**Size: +1 flag (11 lines incl. help text), +6 lines replacing 1, +1 report line.**
No function signature changes. No new imports. No behaviour change on any existing invocation:
the default `notes_plus_evidence` reproduces line 162 exactly, and with `--compressor none`
the branch is dead anyway because `notes` is `""` (`compressor.py:70-76`).

### 1.4 Why `getattr` and not `args.reader_evidence_mode`

`_run(examples, args)` is also reachable with a hand-built namespace, and `--self_test`
(`:350-356`) mutates `args` in place. `getattr(..., default)` keeps the notes branch working for
any caller that constructs args without the new field. One-line cost, removes a whole class of
`AttributeError`.

---

## 2. Blocker 4 — the `mom_notes` / `SelfNotesCompressor` verdict

### 2.1 `--compressor mom_notes` is confirmed unrunnable, and the reason is on disk

`MoMNotesCompressor.__init__` (`longmemeval/compressor.py:106-169`) needs **both**:

1. an `adapter_config.json` parseable by `build_mem_space_config` into a `MemorySpaceConfig`
   (`compressor.py:150-152`, calling `scripts/run_babilong_mem_space.py:180`), and
2. a mem_space adapter `.pt` checkpoint (`compressor.py:161-167`).

`build_compressor` (`compressor.py:229-233`) hard-fails without `--compressor_checkpoint` **and**
`--compressor_adapter_config`. Re-measured on wzc1 this session:

| check | result |
|---|---|
| `find outputs/ -name adapter_config.json \| wc -l` | **45** |
| of those, containing `num_slots` or `slot_dim` | **0** |
| `peft_type` distribution across all 45 | **`LORA` × 45** (0 other values) |
| `ls -d outputs/*mem_space*` | **0 directories** |
| `find outputs/ -name '*.pt' -path '*mem_space*'` | **0 files** |
| `find . -name '*.pt' -path '*mem_space*'` (repo-wide, depth 4) | **0 files** |

All 45 live under exactly four run families — `qcmem_distill_{hy3_j32_r32, llama3_j12_r32_4k,
qwen_j12, qwen_j12_r32_4k, qwen_j9b0_pg19_nctx7}` — i.e. they are QCMem LoRA distillation
adapters, structurally incapable of configuring a slot bank.

**Corroboration that this is real absence and not a search artifact:** the two mem_space adapter
configs that repo scripts still reference by hardcoded path —
`outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json`
(`scripts/_launch_stairs_L5.sh:12`, `scripts/_launch_hnstv2_tree.sh:34`,
`scripts/_eval_hnstv2_bundle.sh:51`) and `outputs/mem_space_perdoc_chunk128/adapter_config.json`
(`scripts/eval_base_longbench_r33.sh:19`) — **both return MISSING on stat.** So several *other*
scripts in this repo are also already broken against wzc1 for the same reason. B08 is not
special-cased; the mem_space asset family is simply gone from this disk.

The only `num_slots` strings left on wzc1 are inside **BABILong result JSONs**
(`tree_step2000/**/*.json` → `model.num_slots = 128`), i.e. *records of past runs*, not loadable
configs. That is exactly the shape of a deleted asset family.

### 2.2 …and this does NOT cost the gate an arm

**This is the load-bearing correction to the framing of the task that commissioned this file.**
The concern was "45 LoRA configs, zero with `num_slots` → is one of the three arms asset-less?"

**No.** The gate's third arm was never `mom_notes`. `STATUS.json.next_gate` already says so, in a
key written before this session:

> `notes_generator_must_be_the_reader_itself`: *"Self-notes: the same Meta-Llama-3-8B that reads
> also writes the notes from the same retrieved evidence. Using mem_space (`--compressor
> mom_notes`) would introduce a SECOND model, confounding 'notes help' with 'the mem_space model
> is good' — and it is unrunnable anyway."*

So `mom_notes`'s death is **doubly harmless**: it was rejected on *scientific* grounds (it breaks
the single-variable design by adding a second model) *before* it was found to be unrunnable on
*asset* grounds. The three arms are

```
A-raw          : --compressor none                                     RUNNABLE TODAY
A-notes+raw    : --compressor self_notes --reader_evidence_mode notes_plus_evidence
A-notes-only   : --compressor self_notes --reader_evidence_mode notes_only
```

and all three need **one** model, `models/Meta-Llama-3-8B`, which is the reader they already share.
**Recommendation: keep three arms. Do not narrow to two. Do not build a mem_space asset.**

### 2.3 `SelfNotesCompressor` — where it goes and what it must reuse

New class in `longmemeval/compressor.py` (append after `MoMNotesCompressor`, ~60 lines), plus one
branch in `build_compressor` (`:221-242`) and one added choice in `--compressor`
(`run_baseline.py:329-331`, `choices=["none", "mom_notes", "self_notes"]`).

Non-negotiable design points, each traceable to a specific hazard:

1. **Share the reader's weights, do not load a second copy.** `LocalHFReader.__init__`
   (`reader.py:206-210`) already holds an 8B bf16 model on `cuda`. A second
   `AutoModelForCausalLM.from_pretrained` would double resident weights for no reason and, worse,
   would make "same model" a claim rather than a fact. Cleanest seam: build the reader first
   (`run_baseline.py:130`), then pass it into `build_compressor` so `SelfNotesCompressor` holds a
   reference to `reader.model` / `reader.tokenizer`. That is a signature change to
   `build_compressor(name, args)` → `build_compressor(name, args, reader=None)`, which is safe
   because `build_compressor` has exactly one caller (`run_baseline.py:132`).
2. **Notes must be generated from the SAME post-budget evidence list** the answer arms see —
   i.e. after `_apply_token_budget` at `:142`, which is already where `compress()` is called
   (`:153`). Do not re-retrieve.
3. **Generate the notes ONCE per question and reuse the identical string across
   `A-notes+raw` and `A-notes-only`.** The prereg freezes "same notes text" as part of the single
   variable (`STATUS.json.next_gate.single_variable`). Regenerating per arm would let decoder
   nondeterminism leak into the contrast. Concretely: run the notes pass once, persist
   `{question_id: notes}` to JSONL, and have both arms read that file. This also makes the notes
   auditable, which the `U` metric needs — `U` is defined against *"that arm's own context"*, so
   the scorer must be able to reconstruct each arm's context exactly.
4. **Greedy, and the same decode settings as the reader.** `do_sample=False, num_beams=1`
   (matching `reader.py:282-289`), `max_new_tokens=128` (the existing
   `--compressor_max_new_tokens` default, `run_baseline.py:342-343`).
5. **Reuse `MoMNotesCompressor`'s instruction verbatim** (`compressor.py:181-185`) so the notes
   prompt is not a new uncontrolled variable:
   `"Summarize the facts from the conversation above that are relevant to answering this
   question: {question}\nRelevant facts:"`.

### 2.4 Rename the leaked `MoM` label — this is a correctness issue, not cosmetics

`run_baseline.py:156-159` hardcodes `session_id="MoM-NOTES"` and `text=f"MoM NOTES: {notes}"`.
With a **self**-notes generator those strings are **factually wrong inside the reader's prompt**,
and `Evidence.as_block` (`backends.py:47-53`) renders `session=` into the prompt the model reads.
So the label is not a comment — it is model input. It must become `SELF-NOTES` / `SELF NOTES:`
when `--compressor self_notes` is active, or the arm's context misdescribes its own provenance.
Cheapest correct form: derive the label from the compressor (e.g. a `label` property on the
`Compressor` ABC, defaulting to `MoM`), so the `mom_notes` path stays byte-identical.

---

## 3. Measured consequences (0 GPU, real tokenizer, this session)

Method: real `models/Meta-Llama-3-8B` tokenizer, real BM25 retrieval at `top_k=10`, real
`--evidence_token_budget 4000`, and `LocalHFReader._build_prompt` (`reader.py:229-254`)
**reimplemented line-for-line** rather than approximated. `CUDA_VISIBLE_DEVICES=""` throughout.

### 3.1 The stratum reproduces exactly

| quantity | STATUS.json says | measured | verdict |
|---|---|---|---|
| stratum n | 134 | **134** | ✅ |
| knowledge-update n | 78 | **78** | ✅ |
| single-session-assistant n | 56 | **56** | ✅ |
| `_abs` in stratum | 6 | **6** | ✅ |
| `_abs` in knowledge-update / SSA | 6 / 0 | **6 / 0** | ✅ |
| `_abs` repo-wide | 30 | **30** | ✅ |
| primary denominator 134−6 | 128 | **128** | ✅ |

### 3.2 ⚠️ The `A-notes-only` arm is ~14.5× cheaper than the cost model assumes

| arm | Σ prefill tokens (n=134) | mean | max |
|---|---|---|---|
| `A-raw` (= `A-notes+raw` basis) | **529,620** | 3952.4 | 4609 |
| **`A-notes-only`** | **36,577** | **273.0** | 312 |

`STATUS.json.gpu_cost_estimate.arithmetic` computes **three answer arms at 529,620 prefill each**.
The third arm's context is a single ≤128-token notes block, so its real prefill is 36,577.
Re-running the same anchor arithmetic (`j0_top12`: 0.15180 ms/prefill-tok, 39.0021 ms/decode-tok):

```
A-raw                  80.4 s prefill + 334.5 s decode = 414.9 s
A-notes+raw            80.4 s prefill + 334.5 s decode = 414.9 s
A-notes-only            5.6 s prefill + 334.5 s decode = 340.0 s   <-- was booked as 414.7 s
notes-generation pass   80.4 s prefill + 669.0 s decode = 749.4 s
core = 1919.2 s = 0.5331 GPU-h ; x2.0 slack = 1.0662 ; + judge 0.045 = 1.111 GPU-h
```

vs the booked **1.153 GPU-h**. **The headline survives** (1.11 vs 1.15, both far under the 2 GPU-h
ceiling) and the direction is favourable, so this is a *precision* correction, not a re-plan. It
is recorded because an unexamined 14.5× error in a per-arm term is the kind of thing that is
harmless at n=134 and not harmless after the K2 escalation to n=500.

Also note the anchor's own limit: `read_len = 6177` for the anchor vs ~3952 mean here, so
prefill is being **extrapolated downward** ~1.6×; and decode dominates 4:1 either way
(334.5 s vs 80.4 s), so the total is really a decode-bound estimate. Both facts already sit in
`gpu_cost_estimate.CORRECTION_20260814_wrong_anchor_arm`; neither is new.

### 3.3 ✅ The `A-notes+raw` arm does NOT silently truncate raw evidence — a real hazard, checked

`LocalHFReader._truncate_evidence_block` (`reader.py:213-227`) caps the evidence section at
`max_prompt_tokens - reserve`. If a prepended notes block pushed the section over that cap, the
arm labelled *"notes prepended, raw kept"* would silently be *"notes prepended, raw partly
deleted"* — and the single-variable claim would be false. Measured:

```
reserve (measured)                                          = 165
cap = 7000 - 165                                            = 6835
evidence-section tokens: mean 3838.4, p90 4278, max 4505
items where RAW section already exceeds cap                 = 0 / 134
items exceeding cap after prepending a ~150-tok notes block  = 0 / 134
```

**0/134 both ways, with ~2330 tokens of headroom at the observed max.** So on this stratum at this
config the arm name is literally true. ⚠️ **This is a property of the config, not of the harness**
— it must be re-measured if `--top_k`, `--evidence_token_budget`, `max_prompt_tokens`, or
`--compressor_max_new_tokens` change, exactly like the closure premise
(`established_measurements.closure_RE_VERIFIED_at_the_gates_own_budget_20260814.standing_rule`).

---

## 4. Recommended build order (all CPU, no card)

| # | task | size | why this order |
|---|---|---|---|
| 1 | `--reader_evidence_mode` + the `:162` 3-branch replacement + report field | ~18 lines, 1 file | Smallest, zero-ripple, and it unblocks arm 3. Ship alone. |
| 2 | `SelfNotesCompressor` + `build_compressor(..., reader=)` + `self_notes` choice + the `SELF-NOTES` label fix (§2.4) | ~80 lines, 2 files | Needs #1 to be observable end-to-end. |
| 3 | **A stratum selector** (§5.1) | ~10 lines | **The gate cannot be run without it.** |
| 4 | The `U` scorer (blocker 3) | new file | The novelty metric. Build on ALCE / FActScore / SummaC machinery and *say so* (`RELATED_WORK.md` MUST-NOT-CLAIM item 5) — the honest statement is "no such scorer exists in **this repo**". |
| 5 | Judge input adapter (blocker 5, §5.2) | ~25 lines | Needed before ACC exists. Bigger than "an adapter": see §5.2. |
| 6 | zwfy6 asset verification (blocker 6) | 1 ssh | **Last, and only when a card is actually being booked.** |

Order matters: #3 is cheaper than #2 but was invisible until this session, and #5 is larger than
its one-line description in `STATUS.json` suggests.

---

## 5. ⚠️ THREE blockers the prior audit did not have

These are new findings, not restatements. #5.1 makes the gate **unrunnable as written**.

### 5.1 NEW BLOCKER 7 — there is no way to select the stratum. The gate cannot run.

`run_baseline.py` has **no** question-type filter. `--limit N` (`:270-271`) takes a prefix via
`load_longmemeval(path, limit=...)` (`data.py:138-141`). Measured on the real file:

```
stratum member indices in load order: 366 … 499   (contiguous: True, len 134)
Counter(types in exs[366:500]) = {knowledge-update: 78, single-session-assistant: 56}
```

The stratum happens to be a **contiguous tail block** — but it is a **suffix, not a prefix**, so
`--limit` cannot express it (`--limit 500` = all 500; `--limit 134` = the *first* 134, which is a
different set entirely). Running all 500 instead is **not** an acceptable substitute: the gate's
whole premise is that retrieval is closed *on this stratum*, and the re-measured off-stratum
numbers in `STATUS.json` show closure **fails** elsewhere
(`single-session-preference` any_hit 0.7667 → 0.7000 at budget=4000). Scoring 500 and subsetting
afterwards would also 3.7× the cost (Σ prefill 2,010,736 vs 529,620) for no added information.

**Fix (~10 lines, same file, same zero-ripple argument as §1.2):** add
`--question_types knowledge-update,single-session-assistant` and filter in `main()` right after
`load_longmemeval` (`:361`), then **assert the survivor count**:

```python
    if args.question_types:
        want = {t.strip() for t in args.question_types.split(",") if t.strip()}
        examples = [e for e in examples if e.question_type in want]
        if args.expect_n is not None and len(examples) != args.expect_n:
            raise SystemExit(
                f"stratum size {len(examples)} != --expect_n {args.expect_n}")
```

The `--expect_n 134` assert is not decoration: it is the mechanical form of
`readout_preregistration.why_an_assert_not_a_nan_check` ("assert `n_scored == expected` per cell,
not just check for NaNs"), applied at *input* time so a silently-wrong stratum cannot reach the
scorer. Do not rely on `--limit`, and do not hardcode `exs[366:500]` — the slice is a property of
today's file ordering, and `data.py` also accepts JSONL (`:163`).

### 5.2 NEW BLOCKER 8 — the judge adapter is a field-name mismatch, not just a shape mismatch

`STATUS.json` blocker 5 says the judge "takes `--result_dirs` of LoCoMo-shaped dirs; longmemeval
emits `{question_id, hypothesis}` JSONL". True, but the gap is wider than a rename. Measured:

- `a02_judge_openweight.py:187-201` `load_preds()` globs **`preds*.jsonl`** inside each
  `--result_dirs` entry and keys on **`item["id"]`** (a hard `KeyError` if absent).
- The judge body then reads `item["pred"]` (`:341,357`), `item["question"]` (`:355`),
  `item.get("answers", [])` (`:356`), `item.get("category")` (`:354`), and
  `item.get("is_abstention", False)` (`:337`).
- `write_submission` (`scoring.py:30-43`) emits **only** `question_id` and `hypothesis`, into a
  **single file** at `--out`, and **discards everything else** — including the gold answer, the
  question text, the question type, and any abstention marker.

So the adapter must **re-join the submission against the source data** to recover `question`,
`answers`, `category`, and `is_abstention`, then re-emit as `preds*.jsonl` with `id`/`pred`.
Two consequences worth pre-committing:

1. **`is_abstention` must be set explicitly.** If it defaults to `False`, the 6 `_abs` items get
   sent to the semantic judge instead of the refusal rule (`:337-340`), which
   `readout_preregistration.abstention_handling` says "corrupts both ACC and U". The 6 items are
   identifiable by the `_abs` suffix on `question_id` (verified: 6 in stratum, 30 repo-wide), and
   they **do** carry a gold answer string (e.g. *"The information provided is not enough. You
   mentioned trying Korean restaurants but not Italian restaurants."*) plus 2 gold sessions each —
   so a naive "no gold answer ⇒ abstention" heuristic would **not** find them.
2. **The judge writes into the result dir.** `judge_meta.json` is written per `--result_dirs`
   entry (`:291-293`) and the cache is appended to `judge_cache_openweight.jsonl` (`:296,346`).
   Give each arm its **own** directory or the three arms will share one cache. The cache is keyed
   on `id` alone with **no arm field**, so a shared directory would make arm 2 silently reuse
   arm 1's verdicts — a same-`id`-different-`pred` collision that produces no error and no NaN.
   **This is the single most dangerous item in this file.**

Also: `a02_judge_openweight.py`'s `_JUDGE_TEMPLATE` (`:48-63`) is written for **LoCoMo**
("*a question about a long, multi-session dialogue (the LoCoMo benchmark)*"). Using it verbatim on
LongMemEval means the judge prompt names the wrong benchmark. Changing it breaks protocol identity
with A02's archived numbers; keeping it is a documented wart. **Recommendation: keep it verbatim**
(protocol identity is worth more than the benchmark noun, and B08 has no archived judge numbers to
be consistent with) **and record the choice in `judge_meta.json`**, PRE-DATA.

### 5.3 NEW (minor) — `notes_examples` keeps only 3, so notes are not auditable

`run_baseline.py:163-166` caps `notes_examples` at **3** entries for the report. `U` is defined
against *"that arm's own context"*, so scoring it requires **all 134** notes strings, verbatim.
The notes-persistence file in §2.3 point 3 is therefore not an optimisation — it is what makes the
primary metric computable at all. The 3-example report field is a debugging nicety; do not mistake
it for provenance.

---

## 5A. ⚠️ The scheduler cannot see any of B08's blockers

Discovered while verifying the `STATUS.json` append. Recorded in
`STATUS.json.ready_queue_visibility_defect_20260815`.

`proposal/ready_queue.py:252-253` hard-codes:

```python
BLOCK_KEYS = ["blocking_dependency", "blocked_by", "required_before_stage0",
              "gpu_policy", "premise_falsified"]
```

B08 carries its blockers under **`prior_gate`** and **`remaining_blockers_all_CPU`**. Neither name
is in that list. Measured via `ready_queue.read_one()` on B08's own file:

```
HEAD (before my append): lifecycle=ready_cpu  n_keys=20  problems=0  live_blockers=0
NOW  (after my append) : lifecycle=ready_cpu  n_keys=24  problems=0  live_blockers=0
```

**`live_blockers = 0` both before and after.** The scheduler has never known B08 has
prerequisites. It reaches `ready_cpu` on a *different* ground — `novelty gate not adjudicated
(absent) -> the actionable task is 0 GPU: run it` — which is now **wrong in its particulars**
(the adjudication exists, is committed as `463dca4`, and is re-verified in `RELATED_WORK.md` §11)
while still landing on the **right answer**.

Two consequences:

1. **The commissioning expectation cannot be met as stated.** The task expected B08 to drop from
   "4 un-cleared blockers" in the queue output. It never showed 4 — it showed **0**. So the count
   could not improve. The honest read-out: the blocker list became **more accurate**, not shorter.
2. **A latent promotion hazard.** If anyone flips `novelty_checked` to `true` — which
   `RELATED_WORK.md` §10.1 shows is one line away, and which its own author deliberately declined
   to write — B08 moves toward `ready_gpu` while **five** CPU blockers (3, 5, 7, 8, 9) remain
   invisible. That is exactly the paperwork-counts-as-readiness failure `ready_queue.py` was
   written to stop, arriving through a key-name gap rather than a logic gap.

**Second, smaller defect:** `STATUS.json` declares `lifecycle: needs_prior_gate`; the reader
reports `ready_cpu`. Only `dead` / `promoted` / `running` / `ready_cpu` short-circuit as
authoritative; `needs_prior_gate` falls through to inference, which overrides the declaration.
Benign for GPU safety (both are `!= ready_gpu`) but it is a declared-vs-inferred mismatch on the
field `LIFECYCLE_SCHEMA.md` calls 唯一的机器可读状态.

**Fix deliberately NOT applied here.** `ready_queue.py` is the shared scheduler for all 15
proposals, so touching `BLOCK_KEYS` re-classifies every one of them at once. Per
`memory/fix-the-class-not-the-instance.md` that belongs in one reviewed change covering the whole
key family — with a before/after diff of `python3 proposal/ready_queue.py` confirming **no**
proposal moves *into* `ready_gpu` — not in a drive-by edit inside a B08 task. Per
`memory/reporting-a-gap-is-not-closing-it.md` it is filed as an explicit task, not left as a
caveat.

---

## 6. What this file does NOT claim

- **No code was changed.** `git status` for `longmemeval/` is clean; the diffs above are plans.
- **No GPU, no ssh, no node touched.** All measurements are CPU tokenizer + BM25 + file reads on
  wzc1 with `CUDA_VISIBLE_DEVICES=""`.
- **Everything is wzc1-scoped.** `/apdcephfs_zwfy6` is not mounted here and ssh was barred, so
  blocker 6 stands untouched: `models/Meta-Llama-3-8B`, `data/longmemeval/longmemeval_s.json`, and
  the Qwen3-8B judge weights must be confirmed **on the target node** before a card is booked
  (`memory/two-disk-rule-applies-to-main-too.md`).
- **This does not make B08 `ready_gpu`.** Blockers 3 (the `U` scorer), 5/8 (judge adapter), 7
  (stratum selector) and 6 (zwfy6) remain open, and #7 is newly *added* by this session. The
  gate also still needs its ~1.1 GPU-h and a free card, and per
  `CLAUDE.md` the whole cluster is currently committed.
- **`SelfNotesCompressor` is specified, not written.** §2.3 is a design contract, not code.

---

## 7. Commit provenance — a concurrent-commit race that RESOLVED ITSELF

**Final state (authoritative): all four artifacts of this task live in commit `c55090d`**, whose
subject is `prereg(B08): record that dd6a4bd mislabels the B08 leg-1 artifacts` —

```
proposal/backlog/B08-memory-applications/LEG1_IMPL_PLAN.md      (this file)
proposal/backlog/B08-memory-applications/RELATED_WORK.md        (§11 appended)
proposal/backlog/B08-memory-applications/STATUS.json            (+4 keys, byte-prefix append)
proposal/shared/code/b08_append_status_keys_20260815.py         (the append script)
```

`git log -- proposal/backlog/B08-memory-applications/` shows `c55090d` on top. Nothing further is
needed; the rest of this section is the audit trail for a transient anomaly.

**What happened.** A concurrent B06 agent was committing in the same working tree. Its first
commit, `dd6a4bd`, ran between this task's `git add` and its own `git commit` and **swept the
staged B08 files into itself** — it contained exactly the four B08/shared files (1119 insertions,
0 deletions) and **none of B06's own**. This section was originally written to record that
mislabelling, on the explicit decision **not** to amend someone else's hash.

**Then it self-corrected.** The B06 agent amended its commit to hold its own five files
(`80bf6d3`, 2509 insertions: `DRIFT_RESOLUTION_VERDICT.md`, its `STATUS.json`, and three
`evidence/` files), which removed the B08 files from its tree. `dd6a4bd` is now **dangling** —
unreachable from any branch, pending GC. The B08 files landed in `c55090d` instead, correctly
titled. So the anomaly the commit message names **no longer exists in reachable history**, and
`c55090d`'s own subject line is now self-referentially stale.

**Why this is recorded rather than rewritten.** Amending `c55090d` to a cleaner subject would be
cosmetic, and the whole reason the race was survivable is that nobody rewrote a hash out from
under a concurrently-running agent. Applying that rule to my own commit too is the consistent
choice. Verified integrity: `git diff c55090d` against the working tree is empty, and its
`STATUS.json` parses with **24 keys** including all four `*_20260815` additions.

**Operational lesson worth keeping:** `git add` followed later by `git commit` is **not atomic**
when another agent shares the working tree. Two agents were committing to one checkout, and the
index is global. Staging and committing should be one step, and a task should verify
`git show --name-only HEAD` lists *its own* files before reporting a commit.
