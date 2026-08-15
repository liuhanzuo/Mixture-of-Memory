# B11 — K3 exit precheck: duplicate scan, mechanical proof of (a), stratified measurement of (b)

**Date 2026-08-16 · GPU spent: ZERO (CPU rescoring + read-only GitHub API only) · node: LOCAL (wzc1)**

Purpose: make B11's `next_gate` (the K3 exit — file an upstream issue on `booydar/babilong`)
**ready to file in one step**, and independently re-verify its two assertions rather than inherit them.

> ## ⛔ THE ISSUE HAS **NOT** BEEN FILED
> Nothing was posted to GitHub. Only read-only API endpoints were called (`GET /repos`,
> `GET /issues`, `GET /issues/comments`, `GET /search/issues`, `GET /branches`, `GET /forks`,
> `raw.githubusercontent.com`). **Filing requires explicit human sign-off**, which this pass does not
> have. See §5 for the exact final state.

**Headline of this pass.** Both assertions survive, but **assertion (a)'s wording must not be filed as
written**: line 31 is **executed on every call** (measured, `sys.settrace`) and is a **guaranteed
no-op**, not unreachable code. And assertion (b) is now *stronger* than the record claimed: the two
independent corpora I measured disagree on the **aggregate sign** (−2.07 pp vs +0.54 pp) while
agreeing on the **stratified** signs (LIST ≈ −8 pp, non-LIST ≈ +1 pp). That disagreement is the
cleanest possible demonstration that the aggregate is not a well-defined quantity.

---

## 1. Step 1 — duplicate check (read-only). RESULT: **no duplicate. K3's exit is still unclaimed.**

Proxy set on separate lines, **positive control run first**:

```
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
curl -sS https://api.github.com/rate_limit          # -> core limit 60, remaining 60  (CONTROL PASSED)
curl -sS https://api.github.com/repos/booydar/babilong
# -> {'full_name':'booydar/babilong','default_branch':'main','open_issues_count':6,
#     'pushed_at':'2026-06-01T18:13:38Z','stargazers_count':253}   (CONTROL PASSED)
```

Both controls returned real data, so the NOT-FOUND results below are valid rather than a silent
proxy failure.

### 1.1 Exhaustive enumeration (`state=all`, so open **and** closed, issues **and** PRs)

`GET /repos/booydar/babilong/issues?state=all&per_page=100` (2 pages) → **18 items = 10 issues + 8
PRs**, which matches the count in `K1_NOVELTY_CHECK.md` §1. Raw JSON persisted at
`k3_precheck/evidence_github_issues_prs.json`. Full list:

| | # | state | created | title |
|---|---|---|---|---|
| ISS | 1 | open | 2024-03-13 | how to show the heatmap on Figure 10 in the paper? |
| PR | 2 | closed | 2024-06-13 | evaluation of popular models on BABILong |
| ISS | 3 | open | 2024-06-13 | Add Claude and Google models into benchmark |
| ISS | 4 | closed | 2024-06-13 | Clarify GPT-4 model |
| ISS | 5 | closed | 2024-11-04 | Leaderboard is down. plz restart the Leaderboard in HF, thx. |
| PR | 6 | closed | 2024-11-14 | add new models evaluations (Gemini, LLama 3.1, 3.2, Qwen 2, 2.5, Phi-3.5, …) |
| ISS | 7 | closed | 2024-11-19 | Cannot run create_tasks.py for qa3 task |
| ISS | 8 | open | 2024-12-19 | Batch evaluations |
| ISS | 9 | closed | 2024-12-25 | Mamba 130m fine-tuning recipe |
| ISS | 10 | closed | 2025-01-03 | Use Babilong training data can impair the model performance |
| ISS | 11 | open | 2025-01-07 | love the benchmark. please test exaone? |
| PR | 12 | closed | 2025-04-04 | add scripts and results for gemma-4, phi-4, xlstm-7b |
| PR | 13 | closed | 2025-05-05 | Feat/babilong evals hf |
| PR | 14 | closed | 2025-05-05 | add scripts for LLama-4-Scout, LLama-3 ProLong, LLama-3.1 UltraLong, … |
| PR | 15 | closed | 2025-05-05 | leaderboard update, new models, and collection of models predictions |
| ISS | 16 | open | 2026-02-05 | Performance mismatch in gemma series model |
| PR | 17 | open | 2026-05-28 | Add BABILong results for Aegyx 0.1 |
| PR | 18 | closed | 2026-05-31 | add babilong results for gemini 3 flash preview |

### 1.2 Keyword scan over **every title, every body, and every comment**

`GET /repos/booydar/babilong/issues/comments?per_page=100` → **20 comments** fetched. Scanned all 18
titles + 18 bodies + 20 comments for: `metrics.py`, `preprocess_output`, `compare_answers`,
`truncat*`, `lowercase`, `lower()`, `split('question`, `first sentence`, `first period`, `scorer`,
`scoring`, `task_labels`, `dead code`, `no-op`, `unreachable`, `exact match`, `postprocess`,
`post-process`, `answer extraction`, `preprocess`.

**3 hits, all 3 read in full and all 3 false positives:**

| where | matched | actual content |
|---|---|---|
| #16 body | `lower()` | Gemma reproduction mismatch; `lower()` appears in the *user's own pasted inference script* (`if 'gemma-3' in model_name.lower()`). The question is about chat template vs SDPA/FA2 attention backend. Zero comments. Nothing about the scorer. |
| #17 body | `scorer`, `scoring` | Aegyx 0.1 results-submission PR. Says "the official BABILong scorer **was not modified**" — i.e. it asserts the scorer is fine, it does not report a defect. |
| #17 comment (Voresot) | `scorer` | Same PR: "scored with the unmodified official BABILong collector/scorer". |

### 1.3 GitHub search API, repo-scoped (each query and its exact result)

| query | `total_count` | hits |
|---|---|---|
| `repo:booydar/babilong metrics.py` | **0** | — |
| `repo:booydar/babilong preprocess_output` | **0** | — |
| `repo:booydar/babilong compare_answers` | **0** | — |
| `repo:booydar/babilong truncation` | **0** | — |
| `repo:booydar/babilong lowercase` | **0** | — |
| `repo:booydar/babilong "first sentence"` | **0** | — |
| `repo:booydar/babilong split` | 1 | ISS #16 (the Gemma one) |
| `repo:booydar/babilong scorer` | 1 | PR #17 (the Aegyx one) |

The last two queries returning **non-zero** is the internal control that the search endpoint was
actually working — the six zeros are real absences, not a broken query path.

### 1.4 Cross-repo search (someone may have reported it elsewhere)

| query | `total_count` | hits |
|---|---|---|
| `babilong preprocess_output` | **0** | — |
| `babilong metrics.py truncation` | **0** | — |
| `babilong split Question lowercase` | **0** | — |
| `preprocess_output split Question` | 1 | `opensoft/dartwing-ocr-pipeline#38` — unrelated OCR pipeline |

### 1.5 Has anyone silently fixed it in a branch or fork?

**No.** All 9 upstream branches enumerated (`main`, `dev`, `feat/babilong_evals_hf`,
`feat/new_models_may_2025`, `feat/start_vllm_server`, `mlspace-0225`, `predictions_06_2024`,
`smollm-eval`, `smollm-evals`); 27 forks enumerated. Fetched `babilong/metrics.py` from `main` of
upstream plus the 4 most-recently-pushed forks (`adam-suliman/babilong-cl`, `RodkinIvan/babilong`,
`Voresot/babilong`, `sylee21/babilong`): **all 5 are byte-identical** (md5
`f0118396c611e5d0b2e440413f1c4a67` as served over HTTP) and **all 5 still have
`output = output.split('Question')[0]` at line 31**.

### 1.6 Verdict on Step 1

**The K3 exit is NOT already satisfied by someone else.** No issue, PR, comment, branch or fork
addresses `metrics.py` / `preprocess_output` / the lowercase-then-split defect / the truncation
behaviour. B11's own upstream-check (carried, unverified, from K1 → flagged in `K1_NARROWING.md` §5
as needing re-checking immediately before filing) is now **re-verified by me, today**.

---

## 2. Step 2 — assertion (a), proven mechanically. VERDICT: **CONFIRMED, but the WORD "unreachable" must be dropped.**

### 2.1 Version and line numbers of record (line numbers drift — these are pinned)

| item | value | how established |
|---|---|---|
| upstream HEAD of `main` | **`7a6efee29f5cac03c3c410e6799c80fd2ffe3610`**, 2026-06-01T18:13:38Z, *"Merge pull request #18 from RodkinIvan/add-results-gemini-3-flash-preview"* | `GET /repos/booydar/babilong/commits/main` |
| `metrics.py` md5 at that HEAD | **`0a5ecc52ade4e337d35b8f9c97c38310`** | fetched `raw.githubusercontent.com/booydar/babilong/7a6efee2…/babilong/metrics.py`; `diff` against the local copy is **empty** |
| local canonical copy | `third_party/babilong-pkg/babilong/metrics.py`, same md5 | `md5sum` |
| local checkout HEAD | `f09a184b43316a751d5059e13de7c557b6daca86` (2025-05-05, PR #15) — **behind upstream**, but `metrics.py` is byte-identical, so line numbers 25/27/31 are the maintainer's | `git log`, plus the `diff` above |
| last commit to touch `metrics.py` | `93d7bfe67ad0` (2025-04-05) | `GET /commits?path=babilong/metrics.py` — agrees with local `git log` |
| commit that INTRODUCED line 31 | **`58e5d20b775b`** (2024-06-04, booydar), *"add split by \"Question\" in preprocess + fix lowercase bug"* | `git blame -L 24,32` |

The introducing commit message is worth quoting in the issue, but **only with the correction
below** — the message misleads about *which* lowercasing it means.

> **CORRECTED 2026-08-16 by MAIN**, from `git show 58e5d20b -- babilong/metrics.py` (2 insertions,
> 1 deletion — the whole diff). An earlier draft of this document, and its ready-to-file issue body,
> said *"the same commit that added the guard also moved/added the lowercasing, which is how the two
> came to collide."* **That is false.** It was inferred from the commit *message* instead of read
> off the *diff*. What `58e5d20b` actually changed:
>
> - `preprocess_output`: **added** `output = output.split('Question')[0]` — the guard, now line 31.
> - `compare_answers`: `label in question` → `label in question.lower()`. **This is the "lowercase
>   bug" the message names, and it is in a different function.**
> - It did **not** touch `output = output.lower()` (now line 25). `git blame -L 25,25` attributes
>   that line to **`1e7893a4`** (yurakuratov, **2024-05-22**) — **13 days earlier**.
>
> The corrected framing is also the stronger one: the lowercasing already existed upstream, and the
> guard was added on top of it by an author who — per their own commit message — was thinking about
> case handling at that very moment and still did not notice. Unlike the false version, this
> survives a maintainer running `git blame`. **Do not restore the deleted sentence.**

**The code, at upstream lines 24–32 of `7a6efee2`:**

```python
24  def preprocess_output(output):
25      output = output.lower()
26      # take only the first sentence from output
27      output = output.split('.')[0]
28      # filter responses when model tries to generate examples
29      output = output.split('<context>')[0]
30      output = output.split('<example>')[0]
31      output = output.split('Question')[0]
32      return output
```

### 2.2 ⚠️ The distinction the record got wrong

`STATUS.json` (`dead_code_recheck_20260814`, `lifecycle_reason`), `K1_NOVELTY_CHECK.md` §3.1 and
`K1_NARROWING.md` §2.1 all call line 31 **"UNREACHABLE" / "dead code"**. **That is false as stated,
and would get an upstream issue closed with "no, it runs."** Measured:

```
$ python3 /apdcephfs_wzc1/.../B11-.../k3_precheck/verify_line31_noop.py
module file : .../third_party/babilong-pkg/babilong/metrics.py
module md5  : 0a5ecc52ade4e337d35b8f9c97c38310
python      : 3.14.6

PART 1 -- is line 31 EXECUTED? (line-level trace)
  input='kitchen Question: Where is Mary? Answer: garden'  lines executed = [25, 27, 29, 30, 31, 32]
  input='kitchen. Question: Where is Mary?'                lines executed = [25, 27, 29, 30, 31, 32]
  input='plain kitchen'                                    lines executed = [25, 27, 29, 30, 31, 32]
  -> line 31 IS executed on every input. It is NOT dead in the control-flow sense.
```

**So the precise claim is: line 31 is EXECUTED on every call and is a GUARANTEED NO-OP.** It is not
never-executed, and it is not unreachable. (This is the same class of error as the standing lesson
*"'the path is unreachable' ≠ 'the line is unreachable'"* — recorded once before in this repo, and
the record still contains the wrong word in three files. The issue body in §4 uses "the guard can
never fire" and carries an explicit terminology note.)

### 2.3 Why it is a no-op — value-level, then exhaustively

```
PART 2 -- is line 31 a provable NO-OP? (value-level)
  in ='kitchen Question: Where is Mary? Answer: garden'
    value entering line 31 = 'kitchen question: where is mary? answer: garden'
    value leaving  line 31 = 'kitchen question: where is mary? answer: garden'
    line 31 changed the value? False
  in ='QUESTION here'                 -> entering/leaving both 'question here'   changed? False
  in ='Question at the very start'    -> entering/leaving both 'question at…'    changed? False
  (the instrumented step-by-step reimplementation is asserted equal to the real
   preprocess_output on every probe, so this is the real code path)

PART 3 -- WHY it can never change the value: str.lower() cannot emit 'Q'
  codepoints whose .lower() contains ASCII 'Q':                          []
  codepoints whose .lower() contains ANY of 'QUESTION':                  []
  multi-char lowerings that still contain an ASCII uppercase:            []
  codepoints where .lower() is NOT idempotent:                           []
  Greek final-sigma context rule check: 'ΟΔΟΣ'.lower() -> 'οδος'

PART 4 -- CONTROL: lines 29/30 DO fire (so the defect is specific to line 31)
  preprocess_output('kitchen <CONTEXT> blah'    ) = 'kitchen '
  preprocess_output('kitchen <EXAMPLE> blah'    ) = 'kitchen '
  preprocess_output('kitchen <context> blah'    ) = 'kitchen '
  preprocess_output('kitchen Question: blah'    ) = 'kitchen question: blah'   <-- guard did not fire

PART 5 -- the ONE-CHARACTER FIX, executed side by side
  in       : 'kitchen Question: Where is Mary? Answer: garden'
    current: 'kitchen question: where is mary? answer: garden'
    fixed  : 'kitchen '
  in       : 'the football is in the kitchen Question: Where is Mary?'
    current: 'the football is in the kitchen question: where is mary?'
    fixed  : 'the football is in the kitchen '

PART 6 -- only in-repo caller of preprocess_output
  metrics.py:24: def preprocess_output(output):
  metrics.py:36: output = preprocess_output(output)
```

The full 0x110000-codepoint sweep is the part that makes this a *proof* and not an observation:
`str.lower()` cannot emit an ASCII uppercase letter anywhere in Unicode, it is idempotent
everywhere, so the chained splits on lines 27/29/30 cannot reintroduce a capital either. The one
context-sensitive rule in `str.lower` (Greek final sigma) also produces no uppercase.

*(Note: this pass ran on Python 3.14.6, not the 3.11.6 in the record. Same result. The argument is
version-independent anyway because it is quantified over all codepoints, and I re-ran the sweep
rather than carrying it.)*

### 2.4 No other reachable path (fixing the class, not the instance)

- `.lower()` at line 25 is the function's **first statement and unconditional** — no argument can
  bypass it.
- `git grep preprocess_output` over the **whole upstream repo at HEAD** → exactly **2 hits**: the
  definition (`metrics.py:24`) and the single call site (`metrics.py:36`, inside `compare_answers`).
  No notebook calls `preprocess_output` directly; the 14 eval notebooks all call `compare_answers`.
- A **second, independent scorer** exists upstream: `babilong/babilong_utils.py:8
  compare_answers(target, output)`. Read in full — it lowercases, truncates at the first period and
  splits `<context>`/`<example>`, but has **no `'Question'` guard at all**. So it is *unaffected by
  point 1* rather than an alternative path, and it **shares point 2** (same `split('.')[0]`). The
  issue body says so, because a maintainer will otherwise point at it.

---

## 3. Step 3 — assertion (b), two-armed and STRATIFIED. VERDICT: **CONFIRMED, and the sign-dependence is stronger than recorded.**

**Design.** Arm CANON = `babilong.metrics.compare_answers` as shipped. Arm NOTRUNC = identical except
line 27 removed. Uniqueness requirement (`len(labels_in_output)==1`) retained in **both**, so
choice-lists still score 0 and there is no chance inflation. Generations are byte-fixed: both arms
read the same CSVs. Effect of truncation on an item = `canon − notrunc` (+1 = truncation **rescued**
a correct answer, −1 = truncation **destroyed** one).

### 3.1 First, a bit-exactness check on the recorded ablation

I re-ran the record's own `analyze_a02_truncation_ablation.py` from the raw per-item CSVs, on a
different node and a different disk from the original run, and diffed the output JSON against the
committed evidence file leaf-by-leaf:

```
total differing leaves: 0
BIT-EXACT: True
md5 new json: db850dd98331ad9c8b1eaf47f7061af0
md5 old json: db850dd98331ad9c8b1eaf47f7061af0
```

**Bit-exact** (`proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder/a02_truncation_ablation.json`).
So the 2/6 repair count, the ρ = −1.000 / p = 0.0167 qa1×32k ladder, and the McNemar p = 0.0703 are
reproducible from raw. Confirms the standing "same-harness runs are bit-identical" expectation, and
this one crossed disks (raw CSVs live on zwfy6; I copied them to wzc1 and computed locally, 0 GPU).

### 3.2 Corpus A — the 6 A02 cells, 5 arms, n = 3000 items (100 per arm-cell, no subsetting)

Provenance: `babilong_results/a02_{dvr_babilong_j0_top12, rtax_babilong_A2_j6, rtax_babilong_A3_j9,
babilong_c2_j12_readlora, rtax_babilong_A5_j18}/qa{1,2,5}_{16k,32k}_*shard{0..7}of8.csv` on zwfy6
(8/8 shards asserted present for every arm × cell; 100 items asserted per arm × cell; 3000 = 5×6×100
asserted). Script: `k3_precheck/stratify_line27_a02cells.py`.

**The stratum that carries the mechanism — output format habit:**

| stratum | n | destroyed | rescued | **net effect of keeping truncation** | canon acc | notrunc acc |
|---|---|---|---|---|---|---|
| **ALL (aggregate)** | 3000 | 101 | 39 | **−2.07 pp** | 28.57 | 30.63 |
| **is_list = 1** (choice-list habit) | **1109** | 92 | 3 | **−8.03 pp** | 1.17 | 9.20 |
| **is_list = 0** (non-list) | **1891** | 9 | 36 | **+1.43 pp** | 44.63 | 43.20 |

**Sign differs between the strata: TRUE.** And the aggregate takes the LIST sign, i.e. it is the
LIST stratum (37% of items) that drives the headline while the majority stratum points the other way.

**Does the sign follow FORMAT or merely follow TASK?** It follows format — within every task:

| stratum | n | destroyed | rescued | net |
|---|---|---|---|---|
| qa1, is_list=1 | 577 | 62 | 3 | **−10.23 pp** |
| qa1, is_list=0 | 423 | 2 | 10 | **+1.89 pp** |
| qa2, is_list=1 | 501 | 25 | 0 | **−4.99 pp** |
| qa2, is_list=0 | 499 | 0 | 2 | **+0.40 pp** |
| qa5, is_list=1 | **31** | 5 | 0 | **−16.13 pp** *(n=31, small)* |
| qa5, is_list=0 | 969 | 7 | 24 | **+1.75 pp** |

3/3 tasks show the sign flip **within** the task, so it is not a task effect wearing a format
costume. Per task the marginal is qa1 −5.10, qa2 −2.30, qa5 **+1.20** (n=1000 each) — the qa5 sign
reversal is exactly the "removing truncation *lowers* qa5" observation in the record, re-derived.

**And within every arm** (this is why it can move an arm ordering):

| arm | is_list=1 net (n) | is_list=0 net (n) | list-format rate on qa1/qa2 cells |
|---|---|---|---|
| A0 (j=0) | −6.42 (187) | +2.42 (413) | 37–54% |
| A2 (j=6) | −8.37 (239) | +0.83 (361) | 53–67% |
| A3 (j=9) | −6.37 (267) | +1.20 (333) | 59–75% |
| A4 (j=12) | −9.09 (275) | +0.62 (325) | 60–75% |
| A5 (j=18) | −10.64 (141) | +1.74 (459) | 25–42% |

5/5 arms flip sign across the format strata. The arms differ in **how many** items land in the
penalised stratum (A4 275 vs A5 141), not in the within-stratum sign — which is precisely the
"format-correlated offset between two configurations" mechanism.

Per-cell list-format rate (%), the covariate:

| cell | A0 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| qa1\|16k | 48.0 | 58.0 | 65.0 | 62.0 | 31.0 |
| qa1\|32k | 54.0 | 67.0 | 75.0 | 75.0 | 42.0 |
| qa2\|16k | 42.0 | 56.0 | 59.0 | 60.0 | 25.0 |
| qa2\|32k | 37.0 | 53.0 | 63.0 | 74.0 | 32.0 |
| qa5\|16k | 2.0 | 2.0 | 2.0 | 2.0 | 6.0 |
| qa5\|32k | 4.0 | 3.0 | 3.0 | 2.0 | 5.0 |

### 3.3 Corpus B — the FULL corpus, deterministic. This **supersedes** the record's sampled figure.

`STATUS.json` `assertion_b.empirical_sign_split_measured_this_gate` reports **500 randomly-sampled**
CSVs / 46 241 items → LIST **−8.86 pp** (n=429) vs non-LIST **+0.25 pp** (n=45 812), overall
+0.16 pp. That number is listed under `k3_exit_MAIN_independent_verification_20260815.not_verified_by_MAIN`.
**I did not reproduce it and I am not quoting it.** Instead I removed the sampling: every `qa*.csv`
under `babilong_results/` on **both disks**, content-deduplicated.

- 5372 CSVs found (973 wzc1 + 4399 zwfy6) → 1114 content-identical duplicates removed (the two disks
  share history) → **4258 unique CSVs**, 18 skipped (unknown task / wrong schema), **0 rows skipped**.
- **n = 195 064 items.** Script: `k3_precheck/stratify_line27_fullcorpus.py`; `is_list_format()`
  copied verbatim from the A02 script.

| stratum | n | destroyed | rescued | **net effect of keeping truncation** | canon | notrunc |
|---|---|---|---|---|---|---|
| **ALL (aggregate)** | 195 064 | 979 (0.50%) | 2039 (1.05%) | **+0.54 pp** | 23.95 | 23.41 |
| **is_list = 1** | **6 337** | 558 (8.81%) | 18 (0.28%) | **−8.52 pp** | 1.86 | 10.38 |
| **is_list = 0** | **188 727** | 421 (0.22%) | 2021 (1.07%) | **+0.85 pp** | 24.69 | 23.84 |

Per task (n = 195 064 across 10 tasks): qa1 +1.08, qa2 +1.04, qa3 +0.01, qa4 +0.14, qa5 +0.10,
qa6 −0.12, qa7 −0.05, qa8 −0.08, qa9 −0.55, qa10 +0.00 — **the sign is not even constant across
tasks in the aggregate view.**

### 3.4 ⭐ The finding that makes assertion (b) sharper than the record claimed

| | A02 6 cells (n=3000) | full corpus (n=195 064) |
|---|---|---|
| **aggregate net** | **−2.07 pp** | **+0.54 pp** |
| LIST net | −8.03 pp (n=1109) | −8.52 pp (n=6337) |
| non-LIST net | +1.43 pp (n=1891) | +0.85 pp (n=188 727) |
| LIST share of items | **37.0 %** | **3.2 %** |

**The two corpora disagree on the aggregate sign while agreeing on both stratified signs.** The
aggregate is a *mixture* weighted by how many list-format items happen to be in the pool — 37% in
the A02 cells (Qwen3-8B on qa1/qa2, which elicit enumeration) vs 3.2% corpus-wide. This is a
demonstration, not an assertion, that **the aggregate net effect of line 27 is not a property of the
metric; it is a property of the answer-format mix of whatever you point it at.** That is exactly the
"sign-dependent trade-off" claim, and it is the strongest form of it B11 has.

Recomputation status, stated plainly as required:

- The record's **A02 6-cell ablation** → re-derived from raw CSVs, **BIT-EXACT** (md5 match, 0
  differing leaves).
- The record's **stratified −8.86 / +0.25 pp** → **NOT bit-exact and not attempted**: it came from a
  *random 500-CSV sample* with no recorded seed or file list, so it is not reproducible by
  construction. My deterministic full-corpus replacement gives **−8.52 / +0.85 pp** — same signs,
  same order of magnitude on the LIST stratum, and it removes the sampling caveat that
  `not_verified_by_MAIN` attached to the old pair. **The issue body quotes my numbers, not the
  sampled ones.**

### 3.5 Every demo verified by execution (`k3_precheck/verify_issue_demos.py`, all asserted)

| demo | canon | notrunc | exhibits |
|---|---|---|---|
| `"kitchen Question: where is Mary? Answer: garden"` | **False** | — | point 1: dead guard → false negative; with the 1-char fix → **True** |
| `"The answer is A. kitchen"` | False | **True** | line 27 destroys an *enumerated* correct answer (`-> 'the answer is a'`) |
| `"John moved several times. He is in the kitchen"` | False | **True** | line 27 destroys a *reason-then-answer* correct answer |
| `"kitchen. Question: Where is Mary? Answer: garden"` | **True** | False | line 27 *rescues* a scaffold-leaking correct answer |
| `"kitchen is wrong. the answer is garden"` | **True** | False | line 27 **manufactures** a correct answer (false positive) |

**NEGATIVE CONTROL, and the reason this section exists.** The string
`"Choices: A. In the kitchen B. In the garden. The answer is kitchen."`, which
`K1_NOVELTY_CHECK.md` §3.2 and `K1_NARROWING.md` §2.2 both present as *"truncation kills a correct
list-format answer"*, **executes to canon=False AND notrunc=False**. It dies on the **uniqueness**
requirement (both `kitchen` and `garden` survive without truncation), **not** on truncation. It does
not demonstrate the claimed mechanism and is **excluded** from the issue body. `STATUS.json`
`assertion_b.row1_DOES_NOT_REPRODUCE` already caught this; I re-executed it rather than trusting the
catch, and it is confirmed. Every string in §4 is asserted to exhibit its claimed mechanism.

---

## 4. FINAL READY-TO-FILE ISSUE

Title:

```
metrics.py: preprocess_output's split('Question') guard can never fire (line 25 lowercases first)
```

Body:

```markdown
### Summary

Two separate points about `babilong/metrics.py`, both reproducible from a clean clone with no extra
data and no GPU:

1. **A defect.** `preprocess_output`'s guard at line 31, `output.split('Question')[0]`, can never
   fire, because line 25 has already lowercased the string. It is the only guard in the function
   against post-answer prompt-scaffold leakage. One-character fix, but it is **not** score-neutral.
2. **A design question, deliberately not filed as a bug.** Line 27's first-period truncation has a
   **format-dependent sign**: it removes correct enumerated / reason-then-answer answers, rescues
   correct scaffold-leaking ones, and can manufacture false positives. We are **not** proposing to
   remove it — we are flagging that the net effect on a score depends on the answer-format habit of
   the model being scored, so it can act as a systematic offset between two systems that differ in
   output style rather than in task ability.

### Environment / provenance

- Repo `booydar/babilong`, branch `main`, HEAD **`7a6efee29f5cac03c3c410e6799c80fd2ffe3610`**
  (2026-06-01, *"Merge pull request #18 …"*).
- `babilong/metrics.py`, md5 **`0a5ecc52ade4e337d35b8f9c97c38310`** at that HEAD; line numbers below
  are that file's.
- `metrics.py` last changed in `93d7bfe67ad0` (2025-04-05). Line 31 was added in
  **`58e5d20b775b`** (2024-06-04), *"add split by \"Question\" in preprocess + fix lowercase bug"*.
  Note that the "lowercase bug" fixed in that same commit is a **different** one — `label in
  question` → `label in question.lower()` inside `compare_answers`. The `output = output.lower()`
  on line 25 predates it: `git blame` attributes that line to `1e7893a4` (2024-05-22), 13 days
  earlier. So the guard was added on top of an already-existing lowercasing.
- Verified on Python 3.14.6. The argument in point 1 is version-independent (it is quantified over
  all Unicode codepoints).

### The code

```python
# babilong/metrics.py, lines 24-32 at 7a6efee2
24  def preprocess_output(output):
25      output = output.lower()
26      # take only the first sentence from output
27      output = output.split('.')[0]
28      # filter responses when model tries to generate examples
29      output = output.split('<context>')[0]
30      output = output.split('<example>')[0]
31      output = output.split('Question')[0]
32      return output
```

---

## Point 1 — line 31's guard can never fire

### Terminology first, to avoid a false start

Line 31 **is executed** on every call — a line-level trace of `preprocess_output` shows lines
`[25, 27, 29, 30, 31, 32]` all running. This is **not** a report of unreachable code. The claim is
that the line is reached and **can never change the value**: it is a guaranteed no-op.

### Minimal reproduction

```python
from babilong.metrics import preprocess_output

s = "kitchen Question: Where is Mary? Answer: garden"
print(repr(preprocess_output(s)))
# actual:   'kitchen question: where is mary? answer: garden'
# intended: 'kitchen '
```

The reproduction string deliberately contains **no period before `Question`**, so line 27 cannot
mask the problem. (With a period present, line 27 happens to truncate at the same place and the dead
guard is invisible — which is probably why this has gone unnoticed.)

Control showing lines 29/30 **do** fire, i.e. the defect is specific to line 31:

```python
print(repr(preprocess_output("kitchen <CONTEXT> blah")))   # -> 'kitchen '
print(repr(preprocess_output("kitchen <EXAMPLE> blah")))   # -> 'kitchen '
print(repr(preprocess_output("kitchen Question: blah")))   # -> 'kitchen question: blah'  <-- no-op
```

### Why "can never fire" is a proof, not an observation

`str.lower()` cannot emit an ASCII uppercase letter, so no input can produce the capital `Q` line 31
needs. Brute-forced over the whole Unicode range:

```python
print([cp for cp in range(0x110000) if 'Q' in chr(cp).lower()])
# []
print([cp for cp in range(0x110000) if any(u in chr(cp).lower() for u in 'QUESTION')])
# []
# no codepoint lowers to a multi-character string that still contains an ASCII uppercase:
print([(hex(cp), chr(cp).lower()) for cp in range(0x110000)
       if len(chr(cp).lower()) > 1 and any('A' <= ch <= 'Z' for ch in chr(cp).lower())])
# []
# str.lower is idempotent everywhere, so the chained splits cannot reintroduce one:
print([cp for cp in range(0x110000) if chr(cp).lower().lower() != chr(cp).lower()])
# []
```

The one context-sensitive rule in `str.lower` (Greek final sigma) also yields no uppercase:
`'ΟΔΟΣ'.lower() == 'οδος'`.

There is no alternative path: the `.lower()` is the function's own first statement and is
unconditional, and `preprocess_output` has exactly one definition and exactly one caller
(`compare_answers`, line 36) — `git grep preprocess_output` returns 2 hits repo-wide, and the eval
notebooks all go through `compare_answers`. (The separate legacy scorer
`babilong/babilong_utils.py:8 compare_answers` has no `'Question'` guard at all, so it is unaffected
by this point.)

### Why it matters: the guard is the only defence against scaffold leakage, and enabling it changes scores

Line 31 is the only protection against a model that keeps generating after its answer and re-emits
the `Question: … Answer: …` prompt scaffold — common for base / non-instruction-tuned models
evaluated without a stop string. Executed, with the one-character fix applied:

```python
from babilong.metrics import TASK_LABELS, compare_answers
L = TASK_LABELS['qa1']; Q = "Where is John?"
raw = "kitchen Question: where is Mary? Answer: garden"

compare_answers("kitchen", raw, Q, L)
# False   -- preprocess_output leaves 'kitchen question: where is mary? answer: garden',
#            so both 'kitchen' and 'garden' survive and the uniqueness test rejects it
# with line 31 changed to split('question'), preprocess gives 'kitchen ' and the same call -> True
```

So the dead guard produces **false negatives**, and fixing it is **not score-neutral**. It would be
reasonable to fix it behind a flag, or to fix it and re-run the leaderboard, but silently fixing it
would make new numbers incomparable to old ones — that call is yours, not ours.

### Suggested fix

One character of case, matching the style of lines 29/30:

```diff
-    output = output.split('Question')[0]
+    output = output.split('question')[0]
```

A case-insensitive variant that also catches the `Answer:` scaffold would be more robust, but is a
behaviour change rather than a bug fix:

```python
import re
output = re.split(r'(?i)\bquestion\b', output)[0]
```

---

## Point 2 — line 27's first-period truncation has a format-dependent sign (design question, not a bug)

We are **not** reporting line 27 as a defect and **not** proposing to delete it. The point is that
its effect on a score has **no fixed sign**, and the sign is tied to the model's answer format
rather than to task ability.

`compare_answers` requires the gold target to be the **unique** surviving task label, and line 27
keeps only the text before the first period. Together, all four of the following are executed
against the unmodified package (`target='kitchen'`, `question='Where is John?'`, which contains no
task label):

**It removes correct answers** whose verdict sits after a period:

```python
compare_answers("kitchen", "The answer is A. kitchen", Q, L)
# False   -- preprocess_output -> 'the answer is a'

compare_answers("kitchen", "John moved several times. He is in the kitchen", Q, L)
# False   -- preprocess_output -> 'john moved several times'
```

**It rescues correct answers** when the model leaks scaffold afterwards — line 27 doing, by accident,
the job line 31 was supposed to do:

```python
compare_answers("kitchen", "kitchen. Question: Where is Mary? Answer: garden", Q, L)
# True    -- preprocess_output -> 'kitchen'
```

**It can manufacture a correct answer the model did not give:**

```python
compare_answers("kitchen", "kitchen is wrong. the answer is garden", Q, L)
# True    -- preprocess_output -> 'kitchen is wrong'; the model's actual answer was 'garden'
```

So the two directions are not symmetric-but-harmless: the harmful direction penalises
enumerating and reasoning-then-answering styles, and the helpful direction includes a false-positive
mode.

### The consequence, measured

Re-scoring **stored generations** with line 27 removed and **nothing else changed** (the uniqueness
requirement is kept, so choice-lists still score 0 and no chance inflation is introduced), over
195 064 items from 4258 prediction CSVs of our own past BABILong runs (many models and
configurations, `qa1`–`qa10`), and stratifying by whether the output is a choice-list enumeration:

| stratum | n | line 27 destroys a correct answer | line 27 rescues one | net effect of keeping line 27 |
|---|---|---|---|---|
| choice-list outputs | 6 337 | 558 (8.81%) | 18 (0.28%) | **−8.52 pp** |
| non-list outputs | 188 727 | 421 (0.22%) | 2 021 (1.07%) | **+0.85 pp** |
| all items pooled | 195 064 | 979 (0.50%) | 2 039 (1.05%) | +0.54 pp |

The sign is **opposite** in the two strata, so the pooled number is a mixture whose sign depends on
how many enumerated answers happen to be in the pool. On a different pool of ours where 37% of
outputs are enumerations (rather than 3.2% corpus-wide), the same computation gives a pooled
**−2.07 pp** — opposite sign to the table above, with both stratified signs unchanged
(−8.03 pp list / +1.43 pp non-list). Per-task pooled values also change sign
(qa1 +1.08, qa2 +1.04, qa9 −0.55).

We are not asking you to treat those numbers as a benchmark result — they come from a heterogeneous
mix of our own historical runs, not a controlled comparison, and we are reporting them only to
establish the **sign-dependence**, never as an effect size for any model comparison. Note also that
the legacy `babilong_utils.compare_answers` has the same `split('.')[0]`, so this point applies to
both scorers.

### What we would suggest, in increasing order of invasiveness

1. Fix point 1 first (`'Question'` → `'question'`), so scaffold suppression no longer depends on
   line 27 as a side effect. That alone makes line 27's role smaller and easier to reason about.
2. Document line 27 in the README / paper appendix as a scoring convention that penalises
   multi-sentence and enumerated answers, so users can interpret comparisons between
   differently-styled models correctly.
3. Optionally expose a strictness flag so results can be reported both ways.

---

## Summary

| # | Line | Nature | Status |
|---|------|--------|--------|
| 1 | `metrics.py:31` | `split('Question')` can never fire because line 25 lowercases first; it is the only guard against post-answer scaffold leakage. Executed, but a guaranteed no-op | defect, one-character fix, **changes published numbers** |
| 2 | `metrics.py:27` | first-period truncation has a format-dependent sign: removes correct enumerated/reasoned answers, rescues scaffold-leaking ones, can manufacture false positives | design question / documentation, **not** proposed for removal |

Happy to open a PR for point 1 (with or without a compatibility flag) if you tell us which behaviour
you want to preserve. Point 2 we would rather leave to your judgement, since any change there affects
the leaderboard.
```

### 4.1 What changed relative to `UPSTREAM_ISSUE_DRAFT.md` (2026-08-15)

| # | change | why |
|---|---|---|
| 1 | Title kept; body's point-1 heading changed from "line 31 is a guaranteed no-op (a defect)" to "line 31's guard can never fire", and the terminology note **moved to the top** of point 1 | a maintainer who reads "no-op/dead" first and the note last may close it as wrong; the disclaimer has to precede the claim |
| 2 | Upstream HEAD short hash `7a6efee29f5c` → **full 40-char sha** `7a6efee29f5cac03c3c410e6799c80fd2ffe3610`; added the `93d7bfe67ad0` / `58e5d20b775b` provenance and quoted the introducing commit message | line numbers drift; the full sha + md5 pins them, and the commit message shows the collision |
| 3 | Draft said "Python 3.11.6" → **3.14.6** | that is what I actually ran; the record's 3.11.6 was a different pass |
| 4 | Point-1 score demo made explicit as a **fix-vs-no-fix pair** with the intermediate `preprocess_output` value shown | the draft asserted "not score-neutral"; this executes it |
| 5 | Point-2 numbers replaced: draft had **no** quantitative table; the record's sampled −8.86/+0.25 pp is **not used** | that pair is `not_verified_by_MAIN` and came from an unseeded random 500-CSV sample, hence unreproducible. Replaced with the deterministic full-corpus **−8.52 / +0.85 pp (n=195 064)** plus the second-pool **−2.07 pp** sign-reversal, which is the actual argument |
| 6 | Added: the legacy `babilong_utils.compare_answers` shares `split('.')[0]`, so point 2 applies there too | pre-empts "that's just one scorer" |
| 7 | Removed the internal pre-filing checklist from the issue text | it was marked "delete before filing" and contains internal state |
| 8 | The `"Choices: A. In the kitchen B. …"` string is **absent** from the issue body | executed: canon=False **and** notrunc=False. It shows the uniqueness requirement, not truncation. It would be refuted on first execution |

### 4.2 Pre-filing checklist (internal — not part of the issue text)

- [x] Upstream HEAD (`7a6efee29f5cac03c3c410e6799c80fd2ffe3610`) recorded; `metrics.py` fetched from
      that exact sha and `diff`ed byte-for-byte against the local copy → identical, so lines
      25/27/31 are the maintainer's line numbers.
- [x] All 18 issues+PRs (open **and** closed) enumerated, plus all 20 repo comments; 3 keyword hits
      read in full, all 3 false positives. 12 search-API queries run (8 repo-scoped, 4 cross-repo);
      2 non-zero results serve as the endpoint's positive control.
- [x] 9 upstream branches + 27 forks enumerated; the 4 most active forks' `metrics.py` fetched and
      confirmed byte-identical with line 31 unchanged. Nobody has fixed it.
- [x] **Every** snippet and demo in the body executed against the unmodified package and
      **asserted** to exhibit its claimed mechanism (`k3_precheck/verify_issue_demos.py`).
- [x] The one demo that does **not** exhibit its claimed mechanism identified and excluded.
- [x] Body contains no internal paths, node addresses, arm names (A0–A5), depth-knob values,
      proposal IDs, or unpublished arm-comparison numbers. The only numbers are the format-stratified
      ones, with their heterogeneity caveat attached in-line.
- [x] Point 2 is phrased as a design question and explicitly does **not** propose deleting line 27
      (which the record lists as a forbidden claim).
- [ ] **Approval to file — NOT GRANTED. Requires explicit human sign-off.**

---

## 5. Status, and what this does NOT do

**The issue has NOT been filed.** No issue was created, no PR was opened, no comment was posted, no
fork was pushed. Every GitHub call in this pass was a read: `GET /repos/booydar/babilong`,
`GET .../issues?state=all`, `GET .../issues/comments`, `GET /search/issues`, `GET .../branches`,
`GET .../forks`, `GET .../commits/main`, `GET .../commits?path=babilong/metrics.py`, and
`raw.githubusercontent.com` file fetches. **Filing publishes content to a third-party repository
under the project's identity and is not reversible; it requires explicit human sign-off, which this
pass does not have.** The body in §4 is intended to be copy-pasteable with no further edits once
that sign-off exists.

### 5.1 Honest gaps

1. **Assertion (a)'s wording is still wrong in three files.** `STATUS.json`
   (`dead_code_recheck_20260814.claim`, `lifecycle_reason`), `K1_NOVELTY_CHECK.md` §3.1 and
   `K1_NARROWING.md` §2.1 say **"unreachable" / "dead code"**. Measured: line 31 executes on every
   call. Those files are dated records and are not edited in place; **anything downstream must use
   "the guard can never fire"**, and the issue body does.
2. **The record's sampled −8.86 / +0.25 pp is not reproducible and remains unverified.** It came
   from a random 500-CSV sample with no recorded seed or file list. I did not attempt to reproduce
   it; I replaced it with a deterministic full-corpus measurement. The old pair should be treated as
   superseded, not as independently confirmed.
3. **Corpus B is heterogeneous by construction.** 4258 CSVs from many models/configs across the
   project's history. It supports the **sign-dependence** and nothing else — it is not a controlled
   arm contrast and must never be quoted as an effect size for a model comparison. This caveat is
   carried inside the issue body itself.
4. **No significance test on the stratified split.** The signs are reported with n per stratum and
   raw destroyed/rescued counts. I did not attach p-values, because the per-item outcomes are
   clustered by CSV/model/config and a naive per-item test would understate the variance. The claim
   made is "the sign differs between strata in both corpora", which the counts support directly
   (LIST 558 destroyed vs 18 rescued; non-LIST 421 vs 2021).
5. **qa5's list stratum in corpus A is n=31.** Its −16.13 pp is reported with the n visible and is
   not load-bearing; the qa1/qa2 list strata (n=577, n=501) carry the within-task result.
6. **Upstream's own results were not re-scored.** I measured line 27's effect on *our* stored
   generations. I did not download the 400+ prediction CSVs in upstream `babilong_evals/` and rescore
   the published leaderboard. That would strengthen point 2 for a maintainer and is a natural
   follow-up, but it changes the issue from "here is a property of your metric" into "here is what
   your published numbers should have been", which is a much larger claim and needs its own review.
7. **K2 (cross-family replication) is untouched and still blocking promotion.** This pass is the K3
   exit only. It does not make B11 a paper, and `K1_NARROWING.md` §3.1's recommendation stands: B11's
   scientific content is two code facts plus a 6-cell measurement on one model family.

### 5.2 Artefacts written by this pass

| path | what |
|---|---|
| `k3_precheck/verify_line31_noop.py` | the (a) proof: settrace, value-level, Unicode sweep, 29/30 control, 1-char fix, caller enumeration |
| `k3_precheck/stratify_line27_a02cells.py` | (b) on the 6 A02 cells, n=3000, stratified 7 ways |
| `k3_precheck/stratify_line27_fullcorpus.py` | (b) deterministic full corpus, n=195 064, 4258 CSVs, both disks |
| `k3_precheck/verify_issue_demos.py` | every issue-body demo, executed **and asserted**, incl. the negative control |
| `k3_precheck/evidence_stratified_a02cells.json` | corpus-A numbers |
| `k3_precheck/evidence_stratified_fullcorpus.json` | corpus-B numbers |
| `k3_precheck/evidence_github_issues_prs.json` | raw API JSON for all 18 issues+PRs (the duplicate-scan input) |

All four scripts are 0-GPU and re-runnable. `stratify_line27_a02cells.py` needs the A02 raw CSVs
(zwfy6-resident; copy to a local dir and point `A02_W` at it).
`stratify_line27_fullcorpus.py` has the two corpus roots at the top.
