<!-- ⛔ STATUS: DRAFT. NOT FILED. Filing to a third-party public tracker is an outward-facing,
     hard-to-reverse action and requires explicit human sign-off. Nothing in this pass was posted:
     only read-only GitHub endpoints (GET /repos, /issues, /issues/comments, /search/issues,
     /branches, /forks, raw.githubusercontent.com) were called. -->

# B11 / K3-EXIT — upstream issue for `booydar/babilong`, ready to paste

**Date 2026-08-16 · GPU spent: ZERO** (no node contacted, no CUDA device touched; all work is CPU
string operations on files already on disk, plus read-only HTTPS through `hy-proxy`).

This file supersedes nothing: `UPSTREAM_ISSUE_DRAFT.md` (08-15) and
`K3_EXIT_PRECHECK_20260816.md` (08-16 earlier) are kept as the dated record. What is new here is
(1) an independent re-derivation of both claims from source by execution, (2) a **correction to the
earlier precheck's corpus measurement**, and (3) two `md5` / counting discrepancies in the record
that are resolved below rather than inherited.

---

## 0. Headline

| | claim as worded in B11's `next_gate` | verdict of my executed check |
|---|---|---|
| **A** | "line 31 `split('Question')` is **unreachable** because line 25 lowercases first" | **PARTLY REFUTED.** The *substance* survives and is provable. The **word "unreachable" is false and must not be filed**: line 31 is **executed on every single call** (measured, `sys.settrace`, 7/7 inputs). The correct claim is *"the guard is executed but can never fire"* — a **guaranteed no-op**. |
| **B** | "line 27's first-period truncation is a sign-dependent trade-off, not a bug" | **SURVIVED**, in a real two-arm comparison, in **both directions**, on hand cases *and* on 418 367 stored items. |
| **duplicate?** | "re-check that upstream still has no issue on this" | **NO DUPLICATE. The exit is still unclaimed.** 10 issues + 8 PRs (open *and* closed) + 20 comments scanned; 9 branches + 27 forks checked; 11 search queries. |

**Also found this pass — a defect in our own earlier precheck, not in the upstream code:** the
08-16 precheck states it measured "**every** `qa*.csv` under `babilong_results/`" and "removed the
sampling", but its glob is `os.path.join(root,"*","qa*.csv")` — **non-recursive**, so it saw
**973 of the 10 886** files present on wzc1. Its `n = 195 064` is therefore not the full corpus it
claims to be. My recursive re-run gives **n = 418 367** and the **same stratified signs with a
larger magnitude** (LIST **−12.71 pp** vs non-LIST **+1.71 pp**). The conclusion is unchanged; the
number quoted in the issue body is mine, and the claim "full corpus" is now true.

---

## 1. Duplicate check — the exit is still unclaimed

Proxy, **positive control first**, so that "no results" is distinguishable from "proxy broken":

```
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
curl -sS https://api.github.com/rate_limit             # HTTP=200, limit 60 remaining 60   CONTROL PASSED
curl -sS https://api.github.com/repos/booydar/babilong # full_name booydar/babilong, open_issues_count 6,
                                                       # pushed_at 2026-06-01T18:13:38Z, stars 253  CONTROL PASSED
```

- **`GET /issues?state=all&per_page=100`** (2 pages) → **18 records = 10 issues + 8 PRs**, i.e. open
  **and** closed. (Note: `open_issues_count: 6` in the repo object counts open PRs too; and the
  record's phrase "18 issues" is really 18 *records*. Both are bookkeeping, neither changes the
  conclusion.)
- **`GET /issues/comments?per_page=100`** → **20 comments**, also scanned.
- Keyword scan over all 18 titles + 18 bodies + 20 comments for `metrics.py`, `preprocess_output`,
  `compare_answers`, `truncat*`, `lowercase`, `lower()`, `split('question`, `first sentence`,
  `first period`, `scorer`, `scoring`, `task_labels`, `dead code`, `no-op`, `unreachable`,
  `exact match`, `postprocess`, `answer extraction`, `preprocess`, `split('.')`
  → **4 hits, all read in full, all 4 false positives**:

| where | matched | what it actually is |
|---|---|---|
| ISS #16 body | `lower()` | Gemma reproduction question. `lower()` occurs 8× **inside the user's own pasted inference script** (`if 'gemma-3' in model_name.lower()`). Topic is chat template vs SDPA/FA2 backend. |
| PR #17 body | `scorer`, `scoring` | Aegyx 0.1 results submission. Says *"The official BABILong scorer **was not modified**"* — it asserts the scorer is fine, it does not report a defect. |
| PR #17 comment | `scorer` | Same PR: *"scored with the unmodified official BABILong collector/scorer"*. |

- **Search API**, repo-scoped — the two non-zero rows are the endpoint's positive control, so the
  six zeros are real absences:

| query | `total_count` |
|---|---|
| `repo:booydar/babilong metrics.py` | **0** |
| `repo:booydar/babilong preprocess_output` | **0** |
| `repo:booydar/babilong compare_answers` | **0** |
| `repo:booydar/babilong truncation` | **0** |
| `repo:booydar/babilong lowercase` | **0** |
| `repo:booydar/babilong "first sentence"` | **0** |
| `repo:booydar/babilong split` | 1 → ISS #16 (the Gemma one) ← control |
| `repo:booydar/babilong scorer` | 1 → PR #17 (the Aegyx one) ← control |

- Cross-repo: `babilong preprocess_output` → **0**; `babilong metrics.py truncation` → **0**.
  (A third cross-repo query hit the 60/h unauthenticated rate limit and is not counted as a zero.)
- **Has anyone quietly fixed it?** No. 9 branches (`main`, `dev`, `feat/babilong_evals_hf`,
  `feat/new_models_may_2025`, `feat/start_vllm_server`, `mlspace-0225`, `predictions_06_2024`,
  `smollm-eval`, `smollm-evals`) and 27 forks enumerated. `babilong/metrics.py` fetched from `main`,
  from `dev`, from the pinned HEAD sha, and from the 4 most-recently-pushed forks
  (`adam-suliman/babilong-cl`, `RodkinIvan/babilong`, `Voresot/babilong`, `sylee21/babilong`):
  **all 7 fetches are md5 `0a5ecc52ade4e337d35b8f9c97c38310`** and all still carry
  `output = output.split('Question')[0]` at line 31.

> **Discrepancy resolved.** The earlier precheck reports the fork `metrics.py` md5 as
> `f0118396c611e5d0b2e440413f1c4a67` while reporting `0a5ecc52…` for the same, supposedly
> byte-identical, file two sections later. I reproduced the second hash: `f0118396…` is the md5 of
> **the same bytes with the trailing newline stripped** (a `$(curl …)` command-substitution
> artifact, since `$( )` eats trailing newlines). Verified: `md5(raw) = 0a5ecc52…`,
> `md5(raw.rstrip(b"\n")) = f0118396…`. Same file, two hashing procedures. **`0a5ecc52…` is the
> hash of the file as served and as stored**, and is the one quoted in the issue body.

---

## 2. Provenance of the bytes under test

| item | value | how established |
|---|---|---|
| upstream HEAD of `main` | `7a6efee29f5cac03c3c410e6799c80fd2ffe3610` (2026-06-01, *"Merge pull request #18 …"*) | `GET /repos/booydar/babilong/commits/main` |
| `metrics.py` md5 at HEAD | `0a5ecc52ade4e337d35b8f9c97c38310` | fetched from `main`, from the pinned sha, and from `dev`; all three identical |
| local copy under test | `third_party/babilong-pkg/babilong/metrics.py`, same md5 | `md5sum`; `diff` against the upstream fetch is **empty** |
| second local clone | `…/pighzliu_code/babilong/babilong/metrics.py`, same md5, HEAD `f09a184` | `md5sum` |
| ⇒ line numbers | **25 / 27 / 31 are the maintainer's own line numbers** | byte-identity, plus `inspect.getsourcelines` asserting the text of each of the three lines |
| line 31 introduced by | `58e5d20b775bd111c52c4a227b8e8883ecd0d683` (booydar, 2024-06-04) | `git log -S"split('Question')"`, `git blame -L 24,32` |
| line 25 introduced by | `1e7893a4` (yurakuratov, **2024-05-22**) — 13 days **earlier** | `git blame -L 25,25` |

**Second disk:** `/apdcephfs_zwfy6` is **not mounted on this node** (`mount | grep apdcephfs`
shows only `/apdcephfs_wzc1` and `/apdcephfs_wzc1_304376610`, the latter an alias of the same
device), and this task forbids SSH. This does not weaken the code claims: the authoritative
comparison for *code* is against **upstream**, which was fetched and is byte-identical. It does
scope the corpus measurement in §4, which is stated there as wzc1-only.

> **Do not restore a deleted sentence.** An earlier draft said the commit that added the guard
> *"also added/moved the lowercasing, which is how the two came to collide."* I read the full diff
> of `58e5d20b` and that is **false**. The whole diff is 2 insertions / 1 deletion: it **adds** line
> 31, and it changes `label in question` → `label in question.lower()` **in a different function**
> (`compare_answers`) — *that* is the "lowercase bug" the message names. It does **not** touch line
> 25. The corrected framing is also the stronger one: the lowercasing already existed, and the guard
> was added on top of it by an author who, by their own commit message, was thinking about case at
> that very moment and still did not notice.

---

## 3. Claim A, verified by execution — and the word that must be dropped

Script: `evidence/k3_exit_reachability_check.py`; full output recorded in §6.

`STATUS.json` (`dead_code_recheck_20260814`, `lifecycle_reason`), `K1_NOVELTY_CHECK.md` §3.1 and
`K1_NARROWING.md` §2.1 all say **"UNREACHABLE" / "dead code"**. Measured, line-level trace:

```
lines=[25, 27, 29, 30, 31, 32]  line31_executed=True   input='kitchen Question: Where is Mary? Answer: garden'
lines=[25, 27, 29, 30, 31, 32]  line31_executed=True   input='plain kitchen'
lines=[25, 27, 29, 30, 31, 32]  line31_executed=True   input=''
lines=[25, 27, 29, 30, 31, 32]  line31_executed=True   input='Question'
...
inputs where line 31 executed: 7/7
VERDICT A1: 'line 31 is UNREACHABLE' -> REFUTED (the line executes every time)
```

Three properties were measured **separately**, because they are not the same property and a
maintainer will reject the wrong one:

| | property | method | result |
|---|---|---|---|
| A1 | is it **executed**? | `sys.settrace`, line-level | **YES, 7/7 inputs** → "unreachable" is **false** |
| A2 | can it **change the value**? | value entering vs leaving line 31, on a step-by-step reimplementation *asserted equal* to the real function on every probe | **NO, 0/10 probes** → guaranteed no-op |
| A3 | **why** can it never fire? | exhaustive sweep of all 0x110000 codepoints | `str.lower()` cannot emit an ASCII capital anywhere, and is idempotent everywhere → **proof, not sample** |

Controls that make the finding specific to line 31 rather than to the whole tail:
`preprocess_output('kitchen <CONTEXT> blah') == 'kitchen '` and
`preprocess_output('kitchen <EXAMPLE> blah') == 'kitchen '` (lines 29/30 **do** fire on their
lowercase literals), while `preprocess_output('kitchen Question: blah') == 'kitchen question: blah'`
(line 31 does not). `git grep preprocess_output` upstream → exactly 2 hits: the definition (line 24)
and the single caller (line 36). Line 25 is the function's first and unconditional statement, so no
argument can bypass it.

**Observable consequence** (a maintainer will ask whether it matters):
`compare_answers('kitchen', 'kitchen Question: where is Mary? Answer: garden', 'Where is John?',
TASK_LABELS['qa1'])` is **False** as shipped, and **True** with the one-character fix. So the dead
guard produces false negatives and **fixing it is not score-neutral**.

---

## 4. Claim B, verified as a real two-arm comparison — SURVIVED

Arm **CANON** = `babilong.metrics.compare_answers` as shipped. Arm **NOTRUNC** = identical except
line 27 removed. Every other filter — lowercase, `<context>`/`<example>`, the
`labels_in_question` subtraction, and crucially the **uniqueness** requirement
`len(labels_in_output) == 1` — is held fixed **in both arms**, so a disagreement isolates line 27
and nothing else. **Arm sanity asserted:** on period-free inputs the two arms are identical (3/3),
which is what makes them differ by line 27 alone.

A demo is accepted **only if the two arms disagree on it**:

| canon | notrunc | isolates line 27 | case |
|---|---|---|---|
| False | **True** | ✅ | `"The answer is A. kitchen"` → canon `'the answer is a'`; truncation **destroys** a correct enumerated answer |
| False | **True** | ✅ | `"John moved several times. He is in the kitchen"`; **destroys** a correct reason-then-answer |
| **True** | False | ✅ | `"kitchen. Question: Where is Mary? Answer: garden"`; truncation **rescues** a correct scaffold-leaking answer |
| **True** | False | ✅ | `"kitchen is wrong. the answer is garden"` (target `kitchen`); truncation **manufactures** a correct answer — a false positive |
| False | False | ❌ **REJECTED** | `"Choices: A. In the kitchen B. In the garden. The answer is kitchen."` |

**The rejected row is the negative control and the reason this section exists.** That string is
presented in `K1_NOVELTY_CHECK.md` §3.2 and `K1_NARROWING.md` §2.2 as *"truncation kills a correct
list-format answer"*. Executed, it is **False under both arms**: without truncation, **both**
`kitchen` and `garden` survive, so it dies on the **uniqueness** test at `metrics.py:56`, **not** on
line 27. It does not exhibit the mechanism it is filed under and is **excluded from the issue body**.
(`STATUS.json.assertion_b.row1_DOES_NOT_REPRODUCE` caught this on 08-15; I re-executed rather than
trusting the catch, and it is confirmed.)

### On real stored generations — recursive, wzc1-only

No GPU: this is re-scoring prediction CSVs that are already on disk. Recursive glob
`babilong_results/**/qa*.csv` → **10 886 files**, content-deduplicated to **10 647**, **0 files
skipped for unknown task or bad schema**, 800 rows skipped (they belong to retrieval-diagnostic
CSVs with columns `idx,length,n_chunks,needle_chunks,…` and **no** `target`/`output`/`question`
field, i.e. they are not scoreable rows at all — inspected, not assumed). `is_list_format()` copied
verbatim from `analyze_a02_truncation_ablation.py`.

| stratum | n | line 27 destroys a correct answer | line 27 rescues one | **net pp of keeping line 27** |
|---|---|---|---|---|
| **choice-list outputs** | 10 377 | 1 332 (12.84 %) | 13 (0.13 %) | **−12.71** |
| **non-list outputs** | 407 990 | 1 248 (0.31 %) | 8 233 (2.02 %) | **+1.71** |
| all items pooled | **418 367** | 2 580 | 8 246 | +1.35 |

**The stratified signs are opposite**, so the pooled figure is a mixture whose sign is set by how
many enumerated answers are in the pool — which is a property of the *models being scored*, not of
the metric. Per-task pooled values do not share a sign either (qa5 **+1.97**, qa1 +1.25, qa2 +1.18,
qa8 **−0.38**, qa9 **−2.67**).

**CAVEAT that must travel with these numbers:** this is a heterogeneous mix of our own historical
runs across many models and configurations, **not** a controlled comparison. It supports the
**sign-dependence** only and must never be quoted as an effect size for any model comparison.

---

## 5. THE ISSUE TEXT — ready to paste, **not posted**

**Title:**

```
metrics.py: preprocess_output's split('Question') guard can never fire (line 25 lowercases first)
```

**Body:**

```markdown
### Summary

Two points about `babilong/metrics.py`, both reproducible from a clean clone with no extra data and
no GPU:

1. **A defect.** The guard on line 31, `output.split('Question')[0]`, can never fire, because line
   25 has already lowercased the string. It is the function's only guard against post-answer prompt
   scaffold leakage. The fix is one character, but it is **not** score-neutral.
2. **A design question, deliberately not filed as a bug.** Line 27's first-period truncation has a
   **format-dependent sign**: it removes correct enumerated / reason-then-answer answers, rescues
   correct scaffold-leaking ones, and can manufacture false positives. We are **not** proposing to
   remove it. We are flagging that its net effect on a score depends on the answer-format habit of
   the model being scored, so it can act as a systematic offset between two systems that differ in
   output style rather than in task ability.

### Provenance

- `booydar/babilong`, branch `main`, HEAD `7a6efee29f5cac03c3c410e6799c80fd2ffe3610` (2026-06-01).
- `babilong/metrics.py`, md5 `0a5ecc52ade4e337d35b8f9c97c38310` at that HEAD. All line numbers
  below are that file's.
- Line 31 was added in `58e5d20b775b` (2024-06-04), *"add split by \"Question\" in preprocess + fix
  lowercase bug"*. Worth noting: the "lowercase bug" fixed in that same commit is a **different**
  one — `label in question` → `label in question.lower()`, inside `compare_answers`. The
  `output = output.lower()` on line 25 predates it; `git blame` attributes it to `1e7893a4`
  (2024-05-22), 13 days earlier. So the guard was added on top of an already-existing lowercasing.
- Verified on Python 3.14.6. Point 1's argument is version-independent (it is quantified over all
  Unicode codepoints).

### The code

```python
# babilong/metrics.py, lines 24-32
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

## Point 1 — the guard on line 31 can never fire

### Terminology first, so this does not start on the wrong foot

Line 31 **is executed** on every call — a line-level trace of `preprocess_output` shows
`[25, 27, 29, 30, 31, 32]` for every input we tried. **This is not a report of unreachable code.**
The claim is narrower and, we think, still worth fixing: the line is reached and **can never change
the value**. It is a guaranteed no-op.

### Minimal reproduction

```python
from babilong.metrics import preprocess_output

s = "kitchen Question: Where is Mary? Answer: garden"
print(repr(preprocess_output(s)))
# actual:   'kitchen question: where is mary? answer: garden'
# intended: 'kitchen '
```

The reproduction string deliberately has **no period before `Question`**, so that line 27 cannot
mask the problem. When a period is present, line 27 happens to truncate at the same place and the
dead guard is invisible — which is probably why this has gone unnoticed.

Control showing lines 29/30 **do** fire, i.e. the defect is specific to line 31:

```python
print(repr(preprocess_output("kitchen <CONTEXT> blah")))   # -> 'kitchen '
print(repr(preprocess_output("kitchen <EXAMPLE> blah")))   # -> 'kitchen '
print(repr(preprocess_output("kitchen Question: blah")))   # -> 'kitchen question: blah'   <-- no-op
```

### Why "can never fire" is a proof rather than an observation

`str.lower()` cannot emit an ASCII capital, so no input can produce the `Q` that line 31 needs.
Brute-forced over the entire Unicode range:

```python
print([cp for cp in range(0x110000) if 'Q' in chr(cp).lower()])
# []
print([cp for cp in range(0x110000) if any(u in chr(cp).lower() for u in 'QUESTION')])
# []
# no codepoint lowercases to a multi-character string that still contains an ASCII capital:
print([(hex(cp), chr(cp).lower()) for cp in range(0x110000)
       if len(chr(cp).lower()) > 1 and any('A' <= ch <= 'Z' for ch in chr(cp).lower())])
# []
# str.lower is idempotent everywhere, so the chained splits on 27/29/30 cannot reintroduce one:
print([cp for cp in range(0x110000) if chr(cp).lower().lower() != chr(cp).lower()])
# []
```

The one context-sensitive rule in `str.lower` (Greek final sigma) also yields no capital:
`'ΟΔΟΣ'.lower() == 'οδος'`.

There is no alternative path either: the `.lower()` is the function's own first statement and is
unconditional, and `git grep preprocess_output` returns exactly 2 hits repo-wide — the definition
(line 24) and the single call site (`compare_answers`, line 36); the eval notebooks all go through
`compare_answers`. (The separate legacy scorer `babilong/babilong_utils.py:8 compare_answers` has no
`'Question'` guard at all, so it is unaffected by this point.)

### Why it matters: this is the only defence against scaffold leakage, and enabling it moves scores

Line 31 is the only protection against a model that keeps generating past its answer and re-emits
the `Question: … Answer: …` scaffold — common for base / non-instruction-tuned models evaluated
without a stop string. Executed:

```python
from babilong.metrics import TASK_LABELS, compare_answers
L = TASK_LABELS['qa1']; Q = "Where is John?"
raw = "kitchen Question: where is Mary? Answer: garden"

compare_answers("kitchen", raw, Q, L)
# False  -- preprocess_output leaves 'kitchen question: where is mary? answer: garden',
#           so both 'kitchen' and 'garden' survive and the uniqueness check rejects the item
# with line 31 changed to split('question'), preprocess gives 'kitchen ' and the same call -> True
```

So the dead guard causes **false negatives**, and repairing it is **not** score-neutral. Fixing it
behind a flag, or fixing it and re-running the leaderboard, are both reasonable; silently fixing it
would make new numbers incomparable with published ones. That call is yours, not ours.

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

## Point 2 — line 27's first-period truncation has a format-dependent sign (design question)

We are **not** reporting line 27 as a defect and **not** proposing to delete it. The point is that
its effect on a score has **no fixed sign**, and that the sign tracks the model's answer format
rather than its task ability.

`compare_answers` requires the gold target to be the **unique** surviving task label, and line 27
keeps only the text before the first period. All four of the following are executed against the
unmodified package, with `target='kitchen'` and `question='Where is John?'` (which contains no task
label):

**It removes correct answers** whose verdict sits after a period:

```python
compare_answers("kitchen", "The answer is A. kitchen", Q, L)
# False  -- preprocess_output -> 'the answer is a'

compare_answers("kitchen", "John moved several times. He is in the kitchen", Q, L)
# False  -- preprocess_output -> 'john moved several times'
```

**It rescues correct answers** when the model leaks scaffold afterwards — line 27 doing, by
accident, the job line 31 was meant to do:

```python
compare_answers("kitchen", "kitchen. Question: Where is Mary? Answer: garden", Q, L)
# True   -- preprocess_output -> 'kitchen'
```

**And it can manufacture a correct answer the model did not give:**

```python
compare_answers("kitchen", "kitchen is wrong. the answer is garden", Q, L)
# True   -- preprocess_output -> 'kitchen is wrong'; the model's actual answer was 'garden'
```

So the two directions are not symmetric-but-harmless: the harmful direction penalises enumerating
and reasoning-then-answering styles, and the helpful direction includes a false-positive mode.

### The consequence, measured

Re-scoring **stored generations** with line 27 removed and nothing else changed — the uniqueness
requirement is kept in both arms, so choice-lists still score 0 and no chance inflation is
introduced — over 418 367 items from 10 647 prediction CSVs of our own past BABILong runs (many
models and configurations, `qa1`–`qa10`), stratified by whether the output is a choice-list
enumeration:

| stratum | n | line 27 destroys a correct answer | line 27 rescues one | net effect of keeping line 27 |
|---|---|---|---|---|
| choice-list outputs | 10 377 | 1 332 (12.84 %) | 13 (0.13 %) | **−12.71 pp** |
| non-list outputs | 407 990 | 1 248 (0.31 %) | 8 233 (2.02 %) | **+1.71 pp** |
| all items pooled | 418 367 | 2 580 (0.62 %) | 8 246 (1.97 %) | +1.35 pp |

The sign is **opposite** in the two strata, so the pooled number is a mixture whose sign depends on
how many enumerated answers happen to be in the pool. Per-task pooled values do not agree on a sign
either (qa5 +1.97, qa1 +1.25, qa9 −2.67, qa8 −0.38).

Please do not read those numbers as a benchmark result: they come from a heterogeneous mix of our
own historical runs, not a controlled comparison, and we report them only to establish the
**sign-dependence**, never as an effect size for any model comparison. Note also that the legacy
`babilong_utils.compare_answers` has the same `split('.')[0]`, so this point applies to both
scorers.

### What we would suggest, in increasing order of invasiveness

1. Fix point 1 first (`'Question'` → `'question'`), so that scaffold suppression no longer depends
   on line 27 as an accidental side effect. That alone makes line 27's role smaller and easier to
   reason about.
2. Document line 27 in the README or paper appendix as a scoring convention that penalises
   multi-sentence and enumerated answers, so users can interpret comparisons between
   differently-styled models correctly.
3. Optionally expose a strictness flag, so results can be reported both ways.

---

## Summary

| # | Line | Nature | Status |
|---|------|--------|--------|
| 1 | `metrics.py:31` | `split('Question')` can never fire, because line 25 lowercases first. It is executed on every call — a guaranteed no-op, not unreachable code — and it is the only guard against post-answer scaffold leakage | defect; one-character fix; **changes published numbers** |
| 2 | `metrics.py:27` | first-period truncation has a format-dependent sign: removes correct enumerated/reasoned answers, rescues scaffold-leaking ones, can manufacture false positives | design question / documentation; **not** proposed for removal |

Happy to open a PR for point 1 (with or without a compatibility flag) if you tell us which behaviour
you want to preserve. Point 2 we would rather leave to your judgement, since any change there
affects the leaderboard.
```

---

## 6. "Verified how" — the exact executed evidence for each claim

Everything below was produced by
`proposal/backlog/B11-generative-scorer-format-fragility/evidence/k3_exit_reachability_check.py`
(exit code captured **directly**, not through a pipe: `rc=0`).

| claim in the issue body | how it was verified | outcome |
|---|---|---|
| lines 25/27/31 are those statements | `inspect.getsourcelines` + `assert` on the text of each of the three lines | **asserted**, would raise on drift |
| line numbers are the maintainer's | upstream fetch from `main`, from pinned sha `7a6efee2…`, and from `dev`; `md5sum` + `diff` vs local | md5 `0a5ecc52…` on all, `diff` **empty** |
| "line 31 is executed" (⇒ *not* unreachable) | `sys.settrace`, line-level, 7 inputs incl. empty string, `'Question'`, and Turkish-I / Sigma / sharp-s adversarials | `[25,27,29,30,31,32]` on **7/7**; A1 **REFUTED** |
| "can never change the value" | step-by-step reimplementation **asserted equal** to the real `preprocess_output` on every probe, then before/after around line 31 | changed on **0/10** probes |
| "no input can put a capital past line 25" | exhaustive sweep of all 0x110000 codepoints: contains-`Q`, contains-any-of-`QUESTION`, multi-char lowerings retaining a capital, non-idempotent lowerings | all four lists **empty** |
| lines 29/30 fire (specificity control) | executed `preprocess_output` on `<CONTEXT>` / `<EXAMPLE>` / `<context>` and on `Question:` | `'kitchen '` vs `'kitchen question: blah'`; **control passed** |
| only one caller | `git grep preprocess_output` in the upstream checkout | 2 hits: def line 24, call line 36 |
| "fixing it is not score-neutral" | two-arm `compare_answers` vs the same code with the one character changed, `TASK_LABELS['qa1']`, question asserted to contain no task label | `False` → `True` |
| line 27 destroys / rescues / manufactures | 5-case two-arm table; a case counts **only if the arms disagree**; arm sanity asserted on period-free inputs (3/3 identical) | 4 accepted, **1 rejected** |
| the record's `"Choices: A. …"` string | executed under both arms | canon `False` **and** notrunc `False` → **excluded**; it dies on uniqueness (`{'garden','kitchen'}` both survive), not on line 27 |
| corpus sign flip | recursive glob (10 886 → 10 647 unique), 418 367 items, both arms, stratified by `is_list_format` copied verbatim from the A02 script; 800 non-scoreable rows inspected and identified as retrieval-diagnostic schema | LIST **−12.71 pp** vs non-LIST **+1.71 pp**; signs differ |

### Claims I did **not** verify, stated plainly

1. **zwfy6-resident CSVs.** Not mounted on this node and SSH is forbidden here, so the corpus figure
   is **wzc1-only**. It does not affect either code claim (those are settled against upstream), and
   the sign flip already reproduces on 418 367 items.
2. **The earlier precheck's `n = 195 064` and its −8.52/+0.85 pp pair.** Not reproduced, and
   deliberately **not quoted**: its glob is non-recursive (see §0), so it is not the "full corpus"
   it says it is. Superseded by the recursive figure above.
3. **The 08-15 record's `−8.86 / +0.25 pp`** (500 randomly-sampled CSVs, no recorded seed or file
   list). Unreproducible by construction; not quoted.
4. **Whether the trade-off changes any ranking.** Untouched by this pass. That rests on the 6 A02
   cells, where the inversion is **not significant** (exact McNemar p = 0.0703, Holm 0.4219). The
   issue body makes no ranking claim, and it must not.
5. **K3 as originally written** ("do LongBench / RULER / LongEval scorers have the same class of
   preprocessing?"). **Not tested.** Only the K3-**exit** step was executed. *"BABILong's scorer is
   uniquely broken"* remains a forbidden claim, and the issue body does not say it.

### Pre-filing checklist

- [x] Duplicate scan: 18 records (open **and** closed) + 20 comments + 11 search queries + 9
      branches + 27 forks. 4 keyword hits read in full, all false positives.
- [x] Line numbers pinned to a byte-identical upstream fetch; both md5 discrepancies in the record
      resolved (trailing-newline artifact) rather than inherited.
- [x] Every snippet in §5 executed against the unmodified package.
- [x] The one demo that does **not** exhibit its claimed mechanism identified and **excluded**.
- [x] Body contains no internal paths, node addresses, arm names, depth-knob values, proposal IDs,
      p-values, or arm-comparison numbers.
- [x] Point 2 framed as a design question; does **not** propose deleting line 27 (a forbidden
      claim); the heterogeneity caveat travels inline with the numbers.
- [x] The word **"unreachable" does not appear as an assertion** anywhere in the issue body; the
      terminology note **precedes** the claim it qualifies.
- [ ] **Approval to file — NOT GRANTED.** Requires explicit human sign-off. Nothing was posted.
