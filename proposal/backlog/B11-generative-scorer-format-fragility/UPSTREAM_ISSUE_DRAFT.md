<!-- STATUS: DRAFT — NOT FILED. Filing requires explicit approval. -->

# Upstream issue draft — booydar/babilong

**Target repo:** https://github.com/booydar/babilong
**Target file:** `babilong/metrics.py`
**Draft prepared:** 2026-08-15
**Filing status:** NOT FILED. Do not submit without explicit approval.

Everything below is written so an upstream maintainer can reproduce it from a clean
clone with no extra data and no GPU. Two independent points: point 1 is a defect
(a guard that can never fire), point 2 is a design question (an operation whose
effect on a score has a format-dependent sign), deliberately not phrased as a bug.

---

## Proposed title

`metrics.py: preprocess_output's split('Question') guard can never fire (line 25 lowercases first)`

---

## Proposed body

### Environment / provenance

- Repo: `booydar/babilong`, branch `main`
- Repo HEAD at the time of writing: `7a6efee29f5c` (2026-06-01, "Merge pull request #18 …")
- File: `babilong/metrics.py`, md5 `0a5ecc52ade4e337d35b8f9c97c38310`
- The file has not changed since `93d7bfe67ad0` (2025-04-05); the line in question was
  added in `58e5d20b775b` (2024-06-04, *"add split by \"Question\" in preprocess + fix lowercase bug"*)
- Python 3.11.6 (the argument below is version-independent; see the brute-force check)

### The code

```python
# babilong/metrics.py, lines 24-32 at HEAD 7a6efee29f5c
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

## Point 1 — line 31 is a guaranteed no-op (a defect)

`preprocess_output` lowercases its input unconditionally at line 25, then at line 31
splits on the **capitalised** literal `'Question'`. Because every value flowing into
line 31 is derived from `output.lower()` by substring slicing only, the string
`'Question'` can never be present, so `split('Question')` always returns a
single-element list and line 31 never changes the value.

This matters because line 31 is the *only* guard in the function against the failure
mode it was added for: a model that keeps generating after its answer and re-emits the
`Question: …` prompt scaffold (a very common behaviour for base / non-instruction-tuned
models evaluated without a stop string). Lines 29 and 30 do fire, because those two
literals are already lowercase (`'<context>'`, `'<example>'`). Only line 31 is affected.

### Minimal reproduction

```python
from babilong.metrics import preprocess_output

s = "kitchen Question: Where is Mary? Answer: garden"
print(repr(preprocess_output(s)))
# actual:   'kitchen question: where is mary? answer: garden'
# intended: 'kitchen '
```

Note the reproduction string deliberately contains **no period before `Question`**, so
line 27 cannot mask the issue. (With a period present, line 27 happens to truncate at
the same place and the dead guard is invisible — which is likely why this has gone
unnoticed.)

Control showing lines 29/30 *do* fire, i.e. the defect is specific to line 31:

```python
print(repr(preprocess_output("kitchen <CONTEXT> blah")))   # -> 'kitchen '
print(repr(preprocess_output("kitchen <EXAMPLE> blah")))   # -> 'kitchen '
```

### Why "can never fire" is not just an empirical observation

`str.lower()` cannot emit an ASCII uppercase letter, so no input can produce the
capital `Q` that line 31 needs. Brute-forced over the entire Unicode range:

```python
print([cp for cp in range(0x110000) if 'Q' in chr(cp).lower()])
# []
print([cp for cp in range(0x110000) if any(u in chr(cp).lower() for u in 'QUESTION')])
# []
# only one codepoint lowers to more than one character, and it contains no ASCII uppercase:
print([(hex(cp), chr(cp).lower()) for cp in range(0x110000)
       if len(chr(cp).lower()) > 1
       and any('A' <= ch <= 'Z' for ch in chr(cp).lower())])
# []
# str.lower is also idempotent for every codepoint, so the chained splits cannot reintroduce one:
print([cp for cp in range(0x110000) if chr(cp).lower().lower() != chr(cp).lower()])
# []
```

The one context-sensitive rule in `str.lower` (Greek final sigma) also produces no
uppercase output: `'ΟΔΟΣ'.lower() == 'οδος'`.

There is no alternative code path: the `.lower()` is the function's own first statement
and is unconditional, so *every* caller of `preprocess_output` — including
`compare_answers` at line 36, which is the only in-repo caller — is subject to it.
(The separate legacy `babilong_utils.compare_answers` does not contain this guard at all.)

### Terminology note

The line **is** executed (a line-level trace shows lines 25, 27, 29, 30, 31, 32 all run);
it is not unreachable code in the control-flow sense. It is reached and has provably no
effect. Please read the report as "the guard can never fire", not "the line is never
executed".

### Suggested fix

One character of case, matching the style of lines 29/30:

```python
-    output = output.split('Question')[0]
+    output = output.split('question')[0]
```

A case-insensitive variant that also handles the `Answer:` scaffold would be more
robust, but is a behaviour change rather than a bug fix:

```python
import re
output = re.split(r'(?i)\bquestion\b', output)[0]
```

### Please note this changes published numbers

Enabling the guard makes some previously-failed outputs pass (any case where a correct
answer is followed, without an intervening period, by regurgitated prompt scaffold), so
it is **not** score-neutral. It would be reasonable to fix it behind a flag, or to fix
it and re-run the leaderboard, but silently fixing it would make new numbers
incomparable to old ones. Maintainers are better placed than we are to decide which.

---

## Point 2 — line 27's first-period truncation has a format-dependent sign (a design question, not a bug)

We are *not* reporting line 27 as a defect. We want to flag that its effect on the
score has **no fixed sign**: whether it helps or hurts depends on the answer *format
habit* of the model being evaluated, which means it can act as a systematic,
format-correlated offset between two models (or two configurations of one model) that
differ in output style rather than in task ability.

`compare_answers` requires the gold target to be the **unique** surviving task label,
and line 27 keeps only the text before the first period. Together:

**It removes correct answers** when the model's verdict sits after any period:

```python
from babilong.metrics import TASK_LABELS, compare_answers
L = TASK_LABELS['qa1']; Q = "Where is John?"

compare_answers("kitchen", "The answer is A. kitchen", Q, L)
# False   (preprocess_output -> 'the answer is a')

compare_answers("kitchen", "John moved several times. He is in the kitchen", Q, L)
# False   (preprocess_output -> 'john moved several times')
```

**It rescues correct answers** when the model leaks scaffold after its answer — this is
line 27 doing the job line 31 was supposed to do:

```python
compare_answers("kitchen", "kitchen. Question: Where is Mary? Answer: garden", Q, L)
# True    (preprocess_output -> 'kitchen')
```

**It can also manufacture a correct answer that the model did not give:**

```python
compare_answers("kitchen", "kitchen is wrong. the answer is garden", Q, L)
# False positive: True, although the model's actual answer is 'garden'
```

So the two directions are not symmetric-but-harmless: the harmful direction penalises
enumerating / reasoning-then-answering styles, and the helpful direction includes a
false-positive mode.

The consequence we would like maintainers to be aware of: because the sign is tied to
answer format, two systems with the same task ability but different output styles can
receive systematically different scores, and the difference is attributable to line 27
rather than to the benchmark's construct. This is a property of the metric, not of any
one model, which is why we are raising it here rather than working around it locally.

We are deliberately **not** proposing to delete line 27. Deleting it is not a fix — it
would remove the scaffold-leak protection that line 31 was supposed to provide and that
line 27 currently provides by accident, and in our own checks removing it lowers scores
on some tasks while raising them on others. Possible directions, in increasing order of
invasiveness:

1. Fix point 1 first (`'Question'` → `'question'`), so scaffold suppression no longer
   depends on line 27 as a side effect. This alone makes line 27's role smaller and
   easier to reason about.
2. Document line 27 in the README / paper appendix as a scoring convention that
   penalises multi-sentence and enumerated answers, so users can interpret comparisons
   between differently-styled models correctly.
3. Optionally expose a strictness flag so results can be reported both ways.

---

## Summary

| # | Line | Nature | Status |
|---|------|--------|--------|
| 1 | `metrics.py:31` | `split('Question')` can never fire because line 25 lowercases first; it is the only guard against post-answer scaffold leakage | defect, one-character fix, but changes published numbers |
| 2 | `metrics.py:27` | first-period truncation has a format-dependent sign (removes correct enumerated/reasoned answers, rescues scaffold-leaking ones, can manufacture false positives) | design question / documentation, **not** proposed for removal |

Happy to open a PR for point 1 (with or without a compatibility flag) if you tell us
which behaviour you want to preserve. Point 2 we would rather leave to your judgement,
since any change there affects the leaderboard.

---

## Pre-filing checklist (internal, delete before filing)

- [x] Upstream HEAD recorded (`7a6efee29f5c`) and `metrics.py` md5 matched byte-for-byte
      against the local copy, so line numbers 25/27/31 are the maintainer's line numbers.
- [x] All 10 issues (open + closed) and all 8 PRs enumerated; keyword scan over every
      title and body for `metrics.py` / `preprocess_output` / `compare_answers` /
      `truncat*` / `lowercase` / `split('Question')` / `scorer` / `first sentence`
      returned only two false positives (#16 is a Gemma reproduction/chat-template
      question, #17 is a results-submission PR). GitHub issue search
      `repo:booydar/babilong+{metrics.py,preprocess_output,truncation,lowercase}` = 0 hits.
- [x] Every snippet in the body was executed against the canonical package.
- [x] Contains no internal paths, node addresses, internal conclusions or unpublished numbers.
- [ ] Approval to file — NOT GRANTED as of 2026-08-15.
