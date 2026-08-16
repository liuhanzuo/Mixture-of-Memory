---
name: a-guard-must-see-the-repair-it-prescribes
description: "★★ 我写的 stale-absence guard 只用 ±260 字符邻近判\"已修正\", 但它自己规定的修法(顶层 dated superseding key)必然在窗口外 → 五个提案照指示修完后它仍全红; 抑制机制要结构化(key 名+指向该文件+断言存在), 不是放宽窗口"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

A checker that stays red **after its own prescribed repair is applied** is worse than no
checker: everyone learns to ignore its exit code, and then it hides the next real defect.
This is the same failure as [[an-informational-nonzero-rc-hides-real-defects]] arriving from
the opposite direction — there the rc was documented as ignorable, here the rc becomes
*de facto* ignorable because it can never go green.

**2026-08-16, measured.** `proposal/check_stale_absence_claims.py` flags STATUS.json records
asserting a file is absent when it is on disk. Its FAIL message prescribes the repair:

> Repair by ADDING a dated superseding key, not by editing the old sentinel.

But its only suppression was textual proximity — a refutation within ±260 characters. A dated
superseding key lives at the **top level of the document**, thousands of characters from the
sentinel it corrects. So after I repaired all five proposals exactly as instructed, the guard
still reported all five.

**Widening the window is the wrong fix.** 2000 chars would have suppressed genuinely
unrelated sentinels in the same file — trading a false positive for a false negative.
Proximity was the wrong *mechanism*, not a wrong *parameter*.

The fix is structural, with three simultaneous conditions so it cannot become a blanket
silencer: the correcting key must (a) match a recognised name pattern, (b) name this specific
file, and (c) **assert existence** rather than merely mention the filename.

**How I knew (c) was needed:** without it, a dated key saying "RELATED_WORK.md is the
blocker; still to be written" would silence the guard. That is the paperwork-counts-as-
readiness bug from [[a-declared-lifecycle-is-not-an-adjudicated-one]] — a record silencing a
requirement by mentioning it.

**Why:** a guard's suppression logic encodes what it believes a *fix* looks like. If that
belief disagrees with the fix the guard itself recommends, the two halves of the tool
contradict each other, and the contradiction only shows up after someone does the work.

**How to apply:**
1. After writing a guard, **apply its own prescribed repair to one real case and re-run it.**
   If it stays red, the suppression mechanism is wrong. This is a two-minute test and it is
   the only one that catches this class.
2. Write negative controls that prove the suppression is NARROW, not just that it fires. I
   used 7: bare sentinel reports; prescribed repair silences; a dated key that only *mentions*
   the file still reports; a correction about a *different* file still reports; existence
   asserted under an unrecognised key still reports; `exists:false` still reports; genuinely
   absent file never reports. The four "still reports" cases are the ones that matter — see
   [[a-green-checker-covers-only-what-it-targets]].
3. Same session, same file, a second boundary bug: the absence regex's 90-char lookahead
   **crossed an intervening filename** and pinned the absence on the citing file
   (`GATE_PREREGISTRATION.md ... cite 'logs/x.log'. THAT PATH DOES NOT EXIST`) — the record
   was CORRECT and the guard was wrong. My first fix excluded only quotes and slashes and
   passed that case **for the wrong reason** (the log path happened to be quoted); an unquoted
   barrier was still crossed, caught only by a fixture. **Barrier must be the filename token,
   not the punctuation that sometimes surrounds it.** Relates to
   [[fix-the-class-not-the-instance]].
4. Corollary found the same round: a **docstring** naming specific proposals will rot.
   `ready_queue.py`'s section headed "the load-bearing rule this encodes" asserted five named
   proposals had no RELATED_WORK.md; all five existed, and over the tool's whole scanned set
   the true count was **zero**. B07 then wrote a blocker citing "ready_queue.py:46-51
   hard-codes B07 …" — those lines are *inside that docstring*
   (`ast.get_docstring` confirms), and no such list exists in code. A comment was read as a
   code path. **State the RULE in prose; let the code state the FACTS.**
5. When claiming a docs-only edit is behaviour-neutral, prove it: AST-with-docstring-stripped
   identical **and** byte-identical stdout. My first attempt used `git stash`, which silently
   did nothing (`fatal: 'locomo/.git' not recognized`) so the diff compared the file to
   itself and printed a meaningless "IDENTICAL" — the same shape as
   [[a-pipe-makes-a-failing-command-report-success]].
