---
name: a-writers-hardcoded-destination-defines-what-it-destroys
description: ★★ refreeze 工具 DEST 硬编码 round_05, docstring 还写着「writes a NEW round_05」(写时为真); 我想建 round_06 却静默覆盖了 6 个 reviewer 已读的 snapshot(4a2235e8→16171eef); 靠 git 恢复; WRITER 的目标路径必须计算, 且已被 review 的 round 即使显式 override 也要拒绝
metadata:
  type: feedback
---

A **writer's** hardcoded destination does not just choose where output goes — it silently
defines **what gets destroyed**. Reading is forgiving of a stale constant; writing is not.

**2026-08-17, measured.** I ran `paperC/code/refreeze_from_manuscript.py` intending to create
round_06, having said one message earlier that I would not re-freeze in place because
reviewers were still reading. It rewrote **round_05** — after all six had finished:

```
5 manuscript files modified, 2 evidence files added
snapshot_sha256  4a2235e8  ->  16171eef
```

`DEST` was `ROOT / "review_rounds" / "round_05" / "submission_complete"`, and the docstring
still read *"This writes a NEW round_05 directory"* — **true when written, false the moment
round_05 was reviewed**. Same class as
[[a-hardcoded-list-in-an-emitter-silently-defines-a-headline]] and
[[append-only-records-outlive-their-own-truth]]: a constant that was accurate at authoring
time and became a hazard without changing a byte.

**Recovered only because the snapshot was committed** (`f694741`). Restoration verified
byte-identical — hash `4a2235e8`, 49 files, 17 cited, clean tree — and the reviews,
aggregate and adjudication were never touched. Had `submission_complete/` been untracked or
gitignored, the artifact six reviews were written against would simply be gone, and every
one of those reviews would have become unfalsifiable.

**Why it slipped past me:** I checked the *intent* ("freeze the next round") and the *result*
("PASS: every path present"), but not the *destination*. The tool printed
`snapshot: review_rounds/round_05/submission_complete` in its own output — the evidence was
on screen and I read past it because the verdict line said PASS. A green verdict does not
license skipping the target line; cf.
[[a-green-checker-covers-only-what-it-targets]].

**How to apply:**
1. **Any tool that writes: compute the destination, never hardcode it.** Here: newest
   *unreviewed* round, else next number.
2. **Make immutability structural, not documentary.** A round containing `reviews*/*.json`
   or `PANEL_AGGREGATE*.json` is now REFUSED — including under an explicit
   `PAPERC_FREEZE_DEST` override. An override should let you choose among *legal*
   destinations, not authorise destroying evidence a review was written against.
3. **Before running a writer, read its DEST/output path out of the source or its own first
   printed line** — not from what you believe the tool does.
4. Controls that matter here are the *refusals*, not the success: default resolves to
   round_06 not round_05; override at round_05 refused; override at round_04 (also
   reviewed) refused; override at a fresh round allowed. 4/4.
5. **Corollary for review loops generally:** commit the frozen snapshot. Its only job is to
   be the fixed thing reviews point at, and that job fails silently if it is mutable and
   unversioned.
