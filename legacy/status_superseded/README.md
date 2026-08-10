# `legacy/status_superseded/` — archived `status/` markdown (2026-08-10)

43 files moved here from `status/` during a doc-hygiene pass. **Nothing here was
deleted** — these have historical or provenance value but are no longer entry
points. If a number in one of these files appears in a paper table, that file was
*kept* in `status/` instead, not archived.

Method: each file got a live-reference scan (`grep` across `*.md/*.sh/*.py/*.tex/*.json`
excluding `legacy/`, `.git/`, worktrees), then an adversarial second pass whose job was
to *refute* every ARCHIVE/DELETE call. **43 of 175 initial calls were overturned back
to KEEP** — mostly because the file turned out to hold a paper-table number, a
retraction, or the only copy of an experiment verdict.

⚠️ **Known defect in the first pass**: the mandated scan used `--include='*.json'`,
which does **not** glob-match `*.jsonl`. Referrers in `status/RESEARCHER_REPORTS.jsonl`
and `status/TRAINER_ACTIVITY.jsonl` were therefore invisible to the triage agents. The
adversarial pass caught this and several files were rescued. MAIN re-verified every
DELETE candidate with a corrected scan before executing.

Always-kept categories (never archived, regardless of age): `SESSION_HANDOFF.md`,
`RUN_REGISTRY.md`, `BENCHMARK_RESULTS.md`, `PENDING_TASKS.md`, `GPU_STATUS.md`,
`TRAINER_ACTIVE.md`, and **anything that is a retraction or postmortem** — those
prevent revived errors.

Recover a file with `git mv legacy/status_superseded/<name> status/`.
