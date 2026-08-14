# MOVED — this directory was a bookkeeping split, not a second proposal

B04 had its analysis scripts here and **all** of its documents (`PROPOSAL.md`,
`STATUS.json`, `SOURCES.md`, `NOVELTY_CHECK.md`, the verdicts, `evidence/`) in a
sibling directory. One direction, two directories — so anyone landing here saw
code with no `STATUS.json` and could not tell what the direction's verdict was.

**Merged on 2026-08-14. The single canonical home is:**

    proposal/backlog/B04-eval-fragility-incubator/

The two analyzers moved (via `git mv`, history preserved) to:

| was | now |
|---|---|
| `proposal/backlog/B04-eval-fragility/analyze_b04_5rung.py` | `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_5rung.py` |
| `proposal/backlog/B04-eval-fragility/analyze_b04_qwen_6rung.py` | `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_qwen_6rung.py` |

Those two files were the **only** copies anywhere (checked both disks: wzc1
`pighzliu_code/` tree, and zwfy6 `/apdcephfs_zwfy6/share_304376610/pighzliu_code/`
via `.73` — zwfy6 never had a `B04-*` directory at all). They are cited as
provenance by both verdicts, `status/SESSION_HANDOFF.md`, and `status/scout_21/*`,
so they were moved, never deleted.

This file is a pointer only. Do not add new B04 material here.
