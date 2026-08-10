# Stale agent worktrees removed 2026-08-10 — branch inventory

`.claude/worktrees/` held 7 full checkouts (175 MB) from 2026-07-07 agent runs.
All 7 directories removed with `git worktree remove --force`; **all 7 branches
are intact** (`git branch --list 'worktree-*'` → 7). Nothing was lost — a
worktree dir is just a second checkout of a branch that still exists.

To recover any of these: `git checkout worktree-<name>` or
`git worktree add .claude/worktrees/<name> worktree-<name>`.

| branch | HEAD | landed on main? |
|---|---|---|
| `worktree-agent-a3d4a8f61430b74e5` | `bb97e43` bench QCMem vs full-ctx latency+memory | **yes** — `scripts/bench_qcmem_vs_fullctx.py` byte-identical on main |
| `worktree-agent-a542419157c7a6a9c` | `114da06` fix YaRN factor for 64k/128k | **yes** — `scripts/eval_p16_kvcompress.py` byte-identical on main |
| `worktree-agent-aff62ec41a60faf79` | `0ee03e2` MemoryLLM port smoke + pyramid v1 design | **yes** — all 4 files byte-identical on main |
| `worktree-agent-af1b6ba405b2133fa` | `43e17d8` | **yes** — 0 commits ahead of main |
| `worktree-agent-a3f9b2f19ca9f221d` | `b6b3611` reader_attn salience selector | **superseded** — main's `eval_qcmem_babilong.py` is newer (`81949b0`, 2026-08-04) |
| `worktree-agent-a88042f2e7d45f99c` | `47a8768` RULER eval driver | **superseded** — main's `eval_ruler_qcmem.py` is newer (`81949b0`, 2026-08-04) |
| `worktree-agent-a33effba058813979` | `edf7a0f` **pyramid P2 dual-cadence read** | ⚠️ **NOT on main** — `scripts/selftest_pyramid_p2.py` missing from main entirely; `src/memory/pyramid/pyramid_model.py` on main still has the P2 read as `NotImplementedError` |

## ⚠️ The one real unmerged asset

`worktree-agent-a33effba058813979` contains a **complete P2 dual-cadence read
implementation** for `PyramidMemory` plus a non-circular self-test with two
correctness gates (`M_far=0` == QCMem near-only; full-pool == MemoryLLM
bit-for-bit). Main's `src/memory/pyramid/pyramid_model.py` is still the skeleton
with the read math stubbed out.

The pyramid direction is currently **dormant** (only cited in `paper*/`
related-work bibs, no `proposal/active/` entry). If it is ever revived, start by
cherry-picking `edf7a0f` rather than re-implementing — the mask/RoPE consistency
gate that blocked it was solved there.
