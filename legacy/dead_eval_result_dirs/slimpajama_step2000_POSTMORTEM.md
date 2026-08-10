# `slimpajama_step2000/` — deleted 2026-08-10, scored first

**What it was**: offline BABILong eval output for the run `mem_space_slimpajama_prediction`
(`scripts/_launch_slimpajama_prediction.sh`, watcher `scripts/_auto_eval_on_slimpajama_done.sh`),
evaluated 2026-07-05 19:46–20:57. 192 files / 368 KB: `{qa1,qa2,qa5} × {2k,4k,8k,16k}`,
8 shards each, `chat_template_no` (compliant with the chat=False rule).

**Why deleted**: the run it measured is gone — `outputs/mem_space_slimpajama_prediction/`
no longer exists on either disk, so the numbers are unreproducible and unattachable to a
checkpoint. It was never harvested into `status/RUN_REGISTRY.md` or
`status/BENCHMARK_RESULTS.md`, was untracked by git (`git ls-files` → 0), and the only
referrer was its own launch watcher. The training log `logs/mem_space_slimpajama_prediction.log`
(151 KB, 2026-07-05) is kept.

**The numbers, scored before deletion** (`contains` metric, n=100/cell, so nothing is lost):

| task | 2k | 4k | 8k | 16k |
|---|---:|---:|---:|---:|
| qa1 | 4.0 | 0.0 | 0.0 | 0.0 |
| qa2 | 8.0 | 8.0 | 1.0 | 0.0 |
| qa5 | 20.0 | 12.0 | 2.0 | 1.0 |

**OVERALL n=1200, contains = 4.7 %.**

**Reading**: this is a dead arm, not a lost result. 15.8 % of generations are degenerate
(`</question` repetition loops — the same post-training generation collapse that killed
RMT v3–v10, see `CLAUDE.md` 已放弃 list). Scores decay to ~0 by 8k on every task. Under the
PPL/quality红线 this arm was already failed; preserving 368 KB of collapsed generations
adds nothing that this table doesn't record.
