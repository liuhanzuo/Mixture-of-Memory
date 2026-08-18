---
name: scaffold-server-heartbeat
description: Inspect, monitor, debug, and safely advance the Scaffold-Coder 8-GPU server workflow. Use when asked to run a heartbeat, 巡检服务器, check GPU/training/queue status, continue experiments, diagnose a registered failure, advance TODO.md, or report periodic Scaffold-Coder progress.
---

# Scaffold Server Heartbeat

Run a deterministic remote inspection, then make only registered and auditable
changes. This skill is the heartbeat action; it does not create a scheduler or
wake a chat by itself.

## Inspect

From the Scaffold-Coder repository, run:

```bash
python .codex/skills/scaffold-server-heartbeat/scripts/inspect_scaffold.py \
  --trigger --ensure-daemon \
  --output ops/artifacts/latest_skill_heartbeat.json
```

The script reads `SERVER_ACCESS.md` without printing the password, invokes one
lock-safe remote heartbeat, and returns GPU, active-run, queue, Git, disk, log,
and history state as JSON.

## Decide

Use this order:

1. If a registered run is healthy, do not interfere. Report progress, loss,
   throughput, memory, ETA, and latest checkpoint.
2. If it completed, validate its success artifact and let the heartbeat advance
   the queue.
3. If it is `NEEDS_DEBUG`, inspect the full registered log, reproduce the
   failure, patch the smallest cause, run tests, commit, sync, archive the
   blocked state, and reset only that queue item.
4. If no run is active and GPUs are idle, invoke one heartbeat to launch the
   first safe `READY` item.
5. If GPUs belong to an unknown or external process, remain read-only and
   continue CPU-side TODO work.

## Safety

- Never print or commit `SERVER_ACCESS.md`.
- Never kill an external or unregistered PID.
- Signal only the process group in `ops/state/active_run.json`.
- Keep the first resumable checkpoint before replacing a run.
- Treat finite loss, all-rank progress, useful GPU utilization, and a verified
  success artifact as completion gates.
- Preserve live remote modifications to `ops/queue.tsv` and `ops/history.tsv`
  when syncing Git.

## Advance research

After the current gate passes, execute the first actionable item in `TODO.md`.
Update tests, metrics, documentation, Git commits, and the remote queue. Report
only material changes when invoked periodically.

For recurring execution, configure an external scheduler or Codex Scheduled
Task to invoke this skill. Keep the existing remote tmux heartbeat as the
execution fallback.
