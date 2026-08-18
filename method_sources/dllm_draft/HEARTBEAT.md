# 30-Minute Heartbeat Policy

## Purpose

The heartbeat is an auditable watchdog. Its idle baseline interval is 30
minutes. To avoid leaving GPUs idle between queued jobs, it polls every five
minutes while work is queued and every minute while a registered run is active.
It performs only pre-authorized, deterministic recovery actions. Research
decisions and novel debugging remain documented in `TODO.md` and the run log.

## Inspection

Each heartbeat records:

1. SSH reachability and remote timestamp.
2. GPU inventory, utilization, memory use, temperature, power, and active PIDs.
3. Training process tree and distributed-worker count.
4. Latest log/checkpoint modification times.
5. Recent log tail and signatures for OOM, NaN/Inf, NCCL errors, tracebacks,
   deadlocks/stalls, disk-full errors, and killed workers.
6. Free disk/inode capacity, RAM, and load.

## Allowed automatic actions

- Mark a run complete when its configured success artifact exists.
- Advance to the first runnable item in the machine-readable queue.
- Restart a previously registered command after a transient failure, subject to
  retry and cooldown limits.
- Terminate only PIDs registered to the active run, never arbitrary processes.
- Stop restarting and mark `NEEDS_DEBUG` after the retry budget is exhausted.
- Disable launches when disk, temperature, GPU health, or checkpoint safety
  thresholds fail.

## Safety and efficiency rules

- Never use `pkill -9`, `killall`, or broad process-name matching.
- Every launched run gets a run ID, PID file, exact command, environment dump,
  log path, checkpoint path, and retry counter.
- Prefer data parallelism/FSDP/ZeRO when the model fits efficiently; use tensor
  parallelism only when memory or kernel efficiency requires it.
- “Use all 8 GPUs” means useful synchronized work, not eight allocated idle
  processes. Throughput, MFU, communication overhead, and memory headroom are
  measured before selecting DP/TP topology.
- Keep at least one resumable checkpoint before replacing or terminating a run.

## Files

- `TODO.md`: human-readable priority queue.
- `ops/queue.tsv`: commands eligible for automatic execution.
- `ops/state/active_run.env`: current run metadata.
- `ops/logs/heartbeat.log`: append-only heartbeat log.
- `ops/logs/<run_id>.log`: per-run stdout/stderr.
- `ops/history.tsv`: run transition history.

## Codex CLI scheduling boundary

The server watchdog above is independent of the Codex conversation. The CLI
does not expose a dedicated scheduling API for an active TUI thread. In this
environment, the verified equivalent is to paste one message into the current
`tcodex` pane and send Enter; when a turn is busy, the TUI queues it as a
follow-up and submits it after the current turn.

`ops/codex_pane_heartbeat.sh` uses a single-pending guard:

- it creates `ops/state/codex_pane_heartbeat_pending.json` before sending;
- later timer ticks skip while that file exists;
- the receiving heartbeat turn first runs
  `ops/ack_codex_pane_heartbeat.sh`;
- this prevents unattended sessions from accumulating duplicate inputs.

The remote `.104` watchdog remains the source of truth for registered-process
monitoring and is independent of this optional conversation trigger.
