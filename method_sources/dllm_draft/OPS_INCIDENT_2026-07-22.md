# GPU occupancy incident — 2026-07-22

## Timeline

- Until 2026-07-22 20:26 +08:00, the pre-existing OLMo run was healthy at
  step `25,860 / 200,000`, with finite loss and approximately 7.81 s/step.
- At 2026-07-22 20:27:41 +08:00, its `torchrun` launcher (PID 490809)
  logged that it **received signal 15 (SIGTERM)** and propagated SIGTERM to
  workers 490879–490886.
- The latest complete checkpoint visible for that unrelated run is
  `step25500.pt`, written at 2026-07-22 19:39:14 +08:00.
- By 2026-07-22 20:35 +08:00, eight different CUDA PIDs
  2418618–2418625 occupied approximately 19.4–19.9 GiB per GPU with substantial
  utilization. Those PIDs are not visible in this container's `ps` namespace,
  so they appear to belong to another container/allocation.

## Safety audit

Scaffold-Coder automation did not terminate the OLMo run:

- it is declared `read_only: true` under `external_runs`;
- `ops/heartbeat.py` only sends signals to the process group stored in
  `ops/state/active_run.json`;
- no Scaffold GPU run was active at 20:27;
- the only Scaffold jobs around the incident were short CPU test jobs.

The source of SIGTERM is therefore external to the Scaffold watchdog. No
attempt was made to signal, restart, or attach to either the stopped OLMo run
or the replacement out-of-namespace GPU processes.

## Current policy

- Keep all Scaffold GPU jobs in `READY` state while any GPU exceeds the idle
  memory/utilization thresholds.
- Continue CPU-side development and validation.
- Automatically begin the queued smoke gates only after all eight GPUs are
  observably idle.
- If the owner wants the old OLMo experiment resumed, resume from step 25,500
  under that project's own launch procedure; this repository will not do so
  without explicit authorization.
