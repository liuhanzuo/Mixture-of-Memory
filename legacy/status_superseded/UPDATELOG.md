[2026-05-17T10:58:25] heartbeat: validate final eval is down to one tail while remote P8-temp20 is healthy
  - `outputs/eval_p11_500step_validate/` now has all `21/21` result CSVs materialized and `20/21` fully complete; only `qa1 32k=90/100` remains, running on local GPU `3` while GPUs `0/1/2/4/5/6/7` are idle
  - latest remaining validate-tail QUERY_DIAG is still near the old flat floor (`qa1 32k top1_sim_mean=0.002091`), so canonical scoring still waits on the final ten examples rather than any routing recovery signal
  - remote `28.59.80.196` has been repurposed for the clean `P8 + selector_temperature=20` ablation in tmux `p8_temp20_500_20260517_105421`; all 8 GPUs are active at roughly `90-95 GiB` / `94-100% util`
  - remote training log `logs/p8_temp20_500_20260517_105421.log` is healthy through `BABI step 40/500`, and its first QUERY_DIAG already shows `top1_sim_mean=0.020874`, well above the old `~0.002` flat-routing floor
  - working tree cleanliness remains a separate warning; heartbeat state was refreshed around the new dual-frontier snapshot

[2026-05-17T10:36:00] heartbeat: validate final eval narrowed to the last three 32k tails
  - `outputs/eval_p11_500step_validate/` now has all `21/21` result CSVs materialized, and `18/21` already reached `100 parsed rows`; only `qa1 32k=20`, `qa2 32k=30`, and `qa5 32k=40` remain incomplete
  - local eval has narrowed to 3 long workers on GPUs `3..5` (about `40.9-55.0 GiB`, `99% util`); GPUs `0..2` and `6..7` are now idle, while remote `28.59.80.196` remains fully idle at `0 MiB / 0% util`
  - latest eval QUERY_DIAG is still pinned near the old flat floor (`~0.00208-0.00221`), with recent `32k` examples at `qa1=0.002205`, `qa2=0.002200`, `qa5=0.002113`
  - a partial `p11_500step_validate_score.csv` has appeared on disk, but canonical scoring still must wait for the final three `32k` tails to reach `100 parsed rows`
  - `git status --short` remains dirty beyond heartbeat status files, so clean-tree / push is still tracked as a separate warning

[2026-05-17T09:49:59] heartbeat: validate training finished and final eval is already running
  - local `p11_fsdp_500step_validate` finished cleanly at `step 500/500`; final `mem_space_adapter.pt` is present and training ended with `non-finite=0`
  - `outputs/eval_p11_500step_validate/` is now active on local GPUs `0..5`; `12/21` CSVs have materialized so far, including `qa1 0k=100`, `qa2 0k=100`, `qa2 1k=100`, `qa5 0k/1k/2k=100`
  - latest train/eval QUERY_DIAG remains near the old flat floor (`~0.00206-0.00222`), so this heartbeat confirms stability more strongly than routing recovery
  - local GPUs `0..5` are busy with eval while `6..7` are idle; remote `28.59.80.196` remains fully idle at `0 MiB / 0% util`
  - `git status --short` is still dirty beyond heartbeat status files, so clean-tree / push stays a separate warning rather than an implied done item

[2026-05-17T08:54:38] heartbeat: scored temp20 final + step500 and rolled into validate run
  - local `temp20 final` is fully scored: overall `35.24`, short avg `45.42`, long avg `21.67`; all 21 cells reached `100 parsed rows`
  - remote `step500` is fully scored: overall `33.81`, short avg `45.92`, long avg `17.67`; remote node `28.59.80.196` is now idle
  - local H20 immediately moved on to `p11_fsdp_500step_validate`; `logs/p11_fsdp_500step_validate_20260517_0851.log` has reached `PG19 step 10/500` and GPUs are actively occupied
  - verdict: `selector_temperature=20` is a real positive lever, but the post-fix validate run is now the primary next decision point

[2026-05-17T08:10:57] heartbeat: step500 eval reached 21/21 files; temp20 final advanced to 19/21 files
  - local temp20 final eval: 5 workers remain active on GPUs 0/1/3/4/5; GPU2 is now idle because `qa5_short` completed
  - local progress: `outputs/eval_p11_temp20_final_20260517_073341/` currently has `19` CSVs with parsed row counts `20-100`; only `qa1 32k` and `qa2 32k` are still missing
  - remote 28.59.80.196: step500 eval now has all `21` CSVs materialized, but final `32k` tails are still incomplete (`qa1 32k=60`, `qa2 32k=80`, `qa5 32k=80`)
  - remote workers: still narrowed to 3 long-worker processes (`302636`–`302638`) on GPUs 3/4/5
  - verdict: both evals remain healthy; the next milestone is not file appearance anymore, but getting every remaining tail to `100 parsed rows` so scoring can start

[2026-05-17T07:48:25] heartbeat: temp20 final eval is genuinely active; step500 eval now at 19/21 CSVs
  - local temp20 final eval: tmux `p11_temp20_final_eval_20260517_073341` is now actively using GPUs 0-5 at roughly `35-57 GiB` / `98-99% util`
  - local progress: `outputs/eval_p11_temp20_final_20260517_073341/` currently has `14` CSVs with parsed row counts spanning `10-100`; completed cells include `qa1 0k/1k`, `qa2 0k/1k`, `qa5 0k/1k/2k/8k`
  - remote 28.59.80.196: step500 eval has narrowed to 3 long-worker processes (`302636`–`302638`)
  - remote progress: `outputs/eval_p11_step500/` now has `19` CSVs with parsed row counts `10-100`; `qa5 32k=10` has materialized, while `qa1 32k` / `qa2 32k` are still missing
  - verdict: both evals are healthy; next milestone is to score remote step500 as soon as the final 2 cells appear, then finish temp20 final and compare

[2026-05-17T07:13:18] heartbeat: temp20 healthy through step410; step500 eval confirmed active
  - local temp20: detached tmux `p11_temp20_500_20260517_063303` is healthy through `step 410/500` with checkpoint saves at `100/200/300/400`
  - local QUERY_DIAG: later `top1_sim_mean` at steps `175/196/219/241/270/293/317/342/370/393` = `0.008606 / 0.016479 / 0.012939 / 0.008606 / 0.011230 / 0.007935 / 0.014893 / 0.009033 / 0.015259 / 0.012085`
  - remote 28.59.80.196: not idle; 6 workers are actively evaluating `mem_space_adapter_step000500.pt` into `outputs/eval_p11_step500/` while the tmux session name still says `p11_step4500_eval_20260517_040951`
  - eval progress: `13` CSVs currently materialized with parsed row counts spanning `10-100`; completed cells so far include `qa1 0k/1k`, `qa2 0k/1k`, `qa5 0k/1k/2k`
  - verdict: temp20 remains the strongest 8B mechanism follow-up so far, and the immediate next milestone is `step 500` completion plus step500-checkpoint scoring

[2026-05-17T06:48:50] heartbeat: temp20 tmux run healthy through step150
  - session: p11_temp20_500_20260517_063303
  - log: logs/p11_temp20_500_20260517_063303.log
  - remote 28.59.80.196: all 8 GPUs idle
  - local evidence: step150 reached; QUERY_DIAG top1_sim_mean at steps 25/49/73/99/124/149 = 0.012207 / 0.012756 / 0.007751 / 0.009521 / 0.007263 / 0.009399
  - verdict: detached tmux fixed the launch-lifetime SIGTERM blocker, and temp20 has clearly lifted routing above the old ~0.002 flat floor in early training

[2026-05-17T05:16:34] P11 FSDP 8B eval scored (no re-launch needed)
  - Adapter: outputs/babilong_sft_phase11_fsdp_full/mem_space_adapter.pt (382 keys, 2026-05-17 01:21)
  - Eval already ran 2026-05-17 01:56 -> 03:48 in 6-proc layout (qa{1,2,5} x {short(0k-4k), long(8k-32k)})
  - All 21 cells complete with 100 examples each in outputs/eval_phase1b_p11_final/
  - Scoring CSV: outputs/eval_phase1b_p11_final/p11_fsdp_score.csv
  - Results:
      qa1 mean: 27.71  (P-1B-v2: 45.86  P8: 68.00)
      qa2 mean: 11.29  (P-1B-v2: 18.43  P8: 39.43)
      qa5 mean: 40.00  (P-1B-v2: 48.00  P8: 70.00)
      overall 21-cell mean: 26.33  (P-1B-v2: 37.43  P8: 59.14)
  - VERDICT: P11 FSDP is a REGRESSION vs both P-1B-v2 (-11.10) and P8 (-32.81).
    Long-context (8k-32k) basically collapses: 0-13% per cell. The L1+L3 (no L2) + FSDP-partial training did not generalize.


[2026-05-17T12:08:00] heartbeat: P11 DDP rerun trained + eval running, ghost-cleanup done, remote P8 eval chain healthy
  - local P11 DDP rerun training completed at 11:50:50 with `step 500/500`, final `mem_space_adapter.pt` saved; 21-cell eval auto-launched same minute via the `launch_p11_ddp_500step_validate.sh full` chain (tmux `p11_ddp_500step_eval`)
  - DDP rerun eval at 12:08 has `16/21` CSVs materialized; `qa5_short@4k=91/100`, `qa1_short@2k=41/100`, `qa1_long@16k=59/100`; ETA ~30 min
  - DDP rerun training-side `top1_sim_mean ∈ [0.002136, 0.002274]` for the entire 500 steps — matches P8 canonical and P11 FSDP, confirming archaeology finding that this metric is normal at temp=1.0 not a failure signal
  - 11:54 ghost-cleanup: killed 22 ghost training processes from `outputs/babilong_sft_phase11_fsdp_nogc_500step/` (tmux `p11_fsdp_nogc_500step_20260517_1133_v3`, the `_v3` suffix proved 3 auto-restarts), removed ghost launcher scripts; this had OOM-evicted the original P8+temp20 final eval (TS `20260517_112550`, 0 CSVs)
  - 11:58 remote `28.59.80.196` re-launched P8 eval chain (single tmux `p8_eval_chain_20260517_1157` runs Eval 1 = P8+temp20 final at TS `20260517_1157`, then auto-fires Eval 2 = P8 historical re-eval of the 5/15 ckpt); 6 worker procs healthy on GPUs 0..5, ~29.8 GiB each
  - heartbeat-facing files refreshed with the corrected reality (TRAINER_ACTIVE.md was rewritten because a worker had injected ghost-run narrative into it)
