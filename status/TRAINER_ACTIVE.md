# TRAINER_ACTIVE.md — Active Training Runs

## Updated 2026-05-13 15:35 CST (heartbeat)

> `H_V2_PLAN.md` 现视为常态化 plan 文件，heartbeat 在处理状态后可以直接更新。
>
> 本轮 heartbeat 继续按用户最新前提执行：**稳定 B200 `b200-1..4` 仍在，replacement B200 `b200-5..8` 视为 unavailable for auto-launch**。
>
> 本轮 heartbeat 的状态结论未反转上一轮关键判断：
> 1. `b200-2` 的 `H-v2 C Phase 1` 仍保持 **已停止** 状态，节点持续空闲；旧日志尾部继续佐证其在更晚窗口仍反复出现 `eval_loss=nan`。
> 2. `h20-1` 的 `MPlus-8B-smoke-grid-20260513` 仍然对应 **punctuation-only generation**（样例输出 `!!!!!!!!!!!!!!!!!!!!`），不是空字符串。
> 3. `h20-2/3/4` 的 RMT retry3 仍然是 **legacy evaluator / v10 checkpoint mismatch**，不是 checkpoint memory 文件损坏。
> 4. 本轮已按 HEARTBEAT.md 的主动调研要求，再次发起两条更聚焦的后台跟进：`general-purpose-20` 研究 RMT v10 官方/兼容 eval path，`general-purpose-21` 审核 MPlus live import / RoPE 修复路径。

## Active H-series / baseline reproduction

### b200-1 — H-v2 A Phase 2 (bs upgraded)
- Status: **running**
- Phase 1 checkpoint: `outputs/h_v2_phase1_A_b2001/checkpoint_final.pt`
- Resumed from: `outputs/h_v2_phase2_A_b2001/segments_4/checkpoint_final.pt`
- Phase 2 log: `logs/h_v2_phase2_A_b2001_bs.log`
- Phase 2 output dir: `outputs/h_v2_phase2_A_b2001/`
- Current stage: curriculum `segments=8` in tmux `v2_phase2_A_b2001_bs`
- Master port: `29610`
- Latest observed progress: `step=340`, recent loss around `0.3457`, log still updating normally (`2026-05-13 15:30 CST`)
- GPUs: local 8×L20A workers still在线；显存约 `129.8 GiB / 183 GiB` 每卡，util 快照有波动但日志持续推进，判定为 **healthy**
- Batch: `--batch_size 32 --gradient_accumulation_steps 2` → **effective_batch=512**

### b200-2 — H-v2 C Phase 1 (ARMT reference)
- Status: **killed by heartbeat due persistent eval anomaly**
- Launch script: `scripts/v2_phase1/train_v2_C_armt.sh`
- tmux: `v2_phase1_C_b2002` (**killed last round; still absent this round**)
- Phase 1 log: `logs/h_v2_phase1_C_b2002.log`
- Output dir: `outputs/h_v2_phase1_C_b2002/`
- Launch timeline:
  - `2026-05-13 13:11 CST`: initial launch used project `.venv` and failed immediately with `ModuleNotFoundError: No module named 'fla'`
  - `2026-05-13 13:31 CST`: heartbeat verified `/opt/conda/envs/torch-base/bin/python` can import `torch/transformers/accelerate/fla` plus `modeling_amt.online_armt`, then relaunched the same script with `PYTHON_BIN=/opt/conda/envs/torch-base/bin/python`
- Failure evidence after relaunch:
  - train loop continued to `step≈384` with finite train loss (`~2.784`)
  - but eval emitted **persistent** `eval_loss=nan` from early checkpoints onward, not a one-off
  - latest observed anomaly window in log tail reached `epoch 0.03569` with `eval_loss='nan'`
- Action taken: heartbeat killed the tmux session and rechecked `nvidia-smi`; b200-2 remains **fully idle** this round
- Next action: inspect `third_party/associative-recurrent-memory-transformer/run_finetuning_lm_rmt_hf.py` eval path / labels / metric handling before any relaunch of C on stable B200
- Previous occupant: HMT `hmt_full_resume35000` completed successfully; final tail shows `PPL on 100 test samples: 11.405261888504029`

### b200-3 — H-v2 D Phase 2 (bs upgraded)
- Status: **running**
- Phase 1 checkpoint: `outputs/h_v2_phase1_D_b2003/checkpoint_final.pt`
- Resumed from: `outputs/h_v2_phase2_D_b2003/segments_4/checkpoint_final.pt`
- Phase 2 log: `logs/h_v2_phase2_D_b2003_bs.log`
- Phase 2 output dir: `outputs/h_v2_phase2_D_b2003/`
- Current stage: curriculum `segments=8` in tmux `v2_phase2_D_b2003_bs`
- Master port: `29611`
- Latest observed progress: `step=320`, recent loss around `0.0061`, log still updating normally (`2026-05-13 15:30 CST`)
- GPUs: remote 8×L20A occupied, `~135.4 GiB / 183 GiB` per GPU；util 快照可见瞬时 0%，但日志持续推进，判定为 **healthy**
- Batch: `--batch_size 32 --gradient_accumulation_steps 2` → **effective_batch=512**
- Follow-up candidate: researcher 最新高置信建议仍是 D 的下一条 Phase 2 ablation 改成 `qa1+qa2+qa3` multi-task mix，但应在当前 qa1 curriculum 跑完后再开

### b200-4 — H-v2 B Phase 2 (bs upgraded)
- Status: **running**
- Phase 1 checkpoint: `outputs/h_v2_phase1_B_b2004/checkpoint_final.pt`
- Resumed from: `outputs/h_v2_phase2_B_b2004/segments_4/checkpoint_final.pt`
- Phase 2 log: `logs/h_v2_phase2_B_b2004_bs.log`
- Phase 2 output dir: `outputs/h_v2_phase2_B_b2004/`
- Current stage: curriculum `segments=8` in tmux `v2_phase2_B_b2004_bs`
- Master port: `29612`
- Latest observed progress: `step=360`, recent loss around `0.0338`, log still updating normally (`2026-05-13 15:30 CST`)
- GPUs: remote 8×L20A occupied, `~165.8 GiB / 183 GiB` per GPU；其中个别卡 util 快照短暂为 0%，但整体日志持续推进，判定为 **healthy**
- Batch: `--batch_size 16 --gradient_accumulation_steps 4` → **effective_batch=512**

### H20 — latest eval/debug state
- Status: **reachable; no active eval workers remain after this round's smokes**
- `h20-1`: `MPlus-8B-smoke-grid-20260513` finished
  - scope: `qa1 × {1k,2k,4k,8k,16k,32k}`
  - result: current smoke csv sample is **not blank**; `qa1_1k` row shows `output='!!!!!!!!!!!!!!!!!!!!'`, `correct=0`
  - conclusion: current blocker is **punctuation-only generation / semantic degeneration**, not runtime crash and not empty-string extraction
  - log evidence: `logs/babilong_mplus_1k_20260513_1414.log`, `logs/babilong_mplus_32k_20260513_1414.log`
- `h20-2`: `RMT PPL smoke retry3` finished and failed
  - log: `logs/rmt_eval_ppl_h20_2_retry3_20260513.log`
  - failure: `legacy/scripts/eval_rmt.py` expected legacy `memory_embeddings/extractor.*` keys but checkpoint contains v10 keys `l0.*`, `recon_head.*`
- `h20-3`: `RMT NIH smoke retry3` finished and failed with the same loader mismatch
  - log: `logs/rmt_eval_nih_h20_3_retry3_20260513.log`
- `h20-4`: `RMT memory smoke retry3` finished and failed with the same loader mismatch
  - log: `logs/rmt_eval_mem_h20_4_retry3_20260513.log`
- Extra validation already on disk:
  - H20-side `legacy/scripts/debug_eval_rmt_v10.py --skip_model` confirms `outputs/rmt_v10_20260419_182044` carries **valid v10 memory keys** (`l0.memory`, `recon_head.proj.*`)
  - Therefore current blocker is the **wrong evaluator entrypoint**, not an obviously wrong `rmt_memory.pt` schema
- Current proactive follow-up in progress:
  - `general-purpose-20`: focused research on the correct v10-aware / official RMT BABILong eval path
  - `general-purpose-21`: focused audit of MPlus live import / RoPE remediation path

### replacement B200 — current policy
- Current node mapping:
  - `b200-5 = 28.89.18.252`
  - `b200-6 = 28.89.20.82`
  - `b200-7 = 28.89.20.27`
  - `b200-8 = 28.89.18.19`
- Status: **treat as unavailable for current auto-launches**
- Current probe result:
  - `b200-5`: `Permission denied`
  - `b200-6`: `Permission denied`
  - `b200-7`: `Connection refused`
  - `b200-8`: `Connection refused`
- Current rule: 在认证与连通性恢复稳定前，不把 replacement B200 作为调度面的一部分
- Password file: `configs/password_b200_ephemeral.txt`

## Current blockers
- `b200-2` 上的 H-v2 C 已被 heartbeat 停止：persistent `eval_loss=nan` 说明当前 Phase 1 eval path 不健康，**未修复前不应直接重启**
- `MPlus-8B` 当前主要 blocker 是 **punctuation-only generation**，不是 runtime crash，也不是 empty-string extraction
- `RMT` H20 retry3 的 blocker 是 **legacy evaluator mismatch**：当前 `legacy/scripts/eval_rmt.py` 不适配 v10 checkpoint
- `ARMT` 的 H20 侧前置数据/代码已补齐，但 **ARMT checkpoint 仍未在 H20 shared mount 上确认 ready**
- replacement B200 仍不可用于 Mixture-of-Memory auto-launch
- repo 当前存在较多未提交代码/状态修改（`git status --short` 非空），heartbeat 仅记录 WARNING，不在本轮擅自整理
- `CODE_CLEANUP_SUGGESTIONS.md` 近 7 天内有更新，但当前仅保留 medium-confidence 人工确认项；未发现新的 high-confidence 清理动作需要在本轮立即执行

## Busy nodes
| Node | Status |
|------|--------|
| **b200-1** | **H-v2 A Phase 2 running** |
| **b200-2** | **idle after heartbeat killed H-v2 C** |
| **b200-3** | **H-v2 D Phase 2 running** |
| **b200-4** | **H-v2 B Phase 2 running** |
| **h20-1** | **idle after MPlus smoke-grid completed** |
| **h20-2** | **idle after RMT PPL retry3 failed** |
| **h20-3** | **idle after RMT NIH retry3 failed** |
| **h20-4** | **idle after RMT memory retry3 failed** |

## Replacement B200
| Node | Status |
|------|--------|
| **b200-5** | **Permission denied** |
| **b200-6** | **Permission denied** |
| **b200-7** | **Connection refused** |
| **b200-8** | **Connection refused** |

## H20 probes
| Node | Status |
|------|--------|
| **h20-1** | **28.58.244.13 reachable, idle after MPlus smoke-grid** |
| **h20-2** | **28.85.54.125 reachable, idle after RMT retry3 failure** |
| **h20-3** | **28.59.5.176 reachable, idle after RMT retry3 failure** |
| **h20-4** | **28.83.52.26 reachable, idle after RMT retry3 failure** |
