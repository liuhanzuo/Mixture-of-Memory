# GPU_STATUS.md — 5 节点 GPU 台账（40 卡）
> 每次启动/kill GPU 任务更新。heartbeat 先读→对照 nvidia-smi→台账说跑但空=补卡。★29.162.226.120=dllm 绝不碰。
> 2026-08-08 15:03 更新：用户指令「B200跑resume，H20跑新方向」→ Paper B resume 迁移到 .21/.73。

## ⚠️ 节点调度规则变更（2026-08-08 15:03 用户指令）

> **用户指令**：「B200可以拿去直接跑resume 然后H20跑得比较慢 可以跑你的新方向. 其他H20你想用可以随时kill paperB的resume」
> - `.21` (L20A) → Paper B resume (keep10 等待数据传输完成后启动)
> - `.73` (H20) → Paper B keep8 resume (已运行)
> - `.82` (H20) → keep10 resume 将在 .21 确认运行后被 kill，腾出给新方向
> - `.104` (H20，已交还用户) → keep12 resume 运行中，用户可随时 kill

## ▶️ 当前在跑（2026-08-08 15:03 +08:00 更新）

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **LOCAL** | 8×L20A wzc1 | **空闲**（2026-08-12 01:39 起） | #181 `keep14fresh2_seed1234` 训练**已于 08-12 00:22 跑满 step200000**（`final.pt` + `DONE`）；随后 01:06→01:39 跑完 seed-variance eval battery（seed42+seed1234 × 6 轴，8 卡，33 min），**8 卡已释放 0 MiB** | ✅ 空闲可投 |
| **.21** | 8×L20A wzc1 | `keep10fresh2 resume` | 等待 dolmino 传输完成（ETA ~17:02 CST）；launch script PID 4516 | ⏳ WAITING |
| **.73** | 8×H20 zwfy6 | `keep8fresh2 resume` | step121000_full.pt → step 121160+, 5.79s/step, 73.5GB REMAPPED ✓ | ▶️ 运行中 |
| **.82** | 8×H20 zwfy6 | `keep10fresh2 resume` | step83980+, 6.80s/step, 82.7GB，将在 .21 keep10 确认后被 kill | ▶️ 运行中（待 kill） |
| **.104** | 8×H20 zwfy6 | `keep12fresh2 resume` | step124220+, 7.87s/step, 91.9GB，用户管理 | ▶️ 运行中（用户控制） |

> ✅ **2026-08-12 01:39 更新（task #181 收尾）**：LOCAL 的 keep14 seed1234 训练已 DONE(200k)，seed-variance eval 也已跑完并释放 8 卡 → **LOCAL 现在空闲可投**。
> 结论见 `paperB/SEEDVAR_KEEP14_VERDICT.md`：PPL 种子稳（10.5613 vs 10.5673，0.06%）但 MMLU-L 摆动 −1.19pp、BoolQ +2.05pp；Holm 校正后仅 BoolQ 显著；n=2→df=1 **不可报 σ_run**。
> ⚠️ 本表 .21/.73/.82/.104 四行是 2026-08-08 的旧状态，本次未核实（本任务只准动 LOCAL），下一个 heartbeat 请对照 nvidia-smi 重核。

## 传输进度（2026-08-08 15:02）

| 任务 | 大小 | 进度 | ETA | PID |
|---|---:|---|---|---|
| dolmino .73→.21:/dev/shm/dolmino_now15b_zwfy6.npy | 118 GB | ~1% | ~17:02 CST | local 2741463 |
| step124000.pt .73→.21 wzc1 outputs/keep12fresh2/ | 41 GB | ~1% | ~15:37 CST | local 2741464 |

## ⚡ 下一步（自动）

1. step124000.pt 传输完成 → 自动校验 md5（transfer_ckpt124k_pipe.sh 会输出 MD5_OK）
2. dolmino 传输完成 → keep10 launcher on .21 自动检测 size=126907244672，启动训练
3. keep10 on .21 首步 log 确认 REMAPPED ✓ / all base_lr=2.00e-05 ✓ / rows=15491607 ✓
4. 然后 kill .82 keep10: `kill -9 1418803`（不要 pkill）
5. 更新 GPU_STATUS.md 和 gpu_runs.jsonl

## 📋 历史已完成（2026-08-08）

- P2.4 六臂 SFT sweep 全完成（keep8/10/12/full32/shortgpt16/keep14 pre→post SFT ΔPPL）
- Within-disk floor v3（0 flips / Δcore6=+0.0000 across 4 arms）
- flip-boundary 根因定位 = torch 2.7 vs 2.13 版本差

## 🚫 .104 已交还用户（2026-08-05 15:4x，用户指令）

> **`.104`（28.83.24.104）心跳不纳管，但 keep12 resume 在上面跑，用户说可随时 kill**。heartbeat 不主动操作 .104。

---
> 旧台账（2026-08-08 06:25-08:28 的 P2.4 SFT sweep 与 within-disk floor v3 记录）已归档到 `status/GPU_STATUS_archive_20260808_0828.md`（如存在）。

---

## 2026-08-11 21:22–21:34 +08:00 — `.73` paperG gate-2 (task #248) 已完成，卡已释放

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.73** | 8×H20 zwfy6 | `gate2_mc_letter_content` (#248) | paperG gate-2 全量复现：MMLU 同口径 letter-vs-content × 6 个非 MMLU MC benchmark × 6 arm × 8 shard = 36 cell；21:22 起，21:34 完；36/36 cell 8/8 shard、n_scored==expected、n_nan=0；失败语法 grep 零命中 | ✅ 完成，GPU 0-7 已释放（0 MiB） |

- 结论：`REPLICATES_PARTIALLY_AND_NARROWS_THE_CLAIM`，详见 `paperG/evidence/SECOND_MC_BENCHMARK_VERDICT.md`。
- 上面 2026-08-08 那张表里 `.73 = keep8fresh2 resume ▶️ 运行中` 已过期：本次占卡前 `nvidia-smi` 实测 8 卡全 0 MiB、无 compute app。**未 kill 任何进程**（.73 上另一 agent 的纯 CPU jsonl 重算未受影响）。
- 结果落盘两盘：`olmo2_mc_letter_content_results/`（zwfy6 + wzc1，各 52 MB，各自校验完整）。

---

## 2026-08-11 22:19 +08:00 — `.73` paperG gate-2 CROSS-FAMILY (task #250) 启动

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.73** | 8×H20 zwfy6 | `gate2_xf` (#250) | paperG gate-2 跨家族扩展：#248 harness 原封不动跑 **非 OLMo** 三家族 × 5 arm × 6 task。arm = {llama2_7b, llama3_8b, qwen3_8b_base} × {base, k8, k10, k12, k14}，damage = **eval 期 front-N truncation（无 fresh block、无 heal）**，与 gate-1 DAMAGED 同构造故与已归档 MMLU 数字直接可比。8 shard/arm，bs=48（与 #248 同值），22:19 起 | ▶️ 运行中 |

- 占卡前 `nvidia-smi` 实测 8 卡全 0 MiB、无 compute app；**未 kill 任何进程**（.73 上另一 agent 的纯 CPU 工作不受影响）。
- 前置：`Qwen3-8B-Base` 原本 **wzc1-only**，已 `scp -O` 16 GB 到 zwfy6，12 个文件 md5 全对。⚠️ zwfy6 原有的 `models/Qwen--Qwen3-8b`（含 `Qwen3-8b-local` symlink）是 **Instruct** 模型（`eos=151645` im_end、有 chat_template、40960 ctx），**不能当 base arm 用**。
- driver `scripts/_run_mc_letter_content_crossfamily_8gpu.sh`（wzc1 写、`scp -O` 到 zwfy6、md5 `aef912ce` 双端一致），log `zwfy6:logs/gate2_xf_DRIVER.log`，结果 `zwfy6:mc_lc_crossfamily_results/`。

## 2026-08-11 22:32 +08:00 — `.73` #250 scoring完成，8 卡已释放（0 MiB，无 compute app）

- 15 arm × 6 task = **90 cell 全部 8/8 shard、`n_scored==EXPECTED_N`、`n_nan=0`**；128 个 shard/merge log 跑失败语法 grep（`Traceback (most recent call last)` / `CUDA out of memory` / `AssertionError` / `*INTEGRITY FAILURE` / `CARDINALITY FAILURE`）**零命中**。
- 判定 `REPLICATES_IN_DIRECTION_ACROSS_FAMILIES_BUT_THE_LADDER_DOES_NOT`，详见 `paperG/evidence/GATE2_CROSSFAMILY_VERDICT.md`。
- ⚠️ driver log 里每个 arm 都有一条**假的** `MERGE FAIL ...: 0/6 tasks merged` —— 是 driver 自检 `grep -c "^\[merge\]"` 锚了 `^` 而 `_log()` 会加时间戳前缀，纯 cosmetic bug，**跑完后已修**。15/15 arm 实际都成功 merge（各 6/6 `summary_<task>.json`，已 `ls` 逐个核对）。
- 后续 nulls/统计是 **CPU-only**，在 LOCAL(wzc1) 跑（MMLU cross-family per-item 记录 190MB 只在 wzc1），未占任何 GPU。
- **未 kill 任何非本任务进程**；.73 上另一 agent 的纯 CPU 工作不受影响。

---

## 2026-08-11 23:51 – 23:55 +08:00 — `.82` A03 seed45 四轴 eval (task #243) 已完成，卡已释放

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.82** | 8×H20 zwfy6 | `A03_1B_dataorder_seed45_step220000` (#243) | A03 dataorder 第三个（也是最后一个）pre-registered sampler seed 的四轴 eval：mmlu_content + popqa/triviaqa + nq_open，8-way sharded GPU0-7。23:51:01 起，23:54:46 全完（MMLU 81s / CB(pt) 104s / CB(nq) 40s） | ✅ 完成，GPU 0-7 已释放（0 MiB） |

- **占卡前实测**：`.82` 8 卡全 0 MiB、无 trainer/watcher 残留进程（seed45 训练已于 23:29:10 由自己的 watcher 停在 step220000，wrapper 已退出）。**未 kill 任何进程。**
- **ckpt 完整性（关键，因为 driver 带的是 v1 停止竞态守卫）**：`step220000.pt` 与 `step205000/210000/215000.pt` **字节数完全相同 = 12,181,311,650 B（delta +0 B）**；ext driver 自带的 `torch.load(weights_only=False)` 探针独立返回 ok。曾把 Arm 4 的 step220000.pt 截到 49% 的那个竞态**这次没触发**。trainer `rc=1` 是 `kill -TERM` 的预期返回码，不是崩溃。
- **shard 完整性**：四轴 ×（arm + baseline）= 8 个 cell **全部 8/8 shard、`n_scored == expected`（popqa 14267 / triviaqa 17944 / nq_open 3610 / mmlu 14042）、0 重复 item_id、0 nan**；MMLU `summary.json` 独立报 `n_valid=14042 n_nan=0`。失败语法 grep（`Traceback (most recent call last)` / `CUDA out of memory` / `loss=nan`）**零命中**。
  ⚠️ 不要用 `grep -icE 'nan'` 判失败——它会命中 harness 自己的**通过**行 `✓ No NaN/Inf in model parameters`。
- **结果**：primary 轴 triviaqa em **θ = −0.3622 pp，CI95 [−0.5517, −0.1838]，SIG 负 → NOT-CONFIRM**；聚合 **0/3 CONFIRM → ARTIFACT**（维持现状，A-2 仍撤回）。σ_run：keep7-20k → S=4/df=3/s=0.4039 pp；pooled → **df=5 / 0.3666 pp / χ² [0.229, 0.899]**。
- 判定文档 `proposal/archive/A03-parametric-vs-external-memory/SEED45_VERDICT.md`；证据三份 JSON 在同目录 `evidence/`（两盘 md5 一致）。
- 后续配对差分 + σ 重算是 **CPU-only**（在 `.82` 上跑，各 <1 min），未额外占卡。
