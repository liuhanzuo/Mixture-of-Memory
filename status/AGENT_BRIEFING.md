# AGENT_BRIEFING.md — 从零开始的 Agent 完整交接文档

> **目标读者**：一个对本项目一无所知的新 agent。读完本文档，你应该能独立完成 heartbeat 巡检、理解当前实验状态、并知道接下来要分析什么。
> **最后更新**：2026-06-25 12:30 GMT+8

---

## 第一部分：我们在做什么（研究背景）

### 1.1 研究问题

我们在研究 **LLM 的长程记忆问题**：如何让一个 8B LLM（Llama-3-8B）在处理几万 token 的超长文档时，仍能准确回忆起文档开头的信息。

**核心发现（过去几个月）**：

我们自己开发的 `mem_space` 系统（一个插入 Llama 各层的 memory bank）存在**"读出鸿沟"**——

- memory bank 里**确实存储了**长程信息（几千 token 之前的内容）
- 但标准的 W0（纯 memory readout）**读不出来**，长程准确率接近随机
- 用 SWA（Sliding Window Attention，让模型在 eval 时能直接"看"到历史 chunk 的部分 token）才能读出来
- W6 比 W0 的 BABILong 32k qa5 分数高出 3× 以上（23 vs 8），证明信息在 bank 里

**类比**：知识存在大脑里，但嘴说不出来——不是记忆问题，是提取问题。

### 1.2 当前方案：方案B（FIFO Memory 对标 MemoryLLM）

我们参考 **MemoryLLM**（一篇顶会论文）的架构，用类似的 FIFO 写入方式重新实现了一套 memory：

- 每个 chunk forward 后，取各层 hidden states 写入 memory bank（FIFO 先进先出）
- buffer 存 `num_blocks` 个块的历史，新块进来最老的块被淘汰
- eval 时全量 memory tokens 做 full attention（类似 MemoryLLM 的做法）

**MemoryLLM 的 BABILong 基准**（这是我们要超越的目标）：
```
任务     0k   1k   2k   4k   8k  16k  32k
qa1      53   42   32   23   14    9    7
qa2      36   35   19   16   15   16   16
qa5      47   50   45   39   39   38   34   ← 32k=34，我们目前最好只有 23
```

### 1.3 方案B 消融实验设计（4臂）

我们在 4 个节点上并行训练，探索两个维度的影响：
- **chunk_size**（每次写入 memory 的 token 数）：512 vs 1024
- **buffer_length**（memory 保留多少历史块）：b25（25块=12800tok） vs b50（50块=25600tok） vs b100（100块=51200tok）

| 臂 | 节点 | chunk_size | buffer_length | 实验目的 |
|---|---|---|---|---|
| A | B200.53 | 1024 | 50 | 更大 chunk，够大显存（L20A 183GB） |
| B | 本机 | 512 | 50 | 基准，对标 MemoryLLM 默认配置 |
| C | .245.174 | 512 | 100 | buffer 加倍，看远程信息是否更多 |
| D | .7.53 | 512 | 25 | buffer 减半，控制变量 |

---

## 第二部分：当前状态（2026-06-25 12:30）

### 2.1 训练状态

**4 臂全部训练完成 step3000/3000：**

| 臂 | 节点 | 完成时间 | 训练时长 | ckpt 位置 |
|---|---|---|---|---|
| chunk1024/b50 | B200.53 | Jun24 ~22:00 | 297.1min | `outputs/mem_space_fifo_b50_chunk1024/` (wzc1盘) |
| chunk512/b50 | 本机 | Jun25 07:12 | 624.1min | `outputs/mem_space_fifo_b50_chunk512/full_model.pt` |
| chunk512/b100 | .245.174 | Jun25 07:03 | 621.3min | `outputs/mem_space_fifo_b100_chunk512/full_model.pt` (diskB) |
| chunk512/b25 | .7.53 | Jun25 07:07 | 622.0min | `outputs/mem_space_fifo_b25_chunk512/full_model.pt` (diskB) |

**注意**：diskB 路径是 `/apdcephfs_zwfy6/share_304376610/...`（304376610），不是本机的 303098609。

### 2.2 Eval 状态（2026-06-25 12:10 更新）

**BABILong eval** 使用 `scripts/_eval_taskpool_2group.sh`，评测 qa1/qa2/qa5 × 0k/1k/2k/4k/8k/16k/32k = 21 个任务，每个任务 100 样本。

每个臂需要跑 **W0**（swa_eval_chunks=0，纯 memory readout）和 **W6**（swa_eval_chunks=6，memory + 6个历史 chunk SWA），分别量化"memory 本身的效果"和"memory+近窗口的上界"。

| 臂 | 节点 | W0 eval | W6 eval |
|---|---|---|---|
| chunk1024/b50 | B200.53 | **✅ 完成**（结果见2.3） | **进行中**（qa5 4k/32k，~1-2h） |
| chunk512/b50 | 本机 | **进行中**（qa1 32k，08:29起） | 排队（W0后自动启动） |
| chunk512/b100 | .245.174 | **进行中**（qa1 32k，09:31起） | 排队（W0后自动启动） |
| chunk512/b25 | .7.53 | **进行中**（qa5 32k，10:43起） | 排队（W0后自动启动） |

**日志位置**：
- B200.53（wzc1盘）：`/apdcephfs_wzc1/share_304376610/.../logs/fifo_b50_c1024_eval_W0.out` / `_W6.out`
- 本机：`logs/fifo_b50_c512_eval_W0.out`（通过 `fifo_b50_c512_eval_driver.log` 驱动）
- .245.174：`logs/fifo_b100_c512_eval_W0.out`（在 diskB）
- .7.53：`logs/fifo_b25_c512_eval_W0.out`（在 diskB）

### 2.3 已有结果

**B200 chunk1024/b50 W0 结果（step3000）**：
```
task   0k   1k   2k   4k   8k  16k  32k
qa1    73   53    6    1    4   21    2   ← 32k 接近随机
qa2    54   30    8    1    2    6    2   ← 同样崩
qa5    35   39   12   21   10   15    8   ← 稍好，但远低于 MemoryLLM 34
```

**初步解读**：chunk1024 配置的 W0 长程很差（qa1 32k = 2），读出鸿沟依然存在。需要等 W6 结果看 bank 里是否有信息。

**历史对照（上一代 mass_coef 系列，chunk512）**：
```
配置                qa5: 8k  16k  32k
s1234 W0            16   12   11
s1234 W6            36   33   23   ← W6 >> W0，确认鸿沟
mass_coef2 W6       54   36   23   ← 最强
MemoryLLM baseline  39   38   34   ← 目标线
```

---

## 第三部分：Heartbeat 巡检指南

### 3.1 什么是 Heartbeat

Heartbeat 是对所有节点的周期性健康巡检，检查：
1. 每个节点的 GPU 状态（显存/利用率）
2. 训练/eval 进程是否存活
3. 日志最新进度
4. 有无 OOM/hang/crash

发现问题时**自主处理**（kill + 重启），完成后更新状态文件。

### 3.2 节点信息

| 节点 | 地址 | 端口 | 密码文件 | 项目根路径 |
|---|---|---|---|---|
| 本机 | localhost | — | — | `/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory` |
| .196 | 本机共享FS，用于多机训练 | — | `configs/password_diskA.txt` | 同本机（共享FS） |
| .245.174 | `28.58.245.174` | 22 | `configs/password_h20_returned.txt` | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory` |
| .7.53 | `28.48.7.53` | 22 | `configs/password_h20_returned.txt` | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory` |
| B200.53 | `28.88.184.53` | **36000** | `configs/password_b200_53.txt` | `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory` |

**注意**：
- `.7.53` 和 `.245.174` 共用同一个密码文件 `password_h20_returned.txt`
- B200.53 SSH 端口是 **36000**，不是默认 22
- B200 项目路径在 **wzc1 盘**（`apdcephfs_wzc1/share_304376610`），不同于 diskB（`zwfy6/share_304376610`）

### 3.3 SSH 命令模板

```bash
# 远程节点通用模板（以 .245.174 为例）
sshpass -f /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/configs/password_h20_returned.txt \
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password \
  root@28.58.245.174 "命令"

# B200.53 专用（端口 36000）
sshpass -f /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/configs/password_b200_53.txt \
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password \
  -p 36000 root@28.88.184.53 "命令"
```

### 3.4 标准巡检步骤

#### Step 1：检查 GPU 状态

```bash
# 在每个节点执行
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

正常状态：
- 训练中：8 卡全满（~80-95GB / 100%）
- Eval 中：GPU 高显存占用，利用率 100%
- 空闲：0 MiB / 0%

异常信号：
- 预期应该在跑，但 GPU 全部 0 MiB → 进程已死，需要重启
- GPU 部分有值、部分为 0 → 可能只启动了部分 shard，查进程

#### Step 2：检查进程

```bash
# 训练进程（torchrun）
ps aux | grep torchrun | grep -v grep

# Eval 进程（run_babilong）
ps aux | grep run_babilong | grep -v grep

# 查所有 python 进程概览
ps aux | grep python | grep -v grep | wc -l
```

#### Step 3：查日志尾部

```bash
# 训练日志（本机示例）
tail -5 logs/mem_space_fifo_b50_chunk512.log

# Eval 日志（看进度）
tail -10 logs/fifo_b50_c512_eval_W0.out
```

训练日志正常输出示例：
```
2026-06-25 07:12:10,105 - INFO - [step 3000/3000] lm=1.305 time=624.1 min
2026-06-25 07:12:10,105 - INFO - Training complete: steps=3000 ...
```

Eval 日志正常输出示例：
```
[Thu Jun 25 11:09:33 CST 2026] GROUP0 done ck0 qa5 2k
[Thu Jun 25 11:09:33 CST 2026] GROUP0 -> ck0 qa5 4k
```

Eval 完成后会打印聚合结果：
```
=== fifo_b50_c1024_step3000_W0 ===
task       0k      1k      2k      4k      8k     16k     32k
qa1       73     53      6      1      4     21      2
```

#### Step 4：更新状态文件

每次巡检完成后，追加一条到 `status/TRAINER_ACTIVITY.jsonl`：

```bash
cat >> /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/status/TRAINER_ACTIVITY.jsonl << 'EOF'
{"ts":"2026-06-25T12:00:00+08:00","event":"heartbeat","note":"简要描述各节点状态"}
EOF
```

重大状态变化（eval 完成、训练完成、发现 crash 等）还需更新 `status/SESSION_HANDOFF.md`。

### 3.5 常见问题处理

#### 问题1：Eval 进程挂了（GPU 显存降到 0，日志不再更新）

诊断：
```bash
# 1. 查日志最后时间戳
tail -3 logs/fifo_XXX_eval_W0.out

# 2. 查进程
ps aux | grep run_babilong | grep -v grep

# 3. 查 results 目录
ls babilong_results/XXX/ | wc -l
```

处理：
```bash
# kill 所有 eval python 进程
ps aux | grep run_babilong | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
# 重启（见第四部分的启动命令）
```

#### 问题2：eval skip 大量任务（`[skip] ... (complete)`）

这是**正常行为**：eval 脚本会检查 results 目录，已有结果的 shard 跳过，直接从上次中断位置继续。不需要处理。

#### 问题3：训练 OOM（仅 H20 节点，95GB 显存）

历史根因：`unfreeze_from=16`（解冻 16 层）导致 optimizer states ~14GB，在 step30 OOM。
修复：必须用 `unfreeze_from=24`（只解冻最后 8 层）。B200 L20A 183GB 不受此限制。

#### 问题4：HF dataset lock 陷阱

OOM 崩溃后 `.hf_cache/datasets/*.lock` 残留，下次启动进程会在 BABILong prefetch 后静默退出（GPU 0%，无 traceback）。

修复：
```bash
rm -f .hf_cache/datasets/*.lock
```

---

## 第四部分：Eval 启动命令（需要时手动重启用）

### 4.1 标准 eval 脚本用法

`scripts/_eval_taskpool_2group.sh` 是通用调度器，通过环境变量传参：

```bash
# 必需参数
RUN_PREFIX=xxx           # 日志目录前缀
CKPT_FILES="path/to.pt"  # ckpt 路径（空格分隔多个）
CK_NAMES="run_name"      # 每个 ckpt 的名称（与 CKPT_FILES 对齐）
ADAPTER_CONFIG="path/to/adapter_config.json"
CHUNK_SIZE=512           # 必须与训练时一致

# 可选参数
EXTRA_ARGS="--swa_eval_chunks 0"  # W0: =0; W6: =6
PROJECT_ROOT=/path/to/project     # 默认本机路径
PYTHON_BIN=/path/to/.venv/bin/python
```

### 4.2 各节点 eval 启动示例

**本机 chunk512/b50 W0（如需重启）：**
```bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
CK=outputs/mem_space_fifo_b50_chunk512/full_model.pt
CFG=outputs/mem_space_fifo_b50_chunk512/adapter_config.json

nohup bash -c "
  export WANDB_MODE=offline
  RUN_PREFIX=fifo_b50_c512_W0 \
  CKPT_FILES=\"$CK\" CK_NAMES=\"fifo_b50_c512_final_W0\" \
  ADAPTER_CONFIG=\"$CFG\" CHUNK_SIZE=512 \
  EXTRA_ARGS=\"--swa_eval_chunks 0\" \
  PROJECT_ROOT=$ROOT PYTHON_BIN=$ROOT/.venv/bin/python \
  bash scripts/_eval_taskpool_2group.sh > logs/fifo_b50_c512_eval_W0.out 2>&1
" &
echo "PID=$!"
```

**远程节点（.245.174）chunk512/b100 W0（如需重启）：**
```bash
ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
sshpass -f configs/password_h20_returned.txt \
  ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password root@28.58.245.174 "
  cd $ROOT && export WANDB_MODE=offline
  CK=outputs/mem_space_fifo_b100_chunk512/full_model.pt
  CFG=outputs/mem_space_fifo_b100_chunk512/adapter_config.json
  nohup bash -c '
    RUN_PREFIX=fifo_b100_c512_W0 \
    CKPT_FILES=\"\$CK\" CK_NAMES=\"fifo_b100_c512_final_W0\" \
    ADAPTER_CONFIG=\"\$CFG\" CHUNK_SIZE=512 \
    EXTRA_ARGS=\"--swa_eval_chunks 0\" \
    PROJECT_ROOT=$ROOT PYTHON_BIN=$ROOT/.venv/bin/python \
    bash scripts/_eval_taskpool_2group.sh > logs/fifo_b100_c512_eval_W0.out 2>&1
  ' &
  echo PID=\$!
"
```

### 4.3 聚合结果

eval 完成后，用以下命令聚合 4 shards：

```bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
.venv/bin/python scripts/score_nested_babilong.py \
  --results_folder babilong_results/fifo_b50_c512_final_W0 \
  --output_name fifo_b50_c512_final_W0
```

或者 eval 脚本结束时会自动打印聚合表格（见日志末尾 `=== xxx ===` 区块）。

---

## 第五部分：接下来要分析什么

### 5.1 即时待办（今天，~Jun25 14:00-16:00）

当所有 W0 eval 完成后，**立即聚合并分析**：

1. **对比 4 臂 W0 结果**：
   - 横轴：chunk_size（512 vs 1024），纵轴：buffer_length（b25/b50/b100）
   - 重点看 8k/16k/32k 长程分数
   - 问题：更大的 chunk 或更长的 buffer 能改善长程吗？

2. **对比 MemoryLLM baseline**：
   - MemoryLLM qa5 32k = 34，我们的 B200 chunk1024/b50 W0 = 8
   - 目标：看哪个配置最接近 MemoryLLM，找到差距主因

3. **W0 vs W6 鸿沟分析**：
   - 等 B200 W6 完成（~13:00）
   - W6 应该能读出更多信息，如果 W6 >> W0，说明信息在 bank 里但读不出
   - 如果 W6 ≈ W0 且都很低，说明 FIFO 写入本身有问题

### 5.2 预期结论形态

**情形A（最可能）**：W0 全部很差，W6 有提升但仍不及 MemoryLLM
- 含义：读出鸿沟依然是主要障碍，FIFO 写入没有根本解决问题
- 下一步：**方案C（蒸馏）**——用 MemoryLLM teacher 蒸馏 mem_space student，直接学习读出能力

**情形B**：某个配置（如 b100/chunk512）W0 接近 MemoryLLM
- 含义：buffer 越长越有利，验证了 FIFO 方向
- 下一步：进一步扩大 buffer，或改进读出机制

**情形C**：W0 全部 ≈ MemoryLLM，W6 >> W0
- 含义：FIFO 写入解决了写入问题，但读出仍是瓶颈（与历史结论一致）
- 下一步：专门优化读出（distillation 或 SWA-train）

### 5.3 全部 eval 完成后的分析流程

```bash
# 1. 在每个节点查 eval 日志尾部，确认完成
# B200
ssh B200.53 "tail -20 .../logs/fifo_b50_c1024_eval_W0.out"
ssh B200.53 "tail -20 .../logs/fifo_b50_c1024_eval_W6.out"
# 本机
tail -20 logs/fifo_b50_c512_eval_driver.log
# .245.174
ssh .245.174 "tail -20 .../logs/fifo_b100_c512_eval_W0.out"
# .7.53
ssh .7.53 "tail -20 .../logs/fifo_b25_c512_eval_W0.out"

# 2. 从每个 eval 日志末尾提取聚合表格（=== XXX === 区块）

# 3. 汇总 4 臂结果，更新 status/RUN_REGISTRY.md 和 status/SESSION_HANDOFF.md

# 4. 决定下一步方向
```

### 5.4 更长期方向（如果 W0 表现差）

**方案C（蒸馏）**：
- Teacher：MemoryLLM（已有模型 `.hf_cache/models--YuWangX--memoryllm-8b-chat`，**只在 diskA/本机**）
- Student：我们的 mem_space（FIFO 版）
- 目标：让 student 的 W0 readout 直接模仿 teacher 的输出分布
- 历史经验：self-study 蒸馏（用 Llama full-context 当 teacher）在 BABILong qa1 8k=19，有效但有限

---

## 第六部分：关键文件索引

```
项目根：/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/

状态文件：
  status/SESSION_HANDOFF.md      — 当前实验快照（最常读的文件）
  status/TRAINER_ACTIVITY.jsonl  — 巡检流水账（时间序列）
  status/RUN_REGISTRY.md         — 所有 run 配置 + 结果对照表（最权威）
  status/BENCHMARK_RESULTS.md    — 结果汇总（含外部论文数字）

核心代码：
  src/memory/mem_space/layer.py  — mem_space 核心层实现
  src/memory/mem_space/config.py — MemorySpaceConfig dataclass
  scripts/_eval_taskpool_2group.sh — 标准 eval 调度脚本
  scripts/run_babilong_mem_space.py — BABILong 单任务 eval 脚本
  scripts/score_nested_babilong.py — 聚合 shard 结果

训练脚本（方案B）：
  scripts/train_mem_space_fifo_b50_chunk1024_b200.sh — B200 臂
  scripts/train_mem_space_fifo_b*.sh — 其他臂

输出目录：
  outputs/mem_space_fifo_b50_chunk1024/  — B200 臂 ckpt（wzc1盘）
  outputs/mem_space_fifo_b50_chunk512/   — 本机臂 ckpt
  babilong_results/fifo_*/               — eval CSV 结果

日志：
  logs/mem_space_fifo_*.log              — 训练日志
  logs/fifo_*_eval_W{0,6}.out           — eval 日志
```

---

## 附录：重要历史结论（不要重走这些弯路）

1. **eval_interval≠0 会导致 NCCL 崩溃**：训练时必须 `eval_interval=0`，不能开内联 eval。

2. **SWA 增益不 transfer 到下游任务**：RULER/LongEval 测试中 combined W0/W2/W4 全部输出 comma-spam，SWA 效果只在 BABILong 上体现，不是真正的"理解"能力。

3. **step500 通常是最佳点**：过训后长程分数单调退化，历史多次验证 step500 > step1000 > step2000。方案B 用 step3000 是为了充分探索 FIFO 收敛，但最终 eval 应该也对比中间 ckpt（如果有保存的话）。

4. **BABILong 0k 分数高不代表 OK**：0k 只需要 in-context 推理，不需要 memory，是 baseline 能力验证。关注 8k/16k/32k 才是长程真实能力。

5. **H20 unfreeze_from 必须 ≥ 24**：16 会在 step30 OOM（optimizer states 太大），已有三个臂踩过这个坑。

6. **MemoryLLM 资产只在 diskA（本机）**：`external/memoryllm_venv` 和模型权重在本机，B200/.245.174/.7.53 没有，方案C蒸馏只能在本机跑。
