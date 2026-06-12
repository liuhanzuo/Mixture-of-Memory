# H-series v2 执行计划书

**创建时间**: 2026-05-11 (最后更新)  
**目标**: 在 BABILong 上获得非零分数，验证 cross-attention memory 架构  
**维护方式**: 该文件现视为常态化 / 持续维护的 plan 文件；heartbeat/main 在处理完状态变化后可以直接更新

---

## 背景

- H-series v1 (Llama-3-8B base + NIAH 训练) 在 BABILong 上 0%
- Root cause (researcher 确认): base model 不会 follow instructions + 训练任务 (NIAH) 和 eval (BABILong) 分布完全不同
- Solution: 跟随 ARMT 两阶段配方 (PG19 LM pretrain → BABILong 任务微调)，换 Llama-3.2-1B (和 ARMT 论文一致)

---

## 训练矩阵

| 变体 | 架构 | Base 策略 | Memory | 节点 | Phase 1 log |
|---|---|---|---|---|---|
| **A** | cross-attn slots (write L8, read L10/12/14) | Frozen base | 64 slots | b200-1 (本地) | `logs/h_v2_phase1_A_b2001.log` |
| **B** | Joint attention (early H-series) | **Full FT** | 128 slots (scaled up) | b200-4 | `logs/h_v2_phase1_B_b2004.log` |
| **C** | ARMT associative memory (reference) | Frozen + memory tokens | 32 tokens, d_mem=64 | b200-2（HMT 完成后已切入） | `logs/h_v2_phase1_C_b2002.log` |
| **D** | cross-attn slots + LoRA | Frozen base + LoRA(r=8, α=32) | 64 slots | b200-3 (等 MemoryLLM 完) | `logs/h_v2_phase1_D_b2003.log` |

**Baselines 对照** (已跑/在跑)：
- ARMT full training: b200-5/6/7/8（已完成，train_loss≈2.70, eval_loss≈2.78）
- RMT full training: 被 SIGHUP 杀掉，**待重启**（优先考虑 replacement B200 空闲节点）
- HMT full training: b200-2（已完成；final test PPL≈11.405，节点已切给 H-v2 C Phase 1）
- MemoryLLM eval: ✅ 完成（b200-3 已释放并转给 H-v2 D）
- Beacon eval: ✅ 完成 (BABILong avg 60.9%)

---

## 2026-05-12 17:08 资源调度决议

### 当前资源图
- stable b200-1..4：全部占用（A/B/D + 新切入的 C Phase 1）
- replacement b200-5..8：全部空闲（32× L20A 可立即使用）
- h20-1..4：全部可达且空闲，但 MemoryLLM 仍需先在 h20-1 完成 verify 修复

### 本轮 heartbeat 强制执行计划
1. **立即在 b200-5 启动 H-v2 C Phase 1**，不再等待 b200-2 上 HMT 完成
2. **b200-6 已用于 RMT 重启**；保留 b200-7..8 作为 follow-up 训练池，优先给 C 的后续阶段、成熟 baseline 扩展、或 researcher/coder 产出的高置信新训练任务
3. **H20 侧先修后跑**：一旦 h20-1 上 MemoryLLM verify 通过，立刻在 h20-1 拉起 `1k/2k/4k/8k/16k/32k` full wave；h20-2..4 保留给后续成熟 eval/repro 扩展
4. **heartbeat 后续检查顺序**：优先确认 `b200-5` 的 C Phase 1 是否稳定输出 step/loss；其次确认 h20-1 verify 是否通过并触发 full wave

## 2026-05-12 18:09 H20-only pivot

- 用户最新澄清：**暂时不在的是 replacement B200 `b200-5..8`，稳定 B200 `b200-1..4` 仍在**。我上一轮把范围误写成了全部 B200 offline。
- 原 heartbeat cron 已暂停，避免继续按错误前提自动执行；若恢复 heartbeat，应恢复对 `b200-1..4` 的正常认知，并只把 `b200-5..8` 视为暂不可用。
- h20-1 上的 MemoryLLM verify 已补齐三处兼容性修复：
  1. `scripts/eval_baseline_babilong.py` 兼容 chat template 返回 `BatchEncoding`
  2. `scripts/eval_baseline_babilong.py` 为 MemoryLLM 补 `generation_config`
  3. `scripts/launch_h20_memoryllm.sh` 默认改用 `${WORKDIR}/.venv/bin/python`
- 之后已在 h20-1 触发 `1k/2k/4k/8k/16k/32k` 六路 MemoryLLM wave（`2026-05-12 18:08 CST`，`scripts/launch_h20_memoryllm.sh`）；当前这些任务已快速结束，日志显示结果文件已存在而 skip。
- 当前主执行面仍偏向 H20 结果核查，但这不等于 `b200-1..4` 不存在；它们仍可在后续 heartbeat/调度中纳入考虑。

## 2026-05-12 H20 剩余 eval 执行计划

### 当前已确认状态
- `Llama-3.2-1B-Instruct`：BABILong 6/6 length 已完成，可直接作为 vanilla reference。
- `Beacon-Qwen2-7B`：BABILong 6/6 length 已完成，当前是最强已发布 baseline。
- `MemoryLLM-8B-chat`：h20-1 结果目录已齐全覆盖 `qa1..qa10 × {1k,2k,4k,8k,16k,32k}` 共 60 个 csv；heartbeat 已重新刷新本地 `status/babilong_baselines_h20.json` / `status/babilong_results.json`，当前完整聚合后 AVG 约维持在 **31-37%**。
- `M+`：`scripts/eval_baseline_babilong.py` 已支持 `--baseline mplus`；h20-1 的 six-wave `1k/2k/4k/8k/16k/32k` 全量 eval 早已完成，`MPlus-8B` 结果目录也已齐全覆盖 **60/60 csv**，但那批 full-wave 分数仍应视为被 runtime bug 污染的无效 0%。后续修复已先移除 runtime crash（device mismatch / CUDA tensor→numpy），本轮 heartbeat 进一步确认 **H20 线上 evaluator 曾经实际跑的是 stale copy**，现已同步为本地带 RoPE `inv_freq` 重建补丁的版本。同步后，`MPlus-8B-smoke-fix4-20260513` 的 `qa1_1k` 单样本输出已不再是 `!!!!!!!!!!!!!!!!!!!!`，而是重复的 `Incomplete...` / `INT...` 片段（`correct=0`）。因此当前 blocker 已从 **punctuation-only collapse** 收敛为 **post-sync semantic corruption / generate-path mismatch**；full wave 仍不应直接重跑。基于本轮人工代码审计，最可疑的剩余差异是 `scripts/eval_baseline_babilong.py` 里 MPlus `_generate()` 仍手工构造 attention mask 并强制 `use_cache=False`，与 `MemoryLLM-source/README.md` 的原始 `model.generate(input_ids=..., max_new_tokens=20)` 用法不一致。
- `ARMT / RMT`：repo 内已有官方 BABILong finetune/eval 入口（`third_party/associative-recurrent-memory-transformer/run_finetuning_babilong_rmt.py` + `third_party/associative-recurrent-memory-transformer/scripts/babilong/*`），但前提是对应 checkpoint 已可读且 layout 对得上。本轮 heartbeat 已通过 SSH+rsync 将 `legacy/`、`babilong/`、`data/armt_pg19_real_tokenized_full/` 推到 H20 shared mount；其中 RMT 还额外补了 H20-side shim `src/memory/rmt -> ../../legacy/memory/rmt`，并在 `h20-2/3/4` 拉起 `rmt_eval_ppl_h20_2`、`rmt_eval_nih_h20_3`、`rmt_eval_mem_h20_4`。这 3 个 retry3 当前都已结束，且一致死在 `legacy/scripts/eval_rmt.py` 的 **legacy/v10 loader mismatch**：evaluator 期待 `memory_embeddings/extractor.*`，实际 checkpoint `rmt_memory.pt` 是 v10 key（`l0.memory`, `recon_head.proj.*`）。H20 上直接运行 `legacy/scripts/debug_eval_rmt_v10.py --skip_model` 已确认该 checkpoint 的 memory keys 本身是 **valid v10**。本轮进一步人工审计表明：repo 内现成的 **v10-aware loader** 在 `legacy/scripts/eval_needle_haystack_v10.py` / `legacy/scripts/eval_needle_haystack_extended.py`，但尚未确认有现成的 **BABILong v10 eval 入口**；因此下一步应基于这条 v10 loader 路径做 BABILong wrapper/adaptation，而不是继续重试 `legacy/scripts/eval_rmt.py`。ARMT checkpoint 本身仍未在 H20 shared mount 上确认 ready，因此本轮仍未直接启动 ARMT BABILong eval。
- `HMT`：当前 repo 未发现直接的 BABILong eval 入口；survey 也标记为 `BABILong: NOT reported`，因此应视为“需要适配”，不是 H20 上立刻可跑的现成 eval。
- replacement B200 当前节点映射：`b200-5=28.89.18.252`, `b200-6=28.89.20.82`, `b200-7=28.89.20.27`, `b200-8=28.89.18.19`；密码文件仍为 `configs/password_b200_ephemeral.txt`。当前更应以较新的 heartbeat 探测为准：`b200-5` / `b200-7` / `b200-8` 现返回 `Permission denied`，`b200-6` 现为 `Connection refused`，因此继续视为“unavailable / not ready for Mixture-of-Memory auto-launch”。历史上该池曾暴露 `share_304376610` 路径与 ARMT `past_key_values` runtime bug，但在认证与连通性重新稳定前，不应继续把它们当作可直接接管 C/RMT 的稳定训练池。

### 执行优先级（给 H20）
1. **P0 — 处理 `M+` 的生成退化问题**
   - 聚合已经完成；`status/babilong_results.json` / `status/babilong_baselines_h20.json` 已纳入 `MPlus-8B`，但当前分数无效。
   - 下一步不是重跑同样的 six-wave，而是先修复 `scripts/eval_baseline_babilong.py` / `MemoryLLM-source/modeling_mplus.py` 路径中的生成语义失真（当前 smoke-fix4 输出为重复 `Incomplete...` / `INT...`，不再是 `!!!!!!!!!!!!!!!!!!!!`）。
   - 优先验证点：让 MPlus `_generate()` 更贴近上游 README 用法（避免额外手工 attention mask，并优先恢复默认 cache generation 语义），再做 `qa1/1k` smoke。
2. **P1 — 跑 ARMT 的 BABILong eval**
   - 前提：ARMT checkpoint 在共享盘上已准备好。
   - 先做 `qa1` smoke / validate-only，确认 checkpoint + tokenizer + data path 全部打通；再扩成全任务全长度。
4. **P3 — 跑 RMT 的 BABILong eval**
   - 前提：RMT restart 产出可评估 checkpoint。
   - 执行方式同 ARMT：先 smoke，后 full sweep。
5. **P4 — HMT 适配**
   - 暂不作为 H20 本轮立即执行项；需要先补一个 BABILong adapter/eval path，再谈 full run。

### 建议实际 run order
1. **先处理 `M+` 的生成退化 bug**（结果已聚合，不再重复同样 six-wave；优先修复 `scripts/eval_baseline_babilong.py` / `MemoryLLM-source/modeling_mplus.py` 路径中当前 `Incomplete...` / `INT...` 重复输出的问题）
2. **优先修复 replacement B200 的 runtime blocker**（路径已切到 `share_304376610`，但本轮 heartbeat 对 b200-5/7/8 现有密码验证失败、b200-6 连接被拒；继续视作 unavailable / not ready）
3. **ARMT qa1 smoke → ARMT full eval**（仅在 checkpoint 路径 + replacement 节点 runtime 都确认可用后执行）
4. **RMT：先补 v10-aware BABILong 入口，再 qa1 smoke → full eval**（当前只确认 NIH/extended evaluator 内已有 v10 loader，不能继续直接复用 `legacy/scripts/eval_rmt.py`）
5. **HMT 暂缓**，先单列 TODO

### 每个方法的完成标准
- 结果目录下应覆盖 `qa1..qa10 × {1k,2k,4k,8k,16k,32k}` 共 **60 个 csv**。
- 每个 csv 应有 **100 行样本**。
- 跑完后必须立刻重跑聚合脚本，更新汇总 JSON；不要再出现 raw 已齐但 summary 仍 partial 的情况。

### 具体落地建议
- **MemoryLLM 聚合刷新**：优先使用 `scripts/aggregate_babilong_baselines.py`；若只想快速生成 baseline summary，也可用 `scripts/score_babilong_baselines.py`。
- **M+ 执行入口**：直接走 `scripts/eval_baseline_babilong.py --baseline mplus`；参数模板尽量照 `scripts/launch_h20_memoryllm.sh` 复用，避免再踩环境/代理/venv 差异。
- **ARMT/RMT 执行入口**：优先沿用 `third_party/associative-recurrent-memory-transformer/run_finetuning_babilong_rmt.py` 和其 `scripts/babilong/*.sh` 逻辑，不要先自行重写 evaluator。
- **HMT**：先补 eval 适配，再安排 H20；本轮只保留为 TODO。

### 本轮明确 TODO
- [x] 刷新 `MemoryLLM-8B-chat` 的聚合 JSON（本地 `status/babilong_baselines_h20.json` 已含 `MemoryLLM-8B-chat`）
- [x] 完成 `M+` BABILong 全量 eval（heartbeat 已完成 60/60 csv 聚合，但结果暴露 runtime bug：当前有效分数为全长度 0%）
- [ ] 修复 replacement B200 的 Mixture-of-Memory runtime blocker（路径已切到 `share_304376610`，`.venv + fla` 已通，但 ARMT 仍在 `language_modeling.py` 的 `past_key_values` 处崩溃）
- [ ] 若 checkpoint ready 且 replacement 环境可用，完成 `ARMT` BABILong eval
- [ ] 若 checkpoint ready，完成 `RMT` BABILong eval
- [ ] 为 `HMT` 补一个可复用的 BABILong eval path
- [ ] 暂不做 `Infini-attention`
- [ ] `MemLong` 延后（依赖更多环境/FAISS，且优先级低于 M+/ARMT/RMT）

---

## Phase 1: PG19 LM Pretrain

**配方**（照抄 ARMT）：
- Dataset: `data/armt_pg19_real_tokenized_full` (已预处理)
- Steps: 50,000
- LR: 1e-5 (memory-only 变体), 2e-5 (full FT 变体 B)
- Scheduler: linear, warmup 5000
- Optimizer: AdamW, wd=0.01, grad clip 1.0
- bf16, 8 GPU, effective batch 64 (per-device 1 × grad_accum 8 × 8 GPU)
- segment_size=512, max_n_segments=2
- `--no_loss_from_first_segment` (只在第2个 segment 上算 loss)

**预计时间**：
- A/D (frozen + memory only): **~3 小时**
- B (full FT): **~4 小时**
- C (ARMT): **~3 小时**

**Phase 1 成功标志**: eval PPL 收敛，train loss < 3.0

---

## Phase 2: BABILong 任务微调

**配方**（照抄 ARMT）：
- Task: qa1 (单任务起步, single-supporting-fact)
- Dataset: BABILong qa1 train split + PG19 noise injection
- Steps: 30,000 (curriculum 2→4→8→16→32 segments, 各 6000 steps)
- LR: 1e-4, linear, warmup 1000
- Loss: **仅在答案 token 上计算** (labels_mask 其他全 -100)
- 使用 chat template 包装 prompt (即使 base model，保持与 eval 格式一致)

**预计时间**：
- 单变体: **10-30 小时** (取决于 curriculum)

**Phase 2 成功标志**: BABILong qa1 eval accuracy > 20% (非零就是里程碑)

### 2026-05-12 当前高置信 follow-up
- **H-v2 D 下一条最值得开的新臂**：把 Phase 2 从 `qa1-only` 改为 `qa1+qa2+qa3` multi-task mix，并保留少量 PG19 noise + instruction-style format wrapper。该建议来自本轮 researcher 高置信结论，优先作为 D 的 follow-up ablation。
- **Baseline 队列**：`M+` 六路跑完后，先聚合 `MPlus-8B`，再推进 `ARMT` BABILong eval；但 ARMT 之前必须先修复 replacement B200 的路径/环境问题。
- **2026-05-13 审计→修复闭环**：上述两点现已落地到代码：`scripts/train_h_v2_babilong.py` 已改为使用 BABILong prompt template 做 answer-only supervision，`scripts/train_h_v2.py` 已去掉 segment 边界的 slot-state `.detach()`；本地 compile 与 dataset sanity 已通过。启动 D 的 multi-task follow-up 时应直接沿用这条已修补路径。

---

## 进度跟踪

### 2026-05-11 里程碑

- [x] PG19 tokenized 数据准备完成 (25552 books, 21GB)
- [x] 4 变体训练脚本准备完成 + smoke 通过 (A/B 跑通, C smoke 需加 `--sample_size 1024` 的 fix)
- [x] ARMT full 训练启动 (b200-5/6/7/8, 4 节点并行)
- [x] Beacon BABILong 完整 eval 完成 (60.9% avg)
- [x] H-v2 A Phase 1 启动 (b200-1)
- [x] H-v2 B Phase 1 启动 (b200-4)
- [x] H-v2 A Phase 1 完成 → Phase 2 启动（2026-05-12 13:31 已在 b200-1 启动 `v2_phase2_A_b2001`）
- [x] H-v2 B Phase 1 完成 → Phase 2 启动（2026-05-12 13:31 已在 b200-4 启动 `v2_phase2_B_b2004`）
- [x] MemoryLLM eval 完成 → b200-3 释放 → H-v2 D Phase 1 启动
- [x] H-v2 D Phase 1 完成 → Phase 2 启动（2026-05-12 13:31 已在 b200-3 启动 `v2_phase2_D_b2003`）
- [x] H-v2 C Phase 1 已在 b200-5 启动（2026-05-12 17:12，tmux `v2_phase1_C_b2005`）；b200-2 上 HMT 继续独立跑完
- [x] ARMT full PG19 在 b200-5/6/7/8 完成（train_loss≈2.70, eval_loss≈2.78）
- [x] RMT 重启已在 b200-6 拉起（2026-05-12 17:12，tmux `rmt_v10_l0l1_b2006`）

### Phase 2 里程碑

- [ ] H-v2 A Phase 2 (BABILong qa1) 启动 + 完成（已于 2026-05-12 13:31 启动，待完成）
- [ ] H-v2 B Phase 2 启动 + 完成（已于 2026-05-12 13:31 启动，待完成）
- [ ] H-v2 C Phase 2 启动 + 完成
- [ ] H-v2 D Phase 2 启动 + 完成（已于 2026-05-12 13:31 启动，待完成）
- [ ] 4 变体 BABILong eval → 对比表

---

## Heartbeat 自主执行规则

### Heartbeat 每次检查必须做的事

1. **读取本文件 + CODEBUDDY.md + HEARTBEAT.md**
2. **检查所有运行中 tmux session**:
   - b200-1: `tmux ls` (本地) 找 `h_v2_A`
   - b200-2: SSH 到 28.89.17.144，检查 `hmt_full` 和 `h_v2_C` (如已启动)
   - b200-3: SSH 到 28.89.17.85，检查 MemoryLLM eval 进程 和 `h_v2_D` (如已启动)
   - b200-4: SSH 到 28.89.19.134，检查 `h_v2_B`
   - b200-5/6/7/8 (28.89.18.252, 28.89.20.82, 28.89.20.27, 28.89.18.19): 检查 replacement B200 节点可用性 / 空闲状态
3. **对每个 RUNNING 任务**: 读 log 末尾，记录 step/loss，判断健康性:
   - loss > 100 或 NaN → kill (符合 CODEBUDDY.md 规则)
   - log > 30 min 无更新 → stalled，kill
   - 正常 → 继续
4. **对每个已 COMPLETED 任务**: 按计划书推进下一步
   - A Phase 1 完成 → 立即启动 A Phase 2 (不需要审批，同 pipeline)
   - MemoryLLM eval 完成 → 立即启动 **D Phase 1** 在 b200-3
   - HMT 完成 → 立即启动 **C Phase 1** 在 b200-2 (smoke 脚本已有，Phase 1 需要加 `--sample_size 1024`)
   - RMT 需要重启 → 等 A 在 b200-1 完成后，或找其他空闲节点

### 触发新任务的条件

**自动启动 (auto_launch=true，无需审批)**:
- 任何 Phase 1 完成 → 立即启动对应 Phase 2 (同 pipeline 延伸)
- 任何 eval 完成 → 立即释放节点并启动下一个 pending 任务
- 节点空闲 且有 pending 任务 → 启动

**需要用户审批**:
- 新的实验方向 (非 ablation)
- 修改训练超参 (LR/memory size 等) (但 researcher 确认的除外)
- 发现架构重大问题需要重构

### 文件路径速查

**Phase 1 launch scripts**: `scripts/v2_phase1/train_v2_{A,B,C,D}_*.sh`  
**训练主脚本**: `scripts/train_h_v2.py` (A/B/D), `third_party/associative-recurrent-memory-transformer/run_finetuning_lm_rmt_hf.py` (C)  
**tokenized 数据**: `data/armt_pg19_real_tokenized_full/`  
**模型**: `/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B`  
**Log 目录**: `logs/h_v2_phase1_{A,B,C,D}_b200{1,2,3,4}.log`  
**Output 目录**: `outputs/h_v2_phase1_{A,B,C,D}_b200{1,2,3,4}/`  

### Remote 节点信息

| 节点 | IP | 密码文件 |
|---|---|---|
| b200-1 | (本地) | - |
| b200-2 | 28.89.17.144 | `configs/password.txt` |
| b200-3 | 28.89.17.85 | `configs/password.txt` |
| b200-4 | 28.89.19.134 | `configs/password.txt` |
| b200-5 | 28.89.18.252 | `configs/password_b200_ephemeral.txt` |
| b200-6 | 28.89.20.82 | `configs/password_b200_ephemeral.txt` |
| b200-7 | 28.89.20.27 | `configs/password_b200_ephemeral.txt` |
| b200-8 | 28.89.18.19 | `configs/password_b200_ephemeral.txt` |

### SIGHUP 规避规则

**所有长任务必须用 tmux** (一小时+的任务)：
```bash
# 本地
tmux new-session -d -s <name> "<full command> 2>&1 | tee <log>"

# 远程 (通过 SSH)
sshpass ... ssh root@<ip> '
tmux new-session -d -s <name> "
<full command with exported env vars>
2>&1 | tee <log>
"
'
```
不使用 `nohup` + background，这在 CodeBuddy Bash tool 下会被 SIGHUP 杀死。

---

## 现有运行任务详情（heartbeat 重点监控）

### b200-1: H-v2 A
- Phase 1 log: `logs/h_v2_phase1_A_b2001.log`
- Phase 1 final ckpt: `outputs/h_v2_phase1_A_b2001/checkpoint_final.pt`
- Phase 2 tmux: `v2_phase2_A_b2001_bs`
- Phase 2 log: `logs/h_v2_phase2_A_b2001_bs.log`
- 最新状态: **Phase 2 持续健康推进**；当前已进入 curriculum `segments=8`，本轮 heartbeat 本地日志尾部更新到 `step=580`、recent loss≈`0.3486`（`2026-05-13 16:28 CST`），8×GPU 持续占用

### b200-4: H-v2 B
- Phase 1 log: `logs/h_v2_phase1_B_b2004.log`
- Phase 1 final ckpt: `outputs/h_v2_phase1_B_b2004/checkpoint_final.pt`
- Phase 2 tmux: `v2_phase2_B_b2004_bs`
- Phase 2 log: `logs/h_v2_phase2_B_b2004_bs.log`
- 最新状态: **Phase 2 持续健康推进**；当前已进入 curriculum `segments=8`，本轮 heartbeat 远程日志尾部更新到 `step=560`、recent loss≈`0.0007`（`2026-05-13 16:17 CST`），8×GPU 持续占用

### b200-5/6/7/8: replacement B200 pool
- 当前映射（以较新的 heartbeat / PENDING_TASKS 为准）: `b200-5=28.89.18.252`, `b200-6=28.89.20.82`, `b200-7=28.89.20.27`, `b200-8=28.89.18.19`
- 当前状态: **本轮 heartbeat 不再视为可用训练池**。实测 `b200-5/6` 使用现有 `configs/password_b200_ephemeral.txt` 返回 `Permission denied`，`b200-7/8` 为 `Connection refused`；因此继续按“replacement B200 unavailable / not ready”处理
- GPU / 路径: 历史上为 8× `NVIDIA L20A 183359 MiB`，且问题根路径曾指向 `share_304376610`；但在认证与连通性重新稳定前，不应作为 auto-launch 目标
- 备注: 旧的 replacement IP 记录与路径描述在历史段落中存在不一致，本节以后续 heartbeat 的实际探测结果为准；旧 ARMT full PG19 完成记录仅保留为历史结果（train_loss≈2.699-2.710, eval_loss≈2.778-2.785）

### b200-2: HMT → H-v2 C 切换位
- HMT 历史: 原始 run 先在 ~11000 step 因 validation `StopIteration` 崩溃；之后的 `resume10000` run 已完成到 35000+ checkpoint，但在训练结束后的 final eval 再次因 `valid_gen` 耗尽触发 `StopIteration`
- HMT 修复结果: heartbeat 已修复 `third_party/HMT-pytorch/tools/training/train_redpajama.py` 中 final eval / test 的 dataloader 迭代逻辑（`StopIteration` 时重建 iterator），并从 `outputs/hmt_pg19_full_b2002_resume10000/model_weights_35000.pth` 续跑完成；最终 test tail 给出 `PPL on 100 test samples: 11.405261888504029`
- H-v2 C 启动经过:
  - `2026-05-13 13:11 CST` heartbeat 在 b200-2 启动 **H-v2 C Phase 1**，tmux=`v2_phase1_C_b2002`，log=`logs/h_v2_phase1_C_b2002.log`，port=`29613`
  - 首次启动立刻因项目 `.venv` 缺少 `fla` 而失败（`ModuleNotFoundError: No module named 'fla'`）
  - heartbeat 随后现场验证 `/opt/conda/envs/torch-base/bin/python` 可导入 `torch/transformers/accelerate/fla` 与 `modeling_amt.online_armt`，并于 `2026-05-13 13:31 CST` 用 `PYTHON_BIN=/opt/conda/envs/torch-base/bin/python` 重新拉起同一脚本
- 最新异常结论:
  - relaunch 后训练确实推进到 `step≈310`，train loss 仍有限（约 `2.784`）
  - 但 eval path 从极早期开始持续输出 `eval_loss=nan`，并一直重复到最近观测窗口（约 `epoch 0.02818`），不属于“一次性抖动”
  - 因此 heartbeat 已在本轮 **kill `v2_phase1_C_b2002` 并释放 b200-2**，避免继续浪费稳定 B200 资源
- 当前状态: **b200-2 现已空闲**。下一步不是直接重启 C，而是先检查 `third_party/associative-recurrent-memory-transformer/run_finetuning_lm_rmt_hf.py` 的 eval path / labels / metric 逻辑，再决定是否重启或迁移

### b200-3: H-v2 D
- Phase 1 log: `logs/h_v2_phase1_D_b2003.log`
- Phase 1 final ckpt: `outputs/h_v2_phase1_D_b2003/checkpoint_final.pt`
- Phase 2 tmux: `v2_phase2_D_b2003`
- Phase 2 log: `logs/h_v2_phase2_D_b2003.log`
- 状态: **Phase 2 已启动**；当前处于 curriculum `segments=2` 启动阶段，8×GPU 已重新占用

---

## Researcher 报告

完整背景报告在：`ops/research_notes/2026-05-11_*`
- `memory_papers_reproducibility_survey.md`
- BABILong 0% 根因分析 (inline in conversation log)
- H-v2 训练配方 (本文件)
