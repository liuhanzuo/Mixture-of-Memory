# Plan — slot_dim=16384 宽 slot + 替代 dual_gate（2026-06-04 夜）

## 目标

把 mem_space L1 slot 维度从 4096（=backbone hidden）扩到 **16384**，每层 **128 slot**，
并用成本可控的写入门控**替代 LM2 式全连接 dual_gate**（slot_dim=16384 时该门控参数爆到 ~34B，必 OOM）。

核心待回答问题：
1. **宽 slot（16384）下，可学习的逐特征门控是否还胜过固定标量 β？** dual_gate 的全连接形式太贵且当前根本没学（见下方 bug），换成低秩/对角后值不值得保留。
2. slot_dim=16384 + 128 slot 能否在 H20（97.8 GiB）上跑起来（带 gradient_checkpointing + gas=4）。

## 已确认的关键事实

- `train_mem_space_dolmino_cpt.py` 此前**没有 `--slot_dim` CLI**，slot_dim 永远 = d_model=4096 → 本次新增。
- **Bug（本次顺手修）**：`_mem_space_params` 漏收 dual_gate 的 `gate_proj_new/gate_proj_mem/gate_bias`，
  导致 `--use_dual_gate` 的门控投影从未进 optimizer，被冻在 xavier 初始化。这解释了为什么之前
  P2 日志只见标量 `gate_val(beta)=0.5`、门控始终"没在学"。
- 架构本身已支持 slot_dim≠d_model（selector K_sel、slot_to_hidden/hidden_to_slot 投影都已处理）。

## 三臂单变量对照（其余参数完全一致）

公共配置：slot_dim=16384, num_slots=128, top_k=16, chunk_size=1024, selector_temperature=40,
gradient_checkpointing, gas=4, routing_pool_mode=slot_query, use_decoupled_read, use_l3_summary,
inject_gate_bias_init=-2.0, no_detach_slots_in_selector, no_slot_delta_clip。
total_steps=500（短跑验证 routing + 门控诊断），save_interval=250。

| 臂 | writeback_mode | 门控形式 | 可训练门控成本/层 | 节点 |
|----|----------------|----------|------------------|------|
| **A** | `lowrank_gate` (r=256) | `U(V_new·s + V_mem·M) + b`，logits=2·slot_dim | ~4·d·r | 本机 H20-1 |
| **B** | `diag_gate` | 逐特征 `a·s + c·M + b`（对角向量） | ~6·d | 远程 H20-2 (28.59.80.196) |
| **C** | `scalar_beta` | 复用 legacy 单门 EMA（标量 β） | ~0 | 待定（盘B 或 H20-2 排队） |

> A vs C 回答"可学门控 > 固定 β？"；B 是 A 的极简对照（对角 = rank-0 逐特征），
> 三者一起看清"门控复杂度—收益"曲线。

## 节点 / 环境

- **本机 H20-1**：`.venv/bin/python`，跑 A。注意当前 H20-1 有 P2 decoupled run（step ~570/2000）占着 8 卡 —
  需先决定：等它完成 / kill 它腾位。P2 的 routing 也还在挣扎（top1_sim 0.04-0.09 抖动），
  且 dual_gate bug 意味着它的门控本来没在学 → **倾向 kill P2 腾本机给 A**（待用户确认或按"显著 bug"红线判断）。
- **远程 H20-2 (28.59.80.196)**：与本机共享 FS（代码免同步），torch-base env 正常，跑 B。
- **盘 B (28.49.196.161 / 29.162.241.149)**：torch-base env **缺 transformers** → 用户已确认
  **可直接 copy 本机可用 env 过去**。若要用盘B 跑 C，先 copy env + rsync 代码。否则 C 在 H20-2 排 A/B 之后。

## 执行步骤

1. [进行中] coder 实现 config/layer/trainer 改动 + 修 bug + 3 个 launch 脚本 + commit。
2. [待] 验证 coder diff：确认 `_mem_space_params` 收齐新门控参、slot_dim 传入正确、向后兼容 dual_gate 不变。
3. [待] 本机起 A（先解决 P2 占位）。直接多卡跑，不预 smoke（项目规则）。盯：
   - **是否 OOM**（16384 slot 的实测显存，单卡目标 < 90 GiB）。OOM → 降 gas 或上更激进 checkpointing。
   - **routing**：top1_sim 是否 > 1/128 噪声地板并稳定；uniq_sel_slots 是否 >> 1。
   - **门控诊断**：g_in_mean / g_forget_mean 是否随 step 变化（证明门控在学，区别于 bug 期的冻结）。
   - lm loss 是否健康（PPL 别 > 100，否则按红线先排查不调参）。
4. [待] 远程 H20-2 起 B（同样盯点）。
5. [待] C 排在 H20-2（A/B 之后）或 copy env 后落盘 B。
6. [待] 500 step 后三臂汇总：派 researcher 比较 routing 质量 + 门控行为 + lm loss，
   决定哪个 writeback_mode 进入长跑（2000+ step）。结果写入 BENCHMARK_RESULTS.md + 新建 versions/vN_widewb.md。

## 风险 / 注意

- **NCCL/CEPH hang**：本机更稳，重要臂（A）放本机；远程偶发 30min watchdog 崩，崩了自动重启。
- **eval 内联会 desync 崩** → 全程 `--eval_interval 0`，eval 离线单独跑 checkpoint。
- **16384 显存**：这是本实验最大不确定性。若 A 在本机 OOM，先降 slot_dim 到 8192 验证代码正确性再回 16384，
  或减 num_slots。实测显存第一时间记到本文件。
- slot 内存：128 slot × 16384 dim × bf16 × 32 层（shared_bank 则 1 份）≈ 单份 4M 参数 *2B = 8MB/份，slot 本身不大；
  真正吃显存的是 16384 维下的 hidden_to_slot/slot_to_hidden 投影激活 + checkpointing 重算。

## 落账

- 启动：TRAINER_ACTIVE.md（Write 覆盖）+ ACTIVE_SWEEPS.jsonl + gpu_runs.jsonl（带 commit_hash, slot_dim, writeback_mode）
- 完成：gpu_runs.jsonl + RESEARCHER_REPORTS.jsonl + UPDATELOG.md
