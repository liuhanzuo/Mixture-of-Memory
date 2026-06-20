# mem_space 多层读出修复实验 — 设计草案

**作者**: landmark-repro  **日期**: 2026-06-20  **状态**: 待 team-lead 审 + Level 1 结果合并
**依据**: S5(单层读出 step1000/2000 长档 0% vs S4b 全层 78-100%)+ methodA-eval go/no-go(reader 选对 chunk 55% 但 W0 答不出)+ Level 0(单块 readout 96-100%)三独立证据，根因 = 跨块信息只在少数层被消费。

---

## 0. 关键前提（先看，影响臂设计）

**⚠️ mem_space 当前 Method A 已经是"多层读出 16,20,24 + unfreeze L16-31"了**（`scripts/launch_rawkv_methodA_h1fix_b200.sh:52-53`），但仍崩。所以"多层"杠杆在 raw-KV 侧**已部分开却没救回**。这意味着修复不是简单"单层→多层"，而要拆清楚两个混在一起的变量：
- **(A) 层覆盖广度**：3 层(16/20/24) vs Landmark 的全 32 层 / 上半 16-31。3 层可能仍欠。
- **(B) 训练目标**：`--last_chunk_loss_only`（只最后一块算 loss，reader 几乎没练"从跨块 KV 生成 token"）vs Landmark 的**全序列 LM loss + 每层每步都在读出**。

**这是本实验要解开的核心混淆**：到底是"层不够"(A) 还是"loss 没教读出"(B)，还是两者都要。

**⚠️ loss 模式的坑**：关 `--last_chunk_loss_only` → 回到 tbptt（`dolmino_train_step_tbptt`），但那条路是"每块用局部 causal attn 预测自己的 next-token → memory 几乎用不上"（train 脚本 :1067-1068 原注）。**所以"全序列 loss"不能简单关 last_chunk_loss_only**——那会让 memory 通道失压，不等于 Landmark 的"全序列 loss + 强制跨块读出"。需要一个新的 loss 模式（见 §3 改动）。

---

## 1. 现有代码支持程度（盘点结论：基础设施基本就绪，差一个 loss 模式）

| 能力 | 现状 | 改动量 |
|---|---|---|
| 多层 readout 注入 | ✅ 已支持 `--rawkv_readout_layers 16,20,24`，每层装自己的 in-attn wrapper、用**自己的 k/v_proj + RoPE** 重投影检索到的 hidden（`layer.py:878-894`，config.py:558-568/611-624）= 已是"每层独立 query 读出"，非复制单层向量 | 0（只改配置） |
| 任意层集合（全/上半/隔层） | ✅ 列表任意指定 | 0 |
| 多层解冻 | ✅ `--unfreeze_backbone --unfreeze_layers_from N`（train 脚本 :527-542 partial unfreeze v2） | 0 |
| 全序列+强制跨块 loss | ❌ 只有两档：`last_chunk_loss_only`(只末块) 或 tbptt(每块局部、memory 失压) | **需新增**（见 §3） |
| 写 owner 单层 / 读多层 | ✅ store write 由 min(readout_layers) 拥有，读由全列表（layer.py:881-883） | 0 |

**结论**：层覆盖 + 解冻是纯配置，0 代码改动。唯一要写代码的是**"全序列 loss 但强制走 memory"的训练目标**（让每个 target chunk 都对前文 memory 算 loss，而非只末块）。

---

## 2. 对照臂设计（单轴拆 A/B）

锚点固定：chunk512、num_slots128、raw-KV readout、T2+pg19、lr2e-5/wd0.1、2000 步 save500、judge=W0 qa1 passkey 0k/4k/8k/16k/32k（与 methodA-eval go/no-go 同口径）。

| 臂 | readout 层 | unfreeze | loss 模式 | 测什么 |
|---|---|---|---|---|
| **A0 (baseline)** | 单层 16 | L16-31 | last_chunk_only | 复现单层 = S5 0% 配置的 mem_space 对照 |
| **A1** | 16,20,24（现状） | L16-31 | last_chunk_only | 复现现 Method A（已知崩，作 anchor） |
| **A2** | 全上半 16-31（16层） | L16-31 | last_chunk_only | 纯测层覆盖广度(A)：3层→16层能否救 |
| **B1** | 16,20,24 | L16-31 | **全序列跨块 loss** | 纯测训练目标(B)：同层数下 loss 改了能否救 |
| **B2** | 全上半 16-31 | L16-31 | **全序列跨块 loss** | A+B 都开 = 最接近 Landmark，上限 |

- A0/A1/A2 是**层覆盖单轴**（held loss=last_chunk）；A1/B1 是**loss 单轴**（held 层=3）；B2 = 组合上限。
- 先跑 A2 + B1（两个单轴各一），若任一显著回升 → 锁定主杠杆；再跑 B2 确认组合。A0 作为"单层=0%"的 mem_space 侧 S5 复现（补 Landmark→mem_space 的桥）。

## 3. 需要的代码改动（仅 §2 的 B 臂）

新增 loss 模式 `--cross_chunk_loss`（暂名）：对**每个** chunk c（c≥1）都计算 loss，但 c 的预测只能通过 memory/raw-KV readout 看到 chunk<c（chunk c 自身走局部 causal，前文走 no_grad 流入 memory 再 detach，复用现有 last_chunk_loss_only 的"context no_grad → detach → target 算 loss"机制，只是把"只末块"改成"滑窗每块都当一次 target"）。
- 改动文件：`scripts/train_mem_space_dolmino_cpt.py`（dolmino_train_step 加多-target 循环；~30-50 行）。
- 风险：每块都算 loss → 计算量 ~×n_chunks，需相应降 total_steps 或 grad_accum 调整。
- 备选（更省）：保持 last_chunk，但**滑动 target chunk 位置**（每步随机选一个 chunk 当 target，前文进 memory）——同样让 reader 在各种 query-needle 距离上练读出，计算量不变。**推荐这个备选**（改动更小、不爆算力）。

## 4. 资源 + 判据
- Group-A 16 卡（本机+.196）或单节点 8 卡。raw-KV readout 是 Llama-3-8B，建议本机 8 卡 FSDP + grad-ckpt（同 Method A 配置）。
- 判据：W0 qa1 passkey 长档（8k/16k/32k）。**A2 或 B1 任一把 8k/16k 从 ≤7 拉到 ≥30** = 该杠杆有效；都不动 = 根因在更下游（reader 架构 / 注入分布 OOD）。
- 与 methodA-eval Level 1（distance×层数 + 建议加 loss on/off 臂）对齐：避免重跑，ta 测 distance 维度，我测层覆盖×loss 维度。

## 5. 待 team-lead 决策点
1. 先跑哪两个单轴臂（建议 A2 + B1）。
2. loss 改动选"每块都算"(贵) 还是"滑动 target"(省，推荐)。
3. 等 Level 1 再起，还是 A2(纯配置0改动)可以先起。
