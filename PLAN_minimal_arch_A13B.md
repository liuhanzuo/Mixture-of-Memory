# PLAN — Hunyuan-A13B 极简架构（front-j + fresh-NTP continue-pretrain）

> 合作者方案的落地计划，**以 Qwen3-8B 和 Hy3-80L 的既有做法为参照细化**。
> 相关：核心结论/日志 `HARU.md`；环境 `status/HY3_ENV_SETUP_lhz.md`；训练器 `scripts/train_hunyuan_a13b_probe2.py`。
> 最后更新：2026-07-14。

---

## 0. 一句话
**"理解在前段、生成在顶层" 分工假设 → 极简架构验证**：从预训练大模型取前 j 层（理解已饱和），丢掉顶部冗余层，接 k 层全新 transformer 专做 next-token 生成，把 (j+k) 层小模型在 SlimPajama 上 continue-pretrain，看能否逼近原完整模型。

---

## 1. 合作者原始 plan（4 步）
1. **确定语义截止层 j**（每个 backbone 不同）。
2. 从预训练模型**取出前 j 层**（带预训练权重）。
3. 在前 j 层后面**新加 k 层（2 或 4）transformer**（随机初始化，专学 NTP）。
4. 把这 (j+k) 层模型在 **continue-pretrain 数据集（SlimPajama）** 上继续训练。

---

## 2. 三个 backbone 的 j（0.4·L 规律，跨 backbone 成立）

| backbone | 总层 L | **j** | j/L | 定 j 的方法 | 状态 |
|---|---|---|---|---|---|
| Qwen3-8B | 36 | **12** | 0.33 | 分工 probe / 经验 | armB 已训到 40k（→200k） |
| Hy3 | 80 | **32** | 0.40 | QCMem depth-partition j-sweep（fidelity smile mid-depth min） | j 已定 |
| **Hunyuan-A13B**（本计划） | **32** | **13** | **0.41** | **QCMem j-sweep**（16k fidelity smile，见 HARU.md） | j 已定，待训 |

⚠️ **定 j 的方法教训（A13B 专属坑）**：Hunyuan 中间层有 massive activation（hidden absmax ±137~165），**不能用 logit-lens / 硬截断**判 j（会爆炸 nll 84）；**必须用 QCMem depth-partition j-sweep**（h_j 喂 layers[j:] 正常重算）。详见 HARU.md。Qwen3/Hy3 无此坑但用同一 QCMem 方法最稳。

---

## 3. Qwen3 / Hy3 是怎么做的（参照，细化 A13B 的依据）

### 3.1 三个 arm（`train_{qwen3,hunyuan_a13b}_arch_probe2.py` 共用设计）
- **arm B（默认，主线）**：`front-j 继承权重 + k 层 fresh`，**训练所有层（不冻结）**，差分学习率（"healing"）。= 合作者 plan 的正解。
- **arm A（`--freeze_front`）**：冻结前 j 层，只训 fresh 层 + norm + head。消融对照——"前 j 层表征是否已足够、只差生成头"。
- **control 2（`--from_scratch`）**：忽略预训练权重、全随机初始化训 (j+k) 层。对照——"继承权重到底带来多少收益"。
- **control 0**：原完整模型（不训），eval ppl 作天花板基线。

### 3.2 关键训练配方（arm B）
- **构造**：`cfg.num_hidden_layers = j+k`；transplant 前 j 层 + embed + final norm（+ lm_head，Qwen3 是 fresh、A13B 因 tie_word_embeddings 是继承）；fresh 尾层随机初始化。
- **差分学习率**（核心）：
  - fresh 新层：`--lr 1e-4`（高，从头学）
  - inherited 前 j 层 + embed + norm：`--lr_inherited 2e-5`（低，保护预训练表征、微调不破坏）
  - cosine 衰减到 `min_lr 1e-5 / min_lr_inherited 2e-6`
- **不冻结**（合作者明确要求）：arm B 不加 `--freeze_front`，前 j 层是低 LR 微调、不是冻死。
- warmup 150、weight_decay 0.1、grad_clip 1.0、seq_len 2048、gradient_checkpointing 1。

### 3.3 batch / 步数 / 并行（Qwen3 armB 实际调度）
- Qwen3 armB：`batch_size 8 × grad_accum 2 × 8 GPU = eff_bs 128`（单节点）；20k→200k 续训时改 2 节点 16 卡 `bs4 × accum2 × 16 = eff_bs 128`（保持 eff_bs 不变）。
- Qwen3 分两段跑：先 20k（`launch_qwen3_arch_probe2.sh`，max_steps 默认 2000，实跑 20k）→ resume 到 200k（`run_qwen3_armB_200k_node.sh`）。**"先 20k 看形态，再拉长到 200k"** 是既定节奏。

---

## 4. Hunyuan-A13B 的落地配置（本计划）

### 4.1 架构
- L=32，**keep_front_layers=13**（j），**n_fresh_layers=2**（先 fresh2；后续并行 fresh4 ablation）→ **15 层小模型**。
- A13B 是 MoE（64 experts, top-8, +1 shared/层）；fresh 尾层用 `post_init` 正确初始化 MoE（experts N(0,0.02)），不手搭层。
- tie_word_embeddings=True：lm_head 权重 == embed，属"继承"（低 LR），与 Qwen3（lm_head fresh）不同。

### 4.2 arm 计划（对齐 Qwen3/Hy3）
| arm | 配置 | 目的 | 节点 |
|---|---|---|---|
| **arm B** keep13+fresh2 | 默认，不冻结，差分 LR | **主线**，合作者 plan 正解 | lhz（8×H200） |
| arm B keep13+**fresh4** | 同上，k=4 | fresh 层数 ablation | lhz2（8×H200） |
| （后续可选）arm A / from_scratch | 消融 | 隔离"前j足够"/"继承收益" | 视资源 |

### 4.3 超参（arm B，对齐 Qwen3 armB）
```
--keep_front_layers 13 --n_fresh_layers 2
（不加 --freeze_front）
--lr 1e-4 --min_lr 1e-5 --lr_inherited 2e-5 --min_lr_inherited 2e-6
--warmup_steps 100~150 --seq_len 2048 --gradient_checkpointing 1
--batch_size 1 --grad_accumulation_steps 8   （A13B 80B MoE，单卡放不下大 batch；FSDP FULL_SHARD）
--max_steps 20000（第一阶段，先看 s/step + 形态）→ 后续 resume 拉长
--save_every 1000 --log_every 10
--data_path data/slimpajama_chunks_2048_hunyuan.npy   （已生成 338514×2048）
```
- **eff_bs**：bs1 × accum8 × 8 GPU = 64（A13B 比 Qwen 大得多，eff_bs 比 Qwen 的 128 小是显存所限，可接受）。
- **A13B 特有坑**（务必）：加载模型必须 `experts_implementation="eager"`（否则 torch2.8-nv grouped_mm kernel 崩，见 HARU.md 坑2）；`cfg.head_dim=128`；FSDP FULL_SHARD + CPU offload 防首步 OOM。

### 4.4 节奏（对齐 Qwen3 的"先20k再200k"）
1. **阶段1**：keep13+fresh2 跑 20k，看 s/step、loss 曲线、是否健康。
2. **阶段2**：健康则 resume 拉长（Qwen 到 200k；A13B 视算力定，合作者 Qwen 计划 200k）。
3. 每阶段末 eval：(a) LM ppl 对比原 32 层完整模型（control 0 天花板）；(b) 可选下游。
4. fresh4 arm 并行跑，对比 k=2 vs k=4。

---

## 5. 评判标准（假设成立/推翻）
- **成立**：15 层（13+2）continue-train 后 LM ppl **逼近** 原 32 层完整模型 → 顶部 19 层大部分冗余、极简架构可行。
- **推翻**：怎么训都追不平 → 中间层做了不可省的"渐进精炼"，分工假设对"删层"不成立（只对"缓存"成立，即 QCMem）。
- 参照：Qwen3 armB 已训到 40k（→200k）是同类实验的进行中基线，A13B 可横向对比"大 MoE 上是否同样成立"。

---

## 6. 当前状态（2026-07-14）
- [x] j 确定：A13B split-j=13/32（QCMem j-sweep）
- [x] 训练数据：`slimpajama_chunks_2048_hunyuan.npy`（338514×2048）
- [ ] 训练器加 `experts_implementation="eager"`（待改，当前 `train_hunyuan_a13b_probe2.py` 加载处没有）
- [ ] lhz launch 脚本（keep13+fresh2，改自旧 apdcephfs 脚本）
- [ ] 起 arm B 训练看 s/step（合作者预期个位数 s/step）
