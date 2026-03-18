# Mixture-of-Memory (MOM) — Agent Memory System v0

> 一个最小但可运行的研究原型，用于验证基于 **Mixture-of-Memory** 的 LLM 长期记忆策略。

---

## 项目概述

本项目围绕**冻结的 HuggingFace 因果语言模型**构建一套外部记忆系统，通过以下核心机制增强模型的长程记忆能力：

1. **Block Evaluator** — 在每个固定长度 block 结束时，评估哪些 token 位置是有价值的 anchor
2. **Anchor-guided Retrospective Gather** — 对选中的 anchor 使用注意力机制聚合 block 级别的潜在信息
3. **Mixture of Memory (MOM)** — 维护 3 个具有不同更新/保留动态的潜在矩阵记忆（fast / medium / slow）
4. **Write / Route / Retain Policy** — 学习决定写入强度、路由目标和保留率
5. **Memory Readout + Fusion** — 在每个 token 步从记忆中读取并通过门控残差融合影响生成

### v0 设计假设

- **不修改** transformer 内部架构
- **不做** 全量 SFT
- **冻结** backbone 模型，所有新功能作为外部模块添加
- 记忆保持**潜在化**（latent）且**注意力兼容**（attention-compatible）
- 记忆仅通过轻量级 readout + fusion head 影响生成（不注入每一层 transformer）
- 使用冻结 backbone 的 hidden states 作为短期分支（SWA）的代理

### 为什么冻结 Backbone？

v0 的目标是**验证记忆策略本身是否有效**，而非重新训练语言模型。冻结 backbone：
- 将实验变量隔离到记忆模块
- 大幅减少训练参数和计算量
- 确保 backbone 的语言能力不会在训练中退化
- 后续可通过 LoRA / partial unfreezing 逐步解锁

---

## 架构流程

```
输入序列 → [冻结 Backbone] → hidden states
                                    │
                          ┌─────────┼─────────┐
                          ▼                   ▼
                   按 block 切分          Memory Readout
                          │                   ▲
                   ┌──────┴──────┐           │
                   ▼             ▼           │
              Evaluator    Block Hidden      │
                   │             │           │
                   ▼             │           │
             AnchorSelector     │       ┌───┴───┐
                   │             │       │  MOM  │
                   ▼             ▼       │ (3×)  │
            Retrospective ──────┘       └───┬───┘
              Gather                        ▲
                   │                        │
                   ▼                        │
              MemoryWriter ─── update ──────┘
                   │
              (key, value, α, ρ, λ)

Memory Readout → FusionHead → fused hidden → lm_head → logits
```

---

## 各模块说明

| 模块 | 路径 | 功能 |
|------|------|------|
| **FrozenLMWrapper** | `src/backbone/lm_wrapper.py` | 加载并冻结 HuggingFace 模型，暴露 hidden states |
| **BlockEvaluator** | `src/anchor/evaluator.py` | 对 block 内每个 token 打分，支持 MLP / Transformer |
| **AnchorSelector** | `src/anchor/selector.py` | 从评分中选 top-k anchor，支持确定性和随机采样 |
| **RetrospectiveGather** | `src/gather/retrospective_attn.py` | 对每个 anchor 做单头注意力聚合 block 信息 |
| **MixtureOfMemory** | `src/memory/mom.py` | 管理 3 个潜在矩阵记忆 (fast/medium/slow) |
| **MemoryWriter** | `src/memory/update.py` | 生成写入决策: key, value, α, ρ, λ |
| **RetentionScheduler** | `src/memory/retention.py` | 为不同记忆提供保留率先验和调度 |
| **MemoryReadout** | `src/memory/readout.py` | 从所有记忆读取并通过学习路由融合 |
| **FusionHead** | `src/fusion/fusion_head.py` | 门控残差融合: h̃ = h + g ⊙ Wᵣr |
| **MemoryAugmentedLoss** | `src/training/losses.py` | 组合 LM loss + utility 辅助 loss |
| **MemoryTrainer** | `src/training/trainer.py` | 完整的 block-wise 训练循环 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
# 或
pip install -r requirements.txt
```

### 2. 在合成数据上训练

```bash
# 使用默认配置（tiny GPT-2）
bash scripts/train_stage1.sh --device cuda --seed 42

# 使用 CPU 快速测试
bash scripts/train_stage1.sh --device cpu --seed 42

# 使用特定 ablation 配置
bash scripts/train_stage1.sh --ablation full_mom --device cuda
```

### 3. 评估合成基准

```bash
bash scripts/eval_synthetic.sh --checkpoint checkpoints/checkpoint_best.pt --device cuda
```

---

## 基线配置 (Ablations)

通过 `--ablation` 参数切换不同基线，所有 ablation 配置位于 `configs/ablations/`：

| 配置 | 说明 |
|------|------|
| `no_memory` | 仅 backbone，无长期记忆 |
| `single_memory` | 单记忆 + 即时写入 |
| `mean_pool_write` | 单记忆 + 延迟写入（block 均值池化，无 evaluator） |
| `full_mom` | 完整 MOM + evaluator-guided write（完整方法） |
| `surprise_write` | 基于 surprise 的写入策略（对比 Titans 风格） |

```bash
# 运行 no_memory 基线
bash scripts/train_stage1.sh --ablation no_memory

# 运行完整方法
bash scripts/train_stage1.sh --ablation full_mom
```

---

## 合成数据集

5 种任务类型用于快速验证记忆能力：

| 任务 | 说明 |
|------|------|
| `simple_recall` | 简单关联回忆 |
| `updated_preference` | 偏好更新后查询最新值 |
| `past_value_query` | 偏好更新后查询旧值 |
| `long_distance_recall` | 长距离从句式回忆 |
| `distractor_heavy` | 大量干扰项中的回忆 |

### 评估指标

- **Exact Match / Accuracy**: 总体准确率
- **Update Accuracy**: 偏好更新后的准确率
- **Temporal Accuracy**: 时序查询准确率
- **Per-task Accuracy**: 各任务类型的细粒度准确率

---

## 训练日志

训练过程中记录以下信息：

- `train_loss` / `val_loss`: 训练和验证损失
- `lm_loss` / `utility_loss`: LM 损失和 utility 辅助损失
- `avg_anchors_per_block`: 每 block 平均选择的 anchor 数
- `write_alpha_mean`: 平均写入强度
- `write_route_entropy`: 路由分布熵
- `rho_mem{i}_mean`: 各记忆的平均路由权重
- `lam_mem{i}_mean`: 各记忆的平均保留因子
- `base_retention_mem{i}`: 基础保留率
- `mem_{name}_frobenius_norm`: 记忆矩阵范数

---

## 项目结构

```
Mixture-of-Memory/
├── configs/
│   ├── base.yaml                    # 基础配置
│   ├── synthetic.yaml               # 合成数据集配置
│   ├── model/tiny.yaml              # Tiny 模型配置
│   └── ablations/                   # Ablation 配置
├── src/
│   ├── backbone/                    # 冻结 backbone 包装器
│   ├── memory/                      # MOM 核心: mom, update, readout, retention
│   ├── anchor/                      # Block evaluator + anchor selector
│   ├── gather/                      # Retrospective attention gather
│   ├── fusion/                      # 门控残差融合
│   ├── data/                        # 数据集: synthetic, RULER, LongMemEval
│   ├── training/                    # 训练: losses, trainer, stages, utils
│   ├── eval/                        # 评估: synthetic, RULER, LongMemEval, metrics
│   └── common/                      # 通用: config, logging, seed, typing
├── scripts/                         # Shell 脚本
├── tests/                           # 单元测试
├── requirements.txt
└── pyproject.toml
```

---

## 运行单元测试

```bash
pytest tests/ -v
```

测试覆盖：
- 记忆更新形状和数值稳定性
- Anchor 选择正确性
- Retrospective gather 输出形状和注意力权重
- Readout 形状和路由权重
- Fusion 输出形状、残差连接和门控初始化

---

## 扩展路线

### 短期（v0.1）
- [ ] 集成 RULER 基准评估
- [ ] 集成 LongMemEval 基准评估
- [ ] 添加 wandb 日志支持

### 中期（v1）
- [ ] 实现真正的 SWA (Sliding Window Attention) 短期分支
- [ ] 添加 LoRA adapter 支持（Stage 2）
- [ ] 支持多 GPU 训练（FSDP）
- [ ] 实现 per-anchor utility target（更精细的 evaluator 监督）

### 长期（v2）
- [ ] 跨 block 注意力
- [ ] 将记忆注入 transformer 层（per-layer injection）
- [ ] 端到端联合训练（Stage 3）
- [ ] KV cache 感知的记忆融合

---

## 核心公式

### 记忆更新
$$M_t^{(i)} = \lambda_t^{(i)} M_{t-1}^{(i)} + \rho_t^{(i)} \alpha_t (k_t v_t^\top)$$

### 记忆读取
$$r_t^{(i)} = q_t^\top M_t^{(i)}, \quad r_t = \sum_i \gamma_t^{(i)} r_t^{(i)}$$

### 融合
$$\tilde{h}_t = h_t + g_t \odot W_r r_t$$

### Utility Target
$$u_n = L_{\text{future}}^{(-n)} - L_{\text{future}}^{(+n)}$$

---

## License

Research prototype — for academic and experimental use.
