# 🧠 MoM-Agent: Mixture-of-Memory for Local-Attention Agents

> **Hierarchical Memory for Local-Attention Agents**
>
> 三级层次化记忆系统，补偿 SWA（Sliding-Window Attention）/ 局部注意力骨干模型丢失的长程上下文访问能力。

---

## 📋 目录

- [研究动机](#研究动机)
- [系统架构](#系统架构)
- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [配置系统](#配置系统)
- [评测任务](#评测任务)
- [训练](#训练)
- [测试](#测试)
- [License](#license)

---

## 研究动机

大型语言模型（LLM）在长上下文对话中依赖全注意力（Full Attention）来维持上下文一致性，
但全注意力的 **O(n²)** 复杂度在超长序列场景下代价极高。滑动窗口注意力（SWA）
将复杂度降至 **O(n·w)**，但代价是**丢失窗口之外的长程信息**。

**MoM（Mixture-of-Memory）** 引入三级层次化记忆，在保持 SWA 高效推理的同时，
恢复对长程上下文的访问能力，使 **SWA + MoM ≈ Full Attention 基线性能**。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      MemoryAgent                            │
│  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │  TurnProcessor │  │ SessionRunner  │  │   Backbone    │  │
│  │  (单轮处理)    │  │ (会话编排)     │  │  (SWA/Full)   │  │
│  └───────┬───────┘  └───────┬────────┘  └───────────────┘  │
│          │                  │                               │
│          ▼                  ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MemoryScheduler (调度器)                │   │
│  │  ┌─────────┐   ┌──────────┐   ┌──────────────────┐ │   │
│  │  │   L1    │   │    L2    │   │       L3         │ │   │
│  │  │ 关联矩阵 │   │ 事件记忆  │   │   语义/画像记忆   │ │   │
│  │  │ (在线)   │   │ (turn级) │   │   (session级)    │ │   │
│  │  └─────────┘   └──────────┘   └──────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 三级记忆

| 层级 | 名称 | 更新频率 | 存储内容 | 模块 |
|------|------|---------|---------|------|
| **L1** | 关联矩阵记忆 | 每 token step（在线同步） | 衰减关联矩阵 (key→value)，门控融合到隐藏状态 | `src/memory/l1/` |
| **L2** | 事件级记忆对象 | 每 turn / chunk 边界（异步） | 从最近消息直接聚合的结构化事件对象，支持合并与检索 | `src/memory/l2/` |
| **L3** | 语义/画像记忆 | 每 session 结束（异步） | 从 L2 抽象总结而来的长期画像条目，支持冲突修订 | `src/memory/l3/` |

### 信息流

```
用户消息 ──→ L1 在线写入 ──→ 门控融合到骨干隐藏状态
         ──→ 消息缓冲区 ──→ chunk/turn 边界 ──→ L2 聚合 ──→ L2 存储
                                              └──→ session 结束 ──→ L3 总结 ──→ L3 存储
检索方向:  Query ──→ L2 检索 (top-k 事件) ──→ L3 检索 (top-k 画像) ──→ 注入 Prompt
```

---

## 安装

### 环境要求

- Python ≥ 3.11
- PyTorch ≥ 2.1.0
- CUDA (可选, 用于 GPU 加速)

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repo-url> mom-agent
cd mom-agent

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装项目 (可编辑模式, 含开发依赖)
pip install -e ".[dev]"

# 或使用 requirements.txt
pip install -r requirements.txt
```

---

## 快速开始

### 1. 交互式对话

```bash
# 使用 SWA + 完整 MoM 记忆
python -m scripts.run_chat \
    --config-name swa_mom \
    model=swa_qwen \
    memory=mom_full

# 使用 Hydra 配置覆盖
python -m scripts.run_chat \
    --config-name swa_mom \
    memory.scheduler.l2_chunk_size=3 \
    experiment.run.seed=123
```

### 2. 运行评测

```bash
# 完整评测 (所有任务)
python -m scripts.run_eval \
    --config-name swa_mom \
    experiment.eval_tasks='[synthetic_update,profile_bench,longhorizon_chat]'

# 单任务评测
python -m scripts.run_eval \
    --config-name swa_mom \
    experiment.eval_tasks='[profile_bench]'
```

### 3. 消融实验

```bash
# 运行全部消融实验 (SWA-only → +L1 → +L1+L2 → +L1+L2+L3 → FullAttn)
python -m scripts.run_ablation \
    experiment.run.seed=42 \
    experiment.run.num_sessions=10
```

### 4. 数据构建

```bash
# 从原始消息日志构建 L2 事件记忆
python -m scripts.build_l2_from_messages \
    --input data/raw/chat_logs.jsonl \
    --output data/processed/l2_objects.jsonl

# 从 L2 事件构建 L3 画像记忆
python -m scripts.build_l3_from_l2 \
    --input data/processed/l2_objects.jsonl \
    --output data/processed/l3_profiles.jsonl
```

---

## 项目结构

```
mom-agent/
├── configs/                        # Hydra 配置文件
│   ├── eval/                       # 评测任务配置
│   │   ├── longhorizon_chat.yaml   #   长程对话评测
│   │   ├── profile_bench.yaml      #   画像准确度评测
│   │   └── synthetic_update.yaml   #   合成更新评测
│   ├── exp/                        # 实验组合配置
│   │   ├── fullattn_baseline.yaml  #   全注意力基线
│   │   ├── swa_only.yaml           #   仅 SWA (无记忆)
│   │   ├── swa_l1.yaml             #   SWA + L1
│   │   ├── swa_l1_l2.yaml          #   SWA + L1 + L2
│   │   └── swa_mom.yaml            #   SWA + 完整 MoM
│   ├── memory/                     # 记忆层配置
│   │   ├── l1_assoc.yaml           #   L1 关联矩阵配置
│   │   ├── l2_episode.yaml         #   L2 事件记忆配置
│   │   ├── l3_profile.yaml         #   L3 画像记忆配置
│   │   └── mom_full.yaml           #   完整 MoM 调度器配置
│   └── model/                      # 骨干模型配置
│       ├── debug_tiny.yaml         #   调试用小模型
│       ├── fullattn_qwen.yaml      #   Qwen 全注意力
│       └── swa_qwen.yaml           #   Qwen SWA
│
├── src/                            # 核心源码
│   ├── agents/                     # Agent 层
│   │   ├── memory_agent.py         #   MemoryAgent 主类
│   │   ├── session_runner.py       #   SessionRunner 会话编排
│   │   └── turn_processor.py       #   TurnProcessor 单轮处理
│   │
│   ├── backbone/                   # 骨干模型抽象
│   │   ├── interfaces.py           #   BackboneModel 抽象接口
│   │   ├── swa_model.py            #   SWA 骨干实现
│   │   ├── full_attention_model.py #   全注意力骨干实现
│   │   └── hidden_state_types.py   #   隐藏状态类型定义
│   │
│   ├── memory/                     # 三级记忆系统
│   │   ├── l1/                     #   L1: 关联矩阵记忆
│   │   │   ├── assoc_memory.py     #     AssociativeMemoryL1 主类
│   │   │   ├── gating.py           #     门控网络 (选择性融合)
│   │   │   ├── reader.py           #     矩阵读取器
│   │   │   └── writer.py           #     矩阵写入器
│   │   ├── l2/                     #   L2: 事件级记忆
│   │   │   ├── aggregator.py       #     消息→事件聚合器
│   │   │   ├── merger.py           #     事件合并器
│   │   │   ├── object_store.py     #     L2ObjectStore 存储
│   │   │   ├── retriever.py        #     L2 检索器
│   │   │   └── types.py            #     ChatMessage / MemoryObject 类型
│   │   ├── l3/                     #   L3: 语义/画像记忆
│   │   │   ├── summarizer.py       #     L2→L3 摘要总结器
│   │   │   ├── profile_store.py    #     ProfileStore 存储
│   │   │   ├── reviser.py          #     冲突检测与修订器
│   │   │   └── formatter.py        #     画像格式化输出
│   │   ├── scheduler.py            #   MemoryScheduler 统一调度器
│   │   └── state.py                #   MoMState / MoMStats 状态管理
│   │
│   ├── eval/                       # 评测模块
│   │   ├── metrics.py              #   基础指标 (ROUGE, BERTScore 等)
│   │   ├── update_eval.py          #   信息更新准确度评测
│   │   ├── summary_eval.py         #   摘要质量评测
│   │   ├── retrieval_eval.py       #   检索效果评测 (Recall@k, MRR)
│   │   └── cost_eval.py            #   开销评测 (延迟, 内存, FLOPS)
│   │
│   ├── tasks/                      # 评测任务定义
│   │   ├── longhorizon_chat_task.py#   长程对话任务
│   │   ├── profile_task.py         #   画像构建任务
│   │   └── synthetic_update_task.py#   合成信息更新任务
│   │
│   ├── training/                   # 训练脚本
│   │   ├── train_gate.py           #   L1 门控网络训练
│   │   ├── train_l2_aggregator.py  #   L2 聚合器训练
│   │   └── train_l3_summarizer.py  #   L3 总结器训练
│   │
│   └── utils/                      # 工具库
│       ├── io.py                   #   文件 I/O (JSON/JSONL/YAML)
│       ├── logging.py              #   日志配置 (Rich)
│       ├── seeds.py                #   随机种子管理
│       ├── text.py                 #   文本处理 (截断/token计数)
│       └── time.py                 #   计时器与时间工具
│
├── scripts/                        # 可执行脚本
│   ├── run_chat.py                 #   交互式对话
│   ├── run_eval.py                 #   评测流水线
│   ├── run_ablation.py             #   消融实验
│   ├── build_l2_from_messages.py   #   离线构建 L2
│   └── build_l3_from_l2.py         #   离线构建 L3
│
├── tests/                          # 单元测试
│   ├── test_l1.py                  #   L1 关联矩阵测试
│   ├── test_l2.py                  #   L2 事件记忆测试
│   ├── test_l3.py                  #   L3 画像记忆测试
│   ├── test_scheduler.py           #   MemoryScheduler 测试
│   └── test_agent_smoke.py         #   Agent 端到端烟雾测试
│
├── data/                           # 数据目录
│   ├── raw/                        #   原始数据
│   ├── processed/                  #   处理后数据
│   └── cache/                      #   缓存
│
├── outputs/                        # 输出目录
│   ├── runs/                       #   实验运行记录
│   ├── metrics/                    #   评测指标
│   └── traces/                     #   对话轨迹
│
├── pyproject.toml                  # 项目配置与依赖
├── requirements.txt                # 依赖列表
└── .gitignore                      # Git 忽略规则
```

---

## 配置系统

项目使用 [Hydra](https://hydra.cc/) + [OmegaConf](https://omegaconf.readthedocs.io/) 进行配置管理，
支持 **YAML 组合** 和 **命令行覆盖**。

### 配置层级

```
configs/
├── exp/swa_mom.yaml          # 实验配置 (顶层入口)
│   ├── defaults:
│   │   ├── /model: swa_qwen  # 引用模型配置
│   │   └── /memory: mom_full # 引用记忆配置 (组合 l1+l2+l3)
│   └── experiment: ...       # 实验参数
└── memory/mom_full.yaml      # MoM 调度器配置
    ├── defaults:
    │   ├── l1_assoc           # L1 配置
    │   ├── l2_episode         # L2 配置
    │   └── l3_profile         # L3 配置
    └── scheduler: ...         # 调度参数
```

### 调度器关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `l1_sync` | `true` | L1 同步更新 |
| `l2_update_on` | `turn_end` | L2 更新触发时机 |
| `l2_chunk_size` | `5` | L2 聚合的 chunk 大小 |
| `l3_update_on` | `session_end` | L3 更新触发时机 |
| `context_budget.max_memory_tokens` | `512` | 记忆注入最大 token 预算 |
| `context_budget.l2_token_ratio` | `0.6` | L2 在预算中的占比 |
| `context_budget.l3_token_ratio` | `0.4` | L3 在预算中的占比 |

### 命令行覆盖示例

```bash
# 调整 L2 chunk 大小 和 L1 衰减率
python -m scripts.run_eval \
    --config-name swa_mom \
    memory.scheduler.l2_chunk_size=3 \
    memory.l1.decay=0.95

# 切换到全注意力基线
python -m scripts.run_eval \
    --config-name fullattn_baseline
```

---

## 评测任务

### Synthetic Update (合成更新)

测试记忆系统对**信息更新**的追踪能力。注入一系列属性声明，后续声明可能更新、推翻前面的值，
评估系统在多轮后是否能正确反映最新状态。

```bash
python -m scripts.run_eval --config-name swa_mom \
    experiment.eval_tasks='[synthetic_update]'
```

### Profile Bench (画像准确度)

测试 L3 画像记忆的**长期一致性**。经过多 session 交互后，评估从 L3 检索到的用户画像
是否与真实画像匹配（Precision / Recall / F1）。

```bash
python -m scripts.run_eval --config-name swa_mom \
    experiment.eval_tasks='[profile_bench]'
```

### Long-Horizon Chat (长程对话)

测试**长对话场景**下的上下文一致性。在 100+ 轮对话中，随机插入需要引用早期信息的问题，
评估系统的回答质量（ROUGE / BERTScore）。

```bash
python -m scripts.run_eval --config-name swa_mom \
    experiment.eval_tasks='[longhorizon_chat]'
```

### 消融实验

系统地对比不同记忆层级组合的效果：

| 配置 | L1 | L2 | L3 | 说明 |
|------|:--:|:--:|:--:|------|
| `swa_only` | ✗ | ✗ | ✗ | 纯 SWA (下界) |
| `swa_l1` | ✓ | ✗ | ✗ | + 关联矩阵 |
| `swa_l1_l2` | ✓ | ✓ | ✗ | + 事件记忆 |
| `swa_mom` | ✓ | ✓ | ✓ | 完整 MoM |
| `fullattn_baseline` | ✗ | ✗ | ✗ | 全注意力 (上界) |

```bash
python -m scripts.run_ablation --multirun \
    experiment.run.seed=42,123,456
```

---

## 训练

### L1 门控网络

训练 L1 门控，学习何时从关联矩阵读取、何时信任骨干输出：

```bash
python -m src.training.train_gate \
    --d-model 2048 \
    --hidden-dim 512 \
    --lr 1e-4 \
    --epochs 10 \
    --data-path data/processed/gate_train.pt
```

### L2 聚合器

训练 L2 聚合器，从消息 chunk 中提取结构化事件对象：

```bash
python -m src.training.train_l2_aggregator \
    --lr 2e-5 \
    --epochs 20 \
    --data-path data/processed/l2_train.jsonl
```

### L3 总结器

训练 L3 总结器，从 L2 事件中抽象出长期画像条目：

```bash
python -m src.training.train_l3_summarizer \
    --lr 2e-5 \
    --epochs 15 \
    --data-path data/processed/l3_train.jsonl
```

---

## 测试

```bash
# 运行全部测试
pytest

# 运行特定模块测试
pytest tests/test_l1.py -v
pytest tests/test_l2.py -v
pytest tests/test_l3.py -v
pytest tests/test_scheduler.py -v
pytest tests/test_agent_smoke.py -v

# 带覆盖率报告
pytest --cov=src --cov-report=html

# 仅运行快速测试 (跳过需要模型的测试)
pytest -m "not slow"
```

---

## 开发

### 代码风格

```bash
# Lint
ruff check src/ tests/ scripts/

# 自动修复
ruff check --fix src/ tests/ scripts/

# 类型检查
mypy src/
```

### 添加新的记忆层

1. 在 `src/memory/` 下创建新目录 (如 `l4/`)
2. 实现存储、聚合/总结、检索三个核心组件
3. 在 `MemoryScheduler` 中注册新层的初始化和调度逻辑
4. 在 `configs/memory/` 下添加对应 YAML 配置
5. 编写单元测试于 `tests/test_l4.py`

### 添加新的评测任务

1. 在 `src/tasks/` 下创建任务类 (继承 `BaseTask`)
2. 在 `src/eval/` 下添加任务专属评测器
3. 在 `configs/eval/` 下添加任务配置
4. 在 `scripts/run_eval.py` 中注册新任务

---

## License

MIT License. See [pyproject.toml](pyproject.toml) for details.
