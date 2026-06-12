# L3 隔离对照实验 - 完整设置总结

**日期**: 2026-06-04  
**作者**: CodeBuddy Code  
**状态**: 设计完成，待启动  

---

## 概览

已设计并实现了一个严格的L3隔离对照实验框架，用于干净地量化L3摘要模块的独立贡献。两条训练run之间唯一的差异是 **L3 的有无**（use_l3_summary 参数），其余所有组件（L1、L2、优化器、数据、随机种子）完全相同。

---

## 已创建的文件清单

### 1. 设计文档
- **`docs/L3_ISOLATION_EXPERIMENT_DESIGN.md`** (完整, 11 sections)
  - 核心原则与隔离条件
  - 配置文件结构
  - 启动命令
  - 对比分析指标
  - 预期结果解释
  - 常见问题排查

### 2. 配置文件 (`configs/l3_isolation/`)
- **`base.yaml`** - 共同基础配置
  - 模型与数据路径
  - 训练超参 (lr=1e-4, total_steps=500, seed=42)
  - L1 配置 (num_slots=128, top_k=16)
  - L3参数模板 (use_l3_summary=false, l3_n_summary=64等)
  
- **`without_l3.yaml`** - L1-only arm
  - 仅覆盖: `use_l3_summary: false`
  - 其他所有参数继承自 `base.yaml`
  
- **`with_l3.yaml`** - L1+L3 arm  
  - 覆盖: `use_l3_summary: true`
  - 设置: `l3_n_summary=64`, `l3_n_layers=2`, `l3_n_heads=8`
  - 其他所有参数继承自 `base.yaml`
  
- **`README.md`** - 快速指南和故障排查

### 3. 启动脚本 (`scripts/`)
- **`launch_l3_isolation_without.sh`** - 启动不含L3的训练
  - 配置: 所有L1参数, `--use_l3_summary false`
  - 输出: `outputs/l3_isolation/without_l3_TIMESTAMP/`
  - 日志: `logs/l3_isolation_without_TIMESTAMP.log`
  
- **`launch_l3_isolation_with.sh`** - 启动包含L3的训练
  - 配置: 所有L1参数 + L3参数
  - 输出: `outputs/l3_isolation/with_l3_TIMESTAMP/`
  - 日志: `logs/l3_isolation_with_TIMESTAMP.log`
  
- **`verify_l3_isolation_config.sh`** - 配置一致性验证工具

---

## 关键设计决策

### 1. 严格参数等价性

| 维度 | Without L3 | With L3 | 为什么相同? |
|------|-----------|---------|-----------|
| 模型 | Llama-3-8B | Llama-3-8B | 控制模型变量 |
| L1 (num_slots) | 128 | 128 | 隔离L3影响 |
| L1 (top_k) | 16 | 16 | 隔离L3影响 |
| 学习率 | 1e-4 | 1e-4 | 控制优化器 |
| 随机种子 | 42 | 42 | 重现性 |
| 总步数 | 500 | 500 | 控制训练长度 |
| 数据 | qa1,qa2,qa5 | qa1,qa2,qa5 | 相同训练集 |
| **L3开关** | **false** | **true** | 唯一差异 |

### 2. L3 参数配置

| 参数 | 值 | 理由 |
|------|-----|------|
| `l3_n_summary` | 64 | 标准Q-Former规模（64个查询） |
| `l3_n_layers` | 2 | 中等容量（~150M params） |
| `l3_n_heads` | 8 | 标准多头配置 |
| `l3_diversity_weight` | 0.1 | 防止查询塌缩 |
| `disable_l1_inject` | false (两个都) | 确保两条都有L1作为基础 |

### 3. 启动策略

两条run应该**并行启动**（相隔几分钟）：
```bash
# 终端1
bash scripts/launch_l3_isolation_without.sh

# 终端2 (稍后)
bash scripts/launch_l3_isolation_with.sh
```

这样可以在完整硬件相同的条件下对比（同一时间段的GPU温度、集群负载等）。

---

## 对比分析指标

### 训练过程指标

从日志/WandB中提取：
- **lm_loss**: 最终损失值
- **top1_sim**: 路由置信度（L3是否改善路由表达性）
- **inject_gate_mean/std**: 注入门参数学习情况
- **uniq_sel_slots**: 唯一选择的slot数（L3是否改善多样性）

**L3-specific 诊断**（仅在 with_l3 arm）：
- **summary_q_mean_cos**: 摘要查询间的平均相似度（< 0.5 为健康）
- **summary_q_max_cos**: 最大相似度（应 < 0.9）

### 最终评分

启动独立的BABILong eval：
```bash
python scripts/run_babilong_mem_space.py \
    --model_path "..." \
    --adapter_checkpoint outputs/l3_isolation/without_l3_*/mem_space_adapter.pt \
    --output_dir outputs/eval_l3_isolation_without
```

对标准指标：
- **Overall accuracy** (qa1, qa2, qa5)
- **Short vs Long performance** (0k-8k vs 16k-32k)
- **PPL delta**: (with_l3_ppl - without_l3_ppl) / without_l3_ppl

---

## 预期结果场景

### Scenario A: L3 显著改善 (PPL 下降 > 5%)

```
without_l3: PPL = 2.8
with_l3:    PPL = 2.65
delta:      -5.4%
```

**结论**: L3 贡献显著，推荐保留并进一步优化  
**下一步**: 消融 L3 超参（层数、摘要数量等）

### Scenario B: L3 轻微改善 (PPL 下降 1-5%)

```
without_l3: PPL = 2.8
with_l3:    PPL = 2.73
delta:      -2.5%
```

**结论**: L3 有潜力但成本高（150M params），收益边际  
**下一步**: 优化 L3 融合策略或检查是否与L1竞争

### Scenario C: L3 无益或有害 (PPL 下降 < 0%)

```
without_l3: PPL = 2.8
with_l3:    PPL = 2.85
delta:      +1.8%
```

**结论**: 当前L3配置不适合该任务  
**根因排查**:
- L3查询塌缩? (summary_q_mean_cos ≈ 1.0) → 增加多样性正则化
- L3与L1竞争? (inject_gate卡住) → 修改融合策略
- L3过拟合? → 需要正则化或dropout

---

## 启动前检查清单

- [ ] 模型文件存在: `models/Meta-Llama-3-8B-Instruct/`
- [ ] 配置文件完整: `configs/l3_isolation/{base,without_l3,with_l3}.yaml`
- [ ] 启动脚本可执行: `chmod +x scripts/launch_l3_isolation_*.sh`
- [ ] 输出目录准备: `mkdir -p outputs/l3_isolation logs`
- [ ] W&B API key设置 (如需在线日志): `export WANDB_API_KEY="..."`
- [ ] 验证配置一致性: `bash scripts/verify_l3_isolation_config.sh` ✓
- [ ] GPU 清理: 确认无残留进程（`nvidia-smi`显示所有GPU为0%）

---

## 文件位置快速参考

```
Mixture-of-Memory/
├── docs/
│   └── L3_ISOLATION_EXPERIMENT_DESIGN.md       ← 完整设计方案
├── configs/l3_isolation/
│   ├── base.yaml                               ← 共同配置
│   ├── without_l3.yaml                         ← Without L3 arm
│   ├── with_l3.yaml                            ← With L3 arm
│   └── README.md                               ← 快速指南
├── scripts/
│   ├── launch_l3_isolation_without.sh           ← 启动脚本 1
│   ├── launch_l3_isolation_with.sh              ← 启动脚本 2
│   └── verify_l3_isolation_config.sh            ← 验证脚本
└── EXPERIMENT_SETUP_SUMMARY.md                 ← 本文档
```

---

## 核心概念回顾

### 为什么要隔离 L3?

当前的多层内存（L1 slot + L2 token压缩 + L3摘要）结果中，不清楚每一层的独立贡献。L3是最新加入的，需要严格的ablation来量化它的真实效果。

### 严格隔离的重要性

在多组件系统中，如果配置不完全相同，差异的根源就会模糊不清：
- 如果L1参数不同 → 无法判断改善来自L3还是更好的L1配置
- 如果随机种子不同 → 无法排除随机性
- 如果学习率不同 → 无法排除优化器效应

### 为什么 disable_l1_inject=False (两个都)

这确保两条run都有完整的L1层。`disable_l1_inject=True` 用于"纯L3"实验（完全移除L1），但本实验的目标是"L1 vs L1+L3"，所以两个都保留L1。

---

## 后续步骤

1. **启动实验** (共 500 steps, 预计 8-10 小时/run)
   ```bash
   bash scripts/launch_l3_isolation_without.sh &
   sleep 300  # 间隔5分钟
   bash scripts/launch_l3_isolation_with.sh &
   ```

2. **监控进度** (每 30 分钟检查一次)
   ```bash
   tail -f logs/l3_isolation_*.log
   ```

3. **完成后提取结果**
   ```bash
   # 从日志提取最终指标
   grep "Training complete" logs/l3_isolation_*.log
   ```

4. **启动BABILong终评**
   ```bash
   # 两个checkpoint都eval
   python scripts/run_babilong_mem_space.py ...
   ```

5. **撰写结果分析** 
   ```
   results/l3_isolation_analysis_YYYYMMDD.md
   ```

---

## 参考文档

- **完整设计**: `docs/L3_ISOLATION_EXPERIMENT_DESIGN.md`
- **快速指南**: `configs/l3_isolation/README.md`
- **L3源码**: `src/memory/mem_space/l3_summary.py`
- **L1源码**: `src/memory/mem_space/layer.py`
- **训练脚本**: `scripts/train_mem_space_babilong.py`

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-06-04 | 初始设计完成 |

---

**准备状态**: ✓ 就绪  
**下一步**: 用户确认后启动两条run
