# RMT-Slot Hybrid + LM2 Analysis 完整计划书

## 当前状态
- b200-2 (28.89.17.144) 空闲 1h15m，8 GPU 可用
- H/H2 在 step 2200 稳定训练，H 领先 H2 0.011 ratio
- MemLong b200-4 已稳定训练 (step 500 checkpoint)
- RMT-Slot 核心代码已存在 (`src/memory/rmt_slot/`, `scripts/train_rmt_slot.py`)

## 第一阶段：LM2 仓库分析 (30分钟)

### 目标
1. 克隆 LM2 仓库并分析其架构
2. 提取 BABILong 测试集和评估代码
3. 识别可借鉴的架构模式

### 步骤
```bash
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/
git clone https://github.com/your-org/LM2.git LM2-Analysis
cd LM2-Analysis
```

**分析重点：**
- `data/` 目录：查找 BABILong 数据集文件
- `eval/` 目录：BABILong 评估脚本
- `src/` 目录：LM2 内存机制实现
- `configs/`：超参数配置模式

### 预期产出
- BABILong 测试集集成路径
- LM2 架构洞察总结文档
- RMT-Slot 改进建议

## 第二阶段：RMT-Slot 混合架构实现 (2-3小时)

### 架构设计
**核心创新：**将 slot memory 的 top-k retrieval 与 RMT 的 sandwich injection 结合

**工作流程：**
1. **Segment 分割**：1024 tokens/segment
2. **Slot 检索**：从 N=64 slots 中检索 top-k=8
3. **Sandwich 构建**：`[retrieved_slots | content | placeholder]`
4. **Backbone 前向**：Llama-3-8B 全量 fine-tune
5. **EMA 写回**：仅更新选中的 top-k slots

### 文件结构
```
src/memory/rmt_slot/
├── __init__.py          # 模块导出
├── rmt_slot_model.py    # 核心混合模型类
└── config.py           # 配置参数

scripts/
├── train_rmt_slot.py    # 训练入口（基于 train_cross_attn_memory.py）
└── launch_rmt_slot.sh   # b200-2 启动脚本

versions/
└── v3_rmt_slot.md       # 架构文档
```

### 关键组件复用
- `TopKSelector` (src/memory/mem_space/selector.py)
- `MemoryBank` (src/memory/mem_space/memory_bank.py) 
- RMT sandwich 逻辑 (src/memory/rmt/rmt_v10.py)
- 训练基础设施 (scripts/train_cross_attn_memory.py)

### 超参数配置
```yaml
backbone: Llama-3-8B (full fine-tune)
num_slots: 64
top_k: 8
segment_length: 1024
learning_rate: 5e-6
gradient_accumulation: 4
max_steps: 2000
ema_gate_init: 0.3
```

## 第三阶段：LM2 学习成果集成 (1小时)

### BABILong 测试集集成
- 将 BABILong 添加到 NIAH 评估管道
- 创建 `eval_babilong.py` 评估脚本
- 在训练中定期运行 BABILong 评估

### LM2 架构模式应用
- 分析 LM2 的 memory gating 机制
- 如果适用，集成到 RMT-Slot 的 EMA 写回逻辑
- 借鉴 LM2 的训练策略和正则化技术

## 第四阶段：训练启动与监控 (持续)

### 启动命令
```bash
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
bash scripts/launch_rmt_slot.sh
```

### 监控指标
- **内存效率**：memory_ratio < H/H2 (目标 < 0.977)
- **NIAH 准确率**：突破 0% (H/H2 目前为 0%)
- **训练稳定性**：无 NaN，loss 单调下降
- **BABILong 表现**：长上下文理解能力

### 验证节点
- Step 100：基础功能验证
- Step 500：初步性能评估
- Step 1000：与 H/H2 中期对比
- Step 2000：最终结果分析

## 风险与缓解

### 技术风险
1. **RMT v10 生成崩溃重现**：使用 EMA 写回避免 "train all K" 稀释
2. **Llama-3 GQA 兼容性**：验证 4D attention mask 与 GQA 的兼容性
3. **位置编码漂移**：使用连续 0-based position IDs，避免 "Zero" 陷阱

### 资源风险
- b200-2 可能被其他实验占用：实时监控节点状态
- 显存不足：bs=1, grad_accum=4 确保 183GB 显存足够

## 成功标准

### 主要指标
- [ ] RMT-Slot 完成 2000 步训练
- [ ] memory_ratio < H/H2 同 step 表现
- [ ] NIAH accuracy > 0% (突破性进展)
- [ ] BABILong 测试集集成成功

### 次要指标
- [ ] LM2 分析产出有价值洞察
- [ ] 训练过程稳定无崩溃
- [ ] 代码可复用性高

## 时间线估计

| 阶段 | 时长 | 开始时间 | 结束时间 |
|------|------|----------|----------|
| LM2 分析 | 30分钟 | T+0 | T+30m |
| RMT-Slot 实现 | 2-3小时 | T+30m | T+3h |
| LM2 集成 | 1小时 | T+3h | T+4h |
| 训练启动 | 即时 | T+4h | - |
| 监控评估 | 持续 | T+4h | T+24h |

## 依赖与假设

### 依赖条件
- b200-2 保持空闲状态
- Llama-3-8B 模型文件可访问
- Dolmino-mix 数据集可用
- 网络连接稳定

### 技术假设
- RMT sandwich 逻辑在 Llama-3 上有效
- TopKSelector 和 MemoryBank 组件稳定
- 分布式训练环境配置正确

## 退出策略

如果 RMT-Slot 在 step 500 表现不佳：
1. 回退到纯 slot-based 方法（类似 H/H2）
2. 或尝试 RMT v10 的简化版本
3. 分析失败原因，迭代改进

---

**计划制定时间**：2026-05-08 23:30 CST  
**计划制定者**：Claude (基于用户需求分析)  
**下次更新**：第一阶段完成后