# Runtime Capacity 64-Task Screening Results

日期：2026-08-05  
硬件：8×H20  
Checkpoint：`scaffold_sft_stage1/global_step_4465`

## 结果

| 配置 | HE+ | Failure | 成功样本平均 NFE | 全任务平均 NFE | 成功样本累计 token |
|---|---:|---:|---:|---:|---:|
| Tiny | 0.00% | 34.38% | **15.60** | 186.23 | **3,089** |
| Small | 3.13% | 37.50% | 44.03 | 219.52 | 9,307 |
| Medium | **10.94%** | **0.00%** | 69.55 | **69.55** | 15,960 |
| Large | **10.94%** | 10.94% | 62.61 | 111.77 | 13,910 |

## Failure 原因

```text
Tiny   22 / 64 depth_capacity_exhausted
Small  24 / 64 depth_capacity_exhausted
Medium  0 / 64
Large   7 / 64 model_call_budget
```

## 关键观察

1. Tiny 成功样本的平均 NFE 只有 15.6，但 34.4% 的任务因为 depth cap
   直接失败，且功能正确率为零。
2. Small 增加了表达容量，但 depth=2 仍造成 37.5% failure。
3. Medium 在该筛选集上消除了全部 generation failure，并达到与 Large
   完全相同的 10.94% HumanEval+。
4. Large 并没有提高功能正确率，反而有 7 个任务运行到 512-call 上限。
5. 因此容量不是越大越好。当前结果呈现：

```text
容量不足 → depth exhaustion
容量适中 → 稳定终止且功能最好
容量过大 → 弹性编辑长尾 / model-call budget failure
```

## 对论文主线的意义

CAP-012 的继续门槛已经满足：

- Tiny/Medium failure 差 34.38 个百分点；
- Small/Medium failure 差 37.50 个百分点；
- Tiny/Medium HE+ 差 10.94 个百分点；
- Medium/Large HE+ 相同，但 Medium 的全任务 NFE 低 37.8%；
- capacity hit 与 failure 一一对应。

这一结果直接支持：

> Runtime capacity 是结构化 dLLM 的一阶变量，并存在优于固定 Tiny 和
> 固定 Large 的中等容量 operating point。

下一步应：

1. 在 full HumanEval+ 上比较 Small/Medium/Large；
2. 在 MBPP+ 上使用完全相同配置；
3. 以 Medium 为目标，实现从 Small 开始、遇到 depth pressure 再扩容的
   Adaptive policy；
4. 研究 Large 的 7 个 model-call failure 是否由 expand/delete 长尾造成。
