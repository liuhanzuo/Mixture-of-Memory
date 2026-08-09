# B03 — Does Layer Reset Survive LLM Pretraining?

## 状态

**HOLD / GATE-ONLY。最低优先级。**

## 不是新方法

top-K block reset 在参数操作上属于 LLF/layerwise reinitialization 家族。
不能使用：

- cyclic prune-regrow 新方法
- depth cycling
- 新 plasticity 机制

## 存活的科学问题

小数据视觉中有效的 forget-and-relearn，在单遍、知识密集的 decoder-only LM
预训练中是否仍有效？若无效，其代价是否表现为：

- PPL 可恢复；
- 参数化知识持续损失；
- 并随 reset 次数或数据 regime 扩大？

## 1B 核心 gate

`2 × 3`：

- data regime：single-pass / repeated-data
- reset count：`N={0,1,3}`

共同条件：

- 明确标注为 LLF operator
- matched token presentations
- matched LR/final architecture
- reset layers 的 optimizer moments 同时重置
- PPL + MMLU-content + closed-book QA + null floors

## 存活条件

至少满足一项：

1. 显著 reset × data-regime interaction；
2. PPL 与知识恢复曲线显著分离，且随 N/损伤时点扩大。

普通统一退化或普通 null 不足以独立成篇。

## 关闭条件

- N=0/1/3 只是单调统一退化；
- single-pass/repeated-data 无交互；
- 两轴不分离。

