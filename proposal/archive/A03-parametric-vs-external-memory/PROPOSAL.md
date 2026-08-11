# A03 — Where Should Lost Knowledge Live?

## 状态

**ACTIVE DESIGN；连接 Paper A 与 Paper B 的最高上限新方向。**

## 核心问题

结构压缩后丢失的知识，应当：

1. 通过 continued pretraining 重新写回参数；
2. 通过 raw-text retrieval 外部提供；
3. 通过 reusable residual/KV memory 提供；
4. 由参数恢复与外部 memory 联合承担？

## 核心假设

- NTP/CPT 能较快恢复 distributional fit，却不一定经济地恢复参数化事实；
- 外部 evidence 可以恢复在 closed-book 设置中丢失的事实；
- residual memory 可降低重复查询成本，但存在 readout-interface tax；
- 最优方案取决于知识是旧知识、新知识、更新知识还是多证据组合。

## 最小 pilot

先在 1B 或较小 Qwen/OLMo 上构造：

1. intact full-depth
2. pruned/shallow
3. pruned + CPT
4. pruned + raw-text RAG
5. pruned + CoMem/residual memory
6. pruned + CPT + memory

知识轴：

- 旧参数知识：MMLU-content、TriviaQA、PopQA
- 新注入事实：训练后新增、未见事实
- 更新知识：old/new conflict、temporal validity
- 多证据组合：RULER multikey、LongMemEval update/temporal

## 必报成本

- training tokens/FLOPs/GPU-hours
- inference latency/TTFT
- model parameters
- persistent bytes
- new-knowledge write cost
- old-knowledge forgetting
- stale/conflict error

## 成功条件

出现清晰分工之一：

- memory 用显著更少训练 token 恢复 factual/update 能力；
- CPT 对 PPL 更有效，但 memory 对事实更新/闭卷缺口更有效；
- joint 方案在总成本约束下形成 Pareto frontier。

## Kill 条件

- raw-text RAG 在所有质量/成本点严格支配 residual memory；
- CPT 能以更低总成本恢复全部目标能力；
- 1B pilot 所有知识指标均处于 floor，无法测量。

## 关键控制

- 每个接口必须高于自己的 null floor；
- 相同 evidence；
- 同 tokenizer/model family；
- 分离 closed-book 与 open-book；
- 不把“给答案原文”误称为恢复参数知识。

