# A04 — Recovery Certification after Structural Injury

## 状态

**ACTIVE DESIGN。现有 Paper B 提供案例和 harness，但不是干净 scaling law。**

## 稳健起点

现有最可信结论是：

> 单条 keep14 recovery 路径上，PPL 持续改善不能单独证明已经接近 intact target。

不应直接使用现有 depth ladder 推导普遍规律，因为它混入：

- 两个大小相差 2.046× 的训练语料；
- 不同 checkpoint steps；
- historical LR grouping bug；
- 未记录的原始 seed；
- partial-shard 与 runtime drift。

## 提案

建立预注册的 recovery certification protocol：

1. 同一物理训练数据，记录 SHA256、rows、tokenizer hash；
2. 同 token presentations/FLOPs，而非只按 optimizer steps；
3. 同 optimizer、LR、batch、runtime；
4. 至少 2 seeds，主结果 3 seeds；
5. checkpoint grid 在看结果前冻结；
6. 每个 checkpoint 联合报告：
   - in-domain/OOD PPL
   - MMLU letter/content 及各自 null
   - closed-book QA
   - core likelihood/MC
   - run-level uncertainty

## 1B MVP

四个结构：

1. prefix + fresh tail
2. contiguous keep-only
3. ShortGPT/non-contiguous policy
4. random trunk + inherited lexical/readout interface

在 3 个 token budgets 上评测，2 seeds。

## 关键研究问题

- likelihood recovery 何时能认证 target recovery？
- construction 的影响是否在 matched-PPL 后仍存在？
- final block、fresh tail、继承层数量分别贡献什么？
- 哪些 stopping rule 会错误提前宣布“恢复完成”？

## 成功条件

- 结构策略差异跨 seed `≥2pp`；
- matched-PPL 下 capability 差异仍显著；或
- 明确证否 PPL certification，并给出可复现的停止规则。

## Kill 条件

- 1B 全部目标指标处于 floor；
- 多 seed 后 construction 差异被训练方差完全吞没；
- 结果只能复述“不同指标不同”，无法提出 certification rule。

