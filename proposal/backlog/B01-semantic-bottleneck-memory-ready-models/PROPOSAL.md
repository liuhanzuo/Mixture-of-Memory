# B01 — Memory-Ready Models via Semantic Bottleneck Pretraining

## 状态

**BACKLOG，科学证据较强；缺真实可部署的 compressed store 和强模型联合训练。**

## 核心主张

vanilla `h_j` 并不天然低秩。通过在语义 handoff 附近插入无 residual 的
feature-axis bottleneck，并从头训练，可显式形成：

- 可压缩的中层 cache representation；
- 对 rank truncation 更鲁棒的 upper-layer readout；
- 可与 depth-partitioned memory 组合的模型。

## 已有结果

- 1B vanilla `dim99≈1825`，bottleneck `≈438`
- 3B vanilla `≈2790`，bottleneck `≈467`
- rank128/256 的 ΔNLL 显著更小
- 固定 LM tax 约 4–8.5%，3B 小于 1B

## 当前缺口

1. 实际 store 仍保存恢复后的 full-width hidden，而非 bottleneck latent。
2. 还没有自然 long-memory task 的完整质量/存储/延迟前沿。
3. 未与 Read-LoRA、Write-LoRA 联合。
4. 强模型/A13B 受实现、初始化和 MoE 系统瓶颈阻塞。

## 最小下一步

四臂：

1. stock + Read-LoRA
2. bottleneck only
3. bottleneck + Read-LoRA
4. bottleneck + Read-LoRA + Write-LoRA

必须实际持久化 `d_bottle` latent，并报告 bytes/token。

## Kill 条件

- 低秩 latent 在 RULER/LongEval 上不保留精确 evidence；
- fixed LM tax 在强模型上扩大而非缩小；
- full-depth RAG 在同存储预算下严格支配。

