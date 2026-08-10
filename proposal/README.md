# Proposal Repository

本目录是 Mixture-of-Memory 项目的**唯一提案索引**。目标不是复制所有实验状态，
而是让每个方向只有一个最新、可执行、可证伪的入口。

最后整理：2026-08-08。

## 状态定义

- `active/`：证据已较完整，当前值得优先补 gate 或写作。
- `backlog/`：科学问题仍成立，但需要前置实验、资源或进一步新颖性核验。
- `archive/`：已死亡、被合并或被新 framing 取代的方向；保留必要 provenance，
  防止旧 claim 被误复活。
- `shared/`：多个提案共用的原始证据、代码和文献审计工具。

每个活跃/候选提案使用：

```text
PROPOSAL.md      最新主张、实验和 kill gate
STATUS.json      机器可读状态
SOURCES.md       证据源路径
```

## Related Work 强制门槛

从 2026-08-08 起，每个 active/backlog proposal 在启动新 GPU 实验前必须：

1. `PROPOSAL.md` 中有独立的 **Related Work / 新颖性边界**章节；
2. 列出最接近工作的具名 collision、重叠点、剩余空缺和“不得主张”；
3. `SOURCES.md` 中保存外部一手来源，而不只列内部结果；
4. benchmark、baseline、系统组件和评价协议都纳入 related-work 审计；
5. 若文献已覆盖核心主张，先收窄或归档 proposal，不能靠换应用包装。

当前逐提案缺口与补齐优先级见：

```text
shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md
```

其中 A03、A02、A04、B01 是最高优先级补洞项。

> ⚠️ **2026-08-10 更正**：本行原写「B09 当前最完整」。那说的是**文献/设计完整度**，
> 但会被读成「最接近可跑」，而 B09 恰恰相反——它的候选池（~10K agent trajectories
> / ~100K SFT rows）**在两个盘上都不存在**，Phase 0 数据审计无法执行，状态已改为
> `backlog_blocked_data_does_not_exist`。见
> `backlog/B09-trajectory-aware-sft-data-selection/DATA_AUDIT_VERDICT_20260810.md`。

## 当前排序

### Active

1. `active/A01-null-calibration-methodology/`
   - 跨 construct 的 input-blind null calibration。
   - MMLU interface failure、SQuAD majority prior、CKA layer-order null 和
     probe/native readout 是同一方法学框架的案例。
   - ⚠️ **2026-08-10：MAJOR REVISION**（外部审计
     `active/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md`
     §2.1+§7）。两条 claim 撤回（family-general step function；Llama-2 content
     strictly monotone）+ 一条降级（tie convention 翻 5/6 → 可执行 convention 翻
     0/6）。**读任何 2026-08-10 之前的 A01 verdict 文件前先读
     `active/A01-null-calibration-methodology/TCODEX_AUDIT_RESPONSE.md`。**
     `STATUS.json:status` 已不再声称 all gates passed。
2. `active/A02-comem-write-read-repair/`
   - 先验证已有 Write-LoRA/overlap repair 是否迁移到自然任务，再重做
     equal-latency frontier。
3. `active/A03-parametric-vs-external-memory/`
   - 结构压缩后，知识应恢复进参数，还是迁移到显式外部 memory interface。
4. `active/A04-recovery-certification/`
   - 用干净、多 seed、同语料同 token 的实验研究 recovery certification，
     而非把现有混杂 depth ladder 当 scaling law。

### Backlog

- `backlog/B01-semantic-bottleneck-memory-ready-models/`
- `backlog/B02-adaptive-depth-and-read-budget/`
- `backlog/B03-cyclic-layer-reset-boundary/`
- `backlog/B04-eval-fragility-incubator/`
  - `NARROWED_TO_OLMO_2_ONLY`。2026-08-10：Qwen cross-family「kill」降级为
    `NON_MATCHED_INCONCLUSIVE`（该 ladder 把 damage 与 training budget 混在一起）
    → 跨家族**未被检验**，而非被证伪。这**不是**晋升理由。见
    `DIRECTION_A_QWEN_LADDER_CONFOUND_ADDENDUM.md`。
- `backlog/B05-semantic-handoff-phase-diagram/`
- `backlog/B06-portable-decompression-adapter/`
- `backlog/B07-mutable-comem-serving/`
- `backlog/B08-memory-applications/`
- `backlog/B09-trajectory-aware-sft-data-selection/`
  - **`backlog_blocked_data_does_not_exist`（2026-08-10）**：候选池两盘皆无，
    先要做数据获取项目，Phase 0 审计才有对象。见 `DATA_AUDIT_VERDICT_20260810.md`。
  - 从 10K agent trajectories 展开的约 100K SFT rows 中选择 5K；
    以 trajectory/decision credit、target relevance 和集合覆盖替代 flat Top-K。

### Archive

- `archive/paperC-v1-frozen-cap/`：SQuAD prune-and-graft capability 与
  forward-probe depth predictor 已死亡。
- `archive/paperD-cross-family-stitching/`：跨家族 layer stitching 方法已死亡；
  存活的 CKA/null 与 affine readout pilot 已迁到 A01/shared。
- `archive/revival-slate/`：旧 proposal slate，包含后来被修正的数字，仅供 provenance。
- `archive/superseded/`：明确错误或已被更正的 proposal 文档。

## 使用规则

1. 新方向先写 `PROPOSAL.md` 和 kill gate，再启动 GPU。
2. 不得用旧 `status/*.md` 或 `versions/*.md` 中的历史 proposal 作为当前决策入口。
3. 原始结果尽量放在 `shared/evidence/`；大模型 checkpoint 和大数据仅在
   `SOURCES.md` 中引用，不复制。
4. dead proposal 的证据可以复用，但其旧 claim 不自动复活。
