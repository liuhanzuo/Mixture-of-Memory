# B08 — Memory Applications Portfolio

## 状态

**BACKLOG PORTFOLIO。包含三个相互关联但风险不同的应用。**

## 1. Query-Conditioned Notes + Raw Evidence

第一阶段 retrieval 已接近饱和时，不再继续优化 naïve cosine reranker，而是：

- BM25+dense RRF 保召回；
- 对候选生成 query-specific notes；
- notes 显式标注事实、日期、说话人、更新链、冲突；
- reader 同时看 notes 和少量 raw evidence。

必须测 notes faithfulness，不能让 summary 成为唯一事实源。

## 2. Typed Personal Memory Ledger

- immutable event memory
- derived profile
- validity interval
- supersedes / tombstone
- source provenance / confidence
- current vs historical vs abstain

任务：overwrite、stale、contradiction、temporary、LongMemEval update/temporal。

## 3. Multi-Tier Pyramid Memory

- near：精确 residual
- far：compressed latent/notes
- profile：结构化长期状态

先做两层 MVP；当前完整 pyramid 实现风险高，不能直接大规模投入。

## Kill 条件

- notes-only 幻觉率高且 notes+raw 不优于 raw；
- typed ledger 不降低 stale/conflict error；
- pyramid 的 far memory 读取成本吞没固定 Read 优势。

