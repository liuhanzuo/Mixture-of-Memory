# Slot-Routed Evidence Memory — Roadmap (capability-upper-bound first)

设计定稿 2026-06-17（用户 directive + team-lead 综合）。目标：先证明 **slot-routed raw-evidence memory 能否把精确事实救回来**，再谈压缩/效率。

## 为什么做这个
已诊断铁证（RUN_REGISTRY）：固定 128-slot bank 保 gist 丢精确。
- LongBench W0 仅保留 base 开卷的 ~47%（narrativeqa 16%, qasper 38%）。
- sliding-PPL：pg19 +109%、proofpile +41%、codeparrot +44%（长程依赖越强退化越大）。
- 对话记忆 base≈3.8×(LME)/7×(LOCOMO)；连单段单跳精确事实都丢。
- 加训练窗口 / slot 128→256 都没破 32k 墙 → 不是容量问题，是 reader/interface 问题（待 oracle 证实）。

## 实验哲学（严格分三阶段，不许跳）
1. **Stage 1 — 大容量证明机制 work**（现在做）。
2. **Stage 2 — 找出哪部分真有用**（capacity curve + oracle 分解）。
3. **Stage 3 — 预算压缩与效率**（evidence pruning / beacon / KV 层裁剪 / hot-cold offload）。

第一版可以"大方"：evidence buffer 大、span 多、KV 层多、retrieval top-k 大、允许 oracle 对照。第一版**不追求比 vanilla 省或快**，只回答：*slot 下有足够 raw evidence 时，模型能否恢复 exact recall？*

## 数据结构（关键：不复制 KV）
**Global Evidence Store 存真实 raw KV；slot 只挂 evidence IDs/pointers。** 同一 span 可被多 slot 路由，复制 KV 会爆显存。

```
Slot_i = (a_i address, m_i state, P_i={evidence_id...}, meta_i)
EvidenceStore E = {e_1..e_M}
e_j = (id, text, token_ids[span_len], kv{layer:(K,V)}, chunk_id, pos_start, pos_end, score, tags{date/number/entity/update/relation}, timestamp, superseded)
```
evidence 同时存 **raw text + token_ids + mid-layer K/V + source position + tags** —— 为了 debug 能回答"模型答错时 retrieved evidence 里到底有没有 gold span"。

## ★ Raw-KV 实现方式（team-lead 判断，记录分歧）
- **当前已落地 MVP（commit 88214c2）**：evidence 作为 **hidden-state prefix token 在单层注入**，经冻结层自身 K/V proj。优点：每 token 只存 1 个向量（非 32 层 KV，不爆）、无需改 attention kernel、对冻结模型最忠实。
- **用户提的 mid-layer raw-KV（16/20/24）**：预计算各层 K/V 直接拼接。这是 **Stage-3 效率变体**，只有在 oracle 证明机制 work 后才值得做。
- **Oracle 测试若单层 prefix 不够，再升级到 mid-layer 注入**（evidence-coder 当前任务已含此 fallback 分支）。

## 第一组实验配置
- **Config A (Strong Evidence v0)**：S=128, chunk512, write/read top-k=16, 每 chunk 抽 64 spans, span_len=8, 每 evidence attach top-4 slots, 每 slot cap B=256, read rerank 取 top-256 spans ≈ 2048 evidence tokens。
- **Config B (Upper Bound v1)**：S=128/256, 每 chunk 128 spans, span_len 8/16, cap B=512, read 512 spans ≈ 4K-8K evidence tokens。
- **Config C (Oracle)**：直接给 gold evidence span + selected slot vectors。**必跑。**
- 上限护栏：32K context 最多 read 4K evidence tokens（仍 8× 压缩；128K→4K=32×），避免退化成 full context。

## Oracle 三对照（必做，防误判）
1. **Oracle slot** —— gold evidence 所属 slot 强制加入 selected。提升大 ⇒ selector/address 不准。
2. **Oracle evidence** —— gold span KV 直接拼给 decoder。能答对"4年9个月" ⇒ reader 没问题；答不对 ⇒ KV-prepend/interface 或冻结 decoder 使用能力有问题。**← 当前 evidence-coder 正在跑的 go/no-go。**
3. **Oracle evidence + slot vector** —— 比 evidence-only 更好 ⇒ slot vector 与 raw evidence 互补。

## Capacity curve（Stage-2 目标）
| Evidence budget | exact QA (LoCoMo/LongEval/NIAH) |
|---|---|
| slot-only | 当前分（低）|
| +512 / 1K / 2K / 4K / 8K tokens | ? 期望单调上升 |
| oracle | 上界 |
期望：slot-only 很差 → evidence budget 增大单调上升 → 2K/4K 后接近 oracle。曲线存在 ⇒ 机制有效。

## Evidence selection（第一版强监督，非作弊）
rule-based extractor 强制保存 date/time/number/人名/地点/组织/关系短语/更新短语/否定/偏好（"started/ended/moved to/changed to/no longer/now prefers/used to/since/until"）。duration 类错误 ⇒ date/time span 必须强制进 store。证明有效后再学习化 extractor。

## 诊断指标（不要只看最终分）
gold evidence written? / attached to retrieved slot? / retrieved@R? / attention mass on gold evidence? / exact numeric-date acc / stale (superseded) recall / latency-memory（Stage-3 才重点）。
"4年9个月→4年3个月" case 必记：gold start/end date written? attached same slot? query retrieved that slot? rerank got both spans? decoder attention on both?

## 推进顺序（路线图）
1. **Oracle evidence**：证明 frozen reader 能用 raw KV 答对 exact fact。← 现在
2. Large evidence store：非 oracle 下显著提升 exact QA。
3. Slot-routed evidence：证明 slot indexing 比全局 retrieval 更好/更省。
4. Budget curve：8K→4K/2K/1K/512 压缩。
5. Learned extractor 替换 rule-based。
6. Metadata overwrite 处理 LoCoMo 事实更新（superseded flag）。

## 当前状态
- MVP prefix 机制已落地（commit 88214c2，use_slot_evidence 默认关，smoke 全过）。
- evidence-coder 正在 .76 跑 oracle go/no-go（NIAH 4k 四臂：OFF / heuristic / oracle / oracle+slot；oracle 若单层不够则升 mid-layer 16/20/24）。
- 决策点：oracle 近满分 ⇒ GREEN（建 Global Evidence Store）；oracle 仍低 ⇒ 先修注入层/interface 再说。
