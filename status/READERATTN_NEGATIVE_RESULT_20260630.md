# reader-attn 选择器修复尝试: 诚实负结果 (2026-06-30)

> 进展快照。本文档记录"训练修 reader-attn 选择器"路线的完整尝试与**证伪**, 以及方法论教训。

## 背景
P1 probe: BABILong qa5 reader-attn needle recall@4≈0.13(纯question query, n100), 远低于免费 bm25(0.52)和 oracle(0.72)。目标: 让学习的 reader-attn 选择器追上/超过 bm25。

## 尝试链与结果

### 1. dilution 修复(失败)
假设: 训练 n_ctx=7 vs eval buffer=25 不匹配。做法: t2_gap8192+curriculum。
结果: select_ce 全程~0(74/75 step needle_rank=0), selector 在合成任务零梯度。**T2 任务对 selector 太简单**。

### 2. hard_distractors / format / 信号难度课程(L1-L5, 部分有梯度但不解决根本)
- name-collision(mention): select_ce~3%, 弱。
- format hard-neg(MEMORIZE对齐): 略强但仍~6%。
- L5 paraphrase(自然语言): select_ce~14-32%, 确实更难, 但 needle_rank 仍多数0。
- 信号难度课程(DifficultyCurriculum)实现并跑通, 但都没显著提升 BABILong recall。

### 3. mean-pool salience(假设最有希望, 但严格对照**证伪**)
假设(background agent): salience 只用 query 末位 token(qv=q_r[:,:,-1,:]), qa5末位是功能词"is"→退化。改 query 全序列 mean-pool。
- 早期 e2_meanpool probe(n7→n59): mean-pool r@4=0.40 vs last 0.14, 看似翻倍 → **过度乐观宣传**。
- **严格同probe对照(n50)证伪**: step2000基线 last r@4=0.38 ≈ mean-pool 0.36(打平!)。mean-pool **不优于** last-token。
- 重训(MEM_SALIENCE_QPOOL=mean)后 mean-pool 反降0.28(有害), last 持平0.40。
- **早期"翻倍"假象根源**: (a)n7小样本噪声; (b)e2_meanpool用整个last_chunk做query(last绝对值~0.4), P1用纯question(last~0.14), 跨probe伪对比。

## 结论
- **"训练/pooling 修 reader-attn 选择器"路线证伪**: reader-attn(任何 query pooling)≈ 0.36-0.40, 始终 < bm25 0.52 < oracle 0.72。
- **学习选择未赢启发式 BM25**。这是诚实的负结果。
- 真瓶颈不在 selector 的 query pooling, 可能在: (a)q·k salience 机制本身(layer16 content match)上限有限; (b)读出端(oracle 0.72 也不满); (c)reforward。

## 方法论教训(重要)
1. **probe 必须同实现同批样本才可比**: e2_meanpool(last_chunk query) vs P1(纯question)绝对值差3×, 跨比无效。
2. **小样本(n≤10)的"翻倍"几乎都是噪声**: n7看到的 mean0.40/last0.00, 满n50变 0.36/0.38。
3. 训练端 select_ce/needle_rank 满分 ≠ 下游 BABILong 提升(分布鸿沟)。
4. 我(assistant)在 mean-pool 上连续几轮过度乐观, 严格对照后自我证伪。教训: 关键结论一开始就用同probe满n对照卡死, 不被中间小样本带节奏。

## 推荐方向(待用户)
1. **方案1: BM25 做部署 selector**(recall 0.52, 免费, 已就绪 swa_bm25_token)。学习选择路线暂时认账。
2. 转攻**读出端**(oracle 0.72<1, token-reforward 读出还有 headroom)而非选择端。
3. 接受"选择器=BM25 启发式"作为系统组件, 把 novelty 放别处(slot净负、token-reforward机制等已坐实结论)。

## 仍坚实的结论(CONTROLLED_FINDINGS_20260629.md, 满n/同probe)
1. slot记忆净负(三档n100)。2. bm25>reader-attn选择。3. bm25>oracle是oracle定义缺陷(只定位target字面chunk)。4. 选择+reforward强依赖训练(distill全崩)。

## 红线
mix=0; 泄漏ckpt不引用; 真SOTA锚点pg19 nctx7 qa5 16k=16/32k=9。
