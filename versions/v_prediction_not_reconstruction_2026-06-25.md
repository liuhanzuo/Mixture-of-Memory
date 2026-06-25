# 范式洞察:Prediction 不是 Reconstruction(2026-06-25 用户提出)

## 一句话

**memory 存储的目标是 prediction(下一 token 预测),不是 reconstruction(重建原文),所以 hidden states 可以激进压缩——大部分 hidden 维度对 prediction 无贡献。**

## 含义重写

### 1. 改写"压缩=丢精度"的旧认知
旧认知:slot 压缩(128 slot vs 12800 token)担心"丢精确事实"→ 长程读出失败。
新认知:**precise token 不需要保**,只需保**"chunk 里有什么 prediction-relevant 信号"**。qa5 不需要重建 "Mary gave the milk to Bill" 这个完整句子,**只需 "有 milk-transfer 事件,actor=Mary, recipient=Bill"** 这个低维命题。

### 2. 解释 L3 token-recon aux 全 REJECTED
历史 L3 token-level reconstruction aux(w=0.3/1.0 全 REJECTED,RUN_REGISTRY §"l3_recon_token_weight sweep"):**目标搞错了**。token-recon 强迫 slot 保留 surface token 信息,与 routing/检索的 prediction-relevant 目标**机制冲突**——这就是为什么 w=1.0 灾难性破坏长程检索。**如果当时用 prediction loss 做 aux 而非 recon,结果可能完全不同**。

### 3. 改写 slot vs hidden trade-off
原 trade-off:slot 压缩省显存但丢精度;hidden 保精度但不压缩。
新理解:**hidden 也可以压**,只要不要求 reconstruction。压缩比可激进到 1:100 甚至 1:1000,只要保 prediction-relevant 信息。

### 4. 对当前进行中方向的直接修正
- **Plan A SnapKV-on-chunks**:chunk score 不应基于 full-hidden cosine 相似度,而应基于 **prediction-relevant 子空间**的 score(可能通过 lm_head 投影或 task-aware projection)
- **树形 hidden memory**:内部节点压缩可极激进(1-10 维 prediction feature),叶子也可以不存 raw hidden。**压缩训练目标必须是 LM CE**(prediction),不是 MSE(reconstruction)
- **Information Bottleneck 视角**:I(X; Y_pred) 保留,I(X; X) 不约束 → 经典 IB 目标,理论框架现成

## 实践改动建议

1. 所有 chunk compression 模块用 **prediction-driven loss**(LM CE)训练,放弃 reconstruction-driven aux
2. 内部节点维度可从 4096 → 32/64/128 大幅压缩,实测 prediction quality
3. 选择器/淘汰策略用 **prediction-relevant 子空间**打分,可能比 raw q·k 更强
4. 对 BABILong eval 影响:0k-4k 在压缩下仍泄漏(背得住答案就行,不需重建),8k-32k 干净测压缩是否保 prediction-relevant 长程信号

## 与项目历史的关系

- 死路 H2 trained selector 崩溃,可能部分因 selector 用 reconstruction-similarity 而非 prediction-relevance 训练
- mass_coef 系列、L3 summary 等 hierarchical 设计,如果改用 prediction loss 训练 summary,可能性能不同
- 未来所有 memory 压缩设计的默认目标 = prediction,不是 reconstruction

## 待办

- [ ] 等树形设计 workflow 综合阶段时把此洞察纳入
- [ ] Plan A 实现时,score function 考虑用 lm_head 投影后的 query
- [ ] 未来 NOLEAK/T2 训练,考虑加 prediction-driven chunk compression aux
