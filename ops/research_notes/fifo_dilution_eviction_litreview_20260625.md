# FIFO Dilution + Buffer Eviction 文献调研(2026-06-25, workflow wbr15ytio)

## H2 发现 novelty 裁决
- ★真原创:MemoryLLM-style 训练 hidden FIFO 上,buffer 越小长档越好(b25>b50>b100,qa1 8k=40/27/12),**反驳 MemoryLLM 论文自己的 N↑→更好 scaling claim**。原因:他们 eval(知识注入-回忆)不 stress softmax dilution,BABILong needle 任务 stress。
- 不原创:dilution 原理(Scalable-Softmax 2501.19399 / Lost-in-Middle 2307.03172 / "Context Length Alone Hurts" 2510.05381 已发表);"按 attention 淘汰"= H2O/SnapKV/ChunkKV 的 chunk 级移植。

## 最接近前作
- ChunkKV(2502.00299):raw KV 按 chunk 分组打分保留 = 最近先例,但是 raw token KV 非训练 hidden buffer
- MemoryLLM(2402.04624):随机淘汰,N↑→更好;M+(2502.00592):trained MLP retriever(我们 gist selector 同类已崩)
- H2O(2306.14048)/SnapKV(2404.14469)/Scissorhands(2305.17118):token 级 attention 淘汰
- Landmark(2305.16300)/InfLLM(2402.04617):reader-native 选择器,无训练,我们的精神祖先
- 定位:"H2O/SnapKV/ChunkKV 应用在 MemoryLLM-style 每层训练 hidden FIFO 上"——文献无此组合

## ★ 推荐第一实验:Design A SnapKV-on-chunks(零训练,现有 b100 ckpt)
- 把 b100 全保留 → reader q·k 打分(obs_window=最后64 token),保留 top-25 chunk + 前2 sink(StreamingLLM)
- score(c_i) = mean_h mean_{q in obs_window} max_{k in c_i}(q·k/√d),跨32层 mean pool
- 零新参数(用 reader 自己 attention,避开 trained selector 死路)
- 修复 b25 "丢早期 needle" 盲区:按重要性保留而非按时间
- eval: qa1/qa2/qa5 × {0k,8k,16k,32k} n=100, 现有 b100 ckpt 不训练
- 预测:强成功 qa1 32k≥30(=b25); 中等 >15; 失败 ≤10
- 证伪线:若不超 b50 qa1 32k=24(2/3 task)→ reader-attn chunk 级不 transfer → 转 softmax-sharpening(SSMax 2501.19399)

## 其他设计
- Design B: H2O-on-chunks + EMA attention + StreamingLLM sink + recent-R floor(medium)
- Design C: 两级 active(b25)+ cold pool(100, M+ style)+ InfLLM 无训练 promotion(high)
