# B02 — Query-Adaptive Method, Split Depth, and Evidence Budget

## 状态

**BACKLOG。先做 oracle-headroom，不应直接训练 router。**

## 假设

最优配置随 query type 变化：

- retrieval 是否有益具有任务符号；
- `j` 越深越快，但 readout cliff 与任务有关；
- 固定大 `k` 会增加噪声，固定小 `k` 会漏多证据。

controller 可联合选择：

- raw replay / CoMem / reusable KV
- split depth `j`
- evidence budget `k`
- retrieval rounds
- 低置信度 fallback

## Stage 0

使用现有 sweep 计算 per-example oracle action 和 regret：

- 若 oracle 相对最佳 fixed config 的收益不足，方向关闭；
- 若有明显 headroom，再训练 GBDT/小 MLP。

## 输入特征

- query length/type
- BM25/dense score gap
- retrieval entropy
- estimated evidence count
- document length
- expected reuse count
- latency/storage SLA

## 成功条件

- held-out task family 上质量距离 oracle `≤1pp`
- 平均 Read latency 降低 `≥20%`
- budget violation 低
- 不退化为恒选最浅 `j` 或最大 `k`

