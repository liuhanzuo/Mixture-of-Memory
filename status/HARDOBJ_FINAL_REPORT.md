# HARDOBJ (`--last_chunk_loss_only`) 实验终版报告

> 最后更新：2026-06-14 12:20 GMT+8
> 状态：harder-objective 这条线**已彻底探尽**，容量轴 eval 收尾中。判据采用用户校准框架：**W0（纯 memory 读出）相对 BASE_mix0 的提升**，不拿 SWA gap 当成败线（SWA 等于开卷、纯 memory 预测本来就难）。

## 0. 一句话结论

`--last_chunk_loss_only`（context chunk no_grad 流式入 memory，LM loss 只在 target chunk）把**纯 memory 长程检索**从纯 LM 基线的个位数（BASE_mix0 qa5 8k/16k/32k = 5/5/3）稳定抬到 **~13/11/8（2-3 倍）**。这个提升：
- **真实**（远超随机 + 远超纯 LM 基线）
- **seed-robust**（ctx3 两 seed、ctx5 两 seed、ctx7 三 seed 全复现）
- **随 curriculum 深度温和单调**（16k 档 ctx3=9 → ctx7=11-14）
- **与 memory 容量无关**（N96/128/192/256 在 ctx7 下无差异）
- 但有明确 **plateau ~11-15，无质变**，远不及 SWA 读出上限（50-60）

→ 逼 memory 的训练目标方向正确，但单靠它**不足以突破 memory 读出瓶颈**。

## 1. depth × seed 矩阵（W0 纯 memory，qa5 长程 8k/16k/32k，step1000）

| 配置 | 8k | 16k | 32k | seed 复现 |
|---|---|---|---|---|
| BASE_mix0（纯 LM 锚点） | 5 | 5 | 3 | — |
| ctx3 (curriculum 0:3) | 13 | 9 | 6 | seed42=13/9/6, seed1234=13/9/6（完全一致）|
| ctx5 (0:5) | 11-14 | 10-11 | 8-9 | seed42=11/11/8, seed1234=14/10/9 |
| ctx7 (0:7) | 13-15 | 11-14 | 8-9 | seed42=15/14/8, seed1234=13/11/9, seed2026=14/11/9 |

- 深度正效应集中在 **16k 档**（ctx3 的 9 → ctx5/7 的 11-14），8k/32k 相近。
- ctx7 是最稳、最强的可跑深度。

## 2. 容量轴（ctx7，W0 qa5 长程，step1000）

| 容量 | 8k | 16k | 32k |
|---|---|---|---|
| N96 | 14 | 12 | 9 |
| N128 | 13-15 | 11-14 | 8-9 |
| N192 | 13 | 11 | 8 |
| N256 | 12 | 12 | 8 |

- **N96→N256 全程持平**（qa5 长程 8k≈12-15, 16k≈11-14, 32k≈8-9），扩/缩容量都不是杠杆——与此前 raw 容量 sweep 证伪一致，在 harder-objective 下依然成立。容量轴判决坐实（N192/seed1234 2nd-seed 在跑做 seed-robust 确认中）。

## 3. SWA 读出上限（W6，作参考非判据）

ctx 各档 swa6 qa5 在 8k/16k 达 50-60（如 ctx3-seed1234 step1000 swa6 = 51/51/43/28 @4k-32k；ctx7-2026 = 63/52/39）。与 W0 的巨大 gap **是预期的**（SWA 直接注意上下文原始 KV = 开卷），不作为 harder-objective 成败判据。它标示的是"memory 通道理论可达上限"——纯 memory 读出离这个上限还很远，是改进空间。

## 4. 不可跑配置

- **ctx10 (curriculum 0:10)**：bs2 与 bs1 均 hang（optimizer-init 后 0 step / GPU 0% / log 冻结）。结构性深度上限（序列过长触发 dataloader/NCCL 死锁），降 batch 无效。**ctx7 是最深可跑档。**

## 5. 下一步候选（待用户定）

harder-objective 已无同线增量。突破 plateau 的候选：
- **T2 合成 recall 任务**（associative recall / multi-hop）：让 target 必须从散落各 chunk 的 key-value 检索——纯 LM 缺失的检索压力，机制上最对症。需写数据构造 + 接入 `dolmino_train_step`。
- **收工**：当前结论已是清晰、可发表的负-正混合结果（逼 memory 有效但有天花板）。
