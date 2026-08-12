---
name: one-sample-is-not-a-trend-or-state
description: "★★一个瞬时采样不能判定趋势或状态 — tqdm s/it 不是 cadence(要用 elapsed/iter), GPU mem 要 >=3 次采样, 日志\"卡住\"要从 resume banner 起算; 2026-08-12 一天内同类错误 4 次"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**一个瞬时采样(instantaneous sample)不能用来判定趋势或状态。** 每次我这样做都得出了错误结论，然后自己推翻。

## 具体口径（都是实测踩出来的）

| 想知道什么 | ❌ 错的读法 | ✅ 对的读法 |
|---|---|---|
| 训练速度 | tqdm 行里的 `93.15s/it`（**瞬时**，checkpoint flush 时会飙到 6-7×） | `elapsed / iter`。采样 s/it 历史（每 150 iter 一个点）看有没有 snap back |
| GPU 显存 | 单次 `nvidia-smi`（flush 后分配器释放的瞬间会读到 17GB vs 真实 127GB） | 至少 **3 次采样、间隔 5s** 再下结论 |
| 日志"卡死" | 从**launch** 起算 | 从 **resume banner 写完**起算；先取同类 run 自己的 resume→first-step 间隔做基线（keep8 实测 2min1s） |
| 单调/连续下降 | 稀疏前缀（4-5 个点） | 只报 **matched iter 的 sign-consistency** + 说明在不在 divergence band 内 |

## 为什么反复犯

因为瞬时值**就在眼前**，而正确的量需要多取一次。省掉那一次的代价是：写出"7× slowdown / 配对错了 / 卡死 8 分钟 / 单调漂移"这类**听起来很重要所以会触发行动**的假警报。

2026-08-12 一天内 4 次：tqdm 93s/it 误判 7× 变慢、17GB 误判两臂配置不匹配、keep10 "frozen 8min"（实际 2min23s）、以及此前的 monotone drift / flip_ratio offset / 5th consecutive decline 三连。

## 规则

**报"异常/趋势"之前，先问：我这个结论建立在几个采样上？如果是 1 个，再取 2 个。** 代价是几秒钟，收益是不发假警报、不误 kill 健康 run。

同源教训见 [[same-harness-runs-bit-identical]]（声称 noise floor 前先跑 same-code 对照）和 [[two-disk-rule-applies-to-main-too]]（声称"不存在"前先两盘都搜）——都是「结论前先补一次测量」。
