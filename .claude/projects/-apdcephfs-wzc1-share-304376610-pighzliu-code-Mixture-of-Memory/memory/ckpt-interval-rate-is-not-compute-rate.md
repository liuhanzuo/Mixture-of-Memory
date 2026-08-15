---
name: ckpt-interval-rate-is-not-compute-rate
description: "★2026-08-15 一天内 5 次把 ckpt-flush / rank-0 采样伪影当成故障: ckpt 间隔算 s/step 在 save_every==采样间距时系统性高估 ~13%; 低 util 单次读数总落在 GPU 0 (rank-0 干额外活), 不是 straggler; 只有 log 自带时间戳是干净源"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**：训练速率**只能**用 log 自带的逐行时间戳、取**不含 checkpoint flush** 的窗口来算。
用 checkpoint mtime 间隔算 `s/step`，在 `save_every == 采样间距` 时会**系统性高估约 13%**。

**Why**（2026-08-15 一天内同类误判 5 次，全部是我自己先报警再自己推翻）：

| 现象 | 我的第一反应 | 真相 |
|---|---|---|
| keep10 ckpt 间隔 1.360 s/step vs 基线 1.200 | 「+13.3%，超阈值」 | 124 个 20-step 区间里**只有 4 个** >1.5 s/step，都是 5.15，位置是每次 500-step save **之后**那个区间；其余 120 个中位数**正好 1.200** |
| paperC ckpt 间隔 6.472 vs 前一个 5.854 | 「+10.6%，不是采样噪声」 | log 时间戳给 5.717 = **比基线快 2.3%**；根因是 `save_every` 在 step 40000 附近从 5000 收紧到 **500**，于是我采的每个区间都含一次 flush |
| keep12 ckpt 已 61 分钟没更新 | 「疑似 stall」 | 它自己的 save cadence 就是 **66.0 分钟** → 在期内 |
| `.104` GPU0 读 2%、`.82` GPU0 读 0%、`.104` GPU0 读 3% | 「掉队/崩了」 | 各连采 3 次全是 99-100%；**三次低读全在 index 0** |

**两条可复用的判据**：

1. **flush 记在它之后的那个区间**。我第一次查 `step % 500 == 0` 想验证「慢区间=save 边界」，
   得到「no」，差点据此排除 save 假设。实际慢的是 90520/91020/91520/92020，即 save 之后的 20 步窗口。
   **别用 `%` 测，直接看慢区间是否紧跟 save 点。**
2. **低 util 单次读数总在 GPU 0 = rank-0 在干额外活**（logging / ckpt 序列化 / all-reduce 收尾），
   不是 straggler。**若低读出现在 rank 1-7，那才值得查。**

**How to apply**：
```bash
# 干净口径：从 log 时间戳取两个相邻 20-step 窗口
grep -aoE '^[0-9-]+ [0-9:]+,[0-9]+ - INFO - \[step [0-9]+/[0-9]+\]' "$LOG" | tail -3
# 然后 Δt/Δstep；两个窗口都算，一致才可信
```
- 报速率时**必须说清是 compute 还是 amortised**（含 flush）。两者差 ~13%，ETA 差 0.2 天。
  keep10：compute 1.200 / amortised 1.352；paperC：compute 5.717 / amortised 6.472。
- **`save_every` 会中途改**（paperC 5000→500）。台账里的 cadence 会过期，每次现算。
- 判 stall 用**该 run 自己的** cadence，不是邻居的（[[one-sample-is-not-a-trend-or-state]] 同型）。

**Related**：[[tqdm-elapsed-and-counter-have-different-origins]]（同一族：速率口径坑）、
[[one-sample-is-not-a-trend-or-state]]、[[read-env-not-source-defaults-for-running-procs]]、
[[absence-on-path-is-not-absence-on-disk]]（同日另一个「先下结论再被实测推翻」）。
