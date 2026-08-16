---
name: wzc1-writes-ckpt-1-7x-slower-than-zwfy6
description: "实测 wzc1 写 ckpt ~469 MB/s vs zwfy6 ~786-818 MB/s (1.7x 慢); 因 B200 compute step 更快, 同一 save_every 在 wzc1 的相对代价 +13.9% vs H20 +1.4%"
metadata: 
  node_type: memory
  type: project
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**2026-08-16 实测（三个 run、两个盘、三种 ckpt 大小，各自独立计时）：ckpt flush 的净写带宽两盘差 1.7×。**

| 盘 | run | ckpt | flush 净耗时 | 写带宽 | 摊到 save_every=500 |
|---|---|---|---|---|---|
| **wzc1** | LOCAL keep10fresh2 | 39.01 GB | 83.15 s | **≈469 MB/s** | 1.20 → 1.37 s/step (**+13.9%**) |
| zwfy6 | .73 keep12fresh2 | 43.87 GB | 55.85 s | ≈786 MB/s | 7.81 → 7.92 (+1.4%) |
| zwfy6 | .104 qwen3base_heal | 38.09 GB | 46.54 s | ≈818 MB/s | 5.74 → 5.83 (+1.6%) |

**Why:** 净耗时 = 含 flush 那一窗的 Δt/Δiter 减去紧邻的基线窗。三次都靠 **ckpt 文件 mtime 落在那一窗内** 确证，不是靠猜。

**How to apply:**
- **报 B200 训练速率时必须说清 compute 还是 amortised。** LOCAL 的相对惩罚（+13.9%）远大于 H20（+1.4%），
  原因是分子（写 39 GB 更慢）和分母（compute step 快 4.8×）同时不利 —— 「B200 快 4.8×」在 dense-ckpt run 上要打折。
- 估 ETA 用 amortised；判「是否变慢」用 compute rate，否则每 500 步都会误报一次异常。
- **只影响写大 ckpt 的训练**，不影响 eval-only 任务（如 B04 的 6-eval 回填）。
- 若要压 wzc1 上的 save 开销，杠杆是 `save_every`，不是换节点。

诊断顺序（比连采 util 更便宜，2026-08-16 验证）：看到 rank-0 低 util 或某窗速率异常 → **先查 output_dir 里最新 ckpt 的 mtime 是否落在该窗内**，命中即定案；再连采 util 确认已恢复。见 [[ckpt-interval-rate-is-not-compute-rate]]、[[one-sample-is-not-a-trend-or-state]]、[[cluster-two-disks-not-shared]]。
