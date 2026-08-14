---
name: tqdm-elapsed-and-counter-have-different-origins
description: "★resume 的 tqdm 里 `n/total` 是全局 iter 号、`[H:MM:SS<` 是本次 resume 段的 elapsed —— 两个字段原点不同, 直接 elapsed/n 相除得到的 s/it 是错的; 必须 Δelapsed/Δiter"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

2026-08-15 实测（SparseForge noslorb resume）：log 末行是

```
Training: 100%|██████████| 7500/7500 [12:01:44<00:00, 49.11s/it, loss=2.26]
```

我拿它算 `43304 s / 7500 it = 5.77 s/it`，与实测的 ~53 s/it 差**九倍**。
原因不是哪个数错了，而是**两个字段的原点不同**：

| 字段 | 含义 | 原点 |
|---|---|---|
| `7500/7500` 的分子 | **全局** iter 号 | 训练最初（含 resume 之前的 6700 步） |
| `[12:01:44<` | elapsed | **本次 resume 启动**那一刻 |
| `49.11s/it` | 累计平均 | 本次 resume 段（43304/800 = 54.13，接近但不等，因为 tqdm 有自己的平滑） |

⇒ **`elapsed / n` 在 resume 里恒为错**（分母多算了 resume 之前的所有步）。
唯一可靠的算法是 **Δelapsed / Δiter**，且窗口取 eval 周期的整数倍。

## 附带教训：一个窗口值不是"当前速率"

同一次读数，不同窗口给出**不同且有意义**的值：

```
100-iter 窗口: 49.99 s/it
200-iter 窗口: 53.30 s/it
400-iter 窗口: 54.86 s/it
```

这不是噪声 —— `hardening_x` 退火到 0.0，**arm 在末段真的加速了**。
我上一轮报的 `56.60` 是个**过时的 200-iter 窗口**，拿它当"当前速率"会得出"变慢了"的反向结论。
⇒ 报速率时要**说清窗口**，并且判趋势要看多个窗口的**方向**，不是一个数字。
同族：[[one-sample-is-not-a-trend-or-state]]。

## How to apply

- 算 s/it：抓两个 iter 号的 `[H:MM:SS<`，`(t2-t1)/(i2-i1)`。**永远不要**用末行的 elapsed 除以 iter 号。
- 报出来时带窗口长度（如 `53.30 (200it)`），并在怀疑趋势时多报一个窗口。
- 若两个窗口方向不一致，先想**是不是训练本身在变**（LR schedule / 退火 / save 边界），
  再想是不是自己读错了。
- 交叉验证：`ps -o etimes` 给进程真实寿命，可以拆出 resume 段长度。
