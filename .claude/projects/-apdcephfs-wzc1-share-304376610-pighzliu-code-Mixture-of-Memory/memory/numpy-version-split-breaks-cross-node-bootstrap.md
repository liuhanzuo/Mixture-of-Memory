---
name: numpy-version-split-breaks-cross-node-bootstrap
description: "★三个 numpy 版本分布在五个节点(LOCAL 2.3.5 / .82 2.4.6 / .73+.104+.21 2.5.1) → 同 seed 的 Generator.multinomial 结果不同, 跨节点 bootstrap 非确定; 比 a04 的 5e-4pp 硬校验松, 会静默 latent-fail"
metadata: 
  node_type: memory
  type: project
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**2026-08-13 实测：五个节点上有三个不同的 numpy 版本。**

| 节点 | numpy |
|---|---|
| LOCAL | **2.3.5** ← 离群最远 |
| `.82` | **2.4.6** |
| `.73` / `.104` / `.21` | **2.5.1** |

（都用 `/opt/conda/envs/torch-base/bin/python`）

## 后果：同 seed ≠ 同结果

A04 的 neighbour-variability 任务（commit `2d57da2`）实测：
`np.random.Generator.multinomial` 在 **2.5.1(.73) vs 2.4.6(.82)** 之间，**同一个 seed**、
**输入向量 sha256 相同**、**底层 RNG bit stream 逐位相同**的情况下，
**10000 行里有 19 行不同**。

影响幅度：动了 24 个 cell 里的 3 个，最大 **0.005294 pp**，只在 triviaqa 上，
没有改变任何 verdict，比它要检测的效应小 211×。**同一节点内输出是 byte-identical。**

## 为什么必须记下来：它比现有的硬校验更松

`proposal/active/A04-recovery-certification/code/a04_keep14_trajectory_ni.py` 里
archived-endpoint 复现的硬失败阈值是 **5e-4 pp**，
而这个跨节点 drift 是 **5.3e-3 pp** —— **drift 比阈值大一个数量级**。

→ 同一份分析代码，**在 `.73` 上跑会 pass，在 `.82` 上跑会 assert-fail**，
且失败原因看起来像"数字对不上/分析有 bug"，而不是"numpy 版本不同"。
这是**节点相关的潜伏失败**，只在换节点时才现形。

## How to apply

- **跨节点做 bootstrap / 任何 RNG 敏感的统计前，先核 numpy 版本**：
  `python -c "import numpy;print(numpy.__version__)"`
- 需要 bit-level 可复现的分析，**把所有 leg 放同一个节点**，或先统一 numpy。
- 看到 "reproduction hard-fail 差了 ~5e-3 pp" 这种量级的失败，**先怀疑 numpy 版本，
  不要先怀疑分析逻辑**。
- 这个 drift **不得**被引用来解释任何 >0.006 pp 的差异（A04 verdict 已明文限定）。
- 与 [[same-harness-runs-bit-identical]] 不冲突：那条说的是**同节点同 harness** byte-identical，
  实测仍成立；本条是**跨节点** RNG 实现差异，是另一个轴。

相关：[[cluster-two-disks-not-shared]]（另一类"看起来一样其实不同"的跨节点陷阱）、
[[one-sample-is-not-a-trend-or-state]]。
