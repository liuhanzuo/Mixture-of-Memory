---
name: a-gate-that-says-never-run-may-already-have-run
description: "我照 STATUS.json 的 gate 文本派活说 emit_slorb_ladder.py「从未执行过」, 但它当天 01:25 就跑完且 PASS; 派活前先 ls 消费目录的 mtime, 别信 gate 里写死的现状描述"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**gate 文本里的「这个工具从未被执行过」是**写 gate 那一刻**的状态描述，不是当前事实。派活前先看盘。**

**Why:** 2026-08-16 我派 agent 跑 B12 的 G0 leg 2，prompt 里照抄了 gate 原文：

> That tool has never been executed (the predecessor `emit_small_slorb_variants.py` says so in its own docstring), so this is a free correctness gate.

但 `proposal/backlog/B12-slorb-rank-efficiency/evidence/` 里已经有 **当天 01:25 / 01:44** 写的
`g0_leg2_rungA_manifest_20260816.json` + `g0_leg2_rungA_selfcheck_20260816.json`，`verdict = PASS`，
两个预登记常量都精确命中（`live_branch_params == 404750336`、`density == 0.5625`）。
**一条 `ls -la` 就能发现。** 我在派活前查了 `ready_queue.py`、读了 `STATUS.json` 的 `next_gate`，
**唯独没查 gate 所指的产物目录**。

代价：一个 agent 花 50 次工具调用重做了一遍已完成的腿。它做得很好（还独立负控了自己的 harness、
并额外 dry-run 了 rung P），但那不是我要的那份工作。

**它还发现了两件我原本不会知道的事**，都不是「重跑一遍」的产物：
1. **gate 自己写的命令行是错的，且错得静默**：`emit_slorb_ladder.py:220` 的 `ArgumentParser` 没传
   `allow_abbrev=False`，于是 gate 文本里的 `--mode ladder` 被 argparse **绑到 `--model`**，rc=0 通过。
   我用最小复现证实：`allow_abbrev=True` 时 `--mode ladder` → `Namespace(model='ladder')`；
   `allow_abbrev=False` 时才 SystemExit 2。**「照 gate 原文跑通了」不等于「跑了 gate 想跑的东西」。**
2. **rung A 的断言在算术上是被强制的**：`c=1` ⇒ `live == scope/16` 恒成立，
   所以那个 PASS 检验不到 `c>1` 的 coarsening 记账，也完全没碰 SVD 路径。
   **一个恒真的断言通过了，不构成证据。**

**How to apply:**
- 派活前**按消费者路径 `ls -la` 一遍**（`find <dir> -newermt '-1 day'` 更快）。gate 里的现状描述与
  `has never been executed` / `does not exist` / `is absent` 这类断言，**一律当作待核事实**。
  同族教训见 [[two-disk-rule-applies-to-main-too]]、[[absence-on-path-is-not-absence-on-disk]]。
- **照抄 gate 的命令行之前先读被调用脚本的 argparse**。缩写匹配会让错误 flag 静默生效；
  gate 说的文件名也可能是前身工具的（这里 gate 写 `slorb_variant_manifest.json`，
  而本工具实际写 `slorb_ladder_manifest.json`）。
- **断言通过后追问一句「它可能不通过吗？」** 若在该配置下恒真，就把它标成 `FORCED`，
  不要当成 checked。同 [[selftest-over-invented-inputs-proves-nothing-about-the-pipeline]]。
- 发现派错了就**立刻发 SendMessage 改向**（不要另起 agent，见
  [[continue-agent-with-sendmessage-not-agent]]），并如实说明前提错在我。
