---
name: a-pipe-makes-a-failing-command-report-success
description: "同一天两次: gate 负控和 eval driver dry-run 都因 `cmd | tail` 把真实 rc=2 报成 rc=0; 判 gate/断言是否生效必须 `cmd > file; echo $?`"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**`cmd | tail` 之后的 `$?` 是 `tail` 的退出码，不是 `cmd` 的。** 一个正确失败的断言会被报成成功。

**Why:** 2026-08-16 同一天骗过我**两次**：

1. 给 `gate_null_expectation_bound.py` 做负控：还原修复前的数字后，gate **正确判 FAIL 并点名了行**，但我用 `python gate.py 2>&1 | tail -6; echo "rc=$?"` 读到 **`rc=0`**。真实退出码是 **2**。
2. 在 `.73` 上 dry-run `eval_paperb_ladder_200k.sh`：它**正确**在 ckpt 不存在处 `ASSERT-FAIL` + `FATAL`，我用 `| tail -25` 读到 **`DRYRUN_RC=0`**。去掉管道重测：**`TRUE_RC=2`**。

第二次尤其危险，因为**输出里明明印着 FATAL**，而退出码说成功。如果我只信退出码（自动化就会只信它），我会得出「driver 通过了 dry-run」的结论，而它其实在正确地拒绝运行。

这条我 memory 里早有一句（记在 `selftest-over-invented-inputs...` 末段），但它埋在别的教训里，**没有独立条目，所以复发了**。

**How to apply:**
- **判断 gate / 断言 / 前置检查是否生效，一律 `cmd > /tmp/out 2>&1; echo "rc=$?"`**，再单独 `grep`/`tail` 那个文件。**不要**在同一条命令里管道到 `tail`/`head`/`grep` 之后取 `$?`。
- 需要管道时用 `set -o pipefail`，或取 `${PIPESTATUS[0]}`。但**最稳的是先落文件**，因为落了文件还能重读、能 grep 别的模式，而管道只给你一次。
- **「输出里有 FATAL 但 rc=0」是自相矛盾的证据 → 说明我读错了退出码**，不是说明脚本坏了。自相矛盾时先怀疑自己的测量方式（同 [[read-env-not-source-defaults-for-running-procs]]）。
- 复发的元教训：**一条埋在别的 memory 末段的教训不会被想起。** 值得独立条目的，就给它独立条目。

见 [[selftest-over-invented-inputs-proves-nothing-about-the-pipeline]]、[[one-sample-is-not-a-trend-or-state]]。
