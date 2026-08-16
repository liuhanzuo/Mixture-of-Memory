---
name: mtime-newest-log-is-unsafe-on-shared-disk-nodes
description: "在 .212 上跑 ls -t logs/*.log 拿到的是 LOCAL 的 log —— 两台共享 wzc1 盘, mtime 排序选的是最近写盘的那台; 认远程任务必须先 pgrep 再从 --output_dir 反查 log"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**「按 mtime 找最新 log」这条规则在共享盘的节点对上是错的。** 它会静默返回**另一台机器**的 log，而且内容看起来完全正常 —— 有 step、有 loss、有时间戳、还在推进。

**Why:** 2026-08-16 heartbeat，我 ssh 到 `.212` 跑 `ls -t logs/*.log | head -1`，拿到 `logs/olmo2_7B_keep10fresh2_resume200k_local_0815.log` —— 那是 **LOCAL 的 keep10 训练**。因为 **LOCAL 和 `.212` 共享 wzc1 盘**，`logs/` 是同一个目录，mtime 排序只会选出「最近写盘的那台」，与我登录的是哪台无关。

如果我照它报账，就会把 LOCAL 的 step/loss/rate 当成 `.212` 的报出去，而 `.212` 真正在跑的 distill 完全没被看到。**两个节点会报出同一份数字，看起来「都健康」。**

顺带两个更浅的坑同一轮踩到：
- ssh 过去**落地在 `/root`**，不是仓库根 → `ls: logs/*.log: No such file or directory`。必须显式 `cd`。
- 我猜 log 名含 `distill` 去 glob，`.212` 的实际文件叫 `olmo2_7B_keep14_distill_212_0815.log`（在**本机**也能直读，因为同盘）。**猜文件名 = 静默返回空。**

**How to apply:**
- **共享盘节点（LOCAL + `.212`）认任务一律走 `pgrep -af 'torch.distributed.run|train_'`**，从命令行里读 `--output_dir`，再由 output_dir 反查它自己的 log。**不要用 mtime。**
- zwfy6 三台（`.73/.82/.104`）之间也共享 zwfy6 盘 —— **同一个坑对它们同样成立**，只是我这次没在那边踩到（三台的 run 名恰好互不相同）。
- 判据：**log 里的 run 名 / output_dir 必须和你以为在查的那台的任务对得上**，对不上就是拿错了文件。
- 相关但独立：同一轮我还用 JSON 形状 `"overall":` 去 grep reviewer 的 markdown `- **overall**:`，得到「还没有分数」的错误结论。**三份 Claude review 用了三种不同写法**（`- **overall**: 5 / 10`、`- **Overall**: 5`、`- Overall: **5 / 10**）。抽字段要用格式无关的正则，别 grep 自己预期的格式 —— 见 [[read-what-the-consumer-reads-not-the-bare-key]]。

见 [[cluster-two-disks-not-shared]]、[[one-sample-is-not-a-trend-or-state]]、[[read-env-not-source-defaults-for-running-procs]]。
