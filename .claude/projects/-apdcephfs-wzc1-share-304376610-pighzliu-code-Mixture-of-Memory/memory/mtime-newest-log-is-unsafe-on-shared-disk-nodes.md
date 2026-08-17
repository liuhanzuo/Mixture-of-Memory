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

---

## ★ 2026-08-17 复发（zwfy6 侧），而且本文件上面已经预言了它

上面写着「zwfy6 三台之间也共享 zwfy6 盘 —— **同一个坑对它们同样成立**，只是我这次没在那边踩到
（三台的 run 名恰好互不相同）」。**今天在那边踩到了。**「run 名恰好不同」不是防护，是运气；运气用完了。

这次的外衣不是 `ls -t`，是**过松的 glob**。我在一条 for 循环里给三台各传一个 pattern：

| 节点 | 我传的 pattern | 匹到的文件 | 真相 |
|---|---|---|---|
| `.73` | `olmo2_7B_keep12fresh2_resume200k_73_0814` | ✅ 正确 | step 194840，mtime 秒级新鲜 |
| `.82` | `olmo2_7B_keep8` | ❌ `olmo2_7B_keep8fresh2.log`（**8-01**，step 48060，早死的 run） | 应是 `..._keep8fresh2_resume200k_82_0814.log`，step 169940 |
| `.104` | `qwen3` | ❌ `a01_qwen3_fine_73.log`（**8-09**，而且是 **.73 的**文件） | 应是 `paperC_qwen3base_heal_k8f2.log`，step 70400 |

`ls logs/*<pat>*.log | head -1` 里的 **`head -1` 是字典序第一个，不是最新的** ——
所以短 pattern 会稳定地、静默地选中一个**几周前的死 log**。
`.82` 那条尤其危险：它有 step、有 `/200000` 总数、格式完全正常，只是 step 号是 48060。
**如果我当时正在找 stall，"步号两轮没动" 会立刻成立 —— 因为那个文件永远不会再动。**

### 治法（比「用 pgrep」更具体一层）

- **两个判据同时用**：(1) `find logs -name '*.log' -newermt '-6 minutes'` —— 活的 log 必须刚被写过；
  (2) 文件名/内容含该节点 `--output_dir` 推出的 run 名。**只满足 (2) 会选到死 log；只满足 (1) 会选到同盘邻居的 log。**
- **顺手打印「最近 6 分钟被写过的所有 log」**。这一步这次直接暴露了真相：
  三台各自的 live log 全在列表里，且**三台看到的是同一张列表**（共享盘的铁证）。
- **`head -1` 不等于「最新」**。要最新就 `-newermt` 或 `stat -c %Y` 排序，不要靠 glob 顺序。
- **pattern 必须足够长以唯一确定 run**（含 `resume200k` / 节点号 / 日期这类判别性片段），
  宁可先 `ls` 看命中几个，再决定。命中 >1 个就说明 pattern 不够。
