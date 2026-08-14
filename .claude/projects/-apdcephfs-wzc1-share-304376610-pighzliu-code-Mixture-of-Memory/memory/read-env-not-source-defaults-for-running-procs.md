---
name: read-env-not-source-defaults-for-running-procs
description: "★一个跑着的进程的配置在 /proc/<pid>/environ 里, 不在源码的 ${VAR:-default} 里; 2026-08-14 我照源码默认值判 watcher 盯着 21h 前的死 log, 差点误 kill 两个正确武装的 watcher"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

判断一个**正在运行**的进程「配置成什么样」时，权威是 `/proc/<pid>/environ`（及 `/proc/<pid>/cmdline`），
**不是**源码里的 `${VAR:-default}`。源码给的是「没人 override 时会用什么」，不是「这次实际用了什么」。

**2026-08-14 实测**：`_run_sparseforge_tokenmatched_union9_watcher.sh:170/176` 的默认
`TRAIN_LOG=logs/sparseforge_tokenmatched_{slorb,noslorb}_08{11,12}_*.log`。我 grep 出这两个默认名、
`stat` 一看 **mtime 是 21 小时前**，于是推断「watcher 盯着一个已死的 log」。因为脚本里
`LOG_STALE_S=1800`，这个推断还自带矛盾（真盯着它早该在 21 小时前就触发了）——
矛盾本身就是「我读错了文件」的信号，不是「watcher 坏了」的信号。

真相：两个 watcher 都 **override 了** `TRAIN_LOG`，指向当天的
`sparseforge_tm_{noslorb,slorb}_RESUME_0814_*.log`（实测 age = 0 s / 37 s，活的）。
脚本第 156 行明写「TRAIN_NODE / TRAIN_LOG are OVERRIDABLE，must stay so」，
仓库里还有专门的 `_rearm_sparseforge_union9_watchers.sh` 干这件事。**我读了源码却没读环境。**

**Why**：差一点就以「watcher 盯错文件」为由去 kill 两个**完全正确武装**的 watcher，
而它们是 union-9 全部 9 个任务唯一的产出路径（in-run 的 `--finalize_lm_eval` 是死 flag，见同日 GPU_STATUS 19:27 节）。
kill 掉 = 训练跑完无人接手 = 静默丢掉整轮下游数字。

**How to apply**：
- 认一个在跑的 job 的实际配置：`tr '\0' '\n' < /proc/<pid>/environ`，
  以及 `tr -d '\0' < /proc/<pid>/cmdline`。**grep 环境时不要只 grep 自己想到的变量名**——
  我第一次只 grep 了 `ARM|ROOT|CKPT|TASKS|...`，恰好漏掉 `TRAIN_LOG`，才误判。宁可先整份打印出来看。
- 脚本自己在启动时 `note` 出来的那行（本例 `train_node=... train_log=...`）是最省事的权威，**先 tail 它**。
- 看到 `${VAR:-default}` 就当作「这里有 override 入口」，去找 rearm/launch wrapper。
  同族教训见 [[continue-agent-with-sendmessage-not-agent]]（「说 job 需人工 kill 前先读它的 wrapper」）。
- **自相矛盾的证据要当线索用**：如果按我的解读那个阈值早该触发了，那多半是我的解读错，不是系统错。

与 [[one-sample-is-not-a-trend-or-state]] 同源：都是把「我这层观察到的东西」当成「我想知道的东西」。
另一变体见同日的 `--finalize_lm_eval`：**flag 被传进去 ≠ 那段代码会被走到**；
这里是 **源码写了默认 ≠ 这次用的是默认**。
