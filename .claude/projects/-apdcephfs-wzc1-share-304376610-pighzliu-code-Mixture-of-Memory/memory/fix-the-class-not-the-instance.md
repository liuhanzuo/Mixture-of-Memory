---
name: fix-the-class-not-the-instance
description: "★修好一处嵌套查找 bug 后必须问『同一个 reader 还有哪些 key 也只扫顶层』——我为 gate 加了 NESTED_GATE_CONTAINERS 却没管 BLOCK_KEYS, 三天后 A02 的嵌套 gpu_policy 让一个 CLOSED 提案报出 ready_gpu"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

2026-08-15：`proposal/ready_queue.py` 的 `_walk_blockers` **只扫顶层** `BLOCK_KEYS`，
于是 A02 的 `disposition_2026_08_12.gpu_policy`（原文「**NO further A02 GPU**. Resurrection
requires a NEW MECHANISM」）对调度器**不可见** → A02 报 `ready_gpu`，
调度器把一个**它自己记录里已关闭 GPU 的方向**当成可投任务递给我。

## 为什么这是「同一个 bug 的第二次」

几天前我**已经**修过同一个 reader 的同类缺陷：`KILL_KEYS` + `_first` 只看顶层，
导致 A04 那个写好的三条款 kill gate（在 `gate_design.kill_condition_verbatim`）被报成
"no kill_gate field"。当时我加了 `NESTED_GATE_CONTAINERS` 白名单——
**只修了 gate，没问「这个 reader 还有哪些 key 族也只扫顶层」**。
`BLOCK_KEYS` 就在同一个文件里，隔几十行。

## 判据与修法

- **两个方向都要实测**，别只测「修好了」：
  - 只留嵌套子句 + `RELATED_WORK.md` 存在 → `1 ready_gpu`（bug 复现）
  - 同一份文件把 policy 也抄到顶层 → `0 ready_gpu, blocker STILL LIVE`
  - 修复后只留嵌套子句 → `0 ready_gpu`，且 blocker 路径打印为真实位置
    `[disposition_2026_08_12.gpu_policy]`
- **用显式一层白名单，绝不盲目深走**。深走会把「散文里提到某个 blocker」读成
  真的 blocker —— 那是 over-report 失效模式，和 under-report 一样会卡死工作。
  白名单前缀匹配（`disposition` 覆盖 `disposition_2026_08_12`），因为
  **带日期的 wrapper 正是收尾裁决最自然的落点**，属常见情形而非异常。
- **回归测试必须对旧代码 FAIL**：新增 (j)/(k) 后，旧 `ready_queue.py` 得 43/45 FAIL，
  新版 45/45 PASS。只在新代码上 PASS 的测试证明不了任何事（同族教训见
  [[a-declared-lifecycle-is-not-an-adjudicated-one]] 里「reverse-test 用 HEAD 是无意义的」）。

## 另一半：不要照抄 agent 的因果归因

报这个 bug 的 agent 说「**只**加 `related_work_status` 就把 A02 从 ready_cpu 翻成 ready_gpu」。
我拿 committed 版 STATUS.json 反向测：**仍是 0 ready_gpu**——因为 `RELATED_WORK.md` 缺失
本身就独立扣着它。翻转需要**两个条件同时**满足，而真正的缺陷是那个隐形 blocker。
**agent 找到真 bug ≠ agent 的因果链正确**；发现值得采纳，归因要自己复算。

## How to apply

修完任何「查找/解析只覆盖了一层」的缺陷后，**立刻 grep 同一个 reader 里所有同类 key 族**
（`*_KEYS`、`*_CONTAINERS`、`_first*`、`_walk*`），问「这一族也只扫顶层吗」。
一个 schema reader 里的嵌套盲区几乎从不孤立存在。

同族：[[agent-output-must-be-persisted-to-the-consumers-file]]（派活前先读消费者的字段名）、
[[a-declared-lifecycle-is-not-an-adjudicated-one]]。
