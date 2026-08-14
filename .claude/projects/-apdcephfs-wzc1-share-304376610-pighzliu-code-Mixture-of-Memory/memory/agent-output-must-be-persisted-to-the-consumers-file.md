---
name: agent-output-must-be-persisted-to-the-consumers-file
description: "★workflow 的 return 值不落盘 = 不存在; 2026-08-14 我派 6 个 proposal gate 起草却从没让 agent 写 STATUS.json, 而 ready_queue.py 是读盘判断的, 于是连报两轮「它落地后 ready_gpu 就不是 0 了」——那一步我根本没实现"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

派 agent 产出「要被某个工具消费」的东西时，**必须在 prompt 里显式指定落盘路径和字段名**，
并在 workflow 结束后**自己核实文件真的变了**。`return {...}` 只回到我的上下文，**盘上什么都没发生**。

**2026-08-14 实测**：我派 `wf_c2a3b490-e40` 给 6 个 proposal 补写 kill gate / next_gate，
理由是 `proposal/ready_queue.py` 报 `0 ready_gpu, 12 ready_cpu`。脚本里第 145 行只让 agent **读**
`STATUS.json`，然后 `return { proposals: rows }`。**全程没有一次写文件。**
而 `ready_queue.py` 是**读 `STATUS.json` 的字段**来判 lifecycle 的 →
**gate 写在返回值里，它永远看不到。**

实测证据：那 6 个目录最近改动是 3906s ~ 542857s 前（几小时到 6 天），**没有一个是 workflow 写的**。

**Why**：我因此连着两轮 heartbeat 报「它落地后 `ready_gpu` 应不再是 0」——
那句话建立在一个**我自己从没实现的步骤**上。用户问「ready_cpu 还没搞好么」才暴露。
这是**同一根因的第三次**：
- `--finalize_lm_eval` —— **flag 被传进去 ≠ 那段代码会被走到**
- 源码 `${VAR:-default}` —— **默认值写在源码里 ≠ 这次用的是它**（[[read-env-not-source-defaults-for-running-procs]]）
- 这次 —— **agent 产出了内容 ≠ 内容到了消费者读得到的地方**

统一形式：**我观察到的那一层 ≠ 我想知道的那一层。**

**How to apply**：
- prompt 里写死「**把结果写到 `<确切路径>`，字段名用 `<确切 key>`**」，并要求 agent 回报它写了哪些文件。
- 先去读**消费者**（这里是 `ready_queue.py` 的 `NEXT_GATE_KEYS` / `KILL_KEYS` / `VALID_LC`）
  确认字段名，再决定写什么 —— 否则写了它也不认。
- workflow 返回后**必须实测**：`find <dir> -newermt '<启动时刻>'`，而不是看返回值里的 `n_gates_survived`。
- ⚠️ **本例特有的陷阱，落盘时必须一起处理**：`ready_queue.py` 自己的注释记着
  「*Filling in a proposal's paperwork made a killed direction look like the single most
  dispatchable item in the queue, and it would have been handed 8 idle H20s*」——
  **补完文书会让已死方向看起来最该派活**。所以写 gate 的同时**必须写声明式 `lifecycle`**
  （`dead`/`promoted`/`running` 是权威、不可被文书翻案），否则就是在重演那个坑。

同族：[[reporting-a-gap-is-not-closing-it]]（报缺口不等于补缺口）——
这次更进一步：**派了任务也不等于补缺口，任务的产出得能被消费者读到。**
