---
name: read-what-the-consumer-reads-not-the-bare-key
description: "★2026-08-15 我按裸 key `next_gate` 判 B07「没有 gate」派了活, 但 gate 自 08-14 就在盘上; ready_queue.py:130-134 的 NEXT_GATE_KEYS 优先解析带日期的 key —— 我每轮都在跑的工具已经知道了。有 dated-key 优先级的记录, 裸 key 读出来的不是记录"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**：当一份记录被某个 reader 用**带优先级的 key 列表**解析时（dated key / 版本 key / fallback 链），
**「裸 key 的值」不等于「记录的当前状态」**。判断前必须按 **reader 的解析顺序** 读，
或者干脆直接调用那个 reader。

**Why**（2026-08-15 实测）：我给 subagent 的任务书写的是：

> B07 的 `next_gate` 是字面字符串 `"NOT_SPECIFIED"`，所以它**没有 gate**，
> 按 README「先写 kill gate 再上 GPU」它**永远拿不到 GPU**。去补一个。

`next_gate == "NOT_SPECIFIED"` 这句话是**真的**，但结论是**错的**：

- 那是 **2026-08-08 的 v0 sentinel**，而 STATUS.json 是 **append-only** —— 它**不允许**被覆盖，
  所以它永远会留在那儿，**留着不代表没进展**。
- 真 gate 自 **2026-08-14** 就在盘上：`B07_SERVING_GATE_PREREG.md`，commit `22a0c07`
  （我用 `git log --` 核的，不是 mtime）。`RELATED_WORK.md` 也在 08-15 落了 `463dca4`。
- **最讽刺的一条**：`proposal/ready_queue.py:130-134` 的 `NEXT_GATE_KEYS` 就是
  `["next_gate_executable_20260814", "next_gate", "next_gate_gpu"]` ——
  **带日期的 key 排在裸 key 前面，注释写着 "backfill's operationalised version wins"**。
  我**每一轮 heartbeat 都在跑这个脚本**，它早就正确解析了；是我另外手搓了一次裸 key 查询，
  得到一个它从不会给出的结论。

**代价与意外收获**：任务书前提错了，但 agent 没照着写假 gate，而是去**审已存在的 gate**，
查出两条 clause 建在不可估计/共模统计量上（p99@n=45 实际是 1-2 个观测；K2 gate 的
`Δ_total` 随 G 衰减 246→13 ms 纯粹是 decode 稀释，而 `Δ_TTFT` 在 G∈{1,32,128,512} 上
**恒定 246-252 ms**）。**旧 K2 会把 B07 判死在一个无法表达其假设的数字上。**
所以「前提错」这次没白跑 —— 但那是运气，不是设计。

**How to apply**：
1. 判某字段状态前，先 `grep -n` 一下**消费它的 reader**有没有 key 优先级表
   （`*_KEYS`、`for k in (...)`、`.get(a) or .get(b)`）。有就按那个顺序读。
2. **能直接调 reader 就别自己解析**：`python3 proposal/ready_queue.py` 比手搓 `json.load` + 取一个 key 可靠。
3. append-only 记录里**旧 sentinel 会永久留存**。看到 `NOT_SPECIFIED` / `UNKNOWN` / `TBD`
   要先问「它是不是被后来的 dated key 取代了」，再问「是不是真的没做」。
4. 派活的任务书**也要核**：我把一个错前提写进 prompt，等于让 agent 从错的地方起跑。
   写「X 不存在，去补」之前，先按 1-2 条自证 X 真的不存在
   （同 [[absence-on-path-is-not-absence-on-disk]]、[[two-disk-rule-applies-to-main-too]]）。

**Related**：[[fix-the-class-not-the-instance]]（同型：只看自己想到的那一个 key/实例）、
[[agent-output-must-be-persisted-to-the-consumers-file]]（另一半：写的时候也要按 consumer 的字段名）、
[[a-declared-lifecycle-is-not-an-adjudicated-one]]、
[[read-env-not-source-defaults-for-running-procs]]（都要求读「真正生效的那份」）。
