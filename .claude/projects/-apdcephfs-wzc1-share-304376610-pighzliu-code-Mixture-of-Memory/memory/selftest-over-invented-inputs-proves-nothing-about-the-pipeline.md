---
name: selftest-over-invented-inputs-proves-nothing-about-the-pipeline
description: "B04 的 --selftest 从 08-14 起一直通过, 因为它喂手写向量; 真实读盘路径根本没有代码。加了 fixture selftest 后首跑就抓到 str/int key bug —— 它不崩, 只是同时打印「找到 X」和「全缺」"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**一个只跑手写输入的 selftest，证明的是「函数可 falsify」，对「流水线能不能跑」一字未答。**

**Why:** 2026-08-16 的 B04（eval-fragility）实例最清楚：
- `phi_budget` 有且只有 3 个调用点，**全在 `selftest_phi()` 里**，喂的是手写 y 向量。`--selftest` 在 08-14、08-15、08-16 都通过。
- 但**从盘上读到的任何东西都从不进入 `phi_budget`** —— 也就是说 gate 从未能对真实数据求值，而 selftest 每天报绿。
- 更细的一条：analyzer **确实**读 read-out 目录（开 `per_example_*.jsonl`、数 `shard*of8.json`、读 `summary.json`）用于 donor margin 和完整性检查。所以「它根本不读目录」是**说过头**；准确说法是「读了，但读到的不进入决策统计量」。这个精度差别决定了修法。

**加上一个走真实读盘路径的 fixture selftest 后，首跑就抓到一个真 bug**：resolved-census 的 key 是 `str` 而 grid 查找用 `int`。它**不崩**，而是**同时打印「找到 step200000」和「5/5 全缺」** —— 自相矛盾却看起来正常。纯算术 selftest 抓不到，因为算术从没碰过磁盘的 key。修法不是统一 key，而是**显式分离两个用途**并留注释（int 供查找、JSON/打印侧现场转 str）。

**How to apply:**
- 任何 gate/判定代码，**除算术 selftest 外必须有一个走真实 I/O 路径的 fixture selftest**（`mkdtemp` 建合成目录树，`finally` 里 `rmtree`，**绝不 symlink 到 live 证据目录** —— 见 [[repo-checkers-are-writers-not-probes]]）。
- fixture 要注入**看起来完整的坏数据**，不只是明显坏的：5/8 分片但 `summary.n_shards=5`（静默 partial merge）、`per_example` 被截断而 summary 干净、名字对但 ckpt 错、`ckpt_step` 不匹配。
- **「selftest 通过」不得作为「gate 可运行」的证据**写进任何 STATUS/报告。见 [[a-declared-lifecycle-is-not-an-adjudicated-one]]。
- 判 abort 要看**退出码**：这次 abort 从 exit 0 改成 exit 3。我第一次核时 `| tail` 吃掉了退出码得到 EXIT=0，差点误报 —— 管道会掩盖 exit code，要 `cmd > file; echo $?`。

见 [[hand-composed-demo-strings-must-be-executed]]、[[rank-local-counters-and-gated-postfix-fake-failures]]。
