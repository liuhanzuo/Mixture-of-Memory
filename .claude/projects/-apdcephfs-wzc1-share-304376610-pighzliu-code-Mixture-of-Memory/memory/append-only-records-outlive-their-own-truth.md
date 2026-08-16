---
name: append-only-records-outlive-their-own-truth
description: "我照 B08 STATUS.json 写「RELATED_WORK.md 不存在」派活, 但它 08-15 就在盘上且同一文件里已有自我更正; 全仓 17 个 proposal 有 8 处这种过期缺席断言"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**append-only 记录里的「X 不存在」是一个写下时为真的时间戳，不是当前事实。派活前必须 `ls` 一次；同一个文件里往往已经有更正，只是在更后面。**

**Why:** 2026-08-16 我派 B08 novelty agent，prompt 写着 `RELATED_WORK.md` 缺失、需要新建。
它**08-15 就已提交**（`463dca4`，当时 39,604 B，现 59,799 B），本轮 agent 只**追加**了 §12
（122 insertions, **0 deletions**）。agent 花了 33 分钟里的一部分去推翻我的前提。

我不是凭空编的 —— B08 自己的 `STATUS.json` **三处**写着
「`RELATED_WORK.md (leg-1-only)` does not exist」。但**同一个文件里还写着**：

> 「blocker 1 said RELATED_WORK.md does not exist. **IT DOES.** Written 2026-08-15 07:14 …」

append-only 的代价就是：**为真时写下的 sentinel 会比真相活得更久**，而先命中的那条通常是旧的。

**扫全仓发现这是系统性的，不是 B08 的个例** —— 17 个 proposal 里 **8 处**过期缺席断言：
A01(`SOURCES.md`)、B01、B03(`RELATED_WORK.md` + `GATE_PREREGISTRATION.md`)、B06、B07、B08、B12。
每一处都是下一个 agent 的绊线。

**How to apply:**
- **派活前对 prompt 里每一个「不存在 / 缺失 / 尚未」都跑一次 `ls`。** 一条命令的成本，
  对面是一个 agent 的半小时。
- **读 append-only 记录要读到底再判**，不能停在第一个命中 —— 更正通常在更后面。
  搜 `grep -n` 全部命中而不是 `grep -m1`。
- **修法是追加带日期的新 key 覆盖旧的，不是改旧句子** —— 历史本身就是证据。
  我写了 `proposal/check_stale_absence_claims.py` 只**报告**不改写。
- **「文件在」≠「文件够」**：这 8 处里有几个确实还卡在文件**内容**质量上，
  guard 只说「别再声称它缺失」。
- 同族：[[read-what-the-consumer-reads-not-the-bare-key]]（上次是**裸 key** 过期，
  这次是**正文散句**过期，同一个病换了外衣）、
  [[a-gate-that-says-never-run-may-already-have-run]]（gate 写「从未执行」而当天已 PASS）。
  三条共同点：**我把一份记录当成了当前状态。**
- 附带教训：我给这个 guard 做第二个负控制时，用 `'B08' in line` 抓行，
  结果把**失败横幅里提到的 B08** 当成数据行，误报了一次 false positive。
  **控制的判据必须锚定行格式（`startswith`），不是关键词出现**——正是这个 checker 要防的同一类松匹配。

---

## ★ 2026-08-17 复发：同一个错，而且 guard 早就写好了

**我又干了一遍，对象是 B06 和 B09。** 派 agent 去「写这两个缺失的 `RELATED_WORK.md`」——
两个文件 **08-15 就在盘上**（28,761 B / 33,942 B，commit `463dca4` / `6d3db4f`），
agent 花了一部分 run 去推翻我的前提，并正确地拒绝覆盖。

**本条 memory 上面第 24 行就点名了 B06。我读了这条 memory，然后没照它做。**
所以「派活前 `ls` 一次」这条**指令强度不够** —— 它依赖我每次记得。

### 根因有两层，都比「忘了 ls」更具体

1. **我把 reader 的输出当成了文件系统的事实。** 队列打印
   `novelty gate not adjudicated (absent)`，那个 `(absent)` 是
   `rec["novelty_evidence"]`，在 `ready_queue.py:704` **早于任何文件检查**就被设成
   `"absent"`，含义是「STATUS.json 里没有 verdict **key**」。文件缺失是**另一条**
   `problems` 行（`:736`），而它**没有被打印**。
   → **`(absent)` 说的是 key，不是 file。**
   更糟：reader 的 docstring 往上六行就用粗体写着
   「among the proposals this tool actually reports on, the count of missing
   RELATED_WORK.md is **ZERO**」，而且是 **08-16 就更正好的**。我读了输出，没读 docstring。
   同族 → [[read-what-the-consumer-reads-not-the-bare-key]]、
   [[a-green-checker-covers-only-what-it-targets]]。

2. **`proposal/check_stale_absence_claims.py` 早就存在、判定正确、会精确抓到这两个**
   （连字节数一起打出来），它自己的输出还写着
   「each row above tells the next agent to produce a file that exists」。
   **我一次都没跑。**
   → **只有人记得去拉才会响的 tripwire 不是 tripwire。**

### How to apply（强化上面那条「派活前 ls」）

- **把 guard 接到「我每轮一定会跑的那个工具」的输出里，不要留成 opt-in 脚本。**
  已做（commit `cc07b1e`）：`ready_queue.py` 现在每次都调 `check_stale_absence_claims.py`，
  在 SUMMARY 后打印 `⚠ N STALE ABSENCE ASSERTION(S)`。
  **advisory-only**：不改 exit code、不改 lifecycle（presence ≠ sufficiency，
  真正卡在文件**内容**上的 blocker 依然有效，只是别再说文件缺失）。
  解析**只认 checker 自己的 `stale absence assertions: N` 行**，
  找不到该行就**报警而不是当 0**（fail-loud）；4 个控制：live 触发 / rc 仍 0 /
  NC1 报 0 时静默 / NC2 无计数行时报警且不谎称 stale。
- **agent 说「你的前提是错的」时，先自己核实再接受**——这次 agent 是对的，
  核实成本只是一条 `ls` + 一条 `git log`，而结论会改变整轮的记账。
- **checker 从 8 条降到 1 条**，正是「追加带日期的新 key」这个处方在起作用；
  剩下那 1 条（B09）是新 key 里合法提到文件名，不是过期断言。

