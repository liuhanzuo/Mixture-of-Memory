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
