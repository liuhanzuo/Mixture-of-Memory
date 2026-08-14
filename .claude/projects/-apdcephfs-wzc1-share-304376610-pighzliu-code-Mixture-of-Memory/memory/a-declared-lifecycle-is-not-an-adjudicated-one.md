---
name: a-declared-lifecycle-is-not-an-adjudicated-one
description: "★agent 自己往 STATUS.json 写 lifecycle:ready_gpu ≠ 它的 gate 通过了审查; 2026-08-14 我据此报「ready_gpu 0→2、Persist 修复生效」两处皆错——文件是 agent 22:36/22:49 自己写的, 而 6/6 对抗性 lens 判 NEEDS_REVISION、2 条判 gate 根本不可判定"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

一个 agent **声明**的状态和一个**被裁决过**的状态是两回事。
`lifecycle: ready_gpu` 出现在 `STATUS.json` 里，只说明**有人写了它**，不说明它**通过了 gate 审查**。

**2026-08-14 实测（我连错两处）**：

1. 我报「`ready_gpu` 0→2，是我 22:04 加的 Persist 阶段生效了」——**两半都错**：
   - `stat` 显示 B04 的 STATUS.json 改于 **22:36**、B05 改于 **22:49**，都**早于** workflow 结束；
   - 我的 Persist 阶段**只 emit 一份 patch**（沙箱不能写文件），**我从没 apply 过**。
   ⇒ 是**起草 agent 自己写的**。我把「我做了一个修复」和「状态变好了」两件事**凭时间接近就归了因**。
2. workflow 自己的返回值直接反驳了促升：
   `n_gates_survived_adversarial: 0`、`would_become_ready_gpu: []`，且 B04/B05 都在 `stays_ready_cpu` 里。
   逐条看：**6/6 对抗性 lens 全判 `NEEDS_REVISION`**，其中**两条判 gate 根本 `decidable=False`**
   —— B04 的 `phi` 用了 rescale span **116500**，而 read-out 自己的 span 是 **175000**（统计量不是被测量）；
   B05 的 prereg **完全没有 CELL→RUNG 映射规则**（4 个 cell 不一致时 rung 的相位标签无定义）。

⇒ **agent 在自己的 gate 被驳倒之前就把 `ready_gpu` 写进了盘。** `ready_queue.py` 如实读盘、如实上报，
它没错；错的是我**只读了聚合计数（0→2）就当好消息发布**，没去读同一份返回值里的裁决。

**Why**：这正是 `ready_queue.py` 文件头自己警告的那个坑 ——
「*Filling in a proposal's paperwork made a killed direction look like the single most dispatchable
item in the queue, and it would have been handed 8 idle H20s*」。
我上一轮甚至**引用了这句话**来「核实这不是刷文书刷出来的」，却只核了「有没有数字化阈值 + seed」，
**没核那些阈值有没有通过审查**。有阈值 ≠ 阈值正确。

**How to apply**：
- 促升到 `ready_gpu` 前，**必须**去读该 gate 的**对抗性裁决**，不是只看它字段齐不齐。
  判据：**所有 lens 都 `SOUND`**；任何一条 `NEEDS_REVISION` / `REFUTED` → 留在 `ready_cpu`。
- **`decidable=False` 是一票否决**，哪怕阈值写得很像样（B04 有精确到小数点后 6 位的常数，照样不可判定）。
- 报「某个指标变好了」之前先问：**是我做的动作让它变好的吗？** 用 `stat -c %y` + `git log` 对时间线，
  别用「我刚改了 X，然后 Y 变了」推因果。见 [[agent-output-must-be-persisted-to-the-consumers-file]]
  —— 那次我漏了落盘这一步，这次我又把**别人**的落盘当成了**我的**修复生效。
- 落账要写清**权威来源**：我把更正写进 `lifecycle_corrected_by_MAIN_20260814.authority =
  "adversarial verdicts in the workflow result, not MAIN's opinion"`，这样下一个 agent 不会以为
  是我拍脑袋降级的。

同族：[[reporting-a-gap-is-not-closing-it]]、[[one-sample-is-not-a-trend-or-state]]。
统一形式仍是那句：**我观察到的那一层 ≠ 我想知道的那一层** ——
「盘上写着 ready_gpu」是一层，「这个 gate 站得住」是另一层。
