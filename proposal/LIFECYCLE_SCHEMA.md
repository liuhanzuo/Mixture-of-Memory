# proposal/ 生命周期 schema

> **为什么存在。** 2026-08-14 一天之内，我（MAIN）因为「调度器读不懂队列」造成了五类事故：
>
> | 事故 | 真实原因 |
> |---|---|
> | B02 被我判成「blocked，无事可做」 | `status` 字符串描述的是**旧路径被证伪**，不是没活。它的 `next_gate` 写得很具体（固定样本集重跑 j-sweep，1 GPU × 8 configs） |
> | B11 的 K1（**0 GPU 且 blocking**）躺着没人跑 | 它自己 `gpu_policy` 就写着「NO GPU until K1 passes」——**信息在，但没进调度视野** |
> | B06 一个 `p=2.6e-67` 的已确认结果被我归进「薄 backlog」 | PROPOSAL.md 写得很清楚，**STATUS.json 里只剩 `backlog_confirmed_seed` 三个词** |
> | paperC 读出缺陷我连报三轮「待办且有时效」 | 08-13 就做完了（commit `078e464`）。**我凭记忆报的，一次没查盘** |
> | 16 张 H20 被我记成「给 union-9 预留」 | 那任务的归档 arm 是 `sm_100`，**H20 从一开始就不合法**——预留了个架构不对的节点 |
>
> 共同结构：**信息存在于 PROPOSAL.md / git / 代码里，但调度器的输入格式没有承载调度决策。** 于是调度器只能猜，或者跳过。
>
> 本 schema 要修的不是「更勤快地扫 backlog」，而是**让「现在能跑什么」变成一个可计算的属性**。

---

## 0. 迁移原则：只加不改

现状实测（2026-08-14）：**15 个 proposal，`status` 字段有 14 个不同取值**，其中 A03 那个是**约 900 字的散文段落**。

那些散文**有科学价值**（A03 那段记着 seed45 落地 NOT-CONFIRM、0/3 CONFIRM、目录已物理迁移、zwfy6 侧仍在旧路径）——**不得删除**。

所以：

- **新增 `lifecycle` 字段**承载机器可读状态；
- **`status` 原样保留**为人类可读注记，永不覆盖；
- 一切修改遵守 **append-only**（JSON 唯一允许的字节变动是把收尾 `}` 变成 `,`）。

---

## 1. `lifecycle` —— 唯一的机器可读状态

**⛔ `blocked` 这个词在本 schema 中被禁用。** 它在 2026-08-14 把三件不同的事混成了一个字符串，直接导致上表第 1、2 行的事故。

| `lifecycle` | 含义 | 调度含义 |
|---|---|---|
| `ready_gpu` | 已完全指定，**只缺卡** | 空卡时**可直接投** |
| `ready_cpu` | 已完全指定，**不需要卡** | **随时可派，不占卡额度** |
| `needs_prior_gate` | 前面有 gate 挡着 | 看 `prior_gate_needs_gpu` 决定它是否其实立刻可派 |
| `running` | 正在跑 | 不重复投；须带 `running_on` |
| `promoted` | 已晋升 `paper<X>/` | 不再进 ready 队列 |
| `dead` | 已证伪 / 已关闭 | 移入 `proposal/archive/`，claim 不自动复活 |

### 1.1 `needs_prior_gate` 必须带 `prior_gate_needs_gpu`

```json
"lifecycle": "needs_prior_gate",
"prior_gate": "K1 novelty check",
"prior_gate_needs_gpu": false
```

**这个布尔值是 B11 事故的直接修复。** `prior_gate_needs_gpu: false` 的项，生成器必须把它**归入 `ready_cpu` 一并输出**——一个零成本的 blocking gate 躺着不跑，是纯粹的浪费。

---

## 2. 必填字段（缺了就报错，**不许静默略过**）

> B06 消失就是静默略过的结果。所以生成器遇到缺字段**必须对那个 proposal 报错**，而不是跳过它继续。

| 字段 | 规则 | 消费者 |
|---|---|---|
| `lifecycle` | 上表枚举之一 | ready 队列生成器 |
| `next_gate` | **可执行**的一句。写不出来就写 `"NOT_SPECIFIED — <为什么>"`，**不许编** | 生成器 / 我 |
| `gpu_cost_estimate` | 数值 + **估算依据**（哪个已有 run 的实测速率外推）。估不出写 `"UNKNOWN — 需先做 1-cell 计时"` | 调度排序 / 周摘要 |
| `needs_arch` | `sm_100` / `sm_90` / `any` | **admissibility**（见 §3） |
| `kill_gate` | 有就照搬；**没有必须显式写** `"NO_KILL_GATE_DEFINED"` 或 `"NO_KILL_GATE_BY_DESIGN — <理由>"`，**不许留空** | 晋升检查 |
| `novelty_checked` | `true` / `false` | 晋升检查 |

**没有消费者的字段不许存在。** 否则它就是下一个 `backlog_confirmed_seed`——存在但没人读，等于不存在。

---

## 3. admissibility 必须由机器判定，不靠人记

2026-08-14 的教训：**同 harness 复现必须同架构**，否则 stack drift 和 hardware drift 混在一起，连 FAIL 都无法解释。当天 union-9 对照被迫留在 `sm_100`（LOCAL/`.212`），因为全部 37 个归档 `results_*.json` 都是 `sm_100`；而我却把 16 张 `sm_90` 的 H20 记作「为它预留」。

所以一个 gate 可投的判据是**三者同时成立**：

1. `lifecycle == ready_gpu`
2. 有空节点满足 `needs_arch`（`sm_100` = LOCAL/`.212`；`sm_90` = `.73`/`.82`/`.104`）
3. 该节点**没有其他 agent 占用**（两个 agent 同占一节点在 2026-08-08 毁过 4/5 rung）

「空闲」本身的判据见 `.claude/commands/heartbeat.md` Step 2：`memory.used`≈0 **且** `utilization.gpu`=0 **且** `--query-compute-apps` 里该卡无 PID，三者缺一不可。

---

## 4. 时效：deadline 挂在事件上，不挂在日历上

paperC 的读出缺陷是 **pre-hoc-or-never**：必须在 `step121000` 打分**之前**落地，晚了预注册就变成事后辩解。这种 deadline 挂在**另一个 job 的进度**上，日历表达不了。

```json
"deadline": { "before_event": "paperC_qwen3base_heal_k8f2@step121000" }
```

生成器须把**逼近中**的这类项顶到队首。

---

## 5. 空卡必须是带原因的告警

规则：**空卡 + ready 队列非空 = 要么投，要么把「为什么不投」写进机器可读字段。**

2026-08-14 我报「这些卡是给 union-9 预留的」——错了两层（架构不合法 + 那任务并不需要它们），而且**没有任何东西会去核对这句话**。写成字段就可审计。

---

## 6. 晋升：自动执行 + 通知，不设审批

**2026-08-14 用户指令**：「第一 不需要 (a), 告诉我即可,反正想法也算是你提的」。

即：满足门 → **自动建 `paper<X>/` 并通知用户**，不等审批。

### 6.1 门（全部满足才自动晋升）

沿用 `proposal/README.md` 已有规则，此处只是**机械化**它：

1. `kill_gate` 跑过且**没被杀掉**
2. 至少一条经**独立复核**的显著发现（不是「算出了一个数字」）
3. provenance 完整（原始 json/csv 在盘上，**数字可重算**）
4. `novelty_checked == true`，且按**正确家族**核实过 venue

> **venue 核实分两套家族，不可混用**：OpenReview 系（ICLR/NeurIPS/ICML）用 `venueid` + `Camera_Ready_Revision`；**ACL 系（含 Findings）必须 aclanthology + DBLP**。S2/DBLP 对 2026 会议论文常滞后返 arXiv.org，**不可只走 S2**。
> 判据是「**完全相同/抄袭**」才算被占，**「有重叠」不是放弃理由**。

### 6.2 `abandoned` —— 强制字段，这是用户唯一的诚实性抓手

用户**不 review proposal**，只看 paper。这产生一个用户抓不到的失败模式：

> **把 claim 一直收窄到平凡为真，也能通过晋升门。**

A04（判成 BRANCH B）和 B04（general claim 被 Qwen 跨家族复现打死，ρ=+0.43 p=0.42，只剩 OLMo-2-only 的 ρ=±1.00 p=0.0028）这两次收窄是**诚实**的。但同样的动作动机不纯地做，会产出「可发表但空洞」的结果。

所以晋升记录**必须**带：

```json
"abandoned": {
  "original_claim": "<最初想证明什么>",
  "killed_by": "<被哪个实验/证据杀掉>",
  "surviving_scope": "<剩下的确切范围>"
}
```

**这是用户判断收窄是否诚实的唯一抓手。** 缺这一栏 → 晋升检查**必须失败**。

---

## 7. 三层强制（只有第一层是文档）

2026-08-14 的反例：`.claude/commands/heartbeat.md` 是一份写得很详细的流程文件，**过期两个月**（点名的 4 个 log 全不存在、6 个 IP 全下线）我照样每轮「执行」它——因为**静默返回空读起来就像一切正常**。

**文档不会强制自己被遵守。** 所以：

| 层 | 机制 | 强制力来源 |
|---|---|---|
| 1 | 本文件 + `/proposal` skill | 说清规则。**可能被忽略** |
| 2 | `proposal/ready_queue.py` 生成器 | heartbeat **不再手读 15 个目录**，而是读它的输出。我判错 B02/B11 正是因为在手读 |
| 3 | `PostToolUse` hook | 写 `proposal/**/STATUS.json` 后**自动校验 schema**，不合规当场把错误回给我。**不依赖我记得遵守** |
| 4 | CI 兜底 | 每日全量校验 + 输出 ready 队列。即使 hook 被绕过，第二天也会暴露 |

> 现状：`.claude/settings.json` 的 `hooks` 为 `NONE`。第 3 层是**新增**的。

---

## 8. 周摘要（用户要）

**2026-08-14 用户指令**：「第二 需要」。

不是 review proposal，而是一页：

- 本周投了多少 GPU-h
- 几个 gate 通过 / **被杀**
- 几个进 `archive/`
- 当前 ready 队列长度

**`kill 率`必须显式列出。** 若它长期为 0，意味着门形同虚设——这是用户该看见的系统性偏差信号，而不需要知道 B07 具体在干什么。

---

## 9. 待对齐（2026-08-14 未定稿部分）

`next_gate` / `gpu_cost_estimate` / `kill_gate` 的**确切 JSON 形状**待与正在运行的 STATUS-backfill agent（`ac9fce12f0599d821`）实际写入的字段名对齐，**避免让它返工**。届时补进 §2 并落 CI 校验。

字段命名以本仓库既有用法为准（A04 / B10 / B11 的 STATUS.json 粒度最细），**不自创一套**。
