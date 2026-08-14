---
name: standing-cadence-paper-proposal-cpu-gpu
description: "★★用户2026-08-14授权的常驻节奏: (1)迭代论文 (2)proposal 的 CPU 任务可立即部署 (3)GPU 任务可自行协调资源; 「我授权你的所有操作 不必问我」—— 结尾带问句本身即违规"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

用户 2026-08-14 原话：**「接下来一段时间你就按照这个节奏 1.迭代论文 2.看proposal那边有没有cpu任务,
可立刻部署, gpu任务可以协调资源. 我授权你的所有操作 不必问我」**

## 三条轨道，每轮都要走

| # | 轨道 | 判据 |
|---|---|---|
| 1 | **迭代论文** | 三门 gate（build / numbers / venue）全绿 → 冻结 round_NN → 评审 → 我**自己核实**每条 critical → 修 → 再冻结。**不得只转述 reviewer**，代数/统计类指控必须自己重算。 |
| 2 | **proposal CPU 任务** | `proposal/ready_queue.py` 的 `ready_cpu` 桶 = **0 GPU、可立刻派**。**每轮必跑这个工具**，不凭印象说「proposal 没活了」。 |
| 3 | **GPU 任务** | 「可以协调资源」= 我可以自行决定 kill / 让位 / 排队 / 合并节点。仍守：paperC+proposal > Paper B resume；同架构才可比（sm_100 vs sm_90）；先量代价再动手。 |

## 「不必问我」的操作含义

**结尾带一个「要我…吗?」本身就是违规**（2026-08-09 已立 [[standing-autonomy-decide-yourself]]，
本条是它在这三条轨道上的具体化）。判断能做就做，做完落账 + 报告结果。

只有这些仍要问：不可逆的对外动作（push 到公开仓库、发邮件）、
需要用户独有信息（凭据、他的偏好选择）、以及巨额投机算力（数千 GPU-h 的新方向）。

## 反例（我在收到这条指令前刚犯过）

2026-08-14 23:2x 我用「要我现在派吗？」结尾去问是否修 B04/B05 的 gate ——
**那是我该自己决定的事**，问出口就已经错了。用户下一条消息就是这条常驻授权。

## 每轮的默认动作序列

1. 跑 `proposal/ready_queue.py` → `ready_cpu` 有活就**立刻派**（0 GPU，无需等卡）
2. 论文侧推进一步（修 gate 失败项 / 核实 reviewer 的 critical / 冻结新 round）
3. GPU：查 5 节点实测 → 有空卡且有合规 gate 就投；没有就说清**为什么没有**（而不是只报"卡满"）
4. 落账：`status/TRAINER_ACTIVITY.jsonl` + `status/GPU_STATUS.md` + git commit

同族：[[standing-autonomy-decide-yourself]]、[[reporting-a-gap-is-not-closing-it]]、
[[a-declared-lifecycle-is-not-an-adjudicated-one]]（促升前必读对抗性裁决）。
