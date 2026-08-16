---
name: reporting-a-gap-is-not-closing-it
description: ★发现协议缺口/待办后不要只在 heartbeat 里当 caveat 反复报; 能当轮派 subagent 就当轮派 — 且派出去之后往往发现缺口比自己以为的严重得多
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**2026-08-13：我连续两个 heartbeat 把同一个协议缺口写成 caveat 报给用户，一次都没派人修。**

缺口：`scripts/_run_sparseforge_tokenmatched.sh:294` 的 in-run eval 只有 7 个 task
（无 boolq / rte），而 union-9 表口径是 9 个 → #246 两个 arm 的数字无法进表。
我两轮都写了「known gap still open」，然后什么也没做。

第三轮派出去（0 GPU，只写脚本 + 干跑验证），**agent 立刻发现缺口比我以为的严重得多**：

> `--finalize_lm_eval True` 在 `FINAL_FT=0` 下是**死 flag**。gate 在
> `main_llama.py:2248` 要求 `finalization_done`，而它只在 `:3215`（`iter_num > max_iters` 之后）
> 置 True；`final_finetune_iters=0` 时 `:3467-3470` 的 else 直接 `break` 出 `while True`，
> 再也不回到 `:2102` 的 eval 块 → **一个 zero-shot 数字都不会有，不是少两个。**

也就是说：那不是「补两个 task」，那个 watcher 是这两个 arm **唯一**的 zero-shot / PPL /
2:4 verify 来源。**不派的话，两台机 ~92 GPU-h × 8 卡跑完会一张表都没有** ——
正是 #114 的失败模式（标记 completed，两盘零 ckpt）。

## Why

「报告缺口」在感觉上像是尽责了（它出现在报告里、用户看得见），但**缺口不会因为被提及而变窄**。
而且我对缺口严重性的估计本身就是未经检验的猜测 —— 只有真去查代码才知道是「少 2 个」还是「全没有」。

## How to apply

- heartbeat 发现缺口 → 判断「能不能当轮派」。**0 GPU 的（写脚本 / 查代码 / 干跑验证）一律当轮派**，
  不要等卡空、不要等下一轮。
- **同一个缺口不允许在两个 heartbeat 里以 caveat 形式重复出现**。第二次看见它，
  要么派掉，要么写进 `status/PENDING_TASKS.md` 并注明为什么现在不能做。
- 派的时候把自己的前提**标成前提**（「我认为是少 boolq+rte」），并明确要求
  「如果发现我说错了就停下来报告」—— 这次正是这句话让 agent 纠正了我而不是照做。
  见 [[read-the-trainer-docstring-before-designing-a-control]]。
- 训练在跑、卡满 ≠ 没事可做。**0 GPU 的收口工作永远可以并行推进。**
- 相关：[[standing-autonomy-decide-yourself]]（用户已授权自主投任务，不必等拍板）、
  [[long-running-subagents-stall-silently]]（但派出去要盯 artefact 时间戳，
  ⚠️ 注意「读代码期间的沉默」和「卡死的沉默」不同 —— 这次 agent 静默 50 分钟正是在追控制流）。
