---
name: rank-local-counters-and-gated-postfix-fake-failures
description: "★★我把 GATE0 误判为 FAILED 两条都错: nm_2_4_tile_stats 的 processed/skipped 是 all_reduce 前的 rank-local 计数(判据是同行 total_tiles==elems/4), tqdm postfix 被 output_flip_every 门控所以「loss 20 步不动」是一个 iter-0 值重画 24 次"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

2026-08-15：我把 ALPS+SLoRB 的 GATE0 判成 **FAILED** 并写进 `status/` + 报给用户。
**两条「失败信号」全是探针自身配置的伪影**，subagent 反驳后我复核，**撤回自己的结论**。

## 伪影 1：rank-local 计数器 ≠ 全局覆盖率

日志：`sparse_linear_count: 224, processed: 96, skipped 128 (shape=torch.Size([0]))`。
我算出 96/224 = 42.9%，判定「mask 只覆盖不到一半」。

**错。** `processed`/`skipped` 是 **FSDP `all_reduce` 之前**的 rank-0 本地累加；
`torch.Size([0])` 只表示该 shard 在本 rank 上没有份额。

**决定性判据在同一行**：`total_tiles = 1619001344.0`，而 `mask_validation.json` 的
`elems/4 = 6476005376/4 = 1619001344.0` —— **精确相等**，说明 tile 统计覆盖的是**整个模型**。
若真只覆盖 96/224，reduce 后的总数应约 `693857719`。
⇒ **同一行里就有能自证的全局量，我却只读了那个局部量。**

## 伪影 2：tqdm postfix 被 `output_flip_every` 门控

`loss=27.9` 在 20 步 24 次读数里**字节相同**，我判定「梯度没到权重」。

**错。** `main_llama.py:2907/2913/2917` 三处都是
`if iter_num % args.output_flip_every == 0:` 才刷新 postfix，而我派的探针传了
`--output_flip_every 1000000` ⇒ **一个 iter-0 的值被重画了 24 次**。
改 `=1` 重跑：`27.9 → 26.9 → 26.6 → … → 24.0` **单调下降**。

## 伪影 3：对蒸馏 loss 套 PPL 规则

我拿 `exp(27.9)=1.3e12` 去套 CLAUDE.md 的「PPL > 1000 = 模型被污染」。
但 `main_llama.py:2718` 是 `loss = hardness_task*task_loss + hardness_kldiv*kl_loss`
—— **蒸馏目标，不是交叉熵**，`exp()` 它没有 perplexity 含义。
（该臂从 ALPS-pruned 起步、teacher 是 dense，所以 4·KL 项本来就大。）
另外 `flip_ratio=0` 在 `change_mask=False` 下是**正确**的，出现 flip 才是 bug。

## How to apply

- **报 gate 失败前，先问「这个数是 per-rank 还是 global」。** 分布式日志里
  `processed/skipped/count` 这类计数器默认怀疑是 rank-local；找同一行/同一份 JSON 里
  **能交叉验证的全局量**（这里是 `total_tiles` vs `elems/4`）。
- **进度条上的量可能是被采样/门控的。** 判「有没有在变」要看**产生该数字的代码路径**
  （`% every == 0`），不能只看数字重复。同族坑见
  [[tqdm-elapsed-and-counter-have-different-origins]] —— 同一个 tqdm 行，两次栽在不同字段上。
- **套用阈值规则前先确认量纲**：`loss` 是不是 CE？`exp(loss)` 才是 PPL。
  蒸馏/多项加权 loss 不能用 PPL 判据。
- **让探针自己不制造伪影**：诊断用的 run 要把 `output_flip_every=1`、
  日志粒度开到最细，否则你是在诊断自己的采样配置。
- 我的 FAILED 结论**已发布给用户**才被推翻 —— 撤回时要在原文件加 banner 并保留原文
  （`status/ALPS_SLORB_GATE0_FAILED.md` 的 `⛔ SUPERSEDED` 头），不要静默改写。

同族：[[one-sample-is-not-a-trend-or-state]]、[[read-env-not-source-defaults-for-running-procs]]。
统一形式仍是：**我观察到的那一层 ≠ 我想知道的那一层。**
