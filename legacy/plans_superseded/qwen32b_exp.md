# Qwen3-32B QCMem 实验记录与 no-thinking 计划

更新：2026-07-16 13:55 GMT+8

## 0. 当前结论摘要

Qwen3-32B 是 thinking 型 instruct 模型。当前已确认：

1. **BABILong qa5 原始 no-chat/raw 结果不可信**：0k 低、32k 高主要是 output-format / thinking artifact。
2. **BABILong qa5 在 `chat_template + enable_thinking=False` 后恢复正常**：n=500 正式表已完成，0k-32k 约 79-89%。
3. **RULER variable_tracking 的异常曲线也有 output-format 成分，但 naive chat-template disable-thinking probe 反而更差**：说明 VT 不能简单套 chat wrapper；需要进一步区分“answer-only raw prompt”和“chat no-thinking prompt”两种口径。
4. **LongBench strict-empty gate 已废弃为正式口径**：之后与 Qwen3-8B 一样，LongBench empty prediction 保留并计 F1=0，不让整 shard invalid。
5. **后续 Qwen3-32B 自动判分 benchmark 的正式口径建议统一为 disable-thinking / answer-only**，但每个 benchmark 要做小样本 sanity，不能盲目套 chat template。

---

## 1. 固定协议（当前 32B 主协议）

| 项 | 值 |
|---|---|
| 模型 | `models/Qwen3-32B`，stock/no-adapter |
| 层数 | 64 |
| resume depth | 当前实测主用 `j=16`（不是计划表早期的 j=21） |
| chunk_size | 512 |
| sink | bos |
| dtype/attn | bf16 / SDPA |
| QCMem selector | BABILong/RULER NIAH: bm25 topk12；RULER VT: iter_bm25 topk16 rounds4 hop4 |
| no-thinking 口径 | `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` |
| 运行环境 | lhz/lhz2/dev4 使用 `/volume/haru/Mixture-of-Memory/.venv_hy3/bin/python`，不要用 `/volume/haru/envs/dllm` |

---

## 2. BABILong 结果

### 2.1 旧 raw/no-chat 正式表（已判为受 thinking 污染）

路径：`babilong_results/qwen32_zerotrain_n500_j16_chunk512`（lhz2）

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa1 | 99.8 | 91.2 | 84.4 | 80.4 | 80.8 | 59.0 | 29.6 |
| qa2 | 69.8 | 59.8 | 52.6 | 48.6 | 38.6 | 28.4 | 10.8 |
| qa5 | 18.2 | 19.4 | 21.4 | 34.4 | 42.4 | 48.4 | 50.8 |

判定：qa5 的 0k 低、32k 高反常；原始输出大量 `To answer this question...`、自问自答、`Answer: Answer...` 复读。此表保留为 diagnostic，不作为 headline。

### 2.2 qa5 no-thinking 正式表（n=500，已完成）

路径：`babilong_results/qwen32_qa5_disablethinking_n500_j16_chunk512/_summary.json`

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa5 disable-thinking | **89.2** | **87.6** | **85.2** | **84.0** | **79.2** | **79.2** | **81.4** |

结论：no-thinking 直接修复 qa5，说明原 qa5 主要是 output protocol artifact。

### 2.3 qa1/qa2 no-thinking 正式表（n=500，进行中）

节点：dev4 `183.242.150.6:32679`，4×H200
脚本：`scripts/_qwen32_qa12_disablethinking_n500_pool.sh`
输出：`babilong_results/qwen32_qa12_disablethinking_n500_j16_chunk512`
日志：`logs/qwen32_qa12_disablethinking_n500/`
状态：2026-07-16 13:55 正在跑，已从 qa1 0k/1k/2k 推进到 qa1 4k。

待办：完成后合并 qa1/qa2 + 已完成 qa5，形成 BABILong disable-thinking 正式表。

---

## 3. RULER 结果

### 3.1 raw/no-chat n=500 正式表

路径：`ruler_results/qwen32_zerotrain_n500_j16_chunk512`

| task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| niah_single_2 | 100.0 | 99.6 | 92.8 | 97.2 | 96.8 |
| niah_multikey_1 | 96.0 | 89.6 | 97.4 | 94.4 | 96.0 |
| variable_tracking | 15.6 | 34.2 | 32.0 | — | — |

说明：VT 按 `Qwen模型结果.md` 只要求 8k/16k/32k，因此 64k/128k 是未跑，不是 0。

### 3.2 VT raw/no-chat 异常分析

VT raw/no-chat 的 8k 最低不是正常能力曲线。n=500 输出风格统计：

| length | score | 0分样本 | 部分分样本 | 满分样本 | `To answer...` 开头 | 直接/其它输出 |
|---|---:|---:|---:|---:|---:|---:|
| 8k | 15.6 | 336/500 | 116/500 | 48/500 | 284/500 | 63/500 |
| 16k | 34.2 | 304/500 | 41/500 | 155/500 | 176/500 | 170/500 |
| 32k | 32.0 | 314/500 | 45/500 | 141/500 | 85/500 | 203/500 |

解释：8k 更容易触发 think-style 解释，60 token 内没有列出 5 个变量；16k/32k 更常直接吐变量列表，所以分数更高。这是 output-format artifact 与任务难度叠加。

### 3.3 VT naive chat-template disable-thinking probe（n=100，已完成）

路径：`ruler_results/qwen32_vt_disablethinking_n100_j16_chunk512/_summary.json`

| task | 8k | 16k | 32k |
|---|---:|---:|---:|
| variable_tracking, chat + enable_thinking=False | 4.6 | 3.2 | 2.4 |

结论：简单把 RULER prompt 包进 chat template 并关 thinking **会伤害 VT**。这说明 VT 不能直接沿用 BABILong 的 chat-template no-thinking 口径；需要新 probe：

- raw prompt + 强 answer-only 指令；
- 或 chat no-thinking + 重写 prompt，让 assistant 只输出变量名列表；
- 或增大/调整 max_new_tokens 与 stop rule；
- 同时记录 selected chunks / retrieval hit，区分检索链条问题和输出协议问题。

---

## 4. LongBench 结果

### 4.1 raw/no-chat，按 Qwen3-8B 兼容口径重算（已完成）

路径：`longbench_results/qwen32_zerotrain_j16_chunk512/scores.json`

口径修正：empty prediction 保留为合法预测，F1=0；不再让整个 shard invalid。这个口径与 Qwen3-8B LongBench 历史结果一致。

| task | F1 | EM | n |
|---|---:|---:|---:|
| narrativeqa | 6.24 | 0.00 | 200 |
| qasper | 14.24 | 0.00 | 200 |
| hotpotqa | 12.25 | 0.00 | 200 |
| 2wikimqa | 14.27 | 0.50 | 200 |
| musique | 7.71 | 0.00 | 200 |
| multifieldqa_en | 28.54 | 0.00 | 150 |
| **AVERAGE** | **13.87** | **0.08** | 6 tasks |

### 4.2 与 Qwen3-8B LongBench 对比

Qwen3-8B 参考来自 `status/QCMEM_PAPER_DRAFT.md` §2.8（raw/no-chat 官方 F1）。

| task | Qwen3-8B QCMem | Qwen3-32B zero-train QCMem | Δ 32B-8B |
|---|---:|---:|---:|
| narrativeqa | 3.93 | 6.24 | +2.31 |
| qasper | 11.07 | 14.24 | +3.17 |
| hotpotqa | 11.64 | 12.25 | +0.61 |
| 2wikimqa | 11.69 | 14.27 | +2.58 |
| **AVG shared 4** | **9.58** | **11.75** | **+2.17** |
| musique | — | 7.71 | — |
| multifieldqa_en | — | 28.54 | — |
| **AVG 32B all 6** | — | **13.87** | — |

### 4.3 LongBench 问题与计划

现有 raw/no-chat 结果中有 4 条 deterministic decoded-empty prediction，但它们现在按 F1=0 纳入，不再导致 shard failed。

仍需做 disable-thinking LongBench probe：

- tasks：narrativeqa / qasper / hotpotqa / 2wikimqa / musique / multifieldqa_en；
- 每 task 20-50 samples；
- 口径：chat template + `enable_thinking=False`；
- 指标：F1/EM、empty count、avg output words、是否还有 `<think>` / `</think>`；
- 若明显改善，再跑正式全测试集。

---

## 5. 当前在跑任务

| 节点 | 任务 | 输出 | 状态 |
|---|---|---|---|
| lhz `183.242.150.6:32668` | VT disable-thinking n=100 probe | `ruler_results/qwen32_vt_disablethinking_n100_j16_chunk512` | ✅ 已完成，结果 4.6/3.2/2.4 |
| dev4 `183.242.150.6:32679` | BABILong qa1/qa2 disable-thinking n=500 | `babilong_results/qwen32_qa12_disablethinking_n500_j16_chunk512` | RUNNING |
| lhz2 `183.242.150.6:32669` | 无当前 Qwen32 任务 | — | 空闲/未使用 |

---

## 6. `Qwen模型结果.md` 对 32B 的 benchmark 要求

主表一共 13 行：

1. RULER niah_single
2. RULER niah_multikey
3. RULER vt
4. BABILong qa1/qa2/qa5
5. LongBench
6. LongEval
7. LoCoMo
8. LongMemEval（harness 待接）
9. InfiniteBench / ∞Bench（harness 待接）
10. HELMET（harness 待接）
11. vs-Dense 效果
12. vs-Dense 速度
13. split-j sweep

当前已有脚本可执行硬项：RULER / BABILong / LongBench / LongEval / LoCoMo / vs-Dense 效果 / vs-Dense 速度 / split-j sweep。

---

## 7. 后续计划（优先级）

### P0：收尾当前正在跑的 BABILong qa1/qa2 no-thinking

- 等 dev4 qa1/qa2 n=500 完成；
- 聚合 qa1/qa2 + 已完成 qa5；
- 形成 BABILong disable-thinking 正式表；
- 与 raw/no-chat 表对照，写出 artifact 诊断。

### P1：修正 RULER VT no-thinking 口径

naive chat-template no-thinking 失败，下一步要跑小样本对照：

1. raw prompt + 追加硬约束：`Output only the variable names, separated by spaces. Do not explain.`
2. chat-template + enable_thinking=False + 重写 VT answer prefix；
3. topk/iter 检索诊断：检查 selected chunks 是否覆盖 5 个变量链；
4. 如果 answer-only prompt 能修复，再跑 VT n=500 正式表。

### P2：LongBench disable-thinking probe

- 先 n=20/50 per task；
- 若 empty 消失、F1 上升或输出更短，再跑全量；
- 正式口径仍保留 empty=F1 0，不做 shard invalid。

### P3：32B 其余 benchmark

- LongEval；
- LoCoMo；
- vs-Dense 效果/速度；
- 必要时补 split-j sweep 正式表；
- LongMemEval / InfiniteBench / HELMET 需要先接 harness。

---

## 8. 操作备忘

### lhz/lhz2/dev4 环境

```bash
# lhz
ssh -i /root/.ssh/mac_gpu_key -o IdentitiesOnly=yes -p 32668 root@183.242.150.6

# lhz2
ssh -i /root/.ssh/mac_gpu_key -o IdentitiesOnly=yes -p 32669 root@183.242.150.6

# dev4, 4×H200
ssh -i /root/.ssh/mac_gpu_key -o IdentitiesOnly=yes -p 32679 root@183.242.150.6
```

项目根：`/volume/haru/Mixture-of-Memory`
Python：`/volume/haru/Mixture-of-Memory/.venv_hy3/bin/python`

不要用 `/volume/haru/envs/dllm/bin/python` 跑 QCMem；dllm 环境 transformers 4.57 与 QCMem mask API 不兼容。

### no-thinking patch 位置

- BABILong 临时 evaluator：`scripts/_eval_qcmem_babilong_disable_thinking_tmp.py`
- RULER 临时 evaluator：`scripts/_eval_ruler_qcmem_disable_thinking_tmp.py`

关键调用：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```
