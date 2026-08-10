# Qwen3-32B thinking / no-thinking 问题排查记录

更新：2026-07-16

本文记录本轮对话中围绕 **Qwen3-32B QCMem 评测被 thinking/output-format 污染** 的排查结论、已跑实验和后续口径决策。

---

## 0. 一句话结论

Qwen3-32B 是 thinking 型 instruct 模型。对于 BABILong 这类短答案 benchmark，raw/no-chat 或默认 chat-template 会让模型输出推理前言、`<think>`、自问自答或复读，从而严重污染自动判分。**BABILong 应改用 `chat_template + enable_thinking=False` 作为正式口径。**

但不是所有 benchmark 都应该强行套 chat-template no-thinking：

- **BABILong**：可以自然一键 no-thinking，已验证大幅修复 qa5。
- **LongBench**：技术上可以一键 no-thinking，但历史 Qwen3-8B 是 raw/no-chat；当前先按 8B 兼容 raw 口径保留，另可跑 no-thinking 附表。
- **RULER**：原始 prompt 是 completion-style，不能直接包 chat template，否则会改变任务结构；RULER 主表暂保持 raw/no-chat，no-thinking 仅作为诊断 probe。

---

## 1. 背景：发现异常

用户/合作者注意到 Qwen3-32B BABILong qa5 原表很反常：

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa5 raw/no-chat | 18.2 | 19.4 | 21.4 | 34.4 | 42.4 | 48.4 | 50.8 |

0k 极低而 32k 更高，不符合常识。检查原始输出后发现：

- 0k 常输出：`To answer this question, I need to ...`，答案词没有在判分器期望位置出现；
- 32k 常输出：`Answer: Answer: football ...`，答案词很早出现，因此被判对；
- 默认 chat-template 会输出 `<think> Okay, let's see...`，128 token 内都不结束 thinking，判分为 0。

因此异常主要不是 QCMem 检索能力，而是 **Qwen3-32B 的 thinking/output-format 与短答案判分协议不匹配**。

---

## 2. BABILong qa5 probe：确认 thinking 是主因

### 2.1 小样本 probe 设置

- 模型：Qwen3-32B stock/no-adapter
- QCMem：`resume_j=16`，`chunk_size=512`，bm25 topk12，bos sink
- task：BABILong qa5
- lengths：0k / 8k / 32k
- n=20/cell
- 对比：
  1. raw/no-chat，`max_new_tokens=20`
  2. raw/no-chat，`max_new_tokens=128`
  3. chat template 默认 thinking
  4. chat template + `enable_thinking=False`

### 2.2 结果

| 模式 | max_new_tokens | 0k | 8k | 32k | 输出特征 |
|---|---:|---:|---:|---:|---|
| raw/no-chat | 20 | 15% | 35% | 50% | 大量 `To answer...` / `Answer: Answer...` |
| raw/no-chat | 128 | 15% | 35% | 50% | 多给 token 没用，继续啰嗦/复读 |
| chat 默认 thinking | 20 | 0% | 0% | 0% | 全部 `<think>`，没有最终答案 |
| chat 默认 thinking | 128 | 0% | 0% | 0% | 仍然卡在 `<think>` |
| chat + `enable_thinking=False` | 20 | 100% | 85% | 90% | 直接输出短答案 |
| chat + `enable_thinking=False` | 128 | 100% | 85% | 90% | 与20一致，答案已是短输出 |

关键结论：

- `max_new_tokens` 不是主因：raw 20 和 128 一样；
- 默认 chat-template 会显式进入 `<think>`，更糟；
- `enable_thinking=False` 直接修复短答案输出。

---

## 3. BABILong qa5 正式 no-thinking n=500

用户要求正式重跑 qa5。lhz 上完成：

路径：`babilong_results/qwen32_qa5_disablethinking_n500_j16_chunk512/_summary.json`

口径：`chat_template=True + enable_thinking=False`，`max_new_tokens=20`。

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa5 no-thinking n=500 | **89.2** | **87.6** | **85.2** | **84.0** | **79.2** | **79.2** | **81.4** |

结论：原 qa5 的反常曲线基本坐实是 thinking/output-format artifact。

---

## 4. BABILong qa1/qa2 no-thinking 正式重跑

用户提供新机器：

```bash
ssh root@183.242.150.6 -p 32679
```

实际连接需要 key：

```bash
ssh -i /root/.ssh/mac_gpu_key -o IdentitiesOnly=yes -p 32679 root@183.242.150.6
```

节点：dev4，4×H200。

已启动：Qwen3-32B BABILong qa1/qa2 disable-thinking n=500。

- 脚本：`scripts/_qwen32_qa12_disablethinking_n500_pool.sh`
- 输出：`babilong_results/qwen32_qa12_disablethinking_n500_j16_chunk512`
- 日志：`logs/qwen32_qa12_disablethinking_n500/`
- 协议：`chat_template + enable_thinking=False`，qa1/qa2 × 0k/1k/2k/4k/8k/16k/32k × 4 shards，每 cell n=500
- 状态：2026-07-16 13:55 时正在跑 qa1 4k；最终结果待聚合。

---

## 5. RULER 情况：不能直接套 chat no-thinking

### 5.1 当前 Qwen3-32B raw/no-chat RULER 正式表

路径：`ruler_results/qwen32_zerotrain_n500_j16_chunk512`

| task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| niah_single_2 | 100.0 | 99.6 | 92.8 | 97.2 | 96.8 |
| niah_multikey_1 | 96.0 | 89.6 | 97.4 | 94.4 | 96.0 |
| variable_tracking | 15.6 | 34.2 | 32.0 | — | — |

VT 按 `Qwen模型结果.md` 只要求 8k/16k/32k，因此 64k/128k 是未跑，不是 0。

### 5.2 VT 异常：8k 最低

VT 原表里 8k 反而最低。n=500 输出风格统计：

| length | score | 0分样本 | 部分分样本 | 满分样本 | `To answer...` 开头 | 直接/其它输出 |
|---|---:|---:|---:|---:|---:|---:|
| 8k | 15.6 | 336/500 | 116/500 | 48/500 | 284/500 | 63/500 |
| 16k | 34.2 | 304/500 | 41/500 | 155/500 | 176/500 | 170/500 |
| 32k | 32.0 | 314/500 | 45/500 | 141/500 | 85/500 | 203/500 |

解释：8k 更容易触发解释型输出，60 token 内没有列出完整 5 个变量；16k/32k 更常直接列变量名，因此满分样本更多。由于 8k read_len 基本等于全 prompt 长度，检索不是最可能主因。

### 5.3 Qwen3-8B VT 是怎么做的

Qwen3-8B VT 历史口径是 raw/no-chat completion，不是 chat template：

```text
Question: Find all variables that are assigned the value ...
Answer: According to the chain(s) ..., they are:
```

同时 RULER VT 自带一个 in-context worked example：

```python
_make_vt_icl(...): a full mini VT prompt INCLUDING its answer
```

因此 Qwen3-8B 的 VT 依赖 completion-style 续写，不是 user→assistant chat 问答。

### 5.4 naive chat-template no-thinking VT probe（失败）

路径：`ruler_results/qwen32_vt_disablethinking_n100_j16_chunk512/_summary.json`

| probe | 8k | 16k | 32k |
|---|---:|---:|---:|
| VT chat-template + `enable_thinking=False` n=100 | 4.6 | 3.2 | 2.4 |

结论：不能直接把 RULER VT 原 prompt 包成 chat-template。因为原 prompt 的 `they are:` 是 assistant 要续写的位置；包进 chat 后它变成 user message 内容，任务结构被改变。

### 5.5 raw completion + empty-think stub probe（进行中）

用户要求“其它保持和 Qwen3-8B zero-shot VT 一样，只加 no-think”。为此启动了更合理的诊断：

```text
原 RULER VT raw completion prompt + "<think>\n\n</think>\n\n"
```

保持：

- raw/no-chat completion；
- VT ICL worked example；
- answer prefix；
- iter_bm25 topk16 rounds4 hop4；
- string_match_all；
- max_new_tokens=60。

路径：

```bash
ruler_results/qwen32_vt_raw_nothink_n100_j16_chunk512
logs/qwen32_vt_raw_nothink_n100/
```

状态：2026-07-16 14:16 仍在跑，8 张 H200 满载，8k/16k shard 进行中。该结果将判断 raw completion 下只加 no-think 是否能修复 VT 8k 反常。

---

## 6. LongBench 情况

### 6.1 strict-empty gate 修正

Qwen3-32B LongBench 曾因 4 条 deterministic decoded-empty prediction 被 strict gate 判为 incomplete。但这与 Qwen3-8B LongBench 历史口径不一致。

修正后口径：

- raw/no-chat prediction 原样保留；
- empty string 是合法 prediction；
- empty 的 F1/EM = 0；
- 不因为某 row empty 让 shard invalid。

重算路径：`longbench_results/qwen32_zerotrain_j16_chunk512/scores.json`

| task | F1 | EM | n |
|---|---:|---:|---:|
| narrativeqa | 6.24 | 0.00 | 200 |
| qasper | 14.24 | 0.00 | 200 |
| hotpotqa | 12.25 | 0.00 | 200 |
| 2wikimqa | 14.27 | 0.50 | 200 |
| musique | 7.71 | 0.00 | 200 |
| multifieldqa_en | 28.54 | 0.00 | 150 |
| **AVERAGE** | **13.87** | **0.08** | 6 tasks |

### 6.2 与 Qwen3-8B LongBench 对比

Qwen3-8B 参考来自 `status/QCMEM_PAPER_DRAFT.md` §2.8，raw/no-chat 官方 F1。

| task | Qwen3-8B QCMem | Qwen3-32B zero-train QCMem | Δ |
|---|---:|---:|---:|
| narrativeqa | 3.93 | 6.24 | +2.31 |
| qasper | 11.07 | 14.24 | +3.17 |
| hotpotqa | 11.64 | 12.25 | +0.61 |
| 2wikimqa | 11.69 | 14.27 | +2.58 |
| **AVG shared 4** | **9.58** | **11.75** | **+2.17** |
| musique | — | 7.71 | — |
| multifieldqa_en | — | 28.54 | — |
| **AVG 32B all 6** | — | **13.87** | — |

### 6.3 LongBench no-thinking 计划

LongBench 技术上可以用 chat-template + `enable_thinking=False`，但历史 Qwen3-8B 是 raw/no-chat。当前建议：

- raw/no-chat 保留为与 Qwen3-8B 可比的主记录；
- 另跑 LongBench no-thinking 小样本 probe/附表，观察 empty count、F1、输出长度是否改善。

---

## 7. 当前口径决策

### 7.1 能自然一键 no-thinking 的，开 no-thinking

适用：

- BABILong：context + question → short answer，已验证；
- LongBench：技术上适用，但要区分 raw/no-chat 历史可比口径与 no-thinking 新口径。

### 7.2 不能自然一键 no-thinking 的，保持原 benchmark 结构

适用：

- RULER NIAH / VT：completion-style prompt，不直接套 chat-template；
- LongEval 若是 completion-style，也不强行改。

### 7.3 诊断可以做，但不要直接替换主表

包括：

- raw completion + `<think>\n\n</think>` stub；
- answer-only prompt；
- chat-style 重写 RULER prompt。

这些都需要单独标注，不与原始 benchmark 主表混淆。

---

## 8. 已写入/相关文件

- `qwen32b_exp.md`：更宽的 Qwen3-32B no-thinking 结果与计划表。
- `thinking.md`：本文。
- `UPDATELOG.md`：已追加 LongBench keep-empty 重算记录。
- `status/TRAINER_ACTIVITY.jsonl`：已追加 LongBench keep-empty 重算记录。
- `status/GPU_STATUS.md`：记录 lhz/dev4 当前任务。

---

## 9. 后续待办

1. 等 dev4 上 BABILong qa1/qa2 no-thinking n=500 完成，聚合成完整 BABILong no-thinking 表。
2. 等 lhz 上 VT raw completion + empty-think stub n=100 完成，判断 VT 反常是否能修复。
3. 如 VT stub 有效，再决定是否跑 VT n=500 正式表；如无效，则保持 raw/no-chat 主表，并将 VT 解释为 output-format + 链式检索/readout 难题。
4. LongBench 可跑 no-thinking 小样本 probe，但正式对比 8B 仍保留 raw/no-chat keep-empty F1 口径。
5. 后续所有 Qwen3 benchmark 先判断 prompt 类型：chat QA 型可一键 no-thinking；completion 型不强行一键。
