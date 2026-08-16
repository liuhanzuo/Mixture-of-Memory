---
name: paperb-olmo2-base-not-chat
description: Paper B 剪层-heal OLMo-2 是 BASE LM — 只能 base 口径 eval（ppl + LL-based MC）对照 OLMo-2 vanilla base，绝不能 chat-style
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

Paper B 的 prune-then-heal OLMo-2（keep-front-j + n_fresh，在 Dolmino DCLM = base 预训练语料上续训，NTP 目标，**无 SFT/chat 微调**）是一个 **BASE 语言模型**，定位是「更小的 base 模型」，不是 chat 助手。

**因此它不能像 Paper A 的 Qwen3-8B CoMem 那样 chat-style 交互/eval。** eval 协议必须是 base-model 口径：
- held-out NTP perplexity（`scripts/eval_olmo2_probe2_ppl.py`）
- likelihood-based zero-shot MC / knowledge（`scripts/eval_olmo2_probe2_downstream.py`：对 teacher-forced continuation 的 log-prob argmax，**零 generation、无 chat_template、无 BOS**，match 官方 OLMo-2 lm-eval）
- 分母/对照 = **OLMo-2 vanilla BASE**（1B full 16L held-out ppl 10.64；7B full 32L 7.40），**不是 OLMo-2-Instruct**。base_full 已复现公开数字（7B MMLU .605 / HS .805）→ driver 验证过。

**Why:** 用户 2026-07-19 指令——"想清楚这个模型的定位，可能不能像其他模型一样 chat 方式交互，OLMo2 训的和 OLMo2 比较"。chat-eval 会误表这个 base 模型的能力。

**How to apply:** Paper B 的任何 eval 都走上面两个 base driver，绝不掺 Paper A 的 chat+no-think 长上下文生成 harness；报"vs vanilla" 一律用 OLMo-2 BASE 分母。相关：[[bottleneck-layer-sweep-monotone]]。
