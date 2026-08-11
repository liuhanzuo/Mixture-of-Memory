# LongBench 的 near-tie 不是模型上限，是解析伪影 —— 而修好之后我们从「第 2」掉到「第 4」

**日期**: 2026-08-11 12:40 GMT+8
**触发**: 用户假设「表现相近是因为受模型上限 bound」。假设方向对（确有天花板），但**根因不是模型能力，是输出解析**，且修正后**排序对我们不利**。
**状态**: 需要决策，不要先写进论文

---

## 1. 我的 F1 实现已对齐官方（先证明可信）

用我自己写的 LongBench token-F1 重算 `longbench_results/qcmem_8b_zeroshot_j9_chatFALSE/`
的 per-example 预测，与 `paperA/sections/tab_longbench.tex` 的 `CoMem frozen (j=9)` 行逐个数字对比：

| | NQA | Qasper | Hotpot | 2Wiki | MultiF. | Musique | macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| 论文 | 4.63 | 11.23 | 9.41 | 10.71 | 22.05 | 5.72 | 10.63 |
| 我重算 | 4.63 | 11.23 | 9.41 | 10.71 | 22.05 | 5.72 | 10.62 |

**6/6 全等。** 所以下面的分析用的是同一口径，不是我换了 metric 换出来的。

## 2. 真实病因：模型把 thinking 文本写进了答案

抽 `narrativeqa` 前几条预测：

```
GOLD: ['The Atlas Mountains']
PRED: 'The witch lives in a cave on Atlas.\nOkay, so I need to figure out where the witch lives based on the provided text. Let me read th'

GOLD: ['Her face is blurry']
PRED: 'Reiko sees her own face in the mirror.\nOkay, so I need to figure out what Reiko sees in the photograph that her ex-husband takes o'
```

**答案本身在第一句，后面粘了一整段 chain-of-thought。** token-F1 的分母是预测长度，
所以一段 90+ 词的 thinking 会把 precision 摊薄到接近 0 —— 哪怕答案完全正确。

泄漏率（正则匹配 `Okay,|Alright,|Let me|So I need|Wait,|Hmm,`）：

| dataset | n | 泄漏 | 泄漏率 | pred 平均词数 |
|---|---:|---:|---:|---:|
| narrativeqa | 200 | 163 | **81.5%** | 96.2 |
| qasper | 200 | 105 | 52.5% | 97.1 |
| hotpotqa | 200 | 108 | 54.0% | 22.8 |
| 2wikimqa | 200 | 112 | 56.0% | 21.3 |
| multifieldqa_en | 150 | 73 | 48.7% | 48.0 |
| musique | 200 | 93 | 46.5% | 23.8 |
| **合计** | **1150** | **654** | **56.9%** | |

注：这是 `chat_template=False` 协议的**直接后果** —— 没有 chat template
就没有 assistant turn 边界，模型不知道该在哪停，于是把「先答再想」的 base-LM
行为整段吐出来。协议本身是对的（用户 2026-07-22 指令，模型无 SFT/RL），
**但配 token-F1 这个 metric 就产生了系统性低估**。

## 3. 修掉之后：F1 涨 2.3 倍，但**我们的相对位置变差**

只取第一句（最保守的修法，不做任何 answer-extraction 调参）：

| dataset | as-is | 剥 thinking | 首句 only | gain |
|---|---:|---:|---:|---:|
| narrativeqa | 4.63 | 15.35 | **18.60** | +13.97 |
| qasper | 11.23 | 18.86 | **28.50** | +17.26 |
| hotpotqa | 9.41 | 18.97 | **23.24** | +13.83 |
| 2wikimqa | 10.71 | 20.59 | **25.78** | +15.06 |
| multifieldqa_en | 22.05 | 34.79 | **39.90** | +17.85 |
| musique | 5.72 | 7.33 | **9.34** | +3.62 |
| **micro** | **10.13** | 18.64 | **23.55** | **+13.41** |

### ★★ 关键：泄漏率**不是**各 arm 对称的

严格同 4-ds 子集（narrativeqa/qasper/hotpotqa/2wikimqa，每臂 n=800，避免 macro 不可比）：

| arm | n | 泄漏率 | as-is F1 | 首句 F1 | gain |
|---|---:|---:|---:|---:|---:|
| CoMem frozen j9 | 800 | **61.0%** | 9.00 | 24.03 | +15.03 |
| CoMem+LoRA j12 | 800 | 42.1% | 9.58 | 29.43 | +19.84 |
| KV-Direct | 800 | 25.2% | 10.07 | **35.19** | +25.11 |
| InfLLM | 800 | **6.6%** | 9.73 | **34.73** | +25.00 |

**泄漏率从 6.6% 到 61.0%，跨 arm 差 9 倍，而且我们的 arm 泄漏最严重。**

排序变化：

```
as-is    : KV-Direct(10.07) > InfLLM(9.73) > CoMem+LoRA(9.58) > CoMem frozen(9.00)
首句only  : KV-Direct(35.19) > InfLLM(34.73) > CoMem+LoRA(29.43) > CoMem frozen(24.03)
```

as-is 下四臂挤在 9.00–10.07（1pp 带内，看着像"受模型 bound 的 near-tie"）。
修掉解析后带宽张开到 24.03–35.19（**11pp**），**但我们在两个口径下都是最后一名**，
而且差距被放大了：CoMem+LoRA 距 KV-Direct 从 −0.49pp 变成 **−5.76pp**。

per-dataset 首句 F1（确认不是单个 ds 独扛）：

| arm | narrativeqa | qasper | hotpotqa | 2wikimqa |
|---|---:|---:|---:|---:|
| CoMem frozen j9 | 18.60 | 28.50 | 23.24 | 25.78 |
| CoMem+LoRA j12 | 18.27 | 34.78 | 37.41 | 27.24 |
| KV-Direct | 25.23 | 39.05 | 45.75 | 30.72 |
| InfLLM | 22.92 | 41.75 | 43.11 | 31.13 |

**4/4 个 ds 上我们都低于 KV-Direct。** 不是一个 ds 的噪声。

## 4. 这意味着什么

**好消息**：LongBench 的 near-tie 是可修的测量问题，不是「模型上限」。修完有 11pp 动态范围，
benchmark 恢复了区分能力。这本身是个值得写的方法论点（`chat_template=False` × token-F1 的交互）。

**坏消息，必须直说**：修完之后 **LongBench 不再支持我们**。目前 `tab_overview` 里
LongBench 那列（12.15 vs 12.17）被描述成「nearly unchanged」，读起来像「我们持平」；
修正后的真相是**我们在这个 benchmark 上明确落后 KV-Direct 约 5.8pp**。

**不能做的事**：只报「修正后 F1 涨了」而不报排序变化。也不能因为结果不利就不修
—— 泄漏率跨 arm 差 9 倍，as-is 的 near-tie 是**伪影**，继续用它等于用一个偏向
我们的 bug（我们泄漏最多 → 但 as-is 下差距最小，因为所有人都被压到地板）。

## 5. 待决策（不要先动论文）

1. **修法要预注册**。「取第一句」是我选的最保守规则，但它仍是我在**看过结果之后**选的。
   正确做法：先写死抽取规则（不看分数），再跑。候选：(a) 首句；(b) 剥 thinking 正则；
   (c) 官方 LongBench 的 `max_gen_len` 截断 + 停止词。三者结果分别是 23.55 / 18.64 / —— 差异不小。
2. **是否重跑而非重打分**。更干净的做法是重新生成时加停止条件（`\nOkay`, `\n\n` 等），
   而不是事后剪。事后剪对所有 arm 同样施加，可辩护，但重跑更无争议。
3. **这个发现独立于 Qwen3.5**。换模型不会消掉解析 bug；应该先修口径，再谈换模型，
   否则会把「解析修好带来的涨幅」误记成「新模型更强」。

## 6. Provenance

- 预测: `longbench_results/{qcmem_8b_zeroshot_j9_chatFALSE,qcmem_j12,kvdirect,infllm_8b_chatFALSE}/`
- 论文行: `paperA/sections/tab_longbench.tex`（j=9 行逐数字对齐，见 §1）
- ⚠️ `qcmem_j12` 与 `kvdirect` **缺 multifieldqa_en / musique**（各 0 条），所以 §3
  的跨 arm 表只用 4 个共同 ds。6-ds macro 的跨 arm 比较**当前不可做**，需补跑这两个 cell。
