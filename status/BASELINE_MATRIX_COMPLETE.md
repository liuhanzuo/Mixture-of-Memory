# Paper A Baseline Matrix — COMPLETE (2026-07-19)

回答用户"矩阵缺格 + 跑 LLoCO"决策的第一半（**矩阵缺格已全部补齐**）。
统一协议：Qwen3-8B backbone（除 MemoryLLM=Llama-3、StreamingLLM=Qwen3、LLoCO=LLaMA-2 异基座），chat_template=ON、no-think、iter_bm25 selector（CoMem）、官方 scorer（RULER string_match / BABILong compare_answers+TASK_LABELS / LongBench qa_f1 / LoCoMo F1-EM-acc / LongEval acc）。

## 覆盖矩阵（6 主方法 × 5 benchmark）

| 方法 | RULER | LongBench | LongEval | LoCoMo | BABILong | 账本 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **CoMem (ours, Qwen3-8B)** | ✅ | ✅35.79 | ✅ | ✅19.51/28.65 | ✅ | tab_*, PAPER_CONFIG_AUDIT F14b |
| **InfLLM** | ✅ | ✅41.54 | ✅ | ✅25.76/26.38 | ✅ | INFLLM_BASELINE_RESULTS.md |
| **MemoryLLM-8B-chat** | ✅ | ✅12.80 | ✅ | ✅9.93/9.72 | ✅ | MEMORYLLM_REALQA_RESULTS.md |
| **HCache (retrieval-free mid-layer)** | ✅ | ✅19.27 | ✅ | ✅**7.82/8.06** | ✅ | 本文件 §HCache-LoCoMo + F14b |
| **KV-Direct (full-ctx j=0)** | ✅ | ✅42.97 | ✅ | ✅**40.06/43.05** | ✅**本次** | 本文件 §KV-Direct |
| **StreamingLLM (equal-budget)** | ✅ | ✅**37.20** | ✅ | ✅**12.73/17.57** | ✅**本次** | STREAMINGLLM_EQUALBUDGET_RESULTS.md |

> **矩阵状态：全满（零星号，最后一格 StreamingLLM LongBench 已补，AVG F1 37.20，n=1150，empty=0，官方 qa_f1）。** 以下为 2026-07-19 本会话补齐/验证的 cell（铁律2 独立验证：非空 preds + 官方 scorer）。
> StreamingLLM LongBench 37.20（KV-Direct 42.97 > InfLLM 41.54 > SLM 37.20 > CoMem 35.79 > HCache 19.27 > MemoryLLM 12.80）——equal-budget recency 在 real-doc QA（证据多在文档首尾+多数上下文 <6657 token 未被截断）足够甚至略胜 CoMem 检索；recency 真正崩盘在 RULER 多针/vt、LoCoMo、BABILong 长档中段证据（见 STREAMINGLLM_EQUALBUDGET_RESULTS.md §LongBench story）。

---

## §HCache-LoCoMo（✅ 验证 2026-07-19，preds 真实非空）

同 `.73` scorer（F1/EM/acc，无 GPT-4o judge），`locomo_results/hcache_8b_chatnothink/`（8 shard×preds_shard*.jsonl=1986 行，各 ~112KB 非空）。

| method | F1 | EM | acc | cat1 mh | cat2 sh | cat3 temporal | cat4 open | cat5 adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **HCache** | **7.82** | 0.05 | **8.06** | 8.89 | 9.51 | 9.12 | 10.69 | 0.22 |

- 铁律2：preds 非空、well-formed（例 pred="Caroline went to...25 August 2023" vs gold ["7 May 2023"]=真检索失败错值，非退化/空答）。cat5 adversarial 0.22=几乎不拒答（与 CoMem 1.35 同模式）。
- **story**：HCache retrieval-free mid-layer recompute 在多会话记忆检索上 F1 7.82 ≈ CoMem(19.51) 的 40%、KV-Direct(40.06) 的 20% → LoCoMo 上 retrieval-free 显著弱，与其在 LongEval(~0)/LongBench(19.27) 全线最弱一致。

## §KV-Direct-LoCoMo（✅ 验证 2026-07-19，1986/0 empty）

`locomo_results/kvdirect_8b_chatnothink/`。full-ctx j=0（无检索，pack 全 chunk）。

| method | F1 | EM | acc | cat1 mh | cat2 sh | cat3 temporal | cat4 open | cat5 adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **KV-Direct** | **40.06** | 19.59 | **43.05** | 42.75 | 42.94 | 23.10 | 58.26 | 5.61 |

- **story**：full-ctx upper-bound 参照——LoCoMo 上 KV-Direct F1 40.06 是全 baseline 最强（>InfLLM 25.76 > CoMem 19.51 > MemoryLLM 9.93 > HCache 7.82）。CoMem 恒定 read 用 ~1/6.6 read budget 达 KV-Direct acc 的 66%（28.65/43.05），token-F1 差距更大（LoCoMo 是短对话档，full-ctx 直接装得下→检索压缩劣势显现；对比 LongBench CoMem=full 的 83%）。

## §KV-Direct-BABILong（✅ 本次打分 2026-07-19，官方 compare_answers，empty=0 全档）

`babilong_results/kvdirect_8b_chatnothink/`（168 csv=21 cell×8 shard→本次 `scores.json`）。full-ctx j=0。

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| **KV-Direct** qa1 | 99 | 94 | 93 | 89 | 83 | 78 | 71 |
| **KV-Direct** qa2 | 61 | 54 | 55 | 52 | 52 | 47 | 41 |
| **KV-Direct** qa5 | 83 | 79 | 75 | 74 | 77 | 72 | 69 |

- 铁律2：本次 `PYTHONPATH=third_party/babilong-pkg` 内联官方 `compare_answers(target,output,question,TASK_LABELS[task])` 逐 cell 打分，empty_output=0 全档，output well-formed（例 "The most recent location of Mary is bathroom."）。
- **story**：full-ctx BABILong 全档最强（0k-32k 单调缓降，32k 仍 qa1=71/qa5=69）——BABILong ≤32k full-ctx 直接装下 → KV-Direct = CoMem 的强 upper-bound 参照。对比 InfLLM qa1 长档掉更快(32k=37)、CoMem 见 tab_babilong。

---

## LLoCO（用户决策第二半：跑 LLoCO 实证）——见 status/LLOCO_REPRO_PLAN.md + LLOCO_BASELINE_RESULTS.md

- **异基座**（LLaMA-2-7B-chat + AutoCompressor，非 Qwen3-8B）；per-domain supervised LoRA（推理需对应 domain LoRA）。
- 权重已发布：`princeton-nlp/AutoCompressor-Llama-2-7b-6k` + 5 domain LoRA（`xiuyul/Lloco-7b-{nqa,qasper,hqa,qmsum,quality}`）。
- **path (a) 实证跑 ✅ 完成+验证（2026-07-19 18:46，coder ab5d92c，commit `d3e5691` 未 push）**：.73 EVAL-ONLY 自跑 3 个有 LoRA 的 LongBench 任务，n=200/task，**0 empty**（铁律2 直读 scores.json + 独立复算 pred 计数一致）：**narrativeqa F1 24.21/EM 9.0（Table4 23.1，+1.11）、qasper F1 24.45/EM 13.0（26.1，−1.65）、hotpotqa F1 44.24/EM 33.5（46.2，−1.96）、AVG F1 30.97**。三项全在 Table 4 ±2pt 内 → **验证我们 LongBench F1/EM 口径 = LLoCO 官方口径**，compress→softprompt→LoRA pipeline 接线正确；此 3-task 行与 CoMem/InfLLM/KV-Direct/HCache/MemoryLLM 同 eval-set + 同 scorer 直接可比。（坑：bf16 在 flash-attn2.5.6 kv-cache decode kernel 触 SIGFPE core dump→改 fp16 修复=faithful dtype；用预编译 wheel `flash_attn-2.5.6+cu122torch2.1` 避 nvcc13.2 源码编译；isolated conda `lloco_env` py3.10/torch2.1.2/tf4.37.2/peft0.5.0；仅 H20 sm_90 可跑，L20A sm_100 不行。raw：`.73:lloco_results/longbench/`；driver `scripts/eval_lloco_longbench.py`。详见 `status/LLOCO_BASELINE_RESULTS.md`。）
- **path (a.5) 引用 Table 4**（零 GPU，覆盖全 9 任务）：论文 head-to-head 引 LLoCO 官方 LongBench F1（同 scorer 口径），标注 backbone/方法差异——task#10 论文整合时一并写。
- **RULER/BABILong/LoCoMo 排除 LLoCO**（无对应 LoRA，方法不兼容，脚注说明）。
