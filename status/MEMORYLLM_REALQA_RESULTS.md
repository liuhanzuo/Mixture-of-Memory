# MemoryLLM-8B-chat — Real-QA Baseline Results

> **目的**：为论文 baseline matrix 补齐 "same-class stateful-memory baseline" 一列（AGENDA §0 gate #3）。
> MemoryLLM-8B-chat = 固定容量 stateful memory pool（32 层 × 12800 × 4096；num_blocks=50、num_tokens=256、FIFO 溢出丢弃），Llama-3 backbone。
> 长文档按 chunk 注入 memory pool（`inject_memory`），再从短问题 prompt greedy 解码。
>
> **口径（2026-07-17 统一协议）**：chat template ON（此为 -chat 模型）；无 thinking mode（Qwen3-only，不适用）；无 bm25 selector（stateful memory 自带压缩）。dtype=bfloat16，8-shard（每卡一 shard，`[shard_index::8]` strided）。
>
> **⚠️ Caveats（横向对比时必读）**：
> 1. **独立 driver**：MemoryLLM 用独立的 `scripts/eval_memoryllm_*.py` driver（复用各 benchmark 的 data/prompt/scoring 框架，只换 model forward）。
> 2. **Llama-3 backbone**（非 Qwen3）：与 QCMem/CoMem（Qwen3-8B）backbone 不同，是不同基座模型的对比，不是纯算法对比。
> 3. **matched-protocol，非 same-samples**：同 benchmark、同 scorer、同 shard 划分，但 MemoryLLM 走 memory-pool 注入范式，样本构造经模板边界 peel，与 QCMem retrieval-over-chunks 范式不同。
>
> **运行节点**：diskB `.73` = `28.85.35.73`（8×H20）。Python = `PYTHONPATH=external/memoryllm_venv/lib/python3.11/site-packages /usr/bin/python3.11`（py3.11.6 / tf4.43.4 / torch2.6.0+cu124）。模型 `../baselines/memoryllm-8b-chat-hf`。
> **启动时间**：2026-07-19 CST（node `date` 实测）。

---

## 0. Blocker 修复（前置，已完成）

`transformers.masking_utils.create_causal_mask` 在 MemoryLLM 的 tf4.43.4/py3.11 环境不存在，import 链断裂。非侵入式 try/except guard（`create_causal_mask=None`，仅用于 QCMem READ-path，baseline driver 不调用）：

| commit | 文件 | 说明 |
|--------|------|------|
| `bc48926` | `scripts/qcmem_model.py:33` | guard import；不破坏 QCMem tf5.5.4 栈 |
| `311778e` | `src/memory/qcmem/qcmem_model.py:33` | 第二份拷贝（eval_qcmem_locomo→eval_qcmem_babilong→src.memory.qcmem 命中）同 guard |
| `5fcfd2f` | `scripts/eval_memoryllm_locomo.py` + `scripts/eval_memoryllm_longbench.py` | 新增两个 MemoryLLM QA driver（复用 LoCoMo/LongBench 框架 + memoryllm_common forward） |

双向验证：torch-base tf5.5.4 `import scripts.qcmem_model` OK；py3.11 栈 import MemoryLLM driver OK。
committer=LiuHanzuo，`git add <具体文件>`（非 -A），无 AI trailer，未 push（交 main /gitpush）。

---

## 1. LongEval（topic/line-retrieval，最高优先级）

- driver：`scripts/eval_memoryllm_longeval.py`（已有）
- 配置：`--lengths 8k 16k 32k 64k 128k --num_samples 50 --max_new_tokens 48 --chunk_size 1024 --output_name memoryllm_longeval_chatnothink`
- raw：`.73:/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/longeval_results/memoryllm_longeval_chatnothink/longeval_{len}_shard{0..7}of8.json`
- **铁律2 验证**：8k / 16k / 32k / 64k / 128k 各 50 records（128k 抽样 n=50），**empty preds≈0**（32k=0、64k=1、128k=0），pred 为规范 6 位数字（模型输出 "The <REGISTER_CONTENT> in line X is Y." → pred 抽出数字）。非空、well-formed；错例是"抽出了一个 register 数但值不对"（如 exp=323230 pred=708328），非空答/退化 → **真检索失败，非截断 artifact**。

### 1.1 Accuracy（merged 8-shard，n=50/length）

| method | 8k | 16k | 32k | 64k | 128k |
|--------|----|----|----|----|----|
| **MemoryLLM-8B-chat（本次）** | **0.200** | **0.200** | **0.180** | **0.080** | **0.040** |
| CoMem (ours, Qwen3-8B) | 1.00 | 0.96 | 1.00 | 0.94 | 0.98 |
| KV-Direct | 1.00 | 0.98 | 0.96 | 0.36 | 0.00 |
| InfLLM | 0.60 | 0.26 | 0.12 | 0.04 | 0.02 |
| HCache | 0.02 | 0.00 | 0.02 | 0.00 | 0.00 |

（raw merged summary：`.73:.../memoryllm_longeval_chatnothink/_summary_merged.json`；correct 计数 8k/16k=10/50、32k=9/50、64k=4/50、128k=2/50。）

### 1.2 结论（LongEval 全 5 档 ✅ 完成 2026-07-19 08:21）

MemoryLLM 在 LongEval line-retrieval 上 **8k=0.20、16k=0.20、32k=0.18、64k=0.08、128k=0.04**——短档即低位平坦（~0.20，远低于 CoMem 1.00、InfLLM 0.60），长档随长度**缓慢下滑**至 128k=0.04（接近 HCache 全崩档位）。
错误模式不是"不会说话"（empty≈0，输出结构规范），而是**注入后从 memory pool 里读出了错误的 register 值**（~80-96% 定位错，且长档 FIFO 丢弃早期内容使定位更难）。
说明：**MemoryLLM 的固定 memory pool 对"精确定位单行 register 值"这类无损检索任务天然不利**——注入即有损压缩，精确 key-value 定位能力弱；长档 FIFO 丢弃进一步恶化（64k→0.08、128k→0.04）。这正是 CoMem（可缓存语义 + 检索，恒定 read）相对 stateful-compression baseline 的卖点：CoMem LongEval 恒 0.94-1.00，MemoryLLM 0.20→0.04。
**LongEval 五方对照完整**：CoMem(恒~1.0) ≫ InfLLM(0.60→0.02，block-memory 随长档崩) ≈ MemoryLLM(0.20→0.04，stateful-pool 全档低) > HCache(≈0，retrieval-free)；KV-Direct 短档强(1.00)但 128k 窗口溢出→0。

---

## 2. LoCoMo（long-conversation memory，优先级 2）

- driver：`scripts/eval_memoryllm_locomo.py`（新增 `5fcfd2f`，复用 `eval_qcmem_locomo` 的 data/prompt/scoring；只换 forward）
- 配置：n=1986，8-shard，`--max_new_tokens 48`，无 GPT-4o judge（metrics only：F1/EM/acc）
- context/query 切分：按 QCMem prompt 固定模板边界 peel（`\n\n# Conversation history\n` ... `\n\n# Question\n`），CPU 单测已验证（CONTEXT=62995 字符 history，QUESTION_PROMPT=380 字符 instruction+question，无泄漏）
- raw：`.73:.../locomo_results/memoryllm_chatnothink/preds_shard{0..7}of8.jsonl`
- **状态**：✅ **DONE 2026-07-19 08:57**（8-shard 生成 + score）。铁律2 abstain-check：1986 preds，**0 empty**，均为规范会话式答案（非退化/非空答）。同 `.73` scorer（F1/EM/acc，无 GPT-4o judge），与 InfLLM/CoMem 行同口径 apples-to-apples。

| method | F1 | EM | acc |
|--------|----|----|-----|
| **MemoryLLM-8B-chat（本次）** | **9.93** | **0.96** | **9.72** |
| InfLLM | 25.76 | — | 26.38 |
| CoMem (iter_bm25) | 19.51 | — | 28.65 |

**per-category F1（MemoryLLM）**：cat1 multi_hop (n282) 13.11 / cat2 single_hop (n321) 9.16 / cat3 temporal (n96) 8.79 / cat4 open_domain (n841) 14.19 (acc16.41) / cat5 adversarial (n446) **0.67**（几乎全答→token-F1 极低，与 CoMem 1.35 同模式：都不擅拒答）。
**结论**：MemoryLLM LoCoMo overall F1 **9.93 ≈ CoMem(19.51) 的一半、InfLLM(25.76) 的 ~38%**——固定 stateful memory pool 在多会话记忆检索上显著弱于 retrieval-based（CoMem）与 block-memory（InfLLM）。cat4 open_domain 最强(14.19)但仍 < CoMem/InfLLM；adversarial 与 CoMem 一样近 0（不拒答）。raw scores：`.73:.../locomo_results/memoryllm*/scores.json`。

---

## 3. LongBench（real long-doc QA，优先级 3）

- driver：`scripts/eval_memoryllm_longbench.py`（新增 `5fcfd2f`，复用 `eval_longbench_mem_space` 的 data/prompt/DATASET2MAXGEN/scoring）
- 6 QA 子集：narrativeqa / qasper / hotpotqa / 2wikimqa / multifieldqa_en / musique（官方 qa_f1）
- 数据：`data/longbench_raw/data`（离线 jsonl，1150 样本），8-shard，per-dataset DATASET2MAXGEN 生成预算
- raw：`.73:.../longbench_results/memoryllm_chatnothink/{ds}_shard{0..7}of8.jsonl`
- **状态**：✅ **DONE 2026-07-19 09:15**（8-shard 生成 + score）。铁律2 abstain-check：1150 preds，**0 empty**（均非空答）。同 `eval_longbench_mem_space` 官方 qa_f1，与其它方法同口径。

| method | narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **AVG** |
|--------|----|----|----|----|----|----|----|
| **MemoryLLM-8B-chat（本次）** | 17.71 | 17.37 | 7.11 | 7.42 | 22.75 | 4.45 | **12.80** |
| KV-Direct | 26.17 | 45.62 | 57.83 | 42.46 | 53.05 | 32.65 | 42.97 |
| InfLLM | 21.45 | 47.10 | 57.43 | 40.72 | 52.32 | 30.24 | 41.54 |
| CoMem | 21.09 | 37.11 | 49.54 | 37.67 | 45.66 | 23.67 | 35.79 |
| HCache | 8.15 | 28.72 | 16.95 | 19.76 | 34.03 | 8.01 | 19.27 |

**结论**：MemoryLLM LongBench AVG F1 **12.80 = 全 baseline 最弱**，甚至低于 retrieval-free HCache(19.27)、远低于 CoMem(35.79)/InfLLM(41.54)/KV-Direct(42.97)。multi-hop（hotpotqa 7.11 / 2wikimqa 7.42 / musique 4.45）几乎全崩——固定 memory pool 对真实长文档多跳 QA 的可组合检索能力最弱。n=200/ds（multifieldqa_en n=150），全集。raw scores：`.73:.../longbench_results/memoryllm_chatnothink/scores.json`。

---

## 3.5 MemoryLLM 5-benchmark 汇总（✅ 全 5 benchmark COMPLETE 2026-07-19 13:0x）

统一协议（chat template ON、no bm25、bf16、8-shard、官方 scorer、Llama-3 backbone）下 MemoryLLM **全 5 大 real-QA benchmark 全部完成 + 铁律2 独立验证**：
- **LongEval**：8k0.20 / 16k0.20 / 32k0.18 / 64k0.08 / 128k0.04（stateful-pool 全档低，长档 FIFO 恶化）
- **LoCoMo**：overall F1 9.93 / acc 9.72（≈ CoMem 一半、InfLLM ~38%）
- **LongBench**：AVG F1 12.80（全 baseline 最弱，低于 HCache）
- **RULER**（n=100 canonical，§3.6）：single 21/31/24/8/14、multikey 30/27/26/13/8、vt 0.8/2.2/0.6/0.4/0.0（8k-128k）——精确 needle 从短档就弱、VT 全崩
- **BABILong**（n=100 canonical，§3.7）：qa1 53→7、qa2 37→16、qa5 48→37（0k→32k）——单事实短档中等、长档单调掉；qa2/qa5 多事实一直低位
- **一句话 story**：MemoryLLM 固定容量 stateful memory pool 在**精确检索 / needle / 多跳 / 多会话记忆 / 多事实推理**五类任务上系统性弱于 retrieval-based（CoMem）与 block-memory（InfLLM）——为 CoMem "可缓存语义 + 恒定 read 检索" 提供了最直接的同类 stateful-memory 对照。**5 benchmark 无一档追平 CoMem。**
- **收尾溯源**：2026-07-19 09:2x heartbeat 派 coder aa70b71 在 .73 启动 canonical cohort chain（PID 1798123，setsid nohup，13:0x 前完成、PID 已退、7 DONE marker 全在）：`scripts/eval_memoryllm_ruler.py`（single/multikey/vt × 8k-128k n=100 string_match 8-shard）→ **新建** `scripts/eval_memoryllm_babilong.py`（qa1/qa2/qa5 × 0k-32k n=100 官方 `compare_answers`+`TASK_LABELS` 8-shard）。⚠️ 新 driver 复用 **ported forward**（`eval_memoryllm_common`），与其它 4 列一致——**取代**旧 `run_babilong_memoryllm.py`（走原生 `generate()`，forward path 不一致，弃用）。输出 `ruler_results/ruler_memoryllm_canonical/` + `babilong_results/babilong_memoryllm_canonical/`。

---

## 3.6 RULER（niah_single_2 / niah_multikey_1 / variable_tracking × 8k-128k, n=100, string_match）— ✅ COMPLETE (2026-07-19 12:2x)

- driver：`scripts/eval_memoryllm_ruler.py`（ported forward，8-shard，`--limit 100 --max_new_tokens 48 --chunk_size 1024`）
- raw：`.73:.../ruler_results/ruler_memoryllm_canonical/{task}_{len}_shard{0..7}of8.json` + `_summary_merged.json`
- **铁律2 验证**：main 独立复算（读每 shard `records[].recall` 求 100.0×Σrecall/n）与 `_summary_merged.json` **逐 cell 完全一致**；15 cell 全部 **n=100, empty output=0**（record 字段 = `output`+`recall`，非 `pred`；sample output 规范如 "The special magic number for ... is 10" vs answers=['3797280'] → 非空/well-formed，错例是**读出错值**=真检索失败）。

| task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| **MemoryLLM** niah_single_2   | 21.0 | 31.0 | 24.0 | 8.0  | 14.0 |
| **MemoryLLM** niah_multikey_1 | 30.0 | 27.0 | 26.0 | 13.0 | 8.0  |
| **MemoryLLM** variable_tracking | 0.8 | 2.2 | 0.6 | 0.4 | 0.0 |
| CoMem single (对照) | 100 | 100 | 100 | 100 | 96 |
| InfLLM single (对照) | 100 | 99 | 95 | 54 | 53 |
| InfLLM multikey (对照) | 99 | 93 | 65 | 37 | 24 |
| InfLLM vt (对照) | 100 | 98.2 | 90.8 | 0 | 0 |

- **结论**：MemoryLLM RULER 全档低位（single 8-31、multikey 8-30、VT≈0）——固定 stateful memory pool 对精确 needle 检索**从短档就弱**（8k single 仅 21 vs InfLLM 100 / CoMem 100），长档更差；**variable_tracking 全档≈0**（0.8/2.2/0.6/0.4/0.0）说明 pool 完全无法维护引用链。与历史 tab_h2h MemoryLLM 数（single 22/22/28/6/16、multikey 30/14/18/8/6）同量级，本次为 canonical n=100 单一 cohort。→ RULER = MemoryLLM 相对 retrieval-based（CoMem 恒~100）与 block-memory（InfLLM ≤32k 强）最悬殊的一列。

---

## 3.7 BABILong（qa1/qa2/qa5 × 0k-32k, n=100/cell, 官方 compare_answers）— ✅ COMPLETE (2026-07-19 13:0x)

- driver：`scripts/eval_memoryllm_babilong.py`（新建，ported forward `eval_memoryllm_common`，8-shard，`--max_new_tokens 20 --chunk_size 1024`，官方 `compare_answers`+`TASK_LABELS`）；memory_isolation = 每样本前从 clean snapshot 恢复 `model.memory`+`model.initialized`。
- raw：`.73:.../babilong_results/babilong_memoryllm_canonical/{task}_{len}_..._shard{0..7}of8.csv`（cols=`target,output,question`）+ `_summary_merged.json`。
- **铁律2 验证（main 独立复算，非采信 coder）**：`PYTHONPATH=third_party/babilong-pkg` 读全 168 shard CSV，逐 cell 用官方 `compare_answers(target,output,question,TASK_LABELS[task])` 重算 → 与 `_summary_merged.json` **21 cell 全部逐格完全一致**；每 cell **n=100，empty output=0**（sample output 规范如 "The most recent location of Mary is bathroom." vs target "bathroom" → 非空/well-formed，错例是**读出错值**=真检索失败，非截断/退化）。

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MemoryLLM** qa1 | 53 | 42 | 35 | 23 | 18 | 10 | 7  |
| **MemoryLLM** qa2 | 37 | 34 | 16 | 15 | 15 | 15 | 16 |
| **MemoryLLM** qa5 | 48 | 50 | 47 | 40 | 40 | 36 | 37 |
| InfLLM qa1 (对照) | 100 | 94 | 92 | 90 | 85 | 59 | 37 |
| InfLLM qa2 (对照) | 59 | 56 | 54 | 58 | 48 | 43 | 31 |
| InfLLM qa5 (对照) | 80 | 77 | 75 | 74 | 78 | 64 | 55 |

- **结论**：MemoryLLM BABILong **全档、全 task 弱于 InfLLM**。qa1（单事实 supporting）0k=53 就远低于 InfLLM 100，且随长度**单调掉到 32k=7**（FIFO 丢弃早期事实）；qa2（2-fact）0k=37→稳定低位 15-16（多事实关系本就难，pool 无法组合）；qa5（3-fact relational）0k=48 最耐长（32k 仍 37），但仍显著低于 InfLLM 55、且远低于 CoMem。→ 与 RULER/LongEval 一致：**固定 stateful memory pool 对精确/多事实检索系统性弱**，是 CoMem retrieval-based 恒定 read 的强同类对照。




`/tmp/run_memllm_qa_chain.sh`（setsid nohup，pid 1777348）：等 `logs/memllm_longeval_DONE` → LoCoMo 8-shard+score（写 `logs/memllm_locomo_DONE`）→ LongBench 8-shard+score（写 `logs/memllm_longbench_DONE`）→ `logs/memllm_qa_chain_DONE`。
LongEval 聚合器：`.73:/tmp/agg_longeval.py`（读各 length shard json 汇总 merged accuracy）。

**收尾（后续 heartbeat）**：chain DONE 后读 `locomo_results/.../` 和 `longbench_results/.../` 的 score 输出 + 再跑一次 `/tmp/agg_longeval.py`，回填本文件 §1.1 / §2 / §3 表格。收 raw 前务必 abstain-check（非空 preds）。

---

## 5. Main 独立验证 + 待办（2026-07-19 05:2x heartbeat）

**铁律2 独立复核（非采信 coder 声明）**：SSH .73 直读 `longeval_results/memoryllm_longeval_chatnothink/longeval_{8k,16k}_shard*of8.json` →
- 8k：n=50，empty_preds=**0**，correct=10，acc=**0.200**，sample pred=`'268265'`（规范 6 位数）
- 16k：n=50，empty_preds=**0**，correct=10，acc=**0.200**，sample pred=`'764209'`
→ 确认非退化/非空答，错例是"抽出错误 register 值"。LongEval 8k/16k=0.200 数字可信。

**待办（下一 heartbeat / chain DONE 后）**：
1. **回填 §1.1/§2/§3**：等 `.73:logs/memllm_qa_chain_DONE`（LongEval 32k/64k/128k + LoCoMo n=1986 + LongBench 6-ds），读 score 输出 + `/tmp/agg_longeval.py`，abstain-check 后回填。预计 LoCoMo ~6h + LongBench ~4-5h。
2. **push 未推代码（现 4 项待推）**：`.73` diskB checkout 待 commit/push——`bc48926`（scripts/qcmem_model.py import guard）+ `311778e`（src/memory/qcmem/qcmem_model.py import guard）+ `5fcfd2f`（eval_memoryllm_locomo.py + eval_memoryllm_longbench.py）+ **新 `scripts/eval_memoryllm_babilong.py`**（ported-forward-consistent BABILong driver，coder aa70b71 建，尚未 commit），author LiuHanzuo，无 *.pt/password，交 main /gitpush（需 subagent review→APPROVED→star-proxy push；从 .73 diskB checkout 推）。当前 ceph 持久无丢失风险，待 RULER+BABILong chain DONE 批量 review 后推。
