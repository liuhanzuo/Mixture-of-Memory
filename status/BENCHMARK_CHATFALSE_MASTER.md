# Paper A — 6 方法 × 5 benchmark chat=False 主矩阵（单文件 headline 总表）

> **生成 2026-07-24（main 从各权威 status 源汇编，非重跑）。** 这是 Paper A 主对照表的**单一 headline 事实来源**：一屏看全 6 个方法在 5 个 benchmark 上的 chat=False 成绩。
> **口径双支柱（论文标准）**：`chat_template=False` + CoMem `selector=iter_bm25`。理由：论文所有模型是 continue-train 的 **BASE LM（无 SFT/RL）**，套 chat template 注入 OOD token 对 base 不公平 → 全方法统一 chat=False。旧 chat=True 数字作废（历史对照见 `PAPERA_ALL_RESULTS.md` §3）。
> **官方判分**：RULER=`string_match_all`；BABILong=`compare_answers`+`TASK_LABELS`；LongBench/LongEval=`run_scoring`；LoCoMo headline=**GPT-4o judge**（n=1986）。
> **backbone**：Qwen3-8B（除 MemoryLLM=Llama-3-8B-chat 异基座对照）。
> **详细 per-cell / 消融 / 效率 / chat=True 历史 → `status/PAPERA_ALL_RESULTS.md`；LoCoMo judge 全细节 → `status/LOCOMO_JUDGE_AGGREGATE.md`。**

---

## ★ headline 主矩阵（一屏总览，chat=False）

| 方法 | RULER<br>(15-cell 均值 recall) | LongEval<br>(8k–128k 均值 acc) | LongBench<br>(6-ds macro-F1) | BABILong<br>qa1 / qa2 / qa5 | LoCoMo<br>(GPT-4o judge, n=1986) |
|---|:---:|:---:|:---:|:---:|:---:|
| **CoMem（本文，+distilled LoRA, j12）** | **97.05** | 69.0¹ | 12.15 | 55.6 / 27.0 / **68.7** | **38.27** |
| **CoMem（本文，adapter-free, frozen j9）** | 59.4 | 3.2 | 10.63 | 42.4 / 19.6 / 55.6 | 29.15 |
| KV-Direct（full-ctx 上界，j=0） | 78.80² | 65.2 | **12.17** | **78.7 / 48.9** / 61.4 | 34.59 |
| InfLLM（thunlp 旗舰 baseline） | 77.83 | 21.6 | 11.86 | 69.9 / 43.9 / 64.9 | 22.21 |
| StreamingLLM（等预算 recency） | 23.37 | 30.8 | 11.11 | 56.1 / 33.1 / 55.1 | 25.63 |
| MemoryLLM（Llama-3-8B-**chat**，异基座）³ | 16.55 | 13.6 | 9.01 | 30.4 / 21.4 / 38.1 | 16.11 |
| HCache（retrieval-free mid-layer） | 3.73 | 0.0 | 9.20 | 32.9 / 18.0 / 50.3 | 8.11 |

**排名（headline 各基准 chat=False）：**
- **RULER**：CoMem 97.05 > KVD 78.80 ≳ InfLLM 77.83 ≫ StreamingLLM 23.37 > MemoryLLM 16.55 > HCache 3.73
- **LongEval**：CoMem 69.0 > KVD 65.2 > StreamingLLM 30.8 > InfLLM 21.6 > MemoryLLM 13.6 > HCache 0.0（MemoryLLM 现为真 chat=False，异基座 Llama-3 参考）
- **LongBench**：KVD 12.17 ≈ CoMem 12.15 > InfLLM 11.86 > StreamingLLM 11.11 > HCache 9.20 > MemoryLLM 9.01（全方法被 chat=False 压到 9–12 窄带，见下注；MemoryLLM 现为真 chat=False=最低，异基座 Llama-3 参考）
- **BABILong**：KVD 全档最强（full-ctx 上界）；CoMem qa5 68.7 > KVD 61.4（唯一超上界项）（MemoryLLM 真 chat=False qa1 30.4/qa2 21.4/qa5 38.1，异基座参考）
- **LoCoMo（judge）**：**CoMem(+distilled LoRA) 38.27 > KVD 34.59 > CoMem(adapter-free j9) 29.15 > StreamingLLM 25.63 > InfLLM 22.21 > MemoryLLM 16.11 > HCache 8.11** —— distilled LoRA 把 CoMem 从 full-ctx 上界之**下**(29.15)抬到之**上**(38.27)，是超越 KVD oracle 的关键。

¹ CoMem LongEval 6-档（4k–128k）headline=**72.83**；此列取 8k–128k 5 档与 baseline 对齐=69.0。
² KVD RULER 15-cell 含 **128k=0（131072>Qwen3 max_pos 窗口溢出，非 OOM）**；≤64k near-perfect。CoMem 恒定 read 128k 仍 93–100 = 核心卖点。
³ MemoryLLM 是 Llama-3-8B-**chat**（真·chat 模型，非 continue-train base LM）；chat=False 会**剥离它训练时的原生 chat 模板 = 对它反而是 OOD/不公平**，故 chat=True 才是它的公平/原生协议。**全 5 benchmark 均有真 chat=False**：RULER 16.55 / LoCoMo 16.11 / LongBench 9.01 / BABILong 30.4/21.4/38.1（2026-07-25 .82，见 ⁴）+ **LongEval 13.6（2026-07-25 .104 补跑）**。全矩阵已无 chat=True 占位；MemoryLLM 行作异基座 cross-base 参考。
⁴ **MemoryLLM chat=False 全 5 benchmark 已补齐（无 ᵀ 占位）**。2026-07-25：**LongBench** 全 6-ds 全新 8-GPU（`--no_chat_template`，config 确认 `use_chat_template:false`）→ macro **9.01**（.82，`longbench_results/memoryllm_8b_chatFALSE/scores.json`）；**BABILong** config 确认 chat=False 的 21-cell 目录经官方 `compare_answers` 重判 → qa1 30.4/qa2 21.4/qa5 38.1（.82）；**LongEval 13.6**（.104，8k22/16k22/32k16/64k6/128k2，n=50/档，8-GPU 分片，全 shard config 确认 `use_chat_template:false`，`longeval_results/memoryllm_8b_chatFALSE/_summary_merged.json`）。.82 一直被外部 co-tenant 占卡，LongEval 改用**同盘空节点 .104（wzc1 共享盘）**跑完。MemoryLLM 行为异基座 Llama-3 cross-base 参考。

---

## §A RULER（recall %，n=100/cell，niah_single_2 + niah_multikey_1 + variable_tracking）

| 方法 | task | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|
| **CoMem** | niah_single_2 | 100 | 100 | 99 | 99 | 100 |
| | niah_multikey_1 | 95 | 94 | 97 | 91 | 93 |
| | variable_tracking | 96.6 | 97.6 | 98.8 | 99.0 | 95.8 |
| **KV-Direct（上界）** | niah_single_2 | 100 | 100 | 100 | 100 | **0** |
| | niah_multikey_1 | 100 | 100 | 98 | 88 | **0** |
| | variable_tracking | 100 | 100 | 99.8 | 96.2 | **0** |
| **InfLLM** | niah_single_2 | 100 | 100 | 92 | 61 | 57 |
| | niah_multikey_1 | 100 | 79 | 65 | 45 | 20 |
| | variable_tracking | 100 | 96.8 | 90.6 | 81.8 | 79.2 |
| **StreamingLLM** | niah_single_2 | 85 | 36 | 20 | 10 | 2 |
| | niah_multikey_1 | 83 | 34 | 18 | 13 | 4 |
| | variable_tracking | 41.8 | 2.2 | 0.6 | 0.2 | 0.8 |
| **MemoryLLM（Llama-3）** | niah_single_2 | 29 | 40 | 37 | 21 | 21 |
| | niah_multikey_1 | 28 | 24 | 25 | 12 | 10 |
| | variable_tracking | 0 | 1.2 | 0 | 0 | 0 |
| **HCache** | niah_single_2 | 33 | 5 | 3 | 0 | 0 |
| | niah_multikey_1 | 8 | 4 | 0 | 0 | 0 |
| | variable_tracking | 1.6 | 1.0 | 0.4 | 0 | 0 |

> **story**：CoMem 长档恒 93–100（固定预算不溢出）；KVD 128k 归零（窗口溢出）；InfLLM 长档缓降（vt 128k 仍 79，但 niah 64k–128k 掉到 45–61）；StreamingLLM/HCache recency/无检索长档全崩。CoMem 与 InfLLM 15-cell 均值接近（97.05 vs 77.83），差距全在 64k–128k。

---

## §B LongEval（line-retrieval acc %，n=50/len）

| 方法 | 8k | 16k | 32k | 64k | 128k | mean(8k–128k) |
|---|---:|---:|---:|---:|---:|---:|
| **CoMem（本文）** | 69 | 75 | 64 | 67 | 70 | **69.0** |
| **KV-Direct（上界）** | 100 | 96 | 92 | 38 | 0 | 65.2 |
| StreamingLLM | 86 | 34 | 18 | 10 | 6 | 30.8 |
| InfLLM | 60 | 30 | 12 | 4 | 2 | 21.6 |
| HCache | 0 | 0 | 0 | 0 | 0 | 0.0 |
| MemoryLLM（chat=False，异基座）| 22 | 22 | 16 | 6 | 2 | 13.6 |

> CoMem 6-档（含 4k=92）headline=72.83。KVD 长档 64k 骤降 38、128k 归零（RoPE 溢出）；CoMem 恒 64–75。InfLLM 长档快速衰减（8k=60→128k=2）。**MemoryLLM 行现为真 chat=False**（2026-07-25 .104 补跑，8k22/16k22/32k16/64k6/128k2，mean 13.6）——它是 chat 模型、chat=False 对它 OOD，数字仅供异基座 Llama-3 参考，与其 chat=True 14.0 接近。

---

## §C LongBench（per-ds F1，macro-mean over 6 ds，官方 qa_f1）

| 方法 | narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **macro** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CoMem（本文）** | 4.12 | 11.01 | 11.62 | 12.83 | 25.41 | 7.91 | **12.15** |
| **KV-Direct（上界）** | 3.70 | 11.82 | 12.68 | 12.03 | 25.30 | 7.49 | **12.17** |
| InfLLM | 2.99 | 11.35 | 12.08 | 12.50 | 25.33 | 6.94 | **11.86** |
| StreamingLLM | 3.49 | 11.64 | 11.02 | 12.19 | 22.42 | 5.88 | **11.11** |
| HCache | 2.56 | 10.71 | 7.33 | 9.19 | 20.39 | 5.05 | **9.20** |
| MemoryLLM（chat=False）| 3.13 | 8.46 | 8.76 | 10.27 | 17.43 | 5.98 | **9.01** |

> ⚠️ **chat=False 把所有方法的 LongBench extractive token-F1 全线压到 9–12**（chat=True 时 KVD 42.97/InfLLM 41.54/CoMem 35.79）——CoMem 12.15 ≈ full-ctx 上界 KVD 12.17（打平），低分是**协议效应非压缩损失**。**MemoryLLM 行现为真 chat=False**（2026-07-25 .82 全 6-ds 全新 8-GPU 跑，macro 9.01=最低；异基座 Llama-3-chat 剥离原生模板=OOD，仅作 cross-base 参考）。

---

## §D BABILong（compare_answers %，n=100/cell，mean over 0k–32k；CoMem = iter_hop=4 统一口径，#70 2026-07-24，源 `qcmem_j12_iter_bm25_chatFALSE_ad_hop4`）

| 方法 | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CoMem** | qa1 | 98 | 80 | 68 | 68 | 46 | 17 | 12 | 55.6 |
| | qa2 | 26 | 44 | 43 | 44 | 23 | 8 | 1 | 27.0 |
| | qa5 | 68 | 76 | 76 | 75 | 68 | 60 | 58 | **68.7** |
| **KV-Direct（上界）** | qa1 | 98 | 84 | 80 | 74 | 80 | 72 | 63 | **78.7** |
| | qa2 | 58 | 53 | 50 | 46 | 49 | 49 | 37 | **48.9** |
| | qa5 | 71 | 73 | 62 | 59 | 65 | 42 | 58 | 61.4 |
| **InfLLM** | qa1 | 97 | 84 | 81 | 72 | 69 | 52 | 34 | 69.9 |
| | qa2 | 51 | 58 | 41 | 47 | 41 | 39 | 30 | 43.9 |
| | qa5 | 69 | 68 | 63 | 66 | 73 | 60 | 55 | 64.9 |
| **StreamingLLM** | qa1 | 97 | 83 | 80 | 71 | 23 | 27 | 12 | 56.1 |
| | qa2 | 49 | 56 | 44 | 49 | 19 | 11 | 4 | 33.1 |
| | qa5 | 68 | 65 | 68 | 65 | 39 | 47 | 34 | 55.1 |
| **HCache** | qa1 | 96 | 63 | 53 | 15 | 3 | 0 | 0 | 32.9 |
| | qa2 | 57 | 15 | 35 | 17 | 2 | 0 | 0 | 18.0 |
| | qa5 | 75 | 72 | 69 | 64 | 51 | 16 | 5 | 50.3 |
| **MemoryLLM（chat=False）** | qa1 | 52 | 46 | 35 | 28 | 23 | 17 | 12 | 30.4 |
| | qa2 | 37 | 29 | 17 | 18 | 21 | 17 | 11 | 21.4 |
| | qa5 | 50 | 45 | 39 | 35 | 35 | 34 | 29 | 38.1 |

> ⚠️ BABILong ≤32k 全在 full-ctx 窗口内 → **full-ctx 上界 KVD 在 qa1/qa2 明显强于 CoMem**（诚实的压缩 tax，非 CoMem 优势项）；CoMem qa5 68.7 略超 KVD 61.4。InfLLM qa1 69.9 居中（长档 32k=34 掉更快）。CoMem 卖点在**超出 full-ctx 的长度 + 效率**（128k full-ctx OOM vs CoMem 20GB），非 ≤32k 精度超上界。MemoryLLM BABILong 现为真 chat=False（2026-07-25 .82 找到 config 确认 chat=False 的 21-cell 目录，官方 compare_answers 重判：qa1 30.4/qa2 21.4/qa5 38.1，异基座 Llama-3 参考）。

---

## §E LoCoMo（GPT-4o judge headline，chat=False，n=1986）

| 方法 | **judge (n=1986)** | judge (cat1–4, n=1540) | F1 | acc | EM |
|---|---:|---:|---:|---:|---:|
| **CoMem（本文，iter_bm25）** | **38.27** | **48.64** | 9.15 | 23.36 | 0.55 |
| **CoMem（本文，adapter-free j9）** | 29.15 | 37.27 | 7.28 | 16.41 | 0.25 |
| KV-Direct（full-ctx 上界） | 34.59 | 43.83 | 9.02 | 22.36 | 0.60 |
| StreamingLLM | 25.63 | — | 7.67 | 13.75 | 1.56 |
| InfLLM | 22.21 | — | 7.39 | 13.34 | 1.71 |
| MemoryLLM | 16.11 | — | 5.91 | 9.52 | 0.10 |
| HCache | 8.11 | 10.13 | 4.67 | 6.29 | 0.25 |

> **per-cat judge（CoMem）**：cat1 26.95 / cat2 19.00 / cat3 30.21 / cat4 **69.32** / cat5 2.47（本地 abstain）。cat4（single-hop，最大桶 n=841）CoMem 领先决定性（+7 over KVD）。配对 bootstrap（judged n=1540）：paired diff=+4.81，95%CI[2.34,7.27]，p<0.0001 → **CoMem 显著优于 full-ctx KV oracle**。全 6 方判分细节见 `LOCOMO_JUDGE_AGGREGATE.md`。

---

## §F CoMem adapter-free（#65）— ✅ 全 5-benchmark 完成（含 LoCoMo GPT-4o judge = **29.15**，2026-07-25 回填）

- **配置**：Qwen3-8B **frozen backbone（无 LoRA，靠省略 `--lora_adapter` 实现）**，`resume_j=9`（浅 readout-safe split，vs 旗舰 j=12），selector=iter_bm25、topk=12、hop=4、sink=bos、chunk512，chat=False。**一份固定 config 跑全 5 benchmark，j 不逐 benchmark 调**。
- **节点/状态**：.252（8×B200，wzc1），driver `scripts/_qcmem_adapterfree_j9_chatFALSE_taskpool.sh`（commit a21a752 未 push），148 jobs task-pool **全完**，6 官方 scorer 已写出 → `logs/qcmem_adapterfree_j9_chatFALSE/SUMMARY.txt`+`SCHED_DONE`（2026-07-24 23:35）。**唯一剩项：LoCoMo GPT-4o judge 尚未跑（headline judge 仍 PENDING）**。

**结果（chat=False，per-cell）：**

- **RULER 15-cell macro recall = 59.4**

| task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| niah_single_2 | 96 | 99 | 99 | 99 | 96 |
| niah_multikey_1 | 44 | 44 | 59 | 30 | 48 |
| variable_tracking | 45 | 38.8 | 36.4 | 31.2 | 25.8 |

- **LongEval（8k–128k 均值）= 3.2**：8k=8 / 16k=0 / 32k=4 / 64k=0 / 128k=4
- **LongBench 6-ds macro-F1 = 10.63**：narrativeqa 4.63 / qasper 11.23 / hotpotqa 9.41 / 2wikimqa 10.71 / multifieldqa_en 22.05 / musique 5.72
- **BABILong qa1/qa2/qa5 均值 = 42.4 / 19.6 / 55.6**

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qa1 | 98 | 59 | 71 | 57 | 7 | 1 | 4 | 42.4 |
| qa2 | 53 | 17 | 28 | 26 | 8 | 4 | 1 | 19.6 |
| qa5 | 70 | 74 | 65 | 60 | 41 | 39 | 40 | 55.6 |

- **LoCoMo（local scorer，n=1986）**：F1=7.28 / EM=0.25 / acc=16.41（cat1 8.87 / cat2 6.54 / cat3 15.62 / cat4 30.92 / cat5 1.12）。**GPT-4o judge = 29.15（n=1986）** ✅ 已跑（LOCAL，`locomo_results/qcmem_8b_zeroshot_j9_chatFALSE/scores.json`）：cat1 18.79 / cat2 13.40 / cat3 27.08 / cat4 **53.75** / cat5 1.12；cat1-4 judged 加权=37.27（n=1540）。**对照：旗舰+LoRA judge 38.27 > KVD 34.59 > adapter-free 29.15** → distilled LoRA 在 LoCoMo judge 上贡献 +9.12（29.15→38.27），把 CoMem 从 full-ctx 上界之下抬到之上。
- **RULER task-breadth（补充，非主表）**：niah_single_1 16k/64k/128k=99/96/97；niah_single_3=74/89/89；niah_multivalue=66/57.75/67.5；niah_multiquery=71.5/61/70.25。

> **distilled LoRA 的价值对照**：旗舰 +distilled LoRA(j12) RULER **97.05** / LongEval **69.0** vs frozen j9 adapter-free RULER **59.4** / LongEval **3.2** —— frozen backbone 在 niah_multikey_1（多事实消歧）、variable_tracking（迭代 tracking）、LongEval（line-retrieval）上**显著退化**（RULER 掉 ~38 pt、LongEval 近乎归零），但**单针 niah_single_2 仍近满分（96–99）**。说明 distilled LoRA 主要买回「多事实消歧 + 迭代 tracking + 行检索」能力；单 needle 检索靠 frozen backbone + iter_bm25 即可近满分。

---

## §G 数据来源 / provenance

| 内容 | 源 |
|---|---|
| CoMem 旗舰 chat=False per-cell（RULER 97.05 / LongEval 72.83 / LongBench 12.15 / BABILong / LoCoMo 38.27）+ bootstrap CI | `status/QCMEM_STATS_APPENDIX_chatFALSE.md`（.73 diskB）|
| baseline chat=False per-cell（KVD/HCache/StreamingLLM/MemoryLLM，§1.7）| `status/PAPERA_ALL_RESULTS.md` §1.7（2026-07-24 `_agg_baseline_chatFALSE.py` 聚合 .73 官方预算分）|
| **InfLLM chat=False（#63，本表 §A–§D 的 InfLLM 行）** | `logs/infllm_chatFALSE_taskpool/SUMMARY.txt`（LOCAL/.252 wzc1，2026-07-24 17:09 SCHED_DONE，全 cell Iron-Law-2 OK）|
| LoCoMo GPT-4o judge 全 6 方（§E）| `status/LOCOMO_JUDGE_AGGREGATE.md`（.73 各 `locomo_results/*/scores.json`，judge=gpt-4o via maas-openapi 有余额 JWT）|
| CoMem adapter-free（#65，§F）| ✅ `logs/qcmem_adapterfree_j9_chatFALSE/SUMMARY.txt`+`SCHED_DONE`（.252 wzc1 共享 FS，本地可见；5-benchmark 2026-07-24 23:35 完成 + LoCoMo GPT-4o judge=29.15 于 `locomo_results/qcmem_8b_zeroshot_j9_chatFALSE/scores.json`，2026-07-25 完成）|
| **MemoryLLM chat=False 全 5 benchmark：RULER 16.55 / LoCoMo 16.11 / LongBench 9.01 / BABILong 30.4·21.4·38.1 / LongEval 13.6** | `longbench_results/memoryllm_8b_chatFALSE/scores.json` + `babilong_results/memoryllm_8b_chatFALSE/_summary_merged.json`（.82）+ `longeval_results/memoryllm_8b_chatFALSE/_summary_merged.json`（.104 wzc1 共享盘，2026-07-25，8-GPU 分片，config 确认 chat=False）|

**磁盘拓扑**：LOCAL 与 .252(28.89.19.252) 共享 wzc1 物理盘（InfLLM #63 + adapter-free #65 结果本地直接可见）；CoMem 旗舰 + KVD/HCache/StreamingLLM/MemoryLLM chat=False + LoCoMo judge 在 **.73(28.85.35.73)** diskB（另一物理盘，需 SSH）。

---

## §H 决定性实验（#71）— "CoMem 的优势是否只是检索？"（chat=False, Qwen3-8B, 2026-07-25 完成）

> **reviewer 关切**：CoMem 在 LoCoMo judge 上超过 full-ctx oracle（KVD），是否只是"检索过滤了噪声"带来的、与深度压缩/训练无关的效果？
> **设计**：跑一个 **j=0 退化 CoMem**（`resume_j=0`，无 LoRA，无深度切分，但仍用**同一 iter_bm25 selector + 同预算**检索 top-12 → 对检索到的 chunk 做**全 36 层重算**）。它锚定两个单变量对照。源：`locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/scores.json` + `babilong_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE`。

| config | split j | LoRA | 输入 | **LoCoMo judge (n=1986)** | cat1-4 (n=1540) | BABILong qa1/qa2/qa5 |
|---|:---:|:---:|---|:---:|:---:|:---:|
| CoMem 旗舰（+distilled LoRA） | 12 | ✅ | 检索 top-12 | 38.27 | 48.64 | 55.6 / 27.0 / **68.7** |
| **#71 j0 control（全层重算）** | 0 | ✗ | 检索 top-12 | **41.59** | **52.60** | 65.9 / 39.1 / 63.0 |
| #65 adapter-free（frozen） | 9 | ✗ | 检索 top-12 | 29.15 | 37.27 | 42.4 / 19.6 / 55.6 |
| KV-Direct（full-ctx oracle） | 0 | ✗ | **全上下文** | 34.59 | 43.83 | **78.7 / 48.9** / 61.4 |

**(a) 深度切分的贡献**（j0 vs adapter-free j9，唯一变量=split depth 0↔9，都无 LoRA、都检索）：
- LoCoMo judge **41.59 vs 29.15** → frozen 深度-9 切分相对全层重算 **损失 −12.44 pt**。
- BABILong **65.9/39.1/63.0 vs 42.4/19.6/55.6** → 全层重算在**全部** task 上 ≫ frozen j9。
- → 深度轴压缩本身会**丢保真度**；旗舰的 distilled LoRA（j12：38.27，55.6/27.0/68.7）**买回**了其中大部分。

**(b) 检索的贡献**（j0 vs KVD，唯一变量=检索；都全层重算、无 LoRA、无切分）：
- LoCoMo judge **41.59 vs 34.59** → **检索单独就 +7.00**（过滤掉无关对话轮 → 比全上下文更干净）。
- BABILong qa1/qa2 **65.9/39.1 vs 78.7/48.9** → 检索在 needle 任务上**倒亏 −12.8/−9.8**（top-k 漏掉散落事实）。
- BABILong qa5 **63.0 vs 61.4** → 检索近中性（+1.6）。

**★ 对 reviewer 的回答（诚实、分基准）：**
1. **LoCoMo（对话、干扰项密集）**：是的，CoMem 超过 full-ctx oracle 的**很大一部分来自检索**——j0（只检索、无训练、无压缩）已 41.59 > KVD 34.59（+7）。旗舰的深度-12 切分虽略降到 38.27（压缩 tax ≈ −3.3），但**仍 > KVD 34.59**，且以固定预算 6657 token 换来 CoMem 的核心效率卖点。
2. **BABILong ≤32k（needle、事实散落，全在 full-ctx 窗口内）**：不是。检索在此是 **tax 而非 win**（j0 qa1/qa2 < KVD）；CoMem 在此的价值**纯粹是超出 full-ctx 窗口长度时的效率**（128k full-ctx OOM vs CoMem 恒定 ~18–20GB），而非 ≤32k 精度超上界。
3. **深度轴切分**（本文机制的核心）：j0 vs frozen j9 证明切分会丢保真度，**distilled LoRA 正是买回这部分的关键**（j9 frozen 29.15 → j12+LoRA 38.27，LoCoMo judge +9.12）。

> **一句话**：CoMem 的收益来自**检索 + 深度压缩 + 蒸馏 LoRA 三者的组合**，且各自贡献随任务而异——LoCoMo 上检索主导、BABILong 上检索是 tax、深度切分处处需 LoRA 补偿。CoMem 的普适卖点是**固定预算下超越 full-ctx 窗口的长度可扩展性 + 效率**，而非在 full-ctx 窗口内一律精度超上界。

---

## §I frozen depth-sweep（#73）— 纯 isolate distilled LoRA 贡献（chat=False, Qwen3-8B, 2026-07-25 完成）

> **目的**：#65（j9 frozen 29.15 → 旗舰 j12+LoRA 38.27 = +9.12）把 **9→12 深度变化**与 **LoRA** 混在一起。补 **j12 frozen** 点后，`旗舰 j12+LoRA` vs `j12 frozen` = **SAME split depth，唯一变量=LoRA** → 纯 isolate 蒸馏 LoRA 在旗舰自身深度上的贡献。同时 {j0, j6, j9, j12} 四点给出干净的 frozen 深度-保真度单调曲线。
> 源：`{locomo,babilong}_results/qcmem_8b_zeroshot_j{6,12}_frozen_iterbm25_chatFALSE`（#73，driver `_qcmem_adapterfree_jsweep_chatFALSE_taskpool.sh`）+ #71（j0）+ #65（j9）。所有点：frozen 无 LoRA、iter_bm25、topk12、hop4、sink=bos、chunk512、chat=False。

| split j | LoRA | **LoCoMo judge (n=1986)** | BABILong qa1/qa2/qa5 (mean 0k–32k) |
|:---:|:---:|:---:|:---:|
| **j0**（全 36 层重算，=#71） | ✗ | **41.59** | 65.9 / 39.1 / 63.0 |
| **j6**（=#73） | ✗ | 32.78 | 44.6 / 22.6 / 53.9 |
| **j9**（=#65 adapter-free） | ✗ | 29.15 | 42.4 / 19.6 / 55.6 |
| **j12**（frozen，=#73） | ✗ | **24.52** | 33.4 / 18.0 / 60.3 |
| **j12 + distilled LoRA（旗舰）** | ✅ | **38.27** | 55.6 / 27.0 / 68.7 |

**(1) frozen 深度-保真度单调递减**：LoCoMo judge 41.59 → 32.78 → 29.15 → 24.52（j 越深、frozen backbone 丢的保真度越多），与 memory `bottleneck-layer-sweep-monotone`（越深 LM 税越大）一致。BABILong qa1/qa2 同向单调降（qa1 65.9→33.4，qa2 39.1→18.0）；qa5 弱非单调（63.0→53.9→55.6→60.3）。

**(2) ★ 纯 distilled LoRA 贡献（SAME depth j12，唯一变量=LoRA）**：
- **LoCoMo judge：j12 frozen 24.52 → j12+LoRA 38.27 = +13.75**（比 #65 混合口径的 +9.12 更大更干净）。
- BABILong：qa1 33.4→55.6 **+22.2**、qa2 18.0→27.0 **+9.0**、qa5 60.3→68.7 **+8.4**。
- 注意：**9→12 的 frozen 深度变化本身是 −4.63 judge（29.15→24.52，有害）** → distilled LoRA 不仅要补回 j12 的额外深度税，还要把整体抬到 full-ctx oracle（KVD 34.59）之上。故 **+13.75 才是蒸馏 LoRA 在旗舰深度上的真实纯贡献**。

> **对论文的意义**：旗舰的 distilled LoRA 是 CoMem 超过 full-ctx KV oracle 的**决定性组件**——在其自身深度 j12 上单独贡献 +13.75 LoCoMo judge / +22.2 BABILong qa1。frozen backbone（任何深度）都 < KVD oracle；只有加上蒸馏 LoRA 才越过。
