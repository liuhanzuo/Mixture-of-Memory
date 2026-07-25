# Paper A（QCMem / CoMem）全套结果汇总 — 单文件总账

> **生成 2026-07-24（main 从各权威 status 源文件汇编，非重跑）。** 目的：把 Paper A 现有所有结果放进一个文件，便于横向查阅。
> **口径双支柱（论文标准）**：`selector=iter_bm25` + `chat_template=False`。理由：论文所有模型是 continue-train 的 **BASE LM（无 SFT/RL）**，套 chat template 注入 OOD token 对 base 不公平 → **所有方法统一 chat=False**。旧 chat=True（`*_chatnothink`）数字**作废**，仅作历史对照保留在 §3。
> **官方判分**：RULER=`string_match_all`；BABILong=`compare_answers`+`TASK_LABELS`（禁 re.search）；LongBench/LongEval=`run_scoring`；LoCoMo headline=GPT-4o judge。全 chat=False cell 通过 Iron-Law-2（8/8 shard、empty=0、复算 0 mismatch）。
> **backbone**：Qwen3-8B（旗舰）；例外：MemoryLLM=Llama-3-8B-chat（异基座，同类 stateful-memory 对照）。
> **CoMem flagship 配置**：Qwen3-8B + LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`，resume_j=12，selector=iter_bm25，topk=12，sink=bos，chunk1024。

---

## §0 覆盖矩阵（权威，T=chat=True / F=chat=False；2026-07-24 SSH 实地清点）

| 方法 | RULER | LongBench | LongEval | LoCoMo | BABILong |
|---|---|---|---|---|---|
| **CoMem（本文）** | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| KV-Direct（full-ctx 上界） | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| HCache | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| StreamingLLM（等预算） | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| MemoryLLM（Llama-3-**chat**） | T✅ F✅ | T✅ **F△→ᵀ12.80** | T✅ **F❌→ᵀ14.0** | T✅ F✅(judge 16.11) | T✅ **F❌→ᵀ26.9/21.1/42.6** |
| **InfLLM（旗舰 baseline）** | T✅ **F✅** | T✅ **F✅** | T✅ **F✅** | T✅ F✅ | T✅ **F✅** |

> **📌 单文件 headline 主矩阵（一屏看全 6 方法 × 5 benchmark）→ `status/BENCHMARK_CHATFALSE_MASTER.md`**（本文件是 per-cell + 消融 + chat=True 历史的详账）。

- **chat=True：30/30 全齐**（数字见 §3）。
- **chat=False（论文口径，per-cell 全部回填，见 §1.7）**：CoMem/KV-Direct/HCache/StreamingLLM/**InfLLM** **五 benchmark 全齐**（InfLLM chat=False #63 已于 2026-07-24 17:09 完成，全 cell Iron-Law-2 OK，数字见 §1.7 各表 InfLLM 行）；MemoryLLM 部分残缺（RULER✅ 16.55、LoCoMo✅ judge 16.11、LongBench 仅 narrativeqa、LongEval 无、BABILong 目录误命名实为 chat=True）——MemoryLLM 是 Llama-3-8B-**chat**（真·chat 模型非 base LM），chat=False 剥离其原生模板=对它 OOD/不公平。**用户 2026-07-24 决定「两者都要」**：(a) 现用 §3.2 chat=True 数字**立即填满** master 矩阵 ᵀ 格（LongEval 14.0/LongBench 12.80/BABILong 26.9·21.1·42.6，标 ᵀ 且不进 chat=False 排名）；(b) 待 diskB 节点空出跑 MemoryLLM chat=False LongEval/LongBench(6-ds)/BABILong 覆盖成双行=**pending #68**。
- **LoCoMo GPT-4o judge headline（有余额 JWT 全跑）**：CoMem 38.27 / KV-Direct 34.59 / StreamingLLM 25.63 / InfLLM 22.21 / MemoryLLM 16.11 / HCache 8.11（全 6 方，见 §1.7e + `LOCOMO_JUDGE_AGGREGATE.md`）。
- **LoCoMo GPT-4o judge headline（有余额 JWT 全跑）**：CoMem 38.27 / KV-Direct 34.59 / StreamingLLM 25.63 / InfLLM 22.21 / MemoryLLM 16.11 / HCache 8.11（全 6 方，见 §1.7e + `LOCOMO_JUDGE_AGGREGATE.md`）。
- **pending**：(1) ✅ **CoMem adapter-free（frozen j9）chat=False 全 5-benchmark 已完成回填（#65，SUMMARY 2026-07-24 23:35，148 jobs 全完）**：RULER 59.4 / LongEval 3.2 / LongBench 10.63 / BABILong 42.4·19.6·55.6 / LoCoMo local acc 16.41（**GPT-4o judge 待补**）→ 见 `BENCHMARK_CHATFALSE_MASTER.md` §F 主表 `CoMem (adapter-free)` 行；(2) **MemoryLLM chat=False LongEval/LongBench(6-ds)/BABILong（#68，需 diskB，当前 ᵀ chat=True 占位）** → 覆盖成双行。

---

# §1 chat=False 结果（论文口径，权威）

## §1.1 CoMem 旗舰 headline（chat=False，iter_bm25）
| benchmark | 分数 | 定义 |
|---|---:|---|
| RULER（niah 主体 15-cell 均值） | **97.05** | 简单算术均值 over 3 task × 5 len(8k–128k)；见 §1.2 |
| LongEval（6 档均值） | **72.83** | 4k–128k register-content acc，n=100/len |
| LongBench（6-ds macro-F1） | **12.15** | ⚠️ chat=False 对 extractive token-F1 QA 大幅偏低（chat=True=35.79）；micro-F1=11.57 |
| LoCoMo（GPT-4o judge，n=1986） | **38.27** | headline；judged-only cat1–4 n=1540=48.64 |
| BABILong qa1 / qa2 / qa5（0k–32k 均值） | **55.6 / 27.0 / 68.7** | n=100/cell，官方 compare_answers（iter_hop=4 统一口径，#70） |

## §1.2 RULER 主体（CoMem chat=False，15-cell，n=100/cell）
源 `ruler_results/qcmem_8b_iter_chatFALSE_ad`（.73/diskB）。
| task | 8k | 16k | 32k | 64k | 128k | (256k) |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_2 | 100.0 | 100.0 | 99.0 | 99.0 | 100.0 | (96.0) |
| niah_multikey_1 | 95.0 | 94.0 | 97.0 | 91.0 | 93.0 | (91.0) |
| variable_tracking | 96.6 | 97.6 | 98.8 | 99.0 | 95.8 | (99.0) |

**97.05 = 15 cell(8k–128k)简单均值**（256k 排除；含 256k 的 18-cell 均值=90.99，勿混）。

## §1.3 LongEval（CoMem chat=False，n=100/len）
源 `longeval_results/qcmem_8b_iter_chatFALSE`。
| 4k | 8k | 16k | 32k | 64k | 128k | mean |
|---:|---:|---:|---:|---:|---:|---:|
| 92.0 | 69.0 | 75.0 | 64.0 | 67.0 | 70.0 | **72.83** |

## §1.4 LongBench（CoMem chat=False，per-ds F1，n-weighted）
源 `longbench_results/qcmem_8b_iter_chatFALSE`。
| narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **macro-mean** |
|---:|---:|---:|---:|---:|---:|---:|
| 4.12 | 11.01 | 11.62 | 12.83 | 25.41 | 7.91 | **12.15** |

## §1.5 BABILong（CoMem chat=False，n=100/cell，官方 compare_answers）
源 `babilong_results/qcmem_j12_iter_bm25_chatFALSE_ad_hop4`（iter_hop=4 统一口径，#70 2026-07-24；旧 hop=2 目录 `_ad` 的 53.6/25.6/66.7 已作废替换）。
| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qa1 | 98 | 80 | 68 | 68 | 46 | 17 | 12 | 55.6 |
| qa2 | 26 | 44 | 43 | 44 | 23 | 8 | 1 | 27.0 |
| qa5 | 68 | 76 | 76 | 75 | 68 | 60 | 58 | 68.7 |

## §1.6 LoCoMo（chat=False，GPT-4o judge + 官方 scorer，n=1986）— headline 对照
源 `locomo_results/qcmem_8b_iter_chatFALSE`（CoMem）+ `kvdirect_8b_chatFALSE`（KV-Direct 上界）。
| method | **judge (n=1986)** | judge (cat1–4, n=1540) | F1 | acc | EM |
|---|---:|---:|---:|---:|---:|
| **CoMem（iter_bm25）** | **38.27** | **48.64** | 9.15 | 23.36 | 0.55 |
| KV-Direct（full-ctx 上界） | 34.59 | 43.83 | 9.02 | 22.36 | 0.60 |
| **Δ (CoMem − KVD)** | **+3.68** | **+4.81** | +0.14 | +1.00 | −0.05 |

- **per-cat judge（CoMem）**：cat1 multi_hop 26.95 / cat2 single_hop 19.00 / cat3 temporal 30.21 / cat4 open_domain 69.32 / cat5 adversarial 2.47(本地 abstain)。
- **配对显著性**（judged n=1540，10000-resample bootstrap seed1234）：**paired diff=+4.81，95%CI[2.34,7.27]，p<0.0001，P(CoMem>KVD)=1.0 → CoMem 显著优于 full-ctx KV oracle**（unpaired CI 重叠是配对设计 power artifact）。
- judge endpoint=maas `gpt-4o`（seed=1，无 client 端 dated snapshot），prompt 全文见 `QCMEM_STATS_APPENDIX_chatFALSE.md` §1d。cat5(n=446)不送 judge，本地 abstention folded 进 headline（非丢弃）。
- ⚠️ 旧值 F1 19.51/EM 5.99/acc 28.65 = **chat=True**，作废。

## §1.7 baseline chat=False（完整 per-cell，2026-07-24 从 diskB 官方预算分聚合）
> 2026-07-24 用 `scripts/_agg_baseline_chatFALSE.py` 只读聚合 diskB `.73`（与 .82 共享 FS）各 `*_chatFALSE/` 的 per-shard json（官方 scorer 预写分：RULER=string_match_all `summary.score`、LongBench=`f1`、LongEval=`correct/total`、BABILong=`compare_answers+TASK_LABELS` `score`），8/8 shard 加权，n=100/cell（LongEval/LongBench 除外，见下）。全部与 CoMem 同口径同判分。

### §1.7a RULER（recall %，n=100/cell）
| method | task | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|
| **CoMem（本文，参考）** | niah_single_2 | 100 | 100 | 99 | 99 | 100 |
| | niah_multikey_1 | 95 | 94 | 97 | 91 | 93 |
| | variable_tracking | 96.6 | 97.6 | 98.8 | 99.0 | 95.8 |
| **KV-Direct（full-ctx 上界）** | niah_single_2 | 100 | 100 | 100 | 100 | **0** |
| | niah_multikey_1 | 100 | 100 | 98 | 88 | **0** |
| | variable_tracking | 100 | 100 | 99.8 | 96.2 | **0** |
| **StreamingLLM（等预算）** | niah_single_2 | 85 | 36 | 20 | 10 | 2 |
| | niah_multikey_1 | 83 | 34 | 18 | 13 | 4 |
| | variable_tracking | 41.8 | 2.2 | 0.6 | 0.2 | 0.8 |
| **HCache** | niah_single_2 | 33 | 5 | 3 | 0 | 0 |
| | niah_multikey_1 | 8 | 4 | 0 | 0 | 0 |
| | variable_tracking | 1.6 | 1.0 | 0.4 | 0 | 0 |
| **MemoryLLM（Llama-3）** | niah_single_2 | 29 | 40 | 37 | 21 | 21 |
| | niah_multikey_1 | 28 | 24 | 25 | 12 | 10 |
| | variable_tracking | 0 | 1.2 | 0 | 0 | 0 |
| **InfLLM（旗舰 baseline，#63）** | niah_single_2 | 100 | 100 | 92 | 61 | 57 |
| | niah_multikey_1 | 100 | 79 | 65 | 45 | 20 |
| | variable_tracking | 100 | 96.8 | 90.6 | 81.8 | 79.2 |
> InfLLM 15-cell 均值=77.83（≈KVD 78.80），差距全在 64k–128k（niah 长档掉 45–61，但 vt 128k 仍 79.2）；CoMem 97.05 长档恒定优势明显。
> KV-Direct 8k–64k near-perfect，**128k=0=窗口溢出（131072>Qwen3 max_pos，非 OOM）**；CoMem 128k 仍 93–100=固定预算不溢出的核心优势。KVD `no_retrieval=true`（full-ctx，无 selector），errata §8c 的 bm25/iter_bm25 之争对 KVD/HCache 不适用（二者非 selector-based）。

### §1.7b LongBench（per-ds F1，macro-mean over 6 ds）
| method | narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **macro** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CoMem（本文）** | 4.12 | 11.01 | 11.62 | 12.83 | 25.41 | 7.91 | **12.15** |
| **KV-Direct（上界）** | 3.70 | 11.82 | 12.68 | 12.03 | 25.30 | 7.49 | **12.17** |
| StreamingLLM | 3.49 | 11.64 | 11.02 | 12.19 | 22.42 | 5.88 | **11.11** |
| InfLLM（旗舰 baseline，#63） | 2.99 | 11.35 | 12.08 | 12.50 | 25.33 | 6.94 | **11.86** |
| HCache | 2.56 | 10.71 | 7.33 | 9.19 | 20.39 | 5.05 | **9.20** |
| MemoryLLM | 4.09 | — | — | — | — | — | (仅 narrativeqa) |
> ⚠️ **chat=False 把所有方法的 LongBench extractive token-F1 QA 全线压到 9–12**（chat=True 时 KVD 42.97/InfLLM 41.54/CoMem 35.79，见 §3.4）——**CoMem 12.15 ≈ full-ctx 上界 KVD 12.17（打平）**，说明 chat=False 下 LongBench 的低分是协议效应非压缩损失。⚠️ MemoryLLM 只有 narrativeqa（#50 标非论文格，其余 5 ds chat=False 未跑）→ master 用 **ᵀ 12.80**（chat=True 6-ds 占位，#68 待跑 chat=False）。

### §1.7c LongEval（accuracy %，8k–128k，n=50/len）
| method | 8k | 16k | 32k | 64k | 128k | mean(8k–128k) |
|---|---:|---:|---:|---:|---:|---:|
| **CoMem（本文，8k–128k）** | 69 | 75 | 64 | 67 | 70 | **69.0** |
| **KV-Direct（上界）** | 100 | 96 | 92 | 38 | 0 | 65.2 |
| StreamingLLM | 86 | 34 | 18 | 10 | 6 | 30.8 |
| InfLLM（旗舰 baseline，#63） | 60 | 30 | 12 | 4 | 2 | 21.6 |
| HCache | 0 | 0 | 0 | 0 | 0 | 0.0 |
| MemoryLLM | **F❌ 无目录**（chat=False LongEval 未跑）→ **ᵀ 14.0**（chat=True 占位，#68 待跑 chat=False）| | | | | |
> CoMem 6-档含 4k=92 → headline **72.83**（§1.3）；此处取 8k–128k 匹配 baseline 起档 = **69.0 > KVD 65.2**（长档 KVD 64k 骤降 38、128k 归零，CoMem 恒 64–75）。

### §1.7d BABILong（compare_answers %，n=100/cell，mean over 0k–32k）
| method | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CoMem（本文）** | qa1 | 98 | 80 | 68 | 68 | 46 | 17 | 12 | 55.6 |
| | qa2 | 26 | 44 | 43 | 44 | 23 | 8 | 1 | 27.0 |
| | qa5 | 68 | 76 | 76 | 75 | 68 | 60 | 58 | 68.7 |
| **KV-Direct（上界）** | qa1 | 98 | 84 | 80 | 74 | 80 | 72 | 63 | **78.7** |
| | qa2 | 58 | 53 | 50 | 46 | 49 | 49 | 37 | **48.9** |
| | qa5 | 71 | 73 | 62 | 59 | 65 | 42 | 58 | 61.4 |
| StreamingLLM | qa1 | 97 | 83 | 80 | 71 | 23 | 27 | 12 | 56.1 |
| | qa2 | 49 | 56 | 44 | 49 | 19 | 11 | 4 | 33.1 |
| | qa5 | 68 | 65 | 68 | 65 | 39 | 47 | 34 | 55.1 |
| HCache | qa1 | 96 | 63 | 53 | 15 | 3 | 0 | 0 | 32.9 |
| | qa2 | 57 | 15 | 35 | 17 | 2 | 0 | 0 | 18.0 |
| | qa5 | 75 | 72 | 69 | 64 | 51 | 16 | 5 | 50.3 |
| **InfLLM（旗舰 baseline，#63）** | qa1 | 97 | 84 | 81 | 72 | 69 | 52 | 34 | 69.9 |
| | qa2 | 51 | 58 | 41 | 47 | 41 | 39 | 30 | 43.9 |
| | qa5 | 69 | 68 | 63 | 66 | 73 | 60 | 55 | 64.9 |
| MemoryLLM | — | ❌ 见下注 | | | | | | | — |
> ⚠️ BABILong ≤32k 全在 full-ctx 窗口内 → **full-ctx 上界 KVD 在 qa1/qa2 明显强于 CoMem**（qa1 78.7 vs 55.6、qa2 48.9 vs 27.0=可容纳长度下压缩的固有 tax）；CoMem qa5 68.7 略超 KVD 61.4。这是诚实的压缩 tax，非 CoMem 优势项——CoMem 卖点在**超出 full-ctx 的长度 + 效率**（§2.6 128k full OOM），非 ≤32k 精度超上界。（CoMem 行 = iter_hop=4 统一口径，#70）
> ⚠️ **MemoryLLM BABILong chat=False 不采用**：`babilong_results/memoryllm_8b_chatFALSE/` 实测文件名为 `chat_template_yes_system_prompt_no`（**误命名，实为 chat=True**）且 json 无预计算 score 字段 → 非有效 chat=False。master 用 §3.2 chat=True **ᵀ qa1 26.9/qa2 21.1/qa5 42.6** 占位（#68 待跑真 chat=False）。**MemoryLLM 本身是 Llama-3-8B-chat（真·chat 模型非 base LM），chat=False 剥离其原生模板=对它 OOD/不公平，chat=True 才是其原生基线。**

### §1.7e LoCoMo（GPT-4o judge headline + 官方 F1/acc/EM，n=1986）
| method | judge | F1 | acc | EM | cat1–4 judge (n1540) |
|---|---:|---:|---:|---:|---:|
| **CoMem（本文）** | **38.27** | 9.15 | 23.36 | 0.55 | 48.64 |
| **KV-Direct（上界）** | 34.59 | 9.02 | 22.36 | 0.60 | 43.83 |
| StreamingLLM | 25.63 | 7.67 | 13.75 | 1.56 | — |
| InfLLM | 22.21 | 7.39 | 13.34 | 1.71 | — |
| MemoryLLM | 16.11 | 5.91 | 9.52 | 0.10 | — |
| HCache | 8.11 | 4.67 | 6.29 | 0.25 | 10.13 |
> **GPT-4o judge 全 6 方已跑齐**（有余额 JWT，2026-07-24 #64 回填 StreamingLLM/InfLLM/MemoryLLM）：**CoMem 38.27 > KVD 34.59 > StreamingLLM 25.63 > InfLLM 22.21 > MemoryLLM 16.11 > HCache 8.11**；配对 bootstrap 见 §1.6。KV-Direct per-cat judge：c1 24.1/c2 18.7/c3 25.0/c4 62.2/c5 2.7；HCache：c1 6.7/c2 2.5/c3 9.4/c4 14.3/c5 1.1。全 6 方 per-cat 见 `LOCOMO_JUDGE_AGGREGATE.md`。

---

# §2 消融（chat=False，RULER 内部相对比较，全 Iron-Law-2 OK）

## §2.1 tab_selector — CoMem 单遍 selector 消融（RULER n=100，峰值 top-k over {4,8,12,16,24}）
| Task | Len | BM25 | Recency | ReaderAttn | Oracle |
|---|---|---:|---:|---:|---:|
| niah_single | 8k/16k/32k | 100/100/100 | 100/100/82 | 100/100/73 | 100/100/100 |
| niah_multikey | 8k/16k/32k | 99/99/99 | 98/88/54 | 97/90/60 | 100/100/100 |
| var-track | 8k/16k/32k | 99.4/92.6/32.0 | 99.2/92.4/41.2 | 99.8/92.4/27.8 | N/A |
> Oracle 两 needle 任务恒 100（读出无损）；BM25 单遍 niah 追平 Oracle；**VT 单遍 32k 全崩（32/41/28）→ 迭代检索动机**。

## §2.2 tab_itervt — 迭代检索（RULER variable_tracking，n=100）
| arm | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| 单遍 bm25（无跳） | 48.0 | 25.0 | 23.4 | 21.2 | 20.4 |
| **iter_bm25 3 跳（flagship `_ad`：top12/hop4/chunk512，read≈6.6k）** | **96.6** | **97.6** | **98.8** | **99.0** | **95.8** |
| iter_bm25 4 跳（`ablation10`：top16/hop4/chunk1024，read≈17k） | 99.0 | 95.6 | 89.8 | 89.8 | 87.4 |
> 迭代把 VT 从单遍 20–48 全崩救到 96–99 长档恒定。`rounds:0`=auto=ceil(topk/hop)=多跳（非单遍）。3 跳小预算 flagship 长档略优于 4 跳大预算。

## §2.3 tab_chunk — chunk-size（RULER niah_multikey，n=100）
| chunk | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|
| 128 | 91 | 90 | 81 | 85 |
| 256 | 80 | 90 | 90 | 94 |
| 512 | 89 | 95 | 97 | 94 |
| 1024 | 100 | 89 | 92 | 94 |
> chunk 512–1024 长档最稳；flagship 用 chunk1024 一致。

## §2.4 tab_crosschunk — cross-chunk attention（selector=iter_bm25 tk12）
| Task | Full | Block-diag | Δ |
|---|---|---|---|
| niah_single 8k/16k (n=50) | 100/100 | 100/96 | 0/+4 |
| niah_multikey 8k/16k (n=50) | 96/94 | 60/32 | **+36/+62** |
| BABILong qa2 8k/16k (n=100) | 36/20 | 17/13 | +19/+7 |
| BABILong qa5 8k/16k (n=100) | 78/69 | 78/55 | 0/+14 |
> cross-chunk recompute 对多事实消歧 load-bearing，对单 needle 无关。

## §2.5 tab_slm — 等预算档 CoMem vs StreamingLLM（budget=sink4+window6653=6657 tok ≈ CoMem 恒定 read）
| RULER task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| **CoMem** niah_single | 100 | 100 | 100 | 100 | 100 |
| **CoMem** var-track（等预算 3 跳） | 96.6 | 97.6 | 98.8 | 99.0 | 95.8 |
| StreamingLLM single | 90 | 42 | 18 | 16 | 4 |
| StreamingLLM multikey | 86 | 48 | 26 | 8 | 6 |
| StreamingLLM vt | 38 | 3.6 | 1.2 | 0 | 0 |
> 等预算下唯一变量=保留**哪些** token；CoMem relevance-based + 迭代恒定，recency 截断全崩（single 25× gap）。

## §2.6 效率（P1#4，chunk1024，H20，median-of-3）
| Len | Full prefill | CoMem prefill | Speedup | Full peak | CoMem peak |
|---|---:|---:|---:|---:|---:|
| 8k | 1.39s | 1.15s | 1.21× | 19.9GB | 17.8GB |
| 16k | 2.60s | 2.08s | 1.25× | 24.6GB | 19.4GB |
| 32k | 6.26s | 2.75s | 2.27× | 33.8GB | 19.5GB |
| 64k | 18.05s | 4.10s | 4.40× | 52.3GB | 19.8GB |
| 128k | **OOM** | 7.98s | ∞ | **OOM** | **20.3GB** |
> headline：128k full-ctx 在 H20 OOM（all-pos logits bf16=39.8GB 单次 alloc）而 CoMem 20.3GB 跑通。

## §2.7 迭代检索开销（P1#6，CPU micro-bench，median-of-5）
| Len | one-shot ms | iter ms | ratio |
|---|---:|---:|---:|
| 8k | 2.34 | 9.82 | 4.20× |
| 32k | 9.62 | 45.74 | 4.75× |
| 128k | 41.04 | 188.77 | 4.60× |
> iter_bm25 ≈ one-shot 4.2–4.9×，占端到端 ~0.1%，换 VT 20→100 = 基本免费。

## §2.8 P0#2 — VT selector-fairness 2×2（chat=False，RULER var-track，n=100）
| Len | KVD+iter_bm25 | KVD+bm25(1-shot) | CoMem+bm25(1-shot) | CoMem+iter_bm25(flagship) |
|---|---:|---:|---:|---:|
| 8k | 100.0 | 48.4 | 48.0 | 96.6 |
| 16k | 100.0 | 26.0 | 25.0 | 97.6 |
| 32k | 100.0 | 22.4 | 23.4 | 98.8 |
| 64k | 100.0 | 22.6 | 21.2 | 99.0 |
| 128k | 100.0 | 21.2 | 20.4 | 95.8 |
> 归因：固定 selector 下 KVD≈CoMem（大杠杆=selector 非架构）；**CoMem 架构价值=效率**（同检索 match KVD 且距 uncompressed reader 仅几 pp）。论文口径="以极低显存/算力 match KVD 精度"，非"VT 精度超 KVD"。

---

# §3 chat=True 结果（★已作废，仅历史对照；旧 `*_chatnothink` dir）

> 用户 2026-07-22 指令：全论文统一 chat=False，以下 chat=True 数字**不进正文**，保留供审计/对照。

## §3.1 InfLLM chat=True 全 5-benchmark（thunlp paper-faithful，Qwen3-8B）
**RULER**（n=100，string_match）：
| Task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| niah_single_2 | 100 | 99 | 95 | 54 | 53 |
| niah_multikey_1 | 99 | 93 | 65 | 37 | 24 |
| variable_tracking | 100 | 98.2 | 90.8 | 0 | 0 |

**LongBench**（6-ds F1）：narrativeqa 21.45 / qasper 47.10 / hotpotqa 57.43 / 2wikimqa 40.72 / multifieldqa_en 52.32 / musique 30.24 → **AVG 41.54**。
**LongEval**（n=50）：8k 0.60 / 16k 0.26 / 32k 0.12 / 64k 0.04 / 128k 0.02。
**LoCoMo**（n=1986，.73 scorer 无 judge）：F1 25.76 / EM 11.33 / acc 26.38。
**BABILong**（n=100）：qa1 100/94/92/90/85/59/37；qa2 59/56/54/58/48/43/31；qa5 80/77/75/74/78/64/55（0k→32k）。

## §3.2 MemoryLLM-8B-chat 全 5-benchmark（Llama-3 backbone，chat ON，异基座对照）
**LongEval**（n=50）：8k 0.20 / 16k 0.20 / 32k 0.18 / 64k 0.08 / 128k 0.04。
**LoCoMo**（n=1986）：F1 9.93 / EM 0.96 / acc 9.72。
**LongBench**（6-ds F1）：narrativeqa 17.71 / qasper 17.37 / hotpotqa 7.11 / 2wikimqa 7.42 / multifieldqa_en 22.75 / musique 4.45 → **AVG 12.80**（全 baseline 最弱）。
**RULER**（n=100）：single 21/31/24/8/14；multikey 30/27/26/13/8；vt 0.8/2.2/0.6/0.4/0（8k→128k，VT 全崩）。
**BABILong**（n=100）：qa1 53/42/35/23/18/10/7；qa2 37/34/16/15/15/15/16；qa5 48/50/47/40/40/36/37（0k→32k）。

## §3.3 StreamingLLM 等预算 全 benchmark（chat ON，budget=6657 tok）
**RULER**（n=50/100）：single 90/42/18/16/4；multikey 86/48/26/8/6；vt 38/3.6/1.2/0/0（8k→128k）。
**LoCoMo**（n=1986）：F1 12.73 / EM 5.24 / acc 17.57。
**LongBench**（6-ds F1，n=1150）：hotpotqa 50.25 / narrativeqa 20.51 / qasper 46.52 / multifieldqa_en 43.04 / 2wikimqa 42.36 / musique 20.52 → **AVG 37.20**。
**BABILong**（n=100）：qa1 100/94/92/89/48/30/23；qa2 60/56/54/57/34/17/3；qa5 81/77/75/76/75/67/53（0k→32k）。

## §3.4 LongBench 6-方对照（chat=True，同 6-ds 官方 qa_f1，n=1150）
| 方法 | AVG F1 |
|---|---:|
| KV-Direct（full-ctx） | 42.97 |
| InfLLM | 41.54 |
| StreamingLLM（等预算） | 37.20 |
| CoMem | 35.79 |
| HCache | 19.27 |
| MemoryLLM | 12.80 |
> ⚠️ chat=True 口径。CoMem chat=**False** LongBench macro-F1=**12.15**（§1.4）——chat=False 大幅利好 exact-match 但压低 extractive token-F1 QA。

## §3.5 LongEval 5-方对照（chat=True，n=50，max_new_tokens=48）
| 方法 | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| CoMem（per-task 最优 k∈{4,6}） | 1.00 | 0.96 | 1.00 | 0.94 | 0.98 |
| CoMem（iter_bm25 matched，n=100） | 0.73 | 0.76 | 0.79 | 0.72 | 0.76 |
| KV-Direct | 1.00 | 0.98 | 0.96 | 0.36 | 0.00 |
| InfLLM | 0.60 | 0.26 | 0.12 | 0.04 | 0.02 |
| MemoryLLM | 0.20 | 0.20 | 0.18 | 0.08 | 0.04 |
| HCache | 0.02 | 0.00 | 0.02 | 0.00 | 0.00 |

## §3.6 LoCoMo 3-方对照（chat=True，.73 scorer 无 judge，n=1986）
| 方法 | F1 | acc |
|---|---:|---:|
| InfLLM | 25.76 | 26.38 |
| CoMem（iter_bm25） | 19.51 | 28.65 |
| MemoryLLM | 9.93 | 9.72 |

## §3.7 RULER task-breadth（chat=True，4 新 task type，CoMem vs 4 baseline，n=100）
copy-hard needle × {64k, 128k}，recall%：
| task | len | CoMem | InfLLM | StreamingLLM | KV-Direct | HCache |
|---|---|---:|---:|---:|---:|---:|
| niah_single_3(uuid) | 64k | **98.0** | 25.0 | 11.0 | 99.0 | 0.0 |
| niah_single_3(uuid) | 128k | **97.0** | 5.0 | 1.0 | 0.0 | 0.0 |
| niah_multivalue | 64k | **92.5** | 36.25 | 14.75 | 96.75 | 0.0 |
| niah_multivalue | 128k | **95.25** | 23.0 | 4.5 | 0.0 | 0.0 |
| niah_multiquery | 64k | **94.75** | 35.0 | 13.75 | 97.25 | 0.0 |
| niah_multiquery | 128k | **97.0** | 18.75 | 1.5 | 0.0 | 0.0 |
> 唯 CoMem 长档恒 92–98；所有 fixed-budget/recency/no-retrieval baseline 崩。

---

# §4 我们的方法早期结果（BABILong 全 10-task，历史里程碑）

- **P8（Llama-3-8B-Instruct，L1+L3 dual-gate，500 步）**：10-task×7-len 均值 **49.1**（qa1 66.3 / qa5 68.7 / qa9 69.9 最强；qa7/qa8 长档崩）。
- v2（Llama-3.2-1B-Instruct，10000 步）均值 31.6；v2-base 30.5；plain 1B baseline 9.9。
- 参考：LM2-1.7B（from-scratch）、BABILong paper Llama-3-8B-It ~42.6。
> 详见 `status/BENCHMARK_RESULTS.md` §我们的方法。

---

# §5 全局洞察（写论文时处理，task#10）

- **chat=False 大幅利好 exact-match/completion 任务**（如 BM25 VT16k 27.6→92.6、Recency niah_single 16k 72→100），**但压低 extractive token-F1 QA**（CoMem LongBench 35.79→12.15）。因公平性统一 chat=False，全套表都换 chat=False 数字。
- **LoCoMo headline** 从旧 chat=True F1 9.05/acc24.1（05_exp:167，作废）→ **GPT-4o judge 38.27**（配对显著优于 KVD 34.59）。
- **归因主线**：VT 精度来自迭代 selector（非架构）；CoMem 架构价值=效率（128k full-ctx OOM vs CoMem 20GB，恒定 read）。

---

# §6 InfLLM chat=False 补齐进度（✅ 已完成 2026-07-24 17:09）

- **节点**：.252（8×B200，wzc1，暂停 keep12@step111500 腾出）。driver `scripts/_infllm_chatFALSE_taskpool.sh`（commit 4d5c8bc）。
- **配置**：Qwen3-8B（`models/Qwen3-8b-local`），InfLLM paper-faithful `DEFAULT_MEM_CONFIG`（block128/n_init128/n_local4096/topk16/repr4/chunk8192/base1e6），唯一差异=去 `--use_chat_template`；iter_bm25 对 InfLLM N/A（自带 block 检索）。
- **覆盖**（对齐 chat=True `infllm_8b`）：RULER main{niah_single_2,niah_multikey_1,vt}×{8k-128k}+task-breadth、LongBench 6ds、LongEval 5 档 max48、BABILong qa1/2/5×0k-32k。140 jobs 8 卡 task-pool，末尾自动跑 5 官方 scorer → `logs/infllm_chatFALSE_taskpool/SUMMARY.txt`+`SCHED_DONE`。
- **状态（✅ 2026-07-24 17:09 SCHED_DONE）**：全 148 jobs 完成，`logs/infllm_chatFALSE_taskpool/SUMMARY.txt` 全 cell Iron-Law-2 OK（0 empty / 0 mismatch / 8-8 shard）。数字已回填 §1.7a–d 各 InfLLM 行：RULER 15-cell 均值 77.83、LongEval(8k–128k) 21.6、LongBench macro-F1 11.86、BABILong qa1 69.9/qa2 43.9/qa5 64.9；LoCoMo judge 22.21（#64 回填）。.252 现跑 CoMem adapter-free（#65）。

---

# §7 数据来源（本文件汇编自）

| 内容 | 源文件 |
|---|---|
| CoMem chat=False headline + per-cell + LoCoMo judge + CI | `status/QCMEM_STATS_APPENDIX_chatFALSE.md`（.73 diskB）|
| chat=False 段 + 消融 + 效率 + 洞察 | `status/BENCHMARK_RESULTS.md` §顶部 chat=False 段 |
| InfLLM chat=True 全 5-benchmark | `status/INFLLM_BASELINE_RESULTS.md` |
| MemoryLLM 全 5-benchmark | `status/MEMORYLLM_REALQA_RESULTS.md` |
| StreamingLLM 等预算 全 benchmark | `status/STREAMINGLLM_EQUALBUDGET_RESULTS.md` |
| RULER task-breadth 多方对照 | `status/RULER_TASKBREADTH_RESULTS.md` |
| 覆盖矩阵 T/F 定论 | `status/PAPERA_CHAT_TF_MATRIX_AUDIT_20260724.md` |
| 早期方法结果（P8/v2/plain） | `status/BENCHMARK_RESULTS.md` §我们的方法 |
| **baseline chat=False per-cell（§1.7）** | 2026-07-24 `scripts/_agg_baseline_chatFALSE.py` 只读聚合 diskB `.73` 各 `*_chatFALSE/` per-shard 官方分 json |

**✅ 2026-07-24 更新**：baseline（KV-Direct/HCache/StreamingLLM/MemoryLLM）chat=False 的 RULER/LongBench/LongEval/BABILong/LoCoMo 完整 per-cell 已全部聚合回填 §1.7（官方预算分，n=100/cell）。
**⚠️ 仍缺（2026-07-24 21:xx 更新）**：(1) ✅ **InfLLM chat=False 已完成**（#63，17:09 SCHED_DONE，全 cell Iron-Law-2 OK，数字入 §1.7）；(2) ✅ **LoCoMo GPT-4o judge 全 6 方已跑齐**（#64 用有余额 JWT 补 InfLLM/StreamingLLM/MemoryLLM=22.21/25.63/16.11）；(3) **MemoryLLM chat=False LongEval/LongBench(6-ds)/BABILong** — 用户决定「两者都要」：现用 chat=True **ᵀ 占位**（LongEval 14.0/LongBench 12.80/BABILong 26.9·21.1·42.6，见 master），待 diskB 空出跑真 chat=False 覆盖=**pending #68**（MemoryLLM env/权重仅在 diskB；当前 wzc1 满载、diskB H20 归用户）；(4) ✅ **CoMem adapter-free（frozen j9）chat=False 全 5-benchmark eval 完成回填**（#65，SUMMARY 2026-07-24 23:35，148 jobs 全完）：RULER 59.4 / LongEval 3.2 / LongBench 10.63 / BABILong 42.4·19.6·55.6 / LoCoMo local acc 16.41；**仅剩 LoCoMo GPT-4o judge 待跑**（见 `BENCHMARK_CHATFALSE_MASTER.md` §F）。
</content>
</invoke>
