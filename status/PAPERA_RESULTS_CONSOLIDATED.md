# Paper A（CoMem / QCMem）— 结果总集（单文档权威汇编）

> **生成 2026-07-25 11:XX（main 汇编，非重跑）。** 这是 Paper A 提交口径下**全部当前结果的单一文档**：一处读全 headline 主矩阵、逐 benchmark 明细、决定性/消融实验、效率、LoCoMo judge、统计显著性、可复现配置、唯一待补项。
> **用途**：等论文正文重构（sections 02–07 + Overleaf 往返）定稿后，直接以本文档为准整合进各表 + 05_experiments 正文（= pending #10）。
> **口径双支柱（论文标准）**：`chat_template=False`（全模型是 continue-train 的 BASE LM，无 SFT/RL，套 chat template 对 base 不公平）+ CoMem `selector=iter_bm25`（topk=12, hop=4, sink=bos, chunk_size=512）。
> **官方判分**：RULER=`string_match_all`；BABILong=`compare_answers`+`TASK_LABELS`；LongBench/LongEval=`run_scoring`；LoCoMo headline=**GPT-4o judge**（n=1986）。
> **backbone**：Qwen3-8B（除 MemoryLLM=Llama-3-8B-chat 异基座对照）。
> **明细/provenance 源**：`BENCHMARK_CHATFALSE_MASTER.md`（headline）、`PAPERA_ALL_RESULTS.md`（per-cell + chat=True 历史）、`LOCOMO_JUDGE_AGGREGATE.md`（judge 全细节）、`QCMEM_STATS_APPENDIX_chatFALSE.md`（per-len/per-ds + bootstrap CI）、`PAPERA_REPRO_HYPERPARAMS.md`（超参）。

---

## §0 完成度总览（2026-07-25）

| 模块 | 状态 |
|---|---|
| Headline 6 方法 × 5 benchmark 主矩阵（chat=False） | ✅ 完整（唯一例外见下 #68） |
| LoCoMo GPT-4o judge（全 6 方 headline） + 配对 bootstrap 显著性 | ✅ |
| 决定性实验：§F adapter-free（#65）、§H j0 控制（#71）、§I frozen depth-sweep（#73） | ✅ 全闭合 |
| 效率（#67 LoRA-on 控制 + H20 headline） | ✅（口径：report H20，见 §4） |
| 统计附录（per-len/per-ds + bootstrap 95%CI，#54） | ✅ |
| 可复现超参表（#69）、LoRA 训练成本（#66）、BABILong hop=4 统一（#70）、bib（#72） | ✅ |
| **✅ #68 MemoryLLM chat=False——全 5 benchmark 补齐（LongEval 13.6 于 2026-07-25 .104 补跑）** | ✅ 全闭合；异基座次要参考行 |
| paper 整合（#10：InfLLM 入表 + LoCoMo iter_bm25 errata + tab_eff 标 H20 + 最终换表 + 正文） | ⏸ 待正文重构定稿 |

**一句话**：Paper A **实验数据全部补齐**（#68 MemoryLLM LongEval chat=False=13.6 于 2026-07-25 在 .104 补完，全矩阵已无 chat=True 占位）；仅剩论文写作整合（#10，非数据）。

---

## §0.5 ★ 四项口径裁决（2026-07-25，从一手 config/JSON/驱动脚本核账，非抄旧文本）

> 用户 review 提出 4 处口径冲突，必须以**实际 config 文件**为准裁决（不能信任已含矛盾的旧文本）。以下每项附一手出处，已回填本文档对应节。

### 裁决 1 — 检索配置全 benchmark 统一 `topk=12, hop=4`（**否定** repro sheet「LoCoMo k=8 / LongEval k=4–6」）
- **ground truth（config-recorded）**：
  - `locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/eval_config_shard0of8.json` → `topk=12, iter_hop_topk=4, iter_rounds=0, chunk_size=512, sink_tokens=bos, selector=iter_bm25`。
  - `longeval_results/qcmem_8b_zeroshot_j9_chatFALSE/eval_config_shard0of8.json` → **同样 `topk=12, iter_hop_topk=4`**（**不是** k=4–6）。
  - 驱动脚本 `scripts/_qcmem_adapterfree_j9_chatFALSE_taskpool.sh:54-60` 定义单一 `COMMON="--resume_j 9 --selector iter_bm25 --topk 12 --iter_hop_topk 4 --sink_tokens bos --chunk_size 512"`，并在 `:138-172` 的 **5 个 benchmark（ruler/longeval/locomo/longbench/babilong）invocation 里逐一 append `$COMMON`**，无任何 per-benchmark topk/hop 覆写。
  - 效率 bench `ruler_results/bench_qcmem_vs_fullctx.json` config → `topk=12`。
- **裁决**：**旗舰及全部决定性实验族（adapter-free j9 / j0 / depth-sweep j6·j12）在全 5 benchmark 上统一 `topk=12, iter_hop_topk=4, iter_rounds=auto=⌈12/4⌉=3 轮, chunk_size=512, sink=bos, chat=False`**。repro sheet 旧行「LoCoMo k=8 / LongEval k=4–6」是**过期探索遗留**，被实录 eval_config 直接否证 → 已作废。BABILong hop 早前 2→4 修正并重跑（#70）。
- **每 benchmark 真正差异的只是 eval-harness 参数**（非检索配置）：n/长度/max_new_tokens/scorer（见 §5.4 表）。旗舰在 .73 上（配置同族，仅 j/LoRA 不同），选择器配置按设计与本地族一致。

### 裁决 2 — LoRA「4k」= **4000 steps**，训练序列长 = **2048 tokens**（无冲突）
- **ground truth**：`outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json` → `total_steps=4000`、`n_ctx=3`、`chunk_size=512`。
- 序列长 = `(n_ctx+1)×chunk = (3+1)×512 = 2048 tokens`；总 token = `4000 steps × 8 GPU × 2048 = 65.5M`。
- **裁决**：adapter 名里的「4k」= **4000 training steps**，**不是** 4096 序列长。论文写「**4000 steps / 2048-token windows / 65.5M tokens**」。§5.2 已改。

### 裁决 3 — Qwen3-8B（dense）128k full-ctx **不 OOM**（89.36 GB status=ok，删除 §2.4「128k full-ctx OOM」措辞）
- **ground truth**：`ruler_results/bench_qcmem_vs_fullctx.json` 128k 行 → `full_ctx.status="ok"`, `peak_gb=89.36`, `prefill_s=15.014`；`qcmem.peak_gb=18.26`, `prefill_s=1.917`, `speedup=7.83`。**89.36 < 96 → 未 OOM**。
- **裁决**：8B **dense** 模型 128k full-ctx **在 96 GB 卡上成功跑完、峰值 89.36 GB**，**不能说 OOM**。正确措辞：「At 128k, full-context prefill peaks at **89.36 GB** on a 96 GB GPU while CoMem uses **18.26 GB** and is **7.83× faster**.」§2.4 已删「128k full-ctx OOM」并改成此措辞。**唯一真 OOM 是 Qwen3-30B-A3B（MoE）128k dense**——那行保留。

### 裁决 4 — 效率 headline（7.83× / 18.26 GB）与 LoRA-on 控制（2.74× / 18.54 GB）分开报告，硬件不同
- **ground truth**：headline JSON `bench_qcmem_vs_fullctx.json` config `lora_adapter=None`（**LoRA-OFF**）；`logs/bench_chunk512_full.log:1-7` **只记 `device=cuda:0`，无 GPU 名**——「H20」是**推断非实录**。其 dense prefill 15.014s@128k ≈ #67 L20A 控制（6.035s）的 **2.5×**，与「更慢的 H20 级 bf16」一致。
- LoRA-on 控制 `ruler_results/bench_*loraON_L20A.json` 在 **L20A（183 GB）** 跑：128k 2.74× / 18.54 GB。
- **裁决**：
  1. **不得把 2.74× vs 7.83× 当作纯 LoRA 减速**——二者硬件不同（headline=未记名 H20 级；控制=L20A）。
  2. **纯 LoRA 开销由同硬件 L20A 配对隔离**：LoRA-OFF 3.23× / 18.29 GB vs LoRA-ON 2.74× / 18.54 GB → LoRA 仅 **+0.25 GB 峰值、+~18% prefill**，保留渐进效率趋势。
  3. **显存比值硬件无关**（~5× 优势，安全）；**加速比值硬件相关**（论文按 report headline 处理，须标明硬件档 = 96 GB/H20 级）。§4 已按此拆分。

---

## §1 Headline 主矩阵（一屏总览，chat=False）

| 方法 | RULER<br>(15-cell 均值 recall) | LongEval<br>(8k–128k 均值 acc) | LongBench<br>(6-ds macro-F1) | BABILong<br>qa1 / qa2 / qa5 | LoCoMo<br>(GPT-4o judge, n=1986) |
|---|:---:|:---:|:---:|:---:|:---:|
| **CoMem（本文，+distilled LoRA, j12）** | **97.05** | 69.0¹ | 12.15 | 55.6 / 27.0 / **68.7** | **38.27** |
| **CoMem（本文，adapter-free, frozen j9）** | 59.4 | 3.2 | 10.63 | 42.4 / 19.6 / 55.6 | 29.15 |
| KV-Direct（full-ctx 上界，j=0） | 78.80² | 65.2 | **12.17** | **78.7 / 48.9** / 61.4 | 34.59 |
| InfLLM（thunlp 旗舰 baseline） | 77.83 | 21.6 | 11.86 | 69.9 / 43.9 / 64.9 | 22.21 |
| StreamingLLM（等预算 recency） | 23.37 | 30.8 | 11.11 | 56.1 / 33.1 / 55.1 | 25.63 |
| MemoryLLM（Llama-3-8B-**chat**，异基座）³ | 16.55 | 13.6 | 9.01 | 30.4 / 21.4 / 38.1 | 16.11 |
| HCache（retrieval-free mid-layer） | 3.73 | 0.0 | 9.20 | 32.9 / 18.0 / 50.3 | 8.11 |

**排名（headline，chat=False）：**
- **RULER**：CoMem 97.05 > KVD 78.80 ≳ InfLLM 77.83 ≫ StreamingLLM 23.37 > MemoryLLM 16.55 > HCache 3.73
- **LongEval**：CoMem 69.0 > KVD 65.2 > StreamingLLM 30.8 > InfLLM 21.6 > MemoryLLM 13.6 > HCache 0.0（MemoryLLM 现为真 chat=False，异基座参考）
- **LongBench**：KVD 12.17 ≈ CoMem 12.15 > InfLLM 11.86 > StreamingLLM 11.11 > HCache 9.20 > MemoryLLM 9.01（chat=False 把全方法压到 9–12 窄带 = 协议效应非压缩损失；MemoryLLM 现真 chat=False=最低，异基座 Llama-3 参考）
- **BABILong**：KVD 全档最强（full-ctx 上界）；CoMem qa5 68.7 > KVD 61.4（唯一超上界项）
- **LoCoMo（judge）**：**CoMem 38.27 > KVD 34.59 > CoMem(adapter-free) 29.15 > StreamingLLM 25.63 > InfLLM 22.21 > MemoryLLM 16.11 > HCache 8.11** —— distilled LoRA 把 CoMem 从 full-ctx 上界之**下**(29.15)抬到之**上**(38.27)。

¹ CoMem LongEval 6-档（4k–128k）headline=72.83；此列取 8k–128k 5 档与 baseline 对齐=69.0。
² KVD RULER 15-cell 含 128k=0（131072>Qwen3 max_pos 窗口溢出）；≤64k near-perfect。CoMem 恒定 read 128k 仍 93–100。
³ MemoryLLM 是真·chat 模型（非 continue-train base LM）；chat=False 剥离其原生 chat 模板 = 对它 OOD/不公平，chat=True 才是其公平协议。**全 5 benchmark 均有真 chat=False**：RULER 16.55 / LoCoMo 16.11 / LongBench 9.01 / BABILong 30.4·21.4·38.1（2026-07-25 .82）+ **LongEval 13.6（2026-07-25 .104 补跑，8k22/16k22/32k16/64k6/128k2，config 确认 use_chat_template:false）**。全矩阵无 chat=True 占位；MemoryLLM 行作异基座 cross-base 参考。

---

## §2 逐 benchmark 明细

### §2.1 RULER（recall %，n=100/cell，niah_single_2 + niah_multikey_1 + variable_tracking）

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

> CoMem 长档恒 93–100（固定预算不溢出）；KVD 128k 归零（窗口溢出）；InfLLM 长档缓降（niah 64k–128k 掉到 45–61）；StreamingLLM/HCache recency/无检索长档全崩。

### §2.2 LongEval（line-retrieval acc %，n=50/len）

| 方法 | 8k | 16k | 32k | 64k | 128k | mean(8k–128k) |
|---|---:|---:|---:|---:|---:|---:|
| **CoMem（本文）** | 69 | 75 | 64 | 67 | 70 | **69.0** |
| **KV-Direct（上界）** | 100 | 96 | 92 | 38 | 0 | 65.2 |
| StreamingLLM | 86 | 34 | 18 | 10 | 6 | 30.8 |
| InfLLM | 60 | 30 | 12 | 4 | 2 | 21.6 |
| HCache | 0 | 0 | 0 | 0 | 0 | 0.0 |
| MemoryLLM（chat=False，异基座）| 22 | 22 | 16 | 6 | 2 | 13.6 |

> CoMem 6-档（含 4k=92）headline=72.83。KVD 64k 骤降 38、128k 归零（RoPE 溢出）；CoMem 恒 64–75。

### §2.3 LongBench（per-ds F1，macro-mean over 6 ds，官方 qa_f1）

| 方法 | narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **macro** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CoMem（本文）** | 4.12 | 11.01 | 11.62 | 12.83 | 25.41 | 7.91 | **12.15** |
| **KV-Direct（上界）** | 3.70 | 11.82 | 12.68 | 12.03 | 25.30 | 7.49 | **12.17** |
| InfLLM | 2.99 | 11.35 | 12.08 | 12.50 | 25.33 | 6.94 | **11.86** |
| StreamingLLM | 3.49 | 11.64 | 11.02 | 12.19 | 22.42 | 5.88 | **11.11** |
| HCache | 2.56 | 10.71 | 7.33 | 9.19 | 20.39 | 5.05 | **9.20** |
| MemoryLLM（chat=False）| 3.13 | 8.46 | 8.76 | 10.27 | 17.43 | 5.98 | **9.01** |

> ⚠️ chat=False 把所有方法的 LongBench extractive token-F1 全线压到 9–12（chat=True 时 KVD 42.97/InfLLM 41.54/CoMem 35.79）——CoMem 12.15 ≈ full-ctx 上界 KVD 12.17（打平），低分是**协议效应非压缩损失**。MemoryLLM 行现为真 chat=False（2026-07-25 .82 全 6-ds，macro 9.01=最低；异基座 Llama-3-chat OOD，仅 cross-base 参考）。

### §2.4 BABILong（compare_answers %，n=100/cell，mean over 0k–32k；CoMem = iter hop=4 统一口径 #70）

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

> ⚠️ BABILong ≤32k 全在 full-ctx 窗口内 → **full-ctx 上界 KVD 在 qa1/qa2 明显强于 CoMem**（诚实的压缩 tax，非 CoMem 优势项）；CoMem qa5 68.7 略超 KVD 61.4。CoMem 卖点在**超出 full-ctx 的长度 + 效率**，非 ≤32k 精度超上界。
> **★ 128k 措辞裁决（§0.5 裁决 3）**：Qwen3-8B（**dense**）128k full-ctx **不 OOM**——在 96 GB 卡上成功跑完，峰值 **89.36 GB**（vs CoMem 18.26 GB，7.83× faster）。**不要写「128k full-ctx OOM vs CoMem」**（8B dense 处错误）。**唯一真 OOM = Qwen3-30B-A3B（MoE）128k dense**（该行可保留）。

### §2.5 LoCoMo（GPT-4o judge headline，chat=False，n=1986）

| 方法 | **judge (n=1986)** | judge (cat1–4, n=1540) | F1 | acc | EM |
|---|---:|---:|---:|---:|---:|
| **CoMem（本文，iter_bm25）** | **38.27** | **48.64** | 9.15 | 23.36 | 0.55 |
| **CoMem（本文，adapter-free j9）** | 29.15 | 37.27 | 7.28 | 16.41 | 0.25 |
| KV-Direct（full-ctx 上界） | 34.59 | 43.83 | 9.02 | 22.36 | 0.60 |
| StreamingLLM | 25.63 | — | 7.67 | 13.75 | 1.56 |
| InfLLM | 22.21 | — | 7.39 | 13.34 | 1.71 |
| MemoryLLM | 16.11 | — | 5.91 | 9.52 | 0.10 |
| HCache | 8.11 | 10.13 | 4.67 | 6.29 | 0.25 |

> **per-cat judge（CoMem 旗舰）**：cat1 26.95 / cat2 19.00 / cat3 30.21 / cat4 **69.32** / cat5 2.47（本地 abstain）。cat4（single-hop，最大桶 n=841）领先决定性（+7 over KVD）。
> **★ 配对 bootstrap 显著性（judged n=1540，#60）**：CoMem 旗舰 vs KVD full-ctx oracle paired diff=**+4.81**，95%CI **[2.34, 7.27]**，**p<0.0001** → **CoMem 显著优于 full-ctx KV oracle**。

---

## §3 决定性 / 消融实验（回应 reviewer「优势是否只是检索」）

### §3.1 CoMem adapter-free（#65，frozen j9，全 5-benchmark）

- **配置**：Qwen3-8B **frozen backbone（无 LoRA，靠省略 `--lora_adapter` 实现）**，`resume_j=9`，selector=iter_bm25/topk12/hop4/sink=bos/chunk512，chat=False。一份固定 config 跑全 5 benchmark。
- **结果**：RULER 59.4 / LongEval 3.2 / LongBench 10.63 / BABILong 42.4·19.6·55.6 / **LoCoMo judge 29.15**（cat4 53.75；cat1-4 加权 37.27）。
- **意义**：frozen backbone 在 niah_multikey（多事实消歧）、variable_tracking（迭代 tracking）、LongEval（行检索）上**显著退化**（RULER 掉 ~38pt、LongEval 近乎归零），但单针 niah_single_2 仍近满分（96–99）。→ **distilled LoRA 主要买回「多事实消歧 + 迭代 tracking + 行检索」**；单 needle 靠 frozen backbone + iter_bm25 即近满分。

### §3.2 j=0 退化控制（#71，全 36 层重算，无 LoRA、仍检索 top-12）

| config | split j | LoRA | 输入 | **LoCoMo judge (n=1986)** | cat1-4 (n=1540) | BABILong qa1/qa2/qa5 |
|---|:---:|:---:|---|:---:|:---:|:---:|
| CoMem 旗舰（+distilled LoRA） | 12 | ✅ | 检索 top-12 | 38.27 | 48.64 | 55.6 / 27.0 / **68.7** |
| **#71 j0 control（全层重算）** | 0 | ✗ | 检索 top-12 | **41.59** | **52.60** | 65.9 / 39.1 / 63.0 |
| #65 adapter-free（frozen） | 9 | ✗ | 检索 top-12 | 29.15 | 37.27 | 42.4 / 19.6 / 55.6 |
| KV-Direct（full-ctx oracle） | 0 | ✗ | **全上下文** | 34.59 | 43.83 | **78.7 / 48.9** / 61.4 |

- **(a) 深度切分贡献**（j0 vs adapter-free j9，唯一变量=split depth）：LoCoMo judge 41.59 vs 29.15 → frozen 深度-9 切分损失 **−12.44 pt**；BABILong 全 task ≫ frozen j9。→ 深度轴压缩**丢保真度**，旗舰的 distilled LoRA 买回大部分。
- **(b) 检索贡献**（j0 vs KVD，唯一变量=检索）：LoCoMo judge 41.59 vs 34.59 → **检索单独 +7.00**（过滤无关对话轮）；但 BABILong needle qa1/qa2 **65.9/39.1 vs 78.7/48.9** → 检索在 needle 任务上**倒亏**（top-k 漏散落事实）。
- **★ reviewer 回答（诚实、分基准）**：LoCoMo 上 CoMem 超 oracle 很大一部分来自检索（j0 已 41.59>KVD）；BABILong ≤32k 检索是 tax 而非 win；深度轴切分处处需 LoRA 补偿。CoMem 普适卖点=**固定预算下超越 full-ctx 窗口的长度可扩展性 + 效率**，非窗口内一律精度超上界。

### §3.3 frozen depth-sweep（#73）— 纯 isolate distilled LoRA 贡献

| split j | LoRA | **LoCoMo judge (n=1986)** | BABILong qa1/qa2/qa5 (mean 0k–32k) |
|:---:|:---:|:---:|:---:|
| **j0**（全 36 层重算，=#71） | ✗ | **41.59** | 65.9 / 39.1 / 63.0 |
| **j6**（=#73） | ✗ | 32.78 | 44.6 / 22.6 / 53.9 |
| **j9**（=#65 adapter-free） | ✗ | 29.15 | 42.4 / 19.6 / 55.6 |
| **j12**（frozen，=#73） | ✗ | **24.52** | 33.4 / 18.0 / 60.3 |
| **j12 + distilled LoRA（旗舰）** | ✅ | **38.27** | 55.6 / 27.0 / 68.7 |

- **(1) frozen 深度-保真度单调递减**：LoCoMo judge 41.59 → 32.78 → 29.15 → 24.52（j 越深、frozen backbone 丢的保真度越多）。
- **(2) ★ 纯 distilled LoRA 贡献（SAME depth j12，唯一变量=LoRA）**：LoCoMo judge **j12 frozen 24.52 → j12+LoRA 38.27 = +13.75**（比混合口径 +9.12 更干净大）；BABILong qa1 +22.2/qa2 +9.0/qa5 +8.4。注意 9→12 的 frozen 深度变化本身 −4.63 judge（有害），故 **+13.75 才是蒸馏 LoRA 在旗舰深度上的真实纯贡献**。
- **对论文的意义**：distilled LoRA 是 CoMem 超过 full-ctx KV oracle 的**决定性组件**——frozen backbone（任何深度）都 < KVD oracle；只有加蒸馏 LoRA 才越过。

---

## §4 效率（#67，2026-07-25 LoRA-on 控制完成）

> **口径决定（§0.5 裁决 4）**：**headline（7.83× / 18.26 GB）与 LoRA-on 控制（2.74× / 18.54 GB）分开报告，硬件不同**——headline JSON 是 LoRA-OFF、GPU 未记名（H20 级推断）；#67 LoRA-on 控制在 L20A（183 GB）跑。**纯 LoRA 开销由同硬件 L20A 配对隔离**（LoRA-OFF 3.23× / 18.29 GB vs LoRA-ON 2.74× / 18.54 GB → LoRA 仅 +0.25 GB 峰值、+~18% prefill）。**不得把 2.74× vs 7.83× 当纯 LoRA 减速。** 结论：加载 distilled LoRA 不破坏效率故事；显存 ~5× 优势硬件无关。

### §4.1 Headline（`bench_qcmem_vs_fullctx.json`，**LoRA-OFF**，full-write-inclusive，median-of-3，chunk512）

> **硬件说明（§0.5 裁决 3/4）**：headline JSON config `lora_adapter=None`（LoRA-OFF）；驱动 log 只记 `device=cuda:0`，**未记 GPU 名**——「H20」是**推断非实录**（其 dense prefill 15.014s@128k ≈ L20A 控制 6.035s 的 2.5×，与更慢的 H20 级 bf16 一致）。论文报此列须标硬件档为「96 GB / H20 级」。

| Length | Prefill speedup (full/CoMem) | Full mem | CoMem mem | full 128k status |
|---|:---:|---:|---:|:---:|
| 8k | 0.97× | 19.9 GB | 17.3 GB | — |
| 16k | 1.59× | 24.5 GB | 17.4 GB | — |
| 32k | 2.48× | 33.8 GB | 17.5 GB | — |
| 64k | 4.36× | 52.3 GB | 17.8 GB | — |
| **128k** | **7.83×** | **89.36 GB** | **18.26 GB** | **ok（不 OOM）** |

> CoMem 显存 context-independent（~17–18 GB）；full attention 增长到 89.36 GB **但在 96 GB 卡上仍成功跑完（status=ok，未 OOM）**。crossover ≈16k。**7.83× 是硬件相关数（论文须标 96 GB/H20 级硬件）；显存比值 ~5× 硬件无关。**

### §4.2 LoRA-on 控制（#67，L20A 8×，复现论文方法学的 full-write bench）

| len | DENSE(vanilla) pref_s / peak_GB | CoMem **LoRA-OFF** pref_s / peak_GB / spd | CoMem **LoRA-ON** pref_s / peak_GB / spd |
|----:|---:|---:|---:|
| 8k  | 0.165 / 19.92 | 0.230 / 17.35 / 0.72× | 0.309 / 17.60 / 0.54× |
| 16k | 0.356 / 24.55 | 0.313 / 17.41 / 1.14× | 0.441 / 17.67 / 0.81× |
| 32k | 0.826 / 33.82 | 0.537 / 17.54 / 1.54× | 0.681 / 17.79 / 1.21× |
| 64k | 2.104 / 52.34 | 0.962 / 17.79 / 2.19× | 1.200 / 18.04 / 1.75× |
| 128k| 6.035 / 89.39 | 1.867 / 18.29 / **3.23×** | 2.202 / 18.54 / **2.74×** |

**Decode 延迟（KV-cache，ms/tok median / p95）：**

| len | Dense | CoMem LoRA-off | CoMem LoRA-on |
|----:|---:|---:|---:|
| 32k | 21.70 / 21.92 | 23.79 / 23.94 | 32.05 / 33.02 |
| 64k | 21.65 / 21.83 | 23.91 / 24.31 | 31.48 / 31.64 |
| 128k| 21.85 / 21.87 | 23.78 / 23.85 | 31.70 / 31.72 |

### §4.3 结论（对 reviewer「efficiency only measured adapter-free」）

1. **★ 显存声明精确复现且 LoRA-on 成立**：LoRA-off@128k CoMem **18.29 GB** / Dense **89.39 GB**（与论文 18.26 / 89.36 到 2 位小数吻合）；LoRA-on 仅 **+0.25 GB**（232 MB adapter 参数）→ 18.54 GB。**~5× 显存优势是硬件无关的基本卖点，安全，且 LoRA-on 不破坏。**
2. **加速 LoRA-on 保留、LoRA 开销小**：LoRA @128k prefill +18%（绝对 0.14–0.41s），CoMem 仍远快于 dense。
3. **7.83× 是硬件相关（H20 级，未记名推断），非 LoRA artifact**：CoMem prefill + peak-mem 精确复现 headline，**只有 dense prefill 因慢卡 bf16 被 throttle 而更慢**（headline json dense 15.01s vs L20A 6.04s），抬高比值。同代码在 L20A 上 = 3.23× LoRA-off / 2.74× LoRA-on。**显存比值硬件无关；加速比值硬件相关**——论文报 headline 时标明硬件档（96 GB/H20 级）即诚实。
4. **decode**：LoRA +~33% ms/tok（context-independent，仍 O(1)/step）。
5. **OOM 措辞（§0.5 裁决 3）**：**8B dense 128k peak=89.36 GB < 96 GB → 不 OOM**（在 96 GB/H20 级卡也 fit，chunk512 实测）。**不要把 8B dense 128k 当 OOM 卖点**。真正 OOM 只在 (a) **Qwen3-30B-A3B（MoE）128k dense**，或 (b) 更小卡 / 更长上下文 / 更大模型。论文效率主张 = **显存 ~5× ratio + 超窗口长度可扩展**，不依赖 8B dense OOM。

- 结果 JSON：`ruler_results/bench_{fullwrite_,}lora{OFF,ON}_L20A.json` + `bench_lora_sanity_32k.json`。

---

## §5 可复现配置

### §5.1 旗舰 config（全 benchmark 固定 — §0.5 裁决 1）
- backbone Qwen3-8B（`models/Qwen3-8b-local`），split depth **j=12**，distilled LoRA（rank32/alpha64，layers 12–35）= `outputs/qcmem_distill_qwen_j12_r32_4k/final`。
- **检索/读配置全 5 benchmark 统一**：selector=**iter_bm25**，**topk=12**，**iter_hop_topk=4**（`iter_rounds=0`→auto=⌈12/4⌉=**3 轮**），sink=**bos**，chunk_size=**512**，read_len≈6.6k（128k 恒定 seq_len=6657），bf16，sdpa，**chat=False**。**无 per-benchmark topk/hop 调整**（config-recorded 证据：`{locomo,longeval}_results/.../eval_config_shard0of8.json` 均 topk=12/hop=4；驱动 `_qcmem_adapterfree_j9_chatFALSE_taskpool.sh:54-60` 单一 `$COMMON` 应用于 5 benchmark）。**旧 repro sheet「LoCoMo k=8 / LongEval k=4–6」已作废。**
- BM25：token-ID 级，k1=1.5，b=0.75，Robertson IDF+1（`run_babilong_mem_space.py:754`）。
- adapter-free 变体：省略 `--lora_adapter`（无 `--zero_training_no_adapter` flag），resume_j=9（frozen backbone）；depth-sweep 变体 resume_j∈{0,6,9,12} frozen。

### §5.2 LoRA 训练成本（#66，§0.5 裁决 2）
- distilled LoRA：rank32 / alpha64 / dropout0，训练层 **12–35（24 层，58.20M=backbone 的 0.71%）**，backbone 全 frozen。
- **训练步数 = 4000 steps**（adapter 名「4k」= 4000 **steps**，非序列长）；**训练序列长 = 2048 tokens**（`(n_ctx=3 + 1) × chunk=512`）；**总 token = 4000 × 8 GPU × 2048 = 65.5M**（`distill_args.json` total_steps=4000/n_ctx=3/chunk=512）。
- self-distillation：teacher = 同一 Qwen3-8B（`disable_adapter()`，teacher split j=0 full-ctx），teacher top-64 logits，loss=`0.6·KL(T‖S)+0.4·KL(S‖T)`（仅 query segment，无 hard CE），T=1。数据=PG19 train 流式。
- 优化：AdamW(0.9,0.95)，peak lr 8e-5，warmup 100→cosine→0@4000，wd 0，grad clip 1.0，grad_accum 1，seed 42，bf16，sdpa。8×L20A DDP，~24.5 samp/s，wall-clock ~22 min。详见 `PAPERA_REPRO_HYPERPARAMS.md`。

### §5.3 节点 / env
- CoMem 旗舰 + KVD/HCache/StreamingLLM/MemoryLLM chat=False + LoCoMo judge：**.73（28.85.35.73，diskB）**。
- InfLLM #63 + adapter-free #65：**LOCAL / .252（wzc1 共享盘）**。
- 效率 #67：**LOCAL（8×L20A）**（H20 headline 源自历史 `ruler_results/bench_qcmem_vs_fullctx.json`）。
- 磁盘拓扑：LOCAL 与 .252 共享 wzc1 物理盘；.73 在 diskB（另一物理盘）。

### §5.4 每 benchmark eval-harness 参数（检索配置统一，仅这些 harness 参数不同）

| benchmark | tasks | lengths | n/cell | max_new | scorer | seed |
|---|---|---|---:|---:|---|---|
| RULER | niah_single_2, niah_multikey_1, variable_tracking | 8k–128k(5) | 100 | 48（vt=60） | `string_match_all`（大小写无关 substring recall） | 42 |
| LongEval | lines-retrieval | 4k–128k(6，主表取 8k–128k) | 50 | 16（旗舰）/48（adapter-free 族） | `extract_prediction`（首个 ≥4 位数字串）== expected | 1234 |
| LongBench | narrativeqa/qasper/hotpotqa/2wikimqa/multifieldqa_en/musique | — | 200（mfqa=150） | 32/64/128 按 ds | SQuAD token-F1 多参考取 max（macro over 6 ds） | 42 |
| BABILong | qa1/qa2/qa5 | 0k–32k(7) | 100 | 20 | `babilong.metrics.compare_answers`+`TASK_LABELS` | 42 |
| LoCoMo | 全 cat（10 conv, 1986 QA） | — | 1986 | 48 | headline=**GPT-4o judge**(n=1986)；local F1/EM/acc | judge 1234 |

> ⚠️ 检索配置（topk=12/hop=4/rounds3/chunk512/sink=bos）**全 benchmark 统一**；上表只是评测脚手架差异。`max_new_tokens`：旗舰 LongEval=16、adapter-free 族=48（对 line-retrieval 单数字答案两者皆足够，非结论敏感）。

### §5.5 read-pack 构造 + attention mask（P0 建议 #4 — reviewer 常问）

**write（缓存阶段）**：每个 chunk（≤512 token）独立前向到 split depth **j=12** 层，缓存该层 hidden `h_j`（bf16，无 offload/量化）。chunk-local RoPE：每 chunk 位置从 0 起（`positions=arange(T)`），chunk 间相互隔离。

**read（回答阶段）**：iter_bm25 选出 top-12 chunk 后，拼装单一 read-pack：
```
pack = [ sink(BOS 的 h_j, 1 token) ]           # sink=bos
     + [ selected chunk_i 的 h_j, i=1..12 ]     # 按文档原序排列
     + [ query 的 h_j ]                          # 问题
# read 位置编码：全新连续 RoPE 0..H-1（非原始 chunk 位置），H=seq_len≈6657
# 从 layer[12] 起用 layers[12:36]（24 层，+distilled LoRA）重算到 logits
```
**mask（默认，block_diagonal=False）**：整个 pack 是**单一 causal 序列** →
- **query 能看到全部 12 个检索 chunk**（它们在 query 之前）✅；
- **chunk 之间可互相 attend（cross-chunk full attention）**——非块对角隔离；
- 全 pack 一个 causal mask（后 token 看前 token），fresh 连续 position。
- **消融**：`block_diagonal=True`（tab_crosschunk）= chunk 只能看自身+sink，query 看全部 → 论文里作 cross-chunk 消融行，**旗舰用 full（block_diagonal=False）**。
- 出处：`qcmem_model.py:89,117-132`（mask 构造）、`:515-525`（pack 顺序）、`:529`（read RoPE）、`:384`（write chunk-local RoPE）。

---

## §6 provenance（源文件）

| 内容 | 源 |
|---|---|
| CoMem 旗舰 chat=False per-cell + bootstrap CI | `QCMEM_STATS_APPENDIX_chatFALSE.md`（.73 diskB）|
| baseline chat=False per-cell（KVD/HCache/StreamingLLM/MemoryLLM）| `PAPERA_ALL_RESULTS.md` §1.7 |
| InfLLM chat=False（#63）| `logs/infllm_chatFALSE_taskpool/SUMMARY.txt`（LOCAL/.252 wzc1）|
| LoCoMo GPT-4o judge 全 6 方 | `LOCOMO_JUDGE_AGGREGATE.md`（.73 各 `locomo_results/*/scores.json`）|
| CoMem adapter-free（#65，§3.1）| `logs/qcmem_adapterfree_j9_chatFALSE/SUMMARY.txt` + `locomo_results/qcmem_8b_zeroshot_j9_chatFALSE/scores.json` |
| j0 控制（#71，§3.2）| `{locomo,babilong}_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE` |
| frozen depth-sweep（#73，§3.3）| `{locomo,babilong}_results/qcmem_8b_zeroshot_j{6,12}_frozen_iterbm25_chatFALSE` |
| 效率（#67，§4）| `ruler_results/bench_{fullwrite_,}lora{OFF,ON}_L20A.json`；H20 headline `ruler_results/bench_qcmem_vs_fullctx.json` |
| MemoryLLM chat=False 全 5 benchmark（LongBench 9.01 / BABILong 30.4·21.4·38.1 于 .82；LongEval 13.6 于 .104）| `longbench_results/memoryllm_8b_chatFALSE/scores.json` + `babilong_results/memoryllm_8b_chatFALSE/_summary_merged.json` + `longeval_results/memoryllm_8b_chatFALSE/_summary_merged.json`（2026-07-25 .104 wzc1 共享盘，本地可见）|
| headline 汇编 | `BENCHMARK_CHATFALSE_MASTER.md` §A–§I |

---

## §7 唯一待补项 + 整合计划

### ✅ #68 — MemoryLLM chat=False overlay（全 5 benchmark 补齐，CLOSED）
- **LongBench** 全 6-ds 全新 8-GPU chat=False → macro **9.01**（.82）；**BABILong** config 确认 chat=False 的 21-cell 目录经官方 compare_answers 重判 → qa1 30.4/qa2 21.4/qa5 38.1（.82）。
- **LongEval chat=False = 13.6**（2026-07-25 .104 补跑）：8k22/16k22/32k16/64k6/128k2，n=50/档，8-GPU 分片，`--score_only` 合并，全 8 shard `eval_config` 确认 `use_chat_template:false`，无 OOM。→ `longeval_results/memoryllm_8b_chatFALSE/_summary_merged.json`。**.82 一直被外部 co-tenant 占卡；改用同盘空节点 .104（wzc1 共享盘，结果本地可见）跑完。**
- 性质：MemoryLLM 是 Llama-3-8B-**chat** 异基座，chat=False 对它 OOD/不公平（chat=True 才是其原生协议）→ **次要 cross-base 参考行**，不影响任何核心结论。**#68 全部完成。**

### ⏸ #10 — 论文整合（GPU-free，待正文重构定稿）
- InfLLM 行入表（tab_overview/tab_h2h 等）；CoMem LoCoMo errata 改 iter_bm25；**tab_eff caption 按 §0.5 裁决 3/4 修正**：标硬件档「96 GB / H20 级」+ **删「8B dense 128k full-ctx OOM」措辞**（89.36 GB status=ok，仅 30B-A3B MoE 真 OOM）+ headline(7.83×/LoRA-OFF) 与 L20A LoRA-on 控制(2.74×) **分开报告**；BABILong §2.4 同步删 OOM 措辞；§5.1/§5.2 用统一 topk=12/hop=4 + LoRA「4000 steps/2048 tokens/65.5M」口径；正文加 §8 主线叙事 + §5.5 mask 伪代码 + §9 reviewer 预防（retrieval-matched HCache 等价性、conversation-cluster bootstrap、超参冻结声明）；最终 coherent 换表（tab_overview/h2h/scaling/selector/itervt/chunk/crosschunk/slm/locomo）+ 05_experiments 正文。
- 阻塞原因：paper/sections 02–07 正重构 + Overleaf 往返；等定稿后以**本文档为准**整合。

---

## §8 主线叙事（论文 framing，用户 2026-07-25 调整）+ 决定性小表

> **不要讲成「depth-cut 本身提升精度」**。正确主线（6 步）：

1. **检索把任意长的 store 映射到固定 read pack**（iter_bm25 top-12，恒定 read_len≈6.6k）——这是长度可扩展性的来源。
2. **j=0 检索式全重算（retrieved full recompute）质量最好但成本最高**（对检索到的 chunk 做全 36 层重算；LoCoMo judge 41.59，BABILong 65.9/39.1/63.0）。
3. **中层缓存（split at j）降低 write/read 成本，但引入随 j 增大的保真度税**（frozen judge j0 41.59 → j6 32.78 → j9 29.15 → j12 24.52，单调递减）。
4. **轻量 self-distilled LoRA 在同一 j=12 上买回 +13.75 LoCoMo judge**（j12 frozen 24.52 → j12+LoRA 38.27；BABILong qa1 +22.2/qa2 +9.0/qa5 +8.4）——LoRA 在通用 PG19 上蒸馏（teacher=j0 full-recompute，student=j12 mid-recompute），**无检索、无任务数据**。
5. **CoMem = 质量-显存-算力的良好折中**：以固定预算 6657 token 换取 ~5× 显存优势 + 长档速度，且旗舰 LoCoMo judge 38.27 **超过** full-ctx KV oracle（34.59）。
6. **超窗口优势来自「固定检索预算 + 深度复用」的组合，而非单纯深度机制**——各成分贡献随任务而异（LoCoMo 上检索主导、BABILong ≤32k 检索是 tax、深度切分处处需 LoRA 补偿）。

### ★ 决定性小表（论文正文建议直接放，5 行；源 §3.2/§3.3，chat=False, Qwen3-8B）

| config | split j | LoRA | **LoCoMo judge (n=1986)** | BABILong qa1 / qa2 / qa5 |
|---|:---:|:---:|:---:|:---:|
| Retrieved full recompute（j0，No-LoRA） | 0 | ✗ | **41.59** | 65.9 / 39.1 / 63.0 |
| CoMem frozen j6 | 6 | ✗ | 32.78 | 44.6 / 22.6 / 53.9 |
| CoMem frozen j9 | 9 | ✗ | 29.15 | 42.4 / 19.6 / 55.6 |
| CoMem frozen j12 | 12 | ✗ | 24.52 | 33.4 / 18.0 / 60.3 |
| **CoMem distilled j12 + LoRA（旗舰）** | 12 | ✅ | **38.27** | 55.6 / 27.0 / **68.7** |

> 读法：**上 4 行（frozen）随 j 单调掉保真度**（深度税）；**末行 LoRA 在同 j=12 把 24.52 拉到 38.27（+13.75），越过 KV oracle 34.59**。j0 行（41.59）证明「检索 + 全重算」上限最高但最贵——LoRA 让 j=12 中层缓存在低成本下逼近它。

---

## §9 reviewer 预防性补充（P0 建议，2026-07-25 用户提出）

### §9.1 retrieval-matched HCache（P0 #1）— 已由现有实验回答，无需新跑
- **工程等价性**：论文里的 HCache = 「中层缓存 + **无检索**（读全部 chunk）+ 无 LoRA」。给它接上**同一 BM25 检索（同 top-12、同 j、无 LoRA）**得到的，正是 **CoMem adapter-free arm（#65 / §3.1；depth-sweep §3.3）** ——即「HCache-read + 检索」≡ CoMem 去掉 LoRA 的那一支。所以「retrieval-matched HCache vs CoMem-read」这个对照**已经存在**：§3.3 的 frozen j9=29.15 / j12=24.52（检索版）就是它。**建议在论文里直接点明此等价性以预防 reviewer。**
- **补充硬证据（P1 portable-adapter，#75，2026-07-25 LOCAL）**：在 **retrieval-free** HCache（`no_retrieval=True`）read path 上做**单变量 LoRA on/off toggle**（同 node/commit/harness，resume_j=12，chat=False）：

  | arm | 配置 | **LoCoMo judge (n=1986)** | 源 |
  |---|---|:---:|---|
  | A（control，无 LoRA） | HCache j12, no retrieval | **13.29** | `locomo_results/hcache_j12_noLoRA_chatFALSE/scores.json` |
  | B（treatment，+CoMem distilled LoRA） | 同上 + `--lora_adapter …/final --force_lora_with_baseline` | **31.17** | `locomo_results/hcache_j12_LoRA_chatFALSE/scores.json` |

  - **Arm B − Arm A = +17.88（13.29→31.17，2.3×）**，cat4 open_domain 23.31→55.77（+32.46，最大驱动）。**零检索**下 LoRA 独立买回大部分 readout gain（31.17 > adapter-free-with-retrieval 29.15，接近旗舰 38.27）。
  - **意义**：LoRA 不是 CoMem 专用 patch，而是**可移植、与压缩方式无关的「中层重算读出」修复器**——把「LoRA decisive」升级为「reusable KV-decompression/readout adapter」。
  - **诚实 caveat**：Arm A 本地=13.29，canonical HCache headline=8.11（在 diskB .73/.104，跨 node harness/judge 差异 ~5pt）。**单变量结论不受影响**（A/B 同 node/commit，LoRA 唯一变量→ +17.88 干净）；若锚定 8.11 则 delta=+23（仍在预登记区间）。驱动 `scripts/_p1_hcache_lora_toggle.sh`（commit 0b55791）。

### §9.2 LoCoMo 会话级配对 bootstrap / leave-one-conversation-out（P0 #2）— GPU-free，建议补
- **现状**：显著性用的是 **item-level 配对 bootstrap**（judged n=1540，10000 resample，seed=1234）：CoMem 旗舰 vs KVD paired diff=+4.81，95%CI[2.34,7.27]，p<0.0001（§2.5）。
- **建议（预防 reviewer「10 个对话非独立」）**：改用 **conversation-clustered（10-cluster）配对 bootstrap** 或 **leave-one-conversation-out**——resample 单位=对话（cluster）而非单条 QA，正确反映 LoCoMo 仅 10 个独立对话的有效样本量。**GPU-free**：直接用现有 per-item judge 输出（各 `locomo_results/*/judge_cache.jsonl` + preds 里的 conv_id）在 CPU 上重算即可。**旗舰 + KVD 的 judge cache 在 .73（diskB）**（j0/j9 在本地 wzc1），需在 .73 上 CPU 跑或把 cache 取回本地。**待办：整合期补此稳健性分析（不改结论方向，只让 CI 更保守/诚实）。**

### §9.3 超参冻结声明（P0 #3）— 论文须明写
- **如何选 j / top-k / hop**：
  - **split depth j=12**：基于 layer-sweep 分析选定——「可缓存语义上限 vs LM 税」的折中（越深 LM 税越大，单调；见 memory `bottleneck-layer-sweep-monotone` + §3.3 frozen sweep）。**非税最小点**（税最小是 j0，但 j0 无压缩收益）。
  - **top-k=12 / hop=4 / rounds=auto(3)**：hop=4 为定值裁决（`iter_rounds=0`→⌈12/4⌉=3 轮）；全 5 benchmark 统一（§0.5 裁决 1），**不逐 benchmark 调**。
  - **chunk_size=512、sink=bos**：固定。
- **是否 test 前冻结**：**须在论文明确声明**——j/k/hop 是在**分析/开发观察**上选定后**冻结**，主表 5 benchmark 用同一份 config 跑（config-recorded 证据 §0.5 裁决 1）。**⚠️ 待核实并写明**：(a) 选 j/k 时用的具体 dev split（避免「在 test 上调」的质疑）；(b) baseline（StreamingLLM/InfLLM）的调参预算是否对等——目前 StreamingLLM=**等预算 6657 token 严格对齐**、InfLLM=**paper-faithful defaults**（须在论文说明「baseline 用其原论文推荐配置，未在本 test 上额外调优」）；(c) **tab_selector 是否用 per-cell peak-k**（若消融表某些 cell 取了各自最优 k，须在 caption 标注，否则=cherry-pick 质疑）——**此项整合期务必核对 tab_selector 生成脚本**。

### §9.4 block-diagonal mask 描述（P0 #4）— 已补
- read-pack 构造 + causal/cross-chunk mask 的完整描述（含伪代码 + 出处 + block_diagonal 消融）见 **§5.5**。论文正文建议配一张 read-pack 示意图 + 该伪代码。

---

*（本文档随新结果更新；#68 完成 / 论文整合推进时同步刷新对应节。四项口径裁决见 §0.5，主线叙事见 §8，reviewer 预防见 §9。）*
