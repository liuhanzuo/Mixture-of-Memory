# InfLLM Baseline Results (Paper A / CoMem head-to-head)

用户指令：InfLLM 是 flagship baseline，需在全部 5 benchmark（RULER/BABILong/LongBench/LongEval/LoCoMo）上按统一 CoMem 协议测。
本文件汇总 InfLLM measured 结果（chat_template=True, enable_thinking=False, seed 42, bf16, 官方 scorer）。
InfLLM 配置：thunlp/InfLLM paper-faithful defaults — block_size=128, n_init=128, n_local=4096, topk=16, repr_topk=4, chunk_size=8192, base=1e6。
Backbone = Qwen3-8B（同 CoMem）。结果原始文件在 .73/diskB `ruler_results/infllm_8b/`。

## RULER (n=100, string_match_all) — ✅ COMPLETE (2026-07-19)

| Task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| niah_single_2   | 100  | 99   | 95   | 54   | 53   |
| niah_multikey_1 | 99   | 93   | 65   | 37   | 24   |
| variable_tracking | 100 | 98.2 | 90.8 | 0    | 0    |

- 8-32k：earlier per-cell dirs `ruler_results/infllm_8b/{niah_single,niah_multi,vt}_{8k,16k,32k}/`（CSV recall col 聚合 n=100）。
- 64k/128k：`ruler_results/infllm_8b/ruler_infllm/*_shard{0..7}of8.json` + `_summary_shard*`（n=sum shards=100）。
- **story**：InfLLM 在 ≤32k 强（sliding-window + block-memory retrieval 有效），但 fixed retrieval budget（topk=16 blocks×128 + 4096 local ≈ 6.3k）在 64k-128k 崩（single 54/53、multikey 37/24、VT→0）。对比 CoMem 恒定 read 下 64k-128k 仍 ~96-100 → InfLLM 是"训练-free block memory 在长档失效"的强对照。
- **vs tab_h2h 其他行**（64k/128k）：CoMem single 100/96、KV-Direct 0（窗口溢出）、HCache 0、MemoryLLM 6/16。InfLLM 53/54 介于两者——比崩到 0 的强，比 CoMem 弱。

## LongBench (6-ds, F1, chat+no-think) — ✅ COMPLETE (2026-07-19, `longbench_results/infllm_8b/scores.json`)

| ds | narrativeqa | qasper | hotpotqa | 2wikimqa | multifieldqa_en | musique | **AVG** |
|---|---:|---:|---:|---:|---:|---:|---:|
| InfLLM F1 | 21.45 | 47.10 | 57.43 | 40.72 | 52.32 | 30.24 | **41.54** |
| (EM) | 4.5 | 18.0 | 40.5 | 33.5 | 22.0 | 21.5 | 23.3 |

- n=200/ds（multifieldqa_en n=150）= 全集。**story**：InfLLM AVG F1 **41.54 ≈ KV-Direct full-ctx(42.97)，且 > CoMem(35.79) > HCache(19.27)**。与 InfLLM 强 ≤32k RULER 一致（LongBench 多为 ≤32k）→ InfLLM 在中等长度非常强。

## LongEval (8k-128k, n=50) — ✅ COMPLETE (REDO, max_new_tokens=48, 2026-07-19 02:xx)

| 8k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 0.600 | 0.260 | 0.120 | 0.040 | 0.020 |

- 30/13/6/2/1 correct of 50（redo 全 5 档确认完成，128k=1/50=0.02，非之前的 stale 0）。**story**：InfLLM 在 LongEval(lines-retrieval)随长度**急剧崩**——8k 0.60→32k 0.12→64k 0.04→128k 0.02。对比 **CoMem 1.00/0.96/1.00/0.94/0.98**(paper, per-task 最优 k∈{4,6})→ InfLLM 完全被碾压。与 RULER 64k+ 崩(VT→0)+ block-memory 固定 budget 一致。
- ⚠️ **首跑作废(截断 bug)**：max_new_tokens=16 默认全 0.000——InfLLM chat 模式回显长随机 label(如 enotbfnqg-kqexdvxj 多 token)耗尽预算,`extract_prediction`(首个 ≥4 位数字串)找不到→pred=""。REDO 用 `--max_new_tokens 48` 修复:8k 50/50 非空 pred,输出 "...is <836925>" 正确抽取。redo ALL DONE 02:10。
- ⚠️ **32k/64k 的 pred="" 是真失败非 artifact**：InfLLM 输出 "...is **not present** in the provided context"(真检索 miss),不是截断。

### ★ LongEval 全 baseline cohort（统一协议 chat+no-think, max_new_tokens=48, n=50, 8-shard）——待入 tab_longeval

| method | 8k | 16k | 32k | 64k | 128k | 备注 |
|---|---:|---:|---:|---:|---:|---|
| CoMem (per-task 最优 k∈{4,6}) | 1.00 | 0.96 | 1.00 | 0.94 | 0.98 | paper headline（retrieval，per-task 最优 k，n=50） |
| **CoMem (iter_bm25 matched fixed-selector)** | **0.73** | **0.76** | **0.79** | **0.72** | **0.76** | ✅ 2026-07-19 补齐（coder a23655ca 补 64k/128k）：**恒定 read≈6657 的 matched-selector 行**，n=100，chat+no-think，max64。4k=**0.95**（4k 全落进 6144-token read budget→高）。**8k-128k 约 0.72-0.79 FLAT（length-invariant，恒定 read）**，非 headline 0.94-1.00（后者用 per-task 最优 k）。仍碾压全 baseline @128k |
| **InfLLM** | 0.60 | 0.26 | 0.12 | 0.04 | 0.02 | block-memory retrieval，随长度崩 |
| **HCache** | 0.02 | 0.00 | 0.02 | 0.00 | 0.00 | ✅ REDO 2026-07-19 02:40（`hcache_8b_chatnothink/_summary_merged.json`）retrieval-free 全崩 |
| KV-Direct (full-ctx, j=0) | 1.00 | 0.98 | 0.96 | 0.36 | 0.00 | ✅ REDO 2026-07-19 03:13（`kvdirect_8b_chatnothink/`）**128k 窗口溢出→0 已验真**（read_len 131471≫40960 native window）；与 paper 断言 1.00/0.96/1.00/0.34/0.00 在 n=50 噪声内一致 |

- **★ HCache LongEval 已验真非截断**（铁律2）：max_new_tokens=48 足够（输出完整 "line X: REGISTER_CONTENT is <596193>"），preds 非空。近 0 acc 是**真失败**——retrieval-free mid-layer recompute（chunk-local RoPE 写 + recompute 上层）**破坏精确 line→value 映射**：模型要么幻觉一个似是而非的错 6 位数，要么答 "line not present"。与论文 HCache 故事（"crashes otherwise, retrieval-missing"）一致。
- **★ LongEval 三段对照故事**：8k 上 **有 retrieval 才行**——CoMem ~1.0 / InfLLM(block-memory retrieval) 0.60 / HCache(no retrieval) 0.02。随长度：CoMem 恒定 ~1.0（per-task k）、InfLLM 急崩（fixed budget）、HCache 恒 0、KV-Direct 128k 窗口溢出崩 0。→ LongEval = "retrieval-based memory 的主场"最强证据。
- **★ matched-selector 诚实点（tab_longeval 必标）**：把 CoMem 也钉到统一 iter_bm25 恒定-read 协议（同 baseline 口径，非 per-task 调 k）后，CoMem = **8k-128k 约 0.72-0.79 FLAT**（4k=0.95 因全落进 read budget）——**length-invariant 但非满分**。paper 若引 headline 0.94-1.00 须注明"per-task 最优 k∈{4,6}"；apples-to-apples matched 行是 0.73/0.76/0.79/0.72/0.76。**即便 matched，128k=0.76 仍碾压全 baseline**（InfLLM 0.02、MemoryLLM 0.04、HCache 0、KV-Direct 0）。

### ★★ CoMem zero-shot vs +adapter（"adapter 是关键杠杆" ablation，✅ 2026-07-19 补齐 4k-128k）

同 iter_bm25 chat+no-think 协议，唯一变量 = 有无 distilled LoRA adapter（+ 训练把 split 从 zero-shot 安全的 j=9 推到 j=12）：

| length | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|---:|
| **CoMem +adapter**（j=12, LoRA r32/α64 4000步 distill） | **0.95** | **0.73** | **0.76** | **0.79** | **0.72** | **0.76** |
| **CoMem zero-shot**（j=9, 无 adapter） | 0.14 | 0.06 | 0.05 | 0.07 | 0.06 | 0.07 |

- **story**：zero-shot 全档钉在 **0.05-0.14 near-floor**，adapter 把每一档拉到 0.72-0.95 → distilled adapter 是 **~10-15× 杠杆**，是 retrieval-over-chunks LongEval 能工作的**关键条件**（不是恒定 read 本身，是 read 内容被 adapter 学会"从检索到的 chunk 里精确定位 line→value"）。
- **铁律2 验证（coder ab16c2b + main 复核）**：merge=独立复算逐档一致，每档 n=100 索引 0-99 无重无缺。zero-shot 错误性质：**长档（8k-128k）= well-formed 错值**（正确点名 line label 如 `vonculzso-tpbeioj` 但读出错的 6 位数 460523≠期望 104362）=真检索/读取失败；**4k 异常** = 61 空/拒答（"line not present"，zero-shot 短档格式跟随 artifact，非 merge bug）→ 4k=0.14 被拒答压低，仍 near-floor。
- ⚠️ **j 值差异（诚实标注）**：zero-shot=j9（zero-shot 安全 readout split），+adapter=j12（训练后可推更深）。所以杠杆 = adapter LoRA **＋** 训练允许的更深 split 的联合效应，不是纯"加不加 LoRA"单变量。
- raw：`.73:longeval_results/qcmem_8b_zs_iter_chatnothink/longeval_8b_zs/`（4k-32k 4-shard + 64k/128k 8-shard，`_summary_merged.json`）。DONE marker `logs/qcmem_longeval_zs_iter_64k128k_DONE`。

## LoCoMo (n=1986, F1/EM/acc; GPT-4o judge deferred) — ✅ COMPLETE (2026-07-19 01:56)

| LoCoMo (same .73 scorer) | OVERALL F1 | EM | acc | cat1 multi_hop | cat2 single_hop | cat3 temporal | cat4 open_domain | cat5 adversarial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **InfLLM** | **25.76** | 11.33 | **26.38** | F1 29.96 (n282) | 29.70 (n321) | 20.41 (n96) | 31.65 (n841) | 10.31 (n446) |
| CoMem (iter_bm25/chat) 对照 | 19.51 | 5.99 | 28.65 | 19.59 | 20.14 | 11.59 | 29.77 | 1.35 |

- **★ apples-to-apples 关键结论**：同 .73 scorer 下 InfLLM F1 25.76 / acc 26.38 **≈** CoMem(iter_bm25,chat+no-think) F1 19.51 / acc 28.65——**大致打平**(InfLLM token-F1 略高,CoMem acc 略高),**非 3× 差距**。
- **★ 论文 errata 发现**：`05_experiments.tex:167` 写 CoMem LoCoMo **F1 9.05 / acc 24.1** 是**过时数**(旧 bm25 run,已被用户 iter_bm25 指令作废)。canonical iter_bm25 重打分 = **19.51 / 28.65**。→ tab_locomo + 05_exp 叙事须把 CoMem 更新到 iter_bm25(19.51/28.65),同时加 InfLLM 行(25.76/26.38)。
- **story**：InfLLM cat4 open_domain F1 最强(31.65)但 acc 只 33.41(CoMem 50.54 强);cat5 adversarial InfLLM 10.31 vs CoMem 1.35(CoMem 几乎全拒答→token-F1 极低)。judge 待补(GPT-4o CORRECT/WRONG,与 CoMem 39.5 口径)。
- score_only stale-signature bug：.73/diskB 的 `eval_qcmem_locomo.py` 是旧 `run_scoring(output_dir, use_bertscore=False)` 签名(未从 wzc1 同步 judge 版)→InfLLM harness 传 `use_llm_judge=`/`judge_workers=` 崩。generation 已完成(8 shard=1986 preds),直接用旧签名 CPU 重打分修复。**待办**:同步 wzc1 新版 eval_qcmem_locomo.py 到 diskB。

## BABILong (qa1/qa2/qa5 × 0k-32k, n=100/cell, 官方 compare_answers) — ✅ COMPLETE (2026-07-19)

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa1 | 100 | 94 | 92 | 90 | 85 | 59 | 37 |
| qa2 | 59  | 56 | 54 | 58 | 48 | 43 | 31 |
| qa5 | 80  | 77 | 75 | 74 | 78 | 64 | 55 |

- 原始 flat CSV：`babilong_results/infllm_n100/qa{1,2,5}_{L}_..._shard{0..3}of4.csv`(每 cell 4 shard=100)；官方 `third_party/babilong-pkg/babilong/metrics.compare_answers`(+TASK_LABELS[task])重打分(非 re.search)。⚠️ 该 flat 布局不匹配 `score_nested_babilong.py` 的 `<root>/<run>/<run>_<length>/` 嵌套预期→用内联复刻 score_cell 打分。
- **story**：InfLLM qa1 短档强(0k=100,8k=85)但 16k→59/32k→37 明显掉;qa2(2-fact)本就弱(≤59);qa5(3-fact relational)最耐长(32k 仍 55)。与 CoMem 对照见 tab_babilong(CoMem qa1 长档更强、qa5 relational 更强)。

---
## ✅ InfLLM 全 5-benchmark sweep 完成（2026-07-19）——待入论文
5 个 benchmark 全部 measured(统一协议:chat_template=True, enable_thinking=False, seed 42, bf16, 官方 scorer, Qwen3-8B, thunlp/InfLLM paper-faithful defaults)：**RULER✅ / LongBench✅ / LoCoMo✅ / LongEval✅(redo) / BABILong✅**。
下一步(GPU 无关,纯论文写作)：
1. 加 InfLLM 行到 `tab_h2h.tex`(RULER single/multikey 8k-128k)、`tab_longbench.tex`、`tab_longeval.tex`、`tab_locomo.tex`、`tab_babilong.tex`。
2. **同步修 CoMem LoCoMo errata**：05_exp:167 F1 9.05/acc24.1(旧bm25,作废)→ iter_bm25 **19.51/28.65**。
3. `05_experiments.tex` 叙事加 InfLLM 对照：≤32k 强(RULER/LongBench 追平甚至超 CoMem)但 64k+ block-memory 崩(RULER VT→0、LongEval→0、LoCoMo 打平)——"training-free block memory 在长档失效" 的强对照。
