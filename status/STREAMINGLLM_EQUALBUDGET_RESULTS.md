# StreamingLLM Equal-Budget Results (Paper A / tab_slm extension)

统一协议：Qwen3-8B backbone，equal budget = sink=4 + window=6653 = **6657 tokens ≈ CoMem 恒定 read**（≈16.9 GB），
`--max_new_tokens 48`，chat+no-think，n=50，8-shard，RULER `string_match_all`。driver `scripts/eval_ruler_streamingllm.py`。
结果原始文件在 .73/diskB `ruler_results/streamingllm_qwen_{niah,mkvt,32k}/`。

## Equal-budget RULER grid（StreamingLLM vs CoMem）

| task | 8k | 16k | 32k | 64k | 128k | 源 |
|---|---:|---:|---:|---:|---:|---|
| **SLM** niah_single_2   | 90.0 | 42.0 | 18.0 | 16.0 | 4.0  | `streamingllm_qwen_niah`(8/16/64/128) + `streamingllm_qwen_32k` |
| **SLM** niah_multikey_1 | 86.0 | 48.0 | 26.0 | 8.0  | 6.0  | `streamingllm_qwen_mkvt`(8/16/64/128) + `_32k` |
| **SLM** variable_tracking | 38.0 | 3.6 | 1.2 | 0.0  | 0.0  | `streamingllm_qwen_mkvt`(8/16/64/128) + `_32k` |
| CoMem single (对照) | 100 | 100 | 100 | 100 | 100 | tab_slm/tab_h2h |
| CoMem multikey (对照, tab_h2h) | 88 | 92 | — | (64k/128k 见 tab_h2h) | | tab_h2h |
| CoMem vt (iter_bm25, 对照) | 95.2 | 93.8 | 96.8 | (95.x) | (95.x) | tab_itervt |

- **★ tab_slm story（强化）**：equal budget 下唯一变量 = 保留**哪些** token。
  - **单针（single）**：recency 截断 90→4（8k→128k），CoMem 恒 100 → 128k **25× gap**（已在正文）。
  - **多针（multikey）**：SLM 86→6，CoMem 88-100 恒定 → recency 截断**丢中段多事实针**，长档几乎全崩。
  - **变量追踪（vt）**：SLM 38→0（16k 就崩到 3.6），CoMem iter_bm25 93-97 恒定 → 引用链需要 relevance-based 选取 + 迭代检索，recency 完全无能。
- **结论**：fixed budget 必要但不充分；**relevance-based selection**（CoMem）才让 budget 起作用。多针 + tracking 比单针更能拉开差距（single 已 25×，multikey/vt 差距更极端：SLM 长档→0-6 vs CoMem 88-97）。

## LoCoMo（equal-budget，✅ 完成+验证 2026-07-19，n=1986）

同 `.73` scorer（F1/EM/acc，无 GPT-4o judge），`locomo_results/streamingllm_8b_chatnothink/scores.json`。sink4+window6653=6657 恒定 budget。

| method | F1 | EM | acc | cat1 mh | cat2 sh | cat3 temporal | cat4 open | cat5 adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **StreamingLLM** | **12.73** | 5.24 | **17.57** | 8.74 | 5.50 | 7.33 | 11.84 | 23.32 |
| CoMem (iter_bm25) 对照 | 19.51 | 5.99 | 28.65 | 19.59 | 20.14 | 11.59 | 29.77 | 1.35 |
| KV-Direct (full-ctx) 对照 | 40.06 | 19.59 | 43.05 | — | — | — | — | — |

- **story**：equal-budget recency 截断在多会话记忆上 F1 12.73 < CoMem 19.51 < full-ctx KV-Direct 40.06 → 同 budget 下 recency 保留策略弱于 relevance-based 检索。cat5 adversarial StreamingLLM 23.32 高（recency 窗口外内容"看不到"→更常拒答/答不知道，与 adversarial gold 撞对），但 cat1-4 全线低于 CoMem。

## BABILong（equal-budget，✅ 完成+验证 2026-07-19，n=100/cell，官方 compare_answers，empty=0）

`babilong_results/babilong_streamingllm_8b_chatnothink/scores.json`。

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SLM** qa1 | 100 | 94 | 92 | 89 | 48 | 30 | 23 |
| **SLM** qa2 | 60 | 56 | 54 | 57 | 34 | 17 | 3 |
| **SLM** qa5 | 81 | 77 | 75 | 76 | 75 | 67 | 53 |

- 铁律2：empty_output=0 全 21 cell（n_shards=8）。qa1/qa2 逐值本会话直读 scores.json 核实；qa5 0k=81 本读确认，1k-32k 承前次全读。
- **story**：≤4k（全落进 6657 budget）qa1 ~89-100 强，**8k 起 recency 截断丢中段 supporting fact → qa1 48/30/23、qa2 34/17/3 急崩**；qa5(3-fact relational) 最耐长(32k 仍 53)因关系事实更可能落在近窗。对比 KV-Direct(full-ctx) qa1 32k=71、CoMem 见 tab_babilong → **equal-budget 下 recency 在需要中段精确事实的长档系统性失败**，与 RULER(single 90→4、vt 38→0)、LongEval 一致：fixed budget 必要不充分，relevance selection 才让 budget 起作用。

## LongBench（equal-budget，✅ 完成+验证 2026-07-19，n=1150，官方 qa_f1，empty=0）

`.73` 8-shard 跨 8 GPU，`longbench_results/streamingllm_8b_chatnothink/scores.json`。sink4+window6653=6657 恒定 budget（`kept_tokens` max=6657，长档全部截到 budget，短档保留全文）。官方 LongBench scorer（`eval_longbench_mem_space.run_scoring` → `compute_f1_multi`/`compute_em_multi`，禁 re.search），chat+no-think，Qwen3-8B。6 数据集与 InfLLM/KV-Direct/CoMem/MemoryLLM/HCache cohort 完全相同。

| dataset | F1 | EM | n |
|---|---:|---:|---:|
| hotpotqa | 50.25 | 38.50 | 200 |
| narrativeqa | 20.51 | 6.00 | 200 |
| qasper | 46.52 | 16.50 | 200 |
| multifieldqa_en | 43.04 | 18.00 | 150 |
| 2wikimqa | 42.36 | 34.50 | 200 |
| musique | 20.52 | 12.00 | 200 |
| **AVERAGE** | **37.20** | **20.92** | 1150 |

对照（同 6-ds LongBench AVG F1，同 scorer/协议）：**KV-Direct(full-ctx) 42.97 > InfLLM 41.54 > StreamingLLM 37.20（本次）> CoMem 35.79 > HCache 19.27 > MemoryLLM 12.80**。

- 铁律2：全 6 ds empty_output=0 / oom=0 / no_answers=0（独立 glob 8-shard jsonl dedup by index 复算，TOTAL n=1150），preds well-formed；官方 qa_f1 scorer 直读 scores.json 一致。
- **story**：real-doc QA 上 equal-budget recency（StreamingLLM 37.20）反而 **> relevance-selection CoMem 35.79**，且逼近 full-ctx（KV-Direct 42.97 的 87%）——与 RULER/LoCoMo/BABILong 里 recency 长档崩盘的结论**并不矛盾**：LongBench QA 上下文中位数远短（大量样本 <6657 token → `kept_tokens` < budget = 未截断=近似 full-ctx），且答案证据多在文档首尾（尾部落进 recency 窗、首部落进 sink），所以 recency 截断损失小；InfLLM/CoMem 的检索优势在这种"证据非中段深埋"的短-中档 QA 上体现不出来。真正拉开 recency vs relevance 的是 RULER 多针/vt（SLM→0-6 vs CoMem→90+）、LoCoMo 多会话（12.73 vs 19.51）、BABILong 长档中段事实（qa1 32k 23 vs KV-Direct 71）——即"证据在被丢弃的中段"时 recency 才系统性失败。故 LongBench 单格补齐后，equal-budget 叙事应强调：**fixed budget 是否够用取决于证据位置**——LongBench 首尾证据下 recency 足够；needle/多针/长档中段证据下必须 relevance-based selection。

## 完成记录
- 2026-07-19 06:5x — LongBench equal-budget cohort 完成+铁律2 验证（矩阵最后一格补齐；AVG F1 37.20，n=1150，empty=0）。raw `.73:longbench_results/streamingllm_8b_chatnothink/`。
- 2026-07-19 06:0x — LoCoMo + BABILong equal-budget cohort 完成+铁律2 验证（补齐 StreamingLLM 全 benchmark，现 RULER/LongEval/LoCoMo/BABILong 齐；LongBench 可选）。见 status/BASELINE_MATRIX_COMPLETE.md。
- 2026-07-19 04:05:59 — mk+vt 8k/16k/64k/128k ALL DONE（`streamingllm_qwen_mkvt/_merged.txt`）。Task #13 completed。
- 2026-07-19 04:09:02 — 32k-fill（single/mk/vt）launched（`logs/streamingllm_32kfill.sh` → `streamingllm_qwen_32k`）。补齐 32k 列后本表即完整 5-length × 3-task。

## 待办（Task #10 paper 整合时一并处理）
- 32k 列跑完 → 填入本表 + `tab_slm.tex`（当前 tab_slm 仅 single 90/42/16/4，缺 32k + 缺 multikey/vt 两行）。
- `05_experiments.tex` §Equal-budget（L73-83）叙事可从"single 25× gap"扩展到"multi-fact/tracking 差距更极端"（multikey SLM→6 vs CoMem→~90+；vt SLM→0 vs CoMem→~95）。
