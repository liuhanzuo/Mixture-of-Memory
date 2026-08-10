# Paper A — 可复现性超参总表（配置 / 训练 / 评测 三层）

> **生成 2026-07-24（main 从一手 config/源码/日志/结果文件核账，非重跑）。** 目标：任何人拿这张表 + 代码就能复现 CoMem 旗舰的**配置、训练、评测**三层。每个值附 `文件:行号` 出处；无出处的写 NOT FOUND。
> **口径双支柱（全论文强制）**：`chat_template=False`（所有模型是 continue-train BASE LM，无 SFT/RL，套 chat 模板注入 OOD token 不公平）+ CoMem `selector=iter_bm25`。
> **backbone**：Qwen3-8B（唯一例外 MemoryLLM=Llama-3-8B-chat 异基座对照）。

---

## ★ P0 冲突裁决（5 项，reviewer 最关心）

### P0-1 旗舰 chunk size = **512**（权威）
- 一手：`outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json:11`（chunk_size=512）；`ruler_results/bench_qcmem_vs_fullctx.json:8`；`paper/sections/05_experiments.tex:5`（"split j=12 with chunk size 512"）+ 14 处证据。
- **1024 仅出现在消融表**（tab_selector / tab_itervt / tab_crosschunk / tab_chunk 对比行）。task #56 的 "chunk1024" = tab_chunk 消融行，**不是旗舰**。
- **裁决：主表/旗舰一律 512。**

### P0-2 LoRA "4k" = **4000 steps**，训练序列长 = **2048 tokens**（无冲突）
- adapter 名 `r32_4k` 的 `4k` = 4000 training steps（`distill_args.json:16` total_steps=4000），**不是**序列长度。
- 训练序列长 = `(n_ctx+1)×chunk_size = (3+1)×512 = 2048 tokens`（`distill_args.json:10-11` n_ctx=3/chunk=512；`train_qcmem_distill.py:228` window_len 公式）。
- **与论文 "2,048-token windows" 完全一致，无冲突。** （launch 脚本默认 N_CTX=7→4096，但此 run 覆写为 3→2048，distill_args.json 是永久存档权威值。）

### P0-3 训练步数 = **4000**（config + log 双证，无冲突）
- config：`distill_args.json:16` total_steps=4000。
- log：`logs/qcmem_distill_qwen_j12_r32_4k.log:421` `step 4000/4000 loss 0.0555 lr 0.00e+00`；`:422` saved step4000；`:423` final adapter。final ckpt = step4000。

### P0-4 效率实验主报告 = **chunk 512 那套**（权威）
- tab_eff.tex 权威源 = `ruler_results/bench_qcmem_vs_fullctx.json`（`status/QCMEM_PAPER_DRAFT.md:5` 明确）。`tab_eff.tex:1` 注释 "chunk 512 (verified)"，数字与 JSON 完全吻合。
- 128k：加速 **7.83×**（7.8299，`bench...json:104-120`），CoMem 峰值 **18.26 GB**，full-ctx **89.36 GB**（L20A 183GB 上 status=**"ok" 不 OOM**）。
- chunk1024 的 "128k full-ctx OOM vs CoMem 20.3GB" 是在 **H20（97.8GB）** 上的独立观测（SESSION_HANDOFF:30，task#56），属**消融补充描述**，不进 tab_eff。
- **caption "OOMs" = 指 H20 部署场景，无冲突**：`tab_eff.tex:17` caption "grows to 89 GB and **OOMs**" 指的是 **H20(97.8GB) 部署**——full-ctx 128k 的 ~89GB 峰值在 H20 上会 OOM（chunk1024 128k full-ctx H20 已实测 OOM，SESSION_HANDOFF:30）。L20A(183GB) bench 之所以测到 89GB 且 status=ok，只是因为该卡显存足够大能容下并测出此峰值；同一 89GB footprint 放到 H20 就 OOM。故 caption 正确，**不是冲突，无需改**。（可选：caption 里点明 "on H20-class 80GB GPUs" 更清晰，但非必须。）

### P0-5 StreamingLLM 等预算 = **6657 tokens**，严格对齐 CoMem read pack（无冲突）
- SLM：sink=4（`_run_streamingllm_ruler_8gpu.sh:34`）+ window=6653（`:35`）= budget **6657**（`STREAMINGLLM_EQUALBUDGET_RESULTS.md:3`）。实现 = 截断近似 `concat(ids[:,:4], ids[:,-6653:])`（`eval_ruler_streamingllm.py:22-28`；transformers 5.x 移除 SinkCache）。
- CoMem read pack `seq_len=6657`（sink(1) + topk×chunk_h_j + query_h_j，`bench...json:113`，全长度恒定）。
- **SLM budget(6657) = CoMem read pack(6657)，严格等预算，单一变量=保留哪些 token。** CoMem 实测平均 read≈6.5k（8k cell avg_read_len=6565，`QCMEM_STATS_APPENDIX_chatFALSE.md:127`）。

---

## 一、[Backbone]
| 项 | 值 | 出处 |
|---|---|---|
| 模型 | Qwen3-8B（`models/Qwen--Qwen3-8b`） | `models/Qwen--Qwen3-8b/config.json` |
| 架构 | qwen3, num_hidden_layers=**36 (L)**, hidden_size=4096, head_dim=128 | config.json |
| 注意力 | num_attention_heads=32, num_key_value_heads=8（GQA） | config.json |
| MLP | intermediate_size=12288 | config.json |
| 词表 | vocab_size=151936 | config.json |
| 位置 | max_position_embeddings=**40960**（原生窗口）, rope_theta=1e6, **rope_scaling=null（YaRN 未激活）** | config.json |
| dtype | bfloat16 | config.json |
| attn_impl | sdpa | eval/train 全默认 |
| ⚠️ 注意 | 论文里 131072 = **未激活的 YaRN 上限**，实际原生 native context = 40960（`05_experiments.tex:6-7`）。KVD RULER 128k=0 是窗口溢出非 OOM。 | `05_experiments.tex:6-7` |

---

## 二、[CoMem flagship]（旗舰：+distilled LoRA, j12）
| 项 | 值 | 出处 |
|---|---|---|
| split depth j | **12**（缓存第 12 层 hidden；读时重算 layers[12:36]=24 层） | `distill_args.json:3` resume_j=12 |
| chunk_size | **512** | `distill_args.json:11`（P0-1） |
| LoRA adapter | `outputs/qcmem_distill_qwen_j12_r32_4k/final` | 目录 |
| selector | **iter_bm25** | `QCMEM_STATS_APPENDIX_chatFALSE.md:6-8` |
| topk（检索 chunk 数） | **12**（RULER/BABILong 主）；per-task 最优：LoCoMo k=8、LongEval k=4–6 | STATS_APPENDIX + eval 脚本 |
| iter_hop_topk | **4（全 benchmark 统一，2026-07-24 用户裁决"就用4"）**：RULER/LongEval/LongBench/LoCoMo 本就=4；BABILong 脚本默认已由 2→4 修正（`eval_qcmem_babilong.py:905` argparse + `:347` 函数签名，commit `5158e70`）。✅ 旗舰 BABILong 已 hop=4(3 轮) 重跑完成(#70，dir=`qcmem_j12_iter_bm25_chatFALSE_ad_hop4`)：**qa1 55.6 / qa2 27.0 / qa5 68.7**（旧 hop=2/6 轮为 53.6/25.6/66.7，三档均略升 +1.4~2.0，结论稳定），已回填 §D/headline/PAPERA_ALL_RESULTS。 | 脚本默认（已修） |
| **iter_rounds** | **0 = auto** → `rounds = ⌈topk/hop⌉ = ⌈12/4⌉ = 3 轮`（**不是单遍！**） | `eval_qcmem_babilong.py:361` `rounds = int(iter_rounds) if iter_rounds>0 else -(-k//hop)` |
| 迭代检索定义 | round1: query→BM25 top-hop；后续 round: 用**上一轮新加入 chunk 的 token 作 frontier query**（one-hop 扩展）→ 再 BM25 top-hop；dedup via selected_set；早停：剩余≤0 / frontier 空 / 无正分候选；累计到 topk=12 上限 | `eval_qcmem_babilong.py:361-393` |
| sink | **1 个 BOS token**（sink=bos） | Agent B `qcmem_model.py` |
| pack 顺序 | `[sink BOS h_j] → [selected h_j ×topk, 按文档顺序排] → [query h_j]`；causal | `qcmem_model.py:515-525` |
| read 位置编码 | 读时用**全新连续 RoPE 0..H-1**（非原始位置） | `qcmem_model.py:529` |
| cross-chunk attention | **full（block_diagonal=False 默认）** | `qcmem_model.py:89,117-132` |
| h_j 存储 | 模型 dtype（bf16），无 offload/量化 | `qcmem_model.py:140,392` |
| chunk-local RoPE（write） | 每 chunk 位置从 0 开始（`positions=arange(T)`） | `qcmem_model.py:384` |
| **实际 read budget** | pack 长度 = sink(1)+topk×?+query；128k 恒定 **seq_len=6657**；实测平均 read≈6.5k（8k cell=6565） | `bench...json:113`, `STATS_APPENDIX:127`（P0-5） |

---

## 三、[BM25]（iter_bm25 底层打分）
| 项 | 值 | 出处 |
|---|---|---|
| 函数 | `_bm25_scores(docs, query_ids, k1=1.5, b=0.75)` | `run_babilong_mem_space.py:754` |
| k1 | **1.5**（硬编码签名默认） | `run_babilong_mem_space.py:754` |
| b | **0.75** | `run_babilong_mem_space.py:754` |
| 操作对象 | **token IDs**（非文本；无 lowercase/去停用词/stemming） | `run_babilong_mem_space.py:754-779` |
| IDF | Robertson IDF + 1 平滑 | `run_babilong_mem_space.py:779` |
| doc 单位 | 每个 chunk 的 token IDs = 一个 doc；query = bare_question token IDs | Agent B |

---

## 四、[LoRA]（distilled LoRA 结构 + 蒸馏 + 训练成本）
### 结构
| 项 | 值 | 出处 |
|---|---|---|
| rank / alpha / dropout | 32 / 64 / 0.0 | `distill_args.json:5-7` / `final/adapter_config.json:44,46,54` |
| target modules | q,k,v,o,gate,up,down（7 个） | `distill_args.json:8` / `adapter_config.json:57-65` |
| 注入层 | **仅 layers 12–35（24 层）**；layers 0–11 无 LoRA 且 frozen | `adapter_config.json:17-41` layers_to_transform=[12..35] / `log:14` |
| 可训练参数 | **58.20M = backbone 的 0.71%** | `log:14` / `FLAGSHIP_TRAINING_COST.md:17-18` |
| backbone | **完全 frozen** | `train_qcmem_distill.py:108` / `FLAGSHIP_TRAINING_COST.md:10` |

### 蒸馏
| 项 | 值 | 出处 |
|---|---|---|
| teacher | 同一 Qwen3-8B 实例（无第二拷贝），`disable_adapter()`+`no_grad()` | `train_qcmem_distill.py:530-532` |
| teacher split j | **0**（full-context 全深度） | `train_qcmem_distill.py:533` |
| teacher top-k logits | **64** | `distill_args.json:13` / `train_qcmem_distill.py:338` |
| loss | `0.6·KL(teacher‖student) + 0.4·KL(student‖teacher)`，在 teacher top-64 support 上，仅 query segment | `train_mem_space_dolmino_cpt.py:2585-2606` |
| distill_lambda | 0.6 | `distill_args.json:14` |
| CE 权重 | 0.0（无 hard-label CE） | `distill_args.json:15` |
| temperature | NOT FOUND（无字段；KL 在原始 logits 上算，T=1 等效） | `train_mem_space_dolmino_cpt.py:2597-2600` |
| 数据 | PG19 train（`data/pg19_train.jsonl`），流式无限循环 step-bounded | `distill_args.json:9` / `train_qcmem_distill.py:236` |

### 训练超参
| 项 | 值 | 出处 |
|---|---|---|
| seq length | **2048 tokens**（(3+1)×512） | `distill_args.json:10-11`（P0-2） |
| per-GPU batch | 1 | 代码结构 / grad_accum=1 |
| grad accum | 1 | `distill_args.json:20` |
| effective global batch | **8 samples/step**（8 GPU） | `log:4` world_size=8 |
| total steps | **4000** | `distill_args.json:16`（P0-3） |
| optimizer | AdamW, betas=(0.9,0.95) | `train_qcmem_distill.py:473,475` |
| peak lr | 8e-5 | `distill_args.json:17` |
| weight decay | 0.0 | `distill_args.json:19` |
| warmup | 100 steps linear | `distill_args.json:18` / `train_qcmem_distill.py:486-487` |
| scheduler | warmup 后 cosine → 0 at step4000 | `train_qcmem_distill.py:488-489` |
| grad clip | 1.0 | `distill_args.json:21` |
| seed | 42 | `distill_args.json:28` |
| dtype | bfloat16 | `distill_args.json:27` |
| grad checkpointing | False（此 run） | `distill_args.json:22` |
| attn_impl | sdpa | `distill_args.json:31` |

### 训练成本
| 项 | 值 | 出处 |
|---|---|---|
| GPU | 8× NVIDIA L20A（183GB/卡），单节点 DDP | `FLAGSHIP_TRAINING_COST.md:48` / `log:4` |
| throughput | ~24.5 samp/s（global） | `log:421` |
| wall-clock | **~22 分钟**（吞吐推算；log 无时间戳） | `FLAGSHIP_TRAINING_COST.md:54` |
| 峰值显存 | NOT FOUND（log 无 nvidia-smi） | looked in log 全文 |
| 总 token | 4000×8×2048 = **65.5M tokens** | `FLAGSHIP_TRAINING_COST.md:68` |

---

## 五、[Adapter-free]（frozen backbone，无 LoRA — #65 ✅ 全 5-benchmark 完成，含 LoCoMo judge=29.15）
| 项 | 值 | 出处 |
|---|---|---|
| split depth j | **9**（浅 readout-safe split，vs 旗舰 j12） | #65 config |
| LoRA | **无**（通过 **省略 `--lora_adapter`** 实现；不存在 `--zero_training_no_adapter` flag） | 上下文校正（standing） |
| backbone | frozen | — |
| 其余 | chunk512、selector=iter_bm25、topk=12、hop=4、sink=bos、chat=False；一份固定 config 跑全 5 benchmark，j 不逐 benchmark 调 | `_qcmem_adapterfree_j9_chatFALSE_taskpool.sh`（commit a21a752 未 push） |
| 状态 | ✅ **完成**（SUMMARY 2026-07-24 23:35，148 jobs 全完；LoCoMo GPT-4o judge=29.15 于 2026-07-25 补齐） | #65 |
| 结果 | **RULER 59.4 / LongEval 3.2 / LongBench 10.63 / BABILong 42.4·19.6·55.6 / LoCoMo judge=29.15（local acc 16.41）** | `logs/qcmem_adapterfree_j9_chatFALSE/SUMMARY.txt` + `locomo_results/qcmem_8b_zeroshot_j9_chatFALSE/scores.json` |

---

## 六、[Generation]
| 项 | 值 | 出处 |
|---|---|---|
| 解码 | **greedy**（do_sample=False, num_beams=1, 无 temp/top_p） | eval 脚本全默认 |
| max_new_tokens | RULER niah=48 / vt=60；LongEval=16；LongBench 按 ds(32/64/128)；BABILong=20；LoCoMo=48 | 见下各 benchmark 表 |
| enable_thinking | False | STATS_APPENDIX:6-8 |

---

## 七、[Efficiency]（chunk512，权威 = bench_qcmem_vs_fullctx.json）
| 项 | 值 | 出处 |
|---|---|---|
| chunk_size / j / topk | 512 / 12 / 12 | `bench...json:5,7,8` |
| GPU | NVIDIA L20A 183GB，单卡 cuda:0 | `logs/bench_chunk512_full.log:1-7` |
| software | torch 2.10.0+cu128, transformers 5.5.4, bf16, sdpa | CLAUDE.md wzc1 .venv + json:9-10 |
| warmup / repeat / decode | warmup=1（不计时）/ n_repeat=3（median）/ n_decode=20 | `bench...json:11-13` |
| speedup 定义 | full_ctx_prefill_s / qcmem_prefill_s（prefill ratio） | `bench_qcmem_vs_fullctx.py:459` |
| 计时边界 prefill | full: L-token 前向到 logits；QCMem: write ALL N chunks(O(L)) + bm25 CPU select + query write + read→logits（端到端） | `bench_qcmem_vs_fullctx.py:21-38` |
| 计时边界 decode | 20 步 greedy；QCMem 每步 re-encode growing query + read（无 KV cache，faithful eval path） | `bench_qcmem_vs_fullctx.py:35-36` |
| peak_gb | `torch.cuda.max_memory_allocated()` prefill+decode 最大 | `bench_qcmem_vs_fullctx.py:38-39` |
| **128k 结果** | 加速 **7.83×**；CoMem 峰值 **18.26 GB**；full-ctx **89.36 GB** | `bench...json:104-120`（P0-4） |
| caption "OOMs" | ✅ 无冲突：89GB 峰值在 **H20(97.8GB) 部署会 OOM**（H20 已实测）；L20A(183GB) bench 显存足够故 status=ok 并测出该峰值 | P0-4 |

---

## 八、[P1] 各 benchmark 评测配置（复现评测层）
### RULER
tasks=niah_single_2/niah_multikey_1/variable_tracking（3）；lengths=8k/16k/32k/64k/128k（5）；n=100/cell；shards=8；seed=42（per-cell=42+hash%1e5，`eval_ruler_qcmem.py:303-304`）；max_new=48（vt=60）；scorer=`_string_match_all_one`（大小写无关 substring recall；cell=mean recall×100，`eval_ruler_mem_space.py:600-605`）；聚合=**15 cell 等权 macro mean**=1455.8/15=97.05；bootstrap=1000 resample seed=2024；haystack: niah=PG19 prose / vt=RULER noise。

### LongEval
lines-retrieval；lengths=4k/8k/16k/32k/64k/128k（6）；n=100/len（total 600）；shards=8；seed=1234（per-len=1234+CRC32%1e5，`eval_qcmem_longeval.py:313`）；max_new=16；scorer=`extract_prediction`（首个≥4位数字串）==expected；聚合=6 len 等权 macro=72.83（含 4k=92；主表取 8k–128k 5 档对齐 baseline=69.0）；bootstrap overall n=600 95%CI[69.33,76.33]。

### LongBench
6 ds=hotpotqa/narrativeqa/qasper/multifieldqa_en/2wikimqa/musique；n=200（multifieldqa_en=150），total 1150；max_samples=-1；shards=8；seed=42；max_new 按 ds（hotpot/2wiki/musique=32, narrative/qasper=128, multifield=64，`eval_longbench_mem_space.py:103-110`）；scorer=`compute_f1_multi`（SQuAD token-F1 多参考取 max）+`compute_em_multi`；聚合=**macro mean over 6 ds 等权=12.15**（micro n-weighted=11.57）；prompt=DATASET2PROMPT；数据=`data/longbench_raw/data`；chat=False；bootstrap NOT FOUND（STATS_APPENDIX §3.6 无 LongBench 单独 CI）。

### BABILong
dataset=RMT-team/babilong；tasks=qa1/qa2/qa5；lengths=0k/1k/2k/4k/8k/16k/32k（7）；n=100/cell；shards=4；seed=42（greedy seed-invariant）；max_new=20；prompt=babilong DEFAULT_PROMPTS get_formatted_input（use_instruction/examples/post_prompt=True, system_prompt=""）；chat=False；scorer=`babilong.metrics.compare_answers`+`TASK_LABELS[task]`；聚合=21 cell 各自 acc（grid，无额外加权）；bootstrap 5 cells 95%CI（1000 resample seed=2024）。

### LoCoMo
dataset=locomo10.json（10 conv, 1986 QA）；n=**1986**（max_samples=-1, all cat）；shards=4；max_new=48；local scorer=F1/EM/acc（cat5=abstain-acc）；**headline=GPT-4o judge over all n=1986**（cat1-4 n=1540 送 API，cat5 本地 abstain）；judge endpoint=`maas-openapi.wanjiedata.com/api/v1`, model=gpt-4o（不 pin snapshot）；judge prompt=STATS_APPENDIX §1(d)（输出 CORRECT/WRONG）；旗舰 judge=**38.27** vs KVD 34.59（paired diff +4.81, 95%CI[2.34,7.27], p<0.0001）；bootstrap=1000 resample seed=1234（unpaired）+paired 10000 seed=1234；judge cache=judge_cache.jsonl（1540 行，4 retries exp backoff，API 失败→0.0）；chat=False。

---

## 九、[P1] 消融矩阵（指针）
- **split depth j sweep**：bottleneck 越深 LM 税越大（单调）；j=12 是"可缓存语义上限 vs LM 税"折中（非税最小点）。见 memory `bottleneck-layer-sweep-monotone`。
- **selector 消融**（tab_selector）、**iter vs single VT**（tab_itervt）、**cross-chunk**（tab_crosschunk）、**chunk 512 vs 1024**（tab_chunk）——这些表用 chunk=1024，是消融专用，非旗舰。
- **LoRA 消融**：adapter-free（j9 frozen，#65）vs +distilled LoRA（j12）= 主表两行制核心对照。

---

## 十、待补 / open items
1. ✅ **#65 adapter-free 5-benchmark eval + LoCoMo judge 全部完成回填**（SUMMARY 2026-07-24 23:35 + judge 2026-07-25）→ [Adapter-free] 结果 + 主表已填（RULER 59.4 / LongEval 3.2 / LongBench 10.63 / BABILong 42.4·19.6·55.6 / **LoCoMo GPT-4o judge=29.15**，local acc 16.41）。**本项 close。** 对照：旗舰+LoRA 38.27 > KVD 34.59 > adapter-free 29.15 → distilled LoRA 在 LoCoMo judge 贡献 +9.12。
2. **#68 MemoryLLM chat=False overlay**（LongEval/LongBench6ds/BABILong）→ gated on 空闲 diskB 节点。
3. **tab_eff.tex caption "OOMs" 措辞修正**（P0-4，须加硬件限定，否则与 bench status=ok 矛盾）。
4. ✅ **iter_hop_topk 已统一 = 4 且旗舰 BABILong hop=4 重跑完成回填（2026-07-24，#70 CLOSED）**：BABILong 脚本默认由 2→4 修正（commit `5158e70`，`eval_qcmem_babilong.py:905`+`:347`）；其余 4 benchmark 本就=4。旗舰 BABILong hop=4(3 轮) 重跑（.73 GPU0-3，dir=`babilong_results/qcmem_j12_iter_bm25_chatFALSE_ad_hop4`）已完成并 rescore → 新数字 **qa1 55.6 / qa2 27.0 / qa5 68.7**（旧 hop=2 的 53.6/25.6/66.7，三档均略升），已回填 §D/headline/PAPERA_ALL_RESULTS。**本项 close。**
5. 训练**峰值显存** NOT FOUND（log 无记录）——若论文要报，需重跑一次带 nvidia-smi 采样或 `torch.cuda.max_memory_allocated`。
6. 蒸馏 **temperature** 无显式字段（T=1 等效）——论文若写 KD 温度，标 T=1。
7. LongBench **bootstrap CI** 缺（STATS_APPENDIX §3.6 未给）——若论文要报 LongBench CI 需补算。
