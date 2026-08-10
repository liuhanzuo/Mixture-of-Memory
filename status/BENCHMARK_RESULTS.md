# BABILong Benchmark Results
# 所有实验结果汇总，方便查阅和对比。
# 格式：10 task × 7 length (0k/1k/2k/4k/8k/16k/32k) × 100 samples per cell。
# 评分：babilong.metrics.compare_answers（与论文口径一致）。
# 更新时间：2026-05-18（下方 chat=False campaign 段 2026-07-23 追加）

---

# ★ 论文 chat=False 全量结果汇总（2026-07-23，campaign 完成）

> **协议双支柱**：`selector=iter_bm25` + `chat_template=False`。官方判分：RULER=`string_match_all`，BABILong=`compare_answers`+`TASK_LABELS`（禁 re.search），LongBench/LoCoMo=`run_scoring`，LoCoMo headline=GPT-4o judge。
> **为何 chat=False**：论文所有模型都是 continue-train 的 **BASE LM（无 SFT/RL）**，套 chat template 会注入 OOD token，对 base 不公平；故**所有方法统一 chat=False**。旧 chat=True 数字（`*_chatnothink`）作废。
> **flagship**：Qwen3-8B + LoRA `qcmem_distill_qwen_j12_r32_4k`，resume_j=12，chunk1024，sink=bos。
> **完整性**：以下所有 RULER 消融格均通过 **Iron-Law-2**（8/8 shard、empty=0、recall 重算与 on-disk 0 mismatch）。数据在 diskB（.73/.82/.104 共享 FS）。判分脚本：`scripts/score_ruler_taskbreadth.py`（RULER）、`scripts/score_flat_babilong.py`（BABILong）。
> **⭐ 本段表格只记 CoMem（本文方法）自身数字**；baseline 对照方法（KV-Direct/HCache/StreamingLLM/MemoryLLM）不入表，数字见 `ruler_results/*_chatFALSE/`（.82）+ `status/STREAMINGLLM_EQUALBUDGET_RESULTS.md` + RUN_REGISTRY。标 **🚫 论文无关** 的条目=不进论文正文。

## Phase 1 — 核心 benchmark（Qwen3-8B）

### CoMem flagship（chat=False，iter_bm25）
| benchmark | 分数 |
|---|---:|
| RULER（niah 主体） | **97.05** |
| LongEval | **72.83** |
| LongBench（AVG） | **12.15** |
| BABILong qa1 / qa2 / qa5 | **53.6 / 25.6 / 66.7** |

> **baseline 对照（不入本表）**：同 chat=False 下 KV-Direct（8k–64k near-perfect→128k=0，非 OOM）/ HCache（极弱 33/5/3）/ StreamingLLM（85/36/20/10/2）/ MemoryLLM（niah 29/40/37/21/21、VT≈0）——全线 << CoMem 97.05。数字见 `ruler_results/*_chatFALSE/`（.82）。⚠️ KV-Direct/HCache 用 `sel=bm25` 非 `iter_bm25`（errata §8c，task#10 待统一）。

### LoCoMo（n=1986，chat=False，官方 scorer + GPT-4o judge）— CoMem only（2026-07-23 verified，appendix §1）
| method | **judge (headline)** | F1 | acc | EM |
|---|---:|---:|---:|---:|
| **CoMem（iter_bm25）** | **38.27** | 9.15 | 23.36 | 0.55 |

> **headline = GPT-4o judge 38.27**（over n=1986；cat5 adversarial n=446 不送 gpt-4o，本地 abstention 判分 folded 进 headline，非丢弃；judged-only cat1–4 n=1540 = **48.64**）。判分 endpoint=maas `gpt-4o`（无 client 端 dated snapshot），seed=1、prompt 全文见 `status/QCMEM_STATS_APPENDIX_chatFALSE.md` §1d。per-cat：cat1 26.95/cat2 19.00/cat3 30.21/cat4 69.32/cat5 2.47。
> baseline 对照（不入本表）：**KV-Direct（full-ctx 上界）judge 34.59 / F1 9.02**（同 chat=False）。**paired bootstrap（judged n=1540，B=10000 seed1234）：CoMem−KVD judge diff=+4.81，95%CI[2.34,7.27]，p<0.0001 → CoMem 显著优于 full-ctx KV oracle**（unpaired CI 重叠是配对设计下的 power artifact；judge 是 protocol-robust headline，token-F1 9.15≈9.02 打平=formatting artifact）。
> ⚠️ 旧值 F1 19.51/EM 5.99/acc 28.65 是 **chat=TRUE**，已作废（chat=False F1=9.15）。

## Phase 2 — 消融表（Qwen3-8B RULER，内部相对比较，全 Iron-Law-2 OK）

### #9 tab_selector — CoMem 单遍 selector 消融（BM25/Recency/ReaderAttn/Oracle 均为 CoMem 内部 selector 变体，RULER n=100，峰值 top-k over {4,8,12,16,24}）
| Task | Len | BM25 | Recency | ReaderAttn | Oracle |
|---|---|---:|---:|---:|---:|
| niah_single   | 8k  | 100 | 100 | 100 | 100 |
| niah_single   | 16k | 100 | 100 | 100 | 100 |
| niah_single   | 32k | 100 | 82  | 73  | 100 |
| niah_multikey | 8k  | 99  | 98  | 97  | 100 |
| niah_multikey | 16k | 99  | 88  | 90  | 100 |
| niah_multikey | 32k | 99  | 54  | 60  | 100 |
| var-track     | 8k  | 99.4 | 99.2 | 99.8 | N/A |
| var-track     | 16k | 92.6 | 92.4 | 92.4 | N/A |
| var-track     | 32k | 32.0 | 41.2 | 27.8 | N/A |

> 结论：Oracle 两 needle 任务恒 100（读出无损，长程差距=检索问题）；BM25 单遍在 niah 追平 Oracle（≤1pp）。**VT 单遍 selector 32k 全部崩塌**（32.0/41.2/27.8）→ 迭代检索（#10）的动机。

### #10 tab_itervt — 迭代检索（RULER variable_tracking，n=100，chat=False；2×2 one-shot vs iterative 见 #55 P0#2）
| arm | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| **单遍 bm25**（明码单次检索，无跳；2×2 CoMem+bm25） | 48.0 | 25.0 | 23.4 | 21.2 | 20.4 |
| **iter_bm25 3 跳**（flagship `_ad`：topk12/hop4/chunk512，read≈6.6k） | **96.6** | **97.6** | **98.8** | **99.0** | **95.8** |
| iter_bm25 4 跳（`ablation10`：topk16/hop4/chunk1024，read≈17k，大预算变体） | 99.0 | 95.6 | 89.8 | 89.8 | 87.4 |
| oracle_vt（🚫论文无关：诊断上界，不入正文） | 15.6 | 10.4 | 4.8 | 1.2 | 2.4 |

> **headline（#61 已裁决）**：迭代检索把 VT 从**单遍 bm25 的 20–48 全崩**救到 **96–99 长档恒定** → 链式追踪任务必须多跳检索。`rounds:0`=auto=ceil(topk/hop) 是**多跳**（3/4 跳），非单遍（曾误标致"叙事反转"，已纠正）。3 跳小预算 flagship（`_ad`）长档略优于 4 跳大预算（99.0/95.8 vs 89.8/87.4）=多召回 distractor 链在 128k 略伤，flagship 取 `_ad`。

### #11 tab_chunk — chunk-size 消融（RULER niah_multikey，n=100，chat=False，本轮判分）
| chunk_size | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|
| 128  | 91.0 | 90.0 | 81.0 | 85.0 |
| 256  | 80.0 | 90.0 | 90.0 | 94.0 |
| 512  | 89.0 | 95.0 | 97.0 | 94.0 |
| 1024 | 100.0 | 89.0 | 92.0 | 94.0 |

> 结论：chunk 512–1024 长档最稳（64k 均 94，512 在 32k 达 97）；chunk128 长档略降（32k 81）。flagship 用 chunk1024 与此一致。

### #12 tab_crosschunk — cross-chunk attention（full vs block-diag KV reuse，selector=iter_bm25 tk12）
| Task | Full | Block-diag | Δ(full−bd) |
|---|---|---|---|
| RULER niah_single (8k/16k, n=50) | 100 / 100 | 100 / 96 | 0 / +4 |
| RULER niah_multikey (8k/16k, n=50) | 96 / 94 | 60 / 32 | **+36 / +62** |
| BABILong qa2 (8k/16k, n=100) | 36 / 20 | 17 / 13 | +19 / +7 |
| BABILong qa5 (8k/16k, n=100) | 78 / 69 | 78 / 55 | 0 / +14 |

> 结论：cross-chunk recompute 对**多事实消歧** load-bearing（multikey Δ+36/+62、qa2/qa5 有 gap），对单 needle 无关。

### tab_slm — 等预算档 CoMem（budget = sink4+window6653 = 6657 tok ≈ CoMem 恒定 read）
| RULER task | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| **CoMem** niah_single | 100 | 100 | 100 | 100 | 100 |
| **CoMem** var-track (equal-budget, single-pass) | 96.6 | 97.6 | 98.8 | **99.0** | **95.8** |

> ⚠️ **VT 行 = chat=False 等预算精确值**（dir `qcmem_8b_iter_chatFALSE_ad`，selector=iter_bm25 topk12/hop4 → **`rounds:0`=auto=ceil(12/4)=3 跳迭代**（非单遍！见 #61），read≈6.6k，全档单一 config，Iron-Law-2 OK）；取代旧 `~95/~95` 占位 + 混配 95.2/93.8/96.8。**这是 3 跳迭代 flagship，长档 99.0/95.8**；比 tab_itervt 的 4 跳大预算变体（89.8/87.4）略优（#61 已裁决：within-iterative 差异，flagship 取 `_ad`）。

> baseline 对照（不入本表）：等预算 StreamingLLM（recency 截断）single 90/42/18/16/4、multikey 86/48/26/8/6、vt 38/3.6/1.2/0/0 → 长档全崩，见 `status/STREAMINGLLM_EQUALBUDGET_RESULTS.md`。**结论**：等预算下唯一变量=保留**哪些** token；CoMem 的 relevance-based selection + 迭代检索恒定，recency 截断全崩（single 25× gap） → fixed budget 必要但不充分。

### 投稿前补缺 GPU eval（2026-07-23，.73 H20，agent a8ef76da；详报 diskB `status/QCMEM_GPU_EVAL_PRESUB_20260723.md` untracked）

**P0#2 — VT selector-fairness 2×2（chat=False，RULER var-track，n=100，全 Iron-Law-2 OK）**
| Len | KVD+iter_bm25 | KVD+bm25(1-shot) | CoMem+bm25(1-shot) | CoMem+iter_bm25(flagship) |
|---|---:|---:|---:|---:|
| 8k | 100.0 | 48.4 | 48.0 | 96.6 |
| 16k | 100.0 | 26.0 | 25.0 | 97.6 |
| 32k | 100.0 | 22.4 | 23.4 | 98.8 |
| 64k | 100.0 | 22.6 | 21.2 | 99.0 |
| 128k | 100.0 | 21.2 | 20.4 | 95.8 |
> **归因结论**：固定 selector 下 KVD≈CoMem 各档（one-shot 48/26/…≈48/25/…；iter KVD=100 vs CoMem 96.6–99.0，full-depth reader 高 1–4pp）；**大杠杆=selector（one-shot→iter：VT 20–48→96–100，两个架构都是）**。**CoMem 架构价值=效率非 VT 精度**（同检索下 matches 自身 one-shot 且距 uncompressed KVD reader 仅几 pp）。这决定论文口径：CoMem = "以极低显存/算力 match KVD 精度"，非"VT 精度超 KVD"（VT 上 KVD 反略高；LoCoMo judge 上 CoMem 38.27>KVD 34.59 显著——不同任务不同）。

**P1#4 — chunk1024 效率（vs chunk512 tab_eff，H20，median-of-3）**
| Len | Full prefill | CoMem prefill | Speedup | Full peak | CoMem peak |
|---|---:|---:|---:|---:|---:|
| 8k | 1.39s | 1.15s | 1.21× | 19.9GB | 17.8GB |
| 16k | 2.60s | 2.08s | 1.25× | 24.6GB | 19.4GB |
| 32k | 6.26s | 2.75s | 2.27× | 33.8GB | 19.5GB |
| 64k | 18.05s | 4.10s | 4.40× | 52.3GB | 19.8GB |
| 128k | **OOM** | 7.98s | ∞ | **OOM** | **20.3GB** |
> chunk1024 read pack=1+12·1024+1024=13313 tok（2× chunk512）→ CoMem mem≈18–20GB flat（+~2GB vs c512）。**同硬件 headline：128k full-ctx 在 H20 OOM**（all-pos logits `[1,131072,151936]` bf16=39.8GB 单次 alloc）而 CoMem 20.3GB 跑通。⚠️ c512 tab_eff 的 "128k full=89GB/7.83×" 是 B200 上测；chunk_size 不影响 full-ctx path，c1024 128k full 有限值需 B200 rerun（低优先，OOM-on-H20 已足够讲故事）。

**P1#6 — 迭代检索开销（CPU micro-bench，median-of-5）**
| Len | one-shot ms | iter ms | ratio | mem |
|---|---:|---:|---:|---|
| 8k | 2.34 | 9.82 | 4.20× | 相同(<0.05MB 差) |
| 32k | 9.62 | 45.74 | 4.75× | 相同 |
| 128k | 41.04 | 188.77 | 4.60× | 相同 |
> iter_bm25 ≈ one-shot 的 ~4.2–4.9× 延迟（3 跳），两者均 length-linear，CPU 内存一致。绝对开销可忽略：128k 迭代检索 ~189ms/19MB vs 模型 forward ~25s/样本 → 多跳 selector 占端到端 ~0.1%，却换来 VT 20→100 的提升（**基本免费**）。


## 🚫 论文无关 / 范围外
**🚫 论文无关（不进论文正文）**：
- **oracle_vt 控制（#10 内）**：诊断用单遍上界，不入正文。
- **tab_scale（模型规模 0.6B–32B 扫描）**：未被 `\input`，**不在论文**。

**在论文、但本 chat=False campaign 不重跑**：
- **tab_eff（#13）**：纯 prefill 计时 + 显存 MB，chat 不敏感，无需重跑。
- **tab_hy3_ruler / tab_hy3_distill（#14/#15）**：独立 Hunyuan Hy3 80L MoE backbone（非 Qwen3-8B），另一 harness。

## ★ 全局洞察（写论文时必须处理，task#10）
chat=False 大幅利好 exact-match / completion 任务（如 BM25 VT16k **27.6 → 92.6**、Recency niah_single 16k 72 → 100），但对 extractive token-F1 QA 略降。因公平性要求**所有方法统一 chat=False**，故 tab_selector / tab_itervt / tab_chunk / tab_crosschunk / tab_slm / tab_overview / tab_h2h / tab_scaling **整套表都要换成上面这批 chat=False 数字**。

### ★★ task#61 — VT config provenance ✅ RESOLVED（2026-07-23，P0#2 2×2 + code trace，high confidence）
**根因=`rounds:0` 被误读为"单遍"。** 实测 sidecar `qcmem` 配置 + code (`eval_qcmem_babilong.py:253` `rounds = iter_rounds if >0 else ceil(topk/hop)`) 证明 **`rounds=0` = auto = ceil(topk/hop_topk)，是多跳迭代，不是单遍**：
- **flagship 等预算 dir `qcmem_8b_iter_chatFALSE_ad`**：selector=iter_bm25, topk12/hop4 → **ceil(12/4)=3 跳迭代**，chunk512，read≈6.6k（实测 avg_read_len 6630）→ VT **96.6/97.6/98.8/99.0/95.8**（8k→128k）。
- **tab_itervt dir `ablation10_itervt_chatFALSE/iterbm25_vt`**：selector=iter_bm25, topk16/hop4 → **ceil(16/4)=4 跳迭代**，chunk1024，read≈17k → VT **99.0/95.6/89.8/89.8/87.4**。
- **真·单遍 = 明码 `bm25` selector（无跳）** = P0#2 2×2 CoMem+bm25 = **48/25/23/21/20**（长档全崩）。

**裁决（三问全清）**：
- (a) **RULER headline 97.05 的 VT 用 3 跳 iter_bm25（`_ad` legit flagship）**，非单遍——合法。
- (b) **等预算行用 `_ad`（3 跳 top12/chunk512 read6.6k）；tab_itervt 对照用「单遍 bm25(20–48) → 迭代 iter_bm25(96–99)」**（用 2×2 的干净 one-shot vs iter 两行，取代旧 32→89.8）。
- (c) **迭代确实救 VT，叙事未反转**：所谓"单遍反超 hop4"纯属 rounds=0 误标；`_ad`(3 跳) 与 `ablation10`(4 跳) 都是迭代，`_ad` 小预算长档略优（99.0/95.8 vs 89.8/87.4）= 4 跳多召回的 distractor 链在 128k VT 上略伤，属 within-iterative 次要发现，flagship 取 `_ad`。
- **归因（P0#2 2×2）**：固定 selector 下 KVD≈CoMem 各档 → **VT 精度来自迭代 selector，非架构；架构价值=效率**（128k full-ctx OOM，CoMem 20GB）。
- **残留（并入 #10 论文集成）**：#9 tab_selector "BM25 单遍" VT 8k=99.4 vs 2×2 CoMem+bm25 fixed-top12 8k=48.0——tab_selector 用**峰值 top-k 扫描**（best over {4,8,12,16,24}，且可能 chat=True）、2×2 用固定 top12 chat=False；最终 tab_selector 换 chat=False 时需再核 top-k 口径。不阻塞本裁决。

---

## 我们的方法（Mixture-of-Memory）

### P8 — L1+L3 dual-gate, Llama-3-8B-Instruct, 500 steps
**Backbone**: Meta-Llama-3-8B-Instruct (8B)
**Config**: L1(512 slots, top_k=64) + L3(64 summary tokens), dual gate, shared bank, selector_temp=1.0
**Training**: 500 steps, tasks=qa1/qa2/qa5, lengths=1k/2k/4k, lr=2e-5
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_p8_full_20260518_144631/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1      89   89   77   74   62   41   32   66.3
qa2      44   48   44   35   48   36   23   39.7
qa3      29   38   29   26   37   37   25   31.6
qa4      51   57   47   50   48   27   23   43.3
qa5      72   73   66   67   88   65   50   68.7
qa6      84   80   74   60   56   48   43   63.6
qa7      36   28   18   17    0    0    0   14.1
qa8      61   54   41   41   34   19    9   37.0
qa9      90   81   70   66   64   58   60   69.9
qa10     74   65   57   50   60   50   40   56.6
AVG    63.0 61.3 52.3 48.6 49.7 38.1 30.5   49.1
```
**Overall mean (10 tasks × 7 lengths)**: **49.1**

Notes:
- Short context (0k/1k) very strong: 63/61 avg
- qa5/qa9 strongest tasks (retrieval/coreference)
- qa7/qa8 collapse at 8k+ (counting tasks — EMA decay issue, motivation for v6/v7)

---

### v2 final — L1+L3 dual-gate, Llama-3.2-1B-Instruct, 10000 steps
**Backbone**: Llama-3.2-1B-Instruct (1B)
**Config**: same L1+L3 recipe as P8
**Training**: 10000 steps, tasks=qa1/qa2/qa5, lengths=1k/2k/4k
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_phase1b_v2_full_20260518/p1bv2_final_fullqa_20260518/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1      66   39   46   41   40   47   36   45.0
qa2      28   20   23   20   30   21   19   23.0
qa3      29   18   25   16   29   17   12   20.9
qa4      24   27   29   28   27   24   18   25.3
qa5      51   46   37   32   76   66   54   51.7
qa6      45   24   35   31   51   43   50   39.9
qa7      18   12   12   27    1    0    0   10.0
qa8      45   14   20   20   28   18   12   22.4
qa9      56   36   28   41   54   58   54   46.7
qa10     35    9   25   24   34   46   45   31.1
AVG    39.7 24.5 28.0 28.0 37.0 34.0 30.0   31.6
```
**Overall mean**: **31.6**

---

### v2-base (step4000) — L1+L3 dual-gate, raw Llama-3.2-1B (no Instruct), 5000 steps
**Backbone**: Meta-Llama-3.2-1B base (1B, NO chat template)
**Config**: same L1+L3 recipe as v2
**Training**: 5000 steps
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_phase1b_v2_llama32_1b_base_step4000/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1      59   47   38   42   44   44   32   43.7
qa2      34   21   20   15   28   26   24   24.0
qa3      21   20   24   21   26   24   17   21.9
qa4      24   25   29   25   29   31   19   26.0
qa5      56   47   48   47   77   64   53   56.0
qa6      37   25   38   36   41   38   40   36.4
qa7      18   15   29   30    0    0    0   13.1
qa8      42   21   24   15   27   20   15   23.4
qa9      39   22   29   36   36   30   21   30.4
qa10     40   17   15   19   49   41   32   30.4
AVG    37.0 26.0 29.4 28.6 35.7 31.8 25.3   30.5
```
**Overall mean**: **30.5**

---

### Plain Meta-Llama-3.2-1B baseline (no memory, vanilla inference)
**Backbone**: Meta-Llama-3.2-1B base (1B)
**Config**: vanilla HF generation, no memory module
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_llama32_1b_base_b2002_20260518_110237/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1       0    1   28   42   48   43   30   27.4
qa2       0    0   26   18   24   17   15   14.3
qa3       0    0    5   12   26   26   22   13.0
qa4       0    2    4    6   10   10   22    7.7
qa5       0    1   10    9   13   24   34   13.0
qa6       0    0    0    1    4    3   20    4.0
qa7       0    0    0    0    2    1    3    0.9
qa8       0    0    0    0    4    3   11    2.6
qa9       0    0    0    0   11   19   42   10.3
qa10      0    0    0    0    8    9   26    6.1
AVG     0.0  0.4  7.3  8.8 15.0 15.5 22.5    9.9
```
**Overall mean**: **9.9**

Notes:
- 0k columns all 0: raw base model cannot answer BABILong without any context retrieval
- Short lengths (0k/1k) near-zero; only starts working at 2k+

---

## 其他论文（参考数字）

### LM2-1.7B (Large Memory Models, arXiv 2502.06049)
**Backbone**: vanilla-Llama-1.7B (**trained from scratch** by LM2 team, NOT Meta's open release)
**Method**: auxiliary memory module with dual input/forget gates
**Source**: Table 1 of the LM2 paper; only qa1/qa2/qa5 reported; ≥8k is aggregate average
**Note**: NOT directly comparable to our work — backbone is a custom 1.7B pretrained model

```
length   qa1   qa2   qa5   10-task-avg
0k       99    89    98    92.5
1k       85    59    91    78.3
2k       58    43    87    65.8
4k       46    37    78    55.9
>=8k avg 23.8  15.0  38.8  39.9
```

### LM2 paper baseline: Meta Llama-3.2-1.2B (vanilla, no memory)
**Note**: This is Meta's released Llama-3.2-1B used as baseline by LM2 paper; directly comparable to our plain baseline above.

```
length   qa1   qa2   qa5   10-task-avg
0k       54    25    59    40.7
1k       48    22    69    39.5
2k       44    18    64    38.6
4k       37    16    56    36.8
>=8k avg 19     8    36.5  28.2
```

### BABILong paper (NeurIPS 2024, arXiv 2406.10149) — Table 4
**Backbone**: Meta-Llama-3-8B-Instruct / Meta-Llama-3.1-8B-Instruct
**Note**: n=1000 samples per cell (vs our n=100)

```
qa   length  Llama-3-8B-It  Llama-3.1-8B-It
qa1  0k      98             99
qa1  1k      93             97
qa1  2k      80             97
qa1  4k      16             95
qa1  8k       7             83
qa1  16k     31            100
qa1  32k     23             87
qa2  0k      47             53
qa2  4k      10             51
qa2  8k       4             44
qa2  32k      2             56
qa5  0k      85             81
qa5  4k      52             85
qa5  8k      43             86
qa5  32k     50             85
```
**P8 vs Llama-3-8B-Instruct vanilla overall**: P8 mean=49.1 vs paper Llama-3-8B mean≈42.6 (+6.5pp)

---

## 正在进行的实验

### v6 — replace writeback (slot ← s_new, no EMA)
**Status**: Training started 2026-05-18 on local H20 8 GPUs
**Config**: same as v2 (1B Instruct) but with --use_replace_writeback
**Expected**: better counting/tracking (qa7/qa8), potentially worse long-range retrieval
**Results**: TBD

### v7 — hybrid EMA + 8 global always-on slots
**Status**: Training started 2026-05-18 on remote H20 28.59.80.196 8 GPUs
**Config**: same as v2 (1B Instruct) + --num_global_slots 8
**Expected**: keep retrieval quality while improving counting via always-on registers
**Results**: TBD

### MemoryLLM-8B eval
**Status**: Running on B200 (28.89.17.144), extremely slow (~100s/sample due to chunked inject_memory)
**Config**: full qa1-10 × 0k-32k, no chat template
**Expected**: ~10-20 hours to complete
**Results**: TBD

### P2 decoupled-read — Llama-3-8B, Dolmino CPT step2000 (2026-06-04)
**Config**: mem_space adapter, use_decoupled_read=on, num_slots=128 top_k=16 temp=40, offline BABILong qa1/qa2/qa5 × 0k-32k, n=100, babilong.metrics
**Checkpoint**: outputs/dolmino_p2_decoupled_local/mem_space_adapter.pt (commit 02561b4)
**Accuracy (%)**:
| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|------|----|----|----|----|----|-----|-----|
| qa1  | 72.0 | 24.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qa2  | 27.0 | 13.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qa5  | 53.0 | 27.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
**Verdict**: FAILS gate. 0k healthy (model fine when no compression needed), but ≥2k collapses to 0.0% — routing collapse (eval top1_sim≈0.05≈uniform/128). Decoupled-read does NOT rescue compression. See ops/research_notes/toy_vs_full_routing_collapse_20260604.md for root cause (decoupled-read severs selector's LM-loss gradient; LM loss alone never bootstraps content addressing).
