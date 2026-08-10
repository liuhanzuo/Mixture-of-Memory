# 论文 chat=False 全量清单（2026-07-22，用户指令"把论文要的所有 chat=false 都测了"）

**范围钉死**：论文 = **Qwen3-8B 单一 backbone**。`tab_scale.tex`（0.6B–32B 模型规模扫描）**未被 `\input`，不在论文** → **无模型规模扫描，chat=False 不含 sweep**。
`tab_scaling` 是**长度扫描**（8k→256k）on 8B，不是模型规模。

协议双支柱：**selector=iter_bm25 + chat_template=False**（no-think 天然含在 chat=False 里）。官方判分：BABILong=compare_answers，RULER=string_match，LongBench/LoCoMo=run_scoring，LoCoMo headline=GPT-4o judge。

15 张 `\input` 表分类：

## A. chat-sensitive，需 chat=False（Qwen3-8B eval）
| # | 表 | 内容 | 覆盖状态 |
|---|-----|------|---------|
| 1 | tab_overview | 5-benchmark(RULER niah×2/LongEval/BABILong qa1qa5/LongBench AVG/LoCoMo) CoMem+KV-Direct+HCache | aa5c3802(4bench)+LoCoMo |
| 2 | tab_h2h | RULER niah_single/multikey CoMem/KVD/HCache/MemoryLLM n=50 | aa5c3802 RULER |
| 3 | tab_babilong | BABILong qa1/qa5 CoMem/KVD/HCache/MemoryLLM | aa5c3802 BABILong |
| 4 | tab_longbench | LongBench(dir 实测 use_chat_template=true → 必重跑) | aa5c3802 LongBench |
| 5 | tab_longeval | LongEval CoMem/KVD/HCache | aa5c3802 LongEval |
| 6 | tab_locomo | LoCoMo(headline=GPT-4o judge) | 主体已 chat=False；baseline 三方 abaed0fc(.82) |
| 7 | tab_scaling | RULER 长度 8k→256k CoMem vs full-ctx n=50 | aa5c3802 到 128k；**256k CoMem 需补**(full-ctx@256k OOM 无需) |
| 8 | tab_slm | StreamingLLM 等预算 RULER niah_single CoMem vs SLM n=50 | **需确认 aa5c3802 baseline 含 SLM 等预算档** |

## B. chat-sensitive 消融（Qwen3-8B RULER，内部相对比较）— 一致性需 chat=False，Phase 2 低优先
| # | 表 | 内容 |
|---|-----|------|
| 9 | tab_selector | selector 消融(bm25 vs iter_bm25) RULER j=12 chunk1024 32slots |
| 10 | tab_itervt | 迭代检索 RULER var-track n=100 |
| 11 | tab_chunk | chunk-size 消融 |
| 12 | tab_crosschunk | cross-chunk attention 消融 |

## C. 非 chat-sensitive — 不重跑
| # | 表 | 理由 |
|---|-----|------|
| 13 | tab_eff | prefill 加速/显存 MB，纯计时+内存，无生成文本判分 |

## D. 独立 backbone（Hunyuan Hy3 80L MoE，非 Qwen3-8B）— 本 16-H20 Qwen campaign 范围外
| # | 表 | 理由 |
|---|-----|------|
| 14 | tab_hy3_ruler | Hy3 backbone RULER，另一模型+harness |
| 15 | tab_hy3_distill | Hy3 自蒸馏 ppl-ratio/top1，ppl 口径非 chat-dependent |

## 执行分期
- **Phase 1（进行中）**：aa5c3802（16 H20 .73+.104）跑核心 4 benchmark（RULER/LongEval/LongBench/BABILong）CoMem 8B flagship + baseline，chat=False，iter_bm25，官方判分 → 覆盖表 1-5 + 7/8 的 RULER 主体。abaed0fc(.82) 跑 LoCoMo baseline → 表 6。
- **Phase 2（Phase 1 报完 / GPU 空出后）**：补 (a) RULER 256k CoMem(tab_scaling)；(b) tab_slm StreamingLLM 等预算档确认；(c) 消融表 9-12 chat=False 重跑。
- **不动**：tab_eff（13）、tab_hy3_*（14/15）。

## GPU 现状
- 训练暂停：.73/.104 H20（见 PAUSE_RESUME_H20_20260722.md，eval 完 resume）。
- LOCAL 8×L20A + .252 B200 仍在 Paper B 训练。
- eval GPU 全占（.73+.104 跑 aa5c3802，.82 跑 abaed0fc）→ Phase 2 排队。
