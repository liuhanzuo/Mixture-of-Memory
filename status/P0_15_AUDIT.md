# P0.15 — 提交前可审计性、读长口径与匿名化收口（Paper A）

**Date:** 2026-08-02 · **Type:** existing-artifact recompute + Paper-only audit · **NO GPU / NO model forward.**
所有分项数值均由现有 raw predictions / scorer 输出（CPU-only 打分）核实，未跑任何模型。diskB 只读访问节点 `.73`
(`28.85.35.73:36000`, `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`)。本文件只报告与建议，
**不修改任何 `.tex` / 已有 `status/*.md` / `paperA/TODOList.md`**。

---

## Part A — j=0 cell-level 分解

**配置（config #2，全部 cell 完全一致）：** Qwen3-8B `models/Qwen3-8b-local`, `--resume_j 0`（full 36-layer
recompute of retrieved pack = 自蒸馏 teacher / full-depth 检索基线）, `--selector iter_bm25 --topk 12
--iter_rounds 0 --iter_hop_topk 4 --chunk_size 512 --sink_tokens bos`, `--dtype bfloat16 --attn_impl sdpa
--seed 42`, **NO LoRA adapter**, **chat_template=FALSE, enable_thinking=FALSE**。这是 task #108 的 j=0 text-RAG
5-benchmark sweep 产物（原始记录 `status/P0_2_CONFIG2_JRAG_RESULTS.md`），本节全部逐 cell 重新打分核对。

聚合口径：RULER = 逐 shard `summary.score` 按 n 加权（string_match recall × 100）；LongBench = 逐 shard `f1`
按 num_samples 加权，6-QA macro；LongEval = `_summary_merged.json` 逐长档 accuracy；BABILong = 官方
`babilong.metrics.compare_answers + TASK_LABELS`（`scripts/score_flat_babilong.py`，8-shard 求和）；LoCoMo =
`scores.json`（F1/acc/EM + gpt-4o judge，按 category 分解）。

### A.1 RULER — Cohort B (`niah_single_3`)，**单列附表，严禁并入 Cohort-A / `niah_single_2`**

> ⚠️ Paper A 的 RULER Cohort A（all-method Table `tab:h2h`）用 `niah_single_2`（CoMem macro 97.05）；Cohort B
> 用 `niah_single_3`（CoMem+LoRA macro 96.07，见 `tab:ruler-statistics`）。j=0 的 paired RULER 数据只能进
> **Cohort B**，与 P0.13 的 15-cell paired（Arm A `resume_j=0`+LoRA / Arm B `resume_j=12`+LoRA）共表，
> **不得与 `niah_single_2` 行拼接或跨 cohort 相减**。

**（i）本节点核实的 j=0 no-LoRA RULER（config #2，n=100/cell，diskB 8-shard 加权，附实测 read_len）**

| task | 8k | 16k | 32k | 64k | 128k | mean read_len (tok) |
|------|----|----|----|----|----|----|
| niah_single_3   |  99 |  99 |  98 |  96 |  99 | 6195–6639 |
| niah_multikey_1 | 100 | 100 |  99 |  99 |  99 | 6177–6561 |
| variable_tracking | 100 | 100 | 100 | 100 | 100 | 6557–6630 |

- **15-cell macro = 99.20**（单列，config #2 j=0 no-LoRA）。全 15 cell n=100，0 空、0 OOM，逐 cell IL2 OK。
- raw path（diskB `.73`）：NIAH `ruler_results/p0_2_c2_j0_iterbm25_niah_chatFALSE/`（FRESH，8-shard `*.json` 各带 `summary.score/n/avg_read_len`）；
  VT `ruler_results/presub_A_kvdirect_iterbm25_vt/variable_tracking_{L}/_summary_shard*of8.json`（REUSED；dir 名含 `kvdirect` 但记录的 `eval_config` = config #2 j=0，dir 名误导）。
- 实测 read_len 6.18–6.64k（短尾 chunk/query）——见 Part B。

**（ii）P0.13 paired Cohort-B 15-cell（同一 Cohort B，`niah_single_3` 家族，paired j=0 vs j=12，均 +LoRA）——单列 paired 附表，复用 P0.13，不再重算**

| Task | 8k (A/B) | 16k (A/B) | 32k (A/B) | 64k (A/B) | 128k (A/B) |
|------|----|----|----|----|----|
| niah_single_3 | 100.0/100.0 | 97.0/91.0 | 97.0/97.0 | 99.0/98.0 | 98.0/98.0 |
| niah_multikey_1 | 100.0/94.0 | 100.0/91.0 | 100.0/99.0 | 97.0/90.0 | 100.0/93.0 |
| variable_tracking | 99.8/96.2 | 100.0/98.0 | 100.0/98.2 | 100.0/98.6 | 100.0/99.0 |

- Arm A (`resume_j=0`+flagship LoRA) macro **99.19** vs Arm B (`resume_j=12`+同 LoRA) macro **96.07**；paired diff +3.12 pp，95% CI [2.36,3.93]，McNemar p≈8.79e-24。n_paired=1500，packs_paired_1to1=True。
- raw path（diskB `.82` `28.82.250.82:36000`）：`bench_results/p0_13_quality_latency/`（`manifest.json`/`summary.json`/`stats.json`/`quality/` 120 files/`latency/`）。pack-pairing manifest = `manifest.json`（`packs_paired_1to1=True`，iter_bm25 forward-free → packs resume_j-独立；LoRA SHA `dd09cd17…` `lora_sha_match=true`）。原始记录 `status/P0_13_QUALITY_LATENCY.md`。
- **口径区别**：(i) 是 j=0 **no-LoRA** teacher（用于 depth-tradeoff / teacher 上界）；(ii) 是 j=0 **+LoRA** 与 j=12+LoRA 的 read-path paired（P0.13）。二者都在 Cohort B `niah_single_3` 家族内，但 LoRA 挂载不同——报告时须分别标注，不可混为一行。

### A.2 LongBench（6-QA，SQuAD token-F1，macro；chat=False，j=0 no-LoRA）

| dataset | F1 | n |
|---------|----|----|
| 2wikimqa        | 12.42 | 200 |
| hotpotqa        | 12.17 | 200 |
| multifieldqa_en | 26.18 | 150 |
| musique         |  7.47 | 200 |
| narrativeqa     |  3.88 | 200 |
| qasper          | 11.73 | 200 |
| **6-QA macro**  | **12.31** | 1150 |

- raw path（diskB `.73`）：`longbench_results/p0_2_c2_j0_iterbm25_chatFALSE/`（FRESH，48 = 6 ds × 8 shard `*_metrics.json`，各带 `f1`/`num_samples`）。
- 逐 shard f1 按 num_samples 加权重算 → 与 P0.2 表逐位一致。**6-QA macro 为权威**；`P0_2_PARETO_RESULTS.md` §4b 的 “4-QA” 子集在本 harness 未正式定义（`lb.DEFAULT_DATASETS` = 上述 6），如需 4-QA 须先固定子集再从上表 per-ds 派生。

### A.3 LongEval（line-key retrieval accuracy，n=100/length，chat=False，j=0 no-LoRA）

| 4k | 8k | 16k | 32k | 64k | 128k |
|----|----|----|----|----|----|
| 98.0 | 100.0 | 96.0 | 99.0 | 94.0 | 97.0 |

- mean(8k–128k) = **97.2%**（5 长档）· mean(4k–128k) = 97.3%（6 长档）。全长档 n=100、8-shard。
- raw path（diskB `.73`）：`longeval_results/p0_2_c2_j0_iterbm25_chatFALSE/longeval_8b/_summary_merged.json`（FRESH）。
- **实测 avg_read_len**：4k 4237.9 / 8k 6291.0 / 16k 6307.1 / 32k 6333.8 / 64k 6398.0 / 128k 6455.7 tok → L-independent 检索 pack（Part B 证据）。4k 档 read_len < 6.2k 因文档本身短于 pack 上限。

### A.4 BABILong（qa1/qa2/qa5 × 7 lengths，n=100/cell，官方 compare_answers，chat=False，j=0 no-LoRA）

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|------|----|----|----|----|----|----|----|
| qa1 | 98 | 84 | 80 | 73 | 60 | 34 | 35 |
| qa2 | 58 | 53 | 51 | 44 | 37 | 17 | 12 |
| qa5 | 70 | 73 | 61 | 57 | 69 | 53 | 60 |

- raw path（**权威**，diskB `.73`）：`babilong_results/p0_2_c2_j0_iterbm25_chatFALSE/`（FRESH，flat 168 CSV = 3 task × 7 len × 8 shard，逐 cell IL2 OK，n=100/cell）。本节点用 `scripts/score_flat_babilong.py --num_shards 8` 重算 → 与 P0.2 表逐位一致（21-cell mean 56.1）。
- **本地交叉核对**（wzc1 local）：`babilong_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/`（nested，4-shard，n=100/cell，`eval_config` 确认为同 config #2）用 `scripts/score_nested_babilong.py` 重算 = qa1 98/84/79/73/60/32/35、qa2 58/55/53/44/36/16/12、qa5 69/72/62/55/71/51/61 —— 与权威 8-shard run 逐 cell 差 0–2 pp（独立采样噪声，非 config 差异）。**报告以 diskB 8-shard p0_2_c2 为准**（与 P0.2 headline 一致）；本地 4-shard 仅作 provenance 交叉验证。

### A.5 LoCoMo（full，n=1986，chat=False，j=0 no-LoRA，gpt-4o judge）

| F1 | acc | EM | GPT-4o judge |
|----|-----|----|--------------|
| 9.90 | 25.23% | 0.81% | 41.59% |

per-category judge（`scores.json` by_category）：

| cat | n | F1 | acc% | judge% |
|----|----|----|----|----|
| 1 | 282 | 10.46 | 13.12 | 31.21 |
| 2 | 321 | 7.40 | 10.90 | 23.05 |
| 3 | 96 | 7.55 | 17.71 | 40.63 |
| 4 | 841 | 14.28 | 47.09 | 72.41 |
| 5 | 446 | 3.59 | 3.59 | 3.59 |
| **overall** | **1986** | **9.90** | **25.23** | **41.59** |

- raw path（**wzc1 local**，非 diskB）：`locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/`（`scores.json` + `preds_shard{0..7}of8.jsonl` + `judge_cache.jsonl` + `eval_config_shard*of8.json`）。`eval_config` 核实 = config #2（`resume_j=0, selector=iter_bm25, topk=12, iter_hop_topk=4, sink_tokens=bos, chunk_size=512, lora_adapter="", use_chat_template=false, enable_thinking=false, Qwen3-8b-local`）。REUSED——judge（gpt-4o）已算，无需重跑。

### A.6 [BLOCKED-DATA]

- **无 BLOCKED-DATA 项。** 全部 5 benchmark 的 j=0（config #2）raw predictions 均可达（RULER/LongBench/LongEval/BABILong 在 diskB `.73`，LoCoMo + BABILong 交叉在 wzc1 local），且已逐 cell CPU 重算核实。无任何 cell 需从 macro 反推。

---

## Part B — 统一读长术语（静态扫描，NOT modifying `.tex`）

**权威口径：** 名义满 pack = **6,657 tokens**（BOS 1 + top-12×512=6144 + query ≤512）；RULER/store-scaling
**实测均值 = 6.2–6.5k**（短尾 chunk/query，即 A.1/A.3 的 avg_read_len 6.18–6.64k、LongEval 4k 档更低）。
`approximately 6.5k` 仅作简写。

扫描命令：
```
grep -rniE "6,?657|6\.2|6\.3|6\.4|6\.5k|read.?len|read.?length|pack.*token|top-?12|nominal" \
  paperA/sections/00_abstract.tex paperA/sections/04_methodology.tex \
  paperA/sections/05_experiments.tex paperA/sections/08_appendix.tex \
  paperA/sections/08_statistics_appendix.tex paperA/sections/tab_*.tex
```

### B.1 结论：**未发现硬矛盾（no contradictory-configuration sentence）**

全文一致地维持了 nominal(6,657) vs actual(6.2–6.5k) 的区分。canonical 调和句在
`08_appendix.tex:117` —— *"The top-12 pack is 6,657 tokens and averages about 6.5k on RULER."* —— 正确。
`tab_store_scaling.tex` 表体给出实测 6,208–6,211 (NIAH) / 6,463–6,464 (VT)，caption 写 "about 6.2–6.5k"，
是 nominal cap / per-example actual / aggregate mean 三者区分的正面范例。**无任何句子把 6,657 与 6.2–6.5k
写成互斥/矛盾的同一量。**

### B.2 命中清单（loose-wording，非矛盾，可选收紧；行号 + 原文 + 建议）

| 文件:行 | 原文（节选） | 判定 | 建议改法（等用户签字） |
|---|---|---|---|
| `04_methodology.tex:17` | "The flagship reads approximately 6.5k tokens regardless of stored-context length." | OK（简写） | 保留；如求严谨可加 "(nominal cap 6,657)"。 |
| `00_abstract.tex:22` | "A fixed model-side read remains about 6.2--6.5k tokens for stores tested to 4M" | OK（actual mean） | 保留。 |
| `05_experiments.tex:11-12` | "the read is about 6.5k tokens" | OK（简写） | 保留。 |
| `05_experiments.tex:73` | "At the same nominal $\sim$6.5k-token read budget" | 轻微不精确：把 6.5k 标为 "nominal"，但 nominal cap 是 6,657，6.5k 是四舍五入的**实测**均值 | 建议改 "nominal 6,657-token pack (measured $\sim$6.5k)" 或删 "nominal" 只留 "$\sim$6.5k-token measured read budget"。 |
| `08_appendix.tex:117` | "The top-12 pack is 6,657 tokens and averages about 6.5k on RULER." | **正确（canonical 调和）** | 保留原样；建议把此句的显式分解（BOS 1 + 12×512 + query ≤512 = 6,657）在此或 methodology 补一次，便于 reviewer 核 nominal cap。 |
| `08_appendix.tex:217` | "a roughly 6,657-token pack" | 轻微不当：6,657 是**精确** nominal cap，"roughly" 应挂到实测 6.5k，不该挂到 6,657 | 建议改 "a 6,657-token nominal pack (measured $\sim$6.5k)"。 |
| `tab_slm.tex:22-23` (caption) | "a measured read of about 6.5k tokens, while StreamingLLM retains the same nominal budget" | OK（CoMem measured / SLM nominal，二者本就不同口径，已区分） | 保留；如求一致可点明 CoMem nominal 亦为 6,657。 |
| `tab_chunk.tex:14` | "512 & 6,657 & ..."（read tokens 列） | OK（chunk-size cap 列，nominal） | 保留（此列是 nominal read cap，正确）。 |

**B 净结论：** 无强制修改项。术语体系自洽；仅 `05_experiments.tex:73` 与 `08_appendix.tex:217` 两处把
"nominal/roughly" 挂错到了另一口径，属可选微调；建议全文补一次 6,657 的显式分解式。

---

## Part C — 匿名与可复现性扫描（静态扫描，只报告，NOT modifying files）

扫描对象：`paperA/main.tex` + `paperA/sections/*.tex` + `paperA/qcmem.bib` + 编译稿 `paperA/main.pdf`（正文文本 + metadata）。
匿名状态：`\author{Anonymous ACL Submission}`，`\usepackage[review]{acl}`，匿名仓库 `https://anonymous.4open.science/r/COMem-Anonymous/`。

扫描命令：
```
# .tex / .bib
grep -rniE "wanjiedata|maas-openapi|star-proxy|hy-proxy|woa\.com|\.oa\.com|openapi|proxy" paperA/sections/*.tex paperA/main.tex
grep -rniE "outputs/|/apdcephfs|share_30|zwfy6|wzc1" paperA/sections/*.tex paperA/main.tex
grep -rniE "\b(28|29|30)\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b" paperA/sections/*.tex paperA/main.tex
grep -rniE "pighzliu|LiuHanzuo|lhz24|tsinghua|mails\.tsinghua" paperA/sections/*.tex paperA/main.tex paperA/qcmem.bib
# PDF 正文
pdftotext paperA/main.pdf - | grep -niE "wanjiedata|maas-openapi|star-proxy|hy-proxy|woa\.com|\.oa\.com|pighzliu|LiuHanzuo|lhz24|apdcephfs|zwfy6|wzc1|share_30|(28|29|30)\.[0-9.]+|outputs/qcmem|Tsinghua"
# PDF metadata
pdfinfo paperA/main.pdf   # Creator/Producer/Author/...
```

### C.1 命中与人工判定

| # | 位置 | 命中内容 | 类别 | 人工判定 | 建议 |
|---|---|---|---|---|---|
| 1 | `08_statistics_appendix.tex:79-81`（PDF 正文 line ~4277 同现） | `\url{https://maas-openapi.wanjiedata.com/api/v1/chat/completions}` | **真实第三方 judge endpoint 域名** | **需泛化**——非匿名 vendor 端点，reviewer 可据此识别机构/供应商 | 改为 "an OpenAI-compatible \texttt{gpt-4o} chat-completions endpoint"（**保留** model name `gpt-4o`、`seed=1`、无 client 温度/top-p、"endpoint does not expose a dated model snapshot"、4×重试指数退避、失败保守判 WRONG、CORRECT/WRONG 模板——这些是必要复现信息）。完整 endpoint 保留在非投稿 status/raw manifest。 |
| 2 | `08_statistics_appendix.tex:9` + `08_appendix.tex:139` | `\path{outputs/qcmem_distill_qwen_j12_r32_4k/final}` | 内部 checkpoint 相对路径 | **borderline / 倾向保留**——已是**匿名相对路径**（无 `/apdcephfs`、无用户名、无 IP），且原文已注 "this internal path is reported only for reproducibility"，并配套给出 **权威 Final weight SHA-256 `dd09cd17…`** | 可保留；若审稿求稳可软化为 "the released adapter artifact (SHA-256 below)"。**不要删 SHA-256 / rank/α/mounted modules**——那是权威复现标识。 |

### C.2 CLEAN（零命中，无需动作）

- **节点/IP**：`.tex`、`.bib`、PDF 正文均**无** `28.*/29.*/30.*` IP。
- **用户名**：无 `pighzliu` / `LiuHanzuo` / `lhz24` / `Tsinghua` / `mails.tsinghua`。
- **内部绝对路径**：无 `/apdcephfs` / `zwfy6` / `wzc1` / `share_30…`（仅有 #2 的匿名相对 `outputs/…`）。
- **其他 proxy 域名**：无 `star-proxy.oa.com` / `hy-proxy.woa.com` / `woa.com` / `.oa.com`。
- **PDF metadata**：`Creator: LaTeX with hyperref`，`Producer: xdvipdfmx (0.1)`，**无 Author/Title/Subject/Keywords 泄露**（clean）。
- **author 块**：`Anonymous ACL Submission`；匿名仓库用 `anonymous.4open.science`（合规）。
- **bib**：无身份泄露。

### C.3 必须保留（KEEP）的复现信息——已确认在稿中且不建议删

- model name `gpt-4o`；judge `seed=1`；scoring policy（CORRECT/WRONG、4×重试、失败保守判 wrong、gold "OR" 拼接、fallback substring vote）。
- "The endpoint does not expose a dated model snapshot" 限制说明（`08_statistics_appendix.tex:82`）。
- weight SHA-256 `dd09cd17457c63578c0f38dab79b287ab5da6e3f14c119aedafec1c34400536f`；backbone revision `b968826d9c46`。
- seed 42（RULER/训练）、seed 1234（LongEval）；chunk 512 / top-12 / BOS sink / iter_bm25 / rounds=0 / chat_template=False。
- 匿名相对 artifact 路径（`outputs/qcmem_distill_qwen_j12_r32_4k/final`，见 C.1#2 判定）。

---

## MAIN 待办（需 MAIN/用户在 `.tex` 上执行的具体编辑；subagent 与 MAIN 当前均不改 `.tex`，等用户签字）

1. **[C — 必改，唯一硬 action]** `paperA/sections/08_statistics_appendix.tex:79-81`：把 judge endpoint URL
   `https://maas-openapi.wanjiedata.com/api/v1/chat/completions` 泛化为 "an OpenAI-compatible \texttt{gpt-4o}
   chat-completions endpoint"。保留 model name / seed=1 / 无温度top-p / "no dated snapshot" / 重试与保守判错策略。
2. **[C — 可选]** `08_statistics_appendix.tex:9` & `08_appendix.tex:139` 的 `outputs/qcmem_distill_qwen_j12_r32_4k/final`：
   如审稿求稳，软化为 "the released adapter artifact (SHA-256 below)"；否则可保留（已是匿名相对路径 + 有 SHA-256）。
3. **[B — 可选微调]** `05_experiments.tex:73` 把 "nominal $\sim$6.5k-token read budget" 改为
   "nominal 6,657-token pack (measured $\sim$6.5k)"。
4. **[B — 可选微调]** `08_appendix.tex:217` 把 "a roughly 6,657-token pack" 改为
   "a 6,657-token nominal pack (measured $\sim$6.5k)"（"roughly" 挂到实测 6.5k，不挂 6,657）。
5. **[B — 建议增补]** 在 `04_methodology.tex` 或 `08_appendix.tex:117` 补一次 6,657 的显式分解：
   BOS 1 + top-12×512 (=6,144) + query ≤512 = 6,657（nominal cap），实测均值 6.2–6.5k。
6. **[A — 呈现]** 若要在附录补 j=0 cell 表：RULER j=0 用 **Cohort B 单列附表**（A.1(i)，`niah_single_3`），
   **严禁并入 `tab:h2h` Cohort-A 或与 `niah_single_2` 行拼接**；P0.13 paired 15-cell（A.1(ii)）单列 paired 附表。
   LongBench(A.2)/LongEval(A.3)/BABILong(A.4)/LoCoMo(A.5) 直接用本文表格 + raw path + n。
7. **[PDF]** 完成上述编辑后 MAIN/用户重新编译 `paperA/main.pdf` 并复核页数与 metadata（当前 metadata 已 clean）。

> 本任务未跑任何 GPU / 模型 forward；A 全部 cell 由现有 raw predictions CPU 重算核实（无 BLOCKED-DATA）；
> B/C 为静态扫描 + 建议，未改任何 `.tex`。
