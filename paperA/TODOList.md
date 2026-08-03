# Paper A：ARR 修改与补充实验清单

> 截至 2026-08-02，Pareto、probe、store/I/O scaling、same-depth controls、P0.12/P0.13 paired quality--latency、P0.15 audit、P1.6 SnapKV/PyramidKV、P1.7 continuous-prefix oracle 和 **P2.4 多深度 quality--latency 曲线（j=6/9/18 训练+eval 全部验收，2026-08-02）** 均已完成并集成。
>
> **2026-08-03 独立 ARR 重审后新增实验缺口：P0.16--P0.20 为决定投稿强度的优先诊断；P1.8--P1.10 为系统/基线补强；P2.5 为可选泛化验证。P0.16 已完成并证明 document context 可全额回收 multikey gap；P0.17 overlap Write 正在运行。用户指定 P0.20 equal-latency retrieval-budget comparison 为当前最高优先级的新实验；write-path 训练仍需等待 P0.18 的机制判定。**
>
> 实验 agent 完成条目后，必须填写 raw path、checkpoint/config、代码版本、命令、硬件、统计口径和对论文结论的影响。负面结果不得删除。

## 当前仍缺的提交前工作

1. **当前运行**：完成 P0.17 overlapping Write，不中断已有 run。
2. **最高优先级新实验**：P0.20 在相同在线延迟预算下比较 text RAG 与 CoMem 的可读证据量和质量；先用现有 BM25 路径闭合 harness，再用 P1.9 的 dense retriever 复现主结果。
3. **机制补强**：完成 P0.18 位置/上下文拆解和 P0.19 retrieval/readout decomposition。
4. **部署闭环**：用 P1.8 给出 repeated-query serving crossover。
5. **条件任务**：P1.10 write-path distillation 只能在 P0.18 证明 Write 表示或位置接口可学习修复后启动。

## 状态规范

- `[TODO]`：未开始
- `[RUNNING]`：填写节点、PID/job ID、开始时间和预计完成时间
- `[VERIFY]`：结果可能已存在，必须先核验 provenance、协议和原始文件
- `[BLOCKED]`：写明缺少的 checkpoint、数据或代码
- `[DONE]`：结果、路径、复现方式和论文修改均已完成
- `[NEGATIVE]`：实验完成但不支持预期；仍须完整保留

---

# P0：提交前必须处理

## P0.1 解释或重测 Table 3 与 Table 9 的效率冲突

- **状态**：`[DONE]`（2026-07-31：已确认两数来自 persistent-index online-query 与 full-ingest phase sweep 两种计时约定。正文改用同平台 L20A、LoRA-on 的保守 headline：128k prefill `2.74×`、18.54 vs 89.39 GB；旧 `7.83×` 仅作为不同口径 cohort 说明，不再混入主结论。）
- **类型**：系统实验 + Paper 修改；最高优先级。
- **问题**：Table 3 报告 128k CoMem `j=12` full-write-inclusive prefill `1.917 s`，但 Table 9 报告同深度 Write alone `7.79 s`、Read `0.722 s`。若计时定义一致，两者不可能同时成立；摘要级 `7.83×` 依赖前者。
- **行动**：
  1. 查明两张表原始日志、commit、节点、实际 GPU 型号、batching、chunk Write 实现、warmup/compile、计时边界；
  2. 在同一节点、同一 commit、同一 dtype/SDPA、同一 chunk/top-k 下重测 full context、`j=0`、`j=12 frozen`、`j=12 + LoRA`；
  3. 分开报告：index build、document Write、query Write、retrieval、host→device transfer、Read/prefill、decode、peak GPU memory、peak host memory；
  4. 报告 median、P5/P95 或 IQR、warmup 次数、重复次数；
  5. 在冲突解决前，Paper 中弱化或移除 `7.83×` headline。
- **验收条件**：所有方法使用同一 harness；raw timestamps 可复算表格；论文明确解释旧 `1.917` 与 `7.79` 的来源。

**结果填写（2026-07-31 核实：非真冲突，两种 write 约定）**

- **核心结论**：`1.917 s`（Table 3=`tab_eff.tex` panel a）与 `7.79 s`（Table 9=`tab_pareto.tex` panel b）测的是**不同量**，来自**两个不同 bench 脚本 + 两种 write 约定**，可同时成立：
  - `1.917 s` ← `scripts/bench_qcmem_vs_dense.py`（=benchmark.md §1c 口径）：**select-first 部署路径**——先 bm25 选 top-12，再只把**检索出的 pack（≈6,657 tok）+ query** 过 forward（写下层 + 读上层）。脚本 docstring 明写 "select topk FIRST and write only those (**constant write**)"→ prefill 对上下文长度**恒定**（8k/32k/128k ≈1.82/1.90/1.92 s）。7.83× headline (=15.014/1.917) 即此口径，是**在线 per-query 部署成本**。
  - `7.79 s` ← `scripts/bench_qcmem_vs_fullctx.py`（P3.1 depth-Pareto，task #78）：**write-all 约定**——按 benchmark spec **把全部 128k context chunks 都过 `embed+layers[0:12]`**（`write_s`=O(L) 一次性离线 ingest），并**单列** `read_s`=0.722 s（per-query 上层读）。docstring line 50-53 明写 "We WRITE ALL N context chunks... we **intentionally show the linear write-all number** here"。
  - ∴ `1.917 s`（per-query，写 6.7k tok read-pack）与 `7.79 s`（一次性，写 128k 全语料）**token 数差≈19×、频次不同**，非矛盾。混淆点=tab_eff 把 1.917 s 叫 "full-write-inclusive prefill"，字面像"含全量写"，实则只含**该 query read-pack 的下层写**。
- **∴ labeling 问题，非 data 问题**。剩余动作（paper-only）：把 tab_eff panel(a) "full-write-inclusive prefill" 改名为 "**online query prefill (persistent depth-$j$ index; writes retrieved pack + reads upper band)**"，并在 §efficiency 明确 7.79 s 是**一次性离线 ingest（O(L)，跨 query 摊销，break-even≈26 queries，见 P0.2）**。tab_pareto 已正确单列离线 write，无需改数。
- Table 3 raw log/脚本：`scripts/bench_qcmem_vs_dense.py`（median-of-3, chunk512, bf16/SDPA, 1×H20, peak=`max_memory_allocated`）；汇总 `bench_qcmem_vs_dense_result.txt`（§1c 全 scale：8B@128k prefill 1.92 s / mem 恒 17.9 G / prefill× 57.5）。
- Table 9 raw log/脚本：`scripts/bench_qcmem_vs_fullctx.py`（3-phase 计时 `prefill_s=write_s+select_s+read_s`；write_s@128k=7.79 s，read_s=0.722 s，median-of-3，同一 top-12 pack ≈6,657 tok，只变 j）。
- 重测节点/GPU：1×H20（97.8 GiB），bf16 + SDPA。
- 原因解释：见上"核心结论"——两脚本 write 约定不同（constant-write vs write-all）。

| Method (128k) | Write (one-time, O(L)) | Read/prefill (per-query) | Decode | Peak GPU | Raw path |
|---|---:|---:|---:|---:|---|
| full context | — | 15.014 s | 28.14 ms/tok | 89.36 GB | tab_eff (a/b) |
| `j=0` (RAG, retrieve+full recompute) | 0.23 s | 1.01 s | — | 18.3 GB | tab_pareto |
| `j=12` frozen (constant-write deploy) | (index 一次性) | **1.917 s** | 26.39 ms/tok | 18.26 GB | tab_eff (a/b) |
| `j=12` frozen (write-all Pareto) | **7.79 s** | 0.722 s | — | 18.3 GB | tab_pareto |
| `j=12` + LoRA | (同 frozen) | +18% prefill | 31.70 ms/tok | 18.54 GB | tab_eff-lora |

- 最终可保留的 headline：**7.83× online prefill @128k（1.917 s vs 15.014 s），恒定 ≈18 GB vs 89 GB**（select-first 部署口径）；一次性离线 ingest 7.79 s 单列摊销报告。
- ✅ 可选补强（2026-07-31 **已交付**）：用**同一 harness**（单次 model-load）一次跑齐 full / j0 / j12-frozen / j12+LoRA 的分项 timestamps，写入 `status/P0_1_UNIFIED_TIMING.md`（flagship Qwen3-8B j=12+LoRA，1×H20，8k/32k/128k）：write-all decomp（`bench_qcmem_vs_fullctx.py`）write_s 0.355/1.444/5.826=O(L)、read_s 恒 ~0.849、peak 17.6→18.54 GB、full-ctx@128k OOM；dense-speed（`bench_qcmem_vs_dense.py`）QCMem 恒定 prefill ~1.05–1.10s vs Dense 1.20/7.33/71.37s = **64.9× prefill @128k**、decode 20–32×；**fresh LoRA-on break-even ≈20 queries**。read pack 6657 tok + 128k peak 18.54 GB 与正文精确一致。paper 效率表未改（cohort 差异已 flag，非错）。commit `6dffa59`（未 push）。
- Paper 修改位置：abstract / §efficiency / tab_eff caption（relabel）；tab_pareto 无需改。

## P0.2 将 `j=0` / text retrieval 升为核心端到端基线

- **状态**：`[DONE]`（2026-08-01：统一 4-config Pareto 已交付 `status/P0_2_PARETO_RESULTS.md`——quality×latency×storage 三轴 + Q={1..128} 摊销 + frozen/adapted 分行。fresh GPU 延迟/存储实测 on .104：CoMem 8192 B/tok=18× less than full KV；flat 18.5 GB @128k vs full-ctx ~89.4 GB（footprint：L20A/183GB 实测 89.39 GB 成功；同 single-forward write-all harness 在 H20/97.8GB OOM——见 ★harness caveat）；prefill 恒 0.8–1.3 s vs full-ctx 50.59 s@128k=38× speedup；h₁₂ pack 54.5 MB H2D 1.20 ms 恒定；break-even write-all>select-first @Q≥12、>j0-RAG @Q≥~17–20、>full-ctx @Q=1。RULER macro：full-ctx 78.73（128k→0）/ j0-RAG VT100（NIAH N/F）/ frozen-j12 **8.01**（dominated）/ **CoMem+LoRA 96.07**（唯一 top-quality+128k-feasible）→ 88pp gap=LoRA/蒸馏贡献，非 depth-cache。**剩余开口已闭合：config #2 j=0 text-RAG 5-benchmark 已补齐（2026-08-01，`status/P0_2_CONFIG2_JRAG_RESULTS.md`；chat=False / iter_bm25 / tk12 / resume_j=0 / NO-LoRA / Qwen3-8B）：RULER 15-cell macro **99.20**（niah_single 99/99/98/96/99、multikey 100/100/99/99/99、VT 全 100，n=100）；LongBench 6-QA macro F1 **12.31**（4-QA 子集 harness 未正式定义→6-QA 权威）；LongEval mean(8k–128k) **97.2%**；BABILong qa1 98/84/80/73/60/34/35、qa2 58/53/51/44/37/17/12、qa5 70/73/61/57/69/53/60（0k–32k，n=100）；LoCoMo F1 9.90 / acc 25.23% / gpt-4o judge **41.59%**（n=1986）。**关键：config#2 = j=0 全深度检索 teacher，在 RULER(99.2>96.07)/LongBench(12.31>12.15)/LoCoMo/尤其 LongEval(97.2%≫69.0%) 上 match 或超过 flagship #4 → 证实为蒸馏上界**；唯一例外 BABILong qa1 长档（full-context #1 的 63 优于 BM25 检索的 35）。provenance：NIAH fresh `ruler_results/p0_2_c2_j0_iterbm25_niah_chatFALSE/`、VT reused-verified `presub_A_kvdirect_iterbm25_vt/`、LongBench/LongEval/BABILong fresh `*_p0_2_c2_j0_iterbm25_chatFALSE/`、LoCoMo reused-verified `locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/`（在 wzc1 盘，非 diskB）。§4c 数值化后 P0.2 完全无 NOT-FOUND。**）
- **原状态**：`[TODO]`
- **类型**：系统实验 + 评测 + Paper 修改。
- **问题**：现有 `j=0` 结果已证明 selection 本身很强：LoCoMo `41.59` 高于 CoMem `38.27`，BABILong qa1/qa2 也更高。当前稿缺少贯穿质量、延迟、写入和存储的统一 Pareto，无法回答 depth cache 相对普通 bounded text retrieval 的额外价值。
- **行动**：
  1. 对主任务统一比较 full context、`j=0`、`j=12 frozen`、`j=12 + LoRA`；
  2. 报告 persistent bytes/token 与 128k、1M、1B token 的绝对容量；
  3. 报告初始 index/write cost、每查询 retrieval/transfer/read/decode cost；
  4. 计算 `Q={1,2,4,8,16,32,64,128}` 查询的累计和摊销成本；
  5. 绘制 quality–latency–persistent-storage Pareto；
  6. 分别给出 frozen 与 adapted CoMem，避免把 LoRA 质量和 adapter-free 系统数字混成一个配置。
- **必须说明**：CoMem bf16 residual 为 `8192 bytes/token`；约 `1 GiB/128k tokens`、`8.2 GB/1M tokens`，并与 text/token IDs、index 和 full KV 比较。

**结果填写（2026-07-31：质量/延迟/存储三轴大部分已有数据，可拼装）**

- 统一 benchmark/tasks：RULER 15-cell macro + LoCoMo GPT-4o judge + BABILong 3-task + LongBench macro-F1 + LongEval（见 `tab_overview.tex`）；延迟/存储见 `tab_pareto.tex`（同机 H20 split-depth sweep）。
- harness/commit/hardware：质量=chat=False 统一协议（8 shards，n=100/cell RULER，全集 LoCoMo）；延迟=`bench_qcmem_vs_fullctx.py`（3-phase）+ `bench_qcmem_vs_dense.py`；1×H20 bf16/SDPA，median-of-3。
- **bytes/token**：text token-IDs ≈ **2–4 B/tok**（int16/int32）；`j=0` BM25 index ≈ token 存储量级（few B/tok）；**CoMem = 8192 B/tok**（Qwen3-8B hidden 4096 × bf16 2B，单个 depth-$j$ residual）；full KV = 全 36 层 K+V（无界，→ 128k 时总显存 89.36 GB）。
- **CoMem 绝对容量**：≈ **1 GiB / 128k tok**（8192×131072=1,073,741,824 B）、**8.2 GB / 1M tok**、**8.2 TB / 1B tok**——比 full KV 省，但比纯 text 存储贵 ~2000×（换来免重算下层）。
- **延迟-存储 Pareto（128k，1×H20，median-of-3，同一 top-12 pack ≈6,657 tok，只变 $j$，无 LoRA）**：

| Split $j$ | 上层数 | Read (s, per-query) | Write (s, one-time O(L)) | Peak (GB) |
|---:|---:|---:|---:|---:|
| 0 (RAG: 检索+全重算) | 36 | 1.01 | 0.23 | 18.3 |
| 6 | 30 | 0.86 | 4.61 | 18.3 |
| 9 | 27 | 0.79 | 6.11 | 18.3 |
| **12 (CoMem)** | 24 | **0.72** (−29% vs j0) | 7.79 | 18.3 |

- Table-9-implied旧 break-even：约 `27 queries`（仅供核验，不可直接作为最终部署结论）
- **新 break-even（read-only 口径）**：j=12 比 j=0 多付一次性 write 7.79−0.23=**7.56 s**，每 query 省 read 1.01−0.72=**0.29 s** → break-even ≈ **26 queries**（与旧 27 一致）。若把 decode/prefill 全省算入更快；**Q<26 用 j=0（RAG）更省，Q≥26 用 CoMem depth-cache 更省**。
- Pareto 图：已存在——`paperA/sections/tab_pareto.tex`（两 panel：Read latency × 长度 + 128k step/write/mem）+ `paper/sections/tab_pareto.tex`；图版见 `paperA/figures/`。
- raw results：质量 `outputs/`+`ruler_results/`（见 §stats appendix，从 saved shards 重算无需 GPU）；延迟 `bench_qcmem_vs_*_result.txt`。
- **结论：CoMem 何时优于 `j=0`**：
  - **质量**：`j=0`（RAG，41.59）> CoMem（38.27）on LoCoMo；`j=0` 亦 ≥ CoMem on BABILong qa1/qa2（selection 本身强，task #71 已证）。CoMem 优势在 **qa5（聚合）+ 超原生窗口稳定性（128k full-ctx=0，CoMem 存活）+ 摊销后在线 read/decode 成本更低**。
  - **延迟/存储**：Q≥26 时 CoMem 每 query read −29% / decode −28%，显存跨长度恒定 ≈18 GB（vs full 89 GB）。
  - ∴ 贡献表述改为 **"new trade-off"**（见 P0.6）：高查询频次 + 超窗口 + 需 qa5 类聚合时值得付 residual-store 与 Write；否则 bounded text retrieval（j=0）already strong。
- **★ harness caveat（用户 2026-08-01，写 .tex 必须遵守）**：full-context @128k 的 "不可行" 是 **硬件 + harness 特定**，**不得笼统写 "full context 128k infeasible"**。事实：single-forward write-all footprint ~89.4 GB（weights 16.4 + KV 19.3 + full-seq logits [1,131072,151936]×2B 39.8）——在 **L20A/183GB 上实测 89.39 GB 成功运行**，在 **H20/97.8GB 上同 harness OOM**（89 GB 稳态 + logits 分配瞬时峰值越过 97.8）。论文以 **H20 cohort** 为主口径报告即可（用户指示"直接全写 H20 的结果"），但**必须写明 "on H20 (97.8 GB), single-forward write-all harness"**，并注明 L20A 同 footprint 可跑通 → 结论是 "在 H20 write-all harness 下 full-ctx@128k OOM、CoMem 是唯一可行 operating point"，**非** "128k full-context 普遍不可行"。（line 85 的 "128k 时总显存 89.36 GB" 是 footprint 陈述、非 OOM 断言，保留。）

## P0.3 原生窗口与长度扩展 baseline

- **状态**：`[DONE]`（2026-07-31：已移植完整五行 scaling 表；摘要、正文、结论和 teaser 使用 CoMem+LoRA 无 YaRN `n=50` 对 KV-Direct+YaRN `n=100`，并报告 CoMem+LoRA+YaRN tax。原生长窗口自然任务仍单列于 P2.1。**2026-07-31 追加：mixed-n caveat 已彻底关闭——CoMem 两行（±YaRN）在 .104 重跑到 `n=100`，与 KVD 行全部 matched-n=100，见下「★ matched-n=100 更新」块。**）
- **类型**：先审计已有实验；必要时补评测；Paper 修改。
- **问题**：Paper A 的 `97.05 vs 78.80` 包含 Qwen3-8B 原生 40,960 窗口之外的未扩展 KV-Direct，不能解释为普遍准确率优势。8k–32k 内 KV-Direct 实际更强。
- **重要资产**：并行稿 `paper/` 已出现 YaRN 结果（如 `paper/sections/tab_scaling.tex`、`tab_yarn_tax.tex`，包含 KVD+YaRN 和 CoMem+LoRA+YaRN）。**先核验 raw data、样本数、模型配置和 commit；核验通过则移植，禁止无故重复跑。**
- **行动**：
  1. 主表/摘要分开报告 native-window (`<=32k`) 与 extrapolation-region (`64k/128k`) aggregate；
  2. 核验并移植同 backbone YaRN baseline；如现有数据不可靠，重跑；
  3. 最好补一个原生 128k+ backbone 的自然任务，而不只 RULER needle；
  4. 将主张改成 bounded working set / continued usability，而非一般 accuracy superiority；
  5. 不得把 unextended full-context 叫作 length-extended upper bound。

**结果填写（2026-07-31：可从 `paper/sections/tab_scaling.tex` + `tab_yarn_tax.tex` 直接移植；仅需 native/extrapolation 分列改写 + n 标注）**

- `paper/` YaRN raw path：`ruler_results/p32_from_82/kvd_yarn_*` vs `ruler_results/p32_from_82/kvd_unext_*`（YaRN delta 表源）；scaling 主表 = `paper/sections/tab_scaling.tex`（含 CoMem / full-ctx / KVD+YaRN / CoMem+LoRA / CoMem+LoRA+YaRN 五行 × 8k–256k）。
- provenance/commit/checkpoint：Qwen3-8B backbone；YaRN `factor=4`（effective window **163,840 tok**），原生窗口 **40,960（~40k）**→ **≤32k=native，64k/128k=extrapolation**。selector = `iter_bm25`（hop4, top-12, chunk512, sink=bos）。
- **样本数（关键 caveat）**：KVD 行（含 KVD+YaRN）`n=100`；CoMem / CoMem+LoRA / CoMem+LoRA+YaRN 行 `n=50`。移植主表须标 mixed-n，勿把 n=50 与 n=100 直接当同精度对齐。
- 是否可直接移植：**可**——数据已在并行 `paper/` 稿，只需在 Paper A 主表/摘要按 native vs extrapolation 分列改写，无需重跑（YaRN baseline 已存在且 provenance 清晰）。
- **native-window（≤32k）RULER macro**（8k/16k/32k 三档均值，逐 task）：
  - niah_single：CoMem+LoRA 95.3（100/86/100）；KVD+YaRN 100（100/100/100）
  - niah_multikey：CoMem+LoRA 96.7（94/96/100）；KVD+YaRN 96（98/94/96）
  - var-track：CoMem+LoRA 98.4（98.0/98.4/98.8）；KVD+YaRN 75.1（99.2/99.4/**26.6**←YaRN 在 32k 有 −73.4pp in-window tax）
  - **→ native macro：CoMem+LoRA ≈ 96.8，KVD+YaRN ≈ 90.4**。**结论：native window 内 KV-Direct（unextended）single/multikey 与 CoMem 互有胜负、并不弱；97.05 vs 78.80 的整体优势主要来自 extrapolation 区。**
- **extrapolation-region（64k/128k）**：
  - niah_single：CoMem+LoRA 100/98；KVD+YaRN 100/100
  - niah_multikey：CoMem+LoRA 98/96；KVD+YaRN 91/89
  - var-track：CoMem+LoRA 97.6/**98.4**；KVD+YaRN 67.2/**57.8**；unextended full-ctx 128k = **0**
- **64k/128k YaRN 结果（headline）**：128k var-track 排序 **CoMem+LoRA 98.4 > CoMem+LoRA+YaRN 87.6 > KVD+YaRN 57.8 > unextended full-ctx 0**（**+40.6pp** CoMem 相对 length-extended reference）。YaRN tax 表：128k rescue single **+100** / multikey **+89** / var-track **+57.8**（相对 unextended KVD）；但 32k var-track YaRN in-window tax **−73.4**（RoPE rescaling 与短程注意力冲突）。
- native 长窗口 backbone/natural task：**未补**（P2.1 待办；当前 extrapolation 证据全为 RULER synthetic needle，无原生 128k backbone 自然任务）。
- 修改表/图/段落：(1) 摘要/主表把 `97.05 vs 78.80` 拆成 native (≤32k) 与 extrapolation (64k/128k) 两 aggregate；(2) 主张改为 **bounded working set / continued usability past native window**，非 general accuracy superiority；(3) 明确 unextended full-ctx=0 是"未扩展崩塌"，**不得**称作 length-extended upper bound；(4) 移植 `tab_scaling`+`tab_yarn_tax` 到 Paper A 并标 mixed-n。

**★ matched-n=100 更新（2026-07-31，.104 8×H20，subagent ab0993f；关闭 mixed-n caveat）**——完整记录 `status/P0_3_MATCHED_N100.md`（commit `2bd16ff`，LiuHanzuo，未 push）。CoMem ± YaRN 两行在 .104 全部重跑 `n=50→n=100`，写入 NEW 目录 `ruler_results/comem_lora_native_n100/`（Arm A，native `Qwen3-8b`）+ `ruler_results/comem_lora_yarn_n100/`（Arm B，`Qwen3-8b-yarn` factor-4 eff.163,840），n=50 原目录未动。config 与旗舰同口径：resume_j=12 · iter_bm25(topk12/hop4/rounds0) · sink=bos · chunk512 · chat=False · seed42 · bf16/sdpa · adapter `qcmem_distill_qwen_j12_r32_4k/final` · needle=`niah_single_3`（匹配 KVD）。30 cell 全过 sanity（n=100，0 空，0 OOM）。

- **Arm A CoMem+LoRA（native）n=100** [8k/16k/32k/64k/128k]：single 100/91/97/98/98 · multikey 94/91/99/90/93 · **var-track 96.2/98.0/98.2/98.6/99.0**。macro(15)=**96.07**。
- **Arm B CoMem+LoRA+YaRN n=100**：single 100/90/97/97/98 · multikey 82/85/94/88/93 · **var-track 88.6/90.0/89.4/95.0/93.6**。macro(15)=**92.04**。
- **YaRN in-window tax on CoMem ≈ −4.0pp macro**（single −0.4 / multikey −5.0 / var-track −6.7）——YaRN 对 CoMem **不是 no-op**（RoPE 对全 position 重标度，即便 read pack 恒在窗内也有 in-window tax）。
- **HEADLINE @128k var-track（全部 n=100）**：CoMem+LoRA native **99.0** > CoMem+LoRA+YaRN **93.6** > KVD+YaRN 57.8 > full-ctx 0。排序 **PRESERVED**；**CoMem 相对 length-extended reference = 99.0−57.8 = +41.2pp**（原 n=50 是 98.4→+40.6pp）——**headline HOLDS 且略微 STRENGTHENS**。（+YaRN vs KVD+YaRN = 93.6−57.8 = +35.8pp。）
- **真 +YaRN 数字解决**：vt@128k = **93.6**（n=100）。provenance 裁定：`paper/sections/tab_scaling.tex` 的 degraded `CoMem+LoRA+YaRN` 行（n=50 vt 81.2/86.0/90.4/96.8/87.6）是**正确 provenance**；`paperA/sections/tab_scaling.tex` 现有 "+YaRN" 行（vt 99.2…/95.8）实为 **native mislabeled**（源 `qcmem_8b_iter_chatFALSE_ad`，niah_single_2），应替换为 Arm B 真 YaRN 数（88.6/90.0/89.4/95.0/93.6）；`comem_yarn_128k`=0.8 是坏 run，作废。
- **tex 待办（main 后续，非本轮）**：把 `paperA/sections/tab_scaling.tex` 的 CoMem+LoRA+YaRN 行换成 Arm B 真数、CoMem+LoRA 两行标 n=100（与 KVD 行 matched），mixed-n caveat 从"仅 KVD n=100"改为"全 matched-n=100"。old→new 逐 cell 对照见 `status/P0_3_MATCHED_N100.md`；delta 均在 n=50→100 采样方差内，无 cell 质变。

## P0.4 匿名化提交版本

- **状态**：`[DONE]`（2026-07-31：`main.tex` 增加匿名开关且默认匿名；编译 PDF 首页与 metadata 未显示作者身份，保留匿名 artifact 链接。注意正式 ARR 上传前仍需切换/核验 ACL review template。）
- **类型**：仅改 Paper/构建流程。
- **问题**：当前 `paperA/main.pdf` 首页含作者、单位、邮箱和实名 GitHub；若直接提交 ARR，存在 anonymity desk-reject 风险。公开非匿名预印本不等于 review PDF 可以非匿名。
- **行动**：
  1. 建立匿名构建开关或独立匿名入口；
  2. 删除作者/单位/邮箱、acknowledgments、实名仓库和可识别路径；
  3. 自引使用第三人称，不暴露身份；
  4. 匿名仓库链接只指向匿名 artifact；
  5. 对 PDF 文本和 metadata 做 identity scan。

**结果填写**

- 匿名 PDF：`paperA/main.pdf`（用 `\usepackage[review]{acl}` 构建，首页作者块受 ARR review 模式抑制）
- identity scan 命令/结果：`[DONE 2026-08-02]`；PDF 文本对真实 judge 域名、内部路径、节点/IP、用户名和单位均零命中；metadata 仅含 LaTeX/xelatex producer 信息，无 Author/Title/Subject/Keywords 泄露。P2.4 最终编译后再复扫。
- 匿名 artifact URL：`https://anonymous.4open.science/r/COMem-Anonymous/`（已在 `main.tex` footnote，仅指向匿名 artifact）

---

# P0：仅修改 Paper，可直接完成

## P0.5 收紧 headline 与 benchmark 命名

- **状态**：`[DONE]`（2026-07-31：已明确 15-cell/3-family RULER、qa1/qa2/qa5 BABILong、6-QA LongBench；删除异质指标 `Avg.` 列并区分原生窗口与 YaRN。）
- **类型**：仅改 Paper。
- **行动**：
  1. Abstract/Intro/Table 1 中将 `RULER` 改为 `three-family / 15-cell RULER subset`；
  2. 将 `BABILong` 改为 `qa1/qa2/qa5 BABILong subset`；
  3. 将 `LongBench` 改为 `six-dataset LongBench QA subset`；
  4. 明确这些 subset 的选择原则、是否在看结果前确定；
  5. 删除 Table 1 将五种异质指标等权平均的 `Avg.` 列，或将其降为非排名性描述；
  6. Abstract 明确 `97.05 vs 78.80` 含超原生窗口 stress cells；若保留则同时给 native-window aggregate；
  7. 不使用完整 benchmark 名暗示全套覆盖。

**完成记录**

- 修改文件：`00_abstract.tex`、`01_introduction.tex`、`05_experiments.tex`、`tab_overview.tex` 及各分项附表。
- 最终命名：three-family/15-cell RULER subset；BABILong qa1/qa2/qa5；six-dataset LongBench QA subset。
- subset selection rationale：显式列出任务支持与 scorer，不声称覆盖完整 benchmark suite；异质跨基准平均已删除。

## P0.6 重新定位核心贡献与 `j=0` 的关系

- **状态**：`[DONE]`（2026-07-31，Paper-only：摘要、正文和结论已明确 selection gain 与 depth-reuse gain；`j=0=41.59` 高于 CoMem `38.27`，贡献改为 quality--storage--query-cost trade-off。完整部署 Pareto 仍是 P0.2。）
- **类型**：仅改 Paper；若 P0.2 完成后再更新数字。
- **行动**：
  1. 明确区分 selection gain 与 depth-reuse gain；
  2. 将 `j=0` 作为主 baseline，而非附属 ablation；
  3. 正面陈述 `j=0` 在 LoCoMo、BABILong qa1/qa2 上更强，CoMem 在 qa5/online read cost 上有优势；
  4. 用“new trade-off”而非“全面优于 retrieval”描述贡献；
  5. 给出何时值得支付 residual-store 和 Write cost 的部署条件。

**完成记录**

- 修改文件：`00_abstract.tex`、`01_introduction.tex`、`04_methodology.tex`、`05_experiments.tex`、`06_conclusion.tex`、`07_limitations.tex` 及 Pareto/replay 表。
- 一句话核心定位：retrieval bounds the token working set; CoMem additionally materializes lower-layer computation for repeated queries, trading persistent storage and quality for cheaper model-side Read.

## P0.7 收紧“Understanding Is Done Early”机制表述

- **状态**：`[DONE]`（2026-07-31，Paper-only：统一改为 specified linear probe / truncation intervention / probe-dependent correlate，不再把 readout depth 写成因果定位。完整 probe 方法与稳健性仍是 P1.2。）
- **类型**：Paper 修改；完整支撑该主张仍需 P1.2。
- **行动**：
  1. 将强因果/普遍性表述改为 `task-relevant semantic information is linearly accessible by mid-depth under the specified probe`；
  2. 不把 RULER single-needle readout crash 等同于一般 understanding；
  3. 明确 content depth、readout-crash 和下游质量使用不同测量；
  4. 将 `causal truncation` 的因果范围限制为所执行的模型干预；
  5. 如果标题保留 `Understanding Is Done Early`，正文必须明确它是操作性 probe 结论，而非认知机制定论；否则考虑收紧标题。

**完成记录**

- 修改文件：`00_abstract.tex`、`01_introduction.tex`、`03_motivation.tex`、`05_experiments.tex`、`07_limitations.tex` 及 probe/readout 附表。
- 最终机制主张：specified linear probes expose task-relevant information before native readout under the tested tasks; this motivates an interface but does not causally localize understanding or a universal split depth.

## P0.8 修正系统描述和 Figure 1 配置混合

- **状态**：`[DONE — TEX INTEGRATED 2026-08-02]`。论文已统一为冻结的 released `Qwen/Qwen3-8B`（revision/hash 保留），明确我方不做 backbone continued pretraining、唯一学习权重为自蒸馏 LoRA；HBM-resident 主 timing 与独立 CPU/NVMe/network I/O microbenchmark 的边界也已写明。
- **类型**：仅改 Paper/系统 provenance。
- **已完成**：
  1. Figure 1 改为 YaRN accuracy + 同平台 LoRA-on memory/prefill，不再混合 adapter-free `7.83×`；
  2. “exactly resumes” 已限定为 independently written chunk-local states 条件下的 upper-layer continuation；
  3. 已说明 BOS sink 是 BOS token 经过 `[0:j]` 得到的单 token hidden；
  4. 已写明 generation 的 lower/upper KV caches、两个 position counters 和逐 token 同步。
- **仍需补**：
  1. 主 backbone 的精确 checkpoint ID/hash，并解释 `continued-trained Qwen3-8B base LM`；
  2. 部署态 residual store 在 GPU/CPU/NVMe/network 的位置，以及 retrieval/transfer 是否进入各 timing cohort。

**完成记录**

- Figure 1：`paperA/figures/teaser_results.pdf`
- decoding algorithm：`paperA/sections/04_methodology.tex`
- checkpoint ID/hash（✅ 2026-08-01 main 实测）：
  - **Backbone（冻结）** = 发布版 **Qwen/Qwen3-8B**（本地 `models/Qwen--Qwen3-8b`），HF snapshot revision `b968826d9c46dd6066d109eabc6255188de91218`；标准 config（36 层、hidden 4096、32 attn / 8 KV heads、native window 40960、rope_theta 1e6、bf16、transformers 4.51.0、apache-2.0）。其 model card `base_model: Qwen/Qwen3-8B-Base` → **发布版 Qwen3-8B 本身是 Qwen 官方从 Qwen3-8B-Base post-train 得到的模型；我方对 backbone 不做任何 continued-pretraining，直接冻结使用。** ⚠️ 论文若写"continued-trained Qwen3-8B base LM"属措辞不准，应改为"frozen released Qwen3-8B（Qwen 官方 post-trained，我方唯一学习权重是自蒸馏 LoRA）"。
  - **Flagship 唯一学习权重 = 自蒸馏 LoRA adapter** `outputs/qcmem_distill_qwen_j12_r32_4k/final/`：
    - `adapter_model.safetensors` **sha256 `dd09cd17457c63578c0f38dab79b287ab5da6e3f14c119aedafec1c34400536f`**（232,829,168 B，产于 2026-07-07 00:25）
    - `adapter_config.json` sha256 `244fb9e0fbccd0ef144fa8773986f7105921057e5c2199cb6cfdac562de7c059`
    - 配置：r=32、α=64、dropout=0、`layers_to_transform=12..35`（上 24/36 层，对应 j=12 split）、target_modules={q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}、peft 0.19.0、base=`Qwen--Qwen3-8b`。
    - 训练超参/数据/teacher 见 #66（LoRA 训练成本）、#69（复现超参表）、#77（seq_len=4096）；teacher = config #2（j=0 全深度 RAG，同 top-12 pack）。adapter README 为默认 PEFT stub（无额外 provenance）。
- storage placement 描述（✅ 2026-08-01 main，据 `status/P0_2_PARETO_RESULTS.md` §1c/§2D）：CoMem residual store = **write-once bf16 `h₁₂` 张量，8,192 B/tok**；在**本文所报的全部 timing cohort 中该 store 常驻 GPU HBM**（@128k flat ~18.5 GB），检索命中的**固定 54.5 MB top-12 pack 的 H2D 传输 = 1.20 ms（pinned）已计入 read cohort**（constant in L，不随上下文长度增长）。CPU-pinned / NVMe / network 分层部署的真实 I/O、传输与并发**不在当前 cohort**，属 **P2.2** future work（Tier-3 systems build）；论文应明确当前数字是"store 常驻 HBM、pack H2D 计入 read"的口径。

## P0.9 调整 MemoryLLM 与 LoCoMo judge 表述

- **状态**：`[DONE]`（2026-07-31：judge protocol/denominator/endpoint/paired-item CI/DeepSeek audit 均已在 `sections/08_statistics_appendix.tex`；最后缺口 = conversation-cluster bootstrap 的**数值 95% CI** 现已补齐 = **[+1.27, +8.32]**，point +4.81，bootstrap p≈0.004，commit `787427b`，见 `status/P0_9_CLUSTER_BOOTSTRAP.md`）
- **类型**：仅改 Paper；prompt sensitivity 和 judge 扩展见 P1。
- **行动**：
  1. 将无 native chat template 的 MemoryLLM 从主平均排名中移出或单列为 OOD diagnostic；
  2. 明确 no-chat protocol 是控制变量，不代表正常部署效果；
  3. 报告 conversation-cluster bootstrap 的**数值 95% CI**，不能只写 excludes zero；
  4. 明确第三方 `gpt-4o` endpoint 无 dated snapshot；
  5. 在许可范围内发布 judge prompts/inputs/outputs；
  6. 不让 10-conversation cluster inference 支撑过强普遍结论。

**完成记录（2026-07-31：大部分已在 `sections/08_statistics_appendix.tex`，数值 cluster CI 仍缺）**

- **denominator/protocol（已有）**：LoCoMo 全集 1,986 items = GPT-4o judge cat1–4 的 1,540 answerable + cat5 的 446 local abstention（空/拒答规则）。canonical 全集 point：CoMem+LoRA **38.27** vs KV-Direct **34.59**；common judged subset (n=1540)：CoMem **48.64** vs KVD **43.83**。
- **paired-item bootstrap（已有）**：10,000 resamples seed 1234 → **+4.81，95% CI [2.34, 7.27]**（per-item，全部 > 0）。
- **conversation-cluster bootstrap（✅ 数值 CI 已补齐 2026-07-31）**：paired cluster bootstrap（resample 10 conversations，10,000 resamples seed 1234）= **+4.81，95% CI [1.27, 8.32]**，比 per-item 区间宽（仅 10 clusters 意料之中）但**仍完全 > 0**（bootstrap two-sided p≈0.004）；10 会话级差异中 8 favor CoMem（conv4 −1.12、conv6 −4.67 favor KVD）。种子稳健（lower bound 跨 seeds{1234,1,42,2024,12345,7} 稳定 +1.17…+1.30）。appendix 定性表述已被 commit `787427b` 替换为该数值区间；脚本 `scripts/locomo_cluster_bootstrap.py`，报告 `status/P0_9_CLUSTER_BOOTSTRAP.{md,json}`。
- **DeepSeek 独立 judge audit（已有）**：200-item 分层子集，与 GPT-4o agreement **0.81 (κ=0.626)**；两 judge 同序（GPT-4o 36.5/32.5 = +4.0，DeepSeek 56.0/49.0 = +7.0），正 gap judge-robust。
- **judge endpoint（已有，满足"无 dated snapshot"要求）**：`https://maas-openapi.wanjiedata.com/api/v1/chat/completions`，model `gpt-4o`，`seed=1`，无 client temperature/top-p；appendix 已明确该 endpoint 不暴露 dated snapshot、4 次退避重试、不可解析→保守判错。
- **per-method item-bootstrap CI（已有）**：CoMem 全集 [36.20, 40.58]，KVD [32.48, 36.71]（1,000 resamples seed 1234）。
- **cluster 95% CI**：**[+1.27, +8.32]**（point +4.81，p≈0.004；已填入 appendix，commit `787427b`）
- **修改表/段落**：(1) MemoryLLM 已在 task #83 有 native-chat appendix → 主平均中移出/单列 OOD diagnostic（P0.5/Table 1 联动）；(2) appendix 补 cluster 数值 CI；(3) 明确 no-chat 是控制变量非部署效果。
- **judge artifacts**：verbatim prompt template 已随 eval script 发布（`eval_qcmem_locomo.py`）；⚠️ 待确认是否在许可范围内额外发布 judge inputs/outputs JSONL。

## P0.10 补充隐私与安全边界

- **状态**：`[DONE]`（2026-07-31：Ethics 已加入 residual inversion、membership inference、加密、审计、访问控制和可验证删除。）
- **类型**：仅改 Paper。
- **行动**：在 Ethics/Limitations 中增加 residual-state store 可能受到 inversion、membership inference、unauthorized retrieval 的风险；说明其保护等级不应低于原文本，并讨论删除、访问控制、加密和审计。

**完成记录**

- 修改文件：`paperA/sections/07_ethics.tex`（已含 residual inversion / membership-inference 风险、"保护等级不低于原文本"、encrypt+audit persistent store、access control、verifiable deletion）

## P0.11 补齐 frozen CoMem `j=12` 主表同深度对照

- **状态**：`[DONE]`（2026-07-31 **交付**，.73 8×H20 subagent a62bdb0，80 jobs 全过 0 failure，SCHED_DONE，全 15 RULER cell 过 Iron-Law-2：n=100/8 shard/empty=0/recompute-mismatch=0）。三缺指标已补齐并填入下表；记录 `status/P0_11_FROZEN_J12.md`（commit `aa0ce88`，LiuHanzuo，未 push）。原缺失原因（本地只有 BABILong/LoCoMo；`longbench_results/qcmem_j12` 是 +LoRA 错配置）已由本轮专门 GPU 离线跑覆盖。
- **类型**：额外评测 + 主表更新。
- **目的**：主表当前比较 distilled `j=12` 与 frozen `j=9`，容易混淆 split-depth 与 LoRA adaptation。加入 frozen `j=12` 后可在完全相同深度直接比较 distillation 增益。
- **固定配置**：Qwen3-8B、`resume_j=12`、无 LoRA、`iter_bm25`、top-12、hop-4、`rounds=0`（自动三轮）、chunk-512、BOS sink、`chat_template=False`，评测协议与主表一致。
- **已有结果**：
  - BABILong qa1/qa2/qa5 means：**33.43 / 18.00 / 60.29**；21-cell macro：**37.24**（由 84 个 raw shard JSON 重算）；raw shards：`babilong_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/`；
  - LoCoMo GPT-4o judge：**24.5217**（`n=1986`）；raw：`locomo_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/scores.json`。
- **仍缺结果**：
  1. RULER 3-family/15-cell macro，`n=100/cell`；
  2. LongEval 8k--128k aggregate；
  3. LongBench 6-QA macro F1。
- **完成后修改**：在 `paperA/sections/tab_overview.tex` 增加完整的 `CoMem frozen ($j=12$)` 行，并在正文明确：`j=12` frozen→LoRA 是 adaptation effect；`j=9` frozen 仅保留为较浅 split 的跨 benchmark operating point。

**结果填写**（2026-07-31 交付，n=100/cell，chat=False，iter_bm25/top-12/hop-4/rounds=0/chunk-512/BOS sink/seed=42，Qwen3-8B resume_j=12 frozen）

| Method | RULER | LongEval | LongBench | BABILong | LoCoMo | Raw path |
|---|---:|---:|---:|---:|---:|---|
| frozen `j=12` | **8.01** | **0.2** | **9.96** | **37.24** | **24.52** | `{ruler,longeval,longbench,babilong,locomo}_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/`（diskB .73） |

- **RULER 15-cell macro = 8.01**（120.2/15，string_match）：niah_single_2 36/7/22/18/9、niah_multikey_1 8/3/1/6/4、variable_tracking 2.4/1.2/2.0/0.2/0.4（8k/16k/32k/64k/128k）。
- **LongEval mean(8k–128k) = 0.2%**：per-length 0.0/0.0/1.0/0.0/0.0。
- **LongBench 6-QA macro F1 = 9.96**：narrativeqa 3.85 / qasper 10.67 / hotpotqa 8.85 / 2wikimqa 9.55 / multifieldqa_en 20.76 / musique 6.06。
- **核心结论（adaptation gain 隔离）**：frozen j=12 相对 matched-n flagship CoMem+LoRA(j=12) RULER **96.07** 崩塌到 8.01，甚至低于 CoMem frozen(j=9) RULER **59.41**——**在更深 j=12 split，frozen backbone 无 distillation/LoRA 无法读 memory buffer**。这在完全相同深度隔离出 LoRA adaptation 增益（j=12 frozen→LoRA 是 adaptation effect；j=9 frozen 仅作较浅 split 的跨 benchmark operating point）。
- **tex 待办（main 后续，非本轮）**：在 `paperA/sections/tab_overview.tex` 增 `CoMem frozen ($j=12$)` 行（上述 5 数），正文点明 same-depth adaptation 对照。详见 `status/P0_11_FROZEN_J12.md`。

---

## P0.12 同包、同 LoRA 的 replay 起始层延迟对照

- **状态**：`[DONE]`（2026-08-01 严格验收并完成论文集成）。权威结果 = `bench_results/p0_12_acceptance/`（严格 provenance、分项计时）+ `bench_results/p0_12_naturaltext/`（自然文本一致性）；`bench_results/p0_12_depth_replay/` 为独立复现且结论一致。每臂 3 独立进程 × 20 raw timings + 3 warmups，记录 pack/LoRA hash、168 个挂载模块、完整环境和 component timing；17 个当前 JSON 递归 finite 检查为 0 个异常值。旧 `bench_results/p012/` 已作废，不用于论文。
- **★ 论文采用结果（严格 acceptance bench；med-of-process-medians，n=3 进程×20）**：`qc.read()` **1.08085 s（j=0）→ 0.78707 s（j=12）= 1.373×、降低 27.18%**；upper-transformer forward **1015.86 → 726.66 ms = 1.398×**；final norm + LM head 两臂均约 59 ms；显存均约 17.66 GB。原 `p0_12_depth_replay` 的 **1.374×** 为独立复现，方向和量级一致。
- **⚠️ 表述订正（旧 TODO/status 4 处错误，已在本块改正）**：
  1. j=12 是**预计算并跳过 replay 的 lower 12 层（layer 0–11）**、replay 从 layer 12 起跑上 24 层——**不是"跳过 upper 12 层"**（旧写反了）。
  2. 存储对象 = **residual state h₁₂（depth-12 hidden）**，**不是"lower-12 store KV"**。
  3. 两臂虽同 LoRA、同 packed IDs，但**送入 layer 12 的 hidden 不同**：j=0 整个 pack 过 lower layers 的**全局 causal attention**；j=12 每个 chunk **独立**过 lower layers 再拼接 → **非语义等价对照**。
  4. ∴ 本实验对照 replay 起始层变化下的模型侧 read path；自然文本一致性支持 near-equivalent output，但不能证明 bit identity、benchmark-level equal quality 或端到端 serving speedup。
- **✅ 论文采用 claim**：
  > Holding the adapter and packed token IDs fixed, starting model-side replay at layer 12 rather than layer 0 reduces `qc.read()` latency from 1.081 s to 0.787 s on H20, a **1.373× model-side read-path speedup** across three independent processes; upper-transformer forward improves by **1.398×**, while final norm and LM-head time is unchanged.
  必须同时说明：(a) lower-12 层计算转移到 reusable Write；(b) 不含 retrieval、Write 和 persistent-store I/O；(c) 自然文本输出 near-equivalent 但非 identical，未证明 benchmark-level equal quality；(d) 不是端到端 serving speedup。
- **严格验收项进度（10 项；2026-08-01 acceptance bench 关闭 5 项 + status 文档订正）**：✅ **(1) backbone 权重 hash、(2) LoRA module 枚举、(4) 完整版本、(6) upper-layer forward vs LM-head 分开计时、(7) 输出一致性** —— 由 `scripts/bench_p0_12_acceptance.py`（commit `89914d3`，`.82` H20）跑出，落盘 `bench_results/p0_12_acceptance/{armA,armB}_rep{1,2,3}.json + consistency.json`（已 scp 回 wzc1）；✅ **(9)** `status/P0_12_DEPTH_REPLAY_LATENCY.md` 表述已订正（见该文件顶部 banner）。⏳ 剩余：**(3)** 显式 NaN/finite 检查（bench 以 pack-sha 硬 abort 兜底，但未单独报 finite check）；**(5)** 完整运行命令 + stdout/stderr 留档（launcher `scripts/launch_p0_12_acceptance_82.sh` + `.82:logs/p012acc_*.out` 已存，待汇总入 status）；**(8)** 用 16k JSON 重生成 status 正式汇总；**(10)** 再决定论文集成。
- **✅ Acceptance bench 结果（2026-08-01，`.82` H20，`bench_results/p0_12_acceptance/`）**：
  - **同 pack 校验**：7 进程（6 计时 + 1 一致性）独立重建，`packed_ids_sha256=f7fc7617…` 全部一致且与 depth_replay 权威集 **byte-identical**；`sel_idx=[2,4,7,10,15,17,18,19,20,25,27,29]`、read_len=6657、LoRA sha `dd09cd17…` 一致 ✓（脚本对任何 sha mismatch 硬 abort，off-spec 结果写不出来）。
  - **(1) backbone**：`models/Qwen3-8b-local`、`qwen3`/`Qwen3Config`、36 层、hidden 4096、vocab 151936、`tie_word_embeddings=False`；key-tensor sha 已留（embed_tokens/norm/lm_head/layer0.q_proj/layer35.mlp.down_proj）。
  - **(2) LoRA**：**168 module** = layer 12–35（24 层）× {q,k,v,o,gate,up,down}_proj，adapter=`default`，与 `layers_to_transform=[12..35]` 完全吻合。
  - **(4) 版本**：torch 2.13.0 / CUDA 13.2 / driver 535.247.01 / transformers 5.5.4 / peft 0.19.1 / git `21c124e` / py3.14.6 / NVIDIA H20。
  - **(6) read-path 拆分**（med-of-process-medians，n=3×20）：read_s j0=1080.85 ms → j12=787.07 ms = **1.373×**（复现权威 1.374×）。**加速几乎全部来自 upper-layer transformer forward：1015.86 → 726.66 ms = 1.398×；final-norm+LM-head 两臂都 ~59 ms（depth-independent，同 `[1,6657,V]` 投影，符合预期）。** → 现可精确表述"加速源于上层 transformer forward 计算减少，与 LM-head 无关"，但仍**不得**升级为无条件 "kernel speedup"（拆分证明加速在 forward，但仍无同质量证明）。
  - **(7) 输出一致性（j=0 vs j=12，同 pack）**：末位 next-token cosine **0.9784 / top-1 一致 / KL 0.0407**；greedy 16 步 **top-1 一致率 93.75%（15/16，仅第 1 步不同）**、cos_mean 0.9309、KL_mean 0.0801；query-tail 512 位 cos_mean 0.9551、top-1 一致率 59.18%、KL ~0.048。**判定：高度一致但非 bit-identical**——两臂按设计送入不同 layer-12 hidden，但输出近等价（同 next token、15/16 decode 一致、logit cos 0.93–0.98、KL 仅 0.04–0.08 nats）→ **正面支持（非推翻）caveat (c)：replay 起始层改到 12 只扰动、不实质改变输出**。⚠️ **诚实 caveat**：一致性测于**合成随机-token pack**（满足 pack-sha 一致所必需），非自然文本；query-tail top-1 59% 是随机位 argmax 噪声，末位/decode 一致才是有效信号。**升级为无条件 "same-quality" 前仍需一次自然文本（不同 sha）一致性检查——✅ **已补，见下条**。
- **✅ 自然文本一致性补充（2026-08-01，`.82` GPU0，`bench_results/p0_12_naturaltext/`，bench `scripts/bench_p0_12_naturaltext_consistency.py` commit `8e2c728`）——把验收项 7 从"仅合成 pack"补齐到"含自然文本"**：pack 源 = 真实 wikitext（`data/rmt_train_wikitext.jsonl`），3 段不同文档，几何同 acceptance（6657-tok、top-12/31 ctx chunk），每段两臂 `packed_ids` sha 一致（doc0 `79b5ab5e…` / doc1 `15ef601e…` / doc2 `7223d953…`，均 ≠ 合成 `f7fc7617…`）。**j=0 vs j=12 自然文本一致性（3 段聚合）**：末位 next-token cos **0.9758 / top-1 一致 3/3（100%）** / KL 0.445（doc2 单点高熵拉高均值，top-1 仍一致）；query-tail 512 位 cos 0.9773、**top-1 一致率 0.848**、KL ~0.134；greedy 16 步 **top-1 一致率 0.854**、cos 0.965、KL 0.131。**判定：自然文本上仍高度一致、非 bit-identical，且 teacher-forced top-1 比合成 pack 更强（query-tail 0.59→0.85——自然文本 argmax 分布更尖锐、对 layer-12 输入差异更鲁棒）。** → **支持把 P0.12 表述从严格 "read-path 对照" 适度软化为 "near-equivalent output / 近同质量 read-path 对照"**，但必须保留 "near-same 非 identical"（KL 非零、偶发高熵位发散如 doc2）。至此验收项 7 完全闭环（合成 + 自然文本双证据）。
- **论文集成**：`[DONE]`。摘要、引言、实验、结论、限制及复现附录均已更新；详细分项表置于附录以保持 ACL 八页正文。正文使用 **1.373× model-side read-path** 和 **1.398× upper-transformer-forward**，不使用无条件 pure-depth/kernel/end-to-end 或 equal-quality 表述。
- **类型**：低成本系统消融；无需训练，仅做同平台延迟/显存评测。
- **目的**：在同 pack、同 LoRA 条件下对照 replay 起始层从 0→12 的 model-side read-path 差异；两臂进入 layer 12 的 hidden 构造不同，因此配套报告自然文本输出一致性而不声称 bit identity 或 benchmark-level equal quality。
- **核心对照**：
  1. `j=0 + flagship rank-32 LoRA`；
  2. `j=12 + 同一 flagship rank-32 LoRA`；
  3. 保留现有 `j=0 no-LoRA` 与 `j=12 + LoRA` 作为部署 operating points，但不与本实验混作单因素消融。
- **固定配置**：Qwen/Qwen3-8B 同一 revision；同一 LoRA artifact/SHA-256；同一 top-12 retrieved IDs、完全相同 pack 顺序与约 6,657-token read；chunk-512、BOS sink、bf16/SDPA；同一 H20；固定 query、生成长度和 KV-cache 策略；预热后至少 20 次计时，报告 median、p10/p90 或 bootstrap 95% CI。
- **必须记录**：
  - query latency：总时间及可拆分的 upper-layer forward / LM-head / decode；
  - peak GPU memory；
  - pack token 数与 retrieved ID 哈希，证明两臂输入一致；
  - LoRA 实际挂载层和参数哈希，证明两臂 adapter 一致；
  - 若 LoRA 无法语义一致地挂到 `j=0` 全层路径，必须标记为实现不可比，不能强行报告为 pure-depth control。
- **验收标准**：两臂除 replay 起始层外完全一致；三次独立进程重复的 median 差异方向一致；无 OOM/NaN；原始 per-run timing JSON、环境和 GPU 信息齐全。
- **完成后修改**：已更新 `paperA/sections/tab_pareto.tex`、`05_experiments.tex`、`00_abstract.tex`、`06_conclusion.tex` 和复现附录。论文报告 `1.373×` model-side `qc.read()` path speedup 与 `1.398×` upper-transformer-forward speedup，同时明确两臂 hidden states 非同一计算图、输出近似而非严格等价，且不包含 retrieval、store/query write 或 persistent-store I/O。

---

## P0.13 同 pack、同 LoRA、同 examples 的 benchmark quality--latency 闭环

- **状态**：`[DONE — QUALITY-LATENCY TRADEOFF]`（2026-08-02，coder a0853dad 完成于 .82；本地 commit `29243d9` 未 push，runtime manifest git_commit `21c124e`；结果由 MAIN 复核 JSON 后回填，见下方 RESULTS）。这是投稿前**最后一个不可由现有数据重算、文案收缩或删除次要结果替代的模型评测 run**，至此不再计划新增必要模型实验；其余开放项均为现有数据分析或可选增强。
- **核心问题**：P0.12 已在同 pack、同 LoRA 下测得 `j=0: 1.081 s`、`j=12: 0.787 s`（`1.373×`），但只用三个自然文本 pack 检查输出近似，尚未建立 benchmark-level quality 差异。P0.13 必须回答：在相同 adapter、相同检索内容和完全相同测试样本上，这一 read-path 差异对应多少真实任务质量变化？
- **必须比较的两臂**：
  1. `j=0 + flagship rank-32 LoRA`：相同 top-12 pack，全 36 层 replay；
  2. `j=12 + 同一 LoRA`：相同 pack，从缓存 residual state `h12` replay 上 24 层。
  可选第三臂：`full-context + same LoRA`，仅当同一 adapter 可语义合理地用于 full-context 路径时加入；否则记录为不适用，不能强行解释。
- **最低任务范围**：优先完整 RULER Cohort B 15 cells（`niah_single_3`、`niah_multikey_1`、`variable_tracking`，8k/16k/32k/64k/128k，`n=100`）；若成本阻塞，可先跑预注册代表子集：三任务 × 16k/32k/128k，每 cell `n>=100`，但只有完整 15-cell 才可更新论文 headline。建议再加 BABILong qa1/qa2/qa5；LoCoMo answerable subset 为可选增强而非验收必需。
- **严格固定项**：同 Qwen/Qwen3-8B revision/hash；同 flagship LoRA SHA-256 和 168 个挂载 module；同 example IDs、prompt、gold、decode 参数；每例相同 retrieved chunk IDs、顺序、pack token IDs/hash 和 read length；同 H20、bf16/SDPA、KV-cache、batch、生成长度与 scorer。除 `resume_j=0/12` 及其必然的 hidden-state 构造外不得改变其他因素。
- **每例必须保存**：`example_id/task/length/gold/prediction/score`，retrieved IDs、pack hash、LoRA hash，Write/retrieval/`qc.read()`/decode/total latency，peak memory，NaN/finite flag；保存完整运行命令、环境、git commit 和 stdout/stderr。
- **统计与报告**：
  - 分别报告两臂 cell score、15-cell macro 和差值；
  - 对同一 examples 做 paired bootstrap 95% CI；离散正确性同时做 McNemar 或 exact paired test；
  - 报告 per-example prediction agreement、next-token/decode agreement 与失败类别；
  - 延迟至少 3 个独立进程，每进程 3 warmups + 20 timings，报告 median、p10/p90；
  - 质量与延迟必须来自同一配置 manifest，不能将 P0.12 timing 与不同 run 的 quality 拼接成因果结论。
- **验收标准**：两臂 manifest 除 replay start 外一致；所有 examples 和 pack hashes 一一配对；无 OOM/NaN；三进程延迟方向一致；paired 统计可复算；原始 predictions/timings 齐全。质量不要求预设“不下降”阈值——无论结果正负都必须报告。
- **论文决策规则**：
  - 若 quality 差异小且 CI 支持预注册容忍区间（建议 RULER macro 非劣界 `-1.0` point），可写“在该 benchmark/support 上近似保质的 `1.373×` model-side read-path reduction”；
  - 若质量显著下降，则报告完整 quality--latency trade-off，不使用 quality-preserving/pure-depth 表述；
  - 若 `j=0 + same LoRA` 实现语义无效或 pack 无法严格配对，则 P0.13 判失败，保留当前 P0.12 caveat，不用 workaround。
- **完成后修改**：更新 `tab_replay_latency.tex` 或新增 paired quality--latency 表、`05_experiments.tex`、`00_abstract.tex`、`06_conclusion.tex`、`07_limitations.tex` 和复现附录；随后做一次最终 ARR 盲审。

### RESULTS — DONE 2026-08-02（.82 8×H20 diskB，torch-base 同 P0.12 环境；MAIN 已核对 `summary.json`/`stats.json`/`latency.json`）

**两臂 headline（15 cells，n=100/cell，n_paired=1500，packs_paired_1to1=True，OOM=0，non-finite=0）：**
- Arm A（`resume_j=0` + flagship rank-32 LoRA，全 36 层 replay）：**macro = 99.19**
- Arm B（`resume_j=12` + 同一 LoRA，从缓存 residual h₁₂ replay 上 24 层）：**macro = 96.07**
- **macro diff (A−B) = +3.12 pp**，paired bootstrap 95% CI **[2.36, 3.93]**（10k resamples）
- **McNemar** exact two-sided **p = 8.79e-24**（A-only-correct b=83, B-only-correct c=1, both=1404, neither=12）
- per-example agreement：prediction-exact **2.8%**，first-token **90.8%**，first-token cosine **0.977**，decode top-1 **42.1%**

**Per-cell（armA / armB，diff = A−B，n=100 each）：**

| Task | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| niah_single_3 | 100.0/100.0 (+0.0) | 97.0/91.0 (+6.0) | 97.0/97.0 (+0.0) | 99.0/98.0 (+1.0) | 98.0/98.0 (+0.0) |
| niah_multikey_1 | 100.0/94.0 (+6.0) | 100.0/91.0 (+9.0) | 100.0/99.0 (+1.0) | 97.0/90.0 (+7.0) | 100.0/93.0 (+7.0) |
| variable_tracking | 99.8/96.2 (+3.6) | 100.0/98.0 (+2.0) | 100.0/98.2 (+1.8) | 100.0/98.6 (+1.4) | 100.0/99.0 (+1.0) |

质量差集中在 **niah_multikey_1**（4-distractor-key 检索，全长档 +1..+9）与较轻的 VT；niah_single_3 两臂基本持平 → B 丢失的正是 distractor 干扰下的检索消歧，缓存 h₁₂ 跳过 lower-12 层 replay 会削弱它。

**Latency（3 独立进程 × 3 warmup + 20 timings，固定 16k niah_single_3 pack）：**
- Arm A read median = **931.9 ms**（p10 931.6 / p90 942.0）
- Arm B read median = **664.4 ms**（p10 663.8 / p90 667.1）
- **read speedup A/B = 1.4027×**（三进程各 1.403/1.402/1.404，方向一致，B always faster；与 P0.12 的 ~1.373× 一致）
- total-decode median ~相等（~2.76–2.86 s 两臂）——read 是 16k 端到端的小份额，故加速是 read-phase（prefill/QC-read）效应，非总 wall-clock。

**Provenance（strict_fixes）：** lora_sha `dd09cd17…`（`lora_sha_match=true`）、168 LoRA modules、layers-to-transform [12..35]、selector=iter_bm25 topk=12 iter_hop_topk=4 sink=bos chunk_size=512 chat=False enable_thinking=False bf16 sdpa；backbone key-tensor sha `7a478390…`（== P0.12）；`abort_reasons=[]`。产物在 .82 diskB `bench_results/p0_13_quality_latency/`（`manifest.json`/`summary.json`/`stats.json`/`latency.json` + `quality/` 120 files + `latency/` 3 proc files）。

**论文决策规则 → QUALITY-LATENCY TRADEOFF 分支（非"近似保质"分支）：** macro diff +3.12 pp、CI [2.36,3.93] 整段在 0 以上、且远超预注册非劣界 −1.0 pt → B **不满足非劣**，显著更弱（McNemar p≈9e-24）。故论文必须报告完整 quality--latency trade-off：Arm A（全 replay）为精度最优配置；Arm B（`resume_j=12`）以 **~3 pp** macro 精度（集中于 distractor-heavy multi-key 检索）换取一致的 **1.40× read-phase 加速**。**不得**使用 quality-preserving / pure-depth / 端到端加速 表述。

> `.tex` 集成已完成（2026-08-02）：已更新 paired quality--latency 表，以及 abstract/introduction/experiments/conclusion/limitations/reproducibility appendix；统一采用“1.403× read-phase speedup at 3.12-pp RULER cost”，并明确非 quality-preserving、非端到端加速。

## P0.14 InfiniteBench / PG-19 污染审计（不需模型重跑）

- **状态**：`[DONE-NEGATIVE — NO MODEL RUN]`（2026-08-02，CPU-only，未用 GPU；coder commit `362a22f`，未 push）。审计产物位于 `bench_results/p0_14_contamination/`（`README.md` / `audit_summary.json` / `match_list.json` / `per_record_verdict.jsonl`(580 行) / `threshold_sensitivity.json` / `clean_subset_ids.json` / `data_manifest.json` / `verification.txt`）。
- **方法（三种）**：(a) title/author/PG-ID 交集 = **不可计算**（InfiniteBench 记录与 `data/pg19_train.jsonl` 均无 title/author/PG-ID 元数据）；(b) book-level 精确哈希 = **0**（eval 为匿名化整书，train 为无边界拼接行，结构上不可能整书匹配）；(c) **13-gram containment MinHash sketch**（lowercase+去标点、xxh64 seed0、1/32 bottom-hash 下采样、train sketch 59.1M unique hashes）= **决定性**。⚠️ 修正：首轮按 PG-19 原始行分词，但 PG-19 硬换行 ~13 词/行 → 行内 13-gram 几乎捕获不到 → 假 ~0；已 reflow 成连续 token 流匹配 eval 侧，双向逐字复核（0.999 书的叙述能在 PG-19 找到、0.000 书找不到；QA#0 = Woolf《To the Lighthouse》1927，晚于 PG-19 1919 截止 → 正确判 CLEAN 0.000）。
- **结果**：86 本 unique 书（580 records = QA 351 + choice 229）。containment 分布**强双峰**：31 本 <0.10（30 本 ≈0.000），0.18–0.60 空档，54 本 ≥0.60。故污染率对阈值稳健：

| 阈值 cut | contaminated 书 | 书占比 | contaminated records | record 占比 | QA below cut | choice below cut |
|---:|---:|---:|---:|---:|---:|---:|
| ≥0.80（保守 headline，**低估**） | 24 | 27.9% | 163 | **28.1%** | 249 | 168 |
| 任意 ∈[0.20, 0.60]（诚实 headline） | 54 | **62.8%** | 387 | **66.7%** | 113 | 80 |
| ≥0.90 | 9 | 10.5% | 60 | 10.3% | 319 | 201 |

  ≥0.80 低估是因为 InfiniteBench **匿名化人名**（"Mrs Ramsay"→"Mrs Bronwyn"）打断了部分 13-gram，把本在训练集中的书从 1.0 拉到 0.60–0.92。**诚实结论：~63% 的书 / ~67% 的 eval records 命中 PG-19 训练集（flagship LoRA `outputs/qcmem_distill_qwen_j12_r32_4k` 仅在 `data/pg19_train.jsonl` 上蒸馏，已由 `distill_args.json` 核实）。**
- **处理**：clean-subset 重算（CLEAN<0.10 → QA 113 / choice 76）**本节点不可行**——四臂 per-example predictions 在禁访 GPU 节点 `.73`（zwfy6），wzc1 无副本。`scripts/recompute_p0_14_clean_subset.py` 已就绪（复用 `eval_qcmem_infbench.py` 打分器，无需模型 run），predictions 一旦转来即可跑。**结论/建议：鉴于 ~63–67% 污染且本节点 predictions 不可恢复 → WITHDRAW/RELABEL `tab:infbench`（Book-QA F1 6.06 / Book-choice acc 17.47），仅保留其作为 bounded-read coverage / memory-stress 的定位，删除 QA-F1 / choice-acc 质量结论（或在 predictions 转来后于 GPU 节点用 clean subset 重算再决定）。** ⚠️ `.tex` 集成非本任务范围（MAIN 不改 `.tex`）；该负面结果不触发模型重跑。

## P0.15 提交前可审计性、读长口径与匿名化收口

- **状态**：`[DONE — TEX INTEGRATED 2026-08-02]`。审计报告 `status/P0_15_AUDIT.md`（commit `6bfcc55`）的唯一必改和四项可选收口均已执行：judge 域名已泛化；内部 adapter 路径改为 SHA/config 引用；nominal 6,657 与实测 6.2--6.5k 已统一；附录给出 `1+12×512+512` 分解；`roughly 6,657` 已修正。j=0 分项及 cohort 隔离规则保留在审计报告，未把 Cohort B 混入 Cohort A。
- **类型**：现有 artifact 重算 + Paper-only 审计；不得启动训练或完整模型推理。
- **目的**：消除 reviewer 仍可合理提出的三项复现/呈现问题，而不制造新的 cohort 混用。

### A. j=0 cell-level 分解

- 从现有 raw predictions/scorers 补齐 j=0 的 LongEval、LongBench、BABILong、LoCoMo 分项/类别结果，并在附录给出 raw path、sample count、聚合方式。
- RULER 的 paired j=0 数据是 **Cohort B (`niah_single_3`)**。不得把它直接塞进 Cohort-A Table 13 或与 `niah_single_2` 行拼接；应新增 Cohort-B paired 表/附表，复用 P0.13 的 15-cell结果和 pack-pairing manifest。
- 若某项 raw predictions 不在当前节点，标记 `[BLOCKED-DATA]` 并列出远端路径；不得从 macro 反推 cell。

### B. 统一读长术语

- 全文统一定义：**名义满 pack = 6,657 tokens**（BOS 1 + top-12×512 + query 最多 512）；**RULER/store-scaling 实测均值 = 6.2--6.5k**，来自短尾 chunk/query 或样本实际长度。
- `approximately 6.5k` 仅作为简写；表格必须区分 nominal cap、per-example actual length 与 aggregate mean/range。
- 静态扫描 `00_abstract.tex`、`04_methodology.tex`、`05_experiments.tex`、`08_appendix.tex`、所有 table captions，确认不存在把 6,657 与 6.2--6.5k 写成矛盾配置的句子。

### C. 匿名与可复现性扫描

- 从匿名 review PDF 移除/泛化真实第三方 judge 域名、内部 `outputs/...` checkpoint 路径、节点/IP/用户名和非匿名 artifact 痕迹；保留 model name、seed、scoring policy、weight hash、匿名 artifact 相对路径和“无 dated snapshot”限制。
- 对 PDF 文本和 metadata 执行 identity scan；记录命令、命中项、人工判定和最终零敏感命中结果。
- 不删除复现信息的权威内部记录；完整端点和内部路径保留在非投稿 status/raw manifest 中。

**验收交付**

- 修改文件/表：`05_experiments.tex`、`08_appendix.tex`、`08_statistics_appendix.tex`。
- raw 重算命令与结果：见 `status/P0_15_AUDIT.md` Part A；无模型 forward，全部现有 predictions CPU 重算。
- nominal/actual read-length 扫描：PASS；nominal cap 6,657，实测约 6.2--6.5k，无配置矛盾。
- anonymity scan：真实 judge 域名和内部 adapter 路径已从投稿源移除；节点/IP/用户名/绝对路径保持零命中。
- PDF 编译与页数检查：PASS；最终 P2.4 集成后需再执行一次。

## P0.16 E0：document-contextual Write control（先做，零训练）

- **状态**：`[DONE — 2026-08-03，.104 8×H20 diskB，harness commit 2ae5917（author LiuHanzuo，未 push），MAIN 复核 JSON]`
- **优先级**：当前最高；结果决定 P0.17/P0.18/P1.10 的方向。
- **结果**（`bench_results/p0_16_e0_write_control/{summary,stats}.json`，n_paired=200，2 cells）：
  - macro：**A(full replay)=100.0 / C(continuous-pack oracle)=100.0 / E0(doc-contextual Write)=100.0 / B(chunk-local deployable)=92.5**。
  - paired：**A−E0=+0.0 CI[0,0] McNemar p=1；C−E0=+0.0 CI[0,0] p=1**（E0 与 A/C 逐位一致，both=200）；**E0−B=A−B=C−B=+7.5pp CI[4.0,11.5] McNemar b=15/c=0 p=6.1e-5**。
  - per-cell：8k B=94.0（E0−B=+6.0）、16k B=91.0（E0−B=+9.0）。fail-closed：`packs_paired_1to1=True p013_sha_match=True oom=0 nonfinite=0`；前置 e0_h12_sanity `max_abs=0.000e+00`（tol=5e-2）PASS → E0 下-12 前向与 stock 逐位一致。agreement A_vs_E0 first_token=0.855 cos=0.9963、B_vs_E0 first_token=0.91。
- **裁决**（命中上文“E0 接近 j=0/oracle”规则）：E0 ≈ A/C 且 ≫ B → deployable A-B gap 全部来自 chunk 独立写入缺少文档上下文，Write→Read 重定位近乎无损 → **优先 P0.17（E2 overlap Write）**，已启动（见下）。
- **问题**：现有 continuous-pack oracle 是 query-dependent：它对每个 selected pack 连续运行 layers `[0:12)`，不能作为跨 query 复用的 Write。它证明上层 continuation 无损，但没有区分“独立 chunk 缺少文档上下文”与“Write/Read 坐标重映射”造成的误差。
- **实验臂**（同 examples、selector、top-12、pack 顺序、旗舰 LoRA、Read 实现）：
  1. `j=0` full-depth selected-pack replay；
  2. 现有 continuous-pack `h12` oracle；
  3. **E0 document-contextual Write**：对完整文档按原始 causal 顺序运行 layers `[0:12)`，逐 token 保存 query-independent `h12`，再按 BM25 命中的 chunk 切片组成 store pack，Read 仍从 layer 12 开始。
- **最小协议**：`niah_multikey_1` 8k/16k，`n=100/cell`，复用 P1.7 的 200 个 paired examples；若 E0 与任一端点差异明确，再扩展 Cohort-B 15 cells。
- **必须记录**：原始文档 token 坐标、selected chunk IDs、切片边界、pack hash、LoRA hash、Write/Read RoPE position IDs、逐例 prediction、OOM/finite、Write/Read latency和 peak memory。
- **解释规则**：
  - E0 接近 `j=0/oracle`：主要损失来自 chunk 独立写入缺少文档上下文，优先做 P0.17；
  - E0 接近 deployable `j=12`：主要损失更可能来自 Write→Read 重新定位，暂停 overlap 与 write-LoRA，优先做 P0.18；
  - E0 居中：两类因素均存在，完整执行 P0.17+P0.18。
- **措辞限制**：称为“可跨 query 复用的 document-contextual control”，不称严格 upper bound；其 Write 为 `O(L)`，长文位置扩展和文档更新成本必须报告。
- **验收**：三臂严格 paired；E0 state 在短文档上与 stock lower-12 full-document hidden 数值核验；无 cohort 混用；无论结果正负均进入机制表。

## P0.17 E2：overlapping chunk Write（条件执行，零训练）

- **状态**：`[DONE — 2026-08-03，.104 8×H20，task #136，harness commit 873deb2 + notes be2ae80（paperA/P0_17_E2_NOTES.md，author LiuHanzuo，未 push）；预注册主目标 ≥97.0 达成]`
- **结果**（n=200 paired，niah_multikey_1 {8k,16k}，真 Qwen3-8B + 旗舰 LoRA，6 臂全过 fail-closed gate）：deployable multikey pooled **92.5（w0=Arm B）→ 99.0（best w=128）**。E2_w32=98.5（+6.0 [3.0,9.5] p=4.9e-4 b=12/c=0）、E2_w64=98.5（+6.0，同）、E2_w128=99.0（+6.5 [3.5,10.0] p=2.4e-4 b=13/c=0）；E0 天花板=100.0。每个宽度显著超 deployable baseline，回收 ~80–87% 的 E0−B document-context gap，距 E0 残差 −1.0~−1.5pp（p≥0.25 不显著）。成本：一次性 lower-12 Write FLOPs +5.7%（w32）~+22.9%（w128）；persistent bytes/token + Read + decode 与 w0 完全相同。gates 全 PASS（LoRA sha dd09cd17…168mod layers[12..35]；e2_sanity 两项 max_abs=0.000e+00 证 w0≡Arm B、E0-lower12≡stock；packs_paired_1to1=True；pack sha 200/200==P1.7；oom=0 nonfinite=0）。measured per-arm latency 微bench @.104 GPU0（task #136 收尾）。
- **裁决**：确认 P0.16 归因——deployable gap 来自 chunk-local Write 缺文档上下文，非 Read 重定位。E2 是**可部署修复**（w=32 已近最优且最省），候选并入 Cohort-B。
- **触发条件**：P0.16 显示 document context 能明显回收质量；否则不启动。**→ 已满足**（E0=A=C=100.0，B=92.5，document context 恰好补齐 deployable gap）。
- **方法**：写入 chunk 时前置左上下文 `w∈{32,64,128}`，运行 lower 12 层后丢弃 prefix states，只存原 512-token chunk 的 `h12`。persistent bytes/token、Read pack 和 Read 计算保持不变，仅增加一次性 Write 计算。
- **最小协议**：与 P0.16 完全相同的 multikey 8k/16k paired 200 examples，另含 `w=0` deployable baseline 和 E0 control。
- **报告**：accuracy、paired bootstrap CI、McNemar、Write latency/peak、相对 `w=0` 的额外 FLOPs/token；禁止只报最优 `w`，四个宽度全部保留。
- **成功标准**：预注册主目标为 multikey pooled `92.5 → ≥97.0`，且 store/Read 成本不变；若仅有小幅收益，也按负面/边界结果报告。
- **后续**：达到目标后扩展 Cohort-B 15 cells，并将其作为新的 deployable Write 变体；未达到则停止扩大评测，转 P0.18。

## P0.18 E4：Write 上下文与位置重映射二因素拆解（零训练）

- **状态**：`[HARNESS READY，GPU 排队 — 2026-08-03，5-臂 2×2 拆解 harness（scripts/eval_p018_e4_2x2_writecontrol.py + scripts/_run_p018_e4_8gpu.sh），workflow wg28ofr1v 构建，commit c32a2c9（author LiuHanzuo，未 push）。CPU 全通过（py_compile + import + aggregate path，transformers 5.14.1）。臂：A=j0 / BB=Arm B（chunk-local,local-pos）/ E0=P0.16 E0（doc-ctx,local-pos）/ X=(chunk-local,doc-origin-pos 新)/ Y=(doc-ctx,doc-origin-pos 新)；单因素 control BB→E0（factor1 上下文）、BB→X（factor2 位置）、E0→Y、X→Y、joint BB→Y + 交互残差；A/BB/E0 verbatim 走 p016/p017 → 与 headline 行 bit-identical。fail-closed 门：manifest(exit3, LoRA sha dd09cd17/168 mod/layers[12..35]) + pos_sanity(exit4, doc-origin read==read_prefill tol5e-2) + quality --verify（doc-ctx-h12==stock-lower12 + pos-plumbing assert）。GPU run 排队 .104（P0.20 之后；当前无空闲节点，.55 UNAVAILABLE）。P1.10 解锁依赖本实验裁决。]`
- **目的**：拆开当前 Limitations 中混在一起的两个因素：lower layers 是否看见跨 chunk 文档上下文，以及 cached states 从文档坐标移到 selected-pack 坐标的 RoPE 不一致。
- **设计**：构造可验证的 `2×2` diagnostic：`chunk-local vs document-contextual lower-layer attention` × `local/reset vs document-origin position IDs`。若某臂因 Qwen RoPE/cache API 无法严格实现，必须记录数学定义、失败原因，并至少完成能单独改变一个因素的两条 control。
- **协议**：先用 P0.16 的 paired 200 examples；保存 layer-12 state cosine/L2、最终 logits KL/top-1 agreement和任务 accuracy，不只报最终分数。
- **判定**：明确哪一因素解释主要差距，并据此决定 P1.10 应训练 writer representation、学习位置接口，还是不再训练。
- **验收**：每臂只改变声明的一个因素；position IDs、attention mask、segment mapping 和 state slicing 有 fail-closed assertions。

## P0.19 Retrieval recall 与 in-pack readout 分解（零训练/现有数据优先）

- **状态**：`[DONE（CPU 重算）— 2026-08-03，task #131，from existing predictions/manifests；记录 paperA/P0_19_decomp_NOTES.md，commit b9dc847（author LiuHanzuo，未 push）。RULER paired GPU leg 见 #135 = 低优先/可选（seed-pairing bug 已修 d1e1389；无 paper table 依赖那批 cross-run 配对），仅当该 decomposition 进正文才需重跑 paired j0/j12 RULER。]`
- **目的**：回答为何 CoMem 相对 `j=0` 在 RULER 仅小幅下降，却在 LongEval/BABILong 某些任务下降更大；区分 selector miss 与 cached-state readout failure。
- **任务**：至少覆盖 RULER multikey、BABILong qa1/qa2 和 LongEval；对每个样本标注 gold support 是否进入 top-12 pack，并分别报告：
  1. retrieval recall@12；
  2. `j=0` 在 recall-hit 子集的 answer accuracy；
  3. `j=12` 在同一 hit 子集的 accuracy；
  4. recall-miss 子集表现。
- **实现顺序**：优先从现有 selected chunk manifests、gold facts 和逐例 predictions CPU 重算；只有缺少 support mapping 时才补模型 eval，且必须复用相同 examples/pack。
- **验收**：给出逐任务 decomposition、paired CI 和 raw sample IDs；不得用 answer accuracy 反推 recall。

## P0.20 Equal-latency retrieval-budget frontier：text RAG vs CoMem（先测）

- **状态**：`[RUNNING 阶段A — 2026-08-03 起，task #137，agent ac5056a0 在 LOCAL 建 harness（复用 config#2 j0-RAG/P0.13/P1.7/P0.2 资产）→ rsync .104 → 8×H20 跑 BM25 equal-latency k-sweep；用户指定最高优先级。阶段B（dense）绑定 P1.9。]`
- **核心问题**：固定在线延迟预算时，CoMem 能否利用省下的 lower-layer 计算读取更多 evidence，并在质量上达到或超过 raw-text RAG；这比固定 `topk=12` 只报告约 `1.4×` Read 加速更直接地检验 CoMem 的实际价值。
- **符号与主锚点**：`j` 仅表示深度，旗舰固定 `j=12`；`k` 表示 retrieved chunks。主比较固定 `CoMem(j=12,k=12)`，寻找最大的整数 `k_RAG*` 使 text RAG 的部署配置在线延迟与其匹配（预注册容差 `±5%`），再比较两者质量。
- **两条路径**：
  1. **Text RAG**：同一 selector 排名的 top-`k` 原文 chunk，按相同顺序组 pack，Qwen3-8B 从 layer 0 完整读取；
  2. **CoMem**：同一 selector 排名的 top-`k` chunk IDs，fetch 对应 persistent `h12`，使用旗舰 LoRA 从 layer 12 续算。
- **阶段 A（立即执行）**：复用现有 iterative token-ID BM25、相同 chunking/index/examples，扫描 `k∈{2,4,6,8,10,12,14,16,20,24}`，先闭合统一 harness 和 BM25 equal-latency 结论。
- **阶段 B（正文主结果候选）**：P1.9 固定 BGE/E5 retriever 后，用完全相同协议重跑；dense text-RAG 与 dense CoMem 必须共享同一排序列表、chunk IDs 和顺序。BM25 结果保留为 selector ablation，不删除负面结果。
- **延迟主口径**：同节点、同 GPU、同进程配置测 TTFT，必须包含 `query encoding/selection + lookup + raw/residual fetch + H2D + model prefill/Read`；另单列 model-only Read，禁止把 H20/L20A 或不同 harness 数字直接相减。GPU-resident 与 CPU-pinned store 分开报告；若 external tier 无法完成，必须明确限制。
- **校准与冻结**：只在独立 calibration split 上根据 latency 选择 `k_RAG*`，不得查看质量后挑 `k`；冻结 `k_RAG*` 后评测全部任务。若没有整数点落入 `±5%`，报告两侧相邻点并仅对 latency 插值，质量不插值。
- **质量任务**：主任务为 BABILong qa1/qa2、LongEval、LoCoMo，附 RULER multikey；不得只用 lexical single-needle。每个任务使用双方相同 sample IDs、query、排序前缀和生成设置。
- **次锚点（同一 sweep 顺带报告）**：固定 `text RAG(k=12)`，寻找最大 `k_CoMem*` 满足相同 `±5%` 延迟预算，检验 CoMem 在标准 top-12 RAG 预算下能读取多少额外 evidence。
- **统计与输出**：每个 latency 点 warmup 后至少 20 次，至少 3 个独立进程，报告 median/p95；质量报告 paired bootstrap CI 与逐例 predictions。主图为 quality--TTFT frontier，主表同时给 `k`、read tokens、recall@k、accuracy/Judge、fetch、model Read、TTFT 和 peak memory。
- **成功标准**：主锚点下 CoMem 的质量不低于 latency-matched text RAG，且至少一个非 lexical benchmark 显示显著或稳定优势；若优势只存在于 model-only、加入 residual fetch 后消失，必须限定为 compute-side result。若 CoMem 即使读取更多 chunks 仍更差，则判定 bottleneck 为 cached-state readout，转回 P0.17/P0.18/P1.10，不包装成正面 Pareto。
- **论文影响**：成功时用“equal online latency 下读取更多 evidence/获得更高质量”替代单独强调固定-`k` 的 `1.4×`；无论结果如何，`64.9×` 仍只能归因于 bounded selection + depth reuse 的整体 operating point。

---

# P1：显著增强，通常需要实验或分析

## P1.1 大规模外部存储 / distractor scaling

- **状态**：`[DONE]`（2026-08-01，`.82` 8×H20 与 dllm sibling co-reside；harness `scripts/eval_p1_scaling.py`；完整记录 `status/P1_1_P1_5_SCALING_RESULTS.md`，commit `84720fe`）
- **原状态**：`[TODO]`
- **类型**：额外评测 + 系统实验。
- **目的**：验证 “unbounded-context” 更准确地是 bounded-read over extensible store，并测量 store 增大时的 retrieval recall 与 latency。
- **设计**：固定相关证据和 read budget，store 从 `128k → 256k → 1M → 4M+ tokens` 增加 distractors；报告 recall@k、answer score、retrieval latency、index size、transfer/read latency。
- **任务**：至少一个 single evidence、一个 multi-hop/distributed evidence；加入所需证据数超过 top-12 的压力测试。

**结果填写（single-evidence NIAH, E=1；token-space gold-chunk ground truth；iter_bm25 top-12, chunk512, j=12, chat=False）**

| Store size | Evidence count | Recall@k | Score | Retrieval ms | Index GB | Read tokens | Raw path |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128k | 1 | **1.000** | 100.0 | 80 | 1.07 | 6211 | `ruler_results/p1_scaling_{retrieval,full}/` |
| 256k | 1 | **1.000** | 100.0 | 164 | 2.15 | 6209 | ″ |
| 512k | 1 | **1.000** | 100.0 | 328 | 4.29 | 6209 | ″ |
| 1M | 1 | **1.000** | 100.0 | 697 | 8.59 | 6209 | ″ |
| 2M | 1 | **1.000** | 90.0 | 1407 | 17.18 | 6211 | ″ |
| 4M+ | 1 | **1.000** | 100.0 | 2852 | 34.36 | 6208 | ″ |

**核心结论（P1.1）**
- **recall + answer score 对 store 大小不变**（1.000 / ~100，128k→4M）；**read tokens 恒定 ~6210**（O(1) read，pack 不随 store 增长）。2M 的 90 是 n=10 抽样 miss（recall 仍 1.000）。
- 只有 **store build（`h_j` index 1.07→34.4 GB）** 与 **lexical BM25 retrieval（80 ms→2.85 s）** 随 store 线性增；BM25 倒排索引仅 0.5–16.8 MB，`h_j` index 从不常驻 GPU → **单张 H20 寻址 4M-token store**。
- **压力律 recall@k = min(1, top-k/E)**（VT chain-length sweep @128k，E=4/5/8/12→1.000；E=16→0.750；E=24→0.500；E=32→0.375，精确）——这是 “unbounded context = bounded read over extensible store” 的诚实边界：store 可无界，但答案完整性仅在**所需**证据 ≤ read budget 时成立。

## P1.2 完整 content-depth probe 方法与稳健性

- **状态**：`[DONE]`（2026-08-01，`.73`，记录 `status/P1_2_PROBE_RESULTS.md`，commit `6b0b72f`）
- **类型**：机制实验 + Paper 方法补充。
- **必须补齐**：probe 数据与标签、sample count、train/dev/test split、probe architecture、regularization、optimizer、`knee98` 数学定义、3+ seeds、置信区间。
- **控制**：lexical-only、position-only、random-label、class-balance；至少一个非 Qwen 家族；比较 linear probe 与更受控的 readout。
- **目标**：区分“线性可读”与“被模型实际使用”，避免把 probe accessibility 直接称为 understanding。

**结果填写**

- 数据/标签：SST2（`nyu-mll/glue`）、WiC（`aps/super_glue`）、RTE（`nyu-mll/glue`）；固定分层池 n≈3000（pool_seed=0）；per-seed 分层 60/20/20 train/dev/test。
- probe/优化器：L2 logistic regression（sklearn lbfgs，max_iter=1000），输入 = StandardScaler 归一化的池化 hidden state；C∈{0.1,1,10} 用 dev 选。
- knee98 定义：knee98 = min{ l∈{0..L} : a(l) ≥ 0.98·max_l a(l) }，fractional depth = knee98/L。
- seeds/CI：5 seeds + Student-t 95% CI（≥3 达标）。
- controls：lexical-only、position-only、random-label（+Hewitt-Liang selectivity）、class-balance（majority/balanced-train/macro-recall），各独立 run。SST2 probe peak ~0.90 ≫ lexical 0.71 ≫ position 0.56 ≈ majority 0.56；random-label peak 0.54（selectivity +0.36）。WiC/RTE 低 selectivity（peaks 0.63–0.70）。
- 非 Qwen 结果：Meta-Llama-3-8B(base) + OLMo-2-1124-7B(base) 均复现。**content-j（linear-probe knee）：Qwen3-8B 0.393L / Llama-3-8B 0.269L / OLMo-2-7B 0.285L**；native（模型自身 logit-lens verbalizer）readout knee 明显更深 → **readout gap +0.43L(Qwen) / +0.59L(OLMo) / +0.68L(Llama)**（例：Qwen SST2 layer12=0.33L 已线性可读 0.90，但模型自身通路到 ~layer23=0.64L 才脱离 chance）。
- raw/code：本地 `results/p1_2/`（gitignored）+ `.73`；harness 随 commit `6b0b72f`。
- **Verdict（交 main 决定 paper 措辞）**：mechanism（content 中段线性可读、远早于模型自身 readout 使用 → gap = adapter 的活）**跨 3 家族稳健**；但 **"content-j ≈ 0.45L 近尺度不变" 是 Qwen 家族/任务平均专属、非普适**（跨家族 0.27–0.39L）。paper 的 0.44L(8B) 落在 Qwen CI [0.32,0.47] 内但 Llama/OLMo 未复现。**建议**：把 constant 软化为 family/task-qualified band，把不变性主张放在 *ordering*（content 早于 readout）而非 constant 上。caveat：Llama-WiC native readout 退化（peak≈chance），已从 gap 平均排除。

## P1.3 Prompt/chat-template sensitivity

- **状态**：`[DONE]`（2026-08-01：主 Qwen CoMem/KV-Direct chat T/F 已完成；MemoryLLM native-chat 的 RULER/BABILong diagnostic 已写入附录；MemoryLLM official-template LoCoMo judge 最后一格已补 = **14.75**，记录 `status/P1_3_MEMORYLLM_LOCOMO_JUDGE.md`。）
- **类型**：额外评测。
- **设计**：对主 Qwen3-8B LoCoMo 比较 no-chat controlled protocol 与标准推荐模板；对 MemoryLLM 使用其官方模板，并与当前 OOD no-chat diagnostic 分开。
- **报告**：所有系统的 prompt、generation settings、score；不把不同模板结果混入一个平均值。

**结果填写（2026-07-31：task #44 数据已在，可直接移植；selector=iter_bm25 固定）**

| Model/method | Prompt protocol | GPT-4o judge | token-F1 | Raw path |
|---|---|---:|---:|---|
| Qwen CoMem (iter_bm25) | **chat=False**（controlled，headline） | **38.27** | 9.15 | `locomo_results/qcmem_8b_iter_chatFALSE` |
| Qwen CoMem (iter_bm25) | chat=True | 37.76 | 19.51 | `locomo_results/qcmem_8b_iter_chatnothink` |
| Qwen KV-Direct (oracle) | **chat=False** | **34.59** | 9.02 | `locomo_results/kvdirect_8b_chatFALSE` |
| Qwen KV-Direct (oracle) | chat=True | 38.22 | 40.06 | `locomo_results/kvdirect_8b_chatnothink` |
| MemoryLLM (Llama-3-8B-chat) | official chat=True | **14.75** | — | `locomo_results/memoryllm_officialtmpl_locomo` |
| MemoryLLM (Llama-3-8B-chat) | chat=False (OOD diag) | 16.11 | — | `locomo_results/memoryllm_8b_chatFALSE` |

- **核心发现（judge 是 protocol-robust 的 canonical metric，token-F1 是 formatting artifact）**：
  1. GPT-4o judge 下 CoMem ≈/> KVD **两种协议**：chat=True 37.76 vs 38.22（tie）；chat=False **38.27 vs 34.59（CoMem +3.68 领先）**。
  2. chat=True 的 token-F1 KVD 40.06 >> CoMem 19.51（+20.55），但该 gap 在 chat=False 下**消失**（9.15 ≈ 9.02）→ 是 **chat-template artifact 非能力差**。
  3. CoMem judge 跨协议稳定（37.76→38.27）；KVD judge 掉（38.22→34.59）→ CoMem（base-trained 压缩法）对其原生 no-chat 更鲁棒。
- **已补（2026-08-01）**：MemoryLLM **官方 chat 模板**下的 LoCoMo judge = **14.75**（1986 items；cat1 13.12 / cat2 9.03 / cat3 13.54 / cat4 25.09 / cat5 abstention 0.67；1540 answerable 全 gpt-4o judge，0 API fail）。用**预存预测** `locomo_results/memoryllm_chatnothink`（native Llama-3 chat template + README BOS-drop = 官方模板）judge-only 得出，**零新增 GPU**。**发现**：官方模板 14.75 **低于** chat=False OOD 诊断 16.11（≈ −1.4pp）→ 把 LoCoMo 长注入上下文包进原生 chat 模板对 MemoryLLM 无益。judge 口径与 CoMem/KVD headline 完全一致（`eval_qcmem_locomo.py --score_only --use_llm_judge --judge_model gpt-4o`，endpoint maas-openapi.wanjiedata.com，seed=1，4 retries），可直接对照。记录：`status/P1_3_MEMORYLLM_LOCOMO_JUDGE.md`。
- **不混平均**：judge 作 LoCoMo headline；token-F1 若展示必须标 chat 状态 + formatting-sensitive 注，禁单报 chat=True F1（制造虚假 KVD win）。

## P1.4 LoRA 训练 seed 与 judge 稳健性

- **状态**：`[DONE]`（2026-07-31：两个追加 adapter seed、DeepSeek-V3 audit 和 conversation-cluster CI 均已完成并写入论文。**n-matched backfill 亦已完成**：flagship 在 seed1/2 完全相同的 cell 上重跑 n=50（新目录 `ruler_results/ruler_qcmem_seed42_n50` / `babilong_results/babilong_qcmem_seed42`，n=500 旗舰目录未动），得**真 3-seed matched-n**：RULER max **2.31pp**/median 0.40pp、BABILong max **4.36pp**/median 1.73pp、18-cell median **1.34pp**（旧 seed1-vs-seed2-only 为 median 0.71 / max 3.54）。flagship-n=500-vs-seeds-n=50 caveat 已真正清除；论文 abstract/§05/§07/§08 已更新为 "1.34 … max 4.36"。仅剩 effective-batch-3-vs-8 二阶 caveat 保留（data-seen 已 matched）。commit `28b22a8`（未 push）。）
- **类型**：训练 + 评测。
- **设计**：主 `j=12` self-distillation LoRA 至少 3 seeds；报告 benchmark mean±std。扩大可复现 judge audit，或使用可固定版本的开放 judge；报告 conversation-level 数值 CI。

**结果填写（2026-07-31：seed variance 已跑，n=50 RULER / n=100 BABILong，chat=False iter_bm25，data-budget matched）**

- **seeds**：flagship=42（旗舰，`outputs/qcmem_distill_qwen_j12_r32_4k/final`）+ seed1=1 + seed2=2（`..._seed{1,2}/final`），三者同超参（j=12, rank=32, chunk512, n_ctx=7, λ=0.6, topk64），seed1/2 用 matched data budget（3-GPU×10667≈32000 samples = flagship 8-GPU×4000）。
- **RULER seed variance（seed1 vs seed2，n=50）**：max std = **2.83pp**（niah_multikey@64k/128k）；多数 cell ≤1.5pp。128k：single 100/100、mk 100/96、vt 99.6/99.6。**headline 跨 seed 高度稳定，无 seed-lucky。**
- **BABILong seed variance（n=100）**：max std = **3.54pp**（qa2@4k）；多数 cell ≤2.12pp。qa5：4k 76.5±2.12 / 16k 66.5±2.12 / 32k 67.5±0.71。
- **整体**：18 cell overall median std = **0.71pp**，max std = 3.54pp → distilled LoRA 非 seed-lucky，flagship 落在 seed1/2 区间内。
- **caveat（已写入 paper）**：seed1/2 effective batch=3 vs flagship=8（data seen 与 evaluation n 已 matched，batch/optimization-noise 二阶差异保留）。
- **expanded judge audit ✅**：DeepSeek-V3 200-item audit（agreement 0.81，κ=0.626）与 10-conversation cluster bootstrap CI `[1.27,8.32]` 均已完成并集成。

| Seed | RULER 128k (single/mk/vt) | BABILong qa5 (4k/16k/32k) | Checkpoint/raw |
|---:|---|---|---|
| 42 (flagship) | (canonical n=500 dir) | — | `outputs/qcmem_distill_qwen_j12_r32_4k/final` |
| 1 | 100/100/99.6 | 75/65/68 | `outputs/qcmem_distill_qwen_j12_r32_4k_seed1/final` |
| 2 | 100/96/99.6 | 78/68/67 | `outputs/qcmem_distill_qwen_j12_r32_4k_seed2/final` |

- **mean±std**：RULER overall max std 2.83pp / BABILong max std 3.54pp / 18-cell median 0.71pp。
- **expanded judge audit**：GPT-4o + DeepSeek-V3 双 judge 同序（见 P0.9）；数值 cluster CI 待补。

## P1.5 任务覆盖扩展

- **状态**：`[DONE]`（2026-08-01，同 P1.1 run，`scripts/eval_p1_scaling.py`；记录 `status/P1_1_P1_5_SCALING_RESULTS.md`，commit `84720fe`）
- **原状态**：`[TODO]`
- **类型**：额外评测。
- **目的**：覆盖 fixed top-k retrieval 最可能失败的任务。
- **优先任务**：需要跨许多 chunks 聚合、全局统计、证据数量随 context 增长、长篇生成的任务；不要只补 needle retrieval。
- **必须报告**：evidence recall、最终 answer quality、read budget、失败案例。

**结果填写**

- **tasks**：single-evidence NIAH（E=1 control）、multi-hop VT chain（E=5 distributed）、VT chain-length **stress** sweep（E=4→32 crossing top-12）、cross-chunk aggregation `niah_multivalue`（one key, E scattered values）、global-statistics `cwe`（common-word frequency）。
- **selection rationale（结果前确定）**：`是`——任务在 `build_jobs()` 中于任何 run 之前固定，针对 fixed top-k read 区别于 single-needle 的三种失败路径（evidence 超预算 / 词法共指不可分 / 全局聚合无局部证据集）。
- **scores/raw paths**：`ruler_results/p1_scaling_{retrieval,full}/`（.82 diskB）；multivalue @128k E=1/4/8/12/16/24/32 → recall 1.00/1.00/0.62/0.40/0.31/0.23/0.17，score 100/95/56/36/29/20/14；cwe coverage 128k=0.045 / 512k=0.012，score 0.0。
- **失败模式（四类）**：
  1. **证据数 > read budget** — 硬顶 recall = min(1, top-k/E)（VT stress E=16/24/32 → 0.75/0.50/0.375）；缓解=更大 top-k 或多轮 iter_rounds，与 store 大小无关。
  2. **词法共指证据**（niah_multivalue）— 同 key 的多 needle BM25 无法单独排序，E<budget 也退化（E=8→0.62 vs VT chain E=8=1.000）；是 *selector* 限制非容量限制（语义/学习式 selector 可缓解）。
  3. **全局聚合任务**（cwe）— 任何 bounded top-k read 根本无法回答，coverage 随 store 增长 →0（128k 4.5%→512k 1.2%）；retrieval-based memory 在此为错误工具。
  4. **多跳链推理**（VT）— retrieval 已解（recall 1.000 到 4M）但 *answer* 弱（18–54）：depth-12 `h_j` read 携带链接但模型 compose 传递链不稳定；瓶颈是对 bounded read 的**推理**而非检索——对 “bounded read” claim 的重要细节。

## P1.6 同 retained-token budget 的 selected/compressed-KV 基线

- **状态**：`[DONE — 全部完成 2026-08-02]`（task #120，.252 8×L20A 质量 + .104 1×H20 timing，wzc1）。**SnapKV+PyramidKV 的 native 15-cell + yarn 64k/128k + LoCoMo + full-prefill/peak/decode timing 全部完成并回填（见下方结果表 + timing 明细）；无 TBD。**harness `scripts/eval_p16_kvcompress.py` + `_run_p16_baselines.sh`（vendored SnapKV FasterDecoding@e216ddc + PyramidKV Zefan-Cai@94255b6，GQA-aware，Qwen3 tf-5.14 faithful wrapper，budget=6657/window=32），commit `1f1783c`。fail-closed 忠实度自检 gate：SnapKV PASS（short<budget bit-identical / long uniform retained==budget）；PyramidKV 自检 false-negative 已修（floor=window_size + 非递增单调，commit `bf9bc41`，离线验证 OVERALL_PASS）。
  - **SnapKV / PyramidKV native ✅ DONE**：两者 Cohort-A 15-cell、LoCoMo 与 fidelity gates 全部通过。
  - **YaRN 64k/128k ✅ DONE**：两方法均使用 factor 4、完整 prompt、无 native-window truncation；全部 cell 已回填。
  - **timing/peak/decode ✅ DONE**：同一单 H20 timing harness 完成 8k/32k/128k full-prefill、peak memory、64-token decode 与 retained-KV 统计。
  - **`.tex` ✅ INTEGRATED**：正文 equal-budget 段与 Appendix Table `tab:kvcompress` 已加入；明确 native 长档 truncation、YaRN full-prompt 和 full-prefill-before-eviction 边界。
- **优先级判断**：这是当前最有价值的外部竞品定位实验，但不是修复内部有效性的必需项。标准 SnapKV/PyramidKV/H2O 通常先 full-prefill 再压缩 KV，与 CoMem 的 persistent-store bounded Read 不同构；结果必须按两条系统轴解释。
- **方法选择**：优先标准 **SnapKV + PyramidKV**；若官方实现与 Qwen3/SDPA 不兼容，可用 H2O 替换，但必须记录版本、patch 和忠实度自检，禁止用仓库内自研 `PyramidMemory` 或 “SnapKV-on-chunks” 冒充标准方法。
- **共同协议**：同 Qwen3-8B revision、chat=False、enable_thinking=False、generation/scorer、examples；retained KV/token budget 固定为 **6,657**（或方法支持的最接近值并报告差异）。
- **任务**：RULER Cohort A 15 cells + LoCoMo full set；保存逐例 prediction。64k/128k 若需 YaRN，必须与方法名绑定报告，并另给原生窗口内结果，不能把 YaRN 只加给某一方法后称为纯压缩比较。
- **必须同时报告**：质量、full-prefill latency/peak memory、压缩后 retained KV bytes、decode latency、是否需要看到完整 prompt、OOM/fallback。不能仅比较“最终留下 6.5k tokens”而隐去 full-prefill 成本。
- **公平性解释**：
  1. 与 CoMem 的质量对比是 equal-retained-token diagnostic；
  2. 与 CoMem 的在线系统对比必须同时纳入 full-prefill、persistent storage 和 repeated-query reuse；
  3. 若方法无法处理超原生窗口，按能力边界报告，不做 workaround。
- **验收**：两种标准方法均完成 ≥15 RULER cells 与 LoCoMo；官方/stock full-KV sanity 在短输入上匹配；raw predictions、timing、peak memory、config、commit、GPU、失败样本齐全；无预设胜负阈值。

**结果填写**

RULER Cohort-A native（string_match_all，n=100/cell，8 shard，budget=6657 window=32，chat=False think=False，Qwen3-8B，bf16 SDPA greedy；64k/128k left-trunc 40960，原生窗口内；raw `.252:ruler_results/p16_snapkv_native/`）：

| SnapKV native | 8k | 16k | 32k | 64k | 128k | 行均 |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_2 | 100.0 | 100.0 | 100.0 | 61.0 | 29.0 | 78.0 |
| niah_multikey_1 | 100.0 | 100.0 | 100.0 | 59.0 | 17.0 | 75.2 |
| variable_tracking | 100.0 | 100.0 | 99.6 | 15.0 | 4.4 | 63.8 |

- RULER A macro（15-cell mean）= **72.33**（IRON-LAW-2 全 cell OK）。
- LoCoMo full set n=1986（raw `.252:locomo_results/p16_snapkv/scores.json`）：OVERALL **F1=9.21 / EM=1.11 / acc=22.05**；per-cat F1（EM/acc）—— multi_hop 11.22(0/17.38,n=282)、single_hop 7.12(0/11.21,n=321)、temporal 6.89(0/18.75,n=96)、open_domain 11.87(0/37.22,n=841)、adversarial 4.93(4.93/4.93,n=446)。
- YaRN 64k/128k：**SnapKV / PyramidKV ✅ DONE**（2026-08-02，factor 4，n=100/cell，完整 prompt，无 native truncation）：SnapKV single/multikey/tracking = 100/100、94/91、80.0/84.8；PyramidKV = 100/100、93/89、88.6/86.8。raw `ruler_results/p16_{snapkv,pyramidkv}_yarn/`。
- **Full-prefill latency / peak GB / decode / retained-KV ✅ DONE**（2026-08-02，.104 1×H20，bench `scripts/bench_p16_kvcompress_timing.py` commit `c577ef5` 未 push；复用 P1.6 质量 eval 的同一 KV-compress hijack；median-of-3，warmup≥1，sync-bracketed，n_decode=64；raw `.104:outputs/p16_timing/p16_timing_full.json`）。**⚠️ env：.104 `.venv` 已被 reset 成坏的 py3.14（无包）→ 用 `/opt/conda/envs/torch-base/bin/python`（transformers 5.5.4，与 P1.6 质量 eval 同版本），hijack API 核对在位。**
  | Method | Len | Full-prefill (ms) | Peak (GB) | Decode (ms/tok) | Retained KV (MB) |
  |---|---|---:|---:|---:|---:|
  | SnapKV | 8k | 1212.8 | 18.59 | 24.7 | 945 |
  | SnapKV | 32k | 6342.2 | 25.73 | 24.6 | 945 |
  | SnapKV | 128k | **51519.9** | **54.30** | 24.7 | 945 |
  | PyramidKV | 8k | 1203.1 | 18.59 | 24.7 | 945 |
  | PyramidKV | 32k | 6301.0 | 25.74 | 26.7 | 946.6 |
  | PyramidKV | 128k | **51520.2** | **54.31** | 26.8 | 946.6 |
  - **★ two-system headline**：SnapKV/PyramidKV 必须 full-prefill 整个 prompt → 128k prefill **≈51.5 s** + peak 随 L 涨 **18.6→54.3 GB**；对照 CoMem peak context-flat ≈17.3/17.5/18.3 GB（8k/32k/128k）+ L-independent Read。即"最终留 6657 tok"不能掩盖 baseline 的 full-prefill 成本——这正是 P1.6 要求披露的关键系统差异。retained-KV ≈945 MB（含 64 decode tok；纯 post-prefill ≈936 MB）。

| Method | Retained budget | RULER A macro | LoCoMo F1/acc | Full-prefill ms/GB | Decode ms/tok | Raw path |
|---|---:|---:|---:|---:|---:|---|
| **SnapKV** (native win) | 6657 | **72.33** | **9.21 / 22.05** | 128k: 51520/54.3GB (8k 1213/18.6) | 24.7 | `ruler_results/p16_snapkv_native/`, `locomo_results/p16_snapkv/scores.json` |
| **SnapKV-yarn** 64k/128k | 6657 | ns2 100/100 · mk1 94/91 · vt 80/84.8 | — | 128k 51520/54.3GB (同 timing 表) | 24.7 | `ruler_results/p16_snapkv_yarn/` |
| **PyramidKV** (native win) | 6657 | **72.32** | **9.13 / 22.10** | 128k: 51520/54.3GB (8k 1203/18.6) | 26.8 | `ruler_results/p16_pyramidkv_native/`, `locomo_results/p16_pyramidkv/scores.json` |
| **PyramidKV-yarn** 64k/128k | 6657 | ns2 100/100 · mk1 93/89 · vt 88.6/86.8 | — | 128k 51520/54.3GB (同 timing 表) | 26.8 | `ruler_results/p16_pyramidkv_yarn/` |

PyramidKV native 15-cell（n=100，IRON-LAW-2 全 OK，raw `ruler_results/p16_pyramidkv_native/`）：ns2 = 100/100/100/61/29；mk1 = 100/100/100/59/17；vt = 100/100/99.6/15.2/3.0（8k→128k）；A macro(15)=72.32。LoCoMo n=1986：F1 9.13 / acc 22.10。**两 baseline（SnapKV 72.33 / PyramidKV 72.32）native macro 几乎相同，同 6657 retained budget 下均低于 Cohort-A CoMem+LoRA(97.05)** → P1.6 质量对照完成。full-prefill latency/peak-mem/decode（two-system 轴）**✅ 已补齐**（见上 timing 明细：128k full-prefill ≈51.5s + peak 54.3GB vs CoMem context-flat ~18GB）→ **P1.6 全部完成，无 TBD**。

## P1.7 Continuous-prefix $h_{12}$ 归因 oracle

- **状态**：`[DONE — both cohorts complete 2026-08-02，.82]`（task #121；结项）
- **目的**：分解 P0.13 的 3.12-point gap 中，连续下层上下文计算与 deployable chunk-local Write/repositioning 的贡献；不把 oracle 当可部署方法。
- **三臂**：
  1. `j=0 full replay + flagship LoRA`（P0.13 Arm A）；
  2. `j=12 chunk-local cached h12 + same LoRA`（P0.13 Arm B）；
  3. **oracle**：对完全相同的 selected pack 用连续位置/全 causal attention 跑 layers 0--11，一次性截取 pack-level `h12`，再以相同 LoRA 跑 layers 12--35。
- **关键约束**：三臂同 example、pack IDs/order/token hash、LoRA hash、decode/scorer。Oracle 的 layer-12 state 必须与同 pack 的 stock lower-12 forward 数值匹配；不得从独立 chunk cache 拼接。
- **最小任务**：`niah_multikey_1` 8k/16k，n=100/cell；推荐扩展至 P0.13 Cohort-B 15 cells。
- **统计**：逐例 predictions；paired bootstrap CI + McNemar；报告 A/B/C 两两差、first-token/decode agreement、read latency和 peak memory。
- **解释规则**：
  - oracle 接近 j=0：gap 主要来自 chunk-local Write/repositioning；
  - oracle 接近 deployable j=12：跳过/接口适配占主要部分；
  - 任何结果都不得宣称知识“位于”某层。
- **部署 caveat**：oracle 需要每个 query 对 selected pack 重跑下 12 层，无法作为跨查询缓存；它只用于归因。

**结果填写**

| Cell | j=0 (A, full) | continuous-h12 oracle (C) | chunk-local h12 (B, deployable) | Pairwise CI/tests | Raw path |
|---|---:|---:|---:|---|---|
| multikey 8k+16k（pooled min-cohort，n_paired=200） | 100.00 | 100.00 | 92.50 | A−C=+0.00 CI=[0,0] p=1；A−B=C−B=+7.50 CI=[4.0,11.5] McNemar p=6.1e-05 | `.82(diskB):bench_results/p1_7_h12_oracle`（`logs/p1_7_oracle.out` 聚合） |
| Cohort-B（15 cells，macro，n_paired=1500） | 99.19 | 99.19 | 96.07 | A−C=+0.00 CI=[0,0] p=1；A−B=C−B=+3.12 CI=[2.36,3.93] McNemar p=8.79e-24 | `.82(diskB):bench_results/p1_7_h12_oracle`（`logs/p1_7_oracle_cohortb.out`） |

- **min-cohort 结论（2026-08-02，n_paired=200，2 cells）**：oracle（连续位置 pack-level h12）与 j=0 full replay **逐位 bit-identical**（A−C=+0.00，McNemar p=1，`p013_sha_match=True`），说明"跳过下 12 层计算"本身**不损失质量**；deployable chunk-local h12 相对二者 **−7.50pp**（CI=[4.0,11.5]，McNemar p=6.1e-05）。∴ P0.13 的 deployable gap **完全归因于 chunk-local h12 缓存/repositioning**，而非跳过下层计算——支持"depth-as-reuse-axis 的代价来自 Write-side 的 chunk-local 近似，可用 oracle 上界界定"。`oom=0 nonfinite=0`，`packs_paired_1to1=True`。
- **oracle 有效性 gate**：`--mode h12_sanity` 早前 PASS（continuous-oracle-h12 == stock lower-12，max_abs=0.000e+00），确认 oracle 的 layer-12 state 与同 pack stock 下 12 层数值一致，非拼接。
- **部署 caveat（写 .tex 必遵守）**：oracle 需每 query 对 selected pack 重跑下 12 层，不可跨查询缓存；仅作归因上界，**不得**表述为可部署方法。
- **Cohort-B 结论（2026-08-02，n_paired=1500，15 cells，.82）**：全 15-cell macro 上 oracle(C)=99.19 与 j=0 full replay(A)=99.19 **仍逐位 bit-identical**（A−C=+0.00，McNemar p=1，`p013_sha_match=True`）；deployable chunk-local h12(B)=96.07 相对二者 **−3.12pp**（CI=[2.36,3.93]，McNemar p=8.79e-24）。此 3.12pp **精确等于 P0.13 的 deployable gap** → 在完整任务/长度分布上确认："跳过下 12 层计算零损失，全部 gap 归因于 chunk-local h12 缓存/repositioning 的 Write-side 近似"。`oom=0 nonfinite=0`，`packs_paired_1to1=True`。**#121 结项。**
- **`.tex` ✅ INTEGRATED**：正文机制段、Limitations 与 Appendix Table `tab:h12-oracle` 已加入；明确 oracle 每 query 重跑下 12 层、不可作为可部署缓存。

## P1.8 真实 repeated-query serving 曲线：CoMem vs `j=0`

- **状态**：`[HARNESS READY，GPU 排队 — 2026-08-03，serving-curve harness（scripts/bench_p1_8_serving_curve.py + scripts/_run_p1_8_serving.sh），workflow wg28ofr1v，commit c32a2c9（author LiuHanzuo，未 push）。CPU 通过（py_compile + import + crossover 合成 fixture：Q* 随 G 下降、winner grid 翻转、P0.2 解析交叉核对）。fail-closed 5 门（LoRA sha / store-fetch 选中 h12==fresh recompute max_abs 0 / 单 pack / persistent_bytes 精确 / finite logits）。L=1M store cell ~16GB（H20 97.8G 可跑；逼近 OOM 用 --tier cpu pinned host store，launcher 已单独 fan）。默认矩阵 L∈{32k,128k,1M}×tier{gpu,cpu}×3 proc。GPU run 排队 .104（P0.20 之后）。]`
- **目的**：正面回答 CoMem 在何种 workload 下相对 matched raw-text replay 严格占优，而不是只给解析式 break-even。
- **对照**：`j=0` BM25 raw-text replay、CoMem `j=12+LoRA`；可附 full context 作为参考，但主判断必须是 CoMem vs `j=0`。
- **矩阵**：context/store `L∈{32k,128k,1M}`；同文档查询数 `Q∈{1,4,16,32,64}`；generation length `G∈{1,32,128,512}`；至少测试 GPU-resident 与 CPU-pinned 两个 store tier，CEPH/NVMe 可用 P2.2 实测组件或补统一运行。
- **计时边界**：一次性 index/Write、selection、fetch/H2D、model prefill/Read、decode 全部分列，同时报告累计 latency、amortized latency/query、peak GPU/host、persistent bytes和吞吐。不得把不同硬件 cohort 相减。
- **输出**：一张 `Q×G` crossover heatmap/curve，明确 CoMem 何时胜、何时 `j=0` 胜；报告中位数和尾延迟，至少 3 个独立进程。
- **验收**：原始 timestamps 可重算；真实 crossover 与 P0.2 解析估计一致或解释差异；负面区域完整保留。

## P1.9 Dense retriever + native prompting 的标准 RAG reference

- **状态**：`[HARNESS READY，GPU 排队 — 2026-08-03，dense-RAG reference harness（scripts/eval_p1_9_dense_rag.py + scripts/_run_p1_9_dense_rag_8gpu.sh + paperA/P1_9_DENSE_RAG_NOTES.md），workflow wg28ofr1v，commit c32a2c9（author LiuHanzuo，未 push）。retriever=models/bge-large-en-v1.5（frozen，weight sha 45e19549…== 硬编码门，--mode provenance exit0）；chunk=512、Qwen3-8B reader no-LoRA j=0；raw-text RAG 与 CoMem 共用同 examples + 同排序列表 top-k 前缀。默认 cohort 44 jobs（babilong qa1/qa2×{4k,8k,16k}+longeval{8k,16k}+locomo+ruler niah_multikey_1×{8k,16k}，n=100，4 shards）；分解报 recall@k / hit-conditional reader acc / e2e quality / query-enc+ANN latency / index size；READER_PROMPTS="plain native" 做 template-sensitivity。GPU run 排队 .104（P0.20 之后；本实验亦为 P0.20 阶段B dense equal-latency 前置）。]`
- **目的**：补齐 BM25 `j=0` 之外更接近真实部署的 RAG reference，避免系统结论仅依赖 lexical selector；该实验不替代 matched BM25 路径，也不与 MemoryLLM 混为同类。固定 dense retriever 后，必须同时服务 raw-text RAG 与 CoMem，形成 P0.20 的 dense equal-latency 主比较。
- **建议配置**：固定一个公开 dense retriever（优先 BGE 或 E5，冻结版本/hash），同 chunk=512、Qwen3-8B reader；先完成 top-12 sanity，再按 P0.20 扫描 `k∈{2,4,6,8,10,12,14,16,20,24}`。同时给统一 no-chat 主协议和 reader 原生 prompt/template sensitivity。若 retriever 需要 query instruction，必须按官方说明固定。
- **任务**：优先 BABILong qa1/qa2、LongEval、LoCoMo；附 RULER multikey但不以 lexical needle 为唯一结论。
- **分解**：同时报告 recall@k、reader conditional-on-hit accuracy、end-to-end quality、query encoding/ANN latency和 index size；raw-text RAG 与 CoMem 必须使用相同 examples、同一排序列表及其 top-`k` 前缀。
- **验收**：retriever 模型、corpus index、distance metric、pooling、normalization、query instruction、版本和硬件完整；不得只展示 dense retriever 获胜的任务。最终必须回填 P0.20 阶段 B 的 latency-matched `k_RAG*`/`k_CoMem*`，不能只报告固定 top-12。

## P1.10 E1：Write-path representation/interface distillation（仅条件启动）

- **状态**：`[BLOCKED — 等 P0.16/P0.18 决策；不得提前训练]`
- **触发条件**：P0.16/P0.18 证明主要误差可由 query-independent Write representation 或可学习位置接口修复；若主要问题是不可消除的 pack-coordinate mismatch，则取消本任务。
- **实现要求**：
  1. 新建独立训练脚本/config，不污染旗舰 adapter；
  2. LoRA 挂 lower layers `[0:12)` 或明确的位置接口模块；
  3. 拆出可带梯度的 `write_chunk_core`，移除 Write 路径的 `no_grad`；
  4. teacher 必须在 `disable_adapter()+no_grad()` 下生成；
  5. target 必须 query-independent，禁止直接把 continuous-pack oracle 当 writer target；
  6. 更新所有假设 LoRA 只在 `[12:36)`、module count=168 的 eval hard assertions。
- **目标/评测**：先做小规模 representation sanity，再决定是否 4k-step；最终必须跑 RULER-B paired 15 cells、LoCoMo、Write/Read timing和 persistent storage。
- **成功标准**：旗舰 RULER-B `96.07 → ≥98.5`，同时保留至少 `1.35×` model-side Read speedup，Write/storage 增量完整披露；否则作为负面结果，不替换旗舰。

---

# P2：可选增强

## P2.1 原生长上下文自然任务复现

- **状态**：`[DONE]`（2026-08-01，worker 交付 `status/P2_1_NATURAL_LONGCTX.md`（在 **LOCAL wzc1**，01:31 版仅含 QCMem 臂；Dense 臂 03:44 完成，最终数字见下），main 回填）
- **类型**：额外模型评测（无需训练新 adapter，复用旗舰 LoRA）。
- **目的**：在**原生自然长文档**任务上比较 QCMem（bounded read）vs Dense full-context，避免所有超窗口结论依赖 RoPE extension（synthetic RULER needle）。
- **基准**：**InfiniteBench (∞Bench)** —— `longbook_qa_eng`（自然长篇 QA，F1）+ `longbook_choice_eng`（多跳阅读理解 MC，acc）。原生长文档（90k–266k+ tokens），非 synthetic single-needle。
- **⚠️ 验收口径修正（用户 2026-08-01，写 .tex 必须遵守）**：原 TODO 要求 "**原生 128k+ backbone**"；实际 backbone = Qwen3-8B，**原生窗口仅 40,960（~40k）**，**不满足 "128k+ 原生 backbone" 验收条件**（worker 记录 `P2_1_NATURAL_LONGCTX.md` 里的 "131072 native window" 是笔误——131072 是 YaRN 扩展窗口，非原生）。∴ .tex **只能**表述为 "**natural long-document, over-native-window stress test**"（自然长文档、超原生窗口压力测试），**严禁**写成 "在原生 128k backbone 上复现 / reproduced on a native 128k backbone"。

**结果填写**

- backbone/checkpoint：Qwen3-8B（`models/Qwen3-8b-local`）；QCMem 臂 LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`；Dense/KV-Direct 臂 = 同一 Qwen3-8B base，无 adapter，读**整篇文档**。
- native window：Qwen3-8B **原生窗口 40,960（~40k）**（131072 是 YaRN 扩展窗口，非原生；worker 记录误标为 "131072 native"，此处订正）→ InfiniteBench 所有文档（90k–266k+ tok）均**远超原生窗口**；QCMem 走 bounded top-12 pack（constant ~18.5 GB/card），Dense=**native-window（resume_j=0，无 YaRN）**试图 full-context（read_len 数十万 tok，随文档线性涨到 49–95 GB → 长档 OOM，且远出 RoPE 训练范围）。
- adapter config：`--resume_j 12 --selector iter_bm25 --topk 12 --chunk_size 512 --sink_tokens bos --dtype bfloat16 --attn_impl sdpa --seed 42`，**chat_template=False**（旗舰口径）。
- 8-way GPU shard（`sample % 8`），per-cell merge。

| Method | longbook_qa_eng F1 (n) | longbook_choice_eng acc (n) | MACRO | Peak mem/card | OOM |
|---|---:|---:|---:|---:|---:|
| **QCMem** (j=12, iter_bm25, LoRA) | **6.06** (351) | **17.47** (229) | **11.76** | ~18.5 GB (flat) | **0** |
| Dense / KV-Direct (full ctx) | 2.16 (351) | 2.86 (**35**) | 2.51 | 49–95 GB (↑with doc) | 大量长档 |

**★ LL-based MC rescore（#112 已交付 2026-08-01；关闭下方 caveat(2)）** —— choice metric 从 chat=False 下的 clean-letter 抽取改为 `--mc_ll`（LL-based MC：对每个选项算 leadin+option 的条件对数似然取 argmax；scorer = commit `1efa4d0`，已 in-sync 到 diskB harness，grep=5）。QCMem 臂配置与上表完全一致（resume_j=12 · iter_bm25 · topk12 · chunk512 · sink=bos · chat=False · LoRA `qcmem_distill_qwen_j12_r32_4k/final`），full n=229，**0 OOM**。**Dense 臂在 LL-MC 下走 native-window 截断（resume_j=0，非 full-ctx）**，故能给 matched-n=229（20 个超长档在 native window 下仍 OOM，按 fallback 计分）——这是与上表 full-ctx n=35 臂**不同的 feasible-Dense operating point**，两者并存不互斥。orchestrator `scripts/_infb_orchestrate_p21.sh` S3/S4（.82 diskB，02:52–03:43）。

| Method (LL-MC, chat=False, n=229) | longbook_choice_eng acc_ll | acc_norm | OOM(fallback) | Raw path |
|---|---:|---:|---:|---|
| **QCMem** (j=12, iter_bm25, LoRA, bounded read) | **48.03** | 46.29 | 0 | `infbench_results/qcmem_8b_j12_lora_llmc/scores.json` |
| Dense native-window (resume_j=0, LL-MC) | 32.31 | 32.75 | 20 | `infbench_results/kvdirect_8b_llmc/scores.json` |

- **关键**：LL-MC 把 QCMem choice 从 clean-letter **17.47 → 48.03（+30.56pp）**——证实 caveat(2) 假设：**chat=False 下 clean-letter 抽取严重低估 choice acc**（模型有正确选项偏好但不输出干净字母）。matched-n=229 同 LL-MC scorer 下 **QCMem 48.03 > native-window Dense 32.31（+15.72pp）**，比原 229-vs-35 非对称口径**更干净的同分母比较**。QCMem MACRO 若用 LL-MC choice 则 (6.06+48.03)/2=**27.05**（仍按用户 line 469 指令不作公平质量并列，coverage/feasibility 为主轴）。
- **写 .tex 口径（main，本轮不改 .tex）**：choice 质量以 LL-MC **acc_ll=48.03** 为 headline choice 数（替换 clean-letter 17.47）；Dense 32.31 注明为 native-window-truncated + LL-MC 的 feasible operating point（≠ full-ctx n=35 崩塌臂，后者仍单列作 coverage/OOM 证据）；QA F1（6.06 vs 2.16，n=351）不受影响。

- **Headline（用户 2026-08-01 修正：不得把 MACRO 11.76 vs 2.51 当同分母公平质量比）**：主结论以 **coverage/feasibility** 为轴，非单一 MACRO 比值——
  - **Feasibility/coverage（核心卖点）**：QCMem 在全部 **351/351 QA + 229/229 choice** 文档跑通、**0 OOM**、显存恒定 ~18.5 GB/card；Dense（native-window，无 YaRN）在 **194/229 choice OOM**（仅 **35/229** 被打分），QA 虽全跑但远出窗口 → "bounded read over extensible store" 在超原生窗口的**鲁棒性/可行性**优势（不依赖 RoPE extension）。
  - **Quality（分开报 + 带 caveat，不可直接并列）**：QCMem QA F1 **6.06** (n=351) / choice acc **17.47** (n=229)；Dense QA F1 **2.16** (n=351) / choice acc **2.86** (**n=35 only**)。choice 两臂**分母不对称（229 vs 35）**且 choice acc **受 chat=False 下 clean-letter 抽取影响** → 只能做两个受控口径：**(a) 全集 QA F1 同 n=351 直接比（6.06 vs 2.16）**、**(b) 仅在 Dense 未 OOM 的 35 档上做同档配对 choice 比较**；**严禁**把 11.76 vs 2.51 的 MACRO 当公平质量比。
- **⚠️ 回填 caveat（main 在写 .tex 前须核实）**：(1) Dense choice **n=35 ≠ QCMem n=229**（Dense 在 194/229 choice 文档 OOM 未打分）—— 分母不对称，正文应以 "coverage（能否装下文档）+ 同档配对" 双口径陈述，不可直接并列 acc；(2) ✅ **已解决（#112，见上 LL-MC block）**：原 clean-letter 抽取（metric=`acc`，非 LL-based MC）在 chat=False 下确实低估 choice acc——用 `--mc_ll`（commit 1efa4d0 LL-MC scorer）重打分后 QCMem choice = **48.03（n=229，+30.56pp vs clean-letter 17.47）**，且 LL-MC 给出 matched-n=229 的 native-window Dense 对照 32.31（QCMem +15.72pp）。
- **provenance**：harness `scripts/eval_qcmem_infbench.py`（wzc1 commit 4bbfece + LL-MC 1efa4d0，未 push）；orchestrator `scripts/_infb_orchestrate_p21.sh`（节点 .73）；data `data/infinitebench/{longbook_qa_eng,longbook_choice_eng}.jsonl`；raw `infbench_results/{qcmem_8b_j12_lora,kvdirect_8b}/{*_shard*of8.jsonl,*_metrics.json,scores.json}`（diskB `.73`）；logs `logs/infb_*`。

## P2.2 Persistent-store 实际 I/O 与网络部署

- **状态**：`[DONE]`（2026-08-01，`status/P2_2_PERSISTENT_STORE_IO.md`；worker 交付，main 回填）

## P2.2 Persistent-store 实际 I/O 与网络部署

- **状态**：`[DONE]`（2026-08-01，`status/P2_2_PERSISTENT_STORE_IO.md`；worker 交付，main 回填）
- **类型**：系统实验。
- **设计**：GPU resident（HBM）、CPU pinned host RAM、local NVMe（overlay `/`）、networked store（CEPH `dop-fuse`）四种后端；write-once bf16 `h₁₂` store [n_chunks,512,4096]=8192 B/tok，top-12 read pack=50.3 MB；测 write throughput、random retrieval、H2D transfer、concurrent QPS(1/4/16)、peak GPU/host。store size 扫 128k/1M/4M/8M/16M。
- **环境**：node `.104`（H20，driver 535.247.01，torch 2.13.0/CUDA 13.2），GPU 0 only，全程 uncontended（I/O 保真）。N=7 中位数 + 2 warmup。
- **mount 验证**：overlay `/`（local NVMe）raw write **6.1 GB/s** vs CEPH `dop-fuse` **1.4 GB/s**（~4.3×）→ 本地盘确为本地非网络后端；O_DIRECT 在两者均生效（file read 绕过 page cache = 诚实介质）。

**结果填写**（代表性 store size = 4M/32GB；QPS 取 1/4/16 并发峰值。全 size 扫见下方 crossover）

| Backend | Write GB/s | Retrieve ms | Transfer ms | peak QPS | Peak GPU | Peak host | Raw path |
|---|---:|---:|---:|---:|---:|---:|---|
| GPU (HBM) | 1496 | 0.06 | 0 | 15709 | 32.98 GB | — | `ruler_results/p2_2/p2_2_gpu_fixed.json` |
| CPU pinned | 91 | 1.29 | 1.38 | 973 | — | 43 GB | `ruler_results/p2_2/p2_2_full.json` |
| NVMe (overlay) | 2.55 | 13.5 | 0.90 | 276 | — | ~2 GB | `ruler_results/p2_2/{p2_2_full,p2_2_file_isolated}.json` |
| network (CEPH) | 1.52 | 75.9 | 1.19 | 44 | — | ~2 GB | `ruler_results/p2_2/{p2_2_full,p2_2_file_isolated}.json` |

**★ Crossover（单-H20 store 容量天花板）**：GPU-resident `h₁₂` 在单张 H20 上**放得下到 8M tokens（64 GB，peak 64.98 GB）**、**16M（128 GB > 95 GiB HBM）时 OOM** → 单-H20 上限在 **8M–16M tokens 之间**。CPU-pinned 把上限外推 ~2×（16M/128 GB store，235 GB host）@ ~2 ms/query、~940 QPS；NVMe（~13 ms）与 network/CEPH（~79 ms）外推到几乎无界磁盘。**固定 50.3 MB top-12 pack 使 per-query H2D 在所有 off-GPU 后端恒为 ~1 ms（与 P0.2 的 1.2 ms 一致）**——read 成本与 store size 无关（O(1) read 的系统层证据）。file 后端 peak-host 全 size ~1.9–2.2 GB（从不把 store 常驻 RAM）。

**Provenance / 交付合规**：
- harness `scripts/bench_persistent_store_io.py`，commit **`dcc66df`**（author LiuHanzuo，无 AI trailer，仅该单文件，**未 push**）。
- raw JSON `ruler_results/p2_2/{p2_2_full,p2_2_gpu_fixed,p2_2_file_isolated}.json`；logs `logs/p2_2/`。
- **排除 run（2 个方法学 bug，已修，上表为干净复跑）**：(1) 首次 combined run 有 CUDA allocator 跨-cell 碎片化 bug 误 OOM GPU@8M → cell 边界加 `gc.collect()`+`empty_cache()`（已入 `dcc66df`），复跑 `p2_2_gpu_fixed.json` 得真实 8M-fits/16M-OOM crossover；(2) file 后端 peak-host 被前一个 CPU-pinned cell 的 RAM 污染 → 隔离复跑 `p2_2_file_isolated.json` 得真实 ~2 GB。详见 `status/P2_2_PERSISTENT_STORE_IO.md` 排除列表。
- **一句结论**：CoMem 的持久化 `h₁₂` store 可跨 GPU-HBM/CPU-pinned/NVMe/CEPH 四级介质部署，单-H20 HBM 容纳 ~8M-token store（16M OOM），off-GPU 后端以恒定 ~1 ms H2D、50.3 MB pack 把容量外推到无界磁盘、代价是 retrieval 延迟从 0.06 ms 升到 ~13 ms(NVMe)/~79 ms(CEPH)——read 成本与 store size 解耦。
- **待改 section/table**：论文 §Systems/Deployment 的 persistent-store 表（main-only .tex，本轮不改）。

## P2.3 附录 `tab_scale`：Qwen3 model-family scale RULER（chat=False 口径统一 refresh）

- **状态**：`[DONE]`（2026-08-01，task #62；worker 交付 `status/P62_SCALE_RULER_CHATFALSE.md`，commit `3ca261d`(driver+skeleton)+`4f7051a`(final)，author LiuHanzuo，**未 push**；main 回填）。
- **类型**：附录表 refresh（zero-training / no-adapter，无需训练）。这是既有 `tab_scale` 的 **chat_template=False 口径统一重跑**，不属于"当前仍缺的数据"优先级 gap，独立追踪。
- **目的**：把 model-family RULER scale sweep 全部拉到全论文强制的 **chat_template=False** 口径，补齐缺失的 0.6B / 1.7B / 30B-A3B 三档，与既锁定的 4B / 14B / 32B 参考行同口径横向对照。
- **协议（照搬 locked 4B/14B/32B run，`scripts/_qwen_scale_zerotrain_ruler_pool.sh`）**：`chat_template=False`、`enable_thinking=False`、`sink=bos`、`chunk_size=512`、bf16/SDPA、zero-training（无 LoRA / 无 bottleneck ckpt）；每档 split depth j 随模型深度缩放。**per-task selector（非单一全局 selector，与 locked run 完全一致以保证跨档可比）**：`niah_single`/`niah_multikey`→`bm25 topk12 rounds0`；`vt`→`iter_bm25 topk16 rounds4 hop4`。13 cell = niah_single{8k,16k,32k,64k,128k}+niah_multikey{8k,16k,32k,64k,128k}+vt{8k,16k,32k}，n=500/cell（4 shard×125）。driver `scripts/_run_scale_ruler_remaining_p62.sh`（row-count gate=125行/shard，绕开 32B-hardcoded 的 `qwen32_zerotrain_results.py --is-complete`）；scorer `scripts/_score_chatFALSE.py --ruler`（官方 `_string_match_all_one`，weighted-mean recall）。

**结果填写（RULER recall %，chat=False，n=500；粗体 = 本任务新档，其余为既锁定参考行）**

| size | j | niah_single 8k/16k/32k/64k/128k | niah_multikey 8k/16k/32k/64k/128k | vt 8k/16k/32k | **RULER mean (13 cell)** | n |
|---|---|---|---|---|---:|---:|
| **0.6B** | 2 | 100.0/99.8/100.0/99.6/99.6 | 87.6/80.8/83.6/84.4/91.0 | 60.8/65.0/61.6 | **85.68** | 500 |
| **1.7B** | 3 | 99.8/98.2/99.4/99.4/98.6 | 62.4/43.4/41.8/26.4/66.4 | 42.9/42.0/47.7 | **66.81** | 500 |
| 4B | (locked) | 92.2/97.4/95.4/98.0/98.8 | 34.0/35.8/40.8/32.0/34.8 | 56.4/36.5/34.6 | 60.52 | 500 |
| 14B | 13 | 99.8/82.8/97.6/98.4/97.8 | 49.2/42.6/9.8/34.2/36.4 | 14.2/11.6/13.5 | 52.91 | 500 |
| **30B-A3B** (MoE,128e/8-tok) | 12 | 99.4/99.6/99.0/99.4/99.8 | 67.8/51.6/53.2/52.4/62.8 | 89.2/88.0/84.0 | **80.48** | 500 |
| 32B | 27 | 100/100/100/100/100 | 96.0/100.0/84.0/88.0/100.0 | 46.4/64.8/67.2 | 88.18 | 25（footnote 特例，未重跑）|

**核心结论（tab_scale）**
- **selector 一致性已核验**：本任务的 per-task `bm25`(niah)/`iter_bm25`(vt) 与 locked 4B/14B/32B run 逐行一致 → 新档与已发布档**同口径可比**。（coordinator 口径为 selector=iter_bm25；此处用 per-task 是为保 tab_scale 跨档一致，已在 worker 记录标注。）
- **scale 曲线非单调（真实结果，n=500 验证非缺数据 artifact）**：0.6B(85.68) > 30B-A3B(80.48) > 1.7B(66.81) > 4B(60.52) > 14B(52.91)；14B 最弱（其 vt 塌到 ~12）。0.6B 意外最高——近满分 niah_single(~100)+强 multikey(80–91)+小模型最佳 vt(60–65)。
- **30B-A3B（MoE）** vt 为全家族最强（84–89，其余档 11–67），niah_single 已饱和(~99–100)，但 niah_multikey(51–68) 拖低均值到 80.48；与 32B(88.18) 不完全可比（32B 仅 n=25 footnote 特例）。
- **raw CSV**：`ruler_results/qcmem_scale_{0p6b,1p7b,30ba3b}_chatFALSE_ruler/qcmem_scale_<sz>_<cell>/<cell>_shardNof4.{csv,json}`（NESTED，非 flat jsonl；diskB .73/.104 共享 FS）；pool 状态 `ruler_results/_p62_scale_pool_{73,104}/`。
- **待改 section/table**：论文附录 `tab_scale`（main-only .tex，本轮不改）。

## P2.4 蒸馏后的多深度 quality--latency 曲线

- **状态**：`[DONE — 2026-08-02，.82(j6/j9) + .73(j18) diskB]`（task #122；launcher commit `ebfe475`，pooled eval launcher `.82:2d98c5a`）。**三深度训练全完 + eval 全部验收**（RULER Cohort-B 15-cell n=100/cell、LoCoMo n=1986 GPT-4o judge、固定 16k same-pack Read/Write timing，per-depth resume_j，selector=iter_bm25/topk12/hop4/chunk512/sink=bos，chat=False/no-think，bf16+SDPA+greedy seed42）。全部口径逐 run manifest-gated 验证通过（packs_paired_1to1=True，oom=0，nonfinite=0）。
  - **provenance 更正**：任务原指 `scripts/_launch_contentj_distill.sh` **仓库不存在**；coder 改用 flagship 权威记录 `outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json`（对照 log 交叉核实）逐项匹配：backbone `Qwen--Qwen3-8b` · r32/α64/dropout0 · 7-proj targets · PG-19 chunk512/n_ctx3(2048窗) · 双向 top-64 KL/λ0.6/ce0 · total4000/lr8e-5/warmup100/wd0/GA1/clip1.0 · **GC=off** · bf16/sdpa/**seed42**/8-GPU。PG-19 jsonl 在 wzc1 与 diskB byte-identical（11450766349 B）→ seed42+ws8+n_ctx3 数据顺序与 flagship 相同。（⚠️ diskB 上 `_seed1/_seed2` 副本是不同配方 n_ctx7/α32/10667步/GC-on，未采用。）
- **定位**：把“深度是可调 reuse 轴”从 `j=0/12` 两点扩展为曲线；不阻断当前投稿，优先级低于 P1.6/P1.7。
- **深度**：在旗舰 `j=12` 外新增 `j∈{6,9,18}`；每个深度单独训练与其 split 匹配的 rank-32 LoRA，不能复用 j=12 adapter 伪装成 distilled sweep。
- **严格匹配**：同 backbone revision、PG-19 数据顺序/总 tokens、4000 steps、effective batch、optimizer/LR schedule、seed、LoRA rank/alpha/target-module规则；若 target layer 数不同导致 trainable params 不同，必须报告并避免称 compute matched。
- **评测**：完整 RULER Cohort B 15 cells（n=100/cell）+ 固定 16k same-pack Read timing；推荐补 LoCoMo。每深度保存逐例 predictions、adapter hash、实际 mounted modules、pack hashes、3-process latency。
- **报告**：`(persistent bytes/token, Write cost, Read ms, RULER macro, LoCoMo)` Pareto；paired CI/McNemar；不得只选最优深度或删除负面点。
- **验收**：三个新 adapter provenance 完整；评测无 cohort 混用；质量和 latency 来自同一 config manifest；与 j=12 共同绘制完整曲线。

**结果填写**（LoRA modules/params 列 = 启动时已知，已回填；j=6 从 live log 确认，j=9/j=18 为算术预测待各自 log 确认；eval 列训练完回填）

⚠️ **非严格 compute-matched**：三深度 LoRA span 随 split 变化（layers[j:36]）→ trainable params 不同，**论文须报告逐深度参数量且不得称 compute-matched**。除参数量外（数据顺序/seed/eff-batch/steps/LR schedule/KL loss）与 flagship 完全一致。逐层 r32 LoRA(7 proj) = 2.4247M/层。

| j | LoRA modules/params | RULER B | LoCoMo | Read ms | Write ms | Raw/checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 6 | 210 / 72.74M（30 层[6:36]）| 98.29 | 40.38 | 830.3 | 133.4 | `.82:outputs/qcmem_distill_qwen_j6_r32_4k/final`（DONE）|
| 9 | 189 / 65.47M（27 层[9:36]）| 97.55 | 39.02 | 748.3 | 196.9 | `.82:outputs/qcmem_distill_qwen_j9_r32_4k/final`（DONE）|
| 12 | 168 / 58.20M | 96.07 | 38.27 | 664.4 | existing | P0.13 / flagship |
| 18 | 126 / 43.64M（18 层[18:36]）| 55.41 | 28.65 | 499.5 | 390.3 | `.73:outputs/qcmem_distill_qwen_j18_r32_4k/final`（DONE）|

**回填说明（2026-08-02，headline = 部署配置 Arm B `resume_j=j`）**：
- **RULER B** = Arm B macro（15-cell n=100，n_paired=1500）。同深度 Arm A（`resume_j=0` 全 36 层 replay 上界）≈99.20 恒定，A−B/95%CI/McNemar：j6 0.91[0.49,1.39]/p=3.05e-05·j9 1.65[1.04,2.31]/p=7.45e-09·j12 3.12[2.36,3.93]·j18 43.79[41.77,45.85]/p=6.2e-266（b/c 全为 16/0、28/0、882/0，Arm B 从不优于 Arm A）。
- **LoCoMo** = n=1986 GPT-4o judge overall。F1/EM/acc：j6 9.90/0.81/25.28·j9 9.14/0.25/24.37·j18 8.12/0.35/16.21。
- **Read/Write ms** = 固定 16k same-pack niah_single_3，3-proc 中位数（Read 三 proc 方向一致，Arm B 恒快）。read_speedup A/B：j6 1.166×·j9 1.273×·j12 1.403×·j18 1.807×。深 j 把成本从 Read 移到 Write（133→197→390ms），端到端 total speedup 恒 ~1.0（decode 主导）。
- **单调 knob 结论**：j 是 quality↔latency 单调旋钮——浅 j（j6/j9）质量甚至略高于旗舰 j12（RULER 98.29/97.55、judge 40.38/39.02），read speedup 较小（1.17×/1.27×）；深 j18 买到 1.81× read speedup 但质量坍塌（RULER 55.41、judge 28.65，distractor-heavy niah_multikey 仅 28–44%）。**非严格 compute-matched**（modules 210/189/126 vs 旗舰 168），论文须报告逐深度参数量、勿称 compute-matched。
- **artifacts**：`bench_results/p2_4_depth_quality_latency/j{6,9,18}/`（manifest/summary/stats/latency/quality per-example）+ `locomo_results/p2_4_qcmem_8b_j{6,9,18}_iter_chatFALSE/`（j6/j9 on .82，j18 on .73）。

## P2.5 跨模型验证 depth cliff / Write 误差机制

- **状态**：`[TODO — 可选，非当前提交阻断项]`
- **目的**：检验 Qwen3-8B 上的 depth frontier 与 `j=18` cliff 是否为可复用规律，而非单模型/adapter artifact。
- **最小设计**：选择一个架构不同且许可清晰的 7B/8B backbone；按相对深度取约 `0.25L/0.33L/0.5L`，先跑 frozen continuous-pack oracle 与 chunk-local Write diagnostic，不立即为每个深度训练完整 adapter。
- **升级条件**：只有 frozen/oracle 结果复现相似 knee 后，才训练最多两个深度的 interface adapter。
- **报告**：相对深度、层数、hidden size、storage/token、oracle gap、chunk-local gap、Read latency；不得从两个模型声称普适定律，只能写初步跨模型证据。

---

# 建议执行顺序

1. **完成已在跑的 P0.17**：不打断 overlap Write run；结果按预注册标准验收。
2. **P0.20 阶段 A（最高优先级新实验）**：立即用现有 BM25 排名完成 equal-latency `k` sweep，先回答“同延迟下 CoMem 是否能以更多 evidence 达到更高质量”。
3. **P1.9 + P0.20 阶段 B**：固定 BGE/E5 后复现 dense text-RAG vs dense CoMem 的 equal-latency frontier，作为正文主结果候选。
4. **P0.18 + P0.19**：完成位置/上下文拆解与 retrieval-hit/readout decomposition，解释 P0.20 的胜负来源。
5. **P1.8**：统一 repeated-query serving harness，给出 CoMem 相对 text RAG 的真实摊销区间。
6. **P1.10**：仅在 P0.18 支持可学习 Write/位置接口修复时训练；否则取消。
7. **P2.5**：资源允许再做，不阻断当前投稿。
8. 所有已有 `[DONE]` 条目不得重复运行；新实验必须复用其 cohort、manifests 和统计口径。

# 实验 agent 的统一交付要求

每个 `[DONE]` / `[NEGATIVE]` 条目必须提供：

1. raw output、checkpoint 和日志路径；
2. git commit/hash、模型 checkpoint ID/hash；
3. 完整配置、seed、数据版本、命令；
4. 实际 GPU 型号、驱动/CUDA/PyTorch、kernel、batching、warmup 和重复次数；
5. metric 实现、sample count、聚合方式和置信区间；
6. 失败/异常 run 与排除理由；
7. 一句不过度外推的结论；
8. 需要修改的 section/table/figure；
9. 若 `paperA/` 与其他稿件目录同步，必须注明源数据和最终权威版本。
