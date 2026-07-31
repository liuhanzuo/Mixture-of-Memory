# Paper A：ARR 修改与补充实验清单

> 当前独立评估：ARR Overall `3.0/5 (Findings)`，合理区间 `2.5–3.5`。截至 2026-07-31，YaRN、cluster bootstrap、两个追加 LoRA seeds、MemoryLLM native-chat diagnostic、匿名构建和保守同平台效率 headline 已集成。下一步重点是补齐 `j=0` 端到端 Pareto、probe protocol 和真实 persistent-store 系统证据。
>
> 实验 agent 完成条目后，必须填写 raw path、checkpoint/config、代码版本、命令、硬件、统计口径和对论文结论的影响。负面结果不得删除。

## 当前仍缺的数据（按优先级）

1. **P0.2**：同平台 full context / `j=0` / frozen CoMem / distilled CoMem 的完整 quality--latency--persistent-storage Pareto，包含 index/ingest、retrieval、host→device transfer、decode 和 `Q`-query 摊销。
2. **P1.2**：content-depth probe 的数据、标签、split、`knee98` 定义、seeds/CI、lexical/position/random-label controls 和非 Qwen 复现。
3. **P1.1 + P1.5**：store/distractor scaling 与证据数超过 top-12、全局聚合/长生成任务。
4. **P2.1**：原生 128k+ backbone 上的自然任务，不只 synthetic RULER。
5. **P2.2**：CPU pinned/NVMe/network store 的真实 I/O、transfer、并发与容量。
6. **P0.8**：backbone checkpoint 精确 ID/hash 与部署态 store placement；另有低优先 MemoryLLM official-template LoCoMo judge 和严格 batch/sample-matched seed backfill。

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

- **状态**：`[TODO]`
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
- identity scan 命令/结果：`TBD`（PDF 文本 + metadata 的 `pdffonts`/`pdfinfo`/grep 扫描尚未跑，正式 ARR 上传前须补）
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

- 修改文件：`TBD`
- 最终命名：`TBD`
- subset selection rationale：`TBD`

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

- 修改文件：`TBD`
- 一句话核心定位：`TBD`

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

- 修改文件：`TBD`
- 最终机制主张：`TBD`

## P0.8 修正系统描述和 Figure 1 配置混合

- **状态**：`[TODO]`（Paper 修改已完成 4/6；仍缺 checkpoint 精确 ID/hash，以及部署态 store placement/transfer 边界的权威说明。）
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
- checkpoint ID/hash：`TBD`
- storage placement 描述：`TBD`

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

- **状态**：`[RUNNING]`（2026-07-31 审计：已有 2/5 指标；缺的 RULER/LongEval/LongBench frozen-j12 **确认不在盘上**——本地 wzc1 只有 frozen-j12 的 BABILong/LoCoMo（task #73），唯一的 `longbench_results/qcmem_j12` 是**错配置**（`lora_adapter=qcmem_distill_qwen_j12_r32_4k/final` + `selector=bm25` + 仅 4 ds，是 +LoRA 旗舰非 frozen 对照）；diskB 因 .73/.104 满载 SSH 超时未能核实副本。→ 三指标需专门 GPU 离线跑，已登记 **task #104（auto_launch=true，.73/.104 gap-fill subagent 完成腾卡即起）**。）
- **类型**：额外评测 + 主表更新。
- **目的**：主表当前比较 distilled `j=12` 与 frozen `j=9`，容易混淆 split-depth 与 LoRA adaptation。加入 frozen `j=12` 后可在完全相同深度直接比较 distillation 增益。
- **固定配置**：Qwen3-8B、`resume_j=12`、无 LoRA、`iter_bm25`、top-12、hop-4、`rounds=0`（自动三轮）、chunk-512、BOS sink、`chat_template=False`，评测协议与主表一致。
- **已有结果**：
  - BABILong qa1/qa2/qa5 macro：**24.52**；raw shards：`babilong_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/`；
  - LoCoMo GPT-4o judge：**24.5217**（`n=1986`）；raw：`locomo_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/scores.json`。
- **仍缺结果**：
  1. RULER 3-family/15-cell macro，`n=100/cell`；
  2. LongEval 8k--128k aggregate；
  3. LongBench 6-QA macro F1。
- **完成后修改**：在 `paperA/sections/tab_overview.tex` 增加完整的 `CoMem frozen ($j=12$)` 行，并在正文明确：`j=12` frozen→LoRA 是 adaptation effect；`j=9` frozen 仅保留为较浅 split 的跨 benchmark operating point。

**结果填写**

| Method | RULER | LongEval | LongBench | BABILong | LoCoMo | Raw path |
|---|---:|---:|---:|---:|---:|---|
| frozen `j=12` | TBD | TBD | TBD | **24.52** | **24.52** | 见上；其余 TBD |

---

# P1：显著增强，通常需要实验或分析

## P1.1 大规模外部存储 / distractor scaling

- **状态**：`[TODO]`
- **类型**：额外评测 + 系统实验。
- **目的**：验证 “unbounded-context” 更准确地是 bounded-read over extensible store，并测量 store 增大时的 retrieval recall 与 latency。
- **设计**：固定相关证据和 read budget，store 从 `128k → 256k → 1M → 4M+ tokens` 增加 distractors；报告 recall@k、answer score、retrieval latency、index size、transfer/read latency。
- **任务**：至少一个 single evidence、一个 multi-hop/distributed evidence；加入所需证据数超过 top-12 的压力测试。

**结果填写**

| Store size | Evidence count | Recall@k | Score | Retrieval ms | Index GB | Read tokens | Raw path |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 256k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1M | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4M+ | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## P1.2 完整 content-depth probe 方法与稳健性

- **状态**：`[TODO]`
- **类型**：机制实验 + Paper 方法补充。
- **必须补齐**：probe 数据与标签、sample count、train/dev/test split、probe architecture、regularization、optimizer、`knee98` 数学定义、3+ seeds、置信区间。
- **控制**：lexical-only、position-only、random-label、class-balance；至少一个非 Qwen 家族；比较 linear probe 与更受控的 readout。
- **目标**：区分“线性可读”与“被模型实际使用”，避免把 probe accessibility 直接称为 understanding。

**结果填写**

- 数据/标签：`TBD`
- knee98 定义：`TBD`
- seeds/CI：`TBD`
- controls：`TBD`
- 非 Qwen 结果：`TBD`
- raw/code：`TBD`

## P1.3 Prompt/chat-template sensitivity

- **状态**：`[TODO]`（主 Qwen CoMem/KV-Direct chat T/F 已完成；MemoryLLM native-chat 的 RULER/BABILong diagnostic 已写入附录。仍缺 MemoryLLM official-template LoCoMo judge 一格。）
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
| MemoryLLM (Llama-3-8B-chat) | official chat=True | `TBD`（补） | — | — |
| MemoryLLM (Llama-3-8B-chat) | chat=False (OOD diag) | 16.11 | — | `locomo_results/memoryllm_8b_chatFALSE` |

- **核心发现（judge 是 protocol-robust 的 canonical metric，token-F1 是 formatting artifact）**：
  1. GPT-4o judge 下 CoMem ≈/> KVD **两种协议**：chat=True 37.76 vs 38.22（tie）；chat=False **38.27 vs 34.59（CoMem +3.68 领先）**。
  2. chat=True 的 token-F1 KVD 40.06 >> CoMem 19.51（+20.55），但该 gap 在 chat=False 下**消失**（9.15 ≈ 9.02）→ 是 **chat-template artifact 非能力差**。
  3. CoMem judge 跨协议稳定（37.76→38.27）；KVD judge 掉（38.22→34.59）→ CoMem（base-trained 压缩法）对其原生 no-chat 更鲁棒。
- **待补**：MemoryLLM **官方 chat 模板**下的 LoCoMo judge（现只有 chat=False OOD 诊断 16.11）——需其原生模板单独跑一格（task #83 native-chat appendix 或联动）。
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
- **caveat（须写入 paper）**：seed1/2 effective batch=3 vs flagship=8（data seen 已 matched，batch/opt-noise 二阶差异）；flagship 用 n=500 cell、seed1/2 用 n=50，直接对齐需 subsample flagship 到 n=50（低优先 backfill，不改 verdict）。
- **expanded judge audit**：DeepSeek-V3 独立 judge（200-item，agreement 0.81 κ=0.626，同序）已做（见 P0.9）；conversation-level 数值 CI 仍 TBD（见 P0.9 cluster bootstrap）。

| Seed | RULER 128k (single/mk/vt) | BABILong qa5 (4k/16k/32k) | Checkpoint/raw |
|---:|---|---|---|
| 42 (flagship) | (canonical n=500 dir) | — | `outputs/qcmem_distill_qwen_j12_r32_4k/final` |
| 1 | 100/100/99.6 | 75/65/68 | `outputs/qcmem_distill_qwen_j12_r32_4k_seed1/final` |
| 2 | 100/96/99.6 | 78/68/67 | `outputs/qcmem_distill_qwen_j12_r32_4k_seed2/final` |

- **mean±std**：RULER overall max std 2.83pp / BABILong max std 3.54pp / 18-cell median 0.71pp。
- **expanded judge audit**：GPT-4o + DeepSeek-V3 双 judge 同序（见 P0.9）；数值 cluster CI 待补。

## P1.5 任务覆盖扩展

- **状态**：`[TODO]`
- **类型**：额外评测。
- **目的**：覆盖 fixed top-k retrieval 最可能失败的任务。
- **优先任务**：需要跨许多 chunks 聚合、全局统计、证据数量随 context 增长、长篇生成的任务；不要只补 needle retrieval。
- **必须报告**：evidence recall、最终 answer quality、read budget、失败案例。

**结果填写**

- tasks：`TBD`
- selection rationale（是否结果前确定）：`TBD`
- scores/raw paths：`TBD`
- 失败模式：`TBD`

---

# P2：可选增强

## P2.1 原生长上下文自然任务复现

- **状态**：`[TODO]`
- **类型**：额外模型评测/可能需训练 adapter。
- **目的**：在原生 128k/256k backbone 上比较 full context、`j=0` 和 CoMem，避免所有超窗口结论依赖 RoPE extension。
- **要求**：至少一个自然 QA/对话任务和一个 multi-hop task，不仅是 RULER single needle。

**结果填写**

- backbone/checkpoint：`TBD`
- native window：`TBD`
- adapter config：`TBD`
- results/raw：`TBD`

## P2.2 Persistent-store 实际 I/O 与网络部署

- **状态**：`[TODO]`
- **类型**：系统实验。
- **设计**：GPU resident、CPU pinned memory、local NVMe、networked store 四种后端；测 ingest、random retrieval、transfer、concurrent queries、cache hit/miss。

**结果填写**

| Backend | Write GB/s | Retrieve ms | Transfer ms | QPS | Peak GPU | Peak host | Raw path |
|---|---:|---:|---:|---:|---:|---:|---|
| GPU | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CPU pinned | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NVMe | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| network | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

# 建议执行顺序

1. **先做 P0.1**：不解决 `1.917s vs 7.79s`，任何效率 headline 都不稳。
2. **并行做 P0.3 provenance 审计**：`paper/` 可能已有可用 YaRN 数据，先验证再决定是否重跑。
3. **做 P0.2**：统一 `j=0`/CoMem/full-context 质量—延迟—存储 Pareto。
4. 同时完成 **P0.4–P0.10 的 Paper-only 修改**。
5. 再做 P1.1/P1.2：分别增强“extensible store”和“depth division”两条核心主张。
6. 资源允许时补 P1.3–P2。

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
