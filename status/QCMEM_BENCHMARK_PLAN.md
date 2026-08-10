# QCMem (CoMem) Paper — Benchmark 计划表（协作分工用，全量方案）

> 建立 2026-07-14。QCMem 用 **instruct 模型**（需 follow instruction）。**Qwen3 全家做 scale，全 benchmark 全量跑，每个 scale 都测全 baseline。**
> 用法：认领后在「Owner」列写名字，跑完在状态列更新（✅=n100完成 / 🟡=部分 / ⬜=待测 / ⏳=模型/harness未就位）。

---

## 0. 固定协议（所有 cell 统一，保证同口径）

| 项 | 规定 |
|---|---|
| 模型 | **Qwen3-Instruct 全家 7 档**：0.6B / 1.7B / 4B / 8B / 14B / 32B / 30B-A3B(MoE) |
| split-depth **j** (双 j，2026-07-16 定案，覆盖旧 ≈0.33L) | **zero-shot 用 readout-safe j**（single≥90 最深，随 scale 变深）：0.6B→**2**(0.07L) / 1.7B→**3**(0.11L) / 4B→**9**(0.25L) / 8B→**9**(0.25L) / 14B→**13**(0.325L) / 32B→**27**(0.42L)。**+adapter 用 content-j ~0.45L**（probe 语义峰）：0.6B/1.7B→13 / 4B/8B→16 / 14B→18 / 32B→27 / 30B-A3B→~22。旧固定 ≈0.33L/j3 作废，见 `status/QCMEM_J_DETERMINATION.md`。 |
| chunk_size | 512 |
| selector | 主 `bm25`；vt 加 `iter_bm25`(hop4,k16)；对照 `reader_attn`/`oracle`/`recency` |
| topk | 主 12（RULER 网格扫 4/8/12/16/24） |
| adapter | 自蒸馏 LoRA（in-window 追平 dense）；报告标注 with/without |
| 样本 n | **RULER/BABILong = n=500/cell**（对齐官方 RULER，2026-07-14 用户定）；LongBench/LoCoMo/LongMemEval/∞Bench/HELMET = **全测试集**（固定大小，无 n 选择）；sanity n=30。⚠️现有 8B 的 RULER/BABILong 是旧 n=100，需**重跑到 n=500** |
| 判分 | RULER=`string_match`(官方)；BABILong=`TASK_LABELS`+`compare_answers`(禁 re.search)；LongMemEval/LoCoMo=QA acc(judge/EM) |
| 长度档 | 合成(RULER)：8k/16k/32k/64k/128k(**含超窗口=卖点**)；BABILong：0k-32k；真实任务按数据集原长 |

## 0b. 模型就位状态（下载队列）
| 模型 | L | j | 位置 | 状态 |
|---|---|---|---|---|
| Qwen3-0.6B | 28 | 9 | 本地+diskB | ✅就位 |
| Qwen3-1.7B | 28 | 9 | 本地+diskB | ✅就位 |
| Qwen3-4B | 36 | 12 | — | ⏳需下载 |
| Qwen3-8B(instruct) | 36 | 12 | 本地+diskB | ✅就位(主) |
| Qwen3-14B | 40 | 13 | diskB | ✅就位 |
| Qwen3-32B | 64 | 21 | diskB | ✅就位(实测用 j3) |
| Qwen3-30B-A3B(MoE) | 48 | 16 | — | ⏳需下载(~60GB) |

---

## 1. 主表：RULER × 模型（framing A — 双 j 分列，2026-07-16 重建）

> **双 j 口径**（依据 `status/QCMEM_J_DETERMINATION.md`，覆盖旧固定 0.33L/j3）：
> - **zero-shot 行** = per-model **readout-safe j**（zero-shot readout 不塌、single-recall 近满 ≥90 的最深 j；随 scale 变深：0.6B j2 / 1.7B j3 / 4B j9 / 8B j9 / 14B j13 / 32B j27）。
> - **+adapter 行** = 目标 **content-j（~0.45L，probe 语义峰）**；本轮先填**现有 ~0.33L adapter**（标注实际 j），**content-j adapter（0.6B j13 / 1.7B j13 / 4B j16 / 8B j16 / 14B j18）训练中，出来后更新本列**。
> 判分 RULER=`string_match`，n=100，chunk512，selector：single/multikey=bm25、vt=iter_bm25（固定多跳）。

| model | config | niah_single 8k/16k/32k | niah_multikey 8k/16k/32k | vt 8k/16k/32k | **vs-Dense 128k 超窗口**(Dense→QCMem, single) |
|---|---|---|---|---|---|
| **0.6B** (L28) | zero-shot @ **j2** (0.07L) | 100/100/100 | 77/89/79 | 58/67/79 | 0→**56** |
| | +adapter @ content-j (**j13**，single↑ 但 mk/vt<zs) | 95/98/99 | 24/24/22 | 0/4.8/0 | — |
| **1.7B** (L28) | zero-shot @ **j3** (0.11L) | 97/100/100 | 53/40/33 | 53/64/60 | 0→**83** |
| | +adapter（现有 ~j9, 0.33L） | —(zs 已高) | 56/39/20 | 62/62/66 | — |
| **4B** (L36) | zero-shot @ **j9** (0.25L) | 93/98/94 | 38/35/41 | 58/61/54 | 0→**99** |
| | +adapter（现有 ~j12, 0.33L） | —(zs 已高) | 95/97/94 | 96/96/97 | — |
| **8B** (L36) | zero-shot @ **j9** (0.25L) | 100/97/99 | 42/36/31 | 46/42/39 | 0→**100** |
| | **+adapter @ j12** (0.33L, 已验证) | 100/100/100 | **91/91/92** | **97/97/98** | — |
| **14B** (L40) | zero-shot @ **j13** (0.325L) | 99/89/98 | 51/51/11 | 18/15/11 | 11→**100** |
| | +adapter（现有 ~j13, 0.33L） | 100/100/100 | **100/99/99** | **99/100/100** | — |
| **32B** (L64) | zero-shot @ **j27** (0.42L) | 100/100/100 | 98/88/86 | 42/33/37 | **OOM→100** |
| | +adapter | 几乎不需（gap~0，readout 已达 content 峰） | — | — | — |

> vs-Dense 128k 超窗口列 = 128k niah_single 上 **Dense（Dense=full-context baseline，超窗口 崩0/OOM）→ QCMem（存活）**，QCMem 显存恒定（详见 §1c 速度附表 + §1a）。multikey 同向：14B 5→98 / 32B OOM→98。这是 QCMem **全 scale 通用的核心卖点**（超窗口唯一可用）。
> **★ 32B 的 split-j 两套并记（方法不同，结论不同，team 待定论文口径）**：**我们=j27（0.42L，truncation+probe 语义峰 + readout cliff-bracket 反核不塌）**；**collaborator=j16（0.25L，intrinsic PPL/KL/top1 SlimPajama sweep + needle probe）**，其 accuracy-first 备选 j12。两者数值不冲突（不同文件/方法），主表 zero-shot 现用我们的 j27。

**★ 3 深度故事**（三个 j 分离，详见 `status/QCMEM_J_DETERMINATION.md`）：
1. **content 深度 ~0.45L 近 scale-invariant**（probe：0.42–0.48L，均值 ~0.45L）—— 语义信息最富的"可缓存深度"上限，跨 scale 稳定。
2. **zero-shot readout 崩点随 scale 变深，NOT scale-invariant**：0.6B ~0.09L → 1.7B ~0.22L → 4B/8B ~0.30L → 14B ~0.375L → 32B >0.42L（无崩）。这是 zero-shot 报告 j 的上限——超过则 single recall 塌。
3. **gap = content − readout = adapter 要补的缺口，随 scale 缩小到 32B~0**（0.6B ~0.39L 巨大 → 14B ~0.085L → 32B ~0）。**小模型 adapter 价值最大；32B readout 几乎已到 content 峰，几乎不需 adapter。**

> ⚠️ **zero-shot 行读法**：readout-safe j 按 **single recall ≥90** 选深；此深度上 **multikey/vt（更硬）可能已衰减**（尤 14B：j13 single 99/89/98 尚可，但 mk 51/51/11、vt 18/15/11 已塌）—— 这正是 adapter 要补的 gap（14B +adapter：mk 100/99/99、vt 99/100/100）。0.6B/32B 因 gap 小，zero-shot mk/vt 也高。
> ⚠️ **adapter 行现状**：填的是**现有 ~0.33L adapter**（8B@j12 已验证；1.7B/4B/14B 标注 ~0.33L，其 zero-shot 对照基线为旧 recall-optimal j，故绝对增益口径与 zs 行 readout-safe j 略有错位）。**content-j（~0.45L）adapter 训练中，出来后替换本列。**
>
> **旧 j3/0.33L 数值保留为参考**（勿删）：旧 recall-optimal ~j3 = **recall 上界参考**（mk/vt 在浅 j3 更高，如 14B mk 98/95/90、vt 89/82/96；32B/30B-A3B 主 benchmark 亦用 zs j3）；旧固定 0.33L（8B j12/32B j21）= **保守下界**（~95% 语义 knee95）。见下补充表与 RUN_REGISTRY。

### 1a. 补充 benchmark（非 RULER，沿用原 zero-shot j3/j9 或 adapter；详数见 RUN_REGISTRY 大表）

> 下表数值来自各 scale 原 benchmark run（zero-shot 多为 recall-optimal j3/j9、8B/14B 另有 adapter 臂），**未按 readout-safe j 重跑**，仅作横向参考。

| Benchmark | 0.6B(zs) | 1.7B(zs) | 4B(zs) | **8B zs / +adapter** | **14B zs / +ad** | 32B(zs,j3) | 30B-A3B |
|---|---|---|---|---|---|---|---|
| **j (used, zero-shot)** | **3** | **4** | **9** | **9** zs / **12** +ad | **3** zs / **13** +ad | **3** | **12** |
| **BABILong** qa1/2/5 官方 | ✅11.0 | ✅34.2 | ✅49.3 | ✅**48.4** / **57.1**〔clean〕 | ✅32.7 / **46.6** | ✅41.7 | ✅**32.3** |
| **BABILong CLEAN**〔zs, readout-safe j, chat+no-think+iter_bm25, T24 2026-07-20〕| **33.1**(j2) | **41.2**(j3) | **46.7**(j9) | **48.4**(j9) | **54.3**(j13) | **64.1**(j27) | —(未跑) |
| **BABILong +ad content-j** overall〔+adapter content-j ~0.45L, chat+no-think+iter_bm25, 21-cell 均值, T27b 2026-07-21〕| **28.2**(j13,↓4.9) | **38.6**(j13,↓2.6) | **49.1**(j16,↑2.4) | **49.2**(j16,↑0.8) | **52.8**(j18,↓1.5) | —(无adapter) | —(无adapter) |
| **LongBench** qa_f1 | ✅4.51 | ✅6.07 | ✅8.51 | — / **9.76** | ✅9.63 | ✅12.37 | ✅6.61 |
| **LongBench CLEAN** AVG-F1〔zs, readout-safe j, chat+no-think+iter_bm25, 6-ds, T22 2026-07-20〕| **20.3**(j2) | **21.3**(j3) | **28.8**(j9) | **26.3**(j9) / — | **31.3**(j13) / — | **36.3**(j27) | —(未跑) |
| **LongBench +ad content-j** MACRO-F1〔+adapter content-j ~0.45L, chat+no-think+iter_bm25, 6-ds n=200, T27b 2026-07-21〕| **15.98**(j13,↓4.3) | **22.96**(j13,↑1.7) | **30.35**(j16,↑1.6) | **34.05**(j16,↑7.8) | **37.15**(j18,↑5.9) | —(无adapter) | —(无adapter) |
| **LongEval** | ✅ | ✅ | ✅ | ✅ / **92/71/74/65** | ✅99/99/97/100 | ✅99/100/99/100 | ✅43/30/-/28 |
| **LongEval CLEAN** AVG-acc〔zs, readout-safe j, chat+no-think+iter_bm25, 6 长档 4k-128k n=100, T26 2026-07-20〕| **37.3**(j2) | **15.8**(j3) | **15.2**(j9) | **7.5**(j9) | **20.3**(j13) | **97.5**(j27) | —(未跑) |
| **LongEval +ad content-j** AVG-acc〔+adapter content-j ~0.45L, chat+no-think+iter_bm25+max_new48, 6 长档 4k-128k n=100, T27 2026-07-20〕| **1.2**(j13,↓36) | **0.8**(j13,↓15) | **15.5**(j16,≈) | **13.0**(j16,↑5.5) | **40.2**(j18,↑20) | —(无adapter) | —(无adapter) |
| **LoCoMo** | 🟡跑完待聚合 | 🟡 | 🟡 | 🟡 / 🟡 | ✅acc1.4/F2.2 | ✅acc6.6/F4.1 | ✅acc7.4/**F5.0** |
| **LoCoMo CLEAN** F1/acc〔zs, readout-safe j, chat+no-think+iter_bm25, n=1986, T22 2026-07-20〕| **F12.2/a14.8**(j2) | **F9.8/a16.4**(j3) | **F12.6/a20.9**(j9) | **F14.3/a22.4**(j9) / — | **F13.9/a22.3**(j13) / — | **F17.3/a27.6**(j27) | —(未跑) |
| **LoCoMo +ad content-j** F1/acc〔+adapter content-j ~0.45L, chat+no-think+iter_bm25, n=1986, T27b 2026-07-21〕| **F9.97/a9.26**(j13,↓2.2/↓5.5) | **F8.33/a11.2**(j13,↓1.5/↓5.2) | **F14.2/a19.3**(j16,↑1.6/↓1.6) | **F16.7/a22.1**(j16,↑2.4/≈) | **F18.9/a26.1**(j18,↑5.0/↑3.8) | —(无adapter) | —(无adapter) |
| **RULER CLEAN** string_match s/mk/vt〔zs, readout-safe j, chat+no-think+iter_bm25, CORE 8-32k n=100 均值, T25 2026-07-20；64k/128k 跑中〕| **100/85/81**(j2) | **40/24/48**(j3) | **54/36/20**(j9) | **97/41/2**(j9) | **15/39/1**(j13) | **99/94/37**(j27) | —(未跑) |
| **RULER +ad content-j** s/mk/vt〔+adapter content-j ~0.45L, chat+no-think+iter_bm25, CORE 8-32k n=100 均值, T27 2026-07-20〕| **98/30/1**(j13,mk↓55/vt↓80) | **87/29/21**(j13,s↑47/vt↓27) | **76/44/49**(j16,全↑) | **99/42/12**(j16,vt↑10) | **98/73/19**(j18,s↑83/mk↑34/vt↑18) | —(无adapter) | —(无adapter) |
| **vs-Dense 128k崩塌** | ✅0→56 | ✅0→83 | ✅0→99 | — / **0→100** | ✅s11→100 mk5→98 | ✅s OOM→100 mk OOM→98 | ✅s OOM→100 / mk ⏸未完(卡被OLMo占,待空卡续) |
| **topk ablation** | — | — | ✅(tk4>tk32) | ✅**tk4=98最优,越大越差** | — | — | — |
| **LongMemEval/∞Bench/HELMET** ⏸**暂不评（需 API/GPT-4o judge，用户 2026-07-16 定）** | — | — | — | — | — | — | — |
| **vs-Dense 速度**（128k prefill 加速×；QCMem 显存跨长度恒定） | ✅103× | ✅69× | ✅80× | ✅57× | ✅50× | ✅Dense**OOM** | ✅Dense**OOM** |

> **判分口径 & 暂不评清单（用户 2026-07-16 定）**：
> - **✅ RESOLVED — 全 scale zero-shot LongBench + LoCoMo CLEAN 曲线补齐（T22，2026-07-20，节点 .73）**：T24 BABILong clean 曲线的 **real-QA 延伸**，把 zero-shot clean 口径（chat+no-think+iter_bm25，readout-safe j：0.6B j2/1.7B j3/4B j9/8B j9/14B j13/32B j27）从合成 BABILong 扩到 **LongBench(6-ds real-doc QA, n=1150/scale) + LoCoMo(多会话记忆, n=1986)**，与 BABILong scale 曲线同口径可比。判分：LongBench=官方 `eval_longbench_mem_space.run_scoring`（compute_f1_multi/em，禁 re.search）；LoCoMo=项目 F1/EM/acc（无 GPT-4o judge）。**每 scale：LongBench 1150 + LoCoMo 1986 records，全 empty_output=0，well-formed**。来源：**8B = 已有 clean dir 仅打分**；**0.6B/1.7B/4B/14B/32B = 全部新 GPU 重跑**（旧 `qcmem_{14b,32b,1p7b_j*,0p6b_j*,4b*}` dir 经 config 核实三轴皆污染 bm25/chat_off/浅 j → 弃用）。**LongBench AVG F1 整体随 scale 升、32B 最优**：0.6B 20.3→1.7B 21.3→4B 28.8→8B 26.3(唯一非单调点，<4B)→14B 31.3→32B **36.3**。**LoCoMo acc 近单调**：14.8→16.4→20.9→22.4→22.3→**27.6**（F1 更 noisy：12.2→9.8→12.6→14.3→13.9→**17.3**，32B 最优）。**结论**：两 real-QA 基准都印证「scale up 有帮助/32B 全面最优」，趋势与 BABILong 一致，但 real-doc token-F1 比合成 exact-match noisier（LongBench 8B<4B、LoCoMo 1.7B/14B 小回落），单调性弱于 BABILong。逐 cell grid 见 `RUN_REGISTRY.md`「T22 全 scale LongBench+LoCoMo clean 曲线」。数据：`.73:{longbench,locomo}_results/qcmem_{0p6b,1p7b,4b,8b,14b,32b}_zs_iter_chatnothink`。
> - **✅ RESOLVED — 8B BABILong clean chat+no-think + iter_bm25 重跑完成并回填（T23，2026-07-20，节点 .73）**：干净口径 = `--use_chat_template` + `enable_thinking=False`(no-think) + `selector=iter_bm25`（2026-07-17 统一标配）。官方 `babilong.metrics.compare_answers`+TASK_LABELS（首句 + 恰好一个 label 判分），4-shard 求和合并（`score_nested_babilong.py`），n=100/cell，**全 21 cell empty_output=0，输出 well-formed**（抽查 adapter qa1 16k：干净单答 "Mary is in the garden." 类=真检索命中/失误，非 thinking ramble）。**8B 干净 overall(21 cell)：zero-shot(j9)=48.4 / +adapter(j12)=57.1**。数据：`.73:babilong_results/qcmem_j9_iter_bm25_chatnothink_zs`（zs）+ `qcmem_j12_iter_bm25_chatnothink_ad`（adapter）。
>   - **vs 旧 2026-07-14 污染版**（bm25 + chat_template_no + thinking-era；`qcmem_8b_zeroshot_babilong` j12 / `qcmem_8b_adapter_babilong_mid` j12）：**zero-shot 39.2→48.4（+9.2）**、**adapter 55.5→57.1（+1.6，基本持平）**。增益集中在 **0k–8k**（chat+no-think 让指令跟随更干净：如 zs qa2 1k 14→53、qa5 8k 38→57）。
>   - **⚠️ 修正旧估计**：旧 caveat 猜 "32k 真值 35–50" **不成立**——干净口径下 32k qa1/qa2 仍很低（adapter 27/6、zs 1/1），仅 qa5 32k 保持 ~41–61。thinking 污染对 32k 的压低幅度被**高估**；QCMem 在 32k 的 qa1/qa2 是**真·长程检索失败**（非判分 artifact）。
>   - **⚠️ NEW selector caveat（iter_bm25 vs bm25）**：统一 iter_bm25 对 **qa1 单事实检索中档反而掉分**（adapter qa1 16k 旧 bm25=55 → 干净 iter_bm25=23；输出 well-formed 单答=真检索失误，非格式坏）。iter_bm25 多跳 BFS 对 single-fact QA 会过度扩召、摊薄 topk 预算 → qa1 mid 掉。bm25 对 qa1 更优，但 iter_bm25 是统一口径标配。~~旧 4B/1.7B/0.6B 未重跑~~ → **全 scale 已重跑 clean（见 T24，2026-07-20）**。
> - **✅ RESOLVED — 全 scale zero-shot RULER CLEAN scale-consistency 曲线补齐（T25，2026-07-20，节点 .73）**：补齐 **4-benchmark cross-scale 故事的第 4 条曲线**（BABILong T24 / LongBench+LoCoMo T22 已完成），全 6 scale × 3 task × **5 长档(8k/16k/32k/64k/128k)** 完成。完全同 clean recipe（chat+no-think+iter_bm25，readout-safe j：0.6B j2/1.7B j3/4B j9/8B j9/14B j13/32B j27，原始 hybrid Qwen3 非 2507）；判分 = 官方 `scripts.eval_ruler_mem_space._string_match_all_one`（`string_match`，禁 re.search）；任务 = niah_single_2/niah_multikey_1/variable_tracking。CORE 8k/16k/32k n=100（8-shard 合并）、超窗 64k/128k n=50。**全 90 cell empty_output=0（抽查 32B/8B/14B 128k 共 300 records 全非空、well-formed：8B VT 真理解任务但多跳链未重建、14B niah_single 命中 rec=1.0/失误改述干草堆=真 readout drift；均非空/非 ramble），oom=0，total_n=1200/scale**。去重：0.6B/1.7B/4B/8B 的 niah_single/niah_multikey @8-32k = 旧 clean dir 复用仅打分（三轴 iter_bm25/chat/j 核实干净）；variable_tracking @全长 全 scale 新跑（旧 VT 轴 chat=false 污染）；14B/32B @8-32k 全新跑；64k/128k 全新跑。**★ 与 3 姊妹曲线明显不同——RULER zero-shot@readout-safe-j 强非单调**：niah_single 0.6B **100**/8B 95-98/32B **98-100** 三点近满但中段 1.7B/4B/14B 大塌（14B@j13 全长 0-28、128k=0 = readout drift）；VT **tiny 0.6B 最强(70-92)**（浅 j 保留 VAR token 链）、8B/14B(≈0-4) 几乎全崩、32B(27-43) 部分恢复；唯一稳健信号 = **32B 全 task 最强 + 全长档最稳**（32B niah 64k/128k 仍 89-100，dense 128k 必 OOM 处稳定跑通 = RULER 侧印证固定读卖点）+ 0.6B 因浅 j 在字面任务反常强。机制 = RULER exact-match 对「readout-safe j 是否保留原始 token」极敏感，该 j 按 single-recall 标定不迁移到 multikey/VT 且中段 scale drift。运维：CORE 由本 agent phase1 跑完，64k/128k 由 phase2 ext(0.6/1.7/4B) + recovery task-pool `t25_pool.sh`(8B/14B/32B，phase2 driver 4B 后意外退出后同口径同目录接管) 跑完。逐 cell grid 见 `RUN_REGISTRY.md`「T25 全 scale RULER CLEAN 曲线」。数据：`.73:ruler_results/qcmem_{scale}_zs_ruler_iter_chatnothink`。
> - **✅ RESOLVED — 全 scale zero-shot BABILong CLEAN 曲线补齐（T24，2026-07-20，节点 .73）**：8B clean(T23) 的直接延伸，把 0.6B/1.7B/4B/14B/32B 补到与 8B(48.43) **完全一致口径**（chat+no-think+iter_bm25，zero-shot，每 scale 用 §1 主表 readout-safe j：0.6B j2/1.7B j3/4B j9/8B j9/14B j13/32B j27）。**overall(21 cell) 单调随 scale 上升：0.6B 33.1 → 1.7B 41.2 → 4B 46.7 → 8B 48.4 → 14B 54.3 → 32B 64.1**。官方 `compare_answers`+`score_nested_babilong.py`（8-shard 合并），**全 scale 各 21 cell empty_output=0（2100 records/scale）**，输出 well-formed。来源：0.6B/4B 仅打分已有 clean dir（`qcmem_{0p6b,4b}_zs_iter_chatnothink`）；**1.7B/14B/32B = 新 GPU 重跑**（`qcmem_{1p7b,14b,32b}_zs_iter_chatnothink`）。见 `RUN_REGISTRY.md`「T24 全 scale 曲线」逐 cell grid。⚠️ clean 曲线用 readout-safe j（非旧 recall-optimal 浅 j），与本 §1a 旧 legacy BABILong 行（j3/j4/j3）逐 cell 不直接可比；clean 曲线内部严格同口径。
> - **✅ RESOLVED — 全 scale zero-shot LongEval CLEAN 曲线补齐（T26，2026-07-20，节点 .73）＝ 5-benchmark cross-scale 故事的第 5 条（最后一条）曲线**：BABILong(T24) / LongBench+LoCoMo(T22) / RULER(T25) 已完成，本次补齐 LongEval。完全同 clean recipe（chat+no-think+iter_bm25，readout-safe j：0.6B j2/1.7B j3/4B j9/8B j9/14B j13/32B j27）；driver `scripts/eval_qcmem_longeval.py` 官方 LongEval accuracy（line-key→register-content 精确抽取匹配，禁 re.search）；6 scale × 6 长档(4k/8k/16k/32k/64k/128k) = **36 cell，n=100/cell**（8-shard 合并；8B 4k-32k 复用旧 clean of4 run〔config 核实 iter_bm25/chat/j9 干净〕+ 64k/128k of8 新跑，其余 5 scale 全 clean 重跑）。**全 36 cell empty=0 oom=0，well-formed**（抽查 32B 16k：expected 153333→output "…is <153333>"→pred=153333 correct=True=真检索命中；read_len≈6315 恒定=固定读、n_selected_chunks=12=topk12）。逐 cell grid（accuracy%，长档 4k/8k/16k/32k/64k/128k）：0.6B 58/28/41/25/35/37 · 1.7B 33/11/4/12/20/15 · 4B 30/11/16/15/8/11 · 8B 14/6/5/7/6/7 · 14B 36/13/16/14/18/25 · **32B 99/97/97/98/97/97**（AVG-acc：0.6B 37.3 / 1.7B 15.8 / 4B 15.2 / 8B 7.5 / 14B 20.3 / **32B 97.5**）。**★ 与 RULER 同型（强非单调，非 BABILong/LongBench/LoCoMo 的单调随 scale 升）**——LongEval = 按 line-key 检索精确 register 值 = exact-match 字面任务：①**32B(j27) 碾压且长度无关（97-99 全档，128k≈4k，dense 128k 会 OOM）** = 跨 5 benchmark 稳健信号（32B 全 benchmark 最优）+ 固定读长度不变卖点；②**0.6B(j2) 反常强（25-58）** vs 中段 = 浅 j2 保留字面 REGISTER token（同 RULER 机制）；③**中段塌陷、8B 最弱(5-14)** = readout-safe j9 在 exact-match 字面任务丢原始 token（readout drift，同 RULER 8B/14B niah）。运维：T26 agent a1f86c58 完成 GPU eval（36 cell）后在 well-formed 抽查阶段意外终止未落账；本 heartbeat 独立复算全 grid（of8 逐-length 求和 + 8B of4 短档修正，铁律2 核实非空 well-formed）并回填。逐 cell grid 亦见 `RUN_REGISTRY.md`「T26 全 scale LongEval CLEAN 曲线」。数据：`.73:longeval_results/qcmem_{0p6b,1p7b,4b,8b,14b,32b}_zs_iter_chatnothink`。
> - **✅ RESOLVED — +adapter content-j 曲线（RULER + LongEval 两 exact-match 基准）补齐（T27，2026-07-20，节点 .73）＝ T25/T26 zero-shot readout-safe-j 曲线的 content-j 配对臂**：核心论文问题——**蒸馏 content-j LoRA adapter 能否修复 zero-shot readout-safe-j 在字面 exact-match 任务上的塌陷**（zs 已知塌：8B LongEval 5-14、RULER VT≈0-4、14B niah≈0-28）。完全同 clean recipe（`--use_chat_template` + no-think + `--selector iter_bm25`，topk12/chunk512/sink=bos/bf16），加载各 scale content-j LoRA adapter（0.6B/1.7B→j13、4B/8B→j16、14B→j18；32B/30B-A3B 无 content-j adapter 跳过）。RULER driver `eval_ruler_qcmem.py`（官方 `_string_match_all_one` string_match，禁 re.search，max_new_tokens=48），任务 niah_single_2/niah_multikey_1/variable_tracking @8k/16k/32k n=100（8-shard 合并）。LongEval driver `eval_qcmem_longeval.py`（官方 line-key→register 精确抽取，禁 re.search），4k-128k n=100（8-shard 合并）。**⚠️ LongEval 需显式 `--max_new_tokens 48`**（driver 默认 16 会截断 adapter 冗长 chat 输出 "The <REGISTER_CONTENT> in line X is …" → number 未出 → pred=''；48 = 统一协议值，与 T26 zs 输出同样冗长完整、口径一致，RULER 默认已 48）。**全 RULER 15 cell + LongEval 30 cell empty=0、n=100，铁律2 OK**（抽查 8b LongEval 4k：output "…is **<759233>**." pred 抽取成功、correct 为真值匹配=真检索命中/失误）。**★ 核心发现：adapter 修复能力随 scale 涌现（monotone scale-dependent），非普适**：
>   - **小模型（0.6B/1.7B）adapter 反而伤字面 exact-match**：LongEval 0.6B 37.3→1.2(↓36)、1.7B 15.8→0.8(↓15)；RULER 0.6B mk 85→30/vt 81→1、1.7B vt 48→21（虽 1.7B niah 40→87 因 adapter 反升）。深 content-j 丢字面 token，浅 readout-safe zs 远好，蒸馏 adapter **无法修复反倒破坏**。
>   - **中段（4B）break-even→helpful**：LongEval 15.2→15.5(≈)；RULER 全 task↑（niah 54→76、mk 36→44、vt 20→49）。
>   - **大模型（8B/14B）adapter 明显修复 zs 塌陷**：LongEval 8B 7.5→13.0(↑5.5)、14B 20.3→40.2(↑20，翻倍)；RULER 14B 最戏剧（niah **15→98**、mk 39→73、vt **1→19**，把 j13 塌陷完全救回），8B vt 部分修复（2→12，16k 4→25）。
>   - **结论**：content-j adapter 对字面 exact-match 的修复价值 **随 scale 单调涌现**——大模型（8B/14B）YES（尤 14B 把 readout-safe-j 塌陷 niah 15→98 完全修复），小模型（0.6B/1.7B）NO（深 j 破坏字面检索，adapter 训不回来）。这与合成/real-QA（BABILong/LongBench/LoCoMo）上 adapter 普遍有益的图景**不同**——字面 exact-match 更依赖浅层原始 token，深 content-j 是双刃剑，仅大模型有足够容量在深层重建字面读出。逐 cell grid 见 `RUN_REGISTRY.md`「T27 +adapter content-j 曲线」。数据：`.73:{ruler_results/qcmem_{scale}_adapter_contentj{j}_iter_chatnothink, longeval_results/同名}`。
> - **`j (used, zero-shot)` 行（2026-07-16 加）**：标明本 §1a 各非-RULER benchmark（BABILong/LongBench/LongEval/LoCoMo/vs-Dense）实际用的 split-depth j，逐列从结果目录名 + `eval_config.json` 的 `resume_j` 核实：0.6B=**j3** / 1.7B=**j4** / 4B=**j9** / 8B=**j9(zs)·j12(+adapter，`qcmem_distill_qwen_j12_r32_4k`)** / 14B=**j3(zs)·j13(+adapter，`qcmem_distill_14b_j13_r32`)** / 32B=**j3** / 30B-A3B=**j12**。均为各 scale 的 recall-optimal 浅 j（非 §1 双 j 主表的 readout-safe / content-j）；同一模型 5 个 benchmark 的 j 一致。⚠️ RUN_REGISTRY 旧注误把 30B-A3B 归为 zs j3，实测全 benchmark 均 **j12**（已在 RUN_REGISTRY 更正）。> - **LoCoMo 用 F1**（token-level SQuAD-F1，与 LoCoMo 原论文口径可比）；`eval_qcmem_locomo.py` 的 substring "acc" 仅内部代理（脚本注释亦自承是 judge 的 proxy），**非 LLM-judge** → **报告以 F1 为准**（30B-A3B F1=5.0 / 32B F1=4.1 / 14B F1=2.2）。
> - **⏸ 暂不评（需 API / GPT-4o judge，暂缓）**：**LongMemEval**（官方 = GPT-4o auto-judge）、**∞Bench / HELMET** 的 judge 型子任务。现成 harness = 仓库里 `longmemeval/` 包（Track B RAG baseline，含 loader `load_longmemeval` + 官方提交 JSONL 格式 + `MoMSlotReranker` 接入 stub），**待有 API judge 时接，不用从头建**；若之后要 QCMem-native 口径，只复用它的 loader+scoring。
> - **速度已回填全 scale（2026-07-16，`bench_qcmem_vs_dense_result.txt`（原 8B 于 resume_j=12 单测，其余 scale 2026-07-16 sweep；速度对 j 不敏感，QCMem read pack 恒 6657 tok））**：prefill 加速随 ctx 长度增长（8k ~1–1.6× → 32k ~7–18× → 128k **50–103×**，Dense 能装下的档）；decode 8k 略慢(0.6–0.7×)、32k/128k 快(**1.6–2.6×**)；**QCMem 峰值显存跨 8k/32k/128k 恒定**（读 pack 固定 6657 tok：0.6B 2.1G / 1.7B 4.5G / 4B 9.5G / 14B 31.4G / 30B-A3B 63G / 32B 68G），Dense 显存随长度线性增 → **32B/30B-A3B @128k Dense OOM，QCMem 恒定跑通**（headline 超 scale 卖点）。

**★ agent 侧(8B→0.6B) 全 scale benchmark 基本完成**（详数见 RUN_REGISTRY「★ QCMem 全 scale benchmark」+ bench_qcmem_vs_dense_result.txt）。核心结论：①128k Dense=0 全 scale 崩、QCMem 存活（普适卖点）②adapter 是硬任务/长档杠杆（8B multikey 42→91, vt 46→97）③**zero-shot readout 崩点随模型变小而变浅**（0.6B ~0.09L → 32B >0.42L），gap=adapter 缺口随 scale 缩小到 32B~0 ④multikey topk 甜点=4。**14B/32B(collaborator, zero-shot j3) 全 benchmark 聚合完成**（2026-07-16）：RULER single/multikey 满分级、vs-Dense 128k Dense崩(11/OOM)→QCMem 存活(98-100)、LongEval~99、LongBench qa_f1 32B=12.4/14B=9.6、LoCoMo 32B acc6.6/14B acc1.4。**32B VT 异常弱(iter_bm25 j3峰值~24，深 j 不升→selector/模型瓶颈)，14B VT 却强(89-96)。30B-A3B（j12）BABILong=32.3 / LoCoMo acc7.4·F5.0 / vs-Dense 128k single Dense OOM→QCMem 100 已聚合，multikey 补跑中**；LongMemEval/∞Bench/HELMET 待接 harness。

### 1c. vs-Dense 速度对比全表（2026-07-16，`bench_qcmem_vs_dense_result.txt`（原 8B 于 resume_j=12 单测，其余 scale 2026-07-16 sweep；速度对 j 不敏感，QCMem read pack 恒 6657 tok））

> zero-shot（无 adapter），`bench_qcmem_vs_dense.py --mode speed`，topk12 chunk512 bf16，1×H20/卡，per-model readout-safe j，QCMem 读 pack 固定 6657 tok。prefill×=Dense/QCMem prefill 时间比；decode×=QCMem tok/s ÷ Dense tok/s。列格式 8k/32k/128k。

| model | j | Dense prefill | QCMem prefill | **prefill×** | Dense mem | QCMem mem | decode× |
|---|---|---|---|---|---|---|---|
| 0.6B | 2 | 0.12/1.23/15.41s | 0.12/0.07/0.15s | 1.0/17.6/**102.7×** | 2.4/5.9/19.8G | **2.1G(恒定)** | 6.6/0.7/2.6× |
| 1.7B | 3 | 0.25/1.74/17.29s | 0.16/0.22/0.25s | 1.6/7.9/**69.2×** | 4.9/9.0/25.6G | **4.5G(恒定)** | 0.6/2.6/2.4× |
| 4B | 9 | 0.65/4.34/43.83s | 0.45/0.52/0.55s | 1.4/8.3/**79.7×** | 10.0/15.6/37.9G | **9.5G(恒定)** | 0.7/2.2/2.2× |
| 8B | 9 | 2.37/13.04/110.36s | 1.82/1.90/1.92s | 1.3/6.9/**57.5×** | 18.5/24.8/49.8G | **17.9G(恒定)** | 0.9/32.4/68.7× |
| 14B | 13 | 1.88/9.98/76.26s | 1.44/1.50/1.53s | 1.3/6.7/**49.8×** | 32.1/39.7/70.2G | **31.4G(恒定)** | 0.7/2.0/1.6× |
| 30B-A3B | 16 | 0.75/5.19/**OOM** | 0.55/0.62/0.65s | 1.4/8.4/**Dense OOM** | 63.7/71.3/**OOM** | **63.0G(恒定)** | 0.7/1.6/— |
| 32B | 27 | 4.47/23.91/**OOM** | 3.41/3.46/3.49s | 1.3/6.9/**Dense OOM** | 69.3/80.5/**OOM** | **68.0G(恒定)** | 0.7/1.6/— |

**三点**：① prefill× 随 ctx 长度飙升（8k ~1–1.6× → 32k ~7–18× → **128k 50–103×**，QCMem prefill 恒定 <3.5s）；② decode 8k 略慢(短档开销 0.6–0.7×)、32k/128k **快 1.6–2.6×**；③ **QCMem 峰值显存跨 8k/32k/128k 完全恒定**（读 pack 固定 6657 tok），Dense 随长度线性增 → **32B/30B-A3B @128k Dense OOM，QCMem 恒定跑通**（headline 超 scale 卖点）。

## 1b. 蒸馏策略（2026-07-14 用户定：先只 8B）

> **★ 双 j 更新（2026-07-16，覆盖旧固定 0.33L/j3）**：见 `status/QCMEM_J_DETERMINATION.md`。zero-shot 报 **readout-safe j**（single≥90 最深，per-model：0.6B j2/1.7B j3/4B j9/8B j9/14B j13/32B j27）；adapter 目标 **content-j ~0.45L**（probe 语义峰，近 scale-invariant）。adapter 的作用重述为「**把可读深度从 readout 崩点推向 content 语义峰（~0.45L）**」——gap 随 scale 缩小（0.6B ~0.39L 巨大 → 32B ~0），故**小模型 adapter 价值最大，32B 几乎不需**。表3 的 `resume_j=12（≈0.33L）`是旧默认；**content-j adapter（0.6B j13 / 1.7B j13 / 4B j16 / 8B j16 / 14B j18）正在另训**，出来后替换主表 +adapter 列（当前列填的是现有 ~0.33L adapter）。

### 表1 — 每模型是否蒸馏（每 cell 的 QCMem 报法）
| 模型 | zero-shot(免训练) | +self-distill adapter | adapter 路径 | 说明 |
|---|:---:|:---:|---|---|
| Qwen3-0.6B | ✅报 | ✗ | — | 只 zero-shot |
| Qwen3-1.7B | ✅报 | ⬜可选(P2) | 待蒸 | scale 一致性可补 |
| Qwen3-4B | ✅报 | ✗ | — | 只 zero-shot |
| **Qwen3-8B(主)** | ✅报 | **✅报** | `outputs/qcmem_distill_qwen_j12_r32_4k` | ★唯一必蒸 |
| Qwen3-14B | ✅报 | ⬜可选(P2) | 待蒸 | scale 一致性可补 |
| Qwen3-32B | ✅报 | ✗ | — | 只 zero-shot |
| Qwen3-30B-A3B | ✅报 | ✗ | — | 只 zero-shot |
- **zero-shot** = 免训练、stock backbone 直接跑（核心卖点，所有模型都报）。
- **+adapter** 作用：支撑 abstract「in-window matches dense」+ 把可用 j 推深。仅 8B 必做，1.7B/14B 可选（P2 不阻塞）。

### 表2 — 蒸馏方法（self-distillation，无外部 teacher / 无标注）
| 角色 | 配置 | 梯度 | 作用 |
|---|---|---|---|
| **Teacher** | QCMem `resume_j=0`（adapter DISABLED, `no_grad`）= 精确全 forward | 冻结 | 取每 loss token top-k=64 logit 支撑（无损上界） |
| **Student** | QCMem `resume_j=j`（默认12）+ LoRA on `layers[j:]` | 仅 LoRA 可训, backbone 冻结 | 从第 j 层浅缓存重算上层, 学着还原 teacher |
| **同一份权重** | adapter on/off 切换充当 student/teacher | — | 不额外占显存、无需外部 teacher/标签 |

**Loss** = teacher top-k 支撑上的**双向 top-k KL** `λ·KL(p‖q)+(1-λ)·KL(q‖p)`（λ=0.6）+ 可选极小 CE-to-argmax（默认0）。纯蒸馏，让「深度 j=12 的读出」逼近「深度 j=0 全 forward 的读出」。

### 表3 — 蒸馏超参（默认，per-backbone 各一个 LoRA）
| 项 | 值 | 项 | 值 |
|---|---|---|---|
| resume_j | 12（=split 在第12层, ≈0.33L）| lora_rank | 32 |
| chunk_size | 512 | n_ctx | 7 →(7+1)×512=**4096-tok 窗口** |
| teacher_topk | 64 | distill_lambda | 0.6 |
| ce_weight | 0（默认关）| total_steps | 1000 |
| lr | 1e-4 | warmup | 50 |
| grad_ckpt | on | 成本 | ~1-2 GPU 时（很便宜）|

- **数据**：PG19 文本，on-the-fly chunk。**效果**：in-window 追平 dense + 可用 j 从 ~9 推到 ≥12（脚本注释例：BABILong qa5 .14→.67）。
- 脚本：Mixture-of-Memory `scripts/train_qcmem_distill.py` / COMem `train/distill.py`（`python -m train.distill --model <hf> --j auto`）。

## 2. 方法对照（每个 cell 内全部跑 = 全 baseline，每 scale 都测）
| 方法 | 说明 |
|---|---|
| **QCMem** bm25 / iter_bm25 / +adapter | 我们（主） |
| Dense / full-ctx | 窗口内上界，超窗口崩塌对照 |
| KV-Direct | 全深度重算无检索(resume_j=0) |
| HCache | 中层无检索 |
| StreamingLLM | recency 固定预算 |
| MemoryLLM | 外部固定记忆 |

## 3. Benchmark 就位度（harness 工程量）
| benchmark | 脚本 | 状态 |
|---|---|---|
| RULER | `eval_ruler_qcmem.py` | ✅就位(上限32k, 超窗口走 bench 脚本) |
| BABILong | `eval_qcmem_babilong.py` | ✅ |
| LongBench | `eval_qcmem_longbench.py` | ✅ |
| LongEval | `eval_qcmem_longeval.py` | ✅ |
| LoCoMo | `eval_qcmem_locomo.py` | ✅ |
| vs-Dense | `bench_qcmem_vs_dense.py` | ✅(含128k) |
| **LongMemEval** | — | ⏳需接(长期记忆chat, ~500q, 5能力: 抽取/多会话/时序/知识更新/弃权; LongMemEval_s~115k) |
| **∞Bench** | — | ⏳需接(avg 100k+, 12任务) |
| **HELMET** | — | ⏳需接(7类别, 可控到128k, 含RAG/re-rank/QA/summ) |

## 4. 分工建议
- **一人认领「一列（模型）」**最干净：跑该模型全 benchmark × 全 baseline，output `ruler_results/qcmem_<model>_<bench>/`。
- diskB 三节点共享 FS → task-pool 动态调度多卡排空。
- 大模型(32B/30B-A3B)显存：H20 97GB 单卡可放，QCMem read pack 恒定省显存；Dense baseline 长档可能吃紧，必要时分片。
- 跑完更新本表 + `status/RUN_REGISTRY.md`。

## 5. 待办（工程 prep，非 eval）
1. **下载** Qwen3-4B / 32B / 30B-A3B(instruct) → 本地+diskB。
2. **接 harness**：LongMemEval / ∞Bench / HELMET 的 QCMem 适配（loader + QCMem generate 接口 + 官方判分）。
3. 各新模型跑 QCMem **self-test**(j=0 与全 forward logit diff<1e-4) 确认代码适配。
4. 各新模型自蒸馏 **adapter**（in-window 追平 dense 需要）。

---
> ⚠️ 全量方案工作量巨大（7 模型 × 10+ benchmark × 6 方法）。建议执行顺序：①下载+接harness ②self-test各模型 ③8B 收尾(基准) ④按模型列铺开 scale。

---

## 6. 启动命令（copy-paste，权威）

> **COMem 仓库**（collaborator 用，`git@github.com:liuhanzuo/COMem.git`）：`--n` 默认已=500（RULER/BABILong），真实集全量。
> 环境：`pip install -r requirements.txt`；模型传 HF 路径或本地路径；`--j auto` 自动查 model_registry。

### 6.1 COMem 一行跑单 cell（collaborator）
```bash
cd COMem
# RULER (n=500默认): 单模型全长度
./run_cell.sh ruler <model_path> --lengths 8k,16k,32k,64k,128k --selector bm25 --j auto
# vt 用 iter_bm25
./run_cell.sh ruler <model_path> --tasks vt --selector iter_bm25 --lengths 8k,16k,32k --j auto
# BABILong (n=500默认)
./run_cell.sh babilong <model_path> --tasks qa1,qa2,qa5 --lengths 0k,1k,2k,4k,8k,16k,32k --j auto
# LongBench / LoCoMo (全测试集, 不传--n)
./run_cell.sh longbench <model_path> --j auto
./run_cell.sh locomo   <model_path> --locomo_data <path> --j auto
# 带自蒸馏 adapter (仅8B): 加 --adapter
./run_cell.sh ruler <model_path> --adapter <adapter_dir>/final --j auto
# baseline (每 scale 都测): --baseline dense|kvdirect|hcache|streamingllm
./run_cell.sh ruler <model_path> --baseline dense --j auto
# 或 dispatcher 形式
python -m eval.run --benchmark ruler --model <model_path> --j auto --n 500 ...
```
每模型 j 由 `--j auto` 自动取（0.6/1.7B→9, 4/8B→12, 14B→13, 32B→21, 30B-A3B→16）。

### 6.2 自蒸馏 adapter（仅 8B 必做，1.7B/14B 可选）
```bash
cd COMem
# 单卡
python -m train.distill --model <Qwen3-8B_path> --j auto --data <pg19.jsonl> --out outputs/comem_distill_8b_j12
# 多卡
torchrun --nproc_per_node 8 -m train.distill --model <path> --j auto --data <pg19.jsonl> --out <dir>
# 产出 <dir>/final 喂给 eval 的 --adapter
```

### 6.3 内部（Mixture-of-Memory，diskB 三节点 task-pool 批量铺 n=500）
```bash
# 8B RULER n=500 重铺(旧n100作废): 队列worker多卡排空,每卡抢cell
export MODEL_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
export LORA_CK=outputs/qcmem_distill_qwen_j12_r32_4k/final   # 带adapter版; zero-shot则 LORA_CK=none
export OUT_DIR=ruler_results/qcmem_n500_8b SELECTOR=bm25 NAME_PREFIX=qcmem_n500
export PYTHON_BIN=/opt/conda/envs/torch-base/bin/python
# ⚠️ drain脚本内 --limit 需从100改500 (scripts/_qcmem_n100_drain.sh 复制为 _qcmem_n500_drain.sh 改 --limit 500)
for dev in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$dev setsid nohup bash scripts/_qcmem_n500_drain.sh q 1 $dev >logs/qcw_n500_dev$dev.out 2>&1 &
done
# eval入口(单cell): scripts/eval_ruler_qcmem.py --model_path $M --lora_adapter $CK --resume_j 12 \
#   --selector bm25 --topk 12 --ruler_tasks niah_single --lengths 16k --limit 500 --chunk_size 512
# BABILong n=500: scripts/eval_qcmem_babilong.py ... --limit 500
# 超窗口崩塌(vs-Dense, 128k): scripts/bench_qcmem_vs_dense.py --mode accuracy --n_acc 100 (崩塌用n100够)
```

### 6.4 节点/环境速查
- diskB 三节点(28.85.35.73/28.82.250.82/28.83.24.104, 36000端口): `PYTHON_BIN=/opt/conda/envs/torch-base/bin/python`, 共享FS免同步。
- 本机 wzc1 L20A: `.venv/bin/python`(torch2.13)。
- 判分: RULER `string_match` / BABILong `TASK_LABELS`+`compare_answers`(禁 re.search)。
- ⚠️ RULER eval_ruler 路径长度上限 32k; 64k/128k 超窗口档走 `bench_qcmem_vs_dense.py`。
