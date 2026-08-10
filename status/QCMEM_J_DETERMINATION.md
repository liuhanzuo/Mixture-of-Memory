# QCMem split-depth j — determination via truncation-downstream probe

> 2026-07-16. 用户指令:QCMem 的 split-depth j 应取"前 j 层语义信息尽可能多"的 balance 点,用**切分 + 小 head + 下游任务**实测确定,而非拍 0.33L。
> 脚本 `scripts/probe_truncated_downstream.py`(truncate-and-head == 第 j 层线性 probing)。任务 SST2/WiC/RTE。结果 `probe_results/trunc_{model}.json`(diskB .82.250)。

## 实测语义饱和 j(knee98 = 达峰值 98% 的最浅深度)

| model | L | semantic j (knee98) | /L | hard-task(RTE/WiC) peak |
|---|---|---|---|---|
| 0.6B | 28 | **L13** | 0.48L | 0.64L |
| 1.7B | 28 | **L13** | 0.48L | 0.64L |
| 4B | 36 | **L16** | 0.44L | 0.72L |
| 8B | 36 | **L16** | 0.44L | 0.44L |
| 14B | 40 | **L18** | 0.46L | 0.49L |
| 32B | 64 | **L27** | 0.42L | 0.44L |
| 30B-A3B | 48 | ~**L22**(0.45L 比例外推,未 probe) | 0.45L | — |

## 结论
1. **语义 j ≈ 0.42–0.48L(均值 ~0.45L),近 scale-invariant** —— 比旧 0.33L 假设更深。0.33L 只买到 ~95% 峰值语义(knee95);~0.45L 才 near-max(98%)。
2. **所有 model×task 都 mid > top**(语义中层最富,顶层用 richness 换 next-token 特化)→ 印证"理解在前中层,生成在顶层"分工。
3. **"至少十几层"证实**:balance j 均 ≥13 层,随规模增长(0.6B L13 → 32B L27)。
4. 对照:8B RTE 峰 L16 ≈ 已知 L17,pipeline 可信。
5. **32B 该用 ~j27–28(不是旧 j21/0.33L)。**

## ⚠️ 关键 nuance(probe j vs QCMem readout j)
- probe 测的是**语义 CONTENT 深度**(线性可解码)。QCMem 真实 readout 是 truncate-and-**recompute**,对 faithful recompute **j-不变**(=完整模型);QCMem ≠ 完整只因**压缩 hidden[j]**(分块/选择丢上下文)。
- 所以两个 j 可能不同:(a) 语义 content j ≈0.45L(本文);(b) QCMem readout 压缩容忍 j(single recall 在哪档掉,由 32B cliff-bracket j12/16/21/24 实测)。
- **报告 j 的原则**:取"语义足够富 + readout 不塌"的深度。若语义 j(0.45L)深于 readout cliff → 说明需 adapter 才能缓存那么深(adapter 的价值);若 readout 能撑到 0.45L → 直接用语义 j。
- **待反核**:32B cliff-bracket(在跑)+ deep-j RULER sweep 出来后,把 recall-vs-j 与本 probe 曲线并置,定最终 reporting j + 重建主表。

## ★ 反核完成(2026-07-16):probe 与 QCMem readout 一致 → 缓存更深

**32B cliff-bracket RULER recall(n=100, bm25, 8k/16k/32k):**
- **niah_single = 100 在 j3/j12/j16/j21/j24 每一档** → 到 **j24(0.375L)无 cliff**,readout 完美。
- **niah_multikey 缓慢侵蚀**:j12=98.3 → j16=96.3 → j21=95.3~97.3 → j24=94.3(12 层掉 ~4pt,非 cliff)。
- vt 弱(≤23,深 j 不救)= selector/多跳瓶颈,与 j 无关。

**结论**:probe 说 32B 语义峰 ~j27(0.42L);readout 说到 j24 都不塌 → **两线一致:缓存该比 j21 更深**。QCMem 忠实重算 + 检索让 single 在深 j 仍完美,所以能安全缓存到语义峰附近。

### ★ 定案报告 j(语义 ~0.45L,probe 实测 + readout 反核)
| model | L | 报告 j | /L |
|---|---|---|---|
| 0.6B | 28 | 13 | 0.48L |
| 1.7B | 28 | 13 | 0.48L |
| 4B | 36 | 16 | 0.44L |
| 8B | 36 | 16 | 0.44L |
| 14B | 40 | 18 | 0.46L |
| 30B-A3B | 48 | ~22 | 0.45L |
| **32B** | 64 | **27**(实测安全到 j24) | 0.42L |

**旧 j3(balance-recall)= recall 上界参考;旧 0.33L(8B j12/32B j21)= 保守下界(~95% 语义)。主表报 ~0.45L。**
**待办**:按上表 ~0.45L j 重跑 QCMem RULER(注:single 对 j 不敏感恒 100,multikey/vt 略降)→ 重建主表(task#2)。

## ★★ 修正(2026-07-16 晚):0.45L zero-shot readout 崩 → content-j ≠ readout-j

**上面"两线一致,缓存到 0.45L"是错的**——由 32B cliff-bracket 只测到 j24(0.375L)恒 100 过度外推。全量 0.45L RULER sweep 推翻:

| model | j(/L) | zero-shot single(8k/16k) |
|---|---|---|
| 0.6B | 13(0.48L) | 6/3 |
| 1.7B | 13(0.48L) | 8/0 |
| 4B | 16(0.44L) | 25/3 |
| 8B | 16(0.44L) | **3/0** |
| 14B | 18(0.46L) | 18/36 |
| **8B j9(0.25L) control** | | **100/99** ← 对照:浅 j 完好 |

- **真·深度效应(非 bug)**:8B j9=100 vs j16=崩。zero-shot QCMem readout 在 ~0.45L 全 scale 崩塌。
- **三个深度分离**:recall 最优 ~j3 < **zero-shot readout 容忍上限 ~0.25L** < 语义 content 峰 ~0.45L(probe)。
- **content 住在 0.45L,zero-shot 只够到 ~0.25L → gap = adapter 的意义**(adapter 把可读深度推向语义深度)。
- **∴ zero-shot 主表用 readout-safe j(~0.25L);0.45L 语义深度 = probe 证明 + adapter 读出**,不能 zero-shot 报在 0.45L。
- 32B j27 真跑在进行(定它是否也崩;j24=0.375L 撑住可能只因未到崩点)。
- **下一步**:bracket 各模型 zero-shot single 崩点(readout 容忍上限,~?L)+ 等 32B j27 → 定 zero-shot 报告 j;主表 zero-shot 用 readout-safe j,adapter 行报 content-j。

## ★★★ Bracket 完成(2026-07-16):readout 崩点单调随 scale 变深,非 0.25L 恒定

**方法**:zero-shot QCMem(无 adapter),niah_single **只 16k 一档**,n=100,selector bm25,topk12,chunk512。逐 j 找 single recall 从 ~100 掉到 ~0 的崩点。结果 `ruler_results/qcmem_<model>_j<j>_readoutbracket_n100/`(diskB;32B 用已跑的 `qcmem_32b_j27_semantic_n100`)。

**single recall(16k, n=100) vs j:**
| model | L | recall vs j(j:recall) | 50%崩点 j(/L) | 近满(≥90)最深 j(/L) |
|---|---|---|---|---|
| 0.6B | 28 | j2:**100** · j3:12 · j5:15 · j7:36 · j9:33 · j11:17 · (j13:6) | **~2.6 (0.09L)** | j2 (0.07L) |
| 1.7B | 28 | j2:**100** · j3:**100** · j5:74 · j7:33 · j9:15 · j11:6 · (j13:8) | **~6 (0.22L)** | j3 (0.11L) |
| 4B | 36 | j9:**99** · j11:52 · j13:9 · (j16:25) | **~11 (0.31L)** | j9 (0.25L) |
| 8B | 36 | j9:**100** · j10:81 · j12:9 · j14:0 · (j16:3) | **~11 (0.30L)** | j9 (0.25L) |
| 14B | 40 | j10:98 · j13:89 · j16:29 · (j18:18) | **~15 (0.375L)** | j13 (0.325L) |
| 32B | 64 | j24:**100** · j27:**100** (8k/16k/32k 全 100) | **>27 (>0.42L,无崩)** | ≥j27 (0.42L) |

**核心结论:**
1. **zero-shot readout 崩点单调随 scale 变深,NOT scale-invariant**:0.6B ~0.09L → 1.7B ~0.22L → 4B/8B ~0.30L → 14B ~0.375L → 32B >0.42L(无崩)。旧"~0.25L 恒定"只是 4B/8B 中段的粗值。
2. **content 深度 ~0.45L(probe,近 scale-invariant)vs readout 崩点 → gap = adapter 要补的量,随 scale 缩小**:
   | model | content/L | readout(50%)/L | gap/L(adapter 缺口) |
   |---|---|---|---|
   | 0.6B | 0.48 | 0.09 | **~0.39(巨大)** |
   | 1.7B | 0.48 | 0.22 | ~0.26 |
   | 4B | 0.44 | 0.31 | ~0.13 |
   | 8B | 0.44 | 0.30 | ~0.14 |
   | 14B | 0.46 | 0.375 | ~0.085 |
   | 32B | 0.42 | >0.42 | **~0(readout 已达 content)** |
3. **三深度分离坐实**:recall 最优 ~j3 < zero-shot readout 容忍上限(scale 依赖 0.09-0.42L) < 语义 content 峰 ~0.45L(probe)。**小模型 gap 巨大 → adapter 价值最大;32B readout 几乎已到 content 峰 → 几乎不需 adapter。**
4. **主表 reporting**:zero-shot 行用 per-model readout-safe j(近满 ≥90 那列:0.6B j2/1.7B j3/4B j9/8B j9/14B j13/32B j27);content-j(~0.45L)只在 probe + adapter 行报。

## ★★★ Adapter @ content-j 实测(2026-07-16):gap-vs-scale 论断验证 + 深度 trade-off

按上表 content-j(~0.45L)训 self-distill LoRA(PG19,teacher j0,student resume_j+LoRA layers[j:],KL λ0.6,r32,1000步)并 eval RULER n=100。j:0.6B/1.7B=13,4B/8B=16,14B=18。**32B skip(gap~0)**。

| model | content-j | +ad single 8k/16k/32k | +ad multikey | +ad vt | (对照)zs@content-j single崩 |
|---|---|---|---|---|---|
| 0.6B | j13(0.48L) | **95/98/99** | 24/24/22 | 0/4.8/0 | 6/3 |
| 1.7B | j13(0.48L) | **98/98/96** | 26/29/17 | 22.8/4.4/15 | 8/0 |
| 4B | j16(0.44L) | **100/100/100** | 46/33/37 | 72.8/65.4/79.8 | 25/3 |
| 8B | j16(0.44L) | **100/100/100** | 49/32/22 | 50.2/51.4/41.4 | 3/0 |
| 14B | j18(0.46L) | **100/100/99** | 83/77/71 | 80.8/89.4/64.8 | 18/36 |

1. **single 全 scale 通用恢复到 ~95-100(含 tiny)**：adapter 把可读深度推到语义 content 峰(~0.45L) → 直接证实"content 住 0.45L,zero-shot 只够 ~0.25L,gap=adapter 意义"（第2条 gap 表）**对 needle 检索成立**。
2. **hard 任务(mk/vt) scale-gated,严格随 gap 收窄而成功**：14B(gap0.085L)最强(mk71-83/vt65-89)≫其 zs@readout-safe → 4B/8B(gap0.13-0.14L)partial-ok → **1.7B/0.6B(gap0.26-0.39L)FAIL**(vt 0-23,mk 17-29,反低于其浅 j zs)。**坐实"tiny 模型 gap 巨大→深 j compositional readout 蒸不出来"**（对齐旧 0.6B adapter@j9<zs@j2）。
3. **⚠️ 深度 trade-off**：content-j(0.45L)adapter 的 hard 任务**弱于**旧 ~0.33L adapter(8B mk91→49/vt97→50；14B mk100→83)。→ **single 可读到语义峰,但 hard 任务读出随缓存深度单调变难,~0.33L 是 hard 任务甜点**。主表 adapter 行:追 hard 峰值用 ~0.33L;强调缓存到语义 content 深度用 0.45L(single 满、hard 衰减)。详见 `RUN_REGISTRY.md`「+adapter @ content-j」表。
