# B06 — Portable Mid-Recompute Readout Adapter

## 状态

**BACKLOG / CONFIRMED SINGLE-VARIABLE RESULT。适合作为 Paper A extension 或短文。**

## 核心结果

> ⚠️ **2026-08-10 ERRATA — 下面三个数字已被撤回并替换。**
> `13.29 / 31.17 / +17.88` 取自 `scores.json` 的 `overall_judge`，而该字段是
> **两种仪器的加权混合**：cat1–4 的 1,540 条走 GPT-4o judge，cat5 的 446 条
> **从不进 judge**，由 refusal 正则本地打分（`scripts/eval_qcmem_locomo.py:687-690`），
> 占 22.5% 权重。
> **正确的单仪器数字（Judge$_{1:4}$，n=1540，两臂 id 完全配对）：
> noLoRA `16.69`、+LoRA `39.81`、增益 `+23.12`**
> （McNemar 414/58，exact two-sided p=2.6e-67；paired item bootstrap 95% CI [20.58, 25.58]）。
> 机制、复算与口径选择依据：`paperA/ERRATA_LOCOMO_MIXED_INSTRUMENT_20260810.md`。
> **对 B06 论点的影响：方向不变、且更强**（+17.88 → +23.12，2.4×），
> 因为混合口径把一个与 adapter 无关的常数摊进了分母。
> 下面原文保留作历史记录。

相同 HCache path、相同 Qwen3-8B、相同 `j=12`、相同 node/commit：

- HCache no LoRA：LoCoMo judge `13.29`
- HCache + CoMem Read LoRA：`31.17`
- 增益：`+17.88`

该路径无 retrieval，因此提升不能来自 selector。它支持：

> 自蒸馏 LoRA 学到的是 shared mid-recompute readout/decompression skill，
> 而不只是 CoMem retrieval-pack 特化。

## 下一步

1. 在 canonical diskB HCache predictions 上同 harness rejudge，消除 8.11 vs 13.29 drift；
   ⚠️ **该 drift 是拿旧混合口径的 13.29 量出来的**，rejudge 前先把 canonical 8.11 也
   换算到 Judge$_{1:4}$（n=1540）尺度，否则比的是两把尺子。
2. BABILong/RULER/LongEval 复现；
3. 第二个 residual/checkpoint compressor；
4. adapter size/layer-band 消融。

## 成功条件

- 多任务、多 compressor 保持显著 lift；
- layer/module ablation 定位共享 readout repair；
- 不依赖特定 retrieval pack。

## Kill 条件

- 只在 LoCoMo open-domain category 有益；
- 统一 harness 后增益消失；
- 换 compressor 完全不迁移。

