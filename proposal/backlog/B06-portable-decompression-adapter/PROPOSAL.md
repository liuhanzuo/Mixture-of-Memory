# B06 — Portable Mid-Recompute Readout Adapter

## 状态

**BACKLOG / CONFIRMED SINGLE-VARIABLE RESULT。适合作为 Paper A extension 或短文。**

## 核心结果

相同 HCache path、相同 Qwen3-8B、相同 `j=12`、相同 node/commit：

- HCache no LoRA：LoCoMo judge `13.29`
- HCache + CoMem Read LoRA：`31.17`
- 增益：`+17.88`

该路径无 retrieval，因此提升不能来自 selector。它支持：

> 自蒸馏 LoRA 学到的是 shared mid-recompute readout/decompression skill，
> 而不只是 CoMem retrieval-pack 特化。

## 下一步

1. 在 canonical diskB HCache predictions 上同 harness rejudge，消除 8.11 vs 13.29 drift；
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

