# Source Audit and Cleanup Decisions

整理日期：2026-08-08。

## `paperC_probe_results_zwfy6`

判决：**不服务任何当前 active proposal，但 7 个逐实验 CKA JSON 是历史 raw evidence。**

- 移入：
  `proposal/archive/paperC-v1-frozen-cap/evidence/adaptation_cka/`
- 删除：
  `all_curves.json`，因为它只是七个文件的聚合重复，并省略 `tag/base_meta`。
- 原目录在搬空后删除。

这些文件只能用于记录旧 P-C2 为什么被否证，不能再支持 universal
“adaptation onset depth”。

## Paper C v1

- 归档：
  - 完整 scoping/postmortem；
  - 三份正式 reviewer；
  - per-checkpoint adaptation CKA raw evidence。
- 删除：
  - 临时 brief；
  - 中断 salvage；
  - prompt+model-output rerun；
  - 含已知 citation 错误的中间产物。

最终判决：SQuAD capability proposal 和 forward-only depth predictor 已死亡。

## Paper C v2

- 保留为 backlog 的只有：
  - decoder-only LLM pretraining 中的 block-reset regime boundary；
  - single-pass vs repeated-data；
  - PPL vs parametric-knowledge 双轴代价。
- 删除方法包装：
  - “新 cyclic prune-regrow 方法”；
  - “depth cycling”；
  - “新 plasticity 机制”。
- 原始搜索 HTML/XML/PDF text cache 删除；保留 consolidated forward-citation audit。

## Paper D

- 跨家族 layer stitching 方法归档为 dead。
- 迁入 A01/shared：
  - 91-pair CKA matrices；
  - layer-order shuffle null；
  - R4 memo；
  - affine readout bandwidth pilots。
- 删除的主张：
  - depth mismatch 大于 family mismatch；
  - random-init floor 是 layer correspondence null；
  - affine pilot 证明所有 nonlinear bridge 都不可能。

## Paper E / F

- Paper E 并入 A01，作为 MMLU interface case 和 self-falsification。
- Paper F 留在 incubator；目前只证明 margin compression/near-tie 增多，
  尚未证明完整的 damage→flip fragility。
- 明确错误的 `PAPERF_ACCNORM_REDO.md` 归档为 superseded。

