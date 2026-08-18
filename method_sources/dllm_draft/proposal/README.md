# dLLM Proposal Repository

本目录只保存 `dllm_draft` 当前可执行、可证伪的研究提案。

状态规则：

- `active/`：已有仓库资产和明确 gate，值得优先实施；
- `backlog/`：问题可能成立，但必须先完成前置验证；
- `archive/`：旧主张已被更正、撤回或替代，仅保留 provenance。

每个 proposal 至少包含：

```text
PROPOSAL.md   主张、方法、实验、kill gate、Related Work
SOURCES.md    内部证据与外部一手文献
STATUS.json   机器可读状态
```

## 强制 Related Work 规则

任何 proposal 在启动 GPU 前，必须：

1. 在 `PROPOSAL.md` 中包含独立的 **Related Work 与新颖性边界**章节；
2. 列出最接近的工作、重叠点、剩余空缺以及明确不能主张的内容；
3. 在 `SOURCES.md` 中保存可核查的一手来源，不能只写模糊类别；
4. 新文献若直接覆盖核心主张，先收窄或归档 proposal，再启动实验；
5. benchmark、baseline 和方法组件本身也属于 Related Work，不只引用模型论文。

## 当前提案

### Active

1. `active/A01-execution-grounded-typed-repair/`
   - 强 dLLM checkpoint + execution-grounded typed subtree repair；
   - 第一阶段只测 oracle-localized repair operator；
   - oracle gate 不通过，不训练 localizer。

## 历史入口

- `../ELASTIC_SCAFFOLD_PROPOSAL.md` 已标为 superseded；
- `../DLLM_SALVAGE_ROADMAP_20260808.md` 是仓库级方向审计；
- `../DLLM_RESULTS_20260807.md` 是结果与撤回时间线。

