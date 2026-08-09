# B05 — Semantic Handoff Phase Diagram

## 状态

**BACKLOG MEASUREMENT。与已死亡的“forward probe 预测最佳 adaptation depth”不同。**

## 核心问题

在同一模型上测量：

- `j_content`：信息 probe-readable 的深度
- `j_native`：原生 suffix 可零训练读取的最深 split
- `j_adapt`：小 adapter 可恢复质量的最深 split

并将其展开为：

```text
split depth × readout capacity × task family
```

任务族：

- exact retrieval/copy
- multi-evidence composition
- semantic QA
- format/verbalization
- parametric knowledge

## 已有观察

- semantic feature depth 浅；
- knowledge/native readout 更深；
- exact-copy 任务可能偏好更浅 cache；
- upper cross-chunk attention 对组合任务 load-bearing；
- scale 增大时 readability gap 可能缩小。

## 必须避免

- universal semantic cut；
- 把 logit-lens onset 当 causal storage；
- 用一个 forward-only probe 预测最佳 graft depth。

## 下一步

Qwen3-8B 上做：

- `j={shallow,native,content,deep}`
- readout={native suffix, affine/tuned lens, LoRA, small decoder}
- task={copy,retrieval,composition,format}

若 phase diagram 不清晰，则并入 Paper A/B 机制小节，不独立成篇。

