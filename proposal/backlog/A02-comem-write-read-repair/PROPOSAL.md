# A02 — Context-Aware Write and Read Repair for Reusable Residual Memory

## 状态

**ACTIVE；近期最高 ROI 的模型/系统实验。**

## 问题

CoMem 在同 pack 下减少 Read 层数，但有可测质量税：

- `j=0 → j=12`：Read `931.9 → 664.4 ms`，`1.403×`
- RULER `99.19 → 96.07`，损失 `3.12pp`

continuous-prefix oracle 完全恢复，说明主要问题是 reusable interface，
尤其是独立 chunk Write 缺少 lower-layer document context。

## 已有证据

- chunk-local Write：`92.5`
- document-contextual oracle：`100.0`
- overlap `w=32`：`98.5`
- overlap `w=128`：`99.0`
- lower-layer Write LoRA：`98.5`

但这些只在 synthetic multikey 8k/16k 上成立。

## 第一阶段：零新增训练

组合已有 Read/Write adapter，固定相同 retrieval pack，比较：

1. `j=0` full-depth replay
2. `j=12` Read-LoRA only
3. `j=12 + overlap w32`
4. `j=12 + Write-LoRA`
5. `j=12 + Write-LoRA + Read-LoRA`

任务：

- LongEval 8k/16k/64k/128k
- BABILong qa1/qa2/qa5 @4k/16k/32k
- RULER multikey/VT
- LoCoMo paired subset，再补 full judge
- LongBench QA

## 第二阶段：联合训练

仅当第一阶段自然任务有正信号时：

- 1000–1500 steps
- 奇数 step 更新 Write LoRA，冻结 Read
- 偶数 step 更新 Read LoRA，冻结 Write
- teacher：document-contextual lower Write + `j=0` full replay

## 核心成功条件

- 至少三类任务关闭原 `j0-j12` gap 的一半；
- LongEval 或 multikey 相对 Read-only `≥2pp`；
- LoCoMo judge `≥1.5pp`；
- persistent bytes 和 per-query Read 不增加；
- 普通文本 PPL 恶化不超过 5%。

## 决定性系统 gate

用修复后的 Write 重做 equal-latency frontier：

- raw/dense replay latency-matched `k*`
- CoMem-w0
- CoMem-overlap-w32
- CoMem-Write-LoRA
- Joint

若 paired quality CI 仍显著低于 0，则停止“CoMem 优于 RAG”的叙事，
定位为高复用 workload 的 storage/read-compute 方案。

