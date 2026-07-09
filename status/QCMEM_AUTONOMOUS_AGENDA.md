# QCMem 自主议程（用户 2026-07-10 指令，heartbeat 必读）

> **本文件是用户交给 main/heartbeat 的自主研究议程。** 当前 benchmark column 全部跑完后，按此推进（无需再问用户，属已授权自主范围）。
> heartbeat 每轮巡检时：若「当前 column 任务（见 §0 门槛）已全清 + GPU 有空」→ 从 §1 议程按优先级挑下一个启动。
> 维护：每推进一项，更新对应状态；出结论沉淀到 QCMEM_PAPER_DRAFT.md + RUN_REGISTRY.md。

---

## §0 前置门槛（这些 column 跑完才进 §1 自主议程）

对应 TaskList #1-#4，全绿才算 benchmark 矩阵完整：
- [ ] #1 LoCoMo baseline 三方（HCache/KV-Direct）→ 补 draft §2.9
- [ ] #2 MemoryLLM × RULER（同类方法对照）→ 补 draft
- [ ] #3 MemoryLLM × LongEval/LongBench/LoCoMo
- [ ] #4 babilong 三方补全档（0k-4k baseline）

门槛达成 = 5 benchmark × {QCMem, KV-Direct, HCache, MemoryLLM} 矩阵基本齐 + 都用 per-task 最优 topk。

---

## §1 自主议程（用户 2026-07-10 指定的 4 个方向，按建议优先级）

### 方向 1：查漏补缺（benchmark / baseline / 实验）—— 低成本先做
- 检查 5 benchmark 矩阵还有哪些空 cell（尤其 MemoryLLM 全 benchmark、babilong 低档、LoCoMo 三方）。
- 审视是否缺关键 baseline（StreamingLLM 已做；是否要 InfLLM/Activation-Beacon 等 novelty 清单里的 Tier-1，见 QCMEM_RELATED_WORK.md）。
- 是否缺 ablation（selector bm25 vs 其它、sink 数、resume_j sweep 复核）。
- 产出：补齐的 cell 进 draft；缺口列进 TaskList。

### 方向 2：pretrain scale（7B + 1B/3B 其它方向）
- **7B semantic-bottleneck**：现有 1B/3B from-scratch + 真实 Qwen3-8B funnel continued（gap ~15%）。下一步可选：
  - (a) 真实 Qwen3-8B funnel continued **更长步数**（现 2000 步，看 gap 能否收窄）
  - (b) **funnel-Qwen + 自蒸馏 LoRA 叠加**（三列对照证明单 funnel readout 弱、需蒸馏；叠加是正确组合，验证端到端是否超 vanilla+distill）
  - (c) 7B from-scratch bottleneck（若要纯 scale 曲线，但收敛慢）
- **1B/3B 其它方向**：bottleneck_dim sweep（现固定 512，扫 256/1024 找甜点）、bottleneck_layer sweep、更收敛步数验证趋势。
- 依据：QCMEM_PAPER_DRAFT §3.3/§3.4（端到端 ΔNLL + (j,dim) sweep），教训=funnel 供可压结构、readout 需蒸馏。

### 方向 3：QCMem infra / kernel 优化（自主）
- 当前瓶颈：decode ~2.4s（每步重算 layers[j:]，attend 整个 read pack）> full 0.3-0.5s。
- 可做：resumed-band KV cache（READ 阶段 layers[j:] 的 KV 缓存复用，避免每 decode step 重算）；write 阶段 batch chunk 前向；MemoryLLM inject 的串行瓶颈（若也要优化）。
- 目标：把 QCMem 的 decode 从"慢于 full"优化到可接受，补强效率章节（§2.3 现在诚实标了 decode 慢是软肋）。
- 派 coder 做，先 profile 定位再改 kernel。

### 方向 4：★ 极简架构假设验证（最有想象力，可能最重要）
- **假设**：既然前 ~j 层（j=6 或 12，具体见 §3.1 probing 的语义饱和点）就承载了语义，顶部多数层是冗余的——**能否直接用「前 j 层 + 1 层专做 NTP」构成一个更小的 transformer，去掉中间冗余层？**
- 这是把"分工命题"从"解释 QCMem"推进到"直接指导架构精简"——若成立，是独立的强 claim（比 QCMem 本身更大）。
- 实验设计（待细化）：
  - from-scratch 训一个「j 层理解 + k 层（k 小，如 1-2）NTP head」的模型 vs 同参数量的标准 j+k 层 vs 标准全层，比 LM ppl + 下游。
  - 关键问题：省掉的中间层，是"冗余"还是"渐进精炼"？probing 说语义在中层饱和，但生成质量可能仍需深度。要诚实测——可能证否（中层不冗余只是功能不同）。
  - 先小规模（1B 级）快速证伪/证实，再决定投入。
- ⚠️ 这个方向风险高、可能推翻（"语义饱和 ≠ 可删层"，§3.1 已知中层删除鲁棒但早/末层删除最伤——见 novelty note 的 Stages-of-Inference 2406.19384）。先做便宜的探针实验判可行性，别直接烧大模型。

---

## §2 调度原则（heartbeat 遵守）
- 节点：本机 wzc1 B200/L20A(8) + H20 28.83.53.31 diskB(8) = 16 卡。**29.162.226.120 是 dllm 专用，绝不碰**（见 memory dllm-h20-node）。
- 优先级：先清 §0 门槛（benchmark 矩阵）→ 方向1（查漏，低成本）→ 方向4（极简架构，先便宜探针）/ 方向2（pretrain scale）并行 → 方向3（infra，派 coder 不占训练卡）。
- 大投入（7B from-scratch / 烧多节点）前，若不确定，在 heartbeat 报告里标注并可继续（用户已授权 ablation/scale 延伸自主执行）；只有「全新方向的重构」才需用户拍板。
- 每个新实验落账 gpu_runs.jsonl + RUN_REGISTRY.md；结论进 draft。
