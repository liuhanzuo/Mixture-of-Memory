---
name: bottleneck-layer-sweep-monotone
description: "方向2 layer sweep 结论——bottleneck 越深 LM 税越大（单调），QCMem 的 j=12 是\"可缓存语义上限 vs LM 税\"的折中"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0dac7a11-5048-4ecf-85c3-ff6b9fab88d3
---

方向2 semantic-bottleneck pretrain 的 bottleneck_layer sweep（1B from-scratch，dim512 固定，16000 步收敛，2026-07-10）结论：

**LM 税随 bottleneck 深度单调递增**（末20步均值 ppl）：baseline 25.34 / layer1 26.40(+4.2%) / layer3 26.82 / layer6 26.85 / layer9 27.74 / layer12 27.77(+9.6%)。**越往后放 funnel，信息密度损伤越大**——浅层是低阶/局部信息压缩损失小，越深越接近"生成前精炼表征"每维 load-bearing。

**关键 framing（用户 2026-07-10 点出）**：单看 LM 税缓存点应放浅层（L1 最省），但太浅则可缓存语义不足（§3.2 j-sweep：j≤9 检索饱和、j12 崖跌）。→ **QCMem 的 j=12 是"可缓存语义上限"与"LM 税"的折中，不是税最小点**。layer sweep 量化了折中的代价端，坐实 j=12 是权衡而非任意。

**Why**：这是对 QCMem 设计选择（为什么 split 在 j=12）的正面因果解释，也和分工命题 [[——见 draft §3.1]] 一致。

**How to apply**：写 QCMem 论文 §3.4 时，用这条把"j=12"从"拍脑袋选的"变成"两条曲线（可缓存语义 vs LM 税）交点的折中"。配套 dim sweep 结论：bottleneck 越窄 LM 税越高但越可压（trade-off）。完整数据见 status/RUN_REGISTRY.md 方向2 sweep 区 + draft §3.4（commit 0bf7182）。ckpt 在 outputs/sembott_1b_*_16k/final.pt。
