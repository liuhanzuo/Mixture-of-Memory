# v-minimal-arch: 极简架构假设验证（方向4，probe#2）

> 2026-07-10。用户 idea：既然分工命题（§3.1）说语义在前 ~半深度饱和，顶部多数层是否冗余——能否用「前 j 层 + 少数新层做 NTP」构成更小的 transformer？

## 假设
前 j 层承载语义理解（probing：~半深度饱和）。顶部多数层是"渐进精炼"，可用少数几层新层 + continue-train 替代，去掉冗余中/上层。

## 实验设计（1B from-scratch 基座）
- 基座 = 已训好的 1B baseline（16 层，`outputs/sembott_1b_base_16k/final.pt`）。
- 构造：加载前 12 层 + embed/norm/lm_head，丢顶部 4 层，接 2 个 fresh 层 → 14 层 1.11B。
- 四臂（同 slimpajama seq2048 eff_bs48 lr3e-4，2000 步同口径）：
  - **armA**：冻结前12，只训 2 fresh + head（trainable 0.384B）
  - **armB**：全训 14 层（trainable 1.114B，heal 模式，sanity max|diff|=0 前12精确继承）
  - **scratch14**：14 层随机初始化全训（隔离"是不是只是层数少了"）
  - **base_2k**：全 16 层随机初始化（同 2000 步上界对照）
- 脚本 `scripts/train_minimal_arch_probe2.py` + `launch_minimal_arch_probe2.sh`（commit 252a007）。

## 结果（2000 步同口径，末10均值 ppl）
| arm | 结构 | ppl |
|--|--|--|
| **armB** | 前12继承+2fresh 全训 | **25.77** |
| **armA** | 前12继承+2fresh 冻结前12 | 26.24 |
| scratch14 | 14层随机 | 48.60 |
| base_2k | 全16层随机 | 48.14 |
| （参考）base 16000步 | 全16层充分训练 | 25.34 |

## 初步结论（2000 步，正面信号，但需 16000 步收敛确认）
1. **继承前12层 = 巨大加速**：同 2000 步预算，armA/armB(26) 碾压从头训的 scratch14/base_2k(48)。
2. **armA≈armB**：冻结前12只训 2 新层（armA 26.24）几乎追平全训（armB 25.77）→ **前12表征直接可复用，顶层只需少量新层重学 NTP**。
3. **★逼近充分训练的全模型**：armB 25.77 距 base 充分训练 16000 步的 25.34 仅差 1.7%——**前12+2层只训 2000 步，逼近全16层训 16000 步**。
4. **诚实 caveat**：2000 步 base 未收敛（48 vs 充分训 25.34），所以本对比主要证"继承加速"。"顶层永久冗余"需 armA/armB 训到 16000 步收敛后与 base 25.34 严格比——**已启动延伸（armA/armB @16000 步，outputs/minarch_1b_arm{A,B}_f12k2_16k）**。

## Relationship to prior work
- 接近"深层剪枝 + 修复"（*Unreasonable Ineffectiveness of Deeper Layers* 2403.17887 剪深层+LoRA heal；ShortGPT/LaCo）。
- **我们的 delta**：由分工探测（§3.1）驱动 + **加 fresh 层重学 NTP**（非纯剪枝修复），且和 QCMem 的 depth-partition 同源（前 j 层=语义/可缓存，顶层=可替换生成头）。

## Known issues
- 2000 步对照未收敛，终局待 16000 步延伸。
- 只 1B 规模；若结论成立需 scale 验证。
- probe#1（现成模型硬跳层，无训练）ppl 爆炸 6-21× 是 off-manifold artifact，非本假设的证否（见 logs/probe_minimal_arch.log）。
