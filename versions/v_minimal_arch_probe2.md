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

## ★ 16000 步延伸终值（2026-07-11，方向4 硬结果）
| arm | 16000步 ppl | vs base 25.34 |
|--|--|--|
| **armB**（前12继承+2fresh 全训） | **19.54** | **-23%** |
| scratch14（14层随机初始化） | **21.25** | -16% |
| **armA**（冻结前12，只训2新层） | 23.98 | -5.4% |
| base（全16层充分训练） | 25.34 | — |

### ★★ 因果分解（scratch14 收敛后，诚实结论）
- **深度效应（大头）= scratch14 vs base = 25.34→21.25 = −4.1**：14 层比 16 层在同 16000 步下训得更好（浅模型同预算收敛更充分）。这是 armB<base 的**主要**来源。
- **继承效应（小头但明确）= armB vs scratch14 = 21.25→19.54 = −1.7**：在剔除深度效应后，继承前12层仍带来额外 −1.7 增益（前12表征直接可复用，省了这些层的重学）。
- **armA（冻结前12）= 23.98 反而劣于 scratch14 21.25**：只训 2 新层、冻结前12太死，容量不足 → 继承前12的正收益只在"允许微调（armB）"时兑现，纯冻结不够。
- **诚实结论**：前12+2fresh 的 14 层继续训（armB）确实反超全16层，但**主因是"14 层浅架构同预算收敛更好"（深度效应 −4.1），继承前12是次要但真实的加成（−1.7）**。不能宣称"顶层纯冗余/继承是全部功劳"——deep effect 是大头。这个 nuance 对论文诚实性关键。
- **对 QCMem 的意义**：印证 depth-partition（前 j 层语义可复用）——前12层表征迁移到更浅架构仍有效（−1.7），支持"前 j 层承载可迁移语义"的核心命题；但"精简架构比全模型更优"的强 claim 需谨慎（深度 confound）。
