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

---

## ★★ Qwen3-8B scale 验证（2026-07-12，方向4 大模型硬结果）

> 用 **Qwen3-8B**（36 层，hidden 4096）剪层版复现 1B 结论。区别于上文 1B sembott 基座：此处基座是 Qwen3-8B 官方权重，剪层后各臂 **3.95B 参数**（14 层）。数据 `slimpajama_chunks_2048_qwen3.npy`，fp32 master weights，cosine schedule。

### 配置表 + 终值（末10 log 点均值 ppl，本次亲自复核）

| arm | 结构 | steps | eff_bs | token 预算 | 参数 | 终值 ppl（末10均值） | log / ckpt |
|--|--|--|--|--|--|--|--|
| **armB** | Qwen3-8B keep12+fresh2，全训 heal（fresh 1e-4 / inherited 2e-5，`transplant_max_abs_diff=0`） | **20000** | **128** | **5.24B tok** | 3.95B | **14.4950**（step20000=14.42） | `logs/qwen3_armB_f12k2_20k.log` / `outputs/qwen3_minarch_armB_f12k2_20k/{final,step20000}.pt` |
| **scratch14** | Qwen3-8B 14 层随机初始化全训 | **2000** ⚠️ | **24** ⚠️ | **98.3M tok** ⚠️ | 3.95B | **108.8790** | `logs/qwen3_scratch_f12k2.log` / `outputs/qwen3_minarch_scratch_f12k2/final.pt` |

- 两 run 均 `DONE` 落盘、**0 个 nan**（`grep -ic nan` = 0）。armB 到 step20000（lr 已降到 cosine 底 2e-6），末10 ppl 在 14.4–14.5 间小幅震荡、仍缓降 → 是"20000 步终值"而非彻底收敛（用户计划续到 200k）。
- scratch14 离线 val（`slimpajama_val`，200 chunk）复核 = PPL 109.99（`logs/qwen_ppl_scratch_f12k2.log`），与训练末10均值 108.88 一致，数字可信。

### ⚠️ 重大 caveat：8B 版 armB vs scratch14 预算严重不对等（非同口径）

**任务描述里写 scratch14"同 20000 步"是不准确的——实测 scratch14 只跑到 step 2000（`max_steps=2000`）、eff_bs=24。** 对比 armB 的 20000 步 / eff_bs=128：

- steps 差 **10×**，batch 差 **5.3×**，**token 预算差 53×**（scratch14 只吃了 98.3M tok，仅 armB 5.24B 的 **1.9%**）。
- 对照 1B 版：1B armB vs scratch14 是**严格同预算**（都 16000 步、eff_bs=96）→ 那里的 −1.7 继承效应干净可信。**8B 版不是同预算对照，`14.49 vs 108.88` 的 gap 无法归因为继承效应。**

### 因果分解（诚实，以实测预算为准）

- **1B 尺度（同预算，干净）**：armB 19.54 vs scratch14 21.25 → 继承效应仅 **−1.7**，深度效应（scratch14 vs base 全16层 = 21.25 vs 25.34 = −4.1）才是大头。这是本方向唯一干净的分解。
- **8B 尺度（预算不对等，不能做分解）**：armB 20000 步 ppl=14.49 说明**剪层继承版在 8B 尺度能训得很好**（这是真结论）。但 scratch14 ppl=108.88 **主要反映它只吃了 1.9% 的 token**——8B/14 层随机从头训在 98M tok（step2000）下几乎没起来（ppl 108 属"基本没训起来"级别，见 PPL>100=模型被污染准则）。
- **因此：`14.49 vs 108.88` 的 ~7.5× gap 绝大部分是 token 预算差（53×）造成，不是继承带来的净增益。** 想在 8B 尺度干净分解"继承 vs 深度"，必须补一个**预算对齐的 scratch14**（20000 步 / eff_bs=128 / 5.24B tok），甚至可能需要 100k+ 步让 8B 从头训收敛，才能下定论。**当前数据不支持"8B 继承带来 7× 增益"这种强 claim，别夸大。**

### 可下的诚实结论

1. **armB 3.95B（keep12+fresh2）在 8B 尺度可训、终值 ppl=14.49，且仍在缓降** → "前 j 层表征可直接复用 + 少数 fresh 层重学 NTP"这一极简架构假设在 8B 尺度**工程上成立**（继承迁移无爆炸、heal 正常）。呼应 QCMem 的 depth-j 缓存前提（前 j 层承载可迁移语义）。
2. **"继承 vs 深度"的净分解在 8B 尺度尚未完成**——8B scratch14 欠训（1.9% 预算）使其不能作为有效对照。1B 尺度的干净分解（继承 −1.7，深度 −4.1）是目前唯一可引用的因果结论。
3. **对 QCMem 主线**：8B armB 训得起来，支持"前 j 层表征在大模型尺度依然可迁移/可复用"（方向4 极简架构假设 + QCMem depth-partition 的大模型旁证）；但不宣称精简架构在 8B 上优于全模型（既缺全模型同 eval 集 ppl，也缺预算对齐的 scratch14）。

### 待办（若要补齐 8B 干净分解）
- 补 **budget-matched scratch14**（Qwen3-8B 14 层随机，20000 步 eff_bs=128，与 armB 严格同口径）。
- 补 **Qwen3-8B 全 36 层 base 在同一 slimpajama-val 上的 ppl** 作对照（当前 logs 无此口径记录）。
- armB 若续训到 200k 收敛，更新终值。
