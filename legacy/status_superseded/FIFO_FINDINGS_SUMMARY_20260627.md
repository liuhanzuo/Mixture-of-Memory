# FIFO 长程记忆 — 完整机制结论（2026-06-27 交付总结）

> 一夜系统排查的收口。所有数字均为干净 NOLEAK ckpt（babilong_mix=0），泄漏数字一律不用。
> 真 SOTA 锚点：pg19 nctx7 qa5 16k=16/32k=9。

---

## 一句话结论

**FIFO 长程记忆的 W0/W6 gap 由两半组成：(1) 读出——已被 "token-reforward" 解决（oracle 完美选时 qa1 8k 12→50，全长档）；(2) 选择——无监督选对 needle chunk 是另一半墙，frozen reader-attn precision 不够，零训练（换层/换 k）救不动，更大 k 因稀释反而更糟。**

## 完整判据链（qa1 8k，干净 NOLEAK）

| 配置 | qa1 8k | 16k | 32k | 含义 |
|---|---|---|---|---|
| 纯 memory 基线(W0) | 12 | 8 | 2 | FIFO 现状 |
| hidden-oracle(隔离 needle 的 **hidden 快照**) | 20 | 24 | 22 | 死快照 readout 到顶 ~20 |
| **oracle-token(隔离 needle 的 **原始 token** 重 forward)** | **50** | 28 | 33 | ★读出机制解决 |
| reader-attn-token k4 L16(可部署) | 11 | 5 | - | frozen 选择墙 |
| reader-attn-token k4 **L8**(换层) | 11 | 5 | - | 换层无效 |
| reader-attn-token **k6**(更高召回) | ~0(含OOM) | 0 | - | 更大 k 反而更糟(稀释) |

qa5(多 mention): oracle-token 25/15/22, reader-attn 20(温和, 多 mention 召回容忍度略高于单 needle)。

## 机制(为什么)

1. **死 hidden 快照 vs 活 token**: FIFO 存的是 chunk "当初作为 current 时算的 hidden 快照", 从没 attend 过 query → needle↔query 多跳耦合断裂。SWA/token-reforward 把原始 token 重新过 backbone, 每层重新 attend query → 读得出。(logit-lens probe: needle hidden L31 rank=21, "存了但弱"; researcher HIDDEN_VS_SWA 分析)
2. **位置坍缩是次要**: ArmC(训练时 real 位置)长档 ≈ 基线, 证伪 "pos-0 坍缩是主因"。
3. **选择是真墙**: oracle(作弊知道 needle 在哪)→ 50; frozen reader-attn 选 → 11。差距 = 选择 precision。单 needle(qa1)选不中, 多 mention(qa5)略好。

## 还没做的（等用户决策）

- **supervised-selection 训练**(status/LEARN_TO_SELECT_DESIGN_20260627.md): 用 T2 合成 needle 的已知位置监督选择器 + token-reforward 读出。**confidence MED-LOW**: 2 个 death-list 风险(① 历史所有 trained selector 崩到随机精度 ② T2 不泛化到 BABILong, rawkv_methodA 教训)。~10h, mix=0。
- 多层投票选 chunk(需改代码, 边际, 未做)。

## 旁证（干净基线）

- NOLEAK b25 W0 ≈ pg19 SOTA(qa5 8k=16/16k=7), 确认 b25 "破墙" 100% 泄漏。
- ArmA(T2 格式对齐)W0: qa1 16k 8→15, 32k 2→10, qa5 4k 12→26 — 格式对齐温和一致正向, 但不解长程瓶颈(0k 仍 0, 格式 bug)。

## 交付价值

无论是否继续训练, 这条排除链把 "hidden 记忆效果不好" 从模糊感觉 → 精确机制: **读出可解(token-reforward), 卡在无监督选择**。这是干净、可信、有完整判据支撑的负-转-正结果(读出方向是真贡献)。
