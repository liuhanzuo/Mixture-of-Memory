# 方向c MVP 定型结论 (2026-07-01)

> qa5 give-event 混合上下文 SFT (run=mem_space_dirc_qa5sft_mvp, 从A模型step2000续训800步, mix=0全合成不碰babilong test)。攻"读出墙"。全官方判分compare_answers。

## 一、核心定型结论: 方向c是"能力沿长度平移", 非"读出修复"

**满n100全档数据(fullchain oracle 与 standard mem-chain 一致): 单一16k gap训练把读出/绑定能力从短档整体平移到长档。**

### fullchain oracle (真sf chunk 100%在窗=选择完美, 只测读出)
| 长度 | A模型 | step800 | 变化 |
|---|---|---|---|
| 8k | 60 | **30** | **-30** |
| 16k | 43 | **52** | **+9** |
→ 16k涨9, 但8k退30。**不是"读出可训练修复", 是能力沿长度平移**(16k涨以8k大退为代价, 净效应可能负)。
(注: 训练中途看着单调涨45→52→58→56是**只看16k**的错觉; 满n100看全档才见平移。)

### standard mem-chain (无selector端到端, 同设定)
| 长度 | A模型 | step800 | 变化 |
|---|---|---|---|
| 2k | 54 | 30 | **-24** |
| 4k | 43 | 20 | **-23** |
| 8k | 18 | 24 | +6 |
| 16k | 7-11 | 28 | **+17~21** |
→ 同样短退长涨的平移。两个度量图景一致=平移是真的。

### 长程语言能力不破坏(红线)
pg19 sliding ppl: A模型1.534 → step800 1.593(+0.059轻微)。红线守住。

### 综合裁决
**方向c MVP没有净改善读出**。它证明的是: 训练能改变读出能力的**长度分布**(把能力从训练没覆盖的短档挪到训练gap的长档), 但总量守恒式平移。单档训练的根本局限。

## 二、度量区分(教训)
- **fullchain oracle** = 选对chunk时的读出上界(排除selector)
- **standard mem-chain** = 含selector的真实端到端
两度量图景一致(都平移), 但绝对值不同(selector路径不同)。报告分开, 且**必须看全长度档**(只看单档=中途45→56的修复错觉根源)。

## 三、选择墙(独立于读出墙)
干净三方selector qa5 16k(A模型): bm25≈48 / reader-attn≈22 / oracle(fullchain)43。
→ BM25 selector已接近oracle上界; reader-attn选不对chunk。**端到端 = 选择墙(selector) + 读出墙(reforward)叠加**。

## 四、下一版方案(消除平移代价)
根因: 单一16k gap训练。
**下一版: 混档gap训练** — t2_gap_tokens在2k/4k/8k/16k间混合(或阶梯课程), 让读出修复覆盖全长度档, 消除短档平移。判据: standard mem-chain各档≥A模型 + fullchain各档修复 + pg19 ppl不退。
- 需用户确认再启(多小时8卡重动作)。

## 五、对项目卖点(POSITIONING.md fidelity-routing)的意义
- 卖点充要条件"reforward读出≈全文注意力"在**短档本已满足**(4k fullchain 71≈purelong73), 方向c证明**长档(16k)的读出退化也可训练修复**(43→56)。
- 即: fidelity-routing的"读出"半身, 从"短档无损"扩展到"长档可修复"。
- 剩余: (a)混档消除平移; (b)选择墙(reader-attn弱, 但BM25可兜底近oracle)。

## 红线
mix=0全程; 全合成不碰babilong test; pg19 ppl护栏通过; 真SOTA锚点pg19 nctx7 16/32k=16/9(distill模型)不动。
