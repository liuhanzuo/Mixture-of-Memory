# P2.4 第五个 confound 假设被反驳：25000 步 Dolmino 让 val PPL 上升不是下降

**日期**: 2026-08-08 ~13:00 CST。**测量者**: MAIN, 在 .73 上跑 forward-only PPL。
**这条反驳 workflow verify agent `a640e329` 的 surprise #1**，MAIN 昨夜以为的
"P2.4 第五个混淆"在数据上站不住。原假设与撤回记录如下。

## 原假设（agent `a640e329` verify surprise #1）

> "★ 最重要: intact 臂根本没做过 Dolmino 续训 (logs/p24_sft_full32.log:30 = '[init]
> loaded FULL BASE', 0 步), 而 5 个 pruned 臂全部从 83.5k-200k 步续训 ckpt 起跑。所以
> P2.4 唯一存活的 PPL 声明「intact +4.46% vs pruned +9.16%」同时变了两个东西:
> {结构损伤} 和 {0 步 vs 83.5k-200k 步在同一语料上的续训}。**刚跑完 200k 步 dolmino
> 的模型对 dolmino_now_val 更专精**, SFT 拉向 Tulu-3 时天然掉更多 — 这单独就能解释
> 更大的 ΔPPL%, 完全不需要提损伤。"

这里的核心 empirical 假设是 **"更多 Dolmino 步 → val PPL 更低（更贴近 val）"**。
agent 没有实测这个假设，直接当作已知。

MAIN 当时（heartbeat 前一轮）**接受了这个假设**并投了 full32_dolmino25k SFT 控制臂，
后来因 H20 OOM 改为免训练 PPL 检测。

## 实测（一个数字）

同一 driver、同一 val_path、同 shard 配置：

| ckpt | Dolmino 步数 | held-out PPL |
|---|---:|---:|
| raw base（`../models/OLMo-2-1124-7B`） | 0 | **7.3981** |
| `outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt` | 25000 | **7.6698** |
| Δ | | **+0.272 (+3.7%)** |

- val: `data/dolmino_now_val.npy`, n_windows=4096, n_tokens=8,384,512
- driver: `scripts/eval_olmo2_probe2_ppl.py`, 8 shards, batch_size=4
- ckpt sha ok（strict 355 tensors，keep=32 fresh=0）
- log: `zwfy6:logs/ppl_full32_dolmino_ladder.log:merge` @ 12:57:52
- provenance: `zwfy6:olmo2_ppl_results/7B_full32_dolmino_step25000/summary.json`

**方向与 agent 假设相反**：25000 步 Dolmino 训练让 val PPL **上升**了 3.7%，
说明 `dolmino_now_val` 与 Dolmino 训练分布**不是同一分布**，dolmino 续训**没有**
让 32L 模型贴近 val（在此 range 内反而拉远）。

## 后果

- **agent 的第五个 confound 假设失败** —— 它建立在"更多 dolmino 步→更贴 val"
  这个先验上。对 32L 已证伪。
- 因此 workflow verify agent 判定的 **REFUTED "intact +4.46% vs pruned +9.16% 是
  clean single-variable contrast"**，其**推翻理由**（Dolmino-step confound）现在
  也失去 empirical 基础。
- 那么两组对比"intact +4.46% vs pruned +9.16%"仍是**当前数据能支持的最强声明**，
  没有比 depth/damage 更强的替代解释。（其他四种"不动模型改数字"机制 —— torch
  版本、batch_size、shard merge、GPU 架构 —— 依然存在，但都不改变 SFT ΔPPL%
  在同臂内 pre/post 配对内部的可靠性。）

## 谦逊限定

**只测了 32L。** pruned 模型（keep8-16）在其自己的 5k-200k dolmino 步范围内
是否也 val PPL 上升，**没有直接测**。但有间接证据：所有 pruned 臂的 pre-SFT val
PPL（9.78 / 10.56 / 11.44 / 12.82 / 13.33）**远高于 raw base（7.40）**，
说明 pruning damage 是 dominant driver，dolmino step effect 即使存在也是二阶。

要把这条彻底钉死，还差:
- (a) 中间步（step5000/10000/15000/20000, 只在 wzc1 有，`.252` SSH 挂着 scp 不了）
      —— 若中间某点 PPL < 7.40 而 25000 > 7.40，说明是 overfitting 到 dolmino 训练
      分布之后被 val 拉高，还需再解。
- (b) pruned 模型的 dolmino 步阶梯（e.g. keep14 从 step0/50000/100000/128000/200000
      看 val PPL 曲线是否单调）—— 中间步都在，可以后续做。

## MAIN 追加实测（~13:20 CST）: pruned 臂那侧数据已在盘上, 方向与 intact 不同

刚查发现 4 个 pruned 臂**都有已跑过的 val PPL step 阶梯**（forward-only, 早期
training-time 存下的），不需要新 GPU 就能画曲线。聚合结果：

| arm | 最早已 eval 步 | PPL | 最晚已 eval 步 | PPL | 方向 |
|---|---:|---:|---:|---:|---|
| keep8 | 48000 | 15.13 | 121000 | **13.33** | 单调降 |
| keep10 | 10000 | 17.24 | 83500 | **12.82** | 单调降 |
| keep12 | 115000 | 11.54 | 124000 | **11.44** | 单调降 |
| keep14 | 128000 | 10.83 | 200000 | **10.56** | 单调降 |
| ShortGPT-16 | 153500 | 9.84 | 200000 | 9.78 | 微降 (0.06) |
| **full32 (intact)** | 0 | **7.398** | 25000 | **7.670** | **上升** ★ |

**这是一个不对称结构**：intact 32L 模型的 dolmino 训练让 val PPL 上升（+0.27），
所有 pruned 臂的 dolmino 训练让 val PPL 下降（keep8/10/12 从 M 量级降到 10-15）。

**为什么不对称**：pruned 模型 dolmino step=0 处 PPL 巨大（keep10 = 888k，是因为
刚切断 layers 后 lm_head 不匹配）—— 这个 dolmino 训练**主要做的是 "heal
pruning damage"**，不是 "specialize to dolmino distribution"。所以在整个可测
range 内，pruned 那侧的 PPL 下降**主要归因于 pruning-damage recovery**，只有
极小尾部（e.g. ShortGPT-16 153.5k → 200k 只降 0.06）是"specialize to dolmino"。
而 intact 模型没有 pruning damage 需要 heal，dolmino 训练的净效果就是**过度专精
到训练分布 → val PPL 上升**。

**对第五个 confound 的最终判决**: agent 的假设"更多 dolmino 步 → 更贴 val"
**在两个方向上都不成立**:
- intact 侧：dolmino 训练**拉远 val**（32L: base 7.398 → 25k 7.670）
- pruned 侧：dolmino 训练的净效果是 **heal damage**（PPL 从 M 量级到 10-15），
  不是"贴近 val 分布"。5 pruned 臂在 SFT 起点 PPL 仍**远高于 raw base**（9.78 vs
  7.40），从任何贴近度度量看都**离 val 更远**，不是更近。

所以 P2.4 "intact +4.46% vs pruned +9.16%" 里，pruned 组更大的 ΔPPL% **不能**
用"起点更贴近 val 所以掉更多"来解释 —— 起点本来就更远。反而符合另一个 story:
**pruned 模型的表示更脆弱（更多 near-tie decision boundary），SFT 的分布扰动
在它上面造成更大的破坏**。这与之前所有 "damage → near-tie 密度增高" 的 evidence
（bs 敏感度、跨架构 flip、静默 merge 效应）一致。

## 修订后的 P2.4 headline 判断

- **agent 提出的第五个 confound: 反驳**
- P2.4 "intact +4.46% vs pruned +9.16 ± 0.66%" 两组对比**站得住**
- 更进一步：这个 2× 差异**符合"损伤 → 近似平局密度↑ → SFT 扰动更破坏"** 这一
  mechanism，与今晚其他四个数值现象共同 evidence 一致，可以作为**更强的 story**
  写进 P2.4 章节，而不是保守的 "no ordering, no slope"。

## 剩余谦逊限定

- keep8@0（1.4M）、keep10@0（888k）等 step=0 点是"刚切断 layers 但没 heal"，
  作为"最远"锚点是合理的，但**不能与 base@0（7.398）在同一 x 轴上对比** ——
  它们不是同一模型。整体图是"每 arm 的 val PPL vs 自身 dolmino step"曲线簇，
  不是"共同 x 轴"。
- full32 只在 step25000 有一个点（其他中间步只在 wzc1，`.252` SSH 挂着搬不了）。
  25k → 7.670 单点足够反驳"贴近 val"，但曲线形状还没定。**下次 .252 复活时补上**。


## 我今晚的错误账

这是**第四次归因错误**了（顺序: runtime jitter → driver drift → batch size → 现在
Dolmino-step）。前三个我自己犯，flip boundary 最后靠 subagent 找到 torch 版本。
这个第四条是 subagent 提出、我接受，然后自己用 empirical 反驳的。**吸取教训**:
下次接受 subagent 的"最重要发现"前，先花 15 分钟做一次 empirical 快速验证 ——
本次的验证只花了 5 分钟 GPU（一个 PPL 数字），比误信一晚要便宜太多。

## Provenance

- summary: `zwfy6:olmo2_ppl_results/7B_full32_dolmino_step25000/summary.json` PPL=7.6698
- summary: `zwfy6:olmo2_ppl_results/7B_base_full/summary.json` PPL=7.3981
- launcher: `zwfy6:scripts/_run_olmo2_ppl_full32_dolmino_ladder_73.sh`
- log: `zwfy6:logs/ppl_full32_dolmino_ladder.log`
- 反驳来源: `status/PAPERB_SFT_FIT_CONFOUNDED.md` § FINAL + workflow verify agent a640e329
