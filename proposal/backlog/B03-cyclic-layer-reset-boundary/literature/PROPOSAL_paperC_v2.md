# PROPOSAL — 新 Paper C（cyclic prune-regrow pretraining / depth cycling）综合裁决

综合来源：`AUDIT0` / `AUDIT1` / `AUDIT2` / `AUDIT3` / `SKEPTIC1` / `SKEPTIC2` / `SKEPTIC3` +
`paperA/rebuttal_snippets/R1_depth_convergence.md` / `R2_readout_bandwidth.md`
日期 2026-08-06 ｜ 本文件只由本次 synth agent 写入，未改任何 .tex / status / versions / TODOList。

---

## 1. 一句话判决

# **GO-with-narrowed-scope**（但**必须放弃「新方法/新机制」定位**，改写成
# 「**scale-and-regime boundary + 一个 CNN 上物理不可观测的代价分解测量**」；
# 且必须先过一个半天的 kill gate，gate 不过就 NO-GO 并把残值并入 Paper B）

**为什么不是 GO**：原始构造（层级 + 循环 + 终点等尺寸 + 随机新层 + 目的是最终质量/塑性）在**算子层面已被
peer-reviewed 工作逐字占据**，三个 skeptic 独立收敛到同一篇：

- **`arXiv:2202.00155` LLF (Fortuitous Forgetting), ICLR 2022, §3.2**：
  `M^l_LLF = 1 if l < L ; 0 if l ≥ L`（把第 L 层以上全部重新随机初始化），
  脚注 3 明确 `reset/reinitialize` = **a new initialization**（不是 rewind），
  Table 1 跑 **N3 / N8 / N10 代**循环，全程同一架构（终点等尺寸），目的就是泛化。
  → 我们的 `--keep_front_layers K_f` + `--n_fresh_layers (L − K_f)`（保总深度）在**参数空间里与之同一个操作**
  （同架构、同 shape、同随机初始化分布）。SKEPTIC1 §1.3 与 SKEPTIC2 §1.4 独立得出同一结论。
- **`arXiv:2109.00267` layerwise reinit (`lw`)**（**arXiv preprint**，但被 LLF §4.1 点名对照）
  Algorithm 1 第 12 行 `Reinitialize all layers above block k` + 双层 `for` = **层级 × 多轮**，
  且 §4.2 已经用「Gaussian 扰动探针」做完了 flatness 论证（= AUDIT3 推荐给我们的机制 (c) 验证路径）。
- **`arXiv:2304.04858` SEAL, CVPR 2023, Abstract**：把 LLF 称作
  `a state-of-the-art method in this category`（"iterative learning methods"）——**这个子领域有 SOTA、有专门打它的后继论文。**
- **`arXiv:2307.01163` Active Forgetting, NeurIPS 2023, Abstract**：
  `resetting the embedding layer every K updates during pretraining` + Figure 4 的
  `episodic pattern / loss spike / repeats of forget-relearn` 叙事 —— **「LLM 预训练里周期 reset 一个模块买塑性」这个 framing 已被占，连措辞都占了。**

**为什么不是 NO-GO**：三个 skeptic 全部判 **WEAKENED，无一判 REFUTED**，且各自都留了同一条缝：

- SKEPTIC1 §8：`LLF/SEAL/lw 全是 ResNet/CIFAR/Flower/Tiny-ImageNet 图像分类`；Active Forgetting 是
  **RoBERTa-base 125M 且只 reset embedding 层**；**≥1B decoder-only LM 预训练 + 层级 reset 是真空**。
- SKEPTIC2 §5 结尾亦承认："K1 的三篇全是 CNN / 图像分类 / 小数据 ... 所以技术上仍有一条缝"。
- SKEPTIC3 §5 第 4 条（它唯一攻不下来的点）：
  **「结构破坏后 PPL 可恢复而参数化知识不可恢复」这个可分离的二维代价，在 vision 分类 / RL / ≤0.1B LM 上物理不可观测**，
  它用 6 个 query（S2 3 + arXiv 2 + OpenReview 1）搜过，**未命中**占位。

**死不掉但也活不成方法论文的确切位置**：卖点必须从「we propose」搬到「we measure the boundary」。

---

## 2. 三个 skeptic 的裁决汇总

### 2.1 SKEPTIC1（vs 单调生长）→ **WEAKENED**（自评「距 REFUTED 只差『有人在 LLM 规模上做过』这一条」）

| AUDIT1 的主张 | skeptic 结论 | 收窄后的精确表述 |
|---|---|---|
| 「大→小→大的循环、终点回到原尺寸，没有一篇做过」（18 篇全单调生长） | **REFUTED（该条字面主张错）** | 生长文献那一支的判断是对的（RaPTr 的 bypass≠delete、SOLAR 的删层是一次性拼接前修整、CGLS/LESA/OpT-DeUS/LLaMA-Pro 全单调），但**「没有一篇做过」不成立**：LLF (ICLR 2022) 一篇同时满足 AUDIT1 §4 的**全部四支柱**（destructive 非 bypass / 多轮非单程 / 终点等尺寸 / 层级非权重级）+ 目的是质量 → **AUDIT1 §4「四格全占的组合不存在」的三列表格失效** |
| AUDIT1 §4 支柱 2「多轮循环而非单程，否则 = DSD 单轮」 | **REFUTED** | DSD 原文 Algorithm 1 末行 `goto Sparse Phase for iterative DSD`；§4.3 `A second DSD iteration can further improve the accuracy`；DeepSpeech Table 7/9 跑满 2 完整轮 → **「多轮」在权重级上也不是我们的差异**。另 **AC/DC (NeurIPS 2021, `2106.12379`) §3.2/§4**：`alternate between compression and de-compression every 5 epochs`，且 `if our goal is to return a dense model matching the baseline accuracy, we take the best dense checkpoint ... and fine-tune it`（多轮 + 终点等尺寸 dense + 也 reset 优化器动量）→ AUDIT1 §4 表里 DSD 那行「循环 ✗ / 终点等尺寸 ✓」被 AC/DC 变成「✓ / ✓」 |
| AUDIT1 §5 R4「plasticity 那支几乎全在 RL，LLM 预训练侧是空白」 | **REFUTED** | 三篇反例，两篇 peer-reviewed：Active Forgetting (NeurIPS 2023, RoBERTa 预训练里周期 reset embedding)、`2606.24752`（preprint，5M–314M GPT-style 上做完 dormant unit / lazy head / effective rank 全套 correlate，并把 reset 类列为**综述里的标准 mitigation 家族**）、`2602.11137` Weight Decay Improves LM Plasticity（把 LLM 预训练超参当作最终塑性自变量的 RQ 已被占）。**AUDIT1 建议「借 RL 侧标准量填 LLM 空白」这个空位已被占** |
| （AUDIT1 引得准、skeptic 核实无误的部分） | **SURVIVES** | RaPTr 的 `maintain a common base model of interest`（§2 P1）、`drops fewer layers as training proceeds`（§3）逐字核对一致 → **「bypass ≠ delete」+「RaPTr 是单调、方向与 PLD 相反」这两个判断成立** |

**SKEPTIC1 给出的可活表述（§8）**：
> 在 **decoder-only LLM 的大规模预训练/续训**（7B 量级、Dolmino/DCLM 级 token 预算）上，
> 周期性把 top-K transformer 层随机重初始化，把评价落在**知识型能力与 PPL 的分离**上——
> 检验 LLF 类破坏式 forget-and-relearn 课程在**知识密集的 LM 预训练**中是否仍有效，
> 还是会被 Paper B 已量化的「PPL 能 heal 但知识不能」吞掉。

### 2.2 SKEPTIC2（vs 循环 / 权重级）→ **WEAKENED**（自评「接近 REFUTED」）

| AUDIT2 的主张 | skeptic 结论 | 收窄后的精确表述 |
|---|---|---|
| 「层级 + 循环 + 训练期 + 终点原尺寸从没人做过，只有 filter 级 RePr」 | **REFUTED** | LLF (ICLR22) / SEAL (CVPR23) / `lw` 三篇就是层级循环；AUDIT2 把 `2109.00267` 标成「非循环、非层级 regrow」是**读错**（那是该论文主贡献） |
| 「RePr Table 3 证明结构级(6.9) > 权重级 DSD(7.8)」→ 支撑「换粒度会换机制」 | **REFUTED（读反了原文）** | RePr 原文紧接 Table 3：`DSD and RePr (Weights) perform roughly the same function ... Thus, we observe similar performance between these techniques`（7.8 vs **7.7**）；6.9 那档来自**换 metric（inter-filter orthogonality）**。→ **粒度本身几乎不带增益，文献证据对「粒度重要」是 negative**。⚠️ **这条误读若照抄进 .tex 即 misrepresentation，reviewer 查原文即破** |
| 「DSD 从没在 transformer/LLM 上被复现，全库只 6 条命中，是最强空白证据」 | **REFUTED（检索措辞造成的假空白）** | `all:"sparse pre-training and dense fine-tuning"` 一次命中 **SPDF, UAI 2023 (`2303.10464`)**：`induce up to 75% sparsity into a 1.3B parameter GPT-3 XL model`（sparse-pretrain → dense-finetune = DSD 的 LLM 版）。诚实限定：SPDF 是**一次性、非循环、动机是训练 FLOPs、粒度非结构化权重**，故**不直接占我们的构造**，但它摧毁「规模真空」这条论证 |
| 「没有任何一篇在 ≥1B LM 上做过 plasticity 干预并报告下游能力」 | **REFUTED** | **Springer et al., ICML 2025 (`2503.19206`)** 在 **OLMo-1B/3T + OLMo-2-7B + Amber-7B** 上做 catastrophic overtraining + 机制（progressive sensitivity）+ 理论；`2502.07274` Cho et al.（**S2 `venue=''` → preprint**，他人引作 ICLR 2026）在 >1B 上给了维持塑性的算法。**这两条就写在 AUDIT2 声称精读过的 `2606.24752` 的 §II 里 → 是阅读失误不是检索失误** |
| 「层级 / 循环 / 终点原尺寸 / 目的是质量」四个差异点 | **四个全部 REFUTED 为无效差异** | LLF/SEAL/lw/KE/RePr/DSD 全部终点原尺寸、全部循环、全部目的是最终质量；层级是这个 subfield 的**默认轴**不是新轴 |
| 「与 Paper B 知识/PPL 不对称衔接」（AUDIT2 的 ★★★） | **SURVIVES（唯一真差异）**，但被反向解读 | **同一个事实读作先验时是最强的失败预测**：我们已自证单次层级损伤的知识税近乎不可逆 → cycling = 反复交这个税 → **先验预测：知识侧随轮数 N 单调变差** |

**SKEPTIC2 的可活表述（§7.1）**：
> "Cyclic layer-level reinitialization does not transfer to LLM pretraining, and we explain why."
> 三段：(1) 层级周期 reinit 在小数据图像分类上是有效正则，机制被原作者归因为**抑制困难样本的记忆化**，
> 且原作者已报告大数据下收益消失、迁移性能下降；(2) LLM 预训练在制度上位于该机制的失效区
> （单遍、不过拟合、且**知识记忆本身是目标而非病灶**）；(3) 我们给出该家族方法的**首个 ≥1B 规模化检验**，
> 并用 **PPL vs 知识两轴分离**（CNN 分类上物理不可观测）定量给出失效的**代价结构**。

⚠️ **SKEPTIC2 §7.3 的工程警告（必须传给 coder）**：
AUDIT2 §7.3 说「缺的只是循环调度 + optimizer state 局部重置两块胶水代码」是**错判**——
只加胶水 = 复现 LLF。要做出「从深度栈中间挖 K 层」的差异，需要**新写「任意 index 集合 block 移除 + 指定位置插入新 block」**的能力。
（**部分现成**：`scripts/train_olmo2_shortgpt_fresh.py` 已有 `--keep_layer_indices` / `--selection_json` 支持任意 index 保留 + `--n_fresh_layers` 追加 fresh 尾，
但**新层只能加在尾部**，仍不支持「在中间位置插入」。这一点 MAIN 派 coder 前必须知道。）

### 2.3 SKEPTIC3（vs plasticity 机制）→ **WEAKENED**

AUDIT3 保留了四扇门，skeptic 逐一裁决：

| AUDIT3 的门 | skeptic 结论 | 杀手 / 收窄后表述 |
|---|---|---|
| ★★★ 门1「regime 边界：forget-and-relearn 在大数据/大模型上是否消失，文献明确没答」 | **REFUTED** | `2109.00267` §5 结论段最后一句：`For large datasets, however, reinitialization does not seem to offer a benefit.`；§3.1 决策树的分裂特征字面是 `Training Set Size ≥ 35K?`。**AUDIT3 引了这篇（转引自 2307.01163 的 flatness 半句）却没读它的结论**。另 LLF 自己 §A2.4/Table A8：换 WRN-28-10 / DenseNet-BC 后 `we do not see any improvements from LLF over our baselines`（CIFAR-10 96.32→95.91，CIFAR-100 81.29→80.95，**LLF 更差**）→ **收窄：不是「开放问题」，是「已知 regime 边界的一次外推确认」，属 replication 而非 finding** |
| ★★★ 门2「PPL 能 heal 而知识不能，是该文献里不存在的现象」 | **SURVIVES（唯一活着的）** | 用 6 个 query 未命中占位（含 S2 `knowledge recovery layer pruning healing large language model` total 168 全是压缩向）。但收窄：**它是 Paper B 的既有资产、不是 cycling 的产物**，且其 trade-off 措辞已被 **FIRE (ICLR 2026 Oral)** 的 background 句抢先：`conservative reinitializations fail to restore plasticity, while aggressive ones erase useful knowledge` → **可主张的是「测量」不是「机制发现」** |
| ★★ 门3「时机（when in training）无人系统扫过」 | **REFUTED** | **Springer et al. ICML 2025 §3.3**：`Progressive sensitivity to noise: For a fixed magnitude of perturbation, the change in perplexity between the base model and the perturbed model increases monotonically with the number of pre-training tokens.` 且噪声协方差取的正是**初始化分布的协方差** → 我们的「丢 K 层补随机 K 层」= 该操作在 λ→1 且只作用于后 K 层的**离散极端情形**；magnitude 依赖亦已扫（大扰动 → inflection point 出现在更低 token budget）→ **brief 里「早期做代价小」这条猜测已被预测且方向一致**。更早的占位链：**Critical Learning Periods, ICLR 2018**（`Final test accuracy as a function of the onset of a short 40-epoch deficit`）+ `1905.13277`（`not just whether or how, but when to regularize`） |
| ★ 门4「LLM 缺 replay buffer，RL reset 机制不可外推」 | **WEAKENED 到不足以支撑** | **DASH, NeurIPS 2024 (`2410.23495`) Appendix C.1**：`Table 2 shows L2 INIT (Kumar et al., 2023) and Reset (Nikishin et al., 2022) cannot be a solution in our setting`（stationary 数据下 RL reset 无效，机制归因是 **noise memorization** 不是 replay buffer）→ 我们复述「不可外推」= 复述 NeurIPS 2024 的一张附录表；且这是**对我们方向的直接负面先验**（LLM 预训练近 stationary） |

**SKEPTIC3 的额外净效果（对我们有利的两条）**：
1. **RaPTr 应上调为 ICLR 2024**（AUDIT3 误标 preprint；S2 HTTP200 `type=conference, year=2024, 10 cites`）→ 我们「层级扰动能改善 inductive bias 而非只加速」这条弹药**更硬**。
   ⚠️ 但 AUDIT3 引的 "1.5%" 是**引数错误**，原文 ar5iv 是 `1-5%` / `1-2% better than baseline and stacking` —— 不要照抄。
2. **DASH §5**：`Table 1 shows that warm-starting with SAM does not outperform cold-starting with SAM, indicating that SAM alone is not an effective method in our case` → **B1(SAM) 在 warm-start/stationary 口径下威胁等级可下调半档**。

---

## 3. 若 GO：精确的构造定义（narrowed scope 版）

### 3.1 命名与定位（先定，避免写作时又滑回「新方法」）

- **不叫** cyclic prune-regrow / depth cycling / layer scaling curriculum（前者撞 LLF/lw，后者撞 CGLS）。
- **叫** *"Layer-Reset Curricula at LM-Pretraining Scale: a regime-boundary audit"*，
  论文的 contribution 句式是 **"we test whether X transfers, and we decompose the cost"**，不是 "we propose X"。
- **LLF 必须在 Abstract / Intro 第一段就被点名为 prior method**，我们的臂 R-top 显式标注为「LLF 的 LM-pretraining 复现」。

### 3.2 构造（三个可执行变体，R-top 是**复现**不是我们的方法）

设基座深度 $L$，每轮丢弃/重置 $K$ 层，循环 $N$ 轮，两阶段步数 $T_1$（full）/ $T_2$（reset 后 relearn）。

| 变体 | 操作 | 定位 |
|---|---|---|
| **R-top**（= LLF） | 每轮把 layer index $[L-K, L)$ 原地重新随机初始化（等价于「丢顶 K 层 + 尾部补 K 随机层」） | **prior-method 复现臂**，不是我们的贡献。用现成 `--keep_front_layers=L-K --n_fresh_layers=K` |
| **R-mid**（唯一有构造差异的臂） | 每轮从**深度栈中间**移除一个连续 index 段 $[a, a+K)$（$a$ 由下面的 probe 规则定），并把 K 个新层插回**同一位置**（不是尾部），总深度守恒 | **需要新代码**（现有 flag 只能 top-K 或任意保留 + 尾部补新）。这是与 LLF 的**唯一构造级差异**，必须能用「层级功能分工」正面论证，否则退化成 LLF |
| **R-rand** | 每轮随机抽 K 个非相邻 index 重置 | 隔离「位置是否重要」，同时是 RePr §5「metric 重要」在层级的对应实验 |

**位置选择规则（R-mid 的 $a$）**：用我方独家 probe 资产把 $a$ 钉在 **knowledge-onset 之下 vs 之上** 两个物理机制上：
`results/knowledge_logit_lens_OLMo-2-1124-7B.json` 给 OLMo-2-7B onset L18（0.562L）、L18→L19 由 **0.326 → 0.544**；
Qwen3-8B onset L25（0.694L）、L24→L25 由 **0.236 → 0.621**（R1 已核实同源磁盘 JSON）。
→ R-mid 至少两个 arm：$a$ 落在 onset 之下（$a{+}K \le 18$）与跨过 onset（$a < 18 < a{+}K$）。
**这是 LLF 的 ResNet block 3/4 阈值没有对应物的地方**，也是 SKEPTIC2 §2 说的「RePr 的真正 lesson 是 metric > 粒度」在层级上唯一还没被 Paper B 吃掉的 metric 轴（Paper B 吃掉的是 keep-front / ShortGPT 的**保留集合**选择，不是**循环中每轮重置哪一段**）。

### 3.3 循环调度（数字先钉死，避免各臂 schedule 不可比）

- **轮数**：$N \in \{0, 1, 2, 3\}$。理由：DSD `Adding more DSD iterations has a diminishing return`、RePr §6 `two to three iterations is sufficient`、LLF 用 N3/N8/N10。**N=0 是 compute-matched 的纯 continue-train，必须存在**（LLF Table A7/A8 的关键 baseline 是 `Smth long (N3/N10)`）。
- **总步数固定**（compute-matched 是硬约束）：$T_{\text{total}}$ 固定，$T_1 + N\cdot T_2 = T_{\text{total}}$，每轮之间**共享同一条 cosine LR schedule 的连续段**（不重启 schedule；重启 LR 会引入 SKEPTIC3 §1 里 Springer 已扫过的 LR 混淆，且 `2605.02105 §3.4` 报告 peak LR / 退火长度本身就改结果）。
- **重置的粒度必须包含 optimizer state**（RePr §4 原文点名：`it is important to re-initialize the learning rates corresponding to the weights that are part of the pruned filters. Corresponding Batch Normalization parameters must also be re-initialized.`）→ 重置层的 AdamW `exp_avg` / `exp_avg_sq` + 该层所有 norm 参数一并重置。
  ⚠️ 另 `2606.24752 §IV`：他们每个 task 起点 reset optimizer，`implies that the plasticity degradations we observed were due to inherent plasticity loss in the weights rather than simply stale optimizer states` → **不重置 optimizer 会让结论无法归因到权重**。
- **每轮必须是新的随机 draw**（LLF §5.1 实测：固定同一初始化 `much worse than the version with a different reinitialization each generation`）→ 我们照做，并把「固定 draw」作为一个 ablation（这是**复现 LLF 的 ablation**，不能当我们的发现）。

### 3.4 终点必须与哪个 baseline 同尺寸 / 同参数量 / 同 serving cost

**终点模型 = 与 N=0 臂（同一基座、同一深度 $L$、同一 hidden/vocab、同 token 预算、同 LR schedule、同 seed、同数据顺序）逐字节同 shape。**
- 参数量匹配的算法：`num_hidden_layers` 恒等 $L$，embed / final norm / lm_head 不动；每轮重置只改**值**不改 **shape** → 参数量恒等，**无需推算，可 assert**。落地检查：训练脚本已有的 `assert len(keep_keys) == N_NONLAYER_KEYS` / `assert missing_layer_ids == expected_fresh_ids` 系列（`scripts/train_olmo2_arch_probe2.py:197-208, 304-322`）直接复用。
- serving cost 匹配：同 $L$、同 hidden、同 head 数 → 单 token FLOPs 与 KV footprint 恒等，**不需要 latency bench 来证**（这一点 DSD 自己也主张 `doesn't change the network architecture or incur any inference overhead`，**故不得当亮点写**，AUDIT2 §5-Q2 第 4 条已警告）。
- 训练 FLOPs 匹配：见 §4.4。

### 3.5 基座模型 + 数据（只用盘上有的，已实测确认）

| 阶段 | 模型 | 数据 | 磁盘证据 |
|---|---|---|---|
| **Gate（§6）+ 主实验** | **OLMo-2-0425-1B**（`num_hidden_layers=16, hidden=2048, vocab=100352`，实测自 config.json） | `data/dolmino_now15b.npy`（**62.0 GB**，uint16 → ~31B token）；held-out `data/dolmino_now_val.npy`（33.5 MB） | 两文件均在 wzc1 实测存在；`data/dolmino_olmo2_shards/` 另有 119 GB 分片 |
| **（若 gate 过）scale 验证** | **OLMo-2-1124-7B**（$L=32$） | 同上（与 Paper B **同源同 tokenizer**，故 PPL 可与 Paper B 的 base 7.398 / keep14 10.561 直接同口径对照） | Paper B 全部 200k-step 臂就用它 |
| **OOD PPL（必报，防「只在 in-domain 有效」）** | 同上 | `data/ood_ppl/pg19_test.npy`、`data/ood_ppl/wikitext103_test.npy` | 实测存在，且 Paper B B-P1.2 已用过同一套 |
| **单遍 vs 多遍制度轴（SKEPTIC2 §7.2 最有科学价值的臂）** | 1B | 同一 Dolmino 分片**限流到 1/8 token 数并重复 8 epoch**（制造过拟合制度） vs 全量单遍 | 用 `--max_rows` 限行即可（`train_olmo2_arch_probe2.py` 已有该 flag） |

⚠️ **不要用 `train_olmo2_arch_probe2_distill.py`**：module 级 `import bitsandbytes` 把节点锁死在 .73/.104，且其 `_classify_param` 缺 `module.` 前缀剥离 → **差分 LR 是 no-op**（CLAUDE.md 已记；`train_olmo2_arch_probe2.py:420-450` 已修）。
⚠️ **本项目的 1B keep7 ckpt 不在两盘上**（实测 `outputs/*1B*` 与 zwfy6 同名目录均为空，只有 `scripts/_run_olmo2_probe2_downstream_8gpu.sh:34-37` 引用了 `outputs/olmo2_probe2_1B_keep7fresh2_16card/` 的路径）→ **1B 臂必须重训，不能复用**。

---

## 4. 必须的对照臂（这是能否发表的核心）

### 4.1 臂表（每条注明「排除什么混淆 / 被哪篇打」）

| # | 臂 | 排除的混淆 | 不做会被谁打 |
|---|---|---|---|
| **A0** | **N=0 compute-matched 纯 continue-train**（同 token、同 LR schedule、同 seed、同数据顺序） | 「只是多训了」 | LLF Table A7/A8 的 `Smth long`；DSD Table 2 的 LLR 行；RePr Fig.1 orange line。**没有它任何正结果都无效** |
| **A1** | **R-top = LLF 精确复现**（top-K 原地 reinit，N∈{1,2,3}） | 「我们的方法 = LLF」 | `2202.00155` §3.2。**必须自己先跑出来并承认等价**，否则 reviewer 一句话归零 novelty |
| **A2** | **R-mid（onset 之下）+ R-mid（跨 onset）** | 「位置无关」/「层级 metric 不存在」 | RePr §5（metric > 粒度）；SKEPTIC2 §2 |
| **A3** | **R-rand（随机非相邻 index）** | 「连续段 vs 散点」 | 同上 |
| **A4** | **bypass-not-delete**（同 step 同 K 层被跳过但权重保留 = RaPTr/LayerDrop 式） | 「销毁」这个变量 | RaPTr **ICLR 2024**（`bypass`，权重从未离开模型）；LayerDrop ICLR 2020；Stochastic Depth ECCV 2016 |
| **A5** | **delete-and-copy**（丢 K 层但补**复制层**而非随机层） | 「随机性」这个变量 | SOLAR DUS (NAACL 2024 **Industry Track**)、LLaMA-Pro (ACL 2024 main)、TLI。⚠️ LLF §5.1 已做过「固定 draw vs 每轮新 draw」，我们这条是**不同变量**（copy vs random），不重复 |
| **A6** | **shrink-and-perturb 同周期同轮数**（θ←λθ+noise 全网，λ=0.4/noise 1e-4） | 「必须是层级、必须是离散结构」 | **Ash & Adams `1910.08475`**（§4 标题字面 `Shrink, Perturb, Repeat`；SKEPTIC3 已把它上调为 **NeurIPS peer-reviewed**，S2 首条 271 cites，⚠️ **勿引 24-cite 的旧标题条目**）。**AUDIT3 §4 判定这是生死线** |
| **A7** | **DASH**（NeurIPS 2024，方向感知 shrink） | 「打赢 S&P 就够了」 | SKEPTIC3 §3.C2：**DASH Figure 1 已在同一张图里打赢 S&P** → 生死线比 AUDIT3 说的高一档 |
| **A8** | **SAM 退火期**（只在 LR 退火期换 optimizer） | 「改 optimizer 就够了」 | `2605.02105`（**S2=arXiv preprint，作者 COMMENT 自述 ICML 2026，SKEPTIC3 独立核实 S2 侧仍是 preprint → 必须标 preprint**）。⚠️ 威胁等级可下调半档（DASH §5：SAM 在 warm-start 口径无效） |
| **A9** | **weight decay sweep**（0.1 / 0.3 / 0.5 / 1.0） | 「收益 = wd 从 0.1 调到 0.5」 | `2602.11137`（**preprint**）：OLMo-2-1B-140x 上 wd=0.3/1.0 的 val CE 更差（2.6208/2.7064 vs 2.6088）但 fine-tune 后明显更好。**改一个标量的对手，成本最低** |
| **A10** | **peak LR × 退火长度**（各 2 档） | 「零成本超参就够了」 | `2605.02105 §3.4`：`Shortening WSD annealing also helps: 10% annealing beats the base-model-optimal 20%` |
| **A11** | **LayerNorm Scaling (LNS)** | 「深层退化的根治是改 LN 不是循环丢层」 | **Curse of Depth, NeurIPS 2025 (`2502.05795`)**。⚠️ 同时是**动机反噬点**：若引 CoD 说「深层本来低效所以丢掉便宜」，就同时承认「我们丢的是本来没干活的层」→ **两种叙述不能同时用** |
| **A12** | **权重级同 FLOPs 对照（SPDF 式 sparse→dense）** | 「层级 vs 权重级」 | **SPDF, UAI 2023 (1.3B)**（不是只 cite DSD）；`2508.00212`（preprint）「reinit weights 比 reinit units 在更多设定下稳」 |
| **A13** | **`lw` 式 bottom-up 递减**（每轮少 reinit 一层） | 「调度形状」 | `2109.00267` Alg.1；LLF Table 1 自己做了这个对比 |
| **A14** | **剪了不长回**（终点更浅，同 token 预算） | 「等尺寸这个约束有没有用」 | 全部 pruning 文献 + 我方 Paper B keepN ladder（可直接当参照，无需重训） |
| **A15** | **单遍 vs 多遍（重复数据）制度轴** | 「这个机制只在过拟合制度有效」 | `2109.00267 §5` + §3.1 决策树（`Training Set Size ≥ 35K?`）。**这是最有科学价值的一臂，也是「regime boundary」定位的实证支柱** |

### 4.2 对齐铁律（Paper C v1 就死在 lr mismatch）

**每一臂必须逐项相同，且必须在 `arch_meta.json` 里落账后再启动**：
- `lr`（**新层 LR == 旧层 LR**，差分 LR 只能作为显式 ablation 且必须先确认不是 no-op）
  ⚠️ DSD §2 的 re-dense 用 **1/10 LR** 且 pruned 权重初始化为 **0** → 我们补随机新层若用同 LR，**不是 DSD 的对照**，必须分别列臂。
- `optimizer`（AdamW，fp32 master）、`weight_decay`、`grad_clip`、`warmup`、`cosine` 参数、`seq_len=2048`、`eff_bs=128`、`seed`、**数据顺序（同一 npy + 同 sampler seed）**。
- Paper B 的既有 recipe 可直接复用：`scripts/run_olmo2_7B_keepN.sh` 头注记录了 keep14 的权威配置
  `world_size=8 bs=16 gaccum=1 eff_bs=128 seq_len=2048 lr_fresh=1e-4 lr_inh=2e-5 max_steps=200000 warmup=150 fp32 master + grad_checkpointing`。
  ⚠️ **注意 keep14 用的是差分 LR（1e-4 / 2e-5）**，所以「与 Paper B 同口径」和「各臂 LR-matched」是**两个不同要求**：主对照必须 LR-matched（单一 LR），与 Paper B 的横比只能在**同为差分 LR 的臂**之间做。

### 4.3 常量地板（项目铁律，必报）

- MMLU chance = **0.25**（4-way）。Paper B 实测 keep8 到 200k **MMLU 始终没超 chance**。
- closed-book QA 的恒定拒答/常量基线（教训：某 SQuAD val 有 **49.85%** 是同一句拒答，常量函数 EM 就 49.85）。
- PPL 的「不看输入」上界：$\ln|V| = \ln 100352 = 11.52$ nat。
- ⚠️ 引用 `2503.04429` 的 MMLU 0.373→**0.004** 时必须标「**低于 4-way chance 0.25 → 是格式崩塌不是纯能力损失**」（R2 §3.3 已警告）。

### 4.4 参数量与 FLOPs 怎么算（写进 appendix，reviewer 一定问）

- **参数量**：恒等，`assert` 而非估算（见 §3.4）。
- **训练 FLOPs**：$6 \cdot P \cdot T_{\text{tokens}}$ 的标准近似**在此处足够**，因为所有臂的 $P$ 与 $T_{\text{tokens}}$ 逐项相同（同 $L$、同 eff_bs、同 seq_len、同 max_steps）→ **FLOPs 匹配是构造性成立的，不需要测**。
  唯一例外：A4（bypass）在被跳层的 step 上前后向都省了那 K 层 → **A4 的 FLOPs 天然更低**，必须**按 token 数补偿**（延长 A4 的 max_steps 使总 FLOPs 相等）并在表里注明补偿方式。
- **实测挂钩点**（可直接引，不需新 bench）：7B keep14 在 8×L20A 上 `1.56 s/step maxmem=122.3GB`（`logs/olmo2_7B_keep14fresh2.log` 尾部实测）→ eff_bs 128 × seq 2048 ⇒ **168 ktok/s/node**；200k step = **86.7 node-hours = 693 GPU-hours**。

---

## 5. 预注册 kill 条件（具体数字，5 条）

**在跑之前写进文件、不可事后修改。任何一条触发 → 该定位死，按 §5 尾部的降级路径处理。**

**K1（生死线，最先判）— 必须打赢 non-architectural 对手。**
若在 1B / Dolmino / 同 FLOPs 下，**R-mid 或 R-top 的最佳 N**（N∈{1,2,3}）在 **held-out Dolmino PPL** 上
**不能比 A6(S&P 同周期同轮数) 与 A9(wd sweep 最佳档) 中较强者改善 ≥ 0.03 nat/token**（≈ PPL 相对改善 3%），
**且** 在 **MMLU-C + PopQA/TriviaQA/NQ-open 的 4-任务均值**上不能改善 **≥ 1.5 points**，
→ **「必须动结构」不成立，方向作为方法论文死**（AUDIT3 §4 与 SKEPTIC3 §0 均判此为生死线；SKEPTIC3 把门槛抬到 A7=DASH，故 DASH 若跑得起也要一并过）。

**K2 — cycling 必须打赢「不 cycling」。**
若 **N=3 在任一指标上都不能超过 A0（N=0 compute-matched）超过其 seed 噪声的 2σ**，
→ **循环本身无效，方向死**（SKEPTIC2 §9 原话：「如果 N=3 打不过 N=0，方向当场结束，不要再投入 40 卡」）。
σ 的取法：A0 与最佳 cycling 臂各跑 **≥2 seed**（1B 尺度可负担），σ 用两 seed 极差 / 2 的保守估计。

**K3 — 必须有构造级差异，不能是 LLF 的换域。**
若 **R-mid（含 onset-crossing 与 onset-below 两 arm）在所有指标上都与 A1(R-top=LLF) 落在彼此的 2σ 内**，
→ **我们没有构造差异，只是 LLF 的 LM 复现**；此时**不得作为独立论文**，只能作为「LLF 的规模化复现报告」并入 Paper B 的一节。
（这是 SKEPTIC1 §9 与 SKEPTIC2 §1.4 的共同要求。）

**K4 — 知识税不能随轮数累积（这是我们自己数据给出的先验失败模式）。**
若 **知识轴（MMLU-C + 3 个 closed-book QA 的 4-任务均值，均报常量地板）随 N 单调下降且 N=3 相比 N=0 下降 ≥ 3 points**，
→ **「cycling 是更好的预训练方法」彻底死**（先验支持这个结果：Paper B keep14@200k 的恢复分数
MMLU-L **19.4%** / MMLU-C **60.2%** / PopQA **55.3%** / TriviaQA **46.2%** / NQ-open **29.3%**，
按 `paperB/sections/tab_main_results.tex` 的 base/keep14/floor 三元组现算，**MMLU-L 的 19.4% 与 brief 的 19.5% 一致**，
`19.5` 字面在 paperB/sections/*.tex 里 grep 不到，是**恢复分数派生量**，写作时必须报口径）。
**但 K4 触发时方向不必全死**：它正是 §7.1 的 negative-result 定位所预测的，可转 measurement 论文——**前提是 K5 也满足**。

**K5 — 若转 negative/measurement 定位，代价分解必须真的分离。**
若 **PPL 恢复曲线与知识恢复曲线的分离**在 1B 上不显著（定义：在同一 $N$ 下，
「PPL 相对 A0 的退化百分比」与「知识 4-任务均值相对 A0 的退化百分比」之差 **< 10 个百分点**，
bootstrap 95% CI 覆盖 0），
→ **门2（唯一存活的门）在 1B 上不成立**；此时必须要么上 7B 复测（+693 GPU-hours/臂，需重新审批），
要么**整个方向 NO-GO**（因为门1/门3/门4 已被 SKEPTIC3 判死，门2 是唯一支柱）。

---

## 6. 第一步最小赌注实验（半天出 go/no-go）

### 6.1 配置

| 项 | 值 | 依据 |
|---|---|---|
| 节点 | **1 个节点，8 卡**。首选 **LOCAL（8×L20A，183GB/卡，实测 8 卡全空闲 0 MiB / 0%）** 或 **.252（8×B200）** | 都在 wzc1，与 `data/dolmino_now15b.npy` 同盘，免 rsync |
| 模型 | **OLMo-2-0425-1B**（$L=16$, hidden 2048, vocab 100352） | 盘上实测存在；**1B keep7 ckpt 两盘皆无 → 必须新训** |
| 数据 | `data/dolmino_now15b.npy`（训）+ `data/dolmino_now_val.npy`（held-out PPL） | 与 Paper B 同源 → PPL 口径可继承 |
| 脚本 | `scripts/train_olmo2_arch_probe2.py`（**不是** distill 版）+ 新写的循环调度 wrapper | `--keep_front_layers` / `--n_fresh_layers`（断言真随机初始化）/ `--random_trunk` / fp32 master / grad-ckpt 现成 |
| recipe | `seq_len=2048`, `eff_bs=128`（bs×gaccum×8=128）, AdamW fp32 master, cosine, warmup 150, **单一 LR = 2e-5（LR-matched，不用差分）**, wd=0.1, grad_clip=1.0, seed=42 | 与 Paper B keep14 recipe 一致，只把差分 LR 拍平 |
| **总步数** | **$T_{\text{total}}$ = 6,000 step**（= 1.57B token presentations），每臂固定 | 见下面时长估算 |
| 调度 | $T_1 = 3000$；N=3 时 $T_2 = 1000$（3000+3×1000=6000）；N=0 时纯 6000 | compute-matched 构造性成立 |
| $K$ | **4**（= $L/4$，对齐 LLF 的 block 粒度直觉） | |
| 臂数 | **6 臂**（下表） | |

### 6.2 6 个臂（gate 只跑最小可判集）

| 臂 | 内容 | 判什么 |
|---|---|---|
| G0 | **A0** N=0 纯 continue-train，seed 42 | K2 的分母 |
| G0b | **A0** N=0，seed 1234 | σ 估计（K2 需要） |
| G1 | **A1 = R-top（LLF）** N=3, K=4, 重置 layer [12,16) | K3 的对照 + prior-method 复现 |
| G2 | **R-mid（onset-below）** N=3, K=4, 重置 layer [6,10)（1B 的 knowledge onset 需先用现成 `knowledge_logit_lens` harness 在 1B 上跑一次定位；若 onset ≈ 0.56L ⇒ L9 附近） | K3 的我方臂 |
| G3 | **A6 = S&P 同周期同轮数** λ=0.4, noise 1e-4, 在同样 3 个时点做 | **K1 生死线** |
| G4 | **A9 = wd=0.5**（单档，先探最强对手方向） | **K1 生死线** |

> **为什么不在 gate 里放 SAM / DASH / LNS / SPDF**：它们是 K1 的**次强**对手（DASH §5 已报 SAM 在 warm-start 无效），
> 且实现成本更高。gate 只用 **S&P + wd** 这两个「改几行 / 改一个标量」的对手——**如果连这两个都打不过，后面不用跑。**

### 6.3 GPU 小时估算（基于实测）

- 实测锚点：**7B** keep14 在 8×L20A 上 `1.56 s/step` @ eff_bs 128 / seq 2048（`logs/olmo2_7B_keep14fresh2.log`）。
- 1B 与 7B 的参数比 ≈ 1:6（`hidden 2048/L16` vs `hidden 4096/L32`）→ 保守按 **1/5 的 step time = 0.32 s/step**（不按 1/6，留 20% 余量给小模型的固定开销占比上升）。
- 单臂 6,000 step ≈ **0.53 node-hour ≈ 4.3 GPU-hour**。
- **6 臂串跑 ≈ 3.2 node-hour ≈ 26 GPU-hour**；若 6 臂两两并行（8 卡分 3 组不划算 → 直接串跑）**~3.2 小时**。
- 加 eval：held-out PPL（`scripts/eval_olmo2_probe2_ppl.py`，`data/dolmino_now_val.npy`）+ MMLU-C/L + 3 个 closed-book QA（`scripts/eval_olmo2_probe2_downstream.py`），6 臂 × 8 卡分片 ≈ **1 小时**。
- **合计 ≈ 4.5 小时 / 约 36 GPU-hour，单节点，半天内出结论。** ✅

### 6.4 判定阈值（gate 版，比 §5 宽一档，只用来决定「要不要继续投 7B」）

| 判定 | 阈值 | 动作 |
|---|---|---|
| **PASS-strong** | max(G1,G2) 相对 max(G3,G4) 在 held-out PPL 上 **≥ 0.03 nat/token** 改善 **且** 4-任务知识均值 **≥ 1.5 pt** 改善；**且** max(G1,G2) 相对 G0 超过 \|G0−G0b\| 的 2 倍 | 继续：补齐 §4 全臂 + 上 7B |
| **PASS-weak** | 只满足 PPL 一侧 或 只满足知识一侧 | 只补 **A7(DASH) + A8(SAM) + A15(单遍 vs 多遍)** 三臂再判一次；**不上 7B** |
| **FAIL-K3** | G2 与 G1 在所有指标上落在 2σ 内 | **无构造差异** → 转「LLF 规模化复现」并入 Paper B，不开新论文 |
| **FAIL-K1/K2** | G3 或 G4 打平/更好；或 G1/G2 打不过 G0 | **方法定位死** → 走 §7.1 negative/measurement 定位，且**必须先确认 K5**（在 gate 数据上直接算 PPL-vs-知识分离度）；分离度不足 → **NO-GO** |

**gate 的额外必做（零 GPU 成本，但不做会毁掉 K3）**：先在 1B 上跑一次 `knowledge_logit_lens` 定位 onset 层，
否则 G2 的 $a$ 是拍的。7B/8B 的 onset 已有（OLMo-2-7B L18 = 0.562L；Qwen3-8B L25 = 0.694L），
**1B 的没有** → gate 前先补，约 10 分钟单卡。

---

## 7. Paper A rebuttal 弹药就绪度

### 7.1 R1（三方法论独立收敛，depth structure 非 probe artifact）

| 项 | 判定 |
|---|---|
| **可直接用** | ✅ **§3 的 228-word LaTeX 段落可直接贴 response letter**。三篇全文均抓到（51.8k/72.0k/90.0k chars），venue 三路核实完毕：`2606.07978` / `2510.18871` / `2607.25663` **全部 arXiv preprint**（与 AUDIT0 主表一致；AUDIT0 额外指出 `2510.18871` 有 KnowFM workshop 录用 + ARR 在审 + ICLR 2026 **Rejected_Submission**，主会无录用 → 判 preprint 无误）。段落已写 "three concurrent preprints"，未暗示 peer-reviewed。 |
| **需补什么** | (1) 段落里 `\citet{mechlens2026}` 等三个 key 与 §4 的 .bib 条目要一起入库（§8 已列）。(2) 若要从「三篇」扩到「四—五篇」，**Csordás et al. `2505.13898` "Do language models use their depth efficiently?"** 是最该抓的下一篇（题目正对，且被 Gupta 列在 Related Work）——**但 R1 §5 明确声明未读其全文，现在不得引其结论**。 |
| **诚实性风险** | ⚠️ **有三处，R1 已自己列出并给了处置方案，必须照做**：<br>(a) **MechLens §7.7 的 factual-specificity control 与我们 tab_depth 的任务选择冲突**：Qwen2.5-7B 在 SST-2 上 late crystallization 只 **0.5%**（factual 84.9%，170× gap），结论 `Late Crystallization is specific to factual knowledge retrieval rather than a generic property`；而我们 tab_depth 的 native knee **就是 SST2/WiC/RTE 三个非事实任务的均值**（Qwen 0.824 = RTE 0.944 + SST2 0.639 + WiC 0.889）。→ **处置：MechLens 只用来背书 MMLU knowledge-logit-lens 那条（0.778–0.844L），绝不用它背书 SST2/WiC/RTE 的 native knee**；措辞用区间语言 "brackets"。R1 的段落已照此写。<br>(b) **Gupta §5 有一句可反向引用打我们**：早层 top-1 被高频 token 主导 `is not a consequence of probe bias but rather a reflection of the information content in the early layer representations`；§3.1 `the model is also unable to access the factual knowledge stored in its parameters`。→ **处置：承认它只说明 native 解码晚，不说明 trained 解码早**；「trained 解码早」的支撑换成我方 linear probe + `tab_h12_oracle` 的 continuous-prefix oracle + Ramnauth 的 lexical binding。<br>(c) **Ramnauth 全是 instruction-tuned 模型**（原文 `open-weight, instruction-tuned transformers`），而我们全论文 **chat_template=False / base 口径（项目铁律）** → 跨口径引用必须标注；且其 model identity 解释方差 **34.2% > objective identity 25.0%**，作者自述 `not a strong architecture-invariant account` → **我们也不得主张 architecture-invariant knee**，只主张 "linear precedes native within each family"。<br>另：**MechLens 自身数字内部不一致**（Mistral 26.8 vs 27.1；Llama 70.4 vs 71.0；下界 82.2 vs 82.3）→ **只引它稳定的三个数**：tuned-lens Δ=0.2pp、FEP Depth 一律 >80% depth、MMLU 98.2%。<br>**最后：三篇无一评长记忆任务** → `tab_depth` caption 的 `do not ... establish the same knees on long-memory tasks` 免责声明**必须保留，不能因这三篇就删**。 |

### 7.2 R2（readout 信息带宽梯度）

| 项 | 判定 |
|---|---|
| **可直接用** | ✅ **§5 的 237-word LaTeX 段落可直接贴**。四篇全文全抓到；venue 核实完毕且与 AUDIT0 一致：**`2503.04429` = ICML 2025（唯一 peer-reviewed，双向核实：S2 `publicationVenue.name="International Conference on Machine Learning"` + COMMENT `75 pages. Accepted to ICML 2025`）**；`2606.16897` / `2410.06981` = **arXiv preprint**；`2312.02730` = **UniReps@NeurIPS 2023 workshop extended abstract**（AUDIT0 亦判 preprint）。MAIN 对 CKA_Delta 的三处引用**逐句核实无误**。 |
| **需补什么** | (1) 段落是**自足的**（只用别人的数字 + 我方 `tab_replay_latency` 已发表的 99.19/96.07/3.12/CI[2.36,3.93]/1.403×），**无需新 GPU**。(2) 若 reviewer 追问「跨模型呢」，口头补充用 R2 §3.1 的 oracle 表（olmo2_1b→llama32_1b k=12: CE 6.390 / **ppl 596.1** vs A_full ppl 18.7，ΔCE +3.457；自拼自 ΔCE **+0.636**；llama32→qwen3_1.7b k=8 ΔCE +3.515 / 自拼自 +0.216 —— 我已复核 `paperD_research/smoke_out/oracle_olmo2_1b_llama32_1b_k12.json` 的 `scale` arm 数字逐项一致），**并明说是 preliminary / pilot（n_ce_texts=50，1B 模型对）**。(3) **可选低成本增量**（无新 GPU）：在同一 harness 上加一个中间带宽点（oracle affine 后只做 4-way MC readout），把梯度从两端补成三点单调曲线，直接对上 `2503.04429` 的 MMLU 数据点。 |
| **诚实性风险** | ⚠️ **三条，R2 §3.3 已列，必须写进措辞**：<br>(a) **不同 protocol 不能横比**（项目铁律）→ 表只作**定性排序**，措辞必须是 `these are not commensurable measurements; we use them only to order readouts by information demand`。<br>(b) **#9（Paper A j=12）与 #10（跨家族 oracle）不是同一个变量**：#9 是**同模型内深度复用**，#10 是**跨家族跨 tokenizer 拼接** → **#9 的 3.12 点小损失绝不能用来说「跨模型也行」**。R2 的段落已明确写 `we do not claim general cross-model reusability; we claim that at generation-level bandwidth, depth reuse survives`。<br>(c) **`2503.04429` 的 MMLU 0.004 低于 4-way chance 0.25**，是格式崩塌（作者自陈 trigger 率 4%→45%）→ 引用时必须带 caveat 并报 chance floor。R2 段落已写 `(below the $0.25$ chance floor)`。<br>另：**「可迁移性随 readout 带宽单调衰减」不得当 novelty claim 卖**，只当 reporting requirement；措辞必须列出搜过的 query 并写 `we do not claim priority`（R2 §4 已备好这句）。 |
| **顺带净收益** | **latency provenance 已闭环，无漂移**：`paperA/rebuttal_snippets/README.md` 记 #167 已解决——`bench_results/p0_13_quality_latency/latency/latency_proc{0,1,2}.json` 池化 3×20 raw reads 得 **931.9195 / 664.3577 ms, ratio 1.40274, p10/p90 931.51/941.94 & 663.71/667.10**，六项逐项对上 tex；.82 独占重跑 936.97/667.53（ratio 1.40365）→ **tex 不改，用 `latency_reproducibility.tex`**；`latency_provenance_own_drift.tex` **已作废**（它 own 一个不存在的漂移，文件头已加 ⛔ SUPERSEDED）。**rebuttal 里绝不要引用那个作废片段。** |

---

## 8. .bib 就绪清单

venue 一律按 **AUDIT0 最终判定**（AUDIT0 五路核实、49/49 全 HTTP 200、0 行 UNRESOLVED），
并吸收 SKEPTIC 的三处上调/下调。**preprint 一律显式标 `note={Preprint}`。**

### 8.1 新 Paper C 核心引用（必须进 .bib）

| key 建议 | arXiv | venue（最终） | 备注 |
|---|---|---|---|
| `llf2022` | 2202.00155 | **ICLR 2022** (peer-reviewed) | 三路一致：S2 `type=conference` + COMMENT `ICLR Camera Ready` + Journal-ref `ICLR 2022`。**必须在 Abstract/Intro 点名为 prior method** |
| `seal2023` | 2304.04858 | **CVPR 2023** (peer-reviewed) | S2 conference + DBLP `conf/cvpr/SarfiKCKRMB23` + DOI `10.1109/CVPR52729.2023.01935`。引它的负面结论（LLF 特征损害 transfer） |
| `activeforget2023` | 2307.01163 | **NeurIPS 2023** (peer-reviewed) | S2/DBLP `conf/nips/ChenMRAS0A23` + COMMENT。占掉「LLM 预训练周期 reset 买塑性」framing |
| `layerwisereinit2021` | 2109.00267 | **arXiv preprint** ⚠️ | S2 `venue='arXiv.org'`, DBLP corr-only。**社区常引 AAAI 2022，未核实到，不得写**。但被 LLF §4.1 点名对照 → 内容有 peer 背书。**它的 §5 takeaway 是我们门1 的杀手，必须引** |
| `springer2025overtrain` | 2503.19206 | **ICML 2025** (peer-reviewed) | S2 conference, year 2025, 68 cites。progressive sensitivity |
| `dash2024` | 2410.23495 | **NeurIPS 2024** (peer-reviewed) | S2 conference + COMMENT。**A7 生死线对手** |
| `acdc2021` | 2106.12379 | **NeurIPS 2021（论文自述）** ⚠️ | arXiv COMMENT `Accepted at NeurIPS 2021`；**S2 查询 429 未取到** → 写「NeurIPS 2021」并在 bib note 注明 S2 未核到。**SKEPTIC1 §4 的多轮+等尺寸 dense 终点占位** |
| `dsd2017` | 1607.04381 | **ICLR 2017（论文自述）** ⚠️ | S2 只收 `arXiv.org`；COMMENT `Published as a conference paper at ICLR 2017`。**引用写「ICLR 2017（论文自述，S2 未收录会议版）」** |
| `repr2019` | 1811.07275 | **CVPR 2019** (peer-reviewed) | S2 `Computer Vision and Pattern Recognition` + DOI `10.1109/CVPR.2019.01092`。⚠️ **Table 3 只能按原文读法引**（结构级 7.7 ≈ 权重级 7.8，增益来自 metric），**不得写成「结构级 > 权重级」** |
| `spdf2023` | 2303.10464 | **UAI 2023** (peer-reviewed) | S2 `Conference on Uncertainty in Artificial Intelligence` + COMMENT。1.3B GPT-3 XL |
| `raptr2024` | 2402.05913 | **ICLR 2025** ⚠️ **年份冲突，见 §8.4** | AUDIT0 主表判 **ICLR 2025**（S2 match + DBLP `conf/iclr/PanigrahiSLMRKK25` + OpenReview `ICLR.cc/2025/Conference` Poster，另有 ICLR 2024 `Rejected_Submission`）；SKEPTIC3 判 **ICLR 2024**（S2 `type=conference, year=2024`）。**采 AUDIT0 = ICLR 2025**（DBLP key 带 25 + OpenReview 2024 是 Rejected）。⚠️ 引其增益数字必须写 **`1-5%`**（原文），**不是 AUDIT3 的 1.5%** |
| `warmstart2020` | 1910.08475 | **NeurIPS**（S2 首条 271 cites, conference, 2019） ⚠️ | SKEPTIC3 上调（AUDIT3 曾按 preprint）。⚠️ **S2 有两条记录**：会议条目（新标题 "On Warm-Starting..."，271 cites，**无 arXiv id 字段**）与 `1910.08475`（旧标题 "On the **Difficulty** of Warm-Starting..."，`venue=arXiv.org`，24 cites）。**引用时必须避开 24-cite 那条**。⚠️ **具体年份/会议届次仍需 MAIN 手工核**（见 §8.4） |
| `primacybias2022` | 2205.07802 | **ICML 2022**（COMMENT 自述） | `reset the last 3 out of 7 layers ... with a periodicity of N steps` |
| `redo2023` | 2302.12902 | **ICML 2023 Oral**（COMMENT 自述） | |
| `plasticityinjection2023` | 2305.15555 | **NeurIPS 2023**（COMMENT 自述） | `without changing the number of trainable parameters` → **「终点等尺寸」这个约束已被命名** |
| `cbp_arxiv2023` | 2306.13812 | **arXiv preprint** | S2 `venue='arXiv.org'`, 53 cites |
| `cbp_nature2024` | （无 arXiv id） | **Nature 2024** | S2 search `venue='Nature', year=2024`。**与 2306.13812 是不同条目，引用时勿混** |
| `canscale2026` | 2606.24752 | **arXiv preprint** | S2 `venue=''`（空，HTTP200）。5M–314M。⚠️ **不得用它论证「整个领域空白」**，它自己 §II 就 cite 了 ICML 2025 与 Cho 2026 |
| `wdplasticity2026` | 2602.11137 | **ICML 2026** (peer-reviewed) | **AUDIT0 判定权威**：OpenReview `venueid=ICML.cc/2026/Conference` + `Submission26903/-/Camera_Ready_Revision` + `_bibtex` 为 `@inproceedings{...booktitle={Forty-third International…}}`。⚠️ **覆盖 AUDIT2/SKEPTIC1/SKEPTIC3 的「preprint / 疑 / 429 未核」** —— S2/DBLP 对 2026 会议有滞后 |
| `sam2026` | 2605.02105 | **ICML 2026** (peer-reviewed) | **AUDIT0 判定权威**：OpenReview `venueid=ICML.cc/2026/Conference`（`venue='ICML 2026 regular'`）+ `Submission19173/-/Camera_Ready_Revision` + COMMENT 自述一致。⚠️ **覆盖 AUDIT3/SKEPTIC3 的「S2=preprint」** —— 同上滞后 |
| `curseofdepth2025` | 2502.05795 | **NeurIPS 2025** (peer-reviewed) | S2+DBLP `conf/nips/SunSLYZL25`+OpenReview+COMMENT 四路一致 |
| `criticalperiods2018` | （ICLR 版无 arXiv id）/ 1711.08856 | **ICLR 2018**（会议条目，170 cites）/ **arXiv preprint**（1711.08856, 124 cites） | ⚠️ **两条是不同 S2 记录，勿混** |
| `criticallinear2023` | 2308.12221 | **ICLR 2023** | S2 conference, 14 cites。SKEPTIC3 未抓全文 → 只作背景引 |
| `shortgpt2025` | 2403.03853 | **ACL 2025 Findings** ⚠️ | DBLP booktitle `ACL (Findings)`。**勿写 ACL 2025 main** |
| `deeperlayers2025` | 2403.17887 | **ICLR 2025** | DBLP `conf/iclr/GromovTSGR25`；⚠️ **S2 `year=2024` 是 arXiv 年，正式年份 2025** |
| `solar2024` | 2312.15166 | **NAACL 2024 Industry Track** ⚠️ | DBLP booktitle `NAACL (Industry Track)`。**勿写 NAACL main** |
| `llamapro2024` | 2401.02415 | **ACL 2024 main** | DBLP `ACL (1)` + COMMENT |
| `cgls2026` | 2506.11389 | **ICML 2026** (peer-reviewed) | **AUDIT0**：OpenReview `ICML.cc/2026/Conference` + `Submission6733/-/Camera_Ready_Revision`（另有 ICLR 2026 Rejected）。⚠️ 覆盖 AUDIT1/AUDIT2 的「自述/preprint」 |
| `scale2026` | 2511.03270 | **ACL 2026 Findings** ⚠️ | DBLP `conf/acl/LeeCHCKYLJPPJ26` booktitle `ACL (Findings)`。⚠️ **AUDIT1 §6 曾标「存疑，需再核」→ AUDIT0 已裁定 Findings**；**勿写 ACL main**；正式年份 2026（S2 `year=2025` 是 arXiv 年） |
| `lesa2025` | 2502.13794 | **ACL 2025 main** | DBLP `ACL (1)` |
| `optdeus2025` | 2508.08011 | **arXiv preprint** | OpenReview 仅 `TMLR/Rejected` |
| `panguligh2025` | 2505.20155 | **arXiv preprint** | |
| `midas2024` | 2409.19044 | **NeurIPS 2024** | AUDIT1 §6 已核。**最需正面交锋的单调生长论文** |
| `gstack2024` | 2405.15319 | **NeurIPS 2024 Spotlight** | COMMENT 自述 |
| `reinitvsunits2025` | 2508.00212 | **arXiv preprint** | ⚠️ SKEPTIC3 补：OpenReview 有 **ICLR 2025 Withdrawn Submission** → **撤稿不算录用** |
| `resetitforgetit` | 2310.07996 | **ECAI 2024**（COMMENT 自述） vs **ECAI 2023**（SKEPTIC2 S2） ⚠️ | **年份冲突，见 §8.4** |
| `ke2021` | 2103.05152 | **CVPR 2021** (peer-reviewed) | S2 conference + COMMENT `CVPR Oral 2021`。⚠️ SKEPTIC2 **全文未抓到（两通路均 <20KB），仅摘要** |
| `rifle2020` | 2007.03349 | **ICML 2020** (peer-reviewed) | S2 conference。⚠️ SKEPTIC2 仅有标题 + venue，**未读全文** |
| `mobius2026` | 2607.17843 | **arXiv preprint**（自述 Preliminary Technical Report） | 标题含 "cyclic depth" → **必须引用并划界**（它是 block **order** 循环移位，不删层不重初始化） |
| `reassessing2026` | 2411.15558 | **ICLR 2026 Poster** (peer-reviewed) | **AUDIT0 §1.1 权威裁决**：OpenReview `venueid=ICLR.cc/2026/Conference` + `Camera_Ready_Revision`（Submission2804）；S2/DBLP 滞后。**MAIN 2026-08-05 的「preprint」结论作废**。§1.2 三重证据：它训的是**剩余模型的最后 1–3 个预训练层 + lm_head**，**不是随机初始化新层** → **不占 fresh-cap 构造** |

### 8.2 Paper A rebuttal 三篇（R1）

三条全部 **arXiv preprint**，R1 §4 已给出完整 bibtex（含 venue 核实注释），**可直接搬**：
`mechlens2026` = 2606.07978；`gupta2026depth` = 2510.18871；`ramnauth2026localized` = 2607.25663。
⚠️ `2606.07978` 的 repo 名含 `MechLens-EMNLP2026` 是**投稿意向不是录用**；
⚠️ `2510.18871` HTML 里的 "Machine Learning, ICML" 是 **ICML LaTeX keywords 行不是 venue**。

### 8.3 Paper A rebuttal 四篇（R2）

| key | arXiv | venue |
|---|---|---|
| `ckadelta2026` | 2606.16897 | **arXiv preprint**（S2 `venue=''`, `publicationVenue=null`） |
| `saeuniversality2024` | 2410.06981 | **arXiv preprint** |
| `klabunde2023` | 2312.02730 | **preprint / UniReps@NeurIPS 2023 workshop extended abstract**（COMMENT 自述；AUDIT0 判 preprint） |
| `activationtransfer2025` | 2503.04429 | **ICML 2025** (peer-reviewed) —— 四篇里唯一 |

### 8.4 UNRESOLVED / MAIN 需手工核的条目

AUDIT0 主表 **0 行 UNRESOLVED**，但**跨报告冲突 + 单报告未核**共 5 处，MAIN 必须手工定：

1. **`2402.05913` RaPTr 的年份**：AUDIT0 判 **ICLR 2025**（DBLP `conf/iclr/PanigrahiSLMRKK25` + OpenReview `ICLR.cc/2025/Conference` Poster + ICLR 2024 `Rejected_Submission`）；SKEPTIC3 判 **ICLR 2024**（S2 `year=2024`）。
   **MAIN 需核**：直接看 DBLP key 尾数与 OpenReview venueid。**我的判断：AUDIT0 对**（S2 的 year 常是 arXiv 年，AUDIT0 §4 已把这列为系统性坑）。
2. **`1910.08475` Ash & Adams 的确切 venue/年份**：SKEPTIC3 只拿到「S2 首条 271-cite conference 条目，year 2019」，且该条**无 arXiv id 字段**（可能是不同论文）。**MAIN 需核**：DBLP 作者检索 `Ash Adams warm-starting` 确认是否 NeurIPS 2020 且与 1910.08475 同文。**不核清前，bib 里写 `note={Conference version; venue verified via S2 title match only}`**。
3. **`2310.07996` Reset It and Forget It**：SKEPTIC1 记 **ECAI 2024**（COMMENT 自述），SKEPTIC2 记 **ECAI 2023**（S2 `European Conference on Artificial Intelligence`）。**MAIN 需核**：ECAI 是双年会（2023 Kraków / 2024 Santiago），查 DBLP `conf/ecai/`。
4. **`2306.13812` vs Nature 2024 的对应关系**：AUDIT2 怀疑 2306.13812 已有 Nature 2024 版但未核实；AUDIT3 从 S2 search 拿到独立的 `venue=Nature, year=2024` 条目。**MAIN 需核**：确认 Nature 2024 "Loss of plasticity in deep continual learning" 是否即 2306.13812 的期刊版。**不核清前，两条分开引，绝不把 2306.13812 当 Nature 引。**
5. **FIRE (ICLR 2026 Oral)**：SKEPTIC3 §3.C1 从 OpenReview 拿到 `venue="ICLR 2026 Oral"`，但**未取得 arXiv id、未抓全文，仅 OpenReview abstract**。**MAIN 需核**：拿 arXiv id 后抓全文。它的 background 句（`aggressive ones erase useful knowledge`）是门2 的抢先者，**在 related work 里必须引，但引之前要有 id**。

**另外三条「有 venue 但数字不可用」的警告（不是 UNRESOLVED，是使用禁令）**：
- **`2509.06518` Crown/Frame/Reverse**（preprint）：作者 COMMENT 自陈 `The reported results are skewed due to a data type mismatch ... every other token is zero` → **绝不引其任何 PPL/loss 数字**，只可引「层级异构分配」概念并标 preprint + 已知数据缺陷。
- **`1903.01611`**：已被作者撤并到 `1912.05671`（ICML 2020），COMMENT 明写 `Please read/cite that article instead` → **不要引 1903.01611**。
- **`1710.01878` GMP**：S2 十次重试全 429（不是「无 venue」），arXiv 无 COMMENT/JREF → **只能标 arXiv preprint**；社区常引 ICLR 2018 Workshop，**未核实，不得写入论文**。

### 8.5 Findings / Industry / Workshop 的写法禁令（AUDIT0 §4.3，抄进 bib 注释）

- ACL **Findings**（勿写 main）：`2403.03853` ShortGPT、`2511.03270` SCALE、`2406.11753`
- EMNLP **Findings**（勿写 main）：`2410.02330`、`2210.10041`、`2004.14975`
- NAACL **Industry Track**：`2312.15166` SOLAR
- 仅 **workshop**（本项目判 preprint）：`2312.02730`(UniReps@NeurIPS'23)、`2407.16286`(TF2M@ICML'24)、`2402.02834`(ME-FoMo@ICLR'24，且只是 v1 preliminary)、`2510.18871`(KnowFM@ACL'26)
- `2109.08406` 是 BlackboxNLP workshop 但有 DBLP `conf/` key → AUDIT0 计为 peer-reviewed；**若 MAIN 想统一「只算主会」，此行需降级，请显式决定**

---

## 9. 给 MAIN 的三条硬话（收尾）

1. **不要把 §3 的构造当「我们的方法」写。** 三个 skeptic 独立收敛：`--keep_front_layers` + `--n_fresh_layers`（保总深度）**逐字等于 LLF 的 mask**。只有 **R-mid（中间挖 + 原位插回）** 才是构造级差异，而**现有 flag 做不到**（`train_olmo2_shortgpt_fresh.py` 的 `--keep_layer_indices` 能任意保留但新层只能加尾部）→ **派 coder 前必须说清这是新功能，不是胶水。**

2. **文献先验是三重 negative，必须提前接受负结果**：
   (a) `2109.00267 §5`「大数据上 reinit 无收益」；(b) `DASH` Appendix C.1「stationary 数据下 Reset 不是解」；
   (c) 我方 Paper B 自己的知识税（keep14@200k 恢复分数 MMLU-L **19.4%**）。
   **如果 §6 的 gate FAIL-K1/K2，请立刻走 §7.1 的 measurement 定位或并入 Paper B，不要在 40 卡上追。**

3. **两段 Paper A rebuttal 现在就能用，且是本轮唯一「零风险正收益」的产出。** R1/R2 的 LaTeX 段落已 venue-clean、已自带 caveat、已避开所有反打词（"understanding"/"localize"/"causally establishes"）；latency provenance 亦已闭环无漂移（**用 `latency_reproducibility.tex`，`latency_provenance_own_drift.tex` 已作废**）。**建议先落 rebuttal，再决定 Paper C 的 gate 排期。**
