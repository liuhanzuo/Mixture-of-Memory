# Finding 2 chance-corrected re-analysis (2026-08-06)

## 问题
原 Finding 2 claim "崩坏 arm content > letter, 健康 arm content < letter, 符号追随能力"
是基于 raw content_norm_acc − letter_acc。但 content_norm 的 empirical chance
未必等于 letter 的 0.25。

## 校正方法
用 Random-16L (fully random-init, 200k step, PPL=11.499) 作为
"uninformative model" 代理:
  - letter_acc     = 0.2470  ≈ 理论 chance 0.25 ✓
  - content_norm   = 0.3598  ⇒ empirical chance for content = 35.98%
差异 +10.98pp 是 "按选项 log-lik 挑最大, 长度归一化" 的先验产物,
无关任何能力.

## chance-adjusted delta = (C − 0.3598) − (L − 0.25)
                        = (C − L) − 0.1098

10-arm 全表:

  arm             PPL      C-L raw    chance-adj
  base            7.40    -13.48pp    -24.46pp
  full32@25k      7.67    -12.14pp    -23.12pp
  ShortGPT-16     9.78     -7.31pp    -18.28pp
  keep14@200k    10.56     +6.48pp     -4.50pp
  Random-16L     11.50    +11.28pp     +0.30pp
  keep14@67500   11.53    +11.66pp     +0.68pp
  keep12@124k    11.57     +9.02pp     -1.96pp
  frozen@200k    12.80     +9.81pp     -1.17pp
  keep10@83.5k   12.82     +7.24pp     -3.74pp
  keep8@121k     13.33     +8.72pp     -2.25pp

## 结论
- 10 arm 中 8 个 chance-adj < 0; 只有 Random-16L / keep14@67500 略正 (~+0.5pp,
  分明在 chance 噪声内).
- 原 "崩坏 arm C > L" 是 SCORING METRIC 的 base-rate artifact, 不是能力差异.
- Rebuttal 时不能用 raw C−L 说事; 必须改口径:
    "letter above-chance headroom" (L − 0.25) 随 PPL 单调塌:
      base +35.5pp → keep14@200k +6.8pp → keep8 +0.5pp
    这是 letter 单轴上就成立的能力信号, 不依赖两个 metric 的比较.

## 影响
- Paper B 若已用 raw C−L 论证 "崩坏 arm 依赖 content 信号 / 无 letter 能力":
  这个结论技术上是对的 (letter dropped to chance), 但语气不能说 "content > letter"
  暗示 content 有信号 --- content 也在 chance (35.98%) 附近, 只是 chance 高.
- Cleaner claim: "letter accuracy collapses to chance while content_norm stays
  at content's inflated empirical chance (both = uninformative behavior)".
- 这是弱化, 不是推翻, 但是 rebuttal 必须主动 own 这个校正.

## Letter above-chance headroom (rebuttal-ready, 2026-08-06 01:12)

单变量 clean 口径, 与 content_norm 完全解耦; 用 letter chance = 25% 的 Wilson 95% CI
+ one-sided binomial 检验:

| arm          | PPL     | letter    | headroom | 95% CI (pp)      | p vs chance | sig |
|--------------|---------|-----------|----------|------------------|-------------|-----|
| base         |  7.398  | 60.54%    | +35.54pp | [+34.73, +36.35] | < 1e-100    | *** |
| full32@25k   |  7.670  | 58.77%    | +33.77pp | [+32.95, +34.58] | < 1e-100    | *** |
| ShortGPT-16  |  9.780  | 47.42%    | +22.42pp | [+21.60, +23.25] | < 1e-100    | *** |
| keep14@200k  | 10.561  | 31.84%    |  +6.84pp | [+6.07,   +7.62] | 1.7e-78     | *** |
| **Random-16L**   | 11.498  | 24.70%    |  -0.30pp | [-1.01,  +0.42]  | 0.80        | **ns** |
| **keep14@67500** | 11.533  | 24.92%    |  -0.08pp | [-0.79,  +0.64]  | 0.59        | **ns** |
| keep12@124k  | 11.566  | 27.28%    |  +2.28pp | [+1.54,  +3.02]  | 2.4e-10     | *** |
| frozen@200k  | 12.797  | 26.24%    |  +1.24pp | [+0.51,  +1.97]  | 3.6e-04     | *** |
| keep10@83.5k | 12.816  | 27.20%    |  +2.20pp | [+1.47,  +2.95]  | 8.1e-10     | *** |
| keep8@121k   | 13.333  | 25.50%    |  +0.50pp | [-0.21,  +1.23]  | 0.085       | **ns** |

★ PPL 匹配 pair (Random-16L PPL=11.50 vs keep14@67500 PPL=11.53) letter headroom
  分别 -0.30pp (p=0.80) 与 -0.08pp (p=0.59) 都在 chance 内 → letter-space lockstep null.
★ keep8@121k (+0.50pp, p=0.085) 不达 α=0.05 → 最深剪层的 letter 能力**已回到 chance**.
★ 单调性: headroom 随 PPL 上升逐步塌 (35.5 → 33.8 → 22.4 → 6.8 → chance).
★ 与 content_norm 无关, 直接从 letter_acc 得, rebuttal 站得住.

TSV 数据: paperB/audit_20260805/finding2_letter_headroom.tsv

## Paper B tex 现有措辞审计 (2026-08-06 01:59, rebuttal prep 关键修正)

上一轮说的"三重证伪 Paper B 主命题"打的是**口头概括的靶子**, 不是 tex 里的正式文本.
逐节 review 现有 paperB/sections/*.tex 措辞:

### 00_abstract.tex
- 主命题: "**multi-interface recovery audit**" 作为 methodology contribution
- "Perplexity and downstream performance **can diverge** after compression"
  ---把 dissociation 当**已知前提**引用 costcompression/jaiswal2024truth/beyondperplexity,
  不是**主张** dissociation
- 主结论是 "six-axis reporting checklist" ---audit protocol 而非因果理论

### 01_introduction.tex
- "The open measurement problem is therefore not whether the metrics can disagree,
   but how to audit a recovery claim"
- Contributions 都是 measurement / diagnostic / actionable finding + audit protocol
- 没有 "perplexity heals while knowledge lags" 类硬断言

### 04_experiments.tex
- **Finding 1**: "likelihood recovery overstates target recovery"
  - 具体证据: keep14 200k PPL 10.561 但 MMLU 仍距 base 差 28.74pp
  - 这是**单轨迹内**绝对差距观察, 不涉及跨 arm lockstep
  - 我上轮三重证伪 (Spearman +0.94 lockstep, matched-PPL +0.32pp, ShortGPT beats keep14)
    针对的是 **跨 arm dissociation**, **与 Finding 1 tex 措辞正交**
- **Finding 2**: 已明确 own content 有 high random floor (.360) + null needed
  - 与我 letter-only 校正表 (Random-16L content_norm=.3598) 完全一致
  - tex 已经含 "This reversal changes the interpretation of the interface gain"
- **Finding 3**: 用 letter-only 差 (15.6pt vs 1.8pt) 讨论 ShortGPT vs keep14, 干净

### 02_related.tex line 28
- **"nor loss--task dissociation originate here"** ---Paper B 主动 disavow "自己是提出者"
- Reviewer 若挑 dissociation, tex 已经 defensively 回答了

## Rebuttal 实操 impact

不需要软化 Finding 1 tex 措辞. 我上一轮 audit 出的所有 evidence 可作 supporting
appendix material, 但**主 rebuttal 论证不需要重写 Finding 1**.

若 reviewer 挑:
- "你们说 perplexity heals but knowledge lags 是不是被证伪?"
  → tex 从没这样说. Finding 1 说的是 "likelihood recovery **overstates** target
    recovery", 不是 dissociation. 200k keep14 仍距 base MMLU 28.74pp 是硬事实
    (paired 14042 items, boostrap CI [1.08, 2.29] 只 close 1.68pp), 这个观察站得住.
- "Random-16L 和 keep14@matched-PPL 是不是 letter chance-level? 那你们的差距会不会
   只是 chance 噪声?"
  → 见 finding2_letter_headroom.tsv. 是的, PPL 匹配 pair 都在 chance 内 (p=0.80/0.59),
    这正是 Paper B Finding 2 想说的 "null needed" 现象, 支持 tex 现有论证.
- "Cross-arm PPL vs MMLU 是不是 lockstep? 那 Finding 1 的 headline claim 是不是
   scale artifact?"
  → tex Finding 1 从没做 cross-arm claim. 只做 within-path 观察.

## 结论
Paper B tex 已经足够 defensive. 我上轮的三重证伪 audit 主要用于:
1. Preventing overclaim (自身写 rebuttal 时不要 overclaim)
2. Supporting appendix (若 reviewer 挑, 我们有更强证据)
3. Paper D/C 论证时不要基于错误 Paper B 结论推导

Task #164 "Paper B 主命题证伪三连" 描述改为 "Paper B 主命题保护:
tex 措辞已 defensive, 三重 audit 只是 supporting evidence, 不需要 rewrite".
