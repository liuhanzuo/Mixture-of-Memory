# Paper A + Paper B rebuttal 弹药总索引 (2026-08-06 sprint 产出)

若 reviewer 挑战某个点, 从下表定位对应答复材料.

## Paper B — Finding 1 (likelihood recovery overstates target recovery)

| Reviewer 可能挑战 | 答复材料 | 位置 |
|---|---|---|
| "你们主张 PPL heals / knowledge lags 已被 lockstep 数据证伪" | tex 从没做此硬断言, 只做 within-path 残差观察 | `paperB/audit_20260805/finding2_chance_correction.md` §Paper B tex 现有措辞审计 |
| "cross-arm dissociation 站不住" | 我们的 tex 明确 disavow "nor loss--task dissociation originate here" | `paperB/sections/02_related.tex:28` |
| "keep14 200k → 200k MMLU 差 28.74pp 只是量级问题" | Table 4 residual gap + paired bootstrap CI [1.08, 2.29] | `paperB/sections/04_experiments.tex:19-24` |

## Paper B — Finding 2 (interface gains require null floor)

| Reviewer 可能挑战 | 答复材料 | 位置 |
|---|---|---|
| "raw C-L sign 是 scoring metric artifact 不是能力" | Random-16L content_norm=0.3598 是 uninformative chance, 我们 own | `paperB/audit_20260805/finding2_chance_correction.md` |
| "崩坏 arm 的 C>L 只是 chance floor 差" | 单变量 letter above-chance headroom 表 + Wilson CI | `paperB/rebuttal_snippets/tab_letter_headroom.tex` (drop-in) |
| "PPL 匹配的 arm 差异只是噪声" | Random-16L PPL 11.50 letter -0.30pp p=0.80 vs keep14@67.5k 11.53 letter -0.08pp p=0.59 都 chance-level | `paperB/audit_20260805/finding2_letter_headroom.tsv` |
| "keep8 letter 有 +0.5pp 是能力" | 单侧 binomial p=0.085 NS | 同上 |

## Paper B — 数字可信度

| Reviewer 挑战 | 答复 | 位置 |
|---|---|---|
| "tex 里 X 数字 vs 磁盘不一致" | 4 MMLU + 12 closed-book QA 全部一致 max diff 0.001 (16/16 ✓) | `paperB/audit_20260805/tex_numbers_vs_disk.md` |
| 具体数字溯源 | 每-item score in `paperB/anonymous_artifact/scores/closedbook/` | 已在 release |

## Paper A — Table 3 latency (`tab_replay_latency`, `tab_pareto`, `tab_core_tradeoff`)

| Reviewer 挑战 | 答复 | 位置 |
|---|---|---|
| "931.9/664.4 ms 复现不出" | own <2% 漂 rebuttal 段 (option a) | `paperA/rebuttal_snippets/latency_provenance_own_drift.tex` |
| "1.403× speedup 有多鲁棒" | 方向 1.37-1.41× 在 P0.12 + P0.12 acceptance + P1.8 4 cell × 4 G 共 20+ 独立配置全部成立 | `paperA/audit_20260806/latency_provenance_audit.md` |
| "99.20 vs 99.19 内部不一致" | 已修 tab_pareto 99.20→99.19 (磁盘真值 99.187) | commit `9883ef9` |

## Paper A — Table 2/3 quality (`tab_replay_latency`, `tab_pareto` RULER macros)

| Reviewer 挑战 | 答复 | 位置 |
|---|---|---|
| "99.19/96.07 RULER 从哪来" | .82 上 `p0_13_quality_latency/summary.json` 有 `macro/armA=99.187, armB=96.067, diff=3.12`; 已镜像到 wzc1 | `paperA/anonymous_artifact/scores/p0_13_quality_latency/summary.json` |
| "3.12pt gap CI 从哪来" | `stats.json` 有 `paired_bootstrap_95ci=[2.36, 3.9333]` + `mcnemar` 字段 + `all_packs_paired_1to1=True` | 同上 `stats.json` |
| "paired 声明可信度" | `all_packs_paired_1to1=True` 磁盘字段直接印证 | 同上 |

## Paper A — Table 5 write-control overlap (`tab_write_context`)

| Reviewer 挑战 | 答复 | 位置 |
|---|---|---|
| "92.5 → 98.5 (w=32) 是否显著" | 已 pre-registered target: `baseline_w0_pooled=92.5, best_width_pooled=99.0`; pairwise CI [3.0, 9.5]; McNemar p=5e-4 | `paperA/anonymous_artifact/scores/p0_17_e2_overlap/stats.json` |
| "有没有多 w cherry-pick" | w=32/64/128 全部 report (98.5/98.5/99.0), 无 hide | `summary.json` `macro/*` |

## Paper A — abstract BM25 delta

| Reviewer 挑战 | 答复 | 位置 |
|---|---|---|
| "11.56 BM25 lead 是哪个 anchor" | `primary_anchor_diff=-11.56` (符号: raw replay 领先), CI [-14.444, -8.667], n=9000 paired | `paperA/analysis/equal_latency/source/bm25/decision.json` |

## 涉及的 audit commits (sprint 汇总, 时序倒排)

- `813dab9` paperA rebuttal snippet: own-drift 段
- `5f979ce` paperB rebuttal snippet: letter headroom 表 + paragraph
- `82da816` UPDATELOG: sprint 总结
- `a82f7b7` SESSION_HANDOFF: 覆盖 2026-08-06 快照
- `f9fb8c6` mirror P0.13/P0.17 artifact wzc1
- `9883ef9` audit paperA 3/3 primitive precise + tab_pareto fix
- `8c30fc7` audit paperA self-consistency
- `550a81a` audit paperA latency provenance drift
- `dfdbf2d` audit paperB tex-wording
- `638fb04` audit paperB numbers vs disk 16/16
- `51c7349` audit paperB Finding 2 chance correction
- `6a3b6bb` audit paperB letter headroom Wilson CI + binomial
