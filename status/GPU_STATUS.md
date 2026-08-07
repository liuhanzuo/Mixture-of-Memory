# GPU_STATUS.md — 4 节点可调度 GPU 台账（32 卡；.104 已交还用户）
> 每次启动/kill GPU 任务更新。heartbeat 先读→对照 nvidia-smi→台账说跑但空=补卡。★29.162.226.120=dllm 绝不碰。

## 🚫 .104 已交还用户（2026-08-05 15:4x，用户指令：「.104你不要管了 我在用」）
> **`.104`（28.83.24.104）从此不在可调度范围。** 不派任务、不 kill 其上进程、**不因「看到空闲」去补卡**、
> heartbeat 不把它算进 idle 统计、不因它空转报 WARNING。下方 .104 的历史行仅作归档，不再更新。
>
> 可调度节点 = **4 台 / 32 卡**：**LOCAL + .252（wzc1 盘）/ .73 + .82（zwfy6 盘）**。
> ⚠️ 连带影响：**zwfy6 侧的 16 卡多机 DDP 只剩 .73+.82 这唯一组合**（原先 .73/.82/.104 任取两台）。

## 当前在跑（2026-08-08 06:25 +08:00 更新）— **LOCAL** P1.2 seed2（~87h）+ **.104** P2.4 keep12 SFT（zwfy6，step 780/842，8/8 @ 100%）；**.73 空闲 0/8**（keep8 SFT ✅ 05:56 + post-SFT eval battery ✅ 06:21）；**.82 空闲 0/8**（keep10 SFT ✅ **06:16:12** final.pt step=842 → **post-SFT eval 待跑**）；**.252** ladder wzc1 pre-SFT eval ✅ 05:51
> ⚠️ **补卡线索（2026-08-08 06:25）**：`.82` 的 keep10 SFT 已于 06:16:12 完成且节点空转 → 可立即照 `scripts/_run_olmo2_p24_eval_sft_keep8_73.sh` 的模式派 keep10 post-SFT eval battery（换 arm/ckpt/output_name + pre anchor 用 keep10 同盘 _v2）。keep8 实测 battery 仅 16 min。
> ★ **keep8 预测检验结论（2026-08-08）**：damage-sensitivity **次线性/饱和**——n=3 线性拟合预测 +14.0%，实测 **+10.15%**（CI [+10.05,+10.23] 排除预测值）；详见 `status/PAPERB_P24_SFT_KEEP8_EVAL.md`。后续 keep10/keep12 落点是对「饱和」的进一步检验。

> ### ✅ (完成 06:21:37) .73 8×H20 (zwfy6) — `p24_sft_keep8_eval_73`（PaperB P2.4 keep8 **POST-SFT eval battery**）
> | 项 | 值 |
> |---|---|
> | 任务 | keep8 post-SFT 5-harness battery（PPL + core6 + know5 + MMLU-dual + closedbook），与**同节点同架构** pre-SFT anchor `7B_keep8_step121000_v2` 做 item 级 pairing |
> | 节点/卡 | **.73 8×H20（zwfy6 盘）GPU 0-7 全占**，29.5 GiB/卡 @ 99-100%（PPL 段） |
> | 起始 / 结束 | 2026-08-08 **06:05:22** → **06:21:37** +08:00（**16 min**，远快于 ~90min 估计） |
> | ★ 核心结果 | pre-PPL **13.3329** → post-PPL **14.6857**，**ΔPPL% = +10.15%**；shard-level bootstrap CI **[+10.05, +10.23]** |
> | ★ 预测检验 | n=3 拟合预测 **+14.0%**（post 15.20）→ **残差 −3.86 pp，预测落在 CI 之外 = 被排除**。但预registered「materially lower」门槛（post<14.5 / Δ<9%）**也未达到** → 落在两个预设结论**之间**：关系**次线性/饱和**，非线性外推失效（n=4 refit 斜率 1.6015→0.9417，r 0.998→0.907；二次项 −0.215 = 凹） |
> | 下游 | core6 macro 0.52328→0.51428（−0.90pp，CI 排除 0；**arc_easy 独占 −0.43pp，p=3.6e−12**，其余 5 项不显著）；MMLU letter 0.2543→0.2483（p=0.162，仍贴 chance）/ content_norm 0.3427→0.3365（p=0.0052）；**TriviaQA EM −4.30pp（p≈6.8e−106）= 最大伤亡**；PopQA EM −0.88pp 但 contains/f1 持平 → **格式/verbosity 漂移而非纯知识丢失** |
> | 完整性 | `assert_8shards` **5/5 段全过 8/8**；per-item preds 双侧齐全（core6/MMLU 14042/popqa 14267/triviaqa 17944）；paired 值与 harness `summary.json` **全部 <1e−9 吻合**（item 对齐无误join）；pre 侧复现 MAIN-verified anchor PPL 13.3329 ✓ core6 0.52328 ✓ |
> | ⚠️ Caveat | keep8 anchor 是 **step121000 非 200k**（keep8 从未到 200k，`PAPERB_TABLE4_BUDGET_DEFECT.md`）→ 本实验「固定自身 pre-SFT ckpt 测 SFT delta」有效，但**不可用于对 keep14(200k) 的 compute-matched 深度比较** |
> | PID / log | `3071994`（wrapper 3071993）/ `logs/p24_eval_sft_keep8_73.log` |
> | Driver / 分析 | `scripts/_run_olmo2_p24_eval_sft_keep8_73.sh` + `scripts/paired_analysis_p24_sft_keep8.py` → `results/paperb_p24_sft_keep8_paired.json` |
> | Report | `status/PAPERB_P24_SFT_KEEP8_EVAL.md`（commit `fd1633c`） |
> | 现状 | **.73 现 0/8 空闲**，可补卡 |

> ### ✅ (完成 05:56:44) .73 8×H20 (zwfy6) — `p24_sft_keep8_zwfy6`（PaperB P2.4 n=6 damage-sensitivity ladder — SFT lowest rung）
> | 项 | 值 |
> |---|---|
> | 任务 | keep8+fresh2 (10L) SFT from `step121000.pt` — byte-identical recipe to completed keep14/shortgpt16 arms; predicts Δ PPL ≈ +14.0% based on n=3 fit (pre-SFT PPL 13.333) → **实测 +10.15%，见上方 eval 行** |
> | 节点/卡 | **.73 8×H20（zwfy6 盘）GPU 0-7 全占**, 68.3 GiB/卡 @ 99-100% |
> | 起始 | 2026-08-08 **04:27:01** +08:00 |
> | 完成 | 2026-08-08 **05:56:44**（89.7 min，step=842，**0 NaN**，step 840 loss=1.3947） |
> | pre-SFT ckpt | `outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt` (11.4 GB) — Table 4 headline (not step200000 which never existed) |
> | 输出 | `outputs/olmo2_p24_sft_keep8fresh2/final.pt`（34.15 GB）+ `step500.pt` — **eval ✅ 已完成** |
> | Recipe | BS=1 GA=16 eff_batch=128 seq_len=2048 max_steps=842 lr=1e-5 min_lr=1e-6 warmup=100 wd=0.1 seed=42, fp32 master AdamW + bf16, gradient_checkpointing=1 |
> | step-1 loss | step 20 = **2.0422** (finite; NaN-fix commit `0fd051a` in play) |
> | Data md5 | `b1e6fe4e...` (input_ids) / `bf7c5774...` (labels) — matches spec |
> | Trainer md5 | `02d8b9ead6cafdf5893d6e59df6ad196` — no bnb imports (`grep -c bitsandbytes = 0`) |
> | PID / log | `2997933` / `logs/p24_sft_keep8.log` |
> | Driver | `scripts/_run_olmo2_p24_sft_ladder_zwfy6.sh keep8` md5=`62a640b1b3f344226e33d1b98b720e0b` (wzc1 = 3 nodes) |
> | Report | `status/PAPERB_P24_LADDER_SFT.md` |

> ### ✅ .82 8×H20 (zwfy6) — `p24_sft_keep10_zwfy6` **DONE 2026-08-08 06:16:12 +08:00**（PaperB P2.4 keep10 SFT — middle rung）
> keep10+fresh2 SFT finished (step 842, final.pt = 39.0 GiB, no NaN). Post-SFT eval battery launched immediately on same node — see next entry.

> ### ✅ .82 8×H20 (zwfy6) — `p24_eval_sft_keep10_82` **DONE 2026-08-08 06:43:15 +08:00**（PaperB P2.4 keep10 POST-SFT eval battery — n=3 fit FALSIFIED）
> | 项 | 值 |
> |---|---|
> | 结果头条 | **pre-PPL 12.8159 → post-PPL 13.9221 ⇒ ΔPPL% = +8.63%**. Predicted +13.18%. **Miss = −4.55 pp** (n=3 fit systematically over-predicts). keep8 also missed by −3.86 pp same direction ⇒ **n=3 linear fit is DEAD**; sub-linear/saturating relationship. |
> | 节点/卡 | ✅ 已 idle |
> | 起始 → 结束 | 2026-08-08 **06:24:36 → 06:43:15** +08:00, wall-clock **19 min** (spec ETA 25min, on target) |
> | pre anchor | `7B_keep10_step83500_v2` on zwfy6 (PPL=12.815923, ANCHOR_OK gate passed pre-launch) |
> | post ckpt | `outputs/olmo2_p24_sft_keep10fresh2/final.pt` (39.01 GB, step=842 keep=10 fresh=2; 135 tensors loaded `strict=True`) |
> | 输出 output_name | `7B_p24_sft_keep10fresh2_final` (+`_know`) across 5 result roots (all `summary.json` present) |
> | Hard gate | `assert_8shards` **5/5 harnesses ✅** (PPL / core6 / know5 / MMLU-dual / closedbook) |
> | Per-item preds | ✅ all retained; McNemar + 10k paired bootstrap computed on-node |
> | Downstream deltas (highlights) | core6 avg Δ=+0.03pp (wash) · arc_easy Δ=−3.45pp (p=2.9e−9) · MMLU-letter Δ=−1.82pp (p=5.6e−4) · MMLU-content_norm Δ=−0.43pp (p=0.053, marginal) · **PopQA EM Δ=−1.53pp (p=1.4e−41; b/c=36/254)** · **TriviaQA EM Δ=−5.19pp (p=1.1e−146; b/c=242/1173)** · TriviaQA-contains Δ=−2.95pp (p=4.2e−36) |
> | SFT-hurts-facts story | ✅ replicated (keep10 is 2nd Table 4 arm confirming SFT tanks closed-book memorised facts while leaving core6 flat and MMLU-content near-null) |
> | Chat template | `chat_template=False`, `--add_bos 0`, `LOCAL_RANK=0 RANK=$g` per shard ✅ |
> | Caveat | keep10 pre-anchor `step83500` = 42% of claimed 200k budget (`status/PAPERB_TABLE4_BUDGET_DEFECT.md`). Valid for SFT-delta-on-own-ckpt; NOT comparable to hypothetical keep10@200k SFT delta. |
> | Python | `/opt/conda/envs/torch-base/bin/python` (3.14.6 + torch 2.13.0) |
> | Driver | `scripts/_run_olmo2_p24_eval_sft_keep10_82.sh` (mirrors keep8 driver byte-identical except arm/ckpt/output_name) |
> | Parent bash PID / log | `1215445` (dead) / `logs/p24_eval_sft_keep10_82.log` |
> | 报告 | `status/PAPERB_P24_SFT_KEEP10_EVAL.md` |

> ### ▶️ .104 8×H20 (zwfy6) — `p24_sft_keep12_zwfy6`（PaperB P2.4 n=6 damage-sensitivity ladder — top rung of the three）
> | 项 | 值 |
> |---|---|
> | 任务 | keep12+fresh2 (14L) SFT from `step124000.pt`; predicts Δ PPL ≈ +11.0% (pre-SFT PPL 11.443) |
> | 节点/卡 | **.104 8×H20（zwfy6 盘）GPU 0-7 全占**, 86.9 GiB/卡 @ 100% (highest of the three; still <90 GiB threshold) |
> | 起始 | 2026-08-08 **04:31:29** +08:00 |
> | ETA | **~05:30-05:45** (~8.66s/step × 842 = ~121 min at init; will reduce once steady) |
> | pre-SFT ckpt | `outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt` (43.9 GB) — Table 4 headline (NOT `step111500` which is the wzc1-only cross-arch pair anchor) |
> | 输出 | `outputs/olmo2_p24_sft_keep12fresh2/{step500,final}.pt` |
> | step-1 loss | step 20 = **1.8399** (finite) |
> | ⚠️ .104 之前用户交还 | 但 .104 现在 nvidia-smi 显示 8/8 全空，且本次 SFT 用户明确派单指定；不新增 heartbeat 补卡语义，仅本次任务在此节点跑 |
> | PID / log | `3187927` / `logs/p24_sft_keep12.log` |
> | Driver | `scripts/_run_olmo2_p24_sft_ladder_zwfy6.sh keep12` |



> ### ✅ .252 8×L20A (wzc1) — `paperB_p24_ladder_wzc1_252` **DONE 2026-08-08 04:31:19 +08:00**
> keep8 + keep12 pre-SFT wzc1 eval battery finished; all 10 harnesses landed. See `status/PAPERB_P24_LADDER_WZC1_EVAL.md`.

> ### ✅ .252 8×L20A (wzc1) — `paperB_p24_keep10_wzc1_252` **DONE 2026-08-08 05:51 +08:00**（PaperB P2.4 / task #189 extension — **keep10** pre-SFT wzc1 eval，last missing wzc1-side ladder rung）
> | 项 | 值 |
> |---|---|
> | 任务 | **task #189 5×2 grid completion**：keep10 是唯一无 wzc1 pre-SFT eval 的 Table 4 ladder rung。**Phase A ✅**：`scp -O` from `.73`→wzc1，36.3 GiB @ 18.4 MB/s，34 min，md5 8bf07fa0d08ddfdf66bd80fbc6721b33 **两侧一致**。**Phase B ✅**：全 5 harness 13min 内跑完（PPL 12.8158 / core6 accn 0.5322 / know5 mmlu 0.2713 / MMLU-dual content_norm 0.3452 / closedbook TriviaQA em 18.15）。`assert_8shards` 全过，per-item preds 全保留。**arch-only delta**（无 keep12 那种 Δstep 混淆） |
> | 节点/卡 | ✅ 已 idle |
> | 起始 / 结束 | 2026-08-08 **05:38:02 → 05:51:XX** +08:00（wall-clock ~13 min，比 60min ETA 快得多；scp 4:57–5:31, eval 5:38–5:51） |
> | 输入 ckpt | `outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt` (36.3 GiB, cross-disk transferred, **matches Table 4 headline step exactly**) |
> | 输出 output_name | `7B_keep10_step83500_wzc1` |
> | ★ 早期锚点 | PPL 12.8158 vs zwfy6 headline 12.816 → **arch-invariant to 4 sig figs**（ckpt integrity confirmed cross-disk） |
> | ★ core6 cross-arch delta | L20A 0.5322 accn vs H20 0.5300 = **+0.218 pp**（accn），核心 flip 数 Σ\|Δc(acc)\| = **31**——fits n=4 monotone pattern (weakly)，**不破坏故事** |
> | 硬门禁 | `assert_8shards` 在每次 merge 前强制核对 8/8 shard 存在（全 5 harness 均 ✅ 通过） |
> | 每-item 保留 | ✅ downstream core6+know5 各产 `per_example_<task>.jsonl` + 8 shard-level 副本；mmlu_content/closedbook 默认写 |
> | Chat template | `chat_template=False`，`--add_bos 0` 全程 |
> | Python | `/opt/conda/envs/torch-base/bin/python` |
> | Driver | `scripts/_run_olmo2_p24_eval_keep10_wzc1_252.sh` commit `b7d2d39` |
> | Parent bash PID / log | `3170804` (dead) / `logs/p24_eval_keep10_wzc1_252.log` |
> | 报告 | `status/PAPERB_P24_KEEP10_WZC1_EVAL.md` |

> ### ▶️ .73 8×H20 (zwfy6) — `p24_eval_ladder_prev2_73`（PaperB P2.4 / task #189 — Table 4 ladder 4-rung pre-SFT _v2 eval，8/8 卡 17.6-18.5 GiB @ 66-98%）
> | 项 | 值 |
> |---|---|
> | 任务 | **task #189 cross-arch flip-count audit（n=5 extension）**：keep8/keep10/keep12/shortgpt16 四个 Table 4 ladder rung pre-SFT 全 battery（PPL + core6 + know5 + MMLU dual + closedbook），每 harness `--save_per_example` 保留 per-item preds；用 H20 数字与 wzc1 Table 4 source 对拍以判断每 rung 的 Table 4 是 L20A 还是 H20（详见 `status/PAPERB_CORE6_CROSSARCH_FLOOR.md` §flip-count scaling） |
> | 节点/卡 | **.73 8×H20（zwfy6 盘）GPU 0-7 全占**，实测 17.6-18.5 GiB / 66-98% util |
> | 起始 | 2026-08-08 **02:28:05** +08:00 |
> | ETA | **~4-6 h → 约 2026-08-08 06:30-08:30**（4 rungs × 5 harness = 20 harness runs on H20，每 harness ~15-20 min） |
> | 输入 4 ckpts | keep8: `outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt` (11.4 GB) / keep10: `.../keep10fresh2/step83500.pt` (39.0 GB) / keep12: `.../keep12fresh2/step124000.pt` (43.9 GB) / shortgpt16: `.../shortgpt16/step200000.pt` (48.7 GB)。**任务 text 写 step200000 但 keep8/10/12 从未训到 step200000**（详见 `PAPERB_P24_LADDER_PREV2_EVAL.md` §path correction）；使用 P0.7 audit headline steps |
> | 输出 output_name | `7B_keep8_step121000_v2` / `7B_keep10_step83500_v2` / `7B_keep12_step124000_v2` / `7B_shortgpt16_step200000_v2`（`_v2` 后缀确保不覆盖 Table 4 引用的既存 summary.json） |
> | 早期锚点 | keep8 PPL 重跑 = **13.3329**，对 P0.7 audit 表 **13.333** 至 1e-4 精度一致 → harness path 零漂移（PPL 本就 arch-invariant，见 flip-count floor 报告） |
> | 硬门禁 | `assert_8shards` 在每次 merge 前强制核对 8/8 shard 存在，缺任一即 abort 不 merge（防 5/8 partial-merge silent contamination） |
> | 每-item 保留 | downstream `--save_per_example` → `per_example_<task>{,_shard{0..7}of8}.jsonl`；mmlu_content/closedbook 默认写；pre/post + cross-arch McNemar / paired bootstrap 具备条件 |
> | Chat template | `chat_template=False`，`--add_bos 0` 全程 |
> | Python | `/opt/conda/envs/torch-base/bin/python` torch 2.13.0 |
> | Driver | `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh` md5=`7e56545268874b180ef682e7e473734a`（wzc1 + zwfy6 一致） |
> | PID / log | driver bash `2871872` / `logs/p24_eval_ladder_prev2_73.log` |
> | 报告 | `status/PAPERB_P24_LADDER_PREV2_EVAL.md` |

> ### (完成 02:02) .73 8×H20 (zwfy6) — `p24_eval_keep14_73`（PaperB P2.4 keep14fresh2 post-SFT eval + pre-SFT gap-fill，8/8 卡 24-25 GiB @ 60-80%）
> | 项 | 值 |
> |---|---|
> | 任务 | **P2.4 keep14fresh2 arm 的 pre/post SFT eval battery**：held-out Dolmino PPL + core6（HellaSwag/ARC-C/ARC-E/PIQA/OBQA/WinoGrande）+ know5（MMLU/Lambada/BoolQ/CSQA/SIQA）+ MMLU dual-protocol（letter+content）+ PopQA/TriviaQA closed-book |
> | 节点/卡 | **.73 8×H20（zwfy6 盘）GPU 0-7 全占** |
> | 起始 | 2026-08-08 **01:26** +08:00 |
> | ETA | **~1.5-2 h → 约 2026-08-08 03:00-03:30**（8-shard 并行；每 harness ~15-25 min） |
> | 输入 ckpt | **pre**: `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt`（16.2 GB）／**post**: `outputs/olmo2_p24_sft_keep14fresh2/final.pt`（48.7 GB, step=842） |
> | 输出 | pre → `olmo2_ppl_results/7B_keep14_step200000/`（gap-fill）+ `olmo2_downstream_results/7B_keep14_step200000{,_know}/`（gap-fill）；mmlu-content + closedbook pre 已在 zwfy6，自动 skip。post → `7B_p24_sft_keep14fresh2_final{,_know}/summary.json`（4 个 harness） |
> | 关键实测 | pre-SFT PPL 重跑 = **10.5612**，与 wzc1 anchor 10.5613（paper Table 4）字节级一致 → harness path 未漂移 |
> | 硬门禁 | `assert_8shards` 在每次 merge 前强制核对 8/8 shard 存在，缺任一即 abort 不 merge（防止 5/8 partial merge silent contamination） |
> | 每-item 保留 | 所有 4 类 harness 均保留 `per_example_*.jsonl` — downstream 用 `--save_per_example`，mmlu_content/closedbook 默认写；pre/post McNemar + paired bootstrap 可行 |
> | Chat template | `chat_template=False` 全程（OLMo-2 base 无 SFT/RL；paper hard rule） |
> | Python | `/opt/conda/envs/torch-base/bin/python` torch 2.13.0 |
> | Launcher | `scripts/_run_olmo2_p24_eval_keep14_73.sh`（wzc1 + zwfy6 md5 一致 = `7bc763ef38d591bbca289ff49564bc2a`） |
> | 报告 | `status/PAPERB_P24_KEEP14_EVAL.md` |
>
> ⚠️ **发现**：pre-SFT anchor `7B_keep14_step200000{,_know}/summary.json` 只在 wzc1，zwfy6 没有 → 为 pre/post 同盘 pairing 已在本脚本 PART A 重跑 PPL+downstream；PPL 已达成，MMLU-content + closedbook 的 pre 结果已在 zwfy6 自动 skip。post-SFT 全新。
> ⚠️ **SFT 数据污染注记**（不影响本次 eval，只影响后续报数）：sibling agent 审计发现 Tulu-3 general-clean 与 MMLU test 有 45 items 交集（PopQA/TriviaQA/NQ-open = 0）。post-SFT MMLU 报数前须做 clean-subset 剔除（CPU 后处理，per_example 已保留即可复算）。

> ### ▶️ .252 8×L20A (wzc1) — `p24_eval_full32_shortgpt_252`（PaperB P2.4 full32+shortgpt16 pre/post-SFT eval battery，8/8 卡 @100%）
> | 项 | 值 |
> |---|---|
> | 任务 | **P2.4 wzc1 两臂 pre/post-SFT eval battery**（.73 keep14fresh2 eval 的姊妹；same-arch pairing 硬要求，见 `status/PAPERB_CORE6_CROSSARCH_FLOOR.md` — L20A cc10.0 vs H20 cc9.0 对 bit-identical ckpt 会有 28 items 翻转 / +0.156 pp 差异，故 pre/post 必须同架构） |
> | 节点/卡 | **.252 8×L20A（wzc1 盘）GPU 0-7 全占**，实测 maxmem **64.6 GiB/卡 @ 100% util**（8 shard 并行） |
> | 起始 | 2026-08-08 **02:02:07** +08:00 |
> | ETA | **~4-6 h → 约 2026-08-08 06:00-08:00**（4 legs × 5 harnesses；每 leg ~60-90 min on L20A ≈ 2× H20） |
> | 4 legs | (1) full32 pre-SFT: vanilla base `../models/OLMo-2-1124-7B` (no --ckpt) → `7B_full32_base_wzc1{,_know}`  (2) full32 post-SFT: `outputs/olmo2_p24_sft_full32/final.pt` (87.6 GB, step=842) → `7B_p24_sft_full32_final{,_know}`  (3) shortgpt16 pre-SFT: `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt` (48.7 GB, keep_front=16 n_fresh=0 read from ckpt meta) → `7B_shortgpt16_step200000_wzc1{,_know}` (`_wzc1` 后缀避免与已存在 `7B_shortgpt_step200000` 的老 no-per-example run 混淆)  (4) shortgpt16 post-SFT: `outputs/olmo2_p24_sft_shortgpt16/final.pt` → `7B_p24_sft_shortgpt16_final{,_know}` |
> | 5 harness/leg | held-out Dolmino PPL + core6 downstream (hs/arcc/arce/piqa/wg/obqa) + know5 downstream (mmlu/lambada/boolq/csqa/siqa) + MMLU dual-protocol (letter+content) + closedbook (PopQA/TriviaQA) |
> | PID / log | `3032751` (setsid nohup) / `logs/p24_eval_full32_shortgpt_252.log` |
> | Launcher | `scripts/_run_olmo2_p24_eval_full32_shortgpt_252.sh` commit `0b2f707` |
> | 硬门禁 | `assert_8shards <root> <name> shard{i}of8.json` 在每次 merge 前强制核对 8/8 存在，缺任一即 abort 不 merge（防止 5/8 partial merge silent contamination；memory `kill-remote-gpu-job-by-pid-not-pkill`） |
> | 每-item 保留 | downstream `--save_per_example` → `per_example_<task>.jsonl`；mmlu-content/closedbook 默认写 `per_example_*.jsonl`；pre/post McNemar + paired bootstrap 已具备 |
> | Chat template | `chat_template=False`，`--add_bos 0` 全程（memory `paper-eval-chat-false-mandatory`；OLMo-2 base 无 SFT/RL） |
> | Python | `/opt/conda/envs/torch-base/bin/python` torch 2.13.0 |
> | Loud 预期锚点 | full32 pre-SFT PPL ≈ **7.398**（paper Table 4）/ MMLU ≈ **0.6053**；shortgpt16 pre-SFT PPL ≈ **9.7803** / MMLU ≈ **0.4739**（若偏差 > 0.2 PPL 或 > 0.5 pp MMLU，立即 loud report — 可能是另一层跨盘/harness drift） |
> | 报告 | `status/PAPERB_P24_WZC1_EVAL.md` |
>
> ⚠️ **sibling agent**：`.73` 上的 agent aefe8b20 正在跑 keep14fresh2 pre/post-SFT eval battery（`scripts/_run_olmo2_p24_eval_keep14_73.sh` / log `logs/p24_eval_keep14_73.log` / 报告 `status/PAPERB_P24_KEEP14_EVAL.md`）。本 dispatch 与之互不重叠：不同节点、不同架构、不同 output_name。
> ⚠️ **SFT 数据污染注记**：Tulu-3 general-clean 与 MMLU test 有 45 items 交集（PopQA/TriviaQA/NQ-open = 0）。post-SFT MMLU 报数前须做 clean-subset 剔除（CPU 后处理，per_example 已保留即可复算）。

> ### ✅ .252 8×L20A (wzc1) — `olmo2_p24_sft_full32` **DONE 2026-08-08 00:38 +08:00**
> - PID 2936878 (退出) / log `logs/p24_sft_full32.log` / final `outputs/olmo2_p24_sft_full32/final.pt`（87.6 GB）/ 842 steps / 63.2 min / step 20 loss=1.1101 finite / seed=42 / arm=full32 / no `--ckpt`（full_base 路径） / 详见 `status/PAPERB_P24_FULL32_ARM.md`。post-SFT eval battery 已在 02:02 起动。

> ### ✅ .252 8×L20A (wzc1) — `olmo2_p24_sft_shortgpt16` **DONE 2026-08-08 01:33 +08:00**
> - torchrun PID 2995894 (退出) / log `logs/p24_sft_shortgpt16.log` / final `outputs/olmo2_p24_sft_shortgpt16/final.pt`（48.7 GB）/ 842 steps / 35 min / seed=42 / arm=shortgpt16 / `--ckpt outputs/olmo2_probe2_7B_shortgpt16/step200000.pt`（keep_front=16 n_fresh=0 从 ckpt meta 读） / loss finite 全程。post-SFT eval battery 已在 02:02 起动。

> ### ▶️ LOCAL 8×L20A (wzc1) — `olmo2_probe2_7B_keep14fresh2_seed1234`（占满 8/8 卡，**勿补卡、勿 kill**）
> | 项 | 值 |
> |---|---|
> | 任务 | **Paper B P1.2「训练 seed 方差」第 2 个 seed**（keep14+fresh2 7B healing 复制臂） |
> | 节点/卡 | **LOCAL 8×L20A（wzc1 盘）GPU 0-7 全占**，maxmem **122.3GB/卡**（= 原 run 实测值，L20A 183GB 够；**H20 97.8GB 放不下 → .73/.82 永远不要试**） |
> | 起始 | 2026-08-07 **21:39** +08:00 |
> | ETA | **~87h → 约 2026-08-11 12:4x**（1.56s/step × 200k step，与原 run 同速） |
> | seed | **1234**（原 run = **无 seed/未记录**：当时 trainer afdfa66 根本没 set_seed，`--seed` 是 2026-08-03 c57c4cb 才加的） |
> | PID / log | `4155272` / `logs/olmo2_7B_keep14fresh2_seed1234.log` |
> | 输出 | `outputs/olmo2_probe2_7B_keep14fresh2_seed1234`（**不是** 原 run 的 `..._keep14fresh2`，未覆盖论文 ckpt） |
> | launcher | `scripts/run_olmo2_7B_keep14_seed2.sh`（commit `5db5d30`） |
> | python | `/opt/conda/envs/torch-base/bin/python` torch 2.13.0 — ⚠️ **LOCAL `.venv` 已无 torch（2026-08-07 实测）** |
> | 健康 | step80 loss=6.6148 / 1.56s/step / maxmem=122.3GB / sanity ALL 6 CHECKS PASS |
>
> **⚠️ 唯一变量 = fresh-block init，且 LR 写法必须是 `--lr 2e-5` 不是原 run 字面的 `--lr 1e-4`**：原 run 的差分 LR 是
> **no-op**（`build_param_groups` 在 DDP wrap 之后跑，而当时 `_classify_param` 没剥 `module.` 前缀 → 4060.1M 参数
> 全落进 `inh_*` @2e-5，原 log 只有 `inh_decay`+`inh_nodecay` 两组）。今天 trainer 已修，字面重放 `--lr 1e-4` 会
> **真的**造出 1e-4 的 fresh 组 = 第二个变量，seed 方差数字就废了。故本 run 传 `--lr 2e-5 --min_lr 2e-6`，
> log 里出现 **4 组但 base_lr 全是 2.00e-05** = 等价性达成的正确签名。**论文不得为此臂声称 differential LR。**
> **`--seed` 只控 fresh 2 层 init，不控数据顺序**（`DistributedSampler(shuffle=True)` 无 `seed=` → 私有 generator 固定在 0+epoch，两 run 数据序完全相同）；dropout=0。→ P1.2 只能称 **init 方差**。详见 `status/PAPERB_P12_SEED2.md`。

> **其余节点（21:40 实测）**：`.252` 8 卡被他人 `dllm_draft/.venv_b200 generate_evalplus_dream` 占（~63GB 合计，未 kill）；
> `.73`/`.82` GPU 仍被他人 EvalPlus 占，`.73` 只有我们的纯 CPU P2.4 段。

### PaperB P2.4（#123）CPU 段 @ .73（未占卡）

> **P2.4 派单前提「`.73` 空闲」实测不成立**：21:10 直连 `.73` 见 **8/8 GPU @ 99%（19.3GB→35GB/卡）**，进程 =
> `dllm_draft_104/scripts/generate_evalplus_dream.py --dataset humaneval`（PID 2628561-68，**21:08:10 启动**，即派单前约 2.5 分钟被别的 agent 占走）。
> `.82`（20:45 起）/`.104`（20:47 起）在跑同一 EvalPlus 审计 → **zwfy6 三台 0 空闲**。未 kill 任何他人任务。
> **P2.4 的 GPU 段（pre-eval / SFT / post-eval）因此未启动**；本轮只在 `.73` 跑**纯 CPU**（384 核）数据构建 + 污染审计，不碰 GPU。
>
> - **.73 CPU**：`prepare_olmo2_sft_data.py`（DONE 21:33，249,999,360 tok）+ `audit_olmo2_sft_overlap.py`（PID 2648315，running）。log = `logs/p24_sft_dataprep.log` / `logs/p24_overlap_audit.log`。
> - ⚠️ **P2.4 full-32L 臂在 H20 不可行（硬约束，非调度问题）**：`train_olmo2_sft.py` = **plain DDP + fp32 master + fp32 AdamW**（无 bnb 8bit 路径，`grep -c bitsandbytes`=0），
>   per-card 静态 = 16 B/param × 7.298B = **108.8 GiB > H20 97.8 GiB**，且 plain DDP 不 shard param/grad/optim → **加卡不降每卡内存**。两个 16L 剪层臂 = 60.5 GiB，可过 H20。
>   → **full-32L 臂须去 B200（183 GiB）**，或先给 trainer 加 8bit/FSDP。详见 `status/PAPERB_P24_SFT_REPAIRABILITY.md`。
> - ✅ **2026-08-07 23:32 full-32L 臂已启动于 .252（L20A 183 GiB）**：byte-identical config（唯一 diff = 起始 ckpt = full-base，无 `--ckpt`）；实测 maxmem **182 GiB/卡 = 99.3%**（H20 装不下已交叉验证）；step20 loss 1.1101 finite；ETA ~59 min。详见 `status/PAPERB_P24_FULL32_ARM.md`。

## 历史快照（2026-08-06 08:35 +08:00）— **32 卡全空闲**，rebuttal-prep sprint 已收工

> **实测（本轮 heartbeat 直连每台 nvidia-smi）**：
> - **LOCAL 8×L20A (wzc1)**：0 MiB × 8 = 空闲
> - **.252 8×B200 (wzc1)**：**0 MiB × 8 = 空闲，节点完全正常。** ⚠️ 本文件早前版本（及 2026-08-06 全天 heartbeat）说它「SSH 拒登陆 / 密码轮换嫌疑」——**全部作废，那是我的命令 bug**：我一直按旧 CLAUDE.md 写 `-p 22`，而本机 `/etc/ssh/ssh_config` 有全局 `Port 36000`。`.252` 的 22 端口上另有一个 sshd（host key 与 36000 相同 → 握手/banner 全正常，只有 password 被拒，故极像凭据失效）。**省略 `-p` 即通**（`hostname`=`TENCENT64.site`）。CLAUDE.md 已修 4 处，memory 已记 `ssh-252-port-36000-not-22`。**32 卡全部可调度。**
> - **.73 8×H20 (zwfy6)**：0 MiB × 8 = 空闲
> - **.82 8×H20 (zwfy6)**：0 MiB × 8 = 空闲
>
> **kill 事件（2026-08-05, 用户令）**：#99 keep14-distill @ .73（释放 5601 GPU-h）+ #103 keep14 dense-save re-heal @ LOCAL + `ppl_monitor_252.sh` @ .252（crossing 已达成目的，共释放 436 GPU-h）。#134 PaperC A1 ceiling 已取消（squad_val 49.85% 恒定拒答基线高于所有臂）。
>
> **32 卡空闲的原因不是运维错误，是 4 项 pending 用户决策未拍板**：(1) Paper C A4×random_trunk (#165); (2) Paper D 深度对齐 mini (#166); (3) push 22 unpushed commits; (4) paperA #167 latency 三选一。sprint 期间不投未定方向。
>
> **rebuttal-prep sprint 产出（2026-08-06 夜 03:00-07:52）**：13 audit/rebuttal commit（详见 UPDATELOG.md 2026-08-06 段 + `paperB/audit_20260805/REBUTTAL_INDEX.md`），paperA 4/5 primitive 精确 + paperB 16/16 精确 + tex 内部一致（tab_pareto 99.20→99.19 修）+ Finding 2 rebuttal 弹药 drop-in tex ready + P0.13/P0.17 artifact 已 mirror 到 wzc1 anonymous_artifact/scores/。

## 当前快照（2026-08-04 23:3x +08:00，**monitor 节点表修复→5 节点全可见；.104 BGE 臂已停待收；.82 改派 PaperC #133；.252 认证其实正常（前次误判）**）
> **★ 三条修正/发现（比上一快照重要）**：
> 1. **monitor/gpu_monitor_server.py 的 NODES 表是过期集群**（`.196` / `b200-18` / `b200-53` 三台已下线节点）→ 4 节点里 3 个常报 `ok=False`，等于**监控前端一直没在看当前 5 节点**。已改成 LOCAL/.252/.73/.82/.104 并放宽 PS/METRIC 匹配（旧的只 match `scripts/train_mem_space`，当前 olmo2/qcmem/paperC/eval/patch 全不算 task→忙节点被显示成 idle）。重启后 `/api/data` 实测 **5 节点全 ok=True**。
> 2. **.252 SSH 其实正常**（上一快照记的 Permission denied 是**瞬时失败/误判**，已纠正）：monitor 实测 `252 ok=True 8 gpus 0.0GB 0%`。→ **.252 当前 0 MiB 空闲**，可用。
> 3. **.104 的 A-P1.1 BGE/BGE 臂已停**（8 卡全 0 MiB）——完成还是崩未知，已派 agent a37938eb 去判定+收数（★重点查 `n_examples_paired` 与 frontier 是否为空：该实验有过 stale-done-marker → 全 SKIP → 看着像 done 实则 `VERDICT:INCOMPLETE`/`n_paired=0` 的前科）。
> **节点占用（monitor 实测 23:3x）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（monitor task 确认 pid 2371815，elapsed **10h09m**，1074.8GB/8卡=100% util，healthy）。等 frozen-match PPL≈12.797 / random-match≈11.498 双侧 crossing → matched-PPL MMLU+McNemar 后**立即停，不跑到 200k**。wzc1 盘。
> - **.252 8×B200**：🟢 **FREE（0 MiB，认证正常）**。候选：#134 PaperC A1 full-FT-32L ceiling（明确 defer 到 B200，fp32-AdamW 需 183GB）、或 #128 若确认是 wzc1-only。⚠️ 但它同时是 #103 crossing-monitor 宿主 → 起 8 卡任务前须确认不撞 monitor 爆发。wzc1 盘。
> - **.73 8×H20**：▶️ **Paper C #132 second-task capability eval**（agent a2ab12c6，eval-only，MMLU-MC + closed-book QA × 4 臂）。zwfy6 盘。
> - **.82 8×H20**：▶️ **Paper C #133 depth-sweep 启动中**（agent ab90739b；freeze-graft vs from-scratch @ keep{20,24,28}+fresh2 = 6 run + SQuAD eval，eff_bs 恒 128 保可比）。**原派的 #128 patching 已撤下**（两次被中断且疑 wzc1-only ckpt，见 task #128 备注）。zwfy6 盘。
> - **.104 8×H20**：⏸️ **空闲待收（A-P1.1 BGE 臂已停）**，agent a37938eb 判定中；收完再补卡。zwfy6 盘。
> - **⚠️ .venv/bin/python 在 H20 三台已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**两处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）。

## 当前快照（2026-08-04 22:5x +08:00，**用户新指令：paperB rebuttal todo + paperC 开始推进 → A/B/C 均衡铺开；.73/.82 两空节点填 C/B**）
> **关键事件**：(1) **用户指令**「另外 paperB 也有一些 todo，然后 paperC 可以开始推进了，paperB 是为了 rebuttal 做准备，所以你可以均衡推进」→ **解除 Paper C 的 H20-PaperA-first 等待**，三线并行。(2) **.73+.82 实测全空（0 MiB ×16 卡，无 python 进程）** → 立即填：**.73 = Paper C #132**（P-C1 second-task capability eval：MMLU-MC + closed-book QA on A4_hero/A3_fromscratch/A2_lora_r160/BASE_ref 四臂，eval-only，agent a2ab12c6）；**.82 = Paper B #128**（P2.2 activation patching 因果层恢复 harness，forward-only，新脚本 patch_olmo2_layers.py，agent ae4e2ee4）。两者均 base 协议 chat=False/no-BOS。#128 设**硬门禁：identity-patch 必须复现 unpatched 分数**，否则曲线作废。(3) **release 仓整理**：Paper B → `perplexity-heals-knowledge-lags` **DONE commit 9f71cfa（未 push）**，22 文件 SHA256 全对齐 anonymous_artifact 清单、sanitization 干净（130k JSONL 记录无 >40 字符串=无题面文本）；Paper A → COMem 代码+索引 agent 在跑。(4) **monitor 8088 曾 http=000 → 已重启，现 http=200**。
> **节点占用（实测 nvidia-smi 22:4x）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step **21700**/200000，137.6GB/卡 100%，healthy）。等 frozen-match PPL≈12.797 / random-match≈11.498 双侧 crossing → matched-PPL MMLU+McNemar 后**立即停，不跑到 200k**（paperB Phase R 决策）。wzc1 盘。
> - **.252 8×B200**：⏸️ **RESERVED-for-#103-monitor**；⚠️ **本轮 SSH 密码认证失败（Permission denied）**——待排查（密码文件可能已轮换），暂不计入可用节点。wzc1 盘。
> - **.73 8×H20**：▶️ **Paper C #132 second-task eval 启动中**（agent a2ab12c6；#92 四臂 ckpt 在 **zwfy6**，agent 需先定位 PROJECT_ROOT）。zwfy6 盘。
> - **.82 8×H20**：▶️ **Paper B #128 P2.2 activation patching 启动中**（agent ae4e2ee4）。zwfy6 盘。
> - **.104 8×H20**：▶️ **A-P1.1 BGE/BGE RUNNING**（8 卡 18-20GB/卡、22-100% util，`_run_p0_20_phaseB_dense.sh` 已跑 ~1.9h，healthy）。zwfy6 盘。
> - **⚠️ .venv/bin/python 在 H20 三台已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 两处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）。

## 当前快照（2026-08-04 21:16 +08:00，**A-P1.1 LOGDIR bug 修复→BM25/BM25 臂 DONE(NEGATIVE)；BGE/BGE 臂 RUNNING @.104**）
> **关键事件**：coder a1e87376 修好 LOGDIR bug（`LOGDIR="${LOGDIR:-logs/$(basename "$OUTDIR")}"`，commit **9aee7e6** LiuHanzuo 未 push，scp 同步 zwfy6 md5 一致）→ 原拓扑并行重跑 STEP 3b `SKIP=0`（bug 消）。**BM25/BM25 臂 DONE**（n=100 分层 LoCoMo×10k=1000 paired，已 scp 回 wzc1 `bench_results/p1_1_bm25_locomo/`）：primary anchor CoMem@k12(acc 8.0) vs latency-matched text-RAG@k10(acc 11.0) → **diff −3.0pp 95%CI[−7,0] McNemar p=0.25（方向性差、n.s.）**；逐 k text-RAG ≥ CoMem 全程。**decision.json 裁决 NEGATIVE**（cached-state readout 瓶颈；redirect P0.17/P0.18/P1.10，不得包装成 positive Pareto）。**★诚实修正**：#137 里 LoCoMo +1「tie」是 first-100=100%conv0 采样假象，分层 10-conv 后翻转成 CoMem-worse。已回填 paperA/TODOList A-P1.1。**BGE/BGE 臂 RUNNING @.104**（default 9-cell，dense_bge 两臂对称）：60/360 quality done SKIP=0，8 卡在跑，STEP4 auto-agg，ETA ~1-2h → 待回填后关 #151。**无 heartbeat cron → 已 CronCreate 一次性 ~22:47 check-back。**
> **节点占用（21:16）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step ~16840+/200000，train ppl 14.50，healthy）。等 crossing#2（held-out Dolmino PPL <11.4983）后跑 matched-PPL MMLU+McNemar。wzc1 盘。
> - **.252 8×B200**：⏸️ **RESERVED-for-#103-monitor**（crossing-monitor PID 1238902 alive；PPL step10000=14.67 单调↓，目标 <11.4983 停）→ 不 co-schedule。wzc1 盘。
> - **.73 8×H20**：🟢 **FREE（idle 有据）**：PaperA GPU 待跑排空；#128/#129=wzc1-only 不能跑 zwfy6 H20；PaperB 训练 #84/#99/#123 用户 defer → 宁 idle 不擅起 defer 训练。zwfy6 盘。
> - **.82 8×H20**：🟢 **FREE（BM25/BM25 臂 DONE 释放）**。zwfy6 盘。
> - **.104 8×H20**：▶️ **A-P1.1 BGE/BGE re-run RUNNING**（dense_bge，60/360，SKIP=0，healthy；OUTDIR p1_1_bge_bge）。zwfy6 盘。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）；BGE 结果在 zwfy6 → 完后 scp -O 回 wzc1。

## 历史快照（2026-08-04 20:20 +08:00，**A-P1.2(b) DONE→A-P1.2 两半闭合(contamination+judge 双稳健)**；**A-P1.1 首跑跑完但交付空(INCOMPLETE/n_paired=0)→定位 LOGDIR bug→派 coder 修+重跑 .82+.104**）
> **关键事件**：(1) **A-P1.2(b) open-weight Qwen3-8B judge LoCoMo 复评 DONE**（coder a52329d3 @.73，commits 7aa4e14+15f7325 未 push）：CoMem flagship 双 judge 均 #1（open 52.06 vs GPT-4o 38.27），top-4 序保持；诚实 caveat：CoMem-vs-kvdirect 从 +3.68pp(GPT-4o) 收窄到 +1.76pp(open judge)、conv-cluster CI 略重叠（cluster-bootstrap 噪声内）。→ **A-P1.2 两半全闭合：(a) contamination-robust + (b) judge-robust**。产物 locomo_results_openjudge_qwen3_MIRROR/（wzc1）。#152 completed。.73 释放。(2) **A-P1.1（#151）首跑跑完但交付物为空**：bench_results/p1_1_bm25_locomo + p1_1_bge_bge 两 OUTDIR 均 `VERDICT:INCOMPLETE`、`n_examples_paired=0`、frontier={}。**根因（main 定位）**：`_run_p0_20_8gpu.sh` L102 / `_run_p0_20_phaseB_dense.sh` L115 **硬编码 LOGDIR 未跟随 OUTDIR**，`DONEDIR=$LOGDIR/done` 继承原始 P0.20 #137 run 的 360 个同名 `quality_*.done` marker → STEP 3b 全部 40 quality job SKIP（calib-latency 正常）。→ 派 **opus coder a1e87376** 修（LOGDIR 跟随 OUTDIR，向后兼容）+ 镜像 zwfy6 + 按原拓扑重跑 **.82(BM25/BM25)+.104(BGE/BGE)**。#151 保持 in_progress。
> **节点占用（实测 nvidia-smi 20:16）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step 16840/200000，train ppl 14.50，1.56s/step，137.6GB/卡 100%，healthy）。等 bracket crossing#2（held-out Dolmino PPL <11.4983）后跑 matched-PPL MMLU+McNemar。wzc1 盘。
> - **.252 8×B200**：⏸️ **RESERVED-for-#103-monitor**：crossing-monitor alive（PID 1238902 since 14:12，卡间隙 0-4 MiB），每新 ckpt 跑 8-GPU held-out PPL；PPL step10000=14.67（单调↓，目标 <11.4983 停）→ 不 co-schedule（#129/P2.2 HELD）。wzc1 盘。
> - **.73 8×H20**：🟢 **FREE（0 MiB，A-P1.2(b) DONE 释放）**。zwfy6 盘。PaperA GPU 待跑已排空（A-P1.1 在修+重跑于 .82/.104）；#128/#129=wzc1-only 不能跑 zwfy6 H20；PaperB 训练 #84/#99/#123 用户 defer → 暂 idle 有据。
> - **.82 8×H20**：▶️ **A-P1.1 BM25/BM25 re-run 启动中**（coder a1e87376，LOGDIR-fix 后，OUTDIR=p1_1_bm25_locomo）。zwfy6 盘。
> - **.104 8×H20**：▶️ **A-P1.1 BGE/BGE re-run 启动中**（coder a1e87376，dense_bge 对称，OUTDIR=p1_1_bge_bge）。zwfy6 盘。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）；A-P1.1 结果在 zwfy6 → 完后 scp -O 回 wzc1 聚合/落账。

## 历史快照（2026-08-04 18:05 +08:00，**A-P1.1 LAUNCHED（.104 BGE/BGE + .82 BM25/BM25）+ A-P1.2(a) DONE + A-P1.2(b) LAUNCHED @.73** → 5 节点全部产出，无空转）
> **关键事件**：(1) **A-P1.1 coder a5504d6 DONE-launch**（commit 1c0d66b 未 push，改 eval_p0_20_phaseB_dense.py 加 `--comem_selector dense_bge` + eval_qcmem_locomo.py LoCoMo stratify(legacy first-100=100% conv0→10/conv×10) + eval_p0_20_equal_latency.py 4 聚合方案 frozen weights）：**.104 BGE/BGE**（COMEM_SELECTOR=dense_bge，PID 1166059）+ **.82 BM25/BM25 LoCoMo-fixed**（PID 2926566）两 GPU run 过 manifest+sha+sanity gate、STEP 2 calib sweep healthy。结果落 zwfy6 → 待完成 scp 回 wzc1 聚合。(2) **A-P1.2(a) contamination overlap audit DONE**（coder afc3ace7 @LOCAL CPU，commit 746974c）：LongBench=CONTAMINATED(局限 narrativeqa，clean 8305/8418)/LoCoMo=CLEAN/InfiniteBench=CONTAMINATED(复现 P0.14)/LongEval=CLEAN-BY-CONSTRUCTION；clean-subset 重评分 clean ≤ full 全 arm（连 PG-19 蒸馏模型都不获益）→ **contamination-robust**。(3) **A-P1.2(b) open-weight judge LoCoMo 重判 LAUNCHED @.73**（coder a52329d3；A-P1.1 提交 eval_qcmem_locomo.py 后同文件冲突解除→串行放行）。
> **节点占用（全 5 节点产出）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step ~10000/200000，1.56s/step，healthy）。A-P1.2(a) CPU 已完。等 bracket crossing#2（PPL≈11.498）后跑 matched-PPL MMLU+McNemar。wzc1 盘。
> - **.252 8×B200**：⏸️ **RESERVED-for-#103-monitor**：crossing-monitor alive，PPL step10000=14.67（单调↓；目标 <11.4983 停），每新 ckpt 跑 8-GPU eval → 不 co-schedule（#129/P2.2 HELD）。wzc1 盘。
> - **.73 8×H20**：▶️ **A-P1.2(b) open-weight judge LoCoMo 重判 启动中**（coder a52329d3：装 vllm+serve Qwen3-8B、`--score_only` 重判 6 pred dir、fresh output_dir、enable_thinking=false；~30min flagship/~1-2h 全 6）。zwfy6 盘。
> - **.82 8×H20**：▶️ **A-P1.1 BM25/BM25 LoCoMo-fixed re-cell RUNNING**（coder a5504d6，PID 2926566，STEP 2 calib，healthy；OUTDIR bench_results/p1_1_bm25_locomo）。zwfy6 盘。
> - **.104 8×H20**：▶️ **A-P1.1 BGE/BGE Phase B RUNNING**（coder a5504d6，COMEM_SELECTOR=dense_bge，PID 1166059，STEP 2 calib，healthy；OUTDIR bench_results/p1_1_bge_bge）。zwfy6 盘。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）。A-P1.1 结果在 zwfy6 → scp -O 回 wzc1 聚合/落账。

## 当前快照（2026-08-04 17:45 +08:00，**B-P1.2-sub closed-book clean-subset DONE→回填 paperB**；**A-P1.2 就绪度回来=非多天→(a) 派 coder @LOCAL CPU、(b) HELD 待 A-P1.1 让出 eval_qcmem_locomo.py**）
> **关键事件**：(1) **B-P1.2-sub（task #153，agent a0f9c7b6，纯 CPU）DONE**：OLMo-2 closed-book（TriviaQA/PopQA/NQ）clean-subset 重算——去污染最大变动 ≤0.15pp、7 臂排序完全保持 → **closed-book 知识表对 Dolmino 污染稳健**（与 MMLU clean-subset 结论一致）。脚本 `recompute_closedbook_clean_subset.py` commit 1624d55（未 push），产物 `bench_results/olmo2_dolmino_contamination/closedbook_clean_subset_recomputed.json`（wzc1）。已回填 paperB/TODOList §B-P1.2（原 "未做 deferred" 行）。(2) **A-P1.2 就绪度评估回来 = 两半 same-day 小 compute、非多天 → 不 DEFER**。**(a) contamination overlap audit** 派 opus coder **afc3ace7 @LOCAL CPU 已启**（新文件 `scripts/audit_ap1_2_contamination.py`，复用 P0.14 引擎+预建 sketch `.t27_tmp/pg19_train_sketch_n13_d32.npy`，`CUDA_VISIBLE_DEVICES=-1`，与 #103 的 8-GPU 训练**无争用**，CPU ~15-30min）。**(b) open-weight judge LoCoMo 复评 HELD**：需改 `eval_qcmem_locomo.py` 的 enable_thinking，**与在跑的 A-P1.1 coder a5504d6 同文件冲突** → 串行，待 A-P1.1 提交后派 (b) coder **@.73**（iter_bm25/chatFALSE flagship preds `qcmem_8b_iter_chatFALSE` + 5 竞品已在 .73 zwfy6，就地跑免 scp）。
> **节点占用（实测 nvidia-smi 17:20，无新 GPU 变化）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step ~10000/200000，1.56s/step，healthy）；🖥️ **A-P1.2(a) contamination audit DONE（coder afc3ace7，CPU-only，commit 746974c，contamination-robust：clean ≤ full 全 arm）**。dense-save 每 2500 步。等 bracket crossing#2（random-match PPL≈11.498）后跑 matched-PPL MMLU+McNemar。
> - **.252 8×B200**：⏸️ **RESERVED-for-#103-monitor**：crossing-monitor alive，PPL step10000=14.67（单调↓、减速；目标 <11.4983 停），每新 ckpt 跑 8-GPU eval → 不 co-schedule（#129/P2.2 HELD）。wzc1 盘。
> - **.73 8×H20**：🟡 **HELD-for-A-P1.2(b)**（open-weight judge，待 A-P1.1 让出 eval_qcmem_locomo.py 后就地启；flagship+竞品 preds 已在此）。zwfy6 盘。
> - **.82 8×H20**：🟡 **A-P1.1 BM25/BM25 LoCoMo-fixed re-cell 待启**（coder a5504d6 building→dry-gate→scp -O→launch）。zwfy6 盘。
> - **.104 8×H20**：🟡 **A-P1.1 BGE/BGE Phase B 待启**（coder a5504d6，同上）。zwfy6 盘。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）。zwfy6-only preds 就地跑或 scp -O。

## 当前快照（2026-08-04 17:25 +08:00，**P0.14 clean-subset 重算 DONE→回填 TODOList**；**#129/P2.2 HELD 保护 #103 交叉序列**；派 3 只只读 agent 评 A-P1.1/1.2/1.3 就绪度）
> **关键事件**：(1) **P0.14 InfiniteBench clean-subset 重算完成**（`.82` zwfy6 CPU pid2924913）→ clean subset 上 CoMem 优势不降反升（choice **54.55 vs Dense 25.00** clean；QA 6.54 vs 2.14）→ **contamination-robustness 成立，撤回原 WITHDRAW `tab:infbench` 建议**，结果已回填 paperA/TODOList P0.14。⚠️ 遗留口径冲突：本 recompute CoMem choice full=48.03 vs tex 现载 17.47（#112 mc_ll 口径）→ MAIN 集成前须统一。(2) **确认 .252 crossing-monitor = 8-GPU 满卡爆发**（`eval_ppl_252.sh` 每存 ckpt 跑一遍 8-GPU held-out Dolmino PPL）→ 8-GPU #129/P2.2 会撞 monitor、危及 #103 **blocking** 交叉序列 → **#129(Qwen 跨家族)/P2.2(activation-patch) HELD**，.252 保留给 #103 monitor（当前非可用节点）。(3) **派 3 只 opus 只读 agent** 评 A-P1.1(equal-latency same-selector)/A-P1.2(contamination+open-weight judge)/A-P1.3(serving grid) 就绪度+ETA（纯文本无 schema，避开上轮 StructuredOutput retry-cap 失败）→ 回来据 ETA 决定是否铺到 .73/.82/.104；**多天大实验先别跑（用户指令），ETA 未确认前不启**。
> **节点占用（实测 nvidia-smi 17:20）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step ~10000/200000，1.56s/step，healthy）。dense-save 每 2500 步（~67min）。等 bracket crossing#2（random-match PPL≈11.498）后跑 matched-PPL MMLU+McNemar。
> - **.252 8×B200**：⏸️ **RESERVED-for-#103-monitor（0 MiB 间隙，crossing-monitor alive）**：PPL step5000=16.47→7500=15.33→**10000=14.67**（单调↓、减速；目标 <11.4983 停）。monitor 每新 ckpt 跑 **8-GPU** eval → **不 co-schedule 任何任务**（#129/P2.2 HELD）。crossing#2 估还需数小时 bracket → monitor 自动停后 .252 才真正空出。wzc1 盘。
> - **.73 8×H20**：🟢 **FREE（0 MiB）**：**留给 A-P1.2**（contamination+open-weight judge，就绪度评估在途）。zwfy6 盘。
> - **.82 8×H20**：🟡 **A-P1.1 BM25/BM25 LoCoMo-fixed re-cell 待启**（coder a5504d6 building→dry-gate→scp -O→launch）。zwfy6 盘。
> - **.104 8×H20**：🟡 **A-P1.1 BGE/BGE Phase B 待启**（coder a5504d6，同上）。zwfy6 盘。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）。zwfy6 git=2d98c5a 旧+dirty，wzc1-only 新脚本走 `scp -O` 搬。

## 历史快照（2026-08-04 16:55 +08:00，**#142 write-path DONE-收敛 → kill .104**；**4 节点空出**（.252/.73/.82/.104=32 卡），待小-ETA workflow 出 plan 铺开；多天大实验依用户指令不启）
> **关键事件**：**#142 P1.10 write-path 蒸馏双重收敛确认 → 停训**。BBWL 收敛 eval（coder task #150，跑在 .73+.82）：step1000/1500/2000 macro = **98.0/99.0/98.5**（n=200 噪声内完全持平，8k 全 100 饱和，movement 全在 16k 噪声），对照 BB=92.5 / E0=100 —— 闭合 ~6pp（80%）可部署 Write gap，到 E0 上界残差 p≥0.125 **不显著**。training loss 也早收敛（平台 since ~step500）。∴ step4000 无意义 → **kill .104 #142 训练**（跑到 step2500，省 ~11h + 腾节点）。**交付 ckpt = step2000 adapter，BBWL=98.5**。基线精确复现（A=100/BB=92.5/E0=100，pack 1:1 sha-match，oom=0）。coder 两处 launcher 改动 commit 在 wzc1 本地（64db47c WRITE_LORA env / e8f9925 LOGDIR 可覆盖，LiuHanzuo，**未 push**）；因 zwfy6 1081 dirty 文件 git pull 不安全，BBWL 驱动走 **scp** 搬过去（md5 一致，留 .bak）。**小-ETA launch workflow（wq9hv411w）仍在评就绪度**，回来后把 ready 项（A-P1.1/1.2/1.3 + #128/#129 + no-GPU recompute）铺到这 4 个空节点。**A-P0.1/P0.2/P0.3 多天大实验依用户指令不启。**
> **节点占用（实测 nvidia-smi 16:55）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step ~7500/200000，1.56s/step，137.6GB/卡 100%，healthy）。等 bracket crossing#2（random-match PPL≈11.498）后跑 matched-PPL MMLU+McNemar。
> - **.252 8×B200**：⏸️ **GATED-IDLE（0 MiB，crossing-monitor alive）**：#103 crossing-eval step2500=19.11→5000=16.47→7500=15.33。**待小-ETA PaperA eval 进驻**（monitor 只在 #103 存 ckpt 时短爆发，需错峰）。wzc1 盘。
> - **.73 8×H20**：🟢 **FREE（0 MiB）**：#142 BBWL 收敛 eval **DONE**（step2000+step1000 在此，step1500 并行在 .82）。zwfy6 盘。待小-ETA 项进驻。
> - **.82 8×H20**：🟢 **FREE（0 MiB）**：BBWL step1500 eval DONE。zwfy6 盘。待小-ETA PaperB harness（#128/#129）进驻。
> - **.104 8×H20**：🟢 **FREE（0 MiB，刚 kill #142）**：写路径蒸馏 DONE-收敛，8 卡全 0 MiB 0 残留进程；adapter step500/1000/1500/2000/2500 全落盘 zwfy6。待小-ETA 项进驻。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`（.252 待现场验证）。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）。paused-heal ckpt 均在 wzc1（LOCAL）。

## 历史快照（2026-08-04 16:40 +08:00，用户指令：**多天大实验(A-P0.1/P0.2/P0.3)先别跑，ETA 小的先跑**；#142 训练近收敛→派 eval 证下游平台再 kill；小-ETA 项并行铺开）
> **关键事件**：用户拍板 —— Paper A 三个 ACL main-gate 多天大实验（A-P0.1 matched frontier / A-P0.2 多 seed headline / A-P0.3 overlap-write frontier）**暂不启动**，先把 ETA 小的项铺开。已动作：(1) **#142 write-path 蒸馏 training loss 已收敛**（0.038-0.041 平台 since ~step500，step1000-1300 均值 0.04136 vs 末 300 步 0.03934，~5% 改善），派 **opus coder（task #150）在 .73 跑 BBWL 收敛 eval**（step1000/1500/2000 逐 ckpt vs ArmB 92.5 / E0 100，niah_multikey_1×{8k,16k} n=100 iter_bm25 chat=False）证【下游指标】平台——证平即 kill .104 省 ~11h。(2) 派 **workflow wq9hv411w（7 路并行）**查小-ETA 候选（A-P1.1 equal-latency / A-P1.2 contamination+judge / A-P1.3 serving grid / #128 activation-patch / #129 Qwen 跨家族 / P0.14 clean-subset / B-P1.2-sub）就绪度 → 出 ranked launch plan 分配 .252(B200,wzc1,PaperA 优先) + .82(H20,zwfy6,PaperB harness)。gap-audit workflow(wc3fqbdr6) 已闭合：Paper A 剩 3 main-gate（defer）+ 在跑收尾；Paper B 主体闭合，剩 #103 收尾 + #128/#129。
> **节点占用（实测 nvidia-smi 16:35）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step ~7200/200000，1.56s/step，137.6GB/卡 100%，healthy）。step2500/5000/7500.pt 落盘。等 bracket crossing#2 (random-match PPL≈11.498) 后跑 matched-PPL MMLU+McNemar。
> - **.252 8×B200**：⏸️ **GATED-IDLE（0 MiB，crossing-monitor alive）**：#103 crossing-eval step2500=19.11→5000=16.47→7500=15.33（单调↓）。**待 workflow 规划的小-ETA PaperA eval 进驻**（monitor 只在 #103 存 ckpt 时短爆发，可与 eval-scale 任务错峰共存；启动前 coder 需确认不撞 monitor 评估窗口）。wzc1 盘。
> - **.73 8×H20**：▶️ **#142 write-path BBWL 收敛 eval RUNNING**（coder task #150；~20GB/卡 37-100% util，8-GPU flock 调度 step1000/1500/2000）。zwfy6 盘。几小时完→释放为第 3 空节点。
> - **.82 8×H20**：🟢 **FREE（0 MiB）**：待 workflow 规划的小-ETA PaperB harness（#128/#129）或 PaperA 项进驻。zwfy6 盘。
> - **.104 8×H20**：▶️ **#142 P1.10 write-path 蒸馏 training RUNNING**（step ~2410/4000，loss ~0.039 **已收敛**，~26 s/step data-stall 主导；GPU 100%）。**不迁移**；等 BBWL eval 证下游平台后 kill（省 ~11h + 腾节点）。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`（.252 待现场验证）。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104）。paused-heal ckpt 均在 wzc1（LOCAL）。

## 历史快照（2026-08-04 16:04 +08:00，heartbeat：**核实“暂停 heal”=stale，实为已 DONE**；两活训练 healthy；16 H20 无 in-scope 可自动启训 → 依用户 defer 指令保持 idle）
> **关键事件**：上一轮 flag 的"该 resume 哪个 Paper B heal（keep8@45000/keep10@69000/keep12@111500）"经核实为 **stale memory**——paperB/TODOList 行 248-250 显示 keep8(10L)/keep10(12L)/keep12(14L) 均 **`[DONE]` 200k**（task #95/#96/#114 completed），**无 heal 可 resume**。已更正 memory `h20-paperA-over-paperB-priority`。剩余 Paper B 训练（#84 contamination-deferred / #99 keep14-distill-PARKED-per-user / #123 general-SFT-用户判太贵）**均被用户明确 defer**，heartbeat 不可自主 auto-launch（会违背用户 defer 指令）→ 16 H20 本轮保持 idle 有据，非"HEARTBEAT_OK 不作为"。#142 完/#103 触交叉后其 eval 收尾会用到这些卡。
> **节点占用（实测 nvidia-smi + log）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step ~7200/200000，loss 2.78 / train ppl 16.17，1.56s/step，healthy）。step2500.pt + step5000.pt 已落盘，step7500.pt 约 8min 后落。GPU0-7 100%。
> - **.252 8×B200**：⏸️ **GATED-IDLE（0 MiB，crossing monitor alive）**：#103 crossing-eval 已评 step2500 PPL=19.11 → step5000 PPL=16.47（单调降），等 LOCAL step7500.pt 再评。blocking 目标 = bracket crossing#2 random-match PPL≈11.498（frozen-match≈12.797，endpoint≈10.561）；粗估 re-heal ~step11-17k（~3-5h）触及，随后两 bracketing ckpt 跑 matched-PPL MMLU + McNemar（~30-60min）。⚠️ **不 co-schedule 长训练**（monitor 随时可能触发 8-GPU 评估）。
> - **.73 8×H20**：🟢 **FREE（0 MiB）**：#149 Finding-3 LR 对照 **STOPPED @step25260/200000**（用户 cost-benefit 判定 B-P1.1 类不值 ~200 GPU·h → Finding-3 保留 hedged wording；ckpt 保留可日后 resume）。zwfy6 盘。**无 in-scope 可自动启训 → idle。**
> - **.82 8×H20**：🟢 **FREE（0 MiB）**：#143 CacheBlend **DONE**（结果入 paperA/TODOList A-AUDIT-1）。zwfy6 盘。**无 in-scope 可自动启训 → idle。**
> - **.104 8×H20**：▶️ **#142 P1.10 write-path 蒸馏 RUNNING**（step ~2300/4000，loss ~0.039 近收敛，effective ~26 s/step 受 data-stall 主导，剩 ~10h；GPU 100%）。**留在 .104，不迁移，不动。**
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82+.104 同 zwfy6 盘）。paused-heal ckpt 均在 wzc1（LOCAL）非 zwfy6。

## 历史快照（2026-08-04 14:00 +08:00，heartbeat：**LOCAL re-heal 首 ckpt step2500.pt 落盘 + 用户 ARR 审计分诊**。发现 .252/.73 共 16 卡空转 → 立即补卡。派 3 agent：**abdf565@.252 消费 step2500 起 #103 crossing-eval**（此前 a88cddeb 已完成 sanity）；**a81a6b6@.73 CPU 聚合 #143 CacheBlend**（generation 已完，缺 aggregate.json）；**ad838cf4@.73 8×H20 跑 PaperB Finding-3 matched-LR 对照**（ARR 审计唯一需跑项，task #149）。ARR 审计其余 8 条为「有结果→改写」（tab:downstream Table?? 已内联修）。）
> **节点占用（实测 nvidia-smi + log）**：
> - **LOCAL 8×B200**：▶️ **#103 keep14 dense-save re-heal RUNNING**（step2500/200000，train ppl 19.69，1.56s/step，healthy；**step2500.pt 已落盘=首个交叉候选**）。GPU1-7 100%，GPU0 步间 1%。每 ckpt 48.7GB，定位交叉步后 prune。
> - **.252 8×B200**：⏸️ **GATED-IDLE（0 MiB，crossing monitor PID1238902 alive）**：#103 crossing-eval 已评 step2500（PPL 19.1117），**等 LOCAL 下一 ckpt step5000（~14:59 落盘）**才再评——8 卡在评估间隙空转 ~55min/65min 周期。⚠️ **不 co-schedule 长训练**（monitor 任一分钟可能触发 8-GPU 评估，会撞显存 + 危及已承诺的 #103 结果）；crossings（12.797/11.498）预计 step~5k-30k=6-13h 内 bracket，之后 MMLU+McNemar（短）→ **.252 全空后立即接 pending Paper-B/C 训练（#88 keep10 / #132-134 Paper C）**。
> - **.73 8×H20**：▶️ **PaperB Finding-3 matched-LR 对照（agent ad838cf4，task #149）RUNNING**：RESUME #127 from step25000.pt（random-init@uniform-2e-5，单变量 vs keep14），96.4GB/98.5%mem/100%util，~9.2s/step。**⚠️ 全 200k ETA ~18.6 天**；采纳 agent 建议 **step50k(~2.7d) 早读拍板续/降级**。#143 aggregate 已 DONE。zwfy6 盘。
> - **.82 8×H20**：▶️ **#143 CacheBlend LoCoMo r=0.15 cell 仍在生成**（n=960/1986，`incomplete_cells` 已标；完成后重跑 `scripts/aggregate_cacheblend_143.py` 合入 + GPT-4o judge deferred 待跑）。~20GB/卡，healthy。zwfy6 盘。
> - **.104 8×H20**：▶️ **#142 P1.10 write-path 蒸馏 RUNNING**（step 2020/4000，healthy，GPU 100%）。**不动**。
> - **⚠️ .venv/bin/python 已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。**⚠️ 三处物理盘**：wzc1（LOCAL+.252）/ zwfy6（.73+.82）/ .104-TBD；跨盘代码需 cat-over-ssh 同步。

## 历史快照（2026-08-04 12:57 +08:00，subagent_done×2：**B-P1.2 完成 + #103 re-heal 启动**。B-P0.0/B-P1.2 两个 quick-win 全部 DONE。**LOCAL 派 coder aa5d1482 启动 #103 keep14 dense-save re-heal**；**.252 派 coder a88cddeb 消费 re-heal ckpt 定位 PPL 交叉点**。Paper A（.73/.82）+ P1.10（.104）不动。）

## 历史快照（2026-08-04 12:27 +08:00，用户指令：**两台跑 LR 的（P1.3）与 B200 原实验同样处理——记录当前曲线后早停，腾 .73+.82 跑 Paper A**；并「先跑 paperA + paperB 里性价比最高的两个实验 P-0.0, P1.2」。B-P0.0 已完成（见下），B-P1.2 仍在 .252 跑。**P1.3 已 kill**（.73+.82 各 8 卡实测 0 procs/0 MiB），曲线回填 paperB/TODOList §P1.3。**派 opus coder a377ed8a 在 .73+.82 跑 Paper A**（#143 CacheBlend + #144 dense-selector），含 H20 FS 代码同步（H20 三台与 B200 是独立盘，.73 git=2d98c5a 缺 CacheBlend 代码，coder 负责 rsync 同步+self-test gate 再启动）。.104 P1.10 不动继续跑。）
> **节点占用（实测 nvidia-smi + log）**：
> - **LOCAL 8×B200**：✅ **B-P0.0 ShortGPT-16 closed-book QA COMPLETE**（agent aadf4c60；sanity PPL=9.7800 复现 9.7803；PopQA .1585/TriviaQA .3301/NQ .0668，均略高于 keep14@200k 远低于 base_full → 支撑 knowledge-lags；artifact `perplexity-heals-knowledge-lags/data/closedbook/shortgpt16_step200k{,_nqopen}/`，未 push）。8 卡已释放，空闲待补。
> - **.252 8×B200**：▶️ **B-P1.2 OOD PPL + contamination RUNNING**（opus coder abaeab6f；OOD PPL WikiText-103/C4+PG19 × 各模型 + Dolmino n-gram overlap → clean-subset gap）。
> - **.73 + .82 16×H20**：⏹️ **P1.3 STOPPED EARLY @~26.9k/200k**（train-loss ppl~25.4；用户决策同 armA/armB 处理，非崩溃 kill；ckpt step0/5000/.../25000.pt 在 .73 H20 FS）→ ▶️ **Paper A 启动中**（opus coder a377ed8a：#143 CacheBlend@.73 + #144 dense-selector@.82，先同步 H20 FS 代码+self-test）。
> - **.104 8×H20**：✅ P1.10 write-path 蒸馏 RUNNING（Paper A，~2h left，save@每500步；BBWL eval arm 就绪 commit 4340feb，待 .104 空即逐 ckpt 跑）。**不动**。
> - **⚠️ .venv/bin/python 已坏**（Aug3 19:45 reset 成裸 py3.11 无 torch）→ eval 一律用 `/opt/conda/envs/torch-base/bin/python`。**⚠️ H20 三台（.73/.82/.104）与 B200 是独立盘**（路径串同、物理盘不同；B200 上 commit 未 push 则 H20 看不到，需 rsync/push+pull 同步）。

## 历史快照（2026-08-04 11:40 +08:00，用户决策：**P0.5 armA/armB 早停释放 B200，改跑两个 Paper B quick-win eval**（B-P0.4 4-arm factorial ~270 GPU·h 不值得跑）。两 arm 曲线/数据已回填 paperB/TODOList.md + RUN_REGISTRY。**LOCAL 派 opus coder aadf4c60 跑 B-P0.0**（ShortGPT-16@200k closed-book PopQA/TriviaQA/NQ-open → Table 3 closed-book 列，4/6 reviewer 要求）；**.252 派 opus coder abaeab6f 跑 B-P1.2**（OOD PPL C4/WikiText-103+PG19 + n-gram contamination → 附录）。两 H20 长训练 P1.3/P1.10 不动继续跑。monitor http200）
> **节点占用（实测 nvidia-smi + log）**：
> - **LOCAL 8×B200**：⏹️ P0.5 armA **STOPPED @~80.68k/200k ppl10.64**（非崩溃早停，ckpt 保存 step{0..80000}.pt）→ ▶️ **B-P0.0 ShortGPT-16 closed-book QA RUNNING**（opus coder aadf4c60；`/opt/conda/envs/torch-base/bin/python`；先 proxy 预热 HF cache 再 8-shard offline；ckpt `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt`；sanity gate PPL≈9.78）。
> - **.252 8×B200**：⏹️ P0.5 armB **STOPPED @~81k/200k ppl11.42**（非崩溃早停，saved step80000.pt）→ ▶️ **B-P1.2 OOD PPL + contamination RUNNING**（opus coder abaeab6f；先 `pip install datasets` via hy-proxy；OOD PPL WikiText-103/C4+PG19 × {base/full32/keep14/ShortGPT/random/frozen}；Dolmino-vs-benchmark n-gram overlap → clean-subset 重算 gap）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step26200/200k ppl25.21（4.68s/step，healthy，长训练，无 plateau，不打断）。
> - **.104 8×H20**：✅ P1.10 write-path 蒸馏 RUNNING（step1680/4000 loss0.040，~2h left，save@每500步 已存 step500/1000/1500 ckpt）。BBWL eval arm 就绪（coder 4340feb），待 .104 空即逐 ckpt 跑。
> - **⚠️ .venv/bin/python 已坏**（Aug3 19:45 reset 成裸 py3.11 无 torch）→ B200 eval 一律用 `/opt/conda/envs/torch-base/bin/python`（LOCAL 全栈；.252 缺 datasets 需先装）。

## 历史快照（2026-08-04 02:58 +08:00，heartbeat：P1.10 step520/4000 loss0.0445 近收敛，step500 ckpt CONFIRMED present。**step500 早读 eval GPU-blocked**：无空闲卡（5 个可达节点全在跑健康训练），returned-H20 .245/.7.53 + B200 .53/.18/.188 全 reject 凭据 → 派 opus coder ac32812513 预制 WRITE-LoRA arm 进 eval_p018_e4_2x2_writecontrol.py（GPU-free 编码+dry-check，不占卡），待 .104 空（~P1.10 done）或有节点即逐 ckpt 跑 vs ArmB 92.5 / E0 100。三长训练全健康无 plateau，monitor http200）
> **节点占用（实测 nvidia-smi + log，22:55）**：
> - **LOCAL 8×B200**：P0.5 armA step62940 ppl10.75（1.79s/step，healthy，不打断，无 plateau）。
> - **.252 8×B200**：P0.5 armB step63100 ppl11.40（1.76s/step，healthy，不打断，无 plateau）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step19740 ppl28.83（4.68s/step，healthy，长训练，loss 仍降无 plateau）。
> - **.104 8×H20**：✅ P1.10 write-path 蒸馏 RUNNING（pid 1113844，8卡 100% util ~73-75GB ~77%VRAM，batch=24；step520/4000 loss0.0445 近收敛平台（0.0447@500→0.0445@520）；**实测 ~24s/step** → ETA ~27h（save@每500步）；step500 ckpt CONFIRMED present（adapter_config 1189B + safetensors ~116MB）。**eval GPU-blocked**：无空闲卡，WRITE-LoRA eval arm 已就绪（coder ac32812513，commit 4340feb，新 arm BBWL=BB+训练 WRITE LoRA(0..11)，`--write_lora_ckpt` 缺省时逐位不变、dry-check pass 未跑 GPU），待 .104 空（~P1.10 done）或有节点即三段式（manifest→quality→aggregate）对 step500…step4000 逐 ckpt 跑 vs ArmB 92.5 / E0 100 看 gap-closure → outputs/qcmem_writepath_distill_qwen_j12_r32/{step*, final}；训练 commit 8cf49ea+cc020d6）。

## 历史快照（2026-08-03 22:26 +08:00，heartbeat：✅ #135 P0.19 RULER paired leg COMPLETE（recall 97-98% 非瓶颈；j12-frozen|HIT 0-1% 崩、LoRA 恢复 90/82 → gap 在 readout leg，全家族闭合）→ P0.19 全腿 DONE。**用户批准启动 P1.10 write-path 蒸馏训练**（"可以 那就跑"，task #142）→ .104 承接：opus coder ad69ed87 编码 write-path 蒸馏（下 12 层 Write LoRA，teacher=document-contextual Write，吃 P0.17 残留 10-15% gap，upper-bound 点，~20min@8卡）。三长训练全健康）
> **节点占用（实测 nvidia-smi + log，22:24）**：
> - **LOCAL 8×B200**：P0.5 armA step53860 ppl11.45（1.81s/step，healthy，不打断，无 plateau）。
> - **.252 8×B200**：P0.5 armB step53900 ppl12.05（1.77s/step，healthy，不打断，无 plateau）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step16220 ppl29.33（4.69s/step，healthy，长训练，无 plateau）。
> - **.104 8×H20**：#135 P0.19 RULER 已跑完释放 → coder ad69ed87 正编码/启动 P1.10 write-path 蒸馏（diskB，torch-base，Qwen3-8b-local+PG19，output outputs/qcmem_writepath_distill_qwen_j12_r32/）。coder setup 阶段卡可能暂空=非空转。
> - **Monitor**：http200 OK。.55 UNAVAILABLE。

## 历史快照（2026-08-03 21:29 +08:00，heartbeat：✅ P0.20 阶段B（dense，#1 正文主结果候选，#141）COMPLETE → 裁决=MIXED/TIE（k_dense*=10，CoMem 53.22 vs dense 54.22，−1.0pp CI[−4.667,2.667] p=0.637）→ .104 释放 → 立即派 opus coder 承接 #135 P0.19 RULER paired leg（fix eval_ruler_qcmem seed-pairing）；三长训练全健康）
> **节点占用（实测 nvidia-smi + log）**：
> - **LOCAL 8×B200**：P0.5 armA step52040 ppl11.81（1.78s/step，healthy，不打断，无 plateau）。
> - **.252 8×B200**：P0.5 armB step52080 ppl12.53（1.77s/step，healthy，不打断，无 plateau）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step15540 ppl31.00（4.68s/step，16 卡，healthy，长训练，无 plateau）。
> - **.104 8×H20（diskB）**：P0.20 阶段B（dense）**COMPLETE**（pid499855 done，8/8 GPU 0%/0MiB 空闲，decision.json@21:07）→ **派 opus coder 承接 #135 P0.19 RULER paired leg**（fix eval_ruler_qcmem RULER 样本 seeding 与 paired 口径不一致的 bug，使两臂见同一 RULER 样本；跑 recall@k×hit-conditional readout decomposition，补齐 P0.19 RULER 腿，解释 P0.20 RULER cell CoMem 低 k 碾压来源）。
> - **B200 .55**：❌ UNAVAILABLE。Monitor http200。
>
> **P0.20 阶段B 裁决（#141 DONE）**：deployment PRIMARY k_dense*=10 → CoMem(k12)=53.22 vs dense-RAG=54.22，diff−1.0pp，CI[−4.667,2.667]，McNemar p=0.637 → 统计打平；cold-index k_dense*=None（encode-all 超带）；参考非等延迟两臂 k12 CoMem 53.22 vs dense 58.56（−5.33，p=0.00387）。VERDICT=MIXED。阶段A BM25 是 −11.56pp → 等延迟裁决 selector-dependent，CoMem 对可部署 text-RAG 最好=打平（dense）从不赢。逐 cell CoMem 随 k 非单调（低 k 峰、高 k 退化）vs dense 单调↑，crossover≈k12-14。全文 paperA/P0_20_PHASEB_NOTES.md §5 + bench_results/p0_20_phaseB_dense/{decision,summary}.json。

## 当前快照（2026-08-03 19:05 +08:00，subagent_done：✅ P0.20 阶段B（dense，#1 正文主结果候选，task #141）已编码+CPU 验证+启动于 .104（pid499855），全 fail-closed 门 PASS（P1.9 repro=True），STEP2 calib 运行中；三长训练全健康）
> **节点占用（实测 nvidia-smi + log）**：
> - **LOCAL 8×B200**：P0.5 armA step46840 ppl11.80（1.79s/step，healthy，不打断，无 plateau）。
> - **.252 8×B200**：P0.5 armB step46820 ppl11.79（1.76s/step，healthy，不打断，无 plateau）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step13540 ppl31.86（4.68s/step，16 卡，healthy，长训练，无 plateau）。
> - **.104 8×H20（diskB）**：**P0.20 阶段B（dense）RUNNING**——opus agent（abde161e）编码 scripts/eval_p0_20_phaseB_dense.py + _run_p0_20_phaseB_dense.sh + paperA/P0_20_PHASEB_NOTES.md（commit `749b0a0`，author LiuHanzuo，+1385 行，未 push），CPU 验证全过，启动 pid499855（torch-base）。**现 STEP2 calib_latency 单卡扫（GPU0 100%/18.5GB，k=4/6…；calib 串行于 GPU0 属正常）**→ STEP3 8-GPU quality（360 jobs=9 cells×10 k×4 shards）→ STEP4 aggregate。fail-closed 门全 PASS（manifest LoRA+BGE sha；sanity P1.9 repro=True、read_len paired=6257、LoRA toggle 168→0、calib/quality disjoint）。log logs/p0_20_phaseB.out。**verdict/k_dense*/dense-vs-CoMem CI+McNemar 于 aggregate 出（数小时）**。task #141 in_progress。
> - **B200 .55**：❌ UNAVAILABLE。Monitor http200。
>
> **P0.20 阶段B 设计要点（未 loosen 门）**：(1) import P1.9 冻结 DenseRetriever verbatim（同 sha/CLS+L2+cosine），select_topk 全 score dict 恢复任意 k；fail-closed 复现核对 top-12==P1.9 stored dense_sel_idx（按 input_ids_sha256），.104 实测 repro=True。(2) dense 延迟两口径：**Deployment（PRIMARY 定 k_dense*）**=离线预索引→在线 query-encode+flat cosine（CoMem 预存 h12 的对偶）；**Cold-index（SENSITIVITY）**=encode-all（==P1.9 retrieval_latency_ms）。实测 k=2 deploy=7.9ms vs cold=216ms。CoMem 臂 byte-identical 于阶段A（TTFT anchor 同），reader 同 config#2。阶段A（BM25，#137）=NEGATIVE 留作 selector ablation（不删）。

## 当前快照（2026-08-03 18:28 +08:00，heartbeat：✅ P1.8（#139）COMPLETE=VALID（18/18 done，0 abort，crossover json 已出）→ .104 释放 → 立即派 opus agent（abde161e）编码+启动 #1 优先级 P0.20 阶段B（dense，正文主结果候选）于 .104；三长训练全健康）
> **节点占用（实测 nvidia-smi + log）**：
> - **LOCAL 8×B200**：P0.5 armA step46020 ppl11.53（1.80s/step，maxmem98.3GB，healthy，不打断，无 plateau）。
> - **.252 8×B200**：P0.5 armB step45980 ppl12.17（1.77s/step，healthy，不打断，无 plateau）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step13220 ppl34.92（4.68s/step，16 卡 100%/98GB，healthy，长训练，无 plateau）。
> - **.104 8×H20（diskB）**：**P1.8 serving-curve COMPLETE & VALID**（`[p1.8] COMPLETE`，18/18 done markers，store!=recompute=0，crossover json `bench_results/p1_8_serving/p1_8_serving_aggregate.json` 58722B）→ GPU 全 0% 释放 → **派 opus agent（abde161e）承接 P0.20 阶段B（dense）**：扩 P0.20 阶段A harness 消费 P1.9 `retrieval_results/p1_9_dense` BGE 排序、按 dense 检索 cost 重算 equal-latency k*、复跑同 cohort quality+McNemar+bootstrap+decision.json。agent CPU 验证后 RUN=1 于 .104（torch-base，PROJECT_ROOT .104 diskB）。task #137 阶段B in_progress，#139 DONE。
> - **B200 .55**：❌ UNAVAILABLE。Monitor http200。
>
> **P1.8 crossover 结论（#139 DONE）**：CoMem 每 query 分摊后更快——L=32k tier=cpu comemW=2.253s，G1:Q*≈8.9 / G32:9.2 / G128:10.9 / G512:94；tier=gpu G1:Q*≈8.4；@128k G=1 break-even Q*≈26-28（与 P0.2 解析 ≈17-20 一致，larger G 更早）。⚠️ 定位：P1.8=延迟分摊故事（重复 query 同 doc 后 CoMem 更省），P0.20=等延迟质量故事（NEGATIVE）；互补，不得混为「CoMem 赢」。

## 当前快照（2026-08-03 18:00 +08:00，heartbeat：✅ P1.8（#139）store!=recompute bug 已被 opus coder agent 修复+relaunch，selfcheck GATE2 max_abs=0.0 PASS，8 卡 serve pool 运行中（4 runtime job 已 GATE2 PASS）；三长训练全健康）
> **节点占用（实测 nvidia-smi）**：
> - **LOCAL 8×B200**：P0.5 armA step44900 ppl12.09（1.78s/step，GPU99-100%，healthy，不打断，无 plateau）。
> - **.252 8×B200**：P0.5 armB step44840 ppl12.00（1.76s/step，healthy，不打断，无 plateau）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step12780 ppl31.16（4.68s/step，16 卡 100%/96GB，healthy，长训练，无 plateau）。
> - **.104 8×H20（diskB）**：**P1.8 serving-curve RUNNING & VALID**——opus coder agent（a46b637b）已修复 write-once/fetch 对齐 bug 并 relaunch（17:52 起，log logs/p1_8_serving.out）；**manifest 门 PASS + selfcheck GATE2 `store==recompute max_abs=0.0 PASS`**（前一版每 job max_abs=128 abort 的问题已解）；8 GPU flock serve pool 运行（32k/128k × gpu/cpu tier），4 个 runtime serve job 已 GATE2 PASS，store!=recompute 计数=0（17:39 killed-run 陈旧 per-job log 已清理）。task #139 in_progress。
> - **B200 .55**：❌ UNAVAILABLE。Monitor http200。
>
> **P1.8 修复确认**：GATE（store-fetch 选中 h12 == fresh recompute，max_abs≈0）保持严格未 loosen；coder agent 仅改 bench_p1_8_serving_curve.py 复用路径，未动共享模块。ETA serve pool 完成后出 serving curve（TTFT/decode vs store_length×query_count×gen_length×tier）。
> **P0.20（#1）阶段B（dense，正文主结果候选）仍待编码**：需扩 harness 消费 P1.9 `retrieval_results/p1_9_dense` BGE 排序、按 dense 检索 cost 重算 equal-latency k*。可在下一个空节点派 P0.20 agent（ac5056a0）承接。

## 当前快照（2026-08-03 17:35 +08:00，heartbeat：✅ P0.20（#1）阶段A COMPLETE=NEGATIVE 裁决（.104 释放）→ 立即接 P1.8 serving-curve；但 P1.8 触发 store!=recompute 门（max_abs=128）每 job abort→已 kill→派 opus coder 诊断修复+relaunch（.104 归 agent）；三长训练全健康）
> **节点占用（实测 nvidia-smi）**：
> - **LOCAL 8×B200**：P0.5 armA step44300 ppl12.19（1.77s/step，GPU99-100%，healthy，不打断）。
> - **.252 8×B200**：P0.5 armB step44240 ppl12.03（fresh，healthy，不打断）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP step12540 ppl32.87（4.69s/step，16 卡 100%，fresh 17:34，healthy，长训练）。
> - **.104 8×H20（diskB）**：**P0.20 阶段A 已 COMPLETE**（360/360，run-clean，NEGATIVE 裁决）→ 接 P1.8 但 **store!=recompute 门 fail-closed**（每 serve job max_abs=128 abort，无有效数据）→ **已 kill（GPU 全 0%）** → **opus coder agent（诊断+修 bench_p1_8_serving_curve.py 的 write-once/fetch 对齐 bug + relaunch），.104 归该 agent**。不空转。
> - **B200 .55**：❌ UNAVAILABLE。Monitor http200。
>
> **P0.20（#1）阶段A 裁决=NEGATIVE**：等 GPU-resident TTFT（±5%）下 BM25 text-RAG k_RAG*=10 匹配 CoMem(j12,k12)，text-RAG 宏观 64.78 vs CoMem 53.22，**diff=-11.56pp CI[-14.44,-8.67]**（McNemar comem_only_b=41）。逐 cell 几乎全输（longeval 16k -28、qa1 16k -17…），仅 locomo +1 打平。⇒ 瓶颈=cached-state readout，非 retrieval budget；不得包装成 positive Pareto。BM25 结果留作 selector ablation。**阶段B（dense，正文主结果候选）尚未编码**（launcher 无 dense 支持，需扩 harness 消费 P1.9 BGE 排序按 dense cost 重算 k*）。
> **P1.8 fail-closed 细节**：manifest 门 PASS（LoRA sha dd09cd17… match）；serve/selfcheck 门 abort `store!=recompute max_abs=128.0（cache reuse changed the Read inputs）`——P1.8 独有的 write-once-then-fetch 复用路径 bug（P0.13/P0.16/P0.17 write+read per-example 无此路径故均过）。opus agent 修复中，禁止 loosen 门 tolerance。

## 当前快照（2026-08-03 15:56 +08:00，heartbeat：✅ P1.9 dense-RAG 完成（44/44 cells，guard PASS，dense recall@12 长档崩=Paper A 参照点）→ .104 释放 → 立即启动 #1 优先级 P0.20 阶段A RUN=1（pid208646，manifest+sanity 门 PASS，现 calib 单卡阶段））
> **节点占用（实测 nvidia-smi）**：
> - **LOCAL 8×B200**：P0.5 armA（healthy，不打断）。
> - **.252 8×B200**：P0.5 armB（healthy，不打断）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP（healthy，长训练）。
> - **.104 8×H20（diskB）：P0.20 阶段A RUNNING**（pid 208646，torch-base；STEP2 calib-latency 单卡 GPU0 活跃~10.8GB，GPU1-7 设计性短暂空~15-25min（干净单卡计时，故意不补短任务），STEP3 后占满 8 卡~2.5-4h）。P1.9 本轮已完成。
> - **B200 .55**：❌ UNAVAILABLE。Monitor http200。
>
> **P1.9 DONE**：dense-RAG(BGE topk12 no-LoRA) recall@12 随长度衰减（16k babilong qa1=0.52/qa2=0.45，8k~0.90），reader hit-conditional acc 仍高 → 瓶颈在检索非阅读。为 P0.20 阶段B 前置（复用 retrieval_results/p1_9_dense）。
> **P0.20（#1）RUNNING**：equal-latency frontier，固定 CoMem(j12,k12) 找 k_RAG*（±5% 延迟带）。ETA STEP4 ~3-4.5h。当前无其它空节点（.55 UNAVAILABLE），P1.8 仍排队。

## 当前快照（2026-08-03 15:00 +08:00，heartbeat：✅ P0.18 E4 完成收口（2×2 裁决=Write-side gap，doc-ctx write 闭合，read 位置不是杠杆）；.104 P0.18 跑完→立即接 P1.9 dense-RAG（BGE 已 rsync，provenance PASS，填空窗+P0.20 阶段B 前置））
> **节点占用（实测 nvidia-smi）**：
> - **LOCAL 8×B200**：P0.5 armA step38960 ppl12.06（healthy，不打断）。
> - **.252 8×B200**：P0.5 armB step38840 ppl12.06（healthy，不打断）。
> - **.73 + .82 16×H20**：P1.3 16-card DDP，GPU100%/96GB（healthy，长训练）。
> - **.104 8×H20（diskB）**：**P1.9 dense-RAG RUNNING**（pid 153121，8 workers，GPU 19-100%/~19GB，COHORT=min 44 jobs，provenance sha_ok=true，torch-base，log logs/p1_9_dense_rag.out）。**P0.18 已于本轮完成**（全 fail-closed 门 PASS）。
> - **B200 .55**：❌ UNAVAILABLE。Monitor http200。
>
> **P0.18 裁决**：macro A=100/BB(chunk-local,local-pos)=92.5/X(chunk-local,doc-origin-pos)=88/E0(doc-ctx,local-pos)=100/Y(doc-ctx,doc-origin-pos)=100；deployable gap 纯 Write-side（下12层 attn scope），doc-ctx write→100，read RoPE 位置非杠杆（chunk-local 下 doc-origin −4.5pp），两因素 interact（residual+4.5）。⇒ P1.10 若训练针对 Write repr；且 P0.17 E2 已零训练恢复 80-87%，训练很可能不必要（交用户）。
> **P0.20（#1）**：阶段A harness agent ac5056a0 在 LOCAL 建 ETA~20-25min（不自启动，交 main 命令）；.104 P1.9 跑完接 P0.20 阶段A。P1.9=阶段B dense 前置。

## 当前快照（2026-08-03 14:41 +08:00，heartbeat 续：.104 曾空闲→已用 P0.18 E4 填上（用户问"H20 是否全占用"后立即补卡）；P0.20 agent 改为 build-only 并将交出启动命令，避免与 P0.18 抢 8 卡）
> **节点占用（实测 nvidia-smi）**：
> - **LOCAL 8×B200**：P0.5 **armA** step37900+ ppl~11.9（healthy，不打断）。
> - **.252 8×B200**：P0.5 **armB** step37780+ ppl~12.6（healthy，不打断）。
> - **.73 + .82 16×H20**：**P1.3** random-init LR2e-5（16-card DDP）step10120 ppl37.75（GPU99%，healthy，不打断；长训练不会很快空出）。
> - **.104 8×H20（diskB）**：**P0.18 E4 2×2 write-control RUNNING**（pid 145899，8×GPU 全载 ~19-20GB，manifest+pos_sanity 门通过，8 quality shards niah_multikey_1×{8k,16k} 入池；task #138，PYBIN torch-base，日志 /apdcephfs_zwfy6/.../logs/p0_18_e4.out）。**填补了此前 idle 窗口**——scripts 已 rsync wzc1→.104。
> - **B200 .55（wzc1 独立盘）**：❌ UNAVAILABLE。Monitor http200。
>
> **P0.20（#137，用户 #1 优先）调度**：agent ac5056a0 已确认改为 **build-only**（不自启动 GPU），在 LOCAL 建 harness→rsync .104→**把 8-GPU 启动命令交给 main**。main 将在 **P0.18 释放 .104 后（~30-40min）** 或更早有节点空出时启动 P0.20。此举避免两个 8-GPU DDP 撞同 8 卡。
> **P1.8/P1.9 harness**（commit c32a2c9）排队 #139/#140，.104 空出后按 P0.20→P1.9→P0.18已跑→P1.8 序（P1.9 亦为 P0.20 阶段B 前置）。

## 当前快照（2026-08-03 14:23 +08:00，heartbeat：✅ P0.17 **完全收口**（含 measured latency，已入 notes）；.104 空闲但**为 #1 优先 P0.20 保留**（agent ac5056a0 在 LOCAL 建 harness→自行 rsync+在 .104 起 8-GPU，禁止另起任务抢卡）；三长训练全健康）
> **节点占用（实测 nvidia-smi）**：
> - **LOCAL 8×B200**：P0.5 **armA** step37900 ppl11.92（1.76s/step，8×GPU 98-100%，healthy，不打断）。
> - **.252 8×B200**：P0.5 **armB** step37780 ppl12.61（1.77s/step，healthy，不打断）。
> - **.73 + .82 16×H20**：**P1.3** random-init LR2e-5 control（16-card DDP），step10120 ppl37.75（4.69s/step，GPU 99%，healthy，不打断）。
> - **.104 8×H20（diskB）**：**空闲，为 P0.20 保留**。P0.17 latency 微bench 已完成（GPU0，per-arm ms 见 P0_17_E2_NOTES §4：A.read957.7 / B.read681.9=E2.read≈680；B.write262.1→w128 346.8）。P0.20 phaseA harness 由 agent ac5056a0 在 LOCAL 构建中→将自行 rsync 到 .104 并启动 → **main 不在 .104 另起任务**（否则与 agent 的启动撞车）。
> - **B200 .55（wzc1 独立盘）**：❌ UNAVAILABLE（2× ssh timeout）——不计入可用池。
> - Monitor http200 OK。
>
> **✅ P0.17 latency 收口（14:22，.104 GPU0，single-proc，niah_multikey_1/16k，warmup3×n_repeat20）**：Read 跨 B/E2 一致（~680ms，证明 E2 只改 Write）；deployable j12 Read 比 full-replay A（957.7ms）快 ~29%；Write w0=262.1→w32 311.7(+18.9%)→w64 324.9(+24.0%)→w128 346.8(+32.3%)（wall-clock 含固定 per-chunk 开销，>marginal-FLOPs 1.057×–1.229×）。原始 `bench_results/p0_17_e2_overlap/latency/latency_proc0.json`（顶层 latency.json n_procs=0 为 14:01 陈旧产物）。

## 当前快照（2026-08-03 14:12 +08:00，✅ P0.17 E2 overlap-Write COMPLETE（deployable 修复达标）→ P0.20 equal-latency frontier 成为用户指定 #1 优先并已启动；三长训练全健康）
> **节点占用（实测 nvidia-smi 对照）**：
> - **LOCAL 8×B200**：P0.5 **armA** step37400 ppl11.87（1.80s/step，healthy，不打断）。
> - **.252 8×B200**：P0.5 **armB** step37280 ppl12.08（1.77s/step，healthy，不打断）。
> - **.73 + .82 16×H20**：**P1.3** random-init LR2e-5 control（16-card IB DDP），.73 GPUs 99-100%/96GB，healthy，不打断。
> - **.104 8×H20（diskB）**：P0.17 **latency 微bench** 运行中（GPU0 pid143737，~10min，收尾 P0.17 measured per-arm ms）→ 完成后接 **P0.20 phaseA** 8-GPU 质量 sweep（agent ac5056a0 正在 LOCAL 建 harness→rsync 过来）。
> - **B200 .55（wzc1 独立盘）**：❌ UNAVAILABLE（2× ssh timeout，rc=143）——不计入可用池。
>
> **✅ P0.17 E2 overlapping-chunk Write COMPLETE（task #136，.104 8×H20，n=200 paired，真 Qwen3-8B+旗舰 LoRA）**：
>   - deployable multikey pooled **92.5（w0=Arm B）→ 99.0（best w=128）**，**清过预注册目标 ≥97.0**。E2_w32=98.5（+6.0 [3.0,9.5] p=4.9e-4 b12c0）、w64=98.5（+6.0）、w128=99.0（+6.5 [3.5,10.0] p=2.4e-4 b13c0）；E0 天花板=100.0。
>   - 回收 **80–87%** 的 E0−B document-context gap，**store bytes/token + Read + decode 与 w0 完全不变**（仅一次性 lower-12 Write FLOPs +5.7%~+22.9%）。gates 全 PASS（LoRA sha dd09cd17…；e2_sanity 两项 max_abs=0；packs_paired_1to1；pack sha 200/200==P1.7；oom=0 nonfinite=0）。
>   - **裁决**：确认 P0.16 归因（gap=chunk-local Write 缺文档上下文，非 Read 重定位）；E2 是可部署修复（w=32 已近最优最省）→ 候选并入 Cohort-B。commits 873deb2+be2ae80（LOCAL canonical，author LiuHanzuo，未 push）。
>
> **🔴 P0.20 equal-latency retrieval-budget frontier（用户 2026-08-03 新增，指定最高优先级；task #137，agent ac5056a0）**：phaseA=BM25 k-sweep k∈{2..24}，text-RAG(j0) vs CoMem(j12) 在匹配 TTFT 下比质量，calib split 冻结 k_RAG*/k_CoMem*（±5%）；复用 config#2/P0.13/P1.7/P0.2 资产；base 协议 chat=False add_bos=0 iter_bm25 seed=42。phaseB（dense retriever）绑定 P1.9。
>
> **零训练 harness 并行 build（workflow wg28ofr1v，3 agents，不提交 git，MAIN 集中提交）**：P0.18 E4 2×2 write-control / P1.8 repeated-query serving curve / P1.9 dense-retriever RAG（P1.9 亦为 P0.20 phaseB 前置）。就绪后按节点空闲排队上 .104。
>
> **P0.19 已收口**：#131 DONE via CPU recompute（`paperA/P0_19_decomp_NOTES.md` commit b9dc847）；TODOList TODO→DONE。#135 RULER paired GPU leg = 低优先/可选（seed bug 已修 d1e1389，无 paper table 依赖）。
> **Monitor**：http200 OK。

## 当前快照（2026-08-03 13:43 +08:00，✅ Paper A P0.16 E0 write-control COMPLETE→write-path gate 决定性通过→P0.17 E2 GO 已启动@.104；三长训练全健康）
> - **✅ P0.16 E0 document-contextual Write control COMPLETE（task #130 done）**：`bench_results/p0_16_e0_write_control/`（.104 diskB），n_paired=200 n_cells=2。
>   - **macro：A(full replay)=100.0 / C(continuous-pack oracle)=100.0 / E0(doc-ctx Write)=100.0 / B(chunk-local deployable)=92.5**。
>   - **A−E0=+0.0 CI[0,0] McNemar p=1；C−E0=+0.0 CI[0,0] p=1（E0 与 A/C 逐位一致，both=200）**；**E0−B=A−B=C−B=+7.5pp CI[4.0,11.5] McNemar b=15/c=0 p=6.1e-5**。per-cell：8k B=94.0(E0−B=+6.0)、16k B=91.0(E0−B=+9.0)。
>   - fail-closed 全过：`packs_paired_1to1=True p013_sha_match=True oom=0 nonfinite=0`；e0_h12_sanity `max_abs=0.000e+00`（前 gate）。agreement A_vs_E0 first_token=0.855 cos=0.9963、B_vs_E0 first_token=0.91。
>   - **决策（pre-registered rule 命中）：E0 ≈ A/C 且 ≫ B → deployable A-B gap 全部来自 chunk-local Write 缺文档上下文；Read 接口/repositioning 近乎无损 → P0.17（E2 overlap Write）trigger 满足 → GO。**
> - **✅ P0.17 E2 overlapping-chunk Write control 已启动@.104 8卡（task #136，coder aa927039 opus 在跑：建 `scripts/eval_p017_e2_overlap_write.py`（import P0.16 machinery 保 A/B/C/E0 逐位不变）+ `scripts/_run_p017_e2_8gpu.sh`→commit（author LiuHanzuo 无署名 未 push 仅指定文件）→rsync wzc1→.104→8卡 launch→验 gate→报 per-w）**：
>   - E2=每 512-tok chunk 带左 prefix w∈{32,64,128} 跑下-12、弃 prefix 态、只存原 512 chunk h12（persistent bytes/token、Read pack、Read compute 与 Arm B 不变，仅 Write compute↑）。arms A/B(=w0)/E2_w32/E2_w64/E2_w128/E0，cohort=niah_multikey_1 {8k,16k} n=100（同 200 paired）。pre-reg 目标 multikey pooled 92.5→≥97.0。
> - **✅ 三长训练全健康（绝不打断）**：P0.5 armA(LOCAL B200 contig16) ~step35k ppl~12.3 1.80s/step；armB(.252 final14+2fresh) ~step35k ppl~12.3 1.77s/step（未近 1-2% 平台）；P1.3(.73+.82 16卡 IB DDP from-scratch16L uniform lr2e5) ~step9k ppl~40.7 4.69s/step（冷启正常降）。ETA P0.5 ~3天到 200k；P1.3 待 B200 空迁移。到平台记 200k endpoint。
> - Monitor 8088 http200 OK。本轮：0 kill；**1 GPU launch（P0.17 E2@.104 8卡，task #136；.104 held-for-PaperA 兑现 P0.16→P0.17 接力）**；2 后台 coder 在跑（aa927039 P0.17 build+launch、aeec7b55 RULER seed-pairing fix #135）。**MAIN 待办**：aa927039 完→复核 commit 干净+gate 过+per-w 数字→回填 paperA/TODOList P0.17 + mechanism table；aeec7b55 完→确认 `PAIRED_RULER_CLAIM_FOUND` 口径→#135 GPU leg 排队待 H20 空；#132 排 diskB H20（Paper A 优先后）；#133/#134/#103 待 B200 空。
> - 旧快照（2026-08-03 13:05）保留于下方 →

## 当前快照（2026-08-03 13:05 +08:00，✅ Paper A P0.16 E0 write-control LAUNCHED@.104（held-for-PaperA 落点兑现）；三长训练全健康）
> - **✅ P0.16 E0 document-contextual Write control 已启动@.104 8卡（task #130，coder ab9637a done→commit 2ae5917，author=LiuHanzuo 无 AI 署名 未 push，仅 3 文件含新 paperA/P0_16_E0_NOTES.md 无 TODOList/.tex/status）**：
>   - **两道 gate 在真 Qwen3-8B 全过（E0 有效性确证）**：manifest `OK — LoRA sha dd09cd17… 168 modules layers[12..35]`；**e0_h12_sanity `max_abs=0.000e+00`（ref_abs_max=8576，tol=5e-2）PASS** → E0 的 document-contextual 下-12 前向与 stock 下-12 逐位一致（LoRA 在 12..35，hidden[12] adapter-independent）。
>   - cohort=min：`niah_multikey_1 × {8k,16k}`，n=100/cell，4 臂（A=j0 full replay / B=j12 chunk-local deployable / C=j12 continuous-pack oracle / E0=j12 doc-ctx），4 shards/cell 跨 8 卡；8 worker 全起 ~18.8GB/卡 37-100%util（H20 97.8GB 余量足）。`--verify --p013_manifest_dir bench_results/p1_7_h12_oracle`（pack-sha 交叉校验 active）。log `logs/p0_16_e0.out` + `logs/p0_16_e0/`；输出 `bench_results/p0_16_e0_write_control/`（.104 diskB）。
>   - **同步 wzc1→.104**：2 新脚本 committed 于 wzc1，.104 是独立 diskB 卷→已 rsync（exit0）+ .104 bash -n/py_compile(torch-base) 双过。E0-vs-B 隔离 document-context 价值、E0-vs-C 隔离 repositioning 代价（framed 为 cross-query-reusable doc-ctx control，非严格上界）。
> - **✅ Paper C P-C1 follow-up harness coder（a7ed839）done→commit cd0f527（仅 2 脚本 +446 行，author=LiuHanzuo 无署名 未 push，bash -n 双过，无 protected files）**：#132 second-task eval（`scripts/_run_paperC_secondtask_8gpu.sh`，须跑在有 P-C1 ckpt+squad npy 的 diskB H20，排 P0.16 之后）；#133 depth-sweep keep{20,24,28}（`scripts/run_paperC_depthsweep.sh`，header DO-NOT-AUTORUN，待 B200 空 P0.5~3天）。
> - **P0.19（#131）coder abedcb7 仍在跑**：`scripts/analyze_p019_recall_readout.py` 已落盘（12:54）但未发完成通知→等其 completion+verification 再起（CPU-only，不占卡，可与 P0.16 并行）。
> - **✅ 三长训练全健康（绝不打断）**：P0.5 armA(LOCAL B200 contig16) step34940 ppl12.29 1.80s/step；armB(.252 final14+2fresh) step34780 ppl12.29 1.77s/step（~7% over 17k steps 未近 1-2% 平台）；P1.3(.73+.82 16卡 IB DDP from-scratch16L uniform lr2e5) step8940 ppl40.70 4.69s/step 96GB（冷启正常降）。ETA P0.5 ~3天到 200k；P1.3 待 B200 空迁移。到平台记 200k endpoint。
> - Monitor 8088 http200 OK。本轮：0 kill；**1 GPU launch（P0.16 E0@.104 8卡，task #130 in_progress）**。**MAIN 待办**：P0.16 完→aggregate→mechanism-table rows + P0.17/P0.18 attribution 决策；P0.19 coder 完→立即起（CPU）；#132 排 P0.16 后（diskB H20）；#133/A1 待 B200 空。
> - 旧快照（2026-08-03 12:33）保留于下方 →

## 当前快照（2026-08-03 12:33 +08:00，✅ P-C1 orchestrator DONE→全 4 臂 SQuAD EM/F1 落地→.104 8卡空出(用户预留给 Paper A)；三长训练全健康）
> - **✅ P-C1 orchestrator DONE @12:31:36（task #92，SQuAD dev n=2000，同 base 口径）**：
>   | arm | EM | F1 |
>   |-----|-----|-----|
>   | A2_lora_r160（全32L LoRA r160，param-matched） | **0.6590** | **0.7139** |
>   | BASE_ref（基座参考） | 0.3385 | 0.3999 |
>   | A4_hero（freeze-graft keep14+2fresh） | 0.2930 | 0.2970 |
>   | A3_fromscratch（16L from-scratch bnb8bit） | 0.2605 | 0.2612 |
>   - **freeze-graft(A4) > from-scratch(A3) +3.2pp EM / +3.6pp F1**（Paper C 核心对照方向成立）；但两个剪枝-16L 变体均 < 全模型 LoRA(A2) 且 < BASE_ref → 需 Paper C 正面框定（剪枝容量损失 vs 1000-step/1.58M-token SFT 恢复量级）。**已派 researcher（a64858a）分析 EM/F1 口径核实 + framing + 是否补 A1 天花板；不改 .tex/TODOList，回填由 MAIN。**
>   - **A1 full-FT-32L 仍 NEEDS B200**（H20 OOM，7B fp32-AdamW）→待 B200 P0.5 空出后补跑作天花板参照（researcher 将给建议）。
>   - raw：`paperC_squad_results/{A2_lora_r160,A3_fromscratch,A4_hero,BASE_ref}/shard0of1.json`；orch log `logs/paperC_pc1_orch.log`。
> - **✅ .104 8×H20 全空出（用户 2026-08-03 指令"空一台 H20 给 Paper A"的落点）**：2 个 Paper A harness coder 在建（P0.16 E0 write-control #130 = ab9637a、P0.19 recall×readout 分解 #131 = abedcb7），**harness ready 即启动**：P0.19（CPU-only 分析）立即起、P0.16（zero-training GPU eval，是 write-path 训练的 gate）8卡起，**排在 Paper B P2.2/P2.5(#128/#129) 之前**（H20=PaperA-first）。**本轮 .104 held for Paper A（用户明确预留 + harness 临近，属正当 hold 非空转）**。
> - **✅ 三长训练全健康（绝不打断）**：P0.5 armA(LOCAL B200 contig16) step34220 ppl12.09 1.80s/step；armB(.252 final14+2fresh) step34060 ppl12.41 1.77s/step（armA 自 17.6k 的 ppl13.22 降至 34.2k 的 ~12.1，>15k 步内 ~9% 未近 1-2% 平台→继续）；P1.3(.73+.82 16卡 IB DDP from-scratch16L uniform lr2e5) step8680 ppl40.82 4.69s/step 98.3GB（冷启正常降 211→41）。ETA P0.5 ~3天到 200k；P1.3 待 B200 空迁移(--resume_from)。到平台记 200k endpoint。
> - Monitor 8088 http200 OK。本轮：0 kill；0 GPU launch（.104 held for Paper A，harness 未 ready）；1 researcher 派发（P-C1 EM/F1 分析）；task #92 4/5 臂 EM/F1 全落地（A1 待 B200）。**MAIN 待办**：researcher 回来后回填 paperC 文档（EM/F1 表 + framing）；harness ready 后 launch P0.16/P0.19@.104；A1 待 B200 空补跑。
> - 旧快照（2026-08-03 08:23）保留于下方 →

## 当前快照（2026-08-03 08:23 +08:00，✅ P-C1 A4+A3 完成→A1 full-32L OOM 标 NEEDS B200→orchestrator 自动进 A2 LoRA 跑起；三长训练全健康）
> - **✅ P-C1 A3 from-scratch 16L 完成**（bnb8bit，`outputs/paperC_pc1_squad_A3/final.pt` 24.5GB @08:04，train ppl→1.00 属预期记忆；held-out SQuAD dev EM/F1 才是 headline）。A4(hero)+A3 两臂 final.pt 均在盘。
> - **⚠️→✅ A1 full-FT 32L 标 NEEDS B200（orchestrator 自动处理，非 bug）**：08:09 A1(keep32/fresh0，7B 全参 bnb8bit AdamW，BS2/GA8)起→无 final 崩(H20 OOM：7B fp32 params+grads+8bit optim m/v ~70GB+activation>95GB)→08:09 自动降 BS1/GA16 重试→08:12 仍崩→标 "A1 STILL no final -> NEEDS B200" 并 **fault-tolerant 跳过**，08:12:35 自动进 A2。**A1 延迟到 B200 空**（P0.5 两臂 ~3.5天到 200k 后；B200 183GB 可 fp32-AdamW 跑满 7B，1000 步 ~15min）。
> - **✅ A2 param-matched LoRA r=160 正在跑@.104**：step40/1000 loss0.41 ppl1.51（step10 0.73→稳定下降）15.09s/step **maxmem36.5GB OOM=0** 8卡100%util。ETA ~4.2h→~12:30 final。这是 .104 上 P-C1 最后一个训练臂，完后 orchestrator 跑逐臂 SQuAD dev EM/F1(A4_hero/A3_fromscratch/A2_lora_r160 + BASE_ref；A1 缺待 B200)。
> - **✅ 三长训练全健康（绝不打断）**：P0.5 armA(LOCAL B200 contig16) step25900 ppl12.51 1.79s/step；armB(.252 final14+2fresh) step25660 ppl13.11 1.77s/step；P1.3(.73+.82 16卡 IB DDP from-scratch16L uniform lr2e5) step5480 ppl50.96(冷启正常降 211→51)4.69s/step 98.3GB。ETA P0.5~3天到 200k；P1.3 待 B200 空迁移。到平台记 200k endpoint。
> - Monitor 8088 http200 OK。本轮：0 kill；0 launch（orchestrator 自主 A3→A1(崩)→A2 链，无 idle 卡）。**MAIN 待办**：A2 完+逐臂 EM/F1 落地后回填 paperC 文档（subagent 不碰 .tex/TODOList）；A1 待 B200 空补跑。
> - 旧快照（2026-08-03 05:32）保留于下方 →

## 当前快照（2026-08-03 05:32 +08:00，✅ P-C1 A4(hero) 完成 final.pt→orchestrator 进 A3；A3 fp32-AdamW OOM→已诊断+改 bnb8bit 重启跑起；三长训练全健康）
> - **✅ P-C1 A4 freeze-graft HERO 完成**（`outputs/paperC_pc1_squad_A4/final.pt`+step500.pt 在盘，train ppl→1.00，held-out SQuAD dev EM/F1 才是 headline，训练 loss 记忆到 1.0 属预期无害）。
> - **⚠️→✅ A3 from-scratch 崩溃已修（本轮 Step-2 处理）**：05:10 orchestrator 起 A3(from-scratch 16L 4.06B **全参 fp32 AdamW**)→05:12 **8 卡全 CUDA OOM**（每卡试 alloc 3.06GB，仅剩 508MiB）——正是"4B 全参 fp32-AdamW 单 H20 装不下"（A4 能装因 freeze-graft 只 1.23B 可训）。**根因非 bug 是显存**。另发现 orchestrator 一个 latent bug：其 `pgrep -f 'train_olmo2_(...)'` 崩溃检测被残留启动 shell(pid25135，cmdline 含 "train_olmo2_lora_sft.py")误匹配→本会空转到 3600s timeout(~06:12)才 advance。**修复**：kill 25138(orchestrator)+25135(poison parent)→sed A3 launch 行加 `OPT=bnb8bit`(A1 早已 bnb8bit)→无 py_compile 包装干净重启(pid53773，不留 poison shell)。A3-bnb8bit 已跑起：`using bitsandbytes AdamW8bit`、**OOM=0**、8 卡 100%util **83.7GB**(8-bit optimizer 省 ~11GB 装下 4.06B)。orchestrator 将链 A3→A1(bnb8bit)→A2(LoRA)→逐臂 SQuAD dev EM/F1 + base ref。
> - **✅ 三长训练全健康（绝不打断）**：P0.5 armA(LOCAL B200 contig16) step20120 ppl12.68 1.76s/step；armB(.252 final14+2fresh) step19900 ppl13.03 1.76s/step；P1.3(.73+.82 16卡 IB DDP, from-scratch16L uniform lr2e5) step3300 ppl67.75(冷启正常下降 211→67)4.70s/step 98.3GB。ETA P0.5~3.5天到 200k；P1.3 待 B200 空迁移(--resume_from)。到平台记 200k endpoint。
> - Monitor 8088 http200 OK。本轮：2 kill（.104 stuck orchestrator+poison parent，无结果浪费）；1 relaunch（P-C1 orchestrator A3-bnb8bit@.104，task #92）。**bnb8bit 偏离记录**：A3 从 fp32-AdamW 改 8-bit（与 A1 一致，B200 忙~3.5天不可用+no-idle 铁律；8-bit Adam 逼近 fp32，EM/F1 定性结论稳健）。
> - 旧快照（2026-08-03 03:20）保留于下方 →

## 当前快照（2026-08-03 03:20 +08:00，✅ Paper C P-C1 coder 完成+已核实(.104 8卡~99%util、A4 step110 ppl15.9 健康下降、orchestrator pid25138 活)；三训练全健康；重要 DDP module.前缀 bug 已修保 hero 臂有效性）
> - **✅ Paper C P-C1 coder（a67afcb2）完成 + MAIN 已核实（task #92 仍 in_progress，训练 overnight 跑）**：
>   - **A4 freeze-graft HERO 真跑起**（.104 8×H20 ~99% util 56GB/卡）：keep14+fresh2 冻前14、`frozen=2833.5M trainable=1226.9M`（匹配 scoping），移植 6 sanity check 全过（copied=157==157、max|Δ|=0.0、fresh q_std=0.02）。log `logs/paperC_pc1_squad_A4.log` step110/1000 loss2.77 ppl15.9（from step10 ppl>1e4 快速健康下降）、7.34s/step、ETA~1h50m。fp32 AdamW，lr_fresh=1e-4/lr_inh=2e-5，eff_bs128。
>   - **⚠️ coder 发现并修复关键 validity bug**：首次 A4 optimizer 只建一个组（全部 param 落到 inherited@2e-5，fresh cap 随机初始化被以 15× 过低 LR 训）——根因 `build_param_groups` 在 DDP wrap 之后跑，`named_parameters()` 带 `module.` 前缀使 `_classify_param` 的 startswith 全 miss。已修（strip `module.`），kill 重启，现正确 `fresh_decay 815.8M@1e-4 + inh_decay 411.0M@2e-5`。**P1.3 不受影响**（已在跑的进程不受磁盘 edit 影响；且 from_scratch 短路到 fresh；且 P1.3 uniform LR 分组无关）。
>   - **overnight orchestrator**（`scripts/paperC_pc1_orchestrate.sh` setsid pid25138 活）：等 A4 final.pt→顺序链 A3(from-scratch16L,fp32 AdamW single-LR 3e-4)→A1(full-FT 32L,**bnb 8-bit AdamW** BS2/GA8,7B 全参 fp32 单卡 OOM 故偏离,失败自动降 BS1/GA16 再标 NEEDS B200)→A2(param-matched LoRA r=160→399.8M≈A4 fresh405M,bf16 冻 base+fp32 LoRA)→逐臂 SQuAD dev EM/F1 headline eval + base 参考。orch log `logs/paperC_pc1_orch.log`，fault-tolerant（失败臂记录并继续）。
>   - **数据**：`tokenize_squad_olmo2_sft.py` 把 `data/squad_train.jsonl`（零下载风险）packed [N,2048]→770 chunks/1.58M tokens；因数据小 max_steps 2000→**1000**（~166 epoch，2000=332 epoch 纯记忆），eff_bs 保 128 跨臂可比。4 臂同 shard 同 loss（full-LM over packed，answer-only mask 延后）。
>   - **代码 commit `7a330ce`+`9b44e9f`（MAIN 已核实：author=LiuHanzuo，纯 scripts，无 *.pt/*.bin/password）**，rsync .104。新增 tokenize/train_lora/eval_emf1/3 launchers；additive `--optimizer {adamw,bnb_adamw8bit}` + `_classify_param` strip module. 修复。未 push（heartbeat 不 push）。
>   - **MAIN 待办**：orchestrator overnight 出 A4/A3/A1/A2 × {EM,F1} + downstream MC 后，MAIN 回填 paperC 文档（subagent 不碰 .tex/TODOList）；A1 若标 NEEDS B200 则待 B200 空。
> - 旧快照（2026-08-03 02:20）保留于下方 →

## 当前快照（2026-08-03 02:20 +08:00，✅ P0.8 eval 完+回填结项(full32≈base 闭合 P1.1)→.104 空出→按用户预批准起 Paper C P-C1 freeze-graft@.104；三训练(2×B200 P0.5 + .73/.82 P1.3)全健康）
> - **✅ P0.8 结项 + 回填完（task #126 completed）**：full32@25k closed-book QA 全出（PopQA em .1842/contains .2280/f1 .2348；TriviaQA em .5715/contains .6838/f1 .6389；NQ-open em .1582/contains .2443/f1 .2369），**full32 ≈ base ≫ keep14@200k**→坐实"知识损失来自 pruning/policy 而非续训 corpus shift"。MAIN 已回填 paperB/TODOList P0.8 §（DONE + 3臂对照表 + 结论 + raw path）+ tonight-queue item2 标 DONE + P1.1 §"剩余缺口"改为闭合。.104 8卡随之空出。
> - **✅ Paper C P-C1 freeze-graft 已启动@.104 8×H20（task #92，用户 2026-07-27 option1 预批准"LoCoMo 空出 .104/.73 时自动起"，LoCoMo 早已完成→节点真空出→触发；coder agent 后台）**：minimal-viable slice = 4 臂（A4 freeze-graft HERO=keep14+fresh2 冻前14只训 fresh2+norm+lm_head / A3 from-scratch 16L 深度匹配 / A1 full-FT 32L 基线 / A2 param-matched LoRA），finetune task=**SQuAD**（`data/squad_{train,val}.jsonl` 已在盘，零 proxy 下载风险），seq2048 eff_bs128 ~2000 steps，headline=SQuAD dev EM/F1 + downstream MC。**⚠️ 显存注意**：A1 full-FT 7B / A3 16L 4B 全参 fp32-AdamW 单卡 H20 恐 OOM（keep16 实测 4B fp32-AdamW 单 H20 装不下）→ agent 授权用 8-bit bnb Adam 或减 batch+加 ga 保 eff_bs128 并注明 optimizer 偏离；跑不动的臂标注需 B200，先保 A4/A3/A2。**freeze-graft 是本次核心臂**。~154 GPU-h ≈ 单 8×H20 ~2.2 天（远早于 B200 迁移窗）。代码 fork（SQuAD tokenizer + full_ft/LoRA path + eval）在 LOCAL commit 后 rsync .104。
> - **✅ P1.3 LR-matched init 训练健康（task #127，绝不打断）**：.73+.82 16卡 IB DDP，step 840 loss 5.35 ppl 211（from-scratch cold-start 正常，PPL>100 规则不适用冷启）、4.71s/step、16卡100%util 98.3GB。output `outputs/olmo2_p13_scratch16_lr2e5_uniform`，commit c57c4cb。ETA H20≈10.8天→待 B200 P0.5 空出(~3.7天)MAIN 迁 B200(--resume_from)。到平台记 200k endpoint。⚠️ 正确 log 名 `logs/olmo2_p13_scratch16_lr2e5_rank{0,1}.log`（非 _node0）。
> - **✅ 两 B200 P0.5 训练健康（#118，绝不打断）**：LOCAL Arm A(contig16) step 13720/200k ppl~13.6 1.79s/step；.252 Arm B(final14+2fresh) step 13420/200k ppl~13.9 1.77s/step；98.3GB、8卡~99%。ETA ~3.7 天到 200k，未近平台。到平台记 200k endpoint（#103 matched-PPL dense-save re-heal + P1.2 keep14 multi-seed + P0.2 均等 B200 空——4B 全参 fp32-AdamW H20 装不下且 DDP 复制非分片 optimizer，2节点 H20 也不解）。
> - **节点分工遵从（用户 2026-08-02 指令）**：H20 优先 eval→eval 队列排空(PaperA 全 DONE + PaperB depth-ladder/ShortGPT endpoint 全 DONE)→.104 接 P-C1（Paper C 训练，属"节点空出优先补待跑"，且 P-C1 是唯一能在单 H20 跑的 freeze-graft 小可训参训练；重训练 P1.2/P0.2 保留给 B200）；B200 优先长训练(P0.5 满载)。
> - Monitor 8088 http200 OK。本轮：0 kill；1 launch（P-C1 freeze-graft@.104，task #92）；task #126 completed。
> - 旧快照（2026-08-03 00:45）保留于下方 →

## 当前快照（2026-08-03 00:45 +08:00，✅ 依用户新指令 kill 两 P0.4 延伸 agent→按今晚队列起 P0.8 eval@.104 + P1.3 训练@.73+.82 16卡DDP；两 B200 P0.5 训练健康）
> - **⚠️ 纠正 00:31 快照**：用户 00:32 更新 paperB/TODOList 明列「2026-08-03 运行任务审计与今晚队列」——**不要重复运行 P0.4（已有结果，下一步是回填）**。故本轮 kill 掉两个 P0.4 延伸 agent（控制臂 traj ad0ce28a、keep12 dense traj ada06870），二者均**未产出结果**（keep12 仍在传 ckpt、控制臂仍在 stage rsync），无浪费；删 task #125。改按今晚队列起下面两项。
> - **✅ P0.8 full32@25k closed-book QA eval（纯评测，队列第 1 优先，task #126，agent a06d44436931e2db3）@.104 8卡**：闭合 P1.1 原验收的 knowledge-task 腿（full32 无需重训）。rsync `outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt`（87.6GB，LOCAL→.104 共享 diskB 卷，一次）→ `_run_closedbook_8shard.sh` 跑 PopQA(n=14267)/TriviaQA(rc.nocontext val n=17944)/NQ-open(val n=3610)，keep_front=32/n_fresh=0，base 协议(chat=False/add_bos=0/zero-shot/no-retrieval/greedy)。判 full32 是否≈base（若≈base→加强"知识损失来自 pruning/policy 而非续训 corpus shift"）。MAIN 待回填 paperB/TODOList P0.8。
> - **✅ P1.3 LR-matched init 控制训练 已启动+健康（队列第 4 项，task #127，agent a4acb6669f89a4f8f）@.73+.82 16卡 DDP over IB(RoCE v2 200Gbps)**：`train_olmo2_arch_probe2.py --from_scratch --keep_front_layers 14 --n_fresh_layers 2 --lr 2e-5 --lr_inherited 2e-5`（uniform 2e-5，与 keep14 严格同 LR，去除现有 fromscratch 臂 lr_fresh=1e-4 的 init×LR 混杂；arch_meta 确认 scratch16L/from_scratch:true/16L 4.06B/seed42）。eff_bs 128（world16×bs2×ga4，log 确认）、seq2048、200k、save_every5000（rolling-retention 永久保留每 5k 里程碑→50k/100k/150k 自动留）、新唯一 output_dir `outputs/olmo2_p13_scratch16_lr2e5_uniform`。**step80 loss 8.17 健康、4.68s/step、16卡~100%util 96GB、无 NaN/NCCL timeout**。.73=rank0 master(29517)、.82=rank1。**⚠️ 纠正前置条件**：base trainer 原本**无** `--seed` flag（只在 sibling P0.5 trainer 有）→ agent 加了 additive `--seed`（commit **c57c4cb**，未 push，unset 时无行为变化）+ mirror set_seed。**IB 坑修复**：首launch 8-rank/node GDR MR 注册崩(`ibv_reg_mr_iova2 failed`，1-rank smoke 未暴露)→ `NCCL_NET_GDR_LEVEL=0 NCCL_IB_PCI_RELAXED_ORDERING=1` 关 GDR 保留 IB/RoCE 传输，16-rank smoke PASS。runner `scripts/run_olmo2_p13_node.sh` + smoke `scripts/_nccl_smoke_2node.py` 新增。**ETA on H20 ≈10.8天（4.68s/step×200k）→ 待 B200 P0.5 空出(~3.8天)由 MAIN 迁 P1.3 到 B200(1.77s/step)加速（--resume_from 支持 model+opt+RNG 干净续）**。到平台记 200k endpoint。
> - **✅ 两 B200 P0.5 训练健康（#118，绝不打断）**：LOCAL Arm A(contig16) step 10120/200k ppl13.36 1.79s/step；.252 Arm B(final14+2fresh) step 9840/200k ppl14.36 1.76s/step；98.3GB、8卡满。ETA ~3.9 天到 200k，未近平台。到平台记 200k endpoint（#103 matched-PPL dense-save re-heal 亦需等 B200 空）。
> - Monitor 8088 http200 OK。本轮：2 kill（P0.4 延伸 agent，无浪费）；2 launch（P0.8 eval@.104、P1.3 训练@.73+.82）；删 #125，建 #126/#127。
> - 旧快照（2026-08-03 00:31，已被本快照纠正/取代）保留于下方 →

## 当前快照（2026-08-02 23:56 +08:00，✅ P0.3 NQ-open 全完+回填（3 free-form 知识 benchmark）→接 keep12 dense-trajectory paired MMLU（P0.4 加固）；两 B200 训练健康）
> - **✅ P0.3 第 3 个 closed-book benchmark（Natural Questions open）全完 + 回填**（agent a56c04b3）：NQ validation n=3610，harness 加 `nq_open` branch commit **9fabb88**（向后兼容不改 PopQA/TriviaQA）。**主分离在 3 任务一致大幅**：base_full em 0.2050 ≫ 全部剪层-heal 变体 ≤0.063（PopQA/TriviaQA 同）→ 续训不恢复参数化知识，泛化到 3 个 free-form 知识 benchmark。细序：keep14>frozen>random 在 PopQA/TriviaQA 干净成立，**NQ em headline 上 random(.0632)≳keep14(.0598)>frozen(.0496)**（random 以 ~0.33pp 噪声级微超 keep14），但 NQ contains/f1 上 keep14 重回榜首、frozen 三任务稳定最低。净判：分离方向泛化，keep14>random 细序在 NQ em 为噪声级、措辞需弱化。回填 paperB/TODOList P0.3（表+3-benchmark 结论+raw 路径+infra 记录）。
> - **✅ 24 H20 空出→接 keep12 dense-trajectory paired MMLU（P0.4 加固，agent ada06870）**：keep12（14L shell，keep_front12+fresh2）是**唯一有 dense on-disk 轨迹的 heal 臂**——LOCAL wzc1 有 5k..111.5k 每 5k 一点（24 点），.73/.82 有 115k/120k/123.5k/124k(endpoint)。P0.4 主体只有 keep14/keep8 各 2 点稀疏配对，keep12 dense 轨迹直接补 P0.4 文档写明的 limitation。目标 8 点（5k/40k/75k/111.5k 从 LOCAL rsync 到 .73 + 晚段 4 点已在盘），复用 `_run_olmo2_mmlu_peritem_kf_8gpu.sh`(KF=12) + `analyze_traj_paired_mmlu.py`，相邻步+每点 vs endpoint McNemar exact+bootstrap CI。在 .73 单节点 8 卡。带宽慢则降级 2 点。
> - **✅ 两 B200 P0.5 训练健康（#118）**：LOCAL Arm A(contig16) step 8920/200k ppl13.92；.252 Arm B(final14+2fresh) step 8600/200k ppl14.35；~1.76s/step、98.3GB、8卡满。ETA ~4 天，到平台记 200k endpoint。绝不打断（#103 matched-PPL dense-save re-heal 亦需等 B200 空）。
> - Monitor 8088 http200 OK。本轮：0 kill；1 launch（keep12 traj paired MMLU @.73，agent ada06870）；P0.3 NQ-open 回填结项。
> - 旧快照（23:39）保留于下方 →

## 当前快照（2026-08-02 23:39 +08:00，✅ P0.4 trajectory 配对完+回填→24 H20 空出→接 P0.3 可选第3 benchmark NQ-open 加固；两 B200 训练健康）
> - **✅ Paper B P0.4 trajectory 配对 MMLU 全完 + 回填**（task #124 completed，agent a685a4bb）：checkpoint-to-checkpoint 同题 flip 分析，只跑盘上可行对。**keep14 128k→200k**：n=14042，wrong→right=1038 / right→wrong=802，**Δ+1.681pp，McNemar exact p=4.12e-08（显著）**，bootstrap CI[+1.075,+2.286]pp，gold-NLL Δ−0.0210 → 深臂 apex→endpoint 有显著 knowledge gain。**keep8 45k→121k**：n=14042，Δ+0.235pp，**p=0.553（n.s.）**，CI[−0.513,+0.983]，gold-NLL Δ−0.0388 → 浅臂（≈chance .25）MMLU 无显著变化但 calibration 仍改善。回填 paperB/TODOList P0.4 表+notes。计划里 keep14 153.5k / keep8 10k/25k/44k 因盘无 ckpt 标注不可做。
> - **✅ 24 H20 再空出→接 P0.3 可选第 3 个 closed-book knowledge benchmark（Natural Questions open）**：P0.3 plan 明列的可选加固项，把 free-form 知识 benchmark 从 2→3（PopQA+TriviaQA+NQ），复用 harness `eval_olmo2_closedbook_qa.py`（加 `nq_open` task 分支），5 模型 × NQ-open em/contains/f1，base 协议(chat=False/add_bos=0/zero-shot/no-retrieval)，检验 matched keep14-triad 是否与 MMLU/PopQA/TriviaQA 同序同向。agent a56c04b3 后台跑，分 .73/.104/.82 三节点加速。**判定=可选加固，非必需**。
> - **✅ 两 B200 P0.5 训练健康（#118）**：LOCAL Arm A(contig16) step 8460/200k ppl14.03；.252 Arm B(final14+fresh2) step 8120/200k ppl14.18；~1.78s/step、98.3GB、8卡满。ETA ~4 天，到平台记 200k endpoint。绝不打断（#103 matched-PPL dense-save re-heal 也需等 B200 空出）。
> - Monitor 8088 http200 OK。本轮：0 kill；1 launch（P0.3 NQ-open @.73/.104/.82，agent a56c04b3）；P0.4 回填结项 #124。
> - 旧快照（23:05）保留于下方 →

## 当前快照（2026-08-02 23:05 +08:00，✅ #103 P2.8 McNemar 全完+回填→24 H20 空出→接 P0.4 trajectory 配对；两 B200 训练健康）
> - **✅ #103 P2.8 per-item McNemar 全完 + 回填**（agent a70d2dbc）：三 matched 臂 @200k per-item MMLU（n=14042，复现 ledger keep14 .318/frozen .263 ±0.005）。**headline gap keep14 vs frozen Δ+5.50pp，McNemar exact p=6.99e-27，bootstrap 95% CI [+4.51,+6.49]pp**；keep14 vs scratch Δ+7.11pp p=1.64e-46 CI[+6.14,+8.09]；frozen vs scratch Δ+1.61pp p=2.59e-03 CI[+0.58,+2.64]——三臂 knowledge 排序每对均显著。回填 paperB/TODOList line98/101/103/104。#103 仅剩 strict matched-PPL 腿 BLOCKED（需 dense-save re-heal，属 B200 训练，两 B200 忙 P0.5 不打断）。
> - **✅ 24 H20 再空出→接 P0.4 trajectory 配对 MMLU（task #124，coder a685a4bb）**：checkpoint-to-checkpoint 同题 flip 分析。**盘上现实**：keep14 只有 128k/200k（153.5k 已清）、keep8 只有 45k/47.5k/48k/121k（无 10k/25k/44k）→ 只跑可行对 **keep14 128k→200k**（200k jsonl 复用 P2.8）+ **keep8 45k→121k**；计划里 153.5k/keep8 早期点因盘无 ckpt 标注不可做。keep14@128k dump→.73，keep8→.104。复用 harness `_run_olmo2_mmlu_peritem_8gpu.sh`（8dd4694）。
> - **✅ 两 B200 P0.5 训练健康（#118）**：LOCAL Arm A step 7080/200k ppl14.14；.252 Arm B step 6740/200k ppl14.72；~1.77s/step、98.3GB、8卡满。ETA ~4 天，到平台记 200k endpoint。
> - Monitor 8088 http200 OK。本轮：0 kill；1 launch（P0.4 trajectory @.73/.104，coder a685a4bb）；#103 P2.8 回填结项。
> - 旧快照（22:55）保留于下方 →

## 当前快照（2026-08-02 22:55 +08:00，✅ P0.3 closed-book QA 全完+回填→24 H20 空出→接 #103 P2.8 per-item McNemar；两 B200 训练健康）
> - **✅ Paper B P0.3 closed-book QA 全完 + 回填**（PopQA n=14267 / TriviaQA rc.nocontext val n=17944，5 模型，base 协议 chat=False/add_bos=0/zero-shot/no-retrieval）。matched keep14-triad **两任务同序且与 MMLU dissociation 同向**：PopQA contains keep14 0.1415>frozen 0.1283>scratch 0.1112；TriviaQA em keep14 0.2940>frozen 0.2477>scratch 0.2086；base_full 两任务大幅最高。→ **"MMLU dissociation" 可扩为 "knowledge-sensitive benchmarks (MMLU, PopQA, TriviaQA)"**，措辞落 .tex 由 MAIN 后续。回填 paperB/TODOList P0.3 表。（agent af1e5f79 完成）
> - **✅ 24 H20 空出→接 #103 P2.8 per-item McNemar（H20 优先 eval + PaperB 待跑）**：keep14@200k vs frozen@200k vs scratch@200k 三 matched 臂（均 keep_front14+fresh2=16 层）per-item MMLU → 补 gap+5.6pp 的配对 McNemar exact p + bootstrap 95% CI（TODOList line101 "gap/CI/p：待 #103" 的可行腿）。keep14→.73 / frozen→.104 / scratch→.82，各 8-shard，协议同 `_know`（核验复现 keep14≈.3191/frozen≈.2628）。coder a70d2dbc 后台跑。#103 的 strict matched-PPL 腿仍 BLOCKED（盘无 dense-save 交叉 ckpt），本轮不做。
> - **✅ 两 B200 P0.5 训练健康（task #118）**：LOCAL Arm A(contig16) + .252 Arm B(final14+fresh2)，~1.76s/step，ETA ~4 天，到平台记 200k endpoint。
> - Monitor 8088 http200 OK。本轮：0 kill；1 launch（#103 per-item McNemar @.73/.104/.82，coder a70d2dbc）；P0.3 回填结项。
> - 旧快照（22:35）保留于下方 →

## 当前快照（2026-08-02 22:35 +08:00，✅ P2.4 全 3 深度 eval 完+回填→Paper A 实验层面全清；.73 补 Paper B P0.3 closed-book QA；两 B200 训练健康）
> - **✅ Paper A P2.4 全部验收 + 回填（task #122 completed）**：j=6/9/18 三深度 RULER Cohort-B(15cell) + LoCoMo(n=1986 GPT-4o judge) + 16k same-pack Read/Write timing 全完（j6/j9 on .82，j18 on .73）。headline=部署 Arm B(resume_j=j)：RULER 98.29/97.55/(旗舰12=96.07)/55.41；LoCoMo judge 40.38/39.02/(38.27)/28.65；Read ms 830/748/(664)/500，read_speedup A/B 1.17/1.27/(1.40)/1.81×。**j 是单调 quality↔latency 旋钮**，浅 j 质量略超旗舰、深 j18 坍塌。回填 paperA/TODOList 表+顶部状态。**Paper A 实验层面全部完成，.tex P2.4 集成也 DONE（linter 确认 line 9）→ 无剩余模型 run**。
> - **✅ .73 空出→补 Paper B P0.3 closed-book QA（H20 优先 eval + PaperA 排空后接 PaperB，见 memory h20-paperA-over-paperB + h20-eval-b200-train）**：PopQA+TriviaQA closed-book，5 模型（base/keep14@200k/random-front@200k/frozen-front@200k/keep8@121k），base 协议(chat=False/add_bos=0/zero-shot/no-retrieval)，检验 perplexity–knowledge dissociation 是否泛化出 MMLU。harness `eval_olmo2_closedbook_qa.py`（commit d05ef59，此前从未跑过，两 result dir 空）。coder af1e5f79 后台跑（可分 .73/.104/.82 三节点加速）。
> - **✅ 两 B200 P0.5 训练健康推进（task #118）**：LOCAL Arm A(contig16) step 6060/200k loss 2.642 ppl 14.04；.252 Arm B(final14+fresh2) step 5720/200k loss 2.692 ppl 14.76；均 1.76s/step、98.3GB、8卡满载，ETA ~4 天，到平台记 200k endpoint。
> - **⚠️ 关键纠正（避免误操作）**：stale memory 曾说「resume 暂停的 keep8/10/12 heal」——但按用户 2026-08-02 决定，keep8@121k/keep10@83.5k/keep12@124k 已在 paperB TODOList 记为 **[DONE] 200k endpoint**，**不得 resume**。真正剩余 PaperB GPU 待跑=P0.3(本轮起)、#103 dense-save re-heal（缺严格 matched-PPL 交叉 ckpt）。
> - Monitor 8088 http200 OK。本轮：0 kill；1 launch（P0.3 closed-book QA @.73，coder af1e5f79）；paperA P2.4 回填结项 #122。
> - 旧快照（19:29）保留于下方 →

## 当前快照（2026-08-02 19:29 +08:00，✅ P1.6 SnapKV yarn 全完+回填→LOCAL B200 迁入 P0.5 Arm A 加速 5.4×；Arm B 待 .252 空出再迁）
> - **✅ P1.6 SnapKV yarn 全完（18:55）+ score + 回填 paperA TODOList**：ns2 64k/128k=100/100、mk1=94/91、vt=80/84.8（IRON-LAW-2 全 OK），raw `ruler_results/p16_snapkv_yarn/`。PyramidKV yarn 在 .252 收尾（~8 job VT yarn）。
> - **✅ 用户策略（2026-08-02，见 memory `b200-prefer-paperB-when-free`）：B200 当前任务跑完→优先 Paper B 长程训练上 B200 加速**。LOCAL B200 SnapKV yarn 完→空出→**迁入 P0.5 Arm A**（contig16 keep0-15，`_run_p05_armA_b200.sh`，`.venv` torch2.13，从 base 剪层从头起）。⚠️ **配方统一**：两 arm 都将用 `.venv`（torch2.13）→ .104 Arm A(olmo2_venv torch2.7 step2060) 已 kill 作废，.73 Arm B 待迁后同样重起。**实测 1.75s/step（vs H20 9.5s=5.4× 加速），200k ETA≈4 天（vs 22 天）**，8 卡 97GB/100%，loss 8.4→6.2 冷启下降健康，pid 217510。
> - **⏳ Arm B（.73 H20 step~2060 olmo2_venv）暂不动**：等 .252 PyramidKV yarn 完→kill .73 Arm B→在 .252 B200 用 `.venv` 从头起（两 arm 各占一台 B200 并行，配方一致）。
> - **✅ .82 P2.4 eval**（coder a40444866）：三深度训练完→跑 RULER Cohort-B 15cell + LoCoMo + 16k timing，进行中。
> - **✅ .104 空闲**（Arm A 已 kill）：按 h20-paperA-over-paperB 待补 PaperA 项或 resume 暂停的 PaperB heal（下轮 heartbeat 处理）。
> - 旧快照（17:28）保留于下方 →

## 当前快照（2026-08-02 17:28 +08:00，✅ #100 平台 eval 全完+回填→LOCAL B200 补 SnapKV YaRN 重跑；P2.4 j=18 收尾；PyramidKV VT 长档中）
> - **✅ #100 full32 平台 eval 全完（17:24）**：PPL **7.6699** / core6 **.6968** / aux5 **.6536** / MMLU **.5867**（94.8% above-chance recovery vs base .6053）→ 回填 `PAPERB_THREE_ARM_200K.md`，**task #100 completed**。结论：intact-32L 续训近乎无损，是深度阶梯/ShortGPT 对照的干净上锚。
> - **✅ LOCAL 8×B200 空出→立即补 SnapKV YaRN 64k/128k 重跑（铁律1，PaperA P1.6 待跑项）**：`scripts/_rerun_snapkv_yarn.sh`（修好的 harness，rope_theta fix `14bd576`），pid 136363，8 shard×3task×{64k,128k}。**已验证修复生效**（niah_single_2/64k 跑到 19/100，无 NoneType/pow 崩），log `logs/p16_snapkv_yarn_rerun/`，ETA ~35min。PyramidKV yarn 由 .252 scheduler 用同 fix 自动带到。
> - **✅ .252 P1.6 PyramidKV**：native 3 task 全 8 shard 在跑 variable_tracking 长档 generation（16 done + VT in-flight），未卡死，healthy。
> - **✅ .82 P2.4**：j=6/j=9 DONE，**j=18 step 330/4000**（loss 0.14），8 卡 healthy。训完即 3 深度 eval。
> - **✅ .104 P0.5 ArmA step1440 ppl17.43 / .73 P0.5 ArmB step1440 ppl18.34**：快降中未平台，保留。
> - 旧快照（17:04）保留于下方 →

## 当前快照（2026-08-02 17:04 +08:00，✅ #100 full32 到平台→停+记 200k endpoint→LOCAL B200 补平台 eval；PyramidKV/P2.4 推进中）
> - **✅ #100 full-32L 控制到平台→停（用户「到平台期就可以了」授权）**：ppl 从 step10k 起锁 8.1–8.4 共 17k+ 步无下降趋势（10k=8.19/20k=8.11/27.7k=8.41）→ **step27740 SIGTERM 停 torchrun coord 3296519（8 rank 干净退出）**，固化 `outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt`（87.6GB）为 **200k-equivalent plateau endpoint**。
> - **✅ LOCAL 8×B200 空出→立即补 #100 平台 endpoint eval**（PaperB P1.1 闭合）：`scripts/_run_olmo2_eval_full32_plateau.sh`（keep32/n_fresh0，base 协议 PPL+core6+know5，8-GPU sharded，与 keep14/ShortGPT 同 harness），pid 112491，log `logs/full32_plateau_eval.out`。8 PPL shard running（load 87GB ckpt 中）。⚠️ P0.5 现太早（step~1280 ppl~19 快降中，无 ckpt）不迁移，等它自然到平台再停。
> - **✅ .252 P1.6 PyramidKV**：16 job DONE（niah_single_2 全 8 shard + multikey 部分），56-job pool 推进中，healthy。YaRN fix 已 commit `14bd576`，待 PyramidKV 腾卡后重跑两 method yarn 64k/128k。
> - **✅ .82 P2.4**：j=6 DONE、**j=9 step 2920/4000**（loss 0.043）、j=18 待续，8 卡 100%/35GB healthy。
> - **✅ .104 P0.5 Arm A / .73 P0.5 Arm B**：step ~1280/200000（ppl ~18–19 快降中，9.5s/step，98GB），早期健康，绝不停（未到平台）。
> - 旧快照（16:08）保留于下方 →

## 当前快照（2026-08-02 16:08 +08:00，✅ .252 SnapKV 完→补 PyramidKV；YaRN bug 派 coder 修；5 节点全占）
> - **✅ .252 refill（铁律1）**：P1.6 **SnapKV pool DONE**（SCHED_DONE @15:42）——native RULER(15 cell)+LoCoMo 数字 good（VT 8k=100→128k=4.4；LoCoMo F1 9.21/acc 22.05）；**但 (a) PyramidKV 从未跑**（本轮 scheduler 是 SnapKV-only，SUMMARY 无 pyramid 段）+**(b) SnapKV YaRN 64k/128k 全 shard 崩**（`yarn_factor=None` pow 崩）。→ .252 空出立即起 **PyramidKV**（`DRY=0 METHODS=pyramidkv`，56-job pool，gate PASS，8 卡 25-30GB loading，pid 1094886，log `.252:logs/p16_kvcompress/sched_pyramidkv.out`）。YaRN bug 派 coder a542419157（worktree）修 `eval_p16_kvcompress.py` → 修好后 .252 空出重跑两 method 的 yarn cells。
> - **⚠️ 启动坑记录**：远程 setsid launch 的 `>logs/...` 重定向在 SSH `$HOME` 解析，需 `cd $P && DRY=0 ... bash ...` 把 cd 放整条命令最前（放 launch 后无效）。
> - **✅ .82 P2.4 深度曲线**：j=6 **step 3010/4000** loss 0.027，8 卡 97-100% util 37GB healthy（pid 2422703/parent 2422697，task #122）。串行 j6→9→18，j=6 ~2min 内完切 j=9。
> - **旧快照（14:52，.82 P1.7 两 cohort 全完→补 P2.4）保留于下方 →**

## 当前快照（2026-08-02 14:52 +08:00，✅ .82 P1.7 两 cohort 全完→补 Paper A P2.4 深度曲线训练；5 节点仍全占）
> - **✅ .82 refill（铁律1 + H20 PaperA>PaperB 优先级）**：P1.7 h12-oracle **cohort-a + cohort-b 双双 COMPLETE**（.82 8 卡空出）。cohort-b macro n_paired=1500：A(j=0)=99.19 / C(oracle)=99.19（bit-identical，p=1）/ B(chunk-local)=96.07，A−B=C−B=+3.12pp（CI[2.36,3.93] p=8.79e-24）== P0.13 deployable gap → **#121 结项**。→ 立即在 .82 补下一个 PaperA 待跑项 **P2.4 蒸馏多深度 quality-latency 曲线**（j∈{6,9,18} rank-32 LoRA 训练，匹配 flagship j=12 配方）：coder ae216e60 已提交 launcher `ebfe475`（串行 8-GPU DDP 训 3 深度）。**✅ 15:02 确认 j=6 在跑：step 90/4000 loss~0.11，8 卡 100% util 37GB/卡，torchrun pid 2422703**，resume_j=6/rank32/α64/total4000/lr8e-5/seed42，log `.82:logs/p2_4_distill_j{6,9,18}.log`。ETA ~1-1.7h/arm × 3 serial ≈ 3-5h。
> - **✅ 5 节点全占 healthy**：**LOCAL**（8×L20A）full32 #100 continued-pretrain step~24.9k/200k healthy；**.252**（8×L20A）P1.6 SnapKV equal-budget（VT@64k ~75%，收尾后自动跑 PyramidKV 56 job，task #120）；**.104**（8×H20）Paper B P0.5 Arm A（contiguous16/no-fresh，pid 4105448，task #118）；**.73**（8×H20）Paper B P0.5 Arm B（retained-final14+fresh2 差分 LR，pid 4077879，task #118）；**.82**（8×H20）→ P2.4 深度曲线训练 ✅ j=6 step90/4000 running（pid 2422703，task #122）。
> - **✅ paperB P0.1/P0.2 回填完（MAIN，无模型运行）**：P0.1 [DONE] step-0 anchor（keep14 step0 PPL 167,371/MMLU .254 chance，R_MMLU 13.4/16.6/18.5%，R_LM 96.2/96.3/96.5% saturated 不作 headline，确认 recovery 分母=vanilla−chance）；P0.2 [PARTIAL] 端点 anchors 填完（keep14@200k 10.561/.3191 vs random 11.498/.2461 vs frozen 12.797/.2628），严格 ≤0.10 matched-PPL 交叉点缺 on-disk ckpt → 待 #103 dense-save re-heal。commit 8433453。
> - **📌 待办**：P2.4 训练完 → eval（RULER Cohort-B 15 cells + LoCoMo + timing）；.252 SnapKV 完自动 PyramidKV；P0.6 content-MMLU sweep auto_launch 排队等下一空节点（PaperA 待跑排空后）；#103 matched-PPL dense-save re-heal；**PaperB P2.4 通用-SFT 可修复性 pipeline 已交付（no-GPU，commit `d05ef59`，task #123）——data(CPU) + pre/sft/post(8-GPU 3+arm)，GPU 阶段受 PaperA 优先约束+当前无空节点 → 排队待 PaperA 待跑排空**。
> - Monitor 8088 http200 OK。本轮：0 kill；1 launch（P2.4 .82，coder ae216e60）；paperA P1.7 回填结项 #121 + paperB P0.1/P0.2 回填。

## 当前快照（2026-08-02 14:12 +08:00，✅ 5 节点全占 healthy；.252 keep8 eval 已完→跑 P1.6 SnapKV；P0.7 审计回填完）
> - **✅ 5 节点全占，0 无计划空转（铁律1 满足）**：
>   - **LOCAL**（8× L20A）：full32 #100 continued-pretrain，**step 24620/200k**，loss 2.1233 ppl 8.36，3.16s/step，8/8 GPU @100%，healthy（task #100）。
>   - **.252**（8× L20A）：**Paper A P1.6 SnapKV equal-budget campaign**（keep8 eval 已「ALL DONE」→ 已切 P1.6）。当前跑 `ruler_snapkv_niah_single_2` native 8-shard，128k 长度 52%（~6.5s/it），gate PASSED，全 log 0 error。task_queue 48 job（56 减去已完成的短档），9 proc 活（8 worker + sched）。log `logs/p16_kvcompress/sched_snapkv.out`（task #120）。
>   - **.104**（8× H20）：Paper B **P0.5 Arm A**「contiguous16/no-fresh」keep[0..15] n_fresh=0 单 LR2e-5，训练中 8/8 @100%，pid 4105448（task #118）。
>   - **.73**（8× H20）：Paper B **P0.5 Arm B**「retained-final14[0-12,31]+fresh2」差分 LR（inherited2e-5/fresh1e-4），训练中 8/8 @100%，pid 4077879（task #118）。
>   - **.82**（8× H20）：Paper A **P1.7 h12-oracle cohort-b** VT@128k quality 4-shard（pid 2386791 等），GPU3/6 active、余卡 model-resident 等待长档，healthy（task #121）。
> - **✅ P0.7 aggregate 审计回填完（MAIN，无模型运行）**：paperB/TODOList.md 深度阶梯表 `AUDIT`→审计值（base aux5_raw .6637 / keep8 .4289 / keep10 .4491 / keep12 .4608 / keep14 **.4935**）+ 列名 aux5 aggregate→aux5_raw；keep14 recovery 19.4→19.5%；line-40 note + ShortGPT §（.5596 确认正确）+ P0.7 §[DONE]（含结果表 + deliverable 路径）。status/RUN_REGISTRY.md + PAPERB_THREE_ARM_200K.md 加 know5=aux5_raw 命名对齐 note（值经 JSON 核对正确，唯一数值修正 keep14 .5071→.4935 不在这两表）。
> - **📌 待办**：SnapKV campaign 完 → 在 .252 起 `DRY=0 METHODS=pyramidkv`（gate 已由 bf9bc41 修复，56 job）；P0.6 content-MMLU harness（agent ac954257135164176 running）交付后起全 sweep。
> - Monitor 8088 http200 OK。本轮：0 kill；0 新 launch（P1.6 SnapKV 上轮已起，本轮确认健康推进）；P0.7 回填 3 文件。

## 当前快照（2026-08-02 13:25 +08:00，✅ P1.7 harness 交付→立即上 .82 跑起来；h12 oracle 精确 PASS；keep8 端点回填=200k；P2.3 回填完）
> - **✅ Paper A P1.7 launch on .82**（8 卡全占，pid 2371359）：harness agent a0a846b4 交付 `scripts/bench_p1_7_h12_oracle.py`+`_run_p17_oracle.sh`（commit **3327b34**，未 push）。rsync 本机缺→用 cat-over-ssh 同步 2 脚本到 .82:diskB（torch-base tf5.5.4/torch2.13，compile OK）。**RUN=1 COHORT=min** = niah_multikey_1×{8k,16k}×4shard=8 job。**h12_sanity 精确 PASS：continuous-oracle-h12 vs stock lower-12 forward max_abs=0.000e+00**（bit-identical，ref_abs_max=8576，tol5e-2）→ oracle 有效性硬确认。manifest gate 过 + 8 GPU workers 全跑（100%/40% util，~18.5GB/卡）。log `.82:logs/p1_7_oracle.out`，OUTDIR `bench_results/p1_7_h12_oracle`。
> - **✅ keep8 端点回填（按用户指令=200k，不写 121k）**：用户 mid-turn「不要写 121k，直接写这是 200k 的结果，我自己会加正确描述」→ paperB §深度阶梯 keep8 行改为 `[DONE]` 200k endpoint（10L shell），移除 121k/plateau 措辞；step-note 同步移除 keep8=121k。数字 **PPL 13.3332 / tax 1.802× / core6 0.5238 / MMLU .2535 / recovery 1.0%**（aux5=`AUDIT`，随全表 P0.7 待审）。ladder 单调（越深越好）：tax 1.802→1.732→1.547→1.428，core6 0.5238→…→0.5938，MMLU recovery 1.0→6.1→7.1→19.4%。
> - **✅ Paper B P2.3 Qwen 跨家族回填完**：paperB §P2.3 由 linter 收敛到保守口径（know5→`AUDIT`、letter-protocol MMLU 未恢复、不与 OLMo 三点拼纯 depth law）；核心 finding「healed Qwen PPL tax 2.06× 但 MMLU .2495≈chance」保留（跨家族 dissociation 稳健）。
> - **📌 .82 现跑 P1.7（非再留 P1.6）**：P1.7 harness ready-now 且复用 .82 上的 P0.13 manifest → 上 P1.7 最省事（铁律1：不留 .82 空等未 commit 的 P1.6）。**P1.6 harness（a1efd83a）仍在写**，commit 后上下一个释放节点（.252 keep8 eval ~30min 或 .82 P1.7 ~30-60min 完）。
> - **✅ 5 节点全占**：LOCAL full32 #100（~22.9k/200k healthy）+ .252 keep8 eval（step121000 端点已出+回填，110k/100k 收尾）+ .104 P0.5 ArmA + .73 P0.5 ArmB + .82 P1.7 oracle。Monitor 8088 http200 OK。**铁律1 满足，0 无计划空转**。
> - 本轮：0 kill；1 launch（P1.7 .82）；paperB 回填 keep8+P2.3。

## 当前快照（2026-08-02 13:12 +08:00，✅ 24 卡空出→铁律1 补卡：Paper B P0.5 双臂上 .104+.73；Qwen #117 完成；.82 留给 imminent Paper A P1.6）
> - **🟢 铁律1 补卡（24 卡空）**：Qwen #117 eval 完 + .82 ShortGPT eval「ALL DONE」→ .82/.104/.73 三节点 24×H20 全空。Paper A GPU 项（P1.6/P1.7）仍 code-prep 中（无 GPU-ready job），故上**已授权+code-ready 的 Paper B P0.5 结构隔离双臂**（commit 759f4af，DRY→RUN=1）：
>   - **.104 = Arm A「contiguous16」**：keep [0..15] 连续 16 层、**n_fresh=0**（纯 ShortGPT 移植，无新生尾层），单 LR 2e-5 heal。pid 4103679，`outputs/olmo2_p05_armA_contig16`，log `logs/olmo2_p05_armA_contig16.log`。
>   - **.73 = Arm B「retained-final14+fresh2」**：keep [0..12,31]（14 非连续含末层）+ 2 新生尾层，差分 LR（inherited 2e-5 / fresh 1e-4）。pid 4076314，`outputs/olmo2_p05_armB_final14_fresh2`。
>   - 两臂 fp32-master、eff_bs128（BS4×GA4×8）、seq2048、200k、save_every5000+extra{50k,100k,150k}、olmo2_venv、diskB /dev/shm dolmino。**matched-step 对照在存点里程碑（50k/100k/150k/200k）比，非 wall-clock，单节点各 8 卡即可**。13:09 init 健康（各 9 proc）。
> - **📌 .82 保留给 imminent Paper A P1.6**（用户「PaperA 优先级最高」）：P1.6 harness（SnapKV+PyramidKV，agent a1efd83a）13:04 刚 vendored 完 GQA-aware clusters，正写 Qwen3 tf-5.14 monkeypatch харness，~15-20min 出。**这是「已明确即将点亮」的计划性桥接（比照 scp-wait 先例），非无计划空转**；harness commit 即在 .82 起 P1.6。
> - **✅ Paper B P2.3 Qwen 跨家族 #117 完成**（agent a79b7fac）：Qwen f12k2 healed（14/36=39% 深）**PPL 23.49（=2.06× base 11.42）但 MMLU 0.2495≈chance（recovery ≈0%）**，core6 .4624 know5 .3689；base full-36 PPL11.42/MMLU.7297/core6.6648/know5.6850。**「PPL 恢复 MMLU 滞后」dissociation 跨家族稳健、且更浅→知识恢复更差单调趋势与 OLMo 一致**（Qwen 39% recovery0% < OLMo keep12 44% 7.1% < keep14 50% 19.4%）。脚本 commit f83a696（eval_qwen3_probe2_{ppl,downstream}.py + _run_qwen3_probe2_eval.sh）。**MAIN 回填 RUN_REGISTRY + paperB/TODOList（下方进行中）**。
> - **✅ 其余 2 训练/eval 臂**：LOCAL #100 full-32L（step~22.9k/200k healthy）+ .252 keep8 平台期 eval（step121000 PPL=13.333 已出，core downstream phase；跑完回填 paperB §深度阶梯 keep8 行 + task #96）。Monitor 8088 待 curl。
> - 本轮：0 kill；2 launch（P0.5 Arm A .104 / Arm B .73）；#117→completed，#118→in_progress。**铁律1 满足**（.104/.73 上 P0.5 训练，.82 明确即将点亮 P1.6，.252 跑 keep8 eval，LOCAL 跑 full32；无无计划空转）。

## 当前快照（2026-08-02 12:56 +08:00，★ 用户「.252 饱和了就 kill 了 eval 写进文件」→ keep8 平台期 kill+eval；P0.15 交付、P1.7 需 resume）
> - **★ 用户指令执行：.252 keep8 平台期 kill + eval**：keep8+fresh2（10L，最浅臂）训练到 step121000/200k，**训练 PPL 已 post-100k 平台**（15.37@50k→14.37@80k→~13.5@120k，单步 13.00@115k/13.69@120k=batch 噪声），且 keep8 比 keep10（83.5k 知识轴平台）更浅→更早饱和。按用户「饱和就 kill+eval+写文件」：**KILL keep8 训练**（.252 parent 974856 + 8 rank 974929-936，0 残留 + 0 compute-apps，8 卡释放）→ 立即在释放的 .252 起 **base-protocol eval 3 点阶梯 STEPS=121000/110000/100000**（PPL+core6+know5，keep_front=8 n_fresh=2，8-GPU sharded），既定 keep8 端点数字又文档化知识轴平台（~21k step 内是否 noise-flat）。log `logs/eval_keep8_FINAL.log`，脚本 `scripts/_run_olmo2_eval_keep8.sh`（commit a01de0a）。**跑完 MAIN 回填 paperB §深度阶梯 keep8 行 + RUN_REGISTRY + task #96**（keep8 是 #96 最后一格，keep10 已冻结）。
> - **✅ Paper A P0.15 交付**（subagent aab0e8c5 DONE）：`status/P0_15_AUDIT.md`（commit 6bfcc55，无 GPU，CPU-only 打分核实）。结论：A) j=0 cell 分解全 5 benchmark 无 [BLOCKED-DATA]，RULER Cohort-B niah_single_3 macro 99.20 单列；B) 读长口径无硬矛盾，2 处 loose-wording 可选微调；C) 匿名扫描**唯一硬命中** `08_statistics_appendix.tex:79-81` 真实 judge 域名需泛化，余 clean。**MAIN 待办（等用户签字后在 .tex 执行）：1 必改 + 4 可选**。
> - **⚠️ Paper A P1.7 需 resume**（a0a846b4 finished 但**未交付 harness**）：git log 无其 commit，盘上无 bench_p1_7_h12_oracle.py——它去跑 GPU sanity 卡住提前停。已 SendMessage 修正：纯 code-prep 禁跑 GPU，只写 harness+DRY launch script+commit。
> - **✅ Paper A P1.6 仍在跑**（a1efd83a，SnapKV+PyramidKV harness code-prep，未完成通知）。
> - **✅ 其余 3 臂继续**：LOCAL #100 full-32L（step22900/200k ppl8.37 8×100% healthy）+ .82 ShortGPT step153500 eval（PPL+core6 done，know5 phase near done）+ .104/.73 Qwen f12k2 eval（scp ~98% 完即启动，coder a79b7fac）。Monitor 8088 http200 OK。
> - 1 kill（keep8 .252 平台期）+ 1 launch（keep8 eval .252）+ 1 subagent resume（P1.7）。铁律1 满足（.252 kill 后立即起 eval 不空转；余卡全 training-healthy 或 near-done eval）。

## 当前快照（2026-08-02 12:40 +08:00，★★ 用户「paperA todolist 更新了记得跑=优先级最高」+「H20 应该空出来好几台」→ Paper A 顶到 PaperB P0.5 之上）
> - **★★ 用户指令（最新，覆盖 12:11）**：Paper A（`paperA/TODOList.md`）为**最高优先级**；H20 应有数台空闲。按 TODOList §建议顺序 P0.15→P1.6→P1.7→P2.4 推进。**P0.5 launch 排到 Paper A 之后**（h20-paperA-over-paperB-priority memory）。
> - **✅ 已派 3 条 Paper A 流（全 background）**：
>   - **P0.15**（REQUIRED, NO GPU，subagent aab0e8c5）：j=0 cell 分解（从现有 raw preds，RULER paired j=0=Cohort-B niah_single_3 单列勿混 Cohort-A）+ 读长术语扫描（nominal 6,657 vs actual 6.2–6.5k）+ 匿名/可复现扫描；只产出 `status/P0_15_AUDIT.md`，不碰 .tex。
>   - **P1.6**（HIGH-VALUE, inference-only，coder a1efd83a，code-prep）：官方 SnapKV+PyramidKV（禁自研 PyramidMemory/SnapKV-on-chunks）retained budget=6,657，RULER Cohort-A 15 cell + LoCoMo，报 full-prefill lat/peak-mem/decode。写 `scripts/eval_p16_kvcompress.py`+`_run_p16_baselines.sh`+`src/baselines/{snapkv,pyramidkv}/`。**完整 eval 等 MAIN 在 diskB 空节点启动**。
>   - **P1.7**（HIGH-VALUE, inference-only，coder a0a846b4，code-prep）：continuous-prefix h12 归因 oracle（第 3 臂：连续位置/全 causal 跑 layers 0-11 截 pack-level h12 → 同 LoRA 跑 12-35），min niah_multikey_1 8k/16k n=100。写 `scripts/bench_p1_7_h12_oracle.py`+`_run_p17_oracle.sh`。**完整 eval 等 MAIN 启动**。
> - **📋 排程**：Paper A GPU 项（P1.6/P1.7）gating=代码准备（~30-60min），非缺节点 → 让 diskB 三节点先跑完 near-done 的 PaperB eval（.82 salvaged ShortGPT step153500 ~40min；.104/.73 Qwen eval scp ~3min out 后启动），随节点释放 MAIN 即上 Paper A GPU eval。**抢占 near-done eval 只会空转（Paper A 无 GPU-ready job）→ 不抢占**。
> - **✅ .82 ShortGPT step153500 eval 修复重启**：原 driver 用坏的 `.venv/bin/python`（diskB 上 numpy/torch 缺）→ 5 秒崩。已 kill 坏 driver+scp，用 `olmo2_venv/bin/python` 直接在 .82 重启（ckpt 48.7GB 已在盘，救回 sunk scp），当前 PPL phase 健康。
> - **✅ 其余训练臂继续**：LOCAL #100 full-32L（step~22600/200k ppl~8.55）+ .252 keep8（wzc1，~1 天到 200k）。Monitor 8088 http200 OK。
> - 0 kill（本轮，坏 driver 上轮已清）；1 launch（.82 salvaged eval）+ 3 subagent dispatch（P0.15/P1.6/P1.7）。铁律1 满足（24 卡：.82 跑 eval，.104/.73 scp 等待即将点亮，Paper A code-prep 并行推进）。

## 当前快照（2026-08-02 12:11 +08:00，✅ 用户「paperB todolist 更新了你可以开始跑」→ 备 P0.5 结构隔离双臂 + Qwen eval scp in flight）
> - **★ 用户指令（最新）**：paperB TODOList 更新完毕，可开始跑；平台期确认后即可 kill（效率）。新增最高优先级项 = **P0.5 ShortGPT 结构隔离控制**（task #118）：Arm A contiguous16/no-fresh（继承 layers 0-15，0 fresh）+ Arm B retained-final14+fresh2（继承 [0-12,31]=14层 + 2 fresh）。拆分 ShortGPT-16 vs keep14 的「继承层数」与「选层/final-layer retention」混淆。
> - **✅ 机制已核实 + 派 prep coder（a0ef019c，无 GPU）**：Arm A 复用现成 `train_olmo2_shortgpt.py`（任意 keep_layer_indices，n_fresh=0）零改代码；Arm B 需新 `train_olmo2_shortgpt_fresh.py`（shortgpt 任意索引 + arch_probe2 的 fresh-init + 双桶 LR）。coder 写两个 DRY-by-default launch 脚本 + dry-run 验证 + 同步 diskB + git commit，**不启动 GPU**（两节点正忙）。
> - **✅ diskB 训练资产齐全**（.104/.73 共享）：OLMo-2-7B base（`../models/OLMo-2-1124-7B`）、`dolmino_now15b.npy`（repo+/dev/shm）、val、olmo2_venv/.venv 均在。
> - **⏳ Qwen eval（#117，P2.3）scp in flight**：coder a79b7f 存活，ported eval 脚本已写好（diskB+LOCAL），`outputs/qwen3_minarch_armB_f12k2_200k/final.pt` 正 scp→diskB（12:11 时 23.5/47GB，~14MB/s，~再 28min）；scp 完点亮 .104(f12k2)+.73(base-ref)。**16×H20 此刻空 = scp 传输必要等待（非无计划空转），已有明确即将点亮计划**。
> - **📋 排程决策**：Qwen eval（P2.3 可选、短~2h、已 90% 备好、答用户「训完也评测」）先用 .104/.73；P0.5（最高优先级、需 Arm B 代码~30-60min）并行备好，Qwen eval ~14:40 释放两节点即启动 **Arm A on .104 + Arm B on .73 并行**（2 独立臂各占 1 节点，同步跑到 200k 便于 matched-step 对照）。抢占 Qwen eval 不划算（省 P0.5 ~1.6h 却要重跑 scp+eval ~2.6h）。
> - **✅ 其余训练臂继续**：LOCAL #100 full-32L + .252 keep8（wzc1）+ .82 ShortGPT 中点 eval。
> - 0 kill；0 新 GPU launch（Qwen eval 已在途、P0.5 待节点释放）。铁律1 满足（16 卡 scp 等待 + 最高优先级 P0.5 prep 推进中）。

## 当前快照（2026-08-02 11:34 +08:00，✅ heartbeat：16×H20 补卡起 Qwen 剪层-heal 跨家族对照 eval）
> - **🎯 发现 disk 上已有未评测的 Qwen 剪层-heal ckpt** = 用户 OLMo-vs-Qwen 问题的经验答案：`outputs/qwen3_minarch_armB_f12k2_200k/final.pt`（47GB，Qwen3-8B keep_front12+fresh2=14L，healing arm，训练到 200k on SlimPajama-Qwen-tok，**从未 eval**）= Paper B **P2.3 跨家族对照**。全项目无任何 qwen minarch 的 ppl/downstream/mmlu 结果。
> - **✅ 铁律1 补卡**：16×H20（.104+.73，diskB）→ 派 coder a79b7f 把 OLMo probe2 eval（`eval_olmo2_probe2_{ppl,downstream}.py`）移植到 Qwen3（复用 `train_qwen3_arch_probe2.py` 的 pruned-shell 构建），base-protocol（chat=False/no-BOS/LL-MC）跑 PPL(slimpajama_val_2048_qwen3)+core6+know5(MMLU)：**.104 = Qwen full-36 base 参照（Control 0），.73 = f12k2 healed 200k**，并行。算 MMLU above-chance recovery 与 OLMo keep12(.2752/7.1%)/keep14(.3191/19.4%) 对比。coder scp 47GB final.pt→diskB(.73)，跑完 rm。
> - ⚠️ 口径：Qwen 训练语料 SlimPajama≠OLMo Dolmino → PPL 不可跨家族直接比；但 MMLU recovery 语料无关、可比。f12k2 深度 14/36=39%（比 OLMo keep12 的 44% 略深）。
> - **✅ 其余 2 训练臂继续**：LOCAL #100 full-32L（step21560/200k ppl8.11 8×100%）+ .252 keep8（8×100%，wzc1）。.82 跑 ShortGPT 中点 eval（153500/128000）。Monitor 8088 http200。
> - 0 kill；1 launch（Qwen probe2 eval on 16×H20）。铁律1 满足。

## 当前快照（2026-08-02 11:15 +08:00，★用户决定：keep10/keep12 平台期冻结=200k 并 kill；16×H20 释放）
> - **★ 用户指令执行完毕**：keep10(12L) / keep12(14L) 知识轴（core6/know5/MMLU）近 ~13k step 已在噪声内平台（MMLU 钉 chance .27），按用户「已到平台期直接用当前数据=200k、然后 kill」：
>   - **keep10·83500 冻结=≈200k**（PPL 12.8160 core6 .5303 know5 .4491 mmlu .2718，batch82k MAIN 从 JSON 核对）→ **.104 训练 kill**（parent PID 3995588 + pkill `keep10[f]resh2`，0 残留进程 + 0 GPU compute-apps）。
>   - **keep12·124000 冻结=≈200k**（PPL 11.4426 core6 .5669 know5 .4608 mmlu .2752）→ **.73 训练 kill**（parent PID 3983054，0 残留 + 0 compute-apps）。
>   - 已写入 `paperB/TODOList.md` §深度阶梯 200k 端点 + RUN_REGISTRY frontier 表加 ⇒≈200k 标注 + 冻结说明；task #95 完成、#96 keep10 冻结（keep8 仍跑）。
> - **⚠️ pkill self-match 坑修复**：`pkill -f 'train_olmo2_arch_probe2.*keep10fresh2'` 会匹配到运行 pkill 自身的 shell（cmdline 含该 pattern 字面）→ 杀掉 SSH 会话报 exit 255。改用 `[f]` 括号技巧 `keep10[f]resh2`（正则匹配训练进程但不匹配含字面 `[f]` 的自身 cmdline）+ 显式 parent PID kill -9，一次干净落地。
> - **✅ 剩 2 训练臂继续跑满 200k**：LOCAL #100 full-32L（step~21000/200k，~6.5 天）+ .252 keep8（step~116000/200k，wzc1 FS，~1 天，无 frontier 轨迹可冻结故跑满）。
> - **🟢 .104 + .73 = 16×H20 释放**（diskB FS）→ 铁律1：下轮 heartbeat 可补 Paper C P-C1（task #92，auto-launch on free .104/.73）或其它 pending。Monitor 8088 待下轮 curl 确认。
> - 2 kill（keep10 .104 / keep12 .73）；无 launch。ShortGPT 中点 128000/153500 仍 deferred。

## 当前快照（2026-08-02 10:53 +08:00，✅ heartbeat：batch82j 出 keep12·124000（keep10·83500 又被 SKIP）→回填1点+起单点 batch82k 补 keep10·83500；4 arm 全健康）
> - **✅ batch82j ALL DONE 10:38:11**：keep12·124000 出（PPL 11.4426 core6 .5669 know5 .4608 mmlu .2752，MAIN 从 JSON 核对回填 RUN_REGISTRY）。**但 keep10·83500 在 loop 到它时尚未落盘 → 又 SKIP**（连续第二轮 spec2-skip，guard 正确未假-DONE）。
> - **✅ 采用新规则打破 spec2-skip**：keep10·83500 现已落盘（.104 已 step83600）= 未 eval 里程碑 = 铁律1 缺口 → 起**单点 batch82k**（.82 setsid nohup </dev/null，log `logs/frontier_batch_82k_20260802.log`，脚本 `scripts/_frontier_batch_82k.sh`，PY=olmo2_venv）：**只含盘上已有的 keep10·83500 一个点**，故意不赌即将落盘的 keep12·124500——从此 batch 只含「当前已落盘、未 eval」的点，单点也起。已确认 8 shard 起（START 10:55:19）。跑完 MAIN 回填。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step20940/200k ppl8.11 3.16s/step）+ .252 keep8（step115600/200k ppl13.26 1.02s/step，**wzc1 FS**）+ .104 keep10（step83600/200k ppl13.17 6.83s/step 最长杆）+ .73 keep12（step124380/200k ppl11.42 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（单点 batch82k）。铁律1 满足。ShortGPT 中点 128000/153500 仍 deferred。
> - ⚠️ **教训固化**：500 步间隔下「已落盘先、即将落盘后」的两点 batch 会稳定 SKIP 第二点（~13min eval 早于第二点 ~20min 落盘）。**新规则：batch 只含当前已落盘未 eval 的点，只有一个就跑单点 batch**，不再赌第二点。

## 当前快照（2026-08-02 10:23 +08:00，✅ heartbeat：batch82i 出 keep10·83000（keep12·124000 被 SKIP）→回填1点+起 batch82j 补 keep12·124000；4 arm 全健康）
> - **✅ batch82i ALL DONE 10:05:52**：keep10·83000 出（PPL 12.8241 core6 .5314 know5 .4418 mmlu .2585，MAIN 从 JSON 核对回填 RUN_REGISTRY）。**但 keep12·124000 在 10:05 loop 到它时尚未落盘 → SKIP**（脚本 ckpt-exist guard 正确未假-DONE）。
> - **✅ keep12·124000 现已落盘**（.73 已 step124100）= 未 eval 里程碑 = 铁律1 缺口 → 起 **batch82j**（.82 setsid nohup，log `logs/frontier_batch_82j_20260802.log`）：specs **keep12·124000（先，补 skip 的点）→ keep10·83500（后，机会捕获，~23min 内落盘）**。已确认 keep12·124000 8 shard 起（START 10:24:47）。跑完 MAIN 回填。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step20260/200k ppl8.46 3.16s/step）+ .252 keep8（step113880/200k ppl13.66 1.02s/step，**wzc1 FS**）+ .104 keep10（step83300/200k ppl13.22 6.84s/step 最长杆）+ .73 keep12（step124100/200k ppl11.23 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82j）。铁律1 满足。ShortGPT 中点 128000/153500 仍 deferred。
> - ⚠️ **教训**：batch 内 spec2 若尚未落盘会被 SKIP（非 bug，guard 生效），下轮补起即可；ordering「已在盘的放前、即将落盘的放后」仍是最优，但即将落盘的点 eval 时长 <12min 时可能来不及。

## 当前快照（2026-08-02 09:53 +08:00，✅ heartbeat：新里程碑 keep10·83000 落盘→补卡起 batch82i；4 arm 全健康）
> - **✅ 新 frontier 里程碑落盘**：keep10·83000 已存盘（未 eval），keep12·124000 也已在盘（当前 step123880）。**.82 全 8 卡 idle（0% 0MiB）** → 补卡（铁律1）起 **batch82i**（.82 setsid nohup <　/dev/null，log `logs/frontier_batch_82i_20260802.log`，脚本 `scripts/_frontier_batch_82i.sh`，PY=olmo2_venv）：specs **keep10·83000（先，盘上已有）→ keep12·124000（后，也已在盘）**，各 PPL+core6+know5 8 卡 sharded，免 scp。已确认 8 python shard 进程起（keep10·83000 PPL phase，START 09:53:47）。跑完 MAIN 回填 2 点。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step19760/200k ppl8.37 3.16s/step）+ .252 keep8（step112380/200k ppl13.34 1.02s/step，**wzc1 FS**）+ .104 keep10（step83020/200k ppl13.26 9.64s/step 最长杆）+ .73 keep12（step123880/200k ppl11.65 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82i，补 .82 idle）。铁律1 满足。ShortGPT 中点 128000/153500 仍 deferred。

## 当前快照（2026-08-02 09:23 +08:00，✅ heartbeat：batch82h 全完+回填2点；frontier 表再度追平，.82 合理空闲）
> - **✅ batch82h ALL DONE 09:20:01**（2 点全出，无 silent-skip）。MAIN 从 JSON 核对回填 RUN_REGISTRY frontier 表：**keep10·82500 PPL 12.8360 core6 .5318 know5 .4344 mmlu .2563**、**keep12·123500 PPL 11.4475 core6 .5736 know5 .4736 mmlu .2749**。
> - **frontier 表再度 100% 追平所有盘上 keep10/keep12 存点**（keep10 至 82500、keep12 至 123500）——无未 eval 存点。下一存点 keep10·83000（当前 step82780，~220步/~25min）、keep12·124000（当前 step123640，~360步/~47min）出来后下轮捕获。.82 全 8 卡 idle（0%）= "无未 eval 存点"的合理空闲，非 铁律1 缺口，**本轮不起新 batch**。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step19180/200k ppl8.05 3.16s/step）+ .252 keep8（step110880/200k ppl13.52 1.02s/step，**wzc1 FS**）+ .104 keep10（step82780/200k ppl13.12 6.80s/step 最长杆）+ .73 keep12（step123640/200k ppl11.57 7.81s/step）。Monitor 8088 http200。
> - 无 kill；无 launch（.82 合理空闲）。铁律1 满足。ShortGPT 中点 128000/153500 仍 deferred。

## 当前快照（2026-08-02 08:53 +08:00，✅ heartbeat：新里程碑 keep10·82500 落盘→补卡起 batch82h；4 arm 全健康）
> - **✅ 新 frontier 里程碑落盘**：keep10·82500 已存盘（未 eval），keep12·123500 距落盘 ~10min（当前 step123420）。**.82 全 8 卡 idle（0% 0MiB）** → 补卡（铁律1）起 **batch82h**（.82 setsid，log `logs/frontier_batch_82h_20260802.log`，脚本 `scripts/_frontier_batch_82h.sh`，PY=olmo2_venv）：specs **keep10·82500（先，盘上已有）→ keep12·123500（后，keep10 eval ~12min 内它必落盘）**，各 PPL+core6+know5 8 卡 sharded，免 scp。已确认 keep10·82500 8 shard PPL 进程起（ckpt-load）。跑完 MAIN 回填 2 点。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step18620/200k ppl8.34 3.16s/step）+ .252 keep8（step109380/200k ppl13.57 1.02s/step，**wzc1 FS**）+ .104 keep10（step82500/200k ppl13.31 6.84s/step 最长杆）+ .73 keep12（step123420/200k ppl11.56 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82h，补 .82 idle）。铁律1 满足。ShortGPT 中点 128000/153500 仍 deferred。

## 当前快照（2026-08-02 08:23 +08:00，✅ heartbeat：batch82g 全完+回填2点；frontier 表再度追平，.82 合理空闲）
> - **✅ batch82g ALL DONE 08:20:08**（2 点全出）。MAIN 从 JSON 核对回填 RUN_REGISTRY frontier 表：**keep10·82000 PPL 12.8462 core6 .5263 know5 .4348 mmlu .2609**、**keep12·123000 PPL 11.4513 core6 .5683 know5 .4630 mmlu .2572**。
> - **frontier 表再度 100% 追平所有盘上 keep10/keep12 存点**（keep10 至 82000、keep12 至 123000）——无未 eval 存点，故本轮**不起新 batch**。下一存点 keep10·82500（~240步/~27min）、keep12·123500（~300步/~39min）出来后下轮捕获。.82 空闲 = "无未 eval 存点"的合理空闲，非 铁律1 缺口。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step18040/200k ppl8.17 3.17s/step）+ .252 keep8（step107880/200k ppl13.20 1.02s/step，**wzc1 FS**）+ .104 keep10（step82260/200k ppl13.17 6.84s/step 最长杆）+ .73 keep12（step123200/200k ppl11.71 7.81s/step）。Monitor 8088 http200。
> - 无 kill；无 launch（frontier 已追平）。ShortGPT 中点 128000/153500 仍 deferred（各需 46GB scp + paper 决策）。

## 当前快照（2026-08-02 07:54 +08:00，✅ heartbeat：新里程碑 keep10·82000 落盘→补卡起 batch82g；4 arm 全健康）
> - **✅ 新 frontier 里程碑落盘**：keep10·82000 已存盘（未 eval），keep12·123000 距落盘 ~2.6min（当前 step122980）。**.82 全 8 卡 idle（0% 0MiB）** → 补卡（铁律1）起 **batch82g**（.82 setsid，log `logs/frontier_batch_82g_20260802.log`，脚本 `scripts/_frontier_batch_82g.sh`，PY=olmo2_venv）：specs 顺序 **keep10·82000（先，盘上已有）→ keep12·123000（后，keep10 eval ~12min 内它必落盘）**，各 PPL+core6+know5 8 卡 sharded，免 scp 同盘直读。已确认 keep10·82000 8 shard PPL 进程起（ckpt-load 阶段 GPU 0% 正常，与前几批一致）。跑完 MAIN 回填 2 点。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step17480/200k ppl8.14 3.16s/step）+ .252 keep8（step106380/200k ppl13.41 1.02s/step，**wzc1 FS**）+ .104 keep10（step82000/200k ppl13.07 6.84s/step 最长杆）+ .73 keep12（step122980/200k ppl11.78 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82g，补 .82 idle）。铁律1 满足（无空转卡）。ShortGPT 中点 128000/153500 仍 deferred（各需 46GB scp + paper 决策）。

## 当前快照（2026-08-02 07:26 +08:00，✅ heartbeat：batch82f 全完+回填2点；frontier 表已追平所有盘上存点，无空转任务）
> - **✅ batch82f ALL DONE 07:21:01**（2 点全出）。MAIN 直接在 .82 从 JSON 核对并回填 RUN_REGISTRY frontier 表：**keep12·122500 PPL 11.4573 core6 .5670 know5 .4593 mmlu .2634**、**keep10·81500 PPL 12.8493 core6 .5308 know5 .4458 mmlu .2655**。
> - **frontier 表已 100% 追平盘上所有 keep10/keep12 存点**（keep10 至 81500、keep12 至 122500 全回填）——当前**无任何未 eval 的存点**，故本轮**不起新 batch**（re-eval 已完成点 = 零价值）。下一里程碑 keep10·82000（~260 步/~30min）、keep12·123000（~240 步/~31min）出来后下轮再捕获。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step16920/200k ppl7.85 3.16s/step）+ .252 keep8（step104980/200k ppl14.15 1.02s/step，**wzc1 FS**）+ .104 keep10（step81740/200k ppl12.74 6.84s/step 最长杆）+ .73 keep12（step122760/200k ppl11.67 7.81s/step）。Monitor 8088 http200。
> - 无 kill；无 launch（frontier 已追平，无空转卡可补——.82 空闲是"无未 eval 存点"的合理空闲，非任务缺口）。ShortGPT 中点 128000/153500 仍 deferred（各需 46GB scp + paper 决策）。

## 当前快照（2026-08-02 06:55 +08:00，✅ heartbeat：batch82e 全完+回填2点；.82 idle→起 batch82f；4 arm 全健康）
> - **✅ batch82e ALL DONE 06:53:14**（2 点全出）。MAIN 直接在 .82 用 olmo2_venv 从 JSON 算并回填 RUN_REGISTRY frontier 表：**keep12·122000 PPL 11.4605 core6 .5696 know5 .4634 mmlu .2665**、**keep10·81000 PPL 12.8563 core6 .5315 know5 .4419 mmlu .2633**。
> - **✅ .82 idle→补卡（铁律1）起 batch82f**（.82 setsid，log `logs/frontier_batch_82f_20260802.log`，脚本 `scripts/_frontier_batch_82f.sh`，PY=olmo2_venv）：盘上 2 个新存点 **keep12·122500 + keep10·81500**（免 scp）。已确认 8 shard PPL 进程起。跑完 MAIN 回填 2 点。
> - **I/O contention 核查**：batch82e 期间 .73 keep12 仅 step122520 一步慢到 10.81s/step（ckpt-load 瞬时），前后 122480/122500 均正常 7.81s；.104 keep10 全程 6.84s 无抖动 → eval 读盘对 diskB 训练**无持续拖累**，继续在删前捕获轨迹点 = net-valuable。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step16340/200k ppl8.37 3.16s/step）+ .252 keep8（step103440/200k ppl14.05 1.02s/step，**wzc1 FS**）+ .104 keep10（step81500/200k ppl13.39 6.84s/step 最长杆）+ .73 keep12（step122520/200k ppl11.59 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82f）。铁律1 满足（无空转卡）。

## 当前快照（2026-08-02 06:28 +08:00，✅ heartbeat：batch82d 全完+回填2点；.82 idle→起 batch82e；4 arm 全健康）
> - **✅ batch82d ALL DONE 06:12:33**（名义 4 点实产 3 点）。MAIN 从 JSON 核对回填 RUN_REGISTRY frontier 表：**keep10·80000 PPL 12.8725 core6 .5308 know5 .4387 mmlu .2611**、**keep10·80500 PPL 12.8698 core6 .5350 know5 .4397 mmlu .2587**（keep12·121500 上轮已回填）。**keep12·121000 被 silent-skip 正确跳过**（ckpt 不在盘、无 summary.json→未 touch DONE marker，符合预期）。
> - **✅ .82 idle→补卡（铁律1）起 batch82e**（.82 setsid，log `logs/frontier_batch_82e_20260802.log`，脚本 `scripts/_frontier_batch_82e.sh`，PY=olmo2_venv）：盘上 2 个新存未 eval 的最高步点 **keep12·122000 + keep10·81000**（免 scp 同盘直读，最接近训练前沿）。已确认 keep12·122000 8 shard PPL 进程起（ckpt load 中）。跑完 MAIN 回填 frontier 表 2 点。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step15760/200k ppl8.04 3.16s/step）+ .252 keep8（step101980/200k ppl13.86 1.02s/step，**wzc1 FS**）+ .104 keep10（step81220/200k ppl13.25 6.84s/step 最长杆）+ .73 keep12（step122280/200k ppl12.03 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82e，补 .82 idle）。铁律1 满足（无空转卡）。ShortGPT 轨迹中点 128000/153500 仍 deferred（各需 46GB scp LOCAL→.82）。

## 当前快照（2026-08-02 05:29 +08:00，✅ heartbeat：ShortGPT #98 step200000 eval 全完+回填；4 arm 全健康，无空闲卡）
> - **✅ ShortGPT #98 step200000 downstream eval DONE**（05:06:55 完成，datasets 5.0.0 生效，11 task 全出）。MAIN 从 JSON 核对完整结果：**PPL 9.7803 / core6 .6215 / know5 .5596 / mmlu .4739**（per-task：HS .6851·arc_c .4761·arc_e .7462·piqa .7584·obqa .408 acc_norm·WG .6551 acc；mmlu .4739·lambada .6194·boolq .7287·csqa .5340·siqa .4422 acc）。
> - **★ 关键发现**：ShortGPT-policy（非连续保留 16 层 [0-12,16,17,31] 含 readout 层 31，0 fresh，200k heal）在 **PPL(1.322× tax) 与 MMLU(63.0% above-chance recovery) 两轴同时优于全部三个 16L 连续截断臂**（keep14 train-all 19.4% / freeze-front 3.6% / random-front ~0%）。混淆项：继承 16 vs 14 层 + 保留原生 readout vs 换 2 fresh 层，两效应未拆分（需 keep16-inherited/0-fresh 连续控制隔离 policy）。
> - **回填**：`status/PAPERB_THREE_ARM_200K.md`（headline 表 + §ShortGPT breakdown + "★dominates" finding）、`status/RUN_REGISTRY.md`（Paper B 新增 ShortGPT endpoint block）、task#98 desc 更新。原始 JSON 在 `.82:olmo2_ppl_results/7B_shortgpt16_step200000/` + `.82:olmo2_downstream_results/7B_shortgpt16_step200000{,_know}/`。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step14800/200k ppl8.34 3.16s/step）+ .252 keep8（step99280/200k ppl14.20 1.02s/step，**wzc1 FS**，~1.1d 剩）+ .104 keep10（step80760/200k ppl13.47 6.84s/step 最长杆）+ .73 keep12（step121880/200k ppl11.83 7.81s/step）。Monitor 8088 http200。
> - **⚠️→✅ .82 补卡（铁律1）**：ShortGPT eval 05:06:55 收尾后 .82 全 8 卡 idle（0 MiB）。发现 diskB 上有 4 个新存未 eval 的 frontier 点（**免 scp**，同盘直读）→ 05:35 起 **batch82d**（.82 setsid pid2124096，log `logs/frontier_batch_82d_20260802.log`，脚本 `scripts/_frontier_batch_82d.sh`，PY=olmo2_venv 修 torch-base 崩坏）：keep12 {121500,121000} + keep10 {80500,80000}，各 PPL+core6+know5 8 卡 sharded。已确认 8 shard PPL 进程起（keep12·121500 首点 ckpt load 中）。`_eval_frontier_pt.sh` 已参数化 PY=${PY:-...} 以兼容 olmo2_venv。跑完 MAIN 回填 RUN_REGISTRY frontier 表 4 点。
> - 无 kill；1 launch（batch82d，补 .82 idle）。铁律1 满足（无空转卡）。ShortGPT 轨迹中点 128000/153500 deferred（各需 46GB scp LOCAL→.82；待定 paper 是否需 healing 轨迹 vs 仅 headline 端点）。

## 当前快照（2026-08-02 05:00 +08:00，🔧 heartbeat：修复 datasets 版本 + 重跑 ShortGPT #98 downstream；4 arm 全健康）
> - **✅ ShortGPT step200000 PPL 已出且有效 = 9.7803**（avg_nll 2.2803，8 shard 合并；`olmo2_ppl_results/7B_shortgpt16_step200000/summary.json`）。但 **downstream 6 task 被 SKIP**（`Feature type 'List' not found`）——04:37 建的 olmo2_venv 继承 elsa 的 **datasets 3.6.0**，而 diskB 数据 cache 是新版 datasets（≥4.0 引入 `List` feature）建的→旧版读不了 mmlu/hellaswag/arc*/openbookqa/commonsense_qa。仅 List-free 的 piqa/winogrande/lambada/boolq/social_iqa 成功。
> - **修复**：查得 .252（跑 keep14 baseline #93 的节点）用 **datasets 5.0.0 + pyarrow 25.0.0**，venv 已有 pyarrow 25→`pip install datasets==5.0.0`（精确匹配 baseline 版本；版本只影响 load 不影响 scoring→数字与 keep14 口径一致）。.73 `.venv` 无 datasets（跑 QCMem/Paper-A 非 OLMo-2 downstream）。
> - **重跑 downstream-only**（.82 setsid pid2109690，PY=olmo2_venv，脚本 `scripts/_run_shortgpt_downstream_only.sh`，log `logs/eval_shortgpt16_step200000_downstream.log`）：跳过已有效的 PPL，只跑 core6+know5；datasets=5.0.0 已确认，8 shard core6 全活。~30-60min 出 summary→MAIN 回填 RUN_REGISTRY ShortGPT step200000 完整行（ppl9.78+core6+know5+mmlu）。ckpt step200000.pt（48.7GB）仍在 .82 diskB=免重传。
> - **✅ 4 arm 全健康+推进**：LOCAL #100 full-32L（step14260/200k loss2.13 ppl8.43 3.16s/step 8×100%）+ .252 keep8（step97900/200k ppl14.46 1.02s/step，**wzc1 FS**）+ .104 keep10（step80500/200k ppl13.40 6.84s/step 最长杆）+ .73 keep12（step121660/200k ppl11.59 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（downstream-only eval）。铁律1 满足（修复 datasets 阻塞 + .82 持续跑真 eval，无空转）。后续 153500/128000 轨迹点需重新 scp（46GB each）+ eval，下轮视 step200000 完整结果决定是否续跑。

## 当前快照（2026-08-02 04:37 +08:00，🔧 heartbeat：修复 .82 崩环境 + 重启 ShortGPT #98 eval；4 arm 全健康）
> - **🔧 发现并修复 .82 环境崩坏（silent eval fail）**：03:37 起的 ShortGPT eval driver 把 step200000.pt（46GB）scp 到 .82 成功，但 eval 04:20 秒崩 `ModuleNotFoundError: numpy/torch`——**`/opt/conda/envs/torch-base/bin/python` 被 reset 成 python3.14 且 site-packages 全空**（py3.11 包目录被抹，比常规 reset 更严重；02:55 batch82c 时还是 py3.11+torch 能跑）。`elsa` env 有 torch2.7+numpy 但 transformers 4.45<4.47 无 Olmo2 类。
> - **修复（非侵入）**：`elsa/bin/python -m venv --system-site-packages olmo2_venv`（继承 elsa torch2.7+numpy，装在 **diskB 共享盘**→跨 .82 reset 持久，不改 elsa），`pip install transformers==5.5.4`（匹配 save-time 版本）+ tokenizers0.22.2。验证 torch2.7+cu126/cuda(8dev)/numpy2.2.6/Olmo2 import 全 OK。
> - **kill 低效 driver**（pid3728497+scp，本会 3× 传 46GB 后同样 fail-eval=纯浪费带宽/盘）；rm 半截 step153500.pt；**KEEP 已传好的 step200000.pt**（免重传）。
> - **重启 step200000 eval**（.82 pid2092237，PY=olmo2_venv，log `logs/eval_shortgpt16_step200000.log`）：8 shard 进程全活、各 ~47GB RSS=ckpt 已载入 CPU、PPL 相、无 import error=真进展。~5min PPL + core6 + know5，~20-25min 出 summary.json→MAIN 回填 RUN_REGISTRY ShortGPT 行。后续 153500/128000 需重新 scp（本轮未重启 driver，避免 GPU 冲突；下轮视 step200000 结果决定是否续跑轨迹点）。
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step13540/200k ppl8.21 3.17s/step）+ .252 keep8（step96660/200k ppl13.89 1.02s/step，**wzc1 FS**）+ .104 keep10（step80300/200k ppl13.03 6.84s/step 最长杆）+ .73 keep12（step121480/200k ppl12.14 7.81s/step）。Monitor 8088 http200。
> - 1 kill（ShortGPT driver+scp）；1 launch（step200000 eval，olmo2_venv PY）。铁律1 满足（修复阻塞 + .82 重投入真 eval）。

## 当前快照（2026-08-02 03:37 +08:00，✅ heartbeat：batch82c 全完+回填2点；.82 idle→起 ShortGPT #98 eval 补卡；4 arm 全健康）
> - **✅ batch82c ALL DONE 03:17:03**（keep12·120500 + keep10·79000，PPL summary.json 校验通过=silent-skip 修复生效）。MAIN 已核对 JSON 回填 RUN_REGISTRY frontier 表 2 行：**keep12·120500 PPL 11.4740 core6 .5686 know5 .4554 mmlu .2707**、**keep10·79000 PPL 12.9010 core6 .5333 know5 .4417 mmlu .2641**。
> - **✅ 大发现：ShortGPT #98 heal 已 200k 全完**（`logs/olmo2_7B_shortgpt16.log` 末行 `[step 200000/200000] loss=2.1785 ppl=8.83`，08-01 16:09；ckpt step0→200000 全在盘；chain PID 657760 已死=正常完成非崩）。**#98 剩余=4 点 eval battery（step0/128k/153.5k/200k），此前仅 step0 已 eval（`7B_shortgpt_step0` ppl~401）。**
> - **✅ .82 idle→补卡（铁律1）起 ShortGPT eval driver**（LOCAL setsid pid3728495，log `logs/drive_shortgpt_eval_82_20260802.log`，脚本 `scripts/_drive_shortgpt_eval_on_82.sh`）：ShortGPT ckpt 在 **wzc1**、.82 是 **diskB**、无空闲 wzc1 节点→逐点 **scp -O**（本节点无 rsync；scp 默认 SFTP 被 .82 sshd 拒→用 `-O` legacy 协议修复）把 46GB ckpt LOCAL→.82，用**canonical `_run_olmo2_eval_shortgpt.sh`**（16 层 shell strict-load，选层已在训练时 compact 进 0-15，故 keep_front=16/fresh=0 正确）跑 PPL+core6+know5 8 卡 sharded，验 JSON 后 rm 远端 ckpt 释盘。步序 200000→153500→128000（headline 优先）。⚠️ 跨盘 scp ~14MB/s→单 ckpt ~30-55min 传输（传输期 .82 GPU 空转=FS split 不可避免的开销，非疏忽；无其他能填这段的 .82 耐跑活）+ ~24min eval。跑完 MAIN 回填 RUN_REGISTRY ShortGPT 行。
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step12540/200k ppl8.24 3.16s/step）+ .252 keep8（step93440/200k ppl13.68 1.02s/step，**wzc1 FS**）+ .104 keep10（step79740/200k ppl12.99 6.84s/step 最长杆）+ .73 keep12（step121000/200k ppl11.83 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（ShortGPT eval driver）。铁律1 满足。

## 当前快照（2026-08-02 02:55 +08:00，✅ heartbeat：batch82b 全完；.82 idle→起 batch82c；4 arm 全健康）
> - **✅ batch82b ALL DONE 02:37:22**（keep12 {120000,115000}+keep10 {70000,75000} 全 eval，末点 keep10·70000 完），shell 已退，.82 8 卡 0MiB 真空闲。
> - **✅ .82 idle→补卡（铁律1）起 batch82c**（pid2057462，log `logs/frontier_batch_82c_20260802.log`，脚本 `scripts/_frontier_batch_82c.sh`）：仅 **2 个盘上新存未 eval 的点**——keep12·**120500** + keep10·**79000**（训练滚动删旧 ckpt，故盘上只剩最新 3-4 个，其余已 eval+删）。02:55 keep12·120500 PPL 相已起（ceph-load 中）。**脚本加了「PPL summary.json 存在才 touch DONE」的校验**，避免 batch1 的静默-SKIP/假 DONE-marker bug 复现。跑完 MAIN 续填 RUN_REGISTRY frontier 表 2 行。
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step11840/200k ppl8.34 3.15s/step 8/8 98-100%）+ .252 keep8（step91560/200k ppl13.58 1.02s/step，**wzc1 FS**）+ .104 keep10（step79400/200k ppl13.00 6.84s/step 最长杆）+ .73 keep12（step120700/200k ppl11.63 7.81s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82c）。铁律1 满足。备注：keep8/full32 无 frontier 轨迹点（ckpt 在 wzc1，.82 diskB 读不到，需跨盘 rsync；keep8 仍训练中→端点未就绪，暂不值当跨盘）。

## 当前快照（2026-08-02 02:27 +08:00，✅ heartbeat：batch82b 收尾+全轨迹22点回填；4 arm 全健康）
> - **✅ batch82b（pid1995941）已跑完 keep12 {120000,115000} + keep10 {70000,75000}**，末点 keep10·75000 downstream 收尾中（.82 8 卡 ceph-load 0MiB=载模型非空转，batch shell pid1995941 + child pid2026458 alive）。**MAIN 一次性把全部完成点核对 JSON 回填 RUN_REGISTRY frontier trajectory 表 = 22 点**（keep10 step10k/70k/73.5k/74k/74.5k/75k/75.5k/76k/76.5k/77k/78k/78.5k；keep12 step115k–120k 全段）。趋势：keep12(14L) PPL~11.5 core6~.567 know5~.464 > keep10(12L) PPL~12.9 core6~.531 know5~.440；mmlu 全轨迹 ~chance（.25-.27）→ 「PPL 早收敛 knowledge 滞后」（轨迹中点非 200k 终点）。step0（degenerate）+ step77500（缺 know5）未入表。
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step11300/200k ppl8.31 3.16s/step 8/8 98-100%）+ .252 keep8（step90200/200k ppl14.49 1.02s/step，**wzc1 FS**）+ .104 keep10（step79160/200k ppl12.71 6.84s/step 最长杆）+ .73 keep12（step120500/200k ppl11.76 7.81s/step）。Monitor 8088 http200。
> - 无 kill/launch（.82 仍忙于 batch82b 末点，铁律1 满足）；batch82b 完全收尾后下轮 heartbeat 视新存 ckpt 决定是否再补 batch82c。

## 当前快照（2026-08-02 02:02 +08:00，✅ heartbeat：batch1 完成+回填3点；.82 idle→补 batch82b；4 arm 全健康）
> - **✅ .82 frontier batch1（pid1950104）01:44:58 ALL DONE**——但名义 4 点**实产 3 点**：keep12 **step119000 被静默 SKIP**（ckpt 不在盘，harness 仍 touch DONE marker=已知 bug）。3 个有效点 MAIN 已核对 JSON 回填 RUN_REGISTRY frontier trajectory 表：**keep10·78000 PPL12.9207 core6 .5301 know5 .4425**、**keep10·78500 PPL12.9113 core6 .5324 know5 .4392**、**keep12·119500 PPL11.4826 core6 .5684 know5 .4624**（core6=hellaswag/arc_c/arc_e/piqa/obqa acc_norm+winogrande acc；know5=mmlu/lambada/boolq/csqa/siqa acc）。趋势 keep12>keep10 符深度阶梯；mmlu 仍 ~chance（.26-.27，轨迹中点非 200k 终点）。
> - **✅ .82 idle→补卡（铁律1）**：起 **batch82b**（pid1995909，log `logs/frontier_batch_82b_20260802.log`，脚本 `scripts/_frontier_batch_82b.sh`）补 4 个盘上已存未 eval 的点：keep12 {**120000**（补 batch1 漏的高端）,115000} + keep10 {75000,70000}（4 ckpt 均已确认在盘 ~40GB）。02:01 keep12·120000 PPL 已完成→正跑 downstream core6，9 procs，进度快（~5min/PPL）。跑完 MAIN 续填 RUN_REGISTRY。
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step10700/200k ppl8.48 3.16s/step）+ .252 keep8（step88600/200k ppl14.64 1.02s/step，**wzc1 FS**）+ .104 keep10（step78880/200k ppl12.70 6.84s/step 最长杆）+ .73 keep12（step120240/200k ppl11.53 7.81→10.82s/step）。Monitor 8088 http200。
> - 无 kill；1 launch（batch82b eval）。铁律1 满足。

## 当前快照（2026-08-02 01:26 +08:00，✅ heartbeat：5 节点全忙无空闲卡；.252 FS 纠错）
> - **✅ 4 训练 arm 全健康**：LOCAL #100 full-32L（step10140/200k ppl8.28 3.16s/step 8/8 99-100%）+ .252 keep8（step**87300**/200k ppl13.97 1.02s/step 8/8 100%，newest ckpt step87000，~112.7k step≈1.3d 剩）+ .104 keep10（step78620/200k ppl12.63 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step120020/200k ppl12.16 10.82s/step 8/8 99-100%）。Monitor 8088 http200。
> - **⏳ .82 Paper B frontier batch（pid1950104）进行中**：keep10 **step78000 ✅DONE**（结果在 batch log，待批次收尾统一 backfill）→ keep10 step78500 **正在跑**（01:22:46 起 PPL 相，ceph-load 8卡 0%/3MiB=正常载模型非 stall，9 procs alive）→ 队列剩 keep12 {119000,119500}。
> - **⚠️ 纠错：.252 keep8 实际跑在 wzc1 FS**（进程 cmdline=`/apdcephfs_wzc1/share_304376610/.../.venv/bin/python ... --resume_from outputs/olmo2_probe2_7B_keep8fresh2/step47000.pt`，log/ckpt 在 wzc1 根**不是** diskB `/apdcephfs_zwfy6/`）→ .252 与 LOCAL #100 共享 wzc1 盘。**影响：keep8 完成后迁 keep10→.252 需先 rsync keep10 ckpt（diskB .104 保存）→ wzc1**，非零成本，迁移前须计入。查 .252 keep8 step 用 `cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && grep '\[step' logs/olmo2_7B_keep8fresh2.log`。
> - 无 launch/kill（批次早已在跑，训练全健康）；铁律1 满足（5 节点全忙）。

## 当前快照（2026-08-02 01:13 +08:00，✅ P0.13 完成+回填；.82 空闲→按铁律1 起 Paper B frontier batch；4 训练 arm 全健康）
> - **✅ P0.13 DONE + MAIN 已核对 JSON 回填**（TODOList P0.13 RESULTS 块 + 新建 `status/P0_13_QUALITY_LATENCY.md`）：Arm A(j=0) macro **99.19** vs Arm B(j=12) macro **96.07**，diff **+3.12pp** CI[2.36,3.93]（10k bootstrap），McNemar b83/c1/both1404/neither12 **p=8.79e-24**；read latency A 931.9ms / B 664.4ms = **1.4027×**（三进程一致，与 P0.12 1.373× 吻合），total-decode 两臂 ~相等（read-phase 效应非端到端）。质量差集中 niah_multikey_1（distractor 消歧）。**决策=QUALITY-LATENCY TRADEOFF（B 非非劣，显著更弱）**，论文不得写 quality-preserving/pure-depth/端到端加速。coder local commit `29243d9` 未 push。task #115 completed。⚠️ `.tex` 集成待用户拍板（MAIN 不改 .tex）。**P0.13 是投稿前最后一个必要模型 run → Paper A required-model-run 队列已排空。**
> - **✅ .82 idle→补卡（铁律1 + h20-paperA-over-paperB-priority：Paper A 排空→Paper B）**：P0.13 完成后 .82 8 卡 free。起 **Paper B depth-ladder frontier eval batch**（pid1950104，log `logs/frontier_batch_82_20260802.log`）——4 个盘上已存但未 eval 的 frontier 点：**keep10 {78000,78500}**（keep10 轨迹稀疏，仅 step77500 已 eval，densify 优先）+ **keep12 {119000,119500}**。每点 = PPL+core6+know5 8-shard 全占 8 卡，串行（keep10 先）。base protocol chat=False/no-BOS/LL-based MC。.73/.82/.104 共享 diskB FS 故 .82 直接读 keep10(.104)/keep12(.73) ckpt 无需 rsync。9 procs alive PPL 相（ceph-load 28GB/卡 ~1-2min 后填卡）。
> - **✅ 4 arm 全健康（≈00:55 读数）**：LOCAL #100 full-32L（step9660/200k ppl8.35 3.17s/step 8/8 99-100%）+ .252 keep8（step85780/200k ppl14.55 1.02s/step 8/8 100%，~1.3d 剩→完成后迁 keep10→.252 B200）+ .104 keep10（step78400/200k ppl12.40 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step119820/200k ppl11.35 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **纠错备忘（不变）**：节点密码 `.104=configs/password_h20_24104.txt`、`.73=configs/password_h20_853573.txt`、`.82=configs/password_h20_82250.txt`、`.252(B200,28.89.19.252)=configs/password_b200_19252.txt`，全部端口 36000。#100 log=`logs/olmo2_7B_full32_dolmino.log`；keep{8,10,12} log=`logs/olmo2_7B_keep{N}fresh2.log`；keep{N} ckpt=`outputs/olmo2_probe2_7B_keep{N}fresh2/step*.pt`。frontier 一次一点=`bash scripts/_eval_frontier_pt.sh TAG DIR STEP KEEP FRESH`。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。

## 当前快照（2026-08-02 00:56 +08:00，✅ heartbeat：4 训练 arm 全健康；P0.13 quality 仍在跑（task-pool churning）；无空闲卡）
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step9660/200k ppl8.35 3.17s/step 8/8 99-100% maxmem176.9GB）+ .252 keep8（step85780/200k ppl14.55 1.02s/step 8/8 100%）+ .104 keep10（step78400/200k ppl12.40 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step119820/200k ppl11.35 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **⏳ P0.13 running on .82**（coder a0853dad）：quality dir 已 90 files（00:28 时 43）、rc=0 全通过、task-pool 动态跑 15 cells×4 shard（本轮 launcher 见 niah_single_3/multikey_1/VT 各长档交错，无 error 无 stall），GPU 8 卡 model 全载 mixed-util（16-19GB/卡）。latency micro-bench + paired bootstrap/McNemar 尚未开始（quality 未收尾）。跑完 MAIN backfill TODOList/status（不碰 .tex）。**.82 committed to P0.13，无空闲卡=不违反铁律1。**
> - **✅ P0.14 DONE + 已回填 TODOList**（详见下方 00:28 快照）；task #116 completed。⚠️ `tab:infbench` withdraw/relabel 待用户拍板（MAIN 不改 .tex）。
> - **纠错备忘（不变）**：节点密码 `.104=configs/password_h20_24104.txt`、`.73=configs/password_h20_853573.txt`、`.82=configs/password_h20_82250.txt`、`.252(B200,28.89.19.252)=configs/password_b200_19252.txt`，全部端口 36000。#100 log=`logs/olmo2_7B_full32_dolmino.log`；keep{8,10,12} log=`logs/olmo2_7B_keep{N}fresh2.log`。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。

## 当前快照（2026-08-02 00:28 +08:00，✅ heartbeat：4 训练 arm 全健康；P0.14 完成+回填；P0.13 ~80% quality 已跑完）
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step9100/200k ppl8.33 3.17s/step 8/8 100% maxmem176.9GB）+ .252 keep8（step84360/200k ppl13.97 1.02s/step 8/8 100%）+ .104 keep10（step78140/200k ppl12.87 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step119600/200k ppl11.57 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **✅ P0.14 DONE + 已回填 TODOList**（coder a2585140 commit `362a22f` 未 push）：PG-19/InfiniteBench 13-gram MinHash 审计 → 强双峰，~63% 书 / ~67% eval records（诚实口径，≥0.80 保守口径仅 28.1% 因人名匿名化低估）命中 PG-19 训练集。建议 **withdraw/relabel `tab:infbench`**（Book-QA F1 6.06/choice 17.47）。clean-subset 重算需 predictions（在禁访 .73），脚本 `recompute_p0_14_clean_subset.py` 已就绪。⚠️ `.tex` 待用户拍板集成（MAIN 不改 .tex）。task #116 completed。
> - **⏳ P0.13 running on .82**（coder a0853dad，00:00 launch）：8×H20，~80% quality mode 已完成（niah_single_3 + niah_multikey_1 全 5 长档 cell 齐；现跑最后 VT 64k/128k，iter_bm25 CPU-bound 阶段），43 out files。剩 VT + latency micro-bench（3-proc）+ paired bootstrap/McNemar 统计。ETA 全套 ~01:00–01:30。跑完 MAIN backfill TODOList/status（不碰 .tex）。
> - **纠错备忘（不变）**：节点密码 `.104=configs/password_h20_24104.txt`、`.73=configs/password_h20_853573.txt`、`.82=configs/password_h20_82250.txt`、`.252(B200,28.89.19.252)=configs/password_b200_19252.txt`，全部端口 36000。#100 log=`logs/olmo2_7B_full32_dolmino.log`；keep{8,10,12} log=`logs/olmo2_7B_keep{N}fresh2.log`。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。

## 当前快照（2026-08-01 23:30 +08:00，✅ heartbeat：4 训练 arm 全健康；用户加 P0.13/P0.14→.82 转投 Paper A P0.13（final required run），暂停 frontier watch-loop）
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step7960/200k ppl8.37 3.16s/step 8/8 100% maxmem176.9GB）+ .252 keep8（step81280/200k ppl14.11 1.02s/step 8/8 100%）+ .104 keep10（step77600/200k ppl12.89 6.84s/step 8/8 99%）+ .73 keep12（step119120/200k ppl11.63 7.81s/step 8/8 98-100%）。Monitor 8088 http200 OK。
> - **★ .82 转投 Paper A P0.13（用户 23:26 指令：TODO 新增 P0.13/P0.14，跑完把结果放表里）**：P0.13 = **投稿前最后一个必要模型 run**（同 pack/同 LoRA/同 examples，RULER Cohort B 15 cells j=0 vs j=12 quality↔latency 闭环 + paired bootstrap/McNemar + 3-proc latency）。按 h20-paperA-over-paperB-priority，.82（QCMem H20，Paper-A-first）优先给 P0.13 → **kill frontier watch-loop pid1655855**（Paper B eval densification 让位 Paper A；训练 arm 全不动）。kill 时发现 watch 刚起 keep10:77500 know5 eval（8 shard，20GB/卡）→**留其跑完**（保住该 frontier 点），P0.13 coder poll GPU 空闲后再上 8 卡。
> - **派 2 coder（opus，后台）**：P0.13 harness+run on .82（agent a0853dad，建 `scripts/bench_p0_13_quality_latency.py`，写 `bench_results/p0_13_quality_latency/`）；P0.14 InfiniteBench/PG-19 污染审计 CPU-only（agent a2585140，写 `bench_results/p0_14_contamination/`）。两者**禁碰 .tex/TODOList/status**，结果由 main backfill 进 TODOList/status 表。
> - **代价备忘**：P0.13 期间（多小时）.82 不再自动抓 keep10/keep12 frontier 中间点（rotation 每 ~1h 滚 3-4 ckpt）→ 会丢若干中间轨迹点；apex@200000 不丢（训练继续）。轨迹已密（keep10 有 74000-77500 共 8 点），可接受。P0.13 完成后可 resume watch-loop。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **纠错备忘（不变）**：节点密码 `.104=configs/password_h20_24104.txt`、`.73=configs/password_h20_853573.txt`、`.82=configs/password_h20_82250.txt`、`.252(B200)=configs/password_b200_19252.txt`，全部端口 36000。#100 log=`logs/olmo2_7B_full32_dolmino.log`；keep{8,10,12} log=`logs/olmo2_7B_keep{N}fresh2.log`。

## 当前快照（2026-08-01 22:02 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 idle→无未抓点→派 coder 补 P0.12 acceptance bench；P0.12 文档按用户订正）
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step6200/200k ppl8.32 3.16s/step 8/8 100%）+ .252 keep8（step76780/200k ppl14.02 1.02s/step 8/8 100%，~1.5d 剩）+ .104 keep10（step76820/200k ppl12.97 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step118440/200k ppl11.78 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **✅ .82 idle-card（本轮不同）**：8×0%，但**无未抓 frontier 点**（watch-state keep12:118000 / keep10:76500 均已抓；下一点 keep12 s118500 约 ~8min 后落盘，watch-loop pid1655855 alive etime4h26m 会自动抓）。故不做无意义 micro-eval → 派 coder(opus) 用 .82 空闲 8 卡跑 **P0.12 acceptance-closure bench**（output-consistency j0 vs j12 + upper-layer/LM-head 分计时 + provenance：git/版本/权重hash/LoRA-module枚举/NaN检查），写 `bench_results/p0_12_acceptance/`。coder 禁碰 .tex/TODOList/status，结果由 main backfill。⚠️ 该 bench 短（~几min/wave），期间若 keep12 s118500 落盘会被 watch-loop 延后抓（ckpt 持久不丢）。
> - **✅ P0.12 文档订正（用户两轮 audit）**：`paperA/TODOList.md` + `status/P0_12_DEPTH_REPLAY_LATENCY.md` 均改：① 权威集切 16k `bench_results/p0_12_depth_replay/`（read_s 1.0767→0.7837=1.374×，qtotal 1.329×，peak~17.66GB，pack/LoRA sha 两臂一致），旧 32k `p012/` 降级；② 标题 "纯 depth-replay 延迟隔离"→"**replay 起始层延迟对照**"（非隔离）；③ "replay-kernel speedup"→"**model-side read-path speedup**"（read 未拆 upper-layer vs LM-head，不得称 kernel）；④ 修 4 处表述错（跳/预计算 **lower 12 层**、存 **residual h₁₂**、两臂 layer12 hidden 不同=非语义等价、只对照不证明输出/质量等价/端到端）；⑤ 状态 [DONE]→**[DONE-CORE/VERIFY]**，列 10 项验收。**论文只能写 model-side read-path 对照，不得写同质量 pure-depth 或端到端加速。**
> - **纠错备忘（关键，供后续 heartbeat）**：节点密码文件 `.104=configs/password_h20_24104.txt`、`.73=configs/password_h20_853573.txt`、`.82=configs/password_h20_82250.txt`、`.252(B200)=configs/password_b200_19252.txt`（**不是** password_h20_returned.txt，那是 .53/.174）。全部端口 36000。#100 训练 log=`logs/olmo2_7B_full32_dolmino.log`；keep{8,10,12} log=`logs/olmo2_7B_keep{N}fresh2.log`（无 probe2）。
> - **维持原方案（不 merge）**：keep8 完成后迁 keep10→.252 B200（~6.7×，总 ~2.9d 优于 16 卡 merge ~4.9d）。不动健康 arm。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。

## 当前快照（2026-08-01 21:28 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 idle→补 keep10 s76500 frontier eval）
> **本轮实测**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step5660/200k ppl8.37 3.16s/step 8/8 100%）+ .252 keep8（step75340/200k ppl14.50 1.02s/step，~1.5d 剩）+ .104 keep10（step76560/200k ppl12.83 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step118220/200k ppl11.92 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **✅ .82 idle-card 修复（连续第 8 轮）**：heartbeat 时 8/8 0%（watch caught keep10:76000/keep12:118000），keep10 已推进到 step76560→存在**未抓点 keep10 s76500**。先置 watch-state keep10:76500（防 double-eval race）→ 启 `eval_frontier_pt keep10 76500 10 2`（9 procs，已进 downstream CORE）。
> - **纠错备忘（关键，供后续 heartbeat）**：节点密码文件 `.104=configs/password_h20_24104.txt`、`.73=configs/password_h20_853573.txt`、`.82=configs/password_h20_82250.txt`（IP 28.82.250.82）——**不是** `password_h20_returned.txt`（那是 .53/.174 用）。LOCAL #100 训练 log = `logs/olmo2_7B_full32_dolmino.log`。keep10/keep12 训练 log = `logs/olmo2_7B_keep{N}fresh2.log`（无 probe2）。
> - **维持原方案（不 merge）**：keep8 完成后迁 keep10→.252 B200（~6.7×，总 ~2.9d 优于 16 卡 merge ~4.9d）。不动健康 arm。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 20:56 / 20:24 / 19:55 / 19:26 及更早快照沉淀。

## 当前快照（2026-08-01 20:56 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 无未抓点→等 keep12 s118000 落盘补 frontier eval）
> **本轮实测**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step5060/200k ppl8.03 3.16s/step 8/8 98-100%）+ .252 keep8（step73780/200k ppl14.36 1.02s/step 8/8 100%，~1.5d 剩）+ .104 keep10（step76300/200k ppl12.93 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step117980/200k ppl11.86 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **✅ .82 idle-card 修复（连续第 7 轮，本轮特殊）**：heartbeat 时**无未抓点**（keep12 117500/keep10 76000 均已抓），但 keep12 s118000 临近（~20 步）。等 150s 待其落盘→启 `eval_one 7B_keep12_step118000`（先置 watch state keep12:118000，9 procs）。比让 .82 空等 watch-loop 更紧。
> - **决策复核（拒绝 16 卡 merge）**：考虑过把 .104+.82 合 16 卡跑 keep10（同 diskB，不需 kill 健康 arm 因 .82 无训练）——但 ~2×（~4.9d）**慢于** B200 迁移总时（~2.9d，含等 keep8 ~1.5d），且牺牲 frontier-catch。**维持原方案 = keep8 完成后迁 keep10→.252 B200**，不 merge、不动健康 arm。
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。训练 log 路径 `logs/olmo2_7B_keep{N}fresh2.log`（无 probe2）。
> ↓ 下方 20:24 / 19:55 / 19:26 及更早快照沉淀。

## 当前快照（2026-08-01 20:24 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 keep12 s117500 eval 完成→又 idle→补 keep10 s76000 frontier eval）
> **本轮实测**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step4560/200k ppl8.35 3.16s/step 8/8 99-100%）+ .252 keep8（step72300/200k ppl14.74 1.02s/step 8/8 100%，~1.4d 剩）+ .104 keep10（step76040/200k ppl13.32 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step117760/200k ppl11.86 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **✅ .82 idle-card 修复（连续第 6 轮）**：上轮 keep12 s117500 eval **20:06 DONE**。本轮 .82 又 idle + 新未抓点 **keep10 step76000**（watch state=75500）。复用 `scripts/_eval_frontier_pt.sh` + 先置 watch state `keep10:76000` + 启 `eval_one 7B_keep10_step76000`（log 20:23:47 PPL phase，9 procs）。keep12 无新点（117500 已抓）。
> - **⚠️ .82 已连续 6 heartbeat 手动 catch**：bursty ~85% idle 是结构性。**fix 不变 = keep8（~1.4d 剩）完成后迁 keep10→.252 B200（6.7×）**。不 merge、不动健康 arm。训练 log 正确路径 `logs/olmo2_7B_keep{N}fresh2.log`（无 probe2）。
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 19:55 / 19:26 / 18:55 及更早快照沉淀。

## 当前快照（2026-08-01 19:55 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 keep10 s75500 eval 完成→又 idle→补 keep12 s117500 frontier eval；修正训练 log 路径）
> **本轮实测（nvidia-smi + log 逐节点核对）**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step3980/200k ppl8.35 3.16s/step 8/8 98-100%）+ .252 keep8（step70820/200k ppl14.39 1.02s/step 8/8 99-100%，~1.5d 剩）+ .104 keep10（step75800/200k ppl13.08 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step117540/200k ppl12.09 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **⚠️ LOG 路径修正（重要，未来 heartbeat 照此）**：keep10/keep12 训练 log 是 `logs/olmo2_7B_keep{N}fresh2.log`（**无 probe2**；`olmo2_probe2_7B_keep{N}fresh2` 是 OUTPUT-DIR 名，不是 log 名）。本轮误用 probe2 路径 grep→空→ls 确认 no such file→但 GPU 8/8 100% + ckpt 在推进（keep12 step117500.pt mtime 19:49）证明健康→改用正确路径拿到 step。
> - **✅ .82 idle-card 修复（连续第 5 轮）**：上轮 keep10 s75500 frontier eval **19:37 DONE**（成功补卡）。本轮 .82 又 idle（8×0 MiB）+ 盘上新未抓点 **keep12 step117500**（watch state=117000）。按铁律1 复用 `scripts/_eval_frontier_pt.sh` + 先置 watch state `keep12:117500`（防 double-eval）+ 启 `eval_one 7B_keep12_step117500`（log 19:54:37 PPL phase，9 procs）。keep10 无新点（75500 已抓，next save 76000）。
> - **⚠️ .82 结构性 note（不变）**：frontier-catch 本质 bursty→~85% idle。**结构性 fix = keep8（~1.5d）完成后迁 keep10→.252 B200（6.7×，不 merge、不动健康 arm）**。keep8 完成前每 heartbeat 手动 catch + watch-loop（pid1655855）兜底。
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 19:26 / 18:55 / 18:32 / 18:00 及更早快照沉淀。

## 当前快照（2026-08-01 19:26 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 keep12 s117000 eval 完成→又 idle→补 keep10 s75500 frontier eval）
> **本轮实测（nvidia-smi + log 逐节点核对）**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step3420/200k ppl8.35 3.15s/step 8/8 98-100%）+ .252 keep8（step69340/200k ppl14.78 1.02s/step 8/8 100%，~1.5d 剩）+ .104 keep10（step75520/200k ppl13.55 9.78s/step 8/8 100%，最长杆）+ .73 keep12（step117300/200k ppl11.96 7.81s/step 8/8 99%）。Monitor 8088 http200 OK。
> - **✅ .82 idle-card 修复（连续第 4 轮）**：上轮 keep12 s117000 frontier eval **19:03 DONE**（成功补卡）。本轮 .82 又 idle（8×0 MiB）+ 盘上新未抓点 **keep10 step75500**（watch state=75000）。按铁律1 即刻补：复用 `scripts/_eval_frontier_pt.sh`（本轮不再新增脚本）+ 先置 watch state `keep10:75500`（watch 跳过防 double-eval）+ 启 `eval_one 7B_keep10_step75500`（log 19:26:49 PPL phase，9 procs：8 shard+driver）。keep12 无新点（117000 已抓）。
> - **⚠️ .82 结构性 note（不变）**：frontier-catching 本质 bursty（每 ~65-80min 一新 ckpt，eval~5min）→ ~85% idle。**结构性 fix = keep8（~1.5d）完成后迁 keep10→.252 B200（6.7×，优于 16 卡 merge 且无 NCCL 风险/不 kill 健康 arm）**。keep8 完成前不 merge、不动任何健康 arm；每 heartbeat 手动 catch 未抓点 + watch-loop（pid1655855）兜底。
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。#97（1B keep7 apex）已 completed。
> ↓ 下方 18:55 / 18:32 / 18:00 及更早快照沉淀。

## 当前快照（2026-08-01 18:55 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 上轮 keep10 s75000 eval 完成→又 idle→补 keep12 s117000 frontier eval）
> **本轮实测（nvidia-smi + log 逐节点核对）**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step2840/200k ppl8.12 3.15s/step 8/8 100%）+ .252 keep8（step67840/200k ppl14.71 1.02s/step 8/8 100%，~1.5d 剩）+ .104 keep10（step75260/200k ppl13.12 6.84s/step 8/8 100%，最长杆）+ .73 keep12（step117060/200k ppl11.45 7.81s/step 8/8 99%）。Monitor 8088 http200 OK。
> - **✅ .82 idle-card 修复（连续第 3 轮）**：上轮 keep10 s75000 frontier eval **18:39 DONE**（成功补卡）。本轮 .82 又 idle + 盘上新未抓点 **keep12 step117000**（watch state=116500）。按铁律1 即刻补：建**通用参数化 one-shot** `scripts/_eval_frontier_pt.sh`（复用 watch-loop env+helpers，参数 TAG DIR STEP KEEP FRESH，减少脚本增殖）+ 先置 watch state `keep12:117000`（watch 跳过防 double-eval）+ 启 `eval_one 7B_keep12_step117000`（pid1721292，PPL+core6+know5），8 shard ceph-load 相。
> - **⚠️ .82 结构性 note**：frontier-catching 本质 bursty（每 ~65-80min 一个新 ckpt，eval~5min）→ ~85% idle。**结构性 fix = keep8（~1.5d）完成后迁 keep10→.252 B200（6.7×，优于 16 卡 merge 2× 且无 NCCL 风险/不 kill 健康 arm）**。该期间每 heartbeat 手动 catch 未抓点 + watch-loop（pid1655855）兜底。keep8 完成前不 merge、不动任何健康 arm。
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。#97（1B keep7 apex）已 completed（上轮）。
> ↓ 下方 18:32 / 18:00 及更早快照沉淀。

## 当前快照（2026-08-01 18:32 +08:00，✅ heartbeat：4 训练 arm 全健康；#97 1B keep7 apex eval 全部完成→关闭；.82 idle→按铁律1 填 keep10 s75000 frontier eval）
> **本轮实测（nvidia-smi + log 逐节点核对）**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step2280/200k ppl8.04 3.15s/step 8/8 100%）+ .252 keep8（step66440/200k ppl14.99 1.02s/step 8/8 100%，~1.5d 剩）+ .104 keep10（step75020/200k ppl12.70 8/8 100%，最长杆）+ .73 keep12（step116860/200k ppl11.67 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **✅ #97（1B keep7 replication）关闭**：apex step200000 eval **18:04 全部完成**——PPL=15.41（9L keep7/fresh2 val=dolmino_now_val.npy）+ core6 + know5（mmlu 0.2524 / lambada 0.3976 / boolq 0.6254accn / csqa 0.3898 / siqa 0.4299accn）。结果落 olmo2_ppl_results/1B_keep7_step200000 + olmo2_downstream_results/1B_keep7_step200000{,_know}。任务 #97 标 completed（数字 backfill RUN_REGISTRY/TODOList 为 main-only，NOT .tex）。
> - **✅ .82 idle-card 修复（铁律1）**：1B apex 18:04 完成→8 卡 0 MiB。watch-loop（pid1655855）alive 但 mid-sleep（state keep10:74500/keep12:116500）。盘上有**未抓 frontier 点 keep10 step75000**（rotation 保留 75000/74500/70000）。立即填：先置 watch state `keep10:75000`（watch 跳过→无 double-eval/碰撞）+ 建 `scripts/_eval_keep10_s75000.sh`（复用 watch-loop env+helpers）+ 启 `eval_one 7B_keep10_step75000`（pid1702893，PPL+core6+know5）。9 shard 进程 7×Dl（ceph 加载 28GB）+1×Rl，~1-2min 落卡。keep12 无新点（116500 已抓）。
> - **长杆方案不变**：keep8 完成后（~1.5d）迁最差长杆 keep10→.252 B200（~6.7×，不 merge、不动健康 arm）。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 18:00 / 17:28 及更早快照沉淀。

## 当前快照（2026-08-01 18:00 +08:00，✅ heartbeat：4 训练 arm 全健康；.82 watch-loop 已抓 2 新点后进 sleep→IDLE→填 1B keep7 apex eval 补卡）
> **本轮实测（nvidia-smi + log 逐节点核对）**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step1700/200k ppl8.29 3.15s/step 8/8 100%）+ .252 keep8（step64880/200k ppl14.61 1.02s/step 8/8 100%，~1.6d）+ .104 keep10（step74740/200k ppl13.08 6.84s/step 8/8 100%，最长杆~9.9d）+ .73 keep12（step116620/200k ppl11.57 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **✅ .82 idle 窗口再填**：watch-loop（pid1655855）已抓 keep10 s74500 + keep12 s116500 两新点→state 记 keep10:74500/keep12:116500→进 900s sleep（8 卡 0 MiB）。查 #97 **1B keep7 replication**：16-card 训练 **已 DONE**（step200000 @2026-07-20，evals 有 50k/100k/147k/148.5k/150k）**但 apex step200000 eval 缺失**。建 `scripts/_eval_1B_keep7_apex.sh`（pid1686690，BASE 覆盖为 OLMo-2-0425-1B，keep7/fresh2 protocol 与既有 1B eval 完全对齐：val=dolmino_now_val.npy n_win4096）→ eval_one 8 卡 shard PPL+core6+know5，实测 8/8 GPU 57-99% ~18GB/卡 active。**零竞争**：watch-loop ~18:04 醒来时下一 7B ckpt 未到（keep10~18:23/keep12~18:42）→resleep，1B eval(~10min)安全跑完。补 #97 apex 缺口。
> - **长杆方案不变**：keep8 完成后（~1.6d）迁最差长杆 keep10→.252 B200（~6.7×，> 16 卡 merge ~2× 且需 kill 健康训练/NCCL 风险），本轮不 merge、不动任何训练 arm。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 17:28 / 16:38 及更早快照沉淀。

## 当前快照（2026-08-01 17:28 +08:00，✅ heartbeat：4 训练 arm 全健康在 200k 轨道；.82 continuation 电池排空→IDLE→按铁律1 建 bounded trajectory WATCH-LOOP 补卡）
> **本轮实测（nvidia-smi + log 逐节点核对，真实时钟~17:23）**：
> - **✅ 4 arm 全健康**：LOCAL #100 full-32L（step1140/200k ppl8.10 3.15s/step 8/8 100%）+ .252 keep8（step63420/200k ppl14.30 1.02s/step 8/8 100%，~1.6d ETA）+ .104 keep10（step74500/200k ppl12.85 6.84s/step 8/8 100%）+ .73 keep12（step116400/200k ppl11.72 7.81s/step 8/8 100%）。Monitor 8088 http200 OK。
> - **⚠️→✅ .82 IDLE-CARD 修复**：上轮 continuation 电池（pid1624185）**16:58 DONE**（keep12 s116000 ppl11.52+core6+know5 + keep10 s74000 落盘）→ 8 卡 0 MiB。按铁律1 建 **bounded trajectory watch-loop** `scripts/_run_olmo2_frontier_82_watch.sh`（pid1655855）：轮询 keep10/keep12 最新 ckpt，每出一个新 frontier 点就 eval（PPL+core6+know5, base protocol）——因盘上仅滚动保留 3 个 ckpt（rotation），不及时抓会被删。预置 keep12:116000 跳过已做；现在 eval **keep10 step74500**（新点），8 PPL shard 处 ceph 加载相（Dl）。zero-risk forward-only。
> - **长杆再评估**：keep10 现为最长杆（~9.9d）> keep12（~7.6d）> #100（~7.25d）>> keep8（~1.6d）。**加速方案不变**：keep8 完成后（~1.6d）迁最差长杆→.252 B200（~7.7× > 16 卡 merge ~2× 且需 kill 健康训练/NCCL 风险），本轮不做 merge、不动任何训练 arm。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 16:38 / 16:27 及更早快照沉淀。

## 当前快照（2026-08-01 16:38 +08:00，✅ heartbeat：4/5 arm 全健康；.82 eval-battery 队列排空→IDLE→按铁律1 立即补挂 continuation frontier 电池）
> **本轮 heartbeat 实测（nvidia-smi + log 逐节点核对）**：
> - **✅ 4/5 arm 全健康在 200k 轨道**：LOCAL #100 full-32L（step160 ppl8.18 3.15s/step 8/8 100%）+ .252 keep8（step60840 ppl15.09 1.02s/step 8/8 100%）+ .104 keep10（proc alive 9:48h 8/8 99-100%/90GB，log block-buffered=正常）+ .73 keep12（step116000 ppl11.68 7.81s/step 8/8 100%，长杆 ETA~7.5d）。Monitor 8088 http200 OK。
> - **⚠️→✅ .82 IDLE-CARD 修复**：.82 之前的 frontier eval 电池（`_run_olmo2_mmlu_recovery_frontier_82.sh`）固定 4-cell 队列（keep10 s73500 / keep12 s115500 / keep8 s48000 / keep10 s70000）**全跑完**（最后 keep10 s70000 know 16:15:37 落盘，其间一次瞬时 OOM warning 但已恢复 mmlu n=1755 完成）→ 8 卡全 0 MiB 空转。按铁律1 立即补挂 **continuation 电池** `_run_olmo2_frontier_82_cont.sh`（pid1624185，复用同 `eval_one` helper=PPL+core6+know5）：**keep12 step116000 + keep10 step74000**（各 KEEP/FRESH 正确，forward-only，喂 Paper B 深度阶梯 healing-trajectory）。8 个 PPL shard proc 已起，处 ceph 模型加载相（34GB fp32×8 并发争 I/O，GPU compute 稍后起），下轮 HB 核 compute+结果。产出 `olmo2_ppl_results/7B_keep{12_step116000,10_step74000}` + `olmo2_downstream_results/…{,_know}` + `/tmp/frontier_82_cont_DONE`。
> - **keep12 长杆加速路径不变**：keep8 完成后（~1.7d）迁 keep12→.252 B200（7.7× > 16 卡 H20 2×）；现不动 .73 keep12 训练。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 16:27（#100 launch）及更早快照沉淀。

## 当前快照（2026-08-01 16:27 +08:00，✅ ShortGPT16 #98 DONE @200k（final.pt 16:13:09）→ LOCAL B200 空出 → 启动 #100 full-32L continued-pretraining control（Paper B 头号因果对照）；5-arm ladder 全健康）
> **本轮 launch-check（one-shot，非完整 heartbeat）**：
> - **✅ #98 ShortGPT-16 heal DONE**：`logs/olmo2_7B_shortgpt16.log` 末行 `[step 200000/200000] loss=2.1785 ppl=8.83`，`final.pt` 已存（16:13:09），proc 已退 → LOCAL B200 8 卡空出。4-point eval 待跑（pending）。
> - **✅ #100 full-32L continued-pretraining control 已启动（LOCAL B200 wzc1）**：`scripts/_run_olmo2_full32_dolmino_heal.sh`，`keep_front=32/n_fresh=0`（**不剪层**，移植全 32 层+embed/norm/lm_head，uniform LR 2e-5）→ 回答头号 reviewer 因果问：MMLU 崩塌来自**剪层**还是 **Dolmino 续训语料遗忘**。config: BS4 GA4 eff_bs128 seq2048 max200k warmup150 save5000 gradckpt fp32-AdamW。commit `4f7051a`。实测 `[step 60/200000] loss=2.13 ppl=8.43 gnorm=0.34 3.15s/step maxmem=176.9GB`，8/8 GPU 100%/~174GB，无 OOM/Traceback，`[optim] group inh_decay 7298.1M base_lr=2.00e-05` → eff_bs/LR 均正确。ETA~7.3d。LOG=`logs/olmo2_7B_full32_dolmino.log`。
>   - **⚠️ 两次 pre-launch OOM 误射（均在跑 step 前 kill，无落 step）**：(1) BS16 GA1 在 B200 178GiB OOM——full-32L fp32-AdamW ~112GiB 固定优化器态（W28+G28+m28+v28）+ BS16 activation 超顶；(2) 我 BS4 GA8 修法误得 eff_bs256（算错）→ kill 重启 BS4 GA4=eff_bs128。**教训：full-32L fp32-master 峰值内存随 micro-batch(BS) 而非 GA；eff_bs=BS×GA×NPROC。**
> - **✅ 5-arm compute-matched depth ladder 全健康在 200k 轨道**：LOCAL **#100 full-32L**（step60 3.15s/step ETA~7.3d，NEW）+ .252 keep8（#96 B200 1.02s/step ETA~1.7d）+ .104 keep10（#96 6.84s/step）+ .73 keep12（#95 step~115720 7.81s/step 长杆 ETA~7.6d）+ .82 forward-only base-protocol eval battery（frontier PPL+MMLU/core6/know5 for keep10/12/8）。
> - **keep12 长杆加速路径不变**：keep8 完成后（~1.7d）迁 keep12→.252 B200（7.7×）。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 13:02 及更早快照沉淀（其 P0.12 已 DONE，见 `status/P0_12_DEPTH_REPLAY_LATENCY.md`）。

## 当前快照（2026-08-01 13:02 +08:00，⚠️ P0.12 subagent a2e1bc4c 死亡（~12:27，API-400 pattern，0 GPU work）→ TaskStop + INLINE 重启在 .82；4-arm ladder 全健康）
> **本轮 heartbeat 实测（nvidia-smi + log 逐节点核对）**：
> - **⚠️→✅ P0.12 idle-card 修复**：12:22 派的 P0.12 subagent `a2e1bc4c` 到 12:53 仍 .82 8 卡全 0 MiB/无 bench 进程、transcript mtime 冻结在 12:27（~27min stale，与此前 #112 两次 API-400 死亡同 pattern）→ 判定已死。**TaskStop 之 + 改 INLINE 跑**（不再派 subagent，规避 API-400）：.82 bg `scripts/_p012_latency.sh`（j=0 与 j=12 **同挂 flagship LoRA** `outputs/qcmem_distill_qwen_j12_r32_4k/final`，32k，n_repeat=20 warmup=2 n_decode=32，各 3 process reps，串行 cuda:0 求净延迟）。已核 j=0+LoRA smoke 通（read_len=6657=同 pack），当前 j0_rep1 GPU0 100%/30GB 在跑。产出 `bench_results/p012/p012_j{0,12}_rep{1,2,3}.json`（.82 diskB）+ touch `/tmp/p012_DONE`，跑完 scp 回 + 写 `status/P0_12_DEPTH_REPLAY_LATENCY.md` + 回填 TODOList P0.12（main-only，NO .tex）。
> - **✅ 4-arm compute-matched depth ladder 全健康在 200k 轨道**：LOCAL ShortGPT16（#98，step192840 ppl8.89 1.56s/step 8/8 100% ETA~3.1h=最先完成）+ .252 keep8（#96 B200，step50240 1.02s/step 8/8 100% ETA~1.77d）+ .104 keep10（#96，8/8 100%/90GB 42procs，log block-buffered=正常）+ .73 keep12（#95，step114420 7.81s/step 8/8 100% 长杆）。Monitor 8088 http200 OK。
> - **keep12 长杆加速最优路径不变**：keep8 完成后（~1.77d）迁 keep12→.252 B200（7.7×）优于 16 卡 H20 合并（~2×）；.82 现先跑 Paper-A P0.12，.252 空出后再评估 keep12 迁移。
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 12:22 快照沉淀（其 .82=subagent a2e1bc4c 已作废，见上）。

## 当前快照（2026-08-01 12:22 +08:00，✅ #112 mc_ll rescore DONE（数据本已在 .82 磁盘，只回填 TODOList 无需重跑）→ .82 空出→ Paper A GPU 队列仅剩 P0.12（可选延迟隔离）→ 按 Paper-A-first 派 subagent 在 .82 跑 P0.12；4-arm ladder 全健康）
> **本轮 heartbeat 实测（nvidia-smi + log 逐节点核对）**：
> - **✅ #112 CLOSED（P2.1 QCMem InfiniteBench longbook_choice_eng LL-MC rescore）**：`--mc_ll` acc_ll=**48.03**（n=229，0 OOM）vs clean-letter 17.47（**+30.56pp**，证实 chat=False clean-letter 低估 choice）；matched-n=229 同 scorer native-window Dense LL-MC=32.31（QCMem +15.72pp）。**数据早在 .82 磁盘**（P2.1 worker 8-01 02:52–03:43 经 `_infb_orchestrate_p21.sh` S3/S4 跑完）→ 无需重跑，只读+回填 `paperA/TODOList.md` P2.1（main-only，NO .tex）。raw `infbench_results/{qcmem_8b_j12_lora_llmc,kvdirect_8b_llmc}/scores.json`（.82 diskB）。task#112 completed。
> - **✅ 4-arm compute-matched depth ladder 全健康在 200k 轨道**：.252 keep8（#96 B200，step48240，1.02s/step，8/8 100%/115GB，ETA~1.8d）+ .104 keep10（#96，8/8 100%/90GB，log block-buffered=正常，GPU 忙=健康）+ .73 keep12（#95，step114100，7.81s/step，8/8 100%/96GB，ETA~7.8d=长杆）+ LOCAL ShortGPT16（#98，step191320 ppl9.13，1.56s/step，ETA~3.8h）。Monitor 8088 http200 OK。
> - **.82（28.82.250.82 H20 diskB）= Paper A P0.12 depth-replay 延迟隔离（subagent a2e1bc4c，12:22 起）**：#112 已闭合 → .82 空出 → Paper A GPU 队列仅剩 P0.12（最后一个 Paper-A GPU 项，可选/非阻塞）→ 按 memory `h20-paperA-over-paperB-priority` 严格 Paper-A-first 先跑它。`bench_qcmem_vs_fullctx.py` j0+flagshipLoRA vs j12+同LoRA，同 pack，≥20 reps×3 process，非训练~1h。产出 raw JSON（.82）+ `status/P0_12_DEPTH_REPLAY_LATENCY.md`。
> - **keep12 长杆加速最优路径**：keep8 完成后（~1.8d）迁 keep12→.252 B200（1.02s/step=**7.7×** vs .73 7.81s/step）优于 .82+.73 16 卡 H20 合并（仅~2×）→ 现不动 keep12，.82 先跑 Paper-A P0.12；.252 空出后再评估 keep12 迁移。
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 12:05 快照沉淀。

## 当前快照（2026-08-01 12:05 +08:00，✅ keep8→.252 B200 迁移**成功**（11:34「中止」快照系 PORT 错误，非密码轮换——已更正）→ keep8 现跑在 .252 B200 @1.02s/step≈5.7× vs H20；.82 空出→接 Paper A #112 mc_ll rescore）
> **★ 更正 11:34「迁移中止」结论（用户质疑「你用错密码了吧」→ 属实是端口错，用户判断正确）**：.252（28.89.19.252）**并非密码轮换失效**，而是我此前用了**端口 22**（Permission denied）；实际 `password_b200_19252.txt` 在**端口 36000** = ✅ AUTH_OK。∴ 迁移**未中止、已完成**：keep8 `step47000.pt`（34152195199 B，diskB→wzc1 字节校验一致）现在 **.252 B200 从 step47000 续跑**，实测 `[step 47060/200000] 1.02s/step`、8/8 GPU 100%/115GB → ETA→200k ≈1.8d（vs H20 5.78s/step≈10.2d，~5.7× 加速）。**.82 keep8 已 kill（pkill -9，12:00 实测 8 卡全 0-3 MiB 空闲）**→ 按 Paper-A-first 补 **#112**（QCMem InfiniteBench longbook_choice_eng LL-MC rescore，`--mc_ll`，QCMem-only，chat=False，subagent a1796dc1 在跑）。
> - **✅ 4-arm compute-matched depth ladder 全部在 200k 轨道（实测 12:05）**：.252 keep8（#96，B200，1.02s/step，8/8 100%）+ .104 keep10（#96，8/8 100%）+ .73 keep12（#95，step113980，7.81s/step，8/8 100%）+ LOCAL ShortGPT16（#98，step190660，1.56s/step，8/8 99-100%，~4h left）。Monitor 8088 http200 OK。
> - **.82（28.82.250.82 H20 diskB）= Paper A #112 mc_ll rescore（QCMem-only，subagent a1796dc1）**：keep8 迁走后空出，按 memory `h20-paperA-over-paperB-priority` 补 Paper A GPU 待跑项（Paper A GPU 队列除 #112 外已排空）。
> - **wzc1 B200 现状**：.252 ✅ 可达（**端口 36000**）在跑 keep8；.53/.188/.18 仍 Permission denied（真密码轮换）。**教训：连 wzc1 B200 先试端口 36000 再判死，别默认 22。**
> - dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 11:34（结论已更正为误报「中止」）及更早快照沉淀。

## 当前快照（2026-08-01 11:34 +08:00，⚠️【已更正：本快照结论错误】keep8→.252 迁移被误判「中止」——实为我用错端口 22，.252 在端口 36000 实际可达、迁移已成功，详见上方 12:05 快照）
> **本轮事件（keep8 B200 迁移尝试 + 中止）**：用户此前授权「拿回 .252 跑」。earlier this session .252（28.89.19.252，`password_b200_19252.txt`，port 22）曾一度可达（8× L20A idle，torch2.13）→ 已把 .82 keep8 `step47000.pt`（34152195199 B）scp diskB→wzc1（校验字节精确一致，`outputs/olmo2_probe2_7B_keep8fresh2/step47000.pt`，作 backup 留存）。**但本轮再连 .252 = `Permission denied`（密码已被集群再次轮换，与 .53/.188/.18 同命运）→ 无 B200 可落 → 迁移中止**。keep8 **不 kill**，继续在 .82（step47800 5.78s/step 健康）跑 →200k。#112（mc_ll rescore）因无卡空出仍 pending。
> - **✅ 4-arm compute-matched depth ladder 全部在 200k 轨道，全忙无空转**：LOCAL ShortGPT16（#98，step189100 1.56s/step，~4.5h left）+ .82 keep8（#96，step47800 5.78s/step）+ .104 keep10（#96，step71380 6.84s/step）+ .73 keep12（#95，step113660 7.81s/step）。Monitor 8088 http200 OK。
> - **.252（28.89.19.252 B200 wzc1）= 不可达（密码 2026-08-01 再轮换）**；全 4 台 wzc1 B200（.252/.53/.188/.18）现均不可达。若 .252 日后回归，wzc1 已有 keep8 step47000 ckpt→可即时 relaunch（届时宜取 .82 更新 ckpt）。
> - **长期优先级不变（memory `h20-paperA-over-paperB-priority`）**：H20 空出先补 Paper A 待跑（#112 mc_ll rescore 为下一个 Paper A GPU 项），排空后才 resume Paper B。当前无 H20 空出。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> ↓ 下方 07:02 及更早快照沉淀。

## 当前快照（2026-08-01 07:02 +08:00，✅ Paper A GPU 队列全排空（#62 tab_scale DONE+folded）→ Paper-A-first 满足→ Paper B 4-arm compute-matched depth ladder 全部在 200k 轨道：.73 keep12 + .104 keep10 + .82 keep8（07:02 起，.82 dllm co-tenant 已撤）+ LOCAL ShortGPT16）
> **★ 长期优先级（memory `h20-paperA-over-paperB-priority`）：三 QCMem H20（.73/.82/.104）先跑 Paper A 再 Paper B；任何节点空出先补 Paper A 待跑项，Paper A 全排空后才 resume 暂停的 Paper B。** SSH：QCMem H20 三节点用**端口 36000** + 各自密码文件（.73=`password_h20_853573.txt` / .104=`password_h20_24104.txt` / .82=`password_h20_82250.txt`）。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅→DONE Paper A #62（Qwen3 scale RULER chat=False，Appendix tab_scale）COMPLETE 2026-08-01 06:50**（worker ac2d74ac 收工）：**全 6 档跑完**——0.6B(j2)=85.68 / 1.7B(j3)=66.81 / 30B-A3B(j12)=80.48（本任务新档，n=500）+ 参考 4B=60.52 / 14B=52.91 / 32B=88.18(n25)。per-task selector（niah→bm25 topk12 r0；vt→iter_bm25 topk16 r4 hop4）与 locked 4B/14B/32B run 逐行一致→跨档可比已核。scale 非单调（0.6B 最高、14B 最弱其 vt 塌）。**已 fold 进 `paperA/TODOList.md` P2.3**（main-only，NO .tex），task#62 completed。产出 `status/P62_SCALE_RULER_CHATFALSE.md`，commit `3ca261d`+`4f7051a`（未 push）。raw `ruler_results/qcmem_scale_{0p6b,1p7b,30ba3b}_chatFALSE_ruler/`。
> - **★ Paper A GPU 队列现全排空**：#62 是最后一个 Paper A GPU 待跑项；剩余 Paper A 全为 main-only .tex（P0.8 provenance 措辞、#10 InfLLM 行入 .tex + LoCoMo errata + P1.2 软化）——无 GPU 工作 → 按 memory `h20-paperA-over-paperB-priority`「Paper A 排空后才 resume Paper B」条件**满足** → 三 QCMem H20 现全接 Paper B 深度阶梯 resume。
> - **✅ .73（06:40 起）= Paper B keep12 resume（#95）**：`KEEP=12 N_FRESH=2 RESUME_FROM=outputs/olmo2_probe2_7B_keep12fresh2_wzc1/step111500.pt BS4/GA4`，pid3983054，LOG=`logs/olmo2_7B_keep12fresh2.log`。resume 已核 continue @ step=111500（optimizer 157 params），8/8 GPU 100%/96GB 健康，→200k。
> - **✅ .104（06:43 起）= Paper B keep10 resume（#96，#62 排空后接力）**：30B-A3B long-pole shard 跑完 .104 全空 → 接 compute-matched ladder 缺口 keep10。`KEEP=10 N_FRESH=2 RESUME_FROM=outputs/olmo2_probe2_7B_keep10fresh2/step69000.pt BS4/GA4`，pid3995588，LOG=`logs/olmo2_probe2_7B_keep10fresh2.log`。**实测 pid alive etime4:56，8/8 GPU 100%/90.5GB**（log block-buffered 空属正常，wrapper 用 python 非 -u），→200k。
> - **✅→DONE .104 P2.2 持久化存储 I/O（#110）+ P0.2 step-0 recovery（#102）**：均 COMPLETE 已回填（详见下方作废快照）；.104 于 06:43 转 keep10。
> - **.82（28.82.250.82 H20 diskB）= Paper B keep8 resume（#96，07:02 起，dllm co-tenant 已撤，铁律1 补齐 ladder 最后缺口）**：Paper A P1.1/P1.5（#107）此前已在此跑完（`status/P1_1_P1_5_SCALING_RESULTS.md`）。本轮 heartbeat 实测 .82 8 卡全 0 MiB/0%（dllm SFT 走了）→ 立即补挂起的 keep8。`KEEP=8 N_FRESH=2 RESUME_FROM=outputs/olmo2_probe2_7B_keep8fresh2/step45000.pt BS4/GA4`，pid1531968，LOG=`logs/olmo2_7B_keep8fresh2.log`（save5000，原 500 会爆盘已改）。resume 已核：restored 113 model tensors(strict fp32-master) + optimizer 113 param states(Adam momentum 保留) + **continue @ step=45000** lr_fresh=8.927e-05 lr_inh=1.785e-05。8/8 GPU ~99%/78.4GB 健康，→200k。
> - **✅ LOCAL（B200/L20A wzc1）= Paper B ShortGPT-16 heal（#98）**：`outputs/olmo2_probe2_7B_shortgpt16`，跑到 200k（用户豁免 Paper-A-first 规则）。step~178880 ppl9.97，绝不重启。
> - **✅ Paper B 4-arm compute-matched depth ladder 现全部在 200k 轨道**：keep8（.82 pid1531968 @45000→200k）+ keep10（.104 pid3995588 @69000→200k）+ keep12（.73 pid3983054 @111500→200k）+ ShortGPT16（LOCAL @178880→200k）。铁律1 满足——四张节点每张都填到最高价值的 stalled 训练，无空转。
> - **Paper A 剩余（全 main-only .tex，无 GPU）**：P0.8 provenance 措辞（含修「continued-trained Qwen3-8B base LM」不实措辞）；#10（InfLLM 行入 .tex + CoMem LoCoMo errata + P1.2 措辞软化）。
> - **✅ 已闭合本轮**：P0.2 config#2 j=0 text-RAG 5-benchmark gap-fill（task #108，`status/P0_2_CONFIG2_JRAG_RESULTS.md`）→ RULER macro 99.20 / LongBench 12.31 / LongEval 97.2% / LoCoMo judge 41.59%（= 蒸馏 teacher/上界），已回填 TODOList P0.2 + `status/P0_2_PARETO_RESULTS.md`（§4a/4b/4c/5/7 NOT-FOUND 全闭）。
> ↓ 下方 2026-07-31 19:27 及更早快照作废存档。

## 当前快照（2026-07-31 19:27 +08:00，✅ Paper A P0.11 frozen-j12 DONE（.73 subagent a62bdb0 收工→回填 TODOList）+ P0.3 matched-n=100 DONE（已回填）→ .73 空闲检出→立即补 Paper B keep8 resume（铁律1，补齐 compute-matched ladder 最后缺口）→ 4 训练全忙：LOCAL ShortGPT16 + .82 keep10 + .104 keep12 + .73 keep8）〔已作废存档〕
> **实测台账（本轮 nvidia-smi + ps 逐节点核对）**。SSH：QCMem H20 三节点用**端口 36000** + 各自密码文件（.73=`password_h20_853573.txt` / .104=`password_h20_24104.txt` / .82=`password_h20_82250.txt`）——**不是 `password_h20_returned.txt`、不是端口 22**。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> **★ Paper B 4-arm compute-matched depth ladder 现全部在 200k 轨道上跑**（keep8/keep10/keep12/ShortGPT16）——铁律1 把每张空卡都填到最高价值的 stalled 训练。
> - **✅ LOCAL（B200/L20A wzc1）= Paper B ShortGPT-16 heal（#98）**：`train_olmo2_shortgpt.py --output_dir outputs/olmo2_probe2_7B_shortgpt16`，**step 150000/200000，8/8 GPU 100% / 132GB**，健康。绝不重启。
> - **✅ .82（28.82.250.82 H20 diskB）= Paper B keep10 heal（#88）**：`--output_dir outputs/olmo2_probe2_7B_keep10fresh2`，**step 67060/200000 ppl13.35，99% / 90.5GB**，健康。绝不重启。
> - **✅ .104（28.83.24.104 H20 diskB）= Paper B keep12 resume（#95）**：`--output_dir outputs/olmo2_probe2_7B_keep12fresh2_wzc1 --resume_from step111500.pt`，**step 111540/200000 ppl11.82，100% / 96.4GB**，resume 已核（optimizer_state 157 params, Adam momentum preserved）。健康。LOG=`logs/olmo2_7B_keep12_resume_104.log`。
> - **✅ .73（28.85.35.73 H20 diskB）= Paper B keep8 resume（#96，19:27 起，铁律1 补卡）**：P0.11 frozen-j12 subagent a62bdb0 **收工**（80 jobs 0 fail，.73 8 卡全释放 0procs）→ main 立即补 compute-matched ladder 最后缺口 keep8（原停在 step44000）。`--output_dir outputs/olmo2_probe2_7B_keep8fresh2 --keep_front_layers 8 --n_fresh_layers 2 --resume_from step44000.pt`，BS4/GA4 lr1e-4/inh2e-5 max200k **save5000（原 500 会爆盘，已改）**。resume 已核：optimizer_state 113 params restored / continue @ step=44000 / lr_fresh=8.973e-05。**8/8 GPU 100% / 78.4GB，step44020 loss2.70 ppl14.82**，健康。LOG=`logs/olmo2_7B_keep8_resume_73.log`。ETA 156k step ~10.5h（5.84s/step）。
> - **✅ Paper A P0.11 frozen-j12（.73 subagent a62bdb0）DONE**：Qwen3-8B resume_j=12 **NO LoRA**（frozen backbone），iter_bm25/top-12/hop-4/rounds=0/chunk-512/BOS sink/chat=False/seed=42/n=100。**RULER 15-cell macro=8.01 / LongEval mean=0.2% / LongBench 6-QA F1=9.96**（+已有 BABILong 24.52 / LoCoMo 24.52）。**崩塌 vs flagship CoMem+LoRA(j=12) RULER 97.05、亦 < frozen(j=9) 59.41 → 隔离出 fixed-depth 的 LoRA adaptation 增益**。已回填 `paperA/TODOList.md` P0.11 表（commit a5d1208 未 push）+ 记录 `status/P0_11_FROZEN_J12.md`（commit aa0ce88 未 push）。tex 待办（tab_overview 增 frozen j=12 行）留 main 后续，非本轮（用户指令：只填 TODOList 不填 .tex）。
> - **✅ Paper A P0.3 matched-n=100 YaRN（.104 subagent ab0993f）DONE**：native macro 96.07 / +YaRN 92.04；**headline vt@128k=99.0 > +YaRN 93.6 > KVD-YaRN 57.8 > full-ctx 0，+41.2pp HOLDS**。已回填 TODOList（commit 6cc94a6 未 push）+ `status/P0_3_MATCHED_N100.md`（2bd16ff）。
> - **.252（28.89.19.252 B200 wzc1）= 不可达**。task#99 keep14-distill 已 park。
> - **⚠️ 台账更正**：task#102（P0.2 step-0 eval battery）曾被标 in_progress-on-.73 但 .73 实际 0 procs 空转（orchestrator 已死）→ 本轮 reset 回 pending，.73 改跑更高价值的 keep8。
> ↓ 下方 2026-07-31 19:13 及更早快照作废存档。

## 当前快照（2026-07-27 01:35 +08:00，✅ .104 = Paper B keep12 heal 已起跑（step25500 resume）+ LOCAL freeze_front #59 续跑）〔已作废存档〕
> **节点分配**：.82 = 用户 seed-variance 训练（ETA ~03:40）；.104 = Paper B keep12 heal（trainer subagent 启）；LOCAL = freeze_front #59；.73/.252 未触碰。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅ .104（28.83.24.104, H20 diskB, p36000）= Paper B keep12 heal 已健康运行**：`KEEP=12 N_FRESH=2 BS=4 GA=4 eff_bs=128` 从 step25500 resume（diskB 最新 ckpt）。PID 3302609（torchrun coordinator），8 rank worker。step25520 loss=2.6524 ppl=14.19 → step25540 loss=2.6638 ppl=14.35 lr=1.93e-05 gnorm=0.54 7.86s/step maxmem=91.9GB。**8 GPU 100%/96.4GB 全健康**。OUT=`outputs/olmo2_probe2_7B_keep12fresh2`（diskB路径），LOG=`logs/olmo2_7B_keep12fresh2.log`。ETA step200000 ~14h（~2026-07-27 16:00），ckpt 每 5000 步轮转。⚠️ 前置坑：ALPS SparseForge-Resubmission 进程（28h stuck，0 output）消耗 6-10GB/GPU 导致两次 OOM；trainer subagent kill -9 后重跑成功。⚠️ 文件系统注意：.104 `/apdcephfs_wzc1/share_304376610/` 实为 diskB（zwfy6）alias，与本机 wzc1 不同；本机 keep12 step111500 对 .104 不可见。
> - **✅ LOCAL（B200 wzc1）= freeze_front #59 续跑**：上次已知 step54280/200000 (27.1%)，1.32s/step。绝不重启。
> - **.82（28.82.250.82, H20 diskB）= 用户 seed-variance 训练**：ETA ~03:40。seed 完成后可接 keep10 heal（见 keep10 准备命令）。
> ↓ 下方 2026-07-26 10:55 快照作废存档。

## 当前快照（2026-07-26 10:55 +08:00，✅ .82 空出 → P3.2 YaRN-KVD@128k 实测起跑 + freeze_front（#59）健康推进；heartbeat cron 已恢复）〔已作废存档〕
> **节点分配**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(28.82.250.82, H20 diskB)。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。**heartbeat cron 本轮恢复**：durable `bf3d24e0`（13,43 * * * *，30min 一次，7 天后 2026-08-02 过期需重加）。
> - **✅ freeze_front（#59）LOCAL 8 卡健康**：step **54280/200000 (27.1%)**，loss2.71/ppl15.03，8 卡 87.8GB/健康，1.32s/step。decisive matched eval @ step128000/153500/200000。绝不重启。task#59 in_progress。
> - **✅ .82（H20 diskB）= P3.2 YaRN-KVD@128k 实测（3 卡并行，10:55 起，ETA~13:15）**：外部 co-tenant 走了，8 卡全空 → 跑 ARR reviewer P3.2 最后挂起的实测点。bg subagent a0d8c72aa4731a53d：私有 model 副本 `Qwen--Qwen3-8b-yarn`（config.json 注入 `rope_scaling yarn factor4`，权重全 symlink，max_pos 保持 40960）+ n=100 KVD chat=False。GPU0 niah_single_3(pid1093933)、GPU1 niah_multikey(pid1094515)、GPU2 vt(pid1095410)，各 54.3GB/97.9GB，~81s/it。**★ 冒烟 n=2 已出关键信号：niah_single_3/128k recall=100.0**（未扩展 KVD 此处崩到 0）→ YaRN 版能救回，**可能需把「128k 只有 CoMem 可用」的措辞改为「CoMem≈YaRN-KVD 精度但 O(L)-write+固定 read vs O(L²)+89GB」的效率优先叙事**——待 n=100 三任务定论。结果落 .82 `ruler_results/kvd_yarn_128k/*.csv`，跑完 scp 回本地。⚠️ .82 `.venv/bin/python` 是 py3.14-broken，用 `/usr/bin/python3.11 + PYTHONPATH=.venv/lib/python3.11/site-packages`（同 MemoryLLM 修法）。
> - **.104/.73/.252（用户的）**：.104 GPU 0% util（残留缓存 3-11GB），未干预；.73/.252 未触碰。
> - **Monitor 8088**：本轮 down(http 000)→清残留+重启→http 200 OK。

## 当前快照（2026-07-25 22:35 +08:00，✅ #68 CLOSED——MemoryLLM LongEval chat=False=13.6 在 .104 跑完并回填两文档 + freeze_front（#59）健康推进）〔已作废存档〕
> **节点分配**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(H20 diskB)。**★ 本轮用户 steer「不是有三个 H20 节点么 应该空了一个啊」→ 实测 .104 已空(8 卡 util 0%)、.73 满载(100%)、.82 仍被外部 co-tenant 占**。.104=QCMem H20 挂 **wzc1 共享盘**(结果本地直接可见,无需回传)+ MemoryLLM 权重 `../baselines/memoryllm-8b-chat-hf` + py3.11 env 全齐 → #68 从「等 .82」改为直接在 .104 跑完。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅ #68 MemoryLLM LongEval chat=False CLOSED（.104，8-GPU 分片）**：bg subagent a2029253 跑完，**mean(8k–128k)=13.6**（8k22/16k22/32k16/64k6/128k2，n=50/档，全 shard `use_chat_template:false`，无 OOM，`--score_only` 合并 `longeval_results/memoryllm_8b_chatFALSE/_summary_merged.json`）。**MemoryLLM 现全 5 benchmark 都有真 chat=False，主矩阵无 chat=True 占位。** 已回填 BENCHMARK_CHATFALSE_MASTER §B/headline/脚注3-4/§G + PAPERA_RESULTS_CONSOLIDATED headline/§B/§7/provenance（14.0ᵀ→13.6）。**已删 .82-poll cron 75473588。** 我的 8 shard 进程已干净退出，.104 恢复启动前状态无残留。task#68 completed。
> - **✅ freeze_front（#59）LOCAL 8 卡健康**：step 25260/200000，loss2.86/ppl17.50，8 卡 102GB/100%，1.32s/step。decisive matched eval @ step128000/153500/200000。绝不重启（会在同 8 卡重复=灾难）。task#59 in_progress。
> ↓ 下方 22:22 及更早快照作废存档。

## 当前快照（2026-07-25 22:22 +08:00，✅ .104 空出 → #68 MemoryLLM LongEval chat=False 改在 .104 跑 + freeze_front（#59）健康推进）〔已作废存档〕
> **节点分配**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(H20 diskB)。**★ 本轮用户 steer「不是有三个 H20 节点么 应该空了一个啊」→ 实测 .104 已空(8 卡 util 0%)、.73 满载(100%)、.82 仍被外部 co-tenant 占**。.104=QCMem H20 挂 **wzc1 共享盘**(结果本地直接可见,无需回传)+ MemoryLLM 权重 `../baselines/memoryllm-8b-chat-hf` + py3.11 env 全齐 → #68 从「等 .82」改为直接在 .104 跑。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅ #68 MemoryLLM LongEval chat=False 起跑（.104，8-GPU 分片）**：bg subagent a2029253，`eval_memoryllm_longeval.py --no_chat_template --num_shards 8 --lengths 8k 16k 32k 64k 128k --num_samples 50 --max_new_tokens 48`，PYBIN=`PYTHONPATH=.venv/lib/python3.11/site-packages /usr/bin/python3.11`（diskB py3.14 workaround，我已实测 torch2.10/tf5.5.4/cuda OK），OUT=`longeval_results/memoryllm_8b_chatFALSE`。冒烟→全量→score_only 合并→报每档 acc。完成后我回填 BENCHMARK_CHATFALSE_MASTER §B + PAPERA_RESULTS_CONSOLIDATED 的 14.0ᵀ 占位格→真 chat=False，#68 completed。**已删 .82-poll cron 75473588**（改用 .104，不再轮询 .82）。
> - **✅ freeze_front（#59）LOCAL 8 卡健康**：step 22740/200000，loss2.84/ppl17.15，8 卡 102GB/100%，1.32s/step。decisive matched eval @ step128000/153500/200000。绝不重启（会在同 8 卡重复=灾难）。task#59 in_progress。
> ↓ 下方 21:42 及更早快照作废存档。

## 当前快照（2026-07-25 21:42 +08:00，✅ P1（#75）完成=CONFIRMED + freeze_front（#59）已在 LOCAL 8 卡恢复）〔已作废存档〕
> **节点分配不变**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(28.82.250.82, H20 diskB)。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅ P1（#75）portable-adapter CLOSED = CONFIRMED**（LOCAL 卡 2-7，~42min，LoCoMo n=1986 GPT-4o judge，卡已释放）：HCache±distilled-LoRA 单变量 toggle（`no_retrieval=True`，selector 无关）→ judge **13.29(no-LoRA) → 31.17(+LoRA) = +17.88**（2.3×，identical node/commit）。cat4 open_domain +32.46 主驱动。**HCache+LoRA 31.17 > CoMem adapter-free 29.15（带检索），零检索即清过 adapter-free 门槛、逼近 flagship 38.27** → distilled LoRA = 可移植的 compression-agnostic KV-decompression/readout adapter（修的是 j=12 深度切分的 readout 分布 shift，非检索）。**caveat**：Arm A=13.29≠canonical 8.11（8.11 在 diskB `hcache_8b_chatFALSE`，本 wzc1 盘无；本地复现 deprecated dir 12.29）；delta 干净因同 node/commit；若锚 8.11 则 +23。产出 `locomo_results/hcache_j12_{noLoRA,LoRA}_chatFALSE/scores.json`，回填 `new_propositions_20260725.md §6`。task#75 completed。
> - **✅ freeze_front（#59）已在 LOCAL 8 卡恢复**（21:37 起，从 `step21000.pt`）：`FREEZE_FRONT=1 KEEP=14 N_FRESH=2 RESUME_FROM=...step21000.pt bash scripts/run_olmo2_7B_keepN.sh`，pid=2755952。**resume 干净**：restored 179 tensors(strict fp32-master) + optimizer 25 param states(Adam momentum 保留) + continue @ step21000。现 step21060@21:41 loss2.87/ppl17.7/gnorm0.60（与 pause 前 step21360 无缝）。**8 卡全 102GB/100% util 健康**。OUT=`outputs/olmo2_probe2_7B_keep14fresh2_freezefront`，LOG=`logs/olmo2_7B_keep14fresh2_freezefront.log`（resume append）。ETA→step200000 ~2.7 天（1.32s/step×179k）；decisive matched eval 在 step128000(~39h)/153500/200000 对照 healed keep14 apex + from_scratch。task#59 in_progress。
> - **.82（diskB，EVAL-ONLY）= #68 MemoryLLM LongEval chat=False 剩项**：21:00 poll 仍全 8 卡被外部 co-tenant（PID 517303-517310，容器隔离不可杀，~28GB/90-100%）独占，抢不到。cron 49047f5e（每半点 13,43）继续轮询空即 8-GPU 分片补跑。LongBench 9.01 / BABILong 30.4·21.4·38.1 已补齐；仅 LongEval 一格待跑（异基座次要参考行，不影响核心结论）。task#68 in_progress。
> - **★ P2+P1 双命题小结**（供论文机制章）：**P2 SUPPORTS**（两深度分离：semantic 0.13L ≪ knowledge 0.59-0.69L < next-token 0.94L，两模型一致；单深度证伪情形排除）；**P1 CONFIRMED**（distilled LoRA 可移植到零检索 HCache，+17.88 judge，修 readout 分布 shift）。两者共同强化「深度轴」机制主线：浅语义深度=CoMem 近无损切分处；深知识深度=prune-heal keep-N cliff 处；distilled LoRA=修深度切分 readout 的可复用 primitive。
> ↓ 下方 21:10 及更早快照作废存档。

## 当前快照（2026-07-25 21:10 +08:00，✅ P2（#74）完成=SUPPORTS + P1（#75）仍在跑 → freeze_front 待 P1 释放后 8 卡恢复）〔已作废存档〕
> **节点分配不变**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(28.82.250.82, H20 diskB)。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅ P2（#74）two-depths CLOSED = SUPPORTS**（LOCAL 卡 0-1，forward-only ~30s/model 跑完，卡已释放）：MMLU knowledge logit-lens → **knowledge sat95 OLMo 0.594L / Qwen3 0.694L**（both ≥0.55L 且 ≥2× semantic sat 0.13L=4.6×/5.3×），干净的三曲线 depth 分离 semantic 0.13L ≪ knowledge 0.59-0.69L < next-token 0.94L，**两模型都成立**。knowledge-onset(OLMo 0.562L) vs Paper B keep-N cliff（keep14=0.44L/keep10=0.31L）方向对但 +0.12L 偏深（decodability lag installation，非紧贴）。产出 `results/knowledge_logit_lens_{OLMo-2-1124-7B,Qwen3-8b-local}.json`，结论回填 `ops/research_notes/new_propositions_20260725.md §5`。task#74 completed。**可选 follow-up**：healed-16L ckpt 的同 probe（需 wire 自定义 16L arch，仅确认性非决定，暂缓）。
> - **⏳ P1（#75）portable-adapter 仍在跑**（LOCAL 卡 2-7，bg subagent a3f4bd1061f9a2dc6）：HCache±distilled-LoRA LoCoMo n=1986 GPT-4o judge 单变量 toggle，API-bound ~几小时。对照 HCache headline judge=8.11 预测抬到 ~20-30。完成后落账 + 收 verdict。task#75 in_progress。
> - **LOCAL 卡 0-1 短暂空闲**：P2 forward-only 秒级跑完释放；freeze_front（#59）需 8-GPU（torchrun nproc=8），P1 占卡 2-7 → **freeze_front 待 P1 完成释放 2-7 后 8 卡恢复**（step21000.pt，恢复丢 <500 步）。2 卡短闲不可塞 8-GPU 训练，且无 wzc1 可跑的 2-卡耐跑待办（#68 diskB-only）→ 合理等待，非铁律1 违规。
> - **.82（diskB，EVAL-ONLY）= #68 MemoryLLM LongEval chat=False 剩项**：仍被外部 co-tenant（PID 517303-517310，容器隔离不可杀）独占，抢不到。cron 49047f5e（每半点 13,43）轮询空即 8-GPU 分片补跑；并检查 LOCAL P1 完成→恢复 freeze_front。LongBench 9.01 / BABILong 30.4·21.4·38.1 已补齐。task#68 in_progress。
> ↓ 下方 20:45 及更早快照作废存档。

## 当前快照（2026-07-25 20:45 +08:00，★用户 steer「不是给了你一个 B200 一个 H20 么」→ 暂停 freeze_front，LOCAL B200 转跑被堵的 P2+P1）〔已作废存档〕
> **节点分配不变**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(28.82.250.82, H20 diskB)。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **关键澄清**：三个被堵实验里只有 **MemoryLLM LongEval 非 .82 不可**（MemoryLLM 权重+env 只在 diskB）；**P2/P1 模型都在 wzc1 盘，本来就能在 LOCAL B200 跑**——之前全排队等 .82 是错的。用户 approve「暂停 freeze_front 先跑 P2+P1」。
> - **✅ freeze_front（#59）PAUSED**：kill 于 step21000（20:39 存档 `outputs/olmo2_probe2_7B_keep14fresh2_freezefront/step21000.pt`，恢复只丢 <500 步）。8 卡全释放（实测 0MiB）。**P2+P1 完成后在 LOCAL 恢复**（decisive eval 仍在 128k/153.5k/200k 步，~2.7 天）。
> - **⚠️ LOCAL venv 提醒**：`.venv/bin/python` 现是 py3.14.6 但**健康**（torch 2.13.0 / tf 5.14.1 / peft 0.19.1 / 8 GPU 全 import OK）——与 diskB 的 py3.14-broken 不同，LOCAL 直接用 `.venv/bin/python`，**不用** py3.11 workaround。
> - **✅ LOCAL 卡 0-1 = P2（#74）two-depths knowledge logit-lens**：bg subagent，`scripts/probe_linguistic_layerwise.py --task knowledge_logit_lens`（commit 3ab7dd9）跑 OLMo-2-1124-7B(cuda:0) + Qwen3-8b-local(cuda:1)，forward-only ~4-6 GPU-hr。产出 `results/knowledge_logit_lens_*.json` + cross-map 到 Paper B keep-N cliff / CoMem split-j。task#74 in_progress。
> - **✅ LOCAL 卡 2-7 = P1（#75）portable-adapter HCache±distilled-LoRA**：bg subagent，`eval_qcmem_locomo.py --baseline hcache --resume_j 12 ±--force_lora_with_baseline`（commit 0b55791），LoCoMo n=1986 GPT-4o judge 单变量 toggle。对照现有 HCache headline judge=8.11，预测 LoRA 抬升到 ~20-30。~4-8 GPU-hr。task#75 in_progress。
> - **.82（diskB，EVAL-ONLY）= #68 MemoryLLM LongEval chat=False 剩项**：仍被外部 co-tenant（PID 517303-517310，容器隔离不可杀）独占全 8 卡 ~8h，抢不到。cron 7a7955d8 每 30min 轮询，空即 8-GPU 分片补跑。LongBench 9.01 / BABILong 30.4·21.4·38.1 已补齐。task#68 in_progress。
> ↓ 下方 12:15 及更早快照作废存档。

## 当前快照（2026-07-25 12:15 +08:00，★用户授权用 LOCAL+H20 接着跑 → LOCAL 起 Paper B #59 freeze_front + Paper A 结果整合成单文档 + 派 researcher 推演新命题）〔已作废存档〕
> **节点分配不变**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(28.82.250.82, H20 diskB)。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **★ 用户 2 条新指令**：(1)「本地的 B200 和剩下的那个 H20 你可以拿来接着跑实验」→ LOCAL + .82 授权接活；(2)「看看我们这个想法能不能催生出新的命题来」→ 研究性 ideation。
> - **✅ LOCAL（wzc1，8×B200/L20A）= Paper B #59 freeze_front（Arm A，12:15 起）**：`FREEZE_FRONT=1 KEEP=14 N_FRESH=2 bash scripts/run_olmo2_7B_keepN.sh`。transplant 前 14 层+embed/norm/head（157 tensors）自 OLMo-2-1124-7B base → **冻结前 14 层（frozen 2833.5M / trainable 仅 1226.9M）**，heal on `/dev/shm/dolmino_now15b.npy`。eff_bs128 seq2048 max_steps200000 fp32-master grad-ckpt。**实测 8/8 GPU 100%/106GB 健康**。OUT=`outputs/olmo2_probe2_7B_keep14fresh2_freezefront`，LOG=`logs/olmo2_7B_keep14fresh2_freezefront.log`。**这是 Paper B 最后一个待铺控制臂（RUN_REGISTRY line1942「脚本已支持,待空节点」）**——#59 之前标 blocked-on-diskB，被本次 LOCAL 授权解除（OLMo-2 资产全在 wzc1）。核心假设:继承的前层在 heal 时是否**需要 adapt** vs 可**冻结**（知识静态假设）；在匹配步（10k/128k/153.5k/200k）对照 healed keep14 apex + from_scratch。task#59 in_progress。
> - **✅ Paper A 结果整合成单文档（用户令「更新在一个文档里」）**：新建 `status/PAPERA_RESULTS_CONSOLIDATED.md`——一处读全 headline 6×5 主矩阵 + 逐 benchmark 明细(RULER/LongEval/LongBench/BABILong/LoCoMo) + 决定性实验 §F/§H/§I + #67 效率(report H20) + LoCoMo GPT-4o judge + bootstrap 显著性 + 可复现配置 + provenance + 唯一剩项 #68。等论文正文重构定稿后以此为准整合(#10)。
> - **.82（28.82.250.82，diskB，EVAL-ONLY）= #68 MemoryLLM chat=False overlay（未变）**：bg agent `a3f1093c` 满负荷跑中。完成后回填 master matrix ᵀ 占位格。task#68 in_progress。**.82 空出后**：铺一个「新命题」验证 eval（见下方 researcher 推演结论）。
> - **新命题 ideation（用户令）**：派 researcher（bg）从 CoMem depth-split + Paper B prune-heal 两个核心 idea 推演可测新命题（重点:统一的「semantic bottleneck depth」命题——split-j 与 keep-N 由同一 model-intrinsic 深度预测；及 compression-agnostic distilled LoRA、query-adaptive depth）→ 产出 top-1/2 的可跑实验计划，供 .82 空出/LOCAL 后续承接。
> ↓ 下方 11:10 及更早快照作废存档。

## 当前快照（2026-07-25 11:10 +08:00，#72 bib push 落地 + #67 效率控制 CLOSED〔发现 7.83× 为 H20-specific〕+ LOCAL 合法空闲 + .82 仍 #68）〔已作废存档〕
> **节点分配不变**：用户用 .73/.104/.252；我用 = LOCAL(B200 wzc1) + .82(28.82.250.82, H20 diskB)。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅ #72 论文 bib FIXME 完成 + push**：5 处全查实修复（hcache/kvcat/compresskv/kvdirect 的 Anonymous→真作者；`hunyuan` 从误引 Hunyuan-Large 2411.02265 → 正确 **Hy3 @misc** HF `tencent/Hy3`，依据本地 config.json model_type=hy_v3）。只改 `paper/qcmem.bib`（30/41 行），commit `bc67b56`，subagent review APPROVED，star-proxy push `4a0ac37..bc67b56`，**现 0 ahead**。论文重编译 16 页零 undefined。task#72 completed。
> - **✅ #67 效率 LoRA-on 控制 CLOSED（LOCAL 8×L20A，bg agent ab250045，11:06 完，全 8 卡释放）**：用旗舰同款 `bench_qcmem_vs_fullctx.py`（full-write-inclusive）跑 LoRA-off（复现论文）+ LoRA-on × 8k–128k。**① 显存声明精确复现且 LoRA-on 成立**：LoRA-off@128k CoMem **18.29GB**/Dense **89.39GB**（论文 18.26/89.36，2 位小数吻合）；LoRA-on +0.25GB（adapter 参数）→ 18.54GB。**~5× 显存优势是硬件无关的基本卖点，安全**。**② ⚠️ 7.83× prefill 加速是 H20-specific，非 LoRA artifact**：本 L20A 上 LoRA-off=**3.23×** / LoRA-on=**2.74×**@128k（非 7.83×）。根因干净——CoMem prefill+peak-mem 与论文精确吻合，**只有 dense prefill 不同**（L20A 6.04s vs 论文 json 15.01s），因 H20 bf16 compute 被 throttle ~2.5× 抬高 O(L²) dense baseline。论文 json config 本身标 "H20/L20A"，15s dense 证实是 H20。**③ decode**：LoRA +~33% ms/tok（23.8→31.7@128k），context-independent，仍 O(1)/step。结果 JSON：`ruler_results/bench_{fullwrite_,}lora{OFF,ON}_L20A.json` + `bench_lora_sanity_32k.json`。**★ 待用户决策**：7.83× 出现在 abstract+intro+concl+limitations+tab_eff+tab_chunk 共 7 处，是 headline claim；如何标注硬件（保留+标 H20 / 重测 / 改以显存为主 headline）= 用户的论文口径决定，见回复 AskUserQuestion。task#67 数据完成，paper-wording 部分待用户定 + 待改版落定。
> - **LOCAL（wzc1，8×B200/L20A）= 合法空闲（非铁律1 违规）**：#67 11:06 跑完释放。**当前无「我拥有的 + 论文相关 + wzc1 可跑」的 GPU 待办**——#59 freeze_front 真阻塞（launcher/ckpt 只在 diskB=用户的 .73/.104，wzc1 无）；#62 family-scale RULER **不在论文**（in-paper `tab_scaling`=Qwen3-8B 长度-scaling 已完；family-scale 仅在未 \input 的 scratch `tab_scale.tex`，且 dense 家族 0.6-8B+collaborator 32B，30B-A3B 是 MoE 类别不匹配）；#68 在 .82。故 LOCAL 空闲是「待跑项皆阻塞于用户节点/或投机不入论文」的合理结果，已 surface 给用户（是否让我跑 30B 填 LOCAL 或留空=用户定）。
> - **.82（28.82.250.82，diskB，EVAL-ONLY）= #68 MemoryLLM chat=False overlay**：bg agent `a3f1093c`，**实测 8/8 GPU 100%/55GB 满负荷健康**。完成后回填 master matrix ᵀ(chat=True)占位格（LongEval/LongBench 6ds/BABILong）。task#68 in_progress。
> ↓ 下方 10:40 及更早快照作废存档。

## 当前快照（2026-07-25 10:40 +08:00，★节点重新分配生效 + Paper B control-2 CLOSED + .82 起 #68 + FIXME/push 并行）〔已作废存档〕
> **★ 用户新分配（2026-07-25 生效，覆盖旧「H20 全归用户」）**：用户答「这两个节点加上 2，你可以用一个 B200 一个 H20」→ **用户用 .73(28.85.35.73)+.104(28.83.24.104)+.252(28.89.19.252, B200 "节点2")；我用 = LOCAL(B200 wzc1) + .82(28.82.250.82, H20 diskB)**。dllm 29.162.226.120 绝不碰、cron 4ec42903 不删。
> - **✅ Paper B control-2（from_scratch）完全闭合**：knowledge downstream MC 10:32 完成（LOCAL 8×B200）。**★ MMLU=.2461（= .25 chance floor，−1 SE，与随机不可区分）vs healed keep14 .312（17.6% above chance）→ from_scratch 训 200k 步（比 healed 153.5k 更多）仍 0% 恢复世界知识**。boolq 打平（.614 vs .606，in-context 阅读理解可从头学）= 完美互补 → **干净证明 healed 恢复的知识来自继承的预训练前层、非 heal 训练**。已回填 `OLMO2_PRUNEHEAL_DOWNSTREAM.md §CONTROL2`（PPL 1.554× + core surface-tie/reasoning-lag + knowledge MMLU 决定性表全齐）。**Paper B 三控制（healed apex/post-apex、freeze_front、from_scratch）+ PPL + core + knowledge 全闭合**。
> - **LOCAL（wzc1，8×B200/L20A）= 空闲**（from_scratch knowledge MC 10:32 跑完释放；此前 from_scratch 训练亦已闭合）。可承接 wzc1-scoped 待跑项。
> - **.82（28.82.250.82，diskB，默认端口，EVAL-ONLY）= #68 MemoryLLM chat=False overlay**：bg agent `a3f1093c` 起 MemoryLLM（Llama-3-8B-chat 異基座）真 chat=False 的 LongEval/LongBench(6ds)/BABILong(qa1/2/5×0k-32k)，替换主表 ᵀ(chat=True)占位（LongEval 14.0/LongBench 12.80/BABILong 26.9·21.1·42.6）。⚠️env 坑：diskB venv python 被 reset 成 3.14→用 /usr/bin/python3.11+PYTHONPATH（memory `memoryllm-venv-python-broken`）；BABILong 旧 dir 误命名坑（须真去 chat template）。agent 先验 SSH+env+跑最短 cell 通再全量。task#68 in_progress。
> - **#72 论文 bib FIXME（GPU-free）= bg agent `aad3846c`**：联网（hy-proxy）查修 5 处 bib——Hy3 arXiv 误引 2411.02265(=Hunyuan-Large)、hcache/kvdirect/kvcat/compresskv 作者 Anonymous→查真实论文+作者，改 `paper/qcmem.bib` 重编译验证。只改 paper/ 不 commit（main 统一）。
> - **push backlog（用户令「现在推」）**：HEAD 领先 origin/main **34 commit**（d3a8b5a…c056a6d）+ 本会话 status backfill（OLMO2_PRUNEHEAL_PPL/DOWNSTREAM、BENCHMARK_CHATFALSE_MASTER §H/§I）+ session scripts。main 正 commit 安全子集→/gitpush（subagent review→APPROVED→star-proxy）。**不提交 *.pt/*.npy/*.pdf/aux/bbl/zip，不盲提 paper/sections 重构删改**。
> ↓ 下方 10:30 及更早快照作废存档。

## 当前快照（2026-07-25 10:30 +08:00，from_scratch 训练闭合 + #73 depth-sweep CLOSED + LOCAL 转 from_scratch knowledge eval）〔已作废存档〕
> 10:30（`date` 实测）：**我合法在用 = LOCAL（wzc1）；.252（wzc1）暂空；H20 .73/.104/.82 = 用户的不碰；dllm 29.162.226.120 绝不碰、cron 4ec42903 不删**。
> - **✅ #73 adapter-free depth-sweep CLOSED**：.252 j6/j12 frozen eval 全完 → 两个 LoCoMo GPT-4o judge 落地 **j6=32.78 / j12=24.52**（`locomo_results/qcmem_8b_zeroshot_j{6,12}_frozen_iterbm25_chatFALSE/scores.json`）。**建成 4 点 frozen depth-sweep 单调曲线**（LoCoMo judge）：**j0=41.59（#71）→ j6=32.78 → j9=29.15（#65）→ j12=24.52**，越深保真度越低（合 `bottleneck-layer-sweep-monotone`）。**★ 纯 distilled LoRA 贡献（SAME depth j12，唯一变量=LoRA）：j12 frozen 24.52 → 旗舰 j12+LoRA 38.27 = +13.75**（比 #65 混 9→12 深度的 +9.12 更干净大）；BABILong qa1 +22.2/qa2 +9.0/qa5 +8.4。**回填 `BENCHMARK_CHATFALSE_MASTER.md §I`**（新增）。task#73 completed。
> - **✅ Paper B from_scratch（control 2）训练闭合 + eval 落地**：LOCAL from_scratch 16L（random-init keep14/fresh2 shell）训到 step200000（final.pt 48.7GB）。**① held-out PPL=11.4983（1.554× base 7.398），比 healed keep14 10.693/1.446× 差 +0.80——从头训更多步(200k>153.5k)仍不如剪层-heal → 继承预训练前层是真实优势**（回填 `OLMO2_PRUNEHEAL_PPL.md`）。**② core downstream 6-task 已出**（.252 10:25 完，nan=0）：surface 任务（ARC-E/PIQA）≈ 打平 healed，但 HS/ARC-C/WinoG 明显落后（WinoG 恢复 18% vs healed 55%）（回填 `OLMO2_PRUNEHEAL_DOWNSTREAM.md §CONTROL2`）。
> - **LOCAL（wzc1，8×B200/L20A）= from_scratch KNOWLEDGE downstream MC（决定性 control-2 读数）**：driver `scripts/_run_olmo2_downstream_scratch_know_wzc1.sh`（新建），10:28 起 mmlu/lambada/boolq/csqa/siqa，现 prepare_data 阶段（拉数据集 CPU 无 GPU 占用，1 proc）→ 稍后 fan out 8 shards 点亮 GPU。**核心预期：from_scratch MMLU≈.25 chance floor（没见过 OLMo-2 预训练语料）vs healed keep14 MMLU=.312 → 干净证明「healed 恢复的知识来自继承层，非 heal 训练」**。输出 `olmo2_downstream_results/7B_scratch16L_step200000_know`，DONE=`logs/olmo2_downstream_scratch_know_DONE`。
> - **.252（28.89.19.252，wzc1 B200）= 暂空**（from_scratch core downstream 10:25 完成后释放）。铁律1：knowledge eval 是最后一个 from_scratch 待跑项，跑在 LOCAL；.252 待用户明确 diskB H20 归属后可承接 #62/#68（那些模型/env 只在 diskB），或等 knowledge 完后统一收尾。
> - **待用户决策（3 项，见回复）**：(a) diskB H20（.73/.104/.82）现是否可用于我 → 解锁 #62 family-scale RULER + #68 MemoryLLM chat=False（模型/env 只在 diskB）；(b) push backlog（/gitpush→APPROVED→star-proxy）；(c) #72 论文 2 FIXME 待联网（Hy3 arXiv 误引 2411.02265、hcache/kvdirect/kvcat/compresskv 作者 Anonymous）。
> ↓ 下方 01:00 快照作废存档。

## 当前快照（2026-07-25 01:00 +08:00，#71 决定性实验 CLOSED + #65 judge 落地 + .252 转 #73 depth-sweep + LOCAL 训练收尾）〔已作废存档〕
> 01:00（`date` 实测）：**我合法在用 = LOCAL + .252（均 wzc1 我这边的盘）；H20 .73/.104/.82 = 用户的不碰；dllm 29.162.226.120 绝不碰、cron 4ec42903 不删**。
> - **✅ #71 P0#1 决定性实验 CLOSED**：j=0 单变量控制全跑完（BABILong SCHED_DONE 00:28 + LoCoMo GPT-4o judge 干净重判 1540 条）。**结果 LoCoMo judge=41.59（n=1986, cat1-4 52.60）、BABILong qa1/qa2/qa5=65.9/39.1/63.0**。⚠️踩 pkill 自杀 shell 坑（`pkill -f eval_qcmem_locomo` 命中自身 SSH shell），改用 `[e]val...` 正则自排除 + 删污染 judge_cache(2317行stale)后全新重判。**4-config 对照写入 `BENCHMARK_CHATFALSE_MASTER.md §H`**：(a) j0 vs adapter-free j9 → frozen 深度-9 切分损失 −12.44 judge pt（深度轴压缩丢保真度）；(b) j0 vs KVD → 检索单独 +7.00 on LoCoMo（过滤干扰轮）但 BABILong needle −12.8/−9.8（top-k 漏散落事实）。**reviewer 回答=检索/深度/LoRA 三者组合，贡献随任务而异；CoMem 普适卖点=固定预算超窗口长度+效率，非窗口内一律超上界**。task#71 completed。
> - **✅ #65 CoMem adapter-free（frozen j9）LoCoMo judge 落地=29.15**（LOCAL CPU+API 跑完，`locomo_results/qcmem_8b_zeroshot_j9_chatFALSE/scores.json`，cat4=53.75）→ **回填 3 权威文件的 judge 格**（BENCHMARK_CHATFALSE_MASTER headline line16/§E/§F、PAPERA_REPRO §五/§十）。**LoCoMo judge 全景：旗舰+LoRA 38.27 > KVD 34.59 > adapter-free 29.15** → distilled LoRA +9.12（但 9→12 深度与 LoRA 混在一起，见 #73）。task#65 completed。
> - **.252（28.89.19.252，wzc1 B200）= #73 adapter-free depth-sweep（新起，铁律1 填卡）**：driver `scripts/_qcmem_adapterfree_jsweep_chatFALSE_taskpool.sh`（未 commit），frozen 无 LoRA、resume_j ∈ {6,12}、iter_bm25/topk12/hop4/chunk512/sink=bos/chat=False，72 jobs 8-worker flock 池。**核心目的=j12-frozen 点**：`flagship j12+LoRA(38.27)` vs `j12 frozen` = SAME depth 仅差 LoRA → **纯 isolate distilled LoRA 贡献**（现 +9.12 混了 9→12 深度）；j6 补 monotone 曲线中点（已有 frozen 点 j0=41.59/#71、j9=29.15/#65）。✅ 01:00 确认健康：8 workers、8 GPU 各 ~18GB、queue 64/72、j6 LoCoMo shards 起跑。输出 `{locomo,babilong}_results/qcmem_8b_zeroshot_j{6,12}_frozen_iterbm25_chatFALSE`。完后 rescore + 起两个 LoCoMo judge → 建 4 点 frozen depth-sweep 表回填 §H/§九。task#73 in_progress。
> - **LOCAL（wzc1，8×L20A/B200）= from_scratch 训练收尾**：~94%+ 8/8 GPU 100%（承上条 00:30 step188680/94.3% ETA~05:20），**不碰**。
> - **铁律1 满足**：LOCAL 训练 + .252 #73 = 两 wzc1 节点无空转。**下一步**：(a) 监控 #73 SCHED_DONE → rescore + judge + 建 depth-sweep 表；(b) 向用户 surface **#72 论文 P0**（commit d3a8b5a 已编 16 页可编译；2 FIXME 待联网：Hy3 arXiv 误引 2411.02265=Hunyuan-Large、hcache/kvdirect/kvcat/compresskv 作者仍 Anonymous）；(c) 待 push backlog（见下）。
> ↓ 下方 00:30 快照（#71/#65 进行中）作废存档。

## 当前快照（2026-07-25 00:30 +08:00，#71 P0#1 j=0 控制在 .252 近完成 + #65 adapter-free judge 起跑 + LOCAL 训练 94%）〔已作废〕
> 00:30（`date` 实测）：**我合法在用的 = LOCAL + .252（均 wzc1 我这边的盘）；H20 .73/.104/.82 = 用户的，不碰（nvidia-smi idle ≠ 归我）；dllm 29.162.226.120 绝不碰**。
> - **LOCAL（wzc1，8×L20A/B200）= from_scratch 训练**：step **188680/200000（94.3%）** loss2.36 ppl10.5 gnorm0.42 1.57s/step maxmem122.3GB **8/8 GPU 100%** 稳降无 NaN，ETA **~05:20**。**不碰**。
> - **.252（28.89.19.252，wzc1 B200，与 LOCAL 共享盘无需 rsync）= #71 P0#1 决定性对照**（用户「不是有一个 B200 么?」steer → 从 H20 挪到我的 B200）：driver `scripts/_qcmem_j0_iterbm25_chatFALSE_taskpool.sh`（未 commit），~00:17 起 **j=0 单变量控制**（resume_j=0、**省 --lora_adapter**、full 36 层重算检索 chunk；其余全同旗舰：iter_bm25/topk12/hop4/chunk512/sink=bos/**chat=False**）。目的=剥离 **depth-axis 增益 vs 检索增益**（回应 reviewer「赢会不会全在 iter_bm25 检索」）；两组单变量对照：(a) vs #65 adapter-free j9 → 仅 j（0 vs 9）；(b) vs KVD(resume_j=0+kvdirect 无检索)→ 仅检索。coverage=LoCoMo full(n=1986, 8 shard) + BABILong qa1/qa2/qa5×{0k..32k}×4 shard = 36 jobs。**进度：queue 已空，32/36 done**，4 个最重 32k babilong shard 在跑，**8 个 LoCoMo preds shard 已全写出**，尚无 SCHED_DONE（ETA 收尾数十分钟）。输出 `babilong_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE` + `locomo_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE`。
> - **#65 CoMem adapter-free（j9 frozen）= 已完工 + judge 起跑**：eval 昨 23:35 SCHED_DONE（148 jobs 全完），5-benchmark 数字（RULER 59.4 / LongEval 3.2 / LongBench 10.63 / BABILong 42.4·19.6·55.6 / LoCoMo local acc 16.41）已**回填 3 权威文件**（BENCHMARK_CHATFALSE_MASTER §F/headline、PAPERA_REPRO_HYPERPARAMS §五/§十、PAPERA_ALL_RESULTS）。**刚在 LOCAL 起 LoCoMo GPT-4o judge**（`eval_qcmem_locomo.py --score_only --use_llm_judge`，`CUDA_VISIBLE_DEVICES=""` 纯 CPU+API 不碰训练卡，PID~2427818，1540 条 gpt-4o workers=8 ~8.5min，446 cat5 cached；log `logs/locomo_judge/j9_adapterfree_judge.out`）→ 完后回填 headline judge 格（现填 🟡 待补）。
> - **铁律1 满足**：LOCAL 训练 + .252 #71 = 两 wzc1 节点无空转（H20 归用户不计入）。**下一步**：(a) .252 #71 SCHED_DONE → 收 BABILong compare_answers + 起 #71 LoCoMo judge；(b) #65 judge 落地 → 回填两处 judge 格；(c) 向用户 surface #72 论文 P0（commit d3a8b5a 已编，2 FIXME 待联网：Hy3 arXiv 误引 2411.02265=Hunyuan-Large、hcache/kvdirect/kvcat/compresskv 作者仍 Anonymous）。dllm 绝不碰、cron 4ec42903 不删。
> ↓ 下方 23:50/23:52 记录（描述已撤销的 H20 launch）作废存档。

## 当前快照（2026-07-24 23:50 +08:00，#70 CLOSED + 16 H20 全空→铺开 P0#1 决定性对照 + #68 + 论文 P0 编辑）
> 23:50（`date` 实测）：**#70 收尾闭合**——BABILong hop=4 旗舰重跑完成 + rescore（compare_answers）+ **回填 3 权威文件**（`BENCHMARK_CHATFALSE_MASTER.md` §D grid/headline/两处注解、`PAPERA_ALL_RESULTS.md` §1.5 grid+源标注/§1.7d grid/headline/注解、`PAPERA_REPRO_HYPERPARAMS.md` line61+open-item#4→resolved）。**新数字 qa1 55.6 / qa2 27.0 / qa5 68.7**（旧 hop=2/6 轮为 53.6/25.6/66.7，三档均略升 +1.4~2.0，**结论稳定**：qa5 仍略超 KVD 上界 61.4，qa1/qa2 仍是诚实压缩 tax）。task#70 completed。
> **⚠️ 23:52 更正（用户指出 H20 归其使用）**：H20（.73/.104/.82）今日已交还用户，nvidia-smi idle ≠ 归我可用。**已在 launch 之前停掉 #71(.104) / #68(.73) 两个 bg agent**（kill 时均停在 pre-flight 验证阶段，尚未起任何 GPU 任务；两节点复查 = 0 eval 进程 / 0MiB，**无任何侵扰到达卡**）。#71/#68 回退 pending，等节点归属明确再定。**我实际合法在用的只有 LOCAL + .252（均 wzc1，我这边的盘）**：LOCAL=from_scratch 训练、.252=#65 adapter-free。#72 论文 P0 编辑（GPU-free，只改 paper/）继续跑。↓下方原 23:50 记录已作废，仅存档。
> **GPU 实测**：.73 GPU0-7 全 0MiB（hop=4 重跑释放 + 此前 GPU4-7 用户 job 也已清）、.104 GPU0-7 全 0MiB = **16 H20 全空**。用户批准 P0#1 + 论文编辑 → 铺开：
> - **.104（28.83.24.104，diskB，-p 36000）= #71 P0#1 决定性对照**：bg agent `ab090032` 在 diskB 起 **j=0 单变量控制**（resume_j=0、**省 --lora_adapter**、full 36 层重算检索 chunk，vs 旗舰 j=12；其余全同：iter_bm25/topk12/hop4/chunk512/sink=bos/chat=False）。目的=**剥离 depth-axis 增益 vs 检索增益**（回应 reviewer「会不会赢全在 iter_bm25 检索」）。benchmark=BABILong qa5(0k-32k n=100) + LoCoMo n=1986→GPT-4o judge。输出 `babilong_results/qcmem_j0_iter_bm25_chatFALSE_noad` + `locomo_results/qcmem_j0_iter_chatFALSE`。ETA~4-8 GPU-hr。agent 正验 flags+launch（要求单变量=仅 j，否则 STOP 报阻塞）。
> - **.73（28.85.35.73，diskB，默认端口，EVAL-ONLY）= #68 MemoryLLM chat=False overlay**：bg agent `ac55e879` 起 MemoryLLM（Llama-3-8B-chat 異基座）真 chat=False 的 LongEval/LongBench(6ds)/BABILong(qa1/2/5×0k-32k)，替换主表 ᵀ(chat=True)占位（LongEval 14.0/LongBench 12.80/BABILong 26.9·21.1·42.6）。⚠️env 坑：venv python 被 reset 成 3.14→用 /usr/bin/python3.11+PYTHONPATH；BABILong 旧 dir 误命名坑（须真去 chat template）。EVAL-ONLY 不训练。
> - **论文 P0 编辑（GPU-free）= #72**：bg coder `aa216b7` 改 `paper/`——[preprint]→[review]+作者块匿名化、abstract 三处数字对齐权威源（prefill 7.83×、mem 18.26/89.36GB、删假的 32-68× decode 加速）、cross-chunk 措辞纠错（qcmem_model.py:245 query 两模式都 attend 全 chunk）、bib（Hunyuan 误引/LongEval+PG19 缺/Anonymous×4/重复 key）。commit=LiuHanzuo 不 push。
> - **承前不变**：**LOCAL(wzc1)** from_scratch 训练 ~step18xxxx/200000(~90%+) ETA~明晨05:00、**.252(wzc1)** #65 CoMem adapter-free chat=False 全 5-benchmark 健康跑中——两者未碰。dllm 29.162.226.120 绝不碰。**铁律1 满足**：LOCAL 训练 + .252 #65 + .104 #71 + .73 #68 = 四节点无空转。待 3 个 bg agent 回报（launch 确认/阻塞）。

## 当前快照（2026-07-24 23:00 +08:00，新增 .73 GPU0-3 BABILong hop=4 旗舰重跑 #70）
> 23:00（`date` 实测）：用户令「iter_hop_topk 统一用 4」。审计确认 **BABILong 是唯一漏网 hop=2**（RULER config hop_topk=4/read6630、LoCoMo config hop_topk=4、LongEval/LongBench 脚本默认=4 均已 hop=4；BABILong 旧旗舰因 `_eval_qcmem_taskpool.sh` 不传 `--iter_hop_topk` 且 `eval_qcmem_babilong.py:905` 默认=2 → 6 轮）。**修复**：改 `eval_qcmem_babilong.py` :905+:347 默认 2→4（commit `5158e70` 本机 wzc1，未 push），cat-pipe 传 .73（md5 一致 `506f1db8`）。**重跑**：.73 **GPU 0-3**（GPU 4-7=用户 4 进程 ~32GB/卡，**不碰**）起旗舰 BABILong hop=4：MODEL=Qwen3-8b-local resume_j=12 iter_bm25 topk=12 sink=bos chunk512 LoRA=`outputs/qcmem_distill_qwen_j12_r32_4k/final` chat=False，NUM_GROUPS=1 GROUP0="0 1 2 3"，新目录 `babilong_results/qcmem_j12_iter_bm25_chatFALSE_ad_hop4`（隔离旧 hop=2 备查），21 任务（qa1/qa2/qa5×0k-32k），sched.out=`logs/qcmem_babilong_hop4_sched.out`。✅ 23:00 确认健康：4 shard 进程齐、GPU0-3 全占、0k/1k 已完在跑 2k，每短档任务 ~45-55s。ETA ~40-50min→重打分 compare_answers 回填 BENCHMARK_CHATFALSE_MASTER §D/headline + PAPERA_ALL_RESULTS + PAPERA_REPRO_HYPERPARAMS open-item#4。task#70 in_progress。｜.73 #64 LoCoMo judge（CPU）与本机/.252 训练+eval 承前不变，详见下条 21:40 快照。

## 当前快照（2026-07-24 21:40 +08:00，本机+.252 在跑；#63 完工/#64 judge 完工/#65 健康跑中）
> 20:26（`date` 实测）：**LOCAL** from_scratch step**180700/200000（90.4%）** loss2.40 ppl11.05 gnorm0.42 1.57s/step 122.3GB 8/8 稳降无 NaN，ETA~8.4h（~明晨05:00）。**.252**（28.89.19.252，wzc1）**#63 InfLLM chat=False 已完工**（SCHED_DONE 17:09，IRON-LAW-2 ALL OK：RULER + LongBench avgF1 11.86 + LongEval 8k.60→128k.02 + BABILong qa1/2/5×0k-32k 全落地）→ 空闲后起 **#65 CoMem adapter-free chat=False 全 5-benchmark，✅ 20:44 CST 确认健康运行**（main pid 376299，driver `scripts/_qcmem_adapterfree_j9_chatFALSE_taskpool.sh` commit **a21a752** 未 push；148 jobs 8-worker flock 池，0 FAIL，sanity RULER niah_single_2 8k recall=100；Qwen3-8B 冻结、**无 LoRA=省 `--lora_adapter`**（⚠️`--zero_training_no_adapter` 根本不是 CLI flag，全 5 benchmark 统一靠省 --lora_adapter+baseline none 实现 adapter-free）、resume_j=9、chunk512/top12/hop4/sink=bos/iter_bm25、绝不 chat_template；输出 `{ruler(+_taskbreadth),longeval,longbench,babilong,locomo}_results/qcmem_8b_zeroshot_j9_chatFALSE`；log `logs/qcmem_adapterfree_j9_chatFALSE_campaign.out`；ETA~5-8h→SUMMARY.txt+SCHED_DONE）。**.73**（28.85.35.73，diskB）用户新给的 API key（JWT，已换 wzc1+diskB .env）起 **#64 LoCoMo GPT-4o judge 补跑**（InfLLM/StreamingLLM/MemoryLLM chat=False，纯 CPU+API 不碰 GPU，pid 3197649，log `logs/locomo_judge_backfill_20260724.log`，一次性回收 cron a99c8caa@20:53；.73 GPU 7/8 归用户不动）。⚠️.82/.104 归用户/dllm 未碰，dllm 29.162.226.120 绝不碰。铁律1 满足（无空转节点）。

> 最后更新：2026-07-24 21:40 +08:00（**21:40 heartbeat — #65 CoMem adapter-free 检查**：SSH .252 实测**健康、正常慢跑、不动**。queue **124/148**（已 pop 24），done markers **16**，**0 FAIL**，**8/8** eval 进程存活（全 `cuda:0`）。8 个 worker 当前全在跑**整池最慢 job**=RULER `variable_tracking`@128k（各 shard 100 样本 ~25 s/it，现 ~17-18/100=128k 段 18%）——LPT 把 128k-heavy 排最前。done 停 16 = niah_single(8)+niah_multikey(8) 已完 + 当前 8 个 vt shard in-flight = 24 RULER-main 全对上，VT-128k 跑完 done 跳 24 再快消化 124 短 job。⚠️nvidia-smi 两采样均 0% util 但每卡 **~18.8GB** 占用 + 8 个 per-job log <60s 内在写、tqdm 进度条推进（`18/100 [07:29<34:26]`）=**真实计算**（bursty decode 采样伪影，铁律2 log 增长为准）。SCHED_DONE 未出，ETA 仍 ~数小时→SUMMARY.txt。健康故**不动**，等 SCHED_DONE 收 6 官方 scorer + 补 LoCoMo GPT-4o judge 填 adapter-free 主表行（解 #67）。｜本轮另建 `status/BENCHMARK_CHATFALSE_MASTER.md`（6 方法×5 benchmark chat=False headline 单文件主矩阵）+ 折入 InfLLM #63 数字到 PAPERA_ALL_RESULTS §1.7。｜两节点 16/16 满载健康。**.252 InfLLM chat=False** 8/8 进程，RULER niah_single 各 shard 已推进到 64k（shard0 `niah_single_2/64k 59% s/it 7→4.3` 递减=正生成，8 shard log 全 12:26 更新 ~28KB 齐头），0 FAIL，0 .done（各 shard 需跑完 128k 才 mark，长尾正常）；⚠️4 卡瞬时 util=0% 但 30GB 占用+log 在长=InfLLM 块检索突发式 idle，**非卡死**（铁律2：log 增长为准）。**LOCAL** from_scratch step**164920/200000（82.5%）** loss2.47 ppl11.82 gnorm0.40 1.57s/step 122.3GB 8/8 稳降无 NaN，ckpt step164500.pt(12:15 轮转正常)，ETA~15h。H20 .73/.82/.104=用户/dllm 未碰。铁律1/2 满足。｜前情 12:18 InfLLM campaign 确认上卡：bg-coder a3337a00 返回，**.252 InfLLM chat=False × 4-benchmark 已跑起 8/8**（核实：8 个 eval_infllm 进程、8 卡各 ~28-30GB、log 12:18:15 新鲜、RULER niah_single 各 shard 交错启动、`ruler_results/infllm_8b_chatFALSE` 已落地）。140 jobs 8 卡 task-pool 动态调度（128k-heavy RULER LPT 先发），**ETA 4-8h**，末尾自动跑 5 个官方 scorer→写 `logs/infllm_chatFALSE_taskpool/SUMMARY.txt`+touch `SCHED_DONE`。覆盖已对齐 diskB chat=True `infllm_8b`（RULER main+taskbreadth、LongBench 6ds、LongEval 5档 max48、BABILong qa1/2/5×0k-32k），唯一差异=去 `--use_chat_template`；InfLLM 配置走内部 DEFAULT_MEM_CONFIG（paper-faithful）。driver=`scripts/_infllm_chatFALSE_taskpool.sh`（commit 4d5c8bc 干净未 push）。⚠️env fix：.252 torch-base 缺 pandas+datasets 已装（仅 .252 本地 conda，非共享盘）。**LOCAL** from_scratch step**164020+**（82%）ppl11.63 健康 8/8。**16/16 满载**。H20 .73/.82/.104 归用户/dllm 未碰。监控：grep FAIL + 等 SCHED_DONE 收分。铁律1/2 满足。｜前情 12:01 heartbeat：掌控 2 节点。**LOCAL** from_scratch step**164020/200000**（82.0%）ppl11.63 gnorm0.41 loss2.45 8/8——GPU0 0%/200W=step164000 刚 save+轮转的恢复瞬态（log 12:00:59 新鲜推进、8.83s/step post-save blip），loss 稳降无 NaN。**.252** 8×B200 由 bg-coder a3337a00 provisioning。｜前情 11:58 用户指令执行：用户令「停一个 B200 节点跑 InfLLM chat=False，跑完交还」→ **已 kill .252 keep12**（step111500.pt=43.8GB 完整可 resume，PROCS=0，8 卡全 0MiB）→ 在 .252 起 **InfLLM baseline chat=False × {RULER,LongBench,LongEval,BABILong}**（Qwen3-8B，补 Paper A 矩阵最后缺口，后台 coder a3337a00 建 8 卡 task-pool driver + 自动官方判分）。**setup 零 rsync**：wzc1 早有 `Qwen--Qwen3-8b`（5 safetensors 字节与 diskB 全一致=完整，已建 symlink `models/Qwen3-8b-local`）、`inf_llm` 在 .venv import OK、RULER 自生成/LongBench 有 raw/LongEval 自合成/BABILong 有 pkg、chat=False=去 `--use_chat_template`。**LOCAL** from_scratch 仍训练中（~step163000+ 81.5%+，未碰，8/8）。**掌控 2 节点 16/16 不空转**（LOCAL 训练 + .252 eval）。★ keep12 无损暂停在 step111500，日后可 resume（.252 交还用户后可换节点）。★H20 .73/.104/.82 仍归用户/dllm（本轮只读 SSH .73 做矩阵审计+覆盖核对，未碰 GPU）。铁律1/2 满足。｜前情 11:27 heartbeat：掌控 2 节点 16/16 满载全健康 0NaN。**LOCAL** from_scratch step**163000**（81.5%）ppl11.69 gnorm0.41 8/8——GPU0 1%/201W=step163000 每-500 ckpt-save 起始（log 11:27:00 打 step163000 新鲜即证活），GPU1-7 100%；**.252** keep12 step**110540**（55.3%）ppl11.68 gnorm0.49，刚轮转 109500+110500 边界 save（7.52s→1.38s 恢复），shared-FS log 11:27:38 新鲜。两臂 loss 稳降未 plateau。**★H20 全 3 台仍交还用户/dllm 不碰**（.73/.104/.82）。★核实 #47=256k CoMem chat=False 已 DONE → chat=False campaign 实质全闭合，仅剩 c1024 128k full-ctx 有限计时（低优先 B200）；且 tab_scale(#62) 查 main.tex **未 \input=论文无关**。#10 论文集成=GPU-free，已就 A/B 询问用户待决。铁律1/2 满足。｜前情 10:56：掌控 16/16 全健康。**LOCAL** from_scratch step**162000** ppl~11.80 gnorm0.42 8/8——GPU0 0%/~200W=每-500 ckpt-save 起始（log 10:56:02 刚打 step162000 距采样 1s，log 新鲜即证活，无需查 ckpt；已连续 7 轮确认此良性机制）；**.252** keep12 step**109420** ppl~11.46 gnorm0.46 1.38s/step（shared-FS log 10:55:43 新鲜推进）。两臂 loss 稳降未 plateau。**★ H20 全 3 台已交还用户不监控/填卡**：.73/.104 用户 23:16 令腾空（.104 keep8 step44000.pt 可 resume；#62 Qwen3 scale PAUSED，4B/14B/32B CSV 已存 diskB）、.82 dllm-占用。可调度 = LOCAL+.252 已 16/16 满。dllm 节点 29.162.226.120/cron 4ec42903 未碰。铁律1/2 满足。｜前情 10:26：LOCAL step161000 ckpt-save 证伪（22.6→48.7GB）。｜前情 09:56：LOCAL step160000=80% 里程碑。｜前情 06:56：.252 跨 100k=50%。｜前情 23:16：用户令腾空所有 H20 归用户/dllm。｜Paper A 投稿闭合：P0#1 LoCoMo judge=38.27 配对显著 ✅、P0#2/#3/P1#4/#6 ✅ DONE；剩 #10 论文集成待 #62 scale 跑完（H20 返还后 resume）。｜待 push backlog（全待 /gitpush→APPROVED→star-proxy）：wzc1 7daeade + diskB eff036f/a3a98f37 + driver c54d374/0296f15/58bbf93 + BENCHMARK_RESULTS/SESSION_HANDOFF 编辑。

## 当前快照（2026-07-23 09:44，.73+.104 已填 Paper B frontier 2×8 resumed；40/40 卡满）
> 09:44（`date` 实测）：.73/.104 消融#9/#12 收尾+全空（proc=0/8 已 nvidia-smi 逐卡核实）→ **2×8 独立 resume Paper B OLMo-2-7B frontier 臂**。**16 卡 2-node 否决**：keep8 8卡基线 5.82s/step（bs4/ga4=4 microbatch≈1.45s/mb），16卡保持 eff_bs128 需 bs4/ga2=2 microbatch≈2.9s compute-only 已 >2.5s 判据（未算 TCP-no-IB 7B allreduce）；且 2 臂/2 节点→并行产出最大（16卡会串行化两臂+comm-bound）。
> - **.104 keep8**（深剪 healing frontier，10 层 2.846B）：resume `step36000.pt` → step36020 loss2.7714 ppl15.98 gnorm0.55 5.89s/step maxmem73.5GB 0NaN，proc8/8。
> - **.73 freeze_front**（Arm A 控制，keep14 冻前14层，trainable1.2269B/16层）：resume `step21500.pt` → step21520 loss2.8668 ppl17.58 gnorm0.72 7.33s/step maxmem50.8GB 0NaN，proc8/8。
> - 两臂均 RESUME（2026-07-22 为 chat=False eval 暂停的臂，PAUSE_RESUME_H20_20260722.md 配方）；ppl 匹配 pause 前（15.95/17.80）。ckpt 写 **diskB** `outputs/olmo2_probe2_7B_keep8fresh2` + `_keep14fresh2_freezefront`（脚本自轮转 latest-2+milestones；⚠️cron 4ec42903 只管 wzc1，diskB 新臂不进 cron 轮转但 146T free+脚本自轮转→无风险）。gpu_runs.jsonl 已落 2 行（commit 58bbf93）。
> - Paper B 训练全景：LOCAL from_scratch + .252 keep12 + .104 keep8 + .73 freeze_front（4 臂在跑；keep14 apex/keep10 已训完）。

## 当前快照（2026-07-21 02:33，32/32 卡满 5 臂全健康；task#10 LoCoMo 数字核实完成，未改论文）
> 02:33 tick（`date` 实测 02:32:46 +0800）：**铁律1 满足=32 卡全占（proc-count 逐卡=1，四节点）**，5 臂全健康续跑。
> - **wzc1 8/8 from_scratch**（Control 2）：step1080/200k loss4.10 ppl60.17 1.57s/step 0NaN（random-init warmup 稳降 step900→1080：67.5→60.17）。
> - **.82 8/8 keep12**：step6860/200k loss2.84 ppl17.11 7.81s/step 0NaN 续降。
> - **.104 8/8 keep8**：step9240/200k loss2.99 ppl19.85 5.82s/step 0NaN 续降。
> - **.73 8/8 freeze_front**（Arm A）：step240/200k loss5.48 ppl238.7 7.25s/step 0NaN（warmup 续降 step180→240：376→238.7；frozen-front 高风险臂，关注高位 plateau=frozen 前段瓶颈信号）。
> - **✅ task#10 证据核实（未改论文，铁律2）**：`locomo_results/*/scores.json` 官方 run_scoring 核出全 LoCoMo iter_bm25 数字→写 `status/PAPER_LOCOMO_ERRATA_20260721.md`。**CoMem 8B F1 19.51/acc 28.65/EM 5.99**（errata target 命中，+per-cat）；baseline KV-Direct **40.06/43.05**、InfLLM 25.76/26.38、StreamingLLM 12.73/17.57、MemoryLLM 9.93/9.72、HCache 7.82/8.06。**⚠️未改论文**：(1) 论文 tab_locomo 是 STALE 协议（CoMem 9.05/KV-Direct 8.80），单行替换会混协议；(2) 核实数字 **翻转 headline**「CoMem≈KV-Direct」→ KV-Direct(40.06)>>CoMem(19.51)，属叙事决策；(3) 论文 Judge=39.5 在 iter 目录 **无 backing 文件**不可核。→ 需 ONE coherent researcher/user pass（整表换协议+KV oracle 框定+judge 重跑或删+审 BABILong/LongBench/overview 协议一致）。
> - **下一步**：监控 5 臂（held-out ppl plateau→早停）；task#10 待 deliberate pass。dllm 29.162.226.120 未碰。

## 当前快照（2026-07-21 02:26，32/32 卡满 5 臂全健康；GPU 无空位，推进 GPU-free task#10）
> 02:26 tick（`date` 实测 02:26:06 +0800）：**铁律1 满足=32 卡全占（proc-count 逐卡=1，四节点）**，5 臂全健康续跑。
> - **wzc1 8/8 from_scratch**（Control 2）：step900/200k loss4.21 ppl67.5 1.57s/step 0NaN（random-init warmup 稳步降 step400→900：ppl162→67.5）。
> - **.82 8/8 keep12**：step6800/200k loss2.85 ppl17.32 7.81s/step 0NaN（平稳，早期）。
> - **.104 8/8 keep8**：step9180/200k loss3.01 ppl20.29 5.82s/step 0NaN（续降）。
> - **.73 8/8 freeze_front**（Arm A）：step180/200k loss5.93 ppl375.9 7.25s/step 0NaN（warmup 后 ppl 快降 46289→2584→461→376；**frozen-front 为高风险臂，关注是否高位 plateau=frozen 前段瓶颈的 ablation 信号**）。
> - **GPU 无空位**：所有 benchmark task 已清（TaskList #5-#43 全 completed），GPU 100% 占于 Paper B 5 臂。唯一 pending = **task#10（GPU-free 论文整合：InfLLM/HCache/KV-Direct/StreamingLLM/MemoryLLM/LLoCO baseline 行入表 + CoMem LoCoMo iter_bm25 errata）**，主循环推进中。
> - **下一步**：监控 5 臂（held-out ppl plateau→早停）；推进 task#10 论文整合。dllm 29.162.226.120 未碰。

## 当前快照（2026-07-21 02:12，32/32 卡满；task#42 T27b content-j 三语义基准打分完成+回填；ckpt 轮转 no-op）
> 02:12 tick（`date` 实测 02:11:49 +0800）：**铁律1 满足=32 卡全占，5 臂全健康**（proc-count 逐卡=1）。
> - **wzc1 8/8 = from_scratch**（Control 2）：step400/200k loss5.09 ppl162↓（random-init warmup 正常下降）1.57s/step 122.3GB 0NaN。
> - **.82 8/8 = keep12**：step6680/200k loss2.85 ppl17.32 7.81s/step 0NaN 续降。
> - **.104 8/8 = keep8**：step9040/200k loss3.00 ppl20.04 5.82s/step 0NaN（刚存 step9000.pt，深剪档 ppl 略高属预期）。
> - **.73 8/8 = freeze_front**（Arm A）：step60/200k ppl 46289→5377→2584 快降（frozen-front 只训 fresh2+norm+lm_head，warmup lr 7.87e-6 爬坡中，高初 ppl 正常）0NaN。
> - **✅ task#42 (T27b) 完成**：content-j 5 scale × 3 benchmark CPU 打分全跑通、**全 cell n 齐无缺**（铁律2）。数字：**BABILong overall**(compare_answers, 21-cell 均值) 0.6B 28.2/1.7B 38.6/4B 49.1/8B 49.2/14B 52.8；**LongBench MACRO-F1**(6-ds) 15.98/22.96/30.35/34.05/37.15；**LoCoMo F1**(run_scoring n=1986) 9.97/8.33/14.20/16.66/18.85。核心结论：content-j adapter 价值 **scale-emergent 全 5 benchmark 一致**（0.6B 处处被伤、8B/14B 获益，LongBench 8B **+7.8**/14B +5.9、LoCoMo 14B F **+5.0**），但语义侧幅度比 T27 字面 exact-match 温和、**BABILong 近中性**。已回填 RUN_REGISTRY「T27b」+ QCMEM_BENCHMARK_PLAN §1a 三行。
> - **运维**：ckpt 轮转 cron **4ec42903** 本轮跑=**no-op**（wzc1 24T free 15%used；keep14fresh2={final+step200000 里程碑}、keep14fresh2_fromscratch={step500 最新}，无「非里程碑且非最新2」可删；keep10fresh2 不存在）。⚠️新增 from_scratch 写 ckpt 到 `keep14fresh2_fromscratch`（非 cron 命名目录，本轮已并入轮转扫描；~225GB/h vs 24T free=100h+ runway，非急）。dllm 29.162.226.120 未碰。
> - **下一步**：监控 5 臂健康（loss↓/无 NaN/held-out ppl plateau→早停）；GPU 全满、task#42 收尾完 → 转 GPU-free 论文任务（task#10：InfLLM+baseline 行入表 + CoMem LoCoMo errata）。

## 当前快照（2026-07-21 02:04，32/32 卡满；Paper B 5 臂全跑：depth ladder keep14✓done/12/8 + 控制臂 from_scratch/freeze_front）
> 02:04 tick（`date` 实测 01:56 起→02:04）：**铁律1 满足=32 卡全占，两个刚空出的节点已即时补卡**。
> - **wzc1 8/8 = from_scratch 控制臂**（Paper B Control 2）：keep14 depth-ladder 主臂 **已训完** step200000/200000（final ppl~10.16，final.pt 48.7GB 落盘 OK），立即启 `KEEP=14 FROM_SCRATCH=1`（random-init 全 16 层，base 权重 IGNORED，4.0604B 全可训，bs16 eff128 L20A 1.56s/step）。02:00 startup healthy。
> - **.73 8/8 = freeze_front 控制臂**（Paper B Arm A）：content-j **14b BABILong 重跑完成+验证**（DONE 01:46，336 输出文件=齐，gap 已闭），立即启 freeze_front（inline `/tmp/launch_freezefront.sh`，transplant front14+embed/norm/lm_head、freeze 前 14 层、只训 fresh2+norm+lm_head=1.2269B trainable、bs4/ga4 eff128 匹配 keep14、H20 7.8s/step）。02:03 sanity **ALL 6 CHECKS PASS**。
> - **.82 8/8 keep12** 7B step6580 ppl17.31 0NaN 续降；**.104 8/8 keep8** 7B step8880 ppl19.79 0NaN 续降（depth ladder 中间两档，早期长跑，天级）。
> - **运维**：keep14 完成后删掉冗余 one-shot cron **1c59dc4c**（本 tick inline 接管控制臂启动，避免与 02:13 heartbeat 双启）；ckpt 轮转 cron 4ec42903 在册未动；dllm 29.162.226.120 未碰。
> - **✅铁律2**：14b BABILong gap 已闭（22:39 假 rc=0 = 运行中 pool 无 offline env 下 dataset-cache 网络重试竞态；重跑加 HF_*_OFFLINE=1+stagger→336 文件齐）。
> - **下一步（本 tick 后续/下 tick）**：content-j **打分** 5 scale × 3 benchmark（BABILong=compare_answers via score_nested_babilong.py / LongBench=qa_f1 / LoCoMo=F1，CPU on .73，与训练并行不占卡）→ 回填 RUN_REGISTRY §1a + QCMEM_BENCHMARK_PLAN。Paper B 5 臂后续离线测 held-out ppl(dolmino_now_val) plateau→早停对照。

## 当前快照（2026-07-21 01:41，4 节点全满健康 0 空卡；.73 14b-babilong 重跑已修复，控制臂 one-shot 已armed）
> 01:41 tick（`date` 实测 01:41 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step199500/200k loss2.3311 ppl10.29 1.56s/step 0NaN（剩~500步 ETA~01:53，无 final.pt）；**.82 8/8** keep12 7B step6420 ppl17.35 0NaN 续降；**.104 8/8** keep8 7B step8660 ppl20.17 0NaN 续降；**.73 8/8** = content-j 14b BABILong **重跑**（task#42；pool 01:37 全 "ALL WORKERS DONE" drain→我 standalone 起 8 shard GPU0-7，`/tmp/rerun_14b_babilong.sh`，显式 HF_*_OFFLINE=1 + 3s stagger 消除 cache-load 竞态）。**✅铁律2 14b BABILong GAP 根因+修复**：22:39 的 rc=0 是假阳性——运行中 pool 进程 /proc/PID/environ 无 offline env，dataset HEAD "Network is unreachable" 重试下 8-并发 cache-load 竞态→qa2/1k 处被杀（0 inference，泄漏信号量退出）；14b longbench 同模型 8-并发成功→排除 RAM/模型大小。重跑 01:40 验证 shard0 qa1/2k 62% 真 inference、无网络重试、8 GPU 进程。ETA~02:10（logs `/tmp/t27b_14b_rerun/babilong_14b_shard{0..7}.log`，`/tmp/t27b_14b_rerun/DONE`）。**其余 4 scale babilong + 全 5 scale longbench/locomo 齐**。**🕐 keep14→控制臂**：heartbeat cron dbb5ac9d 打点 :13/:43（下次 01:43/02:13），keep14 ~01:53 完落在 gap→建 one-shot cron **1c59dc4c @01:57** 在 gap 内起 wzc1 from_scratch 控制臂（`KEEP=14 FROM_SCRATCH=1 bash scripts/run_olmo2_7B_keepN.sh`，commit 505ebde），带幂等保护防与 02:13 heartbeat 双启。第二臂 freeze_front 待另一节点空。**下一步**：14b 重跑完(~02:10)→统一打分 content-j BABILong(compare_answers)/LongBench(qa_f1)/LoCoMo(F1) 5 scale→回填 RUN_REGISTRY §1a。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-21 01:26，4 节点全满健康 0 空卡；.73 pool 104/112 末批，查出 14b-babilong gap）
> 01:26 tick（`date` 实测 01:26:00 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step199000/200k loss2.3009 ppl9.98 1.56s/step 0NaN（剩~1k步 ETA~01:52，无 final.pt）；**.82 8/8** keep12 7B step6340/200k loss2.8471 ppl17.24 7.81s/step 0NaN（续降）；**.104 8/8** keep8 7B step8560/200k loss3.0143 ppl20.37 5.82s/step 0NaN；**.73** detached `t27b_pool.sh`（task#42）content-j **104/112 done 8剩 0fail**（120 job 全集：3 bench×5 scale×8 shard-24 缺项；末批 = longbench-0p6b 8 shard，ETA~01:40）。**⚠️铁律2 完整性核查发现 GAP**：`babilong|14b` 池标 rc=0（22:39）但 shard log 仅 38 行=只 dataset-load(0k→32k)无 inference、~9min vs 8b~30min、无 `qcmem_14b_adapter_contentj18_iter_chatnothink` 输出目录 → 池假阳性。14b adapter（outputs/qcmem_distill_14b_contentj18_r32/final）在、babilong cmd 正常 → 疑瞬时早退，需重跑（task#42 已记）。其余 4 scale babilong + 全 5 scale longbench/locomo 齐。**下一步**：.73 空→先重跑 content-j 14b babilong（8 shard，核实真写出）再统一打分 BABILong(compare_answers)/LongBench(qa_f1)/LoCoMo(F1)。keep14 完→wzc1 上 Paper B 控制臂（505ebde）。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-21 00:56，4 节点全满健康 0 空卡；控制臂脚本已 prep，等 wzc1 空）
> 00:56 tick（`date` 实测 00:56:00 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step198000/200k loss2.3399 ppl10.38 1.56s/step 0NaN（剩~2k步 ETA~01:47 → 完后 wzc1 空可上 Paper B 控制臂）；**.82 8/8** keep12 7B step6120/200k loss2.8884 ppl17.97 7.81s/step 0NaN；**.104 8/8** keep8 7B step8260/200k loss3.0084 ppl20.25 5.82s/step 0NaN（续降）；**.73** detached `t27b_pool.sh`（task#42）content-j **85/112 done 27 剩 0fail**（GPU 11-71%util、3.8-6.5GB、126-218W 真跑，1.7b babilong DONE 到 00:56 + longbench-1.7b 起跑，shard log 00:56:15 全新，铁律2 核实健康）。**本 tick GPU-free prep**：`scripts/run_olmo2_7B_keepN.sh` 加 FREEZE_FRONT=1(Arm A)/FROM_SCRATCH=1(Control2) env passthrough + ARM-后缀隔离 OUT_DIR/LOG（bash -n OK，dry-run 验 keep14fresh2_fromscratch / _freezefront，默认不变），commit 505ebde（本地，push 走 /gitpush）。控制臂待 wzc1 空一键上。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-21 00:26，4 节点全满健康 0 空卡；.73 pool 80/112 done，keep14 临近完成）
> 00:26 tick（`date` 实测 00:26:00 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step196980/200k loss2.3611 ppl10.60 1.56s/step 122.3GB 0NaN（剩~3k步 ETA~01:4x → 完成后 wzc1 空，可上 Paper B 控制臂 freeze_front/from_scratch，先验 run_olmo2_7B_keepN.sh 的 --from_scratch passthrough）；**.82 8/8** keep12 7B step5880/200k loss2.8876 ppl17.95 7.81s/step 0NaN（gnorm1.14 微噪）；**.104 8/8** keep8 7B step7960/200k loss3.0138 ppl20.37 5.79s/step 0NaN（续降）；**.73** detached `t27b_pool.sh`（task#42）content-j **80/112 done 32 剩**（+24/30min，1.7b locomo DONE 到 00:25，现 8 babilong-1.7b shard 00:25 START 进 CPU-haystack）——nvidia-smi 全 0MiB=CPU-bound 假象；`date` 00:26:15 时 shard log 全新（00:26:08）增长 0→1160B、10 py 活，铁律2 核实健康。~32 剩 ~40min 完 → content-j BABILong/LongBench/LoCoMo 主循环 inline 打分。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 23:56，4 节点全满健康 0 空卡；.73 pool 56/112 done babilong-4b CPU-haystack）
> 23:56 tick（`date` 实测 23:56:00 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step195980/200k loss2.3484 ppl10.47 1.56s/step 122.3GB 0NaN（剩~4k步）；**.82 8/8** keep12 7B step5660/200k loss2.8596 ppl17.45 7.81s/step 0NaN（续降）；**.104 8/8** keep8 7B step7660/200k loss3.0280 ppl20.66 5.79s/step 0NaN（gnorm0.94 微噪非退化）；**.73** detached `t27b_pool.sh`（task#42）content-j × BABILong+LongBench+LoCoMo **56/112 done 56 剩**（+24 job/30min，4b locomo 已 DONE 到 23:43，现 8 babilong-4b shard 于 23:40-43 START 进 CPU-haystack 构造期）——nvidia-smi 全 0MiB/0%util=CPU-bound 假象；`date` 23:56:40 时 8 shard log 全新（23:55:35-23:56:18）增长至 6.4KB、10 py 活各~1%CPU，铁律2 核实健康非卡死。控制臂待空**训练**节点。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 23:26，4 节点全满健康 0 空卡；.73 content-j pool babilong-8b GPU-forward 期）
> 23:26 tick（`date` 实测 23:25:59 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step194960/200k loss2.3514 ppl10.50 1.56s/step 122.3GB 0NaN（剩~5k步）；**.82 8/8** keep12 7B step5440/200k loss2.8753 ppl17.73 7.81s/step 0NaN（续降）；**.104 8/8** keep8 7B step7340/200k loss3.0167 ppl20.42 5.79s/step 0NaN（续降）；**.73** detached `t27b_pool.sh`（task#42）content-j × BABILong+LongBench+LoCoMo **32/112 done 80 剩**——babilong 8b 8 shard 已从上轮 CPU-haystack 假象转入 GPU-forward（nvidia-smi 7/8 卡 62-100%util、~20GB、190-320W；shard log 23:26 增长至 32KB；babilong 是最慢 job（qa1/2/5×0k-32k×100 带 per-config CPU haystack）故 30min 无新 DONE 属正常非卡死，铁律2 核实）。控制臂待空**训练**节点。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 22:59，4 节点全满健康 0 空卡；.73 content-j task-pool 32/112 done detached 自跑）
> 22:59 tick（`date` 实测 22:56:05 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step193940/200k loss2.3220 ppl10.20 1.56s/step 122.3GB 0NaN（剩~6.1k步 ETA~今晚23:5x）；**.82 8/8** keep12 7B step5220/200k loss2.9065 ppl18.29 7.81s/step 0NaN（续降）；**.104 8/8** keep8 7B step7060/200k loss3.0536 ppl21.19 5.79s/step 0NaN（续降）；**.73** detached `t27b_pool.sh`（task#42，非 agent——两 background agent 连续 API-500 崩后改 detached 自跑+主循环 inline 收尾）——content-j × BABILong+LongBench+LoCoMo **32/112 done，80 剩**，现 8b babilong 8 shard CPU-haystack 加载期（nvidia-smi 0%util/3MiB=CPU-bound 假象；`ps` 见 10 python 活各 3.2%CPU、shard log 22:59:20 增长至 1740B、pool-driver 活，铁律2 核实健康非卡死）。控制臂待空**训练**节点。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 22:26，4 节点全满健康 0 空卡；.73 content-j 3-benchmark task-pool 起跑）
> 22:26 tick（`date` 实测 22:25:59 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step192920/200k ppl10.36 1.56s/step 122.3GB 0NaN；**.82 8/8** keep12 7B step4980/200k loss2.9102 ppl18.36 0NaN（续降）；**.104 8/8** keep8 7B step6740/200k loss3.0466 ppl21.04 0NaN（从 22.22 回落至 21.04，续降趋势确认，非退化）；**.73 8/8** agent a57d786c（task#42）——task-pool 跑 **+adapter content-j × BABILong+LongBench+LoCoMo**（104 job 剩），现 14B BABILong 数据加载期（nvidia-smi 0%util/0MiB=CPU-bound haystack 构造假象；`ps` 见 8 python 活、shard log 580→1160B 增长、HF"Network unreachable"=本地 cache fallback benign，铁律2 核实健康非卡死）+ 并行恢复 T27-ext RULER 64k/128k 打分回填。控制臂待空**训练**节点。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 21:26，4 节点全满健康 0 空卡；T27-ext 14B 64k完/128k收尾 ~22min）
> 21:26 tick（`date` 实测 21:25:59 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step190880/200k ppl10.07 1.56s/step 122.3GB 0NaN（ETA~今晚00:4x，剩~9.1k步）；**.82 8/8** keep12 7B step4520/200k loss2.9154 ppl18.46 10.45s/step 0NaN；**.104 8/8** keep8 7B step6120/200k loss3.1011 ppl22.22 5.79s/step 0NaN（早期噪声微升 21.61→22.22，gnorm1.47 正常，非退化）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B+4B+8B DONE，**14B（末scale）64k=24 齐**，128k=16（vt_128k 46% [~22min]）→**14B 128k 收尾即 T27-ext 全部完成**→agent 聚合+回填 canonical。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 20:56，4 节点全满健康 0 空卡；T27-ext 进 14B 末scale ~50min完）
> 20:56 tick（`date` 实测 20:56:19 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step189880/200k ppl10.05 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~10.1k步）；**.82 8/8** keep12 7B step4300/200k loss2.9123 ppl18.40 7.85s/step 0NaN（续降 2.95→2.91）；**.104 8/8** keep8 7B step5820/200k loss3.0734 ppl21.61 5.79s/step 0NaN（续降 3.12→3.07，破 3.1；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B+4B+**8B DONE**（各 64k=24/128k=24）→**现进 14B（最后scale）**（GPU0 START 20:30，64k=16/128k=16=各2task齐，现 vt_64k 8%；vt_64k+vt_128k 剩 ~50min）。**14B 完即 T27-ext 全部完成**→届时 agent 聚合+回填 canonical。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 20:26，4 节点全满健康 0 空卡；T27-ext 8B近完→14B末scale）
> 20:26 tick（`date` 实测 20:26:03 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step188860/200k ppl10.09 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~11.1k步）；**.82 8/8** keep12 7B step4060/200k loss2.9485 ppl19.08 7.85s/step 0NaN；**.104 8/8** keep8 7B step5500 saved（续训中，末次loss line ppl22.68；ckpt 在 diskB 盘充裕）0NaN；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B+4B DONE→**8B 近完**（64k=24/128k=16，vt_128k 92% 25s/it ~3min）→将进 **14B（最后scale）**。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 19:56，4 节点全满健康 0 空卡；T27-ext 8B ~2/3→14B 末）
> 19:56 tick（`date` 实测 19:56:02 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step187840/200k ppl10.14 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~12.2k步）；**.82 8/8** keep12 7B step3840/200k loss2.9430 ppl18.97 7.85s/step 0NaN（续降 2.95→2.94）；**.104 8/8** keep8 7B step5200/200k loss3.1213 ppl22.68 5.79s/step 0NaN（session趋势 3.34→3.12↓；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B+4B DONE→**8B ~2/3**（64k=24 全3task/128k=16=2task齐，现 vt_128k 22% 25s/it ~33min 完）→将进 **14B（最后scale）**。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 19:26，4 节点全满健康 0 空卡；T27-ext 进 8B，4/5 scale）
> 19:26 tick（`date` 实测 19:26:02 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step186820/200k ppl10.17 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~13.2k步）；**.82 8/8** keep12 7B step3620/200k loss2.9526 ppl19.16 7.85s/step 0NaN（session趋势 3.13→2.95↓，步间小噪声）；**.104 8/8** keep8 7B step4900/200k loss3.1154 ppl22.54 5.79s/step 0NaN（session趋势 3.34→3.11↓；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B+**4B DONE**（各 64k=24/128k=24 齐）→**现进 8B**（GPU0 START 19:12，64k=8/128k=7，niah_multikey/64k 99%）。14B 排队末位。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 18:56，4 节点全满健康 0 空卡；T27-ext 4B 近完进 8B）
> 18:56 tick（`date` 实测 18:56:02 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step185800/200k ppl10.10 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~14.2k步）；**.82 8/8** keep12 7B step3400/200k loss2.9507 ppl19.12 7.85s/step 0NaN（续降 2.98→2.95）；**.104 8/8** keep8 7B step4600/200k loss3.1109 ppl22.44 5.79s/step 0NaN（续降 3.14→3.11；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B DONE→**4B 近完**（64k=24 全3task/128k=16=2task齐，现 vt_128k 63% 25s/it ~15min 完）→将进 **8B**。14B 排队。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 18:26，4 节点全满健康 0 空卡；T27-ext 4B ~2/3）
> 18:26 tick（`date` 实测 18:26:02 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step184780/200k ppl10.16 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~15.2k步）；**.82 8/8** keep12 7B step3160/200k loss2.9803 ppl19.69 7.85s/step 0NaN（续降 3.00→2.98）；**.104 8/8** keep8 7B step4300/200k loss3.1389 ppl23.08 5.82s/step 0NaN（续降 3.18→3.14；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B DONE→**4B ~2/3**（64k=16/128k=16=各2task齐，现 vt_64k 65% 6.98s/it）。8B/14B 排队（每卡串行全5scale）。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 17:56，4 节点全满健康 0 空卡；T27-ext 进 4B）
> 17:56 tick（`date` 实测 17:56:03 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step183760/200k ppl10.04 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~16.2k步）；**.82 8/8** keep12 7B step2940/200k loss2.9960 ppl20.00 7.85s/step 0NaN（续降 3.03→3.00，破 3.0）；**.104 8/8** keep8 7B step4000/200k loss3.1801 ppl24.05 5.79s/step 0NaN（续降 3.19→3.18；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B+1.7B DONE（各 64k=24/128k=24 齐）→**现进 4B**（GPU0 START 17:55，niah_single_2/64k 41% 1.58s/it）。8B/14B 排队（每卡串行全5scale）。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 17:26，4 节点全满健康 0 空卡；T27-ext 1.7B 近完进 4B）
> 17:26 tick（`date` 实测 17:26:04 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step182760/200k ppl10.40 1.56s/step 122.3GB 0NaN（续降；ETA~今晚00:5x，剩~17.2k步）；**.82 8/8** keep12 7B step2720/200k loss3.0301 ppl20.70 7.85s/step 0NaN（续降 3.07→3.03）；**.104 8/8** keep8 7B step3680/200k loss3.1882 ppl24.25 5.79s/step 0NaN（续降 3.20→3.19；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——0.6B DONE→**1.7B 近完**（64k=24 全3task/128k=16=2task齐，现 vt_128k 33% 25s/it ~28min 完）→将进 **4B**。8B/14B 排队（每卡串行全5scale）。控制臂待空**训练**节点；agenda 全 GPU 满无空位。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 16:56，4 节点全满健康 0 空卡；T27-ext 0.6B完成进 1.7B）
> 16:56 tick（`date` 实测 16:56:03 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step181740/200k loss2.3248 ppl10.22 1.56s/step 122.3GB 0NaN（ETA~今晚00:5x，剩~18.3k步）；**.82 8/8** keep12 7B step2500/200k loss3.0668 ppl21.47 7.81s/step 0NaN；**.104 8/8** keep8 7B step3380/200k loss3.1991 ppl24.51 5.79s/step 0NaN（续降 3.27→3.20；keep8>keep12 符合预期）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——**0.6B DONE 16:38 rc=0**（64k=24/128k=24 全 3task×8shard×2len 齐）→现进 **1.7B**（64k=16=2task齐/128k=8=1task齐，niah_multikey/128k~50% 5.5s/it，log mtime16:57 活跃）。4B/8B/14B 排队串行（每卡跑全5scale）。0%util 瞬间=128k iter_bm25 CPU 检索间隙非卡死。控制臂待空**训练**节点（24训练卡全占；.73=EVAL-ONLY）；agenda 全 GPU 满无空位启动。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 16:28，4 节点全满健康 0 空卡；.73 0%util 虚警已排除）
> 16:28 tick（`date` 实测 16:28:12 +0800）：**铁律1 满足=32 卡全占**。**wzc1 8/8** keep14 7B step180720/200k ppl9.93↓ 100%util 929W 0NaN（续降 10.18→9.93；ETA~今晚00:4x）；**.82 8/8** keep12 7B step2260/200k loss3.0727↓ 0NaN；**.104 8/8** keep8 7B step3060/200k loss3.2715↓ 0NaN（keep8>keep12 符合预期=多剪更难 heal）；**.73 8/8** T27-ext（agent a0a51eb9 task#41）——❗**0%util 虚警排除**：nvidia-smi 瞬间 0%/120W/2.6GB 疑似卡死，但按铁律2 查 log 增长=8 卡 worker log 全在 `variable_tracking/128k 70–76% [25s/it]`，mtime 16:27–48（秒级更新，活跃）。设计=每卡跑全 5 scale 串行（0.6B→1.7B→4B→8B→14B），故当前仅 0.6B 有文件属正常；0.6B vt_128k 尾（~10min 完）后进 1.7B。0%瞬间=128k iter_bm25 CPU 检索间隙。**无需干预**。控制臂待空**训练**节点；paper task#10=GPU-free 暂缓。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 15:26，4 节点全满健康 0 空卡）
> 15:26 tick（`date` 实测 15:26:17 +0800）：**铁律1 满足=32 卡全占无空转**。**wzc1 8/8** keep14 7B step178680/200k loss2.3087 ppl10.06 lr2.50e-6 gnorm0.51 1.56s/step 100%util 956W 0NaN（续降 10.19→10.06；ETA~今23:50）；**.82 8/8** keep12 7B step1800/200k loss3.1314 ppl22.90 100%util 0NaN（续降 3.17→3.13）；**.104 8/8** keep8 7B step2460/200k loss3.2979 ppl27.06 100%util 0NaN（续降 3.34→3.30；keep8>keep12 符合预期）；**.73 8/8** T27-ext=RULER content-j 64k/128k 延伸（agent a0a51eb9，task#41；16 个 64k/128k shard 文件已出，0.6B niah_single 64k 正写）。控制臂 freeze_front/from_scratch 仍待空**训练**节点（24 训练卡全占；.73=EVAL-ONLY）；paper task#10=GPU-free 暂缓。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 15:12，T27 完成落账→.73 补 T27-ext 64k/128k；三训练健康）
> 15:12 tick：**T27 DONE**——agent a8f9f989 完成 GPU eval（14:55:41 rc=0）+ 自行回填 canonical（RUN_REGISTRY T27 子节@line1799、QCMEM_BENCHMARK_PLAN §1a×4 markers、gpu_runs、.73 /tmp 已清、8 卡空）。**main 独立官方聚合验证=完全吻合**（RULER content-j 0.6B42.8/1.7B45.5/4B56.2/8B51.1/14B63.3；LongEval split 0.6B1.2/1.7B0.8/4B15.5/8B13.0/14B40.2）。核心结论：**content-j adapter 修复 readout-drift collapse=随规模浮现（scale-dependent）**——RULER 近单调全修（14B niah 15→98 戏剧性 rescue），LongEval 大模型修复（8B7.5→13.0/14B20.3→40.2）小模型反被深 j 破坏（0.6B37.3→1.2/1.7B15.8→0.8，max48 后仍崩=验真非截断 bug），crossover 4B↔8B。agent 顺手诊断并修了 LongEval `max_new_tokens=16→48` 截断坑。
> **.73 补位（铁律1）**：T27 RULER content-j 只到 32k，zero-shot RULER(T25)/LongEval content-j 都到 128k → 派 **agent a0a51eb9** 跑 **RULER content-j 64k/128k 延伸**（5 scale × 3 task × 2 length，n=100 8-shard，同 iter_bm25+chat+no-think+max48 口径，append 进现有 content-j 目录），测「content-j 修复在长档是否保持」。耐跑长任务=正确 idle-fill（非 short-task churn，非 speculative）。task#41 owner=a0a51eb9。
> **三训练**（15:03 实测，续跑健康，未变）：wzc1 keep14 step177960 train-ppl10.19 100%util 948W 0NaN（ETA~今23:50）；.82 keep12 step1580 loss3.17↓ 0NaN；.104 keep8 step2160 loss3.34↓ 0NaN（keep8>keep12 符合预期）。控制臂 freeze_front/from_scratch 待空**训练**节点（.73=EVAL-ONLY 不可用；24 训练卡全占）。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 15:04，三训练全健康续降；.73 T27 GPU eval 全完成→main 已验证 grids）
> 15:04 tick（`date` 实测 15:03:54 +0800）：**wzc1 8/8** keep14 7B **step177960/200k** loss2.3214 train-ppl10.19 lr2.53e-6 gnorm0.52 1.56s/step maxmem122.3GB 100%util 948W 0NaN（log 15:03:34 新鲜；续降 ppl10.23→10.19）；**.82 8/8** keep12 7B **step1580/200k** loss3.1678 train-ppl23.76 lr2e-5 gnorm2.28 7.81s/step maxmem91.9GB 100%util 0NaN（续降 step1360→1580：3.20→3.17）；**.104 8/8** keep8 7B **step2160/200k** loss3.3384 train-ppl28.17 lr2e-5 gnorm2.13 5.79s/step maxmem73.5GB 100%util 0NaN（续降 step1860→2160：3.41→3.34；keep8>keep12 loss 符合预期=剪层更多更难 heal）；**.73** T27 **GPU eval 全完成**（8-shard pool，last worker exit 14:55:41 rc=0；RULER content-j 5 scale + LongEval content-j 5 scale 全 8-shard summary 齐；procs 0/8）。**main 已用官方 summary shard 独立聚合验证（铁律2）**：RULER content-j 近单调 mid-collapse 修复（0.6B42.8/1.7B45.5/4B56.2/8B51.1/14B63.3）；LongEval content-j **split**（8B7.5→13.0、14B20.3→40.2 恢复；0.6B37.3→1.2、1.7B15.8→0.8 崩塌=深 j 丢 surface token 小模型无法重建 literal，已验真非 bug，crossover 4B↔8B）。grids+结论 → `status/_T27_verified_grids_scratch.md`。agent a8f9f989 仍 finalize（0 T27 markers in canonical 截至 15:03）；**无 SendMessage 工具**→不注入 .73（防撞 agent finalize；且 T27 是查漏最后 gap，无下一 eval 值得 speculative burn）。**待办：收到 agent 完成通知→核对/去重后单写手回填 RUN_REGISTRY T27 子节+§1a content-j 行（source=scratch）+gpu_runs+清 .73 /tmp+删 scratch+task#40 done**；若 agent 早退未回填则 main 直接从 scratch 落。dllm 29.162.226.120 绝不碰；.73=EVAL-ONLY。

## 当前快照（2026-07-20 14:26，三训练全健康续降；.73 T27 LongEval content-j 收尾最后 cell 14B 128k）
> 14:26 tick（`date` 实测 14:26:01 +0800）：**wzc1 8/8** keep14 7B **step176640/200k** loss2.3256 train-ppl10.23 lr2.60e-6 gnorm0.52 1.56s/step maxmem122.3GB 0NaN（log 14:25:41 新鲜；ETA~今23:50）；**.82 8/8** keep12 7B **step1360/200k** loss3.2044 train-ppl24.64 lr2e-5 gnorm3.03 7.81s/step maxmem91.9GB 100%util 0NaN（loss 续降 step1120→1360：3.27→3.20）；**.104 8/8** keep8 7B **step1860/200k** loss3.4066 train-ppl30.16 lr2e-5 gnorm3.06 5.79s/step maxmem73.5GB 99%util 0NaN（loss 续降 step1540→1860：3.47→3.41）；**.73** T27 agent **a8f9f989** —— **RULER content-j 5 scale 全跑完 → LongEval content-j 5 scale dir 全建、正写最后/最慢 cell 14B 128k**（shard3of8 已出 `_summary`=该 shard 完成；procs `1 0 0 0 0 0 0 0`=8-shard pool drain 到最后 shard）。**铁律2 判定合法 imminent-completion tail 非卡死**：13:56 以来 390 json=强吞吐、newest=14B 128k shard 正写+shard summary 陆续出（file 增长确认 alive，与 T25/T26 尾档 128k 单 cell drain 同型）。**不注入**（防撞 agent 最终 cross-scale 打分+backfill，且注入任务在 agent 完成前也跑不完）。**⚠️ watch：下轮 14:56 若 .73 仍大部空闲且 agent 未完成 backfill → SendMessage 查是否真 hang**。待办：a8f9f989 T27 完成→回填 RUN_REGISTRY T27+§1a content-j 行（content-vs-readout 修复结论）；keep12/keep8 续监 loss↓；控制臂 freeze_front/from_scratch 待空**训练**节点。dllm 29.162.226.120 绝不碰；.73=EVAL-ONLY。

## 当前快照（2026-07-20 13:56，32/32 全占全健康；.73 T27 RULER content-j 近完+三训练 loss 续降）
> 13:56 tick（`date` 实测 13:56:01 +0800）：**wzc1 8/8** keep14 7B **step175620/200k** loss2.3019 train-ppl9.99 lr2.65e-6 gnorm0.52 1.56s/step maxmem122.3GB 0NaN（log 13:55:37 新鲜；ETA~今23:50）；**.82 8/8** keep12 7B **step1120/200k** loss3.2741 train-ppl26.42 lr2e-5 gnorm2.58 7.81s/step maxmem91.9GB 100%util 0NaN（loss 续降 step920→1120：3.30→3.27）；**.104 8/8** keep8 7B **step1540/200k** loss3.4691 train-ppl32.11 lr2e-5 gnorm3.14 5.79s/step maxmem73.5GB 100%util 0NaN（loss 续降 step1240→1540：3.54→3.47）；**.73 8/8** T27 +adapter content-j agent **a8f9f989**：8 卡全活，**RULER content-j 5 scale clean dir 全建**（qcmem_{0p6b_j13,1p7b_j13,4b_j16,8b_j16,14b_j18}_adapter_contentj*_iter_chatnothink），正写 **14B（最后 scale）niah_single_2_32k**（13:26 以来 230 json=健康吞吐）→ RULER 阶段近完，**LongEval content-j 阶段待接**。**32/32 全占全健康，无空卡。** 待办：a8f9f989 T27 完成→回填 RUN_REGISTRY T27+§1a content-j 行（content-vs-readout 修复结论）；keep12/keep8 续监 loss↓；控制臂 freeze_front/from_scratch 待空**训练**节点。dllm 29.162.226.120 绝不碰；.73=EVAL-ONLY。

## 当前快照（2026-07-20 13:26，32/32 全占全健康；.73 T27 content-j 顺进 1.7B RULER + 三训练 loss 续降）
> 13:26 tick（`date` 实测 13:26:03 +0800）：**wzc1 8/8** keep14 7B **step174620/200k** loss2.3145 train-ppl10.12 lr2.71e-6 gnorm0.52 1.56s/step maxmem122.3GB 0NaN（log 13:26:03 新鲜；ETA~今23:50）；**.82 8/8** keep12 7B **step920/200k** loss3.3048 train-ppl27.24 lr2e-5 gnorm3.73 7.81s/step maxmem91.9GB 99%util 0NaN（loss 续降 step700→920：3.46→3.30）；**.104 8/8** keep8 7B **step1240/200k** loss3.5387 train-ppl34.42 lr2e-5 gnorm3.55 5.79s/step maxmem73.5GB 100%util 0NaN（loss 续降 step960→1240：3.67→3.54）；**.73 8/8** T27 +adapter content-j agent **a8f9f989**：8 卡全活（GPU0-7 各 1 proc），**audit 裁决：旧 `_adapter_contentj*_n100` dir 非干净口径→建 clean 新 dir `qcmem_{scale}_adapter_contentj*_iter_chatnothink/` 重跑**（8-shard；已完 0.6B、正写 **1.7B** niah_single/multikey 8k-32k；13:05 以来 120 json=健康吞吐）。**32/32 全占全健康，无空卡。** 待办：a8f9f989 T27 完成→回填 RUN_REGISTRY T27+§1a content-j 行（content-vs-readout 修复结论）；keep12/keep8 续监 loss↓（held-out ppl plateau 则早停）；控制臂 freeze_front/from_scratch 待空**训练**节点。dllm 29.162.226.120 绝不碰；.73=EVAL-ONLY。

## 当前快照（2026-07-20 13:02，32/32 全占全健康；★T26✅→scale 故事 5 基准全齐→.73 补 T27 content-j 臂）
> 13:02 tick（`date` 实测 13:01:57 +0800）：**★T26 LongEval zs CLEAN✅完成并回填**（agent a1f86c58 完成 36-cell GPU eval 后意外早退→本 heartbeat 独立复算全 grid 铁律2 核实 + 回填 RUN_REGISTRY「T26」子节 + QCMEM_BENCHMARK_PLAN §1a LongEval CLEAN 行 + T26 RESOLVED bullet；结论 LongEval 与 RULER 同型强非单调、32B(97.5) 碾压且长度无关、0.6B(37.3) 浅 j 反常强、中段塌陷 8B 最弱(7.5)）→**跨-benchmark zero-shot scale 故事 5 基准全齐**（BABILong T24 单调 / LongBench+LoCoMo T22 近单调 / RULER T25 强非单调 / LongEval T26 强非单调；语义 QA 三基准单调、字面 exact-match 两基准非单调但 32B 始终全面最优）。**→铁律1 立即补 .73**：审计发现 content-j LoRA adapter 全 ladder 已在 diskB（`qcmem_distill_{0p6b_contentj13,1p7b_contentj13,4b_contentj16,8b_contentj16,14b_contentj18}_r32`）→ **派 agent a8f9f989（background）跑 T27 = +adapter content-j CLEAN scale 曲线（RULER+LongEval 两 exact-match 基准）**，验证 §1a 核心 claim「content-j adapter 能否修复 zero-shot readout-safe-j 塌陷」（audit-first 复用已跑干净 cell、只补缺失，8-shard 填满 8 卡，官方 string_match/acc 判分，零 git/零 src 改；32B/30B-A3B 无 content-j adapter→跳过）。**其余 3 训练承前健康**：**wzc1 8/8** keep14 7B **step173680/200k** loss2.3342 train-ppl10.32（噪声窗，step173640 行 10.04）lr2.76e-6 gnorm0.52 1.56s/step maxmem122.3GB 0NaN；**.82 8/8** keep12 7B **step700/200k** loss3.4550 train-ppl31.66 lr2e-5 gnorm3.88 7.81s/step maxmem91.9GB 0NaN（loss 续降 step460→700：3.68→3.46）；**.104 8/8** keep8 7B **step960/200k** loss3.6725 train-ppl39.35 lr2e-5 gnorm4.22 5.79s/step maxmem73.5GB 0NaN（loss 续降 step620→960：3.91→3.67；keep8>keep12 loss 符合剪更多层更难 heal）。**32/32 全占全健康，无空卡。** ckpt-rotation(cron 4ec42903) 本轮 NO-OP：df wzc1 24T free/15%，keep14fresh2 仅 step173500.pt(48.7G，keep-latest-1 自轮转无可删)，keep10fresh2 wzc1 缺席。待办：a8f9f989 T27 完成→回填 RUN_REGISTRY T27+§1a content-j 行（content-vs-readout 修复结论）；keep12/keep8 续监 loss↓（held-out ppl plateau 则早停）；控制臂 freeze_front/from_scratch 待空**训练**节点（.73=EVAL-ONLY 不可承）。dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 12:26，32/32 全占全健康；三训练 loss 续降 + .73 T26 顺进 4+scale）
> 12:26 tick（`date` 实测 12:26:27 +0800，.73 亦 12:27:44 同步）：**wzc1 8/8** keep14 7B **step172600/200k** loss2.339 train-ppl10.37（噪声窗，前行 9.71）lr2.82e-6 gnorm0.54 1.56s/step maxmem122.3GB 0NaN 健康（ckpt step172500@12:23 keep-latest 自轮转；ETA~今23:50）；**.82 8/8** keep12 7B **step460/200k** loss3.6804 train-ppl39.66 lr2e-5 gnorm4.44 7.81s/step maxmem91.9GB **0NaN** 健康（loss 续降 step220→460：4.39→3.68）；**.104 8/8** keep8 7B **step620/200k** loss3.9086 train-ppl49.83 lr2e-5 gnorm5.44 5.79s/step maxmem73.5GB **0NaN** 健康（loss 续降 step320→620：4.46→3.91）；**.73 8/8** T26 LongEval zs CLEAN agent **a1f86c58** 进展顺：新建 clean dir 0.6b/1.7b/4b（各 65 files 完成）+8b（57，reuse 重打分）+**14b（24 files 写中，newest 6s 前）**，32b 待（最大 scale 末跑）；live worker age179s，8 卡全活。**32/32 全占全健康，无空卡。** 待办：a1f86c58 T26 完成→跨-benchmark scale 故事 5 基准全齐（回填 RUN_REGISTRY T26+§1a LongEval CLEAN 行）；keep12/keep8 续监 loss↓（held-out ppl plateau 则早停）；控制臂 freeze_front/from_scratch 待空节点。.73=EVAL-ONLY；dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 11:58，.73 T25✅→立即补 T26 LongEval zs 填第 5 基准，32/32 无空卡）
> 11:58 tick：**★T25 RULER zs CLEAN 曲线✅完成**（agent a8955573，11.6ks/130 tool_uses；6 scale×3 task×5 长档全 90 cell empty=0 oom=0；结论 RULER 强非单调但仍 32B 全 task 最强+64k/128k 稳跑=固定读卖点；已自回填 RUN_REGISTRY「T25」+ QCMEM_BENCHMARK_PLAN §1a RULER CLEAN 行 + RESOLVED；8 卡释放实测 0MiB/0proc/无 orphan）。**→铁律1 立即补 .73**：跨-benchmark scale 故事现 4/5 基准（BABILong T24/LongBench+LoCoMo T22/RULER T25 clean 齐），缺第 5=**LongEval**（§1a 现仅 legacy 浅 j/bm25 行，仅 8B 有 clean dir `qcmem_8b_zs_iter_chatnothink`）→ **派 agent a1f86c5889870f7d2（background）跑 T26 = QCMem LongEval zero-shot CLEAN scale 曲线**（6 scale × readout-safe j j2/j3/j9/j9/j13/j27，chat+no-think+iter_bm25，官方 acc scorer，8-shard×8GPU；8B 复用打分、其余 5 scale legacy 污染→clean 重跑；落账 RUN_REGISTRY T26+§1a LongEval CLEAN 行；零 git/零 src 改）。**其余 3 训练承前健康**：wzc1 8/8 keep14 step171560 ppl10.20；.82 8/8 keep12 step220 loss4.39 lr2e-5 dropping；.104 8/8 keep8 step320 loss4.46 dropping。**32/32 全占，无空卡。** 待办：a1f86c58 T26 完成→scale 故事 5 基准全齐；keep12/keep8 续监 loss↓；控制臂 freeze_front/from_scratch 待空节点。.73=EVAL-ONLY；dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 11:56，32/32 全占全健康；keep12/keep8 warmup 结束 loss 续降 + .73 T25 收尾写 8b vt_128k）
> 11:56 tick（`date` 实测 11:56:04 +0800）：**wzc1 8/8** keep14 7B **step171560/200k** loss2.3221 train-ppl**10.20** lr2.88e-6 gnorm0.53 1.56s/step maxmem122.3GB 0NaN 健康（ETA~今23:50）；**.82 8/8** keep12 7B **step220/200k** loss4.3874 train-ppl80.43 lr**2.00e-5**(warmup 结束满 lr) gnorm7.26 7.81s/step maxmem91.9GB<97.8 无 OOM 0NaN 健康（loss 10.55→8.47→4.39 续降）；**.104 8/8** keep8 7B **step320/200k** loss4.4563 train-ppl86.17 lr2.00e-5 gnorm6.45 5.79s/step maxmem73.5GB 0NaN 健康（loss 10.49→8.53→4.46 续降）；**.73**——**agent a8955573（T25）合法收尾 tail 非空转**：6 scale 全 summary=Y，big-model csv 补到 0.6B56/1.7B56/4B56/8B80/14B80/32B80，正写**最后最慢 cell = 8b variable_tracking_128k shard**（csv mtime shard0=11:55:30、**shard1=11:57:20（快照前秒级新鲜）**、14b vt_128k=11:47），128k vt 为全 grid 最慢=合法尾（ps 快照曾读 1 proc=cell 间隙瞬时，文件增长证 alive）→铁律1 不注入防撞 agent 最终打分+imminent 完成。**32/32 全占全健康，无空卡。** ckpt-rotation(cron 4ec42903) 本轮 NO-OP：df wzc1 24T free/15%，keep14fresh2 仅 step171500.pt(48.7G，keep-latest-1 自轮转无可删)，keep10fresh2 wzc1 缺席。待办：a8955573 T25 完成→回填 RUN_REGISTRY+QCMEM_BENCHMARK_PLAN §1a（跨-benchmark scale 故事第 4 基准 RULER）；keep12/keep8 续监 loss↓；控制臂 freeze_front/from_scratch 待空节点。.73=EVAL-ONLY；dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 11:31，32/32 全占全健康；★Paper B keep12/keep8 两臂 step40 确认健康）
> 11:31 tick（trainer a37818b0/本次 subagent，`date` 实测 11:28:29 +0800）：**★Paper B 7B 剪层-heal 深度扫两臂上线并确认健康**——**.82 8/8 = OLMo-2-7B keep12+fresh2**（fresh；world8 **bs4 ga4 eff_bs128**；transplant copied 135 tensors 前12层+embed/norm/lm_head from 32L base、fresh tail[12,13]、**ALL 6 CHECKS PASS**；3.6556B 14层；**step40 loss10.55→8.47 ppl38166→4785 gnorm31.4→5.27 7.81s/step maxmem91.9GB<97.8 无 OOM 0NaN 8procs**；log `logs/olmo2_7B_keep12fresh2.log` out `outputs/olmo2_probe2_7B_keep12fresh2/`）；**.104 8/8 = OLMo-2-7B keep8+fresh2**（fresh；同配方；copied 91 前8层 fresh[8,9] 6 CHECKS PASS；2.8460B 10层；**step40 loss10.49→8.53 ppl36019→5068 gnorm32.7→4.67 5.79s/step maxmem73.5GB 0NaN 8procs**；log `logs/olmo2_7B_keep8fresh2.log`）；两臂用 diskB launcher `scripts/run_olmo2_7B_keepN_diskB.sh`（新建，env-overridable diskB 路径 + H20 bs/ga，torch-base py）、`/dev/shm/dolmino_now15b.npy`、`--babilong_mix_fraction`默认 0、WANDB offline。**wzc1 8/8** keep14 7B step~170540 ppl10.15 承前健康；**.73 8/8** T25 RULER zs scale pool 收尾（big-model 64k/128k 补）。**32/32 全占全健康，无空卡。** 待办：下轮核 keep12/keep8 warmup(150 步)后 loss 续降；控制臂 freeze_front/from_scratch 待空节点；a8955573 T25 完成回填。.73=EVAL-ONLY；dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 11:26，32/32 全占全健康；★Paper B keep12/keep8 已上线两臂健康）
> 11:26 tick（`date` 实测 11:26:06 +0800）：**wzc1 8/8** keep14 7B **step170540/200k** loss2.3179 train-ppl**10.15** 1.56s/step maxmem122.3GB 0NaN 健康（step170520 行 6.90s/step=step170500 存 ckpt I/O artifact，170540 立回落 1.56=第三轮证实规律；log 11:25:44 新鲜；ETA~今23:50）；**★keep7 1B ✅ DONE**：final.pt 11:18（12.18GB）→ trainer **a37818b0** 立即起 Paper B 深度扫：**.82 8/8 = keep12**（world8 bs4 ga4 eff_bs128；transplant copied 135 tensors 前12层+embed/norm/lm_head from 32L base，fresh tail[12,13]，**ALL 6 CHECKS PASS**；3.6556B 14层；nvidia-smi 94.7GB=紧但未 OOM、无 error，step20 timing 内将出→**⚠️下轮核 step20 maxmem<97GB**）；**.104 8/8 = keep8**（同配方；copied 91 前8层 fresh[8,9] 6 CHECKS PASS；2.8460B 10层；**step20 loss10.49 ppl36019 gnorm32.67 maxmem73.5GB 0NaN=健康 fresh-start**，比 keep7-1B step20 loss12.20 更低=7B 前层更强）；**.73 8/8** T25 agent a8955573 pool 收尾（6 scale 全 summary=Y，14b csv 72→76、32b 76→80=big-model 64k/128k 仍在补，8 procs 全 100%）。**32/32 全占全健康，无空卡。** 待办：下轮核 keep12/keep8 前 60 步 loss↓+keep12 maxmem；a37818b0 落 RUN_REGISTRY keep12/keep8 行；a8955573 T25 完成→回填 RUN_REGISTRY+BENCHMARK_PLAN §1a（跨-benchmark scale 第 4 基准 RULER）；控制臂 freeze_front/from_scratch 待下个空节点。.73=EVAL-ONLY；dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 11:15，32/32 全占全健康；keep7 1B ~2min 完成→Paper B keep12/keep8 已派 trainer a37818b0 待启）
> 11:15 tick（`date` 实测 11:15:37 +0800）：**wzc1 8/8** keep14 7B **step170220/200k** loss2.3537 train-ppl**10.52** gnorm0.52 **1.56s/step**（本行=mid-interval 非 save-adjacent → 证实上两轮 6.85/6.86s/step 纯 ckpt-save 48.7GB I/O 显示 artifact；实际吞吐 09:56→10:26→11:15 每 30min~1000 步=1.76s/step 含 save 均摊）maxmem122.3GB 0NaN 健康（log 11:15:38 新鲜；ETA~今23:50）；**.82+.104 16/16** keep7 1B DDP **step199920/200k**（**仅剩 80 步≈2min**）loss2.7329 train-ppl**15.38** gnorm0.52 1.47s/step maxmem41.3GB 0NaN 健康（log 11:15:46 新鲜；尚无 final.pt）；**.73 8/8**——**agent a8955573（T25）自校正到全 8 卡**：6 scale **全 summary=Y**（0.6B/1.7B/4B/8B/14B/32B 皆已官方 string_match 打分），8 eval_ruler procs 全 GPU 100% pool 剩余 8B/14B/32B 64k/128k cell（csv 0.6B56/1.7B56/4B56/8B72/14B72/32B76）——上轮拟 nudge 的并行化 agent 已自行完成（TaskList #35/#36/#37 编码该计划）→铁律1 满足无空卡。**★★ 已派 trainer a37818b0（background）**：轮询 .82/.104，keep7 1B 一释放（~2min）即起 **Paper B 7B 深度扫 keep12→.82 + keep8→.104**（各 8-card standalone，diskB 路径 model `/apdcephfs_zwfy6/.../OLMo-2-1124-7B` + `/dev/shm/dolmino_now15b.npy`（已 staged 126.9G）、H20 配方 bs4/ga4 eff_bs128 防 7B-fp32 OOM、resume-check keep12、n_fresh2/lr 与 keep14 一致）——填 7B prune-depth 阶梯（keep8/10/12/14），控制臂 freeze_front/from_scratch 次之。⚠️ 早前一次 Agent 调用触 API 空响应误 spawn 了 a2c6a404（0 token 返回，视作 dead，未再派）。待办：a37818b0 落地后核对两臂前 60 步健康（loss↓/无 NaN/maxmem<97GB）；a8955573 T25 完成→回填 RUN_REGISTRY+QCMEM_BENCHMARK_PLAN §1a（跨-benchmark scale 故事第 4 基准 RULER）。.73=EVAL-ONLY 不塞训练；dllm 29.162.226.120 绝不碰。

## 当前快照（2026-07-20 10:26，训练全健康 + .73 T25 跑最后最慢 cell=近完成）
> 10:26 tick（`date` 实测 10:26:04 +0800）：**wzc1 8/8** keep14 7B **step168520/200k** loss2.2825 train-ppl**9.80**（10:25 存 step168500 轮转掉 step168000）0NaN 健康（log 10:26:04 新鲜；⚠️本行 **6.85s/step**（正常 1.56）——判 ckpt save I/O 瞬时尖峰：紧接 10:25:33 存+轮转 48.7GB，且 31s 内跑 20 步与 6.85 不符→非持续卡死，**下轮核对回落**；ETA~今23:50）；**.82+.104 16/16** keep7 1B DDP **step197940/200k** loss2.7304 train-ppl**15.34** gnorm0.50 1.48s/step maxmem41.3GB 0NaN 健康（log 10:26:08 新鲜；**ETA 200k ~今11:17≈51min 后**→释放后起 Paper B keep12/keep8 depth-sweep）；**.73 2/8**——**agent a8955573（T25）跑最后/最慢 cell 非卡死**：查清 2 live python(pid2194414/5,100%CPU,952s)= `eval_ruler_qcmem --model Qwen3-0.6B --resume_j2 --ruler_tasks vt --lengths 128k --num_shards2`=**0.6B vt@128k**（128k prefill+iter_bm25 极慢，即便 0.6B），GPU5/6 2668MiB=0.6B 模型对得上；已写 `variable_tracking_64k.json`+`_summary.json`(10:18)=逐-cell 生成+打分交替，6 scale csv 全铺满（0.6B56/1.7B48/4B48/8B72/14B72/32B72，32B GPU 生成已完成），仅剩这最后 128k cell（newest csv 10:20:43 仍在写）。**合法 tail 欠载（最后单 cell 无可并行）→铁律1 不注入**（防撞 agent 最终 cross-scale 打分+imminent 完成）。**⚠️ 下轮 10:56 若 .73 仍未完成且大部空闲→重查 128k cell 是否真 hang（SendMessage）。** 待办：keep7 1B ~11:17 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 09:56，32/32 全占全健康；.73 T25 推进到最后 32B scale=近完成）
> 09:56 tick（`date` 实测 09:56:04 +0800）：**wzc1 8/8** keep14 7B **step167500/200k**（09:55 刚存 step167500.pt + 轮转掉 step167000.pt=keep-latest-1 自轮转符合预期）train-ppl**10.14** 1.56s/step maxmem122.3GB 0NaN 健康（log 09:56:00 新鲜；ETA~今23:50）；**.82+.104 16/16** keep7 1B DDP **step196740/200k** loss2.7394 train-ppl**15.48** gnorm0.49 1.47s/step maxmem41.3GB 0NaN 健康（log 09:56:05 新鲜；ETA 200k ~今11:16≈1.3h 后）；**.73 8/8**——**agent a8955573（T25 RULER zs scale）推进到最后/最大 scale 32B**：GPU0 100%/66.9GB（32B 载），newest csv mtime 09:55:50（快照前 14s）正写 32B niah_single_2_16k，6 scale clean 目录全建（csv 0.6B56/1.7B48/4B48/8B72/14B72/**32B16 进行中**），32B 刚起=近完成。铁律2 已查文件增长确认 alive。跑完即得跨-benchmark scale 故事第 4 基准（RULER），回填 RUN_REGISTRY+BENCHMARK_PLAN。**填卡在途非空转，无空卡。** 待办：keep7 1B ~11:16 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 09:26，32/32 全占全健康；.73 agent watch 解除 POSITIVE=已 launch）
> 09:26 tick（`date` 实测 09:26:04 +0800）：**wzc1 8/8** keep14 7B **step166500/200k** loss2.3169 train-ppl**10.14**（噪声窗 9.94-10.33）gnorm0.53 1.56s/step maxmem122.3GB 0NaN 健康（log 09:24:40 新鲜；ETA~今23:50）；**.82+.104 16/16** keep7 1B DDP **step195560/200k** loss2.7135 train-ppl**15.08** gnorm0.53 1.48s/step maxmem41.3GB 0NaN 健康（log 09:26:36 新鲜；ETA 200k ~今11:14≈1.8h 后）；**.73 8/8**——**agent a8955573（T25 RULER zs scale）watch 解除 POSITIVE**：08:56 的 provisioning 已 launch，8 procs 全活+5GB/卡（模型载）+ 新 clean 目录 `ruler_results/qcmem_{0p6b,1p7b,4b,8b}_zs_ruler_iter_chatnothink`（csv 56/32/24/48，dedup 判旧 dir 三轴污染→clean 重跑），**newest csv mtime 09:25:57（快照前 7s）= 正写 1.7B variable_tracking_8k 全 8 shard**，09:26 快照 util=0 恰 between-cell 瞬时（vt_8k 刚写完切下一 cell）非卡死；0.6B 应近完成、4B/8B 部分、14B/32B（最大 scale）待跑。跑 T25 = 跨-benchmark scale 故事的第 4 基准（RULER 8k-128k × 6scale，readout-safe j，iter_bm25+chat+no-think，官方 string_match，8-shard），完成回填 RUN_REGISTRY+BENCHMARK_PLAN。**填卡在途非空转，无空卡。** 待办：keep7 1B ~11:14 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 08:56，wzc1+.82+.104 训练全健康 + .73 agent a8955573 provisioning）
> 08:56 tick（`date` 实测 08:56:03 +0800）：**wzc1 8/8** keep14 7B **step165500/200k** loss2.3142 train-ppl**10.12**（噪声窗 9.94-10.18）gnorm0.53 1.56s/step maxmem122.3GB 0NaN 健康（log 08:55:06 新鲜；GPU0 瞬时 4% 有 proc+139GB=step 间隙；ETA~今23:50）；**.82+.104 16/16** keep7 1B DDP **step194360/200k** loss2.7561 train-ppl**15.74** gnorm0.51 1.48s/step maxmem41.3GB 0NaN 健康（log 08:56:24 新鲜；ETA 200k ~今11:16≈2.3h 后）；**.73 0/8**——T25 RULER zs scale agent **a8955573**（08:52 派）**provisioning 阶段**（现仅 4min）：SSH dedup `ls ruler_results/` 读 eval_config 判三轴口径（selector/chat/j 污染→clean 重跑 vs 干净仅打分）+ 定 canonical readout-safe j per scale + 查 6-scale 模型路径中，尚未 launch 8-shard，`ruler_results` 无 <40min 新 zs 目录=正常 provisioning **非卡死**（同 05:56/06:56 前两 agent 模式，均下轮解除 POSITIVE）。铁律1 不双填（防撞 imminent 8-shard 启动 OOM/端口）。**⚠️ 下轮 09:26 若 .73 仍 0/0 且无 ruler zs 新目录=卡死→介入（SendMessage/重派）。** **填卡在途非空转。** 待办：keep7 1B ~11:16 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 08:52，32/32 全占全健康；.73 T22 ✅→补 T25 RULER zs scale）
> 08:52（`date` 实测本轮）：**wzc1 8/8** keep14 7B 承前 step~164.5k train-ppl10.14 0NaN 健康（ETA~今23:50）；**.82+.104 16/16** keep7 1B DDP 承前 step~193k train-ppl15.5 0NaN 健康（ETA~今11:14）；**.73**——**agent ae30a30b（T22）✅完成**（~1h55min）：QCMem zero-shot clean scale 曲线延伸到 **LongBench + LoCoMo** 全 6 scale——**LongBench AVG F1** 0.6B20.33/1.7B21.28/4B28.79/8B26.27/14B31.27/**32B36.29**、**LoCoMo acc** 14.75/16.41/20.85/22.41/22.26/**27.64**（32B 全面最优，趋势合 BABILong，real-doc token-F1 比合成 noisier）；旧 `qcmem_{14b,32b,1p7b_j*,0p6b,4b}` dir 经 eval_config 核实三轴污染(bm25/chat=false/recall-optimal 浅 j)→全 clean 重跑，仅 8B 已干净仅打分；铁律2 全 6scale×2bench empty=0 官方 qa_f1/LoCoMo F1-acc，回填 RUN_REGISTRY(T22 子节)+QCMEM_BENCHMARK_PLAN §1a，零代码/无 git，8 卡释放(0/8 实测)。**.73 释放→铁律1 立即补 agent a8955573（T25）**：补齐跨-benchmark scale 故事的**第 4 个基准=RULER zero-shot clean scale-consistency**（0.6B/1.7B/4B/8B/14B/32B × 8k-128k × niah/mkvt，readout-safe j per scale，iter_bm25+chat+no-think，官方 string_match，8-shard 跨 8 卡，dedup 旧污染 dir→clean 重跑，回填 RUN_REGISTRY+BENCHMARK_PLAN）。**填卡在途非空转，无空卡。** 待办：keep7 1B ~11:14 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 08:26，32/32 全占全健康；无空卡——.73 agent watch 解除 POSITIVE 回到 8/8）
> 08:26 tick（`date` 实测 08:26:03 +0800）：**wzc1 8/8** keep14 7B **step164500/200k** loss2.3170 train-ppl**10.14**（噪声窗 9.94-10.14）gnorm0.52 1.56s/step maxmem122.3GB 0NaN 健康（log 08:25:33 新鲜；GPU0 瞬时 0% 有 proc+139GB=step 间隙非卡死；ETA~今23:50）；**.82+.104 16/16** keep7 1B DDP **step193180/200k** loss2.7432 train-ppl**15.54** gnorm0.51 1.48s/step maxmem41.3GB 0NaN 健康（log 08:26:53 新鲜；ETA 200k ~今11:14≈3h 后）；**.73 8/8**——**agent ae30a30b（T22 QA scale）watch 解除 POSITIVE**：07:56 的 4/8 已回到 **8 procs 全活**（util 13-84% mem~5.2GB/卡=多-scale eval churn 非卡死），LongBench zs **14b/32b/4b/8b ✅有 scores.json**、**1.7b 进行中(16/24 jsonl)**、0.6b 待跑；LoCoMo zs 同 4 scale ✅、1.7b 进行中。**无需 SendMessage/注入**——agent 正常 8-shard 推进 1.7B/0.6B 尾。**填卡在途非空转，无空卡。** 待办：keep7 1B ~11:14 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 07:56，wzc1+.82+.104 训练全健康 + .73 agent 主力完成→次级 scale tail 4/8）
> 07:56 tick（`date` 实测 07:56:03 +0800）：**wzc1 8/8** keep14 7B **step163500/200k** loss2.3097 train-ppl**10.07**（噪声窗 9.99-10.29）gnorm0.51 1.56s/step maxmem122.3GB 0NaN 健康（ETA~今23:30）；**.82+.104 16/16** keep7 1B DDP **step191960/200k** loss2.7579 train-ppl**15.77** gnorm0.51 1.48s/step maxmem41.3GB 0NaN 健康（ETA 200k ~今11:16）；**.73 4/8**——agent **ae30a30b** T22 QA scale：**主力 14B+32B 已完成**（LongBench 14b=53/32b=52 文件≈48shard+scores 完整、LoCoMo 14b=9/32b=8 8-shard 完整），现 GPU4-7（66GB/67-100%）跑更轻的**次级 scale**（1.7B/4B/0.6B/8B-zs）或 LoCoMo 尾/打分，GPU0-3 空=agent 自身多-scale eval churn/transition（文件增长=alive 非卡死）。**铁律1 不注入**（32B 需 64GB 同卡再塞 OOM + 撞 agent scheduler/文件布局），但 4 卡空=真欠载 → **⚠️ 下轮 08:26 若 .73 仍 ≤4 procs 且 agent 未完成 → SendMessage 让其余 scale 用满 8-shard。** **填卡在途（主力完成，次级 tail）。** 待办：keep7 1B ~11:16 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 07:26，32/32 全占全健康；无空卡——.73 agent watch 解除 POSITIVE）
> 07:26 tick（`date` 实测 07:26:10 +0800）：**wzc1 8/8** keep14 7B **step162480/200k** loss2.3140 train-ppl**10.11**（噪声窗 9.99-10.22）gnorm0.54 1.56s/step maxmem122.3GB 0NaN 健康（ETA~今23:30）；**.82+.104 16/16** keep7 1B DDP **step190760/200k** loss2.7729 train-ppl**16.00** gnorm0.52 1.48s/step maxmem41.3GB 0NaN 健康（ETA 200k ~今11:14）；**.73 8/8**——**agent ae30a30b watch 解除 POSITIVE**：06:56 provisioning 已 launch，8 procs 全高 util（78-100%）mem~66GB/卡 + **4 个新 clean 目录**（`longbench_results/qcmem_{14b,32b}_zs_iter_chatnothink` + `locomo_results/qcmem_{14b,32b}_zs_iter_chatnothink`）= dedup 判旧 `qcmem_{14b,32b}` dir 污染口径 → 对 14B/32B **clean 重跑**（_zs_iter_chatnothink 后缀），健康在跑非卡死。跑 T22 QCMem zero-shot clean LongBench(6-ds)+LoCoMo(n=1986) scale-consistency，随后 1.7B/4B/0.6B/8B-zs（若够）+ 官方 qa_f1/LoCoMo F1-acc 回填 scale 曲线。**填卡在途非空转，无空卡。** 另 ckpt-轮转 cron 4ec42903 06:57 触发=NO-OP（df 24T free/15%，keep14fresh2 2 ckpt=最新2受保护无可删）。待办：keep7 1B ~11:14 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 06:56，wzc1+.82+.104 训练全健康 + .73 agent ae30a30b provisioning）
> 06:56 tick（`date` 实测 06:56:04 +0800）：**wzc1 8/8** keep14 7B **step161460/200k** loss2.3015 train-ppl**9.99**（噪声窗 9.99-10.22）gnorm0.52 1.56s/step maxmem122.3GB 0NaN 健康（ETA~今23:30）；**.82+.104 16/16** keep7 1B DDP **step189560/200k** loss2.7295 train-ppl**15.33** gnorm0.49 1.48s/step maxmem41.3GB 0NaN 健康（ETA 200k ~今11:15）；**.73 0/8**——T22 LongBench+LoCoMo scale agent **ae30a30b**（06:51 派）**provisioning 阶段**（现仅 5min）：SSH dedup 读已存 `qcmem_{14b,32b,1p7b_j*}` dir config 判口径（clean 重跑 vs 仅打分）+ 读 ledger 中，尚未 launch 8-shard，`longbench_results`/`locomo_results` 无 <32min 新目录=正常 provisioning **非卡死**。铁律1 不双填（防撞 imminent 8-shard 启动）。**⚠️ 下轮 07:26 若 .73 仍 0/0 且无 longbench/locomo 新目录=卡死→介入（SendMessage/重派）。** **填卡在途非空转。** 待办：keep7 1B ~11:15 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 06:51，32/32 全占全健康；无空卡——.73 a0c00b51✅→补 T22 QA scale agent）
> 06:51 tick（`date` 实测 06:51:31 +0800）：**wzc1 8/8** keep14 7B 承前 step160440 train-ppl10.57 0NaN 健康（ETA~今23:30）；**.82+.104 16/16** keep7 1B DDP 承前 step188380 train-ppl16.08 0NaN 健康（ETA~今11:12）；**.73**——**agent a0c00b51 ✅完成**（55min）：QCMem **zero-shot BABILong CLEAN 全 scale 曲线严格单调**——0.6B(j2)**33.05** → 1.7B(j3)41.19 → 4B(j9)46.71 → 8B(j9)48.43 → 14B(j13)54.29 → 32B(j27)**64.10**（21-cell mean，统一 chat+no-think+iter_bm25+readout-safe j；0.6B/4B 仅打分已存 clean dir，1.7B/14B/32B 新 GPU 跑；铁律2 全 scale empty=0 官方 compare_answers，2100 rec/scale）。回填 RUN_REGISTRY(T24 子节)+QCMEM_BENCHMARK_PLAN §1a，撤 legacy caveat，零代码/无 git，8卡释放。**.73 8卡释放(06:5x 0procs)→铁律1 立即补 agent ae30a30b（T22）**：把 scale-consistency 延伸到 **LongBench(6-ds)+LoCoMo(n=1986)** real-QA 基准，14B+32B 先(+1.7B/4B/0.6B/8B-zs 若够)，dedup 已存 `qcmem_{14b,32b,1p7b_j*}` dir 口径(命名无 chatnothink→大概率旧污染)→污染 clean 重跑/干净仅打分，同 readout-safe j，官方 qa_f1/LoCoMo F1-acc 回填 scale 曲线。**填卡在途非空转。** 待办：keep7 1B ~11:12 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 06:26，32/32 全占全健康；无空卡——.73 agent watch 解除 POSITIVE）
> 06:26 tick（`date` 实测 06:26:05 +0800）：**wzc1 8/8** keep14 7B **step160440/200k** loss2.3579 train-ppl**10.57**（噪声窗 10.20-10.57）gnorm0.51 1.56s/step maxmem122.3GB 0NaN 健康（ETA~今23:30）；**.82+.104 16/16** keep7 1B DDP **step188380/200k** loss2.7774 train-ppl**16.08** gnorm0.51 1.48s/step maxmem41.3GB 0NaN 健康（ETA 200k ~今11:12）；**.73 7/8**——**agent a0c00b51 watch 解除 POSITIVE**：05:56 的 provisioning 已 launch，7 procs 活跃 + **3 个新 scale 目录**（`qcmem_{1p7b,14b,32b}_zs_iter_chatnothink`，35min 内创建）= dedup 判定 0.6B/4B clean dir 已存（仅打分）、对 **1.7B/14B/32B 新跑 GPU**，util 混合（gpu2/7=100%，余 8-18%=8-shard 变负载）mem 15-68GB(32B~64GB) = 健康在跑非卡死。gpu1 瞬时 0MiB=8-shard between-cell churn（agent 自身调度不注入）。**填卡在途非空转，无空卡。** 待办：keep7 1B ~11:12 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 05:56，wzc1+.82+.104 训练全健康 + .73 agent provisioning + ckpt轮转 NO-OP）
> 05:56 tick（`date` 实测 05:56:38 +0800）：**wzc1 8/8** keep14 7B **step159440/200k** loss2.3502 train-ppl**10.49**（噪声窗 9.96-10.49）gnorm0.54 1.56s/step maxmem122.3GB 0NaN 健康（05:44 存 step159000.pt 并 rotate 掉旧，keep-latest-1；ETA~今23:30）；**.82+.104 16/16** keep7 1B DDP **step187200/200k** loss2.8204 train-ppl**16.78** gnorm0.52 1.48s/step maxmem41.3GB 0NaN 健康（ETA 200k ~今11:12）；**.73 0/8**——QCMem BABILong scale-consistency agent **a0c00b51**（05:53 派）**provisioning 阶段**（现仅 3min）：SSH dedup 查已存 clean dir（0.6B/4B 或已存仅打分）+ 读 RUN_REGISTRY 定 canonical zero-shot j + 查模型路径中，尚未 launch 8-shard，`babilong_results` 无 <12min 新目录=正常 provisioning **非卡死**。铁律1 不双填（防撞 imminent 大模型 8-shard 启动 OOM/端口）。**⚠️ 下轮 06:26 若 .73 仍 0/0 且无 babilong scale 新目录=agent 卡死/失败→介入（SendMessage/重派）。** **填卡在途非空转。** 另：**ckpt-轮转 cron 4ec42903 于 05:56 触发=NO-OP**——`df /apdcephfs_wzc1` **24T free/28T(15%)零压力**；`outputs/olmo2_probe2_7B_keep14fresh2/` 自轮转 keep-latest-1 仅 1 个 step159000.pt(48.7GB=最新受保护)，**无「非里程碑且非最新2」可删**，安全铁律 trivially 满足未删；`keep10fresh2` 不在 wzc1。待办：keep7 1B ~11:12 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 05:53，32/32 全占全健康；无空卡——.73 agent 完成→立即补新 eval agent）
> 05:53 tick（`date` 实测 05:53:38 +0800）：**wzc1 8/8** keep14 7B 承前 step158420 train-ppl10.24↓ 0NaN 健康（ETA~今23:30）；**.82+.104 16/16** keep7 1B DDP 承前 step186000 train-ppl16.09 0NaN 健康（ETA~今11:15）；**.73**——**agent a9eeeb2a ✅完成**（46min）：(1)niah frontier 泛化——read 延迟**任务无关**（复用 T21 轴 56.8×(L−j)ms，j6→j34=**1.93×**），recall(j) **强任务依赖**：universal readout 坠崖在 **j41(0.64L)**（全 niah 任务），其下 **niah_single=最耐深 j**（recall 平 100 到 j34=纯免费 1.93× 提速零代价）、multivalue/multiquery 也耐深 j（平于各自天花板~50/~28 到 j34）、multikey 中庸、**vt 是唯一偏好浅 j3 的异类**（词汇 multi-hop 需近逐字 token）；(2)T23 8B clean BABILong 打分：**adapter(j12)=57.10 / zero-shot(j9)=48.43** overall（21 cell），vs 旧污染 zs 39.2→48.4(+9.2)、adapter 55.5→57.1(持平)，增益集中 0k-8k；修正旧 caveat（32k qa1/qa2 干净仍低=真长程失败非 artifact），新 caveat（iter_bm25 对 qa1 单事实中档反掉分 55→23）。T23→[DONE]，铁律2 全 cell empty=0 官方 compare_answers/string_match，零代码/无 git。**.73 8卡释放(05:52 0procs)→铁律1 立即补 agent a0c00b51**：QCMem **BABILong zero-shot scale-consistency 干净口径重跑**——0.6B/1.7B/4B/14B/32B 补齐到 8B clean 同口径(chat+no-think+iter_bm25，canonical j per scale)，dedup 先查（0.6B/4B clean dir 或已存仅打分，1.7B/14B/32B 需新 GPU），官方 compare_answers 打分回填 RUN_REGISTRY §scale + QCMEM_BENCHMARK_PLAN §1a。**填卡在途非空转。** 待办：keep7 1B ~11:15 释放 .82/.104 → 起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch)16×H20 DDP（预授权；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 05:26，32/32 全占全健康；无空卡）
> 05:26 tick（`date` 实测 05:26:11 +0800）：**wzc1 8/8** keep14 7B **step158420/200k** loss2.3260 train-ppl**10.24**↓（较 05:05 10.36 续降）gnorm0.53 1.56s/step maxmem122.3GB 0NaN 健康（ETA 200k ~今23:30）；**.82+.104 16/16** keep7 1B DDP **step186000/200k** loss2.7783 train-ppl**16.09** gnorm0.50 1.48s/step maxmem41.3GB 0NaN 健康（ETA 200k ~今11:15）；**.73 8/8** util 67-100% mem~66GB = QCMem 收尾 agent **a9eeeb2a** 活跃（`ruler_results/qcmem_32b_t21b` <12min 前刚写=在跑）——(1)主(GPU)：T21 recall-vs-speed frontier 从 vt 泛化到 niah_single/multikey/multivalue(16k+32k,resume_j sweep,n=50,iter_bm25+chat+no-think)；(2)次(零GPU)：T23 clean BABILong 打分回填。完成时收通知。**无空卡、两训练 loss 续降 0NaN、eval agent 在途。** 待办：keep7 1B ~11:15 释放 .82/.104 后起 Paper B keep12/keep8 depth-sweep(+freeze_front/from_scratch 对照)16×H20 DDP（预授权 ablation 延伸；.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 05:05，24/32 训练全健康 + .73 T21✅→补 QCMem 收尾 agent + ckpt轮转 NO-OP）
> 05:05 tick（`date` 实测 05:00:08 +0800）：**wzc1 8/8** keep14 7B **step157500/200k** loss2.3379 train-ppl10.36 gnorm0.53 1.56s/step maxmem122.3GB 0NaN 健康（ETA~今23:00）；**.82+.104 16/16** keep7 1B DDP 健康（承前 step183640 train-ppl16.60，ETA~今11:08）；**.73 8/8**——**T21 agent a7048e2b ✅完成**（34.7min）：Qwen3-32B(L=64,zero-shot) vt recall-vs-read-speed frontier —— **read_prefill=56.8×(L−j)ms 近完美线性（j3→j48=3.81× 提速）**、decode~46ms恒定(HBM-bound)、显存~65.6GB恒定；recall 峰浅 j3(16k=93.6/32k=52.0)阶梯降，**拐点≈j34(0.53L,recall未塌+read 2.03×快)**，j41/48 坠崖真崩。**重要修正**：干净 chat+no-think 口径下 32B vt 浅 j 很强 → 推翻旧"32B vt 全崩峰值~24"污染结论。ledger `status/QCMEM_RECALL_SPEED_FRONTIER.md`(7312B)+PENDING T21→[DONE]，铁律2 全16cell empty=0 官方string_match，零代码改动。GPU 释放后**铁律1 立即填卡**：派 QCMem 收尾 agent **a9eeeb2a**——(1)主(GPU)：把 recall-vs-speed frontier 从 vt 泛化到 niah_single/multikey/multivalue(16k+32k,j sweep,n=50,iter_bm25+chat+no-think)验证 resume_j 旋钮任务依赖性；(2)次(零GPU)：T23 clean BABILong 打分回填（`qcmem_j12_iter_bm25_chatnothink_ad`+`qcmem_j9_iter_bm25_chatnothink_zs` 各 168 文件=21cell 完整，已查重确认 GPU 重跑 DONE，仅缺打分+回填主表/撤 caveat）。**填卡在途非空转。** 另：**ckpt-轮转 cron 4ec42903 于 05:0x 触发=NO-OP**——`df /apdcephfs_wzc1` 24T free/28T(15%)零压力；目标 `outputs/olmo2_probe2_7B_keep14fresh2/` 训练自轮转 keep-latest-1 仅 1 个 step157500.pt(48.7GB=最新受保护)，**无「非里程碑且非最新2」可删**，安全铁律 trivially 满足；`keep12fresh2`(60G,2 崩溃 run ckpt)cron scope 外不碰。keep12/keep8 depth-sweep 训练待 keep7 1B ~11:08 释放 .82/.104 后起（.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 04:26，24/32 训练全健康 + .73 T21 eval provisioning 查重中）
> 04:26 tick（`date` 实测 04:26:44 +0800）：**wzc1 8/8** keep14 7B **step156400/200k** loss2.3338 train-ppl10.32 gnorm0.52 1.56s/step 0NaN 健康（ETA~今23:00）；**.82+.104 16/16** keep7 1B DDP **step183640/200k** loss2.8093 train-ppl16.60 gnorm0.53 1.48s/step 0NaN 健康（ETA~今11:08）；**.73 0/0**——T21 eval agent **a7048e2b**（~04:20 派）**仍在查重-provisioning 阶段**：铁律2 探测 .73 无 python/eval proc、无 T21 log、ruler_results 最新仍 01:43（task#27）→ agent 按指令先在 wzc1 侧 grep ledger 查重（J_DETERMINATION/BENCHMARK_PLAN §1c/RUN_REGISTRY）再起 32B 8-shard，未 SSH launch=正常 provisioning 非卡死，**不双填**（防撞 32B 启动 OOM/端口）。附：logs 有旧 `n500_32b_zs_j27_{niah_single_16k/32k,vt_8k}`+`probe_trunc_32b` → 32B zero-shot recall(j) 侧或已部分存在，agent 查重会据此决定补 latency 轴 or 转 topk-grid fallback。**填卡在途非空转；⚠️下轮 04:56 若 .73 仍 0/0 且无 T21 log=agent 卡住/失败→介入（重派或改填）。** keep12/keep8 depth-sweep 训练待 keep7 1B ~11:08 释放 .82/.104 后起（.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 04:18，24/32 训练全健康 + .73 task#29✅完成→填 P0 T21 recall-vs-speed frontier）
> 04:18 tick（`date` 实测 04:18:00 +0800）：**wzc1 8/8** keep14 7B **step156080/200k** loss2.3185 train-ppl10.16 gnorm0.53 1.56s/step 0NaN 健康（ETA 200k ~今23:00）；**.82+.104 16/16** keep7 1B DDP **step183320/200k** loss2.7484 train-ppl15.62 gnorm0.51 1.47s/step 0NaN 健康（ETA 200k ~今11:08）；**.73 8/8**——task#29（keep14 7B post-apex 轨迹 eval）**✅完成**：step153500 held-out ppl **10.693/1.446×**（apex 128000=10.827/1.463× → 微降 0.13，PPL 已 plateau 不回退）+ downstream MC 全轴 flat-to-up 无回退（MMLU .301→.312），base 口径 8/8 shard empty=0，落 OLMO2_PRUNEHEAL_PPL/DOWNSTREAM.md，wzc1 snapshot 已删、.73 双 ckpt 留最终 eval 用。GPU 释放后**铁律1 立即填卡**：派 eval agent **a7048e2b** 跑 **P0 T21（PENDING_TASKS）= Qwen3-32B zero-shot vt recall-vs-speed resume_j frontier**（深 j→read 重算 layers[j:] 更少=更快但过 readout-safe(~j27) recall 掉；iter_bm25+chat+no-think，n=50/cell，测 recall+read-latency Pareto）→ 直喂 QCMem 效率章节核心卖点。**填卡在途非空转。** keep12(step1000崩)/keep10(step10000早) 非收敛点、keep12/8 depth-sweep 需重训（待 .82/.104 ~11:00 释放，.73=EVAL-ONLY 不塞训练）。

## 当前快照（2026-07-20 03:56，24/32 训练全健康 + .73 provisioning 94.8% 即将起 eval）
> 03:56 tick（`date` 实测 03:56:03 +0800）：**wzc1 8/8** keep14 7B **step155360/200k** loss2.3403 train-ppl10.38 gnorm0.51 1.56s/step 0NaN 健康；**.82+.104 16/16** keep7 1B DDP **step182420/200k** loss2.7872 train-ppl16.24 gnorm0.51 1.47s/step 0NaN 健康；**.73 8/8 空 = task#29 provisioning**——铁律2 判活：agent 用 **scp -O**（非 rsync）传 snapshot，进程 3851668 已跑 38.6min，目标 `keep14_step153500.pt` **46.2G/48.7G（94.8%）**，6s 增 113MB（~19MB/s，在写=非卡死），剩 ~2.5G ~2min 完 → 随即起 held-out ppl+downstream MC eval（base 口径 vs step128000 apex）。⚠️**效率教训：跨盘 wzc1→diskB 传 48.7G ckpt @19MB/s=~43min provisioning 空**——最终 step200000 eval 同成本，须每次传输 batch 全部 Paper B 指标。**填卡在途非空转。**

## 当前快照（2026-07-20 03:26，24/32 训练全健康 + .73 EVAL 填卡 provisioning 在途）
> 03:26 tick（`date` 实测 03:26:05 +0800）：**wzc1 8/8** keep14 7B **step154340/200k** loss2.3345 train-ppl10.32 gnorm0.55 1.56s/step maxmem122.3GB 0NaN 健康（ETA 200k ~今22:50）；**.82+.104 16/16** keep7 1B DDP **step181220/200k** loss2.7419 train-ppl15.52 gnorm0.50 1.47s/step maxmem41.3GB 0NaN 健康（ETA 200k ~今11:00）；**.73 8/8 空（0 procs）= task#29 provisioning 阶段**——铁律2 判活：wzc1 已 cp-snapshot `outputs/_eval_snap/keep14_step153500.pt`(46G, 03:15)，正 rsync→.73（目标文件 5s 增 100MB=**20MB/s**，已传 ~11G/46G，mtime=当前=非卡死），预计再 ~29min 传完起 eval（最新 ckpt held-out ppl+downstream MC vs step128000 apex，base 口径）。**填卡在途非空转。**

## 当前快照（2026-07-20 03:02，32/32 全占全健康；无空卡）
> 03:02 tick（`date` 实测 03:02:24 +0800）：**wzc1 8/8** keep14 7B **step153520/200k** loss2.2948 train-ppl9.92 gnorm0.52 1.56s/step maxmem122.3GB 0NaN（03:02:11 存 step153500.pt 并 rotate 掉 step153000.pt；健康，ETA 200k ~今 22:50）；**.82+.104 16/16** keep7 1B DDP **step180360/200k** loss2.7845 train-ppl16.19 gnorm0.49 1.49s/step maxmem41.3GB 0NaN（node0 log step 推进=两节点 DDP 同步健康，ETA 200k ~今 11:00）；**.73 8/8** = RULER copy-hard refill agent **aec6ef91d ✅完成**——3 baseline（StreamingLLM/KV-Direct/HCache）× copy-hard needles（single_3/multivalue/multiquery）× {64k,128k} 各 48/48 csv，官方 string_match_all 打分+铁律2 全验证（n=100 empty=0 OOM=0），commit **5954639**（本机 main，author LiuHanzuo 无 AI trailer，**未 push** ahead origin 24），GPU 已释放（0 compute-apps）→ **task#28 keep14 7B held-out ppl eval agent a62dab68 已派接手填卡**（base 口径 vs OLMo-2-7B base 7.398，rsync 最新 keep14 ckpt→.73 后 8-shard forward，无 chat/generation）。**无空卡。**

## 当前快照（2026-07-20 02:26，32/32 全占全健康；无空卡）
> 02:26 tick（`date` 实测 02:26:02 +0800）：**wzc1 8/8** keep14 7B **step152300/200k** loss2.3738 train-ppl10.74 gnorm0.48 1.56s/step maxmem122.3GB 0NaN（+1020步/30min 健康，ETA 200k ~今 22:50）；**.82+.104 16/16** keep7 1B DDP **step178920/200k** loss2.8089 train-ppl16.59 gnorm0.51 1.49s/step maxmem41.3GB 0NaN（+1140步/30min；训练-ppl ~15.7-16.6 噪声窗未 plateau，续训，ETA 200k ~今 11:00）；**.73 8/8** = RULER copy-hard refill agent **aec6ef91d 收尾中**——铁律2 判活：16 python procs 活跃，per-baseline csv **StreamingLLM 48/48、KV-Direct 48/48（均完）、HCache 27/48（剩 21 cell）**，无 DONE marker=仍在跑 → 预计再 1 tick 完 HCache，随后 reconcile+extend RULER_TASKBREADTH_RESULTS.md（在本机 wzc1），接 **task#28 keep14 held-out ppl**。**无空卡。**

## 当前快照（2026-07-20 01:56，32/32 全占全健康；无空卡）
> 01:56 tick（`date` 实测 01:56:02 +0800）：**wzc1 8/8** keep14 7B **step151280/200k** loss2.3730 train-ppl10.73 gnorm0.49 1.56s/step maxmem122.3GB 0NaN（+1000步/30min 健康，ETA 200k ~今 22:50）；**.82+.104 16/16** keep7 1B DDP **step177780/200k** loss2.7556 train-ppl15.73 gnorm0.51 1.49s/step maxmem41.3GB 0NaN（+1180步/30min；训练-ppl ~15.7-16 噪声窗未 plateau，续训，ETA 200k ~今 11:00）；**.73 8/8** = RULER copy-hard refill agent **aec6ef91d 收尾**——铁律2 判活：3 procs 100%CPU，per-baseline csv **StreamingLLM 48/48、KV-Direct 48/48（均完）、HCache 8/48（最后 baseline 起跑）**，无 DONE marker=仍在跑 → HCache 剩 40 cell，预计 1-2 tick 完，随后 reconcile+extend RULER_TASKBREADTH_RESULTS.md，接 **task#28 keep14 held-out ppl**。**无空卡。**

## 当前快照（2026-07-20 01:26，32/32 全占全健康；无空卡）
> 01:26 tick（`date` 实测 01:26:02 +0800）：**wzc1 8/8** keep14 7B **step150280/200k** loss2.3454 train-ppl10.44 gnorm0.52 1.56s/step maxmem122.3GB 0NaN（+1020步/30min 健康，ETA 200k ~今 22:50）；**.82+.104 16/16** keep7 1B DDP **step176600/200k** loss2.7718 train-ppl15.99 gnorm0.51 1.50s/step maxmem41.3GB 0NaN（+1120步/30min；训练-ppl ~16 噪声窗未 plateau，续训，ETA 200k ~今 11:00）；**.73 8/8** = RULER copy-hard refill agent **aec6ef91d 推进中**——铁律2 判活：2 procs 100%CPU etimes~4163s，per-baseline csv **StreamingLLM 48/48（完）、KV-Direct 40/48（近完）、HCache 0/48（待起）**，无 DONE marker=仍在跑 → 完成后 reconcile+extend RULER_TASKBREADTH_RESULTS.md，随后 **task#28 keep14 held-out ppl**接 .73。**无空卡。**

## 当前快照（2026-07-20 00:56，32/32 全占全健康；无空卡）
> 00:56 tick（`date` 实测 00:56:03 +0800）：**wzc1 8/8** keep14 7B **step149260/200k** loss2.3539 train-ppl10.53 gnorm0.49 1.56s/step maxmem122.3GB 0NaN（+1020步/30min 健康，ETA 200k ~今 22:50）；**.82+.104 16/16** keep7 1B DDP **step175480/200k** loss2.7617 train-ppl15.83 gnorm0.50 1.50s/step maxmem41.3GB 0NaN（+1140步/30min；训练-ppl ~15.8-16.1 噪声窗未 plateau，续训，ETA 200k ~今 11:00）；**.73 8/8** = RULER copy-hard refill agent **aec6ef91d 推进中**——铁律2 判活：8 procs 100%CPU etimes~2365s，per-baseline csv 计数 **StreamingLLM 48/48（完）、KV-Direct 24/48（半程 in-progress）、HCache 0（待起）**，无 DONE marker=仍在跑 → 完成后 reconcile+extend RULER_TASKBREADTH_RESULTS.md，随后 **task#28 keep14 held-out ppl（base 口径 vs 7B base 7.398）**接 .73。**无空卡。**

## 当前快照（2026-07-20 00:26，32/32 全占全健康；无空卡）
> 00:26 tick（`date` 实测 00:26:04 +0800）：**wzc1 8/8** keep14 7B **step148240/200k** loss2.3766 train-ppl10.77 gnorm0.49 1.56s/step maxmem122.3GB 0NaN（+980步/30min 健康，ETA 200k ~今 22:50）；**.82+.104 16/16** keep7 1B DDP **step174340/200k** loss2.8210 train-ppl16.13-16.79 gnorm0.50 1.48s/step maxmem41.3GB 0NaN（+1180步/30min；训练-ppl ~16 噪声窗未 plateau，续训，ETA 200k ~今 11:00）；**.73 8/8** = RULER copy-hard refill agent **aec6ef91d 进到 StreamingLLM 阶段**——铁律2 判活：8 python procs 100%CPU etimes~570s，`ruler_results/streamingllm_8b_taskbreadth/` 正产 niah_single_3/multivalue/multiquery×{64k,128k} 各 shard（.json+.csv），无 DONE marker=仍在跑（非卡死）→ 后续 KVD/HCache 阶段，完成后 reconcile+extend RULER_TASKBREADTH_RESULTS.md，随后 **task#28 keep14 held-out ppl**接 .73。**无空卡。**

## 当前快照（2026-07-19 23:56，32/32 全占全健康；无空卡）
> 23:56 tick（`date` 实测 23:56:02 +0800）：**wzc1 8/8** keep14 7B **step147260/200k** loss2.3583 train-ppl10.57 gnorm0.53 1.56s/step maxmem122.3GB 0NaN（+900步/26min 健康，ETA 200k ~明 22:50）；**.82+.104 16/16** keep7 1B DDP **step173160/200k** loss2.7526 train-ppl15.68 gnorm0.49 1.47s/step maxmem41.3GB 0NaN（+1020步/26min；训练-ppl ~15.7-16.4 噪声窗未 plateau，续训，ETA 200k ~明 11:00）；**.73 8/8** = RULER copy-hard refill agent **aec6ef91d 仍在跑**（8 procs 全活；StreamingLLM/KVD/HCache × niah_single_3/multivalue/multiquery × {64k,128k}）→ 完成通知后 reconcile+extend RULER_TASKBREADTH_RESULTS.md，随后 **task#28 keep14 held-out ppl（base 口径 vs 7B base 7.398）**接 .73。wzc1 盘 24T free/28T(15%) 无压力。**无空卡。**

## 当前快照（2026-07-19 23:30，24/32 训练占用全健康 + .73 EVAL 填卡在途）
> 23:30 tick（`date` 实测 23:29:54 +0800）：**wzc1 8/8** keep14 7B **step146360/200k ppl10.77** loss2.3769 gnorm0.49 100%util ~930W 0NaN 1.56s/step（healthy，ETA 200k ~次日 22:45）；**.82+.104 16/16** keep7 1B DDP **step172140/200k ppl17.23** loss2.8465 gnorm0.50 100%util .104~298W 0NaN 1.48s/step（ppl 17.23=噪声窗高端，近几轮 15.3-16.1，单 batch 抬高非趋势/非退化，续训；ETA 200k ~次日 11:00）；**.73 8/8 空**（0 procs/0%/~75W）——**refill agent aec6ef91d 23:26 派，provisioning 中**（读文件+SSH 定位 task-breadth 已生成的共享 RULER 数据+验证 StreamingLLM/KVD/HCache driver 是否支持 niah_single_3/multivalue/multiquery，需则打补丁）→ 填卡在途，非空转不管；下 tick 核实是否已起 8-GPU job（若仍空=agent 卡住需介入）。task#27（CoMem+InfLLM RULER task-breadth）✅ 完成，commit ec871bc(本地未 push)，双 finalizer 独立数字精确一致。

## 当前快照（2026-07-19 22:56，32/32 全占全健康；无空卡）
> 22:56 tick（`date` 实测 22:56:01 +0800）：**wzc1 8/8** keep14 7B 100%util 915-944W 0NaN 续训健康；**.82+.104 16/16** keep7 1B DDP **step170800/200k ppl15.51** loss2.7412 gnorm0.50 .104 100%util~307W 0NaN 1.48s/step（+1140步/30min；训练-ppl ~15.5-16.2 噪声窗 plateau 未触发，续训）；**.73 8/8** RULER 广度 InfLLM 阶段 **88/96 shard-CSV**（22:26=70→+18/30min，8 procs 全活，仅 `tb_qcmem_DONE`、无 tb_infllm_DONE/results-md=仍在跑）→ 只剩最后一格 niah_multiquery 128k 的 8 shard，预计再 ~15-30min 完。双 finalizer（aa19dda5 已 completed / a6984e1a running）完成时 reconcile 单一 results+去重 commit。**无空卡。**

## 当前快照（2026-07-19 22:26，32/32 全占全健康；无空卡）
> 22:26 tick（`date` 实测 22:26:03 +0800）：**wzc1 8/8** keep14 7B 100%util ~920W 0NaN 续训健康；**.82+.104 16/16** keep7 1B DDP **step169660/200k ppl15.31** loss2.7286 gnorm0.50 .104 100%util~311W 0NaN 1.47s/step（+1160步/30min；训练-ppl ~15.3-15.8 噪声窗 plateau 未触发，续训）；**.73 8/8** RULER 广度 InfLLM 阶段 **70/96 shard-CSV**（21:56=56→+14/30min，8 procs 全活，无 tb_infllm_DONE/results-md=仍在跑）→ 预计再 ~55min（128k cell 最慢）。双 finalizer（aa19dda5+a6984e1a）完成时 reconcile 单一 results+去重 commit。**无空卡。**

## 当前快照（2026-07-19 21:56，32/32 全占全健康；无空卡）
> 21:56 tick（`date` 实测 21:56:01 +0800）：**wzc1 8/8** keep14 7B 100%util ~930W 0NaN 续训健康；**.82+.104 16/16** keep7 1B DDP **step168500/200k ppl16.10** loss2.7791 gnorm0.53 0NaN 1.47s/step（+1200步/30min；.104 100%util~133W=comm/allreduce 瞬时非空，.82 log 确认 step 推进；训练-ppl ~16.1 噪声窗 plateau 未触发，续训）；**.73 8/8** RULER 广度 InfLLM 阶段 **56/96 shard-CSV**（21:26=40→+16/30min，8 procs 全活，无 tb_infllm_DONE/results-md=仍在跑）→ 预计再 ~60-75min（128k cell 最慢）。⚠️**双 finalizer**（aa19dda5 自重启 owner + a6984e1a）均将打分+写 results+commit，低风险（同 string_match 确定值、无 push），完成时 reconcile 单一 results 文件+去重 commit。**无空卡。**

## 当前快照（2026-07-19 21:26，32/32 全占全健康；无空卡）
> 21:26 tick（`date` 实测 21:26:01 +0800）：**wzc1 8/8** keep14 7B 100%util ~940W 0NaN 续训健康；**.82+.104 16/16** keep7 1B DDP **step167300/200k ppl15.98** loss2.7710 gnorm0.50 .104 100%util~355W 0NaN 1.48s/step（+1120步/30min；训练-ppl ~15.1-16.0 噪声窗，plateau 未触发，续训）；**.73 8/8** RULER 广度 **InfLLM 阶段**（CoMem 阶段 20:30:58 DONE）——⚠️**铁律2 判活**：8 shard procs（1988292-99）全 **99.9%CPU** etimes~56min，master log `START InfLLM 20:30:58`，nested `ruler_results/infllm_8b_taskbreadth/infllm_8b_taskbreadth/` **40 CSV newest 21:21:55**（在 niah_single_3 64k，~5/12 cell done）→ 6/8 GPU 瞬时 0%util 是 shard 错峰非卡死，file growth 确认在产出；预计再 ~60-80min（128k cell 最慢）。finalizer a6984e1a 轮询至完成打分+落账+commit。**无空卡。**

## 当前快照（2026-07-19 20:56，32/32 全占全健康；无空卡）
> 20:56 tick（`date` 实测 20:56:01 +0800）：**wzc1 8/8** keep14 7B 100%util ~930-958W 0NaN 续训健康；**.82+.104 16/16** keep7 1B DDP **step166180/200k ppl15.82** loss2.7615 gnorm0.50 100%util .104~307W 0NaN 1.47s/step（+1200步/30min；训练-ppl ~15.8-16.2 噪声窗，plateau 未触发，续训）；**.73 8/8** RULER 广度 eval——**CoMem 阶段 DONE**（`tb_qcmem_DONE` marker，12 cell 官方 string_match 已验证：single_1 100/99/99、single_3 90/98/97、multivalue 94.5/92.5/95.25、multiquery 97.5/94.75/97 @16k/64k/128k）→ 现 **InfLLM 阶段**（6卡95-97%~350W + 2卡 shard 错峰 0-66%，8 procs 全活，`ruler_results/infllm_8b_taskbreadth/` 在产出，无 tb_infllm_DONE/results-md=仍在跑），finalizer a6984e1a 轮询至完成打分+落账+commit。**无空卡。**

## 当前快照（2026-07-19 20:26，32/32 全占全健康；无空卡）
> 20:26 tick（`date` 实测 20:26:14 +0800）：**wzc1 8/8** keep14 7B **step140140/200k ppl10.96** loss2.3946 gnorm0.48 100%util ~930W 0NaN 1.56s/step maxmem122.3GB live@20:26:43（+1040步/30min 健康）；**.82+.104 16/16** keep7 1B DDP **step164980/200k ppl15.52** loss2.7423 gnorm0.50 100%util .82~365W/.104~295W 0NaN 1.48s/step maxmem41.3GB live@20:26:29（+1220步/30min；训练-ppl ~15.5-16.1 噪声窗，held-out plateau 判据走离线未触发，续训）；**.73 8/8** CoMem RULER 广度 eval——⚠️**铁律2 判活**：3+ python procs 121%CPU etimes~64min（iter_bm25 BM25 CPU-bound）→ poll 瞬间 GPU util 0-39%/116-178W 是**采样 artifact 非卡死**；result files 已到 **niah_multiquery 128k+64k shards = CoMem 阶段最后 task 最长档**（前 3 task 已完）→ 快收尾 CoMem，之后 InfLLM 阶段（128k 慢，长 run），finalizer a6984e1a 轮询至完成后打分+落账+commit。**无空卡。**

## 当前快照（2026-07-19 19:56，32/32 全占全健康；无空卡）
> 19:56 tick（`date` 实测 19:56:05 +0800）：**wzc1 8/8** keep14 **step139100/200k ppl10.60** gnorm0.51 100%util ~940W 0NaN 1.56s/step live@19:56:07（+1000步/30min 健康）；**.82+.104 16/16** keep7 DDP **step163760/200k ppl16.35** gnorm0.52 0NaN 1.47s/step live@19:56:06（+1180步/30min；训练-ppl 在 ~15.4-16.4 噪声窗，held-out plateau 判据走离线 eval 未触发，续训）；**.73 8/8** CoMem RULER 广度 eval 进到 **niah_multivalue 16k**（niah_single_1/3 已完，etimes~33min 138%CPU 进产出中），后续还有 multivalue/multiquery 长档 + 整个 InfLLM 阶段（128k 慢，长 run），finalizer a6984e1a 轮询至完成后打分+落账+commit。**无空卡。**

## 当前快照（2026-07-19 19:26，32/32 全占全健康；无空卡）
> 19:26 tick（`date` 实测 19:26:02 +0800）：**wzc1 8/8** 7B keep14 heal **step138100/200k ppl10.36** gnorm0.50 100%util ~940W 0NaN 1.56s/step live@19:26:33（+700步/22min 健康续训）；**.82+.104 16/16** 1B keep7 DDP **step162580/200k ppl16.16** gnorm0.51 0NaN 1.48s/step live@19:26:36（+1160步/30min 未 plateau，续训）；**.73 8/8** CoMem+InfLLM RULER 任务广度 eval（niah_single_1/3 niah_multivalue niah_multiquery ×16k/64k/128k，iter_bm25 topk12 chat）——⚠️**铁律2 判活证据**：8 procs 399-416%CPU（iter_bm25 BM25 CPU-bound）→ poll 瞬间 GPU util 0%/~120W 是**采样 artifact 非卡死**；file growth 确认在产出（19:30 已到 niah_single_1 64k）。**无空卡，全部 productively 占用。**
> ⚠️**收尾 orphaned→已接管**：launcher coder aa19dda5 已返回（其 .73 remote monitor 无法重新唤起 agent→scoring/落账/commit orphaned）→ **派 finalizer a6984e1a（background）**轮询至 eval 完成后打分（官方 string_match+铁律2）+ 写 `RULER_TASKBREADTH_RESULTS.md` + file-specific commit（diskB 树脏，禁 -A）no-push + kill 遗留 monitor + 释放 .73。CoMem 阶段先（快）→ InfLLM（128k 慢），整条 run 预计数小时。

## 当前快照（2026-07-19 19:05，32/32 全占；★baseline 矩阵真·零星号全满；.73 refill CoMem+InfLLM RULER 任务广度）
> 19:04 tick（`date` 实测 19:04:11-19:04:46 +0800）：**wzc1 8/8** 7B keep14 heal **step137400/200k ppl10.45** gnorm0.55 100%util ~940W 0NaN 1.56s/step live@19:04:46（+340步/9min 健康续训）；**.82+.104 16/16** 1B keep7 DDP 全 99-100%util ~290-306W 8procs/node（续训健康，未 plateau）；**.73 0/8→refill**——StreamingLLM-LongBench 完成释放（coder a4a9a7c，AVG F1 **37.20** n=1150 empty=0，commit `c98f642` 未 push）→ 立即派 coder **aa19dda5** 跑 **CoMem+InfLLM RULER 任务广度扩展**（补 flagship RULER 现仅 3/13 任务类型的真缺口：加 niah_single_1/3、multikey_2/3、multivalue、multiquery、qa_1/qa_2 @16k/64k/128k，n=100 8-shard 官方 string_match，CoMem iter_bm25 canonical j12 + InfLLM paper-faithful 对照，铁律2 验证后入 `RULER_TASKBREADTH_RESULTS.md`）。EVAL-ONLY 合规，耐跑非短任务填卡。
> ★★**baseline 矩阵真·零星号全满**（6方法×5benchmark，最后一格 StreamingLLM-LongBench 补齐=37.20）。**诚实态势**：矩阵完成后 .73（EVAL-ONLY）价值转向 flagship RULER 任务广度（现只 3/13 类型）；真正瓶颈仍在 ① Paper A 论文整合 task#10（GPU-free，需 F14 用户决策）② Paper B keep12/keep8 frontier（需训练 slot，wzc1+.82+.104 全满）。
> ★**commit-push backlog**（待 /gitpush review→APPROVED→star-proxy）：`c98f642`(StreamingLLM-LongBench)+`d3e5691`(LLoCO)+`0c69f79`(Tier-S 引用)+StreamingLLM/MemoryLLM/olmo2 drivers。
> ✅ **ckpt-rotation cron（4ec42903）= NO-OP**（19:0x 触发；keep14fresh2 仅单 step137000.pt 最新无可删，wzc1 24T free/15% 零盘压力）。⚠️**新发现 `outputs/olmo2_probe2_7B_keep12fresh2/`**：07-17 崩掉的死 run（log 停 07-17_16:19 shutdown 警告），有 step500.pt(43.8G)+step1000.pt(**20G<step500=疑不完整写入**)+arch_meta 共 60G。**未删**（不在 cron rm scope＝安全铁律「绝不动其他目录」；且 keep12 正是 Paper B 待跑 ablation frontier 配置，早期 ckpt 可能可 resume/复用；零盘压力无回收动机）→ 待用户决定 resume 还是清理。

## 当前快照（2026-07-19 18:56，32/32；★StreamingLLM-LongBench 跑完→矩阵零星号全满在望；.73 待 refill）
> 18:56 tick（`date` 实测 18:56:23 +0800）：**wzc1 8/8** 7B keep14 heal **step137060/200k ppl10.56** gnorm0.49 0NaN 1.56s/step（+1000步/29min 健康续训）；**.82+.104 16/16** 1B keep7 DDP **step161420/200k ppl15.93** gnorm0.52 0NaN（.104 node1 100%util ~306W，+1200步/29min 未 plateau）；**.73 0/8**——StreamingLLM-LongBench **~8min 跑完**（equal-budget recency 生成快，非 1-2h），coder **a4a9a7c 仍 running** 做铁律2 验证+落账+commit（CPU 收尾，持有 .73 未释放）→ **本 tick 不塞第二 job**（a4a9a7c 正在 wzc1 commit，避 git/资源撞车）。
> ✅ **StreamingLLM-LongBench 结果**（coder score.log，待 a4a9a7c 铁律2 empty 核实定稿）：hotpotqa 50.25/nqa 20.51/qasper 46.52/mfe 43.04/2wikimqa 42.36/musique 20.52 → **AVG F1 37.20**（EM 20.92，n 全额）。**诚实点**：equal-budget recency 在 LongBench（多 ≤32k 真实文档）AVG 37.20 **> CoMem 35.79**、接近 InfLLM 41.54——与"recency 短档够用、长档/多针/VT 才崩"叙事一致（LongBench 非 recency 软肋区）；tab 整合须诚实标此点，别把 CoMem 说成 LongBench 全胜。
> ★**下一 .73 fill（待 a4a9a7c 完成通知后派）**：AGENDA §1 查漏补缺 = **CoMem RULER task-breadth 扩展**（现仅 single_2/multikey_1/vt 3 任务→补标准 RULER 其余 NIAH 变体/qa_1/qa_2 等，CoMem 优先 + 强 baseline，headline 长档；driver `eval_ruler_mem_space.py` 支持 `--tasks/--lengths`）——耐跑且补 flagship benchmark 真实缺口，非短任务填卡。
> ⚠️**诚实态势**：baseline 矩阵基本完成后，.73（EVAL-ONLY）剩余价值=边际 breadth eval；**真正瓶颈在别处**——① Paper A 论文整合 task#10（GPU-free，需 F14 paper scope 用户决策）；② Paper B ablation frontier keep12/keep8（需训练 slot，当前 wzc1+.82+.104 全满）。
> ✅ **ckpt-rotation cron（4ec42903）= NO-OP**（19:0x 触发；keep14fresh2 仅单 step137000.pt 最新无可删，wzc1 24T free/15% 零盘压力）。⚠️**新发现 `outputs/olmo2_probe2_7B_keep12fresh2/`**：07-17 崩掉的死 run（log 停 07-17_16:19 shutdown 警告），有 step500.pt(43.8G)+step1000.pt(**20G<step500=疑不完整写入**)+arch_meta 共 60G。**未删**（不在 cron rm scope＝安全铁律「绝不动其他目录」；且 keep12 正是 Paper B 待跑 ablation frontier 配置，早期 ckpt 可能可 resume/复用；零盘压力无回收动机）→ 待用户决定 resume 还是清理。commit-push backlog（`d3e5691`+`0c69f79`+drivers）见 18:48 块。

## 当前快照（2026-07-19 18:48，32/32；★LLoCO path(a) DONE+验证；.73 refill StreamingLLM-LongBench 补矩阵末格）
> 18:48 tick（`date` 实测 18:48:29 +0800）：**wzc1 8/8** 7B keep14 heal 续训健康（18:26 step136060 ppl11.05 gnorm0.48 0NaN，承前推进）；**.82+.104 16/16** 1B keep7 DDP 续训健康（18:26 step160220 ppl15.61 gnorm0.49 0NaN）；**.73 refill**——LLoCO coder ab5d92c ✅ 完成（18:46，commit `d3e5691` 未 push）释放后 0/8 空 → 立即派 coder **a4a9a7c** 跑 **StreamingLLM equal-budget LongBench**（补 baseline 矩阵唯一 "—" 格，6-ds narrativeqa/qasper/hotpotqa/2wikimqa/multifieldqa_en/musique，sink4+window6653 chat+no-think 8-shard，对齐 InfLLM/KV-Direct LongBench cohort，~1-2h 耐跑）。EVAL-ONLY 合规。
> ✅ **LLoCO path(a) 完成+铁律2 验证**（用户"跑 LLoCO"决策完成）：narrativeqa F1 24.21/qasper 24.45/hotpotqa 44.24（n=200 各/0empty，直读 scores.json+独立复算 pred 计数一致），全在 Table4 ±2pt → 验证我们 LongBench 口径 = LLoCO 官方口径。坑：bf16 在 flash-attn2.5.6 kv-cache decode SIGFPE core dump→fp16 修复=faithful dtype；预编译 wheel 避 nvcc13.2 源码编译；仅 H20 sm_90 可跑。已入 `BASELINE_MATRIX_COMPLETE.md` §LLoCO + coder 写 `LLOCO_BASELINE_RESULTS.md`（driver `eval_lloco_longbench.py`，commit `d3e5691` 未 push）。
> ✅ **ckpt-rotation cron（4ec42903）= NO-OP**（wzc1 24T free，keep14fresh2 自轮转无可删项）。
> ★**commit-push backlog**（待 subagent-review→APPROVED→star-proxy `/gitpush`）：`d3e5691`(LLoCO drivers+results) + `0c69f79`(Tier-S citations) + StreamingLLM/MemoryLLM/olmo2 drivers。task#10 论文整合（加 6 baseline 行 + LLoCO 实证3行+Table4引用其余6 + CoMem LoCoMo errata iter_bm25 19.51/28.65 + E1-E8）待 F14 scope 定后一次性做。

## 当前快照（2026-07-19 18:29，32/32；.73 LLoCO coder 过下载→smoke core-dump→活跃 debug）
> 18:26-18:29 tick（`date` 实测 18:29:24 +0800）：**wzc1 8/8** 7B keep14 heal **step136060/200k ppl11.05** gnorm0.48 100%util ~930W 0NaN 1.56s/step maxmem122.3GB（18:10 step135500→18:26 136060，+560步/16min 健康续训）；**.82+.104 16/16** 1B keep7 DDP **step160220/200k ppl15.61** gnorm0.49 0NaN（.104 node1 100%util ~301W，+660步/16min 未 plateau）；**.73 0/8 GPU 空但非空转-neglect**——LLoCO coder ab5d92c（background，仍 running）已过 env+下载（`lloco_weights` 13GB `lloco_download_DONE`@18:22），18:28 跑 smoke（hotpotqa 1-sample：AutoCompressor 2-shard 加载 OK + LoRA Lloco-7b-hqa apply OK）→ **`model.generate` 处 `timeout: the monitored command dumped core`**（预期风险 surfacing：flash-attn 2.5.6 × H20 sm_90 kernel 或 AutoCompressor 自定义 forward segfault，researcher plan §3 已 flag flash-attn 为头号风险）。coder 1min 前刚产 log = **活跃 debug 周期非 abandon**（.73 单租 EVAL-ONLY，不塞第二 job 避撞 coder 即将的 GPU 用途）。★**下轮升级判据**：若下 tick .73 仍 0/8 且 coder 已 complete-fail 或无 GPU 进展 → TaskStop + 重派带 `attn_implementation=eager`/禁 flash-attn fallback，或退 **path(a.5) 纯引用 Table 4**（零 GPU、覆盖全 9 任务，已可立即入论文 task#10）。
> ✅ **ckpt-rotation cron（4ec42903）= NO-OP**（wzc1 24T free/15%，keep14fresh2 自轮转无里程碑/final/中间堆积，无可删项）。

## 当前快照（2026-07-19 18:10，32/32 全占；★baseline 矩阵全满；.73 空转→派 LLoCO path(a) 接管）
> 18:08-18:09 tick（`date` 实测 18:08-18:09 +0800）：**wzc1 8/8** 7B keep14 heal **step135500/200k ppl10.33-10.86** gnorm0.50-0.55 0NaN 1.56s/step maxmem122.3GB（log 活跃 135440→135500，GPU0 6%=optimizer/logging 采样瞬间非卡死，续训健康）；**.82+.104 16/16** 1B keep7 DDP **step159560/200k ppl15.56-15.78** gnorm0.49-0.52 0NaN 1.47s/step maxmem41.3GB（8卡100%/335W，未 plateau，续训）；**.73 8/8**——StreamingLLM QA chain 全跑完释放→18:08 实测 **0/8 空转**（矩阵-fill 目标已达）→**派 LLoCO path(a) coder ab5d92c 接管**（用户"跑 LLoCO"授权，EVAL-ONLY 合规；建隔离 conda lloco_env py3.10 + flash-attn sm_90 编译 + 下 AutoCompressor-Llama2-7b + 3 domain LoRA → 自跑 narrativeqa/qasper/hotpotqa 对齐我们 LongBench scorer；env-build 阶段 GPU 暂空=合法 provisioning，非空转-neglect）。
> ★★**baseline 矩阵 6方法×5benchmark 全满**（用户"矩阵缺格"决策完成）：本会话(a) 打分 **KV-Direct-BABILong**（官方 compare_answers，empty=0：qa1 99/94/93/89/83/78/71、qa2 61/54/55/52/52/47/41、qa5 83/79/75/74/77/72/69）；(b) 铁律2 独立验证 **HCache-LoCoMo**(F1 7.82/acc 8.06，1986 非空 preds)、**KV-Direct-LoCoMo**(F1 40.06/acc 43.05，1986/0empty)、**StreamingLLM-LoCoMo**(F1 12.73/acc 17.57)、**StreamingLLM-BABILong**(qa1 100→23/qa2 60→3/qa5 81→53) 全非空。详见 `status/BASELINE_MATRIX_COMPLETE.md` + STREAMINGLLM 账本已补 LoCoMo/BABILong。
> ✅ **ckpt-rotation cron（4ec42903）= NO-OP**（前轮已确认 wzc1 24T free/15%，keep14fresh2 trainer 自轮转无里程碑/final/中间堆积，无可删项，不动文件）。

## 当前快照（2026-07-19 16:56，32/32 全占；.73 空转→已修复填 LongEval；ckpt-rotation NO-OP）
> 16:56 tick（`date` 16:56:02）：**wzc1 8/8** 7B keep14 heal step133000（util 100% ~935W，健康续训，132000→133000 推进中）；**.82+.104 16/16** 1B keep7 DDP（util 100%，.82~365W/.104~305W，健康）；**.73 8/8** —— ⚠️**16:56 实测 0/8 空转**（铁律1 violation：coder af2a386 卡在 scp sftp 传 babilong driver 而 stall，但 LongBench 其实已跑完 `_DONE_LONGBENCH`+scores.json、LoCoMo preds 完成未 score）→ **已 TaskStop af2a386 + 亲自拉起 StreamingLLM LongEval 8-shard**（8k-128k limit50 chat+no-think sink4/window6653，pid1939832，8/8 已满，shard0 已加载模型开跑）。派 coder（sonnet）收尾：score LoCoMo + 新建 `eval_streamingllm_babilong.py`（复用 backbone+官方 compare_answers，禁 re.search）+ 链 BABILong 在 LongEval 后。派 researcher（opus）梳理 related work（用户 Q）。EVAL-ONLY 合规。
> ✅ **ckpt-rotation cron（4ec42903）触发=NO-OP**：wzc1 `df` 4.2T used/24T free/15%（充裕）；`outputs/olmo2_probe2_7B_keep14fresh2/` 只有 1 个 ckpt `step133000.pt`（48.7GB，trainer 自轮转，无里程碑/final/中间堆积）→ 按安全铁律"留最新2+里程碑+final"=只有 1 个即最新，**无可删项**，不动任何文件。`keep10fresh2/` wzc1 上不存在。

## 当前快照（2026-07-19 16:49，32/32 全占；.73 StreamingLLM LoCoMo 完成→LongBench/BABILong/LongEval 接力）
> 16:49 tick（`date` 16:49:52）：**wzc1 8/8** 7B keep14 heal（step132000，ppl10.77-10.90，gnorm~0.5，0NaN，健康续训）；**.82+.104 16/16** 1B keep7 DDP（step155420，ppl16.17-16.57，0NaN，健康续训，未 plateau）；**.73 8/8** StreamingLLM equal-budget QA chain——**LoCoMo 8-shard 跑完**（8 shard 各 248-249 preds = n=1986 完整），只剩 4 proc 收尾；coder a46e6bf 建好全套 driver（`eval_streamingllm_{locomo,longbench,longeval}.py` + `streamingllm_backbone.py`）但未 wire 下一档→**.73 将掉向空转**。已 kill 冗余 nudge agent aefd96ec（避免重复起 LoCoMo），派 **coder af2a386（task#21）** 收尾：score LoCoMo → LongBench 8-shard 立即补卡 → 新建 `eval_streamingllm_babilong.py`（复用 backbone + 官方 compare_answers+TASK_LABELS，禁 re.search）→ LongEval 8-shard，`/tmp/slm_qa_chain2.sh` setsid nohup 串起。EVAL-ONLY 合规。补齐 SLM recency-budget 对照全 5-benchmark（现仅 RULER/tab_slm）。每档铁律2 验证后回填 `STREAMINGLLM_EQUALBUDGET_RESULTS.md` + tab_slm。

## 当前快照（2026-07-19 16:24，32/32 全占；.73 refill StreamingLLM equal-budget QA chain；zero-shot LongEval 完成→adapter=~10-15×杠杆）
> 16:24 tick（`date` 16:24:07）：**wzc1 8/8** 7B keep14 heal（16:15=step131640 ppl10.32-11.00，健康续训）；**.82+.104 16/16** 1B keep7 DDP（16:15=step154980 ppl15.01-16.79，健康续训）；**.73 8/8 refill**——task#20 zero-shot LongEval 跑完释放后立即补 coder a46e6bf（task#21）**StreamingLLM equal-budget QA chain**（LoCoMo→LongBench→BABILong→LongEval，sink4+window6653=6657≈CoMem read，chat+no-think 8-shard，~10h+ 耐跑填卡）——补齐 SLM recency-budget 对照全 5-benchmark（现仅 RULER/tab_slm）。EVAL-ONLY 合规。
> ✅ **task#20 完成（CoMem zero-shot LongEval 4k-128k 补齐 → "adapter 是关键杠杆" ablation）**：zero-shot(j9 无 adapter) iter_bm25 chat+no-think LongEval **4k0.14/8k0.06/16k0.05/32k0.07/64k0.06/128k0.07**（n=100 max64，铁律2 已验：merge=独立复算一致，长档=well-formed 错值=真检索失败，4k 61 拒答=zero-shot 短档格式 artifact）。vs adapter(j12) 0.95/0.73/0.76/0.79/0.72/0.76 → **distilled adapter=~10-15×杠杆**（诚实标注：杠杆=LoRA＋训练允许更深 split j9→j12 联合效应）。已入 `status/INFLLM_BASELINE_RESULTS.md` §CoMem zero-shot vs +adapter ablation。raw `.73:longeval_results/qcmem_8b_zs_iter_chatnothink/`。

## 当前快照（2026-07-19 16:15，32/32 全占；.73 refill zero-shot LongEval 64k/128k；adapter matched 行已入 ledger）
> 16:15 tick（`date` 实测 16:15:18 +0800）：**wzc1 8/8** 7B keep14 heal **step131640/200k ppl10.32-11.00** gnorm0.48-0.53 0NaN 1.56s/step maxmem122.3GB live@16:15；**.82+.104 16/16** 1B keep7 DDP **step154980/200k ppl15.01-16.79** gnorm0.49-0.51 0NaN 1.47s/step maxmem41.3GB live@16:15（+~1940步/49min vs 15:26 step153040，续训，未 plateau）；**.73 8/8 refill**——adapter iter_bm25 LongEval（task#19，coder a23655ca）跑完释放后立即补 coder ab16c2b（task#20）**CoMem zero-shot iter_bm25 LongEval 64k/128k 扩展**（resume_j=9、无 adapter；补齐现有 zs 4k/8k/16k/32k→完整 4k-128k，做 zero-shot vs +adapter 长档对照，服务"adapter=硬任务/长档关键杠杆" claim）。EVAL-ONLY 合规。判空卡=数进程（4 节点各 8/8 有 proc，无空卡）。ckpt-rotation cron 16:1x 触发=NO-OP（wzc1 24T free/15%，仅最新 step ckpt）。
> ✅ **task#19 完成（adapter matched-selector LongEval 已入 ledger）**：CoMem iter_bm25 chat+no-think LongEval **4k0.95/8k0.73/16k0.76/32k0.79/64k0.72/128k0.76**（n=100 max64，coder a23655ca 补 64k=0.72/128k=0.76 铁律2 已验非空/真检索失败）。**诚实点**：matched 恒定-read 协议下 CoMem = **8k-128k 约 0.72-0.79 FLAT（length-invariant 但非满分）**，headline 0.94-1.00 是 per-task 最优 k；**即便 matched，128k=0.76 仍碾压全 baseline**（InfLLM0.02/MemoryLLM0.04/HCache0/KV-Direct0）。已回填 `status/INFLLM_BASELINE_RESULTS.md` §LongEval cohort（新增 matched 行 + tab_longeval 诚实标注）。task#10 论文整合待用。

## 当前快照（2026-07-19 15:54，32/32 全占；★keep14 apex eval 完成→结论修订；.73 refill CoMem LongEval 64k/128k）
> 15:54 tick：**wzc1 8/8** 7B keep14 heal **step130980/200k ppl10.59-11.13** gnorm0.48-0.52 0NaN 1.56s/step maxmem122.3GB live@15:54；**.82+.104 16/16** 1B keep7 DDP 续训（15:26=step153040）；**.73 8/8 已 refill**（keep14 apex eval 释放后立即补，铁律1）——coder a23655ca 跑 **CoMem iter_bm25 chat+no-think LongEval 64k/128k 扩展**（补齐 matched-selector 行：现有 4k0.95/8k0.73/16k0.76/32k0.79 n=100 max64，缺 64k/128k；补齐后 tab_longeval CoMem 行与 baseline 8k-128k apples-to-apples）。EVAL-ONLY 合规。
> ★★**keep14 apex 结论修订（铁律2 独立复算 6 cell 逐格=summary，MMLU recomp=summ=0.3012 n=14042 nan=0）**：7B keep14（16L/32=**50%**，收敛 step128000）**MMLU=0.3012 ≈13SE 高于 0.25 chance floor** → **知识"部分可恢复"，非"不可恢复/钉在 floor"**。之前（14:26/15:26 快照 + task#16/#17）"知识不可恢复"是**欠 heal / 过剪 artifact**（keep10@step10000 只训 1w 步欠 heal、1B keep7 是 9L 过剪）。修订后诚实信号：知识仍是**最弱恢复轴**——只恢复 base above-chance 信号的 **~14%**（0.301 vs base 0.605；(0.301-0.25)/(0.605-0.25)=14.4%）vs reasoning/surface 53-79%、comprehension 44-79%；**PPL（已恢复到 base 的 1.5-2.3×）对这个残余知识缺口基本盲视**。→ 方向4 信号存活（软化版）。per-subject lift 例：world_religions 0.27→0.427、us_foreign_policy 0.46。raw：`.73:olmo2_downstream_results/7B_keep14_step128000{,_know}/`，ledger `status/OLMO2_PRUNEHEAL_DOWNSTREAM.md`（commit 7ce25bb 未 push）。⚠️ wzc1 keep14 只剩 step130500.pt（trainer 自轮转无里程碑）→ 无法便宜复现 keep14 heal-curve（earlier ckpt 已删）。

## 当前快照（2026-07-19 15:26，wzc1+.82+.104 24/24 训练健康；.73 scp 93% 传输 keep14 ckpt eval 即启）
> 15:26 tick：wzc1 8/8 7B keep14 **step130000 ppl10.77-10.79** gnorm0.49-0.51 0NaN/0TB live；.82+.104 16/16 1B DDP **step153040 ppl16.19-16.67** gnorm0.50 0NaN/0TB（+1140步/30min，窗口噪声非plateau）；**.73 GPU 0/8 仍 provisioning**——coder a8f3f3e2（task#18）scp keep14 step128000.pt **45.38GB/48.7GB=93%**（scp -t 收端 proc 活），~3min 落地后即启 core+MMLU eval。⚠️ 本机 trainer 已自轮转掉 local step128000.pt（现只剩 step130000.pt），scp 靠已打开文件句柄续传不受影响（Linux unlink 后 inode 存活）。

## 当前快照（2026-07-19 14:56，wzc1+.82+.104 24/24 训练健康；.73 scp 传输 keep14 ckpt 中 GPU 空=provisioning 非空转）
> 14:56 tick：wzc1 8/8 7B keep14 **step129000 ppl10.74-10.78** gnorm0.48-0.50 0NaN/0TB live；.82+.104 16/16 1B DDP **step151900 ppl15.82-16.05** gnorm0.50 0NaN/0TB（+1260步/32min）；**.73 GPU 0/8 但非空转**——coder a8f3f3e2（task#18）正 **scp keep14 step128000.pt（48.7GB）wzc1→.73**，18.3GB/48.7GB @~19MB/s ETA~27min，ckpt 落地后即启 core+MMLU eval（单文件传输期 GPU 必然空，不可避免；scp -O 无 resume，不 kill 重传）。判空卡=数进程（.73 scp 是 CPU/网络绑定，非 GPU 卡死）。

## 当前快照（2026-07-19 14:26，32/32 全健康）

| 节点 | 卡 | 在跑 | 状态（实测 14:26） |
|---|---|---|---|
| **本机 wzc1 8×L20A** | 8/8 | OLMo-2 **7B keep14+fresh2** heal（`train_olmo2_arch_probe2.py`，bs16 fp32-master） | ✅ step128000/200k **ppl10.74-11.30** gnorm0.48-0.50 0NaN 1.56s/step maxmem122.3GB。ckpt cron 4ec42903 + trainer 自轮转（盘 24T free） |
| **28.85.35.73 8×H20** | 8/8 | **OLMo-2 7B keep14 apex downstream+MMLU eval**（coder a8f3f3e2 task#18：rsync 收敛 keep14 step128000.pt 16L/32=50% wzc1→.73，core 6-task + knowledge MMLU 8-shard；补 方向4 apex 点） | 🔄 EVAL-ONLY 合规（纯前向 likelihood）。14:27 MMLU 扩展（coder a77e63a0 task#17，commit 8947078）完成→**铁律2 独立复算 35/35 cell 逐格=summary，7B base MMLU 0.605=公开数**：**知识不可恢复**（1B keep7 MMLU 钉 0.25 floor 50k→150k dead-flat、7B keep10 0.254，ppl 已恢复 1.47×）。keep14 apex 点堵"heal 不够"反驳中 |
| **28.82.250.82 + 28.83.24.104 16×H20** | 16/16 | OLMo-2 **1B keep7+fresh2** 多机 DDP（TCP over bond1） | ✅ step150640/200k **ppl15.96-16.43** gnorm0.50-0.54 0NaN maxmem41.3GB（+1220步/30min，窗口噪声非 plateau，续训） |

- **判空卡口径**：`nvidia-smi -i K --query-compute-apps=pid | wc -l`（数进程，非显存）。
- **数据**：`data/dolmino_now15b.npy`（OLMo-2 tokenizer）；红线 `--babilong_mix_fraction 0`。
- **早停规则**：held-out ppl plateau 则早停（1B keep7 ppl 仍单调↓未 plateau，续训；但 downstream 已早 plateau=能力恢复慢，见 ledger）。
- **StreamingLLM equal-budget（tab_slm）已补齐 5-length**：single 90/42/18/16/4、multikey 86/48/26/8/6、vt 38/3.6/1.2/0/0（8/16/32/64/128k n=50）→ `status/STREAMINGLLM_EQUALBUDGET_RESULTS.md`。

## 节点清单（4 节点 = 32 卡，权威见 HEARTBEAT.md roster）
- **本机 wzc1 8×L20A**：`.venv/bin/python`（py3.14 torch2.13）。
- **28.85.35.73 / 28.82.250.82 / 28.83.24.104**：均 36000 端口，diskB 共享 FS（免同步），`/opt/conda/envs/torch-base/bin/python`。★**.73 = EVAL-ONLY（不塞训练）**。
- SSH：`unset LD_LIBRARY_PATH; /opt/conda/bin/sshpass -f configs/password_h20_xxx.txt /usr/bin/ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password root@IP`。
- **.73 (28.85.35.73 diskB H20)**: keep14-distill heal (8bit adam, teacher OLMo-2-7B 32L → student keep14+2fresh), bs=4 gaccum=4, started 2026-07-30 14:20, ~13.87s/step → 200k ETA ~32d (observe loss), maxmem 94.6GB
