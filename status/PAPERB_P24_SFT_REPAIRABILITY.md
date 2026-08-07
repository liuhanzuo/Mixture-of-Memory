# Paper B P2.4 — 通用 SFT 接口可修复性诊断（task #123）

**状态：`[PARTIAL — 数据段 DONE + 1/3 臂 SFT 训练中；裁决未产生]`**
**日期**：2026-08-07 | **节点**：`.73`（8×H20，zwfy6 盘）| **commit**：`5db5d30`（fix `0fd051a`）
**执笔**：experiment agent。⚠️ 本文件只报已实测的数字；**四路裁决（four-way verdict）本轮不触发**，因为 post-SFT eval 未跑完。

---

## 0. 一句话结论

**SFT 数据已构建并通过污染审计（干净，可用）**，但**P2.4 的裁决本轮无法给出**：
(a) 派单所述「`.73` 空闲」不成立，GPU 被他人任务反复抢占；
(b) **full-32L 臂在 H20 上物理不可行**（fp32 DDP 需 108.8 GiB/卡 > 97.8 GiB）；
(c) 修复了两个真 bug（OOM、**loss=nan**）后仅启动了 1/3 臂。
**任何"SFT 修好了 MMLU"的说法目前都没有实测支持——不要写进论文。**

---

## 1. GPU 可用性：派单前提实测不成立

| 时间(CST) | `.73` 实测 | 说明 |
|---|---|---|
| 21:08:10 | 他人任务启动 | `dllm_draft_104/scripts/generate_evalplus_dream.py --dataset humaneval`，8 卡 |
| 21:10 | **8/8 卡 @99%** | 我的首次检查。派单前 ~2.5 分钟被占走 |
| 21:10 | `.82`(20:45 起)/`.104`(20:47 起) | 同一 EvalPlus 审计 → **zwfy6 三台 24 卡 0 空闲** |
| 22:23 | 短暂全空 | 20 秒后即被**另一容器**重占（PID 不在我 namespace，`/proc/<pid>` 不可读）|
| 22:36 | ALL_FREE | 抢到窗口，启动 SFT |
| 22:37 | 我的 run OOM 崩 + 8 卡被他人重占 | 见 §4 |
| 22:41 | ALL_FREE | 再抢窗口 |
| 22:53– | **我的 SFT 稳定运行中** | 96.2 GiB/卡 × 8 |

**未 kill 任何他人任务。** `.82`/`.104`/`.252`/LOCAL 全程未碰。
`.73` 是**共享 GPU 节点**（他人容器 PID 对我不可见）→ 后续 P2.4 排程必须考虑抢占，或换独占节点。

---

## 2. ⚠️ full-32L 臂在 H20 不可行（硬约束，非调度问题）

`scripts/train_olmo2_sft.py` = **plain DDP + fp32 master weights + fp32 AdamW**（`grep -c bitsandbytes` = **0**，无 8-bit 路径）。
plain DDP **只 all-reduce 梯度，不 shard** param/grad/optimizer state → **加卡不降每卡内存**。

| Arm | params（实测 dry-build） | fp32 AdamW 静态 = 16 B/param | H20 97.8 GiB | B200 183 GiB |
|---|---:|---:|:--:|:--:|
| **full 32L** | 7.298 B | **108.8 GiB** | ❌ 超 11 GiB | ✅ |
| keep14+fresh2 (16L) | **4,060,352,512** | 60.5 GiB | ✅ | ✅ |
| ShortGPT-16 (16L) | **4,060,352,512** | 60.5 GiB | ✅ | ✅ |

参数量交叉验证：4,060,352,512 × 4 B = 16.24 GB = **恰好等于** `keep14fresh2/step200000.pt` 的 16,241,486,089 B。

**结论**：full-32L 臂**必须去 B200**（LOCAL/.252，183 GiB），或先给 trainer 加 bnb8bit（`.73` 有 bnb 0.50.0，8-bit 可降至 68.0 GiB 可过 H20）/FSDP。
⚠️ 但 **换 optimizer 会破坏"三臂 byte-identical optimizer"** 的单变量纪律 → **建议 full-32L 与两个剪层臂一起放 B200 跑，而不是混用两种 optimizer**。

---

## 3. SFT 数据 + 污染审计（DONE，干净）

### 3.1 数据源
- **数据集**：`allenai/tulu-3-sft-mixture`（streaming，seed 42，shuffle_buffer 50000），tokenizer = OLMo-2-1124-7B。
- **产物**：`data/olmo2_sft/tulu3_general_{input_ids,labels,text.jsonl,manifest.json}`（zwfy6 `.73`）。
- **规模**：**249,999,360 tok**（122,070 × 2048），**161,118,687 supervised**，用 234,483 conversations，**denied 46,013**。
- **deny-filter 生效**：19 个 pattern（flan/mmlu/arc/triviaqa/popqa/natural_questions/sciq/hendrycks/openbookqa/hellaswag/commonsense/squad/coqa/drop_/exam/…）。

存活 9 个 source（**无学科多选、无 closed-book QA**）：

| conversations | source |
|---:|---|
| 46,774 | evol_codealpaca_heval_decontaminated |
| 46,145 | numinamath_tir_math_decontaminated |
| 45,639 | tulu_v3.9_wildchat_100k |
| 42,766 | personahub_math_v5_regen_149960 |
| 39,351 | tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k |
| 7,109 | oasst1_converted |
| 5,262 | tulu-3-sft-personas-math-grade |
| 768 | tulu_v3.9_personahub_math_interm_algebra_20k |
| 669 | tulu_v3.9_table_gpt_5k |

### 3.2 n-gram overlap 审计（8-gram，threshold 0.5，全量 234,483 records，pooled vocab **74,819,130**）

| Eval | n | hit@0.5 | hit rate | exact substring | exact rate | mean containment |
|---|---:|---:|---:|---:|---:|---:|
| **MMLU** | 14,042 | 63 | 0.45% | **45** | **0.32%** | 0.00728 |
| **PopQA** | 14,267 | **0** | **0.00%** | **0** | **0.00%** | 0.000012 |
| **TriviaQA** | 17,944 | 8 | 0.045% | **0** | **0.00%** | 0.00070 |

**dedup**：丢弃与任一 eval 题 ≥0.5 containment 的 record = **76 / 234,483 = 0.0324%** → clean scale 234,407。

**MMLU 45 条 exact match 的构成**（按 subject）：`high_school_mathematics` **36**、`high_school_statistics` 3、`college_chemistry` 2、`nutrition` 2、`international_law` 1、`logical_fallacies` 1。其中 11 条是 ≤10 词的**退化样板句**（"Inflation"、"IBS"、"which one of the following statements is true"）。
→ 真实重叠 = **数学题**（来自 numinamath / personahub math），量级 0.32%，**已量化、可披露**。**PopQA / TriviaQA 零重叠**（closed-book 知识任务完全干净，这是本诊断最关键的一点）。

**方法学说明（诚实披露）**：仓库自带的 `audit_olmo2_sft_overlap.py` 的 exact-substring 段是 O(46k 题 × 234k records) 的 Python 双循环，实测 >1 h 未出结果（已 kill）。我用**数学等价的快路**替代：若题目 q 逐字出现在某 record 内，则 q 的**每个** 8-gram 都在该 record → containment ≡ 1.0；故 {exact} ⊆ {containment==1.0}，只需对该小子集做子串扫描。**两法 pooled vocab 完全一致（74,819,130），互为交叉验证。** 快路脚本 = `/tmp/fast_audit.py`（结果 `data/olmo2_sft/tulu3_general_fast_audit.json`）。

---

## 4. 修复的两个真 bug

### 4.1 OOM（BS=4）
`torch.OutOfMemoryError: Tried to allocate 3.06 GiB`。**3.06 GiB 恰好 = fp32 logits** `4×2048×100352×4 B` → 词表 100k 下 logits+xent 是峰值主因。
**修法**：`BS=4/GA=4` → **`BS=1/GA=16`**，`eff_batch = 1×16×8 = 128` **与原配置完全相同** → token budget 不变，**单变量纪律不破**。

### 4.2 ★ loss=nan（load-bearing，已修源码并 commit `0fd051a`）
首次成功启动后 **step 20 loss=nan**。根因：
`prepare_olmo2_sft_data.py` 按**拼接**打包、**跨 conversation 边界**切 2048 行，而 `any(mask)` 只在**每个 conversation** 层面检查 → 一个 2048-token 行可能**整行落在无监督区**（长 user turn 内），得到**全 -100 标签行**。
`F.cross_entropy(..., ignore_index=-100)` 对全忽略 target **返回 NaN**（已实测验证），**DDP all-reduce 把 NaN 扩散到每个 rank 的梯度 → 全部权重被污染**。
**实测规模：14,330 / 122,070 行 = 11.74%。**
**修法**：`_flush_full()` 丢弃零监督行 + manifest 记 `n_rows_dropped_zero_supervised`（commit `0fd051a`，已同步 zwfy6）。
清洗后数据：**107,740 行 / 220,651,520 tok / supervised 161,118,687（一个监督 token 都没丢）** → `tulu3_general_clean_{input_ids,labels}.npy`。
**修复生效证据**：relaunch 后 `step 20 loss=1.7675`（有限），step 100 loss=1.3268，持续下降。

> ⚠️ **对未来所有 arm 的影响**：清洗是确定性的、**对三臂完全相同**，故单变量纪律成立。但**任何用未清洗数据跑出的 SFT 结果都是 NaN 污染的，必须作废重跑。**

---

## 5. Arm 配置（config diff = 只有起始 checkpoint）

三臂共享（**byte-identical**）：`--sft_ids/--sft_labels`（同一 clean npy）、`--max_steps 842`、`--batch_size 1`、`--grad_accumulation_steps 16`（eff_batch 128）、`--seq_len 2048`、`--lr 1e-5`、`--min_lr 1e-6`、`--warmup_steps 100`、`--weight_decay 0.1`、`--seed 42`、`--gradient_checkpointing 1`、AdamW betas (0.9,0.95)、fp32 master + bf16 autocast。
**唯一差异 = `--ckpt` / `--arm_name`。**

| Arm | ckpt（zwfy6，均已实测存在） | 状态 |
|---|---|---|
| keep14+fresh2@200k | `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt`（16.2 GB，keep=14 fresh=2 step=200000）| **RUNNING**（22:53 起，842 步，9.6 s/step，ETA ~2.2 h）|
| ShortGPT-16@200k | `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt`（48.7 GB，keep=16 fresh=0 step=200000）| **待跑**（`.73` 排队）|
| full 32L base | `/apdcephfs_zwfy6/.../pighzliu_code/models/OLMo-2-1124-7B`（6 shard safetensors）| **BLOCKED → 需 B200**（§2）|
| keep14 equal-token NTP（compute control，spec 建议）| 同 keep14 ckpt + `--data_mode ntp` | **待跑**；语料 `dolmino_now15b.npy` 在 zwfy6 ✅ |

> ⚠️ **路径纠正**：`_run_olmo2_p24_sft_pipeline.sh` 硬编码 `final.pt`，但两个臂磁盘上**只有 `step200000.pt`**（无 `final.pt`）→ 直接跑 pipeline 会启动即失败。本轮已用显式 `step200000.pt` 绕过；**pipeline 该行仍需修**（未改，因 MAIN 拥有该文件的排程语义）。
> ⚠️ **base 权重位置**：在 `pighzliu_code/models/`（**不在** `Mixture-of-Memory/models/`）；`Mixture-of-Memory/models/` 下无 OLMo，`~/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B` 是**空壳 4.0K**。

---

## 6. Pre-SFT 基线（复用既有 DONE 任务，非本轮重跑）

来自 P0.6（双协议 MMLU，n_valid=14042/n_nan=0）与 P0.3/B-P0.0（closed-book），均 chat=False / no-BOS / LL-MC：

| Arm | PPL | MMLU letter | MMLU content_norm | PopQA contains | TriviaQA em |
|---|---:|---:|---:|---:|---:|
| full base 32L | — | .6054 | .4706 | .2571 | .6355 |
| keep14+fresh2@200k | 10.561 | .3184 | .3832 | .1415 | .2940 |
| ShortGPT-16@200k | 9.78 | .4742 | .4012 | .1585 | .3301 |
| （参照）random-init@200k | 11.498 | .2470 | .3598 | — | — |

**null（构造相称，必须用这两个而非 .2500）**：letter → best-constant-letter **always-D = .2689**（n=14042）；content → longest-option heuristic **.2845**（split-ties 约定；**4805/14042 = 34.2%** 的题有并列最长选项，故该约定 load-bearing 必须写明）。
**bf16 exact ties**：tie rate 随损伤上升（base .0013 vs keep14-reheal .2547），argmax 按 index 决胜 —— post-SFT eval 必须沿用同一 tie 约定并披露。
**关键 pre-SFT 事实**：ShortGPT-16 letter .4742 **已显著高于** keep14 .3184（16 层臂中最强），但两者 closed-book 均**远低于** base → 这是 P2.4 第三条裁决（"keep14 仍显著落后 ShortGPT"）的**起点**，SFT 后必须重测才能判定。

---

## 7. 四路裁决：**本轮不触发**

| 预登记裁决 | 本轮能否判定 |
|---|---|
| MMLU 升 **且** 独立知识任务同步升 → 支持可监督修复 | ❌ post-SFT eval 未跑 |
| **仅** MMLU 升 → 任务格式/多选接口适配 | ❌ 同上 |
| keep14 仍显著落后 ShortGPT → 结构保留比 SFT 更关键 | ❌ ShortGPT 臂未跑 |
| full base 获相似增益 → **不得**归因于压缩恢复 | ❌ full32 臂在 H20 不可行 |

**⇒ 不得在论文中写入任何 P2.4 结论。** 数据段与 harness 已就绪，裁决需下述剩余工作。

---

## 8. 剩余工作（给 MAIN）

1. **keep14fresh2 臂**（running）跑完 → 存 `outputs/olmo2_p24_sft_keep14fresh2/final.pt`。
2. **ShortGPT-16 臂**：同配置换 `--ckpt .../shortgpt16/step200000.pt`（`.73` 可跑）。
3. **full-32L 臂 + keep14 NTP compute control**：**去 B200**（LOCAL/.252）。
   ⚠️ 若为上 H20 而给 full32 换 bnb8bit，则三臂 optimizer 不再一致 → **建议三臂统一在 B200 跑**。
4. **post-SFT eval battery**（每臂）：held-out Dolmino PPL + **两种 MMLU protocol**（各自 null：.2689 / .2845）+ core6 + 逐任务 + PopQA + TriviaQA closed-book；保留逐题 prediction → wrong→right / right→wrong、McNemar（**用 log-space 版**，见下）、paired bootstrap CI。
   - harness：`eval_olmo2_probe2_ppl.py` / `eval_olmo2_mmlu_content.py`（双协议）/ `eval_olmo2_probe2_downstream.py` / `eval_olmo2_closedbook_qa.py` —— **不要新写 scorer**。
   - ⚠️ 已知坑：`mcnemar_exact_p` 的 `math.comb` 版在 full-set merge（n≈数千）会 `OverflowError`，须用 commit `324a44f` 的 log-space 版。
   - ⚠️ sharded eval 必须断言 shard 完整（静默 5/8 merge 会毁口径）。
5. **aggregate 只用 P0.7 固定口径**。

---

## 9. 产物路径（`.73` = `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`）

```
data/olmo2_sft/tulu3_general_{input_ids,labels}.npy        原始（含 11.74% NaN 行，勿用于训练）
data/olmo2_sft/tulu3_general_clean_{input_ids,labels}.npy  ★训练用（107,740 行）
data/olmo2_sft/tulu3_general_clean_manifest.json           清洗账
data/olmo2_sft/tulu3_general_manifest.json                 source histogram / deny 统计
data/olmo2_sft/tulu3_general_text.jsonl                    审计输入（757 MB）
data/olmo2_sft/tulu3_general_fast_audit.json               ★污染审计结果
data/olmo2_sft/p24smoke_*                                  2M-tok smoke（含 slow-audit 交叉验证）
logs/p24_sft_dataprep.log logs/p24_fast_audit.log logs/p24_dedup.log logs/p24_sft_keep14fresh2.log
outputs/olmo2_p24_sft_keep14fresh2/                        SFT 输出（running）
```
代码（wzc1 主仓，commit `0fd051a`）：`scripts/prepare_olmo2_sft_data.py`（本轮修复）+ 既有 `audit_olmo2_sft_overlap.py` / `train_olmo2_sft.py` / `eval_olmo2_closedbook_qa.py` / `_run_olmo2_p24_sft_pipeline.sh`（commit `d05ef59`）。
zwfy6 checkout 原为 `2d98c5a`（缺 P2.4 全部脚本）→ 已 `scp -O` + md5 校验同步。
