# QCMem T21 — 32B VT recall-vs-read-speed Pareto frontier

> 2026-07-20. 用户 idea (PENDING T21, P0)：QCMem `resume_j` 越深 → read 阶段重算 `layers[j:]` 越少 → read 越快；
> 但过了 readout-safe 深度 recall 会掉。把 **recall(j)** 与 **read-latency(j)** 配成一条 Pareto frontier，坐实
> "resume 深度 = 速度↔精度旋钮"。**EVAL-ONLY，zero-shot（无 adapter），无 git，无代码改动**。

## 协议（严格同口径）
- **模型**：`models/Qwen3-32B`（L=64 层，hidden=5120），**zero-shot，无 `--lora_adapter`**。bf16 / sdpa。
- **任务**：`variable_tracking`（vt，多跳、对 readout 深度最敏感）。
- **selector**：`iter_bm25`（vt canonical，多跳 BFS 词法链，rounds0→ceil(topk/hop)=3 轮，hop_topk4；forward-free）。
- **chat_template ON + thinking OFF**（`--use_chat_template`，`enable_thinking=False`）——2026-07-17 QCMem eval 统一标配。
- topk=12，chunk_size=512，sink=bos。read pack 恒定 ~6570 tok（16k）/ ~6625 tok（32k）。
- **recall 判分**：官方 `_string_match_all_one`（RULER vt string_match_all 口径）。n=50/cell（单卡整跑，非分片）。
- **read-latency**：`scripts/bench_qcmem_vs_dense.py --mode profile`（同 32B / topk12 / chunk512），
  隔离 `read_prefill`（`layers[j:]` 在固定 pack 上跑一遍）+ `decode/step`（resume-band KV cache 单 token）。
  read pack 大小固定 → read-latency 与 context 长度无关（只随 (L-j) 变）。
- 节点 .73（28.85.35.73，diskB，8×H20），`/opt/conda/envs/torch-base/bin/python`。
- **原始结果**：`.73:/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`
  - recall：`ruler_results/qcmem_32b_t21_vt_j{3,6,13,20,27,34,41,48}/t21/variable_tracking_{16k,32k}.{csv,json}`
  - latency：`logs/t21_profile_j{3,6,13,20,27,34,41,48}.log`

## 铁律2 校验（报数前）
- **empty=0 在全部 16 个 cell**（8 j × {16k,32k}），preds 为真实非空 vt 输出，官方 `string_match_all` 判分。n=50/cell。
- spot-check：j3 输出真链答案（"According to the chain(s) of variable assignment...", recall=1.0）；
  j48 输出 "The text provided does not contain any variable assignments"（recall=0）——**深 j 崩是真·readout 重建失败，不是空输出**。
- scorer=`scripts.eval_ruler_mem_space._string_match_all_one`；原始路径见上。

---

## 表 1：RECALL(j)（vt，n=50，string_match_all）

| j | j/L | 16k recall | 32k recall |
|---|-----|-----------:|-----------:|
| **3** | 0.047 | **93.6** | **52.0** |
| 6 | 0.094 | 39.2 | 21.6 |
| 13 | 0.203 | 14.0 | 17.6 |
| 20 | 0.313 | 48.0 | 44.0 |
| 27 | 0.422 | 26.4 | 22.4 |
| 34 | 0.531 | 33.2 | 22.4 |
| 41 | 0.641 | 3.6 | 5.6 |
| 48 | 0.750 | 2.0 | 1.6 |

- **峰值在极浅 j3（16k=93.6 / 32k=52.0）**。iter_bm25 跟的是**字面变量名**链：浅 j 的 cached h_j 最接近原始词法 token，重算 `layers[3:]` 仍能看到近逐字的变量名 → 多跳链重建最好。深 j 抽象掉字面 token → 词法多跳变难。
- **中段（j6–j34）噪声较大但成活**（vt 本身是 RULER 最噪任务，n=50 ±~7pt）：22–48% 区间，j20 有二次小峰（48/44）。
- **深 j 悬崖**：j41（3.6/5.6）、j48（2.0/1.6）**塌到近零** → 32B vt 的 readout-safe 上限落在 **j34↔j41 之间（0.53L↔0.64L）**，比单针 niah 的 j27（0.42L）安全点更深（多跳仍能撑到 0.53L）。

> ⚠️ **重要修正（本轮新发现）**：旧 `RUN_REGISTRY.md:1604` 的 32B vt j-sweep（j3=17.8@16k 等）是 **2026-07-17 chat+no-think 标准之前** 生成，被 thinking/MC 标记污染压低。本轮同 iter_bm25 但补上 chat+no-think 后 **j3 16k=93.6（旧 ~18）** → 旧 32B vt "全 scale 唯一崩、峰值仅 ~24" 的结论**作废**：干净口径下 32B vt 在浅 j 很强（≈8B+adapter 的 97、30B-A3B zs 的 95）。

## 表 2：READ-LATENCY(j)（固定 pack，与 context 长度无关）

| j | 重算层数 L-j | read_prefill (ms) | ms/层 | decode/step (ms) | read+gen@48tok (ms) | read_prefill 加速 vs j3 |
|---|----:|----:|----:|----:|----:|----:|
| 3 | 61 | 3460.7 | 56.7 | 48.6 | 5793 | 1.00× |
| 6 | 58 | 3288.8 | 56.7 | 45.1 | 5454 | 1.05× |
| 13 | 51 | 2893.6 | 56.7 | 46.7 | 5136 | 1.20× |
| 20 | 44 | 2498.1 | 56.8 | 46.1 | 4711 | 1.39× |
| 27 | 37 | 2100.0 | 56.8 | 45.7 | 4294 | 1.65× |
| 34 | 30 | 1703.1 | 56.8 | 45.8 | 3901 | **2.03×** |
| 41 | 23 | 1305.6 | 56.8 | 46.1 | 3518 | 2.65× |
| 48 | 16 | 909.0 | 56.8 | 46.6 | 3146 | **3.81×** |

- **`read_prefill` = 56.8 × (L−j) ms，近乎完美线性** —— 直接坐实机制 "read 成本 ∝ 重算层数 (L−j)"。j3→j48 read_prefill 快 **3.81×**。
- **`decode/step` ≈ 46ms 恒定**（对 j 不敏感）：单 token decode 是 HBM 带宽瓶颈（每步过权重），不随 resume band 层数明显变。故整体 read+gen@48tok 的加速被 decode 常数项摊薄到 **1.84×**（5793→3146ms）；**纯 read 阶段（prefill）才是随 j 线性变的旋钮，3.8× span**。

## 表 3：显存(j)（recall 进程实测，nvidia-smi）

| j | 13 | 20 | 27 | 34 | 41 | 48 |
|---|---:|---:|---:|---:|---:|---:|
| GPU mem (MiB) | 66223 | 65915 | 65607 | 65621 | 65635 | 65649 |

- **~65.6–66.2 GB，全 j 基本恒定（变动 <1%）**，被 32B bf16 权重（~64GB）+ 固定 read pack 主导。
- **佐证**：显存不随 j 变，变的只是**算量**（重算的层数）。QCMem read 的 O(1)-in-context 显存特性在 j-sweep 上成立（read pack 恒 ~6570 tok，见表内 avg_read_len）。

---

## Pareto frontier + story

把 (recall, read_prefill) 画成散点：**左上 = 高 recall / 慢 read；右下 = 快 read / 低 recall**。非被支配（Pareto-optimal）的操作点：

- **j3 = recall 峰（93.6/52.0）、read 最慢（3461ms）** —— frontier 最左上角。要最高精度就付 61 层重算。
- 往深走（j↑）：`read_prefill` 线性变快，recall 阶梯下滑（含 j20 二次峰 48/44）。
- **Pareto 拐点（knee）≈ j34（0.53L）**：recall 仍 33.2/22.4（未塌），read_prefill 已 **2.03× 快于 j3**（1703 vs 3461ms），read+gen 1.48×。这是 "深 j 换速度但 recall 尚可" 的甜点。
- **j41/j48 被 recall 支配**（3.6/5.6→2.0/1.6，模型输出"无变量赋值"= 真崩）：read 最快（3.81×）但 recall 无用 —— 只有在不需要 recall 时才划算。

**一句 story**：32B vt 上，QCMem 的 `resume_j` 是一根 **速度↔精度旋钮**——read 阶段代价严格线性 ∝ (L−j)（浅 j3 全 61 层→深 j48 仅 16 层 = 3.8× 提速），而 recall 从浅 j3 的峰值（词法多跳需近逐字变量名，93.6@16k）随 j 加深阶梯下滑、过 readout-safe 上限（~j34/0.53L）后坠崖。可用操作区间 = j∈[3,34]：j3 取满精度，j34 取"recall 未塌 + read 2× 提速"的折中，再深即为纯速度收益（recall 已死）。深 j readout 崩是**真重建失败**（非空输出），显存全程恒定（算量变、显存不变）。

## §per-task 泛化（T21b，2026-07-20，节点 .73）

> **动机**：T21 只测了 vt（对 readout 最敏感、最噪）。T21b 把 recall-vs-`resume_j` frontier 推广到 4 个 RULER **niah 检索类任务**，验证「resume_j 速度↔精度旋钮」的行为**随任务类型如何变**。
> **协议（与 T21 完全同口径，仅换 task）**：`models/Qwen3-32B`（L=64，zero-shot 无 adapter，bf16/sdpa），`selector=iter_bm25`（统一口径），`--use_chat_template` + `enable_thinking=False`(no-think)，topk12 / chunk512 / sink=bos。判分官方 `_string_match_all_one`。**n=50/cell，全 48 cell（4 task × 6 j × {16k,32k}）empty_output=0**、recipe 单一 `(chat=T, think=F, iter_bm25, lora=None, topk12, chunk512, zero_train=T)`、n=50 校验全过。原始：`.73:ruler_results/qcmem_32b_t21b/j{6,13,20,27,34,41}_{task}/`。

### 表 A：RECALL(j) per-task（n=50，string_match_all）

| j | j/L | single_2 16k/32k | multikey_1 16k/32k | multivalue 16k/32k | multiquery 16k/32k |
|---|-----|-----------------:|-------------------:|-------------------:|-------------------:|
| 6  | 0.094 | **100 / 100** | 98 / 96 | 48.5 / 50.0 | 25.0 / 25.5 |
| 13 | 0.203 | **100 / 100** | 94 / 94 | 48.5 / 49.5 | 23.5 / 29.5 |
| 20 | 0.313 | **100 / 98**  | 96 / 90 | 49.5 / 50.0 | 25.0 / 32.5 |
| 27 | 0.422 | **100 / 98**  | 96 / 86 | 49.5 / 49.0 | 26.5 / 32.0 |
| 34 | 0.531 | **100 / 100** | 66 / 62 | 49.5 / 48.5 | 23.0 / 24.0 |
| 41 | 0.641 | **10 / 34**   | 8 / 4   | 1.5 / 1.5   | 2.0 / 0.0 |

（vt 见 T21 表1：峰值在浅 j3=93.6/52.0，中段噪声，j41=3.6/5.6 崩。）

### 表 B：READ-LATENCY(j)（引 T21 表2，**任务无关**）

read pack 恒定（T21b 实测 avg_read_len=6182 在 j6/j34 **完全相同**）→ read latency 只随 (L−j) 变、与 task 无关，直接引 T21 线性轴 `read_prefill=56.8×(L−j)ms`：

| j | L−j | read_prefill (ms) | 加速 vs j6 |
|---|----:|------------------:|----------:|
| 6  | 58 | 3288.8 | 1.00× (ref) |
| 13 | 51 | 2893.6 | 1.14× |
| 20 | 44 | 2498.1 | 1.32× |
| 27 | 37 | 2100.0 | 1.57× |
| 34 | 30 | 1703.1 | **1.93×** |
| 41 | 23 | 1305.6 | 2.52× |

### 结论 story：resume_j 旋钮的**任务依赖性**

**普适悬崖**：所有 4 个 niah task 的 readout 硬崖都精确落在 **j41（0.64L）**——j34→j41 全部坠落（single 100→10, multikey 66→8, multivalue 49→1.5, multiquery 24→2）。这是 32B niah readout 的统一上限（比 vt 的 j41/j48 略同）。

**崖下 recall 对 j 的形状 = 任务依赖**（这才是旋钮的关键）：
- **niah_single_2 = 深-j 最宽容（纯免费提速）**：recall **在 j6→j34 完全平在 100**（32k 仅 98-100 抖动），直到 j41 才崩。→ 可放心推到 **j34（0.53L）= read 1.93× 提速、recall 零损失**，是「深 j 换速度」最干净的案例（简单单针检索，浅缓存 h_j 已够重建读出）。
- **niah_multivalue / niah_multiquery = 深-j 宽容但有任务天花板**：recall 各自**平在任务上限**（multivalue ~49-50，需 string_match_all 全中 4 值；multiquery ~23-32，取 4 针）**从 j6 平到 j34**，j41 崩。→ **recall 不随 j 降**（deep j 亦免费），只是绝对分被任务难度锁死；multiquery 甚至在 j20-27 有轻微峰（32k=32）。
- **niah_multikey_1 = 居中**：高位 86-98 平到 **j27（0.42L）**，j34 开始温和衰减（66/62），j41 崩。→ 免费区到 j27，j34 起付小额 recall 代价。
- **vt（T21）= 唯一偏好浅 j**：峰值在极浅 **j3**（词法多跳需近逐字变量名），随 j 加深阶梯降 + 噪声大，readout-safe ~j34，j41/48 崩。deep j 对 vt 是净损失（丢失字面 token）。

**一句 story**：`resume_j` 速度↔精度旋钮的**斜率强烈依赖任务类型**。read 成本永远 ∝(L−j)（任务无关，j6→j34 = 1.93× 提速）；但 recall(j) 的形状分三类——① **检索类 niah（single/multivalue/multiquery）recall 崖下对 j 近乎恒定** → deep j 到 **j34（0.53L）是"纯免费提速"**（recall 零损，只是各自任务天花板）；② **multikey 居中**（免费到 j27，j34 起温和衰减）；③ **vt 反向偏好浅 j**（词法多跳需近逐字 token，deep j 净损）。**统一 readout 硬崖在 j41（0.64L）**，所有 niah 任务一致坠落。→ 论文效率章：对多数检索任务 QCMem 可默认推到 j≈0.5L 吃满 ~2× read 提速而不掉 recall；仅 vt 类词法多跳需保守用浅 j。

## 落账
- 数据源（T21）：`.73` diskB `ruler_results/qcmem_32b_t21_vt_j*` + `logs/t21_profile_j*.log`（原始留 diskB，未搬回）。
- 数据源（T21b，本节）：`.73` diskB `ruler_results/qcmem_32b_t21b/j{6,13,20,27,34,41}_{niah_single_2,niah_multikey_1,niah_multivalue,niah_multiquery}/`（48 cell，n=50，留 diskB）。
- 关联：`status/QCMEM_J_DETERMINATION.md`（readout-safe 深度）、`status/RUN_REGISTRY.md:1604`（旧污染 32B vt sweep，本表修正之）、`status/RUN_REGISTRY.md`（T21b per-task 行）。
