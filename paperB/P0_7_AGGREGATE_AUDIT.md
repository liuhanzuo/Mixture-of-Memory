# P0.7 — core6 / aux4_raw / aux5_raw 聚合口径 raw-JSON 审计

**状态**：DONE（纯分析，无 GPU/模型 run）
**脚本**：`scripts/audit_olmo2_aggregates.py`（从每个 arm 原始 `summary.json` 重算，核对到 <1e-6）
**机器可读表**：`paperB/P0_7_aggregate_audit.csv` / `paperB/P0_7_aggregate_audit.json`
**日期**：2026-08-02

---

## 1. 固定口径（严格按 TODOList §P0.7，权威）

| aggregate | 成员任务 | metric key |
|---|---|---|
| **core6** | HellaSwag, ARC-Challenge, ARC-Easy, PIQA, OpenBookQA | `acc_norm` |
| | WinoGrande | `acc` |
| **aux4_raw** | LAMBADA(openai), BoolQ, CommonsenseQA, SocialIQA | plain `acc` |
| **MMLU** | MMLU（单独报告，**不并入任何 knowledge aggregate**） | plain `acc` |
| **aux5_raw**（可选） | aux4 四项 + MMLU | plain `acc` |
| **MMLU recovery** | (MMLU − 0.25) / (base_MMLU − 0.25)，base_MMLU=**0.605256** | — |

- `aux4_raw` / `aux5_raw` 仅是异质任务的描述性算术均值，**禁止**称作 "knowledge recovery"。
- `aux5_raw` 的成员集合 = 历史上的 "know5"，但名字必须改为 `aux5_raw`。

---

## 2. 每 arm 重算结果（全部来自 raw JSON）

| Arm | core6 | aux4_raw | MMLU | aux5_raw (optional) | MMLU recovery | PPL |
|---|---:|---:|---:|---:|---:|---:|
| base full (32L, 参照) | **0.7037** | 0.6783 | 0.6053 | 0.6637 | 100.0% | 7.398 |
| keep8 @121k (headline) | 0.5238 | 0.4727 | 0.2535 | 0.4289 | 1.0% | 13.333 |
| keep8 @110k | 0.5218 | 0.4729 | 0.2500 | 0.4284 | 0.0% | — |
| keep8 @100k | 0.5168 | 0.4814 | 0.2490 | 0.4350 | −0.3% | — |
| keep10 @83.5k (headline) | 0.5303 | 0.4934 | 0.2718 | 0.4491 | 6.1% | 12.816 |
| keep12 @124k (headline) | 0.5669 | 0.5073 | 0.2752 | 0.4608 | 7.1% | 11.443 |
| keep14 @200k (headline) | 0.5938 | 0.5371 | 0.3191 | **0.4935** | 19.5% | 10.561 |
| ShortGPT-16 @200k (trained) | 0.6215 | 0.5811 | 0.4739 | 0.5596 | 63.0% | 9.780 |
| ShortGPT-16 @0 (no-heal) | 0.4212 | 0.2780 | 0.2620 | 0.2748 | 3.4% | 401.124 |
| frozen-front @200k | 0.5631 | 0.4928 | 0.2628 | 0.4468 | 3.6% | 12.797 |
| random-init (16L) @200k | 0.5584 | 0.4909 | 0.2461 | 0.4419 | −1.1% | 11.498 |

> PPL 来源：`paperB/data/raw/olmo2_ppl_results/<arm>/summary.json`（keep8@121k 用顶层
> `olmo2_ppl_results/7B_keep8_step121000/`，keep10/keep12 用 RUN_REGISTRY 的 batch82 PPL
> 12.816 / 11.443，其 downstream JSON 已 materialize 到 paperB/data/raw）。
> keep12 arc_easy n=1782（该 shard 集下 n 与其它 arm 的 2376 不同，raw JSON 原样保留，
> acc_norm=0.6891 已按 arm 内比例进入 core6；非口径问题，为该次 eval 的样本计数，已如实记录）。

### core6 与 TODO 表值一致性
| Arm | 重算 core6 | TODO 表值 | 一致? |
|---|---:|---:|:--:|
| base full | 0.703684 | 0.7037 | ✅ |
| keep8 @121k | 0.523821 | 0.5238 | ✅ |
| keep10 @83.5k | 0.530327 | .531 (RUN_REGISTRY ~.531) | ✅ |
| keep12 @124k | 0.566863 | .567 (RUN_REGISTRY ~.567) | ✅ |
| keep14 @200k | 0.593757 | 0.5938 | ✅ |
| ShortGPT-16 @200k | 0.621520 | 0.6215 | ✅ |

**结论：所有已发布 core6 值均正确，无需修正。** core6 从未混用 metric（一直是
acc_norm×5 + winogrande acc），bug 只出在 aux5/know5 那一列。

---

## 3. 旧聚合值审计（`.6639/.4491/.4608/.5071/.5596`）

对每个旧值，同时算两种 5-task 均值：
- **aux5_raw (plain)** = 5 项全用 `acc`（P0.7 正确口径）
- **mixed(normBCS)** = MMLU/LAMBADA 用 `acc`，但 BoolQ/CSQA/SIQA 误用 `acc_norm`（疑似 bug 口径）

| 旧值 | 归属 arm | aux5_raw (plain, 正确) | mixed(normBCS, bug口径) | 诊断 |
|---:|---|---:|---:|---|
| **.6639** | base full | **0.6637** | 0.6577 | ✅ OK — 旧值本就是 plain-acc 5-task 均值（应重命名为 aux5_raw，语义修正） |
| **.4491** | keep10 @83.5k | **0.4491** | 0.4520 | ✅ OK — plain-acc，值正确 |
| **.4608** | keep12 @124k | **0.4608** | 0.4773 | ✅ OK — plain-acc，值正确 |
| **.5071** | keep14 @200k | 0.4935 | **0.5071** | ❌ **BUG** — 旧值用了 acc_norm 混算（BoolQ/CSQA/SIQA 误取 acc_norm）；**正确 aux5_raw = 0.4935** |
| **.5596** | ShortGPT-16 @200k | **0.5596** | 0.5530 | ✅ OK — plain-acc，值正确 |

### 需要修正的具体数字
- **只有 keep14 的 `.5071` 是数值错误**：正确 `aux5_raw = 0.4935`（差 −0.0136）。
  - 错因：BoolQ 0.6382(acc)→误用 0.6887(acc_norm)、CSQA 0.4988→0.4758、SIQA 0.4340→0.4744，
    三项混入 acc_norm 把均值抬高了 ~1.4pp。
- **其余四个旧值（.6639 / .4491 / .4608 / .5596）在数值上都正确**（本就是 plain-acc 5-task 均值）。
  它们**唯一**的问题是**命名/语义**：曾被叫作 "know5 / knowledge aggregate"，
  按 P0.7 必须改称 **`aux5_raw`**，且不得解释为闭卷知识轴。

### 命名/解释层面的统一修正（对所有 arm）
1. 把任何 "know5" / "knowledge aggregate" 列 → 重命名 **`aux5_raw`**（若沿用五项集合）
   或拆成 **`aux4_raw` + 单独 MMLU**（推荐，避免把 MMLU 混进异质均值）。
2. MMLU 一律单列，配 `above-chance recovery`。
3. keep14 的 aux5_raw：`.5071 → .4935`（数值修正）。

---

## 4. 需要回填修正的文件（供 main 处理，本报告不改这些文件）

| 文件 | 位置 | 动作 |
|---|---|---|
| `paperB/TODOList.md` | line 33-49「aux5 aggregate」列 + line 49 shortgpt `.5596` + line 208/220 表 | core6 全对；填 aux4_raw/aux5_raw；keep14 aux5 `.5071→.4935`；把 "know5" 措辞改 aux5_raw |
| `status/RUN_REGISTRY.md` | line 1980(keep10)/1997(keep12)/2013(shortgpt) 的 "know5" 列 | 数值均对（.4491/.4608/.5596），仅把表头 "know5" → aux5_raw |
| `status/GPU_STATUS.md` | line 59/60/68/128 "know5" 提法 | 数值对，改称 aux5_raw；不再叫 knowledge |
| `status/PAPERB_THREE_ARM_200K.md` | line 63 "know5 = .5596" | 数值对，改称 aux5_raw |
| `.tex`（若含 know5/该列） | grep `know5` 确认匿名 PDF 不再出现未定义 aggregate | 由 main 检查 |

> ⚠️ 注意：keep14 MMLU above-chance recovery 精确值 = (0.319114−0.25)/(0.605256−0.25)
> = **19.45% → 19.5%**（TODO line 37 写的 19.4% 是四舍五入偏差，建议统一为 19.5%）。

---

## 5. Provenance（raw JSON 路径）

本机 wzc1（直接读）：
- base full：`paperB/data/raw/olmo2_downstream_results/7B_base_full{,_know}/summary.json`；PPL `paperB/data/raw/olmo2_ppl_results/7B_base_full/`
- keep8 @{100k,110k,121k}：`olmo2_downstream_results/7B_keep8_step{100000,110000,121000}{,_know}/`；PPL `olmo2_ppl_results/7B_keep8_step121000/`
- keep14 @200k：`olmo2_downstream_results/7B_keep14_step200000{,_know}/`；PPL `paperB/data/raw/olmo2_ppl_results/7B_keep14_step200000/`
- shortgpt @0：`paperB/data/raw/olmo2_downstream_results/7B_shortgpt_step0{,_know}/`
- frozen-front / random-init(scratch16L) @200k：`olmo2_downstream_results/7B_{freezefront,scratch16L}_step200000{,_know}/`（PPL 同名 `paperB/data/raw/olmo2_ppl_results/`）

从 diskB .82（28.82.250.82:36000）拉取并 materialize 到本机：
- keep10 @83.5k：`.82:olmo2_downstream_results/7B_keep10_step83500{,_know}/` → `paperB/data/raw/olmo2_downstream_results/7B_keep10_step83500{,_know}/summary.json`
- keep12 @124k：`.82:...7B_keep12_step124000{,_know}/` → 同上目录
- ShortGPT-16 @200k：`.82:...7B_shortgpt16_step200000{,_know}/` → 同上目录

`[PENDING training]`：full32 32L continued-pretraining control（P1.1，LOCAL/wzc1 #100）未到 200k，本审计不含，跑满后按同口径补一行。
`[NOT MISSING]`：frozen-front / random-init 在本机 `olmo2_downstream_results/` 已存在（分别为 `7B_freezefront_step200000` 与 `7B_scratch16L_step200000`，random-init = from-scratch 16L），无需去 .82 找。

> ⚠️ **2026-08-17 补记（#192 A+ 之后，本节上面的 keep8/keep10/keep12 路径已不是 Table 4 的来源）。**
> commit `6d15049` 把 Table 4 的三个浅层行换成了单协议 `_v2` 重测，但**没写任何证据路径**，
> 且这些目录**不在 `outputs/` 也不在 `evals/`**（`evals/` 在本仓根本不存在），
> 所以 `grep -r '0\.6936' outputs evals` 两盘都返回空 —— 那是**检索范围假象，不是文件缺失**。
>
> Table 4 三行的真实来源（完整审计见 `paperB/TABLE4_PROVENANCE_20260817.md`）：
> - **十个非 MMLU 列**：**仅 zwfy6**
>   `olmo2_downstream_results/7B_keep{8,10,12}_step{121000,83500,124000}_v2{,_know}/summary.json`
>   （wzc1 上 `find -maxdepth 6 -type d -name '*keep{8,10,12}_step*_v2*'` 为空；
>   按 `paperB/data/README.md` 的口径属 disk-local，非 portable）。
> - **MMLU 列**：`olmo2_mmlu_content_results/7B_keep{8,10,12}_step{121000,83500,124000}/summary.json`
>   的 **`letter_acc`** 字段，两盘 sha256 一致。
>   ⚠️ 同名的 `*_wzc1`（.2546/.2717/.2713）与 mmlu 树里的 `*_v2`（.2543/.2707/.2724）都是**诱饵**，
>   且 `7B_keep12_step111500_wzc1` **step 也不对**（111500≠124000）。
>
> **已核**：33/33 cell 复现（含从 `shard{0..7}of8.json` 绕过 `summary.json` 重算，
> keep12 arc_easy = 1648/2376 = 0.693603 → `.694`），逐 cell 断言
> `n_scored==n==expected` / `n_nan==0` / `n_shards==8` / `add_bos==false` / `ckpt_step==` 行内 step，
> 三个脚本 `PY_RC=0`。无一 cell 不复现。
>
> **两处仍需注意**：(1) `paperB/data/raw/` 里 keep8@121k **完全没有**，keep12@124k 是**旧的 6/8 缺陷文件**
> （`arc_easy n_scored=1782` 而 `n_shards` 仍写 8 —— 缺陷可在文件中直接验证），
> keep10@83.5k 是另一次（pre-`_v2`）测量，与 Table 4 最大差 0.40pp（keep12 侧 0.49pp，均在 caption 披露的
> ≤0.5pt 同架构跨 stack 界内）。(2) `paperB/scripts/build_appendix_artifacts.py` 只经
> `RAW = ROOT/"data"/"raw"` 取数且断言仍指向旧 rung，**不读任何 `_v2` 路径** ——
> 它跑绿**不构成** Table 4 浅层三行的证据。

---

## 6. 一句话结论

**core6 全部正确无需改。** 历史 "know5" 列里 base/keep10/keep12/shortgpt16 的**数值都对**
（本就是 plain-acc 五项均值），**唯一数值错误是 keep14 `.5071`（应为 `.4935`，acc_norm 污染）**。
所有 arm 的该列都必须**改名为 `aux5_raw`**（或拆成 aux4_raw + 单独 MMLU），
并停止把它当作 "knowledge recovery / 闭卷知识轴"。
