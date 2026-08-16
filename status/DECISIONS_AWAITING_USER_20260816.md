# DECISIONS_AWAITING_USER_20260816.md — 需要你决定的，一次说清

> 你问「把需要我决定的再说一遍」。上一轮我只报了两件（B11 issue、`out_llama/`）。
> 我重新按盘上实测扫了一遍，**实际是四件**，其中两件我之前一直没报出来：
> 任务 **#192**（在 task list 里明确标着 `DECISION`）和 **A04**（`STATUS.json` 里显式
> `USER APPROVAL for GPU`）。这份文件是这四件的完整摊开，每条都带我自己核实过的证据。
>
> 也顺手更正一个我反复漏报的事：**#192 早在 2026-08-08 就有一份完整的决策分析在盘上**
> （`status/PAPERB_192_TABLE4_BUDGET_DECISION.md`，含推荐、成本比、三处前提纠正），
> 而我在多轮 heartbeat 里都没引用它。它不是「待写」，是「待你拍板」。

---

## D1 — `out_llama/` 里哪些 SparseForge run 还要留（**4.69 TiB，唯一有实质空间收益的一件**）

**为什么只有你能答**：99 个目录里只有 3 个被committed文件按名字引用，但它们是通过
`--out_dir`/glob 寻址的，所以「没被点名」≠「没被用到」——这正是 08-12 audit 自己
在 `mem_space` 上否掉的松推理，我不能对 `out_llama` 用双标。而且**#245（复现 ALPS+SLoRB
作为 matched control）仍 pending，跑的正是 Llama-2-7B**，那 2.6 TiB 就在这个 family 里。

| 选项 | 释放 | 代价 | 不可逆? |
|---|---|---|---|
| **D1-a**（我倾向）你给一张「要留的 run 清单」，删补集 | **~3–4 TiB**，wzc1 92%→~89% | 你花几分钟列清单 | 是（但 `args.json`+`best_lm_eval.json` 只有 KB，可全留作记录） |
| D1-b 全留，等 #245 跑完再谈 | 0 | wzc1 继续 92%，还剩 10 T | 否 |
| D1-c 只删非-Llama2 family（Qwen3 423G + 13b 398G + opt 308G + moe 207G + gpt2 144G） | ~1.4 TiB | 若 rebuttal 要跨模型对照就毁了 | 是 |

**背景数字（我实测）**：我们在 wzc1 占 **17.05 TiB = 已用的 15.5%**，35 个用户里排第 2。
zwfy6 的 97% **不是我们造成的**（我们 3.4%，`hunyuan/` 单独 322 TiB = 全盘 48%），
所以那边删什么都没意义。详见 `status/DISK_DECISION_20260816.md`。

---

## D2 — B11 的上游 bug issue 要不要提到 `booydar/babilong`

**为什么等你**：对外发布、不可逆。

草稿 `proposal/backlog/B11-generative-scorer-format-fragility/K3_EXIT_UPSTREAM_ISSUE_DRAFT.md`
（519 行），我自己审过：
- **上游确实还没人提** —— 18 条 issue/PR（`state=all`）+ 20 条 comment 扫 20 关键词，4 命中全是误报；
  9 branch + 27 fork 的 `metrics.py` md5 全同 `0a5ecc52`，line 31 还是 `split('Question')`。
- **措辞已修正**：`unreachable` 是**假的**（`sys.settrace` 显示 7/7 输入都执行），正确说法是
  **executed but can never fire**。草稿里 `unreachable` 只出现在否认它的句子里。若按旧措辞提，
  维护者一句 "no, it runs" 就能关掉。
- **§5 待粘正文（10,328 字符）泄漏扫描 0 命中** —— 内部路径只在 §2 我们自己的 provenance 笔记里。
- 附带价值：修复会**改动已发表数字**（LIST 格式 −12.71 pp vs 非 LIST +1.71 pp，符号翻转）。

| 选项 | 结果 |
|---|---|
| **D2-a**（我倾向）提 | 上游得到一个一字符修复；我们拿到 K3 exit，B11 可推进 |
| D2-b 不提，只留内部记录 | 0 风险，但 B11 的 K3 exit 一直挂着 |
| D2-c 你自己提 | 同 a，署名是你 |

---

## D3 — #192：Table 4 budget defect（task list 里标 `DECISION`，我此前漏报）

**盘上现状我刚核实过，和 8 天前那份分析一致：**
- `tab_downstream.tex` **已经写真实 step**（keep8 121k / keep10 83.5k / keep12 124k），
  不是「统一写 200k」。**主表部分早已修好** —— 我之前引用的「统一 200k」来自
  `main.aux` / `SUBMISSION_STATUS.md` / `TODOList.md` 这些**陈旧产物**。
- **但 keep12 的 ARC-E 仍是 `.689`，那是 6/8 shard 算出来的**（1782 = 6×297，
  shard0/shard5 因 HF 429 被 skip，merge 不把 `n_skipped_shards` 写进 summary 所以看不出来）。
  干净 v2 是满额 2376，值 `.6936`。**这比 budget 问题严重 —— 那是数字本身错。**
- 三个干净 v2 目录**现在都在 zwfy6 上**（我刚 ls 过，PRESENT×3）。

| 选项 | 成本 | 换来什么 |
|---|---|---|
| **D3-a（原分析推荐 A+，我也倾向）** 切到干净 v2 + 修 partial merge + 补 4 处附录/3 处元数据，**并 kill 两条 live resume** | **0 GPU**（结果已在盘）+ 小时级 CPU | 修掉数字错误；`keep8 vs keep12` 这个 budget-matched 对照零成本变成论文加强项 |
| D3-b resume 三 arm 到 200k 重跑 | **9.2 天占三台 H20** | 「齐整」，但用不可见的 optimizer/LR 异质换掉可见的 step 异质 → 分析判定**审稿风险上升**；且**完全不解决 partial merge** |
| D3-c 混合（resume keep10/keep12，keep8 披露） | 中 | 会**毁掉** keep8-vs-keep12 那个对照 |

**A+ 的一个连带后果必须一起处理**：切 v2 后 keep12 MMLU 从 `.2752` → `.2717`，
**与 keep10 的 `.2717` 完全相等**，所以任何依赖「keep12 MMLU 高于 keep10」的措辞都要改
（`app_tab_mmlu.tex:23-24` 那句 "above chance" 仍成立，Wald 下界 .2644/.2678 都 > .25）。

> 注意 D3 与 D1 有耦合：选 D3-a 要 kill 的两条 live resume 就是现在 `.73` keep12（明天 ~20:30 落地）
> 和 `.82` keep8。**我不会自己 kill 它们** —— 这属于「kill 本轮刚要求跑完的任务」。

---

## D4 — A04 的 GPU 授权（`STATUS.json` 明写 `USER APPROVAL`，我此前漏报）

`proposal/active/A04-recovery-certification/STATUS.json` 的
`blocked_by.still_blocking_before_any_gate_gpu` 第 3 条**逐字**是：

> "USER APPROVAL for GPU. The full gate is **1,077–4,309 GPU-h**; nothing beyond Pilot Zero
> may be launched without it."

另两条是 0-GPU 且**我可以自己做**（我没做，因为第 3 条没解决前做了也不能跑）：
1. `PROPOSAL.md` 收窄到 narrowed `safe_residual_claim`（现版本的「提案」很大程度是
   arXiv:2606.14150 的设计、动机很大程度是 2607.00368/2601.22950 的论证）；
2. 一行代码：`scripts/train_olmo2_arch_probe2.py:863` 的 `DistributedSampler(ds, shuffle=True)`
   **没传 `seed=`**，所以 `--seed` 只动 fresh-tail init，**data order 在各 seed 间完全相同** ——
   不修的话 K2 的方差估计漏掉 data-order 成分。

| 选项 | 含义 |
|---|---|
| **D4-a**（我倾向）先只批 **Pilot Zero**（CPU + ≤3 GPU-h），拿它的结果再决定全 gate | 几乎零成本，且 Pilot Zero 的设计目的就是判断「PLATEAU-accepts / NI-rejects 的分歧到底存不存在」 |
| D4-b 批全 gate（1,077–4,309 GPU-h） | 40 卡满载下约 **5.6–22 天** 全集群 |
| D4-c 不批，A04 冻结 | proposal 停在 `needs_prior_gate` |

---

## 我不需要你决定、会继续自己做的

- 五条训练继续跑（`.73` keep12 明天 ~20:30 落地，chain watcher 已验证存活会自动接 ladder eval）
- paperC 迭代 + gate 硬化 + re-freeze（round_05 已建，round_04 hash 未动）
- proposal 的 0-GPU lane（B08 leg-1 代码正在跑）
- `.104` 那个 1.13× 的可证伪预测下轮验
- 磁盘：**一个字节都没删**，也不会在 D1 之前删

---

# RESOLVED 2026-08-16 22:45 — user答复 + 我据授权的裁定

用户原话：**「D1先不管了 D2不提 其他的你自己决定」**

| | 决定 | 谁定的 | 执行 |
|---|---|---|---|
| **D1** `out_llama/` 4.69 TiB | **不管，全留** | 用户 | 停手。**一个字节都没删，也不再提**。wzc1 继续 92%/剩 10 T —— 够用，不构成风险。#245 的 Llama-2-7B family 因此完整保留，这也顺带消除了它的资产风险。 |
| **D2** B11 上游 issue | **不提** | 用户 | 不提。草稿 `K3_EXIT_UPSTREAM_ISSUE_DRAFT.md` 留在盘上作内部记录（它含已执行的可达性证据，本身有价值）。**B11 的 K3 exit 就此按「不对外」结案**，不再挂着等。 |
| **D3** #192 Table 4 | **A+（0 GPU 修数字）** | 我，据授权 | 已派 agent。明确禁止选项 B（kill+resume 三臂）。 |
| **D4** A04 GPU | **只批 Pilot Zero（CPU + ≤3 GPU-h）；全 gate 1,077–4,309 GPU-h 不批** | 我，据授权 | 见下。 |

## D3 的执行边界（已写进派活 prompt）

选 A+ 的理由是**它修的是一个错的数字，不是一个难看的标签**：Table 4 的 keep12 ARC-E `.689`
来自 **6/8 shard**（1782=6×297），干净 v2 是满额 2376 = `.6936`。0 GPU，因为结果已在盘上。

三条硬约束：
1. **整表必须同源同盘** —— 半切的表比现状更糟。做不到就什么都不改并报告。
2. **keep12 MMLU 会 `.2752`→`.2717`，与 keep10 完全相等** —— 依赖「keep12 > keep10」的措辞必须一起改。
3. **每个引入的目录都要断言 per-task `n_scored == expected`** —— 正在修的就是静默 partial merge，
   不能顺手引进第二个。

## D4 的裁定与理由

**批 Pilot Zero，不批全 gate。** 理由：
- Pilot Zero 的**设计目的**就是先判断「PLATEAU-accepts / NI-rejects 的分歧到底存不存在」，
  用的是**已在盘上的 checkpoint**，CPU + ≤3 GPU-h。这是一个 kill-gate 性质的探针，不是投资。
- 全 gate 是 **1,077–4,309 GPU-h** = 40 卡满载 **5.6–22 天**。在分歧是否存在都没确认前投入，
  违反「先跑便宜的判别性实验」。
- 另两条 blocker 是 0 GPU 且**归我**，我会做（下面）。第 3 条 USER APPROVAL 现在按授权由我批到 Pilot Zero 为止。

**明确不批的**：rungs 之外的任何扩展、Qwen3.5-9B generalisation leg（它的成本
`NOT ESTIMATED, deliberately`，因为需要先把 `sparse_modeling.py` 移植到多模态 hybrid-attention
架构 —— 未 scope 的东西不能批）。

## 顺带自己决的两件（不在原四件里）

- **B12 pilot 授权跑**：它的 G0 **两条腿都已 PASS（0 GPU）**，自己的 `gpu_policy` 释放条件已满足，
  pilot 是 **1.46 GPU-h 实测**（0.73 GPU-h/cell，来自 union-9 closeout 自己的 stage 时间戳）。
  要求 **sm_100 + wzc1 = LOCAL 或 .212**。**LOCAL 约 6.1 h 后 keep10 到 200k 自然空出** ——
  不需要 kill 任何东西。已修它的 argparse 缺陷（见下）。
- **`.104` 的 1.13× 我自己的假设被自己的测试推翻了**，已如实记录为「未解释的单窗口波动，已自愈」，
  不再追。见 `TRAINER_ACTIVITY.jsonl` 22:40 的 `MY_HYPOTHESIS_REFUTED`。
