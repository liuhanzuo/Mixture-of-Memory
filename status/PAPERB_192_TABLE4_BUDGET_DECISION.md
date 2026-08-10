# #192 决策分析：Table 4 budget defect —— 如实披露 vs resume 到 200k

作者：subagent（Phase-2 决策分析）  日期：2026-08-08
范围：只做分析，**未改任何 .tex / TODOList / versions**。所有数字都标了磁盘出处。

---

## 0. 三个必须先纠正 MAIN 前提的发现（**这三条改变了决策结构**）

### 0.1 ★ Table 4 已经不写 200k 了 —— defect 的**主表部分已被修好**，MAIN 的前提过期

MAIN 的 brief 说「论文 Table 4 统一写 200k」。**磁盘上不是这样。**
`paperB/sections/tab_downstream.tex`（= Table 4，由 `main.aux:567` 的
`\newlabel{tab:downstream}{{4}...}` 确认编号）当前内容：

```
paperB/sections/tab_downstream.tex:16  keep8 (10L)   & 121k   & ...
paperB/sections/tab_downstream.tex:17  keep10 (12L)  & 83.5k  & ...
paperB/sections/tab_downstream.tex:18  keep12 (14L)  & 124k   & ...
paperB/sections/tab_downstream.tex:40  ... Rows use their actual retained checkpoints; ...
```

真实 step 已经在表里了，caption 也已经改成 "actual retained checkpoints"。
`sections/03_method.tex:29`、`sections/05_analysis.tex:32`、`sections/06_limitations.tex:9`、
`sections/app_tab_ppl.tex:45`、`sections/app_tab_mmlu.tex:10-12` 同样已披露真实 step。

**MAIN 之所以看到「统一写 200k」，是因为读了陈旧产物**：
- `paperB/main.aux:567`（陈旧编译缓存）仍写 "All trained endpoint rows, including full32, are reported at step 200k"
- `paperB/SUBMISSION_STATUS.md:7` 仍写 "Table 4 reports trained endpoints at 200k"
- `paperB/TODOList.md:267` 仍写 "所有深度阶梯 arm 均已完整运行至 200k step，论文 Table 4 的 Budget 列统一为 200k"

→ **选项 A 的主表部分已经完成了。剩下的是 4 处附录残留 + 3 处元数据文件。**
→ 也就是说 **A 的剩余成本从「大改」降到「4 行附录 + 3 行元数据」**，这大幅改变 A/B/C 的权衡。

### 0.2 ★★ 比 budget defect **更严重**的问题：Table 4 的 keep12 行是 **6/8 shard partial merge**

我实测 Table 4 三个 shallow 行的真实来源目录（用 11 个 task 值全位匹配定位，11/11 exact）：

| Table 4 行 | 来源目录 | 盘 |
|---|---|---|
| keep8 (10L) | `olmo2_downstream_results/7B_keep8_step121000` | wzc1 |
| keep10 (12L) | `paperB/data/raw/olmo2_downstream_results/7B_keep10_step83500` | wzc1 |
| keep12 (14L) | `paperB/data/raw/olmo2_downstream_results/7B_keep12_step124000` | wzc1 |

然后对这三个目录跑完整性断言：

```
Table4 keep8 :      DEFECTS=none
Table4 keep10:      DEFECTS=none
Table4 keep12:      DEFECTS=['arc_easy: n_scored=1782 EXPECT 2376 (=6.00/8 shards)']
```

**Table 4 里 keep12 的 ARC-E `.689` 是 6/8 shard 算出来的**（1782 = 6×297）。
这正是 Phase-1 已经定位的 partial-merge 缺陷（shard0/shard5 因 HF 429 rate-limit 被 skip，
merge() 只累加 `n_skipped_shards` 而不写进 summary，所以 summary 表面看不出缺 shard）。
干净 v2 的 `arc_easy` 是满额 2376，值 `.6936`。

**这条比 budget 问题严重得多**：budget 差异是"披露即可"，partial merge 是**数字本身错**。
如果 reviewer 或 artifact 检查者去数 `n_scored`，这是一个可直接指认的数据完整性缺陷。

### 0.3 ★ Table 4 的 5 个非 shallow 行的 MMLU 与其 core-task 来源目录**不同源**

同一定位脚本对 base/keep14/sg16/frozen/random 只能拿到 **10/11 匹配**，
唯一不匹配的永远是 MMLU：

| Table 4 行 | 10 个 core/know task 来源 | 表里 MMLU | 该目录 MMLU |
|---|---|---|---|
| base | `paperB/data/raw/.../7B_base_full` | .6054 | .6053 |
| keep14 | `olmo2_downstream_results/7B_keep14_step200000` | .3184 | .3191 |
| ShortGPT | `paperB/data/raw/.../7B_shortgpt16_step200000` | .4742 | .4739 |
| frozen | `olmo2_downstream_results/7B_freezefront_step200000` | .2624 | .2628 |
| random | `olmo2_downstream_results/7B_scratch16L_step200000` | .2470 | .2461 |

这是**有意为之且已在 caption 声明**的：`tab_downstream.tex:31` 写
"MMLU uses the same-item dual-interface letter snapshot as Table 5"。
但注意**三个 shallow 行是 11/11 匹配 downstream `_know` harness**，也就是
**shallow 行的 MMLU 来自 `_know` harness，另 5 行来自 dual-interface harness**。
Phase-1 已经发现这两个 harness 对同一 ckpt 给不同 MMLU（keep10: .271685 vs .272041）。
→ Table 4 的 MMLU 列**混了两个 harness**。这与 budget 问题独立，但会在同一次修订里被 reviewer 一起问到。

---

## 1. 选项 A：如实披露真实 steps

### 1.1 具体要改哪几处（grep 结果，**我没有改**）

**已经正确、不需要动**（供 MAIN 确认无需重复劳动）：
| 文件:行 | 现状 |
|---|---|
| `paperB/sections/tab_downstream.tex:16-18` | 已写 121k / 83.5k / 124k |
| `paperB/sections/tab_downstream.tex:40` | caption 已写 "actual retained checkpoints" |
| `paperB/sections/03_method.tex:29` | 已写 "keep8 at 121k, keep10 at 83.5k, keep12 at 124k. These rows are neither step-, token-, nor FLOP-matched" |
| `paperB/sections/05_analysis.tex:32` | 已写 "stop at 121k / 83.5k / 124k ... endpoint inventory rather than a matched-step" |
| `paperB/sections/06_limitations.tex:9` | 已写 "shallow rows stop at unequal retained checkpoints" |
| `paperB/sections/app_tab_ppl.tex:18,20,22,45` | 已用 121k$^\dagger$/83.5k$^\dagger$/124k$^\dagger$ + 脚注列真实 step |
| `paperB/sections/app_tab_mmlu.tex:10-12` | 已写 "keep8, 121k / keep10, 83.5k / keep12, 124k" |

**仍然写假 200k、必须改**（4 处 tex，全在附录）：
| 文件:行 | 原文 | 问题 |
|---|---|---|
| `paperB/sections/app_tab_protocol_controls.tex:12` | `keep8, 200k & .2550 & .3219 & .3423` | 真实 121k |
| `paperB/sections/app_tab_protocol_controls.tex:13` | `keep10, 200k & .2720 & .3230 & .3445` | 真实 83.5k |
| `paperB/sections/app_tab_protocol_controls.tex:14` | `keep12, 200k & .2728 & .3419 & .3629` | 真实 124k |
| `paperB/sections/app_tab_protocol_controls.tex:50` | `keep8, 200k & .1246 & .1577 & .0368` | 真实 121k（闭卷 QA 表） |
| `paperB/sections/app_tab_protocol_controls.tex:32` | caption `All pruned rows are reported at step 200k.` | **整句假**，含 keep8/10/12 |
| `paperB/sections/app_tab_protocol_controls.tex:60` | caption `keep8 ... it is reported at step 200k.` | 假 |
| `paperB/sections/app_tab_keep8.tex:9,11-21` | 列头 `\textbf{200k}` + `$\Delta_{44\to200k}$` | 真实是 121k；11 行数据 + 2 处 caption(26-27,31) |

**需要独立核实的一处**（我倾向它是对的，但没找到直接 artifact）：
- `paperB/sections/app_tab_crossfamily.tex:10` `keep12 + fresh2, 200k`（Qwen3-8B 臂）
  这是 **Qwen 臂**，不是 OLMo ladder。`paperB/P2_5_qwen_protocol_NOTES.md:54-55` 记
  `outputs/qwen3_minarch_armB_f12k2_200k/final.pt`，目录名与 tag 都是 200k。
  **我没有打开这个 ckpt 读 `step` 字段核实**（在 zwfy6，47GB）。建议 MAIN 让人核一次，
  因为 OLMo 侧刚发现"目录名/表格标称 ≠ 真实 step"，同类风险在 Qwen 侧不能靠命名排除。

**非 tex 元数据（不影响 PDF，但影响交接与 artifact 一致性）**：
| 文件:行 | 原文 |
|---|---|
| `paperB/SUBMISSION_STATUS.md:7` | "Table 4 reports trained endpoints at 200k" |
| `paperB/TODOList.md:267` | "所有深度阶梯 arm 均已完整运行至 200k step，论文 Table 4 的 Budget 列统一为 200k" |
| `paperB/TODOList.md:275` | "**深度结论边界**：所有臂统一完成 200k 训练，可按 endpoint 直接比较" |
| `paperB/main.aux:567` | 陈旧 caption（重编译自动消失，不用手改） |

⚠️ `app_tab_ppl.tex` 的脚注措辞我建议 MAIN 再看一遍：git 历史里它曾写
"use the unified 200k endpoint convention"（见 `main.aux:1091` 陈旧缓存），
现在已改成如实版本。**"unified 200k endpoint convention" 这个说法本身是委婉化的假陈述**
（把"没跑到 200k"说成"采用 200k 端点约定"），如果 submission_source*/ 或 review_history/
里还有旧 PDF 带这句，rebuttal 时可能被拿来对比。

### 1.2 reviewer 会怎么看"不等预算的 depth ladder"？

**会问，但不致命，前提是自己先说。** 三个层次：

1. **不说 → 严重**。审稿人只要下载 artifact 数 `ckpt_step`，就会发现表里标 200k 而 meta 是 83500。
   这从"设计局限"升级为"misreporting"。这是目前 4 处附录残留的真实风险。
2. **说了但辩解 → 中等**。写成"预算不同但结论不变"会招来"你怎么知道不变"。
3. **说了并把不等预算变成分析对象 → 可接受甚至加分**。见 1.3。

### 1.3 ★ 不等预算是否削弱 Table 4 想论证的东西？—— **分情况，需要逐对拆，不能一句话下结论**

Table 4 想论证 depth↓ → capability↓ 单调。我把每一对的"预算方向"和"深度方向"对齐关系算出来了：

真实数据（core6 = 干净 v2 单协议；MMLU 同源 `_v2_know`）：

| arm | 层数 | steps | % of 200k | core6 | MMLU |
|---|---:|---:|---:|---:|---:|
| keep8 | 10 | 121000 | 60.5% | 0.523284 | 0.254451 |
| keep10 | 12 | 83500 | 41.8% | 0.529988 | 0.271685 |
| keep12 | 14 | 124000 | 62.0% | 0.568880 | 0.271685 |
| keep14 | 16 | 200000 | 100.0% | 0.595324 | 0.317975 |

逐对分析：

| 对 | budget ratio | 更深的是 | 预算更多的是 | 判定 | Δcore6 | ΔMMLU |
|---|---:|---|---|---|---:|---:|
| keep8 vs keep10 | 1.449 | keep10 | **keep8** | **ADVERSE（强化结论）** | +0.67pp | +1.72pp |
| keep8 vs keep12 | 1.025 | keep12 | keep12 | **NEAR-MATCHED（几乎干净）** | +4.56pp | +1.72pp |
| keep10 vs keep12 | 1.485 | keep12 | keep12 | ALIGNED（混淆） | +3.89pp | 0.00pp |
| keep8 vs keep14 | 1.653 | keep14 | keep14 | ALIGNED（混淆） | +7.20pp | +6.35pp |
| keep10 vs keep14 | 2.395 | keep14 | keep14 | ALIGNED（混淆） | +6.53pp | +4.63pp |
| keep12 vs keep14 | 1.613 | keep14 | keep14 | ALIGNED（混淆） | +2.64pp | +4.63pp |

**MAIN 的直觉方向对，但只对一对，而且是最弱的一对。** 具体：

✅ **MAIN 说的"keep8 预算多 45% 却更低 → 单调性是更强证据"确实成立**，
但**只成立于 keep8-vs-keep10 这一对**，而这一对的 Δcore6 只有 **+0.67pp** ——
是整个 ladder 里**最小**的一个 gap，而且 Phase-1 已经证明同 arch 同盘的 flip floor
能到 0.2pp 量级（keep10 wzc1 0.532168 vs H20 0.529988 = 0.218pp 跨架构差）。
所以 **"预算更多反而更差"这个最有说服力的论证，恰好落在信噪比最差的那一对上**。
我建议 MAIN **不要把它当主论证**，只当作"至少 ladder 不是单纯预算驱动"的存在性反例。

✅ **真正被低估的宝石是 keep8 vs keep12：budget ratio 1.025（121k vs 124k，差 2.5%）**。
这是**事实上的 budget-matched 对照**，而且 Δcore6 = **+4.56pp**（大 gap）。
换言之：**"固定 budget 只变 depth"这个实验其实已经存在于磁盘上了**，
就是 keep8@121000 vs keep12@124000（10L vs 14L，预算差 2.5%）。
⚠️ 这直接**推翻 MAIN 的"budget-matched clean experiment 不可行（没有任何共同 step）"**：
MAIN 找的是"完全相同 step"，但 **2.5% 预算差在 52.4B token 尺度上已经足够称 matched**
（论文自己已把 eff_bs=128 的 nominal token 当作预算单位，2.5% = 0.8B token）。
这是一个**零成本、已在磁盘、可立即写进论文**的正面结果，价值远超把三个 arm 补到 200k。

❌ **但 4/6 对是 ALIGNED（深度和预算同向）**，包括所有涉及 keep14 的对。
所以**整条 ladder 的单调性确实被预算混淆**，不能声称"depth 单调降低 capability"。
诚实的表述只能是：ladder 是 endpoint inventory（论文 `05_analysis.tex:32` 已经这么写了，
是对的），**外加一句 keep8-vs-keep12 的 budget-matched 子对照**。

**结论**：不等预算**削弱"整条 ladder 单调"**，但**不削弱**（甚至可以支撑）
一个更窄、更强的命题：在预算相差 2.5% 时，10L 比 14L 低 4.56pp core6。
A 选项如果配上这个重构，**论文实际变强而不是变弱**。

---

## 2. 选项 B：resume 三个 arm 到 200k 重跑 eval

### 2.1 成本（全部用实测 s/step，非估算）

| arm | 起点 | 剩余步数 | s/step（实测来源） | 墙钟 |
|---|---:|---:|---:|---:|
| keep8 | 121000 | 79000 | 5.78（`logs/olmo2_7B_keep8_resume_73.log`, .73） | 126.8 h = **5.3 天** |
| keep10 | 83500 | 116500 | 6.79（今天 .82 live log） | 219.7 h = **9.2 天** |
| keep12 | 124000 | 76000 | 7.91 / live 7.86（今天 .104） | 167.0 h = **7.0 天** |

- **三节点并行的关键路径 = keep10 的 9.2 天**（.82 上 live log @step83620，剩 219.5 h）。
- 串行总计 513.6 GPU·h·8卡 = 21.4 天。
- eval 追加成本很小：从 `logs/p24_eval_ladder_prev2_73.log` 时间戳实测，
  单臂 PPL+core6+know5+mmlu-content 全套 = **02:28:05 → 02:37:52，约 10 分钟/臂**（8×H20）。
  三臂 ≈ 0.5 h。**eval 不是瓶颈，训练是。**
- 机会成本：占用 zwfy6 三台 H20 共 9.2 天。按 memory 里
  `[[h20-paperA-over-paperB-priority]]`（用户 2026-08-01 指令：三台 H20 长期先跑 PaperA），
  **这 9.2 天直接违背该优先级**，除非用户改口。

### 2.2 ★★ B 的致命问题不是 keep8 不对称，而是**三个 arm 全都不忠实**

**先纠正 MAIN 的前提**：MAIN 担心的"keep8 无 optimizer_state 的不对称"**不存在**。
Phase-1 已实测：wzc1 的 `outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt` (31.8 GiB)
带完整 optimizer/rng/train_args；zwfy6 那份 10.6 GiB 是被剥掉 optimizer 的瘦身副本。
三个 arm **对称**，都有带 optimizer 的源 ckpt。

**但真正的问题更糟**：当前 HEAD 的 param-group 结构（4 组）与 ckpt 存的（2 组）不匹配，
`AdamW.load_state_dict` 抛 `ValueError`，trainer line 895-900 catch 掉降级成 WARM-RESTART。
**这不是理论推断 —— 今天 11:56/11:57 起的两个 live run 已经掉进去了**：

```
.82  logs/olmo2_7B_keep10fresh2_resume200k.log  @step83620  lr=6.65e-05  gnorm=0.59  6.79s/step
.104 logs/olmo2_7B_keep12fresh2_resume200k.log  @step124100 lr=3.84e-05  gnorm=0.67  7.86s/step
```
keep10 原轨迹 @83540 是 `lr=1.33e-05`，现在 `6.65e-05` → **fresh cap LR 突跳 5.0×**。
两条 log 都打了 `WARM-RESTART (Adam moments re-init)`。

**所以 B 的自相矛盾比 MAIN 想的更深，且换了位置**：
- MAIN 以为矛盾是「keep8 warm-restart vs keep10/12 忠实」→ **错，三个都 warm-restart**
- 真实矛盾是「**keep14/ShortGPT 的 200k 是一条连续轨迹的 200k；补出来的 keep8/10/12 的 200k
  是"前 N 步一个 optimizer 状态 + 后 M 步 Adam 动量清零 + fresh cap LR 5× 跳变"的拼接轨迹**」。
- 补完之后表格看起来"全是 200k、齐整了"，但**齐整是假的**：ladder 从
  "step 不同但训练过程同质" 变成 "step 相同但训练过程异质"。
- **后者更难被 reviewer 发现，也更难在 limitations 里说清楚。** 这是 B 最坏的性质：
  它把一个**可见、可披露、可量化**的缺陷（step 不同）换成一个
  **不可见、难披露、无法量化**的缺陷（optimizer 断点 + LR 跳变，且各 arm 断点位置不同）。

⚠️ 而且断点位置本身就不对称：keep8 在 60.5% 处断，keep10 在 41.8% 处断，keep12 在 62% 处断。
"三个 arm 的 warm-restart 发生在训练进度的不同位置"是一个**新引入的、无法用现有数据 bound 的**不对称。

**B 是可救的但不是免费的**：Phase-1 已证明 2组→4组 flat-index 重映射是双射
（三 arm 全部 `remap_is_bijective=true` / `all_moments_preserved=true`，工具在 commit 79bf3f6）。
救 B 需要：(1) 在 trainer line 892 前插 remap；(2) resume 时显式 `--lr 2e-5 --min_lr 2e-6`
复现老 bug 的均匀 LR；(3) **kill 掉现在两条已污染的 live run 重启**（现在 kill 只损失约 30 min）。
即使这样，**dataloader 位置仍不恢复**（trainer line 936-941 只 `set_epoch` 然后从头 iter），
keep10 会重放本 epoch 已见的 83500 步数据 → "忠实续跑"在 data order 维度上仍不成立。

### 2.3 B 的额外隐患：轮转会吃掉 resume 起点

今天的 launch 用 `--keep_last_n 3 --milestone_every 5000 --keep_milestones 8 --keep_steps 150000,175000,200000`。
Phase-1 用 `select_rotation_victims` 模拟：keep12 再存 3 个 ckpt 后 step123500/124000 进 victims；
keep8 再存 3 个后 step48000/121000 进 victims。**如果 B 跑起来又想保留旧端点做 paired 对照，
必须先把 83500/124000/121000 加进 `--keep_steps` 或另存备份**，否则 A 的退路也被烧掉。
（keep10 目录现在只剩 step83000/83500，再存 4 个就全没了。）

---

## 3. 选项 C：混合（resume keep10/keep12，keep8 如实披露）

**C 的原始动机已经消失。** C 存在的唯一理由是"keep8 无 optimizer 所以只能 warm-restart"，
而这条前提被 Phase-1 证伪（wzc1 那份带 optimizer）。三个 arm 对称，**没有理由单独区别对待 keep8**。

如果 MAIN 仍在当前（未修 remap）代码下执行 C，结果是**最坏的**：

| rung | 口径 |
|---|---|
| keep8 | 121k，连续轨迹 |
| keep10 | 200k，含 41.8% 处 warm-restart + LR 5× 跳 |
| keep12 | 200k，含 62% 处 warm-restart + LR 5× 跳 |
| keep14 | 200k，连续轨迹（历史上也有 resume，但同代码同 bug，所以与 ladder 内部可比） |

**三个 rung 三种口径**，而且 caption 得同时解释"为什么 keep8 停在 121k"和
"为什么 keep10/12 的 200k 和 keep14 的 200k 不是一回事"。
更糟的是 **C 恰好破坏了 §1.3 发现的那颗宝石**：
keep8@121000 vs keep12@124000 的 budget-matched 对照（ratio 1.025）会因为
keep12 被推到 200k 而**消失**（变成 121k vs 200k，ratio 1.653，ALIGNED 混淆）。

→ **C 严格劣于 A，且劣于修好 remap 后的 B。C 应直接排除。**

---

## 4. 推荐

### 4.1 推荐：**A+（如实披露 + 提取已有的 budget-matched 对照 + 修 partial merge），并 kill 两条 live resume**

理由，按权重排序：

1. **A 的成本已经被前人做掉了 90%**（§0.1）。剩 4 处附录 + 3 处元数据，是**小时级 CPU 工作**。
   B 是 9.2 天占三台 H20，还违反用户的 PaperA 优先级。成本比约 1:200。
2. **B 换来的"齐整"是假的**（§2.2）。它用不可见的 optimizer/LR 异质性换掉可见的 step 异质性，
   是**审稿风险上升而非下降**。而且要救它必须先改 trainer + kill 重启，B 的真实成本还要加上
   代码改动 + 重新验证的风险。
3. **A 配上 §1.3 的 keep8-vs-keep12 对照后，论文实际变强**。
   这个 budget-matched 证据零成本、已在磁盘，且**直接回答 reviewer 最可能问的那个问题**
   （"你怎么知道不是训练时长而是深度？"）。B 反而会**毁掉**这个对照（§3）。
4. **有一个比 budget 更紧急的缺陷**（§0.2 keep12 ARC-E 6/8 shard）。
   优先级应该是：先修数字错的（partial merge）→ 再修标签错的（4 处 200k）→
   预算不等只需披露。B 完全不解决 partial merge（补到 200k 也要重新 eval，
   而重新 eval 才是修 partial merge 的动作 —— 但那 10 分钟的 eval **不需要**先花 9.2 天训练）。
5. **不等预算已经被论文自己披露了三处**（method/analysis/limitations），
   审稿人会看到的是"作者主动交代了预算不匹配并给出 endpoint-inventory 定性"，
   这在 measurement-audit 类论文里是**符合其自我定位的**（这篇论文的卖点就是"审计")。

### 4.2 如果 MAIN 选 A+，接下来具体要做什么

按顺序，前 3 步是必须的：

**Step 1（数字正确性，优先级最高，~10 min GPU 或 0 GPU）**
把 Table 4 的三个 shallow 行从 v1 目录切到**干净 v2 单协议目录**：
- keep8 → `zwfy6/olmo2_downstream_results/7B_keep8_step121000_v2(+_know)`
- keep10 → `..._7B_keep10_step83500_v2(+_know)`
- keep12 → `..._7B_keep12_step124000_v2(+_know)`
这一步同时消除 ARC-E 的 6/8 merge。**不需要 GPU**（结果已在盘上）。
需要改的单元格 delta（我实测，`paper → cleanV2`，pp）：
```
keep8 : hella -0.03  arc_c +0.35  arc_e +0.29  piqa -0.21  winog -0.93  obqa +0.20
        lamb +0.02  boolq -0.14  csqa +0.42  siqa -0.14  mmlu +0.10
keep10: hella -0.03  arc_c -0.25  arc_e +0.42  piqa -0.02  winog +0.18  obqa -0.40
        lamb +0.04  boolq -0.17  csqa -0.19  siqa +0.01  mmlu -0.01
keep12: hella -0.09  arc_c +0.45  arc_e +0.46  piqa +0.08  winog -0.16  obqa +0.40
        lamb -0.08  boolq +0.08  csqa -0.46  siqa +0.35  mmlu -0.35
```
⚠️ 最大单点变化是 keep8 winogrande −0.93pp 和 keep12 MMLU −0.35pp。
**keep12 MMLU 从 .2752 掉到 .2717 后，与 keep10 的 .2717 完全相等** ——
`app_tab_mmlu.tex:23-24` 那句 "keep10/keep12 ... are above [chance]" 仍成立
（Wald CI 下界 .2644/.2678 都 > .25），但"keep12 > keep10"的排序会消失，
**任何依赖 keep12 MMLU 高于 keep10 的措辞都要改**。这必须一起处理，否则会前后矛盾。
⚠️ 切换后 Table 4 的整表会锁在 **zwfy6/H20**，而当前 5 个非 shallow 行有些来自
`paperB/data/raw/`（无法确认盘）。**整表必须同源同盘**，这步要一次做完，不能只切 3 行。

**Step 2（标签正确性，0 GPU）**
改 §1.1 表里列的 4 处 tex：`app_tab_protocol_controls.tex:12,13,14,50` + caption 32,60；
`app_tab_keep8.tex` 列头 9 + 数据 11-21 + caption 26-27,31。
同步 `SUBMISSION_STATUS.md:7`、`TODOList.md:267,275`。
重编译让 `main.aux` 自然更新。

**Step 3（把缺陷变成贡献，0 GPU）**
在 `05_analysis.tex` 加 2-3 句 budget-matched 子对照：
keep8@121k(10L) vs keep12@124k(14L)，预算差 2.5%，Δcore6 = +4.56pp。
明确写这是 ladder 里唯一近似等预算的一对，其余 4/6 对深度与预算同向因而混淆。
⚠️ 措辞里**不要**说"这证明 depth 因果"——它仍是 n=1 单 run，且两 arm 的
per-device micro-batch 配置不同（Phase-1 发现 keep8@121000 的 train_args 是 bs16/ga1 而
keep10@83500/keep12@124000 是 bs4/ga4，eff_bs 都 128）。这一点得如实提一句。

**Step 4（可选，核实）**
核 `app_tab_crossfamily.tex:10` 的 Qwen f12k2 真实 step（读 ckpt 的 `step` 字段）。

**Step 5（GPU 运维决定，需要 MAIN 立刻拍板）**
`kill -9 1372221`（.82 keep10）和 `kill -9 3429286`（.104 keep12）。
现在 kill 只损失约 30 min（keep10 已跑到 83620，即 120 步；keep12 到 124100，即 100 步）。
腾出的两台 H20 按 `[[h20-paperA-over-paperB-priority]]` 回补 PaperA。
**不要用 pkill**（memory `[[kill-remote-gpu-job-by-pid-not-pkill]]`，且会误杀 eval 进程）。
⚠️ 如果 MAIN 决定**保留** B，那必须先改 remap + 显式 `--lr 2e-5 --min_lr 2e-6` 重启，
并把 83500/124000/121000 加进 `--keep_steps` 防轮转吃掉退路（§2.3）。

### 4.3 什么情况下我会改推荐为 B

只有一种：**如果用户/审稿意见明确要求"所有 depth ladder arm 必须等 step"**，
且愿意接受 9.2 天 + 违反 PaperA 优先级。即使那样，也必须**先修 remap 再重启**，
否则得到的是带 5× LR 跳变的假齐整。
**在没有外部硬要求的情况下，B 的成本效益是负的。**

---

## 5. Table 4 caption / footnote 措辞草案

### 5.1 英文版（可直接用）

> **Budget note.** The four continued-pretraining endpoints that reach the full
> schedule (keep14, ShortGPT-16, frozen-front, random-init) are evaluated at 200k
> optimizer steps (52.4B nominal token presentations at effective batch 128 and
> length 2,048). The three shallower rows are evaluated at the deepest checkpoint
> retained by our rotation policy: keep8 at 121k, keep10 at 83.5k, and keep12 at
> 124k steps; intermediate checkpoints were deleted by checkpoint rotation and
> cannot be recovered. Perplexity was still decreasing at each of these
> checkpoints, so the shallow rows are lower bounds on what their depth can reach
> under this recipe. Consequently the depth ladder is an inventory of retained
> endpoints, not a step-, token-, or FLOP-matched depth comparison, and we do not
> claim a budget-controlled monotone depth effect. Two observations bound the
> confound in opposite directions. First, keep8 (10 layers, 121k steps) and keep12
> (14 layers, 124k steps) differ by only 2.5% in optimizer steps, and keep12 is
> 4.56 points higher on the core-six macro-average; this near-matched pair is the
> one place where depth is separated from budget. Second, keep8 received 45% more
> steps than keep10 (121k vs 83.5k) yet scores 0.67 points lower on the same
> aggregate, so the ladder ordering is not produced by training time alone. We
> report both because the remaining four of six pairs vary depth and budget in the
> same direction and therefore cannot separate them.

**更短的表格脚注版**（若空间紧张）：

> $^{\dagger}$Evaluated at the deepest retained checkpoint (keep8 121k, keep10
> 83.5k, keep12 124k), not at 200k; rotation deleted the intermediate
> checkpoints. PPL was still decreasing, so these rows are lower bounds. The
> ladder is thus an endpoint inventory rather than a matched-budget depth
> comparison. Note keep8 and keep12 differ by only 2.5% in steps (121k vs 124k)
> with a 4.56-point core-six gap, and keep8 has 45% more steps than keep10 yet
> scores lower.

### 5.2 中文版

> **预算说明。** 四个跑满完整 schedule 的续训端点（keep14、ShortGPT-16、
> frozen-front、random-init）在 200k optimizer step 处评测（effective batch 128、
> 长度 2048，对应 52.4B nominal token presentation）。三个更浅的行在我们
> checkpoint 轮转策略所保留的最深 checkpoint 处评测：keep8 为 121k，keep10 为
> 83.5k，keep12 为 124k step；中间 checkpoint 已被轮转删除且无法恢复。在这三个
> checkpoint 处 perplexity 仍在下降，因此这三行是该深度在本配方下所能达到水平的
> **下界**。据此，本深度阶梯是"已保留端点的清单"，而**不是** step / token / FLOP
> 对齐的深度比较，我们**不**声称存在受预算控制的单调深度效应。两个观察从相反方向
> 界定了这个混淆：第一，keep8（10 层，121k step）与 keep12（14 层，124k step）的
> optimizer step 仅相差 2.5%，而 keep12 在 core-six 宏平均上高 4.56 个点 ——
> 这是阶梯中唯一一处深度与预算被分离开的配对；第二，keep8 比 keep10 多了 45% 的
> step（121k vs 83.5k），却在同一聚合指标上低 0.67 个点，说明阶梯的排序不是单纯由
> 训练时长产生的。我们同时报告这两点，因为其余 4 组配对中深度与预算同向变化，
> 无法据以分离二者。

### 5.3 措辞设计说明（为什么这样写不自我削弱）

- **不用"unified 200k endpoint convention"这类委婉语** —— 那是把假陈述包装，
  一旦被查 artifact 反而更糟。直接说 121k/83.5k/124k。
- **"lower bounds" 是把缺陷转成方向性信息**：预算不足只会让浅层看起来更差，
  而论文的结论方向本来就是"浅层更差"，所以**缺陷与结论同向 → 结论保守而非夸大**。
  这是唯一诚实又不削弱的框法，且它是真的（PPL 仍在下降有 `app_tab_ppl.tex` 支撑）。
- **主动给出两个 bound**（2.5% near-matched 的大 gap + 45% adverse 的小 gap），
  抢在 reviewer 之前完成"混淆分析"，把 defect 变成 analysis contribution。
- **明确写"不声称受预算控制的单调深度效应"** —— 主动收窄声明范围，
  堵掉 reviewer 最容易攻的点，代价是放弃一个本来也支撑不住的强声明。
- **不提 warm-restart / LR 跳变** —— 因为在 A 路线下 keep8/10/12 都是连续轨迹，
  没有这个问题。（若走 B，caption 必须额外交代，那会长得多也难看得多，这本身也是 A 优于 B 的理由。）
