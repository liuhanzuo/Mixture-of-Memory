# ★ Paper B 全部 arm 的差分 LR 从未生效 —— `_classify_param` 缺 `module.` 剥离，fresh tail 一直按 2e-5 训练

**日期**: 2026-08-08 ~12:1x CST。**发现路径**: subagent `a0c95e03` 在查 resume 的 param-group
不匹配时定位到根因；**MAIN 独立复核后发现影响面比 agent 报告的更大**。
**这条影响 Paper B 每一个 arm 的方法描述，不只是 resume。**

## 根因（已实测）

`build_param_groups` 在 **DDP wrap 之后**调用，此时 `named_parameters()` 的 name 全带
`module.` 前缀。修复前的 `_classify_param` 里 `name.startswith("model.layers.")` 因此
**永远为 False**，所有参数落入 `inherited` 桶，`fresh_*` 两组为空被过滤掉。

修复 commit = **`7a330ce`（2026-08-03 03:12）**，加入：
```python
if name.startswith("module."):
    name = name[len("module."):]
```

> ⚠️ **agent 报告把修复 commit 写成 `d98177c`（08-05），这是错的。** `d98177c` 加的是
> `--random_trunk`。真正加 `module.` 剥离的是 `7a330ce`。MAIN 用
> `git log -S 'module.'` 核实。agent 的其余分析成立。

## 影响面：不是三个 arm，是全部（MAIN 实测，超出 agent 报告范围）

我从**训练 log 里 `[optim] group ...` 的实际打印**统计（比读 ckpt 更权威，这是训练时真实分组）：

| 训练 log | `fresh_*` 组出现 | `inh_*` 组出现 |
|---|---:|---:|
| `olmo2_7B_keep8fresh2.log` | **0** | 6 |
| `olmo2_7B_keep10fresh2.log` | **0** | 6 |
| `olmo2_7B_keep12fresh2.log` | **0** | 10 |
| `olmo2_7B_keep16fresh2.log` | **0** | 2 |
| `olmo2_7B_keep14fresh2_freezefront.log` | **0** | 4 |
| `olmo2_7B_keep14_distill.log` | **0** | 6 |
| `olmo2_1B_keep7fresh2_16card_node0.log` | **0** | 2 |
| `olmo2_1B_keep7fresh2_1node.log` | **0** | 2 |
| — | | |
| `olmo2_7B_keep10fresh2_resume200k.log`（今天） | **2** | 2 |
| `olmo2_7B_keep12fresh2_resume200k.log`（今天） | **2** | 2 |

**每一个历史 arm 都只有 2 组、`base_lr=2.00e-05`。** 7B keep8/keep10/keep12/keep16、
freeze_front、distill、1B keep7 —— 全部。也就是说：

**Paper B 里所有 prune-then-heal arm 的 fresh tail block 从来没有用过 `lr=1e-4`，
而是和 inherited backbone 一样的 `2e-5`。整个 ladder 实际是均匀 LR。**

CLAUDE.md 里已经记过 distill trainer 的同款 bug（"#99 实际是均匀 2e-5，不得声称差分 LR"），
**但当时以为只是 distill 版；实测表明 `train_olmo2_arch_probe2.py` 在 08-03 之前同样中招，
所以整个 keepN ladder 都是均匀 LR。** 好消息是**所有 arm 同 bug 同行为，因此彼此可比**；
坏消息是**论文若写了"fresh block 用更高 LR"就是错的，必须改**。

## 对今天两个 resume run 的后果：已 kill

resume 用的是**修复后的 HEAD**，所以它们**产生了 4 组**：

```
[resume keep10, 今天 11:59]
  group fresh_decay:   815.8M params  base_lr=1.00e-04   <-- 816M 参数按 5× LR 训练
  group fresh_nodecay:   0.0M params  base_lr=1.00e-04
  group inh_decay:    2434.8M params  base_lr=2.00e-05
  group inh_nodecay:     0.2M params  base_lr=2.00e-05

[历史 keep10 原始训练]
  group inh_decay:    3250.6M params  base_lr=2.00e-05   <-- 全部按 2e-5
  group inh_nodecay:     0.2M params  base_lr=2.00e-05
```

**所以 resume 不是"补齐预算"，是换了训练配方**：816M 参数的 fresh tail 会以历史 5 倍的 LR
继续跑 7-9 天。产出的 200k 点与 ladder 上任何其他点都不同源，无法放进同一张表。

**已 kill 两个 run**（`.82` keep10 / `.104` keep12，各 41 个 PID，`kill -9`，确认 0 残留）。
**未产生污染 ckpt** —— resume 只跑了约 4 分钟，`save_every=500` 未触发；两个目录里最新的
仍是 8-02 的 `step83500.pt` / `step124000.pt`，原始 ckpt 完好。这是运气，不是设计；
下次改配方前应先确认 save 间隔。

## 对 #192 的最终影响

选项 B（resume 到 200k）**在当前代码状态下不可用**，理由有两层，第二层是新的：
1. optimizer state 无法加载 → warm-restart（Adam moments 清零）
2. **更严重**：修复后的分组让 fresh tail 拿 5× LR → **配方变了**，不再是同一个实验的延续

要让 B 可行，必须先决定"**补跑要复制历史的 bug 行为（均匀 2e-5）还是用修好的差分 LR**"：
- 复制 bug（`--lr 2e-5` 让 fresh 组名义 LR 与 inh 一致，或临时还原旧分组）→ 与 ladder 同源，
  但等于故意保留一个已知 bug 的行为
- 用修好的差分 LR → 与 ladder 不同源，只能作为独立的新臂

**因此我倾向选项 A（如实披露真实 steps）更强了**，理由和之前一样但更硬：
keep8（10 层，121k 步）比 keep10（12 层，83.5k 步）多 45% 预算，**却 core6 更低**
（.52328 vs .52999）。ladder 单调性在"更浅的还多吃 45% 算力"条件下依然成立，
**这比等预算 ladder 更强的证据**。

## 必须做的写作修正（无论 #192 怎么定）

> ### ✅ 已核实：**论文本来就写对了，不需要修正。** （MAIN, ~12:2x CST）
>
> 我先写了下面那份"必须 grep 全 paperB 改掉差分 LR 声明"的待办，然后去核实，
> 发现**论文早已如实披露此事**，措辞甚至点명了根因：
>
> `paperB/sections/03_method.tex:31-35`:
> > "Executed keep14 and frozen-front training used peak/minimum LR
> > $2\times10^{-5}/2\times10^{-6}$ **for every trainable parameter because the
> > historical distributed parameter grouping assigned all trainable tensors to
> > the inherited group.** Random-init used $10^{-4}/10^{-5}$. ShortGPT and full32
> > used $2\times10^{-5}/2\times10^{-6}$."
>
> `paperB/sections/08_appendix.tex:125-128` 有对应的 appendix 版本。
> `tab_main_results.tex` 的 LR 列全部是 `2\times10^{-5}`，唯一的 `1\times10^{-4}`
> 是 Random-init 行 —— 而该臂本来就是全随机 16L / 全部参数皆 fresh，所以那个值**是对的**。
>
> **所以本文件的价值不在"发现论文写错了"，而在**：
> (a) 把影响面从"keep14 + frozen-front"扩到 **实测的全部 arm**（keep8/10/12/16、distill、1B keep7），
>     论文目前只点名了 keep14 与 frozen-front；appendix 若要完整，可把 ladder 各 rung 一并纳入同一句话。
> (b) **定位了具体根因与修复 commit**（`7a330ce`, 2026-08-03 03:12，`module.` 前缀剥离），
>     论文现在的措辞是"historical distributed parameter grouping"，可以更精确。
> (c) **解释了今天 resume 为何不可用** —— 修复后的 HEAD 会给 fresh tail 5× LR，配方变了。
>
> 下面原始待办**已作废**，保留以显示我的判断过程和纠正。

~~- [ ] **grep 全 paperB/，凡声称"fresh block 用更高学习率 / differential LR / lr_fresh"的地方全部改为"均匀 2e-5"**，
      并如实说明 `lr` 参数在这些 run 里对 fresh 组是 no-op~~ → **不需要，论文已写对**
- [ ] （可选、低优先）把 03_method 与 08_appendix 里那句话的覆盖面从"keep14 and frozen-front"
      扩到 ladder 全部 rung，并把"historical distributed parameter grouping"精确到
      `module.`-prefix 分类 bug（fixed in `7a330ce`）
- [ ] 检查 release repo `perplexity-heals-knowledge-lags/` 与 `versions/*.md` 里的同类声明
      是否也已如实披露（**未查**）
- [x] 好消息可明说：**所有 arm 同 bug 同行为 → ladder 内部可比性不受影响**，
      这一点论文已通过统一的 `2\times10^{-5}` LR 列体现


## 待评估（未做）

subagent 给了候选 A（manual param-group remap，~40 行，已验证 135 个 param 全部 shape 匹配，
可 100% 恢复 Adam moments）。**但即使 remap 成功，上面第 2 层问题仍在** ——
remap 恢复的是 2-group 的 moments，而 rebuild 出的是 4-group 结构，语义不对应。
要真正忠实续跑，需要的是**旧分组 + 旧 LR**，即候选 B（agent 判为"等于保留 bug"，不推荐）。
所以：**忠实 resume 需要故意复现 bug 行为**，这个取舍必须由用户在 #192 里定。

## Provenance

- 修复 commit: `7a330ce`（2026-08-03 03:12），`scripts/train_olmo2_arch_probe2.py`
  加 `module.` 剥离；当前 HEAD 该逻辑在 `:438-439`
- 训练 log（zwfy6）: `logs/olmo2_7B_keep{8,10,12,16}fresh2.log`、
  `logs/olmo2_7B_keep14fresh2_freezefront.log`、`logs/olmo2_7B_keep14_distill.log`、
  `logs/olmo2_1B_keep7fresh2_*.log` —— 全部 `fresh_*` 组出现 0 次
- resume log（今天，已 kill）: `logs/olmo2_7B_keep{10,12}fresh2_resume200k.log`
- agent 报告: `status/PAPERB_CKPT_RESUME_MECHANISM.md`（含候选 A 完整补丁）
- 相关: `status/PAPERB_RESUME_WARM_RESTART_DEFECT.md`（本文件的前身，只覆盖 optimizer 层）、
  `status/PAPERB_TABLE4_BUDGET_DEFECT.md`、CLAUDE.md 里 distill trainer 同款 bug 的记载
