# GPU_STATUS.md — 5 节点 GPU 台账（40 卡）
> 每次启动/kill GPU 任务更新。heartbeat 先读→对照 nvidia-smi→台账说跑但空=补卡。★29.162.226.120=dllm 绝不碰。
> 2026-08-08 15:03 更新：用户指令「B200跑resume，H20跑新方向」→ Paper B resume 迁移到 .21/.73。

## 🔄 2026-08-14 20:00 — 40/40 忙、五臂全健康；paperC round_00 三门全绿已开跑评审

| 节点 | 任务 | 实测 | 状态 |
|---|---|---|---|
| **LOCAL** | SparseForge **noslorb** | iter **7192/7500** (95.9%)、剩 308、**54.53**(200-iter)/**53.57**(400-iter) s/it、8×111-117 GB @100%、ETA **4.67 h** | ▶️ 健康 |
| **`.212`** | SparseForge **slorb** | iter **6986/7500** (93.1%)、剩 514、**53.47**/**53.36** s/it、8×114-121 GB @98-100%、ETA **7.63 h** | ▶️ 健康 |
| **`.73`** | Paper B **keep12** | step **167120/200000**、loss 2.4339、ppl 11.40、7.81 s/step、96.4 GB @100% | ▶️ 健康 |
| **`.82`** | Paper B **keep8** | step **132500/200000**、loss 2.5434、ppl 12.72、5.78 s/step、78.5 GB @100% | ▶️ 健康 |
| **`.104`** | paperC heal | step **32920/200000**（primary read-out **121000**）、loss 2.7140、ppl 15.09、5.74 s/step、78.8 GB @100% | ▶️ 健康（勿动） |

Monitor **http200 OK**（5/5 节点 × 8 卡）。5 份 log **0 error**。每节点 8 个 compute-app、**一卡一主**无抢卡。

### ✅ 速率测量再简化：tqdm 自带累计 elapsed，**一次读取**就能算可公度窗口

不用再隔 120 s 采两次。tqdm 每行是 `| 7188/7500 [7:08:40<4:16:38, 49.35s/it, ...]` ——
**`[` 后第一个时间是累计 elapsed**。解析 (iter, elapsed) 对、在**精确** 200/400-iter 跨度上做差即可，
比"隔两分钟采两次"既快又准（窗口保证是 100 的整数倍，而不是近似）。
交叉验证：noslorb 54.53 vs 53.57（差 1.8%）、slorb 53.47 vs 53.36（差 0.2%），与 19:27/18:57 的 ~53.5 一致。

> ⚠️ 顺带纠正一个**长期错误基线**：`45-48 s/it` 从来不是吞吐，它是 **eval 之间**的瞬时值
> （上面 tqdm 行里的 `49.35s/it` 就是这个）。真实吞吐 ~53.5。以前拿 45-48 当基线，
> 才会一看到 53 就误以为"被抢卡拖慢"。

### ❌ 本轮两个我自己的读数错误（都当场发现并修）

**1. iter 正则锚错，返回空**：我写 `^ *N/7500`，而真实格式是 `Training:  96%|...| 7188/7500 [...]`。
**空的 grep 结果和"健康安静的 log"长得一模一样** —— 若不查，我会沿用上一轮的进度数字当本轮结果。
治法：先 `tail | tr '\r' '\n'` 看清真实格式，再写解析。

**2. ★ zwfy6 是共享盘 → `ls -t logs/ | head -1` 会跨节点串台**：三个节点只返回**两个**不同 run，
且把 `.73` 标成 paperC 的 log、`.104` 标成 keep8 的 log。
**根因**：`.73/.82/.104` 共享同一个 zwfy6 盘，`logs/` 里最新的文件是**全集群最新**，
不是"我 ssh 进去那台机器的 job"。heartbeat 手册说"按 mtime 找最新 log、别写死名字" ——
**这条对 LOCAL/`.212` 对，对共享的 zwfy6 盘错。**
治法：**从各节点自己的进程表**认任务：
```bash
pgrep -af "train_olmo2|torch.distributed.run" | grep -oE "output_dir [^ ]+" | sort -u
```
结果证明**台账一直是对的**：`.73`=keep12、`.82`=keep8、`.104`=paperC。然后按**显式文件名**读各自的 log。
> **泛化**：在共享文件系统上，**文件 mtime 只能告诉你"什么时候"，不能告诉你"在哪台机器"**。
> 节点归属必须来自进程表，不能来自目录列表。

### ✅ paperC round_00：三门全绿 → 已启动 6 审盲评（0 GPU）

| gate | 结果 |
|---|---|
| build | **PASS** — 16 页、0 undefined ref/citation、0 LaTeX error |
| numbers | **PASS** — 524 个数值全部可溯源，**0 untraceable** |
| venue | **PASS ×2** — 11/11 条目核实（OpenReview `venueid` 6 条 + ACL Anthology/DBLP 5 条） |

snapshot `e1a4f5db43c2945d`、31 文件、0 missing dependency。
**venue gate 抓到的真缺陷不是 venue 而是一个编造的标题**（`zheng2025cheating`：
"Null Models **That Always Output a Constant Beat LLM Benchmarks**" → 真实是
"Null Models **Achieve High Win Rates**"），已修 + 重新编译 + **重新冻结**（见 commit `adc127d`）。

**⚠️ 启动前先修了 workflow 自己的一个致命 bug**（commit `f429509`）：
`review_round.js:332` 声明 `const meta = await agent(...)`，与第 1 行**必需的** `export const meta = {...}`
**重名** → 每次调用都在 parse 期就死（`Identifier 'meta' has already been declared`）。
我此前只对三个 Python 脚本做了负向测试（喂坏论文、确认非 0 退出），**从没 parse 过那个 JS**。
**一个 parse 不了的 workflow 不是"评审很严格"，而是"根本没有评审"** —— 正是该文件自己的注释
警告 `.claude/agents/` 会有的静默退化。`node --check` 毫秒级就能抓到，本机有 node，**已纳入流程**。

Action: **GPU 无操作**（纯测量）；启动 paperC round_00 评审 workflow（`wf_ec538a86-f61`，CPU/API only）。

## 🔄 2026-08-14 19:27 — 40/40 忙、节拍稳定；⚠️ **撤回我前三轮的「finalize 已武装、覆盖 7 任务」**

| 节点 | 任务 | 实测 | 状态 |
|---|---|---|---|
| **LOCAL** | SparseForge **noslorb** | iter **7156/7500** (95.4%)、剩 344、**53.85 s/it**（200-iter）/ **53.59**（382-iter）、8×111-117 GB @100%、ETA **5.15 h** | ▶️ 健康 |
| **`.212`** | SparseForge **slorb** | iter **6948/7500** (92.6%)、剩 552、**53.41 s/it**（200-iter）/ **53.66**（376-iter）、8×114-121 GB @100%、ETA **8.19 h** | ▶️ 健康 |
| **`.73`** | Paper B **keep12** | step **166880/200000**、loss 2.3915、ppl 10.93、7.81 s/step、96.4 GB @100% | ▶️ 健康 |
| **`.82`** | Paper B **keep8** | step **132160/200000**、loss 2.5429、ppl 12.72、5.78 s/step、78.5 GB @99-100% | ▶️ 健康 |
| **`.104`** | paperC heal | step **32580/200000**（primary read-out 121000）、loss 2.7559、ppl 15.73、5.74 s/step、77.5 GB @99% | ▶️ 健康（勿动） |

Monitor: **http200 OK**，`latest` 里 5 节点各 8 卡。5 份 log 全部 0 error。Action: **none**（纯测量）。

### ✅ 18:57 引入的「可公度窗口」估计量本轮通过交叉验证

同一时刻用两个不同长度的窗口量同一个 run，**只要两者都是 100 的整数倍**就应该一致。实测：

| 臂 | 200-iter 窗口 | ~380-iter 窗口 | 差异 |
|---|---|---|---|
| noslorb | 53.85 | 53.59 | **0.5%** |
| slorb | 53.41 | 53.66 | **0.5%** |

对比 18:57 之前用 last-60 时，同一个 run 在相邻两轮之间能摆到 **47 ↔ 59（±11%）**。
**估计量修好了，不是运气。** 两臂真实吞吐就是 ~53.5 s/it，**45-48 s/it 那个「单跑基线」是 eval 之间的
瞬时速率，从来不是吞吐**——之前把它当基线，才会一看到 53 就以为「被抢卡拖慢」。

### ❌ 撤回：`--finalize_lm_eval True` 是**死 flag**，in-run 覆盖 **0/9 而非 7/9**

17:57 / 18:27 / 18:57 三轮我都写「收尾阶段已武装、覆盖 union-9 的 7 个任务」。**错了。**

**源码判据**（`/apdcephfs_wzc1/share_304376610/pighzliu_code/main_llama.py`，非本仓库）：

| 行 | 内容 | 后果 |
|---|---|---|
| `:2248` | `if finalization_done and args.finalize_lm_eval:` ← lm_eval 唯一入口 | **在 `while True:` 训练循环内部** |
| `:3215` | `finalization_done = True` | 置位发生在循环**之后**的收尾段 |
| `:3469` | `extra = int(args.final_finetune_iters)`；`extra<=0` 时 `else: break` | **直接跳出 `while True:`** → `:2248` 再也不会被执行 |

运行中进程实证 `--final_finetune_iters 0` ⇒ 走的正是 `break` 那条。
**经验判据**：`out_llama_tokenmatched_{noslorb,slorb}` 两个目录里 **lm_eval 文件数 = 0**；
而那个 17000-iter 的旧 run（`final_finetune_iters=3000`，真的进过收尾微调）**有** `best_lm_eval.json`。

**我错在哪**：我从**进程参数**读到 flag=True，然后**推断了行为**。
**flag 被传进去 ≠ 那段代码会被走到。** 这与今天 paperC E1（「没有 python 有 pyarrow」= 真于所查、
假于磁盘）和 18:57 的窗口伪影是**同一类**：把「我观察到的那一层」当成「我想知道的那一层」。
`_run_sparseforge_tokenmatched_union9_watcher.sh` 的文件头**从 08-13 起就写着这条**，我只读了用法段。

### ✅ 但真正的机制**是活的**，无需补任何东西（先核实，再判断）

LOCAL 上两个 offline watcher 都在跑（`.212` 上正确地没有）：

| PID | ARM | 状态 | 已存活 | 最近一次轮询 |
|---|---|---|---|---|
| **176642** | `noslorb` | `Ss` | ~5.2 h | 19:28:40 `remote training log advanced 29s ago (< 1800s); still running; waiting` |
| **176751** | `slorb` | `Ss` | ~5.2 h | 同上 |

- 轮询 **300 s**，staleness 触发阈 **1800 s**；训练一停就自动开始离线打分。
- 覆盖 **全部 9 个任务**：`boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa`
  —— 即 in-run 缺的 BoolQ + RTE **本来就在 watcher 里**，从头到尾没有缺口。
- 按臂分别处理 SLoRB：`slorb → --mask hard --slorb fold`、`noslorb → --mask hard --slorb drop`。
- PPL 在 **2048 与 4096 两个长度**上各打一遍，seqlen 写进各自的 `ppl_metrics.json`。
- **必须在 LOCAL 打分**，因为 LOCAL **就是 `.21`**，是唯一装了 pinned harness（transformers 4.57.6 /
  git `b86c479`）的机器——所有归档 arm 的 `results_*.json` 都是这个版本打的，换机器数字不可比。

⇒ **本轮不需要动作。** 我把它当缺口去查，结果发现机制已经正确且武装 ——
**「先核实再判断」这次省掉了一次多余的手工 eval，也纠正了一条我自己传播了三轮的错误结论。**
noslorb 约 **5.1 h** 后落地（≈ 08-15 00:35），watcher 会自动接手。

## 🔄 2026-08-14 18:57 — ✅ ckpt 保存已验证；⚠️ **撤回我自己前两轮的「变快了」结论**

| 节点 | 任务 | 实测 | 状态 |
|---|---|---|---|
| **LOCAL** | SparseForge **noslorb** | iter **7120/7500** (94.9%)、剩 380、**53.17 s/it**（200-iter 可比跨度）、ETA **5.6 h** | ▶️ 健康 |
| **`.212`** | SparseForge **slorb** | iter **6910/7500** (92.1%)、剩 590、**53.38 s/it**（200-iter）、ETA **8.8 h** | ▶️ 健康 |
| **`.73`** | Paper B **keep12** | step **166660/200000**、loss 2.4304、ppl 11.36、7.81 s/step、96.4 GB | ▶️ 健康 |
| **`.82`** | Paper B **keep8** | step **131860/200000**、loss 2.5734、ppl 13.11、5.78 s/step、78.5 GB | ▶️ 健康 |
| **`.104`** | paperC heal | step **32280/200000**、ppl 15.52、5.74 s/step、77.5 GB | ▶️ 健康（勿动） |

### ✅ 三个 zwfy6 run 的 ckpt 保存**已实测确认**（关闭唯一的静默失败模式）

新起的 resume 最危险的失败模式是「能训不能存」—— 会静默损失 3-5 天。实测三者都在写：
keep12 `step166500.pt` @18:36、keep8 `step131500.pt` @18:21、paperC `step32000.pt` @18:29。
zwfy6 余量 **36 T**，无磁盘风险。

### ⚠️ 撤回：17:57 与 18:27 两轮报的「SparseForge 变快到 47 s/it」是**窗口伪影**

- 本轮 last-60 读到 **59.30 / 58.34** s/it，看着像掉了 20%；
- 但换成 **200-iter 可比跨度**（每个窗口恰含 2 次 eval）：**53.17 / 53.38**，与全程均值 52.51 / 53.08 **持平**。
- **既没有变快，也没有变慢。**

**根因**：`eval_ppl` 每 100 iter 触发一次、耗时约 3:00，外加一次 `[save]`。60-iter 窗口若**恰好包含** eval
就读 ~59 s/it；**恰好错开**就读 ~47 s/it。本轮实证：noslorb 的 last-60 跨 iter **7069→7121**，
正好横跨 iter-7100 的 eval + save。**我的 47 和我的 59 是同一个伪影的两个方向。**

**正确估计量**：窗口长度取 eval 周期的**整数倍**（200/400 iter），使每个窗口含相同次数的 eval；
或直接读 eval 行上的 tqdm 稳态值（47.13 / 46.43 = **eval 之间**的速率，**不是**决定 ETA 的吞吐）。

> **今天第三次同类错误**（前两次：早上撤回的 1.32× 抢卡倍数、以及那条打印「0.0 s/it」的 VERDICT）。
> heartbeat 契约已有「单次采样不是趋势」，**缺的那条规则是：窗口长度必须与其中任何周期性事件的周期可公度。**
> 已写入本节，下一个 agent 不要再用 last-60 去量一个每 100 iter 抽一次 eval 的 run。

## 🔄 2026-08-14 18:27 — 40/40 全忙；三个 H20 job 互不干扰（各自偏离 ≤2%）

| 节点 | 任务 | 实测 | 状态 |
|---|---|---|---|
| **LOCAL** | SparseForge **noslorb** | iter **7095/7500** (94.6%)、剩 405、~~last-60 47.00 s/it~~ **← 该数字已于 18:57 撤回（见下节，eval 窗口伪影）；真实 53.17** | ▶️ 健康 |
| **`.212`** | SparseForge **slorb** | iter **6884/7500** (91.8%)、剩 616、~~last-60 47.98 s/it~~ **← 已撤回；真实 53.38** | ▶️ 健康 |
| **`.73`** | Paper B **keep12** | step **166440/200000**、loss 2.3783、ppl 10.79、**7.81 s/step**（420 步/3282 s，偏离自身基线 **−1.1%**） | ▶️ 健康 |
| **`.82`** | Paper B **keep8** | step **131560/200000**、loss 2.5515、ppl 12.83、**5.85 s/step**（540 步/3160 s，**−0.3%**） | ▶️ 健康 |
| **`.104`** | paperC Qwen3 heal | step **31980/200000**、ppl 16.30、**5.85 s/step**（7980 步/46708 s，**+2.0%**） | ▶️ 健康（勿动） |

**三个并发 8-GPU job 无相互干扰**：keep12 −1.1%、keep8 −0.3%、paperC +2.0%，全部在各自基线 ±2% 内。

**SparseForge 已稳定在 ~47 s/it**：两臂 last-60 分别 47.00 / 47.98，落在 45-48 s/it **单跑基线**内 ——
证实上一轮判断「union-9 抢卡结束」。⚠️ 全程均值（52.51 / 53.08）仍被早先抢卡期拉高，
**不可当作当前速率引用**。

### ⚠️ paperC 的完成时间必须说清是「哪个 step」（本轮更正）

实测 5.85 s/step ⇒ 到 `max_steps=200000` 还要 **11.4 天**。但那**不是决策点**：
- `HEAL_CONFOUND_PREREGISTRATION.md:87` / `:121` 明写 **primary read-out = step 121000**，
  `max_steps=200000` 只是为了对齐 comparator `olmo2_7b/keep8` 的启动配置；
- 运行中进程实证：`--keep_steps 121000`。

⇒ **真正的决策地平线 = 121000 − 31980 = 89020 步 = 144.7 h ≈ 6.0 天**（约 08-20/21 抵达）。
`SUBMISSION_GAP_AUDIT` 里「08-20 自己会到」的说法**对 121000 成立、对 200000 不成立**。
**任何引用 paperC 完成日期的地方都必须注明是哪个 step。**

## 🔄 2026-08-14 17:57 — 40/40 卡全忙，两个新起的 Paper B 臂已有实测节拍

| 节点 | 任务 | 实测 | 状态 |
|---|---|---|---|
| **LOCAL** | SparseForge **noslorb** | iter **7059/7500** (94.1%)、8×111-117 GB @100%、**last-60 47.35 s/it**（全程均值 53.04）、ETA **6.5 h** | ▶️ 健康，**且变快了** |
| **`.212`** | SparseForge **slorb** | iter **6847/7500** (91.3%)、8×114-121 GB @100%、last-60 58.32（全程 53.77）、ETA **9.75 h** | ▶️ 健康 |
| **`.73`** | Paper B **keep12** resume | step **166200/200000**、loss 2.4548、ppl 11.64、**7.81 s/step**（180 步/1406 s 实测）、96.4 GB @100% | ▶️ 健康 |
| **`.82`** | Paper B **keep8** resume | step **131240/200000**、loss 2.5406、ppl 12.69、**5.78 s/step**（220 步/1272 s 实测）、78.5 GB @100% | ▶️ 健康 |
| **`.104`** | paperC Qwen3 heal | step **31660/200000**、ppl 15.72、5.74 s/step、77.5 GB @100% | ▶️ 健康（勿动） |

**★ LOCAL 变快是真事，不是噪声**：last-60 窗口 **47.35 s/it** 已回到 45-48 s/it 的**单跑基线**区间
（全程均值 53.04 被今天早些时候的 union-9 eval 抢卡拉高）。即**抢卡结束**，不是随机波动。
按 heartbeat 契约「偏离基线 >10% 要报实测值+基线+倍数」，这里是往好的方向偏，同样报出来。

**两个新臂无相互干扰**：keep12 实测 7.81 s/step vs 瞬时 7.81、keep8 实测 5.78 vs 瞬时 5.78 —— 完全一致，
说明两个 H20 job 之间、以及与 `.104` 之间都没有争抢。

~~**收尾阶段已验证（从运行中进程的参数读，不是从 log 正文猜）**：两个 SparseForge 臂都带~~
~~`--finalize_lm_eval True` + 7 个任务（hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,race）。~~
~~⚠️ **已知缺口仍在**：in-run finalize 只覆盖 union-9 的 **7/9**，**BoolQ + RTE 需另跑**。~~

> **❌ 划掉的三行是错的（2026-08-14 19:27 撤回，证据见本文件顶部 19:27 节）。**
> `--finalize_lm_eval` 在 `--final_finetune_iters 0` 下是**死 flag**：in-run 覆盖 **0/9，不是 7/9**。
> 我读了进程的 **flag**，然后推断了它的 **behaviour**——**flag 被传进去 ≠ 那段代码会被走到**。
> 真正的机制是 LOCAL 上两个还活着的 offline watcher（19:27 节有 PID）。

pinned harness `venv_union9` 完好：**lm_eval 0.4.8 + transformers 4.57.6 + torch 2.13.0 + datasets 5.0.1**（**这条仍成立**）。

## 🔄 2026-08-14 17:35 — **16 张空闲 H20 补上 Paper B resume**（40/40 卡全忙）

**★ 台账与实测冲突，已按实测更正**：上一节记 `.73` 在跑 B02 confirmatory，实测 **8×0 MiB / napps=0**。
原因：B02 于 ~16:15 跑完，且**它自己的 kill gate 在两个长度上都触发** → `lifecycle: dead`。
于是 `.73` + `.82` = **16 张 H20** 同时空闲。

| 节点 | 任务 | 实测 | 状态 |
|---|---|---|---|
| **LOCAL** | SparseForge **noslorb** | iter **7013/7500** (93.5%)、8×111-117 GB @100%、**53.41 s/it**(113-iter)、wiki_ppl 6.6506 | ▶️ 健康 |
| **`.212`** | SparseForge **slorb** | iter **6802/7500** (90.7%)、8×114-121 GB @100%、**57.88 s/it**(112-iter)、wiki_ppl 6.1612 | ▶️ 健康 |
| **`.73`** | **Paper B keep12+fresh2 resume**（本轮新起） | step **166020/200000**、loss 2.3887、ppl 10.90、**7.90 s/step**、96.4 GB @100% | ▶️ 新起 |
| **`.82`** | **Paper B keep8+fresh2 resume**（本轮新起） | step **131020/200000**、loss 2.5240、ppl 12.48、**5.87 s/step**、78.5 GB @100% | ▶️ 新起 |
| **`.104`** | paperC Qwen3 heal | step **31300/200000**、ppl 17.57、77.5 GB @100%、5.74 s/step | ▶️ 健康（勿动） |

### 为什么这次可以跑 Paper B（三问全过，且是今天第一次真的过）

CODEBUddy 优先级铁律是「paperC / proposal 有活就不许碰 Paper B」，判据是**它们是不是真没活了**：
1. **proposal**：`ready_queue.py` 实测 **0 ready_gpu / 12 ready_cpu** —— 12 项全卡在自己的 0-GPU 前置门上，
   没有一项是 GPU-eligible。B02 已 dead，B11 收窄后判「不值得做」。
2. **paperC**：manuscript 今天已由 tcodex 产出并**通过 build + numbers gate**（16 页、525/525 数字可溯源）；
   `SUBMISSION_GAP_AUDIT` 里唯一的 GPU 项是 `~1 GPU-h` 且标注「可选、不要阻塞」。
3. **架构对**：keep8/keep10/keep12 的 ckpt + trainer + 126.9 GB 数据**全在 zwfy6（sm_90）**，
   与它们当初训练的架构一致。

### ⚠️ 发现并关闭的陷阱：既有 launch 脚本钉死了**过期 ckpt**

`launch_keep8_resume_h20_73.sh` 钉 `step121000_full.pt`、`_82.sh` 钉 `step124500.pt`、
`launch_keep12_resume_b200_21.sh` 钉 `step124000.pt`。而实测 **keep8 已在 131000、keep12 已在 166000**
⇒ 照原样跑会**静默丢掉 ~10k / ~42k 步**，并画出一条看着完全正常的 loss 曲线。
与今晨 SparseForge 的悬空 `last` symlink 是同一类陷阱。

新脚本 `launch_{keep12_73,keep8_82}_..._0814.sh`：
- **发现**最新 ckpt（`ls -t`），**断言** `>= 记录步数`，若 `>= 200000` 直接 exit 0；
- 由 **H20 实证配方**（`launch_keep8_resume_h20_73.sh`）**机械替换**生成，**15 个超参逐一断言相同**；
- **不**用 `launch_keep12_resume_b200_21.sh`（那是 B200 配方 batch 8×accum 2）——keep12 比 keep8 更深，
  97.8 GB 的 H20 上用 batch 8 有 OOM 风险；改用 batch 4×accum 4，`eff_batch = 128` 不变；
- 启动前 `napps != 0` 就拒绝启动（防两 agent 同占一节点）；
- 跨盘 `scp -O` + md5 双向核对。

实测 preflight 日志：`Resuming from DISCOVERED newest ckpt: .../step166000.pt (step 166000)` /
`.../step131000.pt (step 131000)` —— **拿到的是最新的，不是钉死的那个**。

## 🔄 2026-08-14 15:58 — heartbeat 实测（覆盖上一节的 12:34 台账数字）

| 节点 | 硬件/盘 | 任务 | 实测 | 状态 |
|---|---|---|---|---|
| **LOCAL** | 8×B200 wzc1 | SparseForge **noslorb** resume | iter **6919/7500** (92.3%)、8×111-117 GB @100%、**55.86 s/it**（128-iter 窗口）、loss 2.04、wiki_ppl 6.6329 | ▶️ 健康 |
| **`.212`** | 8×B200 wzc1 | SparseForge **slorb** resume | iter **6709/7500** (89.5%)、8×114-120 GB @100%、**56.29 s/it**（128-iter 窗口）、loss 1.53、wiki_ppl 6.1577 | ▶️ 健康 |
| **`.73`** | 8×H20 zwfy6 | **B02 confirmatory n=200**（8 个 resume_j 并行） | 8×66-67 GB、util **0-99% 抖动**、8 lane 全在 90 s 内写过 CSV | ▶️ 健康 |
| **`.82`** | 8×H20 zwfy6 | **空闲（有据）** | 8×0 MiB、napps=0 | ⏸️ 见下方判据 |
| **`.104`** | 8×H20 zwfy6 | paperC Qwen3 heal | step **30440/200000**、8×77.5 GB @100%、5.74 s/step、ppl 16.17 | ▶️ 健康（勿动） |

**★ 两个「看着像故障、实际不是」的实测（都按 heartbeat 铁律排除了）**
- **LOCAL 15:30 时读到 8×17-22 GB**（vs 现在 111-117 GB）＝ iter 6900 的 `eval_ppl` 里程碑释放了训练
  activation，**不是 OOM**（铁律 4）。同一份 log 里有 `evaluating: iter_num 6900 ... wiki_ppl 6.6329`。
- **`.73` 的 `logs/b02_confirm_main.out` mtime 落后 42 分钟** —— 它是 **launcher**，spawn 完 8 个 worker 就退出了；
  worker 直接写 CSV。改看 per-lane artifact mtime：j=3/6/13/20/27/34/41/48 **全部在 90 s 内有写入**。
  **GPU4 瞬时 0% / GPU2 23% 是 lane 间长短档差异，不是 stall**（铁律 1）。
- **速率一律用累积 tqdm elapsed 算**：两臂 128-iter 窗口都是 ~56 s/it；last-30 窗口读到 67-69 s/it 是
  **eval 里程碑落在短窗口里**的假象，不是变慢。单跑基线 45-48 s/it 是重启前无 union-9 争抢时的数字。

**`.82` 为何不补卡（三问全过，非静默留空）**
1. **proposal 有没有 GPU 活？没有。** `proposal/ready_queue.py`（今日新增 layer-2 生成器）实测
   **0 ready_gpu / 13 ready_cpu** —— 13 项全卡在自己的 0-GPU 前置门上。
2. **架构/资产对不上。** 候选是 #245 AST+SLoRB 参数匹配对照，但 `.82`（zwfy6）**根本没有 `main_llama.py`**，
   且只有 `dolmino_chunks_2048_olmo2.npy`（**olmo2 tokenizer，错的**），没有 dolmino-llama2 分词。
   SparseForge trainer 是 **wzc1-only**。要在这跑得先跨盘搬代码 + 重分词 126 GB，
   而两臂今晚就在**对的盘 + 对的架构**上跑完。
3. **无 agent 冲突**：两个后台 agent 都是 CPU-only。
⇒ 结论：**留空，并把理由写成可审计的字段**（`TRAINER_ACTIVITY.jsonl` 的 `idle_card_justification`），
不是「卡空着没人管」。

## 🔄 2026-08-14 12:34-12:50 — **节点重启后重配 + SparseForge ±SLoRB 双臂忠实 resume**（用户指令：重启因两台 B200 利用率过低）

> **★ roster 更正（我自己实测，覆盖旧记录）**：**LOCAL 就是 `28.89.19.21`**（`ifconfig` 实证）——
> 旧的「.21 远程」与 LOCAL 现在是**同一台机器**，旧 LOCAL 已不存在。唯一**新增**节点是 **`.212`**
> (`28.89.18.212`，密码 `configs/password_b200_18212.txt`)。
> **`.212` 与 LOCAL 同属一个 wzc1 物理盘**（写随机 stamp 跨节点读回验证，`df` 同为 `dop-fuse 120T/109T/91%`）
> ⇒ **两者之间无需 scp**。`.212` 上**没有** `/apdcephfs_zwfy6` 挂载。
> 两台都是**真 B200**：`sm_100` / 148 SM / 192 GB（`L20A` 只是 name 显示 bug，见 [[l20a-name-string-is-really-b200-sm100]]）。

> **⚠️ 重启把工具链也抹了（不只是 job）**：`sshpass` 在 LOCAL 和 `.212` **都已消失** → 改用 pexpect helper `/tmp/sshp.py`。
> conda env 被重置为 **Python 3.14.6 + 仅 torch 2.13.0/numpy 2.5.1**，`transformers`/`tqdm`/`safetensors`/
> `accelerate`/`wandb`/`sentencepiece` 全缺（两台一致）。已全部装回（`transformers` 5.15.0 是**大版本跳跃**，
> 已显式验证 `LlamaConfig/LlamaForCausalLM` + 整条 trainer import 链在两台都通过**才**投 GPU）。

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **LOCAL** (=`.21`) | 8×B200 wzc1 | **SparseForge noslorb resume** | `[RESUME] Resuming from iter_num=6700` + **Optimizer/Scaler state restored**，8 rank 全 sync，8×25260 MiB @100%。剩 800 iter × 44.81 s = **9.96 h ≈ 79.7 GPU-h** | ▶️ 运行中 |
| **`.212`** | 8×B200 wzc1 | **SparseForge slorb resume** | `[RESUME] Resuming from iter_num=6500` + **Optimizer/Scaler state restored**。剩 1000 iter × 48.39 s = **13.44 h ≈ 107.5 GPU-h** | ▶️ 运行中 |
| **`.104`** | 8×H20 zwfy6 | **paperC Qwen3 heal（未打扰）** | pid 3343485，step **28300/200000**，loss 2.868 ppl 17.60，5.74 s/step，maxmem 77.5 GB，已跑 1 d 22 h | ▶️ 运行中（勿动） |
| **`.73`** | 8×H20 zwfy6 | A04 shallow keep14 **已跑完**；现跑 CPU 收割 | 8×0 MiB | ✅ 空/收割中 |
| **`.82`** | 8×H20 zwfy6 | A04 shallow keep13 **已跑完**；现跑 CPU 收割 | 8×0 MiB | ✅ 空/收割中 |

**重启只杀了 6 个文件、全是 SparseForge**（`find logs -newermt '2026-08-13 20:00'` 实证：2 训练 + 2 watcher×2）。
noslorb 死在 iter 6700/7500 (89.3%)、slorb 死在 6500/7500 (86.7%)，**都带 optimizer state 可忠实 resume**。

> ⚠️⚠️ **发现并关闭了一个「静默从 0 重训」陷阱（后续 agent 必读）**：两臂的 `last` symlink **都是坏的**
> （按 `$ROOT` 相对路径写，却从 arm 目录内解析）。`main_llama.py:1570` 用 `os.path.islink()` 判断，而它对
> **悬空链接返回 True** → `realpath` 得到不存在的目录 → `isdir()` 失败 → **只打一行 `[RESUME] Warning` 就从
> iter 0 开始重训**，会白烧 ~43 h 并产出一条看着很正常的 loss 曲线。已修 symlink + 写 `last_dir.txt`，
> 且 resume 脚本**显式传 `--resume_dir`** 不依赖 symlink。
>
> ⚠️ **optimizer state 是 rank-0-LOCAL 而非 FSDP-full**（存档走 `optimizer.state_dict()`，非
> `full_optim_state_dict()`；实测 65 个 flat entry / 1,106,510,336 exp_avg 元素 vs 291 声明参数）
> ⇒ **换 rank 数就丢 momentum**。本次因为用**同样的 8-rank hybrid_sharded 拓扑**重启，实测
> `Optimizer state restored` 成功，**是完全忠实的 resume**。若将来要换卡数，必须先改存档路径。

**resume 脚本 = `scripts/_run_sparseforge_tokenmatched_resume.sh`**，由原脚本**机械字符串替换**生成（保证无超参漂移，
diff 已验证只改 log 名 + resume flag + preflight），**未改原脚本**（它被 provenance 文档引用）。
preflight 断言 resume_dir/model.pt/iter_num 范围，且**校验 ckpt 自己的 `args['SLoRB']` 与 arm 一致** ——
已做**反向测试**：拿 slorb ckpt 喂 `ARM=noslorb` 会正确报 `WRONG ARM'S CHECKPOINT` 退出。

## ✅ 2026-08-13 01:40-02:08 — paperC heal-confound **milestone 轨迹 MMLU-Pro 打分 完成**（.73 + .82，共 **5.57 GPU-h** / 授权 120）

兑现 `paperC/HEAL_CONFOUND_PREREGISTRATION.md` §10 预留的 16 卡（「其余 16 卡留给 milestone 的
offline MMLU-Pro 打分，这是真正的下一个瓶颈」）—— 该预留自 08-12 launch 起一直空置未用。
**两节点已于 02:08 全部释放（实测 8×0 MiB / 0 compute apps，可投）。**

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.73** | 8×H20 zwfy6 | heal 轨迹 `step5000,6000` + `step7000` | 01:40:59 起，各 409/408/410 s。`ALL ARMS DONE`，3/3 MERGE OK | ✅ 完成，已释放 |
| **.82** | 8×H20 zwfy6 | heal 轨迹 `step5500,6500` + **OLMo-2 `keep8@45000`** | 01:44:17 起，各 434/437/407 s，3/3 MERGE OK | ✅ 完成，已释放 |
| **.104** | 8×H20 zwfy6 | **paperC heal 训练本体（未打扰）** | pid 3343471，02:11 实测 step 7260/200000。**实测 elapsed/iter = 5.847 s/step**（非 tqdm 瞬时值；逐区间中位 5.750、最大 8.300 落在 ckpt flush）⇒ 距 prereg read-out step121000 还有 **184.8 h ≈ 7.70 d**。本次只读它的 ckpt | ▶️ 运行中（勿动） |
| **LOCAL / .21** | 8×B200 ×2 wzc1 | SparseForge #246（**未打扰**） | 本次禁用，**全程未连接** | ▶️ 运行中（勿动） |

**结论（详见 `paperC/HEAL_TRAJECTORY_READOUT_1.md`）**：10 cell 全部 `n=12032`/`0 nan`/`0 trunc`/
`chat_template is False`，并**复现**两个已归档 cell（qwen3 k8 −0.881pp p=0.0362 BELOW；olmo2 keep8@121k
−0.116pp p=0.7118 AT）。轨迹 5000→7000 **平**（Δ −0.175..−0.083 pp，p 全 >0.26）。
⚠️ **发现 read-out 统计量缺陷**：`always-<L>` 精度是**非平坦的数据集属性**（A .1166 … J .0785，跨度 3.81pp），
healed Qwen3 塌缩到 **A**（82-91% 的 item）= argmax = **floor 本身**，un-healed 塌缩到 **E**（94.5%，只出 5 个字母）
= always-E **−2.11pp** ⇒ **「AT floor」与「BELOW floor」是同一现象、只差塌缩到哪个字母，不含任何 competence**。
独立性模型 `acc_hat=Σ P(pred=L)P(gold=L)` 把**每个 damaged cell** 解释到 +0.07..+1.13pp，却在 intact 模型上
失效（+35.4/+16.3pp）。故 prereg §8 的 H_heal 判据**可被「换塌缩字母」满足**。建议在 step121000 打分前
（保持 pre-hoc）把 `modal_pred_share` + 独立性残差与每个 cell 并列报告，区分 degenerate-at-floor 与
competent-at-floor —— **0 GPU 成本**。

> ⚠️⚠️ **本次发现并已规避的真实数据丢失（后续 agent 必读）**：live 目录
> `outputs/paperC_qwen3base_heal_k8f2/` 处于 rotation 下（`keep_last_n=3 milestone_every=5000
> keep_milestones=8`）⇒ **非 5000 倍数的 milestone 会被删**。**实测**：`step5500.pt` 01:34 还在、
> **01:46 step7000 落盘时已从 live 目录消失**，仅因已 hardlink 而存活。本次打分前把 5000/5500/6000/6500
> （及后来的 7000）**hardlink** 到 `outputs/paperC_qwen3base_heal_k8f2_pinned/`（同 inode、`df` 不变、
> **0 额外字节**；rotator 只 glob 自己的 output_dir 故 pinned 免疫）。
> **规则：打 mid-run milestone 必须「先 pin 再打」**，直接读 live 目录会与 rotator 竞态、`--ckpt` 可能在
> 枚举与载入之间消失。

**两节点分工=两个独立 8 卡分片，不合 16 卡 DDP**：打分是「一 ckpt 一臂、臂内 8 shard」，臂间无通信，
不同 milestone 分给两节点是 **线性 2×**；而 prereg §10 实测 16 卡 DDP 只有 **1.10×** 且多一个 TCPStore
失败模式。milestone **交错**分配（.73=5000/6000、.82=5500/6500）使单节点故障只丢间隔点、不丢连续半条轨迹。
两节点排空即刻补任务（.73 01:55、.82 02:01），**无空转**。
> 📌 每臂仅 **0.93 GPU-h** ⇒ §10 说的「打分是下一个瓶颈」在 **GPU-小时意义上不成立**；但它确实是**唯一能
> 暴露上述 read-out 缺陷**的动作，且**有时效**（再等到 day 8 会永久丢掉 5 个 milestone 中的 3 个）。

## 🟡 2026-08-13 — `.73` 实测【空闲】8 卡 0 MiB；#99 keep14-distill resume **审查后决定不投**（花费 0 GPU-h）
> ⚠️ 本行的「.73 空闲」是 01:3x 的状态，**已被上面 01:40-02:08 的 paperC 打分任务取代**（该任务已完成、卡已释放）。

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.73** | 8×H20 zwfy6 | **空闲**（本次未占卡） | 派来 resume #99 keep14-distill heal。`nvidia-smi` 实测 GPU 0-7 全 **0% / 0 MiB**（下面 2026-08-12 那行「.73 = keep10fresh2 resume 运行中」**已过期**，该 run 已不在）。**审查后没有启动**：`--save_every 5000` + resume 起点 step5000 ⇒ 下一次落盘在 step10000 = 5000 步 × 13.11 s/step = **18.2 h wall = 146 GPU-h**，是本次 80 GPU-h 授权的 **1.8×**；80 GPU-h 只到 ~step7745，**差 2255 步落不了盘 ⇒ 必然 0 checkpoint**。08-05 那次已经这样烧掉 81 GPU-h（5000→7780，盘上仍只有 step5000.pt），07-31 同样（→5200）；再投就是**第三次全损**。详见 `status/PENDING_TASKS.md` #99 `[BLOCKED]`。**未 kill 任何进程。** | ⏸️ 空闲，待用户就 (a) 放弃 / (b) 迁 B200 / (c) 降 save_every 拍板 |

> 同时查出三个前提性问题（不改代码、仅记录）：① 任务给的「teacher」`outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` 其实是 **keep14 学生自己**（16,241,486,089 B = 4.0604B×4B fp32），真 teacher 是 HF 目录 `../models/OLMo-2-1124-7B`（32L/7.2986B，`from_pretrained` 载**目录**，喂 `.pt` 会崩）。② 差分 LR 缺陷在 zwfy6 **确认存在**（三次启动 log 只有 `inh_decay 4060.1M` + `inh_nodecay 0.3M` 两组、无 `fresh` 组 ⇒ 均匀 2e-5），且 zwfy6 版 trainer **缺** `train_olmo2_arch_probe2.py:912` 的 2→4 组 remap shim ⇒ 若补 `module.` 剥离会让 optimizer `load_state_dict` 失败降级 warm-restart、**丢 Adam 动量**，故只能选「原样均匀 LR」。③ 两盘 trainer md5 不同（LOCAL `228812e8` / zwfy6 `9e824f7d`），zwfy6 版**无** `--seed`、无 rotation flags ⇒ 照 LOCAL 写 flag 会 `unrecognized arguments` 直接崩。

## ⚠️ 节点调度规则变更（2026-08-08 15:03 用户指令）

> **用户指令**：「B200可以拿去直接跑resume 然后H20跑得比较慢 可以跑你的新方向. 其他H20你想用可以随时kill paperB的resume」
> - `.21` (L20A) → Paper B resume (keep10 等待数据传输完成后启动)
> - `.73` (H20) → Paper B keep8 resume (已运行)
> - `.82` (H20) → keep10 resume 将在 .21 确认运行后被 kill，腾出给新方向
> - `.104` (H20，已交还用户) → keep12 resume 运行中，用户可随时 kill

## ▶️ 当前在跑（2026-08-08 15:03 +08:00 更新）

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **LOCAL** | 8×L20A wzc1 | **空闲**（2026-08-12 01:39 起） | #181 `keep14fresh2_seed1234` 训练**已于 08-12 00:22 跑满 step200000**（`final.pt` + `DONE`）；随后 01:06→01:39 跑完 seed-variance eval battery（seed42+seed1234 × 6 轴，8 卡，33 min），**8 卡已释放 0 MiB** | ✅ 空闲可投 |
| **.21** | 8×L20A wzc1 | `keep10fresh2 resume` | 等待 dolmino 传输完成（ETA ~17:02 CST）；launch script PID 4516 | ⏳ WAITING |
| **.73** | 8×H20 zwfy6 | `keep10fresh2 resume`（**另一 agent 于 03:3x 接手**，commit `2796744`） | 03:32 前：task #252 的 15 臂 cross-family MMLU-Pro **重跑完成**（`MAXLEN=2048` + `use_cache=False`，02:17→03:32，8 卡 75 min）→ `ALL ARMS DONE`，15/15 MERGE OK，`n_trunc=0`／`n=12032`／`nan=0` 逐 shard 复核；卡释放后立刻被 Paper B `keep10fresh2 step86500→200000` 占用（PID 2438724，90.5GB/卡，8 卡满载） | ▶️ 运行中（非本任务） |
| **.82** | 8×H20 zwfy6 | `keep10fresh2 resume` | step83980+, 6.80s/step, 82.7GB，将在 .21 keep10 确认后被 kill | ▶️ 运行中（待 kill） |
| **.104** | 8×H20 zwfy6 | `keep12fresh2 resume` | step124220+, 7.87s/step, 91.9GB，用户管理 | ▶️ 运行中（用户控制） |

> ✅ **2026-08-12 03:55 更新（task #252 收尾）**：修完 #251 cross-family 的两个完整性缺陷并重跑 15 臂，`.73` 8 卡于 03:32 释放，**随即被另一 agent 接手跑 Paper B keep10 resume**（零空转，无需补卡）。
> 缺陷 1 = `MAXLEN=1536` 是按 **OLMo-2** tokenizer 量的（max 1226 tok），对 Llama-2（1678）/ Qwen3（1660）过小 → 10/15 cell 的 labelled option body 被左截断，且溢出集**因 tokenizer 而异**故跨家族表**不 item-matched**。修法选 **(a) 抬到 2048**（不是排除 item，那会破坏与已归档 MMLU cell 的全 n 匹配）。实测影响：**0/14 cell 结论变化**，最大 letter acc 变化 **+0.0083 pp**（一个 item），9 次 argmax 翻转全落在受影响 item 上、受影响之外 **0** 翻转。
> 缺陷 2 = `llama2_7b_base` 整臂 OOM（5/8 shard 死，guard 正确拒绝 3/8 merge）。根因是 **KV cache**：Llama-2 无 GQA（`num_kv_heads=32`×32 层 = fp32 KV **72.0 GiB** @B=48/L=1536，而 Llama-3/Qwen3 是 18.0/20.2）→ 只有 intact Llama-2 会死。修法 `use_cache=False`（teacher-forced 单次前向根本不用 cache），94 GiB → 41-50 GiB。
> ⚠️ **`n_trunc` 已从 driver 层 WARNING 升级为 scoring 脚本内的硬 assert** —— 当初正是因为它只是 warning，10 个被截断的 cell 才照样写出了 summary。
> 结论见 `paperC/evidence/POWER_WALL_VERDICT.md` §6：21/21 cell 有功效（hw 0.083-0.968 pp）；**AT-the-floor 主结论不变**；但新增两处自我纠正——below-floor **不是 MMLU-specific**（llama2/k8 p=0.0168、qwen3/k8 p=0.0362 显著低于 floor，分界像是 heal vs no-heal），且 `qwen3_8b_base/k14` 显著**高于** floor（+0.233 pp, p=0.0192）故"damage ⇒ at-or-below"是 **14/15** 而非普适。
> ⚠️ 本表 .21/.82/.104 三行是 2026-08-08 的旧状态，本次未核实（本任务只动 .73），下一个 heartbeat 请对照 nvidia-smi 重核。

## 传输进度（2026-08-08 15:02）

| 任务 | 大小 | 进度 | ETA | PID |
|---|---:|---|---|---|
| dolmino .73→.21:/dev/shm/dolmino_now15b_zwfy6.npy | 118 GB | ~1% | ~17:02 CST | local 2741463 |
| step124000.pt .73→.21 wzc1 outputs/keep12fresh2/ | 41 GB | ~1% | ~15:37 CST | local 2741464 |

## ⚡ 下一步（自动）

1. step124000.pt 传输完成 → 自动校验 md5（transfer_ckpt124k_pipe.sh 会输出 MD5_OK）
2. dolmino 传输完成 → keep10 launcher on .21 自动检测 size=126907244672，启动训练
3. keep10 on .21 首步 log 确认 REMAPPED ✓ / all base_lr=2.00e-05 ✓ / rows=15491607 ✓
4. 然后 kill .82 keep10: `kill -9 1418803`（不要 pkill）
5. 更新 GPU_STATUS.md 和 gpu_runs.jsonl

## 📋 历史已完成（2026-08-08）

- P2.4 六臂 SFT sweep 全完成（keep8/10/12/full32/shortgpt16/keep14 pre→post SFT ΔPPL）
- Within-disk floor v3（0 flips / Δcore6=+0.0000 across 4 arms）
- flip-boundary 根因定位 = torch 2.7 vs 2.13 版本差

## 🚫 .104 已交还用户（2026-08-05 15:4x，用户指令）

> **`.104`（28.83.24.104）心跳不纳管，但 keep12 resume 在上面跑，用户说可随时 kill**。heartbeat 不主动操作 .104。

---
> 旧台账（2026-08-08 06:25-08:28 的 P2.4 SFT sweep 与 within-disk floor v3 记录）已归档到 `status/GPU_STATUS_archive_20260808_0828.md`（如存在）。

---

## 2026-08-11 21:22–21:34 +08:00 — `.73` paperC gate-2 (task #248) 已完成，卡已释放

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.73** | 8×H20 zwfy6 | `gate2_mc_letter_content` (#248) | paperC gate-2 全量复现：MMLU 同口径 letter-vs-content × 6 个非 MMLU MC benchmark × 6 arm × 8 shard = 36 cell；21:22 起，21:34 完；36/36 cell 8/8 shard、n_scored==expected、n_nan=0；失败语法 grep 零命中 | ✅ 完成，GPU 0-7 已释放（0 MiB） |

- 结论：`REPLICATES_PARTIALLY_AND_NARROWS_THE_CLAIM`，详见 `paperC/evidence/SECOND_MC_BENCHMARK_VERDICT.md`。
- 上面 2026-08-08 那张表里 `.73 = keep8fresh2 resume ▶️ 运行中` 已过期：本次占卡前 `nvidia-smi` 实测 8 卡全 0 MiB、无 compute app。**未 kill 任何进程**（.73 上另一 agent 的纯 CPU jsonl 重算未受影响）。
- 结果落盘两盘：`olmo2_mc_letter_content_results/`（zwfy6 + wzc1，各 52 MB，各自校验完整）。

---

## 2026-08-11 22:19 +08:00 — `.73` paperC gate-2 CROSS-FAMILY (task #250) 启动

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.73** | 8×H20 zwfy6 | `gate2_xf` (#250) | paperC gate-2 跨家族扩展：#248 harness 原封不动跑 **非 OLMo** 三家族 × 5 arm × 6 task。arm = {llama2_7b, llama3_8b, qwen3_8b_base} × {base, k8, k10, k12, k14}，damage = **eval 期 front-N truncation（无 fresh block、无 heal）**，与 gate-1 DAMAGED 同构造故与已归档 MMLU 数字直接可比。8 shard/arm，bs=48（与 #248 同值），22:19 起 | ▶️ 运行中 |

- 占卡前 `nvidia-smi` 实测 8 卡全 0 MiB、无 compute app；**未 kill 任何进程**（.73 上另一 agent 的纯 CPU 工作不受影响）。
- 前置：`Qwen3-8B-Base` 原本 **wzc1-only**，已 `scp -O` 16 GB 到 zwfy6，12 个文件 md5 全对。⚠️ zwfy6 原有的 `models/Qwen--Qwen3-8b`（含 `Qwen3-8b-local` symlink）是 **Instruct** 模型（`eos=151645` im_end、有 chat_template、40960 ctx），**不能当 base arm 用**。
- driver `scripts/_run_mc_letter_content_crossfamily_8gpu.sh`（wzc1 写、`scp -O` 到 zwfy6、md5 `aef912ce` 双端一致），log `zwfy6:logs/gate2_xf_DRIVER.log`，结果 `zwfy6:mc_lc_crossfamily_results/`。

## 2026-08-11 22:32 +08:00 — `.73` #250 scoring完成，8 卡已释放（0 MiB，无 compute app）

- 15 arm × 6 task = **90 cell 全部 8/8 shard、`n_scored==EXPECTED_N`、`n_nan=0`**；128 个 shard/merge log 跑失败语法 grep（`Traceback (most recent call last)` / `CUDA out of memory` / `AssertionError` / `*INTEGRITY FAILURE` / `CARDINALITY FAILURE`）**零命中**。
- 判定 `REPLICATES_IN_DIRECTION_ACROSS_FAMILIES_BUT_THE_LADDER_DOES_NOT`，详见 `paperC/evidence/GATE2_CROSSFAMILY_VERDICT.md`。
- ⚠️ driver log 里每个 arm 都有一条**假的** `MERGE FAIL ...: 0/6 tasks merged` —— 是 driver 自检 `grep -c "^\[merge\]"` 锚了 `^` 而 `_log()` 会加时间戳前缀，纯 cosmetic bug，**跑完后已修**。15/15 arm 实际都成功 merge（各 6/6 `summary_<task>.json`，已 `ls` 逐个核对）。
- 后续 nulls/统计是 **CPU-only**，在 LOCAL(wzc1) 跑（MMLU cross-family per-item 记录 190MB 只在 wzc1），未占任何 GPU。
- **未 kill 任何非本任务进程**；.73 上另一 agent 的纯 CPU 工作不受影响。

---

## 2026-08-11 23:51 – 23:55 +08:00 — `.82` A03 seed45 四轴 eval (task #243) 已完成，卡已释放

| 节点 | 硬件 | 任务 | 细节 | 状态 |
|---|---|---|---|---|
| **.82** | 8×H20 zwfy6 | `A03_1B_dataorder_seed45_step220000` (#243) | A03 dataorder 第三个（也是最后一个）pre-registered sampler seed 的四轴 eval：mmlu_content + popqa/triviaqa + nq_open，8-way sharded GPU0-7。23:51:01 起，23:54:46 全完（MMLU 81s / CB(pt) 104s / CB(nq) 40s） | ✅ 完成，GPU 0-7 已释放（0 MiB） |

- **占卡前实测**：`.82` 8 卡全 0 MiB、无 trainer/watcher 残留进程（seed45 训练已于 23:29:10 由自己的 watcher 停在 step220000，wrapper 已退出）。**未 kill 任何进程。**
- **ckpt 完整性（关键，因为 driver 带的是 v1 停止竞态守卫）**：`step220000.pt` 与 `step205000/210000/215000.pt` **字节数完全相同 = 12,181,311,650 B（delta +0 B）**；ext driver 自带的 `torch.load(weights_only=False)` 探针独立返回 ok。曾把 Arm 4 的 step220000.pt 截到 49% 的那个竞态**这次没触发**。trainer `rc=1` 是 `kill -TERM` 的预期返回码，不是崩溃。
- **shard 完整性**：四轴 ×（arm + baseline）= 8 个 cell **全部 8/8 shard、`n_scored == expected`（popqa 14267 / triviaqa 17944 / nq_open 3610 / mmlu 14042）、0 重复 item_id、0 nan**；MMLU `summary.json` 独立报 `n_valid=14042 n_nan=0`。失败语法 grep（`Traceback (most recent call last)` / `CUDA out of memory` / `loss=nan`）**零命中**。
  ⚠️ 不要用 `grep -icE 'nan'` 判失败——它会命中 harness 自己的**通过**行 `✓ No NaN/Inf in model parameters`。
- **结果**：primary 轴 triviaqa em **θ = −0.3622 pp，CI95 [−0.5517, −0.1838]，SIG 负 → NOT-CONFIRM**；聚合 **0/3 CONFIRM → ARTIFACT**（维持现状，A-2 仍撤回）。σ_run：keep7-20k → S=4/df=3/s=0.4039 pp；pooled → **df=5 / 0.3666 pp / χ² [0.229, 0.899]**。
- 判定文档 `proposal/archive/A03-parametric-vs-external-memory/SEED45_VERDICT.md`；证据三份 JSON 在同目录 `evidence/`（两盘 md5 一致）。
- 后续配对差分 + σ 重算是 **CPU-only**（在 `.82` 上跑，各 <1 min），未额外占卡。
