# GPU_STATUS.md — 单一事实来源

**最后实测 2026-08-17 12:50 GMT+8。⚠️ 集群从 5 节点/40 卡 变成 6 节点/48 卡：新增 `.25`（8×B200，第三个盘）。**
32/48 卡占用；LOCAL 8 卡（B01 agent 在用/待用）+ `.25` 8 卡（刚接入，零资产）。四个训练臂全部 0 Traceback/OOM。

## ★★ 2026-08-17 拓扑纠正：我们的「5 节点集群」本身就是 taiji 的 pod

`get_general_train_instance_docker_ip` 按 task_id 反查容器 IP，实测五台全部命中：

| taiji task_id | pod IP | 我们叫它 | namespace | 盘 |
|---|---|---|---|---|
| `basic_train_pighzliu_20260710175554_15fc7d86` | 28.89.19.21 | LOCAL | `…-wzc3` | wzc1 |
| `basic_train_pighzliu_20260713190850_5eba9dcd` | 28.89.18.212 | `.212` | `…-wzc3` | wzc1 |
| `basic_train_pighzliu_20260711153303_3c9c4fd4` | 28.85.35.73 | `.73` | `…-zhongwei-2` | zwfy6 |
| `basic_train_pighzliu_20260712232924_6e275d82` | 28.82.250.82 | `.82` | `…-zhongwei-2` | zwfy6 |
| `basic_train_pighzliu_20260710180127_a0e13a61` | 28.83.24.104 | `.104` | `…-zhongwei-2` | zwfy6 |
| `pighzliu_train3`（inst `8b1d89c79ffa5b95019ffef255bc07b9`） | **28.197.251.25** | **`.25`（新）** | `…-wzz-1` | **wzz** |

**推论**：`root` 密码可由 `auth.passwd` 字段直接取得，**不用扫 `configs/password_*.txt`**
（我扫了 19 个全 `rc=5`，其实一条 MCP 调用就有）。「节点重启抹掉 conda/sshpass」= **pod 被重建**。
**taiji 不是「集群之外的另一个算力池」，它就是同一个池子**，我们只是从 pod 内部看它。

## ★★ 第三个盘 `/apdcephfs_wzz`（share_**303419932**，注意 share ID 都不同）

`.25` 上 `ls -d /apdcephfs*` **只有** `/apdcephfs_wzz`；`/apdcephfs_wzc1` 和 `/apdcephfs_zwfy6`
**都不存在**（不像 `.73` 有 symlink 兜底）。盘上只有 `hunyuan/common` + `test/` ——
**没有仓库、没有模型、没有数据**。41T 已用 39T = **96% 满，仅剩 1.9T**；
另有 **23 TB 空闲本地 overlay `/`**（但随 pod 销毁）。

**跨盘带宽实测 7.88 MB/s = 27.7 GB/h**（1 GiB `scp -O` 130 s，两端 md5 一致
`ec88eb4023898f813471944959e9eec3`，探针已双侧删除）。→ 代码 ~2 GB ≈ **4 min**；
8B 模型/ckpt 16.4 GB ≈ **35 min**；OLMo2-7B ckpt 39 GB ≈ **1.4 h**。
比 wzc1↔zwfy6 的 19.2 MB/s 慢 2.4×，但代码/单模型量级完全可接受。
工具差异：`.25` 有 `rsync`/`scp`/`git` **无 `sshpass`**；本机 **无 `rsync`**（用 `scp -O`）。

## ★ 用户盘策略（2026-08-17，覆盖此前的对称假设）

> 「wzc1 当作主要的盘，以后需要什么就往其他地方 transfer 什么。**模型可以留下但代码最好每次训练都新传**。」

**wzc1 = canonical**。模型/ckpt 远端常驻；**代码每次训练前重传，或先验 sha256 逐文件一致**。
⚠️ 不能用 commit 比对：实测 wzc1 **解析不出** zwfy6 的 HEAD `2d98c5a`
（`git cat-file -t` → `could not get object info`），且 zwfy6 有 **1194** 个未提交改动
→ **「远端是否最新」在 git 层面无法回答**，只能比文件内容。

| 节点 | 硬件 | 盘 | 在跑 | ckpt step | amortised s/step（**ckpt mtime**） | baseline | 判定 |
|---|---|---|---|---|---|---|---|
| LOCAL(=.21) | 8×B200 sm_100 | wzc1 | B01 persist-path agent（0-GPU 主体） | — | — | — | 8×0 MiB；agent 在改代码，GPU 待用 |
| **`.25`（新）** | **8×B200 sm_100**（`nvidia-smi` 老实报 `NVIDIA B200`） | **wzz** | **空闲，零资产** | — | — | — | 8×0 MiB / 0% / 0 PID；192 vCPU、1993 GB RAM、torch 2.8.0 见 8 卡 |
| `.212` | 8×B200 | wzc1 | `olmo2_probe2_7B_keep14fresh2_distill` | 61000 | 2.4520 / 2.4480 | 2.4500 | healthy — ETA 08-21 10:16 |
| `.73` | 8×H20 | zwfy6 | `olmo2_probe2_7B_keep12fresh2` + eval watcher | **196500** | **7.9200**（196000→196500 = 66.0 min） | 7.9160 | healthy 1.000× — 剩 3500 步 ≈ 7.7 h |
| `.82` | 8×H20 | zwfy6 | `olmo2_probe2_7B_keep8fresh2` | 171500 | 5.8580 / 5.8620 | 5.8640 | healthy — ETA 08-19 09:53 |
| `.104` | 8×H20 | zwfy6 | `paperC_qwen3base_heal_k8f2` | 72000 | 5.8380 / 5.8420 | 5.8380 | healthy — ETA 08-26 03:33 |

> ⚠️ `.25` 的 `nvidia-smi` **老实报 `NVIDIA B200`**，而 LOCAL/`.212` 报 `L20A`。
> → **name 字符串不可靠是那两台的局部问题，不是全局规律**（见
> `memory/l20a-name-string-is-really-b200-sm100.md`）。判代际一律看 `compute_cap`（10.0）。

## ★ taiji 可动用配额（实测，2026-08-17）

我们自己的组 `TaiJi_HYAide_Pretrain_Test`，真正能拿的是 **`min(quota_free, phys_free)`**：

| 卡型 | quota_total | used | quota_free | phys_free | **可动用** |
|---|---|---|---|---|---|
| **H800** | 384 | 184 | 200 | 295（nanjing-4） | **200 → 25 个 8 卡节点** |
| H20 | 1016 | 1004 | 12 | 1950（zhongwei-1/2） | 12 |
| L20A | 32 | 8 | 24 | 48（wzz-1） | 24 |
| L20A（第二条） | 0 | 16 | **−16（已超配）** | — | 0 |
| A800 | 16 | 16 | 0 | 16 | 0 |

**只看物理会高估**（H800 物理 295 但配额只给 200）；**只看配额会漏掉超配的负数**。
全 31 组 normal free 合计 1357 卡；elastic 另有 Exp2_GY_L40S free=595、hy_exp_SH_A100H free=488。
两盘 ceph 可达性已验（`query_storage_cluster_free_space` 对 wzc1/zwfy6 均 rc=0）。

**待定的关键问题**：taiji 提交新任务能否指定挂 `share_304376610`（我们的项目盘）？
能 → rsync 问题消失，200 张 H800 立即可用。已派 workflow `wf_d11287fd-a9f` 查证。

## 为什么 LOCAL 之前空着 —— 我报的理由每条都真，但整体结论错了

三条理由（proposal `0 ready_gpu`、keep10 eval 要 sm_90、#245 是 211 GPU-h 未启动）都经得起核，
但它们回答的是「**这 40 卡里有没有活**」，而用户问的是「**为什么空着**」。
我把「集群」默认成 CLAUDE.md 里那 5 节点 —— **范围错误，不是事实错误**，所以自查抓不到。
另外 B01 的 blocker 是**可以派人关的 0-GPU 代码缺口**，我却连报数轮当 blocker
（见 `memory/reporting-a-gap-is-not-closing-it.md`）。

## ⚠️ 仍然有效的测量陷阱

1. **速率只能用 ckpt mtime**：watcher 每 300 s 采样而 log 每 20 步一行 → aliasing，
   众数高估（7.50）、跨界窗口低估（8.33），**两个统计量都不收敛**。真值 7.919。
2. **单点 util 不是状态**：低 util 单次读数总落在 GPU0（rank-0 干额外活）。决定性证据是 ckpt 间隔。
3. **「文件不存在」现在要查三个盘**（wzc1 / zwfy6 / **wzz**）才成立。


> ### ★ 判「ckpt 年龄接近周期」时：用 log 位置做**可证伪的预测**，然后等它落地
> 11:27 `.73` 的 ckpt 年龄 62.8 min vs 65.9 min 周期 —— 本轮最贴边的一次。
> **没有**用「还在周期内」放过：由 log 位置（195980，即 ckpt 195500 之后 480 步）预测
> `step196000` 约 2.6 min 后落盘，**等到 11:31 亲眼看到它落地**（age 1 min，log 同步到 196000）。
> **先预测再观察，不是从阈值反推。** `.82`/`.104` 同理：log 分别领先 ckpt 480/440 步，同一健康模式。

> ### ★★ paperC 没有 standing red —— 那个「1 red」是我自己的 glob（11:57 查明）
> 我历轮用 `for g in paperC/code/gate*.py` 扫 gate，**按文件名前缀选**，于是把
> `gate2_crossfamily_nulls.py` 也扫了进去。它**不是 gate**，是 cross-family 扩展的
> **分析 emitter**（自己的 docstring 写 "the analysis half"，有必填位置参数
> `xf_root out_json`，末尾 `json.dump`）。无参调用它**永远** argparse usage + exit 2。
> 它的产物一直在盘上：`evidence/second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.json`
> (1,176,542 B) + `.csv` (203,150 B)，`paperC/README.md:241` 列为两盘 shipped artifact。
> **正确口径是 9/9 gate 全绿。**
> 已加 `paperC/code/run_all_gates.py`（commit `0d7cf6f`）：**按属性选**（无参可跑才是 gate），
> 用 `ast` 解析 argparse **推导**分类而不是写死跳过名单，且**打印被排除项及理由**（排除不可静默）；
> 同时把解释器钉在 conda（`.venv` 缺 PyMuPDF 会让 build_record gate 把 ModuleNotFoundError 印成 FAIL）。
> 实测 `PASS: 9/9 gates rc=0`。

> ## ⚠️⚠️ `.73` 速率：我今天报错了**两次**，方向相反，两次都是 watcher 采样伪影
>
> | 数值 | 来源 | 结论 |
> |---|---|---|
> | 8.333 | watcher，5 个区间的**均值** | ❌ 05:52 报的「慢 5.2%」 |
> | 7.500 | watcher，**众数**区间（40 步/300 s） | ❌ 06:07 报的「快 5.3%」—— 我自称在纠正，其实是第二个错 |
> | **7.9120–7.9220** | **ckpt mtime**，连续多个区间（500/3500/5000 步） | ✅ **权威，= 基线 1.000×** |
>
> **根因是 aliasing，不只是「量化」。** watcher 每 300 s 采一次，而 log 每 **20 步**才写一行。
> 真速率 7.919 s/step → 300 s = **37.9 步** → watcher 只能看到 `40, 40, 20, 40, 40 …` 交替。
> 于是：**取众数（40）系统性高估速度**（得 7.50），**跨越交替边界的窗口系统性低估**（得 8.33）。
> **两个统计量都不会收敛到真值，因为采样器比信号粗。**
>
> **规则**：速率**只能**用 **ckpt mtime** 这类与被测信号同粒度的源；watcher / tqdm 这类
> **下采样日志**只能用来判「是否在推进」，**不能用来算速率**。
> **换统计量不等于换方法** —— 06:07 我把均值换成众数却没换源，于是又错一次。
> 详见 `memory/a-downsampled-log-cannot-give-a-rate.md`。

## 为什么不填 LOCAL 的 8 张 B200（11:57 复核，理由未变）

**「有空卡」≠「必须马上塞任务」。** 三个候选全部被实测排除，不是漏看：

1. **proposal 侧真的没活**：`proposal/ready_queue.py` 报 `0 ready_gpu`（8 个 `ready_cpu`）。
   每个 proposal 要么被 0-GPU gate 挡着，要么有明确的 no-further-GPU 处置（A02），
   要么缺 **USER APPROVAL**（A04 完整 gate 是 1,077–4,309 GPU-h，我不能自行批准）。
2. **keep10 的 200k eval 不能在这里跑**：`scripts/eval_paperb_ladder_200k.sh:85` 写死 `REQUIRE_SM=9.0`，
   LOCAL 是 sm_100。虽有 `SKIP_ARCH_GUARD`，但用它会让 Table 4 的一个 rung 跑在与其他 rung 不同的架构上
   —— 那正是该 guard 要防的污染（实测 cross-arch floor 0.03–0.16 pp）。ckpt 已 `scp -O` 送到 `.73`
   且两端 md5 一致（`4440fb7f0471d6952b2ffacdbad7d691`，39,009,622,410 B 已核），chain 正在等 `.73` 腾卡。
3. **唯一 wzc1/sm_100-resident 的待跑项 #245 ALPS+SLoRB 是 211 GPU-h**，且
   `status/ALPS_SLORB_GATE0_VERDICT.md` 自己写着 **NOT LAUNCHED**，卡在两个 scoping 决定。
   **加卡救不了**：`global_batch_size=256` 固定，8 卡的 GPU-h 最好也只是持平。

**为了不报「空闲」而投一个 211 GPU-h 的 run，正是「卡满 ≠ 在跑对的东西」这个错误本身。**
下一次自动补卡：`.73` keep12 到 200k（~20:20，剩 3800 步 @7.916 s/step）→ chain 自动投 keep12 eval，keep10 eval 紧随同一节点。

Monitor: `http200 OK`，`latest` 键 5 节点各 8 卡。错误行扫描：四个 live log 各 0 行。

## ★ chain watcher 现况（11:57 实测）

| watcher | PID | 状态 | 等什么 → 然后做什么 |
|---|---|---|---|
| `chain_keep10_ship_and_eval_200k.sh` | 655909 | **alive 09:51:32** | `.73` 连续 2 次 0 PID → 跑 keep10 ladder eval（ckpt 已送达并核过 md5） |
| `chain_keep12_eval_200k.sh`（.73 本地） | **1243702** | **alive**，每 300 s 一行 | keep12 落 `step200000.pt`（**认文件不认 log 行** —— log 行会早于 43.9 GB 落盘） → 自动投 keep12 eval |
| `chain_b12_pilot_on_local_free.sh` | 650568 | **已完成退出**（05:24，rung P rc=0 / Dctl rc=0） | — Q/R/S 按预注册未跑，2.92 GPU-h 未花 |

**两条 live chain 现在被 driver 里的 flock 串行化**（见上方 09:30 修复条）：谁先拿到锁谁跑，
输的那个照常继续 poll 重试，**不会两个同时占同一节点的 8 卡**。

**为什么 keep10 的 eval 必须拆出去**：`eval_paperb_ladder_200k.sh` 写死 `REQUIRE_SM=9.0`，
非 H20 直接 die（Table 4 是单一 H20 口径）。LOCAL 是 sm_100，所以 keep10 的 eval
**不能在它自己跑完的机器上做**。

## ⚠️ keep10 是唯一「ckpt 在错盘」的 rung

zwfy6 的 `outputs/olmo2_probe2_7B_keep10fresh2/` 曾停在 step90000，wzc1 已过 193000
→ `step200000.pt` 是 wzc1-only，已 `scp -O` 到 `.73`（39.01 GB，两端 md5 一致）。
keep8 在 .82、keep12 在 .73 训练，本来就在 zwfy6，就地 eval 即可。

## ★★ 跨盘速率实测：19.2 MB/s，不是 12 MB/s（CLAUDE.md 那条自相矛盾）

CLAUDE.md 写「12MB/s 单流 / 搬两个 45.4GiB 约 42 小时」——**这条跟它自己的算术矛盾**
（45.4GiB×2 @ 12MB/s ≈ 2.3 h，不是 42 h）。实测：**2 GiB 探针 wzc1→.73 `scp -O` 用时 112 s
= 19.2 MB/s，两端 md5 一致**，探针已从两侧删除。
→ keep10 的 39.01 GB ckpt 只需 **~34 分钟**，且发生在 .73 空出前，**不占任何 GPU 时间**。
**不要再拿「42 小时」当理由否掉一次搬运，先实测。**

## ⚠️ 两个测量陷阱（仍然有效，别再踩）

1. **ckpt flush 伪影**：save 边界所在的区间会把 flush 记进去。用**连续两个 500-step ckpt 间隔**测，
   并说清报的是 compute 还是 amortised。
2. **单点 util 不是状态**：低 util 的单次读数总落在 GPU0（rank-0 干额外活）。
   `.212` 曾首采 GPU0=0%，连采 4 次后 `0%→8%→4%→100%→98%` 自行恢复，
   而同期 ckpt 间隔完全没变慢 —— **决定性证据是 ckpt 间隔，不是 util**。
