# GPU_STATUS.md — 5 节点单一事实来源

**最后实测 2026-08-17 09:30 GMT+8（heartbeat）。32/40 卡占用，LOCAL 8 卡空闲（**故意**）。每节点 8 个 compute PID + 单一 owning script = 无抢卡。四臂全部 0 个 Traceback/OOM/ChildFailedError。**

| 节点 | 硬件 | 盘 | 在跑 | ckpt step | log step | 显存/卡 | util | amortised s/step（**ckpt mtime**） | baseline | ckpt 年龄/周期 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOCAL(=.21) | 8×B200 sm_100 | wzc1 | **IDLE** | — | — | 0 MiB | 0% | — | — | — | 三判据齐（0 MiB + 0% + 0 PID）；**故意空闲 → 见下方章节** |
| `.212` | 8×B200 | wzc1 | `olmo2_probe2_7B_keep14fresh2_distill` | **57500** | 57680 | 157.8 GB | 100% | **2.4500 / 2.4500** | 2.4500 | 6.6 / 20.4 min | healthy 1.000× — ETA 08-21 10:16 |
| `.73` | 8×H20 | zwfy6 | `olmo2_probe2_7B_keep12fresh2` + eval watcher | **195000** | 195060 | 96.4 GB | 99% | **7.9220 / 7.9160** | 7.9160 | 8.8 / 65.9 min | healthy 1.000× — **剩 ~4940 步 ≈ 10.9 h** |
| `.82` | 8×H20 | zwfy6 | `olmo2_probe2_7B_keep8fresh2` | **170000** | 170240 | 78.5 GB | 99-100% | **5.8740 / 5.8580** | 5.8640 | 24.0 / 48.8 min | healthy 1.000× — ETA 08-19 09:53 |
| `.104` | 8×H20 | zwfy6 | `paperC_qwen3base_heal_k8f2` | **70500** | 70700 | 78.8 GB | 100% | **5.8540 / 5.8520** | 5.8380 | 19.9 / 48.7 min | healthy 1.002× — ETA 08-26 03:33 |

> ### ★★ 09:30 修复：两个 watcher 会在 `.73` 上撞车（已加锁，实测过）
> `.73` 上**同时**有两个 watcher 准备启动**同一个** 8 卡 eval driver，而**两者都没有锁**：
>
> | watcher | 位置 | 触发条件 | poll |
> |---|---|---|---|
> | `chain_keep12_eval_200k.sh` (PID 1243702) | .73 | keep12 落 `step200000.pt` | 300 s |
> | `chain_keep10_ship_and_eval_200k.sh` (PID 655909) | LOCAL | `.73` 连续 2 次报 0 compute PID | 300 s |
>
> `eval_paperb_ladder_200k.sh` 把一个 7B 模型分片到**全部 8 张卡**（`CUDA_VISIBLE_DEVICES=$g`），
> 两个并发 = 每张卡两个 7B = **2026-08-08 毁掉 4/5 rung 的那个 OOM**。
> **撞车窗口是真的**：keep12 训练退出后、它自己的 eval 抢到卡之前有一段 0 PID；model-load 阶段也读 0 PID。
> 谁先采样谁赢 —— 那是抛硬币，不是设计。
>
> **修法**：锁加在 **driver** 里（两条 chain 都要经过它），所以**不动任何正在跑的 watcher**
> （运行中的 shell 会重读自己的 .sh，改它就是 bug）。fail-closed + 大声报错，不排队 ——
> 输的那个 watcher 本来就会继续 poll 重试，静默排队反而会把调度缺陷藏起来。
> commit `5e013ee`，已 `scp -O` 到 zwfy6（两端 sha256 一致 `3b3a49ad…`），`.73` 上实测
> `flock` 存在 + `bash -n` 通过 + `.locks/` 可写 + 训练未受影响（仍 8 PID）。

> ### 09:30：`.104` 之外三台的 log 归属曾被我搞错（第 2 次）
> 用 `olmo2_7B_keep8` / `qwen3` 这种**过松的 glob** + `head -1`（字典序，**不是最新**），
> 给 `.82` 选中了 **8-01 的死 log（step 48060）**，给 `.104` 选中了 **`.73` 的 8-09 log**。
> 两个文件读起来都完全正常。**`.82` 那条会凭空造出一个 stall 判定**，因为它永远不会再动。
> 正解要**两个判据同时满足**：`find logs -newermt '-6 minutes'`（活的）**且** 文件名/内容含
> 由该节点 `--output_dir` 推出的 run 名（归属对）。
> 附证据：`.73` 和 `.82` 看到的「最近 6 分钟被写过的 log」**是同一张表** —— 共享盘的铁证。

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

## 为什么不填 LOCAL 的 8 张 B200（08:30 复核，理由未变）

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
下一次自动补卡：`.73` keep12 到 200k（~20:41）→ chain 自动投 keep12 eval，keep10 eval 紧随同一节点。

Monitor: `http200 OK`，`latest` 键 5 节点各 8 卡。错误行扫描：四个 live log 各 0 行。

## ★ chain watcher 现况（09:30 实测）

| watcher | PID | 状态 | 等什么 → 然后做什么 |
|---|---|---|---|
| `chain_keep10_ship_and_eval_200k.sh` | 655909 | **alive 07:21:31** | `.73` 连续 2 次 0 PID → 跑 keep10 ladder eval（ckpt 已送达并核过 md5） |
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
