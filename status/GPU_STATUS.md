# GPU_STATUS.md — 5 节点单一事实来源

**最后实测 2026-08-17 07:58 GMT+8（heartbeat）。32/40 卡占用，LOCAL 8 卡空闲（**故意**）。每节点 8 个 compute PID + 单一 owning script = 无抢卡。四臂全部 0 个 Traceback/OOM/ChildFailedError。**

| 节点 | 硬件 | 盘 | 在跑 | step | 显存/卡 | util | amortised s/step（**ckpt mtime**） | baseline | ckpt 年龄/周期 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| LOCAL(=.21) | 8×B200 sm_100 | wzc1 | **IDLE** | — | 0 MiB | 0% | — | — | — | 三判据齐（0 MiB + 0% + 0 PID）；**故意空闲 → 见下方章节** |
| `.212` | 8×B200 | wzc1 | `olmo2_probe2_7B_keep14fresh2_distill` | 55000 ckpt | 157.8 GB | 100% | **2.4506 / 2.4507 / 2.4497** | 2.4500 | 19.0 / 20.4 min | healthy 1.000× — ETA 08-21 10:16 |
| `.73` | 8×H20 | zwfy6 | `olmo2_probe2_7B_keep12fresh2` + eval watcher | **194380** (log) | 96.4 GB | 100% | **7.9171 / 7.9194 / 7.9131** | 7.9160 | 51.1 / 65.9 min | healthy 1.000× — ETA ~20:41 |
| `.82` | 8×H20 | zwfy6 | `olmo2_probe2_7B_keep8fresh2` | **169320** (log) | 78.5 GB | 100% | **5.8606 / 5.8609 / 5.8571** | 5.8640 | 32.0 / 48.8 min | healthy 0.999× — ETA 08-19 09:43 |
| `.104` | 8×H20 | zwfy6 | `paperC_qwen3base_heal_k8f2` | **69780** (log) | 78.8 GB | 98-100% | **5.8463 / 5.8426 / 5.8455** | 5.8380 | 27.7 / 48.7 min | healthy 1.001× — ETA 08-26 03:33 |

> ### 07:58：上一轮 `.104` 的「ckpt 号没动」已确认是正常周期
> 07:28 我看到 `.104` 两轮最新 ckpt 都是 `step69000`，按 ckpt 年龄 46.6 min vs 48.7 min 周期判为**非 stall**。
> 本轮它已落 `step69500`（年龄 27.7 min）、log 到 `step69780` —— **判断得到证实**。
> 表里新增「ckpt 年龄/周期」一列，就是为了让下一个 agent 一眼看出「号没变」是否落在周期内，
> 不必重新推导。**四臂的 ckpt 年龄全部小于各自周期。**

> ### 07:28 补记：`.104` 的 ckpt 号两轮没动 ≠ stall（已排除）
> 上一轮和本轮最新 ckpt 都是 `step69000`。**没有据此报 stall**，而是查了两件事：
> ckpt **年龄 46.6 min** 对 `save_every=500 @5.84 s/step = 48.7 min` 的周期 —— 下一个还差 ~2 min 才到点；
> 且 log 已推进到 **step 69480**、mtime 是秒级新鲜。**「ckpt 号没变」在一个周期内是正常的**，
> 判活要看 log 推进 + ckpt 年龄对周期，不是看 ckpt 号是否变化。

> ### 07:00 补记：`.212` GPU0 读到 0% —— 是 ckpt flush，不是 straggler，已证
> 首次采样 GPU0 = **0%**，其余 7 张 100%。**没有据此下结论**，而是连采 4 次（间隔 ~8 s）：
> `0% → 8% → 4% → 100% → 98%` —— GPU0 自行恢复。
> **决定性证据不是 util 而是 ckpt 间隔**：最新 ckpt `step54000` 当时**只有 0.5 分钟大**，
> 即我正好采样在 rank-0 写盘期间；而 ckpt 间隔 2.4504/2.4521/2.4506 s/step **完全没有变慢**。
> 参见 `memory/ckpt-interval-rate-is-not-compute-rate.md`：低 util 的单次读数总落在 GPU0，
> 因为 rank-0 干额外活。**单点 util 不是状态**；要判生死看 artifact 的推进。

> ## ⚠️⚠️ `.73` 速率：我今天报错了**两次**，方向相反，两次都是 watcher 采样伪影
>
> | 数值 | 来源 | 结论 |
> |---|---|---|
> | 8.333 | watcher，5 个区间的**均值** | ❌ 05:52 报的「慢 5.2%」 |
> | 7.500 | watcher，**众数**区间（40 步/300 s） | ❌ 06:07 报的「快 5.3%」—— 我自称在纠正，其实是第二个错 |
> | **7.9167–7.9215** | **ckpt mtime**，连续 5 个区间（500/2500/5000 步） | ✅ **权威，= 基线 1.0004×** |
>
> **根因是 aliasing，不只是「量化」。** watcher 每 300 s 采一次，而 log 每 **20 步**才写一行。
> 真速率 7.919 s/step → 300 s = **37.9 步** → watcher 只能看到 `40, 40, 20, 40, 40 …` 交替。
> 于是：**取众数（40）系统性高估速度**（得 7.50），**跨越交替边界的窗口系统性低估**（得 8.33）。
> **两个统计量都不会收敛到真值，因为采样器比信号粗。**
>
> **规则（比上一版更强）**：速率**只能**用 **ckpt mtime** 这类与被测信号同粒度的源；
> watcher / tqdm 这类**下采样日志**只能用来判「是否在推进」，**不能用来算速率**。
> 某 arm 的 s/step 若与基线偏离 >2%，先问「这个数是哪个源给的」，再问「那个源的粒度是多少」。
> **换统计量不等于换方法** —— 06:07 我把均值换成众数却没换源，于是又错一次。
## 为什么不填 LOCAL 的 8 张 B200（2026-08-17 05:52）

**「有空卡」≠「必须马上塞任务」。** 三个候选全部被实测排除，不是漏看：

1. **proposal 侧真的没活**：`proposal/ready_queue.py` 报 `0 ready_gpu`。每个 proposal 要么被 0-GPU gate
   挡着（8 个 `ready_cpu`），要么有明确的 no-further-GPU 处置（A02），要么缺 **USER APPROVAL**
   —— A04 完整 gate 是 1,077–4,309 GPU-h，我不能自行批准。
2. **keep10 的 200k eval 不能在这里跑**：`scripts/eval_paperb_ladder_200k.sh:85` 写死 `REQUIRE_SM=9.0`，
   LOCAL 是 sm_100。虽有 `SKIP_ARCH_GUARD`，但用它会让 Table 4 的一个 rung 跑在与其他 rung 不同的架构上
   —— 那正是该 guard 要防的污染（实测 cross-arch floor 0.03–0.16 pp）。ckpt 已 `scp -O` 送到 `.73`
   且两端 md5 一致（`4440fb7f0471d6952b2ffacdbad7d691`），chain 正在等 `.73` 腾卡。
3. **唯一 wzc1/sm_100-resident 的待跑项 #245 ALPS+SLoRB 是 211 GPU-h**，且
   `status/ALPS_SLORB_GATE0_VERDICT.md` 自己写着 **NOT LAUNCHED**，卡在两个 scoping 决定
   （已有的 625M-token 点是否已回答 reviewer / 现存 run 的 4 处配置差异怎么定价）。**加卡救不了**：
   `global_batch_size=256` 固定，8 卡的 GPU-h 最好也只是持平，scaling 不理想则更差。

**为了不报「空闲」而投一个 211 GPU-h 的 run，正是「卡满 ≠ 在跑对的东西」这个错误本身。**
下一次自动补卡：`.73` keep12 到 200k（~21:05）→ chain 自动投 keep12 eval，keep10 eval 紧随同一节点。

Monitor: `http200 OK`。错误行扫描：五个 live log 各 0 行。

## ★ 两个 chain watcher 在跑（2026-08-17 02:00 起，都在 LOCAL 上）

LOCAL 此前**没有任何后继任务在等**（`pgrep 'watch|chain|wait'` 只返回编辑器的 file watcher），
25 分钟后 8 张 B200 会空转。已排两个后继，**它们不可互换**：

| watcher | PID | 等什么 | 然后做什么 |
|---|---|---|---|
| `chain_b12_pilot_on_local_free.sh` | 650568 | LOCAL 上 compute PID 连续 2 次为 0 | B12 pilot pair：先 rung P，P 成功才跑 Dctl。1.46 GPU-h。**Q/R/S 一律不跑** |
| `chain_keep10_ship_and_eval_200k.sh` | 655909 | `step200000.pt` size 稳定 | `scp -O` 到 .73（~34 min）→ **两端核 md5** → 等 .73 空卡 → 跑 ladder eval |

**为什么必须拆成两个**：`eval_paperb_ladder_200k.sh:85` 写死 `REQUIRE_SM=9.0`，非 H20 直接 die
（Table 4 是单一 H20 口径，core6 有实测 0.03-0.16pp 跨架构地板）。LOCAL 是 sm_100，
所以 keep10 的 eval **不能在它自己跑完的机器上做**；而 B12 pilot 反过来**只能**在 sm_100/wzc1 跑。

## ⚠️ keep10 是唯一「ckpt 在错盘」的 rung（2026-08-17 实测）

zwfy6 的 `outputs/olmo2_probe2_7B_keep10fresh2/` **停在 step90000**（08-12），wzc1 已过 193000
→ `step200000.pt` 将是 **wzc1-only**。keep8 在 .82、keep12 在 .73 训练，本来就在 zwfy6，就地 eval 即可。

## ★★ 跨盘速率实测：19.2 MB/s，不是 12 MB/s（CLAUDE.md 那条自相矛盾）

CLAUDE.md 写「12MB/s 单流 / 搬两个 45.4GiB 约 42 小时」——**这条跟它自己的算术矛盾**
（45.4GiB×2 @ 12MB/s ≈ 2.3 h，不是 42 h）。所以我实测了：**2 GiB 探针 wzc1→.73 `scp -O` 用时 112 s
= 19.2 MB/s，两端 md5 一致**，探针已从两侧删除。
→ keep10 的 39.01 GB ckpt 只需 **~34 分钟**，而且发生在 .73 空出前 ~17 h，**不占任何 GPU 时间**。
**这个量级的跨盘搬运是便宜的；不要再拿「42 小时」当理由否掉一次搬运，先实测。**

## ⚠️ 上一轮两个测量陷阱（仍然有效，别再踩）

1. **ckpt flush 伪影**：save 边界所在的区间会把 flush 记进去。用**连续两个 500-step ckpt 间隔**测，
   并说清报的是 compute 还是 amortised。
2. **log 步进只有 ±20 步时，比值是量化噪声不是测量。** 本轮 `.73` 的 tqdm 瞬时值在 7.81–10.26 s/it 之间跳，
   而 ckpt 间隔给出 7.914/7.918 —— **以 ckpt 间隔为准**。
