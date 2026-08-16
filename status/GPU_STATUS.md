# GPU_STATUS.md — 5 节点单一事实来源

**最后实测 2026-08-17 02:10 GMT+8（heartbeat）。40/40 卡占用，0 空闲，每节点 8 个 compute PID（单一主人，无抢卡）。**

| 节点 | 硬件 | 盘 | 在跑 | step | 显存/卡 | util | amortised s/step | baseline | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| LOCAL(=.21) | 8×B200 sm_100 | wzc1 | `olmo2_probe2_7B_keep10fresh2` | 192940/200000 | 123.9 GB | 100% | **1.4148** | 1.3904 (9-interval mean) | healthy 1.018× — **ETA 04:35** |
| `.212` | 8×B200 sm_100 | wzc1 | `olmo2_probe2_7B_keep14fresh2_distill` | 46640/200000 | 157.8 GB | 95-100% | **2.4510** | 2.4500 | healthy 1.000× — ETA 08-21 10:16 |
| `.73` | 8×H20 sm_90 | zwfy6 | `olmo2_probe2_7B_keep12fresh2` | 191660/200000 | 96.4 GB | 100% | **7.9160** | 7.9200 | healthy 0.999× — ETA 19:57 |
| `.82` | 8×H20 sm_90 | zwfy6 | `olmo2_probe2_7B_keep8fresh2` | 165640/200000 | 78.5 GB | 100% | **5.8660** | 5.8640 | healthy 1.000× — ETA 08-19 09:43 |
| `.104` | 8×H20 sm_90 | zwfy6 | `paperC_qwen3base_heal_k8f2` | 66080/200000 | 78.8 GB | 100% | **5.8530** | 5.8380 | healthy 1.003× — ETA 08-26 03:33 |

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
