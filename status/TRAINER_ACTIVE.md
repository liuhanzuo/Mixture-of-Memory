# TRAINER ACTIVE — 2026-06-13 07:50 +08:00

## 大局：ROUTE-A 全旋钮闭账（全证伪）→ 转推唯一未证伪杠杆 = eval-time SWA 深度扫描

ROUTE-A selector sweep 四类旋钮全部证伪闭账：
- loss_free (arm1)：usage_cov 0.88 达标但长程 qa5 未超 P11 base。
- entropy (arm2)：更差。
- selector_temperature {20,40,80}：三档全证伪（temp20 step1000 qa5 长程 10/7/5）。
- load_balance_weight (lbw=0.01, seed42)：qa5 长程 8k/16k/32k = 21/21/17 ≪ P11 base 48/45/44 → REJECTED。

**★关键正向发现**：复盘 RUN_REGISTRY + 重算 eval-time SWA 分，**eval-time cross-chunk SWA 是全 sweep 唯一未证伪且单调增益的杠杆**。P11 chunk512 step500：
- qa5 8k：W0=48 → W1=67 → W2=72（仍在爬，未饱和）
- qa5 16k：45 → 57 → 67
- qa1 8k：20 → 35 → 42

→ 本轮闭环填满 4 节点，纯离线 eval 现有 ckpt（无训练、无架构改动），扫 eval-SWA 三个最有信息量的轴。

## 本机 8×H20 — eval-SWA W-sweep P11 c512 step500 (RUNNING, HEALTHY)
- `scripts/eval_p11_step500_swa_wsweep_local.sh`，W={3,4,6,8} × qa{1,2,5} × 7 lengths。
- 找 eval-SWA 增益的饱和点（W>2 是否继续涨）。8 procs ~15.7GB/卡。
- 结果：`babilong_results/p11_step500_local_swa{3,4,6,8}`。

## .196 (diskA shared FS) 8×H20 — eval-SWA chunk-size sweep (RUNNING, HEALTHY)
- `scripts/eval_p11_chunksize_swa_sweep_196.sh`，P11 **chunk1024 + chunk256** step500，W={0,1,2}。
- 测 eval-SWA 增益是否随 chunk size 变化（chunk1024 是最强基线）。8 procs ~34GB/卡。
- 结果：`babilong_results/p11_c{1024,256}_step500_swa{0,1,2}`。

## .249 (diskB) 8×H20 — eval-SWA F2 long-doc c1024 sweep (RUNNING, HEALTHY)
- `scripts/eval_f2_c1024_swa_sweep_249.sh`，F2 长文档训练 chunk1024 step500，W={0,1,2}。
- 测 eval-SWA 是否也帮长文档训练的模型（diskB 本地 ckpt，无需 rsync）。8 procs ~15.7GB/卡。
- 结果：`babilong_results/f2_c1024_step500_swa{0,1,2}`。

## B200 .188 (wzc1) 8×L20A — eval-SWA F2 c512 sweep (PENDING rsync → auto-launch)
- ckpt rsync (10GB diskA→wzc1) 进行中；B200 上 detached waiter (pid 589382) 监测 ckpt 落地后自动起 `scripts/eval_f2_c512_swa_sweep_b200.sh`。
- 补 F2(长文档) × chunk-size 格点。结果：`babilong_results/f2_c512_step500_swa{0,1,2}`。

## .76 (diskB) 8×H20 — JUSTIFIED IDLE
- diskB 无 F2 c512 ckpt；为冗余确认 churn 10GB rsync 违反 anti-noise 铁律 → 留空。

## GPU UTILIZATION: 4/5 节点跑 eval-SWA sweep（local W深度 + .196 chunk-size + .249 F2长文档 + B200 待 rsync 自启），.76 justified idle。无 kill 无空转。

## 下一 HB
- 收齐 4 sweep 出分 → 用 `score_nested_babilong.py` 评分 → 画 eval-SWA 增益曲线（W × chunk_size × 训练数据 三维）写入 RUN_REGISTRY §3c。
- 检查 B200 waiter 是否成功自启（`/tmp/wait_launch_f2c512.done`）；若 rsync 失败则手动补。
- 为主会话方向决策提供量化依据：eval-SWA 是否值得做成 F3 架构化（训练时也带 SWA 已证伪 D2b，但 eval-time SWA 单调有效）vs 承认 mem-space 此规模不优于纯长上下文。
- needs_code alert (06-12 23:17) 仍 unacked，待主会话定向。
