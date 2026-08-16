# GPU_STATUS.md — 5 节点单一事实来源

**最后实测 2026-08-16 21:02 GMT+8（heartbeat）。40/40 卡占用，0 空闲，每节点 8 个 compute PID（单一主人，无抢卡）。**

| 节点 | 硬件 | 盘 | 在跑 | step | 显存/卡 | util | amortised s/step | baseline | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| LOCAL(=.21) | 8×B200 sm_100 | wzc1 | `olmo2_probe2_7B_keep10fresh2` | 180020/200000 | 123.9 GB | 100% | **1.4100** | 1.2000 compute | healthy（+17.5% = wzc1 ckpt 代价） |
| `.212` | 8×B200 sm_100 | wzc1 | `olmo2_probe2_7B_keep14fresh2_distill` | 39380/200000 | 157.8 GB | 100% | 2.5833 | — | healthy |
| `.73` | 8×H20 sm_90 | zwfy6 | `olmo2_probe2_7B_keep12fresh2` | 189400/200000 | 96.4 GB | 100% | **7.9180** | 7.8000 | healthy 1.02× |
| `.82` | 8×H20 sm_90 | zwfy6 | `olmo2_probe2_7B_keep8fresh2` | 162600/200000 | 78.5 GB | 100% | **5.8640** | 5.8000 | healthy 1.01× |
| `.104` | 8×H20 sm_90 | zwfy6 | `paperC_qwen3base_heal_k8f2` | 63080/200000 | 78.8 GB | 100% | **5.8380** | 5.7500 | healthy 1.02× |

Monitor: `http200 OK`。错误行扫描：三个 zwfy6 live log 各 0 行。

## ⚠️ 本轮两个测量陷阱（都已避开，记下来别再踩）

1. **keep10 的 155 s 窗口读出 3.8750 s/step（=基线 3.2×），是 ckpt flush 伪影不是故障。**
   判据：`step180000.pt` mtime = **21:00:13**，落在我窗口（20:58:09–20:59:44）**之后** ——
   flush 被记在跨过 save 边界的那个区间里。改用**连续两个 500-step ckpt 间隔**得
   1.3700 / 1.4100 s/step，与 compute 基线 1.2000 差 +17.5%，正是已知的 wzc1 写盘代价。
   **报速率必须说清是 compute 还是 amortised。**
2. **zwfy6 三臂的 log 步进只有 +20，那是 log 粒度（±1 个区间），比值 0.69×/1.39× 是量化噪声不是测量。**
   同样改用 ckpt 间隔（500 步）→ 1.01–1.02×。**小 Δ 上的比值不构成测量。**

## 认任务的正确方式（同盘节点共享 logs/，mtime 会骗人）

先 `ps -eo pid,cmd --no-headers | grep "[t]rain_olmo2_arch_probe2.py"` 拿 `--output_dir`，再据此定位 log。
本轮实证：zwfy6 的 `logs/` 里有 **3 个 keep12 log + 4 个 keep8 log**（共享盘历史残留），live 的是 `_0814`；
LOCAL 的 live log 是 `_local_0815`，而 `_21.log` **自 08-08 起就没动过**。按 mtime 或猜名字都会拿错。

## chain watcher（`.73`）

PID **1243702**，`scripts/chain_keep12_eval_200k.sh`，存活已确认。
触发条件 = `outputs/olmo2_probe2_7B_keep12fresh2/step200000.pt` 出现且大小连续两次轮询不变（不是 log 行）。
按 keep12 自己的 amortised 7.9180 s/step，剩 10600 步 ≈ **23.3 h → 约 2026-08-17 20:19 GMT+8** 落地，届时自动起 ladder eval。

## 磁盘（2026-08-16 实测，详见 status/DISK_DECISION_20260816.md）

- wzc1 120T / **110T used / 10T free / 92%**；zwfy6 689T / **667T used / 22T free / 97%**（3 次采样一致，非 fuse 滞后）。
- 我们在 wzc1 占 **17.05 TiB = 已用的 15.5%**（35 个用户里第 2）；在 zwfy6 占 22.98 TiB = **3.4%**，
  而 `hunyuan/` 单独 **322 TiB（全盘 48%）** → **zwfy6 的 97% 不是我们造成的、也不是我们能修的**。
- 待用户决定的只有一件：`out_llama/`（4.69 TiB，99 个 SparseForge sweep，仅 3 个被按名引用，#245 仍 pending）。
