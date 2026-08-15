# TRAINER ACTIVE — 2026-08-15 10:40 CST (GMT+8)

> **Write 覆盖，禁止 Edit**（CLAUDE.md「状态文件更新」）。

## 32/40 卡忙 — 四臂训练并行，`.212` 8 卡空闲（**故意不填**）

| 节点 | 盘 | 任务 | step | 实测 s/step | maxmem | ETA | 状态 |
|---|---|---|---|---|---|---|---|
| **LOCAL** | wzc1 | **Paper B keep10fresh2 resume** ★本轮新启 | **90200**/200000 | **1.200** | 106.7 GB | **1.52 d** | ▶️ 健康 |
| `.212` | wzc1 | **空**（8×0 MiB / 0 PID） | — | — | — | — | ⬜ 空闲，见下 |
| `.73` | zwfy6 | Paper B keep12fresh2 resume | ~171400+ | 7.92 | 91.9 GB | ~2.6 d | ▶️ 健康 **勿动** |
| `.82` | zwfy6 | Paper B keep8fresh2 resume | ~141160+ | 5.85 | 73.5 GB | ~4.2 d | ▶️ 健康 **勿动** |
| `.104` | zwfy6 | paperC qwen3base_heal_k8f2 | ~38800+ | 5.84 | 77.5 GB | ~10.9 d | ▶️ 健康 **勿动** |

**本轮唯一改动 = LOCAL 启动 keep10。三台 H20 全程未碰**（keep10 的 ckpt 是从 `.82` **只读** `dd` 拉的，
`.82` 的 keep8 训练在拉取期间照常推进，log 无中断）。

---

## ★ LOCAL — Paper B keep10+fresh2 resume（本轮新启）

- **launcher**：`scripts/launch_keep10_resume_b200_local_0815.sh`
- **log**：`logs/olmo2_7B_keep10fresh2_resume200k_local_0815.log`
- **torchrun PID 2937858**；worker PID 2938285-2938292（8 卡各 123870 MiB / 99-100%）
- **忠实 resume 已核**：banner `[resume] continue @ step=90000 epoch=0`、
  `[resume] restored 135 model tensors (strict, fp32 master weights)`、
  `[resume] optimizer state restored (135 param states) -> Adam momentum preserved`
- **config**：batch 16 × accum 1 × world 8 = **eff_batch 128**（与 H20 臂的 4×4×8 **完全等价**，
  非近似：DistributedSampler 给出同一 per-rank 排列，drop_last 下两种配置都是 121028 optimizer steps/epoch）
- **optimizer**：fp32 torch AdamW（**未传任何 bnb flag**；B200 无 bitsandbytes，
  该 trainer 只在 `--optimizer bnb_adamw8bit` 下才 import，不同于 `_distill.py` 的 module 级 import）
- **速率口径**：`1.200 s/step` 由 **log 自带时间戳** 的 Δelapsed/Δiter 在
  **180 step / 216 s 窗口**（90020→90200）算出；丢掉前 2 个点仍是 1.200。
  **不是** trainer 自报 postfix（那是冻结值）。vs H20 5.78 → **4.82× 加速**，7.35 d → **1.52 d**。

### 为什么 keep10 现在可以跑（不是「卡空了就填」）

CLAUDE.md 判据是「**paperC / proposal 是不是真的没活了**」，本轮实测：
- `proposal/ready_queue.py` → **0 ready_gpu**，9 ready_cpu 全是 0-GPU 写作/归档；
- `paperC/SUBMISSION_GAP_AUDIT.md:4` 自述 **「compute: CPU + web only. ZERO GPU.」**，
  且 paperC 训练已在 `.104` 上跑着；
- B10 gate 2/3 需 GPU 但 gate 1 刚判 **KILL** → 未授权。

### 两个跨盘资产都已解决（md5 双端一致）

1. **ckpt**：最新 keep10 = step90000（08-12）**原先只在 zwfy6**。用 **6 路并行 `ssh dd` 字节区间**
   从 `.82` 拉，**6.7 分钟**（聚合 ~92 MB/s；单流实测仅 17.7 MB/s）。
   md5 `0112936ed6bb1e3549269bb8b6461a17` **两盘一致**。
   ⚠️ wzc1 本地另有**更旧的** `step83500.pt` —— 从它起跑会**静默丢 6500 步**且与 zwfy6 同名 run 分叉；
   launcher 因此**发现最新 ckpt 再断言 ≥90000**，不信任写死的文件名。
2. **corpus**：15,491,607 行 dolmino（118 GiB）原先也只在 zwfy6。**没有搬** ——
   wzc1 有全部 84 个源 shard，**本地重建 153 秒**（vs 走线 ~3.2 h）。
   实测配方：`concat(sorted 84 shards)` = 15,495,703 行，`[0:4096]` == `data/dolmino_now_val.npy`（val），
   `[4096:]` == 训练语料。md5 `7df19b217e5b0670d58bf6e01e6559d0` 与 `.82:/dev/shm/dolmino_now15b.npy` **逐字节一致**。
   ⚠️ **wzc1 的 `data/dolmino_now15b.npy` 只有 7,570,911 行，是 PARTIAL PREFIX，绝不可用于本臂。**
   重建器：`scripts/build_dolmino_corpus_wzc1.py`（已从 `/tmp` 挪到项目盘，重启不丢）。

---

## ⛔ `.212` 8 卡空闲，本轮**故意不填**

1. **不是 keep10 的第二个节点**：该 trainer 是 **plain DDP**，每个 optimizer step all-reduce
   **13.0 GB** fp32 梯度（ring 26.0 GB/rank）。实测 LOCAL↔`.212` TCP **14.4 Gbps**
   → 纯网络 **~14.5 s/step**（即便假设 4 路多流也 ~3.6 s/step），**比整步 1.2 s 还大**。
   16 卡会**慢一个量级**。8 卡单机既快又稳。
2. **不是别的任务的位置**：`ready_queue.py` 仍是 **0 ready_gpu**。「卡空」不是启动理由。

---

## 运维备注

- `/dev/shm/dolmino_now15b_wzc1.npy`（118 GiB，tmpfs 944 G）**重启即失**；
  重建一行：`/opt/conda/envs/torch-base/bin/python scripts/build_dolmino_corpus_wzc1.py`（153 s）。
  launcher 的 preflight 会**核 rows + md5** 后才启动，不会拿错语料静默开跑。
- 该 trainer **没有** `--eval_interval` flag（无内联 BABILong eval），所以内联 eval 的 NCCL desync 风险不适用。
- 判「训练是否活着」用 `nvidia-smi --query-compute-apps` 的 PID + log 的 **mtime**，不要单次采样推趋势。
