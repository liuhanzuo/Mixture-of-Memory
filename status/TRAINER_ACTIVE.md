# TRAINER ACTIVE — 2026-08-15 21:44 CST (GMT+8)

> **Write 覆盖，禁止 Edit**（CLAUDE.md「状态文件更新」）。
> 速率一律取 **log 自带时间戳的 Δt/Δstep**（compute rate）；不用 trainer 自报 postfix（实测是冻结值），
> 不用 ckpt 间隔（被 flush 高估 ~13%）。

## 40/40 卡全忙 — 五臂训练并行，`.212` 本轮补上

| 节点 | 盘 | 任务 | step | 实测 s/step | maxmem | ETA | 状态 |
|---|---|---|---|---|---|---|---|
| **`.212`** | wzc1 | **Paper B #99 keep14-distill heal** ★本轮新启 21:31:20 | **5180**/200000 | **2.359** | 131.9 GB | **5.32 d** | ▶️ 健康 |
| LOCAL | wzc1 | Paper B keep10fresh2 resume | ~90200+/200000 | 1.200 | 106.7 GB | ~1.5 d | ▶️ 健康 **未碰** |
| `.73` | zwfy6 | Paper B keep12fresh2 resume | ~171400+/200000 | 7.92 | 91.9 GB | ~2.6 d | ▶️ 健康 **未碰** |
| `.82` | zwfy6 | Paper B keep8fresh2 resume | ~141160+/200000 | 5.85 | 73.5 GB | ~4.2 d | ▶️ 健康 **未碰** |
| `.104` | zwfy6 | paperC qwen3base_heal_k8f2 | ~38800+/200000 | 5.84 | 77.5 GB | ~10.9 d | ▶️ 健康 **勿动** |

本轮**只动 `.212`**；其余 4 节点 32 卡一张未碰（启动前后各 `nvidia-smi` 复核，`.212` 上 8×0 MiB / 0 PID 后才起）。

## 本轮新启：`.212` Paper B #99 keep14-distill heal（ladder 唯一缺的臂）

OLMo-2-7B base **32L teacher** → keep14+fresh2 **16L student**，`loss = NTP + 0.6 · KL(top-k=64)`。
**忠实 resume 自 step5000**（不是新 run）。

- torchrun **PID 524842**，worker **525664-525671**，8 卡各 151162 MiB @100%
- launcher `scripts/launch_keep14_distill_resume_212_0815.sh`；log `logs/olmo2_7B_keep14_distill_212_0815.log`
- `bs=16 GA=1 world=8 → eff_bs=128`，**与 H20 配方逐位相同** ⇒ 优化路径未变、同口径可比
- `save_every=500`（**解锁 #99 的关键**）、`gradient_checkpointing=1`（实测必须开）
- 语料 `/dev/shm/dolmino_now15b_wzc1.npy` rows=**15491607**
  md5=`7df19b217e5b0670d58bf6e01e6559d0`（与 keep8/keep12 所用**逐字节一致**）
- resume ckpt md5=`0ec4481adde2314a470616d49aa922e9`，**两盘一致**

### 忠实 resume 的证据（不是重启）

```
[resume] loading ckpt .../step5000.pt (saved at step 5000, has_optimizer=True)
[resume] optimizer state restored (179 param states) -> Adam momentum preserved
[resume] continue @ step=5000 epoch=0 warmup=150 max_steps=200000
[step 5020/200000] loss=3.1694 ppl=23.79 ntp_ppl=16.85 kl=0.5756
```
loss 从 3.169 接续；对照单卡探针里**从头跑**是 `ppl≈2.9e6`。

### #99 解锁的两点（实测，覆盖旧记载）

1. **「distill trainer 锁死 .73/.104」是错的。** 源码 line 63 注释自证 bnb 的唯一理由是
   *"to fit keep14 train-all + teacher in H20 95GB"* —— 与 178.4GB 的 B200 无关。
   `.212` 装 **bnb 0.50.1**，实测 `AdamW8bit` 在 **sm_100 (10,0)** 构造并 step 成功。
   保留 bnb ⇒ ckpt 的 8-bit optimizer state 可**忠实**加载（换 fp32 AdamW 只能从 step0 重跑、丢 5000 步）。
2. **真阻塞是 ckpt cadence。** `save_every 5000` + resume 起点正好 5000 → 下一次落盘在 10000；
   07-31 死于 step5200、08-05 只到 step7780 ⇒ **两次烧完预算、0 ckpt**。本轮 `save_every 500`。

### gradient_checkpointing 是**实测**结论，不是沿用默认

趁 ckpt 传输空窗跑 2 个单卡 40-step 探针（`scripts/_probe_distill_gc_b200.sh`）：
**GC=1 bs=16 → 2.26 s/step / 115.7 GB / 正常**；**GC=0 bs=16 → OOM**（178.35 GiB 已用 177.84 GiB）。
⇒ B200 也不能关 checkpointing；bs 受**激活**限制而非静态显存（静态仅 51.41 GiB）。

### ⚠️ 写作红线：不得声称差分 LR

`_classify_param` 没剥 DDP `module.` 前缀 + `build_param_groups` 在 DDP wrap 之后 ⇒ 8 卡 log 只有
`inh_decay 4060.1M @2e-5` + `inh_nodecay 0.3M @2e-5`，**实际均匀 2e-5**。与 keepN ladder 同 bug 同行为
（故可比），但论文不能写差分 LR。

## 监控要点

- 首个 ckpt 应在 **step5500** 落到 `outputs/olmo2_probe2_7B_keep14fresh2_distill/`。
  **若 step 已过 5500 仍无新 `.pt`，优先查落盘** —— #99 的历史失败模式正是「跑了但没落盘」。
- `/dev/shm` 是 tmpfs 且 **node-local**（`.212` 与 LOCAL 共享 wzc1 项目盘但**不共享 /dev/shm**）：
  重启后语料即失，用 `scripts/build_dolmino_corpus_wzc1.py` 重建（150 s）；launcher 会挡住拿错语料。
  ⚠️ `data/dolmino_now15b.npy` 是 **7,570,911 行 PARTIAL PREFIX**，禁用。
- 该 trainer **无 inline eval、无 `--eval_interval`** ⇒ inline-BABILong 的 NCCL desync 风险结构性不存在。
- 允许跑 Paper B 的判据：`proposal/ready_queue.py` → **0 ready_gpu**（8 ready_cpu 全 0-GPU）。
