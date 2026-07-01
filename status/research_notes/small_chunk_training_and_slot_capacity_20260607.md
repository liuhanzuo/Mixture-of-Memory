# 小 chunk 训练 + slot 容量随 chunk 调整（2026-06-07，纯调研，未改代码/未跑训练）

## 关键代码事实（已核对）
- 训练用 **per_doc** 模式（`launch_progressive_chunk_diskB.sh:84`），每个 sample = `(n_ctx+1)*chunk_size` 连续 token，前 n_ctx 个是 context、最后 1 个是 target（`dolmino_dataset.py:383,423-433`）。`curriculum 0:3` → **n_ctx=3 全程固定**（launch:95）。
- ⚠️ **澄清一个常见误读**：per_doc dolmino 里**每 step 的注入次数 = n_ctx+1 = 4，与 chunk_size 无关**（`train_..._tbptt` 逐 chunk forward+写 bank，`:1163,1184`）。roadmap F1 的"注入频率=seq_len/chunk_size"只在 **babilong-mixed 样本**成立（`babilong_train_step` `n_chunks=ceil(total_len/chunk_size)` `:1282`，babilong_mix=0.15）。
- 当前 4 个 stage **共用同一套 slot 配置**：`num_slots=128 top_k=16 selector_dim=128 slot_dim=None(=4096) num_global_slots=4`（launch:86-88），仅 chunk_size 不同。

## Q1：小 chunk 为何更不稳 + 怎样训练更合适
**真正的小-chunk 不稳来源（按代码定位）：**
1. **每 step 梯度 token 数随 chunk 线性缩水**（high）：tbptt 下梯度承载 token=(n_ctx+1)*chunk_size，chunk128=512、chunk1024=4096，**小 chunk 单步梯度方差天然大 4×**。这是 dolmino 路径的主因，不是注入频率。
2. **局部 SWA 窗口小 → 路由信号噪声大**（high，= F1 机理）：每次 forward 自注意力只看 chunk_size token，小 chunk 时单 chunk 信息少→ selector logits 更易抖。
3. **babilong-mixed 样本注入频率确实随 chunk 翻倍**（high）：这部分小 chunk 注入累积更多，spike 期放大成乱码（F2 实测 chunk128 step1000 PPL~3000）。
4. **warmup/spike-σ 是按 step 计、未按 token 归一**（medium）：warmup=300 step 在 chunk128 只热身 ~0.6M token，chunk1024 热身 ~5M token，**小 chunk 实际热身严重不足**。

**改进候选（小→大，可直接进下一个 launch 脚本）：**
- **[high, 可直接采用] 已是 P11 配方**：关 ST-Gumbel + `--normalize_readout`+`--use_delta_rule_writeback` 注入幅度钉死，半砍方差（稳定性报告实测）。当前脚本已带，保持。
- **[high, 可直接采用] warmup 随 chunk 反比缩放**：chunk128 用 warmup≈600-800、chunk1024≈150-300，保证各 stage 热身 token 量级一致。零风险。
- **[high, 可直接采用] grad_accum 随 chunk 反比缩放**：当前固定 accum=2；建议 chunk128 用 accum=8、chunk512 用 2，使**有效梯度 token/step 跨 stage 大致恒定**，直接压小 chunk 的梯度方差。
- **[medium] loss_spike_sigma 随 chunk 调**：小 chunk 抖动本就大，σ=3 可能误杀正常 batch；建议小 chunk 放宽到 σ=4。
- **[medium] lr 随有效梯度 token 缩放**：小 chunk 单步信号弱，peak lr 略降（1e-4→7e-5）或延后。
- **curriculum 方向**：现链是 chunk **小→大** warm-start，与 F1（大 chunk 更稳更强）一致、合理；不建议改"大→小"（会把已学好的长窗口能力往噪声更大的小窗口上退）。可考虑**直接从 chunk256 起步**跳过最不稳的 chunk128。

## Q2：slot 容量该如何随 chunk 变化
**核心约束（warm-start 链的硬限制，high）**：`num_slots / slot_dim / selector_dim` 决定 bank 参数**形状**，warm-start 跨 stage 改这三者→ckpt 形状不匹配（`load_state_dict strict=False` `:1066` 会静默丢掉整组 bank，等于重训）。**唯一能逐 stage 自由调而不破坏 warm-start 的 slot 旋钮是 `top_k`**（运行期选择数，非参数形状）。

**假设**：chunk 越大→单 chunk 信息越多→需更高有效容量（更多被激活 slot 或更大 slot_dim）；chunk 越小→大容量 over-parameterized、routing 更难学。这正是 roadmap D3，但 **D3 BLOCKED on D1**（slot_dim 16384 唯一 run 启动即崩、无结论）。

**可验证的最小实验（无依赖、可直接跑）：**
1. **top_k 随 chunk 阶梯**（warm-start 安全）：固定 num_slots=128/slot_dim=4096，仅 top_k 随 stage 升（c128:top_k8 → c256:16 → c512:24 → c1024:32），验证"大 chunk 需更多激活 slot"。⚠️注意 RUN_REGISTRY §3.4 已测 chunk512+top_k8 反而劣（qa5_0k=16），说明**减 top_k 伤检索覆盖**，故应是"小 chunk 也别低于 16，仅大 chunk 增"。
2. **固定容量全程**（当前做法）作为对照基线。
3. **依赖 D1**：先补干净的 slot_dim 16384 vs 4096 对照（修 wbmode 启动）→ 再谈"大 chunk 配大 slot_dim"。在 warm-start 链里 slot_dim 不能中途变，若要分容量须**断链分别训**或只在最末大-chunk stage 单独扩容重训。

**建议**：下一轮阶梯**先保持固定容量**（可复现基线），仅把 top_k 作为唯一随 chunk 调的容量旋钮做一次小消融；slot_dim 分级等 D1 有结论再做。
