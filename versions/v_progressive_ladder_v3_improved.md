# v_progressive_ladder_v3_improved — 渐进 chunk 阶梯 V3 改进版（2026-06-07）

脚本：`scripts/launch_progressive_chunk_diskB_v3_improved.sh`
基线：`scripts/launch_progressive_chunk_diskB.sh`（v1，stable P11 配方）
依据：`status/research_notes/small_chunk_training_and_slot_capacity_20260607.md`（confidence=high 的「可直接采用」改进）

## Architecture（warm-start 链）
3 stage（v1 为 4 stage），每 stage warm-start 上一级的 step CHAIN_STEP ckpt：
```
stage1_c256 (scratch) --> stage2_c512 --> stage3_c1024
```
单节点 1×8 H20 DDP。slot 配置（num_slots=128 / top_k=16 / selector_dim=128 / slot_dim=4096 / num_global_slots=4）以及全部 P11 旋钮、offline-babilong / HF_HOME / WANDB offline / PYTHONPATH 设置与 v1 逐字一致。

## 相对 v1 的 4 处改动
1. **跳过 chunk128**：从 chunk256 起步（scratch）。chunk128 单步梯度方差 ~4×，F2 实测 chunk128 step1000 PPL~3000，最不稳。
2. **warmup 随 chunk 反比缩放**（per-stage 形参）：锚 chunk1024→300，warmup≈round(300*1024/chunk)。各 stage 热身 token 量级一致，避免小 chunk 热身严重不足。
3. **grad_accum 随 chunk 反比缩放**（per-stage 形参）：锚 chunk1024→2（v1 值），accum≈round(2*1024/chunk)。有效梯度 token/step 跨 stage 恒定，压小 chunk 梯度方差。
4. **loss_spike_sigma 小 chunk 放宽**（per-stage 形参）：避免小 chunk 自然抖动被误杀。

| stage | chunk | warmup | accum | sigma | port | init |
|-------|-------|--------|-------|-------|------|------|
| stage1_c256 | 256 | 1200 | 8 | 4.0 | 29830 | scratch |
| stage2_c512 | 512 | 600 | 4 | 3.5 | 29831 | stage1 ckpt |
| stage3_c1024 | 1024 | 300 | 2 | 3.0 | 29832 | stage2 ckpt |

## Initialization / 默认值
- `TOTAL_STEPS` 默认 5000（v1 默认 800；为训出强 ckpt 提高，env 可覆盖）。
- `SAVE_INTERVAL` 默认 500，`CHAIN_STEP` 默认 500（必须是 SAVE_INTERVAL 整数倍，脚本内有 guard，否则链 ckpt 不存在→链断）。
- run_stage 签名扩展为 `(stage_name, chunk, port, init_ckpt, warmup, accum, sigma)`。
- master_port 用 29830–29832，避开当前活跃的 29793/29794。

## Relationship to prior work
- 继承 v1 的 stable P11 配方（ST-Gumbel OFF + delta-rule writeback + normalized readout）。
- 仅落实 research note 中 confidence=high 的 token-归一化思路（warmup/accum 按 chunk 反比）+ 跳过最不稳 stage + spike-σ 放宽。

## Known issues
- slot 容量未随 chunk 调（note Q2 建议本轮保持固定容量作为可复现基线，top_k 消融留待后续；slot_dim 分级 BLOCKED on D1）。
- lr 随有效梯度 token 缩放（note 列为 medium）未采纳，lr 仍固定 1e-4。
- 未启动训练/未 rsync 到盘B，仅创建脚本。
