# H20 RoCE / IB 网络配置（2026-07-12，main 探明，供 16 卡 DDP 用）

**用户明确：这些 H20 只有 IB 网口（实为 RoCE over Ethernet）→ NCCL 必须 `NCCL_IB_DISABLE=0`，不能用 landmark 脚本默认的 TCP。**

## 三台 H20 RoCE 硬件（全部实测 State=Active LinkUp）
- 网卡 = Mellanox MT4126，200 Gbps，RoCEv2（Base lid=0，无原生 IB，走 GID）。
- RoCE 设备 = `mlx5_bond_1 .. mlx5_bond_8`（8 个），对应 netdev `eth2/4/6/8/10/12/14/16`（这些 eth **无 IP**，RoCE 用 GID 直连，不走 IP 路由）。
- RoCEv2 GID index 3（IPv4-mapped）解出各节点 RoCE IP：
  - `.24.104` (28.83.24.104): GID `ffff:1c56:8955` → **28.86.137.85**
  - `.85.73` (28.85.35.73): GID `ffff:1c57:61a9` → **28.87.97.169**
  - `.53.31` (28.83.53.31): GID `ffff:1c57:46c2` → **28.87.70.194**
- 每台有多个 bond 网卡跨 28.86.x / 28.87.x 两段（.53.31 实测：bond1=28.83.53.31/26, bond2-5=28.87.x, bond6-9=28.86.x）。RoCE 数据面走 GID/同 fabric，跨 IP 段不影响 RDMA 直连。

## 推荐 NCCL 环境变量（RoCE）
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
export NCCL_IB_GID_INDEX=3          # RoCEv2
export NCCL_SOCKET_IFNAME=bond1     # bootstrap/rendezvous 走主 bond1（有可路由 IP）
export NCCL_DEBUG=WARN
```
- `torchrun --master_addr` 用能互相 ssh 的 bond IP（TCP rendezvous）；RoCE 数据面自动经 mlx5_bond。
- 烟测判据：`NCCL_DEBUG` 显示 `[send] via NET/IB`（非 NET/Socket）+ all_reduce 无 30s timeout = RoCE 通。
- 若 RoCE 死活不通 → 降级单台 8 卡续训，别硬耗 timeout。

## 关键：step20000.pt 是旧格式（model-only）
`outputs/qwen3_minarch_armB_f12k2_20k/step20000.pt` 只有 model_state + step + arch meta，**无 optimizer_state / max_steps / rng**（它是 resume 功能 commit 0654d29 之前存的）。→ resume 会走 **warm-restart**（Adam 动量 re-init，LR 从 cosine 曲线 step20000 处续，非 warmup）。EXTEND log 不会触发（无 stored max_steps），但 get_lr 用新 args.max_steps=200000 重算 cosine，行为正确。
