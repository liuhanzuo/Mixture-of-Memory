# BUG.md — 未解问题记录

> lhz/lhz2 H200 集群上的已知 bug。相关:工作日志 `HARU.md`、计划 `PLAN_minimal_arch_A13B.md`、环境 `status/HY3_ENV_SETUP_lhz.md`。

---

## BUG-1 [OPEN] A13B keep13+fresh2 训练在 FSDP 首个 collective 死锁 / 崩溃(从未跑出过一个 step）

**状态**:未解。2026-07-14 发现,多轮尝试后锁定根因方向,尚未修复。
**影响**:极简架构训练(`scripts/train_hunyuan_a13b_probe2.py`,keep13+fresh2,SlimPajama continue-pretrain)**在 lhz 环境下从未成功打出过一个 training step**。模型构造/transplant 全部正确(见下),卡在进 training loop 前。

### 症状(决定性证据:是 NCCL collective 死锁,不是显存/慢)
单机 8 卡 **seq_len=512 最小配置**(显存宽裕 ~38GB/143GB、单机无跨节点)下:
- 日志停在 `[untie] embed/lm_head UNTIED before FSDP` 之后,**~5 分钟无更新、无 step、无报错**。
- `nvidia-smi` 连续三次采样**完全一致**:**7 张卡恒 100% util、GPU4(rank4)恒 0%**。
- 训练进程 CPU 249%/99%(忙等,不是 D 状态挂起)。
- → 经典 **rank4 掉队 → 其余 7 rank 在 collective(FSDP all-gather/reduce-scatter)里忙等它** 的死锁形态。

### 模型构造是对的(排除 transplant/架构问题)
每次日志都打:`[transplant] copied 185 tensors (front 13 layers) max|model-base|=0.000e+00 (exact)`、`fresh tail [13,14]`、`model params=37.963B num_hidden_layers=15`、untie PASS。**15 层(13 keep + 2 fresh)模型搭对了**,问题纯在 FSDP 分布式执行。

### 各配置尝试链(全部在"进 loop 前"失败,同一根因的不同表现)
| 配置 | 结果 | 表现 |
|---|---|---|
| 单机8卡 seq2048 + fsdp_cpu_offload=1(全开) | 崩 | SIGSEGV(38B 全可训参数 CPU pinned optimizer 爆) |
| 单机8卡 seq2048 offload=0 | 崩 | OOM(差 4.65GB;后 backward_prefetch=BACKWARD_POST 后差 3.0GB) |
| 单机8卡 seq1024 offload=0 | 起来但**无 step** | 卡住(被 cancel) |
| 16卡 seq2048 走 eth0(10Gb/s) | 崩 | NCCL watchdog SIGABRT(跨节点太慢 ~60-90s/step,超时) |
| 16卡 seq2048 走 IB(mlx5 400Gb/s NDR,GDRDMA) | 崩 | rank2 `ncclSystemError`,进 loop ~2min 无 step |
| **单机8卡 seq512 小配置 offload=0** | **死锁** | rank4 恒 0%、其余忙等(上述症状,最干净复现) |

**统一根因判断**:某个 rank 在 **FSDP wrap 后首个 collective** 卡住。多机时表现为 SIGABRT/ncclSystemError/watchdog-timeout,单机时表现为纯死锁。与 seq_len / 显存 / 多机 / 网络**均无关**(小配置全绕开仍死锁)。

### 环境背景(可能相关)
- lhz/lhz2:8× **H200 143GB**,系统 torch **2.8.0a0+...nv25.05**(NVIDIA 优化版),transformers 5.13.1,`.venv_hy3`(共享盘)。
- A13B = 公开 `HunYuanMoEV1`(32层 MoE,64 experts),`experts_implementation="eager"`(为避 torch2.8-nv grouped_mm kernel 的 `GroupMMCommon.cuh:51 delta%16==0` 断言崩溃,见 HARU.md 坑2)。
- FSDP FULL_SHARD + MixedPrecision(bf16 compute/fp32 reduce)+ fp32 master + sync_module_states + BACKWARD_POST + use_orig_params。
- 对比:Hy3(train_hyv3_probe2.py,keep36+fresh2)在**旧集群 L20A 183GB** 上跑过,但**也只是 max_steps=200 的 smoke、本地无成功 step 日志**——不能确定 Hy3 配方在"真出 step"上验证过。IB smoke(纯 all_reduce,含 2GB 大消息)在本集群**是 PASS 的**,所以 IB 硬件/带宽本身没问题,是 FSDP 训练特定的 collective pattern 触发。

### 疑似根因(待验证,优先级从高到低)
1. **特定物理卡故障(GPU4)**:rank4 恒 0% 掉队 → 可能 GPU4 本身有 Xid/ECC/掉总线问题。**验证**:`nvidia-smi -i 4`、`dmesg|grep -i xid`;`CUDA_VISIBLE_DEVICES` 避开 4 跑同样 smoke,看是否就通。
2. **FSDP + MoE eager 的 collective pattern 在 torch2.8-nv 上的 bug**:experts_implementation=eager 的 per-expert loop 可能在某 rank 产生不对称的 collective(不同 rank 走不同 expert 分支 → collective 数量/顺序不一致 → 死锁)。这是 MoE + FSDP 的经典坑。
3. **IB SHARP 集合卸载**(仅多机):日志有 mlx5 SHARP,`ncclSystemError` 可能是 SHARP 在 reduce-scatter 出错 → 试 `NCCL_COLLNET_ENABLE=0`。

### 下一步(未做完的二分,恢复工作时从这里继续)
- **[关键二分 A] `--nproc 1` 单卡 seq512**(无 FSDP、无 collective):出 step → 训练器逻辑 OK,问题纯在 FSDP;卡住 → 模型 forward(MoE eager)本身在 lhz 卡。
- **[关键二分 B] 避开 GPU4**(`CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7` 或换 4 卡子集)跑 smoke:通 → GPU4 坏卡;仍卡 → 逻辑 rank 问题非物理卡。
- 设 `NCCL_DEBUG=INFO TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576`,超时 dump flight recorder 看 rank4 卡在哪个 collective op。
- 若确认是 MoE+FSDP eager 的 collective 不对称:考虑 (a) 换 SHARD_GRAD_OP 而非 FULL_SHARD;(b) 不 wrap MoE expert 层单独 wrap;(c) 换非 nv 版 torch 对照。

### 已确认 OK 的(不用重查)
- 模型 transplant/构造(15层 exact)、j=13 结论(QCMem j-sweep,见 HARU.md)、数据(slimpajama_chunks_2048_hunyuan.npy 338514×2048)、环境(.venv_hy3)、IB 硬件+带宽(all_reduce smoke PASS)、16卡 NCCL 拓扑(world16 通)。
