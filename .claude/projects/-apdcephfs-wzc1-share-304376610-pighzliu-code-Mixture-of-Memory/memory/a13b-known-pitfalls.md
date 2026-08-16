---
name: a13b-known-pitfalls
description: 混元 Hunyuan-A13B-Pretrain 剪层+补层 continue-train 的四连坑 + 修复配方
metadata: 
  node_type: memory
  type: project
  originSessionId: 0dac7a11-5048-4ecf-85c3-ff6b9fab88d3
---

混元 `Hunyuan-A13B-Pretrain`（model_type=hunyuan_v1_moe，80层 MoE，vocab128167，hidden4096，**tie_word_embeddings=True**，~160GB bf16）做 Qwen3-8B 式 prune-heal（keep24+fresh2=26层65B）在 8× B200(178GB) FSDP FULL_SHARD 上，2026-07-12 踩了**四连坑**，逐一修复后才能训。脚本 `scripts/train_hunyuan_a13b_probe2.py`，env `.venv_hy3`(tf5.13.1)，launch `scripts/launch_hunyuan_a13b_keep24_fresh2.sh`。

1. **NCCL init 竞争**：rank0 CPU 载 160GB >10min，rank1-7 等 ncclUniqueId 超时 → commit 1d23b4b：`init_process_group(timeout=2h)` + `device_id=torch.device("cuda",local_rank)` eager 建 comm。
2. **NCCL 心跳监控器 480s 误杀**：rank0 组装 65B transplant 期间 rank1-7 卡 barrier busy-wait，`TORCH_NCCL_ENABLE_MONITORING`(默认on)判 watchdog hang → SIGABRT（**独立于 init timeout，先于它杀**）→ 修复 launch env `TORCH_NCCL_ENABLE_MONITORING=0` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`。
3. **FSDP 首 forward tied-embedding 死锁**：tie_word_embeddings=True 下 rank0(materialized) dedup 成 flat-param `[embed,norm]`，meta rank `to_empty` **静默断 tie** → 多注册 lm_head 成 `[embed,norm,lm_head]` → 各 rank root allgather numel 不同 → embed unshard 死锁（py-spy: rank0 在 F.embedding，其余卡 _use_low_precision_shard）→ commit 7f6f049：wrap 前所有 rank `untie_output_embeddings`（lm_head 独立 Parameter + cfg.tie_word_embeddings=False）。
4. **首 step CUDA OOM**：65B FSDP 每卡 shard≈8.1B，fp32 master(32GB)+AdamW 2×fp32(65GB)+bf16 param 撑爆 178GB → commit 6876ca4：`--fsdp_cpu_offload` 默认改 ON（param+grad+optim 卸 CPU RAM，本机 256核/1.9TB），GPU 只留 bf16 unshard+激活。2卡 smoke maxmem 161GB<178。**代价**：CPU AdamW 慢，需 `OMP_NUM_THREADS=16`（256核，8rank×16=128）。

**通用运维坑**：杀 A13B 进程**绝不用 `pkill -f train_hunyuan_a13b`**——pgrep/pkill 会匹配到含该字符串的**自身 shell 命令行**，`kill -9 $(pgrep...)` 把自己 shell 杀了（表现=命令无输出秒退）。用显式 PID 或 `nvidia-smi --query-compute-apps=pid`。判 A13B "慢 vs 死锁"：多采样功耗，死板恒定=NCCL 自旋死锁，波动=真算。相关 [[heartbeat-preauth-kill-rm]]。