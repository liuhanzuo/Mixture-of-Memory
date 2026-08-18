# Remote Server Inventory

Inventory time: `2026-07-22 11:01–11:04 Asia/Shanghai`

## Access and host

- SSH: `root@28.82.250.82`
- Hostname: `TENCENT64.site`
- OS: TencentOS Server 4
- Kernel: `5.4.241-1-tlinux4-0017.7`
- Runtime appears to be a Kubernetes/container allocation rather than a normal
  persistent bare-metal host.

## Compute

- CPU: 2 sockets × AMD EPYC 9K84, 96 physical cores/socket
- Logical CPUs: 384
- NUMA nodes: 2
- RAM: approximately 2.2 TiB
- Swap/zram: approximately 6.6 TiB

## GPUs and interconnect

- 8 × NVIDIA H20
- Reported memory: 97,871 MiB/GPU, approximately 95 GiB usable
- Driver: `535.247.01`
- CUDA runtime/toolkit reported by the image: `13.2`
- All eight GPUs are connected to one another with `NV18`
- GPUs 0–3 are CPU-NUMA 0; GPUs 4–7 are CPU-NUMA 1

This topology is suitable for 8-way DDP/FSDP and, when necessary, tensor
parallelism. For a 7B model on 96GB GPUs, pure or sharded data parallelism should
be benchmarked before adding TP, because TP may reduce arithmetic efficiency
without being required for memory.

## Software

- Conda: `26.3.2`
- Active environment: `/opt/conda/envs/torch-base`
- Python: `3.14.6`
- PyTorch: `2.13.0`
- Torch CUDA: `13.2`
- Transformers: `5.5.4`
- Accelerate: `1.14.0`
- Datasets: `5.0.0`
- Available: `torchrun`, `tmux`, `uv`, Git, curl, wget
- Not currently installed: DeepSpeed, FlashAttention, TRL, vLLM, verl, wandb
- No Slurm commands were found.
- No usable `crontab` command was found.
- `systemctl` exists, but this container does not expose a normal persistent
  system service setup.

The unusually new Python/PyTorch/CUDA combination is unlikely to match released
Dream/DreamOn requirements exactly. A separate pinned environment will be
created rather than mutating `torch-base`.

## Storage

- Root overlay: approximately 12 TB total, 3.4 TB free
- `/dockerdata`: 9 TB free
- Shared project filesystem:
  `/apdcephfs_zwfy6/share_304376610`, approximately 145 TB free
- `/jizhicfs`: approximately 25 TB free but already 90% utilized
- `/cfs_hy_aide/common`: only approximately 7.1 TB free and 96% utilized

Primary persistent project path:

```text
/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft
```

Use `/dev/shm` or local NVMe only for reconstructible caches. Checkpoints and
source should remain on the shared project filesystem.

## Existing GPU workload

At inventory time all eight GPUs were already occupied by a healthy pre-existing
run:

```text
torchrun --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --output_dir outputs/olmo2_probe2_7B_keep12fresh2 \
  --max_steps 200000 ...
```

- Working directory:
  `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`
- Parent PID at inventory: `490809`
- Progress at 11:01: `21,580 / 200,000`
- Typical speed: `7.81 s/step`
- Loss: finite, around 2.65–2.72 in the inspected window
- GPU utilization: 100% on every GPU
- Peak allocated memory reported by the run: 91.9GB/GPU
- Checkpoint cadence: every 500 steps
- Checkpoint size: approximately 40.9 GiB
- Latest inspected checkpoint: `step21500.pt`
- No OOM, traceback, NCCL failure, NaN loss, or disk error was found.
- Rough remaining time at the current speed: approximately 16 days.

This workload is registered as an **external read-only run**. The Scaffold-Coder
watchdog may report its status but must never terminate or restart it.

## Operational constraints

- `/root/.ssh` is mounted read-only; passwordless key installation failed.
- The 30-minute heartbeat therefore runs directly on the server in a detached
  tmux session.
- A tmux heartbeat survives SSH disconnects but not allocation/container
  replacement. Because neither cron nor a persistent system service is
  available, restart-on-container-recreation cannot be guaranteed from inside
  this allocation.

