# Paper B keep8/keep10/keep12 Resume 200k v2 — 启动记录

Created: 2026-08-08 14:35

---

## 一、补丁 diff 摘要

**commit**: `5578ed0`  
**文件**: `scripts/train_olmo2_arch_probe2.py`  
**改动**: lines ~886-972，原 15 行 try/except 扩展为 ~87 行，新增 `elif n_ckpt_groups == 2 and n_new_groups == 4` 分支

核心逻辑：
1. 从 `resume_ckpt["model_state"]` 推断旧 2-group 方案的 name→index 映射：
   - group 0 = ndim>=2 参数，按 model_state_dict.keys() 顺序编号 0..N2-1
   - group 1 = ndim<2 参数，编号 N2..N2+N1-1
2. 遍历 HEAD optimizer 的 4 groups，用 data_ptr+shape 找到每个参数对应的 bare name
3. 将 old_state[old_i] 的 `{step, exp_avg, exp_avg_sq}` 拷贝到 `optimizer.state[p]`
4. remap 失败时 fallback 到 WARM-RESTART（不崩训练）

**md5 核对**（wzc1 ↔ zwfy6 两盘一致）：`879541f001568ceea16528e2e5d8035f`

---

## 二、三个 arm 的启动命令（逐字）

### keep10 (.82)
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
export WANDB_API_KEY="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"
export NCCL_IB_DISABLE=1; export NCCL_SOCKET_IFNAME=bond1
setsid nohup /opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node 8 \
    scripts/train_olmo2_arch_probe2.py \
    --resume_from outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt \
    --keep_front_layers 10 --n_fresh_layers 2 \
    --batch_size 4 --grad_accumulation_steps 4 --seq_len 2048 \
    --lr 2e-5 --min_lr 2e-6 --lr_inherited 2e-5 \
    --max_steps 200000 --warmup_steps 150 --weight_decay 0.1 \
    --gradient_checkpointing 1 \
    --save_every 500 --milestone_every 5000 --keep_last_n 3 --keep_milestones 8 \
    --keep_steps 83500,121000,124000,150000,175000,200000 \
    --data_path /dev/shm/dolmino_now15b.npy \
    --output_dir outputs/olmo2_probe2_7B_keep10fresh2 \
    --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
    > logs/olmo2_7B_keep10fresh2_resume200k_v2.log 2>&1 < /dev/null &
```
PID: 1418803

### keep12 (.104)
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
# same env vars
setsid nohup /opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node 8 \
    scripts/train_olmo2_arch_probe2.py \
    --resume_from outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt \
    --keep_front_layers 12 --n_fresh_layers 2 \
    ... (same flags, keep12-specific values)
    --output_dir outputs/olmo2_probe2_7B_keep12fresh2 \
    > logs/olmo2_7B_keep12fresh2_resume200k_v2.log 2>&1 < /dev/null &
```
PID: 3475263

### keep8 (.73)
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
# same env vars
setsid nohup /opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node 8 \
    scripts/train_olmo2_arch_probe2.py \
    --resume_from outputs/olmo2_probe2_7B_keep8fresh2/step121000_full.pt \
    --keep_front_layers 8 --n_fresh_layers 2 \
    ... (same flags, keep8-specific values)
    --output_dir outputs/olmo2_probe2_7B_keep8fresh2 \
    > logs/olmo2_7B_keep8fresh2_resume200k_v2.log 2>&1 < /dev/null &
```
PID: 3291542  
注意: resume_from 用的是 `step121000_full.pt`（34.2G scp 版），非 `step121000.pt`（11.4G 被剥版）

---

## 三、log 证据行

### keep10 (step83500 -> 200000)
```
[optim] group fresh_decay: 815.8M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group fresh_nodecay: 0.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_decay: 2434.8M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay: 0.2M params base_lr=2.00e-05 min_lr=2.00e-06
[resume] ckpt has 2 groups, optimizer has 4 groups; applying keep10/12/8 compatibility remap...
[resume] optimizer state REMAPPED 2-group -> 4-group (135/135 param states, Adam moments preserved)
[resume] continue @ step=83500 epoch=0 warmup=150 max_steps=200000 lr_fresh(now)=1.332e-05 lr_inh(now)=1.332e-05
[step 83520/200000] loss=2.5249 ppl=12.49 lr=1.33e-05 gnorm=0.52 6.86s/step maxmem=82.7GB
```

### keep12 (step124000 -> 200000)
```
[optim] group fresh_decay: 815.8M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group fresh_nodecay: 0.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_decay: 2839.5M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay: 0.2M params base_lr=2.00e-05 min_lr=2.00e-06
[resume] ckpt has 2 groups, optimizer has 4 groups; applying keep10/12/8 compatibility remap...
[resume] optimizer state REMAPPED 2-group -> 4-group (157/157 param states, Adam moments preserved)
[resume] continue @ step=124000 epoch=1 warmup=150 max_steps=200000 lr_fresh(now)=7.694e-06 lr_inh(now)=7.694e-06
[step 124020/200000] loss=2.3699 ppl=10.70 lr=7.69e-06 gnorm=0.49 8.02s/step maxmem=91.9GB
```

### keep8 (step121000 -> 200000)
```
[optim] group fresh_decay: 815.8M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group fresh_nodecay: 0.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_decay: 2030.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay: 0.1M params base_lr=2.00e-05 min_lr=2.00e-06
[resume] ckpt has 2 groups, optimizer has 4 groups; applying keep10/12/8 compatibility remap...
[resume] optimizer state REMAPPED 2-group -> 4-group (113/113 param states, Adam moments preserved)
[resume] continue @ step=121000 epoch=1 warmup=150 max_steps=200000 lr_fresh(now)=8.093e-06 lr_inh(now)=8.093e-06
[resume] sampler.set_epoch(1) (deterministic reshuffle for this epoch)
[step 121020/200000] loss=2.5702 ppl=13.07 lr=8.09e-06 gnorm=0.48 5.86s/step maxmem=73.5GB
```

**三个 arm 全部 REMAPPED，无任何 WARM-RESTART，全部 base_lr=2.00e-05。**

---

## 四、s/step 实测与 ETA

| arm | s/step | 缺口 steps | ETA |
|-----|--------|-----------|-----|
| keep10 | 6.80 s/step | 116500 | ~220 h (~9.2 天) |
| keep12 | 7.87 s/step | 76000 | ~166 h (~6.9 天) |
| keep8 | 5.86 s/step | 79000 | ~128 h (~5.3 天) |

---

## 五、keep8 scp 处理过程

- 源：`/apdcephfs_wzc1/.../outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt` (34,152,195,666 B)
- 目标：`/apdcephfs_zwfy6/.../outputs/olmo2_probe2_7B_keep8fresh2/step121000_full.pt` (34,152,195,666 B)
- 命令：`scp -O` (zwfy6 sftp subsystem 已坏，必须用 -O legacy protocol)
- 时间：13:54 ~ 14:27，约 33 分钟，平均约 17 MB/s
- md5 校验：`0e710ce273b8a6c7af71605cef673ab8`（wzc1 源 = zwfy6 目标，完全一致）
- 原 step121000.pt（11.4G 被剥版）保持不变，未覆盖

---

## 六、首次启动的小插曲

第一轮启动因 `--wandb_project mixture-of-memory` 参数不被脚本支持（该脚本无 wandb argparse flag）而崩溃。移除该参数后第二轮启动成功。WANDB_API_KEY 环境变量仍设置（训练脚本内部通过 os.environ 读取）。
