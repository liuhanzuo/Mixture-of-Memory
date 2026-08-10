# H20 训练暂停→eval→resume（2026-07-22 22:3x，用户指令）

用户指令：暂停 H20 上的训练任务，先跑 eval，跑完再 resume。
H20 训练臂 = **.73 freeze_front** + **.104 keep8**（.82 已在跑 eval；LOCAL=L20A、.252=B200 非 H20，继续训不动）。

训练脚本 `scripts/train_olmo2_arch_probe2.py` 支持 `--resume_from`（恢复 model+optimizer+global_step+epoch+RNG，跳过 front-layer transplant）。
数据 `/dev/shm/dolmino_now15b.npy` kill 后仍在（/dev/shm 持久到 reboot），resume 直接复用。

## 暂停时状态（paused @ 2026-07-22 22:3x）
- **.73 freeze_front**：step21840 loss2.88 ppl17.80；最新 ckpt = `outputs/olmo2_probe2_7B_keep14fresh2_freezefront/step21500.pt`
- **.104 keep8**：step36000 loss2.77 ppl15.95；最新 ckpt = `outputs/olmo2_probe2_7B_keep8fresh2/step36000.pt`

## RESUME 命令（eval 跑完后逐节点执行）

**SSH recipe**：`unset LD_LIBRARY_PATH; /opt/conda/bin/sshpass -f configs/password_h20_853573.txt /usr/bin/ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password root@IP '<cmd>'`（.73=password_h20_853573.txt，.104=password_h20_24104.txt）

### .73 freeze_front resume（在 .73 上）
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
WANDB_MODE=offline setsid nohup /opt/conda/envs/torch-base/bin/python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py --data_path /dev/shm/dolmino_now15b.npy \
  --output_dir outputs/olmo2_probe2_7B_keep14fresh2_freezefront \
  --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  --keep_front_layers 14 --n_fresh_layers 2 --batch_size 4 --grad_accumulation_steps 4 \
  --seq_len 2048 --lr 1e-4 --lr_inherited 2e-5 --max_steps 200000 --gradient_checkpointing 1 --freeze_front \
  --resume_from outputs/olmo2_probe2_7B_keep14fresh2_freezefront/step21500.pt \
  >> logs/olmo2_7B_keep14fresh2_freezefront.log 2>&1 &
```

### .104 keep8 resume（在 .104 上）
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
WANDB_MODE=offline setsid nohup /opt/conda/envs/torch-base/bin/python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py --data_path /dev/shm/dolmino_now15b.npy \
  --output_dir outputs/olmo2_probe2_7B_keep8fresh2 \
  --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  --keep_front_layers 8 --n_fresh_layers 2 --batch_size 4 --grad_accumulation_steps 4 \
  --seq_len 2048 --lr 1e-4 --lr_inherited 2e-5 --max_steps 200000 --gradient_checkpointing 1 \
  --resume_from outputs/olmo2_probe2_7B_keep8fresh2/step36000.pt \
  >> logs/olmo2_7B_keep8fresh2.log 2>&1 &
```

## resume 后验证
- `tail -3 logs/olmo2_7B_*.log` 看到 `[resume] continue @ step=...` + step 继续增长
- `nvidia-smi` 8/8 卡 proc=1、util 上来
- resume 后 `[resume] optimizer state restored` + step 从 ckpt step 续（.73 从 21500，.104 从 36000）

## eval（暂停期间跑）
chat=False 全 benchmark 重跑 campaign（用户论文级政策）：CoMem 8B flagship + baseline × RULER/LongEval/LongBench/BABILong，selector=iter_bm25，无 --use_chat_template。coder 在 .73+.104（16 H20）跑。
