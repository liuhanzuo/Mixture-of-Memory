# GPU_STATUS.md — 三节点 GPU 实时台账
> 每次启动/kill GPU任务更新. heartbeat先读→对照nvidia-smi→台账说跑但空=补卡. ★29.162.226.120已归还(2026-07-11用户拿回给QCMem用=28.83.24.104). 最后更新：2026-07-12 (本机启动 Hy3 256k)

## 节点清单(3节点=24卡)
- **本机 wzc1 B200/L20A(8卡)**: .venv torch2.10 sm_100
- **H20 .53.31 (28.83.53.31:36000, diskB)**: .venv→torch-base torch2.10+tf5.5.4, 密码 configs/password_h20_3153.txt
- **★H20 .24.104 (28.83.24.104:22, diskB同.53.31盘, 2026-07-11回归)**: torch-base torch2.13+tf5.5.4+pandas+peft(已补装), 密码 configs/password_h20_24104.txt(末尾逗号). 8×H20 97.8GB. 与.53.31共享diskB→代码/模型/数据无需rsync

## 本机 B200/L20A (wzc1, 8卡) — 满载 Hy3 QCMem 256k RULER eval (2026-07-12 长档最后一档)
- **在跑**: Hy3 (hy_v3 80层MoE 597GB) device_map=auto 8卡分片 + 蒸馏adapter(outputs/qcmem_distill_hy3_j32_r32/final, j32/LoRA r32) QCMem **256k** RULER eval
  - 脚本 scripts/eval_ruler_qcmem_hy3.py (commit 52e5bbf): bm25 topk=8 恒定read + resume_j=32 + 官方 string_match_all 判分 + device_map分片 + jsweep验证的手动LoRA加载(绕peft distributed_operation bug)
  - 任务: niah_single_2/niah_multikey_1 × 256k, limit=50, log logs/hy3_ruler_256k.log, out ruler_results/hy3_qcmem_j32_256k/. PID 4006214, 起 2026-07-12, 预计~90-100min.
  - ✅ 已确认: Hy3 107s加载分片8卡(每卡70-91GB), LoRA真加载(672 tensors, sum|lora_B|=7.24e4>>0), eval跑起 ~52s/sample
  - ★"起不来"根因 = `import eval_ruler_mem_space` 需~22s(fla triton+tf+sklearn) > 15s timeout → --help/前台都被 timeout 杀掉; 脚本本身无bug. 256k 在 _LENGTH_TOKENS 已支持(262144), topk=8恒定read → 与128k同量级(~4.3-4.6k)可跑不OOM.
- 16k-128k 全 DONE: niah_single 16k=98/32k=92/64k=100/128k=98; niah_multikey 16k=100/32k=94/64k=100/128k=100 (蒸馏 16k tax 1.386→1.078)

## H20 .53.31 (8卡) — MemoryLLM RULER补档
## H20 .24.104 (8卡, 新回归) — MemoryLLM RULER补档(niah_single/multikey/vt × 8k/16k/32k, _n3系列)

## 待决/进行
- Qwen3-8b probe#2设计workflow(wdy2jr8my)跑中→出实现spec(方向4 scale到8B真实模型)
- 3b dim已定局(d512+4.7%<1B5.9%/d256+5.8%<8.5% = model越大税越小). 3b layer待L9v2/L3v2收尾
