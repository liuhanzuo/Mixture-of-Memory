# Paper B P2.4 — ShortGPT-16@200k SFT arm on `.252` (task sibling of #123)

**状态**：`[RUNNING]`
**日期**：2026-08-08 CST | **节点**：`.252`（8×L20A cc10.0，183 GiB/卡，wzc1 盘）
**执笔**：experiment agent（ShortGPT-16 分支）。不修改 `PAPERB_P24_SFT_REPAIRABILITY.md`（sibling 拥有）与 `PAPERB_P24_FULL32_ARM.md`（.252 full32 sibling 拥有）。

---

## 1. 一句话现状

ShortGPT-16@200k 臂已在 `.252` 启动，与 `.73` 上 keep14+fresh2 臂 **byte-identical**（唯一逻辑差异 = 起始 checkpoint 与 arm_name/output_dir，加上 base_model 路径的 wzc1 vs zwfy6 磁盘 artifact，HF 内容逐字节一致）。三臂共享清洗后 SFT 数据（wzc1 md5 与 zwfy6 逐字节一致）、token budget、optimizer、seed、grad-ckpt。

**无 NaN**：step 20 loss = **1.6719 finite**；step 40 = 1.4377（下降中）。数据 npy md5 `b1e6fe4e...` / `bf7c5774...`，与 sibling 报告匹配。

---

## 2. 与 `.73` keep14fresh2 臂的 launch 逐行 diff（canonical source = 运行中的 `.73` ps 输出）

`.73` keep14fresh2 命令（截取自 21:xx CST 启动的 running PID 2694898，2026-08-07 22:53 起）:
```
python torchrun --nproc_per_node 8 --master_port 29535 scripts/train_olmo2_sft.py
  --arm_name keep14fresh2
  --ckpt outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt
  --base_model /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B
  --sft_ids data/olmo2_sft/tulu3_general_clean_input_ids.npy
  --sft_labels data/olmo2_sft/tulu3_general_clean_labels.npy
  --output_dir outputs/olmo2_p24_sft_keep14fresh2
  --max_steps 842 --batch_size 1 --grad_accumulation_steps 16 --seq_len 2048
  --lr 1e-5 --min_lr 1e-6 --warmup_steps 100 --weight_decay 0.1
  --seed 42 --gradient_checkpointing 1
```

`.252` shortgpt16 命令（本 arm 的 running PID 2995894，2026-08-08 00:58 起）:
```
python torchrun --nproc_per_node 8 --master_port 29535 scripts/train_olmo2_sft.py
  --arm_name shortgpt16
  --ckpt outputs/olmo2_probe2_7B_shortgpt16/step200000.pt
  --base_model /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B
  --sft_ids data/olmo2_sft/tulu3_general_clean_input_ids.npy
  --sft_labels data/olmo2_sft/tulu3_general_clean_labels.npy
  --output_dir outputs/olmo2_p24_sft_shortgpt16
  --max_steps 842 --batch_size 1 --grad_accumulation_steps 16 --seq_len 2048
  --lr 1e-5 --min_lr 1e-6 --warmup_steps 100 --weight_decay 0.1
  --seed 42 --gradient_checkpointing 1
```

**Diff = 3 语义 + 1 路径 artifact**：
| 参数 | `.73` keep14fresh2 | `.252` shortgpt16 (本 arm) | 类别 |
|---|---|---|---|
| `--arm_name` | `keep14fresh2` | `shortgpt16` | 语义（spec 要求） |
| `--ckpt` | `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` | `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt` | 语义（spec 要求，唯一变量） |
| `--output_dir` | `outputs/olmo2_p24_sft_keep14fresh2` | `outputs/olmo2_p24_sft_shortgpt16` | 语义（spec 要求） |
| `--base_model` | `/apdcephfs_zwfy6/.../OLMo-2-1124-7B` | `/apdcephfs_wzc1/.../OLMo-2-1124-7B` | **路径 artifact** — .73 在 zwfy6 盘，.252 在 wzc1 盘；两份 HF checkpoint 是 byte-identical 6-shard safetensors（此为 sibling full32 status doc §2 已确立的规范：跨盘副本 HF 内容一致，路径差不构成语义差异） |

其余**完全一致**：
- `--max_steps 842 --batch_size 1 --grad_accumulation_steps 16 --seq_len 2048`
- `--lr 1e-5 --min_lr 1e-6 --warmup_steps 100 --weight_decay 0.1 --seed 42 --gradient_checkpointing 1`
- `--sft_ids` / `--sft_labels` 相对路径完全一致（`data/olmo2_sft/tulu3_general_clean_{input_ids,labels}.npy`）
- eff_batch = 1 × 16 × 8 = **128**
- token_budget = 842 × 128 × 2048 = **220,725,248 tok**
- torchrun `--nproc_per_node 8 --master_port 29535`，`/opt/conda/envs/torch-base/bin/python3.14`，torch 2.13.0 / transformers 4.57.6

---

## 3. 与 `.252` full-32L 臂的 launch 逐行 diff

`.252` full32 命令（DONE 00:38 CST；从 log 提取 `arm=full32` + status doc §2）:
```
python torchrun --nproc_per_node 8 --master_port 29535 scripts/train_olmo2_sft.py
  --arm_name full32
  # (no --ckpt — trainer 走 load_base_model / full_base 路径)
  --base_model /apdcephfs_wzc1/.../OLMo-2-1124-7B
  --sft_ids ... --sft_labels ...  --output_dir outputs/olmo2_p24_sft_full32
  --max_steps 842 --batch_size 1 --grad_accumulation_steps 16 --seq_len 2048
  --lr 1e-5 --min_lr 1e-6 --warmup_steps 100 --weight_decay 0.1
  --seed 42 --gradient_checkpointing 1
```

**Diff = 3 处**：
| 参数 | `.252` full32 | `.252` shortgpt16 (本 arm) |
|---|---|---|
| `--arm_name` | `full32` | `shortgpt16` |
| `--ckpt` | **省略**（trainer 走 `load_base_model` full_base 路径） | `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt` |
| `--output_dir` | `outputs/olmo2_p24_sft_full32` | `outputs/olmo2_p24_sft_shortgpt16` |

其余 byte-identical（含 `--base_model` = wzc1 路径完全一致）。

**结论：single-variable 纪律成立。** 三个 arm 相互之间只在 arm_name / ckpt 路径 / output_dir 上不同；其他 optimizer / seed / token_budget / seq_len / grad_ckpt / warmup / lr / weight_decay 完全一致。

---

## 4. `final.pt` hardcode 疑问的实测澄清（对派单文本的小修正）

派单文本担心 trainer 存在「pruned arm ckpt 必须命名 `final.pt`」的 hardcode，需要 `ln -s step200000.pt final.pt` 绕过。**实测该 hardcode 不存在**：
- `scripts/train_olmo2_sft.py` line 130-131：`if args.ckpt: model, meta = load_pruned_model(args.ckpt, ...)` —— 直接使用传入路径，无 `final.pt` 拼接。
- `.73` keep14fresh2 臂的 running command 也直接传 `--ckpt .../step200000.pt`，未用 `final.pt`。
- 唯一使用「final.pt」名字的地方是 pipeline 脚本 `_run_olmo2_p24_sft_pipeline.sh`（sibling 报告 §5 已标注），我们不走 pipeline 而是直接调 trainer，因此不受此影响。

**巧合但无关**：`outputs/olmo2_probe2_7B_shortgpt16/final.pt` 已经存在（48.7 GB，2026-08-01 16:13 生成），这是 shortgpt16 **healing 阶段本身**保存的 final.pt（healing trainer 在最终 step 存的），不是本次 SFT 的产物，也不是任何 agent 手工创建的 symlink。我们**不使用它**——按 spec 明确传入 `step200000.pt`，与 `.73` keep14 臂完全对称。

未改 `scripts/train_olmo2_sft.py`，未创建任何 symlink。

---

## 5. 数据 md5 验证（wzc1 副本 vs sibling zwfy6 源，逐字节一致）

| 文件 | wzc1 md5（本 arm 读的） | sibling 报告 (`PAPERB_P24_SFT_REPAIRABILITY.md` §3 / `FULL32_ARM.md` §3) | 匹配 |
|---|---|---|:-:|
| `tulu3_general_clean_input_ids.npy` | `b1e6fe4e11351e208da24b03d96a762a` | `b1e6fe4e11351e208da24b03d96a762a` | ✅ |
| `tulu3_general_clean_labels.npy` | `bf7c57746f05b1ac73ccdaa07b1481b7` | `bf7c57746f05b1ac73ccdaa07b1481b7` | ✅ |

数据源自 sibling 报告 §4.2 的清洗（commit `0fd051a`，丢弃 11.74% 全 -100 行，得 107,740 行）。log 里 `[data] sft rows=107740 seq_len=2048` 与该数字一致。

---

## 6. 内存与 arm 参数（预期 vs 实测）

| Arm | params | fp32 AdamW 静态 = 16 B/param | 节点 GPU 内存 | 实测 maxmem | fit? |
|---|---:|---:|---:|---:|:-:|
| full-32L (`.252`) | 7.298 B | 108.8 GiB | L20A 183 GiB × 8 | 182 GiB (99.3%) | ✅ |
| keep14+fresh2 (`.73`) | 4.060 B | 60.5 GiB | H20 97.8 GiB × 8 | ~96 GiB (98%) | ✅ |
| **ShortGPT-16 (本 arm, `.252`)** | 4.060 B | 60.5 GiB | L20A 183 GiB × 8 | **101 GiB (55%)** | ✅ 舒服 |

派单文本预测 ~55 GiB 与实测 101 GiB 略有偏差（约 40 GiB 差），可能是 bf16 activations + grad_ckpt 存 checkpoint tensor + NCCL buffer 的 headroom。仍完全在 183 GiB 内，无任何压力。

---

## 7. 命令实测（PID / log / 状态）

- **PID**：`2995894`（torchrun 进程，8× 子 worker）
- **log**：`logs/p24_sft_shortgpt16.log`
- **output_dir**：`outputs/olmo2_p24_sft_shortgpt16/`
- **git commit**：`673f610`（LOCAL + `.252` 同 HEAD；trainer 未改）

step 20 (`2026-08-08 01:00:39`)：loss=**1.6719**（finite），lr=1.90e-06（warmup 中）,  2.23s/step。
step 40 (`2026-08-08 01:01:21`)：loss=**1.4377**（下降），lr=3.90e-06, 2.18s/step。
无 NaN，无 loss 爆炸，warmup + AdamW 状态正常。

---

## 8. Post-launch 健康门（sibling §5 定义）

Step 20 loss = **1.6719 finite** → 通过。
若 step 20 loss = NaN → 立即 kill；诊断可能包括：
- npy 未匹配：`md5sum` 验证（本 arm 已核，通过）
- 数据加载 shift 错误
- OOM 假 NaN

---

## 9. ETA 与完成后交付

- **ETA**：842 × 2.18s ≈ 30.6 min 总运行 → 从 00:58 起，**约 2026-08-08 01:30 CST 完成**。（派单预测 ~40 min，实测更快，因为 2.18 s/step vs 预测 2.5-3 s/step；full32 是 4.5 s/step 的 ~2× 减半符合 16L 半深度。）
- **交付**：
  - `outputs/olmo2_p24_sft_shortgpt16/final.pt`
  - 完整 log `logs/p24_sft_shortgpt16.log`
  - 三臂 post-SFT eval battery（另启，非本 launch 范围；由 MAIN 排程）

**不写入论文任何 P2.4 结论**（sibling §7：三臂 + post-SFT eval 全就位前四路裁决不成立）。

---

## 10. 运行状态

见 `status/gpu_runs.jsonl`（本 arm 已 append，2026-08-08T00:58:00+0800）与 `status/GPU_STATUS.md`（.252 块已换为 shortgpt16）。

`.73` keep14fresh2 应在 ~01:00 CST 完成（step 740/842 at 00:49 → ETA ~11 min at 9.6 s/step）；本 arm 大约同期完成。**三臂全部完成后即可启动 post-SFT eval battery**。
