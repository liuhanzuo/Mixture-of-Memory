# Paper B P2.4 — full-32L base SFT arm on `.252` (task sibling of #123)

**状态**：`[RUNNING]`
**日期**：2026-08-07 CST | **节点**：`.252`（8×L20A cc10.0，183 GiB/卡，wzc1 盘）
**执笔**：experiment agent（full-32L 分支）；不修改 `status/PAPERB_P24_SFT_REPAIRABILITY.md`（sibling 拥有）。

---

## 1. 一句话现状

full-32L base 臂已在 `.252` 启动，与 `.73` 上 keep14+fresh2 臂 **byte-identical 除起始 checkpoint 外**。三臂共享 SFT 数据（干净 npy，md5 通过）、token budget、optimizer、seed、grad-ckpt。

**无 NaN**：step 20 log 报 finite loss；用干净 npy `tulu3_general_clean_*`（`0fd051a` 的修复生效）。

---

## 2. 与 `.73` keep14fresh2 臂的 launch 逐行 diff

**只有 3 处**（arm 语义决定的最小差异）：
| 参数 | `.73` keep14fresh2 | `.252` full32 (本 arm) |
|---|---|---|
| `--arm_name` | `keep14fresh2` | `full32` |
| `--ckpt` | `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` | **省略**（trainer 判 `full_base` 路径，`load_base_model` 走 `from_pretrained`） |
| `--output_dir` | `outputs/olmo2_p24_sft_keep14fresh2` | `outputs/olmo2_p24_sft_full32` |

其余**完全一致**：
- `--base_model /apdcephfs_{wzc1|zwfy6}/.../pighzliu_code/models/OLMo-2-1124-7B`（两盘 HF checkpoint 完全一样，均 6 shard safetensors）
- `--sft_ids/--sft_labels data/olmo2_sft/tulu3_general_clean_{input_ids,labels}.npy`（wzc1 副本 md5 与 zwfy6 源一致，见 §3）
- `--max_steps 842 --batch_size 1 --grad_accumulation_steps 16 --seq_len 2048 --lr 1e-5 --min_lr 1e-6 --warmup_steps 100 --weight_decay 0.1 --seed 42 --gradient_checkpointing 1`
- eff_batch = 1 × 16 × 8 = **128**（与 keep14 臂一致）
- token_budget = 842 × 128 × 2048 = **220,725,248 tok**（与 keep14 臂一致）
- torchrun `--nproc_per_node 8`, `torch-base` conda env（torch 2.13.0, transformers 4.57.6, CUDA cc10.0）
- 数据版本：clean_manifest 由 commit `0fd051a` 生成（丢弃 11.74% 全 -100 行）

**唯一 python 差异**：`.73` 用 `python3.14` conda，`.252` 用 `python3.11` conda（两者均 `/opt/conda/envs/torch-base/bin/python`，torch 2.13.0/transformers 4.57.6 数字与硬件加速路径一致，非语义 drift）。

**未改 `scripts/train_olmo2_sft.py`**（sibling 声明 MAIN 拥有该文件）。full-32L 走 `--full_base` 内建路径，**不需要 `final.pt` 符号链接** —— 该 workaround 只对两个剪层臂有效，本臂 `load_base_model` 直接从 HF 目录读 safetensors。

---

## 3. 数据传输（zwfy6 → wzc1，md5 双端一致）

`.252` 是 wzc1，不能直接 read zwfy6。方案 = 从 `.73`（zwfy6 源）`scp -O` 到 **LOCAL wzc1 共享路径**，`.252` 通过共享 wzc1 直接读。

| 文件 | zwfy6 源大小 | wzc1 副本大小 | zwfy6 md5 | wzc1 md5 | 匹配 |
|---|---:|---:|---|---|:-:|
| `tulu3_general_clean_input_ids.npy` | 882,606,208 B | 882,606,208 B | `b1e6fe4e11351e208da24b03d96a762a` | `b1e6fe4e11351e208da24b03d96a762a` | ✅ |
| `tulu3_general_clean_labels.npy` | 882,606,208 B | 882,606,208 B | `bf7c57746f05b1ac73ccdaa07b1481b7` | `bf7c57746f05b1ac73ccdaa07b1481b7` | ✅ |

`.252` 通过共享 wzc1 mount 直接看到相同 md5（实测通过）。传输耗时 ~43s/文件 @ ~20 MB/s。

**注**：sibling 报告 §9 记录 wzc1 主仓也已由 `scp -O` 同步到 zwfy6，但主仓那次同步的是**代码**，数据 npy 只存在于 zwfy6；本次是**数据**首次进 wzc1。

---

## 4. 内存与 arm 参数（预期 vs 实测）

| Arm | params | fp32 AdamW 静态 = 16 B/param | 节点 GPU 内存 | fit? |
|---|---:|---:|---:|:-:|
| **full-32L** (本 arm) | **7.298 B** | 108.8 GiB | L20A 183 GiB × 8 | ✅ |
| keep14+fresh2 (16L) | 4.060 B | 60.5 GiB | H20 97.8 GiB × 8 | ✅ |

Sibling 硬约束验证：full-32L 在 H20（97.8 GiB）不可行；`.252` 的 L20A 183 GiB 足够。

---

## 5. 命令实测（PID / log / 状态）

- **PID**：见 §8 状态字段（launch 时刻记录）
- **log**：`logs/p24_sft_full32.log`
- **output_dir**：`outputs/olmo2_p24_sft_full32/`
- **git commit**：`d29f4bc`（LOCAL + `.252` 同 head；trainer 未改）

---

## 6. Post-launch 健康门（sibling §5 定义）

Step 20 loss = **finite**（非 NaN），持续下降 → 通过。
若 step 20 loss = NaN → 立即 kill；诊断可能包括：
- npy 未匹配（不用可能 md5 已换）：`md5sum` 验证
- 数据加载 shift 错误
- OOM 假 NaN（activations 溢出）

---

## 7. 完成后交付（对齐 sibling §8 + TODOList §"实验 agent 完成后统一交付要求"）

- `outputs/olmo2_p24_sft_full32/final.pt`
- 完整 log
- 与 keep14fresh2 / ShortGPT-16 三臂 post-SFT eval（另启，非本 launch 范围）

**不写入论文任何 P2.4 结论**（sibling §7：三臂 + post-SFT eval 全就位前四路裁决不成立）。

---

## 8. 运行状态

见 `status/gpu_runs.jsonl` 与 `status/GPU_STATUS.md`。
