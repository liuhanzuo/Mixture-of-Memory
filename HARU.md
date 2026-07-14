# HARU.md — Hunyuan-A13B 极简架构 / QCMem 工作日志

> 个人工作日志（lhz / lhz2 H200 集群）。记录在公开 **Hunyuan-A13B**（32 层 MoE）上推进"理解-生成分工 / 极简架构"方向干了什么。
> 权威运维细节见 `status/HY3_ENV_SETUP_lhz.md`；本文件是"我做了啥 + 结论"。
> 最后更新：2026-07-14。

---

## ★★ 核心结论：A13B 的 split-j ≈ 13/32（0.4·L）

**用对方法（QCMem depth-partition j-sweep）在真实 Hunyuan-A13B 上测出的语义截断层。** 和 Qwen3-8B（12/36≈0.33L）、Hy3-80L（32/80=0.4L）一致，0.4·L 规律跨 backbone 成立。

### 结果表（QCMem j-sweep，`scripts/qcmem_a13b_jsweep.py`，8 docs）
`ppl_gap = QCMem读出ppl / 全上下文ppl`（越接近 1 越好），`top1 = 与全模型 argmax 一致率`：

| j | frac(j/32) | 3k gap | 8k gap | 16k gap | 16k top1 |
|---|---|---|---|---|---|
| 0 | 0.00 | **1.000** | **1.000** | **1.000** | 1.000 |  ← 正确性 gate（==full forward，diff 0）|
| 4 | 0.12 | 1.060 | 1.066 | 1.131 | 0.834 |
| 8 | 0.25 | 1.181 | 1.190 | 1.353 | 0.777 |
| 12 | 0.38 | 1.276 | 1.300 | 1.472 | 0.758 |
| **13** | **0.41** | 1.285 | 1.333 | 1.544 | 0.742 |  ← **选定 split-j** |
| 14 | 0.44 | 1.309 | 1.365 | **1.601 峰** | 0.739 |
| 16 | 0.50 | 1.323 | 1.385 | 1.597 ↓ | 0.745 |
| 20 | 0.62 | 1.364 | 1.389 | 1.591 谷 | 0.748 ↑ |
| 24 | 0.75 | 1.390 | 1.406 | 1.599 | 0.747 |

- **16k 出现 "fidelity smile"**（和 Hy3 在 8k/16k 一致）：gap 在 j=14 见顶(1.601)后 j=16~20 回落、top1 回升 → mid-depth "可缓存语义天花板"。短 ctx 看不出，必须长 ctx。
- 绝对 gap 偏高(1.3~1.6@长档)是 **zero-shot + 全 chunk 选中**的正常现象（Hy3 同样），真实 self-distill / keep+fresh 训练后会往 1.0 收。
- 结果 JSON：`logs/a13b_jsweep_results.json`(短)、`logs/a13b_jsweep_longctx.json`(长)。详见 `versions/v_qcmem_a13b_port.md`。

### ⚠️ 方法教训（重要，别重走）
确定 j **必须用 QCMem depth-partition**（h_j → 喂 layers[j:] 正常重算 → 读出），**不能用 logit-lens / 硬截断**（h_j 直接过顶层 norm→lm_head）。
- 原因：Hunyuan 中间层残差流有**巨型 massive activation**（layer 8~24 的 hidden absmax 达 ±137~165，rms 仅 0.1；直到 layer 31 才降到 16.5）。final RMSNorm 的 per-dim gain 是为顶层学的，套到中间层 hidden 上放大 outlier → logits ±165 → nll 爆炸(84~235)。
- 一开始误用 `probe_minimal_arch_hunyuan.py`（logit-lens）→ 所有 j 全崩 nll 84，被误判"中间层不可省"。换成 QCMem j-sweep 后曲线立刻平滑合理。**logit-lens 那条对 Hunyuan 无效，作废。**

---

## 流程记录（2026-07-13 ~ 07-14）

### 1. 环境（详见 status/HY3_ENV_SETUP_lhz.md）
- **拓扑**：开发机(CPU, 有外网) ↔ **lhz**(8×H200, 共享 `/volume/haru`, 无外网) / **lhz2**(8×H200, 独立盘, 无外网)。
- **`.venv_hy3`**（共享盘）：lhz python3.12 + `--system-site-packages` 继承系统 nv-torch 2.8 + 离线装 transformers **5.13.1** / peft / accelerate / datasets / tiktoken；numpy 强制 **1.26.4**（否则 nv-torch 崩）。
- 离线安装：开发机下 cp312 wheels → 共享盘 `wheelhouse_hy3/`（gitignored）→ lhz `pip --no-index` 装。

### 2. 模型 & 数据
- **Hunyuan-A13B-Pretrain** 权重下到共享盘 `models/Hunyuan-A13B-Pretrain`（150G, 32层, 64 experts, `model_type=hunyuan`）。gitignored。
- **SlimPajama**：validation + 6 个 train 分片下到 `data/slimpajama-6b/`。已生成 hunyuan-tokenized `data/slimpajama_val_2048_hunyuan.npy`（j-sweep 用）。train npy 生成中。

### 3. 代码（本次提交的）
- `scripts/qcmem_a13b_jsweep.py` — A13B(32层) QCMem j-sweep（改自 Hy3 版，用 pre-tokenized npy 省 tokenizer round-trip）。
- `src/memory/qcmem/qcmem_hy3.py` — 新增 `load_a13b_qcmem()`（`QCMemHy3Model` 类架构无关、复用；Hy3 的 `load_hy3_qcmem` 不动）。
- `scripts/probe_minimal_arch_hunyuan.py` / `probe_truncated_downstream_hunyuan.py` — minimal-arch probe 的 A13B port（**已证对 Hunyuan 因 massive activation 失效，保留供参考/别的 backbone 用**）。含两处兼容修复见下。
- `versions/v_qcmem_a13b_port.md` — A13B port 版本文档 + 完整结果。

### 4. A13B on transformers 5.13.1 + torch 2.8-nv 的三个坑（已解，复用于所有 A13B 脚本）
1. **`head_dim=None`**：`HunYuanMoEV1Config.from_pretrained` 后须手动 `cfg.head_dim=128`（否则 attention `head_dim**-0.5` 崩）。
2. **MoE grouped_mm kernel 崩**：默认 experts 走 torch2.8-nv grouped-GEMM，触发 `GroupMMCommon.cuh:51 delta%16==0` 断言 → **必须 `from_pretrained(..., experts_implementation="eager")`**（per-expert index_add_ loop，数值等价）。
3. **`create_causal_mask` 签名**：5.13.1 用 `inputs_embeds=`、无 `cache_position`；4.57 用 `input_embeds=`+`cache_position=`。脚本已 try/except 双兼容。

---

## 下一步
1. **起 keep-13 + fresh-{2,4} 训练**（`train_hunyuan_a13b_probe2.py`，SlimPajama，先 20k steps 看 s/step + 收敛）。
   - ⚠️ backbone **不冻结**（`--freeze_front` 保持默认 off）：前 13 层低 LR(2e-5) 微调、新 NTP 层高 LR(1e-4)。
   - ⚠️ 训练器加载模型处也要加 `experts_implementation="eager"`（同坑2），当前脚本尚未加。
   - 启动脚本 `launch_hunyuan_a13b_keep24_fresh2.sh` 是旧集群路径/max_steps=200，需改 lhz 路径 + keep13 + max_steps 20000。
2. 训练后重测 j-sweep，看 gap 是否往 1.0 收（对标 Hy3 self-distill 把 1.25-1.5× 压向 1.0）。
