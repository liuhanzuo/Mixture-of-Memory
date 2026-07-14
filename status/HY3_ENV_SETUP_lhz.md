# Hy3 / Hunyuan QCMem 环境搭建记录（lhz H200 集群，2026-07-13）

> 本文件记录在**新的 lhz / lhz2 H200 集群**上为 QCMem-on-Hunyuan-MoE 搭建的运行环境。
> 与旧集群（CODEBUDDY.md 里的 apdcephfs 节点）**无关**——这是一套独立的新机器。

## 机器拓扑（实测）
| 机器 | 角色 | GPU | Python/torch | 盘 | 外网 |
|---|---|---|---|---|---|
| 开发机（本会话所在） | 写代码/下载 | 无 GPU | /opt/conda py3.11, torch2.7.1 | /volume/haru (gpfs) | ✅ 有 |
| **lhz** (`lhz-0`) | 训练/eval | 8× H200 143GB | /usr/bin py3.12, **torch 2.8.0a nv25.05** | ⭐**共享 /volume/haru** | ❌ 无 |
| **lhz2** (`lhz2-0`) | 训练/eval | 8× H200 143GB | 同 lhz | ❌ 独立本地盘 /（不挂 /volume/haru） | ❌ 无 |

- **开发机 ↔ lhz 共享 `/volume/haru`**：改代码/建 venv 在开发机做，lhz 直接可见，**零同步**。
- **lhz2 不挂共享盘**：要用得把仓库 + venv rsync 过去（本地盘 /root，44%→有空间）。
- lhz/lhz2 **均无外网**（pypi/HF/腾讯镜像全不通，无代理）；只有开发机能上网 → 离线 wheelhouse 方案。

## 环境：`.venv_hy3`（建在共享盘根 = `/volume/haru/Mixture-of-Memory/.venv_hy3`）
- 用 **lhz 的 python3.12** 建（`python3 -m venv --system-site-packages`），继承系统 **nv 优化版 torch 2.8**（不重装 torch，保留 nv25.05 优化 + sm_90 H200 兼容）。
- venv 内 pip 不受 PEP668 限制，可正常装包。
- 装了：**transformers 5.13.1**（认得 `hy_v3`=HYV3* **和** `hunyuan_v1_moe`=HunYuanMoEV1* 两套 Hunyuan MoE 类）、peft 0.19.1、accelerate 1.14、datasets 4.0。
- ⚠️ **numpy 必须 <2**：wheelhouse 默认拖来 numpy 2.2.6 会让 nv-torch 崩（"compiled using NumPy 1.x cannot run in 2.x"）→ 已强制降 **numpy==1.26.4**（匹配系统 torch 编译版本）。

### 复现命令（开发机执行，lhz 因共享盘直接生效）
```bash
# 1. 开发机(有网)下 cp312 wheels 到共享盘,排除 torch/nvidia(用 lhz 系统的)
/opt/conda/bin/python -m pip download --dest wheelhouse_hy3 --only-binary=:all: \
  --python-version 3.12 --implementation cp --abi cp312 --platform manylinux2014_x86_64 \
  "transformers==5.13.1" "peft>=0.7.0" "accelerate>=0.25.0" "datasets>=2.16.0" "numpy==1.26.4"
mkdir -p wheelhouse_hy3/_excluded_torch
mv wheelhouse_hy3/torch-*.whl wheelhouse_hy3/nvidia_*.whl wheelhouse_hy3/triton-*.whl wheelhouse_hy3/_excluded_torch/

# 2. lhz 上建 venv + 离线装(共享盘,开发机看不到 GPU 所以在 lhz 跑)
ssh lhz 'cd /volume/haru/Mixture-of-Memory && \
  python3 -m venv --system-site-packages .venv_hy3 && \
  .venv_hy3/bin/pip install --no-index --find-links wheelhouse_hy3 --no-deps wheelhouse_hy3/*.whl && \
  .venv_hy3/bin/pip install --no-index --find-links wheelhouse_hy3 --no-deps --force-reinstall "numpy==1.26.4"'
```

### 验证（全 PASS）
```bash
ssh lhz 'cd /volume/haru/Mixture-of-Memory && \
  PYTHONPATH=$PWD:$PWD/third_party/babilong-pkg CUDA_VISIBLE_DEVICES="" \
  .venv_hy3/bin/python scripts/qcmem_hy3_selftest.py --tiny --device cpu --dtype float32 --attn_impl eager --tol 1e-4'
# => HY3 SELF-TEST: ALL PASS — QCMem depth-partition is exact on Hy3 MoE (max|diff|=0.0)
```
- torch 2.8 cuda True, 8 dev；transformers 5.13.1；QCMemModel / QCMemHy3Model import OK。

## ⚠️ 关键：公开 Hunyuan-A13B ≠ 项目 port 的 Hy3（用模型前必看）
| | 公开 **Hunyuan-A13B** (`tencent/Hunyuan-A13B-Instruct`) | 项目 port 的 **Hy3** |
|---|---|---|
| model_type / 类 | `hunyuan_v1_moe` / `HunYuanMoEV1ForCausalLM` | `hy_v3` / `HYV3ForCausalLM` |
| 层数 | **32** | **80** |
| experts | 64 | 192 |
- 现有 `src/memory/qcmem/qcmem_hy3.py` + self-test + trainer **硬编码 `HYV3*` 类、80 层假设**。
- **直接用公开 Hunyuan-A13B 需要适配**：改类名 `HYV3*`→`HunYuanMoEV1*`、层数 80→32、重扫 split-j（旧甜点 j=32/80=0.4L → A13B 应 ~0.4×32≈13）、`first_k_dense_replace` 等 config 细节。QCMem WRITE/READ 核心在父类 `QCMemModel`，不用动。
- self-test 用 tiny 随机 `HYV3Config`，**不需要任何模型权重**（已验证）。

## 待办（真要跑模型时）
1. 下载公开 Hunyuan-A13B 权重（开发机有网；放共享盘 `models/`，lhz 直接可见）。
2. 适配 port 代码 `HYV3*`→`HunYuanMoEV1*`（派 coder；A13B 32 层）。
3. 准备 `data/pg19_train.jsonl`（PG19Packer 按**每行纯文本** encode，不解析 JSON；自蒸馏/j-sweep 需要）。
4. lhz2 要用：rsync 仓库 + `.venv_hy3` 到 lhz2:/root/（不共享盘）。

---

## A13B minimal-arch probe 诊断结论（2026-07-14）
- **kernel 坑**：transformers 5.13.1 的 `grouped_mm_experts_forward`（torch 2.8-nv）对未 pad 的 per-expert token 数触发 `GroupMMCommon.cuh:51 delta%16==0` 断言崩溃。**解法：`from_pretrained(..., experts_implementation="eager")`** 走 per-expert index_add_ loop，数值等价、无对齐约束。probe 脚本已加。
- **create_causal_mask 签名**：5.13.1 用 `inputs_embeds=`（无 `cache_position` 参数），4.57 用 `input_embeds=`+`cache_position=`。probe 脚本已 try/except 兼容。
- **★ training-free 截断 probe 对 Hunyuan-A13B 无效（关键发现）**：原生 output_hidden_states logit-lens 实测：
  - layer 8/16/24：nll=84/90/83（爆炸），hidden **absmax=137~165 但 rms 仅 0.10~0.17** = 极端 massive activation
  - layer 31：nll=3.4（正常），absmax 骤降到 16.5
  - 根因：Hunyuan 中间层残差流有巨型 outlier 维度，final RMSNorm 的 per-dim gain 是为顶层学的，套到中间层 hidden 上把 outlier 放大 → logits ±165 → nll 爆炸。
  - **含义**：naive 硬截断 logit-lens 在 Hunyuan 不 informative（off-manifold），给不出有意义的语义饱和 j。确定 j 应改用：(a) 下游任务 probe（probe B，需 GLUE 数据+网络）；或(b) 直接按 Qwen/Hy3 的 0.33-0.4L 经验值设 j≈11-13，进入真实训练验证（train_hunyuan_a13b_probe2.py，keep+fresh 真训才是用户交付路径）。
