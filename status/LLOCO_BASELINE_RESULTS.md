# LLoCO Baseline — LongBench Reproduction Results

> 作成：2026-07-19 · 节点：.73 (28.85.35.73, 8×H20, diskB, EVAL-ONLY)
> 方法：LLoCO (Tan et al. 2024, arXiv:2404.07979) — 路径 (a)：自跑官方已发布权重
> 目的：为 Paper A / CoMem head-to-head 提供 LLoCO baseline + 验证 eval 口径

---

## 1. 结果（LongBench, n=200/task, 我们的 scorer 口径）

| Task | 我们跑出 F1 | 我们跑出 EM | LLoCO 论文 Table 4 (F1) | ΔF1 | empty_pred |
|------|:----------:|:----------:|:----------------------:|:---:|:----------:|
| narrativeqa | **24.21** | 9.00 | 23.1 | **+1.11** | 0 |
| qasper | **24.45** | 13.00 | 26.1 | **−1.65** | 0 |
| hotpotqa | **44.24** | 33.50 | 46.2 | **−1.96** | 0 |
| **AVG** | **30.97** | 18.50 | (NQA/QAS/HQA 均值 31.8) | −0.83 | — |

**三个任务全部落在 Table 4 的 ±2pt 以内**（|ΔF1| ≤ 1.96）。这验证了：
1. 我们的 LongBench F1/EM scorer 口径与 LLoCO 官方口径一致（同为标准 SQuAD/LongBench token-F1）。
2. LLoCO 压缩 + LoRA 推理管线接入正确（softprompt 注入、native prompt 格式、fp16 dtype）。
3. 该 3 任务的 LLoCO 数字可直接作为 CoMem head-to-head 的一行（与我们其他 LongBench baseline 完全同 eval set + 同 scorer）。

预测质量抽查（非空、语义正确）：
- narrativeqa: `He lives with the Mulvilles.` vs `He is a guest in the home of the Mulvilles.`
- hotpotqa: `Miller v. California` vs `Miller v. California`（EM 命中）；`2013` vs `2013`（EM 命中）
- qasper: 大量 `Unanswerable`（qasper 本身含 unanswerable 问题）

---

## 2. Backbone / 方法差异（论文中必须诚实标注）

| 属性 | LLoCO（本 baseline） | 我们的 CoMem |
|------|--------------------|------------|
| 基座 | LLaMA-2-7B（4k 原窗口）+ AutoCompressor 压缩器 | Qwen3-8B（32k 原窗口） |
| LoRA | **per-domain 监督微调**（in-domain 训练数据）| distill LoRA（通用跨域） |
| 推理时是否需要领域数据 | **需要**（每个任务必须有对应 LoRA）| 不需要（zero-shot） |
| 覆盖范围 | 仅 3/9 LongBench 任务有已发布 LoRA（nqa/qasper/hqa）| 全部 |

**建议表述**："LLoCO 使用 LLaMA-2-7B + AutoCompressor 作为 backbone（与 MemoryLLM 一样属不同基座对比），且每个任务需要 in-domain 监督微调的 LoRA；我们自跑官方权重在 LongBench narrativeqa/qasper/hotpotqa 上复现，F1 与原论文 Table 4 差异 ≤2pt。其余 6 个 LongBench 子任务（MFE/WMQA/MSQ/Gov/QMS/MNews）的 LoRA 官方未发布，如需可直接引用 Table 4 数字（MFE 26.3 / WMQA 35.6 / MSQ 27.3 / Gov 17.6 / QMS 23.4 / MNews 25.0）。RULER/BABILong/LoCoMo 无对应 LoRA，不做 LLoCO 对比。"

---

## 3. 方法细节（driver = scripts/eval_lloco_longbench.py）

- **压缩**：`LlocoAutoCompressorModel.from_pretrained(AutoCompressor-Llama-2-7b-6k)`，`segment_lengths=1536, output_softprompt=True`，truncation=False（full context，≤122880 tokens）——逐 sample 内联压缩（等价于官方 preproc_embs.py，省去 .pth 缓存）。
- **decoder query**：用 LLoCO **native** per-task prompt（`finetune_scrolls.py` 的 nqa/qasper sys_prompt + `finetune_hotpot.py` 的 icl_prompt）+ question + `\nAnswer:`。context 已压进 softprompt，故 decoder 只喂 instruction+question——正是 LoRA 训练时见过的格式。
- **推理**：`model.generate(input_ids=query, softprompt=compressed_ctx, max_new_tokens=...)`（PeftModel 加 domain LoRA），verbatim 复刻官方 inference.py。max_new_tokens 用我们的 DATASET2MAXGEN（nqa/qasper=128, hotpotqa=32，与我们其他 LongBench baseline 同）。
- **scorer**：将 `scripts/eval_longbench_mem_space.py` 的 `normalize_answer/compute_f1/compute_f1_multi/compute_em_multi/run_scoring` **原样拷入 driver**（保证 bit-identical 口径；不 import 该模块是因为它依赖 transformers 5.x 的 `src.memory.mem_space`，与隔离 env 的 transformers 4.37.2 冲突）。
- **eval set**：LongBench narrativeqa/qasper/hotpotqa 本地 JSONL（`data/longbench_raw/data/`，200/task）。注意 LLoCO 原论文 Table 4 是在**完整 tau/scrolls + hotpot_qa validation** 上，本复现用 LongBench 200-sample 子集（与我们其他 baseline 同 set），故与 Table 4 非同一 eval set——但结果仍 ≤2pt，说明口径 + 管线正确。

---

## 4. 环境（隔离 conda env，仅 .73 H20 可用）

- env：`/opt/conda/envs/lloco_env`（conda create -n lloco_env python=3.10）
- torch 2.1.2+cu121 · transformers 4.37.2 · peft 0.5.0 · datasets 2.18.0 · numpy 1.26.4
- **flash-attn 2.5.6**：用**预编译 wheel**（`flash_attn-2.5.6+cu122torch2.1cxx11abiFALSE-cp310`），H20 sm_90 实测可用（未从源码编译——系统 nvcc 是 CUDA 13.2，源码编译 2.5.6 有风险，预编译 wheel 规避且更快）。`modeling_flash_llama.py` 硬依赖 flash_attn（无 eager 回退），故 flash-attn 必装。
- **dtype = fp16**（关键坑）：bf16 会在 flash-attn 2.5.6 的 kv-cache decode kernel 上触发 **SIGFPE core dump**（H20/sm_90）；fp16 既是官方 inference.py 的 dtype（`--fp16 True`）也稳定。driver 默认已改 fp16。
- **wzc1 L20A（sm_100）不能跑**：flash-attn 2.5.6 只编到 sm_90，LLoCO eval 必须在 H20 节点。

---

## 5. 产物路径（.73 diskB，与 wzc1 不共享 FS）

节点 `28.85.35.73:36000`，`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`：
- 预测 raw：`lloco_results/longbench/{narrativeqa,qasper,hotpotqa}_{0..7}.jsonl`（每任务 8 shard，n=200）
- 汇总分数：`lloco_results/longbench/scores.json`
- 权重：`external/lloco_weights/{AutoCompressor-Llama-2-7b-6k, Lloco-7b-nqa, Lloco-7b-qasper, Lloco-7b-hqa}`
- 官方代码 clone：`external/lloco/`
- 日志：`logs/lloco_{task}_g{0..7}.log`, `logs/lloco_runner.log`, `logs/lloco_score.log`

driver（已提交，未 push）：`scripts/eval_lloco_longbench.py` + `scripts/run_lloco_longbench_8gpu.sh`

---

## 6. 遇到的坑（复盘）

1. **conda create / pip 需 proxy**：`repo.anaconda.com` 与 HF 都要 `http_proxy=http://hy-proxy.woa.com:3128`。
2. **flash-attn 源码编译风险**：系统 nvcc=CUDA13.2，2.5.6（2024 初）源码编译可能失败 → 改用预编译 cu122torch2.1 wheel，一次成功。
3. **bf16 → SIGFPE**：见 §4，改 fp16 解决（先用 -u 无缓冲复现才定位到是 generate 阶段崩，因 core dump 吞掉了 buffered stdout）。
4. **eval set 之辨**：LLoCO 官方 inference.py 跑的是 tau/scrolls + hotpot_qa validation（非 THUDM/LongBench）；本复现刻意用 LongBench 子集以对齐我们其他 baseline，因此与 Table 4 是"同任务不同 set"，±2pt 一致已足够验证口径。
