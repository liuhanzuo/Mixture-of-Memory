# LLoCO Baseline 复现可行性评估与执行方案

> 作成时间：2026-07-19  
> 目标：把 LLoCO (arXiv:2404.07979, Tan et al. 2024) 加入 Paper A / CoMem head-to-head

---

## 1. 官方 Release 状态（实测核实）

| 资产 | 位置 | 状态 |
|------|------|------|
| 代码 | https://github.com/jeffreysijuntan/lloco | ✅ 公开，含完整训练/推理代码 |
| 压缩器（AutoCompressor-Llama-2-7b-6k） | https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k | ✅ 公开，2 shard pytorch_model bins，~13GB |
| NarrativeQA LoRA | https://huggingface.co/xiuyul/Lloco-7b-nqa | ✅ `adapter_model.safetensors` + `adapter_config.json` |
| Qasper LoRA | https://huggingface.co/xiuyul/Lloco-7b-qasper | ✅ 同上 |
| HotpotQA LoRA | https://huggingface.co/xiuyul/Lloco-7b-hqa | ✅ 同上 |
| QMSum LoRA | https://huggingface.co/xiuyul/Lloco-7b-qmsum | ✅ 同上 |
| QuALITY LoRA | https://huggingface.co/xiuyul/Lloco-7b-quality | ✅ 同上 |
| MultiDoc LoRA（2wikimqa/musique） | 未发布 | ❌ 论文 Appendix D 提到用 combined finetuned model，但未上传 HF |
| MultiFieldQA-en LoRA | 未发布 | ❌ 需要 GPT-4 生成训练数据（论文描述），不可重现 |
| RULER/BABILong/LoCoMo LoRA | 不存在 | ❌ 这些 benchmark 超出 LLoCO 原论文范围 |

**HF 作者说明**：README 中 "Release pre-trained LoRA weights" 复选框显示未勾选，但 `xiuyul/`（Xiuyu Li，论文共同第一作者）已上传 5 个 LoRA adapter。adapter_config.json 确认：`base_model_name_or_path = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"`，LoRA r=8，target_modules=[q,k,v,o_proj]。

**已发布 LongBench 结果（论文 Table 4）**：

| NQA | QAS | MFE | HQA | WMQA | MSQ | Gov | QMS | MNews | Avg |
|-----|-----|-----|-----|-----|-----|-----|-----|-------|-----|
| 23.1 | 26.1 | 26.3 | 46.2 | 35.6 | 27.3 | 17.6 | 23.4 | 25.0 | 27.8 |

这是官方数字，使用官方 LongBench scorer（与我们的 `compute_f1_multi` / `run_scoring` 口径一致）。

---

## 2. Backbone 差异（必须在论文中诚实标注）

| 属性 | LLoCO | 我们的 CoMem |
|------|-------|------------|
| 基础模型 | LLaMA-2-7B-chat-hf（4k 原始窗口） | Qwen3-8B（32k 原始窗口） |
| 有效上下文 | 128k（经 AutoCompressor 压缩） | 128k+（QCMem mid-depth resume） |
| LoRA 方式 | per-domain 监督微调（in-domain 数据） | distill LoRA（通用，跨域） |
| 推理时是否需要领域数据 | **需要**（必须有对应 LoRA） | 不需要（zero-shot 可用） |

**论文标注建议**："LLoCO 使用 LLaMA-2-7B 作为 backbone（与 MemoryLLM 等一样属于不同基座对比）；其在 LongBench 的数字直接引用自原论文 Table 4（F1/EM 口径一致）。"

**本机模型可用情况**：
- `/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Llama--Llama2-7b`：有 base 模型（safetensors 格式），但 LLoCO 需要 **chat** 版本（`Llama-2-7b-chat-hf`）+ AutoCompressor 改造后的模型（`princeton-nlp/AutoCompressor-Llama-2-7b-6k`，不是普通 Llama-2-7B）。本地 base 模型不能直接用。

---

## 3. 三条路径成本估算

### 路径 (a)：直接用已发布权重 eval（eval-only）

**可 eval 的 LongBench 任务**：NarrativeQA、Qasper、HotpotQA（3/9 任务，对应已发布 LoRA）

**步骤**：
1. 在 .73（28.85.35.73，8×H20 eval-only）建独立 venv（Python 3.10，~2h）
2. 下载 `princeton-nlp/AutoCompressor-Llama-2-7b-6k`（~13GB，~30min）
3. 下载 3 个 LoRA adapter（各 ~100MB，~5min）
4. 克隆 `jeffreysijuntan/lloco` 代码（~1min）
5. 对 LongBench val 数据运行 `preproc_embs.py`（document → compressed embedding cache，3 task × ~200-500 samples，~2-3h 单卡）
6. 运行 `inference.py` 做推理（每 task ~1-2h 单卡）
7. 格式转换 + 接入我们的 LongBench scorer

**预计 wall-clock**：env 2h + 下载 1h + preproc 3h + 推理 4h + 评分 1h = **约 11h 总**（可单卡串行，不占多卡）

**关键风险**：
- **flash-attn 2.5.6 × H20（sm_90）**：H20 = Hopper = sm_90，flash-attn v2.5.6 的 `setup.py` 明确包含 `compute_90,code=sm_90`（已从 raw.githubusercontent.com 确认），**兼容**。但需从源码编译（约 20-30min，需 CUDA 12.x + gcc）。
- **Python 版本**：LLoCO 依赖（transformers==4.37.2 / peft==0.5.0）在 Python 3.14 上可能有 API 兼容问题。需要 `conda create -n lloco python=3.10`，在 .73 节点上是否有 conda 可用：需提前确认。
- **vllm==0.3.3**：只在 baseline eval mode（纯 LLaMA 无压缩）中使用，LLoCO 自身的 autocomp eval mode 不需要。**可跳过 vllm**。
- **Coverage 不完整**：2wikimqa / musique / multifieldqa_en 没有 LoRA，无法用路径 (a) 运行。

**可上 .73？** 是，全程 eval-only，不占训练 GPU。可以等 .73 有空闲缝隙排入。

---

### 路径 (a.5)：直接引用论文 Table 4 数字（零 GPU）⭐ 推荐

**可用数字**：LLoCO 论文 Table 4 = 全部 9 个 LongBench subtask，使用官方 LongBench F1/EM 评分（与我们的 scorer 口径一致）。

**操作**：
- 论文 related work 或 head-to-head 表中直接引用：LLoCO Table 4 的 NQA/QAS/MFE/HQA/WMQA/MSQ/Gov/QMS/MNews 数字
- 标注 backbone 差异（LLaMA-2-7B vs Qwen3-8B）
- 标注方法差异（supervised in-domain LoRA vs distill LoRA）

**工作量**：几乎零，只需整理 LaTeX 表格。

**局限性**：不能做 RULER/BABILong/LoCoMo 对比（LLoCO 原论文无这些数字）。

---

### 路径 (b)：用已发布 compressor + 自训缺失 LoRA（需训练 slot）

**需要训练的**：
- 2wikimqa/musique：需 multi-hop QA 混合训练集（HotpotQA + Wikipedia paragraphs）+ finetune ~3-5h on 4×H20
- multifieldqa_en：需 GPT-4 生成 QA pairs（论文原始方法，昂贵且不完全可重现）；替代方案：用通用 instruction QA 数据代替
- LoCoMo LoRA：需构建对话类训练数据（LoCoMo 本身无官方 train split）

**约束**：当前无空闲训练 slot（wzc1 + .82/.104 满；.73 = EVAL-ONLY）。需等 Paper B 训练完成才能腾出 slot。

**预计训练 wall-clock**：每个 domain LoRA ~3-5h × 4 GPU（4× 8=32 GPU-hours）。

---

### 路径 (c)：完全从头复现

最重。需要：AutoCompressor 自训（论文说用 princeton-nlp 的预训练权重）+ 所有 domain LoRA 从零。不推荐。

---

## 4. 依赖/环境分析

### LLoCO 要求的环境（`requirements.txt` 实测）：

```
torch>=2.1.2
transformers==4.37.2
peft==0.5.0
flash-attn==2.5.6
datasets==2.18.0
deepspeed==0.12.3  # 训练用，推理可跳过
vllm==0.3.3        # baseline mode 用，LLoCO 推理可跳过
fire==0.5.0
```

### 与我们现有 env 的兼容性：

| env | Python | torch | transformers | 兼容 LLoCO？ |
|-----|--------|-------|--------------|--------------|
| wzc1 `.venv` | 3.14 | 2.13 | 5.13 | ❌ transformers 差 1 大版本；flash-attn 2.5.6 不支持 L20A sm_100 |
| diskB `torch-base`（.73/.82/.104） | 3.14 | 2.13 | 5.13 | ❌ transformers 差 1 大版本；但 H20 sm_90 可编译 flash-attn |

### 结论：必须建独立隔离 venv

参考 `external/landmark_venv`、`external/InfLLM` 的做法，在 .73 上新建：
```bash
conda create -n lloco_env python=3.10
conda activate lloco_env
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.2 peft==0.5.0 datasets fire accelerate
# flash-attn 需从源码编译（H20 sm_90 OK）：
MAX_JOBS=8 pip install flash-attn==2.5.6 --no-build-isolation
```

**wzc1 L20A（sm_100）不能跑 flash-attn 2.5.6**（2.5.6 只编到 sm_90）。LLoCO eval 必须在 .73 H20 节点上。

---

## 5. Driver 草案

### 两阶段流程：

#### Stage 1：preproc_embs（一次性，每任务约 30-60min 单卡）

```bash
cd external/lloco
python preproc_embs.py \
    --emb_model_name autocomp \
    --dataset narrativeqa \   # 或 qasper / hotpot_qa
    --split validation \
    --out_path embeddings/narrativeqa_val_embs.pth \
    --truncation False
```
数据源：从我们的 `data/longbench_raw/` 读取（需在 `data.py` 中添加对 JSONL 格式的支持，或直接用 HF datasets 在线加载）。

#### Stage 2：inference wrapper `scripts/eval_lloco_longbench.py`（新建 driver）

```python
"""LLoCO baseline — LongBench eval driver.

复用框架（verbatim, unmodified）：
  - lb.load_longbench_dataset / format_prompt / DATASET2MAXGEN / run_scoring  
    （来自 scripts/eval_longbench_mem_space.py）

只替换的部分：
  - 模型 forward：LlocoAutoCompressorModel（AutoCompressor-Llama-2-7b-6k）
                  + PeftModel（domain-specific LoRA adapter）
                  + precomputed embedding cache（.pth → softprompt 注入）
  - 输入构造：entry["decoder_input_ids"] + entry["context_embeddings"]（来自 LazyScrollsSFTDataset）
  - 输出格式：{index, pred, answers, dataset} JSONL（与其他 drivers 统一，喂给 run_scoring）
"""

# 关键接口点（骨架）：

# 1. 加载 AutoCompressor + LoRA
from external.lloco.auto_compressor import LlocoAutoCompressorModel
from peft import PeftModel

def load_lloco(base_model="princeton-nlp/AutoCompressor-Llama-2-7b-6k",
               peft_path="external/lloco_weights/Lloco-7b-nqa",
               dtype=torch.bfloat16):
    model = LlocoAutoCompressorModel.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, peft_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = "[PAD]"
    return model.cuda().eval(), tokenizer

# 2. 加载 precomputed embeddings（.pth cache）
emb_cache = torch.load("embeddings/narrativeqa_val_embs.pth")  # {pid: tensor}

# 3. 推理：把 softprompt 注入为 context
output_ids = model.generate(
    input_ids=query_ids.unsqueeze(0),
    softprompt=emb_cache[pid].unsqueeze(0).to(dtype),
    max_new_tokens=max_gen
)

# 4. 格式对齐：输出 {index, pred, answers, dataset} JSONL（与 lb.run_scoring 兼容）
```

**关键改动点（相对 LLoCO 原版 inference.py）**：
1. 替换数据集加载：用我们 `data/longbench_raw/{dataset}.jsonl` 而非在线 HF
2. 输出格式改为 JSONL（与我们的 shard-merge scorer 对接）
3. 支持 `--num_shards / --shard_index` 多卡并行（.73 可 8 卡并行 8 shard）
4. DATASET_TO_LORA_MAP = {narrativeqa: "nqa", qasper: "qasper", hotpotqa: "hqa"}（自动选 adapter）

**需要修改的 LLoCO 源码**（最少改动）：
- `finetune_scrolls.py` 中 `LazyScrollsSFTDataset.__init__`：添加从本地 JSONL 加载的路径（HF 可能超时）
- `preproc_embs.py`：添加 hotpot_qa 的本地路径支持

---

## 6. 综合评估与推荐

### 在哪些 benchmark 上做 head-to-head？

| Benchmark | LLoCO 可行性 | 推荐方式 |
|-----------|------------|---------|
| **LongBench NQA/QAS/HQA** | ✅ LoRA 已发布 | Path (a)：自跑；或直接引用 Table 4 |
| **LongBench WMQA/MSQ** | ⚠️ LoRA 未发布 | Path (a.5)：引用 Table 4（WMQA=35.6/MSQ=27.3）|
| **LongBench MFE** | ⚠️ LoRA 未发布（需 GPT-4 data） | Path (a.5)：引用 Table 4（MFE=26.3）|
| **RULER NIAH** | ❌ 无对应 LoRA | 排除或注脚说明 |
| **BABILong** | ❌ 无对应 LoRA | 排除（LLoCO 没有 BABILong LoRA） |
| **LoCoMo** | ❌ 无对应 LoRA + 无训练数据 | 排除（方法不兼容）|

### 最终推荐：路径 (a.5) + 路径 (a) 后验证

**第一阶段（立即可执行，零 GPU）**：
- 在论文 head-to-head 表中直接引用 LLoCO Table 4 LongBench 数字（NQA/QAS/MFE/HQA/WMQA/MSQ/Gov/QMS/MNews）
- 标注：① backbone 差异（LLaMA-2-7B-chat vs Qwen3-8B）；② 方法差异（supervised in-domain LoRA per domain vs distill LoRA）；③ RULER/BABILong/LoCoMo 不比较（LLoCO 需 per-domain LoRA，无法公平适配）
- **零 GPU 成本，可立即插入论文表格**

**第二阶段（可选，验证 eval 口径对齐，约 1 天）**：
- 等 .73 有空闲缝隙（≥8h）时，在 .73 H20 节点建 `conda env lloco_env python=3.10`，运行路径 (a) 的 3 个任务（NarrativeQA/Qasper/HotpotQA）
- 目的：验证我们的 prompt 格式 + scorer 与论文数字一致；若一致则全套引用论文数字，若差异 >2pt 则单独注明使用了不同 prompt 格式
- **不占训练 GPU，仅 eval-only，可排入 .73**

**路径 (b) 建议**：暂不执行，原因：① 当前无训练 slot；② 2wikimqa/musique LoRA 训练需 32+ GPU-hours；③ MFE 需 GPT-4 生成数据（非 open-source 可复现），论文中很难公平声称完整复现了 LLoCO；④ 即使训练出来，backbone 差异（LLaMA-2-7B vs Qwen3-8B）仍会主导性能差异，额外 LoRA 训练的边际价值低。

---

## 7. 最终一句话结论

**推荐走路径 (a.5)（引用论文 Table 4 LongBench 数字）+ 可选路径 (a) 在 .73 后验证 3 任务；RULER/BABILong/LoCoMo 排除 LLoCO。eval-only 路径可立即上 .73，无需等训练 slot。**
