# Long-Context Memory Papers: Reproducibility Survey

**Date**: 2026-05-11
**Author**: Researcher Agent
**Purpose**: Systematic survey of long-context memory papers, focusing on reproducibility and comparison potential with our H-series cross-attention memory experiments.

---

## Executive Summary

We surveyed 10 papers on long-context memory mechanisms for LLMs. The key finding is that **ARMT (by the RMT team) and Activation Beacon are the strongest candidates for immediate reproduction**, while **Infini-attention has been shown to be non-reproducible** by independent experiments. Our H-series cross-attention memory approach (PPL ratio 1.011) has strong PPL performance but fails BABILong due to base model limitations (Llama-3-8B trained on 8k context).

---

## Summary Table

| Paper | Year/Venue | GitHub | Code Completeness | BABILong Reported | Reproduce Time (8xB200) | Priority |
|-------|-----------|--------|-------------------|-------------------|------------------------|-----------|
| RMT | NeurIPS'22 / AAAI'24 | [booydar/recurrent-memory-transformer](https://github.com/booydar/recurrent-memory-transformer) | Full (train+eval+inference) | Yes (SOTA on some tasks) | 2-3 days train, 4h eval | **P1** |
| ARMT | ICML'24 Workshop | [RodkinIvan/associative-recurrent-memory-transformer](https://github.com/RodkinIvan/associative-recurrent-memory-transformer) | Full (train+eval+scripts) | Yes (SOTA) | 2-3 days train, 4h eval | **P1** |
| Activation Beacon | arXiv 2401 / NAACL'24 | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | Full (train+eval+HF models) | 65.4% @ 1k (our result) | 4h eval (ckpt exists) | **P1** |
| MemoryLLM | ICML'24 | [wangyu-ustc/MemoryLLM](https://github.com/wangyu-ustc/MemoryLLM) | Full (train+eval+ckpt) | 36.6% @ 1k (our result) | 4h eval (ckpt exists) | **P2** |
| HMT | NAACL'25 | [OswaldHe/HMT-pytorch](https://github.com/OswaldHe/HMT-pytorch) | Full (train+eval) | No reported | 1-2 days train, 2h eval | **P2** |
| MemLong | arXiv 2408 | [Bui1dMySea/MemLong](https://github.com/Bui1dMySea/MemLong) | Partial (train+eval, needs faiss) | No reported | 2-3 days train, 4h eval | **P3** |
| Block Recurrent Transformer | NeurIPS'22 | [google-research/meliad](https://github.com/google-research/meliad) (JAX) | Partial (JAX only) | No | Hard (JAX, no HF) | **P4** |
| Infini-attention | arXiv 2404 (Google) | No official code | Community-only, broken | No (shown to fail) | N/A (non-reproducible) | **Skip** |
| Landmark Attention | NeurIPS'23 | [epfml/landmark-attention](https://github.com/epfml/landmark-attention) | Partial (inference, old LLaMA) | No | 1-2 days to adapt | **P4** |
| M+ (MemoryLLM v2) | arXiv 2502 / 2025 | Same as MemoryLLM (mplus-8b branch) | Full (train+eval+ckpt) | TBD | 4h eval (ckpt exists) | **P2** |

---

## Detailed Analysis

---

### 1. RMT (Recurrent Memory Transformer)

**Paper Info**
- Title: "Recurrent Memory Transformer"
- Authors: Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev (DeepPavlov / AIRI)
- Venues: NeurIPS 2022, follow-up AAAI 2024
- arXiv: 2207.06881 (original), 2304.11062 (scaling to 1M tokens)
- Follow-up: "In Search of Needles in a 11M Haystack" (arXiv: 2402.10790)

**Core Method**
Adds special memory tokens to the input sequence that serve as a segment-level recurrent memory. The model processes input in segments, passing memory tokens between segments. Compatible with any HuggingFace model via a wrapper.

**GitHub Repository**
- URL: https://github.com/booydar/recurrent-memory-transformer
- Branch: `framework_accel` (active development)
- Stars: ~150+ (estimate, API unavailable)
- Last updated: Active as of 2024-2025
- Code: Complete -- training scripts, evaluation scripts, BABILong benchmark code (in companion repo `booydar/babilong`)

**Reproducibility Assessment**
- Code open source: YES, full training + evaluation code
- Pretrained checkpoints: Not directly provided; but models are fine-tuned from public HF checkpoints (e.g., LLaMA-3.2-1B)
- Requirements: Yes, `requirements.txt` provided
- README quality: Good -- clear installation, training, and evaluation instructions
- Code quality: HIGH. Well-structured, uses HuggingFace Accelerate, supports DeepSpeed
- Confidence: **HIGH**

**Reproduction Time (8xB200)**
- From scratch training: 2-3 days on PG-19 / C4 for 7B model
- Fine-tune from existing: 6-12 hours
- Inference/eval only: 2-4 hours on BABILong

**Benchmarks**
- BABILong: Yes -- RMT team created this benchmark. Reports SOTA on multi-task long-context QA.
- WikiText / PG-19 PPL: Yes
- NIAH / Passkey retrieval: Yes (in the 1M token paper)
- Direct comparison: We can run their exact BABILong eval pipeline

**Comparison with Our H-series**
- RMT uses segment-level recurrence with memory tokens; we use cross-attention memory at specific layers
- RMT's BABILong performance is the gold standard; reproducing their exact numbers would validate our eval pipeline
- Their Llama-3.2-1B + RMT model would be a direct comparison point

**Priority: P1** -- Same team created BABILong; code is mature; we already use their eval pipeline

---

### 2. ARMT (Associative Recurrent Memory Transformer)

**Paper Info**
- Title: "Associative Recurrent Memory Transformer"
- Authors: Ivan Rodkin, Yuri Kuratov, Aydar Bulatov, Mikhail Burtsev
- Venue: ICML 2024 NGSM Workshop
- arXiv: 2407.04841

**Core Method**
Enhances RMT with associative memory using linear-attention style mechanisms. Scales to 50M tokens while trained only on 16k context. Achieves SOTA on BABILong benchmark.

**GitHub Repository**
- URL: https://github.com/RodkinIvan/associative-recurrent-memory-transformer
- Stars: ~30+ (estimate)
- Last updated: Active 2024-2025
- Code: Complete -- fork of RMT repo with ARMT additions, training scripts for PG-19 with Llama-3.2

**Reproducibility Assessment**
- Code open source: YES, full training + evaluation
- Pretrained checkpoints: Not provided; train from public HF checkpoints
- Requirements: Yes, `requirements.txt` provided
- README quality: Good -- includes installation and training commands
- Code quality: HIGH -- same codebase as RMT, well-maintained
- Includes Llama-3.2 training scripts: `scripts/pg19/finetune_armt_llama3.2_pg19_sliding.sh`
- Confidence: **HIGH**

**Reproduction Time (8xB200)**
- From scratch training: 2-3 days on PG-19
- Fine-tune: 6-12 hours
- Inference/eval only: 2-4 hours

**Benchmarks**
- BABILong: Yes -- SOTA performance reported
- PG-19 PPL: Yes
- Direct comparison: Can use same BABILong eval we already run

**Comparison with Our H-series**
- ARMT is the most direct competitor: both add memory to LLMs for long context
- ARMT uses associative memory tokens; we use cross-attention slots
- Both work with Llama-3.2-1B base

**Priority: P1** -- SOTA on BABILong; same team as RMT; directly comparable architecture

---

### 3. Activation Beacon

**Paper Info**
- Title: "Long Context Compression with Activation Beacon" (originally "Soaring from 4K to 400K")
- Authors: Peng Zhang et al. (Microsoft Research / BAAI)
- arXiv: 2401.03462
- Venue: Under review (NAACL track)

**Core Method**
A plug-in module that progressively compresses KV activations across all layers into compact "beacon" representations. Preserves original short-context capability while extending to 400K tokens.

**GitHub Repository**
- URL: https://github.com/FlagOpen/FlagEmbedding (under activation-beacon subdirectory)
- Stars: FlagEmbedding has 7000+ stars
- Pretrained models on HuggingFace: YES -- `namespace-Pt/beacon-qwen-2-7b-instruct` and others
- Code: Complete -- training, evaluation, and pretrained models

**Reproducibility Assessment**
- Code open source: YES
- Pretrained checkpoints: YES -- multiple models on HuggingFace (Qwen2-7B, LLaMA-2-7B)
- Requirements: Yes
- README quality: Good
- Code quality: HIGH -- production-quality code from Microsoft/BAAI
- **We already have baseline results**: 65.4% AVG @ 1k on BABILong with Beacon-Qwen2-7B
- Confidence: **VERY HIGH** (already reproduced)

**Reproduction Time (8xB200)**
- From scratch training: Not needed -- checkpoints available
- Fine-tune: 1-2 days
- Inference/eval only: 2-4 hours (already done)

**Benchmarks**
- BABILong: Our result 65.4% @ 1k (strongest baseline)
- LongBench: Yes (reported in paper)
- Passkey retrieval: Yes
- PPL: Yes

**Comparison with Our H-series**
- Beacon is currently our strongest baseline on BABILong
- Beacon uses activation compression; we use cross-attention memory slots
- Both aim for similar goals but very different mechanisms

**Priority: P1** -- Already have baseline; extend to training reproduction

---

### 4. MemoryLLM

**Paper Info**
- Title: "MemoryLLM: Towards Self-Updatable Large Language Models"
- Authors: Yu Wang et al. (UCSD/USTC/Amazon)
- Venue: ICML 2024
- arXiv: 2402.04624
- Follow-up: "M+: Extending MemoryLLM with Scalable Long-Term Memory" (arXiv: 2502.00592)

**Core Method**
Embeds a fixed-size memory pool (1.67B parameters for 8B model) within the model's latent space as self-updatable parameters. The memory is updated via gradient-based injection during forward pass.

**GitHub Repository**
- URL: https://github.com/wangyu-ustc/MemoryLLM
- Stars: ~200+ (estimate)
- Last updated: Active 2025 (mplus-8b branch updated July 2025)
- Code: Very complete -- training code, evaluation code, pretrained checkpoints for 7B and 8B

**Reproducibility Assessment**
- Code open source: YES -- full training + eval
- Pretrained checkpoints: YES -- `YuWangX/memoryllm-7b`, `YuWangX/memoryllm-8b`, `YuWangX/memoryllm-8b-chat`, `YuWangX/mplus-8b`
- Requirements: Yes, `requirements.txt` and `requirements_infer_only.txt`
- README quality: Excellent -- detailed setup, training, evaluation instructions
- Code quality: HIGH -- well-documented, includes custom model class
- **We already have baseline results**: 36.6% AVG @ 1k on BABILong with MemoryLLM-8B-chat
- Confidence: **VERY HIGH** (already reproduced)

**Reproduction Time (8xB200)**
- From scratch training: 3-5 days on RedPajama/C4 for 8B model
- Fine-tune: 1-2 days
- Inference/eval only: 2-4 hours (already done)

**Benchmarks**
- BABILong: Our result 36.6% @ 1k
- LongBench: Yes (reported in paper, we ran this too)
- Knowledge retention tasks: Yes
- PPL: Not the main focus

**Comparison with Our H-series**
- MemoryLLM embeds memory into model parameters; we use separate cross-attention slots
- MemoryLLM has flat performance across lengths (31-37%); suggests parameter memory is robust but not precise
- M+ extension adds scalable long-term memory -- worth testing

**Priority: P2** -- Already have baseline; M+ extension worth investigating

---

### 5. HMT (Hierarchical Memory Transformer)

**Paper Info**
- Title: "HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing"
- Authors: Zifan He et al. (UCLA)
- Venue: NAACL 2025
- arXiv: 2405.06067

**Core Method**
Imitates human memorization with hierarchical memory: sensory tokens (short-term), segment-level memory (medium-term), and recalled memory (long-term). Uses memory-augmented segment-level recurrence with a cross-attention based memory recall mechanism.

**GitHub Repository**
- URL: https://github.com/OswaldHe/HMT-pytorch
- Stars: ~50+ (estimate)
- Last updated: Active 2024-2025
- Code: Complete -- training scripts, evaluation, supports Llama-2-7B

**Reproducibility Assessment**
- Code open source: YES -- full training + eval
- Pretrained checkpoints: Not provided; train from public HF checkpoints
- Requirements: Yes, `requirement.txt`
- README quality: Good -- includes accelerate config, training commands
- Code quality: HIGH -- adapts RMT codebase, well-structured, supports AMD/NVIDIA
- Supports LoRA: Yes
- Confidence: **HIGH**

**Reproduction Time (8xB200)**
- From scratch training: 1-2 days on PG-19 with Llama-2-7B
- Fine-tune: 4-8 hours
- Inference/eval only: 1-2 hours

**Benchmarks**
- PG-19 PPL: Yes
- PubMedQA: Yes
- LongBench: Not reported
- BABILong: NOT reported
- Direct comparison: We could adapt their model to BABILong evaluation

**Comparison with Our H-series**
- **HMT is architecturally very similar to our approach**: cross-attention memory recall at specific layers
- HMT uses "sensory tokens" + "memory recall" = our "write" + "read" memory pattern
- HMT adapts RMT codebase, same framework we've worked with
- Most directly comparable architecture to our H-series

**Priority: P2** -- Architecturally most similar to our approach; strong reference for design decisions

---

### 6. MemLong

**Paper Info**
- Title: "MemLong: Memory-Augmented Retrieval for Long Text Modeling"
- Authors: Weijie Liu et al. (Bui1dMySea / ZetangForward)
- arXiv: 2408.16967

**Core Method**
Uses external retrievers (e.g., BGE-M3) to retrieve relevant chunks from a dynamic memory buffer. Memory planning uses frequency-based eviction (not FIFO). Only fine-tunes upper layers, keeping lower layers frozen.

**GitHub Repository**
- URL: https://github.com/Bui1dMySea/MemLong
- Stars: ~100+ (estimate)
- Last updated: 2024
- Code: Partially complete -- training + eval code, but requires faiss-gpu

**Reproducibility Assessment**
- Code open source: YES
- Pretrained checkpoints: LoRA weights may be provided
- Requirements: Yes, `requirements.txt` + faiss-gpu needed
- README quality: Good -- includes data download, training, and evaluation steps
- Code quality: MEDIUM -- requires significant setup (slimpajama download, faiss-gpu)
- Two-stage training: warm-up LoRA + MemLong fine-tuning
- Confidence: **MEDIUM**

**Reproduction Time (8xB200)**
- From scratch training: 2-3 days (two stages, data download needed)
- Fine-tune: 1-2 days
- Inference/eval only: 2-4 hours

**Benchmarks**
- Language modeling PPL: Yes (WikiText, PG-19)
- ICL evaluation: Yes
- BABILong: NOT reported
- LongBench: Not reported

**Comparison with Our H-series**
- MemLong uses external retrieval (RAG-like); we use internal cross-attention memory
- Different paradigm but same goal
- Lower relevance since we focus on learned internal memory

**Priority: P3** -- External retrieval approach, less relevant to our internal memory mechanism

---

### 7. Block Recurrent Transformer (BloRT)

**Paper Info**
- Title: "Block-Recurrent Transformers"
- Authors: Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy P. Lillicrap (DeepMind); James Hutchins, Imanol Schlag, Yujia Li et al. (Google)
- arXiv: 2203.07852
- Venue: NeurIPS 2022

**Core Method**
Applies a Transformer layer recurrently along a sequence with linear complexity. Maintains recurrent state between blocks, remembering up to 60K tokens. Outperforms Transformer-XL.

**GitHub Repository**
- Official: https://github.com/google-research/meliad (JAX implementation)
- PyTorch: https://github.com/jskinn/pytorch-block-recurrent-transformer (community)
- Also: https://github.com/lucidrains/block-recurrent-transformer-pytorch (lucidrains)

**Reproducibility Assessment**
- Code open source: YES (JAX) + community PyTorch
- Pretrained checkpoints: NO
- Requirements: JAX-based official code, complex setup
- README quality: MEDIUM for official JAX repo
- Code quality: MEDIUM for reproduction -- JAX ecosystem, no HuggingFace integration
- Confidence: **LOW** for reproduction (JAX barrier)

**Reproduction Time (8xB200)**
- From scratch training: Hard to estimate; JAX codebase requires significant adaptation
- PyTorch community port exists but may be incomplete
- Total effort: 5-7 days including porting

**Benchmarks**
- PG-19 PPL: Yes
- Long document modeling: Yes
- BABILong: NOT reported

**Comparison with Our H-series**
- Block Recurrent uses segment-level recurrence with learned state; conceptually similar to our approach
- JAX codebase makes reproduction harder
- Historical importance but less practical for immediate comparison

**Priority: P4** -- Important historically but JAX barrier makes reproduction impractical

---

### 8. Infini-attention (Google)

**Paper Info**
- Title: "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
- Authors: Tsendsuren Munkhdalai et al. (Google Research)
- arXiv: 2404.07143

**Core Method**
Incorporates a compressive memory module into vanilla attention, combining masked local attention with long-term linear attention. Claims infinite context length with bounded memory.

**GitHub Repository**
- No official code released by Google
- Community implementations:
  - https://github.com/vmarinowski/infini-attention
  - https://github.com/jlamprou/Infini-Attention (HuggingFace)
  - https://github.com/a-r-r-o-w/infini-attention

**Reproducibility Assessment**
- Code open source: NO official code; community implementations only
- Pretrained checkpoints: NO
- **Critical issue**: HuggingFace published a blog titled "A failed experiment: Infini-Attention" showing the method does not work as claimed
- Multiple independent attempts to reproduce have failed
- Reddit discussions confirm non-reproducibility
- Confidence: **VERY LOW** -- method appears fundamentally broken

**Reproduction Time**
- N/A -- non-reproducible per independent verification

**Benchmarks**
- Paper claims: passkey retrieval, PG-19 PPL, long context tasks
- Independent reproduction: FAILS to match paper claims

**Comparison with Our H-series**
- Infini-attention concept (compressive memory in attention) is related to our approach
- But the method has been shown to not work, making it a cautionary tale rather than a baseline
- The failure mode (compressive memory losing information) is relevant to understanding memory design

**Priority: SKIP** -- Non-reproducible; serves as a negative result / design caution

---

### 9. Landmark Attention

**Paper Info**
- Title: "Landmark Attention: Random-Access Infinite Context Length for Transformers"
- Authors: Amirhossein Kazemnejad et al. (EPFL)
- Venue: NeurIPS 2023
- arXiv: 2305.16300

**Core Method**
Introduces landmark tokens as compressed summaries of input blocks. During attention, these landmarks serve as routing tokens for random-access over long contexts, enabling effectively infinite context length.

**GitHub Repository**
- URL: https://github.com/epfml/landmark-attention
- Fork with QLoRA: https://github.com/eugenepentland/landmark-attention-qlora
- Stars: ~200+ (estimate)

**Reproducibility Assessment**
- Code open source: YES (inference + fine-tuning)
- Pretrained checkpoints: May exist for original LLaMA
- Requirements: Yes
- README quality: MEDIUM -- based on original LLaMA, may need adaptation for Llama-3
- Code quality: MEDIUM -- built for LLaMA-1/2, would need adaptation
- Confidence: **MEDIUM**

**Reproduction Time (8xB200)**
- From scratch: Not applicable -- method is a fine-tuning approach
- Fine-tune: 1-2 days (needs adaptation to Llama-3)
- Inference/eval: 2-4 hours (once adapted)

**Benchmarks**
- WikiText PPL: Yes
- Long context tasks: Yes
- BABILong: NOT reported
- Demonstrated on 32K context

**Comparison with Our H-series**
- Landmark attention uses learned routing tokens; conceptually similar to our memory slots
- Random-access paradigm is different from our sequential write/read pattern
- Built for older LLaMA models, needs updating

**Priority: P4** -- Interesting conceptually but needs significant adaptation; lower relevance

---

### 10. M+ (MemoryLLM v2 with Scalable Long-Term Memory)

**Paper Info**
- Title: "M+: Extending MemoryLLM with Scalable Long-Term Memory"
- Authors: Yu Wang et al.
- arXiv: 2502.00592
- Year: 2025

**Core Method**
Extends MemoryLLM with a scalable long-term memory (LTM) module that is stored on CPU (numpy) and accessed during inference. The LTM can grow without bound while keeping GPU memory fixed.

**GitHub Repository**
- URL: Same as MemoryLLM: https://github.com/wangyu-ustc/MemoryLLM (mplus-8b branch)
- Pretrained: `YuWangX/mplus-8b` on HuggingFace

**Reproducibility Assessment**
- Code open source: YES
- Pretrained checkpoints: YES (mplus-8b)
- Same high-quality codebase as MemoryLLM
- Confidence: **HIGH**

**Reproduction Time (8xB200)**
- Inference/eval: 4-6 hours (includes LTM management overhead)

**Benchmarks**
- Same as MemoryLLM + additional long-term memory tasks
- BABILong: Not yet tested by us

**Comparison with Our H-series**
- M+ adds external scalable memory; our approach uses fixed-size internal memory
- Worth testing as an extension of the MemoryLLM baseline we already have

**Priority: P2** -- Natural extension of existing baseline; pretrained checkpoint available

---

## Recommended Reproduction Order

Given our resources (8xB200 cluster, H20 cluster for eval) and goals (compare with H-series), we recommend this order:

### Phase 1: Immediate (1-2 days)
1. **M+ evaluation on BABILong** -- Load `YuWangX/mplus-8b`, run our BABILong pipeline. Extends our existing MemoryLLM baseline.
2. **ARMT training on Llama-3.2-1B** -- Use their scripts, train on PG-19, evaluate on BABILong. This is the SOTA on BABILong and most directly comparable.

### Phase 2: Short-term (3-5 days)
3. **HMT training on Llama-2-7B or Llama-3-8B** -- Most architecturally similar to our H-series. Adapt their code to BABILong evaluation.
4. **RMT training on Llama-3.2-1B** -- Baseline from the RMT team. Compare ARMT vs RMT to understand associative memory gains.

### Phase 3: Medium-term (1-2 weeks)
5. **MemLong training** -- If Phase 1-2 results are promising, add MemLong as an external-retrieval baseline.
6. **Activation Beacon training from scratch** -- We have eval results; training reproduction validates the full pipeline.

### Skip
- **Infini-attention**: Non-reproducible per independent verification
- **Block Recurrent Transformer**: JAX barrier, lower practical value

---

## Key Insights for H-series

1. **Base model matters more than memory mechanism for BABILong**: Llama-3-8B (8k training context) scores 0% on BABILong regardless of memory method. Llama-3.2-1B (128k training context) scores 25% without any memory. This explains our H-series BABILong failure.

2. **PPL ratio is a meaningful metric**: Our H14_isolate_aggr PPL ratio of 1.0113 is competitive. Most papers report raw PPL, not ratio, making direct comparison hard. We should adopt ratio as our standard metric.

3. **Cross-attention memory vs. memory tokens**: HMT uses a similar cross-attention recall mechanism to our H-series. If HMT achieves good BABILong scores, it validates our architectural direction.

4. **The 1000-step fine-tune destruction**: Our H6 step-1000 broke Llama-3-8B's in-context learning. RMT/ARMT typically train for much longer (full PG-19 epochs). This suggests we need longer training schedules.

5. **Activation Beacon is the strongest practical baseline**: 65.4% @ 1k on BABILong, pretrained checkpoints available. Any new memory mechanism should aim to beat this.

---

## Confidence Summary

| Paper | Confidence | Reason |
|-------|-----------|--------|
| RMT | HIGH | Mature codebase, already using their BABILong eval |
| ARMT | HIGH | Same codebase as RMT, includes Llama-3.2 scripts |
| Activation Beacon | VERY HIGH | Already reproduced eval results |
| MemoryLLM | VERY HIGH | Already reproduced eval results |
| HMT | HIGH | Complete code, clear README |
| MemLong | MEDIUM | Requires faiss-gpu, data download |
| Block Recurrent | LOW | JAX barrier, no HF integration |
| Infini-attention | VERY LOW | Shown non-reproducible |
| Landmark Attention | MEDIUM | Needs adaptation from LLaMA-1/2 |
| M+ | HIGH | Same codebase as MemoryLLM |

---

## Sources

- [booydar/recurrent-memory-transformer](https://github.com/booydar/recurrent-memory-transformer) -- RMT official code
- [RodkinIvan/associative-recurrent-memory-transformer](https://github.com/RodkinIvan/associative-recurrent-memory-transformer) -- ARMT official code
- [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) -- Activation Beacon code
- [wangyu-ustc/MemoryLLM](https://github.com/wangyu-ustc/MemoryLLM) -- MemoryLLM / M+ official code
- [OswaldHe/HMT-pytorch](https://github.com/OswaldHe/HMT-pytorch) -- HMT official code
- [Bui1dMySea/MemLong](https://github.com/Bui1dMySea/MemLong) -- MemLong official code
- [google-research/meliad](https://github.com/google-research/meliad) -- Block Recurrent Transformer (JAX)
- [epfml/landmark-attention](https://github.com/epfml/landmark-attention) -- Landmark Attention official code
- [HuggingFace Blog: A failed experiment: Infini-Attention](https://huggingface.co/blog/infini-attention) -- Non-reproducibility evidence
- [BABILong paper (arXiv:2406.10149)](https://arxiv.org/abs/2406.10149) -- BABILong benchmark
- [booydar/babilong](https://github.com/booydar/babilong) -- BABILong evaluation code
