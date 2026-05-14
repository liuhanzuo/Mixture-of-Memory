# Full SFT Training Plan for V4 ChunkMemory

## Problem Statement

LoRA training (rank=16, Q/V only) degraded base PPL from 6.70 to 9.08. The root cause
is that LoRA on a frozen backbone cannot adapt the full model to the memory-bank
attention pattern without hurting its language modeling capability. Full SFT (all
parameters trainable) should allow the model to jointly learn: (1) normal language
modeling and (2) memory-bank-augmented attention, without one hurting the other.

## Design: `scripts/train_v4_full_sft.py`

---

## 1. Model Architecture Changes

### 1.1 Remove LoRA, Use Raw LlamaForCausalLM

The existing `ChunkMemoryModel` wraps a PeftModel with LoRA on Q/V projections. The
new design removes LoRA entirely and operates directly on LlamaForCausalLM with all
parameters trainable.

**What to REMOVE from the current script:**
```python
# DELETE these imports and all LoRA-related code:
from peft import LoraConfig, get_peft_model, TaskType

# DELETE from ChunkMemoryModel.__init__:
lora_config = LoraConfig(...)
for p in base_model.parameters():
    p.requires_grad = False
self.peft_model = get_peft_model(base_model, lora_config)
```

**New ChunkMemoryModel class:**

```python
class ChunkMemoryModel(nn.Module):
    """Wraps LlamaForCausalLM (fully trainable) with per-layer memory banks.

    Key difference from LoRA version:
    - No PeftModel wrapper. Direct LlamaForCausalLM.
    - ALL parameters are trainable.
    - Memory banks remain pure runtime state (no gradients, NOT nn.Module).
    """

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        num_slots: int = 64,
        top_k: int = 8,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.top_k = top_k
        self.epsilon = epsilon

        # Store the base model directly -- no LoRA, all params trainable.
        self.model = base_model

        # Derive model metadata.
        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        # Per-layer memory banks (plain Python objects, NOT nn.Module).
        self.banks: list[ChunkMemoryBank] = [
            ChunkMemoryBank(num_slots, self.d_model) for _ in range(self.num_layers)
        ]

        # Direct references to decoder layers.
        self._decoder_layers: list[nn.Module] = list(
            self.model.model.layers
        )

        # Enable gradient checkpointing on the base model.
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
```

### 1.2 Updated forward_chunk()

The forward_chunk logic stays the same, but references change from
`self.peft_model.base_model.model.model` to `self.model.model`:

```python
def forward_chunk(
    self,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> dict:
    B, T = input_ids.shape
    device = input_ids.device
    dtype = next(self.parameters()).dtype

    # Get internal LlamaModel.
    llama_model = self.model.model  # was: self.peft_model.base_model.model.model
    embed_tokens = llama_model.embed_tokens
    hidden_states = embed_tokens(input_ids).to(dtype)

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    rotary_emb = llama_model.rotary_emb
    position_embeddings = rotary_emb(hidden_states, position_ids)

    # Build base causal mask.
    neg_inf = torch.finfo(dtype).min
    base_causal = torch.triu(
        torch.full((T, T), neg_inf, dtype=dtype, device=device), diagonal=1
    )
    base_causal_4d = base_causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T).contiguous()

    # Pass through each decoder layer with bank injection.
    # IDENTICAL logic to current version -- slots prepended, prefix causal mask,
    # position embedding extension, bank updates (append or top-k EMA).
    # See current lines 211-296. Only change is variable references.

    for layer_idx, layer in enumerate(self._decoder_layers):
        bank = self.banks[layer_idx]
        n_filled = bank.num_filled

        if n_filled == 0:
            # Normal forward, no slots.
            layer_out = layer(
                hidden_states,
                attention_mask=base_causal_4d,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            last_h = hidden_out[:, -1, :].detach()
            bank.append(last_h)
            hidden_states = hidden_out
        else:
            # Slots present: build extended sequence + prefix causal mask.
            # ... identical to current lines 234-296 ...
            # (Phase 1 append or Phase 2 top-k + EMA)
            pass

    # Final layernorm + LM head.
    llama_model_out = llama_model.norm(hidden_states)
    lm_head = self.model.lm_head  # was: self.peft_model.base_model.model.lm_head
    logits = lm_head(llama_model_out)

    result = {"logits": logits}
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        result["loss"] = loss
    return result
```

### 1.3 New Method: forward_plain() for Pretrain Data

Pretrain chunks (90% of data) go through normal Llama forward without memory banks.
This preserves the base LM capability:

```python
def forward_plain(
    self,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> dict:
    """Standard Llama forward, no memory banks. For pretrain data."""
    # Simply delegate to the standard LlamaForCausalLM forward.
    outputs = self.model(input_ids=input_ids, labels=labels)
    return {"logits": outputs.logits, "loss": outputs.loss}
```

This is critical: during pretrain steps, the model does vanilla NTP with no slot
injection. This prevents the base LM from degrading.

---

## 2. Data Loading: Dual-Source Mixed Training

### 2.1 Strategy

Both data sources use the same `pg19_chunks_llama3.npy` file (chunks of seq_len=4096
tokens each). The difference is how chunks are processed:

| Source | Ratio | Processing | Memory Banks |
|--------|-------|-----------|-------------|
| Pretrain | 90% | Individual chunks, flat | OFF (forward_plain) |
| Memory | 10% | Grouped into documents | ON (forward_chunk) |

### 2.2 Dual DataLoader Design

```python
class FlatChunkDataset(Dataset):
    """Pretrain data: each chunk is an independent sample."""

    def __init__(self, npy_path, seq_len, skip, max_chunks):
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip: skip + max_chunks].astype(np.int32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = torch.tensor(self.data[idx], dtype=torch.long)[:self.seq_len]
        return {"input_ids": t, "labels": t.clone(), "mode": "pretrain"}


class DocumentChunkDataset(Dataset):
    """Memory data: groups chunks_per_doc sequential chunks into a document.

    Reuses existing DocumentChunkDataset from train_v4_chunk_memory.py exactly.
    """

    def __init__(self, npy_path, seq_length, skip_chunks, max_chunks,
                 chunks_per_doc=8):
        # ... identical to current implementation ...
        pass

    def __getitem__(self, idx):
        # ... identical to current implementation ...
        # Returns {"chunks": [list of chunk dicts], "mode": "memory"}
        pass
```

### 2.3 Data Allocation

To avoid train/eval overlap, split the npy file:
- Total chunks in pg19_chunks_llama3.npy: need to check exact count
- Pretrain data: chunks 0 to N_pretrain (90% of budget)
- Memory data: chunks N_pretrain to N_pretrain + N_memory (10% of budget)
- Eval data: after N_pretrain + N_memory

```python
# Example allocation for 10,000 chunks:
pretrain_skip = 0
pretrain_count = 9000       # 90% for pretrain
memory_skip = 9000
memory_count = 1000         # 10% for memory training
eval_skip = 10000           # after training data
eval_count = 200
```

### 2.4 Training Loop: Per-Step Mode Selection

Each training step randomly selects pretrain or memory mode based on the 90/10 ratio:

```python
pretrain_ratio = 0.9

while global_step < args.max_steps:
    # Randomly choose mode for this step.
    if random.random() < pretrain_ratio:
        # PRETRAIN MODE: standard NTP, no memory banks.
        batch = next(pretrain_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        result = model(input_ids=input_ids, labels=labels)
        # Note: forward() should dispatch to forward_plain()
        loss = result["loss"]
    else:
        # MEMORY MODE: sequential chunk processing with banks.
        chunks = next(memory_iter)
        model.reset_banks()
        for chunk in chunks:
            input_ids = chunk["input_ids"].unsqueeze(0).to(device)
            labels = chunk["labels"].unsqueeze(0).to(device)
            result = model.forward_chunk(input_ids, labels=labels)
            loss = result["loss"]
            # Backward on EACH chunk in the document (gradients accumulate).
            if torch.isfinite(loss):
                loss = loss / args.chunks_per_doc  # normalize
                loss.backward()

    # Gradient step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    global_step += 1
```

### 2.5 Iteration Strategy

Use separate iterators for each data source, cycling when exhausted:

```python
def cycle_iterator(loader):
    """Infinite iterator that cycles through the DataLoader."""
    while True:
        for batch in loader:
            yield batch

pretrain_iter = cycle_iterator(pretrain_loader)
memory_iter = cycle_iterator(memory_loader)
```

---

## 3. Training Infrastructure

### 3.1 DeepSpeed ZeRO Stage 2 (Recommended)

ZeRO Stage 2 shards optimizer states and gradients across GPUs. With 8 GPUs:

| Component | Per GPU (ZeRO-2, 8 GPUs) |
|-----------|--------------------------|
| Model weights (bf16, replicated) | 16.06 GB |
| Optimizer states (fp32, sharded) | 8.03 GB |
| Gradients (fp32, sharded) | 4.01 GB |
| Activations (with GC) | ~5-10 GB |
| **Total** | **~33-38 GB** |

This fits comfortably in B200 (183 GB) and H20 (97.8 GB).

### 3.2 DeepSpeed Configuration

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "gradient_accumulation_steps": 4
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 1,
  "wall_clock_breakdown": false
}
```

Note: `gradient_accumulation_steps: 4` with `micro_batch_size: 1` gives effective
batch size of 8 GPUs * 1 * 4 = 32 per optimizer step.

### 3.3 Why NOT ZeRO Stage 3

ZeRO-3 shards model weights across GPUs, requiring all-gather during forward pass.
This adds communication overhead and complicates the memory bank hook mechanism
(because we access decoder layers directly). ZeRO-2 is sufficient since 38 GB per
GPU is well within available memory.

### 3.4 Gradient Checkpointing

Enable via `model.gradient_checkpointing_enable()` in ChunkMemoryModel.__init__.
This reduces activation memory from ~5 GB to ~2 GB per GPU by recomputing
activations during backward. The 30% compute overhead is acceptable.

**Important**: Gradient checkpointing must be enabled BEFORE wrapping in DDP/DeepSpeed.

---

## 4. Memory Considerations: Detailed Analysis

### 4.1 Llama-3-8B Memory Footprint

```
Parameters:           8.03B * 2 bytes (bf16)  = 16.06 GB
AdamW momentum:       8.03B * 4 bytes (fp32)  = 32.12 GB
AdamW variance:       8.03B * 4 bytes (fp32)  = 32.12 GB
Gradients:            8.03B * 4 bytes (fp32)  = 32.12 GB
-----------------------------------------------------------
Total (1 GPU):        112.42 GB
Total (ZeRO-2, 8 GPU): 28.10 GB/GPU + activations
```

### 4.2 Batch Size Calculation

With ZeRO-2 on B200 (183 GB):
- Fixed overhead: ~28 GB
- Available for activations: ~155 GB
- Activation per sample (seq_len=4096, bf16): ~5 GB full, ~2 GB with GC
- Conservative estimate with GC: batch_size=1 per GPU is safe
- Could potentially do batch_size=2-4 with GC, but chunk processing is sequential
  within a document, so batch_size=1 is the natural choice

**Recommendation**: batch_size=1 per GPU, gradient_accumulation=4, effective
batch=32 per optimizer step.

### 4.3 Memory Bank Memory

Memory banks store [B, num_slots, d_model] tensors in bf16:
- Per layer: 1 * 64 * 4096 * 2 bytes = 0.5 MB
- 32 layers: 16 MB total
- Completely negligible compared to model/optimizer states.

### 4.4 Extended Sequence Overhead

When slots are prepended, the effective sequence length is `n_slots + T`:
- Phase 1 (filling): up to 64 + 4096 = 4160 tokens
- Phase 2 (top-k): top_k + 4096 (e.g., 4 + 4096 = 4100)
- Attention matrix: [4160, 4160] vs [4096, 4096] -- ~3% increase, negligible.

---

## 5. Hyperparameters

```python
# Full SFT hyperparameters for Llama-3-8B
hp = {
    "lr": 1e-5,                    # Conservative for full SFT (vs 1e-4 for LoRA)
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_steps": 2000,
    "lr_scheduler": "cosine",
    "gradient_accumulation_steps": 4,
    "per_device_batch_size": 1,
    "effective_batch_size": 32,     # 8 GPUs * 1 * 4 accum
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,

    # Data mixing
    "pretrain_ratio": 0.9,
    "chunks_per_doc": 8,

    # Memory bank
    "num_slots": 64,
    "top_k": 8,
    "epsilon": 0.05,
    "ema_decay": 0.9,

    # Data
    "seq_len": 4096,
    "pretrain_max_chunks": 9000,
    "memory_max_chunks": 1000,
}
```

### 5.1 LR Rationale

- Full SFT of 8B models typically uses 1e-5 to 5e-5 (vs LoRA's 1e-4)
- Too high LR will destabilize the pretrained weights
- Start at 1e-5 with cosine decay to ~1e-6
- If training is too slow, can increase to 3e-5

### 5.2 Steps Rationale

- 2000 steps * 32 effective batch = 64K training samples
- Each sample is a 4096-token chunk = ~262M tokens of training
- This is a light SFT, not full pretraining. Sufficient for adapting to memory banks.

---

## 6. Checkpoint Saving and Evaluation

### 6.1 Checkpoint Format

Since there is no LoRA, save the full model state:

```python
def save_checkpoint(model, optimizer, global_step, output_dir):
    # Get the underlying model (unwrap DDP/DeepSpeed).
    root = model.module if hasattr(model, "module") else model

    torch.save({
        "global_step": global_step,
        "model_state_dict": root.model.state_dict(),  # Full LlamaForCausalLM
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(output_dir, f"step_{global_step}.pt"))
```

Alternatively, save as HuggingFace format for easier loading:

```python
root.model.save_pretrained(os.path.join(output_dir, f"step_{global_step}"))
tokenizer.save_pretrained(os.path.join(output_dir, f"step_{global_step}"))
```

### 6.2 Evaluation

Two eval modes, same as current script:

1. **Vanilla PPL**: Reset banks before each chunk, forward through model.
   Tests base LM capability (should stay close to 6.70).
2. **Memory PPL**: Banks active across chunks_per_doc chunks.
   Tests memory-augmented capability (should improve from 9.08).

Eval runs every `eval_interval` steps (default 100).

### 6.3 Go/No-Go Criteria

```
After step 500:
- Vanilla PPL should be < 8.0 (base capability preserved)
- Memory PPL should be < vanilla PPL (memory is helpful, not harmful)
- If vanilla PPL > 10.0: KILL and diagnose (LR too high, data issue, etc.)
```

---

## 7. Launch Script

`scripts/launch_v4_full_sft.sh`:

```bash
#!/bin/bash
set -e
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
mkdir -p logs outputs/v4_full_sft
export PYTHONUNBUFFERED=1
export PYTHONPATH=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory:$PYTHONPATH

torchrun --nproc_per_node=8 scripts/train_v4_full_sft.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
  --num_slots 64 \
  --top_k 8 \
  --epsilon 0.05 \
  --lr 1e-5 \
  --max_steps 2000 \
  --pretrain_max_chunks 9000 \
  --memory_max_chunks 1000 \
  --skip_chunks 0 \
  --seq_len 4096 \
  --chunks_per_doc 8 \
  --pretrain_ratio 0.9 \
  --gradient_accumulation_steps 4 \
  --warmup_steps 100 \
  --eval_interval 100 \
  --output_dir outputs/v4_full_sft \
  2>&1 | tee logs/v4_full_sft_$(date +%Y%m%d_%H%M%S).log
```

---

## 8. Implementation Checklist

Step-by-step changes from `train_v4_chunk_memory.py` to `train_v4_full_sft.py`:

### 8.1 Imports
- [ ] REMOVE: `from peft import LoraConfig, get_peft_model, TaskType`
- [ ] KEEP: All other imports (torch, transformers, ChunkMemoryBank, etc.)
- [ ] ADD: `from torch.optim.lr_scheduler import CosineLRScheduler` or use transformers get_scheduler

### 8.2 ChunkMemoryModel Class
- [ ] REMOVE: `lora_rank` parameter from __init__
- [ ] REMOVE: All LoRA-related code (LoraConfig, get_peft_model, freeze backbone)
- [ ] CHANGE: `self.peft_model = get_peft_model(...)` to `self.model = base_model`
- [ ] ADD: `self.model.gradient_checkpointing_enable(...)` in __init__
- [ ] CHANGE: `_get_decoder_layers()` from `self.peft_model.base_model.model.model` to `self.model.model`
- [ ] CHANGE: All `self.peft_model.base_model.model.model` references to `self.model.model`
- [ ] CHANGE: `self.peft_model.base_model.model.lm_head` to `self.model.lm_head`
- [ ] ADD: `forward_plain()` method for pretrain data (standard Llama forward)
- [ ] KEEP: `reset_banks()`, `forward_chunk()`, `make_prefix_causal_mask()`, `extend_position_embeddings()` -- all identical

### 8.3 Data Classes
- [ ] KEEP: `DocumentChunkDataset` (for memory training data) -- identical
- [ ] ADD: `FlatChunkDataset` (move from inline class in main to top level)
- [ ] ADD: Separate data splits: pretrain (90%) and memory (10%)
- [ ] ADD: `cycle_iterator()` helper for infinite cycling

### 8.4 Training Loop
- [ ] CHANGE: Optimizer uses ALL model parameters (not just LoRA params)
- [ ] ADD: Cosine LR scheduler with warmup
- [ ] CHANGE: Training loop to dual-mode (pretrain vs memory based on ratio)
- [ ] ADD: Gradient accumulation logic (accumulate across chunks within a doc)
- [ ] KEEP: Bank reset at document boundaries
- [ ] KEEP: Phase 1/Phase 2 bank logic (append until full, then top-k)
- [ ] CHANGE: Checkpoint saving from LoRA state dict to full model state dict

### 8.5 Evaluation
- [ ] KEEP: `evaluate_vanilla_ppl()` -- mostly unchanged
- [ ] ADD: Log both vanilla PPL and memory PPL at each eval
- [ ] ADD: Go/No-Go criteria (vanilla PPL < 8.0 at step 500)

### 8.6 Distributed Training
- [ ] CHANGE: Replace plain DDP with DeepSpeed ZeRO-2 (recommended)
- [ ] Alternative: Keep plain DDP if DeepSpeed is not available
- [ ] ADD: DeepSpeed config JSON file or dict

---

## 9. Risk Mitigation

### 9.1 OOM Risk
- Gradient checkpointing reduces activation memory by ~60%
- ZeRO-2 shards optimizer/gradients across 8 GPUs
- batch_size=1 with seq_len=4096 is conservative
- If OOM: reduce seq_len to 2048, or add CPU offload for optimizer

### 9.2 Training Instability
- LR 1e-5 is conservative for full SFT
- Gradient clipping at 1.0 prevents exploding gradients
- Warmup of 100 steps ensures stable start
- bf16 mixed precision is standard for Llama training

### 9.3 Base PPL Degradation
- The 90% pretrain data ensures the model maintains its LM capability
- forward_plain() does not inject memory slots, so 90% of gradient updates
  are pure NTP on normal text
- Eval every 100 steps catches degradation early

### 9.4 Catastrophic Forgetting
- The mixed training strategy (90% pretrain + 10% memory) is specifically
  designed to prevent catastrophic forgetting of base capabilities
- If vanilla PPL starts rising, increase pretrain_ratio to 0.95

---

## 10. What to KEEP Unchanged

The following components are reused verbatim from `train_v4_chunk_memory.py`:

1. **make_prefix_causal_mask()** -- Lines 53-82. Identical.
2. **extend_position_embeddings()** -- Lines 89-102. Identical.
3. **ChunkMemoryBank** -- In `src/memory/mem_space/chunk_memory_bank.py`. Identical.
4. **DocumentChunkDataset** -- Lines 326-371. Identical (used for memory data).
5. **doc_collate_fn()** -- Line 374-378. Identical.
6. **init_distributed()** -- Lines 385-393. Identical.
7. **forward_chunk() core logic** -- Lines 169-316. Same slot injection, same
   prefix mask, same position embedding extension. Only the model reference paths change.
8. **reset_banks()** -- Lines 165-167. Identical.

The key insight: the algorithm (slot injection, prefix causal mask, top-k selection,
EMA update) is sound. The problem was purely the LoRA bottleneck preventing full
adaptation. Switching to full SFT with mixed pretrain/memory data should resolve the
PPL degradation while preserving memory-augmented capability.
