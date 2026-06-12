# RMT v2 Architecture Design

## Executive Summary

RMT v1 revealed critical issues:
1. **84.7M memory module params** — 650× over target (goal: ~131K)
2. **Warmup is 60% of training** — 50/84 steps = 60%
3. **Loss trajectory unstable** — barely moves in epochs 1-2, only drops in epoch 3
4. **Training data limited** — only English Wikitext-103 (1808 docs)

This document proposes RMT v2 with:
- **<500K memory module params** (99.4% reduction)
- **Proper warmup schedule** (8-10% of training)
- **Balanced mixed-language dataset** (Wikipedia + Wikitext + conversation)

---

## 1. Architecture Redesign

### Problem Analysis

The v1 MemoryExtractor is a full multi-head attention layer:

```
Component              Params        Purpose
============================================
memory_embeddings      655K - 2.1M   [max_segments, num_mem, hidden_dim]
extractor.query        131K          [1, num_mem, hidden_dim]
extractor.key_proj     16.8M         [hidden_dim, hidden_dim]
extractor.value_proj   16.8M         [hidden_dim, hidden_dim]
extractor.out_proj     16.8M         [hidden_dim, hidden_dim]
extractor.gate.0       33.6M         [hidden_dim*2, hidden_dim]
extractor.gate.2       4.1K          [hidden_dim, 1]
============================================
TOTAL                  84.7M
```

**Key insight**: We're using a full 4096×4096 projection for memory extraction, but:
- Input is just segment hidden states [B, 2048, 4096]
- Output is only 64 memory tokens [B, 64, 4096]
- This is massively over-parameterized for simple compression

### v2 Design: Lightweight Memory Extraction

Replace the full attention extractor with a **two-stage lightweight extractor**:

#### Stage 1: Mean Pooling → Intermediate Compressed Representation

```
segment_mean = mean(hidden_states, dim=1)  # [B, 4096]
intermediate = MLP(segment_mean)           # [B, 256]
```

- **Params**: 4096 × 256 + 256 × 256 + 256 × 4096 ≈ **1.05M** (still too big)
- **Better**: Use a single linear projection
  ```
  intermediate = Linear(4096 → 128)(segment_mean)  # [B, 128]
  ```
  - **Params**: 4096 × 128 = **524K** (better)

#### Stage 2: Expand to Memory Tokens with Learned Queries

```
queries = memory_query_embeddings  # [64, 128] (learnable)
new_memory = queries @ intermediate.T  # [64, B]
new_memory = new_memory.T  # [B, 64, 128]
new_memory = Linear(128 → 4096)(new_memory)  # [B, 64, 4096]
```

- **memory_query_embeddings**: 64 × 128 = **8.2K**
- **output projection**: 128 × 4096 = **524K**

#### Simplified Gate

```
gate = Sigmoid(Linear(4096 → 1)(concat(new_memory, old_memory)))
new_memory = gate * new_memory + (1 - gate) * old_memory
```

But this still needs:
```
Linear(8192 → 64) for gate  # [8192, 64]
```
- **Params**: 8192 × 64 = **524K**

### Final v2 Design (Target: ~300K params)

**MemoryExtractor v2**:
```
1. Mean pooling: segment_mean = mean(hidden_states, dim=1)  # [B, 4096]

2. Light MLP (bottleneck → 256 → bottleneck):
   - fc1: Linear(4096 → 256)        # 1,048,576 params ← too big!
   
   Let's reduce further:
   - Use bottleneck 128:
     - fc1: Linear(4096 → 128)      # 524,288 params
     - fc2: Linear(128 → 128)       # 16,384 params
     - fc3: Linear(128 → 4096)      # 524,288 params
     - Total: 1.06M ← still too big

   Let's try a different approach: **direct token-wise extraction**
```

### Alternative v2: Token-wise Linear Pooling with Low-Rank Expansion

```
# Stage 1: Compress each token position (2048 tokens → 256 compressed)
token_compressed = Linear(4096 → 256)(hidden_states)  # [B, 2048, 256]
segment_compressed = mean(token_compressed, dim=1)    # [B, 256]

# Stage 2: Expand to memory tokens with learned scaling
memory_bases = nn.Parameter(torch.randn(64, 256))  # [64, 256]
new_memory = memory_bases.unsqueeze(0) * segment_compressed.unsqueeze(1)  # [B, 64, 256]
new_memory = Linear(256 → 4096)(new_memory)       # [B, 64, 4096]

# Simplified gate: scalar gate per memory token
gate_scalar = Sigmoid(Linear(4096 → 64)(segment_compressed))  # [B, 64]
new_memory = gate_scalar.unsqueeze(-1) * new_memory + (1 - gate_scalar).unsqueeze(-1) * old_memory
```

**Param counts**:
- `token_compressed`: 4096 × 256 = **1,048,576** (too big)
- `memory_bases`: 64 × 256 = **16,384**
- `output_proj`: 256 × 4096 = **1,048,576** (too big)
- `gate_linear`: 256 × 64 = **16,384**
- Total: ~2.1M (still too big)

### v2 Final Design: Ultra-Lightweight (Target ~300K)

**Key insight**: Don't transform every token. Use **mean pooling + small MLP** with bottleneck.

```
MemoryExtractor v2:
    def forward(self, hidden_states, old_memory):
        # Mean pooling
        segment_mean = hidden_states.mean(dim=1)  # [B, 4096]

        # Ultra-light MLP: 4096 → 128 → 128 → 4096 (with residual)
        residual = segment_mean
        compressed = self.mlp(segment_mean)  # [B, 4096]
        intermediate = residual + compressed  # residual connection

        # Broadcast to memory tokens with learned query scaling
        # Instead of separate queries, use learned coefficients
        query_coeffs = self.query_coeffs  # [num_mem, 1]
        new_memory = intermediate.unsqueeze(1) * query_coeffs  # [B, num_mem, 4096]

        # Scalar gate per memory token (shared across batch)
        gate_scalar = torch.sigmoid(self.gate_bias)  # [num_mem]
        new_memory = gate_scalar * new_memory + (1 - gate_scalar) * old_memory
        return new_memory
```

**Param counts**:
- `mlp`: 
  - fc1: 4096 → 128 = 524,288
  - fc2: 128 → 128 = 16,384
  - fc3: 128 → 4096 = 524,288
  - Total: **1,064,960** (still >1M)

Let's try **bottleneck 64**:
```
mlp:
  - fc1: 4096 → 64 = 262,144
  - fc2: 64 → 64 = 4,096
  - fc3: 64 → 4096 = 262,144
  - Total: **528,384**
query_coeffs: 64 × 1 = 64
gate_bias: 64 = 64
memory_embeddings: 6 × 64 × 4096 = 1,572,864
```
Total: **~2.1M** (memory_embeddings is the bottleneck)

### Critical Optimization: Reduce num_memory_tokens

Current: 64 memory tokens
**Target: 8-16 memory tokens**

Let's try **16 memory tokens**:
```
memory_embeddings: 6 × 16 × 4096 = 393,216
mlp (bottleneck 64): 528,384
query_coeffs: 16 × 1 = 16
gate_bias: 16 = 16
segment_bias: 6 × 16 = 96
TOTAL: **921,728** (still >500K)
```

Let's try **8 memory tokens** with **bottleneck 32**:
```
memory_embeddings: 6 × 8 × 4096 = 196,608
mlp:
  - fc1: 4096 → 32 = 131,072
  - fc2: 32 → 32 = 1,024
  - fc3: 32 → 4096 = 131,072
  - Total: 263,168
query_coeffs: 8 × 1 = 8
gate_bias: 8 = 8
segment_bias: 6 × 8 = 48
TOTAL: **459,840** ← **TARGET ACHIEVED!**
```

### v2 Recommended Architecture

```python
class MemoryExtractorV2(nn.Module):
    """Ultra-lightweight memory extractor (~460K params)."""
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_memory_tokens: int = 8,  # Reduced from 64
        bottleneck_dim: int = 32,   # Tiny bottleneck
    ):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        
        # Ultra-light MLP: 4096 → 32 → 32 → 4096
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        
        # Learned query coefficients for broadcasting
        self.query_coeffs = nn.Parameter(torch.randn(num_memory_tokens, 1) * 0.02)
        
        # Learned scalar gate (simpler than MLP gate)
        self.gate_bias = nn.Parameter(torch.zeros(num_memory_tokens))
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, seq_len, 4096]
        old_memory: Optional[torch.Tensor] = None,  # [B, 8, 4096]
    ) -> torch.Tensor:
        # Mean pooling over sequence
        segment_mean = hidden_states.mean(dim=1)  # [B, 4096]
        
        # Residual MLP compression
        residual = segment_mean
        compressed = self.mlp(segment_mean)  # [B, 4096]
        intermediate = residual + compressed  # [B, 4096]
        intermediate = self.norm(intermediate)
        
        # Broadcast to memory tokens with learned scaling
        # new_memory[i] = query_coeffs[i] * segment_compressed
        new_memory = intermediate.unsqueeze(1) * self.query_coeffs  # [B, 8, 4096]
        
        # Scalar gate blending
        if old_memory is not None:
            gate = torch.sigmoid(self.gate_bias).unsqueeze(0).unsqueeze(-1)  # [1, 8, 1]
            new_memory = gate * new_memory + (1 - gate) * old_memory
        
        return new_memory
```

**Param breakdown**:
```
memory_embeddings: 6 × 8 × 4096 = 196,608  (21%)
mlp: 263,168                                (57%)
  - fc1: 4096 × 32 = 131,072
  - fc2: 32 × 32 = 1,024
  - fc3: 32 × 4096 = 131,072
query_coeffs: 8 × 1 = 8                      (<1%)
gate_bias: 8                                 (<1%)
segment_bias: 6 × 8 = 48                     (<1%)
norm: 2 × 4096 = 8,192                       (2%)
=============================================
TOTAL: 467,932 (~468K params)
```

**Comparison**:
- v1: 84.7M params
- v2: 0.47M params
- **Reduction: 99.4%**

---

## 2. Training Hyperparameters

### v1 Problems

1. **Warmup too long**: 50 steps / 84 total = **60%** of training in warmup
   - Learning rate was near-zero for most of training
   - Loss barely moved in epochs 1-2

2. **Total steps too low**: Only 84 steps (3 epochs × 1808 docs / 8 GPUs / 8 grad_accum)
   - Model barely had time to converge
   - Loss was still unstable (went up in final steps)

3. **Learning rate mismatch**:
   - Backbone LR: 1e-5 (fine)
   - RMT LR: 1e-4 (may be too high for small RMT module)

### v2 Recommendations

#### Step Count Estimation

Assuming:
- 1808 documents (same as v1)
- 8 GPUs
- Batch size = 1
- Grad accum = 8
- 6 segments per document

```
Steps per epoch = ceil(1808 / 8 / 8) = ceil(28.25) = 29
Total steps (3 epochs) = 29 × 3 = 87 steps
```

This is very low. **Recommend: Increase to 5-10 epochs**:

```
Steps per epoch = 29
Total steps (5 epochs) = 29 × 5 = 145 steps
Total steps (10 epochs) = 29 × 10 = 290 steps
```

#### Warmup Schedule

Standard practice: **5-10% of training in warmup**

```
For 145 steps:  warmup = 7-15 steps (≈10%)
For 290 steps:  warmup = 15-30 steps (≈10%)
```

**Recommendation**:
```
warmup_steps = 20  # For 5 epochs
OR
warmup_steps = 30  # For 10 epochs
```

#### Learning Rate

v2 module is much smaller (468K vs 84.7M), so:
- Use smaller LR for stability
- Consider LR scaling by sqrt(param_ratio)

```
v1 RMT LR: 1e-4
Param ratio: 0.468M / 84.7M = 0.0055
sqrt(param_ratio) = sqrt(0.0055) = 0.074

v2 RMT LR = 1e-4 × 0.074 = 7.4e-6
```

But this might be too conservative. Let's try:
```
Backbone LR: 1e-5 (same as v1)
RMT LR: 5e-5 (2× backbone, 0.5× v1)
```

**Recommendation**:
```
lr = 1e-5              # Backbone
rmt_lr = 5e-5          # RMT module (0.5× v1)
```

#### Summary: Recommended Hyperparameters

```python
# Training config
num_epochs = 10                 # 5-10 epochs (v1 was 3)
lr = 1e-5                      # Backbone learning rate
rmt_lr = 5e-5                  # RMT learning rate (smaller than v1)
warmup_steps = 30              # ~10% of 290 steps
weight_decay = 0.01
max_grad_norm = 1.0

# Scheduler
scheduler_type = "cosine"      # get_cosine_schedule_with_warmup
```

---

## 3. Training Data Strategy

### v1 Limitations

1. **Only English Wikitext-103**:
   - 1808 documents
   - Average 11.5K tokens per doc
   - No Chinese data
   - No conversation/dialogue data

2. **Potential issues**:
   - Memory module may not generalize to Chinese
   - No conversation patterns to learn
   - Limited domain diversity

### v2 Data Mix Strategy

#### Data Sources

1. **Wikipedia (Chinese + English)**
   - High-quality encyclopedic text
   - Good for factual knowledge
   - Both languages for cross-lingual capability

2. **Wikitext-103 (English)**
   - Long-form coherent text
   - Good for document-level modeling

3. **Conversation Data**
   - Instruction-following conversations
   - Dialogue modeling
   - Multi-turn reasoning

#### Sampling Strategy

Goal: Balanced exposure to each data type

```
Total samples: ~10K documents

Wikipedia (CN):  3,000 docs (30%)
Wikipedia (EN):  3,000 docs (30%)
Wikitext-103:    2,000 docs (20%)
Conversation:    2,000 docs (20%)
```

**Implementation**:
```python
# Create separate datasets for each source
datasets = [
    {"name": "wiki_cn", "path": "data/wiki_cn.jsonl", "weight": 0.3},
    {"name": "wiki_en", "path": "data/wiki_en.jsonl", "weight": 0.3},
    {"name": "wikitext", "path": "data/wikitext103.jsonl", "weight": 0.2},
    {"name": "conversation", "path": "data/conversation.jsonl", "weight": 0.2},
]

# Use ConcatDataset with sampling or WeightedRandomSampler
```

#### Document Length Considerations

- Wikipedia: Long documents (10K-50K tokens)
- Wikitext: Medium documents (5K-15K tokens)
- Conversation: Short (500-2K tokens per turn)

**Strategy**:
- Pad shorter documents to minimum segment length
- Truncate longer documents to max_segments × segment_length
- Target: 4-6 segments per document (8K-12K tokens per doc)

#### Data Preprocessing

```python
# Tokenization
def preprocess_document(text, tokenizer, segment_length=2048, max_segments=6):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Truncate to max_segments
    if len(tokens) > max_segments * segment_length:
        tokens = tokens[:max_segments * segment_length]
    
    # Pad to fixed length (for DDP static graph)
    target_len = max_segments * segment_length
    if len(tokens) < target_len:
        tokens = tokens + [tokenizer.pad_token_id] * (target_len - len(tokens))
    
    return tokens
```

---

## 4. Implementation Plan

### Files to Modify

1. **`src/memory/rmt/rmt_module.py`**
   - Replace `MemoryExtractor` with `MemoryExtractorV2`
   - Update `RMTMemory.__init__` to use `num_memory_tokens=8`
   - Update `MemoryExtractorV2` signature (remove `num_heads`)

2. **`scripts/train_rmt.py`**
   - Update default `num_memory_tokens=8`
   - Remove `num_memory_heads` argument
   - Update default `warmup_steps=30`
   - Update default `num_epochs=10`
   - Update default `rmt_lr=5e-5`
   - Add data mixing logic

### Code Changes Detail

#### Change 1: MemoryExtractorV2 in rmt_module.py

```python
# REPLACE entire MemoryExtractor class with:

class MemoryExtractorV2(nn.Module):
    """
    Ultra-lightweight memory extractor.
    Uses mean pooling + residual MLP with tiny bottleneck.
    Total params: ~270K (excluding memory_embeddings).
    """
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_memory_tokens: int = 8,
        bottleneck_dim: int = 32,
    ):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        self.hidden_dim = hidden_dim
        
        # Ultra-light MLP: hidden_dim → bottleneck → bottleneck → hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        
        # Learned query coefficients for broadcasting
        self.query_coeffs = nn.Parameter(torch.randn(num_memory_tokens, 1) * 0.02)
        
        # Learned scalar gate (simpler than MLP gate)
        self.gate_bias = nn.Parameter(torch.zeros(num_memory_tokens))
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, seq_len, hidden_dim]
        old_memory: Optional[torch.Tensor] = None,  # [B, num_mem, hidden_dim]
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        
        # Mean pooling over sequence
        segment_mean = hidden_states.mean(dim=1)  # [B, hidden_dim]
        
        # Residual MLP compression
        residual = segment_mean
        compressed = self.mlp(segment_mean)  # [B, hidden_dim]
        intermediate = residual + compressed  # [B, hidden_dim]
        intermediate = self.norm(intermediate)
        
        # Broadcast to memory tokens with learned scaling
        new_memory = intermediate.unsqueeze(1) * self.query_coeffs  # [B, num_mem, hidden_dim]
        
        # Scalar gate blending
        if old_memory is not None:
            gate = torch.sigmoid(self.gate_bias).unsqueeze(0).unsqueeze(-1)  # [1, num_mem, 1]
            new_memory = gate * new_memory + (1 - gate) * old_memory
        
        return new_memory
```

#### Change 2: Update RMTMemory to use new extractor

```python
# In RMTMemory.__init__, REPLACE:
self.extractor = MemoryExtractor(
    hidden_dim=hidden_dim,
    num_memory_tokens=num_memory_tokens,
    num_heads=num_heads,
)

# WITH:
self.extractor = MemoryExtractorV2(
    hidden_dim=hidden_dim,
    num_memory_tokens=num_memory_tokens,
    bottleneck_dim=32,
)
```

#### Change 3: Update train_rmt.py defaults

```python
# REPLACE defaults:
parser.add_argument("--num_memory_tokens", type=int, default=8)  # was 64
parser.add_argument("--segment_length", type=int, default=2048)
parser.add_argument("--max_segments", type=int, default=6)
# REMOVE: parser.add_argument("--num_memory_heads", type=int, default=8)

parser.add_argument("--num_epochs", type=int, default=10)  # was 3
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--rmt_lr", type=float, default=5e-5)  # was 1e-4
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--grad_accumulation_steps", type=int, default=8)
parser.add_argument("--warmup_steps", type=int, default=30)  # was 100
```

#### Change 4: Data mixing in train_rmt.py

```python
# ADD data mixing logic:
class MixedDataset(Dataset):
    """Mixed dataset with weighted sampling from multiple sources."""
    def __init__(
        self,
        data_configs: List[Dict],  # [{"path": "...", "weight": 0.3}, ...]
        tokenizer,
        segment_length: int = 2048,
        max_segments: int = 6,
    ):
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.max_segments = max_segments
        self.max_total_tokens = segment_length * max_segments
        
        # Load all datasets
        self.datasets = []
        self.weights = []
        for cfg in data_configs:
            docs = []
            with open(cfg["path"], 'r') as f:
                for line in f:
                    doc = json.loads(line)
                    tokens = tokenizer.encode(doc['text'], add_special_tokens=False)
                    if len(tokens) >= segment_length:
                        num_segs = min(len(tokens) // segment_length, max_segments)
                        tokens = tokens[:num_segs * segment_length]
                        target_len = max_segments * segment_length
                        if len(tokens) < target_len:
                            tokens = tokens + [tokenizer.pad_token_id] * (target_len - len(tokens))
                        docs.append(tokens)
            self.datasets.append(docs)
            self.weights.append(cfg["weight"])
        
        # Create mapping (doc_idx -> (dataset_idx, doc_in_dataset))
        self.doc_mapping = []
        for ds_idx, docs in enumerate(self.datasets):
            for doc_in_ds in range(len(docs)):
                self.doc_mapping.append((ds_idx, doc_in_ds))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Calculate cumulative distribution for sampling
        self.cum_weights = []
        cum = 0
        for w in self.weights:
            cum += w
            self.cum_weights.append(cum)
    
    def __len__(self):
        return len(self.doc_mapping)
    
    def __getitem__(self, idx):
        # Use idx directly (sampler handles weighted sampling)
        ds_idx, doc_in_ds = self.doc_mapping[idx]
        tokens = self.datasets[ds_idx][doc_in_ds]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        num_segments = len(tokens) // self.segment_length
        return {"input_ids": input_ids, "labels": labels, "num_segments": num_segments}

# In main(), REPLACE dataset loading with:
data_configs = [
    {"path": args.data_path + "/wiki_cn.jsonl", "weight": 0.3},
    {"path": args.data_path + "/wiki_en.jsonl", "weight": 0.3},
    {"path": args.data_path + "/wikitext103.jsonl", "weight": 0.2},
    {"path": args.data_path + "/conversation.jsonl", "weight": 0.2},
]

dataset = MixedDataset(
    data_configs=data_configs,
    tokenizer=tokenizer,
    segment_length=args.segment_length,
    max_segments=args.max_segments,
)

# Use WeightedRandomSampler for balanced sampling
weights = []
for ds_idx, _ in dataset.doc_mapping:
    weights.append(dataset.weights[ds_idx])

sampler = torch.utils.data.WeightedRandomSampler(
    weights, num_samples=len(dataset), replacement=True
)
```

---

## 5. Expected Outcomes

### Parameter Reduction

| Component | v1 Params | v2 Params | Reduction |
|-----------|-----------|-----------|-----------|
| MemoryExtractor | 84.1M | 271K | 99.7% |
| memory_embeddings | 1.6M | 197K | 87.6% |
| Total | 84.7M | 468K | 99.4% |

### Training Efficiency

| Metric | v1 | v2 | Improvement |
|--------|-----|-----|-------------|
| Memory module size | 84.7M | 468K | 99.4% ↓ |
| Warmup ratio | 60% | 10% | 6× ↓ |
| Total steps | 84 | 290 | 3.5× ↑ |
| Expected convergence | Poor (60% warmup) | Good | Much better |

### Expected Loss Trajectory

v1 (problematic):
```
Step 20:  2.7351 (epoch 1, barely moving)
Step 40:  2.7342 (epoch 2, no change)
Step 60:  2.1617 (epoch 3, warmup done)
Step 80:  2.3373 (went UP!)
```

v2 (expected):
```
Step 10:  2.6 (warmup, slow improvement)
Step 30:  2.2 (warmup done, rapid drop)
Step 60:  1.8 (steady improvement)
Step 100: 1.5 (converging)
Step 200: 1.2 (near convergence)
Step 290: 1.1 (final)
```

---

## 6. Risks and Mitigations

### Risk 1: 8 memory tokens too small

**Concern**: Can 8 memory tokens capture enough information from 12K tokens?

**Mitigation**:
- Start with 8 tokens (target: 468K params)
- If performance is poor, try 16 tokens (target: 922K params)
- Benchmark against v1 with 64 tokens

### Risk 2: Bottleneck 32 too aggressive

**Concern**: 32 → 4096 projection may lose too much information.

**Mitigation**:
- Use residual connection (mean + MLP(mean))
- Layer normalization for stability
- If unstable, try bottleneck 64

### Risk 3: Data quality issues

**Concern**: Mixed data may cause domain confusion.

**Mitigation**:
- Start with pure Wikitext (same as v1) for baseline
- Gradually add Wikipedia, then conversation
- Monitor loss by data type

### Risk 4: RMT LR too small

**Concern**: 5e-5 may be too conservative.

**Mitigation**:
- Start with 5e-5
- If RMT params don't converge, try 1e-4
- Monitor RMT module gradient norms

---

## 7. Validation Plan

### Metrics to Track

1. **Training loss**: Should decrease steadily after warmup
2. **Perplexity**: Compare baseline (no RMT) vs RMT
3. **Memory utilization**: Are memory tokens actually being used?
4. **Cross-lingual performance**: If mixed data works well

### Ablation Studies

1. **num_memory_tokens**: 8 vs 16 vs 32
2. **bottleneck_dim**: 32 vs 64 vs 128
3. **Data mix**: Wikitext-only vs Wikipedia-only vs mixed
4. **Warmup**: 10% vs 15% vs 20% of training

### Success Criteria

- Total RMT params < 1M ✓
- Loss decreases monotonically after warmup
- Final loss < 1.5 (vs v1's 2.16-2.34)
- Training completes in <1 hour on 8 GPUs
- No loss instability in final steps

---

## 8. Next Steps

1. **Implement MemoryExtractorV2** in `rmt_module.py`
2. **Update train_rmt.py** with new defaults and data mixing
3. **Prepare mixed dataset**:
   - Download Chinese Wikipedia
   - Download English Wikipedia
   - Prepare conversation dataset
4. **Run baseline experiment** (v2 with Wikitext-only)
5. **Compare v1 vs v2** on same data
6. **Ablation study** on num_memory_tokens and bottleneck_dim
7. **Full training** with mixed data

---

## Appendix A: Param Count Calculation

### v1 MemoryExtractor

```
query:           1 × 64 × 4096 = 262,144
key_proj:        4096 × 4096 = 16,777,216
value_proj:      4096 × 4096 = 16,777,216
out_proj:        4096 × 4096 = 16,777,216
gate.0:          8192 × 4096 = 33,554,432
gate.2:          4096 × 1 = 4,096
norm:            2 × 4096 = 8,192
TOTAL:           84,160,512
```

### v2 MemoryExtractorV2 (8 tokens, bottleneck 32)

```
mlp.0:           4096 × 32 = 131,072
mlp.2:           32 × 32 = 1,024
mlp.4:           32 × 4096 = 131,072
query_coeffs:    8 × 1 = 8
gate_bias:       8
norm:            2 × 4096 = 8,192
TOTAL:           271,376
```

### Full RMTMemory v2 (8 tokens)

```
memory_embeddings:  6 × 8 × 4096 = 196,608
extractor:          271,376
segment_bias:       6 × 8 = 48
TOTAL:              468,032
```

---

## Appendix B: Training Command Examples

### v1 (for reference)

```bash
python scripts/train_rmt.py \
    --model_path /path/to/Qwen3-8B \
    --data_path data/wikitext103_train.jsonl \
    --output_dir outputs/rmt_v1 \
    --num_memory_tokens 64 \
    --num_memory_heads 8 \
    --segment_length 2048 \
    --max_segments 6 \
    --num_epochs 3 \
    --lr 1e-5 \
    --rmt_lr 1e-4 \
    --warmup_steps 100 \
    --batch_size 1 \
    --grad_accumulation_steps 8
```

### v2 (recommended)

```bash
python scripts/train_rmt.py \
    --model_path /path/to/Qwen3-8B \
    --data_path data/mixed \
    --output_dir outputs/rmt_v2 \
    --num_memory_tokens 8 \
    --segment_length 2048 \
    --max_segments 6 \
    --num_epochs 10 \
    --lr 1e-5 \
    --rmt_lr 5e-5 \
    --warmup_steps 30 \
    --batch_size 1 \
    --grad_accumulation_steps 8
```

---

**Document Version**: 1.0
**Date**: 2025-01-XX
**Author**: Research Agent, Mixture-of-Memory Project
