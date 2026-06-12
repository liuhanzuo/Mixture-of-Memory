# RMT Official Repo Deep Analysis

**Date**: 2026-04-19
**Repo**: `booydar/recurrent-memory-transformer` (branch: `framework_accel`)
**Papers**: NeurIPS 2022, AAAI 2024
**Analyst**: main agent (researcher subagent timed out)

---

## 1. RMT Architecture Overview

The official RMT has **three distinct implementation strategies**:

### Strategy A: MemoryCell + RecurrentWrapper (decoder/LM models)
File: `modeling_rmt/language_modeling.py`

**Memory injection**: **Sandwich** — `[old_mem | segment | placeholder_mem]`
- Memory tokens are **prepended AND appended** to every segment
- `placeholder_mem` at the end is initialized from learned parameters
- After the forward pass, hidden states at the last K positions become the new memory

**Memory initialization**: `nn.Parameter(memory_weights)` initialized with `embedding.weight.std()` scale

**Segment iteration**: `RecurrentWrapper` loops over segments, passing `memory_state` between them

**Gradient management**: `manage_gradients(memory_state, seg_num)`:
- `k2` parameter controls BPTT depth
- If `seg_num + k2 > max_n_segments`, memory is detached
- Default: full gradient flow through all segments

**Loss**: Concatenated logits across all segments, standard shifted CE loss

### Strategy B: RMTBaseModel (encoder / encoder-decoder models)
File: `modeling_rmt/base.py`, `modeling_rmt/conditional_generation.py`

**Memory injection**: Memory tokens are **special tokens** appended to vocabulary
- `model.resize_token_embeddings(vocab_size + num_mem_tokens)`
- Memory positions are at the **beginning** of each segment (after [CLS] if present)
- Memory embeddings from `model.embeddings(mem_token_ids)` are placed at `memory_position`

**Memory update**: After each segment's forward pass:
```python
memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]
```
The hidden states at the memory positions after the full model run become the next segment's memory.

### Strategy C: Horizontal Memory (most sophisticated)
File: `rmt_utils/encoder/horizontal_memory.py`, `rmt_utils/encoder_decoder/horizontal_memory.py`

**Memory injection at EVERY layer**:
```python
if i in rmt_parent.memory_storage:
    hidden_states = torch.cat([layer_memory, hidden_states], dim=1)
    layer_attention_mask = torch.cat((memory_mask, attention_mask), dim=-1)
```

**Memory extraction at EVERY layer**:
- Each layer has a `memory_layer` (a **deep copy** of the actual transformer layer)
- After the normal layer forward, the memory_layer runs again on hidden states
- The updated hidden states at memory positions are extracted
- Memory is **detached** between layers (`.detach()`)

**Key insight**: Different layers maintain different memory representations. Memory at layer 0 captures low-level features, while memory at layer N captures high-level features.

### Strategy D: Memory Layers (simpler layer-wise approach)
File: `rmt_utils/encoder/memory_layers.py`, `rmt_utils/encoder_decoder/memory_layers.py`

- Each transformer layer has a **copy of itself** as a `memory_layer`
- After each layer's forward pass, the memory_layer runs on the hidden states to update memory
- Updated memory replaces hidden states at memory positions:
```python
memory = memory_layer_out[0][:, rmt_parent.memory_position]
hidden_states[:, rmt_parent.memory_position] = memory
```

---

## 2. Key Design Details

### Position Encoding
- **T5 (encoder-decoder)**: Uses T5's relative positional bias. The `positional_encoding.py` file shows they compute a special bias for memory positions (treating all memory positions as "position 0" equivalent).
- **Decoder (MemoryCell)**: No special position handling — just splits tensors and passes them through. Positions flow naturally.
- **Horizontal Memory**: Memory tokens are prepended to hidden states at each layer. The attention mask ensures they attend to content bidirectionally.

### Training Configuration (from `run_finetuning_lm_rmt.py`)
- `--input_size`: max input size per segment (e.g., 128, 256, 512, 2048)
- `--num_mem_tokens`: number of memory tokens
- `--max_n_segments`: max segments per sample
- `--vary_n_segments`: randomly choose segment count (1 to max) — **curriculum learning**
- `--bptt_depth`: max segments for gradient flow
- `--k2`: truncated BPTT depth
- `--sum_loss`: sum losses across segments (vs. averaging)
- `--segment_ordering`: regular, reversed, bidirectional, repeat_first
- `--memory_layers`: which layers get memory processing ("all" or specific indices)
- `--share_memory_layers`: share weights across memory layers
- `--reconstruction_loss_coef`: reconstruction loss weight
- `--freeze_model_weights`: train only memory layers
- `--use_lora`: LoRA support (r, alpha, dropout configurable)
- Data: wikitext or arxiv pile, standard LM pretraining

### Data Preparation
```python
block_size = args.input_size
if args.num_mem_tokens is not None:
    block_size -= 2 * args.num_mem_tokens  # sandwich approach
```
For collation: sequences are reversed, padded, then flipped back (right-aligned padding).

### `vary_n_segments` (Critical for training)
During training, the number of segments is **randomly varied** from 1 to `max_n_segments`. This is essentially curriculum learning — the model first learns with 1 segment (trivial), then gradually learns to use memory for longer contexts.

---

## 3. Comparison with Our Implementation

| Aspect | Official RMT | Our Implementation (v8/v5) |
|--------|-------------|--------------------------|
| **Memory injection** | Sandwich `[mem\|seg\|mem]` (decoder) or beginning (encoder) | Beginning only `[mem\|seg]` |
| **Memory extraction** | Read from hidden states at memory positions after full forward | Take last K hidden states (v8) or cross-attention (v5) |
| **Per-layer processing** | YES — each layer has memory_layer copy that processes/updates memory | NO — single extraction at the very end |
| **Learned memory init** | `nn.Parameter` init from `embedding.std()` | `nn.Parameter` init from `0.02` scale |
| **Memory during forward** | Memory tokens participate in ALL attention computations at EVERY layer | Memory only at input layer, no per-layer processing |
| **BPTT control** | Explicit `k2` parameter, gradient detaching per segment | Per-segment `loss.backward(retain_graph=False)` |
| **Segment variation** | `vary_n_segments`: random segment count for curriculum | Fixed segment count |
| **Loss design** | Sum or average over all segments, single backward | Per-segment backward |
| **Position handling** | Model's native encoding (relative for T5, standard for decoder) | Custom position IDs: mem gets 0..K-1, content gets K..N |
| **Memory gate** | None — pure hidden state passthrough | v8: sigmoid gate (saturates to ~1.0) |

---

## 4. Why Our Memory Becomes Noise — Root Cause Analysis

### Root Cause #1: No Per-Layer Memory Processing
**This is the most critical difference.**

Official RMT (Strategy C/D) processes memory at **every transformer layer**. Each layer has a copy of itself that refines memory. This means:
- Layer 0 memory: captures surface-level token patterns
- Layer 12 memory: captures semantic relationships
- Layer 24 memory: captures abstract task-relevant information

Our implementation extracts memory only from the **final layer's last K positions**. This means the memory is only "seen" at the input of the next segment and must be useful at ALL layers simultaneously — an impossible task. The model can't specialize what information to store per layer.

### Root Cause #2: No Sandwich (Read/Write) Structure
The official decoder approach puts memory at **both ends** of the segment:
```
[old_memory | segment_content | placeholder_for_new_memory]
```

This gives the model two clear roles:
- The prepended memory is **read** (attended to by content tokens)
- The appended placeholder is **written to** (the model learns to compress info there)

Our approach only prepends memory. The model has to both read AND write from the same positions, which is ambiguous.

### Root Cause #3: Position Encoding Mismatch
We use custom position IDs where memory gets positions 0..K-1 and content gets K..N. But:
- The model's Rotary Position Embedding (RoPE) was trained with content starting at position 0
- Memory tokens at positions 0..K-1 steal these positions from real content
- This disrupts the model's positional understanding

The official RMT either:
- Uses relative positional encoding (T5), which is position-agnostic
- Uses the decoder's natural position flow (MemoryCell approach)

### Root Cause #4: No Curriculum Learning (vary_n_segments)
Official RMT randomly varies segment count during training (1 to max). This means:
- 1-segment samples: memory is irrelevant, model learns standard LM
- 2-segment samples: memory starts to matter
- N-segment samples: memory is critical

Our training always uses the same segment count. The model never gets "easy" examples where it can first learn standard LM, then gradually learn to use memory.

### Root Cause #5: Per-Segment Backward Breaks Gradient Flow
We do `loss.backward(retain_graph=False)` after each segment. This means:
- Segment N's loss doesn't flow back through segment N-1
- The memory has no gradient signal to learn WHAT to preserve
- Memory gradients only come from the next segment's loss, not from all future segments

Official RMT (with default BPTT settings) flows gradients through multiple segments, giving the memory a clear learning signal.

### Root Cause #6: Memory Gate Saturation
Our v8 gate is initialized to `sigmoid(2.0) ≈ 0.88`. Combined with no gradient pressure to actually change it, it saturates toward 1.0 (as observed: alpha → 0.999). The gate becomes a no-op.

---

## 5. Recommended Next Steps

### Option A: Implement Official RMT Architecture (High Priority)
Reimplement following the official code closely:
1. **Sandwich memory**: `[old_mem | segment | placeholder_mem]`
2. **Per-layer memory processing** (Strategy C — horizontal memory) if feasible, or at least Strategy A (MemoryCell sandwich)
3. **Fix position encoding**: Either use RoPE correctly (no offset) or disable RoPE for memory positions
4. **Curriculum learning**: `vary_n_segments=True` during training
5. **Full BPTT**: accumulate losses, single backward (or controlled via k2)

### Option B: Minimal Fix (Quick Validation)
Keep our v8 architecture but fix the most impactful issues:
1. **Sandwich**: prepend AND append memory positions
2. **vary_n_segments**: randomize segment count during training
3. **Fix position IDs**: let memory positions share position 0 or use no position encoding
4. **Accumulate losses**: don't backward per-segment, accumulate and backward once
5. **Fix memory init scale**: use `embedding.std()` instead of `0.02`

### Option C: Direct Fork of Official Code
Clone `booydar/recurrent-memory-transformer`, adapt for Qwen3-8B:
- The MemoryCell approach in `language_modeling.py` is the simplest to adapt
- Qwen3 is a causal decoder, so the decoder approach applies directly
- Just need to handle Qwen3's RoPE properly

---

## 6. Key Code Snippets for Reference

### MemoryCell (decoder LM) — the simplest approach to adapt:
```python
class MemoryCell(nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        embeddings = self.model.get_input_embeddings()
        memory_dim = self.model.config.hidden_size
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', nn.Parameter(memory_weights, requires_grad=True))
        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def process_input(self, input_ids, memory_state, **kwargs):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # SANDWICH: [memory | segment | memory]
        inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
        # ... attention mask padding ...
        return seg_kwargs

    def process_output(self, model_outputs, **kwargs):
        # NEW memory = hidden states at the APPENDED positions
        memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
        # LOGITS = only for segment tokens (strip both memory ends)
        out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]
        return out, memory_state
```

### RecurrentWrapper gradient management:
```python
def manage_gradients(self, memory_state, seg_num):
    k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
    if seg_num == 0 or k2 in {-1, None} or seg_num + k2 > max_n_segments:
        return True
    memory_state = memory_state.detach()  # Stop gradient for middle segments
    return False
```
