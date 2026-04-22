# RMT Inference Debug — 2026-04-18

## Root Cause

**The generation phase in eval completely drops the RMT attention mask and position IDs.**

During training, `_forward_single_segment()` passes:
- A custom attention mask where memory tokens have **full bidirectional attention** (attend to each other + all segment tokens)
- Custom position_ids (memory gets 0..N-1, segment gets global positions)

During inference generation (`_run_single_nih_test`, line ~the generation loop), the code calls:
```python
outputs = self.model.model(inputs_embeds=inputs_embeds, output_hidden_states=False)
```
**No `attention_mask`. No `position_ids`.** This means:
1. Qwen3's default causal mask is used — memory tokens can only attend causally (position 0 sees only itself, position 1 sees 0+1, etc.)
2. Memory tokens get default position_ids (0, 1, 2, ...) instead of the global positions they had during training
3. The memory token representations are computed completely differently from training

The memory tokens' hidden states during generation are garbage compared to what the model learned to use during training. The question tokens *can* attend to memory (since memory is at earlier positions in causal ordering), but the memory representations themselves are wrong.

## Specific Code Locations

### Bug 1: Missing attention mask during generation
**File**: `scripts/eval_rmt.py`, `_run_single_nih_test()`, the generation loop (~line 260):
```python
outputs = self.model.model(
    inputs_embeds=inputs_embeds,
    output_hidden_states=False,
)
```
Should include `attention_mask` with bidirectional memory attention, same as training.

### Bug 2: Missing position_ids during generation
Same location — no `position_ids` argument. Memory tokens need to know their "global" position for RoPE to work correctly.

### Bug 3: Inefficient token-by-token full recompute (minor, performance only)
Each generation step re-runs the full model on the entire growing sequence instead of using KV cache. This is correct but ~100x slower than necessary.

## Recommended Fix

Add a `generate_with_memory()` method to `RMTModel` (or fix the eval script) that:

1. **Builds the correct attention mask** for the [memory + question] sequence — memory tokens get full bidirectional attention, question tokens get causal attention to memory + preceding question tokens (same as `build_rmt_attention_mask`)
2. **Passes correct position_ids** — memory tokens at 0..num_memory-1, question tokens starting from the last segment's global position (so RoPE is consistent with training)
3. **Uses KV cache** for efficiency (optional but strongly recommended for speed)

Example fix for `RMTModel`:

```python
def generate_with_memory(self, question_ids, memory, segment_idx, max_new_tokens=20, device=None):
    """Generate answer tokens with memory injection."""
    B = question_ids.shape[0]
    device = question_ids.device
    
    # Build inputs with memory
    inputs_embeds = self._embed_with_memory(question_ids, memory)
    q_len = question_ids.shape[1]
    total_len = self.num_memory_tokens + q_len
    
    # Build attention mask (same as training)
    attn_mask = build_rmt_attention_mask(q_len, self.num_memory_tokens, device)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total, total]
    bool_mask = attn_mask.expand(B, -1, -1, -1).bool()
    attn_mask_4d = torch.zeros(B, 1, total_len, total_len, device=device)
    attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask, float('-inf'))
    
    # Build position_ids — memory at 0..N-1, question continues from last segment
    mem_pos = torch.arange(self.num_memory_tokens, device=device)
    q_pos = torch.arange(q_len, device=device) + segment_idx * self.segment_length
    position_ids = torch.cat([mem_pos, q_pos]).unsqueeze(0).expand(B, -1)
    
    # Prefill
    outputs = self.model.model(
        inputs_embeds=inputs_embeds,
        attention_mask={"full_attention": attn_mask_4d},
        position_ids=position_ids,
        output_hidden_states=False,
        use_cache=True,
    )
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past_kv = outputs.past_key_values
    generated = [next_token]
    
    # Autoregressive decode (simplified — no memory mask needed for decode steps
    # since only the new token is generated, and KV cache handles attention)
    for _ in range(max_new_tokens - 1):
        if next_token.item() == self.tokenizer.eos_token_id:
            break
        token_embeds = self.model.get_input_embeddings()(next_token)
        next_pos = position_ids[:, -1:] + 1
        
        outputs = self.model.model(
            inputs_embeds=token_embeds,
            position_ids=next_pos,
            past_key_values=past_kv,
            use_cache=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past_kv = outputs.past_key_values
        generated.append(next_token)
    
    return torch.cat(generated, dim=1)
```

Then update `_run_single_nih_test` to use it:
```python
from the last segment index
generated_ids = self.model.generate_with_memory(question_ids, old_memory, seg_idx=num_segments-1, max_new_tokens=20)
answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## Validation

After fixing, re-run NIH eval on the same checkpoint:
```bash
python scripts/eval_rmt.py --checkpoint_dir <path> --data_path <path> --eval_type nih --nih_num_trials 5
```

Expected: accuracy should improve significantly from 0% (matching or approaching base model performance for short contexts, with potential degradation only at very long contexts where memory truly matters).

## Risks / Follow-ups

1. **Position ID continuity**: The question tokens' position_ids should continue from where the last segment ended. Verify this is correct.
2. **KV cache + custom attention mask**: During the autoregressive decode phase after prefill, the KV cache already contains memory+question tokens. New tokens can attend to the full cache. This should work correctly with standard causal attention on the cache side.
3. **`{"full_attention": ...}` format**: This bypasses Qwen3's `create_causal_mask` helper. During decode with KV cache, we should NOT use this format — let the model use its standard causal mask for the new token. The fix above handles this by not passing the custom mask during decode steps.
