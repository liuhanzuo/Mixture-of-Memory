#!/usr/bin/env python3
"""
Diagnose the MAG smoke test loss divergence (0.24 → 2.65).

The smoke test uses raw_kv injection on Qwen3-8B with synthetic data.
Alpha starts at 0.01 and climbs to 0.299 (max_alpha=0.3).
Loss goes from 0.24 up to 2.65 monotonically.

This script isolates the cause by comparing:
  1. Baseline loss (no injection, same data/model)
  2. Loss with injection but alpha=0 (should equal baseline)
  3. Loss with injection at various alpha values
  4. Whether the loss increase comes from LM head or auxiliary losses
"""

import sys, os, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0"
print(f"Using device: {device}")


def make_synthetic_batch(tokenizer, seq_len=512, batch_size=2):
    """Create synthetic input IDs (random tokens)."""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    return input_ids, attention_mask


def build_causal_mask(seq_len, device):
    """Standard causal mask in HF format: 0 for allowed, -inf for blocked."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]


def test_baseline_loss(model, tokenizer):
    """Step 1: Get baseline loss (no injection)."""
    print("\n=== Step 1: Baseline Loss (no injection) ===")
    model.eval()
    input_ids, attention_mask = make_synthetic_batch(tokenizer, seq_len=512, batch_size=4)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
        
        # Shift for causal LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        # Perplexity
        ppl = math.exp(loss.item())
    
    print(f"  Baseline cross-entropy loss: {loss.item():.4f}")
    print(f"  Baseline perplexity:         {ppl:.2f}")
    print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"  Logits mean/std: {logits.mean().item():.2f} / {logits.std().item():.2f}")
    
    return loss.item()


def test_alpha_sensitivity(model, tokenizer, kv_injector_class, base_config):
    """Step 2: Test loss at different alpha values."""
    print("\n=== Step 2: Alpha Sensitivity Test ===")
    
    # Re-create model fresh for each test
    from memory.mag.kv_memory_injector import RawKVInjector, KVMemoryInjectorConfig
    
    results = []
    
    # Create fake memory KV (random, same structure as real memory)
    input_ids, attention_mask = make_synthetic_batch(tokenizer, seq_len=512, batch_size=4)
    B, T = input_ids.shape
    
    # Simulate memory text: 128 tokens
    num_mem_tokens = 128
    mem_input_ids = torch.randint(0, tokenizer.vocab_size, (B, num_mem_tokens), device=device)
    mem_mask = torch.ones(B, num_mem_tokens, device=device)
    
    with torch.no_grad():
        model.eval()
        
        # Forward memory text through model to get per-layer hidden states
        mem_outputs = model(
            input_ids=mem_input_ids,
            attention_mask=mem_mask,
            output_hidden_states=True,
        )
        hidden_states_list = mem_outputs.hidden_states  # [0]=embed, [1..36]=layer outputs
        
        # Build virtual KV cache from memory hidden states (raw_kv style)
        config = model.config
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        num_layers = config.num_hidden_layers
        
        # Get injection layers (every 9th: 0, 9, 18, 27)
        injection_layers = list(range(0, num_layers, 9))
        
        virtual_kv_cache = {}
        for layer_idx in injection_layers:
            hs = hidden_states_list[layer_idx + 1]  # +1 because index 0 is embedding
            
            attn = model.model.layers[layer_idx].self_attn
            
            # Compute K, V projections
            k = attn.k_proj(hs)
            v = attn.v_proj(hs)
            
            if hasattr(attn, 'k_norm'):
                k = attn.k_norm(k.view(B, num_mem_tokens, num_kv_heads, head_dim)).view(B, num_mem_tokens, -1)
            
            k_heads = k.view(B, num_mem_tokens, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            v_heads = v.view(B, num_mem_tokens, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            
            virtual_kv_cache[layer_idx] = (k_heads, v_heads)
        
        # Now test with different alpha settings
        causal_mask = build_causal_mask(T, device)
        
        for alpha_setting in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            # Create injector with specific alpha
            test_config = KVMemoryInjectorConfig(
                hidden_dim=config.hidden_size,
                num_layers=num_layers,
                injection_layers=injection_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                init_alpha=alpha_setting if alpha_setting > 0 else 0,
                max_alpha=max(alpha_setting, 0.3) if alpha_setting > 0 else 0.3,
                injection_mode="raw_kv",
            )
            
            injector = RawKVInjector(test_config)
            
            # Forward through model with injection
            layer_outputs = None
            hidden = model.model.embed_tokens(input_ids)
            
            # Get position embeddings for Qwen3
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RopeConfig
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            
            for layer_idx in range(num_layers):
                decoder_layer = model.model.layers[layer_idx]
                
                if layer_idx in injection_layers:
                    # Get position embeddings for this layer
                    # Qwen3 uses rotary embeddings
                    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
                    rotary_emb = decoder_layer.self_attn.rotary_emb
                    cos, sin = rotary_emb(hidden, position_ids)
                    position_embeddings = (cos, sin)
                    
                    h, _ = injector.forward_decoder_layer(
                        layer_idx=layer_idx,
                        decoder_layer=decoder_layer,
                        hidden_states=hidden,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        virtual_kv_cache=virtual_kv_cache,
                        inference_scale=1.0,
                    )
                else:
                    # Normal forward
                    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                    
                    residual = hidden
                    hidden_normed = decoder_layer.input_layernorm(hidden)
                    
                    attn_module = decoder_layer.self_attn
                    head_dim_local = attn_module.head_dim
                    hidden_shape = (B, T, -1, head_dim_local)
                    
                    q = attn_module.q_norm(attn_module.q_proj(hidden_normed).view(hidden_shape)).transpose(1, 2)
                    k = attn_module.k_norm(attn_module.k_proj(hidden_normed).view(hidden_shape)).transpose(1, 2)
                    v = attn_module.v_proj(hidden_normed).view(hidden_shape).transpose(1, 2)
                    
                    rotary_emb = attn_module.rotary_emb
                    cos, sin = rotary_emb(hidden_normed, position_ids)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    
                    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
                    num_kv_groups = attn_module.num_key_value_groups
                    k = repeat_kv(k, num_kv_groups)
                    v = repeat_kv(v, num_kv_groups)
                    
                    attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn_module.scaling
                    attn_weights = attn_weights + causal_mask[:, :, :, :attn_weights.shape[-2]]
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                    attn_output = torch.matmul(attn_weights, v)
                    
                    attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, -1)
                    attn_output = attn_module.o_proj(attn_output)
                    
                    hidden = residual + attn_output
                    
                    residual = hidden
                    hidden = decoder_layer.post_attention_layernorm(hidden)
                    hidden = decoder_layer.mlp(hidden)
                    hidden = residual + hidden
            
            # Final norm + LM head
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            actual_alpha = injector.get_alpha(injection_layers[0]).item() if alpha_setting > 0 else 0.0
            ppl = math.exp(min(loss.item(), 20))
            print(f"  alpha_setting={alpha_setting:.2f}  actual_alpha={actual_alpha:.4f}  loss={loss.item():.4f}  ppl={ppl:.2f}")
            results.append({
                "alpha_setting": alpha_setting,
                "actual_alpha": actual_alpha,
                "loss": loss.item(),
                "ppl": ppl,
            })
    
    return results


def analyze_results(baseline_loss, alpha_results):
    """Analyze whether loss divergence is expected or a bug."""
    print("\n=== Analysis ===")
    print(f"  Baseline loss (no injection): {baseline_loss:.4f}")
    
    loss_at_init_alpha = alpha_results[1]["loss"] if len(alpha_results) > 1 else None
    loss_at_max_alpha = alpha_results[-1]["loss"] if alpha_results else None
    
    if loss_at_init_alpha:
        delta = loss_at_init_alpha - baseline_loss
        print(f"  Loss at init alpha (0.01):   {loss_at_init_alpha:.4f} (delta: +{delta:.4f})")
    
    if loss_at_max_alpha:
        delta = loss_at_max_alpha - baseline_loss
        print(f"  Loss at max alpha (0.3):    {loss_at_max_alpha:.4f} (delta: +{delta:.4f})")
    
    # Check: is the loss increase proportional to alpha?
    print("\n  Loss increase vs alpha:")
    for r in alpha_results:
        delta = r["loss"] - baseline_loss
        ratio = delta / max(r["actual_alpha"], 1e-8) if r["actual_alpha"] > 0 else 0
        print(f"    alpha={r['actual_alpha']:.4f}  loss_delta={delta:.4f}  delta/alpha={ratio:.2f}")


def main():
    print("=" * 60)
    print("MAG Smoke Test Loss Divergence Diagnosis")
    print("=" * 60)
    
    # Load model
    model_path = os.environ.get(
        "BASE_MODEL",
        "/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b"
    )
    
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    # Step 1: Baseline
    baseline_loss = test_baseline_loss(model, tokenizer)
    
    # Step 2: Alpha sensitivity (only if GPUs have enough memory)
    free_mem = torch.cuda.mem_get_info(device)[0] / 1e9
    print(f"\n  Free GPU memory: {free_mem:.1f} GB")
    
    if free_mem < 10:
        print("  WARNING: Low GPU memory, skipping alpha sensitivity test")
        print("  The baseline comparison alone is sufficient to diagnose")
        print(f"\n=== CONCLUSION ===")
        print(f"  Baseline loss on random tokens: {baseline_loss:.4f}")
        print(f"  Smoke test started at loss=0.24 (BELOW baseline!)")
        print(f"  This suggests the smoke test loss metric is NOT standard cross-entropy,")
        print(f"  or uses a different normalization/composition.")
        print(f"  Check the MAG training script's loss computation.")
        return
    
    # Run full analysis
    from memory.mag.kv_memory_injector import KVMemoryInjectorConfig
    alpha_results = test_alpha_sensitivity(model, tokenizer, None, None)
    analyze_results(baseline_loss, alpha_results)


if __name__ == "__main__":
    main()
