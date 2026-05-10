#!/usr/bin/env python3
"""
Debug script for RMT v10 NiH eval pipeline.

Diagnoses why all RMT v10 variants get 0% NiH accuracy while base model gets 73%.

Usage:
  python scripts/debug_eval_rmt_v10.py \
    --base_model models/Qwen--Qwen3-8b \
    --checkpoint outputs/rmt_v10_20260419_182044  # or any v10 checkpoint dir
"""

import os, sys, json, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_v10 import RMTv10Config, RMTv10Model, RMTv10Memory


def diagnose_checkpoint(checkpoint_dir):
    """Check if rmt_memory.pt keys match RMTv10Model expected keys."""
    rmt_path = os.path.join(checkpoint_dir, "rmt_memory.pt")
    if not os.path.exists(rmt_path):
        rmt_path = os.path.join(checkpoint_dir, "final", "rmt_memory.pt")
    if not os.path.exists(rmt_path):
        print(f"[BUG] No rmt_memory.pt found in {checkpoint_dir}")
        return False

    state = torch.load(rmt_path, map_location="cpu", weights_only=False)
    keys = list(state.keys())
    print(f"\n=== rmt_memory.pt keys ({len(keys)} total) ===")
    for k, v in state.items():
        print(f"  {k}: {v.shape}")

    # Expected v10 keys
    expected_prefixes = ["l0.", "recon_head."]
    v10_keys = [k for k in keys if any(k.startswith(p) for p in expected_prefixes)]
    legacy_keys = [k for k in keys if not any(k.startswith(p) for p in expected_prefixes)]

    print(f"\n  v10-matching keys: {len(v10_keys)}")
    print(f"  Legacy/mismatched keys: {len(legacy_keys)}")

    if legacy_keys and not v10_keys:
        print(f"\n[BUG FOUND] All keys are from an older RMT version (v5/v6/v8).")
        print(f"  load_state_dict(strict=False) will load NOTHING.")
        print(f"  The model uses randomly-initialized memory → garbage output → 0% accuracy.")
        print(f"\n  Fix: retrain with v10 code, or use a checkpoint trained with rmt_v10.py")
        return False

    if v10_keys and not legacy_keys:
        print(f"\n[OK] All keys match v10 architecture.")
        return True

    if v10_keys and legacy_keys:
        print(f"\n[WARN] Mixed keys — some v10, some legacy. Check carefully.")
        return True

    print(f"\n[WARN] No keys found?")
    return False


def diagnose_model_loading(base_model_path, checkpoint_dir, device):
    """Load model and check if memory weights were actually loaded."""
    print(f"\n=== Model Loading Diagnosis ===")

    # Build config
    config_path = os.path.join(checkpoint_dir, "rmt_config.json")
    if not os.path.exists(config_path):
        print(f"[SKIP] No rmt_config.json in {checkpoint_dir}")
        return

    with open(config_path) as f:
        rmt_cfg = json.load(f)

    v10_cfg = RMTv10Config()
    for field in ["num_mem_tokens", "segment_length", "max_n_segments"]:
        cfg_key = field.replace("num_", "").replace("_tokens", "").replace("n_", "")
        if field in rmt_cfg:
            setattr(v10_cfg, field, rmt_cfg[field])
        elif cfg_key in rmt_cfg:
            setattr(v10_cfg, field, rmt_cfg[cfg_key])
    v10_cfg.max_n_segments = rmt_cfg.get("max_segments", v10_cfg.max_n_segments)

    weight_dir = checkpoint_dir
    if not os.path.exists(os.path.join(weight_dir, "model.safetensors")):
        final = os.path.join(weight_dir, "final")
        if os.path.exists(os.path.join(final, "model.safetensors")):
            weight_dir = final

    print(f"  Loading base model from: {weight_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        weight_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    print(f"  Wrapping with RMTv10Model...")
    rmt_model = RMTv10Memory(v10_cfg).wrap(model)

    # Print expected state_dict keys
    print(f"\n  RMTv10Model state_dict keys:")
    for k, v in rmt_model.state_dict().items():
        print(f"    {k}: {v.shape}")

    # Load memory
    rmt_path = os.path.join(weight_dir, "rmt_memory.pt")
    if os.path.exists(rmt_path):
        state = torch.load(rmt_path, map_location=device, weights_only=False)
        result = rmt_model.load_state_dict(state, strict=False)
        print(f"\n  load_state_dict result:")
        print(f"    missing_keys: {result.missing_keys}")
        print(f"    unexpected_keys: {result.unexpected_keys}")

        if result.missing_keys:
            print(f"\n  [BUG] Missing keys were NOT loaded — these use random init:")
            for k in result.missing_keys:
                if "l0.memory" in k:
                    print(f"    *** {k}: L0 memory is randomly initialized! ***")
    else:
        print(f"  [BUG] No rmt_memory.pt found at {rmt_path}")

    return rmt_model


def diagnose_forward_path(rmt_model, tokenizer, device):
    """Trace the forward path with a small example."""
    print(f"\n=== Forward Path Diagnosis ===")

    # Simple test: encode a short text, segment it, run through RMT
    test_text = "The secret code is ABC123. This is some filler."
    input_ids = tokenizer.encode(test_text, add_special_tokens=False, return_tensors="pt").to(device)
    B, L = input_ids.shape
    print(f"  Input: {L} tokens")

    cfg = rmt_model.config
    seg_len = cfg.segment_length
    K = cfg.num_mem_tokens
    num_segments = max(1, min(cfg.max_n_segments, L // seg_len))
    print(f"  num_segments: {num_segments}, segment_length: {seg_len}, num_mem_tokens: {K}")

    # Pad to segment_length
    pad_len = seg_len - L
    if pad_len > 0:
        input_ids = torch.cat([input_ids, torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device)], dim=1)
        L = input_ids.shape[1]
    print(f"  Padded input: {L} tokens")

    base_model = rmt_model.base_model
    memory_state = rmt_model.l0.get_initial_memory(B).to(dtype=torch.bfloat16)
    print(f"  Initial memory shape: {memory_state.shape}")
    print(f"  Initial memory norm: {memory_state.norm(dim=-1).mean():.4f}")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        segments = rmt_model._segment_input(input_ids, num_segments)
        for seg_idx in range(num_segments):
            seg_ids = segments[seg_idx].to(device)
            content_embeds = base_model.get_input_embeddings()(seg_ids)
            print(f"\n  Segment {seg_idx}: content_embeds shape = {content_embeds.shape}")

            inputs_embeds, attn_mask_2d = rmt_model.l0.build_sandwich_fast(content_embeds, memory_state)
            print(f"  Sandwich embeds shape: {inputs_embeds.shape} (expected: [{B}, {2*K+seg_len}, D])")
            print(f"  Attn mask shape: {attn_mask_2d.shape}")
            print(f"  Memory prepended: norm = {inputs_embeds[:, :K, :].norm(dim=-1).mean():.4f}")

            # Check mask correctness
            # old_mem should see everything
            old_mem_mask = attn_mask_2d[0, :K, :]
            print(f"  old_mem mask: can see content+placeholder = {old_mem_mask[:, K:].all().item()}")

            attn_mask_4d = rmt_model._make_4d_attn_mask(attn_mask_2d, torch.bfloat16)
            total_len = inputs_embeds.shape[1]
            position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

            outputs = base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                output_hidden_states=True,
            )

            new_memory = rmt_model.l0.extract_new_memory(outputs.hidden_states[-1])
            print(f"  New memory norm: {new_memory.norm(dim=-1).mean():.4f}")
            memory_state, importance = rmt_model.l0.apply_importance_routing(memory_state, new_memory)
            print(f"  Updated memory norm: {memory_state.norm(dim=-1).mean():.4f}")

    # Test generation with memory
    print(f"\n=== Generation Diagnosis ===")
    question = "What is the secret code mentioned in the document?"
    q_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(device)
    q_embeds = base_model.get_input_embeddings()(q_ids)
    gen_embeds = torch.cat([memory_state, q_embeds], dim=1)
    print(f"  Generation input: memory({K}) + question({q_ids.shape[1]}) = {gen_embeds.shape[1]} tokens")

    generated = []
    for step in range(10):
        cur_len = gen_embeds.shape[1]
        causal = torch.tril(torch.ones(cur_len, cur_len, device=device, dtype=torch.bool))
        attn = torch.zeros(1, 1, cur_len, cur_len, device=device, dtype=torch.bfloat16)
        attn.masked_fill_(~causal, float('-inf'))
        pos = torch.arange(cur_len, device=device).unsqueeze(0).expand(B, -1)

        out = base_model.model(inputs_embeds=gen_embeds, attention_mask=attn, position_ids=pos)
        logits = base_model.lm_head(out.last_hidden_state[:, -1:, :])
        tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if tok.item() == tokenizer.eos_token_id:
            break
        generated.append(tok.item())
        gen_embeds = torch.cat([gen_embeds, base_model.get_input_embeddings()(tok)], dim=1)

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  Generated answer: '{answer}'")
    print(f"  Expected substring: 'ABC123'")
    print(f"  Match: {'ABC123' in answer}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--skip_model", action="store_true", help="Only check memory file keys, skip model loading")
    args = parser.parse_args()

    # Step 1: Check memory file keys (no GPU needed)
    weight_dir = args.checkpoint
    if os.path.exists(os.path.join(weight_dir, "final", "rmt_memory.pt")):
        weight_dir = os.path.join(weight_dir, "final")
    diagnose_checkpoint(weight_dir)

    if args.skip_model:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("\n[WARN] No GPU available, skipping model loading tests.")

    # Step 2: Model loading
    rmt_model = diagnose_model_loading(args.base_model, args.checkpoint, device)

    # Step 3: Forward path
    tok_dir = args.checkpoint
    if os.path.exists(os.path.join(tok_dir, "tokenizer_config.json")):
        pass
    elif os.path.exists(os.path.join(tok_dir, "final", "tokenizer_config.json")):
        tok_dir = os.path.join(tok_dir, "final")
    else:
        tok_dir = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if rmt_model is not None:
        diagnose_forward_path(rmt_model, tokenizer, device)


if __name__ == "__main__":
    main()
