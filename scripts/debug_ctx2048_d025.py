#!/usr/bin/env python3
"""
Debug script for ctx2048_d0.25 0% NIH accuracy issue.

Hypothesis testing:
1. Memory overwriting: segment N overwrites needle info from segment N-1
2. Segment boundary artifact: needle falls exactly at segment boundary
3. Base model prior: model answers from pre-training knowledge, not memory
4. Shape mismatch: initial_slots shape incompatibility between training and eval

This script traces through the full eval pipeline for both failing and passing cases.
"""
import os, sys, json, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.slot.slot_memory_compressor import SlotMemoryCompressor, SlotMemoryWrapper

# Replicate eval logic
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_slot_memory import generate_nih_sample

CHECKPOINT_DIR = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/slot_memory_8gpu_20260420_105232_stage1_20260420_105312/final"
BASE_MODEL = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b"
DEVICE = "cuda:0"
SEG_LEN = 1024
NUM_SLOTS = 16
SLOT_DIM = 256

def main():
    device = torch.device(DEVICE)
    
    print("=" * 80)
    print("DIAGNOSTIC: ctx2048_d0.25 0% NIH accuracy")
    print("=" * 80)
    
    # 1. Load model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": device},
    )
    
    # Try loading with max_segments=6 (matching training) and max_segments=4 (eval default)
    for max_seg in [6, 4]:
        print(f"\n[2] Testing compressor with max_segments={max_seg} (num_segments={max_seg+1})...")
        compressor = SlotMemoryCompressor(
            hidden_dim=4096, num_slots=NUM_SLOTS, slot_dim=SLOT_DIM,
            num_iterations=3, dropout=0.1, num_segments=max_seg + 1,
        )
        slot_weights_path = os.path.join(CHECKPOINT_DIR, "slot_weights.pt")
        state = torch.load(slot_weights_path, map_location=device, weights_only=True)
        
        # Check shape compatibility
        saved_initial = state['initial_slots'].shape  # Should be [7, 16, 256]
        model_initial = compressor.initial_slots.shape
        print(f"  Saved initial_slots shape: {saved_initial}")
        print(f"  Model initial_slots shape: {model_initial}")
        
        if saved_initial != model_initial:
            print(f"  *** SHAPE MISMATCH! This would cause load_state_dict to fail ***")
            # Try with strict=False
            try:
                result = compressor.load_state_dict(state, strict=False)
                print(f"  load_state_dict(strict=False): missing={result.missing_keys}, unexpected={result.unexpected_keys}")
            except Exception as e:
                print(f"  load_state_dict error: {e}")
        else:
            try:
                compressor.load_state_dict(state)
                print(f"  load_state_dict: OK")
            except Exception as e:
                print(f"  load_state_dict error: {e}")
    
    # Use max_segments=6 to match training
    compressor = SlotMemoryCompressor(
        hidden_dim=4096, num_slots=NUM_SLOTS, slot_dim=SLOT_DIM,
        num_iterations=3, dropout=0.1, num_segments=7,  # max_segments=6 → num_segments=7
    )
    state = torch.load(os.path.join(CHECKPOINT_DIR, "slot_weights.pt"), map_location=device, weights_only=True)
    compressor.load_state_dict(state)
    compressor.to(device=device, dtype=torch.bfloat16)
    
    slot_model = SlotMemoryWrapper(model=base_model, compressor=compressor, segment_length=SEG_LEN)
    slot_model = slot_model.to(device=device, dtype=torch.bfloat16)
    slot_model.eval()
    
    # 3. Generate test cases
    needle_text = "the special secret code is Blue Monkey 42"
    answer_text = "Blue Monkey 42"
    haystack = "The history of computing spans several centuries. Early mechanical devices " * 50
    
    test_cases = [
        ("ctx2048_d0.0", 2048, 0.0),
        ("ctx2048_d0.25", 2048, 0.25),
        ("ctx2048_d0.5", 2048, 0.5),
        ("ctx4096_d0.0", 4096, 0.0),
        ("ctx4096_d0.25", 4096, 0.25),
    ]
    
    print("\n" + "=" * 80)
    print("[3] SEGMENT ANALYSIS")
    print("=" * 80)
    
    for label, ctx_len, depth in test_cases:
        ctx_ids, q_ids, answer = generate_nih_sample(tokenizer, ctx_len, depth, needle_text, answer_text, haystack)
        num_segs = ctx_ids.shape[0] // SEG_LEN
        
        # Calculate needle position
        needle_len = len(tokenizer.encode(needle_text))
        total_content = ctx_len - needle_len - 50
        insert_pos = int(depth * total_content)
        needle_seg = insert_pos // SEG_LEN
        
        print(f"\n--- {label} ---")
        print(f"  Context length: {ctx_len}, Segment length: {SEG_LEN}, Num segments: {num_segs}")
        print(f"  Needle position: {insert_pos} (in segment {needle_seg})")
        print(f"  Needle segment range: [{needle_seg * SEG_LEN}, {(needle_seg + 1) * SEG_LEN})")
        print(f"  Question tokens: {q_ids.shape[0]}")
        print(f"  Question: {repr(tokenizer.decode(q_ids, skip_special_tokens=True))}")
    
    # 4. Process with slot memory and check intermediate states
    print("\n" + "=" * 80)
    print("[4] MEMORY STATE ANALYSIS (with slot memory)")
    print("=" * 80)
    
    with torch.no_grad():
        for label, ctx_len, depth in test_cases:
            ctx_ids, q_ids, answer = generate_nih_sample(tokenizer, ctx_len, depth, needle_text, answer_text, haystack)
            ctx_ids = ctx_ids.unsqueeze(0).to(device)
            q_ids = q_ids.unsqueeze(0).to(device)
            B = 1
            num_segs = ctx_ids.shape[1] // SEG_LEN
            old_slots = None
            
            print(f"\n--- {label} ---")
            
            for seg_idx in range(num_segs):
                seg_ids = ctx_ids[:, seg_idx * SEG_LEN : (seg_idx + 1) * SEG_LEN]
                
                if old_slots is None:
                    slots = slot_model.compressor.get_initial_slots(seg_idx, B, device, torch.bfloat16)
                else:
                    slots = old_slots
                
                mem_tokens = slot_model.compressor.slots_to_memory_tokens(slots)
                print(f"  Segment {seg_idx}: mem_tokens norm={mem_tokens.norm().item():.4f}, mean={mem_tokens.mean().item():.6f}")
                
                _, _, seg_hidden = slot_model.forward_segment(seg_ids, None, mem_tokens, stage="ce_only")
                
                new_slots, _ = slot_model.compressor(seg_hidden, old_slots=old_slots, compute_recon=False)
                
                # Check slot similarity
                if old_slots is not None:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        new_slots.flatten().unsqueeze(0),
                        old_slots.flatten().unsqueeze(0)
                    ).item()
                    print(f"  Segment {seg_idx}: slots cosine_sim_with_prev={cos_sim:.4f}")
                
                old_slots = new_slots
            
            # Generate answer
            final_mem_tokens = slot_model.compressor.slots_to_memory_tokens(old_slots)
            print(f"  Final mem_tokens norm={final_mem_tokens.norm().item():.4f}")
            
            # Use the eval's generate function
            from eval_slot_memory import generate_with_slot_memory
            gen = generate_with_slot_memory(slot_model, tokenizer, ctx_ids, q_ids, max_new_tokens=30, device=DEVICE)
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            is_correct = answer.lower() in decoded.lower()
            print(f"  Generated: {repr(decoded[:100])}")
            print(f"  Correct: {is_correct}")
    
    # 5. Test with ZERO memory (no slots) — tests base model prior
    print("\n" + "=" * 80)
    print("[5] ZERO MEMORY TEST (base model prior)")
    print("=" * 80)
    
    with torch.no_grad():
        inner = slot_model.model
        backbone = inner.model if hasattr(inner, 'model') else inner.get_base_model().model
        lm_head = inner.lm_head if hasattr(inner, 'lm_head') else inner.get_base_model().lm_head
        
        question = "\n\nBased on the text above, what is the special secret code mentioned in the document?"
        q_ids = tokenizer.encode(question, add_special_tokens=False)
        q_tensor = torch.tensor([q_ids], dtype=torch.long, device=device)
        T = q_tensor.shape[1]
        
        # Test with zero memory
        zero_mem = torch.zeros(1, NUM_SLOTS, 4096, dtype=torch.bfloat16, device=device)
        token_embeds = inner.get_input_embeddings()(q_tensor) if hasattr(inner, 'get_input_embeddings') else backbone.get_input_embeddings()(q_tensor)
        
        # For merged model, use base model directly
        if hasattr(base_model, 'get_input_embeddings'):
            token_embeds = base_model.get_input_embeddings()(q_tensor)
        
        inputs_embeds = torch.cat([zero_mem, token_embeds], dim=1)
        
        K = NUM_SLOTS
        total = K + T
        causal = torch.tril(torch.ones(total, total, device=device)).bool()
        causal[:K, :] = True
        attn_mask_4d = torch.zeros(1, 1, total, total, dtype=torch.bfloat16, device=device)
        attn_mask_4d = attn_mask_4d.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        position_ids = torch.cat([torch.arange(K, device=device), torch.arange(T, device=device) + K]).unsqueeze(0)
        
        outputs = base_model.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask_4d, position_ids=position_ids, use_cache=True)
        logits = base_model.lm_head(outputs.last_hidden_state)
        past_kv = outputs.past_key_values
        
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]
        cur_pos = position_ids[:, -1:] + 1
        
        for step in range(29):
            token_embeds = base_model.get_input_embeddings()(next_token)
            total_so_far = K + T + step + 1
            step_mask = torch.zeros(1, 1, 1, total_so_far, dtype=torch.bfloat16, device=device)
            outputs = base_model.model(inputs_embeds=token_embeds, attention_mask=step_mask, position_ids=cur_pos, past_key_values=past_kv, use_cache=True)
            logits = base_model.lm_head(outputs.last_hidden_state)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_kv = outputs.past_key_values
            cur_pos = cur_pos + 1
            generated.append(next_token)
        
        gen_text = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True).strip()
        print(f"  With ZERO memory: {repr(gen_text[:100])}")
        print(f"  Contains 'Blue Monkey 42': {'blue monkey 42' in gen_text.lower()}")
        
        # Test with NO memory at all (just question, no prefix)
        outputs2 = base_model(q_tensor, use_cache=True)
        logits2 = outputs2.logits
        next_token2 = logits2[:, -1, :].argmax(dim=-1, keepdim=True)
        generated2 = [next_token2]
        past_kv2 = outputs2.past_key_values
        
        for step in range(29):
            outputs2 = base_model(next_token2, past_key_values=past_kv2, use_cache=True)
            logits2 = outputs2.logits
            next_token2 = logits2[:, -1, :].argmax(dim=-1, keepdim=True)
            past_kv2 = outputs2.past_key_values
            generated2.append(next_token2)
        
        gen_text2 = tokenizer.decode(torch.cat(generated2, dim=1)[0], skip_special_tokens=True).strip()
        print(f"  With NO memory (plain generate): {repr(gen_text2[:100])}")
        print(f"  Contains 'Blue Monkey 42': {'blue monkey 42' in gen_text2.lower()}")
    
    print("\n" + "=" * 80)
    print("[6] ANALYSIS")
    print("=" * 80)
    print("""
Key questions to answer:
1. Does the base model output "Blue Monkey 42" with zero/no memory?
   → If YES: the model has memorized the QA pair, slot memory is irrelevant
   → If NO: slot memory is needed, and its failure explains ctx2048_d0.0/d0.25

2. Does slot memory preserve needle info across segments?
   → Check cosine similarity between consecutive segments' slots
   → High similarity = slots not changing much = might preserve info
   → Low similarity = slots completely overwritten = info lost

3. Is the initial_slots shape compatible?
   → Training used max_segments=6 (initial_slots: [7,16,256])
   → Eval default uses max_segments=4 (initial_slots: [5,16,256])
   → SHAPE MISMATCH would cause load_state_dict to crash!

4. Why does ctx4096_d0.0 work but ctx2048_d0.0 doesn't?
   → If base model answers correctly with zero memory, this is explained
   → Otherwise, need to investigate further
""")

if __name__ == "__main__":
    main()
