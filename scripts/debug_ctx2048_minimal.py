#!/usr/bin/env python3
# Use venv python: /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/.venv/bin/python3
"""Minimal debug: trace slot memory for ctx2048 vs ctx4096 with actual model."""
import sys, os, json, torch, torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from src.memory.slot.slot_memory_compressor import SlotMemoryCompressor, SlotMemoryWrapper

CKPT = f"{PROJECT_ROOT}/outputs/slot_memory_8gpu_20260420_105232_stage1_20260420_105312/final"
DEVICE = "cuda:0"
SEG_LEN = 1024

def main():
    print("[1] Loading tokenizer...")
    # Use slow tokenizer from base model
    base_tok_path = f"{PROJECT_ROOT}/../models/Qwen--Qwen3-8b"
    tokenizer = AutoTokenizer.from_pretrained(base_tok_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("[2] Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
    )
    
    print("[3] Loading compressor (max_segments=6)...")
    compressor = SlotMemoryCompressor(
        hidden_dim=4096, num_slots=16, slot_dim=256,
        num_iterations=3, dropout=0.1, num_segments=7,
    )
    state = torch.load(f"{CKPT}/slot_weights.pt", map_location=DEVICE, weights_only=True)
    compressor.load_state_dict(state)
    compressor.to(device=DEVICE, dtype=torch.bfloat16)
    
    slot_model = SlotMemoryWrapper(model=base_model, compressor=compressor, segment_length=SEG_LEN)
    slot_model.to(device=DEVICE, dtype=torch.bfloat16)
    slot_model.eval()
    
    # Build NIH sample
    needle = "the special secret code is Blue Monkey 42"
    answer = "Blue Monkey 42"
    haystack = "The history of computing spans several centuries. Early mechanical devices " * 50
    
    test_configs = [
        ("ctx2048_d0.0", 2048, 0.0),
        ("ctx2048_d0.25", 2048, 0.25),
        ("ctx2048_d0.50", 2048, 0.50),
        ("ctx2048_d0.75", 2048, 0.75),
        ("ctx4096_d0.0", 4096, 0.0),
        ("ctx4096_d0.25", 4096, 0.25),
        ("ctx1024_d0.0", 1024, 0.0),
    ]
    
    device = torch.device(DEVICE)
    
    for label, ctx_len, depth in test_configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        
        needle_len = len(tokenizer.encode(needle, add_special_tokens=False))
        total_content = ctx_len - needle_len - 50
        insert_pos = int(depth * total_content)
        
        # Build context tokens
        hay_tokens = tokenizer.encode(haystack, add_special_tokens=False)
        repeated = []
        while len(repeated) < total_content:
            repeated.extend(hay_tokens)
        repeated = repeated[:total_content]
        
        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
        context_tokens = repeated[:insert_pos] + needle_tokens + repeated[insert_pos:]
        context_tokens = context_tokens[:ctx_len]
        
        question = "\n\nBased on the text above, what is the special secret code mentioned in the document?"
        question_tokens = tokenizer.encode(question, add_special_tokens=False)[:100]
        
        ctx_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
        q_ids = torch.tensor([question_tokens], dtype=torch.long, device=device)
        
        num_segs = ctx_ids.shape[1] // SEG_LEN
        print(f"  ctx_len={ctx_len}, seg_len={SEG_LEN}, num_segs={num_segs}")
        print(f"  insert_pos={insert_pos}, needle_seg={insert_pos // SEG_LEN}")
        
        with torch.no_grad():
            B = 1
            old_slots = None
            slot_history = []
            
            for seg_idx in range(num_segs):
                seg_ids = ctx_ids[:, seg_idx * SEG_LEN : (seg_idx + 1) * SEG_LEN]
                
                if old_slots is None:
                    slots = slot_model.compressor.get_initial_slots(seg_idx, B, device, torch.bfloat16)
                else:
                    slots = old_slots
                
                mem_tokens = slot_model.compressor.slots_to_memory_tokens(slots)
                
                _, _, seg_hidden = slot_model.forward_segment(seg_ids, None, mem_tokens, stage="ce_only")
                new_slots, _ = slot_model.compressor(seg_hidden, old_slots=old_slots, compute_recon=False)
                
                if old_slots is not None:
                    cos = F.cosine_similarity(
                        new_slots.flatten().unsqueeze(0),
                        old_slots.flatten().unsqueeze(0)
                    ).item()
                    print(f"  seg {seg_idx}: slots norm={new_slots.norm().item():.2f}, cos_prev={cos:.4f}")
                else:
                    print(f"  seg {seg_idx}: slots norm={new_slots.norm().item():.2f} (initial)")
                
                slot_history.append(new_slots.clone())
                old_slots = new_slots
            
            # Generate answer
            mem_tokens = slot_model.compressor.slots_to_memory_tokens(old_slots)
            inputs_embeds = slot_model._embed_with_memory(q_ids, mem_tokens)
            
            K = mem_tokens.shape[1]
            T = q_ids.shape[1]
            attn_mask = slot_model._build_attention_mask(T, K, device).unsqueeze(0)
            position_ids = slot_model._build_position_ids(T, K, device).unsqueeze(0)
            
            dtype = torch.bfloat16
            bool_mask_4d = attn_mask.unsqueeze(1)
            attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=dtype)
            attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask_4d, float('-inf'))
            
            inner = slot_model.model.get_base_model() if hasattr(slot_model.model, 'get_base_model') else slot_model.model
            outputs = inner.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                use_cache=True,
            )
            logits = inner.lm_head(outputs.last_hidden_state)
            past_kv = outputs.past_key_values
            
            cur_pos = position_ids[:, -1:]
            total_prior = K + T
            
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [next_token]
            
            for step in range(29):
                token_embeds = inner.get_input_embeddings()(next_token)
                step_pos = cur_pos + 1
                step_total = total_prior + step + 1
                step_mask = torch.zeros(B, 1, 1, step_total, dtype=dtype, device=device)
                outputs = inner.model(
                    inputs_embeds=token_embeds,
                    attention_mask=step_mask,
                    position_ids=step_pos,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                logits = inner.lm_head(outputs.last_hidden_state)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_kv = outputs.past_key_values
                cur_pos = step_pos
                generated.append(next_token)
            
            gen_text = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True).strip()
            is_correct = answer.lower() in gen_text.lower()
            
            print(f"  Generated: {repr(gen_text[:120])}")
            print(f"  Correct: {is_correct}")
            
            # Also test: NO MEMORY (zero memory prefix)
            zero_mem = torch.zeros(1, K, 4096, dtype=torch.bfloat16, device=device)
            zero_embeds = slot_model._embed_with_memory(q_ids, zero_mem)
            
            outputs_z = inner.model(
                inputs_embeds=zero_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                use_cache=True,
            )
            logits_z = inner.lm_head(outputs_z.last_hidden_state)
            past_kv_z = outputs_z.past_key_values
            
            next_token_z = logits_z[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_z = [next_token_z]
            cur_pos_z = position_ids[:, -1:]
            
            for step in range(29):
                token_embeds_z = inner.get_input_embeddings()(next_token_z)
                step_pos_z = cur_pos_z + 1
                step_total_z = total_prior + step + 1
                step_mask_z = torch.zeros(B, 1, 1, step_total_z, dtype=dtype, device=device)
                outputs_z = inner.model(
                    inputs_embeds=token_embeds_z,
                    attention_mask=step_mask_z,
                    position_ids=step_pos_z,
                    past_key_values=past_kv_z,
                    use_cache=True,
                )
                logits_z = inner.lm_head(outputs_z.last_hidden_state)
                next_token_z = logits_z[:, -1, :].argmax(dim=-1, keepdim=True)
                past_kv_z = outputs_z.past_key_values
                cur_pos_z = step_pos_z
                generated_z.append(next_token_z)
            
            gen_text_z = tokenizer.decode(torch.cat(generated_z, dim=1)[0], skip_special_tokens=True).strip()
            is_correct_z = answer.lower() in gen_text_z.lower()
            
            print(f"  [ZERO MEM] Generated: {repr(gen_text_z[:120])}")
            print(f"  [ZERO MEM] Correct: {is_correct_z}")

if __name__ == "__main__":
    main()
