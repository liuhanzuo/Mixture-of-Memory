#!/usr/bin/env python3
"""
Debug v3: Test retrieval WITHOUT answer in prompt - pure memory test.
"""
import os, sys, json, torch, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import (
    RMTMemory, RMTModel, build_rmt_attention_mask, build_rmt_position_ids,
)

CKPT = "outputs/rmt_v10_8gpu_20260419_001626_20260419_001703/final/"
DEVICE = torch.device("cuda:0")

def main():
    with open(os.path.join(CKPT, "rmt_config.json")) as f:
        cfg = json.load(f)
    seg_len = cfg["segment_length"]
    n_mem = cfg["num_memory_tokens"]
    
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": str(DEVICE)}, attn_implementation="eager",
    )
    model.eval()
    
    rmt_memory = RMTMemory(
        hidden_dim=model.config.hidden_size,
        num_memory_tokens=n_mem,
        num_heads=8,
        max_segments=cfg["max_segments"] + 1,
        bottleneck_dim=cfg.get("bottleneck_dim", 256),
        extractor_version=cfg.get("extractor_version", 5),
    ).to(device=DEVICE, dtype=torch.bfloat16)
    rmt_memory.load_state_dict(torch.load(os.path.join(CKPT, "rmt_memory.pt"), map_location=DEVICE))
    rmt_memory.eval()
    
    rmt_model = RMTModel(model, rmt_memory, segment_length=seg_len)
    rmt_model.eval()
    
    haystack_topics = [
        "The history of computing spans several centuries from early mechanical calculators to modern quantum systems.",
        "Photosynthesis is the process by which green plants transform light energy into chemical energy during daylight.",
    ]
    rng = random.Random(42)
    tokens_per_seg = seg_len - n_mem
    
    # === Test 1: Single segment, needle in it, Chinese retrieval question ===
    needle = "阿尔法 的编号是 ABC123。"
    needle_toks = tok.encode(needle, add_special_tokens=False)
    
    hay_tokens = []
    while len(hay_tokens) < tokens_per_seg - len(needle_toks):
        hay_tokens.extend(tok.encode(rng.choice(haystack_topics), add_special_tokens=False))
    
    all_tokens = hay_tokens[:tokens_per_seg - len(needle_toks)] + needle_toks
    padded = all_tokens + [tok.pad_token_id] * (seg_len - len(all_tokens))
    input_ids = torch.tensor([padded], device=DEVICE, dtype=torch.long)
    
    question = "请问 阿尔法 的编号是什么？"
    
    with torch.no_grad():
        # Process segment
        mem = rmt_memory.get_initial_memory(0, 1, DEVICE, torch.bfloat16)
        _, seg_hidden = rmt_model._forward_single_segment(input_ids, None, mem, 0)
        mem_result = rmt_memory.extract_memory(seg_hidden, mem)
        final_mem = mem_result[0] if isinstance(mem_result, tuple) else mem_result
        
        # Generate with memory (no answer in prompt!)
        q_ids = tok.encode(question, add_special_tokens=False)
        q_tensor = torch.tensor([q_ids], device=DEVICE, dtype=torch.long)
        q_len = len(q_ids)
        
        inputs_embeds = rmt_model._embed_with_memory(q_tensor, final_mem)
        attn_mask = build_rmt_attention_mask(q_len, n_mem, DEVICE)
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)
        attn_mask_float = torch.zeros_like(attn_mask_4d, dtype=torch.float32)
        attn_mask_float = attn_mask_float.masked_fill(~attn_mask_4d, float('-inf'))
        pos_ids = build_rmt_position_ids(q_len, n_mem, 0, DEVICE).unsqueeze(0)
        
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_float},
            position_ids=pos_ids,
        )
        last_logit = model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, 0]
        top10 = torch.topk(last_logit, 10)
        top10_tokens = tok.convert_ids_to_tokens(top10.indices.tolist())
        
        print("=== TEST 1: Single segment, Chinese needle 'ABC123' ===")
        print(f"  Question: {question}")
        print(f"  Top 10: {top10_tokens}")
        print(f"  Probs: {[f'{p:.3f}' for p in torch.softmax(top10.values, dim=-1).tolist()]}")
        
        # Check specific answer tokens
        answer = " ABC123"
        answer_ids = tok.encode(answer, add_special_tokens=False)
        answer_text = tok.convert_ids_to_tokens(answer_ids)
        print(f"  Answer tokens: {answer_ids} = {answer_text}")
        for aid, atxt in zip(answer_ids, answer_text):
            print(f"    '{atxt}' (id={aid}): logit={last_logit[aid].item():.3f}, rank={((last_logit > last_logit[aid]).sum().item() + 1)}")
        
        # Greedy decode a few tokens
        print("\n  Greedy decode:")
        gen_embeds = inputs_embeds
        for i in range(10):
            cur_len = gen_embeds.shape[1] - n_mem  # exclude memory
            mask = build_rmt_attention_mask(cur_len, n_mem, DEVICE)
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            mask_float = torch.zeros_like(mask_4d, dtype=torch.float32).masked_fill(~mask_4d, float('-inf'))
            pid = build_rmt_position_ids(cur_len, n_mem, 0, DEVICE).unsqueeze(0)
            
            out = model.model(inputs_embeds=gen_embeds, attention_mask={"full_attention": mask_float}, position_ids=pid)
            nxt_logit = model.lm_head(out.last_hidden_state[:, -1:, :])[0, 0]
            nxt_tok = torch.argmax(nxt_logit).item()
            tok_text = tok.convert_ids_to_tokens([nxt_tok])[0]
            if nxt_tok == tok.eos_token_id:
                print(f"    [{i}] {tok_text} (EOS)")
                break
            print(f"    [{i}] {tok_text} (id={nxt_tok})")
            nxt_embed = model.get_input_embeddings()(torch.tensor([[nxt_tok]], device=DEVICE))
            gen_embeds = torch.cat([gen_embeds, nxt_embed], dim=1)
    
    # === Test 2: Same but WITHOUT memory (zero/initial memory) ===
    print("\n\n=== TEST 2: Same but WITHOUT processed memory (initial) ===")
    with torch.no_grad():
        zero_mem = rmt_memory.get_initial_memory(0, 1, DEVICE, torch.bfloat16)
        inputs_embeds = rmt_model._embed_with_memory(q_tensor, zero_mem)
        attn_mask = build_rmt_attention_mask(q_len, n_mem, DEVICE)
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)
        attn_mask_float = torch.zeros_like(attn_mask_4d, dtype=torch.float32)
        attn_mask_float = attn_mask_float.masked_fill(~attn_mask_4d, float('-inf'))
        pos_ids = build_rmt_position_ids(q_len, n_mem, 0, DEVICE).unsqueeze(0)
        
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_float},
            position_ids=pos_ids,
        )
        last_logit = model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, 0]
        top10 = torch.topk(last_logit, 10)
        print(f"  Top 10: {tok.convert_ids_to_tokens(top10.indices.tolist())}")
        for aid, atxt in zip(answer_ids, answer_text):
            print(f"    '{atxt}' (id={aid}): logit={last_logit[aid].item():.3f}")
        
        # Greedy decode
        print("  Greedy decode:")
        gen_embeds = inputs_embeds
        for i in range(10):
            cur_len = gen_embeds.shape[1] - n_mem
            mask = build_rmt_attention_mask(cur_len, n_mem, DEVICE)
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            mask_float = torch.zeros_like(mask_4d, dtype=torch.float32).masked_fill(~mask_4d, float('-inf'))
            pid = build_rmt_position_ids(cur_len, n_mem, 0, DEVICE).unsqueeze(0)
            out = model.model(inputs_embeds=gen_embeds, attention_mask={"full_attention": mask_float}, position_ids=pid)
            nxt_logit = model.lm_head(out.last_hidden_state[:, -1:, :])[0, 0]
            nxt_tok = torch.argmax(nxt_logit).item()
            tok_text = tok.convert_ids_to_tokens([nxt_tok])[0]
            if nxt_tok == tok.eos_token_id:
                print(f"    [{i}] {tok_text} (EOS)")
                break
            print(f"    [{i}] {tok_text} (id={nxt_tok})")
            nxt_embed = model.get_input_embeddings()(torch.tensor([[nxt_tok]], device=DEVICE))
            gen_embeds = torch.cat([gen_embeds, nxt_embed], dim=1)

    # === Test 3: Question WITH segment_idx matching training ===
    print("\n\n=== TEST 3: Using segment_idx=1 for question (matching training retrieval) ===")
    with torch.no_grad():
        # In training, retrieval uses segment_idx of the needle-containing segment
        # For 2-segment doc with needle at end of seg 0, retrieval would use seg 1's memory
        # Let's match: process 2 segments, use memory after seg 1
        old_memory = None
        for seg_idx in range(2):
            mem = rmt_memory.get_initial_memory(seg_idx, 1, DEVICE, torch.bfloat16) if old_memory is None else old_memory
            _, seg_hidden = rmt_model._forward_single_segment(input_ids, None, mem, seg_idx)
            mem_result = rmt_memory.extract_memory(seg_hidden, old_memory)
            old_memory = mem_result[0] if isinstance(mem_result, tuple) else mem_result
        
        # Generate with this memory and segment_idx=1 position ids
        inputs_embeds = rmt_model._embed_with_memory(q_tensor, old_memory)
        # Use segment_idx=1 for position ids (matching training)
        pos_ids = build_rmt_position_ids(q_len, n_mem, 1, DEVICE).unsqueeze(0)
        attn_mask = build_rmt_attention_mask(q_len, n_mem, DEVICE)
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)
        attn_mask_float = torch.zeros_like(attn_mask_4d, dtype=torch.float32)
        attn_mask_float = attn_mask_float.masked_fill(~attn_mask_4d, float('-inf'))
        
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_float},
            position_ids=pos_ids,
        )
        last_logit = model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, 0]
        top10 = torch.topk(last_logit, 10)
        print(f"  Top 10: {tok.convert_ids_to_tokens(top10.indices.tolist())}")
        for aid, atxt in zip(answer_ids, answer_text):
            print(f"    '{atxt}' (id={aid}): logit={last_logit[aid].item():.3f}")
        
        print("  Greedy decode:")
        gen_embeds = inputs_embeds
        for i in range(10):
            cur_len = gen_embeds.shape[1] - n_mem
            mask = build_rmt_attention_mask(cur_len, n_mem, DEVICE)
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            mask_float = torch.zeros_like(mask_4d, dtype=torch.float32).masked_fill(~mask_4d, float('-inf'))
            pid = build_rmt_position_ids(cur_len, n_mem, 1, DEVICE).unsqueeze(0)
            out = model.model(inputs_embeds=gen_embeds, attention_mask={"full_attention": mask_float}, position_ids=pid)
            nxt_logit = model.lm_head(out.last_hidden_state[:, -1:, :])[0, 0]
            nxt_tok = torch.argmax(nxt_logit).item()
            tok_text = tok.convert_ids_to_tokens([nxt_tok])[0]
            if nxt_tok == tok.eos_token_id:
                print(f"    [{i}] {tok_text} (EOS)")
                break
            print(f"    [{i}] {tok_text} (id={nxt_tok})")
            nxt_embed = model.get_input_embeddings()(torch.tensor([[nxt_tok]], device=DEVICE))
            gen_embeds = torch.cat([gen_embeds, nxt_embed], dim=1)

if __name__ == "__main__":
    main()
