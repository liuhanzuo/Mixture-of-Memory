"""
Minimal RMT v10 inference debug script.
Tests whether memory actually changes between segments and whether generation works.
"""
import os, sys, torch, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import RMTMemory, RMTModel, build_rmt_attention_mask, build_rmt_position_ids

CKPT = "outputs/rmt_v10_8gpu_20260419_001626_20260419_001703/final/"
BASE_MODEL = "../models/Qwen--Qwen3-8b"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # Load config
    with open(os.path.join(CKPT, "rmt_config.json")) as f:
        cfg = json.load(f)
    print(f"Config: extractor_v={cfg['extractor_version']}, num_mem={cfg['num_memory_tokens']}, seg_len={cfg['segment_length']}, bottleneck={cfg['bottleneck_dim']}")

    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    model.eval()

    hidden_dim = model.config.hidden_size

    # Load RMT memory
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=cfg["num_memory_tokens"],
        num_heads=8,
        max_segments=cfg["max_segments"] + 1,
        bottleneck_dim=cfg["bottleneck_dim"],
        extractor_version=cfg["extractor_version"],
    ).to(device=DEVICE, dtype=torch.bfloat16)
    rmt_memory.load_state_dict(torch.load(os.path.join(CKPT, "rmt_memory.pt"), map_location=DEVICE))
    rmt_memory.eval()

    rmt_model = RMTModel(model, rmt_memory, segment_length=cfg["segment_length"])
    rmt_model.eval()

    print(f"\nModel loaded. hidden_dim={hidden_dim}")

    # === Test 1: Check initial memory values ===
    print("\n" + "="*60)
    print("TEST 1: Initial memory values per segment")
    print("="*60)
    for seg_idx in range(3):
        mem = rmt_memory.get_initial_memory(seg_idx, 1, DEVICE, torch.bfloat16)
        print(f"  seg_idx={seg_idx}: shape={mem.shape}, mean={mem.float().mean():.6f}, std={mem.float().std():.6f}, "
              f"min={mem.float().min():.6f}, max={mem.float().max():.6f}")

    # === Test 2: Process 2 segments and check memory extraction ===
    print("\n" + "="*60)
    print("TEST 2: Memory extraction across 2 segments")
    print("="*60)

    seg_len = cfg["segment_length"]
    seg1_text = "The secret code is 7319. Remember it well. " * 100  # ~600 tokens
    seg2_text = "This is filler text about computing history. " * 100

    seg1_ids = tokenizer.encode(seg1_text, add_special_tokens=False)
    seg2_ids = tokenizer.encode(seg2_text, add_special_tokens=False)

    # Trim to segment_length
    seg1_ids = seg1_ids[:seg_len]
    seg2_ids = seg2_ids[:seg_len]

    seg1_tensor = torch.tensor([seg1_ids], dtype=torch.long, device=DEVICE)
    seg2_tensor = torch.tensor([seg2_ids], dtype=torch.long, device=DEVICE)

    B = 1
    with torch.no_grad():
        # Segment 0
        mem0 = rmt_memory.get_initial_memory(0, B, DEVICE, torch.bfloat16)
        print(f"\n  Initial mem (seg0): mean={mem0.float().mean():.6f}, std={mem0.float().std():.6f}")

        _, seg_hidden0 = rmt_model._forward_single_segment(seg1_tensor, None, mem0, 0)
        print(f"  Seg0 hidden: shape={seg_hidden0.shape}, mean={seg_hidden0.float().mean():.6f}, std={seg_hidden0.float().std():.6f}")

        mem_result0 = rmt_memory.extract_memory(seg_hidden0, None)
        mem1 = mem_result0[0] if isinstance(mem_result0, tuple) else mem_result0
        print(f"  Extracted mem (seg0→seg1): mean={mem1.float().mean():.6f}, std={mem1.float().std():.6f}")

        # Check if memory changed
        diff01 = (mem1 - mem0).abs().mean().item()
        print(f"  |mem1 - mem0| mean diff = {diff01:.8f}")

        # Segment 1
        _, seg_hidden1 = rmt_model._forward_single_segment(seg2_tensor, None, mem1, 1)
        print(f"\n  Seg1 hidden: shape={seg_hidden1.shape}, mean={seg_hidden1.float().mean():.6f}")

        mem_result1 = rmt_memory.extract_memory(seg_hidden1, mem1)
        mem2 = mem_result1[0] if isinstance(mem_result1, tuple) else mem_result1
        print(f"  Extracted mem (seg1→seg2): mean={mem2.float().mean():.6f}, std={mem2.float().std():.6f}")

        diff12 = (mem2 - mem1).abs().mean().item()
        print(f"  |mem2 - mem1| mean diff = {diff12:.8f}")

    # === Test 3: Generate with memory ===
    print("\n" + "="*60)
    print("TEST 3: Generation with memory (using generate_rmt approach)")
    print("="*60)

    question = "What is the secret code?"
    q_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    n_mem = cfg["num_memory_tokens"]
    last_seg_idx = 1  # we processed 2 segments (0, 1)

    with torch.no_grad():
        inputs_embeds = rmt_model._embed_with_memory(q_ids, mem2)
        q_len = q_ids.shape[1]

        print(f"  Question: '{question}'")
        print(f"  q_ids shape: {q_ids.shape}")
        print(f"  inputs_embeds shape: {inputs_embeds.shape}")

        # Build attention mask (from eval script approach)
        max_new_tokens = 30
        max_combined = q_len + max_new_tokens
        gen_position_ids = build_rmt_position_ids(max_combined, n_mem, last_seg_idx, DEVICE).unsqueeze(0)
        init_position_ids = gen_position_ids[:, :n_mem + q_len]

        gen_attn_mask = build_rmt_attention_mask(q_len, n_mem, DEVICE)
        gen_attn_mask_4d = gen_attn_mask.unsqueeze(0).unsqueeze(0)
        gen_attn_mask_float = torch.zeros_like(gen_attn_mask_4d, dtype=torch.bfloat16)
        gen_attn_mask_float = gen_attn_mask_float.masked_fill(~gen_attn_mask_4d, float('-inf'))

        print(f"  Position IDs (first 10): {init_position_ids[0, :10].tolist()}")
        print(f"  Position IDs (mem range): {init_position_ids[0, :n_mem].min().item()}..{init_position_ids[0, :n_mem].max().item()}")
        print(f"  Position IDs (text range): {init_position_ids[0, n_mem:].min().item()}..{init_position_ids[0, n_mem:].max().item()}")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = []
            for step in range(max_new_tokens):
                cur_len = inputs_embeds.shape[1]
                if cur_len == n_mem + q_len:
                    cur_mask = gen_attn_mask_float
                    cur_pos = init_position_ids
                else:
                    cur_text_len = cur_len - n_mem
                    cur_mask_bool = build_rmt_attention_mask(cur_text_len, n_mem, DEVICE)
                    cur_mask_4d = cur_mask_bool.unsqueeze(0).unsqueeze(0)
                    cur_mask = torch.zeros_like(cur_mask_4d, dtype=torch.bfloat16).masked_fill(~cur_mask_4d, torch.tensor(float('-inf'), dtype=torch.bfloat16))
                    cur_pos = gen_position_ids[:, :cur_len]

                outputs = rmt_model.model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask={"full_attention": cur_mask},
                    position_ids=cur_pos,
                    output_hidden_states=False,
                )
                logits = rmt_model.model.lm_head(outputs.last_hidden_state[:, -1:, :])
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
                generated_ids.append(next_tok.item())

                if step < 5:
                    top5 = torch.topk(logits[0, -1], 5)
                    top5_tokens = [tokenizer.decode([t]) for t in top5.indices.tolist()]
                    print(f"  Step {step}: top5={list(zip(top5_tokens, [f'{p:.2f}' for p in top5.values.softmax(-1).tolist()]))}")

                next_embed = rmt_model.model.get_input_embeddings()(next_tok)
                inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n  Generated answer: '{answer}'")

    # === Test 4: Same generation WITHOUT memory (baseline) ===
    print("\n" + "="*60)
    print("TEST 4: Generation WITHOUT memory (random mem embeddings)")
    print("="*60)

    with torch.no_grad():
        # Use initial memory from seg0 (no extraction)
        random_mem = rmt_memory.get_initial_memory(0, B, DEVICE, torch.bfloat16)
        inputs_embeds_no = rmt_model._embed_with_memory(q_ids, random_mem)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_no_mem_ids = []
            for step in range(max_new_tokens):
                cur_len = inputs_embeds_no.shape[1]
                if cur_len == n_mem + q_len:
                    cur_mask = gen_attn_mask_float
                    cur_pos = init_position_ids
                else:
                    cur_text_len = cur_len - n_mem
                    cur_mask_bool = build_rmt_attention_mask(cur_text_len, n_mem, DEVICE)
                    cur_mask_4d = cur_mask_bool.unsqueeze(0).unsqueeze(0)
                    cur_mask = torch.zeros_like(cur_mask_4d, dtype=torch.bfloat16).masked_fill(~cur_mask_4d, torch.tensor(float('-inf'), dtype=torch.bfloat16))
                    cur_pos = gen_position_ids[:, :cur_len]

                outputs = rmt_model.model.model(
                    inputs_embeds=inputs_embeds_no,
                    attention_mask={"full_attention": cur_mask},
                    position_ids=cur_pos,
                )
                logits = rmt_model.model.lm_head(outputs.last_hidden_state[:, -1:, :])
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
                gen_no_mem_ids.append(next_tok.item())
                next_embed = rmt_model.model.get_input_embeddings()(next_tok)
                inputs_embeds_no = torch.cat([inputs_embeds_no, next_embed], dim=1)

    answer_no_mem = tokenizer.decode(gen_no_mem_ids, skip_special_tokens=True)
    print(f"  Generated (no memory): '{answer_no_mem}'")

    # === Test 5: Compare logits with/without memory ===
    print("\n" + "="*60)
    print("TEST 5: Logit comparison (with mem vs without mem)")
    print("="*60)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # With memory
            out_mem = rmt_model.model.model(
                inputs_embeds=rmt_model._embed_with_memory(q_ids, mem2),
                attention_mask={"full_attention": gen_attn_mask_float},
                position_ids=init_position_ids,
            )
            logits_mem = rmt_model.model.lm_head(out_mem.last_hidden_state[:, -1, :])

            # Without memory (random)
            out_nomem = rmt_model.model.model(
                inputs_embeds=rmt_model._embed_with_memory(q_ids, random_mem),
                attention_mask={"full_attention": gen_attn_mask_float},
                position_ids=init_position_ids,
            )
            logits_nomem = rmt_model.model.lm_head(out_nomem.last_hidden_state[:, -1, :])

            logit_diff = (logits_mem - logits_nomem).abs()
            print(f"  Logit diff: mean={logit_diff.mean():.6f}, max={logit_diff.max():.6f}")
            print(f"  Top5 with mem: {tokenizer.batch_decode(torch.topk(logits_mem[0], 5).indices.unsqueeze(0))}")
            print(f"  Top5 no mem:   {tokenizer.batch_decode(torch.topk(logits_nomem[0], 5).indices.unsqueeze(0))}")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
