#!/usr/bin/env python3
"""Debug NIH eval with verbose output for a specific setting."""
import os, sys, torch, logging, json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.slot.slot_memory_compressor import SlotMemoryCompressor, SlotMemoryWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import eval helpers
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_slot_memory import generate_nih_sample, generate_with_slot_memory

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--base_model", default="../models/Qwen--Qwen3-8b")
    parser.add_argument("--ctx_len", type=int, default=None)
    parser.add_argument("--depth", type=float, default=None)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context_lengths_8192", action="store_true", help="Run 8192 eval for all depths")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info(f"Loading model from {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": device},
    )

    compressor = SlotMemoryCompressor(
        hidden_dim=4096, num_slots=args.num_slots, slot_dim=args.slot_dim,
        num_iterations=3, dropout=0.1, num_segments=args.max_segments + 1,
    )
    slot_weights_path = os.path.join(args.checkpoint_dir, "slot_weights.pt")
    state = torch.load(slot_weights_path, map_location=device, weights_only=True)
    compressor.load_state_dict(state)

    slot_model = SlotMemoryWrapper(model=base_model, compressor=compressor, segment_length=args.segment_length)
    slot_model = slot_model.to(device=device, dtype=torch.bfloat16)
    slot_model.eval()

    needle_text = "the special secret code is Blue Monkey 42"
    answer_text = "Blue Monkey 42"
    haystack = "The history of computing spans several centuries. Early mechanical devices " * 50

    if args.context_lengths_8192:
        # Task 2: eval at 8192 for all depths
        results = {}
        for depth in [0.25, 0.5, 0.75, 1.0]:
            correct = 0
            outputs = []
            for trial in range(args.num_trials):
                ctx_ids, q_ids, answer = generate_nih_sample(tokenizer, 8192, depth, needle_text, answer_text, haystack)
                ctx_ids = ctx_ids.unsqueeze(0).to(device)
                q_ids = q_ids.unsqueeze(0).to(device)
                with torch.no_grad():
                    gen = generate_with_slot_memory(slot_model, tokenizer, ctx_ids, q_ids, max_new_tokens=30, device=args.device)
                    decoded = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
                    is_correct = answer.lower() in decoded.lower()
                    if is_correct:
                        correct += 1
                    outputs.append({"trial": trial, "generated": decoded, "correct": is_correct})
            acc = correct / args.num_trials
            key = f"ctx8192_d{depth}"
            results[key] = {"accuracy": acc, "correct": correct, "total": args.num_trials, "outputs": outputs}
            logger.info(f"{key}: {acc:.0%} ({correct}/{args.num_trials})")

        print("\n=== 8192 CONTEXT RESULTS ===")
        for k, v in results.items():
            print(f"  {k}: {v['accuracy']:.0%} ({v['correct']}/{v['total']})")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        # Task 1: verbose debug
        results = []
        for trial in range(args.num_trials):
            ctx_ids, q_ids, answer = generate_nih_sample(tokenizer, args.ctx_len, args.depth, needle_text, answer_text, haystack)
            ctx_ids = ctx_ids.unsqueeze(0).to(device)
            q_ids = q_ids.unsqueeze(0).to(device)

            print(f"\n--- Trial {trial} (ctx={args.ctx_len}, depth={args.depth}) ---")
            print(f"  Context shape: {ctx_ids.shape}")
            print(f"  Question shape: {q_ids.shape}")
            print(f"  Expected answer: {answer}")
            print(f"  Question text: {tokenizer.decode(q_ids[0], skip_special_tokens=True)}")

            with torch.no_grad():
                gen = generate_with_slot_memory(slot_model, tokenizer, ctx_ids, q_ids, max_new_tokens=30, device=args.device)
                decoded = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
                is_correct = answer.lower() in decoded.lower()
                print(f"  Generated: {repr(decoded)}")
                print(f"  Correct: {is_correct}")
                results.append({"trial": trial, "generated": decoded, "correct": is_correct})

        correct = sum(r["correct"] for r in results)
        print(f"\n=== ctx{args.ctx_len}_d{args.depth}: {correct}/{args.num_trials} ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
