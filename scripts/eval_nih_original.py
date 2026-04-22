"""Minimal Needle-in-a-Haystack eval for original RMT.

Tests whether the trained RMT model can recall information from earlier segments.
Uses the same MemoryCell + RecurrentWrapper from the original Bulatov code.
"""

import os
import sys
import json
import argparse
import random
import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'recurrent-memory-transformer'))
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper


def generate_needle_haystack(tokenizer, needle, answer, haystack_text, depth, total_len, segment_size):
    """Insert needle into haystack at given depth ratio, with a retrieval question."""
    # Compute where to insert
    insert_pos = int(total_len * depth)

    # Build full text
    haystack_tokens = tokenizer.encode(haystack_text, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)

    # Build question that asks for the specific answer
    question = f"\n\nQuestion: Based on the text above, what is {answer}? Answer concisely:\n"
    question_tokens = tokenizer.encode(question, add_special_tokens=False)

    # Trim haystack to leave room for needle + question
    max_haystack = total_len - len(needle_tokens) - len(question_tokens)
    if len(haystack_tokens) > max_haystack:
        haystack_tokens = haystack_tokens[:max_haystack]

    # Insert needle
    actual_pos = min(insert_pos, len(haystack_tokens))
    full_tokens = haystack_tokens[:actual_pos] + needle_tokens + haystack_tokens[actual_pos:]

    # Trim to total_len (should already fit)
    full_tokens = full_tokens[:total_len]

    # Append question
    all_tokens = full_tokens + question_tokens

    return all_tokens


def eval_single(model, tokenizer, needle, answer, haystack, device,
                num_mem_tokens, input_size, max_n_segments, depth, num_trials=1):
    """Run NiH eval for a single needle/answer pair at given depth."""
    segment_size = input_size
    total_len = segment_size * max_n_segments

    correct = 0
    for _ in range(num_trials):
        tokens = generate_needle_haystack(tokenizer, needle, answer, haystack, depth, total_len, segment_size)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            # Generate answer
            memory_state = None
            # Segment the input
            segments = []
            for i in range(0, len(tokens), segment_size):
                seg = tokens[i:i+segment_size]
                if len(seg) < segment_size:
                    seg = seg + [tokenizer.pad_token_id] * (segment_size - len(seg))
                segments.append(seg)

            if len(segments) == 0:
                segments = [tokens[:segment_size].ljust(segment_size, tokenizer.pad_token_id)]

            # Process all segments except last (for memory accumulation)
            for seg in segments[:-1]:
                seg_ids = torch.tensor([seg], dtype=torch.long, device=device)
                _, memory_state = model.memory_cell(
                    input_ids=seg_ids,
                    memory_state=memory_state,
                    output_hidden_states=True,
                )

            # Generate from last segment
            last_seg = segments[-1]
            last_ids = torch.tensor([last_seg], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(last_ids)
            if tokenizer.pad_token_id is not None:
                attention_mask[last_ids == tokenizer.pad_token_id] = 0

            generated = model.memory_cell.generate(
                input_ids=last_ids,
                memory_state=memory_state,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
            )

            # Decode
            # When using inputs_embeds, HF generate returns ONLY new tokens (no input prefix)
            new_tokens = generated[0]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Check if answer is in response
            if answer.lower() in response.lower():
                correct += 1

    return correct, num_trials, response


def main():
    parser = argparse.ArgumentParser(description="RMT Needle-in-Haystack Eval")
    parser.add_argument("--base_model", required=True, help="Path to trained model")
    parser.add_argument("--memory_path", required=True, help="Path to rmt_memory.pt")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path (default: same as base_model)")
    parser.add_argument("--output_dir", default="outputs/nih_eval_original")
    parser.add_argument("--num_mem_tokens", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--max_n_segments", type=int, default=4)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--depths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_path = args.tokenizer_path or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print(f"[NiH-Eval] Loading model from {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    base_model.gradient_checkpointing_enable()

    # Load RMT memory weights
    print(f"[NiH-Eval] Loading memory from {args.memory_path}...")
    memory_cell = MemoryCell(base_model, num_mem_tokens=args.num_mem_tokens)
    mem_data = torch.load(args.memory_path, map_location=device)
    if isinstance(mem_data, dict) and 'memory' in mem_data:
        memory_cell.memory = torch.nn.Parameter(mem_data['memory'].to(device=device, dtype=base_model.dtype))
    else:
        print("[NiH-Eval] Warning: memory file format unexpected, using random init")

    # RecurrentWrapper for generate
    rmt_config = {
        'segment_size': args.input_size,
        'max_n_segments': args.max_n_segments,
        'bptt_depth': -1,
        'segment_alignment': 'right',
    }
    rmt_model = RecurrentWrapper(memory_cell, **rmt_config)

    # Test cases
    needles = [
        ("The secret code is X7K9M2.", "X7K9M2"),
        ("Alice's phone number is 555-1234.", "555-1234"),
        ("The meeting is scheduled for Tuesday at 3pm.", "Tuesday"),
        ("The treasure is buried under the oak tree at coordinates 42N 71W.", "42N 71W"),
        ("Professor Chen's office is in Room 305 of the Physics Building.", "Room 305"),
    ]

    # Generate haystack (repetitive but varied text)
    haystack = ("The library was quiet that afternoon. Students sat at their desks reading various books about "
                "history, science, and literature. The sun shone through the windows, casting long shadows on the floor. "
                "A cat wandered between the shelves, looking for a comfortable spot to nap. The librarian was cataloging "
                "new arrivals and organizing the periodicals section. Outside, birds sang in the trees lining the campus walkway. "
                "A group of tourists walked past the building, admiring the old architecture. The clock on the wall showed "
                "it was almost time for the evening lecture series to begin. Faculty members discussed research proposals "
                "over coffee in the break room. The email system sent notifications about upcoming deadlines. "
                "A research assistant was calibrating equipment in the basement laboratory. ") * 100

    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[NiH-Eval] Starting eval: {len(needles)} needles × {len(args.depths)} depths × {args.num_trials} trials")
    print(f"[NiH-Eval] Segment size: {args.input_size}, Max segments: {args.max_n_segments}, Mem tokens: {args.num_mem_tokens}")
    print("=" * 60)

    for needle, answer in needles:
        for depth in args.depths:
            correct, total, response = eval_single(
                rmt_model, tokenizer, needle, answer, haystack, device,
                args.num_mem_tokens, args.input_size, args.max_n_segments,
                depth, args.num_trials,
            )
            result = {
                "needle": needle,
                "answer": answer,
                "depth": depth,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else 0.0,
                "sample_response": response,
            }
            results.append(result)
            status = "✓" if correct > 0 else "✗"
            print(f"  {status} depth={depth:.2f} acc={correct}/{total}  "
                  f"needle='{needle[:30]}...'  answer='{answer}'  response='{response[:60]}'")

    # Summary
    total_correct = sum(r["correct"] for r in results)
    total_trials = sum(r["total"] for r in results)
    overall_acc = total_correct / total_trials if total_trials > 0 else 0.0

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": args.base_model,
        "memory_path": args.memory_path,
        "num_mem_tokens": args.num_mem_tokens,
        "input_size": args.input_size,
        "max_n_segments": args.max_n_segments,
        "num_trials": args.num_trials,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_trials": total_trials,
        "results": results,
    }

    out_path = os.path.join(args.output_dir, "nih_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[NiH-Eval] Overall accuracy: {total_correct}/{total_trials} = {overall_acc:.1%}")
    print(f"[NiH-Eval] Results saved to {out_path}")

    # Per-depth breakdown
    print("\nPer-depth breakdown:")
    for d in args.depths:
        d_results = [r for r in results if r["depth"] == d]
        d_acc = sum(r["correct"] for r in d_results) / max(sum(r["total"] for r in d_results), 1)
        print(f"  depth={d:.2f}: {d_acc:.1%}")


if __name__ == "__main__":
    main()
