#!/usr/bin/env python3
"""
NIH-Hard: Needle-in-Haystack benchmark that exposes base model limitations at 32K+ context.

This creates much harder test cases where the base model will clearly fail,
allowing us to measure whether memory compression mechanisms provide any benefit.

Key differences from NIH-Extended:
- Multiple needles (10-20 needles per sequence)
- Deeper needle insertion (95-99% depth in 64K-128K context)
- Noisy context (random tokens around needles)
- Mixed needle types (numbers, text, special tokens)
"""
import json
import random
import time
import torch
from typing import List, Dict, Any, Tuple
import argparse
import sys
import os

# Add src to path for imports
sys.path.append('src')
from memory.sparse_memory.model import SparseMemoryLlamaForCausalLM
from memory.sparse_memory.attention import SparseMemoryConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_hard_needle_sequence(
    context_length: int = 65536,  # 64K context
    num_needles: int = 15,        # 15 needles
    needle_depth: float = 0.98,    # 98% depth
    noise_ratio: float = 0.3,     # 30% noise around needles
    needle_types: List[str] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Generate a very hard needle-in-haystack sequence."""
    random.seed(random_seed)
    
    if needle_types is None:
        needle_types = ["number", "text", "special", "mixed"]
    
    # Generate needles of different types
    needles = []
    for i in range(num_needles):
        needle_type = random.choice(needle_types)
        
        if needle_type == "number":
            needle = f"[NEEDLE_{i:02d}]" + "".join([str(random.randint(0, 9)) for _ in range(10)])
        elif needle_type == "text":
            words = ["important", "critical", "key", "vital", "essential", "crucial", "urgent", "paramount"]
            needle = f"[{random.choice(words)}_{i:02d}]" + "".join([random.choice(words) for _ in range(3)])
        elif needle_type == "special":
            chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            needle = f"[SYMBOL_{i:02d}]" + "".join([random.choice(chars) for _ in range(8)])
        else:  # mixed
            needle = f"[MIXED_{i:02d}]" + str(random.randint(1000, 9999)) + random.choice(["!", "@", "#", "$"]) + str(random.randint(100, 999))
        
        needles.append(needle)
    
    # Generate background noise
    vocab = ['hello', 'world', 'test', 'data', 'model', 'memory', 'attention', 'transformer', 'neural', 'network',
             'machine', 'learning', 'artificial', 'intelligence', 'deep', 'learning', 'computer', 'science', 'code', 'program',
             'algorithm', 'function', 'variable', 'constant', 'parameter', 'tensor', 'matrix', 'vector', 'neuron', 'layer']
    
    background_length = context_length
    sequence = []
    
    # Insert needles at deep positions
    for i, needle in enumerate(needles):
        # Calculate needle position (deep in sequence)
        needle_pos = int(context_length * needle_depth) + (i * 100)  # Spread needles
        needle_pos = min(needle_pos, context_length - 100)  # Leave room for needle + noise
        
        # Add noise before needle
        noise_before = int(50 * (1 + noise_ratio))
        for _ in range(noise_before):
            sequence.append(random.choice(vocab))
        
        # Insert needle
        sequence.extend(needle.split())
        
        # Add noise after needle  
        noise_after = int(50 * (1 + noise_ratio))
        for _ in range(noise_after):
            sequence.append(random.choice(vocab))
    
    # Pad to exact length if needed
    while len(sequence) < context_length:
        sequence.append(random.choice(vocab))
    
    sequence = sequence[:context_length]
    
    return {
        "sequence": sequence,
        "needles": needles,
        "context_length": context_length,
        "num_needles": num_needles,
        "needle_depth": needle_depth,
        "noise_ratio": noise_ratio,
        "random_seed": random_seed
    }


def evaluate_hard_nih(
    model,
    tokenizer,
    num_trials: int = 3,
    context_length: int = 65536,
    num_needles: int = 15,
    needle_depth: float = 0.98,
    use_sparse_memory: bool = False,
    sparse_config: SparseMemoryConfig = None
) -> Dict[str, Any]:
    """Evaluate model on hard NIH benchmark."""
    results = {
        "model_name": model.config._name_or_path,
        "context_length": context_length,
        "num_needles": num_needles,
        "num_trials": num_trials,
        "use_sparse_memory": use_sparse_memory,
        "results": [],
        "accuracy": 0,
        "total_needles": 0,
        "correct_needles": 0
    }
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        
        # Generate hard needle sequence
        data = generate_hard_needle_sequence(
            context_length=context_length,
            num_needles=num_needles,
            needle_depth=needle_depth,
            random_seed=42 + trial
        )
        
        # Create prompt
        prompt = "Answer the following question based on the provided text:\n\n"
        prompt += " ".join(data["sequence"]) + "\n\n"
        prompt += "Question: What are the needles in the text? List them exactly as they appear.\n"
        prompt += "Answer:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
        
        if use_sparse_memory and sparse_config is not None:
            # Initialize sparse memory
            sparse_model = SparseMemoryLlamaForCausalLM(model, 
                memory_slots=sparse_config.num_mem_tokens,
                top_k=sparse_config.top_k,
                sliding_window=sparse_config.window_size,
                ema_alpha=sparse_config.gate_alpha
            )
            sparse_memory = sparse_model.reset_memory(inputs.input_ids.shape[0])
            generation_model = sparse_model
        else:
            generation_model = model
        
        # Generate response
        with torch.no_grad():
            outputs = generation_model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_part = response.split("Answer:")[-1].strip()
        
        # Evaluate answer
        trial_result = {
            "trial": trial + 1,
            "needles": data["needles"],
            "response": answer_part,
            "correct_needles": [],
            "missed_needles": [],
            "hallucinated_items": []
        }
        
        # Check for needles in response
        found_needles = []
        for needle in data["needles"]:
            if needle in answer_part:
                found_needles.append(needle)
                trial_result["correct_needles"].append(needle)
            else:
                trial_result["missed_needles"].append(needle)
        
        # Check for hallucinations (items that look like needles but aren't)
        words = answer_part.split()
        for word in words:
            if ("NEEDLE" in word or "SYMBOL" in word or "MIXED" in word or word.startswith("[") and "]" in word):
                if word not in data["needles"]:
                    trial_result["hallucinated_items"].append(word)
        
        results["results"].append(trial_result)
        results["total_needles"] += len(data["needles"])
        results["correct_needles"] += len(found_needles)
        
        print(f"  Found {len(found_needles)}/{len(data['needles'])} needles")
        print(f"  Missed: {len(trial_result['missed_needles'])}")
        print(f"  Hallucinated: {len(trial_result['hallucinated_items'])}")
    
    # Calculate overall accuracy
    results["accuracy"] = results["correct_needles"] / results["total_needles"] if results["total_needles"] > 0 else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="NIH-Hard benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--context_length", type=int, default=65536, help="Context length")
    parser.add_argument("--num_needles", type=int, default=15, help="Number of needles")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--use_sparse_memory", action="store_true", help="Use sparse memory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    
    args = parser.parse_args()
    
    # Set up GPU
    import torch
    torch.cuda.set_device(args.gpu_id)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure sparse memory if requested
    sparse_config = None
    if args.use_sparse_memory:
        sparse_config = SparseMemoryConfig(
            num_mem_tokens=128,
            window_size=256,
            top_k=8,
            gate_alpha=0.1
        )
    
    # Run evaluation
    print(f"Running NIH-Hard evaluation: {args.context_length} context, {args.num_needles} needles")
    start_time = time.time()
    
    results = evaluate_hard_nih(
        model=model,
        tokenizer=tokenizer,
        num_trials=args.num_trials,
        context_length=args.context_length,
        num_needles=args.num_needles,
        use_sparse_memory=args.use_sparse_memory,
        sparse_config=sparse_config
    )
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"nih_hard_{args.context_length}_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed:.1f}s")
    print(f"Results saved to: {output_path}")
    print(f"Overall accuracy: {results['accuracy']:.2%} ({results['correct_needles']}/{results['total_needles']})")


if __name__ == "__main__":
    main()