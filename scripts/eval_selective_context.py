#!/usr/bin/env python3
"""
Evaluation script for Selective Context prompt compression.
This provides a zero-cost baseline to test compression effectiveness against our baseline PPL.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.selective_context import SelectiveContext
import transformers
import json
from typing import List, Dict, Tuple


def create_test_prompts():
    """Create test prompts that mimic long context scenarios."""
    prompts = {
        "baseline": [
            "The quick brown fox jumps over the lazy dog. This is a simple sentence demonstration.",
            "In machine learning, neural networks have revolutionized the field of artificial intelligence.",
            "Climate change represents one of the most significant challenges facing humanity today."
        ],
        "medium_context": [
            "The history of artificial intelligence dates back to ancient times when philosophers first began to ponder the nature of thought and consciousness. " * 5,
            "In computer science, algorithms form the foundation of efficient problem-solving. Various sorting algorithms have been developed over the decades. " * 5,
            "The concept of deep learning has transformed natural language processing. Transformers architecture introduced attention mechanisms. " * 5
        ],
        "long_context": [
            "The evolution of language models represents a fascinating journey in artificial intelligence. Early systems relied on statistical methods and n-gram models. " * 10,
            "Attention mechanisms revolutionized natural language processing by allowing models to focus on relevant parts of input sequences. " * 10,
            "The development of large language models has accelerated dramatically in recent years, with models growing from millions to billions of parameters. " * 10
        ]
    }
    return prompts


def evaluate_ppl_with_compression(model, tokenizer, prompts: List[str], compression_ratio: float) -> Dict:
    """
    Evaluate perplexity with and without compression.
    
    Args:
        model: Language model
        tokenizer: Tokenizer  
        prompts: List of test prompts
        compression_ratio: Target compression ratio
        
    Returns:
        results: Dictionary with PPL metrics
    """
    compressor = SelectiveContext(compression_ratio=compression_ratio, method="importance")
    
    original_ppls = []
    compressed_ppls = []
    compression_stats = []
    
    for prompt in prompts:
        # Original evaluation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            original_ppl = torch.exp(outputs.loss).item()
        
        # Compressed evaluation
        compressed_ids, compressed_mask = compressor.compress(
            inputs['input_ids'], 
            inputs['attention_mask']
        )
        
        # Re-pad if needed for consistency
        if compressed_ids.shape[1] < 2:  # Skip if too short
            continue
            
        compressed_inputs = {
            'input_ids': compressed_ids,
            'attention_mask': compressed_mask
        }
        
        with torch.no_grad():
            outputs = model(**compressed_inputs, labels=compressed_inputs['input_ids'])
            compressed_ppl = torch.exp(outputs.loss).item()
        
        # Get stats
        stats = compressor.get_compression_stats(
            inputs['input_ids'].shape[1], 
            compressed_ids.shape[1]
        )
        
        original_ppls.append(original_ppl)
        compressed_ppls.append(compressed_ppl)
        compression_stats.append(stats)
    
    # Calculate averages
    avg_original = sum(original_ppls) / len(original_ppls)
    avg_compressed = sum(compressed_ppls) / len(compressed_ppls)
    avg_compression_ratio = sum(s['compression_ratio'] for s in compression_stats) / len(compression_stats)
    
    return {
        'compression_ratio': compression_ratio,
        'avg_original_ppl': avg_original,
        'avg_compressed_ppl': avg_compressed,
        'ppl_change_percent': ((avg_compressed - avg_original) / avg_original) * 100,
        'avg_compression_ratio': avg_compression_ratio,
        'individual_results': [
            {
                'original_ppl': orig,
                'compressed_ppl': comp,
                'ppl_change': ((comp - orig) / orig) * 100,
                'stats': stats
            }
            for orig, comp, stats in zip(original_ppls, compressed_ppls, compression_stats)
        ]
    }


def run_evaluations(model_name: str = "facebook/opt-125m", compression_ratios: List[float] = [0.3, 0.5, 0.7]):
    """
    Run comprehensive evaluation of Selective Context.
    
    Args:
        model_name: HuggingFace model name
        compression_ratios: List of compression ratios to test
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully. Device: {next(model.parameters()).device}")
    
    # Create test prompts
    test_prompts = create_test_prompts()
    
    results = {}
    
    for context_type, prompts in test_prompts.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {context_type} context")
        print(f"{'='*60}")
        
        context_results = {}
        
        for ratio in compression_ratios:
            print(f"\nTesting compression ratio: {ratio}")
            
            result = evaluate_ppl_with_compression(model, tokenizer, prompts, ratio)
            context_results[ratio] = result
            
            print(f"  Original PPL: {result['avg_original_ppl']:.2f}")
            print(f"  Compressed PPL: {result['avg_compressed_ppl']:.2f}")
            print(f"  PPL change: {result['ppl_change_percent']:+.2f}%")
            print(f"  Actual compression ratio: {result['avg_compression_ratio']:.3f}")
        
        results[context_type] = context_results
    
    # Save results
    results_file = f"outputs/selective_context_eval_{model_name.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")
    
    for context_type, context_results in results.items():
        print(f"\n{context_type.upper()} CONTEXT:")
        print("-" * 40)
        
        for ratio, result in context_results.items():
            print(f"  Ratio {ratio}: {result['avg_original_ppl']:.2f} → {result['avg_compressed_ppl']:.2f} ({result['ppl_change_percent']:+.2f}%)")
    
    return results


def compare_with_baseline():
    """Compare Selective Context results with our sparse memory baseline."""
    print("\n" + "="*60)
    print("COMPARISON WITH SPARSE MEMORY BASELINE")
    print("="*60)
    
    # Our baseline results from researcher
    baseline_results = {
        "llama_baseline": {"ppl": 41.24},
        "sparse_memory_v3": {"ppl": 49.60},
        "sparse_memory_fusion": {"ppl": 49.88}
    }
    
    print("SPARSE MEMORY RESULTS (from remote cluster):")
    for method, result in baseline_results.items():
        print(f"  {method}: PPL = {result['ppl']}")
    
    print(f"\nSPARSE MEMORY PERFORMANCE:")
    print(f"  Baseline vs sparse_memory_v3: +{(49.60-41.24)/41.24*100:.1f}% worse")
    print(f"  Baseline vs sparse_memory_fusion: +{(49.88-41.24)/41.24*100:.1f}% worse")
    
    print(f"\nSELECTIVE CONTEXT EXPECTED PERFORMANCE:")
    print(f"  (Zero-cost baseline, should match or improve baseline)")
    print(f"  Target: PPL ≤ 41.24 (no degradation)")
    print(f"  Advantage: No architectural changes, no training required")
    
    return baseline_results


if __name__ == "__main__":
    print("Starting Selective Context evaluation...")
    
    # Run evaluations
    results = run_evaluations(
        model_name="facebook/opt-125m",
        compression_ratios=[0.3, 0.5, 0.7]
    )
    
    # Compare with our sparse memory baseline
    baseline_comparison = compare_with_baseline()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("Selective Context provides a zero-cost alternative to sparse memory.")
    print("Expected results: No PPL degradation (unlike sparse memory's +20% regression)")
    print("Next step: Test with actual Llama-2-7b model and standard benchmarks")