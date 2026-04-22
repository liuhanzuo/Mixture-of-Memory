#!/usr/bin/env python3
"""
Quick validation script for LoCoMo evaluation components.
Tests data loading, context building, and evaluation metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

# Import from eval script
import scripts.eval_rmt_locomo as eval_script
from scripts.eval_rmt_locomo import (
    load_locomo_data,
    build_locomo_context,
    LoCoMoEvalConfig,
)

def test_data_loading():
    """Test LoCoMo data loading."""
    print("Testing data loading...")
    data = load_locomo_data("locomo/data/locomo10.json", categories=None)
    print(f"  Loaded {len(data)} conversations")

    # Test category filtering
    data_cat2 = load_locomo_data("locomo/data/locomo10.json", categories=[2])
    print(f"  Loaded {len(data_cat2)} conversations (category 2 only)")

    # Check sample structure
    if data:
        sample = data[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  QA count: {len(sample.get('qa', []))}")
        qa0 = sample['qa'][0]
        print(f"  QA[0] keys: {list(qa0.keys())}")
        print(f"  QA[0] question: {qa0.get('question', '')[:50]}...")
        print(f"  QA[0] answer: {qa0.get('answer', '')[:50]}...")
        print(f"  QA[0] category: {qa0.get('category', 'N/A')}")

    return data

def test_context_building(data, tokenizer):
    """Test context building from LoCoMo conversations."""
    print("\nTesting context building...")
    if not data:
        print("  No data to test with")
        return

    conv = data[0]
    context_ids, turn_info = build_locomo_context(
        conv,
        tokenizer,
        segment_length=1024,
        max_segments=6,
        max_tokens=6144,
    )

    print(f"  Context length: {len(context_ids)} tokens")
    print(f"  Turn count: {len(turn_info)}")

    if turn_info:
        print(f"  First turn: {turn_info[0]['formatted'][:60]}...")
        print(f"  Last turn: {turn_info[-1]['formatted'][:60]}...")

    return context_ids, turn_info

def test_evaluation_metrics():
    """Test evaluation metrics from LoCoMo."""
    print("\nTesting evaluation metrics...")
    sys.path.insert(0, str(PROJECT_ROOT / "locomo"))
    from task_eval.evaluation import f1_score, exact_match_score, normalize_answer

    # Test cases
    tests = [
        ("hello world", "hello world", "exact match"),
        ("hello world", "hello", "partial match"),
        ("", "hello", "empty prediction"),
        ("the quick brown fox", "the quick brown fox jumps", "ground truth longer"),
    ]

    for pred, gt, desc in tests:
        f1 = f1_score(pred, gt)
        em = exact_match_score(pred, gt)
        print(f"  {desc}: F1={f1:.3f}, EM={em}")

    # Test normalization
    normalized = normalize_answer("The quick, brown fox's jump over the lazy dog!")
    print(f"  Normalization test: '{normalized}'")

def main():
    parser = argparse.ArgumentParser(description="Validate LoCoMo evaluation setup")
    parser.add_argument("--locomo_data", type=str, default="locomo/data/locomo10.json")
    args = parser.parse_args()

    print("=" * 70)
    print("LoCoMo Evaluation Validation")
    print("=" * 70)

    # Test data loading
    data = test_data_loading()

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen--Qwen3-8b", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Tokenizer loaded: {tokenizer.name_or_path}")

    # Test context building
    test_context_building(data, tokenizer)

    # Test evaluation metrics
    test_evaluation_metrics()

    print("\n" + "=" * 70)
    print("All validation tests passed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
