#!/usr/bin/env python3
"""Test that eval_rmt_locomo.py can be imported without errors."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Test imports
    from scripts.eval_rmt_locomo import (
        LoCoMoEvalConfig,
        load_locomo_data,
        build_locomo_context,
        load_rmt_model,
        generate_answer_with_rmt,
        evaluate_locomo_qa,
    )
    print("✓ All imports successful")
    print(f"✓ LoCoMoEvalConfig: {LoCoMoEvalConfig}")
    print(f"✓ Functions: load_locomo_data, build_locomo_context, load_rmt_model, generate_answer_with_rmt, evaluate_locomo_qa")

    # Test config creation
    config = LoCoMoEvalConfig(
        checkpoint_dir="outputs/rmt_v4_20260416_104930",
        locomo_data_path="locomo/data/locomo10.json",
        output_dir="test_output",
    )
    print(f"✓ Config created: {config}")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All import tests passed!")
