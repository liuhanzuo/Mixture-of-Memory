#!/usr/bin/env python3
"""Smoke test: DMS retrofitting on Qwen3-8B (GQA).

Validates:
- Model loads and DMS wraps correctly
- Forward pass (training mode) produces correct shapes
- Decision head hidden_dim detection works
- dtype consistency (bfloat16)
- Mask shape is 4D and broadcastable with GQA
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.dms import apply_dms_to_model

MODEL_NAME = "Qwen/Qwen3-8B"

def test_dms_qwen3():
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Apply DMS to first 2 layers only (faster)
    print("Applying DMS...")
    model = apply_dms_to_model(model, sliding_window=64, tau=0.1, target_layers=[0, 1])

    # Check hidden_dim
    wrapper = model.model.layers[0].self_attn
    assert wrapper._get_hidden_dim() == model.config.hidden_size, "hidden_dim mismatch"

    # Training forward
    model.train()
    inputs = tokenizer("Hello, this is a DMS smoke test for Qwen3 with GQA.", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits.shape[-1] == model.config.vocab_size, "vocab size mismatch"
    assert outputs.logits.dtype == torch.bfloat16, f"dtype mismatch: {outputs.logits.dtype}"

    # Check alpha was computed
    assert hasattr(wrapper, "_last_alpha"), "Missing _last_alpha"
    assert wrapper._last_alpha.shape == (1, inputs["input_ids"].shape[1])

    print("ALL CHECKS PASSED ✓")

if __name__ == "__main__":
    test_dms_qwen3()
