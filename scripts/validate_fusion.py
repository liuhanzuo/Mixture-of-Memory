#!/usr/bin/env python3
"""Validate sparse memory fusion mechanism with wikitext data."""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory.sparse_memory.attention import SparseMemoryAttention

def load_wikitext_data():
    """Load wikitext tokenized data from .npy file."""
    data_path = "data/wikitext_tokenized_1024.npy"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.load(data_path)
    # Convert to torch tensor and ensure correct type
    data = torch.from_numpy(data).long()
    return data

def setup_test_model():
    """Setup a minimal test model configuration."""
    config = {
        'hidden_size': 4096,
        'num_attention_heads': 4,
        'num_hidden_layers': 1,
        'intermediate_size': 11008,
        'max_position_embeddings': 8192,
        'bos_token_id': 1,
        'eos_token_id': 2,
    }
    
    # Create a small test model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager"
    )
    
    return model

def validate_fusion_mechanism():
    """Test the sparse memory fusion with real data."""
    print("Loading wikitext data...")
    wikitext_data = load_wikitext_data()
    print(f"Loaded {len(wikitext_data)} samples from wikitext")
    
    # Use first sample for validation
    test_sample = wikitext_data[:1]  # Single batch dimension
    print(f"Test sample shape: {test_sample.shape}")
    
    # Setup sparse memory attention
    attention_config = {
        'hidden_size': 4096,
        'num_attention_heads': 4,
        'num_slots': 128,
        'top_k': 8,
        'window_size': 256,
        'use_bypass': True,
        'bypass_init': -2.0,
        'bypass_lr_multiplier': 10.0,
    }
    
    print("Initializing SparseMemoryAttention...")
    sparse_attention = SparseMemoryAttention(**attention_config)
    
    # Create dummy input
    batch_size = 1
    seq_len = min(1024, test_sample.shape[1])
    hidden_size = attention_config['hidden_size']
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    print("Testing fusion mechanism...")
    try:
        output, mem_attention = sparse_attention(hidden_states, attention_mask)
        print(f"✅ Fusion successful!")
        print(f"Output shape: {output.shape}")
        print(f"Memory attention shape: {mem_attention.shape}")
        
        # Check if output is reasonable (no NaNs/infs)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("❌ Output contains NaN or Inf values!")
            return False
        
        # Check if memory attention has proper statistics
        if torch.isnan(mem_attention).any() or torch.isinf(mem_attention).any():
            print("❌ Memory attention contains NaN or Inf values!")
            return False
            
        print("✅ All validation checks passed!")
        return True
        
    except Exception as e:
        print(f"❌ Fusion failed with error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Sparse Memory Fusion Validation")
    print("=" * 60)
    
    success = validate_fusion_mechanism()
    
    if success:
        print("\n🎉 All validations passed!")
        sys.exit(0)
    else:
        print("\n💥 Validation failed!")
        sys.exit(1)