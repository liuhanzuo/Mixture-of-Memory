#!/usr/bin/env python3
"""
Test script for Selective Context implementation.
This provides a zero-cost baseline for testing prompt compression effectiveness.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.selective_context import SelectiveContext, SelectiveContextWrapper
import transformers


def test_selective_context():
    """Test Selective Context with a small example."""
    print("Testing Selective Context implementation...")
    
    # Create a simple tokenizer for testing
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts - typical long context scenarios
    test_prompts = [
        "The quick brown fox jumps over the lazy dog. " * 20,  # 400 tokens
        "In machine learning, attention mechanisms have become increasingly important. " * 25,  # 500 tokens
        "The history of artificial intelligence dates back to ancient times when philosophers. " * 30,  # 600 tokens
    ]
    
    compression_ratios = [0.3, 0.5, 0.7]
    
    for ratio in compression_ratios:
        print(f"\n=== Testing Compression Ratio: {ratio} ===")
        compressor = SelectiveContext(compression_ratio=ratio, method="importance")
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}:")
            print(f"Original length: {len(prompt)} chars")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            original_length = inputs['input_ids'].shape[1]
            print(f"Original tokens: {original_length}")
            
            # Compress
            compressed_ids, compressed_mask = compressor.compress(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
            compressed_length = compressed_ids.shape[1]
            
            # Decode and compare
            compressed_text = tokenizer.decode(compressed_ids[0], skip_special_tokens=True)
            
            # Get stats
            stats = compressor.get_compression_stats(original_length, compressed_length)
            
            print(f"Compressed tokens: {compressed_length}")
            print(f"Compression ratio: {stats['compression_ratio']:.2f}")
            print(f"Space saved: {stats['space_saved']} tokens ({stats['space_saved_percent']:.1f}%)")
            print(f"Original preview: {prompt[:100]}...")
            print(f"Compressed preview: {compressed_text[:100]}...")
            print("-" * 50)


def test_method_comparison():
    """Compare different compression methods."""
    print("\n" + "="*60)
    print("COMPRESSION METHOD COMPARISON")
    print("="*60)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a long test prompt
    long_prompt = "The history of artificial intelligence dates back to ancient times when philosophers first began to ponder the nature of thought and consciousness. " * 50
    inputs = tokenizer(long_prompt, return_tensors="pt", truncation=False)
    
    methods = ["random", "importance"]
    
    print(f"Original length: {inputs['input_ids'].shape[1]} tokens")
    
    for method in methods:
        print(f"\n--- Method: {method} ---")
        compressor = SelectiveContext(compression_ratio=0.5, method=method)
        
        compressed_ids, compressed_mask = compressor.compress(
            inputs['input_ids'], 
            inputs['attention_mask']
        )
        
        compressed_text = tokenizer.decode(compressed_ids[0], skip_special_tokens=True)
        stats = compressor.get_compression_stats(
            inputs['input_ids'].shape[1], 
            compressed_ids.shape[1]
        )
        
        print(f"Compressed: {stats['compressed_length']} tokens")
        print(f"Ratio: {stats['compression_ratio']:.2f}")
        print(f"Preview: {compressed_text[:150]}...")


def test_integration_dummy():
    """Test integration with dummy model pass."""
    print("\n" + "="*60)
    print("INTEGRATION TEST WITH DUMMY MODEL")
    print("="*60)
    
    # Create a simple dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size=50257, hidden_dim=768):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
            self.lm_head = torch.nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, input_ids, attention_mask=None):
            hidden = self.embed(input_ids)
            logits = self.lm_head(hidden)
            
            # Dummy loss calculation
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return {'loss': loss, 'logits': logits}
    
    model = DummyModel()
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create wrapper
    class Config:
        selective_context_ratio = 0.5
        selective_context_method = "importance"
    
    config = Config()
    wrapper = SelectiveContextWrapper(config)
    
    # Test batch
    test_text = "This is a test of the selective context compression system. " * 20
    inputs = tokenizer(test_text, return_tensors="pt", truncation=False)
    
    print(f"Original batch shape: {inputs['input_ids'].shape}")
    
    # Apply selective context
    compressed_batch = wrapper.apply_to_inputs(inputs)
    print(f"Compressed batch shape: {compressed_batch['input_ids'].shape}")
    
    # Run through model
    with torch.no_grad():
        outputs = model(compressed_batch['input_ids'], compressed_batch['attention_mask'])
    
    print(f"Model output loss: {outputs['loss']:.4f}")
    print("Integration test completed successfully!")


if __name__ == "__main__":
    test_selective_context()
    test_method_comparison()
    test_integration_dummy()
    print("\n" + "="*60)
    print("All tests completed!")
    print("Selective Context implementation is ready for evaluation.")
    print("Next step: Evaluate against baseline PPL on standard benchmarks.")