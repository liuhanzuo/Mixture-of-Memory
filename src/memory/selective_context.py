"""
Selective Context Implementation
Based on "Selective Context: Context Selection for Long-context Language Models" by Li et al., 2023

This implementation provides prompt compression by pruning redundant tokens at inference time.
No training required - works as a preprocessing step before feeding to the model.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class SelectiveContext:
    """
    Selective Context compressor for long context prompts.
    
    Compresses context by pruning redundant tokens while maintaining performance.
    Based on the principle that compression works best as a pre-processing step,
    not architectural modification.
    """
    
    def __init__(self, 
                 compression_ratio: float = 0.5,
                 method: str = "attention_entropy",
                 window_size: int = 256,
                 entropy_threshold: float = 0.8):
        """
        Args:
            compression_ratio: Target compression ratio (0.5 = keep 50% of tokens)
            method: Method for token selection ("attention_entropy", "random", "importance")
            window_size: Window size for local attention-based methods
            entropy_threshold: Threshold for entropy-based selection
        """
        self.compression_ratio = compression_ratio
        self.method = method
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        
    def compress(self, 
                 input_ids: torch.Tensor, 
                 attention_mask: Optional[torch.Tensor] = None,
                 hidden_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress input context by pruning redundant tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            hidden_states: Optional hidden states from previous model pass
            
        Returns:
            compressed_ids: Compressed token IDs [batch_size, compressed_length]
            compressed_mask: Compressed attention mask [batch_size, compressed_length]
        """
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        if seq_length <= self.window_size:
            # Don't compress very short sequences
            return input_ids, attention_mask
            
        if self.method == "random":
            return self._random_compress(input_ids, attention_mask)
        elif self.method == "importance":
            return self._importance_compress(input_ids, attention_mask)
        elif self.method == "attention_entropy":
            if hidden_states is None:
                # Fallback to random if no hidden states provided
                return self._random_compress(input_ids, attention_mask)
            return self._attention_entropy_compress(input_ids, attention_mask, hidden_states)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")
    
    def _random_compress(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random token selection (baseline method)."""
        batch_size, seq_length = input_ids.shape
        target_length = max(1, int(seq_length * self.compression_ratio))
        
        compressed_ids = []
        compressed_masks = []
        
        for i in range(batch_size):
            mask = attention_mask[i]
            valid_length = mask.sum().item()
            
            if valid_length <= target_length:
                # Keep all tokens if sequence is already short enough
                compressed_ids.append(input_ids[i])
                compressed_masks.append(mask)
                continue
                
            # Random selection
            valid_indices = torch.where(mask)[0]
            selected_indices = valid_indices[torch.randperm(valid_indices.shape[0])[:target_length]]
            selected_indices = selected_indices.sort().values
            
            compressed_ids.append(input_ids[i][selected_indices])
            compressed_masks.append(mask[selected_indices])
            
        return torch.stack(compressed_ids), torch.stack(compressed_masks)
    
    def _importance_compress(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Importance-based compression using simple heuristics."""
        batch_size, seq_length = input_ids.shape
        target_length = max(1, int(seq_length * self.compression_ratio))
        
        compressed_ids = []
        compressed_masks = []
        
        for i in range(batch_size):
            mask = attention_mask[i]
            valid_length = mask.sum().item()
            
            if valid_length <= target_length:
                # Keep all tokens if sequence is already short enough
                compressed_ids.append(input_ids[i])
                compressed_masks.append(mask)
                continue
                
            # Simple heuristics:
            # 1. Keep first and last tokens (preserve beginning and context)
            # 2. Keep tokens at regular intervals
            # 3. Special tokens ([CLS], [SEP], etc.)
            
            valid_indices = torch.where(mask)[0]
            
            # Keep first and last few tokens
            keep_start = min(10, valid_length // 4)
            keep_end = min(10, valid_length // 4)
            
            keep_indices = []
            keep_indices.extend(valid_indices[:keep_start])
            keep_indices.extend(valid_indices[-keep_end:])
            
            # Fill remaining with regular sampling
            remaining_slots = target_length - len(keep_indices)
            if remaining_slots > 0:
                step = (valid_length - keep_start - keep_end) / remaining_slots
                for j in range(remaining_slots):
                    pos = keep_start + int(j * step)
                    keep_indices.append(valid_indices[pos])
            
            keep_indices = sorted(list(set(keep_indices)))  # Remove duplicates
            keep_indices = torch.tensor(keep_indices, device=input_ids.device)
            
            compressed_ids.append(input_ids[i][keep_indices])
            compressed_masks.append(mask[keep_indices])
            
        return torch.stack(compressed_ids), torch.stack(compressed_masks)
    
    def _attention_entropy_compress(self, 
                                   input_ids: torch.Tensor, 
                                   attention_mask: torch.Tensor, 
                                   hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention entropy-based compression."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        target_length = max(1, int(seq_length * self.compression_ratio))
        
        compressed_ids = []
        compressed_masks = []
        
        for i in range(batch_size):
            mask = attention_mask[i]
            valid_length = mask.sum().item()
            
            if valid_length <= target_length:
                # Keep all tokens if sequence is already short enough
                compressed_ids.append(input_ids[i])
                compressed_masks.append(mask)
                continue
                
            # Calculate attention entropy for each token
            token_entropy = self._compute_token_entropy(hidden_states[i], mask)
            
            # Select tokens with lowest entropy (most "informative" tokens)
            valid_indices = torch.where(mask)[0]
            entropy_scores = token_entropy[valid_indices]
            
            # Keep tokens with entropy below threshold or lowest entropy tokens
            if self.entropy_threshold > 0:
                low_entropy_mask = entropy_scores < self.entropy_threshold
                selected_indices = valid_indices[low_entropy_mask]
                
                if len(selected_indices) < target_length:
                    # Fill remaining with lowest entropy tokens
                    remaining_indices = valid_indices[~low_entropy_mask]
                    remaining_scores = entropy_scores[~low_entropy_mask]
                    _, top_k_indices = torch.topk(remaining_scores, target_length - len(selected_indices))
                    selected_indices = torch.cat([selected_indices, remaining_indices[top_k_indices]])
            else:
                # Select lowest entropy tokens
                _, top_k_indices = torch.topk(token_entropy, target_length)
                selected_indices = valid_indices[top_k_indices]
            
            compressed_ids.append(input_ids[i][selected_indices])
            compressed_masks.append(mask[selected_indices])
            
        return torch.stack(compressed_ids), torch.stack(compressed_masks)
    
    def _compute_token_entropy(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy for each token."""
        seq_length = hidden_states.shape[0]
        
        # Compute self-attention weights
        attn_scores = torch.matmul(hidden_states, hidden_states.t()) / math.sqrt(hidden_states.shape[-1])
        
        # Apply mask
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Convert to probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1)
        
        return entropy
    
    def get_compression_stats(self, original_length: int, compressed_length: int) -> dict:
        """Get compression statistics."""
        compression_ratio = compressed_length / original_length
        space_saved = original_length - compressed_length
        
        return {
            'original_length': original_length,
            'compressed_length': compressed_length,
            'compression_ratio': compression_ratio,
            'space_saved': space_saved,
            'space_saved_percent': (space_saved / original_length) * 100
        }


class SelectiveContextWrapper:
    """
    Wrapper to integrate Selective Context into existing training/inference pipeline.
    
    This can be used as a preprocessing step before feeding data to the model.
    """
    
    def __init__(self, config):
        self.config = config
        self.compressor = SelectiveContext(
            compression_ratio=getattr(config, 'selective_context_ratio', 0.5),
            method=getattr(config, 'selective_context_method', 'attention_entropy'),
            window_size=getattr(config, 'selective_context_window_size', 256),
            entropy_threshold=getattr(config, 'selective_context_entropy_threshold', 0.8)
        )
    
    def pre_process_inputs(self, input_ids, attention_mask=None, hidden_states=None):
        """Apply selective context compression as preprocessing step."""
        return self.compressor.compress(input_ids, attention_mask, hidden_states)
    
    def apply_to_batch(self, batch):
        """Apply selective context to a training batch."""
        if 'input_ids' in batch:
            compressed_ids, compressed_mask = self.compressor.compress(
                batch['input_ids'], 
                batch.get('attention_mask')
            )
            batch['input_ids'] = compressed_ids
            batch['attention_mask'] = compressed_mask
        return batch
    
    def apply_to_inputs(self, inputs):
        """Apply selective context to model inputs (same as apply_to_batch)."""
        return self.apply_to_batch(inputs)


# Utility functions for evaluation
def evaluate_compression_effectiveness(model, tokenizer, test_prompts, compression_ratios=[0.3, 0.5, 0.7]):
    """
    Evaluate different compression ratios on test prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        test_prompts: List of test prompts
        compression_ratios: List of compression ratios to test
        
    Returns:
        results: Dictionary with compression statistics
    """
    results = {}
    compressor = SelectiveContext()
    
    for ratio in compression_ratios:
        compressor.compression_ratio = ratio
        
        for prompt in test_prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            original_length = inputs['input_ids'].shape[1]
            
            # Compress
            compressed_ids, compressed_mask = compressor.compress(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
            compressed_length = compressed_ids.shape[1]
            
            # Get compression stats
            stats = compressor.get_compression_stats(original_length, compressed_length)
            
            if ratio not in results:
                results[ratio] = []
            results[ratio].append(stats)
    
    return results