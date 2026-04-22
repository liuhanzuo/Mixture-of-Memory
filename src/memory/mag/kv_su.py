"""
KV-Level Self-Update Function.

Works directly on KV cache (per-head, per-layer):
    K/V shape: (B, num_kv_heads, T_mem, head_dim)

Update rule (per-layer, per-head shared):
    K' = gate * (W_kr @ K + W_ki @ h) + (1-gate) * K + K

Where h is the mean-pooled query hidden state projected to head_dim.

This keeps SU lightweight and compatible with GQA.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KVSelfUpdate(nn.Module):
    """SU that directly modifies KV cache entries.
    
    Per injection layer:
        K' = gate * (linear_retain(K) + linear_inject(h)) + (1-gate) * K + K
        V' = same structure
        
    All linear maps operate on head_dim (128 for Qwen3-8B).
    """
    
    def __init__(
        self,
        num_layers: int = 4,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        use_gate: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Per-layer: retain + inject for K and V
        # Using low-rank (rank=32) to keep params small
        rank = min(32, head_dim)
        self.su_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Key update
                'k_retain_A': nn.Linear(head_dim, rank, bias=False),
                'k_retain_B': nn.Linear(rank, head_dim, bias=False),
                'k_inject_A': nn.Linear(head_dim, rank, bias=False),
                'k_inject_B': nn.Linear(rank, head_dim, bias=False),
                # Value update
                'v_retain_A': nn.Linear(head_dim, rank, bias=False),
                'v_retain_B': nn.Linear(rank, head_dim, bias=False),
                'v_inject_A': nn.Linear(head_dim, rank, bias=False),
                'v_inject_B': nn.Linear(rank, head_dim, bias=False),
            })
            self.su_layers.append(layer)
        
        # Gates: one per layer (init to 1.0 so sigmoid(1)=0.73, inject path active)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        
        # Initialize: retain ≈ identity, inject ≈ 0
        self._init_weights()
        
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[KVSelfUpdate] layers={num_layers}, heads={num_kv_heads}, "
                    f"head_dim={head_dim}, rank={rank}, params={num_params:,}")
    
    def _init_weights(self):
        """Initialize retain ≈ identity path, inject with small random signal."""
        for layer in self.su_layers:
            # Retain: A @ B ≈ I (initialize B = I truncated, A = random small)
            nn.init.eye_(layer['k_retain_B'].weight[:min(layer['k_retain_B'].weight.shape[0], 
                                                          layer['k_retain_B'].weight.shape[1]), :])
            nn.init.eye_(layer['v_retain_B'].weight[:min(layer['v_retain_B'].weight.shape[0],
                                                          layer['v_retain_B'].weight.shape[1]), :])
            nn.init.normal_(layer['k_retain_A'].weight, std=0.02)
            nn.init.normal_(layer['v_retain_A'].weight, std=0.02)
            
            # Inject: small random init (NOT zero — zero means dead gradient)
            nn.init.normal_(layer['k_inject_B'].weight, std=0.01)
            nn.init.normal_(layer['v_inject_B'].weight, std=0.01)
            nn.init.normal_(layer['k_inject_A'].weight, std=0.02)
            nn.init.normal_(layer['v_inject_A'].weight, std=0.02)
    
    def forward(self, kv_cache: dict, query_hidden: dict) -> dict:
        """Update KV cache using SU.
        
        Args:
            kv_cache: {layer_idx: (K, V)} where K,V shape (B, num_kv_heads, T, head_dim)
            query_hidden: {layer_idx: h} where h shape (B, T_query, hidden_dim)
                          (used to derive injection signal)
        
        Returns:
            Updated kv_cache (same structure, with gradients through SU params)
        """
        updated = {}
        su_idx = 0
        
        for layer_idx in sorted(kv_cache.keys()):
            if su_idx >= self.num_layers:
                break
            K, V = kv_cache[layer_idx]  # (B, H, T, D)
            B, H, T, D = K.shape
            
            su_layer = self.su_layers[su_idx]
            gate = torch.sigmoid(self.gates[su_idx])
            
            # Reshape for per-head processing: (B*H, T, D)
            K_flat = K.reshape(B * H, T, D)
            V_flat = V.reshape(B * H, T, D)
            
            # Retain path
            K_retained = su_layer['k_retain_B'](su_layer['k_retain_A'](K_flat))
            V_retained = su_layer['v_retain_B'](su_layer['v_retain_A'](V_flat))
            
            # Inject path: use per-token KV statistics as context signal
            # Using std + mean captures more per-token variation than mean alone
            K_std = K_flat.std(dim=1, keepdim=True).expand_as(K_flat)
            V_std = V_flat.std(dim=1, keepdim=True).expand_as(V_flat)
            K_inject_input = K_flat + 0.1 * K_std  # Add variation signal
            V_inject_input = V_flat + 0.1 * V_std
            
            K_injected = su_layer['k_inject_B'](su_layer['k_inject_A'](K_inject_input))
            V_injected = su_layer['v_inject_B'](su_layer['v_inject_A'](V_inject_input))
            
            # Gated update with residual
            K_new = gate * (K_retained + K_injected) + (1 - gate) * K_flat + K_flat
            V_new = gate * (V_retained + V_injected) + (1 - gate) * V_flat + V_flat
            
            updated[layer_idx] = (
                K_new.reshape(B, H, T, D),
                V_new.reshape(B, H, T, D),
            )
            su_idx += 1
        
        return updated
