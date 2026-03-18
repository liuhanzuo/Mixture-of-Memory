"""
FusionHead: 将 backbone hidden state 与长期记忆 readout 融合的轻量级模块。

核心公式:
    g_t = sigmoid(W_g [h_t; r_t] + b_g)          # 学习门控
    h_tilde_t = h_t + g_t ⊙ (W_r r_t)            # 残差门控融合

设计要点:
    - 残差连接: 确保无记忆时退化为原始 backbone 输出
    - 学习门控: 模型自动决定记忆融合的强度
    - 轻量级: 仅包含两个线性层和一个激活函数
    - 初始化: 门控偏置初始化为负值，训练初期接近零融合（保护 backbone 质量）
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    """将 backbone hidden state 与 memory readout 融合的门控残差模块。
    
    在 v0 中，这是长期记忆影响生成的 **唯一** 入口：
    1. 将 memory readout r_t 通过线性投影到 hidden_dim
    2. 计算门控值 g_t（基于 h_t 和 r_t 的拼接）
    3. 通过残差连接得到融合后的 hidden state
    
    Args:
        hidden_dim: backbone 模型的隐藏层维度
        memory_dim: memory readout 的维度（默认与 hidden_dim 相同）
        gate_init_bias: 门控偏置的初始值，负值使训练初期门控接近 0
        dropout: dropout 概率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: Optional[int] = None,
        gate_init_bias: float = -2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim or hidden_dim
        self.gate_init_bias = gate_init_bias
        
        # --- 记忆投影: 将 memory readout 投影到 hidden_dim ---
        self.memory_proj = nn.Linear(self.memory_dim, hidden_dim, bias=False)
        
        # --- 门控网络: 基于 [h_t; r_t] 计算逐维度门控 ---
        self.gate_proj = nn.Linear(hidden_dim + self.memory_dim, hidden_dim)
        
        # --- Dropout ---
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # --- 初始化 ---
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """参数初始化策略。
        
        - memory_proj: 小幅初始化，避免训练初期记忆信号过强
        - gate_proj: 偏置初始化为负值，使 sigmoid 输出接近 0
          这确保训练初期融合模块几乎不影响 backbone 输出
        """
        # 记忆投影使用较小的初始化
        nn.init.normal_(self.memory_proj.weight, std=0.01)
        
        # 门控投影使用 Xavier 初始化
        nn.init.xavier_uniform_(self.gate_proj.weight)
        # 偏置初始化为负值 → sigmoid 接近 0 → 训练初期几乎不融合
        nn.init.constant_(self.gate_proj.bias, self.gate_init_bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_readout: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：将 backbone hidden 与 memory readout 融合。
        
        Args:
            hidden_states: backbone 输出的 hidden states, shape [B, T, D_h]
                          或 [B, D_h]（单 token 生成时）
            memory_readout: memory 的 readout 向量, shape [B, T, D_m]
                           或 [B, D_m]
        
        Returns:
            fused: 融合后的 hidden states, shape 与 hidden_states 相同
        """
        # 处理 2D 输入（单 token 场景）
        squeeze = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # [B, 1, D_h]
            memory_readout = memory_readout.unsqueeze(1)  # [B, 1, D_m]
            squeeze = True
        
        # 1. 投影 memory readout 到 hidden_dim
        # r_proj: [B, T, D_h]
        r_proj = self.memory_proj(memory_readout)
        r_proj = self.dropout(r_proj)
        
        # 2. 计算门控值
        # gate_input: [B, T, D_h + D_m]
        gate_input = torch.cat([hidden_states, memory_readout], dim=-1)
        # g_t: [B, T, D_h], 值域 (0, 1)
        g_t = torch.sigmoid(self.gate_proj(gate_input))
        
        # 3. 残差门控融合: h_tilde = h + g ⊙ W_r(r)
        fused = hidden_states + g_t * r_proj
        
        if squeeze:
            fused = fused.squeeze(1)  # [B, D_h]
        
        return fused
    
    def get_gate_stats(
        self,
        hidden_states: torch.Tensor,
        memory_readout: torch.Tensor,
    ) -> dict:
        """获取门控统计信息，用于日志和调试。
        
        Args:
            hidden_states: [B, T, D_h]
            memory_readout: [B, T, D_m]
        
        Returns:
            包含门控均值、标准差、最大/最小值的字典
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
            memory_readout = memory_readout.unsqueeze(1)
        
        gate_input = torch.cat([hidden_states, memory_readout], dim=-1)
        g_t = torch.sigmoid(self.gate_proj(gate_input))
        
        return {
            "gate_mean": g_t.mean().item(),
            "gate_std": g_t.std().item(),
            "gate_max": g_t.max().item(),
            "gate_min": g_t.min().item(),
        }
    
    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"memory_dim={self.memory_dim}, "
            f"gate_init_bias={self.gate_init_bias}"
        )
