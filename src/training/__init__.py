"""
training: 各层记忆组件的训练脚本和工具。

包括:
- train_gate: L1 门控网络的训练
- train_l2_aggregator: L2 聚合器 (LLM-based) 的微调
- train_l3_summarizer: L3 总结器 (LLM-based) 的微调
"""

from .train_gate import L1GateTrainer
from .train_l2_aggregator import L2AggregatorTrainer
from .train_l3_summarizer import L3SummarizerTrainer

__all__ = [
    "L1GateTrainer",
    "L2AggregatorTrainer",
    "L3SummarizerTrainer",
]
