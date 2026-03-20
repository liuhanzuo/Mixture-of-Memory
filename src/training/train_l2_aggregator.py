"""
L2 聚合器训练器。

训练目标: 微调 LLM 后端，使其能从对话消息中准确提取结构化记忆对象。

训练策略:
1. 构造 (messages → expected L2 objects) 的训练数据对
2. 将消息序列化为 prompt，期望输出为 JSON 格式的 L2 对象
3. 使用标准 seq2seq / causal LM 微调

当前状态: 框架 + stub 实现
TODO: 接入真实 LLM 微调管线 (如 LoRA + PEFT)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from src.memory.l2.types import ChatMessage, L2MemoryObject

logger = logging.getLogger(__name__)


@dataclass
class L2AggregatorTrainConfig:
    """L2 聚合器训练配置。"""

    # 数据
    train_data_path: str = "data/processed/l2_train.jsonl"
    val_data_path: str = "data/processed/l2_val.jsonl"

    # 训练参数
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # 模型
    base_model: str = "Qwen/Qwen2-1.5B"  # 用于微调的基础模型
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32

    # 保存
    save_dir: str = "outputs/runs/l2_aggregator_training"
    save_every_epoch: bool = True

    # 日志
    log_every: int = 10


# --- 数据格式定义 ---

AGGREGATOR_SYSTEM_PROMPT = """You are a memory extraction assistant. Given a conversation, extract structured memory objects.

Each memory object should have:
- object_type: one of "topic", "preference", "task", "state", "entity"
- summary_text: a concise description
- confidence: a float between 0 and 1

Output as a JSON list of objects."""

AGGREGATOR_INPUT_TEMPLATE = """## Conversation:
{conversation}

## Extract memory objects (JSON list):"""


def format_messages_for_prompt(messages: list[dict[str, str]]) -> str:
    """将消息列表格式化为训练 prompt 的对话部分。"""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def format_objects_as_target(objects: list[dict[str, Any]]) -> str:
    """将期望的 L2 对象格式化为训练目标。"""
    simplified = []
    for obj in objects:
        simplified.append({
            "object_type": obj.get("object_type", "topic"),
            "summary_text": obj.get("summary_text", ""),
            "confidence": obj.get("confidence", 0.8),
        })
    return json.dumps(simplified, ensure_ascii=False, indent=2)


class L2AggregatorDataset(Dataset):
    """L2 聚合器训练数据集。

    数据格式 (JSONL):
    {
        "messages": [{"role": "user", "content": "..."}, ...],
        "objects": [{"object_type": "topic", "summary_text": "...", "confidence": 0.8}, ...]
    }
    """

    def __init__(self, data_path: str, max_seq_len: int = 2048):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.samples: list[dict[str, Any]] = []

        if self.data_path.exists():
            self._load_data()
        else:
            logger.warning(f"[L2 Dataset] 数据文件不存在: {data_path}，使用合成数据")
            self._generate_synthetic()

    def _load_data(self) -> None:
        """从 JSONL 文件加载数据。"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        logger.info(f"[L2 Dataset] 加载了 {len(self.samples)} 条训练样本")

    def _generate_synthetic(self) -> None:
        """生成合成训练数据。"""
        synthetic_samples = [
            {
                "messages": [
                    {"role": "user", "content": "I prefer using Python for data science."},
                    {"role": "assistant", "content": "Python is great for data science!"},
                    {"role": "user", "content": "I'm currently working on a memory system for LLM agents."},
                ],
                "objects": [
                    {"object_type": "preference", "summary_text": "using Python for data science", "confidence": 0.9},
                    {"object_type": "task", "summary_text": "working on a memory system for LLM agents", "confidence": 0.85},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "Can you help me understand transformer attention mechanisms?"},
                    {"role": "assistant", "content": "Sure! Transformer attention uses query, key, value..."},
                    {"role": "user", "content": "I always prefer detailed technical explanations with examples."},
                ],
                "objects": [
                    {"object_type": "topic", "summary_text": "transformer attention mechanisms", "confidence": 0.8},
                    {"object_type": "preference", "summary_text": "detailed technical explanations with examples", "confidence": 0.9},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "I'm researching sparse training for large language models."},
                    {"role": "assistant", "content": "Sparse training is an active area..."},
                    {"role": "user", "content": "Specifically, I'm looking at channel pruning and SLoRB."},
                ],
                "objects": [
                    {"object_type": "topic", "summary_text": "sparse training for large language models", "confidence": 0.9},
                    {"object_type": "state", "summary_text": "researching channel pruning and SLoRB", "confidence": 0.85},
                ],
            },
        ]
        # 复制合成样本以获得更多训练数据
        for _ in range(50):
            self.samples.extend(synthetic_samples)
        logger.info(f"[L2 Dataset] 生成了 {len(self.samples)} 条合成训练样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        sample = self.samples[idx]
        conversation = format_messages_for_prompt(sample["messages"])
        input_text = AGGREGATOR_INPUT_TEMPLATE.format(conversation=conversation)
        target_text = format_objects_as_target(sample["objects"])
        return {
            "input_text": input_text,
            "target_text": target_text,
            "full_text": f"{AGGREGATOR_SYSTEM_PROMPT}\n\n{input_text}\n{target_text}",
        }


class L2AggregatorTrainer:
    """L2 聚合器训练器。

    当前为 stub 实现，提供完整的训练框架但不实际执行 LLM 微调。

    TODO: 接入实际训练:
    1. 加载 HuggingFace 模型和 tokenizer
    2. (可选) 应用 LoRA / PEFT
    3. 构造 DataCollator
    4. 使用 HuggingFace Trainer 或自定义训练循环
    """

    def __init__(self, config: L2AggregatorTrainConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self) -> tuple[L2AggregatorDataset, L2AggregatorDataset | None]:
        """准备训练和验证数据集。"""
        train_dataset = L2AggregatorDataset(
            self.config.train_data_path,
            self.config.max_seq_len,
        )

        val_dataset = None
        if Path(self.config.val_data_path).exists():
            val_dataset = L2AggregatorDataset(
                self.config.val_data_path,
                self.config.max_seq_len,
            )

        return train_dataset, val_dataset

    def train(self) -> dict[str, Any]:
        """执行训练。

        Returns:
            训练结果摘要。
        """
        logger.info("[L2 AggregatorTrainer] 准备训练数据...")
        train_dataset, val_dataset = self.prepare_data()

        logger.info(f"[L2 AggregatorTrainer] 训练集大小: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"[L2 AggregatorTrainer] 验证集大小: {len(val_dataset)}")

        # --- TODO: 实际训练逻辑 ---
        # 以下为 stub，展示预期流程

        logger.warning(
            "[L2 AggregatorTrainer] 当前为 stub 实现。"
            "要接入真实 LLM 微调，请实现以下步骤:\n"
            "  1. 加载 base model 和 tokenizer\n"
            "  2. 应用 LoRA (如 peft.get_peft_model())\n"
            "  3. Tokenize 数据集\n"
            "  4. 使用 transformers.Trainer 或自定义循环训练\n"
            "  5. 保存微调后的适配器权重"
        )

        # 模拟训练结果
        results = {
            "status": "stub_completed",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "epochs": self.config.epochs,
            "base_model": self.config.base_model,
            "message": "Stub training completed. Replace with real LLM fine-tuning.",
        }

        # 保存训练配置记录
        config_path = self.save_dir / "train_config.json"
        import dataclasses
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2, ensure_ascii=False)

        logger.info(f"[L2 AggregatorTrainer] Stub 训练完成。配置已保存至 {config_path}")
        return results

    def evaluate(self, dataset: L2AggregatorDataset) -> dict[str, float]:
        """评估聚合器性能。

        TODO: 实现真实评估:
        1. 用训练后的模型生成 L2 对象
        2. 与 ground truth 对比
        3. 计算 precision / recall / F1

        Returns:
            评估指标。
        """
        logger.warning("[L2 AggregatorTrainer] evaluate() 为 stub 实现。")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "message": "Stub evaluation. Replace with real metrics.",
        }
