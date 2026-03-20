"""
L3 总结器训练器。

训练目标: 微调 LLM 后端，使其能从 L2 记忆对象中生成高质量的
长期语义/画像记忆条目。

训练策略:
1. 构造 (L2 objects → expected L3 profile entries) 的训练数据对
2. 将 L2 对象序列化为 prompt，期望输出为结构化 profile 条目
3. 使用标准 causal LM 微调

当前状态: 框架 + stub 实现
TODO: 接入真实 LLM 微调管线
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from src.memory.l3.summarizer import L3ProfileEntry

logger = logging.getLogger(__name__)


@dataclass
class L3SummarizerTrainConfig:
    """L3 总结器训练配置。"""

    # 数据
    train_data_path: str = "data/processed/l3_train.jsonl"
    val_data_path: str = "data/processed/l3_val.jsonl"

    # 训练参数
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # 模型
    base_model: str = "Qwen/Qwen3-1.7B"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32

    # 保存
    save_dir: str = "outputs/runs/l3_summarizer_training"
    save_every_epoch: bool = True

    # 日志
    log_every: int = 10


# --- Prompt 模板 ---

SUMMARIZER_SYSTEM_PROMPT = """You are a user profile generation assistant. Given a set of episodic memory objects from a conversation, generate long-term profile entries that capture the user's stable traits, interests, and preferences.

Each profile entry should have:
- key: a short identifier (e.g., "research_interest", "preferred_style")
- value: a natural language description
- confidence: a float between 0 and 1
- category: one of "research_interest", "preference", "long_term_project", "identity", "factual"

Output as a JSON list."""

SUMMARIZER_INPUT_TEMPLATE = """## L2 Memory Objects:
{objects_text}

## Generate profile entries (JSON list):"""


def format_l2_objects_for_prompt(objects: list[dict[str, Any]]) -> str:
    """将 L2 对象列表格式化为 prompt 中的描述。"""
    lines = []
    for i, obj in enumerate(objects):
        obj_type = obj.get("object_type", "unknown")
        summary = obj.get("summary_text", "")
        confidence = obj.get("confidence", 0.5)
        lines.append(f"{i + 1}. [{obj_type}] {summary} (conf={confidence:.2f})")
    return "\n".join(lines)


def format_profile_entries_as_target(entries: list[dict[str, Any]]) -> str:
    """将期望的 L3 条目格式化为训练目标。"""
    simplified = []
    for entry in entries:
        simplified.append({
            "key": entry.get("key", ""),
            "value": entry.get("value", ""),
            "confidence": entry.get("confidence", 0.8),
            "category": entry.get("category", "factual"),
        })
    return json.dumps(simplified, ensure_ascii=False, indent=2)


class L3SummarizerDataset(Dataset):
    """L3 总结器训练数据集。

    数据格式 (JSONL):
    {
        "l2_objects": [{"object_type": "topic", "summary_text": "...", "confidence": 0.8}, ...],
        "profile_entries": [{"key": "...", "value": "...", "confidence": 0.9, "category": "..."}, ...]
    }
    """

    def __init__(self, data_path: str, max_seq_len: int = 2048):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.samples: list[dict[str, Any]] = []

        if self.data_path.exists():
            self._load_data()
        else:
            logger.warning(f"[L3 Dataset] 数据文件不存在: {data_path}，使用合成数据")
            self._generate_synthetic()

    def _load_data(self) -> None:
        """从 JSONL 文件加载数据。"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        logger.info(f"[L3 Dataset] 加载了 {len(self.samples)} 条训练样本")

    def _generate_synthetic(self) -> None:
        """生成合成训练数据。"""
        synthetic_samples = [
            {
                "l2_objects": [
                    {"object_type": "topic", "summary_text": "sparse training for LLMs", "confidence": 0.9},
                    {"object_type": "topic", "summary_text": "channel pruning techniques", "confidence": 0.85},
                    {"object_type": "state", "summary_text": "researching model compression", "confidence": 0.8},
                ],
                "profile_entries": [
                    {
                        "key": "research_interest",
                        "value": "The user is actively researching sparse training and model compression for large language models, with a focus on channel pruning.",
                        "confidence": 0.9,
                        "category": "research_interest",
                    },
                ],
            },
            {
                "l2_objects": [
                    {"object_type": "preference", "summary_text": "detailed technical explanations", "confidence": 0.9},
                    {"object_type": "preference", "summary_text": "structured responses with examples", "confidence": 0.85},
                    {"object_type": "preference", "summary_text": "using Python for implementations", "confidence": 0.8},
                ],
                "profile_entries": [
                    {
                        "key": "preferred_response_style",
                        "value": "The user prefers detailed, technical, and structured explanations with concrete examples.",
                        "confidence": 0.9,
                        "category": "preference",
                    },
                    {
                        "key": "preferred_language",
                        "value": "The user prefers Python for implementations and code examples.",
                        "confidence": 0.85,
                        "category": "preference",
                    },
                ],
            },
            {
                "l2_objects": [
                    {"object_type": "task", "summary_text": "building a hierarchical memory system for agents", "confidence": 0.9},
                    {"object_type": "task", "summary_text": "implementing associative matrix memory", "confidence": 0.85},
                    {"object_type": "topic", "summary_text": "agent memory architecture", "confidence": 0.8},
                ],
                "profile_entries": [
                    {
                        "key": "long_term_project",
                        "value": "The user's long-term project is building a hierarchical memory system (MoM) for LLM agents, featuring associative matrix memory for online context compensation.",
                        "confidence": 0.9,
                        "category": "long_term_project",
                    },
                ],
            },
        ]
        # 复制
        for _ in range(50):
            self.samples.extend(synthetic_samples)
        logger.info(f"[L3 Dataset] 生成了 {len(self.samples)} 条合成训练样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        sample = self.samples[idx]
        objects_text = format_l2_objects_for_prompt(sample["l2_objects"])
        input_text = SUMMARIZER_INPUT_TEMPLATE.format(objects_text=objects_text)
        target_text = format_profile_entries_as_target(sample["profile_entries"])
        return {
            "input_text": input_text,
            "target_text": target_text,
            "full_text": f"{SUMMARIZER_SYSTEM_PROMPT}\n\n{input_text}\n{target_text}",
        }


class L3SummarizerTrainer:
    """L3 总结器训练器。

    当前为 stub 实现，提供完整的训练框架。

    TODO: 接入真实 LLM 微调:
    1. 加载 HuggingFace 模型和 tokenizer
    2. (可选) 应用 LoRA / PEFT
    3. 构造 DataCollator
    4. 使用 HuggingFace Trainer 或自定义循环
    5. 保存微调后的权重
    """

    def __init__(self, config: L3SummarizerTrainConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self) -> tuple[L3SummarizerDataset, L3SummarizerDataset | None]:
        """准备训练和验证数据集。"""
        train_dataset = L3SummarizerDataset(
            self.config.train_data_path,
            self.config.max_seq_len,
        )

        val_dataset = None
        if Path(self.config.val_data_path).exists():
            val_dataset = L3SummarizerDataset(
                self.config.val_data_path,
                self.config.max_seq_len,
            )

        return train_dataset, val_dataset

    def train(self) -> dict[str, Any]:
        """执行训练。

        Returns:
            训练结果摘要。
        """
        logger.info("[L3 SummarizerTrainer] 准备训练数据...")
        train_dataset, val_dataset = self.prepare_data()

        logger.info(f"[L3 SummarizerTrainer] 训练集大小: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"[L3 SummarizerTrainer] 验证集大小: {len(val_dataset)}")

        # --- TODO: 实际训练逻辑 ---
        logger.warning(
            "[L3 SummarizerTrainer] 当前为 stub 实现。"
            "要接入真实 LLM 微调，请实现以下步骤:\n"
            "  1. 加载 base model 和 tokenizer\n"
            "  2. 应用 LoRA (如 peft.get_peft_model())\n"
            "  3. Tokenize 数据集\n"
            "  4. 使用 transformers.Trainer 或自定义循环训练\n"
            "  5. 保存微调后的适配器权重"
        )

        results = {
            "status": "stub_completed",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "epochs": self.config.epochs,
            "base_model": self.config.base_model,
            "message": "Stub training completed. Replace with real LLM fine-tuning.",
        }

        # 保存训练配置
        config_path = self.save_dir / "train_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

        logger.info(f"[L3 SummarizerTrainer] Stub 训练完成。配置已保存至 {config_path}")
        return results

    def evaluate(self, dataset: L3SummarizerDataset) -> dict[str, float]:
        """评估总结器性能。

        TODO: 实现真实评估:
        1. 用训练后的模型生成 L3 profile entries
        2. 与 ground truth 对比
        3. 计算语义相似度、覆盖率等指标

        Returns:
            评估指标。
        """
        logger.warning("[L3 SummarizerTrainer] evaluate() 为 stub 实现。")
        return {
            "coverage": 0.0,
            "precision": 0.0,
            "consistency": 0.0,
            "message": "Stub evaluation. Replace with real metrics.",
        }

    def generate_profile_from_l2(
        self,
        l2_objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """使用训练后的模型从 L2 对象生成 profile entries。

        TODO: 实现推理逻辑:
        1. 序列化 L2 对象为 prompt
        2. 调用微调后的模型
        3. 解析输出为 profile entries

        当前退回到规则方法。
        """
        logger.warning("[L3 SummarizerTrainer] generate_profile_from_l2() 为 stub 实现。")
        from src.memory.l3.summarizer import RuleBasedSummarizer, L3ProfileEntry
        from src.memory.l2.types import L2MemoryObject

        # 转换为 L2MemoryObject
        mem_objects = []
        for obj in l2_objects:
            mem_objects.append(L2MemoryObject(
                object_id=obj.get("object_id", "stub"),
                object_type=obj.get("object_type", "topic"),
                summary_text=obj.get("summary_text", ""),
                confidence=obj.get("confidence", 0.5),
                source_turn_ids=obj.get("source_turn_ids", []),
            ))

        summarizer = RuleBasedSummarizer()
        entries = summarizer.summarize(mem_objects)
        return [e.to_dict() for e in entries]
