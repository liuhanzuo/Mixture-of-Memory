"""
LongMemEval 数据加载器 (Scaffold)。

为 LongMemEval 长期记忆评估基准提供数据加载接口。
v0 仅提供框架结构，待后续完善。

LongMemEval 评估 LLM 在长对话中的记忆能力:
  - 信息提取 (Information Extraction)
  - 多会话推理 (Multi-Session Reasoning)
  - 时序理解 (Temporal Understanding)
  - 知识更新 (Knowledge Update)
  - 拒答能力 (Abstention)

参考: https://github.com/xiaowu0162/LongMemEval
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LongMemEvalDataset(Dataset):
    """LongMemEval 基准数据集加载器 (Scaffold)。

    v0 提供接口定义，实际数据加载逻辑待完善。

    Args:
        data_dir: LongMemEval 数据目录路径。
        split: 数据集切分 ("test")。
        category: 评估类别 (None 表示全部)。
        max_samples: 最大样本数。
        tokenizer: HuggingFace tokenizer。
        max_context_length: 最大上下文长度。
    """

    CATEGORIES = [
        "information_extraction",
        "multi_session_reasoning",
        "temporal_understanding",
        "knowledge_update",
        "abstention",
    ]

    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        category: Optional[str] = None,
        max_samples: int = 0,
        tokenizer: Any = None,
        max_context_length: int = 16384,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.category = category
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length

        if category is not None and category not in self.CATEGORIES:
            logger.warning(
                f"未知 LongMemEval 类别: {category}, "
                f"可选: {self.CATEGORIES}"
            )

        self.samples: List[Dict[str, Any]] = []
        self._load_data(max_samples)

    def _load_data(self, max_samples: int) -> None:
        """加载 LongMemEval 数据。

        预期目录结构:
          data_dir/
            {split}.jsonl
        或:
          data_dir/
            {category}/
              {split}.jsonl
        """
        # 尝试按类别加载
        if self.category is not None:
            data_file = self.data_dir / self.category / f"{self.split}.jsonl"
        else:
            data_file = self.data_dir / f"{self.split}.jsonl"

        if not data_file.exists():
            logger.warning(
                f"LongMemEval 数据文件不存在: {data_file}，"
                f"请先下载并准备 LongMemEval 数据集。"
            )
            return

        with open(data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    break
                sample = json.loads(line.strip())
                # 按类别过滤
                if (
                    self.category is not None
                    and sample.get("category") != self.category
                ):
                    continue
                self.samples.append(sample)

        logger.info(
            f"加载 LongMemEval [{self.category or 'all'}] "
            f"split={self.split}: {len(self.samples)} 样本"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回单条样本。

        预期 JSONL 格式:
        {
            "conversation_history": [...],  # 对话历史
            "query": "...",                 # 查询问题
            "answer": "...",                # 正确答案
            "category": "...",              # 评估类别
            "metadata": {...}               # 额外元信息
        }

        Returns:
            dict 包含 text, query, answer, category, 以及可选的 tokenized 结果。
        """
        sample = self.samples[idx]

        # 拼接对话历史为连续文本
        history = sample.get("conversation_history", [])
        if isinstance(history, list):
            context = "\n".join(
                f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                for turn in history
                if isinstance(turn, dict)
            )
        else:
            context = str(history)

        query = sample.get("query", "")
        full_text = f"{context}\n\nQuestion: {query}"

        item: Dict[str, Any] = {
            "text": full_text,
            "context": context,
            "query": query,
            "answer": sample.get("answer", ""),
            "category": sample.get("category", "unknown"),
            "metadata": sample.get("metadata", {}),
        }

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                full_text,
                max_length=self.max_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return item


def build_longmemeval_datasets(
    data_dir: str,
    categories: Optional[List[str]] = None,
    tokenizer: Any = None,
    max_samples: int = 0,
    max_context_length: int = 16384,
) -> Dict[str, LongMemEvalDataset]:
    """构建多类别的 LongMemEval 数据集。

    Args:
        data_dir: LongMemEval 数据根目录。
        categories: 评估类别列表 (None 表示全部)。
        tokenizer: HuggingFace tokenizer。
        max_samples: 每个类别的最大样本数。
        max_context_length: 最大上下文长度。

    Returns:
        datasets: {category: dataset} 字典。
    """
    if categories is None:
        categories = LongMemEvalDataset.CATEGORIES

    datasets: Dict[str, LongMemEvalDataset] = {}
    for cat in categories:
        datasets[cat] = LongMemEvalDataset(
            data_dir=data_dir,
            category=cat,
            max_samples=max_samples,
            tokenizer=tokenizer,
            max_context_length=max_context_length,
        )

    return datasets
