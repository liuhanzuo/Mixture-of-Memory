"""
RULER 数据加载器 (Scaffold)。

为 RULER 长序列基准测试提供数据加载接口。
v0 仅提供框架结构，待后续完善。

RULER 包含以下任务类型:
  - Needle-in-a-Haystack (NIAH)
  - Variable Tracking (VT)
  - Common/Frequent Words (CW/FW)
  - Question Answering (QA)

参考: https://github.com/hsiehjackson/RULER
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RulerDataset(Dataset):
    """RULER 基准数据集加载器 (Scaffold)。

    v0 提供接口定义，实际数据加载逻辑待完善。

    Args:
        data_dir: RULER 数据目录路径。
        task_type: 任务类型，如 "niah_single", "vt", "cwe", "fwe", "qa"。
        context_length: 上下文长度（RULER 支持不同长度）。
        max_samples: 最大样本数（用于调试）。
        tokenizer: HuggingFace tokenizer。
    """

    # 支持的任务类型
    TASK_TYPES = [
        "niah_single",      # 单 needle
        "niah_multikey",     # 多 key needle
        "niah_multivalue",   # 多 value needle
        "niah_multiquery",   # 多 query needle
        "vt",                # Variable Tracking
        "cwe",               # Common Words Extraction
        "fwe",               # Frequent Words Extraction
        "qa_1",              # Question Answering (1 hop)
        "qa_2",              # Question Answering (2 hop)
    ]

    CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

    def __init__(
        self,
        data_dir: str,
        task_type: str = "niah_single",
        context_length: int = 4096,
        max_samples: int = 0,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.context_length = context_length
        self.tokenizer = tokenizer

        if task_type not in self.TASK_TYPES:
            logger.warning(
                f"未知 RULER 任务类型: {task_type}, "
                f"可选: {self.TASK_TYPES}"
            )

        # 加载数据
        self.samples: List[Dict[str, Any]] = []
        self._load_data(max_samples)

    def _load_data(self, max_samples: int) -> None:
        """加载 RULER 数据。

        预期目录结构:
          data_dir/
            {task_type}/
              {context_length}.jsonl
        """
        data_file = (
            self.data_dir / self.task_type / f"{self.context_length}.jsonl"
        )

        if not data_file.exists():
            logger.warning(
                f"RULER 数据文件不存在: {data_file}，"
                f"请先下载并准备 RULER 数据集。"
            )
            return

        with open(data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    break
                sample = json.loads(line.strip())
                self.samples.append(sample)

        logger.info(
            f"加载 RULER [{self.task_type}] "
            f"ctx={self.context_length}: {len(self.samples)} 样本"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回单条样本。

        预期 JSONL 格式:
        {
            "input": "...",     # 输入上下文
            "outputs": ["..."], # 正确答案列表
            "task": "..."       # 任务类型
        }

        Returns:
            dict 包含 text, answers, task_type, 以及可选的 tokenized 结果。
        """
        sample = self.samples[idx]

        item: Dict[str, Any] = {
            "text": sample.get("input", ""),
            "answers": sample.get("outputs", []),
            "task_type": sample.get("task", self.task_type),
        }

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                item["text"],
                max_length=self.context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return item


def build_ruler_datasets(
    data_dir: str,
    task_types: Optional[List[str]] = None,
    context_lengths: Optional[List[int]] = None,
    tokenizer: Any = None,
    max_samples: int = 0,
) -> Dict[str, Dict[int, RulerDataset]]:
    """构建多任务、多长度的 RULER 数据集。

    Args:
        data_dir: RULER 数据根目录。
        task_types: 任务类型列表。
        context_lengths: 上下文长度列表。
        tokenizer: HuggingFace tokenizer。
        max_samples: 每个配置的最大样本数。

    Returns:
        datasets: {task_type: {context_length: dataset}} 嵌套字典。
    """
    if task_types is None:
        task_types = RulerDataset.TASK_TYPES
    if context_lengths is None:
        context_lengths = [4096]

    datasets: Dict[str, Dict[int, RulerDataset]] = {}
    for task in task_types:
        datasets[task] = {}
        for ctx_len in context_lengths:
            datasets[task][ctx_len] = RulerDataset(
                data_dir=data_dir,
                task_type=task,
                context_length=ctx_len,
                max_samples=max_samples,
                tokenizer=tokenizer,
            )

    return datasets
