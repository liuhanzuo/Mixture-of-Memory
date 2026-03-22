"""
MemoryEncoder: 将 L2/L3 文本记忆编码为 hidden-space 向量。

使用 backbone (如 Qwen3) 的 embedding 层对记忆文本进行编码，
通过 mean pooling 得到固定维度的记忆向量表示。

设计要点:
- 共享 backbone 的 embedding 层，不引入额外 embedding 参数
- 支持 L2MemoryObject 和 L3ProfileEntry 两种输入
- 对每条记忆生成一个 (hidden_dim,) 向量
- 支持批量编码以提高效率
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from src.memory.l2.types import L2MemoryObject
from src.memory.l3.summarizer import L3ProfileEntry

logger = logging.getLogger(__name__)


@dataclass
class MemoryEncoderConfig:
    """MemoryEncoder 配置。

    Attributes:
        max_memory_tokens: 每条记忆文本的最大 token 数.
        pooling: 池化策略, "mean" | "last" | "first".
        projection_dim: 如果 > 0, 则在 embedding 后加一个线性投影.
                        设为 0 则直接使用 embedding 维度 (= hidden_dim).
        dropout: 编码后的 dropout 比例.
    """
    max_memory_tokens: int = 64
    pooling: str = "mean"
    projection_dim: int = 0
    dropout: float = 0.0


class MemoryEncoder(nn.Module):
    """将 L2/L3 文本记忆编码为 backbone hidden-space 向量。

    使用 backbone 的 embedding 层 + mean pooling 生成记忆向量。
    可选地通过一个轻量线性投影层进行维度调整。

    Usage::

        encoder = MemoryEncoder(config)
        encoder.set_backbone(backbone_model, tokenizer)

        # 编码一批记忆对象
        l2_objs = [L2MemoryObject(...), ...]
        l3_entries = [L3ProfileEntry(...), ...]
        memory_vecs = encoder.encode(l2_objects=l2_objs, l3_entries=l3_entries)
        # → (num_memories, hidden_dim)
    """

    def __init__(self, config: MemoryEncoderConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = MemoryEncoderConfig()
        elif isinstance(config, dict):
            config = MemoryEncoderConfig(**{
                k: v for k, v in config.items()
                if k in MemoryEncoderConfig.__dataclass_fields__
            })
        self.config = config

        # 延迟初始化 (需要 backbone 信息)
        self._embedding: nn.Embedding | None = None
        self._tokenizer: Any = None
        self._hidden_dim: int = 0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

        # 可选的投影层
        self._projection: nn.Linear | None = None
        self._dropout: nn.Dropout | None = None
        if config.dropout > 0:
            self._dropout = nn.Dropout(config.dropout)

    @property
    def output_dim(self) -> int:
        """编码输出维度。"""
        if self.config.projection_dim > 0:
            return self.config.projection_dim
        return self._hidden_dim

    def set_backbone(
        self,
        backbone_model: nn.Module,
        tokenizer: Any,
        hidden_dim: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """从 backbone 提取 embedding 层并绑定 tokenizer。

        Args:
            backbone_model: HuggingFace CausalLM 或 _DebugTransformer。
            tokenizer: HuggingFace tokenizer.
            hidden_dim: 手动指定 hidden_dim (不指定则自动检测).
            device: 设备.
            dtype: 数据类型.
        """
        self._tokenizer = tokenizer

        if device is not None:
            self._device = torch.device(device) if isinstance(device, str) else device
        if dtype is not None:
            self._dtype = dtype

        # 提取 embedding 层
        self._embedding = self._extract_embedding(backbone_model)

        # 检测 hidden_dim
        if hidden_dim is not None:
            self._hidden_dim = hidden_dim
        elif self._embedding is not None:
            self._hidden_dim = self._embedding.embedding_dim
        else:
            raise ValueError("无法检测 hidden_dim，请手动指定")

        # 初始化投影层
        if self.config.projection_dim > 0:
            self._projection = nn.Linear(
                self._hidden_dim, self.config.projection_dim, bias=False
            ).to(device=self._device, dtype=self._dtype)

        logger.info(
            f"[MemoryEncoder] 绑定 backbone: hidden_dim={self._hidden_dim}, "
            f"output_dim={self.output_dim}, pooling={self.config.pooling}, "
            f"max_tokens={self.config.max_memory_tokens}"
        )

    def _extract_embedding(self, model: nn.Module) -> nn.Embedding | None:
        """从 HuggingFace 模型或 Debug 模型中提取 embedding 层。"""
        # HuggingFace CausalLM 结构: model.model.embed_tokens
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            emb = model.model.embed_tokens
            logger.info(f"[MemoryEncoder] 提取 HF embedding: {emb.num_embeddings} × {emb.embedding_dim}")
            return emb

        # 部分 HF 模型: model.transformer.wte
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            emb = model.transformer.wte
            logger.info(f"[MemoryEncoder] 提取 HF GPT embedding: {emb.num_embeddings} × {emb.embedding_dim}")
            return emb

        # Debug 模型: model.embed
        if hasattr(model, "embed") and isinstance(model.embed, nn.Embedding):
            emb = model.embed
            logger.info(f"[MemoryEncoder] 提取 Debug embedding: {emb.num_embeddings} × {emb.embedding_dim}")
            return emb

        logger.warning("[MemoryEncoder] 未找到 embedding 层，将使用随机初始化")
        return None

    # ------------------------------------------------------------------ #
    #  文本格式化
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_l2_object(obj: L2MemoryObject) -> str:
        """将 L2 对象格式化为编码用的纯文本。"""
        parts = []
        if obj.object_type:
            parts.append(f"[{obj.object_type}]")
        parts.append(obj.summary_text)
        return " ".join(parts)

    @staticmethod
    def format_l3_entry(entry: L3ProfileEntry) -> str:
        """将 L3 条目格式化为编码用的纯文本。"""
        parts = []
        if entry.category:
            parts.append(f"[{entry.category}]")
        if entry.key:
            parts.append(f"{entry.key}:")
        parts.append(entry.value)
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    #  编码核心
    # ------------------------------------------------------------------ #

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """将一批文本编码为向量。

        Args:
            texts: 文本列表.

        Returns:
            memory_vectors: (num_texts, output_dim) 的向量.
        """
        if not texts:
            return torch.zeros(0, self.output_dim, device=self._device, dtype=self._dtype)

        if self._embedding is None or self._tokenizer is None:
            logger.warning("[MemoryEncoder] 未初始化，返回零向量")
            return torch.zeros(
                len(texts), self.output_dim, device=self._device, dtype=self._dtype
            )

        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_memory_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # Embedding (不计算梯度，共享参数但不更新 backbone embedding)
        with torch.no_grad():
            embeddings = self._embedding(input_ids)  # (B, T, D)

        # Pooling
        if self.config.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)  # (B, T, 1)
                pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = embeddings.mean(dim=1)
        elif self.config.pooling == "last":
            if attention_mask is not None:
                # 取每个序列最后一个非 padding token
                lengths = attention_mask.sum(dim=1).long() - 1  # (B,)
                pooled = embeddings[torch.arange(embeddings.size(0)), lengths]
            else:
                pooled = embeddings[:, -1]
        elif self.config.pooling == "first":
            pooled = embeddings[:, 0]
        else:
            raise ValueError(f"未知的 pooling 策略: {self.config.pooling}")

        # (可选) 投影
        if self._projection is not None:
            pooled = self._projection(pooled)

        # (可选) Dropout
        if self._dropout is not None:
            pooled = self._dropout(pooled)

        return pooled  # (B, output_dim)

    def encode(
        self,
        l2_objects: list[L2MemoryObject] | None = None,
        l3_entries: list[L3ProfileEntry] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """编码 L2/L3 记忆对象为向量。

        Args:
            l2_objects: L2 记忆对象列表.
            l3_entries: L3 画像条目列表.

        Returns:
            (memory_vectors, memory_texts):
                memory_vectors: (num_memories, output_dim)
                memory_texts: 对应的格式化文本列表
        """
        texts: list[str] = []
        if l2_objects:
            for obj in l2_objects:
                texts.append(self.format_l2_object(obj))
        if l3_entries:
            for entry in l3_entries:
                texts.append(self.format_l3_entry(entry))

        if not texts:
            return (
                torch.zeros(0, self.output_dim, device=self._device, dtype=self._dtype),
                [],
            )

        vectors = self.encode_texts(texts)
        return vectors, texts

    def encode_with_grad(self, texts: list[str]) -> torch.Tensor:
        """带梯度的文本编码 (用于训练 context selector)。

        与 encode_texts 的区别：不使用 torch.no_grad()，
        允许梯度流过 embedding 层（如果需要微调 embedding）。

        Args:
            texts: 文本列表.

        Returns:
            memory_vectors: (num_texts, output_dim)
        """
        if not texts:
            return torch.zeros(0, self.output_dim, device=self._device, dtype=self._dtype)

        if self._embedding is None or self._tokenizer is None:
            return torch.zeros(
                len(texts), self.output_dim, device=self._device, dtype=self._dtype
            )

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_memory_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # 注意：这里不用 no_grad，允许梯度回传
        embeddings = self._embedding(input_ids)

        if self.config.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
                pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = embeddings.mean(dim=1)
        elif self.config.pooling == "last":
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).long() - 1
                pooled = embeddings[torch.arange(embeddings.size(0)), lengths]
            else:
                pooled = embeddings[:, -1]
        else:
            pooled = embeddings[:, 0]

        if self._projection is not None:
            pooled = self._projection(pooled)
        if self._dropout is not None:
            pooled = self._dropout(pooled)

        return pooled
