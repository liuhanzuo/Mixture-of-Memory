"""L3 总结器: 从 L2 记忆对象中生成长期语义/画像记忆。

L3 通过总结和抽象 L2 的对象, 产生更高层次的长期记忆, 如:
- "用户最近在研究大语言模型"
- "用户偏好技术性、结构化的解释"
- "用户的长期项目是 agent 记忆系统"
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any

from src.memory.l2.types import L2MemoryObject

logger = logging.getLogger(__name__)


# --- L3 数据类型 ---

from dataclasses import dataclass, field


@dataclass
class L3ProfileEntry:
    """L3 层的画像/语义记忆条目。

    Attributes:
        entry_id: 唯一标识符.
        key: 记忆的键 (如 "research_interest", "preferred_style").
        value: 记忆的值 (自然语言描述).
        confidence: 置信度 [0, 1].
        evidence_ids: 支撑证据的 L2 对象 ID 列表.
        created_at: 创建时间.
        last_updated_at: 最后更新时间.
        category: 分类 (如 "research_interest", "preference", "identity" 等).
    """

    entry_id: str
    key: str
    value: str
    confidence: float = 1.0
    evidence_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    category: str = "factual"

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化的字典。"""
        return {
            "entry_id": self.entry_id,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "category": self.category,
        }


# --- 总结器后端 ---

class SummarizerBackend(ABC):
    """总结器后端的抽象接口。"""

    @abstractmethod
    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        """从 L2 对象中生成 L3 profile 条目。"""
        ...


class RuleBasedSummarizer(SummarizerBackend):
    """基于规则的总结器。

    策略:
    - 按 object_type 分组 L2 对象
    - 对每组生成摘要性 profile entry
    - 同时为每个 L2 对象单独生成细粒度条目 (保留关键词)
    - 偏好类对象 → preference 类 entry
    - 话题类对象 → research_interest 类 entry
    - 任务类对象 → long_term_project 类 entry
    """

    TYPE_TO_CATEGORY = {
        "topic": "research_interest",
        "preference": "preference",
        "task": "long_term_project",
        "state": "identity",
        "entity": "factual",
        "relation": "factual",
    }

    def _generate_id(self, key: str) -> str:
        content = f"{key}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        """基于规则从 L2 对象生成 L3 条目。

        生成两种粒度的条目:
        1. 按类型分组的汇总条目 (概括性)
        2. 单个 L2 对象的细粒度条目 (保留原始关键词，用于精确检索)
        """
        if not l2_objects:
            return []

        # 按类型分组
        grouped: dict[str, list[L2MemoryObject]] = defaultdict(list)
        for obj in l2_objects:
            if not obj.is_archived:
                grouped[obj.object_type].append(obj)

        entries: list[L3ProfileEntry] = []

        for obj_type, objs in grouped.items():
            category = self.TYPE_TO_CATEGORY.get(obj_type, "factual")

            # ---- 1. 汇总条目 ----
            summaries = [obj.summary_text for obj in objs]
            combined = "; ".join(summaries[:10])

            if category == "research_interest":
                value = f"The user has been discussing: {combined}"
                key = "recent_research_topics"
            elif category == "preference":
                value = f"The user prefers: {combined}"
                key = "user_preferences"
            elif category == "long_term_project":
                value = f"The user is working on: {combined}"
                key = "active_tasks"
            elif category == "identity":
                value = f"Current user state: {combined}"
                key = "current_state"
            else:
                value = f"Known facts: {combined}"
                key = f"facts_{obj_type}"

            avg_confidence = sum(o.confidence for o in objs) / len(objs)
            evidence_ids = [o.object_id for o in objs]

            entry = L3ProfileEntry(
                entry_id=self._generate_id(key),
                key=key,
                value=value,
                confidence=avg_confidence,
                evidence_ids=evidence_ids,
                category=category,
            )
            entries.append(entry)

            # ---- 2. 细粒度条目 (每个 L2 对象独立生成) ----
            for i, obj in enumerate(objs):
                detail_text = obj.summary_text
                # 去掉 "Discussion topic:" 等前缀
                for prefix in ("Discussion topic:", "[INVALIDATED]"):
                    detail_text = detail_text.replace(prefix, "").strip()

                if len(detail_text) < 5:
                    continue

                detail_key = f"{key}_detail_{i}"
                detail_entry = L3ProfileEntry(
                    entry_id=self._generate_id(detail_key),
                    key=detail_key,
                    value=detail_text,
                    confidence=obj.confidence,
                    evidence_ids=[obj.object_id],
                    category=category,
                )
                entries.append(detail_entry)

        logger.info(f"[L3 Summarizer] Generated {len(entries)} profile entries from {len(l2_objects)} L2 objects.")
        return entries


class LLMSummarizer(SummarizerBackend):
    """基于本地 LLM 的总结器。

    工作流程:
    1. 将 L2 对象按类型分组并序列化为结构化 prompt
    2. 调用本地 HF 模型 generate 生成高层次语义画像 (JSON)
    3. 解析 LLM 输出为 L3ProfileEntry
    4. 如果推理失败，自动 fallback 到 RuleBasedSummarizer

    支持:
    - 直接加载 HF 模型 (model_name_or_path)
    - 加载 LoRA adapter (adapter_path)
    - 外部注入已加载的 model + tokenizer
    """

    # ---- 系统 Prompt: 指导 LLM 生成画像条目 ---- #
    SYSTEM_PROMPT = (
        "You are a user profile summarizer. Given a list of L2 memory objects "
        "(extracted from recent conversations), generate high-level long-term "
        "profile entries. Each entry has:\n"
        '- key: a short identifier (e.g. "research_interest", "preferred_style")\n'
        "- value: a natural language description of the long-term memory\n"
        '- category: one of [research_interest, preference, long_term_project, '
        'identity, expertise, factual]\n'
        "- confidence: float between 0.0 and 1.0\n\n"
        "Return a JSON array of entries. Example:\n"
        '[{"key": "research_interest", "value": "The user has been researching '
        'sparse training for LLMs", "category": "research_interest", "confidence": 0.85}]\n\n'
        "Rules:\n"
        "- Summarize and abstract, do NOT copy L2 objects verbatim.\n"
        "- Merge related L2 objects into single profile entries when appropriate.\n"
        "- Be concise but informative.\n"
        "- Only output the JSON array, no other text.\n"
        "- If no profile can be generated, return an empty array: []"
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,
        adapter_path: str | None = None,
        model: Any = None,
        tokenizer: Any = None,
        device: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        """
        Args:
            model_name_or_path: HF 模型路径. 如果提供 model/tokenizer 则忽略.
            adapter_path: LoRA adapter 路径 (可选).
            model: 已加载的 HF 模型实例 (可选，外部注入).
            tokenizer: 已加载的 tokenizer 实例 (可选，外部注入).
            device: 推理设备 (默认自动检测).
            max_new_tokens: 最大生成 token 数.
            temperature: 生成温度.
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = model
        self._tokenizer = tokenizer
        self._fallback = RuleBasedSummarizer()
        self._model_name_or_path = model_name_or_path
        self._adapter_path = adapter_path
        self._device = device
        self._loaded = model is not None and tokenizer is not None

        if self._loaded:
            logger.info("[L3 LLMSummarizer] 使用外部注入的 model + tokenizer。")
        elif model_name_or_path:
            logger.info(f"[L3 LLMSummarizer] 将从 {model_name_or_path} 延迟加载模型。")
        else:
            logger.warning(
                "[L3 LLMSummarizer] 未提供模型路径，LLM 总结将 fallback 到规则后端。"
            )

    def _ensure_loaded(self) -> bool:
        """延迟加载模型 + tokenizer，成功返回 True。"""
        if self._loaded:
            return True
        if not self._model_name_or_path:
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"[L3 LLMSummarizer] 正在加载模型: {self._model_name_or_path}")
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name_or_path,
                trust_remote_code=True,
                padding_side="left",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name_or_path,
                torch_dtype=torch.float16,
                device_map=device if device != "cpu" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                self._model = self._model.float()

            # 如果提供了 LoRA adapter，合并加载
            if self._adapter_path:
                try:
                    from peft import PeftModel
                    logger.info(f"[L3 LLMSummarizer] 加载 LoRA adapter: {self._adapter_path}")
                    self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
                    self._model = self._model.merge_and_unload()
                    logger.info("[L3 LLMSummarizer] LoRA adapter 合并完成。")
                except ImportError:
                    logger.error("[L3 LLMSummarizer] peft 未安装，无法加载 LoRA adapter。")
                except Exception as e:
                    logger.error(f"[L3 LLMSummarizer] LoRA adapter 加载失败: {e}")

            self._model.eval()
            self._loaded = True
            logger.info("[L3 LLMSummarizer] 模型加载完成。")
            return True

        except Exception as e:
            logger.error(f"[L3 LLMSummarizer] 模型加载失败: {e}", exc_info=True)
            return False

    def _build_prompt(self, l2_objects: list[L2MemoryObject]) -> str:
        """将 L2 对象序列化为 LLM prompt。"""
        lines = []
        for i, obj in enumerate(l2_objects):
            lines.append(
                f"{i+1}. [{obj.object_type}] {obj.summary_text} "
                f"(confidence={obj.confidence:.2f})"
            )
        objects_text = "\n".join(lines)

        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"--- L2 Memory Objects ({len(l2_objects)} total) ---\n"
            f"{objects_text}\n"
            f"--- End ---\n\n"
            f"Generate long-term profile entries as a JSON array:"
        )

    def _parse_llm_output(
        self, output_text: str, l2_objects: list[L2MemoryObject]
    ) -> list[L3ProfileEntry]:
        """解析 LLM 输出的 JSON 为 L3ProfileEntry 列表。"""
        import json as json_mod
        import re

        text = output_text.strip()

        # 尝试解析 JSON
        parsed = None
        # 方法1: 直接解析
        try:
            parsed = json_mod.loads(text)
        except json_mod.JSONDecodeError:
            pass

        # 方法2: 提取 [...] 块
        if parsed is None:
            bracket_start = text.find("[")
            bracket_end = text.rfind("]")
            if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
                try:
                    parsed = json_mod.loads(text[bracket_start:bracket_end + 1])
                except json_mod.JSONDecodeError:
                    pass

        # 方法3: 提取 ```json ... ``` 代码块
        if parsed is None:
            code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            if code_block_match:
                try:
                    parsed = json_mod.loads(code_block_match.group(1))
                except json_mod.JSONDecodeError:
                    pass

        if not parsed or not isinstance(parsed, list):
            logger.warning(
                f"[L3 LLMSummarizer] 无法解析 LLM 输出为 JSON 数组: {text[:200]}..."
            )
            return []

        # 转换为 L3ProfileEntry
        evidence_ids = [obj.object_id for obj in l2_objects]
        valid_categories = {
            "research_interest", "preference", "long_term_project",
            "identity", "expertise", "factual",
        }
        entries: list[L3ProfileEntry] = []

        for item in parsed:
            if not isinstance(item, dict):
                continue
            key = item.get("key", "")
            value = item.get("value", "")
            category = item.get("category", "factual")
            confidence = float(item.get("confidence", 0.7))

            if not key or not value or len(value) < 5:
                continue

            # 验证 category
            if category not in valid_categories:
                category = "factual"

            entry = L3ProfileEntry(
                entry_id=hashlib.md5(
                    f"{key}:{value}:{datetime.now().isoformat()}".encode()
                ).hexdigest()[:12],
                key=key,
                value=value,
                confidence=min(max(confidence, 0.0), 1.0),
                evidence_ids=evidence_ids,
                category=category,
            )
            entries.append(entry)

        logger.info(f"[L3 LLMSummarizer] 从 LLM 输出中解析出 {len(entries)} 条 profile entries。")
        return entries

    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        """使用本地 LLM 从 L2 对象中生成 L3 profile 条目。

        如果 LLM 加载失败或推理出错，自动 fallback 到规则后端。
        """
        if not l2_objects:
            return []

        # 过滤已归档对象
        active_objects = [obj for obj in l2_objects if not obj.is_archived]
        if not active_objects:
            return []

        # 尝试加载模型
        if not self._ensure_loaded():
            logger.info("[L3 LLMSummarizer] 模型不可用，fallback 到规则后端。")
            return self._fallback.summarize(l2_objects, existing_entries)

        try:
            import torch

            prompt = self._build_prompt(active_objects)

            # 使用 chat template (如果 tokenizer 支持)
            if hasattr(self._tokenizer, "apply_chat_template"):
                chat_messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt.split("--- L2 Memory Objects", 1)[-1]},
                ]
                input_text = self._tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                input_text = prompt

            inputs = self._tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=2048,
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # 截取新生成的 token
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, input_len:]
            output_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            logger.debug(f"[L3 LLMSummarizer] LLM 输出: {output_text[:300]}...")

            entries = self._parse_llm_output(output_text, active_objects)

            # 如果 LLM 没有生成任何条目，用规则后端补充
            if not entries:
                logger.info("[L3 LLMSummarizer] LLM 未生成条目，使用规则后端补充。")
                entries = self._fallback.summarize(l2_objects, existing_entries)

            return entries

        except Exception as e:
            logger.error(f"[L3 LLMSummarizer] LLM 推理失败: {e}", exc_info=True)
            logger.info("[L3 LLMSummarizer] Fallback 到规则后端。")
            return self._fallback.summarize(l2_objects, existing_entries)


class L3Summarizer:
    """L3 总结器的统一入口。

    根据配置选择 rule_based 或 llm 后端。

    配置示例::

        # 规则后端 (默认)
        {"summarizer_backend": "rule_based"}

        # LLM 后端
        {
            "summarizer_backend": "llm",
            "model_name_or_path": "/path/to/model",
            "adapter_path": "/path/to/lora",   # 可选
            "max_new_tokens": 512,
            "temperature": 0.2,
        }

        # 外部注入模型
        L3Summarizer(config={"summarizer_backend": "llm"}, model=model, tokenizer=tok)
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.config = config
        backend_type = config.get("summarizer_backend", "rule_based")

        if backend_type == "rule_based":
            self.backend = RuleBasedSummarizer()
        elif backend_type in ("llm", "llm_backed"):
            self.backend = LLMSummarizer(
                model_name_or_path=config.get("model_name_or_path"),
                adapter_path=config.get("adapter_path"),
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.get("max_new_tokens", 512),
                temperature=config.get("temperature", 0.2),
            )
        else:
            raise ValueError(f"Unknown summarizer backend: {backend_type}")

        logger.info(f"[L3 Summarizer] Using backend: {backend_type}")

    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        """从 L2 对象生成 L3 profile 条目。"""
        return self.backend.summarize(l2_objects, existing_entries)
