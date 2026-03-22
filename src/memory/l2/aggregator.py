"""L2 聚合器: 从最近的消息/轮次/chunks 中提取记忆对象。

L2 不从 L1 提升，而是直接从原始对话消息中聚合。
支持:
- rule_based: 基于规则的简单聚合 (默认)
- llm: 基于 LLM 的聚合 (TODO stub)
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.memory.l2.types import ChatMessage, L2MemoryObject

logger = logging.getLogger(__name__)


class AggregatorBackend(ABC):
    """聚合器后端的抽象接口。"""

    @abstractmethod
    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """从消息列表中提取 L2 记忆对象。

        Args:
            messages: 需要聚合的消息列表.
            existing_objects: 已有的记忆对象 (用于避免重复).

        Returns:
            提取出的新记忆对象列表.
        """
        ...


class RuleBasedAggregator(AggregatorBackend):
    """基于规则的聚合器。

    使用简单的启发式规则从消息中提取记忆对象:
    - topic: 从消息内容中提取话题关键词
    - preference: 检测偏好表达模式
    - task: 检测任务/需求表达模式
    - entity: 提取命名实体 (简化版)
    - state: 提取当前状态信息
    """

    # 偏好相关的模式
    PREFERENCE_PATTERNS = [
        r"(?:I prefer|I like|I want|I'd rather|偏好|喜欢|习惯用|倾向于)\s*([^。.！？\n]{2,60})",
        r"(?:please use|always use|don't use|never use|请用|不要用|以后都这样|就是这种)\s*([^。.！？\n]{2,60})",
        r"(?:回答别太长|简洁点|详细一?点|用技术性的方式|结构化|分点说明)([^。.！？\n]{0,50})",
    ]

    # 任务相关的模式
    TASK_PATTERNS = [
        r"(?:help me|please|could you|can you|帮我|请|能否)\s*([^。.！？\n]{2,60})",
        r"(?:I need to|I want to|我需要|我想要|目标是)\s*([^。.！？\n]{2,60})",
    ]

    # 状态相关的模式
    STATE_PATTERNS = [
        r"(?:I am|I'm|currently|right now|我正在|我目前)\s*([^。.！？\n]{2,60})",
        r"(?:working on|researching|studying|在做|在研究|在设计|在开发|在用|在尝试)\s*([^。.！？\n]{2,60})",
    ]

    # 事实声明模式 (中文) — 用于提取 "X是Y" 类的事实
    # 注意: 使用限制性字符类 [^。.！？\n] 代替 (.+?) 避免灾难性回溯
    FACT_PATTERNS = [
        # "X是Y" / "X变成Y" / "X现在变成Y了"
        r"([\u4e00-\u9fffA-Za-z_]{2,20})(?:是|变成了?|改为|更新为|现在(?:变成了?|是))([^。.！？\n]{1,50})",
        # "告诉你一下，X是Y"
        r"(?:告诉你|顺便说|说一下|记一下)[，,]\s*([^。.！？\n]{3,80})",
        # "我的项目叫X" / "我在XX工作"
        r"我(?:的)?(?:项目|工作|城市|编辑器)(?:叫|是|在|用)\s*([^。.！？\n]{1,50})",
        # "更新一下，X现在是Y"
        r"(?:更新一下|纠正一下|修改一下)[，,]\s*([^。.！？\n]{3,80})",
    ]

    # 无效化/过期模式
    INVALIDATION_PATTERNS = [
        r"(?:已经不准|忘掉|不再有效|忘记|清除|过期)([^。.！？\n]{0,50})",
        r"(?:之前说的)([^。.！？\n]{1,50})(?:已经不准|忘掉|不再有效)",
    ]

    def _generate_id(self, text: str, obj_type: str) -> str:
        """生成唯一 ID。"""
        content = f"{obj_type}:{text}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_by_patterns(
        self,
        text: str,
        patterns: list[str],
        obj_type: str,
        turn_id: str,
    ) -> list[L2MemoryObject]:
        """使用正则模式提取对象。"""
        objects = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_text = match.strip() if isinstance(match, str) else match[0].strip()
                if len(match_text) < 3:
                    continue
                obj = L2MemoryObject(
                    object_id=self._generate_id(match_text, obj_type),
                    object_type=obj_type,
                    summary_text=match_text,
                    confidence=0.7,
                    source_turn_ids=[turn_id],
                )
                objects.append(obj)
        return objects

    def _extract_topic(self, messages: list[ChatMessage]) -> list[L2MemoryObject]:
        """从消息中提取话题对象。

        简单策略: 将连续消息的内容合并, 取前 200 字符作为话题摘要。
        """
        if not messages:
            return []

        combined_text = " ".join(m.content for m in messages if m.role == "user")
        if len(combined_text) < 10:
            return []

        summary = combined_text[:200].strip()
        turn_ids = [m.turn_id for m in messages if m.turn_id]

        return [
            L2MemoryObject(
                object_id=self._generate_id(summary, "topic"),
                object_type="topic",
                summary_text=f"Discussion topic: {summary}",
                confidence=0.6,
                source_turn_ids=turn_ids,
            )
        ]

    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """基于规则从消息中聚合 L2 对象。"""
        objects: list[L2MemoryObject] = []

        # 提取话题
        objects.extend(self._extract_topic(messages))

        # 从每条用户消息中提取偏好、任务、状态、事实
        for msg in messages:
            if msg.role != "user":
                continue
            objects.extend(
                self._extract_by_patterns(
                    msg.content, self.PREFERENCE_PATTERNS, "preference", msg.turn_id
                )
            )
            objects.extend(
                self._extract_by_patterns(
                    msg.content, self.TASK_PATTERNS, "task", msg.turn_id
                )
            )
            objects.extend(
                self._extract_by_patterns(
                    msg.content, self.STATE_PATTERNS, "state", msg.turn_id
                )
            )
            # 提取事实声明
            objects.extend(
                self._extract_facts(msg.content, msg.turn_id)
            )
            # 提取无效化声明
            objects.extend(
                self._extract_invalidations(msg.content, msg.turn_id)
            )

        logger.debug(f"[L2 Aggregator] Extracted {len(objects)} objects from {len(messages)} messages.")
        return objects

    def _extract_facts(self, text: str, turn_id: str) -> list[L2MemoryObject]:
        """从消息中提取事实声明 (如 'X是Y')。"""
        objects = []
        for pattern in self.FACT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = " ".join(m.strip() for m in match if m.strip())
                else:
                    match_text = match.strip()
                if len(match_text) < 3:
                    continue
                obj = L2MemoryObject(
                    object_id=self._generate_id(match_text, "entity"),
                    object_type="entity",
                    summary_text=match_text,
                    confidence=0.8,
                    source_turn_ids=[turn_id],
                    metadata={"raw_text": text},
                )
                objects.append(obj)
        return objects

    def _extract_invalidations(self, text: str, turn_id: str) -> list[L2MemoryObject]:
        """提取无效化/过期声明。"""
        objects = []
        for pattern in self.INVALIDATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_text = match.strip() if isinstance(match, str) else match[0].strip()
                obj = L2MemoryObject(
                    object_id=self._generate_id(match_text, "invalidation"),
                    object_type="state",
                    summary_text=f"[INVALIDATED] {text.strip()}",
                    confidence=0.9,
                    source_turn_ids=[turn_id],
                    metadata={"action": "invalidate", "raw_text": text},
                )
                objects.append(obj)
        return objects


class LLMAggregator(AggregatorBackend):
    """基于本地 LLM 的聚合器。

    工作流程:
    1. 将消息序列化为结构化 prompt
    2. 调用本地 HF 模型 generate 提取结构化记忆对象 (JSON)
    3. 解析 LLM 输出为 L2MemoryObject
    4. 如果推理失败，自动 fallback 到 RuleBasedAggregator

    支持:
    - 直接加载 HF 模型 (model_name_or_path)
    - 加载 LoRA adapter (adapter_path)
    - 外部注入已加载的 model + tokenizer
    """

    # ---- 系统 Prompt: 指导 LLM 提取记忆对象 ---- #
    SYSTEM_PROMPT = (
        "You are a memory extraction module. Given a list of conversation messages, "
        "extract structured memory objects. Each object has:\n"
        "- object_type: one of [topic, state, preference, task, entity]\n"
        "- summary_text: a concise summary of the memory\n"
        "- confidence: float between 0.0 and 1.0\n\n"
        "Return a JSON array of objects. Example:\n"
        '[{"object_type": "preference", "summary_text": "User prefers structured explanations", "confidence": 0.85}]\n\n'
        "Rules:\n"
        "- Extract ALL relevant memory objects from the conversation.\n"
        "- Be concise but informative in summary_text.\n"
        "- Only output the JSON array, no other text.\n"
        "- If no memory can be extracted, return an empty array: []"
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,
        adapter_path: str | None = None,
        model: Any = None,
        tokenizer: Any = None,
        device: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
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
        self._fallback = RuleBasedAggregator()
        self._model_name_or_path = model_name_or_path
        self._adapter_path = adapter_path
        self._device = device
        self._loaded = model is not None and tokenizer is not None

        if self._loaded:
            logger.info("[L2 LLMAggregator] 使用外部注入的 model + tokenizer。")
        elif model_name_or_path:
            logger.info(f"[L2 LLMAggregator] 将从 {model_name_or_path} 延迟加载模型。")
        else:
            logger.warning(
                "[L2 LLMAggregator] 未提供模型路径，LLM 聚合将 fallback 到规则后端。"
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

            logger.info(f"[L2 LLMAggregator] 正在加载模型: {self._model_name_or_path}")
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
                    logger.info(f"[L2 LLMAggregator] 加载 LoRA adapter: {self._adapter_path}")
                    self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
                    self._model = self._model.merge_and_unload()
                    logger.info("[L2 LLMAggregator] LoRA adapter 合并完成。")
                except ImportError:
                    logger.error("[L2 LLMAggregator] peft 未安装，无法加载 LoRA adapter。")
                except Exception as e:
                    logger.error(f"[L2 LLMAggregator] LoRA adapter 加载失败: {e}")

            self._model.eval()
            self._loaded = True
            logger.info("[L2 LLMAggregator] 模型加载完成。")
            return True

        except Exception as e:
            logger.error(f"[L2 LLMAggregator] 模型加载失败: {e}", exc_info=True)
            return False

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """将消息列表序列化为 LLM prompt。"""
        lines = []
        for msg in messages:
            role_tag = msg.role.upper()
            lines.append(f"[{role_tag}] {msg.content}")
        conversation_text = "\n".join(lines)

        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"--- Conversation ---\n"
            f"{conversation_text}\n"
            f"--- End ---\n\n"
            f"Extract memory objects as a JSON array:"
        )

    def _parse_llm_output(
        self, output_text: str, messages: list[ChatMessage]
    ) -> list[L2MemoryObject]:
        """解析 LLM 输出的 JSON 为 L2MemoryObject 列表。"""
        import json as json_mod

        # 尝试提取 JSON 数组
        text = output_text.strip()

        # 尝试直接解析
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
            logger.warning(f"[L2 LLMAggregator] 无法解析 LLM 输出为 JSON 数组: {text[:200]}...")
            return []

        # 转换为 L2MemoryObject
        turn_ids = [m.turn_id for m in messages if m.turn_id]
        objects: list[L2MemoryObject] = []

        for item in parsed:
            if not isinstance(item, dict):
                continue
            obj_type = item.get("object_type", "entity")
            summary = item.get("summary_text", "")
            confidence = float(item.get("confidence", 0.7))

            if not summary or len(summary) < 3:
                continue

            # 验证 object_type
            valid_types = {"topic", "state", "preference", "task", "entity"}
            if obj_type not in valid_types:
                obj_type = "entity"

            obj = L2MemoryObject(
                object_id=hashlib.md5(
                    f"{obj_type}:{summary}:{datetime.now().isoformat()}".encode()
                ).hexdigest()[:12],
                object_type=obj_type,
                summary_text=summary,
                confidence=min(max(confidence, 0.0), 1.0),
                source_turn_ids=turn_ids,
                metadata={"source": "llm_aggregator"},
            )
            objects.append(obj)

        logger.info(f"[L2 LLMAggregator] 从 LLM 输出中解析出 {len(objects)} 个对象。")
        return objects

    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """使用本地 LLM 从消息中提取记忆对象。

        如果 LLM 加载失败或推理出错，自动 fallback 到规则后端。
        """
        if not messages:
            return []

        # 尝试加载模型
        if not self._ensure_loaded():
            logger.info("[L2 LLMAggregator] 模型不可用，fallback 到规则后端。")
            return self._fallback.aggregate(messages, existing_objects)

        try:
            import torch

            prompt = self._build_prompt(messages)

            # 使用 chat template (如果 tokenizer 支持)
            if hasattr(self._tokenizer, "apply_chat_template"):
                chat_messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt.split("--- Conversation ---\n", 1)[-1]},
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

            logger.debug(f"[L2 LLMAggregator] LLM 输出: {output_text[:300]}...")

            objects = self._parse_llm_output(output_text, messages)

            # 如果 LLM 没有提取出任何对象，用规则后端补充
            if not objects:
                logger.info("[L2 LLMAggregator] LLM 未提取到对象，使用规则后端补充。")
                objects = self._fallback.aggregate(messages, existing_objects)

            return objects

        except Exception as e:
            logger.error(f"[L2 LLMAggregator] LLM 推理失败: {e}", exc_info=True)
            logger.info("[L2 LLMAggregator] Fallback 到规则后端。")
            return self._fallback.aggregate(messages, existing_objects)


class L2Aggregator:
    """L2 聚合器的统一入口。

    根据配置选择 rule_based 或 llm 后端。

    配置示例::

        # 规则后端 (默认)
        {"aggregator_backend": "rule_based"}

        # LLM 后端
        {
            "aggregator_backend": "llm",
            "model_name_or_path": "/path/to/model",
            "adapter_path": "/path/to/lora",   # 可选
            "max_new_tokens": 512,
            "temperature": 0.3,
        }

        # 外部注入模型
        L2Aggregator(config={"aggregator_backend": "llm"}, model=model, tokenizer=tok)
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.config = config
        backend_type = config.get("aggregator_backend", "rule_based")

        if backend_type == "rule_based":
            self.backend = RuleBasedAggregator()
        elif backend_type in ("llm", "llm_backed"):
            self.backend = LLMAggregator(
                model_name_or_path=config.get("model_name_or_path"),
                adapter_path=config.get("adapter_path"),
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.get("max_new_tokens", 512),
                temperature=config.get("temperature", 0.3),
            )
        else:
            raise ValueError(f"Unknown aggregator backend: {backend_type}")

        logger.info(f"[L2 Aggregator] Using backend: {backend_type}")

    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """聚合消息为 L2 记忆对象。"""
        return self.backend.aggregate(messages, existing_objects)
