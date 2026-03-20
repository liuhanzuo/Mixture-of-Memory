"""
TurnProcessor: 单轮对话处理器。

实现 MoM Agent 的核心单轮处理流程:
1. 接收用户消息
2. 从 L2 / L3 检索相关记忆
3. 将检索到的记忆格式化并注入 prompt 上下文
4. 调用 backbone 生成回复
5. 在线更新 L1 记忆
6. 在 turn 结束时触发 L2 聚合

这是 Agent 的最内层循环。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from src.backbone.interfaces import BackboneModel, MemoryReadableBackbone
from src.backbone.hidden_state_types import BackboneOutput
from src.memory.scheduler import MemoryScheduler
from src.memory.l2.types import L2MemoryObject
from src.memory.l3.summarizer import L3ProfileEntry

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """单轮处理的结果。"""

    response_text: str
    turn_id: str
    # 检索到的 L2 对象
    retrieved_l2: list[tuple[L2MemoryObject, float]] = field(default_factory=list)
    # 检索到的 L3 条目
    retrieved_l3: list[L3ProfileEntry] = field(default_factory=list)
    # 注入的记忆上下文文本
    memory_context: str = ""
    # backbone 输出 (可选, 用于调试)
    backbone_output: BackboneOutput | None = None
    # 耗时统计
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    update_time_ms: float = 0.0


class TurnProcessor:
    """单轮对话处理器。

    封装了完整的「检索 → 生成 → 更新」流程。
    由 MemoryAgent 在每个对话轮次中调用。

    Args:
        scheduler: MoM 记忆调度器.
        backbone: 骨干模型 (可选, 若为 None 则使用规则生成).
        tokenizer: HuggingFace tokenizer (可选).
        max_context_chars: 记忆上下文的最大字符数.
        l2_top_k: L2 检索的 top-k.
        l3_top_k: L3 检索的 top-k.
    """

    def __init__(
        self,
        scheduler: MemoryScheduler,
        backbone: BackboneModel | None = None,
        tokenizer: Any | None = None,
        max_context_chars: int = 3000,
        l2_top_k: int = 5,
        l3_top_k: int = 10,
    ):
        self.scheduler = scheduler
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_context_chars = max_context_chars
        self.l2_top_k = l2_top_k
        self.l3_top_k = l3_top_k

    # ------------------------------------------------------------------ #
    #  主流程
    # ------------------------------------------------------------------ #

    def process_turn(
        self,
        user_message: str,
        turn_id: str,
        conversation_history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> TurnResult:
        """处理一个完整的对话轮次。

        Args:
            user_message: 当前用户输入.
            turn_id: 当前轮次 ID.
            conversation_history: 之前的对话历史 (可选, 用于 prompt 构建).
            system_prompt: 系统提示词 (可选).

        Returns:
            TurnResult 包含回复文本、检索结果、耗时等信息.
        """
        result = TurnResult(response_text="", turn_id=turn_id)

        # ---- Step 1: 记忆检索 ---- #
        t0 = time.monotonic()
        retrieved_l2, retrieved_l3, memory_context = self._retrieve_memory(user_message)
        result.retrieved_l2 = retrieved_l2
        result.retrieved_l3 = retrieved_l3
        result.memory_context = memory_context
        result.retrieval_time_ms = (time.monotonic() - t0) * 1000

        # ---- Step 2: 构建 prompt ---- #
        prompt = self._build_prompt(
            user_message=user_message,
            memory_context=memory_context,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
        )

        # ---- Step 3: 生成回复 ---- #
        t1 = time.monotonic()
        response_text, backbone_output = self._generate(prompt)
        result.response_text = response_text
        result.backbone_output = backbone_output
        result.generation_time_ms = (time.monotonic() - t1) * 1000

        # ---- Step 4: 在线更新 L1 ---- #
        t2 = time.monotonic()
        if backbone_output is not None:
            self._update_l1(backbone_output)
        result.update_time_ms = (time.monotonic() - t2) * 1000

        # ---- Step 5: 推送消息到缓冲区 & 触发 turn-end ---- #
        self._push_turn_messages(user_message, response_text, turn_id)
        self.scheduler.on_turn_end()

        logger.info(
            f"[TurnProcessor] Turn {turn_id} 完成: "
            f"retrieval={result.retrieval_time_ms:.1f}ms, "
            f"generation={result.generation_time_ms:.1f}ms, "
            f"update={result.update_time_ms:.1f}ms, "
            f"L2_retrieved={len(retrieved_l2)}, L3_retrieved={len(retrieved_l3)}"
        )

        return result

    # ------------------------------------------------------------------ #
    #  Step 1: 记忆检索
    # ------------------------------------------------------------------ #

    def _retrieve_memory(
        self, query: str,
    ) -> tuple[list[tuple[L2MemoryObject, float]], list[L3ProfileEntry], str]:
        """从 L2 和 L3 检索相关记忆并格式化为上下文文本。

        Returns:
            (retrieved_l2, retrieved_l3, formatted_context_string)
        """
        # L2 检索
        l2_results = self.scheduler.retrieve_l2(query=query, top_k=self.l2_top_k)

        # L3 检索
        l3_results = self.scheduler.retrieve_l3(query=query, top_k=self.l3_top_k)

        # 格式化为 prompt 上下文
        context_parts: list[str] = []

        # L3 (长期画像) 放在前面
        if l3_results:
            l3_lines = ["[Long-term Profile]"]
            char_count = len(l3_lines[0])
            for entry in l3_results:
                line = f"- [{entry.category}] {entry.key}: {entry.value}"
                if char_count + len(line) > self.max_context_chars // 2:
                    break
                l3_lines.append(line)
                char_count += len(line)
            context_parts.append("\n".join(l3_lines))

        # L2 (事件级记忆)
        if l2_results:
            l2_lines = ["[Recent Memory Objects]"]
            char_count = len(l2_lines[0])
            for obj, score in l2_results:
                if isinstance(obj, L2MemoryObject):
                    line = f"- [{obj.object_type}] {obj.summary_text} (score={score:.2f})"
                else:
                    line = f"- {obj}"
                if char_count + len(line) > self.max_context_chars // 2:
                    break
                l2_lines.append(line)
                char_count += len(line)
            context_parts.append("\n".join(l2_lines))

        memory_context = "\n\n".join(context_parts) if context_parts else ""

        # 统一返回类型
        l3_typed: list[L3ProfileEntry] = []
        for item in l3_results:
            if isinstance(item, L3ProfileEntry):
                l3_typed.append(item)

        return l2_results, l3_typed, memory_context

    # ------------------------------------------------------------------ #
    #  Step 2: 构建 prompt
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        user_message: str,
        memory_context: str,
        conversation_history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """构建最终发送给 backbone 的完整 prompt。

        结构:
        1. system prompt
        2. memory context (L3 profile + L2 objects)
        3. conversation history (recent turns)
        4. current user message
        """
        parts: list[str] = []

        # 系统提示
        if system_prompt:
            parts.append(f"[System]\n{system_prompt}")

        # 记忆上下文
        if memory_context:
            parts.append(f"\n{memory_context}")

        # 对话历史
        if conversation_history:
            history_lines: list[str] = ["[Conversation History]"]
            for msg in conversation_history[-10:]:  # 最近 10 轮
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            parts.append("\n".join(history_lines))

        # 当前消息
        parts.append(f"user: {user_message}")
        parts.append("assistant:")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Step 3: 生成
    # ------------------------------------------------------------------ #

    def _generate(
        self, prompt: str,
    ) -> tuple[str, BackboneOutput | None]:
        """调用 backbone 生成回复。

        如果 backbone 为 None, 返回简单的回显回复 (用于测试/调试)。

        Returns:
            (response_text, backbone_output_or_None)
        """
        if self.backbone is None:
            # 无 backbone: 回显模式 (用于测试)
            response = f"[Echo] I received your message. Memory context was injected."
            logger.debug("[TurnProcessor] No backbone, using echo mode.")
            return response, None

        # 有 backbone: 真实推理
        if self.tokenizer is None:
            logger.warning("[TurnProcessor] Backbone provided but no tokenizer. Using echo mode.")
            return "[Echo] No tokenizer available.", None

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            input_ids = inputs["input_ids"].to(self.backbone.get_device())
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.backbone.get_device())

            # Forward
            with torch.no_grad():
                output = self.backbone.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # 简单的 greedy decode (取最后一个 token 的 logits)
            if output.logits is not None:
                next_token_id = output.logits[:, -1, :].argmax(dim=-1)
                response = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            else:
                response = "[No logits available]"

            return response, output

        except Exception as e:
            logger.error(f"[TurnProcessor] Generation failed: {e}")
            return f"[Error] Generation failed: {e}", None

    # ------------------------------------------------------------------ #
    #  Step 4: L1 在线更新
    # ------------------------------------------------------------------ #

    def _update_l1(self, backbone_output: BackboneOutput) -> None:
        """从 backbone 输出的隐藏状态在线更新 L1 记忆。"""
        try:
            hidden = backbone_output.last_hidden_state
            self.scheduler.on_token_step(hidden_states=hidden)
        except Exception as e:
            logger.warning(f"[TurnProcessor] L1 update failed: {e}")

    # ------------------------------------------------------------------ #
    #  Step 5: 推送消息到缓冲区
    # ------------------------------------------------------------------ #

    def _push_turn_messages(
        self,
        user_message: str,
        response_text: str,
        turn_id: str,
    ) -> None:
        """将当前轮次的用户消息和助手回复推送到 scheduler 缓冲区。"""
        self.scheduler.push_message({
            "role": "user",
            "content": user_message,
            "turn_id": turn_id,
        })
        self.scheduler.push_message({
            "role": "assistant",
            "content": response_text,
            "turn_id": turn_id,
        })
