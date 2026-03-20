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
import re
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

        如果 backbone 为 None, 使用记忆感知的规则回复。

        Returns:
            (response_text, backbone_output_or_None)
        """
        if self.backbone is None:
            # 无 backbone: 记忆感知的规则回复模式
            response = self._rule_based_reply(prompt)
            logger.debug("[TurnProcessor] No backbone, using memory-aware rule reply.")
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

    # ------------------------------------------------------------------ #
    #  记忆感知的规则回复 (无 backbone 时使用)
    # ------------------------------------------------------------------ #

    def _rule_based_reply(self, prompt: str) -> str:
        """基于记忆上下文和对话历史生成规则回复。

        核心策略:
        1. 从 prompt 中提取用户的最后一条消息 (query)
        2. 检查是否有 stale/temporary 类型的无效化声明
        3. 从 prompt 中提取记忆上下文内容 (已由 L2 Retriever 排序)
        4. 从对话历史中搜索与 query 相关的事实
        5. 组合成有意义的回复
        """
        # 提取 user 最后的消息
        user_query = self._extract_last_user_message(prompt)

        # 首先检查是否有无效化/临时状态 (stale/temporary)
        invalidation_reply = self._check_invalidation(prompt, user_query)
        if invalidation_reply:
            return invalidation_reply

        # 提取记忆上下文 (由 L2 Retriever 已按相关度排过序)
        memory_info = self._extract_memory_from_prompt(prompt)

        # 从对话历史中提取相关事实 (排除 query 本身)
        history_facts = self._extract_facts_from_history(prompt, user_query)

        # 组合回复
        reply_parts: list[str] = []

        # 优先使用对话历史中提取的最新事实
        if history_facts:
            # 取最后出现的（最新的）事实
            reply_parts.append(history_facts[-1])

        # 其次使用记忆上下文中的信息 (已按相关度排序，直接取前几条)
        if memory_info:
            for info in memory_info[:3]:
                # 去重：如果和 history_facts 有较大重叠就跳过
                if reply_parts and any(
                    info[:10] in existing or existing[:10] in info
                    for existing in reply_parts
                ):
                    continue
                reply_parts.append(info)

        # 如果没有任何记忆/历史信息，回复无信息
        if not reply_parts:
            reply_parts.append("这个信息我目前没有记录。")

        return "。".join(reply_parts)

    def _extract_last_user_message(self, prompt: str) -> str:
        """从 prompt 中提取最后一条 user 消息。"""
        lines = prompt.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("user:"):
                return line[5:].strip()
        return ""

    def _extract_memory_from_prompt(self, prompt: str) -> list[str]:
        """从 prompt 的记忆上下文部分提取信息。"""
        info_items: list[str] = []
        in_memory_section = False

        for line in prompt.split("\n"):
            line = line.strip()
            if line in ("[Long-term Profile]", "[Recent Memory Objects]"):
                in_memory_section = True
                continue
            if line.startswith("[") and not line.startswith("- "):
                in_memory_section = False
                continue
            if in_memory_section and line.startswith("- "):
                # 提取记忆条目的内容部分
                content = line[2:].strip()
                # 去掉类型标签 [xxx]
                content = re.sub(r"^\[.*?\]\s*", "", content)
                # 去掉尾部的 (score=xxx) 或 (conf=xxx)
                content = re.sub(r"\s*\((?:score|conf)=[\d.]+\)\s*$", "", content)
                # 过滤掉 "Discussion topic:" 前缀的低信息量条目
                if content.startswith("Discussion topic:"):
                    content = content[len("Discussion topic:"):].strip()
                if content and len(content) > 3:
                    info_items.append(content)
        return info_items

    def _filter_relevant(self, items: list[str], query: str) -> list[str]:
        """根据 query 从 items 中筛选相关的条目。"""
        if not items or not query:
            return items

        query_lower = query.lower()
        # 提取 query 关键词
        keywords = set()
        for w in re.split(r"[\s，。？?！!、]+", query_lower):
            if w and len(w) >= 2:
                keywords.add(w)

        # 按与 query 的关键词重叠数排序
        scored: list[tuple[str, int]] = []
        for item in items:
            item_lower = item.lower()
            score = sum(1 for kw in keywords if kw in item_lower)
            # 如果 query 中的实体子串出现在 item 中，额外加分
            entities = re.findall(r"[\u4e00-\u9fffA-Za-z_]{2,}", query)
            for e in entities:
                if e.lower() in item_lower:
                    score += 2
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, sc in scored if sc > 0]

    def _check_invalidation(self, prompt: str, query: str) -> str | None:
        """检查对话历史中是否有对 query 主题的无效化/临时状态声明。

        如果检测到:
        - 明确的失效/遗忘声明 → 回答 "不确定"
        - 临时状态声明 → 回答 "不确定"
        """
        history_text = ""
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("user:"):
                history_text += stripped[5:] + " "

        # 提取 query 中的关键词
        query_keywords = set()
        for w in re.split(r"[\s，。？?！!、]+", query.lower()):
            if w and len(w) >= 2:
                query_keywords.add(w)

        # 检查无效化声明
        invalidation_patterns = [
            r"已经不准", r"忘掉", r"不再有效", r"清除", r"过期",
        ]
        for pat in invalidation_patterns:
            if re.search(pat, history_text):
                # 确保失效声明和 query 主题相关
                for kw in query_keywords:
                    if kw in history_text.lower():
                        return "这个信息已经过期了，我不确定当前的状态。"

        # 检查临时状态声明
        temporary_patterns = [
            r"暂时", r"临时", r"一会儿就好", r"不过一会",
        ]
        for pat in temporary_patterns:
            if re.search(pat, history_text):
                for kw in query_keywords:
                    if kw in history_text.lower():
                        return "之前提到的是临时状态，可能已经过去了，我不确定现在的情况。"

        return None

    def _extract_facts_from_history(self, prompt: str, query: str) -> list[str]:
        """从 prompt 中的对话历史部分搜索与 query 相关的事实。

        核心逻辑:
        - 扫描对话历史中 user 消息包含的事实声明
        - 与 query 进行关键词匹配
        - 返回最相关的事实
        """
        facts: list[str] = []
        history_lines: list[str] = []
        in_history = False

        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped == "[Conversation History]":
                in_history = True
                continue
            if stripped.startswith("[") and not stripped.startswith("user:") and not stripped.startswith("assistant:"):
                if in_history:
                    in_history = False
                continue
            if in_history and (stripped.startswith("user:") or stripped.startswith("assistant:")):
                history_lines.append(stripped)

        # 从历史中提取事实声明
        query_lower = query.lower()

        # 提取 query 中的关键词 (去除常见疑问词)
        stop_words = {
            "什么", "怎么", "哪里", "哪个", "多少", "如何", "为什么", "吗",
            "是", "的", "了", "在", "我", "你", "他", "她", "它",
            "现在", "目前", "最近", "请问", "告诉",
            "what", "where", "how", "which", "who", "when", "is", "are",
            "the", "a", "an", "my", "your", "do", "does",
        }
        query_keywords = set()
        for w in re.split(r"[\s，。？?！!、]+", query_lower):
            if w and w not in stop_words and len(w) >= 2:
                query_keywords.add(w)

        # 扫描历史，找与 query 关键词匹配的 user 声明
        # 排除 query 本身 (最后一条 user 消息)
        last_user_line = ""
        for line in reversed(history_lines):
            if line.startswith("user:"):
                last_user_line = line
                break

        for line in history_lines:
            # 跳过 query 本身和 assistant 消息
            if line == last_user_line:
                continue
            if not line.startswith("user:"):
                continue
            content = line[5:].strip()
            content_lower = content.lower()
            # 检查关键词匹配
            for kw in query_keywords:
                if kw in content_lower:
                    facts.append(content)
                    break

        # 如果没有找到直接关键词匹配，做更宽泛的搜索:
        # 检查 query 中的实体名是否出现在历史中
        if not facts:
            entity_pattern = r"[A-Za-z0-9\u4e00-\u9fff]{2,}"
            query_entities = set(re.findall(entity_pattern, query))
            for line in history_lines:
                if line == last_user_line:
                    continue
                if not line.startswith("user:"):
                    continue
                content = line[5:].strip()
                for entity in query_entities:
                    if entity in content:
                        facts.append(content)
                        break

        return facts

    def _echo_relevant_from_prompt(self, prompt: str, query: str) -> str:
        """最后的回退策略: 尝试从 prompt 中提取一些有用信息。"""
        return ""
