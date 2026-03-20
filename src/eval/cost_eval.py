"""
开销评估器 (CostEvaluator)。

评估 MoM 记忆系统的运行时开销:
- 记忆读写次数 (L1 write/read, L2 aggregate/retrieve, L3 summarize/revise)
- 平均检索条目数
- 估算的 token / context 开销
- 各层操作频率统计

用于验证 MoM 在保持性能的同时，是否真正降低了上下文开销。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.eval.metrics import format_metrics

logger = logging.getLogger(__name__)


@dataclass
class CostSnapshot:
    """单次会话/评测的开销快照。

    Attributes:
        session_id: 会话 ID.
        num_turns: 对话轮次数.
        stats: 来自 MoMStats.to_dict() 的运行时统计.
        avg_l2_retrieved: 每轮平均检索到的 L2 对象数.
        avg_l3_retrieved: 每轮平均检索到的 L3 条目数.
        avg_memory_context_chars: 每轮注入的记忆上下文平均字符数.
        avg_memory_context_tokens: 每轮注入的记忆上下文平均 token 数 (估算).
        total_prompt_tokens: 累计 prompt token 数 (估算).
    """

    session_id: str = ""
    num_turns: int = 0
    stats: dict[str, int] = field(default_factory=dict)
    avg_l2_retrieved: float = 0.0
    avg_l3_retrieved: float = 0.0
    avg_memory_context_chars: float = 0.0
    avg_memory_context_tokens: float = 0.0
    total_prompt_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "num_turns": self.num_turns,
            "stats": self.stats,
            "avg_l2_retrieved": self.avg_l2_retrieved,
            "avg_l3_retrieved": self.avg_l3_retrieved,
            "avg_memory_context_chars": self.avg_memory_context_chars,
            "avg_memory_context_tokens": self.avg_memory_context_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
        }


class CostEvaluator:
    """开销评估器。

    从 SessionTrace / MoMStats 中收集开销数据并生成报告。

    Usage::

        evaluator = CostEvaluator()

        # 从 trace 评估
        trace_dict = session_trace.to_dict()
        report = evaluator.evaluate_trace(trace_dict, stats=agent.get_stats())

        # 批量评估
        report = evaluator.evaluate_batch(traces, stats_list)
    """

    # 粗略的 token 估算: 平均每个 token 约 4 个字符 (英文)
    CHARS_PER_TOKEN: float = 4.0

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.chars_per_token: float = self.config.get(
            "chars_per_token", self.CHARS_PER_TOKEN,
        )

    def evaluate_trace(
        self,
        trace: dict[str, Any],
        stats: dict[str, Any] | None = None,
    ) -> CostSnapshot:
        """评估单个会话 trace 的开销。

        Args:
            trace: SessionTrace.to_dict() 输出.
            stats: 可选的 MoMStats.to_dict() 输出. 若不提供则从 trace 中提取.

        Returns:
            CostSnapshot.
        """
        snapshot = CostSnapshot()
        snapshot.session_id = trace.get("session_id", "")

        turns = trace.get("turns", [])
        snapshot.num_turns = len(turns)

        # 使用提供的 stats 或从 trace 中提取
        if stats is not None:
            snapshot.stats = dict(stats)
        else:
            snapshot.stats = dict(trace.get("stats", {}))

        if not turns:
            return snapshot

        # ---- 统计每轮的检索条目数 ---- #
        l2_counts: list[int] = []
        l3_counts: list[int] = []
        context_char_counts: list[int] = []

        for turn in turns:
            l2_count = turn.get("l2_retrieved_count", 0)
            l3_count = turn.get("l3_retrieved_count", 0)
            l2_counts.append(l2_count)
            l3_counts.append(l3_count)

            # 估算记忆上下文字符数
            memory_ctx = turn.get("memory_context", "")
            if isinstance(memory_ctx, str):
                context_char_counts.append(len(memory_ctx))
            else:
                context_char_counts.append(0)

        n = len(turns)
        snapshot.avg_l2_retrieved = sum(l2_counts) / n
        snapshot.avg_l3_retrieved = sum(l3_counts) / n
        snapshot.avg_memory_context_chars = sum(context_char_counts) / n
        snapshot.avg_memory_context_tokens = (
            snapshot.avg_memory_context_chars / self.chars_per_token
        )

        # 累计 prompt token 估算
        total_chars = sum(context_char_counts)
        snapshot.total_prompt_tokens = int(total_chars / self.chars_per_token)

        return snapshot

    def evaluate_batch(
        self,
        traces: Sequence[dict[str, Any]],
        stats_list: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """批量评估多个会话的开销。

        Args:
            traces: SessionTrace.to_dict() 列表.
            stats_list: 对应的 MoMStats.to_dict() 列表 (可选).

        Returns:
            {
                "snapshots": [CostSnapshot, ...],
                "metrics": {
                    "total_turns", "total_l1_writes", "total_l1_reads",
                    "total_l2_aggregates", "total_l2_retrieves", "total_l2_merges",
                    "total_l3_summarizes", "total_l3_revises",
                    "avg_l2_retrieved_per_turn", "avg_l3_retrieved_per_turn",
                    "avg_memory_context_tokens_per_turn",
                    "total_estimated_prompt_tokens",
                },
                "summary": str,
            }
        """
        snapshots: list[CostSnapshot] = []

        for i, trace in enumerate(traces):
            stats = stats_list[i] if stats_list and i < len(stats_list) else None
            snapshot = self.evaluate_trace(trace, stats=stats)
            snapshots.append(snapshot)

        # ---- 聚合指标 ---- #
        total_turns = sum(s.num_turns for s in snapshots)

        # 从 stats 中聚合各操作计数
        total_l1_writes = sum(s.stats.get("l1_write_count", 0) for s in snapshots)
        total_l1_reads = sum(s.stats.get("l1_read_count", 0) for s in snapshots)
        total_l2_aggregates = sum(s.stats.get("l2_aggregate_count", 0) for s in snapshots)
        total_l2_retrieves = sum(s.stats.get("l2_retrieve_count", 0) for s in snapshots)
        total_l2_merges = sum(s.stats.get("l2_merge_count", 0) for s in snapshots)
        total_l3_summarizes = sum(s.stats.get("l3_summarize_count", 0) for s in snapshots)
        total_l3_revises = sum(s.stats.get("l3_revise_count", 0) for s in snapshots)
        total_prompt_tokens = sum(s.total_prompt_tokens for s in snapshots)

        n_sessions = len(snapshots) if snapshots else 1

        metrics: dict[str, float] = {
            "num_sessions": float(n_sessions),
            "total_turns": float(total_turns),
            # L1
            "total_l1_writes": float(total_l1_writes),
            "total_l1_reads": float(total_l1_reads),
            "l1_writes_per_turn": total_l1_writes / max(total_turns, 1),
            "l1_reads_per_turn": total_l1_reads / max(total_turns, 1),
            # L2
            "total_l2_aggregates": float(total_l2_aggregates),
            "total_l2_retrieves": float(total_l2_retrieves),
            "total_l2_merges": float(total_l2_merges),
            "avg_l2_retrieved_per_turn": (
                sum(s.avg_l2_retrieved * s.num_turns for s in snapshots)
                / max(total_turns, 1)
            ),
            # L3
            "total_l3_summarizes": float(total_l3_summarizes),
            "total_l3_revises": float(total_l3_revises),
            # 上下文开销
            "avg_l3_retrieved_per_turn": (
                sum(s.avg_l3_retrieved * s.num_turns for s in snapshots)
                / max(total_turns, 1)
            ),
            "avg_memory_context_tokens_per_turn": (
                sum(s.avg_memory_context_tokens * s.num_turns for s in snapshots)
                / max(total_turns, 1)
            ),
            "total_estimated_prompt_tokens": float(total_prompt_tokens),
        }

        summary = (
            f"CostEval: {n_sessions} sessions, {total_turns} turns\n"
            + format_metrics(metrics, prefix="  ")
        )

        logger.info(f"[CostEvaluator] {summary}")

        return {
            "snapshots": snapshots,
            "metrics": metrics,
            "summary": summary,
        }

    def compare_configs(
        self,
        reports: dict[str, dict[str, Any]],
    ) -> str:
        """比较不同配置下的开销报告。

        Args:
            reports: 配置名 → evaluate_batch() 返回值的字典.
                例如: {"swa_only": report1, "swa_mom": report2}

        Returns:
            对比表格的文本表示.
        """
        if not reports:
            return "No reports to compare."

        # 收集所有指标键
        all_keys: set[str] = set()
        for report in reports.values():
            all_keys.update(report.get("metrics", {}).keys())

        sorted_keys = sorted(all_keys)

        # 构建表格
        lines: list[str] = []
        header = "| Metric | " + " | ".join(reports.keys()) + " |"
        separator = "|" + "---|" * (len(reports) + 1)
        lines.append(header)
        lines.append(separator)

        for key in sorted_keys:
            row_values: list[str] = []
            for config_name in reports:
                val = reports[config_name].get("metrics", {}).get(key, 0.0)
                if isinstance(val, float):
                    row_values.append(f"{val:.2f}")
                else:
                    row_values.append(str(val))
            row = f"| {key} | " + " | ".join(row_values) + " |"
            lines.append(row)

        return "\n".join(lines)
