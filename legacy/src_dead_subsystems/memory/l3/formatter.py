"""L3 格式化器: 将画像记忆导出为 Markdown 和 JSON 格式。

支持:
- 导出 profile.md (人类可读的画像文档)
- 导出 profile.json (机器可读的结构化数据)
- 格式化单条/多条画像条目为 prompt 上下文
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.memory.l3.summarizer import L3ProfileEntry
from src.memory.l3.profile_store import L3ProfileStore

logger = logging.getLogger(__name__)


class L3Formatter:
    """L3 画像记忆格式化与导出工具。"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.title: str = self.config.get("title", "User Profile")
        self.include_metadata: bool = self.config.get("include_metadata", True)
        logger.info(f"[L3 Formatter] Initialized. title='{self.title}'")

    # ---- Markdown 导出 ----

    def to_markdown(self, entries: list[L3ProfileEntry]) -> str:
        """将画像条目列表格式化为 Markdown 文档。

        按 category 分组, 每组一个二级标题, 每条目一个列表项。
        """
        if not entries:
            return f"# {self.title}\n\n_No profile entries available._\n"

        # 按 category 分组
        grouped: dict[str, list[L3ProfileEntry]] = {}
        for entry in entries:
            grouped.setdefault(entry.category, []).append(entry)

        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"_Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
        lines.append("")

        # 分类标题映射
        category_titles = {
            "research_interest": "🔬 Research Interests",
            "preference": "⚙️ Preferences",
            "long_term_project": "📋 Active Projects",
            "identity": "👤 Identity & State",
            "factual": "📝 Known Facts",
        }

        for category, cat_entries in grouped.items():
            title = category_titles.get(category, f"📌 {category.replace('_', ' ').title()}")
            lines.append(f"## {title}")
            lines.append("")

            # 按置信度降序排列
            cat_entries.sort(key=lambda e: e.confidence, reverse=True)

            for entry in cat_entries:
                confidence_bar = self._confidence_bar(entry.confidence)
                lines.append(f"- **{entry.key}**: {entry.value}")
                if self.include_metadata:
                    lines.append(f"  - Confidence: {confidence_bar} ({entry.confidence:.2f})")
                    lines.append(f"  - Last updated: {entry.last_updated_at}")
                    if entry.evidence_ids:
                        evidence_str = ", ".join(entry.evidence_ids[:5])
                        if len(entry.evidence_ids) > 5:
                            evidence_str += f" ... (+{len(entry.evidence_ids) - 5} more)"
                        lines.append(f"  - Evidence: {evidence_str}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _confidence_bar(confidence: float, length: int = 10) -> str:
        """生成文本形式的置信度条。"""
        filled = int(confidence * length)
        return "█" * filled + "░" * (length - filled)

    def export_markdown(self, store: L3ProfileStore, output_path: str | Path) -> Path:
        """将 ProfileStore 中的所有条目导出为 Markdown 文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        entries = store.list_all()
        md_content = self.to_markdown(entries)

        output_path.write_text(md_content, encoding="utf-8")
        logger.info(f"[L3 Formatter] Exported Markdown profile to {output_path} ({len(entries)} entries)")
        return output_path

    # ---- JSON 导出 ----

    def to_json(self, entries: list[L3ProfileEntry], indent: int = 2) -> str:
        """将画像条目列表格式化为 JSON 字符串。"""
        data = {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(entries),
            "entries": [entry.to_dict() for entry in entries],
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def export_json(self, store: L3ProfileStore, output_path: str | Path) -> Path:
        """将 ProfileStore 中的所有条目导出为 JSON 文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        entries = store.list_all()
        json_content = self.to_json(entries)

        output_path.write_text(json_content, encoding="utf-8")
        logger.info(f"[L3 Formatter] Exported JSON profile to {output_path} ({len(entries)} entries)")
        return output_path

    # ---- Prompt 格式化 ----

    def format_for_prompt(
        self,
        entries: list[L3ProfileEntry],
        max_entries: int = 10,
        max_chars: int = 2000,
    ) -> str:
        """将画像条目格式化为可直接插入 prompt 的文本块。

        Args:
            entries: 画像条目列表 (应已按相关性排序).
            max_entries: 最多包含的条目数.
            max_chars: 最大字符数限制.

        Returns:
            格式化后的 prompt 文本块.
        """
        if not entries:
            return "[No user profile available]"

        lines: list[str] = []
        lines.append("[User Profile Context]")

        total_chars = len(lines[0])
        count = 0

        for entry in entries[:max_entries]:
            line = f"- [{entry.category}] {entry.key}: {entry.value} (conf={entry.confidence:.2f})"
            if total_chars + len(line) + 1 > max_chars:
                lines.append("- ... (truncated)")
                break
            lines.append(line)
            total_chars += len(line) + 1
            count += 1

        lines.append(f"[{count} profile entries loaded]")
        return "\n".join(lines)

    def format_summary(self, store: L3ProfileStore) -> str:
        """生成画像存储的简要统计摘要。"""
        entries = store.list_all()
        if not entries:
            return "Profile store is empty."

        # 按 category 统计
        cat_counts: dict[str, int] = {}
        for e in entries:
            cat_counts[e.category] = cat_counts.get(e.category, 0) + 1

        avg_confidence = sum(e.confidence for e in entries) / len(entries)

        parts: list[str] = [
            f"Total entries: {len(entries)}",
            f"Average confidence: {avg_confidence:.3f}",
            "Categories: " + ", ".join(f"{k}({v})" for k, v in sorted(cat_counts.items())),
        ]
        return " | ".join(parts)
