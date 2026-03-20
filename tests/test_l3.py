"""
L3 语义/画像级记忆的单元测试。

覆盖:
- L3ProfileEntry 数据类
- L3Summarizer (规则总结器)
- L3ProfileStore (增删改查、衰减归档)
- L3Reviser (冲突修订)
- L3Formatter (导出 Markdown / JSON)
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.memory.l2.types import L2MemoryObject
from src.memory.l3.summarizer import (
    L3ProfileEntry,
    L3Summarizer,
    RuleBasedSummarizer,
)
from src.memory.l3.profile_store import L3ProfileStore
from src.memory.l3.reviser import L3Reviser
from src.memory.l3.formatter import L3Formatter


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_entry() -> L3ProfileEntry:
    """创建一个示例 L3 条目。"""
    return L3ProfileEntry(
        entry_id="e001",
        key="research_interest",
        value="The user is researching large language models.",
        confidence=0.9,
        evidence_ids=["obj_001", "obj_002"],
        category="research_interest",
    )


@pytest.fixture
def sample_l2_objects() -> list[L2MemoryObject]:
    """创建示例 L2 对象用于总结。"""
    return [
        L2MemoryObject(
            object_id="obj_t1", object_type="topic",
            summary_text="large language model training",
            confidence=0.8, source_turn_ids=["t1"],
        ),
        L2MemoryObject(
            object_id="obj_t2", object_type="topic",
            summary_text="sparse training research",
            confidence=0.7, source_turn_ids=["t2"],
        ),
        L2MemoryObject(
            object_id="obj_p1", object_type="preference",
            summary_text="using Python for coding",
            confidence=0.9, source_turn_ids=["t3"],
        ),
        L2MemoryObject(
            object_id="obj_task1", object_type="task",
            summary_text="building a hierarchical memory agent",
            confidence=0.85, source_turn_ids=["t4"],
        ),
    ]


@pytest.fixture
def profile_store() -> L3ProfileStore:
    """创建 L3 ProfileStore。"""
    return L3ProfileStore(config={
        "max_entries": 10,
        "decay_rate": 0.1,
        "archive_threshold": 0.2,
    })


@pytest.fixture
def populated_store(profile_store: L3ProfileStore) -> L3ProfileStore:
    """创建一个预填充的 store。"""
    entries = [
        L3ProfileEntry(
            entry_id="e001", key="research_interest",
            value="User researches LLM", confidence=0.9,
            category="research_interest",
        ),
        L3ProfileEntry(
            entry_id="e002", key="preference",
            value="User prefers Python", confidence=0.8,
            category="preference",
        ),
        L3ProfileEntry(
            entry_id="e003", key="long_term_project",
            value="User builds memory agents", confidence=0.7,
            category="long_term_project",
        ),
    ]
    for e in entries:
        profile_store.add(e)
    return profile_store


# ------------------------------------------------------------------ #
#  测试: L3ProfileEntry
# ------------------------------------------------------------------ #

class TestL3ProfileEntry:
    """测试 L3ProfileEntry 数据类。"""

    def test_basic_fields(self, sample_entry: L3ProfileEntry):
        """基本字段应正确。"""
        assert sample_entry.entry_id == "e001"
        assert sample_entry.key == "research_interest"
        assert sample_entry.confidence == 0.9
        assert len(sample_entry.evidence_ids) == 2

    def test_to_dict(self, sample_entry: L3ProfileEntry):
        """to_dict 应返回可序列化字典。"""
        d = sample_entry.to_dict()
        assert d["entry_id"] == "e001"
        assert d["category"] == "research_interest"
        assert isinstance(d["evidence_ids"], list)

    def test_default_values(self):
        """默认值应合理。"""
        entry = L3ProfileEntry(
            entry_id="e_default",
            key="test_key",
            value="test_value",
        )
        assert entry.confidence == 1.0
        assert entry.evidence_ids == []
        assert entry.category == "factual"
        assert entry.created_at  # 应有默认时间戳
        assert entry.last_updated_at


# ------------------------------------------------------------------ #
#  测试: L3Summarizer
# ------------------------------------------------------------------ #

class TestL3Summarizer:
    """测试 L3 总结器。"""

    def test_rule_based_summarizer(self, sample_l2_objects: list[L2MemoryObject]):
        """RuleBasedSummarizer 应从 L2 对象生成 L3 条目。"""
        summarizer = RuleBasedSummarizer()
        entries = summarizer.summarize(sample_l2_objects)
        assert len(entries) >= 1

        # 检查每个条目的基本字段
        for entry in entries:
            assert entry.entry_id
            assert entry.key
            assert entry.value
            assert entry.category

    def test_summarizer_categories(self, sample_l2_objects: list[L2MemoryObject]):
        """应生成不同类别的条目。"""
        summarizer = RuleBasedSummarizer()
        entries = summarizer.summarize(sample_l2_objects)
        categories = {e.category for e in entries}
        # 对应 topic → research_interest, preference → preference
        assert "research_interest" in categories
        assert "preference" in categories

    def test_summarizer_empty_input(self):
        """空输入应返回空列表。"""
        summarizer = RuleBasedSummarizer()
        assert summarizer.summarize([]) == []

    def test_summarizer_skips_archived(self):
        """应跳过已归档的 L2 对象。"""
        summarizer = RuleBasedSummarizer()
        archived_obj = L2MemoryObject(
            object_id="archived_1", object_type="topic",
            summary_text="old topic", is_archived=True,
        )
        active_obj = L2MemoryObject(
            object_id="active_1", object_type="preference",
            summary_text="using Python",
        )
        entries = summarizer.summarize([archived_obj, active_obj])
        # 归档对象不应参与总结
        for entry in entries:
            assert "archived_1" not in entry.evidence_ids

    def test_summarizer_evidence_ids(self, sample_l2_objects: list[L2MemoryObject]):
        """生成的条目应包含来源 L2 对象的 ID。"""
        summarizer = RuleBasedSummarizer()
        entries = summarizer.summarize(sample_l2_objects)
        for entry in entries:
            assert len(entry.evidence_ids) > 0

    def test_l3_summarizer_facade(self, sample_l2_objects: list[L2MemoryObject]):
        """L3Summarizer 外观类应使用默认 rule_based 后端。"""
        summarizer = L3Summarizer(config={"summarizer_backend": "rule_based"})
        entries = summarizer.summarize(sample_l2_objects)
        assert len(entries) >= 1

    def test_l3_summarizer_invalid_backend(self):
        """无效后端应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unknown summarizer backend"):
            L3Summarizer(config={"summarizer_backend": "nonexistent"})


# ------------------------------------------------------------------ #
#  测试: L3ProfileStore
# ------------------------------------------------------------------ #

class TestL3ProfileStore:
    """测试 L3 画像存储。"""

    def test_add_and_get(self, profile_store: L3ProfileStore, sample_entry: L3ProfileEntry):
        """add 和 get 应正确工作。"""
        profile_store.add(sample_entry)
        retrieved = profile_store.get("e001")
        assert retrieved is not None
        assert retrieved.key == "research_interest"

    def test_size(self, populated_store: L3ProfileStore):
        """size 应返回正确数量。"""
        assert populated_store.size() == 3

    def test_get_by_key(self, populated_store: L3ProfileStore):
        """get_by_key 应正确过滤。"""
        entries = populated_store.get_by_key("research_interest")
        assert len(entries) == 1
        assert entries[0].key == "research_interest"

    def test_get_by_category(self, populated_store: L3ProfileStore):
        """get_by_category 应正确过滤。"""
        entries = populated_store.get_by_category("preference")
        assert len(entries) == 1
        assert entries[0].category == "preference"

    def test_update(self, populated_store: L3ProfileStore):
        """update 应修改字段。"""
        result = populated_store.update("e001", confidence=0.5)
        assert result is True
        entry = populated_store.get("e001")
        assert entry is not None
        assert entry.confidence == 0.5

    def test_update_nonexistent(self, populated_store: L3ProfileStore):
        """更新不存在的条目应返回 False。"""
        result = populated_store.update("nonexistent", confidence=0.5)
        assert result is False

    def test_remove(self, populated_store: L3ProfileStore):
        """remove 应删除条目。"""
        result = populated_store.remove("e001")
        assert result is True
        assert populated_store.get("e001") is None
        assert populated_store.size() == 2

    def test_remove_nonexistent(self, populated_store: L3ProfileStore):
        """删除不存在的条目应返回 False。"""
        result = populated_store.remove("nonexistent")
        assert result is False

    def test_list_all_sorted(self, populated_store: L3ProfileStore):
        """list_all 应按置信度降序排列。"""
        entries = populated_store.list_all()
        for i in range(len(entries) - 1):
            assert entries[i].confidence >= entries[i + 1].confidence

    def test_clear(self, populated_store: L3ProfileStore):
        """clear 应清空所有条目。"""
        populated_store.clear()
        assert populated_store.size() == 0

    def test_decay_all(self, populated_store: L3ProfileStore):
        """decay_all 应降低置信度并可能归档。"""
        # 初始置信度: 0.9, 0.8, 0.7; decay_rate=0.1
        # 一次衰减后: 0.81, 0.72, 0.63 (均 > archive_threshold=0.2)
        archived = populated_store.decay_all()
        assert len(archived) == 0

        # 多次衰减直到有条目被归档
        for _ in range(20):
            populated_store.decay_all()
        assert populated_store.size() < 3  # 应有条目被归档

    def test_capacity_eviction(self):
        """超出 max_entries 时应淘汰最低置信度条目。"""
        store = L3ProfileStore(config={"max_entries": 3})
        for i in range(5):
            entry = L3ProfileEntry(
                entry_id=f"e{i:03d}", key=f"key_{i}",
                value=f"value_{i}", confidence=i * 0.2,
                category="factual",
            )
            store.add(entry)
        assert store.size() == 3
        # 应保留置信度最高的 3 个
        all_entries = store.list_all()
        confidences = [e.confidence for e in all_entries]
        assert min(confidences) >= 0.4  # 0.0 和 0.2 应被淘汰

    def test_merge_entry_new_key(self, populated_store: L3ProfileStore):
        """merge_entry 遇到新 key 时应直接添加。"""
        new_entry = L3ProfileEntry(
            entry_id="e_brand_new", key="brand_new_key",
            value="Completely new info",
            confidence=0.6, category="factual",
        )
        populated_store.merge_entry(new_entry)
        assert populated_store.size() == 4

    def test_merge_entry_existing_key(self, populated_store: L3ProfileStore):
        """merge_entry 遇到同 key+category 时应合并。"""
        new_entry = L3ProfileEntry(
            entry_id="e999", key="research_interest",
            value="User now researches multimodal models",
            confidence=0.95,
            evidence_ids=["obj_new"],
            category="research_interest",
        )
        populated_store.merge_entry(new_entry)
        # 不应增加新条目 (合并到 e001)
        assert populated_store.size() == 3
        merged = populated_store.get("e001")
        assert merged is not None
        assert merged.confidence == 0.95  # 取最高

    def test_serialization_roundtrip(self, populated_store: L3ProfileStore):
        """序列化和反序列化应保持一致。"""
        dicts = populated_store.to_list_of_dicts()
        assert len(dicts) == 3

        new_store = L3ProfileStore()
        new_store.load_from_dicts(dicts)
        assert new_store.size() == 3

        for orig, loaded in zip(populated_store.list_all(), new_store.list_all()):
            assert orig.entry_id == loaded.entry_id
            assert orig.key == loaded.key
            assert orig.value == loaded.value


# ------------------------------------------------------------------ #
#  测试: L3Reviser
# ------------------------------------------------------------------ #

class TestL3Reviser:
    """测试 L3 修订器。"""

    def test_no_conflict(self, populated_store: L3ProfileStore):
        """无冲突时 detect_conflicts 应返回空列表。"""
        reviser = L3Reviser(config={"merge_threshold": 0.8, "contradiction_threshold": 0.3})
        new_entry = L3ProfileEntry(
            entry_id="e_new", key="new_info",
            value="User lives in Beijing",
            confidence=0.7, category="factual",
        )
        conflicts = reviser.detect_conflicts([new_entry], populated_store)
        assert len(conflicts) == 0

    def test_redundant_detection(self, populated_store: L3ProfileStore):
        """高度相似的条目应检测为 redundant。"""
        reviser = L3Reviser(config={"merge_threshold": 0.6, "contradiction_threshold": 0.2})
        new_entry = L3ProfileEntry(
            entry_id="e_dup", key="research_interest",
            value="User researches LLM",  # 与 e001 完全相同
            confidence=0.85, category="research_interest",
        )
        conflicts = reviser.detect_conflicts([new_entry], populated_store)
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == "redundant"

    def test_contradiction_detection(self, populated_store: L3ProfileStore):
        """完全矛盾的条目应检测为 contradiction。"""
        reviser = L3Reviser(config={"merge_threshold": 0.8, "contradiction_threshold": 0.3})
        new_entry = L3ProfileEntry(
            entry_id="e_contra", key="research_interest",
            value="The user has no interest in AI or ML at all",
            confidence=0.95, category="research_interest",
        )
        conflicts = reviser.detect_conflicts([new_entry], populated_store)
        assert len(conflicts) >= 1
        # 由于文本差异大, 应为 contradiction 或 update
        assert conflicts[0].conflict_type in ("contradiction", "update")

    def test_apply_revisions_no_conflict(self, populated_store: L3ProfileStore):
        """apply_revisions 无冲突时应返回所有新条目。"""
        reviser = L3Reviser()
        new_entries = [
            L3ProfileEntry(
                entry_id="e_new1", key="hobbies",
                value="User enjoys hiking", confidence=0.6,
                category="factual",
            ),
        ]
        result = reviser.apply_revisions(new_entries, populated_store)
        assert len(result) == 1
        assert result[0].entry_id == "e_new1"

    def test_conflict_log(self, populated_store: L3ProfileStore):
        """冲突记录应被记入日志。"""
        reviser = L3Reviser(config={"merge_threshold": 0.6, "contradiction_threshold": 0.2})
        new_entry = L3ProfileEntry(
            entry_id="e_dup", key="research_interest",
            value="User researches LLM",
            confidence=0.85, category="research_interest",
        )
        reviser.apply_revisions([new_entry], populated_store)
        log = reviser.get_conflict_log()
        assert len(log) >= 1

    def test_clear_conflict_log(self):
        """clear_conflict_log 应清空日志。"""
        reviser = L3Reviser()
        reviser.conflict_log.append("dummy")  # type: ignore
        reviser.clear_conflict_log()
        assert len(reviser.get_conflict_log()) == 0


# ------------------------------------------------------------------ #
#  测试: L3Formatter
# ------------------------------------------------------------------ #

class TestL3Formatter:
    """测试 L3 格式化器。"""

    def test_to_markdown(self, populated_store: L3ProfileStore):
        """to_markdown 应返回合法的 Markdown 文本。"""
        formatter = L3Formatter()
        entries = populated_store.list_all()
        md = formatter.to_markdown(entries)
        assert "# User Profile" in md
        assert "##" in md  # 应有二级标题 (按 category 分组)

    def test_to_markdown_empty(self):
        """空条目应返回含提示信息的 Markdown。"""
        formatter = L3Formatter()
        md = formatter.to_markdown([])
        assert "No profile entries" in md

    def test_to_json(self, populated_store: L3ProfileStore):
        """to_json 应返回合法的 JSON 字符串。"""
        formatter = L3Formatter()
        entries = populated_store.list_all()
        json_str = formatter.to_json(entries)
        data = json.loads(json_str)
        assert data["total_entries"] == 3
        assert len(data["entries"]) == 3

    def test_export_markdown(self, populated_store: L3ProfileStore):
        """export_markdown 应生成 .md 文件。"""
        formatter = L3Formatter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.md"
            result = formatter.export_markdown(populated_store, path)
            assert result.exists()
            content = result.read_text()
            assert "# User Profile" in content

    def test_export_json(self, populated_store: L3ProfileStore):
        """export_json 应生成 .json 文件。"""
        formatter = L3Formatter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            result = formatter.export_json(populated_store, path)
            assert result.exists()
            data = json.loads(result.read_text())
            assert isinstance(data, dict)
            assert data["total_entries"] == 3

    def test_format_for_prompt(self, populated_store: L3ProfileStore):
        """format_for_prompt 应返回可注入 prompt 的文本。"""
        formatter = L3Formatter()
        entries = populated_store.list_all()
        text = formatter.format_for_prompt(entries)
        assert "[User Profile Context]" in text
        assert "profile entries loaded" in text

    def test_format_for_prompt_empty(self):
        """空条目应返回提示文本。"""
        formatter = L3Formatter()
        text = formatter.format_for_prompt([])
        assert "No user profile" in text

    def test_format_for_prompt_truncation(self, populated_store: L3ProfileStore):
        """max_chars 限制应生效。"""
        formatter = L3Formatter()
        entries = populated_store.list_all()
        text = formatter.format_for_prompt(entries, max_chars=50)
        assert len(text) < 200  # 应被截断

    def test_format_summary(self, populated_store: L3ProfileStore):
        """format_summary 应返回统计摘要文本。"""
        formatter = L3Formatter()
        text = formatter.format_summary(populated_store)
        assert "Total entries: 3" in text
        assert "Average confidence" in text

    def test_format_summary_empty(self):
        """空 store 应返回提示文本。"""
        formatter = L3Formatter()
        empty_store = L3ProfileStore()
        text = formatter.format_summary(empty_store)
        assert "empty" in text.lower()

    def test_confidence_bar(self):
        """_confidence_bar 应返回正确长度的条形图。"""
        bar = L3Formatter._confidence_bar(0.7, length=10)
        assert len(bar) == 10
        assert "█" in bar
        assert "░" in bar
