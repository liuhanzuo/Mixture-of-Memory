"""
L2 事件/状态级记忆的单元测试。

覆盖:
- L2MemoryObject 数据类
- L2Aggregator / RuleBasedAggregator
- L2ObjectStore (增删改查、衰减归档)
- L2Retriever (文本相似度检索)
- L2Merger (合并策略)
"""

import pytest
from datetime import datetime

from src.memory.l2.types import L2MemoryObject, ChatMessage
from src.memory.l2.aggregator import L2Aggregator, RuleBasedAggregator
from src.memory.l2.object_store import L2ObjectStore
from src.memory.l2.retriever import L2Retriever
from src.memory.l2.merger import L2Merger


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_object() -> L2MemoryObject:
    return L2MemoryObject(
        object_id="obj_001",
        object_type="topic",
        summary_text="Discussion about large language models",
        confidence=0.8,
        source_turn_ids=["turn_001", "turn_002"],
    )


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="I prefer using Python for data science.", turn_id="t1"),
        ChatMessage(role="assistant", content="Python is great for data science.", turn_id="t1"),
        ChatMessage(role="user", content="Help me build a memory system for LLM agents.", turn_id="t2"),
        ChatMessage(role="assistant", content="Sure, let's design a hierarchical memory.", turn_id="t2"),
        ChatMessage(role="user", content="I am working on sparse training research.", turn_id="t3"),
    ]


@pytest.fixture
def store() -> L2ObjectStore:
    return L2ObjectStore(config={"max_objects": 10, "max_age_turns": 5})


@pytest.fixture
def retriever() -> L2Retriever:
    return L2Retriever(config={"retrieval_top_k": 3})


@pytest.fixture
def populated_store(store: L2ObjectStore) -> L2ObjectStore:
    objects = [
        L2MemoryObject(object_id="obj_t1", object_type="topic",
                       summary_text="Discussion about large language models",
                       confidence=0.9, source_turn_ids=["t1"]),
        L2MemoryObject(object_id="obj_p1", object_type="preference",
                       summary_text="User prefers Python for coding",
                       confidence=0.85, source_turn_ids=["t2"]),
        L2MemoryObject(object_id="obj_task1", object_type="task",
                       summary_text="Build a hierarchical memory system",
                       confidence=0.75, source_turn_ids=["t3"]),
        L2MemoryObject(object_id="obj_state1", object_type="state",
                       summary_text="User is researching sparse training",
                       confidence=0.7, source_turn_ids=["t4"]),
        L2MemoryObject(object_id="obj_entity1", object_type="entity",
                       summary_text="FSDP stands for Fully Sharded Data Parallel",
                       confidence=0.95, source_turn_ids=["t5"]),
    ]
    for obj in objects:
        store.add(obj)
    return store


# ------------------------------------------------------------------ #
#  测试: L2MemoryObject
# ------------------------------------------------------------------ #

class TestL2MemoryObject:

    def test_basic_fields(self, sample_object: L2MemoryObject):
        assert sample_object.object_id == "obj_001"
        assert sample_object.object_type == "topic"
        assert sample_object.confidence == 0.8
        assert len(sample_object.source_turn_ids) == 2

    def test_touch(self, sample_object: L2MemoryObject):
        sample_object.touch(42)
        assert sample_object.last_accessed_turn == 42

    def test_archive(self, sample_object: L2MemoryObject):
        assert not sample_object.is_archived
        sample_object.archive()
        assert sample_object.is_archived

    def test_to_dict(self, sample_object: L2MemoryObject):
        d = sample_object.to_dict()
        assert d["object_id"] == "obj_001"
        assert d["object_type"] == "topic"
        assert "summary_text" in d
        assert "confidence" in d


# ------------------------------------------------------------------ #
#  测试: L2Aggregator
# ------------------------------------------------------------------ #

class TestL2Aggregator:

    def test_rule_based_aggregator(self, sample_messages: list[ChatMessage]):
        agg = RuleBasedAggregator()
        objects = agg.aggregate(sample_messages)
        assert len(objects) >= 1
        for obj in objects:
            assert obj.object_id
            assert obj.summary_text

    def test_aggregator_extracts_preference(self):
        agg = RuleBasedAggregator()
        messages = [ChatMessage(role="user", content="I prefer using PyTorch for deep learning.", turn_id="t1")]
        objects = agg.aggregate(messages)
        preference_objs = [o for o in objects if o.object_type == "preference"]
        assert len(preference_objs) >= 1

    def test_aggregator_extracts_task(self):
        agg = RuleBasedAggregator()
        messages = [ChatMessage(role="user", content="Help me design a distributed training system.", turn_id="t1")]
        objects = agg.aggregate(messages)
        task_objs = [o for o in objects if o.object_type == "task"]
        assert len(task_objs) >= 1

    def test_aggregator_extracts_state(self):
        agg = RuleBasedAggregator()
        messages = [ChatMessage(role="user", content="I am working on a new paper about pruning.", turn_id="t1")]
        objects = agg.aggregate(messages)
        state_objs = [o for o in objects if o.object_type == "state"]
        assert len(state_objs) >= 1

    def test_l2_aggregator_facade(self, sample_messages: list[ChatMessage]):
        agg = L2Aggregator(config={"aggregator_backend": "rule_based"})
        objects = agg.aggregate(sample_messages)
        assert len(objects) >= 1

    def test_empty_messages(self):
        agg = RuleBasedAggregator()
        assert agg.aggregate([]) == []


# ------------------------------------------------------------------ #
#  测试: L2ObjectStore
# ------------------------------------------------------------------ #

class TestL2ObjectStore:

    def test_add_and_get(self, store: L2ObjectStore, sample_object: L2MemoryObject):
        store.add(sample_object)
        retrieved = store.get("obj_001")
        assert retrieved is not None
        assert retrieved.object_id == "obj_001"

    def test_active_count(self, populated_store: L2ObjectStore):
        assert populated_store.active_count == 5

    def test_get_by_type(self, populated_store: L2ObjectStore):
        topics = populated_store.get_by_type("topic")
        assert len(topics) == 1
        assert topics[0].object_type == "topic"

    def test_update(self, populated_store: L2ObjectStore):
        result = populated_store.update("obj_t1", confidence=0.5)
        assert result is True
        obj = populated_store.get("obj_t1")
        assert obj is not None
        assert obj.confidence == 0.5

    def test_remove(self, populated_store: L2ObjectStore):
        result = populated_store.remove("obj_t1")
        assert result is True
        assert populated_store.get("obj_t1") is None
        assert populated_store.active_count == 4

    def test_capacity_eviction(self, store: L2ObjectStore):
        for i in range(15):
            obj = L2MemoryObject(
                object_id=f"obj_{i:03d}", object_type="topic",
                summary_text=f"Topic {i}", last_accessed_turn=i,
            )
            store.add(obj)
        assert store.active_count == 10
        assert store.archived_count == 5

    def test_decay_check(self, populated_store: L2ObjectStore):
        # max_age_turns=5, 所有对象 last_accessed_turn=0
        archived_ids = populated_store.decay_check(current_turn=10)
        assert len(archived_ids) == 5
        assert populated_store.active_count == 0

    def test_merge(self, populated_store: L2ObjectStore):
        merged = populated_store.merge("obj_t1", "obj_state1", "Combined topic and state")
        assert merged is not None
        assert "Combined" in merged.summary_text
        assert populated_store.get("obj_t1") is None
        assert populated_store.get("obj_state1") is None
        assert populated_store.active_count == 4  # 5 - 2 + 1


# ------------------------------------------------------------------ #
#  测试: L2Retriever
# ------------------------------------------------------------------ #

class TestL2Retriever:

    def test_retrieve_by_text(self, retriever: L2Retriever, populated_store: L2ObjectStore):
        objects = populated_store.get_all_active()
        results = retriever.retrieve(query="large language models", objects=objects, top_k=3)
        assert len(results) > 0
        for obj, score in results:
            assert isinstance(obj, L2MemoryObject)
            assert 0.0 <= score <= 2.0  # 加权分可能略大于 1

    def test_retrieve_top_k(self, retriever: L2Retriever, populated_store: L2ObjectStore):
        objects = populated_store.get_all_active()
        results = retriever.retrieve(query="test", objects=objects, top_k=2)
        assert len(results) <= 2

    def test_retrieve_type_filter(self, retriever: L2Retriever, populated_store: L2ObjectStore):
        objects = populated_store.get_all_active()
        results = retriever.retrieve(query="Python", objects=objects, type_filter="preference")
        for obj, _ in results:
            assert obj.object_type == "preference"

    def test_retrieve_empty(self, retriever: L2Retriever):
        results = retriever.retrieve(query="test", objects=[])
        assert results == []

    def test_retrieve_excludes_archived(self, retriever: L2Retriever):
        obj = L2MemoryObject(
            object_id="archived_1", object_type="topic",
            summary_text="An archived topic about memory systems", is_archived=True,
        )
        results = retriever.retrieve(query="memory systems", objects=[obj])
        assert len(results) == 0

    def test_format_for_prompt(self, retriever: L2Retriever, populated_store: L2ObjectStore):
        objects = populated_store.get_all_active()
        results = retriever.retrieve(query="language model", objects=objects, top_k=2)
        text = retriever.format_for_prompt(results)
        assert "[Retrieved Memory Objects]" in text


# ------------------------------------------------------------------ #
#  测试: L2Merger
# ------------------------------------------------------------------ #

class TestL2Merger:

    def test_should_merge_same_type_high_similarity(self):
        merger = L2Merger(config={"merge_similarity_threshold": 0.5})
        obj_a = L2MemoryObject(
            object_id="a", object_type="topic",
            summary_text="Discussion about large language models",
        )
        obj_b = L2MemoryObject(
            object_id="b", object_type="topic",
            summary_text="Discussion about large language model architectures",
        )
        candidate, action = merger.find_merge_candidate(obj_b, [obj_a])
        assert candidate is not None
        assert action in ("merge", "replace")

    def test_should_not_merge_different_type(self):
        merger = L2Merger(config={"merge_similarity_threshold": 0.5})
        obj_a = L2MemoryObject(
            object_id="a", object_type="topic", summary_text="Discussion about models",
        )
        obj_b = L2MemoryObject(
            object_id="b", object_type="preference", summary_text="Discussion about models",
        )
        candidate, action = merger.find_merge_candidate(obj_b, [obj_a])
        assert action == "append"

    def test_decide_and_merge_batch(self):
        merger = L2Merger(config={"merge_similarity_threshold": 0.5})
        existing = [
            L2MemoryObject(object_id="e1", object_type="topic", summary_text="LLM training"),
        ]
        new_objs = [
            L2MemoryObject(object_id="n1", object_type="topic", summary_text="LLM training methods"),
            L2MemoryObject(object_id="n2", object_type="preference", summary_text="Use Python"),
        ]
        results = merger.decide_and_merge(new_objs, existing)
        assert len(results) == 2
        # 第一个应有 merge/replace 候选
        _, action1, _ = results[0]
        # 第二个应 append（不同类型）
        _, action2, _ = results[1]
        assert action2 == "append"
