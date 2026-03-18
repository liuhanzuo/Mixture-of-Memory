"""
合成记忆数据集。

生成多种需要长程记忆能力的文本样本，用于快速调试和验证 memory 系统。

支持的任务类型：
  1. simple_recall          — 简单关联回忆
  2. updated_preference     — 偏好更新后查询最新值
  3. past_value_query       — 偏好更新后查询旧值
  4. long_distance_recall   — 长距离从句式回忆
  5. distractor_heavy       — 大量干扰项中的回忆

每个样本输出为纯文本字符串，由 tokenizer 在 collate 阶段进行编码。
同时提供结构化元信息（答案位置、正确答案等）用于指标计算。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset


# ============================================================
# 实体 & 属性池
# ============================================================

_DEFAULT_NAMES: List[str] = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
    "Henry", "Ivy", "Jack", "Karen", "Leo", "Mona", "Nick", "Olivia",
    "Paul", "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy",
    "Xander", "Yuki", "Zara", "Amber", "Blake", "Cleo", "Derek",
    "Elena", "Felix", "Gina", "Hugo", "Iris", "Jay", "Kira", "Liam",
    "Nora", "Oscar", "Piper", "Reed", "Stella", "Troy", "Vera",
    "Wade", "Xena", "Yara", "Zeke", "Aria",
]

_DEFAULT_ATTRIBUTES: Dict[str, List[str]] = {
    "fruit": ["apple", "banana", "cherry", "grape", "mango",
              "orange", "peach", "pear", "plum", "strawberry",
              "watermelon", "kiwi", "lemon", "lime", "papaya",
              "blueberry", "raspberry", "fig", "coconut", "pomegranate"],
    "color": ["red", "blue", "green", "yellow", "purple",
              "orange", "pink", "black", "white", "gray",
              "cyan", "magenta", "teal", "indigo", "violet",
              "gold", "silver", "brown", "beige", "crimson"],
    "animal": ["cat", "dog", "rabbit", "parrot", "turtle",
               "hamster", "fish", "horse", "snake", "lizard",
               "eagle", "dolphin", "owl", "fox", "deer",
               "wolf", "bear", "penguin", "koala", "panda"],
    "food": ["pizza", "sushi", "pasta", "burger", "salad",
             "tacos", "ramen", "curry", "steak", "soup",
             "sandwich", "dumpling", "pancake", "waffle", "pie",
             "noodle", "risotto", "kebab", "falafel", "paella"],
    "city": ["Tokyo", "Paris", "London", "Berlin", "Sydney",
             "Toronto", "Seoul", "Mumbai", "Cairo", "Rome",
             "Vienna", "Prague", "Dublin", "Oslo", "Lima",
             "Lisbon", "Athens", "Bangkok", "Hanoi", "Nairobi"],
}


# ============================================================
# 样本数据结构
# ============================================================

@dataclass
class SyntheticSample:
    """一条合成样本的完整信息。"""
    text: str                          # 拼接后的纯文本
    task_type: str                     # 任务类型标识
    answer: str                        # 正确答案（用于 exact match）
    query: str                         # 查询文本
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata 可包含: entity, attribute_type, old_value, new_value 等


# ============================================================
# 句子生成工具
# ============================================================

class _SentenceFactory:
    """根据配置随机生成各类句子片段。"""

    def __init__(
        self,
        names: Optional[List[str]] = None,
        attributes: Optional[Dict[str, List[str]]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.names = list(names or _DEFAULT_NAMES)
        self.attributes = dict(attributes or _DEFAULT_ATTRIBUTES)
        self.attr_types = list(self.attributes.keys())
        self.rng = rng or random.Random(42)

    # ---------- 基础采样 ----------

    def sample_entity(self, exclude: Sequence[str] = ()) -> str:
        pool = [n for n in self.names if n not in exclude]
        return self.rng.choice(pool)

    def sample_attr_type(self) -> str:
        return self.rng.choice(self.attr_types)

    def sample_attr_value(self, attr_type: str, exclude: Sequence[str] = ()) -> str:
        pool = [v for v in self.attributes[attr_type] if v not in exclude]
        return self.rng.choice(pool)

    # ---------- 句子模板 ----------

    def fact_sentence(self, entity: str, attr_type: str, value: str) -> str:
        """如: 'Alice likes banana.' """
        templates = [
            f"{entity} likes {value}.",
            f"{entity}'s favorite {attr_type} is {value}.",
            f"{entity} prefers {value}.",
            f"The {attr_type} that {entity} likes is {value}.",
        ]
        return self.rng.choice(templates)

    def update_sentence(self, entity: str, attr_type: str, new_value: str) -> str:
        """如: 'Later, Alice likes apple.' """
        templates = [
            f"Later, {entity} likes {new_value}.",
            f"Now, {entity}'s favorite {attr_type} is {new_value}.",
            f"{entity} changed preference to {new_value}.",
            f"Recently, {entity} prefers {new_value}.",
        ]
        return self.rng.choice(templates)

    def long_distance_fact(self, entity: str, attr_type: str, value: str) -> str:
        """如: 'Alice likes a fruit, which is banana.' """
        templates = [
            f"{entity} likes a {attr_type}, which is {value}.",
            f"{entity} has a favorite {attr_type}, and it is {value}.",
            f"There is a {attr_type} that {entity} really likes, namely {value}.",
        ]
        return self.rng.choice(templates)

    def question_current(self, entity: str, attr_type: str) -> str:
        templates = [
            f"Question: What {attr_type} does {entity} like?",
            f"Question: What is {entity}'s favorite {attr_type}?",
        ]
        return self.rng.choice(templates)

    def question_current_after_update(self, entity: str, attr_type: str) -> str:
        templates = [
            f"Question: What {attr_type} does {entity} like now?",
            f"Question: What is {entity}'s current favorite {attr_type}?",
        ]
        return self.rng.choice(templates)

    def question_past(self, entity: str, attr_type: str) -> str:
        templates = [
            f"Question: What {attr_type} did {entity} like before?",
            f"Question: What was {entity}'s previous favorite {attr_type}?",
        ]
        return self.rng.choice(templates)

    def distractor_sentence(self, exclude_entities: Sequence[str] = ()) -> str:
        """生成一条无关干扰句。"""
        ent = self.sample_entity(exclude=exclude_entities)
        at = self.sample_attr_type()
        val = self.sample_attr_value(at)
        return self.fact_sentence(ent, at, val)


# ============================================================
# 任务生成器
# ============================================================

class _TaskGenerator:
    """为每种任务类型生成 (sentences, question, answer, metadata)。"""

    def __init__(self, factory: _SentenceFactory, num_distractors: int = 10):
        self.f = factory
        self.num_distractors = num_distractors

    def _pad_with_distractors(
        self,
        core_sentences: List[str],
        insert_positions: Optional[List[int]] = None,
        exclude_entities: Sequence[str] = (),
    ) -> List[str]:
        """在 core_sentences 之间随机插入干扰句。

        Args:
            core_sentences: 核心句子列表（按顺序）。
            insert_positions: 如果指定，则仅在这些位置后插入干扰句；
                              否则在所有间隔中均匀分配。
            exclude_entities: 干扰句中不使用的实体名。

        Returns:
            混合后的句子列表。
        """
        result: List[str] = []
        n = len(core_sentences)
        # 在每两个核心句子之间均匀分配干扰句
        per_gap = max(1, self.num_distractors // max(n, 1))

        for i, sent in enumerate(core_sentences):
            # 在当前核心句前插入干扰
            if i > 0:
                count = per_gap if insert_positions is None or (i - 1) in insert_positions else 0
                for _ in range(count):
                    result.append(self.f.distractor_sentence(exclude_entities))
            result.append(sent)

        # 剩余干扰句追加到末尾（问题前）
        remaining = self.num_distractors - (per_gap * max(n - 1, 0))
        for _ in range(max(0, remaining)):
            result.append(self.f.distractor_sentence(exclude_entities))

        return result

    # ---- 1. Simple Recall ----
    def simple_recall(self) -> SyntheticSample:
        entity = self.f.sample_entity()
        attr_type = self.f.sample_attr_type()
        value = self.f.sample_attr_value(attr_type)

        fact = self.f.fact_sentence(entity, attr_type, value)
        question = self.f.question_current(entity, attr_type)

        sentences = self._pad_with_distractors([fact], exclude_entities=[entity])
        sentences.append(question)
        answer_line = f"Answer: {value}"
        sentences.append(answer_line)

        text = " ".join(sentences)
        return SyntheticSample(
            text=text,
            task_type="simple_recall",
            answer=value,
            query=question,
            metadata={"entity": entity, "attr_type": attr_type, "value": value},
        )

    # ---- 2. Updated Preference ----
    def updated_preference(self) -> SyntheticSample:
        entity = self.f.sample_entity()
        attr_type = self.f.sample_attr_type()
        old_value = self.f.sample_attr_value(attr_type)
        new_value = self.f.sample_attr_value(attr_type, exclude=[old_value])

        fact = self.f.fact_sentence(entity, attr_type, old_value)
        update = self.f.update_sentence(entity, attr_type, new_value)
        question = self.f.question_current_after_update(entity, attr_type)

        sentences = self._pad_with_distractors([fact, update], exclude_entities=[entity])
        sentences.append(question)
        answer_line = f"Answer: {new_value}"
        sentences.append(answer_line)

        text = " ".join(sentences)
        return SyntheticSample(
            text=text,
            task_type="updated_preference",
            answer=new_value,
            query=question,
            metadata={
                "entity": entity, "attr_type": attr_type,
                "old_value": old_value, "new_value": new_value,
            },
        )

    # ---- 3. Past Value Query ----
    def past_value_query(self) -> SyntheticSample:
        entity = self.f.sample_entity()
        attr_type = self.f.sample_attr_type()
        old_value = self.f.sample_attr_value(attr_type)
        new_value = self.f.sample_attr_value(attr_type, exclude=[old_value])

        fact = self.f.fact_sentence(entity, attr_type, old_value)
        update = self.f.update_sentence(entity, attr_type, new_value)
        question = self.f.question_past(entity, attr_type)

        sentences = self._pad_with_distractors([fact, update], exclude_entities=[entity])
        sentences.append(question)
        answer_line = f"Answer: {old_value}"
        sentences.append(answer_line)

        text = " ".join(sentences)
        return SyntheticSample(
            text=text,
            task_type="past_value_query",
            answer=old_value,
            query=question,
            metadata={
                "entity": entity, "attr_type": attr_type,
                "old_value": old_value, "new_value": new_value,
            },
        )

    # ---- 4. Long Distance Recall ----
    def long_distance_recall(self) -> SyntheticSample:
        entity = self.f.sample_entity()
        attr_type = self.f.sample_attr_type()
        value = self.f.sample_attr_value(attr_type)

        fact = self.f.long_distance_fact(entity, attr_type, value)
        question = self.f.question_current(entity, attr_type)

        # 使用更多干扰句来增加距离
        extra_distractors = self.num_distractors * 2
        orig_nd = self.num_distractors
        self.num_distractors = extra_distractors
        sentences = self._pad_with_distractors([fact], exclude_entities=[entity])
        self.num_distractors = orig_nd

        sentences.append(question)
        answer_line = f"Answer: {value}"
        sentences.append(answer_line)

        text = " ".join(sentences)
        return SyntheticSample(
            text=text,
            task_type="long_distance_recall",
            answer=value,
            query=question,
            metadata={"entity": entity, "attr_type": attr_type, "value": value},
        )

    # ---- 5. Distractor Heavy ----
    def distractor_heavy(self) -> SyntheticSample:
        """多个实体和属性，干扰项密集。"""
        # 创建多个无关 fact，其中只有一个是目标
        target_entity = self.f.sample_entity()
        attr_type = self.f.sample_attr_type()
        target_value = self.f.sample_attr_value(attr_type)

        target_fact = self.f.fact_sentence(target_entity, attr_type, target_value)
        question = self.f.question_current(target_entity, attr_type)

        # 生成额外的 "真实但无关" 的事实（其他实体同一属性类型）
        confusing_facts: List[str] = []
        used_entities = [target_entity]
        for _ in range(min(5, len(self.f.names) - 1)):
            ent = self.f.sample_entity(exclude=used_entities)
            used_entities.append(ent)
            val = self.f.sample_attr_value(attr_type, exclude=[target_value])
            confusing_facts.append(self.f.fact_sentence(ent, attr_type, val))

        # 将目标事实随机混入
        all_facts = confusing_facts.copy()
        insert_pos = self.f.rng.randint(0, len(all_facts))
        all_facts.insert(insert_pos, target_fact)

        sentences = self._pad_with_distractors(all_facts, exclude_entities=[target_entity])
        sentences.append(question)
        answer_line = f"Answer: {target_value}"
        sentences.append(answer_line)

        text = " ".join(sentences)
        return SyntheticSample(
            text=text,
            task_type="distractor_heavy",
            answer=target_value,
            query=question,
            metadata={
                "entity": target_entity, "attr_type": attr_type,
                "value": target_value, "num_confusing": len(confusing_facts),
            },
        )


# ============================================================
# 任务注册表
# ============================================================

_TASK_REGISTRY = {
    "simple_recall": "_TaskGenerator.simple_recall",
    "updated_preference": "_TaskGenerator.updated_preference",
    "past_value_query": "_TaskGenerator.past_value_query",
    "long_distance_recall": "_TaskGenerator.long_distance_recall",
    "distractor_heavy": "_TaskGenerator.distractor_heavy",
}

ALL_TASK_TYPES: List[str] = list(_TASK_REGISTRY.keys())


# ============================================================
# PyTorch Dataset
# ============================================================

class SyntheticMemoryDataset(Dataset):
    """合成记忆测试数据集。

    在初始化时一次性生成所有样本并缓存，保证可复现。

    Args:
        num_samples: 样本总数。
        task_types: 启用的任务类型列表。如果为 None 则使用全部类型。
        num_entities: 实体名称池大小（从默认池中截取前 N 个）。
        num_attributes: 每种属性类型的可选值数（从默认池中截取前 N 个）。
        num_distractors: 每个样本中的干扰句数。
        seed: 随机种子。
        tokenizer: 可选的 HuggingFace tokenizer，如果提供则同时返回 token ids。
        max_length: tokenizer 的最大长度（仅在提供 tokenizer 时有效）。
    """

    def __init__(
        self,
        num_samples: int = 1000,
        task_types: Optional[List[str]] = None,
        num_entities: int = 50,
        num_attributes: int = 20,
        num_distractors: int = 10,
        seed: int = 42,
        tokenizer: Any = None,
        max_length: int = 512,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.task_types = task_types or ALL_TASK_TYPES
        self.seed = seed
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 验证任务类型
        for t in self.task_types:
            if t not in _TASK_REGISTRY:
                raise ValueError(
                    f"未知任务类型: {t}，可选: {ALL_TASK_TYPES}"
                )

        # 构建受限的名称 / 属性池
        names = _DEFAULT_NAMES[:num_entities]
        attributes = {
            k: v[:num_attributes] for k, v in _DEFAULT_ATTRIBUTES.items()
        }

        rng = random.Random(seed)
        factory = _SentenceFactory(names=names, attributes=attributes, rng=rng)
        generator = _TaskGenerator(factory=factory, num_distractors=num_distractors)

        # 一次性生成所有样本
        self.samples: List[SyntheticSample] = []
        task_methods = {
            "simple_recall": generator.simple_recall,
            "updated_preference": generator.updated_preference,
            "past_value_query": generator.past_value_query,
            "long_distance_recall": generator.long_distance_recall,
            "distractor_heavy": generator.distractor_heavy,
        }

        for i in range(num_samples):
            task_type = self.task_types[i % len(self.task_types)]
            sample = task_methods[task_type]()
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回单条样本。

        Returns:
            dict 包含:
              - text (str): 原始文本
              - task_type (str): 任务类型
              - answer (str): 正确答案
              - query (str): 查询文本
              - metadata (dict): 额外信息
              - input_ids (Tensor, 可选): tokenized ids
              - attention_mask (Tensor, 可选): attention mask
              - labels (Tensor, 可选): 用于 CLM 的 labels
        """
        sample = self.samples[idx]
        item: Dict[str, Any] = {
            "text": sample.text,
            "task_type": sample.task_type,
            "answer": sample.answer,
            "query": sample.query,
            "metadata": sample.metadata,
        }

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sample.text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)
            # CLM labels: 与 input_ids 相同，padding 处设为 -100
            labels = encoded["input_ids"].squeeze(0).clone()
            labels[encoded["attention_mask"].squeeze(0) == 0] = -100
            item["labels"] = labels

        return item

    def get_task_distribution(self) -> Dict[str, int]:
        """统计各任务类型的样本数。"""
        counts: Dict[str, int] = {}
        for s in self.samples:
            counts[s.task_type] = counts.get(s.task_type, 0) + 1
        return counts


# ============================================================
# 便捷构建函数
# ============================================================

def build_synthetic_splits(
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 500,
    task_types: Optional[List[str]] = None,
    num_entities: int = 50,
    num_attributes: int = 20,
    num_distractors: int = 10,
    seed: int = 42,
    tokenizer: Any = None,
    max_length: int = 512,
) -> Tuple[SyntheticMemoryDataset, SyntheticMemoryDataset, SyntheticMemoryDataset]:
    """一次性构建 train / val / test 三个数据集。

    使用不同的种子以确保三个集合之间没有重叠。

    Args:
        train_size: 训练集样本数。
        val_size: 验证集样本数。
        test_size: 测试集样本数。
        task_types: 启用的任务类型。
        num_entities: 实体池大小。
        num_attributes: 每种属性类型的值数。
        num_distractors: 每样本干扰句数。
        seed: 基础随机种子。
        tokenizer: 可选 tokenizer。
        max_length: 最大序列长度。

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    common_kwargs = dict(
        task_types=task_types,
        num_entities=num_entities,
        num_attributes=num_attributes,
        num_distractors=num_distractors,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_ds = SyntheticMemoryDataset(
        num_samples=train_size, seed=seed, **common_kwargs
    )
    val_ds = SyntheticMemoryDataset(
        num_samples=val_size, seed=seed + 10000, **common_kwargs
    )
    test_ds = SyntheticMemoryDataset(
        num_samples=test_size, seed=seed + 20000, **common_kwargs
    )

    return train_ds, val_ds, test_ds


def build_synthetic_from_config(
    cfg: Any,
    tokenizer: Any = None,
) -> Tuple[SyntheticMemoryDataset, SyntheticMemoryDataset, SyntheticMemoryDataset]:
    """从 OmegaConf 配置构建数据集。

    Args:
        cfg: 包含 ``synthetic`` 子配置的 DictConfig。
        tokenizer: 可选 tokenizer。

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    syn = cfg.synthetic
    return build_synthetic_splits(
        train_size=syn.train_size,
        val_size=syn.val_size,
        test_size=syn.test_size,
        task_types=list(syn.task_types) if syn.get("task_types") else None,
        num_entities=syn.num_entities,
        num_attributes=syn.num_attributes,
        num_distractors=syn.num_distractors,
        seed=syn.seed,
        tokenizer=tokenizer,
        max_length=cfg.get("block", {}).get("block_size", 512) * 4,
    )
