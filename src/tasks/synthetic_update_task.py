"""
synthetic_update_task — 合成记忆更新任务。

测试目标：
- 覆写行为（overwrite）：新信息应覆盖旧信息
- 过期记忆处理（stale-memory）：过期状态不应被错误持久化
- 临时状态 vs 长期状态：区分短期事实和长期事实
- 错误持久化（wrong persistence）：矛盾信息的处理
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UpdateSample:
    """单条更新测试样本。"""
    sample_id: str
    # 对话历史（list of (role, content) 元组）
    conversation: list[tuple[str, str]]
    # 在对话中引入的事实更新
    updates: list[dict[str, Any]]
    # 查询：在对话结束后要求系统回答的问题
    query: str
    # 期望答案
    expected_answer: str
    # 标签：overwrite / stale / contradiction / temporary
    update_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SyntheticUpdateTask:
    """
    合成记忆更新任务。

    生成包含事实覆写、过期信息、矛盾信息的多轮对话，
    在对话结束后提问，检验记忆系统是否正确更新。
    """

    # ---- 事实模板 ----
    FACT_TEMPLATES: list[dict[str, Any]] = [
        {
            "category": "location",
            "key": "user_city",
            "values": ["北京", "上海", "深圳", "杭州", "成都", "纽约", "伦敦"],
            "query_template": "我现在住在哪个城市？",
        },
        {
            "category": "preference",
            "key": "favorite_language",
            "values": ["Python", "Rust", "C++", "Go", "TypeScript", "Java"],
            "query_template": "我最喜欢的编程语言是什么？",
        },
        {
            "category": "project",
            "key": "current_project",
            "values": [
                "稀疏化训练框架",
                "Agent记忆系统",
                "多模态检索引擎",
                "分布式推理服务",
                "CUDA算子库",
            ],
            "query_template": "我目前在做什么项目？",
        },
        {
            "category": "status",
            "key": "mood",
            "values": ["很开心", "有点疲惫", "非常专注", "比较焦虑", "心情平静"],
            "query_template": "我现在的状态怎么样？",
        },
        {
            "category": "tool",
            "key": "editor",
            "values": ["VS Code", "Neovim", "PyCharm", "Cursor", "Emacs"],
            "query_template": "我用什么编辑器？",
        },
    ]

    def __init__(
        self,
        num_samples: int = 50,
        num_distractor_turns: int = 3,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_distractor_turns = num_distractor_turns
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 数据生成
    # ------------------------------------------------------------------

    def generate_samples(self) -> list[UpdateSample]:
        """生成全部测试样本，平均分配到四类更新类型。"""
        samples: list[UpdateSample] = []
        types = ["overwrite", "stale", "contradiction", "temporary"]
        per_type = self.num_samples // len(types)

        for update_type in types:
            for _ in range(per_type):
                sample = self._generate_one(update_type)
                samples.append(sample)

        # 补齐余数
        while len(samples) < self.num_samples:
            t = self.rng.choice(types)
            samples.append(self._generate_one(t))

        self.rng.shuffle(samples)
        return samples

    def _generate_one(self, update_type: str) -> UpdateSample:
        """生成单条样本。"""
        tmpl = self.rng.choice(self.FACT_TEMPLATES)
        values = tmpl["values"]

        if update_type == "overwrite":
            return self._gen_overwrite(tmpl, values)
        elif update_type == "stale":
            return self._gen_stale(tmpl, values)
        elif update_type == "contradiction":
            return self._gen_contradiction(tmpl, values)
        elif update_type == "temporary":
            return self._gen_temporary(tmpl, values)
        else:
            raise ValueError(f"未知更新类型: {update_type}")

    def _gen_overwrite(self, tmpl: dict, values: list[str]) -> UpdateSample:
        """
        覆写测试：先告诉系统事实A，再更新为事实B。
        期望系统回答B。
        """
        old_val, new_val = self.rng.sample(values, 2)
        conv = []
        updates = []

        # 第一轮：建立旧事实
        conv.append(("user", f"告诉你一下，{tmpl['key']}是{old_val}。"))
        conv.append(("assistant", f"好的，我记住了，{tmpl['key']}是{old_val}。"))
        updates.append({"turn": 0, "key": tmpl["key"], "value": old_val, "action": "set"})

        # 插入干扰轮
        conv.extend(self._distractor_turns())

        # 第二轮：覆写为新事实
        conv.append(("user", f"更新一下，{tmpl['key']}现在变成{new_val}了。"))
        conv.append(("assistant", f"已更新，{tmpl['key']}现在是{new_val}。"))
        updates.append({"turn": len(conv) // 2 - 1, "key": tmpl["key"], "value": new_val, "action": "overwrite"})

        return UpdateSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            updates=updates,
            query=tmpl["query_template"],
            expected_answer=new_val,
            update_type="overwrite",
        )

    def _gen_stale(self, tmpl: dict, values: list[str]) -> UpdateSample:
        """
        过期记忆测试：先设定事实A，后明确告知A已过期/不再有效。
        期望系统回答"不确定"或"已过期"，而不是继续返回A。
        """
        old_val = self.rng.choice(values)
        conv = []
        updates = []

        conv.append(("user", f"{tmpl['key']}是{old_val}。"))
        conv.append(("assistant", f"好的，记住了。"))
        updates.append({"turn": 0, "key": tmpl["key"], "value": old_val, "action": "set"})

        conv.extend(self._distractor_turns())

        conv.append(("user", f"之前说的{tmpl['key']}已经不准了，忘掉吧。"))
        conv.append(("assistant", f"好的，已经清除了{tmpl['key']}的信息。"))
        updates.append({"turn": len(conv) // 2 - 1, "key": tmpl["key"], "value": None, "action": "invalidate"})

        return UpdateSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            updates=updates,
            query=tmpl["query_template"],
            expected_answer="<unknown>",
            update_type="stale",
        )

    def _gen_contradiction(self, tmpl: dict, values: list[str]) -> UpdateSample:
        """
        矛盾测试：在同一轮中给出矛盾的信息。
        期望系统能识别矛盾，保留后者或标记不确定。
        """
        val_a, val_b = self.rng.sample(values, 2)
        conv = []
        updates = []

        conv.append(("user", f"{tmpl['key']}是{val_a}，不对，其实是{val_b}。"))
        conv.append(("assistant", f"好的，{tmpl['key']}是{val_b}。"))
        updates.append({"turn": 0, "key": tmpl["key"], "value": val_a, "action": "set"})
        updates.append({"turn": 0, "key": tmpl["key"], "value": val_b, "action": "correct"})

        conv.extend(self._distractor_turns())

        return UpdateSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            updates=updates,
            query=tmpl["query_template"],
            expected_answer=val_b,
            update_type="contradiction",
        )

    def _gen_temporary(self, tmpl: dict, values: list[str]) -> UpdateSample:
        """
        临时状态测试：告诉系统一个临时状态（如心情），
        经过多轮后询问，期望系统不会错误持久化临时状态。
        """
        temp_val = self.rng.choice(values)
        conv = []
        updates = []

        conv.append(("user", f"我现在暂时{temp_val}，不过一会儿就好了。"))
        conv.append(("assistant", f"了解，希望你之后会好起来。"))
        updates.append({"turn": 0, "key": tmpl["key"], "value": temp_val, "action": "temporary"})

        # 插入较多干扰轮，模拟时间流逝
        for _ in range(self.num_distractor_turns * 2):
            conv.extend(self._single_distractor())

        return UpdateSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            updates=updates,
            query=tmpl["query_template"],
            expected_answer="<unknown>",
            update_type="temporary",
            metadata={"note": "临时状态不应被长期持久化"},
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    _DISTRACTOR_TOPICS = [
        ("user", "今天天气怎么样？"),
        ("user", "帮我解释一下Transformer的attention机制。"),
        ("user", "推荐一本关于分布式系统的书。"),
        ("user", "Python的GIL是什么？"),
        ("user", "CUDA编程有什么注意事项？"),
    ]

    _DISTRACTOR_RESPONSES = [
        ("assistant", "今天是多云转晴的天气。"),
        ("assistant", "Transformer的attention机制通过Q/K/V计算注意力权重..."),
        ("assistant", "推荐《Designing Data-Intensive Applications》。"),
        ("assistant", "GIL是Python的全局解释器锁，限制了多线程并行..."),
        ("assistant", "CUDA编程需要注意显存管理、线程同步、warp divergence等。"),
    ]

    def _single_distractor(self) -> list[tuple[str, str]]:
        """生成一轮干扰对话。"""
        idx = self.rng.randrange(len(self._DISTRACTOR_TOPICS))
        return [self._DISTRACTOR_TOPICS[idx], self._DISTRACTOR_RESPONSES[idx]]

    def _distractor_turns(self) -> list[tuple[str, str]]:
        """生成若干轮干扰对话。"""
        turns: list[tuple[str, str]] = []
        for _ in range(self.num_distractor_turns):
            turns.extend(self._single_distractor())
        return turns

    # ------------------------------------------------------------------
    # 评估接口
    # ------------------------------------------------------------------

    def evaluate_single(self, sample: UpdateSample, predicted: str) -> dict[str, Any]:
        """
        评估单条样本。

        Args:
            sample: 测试样本
            predicted: 系统给出的回答

        Returns:
            包含 exact_match, type, sample_id 的字典
        """
        expected = sample.expected_answer
        if expected == "<unknown>":
            # 对于"未知"类答案，检查系统是否避免了错误回答
            # 简单启发式：如果回答中包含任何原始值，视为错误
            original_values = [
                u["value"] for u in sample.updates
                if u["value"] is not None
            ]
            is_correct = not any(v in predicted for v in original_values)
        else:
            is_correct = expected in predicted

        return {
            "sample_id": sample.sample_id,
            "update_type": sample.update_type,
            "exact_match": is_correct,
            "expected": expected,
            "predicted": predicted,
        }

    def evaluate_batch(
        self,
        samples: list[UpdateSample],
        predictions: list[str],
    ) -> dict[str, Any]:
        """
        批量评估。

        Returns:
            包含总体和各类型准确率的字典。
        """
        assert len(samples) == len(predictions), "样本数和预测数不匹配"

        results = [
            self.evaluate_single(s, p)
            for s, p in zip(samples, predictions)
        ]

        # 总体准确率
        total_correct = sum(r["exact_match"] for r in results)
        total = len(results)

        # 各类型准确率
        type_stats: dict[str, dict[str, int]] = {}
        for r in results:
            t = r["update_type"]
            if t not in type_stats:
                type_stats[t] = {"correct": 0, "total": 0}
            type_stats[t]["total"] += 1
            if r["exact_match"]:
                type_stats[t]["correct"] += 1

        type_accuracy = {
            t: s["correct"] / max(s["total"], 1)
            for t, s in type_stats.items()
        }

        return {
            "overall_accuracy": total_correct / max(total, 1),
            "total_samples": total,
            "total_correct": total_correct,
            "type_accuracy": type_accuracy,
            "type_stats": type_stats,
            "details": results,
        }
