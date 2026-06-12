"""
profile_task — 用户画像建模合成任务。

测试目标：
- 用户近期研究方向识别
- 长期偏好的响应风格
- 长期项目身份识别
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileSample:
    """单条画像测试样本。"""
    sample_id: str
    # 多轮对话历史
    conversation: list[tuple[str, str]]
    # 画像查询
    query: str
    # 期望的画像标签 / 关键词
    expected_labels: list[str]
    # 子类型: research_topic / response_style / project_identity
    profile_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- 画像模板 ----

_RESEARCH_TOPICS = [
    {
        "topic": "大语言模型稀疏化",
        "keywords": ["稀疏化", "剪枝", "pruning", "sparse"],
        "conversations": [
            ("user", "我最近在研究LLM的结构化剪枝方法。"),
            ("assistant", "结构化剪枝是一个很有前景的方向，常见的方法包括..."),
            ("user", "SparseGPT和Wanda的对比你怎么看？"),
            ("assistant", "SparseGPT使用二阶信息进行权重重构，而Wanda利用激活幅值..."),
            ("user", "我在尝试将Channel Pruning应用到MoE架构上。"),
            ("assistant", "MoE的剪枝比较特殊，需要考虑Expert-level和Intra-expert两个层面..."),
        ],
    },
    {
        "topic": "Agent记忆系统",
        "keywords": ["记忆", "memory", "agent", "层次化"],
        "conversations": [
            ("user", "我在设计一个层次化的Agent记忆架构。"),
            ("assistant", "层次化记忆是一个很好的设计思路，可以分为短期、情景、语义三层..."),
            ("user", "L1用关联矩阵做在线记忆，L2做情景对象，L3做语义画像。"),
            ("assistant", "这个设计很合理，关联矩阵可以高效地进行在线更新..."),
            ("user", "我想让这个系统能和SWA模型配合使用。"),
            ("assistant", "SWA的局部注意力窗口之外的信息可以通过记忆系统来补偿..."),
        ],
    },
    {
        "topic": "分布式训练优化",
        "keywords": ["分布式", "FSDP", "distributed", "训练"],
        "conversations": [
            ("user", "我在用FSDP的HYBRID_SHARD策略训练70B模型。"),
            ("assistant", "HYBRID_SHARD可以在节点内做full shard，节点间做replicate..."),
            ("user", "遇到了NCCL超时的问题，4机32卡的环境。"),
            ("assistant", "NCCL超时常见原因包括网络拥塞、集合通信不匹配..."),
            ("user", "Gradient Checkpointing和CPU Offload的组合效果如何？"),
            ("assistant", "两者配合可以显著降低显存，但会增加计算时间..."),
        ],
    },
]

_RESPONSE_STYLES = [
    {
        "style": "技术性、结构化",
        "keywords": ["技术", "结构化", "详细", "technical"],
        "conversations": [
            ("user", "请用技术性的方式回答我的问题，最好有结构化的分点说明。"),
            ("assistant", "好的，我会用结构化的方式进行技术解答。"),
            ("user", "解释一下RoPE位置编码。"),
            ("assistant", "## RoPE位置编码\n\n1. **核心思想**: 通过旋转矩阵编码相对位置...\n2. **数学形式**: ..."),
            ("user", "很好，就是这种风格，以后都这样回答。"),
            ("assistant", "明白了，以后我会保持这种技术性、结构化的回答风格。"),
        ],
    },
    {
        "style": "简洁直接",
        "keywords": ["简洁", "直接", "concise", "简短"],
        "conversations": [
            ("user", "回答别太长，简洁点就行。"),
            ("assistant", "好的，简洁回答。"),
            ("user", "什么是KV Cache？"),
            ("assistant", "KV Cache缓存已计算的Key/Value，避免自回归推理时重复计算。"),
            ("user", "对，就这个长度挺好。"),
            ("assistant", "收到，保持简洁。"),
        ],
    },
]

_PROJECT_IDENTITIES = [
    {
        "project": "MoM-Agent",
        "keywords": ["MoM", "Mixture-of-Memory", "Agent记忆"],
        "conversations": [
            ("user", "我的项目叫MoM-Agent，全称Mixture-of-Memory Agent。"),
            ("assistant", "了解了，MoM-Agent项目。"),
            ("user", "目标是为局部注意力模型提供层次化记忆补偿。"),
            ("assistant", "这是一个很有意义的研究方向。"),
            ("user", "最终希望能在论文中展示SWA+MoM能逼近全注意力的效果。"),
            ("assistant", "这个实验设计可以很好地证明记忆系统的价值。"),
        ],
    },
    {
        "project": "SLoRB",
        "keywords": ["SLoRB", "低秩", "稀疏", "剪枝框架"],
        "conversations": [
            ("user", "我在开发SLoRB框架，结合稀疏化和低秩分解。"),
            ("assistant", "稀疏+低秩是一种互补的压缩策略。"),
            ("user", "目前支持LLaMA和Qwen系列模型。"),
            ("assistant", "这两个系列覆盖了主流的开源LLM。"),
            ("user", "下一步要加入MoE模型的支持。"),
            ("assistant", "MoE的稀疏化有独特的挑战和机会。"),
        ],
    },
]


class ProfileTask:
    """
    用户画像建模合成任务。

    生成包含明确用户偏好/研究方向/项目信息的对话，
    在对话结束后提问画像相关问题，检验记忆系统的画像建模能力。
    """

    def __init__(
        self,
        num_samples: int = 30,
        num_distractor_turns: int = 2,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_distractor_turns = num_distractor_turns
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 数据生成
    # ------------------------------------------------------------------

    def generate_samples(self) -> list[ProfileSample]:
        """生成全部画像测试样本。"""
        samples: list[ProfileSample] = []
        types = ["research_topic", "response_style", "project_identity"]
        per_type = self.num_samples // len(types)

        for ptype in types:
            for _ in range(per_type):
                samples.append(self._generate_one(ptype))

        while len(samples) < self.num_samples:
            t = self.rng.choice(types)
            samples.append(self._generate_one(t))

        self.rng.shuffle(samples)
        return samples

    def _generate_one(self, profile_type: str) -> ProfileSample:
        """生成单条画像样本。"""
        if profile_type == "research_topic":
            tmpl = self.rng.choice(_RESEARCH_TOPICS)
            query = "我最近在研究什么方向？"
        elif profile_type == "response_style":
            tmpl = self.rng.choice(_RESPONSE_STYLES)
            query = "我偏好什么样的回答风格？"
        elif profile_type == "project_identity":
            tmpl = self.rng.choice(_PROJECT_IDENTITIES)
            query = "我的项目是什么？"
        else:
            raise ValueError(f"未知画像类型: {profile_type}")

        # 构建对话：模板对话 + 干扰轮
        conv = list(tmpl["conversations"])
        conv.extend(self._distractor_turns())

        return ProfileSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            query=query,
            expected_labels=tmpl["keywords"],
            profile_type=profile_type,
            metadata={
                "source_key": tmpl.get("topic") or tmpl.get("style") or tmpl.get("project"),
            },
        )

    # ------------------------------------------------------------------
    # 干扰生成
    # ------------------------------------------------------------------

    _DISTRACTORS = [
        [("user", "最近有什么好看的电影？"), ("assistant", "最近口碑不错的有几部科幻片...")],
        [("user", "Python 3.12有什么新特性？"), ("assistant", "3.12引入了更好的错误提示...")],
        [("user", "帮我写一个快速排序。"), ("assistant", "def quicksort(arr): ...")],
    ]

    def _distractor_turns(self) -> list[tuple[str, str]]:
        """生成干扰轮。"""
        turns: list[tuple[str, str]] = []
        for _ in range(self.num_distractor_turns):
            pair = self.rng.choice(self._DISTRACTORS)
            turns.extend(pair)
        return turns

    # ------------------------------------------------------------------
    # 评估接口
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        sample: ProfileSample,
        predicted: str,
    ) -> dict[str, Any]:
        """
        评估单条画像样本。

        使用关键词命中率作为指标：
        - hit: predicted中包含了多少个expected_labels
        - precision: 命中数 / 期望标签数
        """
        hits = [
            label for label in sample.expected_labels
            if label.lower() in predicted.lower()
        ]
        hit_count = len(hits)
        total_labels = len(sample.expected_labels)

        return {
            "sample_id": sample.sample_id,
            "profile_type": sample.profile_type,
            "hit_count": hit_count,
            "total_labels": total_labels,
            "precision": hit_count / max(total_labels, 1),
            "hits": hits,
            "expected_labels": sample.expected_labels,
            "predicted": predicted,
        }

    def evaluate_batch(
        self,
        samples: list[ProfileSample],
        predictions: list[str],
    ) -> dict[str, Any]:
        """批量评估画像任务。"""
        assert len(samples) == len(predictions), "样本数和预测数不匹配"

        results = [
            self.evaluate_single(s, p)
            for s, p in zip(samples, predictions)
        ]

        # 总体平均精度
        avg_precision = sum(r["precision"] for r in results) / max(len(results), 1)

        # 各类型精度
        type_stats: dict[str, list[float]] = {}
        for r in results:
            t = r["profile_type"]
            if t not in type_stats:
                type_stats[t] = []
            type_stats[t].append(r["precision"])

        type_avg_precision = {
            t: sum(scores) / max(len(scores), 1)
            for t, scores in type_stats.items()
        }

        return {
            "avg_precision": avg_precision,
            "total_samples": len(results),
            "type_avg_precision": type_avg_precision,
            "details": results,
        }
