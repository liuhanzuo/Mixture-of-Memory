"""
longhorizon_chat_task — 长程对话记忆合成任务。

测试目标：
- 长距离实体/事实回忆（long-distance entity/fact recall）
- 跨轮目标追踪（cross-turn goal tracking）
- 长程对话中的远程记忆访问（long-range memory access in long chat）
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LongHorizonSample:
    """单条长程对话测试样本。"""
    sample_id: str
    # 完整多轮对话（可能很长）
    conversation: list[tuple[str, str]]
    # 在对话末尾的查询
    query: str
    # 期望答案（关键词或精确值）
    expected_answer: str
    # 关键事实被提及的轮次索引（0-based，按 user 轮计数）
    fact_turn_index: int
    # 总 user 轮数
    total_turns: int
    # 子类型: entity_recall / goal_tracking / long_range_access
    task_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- 实体模板 ----

_ENTITY_FACTS = [
    {"entity": "张教授", "fact": "研究方向是量子计算", "query": "张教授的研究方向是什么？", "answer": "量子计算"},
    {"entity": "李明", "fact": "在Google Brain工作", "query": "李明在哪里工作？", "answer": "Google Brain"},
    {"entity": "项目Alpha", "fact": "截止日期是12月15日", "query": "项目Alpha的截止日期是什么时候？", "answer": "12月15日"},
    {"entity": "服务器A", "fact": "IP地址是192.168.1.100", "query": "服务器A的IP地址是多少？", "answer": "192.168.1.100"},
    {"entity": "会议室", "fact": "预定在3楼302房间", "query": "会议室在哪里？", "answer": "3楼302房间"},
    {"entity": "数据集X", "fact": "包含50万条训练样本", "query": "数据集X有多少训练样本？", "answer": "50万"},
    {"entity": "模型v2.3", "fact": "在MMLU上达到了78.5分", "query": "模型v2.3在MMLU上的分数是多少？", "answer": "78.5"},
    {"entity": "王博士", "fact": "推荐使用Flash Attention 2", "query": "王博士推荐使用什么注意力实现？", "answer": "Flash Attention 2"},
]

# ---- 目标追踪模板 ----

_GOAL_TEMPLATES = [
    {
        "initial_goal": "完成分布式训练框架的性能优化",
        "sub_goals": [
            "先测试单机8卡的吞吐量",
            "然后对比FSDP和DDP的通信开销",
            "最后写性能报告",
        ],
        "query": "我们最初设定的主要目标是什么？",
        "answer": "完成分布式训练框架的性能优化",
    },
    {
        "initial_goal": "将推理延迟降低到50ms以内",
        "sub_goals": [
            "先做模型量化",
            "然后优化KV Cache管理",
            "接着引入Speculative Decoding",
            "最后做端到端benchmark",
        ],
        "query": "我们的推理优化目标是什么？",
        "answer": "将推理延迟降低到50ms以内",
    },
    {
        "initial_goal": "构建一个支持百万token上下文的Agent",
        "sub_goals": [
            "先实现层次化记忆",
            "然后集成SWA backbone",
            "接着做长文档测试",
        ],
        "query": "我们最初的Agent构建目标是什么？",
        "answer": "构建一个支持百万token上下文的Agent",
    },
]

# ---- 干扰对话模板 ----

_FILLER_EXCHANGES = [
    [("user", "帮我解释一下什么是Mixture of Experts。"), ("assistant", "MoE是一种条件计算架构，通过路由网络选择性激活专家...")],
    [("user", "PyTorch 2.0的compile功能好用吗？"), ("assistant", "torch.compile通过TorchDynamo捕获计算图，配合Triton后端...")],
    [("user", "怎么调试CUDA out of memory？"), ("assistant", "建议先用torch.cuda.memory_summary()查看显存分配...")],
    [("user", "BF16和FP16有什么区别？"), ("assistant", "BF16有更大的指数范围但更低的尾数精度...")],
    [("user", "vLLM的PagedAttention原理是什么？"), ("assistant", "PagedAttention将KV Cache分成固定大小的Block...")],
    [("user", "LoRA和QLoRA有什么区别？"), ("assistant", "QLoRA在LoRA基础上引入了4-bit量化和双重量化...")],
    [("user", "Triton和CUDA相比有什么优势？"), ("assistant", "Triton提供更高层的抽象，自动处理tile/block调度...")],
    [("user", "DeepSeek-V2的MLA机制是什么？"), ("assistant", "MLA是Multi-head Latent Attention，通过低秩压缩KV...")],
    [("user", "GQA和MQA的对比？"), ("assistant", "GQA是MHA和MQA的折中，多个query head共享一组KV head...")],
    [("user", "什么是Ring Attention？"), ("assistant", "Ring Attention通过在设备间环形传递KV块来实现长序列并行...")],
    [("user", "SwiGLU激活函数的优势是什么？"), ("assistant", "SwiGLU结合了Swish和GLU，在LLM中表现优于ReLU/GELU...")],
    [("user", "Gradient Checkpointing的原理？"), ("assistant", "通过在前向传播时丢弃中间激活，反向时重新计算来节省显存...")],
    [("user", "NCCL和Gloo后端的区别？"), ("assistant", "NCCL专为NVIDIA GPU优化，Gloo更通用但GPU性能不如NCCL...")],
    [("user", "怎么理解RoPE的外推性？"), ("assistant", "RoPE的外推性与基频参数有关，NTK-aware和YaRN等方法改进了...")],
    [("user", "FlashAttention的IO复杂度？"), ("assistant", "FlashAttention将IO复杂度从O(N²)降低到O(N²/M)...")],
]


class LongHorizonChatTask:
    """
    长程对话记忆合成任务。

    生成长对话序列，在早期轮次植入关键事实/目标，
    然后插入大量干扰对话，最后提问以测试远程记忆能力。
    """

    def __init__(
        self,
        num_samples: int = 30,
        min_filler_turns: int = 15,
        max_filler_turns: int = 40,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.min_filler_turns = min_filler_turns
        self.max_filler_turns = max_filler_turns
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 数据生成
    # ------------------------------------------------------------------

    def generate_samples(self) -> list[LongHorizonSample]:
        """生成全部测试样本。"""
        samples: list[LongHorizonSample] = []
        types = ["entity_recall", "goal_tracking", "long_range_access"]
        per_type = self.num_samples // len(types)

        for task_type in types:
            for _ in range(per_type):
                samples.append(self._generate_one(task_type))

        while len(samples) < self.num_samples:
            t = self.rng.choice(types)
            samples.append(self._generate_one(t))

        self.rng.shuffle(samples)
        return samples

    def _generate_one(self, task_type: str) -> LongHorizonSample:
        """生成单条样本。"""
        if task_type == "entity_recall":
            return self._gen_entity_recall()
        elif task_type == "goal_tracking":
            return self._gen_goal_tracking()
        elif task_type == "long_range_access":
            return self._gen_long_range_access()
        else:
            raise ValueError(f"未知任务类型: {task_type}")

    def _gen_entity_recall(self) -> LongHorizonSample:
        """
        实体回忆测试：在对话早期提及某个实体和事实，
        插入大量干扰后，在末尾询问该事实。
        """
        fact_tmpl = self.rng.choice(_ENTITY_FACTS)
        num_fillers = self.rng.randint(self.min_filler_turns, self.max_filler_turns)

        conv: list[tuple[str, str]] = []

        # 第1轮：引入实体事实
        conv.append(("user", f"顺便说一下，{fact_tmpl['entity']}{fact_tmpl['fact']}。"))
        conv.append(("assistant", f"好的，我记住了{fact_tmpl['entity']}的信息。"))

        # 插入大量干扰
        conv.extend(self._filler_turns(num_fillers))

        total_user_turns = 1 + num_fillers  # 事实轮 + 干扰轮

        return LongHorizonSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            query=fact_tmpl["query"],
            expected_answer=fact_tmpl["answer"],
            fact_turn_index=0,
            total_turns=total_user_turns,
            task_type="entity_recall",
            metadata={"entity": fact_tmpl["entity"], "distance": num_fillers},
        )

    def _gen_goal_tracking(self) -> LongHorizonSample:
        """
        目标追踪测试：在对话开头设定目标，过程中讨论子目标，
        穿插干扰后，在末尾询问最初的主目标。
        """
        goal_tmpl = self.rng.choice(_GOAL_TEMPLATES)
        num_fillers_per_gap = self.rng.randint(3, 8)

        conv: list[tuple[str, str]] = []

        # 第1轮：设定主目标
        conv.append(("user", f"我们的主要目标是：{goal_tmpl['initial_goal']}。"))
        conv.append(("assistant", f"明白了，主要目标是{goal_tmpl['initial_goal']}。让我们开始吧。"))

        # 逐步讨论子目标，中间插入干扰
        for sub_goal in goal_tmpl["sub_goals"]:
            conv.extend(self._filler_turns(num_fillers_per_gap))
            conv.append(("user", f"下一步我们{sub_goal}。"))
            conv.append(("assistant", f"好的，开始{sub_goal}。"))

        # 最后再加一批干扰
        conv.extend(self._filler_turns(num_fillers_per_gap * 2))

        total_user_turns = 1 + len(goal_tmpl["sub_goals"]) + num_fillers_per_gap * (len(goal_tmpl["sub_goals"]) + 2)

        return LongHorizonSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            query=goal_tmpl["query"],
            expected_answer=goal_tmpl["answer"],
            fact_turn_index=0,
            total_turns=total_user_turns,
            task_type="goal_tracking",
            metadata={"sub_goals": goal_tmpl["sub_goals"]},
        )

    def _gen_long_range_access(self) -> LongHorizonSample:
        """
        远程记忆访问测试：在对话中间某个随机位置植入事实，
        然后继续大量对话后提问。
        测试记忆系统在非对话开头位置植入信息的回忆能力。
        """
        fact_tmpl = self.rng.choice(_ENTITY_FACTS)
        total_fillers = self.rng.randint(self.min_filler_turns, self.max_filler_turns)

        # 事实植入位置：在前1/3到1/2之间
        fact_position = self.rng.randint(total_fillers // 4, total_fillers // 2)

        conv: list[tuple[str, str]] = []

        # 前半段干扰
        conv.extend(self._filler_turns(fact_position))

        # 植入事实
        conv.append(("user", f"对了，{fact_tmpl['entity']}{fact_tmpl['fact']}，帮我记一下。"))
        conv.append(("assistant", f"好的，已记录。"))

        # 后半段干扰
        remaining = total_fillers - fact_position
        conv.extend(self._filler_turns(remaining))

        return LongHorizonSample(
            sample_id=uuid.uuid4().hex[:12],
            conversation=conv,
            query=fact_tmpl["query"],
            expected_answer=fact_tmpl["answer"],
            fact_turn_index=fact_position,
            total_turns=total_fillers + 1,
            task_type="long_range_access",
            metadata={
                "entity": fact_tmpl["entity"],
                "fact_position_ratio": fact_position / max(total_fillers, 1),
                "total_fillers": total_fillers,
            },
        )

    # ------------------------------------------------------------------
    # 干扰生成
    # ------------------------------------------------------------------

    def _filler_turns(self, n: int) -> list[tuple[str, str]]:
        """生成 n 轮干扰对话，从模板中随机选取（允许重复但尽量避免连续重复）。"""
        turns: list[tuple[str, str]] = []
        last_idx = -1
        for _ in range(n):
            idx = self.rng.randrange(len(_FILLER_EXCHANGES))
            # 尽量避免连续重复
            if idx == last_idx and len(_FILLER_EXCHANGES) > 1:
                idx = (idx + 1) % len(_FILLER_EXCHANGES)
            last_idx = idx
            turns.extend(_FILLER_EXCHANGES[idx])
        return turns

    # ------------------------------------------------------------------
    # 评估接口
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        sample: LongHorizonSample,
        predicted: str,
    ) -> dict[str, Any]:
        """
        评估单条长程对话样本。

        Args:
            sample: 测试样本
            predicted: 系统给出的回答

        Returns:
            包含 exact_match, distance, task_type 等信息的字典
        """
        expected = sample.expected_answer
        is_correct = expected.lower() in predicted.lower()

        return {
            "sample_id": sample.sample_id,
            "task_type": sample.task_type,
            "exact_match": is_correct,
            "expected": expected,
            "predicted": predicted,
            "fact_turn_index": sample.fact_turn_index,
            "total_turns": sample.total_turns,
            "distance": sample.total_turns - sample.fact_turn_index - 1,
        }

    def evaluate_batch(
        self,
        samples: list[LongHorizonSample],
        predictions: list[str],
    ) -> dict[str, Any]:
        """批量评估长程对话任务。"""
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
            t = r["task_type"]
            if t not in type_stats:
                type_stats[t] = {"correct": 0, "total": 0}
            type_stats[t]["total"] += 1
            if r["exact_match"]:
                type_stats[t]["correct"] += 1

        type_accuracy = {
            t: s["correct"] / max(s["total"], 1)
            for t, s in type_stats.items()
        }

        # 按距离分桶统计准确率
        distance_buckets: dict[str, dict[str, int]] = {}
        for r in results:
            d = r["distance"]
            if d < 10:
                bucket = "short(<10)"
            elif d < 20:
                bucket = "medium(10-20)"
            elif d < 30:
                bucket = "long(20-30)"
            else:
                bucket = "very_long(30+)"

            if bucket not in distance_buckets:
                distance_buckets[bucket] = {"correct": 0, "total": 0}
            distance_buckets[bucket]["total"] += 1
            if r["exact_match"]:
                distance_buckets[bucket]["correct"] += 1

        distance_accuracy = {
            b: s["correct"] / max(s["total"], 1)
            for b, s in distance_buckets.items()
        }

        return {
            "overall_accuracy": total_correct / max(total, 1),
            "total_samples": total,
            "total_correct": total_correct,
            "type_accuracy": type_accuracy,
            "type_stats": type_stats,
            "distance_accuracy": distance_accuracy,
            "distance_stats": distance_buckets,
            "details": results,
        }
