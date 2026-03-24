#!/usr/bin/env python3
"""
使用 GLM API 生成长对话训练数据，用于 MAG + SWA 预训练。

核心设计:
  1. 生成多 session 长对话 (8K~32K tokens)
  2. 每个对话在后期 session 显式引用前期 session 的关键信息
  3. 自动转换为 MAG 训练格式 (input_text + memory_texts + relevant_indices + target_text)
  4. 确保训练数据的长度 >> SWA window_size，迫使 MAG 必须依赖外部记忆

用法:
    # 基本用法：生成 1000 条对话（顺序请求，自动限流控制）
    python scripts/generate_long_dialogue.py \
        --api_key YOUR_GLM_API_KEY \
        --output data/raw/long_dialogue_train.jsonl \
        --num_dialogues 1000 \
        --num_sessions_per_dialogue 8 \
        --turns_per_session 15

    # 快速测试
    python scripts/generate_long_dialogue.py \
        --api_key YOUR_GLM_API_KEY \
        --output data/raw/long_dialogue_test.jsonl \
        --num_dialogues 5 \
        --num_sessions_per_dialogue 4 \
        --turns_per_session 8

    # 从断点续传
    python scripts/generate_long_dialogue.py \
        --api_key YOUR_GLM_API_KEY \
        --output data/raw/long_dialogue_train.jsonl \
        --num_dialogues 1000 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("generate_long_dialogue")

# ======================================================================
# GLM API 客户端
# ======================================================================

class GLMClient:
    """GLM API 客户端，单线程顺序请求 + 定时间隔限流控制。
    
    策略：
    - 每次请求之间保持固定间隔（默认 30s）
    - 如果请求成功，立刻发下一条（不等间隔）
    - 如果被限流(429)，等到下一个间隔周期再重试
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "glm-4.7",
        max_retries: int = 10,
        timeout: int = 120,
        request_interval: float = 30.0,
        **kwargs,  # 兼容旧参数
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_interval = request_interval  # 限流时的请求间隔（秒）
        self._last_request_time = 0.0  # 上次请求的时间戳
        self._rate_limited = False  # 是否处于限流状态
        self._lock = threading.Lock()  # 多线程保护限流状态
        # 统计
        self.total_requests = 0
        self.total_success = 0
        self.total_retries = 0
        self.total_tokens_est = 0

    def _wait_for_interval(self):
        """如果处于限流状态，等待到下一个间隔周期再发请求。"""
        with self._lock:
            if not self._rate_limited:
                return  # 没被限流，立刻发
            elapsed = time.time() - self._last_request_time
            remaining = self.request_interval - elapsed
        
        if remaining > 0:
            logger.info(f"⏸️ 限流等待中，{remaining:.1f}s 后重试...")
            time.sleep(remaining)

    def _inc_stat(self, field: str, value: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + value)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.9,
        max_tokens: int = 16384,
        top_p: float = 0.9,
        log_prefix: str = "",
    ) -> str:
        """调用 GLM chat API，带详细日志。"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        # 估算输入长度
        input_chars = sum(len(m.get("content", "")) for m in messages)
        prefix = f"[{log_prefix}] " if log_prefix else ""

        for attempt in range(self.max_retries):
            # 如果被限流，等到下一个间隔再发
            self._wait_for_interval()

            self._inc_stat('total_requests')
            with self._lock:
                self._last_request_time = time.time()
            req_start = self._last_request_time
            logger.info(
                f"{prefix}发送 API 请求 (第{attempt+1}次, "
                f"输入~{input_chars}字, max_tokens={max_tokens}, "
                f"model={self.model}, 累计请求#{self.total_requests})"
            )

            try:
                resp = requests.post(
                    url, headers=headers, json=payload,
                    timeout=(10, self.timeout),  # (connect_timeout, read_timeout)
                )
                req_duration = time.time() - req_start

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                # 提取 usage 信息（如果有）
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "?")
                completion_tokens = usage.get("completion_tokens", "?")
                total_tokens = usage.get("total_tokens", "?")

                self._inc_stat('total_success')
                self._inc_stat('total_tokens_est', len(content) // 2)

                # 成功了！解除限流状态，下次立刻发
                with self._lock:
                    self._rate_limited = False
                logger.info(
                    f"{prefix}✅ 请求成功 ({req_duration:.1f}s), "
                    f"响应{len(content)}字, "
                    f"tokens: prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}, "
                    f"累计成功: {self.total_success}/{self.total_requests}"
                )
                return content

            except requests.exceptions.HTTPError as e:
                req_duration = time.time() - req_start

                if resp.status_code == 429:
                    self._inc_stat('total_retries')
                    # 进入限流状态，下次请求会等 request_interval 秒
                    with self._lock:
                        self._rate_limited = True
                    logger.warning(
                        f"{prefix}⚠️ API 限流 (429), 耗时 {req_duration:.1f}s, "
                        f"将在 {self.request_interval:.0f}s 后重试 "
                        f"({attempt+1}/{self.max_retries}), "
                        f"响应: {resp.text[:300]}"
                    )
                    # 不在这里 sleep，让 _wait_for_interval 在下一轮处理
                elif resp.status_code >= 500:
                    self._inc_stat('total_retries')
                    wait = min(30, 5 * (2 ** attempt))
                    logger.warning(
                        f"{prefix}⚠️ 服务端错误 {resp.status_code}, 耗时 {req_duration:.1f}s, "
                        f"等待 {wait:.0f}s 后重试 ({attempt+1}/{self.max_retries}), "
                        f"响应: {resp.text[:300]}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"{prefix}❌ 请求失败: HTTP {resp.status_code}, "
                        f"耗时 {req_duration:.1f}s, "
                        f"响应: {resp.text[:500]}"
                    )
                    raise

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
                req_duration = time.time() - req_start
                self._inc_stat('total_retries')
                wait = min(30, 5 * (2 ** attempt))
                timeout_type = "连接超时" if isinstance(e, requests.exceptions.ConnectTimeout) else "读取超时"
                logger.warning(
                    f"{prefix}⚠️ {timeout_type} ({req_duration:.1f}s), "
                    f"等待 {wait:.0f}s 后重试 ({attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)

            except requests.exceptions.RequestException as e:
                self._inc_stat('total_retries')
                wait = min(30, 5 * (2 ** attempt))
                logger.warning(
                    f"{prefix}⚠️ 网络异常: {e}, "
                    f"等待 {wait:.0f}s 后重试 ({attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"GLM API 调用失败，已重试 {self.max_retries} 次。"
            f"请检查: 1) API Key 是否有效 2) 模型 {self.model} 是否可用 "
            f"3) 账户配额是否充足"
        )

    def print_stats(self):
        """打印累计统计。"""
        logger.info(
            f"📊 API 统计: 总请求={self.total_requests}, "
            f"成功={self.total_success}, "
            f"重试={self.total_retries}, "
            f"估算tokens=~{self.total_tokens_est}"
        )


# ======================================================================
# 对话生成模板
# ======================================================================

def _generate_scenario(
    client: GLMClient,
    dialogue_idx: int,
    num_sessions: int,
    turns_per_session: int,
    rng: random.Random,
) -> dict | None:
    """让 GLM 动态生成一个对话场景（角色、人设、话题）。
    
    不硬编码场景，让模型自由发挥，保证多样性。
    """
    dial_id = f"dial_{dialogue_idx:05d}"
    min_topics = max(12, num_sessions * 2)
    
    prompt = f"""你是一个对话数据集设计专家。请设计一个适合生成多轮长对话的场景。

## 需求
我需要生成一段很长的对话数据，包含 {num_sessions} 次会话（模拟不同日期的多次交流），每次会话有 {turns_per_session} 轮对话。

## 要求
1. 设计一个两人之间的对话场景（关系可以是朋友、同事、师生、家人、医患、商业伙伴、邻居、旅伴、创业伙伴、读书会成员、运动搭子等任何合理关系）
2. 两个角色要有鲜明的个性、不同的背景和说话风格，人设描述要具体（包含年龄、职业、性格特点、爱好等）
3. 场景要有足够的话题深度，能支撑 {num_sessions} 次长对话而不重复
4. 话题要丰富多样（至少 {min_topics} 个），涵盖这段关系中可能讨论的方方面面
5. 角色名请用中文名字（不要用"小明小红"这种太常见的名字，要有个性）
6. 每次生成请发挥创意，不要总是重复相似的场景

## 输出格式
请严格以 JSON 格式输出，不要添加其他内容：
```json
{{
  "scenario": "场景的简要描述",
  "person_a": "角色A的名字",
  "person_b": "角色B的名字",
  "persona_a": "角色A的详细人设（年龄、职业、性格、爱好等）",
  "persona_b": "角色B的详细人设（年龄、职业、性格、爱好等）",
  "topics": ["话题1", "话题2", "话题3", ...]
}}
```"""

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,  # 高温度保证多样性
            max_tokens=4096,
            top_p=0.95,
            log_prefix=f"{dial_id}/场景生成",
        )
        
        # 解析 JSON
        scenario = None
        # 方法1: 直接解析
        try:
            scenario = json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 方法2: 从 ```json ... ``` 中提取
        if scenario is None:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
            if match:
                try:
                    scenario = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # 方法3: 找 { ... } 块
        if scenario is None:
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                try:
                    scenario = json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        
        if scenario is None:
            logger.warning(f"  ⚠️ [{dial_id}] 场景生成 JSON 解析失败, 响应: {response[:300]}")
            return None
        
        # 校验必须字段
        required_fields = ["scenario", "person_a", "person_b", "persona_a", "persona_b", "topics"]
        for field in required_fields:
            if field not in scenario:
                logger.warning(f"  ⚠️ [{dial_id}] 场景缺少字段 {field}")
                return None
        
        if not isinstance(scenario["topics"], list) or len(scenario["topics"]) < 4:
            logger.warning(f"  ⚠️ [{dial_id}] 话题数量不足: {len(scenario.get('topics', []))}")
            return None
        
        logger.info(
            f"  ✅ [{dial_id}] 场景生成成功: \"{scenario['scenario']}\", "
            f"{scenario['person_a']} vs {scenario['person_b']}, "
            f"{len(scenario['topics'])} 个话题"
        )
        return scenario
        
    except Exception as e:
        logger.error(f"  ❌ [{dial_id}] 场景生成失败: {e}")
        return None


def _build_session_prompt(
    scenario: dict,
    session_idx: int,
    total_sessions: int,
    turns_per_session: int,
    previous_key_facts: list[str],
    session_topics: list[str],
) -> str:
    """构建单个 session 的生成 prompt。

    关键设计:
    - 明确要求在后续 session 中引用/回溯前期信息
    - 要求每轮发言有实质内容 (至少 2-3 句话)
    - 输出格式为可解析的 JSON
    """
    person_a = scenario["person_a"]
    person_b = scenario["person_b"]
    persona_a = scenario["persona_a"]
    persona_b = scenario["persona_b"]

    # 基础 prompt
    prompt = f"""你是一个对话数据生成专家。请生成一段自然、丰富的对话。

## 场景设定
{scenario['scenario']}

## 角色
- {person_a}: {persona_a}
- {person_b}: {persona_b}

## 当前会话
这是第 {session_idx + 1}/{total_sessions} 次对话。
本次对话围绕以下话题展开: {', '.join(session_topics)}

## 对话要求
1. 生成 {turns_per_session} 轮对话 (一轮 = {person_a} 说一句 + {person_b} 回一句)
2. 每次发言至少包含 2-3 句话，包含具体的细节、数字、人名、地点等
3. 对话要自然流畅，像真实的人在交流
4. 每轮对话要有新的信息量，不要重复或空泛
5. 可以有情绪表达、玩笑、口语化表达
"""

    # 如果有之前的关键信息，要求引用
    if previous_key_facts:
        prompt += f"""
## 重要：回溯之前的信息
在之前的对话中，提到了以下重要事实。请在本次对话中**自然地引用、追问或回溯**其中的一些内容（至少引用 2-3 条），例如"你之前说...", "上次你提到...", "你那个...后来怎么样了?"等：

"""
        for i, fact in enumerate(previous_key_facts):
            prompt += f"  {i+1}. {fact}\n"

    prompt += f"""
## 输出格式
请以 JSON 数组格式输出，每个元素是一个对话轮次:
```json
[
  {{"speaker": "{person_a}", "text": "发言内容..."}},
  {{"speaker": "{person_b}", "text": "回复内容..."}},
  ...
]
```

请确保:
- JSON 格式正确可解析
- 每条发言文本足够丰富 (至少 30 个字)
- 不要在 JSON 之外添加其他内容
"""
    return prompt


def _build_key_facts_prompt(
    scenario: dict,
    session_turns: list[dict],
    session_idx: int,
) -> str:
    """从一个 session 的对话中提取关键事实。"""
    person_a = scenario["person_a"]
    person_b = scenario["person_b"]

    dialogue_text = ""
    for t in session_turns:
        dialogue_text += f"{t['speaker']}: {t['text']}\n"

    return f"""从以下第 {session_idx + 1} 次对话中，提取 5-8 条关键事实信息。
这些事实应该是具体的、有价值的，在后续对话中可能被引用。

对话内容:
{dialogue_text}

请以 JSON 数组格式输出关键事实:
```json
["事实1", "事实2", "事实3", ...]
```

要求:
- 每条事实要具体，包含人名/数字/时间/地点等
- 不要太泛泛（如"他们聊了工作"），要具体（如"{person_a}提到下周二要去杭州出差"）
- 只输出 JSON，不要其他内容
"""


# ======================================================================
# 核心数据生成流程
# ======================================================================

def generate_single_dialogue(
    client: GLMClient,
    dialogue_idx: int,
    num_sessions: int,
    turns_per_session: int,
    seed: int = 42,
) -> dict | None:
    """生成一条完整的多 session 长对话。

    返回:
        {
            "dialogue_id": "dial_0001",
            "scenario": {...},
            "sessions": [
                {"session_idx": 0, "turns": [...], "key_facts": [...]},
                ...
            ],
            "all_key_facts": [...],  # 所有 session 的 key_facts 汇总
        }
    """
    dialogue_id = f"dial_{dialogue_idx:05d}"
    rng = random.Random(seed + dialogue_idx)

    # 让 GLM 动态生成场景 (角色 + 人设 + 话题)，最多重试 3 次
    scenario = None
    for scenario_attempt in range(3):
        scenario = _generate_scenario(
            client=client,
            dialogue_idx=dialogue_idx,
            num_sessions=num_sessions,
            turns_per_session=turns_per_session,
            rng=rng,
        )
        if scenario is not None:
            break
        if scenario_attempt < 2:
            logger.warning(
                f"⚠️ 对话 {dialogue_id} 场景生成失败 "
                f"(第{scenario_attempt+1}/3次), 5s 后重试..."
            )
            time.sleep(5)
        else:
            logger.warning(f"❌ 对话 {dialogue_id} 场景生成 3 次全部失败, 跳过")
            return None

    logger.info(
        f"🎬 开始生成对话 {dialogue_id}: "
        f"场景=\"{scenario['scenario']}\", "
        f"{scenario['person_a']} vs {scenario['person_b']}, "
        f"{num_sessions} sessions × {turns_per_session} 轮"
    )

    # 为每个 session 分配话题
    all_topics = list(scenario["topics"])
    sessions_data = []
    all_key_facts = []
    dial_start = time.time()

    for si in range(num_sessions):
        session_start = time.time()
        log_prefix = f"{dialogue_id}/S{si+1}"

        # 每个 session 选 2-3 个话题
        n_topics = rng.randint(2, min(3, len(all_topics)))
        session_topics = rng.sample(all_topics, n_topics)

        logger.info(
            f"  📝 [{log_prefix}] 开始生成 session {si+1}/{num_sessions}, "
            f"话题: {session_topics}"
        )

        # 构建 prompt
        # 前几个 session 不要求回溯，后面的 session 逐步增加回溯
        facts_to_reference = []
        if si >= 1 and all_key_facts:
            # 从之前所有 key_facts 中选取一些
            n_ref = min(rng.randint(2, 5), len(all_key_facts))
            facts_to_reference = rng.sample(all_key_facts, n_ref)
            logger.info(f"  📝 [{log_prefix}] 引用 {len(facts_to_reference)} 条历史事实")

        prompt = _build_session_prompt(
            scenario=scenario,
            session_idx=si,
            total_sessions=num_sessions,
            turns_per_session=turns_per_session,
            previous_key_facts=facts_to_reference,
            session_topics=session_topics,
        )

        # 调用 GLM 生成对话（最多重试 2 次解析失败）
        turns = None
        for parse_attempt in range(2):
            try:
                response = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=16384,
                    top_p=0.9,
                    log_prefix=f"{log_prefix}/对话生成" + (f"(重试{parse_attempt+1})" if parse_attempt > 0 else ""),
                )
                turns = _parse_turns_json(response)
                if turns and len(turns) >= 4:
                    logger.info(
                        f"  ✅ [{log_prefix}] 对话生成成功: {len(turns)} 轮, "
                        f"总字数: {sum(len(t['text']) for t in turns)}"
                    )
                    break
                else:
                    logger.warning(
                        f"  ⚠️ [{log_prefix}] 解析得到 {len(turns) if turns else 0} 轮 "
                        f"(需要至少 4 轮), {'重试中...' if parse_attempt == 0 else '放弃'}"
                    )
                    turns = None
            except Exception as e:
                logger.error(f"  ❌ [{log_prefix}] 对话生成失败: {e}")
                if parse_attempt == 0:
                    logger.info(f"  ⏳ 等待 5s 后重试...")
                    time.sleep(5)
                turns = None

        if not turns or len(turns) < 4:
            logger.warning(f"  ❌ [{log_prefix}] 对话生成最终失败, 跳过整条对话")
            return None

        # 提取 key facts
        key_facts = []
        try:
            facts_prompt = _build_key_facts_prompt(scenario, turns, si)
            facts_response = client.chat(
                messages=[{"role": "user", "content": facts_prompt}],
                temperature=0.3,
                max_tokens=1024,
                log_prefix=f"{log_prefix}/事实提取",
            )
            key_facts = _parse_json_array(facts_response)
            if not key_facts:
                # 如果提取失败，手动从对话中截取
                key_facts = [t["text"][:80] for t in turns[:5] if len(t["text"]) > 20]
                logger.info(f"  ℹ️ [{log_prefix}] 事实提取为空, 从对话文本中截取了 {len(key_facts)} 条")
            else:
                logger.info(f"  ✅ [{log_prefix}] 提取到 {len(key_facts)} 条关键事实")
        except Exception as e:
            logger.warning(f"  ⚠️ [{log_prefix}] key facts 提取失败: {e}")
            key_facts = [t["text"][:80] for t in turns[:5] if len(t["text"]) > 20]

        sessions_data.append({
            "session_idx": si,
            "topics": session_topics,
            "turns": turns,
            "key_facts": key_facts,
            "referenced_facts": facts_to_reference,
        })
        all_key_facts.extend(key_facts)

        session_time = time.time() - session_start
        logger.info(
            f"  ✅ [{log_prefix}] Session 完成 ({session_time:.1f}s): "
            f"{len(turns)} 轮对话, {len(key_facts)} 条事实, "
            f"累计事实: {len(all_key_facts)}"
        )

    dial_time = time.time() - dial_start
    total_turns = sum(len(s["turns"]) for s in sessions_data)
    total_chars = sum(
        len(t["text"]) for s in sessions_data for t in s["turns"]
    )
    logger.info(
        f"🎬 对话 {dialogue_id} 全部完成 ({dial_time:.1f}s): "
        f"{len(sessions_data)} sessions, {total_turns} 轮, "
        f"{total_chars} 字, {len(all_key_facts)} 条事实"
    )

    return {
        "dialogue_id": dialogue_id,
        "scenario": {
            "description": scenario["scenario"],
            "person_a": scenario["person_a"],
            "person_b": scenario["person_b"],
            "persona_a": scenario["persona_a"],
            "persona_b": scenario["persona_b"],
        },
        "sessions": sessions_data,
        "all_key_facts": all_key_facts,
    }


def _try_fix_truncated_json(json_str: str) -> str | None:
    """尝试修复被截断的 JSON 数组。
    
    GLM 在 max_tokens 不足时会截断输出，导致 JSON 不完整。
    策略：找到最后一个完整的 },  然后补上 ]
    """
    json_str = json_str.strip()
    if not json_str.startswith('['):
        return None
    
    # 从后往前找最后一个完整的 } (后面可能跟 , 或空白)
    # 逐步截断尝试解析
    for i in range(len(json_str) - 1, 0, -1):
        if json_str[i] == '}':
            candidate = json_str[:i+1] + ']'
            try:
                data = json.loads(candidate)
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"  🔧 修复截断 JSON 成功: 恢复了 {len(data)} 个元素")
                    return candidate
            except json.JSONDecodeError:
                continue
    return None


def _parse_turns_json(response: str) -> list[dict]:
    """从 GLM 响应中解析对话轮次 JSON。
    
    支持：
    1. 直接 JSON 数组
    2. ```json ... ``` 包裹
    3. 文本中的 [ ... ] 块
    4. 被截断的不完整 JSON（自动修复）
    """
    def _filter_turns(data):
        if isinstance(data, list):
            return [t for t in data if isinstance(t, dict) and "speaker" in t and "text" in t]
        return []

    # 方法 1：尝试直接解析
    try:
        data = json.loads(response)
        turns = _filter_turns(data)
        if turns:
            logger.debug(f"  解析方式: 直接 JSON, {len(turns)} 轮")
            return turns
    except json.JSONDecodeError:
        pass

    # 方法 2：尝试从 ```json ... ``` 中提取
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            turns = _filter_turns(data)
            if turns:
                logger.debug(f"  解析方式: ```json``` 块, {len(turns)} 轮")
                return turns
        except json.JSONDecodeError:
            pass

    # 方法 3：尝试找到 [ ... ] 块
    match = re.search(r"\[[\s\S]*\]", response)
    if match:
        try:
            data = json.loads(match.group())
            turns = _filter_turns(data)
            if turns:
                logger.debug(f"  解析方式: [..] 提取, {len(turns)} 轮")
                return turns
        except json.JSONDecodeError:
            pass

    # 方法 4：尝试修复被截断的 JSON（max_tokens 不足导致输出截断）
    # 先提取 ```json 块中的内容（可能没有闭合的 ```）
    json_str = None
    match_open = re.search(r"```(?:json)?\s*\n?(\[.*)", response, re.DOTALL)
    if match_open:
        json_str = match_open.group(1).rstrip('`').strip()
    else:
        # 直接找 [ 开头的内容
        match_bracket = re.search(r"(\[.*)", response, re.DOTALL)
        if match_bracket:
            json_str = match_bracket.group(1).strip()

    if json_str:
        fixed = _try_fix_truncated_json(json_str)
        if fixed:
            try:
                data = json.loads(fixed)
                turns = _filter_turns(data)
                if turns:
                    logger.warning(
                        f"  ⚠️ JSON 被截断, 通过修复恢复了 {len(turns)} 轮 "
                        f"(原始响应{len(response)}字, 可能需要增大 max_tokens)"
                    )
                    return turns
            except json.JSONDecodeError:
                pass

    logger.warning(
        f"无法解析对话 JSON (响应{len(response)}字), "
        f"前200字: {response[:200]}\n"
        f"后200字: ...{response[-200:]}"
    )
    return []


def _parse_json_array(response: str) -> list[str]:
    """从 GLM 响应中解析 JSON 数组。"""
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return [str(item) for item in data if item]
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return [str(item) for item in data if item]
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[[\s\S]*\]", response)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(item) for item in data if item]
        except json.JSONDecodeError:
            pass

    return []


# ======================================================================
# 转换为 MAG 训练格式
# ======================================================================

def dialogue_to_mag_samples(
    dialogue: dict,
    num_memories: int = 15,
    num_hard_negatives: int = 3,
    rng: random.Random | None = None,
) -> list[dict]:
    """将一条长对话转换为多个 MAG 训练样本。

    策略:
    - 对于 session N (N >= 2) 中的每个 turn pair:
      - query = 当前 user 发言
      - target = 对应的 assistant 回复
      - 相关记忆 = 之前 session 的 key_facts + 该 session 引用的事实
      - 硬负例 = 同 session 的其他发言
      - 随机负例 = 其他 session 的随机发言
    """
    if rng is None:
        rng = random.Random(42)

    sessions = dialogue["sessions"]
    if len(sessions) < 2:
        return []

    # 收集全局 utterance 池
    all_utterances = []
    for s in sessions:
        for t in s["turns"]:
            if len(t["text"]) >= 15:
                all_utterances.append(t["text"])

    samples = []
    person_a = dialogue["scenario"]["person_a"]

    for si in range(1, len(sessions)):
        current = sessions[si]
        turns = current["turns"]
        if len(turns) < 4:
            continue

        # 收集之前所有 session 的 key_facts 作为记忆池
        history_facts = []
        history_utterances = []
        for prev_si in range(si):
            history_facts.extend(sessions[prev_si].get("key_facts", []))
            for t in sessions[prev_si]["turns"]:
                if len(t["text"]) >= 15:
                    history_utterances.append(t["text"])

        if not history_facts and not history_utterances:
            continue

        # 同 session 内的其他发言 (硬负例)
        current_utterances = [t["text"] for t in turns if len(t["text"]) >= 15]

        # 遍历 turn pairs
        i = 0
        while i < len(turns) - 1:
            t0, t1 = turns[i], turns[i + 1]
            if t0["speaker"] == t1["speaker"]:
                i += 1
                continue

            query = t0["text"]
            target = t1["text"]
            if len(query) < 10 or len(target) < 10:
                i += 2
                continue

            # --- 相关记忆 ---
            # 优先: key_facts (精炼), 补充: 历史 utterances (原始)
            relevant_memories = []

            # 1. key_facts
            max_facts = min(len(history_facts), num_memories // 3)
            if max_facts > 0:
                relevant_memories.extend(rng.sample(history_facts, max_facts))

            # 2. 历史 utterances (补充)
            remaining_relevant = num_memories // 2 - len(relevant_memories)
            if remaining_relevant > 0 and history_utterances:
                n_hist = min(remaining_relevant, len(history_utterances))
                # 最近的优先
                recent_half = max(1, n_hist // 2)
                recent = history_utterances[-recent_half:]
                older = rng.sample(
                    history_utterances[:-recent_half],
                    min(n_hist - recent_half, len(history_utterances[:-recent_half])),
                ) if len(history_utterances) > recent_half else []
                relevant_memories.extend(recent + older)

            if not relevant_memories:
                i += 2
                continue

            # --- 硬负例 ---
            hard_neg_candidates = [
                u for u in current_utterances if u != query and u != target
            ]
            hard_negs = rng.sample(
                hard_neg_candidates,
                min(num_hard_negatives, len(hard_neg_candidates)),
            ) if hard_neg_candidates else []

            # --- 随机负例 ---
            remaining = num_memories - len(relevant_memories) - len(hard_negs)
            random_negs = []
            if remaining > 0:
                avoid = set(relevant_memories + hard_negs + [query, target])
                candidates = [u for u in all_utterances if u not in avoid]
                if candidates:
                    random_negs = rng.sample(candidates, min(remaining, len(candidates)))

            # 组装并打乱
            all_memories = relevant_memories + hard_negs + random_negs
            relevant_indices = list(range(len(relevant_memories)))

            indices = list(range(len(all_memories)))
            rng.shuffle(indices)
            shuffled_memories = [all_memories[idx] for idx in indices]
            shuffled_relevant = sorted([indices.index(r) for r in relevant_indices])

            samples.append({
                "input_text": query,
                "target_text": target,
                "memory_texts": shuffled_memories,
                "relevant_indices": shuffled_relevant,
                "dialogue_id": dialogue["dialogue_id"],
                "session_idx": si,
            })

            i += 2

    return samples


# ======================================================================
# 主流程
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 GLM API 生成长对话训练数据 (MAG + SWA 预训练)"
    )

    # API 配置
    parser.add_argument("--api_key", type=str, default="",
                        help="GLM API Key (也可通过 GLM_API_KEY 环境变量设置)")
    parser.add_argument("--api_base_url", type=str, default="https://open.bigmodel.cn/api/paas/v4",
                        help="GLM API base URL")
    parser.add_argument("--model", type=str, default="glm-4.7",
                        help="GLM 模型名称 (默认 glm-4.7, 可选 glm-5, glm-4-flash)")

    # 输出
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 文件路径")
    parser.add_argument("--raw_output", type=str, default="",
                        help="原始对话 JSON 输出路径 (可选, 用于调试)")

    # 数据量配置
    parser.add_argument("--num_dialogues", type=int, default=100,
                        help="生成对话数量")
    parser.add_argument("--num_sessions_per_dialogue", type=int, default=6,
                        help="每条对话的 session 数 (越多越长, 建议 4~10)")
    parser.add_argument("--turns_per_session", type=int, default=12,
                        help="每个 session 的对话轮数 (建议 8~20)")
    parser.add_argument("--num_memories", type=int, default=15,
                        help="每个 MAG 样本的候选记忆数")
    parser.add_argument("--num_hard_negatives", type=int, default=3,
                        help="硬负例数量")

    # 请求控制
    parser.add_argument("--request_interval", type=float, default=30.0,
                        help="限流时请求间隔秒数 (默认 30s, 成功则立刻发下一条)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="API 请求超时秒数 (默认 120s)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="并行生成的对话数 (默认 2, 即同时跑 2 条对话)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传: 跳过已生成的对话")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info(f"🚀 GLM 长对话数据生成器启动 (并行模式, {args.num_workers} workers)")
    logger.info(f"  模型: {args.model}")
    logger.info(f"  目标: {args.num_dialogues} 条对话")
    logger.info(f"  每条: {args.num_sessions_per_dialogue} sessions × {args.turns_per_session} 轮")
    logger.info(f"  并行数: {args.num_workers} 条对话同时生成")
    logger.info(f"  请求间隔: {args.request_interval}s (限流时等待, 成功时立刻发)")
    logger.info(f"  超时: {args.timeout}s")
    logger.info(f"  输出: {args.output}")
    logger.info("=" * 60)

    # API key
    api_key = args.api_key or os.environ.get("GLM_API_KEY", "")
    if not api_key:
        logger.error("请通过 --api_key 参数或 GLM_API_KEY 环境变量提供 API Key")
        sys.exit(1)

    client = GLMClient(
        api_key=api_key,
        base_url=args.api_base_url,
        model=args.model,
        timeout=args.timeout,
        request_interval=args.request_interval,
    )

    # 断点续传: 检查已生成的对话
    existing_ids = set()
    existing_samples = []
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    existing_ids.add(sample.get("dialogue_id", ""))
                    existing_samples.append(sample)
        logger.info(f"📂 断点续传: 已有 {len(existing_samples)} 条样本, "
                    f"来自 {len(existing_ids)} 条对话")

    # 确定需要生成的对话
    dialogues_to_generate = []
    for i in range(args.num_dialogues):
        dial_id = f"dial_{i:05d}"
        if dial_id not in existing_ids:
            dialogues_to_generate.append(i)

    if not dialogues_to_generate:
        logger.info("✅ 所有对话已生成, 无需继续")
        return

    est_api_calls = len(dialogues_to_generate) * (args.num_sessions_per_dialogue * 2 + 1)
    logger.info(
        f"📋 需要生成 {len(dialogues_to_generate)} 条对话, "
        f"预计 ~{est_api_calls} 次 API 调用 ({args.num_workers} workers 并行)"
    )

    # ---- 多线程并行生成 + 即时写入 ----
    succeeded = 0
    failed = 0
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples_written = len(existing_samples)
    write_lock = threading.Lock()  # 文件写入锁
    progress_lock = threading.Lock()  # 进度统计锁

    def _generate_one(dial_idx: int) -> None:
        """单个对话的 worker 函数，在线程池中执行。"""
        nonlocal succeeded, failed, total_samples_written
        dial_id = f"dial_{dial_idx:05d}"
        try:
            dialogue = generate_single_dialogue(
                client=client,
                dialogue_idx=dial_idx,
                num_sessions=args.num_sessions_per_dialogue,
                turns_per_session=args.turns_per_session,
                seed=args.seed,
            )

            if dialogue is None:
                with progress_lock:
                    failed += 1
                logger.warning(f"⚠️ 对话 {dial_id} 生成失败, 跳过")
                return

            # 转换为 MAG 样本
            rng = random.Random(args.seed + dial_idx)
            samples = dialogue_to_mag_samples(
                dialogue,
                num_memories=args.num_memories,
                num_hard_negatives=args.num_hard_negatives,
                rng=rng,
            )

            # 线程安全地追加写入文件
            with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    for s in samples:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
            with progress_lock:
                succeeded += 1
                total_samples_written += len(samples)
                _succeeded = succeeded
                _failed = failed
                _total_samples = total_samples_written

            # 进度日志
            elapsed = time.time() - t0
            done = _succeeded + _failed
            total = len(dialogues_to_generate)
            avg_time = elapsed / done if done > 0 else 0
            eta = (total - done) * avg_time
            logger.info(
                f"📊 {dial_id} → {len(samples)} 个 MAG 样本 | "
                f"进度: {done}/{total} (成功{_succeeded} 失败{_failed}) | "
                f"累计样本: {_total_samples} | "
                f"速度: {avg_time:.1f}s/条 | "
                f"已用: {elapsed/60:.1f}min | "
                f"ETA: {eta/60:.1f}min"
            )

        except Exception as e:
            with progress_lock:
                failed += 1
            logger.error(f"❌ 对话 {dial_id} 生成异常: {e}", exc_info=True)

    logger.info(
        f"\n🚀 开始并行生成 "
        f"({args.num_workers} workers, 限流间隔: {args.request_interval}s)...\n"
    )

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(_generate_one, dial_idx)
                for dial_idx in dialogues_to_generate
            ]
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 触发异常传播
                except Exception as e:
                    logger.error(f"❌ Worker 异常: {e}", exc_info=True)

    except KeyboardInterrupt:
        logger.info("\n⚡ 用户中断! 已生成的数据已实时写入文件, 无需额外保存")
    total_time = time.time() - t0

    # 最终统计
    logger.info("")
    logger.info("=" * 60)
    logger.info("🏁 数据生成完成!")
    logger.info(f"  ✅ 成功: {succeeded} 条对话")
    logger.info(f"  ❌ 失败: {failed} 条对话")
    logger.info(f"  📄 总 MAG 样本: {total_samples_written}")
    logger.info(f"  ⏱️ 总耗时: {total_time/60:.1f} 分钟")
    logger.info(f"  📁 输出文件: {output_path}")
    client.print_stats()
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
