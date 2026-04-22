#!/usr/bin/env python3
"""
Hard Long-Context Memory Benchmarks.

NIH-Extended saturates at 100% for modern models, making it useless for
differentiating memory strategies. This script provides harder tasks where:

  1. Base models show measurable degradation with longer contexts
  2. Memory compression (L0→L1→L2, kNN, etc.) can demonstrate clear benefits
  3. Different memory strategies can be meaningfully compared

Benchmark types:
  - multi_needle_recall: Recall N needles scattered across context, answer ALL
  - associative_recall: Recall (key, value) pairs that appeared earlier
  - temporal_ordering: Determine order of events scattered across context
  - counterfactual_retrieval: Find contradictory facts and identify the latest
  - passkey_copy: Classic passkey task at extreme depths & multi-key variants
  - synthetic_reasoning_chain: Multi-hop reasoning over facts in long context

Usage:
    python scripts/eval_memory_hard_benchmarks.py \
        --model_path ../models/Qwen--Qwen3-8b/ \
        --benchmarks multi_needle_recall associative_recall \
        --context_lengths 16384 32768 65536 131072 \
        --output_dir outputs/hard_benchmarks/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hard_benchmarks")

# ======================================================================
# HAYSTACK TEXT POOL (diverse filler)
# ======================================================================

HAYSTACK_PASSAGES = [
    "The history of computing stretches back thousands of years. From the abacus to modern supercomputers, "
    "humanity has continually sought better ways to process information. Early mechanical calculators gave way "
    "to electronic computers in the mid-twentieth century, transforming every aspect of science and industry.",

    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. "
    "These systems improve their performance over time without being explicitly programmed. Deep learning, a subset "
    "using neural networks with many layers, has achieved remarkable results in image recognition and NLP.",

    "The theory of relativity revolutionized our understanding of space and time. Special relativity showed that "
    "the speed of light is constant for all observers, while general relativity described gravity as the curvature "
    "of spacetime caused by mass and energy.",

    "Natural language processing enables computers to understand and generate human language. Key tasks include "
    "translation, summarization, question answering, and sentiment analysis. Recent advances in transformer models "
    "have dramatically improved performance across all these benchmarks.",

    "The water cycle describes the continuous movement of water on, above, and below the Earth's surface. "
    "Water evaporates from oceans and lakes, forms clouds through condensation, and returns as precipitation. "
    "Approximately 97% of Earth's water is stored in the oceans.",

    "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement. Unlike classical "
    "bits, quantum bits can exist in multiple states simultaneously, potentially solving certain problems "
    "exponentially faster than classical computers.",

    "The Roman Empire was one of the largest civilizations in history. At its height, it stretched from Britain "
    "to Mesopotamia, encompassing the entire Mediterranean basin. Roman law, engineering, and language continue "
    "to influence modern societies.",

    "Photosynthesis converts sunlight into chemical energy. Using chlorophyll, plants absorb carbon dioxide and "
    "water to produce glucose and oxygen. This process is fundamental to life on Earth and forms the basis of "
    "most food chains.",

    "Database management systems organize and store data efficiently. Relational databases use structured tables "
    "with defined relationships, while NoSQL databases offer flexible schemas for unstructured data.",

    "The internet transformed global communication and commerce. Starting as ARPANET in the 1960s, billions of "
    "people use it daily for communication, entertainment, education, and business.",

    "Climate change refers to long-term shifts in temperatures and weather patterns. Human activities, "
    "particularly burning fossil fuels, have been the main driver since the Industrial Revolution.",

    "Neuroscience studies the nervous system, including the brain, spinal cord, and peripheral nerves. Modern "
    "neuroscience combines biology, chemistry, physics, and psychology to understand consciousness.",

    "Organic chemistry studies carbon-containing compounds. Carbon's ability to form four covalent bonds makes it "
    "uniquely suited for creating complex molecules essential for life.",

    "The Silk Road was a network of trade routes connecting East and West. It facilitated the exchange of goods, "
    "ideas, and cultures between China, Central Asia, the Middle East, and Europe for over a millennium.",

    "Genetics studies heredity and variation in living organisms. DNA carries the genetic instructions for "
    "development and functioning. The Human Genome Project mapped all human genes by 2003.",

    "Oceanography studies the physical and biological aspects of the ocean. The ocean covers over 70% of Earth's "
    "surface and plays a crucial role in climate regulation and the global carbon cycle.",

    "Cryptography is the practice of securing communication from adversaries. Modern cryptography relies on "
    "mathematical algorithms and is essential for internet security, digital currencies, and data protection.",

    "Volcanic eruptions occur when magma rises to the surface. Volcanoes can be explosive or effusive, and "
    "they shape landscapes, influence climate, and create fertile soil.",

    "The Industrial Revolution began in Britain in the late 18th century. It marked a major turning point in "
    "history, transforming largely agrarian societies into industrial powerhouses.",

    "Renewable energy sources include solar, wind, hydroelectric, and geothermal power. Transitioning from "
    "fossil fuels to renewables is critical for reducing greenhouse gas emissions.",
]

# ======================================================================
# Chinese haystack passages
# ======================================================================

HAYSTACK_PASSAGES_ZH = [
    "计算机科学的发展经历了数十年的演进，从早期的机械计算器到现代的量子计算系统。人工智能是计算机科学的一个重要分支，致力于创建能模拟人类智能的系统。深度学习在图像识别和自然语言处理方面取得了显著成果。",
    "光合作用是绿色植物将光能转化为化学能的过程。植物利用叶绿素吸收二氧化碳和水，产生葡萄糖和氧气。这个过程对地球生命至关重要，构成了大多数食物链的基础。",
    "量子力学是物理学的基本理论，描述原子和亚原子粒子的性质。量子计算利用量子叠加和纠缠等现象，有望在某些问题上实现指数级加速。",
    "水循环描述地球表面水的持续运动。水从海洋和湖泊蒸发，通过凝结形成云，再以降水形式回到地面。地球上约97%的水储存在海洋中。",
    "气候变暖是长期气温和天气模式的变化。自工业革命以来，人类活动特别是化石燃料的燃烧是主要驱动因素。温室气体增加导致全球气温上升、冰川融化和极端天气事件增加。",
    "机器学习算法基于训练数据构建模型，做出预测或决策。监督学习、无监督学习和强化学习是三种主要范式，每种适用于不同的问题类型。",
    "长城是中国古代的军事防御工程，修筑历史可上溯到西周时期。明长城是保存最完好的部分，全长超过两万公里，是世界文化遗产。",
    "基因组学是研究生物体基因组的学科。DNA携带着遗传信息，指导生物体的发育和功能。人类基因组计划于2003年完成了全部人类基因的测序。",
    "海洋学研究海洋的物理和生物学方面。海洋覆盖地球表面的70%以上，在气候调节和全球碳循环中起着关键作用。",
    "密码学是保护通信安全的学科。现代密码学依赖数学算法，对互联网安全、数字货币和数据保护至关重要。",
]


# ======================================================================
# UTILITIES
# ======================================================================

def generate_haystack_tokens(tokenizer, target_tokens: int, rng: random.Random, lang: str = "en") -> List[int]:
    """Generate filler text tokens to pad context to target length."""
    passages = HAYSTACK_PASSAGES_ZH if lang == "zh" else HAYSTACK_PASSAGES
    tokens: List[int] = []
    while len(tokens) < target_tokens:
        passage = rng.choice(passages)
        new_tokens = tokenizer.encode(passage, add_special_tokens=False)
        tokens.extend(new_tokens)
    return tokens[:target_tokens]


def answer_exact_match(model_output: str, ground_truth: str, case_sensitive: bool = False) -> bool:
    """Check if the ground truth appears in the model output."""
    output = model_output.strip()
    truth = ground_truth.strip()
    if not case_sensitive:
        output = output.lower()
        truth = truth.lower()
    return truth in output


def answer_regex_match(model_output: str, pattern: str) -> bool:
    """Check if the model output matches a regex pattern."""
    return bool(re.search(pattern, model_output))


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics from a list of per-sample results."""
    if not results:
        return {}
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


# ======================================================================
# BENCHMARK: Multi-Needle Recall
# ======================================================================

def generate_multi_needle_recall(
    tokenizer, target_ctx_tokens: int, num_needles: int, rng: random.Random,
    lang: str = "en",
) -> Tuple[str, str, Dict]:
    """
    Generate a context with N needles scattered at different depths.
    The question asks for ALL needles — missing any is a failure.
    """
    # Generate needles: (project_name, secret_code)
    needles = []
    codes = set()
    if lang == "zh":
        names = [
            "阿尔法", "贝塔", "伽马", "德尔塔", "艾普西隆", "泽塔", "伊塔", "西塔",
            "卡帕", "拉姆达", "缪", "纽", "克西", "欧米克戎", "派", "柔",
        ]
    else:
        names = [
            "Project Alpha", "Project Beta", "Project Gamma", "Project Delta",
            "Project Epsilon", "Project Zeta", "Project Eta", "Project Theta",
            "Project Iota", "Project Kappa", "Project Lambda", "Project Mu",
            "Project Nu", "Project Xi", "Project Omicron", "Project Pi",
        ]
    rng.shuffle(names)
    for i in range(num_needles):
        name = names[i]
        while True:
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
            if code not in codes:
                codes.add(code)
                break
        needles.append((name, code))

    # Compute needle tokens to know how much haystack to generate
    needle_texts = []
    for name, code in needles:
        if lang == "zh":
            needle_texts.append(f"[INFO] {name} 的编号是 {code}。")
        else:
            needle_texts.append(f"[INFO] The secret code for {name} is {code}.")
    needle_token_sets = [tokenizer.encode(t, add_special_tokens=False) for t in needle_texts]
    total_needle_tokens = sum(len(t) for t in needle_token_sets)

    # Generate question
    if lang == "zh":
        question = "请列出所有项目的编号。"
    else:
        question = "List the secret code for every project mentioned above."

    question_tokens = tokenizer.encode(question, add_special_tokens=False)

    # Haystack padding
    available_for_haystack = max(0, target_ctx_tokens - total_needle_tokens - len(question_tokens) - 50)
    haystack_tokens = generate_haystack_tokens(tokenizer, available_for_haystack, rng, lang)

    # Distribute needles at different depths
    positions = [int((i + 1) * len(haystack_tokens) / (num_needles + 1)) for i in range(num_needles)]
    full_tokens = list(haystack_tokens)
    for pos, needle_tok_set in zip(positions, needle_token_sets):
        full_tokens = full_tokens[:pos] + needle_tok_set + full_tokens[pos:]
    full_tokens.extend(question_tokens)

    context_str = tokenizer.decode(full_tokens, skip_special_tokens=True)

    # Ground truth: all codes in order
    ground_truth = ", ".join(f"{name}={code}" for name, code in needles)

    metadata = {
        "num_needles": num_needles,
        "needles": [(n, c) for n, c in needles],
        "positions": positions,
        "total_tokens": len(full_tokens),
    }

    return context_str, question, ground_truth, metadata


def evaluate_multi_needle_recall(model_output: str, needles: List[Tuple[str, str]]) -> bool:
    """All needles must be present in output."""
    output_lower = model_output.lower()
    for name, code in needles:
        if code.lower() not in output_lower:
            return False
    return True


# ======================================================================
# BENCHMARK: Associative Recall (Key-Value pairs)
# ======================================================================

def generate_associative_recall(
    tokenizer, target_ctx_tokens: int, num_pairs: int, rng: random.Random,
    lang: str = "en",
) -> Tuple[str, str, str, Dict]:
    """
    Scatter (key→value) pairs throughout context, then query by key.
    Must recall the correct value for a randomly selected key.
    """
    # Generate pairs
    if lang == "zh":
        entities = [
            "张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十",
            "陈一", "林二", "黄三", "杨四", "刘五", "徐六", "马七", "高八",
        ]
        attributes = ["电话", "邮箱", "地址", "公司", "职位", "城市"]
    else:
        entities = [
            "Alice Johnson", "Bob Smith", "Carol Williams", "David Brown",
            "Eve Davis", "Frank Miller", "Grace Wilson", "Henry Moore",
            "Ivy Taylor", "Jack Anderson", "Karen Thomas", "Leo Jackson",
            "Mia White", "Noah Harris", "Olivia Martin", "Paul Garcia",
        ]
        attributes = ["phone number", "email", "address", "company", "job title", "city"]

    rng.shuffle(entities)
    pairs = []
    used_values = set()
    for i in range(num_pairs):
        entity = entities[i]
        attr = rng.choice(attributes)
        while True:
            if lang == "zh":
                value = f"{attr}:{''.join(rng.choices(string.digits, k=8))}"
            else:
                value = f"{attr}: {''.join(rng.choices(string.ascii_lowercase + string.digits, k=8))}"
            if value not in used_values:
                used_values.add(value)
                break
        pairs.append((entity, value))

    # Build needle texts
    needle_texts = []
    for entity, value in pairs:
        if lang == "zh":
            needle_texts.append(f"[记录] {entity}的{attr}是 {value.split(':')[1] if ':' in value else value}")
        else:
            needle_texts.append(f"[Record] {entity}'s {attr} is {value.split(': ', 1)[1] if ': ' in value else value}")
    needle_token_sets = [tokenizer.encode(t, add_special_tokens=False) for t in needle_texts]
    total_needle_tokens = sum(len(t) for t in needle_token_sets)

    # Pick query target
    query_idx = rng.randint(0, num_pairs - 1)
    query_entity, query_value = pairs[query_idx]
    if lang == "zh":
        question = f"请问 {query_entity} 的{attr}是什么？请直接回答编号。"
    else:
        question = f"What is {query_entity}'s {attr}? Answer with the value only."

    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    available_for_haystack = max(0, target_ctx_tokens - total_needle_tokens - len(question_tokens) - 50)
    haystack_tokens = generate_haystack_tokens(tokenizer, available_for_haystack, rng, lang)

    positions = [int((i + 1) * len(haystack_tokens) / (num_pairs + 1)) for i in range(num_pairs)]
    full_tokens = list(haystack_tokens)
    for pos, needle_tok_set in zip(positions, needle_token_sets):
        full_tokens = full_tokens[:pos] + needle_tok_set + full_tokens[pos:]
    full_tokens.extend(question_tokens)

    context_str = tokenizer.decode(full_tokens, skip_special_tokens=True)
    ground_truth = query_value.split(":")[-1].strip()

    metadata = {
        "num_pairs": num_pairs,
        "query_entity": query_entity,
        "query_attribute": attr,
        "positions": positions,
        "total_tokens": len(full_tokens),
    }
    return context_str, question, ground_truth, metadata


# ======================================================================
# BENCHMARK: Passkey Retrieval (hard variant)
# ======================================================================

def generate_passkey_hard(
    tokenizer, target_ctx_tokens: int, num_keys: int, rng: random.Random,
    lang: str = "en",
) -> Tuple[str, str, str, Dict]:
    """
    Classic passkey but with multiple keys. The question asks for a specific one.
    Distractor keys are similar-looking to increase confusion.
    """
    # Generate passkeys
    keys = []
    used = set()
    for _ in range(num_keys):
        while True:
            key = "".join(rng.choices(string.ascii_uppercase, k=3)) + "-" + "".join(rng.choices(string.digits, k=4))
            if key not in used:
                used.add(key)
                break
        keys.append(key)

    # Build context blocks, each containing one key at a random position
    needle_token_sets = []
    key_labels = []
    for i, key in enumerate(keys):
        if lang == "zh":
            text = f"第{i+1}批数据：授权码 {key}。请注意保存此码。"
        else:
            text = f"Batch {i+1} data: authorization key {key}. Please keep this code safe."
        needle_token_sets.append(tokenizer.encode(text, add_special_tokens=False))
        key_labels.append(f"Batch {i+1}" if lang == "en" else f"第{i+1}批")

    total_needle_tokens = sum(len(t) for t in needle_token_sets)

    # Query a random key
    query_idx = rng.randint(0, num_keys - 1)
    if lang == "zh":
        question = f"请问第{query_idx+1}批数据的授权码是什么？"
    else:
        question = f"What is the authorization key for batch {query_idx+1}?"

    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    available_for_haystack = max(0, target_ctx_tokens - total_needle_tokens - len(question_tokens) - 50)
    haystack_tokens = generate_haystack_tokens(tokenizer, available_for_haystack, rng, lang)

    positions = [int((i + 1) * len(haystack_tokens) / (num_keys + 1)) for i in range(num_keys)]
    full_tokens = list(haystack_tokens)
    for pos, needle_tok_set in zip(positions, needle_token_sets):
        full_tokens = full_tokens[:pos] + needle_tok_set + full_tokens[pos:]
    full_tokens.extend(question_tokens)

    context_str = tokenizer.decode(full_tokens, skip_special_tokens=True)
    ground_truth = keys[query_idx]

    metadata = {
        "num_keys": num_keys,
        "query_idx": query_idx,
        "keys": keys,
        "positions": positions,
        "total_tokens": len(full_tokens),
    }
    return context_str, question, ground_truth, metadata


# ======================================================================
# BENCHMARK: Counterfactual Retrieval
# ======================================================================

def generate_counterfactual_retrieval(
    tokenizer, target_ctx_tokens: int, num_updates: int, rng: random.Random,
    lang: str = "en",
) -> Tuple[str, str, str, Dict]:
    """
    Facts appear, then get overwritten with counter-facts. Must retrieve the LATEST version.
    This tests whether memory systems preserve temporal ordering.
    """
    if lang == "zh":
        subjects = ["张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十"]
        cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]
        companies = ["腾讯", "阿里", "字节跳动", "百度", "华为", "小米", "京东", "美团"]
    else:
        subjects = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
        cities = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Mumbai"]
        companies = ["Google", "Apple", "Microsoft", "Amazon", "Meta", "Tesla", "Netflix", "Oracle"]

    rng.shuffle(subjects)
    rng.shuffle(cities)

    # For each subject, create 2-3 successive city assignments
    subject_facts = {}
    for i in range(min(num_updates, len(subjects))):
        subj = subjects[i]
        n_versions = min(rng.randint(2, 3), len(cities))
        chosen_cities = [cities[j] for j in range(n_versions)]
        subject_facts[subj] = chosen_cities

    # Build text
    fact_texts = []
    for subj, city_list in subject_facts.items():
        for j, city in enumerate(city_list):
            if lang == "zh":
                fact_texts.append(f"{subj} 搬到了{city}。")
            else:
                fact_texts.append(f"{subj} moved to {city}.")

    needle_token_sets = [tokenizer.encode(t, add_special_tokens=False) for t in fact_texts]
    total_needle_tokens = sum(len(t) for t in needle_token_sets)

    # Query the latest city for each subject
    query_subj = list(subject_facts.keys())[0]
    latest_city = subject_facts[query_subj][-1]
    if lang == "zh":
        question = f"{query_subj} 目前住在哪个城市？"
    else:
        question = f"Where does {query_subj} currently live?"

    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    available_for_haystack = max(0, target_ctx_tokens - total_needle_tokens - len(question_tokens) - 50)
    haystack_tokens = generate_haystack_tokens(tokenizer, available_for_haystack, rng, lang)

    # Evenly distribute all fact texts
    n_facts = len(fact_texts)
    positions = [int((i + 1) * len(haystack_tokens) / (n_facts + 1)) for i in range(n_facts)]
    full_tokens = list(haystack_tokens)
    for pos, needle_tok_set in zip(positions, needle_token_sets):
        full_tokens = full_tokens[:pos] + needle_tok_set + full_tokens[pos:]
    full_tokens.extend(question_tokens)

    context_str = tokenizer.decode(full_tokens, skip_special_tokens=True)

    # All cities for this subject (for distractor detection)
    all_cities = subject_facts[query_subj]

    metadata = {
        "num_updates": num_updates,
        "query_subject": query_subj,
        "all_versions": all_cities,
        "latest_answer": latest_city,
        "positions": positions,
        "total_tokens": len(full_tokens),
    }
    return context_str, question, latest_city, metadata


# ======================================================================
# BENCHMARK: Synthetic Reasoning Chain
# ======================================================================

def generate_reasoning_chain(
    tokenizer, target_ctx_tokens: int, chain_length: int, rng: random.Random,
    lang: str = "en",
) -> Tuple[str, str, str, Dict]:
    """
    Chain of facts: A=B, B=C, C=D, ... Must infer A=D.
    Facts are scattered across long context with distractors.
    """
    if lang == "zh":
        prefixes = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸",
                     "子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉"]
    else:
        prefixes = [f"X{i}" for i in range(20)]

    rng.shuffle(prefixes)
    values = []
    used = set()
    while len(values) < chain_length + 1:
        v = "".join(rng.choices(string.ascii_lowercase, k=4))
        if v not in used:
            used.add(v)
            values.append(v)

    # Build chain facts: prefix[0]=values[0], prefix[0]=prefix[1], prefix[1]=values[1], ...
    # Simpler: entity_0 → value_0, entity_1 → value_1, ..., plus links entity_0=entity_1, entity_1=entity_2
    facts = []
    for i in range(chain_length):
        if lang == "zh":
            facts.append(f"{prefixes[i]} 的代号是 {values[i]}。")
        else:
            facts.append(f"{prefixes[i]}'s codename is {values[i]}.")
    for i in range(chain_length - 1):
        if lang == "zh":
            facts.append(f"{prefixes[i]} 和 {prefixes[i+1]} 是同一个人。")
        else:
            facts.append(f"{prefixes[i]} is the same person as {prefixes[i+1]}.")

    # Query: what is prefixes[0]'s codename? → values[chain_length-1]
    if lang == "zh":
        question = f"根据以上信息，{prefixes[0]} 的代号是什么？"
    else:
        question = f"Based on the above, what is {prefixes[0]}'s codename?"

    ground_truth = values[chain_length - 1]

    needle_token_sets = [tokenizer.encode(f, add_special_tokens=False) for f in facts]
    total_needle_tokens = sum(len(t) for t in needle_token_sets)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    available_for_haystack = max(0, target_ctx_tokens - total_needle_tokens - len(question_tokens) - 50)
    haystack_tokens = generate_haystack_tokens(tokenizer, available_for_haystack, rng, lang)

    n_facts = len(facts)
    positions = [int((i + 1) * len(haystack_tokens) / (n_facts + 1)) for i in range(n_facts)]
    full_tokens = list(haystack_tokens)
    for pos, needle_tok_set in zip(positions, needle_token_sets):
        full_tokens = full_tokens[:pos] + needle_tok_set + full_tokens[pos:]
    full_tokens.extend(question_tokens)

    context_str = tokenizer.decode(full_tokens, skip_special_tokens=True)

    metadata = {
        "chain_length": chain_length,
        "prefixes": prefixes[:chain_length],
        "values": values[:chain_length],
        "ground_truth": ground_truth,
        "positions": positions,
        "total_tokens": len(full_tokens),
    }
    return context_str, question, ground_truth, metadata


# ======================================================================
# BENCHMARK REGISTRY
# ======================================================================

BENCHMARK_REGISTRY = {
    "multi_needle_recall": {
        "generate": generate_multi_needle_recall,
        "evaluate": lambda output, meta: evaluate_multi_needle_recall(output, meta["needles"]),
        "default_params": {"num_needles": 8},
        "param_sweep": {"num_needles": [4, 8, 12, 16]},
        "description": "Recall ALL needles scattered across context. Missing any = fail.",
    },
    "associative_recall": {
        "generate": generate_associative_recall,
        "evaluate": lambda output, meta: answer_exact_match(output, meta["_ground_truth"]),
        "default_params": {"num_pairs": 12},
        "param_sweep": {"num_pairs": [6, 12, 18, 24]},
        "description": "Recall value for a queried key from many scattered key-value pairs.",
    },
    "passkey_hard": {
        "generate": generate_passkey_hard,
        "evaluate": lambda output, meta: answer_exact_match(output, meta["_ground_truth"]),
        "default_params": {"num_keys": 10},
        "param_sweep": {"num_keys": [5, 10, 15, 20]},
        "description": "Retrieve a specific passkey from many similar-looking keys.",
    },
    "counterfactual_retrieval": {
        "generate": generate_counterfactual_retrieval,
        "evaluate": lambda output, meta: answer_exact_match(
            output, meta["_ground_truth"], case_sensitive=False
        ),
        "default_params": {"num_updates": 6},
        "param_sweep": {"num_updates": [3, 6, 9, 12]},
        "description": "Retrieve the LATEST version of a fact after multiple overwrites.",
    },
    "reasoning_chain": {
        "generate": generate_reasoning_chain,
        "evaluate": lambda output, meta: answer_exact_match(output, meta["_ground_truth"]),
        "default_params": {"chain_length": 6},
        "param_sweep": {"chain_length": [3, 5, 7, 9]},
        "description": "Multi-hop inference across scattered facts in long context.",
    },
}


# ======================================================================
# INFERENCE
# ======================================================================

@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256, use_chat_template: bool = False) -> str:
    """Generate a response from the model."""
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=131072)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=131072)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hard long-context memory benchmarks for evaluating memory compression strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks at 32K context
  python scripts/eval_memory_hard_benchmarks.py --model_path ./model/ --context_lengths 32768

  # Run specific benchmark with trials
  python scripts/eval_memory_hard_benchmarks.py --model_path ./model/ \\
      --benchmarks multi_needle_recall passkey_hard --trials 5

  # Custom benchmark config (JSON)
  python scripts/eval_memory_hard_benchmarks.py --model_path ./model/ \\
      --custom_config configs/hard_bench_custom.json

Custom config format:
  {
    "benchmarks": [
      {
        "name": "multi_needle_recall",
        "params": {"num_needles": 10},
        "context_lengths": [16384, 65536],
        "trials": 3
      }
    ]
  }
        """,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        choices=list(BENCHMARK_REGISTRY.keys()),
                        help="Which benchmarks to run (default: all)")
    parser.add_argument("--context_lengths", nargs="+", type=int,
                        default=[16384, 32768, 65536, 131072],
                        help="Context lengths to test")
    parser.add_argument("--trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--chat_template", action="store_true", help="Use model's chat template for prompting (needed for instruct/chat models)")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--custom_config", type=str, default=None,
                        help="Path to JSON config file for custom benchmark setup")
    parser.add_argument("--max_context_tokens", type=int, default=None,
                        help="Cap on context tokens (truncates model max_length if needed)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Only generate test data, don't run inference (for debugging)")
    parser.add_argument("--batch_size", type=int, default=1, help="Unused; kept for compat")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Determine benchmarks to run
    if args.custom_config:
        with open(args.custom_config) as f:
            custom_config = json.load(f)
        benchmark_configs = custom_config.get("benchmarks", [])
    else:
        bench_names = args.benchmarks or list(BENCHMARK_REGISTRY.keys())
        benchmark_configs = [{"name": name, "params": BENCHMARK_REGISTRY[name]["default_params"]}
                             for name in bench_names]

    logger.info(f"Model: {args.model_path}")
    logger.info(f"Benchmarks: {[c['name'] for c in benchmark_configs]}")
    logger.info(f"Context lengths: {args.context_lengths}")
    logger.info(f"Trials: {args.trials}")

    if not args.skip_generation:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": f"cuda:{args.gpu}"},
            trust_remote_code=True,
        )
        model.eval()

        # Determine effective max context
        model_max = getattr(model.config, "max_position_embeddings", 131072)
        if args.max_context_tokens:
            max_ctx = args.max_context_tokens
            # Extend RoPE if needed to support longer contexts
            if args.max_context_tokens > model_max:
                logger.info(f"Extending max_position_embeddings from {model_max} to {args.max_context_tokens}")
                model.config.max_position_embeddings = args.max_context_tokens
                if hasattr(model, 'rotary_emb'):
                    for m in model.modules():
                        if hasattr(m, 'max_seq_len'):
                            m.max_seq_len = args.max_context_tokens
        else:
            max_ctx = model_max
        logger.info(f"Effective max context: {max_ctx} tokens")

    # Run benchmarks
    all_results = []
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for bench_cfg in benchmark_configs:
        bench_name = bench_cfg["name"]
        bench_params = bench_cfg.get("params", BENCHMARK_REGISTRY.get(bench_name, {}).get("default_params", {}))
        ctx_lengths = bench_cfg.get("context_lengths", args.context_lengths)
        trials = bench_cfg.get("trials", args.trials)

        if bench_name not in BENCHMARK_REGISTRY:
            logger.warning(f"Unknown benchmark: {bench_name}, skipping")
            continue

        bench_info = BENCHMARK_REGISTRY[bench_name]
        gen_fn = bench_info["generate"]
        eval_fn = bench_info["evaluate"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark: {bench_name}")
        logger.info(f"Params: {bench_params}")
        logger.info(f"Context lengths: {ctx_lengths}")
        logger.info(f"Description: {bench_info['description']}")

        for ctx_len in ctx_lengths:
            if ctx_len > max_ctx:
                logger.info(f"  Skipping ctx_len={ctx_len} > max_context={max_ctx}")
                continue

            trial_results = []
            for trial in range(trials):
                trial_seed = args.seed + trial * 1000 + hash(bench_name) % 1000
                trial_rng = random.Random(trial_seed)

                logger.info(f"  ctx={ctx_len}, trial={trial+1}/{trials} ...")

                # Generate
                gen_args = {"tokenizer": tokenizer, "target_ctx_tokens": ctx_len, "rng": trial_rng, "lang": args.lang}
                gen_args.update(bench_params)
                result = gen_fn(**gen_args)
                context_str, question, ground_truth, metadata = result
                metadata["_ground_truth"] = ground_truth

                # Truncate if exceeding model max
                prompt = context_str + "\n" + question if question not in context_str else context_str

                if args.skip_generation:
                    correct = None
                    response = ""
                    logger.info(f"    [SKIP] Generated test case, ground_truth={ground_truth}")
                else:
                    t0 = time.time()
                    response = generate_response(model, tokenizer, prompt, args.max_new_tokens, use_chat_template=args.chat_template)
                    gen_time = time.time() - t0

                    correct = eval_fn(response, metadata)
                    logger.info(f"    {'✓' if correct else '✗'} GT={ground_truth!r} | Response={response[:100]!r} | {gen_time:.1f}s")

                trial_results.append({
                    "trial": trial,
                    "correct": correct,
                    "response": response if not args.skip_generation else None,
                    "ground_truth": ground_truth,
                    "metadata": metadata,
                    "gen_time": time.time() if args.skip_generation else None,
                })

            # Aggregate
            scored = [t for t in trial_results if t["correct"] is not None]
            metrics = compute_metrics(scored) if scored else {"accuracy": None, "correct": 0, "total": 0}

            result_entry = {
                "benchmark": bench_name,
                "params": bench_params,
                "context_length": ctx_len,
                "trials": trials,
                "metrics": metrics,
                "trial_details": trial_results,
                "lang": args.lang,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result_entry)

            logger.info(f"  Result: {metrics['accuracy']*100:.1f}% ({metrics['correct']}/{metrics['total']})")

    # Save results
    if output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = output_dir / f"hard_benchmarks_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\nResults saved to {out_file}")

        # Also save a summary table
        summary_file = output_dir / f"hard_benchmarks_{ts}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Hard Benchmark Results — {datetime.now().isoformat()}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"{'='*80}\n\n")
            for r in all_results:
                acc = r["metrics"]["accuracy"]
                acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
                f.write(f"{r['benchmark']:25s} | ctx={r['context_length']:>6d} | "
                        f"{r['metrics']['correct']}/{r['metrics']['total']:>3d} = {acc_str}\n")
        logger.info(f"Summary saved to {summary_file}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for r in all_results:
        acc = r["metrics"]["accuracy"]
        acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
        logger.info(f"  {r['benchmark']:25s} ctx={r['context_length']:>6d}  "
                     f"{r['metrics']['correct']}/{r['metrics']['total']:>3d} = {acc_str}")

    if not args.skip_generation:
        logger.info("\nDone.")


if __name__ == "__main__":
    main()
