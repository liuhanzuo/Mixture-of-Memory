#!/usr/bin/env python3
"""
构造"必须从记忆中回忆"的训练数据。

核心思路:
1. 从现有对话记忆中提取具体事实（人名、数字、日期、具体细节）
2. 生成关于这些事实的查询问题
3. target 必须包含记忆中的具体信息（不能用通用知识回答）
4. 同时保留部分无关记忆作为干扰

数据格式与现有训练格式一致:
{
    "input_text": "基于之前的对话，回答问题：...",
    "target_text": "具体答案（包含记忆中的事实）",
    "memory_texts": ["...", ...],
    "relevant_indices": [0, 2],
    "dialogue_id": "...",
    "session_idx": 0,
    "recall_required": true  # 标记：target 必须依赖记忆
}
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import Optional


def extract_facts_from_memory(memory_text: str) -> list[dict]:
    """从记忆文本中提取可查询的事实"""
    facts = []
    
    # 提取数字+单位的事实
    number_patterns = [
        r'(\d+[\.\d]*)\s*(万|块|元|块|月|年|天|周|个|人|台|次|分|小时|分钟|秒|%|万块)',
        r'(薪资|月薪|年薪|价格|成本|预算|费用|估值|融资|营收|利润|收入)\s*(?:是|为|约|大概|差不多)\s*(\d+[\.\d]*\s*万?块?元?)',
    ]
    
    # 提取人名
    name_pattern = r'(?:叫|名字是|他是|她是)\s*([^\s，。！？]{2,3})(?:的|，|。|！|？|$)'
    
    # 提取具体地点
    place_pattern = r'(?:在|去|到|地址是)\s*([^\s，。！？]{4,15}(?:路|街|区|园|楼|室|号))'
    
    # 提取时间信息
    time_pattern = r'(?:下周|这周|明天|后天|今天|周一|周二|周三|周四|周五|周六|周日|星期[一二三四五六七]|\d+月\d+[日号]|\d+号)'
    
    for line in memory_text.split('\n'):
        # 数字事实
        for m in re.finditer(r'(\d+[\.\d]*)\s*(万|块|元|月|年|天|周|人|台|次|小时|分钟|%)', line):
            num = m.group(1)
            unit = m.group(2)
            context = line[max(0, m.start()-30):m.end()+30]
            facts.append({
                'type': 'number',
                'value': f'{num}{unit}',
                'context': context.strip(),
                'full_line': line.strip()
            })
        
        # 人名+职位/角色
        for m in re.finditer(r'(?:叫|名字)\s*(?:是|叫)?\s*([^\s，。]{2,4})(?:，|。|的)', line):
            name = m.group(1)
            context = line[max(0, m.start()-20):m.end()+20]
            facts.append({
                'type': 'name',
                'value': name,
                'context': context.strip(),
                'full_line': line.strip()
            })
    
    return facts


def generate_recall_query(fact: dict, memory_text: str) -> Optional[tuple[str, str]]:
    """基于事实生成 query 和 target"""
    ftype = fact['type']
    value = fact['value']
    context = fact['context']
    full_line = fact['full_line']
    
    if ftype == 'number':
        # 问具体数字
        queries = [
            f"之前提到的具体数字是多少？",
            f"关于这个问题，具体的数字/金额是多少？",
            f"上次讨论的结果是什么？具体数字是多少？",
        ]
        target = full_line  # 完整的回答包含具体数字
        query = random.choice(queries)
        
    elif ftype == 'name':
        queries = [
            f"之前提到的那个人叫什么名字？",
            f"上次说的那个人是谁？",
            f"之前提到的人名是什么？",
        ]
        target = full_line
        query = random.choice(queries)
    
    else:
        return None
    
    return query, target


def generate_recall_samples(
    input_path: str,
    output_path: str,
    num_samples: int = 50000,
    seed: int = 42,
):
    random.seed(seed)
    
    print(f"读取原始数据: {input_path}")
    with open(input_path) as f:
        raw_data = [json.loads(line) for line in f]
    
    print(f"原始数据: {len(raw_data)} 条")
    
    recall_samples = []
    attempts = 0
    
    while len(recall_samples) < num_samples and attempts < num_samples * 5:
        attempts += 1
        sample = random.choice(raw_data)
        memories = sample.get('memory_texts', [])
        relevant = sample.get('relevant_indices', [])
        
        if not relevant or not memories:
            continue
        
        # 从相关记忆中提取事实
        all_facts = []
        for idx in relevant:
            if idx < len(memories):
                facts = extract_facts_from_memory(memories[idx])
                for fact in facts:
                    fact['memory_idx'] = idx
                all_facts.extend(facts)
        
        if not all_facts:
            continue
        
        # 随机选一个事实
        fact = random.choice(all_facts)
        result = generate_recall_query(fact, memories[fact['memory_idx']])
        if result is None:
            continue
        
        query, target = result
        
        # 构造新样本
        # query: 基于之前的对话记忆，回答问题
        input_text = f"根据之前的对话记忆，回答以下问题：{query}"
        
        # 打乱记忆顺序，但记录新的 relevant indices
        memory_indices = list(range(len(memories)))
        random.shuffle(memory_indices)
        shuffled_memories = [memories[i] for i in memory_indices]
        new_relevant = [memory_indices.index(fact['memory_idx'])]
        
        recall_samples.append({
            'input_text': input_text,
            'target_text': target,
            'memory_texts': shuffled_memories,
            'relevant_indices': new_relevant,
            'dialogue_id': f"recall_{sample.get('dialogue_id', 'unknown')}_{attempts}",
            'session_idx': 0,
            'recall_required': True,
        })
    
    print(f"生成了 {len(recall_samples)} 条 recall 样本")
    
    with open(output_path, 'w') as f:
        for s in recall_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    print(f"已写入: {output_path}")
    
    # 打印几个示例
    print("\n=== 示例 ===")
    for s in recall_samples[:3]:
        print(f"\nQuery: {s['input_text']}")
        print(f"Target: {s['target_text'][:150]}...")
        print(f"Memories: {len(s['memory_texts'])}, Relevant: {s['relevant_indices']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/mag_train_generated_causal.jsonl')
    parser.add_argument('--output', default='data/mag_train_recall.jsonl')
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_recall_samples(args.input, args.output, args.num_samples, args.seed)
