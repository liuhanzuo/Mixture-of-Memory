#!/usr/bin/env python3
"""
Prepare mixed dataset for RMT v2.1 training.
Mixes English and Chinese Wikipedia data with 50/50 ratio.
"""

import json
import random
from pathlib import Path


def load_jsonl(file_path):
    """Load JSONL file and return list of dicts."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save list of dicts to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    # Paths
    project_root = Path("/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory")
    data_dir = project_root / "data"

    en_file = data_dir / "rmt_train_wikitext.jsonl"
    zh_file = data_dir / "rmt_train_wiki_zh_10k.jsonl"
    output_file = data_dir / "rmt_train_mixed.jsonl"

    print(f"Loading English data from {en_file}...")
    en_data = load_jsonl(en_file)
    print(f"  Loaded {len(en_data)} documents")

    print(f"Loading Chinese data from {zh_file}...")
    zh_data = load_jsonl(zh_file)
    print(f"  Loaded {len(zh_data)} documents")

    # Determine number of docs to sample (use minimum to get 50/50)
    num_docs = min(len(en_data), len(zh_data))
    print(f"\nSampling {num_docs} docs from each language for 50/50 mix")

    # Sample from Chinese data (if more than English)
    if len(zh_data) > num_docs:
        zh_sampled = random.sample(zh_data, num_docs)
    else:
        zh_sampled = zh_data

    # Use all English data (if fewer than Chinese)
    if len(en_data) > num_docs:
        en_sampled = random.sample(en_data, num_docs)
    else:
        en_sampled = en_data

    # Interleave: alternate between English and Chinese
    mixed_data = []
    for i in range(num_docs):
        mixed_data.append(en_sampled[i])
        mixed_data.append(zh_sampled[i])

    # Shuffle the mixed data to randomize order
    random.shuffle(mixed_data)

    # Save mixed dataset
    print(f"\nSaving mixed dataset to {output_file}...")
    save_jsonl(mixed_data, output_file)

    # Statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total documents: {len(mixed_data)}")
    print(f"English documents: {len(en_sampled)}")
    print(f"Chinese documents: {len(zh_sampled)}")
    print(f"Ratio: {len(en_sampled) / len(mixed_data):.1%} EN / {len(zh_sampled) / len(mixed_data):.1%} ZH")

    # Calculate total text length
    total_chars_en = sum(len(item.get('text', '')) for item in en_sampled)
    total_chars_zh = sum(len(item.get('text', '')) for item in zh_sampled)
    total_chars = total_chars_en + total_chars_zh

    print(f"\nText length statistics:")
    print(f"English: {total_chars_en:,} characters ({total_chars_en / total_chars:.1%})")
    print(f"Chinese: {total_chars_zh:,} characters ({total_chars_zh / total_chars:.1%})")
    print(f"Total: {total_chars:,} characters")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
