"""
RMT 训练数据预处理脚本

功能:
1. 改造现有对话数据为长文本格式
2. 从 HuggingFace 下载长文本数据集
3. 数据长度过滤和切割
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import random


# ============================================================================
# 方案 1: 改造现有对话数据
# ============================================================================

def transform_dialog_to_longtext(
    input_path: str,
    output_path: str,
    target_length: int = 12288,  # 目标 12K tokens
    min_length: int = 8192,      # 最小 8K tokens
    max_length: int = 16384,     # 最大 16K tokens
    concat_mode: str = "session",  # session, random, dialogue
):
    """
    将对话数据改造为适合 RMT 训练的长文本

    Args:
        input_path: 输入 jsonl 文件
        output_path: 输出 jsonl 文件
        target_length: 目标 token 长度
        min_length: 最小 token 长度
        max_length: 最大 token 长度
        concat_mode: 拼接模式
            - "session": 按 session_idx 拼接同一对话的多轮
            - "dialogue": 按 dialogue_id 拼接相关对话
            - "random": 随机拼接对话
    """
    print(f"读取数据: {input_path}")

    # 读取所有数据
    all_samples = []
    with open(input_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # 提取文本
            text = f"{item['input_text']}\n{item['target_text']}"
            if 'memory_texts' in item and item['memory_texts']:
                text += "\n\n" + "\n\n".join(item['memory_texts'])

            all_samples.append({
                'text': text,
                'dialogue_id': item.get('dialogue_id', ''),
                'session_idx': item.get('session_idx', 0),
                'char_length': len(text),
            })

    print(f"读取到 {len(all_samples)} 条对话")
    print(f"平均字符长度: {sum(s['char_length'] for s in all_samples) / len(all_samples):.0f}")

    # 按模式拼接
    long_texts = []

    if concat_mode == "session":
        print("模式: 按 session_idx 拼接同一对话的多轮")
        # 按 dialogue_id + session_idx 分组
        groups = defaultdict(list)
        for sample in all_samples:
            key = f"{sample['dialogue_id']}_{sample['session_idx']}"
            groups[key].append(sample)

        # 每个组内拼接
        for group_id, group_samples in groups.items():
            # 按 session_idx 排序
            group_samples.sort(key=lambda x: x['session_idx'])

            current_text = []
            current_length = 0

            for sample in group_samples:
                if current_length + sample['char_length'] > max_length:
                    if current_length >= min_length:
                        long_texts.append("\n\n".join(current_text))
                    current_text = [sample['text']]
                    current_length = sample['char_length']
                else:
                    current_text.append(sample['text'])
                    current_length += sample['char_length']

            if current_length >= min_length:
                long_texts.append("\n\n".join(current_text))

    elif concat_mode == "dialogue":
        print("模式: 按 dialogue_id 拼接相关对话")
        # 按 dialogue_id 分组
        groups = defaultdict(list)
        for sample in all_samples:
            groups[sample['dialogue_id']].append(sample)

        for dialogue_id, group_samples in groups.items():
            # 随机打乱增加多样性
            random.shuffle(group_samples)

            current_text = []
            current_length = 0

            for sample in group_samples:
                if current_length + sample['char_length'] > max_length:
                    if current_length >= min_length:
                        long_texts.append("\n\n".join(current_text))
                    current_text = [sample['text']]
                    current_length = sample['char_length']
                else:
                    current_text.append(sample['text'])
                    current_length += sample['char_length']

            if current_length >= min_length:
                long_texts.append("\n\n".join(current_text))

    elif concat_mode == "random":
        print("模式: 随机拼接对话")
        # 随机打乱
        random.shuffle(all_samples)

        current_text = []
        current_length = 0

        for sample in all_samples:
            if current_length + sample['char_length'] > max_length:
                if current_length >= min_length:
                    long_texts.append("\n\n".join(current_text))
                current_text = [sample['text']]
                current_length = sample['char_length']
            else:
                current_text.append(sample['text'])
                current_length += sample['char_length']

        if current_length >= min_length:
            long_texts.append("\n\n".join(current_text))

    else:
        raise ValueError(f"Unknown concat_mode: {concat_mode}")

    # 输出结果
    print(f"\n生成 {len(long_texts)} 条长文本")
    print(f"长度分布:")
    lengths = [len(text) for text in long_texts]
    print(f"  平均: {sum(lengths)/len(lengths):.0f}")
    print(f"  最小: {min(lengths)}")
    print(f"  最大: {max(lengths)}")
    print(f"  中位数: {sorted(lengths)[len(lengths)//2]}")

    # 保存
    with open(output_path, "w") as f:
        for i, text in enumerate(long_texts):
            out_item = {
                "text": text,
                "char_length": len(text),
                "source": input_path,
                "concat_mode": concat_mode,
            }
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print(f"\n保存到: {output_path}")


# ============================================================================
# 方案 2: 从 HuggingFace 下载长文本数据集
# ============================================================================

def download_long_dataset_from_hf(
    dataset_name: str,
    config: str = None,
    split: str = "train",
    output_path: str = None,
    num_samples: int = 10000,
    min_length: int = 8192,
    max_length: int = 16384,
):
    """
    从 HuggingFace 下载长文本数据集

    推荐数据集:
    - wikimedia/wikipedia (zh subset): 中文维基百科
    - skywork/SkyPile-150B: 大规模中文语料
    - deepmind/pg19: 英文书籍
    - allenai/c4: 网页文本

    Args:
        dataset_name: HF 数据集名称
        config: 数据集配置
        split: 数据集 split
        output_path: 输出路径
        num_samples: 下载数量
        min_length: 最小长度
        max_length: 最大长度
    """
    print(f"尝试从 HuggingFace 下载数据集: {dataset_name}")

    try:
        from transformers import AutoTokenizer

        # 使用 Qwen tokenizer 计算 token 长度
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

        from datasets import load_dataset
        print("加载数据集...")

        ds = load_dataset(dataset_name, config, split=split, streaming=True)

        # 如果未指定输出路径,自动生成
        if output_path is None:
            safe_name = dataset_name.replace("/", "_")
            output_path = f"data/rmt_train_{safe_name}.jsonl"

        count = 0
        filtered_count = 0

        with open(output_path, "w") as f:
            for item in ds:
                # 提取文本
                if "text" in item:
                    text = item["text"]
                elif "article" in item:
                    text = item["article"]
                elif "content" in item:
                    text = item["content"]
                else:
                    continue

                # 过滤空文本
                if not text or len(text) < 100:
                    continue

                # 计算 token 长度
                token_length = len(tokenizer.encode(text))

                # 过滤长度
                if token_length < min_length or token_length > max_length:
                    continue

                # 保存
                out_item = {
                    "text": text,
                    "char_length": len(text),
                    "token_length": token_length,
                    "source": dataset_name,
                }
                f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

                count += 1

                if count % 100 == 0:
                    print(f"已保存 {count} 条样本...")

                if count >= num_samples:
                    break

        print(f"\n下载完成!")
        print(f"保存到: {output_path}")
        print(f"总样本数: {count}")

    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("运行: pip install datasets")
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n可能的原因:")
        print("1. HuggingFace 网络不可访问 (可尝试使用镜像: export HF_ENDPOINT=https://hf-mirror.com)")
        print("2. 数据集名称或配置错误")
        print("3. 需要特殊权限")


# ============================================================================
# 方案 3: 推荐的数据集
# ============================================================================

RECOMMENDED_DATASETS = {
    "中文长文本": [
        {
            "name": "wikimedia/wikipedia",
            "config": "20231101.zh",
            "size": "~2-5GB",
            "samples": "100K+",
            "avg_length": "1K-10K tokens",
            "language": "中文",
            "license": "CC-BY-SA 3.0",
            "score": 5,
            "notes": "中文维基百科,质量高,话题多样,立即可用",
        },
        {
            "name": "AI-ModelScope/Wudao",
            "config": None,
            "size": "~200GB",
            "samples": "500K+",
            "avg_length": "变",
            "language": "中文",
            "license": "Apache 2.0",
            "score": 5,
            "notes": "悟道中文语料,大规模,高质量,建议先下载5-10GB子集",
        },
        {
            "name": "Skywork/SkyPile-150B",
            "config": None,
            "size": "~150B tokens",
            "samples": "海量",
            "avg_length": "变",
            "language": "中文",
            "license": "Apache 2.0",
            "score": 4,
            "notes": "大规模中文语料,需要抽样使用",
        },
        {
            "name": "mC4 (zh-CN)",
            "config": "zh",
            "size": "~38TB (全量)",
            "samples": "海量",
            "avg_length": "变",
            "language": "中文",
            "license": "ODC-BY",
            "score": 3,
            "notes": "网页文本,需要过滤,质量参差不齐",
        },
    ],
    "英文长文本": [
        {
            "name": "deepmind/pg19",
            "config": None,
            "size": "~11GB",
            "samples": "~28K",
            "avg_length": "10K-100K tokens",
            "language": "英文",
            "license": "Apache 2.0",
            "score": 5,
            "notes": "Project Gutenberg 书籍,超长文本,适合RMT",
        },
        {
            "name": "EleutherAI/the_pile_deduplicated",
            "config": None,
            "size": "~825GB",
            "samples": "~200K (长文本)",
            "avg_length": "变",
            "language": "英文",
            "license": "Various",
            "score": 4,
            "notes": "包含 arXiv, Books3 等子集,需要过滤",
        },
    ],
}


def print_recommended_datasets():
    """打印推荐的数据集"""
    print("\n" + "="*80)
    print("推荐的长文本数据集 (用于 RMT 训练)")
    print("="*80)

    for category, datasets in RECOMMENDED_DATASETS.items():
        print(f"\n### {category}")
        print()

        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds['name']}")
            print(f"   配置: {ds['config']}")
            print(f"   大小: {ds['size']}")
            print(f"   样本数: {ds['samples']}")
            print(f"   平均长度: {ds['avg_length']}")
            print(f"   语言: {ds['language']}")
            print(f"   许可证: {ds['license']}")
            print(f"   适配性评分: {ds['score']}/5")
            print(f"   备注: {ds['notes']}")
            print()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RMT 训练数据预处理")

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 命令 1: transform - 改造现有数据
    parser_transform = subparsers.add_parser("transform", help="改造现有对话数据为长文本")
    parser_transform.add_argument("--input", type=str, required=True, help="输入 jsonl 文件")
    parser_transform.add_argument("--output", type=str, required=True, help="输出 jsonl 文件")
    parser_transform.add_argument("--target-length", type=int, default=12288, help="目标 token 长度")
    parser_transform.add_argument("--min-length", type=int, default=8192, help="最小 token 长度")
    parser_transform.add_argument("--max-length", type=int, default=16384, help="最大 token 长度")
    parser_transform.add_argument("--concat-mode", type=str, default="session", choices=["session", "dialogue", "random"], help="拼接模式")

    # 命令 2: download - 从 HuggingFace 下载数据集
    parser_download = subparsers.add_parser("download", help="从 HuggingFace 下载数据集")
    parser_download.add_argument("--dataset", type=str, required=True, help="HF 数据集名称")
    parser_download.add_argument("--config", type=str, default=None, help="数据集配置")
    parser_download.add_argument("--split", type=str, default="train", help="数据集 split")
    parser_download.add_argument("--output", type=str, default=None, help="输出路径")
    parser_download.add_argument("--num-samples", type=int, default=10000, help="下载数量")
    parser_download.add_argument("--min-length", type=int, default=8192, help="最小 token 长度")
    parser_download.add_argument("--max-length", type=int, default=16384, help="最大 token 长度")

    # 命令 3: list - 列出推荐数据集
    subparsers.add_parser("list", help="列出推荐的数据集")

    args = parser.parse_args()

    if args.command == "transform":
        transform_dialog_to_longtext(
            input_path=args.input,
            output_path=args.output,
            target_length=args.target_length,
            min_length=args.min_length,
            max_length=args.max_length,
            concat_mode=args.concat_mode,
        )

    elif args.command == "download":
        download_long_dataset_from_hf(
            dataset_name=args.dataset,
            config=args.config,
            split=args.split,
            output_path=args.output,
            num_samples=args.num_samples,
            min_length=args.min_length,
            max_length=args.max_length,
        )

    elif args.command == "list":
        print_recommended_datasets()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
