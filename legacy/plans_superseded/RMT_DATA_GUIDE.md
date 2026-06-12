# RMT 训练数据获取快速指南

> 更新时间: 2025-04-15
> 目标: 为 RMT 训练获取 8K-16K token 的长文本数据

---

## 🚀 快速开始 (3 步获取数据)

### 步骤 1: 验证用数据 (立即可用)

```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

# 改造现有对话数据
python3 data_generation/rmt_data_preprocessing.py transform \
  --input data/mag_train_generated_causal.jsonl \
  --output data/rmt_train_mag_concat_session.jsonl \
  --concat-mode session \
  --min-length 8192 \
  --max-length 16384

# 预期: ~10K-15K 条长文本样本
```

### 步骤 2: 下载高质量数据 (5-10分钟)

```bash
# 使用 HuggingFace 镜像加速下载
export HF_ENDPOINT=https://hf-mirror.com

# 下载中文 Wikipedia
python3 data_generation/rmt_data_preprocessing.py download \
  --dataset wikimedia/wikipedia \
  --config 20231101.zh \
  --output data/rmt_train_wiki_zh_10k.jsonl \
  --num-samples 10000 \
  --min-length 8192 \
  --max-length 16384

# 预期: ~10K 条高质量长文本
```

### 步骤 3: (可选) 下载更多数据

```bash
# 下载英文书籍数据
python3 data_generation/rmt_data_preprocessing.py download \
  --dataset deepmind/pg19 \
  --output data/rmt_train_pg19_5k.jsonl \
  --num-samples 5000 \
  --min-length 8192 \
  --max-length 16384

# 合并数据集
cat data/rmt_train_wiki_zh_10k.jsonl data/rmt_train_pg19_5k.jsonl > data/rmt_train_combined.jsonl
```

---

## 📊 数据集推荐

### 🥇 Top 3 推荐 (中文优先)

| 排名 | 数据集 | 样本数 | 评分 | 用途 |
|------|--------|--------|------|------|
| **1** | 中文 Wikipedia | 100K+ | ⭐⭐⭐⭐⭐ | 主要训练数据 |
| **2** | Wudao (悟道) | 500K+ | ⭐⭐⭐⭐⭐ | 大规模训练 |
| **3** | SkyPile-150B | 海量 | ⭐⭐⭐⭐ | 补充数据 |

### 🥈 备选 (英文)

| 排名 | 数据集 | 样本数 | 评分 | 用途 |
|------|--------|--------|------|------|
| **1** | PG-19 | ~28K | ⭐⭐⭐⭐⭐ | 英文书籍数据 |
| **2** | The Pile | ~200K | ⭐⭐⭐⭐ | 多领域混合 |

### 🥉 快速验证

| 数据集 | 来源 | 样本数 | 用途 |
|--------|------|--------|------|
| 对话拼接 | `data/mag_train_*.jsonl` | ~10K | 概念验证 |

---

## 🔧 数据质量要求

| 要求 | 标准 | 原因 |
|------|------|------|
| 长度 | 8K-16K tokens | RMT 需要足够长度的文本 |
| 最小样本数 | 10K | 概念验证 |
| 推荐样本数 | 50K+ | 正式训练 |
| 语言 | 中文优先 (80-90%) | Qwen3-8B 中文能力相对弱 |
| 格式 | jsonl, 每行一条 | 易于处理 |

---

## ⚠️ 重要提醒

1. **不要使用短文本**: < 4K tokens 的数据不适合 RMT 训练
2. **保留段落边界**: 切割长文本时优先在 `\n\n` 处切分
3. **语义连贯性优先**: 优先使用真实长文本,避免随机拼接
4. **从小规模开始**: 先用 1K-10K 样本验证,再扩展到更大规模

---

## 📈 分阶段计划

| 阶段 | 时间 | 数据来源 | 样本数 | 目标 |
|------|------|----------|--------|------|
| **验证** | 1-2天 | 对话拼接 | ~10K | 验证 RMT 实现 |
| **小规模** | 1周 | 中文 Wikipedia | 10K | 初步训练 |
| **规模化** | 2-4周 | Wikipedia + Wudao + PG-19 | 50K-100K | 生产级模型 |

---

## 🔍 数据检查命令

```bash
# 检查数据长度分布
python3 << 'PYEOF'
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

data_path = sys.argv[1]
lengths = []

with open(data_path) as f:
    for line in f:
        item = json.loads(line)
        text = item['text']
        length = len(tokenizer.encode(text))
        lengths.append(length)

print(f"样本数: {len(lengths)}")
print(f"平均长度: {sum(lengths)/len(lengths):.0f} tokens")
print(f"最小长度: {min(lengths)} tokens")
print(f"最大长度: {max(lengths)} tokens")
print(f"符合8K-16K范围: {sum(8192 <= l <= 16384 for l in lengths) / len(lengths) * 100:.1f}%")

PYEOF

python3 << 'PYEOF'
import json
from transformers import AutoTokenizer
import subprocess

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

# 检查对话拼接数据
with open("data/rmt_train_mag_concat_session.jsonl") as f:
    item = json.loads(f.readline())
    text = item['text']
    length = len(tokenizer.encode(text))
    print(f"对话拼接数据示例:")
    print(f"  长度: {length} tokens")
    print(f"  文本预览: {text[:200]}...")
    print()

# 检查 Wikipedia 数据
try:
    with open("data/rmt_train_wiki_zh_10k.jsonl") as f:
        item = json.loads(f.readline())
        text = item['text']
        length = len(tokenizer.encode(text))
        print(f"Wikipedia 数据示例:")
        print(f"  长度: {length} tokens")
        print(f"  文本预览: {text[:200]}...")
        print()
except FileNotFoundError:
    print("Wikipedia 数据尚未下载,运行以下命令下载:")
    print("  export HF_ENDPOINT=https://hf-mirror.com")
    print("  python3 data_generation/rmt_data_preprocessing.py download \\")
    print("    --dataset wikimedia/wikipedia \\")
    print("    --config 20231101.zh \\")
    print("    --output data/rmt_train_wiki_zh_10k.jsonl \\")
    print("    --num-samples 10000")

PYEOF
```

---

## 📞 遇到问题?

### 问题 1: HuggingFace 下载失败
**解决方案**: 使用镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 2: 数据质量差
**解决方案**: 过滤短文本,手动检查样本
```python
# 过滤短文本
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

with open("input.jsonl") as f_in, open("output_filtered.jsonl", "w") as f_out:
    for line in f_in:
        item = json.loads(line)
        text = item['text']
        length = len(tokenizer.encode(text))
        if 8192 <= length <= 16384:
            f_out.write(line)
```

### 问题 3: 训练不收敛
**解决方案**:
- 从小规模(1K)开始验证
- 检查数据质量
- 调整学习率和 batch size

---

## 📚 更多信息

详细调研报告见: [RESEARCH_LITERATURE.md](./RESEARCH_LITERATURE.md) 中的"训练数据调研"章节。

数据预处理脚本: [data_generation/rmt_data_preprocessing.py](./data_generation/rmt_data_preprocessing.py)

---

**最后更新**: 2025-04-15
**维护者**: pighzliu
