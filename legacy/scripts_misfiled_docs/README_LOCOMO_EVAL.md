# LoCoMo Evaluation for RMT

This document explains how to use the LoCoMo benchmark evaluation script for RMT models.

## Background

LoCoMo (Long-term Conversational Memory) is a benchmark from ACL 2024 that evaluates very long-term conversational memory of LLMs. The dataset contains 10 extended conversations with 1986 QA pairs across 5 categories:

1. **Multi-hop Reasoning** (Category 1): Requires information from multiple parts of the conversation
2. **Single-hop / Needle-in-Haystack** (Category 2): Requires finding specific information
3. **Temporal Reasoning** (Category 3): Requires understanding when events happened
4. **Open-domain / User Profiling** (Category 4): Questions about speakers' preferences and traits
5. **Adversarial** (Category 5): Information not in the conversation (model should say "not available")

## Quick Start

### Basic Evaluation

Run evaluation on a trained RMT checkpoint:

```bash
python scripts/eval_rmt_locomo.py \
    --checkpoint_dir outputs/rmt_v4_20260416_104930 \
    --locomo_data locomo/data/locomo10.json \
    --output_dir eval_results/locomo
```

### Evaluate Specific Categories

```bash
# Evaluate only single-hop (needle-in-haystack) and long-document QA
python scripts/eval_rmt_locomo.py \
    --checkpoint_dir outputs/rmt_v4_20260416_104930 \
    --categories 2,4 \
    --output_dir eval_results/locomo
```

### Quick Test (Debug Mode)

```bash
# Run with just a few QA pairs per conversation
python scripts/eval_rmt_locomo.py \
    --checkpoint_dir outputs/rmt_v4_20260416_104930 \
    --max_qa_per_conv 5 \
    --output_dir eval_results/locomo_test
```

## Arguments

### Required

- `--checkpoint_dir`: Path to RMT checkpoint directory (must contain `rmt_config.json`)
- `--locomo_data`: Path to LoCoMo data file (default: `locomo/data/locomo10.json`)

### Optional

#### RMT Configuration
- `--num_memory_tokens`: Number of memory tokens (default: 16, from config)
- `--segment_length`: Segment length in tokens (default: 1024, from config)
- `--max_segments`: Maximum number of segments (default: 6, from config)
- `--bottleneck_dim`: Memory bottleneck dimension (default: 64, from config)
- `--extractor_version`: Memory extractor version (2 or 3, default: 3, from config)

#### Generation
- `--max_new_tokens`: Maximum tokens for answer generation (default: 50)
- `--temperature`: Sampling temperature (default: 0.0, greedy decoding)
- `--do_sample`: Enable sampling (disabled by default)

#### Evaluation
- `--categories`: Comma-separated category numbers (e.g., `1,2,3`)
- `--max_qa_per_conv`: Max QA pairs per conversation (for debugging)
- `--use_bertscore`: Compute BERTScore in addition to F1/EM (slower)

#### Output
- `--output_dir`: Output directory for results (default: `eval_results/locomo`)

## Output Format

The script generates two JSON files:

1. **Full results**: `{output_dir}/locomo_results_{timestamp}.json`
   - Contains per-sample predictions and scores
   - Format: `{by_category: {...}, by_sample: [...], overall: {...}}`

2. **Summary**: `{output_dir}/locomo_summary_{timestamp}.json`
   - Contains only aggregate statistics
   - Easier to read and compare across runs

### Example Output

```json
{
  "overall": {
    "num_samples": 1986,
    "avg_f1": 0.6523,
    "avg_em": 0.5210,
    "avg_bert": 0.7234
  },
  "by_category": {
    "1": {
      "num_samples": 282,
      "avg_f1": 0.5832,
      "avg_em": 0.4511
    },
    "2": {
      "num_samples": 321,
      "avg_f1": 0.7123,
      "avg_em": 0.6012
    },
    ...
  }
}
```

## Evaluation Metrics

The script uses LoCoMo's official evaluation metrics:

- **F1 Score**: Token-level F1 with stemming (primary metric)
- **Exact Match (EM)**: Word-level exact match
- **BERTScore** (optional): Semantic similarity score

Note: Categories 1-4 use F1 scoring. Category 5 (adversarial) checks if the model correctly responds that information is not available.

## How It Works

1. **Context Building**: Concatenates all conversation sessions into a single long text
2. **Segmentation**: Splits context into segments (default: 1024 tokens each, up to 6 segments)
3. **Memory Processing**: Processes each segment through RMT, extracting compressed memory
4. **Answer Generation**: Uses the final memory to generate answers for each QA pair
5. **Metric Computation**: Compares generated answers with ground truth using F1/EM/BERTScore

## Validation

Run the validation script to test the setup:

```bash
python scripts/validate_locomo_eval.py
```

This tests:
- Data loading
- Context building
- Evaluation metrics
- Tokenizer loading

## Data Format

LoCoMo data format (already present at `locomo/data/locomo10.json`):

```json
[
  {
    "sample_id": "conversation_0",
    "conversation": {
      "speaker_a": "Alice",
      "speaker_b": "Bob",
      "session_1": [{"speaker": "Alice", "text": "...", "dia_id": "D1:1"}, ...],
      "session_1_date_time": "2023-05-01 10:00",
      "session_2": [...],
      ...
    },
    "qa": [
      {
        "question": "When did Alice go to the event?",
        "answer": "May 7, 2023",
        "category": 2,
        "evidence": ["D1:3"]
      },
      ...
    ],
    ...
  }
]
```

## Dependencies

- `bert-score`: BERTScore computation
- `nltk`: Text processing and stemming
- `transformers`: Model and tokenizer
- `torch`: PyTorch

These should be installed via:
```bash
pip install bert-score nltk
```

## Troubleshooting

### "No module named 'bert_score'"
Install bert-score: `pip install bert-score`

### "No module named 'nltk'"
Install nltk: `pip install nltk`

### CUDA OOM
Reduce `--max_segments` or `--segment_length`, or use a smaller model.

### Missing checkpoint
Ensure the checkpoint directory contains:
- `rmt_config.json`: RMT configuration
- `rmt_memory.pt`: Trained RMT memory weights (optional, will use random init if missing)
- Model files (from base model or HuggingFace)

## Notes

- Context is truncated to `max_segments * segment_length` (default: 6144 tokens)
- Memory flows bidirectionally within segments and recurrently across segments
- Answers are generated using greedy decoding by default (set `--temperature > 0` and `--do_sample` for sampling)
- BERTScore computation is slower but provides additional semantic comparison
