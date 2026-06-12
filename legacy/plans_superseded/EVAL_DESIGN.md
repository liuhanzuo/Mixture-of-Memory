# RMT Evaluation Design Document

## Overview

This document describes the design and implementation of `eval_rmt.py`, a comprehensive evaluation script for the Recurrent Memory Transformer (RMT) model trained on Qwen3-8B.

## Design Goals

1. **Comprehensive Evaluation**: Cover language modeling, retrieval, and memory analysis
2. **Modular Design**: Support running individual evaluation types
3. **Production Ready**: Robust error handling, logging, and JSON output
4. **Reproducible**: Consistent with training configuration

## Evaluation Types

### 1. Perplexity (PPL) Evaluation

**Purpose**: Measure language modeling quality on held-out documents.

**Implementation**:
- Process documents in segments (same as training)
- Pass memory forward between segments
- Compute cross-entropy loss on each segment
- Calculate PPL as `exp(mean_loss)`

**Key Considerations**:
- Uses same segment processing as training for consistency
- Memory flows across all segments (recurrent)
- Pads documents to fixed length (max_segments × segment_length)
- Filters out padding tokens using label masking (-100)

**Metrics**:
- Perplexity: Primary metric (lower is better)
- Avg Loss: Raw cross-entropy
- Total Tokens: For transparency

**Design Decision**: Why PPL as primary metric?
- Direct measure of prediction quality
- Comparable across models and baselines
- Standard in language modeling literature
- Sensitive to memory effectiveness (better memory = better prediction)

### 2. Needle-in-Haystack (NiH) Test

**Purpose**: Test long-context retrieval ability across segments.

**Implementation**:
- Generate haystack documents from Wikitext-103 test set
- Insert needle at specified positions (10%, 30%, 50%, 70%, 90%)
- Process through all segments to build memory
- Ask question about needle at the end
- Generate answer and check accuracy

**Test Variations**:
- **Needle Positions**: Test retrieval from different locations
- **Document Lengths**: 4K, 8K, 12K tokens (2-6 segments)
- **Multiple Trials**: 5 trials per configuration

**Metrics**:
- Overall Accuracy: Percentage of correct retrievals
- Accuracy by Position: Performance across document positions
- Accuracy by Length: Performance across document lengths

**Design Decision**: Why these variations?
- Position variation tests if RMT can retrieve from anywhere
- Length variation tests if performance degrades with more segments
- Multiple trials provide statistical significance

**Generation Approach**:
- Simple greedy generation (not sampling) for consistency
- Uses memory from final segment for answering
- Limits generation to 20 tokens (sufficient for short answers)

### 3. Memory Utilization Analysis

**Purpose**: Understand how memory tokens encode information.

**Implementation**:
- Extract memory tokens after each segment
- Compute cosine similarity between consecutive segments
- Analyze if memory changes or collapses to similar representations

**Metrics**:
- Average Memory Similarity: Mean cosine similarity across segments
- Num Segments Analyzed: Sample size for reliability
- Num Memory Tokens: Configuration context

**Interpretation**:
- **Low similarity (< 0.5)**: Memory encodes unique information per segment
- **Medium similarity (0.5-0.8)**: Some information overlap, but distinct
- **High similarity (> 0.8)**: Memory may not be encoding segment-specific info

**Design Decision**: Why cosine similarity?
- Scale-invariant (works for high-dimensional embeddings)
- Standard metric for semantic similarity
- Easy to interpret (0 = orthogonal, 1 = identical)

## Architecture Design

### Class Structure

```
EvalConfig (dataclass)
├── Configuration parameters
└── Type hints for IDE support

RMTEvalDataset (Dataset)
├── Handles data loading and tokenization
└── Pads documents to fixed length

RMTEvaluator (class)
├── evaluate_perplexity()
├── needle_in_haystack_test()
└── analyze_memory_utilization()
```

### Key Design Decisions

#### 1. Fixed-Length Documents
**Rationale**: Training uses fixed-length documents for static DDP graphs. Evaluation should match.

**Trade-off**: Some padding wasted, but consistency with training.

#### 2. Segment-Wise Forward Pass
**Rationale**: Same as training — process each segment separately to avoid OOM.

**Benefit**: Consistent behavior between train and eval.

#### 3. Memory Detaching
**Rationale**: In evaluation mode, don't need gradients. Detach saves memory.

**Location**: After each segment's backward in training, after extraction in eval.

#### 4. JSON Output Format
**Rationale**: Machine-readable for automated parsing and comparison.

**Structure**:
```json
{
  "checkpoint_dir": "...",
  "eval_type": "all",
  "timestamp": "2026-04-15T12:00:00",
  "perplexity": {...},
  "needle_in_haystack": {...},
  "memory_utilization": {...}
}
```

## Implementation Details

### Model Loading

1. **Configuration Loading**: Reads `rmt_config.json` from checkpoint dir
2. **Base Model**: Loads Qwen3-8B with `torch_dtype=bfloat16`
3. **RMT Memory**: Initializes and loads `rmt_memory.pt` weights
4. **RMTModel Wrapper**: Wraps base model and memory for segment processing

### Data Processing

1. **Tokenization**: Uses trained tokenizer (from checkpoint)
2. **Padding**: Pads to `max_segments × segment_length`
3. **Segmentation**: Automatic based on segment_length
4. **Label Masking**: Padding tokens get -100 (ignored in loss)

### Evaluation Flow

#### Perplexity
```
For each document:
    For each segment:
        1. Get memory (initial or from previous segment)
        2. Forward with memory
        3. Compute loss
        4. Extract new memory
        5. Accumulate loss and token count
Calculate PPL = exp(total_loss / total_tokens)
```

#### Needle-in-Haystack
```
For each configuration (length, position):
    For each trial:
        1. Build document with needle at position
        2. Process through all segments (build memory)
        3. Ask question using final memory
        4. Generate answer
        5. Check if answer contains expected result
Aggregate accuracy across all tests
```

#### Memory Utilization
```
For first 10 batches:
    For each segment:
        1. Forward with memory
        2. Extract new memory
        3. Store memory
Compute cosine similarity between consecutive segments
```

## Launch Script

The `run_eval_rmt.sh` script provides:

1. **Argument Parsing**: Converts CLI args to Python args
2. **Configuration Display**: Shows what will run
3. **Virtual Environment**: Auto-activates if `.venv/` exists
4. **Flexibility**: Optional parameters for quick testing

**Usage Examples**:

```bash
# Full evaluation (all metrics)
bash scripts/run_eval_rmt.sh \
    --checkpoint_dir outputs/rmt_v1/final \
    --data_path data/rmt_train_wikitext.jsonl

# Quick test (only perplexity, 10 docs)
bash scripts/run_eval_rmt.sh \
    --checkpoint_dir outputs/rmt_v1/final \
    --data_path data/rmt_train_wikitext.jsonl \
    --eval_type ppl \
    --max_docs 10

# NiH only (50 trials)
bash scripts/run_eval_rmt.sh \
    --checkpoint_dir outputs/rmt_v1/final \
    --data_path data/rmt_train_wikitext.jsonl \
    --eval_type nih \
    --nih_num_trials 50
```

## Future Enhancements

### Potential Additions

1. **Baseline Comparison**: Evaluate base Qwen3-8B (without RMT) for comparison
2. **More NiH Variations**: Vary needle type, quantity, and complexity
3. **Attention Analysis**: Visualize attention weights to memory tokens
4. **Ablation Study**: Test with different num_memory_tokens values
5. **Streaming Generation**: Test continuous generation beyond segment boundary

### Performance Optimizations

1. **Multi-GPU Support**: Distribute evaluation across GPUs
2. **Caching**: Cache tokenized documents for repeated evals
3. **Progressive Sampling**: Early stopping if convergence detected

## Limitations

1. **Fixed Document Length**: Padding may waste computation
2. **Simple NiH Generation**: Greedy generation, not sampling-based
3. **Memory Analysis Sample Size**: Only analyzes 10 batches
4. **Single Baseline**: Doesn't compare against other memory methods

## Conclusion

The evaluation script provides a comprehensive, production-ready framework for assessing RMT models. The three evaluation types (PPL, NiH, Memory) cover different aspects of model performance:

- **PPL**: Overall language modeling quality
- **NiH**: Long-context retrieval ability
- **Memory Analysis**: Understanding of how memory is used

The modular design allows running individual evaluations for faster iteration during development, while the `all` mode provides complete assessment for final reporting.

The JSON output format and launch script make it easy to integrate into automated pipelines and compare across checkpoints or configurations.
