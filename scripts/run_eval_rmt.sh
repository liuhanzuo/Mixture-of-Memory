#!/bin/bash
# Launch script for RMT evaluation
#
# Usage:
#   bash scripts/run_eval_rmt.sh --checkpoint_dir outputs/rmt_v1/final --data_path data/rmt_train_wikitext.jsonl
#   bash scripts/run_eval_rmt.sh --checkpoint_dir outputs/rmt_v1/final --data_path data/rmt_train_wikitext.jsonl --eval_type ppl
#

set -e

# Default values
CHECKPOINT_DIR=""
DATA_PATH=""
OUTPUT_DIR="eval_results"
EVAL_TYPE="all"
MAX_DOCS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --max_docs)
            MAX_DOCS="--max_docs $2"
            shift 2
            ;;
        --num_memory_tokens)
            NUM_MEMORY_TOKENS="$2"
            shift 2
            ;;
        --segment_length)
            SEGMENT_LENGTH="$2"
            shift 2
            ;;
        --nih_num_trials)
            NIH_NUM_TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: --checkpoint_dir is required"
    echo "Usage: bash scripts/run_eval_rmt.sh --checkpoint_dir <path> --data_path <path>"
    exit 1
fi

if [ -z "$DATA_PATH" ]; then
    echo "Error: --data_path is required"
    echo "Usage: bash scripts/run_eval_rmt.sh --checkpoint_dir <path> --data_path <path>"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "RMT Evaluation Configuration"
echo "=========================================="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Eval type: $EVAL_TYPE"
if [ -n "$MAX_DOCS" ]; then
    echo "Max docs: $MAX_DOCS"
fi
if [ -n "$NUM_MEMORY_TOKENS" ]; then
    echo "Num memory tokens: $NUM_MEMORY_TOKENS"
fi
if [ -n "$SEGMENT_LENGTH" ]; then
    echo "Segment length: $SEGMENT_LENGTH"
fi
if [ -n "$NIH_NUM_TRIALS" ]; then
    echo "NiH trials: $NIH_NUM_TRIALS"
fi
echo "=========================================="

# Build command
CMD="python scripts/eval_rmt.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --eval_type $EVAL_TYPE"

if [ -n "$MAX_DOCS" ]; then
    CMD="$CMD $MAX_DOCS"
fi

if [ -n "$NUM_MEMORY_TOKENS" ]; then
    CMD="$CMD --num_memory_tokens $NUM_MEMORY_TOKENS"
fi

if [ -n "$SEGMENT_LENGTH" ]; then
    CMD="$CMD --segment_length $SEGMENT_LENGTH"
fi

if [ -n "$NIH_NUM_TRIALS" ]; then
    CMD="$CMD --nih_num_trials $NIH_NUM_TRIALS"
fi

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run evaluation
echo "Running evaluation..."
echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
