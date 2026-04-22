#!/bin/bash

# NIH-Hard Evaluation Launcher
# This script launches the hard needle-in-haystack benchmark that should expose base model limitations

set -e

# Configuration
MODEL_PATH="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/models/Qwen--Qwen2.5-1.5B-Instruct"
CONTEXT_LENGTH=65536  # 64K context
NUM_NEEDLES=15
NUM_TRIALS=3
GPU_ID=0

echo "=== NIH-Hard Benchmark Launcher ==="
echo "Context length: $CONTEXT_LENGTH"
echo "Number of needles: $NUM_NEEDLES" 
echo "Number of trials: $NUM_TRIALS"
echo "GPU ID: $GPU_ID"
echo "Model: $MODEL_PATH"
echo "==============================="

# Create output directory
OUTPUT_DIR="outputs/nih_hard_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set environment
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "Starting NIH-Hard evaluation..."
echo "Output directory: $OUTPUT_DIR"

# Run base model evaluation first
echo "=== Base Model Evaluation ==="
python scripts/eval_nih_hard.py \
    --model_path "$MODEL_PATH" \
    --context_length $CONTEXT_LENGTH \
    --num_needles $NUM_NEEDLES \
    --num_trials $NUM_TRIALS \
    --gpu_id $GPU_ID \
    --output_dir "$OUTPUT_DIR" \
    --use_sparse_memory false

echo "Base model evaluation completed."

# Run sparse memory evaluation if implemented
echo "=== Sparse Memory Evaluation ==="
python scripts/eval_nih_hard.py \
    --model_path "$MODEL_PATH" \
    --context_length $CONTEXT_LENGTH \
    --num_needles $NUM_NEEDLES \
    --num_trials $NUM_TRIALS \
    --gpu_id $GPU_ID \
    --output_dir "$OUTPUT_DIR" \
    --use_sparse_memory true

echo "Sparse memory evaluation completed."

# Generate summary
echo "=== Generating Summary ==="
python -c "
import json
import os

output_dir = '$OUTPUT_DIR'
files = [f for f in os.listdir(output_dir) if f.endswith('.json')]

summary = {
    'timestamp': '$(date)',
    'context_length': $CONTEXT_LENGTH,
    'num_needles': $NUM_NEEDLES,
    'num_trials': $NUM_TRIALS,
    'model': '$MODEL_PATH',
    'results': []
}

for file in files:
    with open(os.path.join(output_dir, file), 'r') as f:
        result = json.load(f)
        summary['results'].append({
            'file': file,
            'accuracy': result['accuracy'],
            'correct_needles': result['correct_needles'],
            'total_needles': result['total_needles'],
            'use_sparse_memory': result['use_sparse_memory']
        })

# Print summary
print('=== NIH-HARD EVALUATION SUMMARY ===')
for res in summary['results']:
    print(f\"File: {res['file']}\")
    print(f\"Type: {'Sparse Memory' if res['use_sparse_memory'] else 'Base Model'}\")
    print(f\"Accuracy: {res['accuracy']:.2%} ({res['correct_needles']}/{res['total_needles']})\")
    print('---')

# Save summary
with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f\"Summary saved to: {os.path.join(output_dir, 'summary.json')}\")
"

echo "NIH-Hard evaluation completed successfully!"
echo "Results in: $OUTPUT_DIR"