#!/bin/bash

# Selective Context Evaluation Launcher
# Test zero-cost prompt compression approach after sparse memory failure

set -e

# Configuration
EXPERIMENT_NAME="selective_context_evaluation"
REMOTE_NODES=("28.89.17.143" "28.89.17.144" "28.89.17.85" "28.89.17.134")
NUM_NODES=${#REMOTE_NODES[@]}
NODE_INDEX=0

# SSH options
SSH_OPTS="-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 36000"
PASSWORD="4fS6h9nHdbICfm6,"

echo "Launching Selective Context evaluation on $NUM_NODES remote nodes..."
echo "Experiment: $EXPERIMENT_NAME"

# Test each node sequentially
for NODE_IP in "${REMOTE_NODES[@]}"; do
    NODE_NAME="node$NODE_INDEX"
    OUTPUT_DIR="/root/Mixture-of-Memory/outputs/$EXPERIMENT_NAME/$NODE_NAME"
    LOG_FILE="/root/Mixture-of-Memory/logs/eval_selective_context_$NODE_NAME.log"
    
    echo "Launching on $NODE_IP ($NODE_NAME)..."
    
    # SSH and launch evaluation
    sshpass -p "$PASSWORD" ssh $SSH_OPTS root@$NODE_IP << EOF
        cd /root/Mixture-of-Memory
        
        # Check if model exists, download if needed
        if [ ! -f "models/Llama--Llama2-7b/config.json" ]; then
            echo "Downloading Llama-2-7b model..."
            # Check if virtual environment exists, create if needed
            if [ ! -d ".venv" ]; then
                echo "Creating virtual environment..."
                python3 -m venv .venv
                source .venv/bin/activate
                pip install torch transformers
            fi
            bash scripts/download_model.sh Llama--Llama2-7b
        fi
        
        # Activate environment
        source .venv/bin/activate
        
        # Create output directory
        mkdir -p "$OUTPUT_DIR"
        
        # Run Selective Context evaluation
        echo "Starting evaluation on $(date)" > "$LOG_FILE"
        python scripts/eval_selective_context.py \\
            --model_name_or_path models/Llama--Llama2-7b \\
            --data_path /tmp/pg19_small.jsonl \\
            --compression_ratios 0.3 0.5 0.7 1.0 \\
            --context_sizes 256 512 1024 \\
            --output_dir "$OUTPUT_DIR" \\
            --log_file "$LOG_FILE" \\
            >> "$LOG_FILE" 2>&1
        
        echo "Evaluation completed on $(date)" >> "$LOG_FILE"
        
        # Get final PPL results
        python -c "
import json
import os
import glob

# Find latest results file
results_files = glob.glob('$OUTPUT_DIR/*_results.json')
if results_files:
    with open(results_files[-1], 'r') as f:
        results = json.load(f)
    print('Final PPL Results:')
    for compression, metrics in results.items():
        print(f'  {compression}: PPL={metrics.get(\"ppl\", \"N/A\")}, Compression={metrics.get(\"compression\", \"N/A\")}%')
else:
    print('No results files found')
" >> "$LOG_FILE"
EOF
    
    echo "✅ Node $NODE_NAME ($NODE_IP) evaluation launched"
    NODE_INDEX=$((NODE_INDEX + 1))
    sleep 10  # Small delay between nodes
done

echo ""
echo "🎉 Selective Context evaluation launched on all $NUM_NODES nodes!"
echo "Monitor progress with:"
echo "  tail -f /root/Mixture-of-Memory/logs/eval_selective_context_*.log"

# Update remote experiments configuration
cat > configs/remote_experiments.json << EOF
{
    "node0": {
        "ip": "28.89.17.143",
        "name": "selective_node0",
        "experiment": "selective_context_evaluation",
        "status": "running",
        "pid": null,
        "log_path": "/root/Mixture-of-Memory/logs/eval_selective_context_node0.log",
        "output_dir": "/root/Mixture-of-Memory/outputs/selective_context_evaluation/node0",
        "eval_ppl": null,
        "notes": "Selective Context evaluation - zero-cost compression baseline",
        "last_verified": "$(date -Iseconds)"
    },
    "node1": {
        "ip": "28.89.17.144", 
        "name": "selective_node1",
        "experiment": "selective_context_evaluation",
        "status": "running",
        "pid": null,
        "log_path": "/root/Mixture-of-Memory/logs/eval_selective_context_node1.log",
        "output_dir": "/root/Mixture-of-Memory/outputs/selective_context_evaluation/node1",
        "eval_ppl": null,
        "notes": "Selective Context evaluation - zero-cost compression baseline",
        "last_verified": "$(date -Iseconds)"
    },
    "node2": {
        "ip": "28.89.17.85",
        "name": "selective_node2", 
        "experiment": "selective_context_evaluation",
        "status": "running",
        "pid": null,
        "log_path": "/root/Mixture-of-Memory/logs/eval_selective_context_node2.log",
        "output_dir": "/root/Mixture-of-Memory/outputs/selective_context_evaluation/node2",
        "eval_ppl": null,
        "notes": "Selective Context evaluation - zero-cost compression baseline",
        "last_verified": "$(date -Iseconds)"
    },
    "node3": {
        "ip": "28.89.17.134",
        "name": "selective_node3",
        "experiment": "selective_context_evaluation", 
        "status": "running",
        "pid": null,
        "log_path": "/root/Mixture-of-Memory/logs/eval_selective_context_node3.log",
        "output_dir": "/root/Mixture-of-Memory/outputs/selective_context_evaluation/node3",
        "eval_ppl": null,
        "notes": "Selective Context evaluation - zero-cost compression baseline",
        "last_verified": "$(date -Iseconds)"
    }
}
EOF

echo "✅ Remote experiments configuration updated"