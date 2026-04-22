#!/usr/bin/env python3
"""SSH to remote nodes, verify GPU idle, launch experiments."""
import pexpect
import json
import sys
import time

PASSWORD = "4fS6h9nHdbICfm6,"
SSH_OPTS = "-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
PROJECT_DIR = "/root/Mixture-of-Memory"
TORCHRUN = "/opt/conda/envs/torch-base/bin/torchrun"

NODES = {
    "28.89.17.143": {
        "name": "b200-1",
        "output_dir": "outputs/sparse_memory_concat",
        "base_model": "/root/Mixture-of-Memory/models/Qwen--Qwen3-8b",
        "extra_args": "--memory_slots 128 --top_k 8 --sliding_window 256 --ema_alpha 0.1 --gradient_checkpointing --batch_size 1 --grad_accumulation_steps 8",
    },
    "28.89.17.144": {
        "name": "b200-2",
        "output_dir": "outputs/sparse_l01_seg512_mem32",
        "base_model": "/root/Mixture-of-Memory/models/Qwen--Qwen3-8b",
        "extra_args": "--use_l1 --num_mem_tokens 32 --l1_num_tokens 8 --segment_length 512 --max_segments 8 --warmup_steps 50 --lr 3e-5 --recon_loss_coef 0.2 --use_importance_routing --vary_n_segments --gradient_checkpointing",
    },
    "28.89.17.85": {
        "name": "b200-3",
        "output_dir": "outputs/sparse_l01_mem128_recon05",
        "base_model": "/root/Mixture-of-Memory/models/Qwen--Qwen3-8b",
        "extra_args": "--use_l1 --num_mem_tokens 128 --l1_num_tokens 32 --recon_loss_coef 0.5 --lr 5e-5 --use_importance_routing --vary_n_segments --gradient_checkpointing",
    },
    "28.89.19.134": {
        "name": "b200-4",
        "output_dir": "outputs/sparse_memory_concat_l1",
        "base_model": "/root/Mixture-of-Memory/models/Qwen--Qwen3-8b",
        "extra_args": "--use_l1 --num_mem_tokens 64 --l1_num_tokens 16 --bptt_depth 4 --recon_loss_coef 0.3 --lr 5e-5 --warmup_steps 60 --use_importance_routing --vary_n_segments --gradient_checkpointing --batch_size 1 --grad_accumulation_steps 8",
    },
}

CONFIG_PATH = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/configs/remote_experiments.json"

def ssh_run(ip, cmd, timeout=30):
    """Run a command via SSH and return output."""
    child = pexpect.spawn(f"ssh {SSH_OPTS} root@{ip}", timeout=timeout, encoding="utf-8")
    child.expect("[Pp]assword:", timeout=15)
    child.sendline(PASSWORD)
    child.expect("#", timeout=15)
    child.sendline(cmd)
    child.expect("#", timeout=timeout)
    output = child.before
    child.sendline("exit")
    child.close()
    return output

def check_gpu_idle(ip):
    """Check if GPU is idle. Returns (idle, info)."""
    out = ssh_run(ip, "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=15)
    lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
    # Filter out the command echo line
    data_lines = [l for l in lines if l[0].isdigit() or (l and l.split(',')[0].strip().isdigit())]
    if not data_lines:
        # No GPU data lines found — check if output contains 0 MiB
        if '0 MiB' in out:
            return True, "All GPUs idle (0 MiB)"
        return False, f"Could not parse nvidia-smi output: {out[:200]}"
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            mem_str = parts[1].replace(" MiB", "").strip()
            try:
                if int(mem_str) > 500:
                    return False, f"GPU {parts[0]} has {mem_str} MiB used"
            except ValueError:
                pass
    return True, f"All GPUs idle: {len(data_lines)} GPUs checked"

def launch_experiment(ip, node_cfg):
    """Launch the training experiment on the remote node."""
    cmd = (
        f"cd {PROJECT_DIR} && nohup {TORCHRUN} "
        f"--nnodes=1 --nproc_per_node=8 "
        f"scripts/train_sparse_memory.py "
        f"--base_model {node_cfg['base_model']} "
        f"--output_dir {node_cfg['output_dir']} "
        f"--data_path data/rmt_train_mixed.jsonl "
        f"--gradient_checkpointing "
        f"{node_cfg['extra_args']} "
        f"> {node_cfg['output_dir']}/train.log 2>&1 & echo $!"
    )
    out = ssh_run(ip, cmd, timeout=30)
    # Extract PID
    lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
    pid = None
    for line in reversed(lines):
        if line.isdigit():
            pid = int(line)
            break
    return pid, out

def verify_running(ip, node_cfg, pid, wait=10):
    """Wait and verify the process is running and using GPU."""
    time.sleep(wait)
    # Check process
    out = ssh_run(ip, f"ps -p {pid} -o pid,cmd --no-headers 2>/dev/null || echo DEAD", timeout=15)
    if "DEAD" in out or not out.strip():
        return False, f"Process {pid} is dead"
    # Check GPU usage
    gpu_out = ssh_run(ip, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", timeout=15)
    return True, f"Process alive, GPU state: {gpu_out.strip()[:200]}"

def update_config(ip, status, pid=None):
    """Update the config file status for a node."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg["nodes"][ip]["status"] = status
    cfg["nodes"][ip]["pid"] = pid
    cfg["nodes"][ip]["actual_status"] = f"launched via torchrun at {time.strftime('%Y-%m-%dT%H:%M:%S')}"
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

results = {}
for ip, node_cfg in NODES.items():
    print(f"\n{'='*60}")
    print(f"Node {ip} ({node_cfg['name']})")
    print(f"{'='*60}")
    
    # Step 1: Check GPU idle
    print(f"  Checking GPU state...")
    idle, info = check_gpu_idle(ip)
    print(f"  GPU: {info}")
    
    if not idle:
        print(f"  SKIP: GPU not idle!")
        results[ip] = {"status": "skipped", "reason": info}
        update_config(ip, "skipped_gpu_busy")
        continue
    
    # Step 2: Create output dir and launch
    print(f"  Launching experiment...")
    ssh_run(ip, f"mkdir -p {PROJECT_DIR}/{node_cfg['output_dir']}", timeout=10)
    pid, out = launch_experiment(ip, node_cfg)
    print(f"  PID: {pid}")
    print(f"  Output: {out.strip()[:300]}")
    
    if not pid:
        print(f"  FAILED: Could not determine PID")
        results[ip] = {"status": "failed", "reason": "no PID"}
        update_config(ip, "launch_failed")
        continue
    
    # Step 3: Verify after short wait
    print(f"  Verifying (waiting 15s)...")
    ok, info = verify_running(ip, node_cfg, pid, wait=15)
    print(f"  Verify: {info}")
    
    if ok:
        results[ip] = {"status": "running", "pid": pid}
        update_config(ip, "running", pid)
        print(f"  ✓ SUCCESS: {node_cfg['name']} running (PID {pid})")
    else:
        results[ip] = {"status": "verify_failed", "pid": pid, "reason": info}
        update_config(ip, "running_unverified", pid)
        print(f"  ⚠ UNVERIFIED: PID {pid} but verification unclear")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for ip, r in results.items():
    print(f"  {ip}: {r['status']} {r.get('pid', '')} {r.get('reason', '')}")
