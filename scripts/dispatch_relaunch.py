#!/usr/bin/env python3
"""Relaunch sparse memory training on all 4 remote L20A nodes after GQA fix."""

import pexpect
import time
import json
from datetime import datetime

NODES = {
    "28.89.17.143": {"name": "b200-1", "extra_args": "", "output_dir": "outputs/sparse_memory_pg19_l0"},
    "28.89.17.144": {"name": "b200-2", "extra_args": "--use_l1", "output_dir": "outputs/sparse_memory_pg19_l1"},
    "28.89.17.85":  {"name": "b200-3", "extra_args": "--use_l2", "output_dir": "outputs/sparse_memory_pg19_l2"},
    "28.89.19.134": {"name": "b200-4", "extra_args": "--use_l1 --use_l2", "output_dir": "outputs/sparse_memory_pg19_l3"},
}

SSH_OPTS = "-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
PASSWORD = "4fS6h9nHdbICfm6,"
REMOTE_WORKDIR = "/root/Mixture-of-Memory"
LOCAL_PROJECT = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
BASE_MODEL = "/root/Mixture-of-Memory/models/Qwen--Qwen3-8b"
PYTHON = "/opt/conda/envs/torch-base/bin/python3"
TORCHRUN = "/opt/conda/envs/torch-base/bin/torchrun"
MASTER_PORT = "29510"

# Rsync excludes
RSYNC_EXCLUDES = [
    "--exclude=.git", "--exclude=__pycache__", "--exclude=outputs",
    "--exclude=models", "--exclude=.venv", "--exclude=*.egg-info",
    "--exclude=.ipynb_checkpoints", "--exclude=logs/*.log",
]

def ssh_run(ip, cmd, timeout=60):
    """Run command on remote node, return output."""
    full_cmd = f"ssh {SSH_OPTS} root@{ip} '{cmd}'"
    child = pexpect.spawn(full_cmd, timeout=timeout, encoding="utf-8")
    idx = child.expect(["password:", pexpect.EOF, pexpect.TIMEOUT], timeout=20)
    if idx == 0:
        child.sendline(PASSWORD)
        child.expect(pexpect.EOF, timeout=timeout)
    output = (child.before or "").strip()
    child.close()
    return output

def rsync_code(ip):
    """Rsync project code to remote node."""
    print(f"[{ip}] Rsyncing code...")
    excludes = " ".join(RSYNC_EXCLUDES)
    cmd = (
        f"sshpass -p '{PASSWORD}' rsync -avz --delete "
        f"-e 'ssh {SSH_OPTS}' {excludes} "
        f"{LOCAL_PROJECT}/ root@{ip}:{REMOTE_WORKDIR}/"
    )
    child = pexpect.spawn(cmd, timeout=120, encoding="utf-8")
    child.expect(pexpect.EOF, timeout=120)
    output = (child.before or "").strip()
    child.close()
    # Just check last few lines for errors
    lines = output.split("\n")
    for line in lines[-5:]:
        if "error" in line.lower():
            print(f"[{ip}] Rsync error: {line}")
    print(f"[{ip}] Rsync done.")

def kill_existing(ip):
    """Kill any existing training processes."""
    print(f"[{ip}] Cleaning up existing training processes...")
    # Kill torchrun and python training processes
    ssh_run(ip, "pkill -f 'train_rmt_v10\\|train_sparse_memory' 2>/dev/null; sleep 2; echo cleaned", timeout=20)
    # Also kill specific PID if it's still around
    ssh_run(ip, "kill -9 1110849 2>/dev/null; echo done", timeout=10)

def launch_training(ip, node_info):
    """Launch training on remote node. Returns PID."""
    extra_args = node_info["extra_args"]
    output_dir = node_info["output_dir"]
    
    remote_cmd = (
        f"cd {REMOTE_WORKDIR} && "
        f"mkdir -p logs {output_dir} && "
        f"nohup {TORCHRUN} --nproc_per_node=6 --master_port={MASTER_PORT} "
        f"scripts/train_rmt_v10.py "
        f"--data data/rmt_train_mixed.jsonl "
        f"--output_dir {output_dir} "
        f"--base_model {BASE_MODEL} "
        f"--num_mem_tokens 16 --segment_length 1024 --max_segments 4 "
        f"--vary_n_segments --bptt_depth 2 "
        f"--recon_loss_coef 0.1 --use_importance_routing "
        f"--full_finetune --num_epochs 5 "
        f"--lr 1e-5 --rmt_lr 1e-4 "
        f"--batch_size 1 --grad_accumulation_steps 8 "
        f"--warmup_steps 30 --log_every 10 --save_every 100 --eval_every 100 "
        f"--seed 42 --ddp "
        f"{extra_args} "
        f"> logs/train_{node_info['name']}.log 2>&1 & "
        f"echo $!"
    )
    print(f"[{ip}] Launching training...")
    output = ssh_run(ip, remote_cmd, timeout=30)
    lines = output.strip().split("\n")
    pid = lines[-1].strip() if lines else ""
    print(f"[{ip}] Launched with PID={pid}, extra_args='{extra_args}'")
    return pid

def verify(ip, pid):
    """Verify training is running and GPUs are active."""
    time.sleep(10)
    
    # Check process
    ps_out = ssh_run(ip, f"ps -p {pid} -o pid,comm --no-headers 2>/dev/null", timeout=15)
    if not ps_out.strip():
        print(f"[{ip}] ❌ PID {pid} not found!")
        log_tail = ssh_run(ip, f"tail -20 {REMOTE_WORKDIR}/logs/train_*.log 2>/dev/null || echo 'no logs'", timeout=15)
        print(f"[{ip}] Last log:\n{log_tail}")
        return False
    
    # Check GPUs
    gpu_out = ssh_run(ip, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader", timeout=15)
    print(f"[{ip}] ✅ Process alive (PID {pid})")
    print(f"[{ip}] GPU status:\n{gpu_out}")
    
    # Check if GPUs are actually being used
    for line in gpu_out.strip().split("\n"):
        parts = line.split(",")
        if len(parts) >= 3:
            mem_used = int(parts[2].strip().replace(" MiB", ""))
            if mem_used > 100:
                return True
    print(f"[{ip}] ⚠ No GPU memory usage detected")
    return False

def main():
    print(f"=== Relaunch dispatch at {datetime.now().isoformat()} ===\n")

    results = {}

    # Step 1: Rsync code to all nodes
    print("=== Step 1: Rsync code to all nodes ===\n")
    for ip in NODES:
        try:
            rsync_code(ip)
        except Exception as e:
            print(f"[{ip}] ❌ Rsync failed: {e}")

    # Step 2: Kill existing processes
    print("\n=== Step 2: Kill existing processes ===\n")
    for ip in NODES:
        try:
            kill_existing(ip)
        except Exception as e:
            print(f"[{ip}] Kill error: {e}")

    # Step 3: Launch training
    print("\n=== Step 3: Launch training on all nodes ===\n")
    for ip, info in NODES.items():
        try:
            pid = launch_training(ip, info)
            results[ip] = {"pid": pid, "status": "launching"}
        except Exception as e:
            print(f"[{ip}] ❌ Launch failed: {e}")
            results[ip] = {"pid": None, "status": f"error: {e}"}

    # Step 4: Verify
    print("\n=== Step 4: Verify training started ===\n")
    for ip, info in results.items():
        if info.get("pid"):
            try:
                ok = verify(ip, info["pid"])
                info["status"] = "running" if ok else "verify_failed"
            except Exception as e:
                info["status"] = f"verify_error: {e}"
                print(f"[{ip}] ❌ Verify error: {e}")

    # Summary
    print("\n=== FINAL SUMMARY ===")
    for ip, info in results.items():
        node = NODES[ip]["name"]
        print(f"  {ip} ({node}): pid={info['pid']}, status={info['status']}")

    # Save results
    out_file = f"dispatch_relaunch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print PIDs for config update
    print("\n=== PIDs for config update ===")
    print(json.dumps({ip: results.get(ip, {}).get("pid") for ip in NODES}, indent=2))

if __name__ == "__main__":
    main()
