#!/usr/bin/env python3
"""LPT scheduler for EXP-W1 (top_k32) BABILong eval on 8x H20 (local diskA).

Builds 2 ckpt x 7 length work items, shards heavy 16k/32k for parallelism,
LPT bin-packs onto 8 GPUs, emits one runner script per GPU.
Run with --launch to setsid nohup launch all 8 GPU queues detached.
"""
import argparse
import os
import subprocess

PROJECT_ROOT = "/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
PYBIN = f"{PROJECT_ROOT}/.venv/bin/python"
MODEL = "models/Meta-Llama-3-8B"
ADAPTER_CFG = "outputs/expW1_topk32/adapter_config.json"

CKPTS = {
    "expW1_topk32_step500": "outputs/expW1_topk32/mem_space_adapter_step000500.pt",
    "expW1_topk32_step1000": "outputs/expW1_topk32/mem_space_adapter.pt",
}

# rough relative cost per (qa1+qa2+qa5, limit100) cell
LEN_COST = {"0k": 1.0, "1k": 1.2, "2k": 1.6, "4k": 2.6, "8k": 4.6, "16k": 8.6, "32k": 16.6}
# how many sample shards to split each length into (more parallelism for heavy)
LEN_SHARDS = {"0k": 1, "1k": 1, "2k": 1, "4k": 1, "8k": 1, "16k": 2, "32k": 4}

N_GPUS = 8


def build_items():
    items = []
    for run, ckpt in CKPTS.items():
        for length, cost in LEN_COST.items():
            n = LEN_SHARDS[length]
            for si in range(n):
                items.append({
                    "run": run, "ckpt": ckpt, "length": length,
                    "num_shards": n, "shard_index": si,
                    "cost": cost / n,
                })
    return items


def lpt_pack(items, n_bins):
    bins = [[] for _ in range(n_bins)]
    load = [0.0] * n_bins
    for it in sorted(items, key=lambda x: -x["cost"]):
        b = min(range(n_bins), key=lambda i: load[i])
        bins[b].append(it)
        load[b] += it["cost"]
    return bins, load


def cmd_for(it, gpu):
    run = it["run"]
    length = it["length"]
    n = it["num_shards"]
    si = it["shard_index"]
    out_name = f"{run}_{length}" if n == 1 else f"{run}_{length}_shard{si}of{n}"
    shard_args = "" if n == 1 else f" --num_shards {n} --shard_index {si}"
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} {PYBIN} scripts/run_babilong_mem_space.py "
        f"--model_path {MODEL} --checkpoint {it['ckpt']} --adapter_config {ADAPTER_CFG} "
        f"--results_folder babilong_results/{run} --output_name {out_name} "
        f"--tasks qa1 qa2 qa5 --lengths {length} --limit 100 --chunk_size 512 "
        f"--max_new_tokens 20 --dtype bfloat16 --attn_impl sdpa{shard_args}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()

    items = build_items()
    bins, load = lpt_pack(items, N_GPUS)

    os.makedirs(f"{PROJECT_ROOT}/logs/expW1_sched", exist_ok=True)
    os.makedirs(f"{PROJECT_ROOT}/scripts/expW1_gpu", exist_ok=True)

    env_prefix = (
        "export http_proxy=http://hy-proxy.woa.com:3128; "
        "export https_proxy=http://hy-proxy.woa.com:3128; "
        "export all_proxy=http://hy-proxy.woa.com:3128; "
        f"export HF_HOME={PROJECT_ROOT}/.hf_cache; "
        "export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1"
    )

    def label(it):
        sfx = "" if it["num_shards"] == 1 else f"s{it['shard_index']}"
        return f"{it['run'].split('_')[-1]}/{it['length']}{sfx}"

    print("=== LPT assignment ===")
    for g in range(N_GPUS):
        print(f"GPU{g} load={load[g]:.1f}: " +
              ", ".join(label(it) for it in bins[g]))
        path = f"{PROJECT_ROOT}/scripts/expW1_gpu/gpu{g}.sh"
        with open(path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {PROJECT_ROOT}\n")
            f.write(env_prefix + "\n")
            for it in bins[g]:
                logname = f"logs/expW1_sched/gpu{g}_{it['run']}_{it['length']}"
                if it["num_shards"] > 1:
                    logname += f"_s{it['shard_index']}"
                f.write(f"echo '[START] {it['run']} {it['length']} "
                        f"shard{it['shard_index']}of{it['num_shards']}' \n")
                f.write(cmd_for(it, g) + f" > {logname}.log 2>&1\n")
            f.write(f"echo '[GPU{g} DONE]'\n")
        os.chmod(path, 0o755)

    if args.launch:
        for g in range(N_GPUS):
            script = f"{PROJECT_ROOT}/scripts/expW1_gpu/gpu{g}.sh"
            masterlog = f"{PROJECT_ROOT}/logs/expW1_sched/gpu{g}_master.log"
            subprocess.run(
                f"setsid nohup bash {script} </dev/null > {masterlog} 2>&1 &",
                shell=True, executable="/bin/bash",
            )
        print("\n=== launched 8 GPU queues (detached) ===")
    else:
        print("\n(dry run; pass --launch to start)")


if __name__ == "__main__":
    main()
