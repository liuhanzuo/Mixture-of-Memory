#!/usr/bin/env python3
"""Scheduler: run BABILong long-context (8k/16k/32k) eval for Llama-3-8B-Instruct
using sample-level sharding across 8 H20 GPUs (2 procs/GPU = 16 slots).

Queue = 3 lengths x 3 tasks x 4 shards = 36 subprocesses.
Priority: 32k first (slowest), then 16k, then 8k.
Blocks until all shards complete.
"""
import subprocess
import time
import os
from pathlib import Path

ROOT = Path("/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory")
os.chdir(ROOT)
LOGDIR = ROOT / "logs"
LOGDIR.mkdir(exist_ok=True)

PY = ".venv/bin/python"
NUM_GPUS = 8
PROCS_PER_GPU = 2
NUM_SLOTS = NUM_GPUS * PROCS_PER_GPU  # 16
NUM_SHARDS = 4

LENGTHS = ["32k", "16k", "8k"]  # slowest first
TASKS = ["qa1", "qa2", "qa5"]

# Build queue: (length, task, shard)
QUEUE = []
for L in LENGTHS:
    for task in TASKS:
        for s in range(NUM_SHARDS):
            QUEUE.append((L, task, s))

BASE_ENV = dict(os.environ)
BASE_ENV["HF_DATASETS_OFFLINE"] = "1"
BASE_ENV["HF_HUB_OFFLINE"] = "1"
BASE_ENV["PYTHONPATH"] = "third_party/babilong-pkg"


def make_cmd(L, task, shard):
    return [
        PY, "scripts/eval_baseline_babilong.py",
        "--baseline", "plain_hf",
        "--model_path", "models/Meta-Llama-3-8B-Instruct",
        "--output_name", "Meta-Llama-3-8B-Instruct-full",
        "--results_folder", "babilong_results",
        "--tasks", task,
        "--lengths", L,
        "--num_shards", str(NUM_SHARDS),
        "--shard_index", str(shard),
        "--overwrite",
        "--use_chat_template", "--use_instruction",
        "--use_examples", "--use_post_prompt",
        "--limit", "100", "--max_new_tokens", "20",
    ]


def main():
    # slot -> running dict or None. slot index maps to gpu = slot % NUM_GPUS
    slots = [None] * NUM_SLOTS
    qi = 0
    failures = []
    t_start = time.time()
    print(f"[sched] queue size = {len(QUEUE)}, slots = {NUM_SLOTS}", flush=True)

    while qi < len(QUEUE) or any(s is not None for s in slots):
        # Fill free slots
        for slot in range(NUM_SLOTS):
            if slots[slot] is None and qi < len(QUEUE):
                L, task, shard = QUEUE[qi]
                gpu = slot % NUM_GPUS
                env = dict(BASE_ENV)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                logf = LOGDIR / f"instruct_full_{L}_{task}_s{shard}.log"
                fh = open(logf, "w")
                p = subprocess.Popen(make_cmd(L, task, shard), env=env,
                                     stdout=fh, stderr=subprocess.STDOUT)
                slots[slot] = {"p": p, "fh": fh, "job": (L, task, shard),
                               "gpu": gpu, "t": time.time()}
                print(f"[sched] start slot{slot} gpu{gpu} {L}/{task}/s{shard} pid={p.pid}", flush=True)
                qi += 1
                time.sleep(8)  # stagger model loads to avoid CPU/IO thundering herd

        # Poll for finished
        for slot in range(NUM_SLOTS):
            r = slots[slot]
            if r is None:
                continue
            rc = r["p"].poll()
            if rc is not None:
                r["fh"].close()
                L, task, shard = r["job"]
                dt = time.time() - r["t"]
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[sched] done  slot{slot} gpu{r['gpu']} {L}/{task}/s{shard} {status} ({dt:.0f}s)", flush=True)
                if rc != 0:
                    failures.append((L, task, shard, rc))
                slots[slot] = None
        time.sleep(15)

    print(f"[sched] ALL DONE in {time.time()-t_start:.0f}s. failures={failures}", flush=True)
    return failures


if __name__ == "__main__":
    main()
