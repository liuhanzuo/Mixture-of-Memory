#!/usr/bin/env python3
"""Scheduler: offline BABILong eval for 5 mem_space ckpts (F2c512 step500/final,
ladder s1c256 step500/final, ladder s2c512 step500) across 8 H20 GPUs.

Parallel unit = (ckpt x length). qa1/qa2/qa5 run inside one process per unit.
5 ckpts x 7 lengths = 35 units. 8 GPUs x 2 procs/GPU = 16 slots (~40GB/proc).

Output layout (nested, for score_nested_babilong.py):
  babilong_results/<run>/<run>_<length>/<task>_<length>_<prompt>.csv

Blocks until all units finish.
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
MODEL = "models/Meta-Llama-3-8B"
NUM_GPUS = 8
PROCS_PER_GPU = 2
NUM_SLOTS = NUM_GPUS * PROCS_PER_GPU  # 16

LENGTHS = ["32k", "16k", "8k", "4k", "2k", "1k", "0k"]  # slowest first

# run_name -> (checkpoint, adapter_config, chunk_size)
CKPTS = {
    "f2_c512_step500": (
        "outputs/f2_longdoc_chunk512/mem_space_adapter_step000500.pt",
        "outputs/f2_longdoc_chunk512/adapter_config.json", 512),
    "f2_c512_step5000": (
        "outputs/f2_longdoc_chunk512/mem_space_adapter.pt",
        "outputs/f2_longdoc_chunk512/adapter_config.json", 512),
    "ladder_s1c256_step500": (
        "outputs/progressive_chunk_local_v3_topk_ladder/stage1_c256/mem_space_adapter_step000500.pt",
        "outputs/progressive_chunk_local_v3_topk_ladder/stage1_c256/adapter_config.json", 256),
    "ladder_s1c256_step5000": (
        "outputs/progressive_chunk_local_v3_topk_ladder/stage1_c256/mem_space_adapter.pt",
        "outputs/progressive_chunk_local_v3_topk_ladder/stage1_c256/adapter_config.json", 256),
    "ladder_s2c512_step500": (
        "outputs/progressive_chunk_local_v3_topk_ladder/stage2_c512/mem_space_adapter_step000500.pt",
        "outputs/progressive_chunk_local_v3_topk_ladder/stage2_c512/adapter_config.json", 512),
}

# Build queue: (run_name, length). Longest lengths first for load balancing.
QUEUE = []
for L in LENGTHS:
    for run in CKPTS:
        QUEUE.append((run, L))

BASE_ENV = dict(os.environ)
BASE_ENV["HF_DATASETS_OFFLINE"] = "1"
BASE_ENV["HF_HUB_OFFLINE"] = "1"
BASE_ENV["PYTHONPATH"] = f"{ROOT}:third_party/babilong-pkg"


def make_cmd(run, L):
    ckpt, cfg, chunk = CKPTS[run]
    return [
        PY, "scripts/run_babilong_mem_space.py",
        "--model_path", MODEL,
        "--checkpoint", ckpt,
        "--adapter_config", cfg,
        "--results_folder", "babilong_results",
        "--output_name", f"{run}/{run}_{L}",
        "--chunk_size", str(chunk),
        "--tasks", "qa1", "qa2", "qa5",
        "--lengths", L,
        "--limit", "100",
    ]


def main():
    slots = [None] * NUM_SLOTS
    qi = 0
    failures = []
    t_start = time.time()
    print(f"[sched] queue size = {len(QUEUE)}, slots = {NUM_SLOTS}", flush=True)

    while qi < len(QUEUE) or any(s is not None for s in slots):
        for slot in range(NUM_SLOTS):
            if slots[slot] is None and qi < len(QUEUE):
                run, L = QUEUE[qi]
                gpu = slot % NUM_GPUS
                env = dict(BASE_ENV)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                logf = LOGDIR / f"eval_{run}_{L}.log"
                fh = open(logf, "w")
                p = subprocess.Popen(make_cmd(run, L), env=env,
                                     stdout=fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
                slots[slot] = {"p": p, "fh": fh, "job": (run, L),
                               "gpu": gpu, "t": time.time()}
                print(f"[sched] start slot{slot} gpu{gpu} {run}/{L} pid={p.pid}", flush=True)
                qi += 1
                time.sleep(8)  # stagger model loads

        for slot in range(NUM_SLOTS):
            r = slots[slot]
            if r is None:
                continue
            rc = r["p"].poll()
            if rc is not None:
                r["fh"].close()
                run, L = r["job"]
                dt = time.time() - r["t"]
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[sched] done  slot{slot} gpu{r['gpu']} {run}/{L} {status} ({dt:.0f}s)", flush=True)
                if rc != 0:
                    failures.append((run, L, rc))
                slots[slot] = None
        time.sleep(15)

    print(f"[sched] ALL DONE in {time.time()-t_start:.0f}s. failures={failures}", flush=True)


if __name__ == "__main__":
    main()
