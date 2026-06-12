#!/usr/bin/env python
"""Ad-hoc scheduler for diskB offline BABILong eval of D6 arm C (nullsink_off).
2 ckpts (step500, step5000) x 7 lengths = 14 units. 8 GPUs x 1 proc = 8 slots.
Each unit = (output_name, ckpt, adapter_config, chunk_size, length) running tasks qa1 qa2 qa5, limit 100.
Layout for score_nested_babilong: results_folder=babilong_results/<output_name>, output_name=<output_name>_<length>.
"""
import os, subprocess, time

PROJ = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
PY = PROJ + "/.venv/bin/python"
MODEL = "models/Meta-Llama-3-8B"
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
TASKS = ["qa1", "qa2", "qa5"]
NGPU = 8
PROCS_PER_GPU = 1
SLOTS = NGPU * PROCS_PER_GPU

# (output_name, output_dir, ckpt_filename, chunk_size)
RUNS = [
    ("d6nullsinkoff_step500",  "outputs/d6_nullsink_off", "mem_space_adapter_step000500.pt", 512),
    ("d6nullsinkoff_step5000", "outputs/d6_nullsink_off", "mem_space_adapter.pt",            512),
]

# build queue: (output_name, ckpt, adapter_config, chunk, length)
queue = []
for (oname, odir, ckpt, chunk) in RUNS:
    for L in LENGTHS:
        queue.append({
            "oname": oname,
            "ckpt": f"{odir}/{ckpt}",
            "cfg": f"{odir}/adapter_config.json",
            "chunk": chunk,
            "length": L,
        })

print(f"[sched] total units: {len(queue)}, slots: {SLOTS}", flush=True)

ENV = dict(os.environ)
ENV.update({
    "http_proxy": "http://hy-proxy.woa.com:3128",
    "https_proxy": "http://hy-proxy.woa.com:3128",
    "all_proxy": "http://hy-proxy.woa.com:3128",
    "no_proxy": "localhost,127.0.0.1,.oa.com,.woa.com,.local",
    "HF_HOME": PROJ + "/.hf_home",
})

os.chdir(PROJ)
os.makedirs("logs", exist_ok=True)

slot_gpu = [s % NGPU for s in range(SLOTS)]
running = {}
qi = 0

def launch(slot, unit):
    gpu = slot_gpu[slot]
    rf = f"babilong_results/{unit['oname']}"
    on = f"{unit['oname']}_{unit['length']}"
    logf = f"logs/eval_{unit['oname']}_{unit['length']}.log"
    cmd = [PY, "scripts/run_babilong_mem_space.py",
           "--model_path", MODEL,
           "--checkpoint", unit["ckpt"],
           "--adapter_config", unit["cfg"],
           "--results_folder", rf,
           "--output_name", on,
           "--chunk_size", str(unit["chunk"]),
           "--tasks", *TASKS,
           "--lengths", unit["length"],
           "--limit", "100"]
    e = dict(ENV)
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lf = open(logf, "ab")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=e)
    print(f"[sched] LAUNCH slot{slot} gpu{gpu} -> {on} (pid {p.pid})", flush=True)
    return p

for slot in range(SLOTS):
    if qi < len(queue):
        running[slot] = (launch(slot, queue[qi]), queue[qi])
        qi += 1
        time.sleep(3)  # stagger to avoid HF download race

done = 0
while running:
    time.sleep(15)
    for slot in list(running.keys()):
        p, unit = running[slot]
        if p.poll() is not None:
            done += 1
            print(f"[sched] DONE slot{slot} {unit['oname']}_{unit['length']} rc={p.returncode} ({done}/{len(queue)})", flush=True)
            if qi < len(queue):
                running[slot] = (launch(slot, queue[qi]), queue[qi])
                qi += 1
            else:
                del running[slot]

print(f"[sched] ALL DONE. completed {done} units.", flush=True)
