"""BABILong checkpoint watchdog daemon for H-series training runs.

Runs on h20-3 (28.49.38.97). Polls b200-1..4 every POLL_INTERVAL seconds for
new step_N.pt checkpoints in the four H-series output_dirs. When new ckpt
appears, rsyncs to local h20-3 cluster (apdcephfs_zwfy6) and dispatches a
BABILong eval at lengths [1k, 2k, 4k, 8k] using one GPU per job.

Results land in:
    /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/status/babilong_realtime.jsonl

After each ckpt is done, the daemon also rsyncs the JSONL back to b200-1's
project share so the wzc1 path mirror has the same data.

Resilient design:
- On SSH failure → 3 retries with backoff, then mark exp as ssh_lost for 30 min.
- On individual eval crash → log to stderr, continue.
- State persists in .watchdog_state.json so restart picks up where we left off.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST = os.uname().nodename if hasattr(os, "uname") else ""

# We're meant to run only on h20-3
H20_PROJECT_ROOT = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
B200_PROJECT_ROOT = "/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"
LOCAL_CKPT_DIR = os.path.join(H20_PROJECT_ROOT, "watchdog_ckpts")
STATUS_DIR_LOCAL = os.path.join(H20_PROJECT_ROOT, "status")
STATE_FILE = os.path.join(H20_PROJECT_ROOT, ".watchdog_state.json")
RESULTS_LOCAL = os.path.join(STATUS_DIR_LOCAL, "babilong_realtime.jsonl")
RESULTS_REMOTE = os.path.join(B200_PROJECT_ROOT, "status/babilong_realtime.jsonl")
LOG_DIR_LOCAL = os.path.join(H20_PROJECT_ROOT, "logs")

POLL_INTERVAL = 300  # 5 min
SSH_RETRY_BACKOFF = [10, 30, 60]
SSH_LOST_COOLDOWN = 30 * 60  # 30 min
EVAL_LENGTHS = ["1k", "2k", "4k", "8k"]
EVAL_TASKS = ["qa1", "qa2", "qa5"]
EVAL_SAMPLES = 30
NUM_GPUS = 8
PARALLEL_LIMIT = 4  # at most 4 evals running concurrently

PASSWORD_B200 = os.path.join(B200_PROJECT_ROOT, "configs/password.txt")
PASSWORD_B200_LOCAL_FALLBACK = os.path.join(H20_PROJECT_ROOT, "configs/password.txt")
EVAL_SCRIPT = os.path.join(H20_PROJECT_ROOT, "scripts/eval_cross_attn_babilong.py")
PYTHON = "/opt/conda/envs/torch-base/bin/python"

# B200 nodes: each H-series exp lives on a different node
EXPERIMENTS = [
    {"name": "H9", "node": "28.89.17.143",
     "output_dir": "outputs/experiment_h9_contrastive"},
    {"name": "H10", "node": "28.89.17.144",
     "output_dir": "outputs/experiment_h10_aggressive_contrastive"},
    {"name": "H11_v2", "node": "28.89.17.85",
     "output_dir": "outputs/experiment_h11_v2_pure_contrastive"},
    {"name": "H12", "node": "28.89.19.134",
     "output_dir": "outputs/experiment_h12_arch_only"},
]

# We treat <name, step> as a unique evaluation unit. State maps to:
#   "evaluated":   {"H11_v2/500": {"lengths_done": ["1k","2k",...]}, ...}
#   "ssh_lost":    {"H11_v2": <ts_until_retry>, ...}
#   "ckpt_synced": {"H11_v2/500": "<local_path>", ...}
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_results_lock = threading.Lock()
_gpu_lock = threading.Lock()
_gpu_pool = list(range(NUM_GPUS))
_running_jobs = []  # list of dicts {gpu, popen, key, length}


def _now_iso():
    return _dt.datetime.utcnow().isoformat() + "Z"


def log(msg: str):
    line = f"[{_now_iso()}] {msg}"
    print(line, flush=True)


def password_path() -> str:
    if os.path.isfile(PASSWORD_B200):
        return PASSWORD_B200
    return PASSWORD_B200_LOCAL_FALLBACK


def load_state() -> dict:
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            log(f"WARN: state file corrupt, starting fresh: {e}")
    return {"evaluated": {}, "ssh_lost": {}, "ckpt_synced": {}}


def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def ssh_b200(node: str, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a single command on a b200 node via sshpass+ssh."""
    pw = password_path()
    full = [
        "sshpass", "-f", pw,
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "PreferredAuthentications=password",
        "-o", "ConnectTimeout=15",
        f"root@{node}", cmd,
    ]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "ssh timeout"
    except Exception as e:
        return 1, "", str(e)


def ssh_b200_retry(node: str, cmd: str, timeout: int = 30, retries: int = 3):
    """SSH with up to `retries` exponential-ish retries."""
    last = (1, "", "no attempt")
    for i in range(retries):
        rc, out, err = ssh_b200(node, cmd, timeout=timeout)
        if rc == 0:
            return rc, out, err
        last = (rc, out, err)
        sleep_s = SSH_RETRY_BACKOFF[min(i, len(SSH_RETRY_BACKOFF) - 1)]
        log(f"WARN ssh {node} failed (rc={rc}): {err.strip()[:200]} — retry in {sleep_s}s")
        time.sleep(sleep_s)
    return last


def list_remote_ckpts(exp: dict, state: dict) -> list[int]:
    """Return list of step numbers for which step_N.pt exists on b200 node."""
    name = exp["name"]
    cooldown_until = state["ssh_lost"].get(name, 0)
    if time.time() < cooldown_until:
        return []
    cmd = f"ls {os.path.join(B200_PROJECT_ROOT, exp['output_dir'])} 2>/dev/null"
    rc, out, err = ssh_b200_retry(exp["node"], cmd)
    if rc != 0:
        log(f"ERROR ssh {exp['node']} for {name} failed after retries: {err.strip()[:200]}")
        state["ssh_lost"][name] = time.time() + SSH_LOST_COOLDOWN
        return []
    # success → clear ssh_lost
    state["ssh_lost"].pop(name, None)
    steps = []
    for ln in out.splitlines():
        m = re.match(r"step_(\d+)\.pt$", ln.strip())
        if m:
            steps.append(int(m.group(1)))
    steps.sort()
    return steps


def rsync_ckpt(exp: dict, step: int) -> str | None:
    """Rsync remote step_N.pt to local cache. Returns local path or None.

    Skips if a peer rsync (or eval) is already in progress for this ckpt
    (detected by the presence of a `.step_N.pt.*` partial file in the dir).
    """
    name = exp["name"]
    remote = os.path.join(B200_PROJECT_ROOT, exp["output_dir"], f"step_{step}.pt")
    local_dir = os.path.join(LOCAL_CKPT_DIR, name)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    local = os.path.join(local_dir, f"step_{step}.pt")
    # Already fully synced
    if os.path.isfile(local) and os.path.getsize(local) > 5_000_000_000:
        return local
    # Detect concurrent rsync (rsync writes to .step_N.pt.<random>)
    partials = [p for p in os.listdir(local_dir)
                if p.startswith(f".step_{step}.pt.")]
    if partials:
        log(f"rsync {name}/step_{step}: peer rsync in progress ({partials[0]}), waiting")
        return None
    pw = password_path()
    rsh = ("ssh -o StrictHostKeyChecking=no "
           "-o PreferredAuthentications=password -o ConnectTimeout=15")
    cmd = ["sshpass", "-f", pw, "rsync", "-avz", "--partial", "--inplace",
           "-e", rsh,
           f"root@{exp['node']}:{remote}", local]
    log(f"rsync {name}/step_{step} from {exp['node']} ...")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            log(f"ERROR rsync {name}/step_{step} rc={r.returncode}: {r.stderr.strip()[:300]}")
            return None
        log(f"rsync {name}/step_{step} done in {time.time()-t0:.0f}s "
            f"({os.path.getsize(local)/1e9:.1f}GB)")
        return local
    except Exception as e:
        log(f"ERROR rsync {name}/step_{step} exception: {e}")
        return None


def push_results_to_b200():
    """Best-effort push of results JSONL back to b200-1 (project canonical share)."""
    if not os.path.isfile(RESULTS_LOCAL):
        return
    pw = password_path()
    rsh = ("ssh -o StrictHostKeyChecking=no "
           "-o PreferredAuthentications=password -o ConnectTimeout=15")
    target_dir = os.path.dirname(RESULTS_REMOTE)
    # ensure remote dir exists
    ssh_b200("28.89.17.143", f"mkdir -p {target_dir}")
    cmd = ["sshpass", "-f", pw, "rsync", "-az", "-e", rsh,
           RESULTS_LOCAL, f"root@28.89.17.143:{RESULTS_REMOTE}"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        log(f"WARN push_results_to_b200: {e}")


def claim_gpu() -> int | None:
    with _gpu_lock:
        if _gpu_pool:
            return _gpu_pool.pop(0)
    return None


def release_gpu(g: int):
    with _gpu_lock:
        _gpu_pool.append(g)
        _gpu_pool.sort()


def launch_eval(exp_name: str, step: int, length: str, ckpt_local: str, gpu: int) -> subprocess.Popen:
    """Launch a single eval as a subprocess pinned to one GPU."""
    Path(LOG_DIR_LOCAL).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(LOG_DIR_LOCAL, f"babilong_eval_{exp_name}_step{step}_{length}_g{gpu}.log")
    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--ckpt_path", ckpt_local,
        "--exp_name", exp_name,
        "--step", str(step),
        "--length", length,
        "--num_samples", str(EVAL_SAMPLES),
        "--tasks", *EVAL_TASKS,
        "--output_jsonl", RESULTS_LOCAL,
        "--device", "cuda:0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{H20_PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    # Point HF cache at the synced dir; force offline so evals don't need network.
    hf_home = os.path.join(H20_PROJECT_ROOT, ".hf_cache")
    env["HF_HOME"] = hf_home
    env["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    env["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    log(f"LAUNCH gpu={gpu} {exp_name}/step_{step}/{length} -> {log_file}")
    f = open(log_file, "ab")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def schedule_pending_evals(state: dict):
    """Build the full set of (exp, step, length) work items not yet evaluated.

    We yield them lazily and assign to GPUs as they free up.
    """
    pending = []
    for exp_name, info in state.get("ckpt_synced", {}).items():
        ckpt_local = info if isinstance(info, str) else info.get("path")
        # exp_name e.g. "H11_v2/500"
        ev = state["evaluated"].get(exp_name, {"lengths_done": []})
        for L in EVAL_LENGTHS:
            if L not in ev["lengths_done"]:
                exp, step_s = exp_name.split("/")
                pending.append((exp, int(step_s), L, ckpt_local))
    return pending


def reap_running_jobs(state: dict):
    """Move finished jobs out of _running_jobs and update state."""
    still = []
    for j in _running_jobs:
        rc = j["popen"].poll()
        if rc is None:
            still.append(j)
            continue
        # done
        release_gpu(j["gpu"])
        if rc == 0:
            ev = state["evaluated"].setdefault(j["key"], {"lengths_done": []})
            if j["length"] not in ev["lengths_done"]:
                ev["lengths_done"].append(j["length"])
            log(f"DONE {j['key']}/{j['length']} (gpu={j['gpu']}, rc=0)")
        else:
            log(f"FAIL {j['key']}/{j['length']} rc={rc} — see logs/")
        save_state(state)
        push_results_to_b200()
    _running_jobs[:] = still


def fill_gpu_slots(state: dict):
    """Greedily pull pending evals and launch them on free GPUs (up to PARALLEL_LIMIT)."""
    while len(_running_jobs) < PARALLEL_LIMIT:
        pending = schedule_pending_evals(state)
        # filter out those already running
        running_keys = {(j["key"], j["length"]) for j in _running_jobs}
        pending = [p for p in pending if (f"{p[0]}/{p[1]}", p[2]) not in running_keys]
        if not pending:
            return
        exp, step, length, ckpt_local = pending[0]
        gpu = claim_gpu()
        if gpu is None:
            return
        if not os.path.isfile(ckpt_local):
            log(f"WARN ckpt missing for {exp}/step_{step}: {ckpt_local} — skipping")
            release_gpu(gpu)
            # remove from synced state to avoid loop
            state["ckpt_synced"].pop(f"{exp}/{step}", None)
            save_state(state)
            continue
        popen = launch_eval(exp, step, length, ckpt_local, gpu)
        _running_jobs.append({
            "gpu": gpu, "popen": popen,
            "key": f"{exp}/{step}", "length": length,
        })


def poll_and_sync(state: dict):
    """For each experiment, list new ckpts and rsync them locally."""
    for exp in EXPERIMENTS:
        steps = list_remote_ckpts(exp, state)
        for step in steps:
            key = f"{exp['name']}/{step}"
            if key in state["ckpt_synced"] and os.path.isfile(state["ckpt_synced"][key]):
                continue
            local = rsync_ckpt(exp, step)
            if local is not None:
                state["ckpt_synced"][key] = local
                state["evaluated"].setdefault(key, {"lengths_done": []})
                save_state(state)


def initial_sweep(state: dict):
    """First-shot: sync any existing ckpts and queue evals before entering the loop."""
    log("=== Initial sweep: rsyncing existing ckpts and queuing evals ===")
    poll_and_sync(state)
    log(f"Initial sweep: {len(state['ckpt_synced'])} ckpts known")
    fill_gpu_slots(state)


def main_loop():
    state = load_state()
    Path(LOCAL_CKPT_DIR).mkdir(parents=True, exist_ok=True)
    Path(STATUS_DIR_LOCAL).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR_LOCAL).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_LOCAL).touch(exist_ok=True)

    log(f"watchdog start. results -> {RESULTS_LOCAL}, state -> {STATE_FILE}")
    initial_sweep(state)

    last_poll = 0.0
    while True:
        try:
            reap_running_jobs(state)
            fill_gpu_slots(state)
            now = time.time()
            if now - last_poll > POLL_INTERVAL:
                poll_and_sync(state)
                last_poll = now
                fill_gpu_slots(state)
            time.sleep(15)
        except KeyboardInterrupt:
            log("Interrupted, terminating jobs ...")
            for j in _running_jobs:
                try:
                    j["popen"].terminate()
                except Exception:
                    pass
            break
        except Exception as e:
            log(f"main loop EXCEPTION: {e}\n{traceback.format_exc()}")
            time.sleep(30)


if __name__ == "__main__":
    main_loop()
