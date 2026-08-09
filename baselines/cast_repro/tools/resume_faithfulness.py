#!/usr/bin/env python3
"""EMPIRICAL proof that save/resume under ``--parallel zero2`` is faithful.

This does not assert faithfulness; it MEASURES it.  Two runs of the real
``train_cast_llama.main()`` code path, same seed, same data, same world size:

    A (control)  N+M steps straight through
    B (resumed)  N steps -> save ckpt -> fresh process -> resume -> M more steps

then diff the per-step ``loss_trace.jsonl`` over the overlapping steps and report
the max absolute and relative difference.  A faithful resume gives ~0; a warm
restart (Adam moments re-initialised) shows a visible discontinuity in the first
resumed step because the update direction loses its momentum history.

The comparison is deliberately made against a *third* arm as well:

    C (sabotaged) same as B, but the optimizer state is dropped on load

C is the positive control.  Without it, "B matches A" is uninformative -- it could
mean the resume works, or it could mean this toy is too insensitive to tell the
difference.  C proves the test can actually see a warm restart.  This addresses
the exact failure the project has already suffered: three arms silently
warm-restarted and nobody noticed because nothing was ever compared.

Usage (tiny stub model, CPU/1-GPU, seconds -- for logic):
    python tools/resume_faithfulness.py --tiny

Usage (real LLaMA2-7B, 8 GPUs, zero2 -- the one that counts):
    python tools/resume_faithfulness.py --real --n 3 --m 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def read_trace(p: Path) -> dict:
    out = {}
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["step"]] = r
    return out


def compare(a: dict, b: dict, lo: int, hi: int, key: str = "loss") -> dict:
    """Max abs/rel difference of ``key`` over steps [lo, hi)."""
    steps = [s for s in range(lo, hi) if s in a and s in b]
    if not steps:
        return {"n": 0}
    diffs = [(s, a[s][key], b[s][key], abs(a[s][key] - b[s][key])) for s in steps]
    worst = max(diffs, key=lambda d: d[3])
    rel = [d[3] / max(abs(d[1]), 1e-12) for d in diffs]
    return {
        "n": len(steps),
        "steps": [steps[0], steps[-1]],
        "max_abs": worst[3],
        "max_abs_at_step": worst[0],
        "max_abs_pair": (worst[1], worst[2]),
        "max_rel": max(rel),
        "first_step_abs": diffs[0][3],
        "first_step_pair": (diffs[0][1], diffs[0][2]),
        "n_bit_identical": sum(1 for d in diffs if d[3] == 0.0),
    }


# ---------------------------------------------------------------------------
def run_tiny(n: int, m: int, workdir: Path) -> int:
    """Same experiment with the tiny stub model, in-process, no GPU needed.

    Uses the real trainer main() with `transformers` stubbed, exactly like
    tools/integration_tiny.py, so the code path under test IS the production one.
    """
    import types

    import numpy as np
    import torch

    sys.path.insert(0, str(HERE / "tools"))
    from integration_tiny import TinyLlama  # noqa: E402

    V = 512
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(0)
    rs.randint(0, V, size=400_000).astype(np.uint16).tofile(data / "train.bin")

    base = [
        "--project-root", str(workdir),
        "--model", ".",
        "--data", "data",
        # explicit: the stub corpus has no metadata.json, and --data-dtype auto
        # now (correctly) refuses to guess the token width
        "--data-dtype", "uint16",
        "--max-steps", str(n + m),
        "--mask-period", "3",
        "--global-batch", "2",
        "--micro-batch", "1",
        "--seq-len", "64",
        "--lr", "1e-3",
        "--l1-decay", "1e-3",
        "--log-every", "1",
        "--diag-every", "0",
        "--smoke",
        "--lr-schedule", "cosine",
        "--warmup", "1",
        "--min-lr", "1e-5",
    ]

    def launch(out: str, extra: list, sabotage: bool = False):
        # Fresh interpreter state for the model each time; stub transformers.
        stub = types.ModuleType("transformers")
        stub.LlamaForCausalLM = TinyLlama
        sys.modules["transformers"] = stub
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tcl_" + out, str(HERE / "cast" / "train_cast_llama.py")
        )
        tcl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tcl)

        if sabotage:
            # POSITIVE CONTROL: make the resume drop the optimizer state, i.e.
            # reproduce the silent warm restart, and confirm this harness sees it.
            import cast.checkpoint as ckmod

            real_load = ckmod.load_training_state

            def sabotaged(path, **kw):
                meta = real_load(path, **kw)
                # Wipe the moments AFTER a legitimate load, mimicking exactly what
                # torch does silently when the param groups don't line up.
                opt = kw["opt"]
                inner = getattr(opt, "optim", opt)
                for st in inner.state.values():
                    if "exp_avg" in st:
                        st["exp_avg"].zero_()
                        st["exp_avg_sq"].zero_()
                return meta

            tcl.load_training_state = sabotaged
            # the verifier would (correctly) catch the wipe, so bypass it for C
            tcl.assert_optimizer_state_restored = lambda *a, **k: {
                "params": 0, "with_state": 0
            }

        argv = ["train_cast_llama.py", "--out", out, *base, *extra]
        old = sys.argv
        sys.argv = argv
        try:
            tcl.main()
        finally:
            sys.argv = old

    print(f"[A] control: {n + m} steps straight through", flush=True)
    launch("outA", ["--save-every", "0"])

    # NOTE --stop-after, NOT a lowered --max-steps: max_steps feeds
    # alpha_t = t/T, so changing it to stop early would rescale the decay
    # schedule and make A and B different experiments. The resume guard catches
    # exactly that mistake (it fired on the first draft of this harness).
    print(f"[B] resumed: {n + 1} steps, save, resume, rest of {n + m}", flush=True)
    launch("outB", ["--save-every", str(n), "--keep-last", "5", "--stop-after", str(n + 1)])
    ck = workdir / "outB" / f"ckpt_step{n}"
    assert ck.exists(), f"no checkpoint at {ck}: {list((workdir / 'outB').iterdir())}"
    launch("outB", ["--save-every", "0", "--resume", str(ck)])

    print("[C] positive control: same, but optimizer state wiped on load", flush=True)
    (workdir / "outC").mkdir(exist_ok=True)
    shutil.copytree(ck, workdir / "outC" / f"ckpt_step{n}")
    # C's trace starts empty: only the resumed steps matter for the comparison.
    launch(
        "outC",
        ["--save-every", "0", "--resume", str(workdir / "outC" / f"ckpt_step{n}")],
        sabotage=True,
    )

    A = read_trace(workdir / "outA" / "loss_trace.jsonl")
    B = read_trace(workdir / "outB" / "loss_trace.jsonl")
    C = read_trace(workdir / "outC" / "loss_trace.jsonl")

    pre = compare(A, B, 0, n + 1)
    post = compare(A, B, n + 1, n + m + 1)
    sab = compare(A, C, n + 1, n + m + 1)

    print("\n=== RESULT (tiny stub model) ===")
    print(f"A(control) steps: {sorted(A)}")
    print(f"B(resumed) steps: {sorted(B)}")
    print(f"pre-resume  overlap  A vs B: {json.dumps(pre, default=str)}")
    print(f"post-resume steps    A vs B: {json.dumps(post, default=str)}")
    print(f"post-resume steps    A vs C (warm-restart control): {json.dumps(sab, default=str)}")
    ok = post["n"] > 0 and post["max_abs"] < 1e-9
    sees = sab["n"] > 0 and sab["max_abs"] > 100 * max(post["max_abs"], 1e-12)
    print(f"\nfaithful (B==A to <1e-9): {ok}")
    print(f"harness can detect a warm restart (C deviates >100x more): {sees}")
    return 0 if (ok and sees) else 1


# ---------------------------------------------------------------------------
def run_real(n: int, m: int, args) -> int:
    """The real thing: LLaMA2-7B, 8 GPUs, --parallel zero2, torchrun subprocesses."""
    root = Path(args.project_root)
    py = args.python
    torchrun = str(Path(py).parent / "torchrun")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    common = [
        "--project-root", str(root),
        "--data", args.data,
        "--data-dtype", "auto",
        "--parallel", "zero2",
        "--lr", "2e-5", "--l1-decay", "4e-7",
        "--global-batch", "256", "--seq-len", "4096",
        "--mask-period", "10", "--scale-groups", "2",
        "--eta", "0.3333333333333333", "--kl-temperature", "1.0",
        "--lr-schedule", "constant",
        "--micro-batch", "1", "--gradient-checkpointing",
        "--log-every", "1", "--diag-every", "0",
        # max-steps is a CRITICAL resume arg (it sets alpha_t = t/T), so BOTH
        # arms must declare the same horizon even though A runs it to the end and
        # B stops early.
        "--max-steps", str(n + m),
    ]

    def launch(tag: str, out: str, extra: list) -> int:
        cmd = [torchrun, "--nproc_per_node", "8", "cast/train_cast_llama.py",
               "--out", out, *common, *extra]
        log = logs / f"resume_faith_{tag}.log"
        print(f"[{tag}] {' '.join(cmd)}\n      log -> {log}", flush=True)
        with log.open("w") as fh:
            return subprocess.call(cmd, cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT)

    outA, outB = args.out_prefix + "_A", args.out_prefix + "_B"
    for o in (outA, outB):
        d = root / o
        if d.exists():
            shutil.rmtree(d)

    # B first: it is the one that can fail, and it is cheaper (n+1 steps).
    # --stop-after, not a lowered --max-steps: max_steps sets alpha_t = t/T, so
    # shortening it would make B a different experiment than A (and the resume
    # guard would correctly refuse).
    rc = launch("B1", outB, ["--save-every", str(n), "--keep-last", "5",
                             "--stop-after", str(n + 1)])
    if rc != 0:
        print(f"B phase 1 failed rc={rc}")
        return rc
    ck = root / outB / f"ckpt_step{n}"
    if not ck.exists():
        print(f"no checkpoint written at {ck}")
        return 2
    print(f"[B] checkpoint {ck} = {ck.stat().st_size / 2**30:.1f} GiB", flush=True)
    rc = launch("B2", outB, ["--save-every", "0", "--resume", str(ck)])
    if rc != 0:
        print(f"B phase 2 (resume) failed rc={rc}")
        return rc
    rc = launch("A", outA, ["--save-every", "0"])
    if rc != 0:
        print(f"A (control) failed rc={rc}")
        return rc

    A = read_trace(root / outA / "loss_trace.jsonl")
    B = read_trace(root / outB / "loss_trace.jsonl")
    pre = compare(A, B, 0, n + 1)
    post = compare(A, B, n + 1, n + m + 1)
    print("\n=== RESULT (LLaMA2-7B, 8x L20A, zero2) ===")
    print(f"A(control) steps: {sorted(A)}")
    print(f"B(resumed) steps: {sorted(B)}")
    print(f"pre-resume  A vs B: {json.dumps(pre, default=str)}")
    print(f"post-resume A vs B: {json.dumps(post, default=str)}")
    for s in sorted(set(A) & set(B)):
        mark = "  <-- first resumed step" if s == n + 1 else ""
        print(f"  step {s:>3}  A={A[s]['loss']:.12f}  B={B[s]['loss']:.12f}  "
              f"d={abs(A[s]['loss'] - B[s]['loss']):.3e}{mark}")
    return 0 if post["n"] > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="steps before the checkpoint")
    ap.add_argument("--m", type=int, default=3, help="steps after resuming")
    ap.add_argument("--project-root", default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    ap.add_argument("--python", default="/opt/conda/envs/torch-base/bin/python")
    ap.add_argument("--data", default="Mixture-of-Memory/data/dolmino-mix-1124-llama2",
                    help="relative to --project-root. Dolmino (paper Sec. VI-A), NOT C4.")
    ap.add_argument("--out-prefix", default="outputs/cast_resume_faith")
    ap.add_argument("--keep", action="store_true", help="keep the tiny workdir")
    a = ap.parse_args()
    if a.real:
        return run_real(a.n, a.m, a)
    wd = Path(tempfile.mkdtemp(prefix="cast_resume_"))
    try:
        return run_tiny(a.n, a.m, wd)
    finally:
        if a.keep:
            print(f"workdir kept at {wd}")
        else:
            shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
