#!/usr/bin/env python
"""Unit tests for the shared checkpoint-rotation policy.

Pure-function tests on lists of filenames plus a handful of tmpdir tests for the
filesystem wrapper. NO GPU, NO model, NO torch import -- runs in <1s anywhere:

    /opt/conda/envs/torch-base/bin/python scripts/test_ckpt_rotation.py

Every assertion below maps to a stated safety invariant in
``scripts/ckpt_rotation.py``. A rotation bug silently destroys experiments that
cost GPU-weeks, so these are the regression guards.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ckpt_rotation import (  # noqa: E402
    ADAPTER_STEP_DIR_PATTERN,
    MEM_SPACE_ADAPTER_PATTERN,
    STEP_DIR_PATTERN,
    STEP_PT_PATTERN,
    parse_keep_steps,
    rotate_checkpoints,
    select_rotation_victims,
)

_PASS = 0
_FAIL = 0


def check(name, got, want):
    global _PASS, _FAIL
    if got == want:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}\n          got  {got}\n          want {want}")


def victims(names, **kw):
    return sorted(select_rotation_victims(names, **kw))


# --------------------------------------------------------------------------- #
print("== 1. final.pt is NEVER a victim ==")
names = ["final.pt", "step500.pt", "step1000.pt", "step1500.pt", "step2000.pt"]
check("final.pt survives (keep_last_n=1)",
      "final.pt" in victims(names, keep_last_n=1, just_written="step2000.pt"), False)
check("only pre-newest steps die (keep_last_n=1)",
      victims(names, keep_last_n=1, just_written="step2000.pt"),
      ["step1000.pt", "step1500.pt", "step500.pt"])
check("final.pt alone in dir -> nothing deleted",
      victims(["final.pt"], keep_last_n=1), [])
check("final/ dir-pattern never matches",
      victims(["final", "step1000", "step2000"], keep_last_n=1,
              just_written="step2000", pattern=STEP_DIR_PATTERN),
      ["step1000"])

print("== 2. exactly N newest periodic ckpts survive ==")
names = [f"step{s}.pt" for s in (500, 1000, 1500, 2000, 2500, 3000)] + ["final.pt"]
for n in (1, 2, 3, 4):
    v = victims(names, keep_last_n=n, just_written="step3000.pt")
    survivors = sorted(set(x for x in names if x != "final.pt") - set(v))
    check(f"keep_last_n={n} -> {n} survivors", len(survivors), n)
check("keep_last_n=3 keeps the 3 NEWEST",
      sorted(set(f"step{s}.pt" for s in (500, 1000, 1500, 2000, 2500, 3000))
             - set(victims(names, keep_last_n=3, just_written="step3000.pt"))),
      ["step2000.pt", "step2500.pt", "step3000.pt"])
check("keep_last_n >= n_ckpts -> nothing deleted",
      victims(names, keep_last_n=99, just_written="step3000.pt"), [])

print("== 3. keep_steps ALWAYS survive (load-bearing steps) ==")
names = [f"step{s}.pt" for s in (45000, 47500, 48000, 121000)] + ["final.pt"]
check("keep_steps=45000,48000 protected at keep_last_n=1",
      victims(names, keep_last_n=1, keep_steps="45000,48000",
              just_written="step121000.pt"),
      ["step47500.pt"])
check("keep_steps space-separated parses",
      victims(names, keep_last_n=1, keep_steps="45000 48000",
              just_written="step121000.pt"),
      ["step47500.pt"])
check("keep_steps as int list parses",
      victims(names, keep_last_n=1, keep_steps=[45000, 48000],
              just_written="step121000.pt"),
      ["step47500.pt"])
check("keep_steps garbage tokens ignored, valid ones honoured",
      victims(names, keep_last_n=1, keep_steps="45000,abc,,48000",
              just_written="step121000.pt"),
      ["step47500.pt"])
check("parse_keep_steps('') -> empty", parse_keep_steps(""), set())
check("parse_keep_steps(None) -> empty", parse_keep_steps(None), set())
check("parse_keep_steps('128000,153500')",
      parse_keep_steps("128000,153500"), {128000, 153500})

print("== 4. disabled mode (keep_last_n<=0) deletes NOTHING -- #103 opt-out ==")
dense = [f"step{s}.pt" for s in range(2500, 50001, 2500)] + ["final.pt"]
check("keep_last_n=0 -> no victims (20 dense ckpts)",
      victims(dense, keep_last_n=0, just_written="step50000.pt"), [])
check("keep_last_n=-1 -> no victims",
      victims(dense, keep_last_n=-1, just_written="step50000.pt"), [])
check("keep_last_n=0 ignores keep_milestones cap too",
      victims(dense, keep_last_n=0, milestone_every=2500, keep_milestones=2,
              just_written="step50000.pt"), [])
check("keep_last_n=0 -> ALL 20 dense ckpts survive",
      len(set(dense) - set(victims(dense, keep_last_n=0,
                                   just_written="step50000.pt"))), 21)

print("== 5. empty / single-ckpt dirs untouched ==")
check("empty list", victims([], keep_last_n=1), [])
check("no matching names", victims(["arch_meta.json", "README.md"], keep_last_n=1), [])
check("one ckpt, keep_last_n=1", victims(["step500.pt"], keep_last_n=1,
                                         just_written="step500.pt"), [])
check("one ckpt + final, keep_last_n=1",
      victims(["step500.pt", "final.pt"], keep_last_n=1,
              just_written="step500.pt"), [])
check("just_written protected even if not in newest-N",
      victims(["step100.pt", "step200.pt", "step300.pt"], keep_last_n=1,
              just_written="step100.pt"),
      ["step200.pt"])

print("== 6. step0 is protected automatically (paper table row) ==")
names = ["step0.pt", "step5000.pt", "step10000.pt", "step15000.pt"]
check("step0 survives keep_last_n=1",
      victims(names, keep_last_n=1, just_written="step15000.pt"),
      ["step10000.pt", "step5000.pt"])
check("protect_step_zero=False lets step0 rotate",
      victims(names, keep_last_n=1, protect_step_zero=False,
              just_written="step15000.pt"),
      ["step0.pt", "step10000.pt", "step5000.pt"])

print("== 7. milestone cap is the real volume lever ==")
# 200k-step run, save_every=5000 -> 40 milestones, 46 GB each = 1.8 TB
run200k = [f"step{s}.pt" for s in range(5000, 200001, 5000)] + ["final.pt"]
kept_unlimited = set(run200k) - set(victims(
    run200k, keep_last_n=2, milestone_every=5000, keep_milestones=0,
    just_written="step200000.pt"))
check("milestone_every=5000 keep_milestones=0 -> ALL retained (old behaviour)",
      len(kept_unlimited), 41)
kept_capped = sorted(
    set(run200k) - set(victims(run200k, keep_last_n=2, milestone_every=5000,
                               keep_milestones=8,
                               just_written="step200000.pt")),
    key=lambda n: (n != "final.pt", n))
check("keep_milestones=8 -> 8 milestones + final (latest2 subset of them)",
      len(kept_capped), 9)
check("the 8 retained milestones are the NEWEST",
      sorted(int(n[4:-3]) for n in kept_capped if n != "final.pt"),
      [165000, 170000, 175000, 180000, 185000, 190000, 195000, 200000])
check("milestone_every=0 -> milestones disabled entirely",
      len(set(run200k) - set(victims(run200k, keep_last_n=2, milestone_every=0,
                                     just_written="step200000.pt"))), 3)
check("off-grid ckpt not protected by milestone clause",
      "step111500.pt" in victims(
          [f"step{s}.pt" for s in (105000, 110000, 111000, 111500)] + ["final.pt"],
          keep_last_n=1, milestone_every=5000, keep_milestones=0,
          just_written="step111500.pt"), False)

print("== 8. pattern scoping -- never match anything but our own ckpts ==")
noise = ["final.pt", "step1000.pt", "step2000.pt", "arch_meta.json",
         "optimizer.pt", "model.safetensors", "step.pt", "stepABC.pt",
         "mystep3000.pt", "step4000.pt.bak", "step5000.pt.tmp",
         "eval_step1000.pt", "config.json"]
check("only step<N>.pt matched",
      victims(noise, keep_last_n=1, just_written="step2000.pt"),
      ["step1000.pt"])
check("adapter_step<N> pattern scoping",
      victims(["adapter_final", "adapter_step500", "adapter_step1000",
               "merged", "step500"], keep_last_n=1,
              just_written="adapter_step1000",
              pattern=ADAPTER_STEP_DIR_PATTERN),
      ["adapter_step500"])
check("mem_space_adapter_step<N>.pt pattern + zero-padded steps",
      victims(["mem_space_adapter.pt", "mem_space_adapter_step000500.pt",
               "mem_space_adapter_step001000.pt",
               "mem_space_adapter_step001500.pt"], keep_last_n=1,
              just_written="mem_space_adapter_step001500.pt",
              pattern=MEM_SPACE_ADAPTER_PATTERN),
      ["mem_space_adapter_step000500.pt", "mem_space_adapter_step001000.pt"])
check("STEP_DIR_PATTERN does not match step<N>.pt files",
      victims(["step1000.pt", "step2000.pt"], keep_last_n=1,
              just_written="step2000", pattern=STEP_DIR_PATTERN), [])

print("== 9. never leave zero survivors ==")
check("all victims -> refuse (protect_step_zero off, keep_last_n huge-negative "
      "is disabled path, so force via empty keeps)",
      victims(["step1000.pt"], keep_last_n=1, protect_step_zero=False,
              just_written=None), [])
# with just_written=None and keep_last_n=1, step1000 is the newest -> survives.
check("survivor set is never empty for any keep_last_n>=1",
      all(len(set(["step10.pt", "step20.pt"])
              - set(victims(["step10.pt", "step20.pt"], keep_last_n=k))) >= 1
          for k in (1, 2, 3)), True)

print("== 10. filesystem wrapper (tmpdir) ==")
tmp = tempfile.mkdtemp(prefix="ckptrot_")
try:
    def touch(name, size=16):
        with open(os.path.join(tmp, name), "wb") as fh:
            fh.write(b"x" * size)

    for s in (0, 500, 1000, 1500, 2000):
        touch(f"step{s}.pt")
    touch("final.pt")
    touch("arch_meta.json")
    logs = []
    deleted = rotate_checkpoints(tmp, keep_last_n=2,
                                just_written=os.path.join(tmp, "step2000.pt"),
                                log=logs.append)
    left = sorted(os.listdir(tmp))
    check("fs: deleted 2 (step500,step1000); step0/final/meta kept",
          left, ["arch_meta.json", "final.pt", "step0.pt", "step1500.pt",
                 "step2000.pt"])
    check("fs: returns 2 absolute paths", len(deleted), 2)
    check("fs: all returned paths are absolute",
          all(os.path.isabs(p) for p in deleted), True)
    check("fs: every deletion logged",
          sum(1 for m in logs if "removed old checkpoint" in m), 2)

    # failed save -> rotate nothing
    for s in (2500, 3000):
        touch(f"step{s}.pt")
    logs2 = []
    d2 = rotate_checkpoints(tmp, keep_last_n=1,
                            just_written=os.path.join(tmp, "step9999.pt"),
                            log=logs2.append)
    check("fs: missing just_written -> deletes nothing", d2, [])
    check("fs: skip is logged",
          any("missing/empty" in m for m in logs2), True)
    open(os.path.join(tmp, "step9999.pt"), "wb").close()  # zero-byte
    d3 = rotate_checkpoints(tmp, keep_last_n=1,
                            just_written=os.path.join(tmp, "step9999.pt"))
    check("fs: zero-byte just_written -> deletes nothing", d3, [])
    os.remove(os.path.join(tmp, "step9999.pt"))

    # disabled mode on a real dir
    before = sorted(os.listdir(tmp))
    check("fs: keep_last_n=0 deletes nothing",
          rotate_checkpoints(tmp, keep_last_n=0,
                             just_written=os.path.join(tmp, "step3000.pt")), [])
    check("fs: dir unchanged after disabled rotation",
          sorted(os.listdir(tmp)), before)

    # keep_steps on a real dir
    d4 = rotate_checkpoints(tmp, keep_last_n=1, keep_steps="1500,2500",
                            just_written=os.path.join(tmp, "step3000.pt"))
    check("fs: keep_steps honoured on disk",
          sorted(os.path.basename(p) for p in d4), ["step2000.pt"])
    check("fs: step1500/2500 still on disk",
          all(os.path.exists(os.path.join(tmp, n))
              for n in ("step1500.pt", "step2500.pt")), True)

    # stale *.pt.tmp sweep (train_olmo2_sft.py atomic writes)
    touch("step3000.pt.tmp")
    d5 = rotate_checkpoints(tmp, keep_last_n=1, keep_steps="1500,2500",
                            just_written=os.path.join(tmp, "step3000.pt"),
                            sweep_tmp=True)
    check("fs: stale .pt.tmp swept",
          os.path.exists(os.path.join(tmp, "step3000.pt.tmp")), False)
    check("fs: final.pt still present after all rotations",
          os.path.exists(os.path.join(tmp, "final.pt")), True)

    # directory-shaped checkpoints (qcmem LoRA adapters)
    dtmp = tempfile.mkdtemp(prefix="ckptrotdir_")
    for s in (500, 1000, 1500, 2000):
        os.makedirs(os.path.join(dtmp, f"step{s}"))
        with open(os.path.join(dtmp, f"step{s}",
                               "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"y" * 8)
    os.makedirs(os.path.join(dtmp, "final"))
    with open(os.path.join(dtmp, "final", "adapter_model.safetensors"), "wb") as fh:
        fh.write(b"y" * 8)
    d6 = rotate_checkpoints(dtmp, keep_last_n=2,
                            just_written=os.path.join(dtmp, "step2000"),
                            pattern=STEP_DIR_PATTERN, is_dir=True)
    check("fs-dir: rmtree removed 2 step dirs", len(d6), 2)
    check("fs-dir: final/ + newest 2 kept",
          sorted(os.listdir(dtmp)), ["final", "step1500", "step2000"])
    check("fs-dir: empty just_written dir -> deletes nothing",
          rotate_checkpoints(dtmp, keep_last_n=1,
                             just_written=os.path.join(dtmp, "nonexistent"),
                             pattern=STEP_DIR_PATTERN, is_dir=True), [])
    shutil.rmtree(dtmp)

    check("fs: nonexistent output_dir -> no crash, no victims",
          rotate_checkpoints("/nonexistent/path/xyz", keep_last_n=1), [])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("== 11. resumed-run robustness (old pickled args -> getattr defaults) ==")
check("keep_last_n=None coerces to default 3, still rotates safely",
      victims([f"step{s}.pt" for s in (100, 200, 300, 400)],
              keep_last_n=None, just_written="step400.pt"),
      ["step100.pt"])
check("keep_last_n='2' (string from env) coerces",
      victims([f"step{s}.pt" for s in (100, 200, 300)],
              keep_last_n="2", just_written="step300.pt"),
      ["step100.pt"])
check("milestone_every=None tolerated",
      victims([f"step{s}.pt" for s in (100, 200, 300)],
              keep_last_n=1, milestone_every=None,
              just_written="step300.pt"),
      ["step100.pt", "step200.pt"])

print("== 12. end-to-end trainer kwargs paths (rotation_kwargs_from_args) ==")
import types  # noqa: E402

from ckpt_rotation import rotation_kwargs_from_args  # noqa: E402

_tmpdirs = []


def _mkdir_with(names):
    d = tempfile.mkdtemp(prefix="ckptrot_e2e_")
    _tmpdirs.append(d)
    for n in names:
        with open(os.path.join(d, n), "wb") as fh:
            fh.write(b"x" * 32)
    return d


try:
    # (a) probe2 defaults: keep_last_n=3, milestone_every=5000, unlimited milestones
    d = _mkdir_with([f"step{s}.pt" for s in (0, 5000, 10000, 15000, 15500, 16000)]
                    + ["final.pt"])
    args = types.SimpleNamespace(output_dir=d, keep_last_n=3, keep_steps="",
                                 milestone_every=5000, keep_milestones=0)
    rotate_checkpoints(d, just_written=os.path.join(d, "step16000.pt"),
                       **rotation_kwargs_from_args(args,
                                                   default_milestone_every=5000))
    check("e2e probe2 defaults keep final+step0+milestones+latest3",
          sorted(os.listdir(d)),
          ["final.pt", "step0.pt", "step10000.pt", "step15000.pt",
           "step15500.pt", "step16000.pt", "step5000.pt"])

    # (b) #103 DENSE-SAVE OPT-OUT. This is the highest-consequence case in the
    # whole suite: outputs/olmo2_keep14_densesave_reheal exists solely to retain
    # EVERY every-2500 save so the matched-PPL crossing can be bracketed
    # (step27500 = the only STRICT frozen-front match). If rotation ever touches
    # this dir, experiment #103 is destroyed and needs ~57 GPU-hours to redo.
    dense = [f"step{s}.pt" for s in range(2500, 52501, 2500)]
    d = _mkdir_with(dense + ["final.pt"])
    args = types.SimpleNamespace(output_dir=d, keep_last_n=0, keep_steps="",
                                 milestone_every=2500, keep_milestones=0)
    check("e2e #103 opt-out (keep_last_n=0) deletes nothing",
          rotate_checkpoints(d, just_written=os.path.join(d, "step52500.pt"),
                             **rotation_kwargs_from_args(
                                 args, default_milestone_every=5000)), [])
    check("e2e #103 all 21 dense ckpts + final.pt intact",
          sorted(os.listdir(d)) == sorted(dense + ["final.pt"]), True)
    check("e2e #103 the PPL-bracket ckpts specifically survive",
          all(os.path.exists(os.path.join(d, f"step{s}.pt"))
              for s in (25000, 27500, 30000)), True)

    # (c) keepN launcher defaults on a real 200k grid + off-grid keep_steps.
    # Reproduces the shortgpt16 shape: 40 on-grid milestones + 2 off-grid
    # load-bearing ckpts (128000/153500) + step0 + final = 44 files / 2.0 TB.
    grid = [f"step{s}.pt" for s in range(5000, 200001, 5000)]
    offgrid = ["step128000.pt", "step153500.pt"]
    d = _mkdir_with(grid + offgrid + ["step0.pt", "final.pt"])
    args = types.SimpleNamespace(output_dir=d, keep_last_n=3,
                                 keep_steps="128000,153500",
                                 milestone_every=5000, keep_milestones=8)
    rotate_checkpoints(d, just_written=os.path.join(d, "step200000.pt"),
                       **rotation_kwargs_from_args(args,
                                                   default_milestone_every=5000))
    left = sorted(os.listdir(d))
    check("e2e keepN defaults: 44 ckpts -> 12", len(left), 12)
    check("e2e keepN: off-grid keep_steps survive (only --keep_steps saves them)",
          "step128000.pt" in left and "step153500.pt" in left, True)
    check("e2e keepN: final.pt + step0.pt survive",
          "final.pt" in left and "step0.pt" in left, True)

    # (d) the OLD hardcoded behaviour (latest-2 + unlimited milestones) for
    # contrast: it keeps 42 files -- i.e. rotation "worked" and still produced
    # 2.0 TB. This is why --keep_milestones is the actual volume lever.
    d = _mkdir_with(grid + offgrid + ["step0.pt", "final.pt"])
    args = types.SimpleNamespace(output_dir=d, keep_last_n=2, keep_steps="",
                                 milestone_every=5000, keep_milestones=0)
    rotate_checkpoints(d, just_written=os.path.join(d, "step200000.pt"),
                       **rotation_kwargs_from_args(args,
                                                   default_milestone_every=5000))
    check("e2e old behaviour keeps 42 (the 2.0 TB outcome, reproduced)",
          len(os.listdir(d)), 42)

    # (e) resumed run whose pickled train_args predates these flags entirely
    d = _mkdir_with([f"step{s}.pt" for s in (1000, 2000, 3000, 4000)] + ["final.pt"])
    args = types.SimpleNamespace(output_dir=d, milestone_every=5000)
    kw = rotation_kwargs_from_args(args, default_milestone_every=5000)
    check("e2e resumed-run getattr defaults", kw,
          {"keep_last_n": 3, "keep_steps": "", "milestone_every": 5000,
           "keep_milestones": 0})
    rotate_checkpoints(d, just_written=os.path.join(d, "step4000.pt"), **kw)
    check("e2e resumed run keeps final + latest3",
          sorted(os.listdir(d)),
          ["final.pt", "step2000.pt", "step3000.pt", "step4000.pt"])
finally:
    for d in _tmpdirs:
        shutil.rmtree(d, ignore_errors=True)

print()
print(f"RESULT: {_PASS} passed, {_FAIL} failed")
sys.exit(1 if _FAIL else 0)
