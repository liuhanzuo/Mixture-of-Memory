"""Shared checkpoint-rotation policy for all trainers in this repo.

WHY THIS EXISTS
---------------
Before this module, four trainers carried a hand-copied "latest-2 + every-5000
milestone" rotation block and the rest had none at all. That produced multi-TB
output dirs (measured: ``outputs/olmo2_probe2_7B_shortgpt16`` = 2.0 TB / 44
checkpoints at 46 GB each) and forced a manual 9.4 TiB cleanup. The volume
driver was NOT the latest-N clause -- it was the *unbounded* milestone clause
(``step % 5000 == 0`` retained forever, 40 of them over a 200k-step run).

This module centralises the policy as a **pure function** on a list of names
(:func:`select_rotation_victims`) plus a thin filesystem wrapper
(:func:`rotate_checkpoints`). The pure function is unit-tested in
``scripts/test_ckpt_rotation.py`` -- no GPU, no model, no torch.

RETENTION POLICY
----------------
Given the periodic checkpoints present in ONE output_dir, a checkpoint is KEPT
if ANY of the following holds:

1. it is the checkpoint that was just written (``just_written``);
2. it is one of the ``keep_last_n`` highest step numbers;
3. its step is listed in ``keep_steps`` (the load-bearing-step escape hatch:
   paired-trajectory points, PPL-bracketing points, paper-table rows);
4. its step is 0 and ``protect_step_zero`` (step0 is a paper table row and the
   recovery-fraction denominator, and it is *irreproducible* once the original
   init seed is lost);
5. its step is a multiple of ``milestone_every`` AND is among the
   ``keep_milestones`` newest such multiples (``keep_milestones <= 0`` means
   "unlimited milestones", which is the historical behaviour).

Everything else is a victim. Non-matching filenames -- crucially ``final.pt``
and ``final/`` -- are NEVER even considered, because the pattern only matches
the trainer's own periodic-checkpoint naming (``step<N>.pt`` / ``step<N>``).

HARD SAFETY INVARIANTS (each one has a unit test)
-------------------------------------------------
* ``final.pt`` / ``final/`` never deleted -- it cannot match the step pattern,
  and :func:`rotate_checkpoints` additionally hard-blacklists it.
* ``keep_last_n <= 0`` disables rotation entirely and deletes NOTHING. This is
  the opt-out that dense-save runs (Paper B #103 matched-PPL crossing-point
  capture, ``outputs/olmo2_keep14_densesave_reheal``) MUST use: their entire
  purpose is retaining every single save so the PPL crossing can be bracketed.
  Pass ``--keep_last_n 0`` and rotation becomes a no-op.
* We never empty a directory: if the policy would leave zero survivors among the
  matching entries, nothing is deleted.
* If ``just_written`` is supplied but the file/dir does not exist or is empty,
  the save is assumed to have failed and NOTHING is deleted.
* Only names matching the trainer's own pattern inside the SAME output_dir are
  ever considered. No broader globbing, no recursion, no parent dirs.
* Rotation is the caller's responsibility to run on rank 0 only (all call sites
  are already inside an ``is_main`` guard). :func:`rotate_checkpoints` does not
  touch torch.distributed.
* Every deletion is logged explicitly so it shows up in the training log.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from typing import Callable, Iterable, List, Optional, Sequence, Set

__all__ = [
    "STEP_PT_PATTERN",
    "STEP_DIR_PATTERN",
    "ADAPTER_STEP_DIR_PATTERN",
    "MEM_SPACE_ADAPTER_PATTERN",
    "NEVER_DELETE",
    "parse_keep_steps",
    "select_rotation_victims",
    "rotate_checkpoints",
    "add_rotation_args",
    "rotation_kwargs_from_args",
]

# --- periodic-checkpoint naming patterns (anchored; must match the WHOLE name) -
# ``step1500.pt``           -- the olmo2/qwen3/hyv3 *.pt trainers
STEP_PT_PATTERN = r"step(\d+)\.pt"
# ``step1500``              -- the qcmem LoRA distill trainers (directories)
STEP_DIR_PATTERN = r"step(\d+)"
# ``adapter_step1500``      -- train_olmo2_lora_sft.py (directories)
ADAPTER_STEP_DIR_PATTERN = r"adapter_step(\d+)"
# ``mem_space_adapter_step001500.pt`` -- train_mem_space_dolmino_cpt.py
MEM_SPACE_ADAPTER_PATTERN = r"mem_space_adapter_step(\d+)\.pt"

#: Names that are hard-blacklisted from deletion no matter what a pattern says.
#: The "final" convention across this repo, plus the atomic-write temp suffix is
#: handled separately (see ``sweep_tmp`` in :func:`rotate_checkpoints`).
NEVER_DELETE = frozenset({
    "final.pt",
    "final",
    "full_model.pt",
    "mem_space_adapter.pt",
    "adapter_final.pt",
    "adapter_final",
    "merged",
})


def parse_keep_steps(spec) -> Set[int]:
    """Parse a ``--keep_steps`` value into a set of ints.

    Accepts ``None``, ``""``, an iterable of ints, or a comma/space separated
    string (``"128000,153500"``, ``"45000 121000"``). Unparseable tokens are
    ignored rather than raising, so a typo in a launcher can never crash a
    training run that is hours in -- it just protects fewer steps. Negative
    values are dropped.
    """
    if spec is None:
        return set()
    if isinstance(spec, (int,)) and not isinstance(spec, bool):
        return {int(spec)} if int(spec) >= 0 else set()
    if not isinstance(spec, str):
        out = set()
        for tok in spec:  # already an iterable of ints
            try:
                v = int(tok)
            except (TypeError, ValueError):
                continue
            if v >= 0:
                out.add(v)
        return out
    out = set()
    for tok in re.split(r"[,\s]+", spec.strip()):
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if v >= 0:
            out.add(v)
    return out


def select_rotation_victims(
    names: Iterable[str],
    keep_last_n: int = 3,
    keep_steps: Iterable[int] = (),
    just_written: Optional[str] = None,
    milestone_every: int = 0,
    keep_milestones: int = 0,
    protect_step_zero: bool = True,
    pattern: str = STEP_PT_PATTERN,
) -> List[str]:
    """Pure rotation policy: which of ``names`` should be deleted?

    ``names`` are BASENAMES from a single output_dir (order irrelevant).
    Returns the victim basenames sorted by step ascending. Deletes nothing
    (returns ``[]``) when rotation is disabled, when the save failed, or when
    the policy would leave no survivors.

    This function performs NO filesystem access whatsoever -- it is the unit
    under test.
    """
    # Rotation disabled: keep_last_n <= 0 is the documented opt-out sentinel
    # used by dense-save runs (#103). Bail out before parsing anything.
    try:
        keep_last_n = int(keep_last_n)
    except (TypeError, ValueError):
        keep_last_n = 3
    if keep_last_n <= 0:
        return []

    rx = re.compile(r"^(?:%s)$" % pattern)
    matched = []  # (step, name)
    seen = set()
    for name in names:
        base = os.path.basename(str(name))
        if base in NEVER_DELETE or base in seen:
            continue
        m = rx.match(base)
        if not m:
            continue
        seen.add(base)
        matched.append((int(m.group(1)), base))
    if not matched:
        return []

    matched.sort(key=lambda t: (t[0], t[1]))
    steps_desc = sorted({s for s, _n in matched}, reverse=True)

    keep_steps = parse_keep_steps(keep_steps)

    # (2) the keep_last_n newest step numbers.
    latest = set(steps_desc[:keep_last_n])

    # (5) milestones, optionally capped to the newest ``keep_milestones``.
    try:
        milestone_every = int(milestone_every)
    except (TypeError, ValueError):
        milestone_every = 0
    try:
        keep_milestones = int(keep_milestones)
    except (TypeError, ValueError):
        keep_milestones = 0
    if milestone_every > 0:
        ms = [s for s in steps_desc if s % milestone_every == 0]
        kept_ms = set(ms) if keep_milestones <= 0 else set(ms[:keep_milestones])
    else:
        kept_ms = set()

    jw = os.path.basename(just_written) if just_written else None

    victims, survivors = [], []
    for s, name in matched:
        keep = (
            (jw is not None and name == jw)          # (1) just written
            or s in latest                           # (2) newest N
            or s in keep_steps                       # (3) load-bearing steps
            or (protect_step_zero and s == 0)        # (4) step0 is a paper row
            or s in kept_ms                          # (5) retained milestone
        )
        (survivors if keep else victims).append(name)

    # Never empty a directory. If something in the policy went wrong badly
    # enough that nothing survives, refuse to delete anything.
    if not survivors:
        return []
    return victims


def rotate_checkpoints(
    output_dir: str,
    keep_last_n: int = 3,
    keep_steps: Iterable[int] = (),
    just_written: Optional[str] = None,
    milestone_every: int = 0,
    keep_milestones: int = 0,
    protect_step_zero: bool = True,
    pattern: str = STEP_PT_PATTERN,
    is_dir: bool = False,
    log: Optional[Callable[[str], None]] = None,
    sweep_tmp: bool = False,
    dry_run: bool = False,
) -> List[str]:
    """Apply :func:`select_rotation_victims` to ``output_dir`` and delete.

    Call this ONLY from rank 0, and ONLY after the new checkpoint is fully
    written (pass its path as ``just_written`` so it is both protected and used
    as the "did the save actually succeed?" probe).

    ``is_dir=True`` switches to :func:`shutil.rmtree` for directory-shaped
    checkpoints (the qcmem LoRA adapters save ``step<N>/`` dirs, where
    ``os.remove`` would fail).

    ``sweep_tmp=True`` additionally removes stale ``*.pt.tmp`` files left behind
    by an interrupted atomic write (only ``train_olmo2_sft.py`` writes those).

    Returns the list of absolute paths actually deleted. Never raises: any
    per-entry failure is logged and skipped, because a rotation problem must
    never kill a training run.
    """
    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    try:
        if int(keep_last_n) <= 0:
            return []
    except (TypeError, ValueError):
        pass

    if not output_dir or not os.path.isdir(output_dir):
        return []

    # "Rotate only AFTER the new checkpoint is fully written and verified."
    # If the just-written ckpt is missing or empty, the save failed -> rotate
    # nothing, so a failed save can never cost us an older checkpoint.
    if just_written:
        try:
            if is_dir:
                ok = os.path.isdir(just_written) and bool(os.listdir(just_written))
            else:
                ok = os.path.isfile(just_written) and os.path.getsize(just_written) > 0
        except OSError:
            ok = False
        if not ok:
            _log(f"[ckpt-rotate] SKIP: just-written checkpoint {just_written} "
                 f"missing/empty -> rotating nothing")
            return []

    try:
        entries = os.listdir(output_dir)
    except OSError as exc:
        _log(f"[ckpt-rotate] could not list {output_dir}: {exc}")
        return []
    # keep only entries of the right shape (file vs dir) before applying policy
    kind_ok = []
    for name in entries:
        full = os.path.join(output_dir, name)
        if is_dir and os.path.isdir(full):
            kind_ok.append(name)
        elif not is_dir and os.path.isfile(full):
            kind_ok.append(name)

    victims = select_rotation_victims(
        kind_ok,
        keep_last_n=keep_last_n,
        keep_steps=keep_steps,
        just_written=just_written,
        milestone_every=milestone_every,
        keep_milestones=keep_milestones,
        protect_step_zero=protect_step_zero,
        pattern=pattern,
    )

    jw_abs = os.path.abspath(just_written) if just_written else None
    deleted: List[str] = []
    for name in victims:
        full = os.path.join(output_dir, name)
        abs_full = os.path.abspath(full)
        # belt-and-braces: never the just-written path, never a NEVER_DELETE name
        if jw_abs is not None and abs_full == jw_abs:
            continue
        if os.path.basename(abs_full) in NEVER_DELETE:
            continue
        if dry_run:
            deleted.append(abs_full)
            _log(f"[ckpt-rotate] DRY-RUN would remove {abs_full}")
            continue
        try:
            if is_dir:
                shutil.rmtree(abs_full)
            else:
                os.remove(abs_full)
            deleted.append(abs_full)
            _log(f"[ckpt-rotate] removed old checkpoint {abs_full}")
        except OSError as exc:
            _log(f"[ckpt-rotate] could NOT remove {abs_full}: {exc}")

    if sweep_tmp and not dry_run:
        for tmp in glob.glob(os.path.join(output_dir, "*.pt.tmp")):
            if jw_abs is not None and os.path.abspath(tmp) == jw_abs:
                continue
            try:
                os.remove(tmp)
                _log(f"[ckpt-rotate] removed stale temp file {tmp}")
            except OSError as exc:
                _log(f"[ckpt-rotate] could NOT remove stale temp {tmp}: {exc}")

    if deleted:
        _log(f"[ckpt-rotate] {output_dir}: rotated {len(deleted)} old checkpoint(s) "
             f"(keep_last_n={keep_last_n} keep_steps={sorted(parse_keep_steps(keep_steps))} "
             f"milestone_every={milestone_every} keep_milestones={keep_milestones})")
    return deleted


# --------------------------------------------------------------------------- #
# argparse plumbing shared by every trainer
# --------------------------------------------------------------------------- #
_KEEP_LAST_N_HELP = (
    "checkpoint ROTATION: after each successful periodic save, keep only the N "
    "newest periodic checkpoints and delete the rest. 0 (or any value <=0) "
    "DISABLES rotation entirely and keeps everything -- this is the required "
    "opt-out for dense-save runs whose whole point is retaining every save "
    "(e.g. Paper B #103 matched-PPL crossing-point bracketing). final.pt is "
    "never rotated, and --keep_steps protects named load-bearing steps."
)
_KEEP_STEPS_HELP = (
    "comma-separated step numbers that are ALWAYS retained regardless of "
    "rotation (e.g. '128000,153500'). Use this for load-bearing checkpoints: "
    "paired-trajectory points, PPL-bracketing points, and any step cited by a "
    "paper table. step0 is protected automatically."
)
_KEEP_MILESTONES_HELP = (
    "cap on how many --milestone_every multiples are retained (newest first). "
    "0 = unlimited, which is the historical behaviour and the default (a 200k "
    "step 7B run then retains 40 milestones ~= 1.8 TB). Set e.g. 8 to bound a "
    "long run's checkpoint volume; the launchers do this by default."
)


def add_rotation_args(p, default_keep_last_n: int = 3,
                      default_milestone_every: Optional[int] = None,
                      default_keep_milestones: int = 0):
    """Register ``--keep_last_n`` / ``--keep_steps`` / ``--keep_milestones``.

    ``default_milestone_every=None`` means the caller already declares its own
    ``--milestone_every`` (or deliberately has no milestone concept), so we do
    not add it and avoid an argparse conflict.
    """
    p.add_argument("--keep_last_n", type=int, default=default_keep_last_n,
                   help=_KEEP_LAST_N_HELP)
    p.add_argument("--keep_steps", type=str, default="", help=_KEEP_STEPS_HELP)
    if default_milestone_every is not None:
        p.add_argument("--milestone_every", type=int,
                       default=default_milestone_every,
                       help="rolling-retention milestone modulus: periodic "
                            "checkpoints whose step is a multiple of this are "
                            "retained beyond --keep_last_n (subject to "
                            "--keep_milestones). 0 disables milestones.")
    p.add_argument("--keep_milestones", type=int, default=default_keep_milestones,
                   help=_KEEP_MILESTONES_HELP)
    return p


def rotation_kwargs_from_args(args, default_milestone_every: int = 0) -> dict:
    """Extract rotation kwargs from an argparse Namespace defensively.

    Uses ``getattr`` throughout so a run RESUMED from a checkpoint whose pickled
    ``train_args`` predates these flags can never crash.
    """
    return {
        "keep_last_n": int(getattr(args, "keep_last_n", 3) or 0),
        "keep_steps": getattr(args, "keep_steps", "") or "",
        "milestone_every": int(getattr(args, "milestone_every",
                                      default_milestone_every)
                               or default_milestone_every or 0),
        "keep_milestones": int(getattr(args, "keep_milestones", 0) or 0),
    }
