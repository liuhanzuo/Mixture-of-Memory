#!/usr/bin/env python3
"""B03 G0 — the LLF reset operator as CPU-only checkpoint surgery, plus its self-tests.

WHAT THIS IS
------------
`GATE_PREREGISTRATION.md` sec 8.2 (re-affirmed unchanged by `READOUT_RULE_20260816.md`
sec 6 clause 1) pre-registers G0 as: a standalone, 0-GPU checkpoint-surgery script that

  * reads a trainer-written `step{N}.pt` (the format `scripts/train_olmo2_arch_probe2.py::_save`
    writes, line 513),
  * re-initialises `model.layers.{keep_front .. keep_front + n_fresh - 1}` from the SAME
    `Olmo2ForCausalLM(cfg).post_init()` distribution the trainer uses,
  * zeroes `exp_avg` / `exp_avg_sq` and resets `step` for exactly those parameter indices
    inside `optimizer_state`,
  * leaves every other tensor BYTE-IDENTICAL,
  * writes a ckpt that `--resume_from` strict-loads.

This is deliberately NOT a trainer change. `scripts/train_olmo2_arch_probe2.py` is
imported READ-ONLY, and the fresh init is produced by calling the trainer's own
`build_olmo2_minimal(..., transplant=False)` -- i.e. `Olmo2ForCausalLM(cfg)` + `post_init`
under the trainer's own `set_seed`. NOTHING here hand-builds an init with `torch.nn.init`;
the prereg and the trainer docstring at line 264 both forbid that, and a hand-built init
would pass `_assert_fresh_init`'s three loose checks while differing from the real Olmo2
init on every other tensor.

The operator is LLF (Zhou et al., ICLR 2022) with mask `M^l = 1[l < keep_front]`. It is
labelled as LLF and claims no novelty; see `RELATED_WORK.md` sec 2.1 / sec 3 items 1-2.

THE FOUR PRE-REGISTERED SELF-TESTS
----------------------------------
(a) every non-reset tensor is byte-identical to the input (sha256 per tensor);
(b) the reset layers pass the trainer's own `_assert_fresh_init` (`post_attention_layernorm`
    all-ones -- NOT `input_layernorm`, because OLMo-2 is POST-norm -- `q_norm` all-ones,
    `q_proj.weight` std in the Olmo2 init band);
(c) the surgical ckpt strict-loads with ZERO missing/unexpected keys;
(d) the `optimizer_state` param_group count is preserved, so the trainer's 2-group -> 4-group
    compatibility shim (line 912) is NOT silently triggered.

    ** (c) IS RE-SPECIFIED, AND THAT RE-SPECIFICATION IS ITSELF A PRE-DATA FINDING. **
    The prereg's literal wording for (c) is "strict-loads under `--dry_run_build`". That is
    UNFIREABLE. AST-verified (evidence/g0_selftest_c_unfireable_20260817.json): in `main()`,
    `if args.dry_run_build:` is statement 44 at line 714 and CONTAINS A RETURN, while
    `if args.resume_from:` is statement 47 at line 752. So passing `--dry_run_build` together
    with `--resume_from` exits before `torch.load` is ever called: the surgical ckpt is never
    opened, and the run prints `[dry_run_build] ... -> OK` no matter how corrupt the surgery
    was. A literal execution of (c) would be a green check whose target lies entirely outside
    the tested space.

    This script therefore exercises the REAL strict path directly: it builds the shell via
    the trainer's own `build_olmo2_minimal(..., transplant=False)` -- the same constructor
    the resume path uses at line 762 -- and calls
    `load_state_dict(ckpt['model_state'], strict=True)`, asserting `missing == []` and
    `unexpected == []`. That is exactly the statement the trainer executes at line 776,
    reached with the surgical checkpoint actually loaded.

    ** (d) IS ONLY MEANINGFUL AGAINST A GENUINE 4-GROUP TRAINER CKPT. ** `build_param_groups`
    line 480 DROPS empty groups and `_classify_param` routes every
    `model.layers.{lid >= keep_front}` param to `fresh`, so the group count a resume rebuilds
    is a function of the FLAGS AND ARCH, not of the ckpt. An agent can make (d) "pass" by
    feeding a ckpt whose saved group count happens to match. The script therefore RECORDS
    `n_ckpt_groups` before and after AND `n_rebuilt_groups` (the count the trainer's own
    `build_param_groups` produces for the resume flags), and asserts all three agree. If the
    input has 2 groups the test is reported as VACUOUS rather than PASS.

USAGE
-----
  # surgery + all four self-tests, writing a machine-readable result JSON
  python reset_layers_ckpt.py \
      --in_ckpt  outputs/<run>/step{N}.pt \
      --out_ckpt outputs/<run>/step{N}_reset.pt \
      --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-0425-1B \
      --keep_front_layers 7 --n_fresh_layers 2 --seed 42 \
      --result_json <evidence>/g0_result.json

  # self-tests only, no output ckpt kept  (adds --no_keep_out)
Exit code 0 iff every non-skipped self-test PASSES. Non-zero means a real failure; there is
no "informational" non-zero rc here.

0 GPU. Runs entirely on CPU; asserts CUDA is never initialised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# READ-ONLY import of the trainer. We reuse its constructor, its fresh-init assert, its
# param classifier and its group builder so the operator cannot drift from the trainer.
import train_olmo2_arch_probe2 as T  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256_tensor(t: torch.Tensor) -> str:
    """sha256 over the tensor's raw bytes, in a canonical contiguous CPU layout.

    `.contiguous()` is load-bearing: two tensors can be numerically equal yet have
    different strides, and `numpy().tobytes()` on a non-contiguous view would hash a
    different byte sequence for the same value. We want "byte-identical VALUES", which is
    what the prereg means by leaving other tensors byte-identical.
    """
    x = t.detach().to("cpu").contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def _layer_id(name: str):
    """Decoder-layer index of a `model.layers.<i>....` parameter name, else None."""
    if not name.startswith("model.layers."):
        return None
    try:
        return int(name.split(".")[2])
    except (IndexError, ValueError):
        return None


def _reset_layer_ids(keep_front: int, n_fresh: int) -> set[int]:
    return set(range(keep_front, keep_front + n_fresh))


def _param_name_order(model) -> list[str]:
    """`named_parameters()` order of the bare (non-DDP) model.

    This is the order `build_param_groups` walks, and therefore the order that determines
    which flat optimizer index a parameter occupies. Used to map optimizer indices ->
    parameter names so we can zero the moments of exactly the reset layers.
    """
    return [n for n, _ in model.named_parameters()]


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------
def build_fresh_reference(model_path, keep_front, n_fresh, seed, dtype=torch.float32):
    """A fresh `keep_front + n_fresh` shell built EXACTLY as the trainer builds it.

    `T.set_seed(seed)` then `T.build_olmo2_minimal(..., transplant=False)`, which is
    `Olmo2ForCausalLM(cfg)` (post_init runs inside `__init__`) `.to(dtype)`. This is the
    same call the trainer makes on the resume path (line 762 with `do_transplant=False`),
    so the random init we transplant is the trainer's own init distribution, not a
    hand-built one.
    """
    T.set_seed(seed)
    model, cfg, _ = T.build_olmo2_minimal(
        model_path, keep_front, n_fresh, dtype, transplant=False, is_main=False,
    )
    return model, cfg


def reset_optimizer_moments(optim_state, reset_flat_indices):
    """Zero `exp_avg`/`exp_avg_sq` and reset `step` for exactly `reset_flat_indices`.

    `optim_state` is `optimizer.state_dict()` as written by `_save`: `{"state": {int: {...}},
    "param_groups": [...]}`. AdamW's per-param entries are `step`, `exp_avg`, `exp_avg_sq`
    (torch stores `step` as a 0-dim tensor). We zero IN PLACE on the loaded dict, and we
    return a per-index report so the result JSON can be audited rather than trusted.

    Indices with no state entry (e.g. a param that has never received a gradient) are
    reported as `absent`, not silently ignored -- a reset layer whose moments are missing
    would mean the input ckpt was not a real post-`optimizer.step()` save.
    """
    report = {"zeroed": [], "absent": [], "unexpected_keys_seen": {}}
    st = optim_state["state"]
    for idx in sorted(reset_flat_indices):
        # torch may key `state` by int; a JSON round-trip elsewhere could make it str.
        key = idx if idx in st else (str(idx) if str(idx) in st else None)
        if key is None:
            report["absent"].append(idx)
            continue
        entry = st[key]
        seen = sorted(entry.keys())
        if seen != ["exp_avg", "exp_avg_sq", "step"]:
            report["unexpected_keys_seen"][str(idx)] = seen
        for mkey in ("exp_avg", "exp_avg_sq"):
            if mkey in entry and isinstance(entry[mkey], torch.Tensor):
                entry[mkey] = torch.zeros_like(entry[mkey])
        if "step" in entry:
            s = entry["step"]
            entry["step"] = (torch.zeros_like(s) if isinstance(s, torch.Tensor) else 0)
        report["zeroed"].append(idx)
    return report


def surgery(args):
    """Do the reset and run the four self-tests. Returns a result dict."""
    t_start = time.time()
    res = {
        "tool": os.path.abspath(__file__),
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_used": False,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "inputs": {
            "in_ckpt": os.path.abspath(args.in_ckpt),
            "out_ckpt": os.path.abspath(args.out_ckpt) if args.out_ckpt else None,
            "model_path": args.model_path,
            "keep_front_layers": args.keep_front_layers,
            "n_fresh_layers": args.n_fresh_layers,
            "seed": args.seed,
        },
        "trainer_ref": {
            "path": os.path.abspath(T.__file__),
            "sha256": hashlib.sha256(open(T.__file__, "rb").read()).hexdigest(),
        },
        "selftests": {},
        "notes": [],
    }

    keep_front = args.keep_front_layers
    n_fresh = args.n_fresh_layers
    reset_ids = _reset_layer_ids(keep_front, n_fresh)
    res["reset_layer_ids"] = sorted(reset_ids)

    # ---- 1. read the input ckpt -------------------------------------------------
    ck = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    res["in_ckpt_keys"] = sorted(ck.keys())
    res["in_ckpt_step"] = ck.get("step")
    res["in_ckpt_has_optimizer_state"] = "optimizer_state" in ck
    ms = ck["model_state"]
    res["n_model_tensors"] = len(ms)

    # arch cross-check: the ckpt must be the scale/shape the gate is designed at.
    hid = ms["model.embed_tokens.weight"].shape[1]
    n_layers_in_ckpt = len({_layer_id(k) for k in ms if _layer_id(k) is not None})
    res["ckpt_arch"] = {
        "hidden_size": int(hid),
        "vocab_size": int(ms["model.embed_tokens.weight"].shape[0]),
        "n_decoder_layers": int(n_layers_in_ckpt),
        "ckpt_keep_front_layers": ck.get("keep_front_layers"),
        "ckpt_n_fresh_layers": ck.get("n_fresh_layers"),
    }
    assert n_layers_in_ckpt == keep_front + n_fresh, (
        f"ckpt has {n_layers_in_ckpt} decoder layers but "
        f"keep_front+n_fresh = {keep_front + n_fresh}; wrong arch/scale for this gate")

    # ---- 2. hash EVERY tensor of the input, before we touch anything -----------
    sha_before = {k: _sha256_tensor(v) for k, v in ms.items()}

    # ---- 3. build the trainer's own fresh shell and transplant the reset layers -
    fresh, cfg = build_fresh_reference(
        args.model_path, keep_front, n_fresh, args.seed, dtype=torch.float32)
    fresh_sd = fresh.state_dict()
    res["fresh_shell"] = {
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(cfg.vocab_size),
        "initializer_range": float(cfg.initializer_range),
        "built_via": "train_olmo2_arch_probe2.build_olmo2_minimal(transplant=False)"
                     " -> Olmo2ForCausalLM(cfg) post_init",
    }
    assert int(cfg.hidden_size) == int(hid), (
        f"fresh shell hidden_size {cfg.hidden_size} != ckpt hidden_size {hid}")

    n_replaced = 0
    replaced_names = []
    for k in list(ms.keys()):
        lid = _layer_id(k)
        if lid is not None and lid in reset_ids:
            assert k in fresh_sd, f"reset key {k} absent from the fresh shell"
            assert tuple(fresh_sd[k].shape) == tuple(ms[k].shape), (
                f"shape mismatch on {k}: fresh {tuple(fresh_sd[k].shape)} "
                f"vs ckpt {tuple(ms[k].shape)}")
            ms[k] = fresh_sd[k].detach().clone().to(ms[k].dtype)
            n_replaced += 1
            replaced_names.append(k)
    res["n_tensors_replaced"] = n_replaced
    res["replaced_tensor_names"] = sorted(replaced_names)
    expected_replaced = T.N_TENSORS_PER_LAYER * n_fresh
    res["expected_n_tensors_replaced"] = expected_replaced
    assert n_replaced == expected_replaced, (
        f"replaced {n_replaced} tensors but N_TENSORS_PER_LAYER*n_fresh = "
        f"{expected_replaced}")

    # ---- 4. optimizer moments for exactly the reset params ---------------------
    # The flat optimizer index of a param is its position in the order
    # build_param_groups walked, which is model.named_parameters() order on the bare
    # model. We rebuild that order from the trainer's own helpers so we cannot drift.
    n_ckpt_groups_before = None
    optim_report = None
    n_rebuilt_groups = None
    if "optimizer_state" in ck:
        opt = ck["optimizer_state"]
        n_ckpt_groups_before = len(opt["param_groups"])

        shell_for_order, _, _ = T.build_olmo2_minimal(
            args.model_path, keep_front, n_fresh, torch.float32,
            transplant=False, is_main=False)
        names_in_order = _param_name_order(shell_for_order)

        # Reproduce the trainer's group layout so the flat index we compute is the same
        # index the trainer's optimizer.state_dict() used. build_param_groups walks
        # named_parameters() once and appends into 4 buckets, then DROPS empty buckets
        # (line 480); optimizer.state_dict() then numbers params 0..n-1 in the order
        # they appear across the surviving groups.
        class _A:
            keep_front_layers = keep_front
            from_scratch = False
            random_trunk = False
            weight_decay = 0.1
            lr = 1e-4
            min_lr = 1e-5
            lr_inherited = 2e-5
            min_lr_inherited = 2e-6
        buckets = {"fresh_decay": [], "fresh_nodecay": [], "inh_decay": [], "inh_nodecay": []}
        for nm, pp in shell_for_order.named_parameters():
            cls = T._classify_param(nm, keep_front, False, random_trunk=False)
            prefix = "fresh" if cls == "fresh" else "inh"
            key = f"{prefix}_decay" if pp.ndim >= 2 else f"{prefix}_nodecay"
            buckets[key].append(nm)
        surviving = [b for b in ("fresh_decay", "fresh_nodecay", "inh_decay", "inh_nodecay")
                     if buckets[b]]
        n_rebuilt_groups = len(surviving)
        flat_order = [nm for b in surviving for nm in buckets[b]]
        assert sorted(flat_order) == sorted(names_in_order), (
            "reconstructed flat param order is not a permutation of named_parameters()")

        name_to_flat = {nm: i for i, nm in enumerate(flat_order)}
        reset_flat = {name_to_flat[nm] for nm in flat_order
                      if _layer_id(nm) is not None and _layer_id(nm) in reset_ids}
        res["n_reset_optimizer_indices"] = len(reset_flat)
        res["n_optimizer_states_in_ckpt"] = len(opt["state"])
        res["flat_index_group_layout"] = {b: len(buckets[b]) for b in buckets}
        res["surviving_groups"] = surviving

        optim_report = reset_optimizer_moments(opt, reset_flat)
        res["optimizer_reset_report"] = {
            "n_zeroed": len(optim_report["zeroed"]),
            "n_absent": len(optim_report["absent"]),
            "absent_indices": optim_report["absent"][:16],
            "unexpected_keys_seen": optim_report["unexpected_keys_seen"],
        }
        del shell_for_order
    else:
        res["notes"].append(
            "input ckpt has NO optimizer_state; the moment-reset half of the operator "
            "was not exercised and self-test (d) is not evaluable.")

    # ---- self-test (a): every non-reset tensor byte-identical -------------------
    sha_after = {k: _sha256_tensor(v) for k, v in ms.items()}
    nonreset = [k for k in ms if not (_layer_id(k) is not None and _layer_id(k) in reset_ids)]
    a_mismatch = [k for k in nonreset if sha_before[k] != sha_after[k]]
    reset_keys = [k for k in ms if k not in set(nonreset)]
    reset_changed = [k for k in reset_keys if sha_before[k] != sha_after[k]]
    res["selftests"]["a_nonreset_bytes_identical"] = {
        "pass": len(a_mismatch) == 0,
        "n_nonreset_tensors_checked": len(nonreset),
        "n_mismatched": len(a_mismatch),
        "mismatched_names": a_mismatch[:16],
        "n_reset_tensors": len(reset_keys),
        "n_reset_tensors_whose_sha_changed": len(reset_changed),
        "note": "a reset tensor whose sha did NOT change would mean the surgery was a "
                "no-op on it; reported so the test cannot pass vacuously.",
    }
    # Negative control: the reset tensors MUST have changed, otherwise (a) is trivially
    # satisfied by doing nothing at all.
    res["selftests"]["a_negative_control_reset_tensors_did_change"] = {
        "pass": len(reset_changed) == len(reset_keys) and len(reset_keys) > 0,
        "n_reset_tensors": len(reset_keys),
        "n_changed": len(reset_changed),
        "unchanged_names": [k for k in reset_keys if k not in set(reset_changed)][:16],
    }

    # ---- self-test (b): the trainer's own _assert_fresh_init -------------------
    # Load the surgical state into a shell and call the TRAINER's assert, not a copy of it.
    probe, _, _ = T.build_olmo2_minimal(
        args.model_path, keep_front, n_fresh, torch.float32,
        transplant=False, is_main=False)
    miss_b, unexp_b = probe.load_state_dict(ms, strict=True)
    b_err = None
    try:
        ln_ones, qn_ones, q_std = T._assert_fresh_init(probe, keep_front)
        b_pass = True
    except AssertionError as e:  # the trainer's own message is the evidence
        ln_ones = qn_ones = None
        q_std = None
        b_pass = False
        b_err = str(e)
    res["selftests"]["b_assert_fresh_init"] = {
        "pass": bool(b_pass),
        "checked_layer": keep_front,
        "post_attention_layernorm_all_ones": ln_ones,
        "q_norm_all_ones": qn_ones,
        "q_proj_weight_std": q_std,
        "band": "0.01 < std < 0.04",
        "assertion_error": b_err,
        "called": "train_olmo2_arch_probe2._assert_fresh_init (the trainer's own function, "
                  "not a reimplementation)",
        "post_norm_note": "OLMo-2 is POST-norm: the checked RMSNorm is "
                          "post_attention_layernorm, NOT input_layernorm (which does not "
                          "exist in this architecture).",
    }

    # ---- self-test (c): the REAL strict path -----------------------------------
    # Re-specified, see the module docstring. The prereg's literal --dry_run_build form is
    # unfireable (AST-verified), so we execute the statement the trainer executes at line
    # 776 with the surgical ckpt actually loaded.
    res["selftests"]["c_strict_load"] = {
        "pass": (miss_b == [] and unexp_b == []),
        "n_missing": len(miss_b),
        "n_unexpected": len(unexp_b),
        "missing": list(miss_b)[:8],
        "unexpected": list(unexp_b)[:8],
        "how": "build_olmo2_minimal(..., transplant=False).load_state_dict("
               "ckpt['model_state'], strict=True) -- the same statement the trainer runs "
               "at scripts/train_olmo2_arch_probe2.py:776 on the resume path.",
        "prereg_literal_form_was_unfireable": True,
        "prereg_literal_form_defect": (
            "GATE_PREREGISTRATION.md sec 8.2 (c) says 'strict-loads under --dry_run_build'. "
            "In main(), `if args.dry_run_build:` (line 714) CONTAINS A RETURN and precedes "
            "`if args.resume_from:` (line 752), so --dry_run_build never reads the "
            "checkpoint: the run prints '[dry_run_build] ... -> OK' regardless of whether "
            "the surgery produced garbage. AST evidence: "
            "evidence/g0_selftest_c_unfireable_20260817.json."),
    }
    del probe

    # ---- self-test (d): optimizer group count preserved ------------------------
    if "optimizer_state" in ck:
        n_after = len(ck["optimizer_state"]["param_groups"])
        vacuous = (n_ckpt_groups_before != 4)
        d_pass = (n_ckpt_groups_before == n_after == n_rebuilt_groups)
        res["selftests"]["d_optimizer_group_count_preserved"] = {
            "pass": bool(d_pass and not vacuous),
            "n_ckpt_groups_before": n_ckpt_groups_before,
            "n_ckpt_groups_after": n_after,
            "n_groups_the_trainer_rebuilds_on_resume": n_rebuilt_groups,
            "shim_would_trigger": bool(n_ckpt_groups_before == 2 and n_rebuilt_groups == 4),
            "vacuous": bool(vacuous),
            "vacuity_note": (
                "build_param_groups (line 480) DROPS empty groups and _classify_param routes "
                "every model.layers.{lid>=keep_front} param to 'fresh', so the rebuilt group "
                "count is a function of the FLAGS AND ARCH, not of the ckpt. (d) is only "
                "evidence if the INPUT ckpt is a genuine trainer-written 4-group ckpt: a "
                "2-group input would make the counts 'agree' without exercising anything. "
                "n_ckpt_groups_before != 4 is therefore reported as VACUOUS, not PASS."),
            "shim_ref": "scripts/train_olmo2_arch_probe2.py:912 "
                        "(elif n_ckpt_groups == 2 and n_new_groups == 4)",
        }
    else:
        res["selftests"]["d_optimizer_group_count_preserved"] = {
            "pass": None, "skipped": True,
            "reason": "input ckpt has no optimizer_state",
        }

    # ---- write the surgical ckpt ----------------------------------------------
    if args.out_ckpt and not args.no_keep_out:
        ck["b03_reset_provenance"] = {
            "operator": "LLF top-K layer reset (Zhou et al., ICLR 2022, mask M^l=1[l<K_f]); "
                        "NOT a novel method -- see RELATED_WORK.md sec 2.1 / sec 3.",
            "reset_layer_ids": sorted(reset_ids),
            "reset_from": "Olmo2ForCausalLM(cfg).post_init() via "
                          "train_olmo2_arch_probe2.build_olmo2_minimal(transplant=False)",
            "seed": args.seed,
            "source_ckpt": os.path.abspath(args.in_ckpt),
            "source_step": ck.get("step"),
            "tool": os.path.abspath(__file__),
            "written_utc": res["run_utc"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_ckpt)) or ".", exist_ok=True)
        torch.save(ck, args.out_ckpt)
        res["out_ckpt_bytes"] = os.path.getsize(args.out_ckpt)
        res["out_ckpt_written"] = True
    else:
        res["out_ckpt_written"] = False
        res["notes"].append("surgical ckpt not persisted (--no_keep_out or no --out_ckpt); "
                            "self-tests still ran against the in-memory surgical state.")

    assert not torch.cuda.is_initialized(), "CUDA was initialised; this must be a 0-GPU tool"
    res["elapsed_s"] = round(time.time() - t_start, 2)

    # ---- verdict --------------------------------------------------------------
    checks = res["selftests"]
    evaluable = {k: v for k, v in checks.items() if v.get("pass") is not None}
    failed = sorted(k for k, v in evaluable.items() if not v["pass"])
    res["all_evaluable_selftests_pass"] = (failed == [])
    res["failed_selftests"] = failed
    res["skipped_selftests"] = sorted(k for k, v in checks.items() if v.get("pass") is None)
    dv = checks.get("d_optimizer_group_count_preserved", {})
    res["verdict"] = (
        "G0_PASS" if res["all_evaluable_selftests_pass"] and not res["skipped_selftests"]
        else ("NOT_EXPRESSIBLE" if dv.get("pass") is False and not dv.get("vacuous")
              else "G0_INCOMPLETE")
    )
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_ckpt", required=True)
    p.add_argument("--out_ckpt", default="")
    p.add_argument("--no_keep_out", action="store_true",
                   help="run the self-tests but do not persist the surgical ckpt")
    p.add_argument("--model_path", required=True,
                   help="the OLMo-2 base whose config defines the fresh init "
                        "(1B: .../models/OLMo-2-0425-1B)")
    p.add_argument("--keep_front_layers", type=int, required=True)
    p.add_argument("--n_fresh_layers", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result_json", default="",
                   help="write the machine-readable result here (recommended: the "
                        "proposal's evidence/ dir)")
    args = p.parse_args()

    # Hard 0-GPU guarantee: make CUDA invisible before any torch device work.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    res = surgery(args)
    if args.result_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)) or ".", exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(res, f, indent=2, sort_keys=False)
        print(f"[g0] wrote {args.result_json}")
    print(json.dumps({k: res[k] for k in
                      ("verdict", "all_evaluable_selftests_pass", "failed_selftests",
                       "skipped_selftests", "reset_layer_ids", "n_tensors_replaced")},
                     indent=2))
    return 0 if res["all_evaluable_selftests_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
