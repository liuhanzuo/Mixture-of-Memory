#!/usr/bin/env python3
"""Emit B12 rank-ladder variants of a SparseForge checkpoint. NO TRAINING, NO GPU.

*** THIS SCRIPT HAS NEVER BEEN EXECUTED. *** Its first run must be the rung-A
self-check described in proposal/backlog/B12-slorb-rank-efficiency/STATUS.json
(next_gate leg 2): rung A has an INDEPENDENTLY KNOWN answer, so it is a free
correctness gate before any GPU is requested.

    CUDA_VISIBLE_DEVICES="" python emit_slorb_ladder.py --rung A \
        --ckpt <5B ckpt> --output /tmp/x --dry-run
    # must print density=56.2500% and branch_live=404,750,336 EXACTLY

WHY THIS FILE EXISTS RATHER THAN A PATCH TO export_sparseforge_to_hf.py
----------------------------------------------------------------------
`export_sparseforge_to_hf.py` has four guards that are load-bearing and must not
be perturbed (they are quoted in status/sparseforge_union9_closeout.json
.sparsity_asymmetry.exporter_guards_are_real_not_rubber_stamps):

  :181  --slorb fold requested but SLoRB_Weight/x_proj missing -> hard exit
  :213  mask=hard slorb=drop must yield exact 2:4 -- refuses to write otherwise
  :215  folding SLoRB left the weight exactly 2:4 -> impossible -> investigate
  :217-235 key-set invariant vs the dense reference's safetensors index, then a
        real load_state_dict round-trip, then a POST-CAST 2:4 re-verification

Adding a `--slorb ladder` mode inside it would put a new code path *upstream* of
those guards on the checkpoint that produced the project's headline numbers.
So this is a SEPARATE, OPT-IN tool. What it does NOT do is reinvent the save
path: it reuses the exporter's verified sequence (AutoModelForCausalLM skeleton
-> load_state_dict with key-set assertion -> save_pretrained(safe_serialization)).

THE DEFECT IN THE PREDECESSOR THIS FIXES
----------------------------------------
`emit_small_slorb_variants.py` (also never executed) ends with

    torch.save(new_sd, out / "pytorch_model.bin")

i.e. a bare torch.save of a hand-built dict. That skips the key-set invariant,
the load_state_dict round-trip AND the post-cast verification. A key-naming or
dtype error would then surface as a silently mis-scored eval rather than a crash
-- the exact failure class that has already forced one retraction on this
project. This tool asserts the key set and round-trips through the real model.

OPERATOR: basis coarsening with an LS refit (NOT SVD of the composed product)
----------------------------------------------------------------------------
The deployed branch is E = SLoRB_Weight @ x_proj (sparse_modeling.py:818-822).

  r = in_features // SLoRB_k, with SLoRB_k = 16 MEASURED from the ckpt's own
  args (sparse_modeling.py:415-427). r is NOT a free hyperparameter: every
  projection is already at its maximum rank for k=16. The ONLY knob the trainer
  exposes for "a smaller SLoRB" is k. Coarsening k is therefore the faithful
  analogue of "a smaller SLoRB".

  x_proj's on-block entries are EXACTLY 1.0 (measured, 12/12 sampled
  projections), so its init B is a block-sum: (x @ B^T)[i] = sum of x over the
  i-th block of k. That is a segment-sum -- ZERO stored parameters. Reverting
  x_proj to B deletes 443,678,720 params.

  mode LS (default): after fixing the basis to B_coarse, re-fit the coefficients
  by least squares instead of just reusing the trained SLoRB_Weight. Measured
  over ALL 224 tensors, LS gives global ||dW_eff||/||W_eff|| = 0.024523 vs
  0.025603 for the naive revert (4.22% better) at identical parameter cost.

  Because the coarse basis columns are DISJOINT and constant-1 (B_coarse
  B_coarse^T = k_eff * I), the LS solution is available in closed form:
      argmin_S || S @ B_coarse - E ||_F  =>  S* = (E @ B_coarse^T) / k_eff
  i.e. the block MEAN of E's columns. NOT the block sum -- summing inflates the
  scale by k_eff. This is asserted numerically in _fit_blocksum_ls.

  mode svd (--rung Dctl): the density-matched control. t_p is chosen per
  projection so (out+in)*t ~= out*r_eff, matching the coarsen rung's parameter
  count. This is a GENUINE competitor, not a foregone loss: the repo's own
  op_matched_density_sample.json shows SVD winning the density-matched W_eff
  comparison in 12/35 projections at c=4 and 24/35 at c=16.

DENSITY IS THE DEPLOYMENT COUNT, AND THE EXPORT IS DENSE ON DISK
----------------------------------------------------------------
Every rung is written FOLDED (W*mask + E) because lm_eval only reads standard
HF weights. Folding is exact algebra (export_sparseforge_to_hf.py:62,:123,:283).
The manifest's density is the honest live-parameter count of the two-matmul
DEPLOYMENT form, NOT the folded dense tensor. Any table built from these
manifests must carry that distinction IN THE ROW.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
AUX_SUFFIXES = (".mask", ".hessian_diag", ".grad_ema", ".frozen_mask_flags",
                ".scaler_row", ".SLoRB_Weight", ".x_proj", ".cast_scale")

# ---------------------------------------------------------------------------
# THE FROZEN LADDER. Pre-registered 2026-08-16 in
# proposal/backlog/B12-slorb-rank-efficiency/STATUS.json .ladder_frozen_20260816.
# Keys are coarsening factors c per projection family; c=1 means "blocksum only,
# rank unchanged". Values were selected by an exhaustive scan of all 5^7 maps
# over c in {1,2,4,8,16} minimising global ||dW_eff||_F/||W_eff||_F at each psi.
# EXPECTED_* are independent arithmetic, and the tool ASSERTS against them --
# a ladder edit that changes the density silently is a protocol violation.
# ---------------------------------------------------------------------------
LADDER = {
    "A":    {"q_proj": 1, "k_proj": 1, "v_proj": 1,  "o_proj": 1,
             "gate_proj": 1, "up_proj": 1, "down_proj": 1},
    "P":    {"q_proj": 1, "k_proj": 1, "v_proj": 1,  "o_proj": 1,
             "gate_proj": 1, "up_proj": 1, "down_proj": 8},
    "Q":    {"q_proj": 1, "k_proj": 1, "v_proj": 1,  "o_proj": 8,
             "gate_proj": 1, "up_proj": 1, "down_proj": 16},
    "R":    {"q_proj": 1, "k_proj": 1, "v_proj": 4,  "o_proj": 8,
             "gate_proj": 1, "up_proj": 1, "down_proj": 16},
    "S":    {"q_proj": 1, "k_proj": 1, "v_proj": 16, "o_proj": 16,
             "gate_proj": 1, "up_proj": 4, "down_proj": 16},
    # Dctl uses the SAME map as P but the svd operator, so it lands at the same
    # parameter budget (326,098,944 vs 325,844,992 = 0.08% apart).
    "Dctl": {"q_proj": 1, "k_proj": 1, "v_proj": 1,  "o_proj": 1,
             "gate_proj": 1, "up_proj": 1, "down_proj": 8},
}
LADDER_OPERATOR = {"A": "coarsen", "P": "coarsen", "Q": "coarsen",
                   "R": "coarsen", "S": "coarsen", "Dctl": "svd"}
EXPECTED_BRANCH_LIVE = {"A": 404_750_336, "P": 325_844_992, "Q": 290_848_768,
                        "R": 265_682_944, "S": 189_661_184, "Dctl": 326_098_944}
EXPECTED_DENSITY_PCT = {"A": 56.2500, "P": 55.0316, "Q": 54.4912,
                        "R": 54.1026, "S": 52.9287, "Dctl": 55.0355}
EXPECTED_PSI = {"A": 0.5229, "P": 0.6159, "Q": 0.6572,
                "R": 0.6869, "S": 0.7765, "Dctl": 0.6156}
# Anchors, both ALREADY ON DISK -- this tool must never be asked to make them.
ANCHOR_R0_DENSITY_PCT = 63.1011      # hard_fold, union-9 62.4335, DENSE
ANCHOR_R1_DENSITY_PCT = 50.0         # hard_drop, union-9 57.0678, true 2:4
EXPECTED_SCOPE_ELEMENTS = 6_476_005_376
EXPECTED_SURVIVING = 3_238_002_688


def nm_2_4_hard(soft: torch.Tensor) -> torch.Tensor:
    """Exact port of sparse_modeling.py's nm_2_4 hard projection.

    Line-for-line identical to export_sparseforge_to_hf.py:83-104. The mask is
    computed ONCE per tensor from the same soft mask for every rung and is never
    re-derived, so no rung can differ from another by its mask.
    """
    N, M = 2, 4
    out_dim, in_dim = soft.shape
    in_full = (in_dim // M) * M
    hard = torch.ones_like(soft, dtype=soft.dtype)
    if in_full == 0:
        return hard
    grouped = soft.detach().float()[:, :in_full].view(out_dim, in_full // M, M)
    topi = torch.topk(grouped, k=N, dim=-1, largest=True).indices
    gm = torch.zeros_like(grouped, dtype=soft.dtype)
    gm.scatter_(-1, topi, 1.0)
    hard[:, :in_full] = gm.view(out_dim, in_full)
    return hard


def strip_prefix(key: str) -> str:
    """``model.model.layers.0.…`` -> ``model.layers.0.…`` (exporter:107-111)."""
    return key[len("model."):] if key.startswith("model.") else key


def _fit_blocksum_ls(E: torch.Tensor, in_dim: int, k_eff: int):
    """LS-optimal coefficients for the block-sum basis of block width k_eff.

    B_coarse[i, i*k_eff : (i+1)*k_eff] = 1 with DISJOINT support, so
    B_coarse @ B_coarse^T = k_eff * I and the normal equations collapse to
        S* = (E @ B_coarse^T) / k_eff = block MEAN of E's columns.
    Returns (S_star, r_eff). float64 throughout.
    """
    out_dim = E.shape[0]
    r_eff = in_dim // k_eff
    usable = r_eff * k_eff
    if usable != in_dim:
        # Not reachable for Llama-2 (4096 and 11008 are both divisible by every
        # k_eff in the frozen ladder) but a silent tail-drop would be a real bug.
        raise SystemExit(
            f"in_dim={in_dim} is not divisible by k_eff={k_eff}; refusing to "
            f"silently drop the {in_dim - usable}-column tail")
    S_star = E[:, :usable].view(out_dim, r_eff, k_eff).mean(dim=2)
    return S_star, r_eff


def _materialise_blocksum(S_eff: torch.Tensor, in_dim: int, k_eff: int) -> torch.Tensor:
    """E_hat = S_eff @ B_coarse, i.e. repeat each coefficient across its block."""
    return S_eff.repeat_interleave(k_eff, dim=1)[:, :in_dim]


def _branch_svd(S: torch.Tensor, X: torch.Tensor, t: int):
    """Rank-t SVD truncation of E = S @ X, exact, without forming an (out x in) SVD.

    With S = Qs Rs and X^T = Qx Rx, E = Qs (Rs Rx^T) Qx^T, so the SVD of the
    r x r core IS the SVD of E (the remaining min(out,in)-r singular values are
    exactly zero). Verified independently: svdvals(S@B) == sqrt(k)*svdvals(S) to
    1.5e-15 relative on this checkpoint.
    """
    S64, X64 = S.double(), X.double()
    Qs, Rs = torch.linalg.qr(S64, mode="reduced")
    Qx, Rx = torch.linalg.qr(X64.T.contiguous(), mode="reduced")
    U, sv, Vh = torch.linalg.svd(Rs @ Rx.T, full_matrices=False)
    t = max(1, min(int(t), sv.numel()))
    S_new = (Qs @ U[:, :t]) * sv[:t]
    X_new = (Qx @ Vh[:t].T).T
    energy = float((sv[:t] ** 2).sum() / (sv ** 2).sum())
    return S_new, X_new, energy, t


def _svd_t_density_matched(out_dim: int, in_dim: int, r_eff: int) -> int:
    """t such that (out+in)*t ~= out*r_eff -- the coarsen rung's param count.

    SVD must store BOTH factors, which is why a naive t=r/2 control costs
    424,214,528 params against 404,750,336 for the FULL-RANK blocksum branch
    (4.81% MORE than no reduction at all) and is dominated on both axes.
    Matching the budget is what makes the operator comparison meaningful.
    """
    return max(1, round(out_dim * r_eff / (out_dim + in_dim)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Emit a B12 SLoRB rank-ladder rung (CPU only; no training).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rung", required=True, choices=sorted(LADDER),
                    help="frozen ladder rung; see STATUS.json ladder_frozen_20260816")
    ap.add_argument("--coeffs", choices=["ls", "naive"], default="ls",
                    help="ls (default, measured 4.22%% better) re-fits coefficients "
                         "in the coarse basis; naive reuses trained SLoRB_Weight")
    ap.add_argument("--model", default="models/Llama--Llama2-7b")
    ap.add_argument("--project-root",
                    default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + assert the density/param bookkeeping and the "
                         "per-tensor manifest, then STOP without materialising or "
                         "writing any weights. This is the rung-A self-check mode.")
    ap.add_argument("--allow-density-mismatch", action="store_true",
                    help="do not use. Escape hatch if the ladder is deliberately "
                         "edited; it disables the pre-registration assertion.")
    args = ap.parse_args()

    cmap = LADDER[args.rung]
    operator = LADDER_OPERATOR[args.rung]
    root = Path(args.project_root)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path

    print(f"[b12] rung={args.rung} operator={operator} coeffs={args.coeffs} "
          f"dry_run={args.dry_run}")
    print(f"[b12] coarsen map: " +
          " ".join(f"{p.split('_')[0]}={cmap[p]}" for p in PROJECTIONS))

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = blob["model_state_dict"]
    ck_args = blob.get("args", {}) or {}
    k = int(ck_args.get("SLoRB_k", 16))
    if k != 16:
        raise SystemExit(
            f"this ladder's frozen constants were computed for SLoRB_k=16; the "
            f"checkpoint reports {k}. Recompute EXPECTED_* before proceeding.")
    print(f"[b12] ckpt iter_num={blob.get('iter_num')} SLoRB_k={k} "
          f"init_type={ck_args.get('SLoRB_init_type')} "
          f"trainable_projection={ck_args.get('trainable_projection')}")

    tgt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}[args.dtype]

    clean: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    surviving = branch_live = scope_elems = 0

    for key, val in sd.items():
        if any(key.endswith(s) for s in AUX_SUFFIXES):
            continue                                  # aux tensors never exported
        hf_key = strip_prefix(key)
        if not (key.endswith(".weight") and any(f".{p}." in key for p in PROJECTIONS)):
            if not args.dry_run:
                clean[hf_key] = val.to(tgt).clone()
            continue

        proj = next(p for p in PROJECTIONS if f".{p}." in key)
        base = key[: -len("weight")]
        W = sd[key].double()
        mask = nm_2_4_hard(sd[base + "mask"].float()).double()
        Wm = W * mask
        out_dim, in_dim = W.shape
        scope_elems += W.numel()
        surviving += int((mask != 0).sum())

        S = sd[base + "SLoRB_Weight"]
        X = sd[base + "x_proj"]
        r_nom = in_dim // k
        c = cmap[proj]
        r_eff = r_nom // c
        E_true = S.double() @ X.double()

        if operator == "svd":
            t = _svd_t_density_matched(out_dim, in_dim, r_eff)
            S_new, X_new, energy, t = _branch_svd(S, X, t)
            E_hat = S_new @ X_new
            live = int(S_new.numel() + X_new.numel())   # BOTH factors are stored
            row = {"operator": "svd", "t": t, "energy_retained": energy,
                   "xproj_stored": True}
        else:
            k_eff = k * c
            if args.coeffs == "ls":
                S_eff, r_eff2 = _fit_blocksum_ls(E_true, in_dim, k_eff)
            else:
                # naive: block MEAN of the trained SLoRB_Weight's columns. MEAN,
                # not sum -- summing inflates the scale by c.
                r_eff2 = r_nom // c
                S_eff = S.double().view(out_dim, r_eff2, c).mean(dim=2) if c > 1 \
                        else S.double()
            assert r_eff2 == r_eff, (r_eff2, r_eff)
            E_hat = _materialise_blocksum(S_eff, in_dim, k_eff)
            live = int(S_eff.numel())                  # x_proj is parameter-FREE
            row = {"operator": "coarsen", "c": c, "k_eff": k_eff,
                   "xproj_stored": False}

        dE = (E_hat - E_true)
        row.update({
            "tensor": key, "proj": proj, "out": out_dim, "in": in_dim,
            "nominal_rank": r_nom, "effective_rank": r_eff,
            "live_branch_params": live,
            "rel_fro_err_vs_trained_branch": float(dE.norm() / E_true.norm()),
            "rel_pert_of_deployed_W_eff": float(dE.norm() / (Wm + E_true).norm()),
            "rel_pert_if_branch_deleted": float(E_true.norm() / (Wm + E_true).norm()),
        })
        rows.append(row)
        branch_live += live
        if not args.dry_run:
            clean[hf_key] = (Wm + E_hat).to(tgt)       # fold; exact algebra

    # ---- bookkeeping + PRE-REGISTRATION ASSERTIONS -------------------------
    if scope_elems != EXPECTED_SCOPE_ELEMENTS:
        raise SystemExit(f"scope_elements={scope_elems:,} != "
                         f"{EXPECTED_SCOPE_ELEMENTS:,}; wrong checkpoint?")
    if surviving != EXPECTED_SURVIVING:
        raise SystemExit(f"surviving={surviving:,} != {EXPECTED_SURVIVING:,}; the "
                         f"2:4 mask is not what the union-9 rows were scored with")
    density = (surviving + branch_live) / scope_elems
    psi = (ANCHOR_R0_DENSITY_PCT - 100 * density) / \
          (ANCHOR_R0_DENSITY_PCT - ANCHOR_R1_DENSITY_PCT)
    print(f"[b12] scope={scope_elems:,} surviving={surviving:,} "
          f"branch_live={branch_live:,}")
    print(f"[b12] density={100*density:.4f}%  psi={psi:.4f}  "
          f"(expected {EXPECTED_DENSITY_PCT[args.rung]:.4f}% / "
          f"{EXPECTED_PSI[args.rung]:.4f})")

    exp_live = EXPECTED_BRANCH_LIVE[args.rung]
    exp_dens = EXPECTED_DENSITY_PCT[args.rung]
    live_ok = (branch_live == exp_live)
    dens_ok = abs(100 * density - exp_dens) < 5e-4
    if not (live_ok and dens_ok):
        msg = (f"PRE-REGISTRATION MISMATCH for rung {args.rung}: "
               f"branch_live {branch_live:,} vs expected {exp_live:,}; "
               f"density {100*density:.4f}% vs expected {exp_dens:.4f}%. "
               f"The frozen ladder in STATUS.json .ladder_frozen_20260816 does "
               f"not describe what this code produces -- one of them is wrong "
               f"and it must be resolved BEFORE any GPU is spent.")
        if args.allow_density_mismatch:
            print(f"[b12] WARNING (suppressed): {msg}")
        else:
            raise SystemExit(msg)
    else:
        print(f"[b12] OK pre-registration match: branch_live and density both exact")

    manifest = {
        "b12_rung": args.rung,
        "operator": operator,
        "coefficients": args.coeffs if operator == "coarsen" else "svd",
        "coarsen_map": cmap,
        "source_checkpoint": str(ckpt_path),
        "source_iter_num": blob.get("iter_num"),
        "SLoRB_k": k,
        "scope_elements": scope_elems,
        "surviving_2of4": surviving,
        "live_branch_params": branch_live,
        "density_two_matmul_deployment_form": density,
        "psi_density_points_given_back": psi,
        "expected_branch_live": exp_live,
        "expected_density_pct": exp_dens,
        "prereg_match": bool(live_ok and dens_ok),
        "density_note": (
            "Density counts the DEPLOYED two-matmul form: surviving 2:4 weights + "
            "live branch params. The exported tensor is FOLDED (W*mask + E) and is "
            "therefore DENSE ON DISK -- forced, because lm_eval only reads standard "
            "HF weights; folding is exact algebra (export_sparseforge_to_hf.py:62,"
            ":123,:283). Any table built from this manifest MUST carry that "
            "distinction IN THE ROW, not in a footnote."),
        "must_not_claim": [
            "That this is a training-time rank ablation. It is post-hoc surgery on "
            "a checkpoint trained with SLoRB_k=16. A model TRAINED with "
            f"SLoRB_k={16*max(cmap.values())} could reallocate capacity during "
            "training and land anywhere, plausibly better. This cannot bound that.",
            "Any placement of this rung in a 2:4 column. It is folded and dense.",
        ],
        "per_tensor": rows,
    }

    if args.dry_run:
        print("[b12] DRY RUN: no weights materialised, nothing written. "
              "Bookkeeping above is the deliverable.")
        print(json.dumps({kk: vv for kk, vv in manifest.items()
                          if kk != "per_tensor"}, indent=2))
        return 0

    # ---- REUSE THE EXPORTER'S VERIFIED SAVE PATH ---------------------------
    # (export_sparseforge_to_hf.py:217-258). This is the block whose absence is
    # the defect in emit_small_slorb_variants.py.
    from transformers import AutoModelForCausalLM

    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text())["weight_map"]
    ref = set(index)
    missing = {kk for kk in ref - set(clean) if not kk.endswith("rotary_emb.inv_freq")}
    extra = set(clean) - ref
    if missing or extra:
        raise SystemExit(f"key-set mismatch vs the dense reference: "
                         f"missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}")
    print("[b12] key set matches dense reference (modulo rotary inv_freq)")

    print(f"[b12] materialising reference skeleton ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=tgt, low_cpu_mem_usage=True, local_files_only=True)
    incompat = model.load_state_dict({kk: v.to(tgt) for kk, v in clean.items()},
                                     strict=False)
    still_missing = [kk for kk in incompat.missing_keys
                     if not kk.endswith("rotary_emb.inv_freq")]
    if incompat.unexpected_keys or still_missing:
        raise SystemExit(f"load_state_dict mismatch: "
                         f"unexpected={list(incompat.unexpected_keys)[:8]} "
                         f"missing={still_missing[:8]}")

    # Post-cast sanity. A folded export MUST NOT be exactly 2:4 -- if it is, the
    # branch was a no-op and the rung is meaningless (exporter guard :215).
    post_zeros = post_elems = post_viol = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or mod.weight.ndim != 2:
            continue
        if not any(name.endswith(p) for p in PROJECTIONS):
            continue
        w = mod.weight.detach().float()
        r, cc = w.shape
        post_elems += w.numel()
        post_zeros += int((w == 0).sum())
        nz = (w != 0).reshape(r, cc // 4, 4).sum(-1)
        post_viol += int((nz != 2).sum())
    zero_frac = post_zeros / post_elems
    print(f"[b12] post-cast: zero_frac={zero_frac:.9f} 2of4_violations={post_viol}")
    if post_viol == 0 and abs(zero_frac - 0.5) < 1e-6:
        raise SystemExit(
            "the folded export is EXACTLY 2:4, which is impossible unless the "
            "branch is a no-op -- investigate before trusting this rung "
            "(mirrors export_sparseforge_to_hf.py:215)")

    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path, safe_serialization=True)
    copied = []
    for fname in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"):
        src = model_path / fname
        if src.exists():
            shutil.copyfile(src, out_path / fname)
            copied.append(fname)
    if "tokenizer.model" not in copied:
        raise SystemExit(f"{model_path} has no tokenizer.model; refusing to export "
                         f"a broken tokenizer")
    print(f"[b12] copied tokenizer files verbatim: {copied}")

    manifest["post_cast_zero_fraction"] = zero_frac
    manifest["post_cast_2of4_violations"] = post_viol
    manifest["is_dense_on_disk"] = not (post_viol == 0 and abs(zero_frac - .5) < 1e-6)
    (out_path / "slorb_ladder_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[b12] wrote {out_path} (density {100*density:.4f}%, "
          f"branch_live {branch_live:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
