#!/usr/bin/env python3
"""Gate for QCMem's d_bottle-width persist path (B01 blocking_dependency).

WHAT IS BEING TESTED
--------------------
`QCMemModel(persist_bottleneck_latent=True)` stops the funnel one op short at
WRITE and persists `act(down(h))` (width `d_bottle`) instead of the restored
`up(act(down(h)))` (width `hidden_size`), re-applying `up` at READ. Three things
must hold, and the third is the one that protects already-published numbers:

  1. EQUIVALENCE  — persist ON vs OFF must give the same logits. Reported as a
     MEASURED max-abs-diff, never asserted as "bit-identical" without the number.
  2. BYTES/TOKEN  — measured as `numel * element_size() / T` on the tensor WRITE
     actually returns, on both paths. B01's gate names "bytes/token of what is
     written to the store (not of the restored hidden)" as its mandatory reported
     quantity, so the number has to come off the real tensor, not from arithmetic.
  3. LEGACY REGRESSION — with the flag OFF (the default), every write/read/decode
     path must be BIT-IDENTICAL to the code as of the parent commit. Checked
     against the ACTUAL previous file, materialised from git into a temp module,
     not against a hand-written "reference" reimplementation (which would only
     test my belief about what the old code did). Run for the bottleneck arm AND
     for a `bottleneck_dim=0` vanilla arm.

WHY THE FIXTURE SHAPES ARE WHAT THEY ARE
----------------------------------------
`--fixture real` builds `hidden_size=4096 / d_bottle=512 / bf16`, i.e. the exact
width, ratio and dtype of the 8B endpoints this gate exists for
(`outputs/qwenbott_funnel_L12_d512`, `hidden_size 4096`, `bottleneck_dim 512`,
bf16) — only the layer COUNT is reduced, because the equivalence being tested is
a property of the funnel's split point and of the bf16 GEMM shapes, not of depth.
A `hidden_size=256` toy would exercise neither the real matmul shapes nor bf16
rounding, and a selftest over invented widths would prove nothing about the
tensors that actually get persisted. `--fixture tiny` additionally sweeps a small
model in fp32 + bf16 for speed.

Usage:
  python scripts/qcmem_bottleneck_persist_selftest.py                  # both fixtures, CPU
  python scripts/qcmem_bottleneck_persist_selftest.py --device cuda    # same on 1 GPU
  python scripts/qcmem_bottleneck_persist_selftest.py --fixture tiny   # fast subset
Exit code 0 iff every check passes. Capture it as `cmd > file; echo $?` — a pipe
would report the pipe's rc, not this script's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from transformers import LlamaConfig, LlamaForCausalLM          # noqa: E402

from scripts.semantic_bottleneck_model import BottleneckLayer   # noqa: E402
from src.memory.qcmem.qcmem_model import QCMemModel             # noqa: E402

REL_MODEL = "src/memory/qcmem/qcmem_model.py"
FAILURES: list = []
RESULTS: dict = {}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def load_previous_qcmem_module():
    """Import the pre-change `qcmem_model.py` from git as a separate module.

    `git show HEAD:<path>` — so the regression baseline is the file as committed,
    which is the only thing that can prove the default path did not move. If the
    change is already committed, HEAD is the new file and this test degenerates to
    comparing the new code with itself; the caller is told so explicitly rather
    than being handed a silent tautological PASS.
    """
    src = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{REL_MODEL}"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        return None, f"git show failed: {src.stderr.strip()[:200]}"
    text = src.stdout
    tautological = "persist_bottleneck_latent" in text
    tmp = tempfile.NamedTemporaryFile("w", suffix="_qcmem_prev.py", delete=False)
    tmp.write(text)
    tmp.close()
    spec = importlib.util.spec_from_file_location("qcmem_model_prev", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return (mod, "HEAD ALREADY CONTAINS the persist flag -> this comparison is "
                 "self-vs-self and proves nothing; re-run before committing"
            if tautological else ""), None


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def build_model(hidden, inter, n_layers, n_heads, n_kv, head_dim, vocab,
                b_layer, b_dim, dtype, device, seed=1234):
    """Random-init Llama of the given shape, optionally funnel-wrapped at b_layer.

    Deterministic: the same seed gives the same weights for the bottleneck and
    the vanilla arm, so an arm-to-arm comparison is not confounded by init.
    """
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=n_kv, head_dim=head_dim, hidden_act="silu",
        max_position_embeddings=4096, rope_theta=500000.0, rms_norm_eps=1e-5,
        tie_word_embeddings=False, attention_bias=False, attention_dropout=0.0,
    )
    model = LlamaForCausalLM(cfg).to(dtype)
    if b_dim and b_dim > 0:
        inner = model.model.layers[b_layer]
        model.model.layers[b_layer] = BottleneckLayer(inner, hidden, b_dim).to(dtype)
    return model.to(device).eval()


FIXTURES = {
    # Real width / ratio / dtype of outputs/qwenbott_funnel_L12_d512:
    # hidden_size 4096, bottleneck_dim 512 (8x), bf16. Depth reduced only.
    "real": dict(hidden=4096, inter=4096, n_layers=4, n_heads=32, n_kv=8,
                 head_dim=128, vocab=512, b_layer=2, dtypes=("bfloat16",),
                 b_dim=512, T=64, chunks=(48, 48), q=24),
    # Small sweep to exercise fp32 (where any rearrangement error would be
    # visible well below bf16's rounding floor) plus bf16 at a different ratio.
    "tiny": dict(hidden=256, inter=512, n_layers=6, n_heads=8, n_kv=4,
                 head_dim=32, vocab=257, b_layer=3, dtypes=("float32", "bfloat16"),
                 b_dim=32, T=40, chunks=(24, 24, 16), q=12),
}


def make_ids(spec, device, seed=7):
    g = torch.Generator().manual_seed(seed)
    hi = spec["vocab"]
    return {
        "seq": torch.randint(0, hi, (spec["T"],), generator=g).to(device),
        "chunks": [torch.randint(0, hi, (n,), generator=g).to(device)
                   for n in spec["chunks"]],
        "query": torch.randint(0, hi, (spec["q"],), generator=g).to(device),
        "sink": [1],
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def maxabs(a, b):
    """Max |a-b| in float64, plus the count of differing elements."""
    if a.shape != b.shape:
        return float("inf"), -1
    d = (a.double() - b.double()).abs()
    return float(d.max().item()), int((a != b).sum().item())


def bytes_per_token(h):
    """MEASURED bytes/token of a persisted [1, T, w] tensor."""
    return h.numel() * h.element_size() / h.shape[1]


def full_pipeline(qc, ids):
    """Every read-side output a caller can observe, from cached write pieces."""
    sink = qc.write_chunk(ids["sink"])
    ctx = [qc.write_chunk(c) for c in ids["chunks"]]
    q = qc.write_chunk(ids["query"])
    out = {
        "write_sink": sink, "write_q": q,
        "write_ctx_cat": torch.cat(ctx, dim=1),
        "read": qc.read(sink, ctx, q),
        "read_tail": qc.read_core(sink, ctx, q, logits_tail=4),
        "resume_ids": qc.resume_forward_ids(ids["seq"]),
        "write_chunks_cat": torch.cat(qc.write_chunks(ids["chunks"]), dim=1),
    }
    # decode fast path: prefill both bands then two O(1) steps
    q_hj, bottom, q_pos = qc.write_prefill(ids["query"])
    lg, top, pack = qc.read_prefill(sink, ctx, q_hj)
    outs = [lg]
    for k in range(2):
        tok = int(lg[0, -1].argmax().item())
        lg = qc.decode_step(tok, bottom, top, q_pos + k, pack + k)
        outs.append(lg)
    out["write_prefill_hj"] = q_hj
    out["decode"] = torch.cat(outs, dim=1)
    return out


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def run_fixture(fx_name, dtype_name, device, prev_mod, prev_warn):
    spec = FIXTURES[fx_name]
    dtype = getattr(torch, dtype_name)
    tag = f"{fx_name}/{dtype_name}"
    j = spec["b_layer"] + 1                 # resume_j == bottleneck_layer + 1
    print(f"\n=== fixture {tag}  hidden={spec['hidden']} d_bottle={spec['b_dim']} "
          f"layers={spec['n_layers']} funnel@{spec['b_layer']} resume_j={j} "
          f"device={device} ===")
    ids = make_ids(spec, device)
    res = RESULTS.setdefault(tag, {})

    def mk(b_dim):
        return build_model(spec["hidden"], spec["inter"], spec["n_layers"],
                           spec["n_heads"], spec["n_kv"], spec["head_dim"],
                           spec["vocab"], spec["b_layer"], b_dim, dtype, device)

    # ---------------- 1. legacy regression, bottleneck arm ------------------
    bmodel = mk(spec["b_dim"])
    new_off = full_pipeline(QCMemModel(bmodel, resume_j=j), ids)
    if prev_mod is not None:
        old = full_pipeline(prev_mod.QCMemModel(bmodel, resume_j=j), ids)
        worst = max((maxabs(new_off[k], old[k])[0], k) for k in old)
        flips = sum(maxabs(new_off[k], old[k])[1] for k in old)
        res["regress_bottleneck_maxabs"] = worst[0]
        res["regress_bottleneck_flips"] = flips
        check(f"[{tag}] default path == HEAD code (bottleneck arm, {len(old)} tensors)",
              worst[0] == 0.0 and flips == 0,
              f"max_abs={worst[0]:.3e} (worst {worst[1]}) differing_elems={flips}"
              + (f"  ⚠ {prev_warn}" if prev_warn else ""))

    # ---------------- 2. legacy regression, vanilla arm ---------------------
    # This is the arm that protects the repo's published QCMem numbers: no funnel
    # anywhere, so the flag is inapplicable and the default path must not move.
    vmodel = mk(0)
    v_new = full_pipeline(QCMemModel(vmodel, resume_j=j), ids)
    if prev_mod is not None:
        v_old = full_pipeline(prev_mod.QCMemModel(vmodel, resume_j=j), ids)
        worst = max((maxabs(v_new[k], v_old[k])[0], k) for k in v_old)
        flips = sum(maxabs(v_new[k], v_old[k])[1] for k in v_old)
        res["regress_vanilla_maxabs"] = worst[0]
        res["regress_vanilla_flips"] = flips
        check(f"[{tag}] default path == HEAD code (VANILLA bottleneck_dim=0)",
              worst[0] == 0.0 and flips == 0,
              f"max_abs={worst[0]:.3e} (worst {worst[1]}) differing_elems={flips}")

    # ---------------- 3. persist-path equivalence ---------------------------
    qc_on = QCMemModel(bmodel, resume_j=j, persist_bottleneck_latent=True)
    new_on = full_pipeline(qc_on, ids)
    read_keys = ["read", "read_tail", "resume_ids", "decode"]
    per_key = {}
    for k in read_keys:
        m, n = maxabs(new_off[k], new_on[k])
        per_key[k] = {"max_abs": m, "differing_elems": n,
                      "n": int(new_off[k].numel())}
    res["equivalence"] = per_key
    worst_m = max(v["max_abs"] for v in per_key.values())
    worst_k = max(per_key, key=lambda k: per_key[k]["max_abs"])
    tot_flips = sum(v["differing_elems"] for v in per_key.values())
    tot_n = sum(v["n"] for v in per_key.values())
    # bf16 has ~3 decimal digits; a rearrangement that is only mathematically
    # equal can land a few ulps out. Tolerance is scaled to the logit magnitude
    # so it means the same thing at both dtypes, and the MEASURED number is
    # printed either way -- the number is the deliverable, not the boolean.
    scale = max(float(new_off[worst_k].abs().max().item()), 1.0)
    tol = 1e-2 * scale if dtype is torch.bfloat16 else 1e-4 * scale
    check(f"[{tag}] persist ON == persist OFF (read-side logits)",
          worst_m <= tol,
          f"max_abs={worst_m:.6e} (worst {worst_k}, |logit|max={scale:.3f}, "
          f"tol={tol:.3e}) differing_elems={tot_flips}/{tot_n}")
    print(f"        per-tensor: " + "  ".join(
        f"{k}:{per_key[k]['max_abs']:.3e}/{per_key[k]['differing_elems']}"
        for k in read_keys))
    if worst_m == 0.0 and tot_flips == 0:
        print(f"        -> BIT-IDENTICAL (0 differing elements over {tot_n})")

    # ---------------- 4. bytes/token, MEASURED off the tensors --------------
    bt_off = bytes_per_token(new_off["write_q"])
    bt_on = bytes_per_token(new_on["write_q"])
    ratio = bt_off / bt_on
    want = spec["hidden"] / spec["b_dim"]
    res["bytes_per_token"] = {
        "legacy_measured": bt_off, "persist_measured": bt_on,
        "ratio_measured": ratio, "ratio_expected_hidden_over_dbottle": want,
        "legacy_shape": list(new_off["write_q"].shape),
        "persist_shape": list(new_on["write_q"].shape),
        "store_bytes_per_token_api": [QCMemModel(bmodel, resume_j=j).store_bytes_per_token(),
                                      qc_on.store_bytes_per_token()],
    }
    check(f"[{tag}] measured bytes/token ratio == hidden/d_bottle",
          abs(ratio - want) < 1e-9,
          f"{bt_off:.1f} -> {bt_on:.1f} B/tok = {ratio:.4f}x (expected {want:.4f}x); "
          f"shapes {tuple(new_off['write_q'].shape)} -> {tuple(new_on['write_q'].shape)}")
    api = res["bytes_per_token"]["store_bytes_per_token_api"]
    check(f"[{tag}] store_bytes_per_token() agrees with the measured tensors",
          api == [int(bt_off), int(bt_on)], f"api={api} measured=[{bt_off}, {bt_on}]")

    # ---------------- 5. write_chunks stays consistent with write_chunk -----
    # `write_chunks`' docstring GUARANTEES bit-identity with per-chunk
    # `write_chunk`, and the persist path must not degrade that. But the guarantee
    # is ALREADY BROKEN ON HEAD whenever a BottleneckLayer sits in the write band —
    # measured, and reproduced against git HEAD with this file's changes absent:
    #
    #   CUDA bf16, hidden 4096 / d512, funnel in band : 3764/393216 elems differ,
    #                                                   max_abs 1.56e-02
    #   CUDA bf16, SAME shapes, NO funnel             : 0/393216  (bit-identical)
    #   CPU  fp32, hidden 256  / d32,  funnel in band : 11218/16384, max_abs 5.59e-09
    #   CPU  fp32, SAME shapes, NO funnel             : 0/16384
    #
    # Mechanism, isolated to a bare op: `nn.Linear(4096, 512, bias=False)` in CUDA
    # bf16 called at B=2 differs from two B=1 calls in 82/49152 elements
    # (max_abs 3.91e-03) — a batched-vs-unbatched GEMM kernel/blocking difference.
    # The funnel's `down`/`up` are the only plain `nn.Linear`s in the band, which is
    # exactly why the divergence appears only when the funnel is present. It has
    # NOTHING to do with QCMem's chunk-locality argument (attention still never
    # mixes batch entries; the docstring's *semantic* claim is intact) and NOTHING
    # to do with this change. It does mean the word "bit-identical" in
    # `write_chunks`' docstring is too strong for funnel-wrapped backbones.
    #
    # So: measure BOTH paths, require the persist path's RELATIVE deviation not to
    # exceed the legacy path's by more than a dtype floor, and print the numbers.
    # Asserting bit-identity unconditionally would report a FAIL that is a property
    # of the BLAS/cuBLAS batched GEMM; asserting nothing would let a real
    # persist-path regression through.
    def batch_consistency(res_dict):
        per = res_dict["write_ctx_cat"]        # cat of per-chunk write_chunk
        bat = res_dict["write_chunks_cat"]     # cat of batched write_chunks
        m, n = maxabs(per, bat)
        scale = max(float(per.abs().max().item()), 1e-12)
        return {"max_abs": m, "differing_elems": n, "n": int(per.numel()),
                "max_abs_rel": m / scale, "tensor_absmax": scale}
    leg = batch_consistency(new_off)
    per = batch_consistency(new_on)
    res["write_chunks_vs_write_chunk"] = {"legacy": leg, "persist": per}
    rel_floor = 1e-6 if dtype is torch.float32 else 1e-3
    ok = per["max_abs_rel"] <= max(leg["max_abs_rel"], 0.0) + rel_floor
    check(f"[{tag}] persist write_chunks==write_chunk no worse than legacy's",
          ok,
          f"legacy max_abs={leg['max_abs']:.3e} ({leg['differing_elems']}/{leg['n']} "
          f"elems, rel={leg['max_abs_rel']:.3e}) | persist max_abs={per['max_abs']:.3e} "
          f"({per['differing_elems']}/{per['n']}, rel={per['max_abs_rel']:.3e}) "
          f"| rel_floor={rel_floor:.1e}")
    if leg["differing_elems"] == 0 and per["differing_elems"] == 0:
        print("        -> both BIT-IDENTICAL")
    elif leg["differing_elems"] > 0:
        print("        -> legacy is ALSO not bit-identical here: PRE-EXISTING, "
              "reproduced on git-HEAD code alone, and absent when the funnel is "
              "removed at identical shapes => batched-vs-unbatched GEMM in the "
              "funnel's nn.Linear, not a QCMem or persist-path defect")

    # ---------------- 6. negative controls ---------------------------------
    # The flag must REFUSE placements where deferring `up` would silently persist
    # full-width hiddens. A fallback here would report an 8x saving it did not get.
    def refuses(model, rj, why):
        try:
            QCMemModel(model, resume_j=rj, persist_bottleneck_latent=True)
        except ValueError as e:
            return True, str(e)[:90]
        return False, f"accepted {why} -- SILENT full-width persist"

    ok, d = refuses(vmodel, j, "a model with no funnel")
    check(f"[{tag}] refuses persist on bottleneck_dim=0 (no funnel)", ok, d)
    ok, d = refuses(bmodel, spec["b_layer"], "resume_j == bottleneck_layer")
    check(f"[{tag}] refuses resume_j == bottleneck_layer (funnel not last in band)",
          ok, d)
    ok, d = refuses(bmodel, spec["n_layers"], "resume_j past the funnel")
    check(f"[{tag}] refuses resume_j > bottleneck_layer+1 (band reads full width)",
          ok, d)
    ok, d = refuses(bmodel, 0, "resume_j == 0")
    check(f"[{tag}] refuses resume_j == 0 (empty write band)", ok, d)
    # A full-width piece handed to a persist-path read must be rejected, not
    # silently packed (that would be a correctness bug with no symptom).
    try:
        qc_on.read(new_off["write_sink"], [new_off["write_ctx_cat"]], new_off["write_q"])
        ok, d = False, "accepted full-width pieces on a persist read"
    except ValueError as e:
        ok, d = True, str(e)[:90]
    check(f"[{tag}] persist read rejects full-width cached pieces", ok, d)

    del bmodel, vmodel, qc_on
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fixture", default="all", choices=["all", "real", "tiny"])
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()

    torch.manual_seed(0)
    loaded, err = load_previous_qcmem_module()
    if err:
        prev_mod, prev_warn = None, err
        print(f"[WARN] no HEAD baseline: {err} -> regression checks SKIPPED")
    else:
        prev_mod, prev_warn = loaded
        print(f"[baseline] HEAD:{REL_MODEL} loaded as qcmem_model_prev"
              + (f"  ⚠ {prev_warn}" if prev_warn else ""))

    names = ["real", "tiny"] if a.fixture == "all" else [a.fixture]
    for fx in names:
        for dt in FIXTURES[fx]["dtypes"]:
            run_fixture(fx, dt, a.device, prev_mod, prev_warn)

    if prev_mod is None:
        FAILURES.append("HEAD baseline unavailable -> legacy regression NOT proven")
    print("\n" + "=" * 72)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"device": a.device, "failures": FAILURES,
                       "baseline_warning": prev_warn,
                       "results": RESULTS}, f, indent=1)
        print(f"wrote {a.json_out}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
