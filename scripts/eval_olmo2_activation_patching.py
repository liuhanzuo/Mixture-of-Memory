#!/usr/bin/env python3
"""Paper B P2.2 — causal-layer restoration / activation patching (OLMo-2 prune-heal).

PURE FORWARD INFERENCE. NO TRAINING of any kind — every model here is either
(a) the pretrained base, (b) the healed keep14 ckpt, or (c) a *composition* of
the two whose weights are copied verbatim; nothing is optimised.

Question this answers
---------------------
keep14 = the first 14 layers inherited from vanilla OLMo-2-7B (32L base) + 2 FRESH
tail blocks (16-layer shell), healed to 200k. Its MMLU is low. Two competing
explanations:
  (a) READOUT-LIMITED: the front-14 layers already carry the needed information,
      but the fresh tail cannot read it out; or
  (b) COMPUTATION-DELETED: the required computation lived in the deleted upper
      layers (base layers 14..31) and is simply gone.
This harness runs two non-training interventions that dissociate (a) vs (b), and
reports BOTH held-out PPL and full-MMLU (14,042 items) for every intervention
(TODOList hard requirement — never a single metric).

Intervention 1 — boundary hidden-state grafting  (--mode graft)
--------------------------------------------------------------
Run the base 32L forward over the SAME token batch, capture the residual stream
AT THE OUTPUT of base layer L (a forward hook on base.model.layers[L]; OLMo-2
decoder layers return the post-layer residual as a plain tensor, verified against
transformers 5.13 modeling_olmo2.py line 334). Then run keep14 over the same
batch, but a forward PRE-hook on keep14.model.layers[J] (J defaults to
keep_front_layers = 14 = the fresh-tail input) REPLACES layer J's input hidden
with the captured base hidden. keep14's tail (layers 14,15) + final norm + lm_head
then read out. Scan L over {13,14,...,31}.
  * L == keep_front-1 (=13): base's front-14 representation -> keep14 tail (a
    near-in-distribution control: keep14's healed front ~= base's front).
  * L >  13: hands the tail a representation that ALREADY contains deleted-upper-
    layer computation. If MMLU jumps as L grows -> the upper-layer computation is
    what is missing AND keep14's tail CAN read it when present (favours (b)+ tail
    is capable). If it stays flat -> the info was already at the boundary and the
    readout is the bottleneck (favours (a)).
Positions/RoPE align exactly: both models tokenise identically and OLMo-2 computes
position_embeddings once from position_ids (theta 5e5), independent of layer count.

Controls for intervention 1 (all forward-only, same protocol):
  * IDENTITY PATCH   --graft_donor keep14 --graft_layer <keep_front-1>: the donor
    is a second copy of the recipient, so the captured activation IS the tensor it
    replaces -> must be a BIT-EXACT no-op. This is the HARNESS VALIDITY GATE; run
    it with --gate_check (real 7B weights, real val tokens, logits level).
  * PATCH-NOTHING    --graft_layer -1: hooks disabled -> plain keep14 numbers.
  * RANDOM PATCH     --graft_random {shuffle_positions,gauss_matched}: the donated
    activation is replaced by a statistics-matched, information-free surrogate ->
    the metric floor for "the patched pathway is being used at all".

Intervention 2 — progressive upper-layer restoration  (--mode restore)
---------------------------------------------------------------------
Assemble a hybrid Olmo2ForCausalLM by COPYING weights (no training):
  tail_keep14 (default; "restore before the tail", matches the TODOList wording):
      layers[0..13]            <- keep14 healed front layers
      layers[14..14+k-1]       <- base ORIGINAL upper layers 14..14+k-1 (restored)
      layers[14+k, 15+k]       <- keep14 FRESH tail layers (kept on top)
      embed / norm / lm_head   <- keep14
    k=0 -> byte-identical to keep14 (built-in identity check); k=18 -> keep14
    front + all 18 restored upper layers + fresh tail (34L).
  base_head (confound-free variant): drop the fresh tail, read out through base's
      ORIGINAL norm + lm_head:
      layers[0..13]            <- keep14 healed front
      layers[14..14+k-1]       <- base upper 14..14+k-1
      embed <- keep14 ; norm+lm_head <- base
    k=n_deleted (=18) -> keep14 healed front + full base upper stack + base's own
    readout (isolates hypothesis (b) with no fresh-tail readout confound).
Scan k. A rising PPL/MMLU curve with restored depth => the deleted computation was
necessary (favours (b)); a flat curve => it was redundant (favours (a)).

Scoring口径 (NO drift): reuses eval_olmo2_probe2_ppl.score_windows /
eval_olmo2_mmlu_content.{load_mmlu_examples,score_examples,aggregate,merge} and
eval_olmo2_probe2_ppl.merge_shards verbatim, so PPL and letter/content-MMLU are
byte-identical to the depth-ladder harnesses. BASE protocol: add_special_tokens=
False (--add_bos 0), chat_template=False, zero-shot, greedy, MMLU LL-MC, n=14042.

Modes: --mode {graft,restore,plain}; --task {ppl,mmlu}. Sharding + shard-file
layout + --merge + --prepare_data mirror the reused harnesses exactly. --selftest
runs a tiny CPU end-to-end validation (no 7B weights, no GPU): hook capture/inject
shapes, the graft no-op identity, and the restore composition + k=0 identity.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NO口径 drift: reuse the depth-ladder harnesses' loaders / scorers / mergers.
from eval_olmo2_probe2_ppl import (  # noqa: E402
    _log,
    load_base_model,
    load_pruned_model,
    score_windows,
    merge_shards,
)
from eval_olmo2_mmlu_content import (  # noqa: E402
    load_mmlu_examples,
    score_examples,
    aggregate,
    merge as mmlu_merge,
)


# ===========================================================================
# shell builder (arbitrary layer count) + two-source state-dict loader
# ===========================================================================
def build_shell_n(base_path, n_layers, dtype):
    """Instantiate an Olmo2ForCausalLM with num_hidden_layers=n_layers, config
    otherwise copied from the pretrained base. No weight transplant here; the
    caller strict-loads a fully composed state_dict."""
    cfg = Olmo2Config.from_pretrained(base_path, local_files_only=True)
    cfg.num_hidden_layers = int(n_layers)
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * int(n_layers)
        assert len(cfg.layer_types) == int(n_layers)
    model = Olmo2ForCausalLM(cfg).to(dtype)
    return model, cfg


def _load_base_state_dict(base_path):
    """fp32 CPU state_dict of the full pretrained base + its layer count."""
    base = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32, local_files_only=True
    )
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    n = base.config.num_hidden_layers
    del base
    gc.collect()
    return sd, n


def _load_keep_state_dict(ckpt_path, cli_keep_front, cli_n_fresh):
    """keep14 ckpt state_dict + arch meta (keep_front / n_fresh from the ckpt,
    CLI must agree if given)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError(f"ckpt {ckpt_path} missing 'model_state'")
    sd = ck["model_state"]
    kf = ck.get("keep_front_layers", cli_keep_front)
    nf = ck.get("n_fresh_layers", cli_n_fresh)
    if cli_keep_front is not None and kf is not None and int(cli_keep_front) != int(kf):
        raise ValueError(f"--keep_front_layers={cli_keep_front} != ckpt meta {kf}")
    if cli_n_fresh is not None and nf is not None and int(cli_n_fresh) != int(nf):
        raise ValueError(f"--n_fresh_layers={cli_n_fresh} != ckpt meta {nf}")
    if kf is None:
        raise ValueError("keep_front_layers unknown (not in ckpt meta, none passed)")
    if nf is None:
        nf = 2
    return sd, int(kf), int(nf), ck.get("step")


# ===========================================================================
# Intervention 2 — progressive upper-layer restoration (weight composition)
# ===========================================================================
def _sub_layer(sd, idx):
    """{suffix: tensor} for model.layers.{idx}.* in state_dict sd."""
    pref = f"model.layers.{idx}."
    return {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}


def compose_restore_state_dict(base_sd, base_nl, keep_sd, keep_front, n_fresh,
                               k, readout):
    """Compose the restoration state_dict for restore-level k. Returns
    (target_sd, n_layers). Pure key remapping + tensor references (no math)."""
    n_deleted = base_nl - keep_front  # base upper layers keep_front..base_nl-1
    if not (0 <= k <= n_deleted):
        raise ValueError(f"restore_k={k} out of range [0,{n_deleted}]")
    tgt = {}
    # (1) front: keep14 healed layers 0..keep_front-1
    for i in range(keep_front):
        for suf, v in _sub_layer(keep_sd, i).items():
            tgt[f"model.layers.{i}.{suf}"] = v
    # (2) restored upper: base ORIGINAL layers keep_front..keep_front+k-1
    for j in range(k):
        bidx = keep_front + j
        for suf, v in _sub_layer(base_sd, bidx).items():
            tgt[f"model.layers.{bidx}.{suf}"] = v  # target idx == base idx here
    if readout == "tail_keep14":
        # (3) keep14 fresh tail moved on top: keep layers keep_front..+n_fresh-1
        for t in range(n_fresh):
            kidx = keep_front + t
            tidx = keep_front + k + t
            for suf, v in _sub_layer(keep_sd, kidx).items():
                tgt[f"model.layers.{tidx}.{suf}"] = v
        tgt["model.embed_tokens.weight"] = keep_sd["model.embed_tokens.weight"]
        tgt["model.norm.weight"] = keep_sd["model.norm.weight"]
        tgt["lm_head.weight"] = keep_sd["lm_head.weight"]
        n_layers = keep_front + k + n_fresh
    elif readout == "base_head":
        # no fresh tail; read out through base's ORIGINAL norm + lm_head.
        tgt["model.embed_tokens.weight"] = keep_sd["model.embed_tokens.weight"]
        tgt["model.norm.weight"] = base_sd["model.norm.weight"]
        tgt["lm_head.weight"] = base_sd["lm_head.weight"]
        n_layers = keep_front + k
    else:
        raise ValueError(f"unknown readout {readout}")
    return tgt, n_layers


def build_restore_model(base_path, base_sd, base_nl, keep_sd, keep_front,
                        n_fresh, k, readout, device, shell_builder=build_shell_n):
    tgt, n_layers = compose_restore_state_dict(
        base_sd, base_nl, keep_sd, keep_front, n_fresh, k, readout)
    model, cfg = shell_builder(base_path, n_layers, torch.float32)
    missing, unexpected = model.load_state_dict(tgt, strict=True)
    assert not missing and not unexpected, (
        f"restore compose mismatch: missing={missing[:6]} unexpected={unexpected[:6]}"
    )
    model.config.use_cache = False
    model = model.to(device)
    model.eval()
    meta = {
        "mode": "restore",
        "restore_k": k,
        "restore_readout": readout,
        "keep_front_layers": keep_front,
        "n_fresh_layers": n_fresh,
        "base_num_layers": base_nl,
        "n_deleted_upper": base_nl - keep_front,
        "num_hidden_layers": n_layers,
    }
    _log(f"[restore] k={k} readout={readout} -> {n_layers}L "
         f"(front{keep_front}+base_upper{k}"
         f"{'+fresh'+str(n_fresh) if readout=='tail_keep14' else '+base_head'}) "
         f"strict-loaded {len(tgt)} tensors")
    return model, meta


# ===========================================================================
# Intervention 1 — boundary hidden-state grafting (base -> keep14 tail)
# ===========================================================================
class GraftedModel(torch.nn.Module):
    """Wrap (donor, recipient=keep14 16L). forward(input_ids, attention_mask):
    run the DONOR to capture the residual AT THE OUTPUT of donor layer
    `graft_layer`, then run the recipient with that hidden INJECTED as the input
    of recipient layer `inject_at` (default = keep_front = fresh-tail input).
    Returns the recipient's CausalLMOutput (has .logits), so the reused
    score_windows / score_examples work unchanged.

    Controls
    --------
    * graft_layer < 0            -> graft disabled ("patch-nothing" baseline: pure
                                    keep14 through the wrapper; must reproduce the
                                    plain keep14 numbers).
    * donor is a SECOND COPY of the recipient (`--graft_donor keep14`) with
      graft_layer == inject_at-1  -> mathematically an IDENTITY PATCH: the
      captured tensor IS byte-for-byte the tensor it replaces, so logits must be
      bit-identical to plain keep14. This is the harness VALIDITY GATE.
    * random_mode != "none"      -> the captured hidden is replaced by an
      information-free surrogate with matched activation statistics
      (`shuffle_positions`: exact same activation multiset, permuted along the
      sequence axis; `gauss_matched`: Gaussian with the captured tensor's
      per-(batch,position) mean/std). This is the random-activation FLOOR."""

    def __init__(self, base, keep, graft_layer, inject_at,
                 random_mode="none", random_seed=0):
        super().__init__()
        self.base = base
        self.keep = keep
        self.graft_layer = int(graft_layer)
        self.inject_at = int(inject_at)
        self.random_mode = str(random_mode)
        self.random_seed = int(random_seed)
        if self.random_mode not in ("none", "shuffle_positions", "gauss_matched"):
            raise ValueError(f"unknown random_mode {self.random_mode}")
        self._captured = None
        self._call_idx = 0            # makes the random surrogate reproducible

        n_keep = keep.config.num_hidden_layers
        n_base = base.config.num_hidden_layers
        if self.graft_layer >= 0:
            if not (0 <= self.graft_layer < n_base):
                raise ValueError(f"graft_layer {self.graft_layer} outside [0,{n_base})")
            if not (0 <= self.inject_at < n_keep):
                raise ValueError(f"inject_at {self.inject_at} outside [0,{n_keep})")
            if base is keep:
                raise ValueError(
                    "donor and recipient must be DISTINCT module instances "
                    "(the recipient carries an inject pre-hook, so calling it as "
                    "the donor would fire 'inject before capture')")
            base.model.layers[self.graft_layer].register_forward_hook(self._capture)
            keep.model.layers[self.inject_at].register_forward_pre_hook(
                self._inject, with_kwargs=True)

    # forward hook on donor layer L: store its output residual (plain tensor).
    def _capture(self, module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        self._captured = h.detach()
        return None

    def _randomise(self, cap):
        """Information-free surrogate with matched activation statistics."""
        g = torch.Generator(device="cpu")
        g.manual_seed(self.random_seed * 1000003 + self._call_idx)
        if self.random_mode == "shuffle_positions":
            # exact same activation values, permuted along the sequence axis
            # (destroys token<->position content alignment, preserves the
            #  per-batch activation multiset and hence all moments/norms).
            T = cap.shape[1]
            perm = torch.randperm(T, generator=g).to(cap.device)
            return cap.index_select(1, perm)
        # gauss_matched: N(mu, sigma) matched per (batch, position) over hidden dim
        mu = cap.mean(dim=-1, keepdim=True)
        sd = cap.std(dim=-1, keepdim=True)
        noise = torch.randn(cap.shape, generator=g, dtype=torch.float32).to(cap.device)
        return (noise.to(cap.dtype) * sd + mu)

    # forward PRE-hook on recipient layer J: replace hidden_states (pos. arg 0).
    def _inject(self, module, args, kwargs):
        if self._captured is None:
            raise RuntimeError("inject before capture (donor forward did not run)")
        cur = args[0]
        cap = self._captured
        assert cap.shape == cur.shape, (
            f"graft shape {tuple(cap.shape)} != keep-layer{self.inject_at} input "
            f"{tuple(cur.shape)}")
        cap = cap.to(dtype=cur.dtype, device=cur.device)
        if self.random_mode != "none":
            cap = self._randomise(cap)
        return (cap,) + tuple(args[1:]), kwargs

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        if self.graft_layer >= 0:
            self._captured = None
            with torch.no_grad():
                _ = self.base(input_ids=input_ids, attention_mask=attention_mask)
            assert self._captured is not None, "capture hook did not fire"
        out = self.keep(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
        self._call_idx += 1
        return out


def _load_donor(donor, base_path, ckpt_path, cli_keep_front, cli_n_fresh, device):
    """donor='base' -> pretrained 32L OLMo-2; donor='keep14' -> a SECOND,
    independent instance of the healed keep14 ckpt (identity-patch control)."""
    if donor == "base":
        return load_base_model(base_path, device)
    if donor == "keep14":
        m, meta = load_pruned_model(
            ckpt_path, base_path, cli_keep_front, cli_n_fresh, device)
        meta = dict(meta)
        meta["mode"] = "donor_keep14"
        return m, meta
    raise ValueError(f"unknown graft_donor {donor}")


def build_graft_model(base_path, ckpt_path, cli_keep_front, cli_n_fresh,
                      graft_layer, inject_at, device, donor="base",
                      random_mode="none", random_seed=0):
    donor_model, _bmeta = _load_donor(
        donor, base_path, ckpt_path, cli_keep_front, cli_n_fresh, device)
    keep, _kmeta = load_pruned_model(
        ckpt_path, base_path, cli_keep_front, cli_n_fresh, device)
    kf = _kmeta["keep_front_layers"]
    nf = _kmeta["n_fresh_layers"]
    if inject_at is None:
        inject_at = kf  # fresh-tail input
    gm = GraftedModel(donor_model, keep, graft_layer, inject_at,
                      random_mode=random_mode, random_seed=random_seed)
    gm.eval()
    is_identity = (donor == "keep14" and graft_layer == inject_at - 1
                   and random_mode == "none")
    meta = {
        "mode": "graft",
        "graft_donor": donor,
        "graft_layer": graft_layer,
        "inject_at_layer": inject_at,
        "graft_random": random_mode,
        "graft_random_seed": random_seed,
        "is_identity_patch": bool(is_identity),
        "keep_front_layers": kf,
        "n_fresh_layers": nf,
        "donor_num_layers": donor_model.config.num_hidden_layers,
        "keep_num_layers": keep.config.num_hidden_layers,
        "ckpt_step": _kmeta.get("ckpt_step"),
    }
    _log(f"[graft] donor={donor} L{graft_layer}_out -> keep_layer{inject_at}_in "
         f"random={random_mode} identity={is_identity} "
         f"(keep_front={kf} n_fresh={nf}); both models on {device}")
    return gm, meta


# ===========================================================================
# HARNESS VALIDITY GATE (real 7B weights, real val tokens, logits level)
# ===========================================================================
def run_gate_check(args, device):
    """Bit-level validity gate for the graft hooks, on the REAL keep14 weights.

    Builds (a) a hook-free plain keep14 and (b) an IDENTITY-PATCH GraftedModel
    whose donor is a second copy of keep14, capturing layer (keep_front-1)'s
    output and injecting it at layer keep_front. That substitution is a
    mathematical no-op, so the two must produce BIT-IDENTICAL logits. Also
    reports the same-batch NLL for both, and (as a wiring positive control) the
    max|delta| for a base-donor graft and for a random-activation patch, which
    MUST be non-zero.

    Exit status is non-zero if the gate fails, so drivers can hard-stop."""
    kf_cli, nf_cli = args.keep_front_layers, args.n_fresh_layers
    plain, pmeta = load_pruned_model(
        args.keep14_ckpt, args.base_model, kf_cli, nf_cli, device)
    kf = pmeta["keep_front_layers"]
    inject_at = args.inject_at_layer if args.inject_at_layer is not None else kf

    arr = np.load(args.val_path, mmap_mode="r")
    n_win = max(1, int(args.gate_windows))
    windows = np.array(arr[:n_win]).astype(np.int64)
    input_ids = torch.from_numpy(windows).to(device)
    _log(f"[gate] batch = first {n_win} val window(s) of {args.val_path} "
         f"shape={tuple(input_ids.shape)}")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ref = plain(input_ids=input_ids).logits.float()
    ref_nll = _nll_of(ref, input_ids)
    del plain
    gc.collect()
    torch.cuda.empty_cache()

    report = {"gate": "identity_patch", "val_path": args.val_path,
              "n_windows": n_win, "seq_len": int(input_ids.shape[1]),
              "keep_front_layers": kf, "inject_at_layer": inject_at,
              "ckpt": args.keep14_ckpt, "plain_keep14_nll": ref_nll}

    # ---- (1) THE GATE: donor == recipient, capture L=kf-1, inject at kf ----
    gm, gmeta = build_graft_model(
        args.base_model, args.keep14_ckpt, kf_cli, nf_cli,
        kf - 1, inject_at, device, donor="keep14", random_mode="none")
    assert gmeta["is_identity_patch"], gmeta
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        idt = gm(input_ids=input_ids).logits.float()
    d = (idt - ref).abs().max().item()
    idt_nll = _nll_of(idt, input_ids)
    report["identity_max_abs_logit_delta"] = d
    report["identity_patch_nll"] = idt_nll
    report["identity_nll_delta"] = idt_nll - ref_nll
    report["identity_exact_bitwise"] = bool(torch.equal(idt, ref))
    _log(f"[gate] IDENTITY PATCH (donor=keep14 L{kf-1}->inject{inject_at}): "
         f"max|dlogit|={d:.3e} bitwise_equal={report['identity_exact_bitwise']} "
         f"nll {ref_nll:.6f} -> {idt_nll:.6f} (d={idt_nll-ref_nll:+.3e})")
    del gm
    gc.collect()
    torch.cuda.empty_cache()

    # ---- (2) positive control: random-activation patch MUST change logits ----
    gr, _ = build_graft_model(
        args.base_model, args.keep14_ckpt, kf_cli, nf_cli,
        kf - 1, inject_at, device, donor="keep14",
        random_mode="shuffle_positions", random_seed=args.graft_random_seed)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        rnd = gr(input_ids=input_ids).logits.float()
    report["random_patch_max_abs_logit_delta"] = (rnd - ref).abs().max().item()
    report["random_patch_nll"] = _nll_of(rnd, input_ids)
    _log(f"[gate] random(shuffle_positions) patch: "
         f"max|dlogit|={report['random_patch_max_abs_logit_delta']:.4f} "
         f"nll={report['random_patch_nll']:.4f} (plain {ref_nll:.4f})")
    del gr
    gc.collect()
    torch.cuda.empty_cache()

    # ---- (3) positive control: base-donor graft MUST change logits ----
    gb, _ = build_graft_model(
        args.base_model, args.keep14_ckpt, kf_cli, nf_cli,
        kf - 1, inject_at, device, donor="base", random_mode="none")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        bse = gb(input_ids=input_ids).logits.float()
    report["base_donor_L%d_max_abs_logit_delta" % (kf - 1)] = \
        (bse - ref).abs().max().item()
    report["base_donor_L%d_nll" % (kf - 1)] = _nll_of(bse, input_ids)
    _log(f"[gate] base-donor L{kf-1} graft: "
         f"max|dlogit|={report['base_donor_L%d_max_abs_logit_delta' % (kf-1)]:.4f} "
         f"nll={report['base_donor_L%d_nll' % (kf-1)]:.4f}")
    del gb
    gc.collect()
    torch.cuda.empty_cache()

    ok_identity = report["identity_exact_bitwise"] or d <= 1e-6
    ok_random = report["random_patch_max_abs_logit_delta"] > 1e-3
    ok_base = report["base_donor_L%d_max_abs_logit_delta" % (kf - 1)] > 1e-3
    report["PASS_identity_is_noop"] = bool(ok_identity)
    report["PASS_random_patch_changes_logits"] = bool(ok_random)
    report["PASS_base_donor_changes_logits"] = bool(ok_base)
    report["GATE_PASS"] = bool(ok_identity and ok_random and ok_base)

    os.makedirs(args.results_root, exist_ok=True)
    out = os.path.join(args.results_root,
                       (args.output_name or "gate_check") + ".json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    _log(f"[gate] {'PASS' if report['GATE_PASS'] else 'FAIL'} -> {out}")
    if not report["GATE_PASS"]:
        _log("[gate] FAIL detail: " + json.dumps(
            {k: v for k, v in report.items() if k.startswith("PASS_")}))
        sys.exit(3)


def _nll_of(logits, input_ids):
    """Mean teacher-forced NTP NLL of `logits` against `input_ids` (same口径 as
    score_windows, just averaged instead of summed)."""
    import torch.nn.functional as F
    lg = logits[:, :-1, :]
    tg = input_ids[:, 1:].contiguous()
    return float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                                 reduction="mean").item())


# ===========================================================================
# scoring drivers (reuse the depth-ladder scorers; just supply the model)
# ===========================================================================
def run_ppl_shard(model, args, meta):
    arr = np.load(args.val_path, mmap_mode="r")
    windows_all = np.array(arr)
    assert windows_all.ndim == 2, windows_all.shape
    n_total = windows_all.shape[0]
    idx = np.arange(args.shard_index, n_total, args.num_shards)
    shard_windows = windows_all[idx]
    if args.limit and args.limit > 0:
        shard_windows = shard_windows[: args.limit]
    device = next(model.parameters()).device
    _log(f"[ppl] val={args.val_path} shape={windows_all.shape} shard "
         f"{args.shard_index}/{args.num_shards} -> {shard_windows.shape[0]} windows "
         f"bs={args.batch_size}")
    t0 = time.time()
    sum_nll, n_tokens, n_windows = score_windows(
        model, shard_windows, device, args.batch_size)
    dt = time.time() - t0
    assert n_tokens > 0, "empty shard"
    import math
    ppl_shard = math.exp(sum_nll / n_tokens)
    assert math.isfinite(ppl_shard), f"non-finite ppl_shard={ppl_shard}"
    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"shard{args.shard_index}of{args.num_shards}.json")
    meta["val_path"] = args.val_path
    with open(out, "w") as f:
        json.dump({
            "sum_nll": sum_nll, "n_tokens": n_tokens, "n_windows": n_windows,
            "ppl_shard": ppl_shard, "avg_nll": sum_nll / n_tokens,
            "shard_index": args.shard_index, "num_shards": args.num_shards,
            "seconds": dt, "meta": meta,
        }, f, indent=2)
    _log(f"[ppl shard {args.shard_index}/{args.num_shards}] n_windows={n_windows} "
         f"n_tokens={n_tokens} ppl_shard={ppl_shard:.4f} ({dt:.1f}s) -> {out}")


def run_mmlu_shard(model, tok, args, meta):
    device = next(model.parameters()).device
    bos_id = tok.bos_token_id
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    examples = load_mmlu_examples(args.content_desc)
    examples = examples[args.shard_index::args.num_shards]
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]
    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)
    t0 = time.time()
    records, n_trunc = score_examples(
        model, tok, examples, device, args.batch_size, bool(args.add_bos),
        bos_id, pad_id, args.max_len,
        shard_index=args.shard_index, num_shards=args.num_shards)
    dt = time.time() - t0
    shard_agg = aggregate(records, do_subjects=False, n_boot=1000, seed=args.boot_seed)
    meta["add_bos"] = bool(args.add_bos)
    meta["content_desc"] = args.content_desc
    pe_out = os.path.join(
        results_dir, f"per_example_mmlu_shard{args.shard_index}of{args.num_shards}.jsonl")
    with open(pe_out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    shard_json = os.path.join(
        results_dir, f"shard{args.shard_index}of{args.num_shards}.json")
    with open(shard_json, "w") as f:
        json.dump({
            "shard_index": args.shard_index, "num_shards": args.num_shards,
            "n_trunc": n_trunc, "seconds": round(dt, 1),
            "meta": meta, "shard_aggregate": shard_agg,
        }, f, indent=2)
    _log(f"[mmlu shard {args.shard_index}/{args.num_shards}] n={len(records)} "
         f"valid={shard_agg['n_valid']} nan={shard_agg['n_nan']} "
         f"letter={shard_agg['letter_acc']:.4f} "
         f"content_norm={shard_agg['content_norm_acc']:.4f} "
         f"trunc={n_trunc} ({dt:.1f}s) -> {pe_out}")


# ===========================================================================
# model factory
# ===========================================================================
def build_model_for_mode(args, device):
    """Return (model, tok_needed_meta_only, meta). model is either a plain
    Olmo2ForCausalLM or a GraftedModel (both expose forward -> .logits)."""
    if args.mode == "graft":
        if not (args.base_model and args.keep14_ckpt):
            raise ValueError("graft needs --base_model and --keep14_ckpt")
        return build_graft_model(
            args.base_model, args.keep14_ckpt, args.keep_front_layers,
            args.n_fresh_layers, args.graft_layer, args.inject_at_layer, device,
            donor=args.graft_donor, random_mode=args.graft_random,
            random_seed=args.graft_random_seed)
    if args.mode == "restore":
        if not (args.base_model and args.keep14_ckpt):
            raise ValueError("restore needs --base_model and --keep14_ckpt")
        base_sd, base_nl = _load_base_state_dict(args.base_model)
        keep_sd, kf, nf, step = _load_keep_state_dict(
            args.keep14_ckpt, args.keep_front_layers, args.n_fresh_layers)
        model, meta = build_restore_model(
            args.base_model, base_sd, base_nl, keep_sd, kf, nf,
            args.restore_k, args.restore_readout, device)
        meta["ckpt_step"] = step
        del base_sd, keep_sd
        gc.collect()
        return model, meta
    if args.mode == "plain":
        if args.plain_model == "base":
            model, meta = load_base_model(args.base_model, device)
        else:
            model, meta = load_pruned_model(
                args.keep14_ckpt, args.base_model, args.keep_front_layers,
                args.n_fresh_layers, device)
        return model, meta
    raise ValueError(f"unknown mode {args.mode}")


# ===========================================================================
# self-test (tiny CPU OLMo-2; no 7B weights, no GPU)
# ===========================================================================
def _tiny_cfg(base_path, n_layers):
    cfg = Olmo2Config.from_pretrained(base_path, local_files_only=True)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = n_layers
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * n_layers
    return cfg


def _selftest():
    base_path = os.environ.get("SELFTEST_BASE", "../models/OLMo-2-1124-7B")
    device = torch.device("cpu")
    KF, NF = 4, 2                 # tiny keep = 6 layers (front4 + fresh2)
    BASE_NL = 8                   # tiny base = 8 layers (deleted upper = 4)
    torch.manual_seed(0)

    # ---- build tiny base (8L) and tiny keep (6L). keep_ref is a HOOK-FREE copy
    #      used for every "plain" baseline (GraftedModel registers a pre-hook on
    #      the keep submodule it is given, so that submodule can no longer be
    #      called standalone). ----
    base = Olmo2ForCausalLM(_tiny_cfg(base_path, BASE_NL)).to(torch.float32).eval()
    keep_ref = Olmo2ForCausalLM(_tiny_cfg(base_path, KF + NF)).to(torch.float32).eval()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    keep_sd = {k: v.clone() for k, v in keep_ref.state_dict().items()}
    input_ids = torch.randint(0, base.config.vocab_size, (2, 9))
    with torch.no_grad():
        plain_ref = keep_ref(input_ids=input_ids).logits  # hook-free baseline

    def _fresh(n_layers, sd):
        m = Olmo2ForCausalLM(_tiny_cfg(base_path, n_layers)).to(torch.float32).eval()
        m.load_state_dict(sd, strict=True)
        return m

    # ---------- (A) graft: capture/inject shapes + no-op identity ----------
    # Make base's FRONT KF layers + embed EQUAL keep's, so base's output-of-layer
    # (KF-1) residual == keep's output-of-layer(KF-1) residual -> injecting it at
    # keep layer KF (fresh-tail input) is a mathematical NO-OP => grafted logits
    # must equal plain keep logits (a deterministic correctness check of the hook).
    keep2 = _fresh(KF + NF, keep_sd)
    base2 = _fresh(BASE_NL, base_sd)
    with torch.no_grad():
        base2.model.embed_tokens.load_state_dict(keep2.model.embed_tokens.state_dict())
        for i in range(KF):
            base2.model.layers[i].load_state_dict(keep2.model.layers[i].state_dict())
    gm_noop = GraftedModel(base2, keep2, graft_layer=KF - 1, inject_at=KF)
    with torch.no_grad():
        graft = gm_noop(input_ids=input_ids).logits
    assert gm_noop._captured is not None and gm_noop._captured.shape == (2, 9, 64), \
        gm_noop._captured.shape
    assert torch.allclose(plain_ref, graft, atol=1e-4), \
        f"graft no-op identity failed: max|d|={(plain_ref-graft).abs().max().item():.3e}"
    _log("[selftest] OK graft: capture shape [2,9,64]; front-equal no-op identity holds")

    # a DEEPER graft (base layer BASE_NL-1 output, base != keep) must CHANGE logits
    gm_deep = GraftedModel(base, _fresh(KF + NF, keep_sd),
                           graft_layer=BASE_NL - 1, inject_at=KF)
    with torch.no_grad():
        graft_deep = gm_deep(input_ids=input_ids).logits
    assert not torch.allclose(plain_ref, graft_deep, atol=1e-4), \
        "deep graft did not change logits (hook not wired?)"
    _log("[selftest] OK graft: deep-layer graft changes logits (inject wired)")

    # ---------- (B) restore: composition + k=0 identity + strict-load ----------
    # tail_keep14 k=0 -> byte-identical to keep_sd
    tgt0, nl0 = compose_restore_state_dict(base_sd, BASE_NL, keep_sd, KF, NF, 0,
                                           "tail_keep14")
    assert nl0 == KF + NF, nl0
    assert set(tgt0.keys()) == set(keep_sd.keys()), "k=0 keyset != keep"
    for k in keep_sd:
        assert torch.equal(tgt0[k], keep_sd[k]), f"k=0 tensor drift at {k}"
    _log("[selftest] OK restore: tail_keep14 k=0 is byte-identical to keep14")

    # a tiny shell_builder so build_restore_model composes+strict-loads at tiny dims
    def _tiny_builder(bp, n_layers, dtype):
        m = Olmo2ForCausalLM(_tiny_cfg(bp, n_layers)).to(dtype)
        return m, m.config

    # tail_keep14 k=2 -> 4+2+2 = 8 layers, strict-loads, forward runs
    m2, meta2 = build_restore_model(base_path, base_sd, BASE_NL, keep_sd, KF, NF,
                                    2, "tail_keep14", device,
                                    shell_builder=_tiny_builder)
    assert meta2["num_hidden_layers"] == 8
    # front + fresh tensors trace to keep; restored uppers trace to base
    tgt2, _ = compose_restore_state_dict(base_sd, BASE_NL, keep_sd, KF, NF, 2,
                                         "tail_keep14")
    assert torch.equal(tgt2["model.layers.0.self_attn.q_proj.weight"],
                       keep_sd["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(tgt2["model.layers.4.self_attn.q_proj.weight"],
                       base_sd["model.layers.4.self_attn.q_proj.weight"])
    assert torch.equal(tgt2["model.layers.6.self_attn.q_proj.weight"],
                       keep_sd["model.layers.4.self_attn.q_proj.weight"])  # fresh tail
    with torch.no_grad():
        _ = m2(input_ids=input_ids).logits
    _log("[selftest] OK restore: tail_keep14 k=2 -> 8L, layer provenance + forward OK")

    # base_head k=n_deleted -> keep front + full base upper + base readout
    ndel = BASE_NL - KF
    mh, metah = build_restore_model(base_path, base_sd, BASE_NL, keep_sd, KF, NF,
                                    ndel, "base_head", device,
                                    shell_builder=_tiny_builder)
    assert metah["num_hidden_layers"] == KF + ndel == BASE_NL
    tgth, _ = compose_restore_state_dict(base_sd, BASE_NL, keep_sd, KF, NF, ndel,
                                         "base_head")
    assert torch.equal(tgth["model.norm.weight"], base_sd["model.norm.weight"])
    assert torch.equal(tgth["lm_head.weight"], base_sd["lm_head.weight"])
    assert torch.equal(tgth["model.embed_tokens.weight"],
                       keep_sd["model.embed_tokens.weight"])
    with torch.no_grad():
        _ = mh(input_ids=input_ids).logits
    _log("[selftest] OK restore: base_head k=n_deleted -> base readout, forward OK")

    # ---------- (C) controls: identity patch / random patch / patch-nothing ----
    # (C1) IDENTITY PATCH: donor IS a second copy of the recipient, capture layer
    #      (KF-1)'s output, inject at layer KF. The captured tensor is exactly the
    #      tensor it replaces -> must be BIT-identical to plain keep.
    gm_id = GraftedModel(_fresh(KF + NF, keep_sd), _fresh(KF + NF, keep_sd),
                         graft_layer=KF - 1, inject_at=KF)
    with torch.no_grad():
        idt = gm_id(input_ids=input_ids).logits
    assert torch.equal(idt, plain_ref), (
        "IDENTITY PATCH is not bit-exact: max|d|="
        f"{(idt-plain_ref).abs().max().item():.3e}")
    _log("[selftest] OK control: identity patch (donor==recipient, L=KF-1) is "
         "BIT-EXACT vs plain keep")

    # (C2) PATCH-NOTHING: graft_layer<0 disables both hooks -> also identical.
    gm_off = GraftedModel(_fresh(BASE_NL, base_sd), _fresh(KF + NF, keep_sd),
                          graft_layer=-1, inject_at=KF)
    with torch.no_grad():
        off = gm_off(input_ids=input_ids).logits
    assert torch.equal(off, plain_ref), "patch-nothing (graft_layer<0) changed logits"
    _log("[selftest] OK control: patch-nothing (graft_layer=-1) == plain keep")

    # (C3) RANDOM-ACTIVATION PATCH: same donor+layer as the identity patch, but
    #      the activation is replaced by a statistics-matched surrogate -> MUST
    #      change the logits (and shuffle_positions must preserve the multiset).
    for rm in ("shuffle_positions", "gauss_matched"):
        gm_r = GraftedModel(_fresh(KF + NF, keep_sd), _fresh(KF + NF, keep_sd),
                            graft_layer=KF - 1, inject_at=KF,
                            random_mode=rm, random_seed=7)
        with torch.no_grad():
            rnd = gm_r(input_ids=input_ids).logits
        assert not torch.allclose(rnd, plain_ref, atol=1e-4), \
            f"random patch ({rm}) did not change logits"
        _log(f"[selftest] OK control: random patch ({rm}) changes logits "
             f"(max|d|={(rnd-plain_ref).abs().max().item():.3e})")
    # shuffle_positions preserves the exact activation multiset
    gm_s = GraftedModel(_fresh(KF + NF, keep_sd), _fresh(KF + NF, keep_sd),
                        graft_layer=KF - 1, inject_at=KF,
                        random_mode="shuffle_positions", random_seed=7)
    with torch.no_grad():
        _ = gm_s(input_ids=input_ids)
    cap = gm_s._captured
    sh = gm_s._randomise(cap)
    assert torch.equal(cap.flatten().sort().values, sh.flatten().sort().values), \
        "shuffle_positions changed the activation multiset"
    _log("[selftest] OK control: shuffle_positions preserves the activation multiset")

    # (C4) donor must be a distinct instance (a shared instance would deadlock on
    #      'inject before capture').
    shared = _fresh(KF + NF, keep_sd)
    try:
        GraftedModel(shared, shared, graft_layer=KF - 1, inject_at=KF)
        raise AssertionError("shared donor/recipient was not rejected")
    except ValueError:
        _log("[selftest] OK control: shared donor/recipient instance rejected")

    _log("[selftest] ALL CHECKS PASSED (graft hooks + restore composition + "
         "identity/random/patch-nothing controls validated)")


# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["graft", "restore", "plain"])
    p.add_argument("--task", choices=["ppl", "mmlu"])
    p.add_argument("--base_model", type=str, default="",
                   help="pretrained OLMo-2 32L path (cfg + base weights)")
    p.add_argument("--keep14_ckpt", type=str, default="",
                   help="healed keep14 .pt (the pruned prune-then-heal ckpt)")
    p.add_argument("--keep_front_layers", type=int, default=None,
                   help="default read from ckpt meta (keep14 -> 14)")
    p.add_argument("--n_fresh_layers", type=int, default=None,
                   help="default read from ckpt meta (keep14 -> 2)")
    # graft
    p.add_argument("--graft_layer", type=int, default=-1,
                   help="graft mode: base layer index whose OUTPUT residual is "
                        "captured (0..31). <0 disables graft (pure-keep14 control)")
    p.add_argument("--inject_at_layer", type=int, default=None,
                   help="graft mode: keep14 layer index to inject before "
                        "(default keep_front_layers = fresh-tail input)")
    p.add_argument("--graft_donor", choices=["base", "keep14"], default="base",
                   help="graft mode: which model donates the activation. 'base' "
                        "(default) = pretrained 32L OLMo-2; 'keep14' = a second "
                        "copy of the recipient itself -> with "
                        "--graft_layer=keep_front-1 this is the IDENTITY PATCH "
                        "control (must be a bit-exact no-op)")
    p.add_argument("--graft_random", choices=["none", "shuffle_positions",
                                             "gauss_matched"], default="none",
                   help="graft mode: replace the donated activation with a "
                        "statistics-matched but information-free surrogate "
                        "(random-activation patch floor)")
    p.add_argument("--graft_random_seed", type=int, default=0)
    # restore
    p.add_argument("--restore_k", type=int, default=0,
                   help="restore mode: #base upper layers restored before the tail")
    p.add_argument("--restore_readout", choices=["tail_keep14", "base_head"],
                   default="tail_keep14",
                   help="tail_keep14 (default): base uppers BEFORE keep14 fresh "
                        "tail, keep14 head. base_head: no fresh tail, base's own "
                        "norm+lm_head (confound-free upper-restoration test)")
    # plain
    p.add_argument("--plain_model", choices=["keep14", "base"], default="keep14")
    # ppl
    p.add_argument("--val_path", type=str, default="data/dolmino_now_val.npy")
    # mmlu
    p.add_argument("--content_desc", choices=["full", "none"], default="full")
    p.add_argument("--add_bos", type=int, default=0,
                   help="0 (base protocol / OLMo-2 lm-eval, default) or 1")
    p.add_argument("--max_len", type=int, default=1024)
    # sharding / io
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help=">0 caps items for THIS shard (post-stride); sanity only")
    p.add_argument("--output_name", type=str, default="")
    p.add_argument("--results_root", type=str, default="olmo2_actpatch_results")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--prepare_data", action="store_true",
                   help="mmlu: load cais/mmlu (populate cache) then exit")
    p.add_argument("--selftest", action="store_true",
                   help="tiny CPU end-to-end validation (no GPU / no 7B weights)")
    p.add_argument("--gate_check", action="store_true",
                   help="HARNESS VALIDITY GATE on the real 7B weights: identity "
                        "patch (donor=keep14, L=keep_front-1) must be a bit-exact "
                        "no-op vs plain keep14; random + base-donor patches must "
                        "change the logits. Exits 3 on failure.")
    p.add_argument("--gate_windows", type=int, default=2,
                   help="#val windows used by --gate_check")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.prepare_data:
        ex = load_mmlu_examples(args.content_desc)
        _log(f"[prepare] cais/mmlu: {len(ex)} examples cached")
        return

    if args.merge:
        if not (args.output_name and args.task):
            raise ValueError("--merge requires --output_name and --task")
        results_dir = os.path.join(args.results_root, args.output_name)
        if args.task == "ppl":
            merge_shards(results_dir)
        else:
            mmlu_merge(results_dir, n_boot=args.n_boot, seed=args.boot_seed)
        return

    if args.gate_check:
        if not (args.base_model and args.keep14_ckpt):
            raise ValueError("--gate_check needs --base_model and --keep14_ckpt")
        if not torch.cuda.is_available():
            raise RuntimeError("--gate_check needs CUDA (real 7B weights)")
        run_gate_check(args, torch.device("cuda"))
        return

    if not (args.mode and args.task and args.output_name):
        raise ValueError("need --mode, --task, --output_name (or --merge/--selftest)")
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        raise RuntimeError("CUDA required for a real run (use --selftest for CPU dry-run)")
    device = torch.device("cuda")
    model, meta = build_model_for_mode(args, device)

    if args.task == "ppl":
        run_ppl_shard(model, args, meta)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        run_mmlu_shard(model, tok, args, meta)


if __name__ == "__main__":
    main()
