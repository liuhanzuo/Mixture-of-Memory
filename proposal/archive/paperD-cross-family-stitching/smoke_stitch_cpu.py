#!/usr/bin/env python3
"""Paper D feasibility smoke test: can we physically stitch layers from two
DIFFERENT pretrained model families, and how far apart are their residual
streams?

Two sub-commands (all forward-only, NO training):

  --test splice   (a) physically build  A[0:k] + <stitch> + B[k:] + <unstitch>
                      + A.norm + A.lm_head, forward it, report shape/NaN and
                      teacher-forced CE on real text for several stitch variants
                      (identity / scale-matched / random linear / ridge-fitted
                      linear / fresh transformer block).
                      Includes a SELF-SPLICE plumbing check (B := A, stitch=none
                      must reproduce A_full CE exactly).
  --test repr     (b) layer x layer linear-CKA matrix between two models'
                      residual streams on WORD-ALIGNED representations (works
                      across different tokenizers via offset mapping), plus
                  (c) ridge regression A.layer_i -> B.layer_j held-out R^2
                      (a linear lower bound on what a stitch layer can do).

Outputs JSON into paperD_research/smoke_out/.

Why word-level alignment: the families have different tokenizers, so token rows
are not comparable. We split raw text into whitespace words, mean-pool each
model's subword hidden states inside each word span (fast-tokenizer offset
mapping), giving two [N_words, D] matrices that ARE row-aligned.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

MODELS_ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/models"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_out")

MODEL_PATHS = {
    "olmo2_1b": f"{MODELS_ROOT}/OLMo-2-0425-1B",
    "llama32_1b": f"{MODELS_ROOT}/Llama-3.2-1B",
    "olmo2_7b": f"{MODELS_ROOT}/OLMo-2-1124-7B",
    "llama3_8b": f"{MODELS_ROOT}/Llama--Llama3-8b",
    "qwen3_1p7b": f"{MODELS_ROOT}/Qwen--Qwen3-1.7b",
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# text corpus: decode the OLMo-2-tokenised wikitext103 test windows back to text
# ---------------------------------------------------------------------------
def load_texts(n_texts=400, words_per_text=60, seed=0,
               npy="data/ood_ppl/wikitext103_test.npy"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATHS["olmo2_1b"], local_files_only=True)
    arr = np.load(npy)
    rng = np.random.default_rng(seed)
    texts = []
    for wi in rng.permutation(arr.shape[0]):
        words = tok.decode(arr[wi].tolist(), skip_special_tokens=True).split()
        for s in range(0, max(len(words) - words_per_text, 0), words_per_text):
            chunk = " ".join(words[s:s + words_per_text]).strip()
            if len(chunk) > 80:
                texts.append(chunk)
            if len(texts) >= n_texts:
                return texts
    return texts


WORD_RE = re.compile(r"\S+")


def word_spans(text):
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def _load(model_key, device, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = MODEL_PATHS[model_key]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    return m, tok


# ---------------------------------------------------------------------------
# word-aligned hidden-state extraction (vectorised segment-mean via matmul)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_word_hiddens(texts, model, tok, device, max_words_total=4000):
    """Returns (H [L+1, N_words, D] float32 numpy, keys [(text_idx, word_idx)])."""
    L, D = model.config.num_hidden_layers, model.config.hidden_size
    chunks, keys = [], []
    for ti, text in enumerate(texts):
        if len(keys) >= max_words_total:
            break
        spans = word_spans(text)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
                  return_tensors="pt")
        offs = enc["offset_mapping"][0].tolist()
        ids = enc["input_ids"].to(device)
        T = ids.shape[1]
        rows, kept = [], []
        for wi, (ws, we) in enumerate(spans):
            tidx = [j for j, (a, b) in enumerate(offs) if b > ws and a < we]
            if not tidx:
                continue
            r = torch.zeros(T)
            r[torch.tensor(tidx)] = 1.0 / len(tidx)
            rows.append(r)
            kept.append((ti, wi))
        if not rows:
            continue
        room = max_words_total - len(keys)
        rows, kept = rows[:room], kept[:room]
        P = torch.stack(rows).to(device)                     # [W, T]
        hs = model(input_ids=ids, output_hidden_states=True,
                   use_cache=False).hidden_states            # tuple(L+1) [1,T,D]
        stacked = torch.stack([h[0].float() for h in hs], 0)  # [L+1, T, D]
        pooled = torch.einsum("wt,ltd->lwd", P, stacked)      # [L+1, W, D]
        chunks.append(pooled.cpu())
        keys.extend(kept)
        del hs, stacked, pooled
    H = torch.cat(chunks, dim=1).numpy()
    assert H.shape[0] == L + 1 and H.shape[2] == D
    return H, keys


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def linear_cka_cpu64(X, Y):
    X = torch.from_numpy(np.ascontiguousarray(X)).double()
    Y = torch.from_numpy(np.ascontiguousarray(Y)).double()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = ((Y.T @ X) ** 2).sum()
    denom = torch.sqrt(((X.T @ X) ** 2).sum() * ((Y.T @ Y) ** 2).sum())
    return float("nan") if denom.item() == 0 else float((num / denom).item())


def cka_matrix_gpu(Ha, Hb, device):
    """Full (La+1) x (Lb+1) linear-CKA matrix. fp32 GPU matmuls with TF32 off."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    A = [torch.from_numpy(Ha[i]).to(device) for i in range(Ha.shape[0])]
    B = [torch.from_numpy(Hb[j]).to(device) for j in range(Hb.shape[0])]
    A = [x - x.mean(0, keepdim=True) for x in A]
    B = [y - y.mean(0, keepdim=True) for y in B]
    na = [torch.linalg.norm(x.T @ x).double() for x in A]
    nb = [torch.linalg.norm(y.T @ y).double() for y in B]
    M = np.zeros((len(A), len(B)))
    for i, x in enumerate(A):
        for j, y in enumerate(B):
            num = ((y.T @ x).double() ** 2).sum()
            M[i, j] = float((num / (na[i] * nb[j])).item())
    torch.backends.cuda.matmul.allow_tf32 = prev
    del A, B
    torch.cuda.empty_cache()
    return M


def _ridge_fit(X, Y, alpha_grid, train_frac=0.7, seed=0):
    X = torch.from_numpy(np.ascontiguousarray(X)).double()
    Y = torch.from_numpy(np.ascontiguousarray(Y)).double()
    N = X.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    n_tr, n_va = int(N * train_frac), int(N * 0.15)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    xm, ym = X[tr].mean(0, keepdim=True), Y[tr].mean(0, keepdim=True)
    Xc, Yc = X[tr] - xm, Y[tr] - ym
    G, C = Xc.T @ Xc, Xc.T @ Yc
    I = torch.eye(G.shape[0], dtype=torch.float64)

    def ev(W, idx):
        Ye, Yh = Y[idx], (X[idx] - xm) @ W + ym
        ss_res = ((Ye - Yh) ** 2).sum()
        ss_tot = ((Ye - Ye.mean(0, keepdim=True)) ** 2).sum()
        rel = (torch.linalg.norm(Ye - Yh)
               / torch.linalg.norm(Ye - Ye.mean(0, keepdim=True))).item()
        return float((1 - ss_res / ss_tot).item()), rel

    best = None
    for a in alpha_grid:
        W = torch.linalg.solve(G + a * I, C)
        r2v, _ = ev(W, va)
        if best is None or r2v > best[0]:
            best = (r2v, a, W)
    _, a_best, W = best
    r2, rel = ev(W, te)
    return W, xm, ym, {"test_r2": r2, "test_rel_err": rel, "alpha": a_best,
                       "n_train": int(n_tr), "n_test": int(len(te))}


ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)


def _perdim_diag(X, Y, W, xm, ym, info):
    """Anti-inflation diagnostics: a residual stream is dominated by a few
    'massive activation' dims, so a variance-weighted R^2 can look great while
    most dimensions are unpredicted. Adds MEDIAN per-dimension R^2 and the share
    of Y's variance held by its top dims. Mutates + returns `info`."""
    Xt = torch.from_numpy(np.ascontiguousarray(X)).double()
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).double()
    Yh = (Xt - xm) @ W + ym
    ss_res = ((Yt - Yh) ** 2).sum(0)
    ss_tot = ((Yt - Yt.mean(0, keepdim=True)) ** 2).sum(0)
    per_dim = 1 - ss_res / ss_tot.clamp_min(1e-30)
    var = ss_tot / ss_tot.sum()
    srt = torch.sort(var, descending=True).values
    info["per_dim_r2_median"] = float(per_dim.median().item())
    info["per_dim_r2_p10"] = float(torch.quantile(per_dim, 0.10).item())
    info["target_var_share_top1"] = float(srt[0].item())
    info["target_var_share_top8"] = float(srt[:8].sum().item())
    return info


def ridge_fit_full(X, Y, alpha_grid=ALPHAS):
    """_ridge_fit + per-dim diagnostics. Returns (W, xm, ym, info)."""
    W, xm, ym, info = _ridge_fit(X, Y, alpha_grid)
    _perdim_diag(X, Y, W, xm, ym, info)
    return W, xm, ym, info


def ridge_r2(X, Y, alpha_grid=ALPHAS):
    """Held-out R^2 for Y ~ ridge(X), with anti-inflation diagnostics."""
    return ridge_fit_full(X, Y, alpha_grid)[3]


# ---------------------------------------------------------------------------
# (a) the actual stitched model
# ---------------------------------------------------------------------------
class AffineMap(nn.Module):
    def __init__(self, W, b):
        super().__init__()
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def forward(self, x):
        return x.to(self.W.dtype) @ self.W + self.b


class StitchedLM(nn.Module):
    """A.embed -> A.layers[0:k] -> [stitch] -> B.layers[k:] -> [unstitch] ->
    A.norm -> A.lm_head.

    Each family keeps its OWN rotary embedding (mandatory: head_dim and
    rope_theta/rope_scaling differ across families -> cos/sin tables are NOT
    interchangeable). attention_mask stays None so SDPA uses its internal causal
    flag (valid: we never pad here).
    """

    def __init__(self, A, B, k, stitch, unstitch=None, seed=0, scale=None,
                 stitch_map=None, unstitch_map=None):
        super().__init__()
        self.k = k
        self.embed = A.model.embed_tokens
        self.A_layers = A.model.layers[:k]
        self.A_rotary = A.model.rotary_emb
        self.B_layers = B.model.layers[k:]
        self.B_rotary = B.model.rotary_emb
        self.norm = A.model.norm
        self.lm_head = A.lm_head
        self.vocab = A.config.vocab_size
        D = A.config.hidden_size
        assert B.config.hidden_size == D, "hidden mismatch -> needs a projection"
        torch.manual_seed(seed)
        self.stitch_kind, self.unstitch_kind = stitch, unstitch
        if stitch == "none":
            self.stitch = nn.Identity()
        elif stitch == "scale":
            s = float(scale)
            self.stitch = AffineMap(torch.eye(D) * s, torch.zeros(D))
        elif stitch == "linear_rand":
            self.stitch = nn.Linear(D, D, bias=False)
            nn.init.normal_(self.stitch.weight, std=0.02)
        elif stitch == "ridge":
            self.stitch = stitch_map
        elif stitch == "xfmr":
            # fresh transformer block in the TARGET (B) family so its output
            # lands in B's residual convention
            self.stitch = type(B.model.layers[0])(B.config, layer_idx=k)
        else:
            raise ValueError(stitch)
        if unstitch == "ridge":
            self.unstitch = unstitch_map
        elif unstitch == "xfmr":
            self.unstitch = type(A.model.layers[0])(A.config,
                                                    layer_idx=A.config.num_hidden_layers - 1)
        else:
            self.unstitch = nn.Identity()

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        peA = self.A_rotary(h, position_ids=pos)
        for lyr in self.A_layers:
            h = lyr(h, attention_mask=None, position_embeddings=peA, position_ids=pos,
                    past_key_values=None, use_cache=False)
        peB = self.B_rotary(h, position_ids=pos)
        if self.stitch_kind == "xfmr":
            h = self.stitch(h, attention_mask=None, position_embeddings=peB,
                            position_ids=pos, past_key_values=None, use_cache=False)
        else:
            h = self.stitch(h)
        for lyr in self.B_layers:
            h = lyr(h, attention_mask=None, position_embeddings=peB, position_ids=pos,
                    past_key_values=None, use_cache=False)
        if self.unstitch_kind == "xfmr":
            h = self.unstitch(h, attention_mask=None, position_embeddings=peA,
                              position_ids=pos, past_key_values=None, use_cache=False)
        else:
            h = self.unstitch(h)
        logits = self.lm_head(self.norm(h))
        out = {"logits": logits}
        if labels is not None:
            lg = logits[:, :-1].float()
            out["loss"] = nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1))
        return out


class EarlyExit(nn.Module):
    def __init__(self, A, k):
        super().__init__()
        self.embed, self.layers = A.model.embed_tokens, A.model.layers[:k]
        self.rot, self.norm, self.head = A.model.rotary_emb, A.model.norm, A.lm_head

    def forward(self, ids):
        h = self.embed(ids)
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        pe = self.rot(h, position_ids=pos)
        for l in self.layers:
            h = l(h, attention_mask=None, position_embeddings=pe, position_ids=pos,
                  past_key_values=None, use_cache=False)
        return self.head(self.norm(h))


@torch.no_grad()
def ce_on_texts(fwd, tok, texts, device):
    tot_nll, tot_tok = 0.0, 0
    for text in texts:
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 8:
            continue
        loss = fwd(ids)
        if not np.isfinite(loss):
            return float("nan"), tot_tok
        tot_nll += loss * (ids.shape[1] - 1)
        tot_tok += ids.shape[1] - 1
    return tot_nll / max(tot_tok, 1), tot_tok


def run_splice(args):
    device, dt = args.device, torch.float32
    texts = load_texts(n_texts=args.n_ce_texts, words_per_text=60)
    _log(f"[splice] {len(texts)} eval texts | A={args.model_a} B={args.model_b} k={args.k}")
    res = {"model_a": args.model_a, "model_b": args.model_b, "k": args.k,
           "device": device, "n_texts": len(texts), "variants": {}}

    A, tokA = _load(args.model_a, device, dt)
    ce, ntok = ce_on_texts(lambda ids: float(A(input_ids=ids, labels=ids,
                                              use_cache=False).loss.item()),
                           tokA, texts, device)
    res["variants"]["A_full"] = {"ce": ce, "ppl": float(np.exp(ce)), "tokens": ntok}
    _log(f"  {'A_full':38s} CE={ce:.4f} ppl={np.exp(ce):.4g}")
    ce_a_full = ce

    ee = EarlyExit(A, args.k).to(device).eval()
    ce, ntok = ce_on_texts(
        lambda ids: float(nn.functional.cross_entropy(
            ee(ids)[:, :-1].float().reshape(-1, A.config.vocab_size),
            ids[:, 1:].reshape(-1)).item()), tokA, texts, device)
    res["variants"][f"A_earlyexit_k{args.k}"] = {
        "ce": ce, "ppl": float(np.exp(ce)), "tokens": ntok,
        "note": "A's own front-k + A's own norm/lm_head: logit-lens floor, "
                "isolates 'A's readout cannot read layer-k' from stitch quality"}
    _log(f"  {f'A_earlyexit_k{args.k}':38s} CE={ce:.4f} ppl={np.exp(ce):.4g}")

    # ---- SELF-SPLICE plumbing check: B := A, stitch=none must == A_full ----
    selfm = StitchedLM(A, A, args.k, stitch="none").to(device).eval()
    ce, ntok = ce_on_texts(lambda ids: float(selfm(ids, labels=ids)["loss"].item()),
                           tokA, texts, device)
    res["variants"]["SELF_SPLICE_A+A_stitch=none"] = {
        "ce": ce, "ppl": float(np.exp(ce)), "tokens": ntok,
        "delta_vs_A_full": ce - ce_a_full,
        "plumbing_ok": bool(abs(ce - ce_a_full) < 1e-4)}
    _log(f"  {'SELF_SPLICE A+A (plumbing)':38s} CE={ce:.4f} "
         f"delta_vs_A_full={ce-ce_a_full:+.2e} ok={abs(ce-ce_a_full)<1e-4}")
    del selfm

    B, tokB = _load(args.model_b, device, dt)
    ce, _ = ce_on_texts(lambda ids: float(B(input_ids=ids, labels=ids,
                                           use_cache=False).loss.item()),
                        tokB, texts, device)
    res["variants"]["B_full"] = {"ce": ce, "ppl": float(np.exp(ce)),
                                 "note": "B's OWN tokenizer -> not directly "
                                         "comparable to A-tokenised CE"}
    _log(f"  {'B_full (own tokenizer)':38s} CE={ce:.4f} ppl={np.exp(ce):.4g}")

    # ---- residual RMS at the splice point (for the scale-matched stitch) ----
    with torch.no_grad():
        ids = tokA(texts[0], add_special_tokens=False,
                   return_tensors="pt")["input_ids"].to(device)
        ha = A(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        idsB = tokB(texts[0], add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(device)
        hb = B(input_ids=idsB, output_hidden_states=True, use_cache=False).hidden_states
        rmsA = [float(h[0].float().pow(2).mean().sqrt().item()) for h in ha]
        rmsB = [float(h[0].float().pow(2).mean().sqrt().item()) for h in hb]
    res["residual_rms_per_layer"] = {"A": rmsA, "B": rmsB}
    scale = rmsB[args.k] / rmsA[args.k]
    res["scale_match_factor"] = scale
    _log(f"  residual RMS A: {[round(x, 2) for x in rmsA]}")
    _log(f"  residual RMS B: {[round(x, 2) for x in rmsB]}")
    _log(f"  scale-match factor at k={args.k}: {scale:.3f}")

    # ---- ridge-fitted (oracle-linear) stitch + unstitch maps -----------------
    stitch_map = unstitch_map = None
    ridge_info = {}
    if args.fit_ridge:
        _log("  fitting ridge maps on word-aligned hiddens ...")
        ftexts = load_texts(n_texts=args.n_fit_texts, words_per_text=60, seed=7)
        Ha, ka = extract_word_hiddens(ftexts, A, tokA, device, args.max_words)
        Hb, kb = extract_word_hiddens(ftexts, B, tokB, device, args.max_words)
        assert ka == kb, "word alignment broken between A and B"
        W, xm, ym, info = _ridge_fit(Ha[args.k], Hb[args.k],
                                     (1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5))
        ridge_info["stitch_A%d->B%d" % (args.k, args.k)] = info
        stitch_map = AffineMap(W.float().to(device),
                               (ym - xm @ W).float().squeeze(0).to(device)).to(device)
        _log(f"    stitch  A[{args.k}]->B[{args.k}]  test_R2={info['test_r2']:.4f} "
             f"rel_err={info['test_rel_err']:.4f}")
        LB, LA = B.config.num_hidden_layers, A.config.num_hidden_layers
        W2, xm2, ym2, info2 = _ridge_fit(Hb[LB], Ha[LA],
                                         (1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5))
        ridge_info["unstitch_B%d->A%d" % (LB, LA)] = info2
        unstitch_map = AffineMap(W2.float().to(device),
                                 (ym2 - xm2 @ W2).float().squeeze(0).to(device)).to(device)
        _log(f"    unstitch B[{LB}]->A[{LA}]  test_R2={info2['test_r2']:.4f} "
             f"rel_err={info2['test_rel_err']:.4f}")
        del Ha, Hb
    res["ridge_maps"] = ridge_info

    variants = [("none", None), ("scale", None), ("linear_rand", None),
                ("xfmr", None), ("xfmr", "xfmr")]
    if args.fit_ridge:
        variants += [("ridge", None), ("ridge", "ridge"), ("none", "ridge")]
    for stitch, unstitch in variants:
        name = f"stitch={stitch}" + (f"+unstitch={unstitch}" if unstitch else "")
        try:
            m = StitchedLM(A, B, args.k, stitch=stitch, unstitch=unstitch,
                           seed=args.seed, scale=scale, stitch_map=stitch_map,
                           unstitch_map=unstitch_map).to(device).eval()
            ids = tokA(texts[0], add_special_tokens=False,
                       return_tensors="pt")["input_ids"].to(device)
            lg = m(ids, labels=ids)["logits"]
            entry = {"forward_ok": True, "logits_shape": list(lg.shape),
                     "shape_as_expected": tuple(lg.shape) == (1, ids.shape[1],
                                                              A.config.vocab_size),
                     "logits_all_finite": bool(torch.isfinite(lg).all().item()),
                     "logits_std": float(lg.float().std().item()),
                     "n_params_total_M": sum(p.numel() for p in m.parameters()) / 1e6,
                     "n_params_stitch_M": (sum(p.numel() for p in m.stitch.parameters())
                                           + sum(p.numel() for p in m.unstitch.parameters())) / 1e6}
            ce, ntok = ce_on_texts(lambda x, m=m: float(m(x, labels=x)["loss"].item()),
                                   tokA, texts, device)
            entry.update({"ce": ce, "tokens": ntok,
                          "ppl": float(np.exp(min(ce, 700))) if np.isfinite(ce) else None,
                          "delta_ce_vs_A_full": ce - ce_a_full})
            _log(f"  {name:38s} fwd_ok shape={entry['shape_as_expected']} "
                 f"finite={entry['logits_all_finite']} CE={ce:.4f} "
                 f"ppl={entry['ppl']:.4g}")
            del m
        except Exception as e:  # noqa: BLE001
            entry = {"forward_ok": False, "error": f"{type(e).__name__}: {e}"}
            _log(f"  {name:38s} FAILED: {type(e).__name__}: {e}")
        res["variants"][name] = entry
        torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR,
                     args.out or f"splice_{args.model_a}_{args.model_b}_k{args.k}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    _log(f"[splice] wrote {p}")


# ---------------------------------------------------------------------------
# (b)+(c) representation-distance driver
# ---------------------------------------------------------------------------
def zscore(H):
    """Per-dimension standardisation over the row axis.

    MANDATORY for honest cross-family comparison: LLM residual streams contain a
    handful of 'massive activation' / rogue dimensions that hold >70-99% of the
    total variance. Raw linear CKA and raw variance-weighted R^2 are then almost
    entirely a statement about those few dims, so two unrelated models can score
    R^2 ~ 0.98 just by both having a big dim. Z-scoring gives every dimension
    equal weight and is what we quote as the real number.
    """
    mu = H.mean(axis=-2, keepdims=True)
    sd = H.std(axis=-2, keepdims=True)
    return (H - mu) / np.maximum(sd, 1e-6)


def run_repr(args):
    texts = load_texts(n_texts=args.n_texts, words_per_text=args.words_per_text)
    _log(f"[repr] {len(texts)} texts | models={args.models}")
    keys_ref, H = None, {}
    for mk in args.models:
        t0 = time.time()
        m, tok = _load(mk, args.device)
        h, keys = extract_word_hiddens(texts, m, tok, args.device, args.max_words)
        _log(f"[repr] {mk}: L={m.config.num_hidden_layers} D={m.config.hidden_size} "
             f"n_words={h.shape[1]} in {time.time()-t0:.0f}s")
        if keys_ref is None:
            keys_ref = keys
        else:
            assert keys == keys_ref, f"word-key mismatch for {mk} -> alignment broken"
        H[mk] = h
        del m
        torch.cuda.empty_cache()

    if args.random_baseline:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        mk = args.models[0]
        cfg = AutoConfig.from_pretrained(MODEL_PATHS[mk], local_files_only=True)
        torch.manual_seed(1234)
        rnd = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa"
                                               ).to(torch.float32).to(args.device).eval()
        tok = AutoTokenizer.from_pretrained(MODEL_PATHS[mk], local_files_only=True)
        h, keys = extract_word_hiddens(texts, rnd, tok, args.device, args.max_words)
        assert keys == keys_ref
        H["RANDOM_" + mk] = h
        _log(f"[repr] RANDOM_{mk}: n_words={h.shape[1]}")
        del rnd
        torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    Z = {k: zscore(v) for k, v in H.items()}
    out = {"n_words": int(next(iter(H.values())).shape[1]), "n_texts": len(texts),
           "corpus": "wikitext103_test (decoded from data/ood_ppl)",
           "alignment": "whitespace-word mean-pool of subword hidden states",
           "note": "*_z metrics are per-dimension z-scored -> the HONEST numbers. "
                   "Raw metrics are dominated by a few massive-activation dims.",
           "pairs": {}}
    for pair in args.pairs:
        a, b = pair.split(":")
        Ha, Hb = H[a], H[b]
        Za, Zb = Z[a], Z[b]
        La, Lb = Ha.shape[0] - 1, Hb.shape[0] - 1
        _log(f"[cka] {a}({La}L) vs {b}({Lb}L) ...")
        t0 = time.time()
        M = cka_matrix_gpu(Ha, Hb, args.device)
        Mz = cka_matrix_gpu(Za, Zb, args.device)
        # fp64 CPU validation on one entry
        v64 = linear_cka_cpu64(Ha[La // 2], Hb[Lb // 2])
        best = np.unravel_index(np.argmax(M), M.shape)
        bestz = np.unravel_index(np.argmax(Mz), Mz.shape)
        rec = {"n_layers_a": La, "n_layers_b": Lb,
               "cka_matrix_raw": M.round(4).tolist(),
               "cka_matrix_z": Mz.round(4).tolist(),
               "fp32gpu_vs_fp64cpu_check": {
                   "entry": [La // 2, Lb // 2], "gpu_fp32": float(M[La // 2, Lb // 2]),
                   "cpu_fp64": v64,
                   "abs_diff": abs(float(M[La // 2, Lb // 2]) - v64)},
               "best_pair_raw": {"a_layer": int(best[0]), "b_layer": int(best[1]),
                                 "cka": float(M[best])},
               "best_pair_z": {"a_layer": int(bestz[0]), "b_layer": int(bestz[1]),
                               "cka_z": float(Mz[bestz])},
               "per_a_layer_best_z": [{"a": i, "b": int(np.argmax(Mz[i])),
                                       "cka_z": float(Mz[i].max())}
                                      for i in range(La + 1)],
               "relative_depth_diag_z": [
                   {"frac": round(i / min(La, Lb), 3),
                    "a": int(round(i * La / min(La, Lb))),
                    "b": int(round(i * Lb / min(La, Lb))),
                    "cka_raw": float(M[int(round(i * La / min(La, Lb))),
                                       int(round(i * Lb / min(La, Lb)))]),
                    "cka_z": float(Mz[int(round(i * La / min(La, Lb))),
                                      int(round(i * Lb / min(La, Lb)))])}
                   for i in range(min(La, Lb) + 1)],
               "ridge": {}}
        for k in args.ridge_layers:
            ka, kb = min(k, La), min(k, Lb)
            key = f"a{ka}->b{kb}"
            r_raw = ridge_r2(Ha[ka], Hb[kb])
            r_z = ridge_r2(Za[ka], Zb[kb])
            rec["ridge"][key] = {"raw": r_raw, "z": r_z,
                                 "cka_raw": float(M[ka, kb]), "cka_z": float(Mz[ka, kb])}
            _log(f"  ridge {key}: R2_raw={r_raw['test_r2']:.4f} "
                 f"R2_z={r_z['test_r2']:.4f} (perdim_med_z={r_z['per_dim_r2_median']:.4f}) "
                 f"cka_raw={M[ka, kb]:.4f} cka_z={Mz[ka, kb]:.4f}")
        out["pairs"][pair] = rec
        _log(f"  best raw {rec['best_pair_raw']} | best z {rec['best_pair_z']} "
             f"| {time.time()-t0:.0f}s | fp32-vs-fp64 "
             f"{rec['fp32gpu_vs_fp64cpu_check']['abs_diff']:.2e}")

    a0, z0 = H[args.models[0]], Z[args.models[0]]
    ctl = {}
    for k in args.ridge_layers:
        if k + 1 <= a0.shape[0] - 1:
            key = f"{args.models[0]}_l{k}->l{k+1}"
            ctl[key] = {"raw": ridge_r2(a0[k], a0[k + 1]),
                        "z": ridge_r2(z0[k], z0[k + 1]),
                        "cka_raw": linear_cka_cpu64(a0[k], a0[k + 1]),
                        "cka_z": linear_cka_cpu64(z0[k], z0[k + 1])}
            _log(f"[ctl] {key}: R2_raw={ctl[key]['raw']['test_r2']:.4f} "
                 f"R2_z={ctl[key]['z']['test_r2']:.4f} "
                 f"cka_raw={ctl[key]['cka_raw']:.4f} cka_z={ctl[key]['cka_z']:.4f}")
    out["within_model_control"] = ctl

    p = os.path.join(OUT_DIR, args.out or "repr_metrics.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    _log(f"[repr] wrote {p}")


# ---------------------------------------------------------------------------
# (d) ORACLE test: best possible *single linear* unstitch, fitted on the
#     activations that ACTUALLY arrive (token-aligned, in-distribution).
# ---------------------------------------------------------------------------
class TrunkOnly(nn.Module):
    """A.embed -> A[0:k] -> [stitch] -> B[k:]  (returns the raw residual stream,
    no final norm / no lm_head). Used to harvest the activations the readout
    would actually see."""

    def __init__(self, A, B, k, stitch_mod, stitch_is_xfmr=False):
        super().__init__()
        self.embed, self.A_layers = A.model.embed_tokens, A.model.layers[:k]
        self.A_rotary, self.B_rotary = A.model.rotary_emb, B.model.rotary_emb
        self.B_layers = B.model.layers[k:]
        self.stitch, self.stitch_is_xfmr = stitch_mod, stitch_is_xfmr

    def forward(self, ids):
        h = self.embed(ids)
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        peA = self.A_rotary(h, position_ids=pos)
        for l in self.A_layers:
            h = l(h, attention_mask=None, position_embeddings=peA, position_ids=pos,
                  past_key_values=None, use_cache=False)
        peB = self.B_rotary(h, position_ids=pos)
        if self.stitch_is_xfmr:
            h = self.stitch(h, attention_mask=None, position_embeddings=peB,
                            position_ids=pos, past_key_values=None, use_cache=False)
        else:
            h = self.stitch(h)
        for l in self.B_layers:
            h = l(h, attention_mask=None, position_embeddings=peB, position_ids=pos,
                  past_key_values=None, use_cache=False)
        return h


@torch.no_grad()
def harvest_pairs(trunk, A, tok, texts, device, max_tokens=6000):
    """Token-aligned (X, Y): X = composed trunk output, Y = A's own final
    pre-norm residual stream, for the SAME input_ids. Also returns the ids so we
    can score CE on exactly the same text."""
    Xs, Ys = [], []
    n = 0
    for text in texts:
        if n >= max_tokens:
            break
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 8:
            continue
        x = trunk(ids)[0].float()
        y = A(input_ids=ids, output_hidden_states=True,
              use_cache=False).hidden_states[-1][0].float()
        Xs.append(x.cpu())
        Ys.append(y.cpu())
        n += ids.shape[1]
    return torch.cat(Xs).numpy(), torch.cat(Ys).numpy()


def run_oracle(args):
    """The decisive lower-bound test.

    A 1-layer stitch can do AT LEAST what an optimal affine map can do. So we
    give the splice an *oracle* affine readout adapter, fitted by ridge on the
    exact activations that arrive at the readout (token-aligned, in-distribution,
    train/test split). If CE is still broken with that oracle, a single linear
    stitch provably cannot fix the splice; only a genuinely nonlinear /
    attention-bearing stitch (or real training of B's layers) could.
    """
    device = args.device
    fit_texts = load_texts(n_texts=args.n_fit_texts, words_per_text=60, seed=7)
    ev_texts = load_texts(n_texts=args.n_ce_texts, words_per_text=60, seed=0)
    A, tokA = _load(args.model_a, device)
    B, tokB = _load(args.model_b, device)
    k, D = args.k, A.config.hidden_size
    res = {"model_a": args.model_a, "model_b": args.model_b, "k": k,
           "n_fit_texts": len(fit_texts), "n_ce_texts": len(ev_texts), "arms": {}}

    ce_full, _ = ce_on_texts(lambda ids: float(A(input_ids=ids, labels=ids,
                                                use_cache=False).loss.item()),
                             tokA, ev_texts, device)
    res["ce_a_full"] = ce_full
    _log(f"[oracle] A_full CE={ce_full:.4f} ppl={np.exp(ce_full):.4g}")

    # residual scale for the scale-matched stitch
    with torch.no_grad():
        ids = tokA(ev_texts[0], add_special_tokens=False,
                   return_tensors="pt")["input_ids"].to(device)
        rA = A(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        idsB = tokB(ev_texts[0], add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(device)
        rB = B(input_ids=idsB, output_hidden_states=True, use_cache=False).hidden_states
        scale = (float(rB[k][0].float().pow(2).mean().sqrt())
                 / float(rA[k][0].float().pow(2).mean().sqrt()))
    res["scale_match_factor"] = scale

    # word-aligned ridge stitch A[k] -> B[k]
    Ha, ka = extract_word_hiddens(fit_texts, A, tokA, device, args.max_words)
    Hb, kb = extract_word_hiddens(fit_texts, B, tokB, device, args.max_words)
    assert ka == kb, "word alignment broken"
    W_s, xm, ym, info_s = ridge_fit_full(Ha[k], Hb[k])
    res["stitch_ridge_fit"] = info_s
    ridge_stitch = AffineMap(W_s.float().to(device),
                             (ym - xm @ W_s).float().squeeze(0).to(device)).to(device)
    _log(f"[oracle] word-aligned ridge stitch A[{k}]->B[{k}]: "
         f"R2={info_s['test_r2']:.4f} rel_err={info_s['test_rel_err']:.4f}")
    del Ha, Hb

    stitch_mods = {
        "none": (nn.Identity(), False),
        "scale": (AffineMap(torch.eye(D).to(device) * scale,
                            torch.zeros(D).to(device)).to(device), False),
        "ridge": (ridge_stitch, False),
    }
    if args.include_xfmr:
        torch.manual_seed(args.seed)
        stitch_mods["xfmr_untrained"] = (
            type(B.model.layers[0])(B.config, layer_idx=k).to(torch.float32).to(device), True)

    for sname, (smod, is_x) in stitch_mods.items():
        trunk = TrunkOnly(A, B, k, smod, is_x).to(device).eval()
        X, Y = harvest_pairs(trunk, A, tokA, fit_texts, device, args.max_tokens)
        Wu, xmu, ymu, info_u = ridge_fit_full(X, Y)
        oracle_unstitch = AffineMap(Wu.float().to(device),
                                    (ymu - xmu @ Wu).float().squeeze(0).to(device)).to(device)
        arm = {"stitch": sname, "oracle_unstitch_fit": info_u,
               "n_fit_tokens": int(X.shape[0])}
        # CE with the oracle affine unstitch in place
        norm, head = A.model.norm, A.lm_head

        @torch.no_grad()
        def fwd(ids, trunk=trunk, un=oracle_unstitch):
            lg = head(norm(un(trunk(ids))))[:, :-1].float()
            return float(nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item())

        ce, ntok = ce_on_texts(fwd, tokA, ev_texts, device)
        arm.update({"ce_with_oracle_affine_unstitch": ce,
                    "ppl": float(np.exp(min(ce, 700))) if np.isfinite(ce) else None,
                    "delta_ce_vs_A_full": ce - ce_full, "eval_tokens": ntok})
        _log(f"[oracle] stitch={sname:14s} unstitch=ORACLE_ridge "
             f"(fit R2={info_u['test_r2']:.4f}, per-dim median "
             f"{info_u['per_dim_r2_median']:.3f}) -> CE={ce:.4f} "
             f"ppl={arm['ppl']:.4g} (A_full={ce_full:.4f})")
        res["arms"][sname] = arm
        del trunk, X, Y
        torch.cuda.empty_cache()

    # ---- reference: A[0:k] + FRESH RANDOM A-family tail (no info from B) ----
    from transformers import AutoConfig, AutoModelForCausalLM
    cfgA = AutoConfig.from_pretrained(MODEL_PATHS[args.model_a], local_files_only=True)
    torch.manual_seed(999)
    Arand = AutoModelForCausalLM.from_config(cfgA, attn_implementation="sdpa"
                                             ).to(torch.float32).to(device).eval()
    refs = [("REF_random_A_tail", Arand,
             "A[0:k] + FRESH RANDOM A-family tail + oracle affine readout. "
             "'tail carries zero useful computation' floor: any cross-family "
             "stitch must BEAT this to have transferred anything."),
            ("REF_selfsplice_A_tail", A,
             "A[0:k] + A's OWN tail[k:] (i.e. the unmodified model) + oracle "
             "affine readout. Ceiling: shows the oracle-readout harness itself "
             "is not what limits CE.")]
    for rname, Bmod, note in refs:
        trunk = TrunkOnly(A, Bmod, k, nn.Identity(), False).to(device).eval()
        X, Y = harvest_pairs(trunk, A, tokA, fit_texts, device, args.max_tokens)
        Wu, xmu, ymu, info_u = ridge_fit_full(X, Y)
        un = AffineMap(Wu.float().to(device),
                       (ymu - xmu @ Wu).float().squeeze(0).to(device)).to(device)
        norm, head = A.model.norm, A.lm_head

        @torch.no_grad()
        def fwd_r(ids, trunk=trunk, un=un):
            lg = head(norm(un(trunk(ids))))[:, :-1].float()
            return float(nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item())

        ce, ntok = ce_on_texts(fwd_r, tokA, ev_texts, device)
        res["arms"][rname] = {
            "stitch": "none", "oracle_unstitch_fit": info_u,
            "ce_with_oracle_affine_unstitch": ce,
            "ppl": float(np.exp(min(ce, 700))) if np.isfinite(ce) else None,
            "delta_ce_vs_A_full": ce - ce_full, "note": note}
        _log(f"[oracle] {rname:24s} -> CE={ce:.4f} ppl={np.exp(min(ce,700)):.4g} "
             f"(unstitch fit R2={info_u['test_r2']:.4f})")
        del trunk, X, Y
        torch.cuda.empty_cache()
    del Arand
    torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR,
                     args.out or f"oracle_{args.model_a}_{args.model_b}_k{k}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    _log(f"[oracle] wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["splice", "repr", "oracle"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    # splice
    ap.add_argument("--model_a", default="olmo2_1b")
    ap.add_argument("--model_b", default="llama32_1b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ce_texts", type=int, default=50)
    ap.add_argument("--fit_ridge", action="store_true")
    ap.add_argument("--n_fit_texts", type=int, default=120)
    # repr
    ap.add_argument("--models", nargs="+", default=["olmo2_1b", "llama32_1b"])
    ap.add_argument("--pairs", nargs="+", default=["olmo2_1b:llama32_1b"])
    ap.add_argument("--ridge_layers", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--n_texts", type=int, default=300)
    ap.add_argument("--words_per_text", type=int, default=60)
    ap.add_argument("--max_words", type=int, default=4000)
    ap.add_argument("--random_baseline", action="store_true")
    # oracle
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--include_xfmr", action="store_true")
    args = ap.parse_args()
    {"splice": run_splice, "repr": run_repr, "oracle": run_oracle}[args.test](args)


if __name__ == "__main__":
    main()
