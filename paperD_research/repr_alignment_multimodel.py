#!/usr/bin/env python3
"""Paper D / R4 -- multi-model layer-alignment measurement (n>=20 pairs).

Scales R3's 5-pair z-CKA observation to a 10-14 model pool so that the three
hypotheses R3 raised can actually be tested statistically:

  H1  CKA along relative depth is U-shaped (ends high, middle collapses).
  H2  Depth mismatch hurts layer alignment MORE than family mismatch.
  H3  Cross-family mid-layer CKA sits in an "awkward middle band":
      far above a random floor, far below the same-model adjacent-layer ceiling.

Design decisions carried over from smoke_stitch_cpu.py UNCHANGED so the numbers
stay comparable with R3 (that file is NOT modified):
  * corpus            : wikitext103 test windows, decoded back to text with the
                        OLMo-2 tokenizer (data/ood_ppl/wikitext103_test.npy)
  * row alignment     : whitespace words; each model's subword hidden states are
                        MEAN-pooled inside the word's character span via the fast
                        tokenizer's offset mapping -> two [N_words, D] matrices
                        whose rows refer to the same words (works across
                        different tokenizers). add_special_tokens=False, no
                        padding is ever produced (one text per forward).
  * representation    : hidden_states[i] for i in 0..L (i=0 = embedding output),
                        fp32, use_cache=False
  * "z-CKA"           : per-dimension z-score (mean/std over the word axis) of
                        each [N,D] matrix, THEN linear CKA. This is R3's headline
                        metric; it removes the massive-activation dims that
                        otherwise make raw CKA a statement about 2-3 rogue dims.
  * linear CKA        : ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F), i.e. both
                        matrices column-centered (that is the centering; z-CKA
                        centers+scales beforehand as well).
  * midband z-CKA     : mean of the z-CKA matrix over the BLOCK
                        {i : i/L_A in [0.25,0.75]} x {j : j/L_B in [0.25,0.75]}.
                        Verified to reproduce R3's 0.467/0.517/0.606/0.383/0.346
                        /0.126 exactly from paperD_research/smoke_out/*.json.

Two things ARE different from R3, both documented in the output metadata:
  1. R3 asserted the word keys of every model matched exactly. With 7 tokenizer
     families (incl. sentencepiece) that assert would just crash, so instead we
     take the GLOBAL INTERSECTION of word keys over all extracted models and
     subsample it to a single fixed N used by every pair (uniform N => pairs are
     comparable to each other).
  2. R3's 1B triple used N=4000 and its 7B/8B pairs N=3000; here every pair uses
     the same N.

Stages (run separately; extraction is 1 model per GPU, CKA is 1 pair per GPU):
  --stage extract  --model KEY [--random_init]   -> acts/<KEY>.npy + .keys.json
  --stage align                                  -> acts/common_keys.json
  --stage cka      --pairs a:b c:d ...           -> cka/<a>__<b>.json
  --stage selfcka  --model KEY                   -> cka/<KEY>__SELF.json (gate)
  --stage stats                                  -> repr_alignment_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODELS_ROOT = os.environ.get(
    "PAPERD_MODELS_ROOT",
    "/apdcephfs_zwfy6/share_304376610/pighzliu_code/models",
)
ACT_DIR = os.path.join(HERE, "align_acts")
CKA_DIR = os.path.join(HERE, "align_cka")

# key -> (relative path under MODELS_ROOT, family label, expected L)
MODEL_ZOO = {
    "olmo2_1b":     ("OLMo-2-0425-1B",                  "olmo2",     16),
    "olmo2_7b":     ("OLMo-2-1124-7B",                  "olmo2",     32),
    "llama32_1b":   ("Llama-3.2-1B",                    "llama3",    16),
    "llama3_8b":    ("Llama--Llama3-8b",                "llama3",    32),
    "llama2_7b":    ("Llama--Llama2-7b",                "llama2",    32),
    "qwen3_0p6b":   ("Qwen--Qwen3-0.6b",                "qwen3",     28),
    "qwen3_1p7b":   ("Qwen--Qwen3-1.7b",                "qwen3",     28),
    "qwen3_4b":     ("Qwen3-4B",                        "qwen3",     36),
    "gpt2":         ("openai-community--gpt2",          "gpt2",      12),
    "gpt2_medium":  ("openai-community--gpt2-medium",    "gpt2",      24),
    "gpt2_large":   ("openai-community--gpt2-large",     "gpt2",      36),
    "gpt2_xl":      ("openai-community--gpt2-xl",        "gpt2",      48),
    "opt_2p7b":     ("facebook--opt-2.7b",              "opt",       32),
    "openllama_3b": ("openlm-research--open_llama_3b_v2", "openllama", 26),
}
MIDBAND = (0.25, 0.75)

# Coarser grouping used ONLY as a robustness check on H2: "same_family" is
# ambiguous for the Llama-derived architectures (Llama-2 / Llama-3 / OpenLLaMA
# share an architecture but not weights or data). If H2's conclusion flipped
# under this relabelling we would have to report that.
LINEAGE = {"olmo2": "olmo", "llama2": "llama_arch", "llama3": "llama_arch",
           "openllama": "llama_arch", "qwen3": "qwen", "gpt2": "gpt2",
           "opt": "opt"}


def mpath(key):
    return os.path.join(MODELS_ROOT, MODEL_ZOO[key][0])


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# corpus + word-aligned extraction  (verbatim protocol from smoke_stitch_cpu.py)
# ===========================================================================
def load_texts(n_texts=300, words_per_text=60, seed=0,
               npy=None):
    from transformers import AutoTokenizer
    npy = npy or os.path.join(PROJECT_ROOT, "data/ood_ppl/wikitext103_test.npy")
    tok = AutoTokenizer.from_pretrained(mpath("olmo2_1b"), local_files_only=True)
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


def run_extract(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    key = args.model
    out_key = ("RANDOM_" + key) if args.random_init else key
    os.makedirs(ACT_DIR, exist_ok=True)
    path = mpath(key)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    assert tok.is_fast, f"{key}: need a fast tokenizer for offset mapping"

    t0 = time.time()
    if args.random_init:
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        torch.manual_seed(1234)
        model = AutoModelForCausalLM.from_config(cfg)
        model = model.to(torch.float32).to(args.device).eval()
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, dtype=torch.float32,
                attn_implementation="sdpa")
        except Exception as e:                                    # noqa: BLE001
            _log(f"[extract] {key}: sdpa failed ({e}); falling back to eager")
            model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, dtype=torch.float32,
                attn_implementation="eager")
        model = model.to(args.device).eval()
    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    _log(f"[extract] {out_key}: L={L} D={D} loaded in {time.time()-t0:.0f}s")

    texts = load_texts(n_texts=args.n_texts, words_per_text=args.words_per_text)
    chunks, keys = [], []
    t0 = time.time()
    with torch.no_grad():
        for ti, text in enumerate(texts):
            if len(keys) >= args.max_words:
                break
            spans = word_spans(text)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
                      return_tensors="pt")
            offs = enc["offset_mapping"][0].tolist()
            ids = enc["input_ids"].to(args.device)
            T = ids.shape[1]
            rows, kept = [], []
            for wi, (ws, we) in enumerate(spans):
                tidx = [j for j, (a, b) in enumerate(offs) if b > ws and a < we]
                if not tidx:
                    continue
                r = torch.zeros(T)
                r[torch.tensor(tidx)] = 1.0 / len(tidx)
                rows.append(r)
                kept.append([ti, wi])
            if not rows:
                continue
            room = args.max_words - len(keys)
            rows, kept = rows[:room], kept[:room]
            P = torch.stack(rows).to(args.device)                     # [W,T]
            hs = model(input_ids=ids, output_hidden_states=True,
                       use_cache=False).hidden_states                 # (L+1)[1,T,D]
            stacked = torch.stack([h[0].float() for h in hs], 0)      # [L+1,T,D]
            pooled = torch.einsum("wt,ltd->lwd", P, stacked)          # [L+1,W,D]
            chunks.append(pooled.cpu())
            keys.extend(kept)
            del hs, stacked, pooled
    H = torch.cat(chunks, dim=1).numpy().astype(np.float32)
    assert H.shape[0] == L + 1 and H.shape[2] == D
    _log(f"[extract] {out_key}: {H.shape} in {time.time()-t0:.0f}s "
         f"({len(texts)} texts scanned)")

    # per-layer residual-stream RMS (free, and useful context for the paper)
    rms = [float(np.sqrt(np.mean(H[i].astype(np.float64) ** 2)))
           for i in range(L + 1)]
    np.save(os.path.join(ACT_DIR, f"{out_key}.npy"), H)
    with open(os.path.join(ACT_DIR, f"{out_key}.keys.json"), "w") as f:
        json.dump({"model": key, "random_init": bool(args.random_init),
                   "n_layers": L, "hidden_size": D,
                   "family": MODEL_ZOO[key][1],
                   "keys": keys, "layer_rms": rms,
                   "n_texts_pool": args.n_texts,
                   "words_per_text": args.words_per_text,
                   "max_words": args.max_words}, f)
    _log(f"[extract] wrote {ACT_DIR}/{out_key}.npy")


# ===========================================================================
# global word-key intersection
# ===========================================================================
def run_align(args):
    metas, per = {}, {}
    for fn in sorted(os.listdir(ACT_DIR)):
        if not fn.endswith(".keys.json"):
            continue
        k = fn[:-len(".keys.json")]
        m = json.load(open(os.path.join(ACT_DIR, fn)))
        metas[k] = m
        per[k] = [tuple(x) for x in m["keys"]]
    assert per, f"no extracted activations in {ACT_DIR}"
    common = set(per[next(iter(per))])
    for k, v in per.items():
        common &= set(v)
    # deterministic order = order in the first model, then truncate to target N
    ref = next(iter(per))
    ordered = [k for k in per[ref] if k in common]
    N = min(args.target_words, len(ordered))
    sel = ordered[:N]
    assert len(set(sel)) == N, "duplicate word keys in selection"
    idx = {}
    for k, v in per.items():
        pos = {t: i for i, t in enumerate(v)}
        idx[k] = [pos[t] for t in sel]
    out = {"n_common_words": len(ordered), "n_used": N,
           "models": {k: {"n_layers": metas[k]["n_layers"],
                          "hidden_size": metas[k]["hidden_size"],
                          "family": metas[k]["family"],
                          "random_init": metas[k]["random_init"],
                          "n_words_extracted": len(per[k]),
                          "layer_rms": metas[k]["layer_rms"]} for k in per},
           "index": idx, "selected_keys": [list(t) for t in sel]}
    with open(os.path.join(ACT_DIR, "common_keys.json"), "w") as f:
        json.dump(out, f)
    _log(f"[align] {len(per)} models | per-model words="
         f"{ {k: len(v) for k, v in per.items()} } | common={len(ordered)} "
         f"| used N={N}")


def load_aligned(key):
    """[L+1, N, D] float32 restricted to the global common word set."""
    ck = json.load(open(os.path.join(ACT_DIR, "common_keys.json")))
    idx = np.asarray(ck["index"][key], dtype=np.int64)
    H = np.load(os.path.join(ACT_DIR, f"{key}.npy"), mmap_mode="r")
    return np.ascontiguousarray(H[:, idx, :]), ck["n_used"]


def zscore(H):
    mu = H.mean(axis=-2, keepdims=True)
    sd = H.std(axis=-2, keepdims=True)
    return (H - mu) / np.maximum(sd, 1e-6)


# ===========================================================================
# CKA
# ===========================================================================
def linear_cka_cpu64(X, Y):
    Xc = X.astype(np.float64); Xc = Xc - Xc.mean(0, keepdims=True)
    Yc = Y.astype(np.float64); Yc = Yc - Yc.mean(0, keepdims=True)
    num = ((Yc.T @ Xc) ** 2).sum()
    den = math.sqrt(((Xc.T @ Xc) ** 2).sum() * ((Yc.T @ Yc) ** 2).sum())
    return float("nan") if den == 0 else float(num / den)


def cka_matrix_gpu(Ha, Hb, device):
    """Full (La+1)x(Lb+1) linear-CKA matrix, fp32 GPU matmuls, TF32 OFF."""
    import torch
    try:
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:                                              # noqa: BLE001
        prev = None
    A = [torch.from_numpy(np.ascontiguousarray(Ha[i])).to(device)
         for i in range(Ha.shape[0])]
    B = [torch.from_numpy(np.ascontiguousarray(Hb[j])).to(device)
         for j in range(Hb.shape[0])]
    A = [x - x.mean(0, keepdim=True) for x in A]
    B = [y - y.mean(0, keepdim=True) for y in B]
    na = [torch.linalg.norm(x.T @ x).double() for x in A]
    nb = [torch.linalg.norm(y.T @ y).double() for y in B]
    M = np.zeros((len(A), len(B)))
    for i, x in enumerate(A):
        for j, y in enumerate(B):
            M[i, j] = float((((y.T @ x).double() ** 2).sum() /
                             (na[i] * nb[j])).item())
    if prev is not None:
        torch.backends.cuda.matmul.allow_tf32 = prev
    del A, B
    torch.cuda.empty_cache()
    return M


def _pair_record(a, b, Ma, Mz, La, Lb, extra=None):
    rec = {"model_a": a, "model_b": b, "n_layers_a": La, "n_layers_b": Lb,
           "cka_matrix_raw": np.round(Ma, 5).tolist(),
           "cka_matrix_z": np.round(Mz, 5).tolist()}
    if extra:
        rec.update(extra)
    return rec


def run_cka(args):
    import torch
    os.makedirs(CKA_DIR, exist_ok=True)
    # group pairs by first model so A's activations are loaded exactly once
    todo = {}
    for pair in args.pairs:
        a, b = pair.split(":")
        outp = os.path.join(CKA_DIR, f"{a}__{b}.json")
        if os.path.exists(outp) and not args.overwrite:
            _log(f"[cka] skip existing {a}:{b}")
            continue
        todo.setdefault(a, []).append(b)

    for a, bs in todo.items():
        Ha, N = load_aligned(a)
        Za = zscore(Ha.astype(np.float32))
        La = Ha.shape[0] - 1
        for b in bs:
            t0 = time.time()
            Hb, _ = load_aligned(b)
            Zb = zscore(Hb.astype(np.float32))
            Lb = Hb.shape[0] - 1
            Ma = cka_matrix_gpu(Ha, Hb, args.device)
            Mz = cka_matrix_gpu(Za, Zb, args.device)
            # fp64 CPU cross-check on the mid entry of BOTH metrics
            i, j = La // 2, Lb // 2
            chk = {"entry": [i, j],
                   "raw_gpu_fp32": float(Ma[i, j]),
                   "raw_cpu_fp64": linear_cka_cpu64(Ha[i], Hb[j]),
                   "z_gpu_fp32": float(Mz[i, j]),
                   "z_cpu_fp64": linear_cka_cpu64(Za[i], Zb[j])}
            chk["raw_abs_diff"] = abs(chk["raw_gpu_fp32"] - chk["raw_cpu_fp64"])
            chk["z_abs_diff"] = abs(chk["z_gpu_fp32"] - chk["z_cpu_fp64"])
            rec = _pair_record(a, b, Ma, Mz, La, Lb,
                               {"n_words": N, "fp_check": chk})
            with open(os.path.join(CKA_DIR, f"{a}__{b}.json"), "w") as f:
                json.dump(rec, f)
            _log(f"[cka] {a}({La}L):{b}({Lb}L) mid_z={Mz[i,j]:.4f} "
                 f"fp64diff={chk['z_abs_diff']:.2e} {time.time()-t0:.0f}s")
            del Ma, Mz, Hb, Zb
            torch.cuda.empty_cache()
        del Ha, Za


def run_selfcka(args):
    """HARD GATE: a model against itself must give M[i][i] == 1.0.
    Also yields the same-model adjacent-layer CKA ceiling for H3."""
    os.makedirs(CKA_DIR, exist_ok=True)
    k = args.model
    H, N = load_aligned(k)
    Z = zscore(H.astype(np.float32))
    L = H.shape[0] - 1
    Ma = cka_matrix_gpu(H, H, args.device)
    Mz = cka_matrix_gpu(Z, Z, args.device)
    dmax_raw = float(np.abs(np.diag(Ma) - 1.0).max())
    dmax_z = float(np.abs(np.diag(Mz) - 1.0).max())
    _log(f"[selfcka] {k}: L={L} max|diag-1| raw={dmax_raw:.3e} z={dmax_z:.3e}")
    assert dmax_raw < 1e-5 and dmax_z < 1e-5, (
        f"IDENTITY GATE FAILED for {k}: raw={dmax_raw}, z={dmax_z}")
    adj_z = [float(Mz[i, i + 1]) for i in range(L)]
    adj_raw = [float(Ma[i, i + 1]) for i in range(L)]
    mid = [i for i in range(L) if MIDBAND[0] <= i / L <= MIDBAND[1]]
    rec = _pair_record(k, k, Ma, Mz, L, L, {
        "n_words": N, "self": True,
        "identity_max_abs_dev_raw": dmax_raw, "identity_max_abs_dev_z": dmax_z,
        "adjacent_layer_cka_z": adj_z, "adjacent_layer_cka_raw": adj_raw,
        "adjacent_layer_cka_z_midband_mean": float(np.mean([adj_z[i] for i in mid])),
        "adjacent_layer_cka_z_mean_all": float(np.mean(adj_z))})
    with open(os.path.join(CKA_DIR, f"{k}__SELF.json"), "w") as f:
        json.dump(rec, f)


# ===========================================================================
# pure-numpy statistics (no scipy on these nodes)
# ===========================================================================
def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student-t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    p = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return p if t > 0 else 1.0 - p


def t_two_sided_p(t, df):
    return min(1.0, 2.0 * t_sf(abs(t), df))


def t_ppf(q, df):
    """Inverse t CDF by bisection (q in (0,1))."""
    lo, hi = -300.0, 300.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - t_sf(mid, df)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ols(X, y, names=None):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    rss = float(resid @ resid)
    s2 = rss / dof
    XtXinv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtXinv) * s2, 0.0))
    tv = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    tcrit = t_ppf(0.975, dof)
    tss = float(((y - y.mean()) ** 2).sum())
    names = names or [f"x{i}" for i in range(k)]
    return {
        "n": n, "dof": dof, "r2": 1.0 - rss / tss if tss > 0 else float("nan"),
        "rss": rss, "resid": resid.tolist(),
        "coef": {names[i]: {"beta": float(beta[i]), "se": float(se[i]),
                            "t": float(tv[i]),
                            "p": float(t_two_sided_p(tv[i], dof)),
                            "ci95": [float(beta[i] - tcrit * se[i]),
                                     float(beta[i] + tcrit * se[i])]}
                 for i in range(k)},
        "beta": beta.tolist(),
    }


def quad_fit(t, y):
    """y = a + b t + c t^2 ; returns c with t-stat/p (U-shape iff c>0)."""
    t = np.asarray(t, dtype=np.float64)
    X = np.column_stack([np.ones_like(t), t, t ** 2])
    r = ols(X, y, ["const", "t", "t2"])
    return {"a": r["coef"]["const"]["beta"], "b": r["coef"]["t"]["beta"],
            "c": r["coef"]["t2"]["beta"], "c_se": r["coef"]["t2"]["se"],
            "c_t": r["coef"]["t2"]["t"], "c_p": r["coef"]["t2"]["p"],
            "c_ci95": r["coef"]["t2"]["ci95"], "r2": r["r2"], "n": r["n"],
            "vertex_t": (float(-r["coef"]["t"]["beta"] /
                               (2 * r["coef"]["t2"]["beta"]))
                         if r["coef"]["t2"]["beta"] != 0 else float("nan"))}


def binom_p_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test (p=0.5) -- sign-consistency check."""
    def pmf(i):
        return math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                        - math.lgamma(n - i + 1) + i * math.log(p)
                        + (n - i) * math.log1p(-p))
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-9)))


# ===========================================================================
# stats stage
# ===========================================================================
def block_mean(M, La, Lb, lo=MIDBAND[0], hi=MIDBAND[1]):
    ia = [i for i in range(La + 1) if lo <= i / La <= hi]
    jb = [j for j in range(Lb + 1) if lo <= j / Lb <= hi]
    return float(M[np.ix_(ia, jb)].mean()), ia, jb


def rel_diag(M, La, Lb):
    """Relative-depth diagonal: i/La ~= j/Lb, min(La,Lb)+1 points."""
    n = min(La, Lb)
    out = []
    for s in range(n + 1):
        i = int(round(s * La / n))
        j = int(round(s * Lb / n))
        out.append({"frac": s / n, "a": i, "b": j, "cka": float(M[i, j]),
                    "dist": abs(i / La - j / Lb)})
    return out


def _poly_resid_matrix(M, La, Lb, deg=3):
    """Fit CKA_ij ~ poly_deg(|i/La - j/Lb|) on ALL entries, return residuals.
    Controls the trivial 'CKA just decays with relative-layer distance' story.
    Note the relative-depth diagonal has distance ~= 0 at EVERY point, so a
    pure distance model predicts a FLAT diagonal -- that is the discriminator."""
    ii, jj = np.meshgrid(np.arange(La + 1), np.arange(Lb + 1), indexing="ij")
    d = np.abs(ii / La - jj / Lb).ravel()
    y = M.ravel()
    X = np.column_stack([d ** p for p in range(deg + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / tss if tss > 0 else float("nan")
    return (y - pred).reshape(M.shape), beta.tolist(), r2


def shuffle_null(Mz, La, Lb, n_perm=200, seed=0):
    """Null for the midband block: permute B's LAYER ORDER (CKA entries are
    unchanged, only which layers count as 'B midband' changes)."""
    ia = [i for i in range(La + 1) if MIDBAND[0] <= i / La <= MIDBAND[1]]
    jb = [j for j in range(Lb + 1) if MIDBAND[0] <= j / Lb <= MIDBAND[1]]
    rng = np.random.default_rng(seed)
    rows = Mz[np.ix_(ia, list(range(Lb + 1)))]
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation(Lb + 1)
        vals.append(float(rows[:, perm[jb]].mean()))
    return vals


def run_stats(args):
    ck = json.load(open(os.path.join(ACT_DIR, "common_keys.json")))
    minfo = ck["models"]
    pairs, selfs = {}, {}
    for fn in sorted(os.listdir(CKA_DIR)):
        if not fn.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(CKA_DIR, fn)))
        if rec.get("self"):
            selfs[rec["model_a"]] = rec
        else:
            pairs[(rec["model_a"], rec["model_b"])] = rec

    # ---------------- hard gate 1: identity ----------------
    gate = {"identity_checked_models": sorted(selfs),
            "max_abs_dev_z": max([v["identity_max_abs_dev_z"] for v in selfs.values()] or [9]),
            "max_abs_dev_raw": max([v["identity_max_abs_dev_raw"] for v in selfs.values()] or [9])}
    assert selfs, "no self-CKA runs -> identity gate cannot pass"
    assert gate["max_abs_dev_z"] < 1e-5, f"IDENTITY GATE FAILED: {gate}"
    # ---------------- hard gate 2: fp32 vs fp64 ----------------
    fpd = [p["fp_check"]["z_abs_diff"] for p in pairs.values()]
    gate["max_fp32_vs_fp64_abs_diff_z"] = max(fpd) if fpd else None
    assert not fpd or max(fpd) < 1e-4, f"fp precision gate failed: {max(fpd)}"

    rows = []
    for (a, b), rec in sorted(pairs.items()):
        if minfo[a]["random_init"] or minfo[b]["random_init"]:
            continue
        La, Lb = rec["n_layers_a"], rec["n_layers_b"]
        Mz = np.asarray(rec["cka_matrix_z"])
        Mr = np.asarray(rec["cka_matrix_raw"])
        mid_z, ia, jb = block_mean(Mz, La, Lb)
        mid_r, _, _ = block_mean(Mr, La, Lb)
        diag = rel_diag(Mz, La, Lb)
        t = np.array([d["frac"] for d in diag])
        y = np.array([d["cka"] for d in diag])
        qf = quad_fit(t, y)
        R, poly, poly_r2 = _poly_resid_matrix(Mz, La, Lb)
        yres = np.array([R[d["a"], d["b"]] for d in diag])
        qf_res = quad_fit(t, yres)
        bi = int(np.argmin(y))
        bz = np.unravel_index(int(np.argmax(Mz)), Mz.shape)
        fam_a, fam_b = minfo[a]["family"], minfo[b]["family"]
        # non-parametric U geometry (does not assume a quadratic)
        u_geom = {"argmin_frac": float(t[bi]),
                  "argmin_interior": bool(0.05 < t[bi] < 0.95),
                  "both_ends_above_min": bool(y[0] > y.min() and y[-1] > y.min()),
                  "u_depth_meanend_minus_min": float((y[0] + y[-1]) / 2 - y.min()),
                  "diag_start_zcka": float(y[0]), "diag_end_zcka": float(y[-1])}
        rows.append({
            "pair": f"{a}:{b}", "model_a": a, "model_b": b,
            "family_a": fam_a, "family_b": fam_b,
            "L_a": La, "L_b": Lb, "D_a": minfo[a]["hidden_size"],
            "D_b": minfo[b]["hidden_size"],
            "same_family": int(fam_a == fam_b),
            "same_lineage": int(LINEAGE.get(fam_a, fam_a)
                                == LINEAGE.get(fam_b, fam_b)),
            "same_depth": int(La == Lb),
            "depth_ratio": max(La, Lb) / min(La, Lb),
            "log_depth_ratio": math.log(max(La, Lb) / min(La, Lb)),
            "log_width_ratio": math.log(max(minfo[a]["hidden_size"],
                                            minfo[b]["hidden_size"]) /
                                        min(minfo[a]["hidden_size"],
                                            minfo[b]["hidden_size"])),
            "midband_zcka": mid_z, "midband_raw_cka": mid_r,
            "diag_min_zcka": float(y.min()),
            "diag_min_at": f"l{diag[bi]['a']}<->l{diag[bi]['b']}",
            "diag_end_mean_zcka": float((y[0] + y[-1]) / 2),
            "diag_max_dist": float(max(d["dist"] for d in diag)),
            "best_pair_z": {"a_layer": int(bz[0]), "b_layer": int(bz[1]),
                            "cka_z": float(Mz[bz])},
            "quad_c": qf["c"], "quad_c_t": qf["c_t"], "quad_c_p": qf["c_p"],
            "quad_c_ci95": qf["c_ci95"], "quad_r2": qf["r2"],
            "quad_vertex_t": qf["vertex_t"], "quad_n": qf["n"],
            "quad_c_distresid": qf_res["c"], "quad_c_distresid_t": qf_res["c_t"],
            "quad_c_distresid_p": qf_res["c_p"],
            "dist_poly_coef": poly, "dist_poly_r2": poly_r2,
            "u_geometry": u_geom,
            "relative_depth_diag_z": [{"frac": round(e["frac"], 4), "a": e["a"],
                                       "b": e["b"], "cka_z": round(e["cka"], 5)}
                                      for e in diag],
            "shuffle_null_midband": shuffle_null(Mz, La, Lb,
                                                 n_perm=args.n_perm),
        })

    # ================= H1 =================
    cs = np.array([r["quad_c"] for r in rows])
    cr = np.array([r["quad_c_distresid"] for r in rows])
    npos = int((cs > 0).sum())
    nposr = int((cr > 0).sum())
    nsig = int(sum(1 for r in rows if r["quad_c"] > 0 and r["quad_c_p"] < 0.05))
    nsigr = int(sum(1 for r in rows if r["quad_c_distresid"] > 0
                    and r["quad_c_distresid_p"] < 0.05))
    vt = np.array([r["quad_vertex_t"] for r in rows])
    vt = vt[np.isfinite(vt)]
    am = np.array([r["u_geometry"]["argmin_frac"] for r in rows])
    ud = np.array([r["u_geometry"]["u_depth_meanend_minus_min"] for r in rows])
    H1 = {
        "n_pairs": len(rows),
        "quad_coef_positive": npos, "quad_coef_positive_frac": npos / len(rows),
        "sign_consistency_binom_p": binom_p_two_sided(npos, len(rows)),
        "quad_coef_pos_AND_p<0.05": nsig,
        "quad_coef_mean": float(cs.mean()), "quad_coef_median": float(np.median(cs)),
        "quad_coef_min": float(cs.min()), "quad_coef_max": float(cs.max()),
        "vertex_t_median": float(np.median(vt)) if vt.size else None,
        "vertex_t_iqr": [float(np.percentile(vt, 25)),
                         float(np.percentile(vt, 75))] if vt.size else None,
        "nonparametric_u_geometry": {
            "why": "the quadratic is a summary; these do not assume any shape",
            "argmin_frac_median": float(np.median(am)),
            "argmin_frac_iqr": [float(np.percentile(am, 25)),
                                float(np.percentile(am, 75))],
            "n_argmin_interior": int(sum(r["u_geometry"]["argmin_interior"]
                                         for r in rows)),
            "n_both_ends_above_min": int(sum(r["u_geometry"]["both_ends_above_min"]
                                             for r in rows)),
            "u_depth_median": float(np.median(ud)),
            "u_depth_iqr": [float(np.percentile(ud, 25)),
                            float(np.percentile(ud, 75))],
            "u_depth_min": float(ud.min())},
        "quad_c_vs_midband_corr": float(np.corrcoef(
            cs, np.array([r["midband_zcka"] for r in rows]))[0, 1]),
        "distance_control": {
            "why": "if CKA merely decayed with |i/L_a - j/L_b| the relative-depth "
                   "diagonal (where that distance is ~0 by construction) would be "
                   "FLAT, not U-shaped. Two checks: (1) max distance anywhere on "
                   "the diagonal, (2) refit the quadratic on residuals after "
                   "regressing every matrix entry on a cubic in that distance.",
            "max_diag_distance_over_pairs": float(max(r["diag_max_dist"]
                                                      for r in rows)),
            "resid_quad_coef_positive": nposr,
            "resid_quad_coef_positive_frac": nposr / len(rows),
            "resid_sign_consistency_binom_p": binom_p_two_sided(nposr, len(rows)),
            "resid_quad_coef_pos_AND_p<0.05": nsigr,
            "resid_quad_coef_median": float(np.median(cr)),
        },
        "ends_vs_mid": {
            "mean_diag_end_zcka": float(np.mean([r["diag_end_mean_zcka"]
                                                 for r in rows])),
            "mean_diag_min_zcka": float(np.mean([r["diag_min_zcka"]
                                                 for r in rows])),
            "n_pairs_with_ends_above_min": int(sum(
                1 for r in rows if r["diag_end_mean_zcka"] > r["diag_min_zcka"])),
        },
    }

    # ================= H2 =================
    y = np.array([r["midband_zcka"] for r in rows])
    sf = np.array([r["same_family"] for r in rows], dtype=float)
    ldr = np.array([r["log_depth_ratio"] for r in rows])
    lwr = np.array([r["log_width_ratio"] for r in rows])
    X = np.column_stack([np.ones_like(y), sf, ldr])
    main = ols(X, y, ["const", "same_family", "log_depth_ratio"])
    X3 = np.column_stack([np.ones_like(y), sf, ldr, lwr])
    withw = ols(X3, y, ["const", "same_family", "log_depth_ratio",
                        "log_width_ratio"])
    sl = np.array([r["same_lineage"] for r in rows], dtype=float)
    lin = ols(np.column_stack([np.ones_like(y), sl, ldr]), y,
              ["const", "same_lineage", "log_depth_ratio"])
    # H2 restated with absolute depth difference instead of the log ratio
    adf = np.array([abs(r["L_a"] - r["L_b"]) for r in rows], dtype=float)
    absd = ols(np.column_stack([np.ones_like(y), sf, adf]), y,
               ["const", "same_family", "abs_depth_diff"])
    # standardised effect sizes: beta * SD(x) / SD(y)
    sdy = y.std(ddof=1)
    std_eff = {"same_family": float(main["coef"]["same_family"]["beta"]
                                   * sf.std(ddof=1) / sdy),
               "log_depth_ratio": float(main["coef"]["log_depth_ratio"]["beta"]
                                        * ldr.std(ddof=1) / sdy)}
    # dyadic non-independence: QAP node-label permutation + node bootstrap
    models = sorted({r["model_a"] for r in rows} | {r["model_b"] for r in rows})
    ymat = {}
    for r in rows:
        ymat[(r["model_a"], r["model_b"])] = r["midband_zcka"]
        ymat[(r["model_b"], r["model_a"])] = r["midband_zcka"]
    keyorder = [(r["model_a"], r["model_b"]) for r in rows]
    rng = np.random.default_rng(0)
    nb = {"same_family": 0, "log_depth_ratio": 0}
    obs = {k: main["coef"][k]["beta"] for k in nb}
    n_qap_used = 0
    for _ in range(args.n_qap):
        perm = rng.permutation(len(models))
        mp = {models[i]: models[perm[i]] for i in range(len(models))}
        try:
            yp = np.array([ymat[(mp[a], mp[b])] for a, b in keyorder])
        except KeyError:
            continue          # incomplete pair graph -> that relabelling is not
            #                   evaluable; skip it (complete graph => never hit)
        n_qap_used += 1
        rp = ols(X, yp, ["const", "same_family", "log_depth_ratio"])
        for k in nb:
            if abs(rp["coef"][k]["beta"]) >= abs(obs[k]):
                nb[k] += 1
    B = max(n_qap_used, 1)
    qap_p = {k: (nb[k] + 1) / (B + 1) for k in nb}
    # node-level bootstrap CI
    boot = {"same_family": [], "log_depth_ratio": []}
    rng2 = np.random.default_rng(1)
    for _ in range(args.n_boot):
        pick = rng2.choice(len(models), size=len(models), replace=True)
        sub = [models[i] for i in pick]
        idxs = []
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                p = (sub[i], sub[j])
                q = (sub[j], sub[i])
                for k2, r in enumerate(rows):
                    if (r["model_a"], r["model_b"]) in (p, q):
                        idxs.append(k2)
        if len(idxs) < 6:
            continue
        try:
            rb = ols(X[idxs], y[idxs], ["const", "same_family",
                                        "log_depth_ratio"])
        except Exception:                                          # noqa: BLE001
            continue
        for k in boot:
            boot[k].append(rb["coef"][k]["beta"])
    boot_ci = {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                   if len(v) > 20 else None) for k, v in boot.items()}
    # descriptive 2x2
    cells = {}
    for r in rows:
        cells.setdefault((r["same_family"], r["same_depth"]), []).append(
            r["midband_zcka"])
    # leave-one-family-out and leave-one-model-out: is the fitted effect carried
    # by a single family? (it is -- see the report; gpt2 supplies 6/11 of the
    # same-family pairs)
    fams = sorted({r["family_a"] for r in rows} | {r["family_b"] for r in rows})

    def _refit(sub):
        if len(sub) < 6:
            return None
        ys = np.array([r["midband_zcka"] for r in sub])
        fs = np.array([r["same_family"] for r in sub], dtype=float)
        lsr = np.array([r["log_depth_ratio"] for r in sub])
        if fs.sum() in (0, len(fs)):
            o = ols(np.column_stack([np.ones_like(ys), lsr]), ys,
                    ["const", "log_depth_ratio"])
            return {"n": len(sub), "n_same_family": int(fs.sum()),
                    "same_family": None,
                    "log_depth_ratio": o["coef"]["log_depth_ratio"]}
        o = ols(np.column_stack([np.ones_like(ys), fs, lsr]), ys,
                ["const", "same_family", "log_depth_ratio"])
        return {"n": len(sub), "n_same_family": int(fs.sum()),
                "same_family": o["coef"]["same_family"],
                "log_depth_ratio": o["coef"]["log_depth_ratio"]}

    lofo = {f"drop_{f}": _refit([r for r in rows if r["family_a"] != f
                                 and r["family_b"] != f]) for f in fams}
    lomo = {}
    for m in models:
        lomo[f"drop_{m}"] = _refit([r for r in rows if r["model_a"] != m
                                    and r["model_b"] != m])
    sfc = {}
    for r in rows:
        if r["same_family"]:
            sfc[r["family_a"]] = sfc.get(r["family_a"], 0) + 1
    H2 = {
        "n_pairs": len(rows), "n_models": len(models),
        "regression_midband_on_family_and_depth": main,
        "regression_plus_log_width_ratio": withw,
        "robustness_same_lineage_instead_of_family": lin,
        "robustness_abs_depth_diff_instead_of_log_ratio": absd,
        "leave_one_family_out": lofo,
        "leave_one_model_out": lomo,
        "same_family_pair_composition": sfc,
        "standardised_effect_sizes": std_eff,
        "qap_node_permutation_p": qap_p, "n_qap_perm": B,
        "node_bootstrap_ci95": boot_ci, "n_boot": args.n_boot,
        "cells_same_family_x_same_depth": {
            f"same_family={k[0]},same_depth={k[1]}":
                {"n": len(v), "mean_midband_zcka": float(np.mean(v)),
                 "min": float(min(v)), "max": float(max(v))}
            for k, v in sorted(cells.items())},
        "same_family_pairs": sorted([(r["pair"], round(r["midband_zcka"], 3),
                                      round(r["depth_ratio"], 2))
                                     for r in rows if r["same_family"]]),
        "verdict_criterion": "H2 holds iff log_depth_ratio coef is negative and "
                             "significant AND |std effect of log_depth_ratio| >= "
                             "|std effect of same_family|",
    }

    # ================= H3 =================
    rand_rows = []
    for (a, b), rec in sorted(pairs.items()):
        if not (minfo[a]["random_init"] or minfo[b]["random_init"]):
            continue
        La, Lb = rec["n_layers_a"], rec["n_layers_b"]
        m, _, _ = block_mean(np.asarray(rec["cka_matrix_z"]), La, Lb)
        rand_rows.append({"pair": f"{a}:{b}", "midband_zcka": m})
    adj = {k: v["adjacent_layer_cka_z_midband_mean"] for k, v in selfs.items()}
    allshuf = [v for r in rows for v in r["shuffle_null_midband"]]
    xf = [r["midband_zcka"] for r in rows if not r["same_family"]]
    fmean = (float(np.mean([r["midband_zcka"] for r in rand_rows]))
             if rand_rows else None)
    fmax = (float(max(r["midband_zcka"] for r in rand_rows))
            if rand_rows else None)
    per_pair_shuf_p = [float(np.mean(np.array(r["shuffle_null_midband"])
                                     >= r["midband_zcka"])) for r in rows]
    dm = np.array([r["diag_min_zcka"] for r in rows])
    H3 = {
        "observed_midband_zcka": {
            "n": len(rows), "min": float(y.min()),
            "p25": float(np.percentile(y, 25)), "median": float(np.median(y)),
            "p75": float(np.percentile(y, 75)), "max": float(y.max()),
            "mean": float(y.mean()), "sd": float(y.std(ddof=1)),
            "deciles": [float(np.percentile(y, q)) for q in range(0, 101, 10)]},
        "in_R3_claimed_band_0.35_0.61": {
            "all_pairs": int(((y >= 0.35) & (y <= 0.61)).sum()),
            "all_pairs_frac": float(((y >= 0.35) & (y <= 0.61)).mean()),
            "cross_family_only": int(sum(1 for v in xf if 0.35 <= v <= 0.61)),
            "cross_family_frac": (float(sum(1 for v in xf if 0.35 <= v <= 0.61)
                                        / len(xf)) if xf else None)},
        "n_below_random_floor_mean": int((y < fmean).sum()) if fmean else None,
        "n_below_random_floor_max": int((y < fmax).sum()) if fmax else None,
        "n_above_ceiling_min": int((y > min(adj.values())).sum()) if adj else None,
        "diag_min_zcka": {
            "median": float(np.median(dm)), "min": float(dm.min()),
            "max": float(dm.max()),
            "n_below_random_floor_mean": (int((dm < fmean).sum())
                                          if fmean else None),
            "note": "the mid-depth WORST point of the diagonal, i.e. the number "
                    "that matters if you want to stitch at mid depth"},
        "cross_family_only": {
            "n": len(xf), "min": float(min(xf)), "median": float(np.median(xf)),
            "max": float(max(xf)), "mean": float(np.mean(xf))} if xf else None,
        "floor_random_init_models": {
            "n": len(rand_rows), "pairs": rand_rows,
            "mean": float(np.mean([r["midband_zcka"] for r in rand_rows]))
            if rand_rows else None},
        "null_layer_order_shuffle": {
            "n_perm_per_pair": args.n_perm, "n_total": len(allshuf),
            "mean": float(np.mean(allshuf)),
            "p2.5": float(np.percentile(allshuf, 2.5)),
            "p97.5": float(np.percentile(allshuf, 97.5)),
            "per_pair_p_median": float(np.median(per_pair_shuf_p)),
            "n_pairs_p_below_0.05": int(sum(1 for p in per_pair_shuf_p
                                            if p < 0.05)),
            "n_pairs_observed_above_null_mean": int(sum(
                1 for r in rows
                if r["midband_zcka"] > np.mean(r["shuffle_null_midband"]))),
            "note": "permutes B's layer ORDER; tests whether the midband block "
                    "is special, NOT the CKA magnitude floor (which is the "
                    "random-init control)"},
        "ceiling_same_model_adjacent_layers": {
            "per_model_midband_mean": adj,
            "n": len(adj), "min": float(min(adj.values())),
            "median": float(np.median(list(adj.values()))),
            "max": float(max(adj.values()))},
        "separation": None,
    }
    if rand_rows:
        H3["separation"] = {
            "observed_min_minus_floor_mean":
                float(y.min() - np.mean([r["midband_zcka"] for r in rand_rows])),
            "ceiling_min_minus_observed_max":
                float(min(adj.values()) - y.max())}

    # ================= R3 consistency =================
    r3 = {"olmo2_1b:llama32_1b": 0.467, "olmo2_1b:qwen3_1p7b": 0.517,
          "llama32_1b:qwen3_1p7b": 0.606, "olmo2_7b:llama3_8b": 0.383,
          "olmo2_7b:olmo2_1b": 0.346}
    got = {r["pair"]: r["midband_zcka"] for r in rows}
    cons = []
    for k, v in r3.items():
        a, b = k.split(":")
        m = got.get(k, got.get(f"{b}:{a}"))
        cons.append({"pair": k, "R3": v,
                     "R4": round(m, 4) if m is not None else None,
                     "abs_diff": round(abs(m - v), 4) if m is not None else None})

    per_model = {}
    for r in rows:
        per_model.setdefault(r["model_a"], []).append(r["midband_zcka"])
        per_model.setdefault(r["model_b"], []).append(r["midband_zcka"])
    out = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": os.path.basename(__file__),
            "node": os.uname().nodename,
            "models_root": MODELS_ROOT,
            "corpus": "wikitext103 test windows, decoded to text with the OLMo-2 "
                      "tokenizer (data/ood_ppl/wikitext103_test.npy); "
                      f"{minfo[next(iter(minfo))]['n_words_extracted']} words "
                      "scanned per model before intersection",
            "n_texts_pool": args.n_texts_meta,
            "words_per_text": 60,
            "n_words_used_per_pair": ck["n_used"],
            "n_common_words_available": ck["n_common_words"],
            "row_unit": "one whitespace word",
            "pooling": "MEAN over the subword tokens whose character span "
                       "overlaps the word (fast-tokenizer offset mapping); "
                       "add_special_tokens=False; one text per forward so NO "
                       "padding tokens ever enter the pool; use_cache=False",
            "layers": "hidden_states[0..L]; index 0 = embedding output",
            "dtype": "model fp32, CKA accumulations fp64, TF32 disabled",
            "centering": "linear CKA column-centers both matrices",
            "zcka_definition": "per-dimension z-score (mean/std over the word "
                               "axis) of each [N,D] matrix, then linear CKA. "
                               "Identical to smoke_stitch_cpu.py::zscore + "
                               "cka_matrix_gpu -> directly comparable to R3.",
            "midband_definition": "mean of the z-CKA matrix over the block "
                                  "{i: i/L_a in [0.25,0.75]} x {j: j/L_b in "
                                  "[0.25,0.75]} (verified to reproduce R3)",
            "relative_depth_diagonal": "min(L_a,L_b)+1 points, "
                                       "i=round(s*L_a/n), j=round(s*L_b/n)",
            "differences_from_R3": [
                "global word-key INTERSECTION over all models (R3 asserted exact "
                "equality, impossible across 7 tokenizer families)",
                "uniform N words for every pair (R3: 4000 for the 1B triple, "
                "3000 for the 7B/8B pairs)"],
        },
        "gates": gate,
        "models": {k: {"family": v["family"], "L": v["n_layers"],
                       "D": v["hidden_size"], "random_init": v["random_init"],
                       "layer_rms": v["layer_rms"]} for k, v in minfo.items()},
        "H1_u_shape": H1,
        "H2_depth_vs_family": H2,
        "H3_middle_band": H3,
        "per_model_mean_midband_zcka": {k: float(np.mean(v))
                                        for k, v in sorted(per_model.items())},
        "R3_consistency": cons,
        "pairs": rows,
    }
    p = os.path.join(HERE, args.out)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    _log(f"[stats] wrote {p}")
    _log(f"[stats] H1: quad c>0 in {npos}/{len(rows)} (binom p="
         f"{H1['sign_consistency_binom_p']:.2e}); dist-resid {nposr}/{len(rows)}")
    _log(f"[stats] H2: same_family beta="
         f"{main['coef']['same_family']['beta']:+.4f} p="
         f"{main['coef']['same_family']['p']:.4f} (QAP "
         f"{qap_p['same_family']:.4f}) | log_depth_ratio beta="
         f"{main['coef']['log_depth_ratio']['beta']:+.4f} p="
         f"{main['coef']['log_depth_ratio']['p']:.4f} (QAP "
         f"{qap_p['log_depth_ratio']:.4f}) | std eff {std_eff}")
    _log(f"[stats] H3: obs midband median={np.median(y):.3f} "
         f"[{y.min():.3f},{y.max():.3f}] | floor="
         f"{H3['floor_random_init_models']['mean']} | shuffle-null="
         f"{H3['null_layer_order_shuffle']['mean']:.3f} | ceiling median="
         f"{H3['ceiling_same_model_adjacent_layers']['median']:.3f}")


def run_listpairs(args):
    ks = args.models
    print(" ".join(f"{a}:{b}" for a, b in combinations(ks, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["extract", "align", "cka", "selfcka", "stats",
                             "listpairs"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="olmo2_1b")
    ap.add_argument("--models", nargs="+", default=sorted(MODEL_ZOO))
    ap.add_argument("--pairs", nargs="+", default=[])
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--n_texts", type=int, default=300)
    ap.add_argument("--words_per_text", type=int, default=60)
    ap.add_argument("--max_words", type=int, default=4500)
    ap.add_argument("--target_words", type=int, default=4000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_qap", type=int, default=5000)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_texts_meta", type=int, default=300)
    ap.add_argument("--out", default="repr_alignment_results.json")
    args = ap.parse_args()
    {"extract": run_extract, "align": run_align, "cka": run_cka,
     "selfcka": run_selfcka, "stats": run_stats,
     "listpairs": run_listpairs}[args.stage](args)


if __name__ == "__main__":
    main()
