#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper A · P1.2 — full content-depth probe protocol + robustness.

This EXTENDS the truncation-downstream linear probe that produced Paper A's
"content-$j$ = knee98 depth $\\approx 0.45L$, near scale-invariant" claim
(scripts/probe_truncated_downstream.py, scripts/probe_linguistic_layerwise.py,
status/QCMEM_J_DETERMINATION.md). Those scripts specified the probe only
loosely; this one nails down every under-specified piece:

  * data / labels / sample count / TRAIN-DEV-TEST split
  * probe architecture / regularization / optimizer
  * a PRECISE mathematical definition of knee98
  * >= 3 seeds with confidence intervals
  * CONTROLS (each a separate probe run): lexical-only, position-only,
    random-label (+ Hewitt-Liang selectivity), class-balance
  * a native (model-own) logit-lens readout curve, to argue that
    "linearly decodable" != "actually used by the model".

--------------------------------------------------------------------------
knee98 (formal definition)
--------------------------------------------------------------------------
For a task and a fixed backbone with L transformer layers, index hidden
states l in {0, 1, ..., L}, where l=0 is the (contextual-free) embedding
output and l=L is the top layer. Let a(l) be the HELD-OUT TEST accuracy of a
layer-l linear probe (a single affine readout of the pooled layer-l hidden
state, L2-regularized logistic regression). Define the peak decodability

        A = max_{0 <= l <= L} a(l).

The content-depth knee is the SHALLOWEST layer that recovers >= 98% of A:

        knee98 = min { l in {0,...,L} : a(l) >= 0.98 * A }.

Its fractional depth is  knee98 / L. "content-$j$" for a model is the mean of
knee98 over the semantic task set, with a CI taken over seeds and tasks.
(98% rather than 100% because per-layer test accuracy is noisy near the peak;
0.98*A is inside the sampling noise of the peak but robustly above the rising
shoulder, so knee98 marks where the curve has essentially plateaued.)

--------------------------------------------------------------------------
Design
--------------------------------------------------------------------------
ONE process = ONE (model, task) on ONE GPU. We build a FIXED labelled pool
(pool seed 0), extract per-layer pooled features ONCE (one forward pass with
output_hidden_states), then run all seeds + controls on CPU, plus a GPU
logit-lens readout curve. Each seed is a distinct stratified 60/20/20
train/dev/test partition of the same pool, so the reported CI is the
partition-induced variance of knee98 (feature extraction is deterministic;
L2 logistic regression is deterministic given data, so the split is the only
stochastic axis besides the random-label permutation).

Feature extractors / model loader are reused from probe_linguistic_layerwise
(imported, NOT modified). Verbalizer templates are reused from
probe_truncated_downstream (imported, NOT modified).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import probe_linguistic_layerwise as PL          # noqa: E402  (loader + extractors)
import probe_truncated_downstream as TD          # noqa: E402  (verbalizer templates)


# ===========================================================================
# Pool construction (fixed pool, seed 0). Uses HF *train* split (labels sure).
# ===========================================================================
def build_pool(task, n_pool, pool_seed=0):
    """Return dict: {texts:[str], labels:[int], and task-specific inputs}.
    A single fixed, stratified subsample of the HF train split of size n_pool
    (or the whole split if smaller). texts is the surface string(s) used by the
    lexical-only control."""
    load = PL.load_hf
    rng = np.random.RandomState(pool_seed)

    def _subsample(ds, n):
        idx = np.arange(len(ds))
        if n and n < len(ds):
            # stratified by label so class ratio is preserved in the pool
            labels = np.array(ds["label"])
            keep = []
            for c in np.unique(labels):
                ci = idx[labels == c]
                rng.shuffle(ci)
                keep.append(ci[: int(round(n * len(ci) / len(idx)))])
            idx = np.sort(np.concatenate(keep))
        return ds.select([int(i) for i in idx])

    if task == "SST2":
        tr = _subsample(load("nyu-mll/glue", "sst2", split="train"), n_pool)
        texts = [r["sentence"].strip() for r in tr]
        y = [int(r["label"]) for r in tr]
        return {"kind": "single", "texts": texts, "labels": y,
                "sent": texts, "rows": list(tr)}
    if task == "RTE":
        tr = _subsample(load("nyu-mll/glue", "rte", split="train"), n_pool)
        s1 = [r["sentence1"].strip() for r in tr]
        s2 = [r["sentence2"].strip() for r in tr]
        y = [int(r["label"]) for r in tr]
        texts = [a + " [SEP] " + b for a, b in zip(s1, s2)]
        return {"kind": "pair", "texts": texts, "labels": y,
                "s1": s1, "s2": s2, "rows": list(tr)}
    if task == "WiC":
        tr = _subsample(load("aps/super_glue", "wic", split="train"), n_pool)
        s1 = [r["sentence1"] for r in tr]
        s2 = [r["sentence2"] for r in tr]
        sp1 = [(r["start1"], r["end1"]) for r in tr]
        sp2 = [(r["start2"], r["end2"]) for r in tr]
        y = [int(r["label"]) for r in tr]
        texts = [f'{a} [SEP] {b} [W] {r["word"]}' for a, b, r in zip(s1, s2, tr)]
        return {"kind": "wic", "texts": texts, "labels": y,
                "s1": s1, "s2": s2, "sp1": sp1, "sp2": sp2, "rows": list(tr)}
    raise ValueError(task)


def extract_pool_features(task, pool, model, tok, device, max_len, bs, n_layers):
    """Per-layer pooled features for the whole pool. Returns {l: fp16 [N,H]}."""
    if pool["kind"] == "single":
        feats = PL.extract_sentence_pooled(model, tok, pool["sent"], device,
                                           max_len, bs, n_layers)
    elif pool["kind"] == "pair":
        fa = PL.extract_sentence_pooled(model, tok, pool["s1"], device, max_len, bs, n_layers)
        fb = PL.extract_sentence_pooled(model, tok, pool["s2"], device, max_len, bs, n_layers)
        feats = PL.combine_pair(fa, fb, n_layers)
    elif pool["kind"] == "wic":
        fa = PL.extract_target_word(model, tok, pool["s1"], pool["sp1"], device, max_len, bs, n_layers)
        fb = PL.extract_target_word(model, tok, pool["s2"], pool["sp2"], device, max_len, bs, n_layers)
        feats = PL.combine_pair(fa, fb, n_layers)
    else:
        raise ValueError(pool["kind"])
    return {l: feats[l].astype(np.float16) for l in range(n_layers)}


def token_lengths(tok, texts, max_len):
    return np.array([min(len(tok(t, add_special_tokens=False)["input_ids"]), max_len)
                     for t in texts], dtype=np.float32)


# ===========================================================================
# Splits + probes
# ===========================================================================
def stratified_3way(y, seed, fr_train=0.6, fr_dev=0.2):
    """Stratified train/dev/test index partition (test = remainder)."""
    from sklearn.model_selection import train_test_split
    y = np.asarray(y)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=fr_train, random_state=seed, stratify=y)
    dev_frac_of_rest = fr_dev / (1.0 - fr_train)
    dv, te = train_test_split(rest, train_size=dev_frac_of_rest,
                              random_state=seed, stratify=y[rest])
    return tr, dv, te


def _fit_logreg(Xtr, ytr, C, max_iter=1000):
    from sklearn.linear_model import LogisticRegression
    # sklearn default penalty is L2, default solver lbfgs (documented in meta)
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(Xtr, ytr)
    return clf


def probe_one_layer(X, y, tr, dv, te, C_grid, balanced_train=False, seed=0):
    """L2 logistic-regression probe on standardized layer features.
    C selected on dev; report TEST accuracy + balanced (macro-recall) accuracy.
    Returns (test_acc, test_balacc, chosen_C)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    X = np.nan_to_num(X.astype(np.float32), posinf=0.0, neginf=0.0)
    y = np.asarray(y)
    tr_use = tr
    if balanced_train:
        rng = np.random.RandomState(seed)
        classes, counts = np.unique(y[tr], return_counts=True)
        m = counts.min()
        pick = []
        for c in classes:
            ci = tr[y[tr] == c]
            rng.shuffle(ci)
            pick.append(ci[:m])
        tr_use = np.sort(np.concatenate(pick))
    scaler = StandardScaler().fit(X[tr_use])
    Xtr, Xdv, Xte = scaler.transform(X[tr_use]), scaler.transform(X[dv]), scaler.transform(X[te])
    best_C, best_dev, best_clf = None, -1.0, None
    for C in C_grid:
        clf = _fit_logreg(Xtr, y[tr_use], C)
        d = clf.score(Xdv, y[dv])
        if d > best_dev:
            best_dev, best_C, best_clf = d, C, clf
    acc = float(best_clf.score(Xte, y[te]))
    balacc = float(balanced_accuracy_score(y[te], best_clf.predict(Xte)))
    return acc, balacc, float(best_C)


def probe_random_label(X, y, tr, dv, te, C, seed):
    """Random-label (permutation) control: permute TRAIN labels, evaluate on
    REAL test labels. Accuracy should collapse toward chance/majority."""
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(X.astype(np.float32), posinf=0.0, neginf=0.0)
    y = np.asarray(y)
    rng = np.random.RandomState(seed + 777)
    y_perm = y.copy()
    perm = y[tr].copy()
    rng.shuffle(perm)
    y_perm[tr] = perm
    scaler = StandardScaler().fit(X[tr])
    clf = _fit_logreg(scaler.transform(X[tr]), y_perm[tr], C)
    return float(clf.score(scaler.transform(X[te]), y[te]))


# ===========================================================================
# knee98
# ===========================================================================
def knee98(acc_curve):
    """acc_curve: list a(l) over l=0..L. Returns (knee_layer, peak_layer, peak_acc)."""
    a = np.asarray(acc_curve, dtype=np.float64)
    peak_layer = int(np.argmax(a))
    A = float(a[peak_layer])
    thr = 0.98 * A
    knee = int(np.argmax(a >= thr))  # first index reaching threshold
    return knee, peak_layer, A


def ci95(vals):
    """95% CI half-width via Student-t (small n). Returns (mean, lo, hi, std)."""
    from scipy import stats
    v = np.asarray(vals, dtype=np.float64)
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m, 0.0
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(len(v))
    h = float(stats.t.ppf(0.975, len(v) - 1) * se)
    return m, m - h, m + h, sd


# ===========================================================================
# Native (model-own) logit-lens verbalizer readout curve
# ===========================================================================
def _final_norm(model):
    base = getattr(model, "model", model)
    return (getattr(base, "norm", None)
            or getattr(base, "final_layernorm", None)
            or getattr(base, "final_layer_norm", None))


def logit_lens_verbalizer_curve(model, tok, rows, task, device, max_len, bs, n_layers):
    """For each layer l, apply final_norm + lm_head to hidden[l] at the last
    prompt token and argmax over the verbalizer class tokens (the model's OWN
    output pathway). Returns list a_native(l) over l=0..L. Forward-only."""
    import torch
    spec = TD.VERBALIZERS[task]
    cls_tok = TD._first_token_ids(tok, spec["classes"])
    labels_order = list(cls_tok.keys())
    tok_ids = torch.tensor([cls_tok[l] for l in labels_order], device=device)
    prompts = [spec["template"](r) for r in rows]
    gold = [int(r[spec["label_key"]]) for r in rows]
    final_norm = _final_norm(model)
    lm_head = model.get_output_embeddings()
    if final_norm is None or lm_head is None:
        raise RuntimeError("cannot locate final norm / lm_head for logit-lens")
    correct = np.zeros(n_layers, dtype=np.int64)
    total = 0
    prev = tok.padding_side
    tok.padding_side = "left"   # last column == real last token
    try:
        with torch.no_grad():
            for b0 in range(0, len(prompts), bs):
                bp = prompts[b0:b0 + bs]
                bg = torch.tensor(gold[b0:b0 + bs], device=device)
                enc = tok(bp, return_tensors="pt", padding=True, truncation=True,
                          max_length=max_len, add_special_tokens=False)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                B = bg.shape[0]
                for l in range(n_layers):
                    h = out.hidden_states[l][:, -1, :].to(final_norm.weight.dtype)
                    logits = lm_head(final_norm(h)).float()          # (B,V)
                    sub = logits[:, tok_ids]                          # (B,n_cls)
                    pred = sub.argmax(dim=-1)                         # index into labels_order
                    pred_lab = torch.tensor([labels_order[int(p)] for p in pred], device=device)
                    correct[l] += int((pred_lab == bg).sum().item())
                total += B
    finally:
        tok.padding_side = prev
    return [float(correct[l] / total) for l in range(n_layers)], total


# ===========================================================================
# Main (mode=run)
# ===========================================================================
def run(args):
    import torch
    t0 = time.time()
    tag = os.path.basename(os.path.normpath(args.model_path))
    print(f"[{time.strftime('%H:%M:%S')}] load {args.model_path} on {args.device}", flush=True)
    model, tok, n_layers = PL.load_model(args.model_path, args.device, args.dtype)
    L = n_layers - 1
    print(f"  n_hidden_states={n_layers} (L={L} transformer layers), "
          f"hidden={model.config.hidden_size}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    C_grid = [float(c) for c in args.c_grid.split(",") if c.strip()]

    pool = build_pool(args.task, args.n_pool, pool_seed=0)
    y = pool["labels"]
    N = len(y)
    cls_counts = dict(Counter(y))
    majority = max(cls_counts.values()) / N
    print(f"  pool N={N} classes={cls_counts} majority={majority:.4f}", flush=True)

    te0 = time.time()
    feats = extract_pool_features(args.task, pool, model, tok, args.device,
                                  args.max_len, args.batch_size, n_layers)
    feat_dim = feats[0].shape[1]
    lengths = token_lengths(tok, pool["texts"], args.max_len)
    print(f"  features extracted: dim={feat_dim} in {time.time()-te0:.0f}s", flush=True)

    # ---- native (model-own) logit-lens readout curve (GPU) -----------------
    native_curve, native_n = None, 0
    if not args.skip_native and args.task in TD.VERBALIZERS:
        try:
            rr = pool["rows"][: args.n_native] if args.n_native else pool["rows"]
            native_curve, native_n = logit_lens_verbalizer_curve(
                model, tok, rr, args.task, args.device, args.max_len,
                args.batch_size, n_layers)
            print(f"  native logit-lens readout on n={native_n} examples done", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            native_curve = {"error": repr(e)[:200]}

    # free GPU; the rest is CPU sklearn
    del model
    torch.cuda.empty_cache()

    # ---- per-seed probes (main + balanced + random-label) via joblib -------
    from joblib import Parallel, delayed

    def per_layer(Xl, tr, dv, te, seed):
        # Xl passed explicitly (joblib memmaps large arrays) -> no whole-dict copy
        acc, balacc, C = probe_one_layer(Xl, y, tr, dv, te, C_grid, seed=seed)
        bal_train_acc, _bb, _bc = probe_one_layer(Xl, y, tr, dv, te, C_grid,
                                                  balanced_train=True, seed=seed)
        rnd = probe_random_label(Xl, y, tr, dv, te, C, seed)
        return acc, balacc, bal_train_acc, rnd, C

    per_seed = {}
    for seed in seeds:
        tr, dv, te = stratified_3way(y, seed)
        res = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(per_layer)(feats[l], tr, dv, te, seed) for l in range(n_layers))
        acc = [r[0] for r in res]
        balacc = [r[1] for r in res]           # macro-recall of standard probe
        bal_train_acc = [r[2] for r in res]    # accuracy of class-balanced-train probe
        rnd = [r[3] for r in res]
        Cs = [r[4] for r in res]
        k_acc, pk_acc, A_acc = knee98(acc)
        k_bal, _, _ = knee98(bal_train_acc)
        maj_test = max(Counter(np.asarray(y)[te]).values()) / len(te)
        per_seed[seed] = {
            "acc": [round(a, 4) for a in acc],
            "balanced_acc_metric": [round(a, 4) for a in balacc],
            "balanced_train_acc": [round(a, 4) for a in bal_train_acc],
            "random_label_acc": [round(a, 4) for a in rnd],
            "selectivity": [round(acc[l] - rnd[l], 4) for l in range(n_layers)],
            "chosen_C": Cs,
            "knee98_layer": k_acc, "knee98_frac": round(k_acc / L, 4),
            "peak_layer": pk_acc, "peak_acc": round(A_acc, 4),
            "knee98_layer_balancedtrain": k_bal, "knee98_frac_balancedtrain": round(k_bal / L, 4),
            "test_majority": round(maj_test, 4),
            "n_train": int(len(tr)), "n_dev": int(len(dv)), "n_test": int(len(te)),
        }
        print(f"  seed {seed}: knee98=L{k_acc} ({k_acc/L:.3f}L) peak=L{pk_acc} "
              f"acc={A_acc:.4f} | random-label peak={max(rnd):.4f} (maj {majority:.3f})",
              flush=True)

    # ---- lexical-only + position-only controls (per seed) ------------------
    def lexical_only(tr, dv, te, seed):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        texts = np.asarray(pool["texts"], dtype=object)
        vec = CountVectorizer(binary=True, ngram_range=(1, 2), max_features=50000)
        Xtr = vec.fit_transform(texts[tr]); Xte = vec.transform(texts[te])
        best = -1.0
        for C in C_grid:
            clf = LogisticRegression(C=C, max_iter=1000, n_jobs=1)
            clf.fit(Xtr, np.asarray(y)[tr])
            d = clf.score(vec.transform(texts[dv]), np.asarray(y)[dv])
            if d > best:
                best, bclf = d, clf
        return float(bclf.score(Xte, np.asarray(y)[te]))

    def position_only(tr, dv, te):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        X = np.stack([lengths, np.sqrt(lengths)], axis=1)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=1)
        clf.fit(sc.transform(X[tr]), np.asarray(y)[tr])
        return float(clf.score(sc.transform(X[te]), np.asarray(y)[te]))

    lex, pos = [], []
    for seed in seeds:
        tr, dv, te = stratified_3way(y, seed)
        lex.append(lexical_only(tr, dv, te, seed))
        pos.append(position_only(tr, dv, te))
    print(f"  lexical-only acc mean={np.mean(lex):.4f} | position-only mean={np.mean(pos):.4f}",
          flush=True)

    # ---- aggregate across seeds -------------------------------------------
    kn_layers = [per_seed[s]["knee98_layer"] for s in seeds]
    kn_fracs = [per_seed[s]["knee98_frac"] for s in seeds]
    pk_accs = [per_seed[s]["peak_acc"] for s in seeds]
    m_l, lo_l, hi_l, sd_l = ci95(kn_layers)
    m_f, lo_f, hi_f, sd_f = ci95(kn_fracs)
    m_lex, lo_lex, hi_lex, _ = ci95(lex)
    m_pos, lo_pos, hi_pos, _ = ci95(pos)
    # random-label peak across layers, per seed
    rnd_peaks = [max(per_seed[s]["random_label_acc"]) for s in seeds]

    native_knee = None
    if isinstance(native_curve, list):
        nk, npk, nA = knee98(native_curve)
        native_knee = {"knee98_layer": nk, "knee98_frac": round(nk / L, 4),
                       "peak_layer": npk, "peak_acc": round(nA, 4),
                       "curve": [round(a, 4) for a in native_curve], "n": native_n}

    result = {
        "meta": {
            "task": args.task, "model": args.model_path, "model_tag": tag,
            "dtype": args.dtype, "n_hidden_states": n_layers,
            "n_transformer_layers": L, "feat_dim": int(feat_dim),
            "pool_N": N, "pool_seed": 0, "classes": cls_counts,
            "majority_baseline": round(majority, 4),
            "split": "stratified 60/20/20 train/dev/test per seed (fixed pool)",
            "seeds": seeds, "C_grid": C_grid,
            "probe": ("L2 logistic regression (sklearn lbfgs, max_iter=1000) on "
                      "StandardScaler-normalised pooled hidden state; C selected "
                      "on dev split; report test accuracy"),
            "knee98_def": "min{l : a(l) >= 0.98 * max_l a(l)}, l=0..L",
            "max_len": args.max_len, "batch_size": args.batch_size,
        },
        "knee98": {
            "per_seed_layer": kn_layers, "per_seed_frac": kn_fracs,
            "mean_layer": round(m_l, 3), "ci95_layer": [round(lo_l, 3), round(hi_l, 3)], "std_layer": round(sd_l, 3),
            "mean_frac": round(m_f, 4), "ci95_frac": [round(lo_f, 4), round(hi_f, 4)], "std_frac": round(sd_f, 4),
            "peak_acc_mean": round(float(np.mean(pk_accs)), 4),
        },
        "controls": {
            "lexical_only": {"mean": round(m_lex, 4), "ci95": [round(lo_lex, 4), round(hi_lex, 4)], "per_seed": [round(x, 4) for x in lex]},
            "position_only": {"mean": round(m_pos, 4), "ci95": [round(lo_pos, 4), round(hi_pos, 4)], "per_seed": [round(x, 4) for x in pos]},
            "random_label_peak": {"mean": round(float(np.mean(rnd_peaks)), 4), "per_seed": [round(x, 4) for x in rnd_peaks]},
            "class_balance": {"majority_baseline": round(majority, 4),
                              "note": "balanced_acc_metric (macro-recall) and balanced_train_acc are per-layer in per_seed"},
        },
        "native_readout": native_knee,
        "per_seed": per_seed,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE {tag}/{args.task}: "
          f"knee98={m_l:.2f} layers ({m_f:.3f}L) CI[{lo_f:.3f},{hi_f:.3f}] "
          f"| native knee={native_knee['knee98_frac'] if native_knee else 'NA'}L "
          f"-> {args.out}", flush=True)


# ===========================================================================
# Aggregation (mode=aggregate): combine per-(model,task) JSONs -> summary
# ===========================================================================
def aggregate(args):
    import glob
    out_abs = os.path.abspath(args.out)
    files = []
    for fp in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        base = os.path.basename(fp)
        if base.startswith("_"):
            continue                      # skip smoke / scratch files
        if os.path.abspath(fp) == out_abs:
            continue                      # skip the summary output itself
        try:
            with open(fp) as f:
                probe = json.load(f)
        except Exception:
            continue
        if not (isinstance(probe, dict) and "meta" in probe
                and "model_tag" in probe["meta"]):
            continue                      # skip non-probe jsons
        files.append(fp)
    by_model = {}
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        m = r["meta"]["model_tag"]; L = r["meta"]["n_transformer_layers"]
        by_model.setdefault(m, {"L": L, "tasks": {}})
        by_model[m]["tasks"][r["meta"]["task"]] = r
    summary = {}
    for m, d in by_model.items():
        L = d["L"]
        # content-j = pool all per-seed knee98 fracs across the semantic tasks
        all_fracs, all_layers = [], []
        per_task = {}
        for t, r in d["tasks"].items():
            fr = r["knee98"]["per_seed_frac"]; ly = r["knee98"]["per_seed_layer"]
            all_fracs += fr; all_layers += ly
            per_task[t] = {"knee98_layer_mean": r["knee98"]["mean_layer"],
                           "knee98_frac_mean": r["knee98"]["mean_frac"],
                           "ci95_frac": r["knee98"]["ci95_frac"],
                           "peak_acc": r["knee98"]["peak_acc_mean"],
                           "lexical_only": r["controls"]["lexical_only"]["mean"],
                           "position_only": r["controls"]["position_only"]["mean"],
                           "random_label_peak": r["controls"]["random_label_peak"]["mean"],
                           "majority": r["controls"]["class_balance"]["majority_baseline"],
                           "native_knee_frac": (r["native_readout"]["knee98_frac"]
                                                if r.get("native_readout") else None),
                           "native_peak_acc": (r["native_readout"]["peak_acc"]
                                               if r.get("native_readout") else None)}
        m_f, lo_f, hi_f, sd_f = ci95(all_fracs)
        m_l, lo_l, hi_l, sd_l = ci95(all_layers)
        summary[m] = {"L": L,
                      "content_j_layer_mean": round(m_l, 2), "content_j_layer_ci95": [round(lo_l, 2), round(hi_l, 2)],
                      "content_j_frac_mean": round(m_f, 4), "content_j_frac_ci95": [round(lo_f, 4), round(hi_f, 4)],
                      "content_j_frac_std": round(sd_f, 4),
                      "n_points": len(all_fracs), "per_task": per_task}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="run", choices=["run", "aggregate"])
    ap.add_argument("--model_path")
    ap.add_argument("--task", choices=["SST2", "WiC", "RTE"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_pool", type=int, default=3000)
    ap.add_argument("--n_native", type=int, default=1000,
                    help="examples for the native logit-lens readout curve (0=all pool)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--c_grid", default="0.1,1.0,10.0")
    ap.add_argument("--n_jobs", type=int, default=12)
    ap.add_argument("--skip_native", action="store_true")
    ap.add_argument("--results_dir", default="results/p1_2",
                    help="(aggregate mode) dir of probe_*.json")
    args = ap.parse_args()
    if args.mode == "aggregate":
        aggregate(args)
        return
    if not args.model_path or not args.task:
        ap.error("--model_path and --task required for mode=run")
    run(args)


if __name__ == "__main__":
    main()
