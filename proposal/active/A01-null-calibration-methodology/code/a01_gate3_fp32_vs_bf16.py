#!/usr/bin/env python3
"""A01 gate-3 — the fp32-vs-bf16 CAUSAL test on the MMLU letter interface.

Question
--------
The letter interface on damaged OLMo-2 arms produces a large number of *exact
ties* among the four option logits (top1 == top2 bit-for-bit), and `argmax`
then breaks the tie by INDEX -- an input-blind operation.  A01 claims this is
the mechanism by which the letter interface decays into a constant predictor.
The published harness runs the forward under **bf16 autocast** and casts to fp32
only AFTER the logits exist (`log_softmax(out.logits.float())`), so the
precision is already gone.  Two hypotheses:

  H_artifact : the ties are a bf16 REPRESENTATION artifact.  In fp32 the ties
               largely disappear, the argmax becomes input-driven, and letter
               accuracy moves.
  H_real     : the damaged model genuinely puts (near-)identical mass on the
               four letters.  fp32 resolves the ties into arbitrarily tiny but
               real margins; the tie count collapses but accuracy does not move
               beyond chance-level tie-breaking luck, i.e. the interface is
               still input-blind in substance.

Design: STRICT single-variable control
--------------------------------------
* ONE process, ONE model instance, ONE item list, ONE batching order.
* The ONLY thing that differs between the two arms is the forward dtype:
    bf16 arm : `with torch.amp.autocast("cuda", dtype=torch.bfloat16)`
    fp32 arm : no autocast at all (weights are already fp32 masters)
  Everything downstream (`log_softmax(logits.float())`, the teacher-forced
  sum-logprob, the argmax, the length normalisation) is byte-identical code.
* Weights, tokenizer, prompts, truncation rule, item_ids and shard striding are
  imported/copied from `scripts/eval_olmo2_mmlu_content.py` so the bf16 arm must
  reproduce the ARCHIVED per-example scores.  That reproduction is asserted
  (`--verify_against <archived_dir>`), which is what makes the fp32 arm a clean
  contrast rather than a new harness with unknown drift.
* Scores are stored at FULL float precision (repr), not rounded to 6 dp, because
  the whole question is about tiny margins.  (The archived files are rounded to
  6 dp; the verification therefore compares at 1e-6 and additionally checks that
  the *tie pattern* is identical, which rounding cannot change -- see the
  quantisation audit in a01_gate3_tie_baseline.py: the smallest nonzero bf16 gap
  observed is ~2e-5, twenty times the rounding grid.)

Usage
-----
  # one shard on one GPU
  CUDA_VISIBLE_DEVICES=0 python3 a01_gate3_fp32_vs_bf16.py \
      --base_model ../models/OLMo-2-1124-7B \
      [--ckpt outputs/.../step200000.pt --keep_front_layers 14 --n_fresh_layers 2] \
      --output_name 7B_base_dtype --num_shards 8 --shard_index 0 --batch_size 16

  # merge (asserts ALL shards present and n_scored == expected)
  python3 a01_gate3_fp32_vs_bf16.py --merge --output_name 7B_base_dtype \
      --num_shards 8 --expect_n 14042
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_SCRIPTS = os.path.join(_PROJ, "scripts")
for p in (_SCRIPTS, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_olmo2_probe2_ppl import (  # noqa: E402
    _log,
    load_base_model,
    load_pruned_model,
)
from eval_olmo2_probe2_downstream import encode_pair  # noqa: E402
from eval_olmo2_mmlu_content import load_mmlu_examples  # noqa: E402

LETTERS = "ABCD"
DTYPES = ("bf16", "fp32")
RESULTS_ROOT_DEFAULT = os.path.join(_PROJ, "results", "a01_gate3", "dtype_runs")


# ---------------------------------------------------------------------------
# scoring: identical code path for both dtypes; the ONLY branch is the autocast
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_all(model, tok, examples, device, batch_size, add_bos, bos_id, pad_id,
              max_len, shard_index, num_shards, log_every=100):
    """Score every (example, protocol, candidate) under BOTH dtypes.

    Returns (records, n_trunc). Each record carries, for each dtype, the raw
    sum-logprob of all letter candidates and all content candidates plus the
    continuation token counts (dtype-independent, asserted).
    """
    items = []
    n_trunc = 0
    for ei, ex in enumerate(examples):
        for proto, key in (("L", "letter_cands"), ("C", "content_cands")):
            for ci, (ctx, cont, _nc) in enumerate(ex[key]):
                ids, cs, cl = encode_pair(tok, ctx, cont, add_bos, bos_id)
                if len(ids) > max_len:
                    drop = len(ids) - max_len
                    ids = ids[drop:]
                    cs = max(cs - drop, 1)
                    cl = min(cl, len(ids) - cs)
                    if cl <= 0:
                        cs, cl = max(len(ids) - 1, 1), 1
                    n_trunc += 1
                items.append((ei, proto, ci, ids, cs, cl))

    # identical batching order for both dtypes (deterministic stable sort)
    order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
    n_batches = (len(order) + batch_size - 1) // batch_size

    raw = {d: {} for d in DTYPES}
    ntok = {}
    t0 = time.time()
    for bi, b in enumerate(range(0, len(order), batch_size)):
        bidx = order[b:b + batch_size]
        maxl = max(len(items[i][3]) for i in bidx)
        B = len(bidx)
        input_ids = torch.full((B, maxl), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxl), dtype=torch.long)
        for r, i in enumerate(bidx):
            ids = items[i][3]
            input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, :len(ids)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        for dt in DTYPES:
            if dt == "bf16":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attn)
            else:                     # fp32: NO autocast; fp32 master weights
                out = model(input_ids=input_ids, attention_mask=attn)
            logprobs = torch.log_softmax(out.logits.float(), dim=-1)
            for r, i in enumerate(bidx):
                ei, proto, ci, ids, cs, cl = items[i]
                end = cs + cl
                pos = torch.arange(cs - 1, end - 1, device=device)
                tgt = torch.tensor(ids[cs:end], dtype=torch.long, device=device)
                lp = logprobs[r, pos, tgt].sum().item()
                raw[dt][(ei, proto, ci)] = lp
                if dt == "bf16":
                    ntok[(ei, proto, ci)] = cl
            del out, logprobs
        if log_every and (bi % log_every == 0 or bi == n_batches - 1):
            el = time.time() - t0
            _log(f"[shard {shard_index}/{num_shards}] batch {bi+1}/{n_batches} "
                 f"({el:.0f}s, {el/max(bi+1,1):.2f}s/batch, "
                 f"eta {el/max(bi+1,1)*(n_batches-bi-1)/60:.1f}min)")

    records = []
    for ei, ex in enumerate(examples):
        n_opt = ex["n_opt"]
        gold = ex["gold"]
        rec = {
            "item_id": shard_index + ei * num_shards,
            "subject": ex["subject"],
            "gold": gold,
            "gold_letter": LETTERS[gold] if 0 <= gold < 4 else str(gold),
            "n_opt": n_opt,
            "cont_tokens": {LETTERS[k]: ntok[(ei, "C", k)] for k in range(n_opt)},
        }
        nanflag = False
        for dt in DTYPES:
            L = [raw[dt][(ei, "L", k)] for k in range(n_opt)]
            C = [raw[dt][(ei, "C", k)] for k in range(n_opt)]
            if any(not math.isfinite(x) for x in L + C):
                nanflag = True
            rec[dt] = {
                "letter": {LETTERS[k]: L[k] for k in range(n_opt)},
                "content_raw": {LETTERS[k]: C[k] for k in range(n_opt)},
            }
        rec["nan"] = nanflag
        records.append(rec)
    return records, n_trunc


# ---------------------------------------------------------------------------
# analysis helpers (pure, operate on merged records)
# ---------------------------------------------------------------------------
def _letter_vals(rec, dt):
    n = rec["n_opt"]
    return [rec[dt]["letter"][LETTERS[k]] for k in range(n)]


def _content_norm_vals(rec, dt):
    n = rec["n_opt"]
    return [rec[dt]["content_raw"][LETTERS[k]] / max(rec["cont_tokens"][LETTERS[k]], 1)
            for k in range(n)]


def _argmax_idx(vals):
    """argmax with ties broken by INDEX -- exactly what torch/py max() does and
    exactly the input-blind operation under test."""
    return max(range(len(vals)), key=lambda k: vals[k])


def analyse(records, verbose=True):
    n = len(records)
    gold = [r["gold"] for r in records]
    gold_letter = [r["gold_letter"] for r in records]

    # ---- construct-appropriate nulls, recomputed here ----
    gc = Counter(gold_letter)
    marg = {k: gc.get(k, 0) / n for k in LETTERS}
    const_letter, hits = max(gc.items(), key=lambda kv: kv[1])
    const_floor = hits / n

    def longest_floor(conv):
        tot = 0.0
        for r in records:
            ct = r["cont_tokens"]
            keys = [k for k in LETTERS if k in ct]
            top = max(ct[k] for k in keys)
            win = [k for k in keys if ct[k] == top]
            g = r["gold_letter"]
            if conv == "split":
                tot += (1.0 / len(win)) if g in win else 0.0
            elif conv == "first":
                tot += 1.0 if win[0] == g else 0.0
            elif conv == "last":
                tot += 1.0 if win[-1] == g else 0.0
            elif conv == "credit":
                tot += 1.0 if g in win else 0.0
            elif conv == "wrong":
                tot += 1.0 if (len(win) == 1 and win[0] == g) else 0.0
        return tot / n
    longest = {c: longest_floor(c) for c in
               ("split", "first", "last", "credit", "wrong")}

    out = {
        "n": n,
        "gold_letter_marginals": marg,
        "best_constant_letter": const_letter,
        "best_constant_floor": const_floor,
        "longest_option_floor_by_conv": longest,
        "by_dtype": {},
    }

    preds = {}
    for dt in DTYPES:
        Lp, Cp = [], []
        Lcorr, CRcorr, CNcorr = [], [], []
        tie2 = 0
        mult = Counter()
        gaps = []
        tie_correct = 0
        strict_n = strict_c = 0
        for r in records:
            lv = _letter_vals(r, dt)
            cv = [r[dt]["content_raw"][LETTERS[k]] for k in range(r["n_opt"])]
            cn = _content_norm_vals(r, dt)
            pl = _argmax_idx(lv)
            pcr = _argmax_idx(cv)
            pcn = _argmax_idx(cn)
            Lp.append(pl)
            Cp.append(pcn)
            Lcorr.append(1 if pl == r["gold"] else 0)
            CRcorr.append(1 if pcr == r["gold"] else 0)
            CNcorr.append(1 if pcn == r["gold"] else 0)
            mx = max(lv)
            win = [k for k, v in enumerate(lv) if v == mx]
            mult[len(win)] += 1
            srt = sorted(lv, reverse=True)
            gaps.append(srt[0] - srt[1])
            if len(win) >= 2:
                tie2 += 1
                tie_correct += 1 if pl == r["gold"] else 0
            else:
                strict_n += 1
                strict_c += 1 if pl == r["gold"] else 0
        gaps = np.asarray(gaps, dtype=float)
        pos = gaps[gaps > 0]
        preds[dt] = {"letter": np.asarray(Lp), "content": np.asarray(Cp),
                     "Lcorr": np.asarray(Lcorr), "CRcorr": np.asarray(CRcorr),
                     "CNcorr": np.asarray(CNcorr)}
        Lacc = float(np.mean(Lcorr))
        out["by_dtype"][dt] = {
            "letter_acc": Lacc,
            "content_raw_acc": float(np.mean(CRcorr)),
            "content_norm_acc": float(np.mean(CNcorr)),
            "letter_vs_const_floor_pp": 100 * (Lacc - const_floor),
            "content_norm_vs_longest_split_pp":
                100 * (float(np.mean(CNcorr)) - longest["split"]),
            "letter_exact_tie2_count": tie2,
            "letter_exact_tie2_rate": tie2 / n,
            "letter_tie_multiplicity_hist": {str(k): mult[k] for k in sorted(mult)},
            "letter_all4_tied_count": mult.get(4, 0),
            "letter_strict_n": strict_n,
            "letter_strict_acc": (strict_c / strict_n) if strict_n else None,
            "letter_tied_index_luck_acc": (tie_correct / tie2) if tie2 else None,
            "letter_gap_min_positive": float(pos.min()) if pos.size else None,
            "letter_gap_median": float(np.median(gaps)),
            "letter_gap_p05": float(np.percentile(gaps, 5)),
            "letter_pred_dist": {LETTERS[k]: float(np.mean(np.asarray(Lp) == k))
                                 for k in range(4)},
        }

    # ---- the dtype contrast ----
    lb, lf = preds["bf16"]["letter"], preds["fp32"]["letter"]
    cb, cf = preds["bf16"]["content"], preds["fp32"]["content"]
    Lb, Lf = preds["bf16"]["Lcorr"], preds["fp32"]["Lcorr"]
    tie_mask = np.asarray([
        len([1 for k, v in enumerate(_letter_vals(r, "bf16"))
             if v == max(_letter_vals(r, "bf16"))]) >= 2 for r in records])
    out["contrast"] = {
        "letter_argmax_changed": int(np.sum(lb != lf)),
        "letter_argmax_changed_rate": float(np.mean(lb != lf)),
        "letter_argmax_changed_among_bf16_tied":
            int(np.sum((lb != lf) & tie_mask)),
        "letter_argmax_changed_among_bf16_untied":
            int(np.sum((lb != lf) & ~tie_mask)),
        "n_bf16_tied": int(tie_mask.sum()),
        "content_norm_argmax_changed": int(np.sum(cb != cf)),
        "content_norm_argmax_changed_rate": float(np.mean(cb != cf)),
        "letter_acc_delta_fp32_minus_bf16":
            out["by_dtype"]["fp32"]["letter_acc"] - out["by_dtype"]["bf16"]["letter_acc"],
        "content_norm_acc_delta_fp32_minus_bf16":
            out["by_dtype"]["fp32"]["content_norm_acc"]
            - out["by_dtype"]["bf16"]["content_norm_acc"],
        # McNemar on letter correctness bf16 vs fp32 (paired, same items)
        "letter_b_bf16right_fp32wrong": int(np.sum((Lb == 1) & (Lf == 0))),
        "letter_c_bf16wrong_fp32right": int(np.sum((Lb == 0) & (Lf == 1))),
    }
    b = out["contrast"]["letter_b_bf16right_fp32wrong"]
    c = out["contrast"]["letter_c_bf16wrong_fp32right"]
    out["contrast"]["letter_mcnemar_p"] = mcnemar_exact_p(b, c)
    # paired bootstrap on the fp32-bf16 letter accuracy difference
    d = (Lf - Lb).astype(np.float64)
    rng = np.random.default_rng(0)
    bs = np.empty(10000)
    for i in range(10000):
        bs[i] = d[rng.integers(0, d.size, d.size)].mean()
    out["contrast"]["letter_acc_diff_ci95"] = [float(np.percentile(bs, 2.5)),
                                               float(np.percentile(bs, 97.5))]
    out["contrast"]["letter_acc_diff_boot_p"] = float(
        2 * min((bs <= 0).mean(), (bs >= 0).mean()))

    # ---- arm-vs-floor test under EACH dtype (paired bootstrap) ----
    letter_null = np.asarray([1.0 if g == const_letter else 0.0
                              for g in gold_letter])
    content_null = np.zeros(n)
    for i, r in enumerate(records):
        ct = r["cont_tokens"]
        keys = [k for k in LETTERS if k in ct]
        top = max(ct[k] for k in keys)
        win = [k for k in keys if ct[k] == top]
        content_null[i] = (1.0 / len(win)) if r["gold_letter"] in win else 0.0
    for dt in DTYPES:
        for name, vec, null in (("letter", preds[dt]["Lcorr"], letter_null),
                                ("content_norm", preds[dt]["CNcorr"], content_null)):
            m, lo, hi, p = paired_bootstrap(vec.astype(np.float64) - null,
                                            10000, 7)
            out["by_dtype"][dt][f"{name}_vs_null_pp"] = 100 * m
            out["by_dtype"][dt][f"{name}_vs_null_ci95_pp"] = [100 * lo, 100 * hi]
            out["by_dtype"][dt][f"{name}_vs_null_boot_p"] = p
            out["by_dtype"][dt][f"{name}_verdict"] = (
                "AT the floor (indistinguishable)" if p >= 0.05
                else ("BELOW the floor (significantly)" if m < 0
                      else "above the floor"))
        # residual quadruple
        rep = out["by_dtype"][dt]["letter_acc"]
        out["by_dtype"][dt]["letter_residual"] = rep - const_floor
        out["by_dtype"][dt]["letter_residual_fraction"] = (
            (rep - const_floor) / rep) if rep else None
        repc = out["by_dtype"][dt]["content_norm_acc"]
        out["by_dtype"][dt]["content_residual"] = repc - longest["split"]
        out["by_dtype"][dt]["content_residual_fraction"] = (
            (repc - longest["split"]) / repc) if repc else None

    if verbose:
        _print_report(out)
    return out


def paired_bootstrap(d, n_boot=10000, seed=0):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = d[rng.integers(0, d.size, d.size)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(max(p, 1.0 / n_boot))


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    lh = n * math.log(0.5)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + lh
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, 2.0 * math.exp(m + math.log(sum(math.exp(t - m) for t in terms))))


def _print_report(o):
    print(f"\nn = {o['n']}")
    print(f"best-constant letter floor  = always-{o['best_constant_letter']} "
          f"{o['best_constant_floor']:.6f}")
    print(f"longest-option floor (split)= {o['longest_option_floor_by_conv']['split']:.6f}"
          f"   (first {o['longest_option_floor_by_conv']['first']:.6f} / "
          f"last {o['longest_option_floor_by_conv']['last']:.6f})")
    print(f"\n{'':28s} {'bf16':>12s} {'fp32':>12s}")
    rows = [
        ("letter acc", "letter_acc", "{:.4f}"),
        ("content_norm acc", "content_norm_acc", "{:.4f}"),
        ("letter exact-tie2 count", "letter_exact_tie2_count", "{:d}"),
        ("letter exact-tie2 rate", "letter_exact_tie2_rate", "{:.4f}"),
        ("letter all-4 tied", "letter_all4_tied_count", "{:d}"),
        ("letter strict-argmax acc", "letter_strict_acc", "{:.4f}"),
        ("letter min positive gap", "letter_gap_min_positive", "{:.3e}"),
        ("letter gap median", "letter_gap_median", "{:.6f}"),
        ("letter vs floor (pp)", "letter_vs_null_pp", "{:+.2f}"),
        ("letter vs floor boot p", "letter_vs_null_boot_p", "{:.4f}"),
        ("letter residual frac", "letter_residual_fraction", "{:.4f}"),
        ("content vs floor (pp)", "content_norm_vs_null_pp", "{:+.2f}"),
    ]
    for lab, key, fmt in rows:
        vs = []
        for dt in DTYPES:
            v = o["by_dtype"][dt].get(key)
            vs.append("None" if v is None else fmt.format(v))
        print(f"{lab:28s} {vs[0]:>12s} {vs[1]:>12s}")
    print(f"{'letter verdict':28s} {o['by_dtype']['bf16']['letter_verdict'][:12]:>12s} "
          f"{o['by_dtype']['fp32']['letter_verdict'][:12]:>12s}")
    c = o["contrast"]
    print(f"\ncontrast:")
    print(f"  letter argmax changed        {c['letter_argmax_changed']} "
          f"({c['letter_argmax_changed_rate']:.4f}) "
          f"[{c['letter_argmax_changed_among_bf16_tied']} of the "
          f"{c['n_bf16_tied']} bf16-tied items, "
          f"{c['letter_argmax_changed_among_bf16_untied']} of the untied]")
    print(f"  content_norm argmax changed  {c['content_norm_argmax_changed']} "
          f"({c['content_norm_argmax_changed_rate']:.4f})")
    print(f"  letter acc delta fp32-bf16   {c['letter_acc_delta_fp32_minus_bf16']:+.4f} "
          f"CI95 [{c['letter_acc_diff_ci95'][0]:+.4f},{c['letter_acc_diff_ci95'][1]:+.4f}] "
          f"boot p={c['letter_acc_diff_boot_p']:.4f} McNemar p={c['letter_mcnemar_p']:.3e}")
    print(f"  content acc delta fp32-bf16  "
          f"{c['content_norm_acc_delta_fp32_minus_bf16']:+.4f}")


# ---------------------------------------------------------------------------
# merge with HARD completeness assertions
# ---------------------------------------------------------------------------
def read_shards(results_dir, num_shards, expect_n):
    pat = os.path.join(results_dir, "per_example_dtype_shard*of*.jsonl")
    files = sorted(glob.glob(pat))
    got = {}
    for f in files:
        base = os.path.basename(f)
        idx = int(base.split("shard")[1].split("of")[0])
        tot = int(base.split("of")[1].split(".")[0])
        got[idx] = (f, tot)
    tots = {v[1] for v in got.values()}
    assert len(tots) == 1, f"shards disagree on num_shards: {tots}"
    tot = tots.pop()
    if num_shards:
        assert tot == num_shards, f"found num_shards={tot}, expected {num_shards}"
    missing = [i for i in range(tot) if i not in got]
    assert not missing, (
        f"ABORT: incomplete shard set for {results_dir}: missing shard indices "
        f"{missing} (have {sorted(got)}). Refusing to merge half a set.")
    recs = []
    for i in range(tot):
        with open(got[i][0]) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    recs.sort(key=lambda r: r["item_id"])
    ids = [r["item_id"] for r in recs]
    assert len(set(ids)) == len(ids), "duplicate item_id after merge"
    if expect_n:
        assert len(recs) == expect_n, (
            f"ABORT: n_scored={len(recs)} != expected {expect_n}")
    nan = sum(1 for r in recs if r["nan"])
    assert nan == 0, f"ABORT: {nan} nan rows"
    _log(f"[merge] {results_dir}: {tot}/{tot} shards, n_scored={len(recs)}, nan=0")
    return recs, tot


def verify_against_archive(records, archive_dir, tol=1.5e-6):
    """The validity gate: the bf16 arm of THIS harness must reproduce the
    archived per-example letter/content scores (rounded to 6 dp in the archive),
    and the exact-tie PATTERN must match item-for-item."""
    arc = {}
    with open(os.path.join(archive_dir, "per_example_mmlu.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                arc[r["item_id"]] = r
    n_cmp = 0
    max_dl = max_dc = 0.0
    tie_mismatch = 0
    pred_mismatch = 0
    for r in records:
        a = arc.get(r["item_id"])
        if a is None:
            continue
        if a.get("nan"):
            continue
        n_cmp += 1
        for k in LETTERS[:r["n_opt"]]:
            max_dl = max(max_dl, abs(r["bf16"]["letter"][k] - a["letter"]["scores"][k]))
            max_dc = max(max_dc, abs(r["bf16"]["content_raw"][k]
                                     - a["content_raw"]["scores"][k]))
        mine = _letter_vals(r, "bf16")
        theirs = [a["letter"]["scores"][LETTERS[k]] for k in range(r["n_opt"])]
        if (len([1 for v in mine if v == max(mine)]) >= 2) != \
           (len([1 for v in theirs if v == max(theirs)]) >= 2):
            tie_mismatch += 1
        if _argmax_idx(mine) != a["letter"]["pred"]:
            pred_mismatch += 1
    res = {
        "archive_dir": archive_dir, "n_compared": n_cmp,
        "max_abs_letter_score_drift": max_dl,
        "max_abs_content_score_drift": max_dc,
        "tie_pattern_mismatches": tie_mismatch,
        "letter_pred_mismatches": pred_mismatch,
        "tol": tol,
    }
    _log(f"[verify] vs {archive_dir}: n={n_cmp} max|dletter|={max_dl:.2e} "
         f"max|dcontent|={max_dc:.2e} tie_mismatch={tie_mismatch} "
         f"pred_mismatch={pred_mismatch}")
    ok = (max_dl <= tol and max_dc <= tol and tie_mismatch == 0
          and pred_mismatch == 0)
    res["PASS"] = bool(ok)
    if not ok:
        _log("[verify] *** FAILED — the bf16 arm does NOT reproduce the archive; "
             "the fp32 contrast is NOT single-variable. Reporting loudly.")
    return res


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="../models/OLMo-2-1124-7B")
    p.add_argument("--ckpt", default="")
    p.add_argument("--keep_front_layers", type=int, default=None)
    p.add_argument("--n_fresh_layers", type=int, default=None)
    p.add_argument("--content_desc", default="full", choices=["full", "none"])
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--add_bos", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output_name", required=True)
    p.add_argument("--results_root", default=RESULTS_ROOT_DEFAULT)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--expect_n", type=int, default=0)
    p.add_argument("--verify_against", default="")
    a = p.parse_args()

    results_dir = os.path.join(a.results_root, a.output_name)

    if a.merge:
        recs, tot = read_shards(results_dir, a.num_shards, a.expect_n)
        res = analyse(recs, verbose=True)
        res["output_name"] = a.output_name
        res["n_shards"] = tot
        meta = None
        for sf in sorted(glob.glob(os.path.join(results_dir, "shard*of*.json"))):
            meta = json.load(open(sf)).get("meta", meta)
        res["meta"] = meta
        if a.verify_against:
            res["archive_verification"] = verify_against_archive(
                recs, a.verify_against)
        with open(os.path.join(results_dir, "dtype_summary.json"), "w") as f:
            json.dump(res, f, indent=2)
        _log(f"[merge] wrote {results_dir}/dtype_summary.json")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model, local_files_only=True)
    bos_id = tok.bos_token_id
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    if a.ckpt:
        model, meta = load_pruned_model(a.ckpt, a.base_model, a.keep_front_layers,
                                        a.n_fresh_layers, device)
    else:
        model, meta = load_base_model(a.base_model, device)
    # assert the weights really are fp32 masters (the fp32 arm depends on it)
    dts = {tuple(str(q.dtype) for q in [pp])[0] for pp in model.parameters()}
    assert dts == {"torch.float32"}, f"weights not fp32: {dts}"
    meta.update({"base_model": a.base_model, "add_bos": bool(a.add_bos),
                 "content_desc": a.content_desc, "weights_dtype": "float32",
                 "forward_dtypes": list(DTYPES), "batch_size": a.batch_size,
                 "max_len": a.max_len})

    examples = load_mmlu_examples(a.content_desc)
    examples = examples[a.shard_index::a.num_shards]
    if a.limit > 0:
        examples = examples[:a.limit]

    os.makedirs(results_dir, exist_ok=True)
    t0 = time.time()
    recs, n_trunc = score_all(model, tok, examples, device, a.batch_size,
                              bool(a.add_bos), bos_id, pad_id, a.max_len,
                              a.shard_index, a.num_shards)
    dt = time.time() - t0
    pe = os.path.join(
        results_dir,
        f"per_example_dtype_shard{a.shard_index}of{a.num_shards}.jsonl")
    with open(pe, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(
            results_dir,
            f"shard{a.shard_index}of{a.num_shards}.json"), "w") as f:
        json.dump({"shard_index": a.shard_index, "num_shards": a.num_shards,
                   "n": len(recs), "n_trunc": n_trunc, "seconds": round(dt, 1),
                   "meta": meta}, f, indent=2)
    _log(f"[shard {a.shard_index}/{a.num_shards}] n={len(recs)} trunc={n_trunc} "
         f"({dt:.0f}s) -> {pe}")


if __name__ == "__main__":
    main()
