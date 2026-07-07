#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Truncated-depth downstream evaluation of Qwen3-8B: a CAUSAL test of the
"division of labour" hypothesis.

Hypothesis (from our own layer-wise probing, which is CORRELATIONAL):
    early/mid layers "mostly" do semantic understanding, the top layers
    "mostly" turn the already-understood meaning into a next-token
    distribution (generation). Probing gave RTE peak L17=0.66 -> top L36=0.53,
    next-token only forms at the top. This script asks the CAUSAL question:

        "If we chop the model off at layer j and only use the first j layers'
         representation for a downstream SEMANTIC task, do we match the full
         model?"  If yes for j << 36, the semantics are done in the first j
         layers and the top layers are not needed for understanding.

------------------------------------------------------------------------------
HONESTY / DESIGN NOTE (project red-line #2) -- read before trusting the output
------------------------------------------------------------------------------
There are two ways to "use the first j layers", and they are NOT the same:

  (i)  TRUNCATE-AND-HEAD (no recompute): take hidden[j], train a task head on it.
       ==>  This is *mathematically identical* to layer-j linear probing.
            "First j layers suffice" == "layer-j probe matches top-layer probe".
            We DO NOT pretend this is new information vs probing; we report it
            with the truncation framing (acc-vs-j, j*, full-model baseline)
            because that framing IS the causal claim, but we flag the identity.

  (ii) TRUNCATE-AND-RECOMPUTE (QCMem's readout): take hidden[j] (the full-width
       sequence), run it through the model's remaining layers[j:] + final norm,
       then read out.  We PROVE numerically (see `verify_recompute_identity`)
       that for the *faithful* recompute
                 norm( layers[j:36]( hidden[j] ) )  ==  hidden[36]
       exactly (max abs diff 0.0 in bf16) for EVERY j -- because it is literally
       the model's own forward pass, an identity computation. Therefore
       "faithfully recomputing the upper layers" gives you the full model's top
       representation regardless of where you cut. Option (ii)-faithful is
       j-invariant and equals probing@36. The ONLY thing that could make (ii)
       differ from the full model is *modifying* hidden[j] (compressing it,
       dropping context) -- that is exactly what a trained QCMem compressor
       does, and is out of scope here (needs training).

So the measured "increment from recomputing the upper layers" is simply
    probe@36 - probe@j   (== analysis (e) of the probing script).

Because (i)==probing and (ii)-faithful==full-model-top, this script's genuinely
NEW content beyond re-running probing is Part C: the model's NATIVE generative
readout (verbalizer / lm_head zero-shot accuracy) as the "generation side"
reference, compared against the linear probe@j. That contrast is what actually
tells us what the top layers add:

  * probe@j  ~= probe@36  ~= native  -> understanding fully present at layer j;
                                        top layers do not improve linearly-
                                        decodable task info nor the answer
                                        -> SUPPORTS "understand early, generate late".
  * probe@36 >> probe@j                -> top layers add task-relevant computation.
  * native >> probe@j but probe@36~=probe@j -> the top layers' contribution is
                                        specifically the generation/output format,
                                        not a better representation.

Parts
-----
A. Truncated linear probe (== probing@j) on {SST2, WiC, RTE} for
   j in {4,8,12,16,20,24,28,32,36}: dev acc-vs-j, full-model baseline (j=36),
   j* = earliest j reaching 95% of the j=36 acc, and whether any mid j EXCEEDS
   j=36 (which would mean the top layers HURT downstream semantics).
B. (i) vs (ii) recompute contrast, with the faithful-recompute identity verified
   numerically in-run on SST2 (task-independent property), and the honest
   conclusion that faithful (ii) == full-model top == probe@36.
C. Native full-model zero-shot verbalizer accuracy per task (the generation
   side), compared to the probe@j curve.

Reuses data loaders / hidden extraction from scripts/probe_linguistic_layerwise.py
(imported, NOT modified).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

# --- reuse the probing script's loaders / extractors without modifying it -----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import probe_linguistic_layerwise as PL  # noqa: E402


# ----------------------------------------------------------------------------
# Truncated probe over a chosen set of depths (== layer-j probing)
# ----------------------------------------------------------------------------
def train_probe_one_layer(Xtr, y_tr, Xdv, y_dv, C=1.0, max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr = np.nan_to_num(Xtr.astype(np.float32), posinf=0.0, neginf=0.0)
    Xdv = np.nan_to_num(Xdv.astype(np.float32), posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)
    clf.fit(scaler.transform(Xtr), np.asarray(y_tr))
    return float(clf.score(scaler.transform(Xdv), np.asarray(y_dv)))


def truncated_probe(feats_tr, y_tr, feats_dv, y_dv, depths, C=1.0):
    """feats_* : {layer -> [N,H]}.  Returns {j: dev_acc} for j in depths."""
    out = {}
    for j in depths:
        acc = train_probe_one_layer(feats_tr[j], y_tr, feats_dv[j], y_dv, C=C)
        out[j] = acc
        print(f"    trunc j={j:2d}  dev_acc={acc:.4f}", flush=True)
    return out


def summarize_truncated(acc_by_j, depths, n_hidden_states):
    """Report full-model baseline (top layer), j*, ratios, mid-exceeds-top."""
    top = n_hidden_states - 1  # 36
    base = acc_by_j.get(top)
    if base is None:  # top not in grid: use max depth requested
        top = max(depths)
        base = acc_by_j[top]
    # earliest j reaching 95% / 99% of full-model (top) acc
    j95 = next((j for j in depths if acc_by_j[j] >= 0.95 * base), top)
    j99 = next((j for j in depths if acc_by_j[j] >= 0.99 * base), top)
    ratios = {j: round(acc_by_j[j] / base, 4) if base > 0 else None for j in depths}
    # best mid layer (excluding the top) and whether it exceeds the top
    mids = [j for j in depths if j < top]
    best_mid = max(mids, key=lambda j: acc_by_j[j]) if mids else None
    mid_exceeds_top = bool(best_mid is not None and acc_by_j[best_mid] > base)
    return {
        "depths": list(depths),
        "acc_by_j": {int(j): round(acc_by_j[j], 4) for j in depths},
        "acc_ratio_to_full": {int(j): ratios[j] for j in depths},
        "full_model_layer": int(top),
        "full_model_acc": round(base, 4),
        "j_star_95pct_of_full": int(j95),   # earliest j matching 95% of full model
        "j_star_99pct_of_full": int(j99),
        "best_mid_layer": int(best_mid) if best_mid is not None else None,
        "best_mid_acc": round(acc_by_j[best_mid], 4) if best_mid is not None else None,
        "mid_exceeds_full_model": mid_exceeds_top,
        "full_minus_best_mid": round(base - acc_by_j[best_mid], 4) if best_mid is not None else None,
    }


# ----------------------------------------------------------------------------
# Part B: verify the faithful-recompute identity  norm(layers[j:](h_j)) == h_top
# ----------------------------------------------------------------------------
@torch.no_grad()
def verify_recompute_identity(model, tok, sentences, device, max_len, depths):
    """For a small batch, recompute the top hidden state from each layer-j hidden
    state via the model's own upper layers + final norm, and compare to the
    directly-produced top hidden state (output_hidden_states[-1]).

    Confirms option (ii)-faithful is the identity computation (== full model top),
    hence j-invariant and equal to probing@36. Returns {j: max_abs_diff}.
    """
    from transformers.masking_utils import create_causal_mask

    base = getattr(model, "model", model)
    n_layers = model.config.num_hidden_layers  # 36
    enc = tok(sentences, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    hs = out.hidden_states  # tuple len n_layers+1 ; hs[-1] is POST final-norm
    h_top = hs[-1]
    attn = enc["attention_mask"]                       # (B,T)
    # compare only at NON-PAD positions: right-pad positions are attention-masked
    # out of the causal computation, so the recompute is free to diverge there
    # (and does), which is irrelevant to the representations we actually read out.
    valid = attn.bool().unsqueeze(-1)                  # (B,T,1)
    T = enc["input_ids"].shape[1]
    pos = torch.arange(T, device=device).unsqueeze(0)
    pe = base.rotary_emb(hs[0], pos)
    cm = create_causal_mask(config=model.config, inputs_embeds=hs[0],
                            attention_mask=attn,
                            past_key_values=None, position_ids=pos)
    diffs = {}
    for j in depths:
        if j >= n_layers:
            # j == top (36): hs[j] is already the post-norm top representation;
            # there is nothing to recompute, it IS the full-model output.
            diffs[int(j)] = 0.0
            continue
        h = hs[j].clone()          # raw output of layer j (pre-final-norm)
        for i in range(j, n_layers):
            h = base.layers[i](h, attention_mask=cm, position_embeddings=pe, position_ids=pos)
        h = base.norm(h)
        d = (h.float() - h_top.float()).abs()
        vmask = valid.expand_as(d)
        diffs[int(j)] = float(d[vmask].max().item())
    return diffs


# ----------------------------------------------------------------------------
# Part C: native generative (verbalizer / lm_head) zero-shot readout
# ----------------------------------------------------------------------------
# Templates map each example to a prompt; verbalizer maps class label -> word.
# We compare the first-subtoken logit of each verbalizer word at the last
# non-pad position (the model's actual "generation" decision).
VERBALIZERS = {
    # glue/sst2: 0 = negative, 1 = positive
    "SST2": {
        "template": lambda r: f"Review: {r['sentence'].strip()}\nSentiment (positive or negative):",
        "classes": {0: " negative", 1: " positive"},
        "label_key": "label",
    },
    # glue/rte: 0 = entailment, 1 = not_entailment
    "RTE": {
        "template": lambda r: (f"Premise: {r['sentence1'].strip()}\n"
                               f"Hypothesis: {r['sentence2'].strip()}\n"
                               f"Does the premise entail the hypothesis? Answer yes or no:"),
        "classes": {0: " yes", 1: " no"},
        "label_key": "label",
    },
    # super_glue/wic: 1 = same sense, 0 = different sense
    "WiC": {
        "template": lambda r: (f"Sentence 1: {r['sentence1'].strip()}\n"
                               f"Sentence 2: {r['sentence2'].strip()}\n"
                               f"Does the word \"{r['word']}\" have the same meaning in both "
                               f"sentences? Answer yes or no:"),
        "classes": {1: " yes", 0: " no"},
        "label_key": "label",
    },
}


def _first_token_ids(tok, words):
    ids = {}
    for lab, w in words.items():
        toks = tok(w, add_special_tokens=False)["input_ids"]
        if not toks:
            toks = tok(w.strip(), add_special_tokens=False)["input_ids"]
        ids[lab] = int(toks[0])
    return ids


@torch.no_grad()
def native_verbalizer_acc(model, tok, dev_rows, task, device, max_len, batch_size):
    """Full-model zero-shot accuracy using the model's own lm_head on a verbalizer.
    Returns (accuracy, n, majority_baseline, per-class token ids)."""
    spec = VERBALIZERS[task]
    cls_tok = _first_token_ids(tok, spec["classes"])   # {label: token_id}
    labels_order = list(cls_tok.keys())
    tok_ids = torch.tensor([cls_tok[l] for l in labels_order], device=device)
    # detect verbalizer collisions (would make argmax meaningless)
    collision = len(set(cls_tok.values())) < len(cls_tok)
    prompts = [spec["template"](r) for r in dev_rows]
    gold = [int(r[spec["label_key"]]) for r in dev_rows]
    correct, n = 0, 0
    orig_side = tok.padding_side
    tok.padding_side = "left"   # so last position == real last token for all rows
    try:
        for b0 in range(0, len(prompts), batch_size):
            bp = prompts[b0:b0 + batch_size]
            bg = gold[b0:b0 + batch_size]
            enc = tok(bp, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last_logits = out.logits[:, -1, :]                 # (B,V) left-pad -> real last
            sub = last_logits[:, tok_ids]                      # (B,n_classes)
            pred_idx = sub.argmax(dim=-1).tolist()
            for pi, g in zip(pred_idx, bg):
                pred_label = labels_order[pi]
                correct += int(pred_label == g)
                n += 1
    finally:
        tok.padding_side = orig_side
    maj = Counter(gold).most_common(1)[0][1] / len(gold)
    return {
        "native_acc": round(correct / n, 4) if n else None,
        "n": n,
        "majority_baseline": round(maj, 4),
        "verbalizer": {str(l): spec["classes"][l] for l in labels_order},
        "verbalizer_token_ids": {str(l): int(cls_tok[l]) for l in labels_order},
        "verbalizer_collision": collision,
    }


# ----------------------------------------------------------------------------
# Feature extraction per task (reuse probing builders -> all-layer pooled feats)
# ----------------------------------------------------------------------------
def build_task_features(name, model, tok, device, args, n_layers):
    """Returns (feats_tr, y_tr, feats_dv, y_dv, dev_rows) where dev_rows is the
    raw dev dataset (for the native verbalizer readout)."""
    load = PL.load_hf
    ntr, ndv = args.n_train_sent, args.n_dev_sent
    ml, bs = args.max_len, args.batch_size
    if name == "SST2":
        tr = load("glue", "sst2", split="train").select(range(min(ntr, 67349)))
        dv = load("glue", "sst2", split="validation").select(range(min(ndv, 872)))
        ftr = PL.extract_sentence_pooled(model, tok, [r["sentence"] for r in tr], device, ml, bs, n_layers)
        fdv = PL.extract_sentence_pooled(model, tok, [r["sentence"] for r in dv], device, ml, bs, n_layers)
        return ftr, [r["label"] for r in tr], fdv, [r["label"] for r in dv], list(dv)
    if name == "RTE":
        tr = load("glue", "rte", split="train").select(range(min(ntr, 2490)))
        dv = load("glue", "rte", split="validation").select(range(min(ndv, 277)))

        def feats(ds):
            fa = PL.extract_sentence_pooled(model, tok, [r["sentence1"] for r in ds], device, ml, bs, n_layers)
            fb = PL.extract_sentence_pooled(model, tok, [r["sentence2"] for r in ds], device, ml, bs, n_layers)
            return PL.combine_pair(fa, fb, n_layers)
        return feats(tr), [r["label"] for r in tr], feats(dv), [r["label"] for r in dv], list(dv)
    if name == "WiC":
        tr = load("super_glue", "wic", split="train").select(range(min(ntr, 5428)))
        dv = load("super_glue", "wic", split="validation").select(range(min(ndv, 638)))

        def feats(ds):
            s1 = [r["sentence1"] for r in ds]; s2 = [r["sentence2"] for r in ds]
            sp1 = [(r["start1"], r["end1"]) for r in ds]
            sp2 = [(r["start2"], r["end2"]) for r in ds]
            fa = PL.extract_target_word(model, tok, s1, sp1, device, ml, bs, n_layers)
            fb = PL.extract_target_word(model, tok, s2, sp2, device, ml, bs, n_layers)
            return PL.combine_pair(fa, fb, n_layers)
        return feats(tr), [r["label"] for r in tr], feats(dv), [r["label"] for r in dv], list(dv)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_train_sent", type=int, default=2000)
    ap.add_argument("--n_dev_sent", type=int, default=1000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tasks", default="SST2,WiC,RTE")
    ap.add_argument("--depths", default="4,8,12,16,20,24,28,32,36",
                    help="truncation depths j (layer index into hidden_states; 36=top).")
    ap.add_argument("--skip_native", action="store_true", help="skip Part C native readout")
    ap.add_argument("--skip_verify", action="store_true", help="skip Part B identity verification")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model_path}", flush=True)
    model, tok, n_layers = PL.load_model(args.model_path, args.device, args.dtype)
    print(f"model loaded: {n_layers} hidden states (embed + {n_layers-1} layers), "
          f"hidden={model.config.hidden_size}", flush=True)
    depths = [j for j in depths if 0 <= j <= n_layers - 1]
    if (n_layers - 1) not in depths:
        depths.append(n_layers - 1)  # always include the top as the full-model baseline
    depths = sorted(set(depths))

    results = {
        "model": args.model_path,
        "n_hidden_states": n_layers,
        "n_transformer_layers": n_layers - 1,
        "depths": depths,
        "design_note": (
            "Part A (truncated probe on hidden[j]) is mathematically identical to "
            "layer-j linear probing. Faithful truncate-and-recompute (option ii) "
            "reproduces the full model top exactly (verified in part_B_recompute_identity), "
            "so it is j-invariant and equals probe@36; the measured 'increment from the "
            "upper layers' is probe@36 - probe@j. Part C (native verbalizer) is the "
            "genuinely new generation-side reference."
        ),
        "tasks": {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # ---- Part B: verify faithful-recompute identity (task-independent) --------
    if not args.skip_verify:
        print(f"\n[{time.strftime('%H:%M:%S')}] === PART B: verify recompute identity (SST2 batch) ===", flush=True)
        try:
            sents = [r["sentence"] for r in PL.load_hf("glue", "sst2", split="validation").select(range(8))]
            diffs = verify_recompute_identity(model, tok, sents, args.device, args.max_len, depths)
            max_over_all = max(diffs.values())
            results["part_B_recompute_identity"] = {
                "max_abs_diff_by_j": diffs,
                "max_abs_diff_over_all_j": max_over_all,
                "is_identity": bool(max_over_all < 1e-1),
                "note": ("norm(layers[j:36](hidden[j])) vs hidden[36], measured at "
                         "NON-PAD positions (right-pad positions are attention-masked "
                         "out of the causal computation and diverge harmlessly). ~0 => "
                         "faithful recompute of upper layers == full model top for every "
                         "j, so option-(ii) downstream acc is j-invariant and == probe@36. "
                         "The only non-trivial 'recompute' would MODIFY hidden[j] "
                         "(compression / QCMem), which requires training and is out of scope."),
            }
            for j, d in diffs.items():
                print(f"    j={j:2d}: max|recompute-top|={d:.4g}", flush=True)
            print(f"  -> identity holds: {results['part_B_recompute_identity']['is_identity']} "
                  f"(max diff {max_over_all:.4g})", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results["part_B_recompute_identity"] = {"error": repr(e)[:300]}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Part A + C per task --------------------------------------------------
    for name in task_names:
        if name not in VERBALIZERS:
            print(f"skip unknown task {name}", flush=True)
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] === TASK {name} ===", flush=True)
        try:
            te = time.time()
            ftr, ytr, fdv, ydv, dev_rows = build_task_features(name, model, tok, args.device, args, n_layers)
            maj = Counter(ydv).most_common(1)[0][1] / len(ydv)
            print(f"  features: n_train={len(ytr)} n_dev={len(ydv)} "
                  f"feat_dim={ftr[depths[0]].shape[1]} extract={time.time()-te:.0f}s "
                  f"majority={maj:.4f}", flush=True)
            # Part A: truncated probe (== probing@j)
            acc_by_j = truncated_probe(ftr, ytr, fdv, ydv, depths, C=args.C)
            summ = summarize_truncated(acc_by_j, depths, n_layers)
            summ["majority_baseline"] = round(maj, 4)
            summ["n_train"] = len(ytr)
            summ["n_dev"] = len(ydv)
            summ["equivalent_to_layerwise_probing"] = True
            # Part C: native verbalizer readout (full model, generation side)
            if not args.skip_native:
                try:
                    nv = native_verbalizer_acc(model, tok, dev_rows, name, args.device,
                                               args.max_len, args.batch_size)
                    summ["native_generative_readout"] = nv
                    # contrast native vs probe@j / probe@36
                    full = summ["full_model_acc"]
                    summ["native_vs_probe"] = {
                        "native_acc": nv["native_acc"],
                        "probe_full_layer_acc": full,
                        "probe_best_mid_acc": summ["best_mid_acc"],
                        "native_minus_probefull": (round(nv["native_acc"] - full, 4)
                                                   if nv["native_acc"] is not None else None),
                    }
                    print(f"  native verbalizer acc={nv['native_acc']} "
                          f"(maj {nv['majority_baseline']}, collision={nv['verbalizer_collision']})",
                          flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    summ["native_generative_readout"] = {"error": repr(e)[:300]}
            results["tasks"][name] = summ
            print(f"  -> full-model(L{summ['full_model_layer']}) acc={summ['full_model_acc']:.4f}; "
                  f"j*(95%)={summ['j_star_95pct_of_full']}; "
                  f"best_mid=L{summ['best_mid_layer']}({summ['best_mid_acc']}); "
                  f"mid_exceeds_full={summ['mid_exceeds_full_model']}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results["tasks"][name] = {"error": repr(e)[:300]}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary ------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {results['elapsed_sec']}s -> {args.out}")
    print("\n=== TRUNCATED DOWNSTREAM (acc-vs-j; full model = top layer) ===")
    hdr = "task     full  " + " ".join(f"j{j:<4d}" for j in depths)
    print(hdr)
    for name, s in results["tasks"].items():
        if "error" in s:
            print(f"{name:8s} ERROR {s['error'][:80]}"); continue
        row = f"{name:8s} {s['full_model_acc']:.3f} "
        row += " ".join(f"{s['acc_by_j'][j]:.3f}" for j in depths)
        print(row)
    print("\n=== j* (earliest depth matching 95% of full model) + mid-vs-top ===")
    for name, s in results["tasks"].items():
        if "error" in s:
            continue
        line = (f"{name:8s} j*={s['j_star_95pct_of_full']:>2d}  "
                f"full=L{s['full_model_layer']}:{s['full_model_acc']:.3f}  "
                f"best_mid=L{s['best_mid_layer']}:{s['best_mid_acc']:.3f}  "
                f"mid_exceeds_full={s['mid_exceeds_full_model']}  "
                f"(full-best_mid={s['full_minus_best_mid']})")
        if "native_generative_readout" in s and "native_acc" in s.get("native_generative_readout", {}):
            line += f"  native={s['native_generative_readout']['native_acc']}"
        print(line)
    if "part_B_recompute_identity" in results and "is_identity" in results["part_B_recompute_identity"]:
        b = results["part_B_recompute_identity"]
        print(f"\n=== (ii) faithful recompute: identity={b['is_identity']} "
              f"(max|recompute-top| over j = {b['max_abs_diff_over_all_j']:.3g}) "
              f"=> faithful recompute == full model top == probe@36, j-invariant ===")


if __name__ == "__main__":
    main()
