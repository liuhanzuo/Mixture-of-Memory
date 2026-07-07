#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Layer-wise linguistic probing of Qwen3-8B (Tenney edge-probing style).

Question: Where does Qwen3-8B concentrate semantic processing? Is the claim
"the first ~12 layers focus on semantics" supported by data?

Method
------
1. One forward pass of Qwen3-8B per example with output_hidden_states=True
   -> all 37 hidden states (embeddings + 36 transformer layers).
2. On each layer, train a lightweight linear probe (sklearn LogisticRegression
   on standardized features). Token-level tasks use per-word representations
   (mean of subword pieces), sentence tasks use mean-pooled representations.
3. Record dev accuracy per layer, per task; report peak layer and the earliest
   "saturation" layer that reaches 95% of the peak accuracy.

Tasks span a lexical -> syntactic -> semantic gradient:
  POS      (batterydata/pos_tagging)            lexical    token-level
  DEPREL   (universal_dependencies en_ewt)      syntactic  token-level
  CoLA     (glue/cola, grammatical accept.)     syntactic  sentence-level
  WiC      (super_glue/wic, word sense)         semantic   target-word pair
  SST2     (glue/sst2, sentiment)               semantic   sentence-level
  RTE      (glue/rte, NLI entailment)           semantic   sentence-pair

Honesty (project red-line #2): report peak layers as measured. Do NOT massage
data toward the desired "first 12 layers = semantic" conclusion. If semantics
peak/saturate deeper or shallower, say so.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Model / tokenizer
# ----------------------------------------------------------------------------
def load_model(model_path, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1  # + embedding layer
    return model, tok, n_layers


# ----------------------------------------------------------------------------
# Hidden-state extraction helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def _forward_hidden(model, tok, texts, device, max_len):
    """Return (hidden_states tuple [L x (B,T,H)] on cpu-fp32, offset_mapping, attn_mask)."""
    enc = tok(
        texts,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )
    offsets = enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    hs = [h.float().cpu() for h in out.hidden_states]  # each (B,T,H)
    return hs, offsets.cpu().numpy(), enc["attention_mask"].cpu().numpy()


def _spans_to_token_idx(offsets_row, attn_row, char_start, char_end):
    """Indices of subword tokens (non-pad) overlapping char span [start,end)."""
    idx = []
    for t, (s, e) in enumerate(offsets_row):
        if attn_row[t] == 0:
            continue
        if s == e:  # empty / special
            continue
        if s < char_end and e > char_start:  # overlap
            idx.append(t)
    return idx


def extract_token_level(model, tok, sents_words, sents_labels, device, max_len,
                        batch_size, n_layers, max_tokens=None):
    """Per-word representations for token classification (POS / deprel).

    sents_words:  list[list[str]]   words per sentence
    sents_labels: list[list[str]]   label per word (aligned)
    Returns feats {layer: np.float16 [N, H]}, y (list[str]).
    """
    feats = {l: [] for l in range(n_layers)}
    ys = []
    for b0 in range(0, len(sents_words), batch_size):
        bw = sents_words[b0:b0 + batch_size]
        bl = sents_labels[b0:b0 + batch_size]
        texts, spans_per_sent = [], []
        for words in bw:
            text, spans, cur = "", [], 0
            for w in words:
                spans.append((cur, cur + len(w)))
                cur += len(w) + 1  # +1 for the joining space
                text += w + " "
            texts.append(text.rstrip())
            spans_per_sent.append(spans)
        hs, offsets, attn = _forward_hidden(model, tok, texts, device, max_len)
        for bi, (spans, labels) in enumerate(zip(spans_per_sent, bl)):
            for (cs, ce), lab in zip(spans, labels):
                tok_idx = _spans_to_token_idx(offsets[bi], attn[bi], cs, ce)
                if not tok_idx:
                    continue  # word truncated away
                for l in range(n_layers):
                    v = hs[l][bi, tok_idx, :].mean(dim=0)
                    feats[l].append(v.half().numpy())
                ys.append(lab)
        if max_tokens and len(ys) >= max_tokens:
            break
    feats = {l: np.stack(v).astype(np.float16) for l, v in feats.items()}
    return feats, ys


def extract_sentence_pooled(model, tok, sentences, device, max_len, batch_size, n_layers):
    """Mean-pooled sentence representation over non-pad tokens. {layer: [N,H]}."""
    feats = {l: [] for l in range(n_layers)}
    for b0 in range(0, len(sentences), batch_size):
        bs = sentences[b0:b0 + batch_size]
        hs, _off, attn = _forward_hidden(model, tok, bs, device, max_len)
        mask = torch.tensor(attn).float().unsqueeze(-1)  # (B,T,1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        for l in range(n_layers):
            pooled = (hs[l] * mask).sum(dim=1) / denom  # (B,H)
            for bi in range(pooled.shape[0]):
                feats[l].append(pooled[bi].half().numpy())
    feats = {l: np.stack(v).astype(np.float16) for l, v in feats.items()}
    return feats


def extract_target_word(model, tok, sentences, spans, device, max_len, batch_size, n_layers):
    """Mean-pooled representation of a target char span per sentence (WiC)."""
    feats = {l: [] for l in range(n_layers)}
    for b0 in range(0, len(sentences), batch_size):
        bs = sentences[b0:b0 + batch_size]
        bsp = spans[b0:b0 + batch_size]
        hs, offsets, attn = _forward_hidden(model, tok, bs, device, max_len)
        for bi, (cs, ce) in enumerate(bsp):
            tok_idx = _spans_to_token_idx(offsets[bi], attn[bi], cs, ce)
            if not tok_idx:  # fallback: last non-pad token
                nz = np.nonzero(attn[bi])[0]
                tok_idx = [int(nz[-1])] if len(nz) else [0]
            for l in range(n_layers):
                v = hs[l][bi, tok_idx, :].mean(dim=0)
                feats[l].append(v.half().numpy())
    feats = {l: np.stack(v).astype(np.float16) for l, v in feats.items()}
    return feats


def combine_pair(fa, fb, n_layers):
    """[a, b, |a-b|, a*b] per layer."""
    out = {}
    for l in range(n_layers):
        a = fa[l].astype(np.float32)
        b = fb[l].astype(np.float32)
        out[l] = np.concatenate([a, b, np.abs(a - b), a * b], axis=1).astype(np.float16)
    return out


# ----------------------------------------------------------------------------
# Probe training
# ----------------------------------------------------------------------------
def train_probes(feats_tr, y_tr, feats_dv, y_dv, n_layers, C=1.0, max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_tr = np.asarray(y_tr)
    y_dv = np.asarray(y_dv)
    accs = []
    for l in range(n_layers):
        Xtr = feats_tr[l].astype(np.float32)
        Xdv = feats_dv[l].astype(np.float32)
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)
        clf.fit(scaler.transform(Xtr), y_tr)
        acc = float(clf.score(scaler.transform(Xdv), y_dv))
        accs.append(acc)
        print(f"    layer {l:2d}  dev_acc={acc:.4f}", flush=True)
    return accs


def summarize(accs, sat_frac=0.95):
    peak_layer = int(np.argmax(accs))
    peak_acc = float(accs[peak_layer])
    thresh = sat_frac * peak_acc
    sat_layer = next((l for l, a in enumerate(accs) if a >= thresh), peak_layer)
    return {
        "peak_layer": peak_layer,
        "peak_acc": peak_acc,
        "saturation_layer": int(sat_layer),
        "saturation_thresh": float(thresh),
        "per_layer_acc": [round(a, 4) for a in accs],
    }


# ----------------------------------------------------------------------------
# Task builders
# ----------------------------------------------------------------------------
def load_hf(*args, **kw):
    from datasets import load_dataset
    return load_dataset(*args, **kw)


def build_pos(model, tok, dev, max_len, bs, n_layers, n_train, n_dev):
    tr = load_hf("batterydata/pos_tagging", split="train")
    dv = load_hf("batterydata/pos_tagging", split="test")
    tr = tr.select(range(min(n_train, len(tr))))
    dv = dv.select(range(min(n_dev, len(dv))))
    ftr, ytr = extract_token_level(model, tok, [r["words"] for r in tr], [r["labels"] for r in tr],
                                   dev, max_len, bs, n_layers)
    fdv, ydv = extract_token_level(model, tok, [r["words"] for r in dv], [r["labels"] for r in dv],
                                   dev, max_len, bs, n_layers)
    return ftr, ytr, fdv, ydv


def build_deprel(model, tok, dev, max_len, bs, n_layers, n_train, n_dev):
    tr = load_hf("universal-dependencies/universal_dependencies", "en_ewt", split="train")
    dv = load_hf("universal-dependencies/universal_dependencies", "en_ewt", split="dev")
    tr = tr.select(range(min(n_train, len(tr))))
    dv = dv.select(range(min(n_dev, len(dv))))
    def rels(r):
        return [d.split(":")[0] for d in r["deprel"]]  # core relation
    ftr, ytr = extract_token_level(model, tok, [r["tokens"] for r in tr], [rels(r) for r in tr],
                                   dev, max_len, bs, n_layers)
    fdv, ydv = extract_token_level(model, tok, [r["tokens"] for r in dv], [rels(r) for r in dv],
                                   dev, max_len, bs, n_layers)
    return ftr, ytr, fdv, ydv


def build_single_sentence(hf_args, text_key, model, tok, dev, max_len, bs, n_layers, n_train, n_dev):
    tr = load_hf(*hf_args, split="train")
    dv = load_hf(*hf_args, split="validation")
    tr = tr.select(range(min(n_train, len(tr))))
    dv = dv.select(range(min(n_dev, len(dv))))
    ftr = extract_sentence_pooled(model, tok, [r[text_key] for r in tr], dev, max_len, bs, n_layers)
    fdv = extract_sentence_pooled(model, tok, [r[text_key] for r in dv], dev, max_len, bs, n_layers)
    return ftr, [r["label"] for r in tr], fdv, [r["label"] for r in dv]


def build_pair_sentence(hf_args, k1, k2, model, tok, dev, max_len, bs, n_layers, n_train, n_dev):
    tr = load_hf(*hf_args, split="train")
    dv = load_hf(*hf_args, split="validation")
    tr = tr.select(range(min(n_train, len(tr))))
    dv = dv.select(range(min(n_dev, len(dv))))
    def feats(ds):
        fa = extract_sentence_pooled(model, tok, [r[k1] for r in ds], dev, max_len, bs, n_layers)
        fb = extract_sentence_pooled(model, tok, [r[k2] for r in ds], dev, max_len, bs, n_layers)
        return combine_pair(fa, fb, n_layers)
    return feats(tr), [r["label"] for r in tr], feats(dv), [r["label"] for r in dv]


def build_wic(model, tok, dev, max_len, bs, n_layers, n_train, n_dev):
    tr = load_hf("super_glue", "wic", split="train")
    dv = load_hf("super_glue", "wic", split="validation")
    tr = tr.select(range(min(n_train, len(tr))))
    dv = dv.select(range(min(n_dev, len(dv))))
    def feats(ds):
        s1 = [r["sentence1"] for r in ds]
        s2 = [r["sentence2"] for r in ds]
        sp1 = [(r["start1"], r["end1"]) for r in ds]
        sp2 = [(r["start2"], r["end2"]) for r in ds]
        fa = extract_target_word(model, tok, s1, sp1, dev, max_len, bs, n_layers)
        fb = extract_target_word(model, tok, s2, sp2, dev, max_len, bs, n_layers)
        return combine_pair(fa, fb, n_layers)
    return feats(tr), [r["label"] for r in tr], feats(dv), [r["label"] for r in dv]


TASKS = {
    "POS":    {"category": "lexical",   "level": "token",    "builder": "pos"},
    "DEPREL": {"category": "syntactic", "level": "token",    "builder": "deprel"},
    "CoLA":   {"category": "syntactic", "level": "sentence", "builder": "cola"},
    "WiC":    {"category": "semantic",  "level": "word-pair","builder": "wic"},
    "SST2":   {"category": "semantic",  "level": "sentence", "builder": "sst2"},
    "RTE":    {"category": "semantic",  "level": "pair",     "builder": "rte"},
}


def run_task(name, model, tok, dev, args, n_layers):
    b = TASKS[name]["builder"]
    ntr_tok, ndv_tok = args.n_train_token, args.n_dev_token
    ntr_sent, ndv_sent = args.n_train_sent, args.n_dev_sent
    if b == "pos":
        return build_pos(model, tok, dev, args.max_len, args.batch_size, n_layers, ntr_tok, ndv_tok)
    if b == "deprel":
        return build_deprel(model, tok, dev, args.max_len, args.batch_size, n_layers, ntr_tok, ndv_tok)
    if b == "cola":
        return build_single_sentence(("glue", "cola"), "sentence", model, tok, dev,
                                     args.max_len, args.batch_size, n_layers, ntr_sent, ndv_sent)
    if b == "sst2":
        return build_single_sentence(("glue", "sst2"), "sentence", model, tok, dev,
                                     args.max_len, args.batch_size, n_layers, ntr_sent, ndv_sent)
    if b == "wic":
        return build_wic(model, tok, dev, args.max_len, args.batch_size, n_layers, ntr_sent, ndv_sent)
    if b == "rte":
        return build_pair_sentence(("glue", "rte"), "sentence1", "sentence2", model, tok, dev,
                                   args.max_len, args.batch_size, n_layers, ntr_sent, ndv_sent)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_train_token", type=int, default=700)   # sentences (token tasks)
    ap.add_argument("--n_dev_token", type=int, default=400)
    ap.add_argument("--n_train_sent", type=int, default=2000)   # examples (sentence tasks)
    ap.add_argument("--n_dev_sent", type=int, default=1000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tasks", default="POS,DEPREL,CoLA,WiC,SST2,RTE")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading model {args.model_path}", flush=True)
    model, tok, n_layers = load_model(args.model_path, args.device, args.dtype)
    print(f"model loaded: {n_layers} hidden states (embed + {n_layers-1} layers), "
          f"hidden={model.config.hidden_size}", flush=True)

    results = {"model": args.model_path, "n_layers": n_layers, "tasks": {}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for name in args.tasks.split(","):
        name = name.strip()
        if name not in TASKS:
            print(f"skip unknown task {name}", flush=True)
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] === TASK {name} "
              f"({TASKS[name]['category']}) ===", flush=True)
        try:
            te = time.time()
            ftr, ytr, fdv, ydv = run_task(name, model, tok, args.device, args, n_layers)
            print(f"  features: n_train={len(ytr)} n_dev={len(ydv)} "
                  f"feat_dim={ftr[0].shape[1]} extract={time.time()-te:.0f}s", flush=True)
            maj = Counter(ydv).most_common(1)[0][1] / len(ydv)
            accs = train_probes(ftr, ytr, fdv, ydv, n_layers, C=args.C)
            summ = summarize(accs)
            summ.update({
                "category": TASKS[name]["category"],
                "level": TASKS[name]["level"],
                "n_train": len(ytr),
                "n_dev": len(ydv),
                "n_classes": len(set(ytr)),
                "majority_baseline": round(maj, 4),
                "feat_dim": int(ftr[0].shape[1]),
            })
            results["tasks"][name] = summ
            print(f"  -> peak layer {summ['peak_layer']} (acc {summ['peak_acc']:.4f}), "
                  f"saturation layer {summ['saturation_layer']}, "
                  f"majority {maj:.4f}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["tasks"][name] = {"error": repr(e)[:300]}
        # write incrementally so partial results survive
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {results['elapsed_sec']}s -> {args.out}")
    print("\n=== SUMMARY (peak / saturation layer per task) ===")
    print(f"{'task':8s} {'cat':10s} {'peak':>5s} {'peakacc':>8s} {'sat':>4s} {'maj':>6s}")
    for name, s in results["tasks"].items():
        if "error" in s:
            print(f"{name:8s} ERROR {s['error']}")
            continue
        print(f"{name:8s} {s['category']:10s} {s['peak_layer']:5d} "
              f"{s['peak_acc']:8.4f} {s['saturation_layer']:4d} {s['majority_baseline']:6.3f}")


if __name__ == "__main__":
    main()
