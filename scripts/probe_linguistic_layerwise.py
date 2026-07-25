#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Layer-wise linguistic probing of Qwen3-8B (Tenney edge-probing style).

Real question (2026-07-07 user clarification): NOT "the first j layers saturate
semantics" (our own monotone j-sweep already falsified that). The hypothesis to
test is a DIVISION OF LABOUR:

    early/mid layers "mostly" do semantic understanding;
    the top few layers "mostly" do the AR generation strategy
    (turning the already-understood meaning into a next-token distribution).

Implications tested here:
  (e) Semantic ability should be near the top-layer level already in early/mid
      layers -> report j* = earliest layer reaching 95%/99% of the TOP-layer
      accuracy, and the top-layer increment over j* (small => top does no semantics).
  (f) Add a "generation/output" probe: standard logit-lens next-token top-1
      accuracy per layer (each layer's hidden -> model.norm -> lm_head). Overlay
      it against the semantic-understanding curves. If semantics saturate early
      while next-token only forms near the top, the two curves separate and that
      directly supports "mid = understand, top = generate".

Method
------
1. One forward pass of Qwen3-8B per example with output_hidden_states=True
   -> all 37 hidden states (embeddings + 36 transformer layers).
2. On each layer, train a lightweight linear probe (sklearn LogisticRegression
   on standardized features). Token-level tasks use per-word representations
   (mean of subword pieces), sentence tasks use mean-pooled representations.
3. Record dev accuracy per layer, per task; report peak layer, saturation layer,
   and (e) top-relative saturation j*.
4. (f) Compute the logit-lens next-token top-1 accuracy curve on natural text and
   compare semantic vs generation saturation depth.

Tasks span a lexical -> syntactic -> semantic gradient:
  POS      (batterydata/pos_tagging)            lexical    token-level
  DEPREL   (universal_dependencies en_ewt)      syntactic  token-level
  CoLA     (glue/cola, grammatical accept.)     syntactic  sentence-level
  WiC      (super_glue/wic, word sense)         semantic   target-word pair
  SST2     (glue/sst2, sentiment)               semantic   sentence-level
  RTE      (glue/rte, NLI entailment)           semantic   sentence-pair

Honesty (project red-line #2): report layers as measured. Do NOT massage data
toward the division-of-labour conclusion. If the semantic probe ALSO only
saturates at the top, or the two curves do NOT separate (next-token saturates
early too / semantics saturate late), say explicitly that the data does not
support the division-of-labour hypothesis.
"""
import argparse
import json
import os
import re
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
    Returns feats {layer: np.float32 [N, H]}, y (list[str]).
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
                    feats[l].append(v.numpy())
                ys.append(lab)
        if max_tokens and len(ys) >= max_tokens:
            break
    feats = {l: np.stack(v).astype(np.float32) for l, v in feats.items()}
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
                feats[l].append(pooled[bi].numpy())
    feats = {l: np.stack(v).astype(np.float32) for l, v in feats.items()}
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
                feats[l].append(v.numpy())
    feats = {l: np.stack(v).astype(np.float32) for l, v in feats.items()}
    return feats


def combine_pair(fa, fb, n_layers):
    """[a, b, |a-b|, a*b] per layer."""
    out = {}
    for l in range(n_layers):
        a = fa[l].astype(np.float32)
        b = fb[l].astype(np.float32)
        out[l] = np.concatenate([a, b, np.abs(a - b), a * b], axis=1).astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# (f) Logit-lens next-token accuracy curve (the "generation/output" side)
# ----------------------------------------------------------------------------
@torch.no_grad()
def logit_lens_nexttoken_acc(model, tok, sentences, device, max_len, batch_size, n_layers):
    """Standard logit-lens: for each layer's hidden state, apply the model's
    final RMSNorm (model.norm) then the lm_head, and measure top-1 next-token
    prediction accuracy against the actual next token.

    Rationale (division-of-labour test): the "understanding" side (semantic
    probes) is expected to saturate in early/mid layers, while the
    "generation/output" side (this curve) is expected to only take shape near
    the top. If the two curves separate, that supports "mid = understand,
    top = generate".

    Applying model.norm before lm_head is essential: without it, early-layer
    hidden states have a different norm scale than lm_head expects, which
    artificially depresses early-layer accuracy (a known logit-lens pitfall).

    Returns list[float] of per-layer top-1 next-token accuracy (length n_layers).
    """
    # Locate the final norm and the output projection robustly across HF layouts.
    base = getattr(model, "model", model)  # Qwen3ForCausalLM.model
    final_norm = getattr(base, "norm", None)
    lm_head = model.get_output_embeddings()
    if final_norm is None or lm_head is None:
        raise RuntimeError("could not locate model.norm / lm_head for logit-lens")

    # correct count / total count per layer (predict token t+1 from hidden at t)
    correct = np.zeros(n_layers, dtype=np.int64)
    total = 0
    for b0 in range(0, len(sentences), batch_size):
        bs = sentences[b0:b0 + batch_size]
        enc = tok(bs, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        input_ids = enc["input_ids"]           # (B,T)
        attn = enc["attention_mask"]           # (B,T)
        B, T = input_ids.shape
        # valid prediction positions: t in [0, T-2], and both t and t+1 are non-pad
        pred_mask = (attn[:, :-1] > 0) & (attn[:, 1:] > 0)   # (B,T-1)
        targets = input_ids[:, 1:]                            # (B,T-1)
        n_valid = int(pred_mask.sum().item())
        if n_valid == 0:
            continue
        total += n_valid
        for l in range(n_layers):
            h = out.hidden_states[l][:, :-1, :]               # (B,T-1,H)
            # cast to the norm's dtype for the projection
            h = h.to(final_norm.weight.dtype)
            logits = lm_head(final_norm(h))                    # (B,T-1,V)
            pred = logits.argmax(dim=-1)                       # (B,T-1)
            hit = (pred == targets) & pred_mask
            correct[l] += int(hit.sum().item())
    if total == 0:
        return [0.0] * n_layers
    return [float(correct[l] / total) for l in range(n_layers)]


def load_natural_text(tok, n_sent, max_len):
    """A batch of natural-language sentences for the logit-lens curve.
    Reuse WikiText-103 if available; otherwise fall back to GLUE/SST2 sentences,
    which are plain English text and adequate for next-token top-1 measurement."""
    sents = []
    try:
        ds = load_hf("wikitext", "wikitext-103-raw-v1", split="train")
        for r in ds:
            t = r["text"].strip()
            if len(t.split()) >= 12:  # skip headers / blank lines
                sents.append(t)
            if len(sents) >= n_sent:
                break
        if sents:
            print(f"    logit-lens corpus: wikitext-103 ({len(sents)} lines)", flush=True)
            return sents
    except Exception as e:
        print(f"    wikitext unavailable ({repr(e)[:120]}); falling back to SST2", flush=True)
    ds = load_hf("glue", "sst2", split="train").select(range(min(n_sent, 5000)))
    sents = [r["sentence"] for r in ds][:n_sent]
    print(f"    logit-lens corpus: glue/sst2 fallback ({len(sents)} sentences)", flush=True)
    return sents


# ----------------------------------------------------------------------------
# (P2) Knowledge-decodability logit-lens on MMLU (the "knowledge-readout" depth)
# ----------------------------------------------------------------------------
def load_mmlu_examples(n_mmlu):
    """Flan-style MMLU (cais/mmlu 'all' test split), mirroring
    load_task_examples('mmlu') in scripts/eval_olmo2_probe2_downstream.py so the
    knowledge口径 matches the Paper-B downstream MC eval: per-subject description +
    question + 'A./B./C./D.' lettered choices + 'Answer:'; gold = int answer index.

    Returns list[{"prompt": str, "gold": int, "n_choices": int}]. A fixed-seed
    (seed=0) shuffle is taken before selecting n_mmlu so the subset spans subjects
    rather than the alphabetically-first ones."""
    ds = load_hf("cais/mmlu", "all", split="test")
    if n_mmlu and n_mmlu < len(ds):
        ds = ds.shuffle(seed=0).select(range(n_mmlu))
    letters = ["A", "B", "C", "D"]
    out = []
    for ex in ds:
        subject_h = ex["subject"].replace("_", " ")
        desc = ("The following are multiple choice questions (with answers) "
                f"about {subject_h}.\n\n")
        ch = ex["choices"]
        body = "\n".join(f"{letters[i]}. {ch[i]}" for i in range(len(ch)))
        q = desc + ex["question"].strip() + "\n" + body + "\nAnswer:"
        out.append({"prompt": q, "gold": int(ex["answer"]), "n_choices": len(ch)})
    print(f"    MMLU logit-lens corpus: cais/mmlu all/test ({len(out)} questions)",
          flush=True)
    return out


@torch.no_grad()
def knowledge_logit_lens_mmlu(model, tok, examples, device, max_len, batch_size,
                              n_layers):
    """Per-layer logit-lens knowledge decodability on MMLU (4-choice answer-letter).

    For each layer L, take the hidden state at the LAST prompt position (right
    after 'Answer:'), apply the model's final norm then lm_head (logit lens;
    ``model.get_output_embeddings()`` handles tied AND untied heads identically),
    and score:
      * mmlu_acc         = argmax over the {A,B,C,D} letter-token logits hits gold,
      * mmlu_correct_ll  = mean full-vocab log-softmax log-prob of the gold letter.

    Applying the final norm before lm_head is essential (same known logit-lens
    pitfall handled by logit_lens_nexttoken_acc). Forward-only, no grad,
    bf16-autocast on CUDA. Returns (accs[n_layers], lls[n_layers], n_scored)."""
    base = getattr(model, "model", model)             # *ForCausalLM.model
    final_norm = getattr(base, "norm", None)
    if final_norm is None:  # some layouts name it differently
        final_norm = (getattr(base, "final_layernorm", None)
                      or getattr(base, "final_layer_norm", None))
    lm_head = model.get_output_embeddings()
    if final_norm is None or lm_head is None:
        raise RuntimeError("could not locate model.norm / lm_head for logit-lens")

    letters = ["A", "B", "C", "D"]
    # first continuation token of ' A'/' B'/... — 'Answer:' has no trailing space,
    # so the leading-space letter tokenises context-independently for BPE vocabs.
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in letters]

    use_cuda = "cuda" in str(device)
    correct = np.zeros(n_layers, dtype=np.int64)
    ll_sum = np.zeros(n_layers, dtype=np.float64)
    total = 0
    prev_side = tok.padding_side
    tok.padding_side = "right"                          # so last-real-token = len-1
    try:
        lid = torch.tensor(letter_ids, device=device)   # (4,)
        for b0 in range(0, len(examples), batch_size):
            batch = examples[b0:b0 + batch_size]
            prompts = [e["prompt"] for e in batch]
            golds = torch.tensor([e["gold"] for e in batch], device=device)
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            attn = enc["attention_mask"]
            B = attn.shape[0]
            last_idx = attn.sum(dim=1) - 1               # (B,) right-padded
            rows = torch.arange(B, device=device)
            with torch.autocast(device_type=("cuda" if use_cuda else "cpu"),
                                dtype=torch.bfloat16, enabled=use_cuda):
                out = model(**enc)
            for l in range(n_layers):
                h_last = out.hidden_states[l][rows, last_idx, :]   # (B,H)
                h_last = h_last.to(final_norm.weight.dtype)
                logits = lm_head(final_norm(h_last)).float()       # (B,V)
                letter_logits = logits[:, lid]                     # (B,4)
                pred = letter_logits.argmax(dim=-1)                # (B,)
                correct[l] += int((pred == golds).sum().item())
                logp = torch.log_softmax(logits, dim=-1)           # (B,V)
                gold_tok = lid[golds]                              # (B,)
                ll = logp[rows, gold_tok]                          # (B,)
                ll_sum[l] += float(ll.sum().item())
            total += B
    finally:
        tok.padding_side = prev_side
    if total == 0:
        return [0.0] * n_layers, [0.0] * n_layers, 0
    accs = [float(correct[l] / total) for l in range(n_layers)]
    lls = [float(ll_sum[l] / total) for l in range(n_layers)]
    return accs, lls, total


def run_knowledge_logit_lens(model, tok, args, n_layers):
    """--task knowledge_logit_lens driver (P2 two-depths): per-layer MMLU
    knowledge-decodability curve -> results/knowledge_logit_lens_<tag>.json."""
    proot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = os.path.basename(os.path.normpath(args.model_path))
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", tag)
    out = args.out or os.path.join(proot, "results",
                                   f"knowledge_logit_lens_{tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] === KNOWLEDGE LOGIT-LENS (MMLU) ===",
          flush=True)
    tk = time.time()
    examples = load_mmlu_examples(args.n_mmlu)
    accs, lls, n = knowledge_logit_lens_mmlu(
        model, tok, examples, args.device, args.max_len, args.batch_size, n_layers)
    for l, (a, ll) in enumerate(zip(accs, lls)):
        print(f"    layer {l:2d}  mmlu_acc={a:.4f}  correct_ll={ll:.4f}", flush=True)

    chance = 0.25
    denom = max(n_layers - 1, 1)
    per_layer = [{
        "layer_idx": l,
        "frac_depth": round(l / denom, 4),
        "mmlu_acc": round(accs[l], 4),
        "mmlu_correct_ll": round(lls[l], 4),
    } for l in range(n_layers)]

    peak_layer = int(np.argmax(accs))
    top_layer = n_layers - 1
    top_acc = float(accs[top_layer])
    j95 = _earliest_layer_reaching(accs, 0.95 * top_acc)
    j99 = _earliest_layer_reaching(accs, 0.99 * top_acc)
    # signal onset: earliest layer that beats chance by a clear margin (>=5 pts)
    onset = _earliest_layer_reaching(accs, chance + 0.05)
    summary = {
        "peak_layer": peak_layer,
        "peak_acc": round(float(accs[peak_layer]), 4),
        "top_layer": int(top_layer),
        "top_acc": round(top_acc, 4),
        "chance": chance,
        "sat95_top_layer": j95,
        "sat95_frac_depth": round((j95 if j95 is not None else top_layer) / denom, 3),
        "sat99_top_layer": j99,
        "sat99_frac_depth": round((j99 if j99 is not None else top_layer) / denom, 3),
        "onset_layer": onset,
        "onset_frac_depth": (round(onset / denom, 3) if onset is not None else None),
    }
    results = {
        "meta": {
            "task": "knowledge_logit_lens",
            "model": args.model_path,
            "model_tag": tag,
            "dtype": args.dtype,
            "n_mmlu": n,
            "n_layers": n_layers,
            "chance": chance,
            "note": "logit-lens per-layer MMLU (flan-style, cais/mmlu all/test); "
                    "hidden@last-prompt-pos -> final_norm -> lm_head; argmax over "
                    "{A,B,C,D} letter tokens = mmlu_acc; gold-letter full-vocab "
                    "log-softmax = mmlu_correct_ll. Forward-only, no grad.",
        },
        "summary": summary,
        "per_layer": per_layer,
        "extract_sec": round(time.time() - tk, 1),
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> knowledge decodability: peak L{peak_layer} "
          f"(acc {accs[peak_layer]:.4f}), top(L{top_layer})={top_acc:.4f}, "
          f"onset@L{onset}, 95%@L{j95} -> {out}", flush=True)
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
        Xtr = np.nan_to_num(feats_tr[l].astype(np.float32), posinf=0.0, neginf=0.0)
        Xdv = np.nan_to_num(feats_dv[l].astype(np.float32), posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)
        clf.fit(scaler.transform(Xtr), y_tr)
        acc = float(clf.score(scaler.transform(Xdv), y_dv))
        accs.append(acc)
        print(f"    layer {l:2d}  dev_acc={acc:.4f}", flush=True)
    return accs


def _earliest_layer_reaching(accs, target):
    """Earliest layer index whose acc >= target; None if never."""
    for l, a in enumerate(accs):
        if a >= target:
            return int(l)
    return None


def summarize(accs, sat_frac=0.95):
    peak_layer = int(np.argmax(accs))
    peak_acc = float(accs[peak_layer])
    thresh = sat_frac * peak_acc
    sat_layer = next((l for l, a in enumerate(accs) if a >= thresh), peak_layer)

    # --- (e) top-layer-relative saturation (division-of-labour test) ---------
    # j* = earliest layer reaching 95% / 99% of the *final (top) layer* accuracy.
    # If j* << top layer AND top-layer increment over j* is tiny, semantics are
    # essentially resolved in early/mid layers -> supports "top does not do semantics".
    top_layer = len(accs) - 1
    top_acc = float(accs[top_layer])
    j95 = _earliest_layer_reaching(accs, 0.95 * top_acc)
    j99 = _earliest_layer_reaching(accs, 0.99 * top_acc)
    if j95 is None:
        j95 = top_layer
    if j99 is None:
        j99 = top_layer
    return {
        "peak_layer": peak_layer,
        "peak_acc": peak_acc,
        "saturation_layer": int(sat_layer),
        "saturation_thresh": float(thresh),
        # top-relative (analysis e)
        "top_layer": int(top_layer),
        "top_acc": round(top_acc, 4),
        "sat95_top_layer": int(j95),          # j* @ 95% of top-layer acc
        "sat99_top_layer": int(j99),          # j* @ 99% of top-layer acc
        "top_increment_over_j95": round(top_acc - float(accs[j95]), 4),
        "top_increment_over_j99": round(top_acc - float(accs[j99]), 4),
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
    ap.add_argument("--out", default=None,
                    help="output JSON. Required for the linguistic probe suite; "
                         "for --task knowledge_logit_lens it defaults to "
                         "results/knowledge_logit_lens_<model_tag>.json.")
    ap.add_argument("--task", default=None, choices=["knowledge_logit_lens"],
                    help="optional SINGLE mode. 'knowledge_logit_lens' runs ONLY "
                         "the per-layer MMLU logit-lens knowledge-decodability "
                         "probe (P2 two-depths thesis) and exits. Omit for the "
                         "default linguistic + next-token probe suite.")
    ap.add_argument("--n_mmlu", type=int, default=1000,
                    help="MMLU test questions (seed=0 shuffled subset of cais/mmlu "
                         "'all') for --task knowledge_logit_lens (default 1000).")
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
    ap.add_argument("--n_logitlens_sent", type=int, default=500,
                    help="natural-text sentences for the logit-lens next-token curve (f)")
    ap.add_argument("--skip_logitlens", action="store_true",
                    help="skip the logit-lens next-token comparison curve")
    args = ap.parse_args()
    if args.task is None and not args.out:
        ap.error("--out is required for the linguistic probe suite "
                 "(only --task knowledge_logit_lens auto-defaults the path).")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading model {args.model_path}", flush=True)
    model, tok, n_layers = load_model(args.model_path, args.device, args.dtype)
    print(f"model loaded: {n_layers} hidden states (embed + {n_layers-1} layers), "
          f"hidden={model.config.hidden_size}", flush=True)

    # --- P2 single-mode: per-layer MMLU knowledge-decodability logit-lens ---
    if args.task == "knowledge_logit_lens":
        run_knowledge_logit_lens(model, tok, args, n_layers)
        print(f"\n[{time.strftime('%H:%M:%S')}] DONE (knowledge_logit_lens) in "
              f"{time.time()-t0:.1f}s", flush=True)
        return

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

    # ------------------------------------------------------------------
    # (f) Logit-lens next-token top-1 accuracy curve + division-of-labour test
    # ------------------------------------------------------------------
    if not args.skip_logitlens:
        print(f"\n[{time.strftime('%H:%M:%S')}] === LOGIT-LENS next-token curve ===", flush=True)
        try:
            tll = time.time()
            sents = load_natural_text(tok, args.n_logitlens_sent, args.max_len)
            ll_acc = logit_lens_nexttoken_acc(model, tok, sents, args.device,
                                              args.max_len, args.batch_size, n_layers)
            for l, a in enumerate(ll_acc):
                print(f"    layer {l:2d}  nexttok_top1={a:.4f}", flush=True)
            top = len(ll_acc) - 1
            ll_summary = {
                "per_layer_nexttoken_top1": [round(a, 4) for a in ll_acc],
                "top_layer": int(top),
                "top_acc": round(float(ll_acc[top]), 4),
                "sat95_top_layer": _earliest_layer_reaching(ll_acc, 0.95 * ll_acc[top]),
                "sat99_top_layer": _earliest_layer_reaching(ll_acc, 0.99 * ll_acc[top]),
                "n_sentences": len(sents),
                "extract_sec": round(time.time() - tll, 1),
            }
            results["logit_lens_nexttoken"] = ll_summary
            print(f"  -> next-token top1: top(L{top})={ll_acc[top]:.4f}, "
                  f"95%@L{ll_summary['sat95_top_layer']}, "
                  f"99%@L{ll_summary['sat99_top_layer']}", flush=True)

            # --- division-of-labour comparison: semantic understanding vs generation ---
            sem_tasks = [n for n, s in results["tasks"].items()
                         if isinstance(s, dict) and s.get("category") == "semantic"
                         and "sat95_top_layer" in s]
            if sem_tasks:
                sem_j95 = [results["tasks"][n]["sat95_top_layer"] for n in sem_tasks]
                sem_mean_j95 = float(np.mean(sem_j95))
                ll_j95 = ll_summary["sat95_top_layer"]
                # curves "separate" if semantics saturate meaningfully earlier
                # (in fractional depth) than the next-token/generation curve
                frac_sem = sem_mean_j95 / (n_layers - 1)
                frac_ll = (ll_j95 if ll_j95 is not None else (n_layers - 1)) / (n_layers - 1)
                separated = (frac_ll - frac_sem) >= 0.15  # >=15% of depth later
                verdict = ("SUPPORTS division-of-labour (semantics saturate early/mid, "
                           "next-token forms near the top)" if separated else
                           "DOES NOT clearly support division-of-labour "
                           "(curves do not separate by >=15% of depth)")
                results["division_of_labour"] = {
                    "semantic_tasks": sem_tasks,
                    "semantic_mean_sat95_top_layer": round(sem_mean_j95, 2),
                    "semantic_sat95_frac_depth": round(frac_sem, 3),
                    "nexttoken_sat95_top_layer": ll_j95,
                    "nexttoken_sat95_frac_depth": round(frac_ll, 3),
                    "gap_frac_depth": round(frac_ll - frac_sem, 3),
                    "curves_separate": bool(separated),
                    "verdict": verdict,
                    "honesty_note": "threshold-based heuristic; inspect per-layer curves "
                                    "in both fields before drawing conclusions.",
                }
                print(f"  -> DIVISION-OF-LABOUR: sem 95%@~L{sem_mean_j95:.1f} "
                      f"(depth {frac_sem:.2f}) vs nexttok 95%@L{ll_j95} "
                      f"(depth {frac_ll:.2f}) -> {verdict}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["logit_lens_nexttoken"] = {"error": repr(e)[:300]}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {results['elapsed_sec']}s -> {args.out}")
    print("\n=== SUMMARY (peak / sat / top-relative saturation per task) ===")
    print(f"{'task':8s} {'cat':10s} {'peak':>5s} {'peakacc':>8s} {'top':>4s} "
          f"{'j95*':>5s} {'j99*':>5s} {'dTop95':>7s} {'maj':>6s}")
    for name, s in results["tasks"].items():
        if "error" in s:
            print(f"{name:8s} ERROR {s['error']}")
            continue
        print(f"{name:8s} {s['category']:10s} {s['peak_layer']:5d} "
              f"{s['peak_acc']:8.4f} {s['top_layer']:4d} "
              f"{s['sat95_top_layer']:5d} {s['sat99_top_layer']:5d} "
              f"{s['top_increment_over_j95']:7.4f} {s['majority_baseline']:6.3f}")

    if "division_of_labour" in results:
        d = results["division_of_labour"]
        print("\n=== (f) DIVISION-OF-LABOUR (semantic understanding vs next-token generation) ===")
        print(f"  semantic tasks: {d['semantic_tasks']}")
        print(f"  semantic 95%-of-top saturates @ ~L{d['semantic_mean_sat95_top_layer']} "
              f"(depth {d['semantic_sat95_frac_depth']:.2f})")
        print(f"  next-token 95%-of-top saturates @ L{d['nexttoken_sat95_top_layer']} "
              f"(depth {d['nexttoken_sat95_frac_depth']:.2f})")
        print(f"  gap (depth): {d['gap_frac_depth']:+.2f}  ->  {d['verdict']}")


if __name__ == "__main__":
    main()
