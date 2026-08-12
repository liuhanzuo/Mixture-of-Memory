#!/usr/bin/env python3
"""Dual-INTERFACE (letter | content) MC eval for NON-MMLU benchmarks — paperC
gate-2 full replication.

Why this file exists
--------------------
`scripts/eval_olmo2_mmlu_content.py` established paperC's headline on cais/mmlu:
the *letter* interface (question + `A./B./C./D.` labelled options, score the bare
letter token after `Answer:`) collapses to at-or-below its own best-constant
floor under structural damage, while the *content* interface (label-free, score
the option TEXT) does not collapse the same way.

paperC's second open defect is that the non-MMLU replication (A01 gate-2) used a
DIFFERENT interface contrast: **raw sum-LL vs length-normalised acc_norm**, which
is analogous to but NOT identical with letter-vs-content. This harness closes
that gap: it reproduces MMLU's *exact* letter-vs-content contrast on non-MMLU MC
benchmarks, so the replication no longer needs an "the interfaces are actually
different" footnote.

Interface construction (MMLU-parallel, by design)
-------------------------------------------------
For every task the two prompts share the IDENTICAL question view and differ ONLY
in (a) whether the labelled option body is shown and (b) what the candidate is:

  content : `Question: <q>\nAnswer:`                        cand = " <option text>"
  letter  : `Question: <q>\nA. <t0>\nB. <t1>...\nAnswer:`   cand = " A" / " B" / ...

This is exactly the relation MMLU's harness uses (`letter = desc + q + "\n" +
body + "\nAnswer:"`, `content = desc + q + "\nAnswer:"`), with MMLU's per-subject
description replaced by the neutral `Question: ` stem — i.e. MMLU's
`--content_desc none` template family, applied to both interfaces so the contrast
stays clean. `--desc_style flan` prepends a task-level flan description to BOTH
prompts (prompt-sensitivity check; it does not change the interface contrast).

Free cross-validation: for `arc_challenge` / `arc_easy` / `commonsense_qa` the
content prompt above is BYTE-IDENTICAL to the published prompt in
`scripts/eval_olmo2_probe2_downstream.py::load_task_examples`, so this harness's
`content_raw` accuracy must reproduce the published `correct` field item-for-item
on those three tasks. `openbookqa` is the one deviation: the published loader uses
the bare `question_stem` with NO `Answer:` cue, so OBQA's `content_raw` here will
NOT equal the published ledger — that deviation is deliberate (it buys the clean
letter/content relation) and must be reported.

Nulls are NOT computed here
---------------------------
This file only produces paired per-item records. Every construct-appropriate null
(best-constant letter, longest-option × 5 tie conventions), every paired bootstrap
and every McNemar test is computed on CPU by
`proposal/active/A01-null-calibration-methodology/code/a01_gate2_letter_content_nulls.py`
from these records, so the statistics are re-runnable without a GPU.

Protocol discipline (paperC-wide, non-negotiable)
------------------------------------------------
* `chat_template=False`, `add_bos=0` — OLMo-2 is a BASE LM with no SFT/RL.
* fp32 master weights, bf16-autocast forward (identical to the MMLU harness and
  to training).
* Arch construction imported verbatim from `eval_olmo2_probe2_ppl.py`; the
  (context, continuation) tokeniser imported verbatim from
  `eval_olmo2_probe2_downstream.py`. No drift.
* item_id = shard_index + local_idx * num_shards (same stable-id rule as every
  other harness in this repo) so cross-arm and cross-interface pairing lines up.
* An item is *valid* iff no candidate in EITHER interface produced a non-finite
  score, so letter and content are always scored on the identical item set.

Modes
-----
  (default)  score one shard of one or more tasks under one model load
  --merge    concatenate shards per task, assert shard completeness, write
             summary.json (accuracies + within-arm letter-vs-content pairing)
  --selftest CPU-only end-to-end maths check (tiny random OLMo-2, no weights)
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NO arch drift / NO tokenisation drift: reuse the exact same builders + encoder
# the letter-protocol and MMLU-content harnesses use.
from eval_olmo2_probe2_ppl import (  # noqa: E402
    _log,
    load_base_model,
    load_base_model_any_family,
    load_pruned_model,
    load_truncated_any_family,
)
# NOTE the Qwen3 pruned loader is imported LAZILY inside _load_pruned_dispatch,
# NOT here. Importing `eval_qwen3_probe2_ppl` at module scope would add
# Qwen3Config/Qwen3ForCausalLM to the import graph of every OLMo-2 run, so an
# OLMo cell could start failing because of an unrelated Qwen3 transformers
# change. Every archived number came from a process that never imported those
# classes; keeping the lazy import preserves that exactly.
from eval_olmo2_probe2_downstream import (  # noqa: E402
    _safe_lp,
    encode_pair,
)
# reuse the SAME estimators the MMLU harness used (exact McNemar in log space +
# paired bootstrap), so paperC's MMLU and non-MMLU tables share statistics code.
from eval_olmo2_mmlu_content import (  # noqa: E402
    mcnemar_exact_p,
    paired_bootstrap_diff,
)

_LETTERS = "ABCDEFGHIJ"  # up to 10 options (CSQA is 5-way; ARC has 3/4/5-opt items;
#                          MMLU-Pro is up to 10-way -> this is exactly the ceiling)

# Expected cardinality per task, for the merge-time completeness assert. These are
# the counts the published ledger reports (see status/scout_21/lane2_a01_gate2.md
# section 2b/2c), i.e. AFTER dropping malformed items.
EXPECTED_N = {
    "arc_challenge": 1172,
    "arc_easy": 2376,
    "openbookqa": 500,
    "commonsense_qa": 1221,
    "piqa": 1838,
    "winogrande": 1267,
    # task #251 (POWER WALL). n=12032 -- the same ORDER OF MAGNITUDE as MMLU's
    # 14042, which is the whole point: the five #248 benchmarks are 6-28x smaller
    # than MMLU and 52/60 of #250's damaged cells were underpowered to have
    # detected MMLU's own -1.389 pp effect. No additional SMALL benchmark can fix
    # that; only n can.
    "mmlu_pro": 12032,
}

# Where MMLU-Pro's test parquet lives. Read DIRECTLY with pyarrow (not through
# datasets.load_dataset) so 8 concurrent shards cannot race on a HF cache builder
# and so the item order is a pure function of the file. First existing wins;
# MMLU_PRO_PARQUET overrides.
MMLU_PRO_PARQUET_CANDIDATES = [
    "../data/hf_datasets/TIGER-Lab___mmlu_pro/data/test-00000-of-00001.parquet",
    "/apdcephfs_wzc1/share_304376610/pighzliu_code/data/hf_datasets/"
    "TIGER-Lab___mmlu_pro/data/test-00000-of-00001.parquet",
    "/apdcephfs_zwfy6/share_304376610/pighzliu_code/data/hf_datasets/"
    "TIGER-Lab___mmlu_pro/data/test-00000-of-00001.parquet",
]

# Task-level flan-style descriptions, used ONLY with --desc_style flan. Prepended
# identically to BOTH interfaces, so the letter-vs-content contrast is unchanged.
FLAN_DESC = {
    "arc_challenge": "science",
    "arc_easy": "science",
    "openbookqa": "elementary science",
    "commonsense_qa": "common sense",
    "piqa": "physical common sense",
    "winogrande": "coreference resolution",
    "mmlu_pro": "many subjects",
}

ALL_TASKS = list(EXPECTED_N.keys())


# ---------------------------------------------------------------------------
# family dispatch for PRUNED checkpoints (paperC heal-confound arm)
# ---------------------------------------------------------------------------
# `load_pruned_model` (imported above) hardcodes Olmo2Config/Olmo2ForCausalLM via
# `build_pruned_shell`, so it can only rebuild an OLMo-2 pruned shell. paperC's
# healed Qwen3-8B-Base front8+fresh2 arm is a Qwen3 pruned checkpoint and must be
# scored by THIS harness at read-out.
#
# Design, and why this shape and not another:
#
# * ROUTE to the existing sibling, do NOT parameterise build_pruned_shell.
#   `scripts/eval_qwen3_probe2_ppl.py` already implements the identical builder
#   contract for Qwen3 -- same signature, same {'model_state': ...} ckpt format,
#   same ckpt-meta-vs-CLI agreement check, same fp32 + strict load, same returned
#   meta shape -- and it additionally does the one thing the OLMo builder must
#   NOT do: reset `cfg.layer_types` to length keep+fresh unconditionally (Qwen3's
#   layer_types is not recomputed from num_hidden_layers and would stay length-36
#   -> crash). Adding a family branch INSIDE the OLMo builder would duplicate
#   that Qwen3-specific quirk in a second place and put a new conditional on the
#   code path of every archived OLMo-2 number. Routing keeps the OLMo path
#   textually untouched.
#
# * DISPATCH ON THE CHECKPOINT, not on a CLI flag. Every trainer in this repo
#   writes `model_family` into the .pt (`train_olmo2_arch_probe2.py:538`,
#   `train_olmo2_shortgpt.py:271`, `train_qwen3_arch_probe2.py:352`), so the
#   family is a property of the artefact and cannot be mis-specified at the
#   command line. `--model_family` exists only as an override for a ckpt that
#   predates the field, and it is a HARD ERROR if it contradicts the ckpt.
#
# * The OLMo-2 default is bit-preserving. When the resolved family is olmo2 this
#   function calls exactly `load_pruned_model(...)` with exactly the same
#   arguments -- no re-ordering, no extra kwargs, no added logging on that path.
#
# Note what this is NOT protecting against: a family mismatch could never have
# corrupted a number silently. Measured, an OLMo-2 shell strict-loading a Qwen3
# state_dict raises immediately -- OLMo-2 layers have
# `post_feedforward_layernorm.weight` and no `input_layernorm.weight`, Qwen3 the
# reverse, and the vocabularies differ (100352 vs 151936). So today's behaviour
# is a crash on day 8, not a wrong number. This change converts that crash into
# a score.
_PRUNED_LOADERS = {"olmo2", "qwen3"}


def _load_pruned_dispatch(ckpt_path, base_path, keep_front_layers,
                          n_fresh_layers, device, model_family=None):
    """Load a pruned prune-then-heal ckpt of EITHER family.

    Returns exactly what `load_pruned_model` returns: (model, meta_dict).
    """
    # Read only the family tag; `torch.load` of the whole ckpt happens once,
    # inside the family loader, so this does not double the (11-38 GB) read.
    fam_ck = None
    try:
        _hdr = torch.load(ckpt_path, map_location="cpu", weights_only=False,
                          mmap=True)
        if isinstance(_hdr, dict):
            fam_ck = _hdr.get("model_family")
        del _hdr
    except Exception as e:  # noqa: BLE001 - mmap unsupported / odd ckpt: fall back
        _log(f"[dispatch] could not mmap-peek {ckpt_path} for model_family "
             f"({type(e).__name__}: {e}); relying on --model_family")

    if fam_ck is not None and model_family and str(fam_ck) != str(model_family):
        raise ValueError(
            f"--model_family={model_family} contradicts ckpt meta "
            f"model_family={fam_ck} in {ckpt_path}. The ckpt is authoritative; "
            f"drop the flag or fix it.")
    fam = str(fam_ck or model_family or "olmo2").lower()
    if fam not in _PRUNED_LOADERS:
        raise ValueError(
            f"pruned ckpt {ckpt_path} declares model_family={fam!r}, which has "
            f"no pruned-shell builder here (known: {sorted(_PRUNED_LOADERS)}). "
            f"Port the family's build_pruned_shell first -- do NOT score it "
            f"through another family's shell.")
    if fam_ck is None:
        _log(f"[dispatch] ckpt has no model_family field; using {fam!r} "
             f"({'--model_family' if model_family else 'default'})")

    if fam == "olmo2":
        # BIT-PRESERVING: identical call to the pre-change code path.
        return load_pruned_model(ckpt_path, base_path, keep_front_layers,
                                 n_fresh_layers, device)

    # Lazy import (see the note at the top of this file): Qwen3Config /
    # Qwen3ForCausalLM must never enter an OLMo-2 run's import graph.
    from eval_qwen3_probe2_ppl import (  # noqa: PLC0415
        load_pruned_model as load_pruned_model_qwen3,
    )
    _assert_not_instruct(base_path)
    model, meta = load_pruned_model_qwen3(ckpt_path, base_path,
                                          keep_front_layers, n_fresh_layers,
                                          device)
    # The Qwen3 sibling already sets model_family="qwen3"; assert rather than
    # assume, so a future edit there cannot silently produce an untagged cell.
    assert meta.get("model_family") == "qwen3", meta
    return model, meta


# Qwen3-8B-**Base** vs Qwen3-8B-**Instruct** are two directories that differ in
# ~50 bytes of tokenizer metadata and are trivially confusable:
#   models/Qwen3-8B-Base   eos 151643 <|endoftext|>  ctx 32768   <- BASE
#   models/Qwen--Qwen3-8b  eos 151645 <|im_end|>     ctx 40960   <- INSTRUCT
# Both have a chat_template and both share vocab.json/merges.txt byte-for-byte,
# so `model_type`, `architectures`, vocab_size and the encoding of ordinary text
# are ALL identical -- there is no signal in the obvious places. The criterion is
# eos_token_id + max_position_embeddings, verified on disk 2026-08-12.
#
# This matters because the whole project evaluates at chat_template=False on the
# stated grounds that the models have no SFT/RL (memory:
# paper-eval-chat-false-mandatory). An Instruct checkpoint HAS been SFT'd, so
# scoring it under that protocol and labelling it "base" is exactly the defect
# `status/ISSUES.jsonl :: paperB-crossfamily-qwen-instruct-mislabelled-as-base`
# records. Refuse by default; `ALLOW_INSTRUCT_BASE=1` opts in for someone who
# deliberately wants an Instruct arm and will label it as such.
_QWEN3_INSTRUCT_EOS = 151645
_QWEN3_BASE_EOS = 151643


def _assert_not_instruct(base_path):
    """Refuse a Qwen3-8B-Instruct dir where a Base model is expected."""
    cfg_path = os.path.join(base_path or "", "config.json")
    if not base_path or not os.path.exists(cfg_path):
        _log(f"[guard][WARN] cannot check base/instruct identity: no "
             f"config.json at {cfg_path!r} -- proceeding UNVERIFIED")
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    eos = cfg.get("eos_token_id")
    ctx = cfg.get("max_position_embeddings")
    if eos == _QWEN3_INSTRUCT_EOS:
        msg = (f"{base_path} is Qwen3-8B-**INSTRUCT** (eos_token_id={eos} "
               f"= <|im_end|>, max_position_embeddings={ctx}), not Base "
               f"(eos {_QWEN3_BASE_EOS} = <|endoftext|>, ctx 32768). This "
               f"harness scores at chat_template=False on the stated grounds "
               f"that the model has no SFT/RL, which is false for an Instruct "
               f"checkpoint -- and a cell produced from it must not be labelled "
               f"'base' (see status/ISSUES.jsonl :: "
               f"paperB-crossfamily-qwen-instruct-mislabelled-as-base). Use "
               f"models/Qwen3-8B-Base, or set ALLOW_INSTRUCT_BASE=1 to proceed "
               f"and label the resulting row as Instruct-derived.")
        if os.environ.get("ALLOW_INSTRUCT_BASE") == "1":
            _log(f"[guard][WARN] ALLOW_INSTRUCT_BASE=1 -> proceeding. {msg}")
            return
        raise ValueError(msg)
    if eos != _QWEN3_BASE_EOS:
        _log(f"[guard][WARN] {base_path} has unexpected eos_token_id={eos} "
             f"(expected {_QWEN3_BASE_EOS} for Qwen3-8B-Base); not the known "
             f"Instruct signature ({_QWEN3_INSTRUCT_EOS}) either -- proceeding")
        return
    _log(f"[guard] OK: {base_path} is Qwen3 **Base** (eos={eos}, ctx={ctx})")


# ---------------------------------------------------------------------------
# dataset loaders -> {gold, n_opt, letter_cands, content_cands}
# ---------------------------------------------------------------------------
def _mk_pair(stem: str, texts, gold: int, desc: str):
    """Build the letter/content candidate sets for one item from a shared
    question stem. `stem` is the question view BOTH interfaces see."""
    n = len(texts)
    body = "\n".join(f"{_LETTERS[i]}. {texts[i]}" for i in range(n))
    q_letter = desc + stem + "\n" + body + "\nAnswer:"
    q_content = desc + stem + "\nAnswer:"
    return {
        "gold": int(gold),
        "n_opt": n,
        "letter_cands": [(q_letter, " " + _LETTERS[i], len(_LETTERS[i]))
                         for i in range(n)],
        "content_cands": [(q_content, " " + str(texts[i]), len(str(texts[i])))
                          for i in range(n)],
    }


def _find_mmlu_pro_parquet():
    override = os.environ.get("MMLU_PRO_PARQUET", "")
    cands = ([override] if override else []) + MMLU_PRO_PARQUET_CANDIDATES
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "MMLU-Pro test parquet not found. Searched (BOTH disks are in the list; a "
        "file is 'missing' only after both have been checked): "
        + " | ".join(cands))


def load_mc_examples(task: str, desc_style: str = "none"):
    """Load ONE task. Returns list of per-item dicts with both candidate sets.

    The content prompt is `Question: <q>\\nAnswer:` for every task, which is
    byte-identical to the published loader for arc_challenge / arc_easy /
    commonsense_qa (free cross-validation of content_raw) and a deliberate
    deviation for openbookqa (published uses the bare stem, no Answer: cue).
    """
    if desc_style == "flan":
        desc = ("The following are multiple choice questions (with answers) "
                f"about {FLAN_DESC[task]}.\n\n")
    elif desc_style == "none":
        desc = ""
    else:
        raise ValueError(f"unknown desc_style {desc_style}")

    out = []

    if task == "mmlu_pro":
        # ---------------------------------------------------------------------
        # MMLU-Pro (TIGER-Lab/MMLU-Pro), task #251. Read the parquet DIRECTLY so
        # 8 shards cannot race on a HF builder and item order is file-determined.
        #
        # 10-WAY: `_LETTERS` covers A-J exactly. `n_opt` is NOT constant --
        # measured distribution over the 12032 test items is
        #   10: 9981, 9: 801, 8: 320, 7: 158, 6: 93, 5: 52, 4: 606, 3: 21
        # (mean 9.474). Every downstream null already handles per-item `n_opt`
        # (best_constant_letter uses gold_letter; the longest-option null uses
        # `range(r["n_opt"])`), and `chance` is therefore the MEAN of 1/n_opt
        # (0.110877), NOT 0.10. Both are reported in the verdict.
        #
        # NO `.strip()`, NO option filtering, NO 'N/A' pruning: verified on disk
        # that 0/12032 items contain an 'N/A' option and 0/12032 have
        # answer_index disagreeing with the `answer` letter, so the full test
        # split is used and `EXPECTED_N` is the raw row count.
        # ---------------------------------------------------------------------
        import pyarrow.parquet as pq
        path = _find_mmlu_pro_parquet()
        rows = pq.read_table(path, columns=["question_id", "question", "options",
                                            "answer", "answer_index",
                                            "category"]).to_pylist()
        for ex in rows:
            gi = int(ex["answer_index"])
            opts = [str(o) for o in ex["options"]]
            # hard consistency assert: the dataset ships BOTH a letter and an
            # index; if they ever disagree the gold is ambiguous and the cell is
            # not scoreable.
            assert 0 <= gi < len(opts), (ex["question_id"], gi, len(opts))
            assert _LETTERS[gi] == ex["answer"], (
                f"mmlu_pro qid={ex['question_id']}: answer_index {gi} "
                f"({_LETTERS[gi]}) != answer {ex['answer']}")
            assert len(opts) <= len(_LETTERS), (ex["question_id"], len(opts))
            out.append(_mk_pair("Question: " + ex["question"], opts, gi, desc))
        return out

    from datasets import load_dataset

    if task in ("arc_challenge", "arc_easy"):
        cfg = "ARC-Challenge" if task == "arc_challenge" else "ARC-Easy"
        d = load_dataset("allenai/ai2_arc", cfg, split="test")
        for ex in d:
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue  # malformed; skip (same rule as the published loader)
            # NOTE: no .strip() on the question -> content prompt is byte-identical
            # to eval_olmo2_probe2_downstream.py's arc branch.
            out.append(_mk_pair("Question: " + ex["question"],
                                ex["choices"]["text"], labels.index(ans), desc))
        return out

    if task == "openbookqa":
        d = load_dataset("allenai/openbookqa", "main", split="test")
        for ex in d:
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue
            out.append(_mk_pair("Question: " + ex["question_stem"],
                                ex["choices"]["text"], labels.index(ans), desc))
        return out

    if task == "commonsense_qa":
        d = load_dataset("tau/commonsense_qa", split="validation")
        for ex in d:
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue  # test split has no gold
            out.append(_mk_pair("Question: " + ex["question"],
                                ex["choices"]["text"], labels.index(ans), desc))
        return out

    if task == "piqa":
        d = load_dataset("ybisk/piqa", revision="refs/convert/parquet",
                         split="validation")
        for ex in d:
            out.append(_mk_pair("Question: " + ex["goal"],
                                [ex["sol1"], ex["sol2"]], int(ex["label"]), desc))
        return out

    if task == "winogrande":
        # NEGATIVE CONTROL. The published content interface for winogrande uses
        # PARTIAL scoring (both options share the continuation), which makes
        # length normalisation a no-op -> acc == acc_norm exactly, longest-option
        # null exactly 0.5 with a 100% tie rate. Here the content interface is
        # built the same label-free way as every other task (score the option
        # text as the continuation of the blanked sentence) so the letter/content
        # relation is uniform; the degeneracy that makes winogrande a control is
        # reported from the tie diagnostics, not assumed.
        d = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        a2i = {"1": 0, "2": 1}
        for ex in d:
            s = ex["sentence"]
            i = s.index("_")
            stem = s[:i].rstrip() + " _" + s[i + 1:]
            out.append(_mk_pair("Fill in the blank: " + stem,
                                [ex["option1"], ex["option2"]],
                                a2i[ex["answer"]], desc))
        return out

    raise ValueError(f"unknown task {task}")


# ---------------------------------------------------------------------------
# scoring: one batched teacher-forced pass over ALL candidates of BOTH interfaces
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_examples(model, tok, examples, task, device, batch_size, add_bos,
                   bos_id, pad_id, max_len, shard_index=0, num_shards=1,
                   lp_chunk=512, _raw_sink=None):
    """Score both interfaces for every example of ONE task. Returns
    (records, n_trunc). Numerics are identical to
    eval_olmo2_mmlu_content.score_examples (fp32 log_softmax of the fp32-cast
    logits, summed over teacher-forced continuation tokens).

    MEMORY (task #251). The original implementation did
    `log_softmax(out.logits.float(), dim=-1)` over the WHOLE [B, L, V] batch.
    With V = 100352 that is 4*B*L*V bytes of fp32 -- ~19-24 GiB at B=48 on
    MMLU-Pro, which OOMs an H20 (it survived the 4-way tasks only because those
    prompts are short). Since only the teacher-forced continuation positions are
    ever read, this version GATHERS those positions first and casts/normalises
    only them, in chunks of `lp_chunk` rows. `log_softmax` reduces over the
    vocab dim independently per position, so this is the SAME arithmetic on the
    same fp32 inputs -- asserted BIT-IDENTICAL against the old whole-tensor path
    by `--selftest` (`_assert_gather_path_bit_identical`) and, on GPU, by
    `--verify_numerics`. Peak fp32 buffer is now 4*lp_chunk*V (~205 MiB at 512)
    instead of 4*B*L*V.
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

    raw = {}
    ntok = {}
    order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
    for b in range(0, len(order), batch_size):
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
        # use_cache=False: this is a single teacher-forced forward, nothing is
        # ever generated, so the KV cache is written and immediately discarded.
        # It is pure waste -- and on a NON-GQA family it is the dominant
        # allocation and the direct cause of #251's llama2_7b_base OOM (5/8
        # shards died). fp32 KV at B=48, L=1536: Llama-2 has num_kv_heads=32
        # (no GQA) x 32 layers = 72.0 GiB, versus 18.0 GiB for Llama-3 and
        # 20.2 GiB for Qwen3 (both num_kv_heads=8). That is why only the
        # Llama-2 INTACT arm OOMed while every GQA arm and every truncated
        # (=fewer layers, hence smaller cache) Llama-2 rung survived. Dropping
        # the cache cannot change any logit: HF returns identical logits with
        # and without it for a full-sequence forward.
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn,
                            use_cache=False)
        else:
            out = model(input_ids=input_ids, attention_mask=attn,
                        use_cache=False)

        # flat index of every position that is actually read, and its target.
        # Positions of ONE candidate are appended CONTIGUOUSLY and in increasing
        # j, so the per-candidate sum below is a torch fp32 sum over the same
        # values in the same order as the old `logprobs[r, pos, tgt].sum()`.
        rows, poss, tgts, spans = [], [], [], []
        for r, i in enumerate(bidx):
            ei, proto, ci, ids, cs, cl = items[i]
            spans.append(((ei, proto, ci), len(rows), cl))
            for j in range(cl):
                rows.append(r)
                poss.append(cs - 1 + j)
                tgts.append(ids[cs + j])
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        poss_t = torch.tensor(poss, dtype=torch.long, device=device)
        tgts_t = torch.tensor(tgts, dtype=torch.long, device=device)
        vals = torch.empty(len(rows), dtype=torch.float32, device=device)
        for s in range(0, len(rows), lp_chunk):
            e = min(s + lp_chunk, len(rows))
            sel = out.logits[rows_t[s:e], poss_t[s:e]]          # [k, V]
            lp = torch.log_softmax(sel.float(), dim=-1)          # fp32, as before
            vals[s:e] = lp[torch.arange(e - s, device=device), tgts_t[s:e]]
            del sel, lp
        for key3, off, cl in spans:
            # fp32 torch sum over the contiguous continuation positions --
            # identical operation and order to the old whole-tensor path
            raw[key3] = vals[off:off + cl].sum().item()
            ntok[key3] = cl
        del out, vals, rows_t, poss_t, tgts_t

    # UNROUNDED scores, for the bit-identity selftest only. The per-item records
    # store `_safe_lp`-rounded values (6 dp, JSON hygiene), so comparing THOSE
    # against a raw reference measures the rounding, not the arithmetic.
    if _raw_sink is not None:
        _raw_sink.update(raw)

    records = []
    for ei, ex in enumerate(examples):
        n_opt = ex["n_opt"]
        gold = ex["gold"]
        item_id = shard_index + ei * num_shards
        L = [raw[(ei, "L", ci)] for ci in range(n_opt)]
        C = [raw[(ei, "C", ci)] for ci in range(n_opt)]
        Ct = [ntok[(ei, "C", ci)] for ci in range(n_opt)]
        Cn = [C[ci] / max(Ct[ci], 1) for ci in range(n_opt)]
        is_nan = any(not math.isfinite(x) for x in L + C)
        rec = {
            "item_id": item_id,
            "task": task,
            "gold": gold,
            "gold_letter": _LETTERS[gold],
            "n_opt": n_opt,
            "nan": bool(is_nan),
        }
        sc = lambda v: {_LETTERS[k]: _safe_lp(v[k]) for k in range(n_opt)}  # noqa: E731
        if is_nan:
            rec["letter"] = {"pred": None, "correct": False, "scores": sc(L)}
            rec["content_raw"] = {"pred": None, "correct": False, "scores": sc(C)}
            rec["content_norm"] = {
                "pred": None, "correct": False, "scores": sc(Cn),
                "cont_tokens": {_LETTERS[k]: Ct[k] for k in range(n_opt)}}
            records.append(rec)
            continue
        # argmax with ties broken by INDEX (what torch/py max() does; the same
        # operation every other harness in this repo uses for the readout).
        p_l = max(range(n_opt), key=lambda k: L[k])
        p_cr = max(range(n_opt), key=lambda k: C[k])
        p_cn = max(range(n_opt), key=lambda k: Cn[k])
        rec["letter"] = {"pred": p_l, "pred_letter": _LETTERS[p_l],
                         "correct": bool(p_l == gold), "scores": sc(L)}
        rec["content_raw"] = {"pred": p_cr, "pred_letter": _LETTERS[p_cr],
                              "correct": bool(p_cr == gold), "scores": sc(C)}
        rec["content_norm"] = {
            "pred": p_cn, "pred_letter": _LETTERS[p_cn],
            "correct": bool(p_cn == gold), "scores": sc(Cn),
            "cont_tokens": {_LETTERS[k]: Ct[k] for k in range(n_opt)}}
        records.append(rec)
    return records, n_trunc


# ---------------------------------------------------------------------------
# aggregate (accuracies + within-arm letter-vs-content pairing). Nulls live in
# the CPU analysis script, deliberately.
# ---------------------------------------------------------------------------
def aggregate(records, n_boot=10000, seed=0):
    valid = [r for r in records if not r["nan"]]
    n_all, n_valid = len(records), len(valid)
    Lc = np.array([1 if r["letter"]["correct"] else 0 for r in valid], dtype=np.int64)
    CR = np.array([1 if r["content_raw"]["correct"] else 0 for r in valid], dtype=np.int64)
    CN = np.array([1 if r["content_norm"]["correct"] else 0 for r in valid], dtype=np.int64)
    acc = lambda x: (float(x.mean()) if x.size else None)  # noqa: E731
    out = {
        "n": n_all, "n_valid": n_valid, "n_nan": n_all - n_valid,
        "letter_acc": acc(Lc),
        "content_raw_acc": acc(CR),
        "content_norm_acc": acc(CN),
        "modal_letter_share": None,
        "letter_tie_rate": None,
    }
    if n_valid:
        preds = [r["letter"]["pred_letter"] for r in valid]
        out["modal_letter_share"] = max(Counter(preds).values()) / n_valid
        out["letter_pred_hist"] = dict(sorted(Counter(preds).items()))
        # exact-tie rate in the letter readout (paperC's OLMo-2 bf16 tie mechanism)
        nties = 0
        for r in valid:
            v = [r["letter"]["scores"][_LETTERS[k]] for k in range(r["n_opt"])]
            m = max(v)
            if sum(1 for x in v if x == m) > 1:
                nties += 1
        out["letter_tie_rate"] = nties / n_valid
        b = int(np.sum((Lc == 1) & (CN == 0)))
        c = int(np.sum((Lc == 0) & (CN == 1)))
        diff, lo, hi = paired_bootstrap_diff(CN, Lc, n_boot=n_boot, seed=seed)
        out["letter_vs_content_norm"] = {
            "both_correct": int(np.sum((Lc == 1) & (CN == 1))),
            "letter_only_correct": b, "content_only_correct": c,
            "neither_correct": int(np.sum((Lc == 0) & (CN == 0))),
            "agreement": float(np.mean(Lc == CN)),
            "mcnemar_exact_p": mcnemar_exact_p(b, c),
            "bootstrap_diff_content_norm_minus_letter": diff,
            "bootstrap_ci95": [lo, hi], "n_boot": n_boot,
        }
    return out


# ---------------------------------------------------------------------------
# merge, WITH a hard shard-completeness assert (never merge a half set)
# ---------------------------------------------------------------------------
def merge_task(results_dir, task, num_shards, n_boot=10000, seed=0,
               expected_n=None, strict=True):
    pat = os.path.join(results_dir, f"per_example_{task}_shard*of{num_shards}.jsonl")
    files = sorted(glob.glob(pat))
    have = set()
    for f in files:
        base = os.path.basename(f)
        have.add(int(base.split("_shard")[1].split("of")[0]))
    missing = sorted(set(range(num_shards)) - have)
    if missing:
        msg = (f"SHARD INTEGRITY FAILURE {results_dir}/{task}: missing shards "
               f"{missing} of {num_shards} -> REFUSING to merge")
        if strict:
            raise AssertionError(msg)
        _log("[merge] " + msg)
        return None

    recs = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    ids = [r["item_id"] for r in recs]
    assert len(set(ids)) == len(ids), f"{task}: duplicate item_id after merge"
    recs.sort(key=lambda r: r["item_id"])

    exp = expected_n if expected_n is not None else EXPECTED_N.get(task)
    if exp is not None and len(recs) != exp:
        msg = (f"CARDINALITY FAILURE {results_dir}/{task}: n_scored={len(recs)} "
               f"!= expected {exp} -> REFUSING to merge")
        if strict:
            raise AssertionError(msg)
        _log("[merge] " + msg)
        return None

    agg = aggregate(recs, n_boot=n_boot, seed=seed)
    if strict:
        assert agg["n_nan"] == 0, (
            f"{results_dir}/{task}: n_nan={agg['n_nan']} != 0")

    meta = None
    for sf in sorted(glob.glob(os.path.join(results_dir, f"shard*of{num_shards}.json"))):
        with open(sf) as fh:
            meta = json.load(fh).get("meta", meta)

    out = {"output_name": os.path.basename(results_dir.rstrip("/")),
           "task": task, "protocol": "letter+content(dual)",
           "n_shards_found": len(files), "num_shards": num_shards,
           "expected_n": exp, "meta": meta, **agg}
    with open(os.path.join(results_dir, f"summary_{task}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(results_dir, f"per_example_{task}.jsonl"), "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    lv = agg.get("letter_vs_content_norm", {})
    _log(f"[merge] {out['output_name']}/{task}: n={agg['n_valid']} nan={agg['n_nan']} "
         f"| letter={agg['letter_acc']:.6f} craw={agg['content_raw_acc']:.6f} "
         f"cnorm={agg['content_norm_acc']:.6f} | modal={agg['modal_letter_share']:.4f} "
         f"tie={agg['letter_tie_rate']:.4f} McNemar_p={lv.get('mcnemar_exact_p', float('nan')):.3e}")
    return out


# ---------------------------------------------------------------------------
def _score_examples_wholetensor_REFERENCE(model, tok, examples, device,
                                          batch_size, add_bos, bos_id, pad_id,
                                          max_len):
    """The ORIGINAL whole-tensor scoring path, kept ONLY as the reference the
    memory-safe gather path is asserted bit-identical against (task #251).

    `log_softmax(out.logits.float(), dim=-1)` over the whole [B, L, V] batch is
    ~19-24 GiB of fp32 at B=48 on MMLU-Pro and OOMs an H20. It is NOT used for
    scoring any more. It exists so that the claim "the memory fix does not
    change a single number" is a TEST, not a comment: #248 and #250 numbers were
    produced by this path, so the new path must reproduce it exactly or the
    MMLU-Pro cells are not comparable with them.

    NOTE (#251 follow-up): this reference deliberately still calls the model
    WITHOUT `use_cache=False`, while the live path now passes it. That is not an
    oversight -- it turns `--selftest` into a direct test that dropping the
    (unused, immediately discarded) KV cache does not perturb a single
    log-prob."""
    items = []
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
                items.append((ei, proto, ci, ids, cs, cl))
    raw = {}
    order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
    for b in range(0, len(order), batch_size):
        bidx = order[b:b + batch_size]
        maxl = max(len(items[i][3]) for i in bidx)
        B = len(bidx)
        input_ids = torch.full((B, maxl), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxl), dtype=torch.long)
        for r, i in enumerate(bidx):
            ids = items[i][3]
            input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, :len(ids)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn)
        else:
            out = model(input_ids=input_ids, attention_mask=attn)
        logprobs = torch.log_softmax(out.logits.float(), dim=-1)
        for r, i in enumerate(bidx):
            ei, proto, ci, ids, cs, cl = items[i]
            end = cs + cl
            pos = torch.arange(cs - 1, end - 1, device=device)
            tgt = torch.tensor(ids[cs:end], dtype=torch.long, device=device)
            raw[(ei, proto, ci)] = logprobs[r, pos, tgt].sum().item()
        del out, logprobs
    return raw


def _assert_gather_path_bit_identical(model, tok, examples, device, bos_id,
                                     pad_id, max_len, batch_size=8):
    """The memory fix must not move a single bit, or the MMLU-Pro cells are not
    comparable with the #248/#250 cells that the old path produced.

    ⚠️ Compare the UNROUNDED scores (`_raw_sink`), not the per-item records: the
    records store `_safe_lp`-rounded values (6 dp), so a records-level comparison
    reports ~5e-7 "differences" that are pure JSON rounding and would mask or
    fake a real numerical change. That mistake was made and caught while writing
    this test."""
    ref = _score_examples_wholetensor_REFERENCE(
        model, tok, examples, device, batch_size, False, bos_id, pad_id, max_len)
    # exercise BOTH a chunk larger than the row count and one that forces
    # multiple chunks, so the chunk boundary logic is covered
    for chunk in (4096, 3):
        sink = {}
        score_examples(
            model, tok, examples, "selftest", device, batch_size, False,
            bos_id, pad_id, max_len, shard_index=0, num_shards=1,
            lp_chunk=chunk, _raw_sink=sink)
        assert set(sink) == set(ref), (len(sink), len(ref))
        worst = max(abs(sink[k] - ref[k]) for k in ref)
        n_exact = sum(1 for k in ref if sink[k] == ref[k])
        assert worst == 0.0 and n_exact == len(ref), (
            f"lp_chunk={chunk}: gather path is NOT bit-identical to the "
            f"whole-tensor path (max |delta| = {worst}, exact "
            f"{n_exact}/{len(ref)})")
        _log(f"[selftest] OK: lp_chunk={chunk} gather path BIT-IDENTICAL to the "
             f"whole-tensor reference on all {len(ref)} candidate scores "
             f"(unrounded)")


def _selftest():
    from transformers import Olmo2Config, Olmo2ForCausalLM, AutoTokenizer

    _log("[selftest] tiny CPU OLMo-2 + synthetic 4-way and 5-way items ...")
    base = os.environ.get("SELFTEST_BASE", "../models/OLMo-2-1124-7B")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    cfg = Olmo2Config.from_pretrained(base, local_files_only=True)
    cfg.hidden_size, cfg.intermediate_size = 64, 128
    cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads = 2, 4, 4
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    torch.manual_seed(0)
    model = Olmo2ForCausalLM(cfg).to(torch.float32).eval()
    device = torch.device("cpu")
    bos_id = tok.bos_token_id
    pad_id = tok.pad_token_id or tok.eos_token_id or 0

    seeds = [
        ("Question: What is the capital of France?",
         ["Paris", "London city", "Berlin", "Rome"], 0),
        ("Question: Two plus two equals?",
         ["three", "four", "five", "six", "seven"], 1),  # 5-way
        ("Question: Water is made of hydrogen and?",
         ["oxygen", "carbon", "nitrogen"], 0),           # 3-way
        # 10-way, exercising the FULL A-J letter range that MMLU-Pro needs
        # (task #251). If _LETTERS were ever shortened this item raises.
        ("Question: Which ordinal is the tenth?",
         ["first", "second", "third", "fourth", "fifth",
          "sixth", "seventh", "eighth", "ninth", "tenth"], 9),
    ]
    examples = [_mk_pair(s, t, g, "") for s, t, g in seeds]

    # --- structural check on the prompt relation: letter prompt == content prompt
    #     with the labelled body spliced in before "\nAnswer:" ---
    for ex, (stem, texts, _g) in zip(examples, seeds):
        q_l = ex["letter_cands"][0][0]
        q_c = ex["content_cands"][0][0]
        body = "\n".join(f"{_LETTERS[i]}. {texts[i]}" for i in range(len(texts)))
        assert q_c == stem + "\nAnswer:", q_c
        assert q_l == stem + "\n" + body + "\nAnswer:", q_l
        assert q_l.replace("\n" + body, "", 1) == q_c
        for i in range(len(texts)):
            assert ex["letter_cands"][i][1] == " " + _LETTERS[i]
            assert ex["content_cands"][i][1] == " " + texts[i]
    _log("[selftest] OK: letter prompt == content prompt + labelled body "
         "(MMLU-parallel interface relation), candidates letter vs text")

    recs, n_trunc = score_examples(
        model, tok, examples, "selftest", device, batch_size=8, add_bos=False,
        bos_id=bos_id, pad_id=pad_id, max_len=512, shard_index=0, num_shards=1)
    assert len(recs) == len(seeds)
    assert max(r["n_opt"] for r in recs) == 10, "10-way seed did not survive"
    for r, (_s, texts, _g) in zip(recs, seeds):
        n = len(texts)
        assert r["n_opt"] == n and r["nan"] is False
        for proto in ("letter", "content_raw", "content_norm"):
            assert r[proto]["pred"] in range(n)
            assert len(r[proto]["scores"]) == n
        for k in range(n):
            K = _LETTERS[k]
            rv = r["content_raw"]["scores"][K]
            nt = r["content_norm"]["cont_tokens"][K]
            nv = r["content_norm"]["scores"][K]
            assert abs(nv - rv / max(nt, 1)) < 1e-6, (K, rv, nt, nv)
    _log("[selftest] OK: 3/4/5/10-way schema + content_norm == raw / cont_tokens")

    # independent recompute of one letter candidate's sum-logprob
    ctx, cont, _ = examples[1]["letter_cands"][3]
    ids, cs, cl = encode_pair(tok, ctx, cont, False, bos_id)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))
    lp = torch.log_softmax(out.logits.float(), dim=-1)
    manual = float(lp[0, torch.arange(cs - 1, cs + cl - 1),
                      torch.tensor(ids[cs:cs + cl])].sum().item())
    assert abs(manual - recs[1]["letter"]["scores"]["D"]) < 1e-4, (
        manual, recs[1]["letter"]["scores"]["D"])
    _log(f"[selftest] OK: independent letter sum-logprob recompute matches "
         f"({manual:.5f})")

    # the task #251 memory fix must not move a single bit, or the MMLU-Pro cells
    # are not comparable with the #248/#250 cells the old path produced
    _assert_gather_path_bit_identical(model, tok, examples, device, bos_id,
                                     pad_id, 512)

    agg = aggregate(recs, n_boot=200, seed=0)
    assert agg["n_valid"] == len(seeds) and agg["n_nan"] == 0
    assert 0.0 <= agg["modal_letter_share"] <= 1.0
    assert 0.0 <= agg["letter_tie_rate"] <= 1.0
    lv = agg["letter_vs_content_norm"]
    assert (lv["both_correct"] + lv["letter_only_correct"] +
            lv["content_only_correct"] + lv["neither_correct"]) == len(seeds)
    _log("[selftest] OK: aggregate + pairing bookkeeping")

    # --- MMLU-Pro loader, IF the parquet is on this disk (task #251). This is
    #     the only dataset this harness reads outside `datasets`, so its schema
    #     asserts are worth exercising on CPU before a card is touched. ---
    try:
        p = _find_mmlu_pro_parquet()
    except FileNotFoundError as e:
        _log(f"[selftest] SKIP mmlu_pro loader ({e})")
    else:
        ex = load_mc_examples("mmlu_pro", "none")
        assert len(ex) == EXPECTED_N["mmlu_pro"], (len(ex), EXPECTED_N["mmlu_pro"])
        nopt = Counter(e["n_opt"] for e in ex)
        assert max(nopt) == 10 and min(nopt) >= 2, dict(nopt)
        for e in ex[:200] + ex[-200:]:
            n = e["n_opt"]
            assert 0 <= e["gold"] < n
            assert len(e["letter_cands"]) == n == len(e["content_cands"])
            q_l, q_c = e["letter_cands"][0][0], e["content_cands"][0][0]
            body = "\n".join(f"{_LETTERS[i]}. {e['content_cands'][i][1][1:]}"
                             for i in range(n))
            assert q_l == q_c.replace("\nAnswer:", "\n" + body + "\nAnswer:")
            for i in range(n):
                assert e["letter_cands"][i][1] == " " + _LETTERS[i]
        gm = Counter(_LETTERS[e["gold"]] for e in ex)
        best = max(gm.items(), key=lambda kv: kv[1])
        _log(f"[selftest] OK: mmlu_pro loader ({p}): n={len(ex)} "
             f"n_opt={dict(sorted(nopt.items()))} best-constant=always-{best[0]} "
             f"{best[1]/len(ex):.6f} (10-way chance 0.10, mean 1/n_opt "
             f"{sum(1.0/e['n_opt'] for e in ex)/len(ex):.6f}), letter prompt == "
             f"content prompt + labelled A-J body")

    # --- merge-time integrity asserts must FIRE on an incomplete set ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "per_example_arc_challenge_shard0of8.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        try:
            merge_task(td, "arc_challenge", 8, n_boot=10)
            raise AssertionError("merge accepted 1/8 shards -- assert did not fire")
        except AssertionError as e:
            assert "missing shards" in str(e), str(e)
    _log("[selftest] OK: merge REFUSES an incomplete shard set (1/8)")
    with tempfile.TemporaryDirectory() as td:
        for g in range(8):
            with open(os.path.join(td, f"per_example_arc_challenge_shard{g}of8.jsonl"), "w") as f:
                for r in recs:
                    rr = dict(r)
                    rr["item_id"] = g + r["item_id"] * 8
                    f.write(json.dumps(rr) + "\n")
        try:
            merge_task(td, "arc_challenge", 8, n_boot=10)
            raise AssertionError("merge accepted 24 != 1172 items")
        except AssertionError as e:
            assert "CARDINALITY FAILURE" in str(e), str(e)
    _log("[selftest] OK: merge REFUSES a wrong-cardinality set (24 != 1172)")
    _selftest_family_dispatch()
    _log("[selftest] ALL CHECKS PASSED")


def _selftest_family_dispatch():
    """CPU-only proof that the pruned-ckpt family dispatch routes on the CKPT and
    that the Base-vs-Instruct guard fires. Builds TINY real Olmo2/Qwen3 shells,
    saves them in the trainers' `{'model_state': ...}` format, and round-trips
    them through `_load_pruned_dispatch` -- so this exercises the actual
    `build_pruned_shell` of each family, not a mock.

    Cheap on purpose: 2-layer/32-hidden models, so the whole check is <1 s and
    needs no weights, no GPU and no dataset."""
    import tempfile
    from transformers import (Olmo2Config, Olmo2ForCausalLM,
                              Qwen3Config, Qwen3ForCausalLM)

    olmo_base = os.environ.get("SELFTEST_BASE", "../models/OLMo-2-1124-7B")
    qwen_base = os.environ.get("SELFTEST_QWEN3_BASE", "../models/Qwen3-8B-Base")
    dev = torch.device("cpu")

    def _tiny(cfg):
        cfg.hidden_size, cfg.intermediate_size = 32, 64
        cfg.num_attention_heads, cfg.num_key_value_heads = 2, 2
        cfg.num_hidden_layers = 2          # keep_front 1 + n_fresh 1
        if hasattr(cfg, "head_dim"):
            cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
        if getattr(cfg, "layer_types", None) is not None:
            cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
        return cfg

    with tempfile.TemporaryDirectory() as td:
        # ---- write a tiny shrunken copy of each base config -----------------
        specs = {}
        for fam, cfgcls, mcls, real_base in (
                ("olmo2", Olmo2Config, Olmo2ForCausalLM, olmo_base),
                ("qwen3", Qwen3Config, Qwen3ForCausalLM, qwen_base)):
            if not os.path.exists(os.path.join(real_base, "config.json")):
                _log(f"[selftest] SKIP dispatch/{fam}: no config.json at "
                     f"{real_base}")
                continue
            cfg = _tiny(cfgcls.from_pretrained(real_base, local_files_only=True))
            bdir = os.path.join(td, f"base_{fam}")
            os.makedirs(bdir, exist_ok=True)
            cfg.save_pretrained(bdir)
            torch.manual_seed(0)
            m = mcls(cfg).to(torch.float32)
            ck = os.path.join(td, f"{fam}.pt")
            torch.save({"model_state": m.state_dict(), "step": 7,
                        "model_family": fam, "keep_front_layers": 1,
                        "n_fresh_layers": 1}, ck)
            specs[fam] = (ck, bdir, m.state_dict())

        # ---- routes on the ckpt's model_family, and the weights survive -----
        for fam, (ck, bdir, sd) in specs.items():
            model, meta = _load_pruned_dispatch(ck, bdir, None, None, dev)
            got = type(model).__name__
            want = {"olmo2": "Olmo2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}[fam]
            assert got == want, f"{fam}: dispatch built {got}, expected {want}"
            assert meta["num_hidden_layers"] == 2, meta
            assert meta["ckpt_step"] == 7, meta
            new = model.state_dict()
            assert set(new) == set(sd), f"{fam}: key set changed on round-trip"
            worst = max(float((new[k].float() - sd[k].float()).abs().max())
                        for k in sd)
            assert worst == 0.0, f"{fam}: round-trip not exact (max|d|={worst})"
            _log(f"[selftest] OK: dispatch/{fam} -> {want}, "
                 f"state_dict round-trip EXACT ({len(sd)} tensors)")

        # ---- a contradicting --model_family is a hard error ------------------
        if "qwen3" in specs:
            ck, bdir, _ = specs["qwen3"]
            try:
                _load_pruned_dispatch(ck, bdir, None, None, dev,
                                      model_family="olmo2")
                raise AssertionError("dispatch accepted --model_family=olmo2 "
                                     "for a qwen3 ckpt")
            except ValueError as e:
                assert "contradicts ckpt meta" in str(e), str(e)
            _log("[selftest] OK: dispatch REFUSES --model_family contradicting "
                 "the ckpt")

        # ---- an untagged ckpt defaults to olmo2 (pre-change behaviour) -------
        if "olmo2" in specs:
            ck, bdir, _ = specs["olmo2"]
            d = torch.load(ck, map_location="cpu", weights_only=False)
            d.pop("model_family")
            ck2 = os.path.join(td, "untagged.pt")
            torch.save(d, ck2)
            model, meta = _load_pruned_dispatch(ck2, bdir, None, None, dev)
            assert type(model).__name__ == "Olmo2ForCausalLM"
            _log("[selftest] OK: ckpt with NO model_family field still loads as "
                 "olmo2 (pre-change default preserved)")

        # ---- the Base-vs-Instruct guard, on the REAL config.json files -------
        # eos 151643/151645 is the only reliable discriminator; assert both
        # directions so a future config edit cannot quietly disarm the guard.
        inst = os.environ.get("SELFTEST_QWEN3_INSTRUCT", "../models/Qwen--Qwen3-8b")
        if os.path.exists(os.path.join(qwen_base, "config.json")):
            _assert_not_instruct(qwen_base)   # must NOT raise
            _log("[selftest] OK: guard ACCEPTS Qwen3-8B-Base")
        if os.path.exists(os.path.join(inst, "config.json")):
            try:
                _assert_not_instruct(inst)
                raise AssertionError(f"guard accepted Instruct dir {inst}")
            except ValueError as e:
                assert "INSTRUCT" in str(e), str(e)
            _log(f"[selftest] OK: guard REFUSES Instruct dir ({inst})")
            os.environ["ALLOW_INSTRUCT_BASE"] = "1"
            try:
                _assert_not_instruct(inst)   # opt-in must NOT raise
            finally:
                del os.environ["ALLOW_INSTRUCT_BASE"]
            _log("[selftest] OK: ALLOW_INSTRUCT_BASE=1 opts in explicitly")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--any_family", action="store_true")
    p.add_argument("--model_family", type=str, default="",
                   choices=["", "olmo2", "qwen3"],
                   help="OVERRIDE the pruned-ckpt family for a ckpt written "
                        "before trainers recorded `model_family`. Normally "
                        "unnecessary and better left unset: the family is read "
                        "FROM THE CKPT, and if this flag contradicts the ckpt "
                        "it is a hard error. Ignored unless --ckpt is given.")
    p.add_argument("--keep_front_layers", type=int, default=None)
    p.add_argument("--n_fresh_layers", type=int, default=None)
    p.add_argument("--keep_indices", type=str, default="")
    p.add_argument("--tasks", type=str, default="arc_challenge",
                   help="comma-separated; one model load covers all of them")
    p.add_argument("--desc_style", type=str, default="none", choices=["none", "flan"])
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lp_chunk", type=int, default=512,
                   help="rows of [k, vocab] fp32 log_softmax per chunk. Peak "
                        "fp32 buffer is 4*lp_chunk*vocab (~205 MiB at 512 for "
                        "OLMo-2's 100352 vocab). Bit-identical for any value; "
                        "only memory changes. See score_examples' docstring.")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--allow_truncation", action="store_true",
                   help="Opt IN to left-truncating prompts longer than "
                        "--max_len. OFF BY DEFAULT AND YOU SHOULD LEAVE IT OFF. "
                        "Left-truncation eats the labelled `A. ...`/`B. ...` "
                        "option body that the letter interface is defined to "
                        "read, so a truncated cell measures a DIFFERENT "
                        "interface -- and because the overflow set is a "
                        "(tokenizer x prompt length) property it differs per "
                        "family, which silently breaks item-matching across "
                        "families. #251 shipped 10 cells with n_trunc>0 "
                        "because this was only a driver-level WARNING; it is "
                        "now a hard per-shard assert.")
    p.add_argument("--add_bos", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output_name", type=str, default="")
    p.add_argument("--results_root", type=str, default="olmo2_mc_letter_content_results")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--prepare_data", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"unknown task {t} (known: {ALL_TASKS})")

    if args.prepare_data:
        for t in tasks:
            ex = load_mc_examples(t, args.desc_style)
            exp = EXPECTED_N.get(t)
            _log(f"[prepare] {t}: {len(ex)} examples"
                 + (f" (expected {exp})" if exp else ""))
            assert exp is None or len(ex) == exp, f"{t}: {len(ex)} != {exp}"
        return

    if args.merge:
        if not args.output_name:
            raise ValueError("--merge requires --output_name")
        rd = os.path.join(args.results_root, args.output_name)
        for t in tasks:
            merge_task(rd, t, args.num_shards, n_boot=args.n_boot,
                       seed=args.boot_seed)
        return

    if not args.output_name:
        raise ValueError("--output_name required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (use --selftest for a CPU dry run)")
    device = torch.device("cuda")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    bos_id = tok.bos_token_id
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    if args.ckpt:
        model, meta = _load_pruned_dispatch(args.ckpt, args.base_model,
                                            args.keep_front_layers,
                                            args.n_fresh_layers, device,
                                            model_family=args.model_family)
    elif args.any_family:
        if args.keep_front_layers:
            model, meta = load_truncated_any_family(
                args.base_model, args.keep_front_layers, device)
        else:
            model, meta = load_base_model_any_family(args.base_model, device)
    else:
        model, meta = load_base_model(args.base_model, device)
    meta["base_model"] = args.base_model
    meta["add_bos"] = bool(args.add_bos)
    meta["chat_template"] = False  # paperC-wide: every arm here is a BASE LM
    #                                (no SFT/RL) -- OLMo-2, and Qwen3-8B-Base
    #                                for the healed cross-family arm, whose
    #                                base/Instruct identity is asserted by
    #                                _assert_not_instruct before the load.
    meta["desc_style"] = args.desc_style
    if args.keep_indices:
        meta["keep_indices"] = args.keep_indices

    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)
    shard_report = {}

    for task in tasks:
        examples = load_mc_examples(task, args.desc_style)
        exp = EXPECTED_N.get(task)
        assert exp is None or len(examples) == exp, \
            f"{task}: loaded {len(examples)} != expected {exp}"
        examples = examples[args.shard_index::args.num_shards]
        if args.limit and args.limit > 0:
            examples = examples[: args.limit]
        t0 = time.time()
        records, n_trunc = score_examples(
            model, tok, examples, task, device, args.batch_size,
            bool(args.add_bos), bos_id, pad_id, args.max_len,
            shard_index=args.shard_index, num_shards=args.num_shards,
            lp_chunk=args.lp_chunk)
        dt = time.time() - t0
        # HARD ASSERT, not a warning. #251's driver only printed a TRUNCATION
        # WARNING, so 10 of 15 cross-family cells shipped with n_trunc>0 (40 on
        # llama2_7b, 20 on qwen3_8b_base, identical on every rung) and the run
        # still wrote its summaries. Fail the shard instead: an arm that cannot
        # be scored at full prompt length must not produce a summary at all.
        if n_trunc and not args.allow_truncation:
            raise AssertionError(
                f"TRUNCATION FAILURE {args.output_name}/{task} "
                f"shard{args.shard_index}of{args.num_shards}: n_trunc={n_trunc} "
                f"at --max_len {args.max_len}. Left-truncation removes part of "
                f"the labelled option body, so this cell would measure a "
                f"different letter interface than the untruncated cells and the "
                f"overflow set is tokenizer-specific (item-matching across "
                f"families breaks). Raise --max_len until n_trunc == 0 (probe it "
                f"on CPU with paperC/code/mmlu_pro_trunc_audit.py), or pass "
                f"--allow_truncation if you deliberately want the old behaviour.")
        agg = aggregate(records, n_boot=200, seed=args.boot_seed)
        pe = os.path.join(
            results_dir,
            f"per_example_{task}_shard{args.shard_index}of{args.num_shards}.jsonl")
        with open(pe, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        shard_report[task] = {"n": len(records), "n_valid": agg["n_valid"],
                              "n_nan": agg["n_nan"], "n_trunc": n_trunc,
                              "seconds": round(dt, 1),
                              "letter_acc": agg["letter_acc"],
                              "content_raw_acc": agg["content_raw_acc"],
                              "content_norm_acc": agg["content_norm_acc"]}
        _log(f"[shard {args.shard_index}/{args.num_shards}] {task}: "
             f"n={len(records)} valid={agg['n_valid']} nan={agg['n_nan']} "
             f"letter={agg['letter_acc']:.4f} craw={agg['content_raw_acc']:.4f} "
             f"cnorm={agg['content_norm_acc']:.4f} trunc={n_trunc} ({dt:.1f}s)")

    with open(os.path.join(results_dir,
                           f"shard{args.shard_index}of{args.num_shards}.json"), "w") as f:
        json.dump({"shard_index": args.shard_index,
                   "num_shards": args.num_shards,
                   "tasks": tasks, "meta": meta,
                   "per_task": shard_report}, f, indent=2)


if __name__ == "__main__":
    main()
