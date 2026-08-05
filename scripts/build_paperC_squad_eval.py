#!/usr/bin/env python3
"""Rebuild the Paper C SQuAD eval/train sets with a CONTROLLED refusal prior.

WHY THIS EXISTS (task P0-4, 2026-08-05)
=======================================
The legacy `data/squad_{train,val}.jsonl` (built ~2026-04-15, generator script
lost) are unusable as a capability benchmark for two independently fatal reasons,
both re-verified from the raw SQuAD releases by `report_constant_baseline.py` and
the forensics in this file's docstring:

  (1) CONSTANT-REFUSAL CONTAMINATION.  997/2000 = 49.85% of val `target_text`
      values are the single Chinese string "根据提供的信息无法回答这个问题",
      versus 1756/10000 = 17.56% in train -> a 32.29pp train/val skew on the
      dominant label.  A constant function that ignores its input scores
      EM = 49.85, which is ABOVE every measured arm (A4_hero .2930 /
      A3_fromscratch .2605 / BASE_ref .3385; only A2_lora .6590 beats it).

  (2) "UNANSWERABLE" WAS A RETRIEVAL ARTEFACT, NOT A DESIGN.  The legacy rows'
      `relevant_indices` is statistically indistinguishable from
      Uniform(0, len(memory_texts)-1): on the 978 answerable val rows whose gold
      string occurs in some chunk, the pointer lands on a gold-bearing chunk 323
      times while a uniform pointer would land 312.5 +- 12.9 times (z = +0.82,
      cannot reject uniform).  So `relevant_indices` carries ZERO retrieval
      signal.  Meanwhile the gold answer IS present in *some* chunk for 97.5% of
      answerable val rows -- i.e. the passage was never actually pruned, the
      pointer was simply random.

Provenance recovered by matching questions against the raw releases:
  * legacy val   = SQuAD **v2.0 dev**; the refusal label is exactly the v2.0
    `is_impossible` flag (agreement 1998/2000 = 99.90%), and v2.0 dev is itself
    50.07% impossible -- which is where the 49.85% came from.
  * legacy train = SQuAD **v1.1/v2.0 train** (0 overlap with either dev split;
    0 question overlap with legacy val, so no leakage).
  * chunking = `context.split('.')` with empty pieces dropped, then SHUFFLED
    (89.8% of val rows' `memory_texts` are an exact set-permutation of that
    split; only 13.3% preserve the original order).

THE BUG: legacy refusal rows keep the WHOLE paragraph in `memory_texts`
(997/997 chunk sets are a subset of the question's own v2.0 paragraph, 895/997
an exact permutation of it).  For an is_impossible question that is fine in
v2.0's own framing, but combined with (2) it means the dataset never once
*removed* evidence -- "unanswerable" was inherited as a v2.0 label while the
answerable rows' `relevant_indices` was randomised.  A model is therefore graded
on reproducing a label distribution, not on reading.

WHAT THIS SCRIPT DOES DIFFERENTLY
=================================
  * `--refusal_rate` is an EXPLICIT knob, and TRAIN AND VAL USE THE SAME VALUE
    (kills the 32.29pp skew by construction).
  * Refusals are made TRULY unanswerable by DELETING every chunk that contains
    the gold answer (plus, when a v2.0 plausible-answer span is available, the
    chunk holding it), then asserting the gold no longer appears anywhere in
    `memory_texts`.  This is evidence removal, not label inheritance.
  * Answerable rows are asserted to be TRULY answerable: the gold appears in
    `memory_texts[relevant_indices]`, and `relevant_indices` is the exact
    post-shuffle set of gold-bearing chunk positions (a real retrieval target,
    not a random draw).
  * Chunking uses a sentence splitter that does not shred decimals/abbreviations
    (the legacy `split('.')` cut "U.S." and "3.5" apart).
  * Output schema is byte-compatible with the legacy jsonl, so
    `scripts/eval_paperC_squad_emf1.py` and `scripts/tokenize_squad_olmo2_sft.py`
    run unmodified.

OUTPUTS (never touches data/squad_{train,val}.jsonl -- those are the provenance
of already-reported numbers and are read-only):

    data/paperC_squad_v2/{train,val}_refusal{00,25,50}.jsonl
    data/paperC_squad_v2/manifest.json

USAGE
=====
    # all three refusal_rate tiers (default)
    python scripts/build_paperC_squad_eval.py

    # one tier
    python scripts/build_paperC_squad_eval.py --refusal_rate 0.25

Raw SQuAD is read from --raw_dir (default data/squad_raw); missing files are
downloaded via the hy-proxy if --download is set.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from collections import Counter

RAW_URLS = {
    "train-v1.1.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev-v1.1.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    "train-v2.0.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "dev-v2.0.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

REFUSAL_TEXT = "根据提供的信息无法回答这个问题"
CN_PREFIX = "根据以下对话记录，回答问题："

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)

# Sentence splitter: split after . ! ? + whitespace + an opening-looking token,
# but NOT when the period belongs to a short capitalised abbreviation
# (U.S. / Dr. / No. / cf.).  A decimal ("3.14") can never match because the
# regex requires whitespace after the period.  The legacy builder used a bare
# str.split('.'), which shredded all of these.
# NOTE the lookbehinds must include the period itself, since the match position
# is *after* it.
_ABBREV_WORDS = ["Dr", "Mr", "Mrs", "Ms", "St", "No", "vs", "Jr", "Sr", "cf",
                 "al", "Prof", "Gen", "Col", "Sgt", "Lt", "Rev", "Hon", "Rep",
                 "Sen", "Gov", "Capt", "Ft", "Mt", "Op", "Ave", "Inc", "Ltd",
                 "Co", "Corp", "etc", "esp", "approx", "Fig", "Vol", "pp"]
_ABBREV = (r"(?<!\b[A-Z]\.)"                      # single initial: "J."
           + "".join(rf"(?<!\b{w}\.)" for w in _ABBREV_WORDS))
_SENT_SPLIT = re.compile(r"(?<=[.!?])" + _ABBREV + r"\s+(?=[A-Z0-9\"'(\[])")


# ---------------------------------------------------------------------------
# normalisation -- MUST match scripts/eval_olmo2_closedbook_qa.normalize_answer
# so that the answerable/unanswerable asserts use the same string identity the
# EM/F1 scorer will use.
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def gold_in(chunk: str, gold_norm: str) -> bool:
    """True iff the normalised gold string occurs inside the normalised chunk."""
    if not gold_norm:
        return False
    return gold_norm in normalize_answer(chunk)


# ---------------------------------------------------------------------------
# raw SQuAD loading
# ---------------------------------------------------------------------------
def _download(raw_dir: str, fname: str, tries: int = 200) -> None:
    """Resume-download one raw file until json.load succeeds (the proxy truncates)."""
    import subprocess

    path = os.path.join(raw_dir, fname)
    env = dict(os.environ)
    env.setdefault("http_proxy", "http://hy-proxy.woa.com:3128")
    env.setdefault("https_proxy", "http://hy-proxy.woa.com:3128")
    for i in range(tries):
        if _json_ok(path):
            return
        subprocess.run(["curl", "-sS", "-C", "-", "--max-time", "90", "-o", path, RAW_URLS[fname]],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _json_ok(path):
        raise RuntimeError(f"could not fetch a complete {fname} after {tries} resume attempts")


def _json_ok(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except Exception:
        return False


def load_raw(raw_dir: str, fname: str, download: bool):
    path = os.path.join(raw_dir, fname)
    if not _json_ok(path):
        if not download:
            raise FileNotFoundError(
                f"{path} missing/truncated. Re-run with --download (uses hy-proxy), or fetch "
                f"{RAW_URLS[fname]} manually.")
        os.makedirs(raw_dir, exist_ok=True)
        _download(raw_dir, fname)
    with open(path) as f:
        return json.load(f)


def iter_qas(raw):
    """-> (article_title, paragraph_context, qa_dict) for every question."""
    for art in raw["data"]:
        title = art.get("title", "")
        for par in art["paragraphs"]:
            ctx = par["context"]
            for qa in par["qas"]:
                yield title, ctx, qa


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------
def chunk_paragraph(context: str) -> list[str]:
    """Split a SQuAD paragraph into sentence chunks (order preserved, deduped).

    Dedup matters: a handful of SQuAD paragraphs repeat a sentence verbatim, and
    a duplicated chunk would make `relevant_indices` ambiguous.
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(context.strip())]
    out, seen = [], set()
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# example construction
# ---------------------------------------------------------------------------
def build_answerable(qid, title, context, qa, distractors, n_chunks, rng):
    """A row whose gold IS present in memory_texts[relevant_indices].

    Every row is padded/truncated to exactly `n_chunks` chunks so that context
    length carries NO signal about answerability (see build_refusal).  Padding
    chunks come from a DIFFERENT article and are filtered so they cannot contain
    the gold.

    Returns None if the paragraph cannot be split so that some chunk contains
    the gold (e.g. the answer straddles a sentence boundary), or if the gold
    spans more than `n_chunks` chunks.
    """
    answers = qa.get("answers") or []
    if not answers:
        return None
    gold = answers[0]["text"].strip()
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return None
    chunks = chunk_paragraph(context)
    hits = [c for c in chunks if gold_in(c, gold_norm)]
    rest = [c for c in chunks if not gold_in(c, gold_norm)]
    if not hits or len(hits) > n_chunks:
        return None

    keep = list(hits)
    rng.shuffle(rest)
    keep += rest[: max(0, n_chunks - len(keep))]
    keep = _pad_with_distractors(keep, distractors, [gold_norm], title, n_chunks, rng)
    if keep is None:
        return None

    rng.shuffle(keep)
    rel = sorted(i for i, c in enumerate(keep) if gold_in(c, gold_norm))
    # the gold set must be exactly the sentences we inherited from the paragraph
    if len(rel) != len(hits):
        return None

    return {
        "input_text": CN_PREFIX + qa["question"].strip(),
        "target_text": gold,
        "memory_texts": keep,
        "relevant_indices": rel,
        "dialogue_id": qid,
        "session_idx": 0,
        "recall_required": True,
        # provenance / stratification fields (extra keys are ignored by the
        # existing loaders, which read only the 7 canonical fields)
        "_answerable": True,
        "_squad_id": qa.get("id", ""),
        "_title": title,
        "_all_golds": sorted({a["text"].strip() for a in answers}),
    }


def _pad_with_distractors(keep, distractors, golds_norm, title, n_chunks, rng):
    """Top `keep` up to n_chunks with chunks from OTHER articles that contain no gold."""
    if len(keep) >= n_chunks:
        return keep[:n_chunks]
    have = set(keep)
    out = list(keep)
    for _ in range(60 * n_chunks):
        if len(out) >= n_chunks:
            break
        dt, dc = distractors[rng.randrange(len(distractors))]
        if dt == title or dc in have:
            continue
        if any(gold_in(dc, g) for g in golds_norm):
            continue
        out.append(dc)
        have.add(dc)
    return out if len(out) == n_chunks else None


def build_refusal(qid, title, context, qa, gold_pool, distractors, n_chunks, rng):
    """A row that is TRULY unanswerable: every gold-bearing chunk is DELETED.

    `gold_pool` = the answer strings to erase (the paragraph's real answers plus
    the v2.0 plausible_answers span, so the distractor context cannot be mistaken
    for evidence).  The deleted sentences are replaced by chunks from a DIFFERENT
    article, restoring the chunk count to exactly `n_chunks` -- otherwise
    "context is short" would itself predict the refusal label (measured: 393 vs
    788 chars before this fix), reproducing the very shortcut we are removing.

    Returns None if nothing had to be erased (then it is not a *constructed*
    refusal), or if no own-paragraph sentence survives (then the row has no
    topical anchor and abstaining is trivial).
    """
    chunks = chunk_paragraph(context)
    golds_norm = [normalize_answer(g) for g in gold_pool]
    golds_norm = sorted({g for g in golds_norm if g})
    if not golds_norm:
        return None

    kept = [c for c in chunks if not any(gold_in(c, g) for g in golds_norm)]
    n_removed = len(chunks) - len(kept)
    if n_removed == 0:
        return None          # nothing to delete -> not a constructed refusal
    if not kept:
        return None          # no on-topic sentence survives -> trivially abstainable
    rng.shuffle(kept)
    kept = kept[:n_chunks]
    kept = _pad_with_distractors(kept, distractors, golds_norm, title, n_chunks, rng)
    if kept is None:
        return None

    rng.shuffle(kept)

    # HARD ASSERT: the gold must be gone.
    for g in golds_norm:
        for c in kept:
            assert not gold_in(c, g), f"refusal row {qid} still contains gold {g!r}"

    return {
        "input_text": CN_PREFIX + qa["question"].strip(),
        "target_text": REFUSAL_TEXT,
        "memory_texts": kept,
        "relevant_indices": [],
        "dialogue_id": qid,
        "session_idx": 0,
        "recall_required": True,
        "_answerable": False,
        "_squad_id": qa.get("id", ""),
        "_title": title,
        "_n_chunks_deleted": n_removed,
        "_erased_golds": golds_norm,
    }


def collect_pools(raw_v1, raw_v2, split_name):
    """-> (answerable_candidates, refusal_candidates, distractor_chunks)

    answerable = v1.1 questions (always have a real answer span).
    refusal    = v2.0 is_impossible questions, from which we erase the
                 plausible_answers span AND every answer of the *other*
                 (answerable) questions on the same paragraph -- so the deleted
                 chunks are exactly the ones a reader could mine for an answer.
    distractor = (article_title, sentence) over BOTH releases; used to pad every
                 row to a fixed chunk count so length cannot leak the label.
    """
    answerable = []
    for title, ctx, qa in iter_qas(raw_v1):
        if qa.get("answers"):
            answerable.append((title, ctx, qa))

    # map paragraph -> all real answer strings on it (from v2.0 answerable qas)
    par_answers: dict[str, set[str]] = {}
    for _t, ctx, qa in iter_qas(raw_v2):
        if not qa.get("is_impossible"):
            for a in qa.get("answers") or []:
                par_answers.setdefault(ctx, set()).add(a["text"].strip())

    refusal = []
    for title, ctx, qa in iter_qas(raw_v2):
        if not qa.get("is_impossible"):
            continue
        pool = set(par_answers.get(ctx, set()))
        for a in qa.get("plausible_answers") or []:
            pool.add(a["text"].strip())
        if not pool:
            continue
        refusal.append((title, ctx, qa, sorted(pool)))

    seen_par = set()
    distractors = []
    for raw in (raw_v1, raw_v2):
        for art in raw["data"]:
            title = art.get("title", "")
            for par in art["paragraphs"]:
                ctx = par["context"]
                if ctx in seen_par:
                    continue
                seen_par.add(ctx)
                for c in chunk_paragraph(ctx):
                    if 40 <= len(c) <= 400:
                        distractors.append((title, c))
    return answerable, refusal, distractors


# ---------------------------------------------------------------------------
# one (split, refusal_rate) build
# ---------------------------------------------------------------------------
def build_split(split_name, n_total, refusal_rate, answerable_pool, refusal_pool,
                distractors, n_chunks, seed,
                forbid_questions=None, forbid_titles=None):
    rng = random.Random(seed)
    n_ref = int(round(n_total * refusal_rate))
    n_ans = n_total - n_ref
    forbid_questions = forbid_questions or set()
    forbid_titles = forbid_titles or set()

    ans_idx = list(range(len(answerable_pool)))
    ref_idx = list(range(len(refusal_pool)))
    rng.shuffle(ans_idx)
    rng.shuffle(ref_idx)

    rows = []
    seen_q = set()
    skipped = Counter()

    k = 0
    for j in ans_idx:
        if k >= n_ans:
            break
        title, ctx, qa = answerable_pool[j]
        q = qa["question"].strip()
        if q in seen_q or q in forbid_questions:
            skipped["dup_or_forbidden_question"] += 1
            continue
        if title in forbid_titles:
            skipped["forbidden_title"] += 1
            continue
        row = build_answerable(f"squad_{split_name}_ans_{k}", title, ctx, qa,
                               distractors, n_chunks, rng)
        if row is None:
            skipped["answerable_unbuildable"] += 1
            continue
        rows.append(row)
        seen_q.add(q)
        k += 1
    n_ans_built = k

    k = 0
    for j in ref_idx:
        if k >= n_ref:
            break
        title, ctx, qa, pool = refusal_pool[j]
        q = qa["question"].strip()
        if q in seen_q or q in forbid_questions:
            skipped["dup_or_forbidden_question"] += 1
            continue
        if title in forbid_titles:
            skipped["forbidden_title"] += 1
            continue
        row = build_refusal(f"squad_{split_name}_ref_{k}", title, ctx, qa, pool,
                            distractors, n_chunks, rng)
        if row is None:
            skipped["refusal_unbuildable"] += 1
            continue
        rows.append(row)
        seen_q.add(q)
        k += 1
    n_ref_built = k

    rng.shuffle(rows)
    return rows, {"n_answerable": n_ans_built, "n_refusal": n_ref_built,
                  "target_n_answerable": n_ans, "target_n_refusal": n_ref,
                  "n_chunks": n_chunks,
                  "skipped": dict(skipped)}


# ---------------------------------------------------------------------------
# verification -- run on the FINAL rows, independent of the builders
# ---------------------------------------------------------------------------
def verify(rows, label, n_chunks):
    """Re-derive every claim from the written rows. Raises on any violation."""
    st = Counter()
    ref_chars, ans_chars = [], []
    for r in rows:
        assert set(["input_text", "target_text", "memory_texts", "relevant_indices",
                    "dialogue_id", "session_idx", "recall_required"]) <= set(r), \
            f"{label}: schema missing canonical field in {r['dialogue_id']}"
        mts = r["memory_texts"]
        rel = r["relevant_indices"]
        assert isinstance(mts, list) and len(mts) == n_chunks, \
            f"{label}: {r['dialogue_id']} has {len(mts)} chunks, expected {n_chunks}"
        assert all(0 <= i < len(mts) for i in rel), f"{label}: relevant_indices out of range"
        assert len(set(mts)) == len(mts), f"{label}: {r['dialogue_id']} duplicate chunk"
        is_ref = (r["target_text"] == REFUSAL_TEXT)
        gn = normalize_answer(r["target_text"])
        nchar = sum(len(m) for m in mts)
        if is_ref:
            st["refusal"] += 1
            ref_chars.append(nchar)
            assert rel == [], f"{label}: refusal row has non-empty relevant_indices"
            erased = r.get("_erased_golds") or []
            assert erased, f"{label}: refusal row {r['dialogue_id']} erased nothing"
            for g in erased:
                for c in mts:
                    assert not gold_in(c, normalize_answer(g)), \
                        f"{label}: REFUSAL row {r['dialogue_id']} leaks gold {g!r}"
            st["refusal_gold_absent_verified"] += 1
        else:
            st["answerable"] += 1
            ans_chars.append(nchar)
            assert rel, f"{label}: answerable row has empty relevant_indices"
            in_rel = any(gold_in(mts[i], gn) for i in rel)
            assert in_rel, \
                f"{label}: ANSWERABLE row {r['dialogue_id']} gold {r['target_text']!r} " \
                f"not in memory_texts[{rel}]"
            st["answerable_gold_in_relevant_verified"] += 1
            # relevant_indices must be EXACTLY the gold-bearing set (no random pointer)
            allhits = sorted(i for i, c in enumerate(mts) if gold_in(c, gn))
            assert sorted(rel) == allhits, \
                f"{label}: relevant_indices {rel} != gold-bearing chunks {allhits}"
            st["relevant_indices_exact"] += 1

    # No length shortcut: refusal and answerable strata must have comparable
    # context length, else "short context" alone predicts the label.
    if ref_chars and ans_chars:
        mr = sum(ref_chars) / len(ref_chars)
        ma = sum(ans_chars) / len(ans_chars)
        ratio = mr / ma
        st["mean_ctx_chars_refusal"] = round(mr)
        st["mean_ctx_chars_answerable"] = round(ma)
        assert 0.80 <= ratio <= 1.25, \
            f"{label}: LENGTH SHORTCUT -- refusal ctx {mr:.0f} chars vs answerable " \
            f"{ma:.0f} chars (ratio {ratio:.2f})"
    return dict(st)


# ---------------------------------------------------------------------------
def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default="data/squad_raw")
    ap.add_argument("--out_dir", default="data/paperC_squad_v2")
    ap.add_argument("--refusal_rate", type=float, action="append", default=None,
                    help="repeatable; default = 0.0 0.25 0.5")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_chunks", type=int, default=6,
                    help="every row gets exactly this many memory_texts chunks "
                         "(deleted evidence is replaced by other-article "
                         "distractors) so context length cannot leak the label")
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--download", action="store_true",
                    help="fetch missing raw SQuAD via hy-proxy (resume loop)")
    args = ap.parse_args()

    rates = args.refusal_rate or [0.0, 0.25, 0.5]

    print("[raw] loading SQuAD releases ...", flush=True)
    tr_v1 = load_raw(args.raw_dir, "train-v1.1.json", args.download)
    tr_v2 = load_raw(args.raw_dir, "train-v2.0.json", args.download)
    dv_v1 = load_raw(args.raw_dir, "dev-v1.1.json", args.download)
    dv_v2 = load_raw(args.raw_dir, "dev-v2.0.json", args.download)

    print("[pool] collecting candidates ...", flush=True)
    tr_ans, tr_ref, tr_dis = collect_pools(tr_v1, tr_v2, "train")
    va_ans, va_ref, va_dis = collect_pools(dv_v1, dv_v2, "val")
    print(f"  train pool: answerable={len(tr_ans)} refusal={len(tr_ref)} distractors={len(tr_dis)}")
    print(f"  val   pool: answerable={len(va_ans)} refusal={len(va_ref)} distractors={len(va_dis)}")

    # split hygiene: val comes from the dev releases, train from the train
    # releases -> disjoint by construction.  Enforce it anyway (article titles
    # AND question strings), so a future pool change cannot silently leak.
    val_titles = {t for t, _c, _q in va_ans} | {t for t, _c, _q, _p in va_ref}
    val_questions = ({q["question"].strip() for _t, _c, q in va_ans}
                     | {q["question"].strip() for _t, _c, q, _p in va_ref})

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"generator": os.path.basename(__file__), "seed": args.seed,
                "refusal_text": REFUSAL_TEXT, "cn_prefix": CN_PREFIX,
                "raw_dir": args.raw_dir, "n_chunks": args.n_chunks,
                "note": "val<-dev-v1.1/dev-v2.0, train<-train-v1.1/train-v2.0; "
                        "refusals built by DELETING gold-bearing chunks and "
                        "backfilling other-article distractors to a fixed chunk "
                        "count; relevant_indices = exact gold-bearing set",
                "tiers": {}}

    for rate in rates:
        tag = f"refusal{int(round(rate*100)):02d}"
        print(f"\n=== refusal_rate={rate} ({tag}) ===", flush=True)
        tier = {"refusal_rate": rate}
        for split, n_total, ans_pool, ref_pool, dis in (
                ("train", args.n_train, tr_ans, tr_ref, tr_dis),
                ("val", args.n_val, va_ans, va_ref, va_dis)):
            fq = val_questions if split == "train" else None
            ft = val_titles if split == "train" else None
            rows, info = build_split(split, n_total, rate, ans_pool, ref_pool, dis,
                                     args.n_chunks,
                                     seed=args.seed + int(rate * 1000) + (0 if split == "train" else 7),
                                     forbid_questions=fq, forbid_titles=ft)
            v = verify(rows, f"{split}_{tag}", args.n_chunks)
            path = os.path.join(args.out_dir, f"{split}_{tag}.jsonl")
            write_jsonl(path, rows)
            actual = v.get("refusal", 0) / max(len(rows), 1)
            print(f"  [{split}] n={len(rows)} refusal={v.get('refusal',0)} "
                  f"({actual*100:.2f}%) answerable={v.get('answerable',0)} "
                  f"-> {path}")
            print(f"           asserts: {v}")
            if info["skipped"]:
                print(f"           skipped: {info['skipped']}")
            tier[split] = {"path": path, "n": len(rows),
                           "refusal_rate_actual": actual, "build": info,
                           "verify": v,
                           "bytes": os.path.getsize(path)}
        d = abs(tier["train"]["refusal_rate_actual"] - tier["val"]["refusal_rate_actual"])
        tier["train_val_refusal_skew_pp"] = d * 100
        print(f"  train/val refusal skew = {d*100:.2f}pp  "
              f"(legacy squad_{{train,val}}.jsonl skew = 32.29pp)")
        assert d < 0.01, f"train/val refusal rates diverge by {d*100:.2f}pp"
        manifest["tiers"][tag] = tier

    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[manifest] {mpath}")
    print("ALL ASSERTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
