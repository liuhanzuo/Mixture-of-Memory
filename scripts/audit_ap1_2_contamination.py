#!/usr/bin/env python
"""Paper A A-P1.2(a) — CROSS-BENCHMARK train/eval contamination audit (CPU only).

PURPOSE
-------
Generalizes the P0.14 InfiniteBench audit (``scripts/audit_p0_14_contamination.py``)
to EVERY long-context benchmark reported in Paper A, checking each eval document
for verbatim overlap with the flagship LoRA's *only* distillation corpus, PG-19
(``data/pg19_train.jsonl``). PG-19 is public-domain Project-Gutenberg books, so the
one benchmark that also draws on Gutenberg books (LongBench ``narrativeqa``) is the
prime contamination suspect; the others (Wikipedia-, paper-, code-, or synthetic-
sourced) should score near-zero containment and thereby be certified clean.

METHOD (verbatim reuse of the P0.14 engine)
-------------------------------------------
The core functions ``normalize_tokens`` / ``ngram_hashes`` / ``build_train_sketch``
are imported unchanged from ``scripts.audit_p0_14_contamination``. This script only
GENERALIZES the eval side: each benchmark's schema is mapped to a uniform
(doc_text, record_id) list, deduplicated by whitespace-normalized SHA-256, and
scored by long-n-gram containment against a downsampled MinHash-style sketch of the
whole PG-19 training corpus.

  containment(doc) = |unique eval n-gram sketch hashes present in the train sketch|
                     / |unique eval n-gram sketch hashes|

Verdict per unique doc:  >= --contam_high => CONTAMINATED ; < --contam_low => CLEAN ;
else PARTIAL. (Same thresholds / tokenization / xxh64(seed=0) 64-bit hashing /
1-in-``--downsample`` bottom-hash sampling as P0.14, so the two audits are directly
comparable — InfiniteBench is re-audited here as an end-to-end cross-check.)

Membership test: because the PG-19 sketch is a SORTED-unique uint64 array (produced
by ``np.unique`` and reloaded from ``.npy``), presence is computed with
``np.searchsorted`` — mathematically identical to ``np.isin(..., assume_unique=True)``
membership used in P0.14 but O(|eval| log|train|) per doc, which is what makes a
thousands-of-docs cross-benchmark sweep tractable in minutes.

BENCHMARKS AUDITED
------------------
* LongBench + NarrativeQA : ``data/longbench_raw/data/*.jsonl`` (34 tasks); field
                            ``context`` is the eval doc, ``_id`` the record id,
                            row-order ``index`` is the key used by the LongBench
                            scorer (``run_scoring`` dedups by ``index``).
* LoCoMo                  : ``locomo/data/locomo10.json`` (10 convs, 1986 QA). Doc =
                            the flattened conversation transcript
                            (``render_locomo_history``口径, re-implemented here to
                            keep the audit self-contained); every QA of a conv shares
                            the conv's verdict (id = ``conv{c}_qa{q}``, matching the
                            LoCoMo scorer dedup key).
* InfiniteBench           : ``.t27_tmp/infb_eval/{longbook_qa_eng,longbook_choice_eng}.jsonl``
                            re-audited via the P0.14 loader for a full cross-check.
* LongEval                : synthetic lines-retrieval; each context is
                            machine-generated ``line <random-label>: REGISTER_CONTENT
                            is <6-digit number>`` with no natural-language corpus, so
                            it CANNOT overlap PG-19 and is certified
                            CLEAN_BY_CONSTRUCTION (documented, not scored — the stored
                            ``longeval_results/*/longeval_*.json`` hold only
                            predictions, never the synthesized context).

OUTPUTS (under --out_dir, default bench_results/ap1_2_contamination/)
--------------------------------------------------------------------
* ``audit_summary.json``       — per-benchmark books/records CONTAMINATED/PARTIAL/CLEAN,
                                 contamination ratios, clean-subset sizes.
* ``match_list.json``          — per benchmark, per unique doc: containment, verdict,
                                 the (task,id,index) records mapped to it.
* ``per_record_verdict.jsonl`` — one row per eval record with its doc verdict.
* ``clean_subset_ids.json``    — per benchmark, the records whose source doc is CLEAN
                                 (drives the clean-subset re-score).
* ``thresholds.json``          — exact params used.

USAGE
-----
  CUDA_VISIBLE_DEVICES=-1 python scripts/audit_ap1_2_contamination.py \
      --sketch_npy .t27_tmp/pg19_train_sketch_n13_d32.npy \
      --longbench_dir data/longbench_raw/data \
      --locomo_json locomo/data/locomo10.json \
      --longeval_dir longeval_results \
      --infb_dir .t27_tmp/infb_eval \
      --out_dir bench_results/ap1_2_contamination \
      --n 13 --downsample 32 --contam_high 0.80 --contam_low 0.10 --workers 96
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- verbatim reuse of the P0.14 contamination engine --------------------- #
import scripts.audit_p0_14_contamination as p0  # noqa: E402

normalize_tokens = p0.normalize_tokens
ngram_hashes = p0.ngram_hashes
build_train_sketch = p0.build_train_sketch


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _shas(text: str):
    raw = hashlib.sha256(text.encode("utf-8")).hexdigest()
    norm = hashlib.sha256(_norm_ws(text).encode("utf-8")).hexdigest()
    return raw, norm


# --------------------------------------------------------------------------- #
# Parallel per-doc containment (train sketch shared read-only via fork COW)
# --------------------------------------------------------------------------- #
_TRAIN = None   # set in the parent before Pool() so workers inherit it (COW)
_N = 13
_DS = 32


def _pool_init(n, ds):
    global _N, _DS
    _N, _DS = n, ds


def _containment_worker(job):
    """job = (doc_key, text) -> (doc_key, n_sampled, n_hit).

    Presence of each downsampled eval n-gram hash in the sorted-unique PG-19 train
    sketch via searchsorted (== isin membership, but O(log) per hash).
    """
    doc_key, text = job
    eg = ngram_hashes(normalize_tokens(text), _N, _DS)
    if eg.size == 0 or _TRAIN.size == 0:
        return doc_key, int(eg.size), 0
    pos = np.searchsorted(_TRAIN, eg)
    pos = np.minimum(pos, _TRAIN.size - 1)
    n_hit = int(np.count_nonzero(_TRAIN[pos] == eg))
    return doc_key, int(eg.size), n_hit


def compute_containment(docs: dict, n: int, downsample: int, workers: int) -> dict:
    """docs = {doc_key: doc_dict(with '_text')} -> {doc_key: (n_sampled, n_hit, ratio)}."""
    if not docs:
        return {}
    jobs = [(k, b["_text"]) for k, b in docs.items()]
    out = {}
    t0 = time.time()
    # fork context (NOT the 3.14 default forkserver/spawn) so workers inherit the
    # 472MB _TRAIN sketch read-only via copy-on-write — no per-worker reload.
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs)), initializer=_pool_init,
                  initargs=(n, downsample)) as pool:
        for i, (doc_key, n_samp, n_hit) in enumerate(
                pool.imap_unordered(_containment_worker, jobs, chunksize=1)):
            ratio = (n_hit / n_samp) if n_samp else 0.0
            out[doc_key] = (n_samp, n_hit, ratio)
            if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                print(f"    [containment] {i + 1}/{len(jobs)} docs "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return out


def verdict_for(ratio: float, contam_high: float, contam_low: float) -> str:
    if ratio >= contam_high:
        return "CONTAMINATED"
    if ratio < contam_low:
        return "CLEAN"
    return "PARTIAL"


# --------------------------------------------------------------------------- #
# Eval-side loaders: each returns (docs, records)
#   docs    = {norm_sha: {"raw_sha256","norm_sha256","char_len","_text",
#                          "records":[record...]}}
#   records = [ {"task","id","index",...,"norm_sha"} ]   (one per eval example)
# --------------------------------------------------------------------------- #
def _new_doc(text):
    raw, norm = _shas(text)
    return {"raw_sha256": raw, "norm_sha256": norm, "char_len": len(text),
            "_text": text, "records": []}


def load_longbench(longbench_dir: str):
    docs, records = {}, []
    files = sorted(glob.glob(os.path.join(longbench_dir, "*.jsonl")))
    for fp in files:
        task = os.path.basename(fp)[:-6]
        with open(fp, encoding="utf-8") as f:
            idx = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                ctx = o.get("context", "") or ""
                _, norm = _shas(ctx)
                b = docs.get(norm)
                if b is None:
                    b = docs[norm] = _new_doc(ctx)
                rec = {"task": task, "id": o.get("_id", f"{task}_{idx}"),
                       "index": idx}
                b["records"].append(rec)
                records.append({**rec, "norm_sha": norm})
                idx += 1
    return docs, records


def _render_locomo_history(conv: dict) -> str:
    """Verbatim-equivalent of eval_qcmem_locomo.render_locomo_history — the exact
    transcript text the LoCoMo eval fed the model (re-implemented so the audit does
    not import the heavy eval module)."""
    parts, i = [], 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        parts.append(f"\n=== Session {i}{(' (' + date + ')') if date else ''} ===")
        for turn in conv[f"session_{i}"]:
            parts.append(f"{turn.get('speaker', '')}: {turn.get('text', '')}")
        i += 1
    return "\n".join(parts)


def load_locomo(locomo_json: str):
    docs, records = {}, []
    with open(locomo_json, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    for conv_idx, d in enumerate(data):
        conv = d.get("conversation", {})
        if not isinstance(conv, dict):
            continue
        sample_id = d.get("sample_id", f"conv{conv_idx}")
        history = _render_locomo_history(conv)
        _, norm = _shas(history)
        b = docs.get(norm)
        if b is None:
            b = docs[norm] = _new_doc(history)
        for qi, qa in enumerate(d.get("qa", [])):
            if not (qa.get("question", "") or "").strip():
                continue
            rec = {"task": "locomo", "id": f"conv{conv_idx}_qa{qi}",
                   "conv_idx": conv_idx, "sample_id": sample_id,
                   "category": qa.get("category", -1)}
            b["records"].append(rec)
            records.append({**rec, "norm_sha": norm})
    return docs, records


def load_infinitebench(infb_dir: str):
    docs, records = {}, []
    files = [("longbook_qa_eng.jsonl", "longbook_qa_eng"),
             ("longbook_choice_eng.jsonl", "longbook_choice_eng")]
    for fn, task in files:
        path = os.path.join(infb_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                ctx = o.get("context", "") or ""
                _, norm = _shas(ctx)
                b = docs.get(norm)
                if b is None:
                    b = docs[norm] = _new_doc(ctx)
                rec = {"task": task, "id": int(o["id"])}
                b["records"].append(rec)
                records.append({**rec, "norm_sha": norm})
    return docs, records


def scan_longeval(longeval_dir: str) -> dict:
    """LongEval carries no natural-language corpus — its context is synthesized
    ``line <label>: REGISTER_CONTENT is <number>`` and only predictions are stored.
    We document it CLEAN_BY_CONSTRUCTION and record what prediction files exist."""
    pred_files = sorted(glob.glob(os.path.join(longeval_dir, "*", "longeval_*.json")))
    n_records = 0
    per_file = []
    for fp in pred_files:
        try:
            d = json.load(open(fp))
            m = len(d.get("records", [])) if isinstance(d, dict) else 0
        except Exception:
            m = 0
        n_records += m
        per_file.append({"file": os.path.relpath(fp, PROJECT_ROOT), "records": m})
    return {
        "verdict": "CLEAN_BY_CONSTRUCTION",
        "reason": "LongEval is a synthetic lines-retrieval task: every context is a "
                  "machine-generated list of `line <random-label>: REGISTER_CONTENT "
                  "is <6-digit number>` entries with no natural-language source, so "
                  "verbatim overlap with the PG-19 book corpus is impossible. The "
                  "stored longeval_*.json hold only predictions (label/expected/"
                  "output/pred), never the synthesized context, so containment is "
                  "not computed.",
        "n_prediction_files": len(pred_files),
        "n_prediction_records_total": n_records,
        "prediction_files": per_file,
    }


# --------------------------------------------------------------------------- #
def audit_benchmark(name, docs, records, contain, contam_high, contam_low):
    """Attach verdicts, build match_list / per-record / clean-subset for one bench."""
    match_list, sha2verdict = [], {}
    counts = {"CONTAMINATED": 0, "PARTIAL": 0, "CLEAN": 0}
    for nsha, b in sorted(docs.items(), key=lambda kv: -kv[1]["char_len"]):
        n_samp, n_hit, ratio = contain.get(nsha, (0, 0, 0.0))
        v = verdict_for(ratio, contam_high, contam_low)
        sha2verdict[nsha] = v
        counts[v] += 1
        match_list.append({
            "norm_sha256": nsha, "raw_sha256": b["raw_sha256"],
            "char_len": b["char_len"], "n_eval_ngrams_sampled": n_samp,
            "n_ngrams_in_train": n_hit, "containment": round(ratio, 4),
            "verdict": v, "n_records": len(b["records"]),
            "records": b["records"],
        })

    per_record, clean_records, rec_counts = [], [], {}
    for r in records:
        v = sha2verdict[r["norm_sha"]]
        row = {k: r[k] for k in r if k != "norm_sha"}
        row["benchmark"] = name
        row["book_norm_sha256"] = r["norm_sha"]
        row["verdict"] = v
        per_record.append(row)
        t = r["task"]
        rc = rec_counts.setdefault(
            t, {"total": 0, "CONTAMINATED": 0, "PARTIAL": 0, "CLEAN": 0})
        rc["total"] += 1
        rc[v] += 1
        if v == "CLEAN":
            clean_records.append(r)

    n_docs = len(docs)
    n_rec = len(records)
    summary = {
        "n_eval_records": n_rec,
        "n_unique_docs": n_docs,
        "docs": counts,
        "doc_contamination_ratio": round(counts["CONTAMINATED"] / n_docs, 4) if n_docs else 0.0,
        "records_by_task": rec_counts,
        "record_contamination_ratio": round(
            sum(rc["CONTAMINATED"] for rc in rec_counts.values()) / n_rec, 4) if n_rec else 0.0,
        "n_clean_records": len(clean_records),
    }
    return summary, match_list, per_record, clean_records


def build_clean_ids(name, clean_records):
    """Benchmark-specific clean-subset id manifest for the re-score step."""
    if name == "longbench":
        by_task = {}
        for r in clean_records:
            by_task.setdefault(r["task"], {"indices": [], "ids": []})
            by_task[r["task"]]["indices"].append(r["index"])
            by_task[r["task"]]["ids"].append(r["id"])
        return {"key": "index (row order; LongBench scorer dedups by index)",
                "by_task": by_task}
    if name == "locomo":
        ids = [r["id"] for r in clean_records]
        convs = sorted({r["conv_idx"] for r in clean_records})
        sample_ids = sorted({r["sample_id"] for r in clean_records})
        return {"key": "id (conv{c}_qa{q}; LoCoMo scorer dedup key)",
                "qa_ids": ids, "clean_conv_idx": convs,
                "clean_sample_ids": sample_ids}
    if name == "infinitebench":
        by_task = {}
        for r in clean_records:
            by_task.setdefault(r["task"], []).append(r["id"])
        return {"key": "id (per task; InfiniteBench scorer key)", "by_task": by_task}
    return {"records": clean_records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sketch_npy", default=".t27_tmp/pg19_train_sketch_n13_d32.npy",
                    help="prebuilt sorted-unique PG-19 train n-gram sketch (.npy).")
    ap.add_argument("--train_corpus", default="data/pg19_train.jsonl",
                    help="only used if --sketch_npy missing (rebuild the sketch).")
    ap.add_argument("--longbench_dir", default="data/longbench_raw/data")
    ap.add_argument("--locomo_json", default="locomo/data/locomo10.json")
    ap.add_argument("--longeval_dir", default="longeval_results")
    ap.add_argument("--infb_dir", default=".t27_tmp/infb_eval")
    ap.add_argument("--out_dir", default="bench_results/ap1_2_contamination")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--downsample", type=int, default=32)
    ap.add_argument("--contam_high", type=float, default=0.80)
    ap.add_argument("--contam_low", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load / build the shared PG-19 train sketch ---- #
    global _TRAIN
    if args.sketch_npy and os.path.exists(args.sketch_npy):
        print(f"[audit] loading prebuilt train sketch: {args.sketch_npy}", flush=True)
        train = np.load(args.sketch_npy)
    else:
        train = build_train_sketch(args.train_corpus, args.n, args.downsample,
                                   args.workers)
    # searchsorted needs ascending order; np.unique output is sorted, but load
    # defensively (cheap once, ~1s for 59M) so membership is always correct.
    if train.size and not (np.diff(train) >= 0).all():
        train = np.sort(train)
    _TRAIN = train
    print(f"[audit] train sketch: {train.size:,} unique downsampled {args.n}-gram "
          f"hashes", flush=True)

    # ---- load every benchmark's eval side ---- #
    loaders = []
    if os.path.isdir(args.longbench_dir):
        loaders.append(("longbench", load_longbench(args.longbench_dir)))
    if os.path.exists(args.locomo_json):
        loaders.append(("locomo", load_locomo(args.locomo_json)))
    if os.path.isdir(args.infb_dir):
        loaders.append(("infinitebench", load_infinitebench(args.infb_dir)))

    all_match, all_per_record, clean_ids = {}, [], {}
    summaries = {}
    for name, (docs, records) in loaders:
        print(f"\n[audit] === {name}: {len(records)} records -> "
              f"{len(docs)} unique docs ===", flush=True)
        contain = compute_containment(docs, args.n, args.downsample, args.workers)
        summary, match_list, per_record, clean_records = audit_benchmark(
            name, docs, records, contain, args.contam_high, args.contam_low)
        summaries[name] = summary
        all_match[name] = match_list
        all_per_record.extend(per_record)
        clean_ids[name] = build_clean_ids(name, clean_records)
        print(f"[audit] {name}: docs {summary['docs']} | "
              f"clean records={summary['n_clean_records']}/{summary['n_eval_records']}",
              flush=True)

    # ---- LongEval: synthetic, clean-by-construction ---- #
    if os.path.isdir(args.longeval_dir):
        le = scan_longeval(args.longeval_dir)
        summaries["longeval"] = le
        clean_ids["longeval"] = {"verdict": "CLEAN_BY_CONSTRUCTION",
                                 "note": le["reason"]}
        print(f"\n[audit] longeval: CLEAN_BY_CONSTRUCTION (synthetic; "
              f"{le['n_prediction_records_total']} pred records over "
              f"{le['n_prediction_files']} files)", flush=True)

    thresholds = {
        "n": args.n,
        "tokenization": "lowercase; [^0-9a-z]+ -> space; whitespace split",
        "hash": "xxhash.xxh64(seed=0) intdigest, 64-bit",
        "downsample": args.downsample,
        "downsample_rule": "keep hash h iff (h %% downsample) == 0 (bottom-hash sketch)",
        "membership": "np.searchsorted vs sorted-unique train sketch (== isin membership)",
        "containment_definition": "|unique eval n-gram sketch hashes present in train sketch| "
                                  "/ |unique eval n-gram sketch hashes|",
        "contam_high_threshold": args.contam_high,
        "contam_low_threshold": args.contam_low,
        "verdict_rule": ">= contam_high => CONTAMINATED; < contam_low => CLEAN; else PARTIAL",
        "train_corpus": os.path.abspath(args.train_corpus),
        "train_sketch_npy": os.path.abspath(args.sketch_npy) if args.sketch_npy else None,
        "train_sketch_unique_hashes": int(train.size),
    }

    top = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "A-P1.2(a) cross-benchmark contamination audit vs PG-19 training corpus",
        "benchmarks": summaries,
        "thresholds": thresholds,
    }

    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(args.out_dir, "match_list.json"), "w") as f:
        json.dump({"summary": top, "benchmarks": all_match}, f, indent=2)
    with open(os.path.join(args.out_dir, "per_record_verdict.jsonl"), "w") as f:
        for r in all_per_record:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "clean_subset_ids.json"), "w") as f:
        json.dump(clean_ids, f, indent=2)
    with open(os.path.join(args.out_dir, "audit_summary.json"), "w") as f:
        json.dump(top, f, indent=2)

    print("\n[audit] ===== CROSS-BENCHMARK SUMMARY =====")
    print(json.dumps(summaries, indent=2))
    print(f"[audit] artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
