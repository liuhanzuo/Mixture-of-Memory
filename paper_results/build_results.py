#!/usr/bin/env python3
"""
paper_results/build_results.py  —  READ-ONLY QCMem/CoMem benchmark audit aggregator.

WHAT THIS DOES
  Walks the existing benchmark result directories and parses ONLY machine-written
  artifacts (per-cell JSON summaries, scores.json, eval_config.json) into one
  atomic-cell table:  paper_results/results.csv

WHAT THIS DOES *NOT* DO  (by design — audit constraints)
  * No GPU, no model load, no training, no eval, no inference, no network.
  * Does NOT regenerate predictions.
  * Does NOT rescore predictions (RULER already stores a machine score per cell;
    longbench/locomo store scores.json; BABILong stores NO score on disk, so its
    cells are emitted with an EMPTY score and status='needs_rescore' — we never
    invent a number).
  * Does NOT overwrite any original result file.

Run:
    python3 paper_results/build_results.py
Outputs (all under paper_results/, overwriting only prior audit outputs):
    results.csv                 atomic cells
    _warnings.log               non-fatal parse warnings (never silent)
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

# ROOT/OUT can be overridden via env so the SAME parser can run on the diskB
# mirror (which holds the full 0.6B-32B scale sweep) and emit a CSV there that
# we merge into the canonical results.csv. Default = the repo this file lives in.
ROOT = os.environ.get("AUDIT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.environ.get("AUDIT_OUT") or os.path.join(ROOT, "paper_results")
os.makedirs(OUT, exist_ok=True)

WARN = []
def warn(msg):
    WARN.append(msg)

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
SHARD_RE = re.compile(r"_shard(\d+)of(\d+)")

def model_name_from_path(p):
    if not p:
        return ""
    b = os.path.basename(p.rstrip("/"))
    b = b.replace("Qwen--", "").replace("Qwen-", "Qwen").lower()
    table = {
        "qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B",
        "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
        "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B",
        "qwen3-30b-a3b": "Qwen3-30B-A3B",
    }
    if b in table:
        return table[b]
    bl = os.path.basename(p.rstrip("/")).lower()
    if "llama-3-8b" in bl or "meta-llama-3-8b" in bl:
        return "Llama-3-8B"
    if "olmo" in bl:
        return "OLMo-2-7B"
    if "hunyuan" in bl or "hy3" in bl:
        return "Hunyuan-A13B"
    return os.path.basename(p.rstrip("/"))

_HASH_CACHE = {}
def short_hash(path):
    """sha1 of first 4MB of a file (enough to fingerprint an adapter safetensors)."""
    if not path:
        return ""
    ap = path if os.path.isabs(path) else os.path.join(ROOT, path)
    # adapter dirs: point at the weights file
    if os.path.isdir(ap):
        for cand in ("adapter_model.safetensors", "adapter_model.bin"):
            if os.path.exists(os.path.join(ap, cand)):
                ap = os.path.join(ap, cand)
                break
    if ap in _HASH_CACHE:
        return _HASH_CACHE[ap]
    if not os.path.exists(ap):
        _HASH_CACHE[ap] = "MISSING"
        return "MISSING"
    try:
        h = hashlib.sha1()
        with open(ap, "rb") as f:
            h.update(f.read(4 * 1024 * 1024))
        v = h.hexdigest()[:12]
    except OSError as e:
        warn(f"hash failed {ap}: {e}")
        v = ""
    _HASH_CACHE[ap] = v
    return v

def mtime_iso(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).strftime("%Y-%m-%d")
    except OSError:
        return ""

FIELDS = [
    "run_id", "provenance_level", "protocol_group", "protocol_status", "status",
    "model_name", "model_path", "model_revision", "num_layers", "method",
    "adapter_enabled", "adapter_path", "adapter_hash", "benchmark", "task",
    "split", "context_length", "n_planned", "n_valid", "seed", "resume_j",
    "j_fraction", "selector", "topk", "chunk_size", "metric", "score",
    "score_scale", "stderr", "ci95_low", "ci95_high", "code_commit", "run_date",
    "gpu", "config_path", "predictions_path", "scorer_output_path", "log_path",
    "source_summary_path", "source_disk", "notes",
]

ROWS = []
def emit(**kw):
    row = {f: "" for f in FIELDS}
    row.update(kw)
    ROWS.append(row)

# ----------------------------------------------------------------------------
# protocol grouping heuristic (from actual config, NOT dir name)
# ----------------------------------------------------------------------------
def protocol_group(bench, method, adapter, resume_j, num_layers, n_valid, selector, topk):
    if method != "qcmem":
        return "baseline_" + (method or "unknown")
    frac = (resume_j / num_layers) if (resume_j is not None and num_layers) else None
    ntag = ""
    try:
        nv = int(n_valid)
        if nv >= 400:
            ntag = "n500"
        elif nv >= 90:
            ntag = "n100"
        elif nv >= 40:
            ntag = "n50"
        else:
            ntag = f"n{nv}"
    except (ValueError, TypeError):
        ntag = "nNA"
    if adapter:
        if frac is not None and frac >= 0.40:
            depth = "contentj"
        else:
            depth = "adapter033L"
        return f"qcmem_{depth}_{ntag}"
    # zero-shot
    if frac is None:
        depth = "jNA"
    elif frac <= 0.15:
        depth = "recalloptimal"   # shallow j2/j3/j4
    elif frac <= 0.35:
        depth = "readoutsafe"     # ~0.25-0.33L
    else:
        depth = "deepj"           # >=0.4L (mostly zero-shot-collapse probes)
    return f"qcmem_zs_{depth}_{ntag}"

# ----------------------------------------------------------------------------
# RULER parser (per-cell JSON carries machine score + full qcmem config)
# ----------------------------------------------------------------------------
def parse_ruler():
    base = os.path.join(ROOT, "ruler_results")
    if not os.path.isdir(base):
        warn("no ruler_results/ dir")
        return
    for run in sorted(os.listdir(base)):
        rdir = os.path.join(base, run)
        if not os.path.isdir(rdir):
            continue
        # gather every per-cell json (exclude _summary and eval_config)
        cells = {}  # (task,length) -> list of (score,n,cfg,jsonpath,csvpath,nshards)
        jsons = []
        for dp, _dn, fn in os.walk(rdir):
            for f in fn:
                if not f.endswith(".json"):
                    continue
                if f.startswith("_summary") or "eval_config" in f or f == "config.json":
                    continue
                jsons.append(os.path.join(dp, f))
        if not jsons:
            warn(f"ruler run {run}: no per-cell json found")
            continue
        for jp in jsons:
            try:
                d = json.load(open(jp))
            except (json.JSONDecodeError, OSError) as e:
                warn(f"ruler bad json {jp}: {e}")
                continue
            if "summary" not in d or "task" not in d:
                continue
            task = d.get("task", "")
            length = d.get("length", "")
            summ = d.get("summary", {}) or {}
            score = summ.get("score", None)
            n = summ.get("n", None)
            qc = d.get("qcmem", {}) or {}
            baseline = d.get("baseline", "none")
            key = (task, length)
            csvp = jp[:-5] + ".csv"
            cells.setdefault(key, []).append(
                dict(score=score, n=n, qc=qc, baseline=baseline,
                     model=d.get("model", {}), jp=jp,
                     csv=csvp if os.path.exists(csvp) else "")
            )
        # combine shards per (task,length)
        for (task, length), lst in sorted(cells.items()):
            total_n = 0
            wsum = 0.0
            have_score = False
            rep = lst[0]
            for c in lst:
                if c["n"] is not None and c["score"] is not None:
                    total_n += c["n"]
                    wsum += c["score"] * c["n"]
                    have_score = True
            score = round(wsum / total_n, 2) if (have_score and total_n) else ""
            qc = rep["qc"]
            baseline = rep["baseline"]
            model = rep.get("model", {})
            if not isinstance(model, dict):
                model = {}
            if not isinstance(qc, dict):
                qc = {}
            mp = model.get("model_path", "")
            resume_j = qc.get("resume_j", None)
            numl = qc.get("num_layers", None)
            adapter_path = qc.get("lora_adapter") or ""
            adapter_en = bool(adapter_path)
            if baseline and baseline != "none":
                method = baseline
            elif qc.get("selector") is not None or "qcmem" in run.lower():
                method = "qcmem"
            elif resume_j == 0:
                method = "kvdirect"
            else:
                method = "unknown"
            nshards = len(lst)
            csvpaths = [c["csv"] for c in lst if c["csv"]]
            prov = "A" if csvpaths else "B"
            status = "complete" if score != "" else "unverifiable"
            frac = round(resume_j / numl, 3) if (resume_j is not None and numl) else ""
            emit(
                run_id=f"ruler/{run}/{task}/{length}",
                provenance_level=prov,
                protocol_group=protocol_group("ruler", method, adapter_en, resume_j, numl, total_n, qc.get("selector"), qc.get("topk")),
                protocol_status="",  # filled in analysis layer
                status=status,
                model_name=model_name_from_path(mp),
                model_path=mp,
                num_layers=numl if numl is not None else "",
                method=method,
                adapter_enabled=adapter_en,
                adapter_path=adapter_path,
                adapter_hash=short_hash(adapter_path) if adapter_path else "",
                benchmark="RULER",
                task=task,
                split="",
                context_length=length,
                n_planned="",
                n_valid=total_n if total_n else "",
                resume_j=resume_j if resume_j is not None else "",
                j_fraction=frac,
                selector=qc.get("selector") or "",
                topk=qc.get("topk") if qc.get("topk") is not None else "",
                chunk_size=qc.get("chunk_size") if qc.get("chunk_size") is not None else "",
                metric="string_match",
                score=score,
                score_scale="0-100",
                run_date=mtime_iso(rep["jp"]),
                config_path=os.path.relpath(rep["jp"], ROOT),
                predictions_path=os.path.relpath(csvpaths[0], ROOT) if csvpaths else "",
                source_summary_path=os.path.relpath(rep["jp"], ROOT),
                notes=(f"combined {nshards} shard-cell(s)" if nshards > 1 else ""),
            )

# ----------------------------------------------------------------------------
# LongBench parser (scores.json f1/em + eval_config)
# ----------------------------------------------------------------------------
def parse_longbench():
    base = os.path.join(ROOT, "longbench_results")
    if not os.path.isdir(base):
        return
    for run in sorted(os.listdir(base)):
        rdir = os.path.join(base, run)
        if not os.path.isdir(rdir):
            continue
        sp = os.path.join(rdir, "scores.json")
        cfgp = ""
        for c in glob.glob(os.path.join(rdir, "eval_config*.json")):
            cfgp = c
            break
        cfg = {}
        if cfgp:
            try:
                cfg = json.load(open(cfgp))
            except (json.JSONDecodeError, OSError) as e:
                warn(f"longbench cfg {cfgp}: {e}")
        mp = cfg.get("model_path", "")
        resume_j = cfg.get("resume_j", None)
        numl = cfg.get("num_layers", None)
        adapter_path = cfg.get("lora_adapter") or ""
        baseline = cfg.get("baseline", "none")
        method = baseline if (baseline and baseline != "none") else ("qcmem" if adapter_path or cfg.get("selector") else "unknown")
        if not os.path.exists(sp):
            warn(f"longbench run {run}: no scores.json (config only)")
            emit(run_id=f"longbench/{run}", provenance_level="C" if cfg else "U",
                 protocol_status="", status="unverifiable", benchmark="LongBench",
                 model_name=model_name_from_path(mp), model_path=mp, method=method,
                 adapter_enabled=bool(adapter_path), adapter_path=adapter_path,
                 resume_j=resume_j if resume_j is not None else "",
                 config_path=os.path.relpath(cfgp, ROOT) if cfgp else "",
                 notes="no scores.json on disk")
            continue
        try:
            sc = json.load(open(sp))
        except (json.JSONDecodeError, OSError) as e:
            warn(f"longbench scores {sp}: {e}")
            continue
        frac = round(resume_j / numl, 3) if (resume_j is not None and numl) else ""
        for task, v in sc.items():
            if not isinstance(v, dict):
                continue
            n = v.get("n_samples", "")
            f1 = v.get("f1", "")
            emit(
                run_id=f"longbench/{run}/{task}",
                provenance_level="A",
                protocol_group=protocol_group("longbench", method, bool(adapter_path), resume_j, numl, n, cfg.get("selector"), cfg.get("topk")),
                protocol_status="", status="complete",
                model_name=model_name_from_path(mp), model_path=mp,
                num_layers=numl if numl is not None else "",
                method=method, adapter_enabled=bool(adapter_path),
                adapter_path=adapter_path, adapter_hash=short_hash(adapter_path) if adapter_path else "",
                benchmark="LongBench", task=task, split="full",
                n_valid=n, resume_j=resume_j if resume_j is not None else "",
                j_fraction=frac, selector=cfg.get("selector") or "",
                topk=cfg.get("topk") if cfg.get("topk") is not None else "",
                chunk_size=cfg.get("chunk_size") if cfg.get("chunk_size") is not None else "",
                metric="qa_f1", score=round(f1, 3) if isinstance(f1, (int, float)) else f1,
                score_scale="0-100", run_date=mtime_iso(sp),
                config_path=os.path.relpath(cfgp, ROOT) if cfgp else "",
                scorer_output_path=os.path.relpath(sp, ROOT),
                source_summary_path=os.path.relpath(sp, ROOT),
                notes="" if task != "AVERAGE" else "macro-avg over datasets",
            )

# ----------------------------------------------------------------------------
# LoCoMo parser (scores.json f1/em/acc by category + eval_config)
# ----------------------------------------------------------------------------
def parse_locomo():
    base = os.path.join(ROOT, "locomo_results")
    if not os.path.isdir(base):
        return
    for run in sorted(os.listdir(base)):
        rdir = os.path.join(base, run)
        if not os.path.isdir(rdir):
            continue
        sp = os.path.join(rdir, "scores.json")
        cfgp = ""
        for c in glob.glob(os.path.join(rdir, "eval_config*.json")):
            cfgp = c
            break
        cfg = {}
        if cfgp:
            try:
                cfg = json.load(open(cfgp))
            except (json.JSONDecodeError, OSError) as e:
                warn(f"locomo cfg {cfgp}: {e}")
        mp = cfg.get("model_path", "")
        resume_j = cfg.get("resume_j", None)
        numl = cfg.get("num_layers", None)
        adapter_path = cfg.get("lora_adapter") or ""
        baseline = cfg.get("baseline", "none")
        method = baseline if (baseline and baseline != "none") else ("qcmem" if adapter_path or cfg.get("selector") else "unknown")
        if not os.path.exists(sp):
            warn(f"locomo run {run}: no scores.json")
            emit(run_id=f"locomo/{run}", provenance_level="C" if cfg else "U",
                 status="unverifiable", benchmark="LoCoMo",
                 model_name=model_name_from_path(mp), model_path=mp, method=method,
                 adapter_enabled=bool(adapter_path), adapter_path=adapter_path,
                 config_path=os.path.relpath(cfgp, ROOT) if cfgp else "",
                 notes="no scores.json on disk")
            continue
        try:
            sc = json.load(open(sp))
        except (json.JSONDecodeError, OSError) as e:
            warn(f"locomo scores {sp}: {e}")
            continue
        frac = round(resume_j / numl, 3) if (resume_j is not None and numl) else ""
        n_all = sc.get("n_samples", "")
        topk = cfg.get("topk")
        # overall row (report both f1 and the internal substring 'acc' — labelled)
        for metric_key, metric_name, note in [
            ("overall_f1", "token_f1", "SQuAD token-F1 (LoCoMo-paper comparable)"),
            ("overall_acc", "substring_acc", "INTERNAL substring proxy, NOT official LLM-judge"),
            ("overall_em", "em", ""),
        ]:
            if metric_key in sc:
                emit(
                    run_id=f"locomo/{run}/overall/{metric_name}",
                    provenance_level="A",
                    protocol_group=protocol_group("locomo", method, bool(adapter_path), resume_j, numl, n_all, cfg.get("selector"), topk),
                    protocol_status="", status="complete",
                    model_name=model_name_from_path(mp), model_path=mp,
                    num_layers=numl if numl is not None else "",
                    method=method, adapter_enabled=bool(adapter_path),
                    adapter_path=adapter_path, adapter_hash=short_hash(adapter_path) if adapter_path else "",
                    benchmark="LoCoMo", task="overall", split="full",
                    n_valid=n_all, resume_j=resume_j if resume_j is not None else "",
                    j_fraction=frac, selector=cfg.get("selector") or "",
                    topk=topk if topk is not None else "",
                    chunk_size=cfg.get("chunk_size") if cfg.get("chunk_size") is not None else "",
                    metric=metric_name, score=round(sc[metric_key], 3),
                    score_scale="0-100", run_date=mtime_iso(sp),
                    config_path=os.path.relpath(cfgp, ROOT) if cfgp else "",
                    scorer_output_path=os.path.relpath(sp, ROOT),
                    source_summary_path=os.path.relpath(sp, ROOT), notes=note,
                )

# ----------------------------------------------------------------------------
# BABILong parser (config + prediction count ONLY; NO score persisted on disk)
# ----------------------------------------------------------------------------
def parse_babilong():
    base = os.path.join(ROOT, "babilong_results")
    if not os.path.isdir(base):
        return
    for run in sorted(os.listdir(base)):
        rdir = os.path.join(base, run)
        if not os.path.isdir(rdir):
            continue
        # find one representative per-cell json for config
        rep = None
        cell_csvs = {}  # (task,length) -> [csvpaths]
        for dp, _dn, fn in os.walk(rdir):
            for f in fn:
                if f.endswith(".json") and not f.startswith("_summary") and "eval_config" not in f:
                    if rep is None:
                        rep = os.path.join(dp, f)
                elif f.endswith(".csv"):
                    m = re.match(r"(qa\d+)_(\d+k)_", f)
                    if m:
                        cell_csvs.setdefault((m.group(1), m.group(2)), []).append(os.path.join(dp, f))
        if rep is None:
            warn(f"babilong run {run}: no per-cell json (cannot read config)")
            continue
        try:
            d = json.load(open(rep))
        except (json.JSONDecodeError, OSError) as e:
            warn(f"babilong bad json {rep}: {e}")
            continue
        qc = d.get("qcmem", {}) or {}
        model = d.get("model", {}) or {}
        if not isinstance(model, dict):
            model = {}
        if not isinstance(qc, dict):
            qc = {}
        mp = model.get("model_path", "")
        resume_j = qc.get("resume_j", None)
        numl = qc.get("num_layers", None)
        adapter_path = qc.get("lora_adapter") or ""
        method = "qcmem" if (qc.get("selector") or adapter_path) else ("kvdirect" if resume_j == 0 else "unknown")
        chat_tmpl = d.get("prompt", {}).get("chat_template", None)
        frac = round(resume_j / numl, 3) if (resume_j is not None and numl) else ""
        for (task, length), csvs in sorted(cell_csvs.items()):
            nrows = 0
            for cp in csvs:
                try:
                    with open(cp) as fh:
                        nrows += max(0, sum(1 for _ in fh) - 1)  # minus header
                except OSError as e:
                    warn(f"babilong csv {cp}: {e}")
            emit(
                run_id=f"babilong/{run}/{task}/{length}",
                provenance_level="B",  # config + predictions, but NO score on disk
                protocol_group=protocol_group("babilong", method, bool(adapter_path), resume_j, numl, nrows, qc.get("selector"), qc.get("topk")),
                protocol_status="", status="needs_rescore",
                model_name=model_name_from_path(mp), model_path=mp,
                num_layers=numl if numl is not None else "",
                method=method, adapter_enabled=bool(adapter_path),
                adapter_path=adapter_path, adapter_hash=short_hash(adapter_path) if adapter_path else "",
                benchmark="BABILong", task=task, split="",
                context_length=length, n_valid=nrows,
                resume_j=resume_j if resume_j is not None else "",
                j_fraction=frac, selector=qc.get("selector") or "",
                topk=qc.get("topk") if qc.get("topk") is not None else "",
                chunk_size=(qc.get("chunk_size") or model.get("chunk_size") or ""),
                metric="string_match(TASK_LABELS)", score="",
                score_scale="0-100", run_date=mtime_iso(rep),
                config_path=os.path.relpath(rep, ROOT),
                predictions_path=os.path.relpath(csvs[0], ROOT) if csvs else "",
                notes=f"NO score persisted; needs rescore. chat_template={chat_tmpl}. "
                      f"predictions dated {mtime_iso(csvs[0]) if csvs else '?'} "
                      f"(pre-2026-07-16 thinking-fix if <=07-15 & Qwen3)",
            )

# ----------------------------------------------------------------------------
def _fam(task):
    if task.startswith("niah_single"):
        return "single"
    if task.startswith("niah_multikey"):
        return "mkey"
    if task in ("variable_tracking", "vt"):
        return "vt"
    return task

PROTO_SEL = {"single": "bm25", "mkey": "bm25", "vt": "iter_bm25"}

def derive_usable_excluded():
    """Rule-based split of machine-scored cells into a coherent 'usable_now' cohort
    vs 'excluded_results' (with a reason). Pure function of results.csv rows.
    BABILong is entirely excluded (needs_rescore: no score persisted on disk)."""
    usable, excluded = [], []
    def exc(r, reason):
        rr = dict(r); rr["exclude_reason"] = reason; excluded.append(rr)
    for r in ROWS:
        b = r["benchmark"]; m = r["method"]
        if b == "BABILong":
            exc(r, "needs_rescore_no_score_on_disk"); continue
        if r["score"] == "" or r["status"] != "complete":
            exc(r, "no_machine_score"); continue
        if m != "qcmem":
            exc(r, "baseline_not_primary"); continue
        # n gate
        try:
            n = int(r["n_valid"])
        except (ValueError, TypeError):
            n = 0
        if b == "RULER":
            fam = _fam(r["task"])
            want = PROTO_SEL.get(fam)
            if want and r["selector"] and r["selector"] != want:
                exc(r, f"scorer_mismatch_selector_{r['selector']}_expected_{want}"); continue
            if n < 100:
                exc(r, f"n_mismatch_{n}"); continue
            usable.append(r)
        else:  # LongBench / LoCoMo — full test set, any selector ok
            usable.append(r)
    fields = FIELDS + ["exclude_reason"]
    with open(os.path.join(OUT, "usable_now.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in sorted(usable, key=lambda x: (x["benchmark"], x["model_name"], x["run_id"])):
            w.writerow({k: r.get(k, "") for k in FIELDS})
    with open(os.path.join(OUT, "excluded_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in sorted(excluded, key=lambda x: (x["benchmark"], x.get("exclude_reason", ""), x["run_id"])):
            w.writerow({k: r.get(k, "") for k in fields})
    from collections import Counter
    print(f"[build_results] usable_now={len(usable)}  excluded={len(excluded)}")
    print(f"[build_results] exclude reasons: {dict(Counter(r.get('exclude_reason') for r in excluded))}")


def main():
    parse_ruler()
    parse_longbench()
    parse_locomo()
    parse_babilong()
    # tag local rows with the disk they were parsed from
    this_disk = "diskB" if "zwfy6" in ROOT else "wzc1"
    for r in ROWS:
        if not r.get("source_disk"):
            r["source_disk"] = this_disk
    # OPTIONAL MERGE: if a diskB-parsed CSV was pulled here (read-only, generated
    # by running THIS same script on the diskB mirror), fold it in. diskB holds
    # the full 0.6B-32B scale sweep that is NOT present on local wzc1.
    diskb_csv = os.path.join(OUT, "_results_diskB_raw.csv")
    if os.environ.get("AUDIT_OUT") is None and os.path.exists(diskb_csv):
        seen = {r["run_id"] for r in ROWS}
        n_add = 0
        try:
            for r in csv.DictReader(open(diskb_csv)):
                rid = r.get("run_id", "")
                if rid in seen:
                    continue  # same run_id present on wzc1 → keep wzc1 copy
                row = {f: "" for f in FIELDS}
                for k, v in r.items():
                    if k in row:
                        row[k] = v
                row["source_disk"] = "diskB"
                ROWS.append(row)
                seen.add(rid)
                n_add += 1
            print(f"[build_results] merged {n_add} diskB-only cells from {os.path.basename(diskb_csv)}")
        except (OSError, csv.Error) as e:
            warn(f"could not merge diskB csv: {e}")
    # stable sort
    ROWS.sort(key=lambda r: (str(r["benchmark"]), str(r["model_name"]),
                             str(r["method"]), str(r["run_id"])))
    out_csv = os.path.join(OUT, "results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    with open(os.path.join(OUT, "_warnings.log"), "w") as f:
        f.write("\n".join(WARN) + "\n")
    derive_usable_excluded()
    # stderr summary
    from collections import Counter
    by_bench = Counter(r["benchmark"] for r in ROWS)
    by_prov = Counter(r["provenance_level"] for r in ROWS)
    by_status = Counter(r["status"] for r in ROWS)
    print(f"[build_results] wrote {len(ROWS)} atomic cells -> {out_csv}")
    print(f"[build_results] by benchmark: {dict(by_bench)}")
    print(f"[build_results] by provenance: {dict(by_prov)}")
    print(f"[build_results] by status: {dict(by_status)}")
    print(f"[build_results] {len(WARN)} warnings -> paper_results/_warnings.log")

if __name__ == "__main__":
    main()
