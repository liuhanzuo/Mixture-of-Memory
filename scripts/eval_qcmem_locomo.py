#!/usr/bin/env python
"""QCMem mid-depth resume — LoCoMo (long-conversation memory) eval driver.

The long-multi-session-dialogue companion to the other QCMem drivers
(``scripts/eval_qcmem_babilong.py`` synthetic recall,
``scripts/eval_qcmem_longbench.py`` real long-document QA,
``scripts/eval_qcmem_longeval.py`` lines retrieval): this runs the SAME QCMem
write/read primitive (``src/memory/qcmem/qcmem_model.py``) on LoCoMo — the
classic long-term conversational memory benchmark (ACL 2024, snap-research):
10 extended two-speaker conversations (up to ~19 dated sessions each) with 1986
QA pairs across 5 reasoning categories:

  1. Multi-hop reasoning   — needs facts from multiple parts of the dialogue.
  2. Single-hop / NIAH     — needs one specific fact (natural QCMem home turf).
  3. Temporal reasoning    — needs "when did X happen" ordering across sessions.
  4. Open-domain / profile — speaker preferences / traits.
  5. Adversarial           — the fact is NOT in the dialogue; the model should
                             abstain ("not mentioned / I don't know").

Like the RULER / LongBench / LongEval QCMem drivers, this is a thin composition
of one existing, unmodified forward path + a self-contained LoCoMo task frame —
nothing about the QCMem write/read primitive is re-implemented:

  QCMem forward path (imported from ``scripts/eval_qcmem_babilong.py``):
    * ``qcmem_generate``  — chunk the prompt -> write_chunk each chunk to depth j
                            -> selector picks topk context chunks (bm25/recency/
                            oracle/reader_attn) -> read (pack [sink; selected h_j;
                            query h_j], resume layers[j:]) -> greedy decode. Its
                            ``no_retrieval`` arm packs EVERY context chunk (the
                            KV-Direct / HCache baselines).
    * ``run_self_test``   — j=0 correctness gate (QCMem read == full forward,
                            fp32 max|logit diff| < 1e-4).
    * ``QCMemModel``      — the write/read orchestrator (read-only backbone).
    * ``harness``         — ``_locate_needle_chunks`` (oracle needle locator).

  LoCoMo task frame (self-contained here; mirrors the口径 of the mem_space
  LoCoMo eval ``scripts/eval_dialogmem_mem_space.py``, so numbers are comparable):
    * ``build_locomo_samples`` — parse locomo10.json (session_1..N + dates + qa),
                                 flatten each conversation to a dated transcript,
                                 emit one sample per QA (prompt = instruction +
                                 history + question; the question ALSO becomes the
                                 bm25 retrieval query and the query chunk).
    * F1 / EM / substring-acc  — SQuAD-style token-F1 + exact-match + a substring
                                 accuracy proxy (short factual answers), per-answer
                                 max. Category-5 (adversarial) is scored as
                                 abstention-correct (model must refuse).
    * BERTScore (optional, ``--use_bertscore``) — off by default (loads a heavy
                                 roberta-large and needs network on first use); the
                                 primary reported metrics are F1 / EM.

Baselines (``--baseline``, mirrors the other QCMem drivers exactly):
  * ``none``     — normal QCMem (retrieval topk + resume_j + optional LoRA).
  * ``kvdirect`` (2603.19664) — full-depth recompute (forces resume_j=0) + NO
                                retrieval (packs every chunk) + no LoRA
                                (training-free). Read grows O(context).
  * ``hcache``   (2410.05004) — mid-layer recompute (keeps --resume_j) + NO
                                retrieval (packs every chunk) + no LoRA (post-hoc).
                                Read grows O(context).

Model arms (mutually exclusive, mirrors the other QCMem drivers):
  * plain ``--model_path`` backbone (zero-training QCMem),
  * ``--lora_adapter``    — a trained QCMem-distill LoRA (Direction A),
  * ``--bottleneck_ckpt`` — a continued-pretrain funnel-Qwen checkpoint
                            (RECOMMENDED --resume_j == bottleneck_layer+1).

Sharding: samples are the flattened (conversation x QA) list in a STABLE order;
shard ``s`` evaluates the strided slice ``range(N)[s::num_shards]`` and writes
``preds_shard{s}of{S}.jsonl``. ``--score_only`` globs every ``preds_*.jsonl``,
dedups by sample id and recomputes overall + per-category metrics.

Usage (QCMem-distill LoRA on Qwen3-8B, three-way head-to-head):
    # QCMem (retrieval, fixed read)
    python scripts/eval_qcmem_locomo.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12 --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
        --selector bm25 --topk 12 --sink_tokens bos --chunk_size 512 \
        --locomo_data data/dialogmem/locomo10.json \
        --output_dir locomo_results/qcmem_j12 --num_shards 1 --shard_index 0
    # HCache baseline (no retrieval, mid-layer recompute)
    python scripts/eval_qcmem_locomo.py --baseline hcache \
        --model_path .../Qwen--Qwen3-8b --resume_j 12 \
        --locomo_data data/dialogmem/locomo10.json \
        --output_dir locomo_results/hcache_j12
    # Score only (merge all shards):
    python scripts/eval_qcmem_locomo.py --score_only \
        --output_dir locomo_results/qcmem_j12
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# QCMem forward path — reused verbatim, unmodified (same import the other QCMem
# drivers use; loads its explicit-file-path babilong harness on import).
import scripts.eval_qcmem_babilong as qcb  # noqa: E402

QCMemModel = qcb.QCMemModel
qcmem_generate = qcb.qcmem_generate
run_self_test = qcb.run_self_test


# --------------------------------------------------------------------------- #
# category names (LoCoMo, from scripts/README_LOCOMO_EVAL.md)
# --------------------------------------------------------------------------- #
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "single_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


# --------------------------------------------------------------------------- #
# LoCoMo prompt construction  (mirrors scripts/eval_dialogmem_mem_space.py 口径)
# --------------------------------------------------------------------------- #
_LOCOMO_INSTRUCTION = (
    "You are a helpful assistant with memory of a long conversation between "
    "{spa} and {spb}, organized into dated sessions. Read the conversation "
    "history, then answer the question using only the information in the "
    "history. Answer as concisely as possible with a short phrase, date, or "
    "number. Do not explain."
)


def render_locomo_history(conv: dict) -> str:
    """Flatten a LoCoMo conversation (session_1..N + per-session dates) into a
    single dated transcript. Same rendering as eval_dialogmem_mem_space.py so the
    two evals see identical context text."""
    parts = []
    i = 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        parts.append(f"\n=== Session {i}{(' (' + date + ')') if date else ''} ===")
        for turn in conv[f"session_{i}"]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            parts.append(f"{speaker}: {text}")
        i += 1
    return "\n".join(parts)


def _build_dia_id_map(conv: dict) -> dict:
    """dia_id -> turn text, for resolving QA ``evidence`` pointers to the exact
    utterances that support the answer (used by the oracle selector)."""
    dia_map = {}
    i = 1
    while f"session_{i}" in conv:
        for turn in conv[f"session_{i}"]:
            did = turn.get("dia_id", "")
            txt = (turn.get("text", "") or "").strip()
            if did and txt:
                dia_map[did] = txt
        i += 1
    return dia_map


def _resolve_evidence_texts(evidence, dia_map: dict) -> list:
    """Resolve a QA ``evidence`` field (list of dia_ids, possibly grouped like
    ``"D8:9; D9:17"``) into the underlying turn texts. Returns [] when nothing
    resolves. Used only by the oracle selector (an upper-bound retrieval)."""
    texts = []
    if not isinstance(evidence, (list, tuple)):
        return texts
    for eid_raw in evidence:
        if not isinstance(eid_raw, str):
            continue
        for eid in re.split(r"[;,]\s*", eid_raw):
            eid = eid.strip()
            if eid in dia_map and dia_map[eid] not in texts:
                texts.append(dia_map[eid])
    return texts


def build_locomo_samples(data_path: str) -> list:
    """Parse locomo10.json and flatten to one sample per QA.

    Each sample dict carries everything the driver needs downstream:
      id            — stable ``conv{c}_qa{q}`` id (dedup key for --score_only)
      prompt        — full text (instruction + dated history + question + Answer:)
      question      — bare question text (bm25 retrieval query)
      answers       — [str answer]  (category-5 uses adversarial_answer)
      category      — int 1..5 (question type)
      is_abstention — True for category 5 (model should refuse)
      evidence_texts— resolved supporting turn texts (oracle needle candidates)
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())

    samples = []
    for conv_idx, d in enumerate(data):
        conv = d.get("conversation", {})
        if not isinstance(conv, dict):
            continue
        spa = conv.get("speaker_a", "Speaker A")
        spb = conv.get("speaker_b", "Speaker B")
        instr = _LOCOMO_INSTRUCTION.format(spa=spa, spb=spb)
        history = render_locomo_history(conv)
        dia_map = _build_dia_id_map(conv)
        for qi, qa in enumerate(d.get("qa", [])):
            question = (qa.get("question", "") or "").strip()
            if not question:
                continue
            category = qa.get("category", -1)
            # category 5 stores the gold response under adversarial_answer.
            ans = qa.get("answer", None)
            if ans is None:
                ans = qa.get("adversarial_answer", "")
            if isinstance(ans, (int, float)):
                ans = str(ans)
            ans = ans if isinstance(ans, str) else str(ans)
            prompt = (
                f"{instr}\n\n# Conversation history\n{history}\n\n"
                f"# Question\n{question}\n\n# Answer\n"
            )
            samples.append({
                "id": f"conv{conv_idx}_qa{qi}",
                "prompt": prompt,
                "question": question,
                "answers": [ans],
                "category": category,
                "is_abstention": (category == 5),
                "evidence_texts": _resolve_evidence_texts(
                    qa.get("evidence", []), dia_map),
            })
    return samples


# --------------------------------------------------------------------------- #
# oracle needle chunks (LoCoMo has evidence dia_ids -> supporting turn texts)
# --------------------------------------------------------------------------- #
def _oracle_needle_chunks(input_ids, sample, tokenizer, chunk_size):
    """Document-absolute chunk index set that supports the answer, for the oracle
    selector. Prefer the evidence turn texts (LoCoMo's gold supporting utterances,
    the most reliable locator); then fall back to the raw answer string. Returns
    None if nothing can be located (caller's selector then degrades to recency).
    Adversarial (category-5) QAs have no in-context evidence -> None on purpose."""
    probes = list(sample.get("evidence_texts") or [])
    ans = sample["answers"][0] if sample.get("answers") else ""
    if ans:
        probes.append(ans)
    chunks = set()
    for probe in probes:
        probe = (probe or "").strip()
        if not probe:
            continue
        # long evidence utterances tokenise to many ids; _locate_needle_chunks
        # matches the full subsequence, so use a leading discriminative slice
        # when very long to raise the chance of an exact in-context match.
        got = qcb.harness._locate_needle_chunks(
            input_ids, probe, tokenizer, chunk_size)
        if got:
            chunks |= got
    return chunks or None


# --------------------------------------------------------------------------- #
# scoring  (self-contained; mirrors eval_dialogmem_mem_space.py + adds EM)
# --------------------------------------------------------------------------- #
_REFUSAL_RE = re.compile(
    r"\b(i don'?t know|not (mentioned|sure|provided|available|specified)|"
    r"no (information|mention|record)|cannot (find|determine|answer)|"
    r"unanswerable|isn'?t (mentioned|provided)|wasn'?t mentioned)\b",
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc((s or "").lower())))


def compute_f1(pred: str, gt: str) -> float:
    pt = normalize_answer(pred).split()
    gt_t = normalize_answer(gt).split()
    if len(pt) == 0 and len(gt_t) == 0:
        return 1.0
    if len(pt) == 0 or len(gt_t) == 0:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(gt_t)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pt)
    r = num_same / len(gt_t)
    return 2 * p * r / (p + r)


def compute_f1_multi(pred: str, answers: list) -> float:
    return max((compute_f1(pred, a) for a in answers), default=0.0)


def compute_em_multi(pred: str, answers: list) -> float:
    """Exact match after normalization (per-answer max)."""
    np_ = normalize_answer(pred)
    return max((1.0 if np_ == normalize_answer(a) else 0.0 for a in answers),
               default=0.0)


def substring_acc(pred: str, answers: list) -> float:
    """Short-factual-answer accuracy proxy: normalized gold appears as a
    substring of the normalized prediction (or vice-versa). Closer to LoCoMo's
    'judge says correct' than strict EM."""
    np_ = normalize_answer(pred)
    for a in answers:
        na = normalize_answer(a)
        if na and (na in np_ or np_ in na):
            return 1.0
    return 0.0


def score_sample(item: dict) -> dict:
    pred = item.get("pred", "")
    answers = item.get("answers", [])
    is_abs = item.get("is_abstention", False)
    refused = bool(_REFUSAL_RE.search(pred)) or pred.strip() == ""
    if is_abs:
        # adversarial: correct iff the model abstains / says it doesn't know.
        acc = 1.0 if refused else 0.0
        f1 = acc
        em = acc
    else:
        f1 = compute_f1_multi(pred, answers)
        em = compute_em_multi(pred, answers)
        acc = max(substring_acc(pred, answers), 1.0 if f1 >= 0.5 else 0.0)
    out = {"f1": f1, "em": em, "acc": acc, "refused": refused}
    if "bert" in item:
        out["bert"] = float(item["bert"])
    return out


def run_scoring(output_dir: str, use_bertscore: bool = False):
    """Merge every ``preds_*.jsonl`` shard in ``output_dir`` (dedup by id) and
    recompute overall + per-category F1 / EM / acc. Writes ``scores.json``."""
    output_path = Path(output_dir)
    shard_files = sorted(output_path.glob("preds_*.jsonl"))
    if not shard_files:
        print(f"[QCMem-LoCoMo] no prediction files in {output_dir}")
        return None
    preds = []
    seen = set()
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["id"] not in seen:
                    seen.add(item["id"])
                    preds.append(item)
    if not preds:
        print("[QCMem-LoCoMo] no predictions to score")
        return None

    # optional BERTScore over all non-abstention preds (batched once).
    if use_bertscore:
        _attach_bertscore(preds)

    overall = collections.defaultdict(list)
    by_cat = collections.defaultdict(lambda: collections.defaultdict(list))
    for item in preds:
        sc = score_sample(item)
        cat = str(item.get("category", "?"))
        for k in ("f1", "em", "acc"):
            overall[k].append(sc[k])
            by_cat[cat][k].append(sc[k])
        if "bert" in sc:
            overall["bert"].append(sc["bert"])
            by_cat[cat]["bert"].append(sc["bert"])

    n = len(preds)

    def _avg(xs):
        return (sum(xs) / len(xs) * 100) if xs else 0.0

    results = {
        "benchmark": "locomo",
        "n_samples": n,
        "overall_f1": _avg(overall["f1"]),
        "overall_em": _avg(overall["em"]),
        "overall_acc": _avg(overall["acc"]),
        "by_category": {},
    }
    if overall["bert"]:
        results["overall_bert"] = _avg(overall["bert"])

    print(f"\n[QCMem-LoCoMo] locomo  n={n}")
    print(f"  OVERALL   F1={results['overall_f1']:6.2f}  "
          f"EM={results['overall_em']:6.2f}  acc={results['overall_acc']:6.2f}"
          + (f"  BERT={results['overall_bert']:6.2f}"
             if "overall_bert" in results else ""))
    for cat in sorted(by_cat, key=lambda c: (c == "?", c)):
        v = by_cat[cat]
        m = len(v["f1"])
        name = CATEGORY_NAMES.get(int(cat), cat) if cat.lstrip("-").isdigit() else cat
        entry = {"n": m, "f1": _avg(v["f1"]), "em": _avg(v["em"]),
                 "acc": _avg(v["acc"])}
        if v["bert"]:
            entry["bert"] = _avg(v["bert"])
        results["by_category"][cat] = entry
        print(f"  cat{cat:>2} {name:12s} F1={entry['f1']:6.2f}  "
              f"EM={entry['em']:6.2f}  acc={entry['acc']:6.2f}  (n={m})")

    with open(output_path / "scores.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[QCMem-LoCoMo] saved {output_path / 'scores.json'}")
    return results


def _attach_bertscore(preds: list):
    """Attach a per-prediction ``bert`` F1 (only for non-abstention items). Loads
    bert-score lazily; on failure (missing pkg / offline first-use) leaves preds
    untouched and warns."""
    try:
        from bert_score import score as _bscore
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"[QCMem-LoCoMo][WARN] --use_bertscore requested but bert_score "
              f"unavailable ({e}); skipping BERTScore (F1/EM still reported).")
        return
    idxs, cands, refs = [], [], []
    for i, item in enumerate(preds):
        if item.get("is_abstention", False):
            continue
        answers = item.get("answers", [])
        if not answers:
            continue
        idxs.append(i)
        cands.append(item.get("pred", "") or "")
        refs.append(answers[0])
    if not cands:
        return
    try:
        _, _, F = _bscore(cands, refs, lang="en", verbose=False,
                          rescale_with_baseline=False)
    except Exception as e:  # pragma: no cover
        print(f"[QCMem-LoCoMo][WARN] BERTScore computation failed ({e}); "
              f"skipping (F1/EM still reported).")
        return
    for j, i in enumerate(idxs):
        preds[i]["bert"] = float(F[j].item())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="QCMem mid-depth resume — LoCoMo long-conversation eval driver"
    )
    # --- model arm (aligned with the other QCMem drivers) ---
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to plain backbone weights (Qwen3-8B / "
                             "Llama-3-8B). Required unless --score_only.")
    parser.add_argument("--resume_j", type=int, default=12,
                        help="Layer split index j (0=RAG upper bound, L=closed-book).")
    parser.add_argument("--top_prepay_b", type=int, default=0,
                        help="Direction-B top-prepay: run the top b layers "
                             "query-local at read (0=exact connective resume).")
    parser.add_argument("--reuse_kv_blockdiag", action="store_true", default=False,
                        help="QCMem ablation (ii): block-diagonal read attention. "
                             "Only valid with --top_prepay_b 0 and --baseline none.")
    parser.add_argument("--lora_adapter", type=str, default="",
                        help="Optional path to a trained QCMem-distill LoRA "
                             "adapter dir (Direction A). Mutually exclusive with "
                             "--bottleneck_ckpt.")
    parser.add_argument("--bottleneck_ckpt", type=str, default="",
                        help="Optional path to a continued-pretrain funnel-Qwen "
                             "checkpoint (*.pt with 'model_state' + arch_meta.json "
                             "next to it). RECOMMENDED --resume_j == "
                             "bottleneck_layer+1. Mutually exclusive with "
                             "--lora_adapter.")
    parser.add_argument("--baseline", type=str, default="none",
                        choices=["none", "kvdirect", "hcache"],
                        help="Mechanism-level head-to-head baseline (mirrors the "
                             "other QCMem drivers). 'none' = normal QCMem "
                             "(retrieval topk + resume_j + optional LoRA). "
                             "'kvdirect' (2603.19664) = full-depth recompute "
                             "(forces resume_j=0) + NO retrieval (packs every "
                             "chunk) + no LoRA (training-free) — read grows "
                             "O(context). 'hcache' (2410.05004) = mid-layer "
                             "recompute (keeps --resume_j) + NO retrieval (packs "
                             "every chunk) + no LoRA (post-hoc) — read grows "
                             "O(context).")
    parser.add_argument("--selector", type=str, default="bm25",
                        choices=["bm25", "recency", "oracle", "reader_attn"],
                        help="Chunk selector for the read pack. bm25 is the "
                             "deployable default (retrieve context chunks by "
                             "lexical overlap with the question). oracle uses the "
                             "LoCoMo QA evidence dia_ids (upper bound). "
                             "reader_attn scores cached depth-j hiddens.")
    parser.add_argument("--topk", type=int, default=12,
                        help="Number of context chunks to pack into the read.")
    parser.add_argument("--sink_tokens", type=str, default="bos",
                        choices=["bos", "none"],
                        help="Attention-sink anchor at packed position 0.")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="QCMem chunk size (prompt split into chunk_size "
                             "segments; matches the other QCMem drivers).")
    parser.add_argument("--max_new_tokens", type=int, default=48,
                        help="Greedy decode budget per answer (LoCoMo answers are "
                             "short phrases / dates / numbers).")
    # --- LoCoMo task frame ---
    parser.add_argument("--locomo_data", type=str,
                        default="locomo/data/locomo10.json",
                        help="Path to locomo10.json (1986 QA over 10 "
                             "conversations). Relative paths resolve against "
                             "PROJECT_ROOT. Known copy on diskA: "
                             "data/dialogmem/locomo10.json.")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category numbers to keep "
                             "(e.g. '1,2,3'); default = all 5.")
    parser.add_argument("--output_dir", type=str, default="locomo_results/qcmem",
                        help="Directory for per-shard preds JSONL + scores.json.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples total (after category filter; -1 = all).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--use_bertscore", action="store_true", default=False,
                        help="Also compute BERTScore at scoring time (loads a "
                             "heavy roberta-large; needs network on first use). "
                             "Off by default; F1/EM are the primary metrics.")
    parser.add_argument("--use_chat_template", action="store_true", default=False,
                        help="Wrap each prompt in the tokenizer chat template. "
                             "Default OFF (raw-completion prompts, matching the "
                             "other QCMem drivers). Turn ON for instruct backbones.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--score_only", action="store_true",
                        help="Only merge existing per-shard JSONL + recompute "
                             "metrics.")
    parser.add_argument("--self_test", action="store_true", default=False,
                        help="Run the shared QCMem j=0 correctness gate and exit.")
    args = parser.parse_args()

    # --- score-only: merge shards + recompute metrics, then exit ---
    if args.score_only:
        run_scoring(args.output_dir, use_bertscore=args.use_bertscore)
        return

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")
    if not args.model_path:
        parser.error("--model_path is required unless --score_only")

    # --- head-to-head baseline resolution (identical to the other QCMem drivers) --
    no_retrieval = (args.baseline != "none")
    if args.bottleneck_ckpt and args.lora_adapter:
        parser.error("--bottleneck_ckpt (funnel-Qwen arm) and --lora_adapter "
                     "(stock-Qwen LoRA arm) are mutually exclusive; pick one.")
    if args.baseline == "kvdirect":
        if args.resume_j != 0:
            print(f"[QCMem-LoCoMo] baseline=kvdirect -> forcing resume_j "
                  f"{args.resume_j} -> 0 (full-depth K/V recompute).")
        args.resume_j = 0
        if args.lora_adapter:
            print("[QCMem-LoCoMo] baseline=kvdirect is training-free -> "
                  f"ignoring --lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    elif args.baseline == "hcache":
        if args.lora_adapter:
            print("[QCMem-LoCoMo] baseline=hcache is post-hoc (no training) -> "
                  f"ignoring --lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    if no_retrieval and args.reuse_kv_blockdiag:
        parser.error("--reuse_kv_blockdiag is a QCMem ablation and is incompatible "
                     "with --baseline (kvdirect/hcache pack all chunks with the "
                     "standard causal read).")
    if no_retrieval and args.selector == "oracle":
        print("[QCMem-LoCoMo] baseline packs all chunks -> selector 'oracle' "
              "has no effect (ignored).")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    if args.self_test:
        dtype = torch.float32  # tight <1e-4 gate needs fp32

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    data_path = args.locomo_data
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)

    categories = None
    if args.categories:
        categories = {int(c.strip()) for c in args.categories.split(",") if c.strip()}

    print(f"[QCMem-LoCoMo] model_path={model_path}")
    print(f"[QCMem-LoCoMo] locomo_data={data_path}")
    print(f"[QCMem-LoCoMo] baseline={args.baseline} "
          f"(no_retrieval={no_retrieval}) resume_j={args.resume_j} "
          f"top_prepay_b={args.top_prepay_b} reuse_kv_blockdiag={args.reuse_kv_blockdiag} "
          f"selector={args.selector} topk={args.topk} sink={args.sink_tokens} "
          f"chunk_size={args.chunk_size} chat_template={args.use_chat_template} "
          f"dtype={dtype} attn_impl={args.attn_impl}")
    print(f"[QCMem-LoCoMo] categories={categories or 'all'} "
          f"max_samples={args.max_samples} shard={args.shard_index}/{args.num_shards}")

    # local_files_only=True: offline nodes otherwise treat a local dir path as an
    # HF repo_id and error ("Repo id must be in the form ...").
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    L = int(model.config.num_hidden_layers)
    if not (0 <= args.resume_j <= L):
        parser.error(f"--resume_j must be in [0, {L}] for this model; got {args.resume_j}")
    if not (0 <= args.top_prepay_b <= L - args.resume_j):
        parser.error(f"--top_prepay_b must be in [0, {L - args.resume_j}]; got {args.top_prepay_b}")
    if args.reuse_kv_blockdiag and args.top_prepay_b != 0:
        parser.error("--reuse_kv_blockdiag requires --top_prepay_b 0")

    # Direction A: load a trained QCMem-distill LoRA adapter onto the backbone.
    if args.lora_adapter:
        from peft import PeftModel
        print(f"[QCMem-LoCoMo] loading LoRA adapter: {args.lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        model = peft_model.base_model.model

    # Funnel-Qwen arm: rebuild "stock backbone + mid-layer BottleneckLayer funnel"
    # exactly as continued-pretrain saved it, then load the full state_dict.
    # Identical to the eval_qcmem_longbench.py / eval_qcmem_longeval.py funnel loader.
    if args.bottleneck_ckpt:
        from scripts.train_qwen_bottleneck_continued import inject_bottleneck
        meta_path = os.path.join(
            os.path.dirname(os.path.abspath(args.bottleneck_ckpt)), "arch_meta.json")
        if not os.path.exists(meta_path):
            parser.error(f"--bottleneck_ckpt given but arch_meta.json not found "
                         f"next to it at {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        b_layer = int(meta["bottleneck_layer"])
        b_dim = int(meta["bottleneck_dim"])
        print(f"[QCMem-LoCoMo] funnel-Qwen: arch_meta {meta_path} -> "
              f"bottleneck_layer={b_layer} bottleneck_dim={b_dim} "
              f"num_hidden_layers={meta.get('num_hidden_layers')}")
        inject_bottleneck(model, b_layer, b_dim, dtype)
        ck = torch.load(args.bottleneck_ckpt, map_location="cpu")
        state = ck.get("model_state", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if "inv_freq" not in k]
        if bad_missing or unexpected:
            print(f"[QCMem-LoCoMo][WARN] load_state_dict missing={bad_missing[:8]}"
                  f"{'...' if len(bad_missing) > 8 else ''} "
                  f"unexpected={unexpected[:8]}"
                  f"{'...' if len(unexpected) > 8 else ''}")
        model = model.to(device).eval()
        step = ck.get("step")
        print(f"[QCMem-LoCoMo] funnel-Qwen loaded from {args.bottleneck_ckpt} "
              f"(step={step}). RECOMMENDED --resume_j == bottleneck_layer+1 "
              f"(={b_layer + 1}); you passed --resume_j {args.resume_j}.")

    if args.self_test:
        ok = run_self_test(model, tokenizer, device, args.chunk_size)
        sys.exit(0 if ok else 1)

    qc = QCMemModel(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
                    block_diagonal=args.reuse_kv_blockdiag)

    # --- load LoCoMo data + flatten to (conv x QA) samples in a stable order ---
    samples = build_locomo_samples(data_path)
    if categories is not None:
        samples = [s for s in samples if s["category"] in categories]
    print(f"[QCMem-LoCoMo] total samples: {len(samples)}")
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    # strided shard (matches the babilong/ruler [shard_index::num_shards]
    # convention; global id recorded so run_scoring dedups correctly).
    shard = samples[args.shard_index::args.num_shards]
    print(f"[QCMem-LoCoMo] shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard)} samples")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    # record the eval config next to the predictions.
    with open(outdir / f"eval_config{shard_tag}.json", "w") as f:
        cfg = dict(vars(args))
        cfg.update({"no_retrieval": bool(no_retrieval), "num_layers": L,
                    "resolved_model_path": model_path,
                    "resolved_data_path": data_path})
        json.dump(cfg, f, indent=2)

    outfile = outdir / f"preds{shard_tag}.jsonl"
    results_buffer = []
    t0 = time.time()

    for pos, sample in enumerate(tqdm(shard, desc="locomo", leave=True)):
        prompt = sample["prompt"]
        if args.use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        input_ids = ids.to(device)
        n_tokens = int(input_ids.shape[1])
        n_chunks = (n_tokens + args.chunk_size - 1) // args.chunk_size

        bare_q_ids = tokenizer.encode(sample["question"], add_special_tokens=False)

        # oracle needle chunks (LoCoMo evidence turns -> document chunks).
        needle_set = None
        if not no_retrieval and args.selector == "oracle":
            needle_set = _oracle_needle_chunks(
                input_ids, sample, tokenizer, args.chunk_size)

        gen_stats: dict = {}
        try:
            pred = qcmem_generate(
                qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens,
                selector=args.selector, topk=args.topk,
                sink_tokens=args.sink_tokens,
                needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                no_retrieval=no_retrieval, stats=gen_stats,
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            pred = "[OOM]"
            print(f"[OOM] id={sample['id']} n_tok={n_tokens}: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_buffer.append({
            "id": sample["id"],
            "pred": pred,
            "answers": sample["answers"],
            "category": sample["category"],
            "is_abstention": sample["is_abstention"],
            "question": sample["question"],
            "n_tokens": n_tokens,
            "n_chunks": n_chunks,
            "read_len": gen_stats.get("read_len"),
            "n_selected_chunks": gen_stats.get("n_selected_chunks"),
            "n_context_chunks": gen_stats.get("n_context_chunks"),
        })

        if (pos + 1) % 10 == 0 or pos == len(shard) - 1:
            with open(outfile, "w") as f:
                for r in results_buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (pos + 1) % 20 == 0:
            cur = [score_sample(r) for r in results_buffer]
            avg_f1 = sum(c["f1"] for c in cur) / len(cur) * 100
            avg_acc = sum(c["acc"] for c in cur) / len(cur) * 100
            speed = (pos + 1) / (time.time() - t0)
            print(f"  [locomo] {pos+1}/{len(shard)} | {speed:.2f} samples/s | "
                  f"running F1={avg_f1:.1f}% acc={avg_acc:.1f}% | "
                  f"read_len~{gen_stats.get('read_len')} | "
                  f"last_pred='{pred[:60]}'")

    with open(outfile, "w") as f:
        for r in results_buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[QCMem-LoCoMo] shard {args.shard_index}/{args.num_shards} done: "
          f"{len(results_buffer)} samples ({time.time()-t0:.1f}s) -> {outfile}")

    # Single-shard: auto-score (multi-shard: run --score_only after all finish).
    if args.num_shards == 1:
        print("\n[QCMem-LoCoMo] Running scoring (single-shard mode)...")
        run_scoring(args.output_dir, use_bertscore=args.use_bertscore)

    print("\n[QCMem-LoCoMo] Evaluation complete!")


if __name__ == "__main__":
    main()
