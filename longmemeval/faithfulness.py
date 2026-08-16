"""``U`` — the unsupported-claim rate, per B08 leg-1's pre-registration.

PRE-REGISTERED DEFINITION (verbatim, ``B08_LEG1_GATE_PREREG.md`` §5.2)
----------------------------------------------------------------------
    **U — unsupported-claim rate.** Fraction of non-abstention answers
    containing a factual claim not present in **that arm's own context**.
    Denominator **128**.

Three things in that sentence are load-bearing and drive every design choice
below:

1. *"non-abstention"* — the 6 ``_abs`` items in the stratum are **excluded from
   the U denominator**; the prereg's primary denominator is 128 = 134 − 6.
   ``abstention_handling`` says mis-scoring them "corrupts both ACC and U".
2. *"that arm's own context"* — the support set is **per arm**, not the corpus.
   ``A-notes-only``'s context is the notes block alone; ``A-raw``'s is the raw
   evidence. That per-arm conditioning is exactly what makes the notes-only arm
   interpretable, and it is why the harness must log each arm's actual context
   (``run_baseline.py --context_log``) rather than re-deriving it.
3. *"containing a factual claim"* — a per-claim decision aggregated to a
   per-answer boolean. So the scorer emits **per-item and per-claim records**;
   an aggregate with no trail cannot be re-checked.

WHAT IS AND IS NOT NOVEL HERE (binding; ``RELATED_WORK.md`` §7 item 5)
---------------------------------------------------------------------
It is **forbidden** to claim "we introduce a faithfulness / unsupported-claim
metric". **ALCE** (EMNLP 2023, ``2023.emnlp-main.398``), **FActScore** (EMNLP
2023, ``2023.emnlp-main.741``) and **SummaC** (TACL 2022) already define the
machinery: decompose a generation into atomic claims, then check each claim
against a source. This module is a **deterministic re-implementation of that
machinery for this repo's data shapes**. The honest statement is *"no such
scorer exists in THIS repo"* — which is a different claim from *"no such scorer
exists"*.

What B08 contributes is not the metric; it is **measuring it on a notes-only arm
against that arm's own context**, paired against ``A-raw``.

THE SUPPORT TEST IS LEXICAL, AND THAT IS A STATED LIMITATION
------------------------------------------------------------
The prereg fixes the *definition* of U but does **not** fix the operationalisation
of "factual claim" or of "present in the context". Those are free parameters, so
they are pinned **here, PRE-DATA**, and every one of them is written into the
output JSON:

  * **claim unit** = sentence (SummaC's granularity), by a deterministic splitter.
  * **support test** = salient-token grounding. A claim is UNSUPPORTED iff it
    contains at least ``min_ungrounded_salient`` (default 1) *salient* tokens
    that do not occur in the arm's own context. Salient = not a stopword, and
    either numeric or alphabetic with length ≥ ``min_salient_len`` (default 4).
    Normalisation is lowercase + punctuation strip + a plural fold, so trivial
    morphology is not counted as fabrication.
  * **refusals assert nothing.** A claim matching ``_REFUSAL_RE`` (verbatim from
    ``scripts/a02_judge_openweight.py``, itself verbatim from
    ``scripts/eval_qcmem_locomo.py``) contributes no factual claim, so it can
    never be unsupported. Note this is *separate* from abstention **items**,
    which are excluded from the denominator entirely.

This is a **lexical proxy for entailment, not entailment.** It over-flags
paraphrase and under-flags fluent recombination of in-context tokens. Two
mitigations, both implemented:

  * ``score_answer(..., entailment_fn=...)`` is a documented seam: pass an NLI
    model and the plumbing (claim splitting, per-arm support sets, per-item
    records, pairing, bootstrap) is unchanged. The lexical rule is the *default*,
    not the *architecture*.
  * a **sensitivity sweep** over ``min_ungrounded_salient ∈ {1,2,3}`` is always
    emitted next to the primary number, so a reader can see how much of ΔU is an
    artefact of the threshold.

Because ΔU is **paired** on the same items with the same scorer, a constant
lexical bias cancels to first order; a *differential* one does not, which is why
the sweep is mandatory rather than optional.

DETERMINISM
-----------
No model, no sampling, no dict-ordering dependence, and — deliberately — **no
numpy**. The five nodes carry three numpy versions and same-seed
``multinomial`` differs between them
(``memory/numpy-version-split-breaks-cross-node-bootstrap.md``), so the paired
bootstrap uses ``random.Random`` (stdlib, version-stable) with the
pre-registered ``seed=42`` and 10,000 resamples. The same run on any node gives
the same CI.

Usage (0 GPU)::

    python -m longmemeval.faithfulness \\
        --arm A-raw:outputs/b08_leg1/A-raw/context.jsonl \\
        --arm A-notes-only:outputs/b08_leg1/A-notes-only/context.jsonl \\
        --expect_n 134 --expect_scored 128 \\
        --paired A-notes-only:A-raw \\
        --out evidence/b08_leg1_U.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Pre-registered constants
# --------------------------------------------------------------------------- #

#: Refusal detector, verbatim from ``scripts/a02_judge_openweight.py:61-66``
#: (which is itself verbatim from ``scripts/eval_qcmem_locomo.py:305-310``).
#: Reused rather than rewritten so refusal handling is identical to the judge's.
_REFUSAL_RE = re.compile(
    r"\b(i don'?t know|not (mentioned|sure|provided|available|specified)|"
    r"no (information|mention|record)|cannot (find|determine|answer)|"
    r"unanswerable|isn'?t (mentioned|provided)|wasn'?t mentioned)\b",
    re.IGNORECASE,
)

BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 42
#: K2 evaluability precondition (``STATUS.json.kill_gate``): at n=134 the K2
#: clause is only reachable if the observed unsupported-claim discordance is
#: <= this, else the stratum MUST be extended to n=500.
DISC_U_MAX_AT_N134 = 0.0872

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

#: Function words that carry no factual content. Deliberately small and fixed:
#: a long hand-tuned list would be a knob, and the salience floor
#: (``min_salient_len``) already removes most short function words.
_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing done for
from had has have having he her hers him his how i if in into is it its me my
not of on or our ours she so than that the their theirs them then there these
they this those to us was we were what when where which who whom why will with
would you your yours am no yes about after before during over under again more
most some such only own same too very just also there's it's i'm don't
""".split())


# --------------------------------------------------------------------------- #
# Normalisation / tokenisation (deterministic)
# --------------------------------------------------------------------------- #

def _fold(tok: str) -> str:
    """Lowercase + a conservative plural fold. Deterministic, no stemmer dep."""
    t = tok.lower()
    if len(t) > 3 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokens(text: str) -> List[str]:
    return [_fold(m.group(0)) for m in _WORD_RE.finditer(text or "")]


def salient_tokens(text: str, min_salient_len: int = 4) -> List[str]:
    """Content tokens whose absence from the context is evidence of invention."""
    out = []
    for t in tokens(text):
        if t in _STOPWORDS:
            continue
        if t.isdigit() or any(c.isdigit() for c in t):
            out.append(t)
        elif len(t) >= min_salient_len:
            out.append(t)
    return out


def split_claims(answer: str) -> List[str]:
    """Sentence-level claim units (SummaC granularity). Deterministic."""
    text = (answer or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    return [p for p in parts if p]


# --------------------------------------------------------------------------- #
# The support test
# --------------------------------------------------------------------------- #

def context_text_of(record: dict) -> str:
    """Concatenate an item's context blocks EXACTLY as the reader saw them."""
    blocks = record["context_blocks"]
    if not isinstance(blocks, list):
        raise TypeError(
            f"context_blocks must be a list, got {type(blocks).__name__} "
            f"for question_id={record.get('question_id')!r}"
        )
    parts = []
    for i, b in enumerate(blocks):
        if not isinstance(b, dict) or "text" not in b:
            raise KeyError(
                f"context_blocks[{i}] for question_id="
                f"{record.get('question_id')!r} has no 'text' key "
                f"(keys: {sorted(b) if isinstance(b, dict) else type(b).__name__}). "
                "U is defined against that arm's own context, so an unreadable "
                "context block is a protocol failure, not a skippable row."
            )
        parts.append(str(b["text"]))
    return "\n\n".join(parts)


def score_answer(
    answer: str,
    context: str,
    min_ungrounded_salient: int = 1,
    min_salient_len: int = 4,
    entailment_fn: Optional[Callable[[str, str], bool]] = None,
) -> dict:
    """Per-answer unsupported-claim decision with a full per-claim trail.

    ``entailment_fn(claim, context) -> supported: bool`` is the documented seam
    for an NLI backend (ALCE / FActScore / SummaC style). When given, it replaces
    the lexical rule and the per-claim record says so.
    """
    ctx_set = set(tokens(context))
    claims = split_claims(answer)
    claim_records = []
    n_unsupported = 0
    for claim in claims:
        refusal = bool(_REFUSAL_RE.search(claim))
        sal = salient_tokens(claim, min_salient_len)
        ungrounded = sorted({t for t in sal if t not in ctx_set})
        if refusal:
            supported, why = True, "refusal: asserts no factual claim"
        elif entailment_fn is not None:
            supported = bool(entailment_fn(claim, context))
            why = "entailment_fn"
        elif not sal:
            supported, why = True, "no salient token: no checkable claim"
        else:
            supported = len(ungrounded) < min_ungrounded_salient
            why = (f"{len(ungrounded)} ungrounded salient token(s) "
                   f"vs threshold {min_ungrounded_salient}")
        if not supported:
            n_unsupported += 1
        claim_records.append({
            "claim": claim,
            "is_refusal": refusal,
            "n_salient": len(sal),
            "ungrounded_salient": ungrounded,
            "supported": supported,
            "rule": why,
        })
    return {
        "n_claims": len(claims),
        "n_unsupported_claims": n_unsupported,
        # U is per-ANSWER: "answers CONTAINING a factual claim not present".
        "unsupported": n_unsupported > 0,
        "claims": claim_records,
    }


# --------------------------------------------------------------------------- #
# Context-log I/O — fails LOUDLY, never half-loudly
# --------------------------------------------------------------------------- #

_REQUIRED_CONTEXT_FIELDS = ("question_id", "hypothesis", "context_blocks")


def load_context_log(path: str) -> Dict[str, dict]:
    """Load a ``--context_log`` JSONL into ``{question_id: record}``.

    Every failure mode below RAISES. The precedent is a silent 5/8-shard merge
    that corrupted a whole protocol: a loader that skips a bad row and then
    prints a plausible aggregate is worse than one that crashes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"context log not found: {path}")
    out: Dict[str, dict] = {}
    n_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({e})")
            if not isinstance(rec, dict):
                raise TypeError(
                    f"{path}:{lineno}: expected a JSON object, got "
                    f"{type(rec).__name__}"
                )
            missing = [k for k in _REQUIRED_CONTEXT_FIELDS if k not in rec]
            if missing:
                raise KeyError(
                    f"{path}:{lineno}: context record is missing {missing} "
                    f"(keys present: {sorted(rec)}). U cannot be computed "
                    "without the arm's own context; refusing to score a "
                    "partial log."
                )
            qid = rec["question_id"]
            if not isinstance(qid, str):
                # The B04 failure mode, made loud: a non-str id silently misses
                # every str-keyed lookup, so the scorer would report "found N"
                # and "all missing" in the same breath.
                raise TypeError(
                    f"{path}:{lineno}: question_id must be a str, got "
                    f"{type(qid).__name__} ({qid!r}). A non-str id cannot be "
                    "paired against the other arm's log or against the gold "
                    "data, and silently mis-keying it is how an aggregate ends "
                    "up computed over zero rows."
                )
            if qid in out:
                raise ValueError(
                    f"{path}:{lineno}: duplicate question_id {qid!r}. A "
                    "duplicated item would be double-counted in the "
                    "denominator and would break pairing."
                )
            out[qid] = rec
    if not out:
        raise ValueError(f"{path}: no records (read {n_lines} non-empty lines)")
    return out


def is_abstention_qid(qid: str) -> bool:
    """LongMemEval marks abstention questions with an ``_abs`` id suffix."""
    return str(qid).endswith("_abs")


# --------------------------------------------------------------------------- #
# Arm scoring
# --------------------------------------------------------------------------- #

def score_arm(
    records: Dict[str, dict],
    min_ungrounded_salient: int = 1,
    min_salient_len: int = 4,
    entailment_fn: Optional[Callable[[str, str], bool]] = None,
    expect_n: Optional[int] = None,
    expect_scored: Optional[int] = None,
) -> dict:
    """Score one arm. Emits a per-item trail; asserts counts BEFORE metrics.

    ``expect_n`` = items in the log (the cell size, e.g. 134).
    ``expect_scored`` = U's denominator, i.e. non-abstention items (e.g. 128).
    """
    if expect_n is not None and len(records) != expect_n:
        raise ValueError(
            f"arm has {len(records)} items != expect_n {expect_n}. No metric "
            "may be computed on a partial arm (pre-registered read-out point)."
        )
    per_item = []
    for qid in sorted(records):
        rec = records[qid]
        abstention = is_abstention_qid(qid)
        ctx = context_text_of(rec)
        res = score_answer(
            rec["hypothesis"], ctx,
            min_ungrounded_salient=min_ungrounded_salient,
            min_salient_len=min_salient_len,
            entailment_fn=entailment_fn,
        )
        per_item.append({
            "question_id": qid,
            "question_type": rec.get("question_type"),
            "arm": rec.get("arm"),
            "is_abstention": abstention,
            "in_U_denominator": not abstention,
            "hypothesis": rec["hypothesis"],
            "n_context_blocks": len(rec["context_blocks"]),
            "n_context_tokens": len(tokens(ctx)),
            **res,
        })

    scored = [r for r in per_item if r["in_U_denominator"]]
    if expect_scored is not None and len(scored) != expect_scored:
        raise ValueError(
            f"arm has {len(scored)} non-abstention items != expect_scored "
            f"{expect_scored} (abstention items are excluded from U's "
            "denominator; mis-counting them corrupts both ACC and U)."
        )
    n = len(scored)
    k = sum(1 for r in scored if r["unsupported"])
    by_type: Dict[str, Dict[str, float]] = {}
    for r in scored:
        b = by_type.setdefault(str(r["question_type"]), {"n": 0, "k": 0})
        b["n"] += 1
        b["k"] += int(r["unsupported"])
    for b in by_type.values():
        b["U_pct"] = round(100.0 * b["k"] / b["n"], 4) if b["n"] else None

    return {
        "n_items": len(per_item),
        "n_abstention_excluded": len(per_item) - n,
        "n_scored": n,
        "n_unsupported_answers": k,
        "U_pct": round(100.0 * k / n, 4) if n else None,
        "U_by_question_type": by_type,
        "per_item": per_item,
    }


# --------------------------------------------------------------------------- #
# Paired statistics (stdlib RNG only -- cross-node reproducible)
# --------------------------------------------------------------------------- #

def paired_bootstrap_gap(
    pairs: Sequence[Tuple[int, int]],
    iters: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[dict]:
    """95% paired-bootstrap CI of mean(b - a) in pp. Same scaffold shape as
    ``scripts/analyze_p019_recall_readout.py:120``, numpy-free on purpose."""
    if not pairs:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    a = [int(x[0]) for x in pairs]
    b = [int(x[1]) for x in pairs]
    obs = (sum(b) - sum(a)) / n
    diffs = []
    for _ in range(iters):
        sa = sb = 0
        for _ in range(n):
            j = rng.randrange(n)
            sa += a[j]
            sb += b[j]
        diffs.append((sb - sa) / n)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters) - 1]
    return {
        "gap_pp": round(100 * obs, 4),
        "ci95_pp": [round(100 * lo, 4), round(100 * hi, 4)],
        "n_pairs": n,
        "iters": iters,
        "seed": seed,
        "rng": "python random.Random (numpy deliberately avoided: three numpy "
               "versions across the five nodes make same-seed multinomial "
               "node-dependent)",
    }


def pair_arms(arm_b: dict, arm_a: dict) -> List[Tuple[int, int]]:
    """Pair two scored arms on question_id. Any asymmetry RAISES."""
    ida = {r["question_id"] for r in arm_a["per_item"] if r["in_U_denominator"]}
    idb = {r["question_id"] for r in arm_b["per_item"] if r["in_U_denominator"]}
    if ida != idb:
        only_a, only_b = sorted(ida - idb)[:5], sorted(idb - ida)[:5]
        raise ValueError(
            f"arms are not paired: {len(ida)} vs {len(idb)} scored items; "
            f"only in A: {only_a}; only in B: {only_b}. An unpaired bootstrap "
            "would silently report a CI for a different comparison."
        )
    ma = {r["question_id"]: int(r["unsupported"])
          for r in arm_a["per_item"] if r["in_U_denominator"]}
    mb = {r["question_id"]: int(r["unsupported"])
          for r in arm_b["per_item"] if r["in_U_denominator"]}
    return [(ma[q], mb[q]) for q in sorted(ma)]


def discordance(pairs: Sequence[Tuple[int, int]]) -> float:
    """P(discordant pair) — the K2 evaluability quantity."""
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a != b) / len(pairs)


def delta_u(arm_notes_only: dict, arm_raw: dict) -> dict:
    """Delta_U = U(A-notes-only) - U(A-raw), paired, plus the K2 precondition.

    The escalation decision is a function of ``disc_U`` ALONE and is reported
    next to it, exactly as pre-registered: it is a dispersion quantity, decided
    before any Delta's sign is inspected, so it cannot be used to shop for a
    result.
    """
    pairs = pair_arms(arm_notes_only, arm_raw)
    disc = discordance(pairs)
    boot = paired_bootstrap_gap(pairs)
    n = len(pairs)
    return {
        "definition": "Delta_U = U(A-notes-only) - U(A-raw), absolute pp, paired",
        "delta_U_pp": boot["gap_pp"],
        "ci95_pp": boot["ci95_pp"],
        "bootstrap": boot,
        "disc_U": round(disc, 6),
        "n_pairs": n,
        "k2_evaluability": {
            "threshold_disc_U": DISC_U_MAX_AT_N134,
            "resolvable_at_n134": disc <= DISC_U_MAX_AT_N134,
            "mandatory_action_if_not": (
                "extend the stratum to the full n=500 (same three arms, same "
                "frozen retrieval). MANDATORY when triggered, not optional."
            ),
            "note": (
                "Only meaningful at n=134. The 0.0872 bound is the disc_U at "
                "which the 1.96*sqrt(disc/134) paired half-width equals the "
                "5.0 pp K2 threshold."
            ),
        },
        "survival_branch_note": (
            "Delta_U CI entirely above +5.0 pp => 'notes are an adjunct, never "
            "a substitute', MEASURED. Per NOVELTY_VERDICT.md this is now the "
            "sole load-bearing novelty, so K2 is the decisive clause."
        ),
        "ci_entirely_above_5pp": boot["ci95_pp"][0] > 5.0,
        "k2_fires_ub_below_5pp": boot["ci95_pp"][1] < 5.0,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_arm(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"--arm expects NAME:PATH, got {spec!r}")
    name, path = spec.split(":", 1)
    if not name or not path:
        raise SystemExit(f"--arm expects NAME:PATH, got {spec!r}")
    return name, path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m longmemeval.faithfulness",
        description="U (unsupported-claim rate) scorer for B08 leg-1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arm", action="append", default=[], metavar="NAME:PATH",
                   help="Arm name and its --context_log JSONL. Repeatable.")
    p.add_argument("--paired", action="append", default=[], metavar="B:A",
                   help="Report Delta_U = U(B) - U(A), paired. Repeatable. "
                        "For B08: --paired A-notes-only:A-raw")
    p.add_argument("--expect_n", type=int, default=None,
                   help="Assert items per arm (B08 stratum: 134).")
    p.add_argument("--expect_scored", type=int, default=None,
                   help="Assert U's denominator per arm (B08: 128 = 134-6 _abs).")
    p.add_argument("--min_ungrounded_salient", type=int, default=1,
                   help="Primary rule: a claim is UNSUPPORTED at >= this many "
                        "ungrounded salient tokens.")
    p.add_argument("--min_salient_len", type=int, default=4,
                   help="Alphabetic tokens shorter than this are not salient "
                        "(numerals always are).")
    p.add_argument("--sensitivity", type=str, default="1,2,3",
                   help="Also report U at these min_ungrounded_salient values, "
                        "so threshold dependence is visible next to the "
                        "primary number.")
    p.add_argument("--out", type=str, default=None,
                   help="Write the full JSON (incl. per-item records) here.")
    p.add_argument("--per_item_out", type=str, default=None,
                   help="Optional separate JSONL of per-item records.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.arm:
        print("ERROR: at least one --arm NAME:PATH is required.", file=sys.stderr)
        return 2

    arms: Dict[str, dict] = {}
    sources: Dict[str, str] = {}
    for spec in args.arm:
        name, path = _parse_arm(spec)
        if name in arms:
            raise SystemExit(f"duplicate arm name {name!r}")
        recs = load_context_log(path)
        arms[name] = score_arm(
            recs,
            min_ungrounded_salient=args.min_ungrounded_salient,
            min_salient_len=args.min_salient_len,
            expect_n=args.expect_n,
            expect_scored=args.expect_scored,
        )
        sources[name] = os.path.abspath(path)

    sens_levels = [int(x) for x in args.sensitivity.split(",") if x.strip()]
    sensitivity = {}
    for name, spec in zip(arms, args.arm):
        _, path = _parse_arm(spec)
        recs = load_context_log(path)
        sensitivity[name] = {}
        for lvl in sens_levels:
            s = score_arm(recs, min_ungrounded_salient=lvl,
                          min_salient_len=args.min_salient_len)
            sensitivity[name][str(lvl)] = {
                "U_pct": s["U_pct"], "n_scored": s["n_scored"],
                "n_unsupported_answers": s["n_unsupported_answers"],
            }

    paired = {}
    for spec in args.paired:
        b, a = _parse_arm(spec)
        if b not in arms or a not in arms:
            raise SystemExit(f"--paired {spec!r} names an arm that was not scored")
        paired[f"{b}-minus-{a}"] = delta_u(arms[b], arms[a])

    out = {
        "metric": "U -- unsupported-claim rate",
        "prereg_definition": (
            "Fraction of non-abstention answers containing a factual claim not "
            "present in THAT ARM'S OWN context. Denominator 128 = 134 - 6 _abs. "
            "(B08_LEG1_GATE_PREREG.md section 5.2)"
        ),
        "prior_art_ack": (
            "The claim-decomposition + per-claim support machinery is ALCE "
            "(2023.emnlp-main.398) / FActScore (2023.emnlp-main.741) / SummaC "
            "(TACL 2022). NOT a new metric family: the true statement is 'no "
            "such scorer existed in THIS repo'. RELATED_WORK.md section 7 item 5."
        ),
        "operationalisation": {
            "claim_unit": "sentence (deterministic splitter)",
            "support_test": "salient-token grounding (LEXICAL PROXY, not NLI)",
            "min_ungrounded_salient": args.min_ungrounded_salient,
            "min_salient_len": args.min_salient_len,
            "plural_fold": True,
            "refusals_assert_nothing": True,
            "refusal_regex": _REFUSAL_RE.pattern,
            "entailment_fn_seam": (
                "score_answer(entailment_fn=...) swaps in an NLI backend "
                "without changing any other plumbing."
            ),
            "known_bias": (
                "over-flags paraphrase, under-flags fluent recombination of "
                "in-context tokens. Delta_U is paired with one scorer, so a "
                "CONSTANT bias cancels to first order and a DIFFERENTIAL one "
                "does not -- hence the mandatory sensitivity sweep."
            ),
        },
        "asserts": {"expect_n": args.expect_n,
                    "expect_scored": args.expect_scored},
        "sources": sources,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "per_item"}
                 for k, v in arms.items()},
        "sensitivity_min_ungrounded_salient": sensitivity,
        "paired": paired,
    }

    if args.per_item_out:
        parent = os.path.dirname(os.path.abspath(args.per_item_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.per_item_out, "w", encoding="utf-8") as f:
            for name, arm in arms.items():
                for rec in arm["per_item"]:
                    f.write(json.dumps({"arm_name": name, **rec},
                                       ensure_ascii=False) + "\n")
        out["per_item_path"] = os.path.abspath(args.per_item_out)
    else:
        # No aggregate without a trail: inline the per-item records.
        out["per_item"] = {k: v["per_item"] for k, v in arms.items()}

    blob = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(blob)
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
