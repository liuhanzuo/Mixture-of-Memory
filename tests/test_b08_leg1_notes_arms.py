"""B08 leg-1 blockers (2)(3)(4): the A-notes-only arm, the U scorer, self-notes.

WHY THESE TESTS LOOK LIKE THIS
------------------------------
``memory/selftest-over-invented-inputs-proves-nothing-about-the-pipeline.md``:
B04's ``--selftest`` passed every day while **no code path ever fed on-disk data
into the metric**. So every test here runs the *real* harness over a fixture
carved out of the *real* ``data/longmemeval/longmemeval_s.json`` — real field
names, real ``_abs`` ids, real ``answer_session_ids`` linkage — and the last
class feeds **deliberately plausible-but-broken** inputs to prove the code
fails LOUDLY instead of printing "found X" and "all missing" together.

Fixture: ``tests/fixtures/longmemeval_b08_stratum_fixture.json``, built by
``proposal/backlog/B08-memory-applications/code/build_b08_fixture.py``. 5 real
stratum records (3 ``knowledge-update`` incl. 1 ``_abs``, 2
``single-session-assistant``), haystacks trimmed to gold + distractors.

0 GPU: every test uses ``--reader stub`` / hand-built contexts. No model is ever
loaded, and ``SelfNotesCompressor`` is exercised through its cache and its
guard-rails, never through ``model.generate``.

Run (no pytest in any of this repo's interpreters -- this file is executable)::

    CUDA_VISIBLE_DEVICES="" python3 tests/test_b08_leg1_notes_arms.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from longmemeval import faithfulness as F                      # noqa: E402
from longmemeval.backends import Evidence                      # noqa: E402
from longmemeval.compressor import (                           # noqa: E402
    IdentityCompressor,
    SelfNotesCompressor,
    build_compressor,
)
from longmemeval.data import load_longmemeval                  # noqa: E402
from longmemeval.run_baseline import build_arg_parser, main as rb_main  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures",
                       "longmemeval_b08_stratum_fixture.json")
REAL_DATA = os.path.join(ROOT, "data", "longmemeval", "longmemeval_s.json")

# Every quantity below is a property of the REAL file, cross-checked against
# STATUS.json. If the fixture drifts, these fail.
FIXTURE_N = 5
FIXTURE_N_ABS = 1
FIXTURE_QIDS = ["6a1eabeb", "6aeb4375", "6aeb4375_abs", "7161e7e2", "c4f10528"]


# --------------------------------------------------------------------------- #
# tiny harness (no pytest in this repo's interpreters)
# --------------------------------------------------------------------------- #
_RESULTS = []


def test(fn):
    _RESULTS.append(fn)
    return fn


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def expect_raises(exc_types, fn, must_contain=None):
    """Assert fn() raises, and that the message SAYS WHAT IS WRONG.

    A raise with an unhelpful message is only half a fix: the whole point is
    that the next agent is told which field, which line, and why it matters.
    """
    try:
        fn()
    except exc_types as e:
        msg = str(e)
        if must_contain:
            for frag in ([must_contain] if isinstance(must_contain, str)
                         else must_contain):
                check(frag.lower() in msg.lower(),
                      f"raised {type(e).__name__} but message lacks {frag!r}: {msg}")
        return e
    except BaseException as e:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_types}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc_types}, but nothing was raised")


def _args(**over):
    """A real parsed argparse namespace (not a hand-made stand-in)."""
    argv = ["--data", FIXTURE, "--reader", "stub", "--retriever", "bm25"]
    for k, v in over.items():
        flag = "--" + k
        if v is True:
            argv.append(flag)
        elif v is False or v is None:
            continue
        else:
            argv += [flag, str(v)]
    return build_arg_parser().parse_args(argv)


class FakeReader:
    """A reader-shaped object: what SelfNotesCompressor needs to bind to.

    Deliberately NOT a torch model. It proves the compressor accepts the
    reader's OWN loaded weights (duck-typed on .model/.tokenizer) without any
    test loading an 8B checkpoint. Generation itself is never exercised here;
    that needs a card, and this task has zero GPU budget.
    """

    def __init__(self):
        self.model = object()
        self.tokenizer = object()


# --------------------------------------------------------------------------- #
# 0. The fixture really is real data
# --------------------------------------------------------------------------- #

@test
def test_fixture_is_carved_from_the_real_file():
    check(os.path.exists(FIXTURE), f"fixture missing: {FIXTURE}")
    recs = json.load(open(FIXTURE, encoding="utf-8"))
    check(len(recs) == FIXTURE_N, f"fixture n={len(recs)} != {FIXTURE_N}")
    check([r["question_id"] for r in recs] == FIXTURE_QIDS,
          "fixture question_ids drifted")
    n_abs = sum(1 for r in recs if r["question_id"].endswith("_abs"))
    check(n_abs == FIXTURE_N_ABS, f"_abs count {n_abs} != {FIXTURE_N_ABS}")
    for r in recs:
        check(r["question_type"] in ("knowledge-update",
                                     "single-session-assistant"),
              f"{r['question_id']}: off-stratum type {r['question_type']}")
        prov = r["_fixture_provenance"]
        check(prov["source"].endswith("longmemeval_s.json"), "bad provenance")
        check(prov["original_n_sessions"] > prov["kept_n_sessions"],
              "provenance claims no trim")
        # The gold session MUST survive the trim or retrieval can never hit and
        # every downstream number would be measuring the wrong thing.
        check(set(prov["gold_session_ids"]) <= set(r["haystack_session_ids"]),
              f"{r['question_id']}: gold session was trimmed away")
    # And the source file is on disk at the size STATUS.json records.
    if os.path.exists(REAL_DATA):
        check(os.path.getsize(REAL_DATA) == 278025796,
              "longmemeval_s.json size differs from the pre-registered one")


@test
def test_loader_parses_the_fixture_with_real_semantics():
    exs = load_longmemeval(FIXTURE)
    check(len(exs) == FIXTURE_N, "loader lost records")
    abs_items = [e for e in exs if e.is_abstention]
    check(len(abs_items) == FIXTURE_N_ABS, "is_abstention miscounted")
    # The _abs items DO carry a gold answer + gold sessions, so a naive
    # "no gold answer => abstention" heuristic would miss them (STATUS.json
    # NEW_BLOCKER_8 pre-commitment (a)).
    a = abs_items[0]
    check(a.answer.strip() != "", "_abs item has no gold answer string")
    check(len(a.answer_session_ids) >= 1, "_abs item has no gold sessions")


# --------------------------------------------------------------------------- #
# 1. Blocker (2): the three arms, one variable
# --------------------------------------------------------------------------- #

def _run_arm(tmp, name, compressor, mode, notes_cache=None, readonly=False):
    """Run the REAL CLI over the REAL fixture and return (report, contexts)."""
    ctx = os.path.join(tmp, name, "context.jsonl")
    out = os.path.join(tmp, name, "submission.jsonl")
    rep = os.path.join(tmp, name, "report.json")
    argv = ["--data", FIXTURE, "--reader", "stub", "--retriever", "bm25",
            "--top_k", "10", "--evidence_token_budget", "4000",
            "--compressor", compressor, "--reader_evidence_mode", mode,
            "--context_log", ctx, "--out", out, "--report", rep,
            "--question_types", "knowledge-update,single-session-assistant",
            "--expect_n", str(FIXTURE_N)]
    if notes_cache:
        argv += ["--notes_cache", notes_cache]
    if readonly:
        argv += ["--notes_cache_readonly"]
    rc = rb_main(argv)
    check(rc == 0, f"{name}: CLI rc={rc}")
    report = json.load(open(rep, encoding="utf-8"))
    contexts = {json.loads(l)["question_id"]: json.loads(l)
                for l in open(ctx, encoding="utf-8") if l.strip()}
    return report, contexts


@test
def test_three_arms_differ_ONLY_in_context_composition():
    """The single-variable claim, executed rather than asserted.

    ``memory/hand-composed-demo-strings-must-be-executed.md``: a demo that does
    not actually exhibit the mechanism it claims is worse than none. So this
    runs all three arms end-to-end on real records and compares the artifacts.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "notes.jsonl")
        # Pre-seed the notes cache so no GPU is needed: this is EXACTLY the
        # cross-arm mechanism the prereg demands (generate once, both notes
        # arms read the same strings), just with the generation step skipped.
        exs = load_longmemeval(FIXTURE)
        notes_by_qid = {}
        with open(cache, "w", encoding="utf-8") as f:
            for e in exs:
                note = (f"The user asked: {e.question} "
                        f"Relevant fact recorded on {e.question_date}.")
                notes_by_qid[e.question_id] = note
                f.write(json.dumps({"question_id": e.question_id,
                                    "notes": note,
                                    "source": "test_seed"}) + "\n")

        raw, raw_ctx = _run_arm(tmp, "A-raw", "none", "notes_plus_evidence")
        both, both_ctx = _run_arm(tmp, "A-notes+raw", "self_notes",
                                  "notes_plus_evidence", cache, readonly=True)
        only, only_ctx = _run_arm(tmp, "A-notes-only", "self_notes",
                                  "notes_only", cache, readonly=True)

        check(raw["arm"] == "A-raw", f"arm label {raw['arm']!r}")
        check(both["arm"] == "A-notes+raw", f"arm label {both['arm']!r}")
        check(only["arm"] == "A-notes-only", f"arm label {only['arm']!r}")

        for qid in raw_ctx:
            r = [b["text"] for b in raw_ctx[qid]["context_blocks"]]
            b = [b["text"] for b in both_ctx[qid]["context_blocks"]]
            o = [b["text"] for b in only_ctx[qid]["context_blocks"]]
            # (i) notes+raw == notes block PREPENDED to the byte-identical raw
            check(b[1:] == r, f"{qid}: notes+raw altered the raw evidence")
            check(len(b) == len(r) + 1, f"{qid}: notes block not prepended")
            # (ii) notes-only == the notes block ALONE. This is the arm that
            #      did not exist before: run_baseline.py:162 hardcoded
            #      [notes_block] + list(evidence) with no withhold path.
            check(len(o) == 1, f"{qid}: notes_only kept {len(o)} blocks")
            check(o[0] == b[0], f"{qid}: notes block differs between arms")
            # (iii) raw really is WITHHELD, not merely reordered
            for raw_block in r:
                check(raw_block not in o[0],
                      f"{qid}: raw evidence leaked into the notes-only context")
            # (iv) the notes TEXT is byte-identical across both notes arms
            check(notes_by_qid[qid] in o[0],
                  f"{qid}: notes-only lost the cached notes string")
            check(notes_by_qid[qid] in b[0],
                  f"{qid}: notes+raw lost the cached notes string")

        # (v) retrieval is frozen: identical recall in all three arms.
        check(raw["overall_recall"] == both["overall_recall"] ==
              only["overall_recall"],
              "retrieval differs across arms -- the single variable is violated")
        # (vi) the arm is recoverable from the artifact, not shell history.
        for rep in (raw, both, only):
            check("reader_evidence_mode" in rep and "arm" in rep,
                  "report does not record the arm")


@test
def test_default_behaviour_of_existing_arms_is_unchanged():
    """The pre-2026-08-16 default must be reproduced byte-for-byte.

    ``--compressor none`` + the default mode is what every prior B08 recall
    measurement ran; if this drifts, the archived
    ``b08_lme_bm25_recall_topk*.json`` numbers stop being comparable.
    """
    p = build_arg_parser()
    a = p.parse_args(["--data", FIXTURE])
    check(a.reader_evidence_mode == "notes_plus_evidence",
          "default reader_evidence_mode changed")
    check(a.compressor == "none", "default compressor changed")
    check(a.notes_cache is None and a.context_log is None,
          "new flags are not opt-in by default")
    check(a.question_types is None and a.expect_n is None,
          "stratum selector is not opt-in by default")
    # With --compressor none the notes branch is dead (notes == "") so the
    # composition flag cannot affect the baseline at all.
    check(IdentityCompressor().compress("q", "d", [
        Evidence("r", "s", "2024-01-01", "t", 1.0)]) == "",
        "IdentityCompressor stopped returning empty notes")


@test
def test_self_notes_label_is_not_MoM_in_the_prompt():
    """``Evidence.as_block`` renders ``session=`` INTO the prompt: the label is
    model input. A self-notes arm labelled ``MoM`` misdescribes its own
    provenance to the model that reads it."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "notes.jsonl")
        with open(cache, "w", encoding="utf-8") as f:
            for e in load_longmemeval(FIXTURE):
                f.write(json.dumps({"question_id": e.question_id,
                                    "notes": "N"}) + "\n")
        rep, ctx = _run_arm(tmp, "lbl", "self_notes", "notes_only",
                            cache, readonly=True)
        check(rep["notes_label"] == "SELF", f"label {rep['notes_label']!r}")
        blk = next(iter(ctx.values()))["context_blocks"][0]
        check(blk["session_id"] == "SELF-NOTES", f"session {blk['session_id']!r}")
        check(blk["text"].startswith("SELF NOTES:"), f"text {blk['text'][:40]!r}")
        check("MoM" not in blk["text"] and "MoM" not in blk["session_id"],
              "the MoM label leaked into a self-notes context")
        # And the block really is rendered into the reader's prompt.
        rendered = Evidence(blk["round_id"], blk["session_id"],
                            blk["session_date"], blk["text"], 1.0).as_block(1)
        check("session=SELF-NOTES" in rendered, "label not in the prompt")


@test
def test_stratum_selector_and_expect_n_assert_at_input_time():
    """``--limit`` cannot express the stratum (prefix vs suffix); the type
    filter can, and ``--expect_n`` makes a wrong cell fail before the reader."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.jsonl")
        base = ["--data", FIXTURE, "--reader", "stub", "--out", out]
        # right cell, right count -> runs
        check(rb_main(base + ["--question_types", "knowledge-update",
                              "--expect_n", "3"]) == 0, "KU cell rc != 0")
        # right cell, WRONG count -> SystemExit before any model work
        e = expect_raises(SystemExit, lambda: rb_main(
            base + ["--question_types", "knowledge-update", "--expect_n", "78"]),
            ["stratum size", "expect_n"])
        check("3" in str(e), f"error should report the actual size: {e}")
        # a type absent from the data is a typo, not an empty cell
        expect_raises(SystemExit, lambda: rb_main(
            base + ["--question_types", "knowlege-update"]), "absent from the data")


# --------------------------------------------------------------------------- #
# 2. Blocker (3): the U scorer
# --------------------------------------------------------------------------- #

@test
def test_U_is_computed_against_THAT_ARMS_OWN_context():
    """The definitional core: the same answer is unsupported in one arm and
    supported in the other, purely because the arms' contexts differ.

    Built from the fixture's REAL answers and REAL evidence text, not invented
    strings: the notes-only arm's context is a short note, so a detail that
    lives only in the raw evidence is ungrounded there and grounded in A-raw.
    """
    with tempfile.TemporaryDirectory() as tmp:
        exs = load_longmemeval(FIXTURE)
        raw_log = os.path.join(tmp, "raw.jsonl")
        only_log = os.path.join(tmp, "only.jsonl")
        with open(raw_log, "w", encoding="utf-8") as fr, \
             open(only_log, "w", encoding="utf-8") as fo:
            for e in exs:
                gold_text = "\n".join(
                    str(t.get("content", ""))
                    for sid, sess in zip(e.haystack_session_ids,
                                         e.haystack_sessions)
                    if sid in set(e.answer_session_ids)
                    for t in sess)
                hyp = e.answer                      # the REAL gold answer string
                for f, blocks in (
                        (fr, [{"round_id": f"{e.question_id}_r0",
                               "session_id": e.answer_session_ids[0],
                               "session_date": e.question_date,
                               "text": gold_text}]),
                        (fo, [{"round_id": f"{e.question_id}_self_notes",
                               "session_id": "SELF-NOTES",
                               "session_date": e.question_date,
                               "text": "SELF NOTES: the user discussed "
                                       "several topics."}])):
                    f.write(json.dumps({
                        "question_id": e.question_id,
                        "question_type": e.question_type,
                        "arm": "A-raw" if f is fr else "A-notes-only",
                        "hypothesis": hyp,
                        "context_blocks": blocks,
                    }, ensure_ascii=False) + "\n")

        raw = F.score_arm(F.load_context_log(raw_log),
                          expect_n=FIXTURE_N, expect_scored=FIXTURE_N - FIXTURE_N_ABS)
        only = F.score_arm(F.load_context_log(only_log),
                           expect_n=FIXTURE_N, expect_scored=FIXTURE_N - FIXTURE_N_ABS)

        # abstention items are EXCLUDED from U's denominator (prereg 5.2)
        check(raw["n_items"] == FIXTURE_N, "n_items wrong")
        check(raw["n_abstention_excluded"] == FIXTURE_N_ABS,
              f"excluded {raw['n_abstention_excluded']} _abs items")
        check(raw["n_scored"] == FIXTURE_N - FIXTURE_N_ABS,
              "U denominator includes abstention items")
        # the notes-only arm must have a HIGHER U on this construction
        check(only["U_pct"] > raw["U_pct"],
              f"U(notes-only)={only['U_pct']} !> U(raw)={raw['U_pct']} -- the "
              "per-arm support set is not being used")
        # per-item and per-claim trail exists for every scored item
        for arm in (raw, only):
            check(len(arm["per_item"]) == FIXTURE_N, "per-item trail incomplete")
            for it in arm["per_item"]:
                check("claims" in it and isinstance(it["claims"], list),
                      f"{it['question_id']}: no per-claim trail")
                for c in it["claims"]:
                    check({"claim", "supported", "ungrounded_salient", "rule"}
                          <= set(c), f"claim record missing fields: {sorted(c)}")
                # the aggregate must be recomputable from the trail
                check(it["unsupported"] ==
                      any(not c["supported"] for c in it["claims"]),
                      f"{it['question_id']}: aggregate != per-claim trail")

        d = F.delta_u(only, raw)
        check(d["n_pairs"] == FIXTURE_N - FIXTURE_N_ABS, "pairing lost items")
        check(d["delta_U_pp"] > 0, f"Delta_U={d['delta_U_pp']} should be > 0 here")
        check(d["ci95_pp"][0] <= d["delta_U_pp"] <= d["ci95_pp"][1],
              "point estimate outside its own CI")
        check("resolvable_at_n134" in d["k2_evaluability"],
              "K2 evaluability precondition not reported")
        check(abs(d["disc_U"] - F.discordance(F.pair_arms(only, raw))) < 1e-12,
              "disc_U is not the pair discordance")


@test
def test_support_set_is_EXACTLY_the_items_own_context():
    """The invariant behind "that arm's own context", checked exactly.

    WHY THIS EXISTS, AND IT IS NOT REDUNDANT: the aggregate inequality
    U(notes-only) > U(raw) in the test above SURVIVED a deliberately injected
    bug that widened the support set with tokens from outside the item's context
    (negative control 2, run this session). An inequality between two aggregates
    is a weak witness. This test instead recomputes the expected ungrounded set
    from the context alone and demands EQUALITY, so any leakage into the support
    set -- from another arm, from the gold answer, from a global vocabulary --
    fails here.
    """
    exs = load_longmemeval(FIXTURE)
    n_checked = 0
    for e in exs:
        gold_text = "\n".join(
            str(t.get("content", ""))
            for sid, sess in zip(e.haystack_session_ids, e.haystack_sessions)
            if sid in set(e.answer_session_ids) for t in sess)
        notes_text = "SELF NOTES: the user discussed several topics."
        for ctx in (gold_text, notes_text):
            ctx_tokens = set(F.tokens(ctx))
            res = F.score_answer(e.answer, ctx)
            for c in res["claims"]:
                if c["is_refusal"]:
                    continue
                expected = sorted(
                    {t for t in F.salient_tokens(c["claim"]) if t not in ctx_tokens})
                check(c["ungrounded_salient"] == expected,
                      f"{e.question_id}: support set is not exactly the item's "
                      f"own context.\n  claim: {c['claim'][:80]!r}\n"
                      f"  got:      {c['ungrounded_salient']}\n"
                      f"  expected: {expected}")
                n_checked += 1
    check(n_checked >= 8, f"only {n_checked} claims checked -- too weak")

    # And the directional consequence, per item rather than in aggregate: a
    # detail that lives ONLY in the raw evidence must be ungrounded in the
    # notes-only context.
    flipped = 0
    for e in exs:
        if e.is_abstention:
            continue
        gold_text = "\n".join(
            str(t.get("content", ""))
            for sid, sess in zip(e.haystack_session_ids, e.haystack_sessions)
            if sid in set(e.answer_session_ids) for t in sess)
        r = F.score_answer(e.answer, gold_text)
        o = F.score_answer(e.answer, "SELF NOTES: the user discussed several topics.")
        if (not r["unsupported"]) and o["unsupported"]:
            flipped += 1
    check(flipped >= 2,
          f"only {flipped} fixture items flip supported->unsupported when the "
          "raw evidence is withheld; the per-arm conditioning is not biting")


@test
def test_U_bootstrap_is_deterministic_and_numpy_free():
    pairs = [(0, 1)] * 7 + [(1, 0)] * 3 + [(0, 0)] * 20 + [(1, 1)] * 5
    a = F.paired_bootstrap_gap(pairs)
    b = F.paired_bootstrap_gap(pairs)
    check(a == b, "bootstrap is not deterministic across calls")
    check(a["seed"] == F.BOOTSTRAP_SEED == 42, "seed is not the pre-registered 42")
    check(a["iters"] == F.BOOTSTRAP_ITERS == 10000, "iters != 10000")
    check("numpy" not in sys.modules or True, "")   # not an error either way
    src = open(os.path.join(ROOT, "longmemeval", "faithfulness.py"),
              encoding="utf-8").read()
    check("import numpy" not in src,
          "faithfulness.py imports numpy: cross-node bootstrap would not "
          "reproduce (three numpy versions across the five nodes)")


@test
def test_refusals_and_abstention_are_different_things():
    """A refusal STRING asserts no claim; an abstention ITEM is out of the
    denominator. Conflating them corrupts U in opposite directions."""
    r = F.score_answer("I don't know.", "anything at all")
    check(not r["unsupported"], "a refusal was scored as an unsupported claim")
    check(r["claims"][0]["is_refusal"], "refusal not detected")
    # A refusal-shaped answer on a NON-abstention item still counts in the
    # denominator (it is scored, and it is supported).
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "c.jsonl")
        with open(log, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "q1", "hypothesis": "I don't know.",
                                "context_blocks": [{"text": "ctx"}]}) + "\n")
            f.write(json.dumps({"question_id": "q2_abs",
                                "hypothesis": "Zanzibar quokka telemetry.",
                                "context_blocks": [{"text": "ctx"}]}) + "\n")
        arm = F.score_arm(F.load_context_log(log))
        check(arm["n_scored"] == 1, f"denominator {arm['n_scored']} != 1")
        check(arm["n_unsupported_answers"] == 0,
              "the _abs item's fabrication leaked into U")
        check(arm["U_pct"] == 0.0, f"U={arm['U_pct']}")


@test
def test_U_sensitivity_sweep_is_emitted_by_the_CLI():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "c.jsonl")
        with open(log, "w", encoding="utf-8") as f:
            for i in range(6):
                f.write(json.dumps({
                    "question_id": f"q{i}",
                    "question_type": "knowledge-update",
                    "hypothesis": "Alpha bravo charlie delta echo foxtrot.",
                    "context_blocks": [{"text": "alpha bravo charlie"}],
                }) + "\n")
        out = os.path.join(tmp, "U.json")
        rc = F.main(["--arm", f"A-raw:{log}", "--out", out])
        check(rc == 0, f"faithfulness CLI rc={rc}")
        obj = json.load(open(out, encoding="utf-8"))
        sw = obj["sensitivity_min_ungrounded_salient"]["A-raw"]
        check(set(sw) == {"1", "2", "3"}, f"sweep levels {sorted(sw)}")
        # a stricter threshold can only lower or hold U
        check(sw["1"]["U_pct"] >= sw["2"]["U_pct"] >= sw["3"]["U_pct"],
              f"U not monotone in the threshold: {sw}")
        check("per_item" in obj, "aggregate written with no per-item trail")
        check(obj["prior_art_ack"].count("ALCE") == 1,
              "the ALCE/FActScore/SummaC acknowledgement is missing")


# --------------------------------------------------------------------------- #
# 3. Blocker (4): SelfNotesCompressor, without a card
# --------------------------------------------------------------------------- #

@test
def test_self_notes_binds_to_the_readers_own_weights():
    reader = FakeReader()
    c = build_compressor("self_notes", _args(), reader=reader)
    check(isinstance(c, SelfNotesCompressor), "wrong compressor built")
    check(c.model is reader.model, "compressor did not share reader.model")
    check(c.tokenizer is reader.tokenizer, "compressor did not share tokenizer")
    check(c.label == "SELF", "self-notes label is not SELF")
    # No reader at all => refuse, citing the prereg constraint.
    expect_raises(ValueError, lambda: build_compressor("self_notes", _args()),
                  "reader")
    # A reader with no weights (stub/openai) may NOT generate: that is the
    # notes_generator_must_be_the_reader_itself constraint. It must NOT quietly
    # load a second model either.
    expect_raises(ValueError,
                  lambda: SelfNotesCompressor(reader=object()),
                  [".model", "generate", "prereg"])
    # ...but replaying an EXISTING cache read-only needs no weights: the guard
    # protects generation, and requiring 8B of weights to read a JSONL would
    # make the notes arms unscoreable on a CPU node.
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "n.jsonl")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "q", "notes": "n"}) + "\n")
        ro = SelfNotesCompressor(reader=object(), notes_cache_path=cache,
                                 allow_generate=False)
        check(ro.compress("q", "d", [], question_id="q") == "n",
              "read-only replay failed")
    # readonly with NO cache = no source of notes at all: refuse rather than
    # silently degrade the arm to the raw baseline.
    expect_raises(ValueError, lambda: SelfNotesCompressor(
        reader=object(), allow_generate=False), "no source of notes")


@test
def test_self_notes_uses_the_MoM_instruction_verbatim():
    from longmemeval.compressor import _NOTES_INSTRUCTION
    # Byte-identical to the string MoMNotesCompressor inlined before this change.
    expected = ("\n\nSummarize the facts from the conversation above that are "
                "relevant to answering this question: {question}\n"
                "Relevant facts:")
    check(_NOTES_INSTRUCTION == expected,
          "the notes instruction drifted -- the notes PROMPT would become a "
          "second uncontrolled variable when the generator changes")


@test
def test_notes_cache_freezes_the_notes_text_across_arms():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "n.jsonl")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "6a1eabeb",
                                "notes": "cached note text"}) + "\n")
        c = SelfNotesCompressor(reader=FakeReader(), notes_cache_path=cache,
                                allow_generate=False)
        got = c.compress("q", "2024-01-01",
                         [Evidence("r", "s", "2024-01-01", "t", 1.0)],
                         question_id="6a1eabeb")
        check(got == "cached note text", f"cache miss: {got!r}")
        check(c.stats["notes_cache_hits"] == 1, "cache hit not counted")
        check(c.stats["notes_generated"] == 0, "generated despite a cache hit")
        # A cache MISS under readonly must RAISE, not silently regenerate:
        # silent regeneration would unfreeze the frozen single variable and
        # leave no trace in any artifact.
        expect_raises(RuntimeError, lambda: c.compress(
            "q", "2024-01-01", [Evidence("r", "s", "d", "t", 1.0)],
            question_id="NOT_IN_CACHE"),
            ["not in the notes cache", "unfreeze"])


# --------------------------------------------------------------------------- #
# 4. DELIBERATELY BROKEN INPUTS -- the point of this file
# --------------------------------------------------------------------------- #

@test
def test_broken_int_question_id_fails_loudly_not_silently():
    """THE B04 FAILURE MODE, reproduced on purpose.

    An ``int`` question_id looks fine in the file, parses fine as JSON, and then
    misses every str-keyed lookup — so a tolerant loader prints "found 5" and a
    metric computed over 0 rows. It must raise, and the message must name the
    field and the type.
    """
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "c.jsonl")
        with open(log, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "6a1eabeb", "hypothesis": "a",
                                "context_blocks": [{"text": "c"}]}) + "\n")
            f.write(json.dumps({"question_id": 6041, "hypothesis": "b",
                                "context_blocks": [{"text": "c"}]}) + "\n")
        e = expect_raises(TypeError, lambda: F.load_context_log(log),
                          ["question_id", "must be a str", "int"])
        check(":2:" in str(e), f"error does not name the offending line: {e}")

    # Same class in the notes cache: a str/int mismatch there would make one arm
    # regenerate while the other reused the persisted string.
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "n.jsonl")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": 6041, "notes": "x"}) + "\n")
        expect_raises(ValueError, lambda: SelfNotesCompressor(
            reader=FakeReader(), notes_cache_path=cache),
            ["question_id must be a str", "regenerate"])


@test
def test_missing_evidence_field_fails_loudly():
    """A context record with no ``context_blocks``, and a block with no
    ``text``. Both are 'plausible but broken': U is *defined* against the
    context, so scoring such a row would silently measure nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "c.jsonl")
        with open(log, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "q1",
                                "hypothesis": "an answer"}) + "\n")
        expect_raises(KeyError, lambda: F.load_context_log(log),
                      ["context_blocks", "partial log"])

        log2 = os.path.join(tmp, "c2.jsonl")
        with open(log2, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "question_id": "q1", "hypothesis": "a",
                "context_blocks": [{"round_id": "r", "session_id": "s"}],
            }) + "\n")
        recs = F.load_context_log(log2)
        expect_raises(KeyError, lambda: F.score_arm(recs),
                      ["no 'text' key", "protocol failure"])

        # A notes-cache record missing 'notes' is the same class.
        cache = os.path.join(tmp, "n.jsonl")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": "q1", "note": "typo key"}) + "\n")
        expect_raises(ValueError, lambda: SelfNotesCompressor(
            reader=FakeReader(), notes_cache_path=cache),
            ["missing", "protocol failure"])


@test
def test_partial_arm_and_unpaired_arms_are_refused():
    """No metric on a partial arm; no bootstrap on an unpaired one.

    Precedent: a silent 5/8-shard merge corrupted a whole protocol. The fix is
    an ASSERT on the count, not a NaN check.
    """
    with tempfile.TemporaryDirectory() as tmp:
        def write(path, qids):
            with open(path, "w", encoding="utf-8") as f:
                for q in qids:
                    f.write(json.dumps({
                        "question_id": q, "question_type": "knowledge-update",
                        "hypothesis": "alpha bravo",
                        "context_blocks": [{"text": "alpha bravo"}]}) + "\n")

        full = os.path.join(tmp, "full.jsonl")
        short = os.path.join(tmp, "short.jsonl")
        write(full, [f"q{i}" for i in range(5)])
        write(short, [f"q{i}" for i in range(4)])

        # partial arm vs the pre-registered cell size
        expect_raises(ValueError,
                      lambda: F.score_arm(F.load_context_log(short), expect_n=5),
                      ["4 items", "expect_n 5", "partial arm"])
        # wrong U denominator
        expect_raises(ValueError,
                      lambda: F.score_arm(F.load_context_log(full),
                                          expect_scored=128),
                      ["non-abstention", "expect_scored"])
        # unpaired arms
        a = F.score_arm(F.load_context_log(full))
        b = F.score_arm(F.load_context_log(short))
        expect_raises(ValueError, lambda: F.pair_arms(b, a),
                      ["not paired", "only in"])
        # duplicate ids would double-count the denominator
        dup = os.path.join(tmp, "dup.jsonl")
        write(dup, ["q1", "q1"])
        expect_raises(ValueError, lambda: F.load_context_log(dup),
                      ["duplicate question_id", "double-counted"])


@test
def test_broken_fixture_record_is_rejected_by_the_harness():
    """Feed the REAL harness a fixture whose ``answer_session_ids`` points at a
    session that is not in the haystack — the exact shape of a bad trim. The
    recall bookkeeping must show the miss instead of reporting a free 1.0."""
    with tempfile.TemporaryDirectory() as tmp:
        recs = json.load(open(FIXTURE, encoding="utf-8"))
        bad = json.loads(json.dumps(recs[0]))
        bad["question_id"] = "broken_gold"
        bad["answer_session_ids"] = ["session_that_does_not_exist"]
        path = os.path.join(tmp, "broken.json")
        json.dump([bad], open(path, "w", encoding="utf-8"))
        rep = os.path.join(tmp, "r.json")
        rc = rb_main(["--data", path, "--reader", "stub", "--top_k", "10",
                      "--out", os.path.join(tmp, "s.jsonl"), "--report", rep])
        check(rc == 0, f"rc={rc}")
        r = json.load(open(rep, encoding="utf-8"))
        check(r["overall_recall"]["any_hit_recall"] == 0.0,
              f"a nonexistent gold session scored a hit: {r['overall_recall']}")
        check(r["overall_recall"]["n"] == 1, "the item was silently dropped")
        # And the free-recall branch is NOT what fired: n_gold > 0 here.
        # (scoring.py:64-66 returns 1.0 only for an EMPTY gold list.)


# --------------------------------------------------------------------------- #

def run():
    passed, failed = [], []
    for fn in _RESULTS:
        try:
            fn()
            passed.append(fn.__name__)
            print(f"PASS {fn.__name__}")
        except BaseException:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(passed)}/{len(_RESULTS)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
