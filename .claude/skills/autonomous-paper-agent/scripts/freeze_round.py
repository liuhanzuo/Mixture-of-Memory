#!/usr/bin/env python3
"""freeze_round.py — freeze an immutable, hashed blind-review snapshot.

Why not just use the two scripts that already exist
--------------------------------------------------
`scripts/freeze_paper_version.py` is the better of the two: it follows `\\input`
and `\\includegraphics` to build a real dependency closure. But it **hardcodes
the venue** at lines 32-33 -- `acl.sty` / `acl_natbib.bst`, plus a bib whitelist
of only `qcmem.bib` / `paperB.bib`. Measured: paperA now uses
`colm2026_conference.sty` and paperC uses `iclr2026_conference.sty` + `refs.bib`,
so pointing that script at paperC raises FileNotFoundError on a file that is not
supposed to be there. It also names snapshots `vN_source_<stamp>`, which the
upstream `select_best_round.py` cannot see (it matches `round_(\\d+)$`).

Upstream `make_review_snapshot.py` has the opposite problem: it takes an explicit
`--include` list and a prebuilt PDF, so it cannot discover the dependency closure
and will happily freeze a snapshot that omits a section.

This script keeps the good half of each: **discover the closure** (from
freeze_paper_version) and **emit `round_NN/` with a hashed MANIFEST** (from
make_review_snapshot), with the venue style discovered rather than assumed.

Interface: default-include, explicit-exclude (2026-08-16 rewrite)
----------------------------------------------------------------
The v1 interface took `--evidence` as a repeated **whitelist**. It was faithful to
whatever it was handed, and that was the whole problem: a human enumerating ~24
paths by hand will miss some. Measured on paperC round_04 -- MAIN passed two
`--evidence` flags, 23 of the 25 artifact records on disk were silently omitted,
`missing_dependencies` still reported `[]`, and four of six blind reviewers
independently downgraded the paper because "the frozen artifact does not contain
the evidence it repeatedly claims to publish". No number in the paper was wrong.
The packaging step was.

So the polarity is now inverted. `<paper>/evidence/`, `<paper>/gate/` and
`<paper>/code/` are packed **recursively by default**; `--exclude GLOB` removes
things; `--no-default-artifacts` restores v1 behaviour. `--evidence` still works
and is still additive, so existing call sites do not break -- they just stop
being the only thing that gets in.

Three gates, all fatal
----------------------
1. **Closure gate.** Every `\\input`/`\\include`/`\\includegraphics`/
   `\\lstinputlisting`/... target must resolve. (v1 had this.)
2. **Named-artifact gate.** Every artifact path the *prose* names -- via the
   `tab:artifact-map` identifier table, or any `\\texttt{...}` that looks like a
   repo-relative artifact path -- must be present in the snapshot. This is the
   gate that would have caught round_04: v1's `missing_dependencies` only
   followed the LaTeX `\\input` chain, so a paper could name `\\textsf{E-CAL}
   -> evidence/floor_winners_curse_calibration.json` in a caption, ship without
   it, and still report zero missing dependencies. Confirmed by reading
   round_04's own MANIFEST: `missing_dependencies: []`, `n_files: 34`, two
   evidence records.
3. **Blindness gate.** No prior-round material may enter, by path *or by
   content*, and no internal path / node IP / affiliation may enter. v1 checked
   paths only, and only on the closure -- the `--evidence` loop bypassed
   `BLIND_EXCLUDE` entirely. Measured consequence: rounds 00, 01 and 02 each
   shipped `tcodex_out/EVIDENCE_PACK.md` to blind reviewers (md5
   4d0013b52eedc06b48c3b930a76ba014, byte-identical to the author-side file),
   which contains author-facing steering such as "DO NOT claim differential
   learning rates", "attacking it would be an easy referee kill", and "the
   corresponding limitation L1 in the writer prompt". That is a blindness breach,
   not a presentation defect, and v1's own docstring promised it could not happen.

Sanitisation, not rewriting
---------------------------
Internal absolute paths (`/apdcephfs_*/share_*/pighzliu_code/...`), node IPs and
internal hostnames are redacted in the **snapshot copy only**. Source files are
never modified -- that would destroy provenance. Each redacted file records
`sanitized: true`, the rules applied, the substitution count, and the
`source_sha256` of the untouched original, so any reviewer or auditor can verify
the redaction was mechanical.

Usage:
  python freeze_round.py paperC --round 5                 # packs everything
  python freeze_round.py paperC --round 5 --exclude 'gate/venue_*'
  python freeze_round.py paperC --round 5 --dest submission_complete
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# Paths that must never enter a blind snapshot.
BLIND_EXCLUDE = ("review_rounds", "review_history", "tcodex_out",
                 "SCORE_HISTORY", "review_prompts", "WRITER_NOTES")

# Artifact roots packed recursively by default, mapped to snapshot subpaths that
# mirror the paper's own naming (the prose says "relative to paperC/").
DEFAULT_ARTIFACT_ROOTS = (("evidence", "evidence"),
                          ("gate", "evidence/gate"),
                          ("code", "code"))

# Deliberately empty. An earlier draft of this rewrite hand-listed the two gate
# records that discuss the review process, which would have repeated the very
# mistake being fixed: a hand-maintained list of instances instead of a rule for
# the class. Records that disclose the review process are now found mechanically
# by BLIND_CONTENT_FATAL and quarantined, so the list maintains itself.
DEFAULT_EXCLUDE: tuple[str, ...] = ()

# Files never worth shipping and never safe to ship.
HARD_EXCLUDE_GLOBS = ("*.pt", "*.bin", "*.npy", "*.safetensors", "*.ckpt",
                      "*.pyc", "__pycache__/*", "*/__pycache__/*",
                      "password*", "*/password*", ".DS_Store")

# Redaction rules, applied in order (longest / most specific first).
# The workspace-name and hostname rules exist because these tokens also occur
# OUTSIDE an /apdcephfs path: `pighzliu_code/venv_union9` in a code comment, and
# `"node_hostname": "TENCENT64.site"` in an evidence record. Redacting only the
# absolute-path form would leave both.
SANITIZE_RULES = (
    (r"/apdcephfs_[A-Za-z0-9]+/share_\d+/pighzliu_code/Mixture-of-Memory",
     "<REPO_ROOT>"),
    (r"/apdcephfs_[A-Za-z0-9]+/share_\d+/pighzliu_code", "<WORKSPACE>"),
    (r"/apdcephfs_[A-Za-z0-9_]+", "<INTERNAL_MOUNT>"),
    (r"\b(?:28\.89\.19\.21|28\.89\.18\.212|28\.85\.35\.73|28\.82\.250\.82|"
     r"28\.83\.24\.104)\b(?::\d+)?", "<NODE>"),
    (r"\b[A-Za-z0-9][A-Za-z0-9.-]*\.(?:woa|oa)\.com(?::\d+)?",
     "<INTERNAL_HOST>"),
    # hostname form, e.g. TENCENT64.site -- must precede the bare-org rule
    (r"(?i)\btencent[A-Za-z0-9_-]*\.(?:site|com|net|local)\b",
     "<INTERNAL_HOST>"),
    (r"(?i)\bpighzliu[A-Za-z0-9_-]*", "<WORKSPACE_OWNER>"),
    (r"(?i)\btencent[A-Za-z0-9_-]*", "<INTERNAL_ORG>"),
    (r"(?i)\b(?:lhz24|liuhanzuo)[A-Za-z0-9_-]*", "<AUTHOR>"),
    (r"(?i)\btsinghua[A-Za-z0-9_-]*", "<AFFILIATION>"),
)

# Post-pack audit. Two tiers, because "the string tcodex_out appears" and "a
# reviewer can read the writer's steering notes" are not the same event.
#
# FATAL: content that tells a blind reviewer there were prior rounds, or that
# hands them author-side steering. Any hit means the file must not ship. This is
# the tier that would have caught the rounds 00-02 EVIDENCE_PACK.md leak: that
# file contains "DO NOT claim", "MUST NOT claim", "DO NOT describe".
#
# Every pattern below was measured against the whole paperC tree before adoption,
# because an over-broad blindness pattern is not a safe default -- it quarantines
# the evidence the reviewers asked for, which is the defect being fixed. Measured
# false positives on the manuscript + real artifacts: 0. Discriminators that
# turned out to matter:
#   * case. `DO NOT claim` / `Do not write` is a writer instruction;
#     `we do not claim the interface contrast` is the paper's own prose and
#     appears twice in 02_related.tex. Matching case-insensitively flags both.
#   * `double-blind`. `Paper under double-blind review` is in main.tex and in
#     iclr2026_conference.sty -- it is ICLR's own anonymity boilerplate, i.e. the
#     opposite of a leak. Only unqualified `blind reviewers` is a tell.
#   * `rebuttal` alone is ICLR page-policy vocabulary ("raised to 10 pages for
#     the discussion/rebuttal phase"), so it is not a pattern on its own; the
#     files that mention it are caught by their round references instead.
#   * `five of six` is the paper's own result vocabulary (five of six OLMo-2
#     arms), not a panel count, so reviewer-count patterns must be narrower.
BLIND_CONTENT_FATAL = (
    r"SCORE_HISTORY", r"WRITER_NOTES", r"review_prompts",
    r"(?i)meta-?review", r"(?i)\btarget (?:score|median)\b",
    r"(?i)referee kill", r"(?i)\bwriter prompt\b",
    # writer-facing imperative: case-sensitive on purpose (see above)
    r"\b(?:DO NOT|Do not|MUST NOT|must NOT|Never)\s+"
    r"(?:claim|say|write|describe|report|present|state)\b",
    # any round reference near review vocabulary, plus bare `round_NN` idents
    r"\bround[_ -]?0?\d+\b(?=.{0,60}(?i:review|reviewer|submission|snapshot|"
    r"audit|pre-review))",
    r"round_\d",
    r"(?i)\b(?:pre-review|prior[- ]round|previous round|earlier round)\b",
    # panel / reviewer identifiers
    r"\b[XR]\d/[XR]\d\b", r"\breviewers? \([XR]\d",
    r"\b\d+ of six reviewers\b", r"\bsix reviewers\b",
    r"(?<!double-)\bblind reviewers?\b",
)
# WARN: a bare mention of an excluded directory NAME, with no prior-round or
# steering content. paperC's own evidence/internal_paths_in_submission.json
# already adjudicated this class: "NOT an anonymity breach and NOT a blindness
# breach ... a provenance-presentation defect". Recorded, not fatal -- making it
# fatal would withhold the record that documents the defect.
BLIND_CONTENT_WARN = (r"review_rounds", r"review_history", r"tcodex_out")

# Leak detectors. NOTE the deliberate absence of a TRAILING `\b` on the identifier
# tokens. `\bpighzliu\b` does NOT match `pighzliu_code`, because `u`->`_` is not a
# word boundary -- both are word characters. Measured: that one missing case let
# `pighzliu_code/venv_union9` through in code/construct_nulls_length_unit.py, and
# a bounded-vs-bare sweep over the whole paperC tree found the same flaw would
# have missed 22 files for `pighzliu` and 1 for `tencent`. A leading `\b` is
# fine (it prevents matching inside a longer word); a trailing one is a bug for
# any token that can be followed by `_` or a digit.
LEAK_CONTENT_PATTERNS = (
    r"/apdcephfs_", r"(?i)\bpighzliu", r"(?i)\btencent", r"(?i)\blhz24",
    r"(?i)\bliuhanzuo", r"(?i)\btsinghua",
    r"\b28\.(?:89|85|82|83)\.\d{1,3}\.\d{1,3}\b", r"\.(?:woa|oa)\.com",
)
# Extensions worth reading as text during the audit.
TEXT_EXT = {".tex", ".bib", ".sty", ".bst", ".cls", ".clo", ".json", ".tsv",
            ".csv", ".md", ".txt", ".py", ".sh", ".bbl", ".aux", ".log"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def is_blind_path(rel: str) -> bool:
    return any(x in rel for x in BLIND_EXCLUDE)


def closure(main: Path) -> tuple[set[Path], list[str]]:
    """Dependency closure of a LaTeX main file. Returns (files, missing)."""
    root = main.parent
    files: set[Path] = set()
    missing: list[str] = []
    stack = [main]
    while stack:
        cur = stack.pop()
        if cur in files or not cur.is_file():
            continue
        files.add(cur)
        text = re.sub(r"(?<!\\)%.*", "",
                      cur.read_text(encoding="utf-8", errors="replace"))
        # Commands that pull in another .tex and must be recursed into.
        for m in re.finditer(
                r"\\(?:input|include|subfile|subfileinclude|import|includestandalone)"
                r"(?:\[[^\]]*\])?\{([^}]+)\}", text):
            t = m.group(1).strip()
            cand = root / t
            if cand.suffix != ".tex":
                cand = root / (t + ".tex")
            (stack.append(cand) if cand.is_file() else missing.append(t))
        # Commands that pull in a verbatim / graphic asset (no recursion).
        for m in re.finditer(
                r"\\(?:includegraphics|lstinputlisting|verbatiminput|"
                r"includepdf|inputminted)(?:\[[^\]]*\])?"
                r"(?:\{[^}]*\})?\{([^}]+)\}", text):
            t = m.group(1).strip()
            for ext in ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".tex",
                        ".txt", ".py"):
                c = root / (t + ext)
                if c.is_file():
                    files.add(c)
                    break
            else:
                missing.append(t)
        # Bibliography databases, wherever they live (not just the root glob).
        for m in re.finditer(r"\\(?:bibliography|addbibresource)\{([^}]+)\}",
                             text):
            for t in m.group(1).split(","):
                t = t.strip()
                if not t:
                    continue
                cand = root / t
                if cand.suffix != ".bib":
                    cand = root / (t + ".bib")
                (files.add(cand) if cand.is_file() else missing.append(t))
    # venue style + bib + class helpers + prebuilt bbl, DISCOVERED not assumed.
    # The .bbl matters: without it a reviewer who recompiles gets no references.
    for pat in ("*.sty", "*.bst", "*.bib", "*.cls", "*.clo"):
        files.update(root.glob(pat))
    bbl = main.with_suffix(".bbl")
    if bbl.is_file():
        files.add(bbl)
    return files, missing


def unescape_tex(s: str) -> str:
    """`evidence/foo\\_bar.json` -> `evidence/foo_bar.json`."""
    return (s.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")
             .replace("\\#", "#").replace("\\{", "{").replace("\\}", "}")
             .replace("~", "").strip())


ARTIFACT_PATH_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_./+-]*"
    r"(?:\.(?:json|tsv|csv|md|py|txt|jsonl|yaml|yml|sh|bib|tex)|/)$")


def named_artifacts(tex_files: set[Path], paper_dir: Path) -> dict[str, list[str]]:
    """Artifact paths the PROSE names, keyed by the citing source file.

    Two sources, because the paper uses two conventions:

    * the `tab:artifact-map` table, whose rows resolve an identifier
      (`\\textsf{E-CAL}`, `Emitter`, ...) to one or more `\\texttt{path}` cells.
      Only rows whose identifier is actually cited elsewhere -- or whose
      identifier is not an `E-` label at all, e.g. `Emitter` -- are required.
    * any other `\\texttt{...}` in the body that looks like a repo-relative
      artifact path.

    Paths are relative to the paper directory, per the appendix's own statement
    ("Paths are relative to the repository's paperC/ directory"). A leading
    `paperC/` is therefore stripped when present.
    """
    corpus = {}
    for f in sorted(tex_files):
        if f.suffix != ".tex":
            continue
        corpus[f] = re.sub(r"(?<!\\)%.*", "",
                           f.read_text(encoding="utf-8", errors="replace"))
    joined = "\n".join(corpus.values())
    cited_labels = set(re.findall(r"\\textsf\{(E-[A-Za-z0-9-]+)\}", joined))

    out: dict[str, list[str]] = {}
    paper_name = paper_dir.name

    def add(rel: str, src: Path) -> None:
        rel = unescape_tex(rel)
        for pfx in (f"{paper_name}/", "./"):
            if rel.startswith(pfx):
                rel = rel[len(pfx):]
        if not rel or not ARTIFACT_PATH_RE.match(rel):
            return
        # A bare paper-dir reference or a style/section file is not an artifact.
        if rel in ("", "/") or rel.startswith("sections/"):
            return
        out.setdefault(rel, [])
        if str(src) not in out[rel]:
            out[rel].append(str(src))

    for f, text in corpus.items():
        # 1. artifact-map style rows: "IDENT & cell-with-texttt \\"
        for row in re.findall(r"^\s*(\\textsf\{E-[A-Za-z0-9-]+\}|[A-Z][A-Za-z ]*?)"
                              r"\s*&\s*(.+?)\\\\\s*$", text, re.M):
            ident, cell = row
            lab = re.match(r"\\textsf\{(E-[A-Za-z0-9-]+)\}", ident)
            if lab and lab.group(1) not in cited_labels:
                continue  # row exists but no caption cites it
            for t in re.findall(r"\\texttt\{([^}]*)\}", cell):
                add(t, f)
        # 2. any other artifact-looking \texttt path in the body
        for t in re.findall(r"\\texttt\{([^}]*)\}", text):
            add(t, f)
    return out


def sanitize(data: bytes) -> tuple[bytes, list[str], int]:
    """Redact internal identifiers. Returns (new_bytes, rules_applied, n_subs)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, [], 0
    applied: list[str] = []
    total = 0
    for pat, repl in SANITIZE_RULES:
        text, n = re.subn(pat, repl, text)
        if n:
            applied.append(f"{pat} -> {repl}")
            total += n
    return text.encode("utf-8"), applied, total


def steering_spans(text: str) -> list[tuple[int, int, str]]:
    """Spans of author-side steering that could be redacted rather than withheld.

    Only the writer-imperative class qualifies. A round reference or a reviewer
    identifier cannot be redacted safely -- the surrounding sentence would still
    describe the review process -- so those files are always withheld whole.

    Scope is ONE SENTENCE, deliberately. An earlier draft matched the enclosing
    parenthetical, which on the real input swallowed a substantive finding ("The
    other 5 cells are below chance too, so the chance line does not mislead
    there") along with the instruction. Redacting evidence to hide a writer note
    is the wrong trade. Sentences may wrap across lines (the real hit does), so
    newlines are allowed inside a sentence but a blank line ends the search.
    """
    imperative = (r"\b(?:DO NOT|Do not|MUST NOT|must NOT|Never)\s+"
                  r"(?:claim|say|write|describe|report|present|state)\b")
    body = r"(?:(?!\n\s*\n)[^.])"
    pat = (rf"(?:(?<=\.)|(?<=\n)|^)\s*{body}{{0,300}}?{imperative}"
           rf"{body}{{0,300}}?\.")
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(pat, text)]


def redact_steering(data: bytes) -> tuple[bytes, int, list[str]]:
    """Replace author-side steering spans with a disclosed placeholder."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, 0, []
    spans = steering_spans(text)
    if not spans:
        return data, 0, []
    out, prev, removed = [], 0, []
    for s, e, hit in spans:
        out.append(text[prev:s])
        out.append("[AUTHOR-SIDE EDITORIAL NOTE REDACTED FOR BLIND REVIEW]")
        removed.append(hit.strip()[:200])
        prev = e
    out.append(text[prev:])
    return "".join(out).encode("utf-8"), len(spans), removed


def screen_fatal_bytes(data: bytes) -> list[dict]:
    """Fatal-tier hits in a byte buffer (used to check a redaction succeeded)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    hits = []
    for pat in BLIND_CONTENT_FATAL:
        m = re.search(pat, text)
        if m:
            hits.append({"pattern": pat,
                         "line": text.count("\n", 0, m.start()) + 1,
                         "match": m.group(0)[:80]})
    return hits


def screen_fatal(src: Path) -> list[dict]:
    """Pre-pack screen: does this file disclose the review process / steering?

    Run BEFORE copying, so an offending record is quarantined (recorded in
    `excluded_files`) rather than packed and then reported. A post-pack audit
    still runs as a backstop, so a screen miss is a hard failure, not a silent
    ship.
    """
    if src.suffix.lower() not in TEXT_EXT:
        return []
    return screen_fatal_bytes(src.read_bytes())


def audit(dest: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Post-pack backstop. Returns (fatal_blind, warn_blind, leaks)."""
    fatal, warn, leak = [], [], []
    for f in sorted(dest.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(dest))
        if rel == "MANIFEST.json":
            continue  # the manifest legitimately names the exclusion rules
        if is_blind_path(rel):
            fatal.append({"file": rel, "kind": "path",
                          "pattern": next(x for x in BLIND_EXCLUDE if x in rel)})
        if f.suffix.lower() not in TEXT_EXT:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for pat, sink, kind in ([(p, fatal, "content-fatal")
                                 for p in BLIND_CONTENT_FATAL]
                                + [(p, warn, "content-warn")
                                   for p in BLIND_CONTENT_WARN]
                                + [(p, leak, "leak")
                                   for p in LEAK_CONTENT_PATTERNS]):
            m = re.search(pat, text)
            if m:
                sink.append({"file": rel, "kind": kind, "pattern": pat,
                             "line": text.count("\n", 0, m.start()) + 1,
                             "match": m.group(0)[:80]})
    return fatal, warn, leak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--main", default="main.tex")
    ap.add_argument("--dest", default="submission",
                    help="subdirectory under round_NN/ (default: submission)")
    ap.add_argument("--evidence", action="append", default=[],
                    help="EXTRA reviewer-safe artifact to include; additive to "
                         "the default roots. Repeat.")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob (relative to the paper dir) to withhold; repeat")
    ap.add_argument("--no-default-artifacts", action="store_true",
                    help="do not pack evidence/ gate/ code/ recursively "
                         "(restores the pre-2026-08-16 whitelist behaviour)")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="do not redact internal paths/IPs (audit still fatal)")
    ap.add_argument("--allow-dangling-blind-refs", action="store_true",
                    help="downgrade 'prose names a blind-excluded artifact' from "
                         "fatal to recorded warning")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    pd = Path(a.paper) if Path(a.paper).is_absolute() else REPO / a.paper
    main_tex = pd / a.main
    if not main_tex.is_file():
        print(f"error: {main_tex} not found", file=sys.stderr)
        return 2

    rd = pd / "review_rounds" / f"round_{a.round:02d}"
    dest = rd / a.dest
    if dest.exists():
        if not a.force:
            print(f"error: {dest} exists; use --force", file=sys.stderr)
            return 2
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    user_exclude = tuple(a.exclude)
    all_exclude = DEFAULT_EXCLUDE + user_exclude
    excluded: list[dict] = []

    def excluded_reason(rel: str, src: Path | None = None) -> str | None:
        for g in HARD_EXCLUDE_GLOBS:
            if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(Path(rel).name, g):
                return f"hard-excluded glob {g!r} (weights/caches/secrets)"
        for g in DEFAULT_EXCLUDE:
            if fnmatch.fnmatch(rel, g):
                return f"default exclude {g!r}"
        for g in user_exclude:
            if fnmatch.fnmatch(rel, g):
                return f"operator --exclude {g!r}"
        if src is not None and (hits := screen_fatal(src)):
            return ("blindness screen: content discloses the review process or "
                    "carries author-side steering ("
                    + "; ".join(f"{h['pattern']} @L{h['line']}" for h in hits[:3])
                    + ") -- withheld whole rather than redacted, because "
                      "redacting it would destroy the record's meaning")
        return None

    records: list[dict] = []

    def emit(src: Path, snap_rel: str, redact_notes: bool = False) -> None:
        """Copy src to dest/snap_rel, sanitising, and record it."""
        tgt = dest / snap_rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_bytes()
        src_hash = sha256_bytes(raw)
        rec = {"snapshot_path": snap_rel, "source_path": str(src)}
        if a.no_sanitize or src.suffix.lower() not in TEXT_EXT:
            shutil.copy2(src, tgt)
            rec.update(sanitized=False)
        else:
            body = raw
            notes_removed: list[str] = []
            n_notes = 0
            if redact_notes:
                body, n_notes, notes_removed = redact_steering(body)
            new, rules, n = sanitize(body)
            if n or n_notes:
                tgt.write_bytes(new)
                shutil.copystat(src, tgt)
                rec.update(sanitized=True, sanitize_rules_applied=rules,
                           n_substitutions=n, source_sha256=src_hash,
                           sanitize_note="snapshot copy is redacted; the SOURCE "
                                         "file is byte-unchanged and its sha256 "
                                         "is source_sha256")
                if n_notes:
                    rec.update(n_editorial_notes_redacted=n_notes,
                               editorial_notes_redacted=notes_removed,
                               editorial_note_reason=
                               "the prose names this artifact, so withholding it "
                               "would recreate the round_04 defect; the "
                               "author-side editorial asides are removed instead "
                               "and listed here in full")
            else:
                shutil.copy2(src, tgt)
                rec.update(sanitized=False)
        rec["sha256"] = sha256(tgt)
        rec["size_bytes"] = tgt.stat().st_size
        records.append(rec)

    # ---- manuscript closure -------------------------------------------------
    files, missing = closure(main_tex)
    for f in sorted(files):
        try:
            rel = str(f.relative_to(pd))
        except ValueError:
            rel = f.name
        if is_blind_path(rel):
            excluded.append({"path": rel, "reason": "blindness rule "
                                                    f"{BLIND_EXCLUDE}"})
            continue
        if (r := excluded_reason(rel)):
            excluded.append({"path": rel, "reason": r})
            continue
        # NB: no content screen here. A manuscript file that trips the blindness
        # screen must be a hard FAILURE, not a quarantine -- silently dropping a
        # section would freeze a snapshot that is not the paper. The post-pack
        # audit catches it and sets rc=1.
        emit(f, f"manuscript/{rel}")

    pdf = pd / (Path(a.main).stem + ".pdf")
    if pdf.is_file():
        emit(pdf, f"manuscript/{pdf.name}")

    # ---- what does the prose name? (needed BEFORE packing) -----------------
    # Computed here rather than after packing because an artifact the prose names
    # gets different treatment from one it does not: if the only reason to
    # withhold it is a redactable author-side aside, redact and ship it. Dropping
    # it instead is precisely the round_04 defect.
    named = named_artifacts({f for f in files if f.suffix == ".tex"}, pd)

    # ---- artifacts: default-include, explicit-exclude ----------------------
    packed_sources: set[Path] = {Path(r["source_path"]) for r in records}
    explicit_requests: set[Path] = set()
    roots: list[tuple[Path, str]] = []
    if not a.no_default_artifacts:
        for sub, snap in DEFAULT_ARTIFACT_ROOTS:
            if (pd / sub).is_dir():
                roots.append((pd / sub, snap))
    for e in a.evidence:
        src = Path(e) if Path(e).is_absolute() else REPO / e
        if not src.exists():
            print(f"error: evidence not found: {src}", file=sys.stderr)
            return 2
        roots.append((src, "evidence"))
        explicit_requests.add(src.resolve())

    for src, snap_root in roots:
        srcs = ([src] if src.is_file()
                else sorted(x for x in src.rglob("*") if x.is_file()))
        for s in srcs:
            if s in packed_sources:
                continue
            try:
                paper_rel = str(s.relative_to(pd))
            except ValueError:
                paper_rel = s.name
            if is_blind_path(paper_rel):
                # v1 bug: this check existed for the closure but NOT here, so
                # rounds 00-02 shipped tcodex_out/EVIDENCE_PACK.md.
                excluded.append({"path": paper_rel,
                                 "reason": f"blindness rule {BLIND_EXCLUDE}",
                                 "explicitly_requested":
                                     s.resolve() in explicit_requests})
                continue
            # A prose-named artifact whose ONLY blocker is a redactable aside is
            # redacted and shipped, not withheld.
            is_named = paper_rel in named
            redact_notes = False
            if is_named and s.suffix.lower() in TEXT_EXT:
                hits = screen_fatal(s)
                if hits:
                    body = s.read_bytes()
                    cleaned, n_notes, _ = redact_steering(body)
                    if n_notes and not screen_fatal_bytes(cleaned):
                        redact_notes = True
            if not redact_notes and (r := excluded_reason(paper_rel, s)):
                excluded.append({"path": paper_rel, "reason": r,
                                 "named_by_prose": is_named,
                                 "explicitly_requested":
                                     s.resolve() in explicit_requests})
                continue
            if src.is_file():
                snap_rel = f"{snap_root}/{s.name}"
            else:
                snap_rel = f"{snap_root}/{s.relative_to(src)}"
            packed_sources.add(s)
            emit(s, snap_rel, redact_notes=redact_notes)

    snap_paths = {r["snapshot_path"] for r in records}

    # ---- named-artifact gate ----------------------------------------------
    named_missing, dangling_blind, named_ok = [], [], []
    for rel, citers in sorted(named.items()):
        blind = is_blind_path(rel)
        if rel.endswith("/"):
            hit = any(p.startswith(f"evidence/{rel}")
                      or p.startswith(f"{rel}") for p in snap_paths)
        else:
            hit = (f"evidence/{rel}" in snap_paths or rel in snap_paths
                   or f"manuscript/{rel}" in snap_paths)
        entry = {"named_path": rel, "cited_in": citers}
        if hit:
            named_ok.append(rel)
        elif blind:
            entry["reason"] = ("target is excluded from every blind snapshot by "
                               "rule; the prose points reviewers at a file they "
                               "cannot open")
            dangling_blind.append(entry)
        else:
            ex = next((e for e in excluded
                       if e["path"] == rel or e["path"].rstrip("/") == rel.rstrip("/")),
                      None)
            entry["reason"] = (f"withheld: {ex['reason']}" if ex
                               else "named by the prose but absent from the snapshot")
            named_missing.append(entry)

    # ---- blindness / leak audit -------------------------------------------
    blind_viol, blind_warn, leak_viol = audit(dest)

    digest = hashlib.sha256()
    for r in sorted(records, key=lambda x: x["snapshot_path"]):
        digest.update(r["snapshot_path"].encode())
        digest.update(b"\0")
        digest.update(r["sha256"].encode())
        digest.update(b"\n")

    n_sanitized = sum(1 for r in records if r.get("sanitized"))
    # An artifact the operator asked for BY NAME and did not get is always a
    # hard error, even when withholding it was the right call: the operator's
    # intent and the blindness rule are in conflict and a human must resolve it.
    # Silently honouring the rule is how "I passed --evidence X" turns into "X
    # was never in the snapshot" without anyone noticing.
    denied_requests = [e for e in excluded if e.get("explicitly_requested")]

    gate_pass = not (missing or named_missing or blind_viol or leak_viol
                     or denied_requests
                     or (dangling_blind and not a.allow_dangling_blind_refs))

    manifest = {
        "schema_version": "2.0.0",
        "round": a.round,
        "paper": str(pd.relative_to(REPO)) if str(pd).startswith(str(REPO)) else str(pd),
        "snapshot_sha256": digest.hexdigest(),
        "n_files": len(records),
        "n_evidence_files": sum(1 for r in records
                                if r["snapshot_path"].startswith("evidence/")),
        "n_sanitized_files": n_sanitized,
        "freeze_gate_pass": gate_pass,
        "missing_dependencies": missing,
        "explicit_requests_denied": denied_requests,
        "named_artifacts_missing": named_missing,
        "named_artifacts_present": named_ok,
        "named_artifacts_dangling_by_blindness_rule": dangling_blind,
        "blindness_violations": blind_viol,
        "blindness_warnings": blind_warn,
        "internal_leak_violations": leak_viol,
        "excluded_by_blindness_rule": list(BLIND_EXCLUDE),
        "excluded_files": sorted(excluded, key=lambda x: x["path"]),
        "sanitize_rules": [{"pattern": p, "replacement": r}
                           for p, r in SANITIZE_RULES],
        "artifact_policy":
            "default-include: <paper>/{evidence,gate,code} are packed "
            "recursively; --exclude withholds. Inverted from the v1 --evidence "
            "whitelist on 2026-08-16 after round_04 shipped 2 of 25 artifact "
            "records with missing_dependencies=[].",
        "blindness_note":
            "This snapshot deliberately contains NO previous reviews, scores, "
            "response letters, writer notes, or target thresholds. Reviewers "
            "must see only this directory and the rubric. Two tiers are "
            "enforced: blindness_violations (prior-round disclosure or "
            "author-side steering) are FATAL and the file is withheld; "
            "blindness_warnings (a bare mention of an excluded directory name, "
            "with no prior-round content) are recorded and shipped, per the "
            "adjudication in evidence/internal_paths_in_submission.json.",
        "sanitize_note":
            "Snapshot copies of text artifacts are redacted for internal "
            "absolute paths, node addresses, internal hostnames and "
            "affiliation. SOURCE files are byte-unchanged; each redacted record "
            "carries source_sha256 of the original.",
        "files": sorted(records, key=lambda x: x["snapshot_path"]),
    }
    (dest / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items() if k != "files"},
                     indent=2, ensure_ascii=False))
    print(f"\n[freeze_round] {len(records)} files "
          f"({manifest['n_evidence_files']} evidence, {n_sanitized} redacted, "
          f"{len(excluded)} withheld) -> {dest}", file=sys.stderr)
    if blind_warn:
        print(f"[freeze_round] note: {len(blind_warn)} blindness WARNING(s) "
              f"(bare directory-name mentions, adjudicated non-breach): "
              f"{sorted({w['file'] for w in blind_warn})}", file=sys.stderr)

    rc = 0
    if denied_requests:
        print(f"[freeze_round] FAIL: {len(denied_requests)} explicitly requested "
              f"artifact(s) were withheld -- resolve the conflict between your "
              f"--evidence request and the blindness rule: "
              f"{[e['path'] for e in denied_requests][:5]}", file=sys.stderr)
        rc = 1
    if missing:
        # A missing dependency means the frozen snapshot is not the paper.
        print(f"[freeze_round] FAIL: {len(missing)} missing LaTeX dependency/ies: "
              f"{missing[:5]}", file=sys.stderr)
        rc = 1
    if named_missing:        # The round_04 defect. The paper names an artifact it did not ship.
        print(f"[freeze_round] FAIL: {len(named_missing)} artifact(s) named by "
              f"the prose are absent from the snapshot: "
              f"{[e['named_path'] for e in named_missing][:6]}", file=sys.stderr)
        rc = 1
    if dangling_blind:
        msg = (f"{len(dangling_blind)} artifact(s) named by the prose are "
               f"blind-excluded by rule and can never be shipped: "
               f"{[e['named_path'] for e in dangling_blind][:6]} -- fix the "
               f"PROSE (cite a shippable record), do not ship these")
        if a.allow_dangling_blind_refs:
            print(f"[freeze_round] WARN: {msg}", file=sys.stderr)
        else:
            print(f"[freeze_round] FAIL: {msg}", file=sys.stderr)
            rc = 1
    if blind_viol:
        print(f"[freeze_round] FAIL: {len(blind_viol)} blindness violation(s) in "
              f"the packed snapshot: {blind_viol[:3]}", file=sys.stderr)
        rc = 1
    if leak_viol:
        print(f"[freeze_round] FAIL: {len(leak_viol)} internal-identifier "
              f"leak(s) survived sanitisation: {leak_viol[:3]}", file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
