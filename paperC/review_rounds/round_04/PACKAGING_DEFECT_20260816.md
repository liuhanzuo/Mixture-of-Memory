# round_04 packaging defect — the reviewers scored an incomplete artifact

**Date:** 2026-08-16
**Author:** MAIN (packaging repair task)
**Status:** defect confirmed, cause identified, repaired artifact frozen alongside the original.
**Scope of this document:** attribution only. It says which review issues were caused by
packaging rather than by the paper. **It does not change any reviewer's score or verdict**,
and it is not a rebuttal. Re-adjudication is for MAIN and an independent meta-reviewer.

---

## 1. What happened

The six blind reviewers of round_04 scored the artifact whose manifest reads

```
snapshot_sha256: 7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a
n_files: 34
missing_dependencies: []
```

That artifact shipped **2 evidence records**: `build_record.json` and
`claim_evidence_map.tsv`. The paper's appendix (Table 12, `tab:artifact-map`) resolves the
evidence identifiers `E-A` … `E-K` and `E-CAL` to specific files, and the Reproducibility
Statement says "every quantitative claim in this paper is bound to a machine-readable
record". Of the artifact paths that Table 12 and the body name, **10 were absent** from the
snapshot the reviewers were given (measured by replaying the exact round_04 invocation under
the repaired script):

```
evidence/floor_winners_curse_calibration.json   (E-CAL)
evidence/heal_readout_v2_permutation_null.json  (E-D)
evidence/second_mc_benchmark/                   (E-F)
evidence/mmlu_scale_power/                      (E-B)
evidence/construct_nulls_length_unit.json       (E-H)
evidence/SECOND_MC_BENCHMARK_VERDICT.md
evidence/R7_BOOTSTRAP_P_FIX.md
evidence/POWER_WALL_VERDICT.md
code/emit_tab_construct_nulls.py                (the emitter named in Table 12)
code/check_prose_vs_evidence.py                 (the checker named in Table 12)
```

Separately, at the 11:47:19 freeze time there were **29 files under `paperC/evidence/`**
(19 at top level plus three subdirectories) and only 2 were packaged. The 10 above are the
subset the manuscript names by identifier, which is why they are the load-bearing ones.

Four of the six independently identified this, in near-identical terms:

| Reviewer | Issue | Severity | Score |
|---|---|---|---|
| X1 (novelty) | `R1` | major | 5 |
| X2 (soundness) | `PROV-MISSING-EVIDENCE` | major | 4 |
| X4 (clarity) | `X4-07` | major | 4 |
| X5 (repro) | `R1` | major | 4 |
| X6 (adversary) | `TS-4` | major | 4 |

X5 made it the **leading** reason for the score:

> "The main reason is not presentation: it is that the frozen artifact does not contain the
> evidence it repeatedly claims to publish, and two decision-relevant statistical arguments
> are invalid as written."

**The reviewers were right about the artifact.** They were describing a packaging failure,
not a defect in the research: no number in the manuscript is wrong because of this, and the
records they asked for existed on disk the whole time.

## 2. Cause

`freeze_round.py` took evidence as a repeated **whitelist** (`--evidence PATH`, repeatable).
The round_04 freeze was invoked with two flags:

```
python freeze_round.py paperC --round 4 \
  --evidence paperC/gate/build_record.json \
  --evidence paperC/evidence/claim_evidence_map.tsv
```

The script did exactly what it was told. The defect is an interface that requires a human to
enumerate ~50 paths by hand and provides no check that the enumeration is complete — and
which, by reporting `missing_dependencies: []`, actively signalled that nothing was missing.
That field only ever followed the LaTeX `\input`/`\includegraphics` chain. A file named in a
caption via an `E-` identifier was invisible to it. The gate that should have caught this did
not exist.

## 3. What was repaired

`freeze_round.py` (in `.claude/skills/autonomous-paper-agent/scripts/`) was inverted to
**default-include, explicit-exclude**, and three fatal gates were added. Full detail is in
the script's own docstring; the summary:

1. **Artifact packing is now recursive by default** over `<paper>/{evidence,gate,code}`.
   `--exclude GLOB` withholds; `--no-default-artifacts` restores the old behaviour.
2. **Named-artifact gate.** Every artifact path the prose names — parsed from the
   `tab:artifact-map` identifier table and from body `\texttt{...}` paths — must be present,
   or the freeze exits non-zero. This is the gate that would have caught round_04.
3. **Blindness gate, now applied to artifacts as well as to the manuscript closure.**
4. **Anonymity gate.** Internal absolute paths, node addresses, internal hostnames and
   affiliation strings are redacted in the snapshot copy; **source files are never modified**
   and each redacted record carries `source_sha256` of the untouched original.

## 4. The repaired artifact

`review_rounds/round_04/submission_complete/`

```
snapshot_sha256: ffd5fd7d8c3d8b30d44c684f03fb7261972e378e30eb33b032a5640788ab162f
n_files: 76        (32 manuscript + 44 artifacts, incl. 30 under evidence/)
freeze_gate_pass: true
```

**`round_04/submission/` was not touched.** Its hash `7fcb9ccc…` is the provenance of what
the reviewers actually scored; overwriting it would destroy round_04's auditability. Verified
after the repair: still `7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a`.

The manuscript in `submission_complete/` is **byte-identical** to the reviewed manuscript
(32/32 files, `cmp` clean), so the two artifacts differ *only* in packaging. Note that the
working tree's manuscript has since been edited by concurrent work (see §6), so
`submission_complete/` was reconstructed from the reviewed manuscript bytes plus artifacts
that existed at the 11:47:19 freeze time — not from the live tree.

Accounting, asserted mechanically at freeze time:

```
round_04-era artifacts on disk : 51
withheld (blindness screen)    :  7
shipped artifacts              : 44   = 26 evidence + 4 evidence/gate + 14 code
manuscript files               : 32
MANIFEST.json                  :  1
total n_files                  : 76
```

Seven artifacts are **deliberately withheld** and the reason is recorded per-file in the
manifest. Each discloses the review process itself — e.g. `venue_page_budget_round03.json`
("round 3 pre-review audit"), `venue_verification_openreview.json` (cites
`round_00/submission/...`), `s2_02_stratified_ordering.json` and
`s2_03_symmetric_inference.json` (both attribute findings to "the round_01 reviewer"). These
cannot be shipped to a blind reviewer and cannot be redacted without destroying their
meaning. **This is a real residual gap:** `s2_02` and `s2_03` back the stratified-ordering
and symmetric-inference analyses, so two analyses still lack a shippable record. The fix is
editorial — re-emit those two records without the review-process commentary — and is
**not** done here.

## 5. Which review issues are attributable to packaging

**Attributable to packaging (should be re-scored against `ffd5fd7d…`):**

- X1 `R1`, X2 `PROV-MISSING-EVIDENCE`, X4 `X4-07`, X5 `R1`, X6 `TS-4` — all five are the
  "artifact does not ship its evidence" issue. All five are **substantially but not fully**
  resolved: 30 evidence records ship instead of 2, and 5 of the 9 evidence identifiers now
  resolve completely, but `E-I` still resolves to nothing and `E-A`/`E-B`/`E-F` resolve only
  via a co-source (see §6b).
- X5's score in particular was driven **primarily** by this issue by its own statement, so
  its 4/10 is the number most likely to move on a complete artifact.

**NOT attributable to packaging — these are substantive and remain open:**

- The MMLU-Pro winner's-curse calibration using a null incompatible with variable legal
  option counts (X2, X4, X5, X6 all raise it independently). Shipping more files does not
  address it.
- The aggregate *p* = 0.0196 assuming independence the paper itself denies.
- The pre-comparison-gate interpretation not following from subtracting a common constant.
- X1 `N3` (missing ACL 2025 synthesis) — a citation/positioning issue.
- X1 `R2` (**stale build record**) — *verified still true in the repaired artifact*:
  `build_record.json` records `pdf_pages 22`, `pdf_bytes 355196`, `pdf_sha256 56a376e1…`,
  while the shipped PDF is `366583` bytes, sha `1fbaaf998322…`, and
  `pdf_visually_inspected: false`. Packaging did not and could not fix this; the build record
  must be regenerated against the final PDF.

**Caution against over-crediting this repair.** Four reviewers raised the artifact gap *and*
independent statistical objections. Only the former is packaging. The verification tests they
wrote are also stronger than "the files are present" — X1, X2, X5 and X6 all require that a
reader **regenerate the headline tables** from the shipped records on a clean machine. That
end-to-end regeneration **has not been run** and is the correct next gate.

## 6. Two defects found during the repair that are larger than the original report

**(a) Rounds 00, 01 and 02 shipped a blindness leak.** Each of those snapshots contains
`evidence/EVIDENCE_PACK.md`, md5 `4d0013b52eedc06b48c3b930a76ba014` — **byte-identical to the
author-side `paperC/tcodex_out/EVIDENCE_PACK.md`**. `tcodex_out` is on the blindness
exclusion list, but v1 applied that list only to the manuscript closure; the `--evidence`
loop bypassed it entirely. That file contains author-facing steering, e.g. "⚠️ **DO NOT claim
differential learning rates on either side**", "attacking it would be an easy referee kill",
and a reference to "the corresponding limitation L1 in the writer prompt". This is a
**blindness breach**, not a presentation defect, and it means the scores from rounds 00-02
were produced by reviewers with access to author-side strategy notes. The repaired script
blocks this (verified by negative control), but **the historical rounds 00-02 scores should be
treated as compromised.** The prior assessment in `paperC/state/assumptions.md` — "I grepped
the frozen tree … Result: zero leaks" — searched for score/verdict vocabulary and did not
search for writer-steering imperatives, so it missed this.

**(b) The paper still names an unshippable file, and one identifier resolves to nothing.**
`tcodex_out/EVIDENCE_PACK.md` is cited **8 times** in `sections/09_appendix.tex`, including as
a source for `E-A`, `E-B`, `E-D`, `E-E`, `E-F`, `E-H`, `E-I` and `E-K`. It can never enter a
blind snapshot. `evidence/internal_paths_in_submission.json` already diagnosed this class in
round_03 and recommended citing stable evidence IDs instead; that fix was applied to some
captions but the artifact-map table still points at the file. The repaired freeze reports this
as a fatal error by default; `submission_complete` was produced with
`--allow-dangling-blind-refs` so the flag is recorded rather than blocking.

Checked per identifier against `ffd5fd7d…`: `E-CAL`, `E-D`, `E-E`, `E-H` and `E-K` now resolve
to a shipped file; `E-A`, `E-B` and `E-F` resolve to a shipped *co-source* but their
`EVIDENCE_PACK.md` component does not; and **`E-I` resolves to nothing at all**, because
`EVIDENCE_PACK.md` is its only listed source. So the reviewers' complaint is *not* fully
retired by this repair — `E-I` remains unverifiable from the artifact, and that is a prose /
evidence-emission fix which has **not** been made here.

## 7. Reproducing this

```bash
python .claude/skills/autonomous-paper-agent/scripts/freeze_round.py \
  paperC --round <N> --dest submission
```

Exits non-zero if a LaTeX dependency is missing, a prose-named artifact is absent, an
explicitly requested artifact was withheld, a blindness violation reaches the snapshot, or an
internal identifier survives redaction. Eight negative controls were run against a
known-green tree, each injecting one defect and confirming a non-zero exit. Replaying the
**exact** round_04 invocation under the new script reports **10 named-missing artifacts and
exits 1**, where v1 reported `missing_dependencies: []` and exited 0:

```bash
python .../freeze_round.py paperC --round 4 --no-default-artifacts \
  --evidence paperC/gate/build_record.json \
  --evidence paperC/evidence/claim_evidence_map.tsv
```
