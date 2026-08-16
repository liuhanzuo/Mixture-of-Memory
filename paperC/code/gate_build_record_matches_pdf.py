#!/usr/bin/env python3
"""GATE: gate/build_record.json must describe the PDF that is actually on disk.

Round_04 found build_record.json certifying 22 pages / 355196 bytes while main.pdf was
24 pages / 366583 bytes. A build record whose sha256, byte count and page count do not
match the artefact beside it certifies nothing -- and the page count is the field a
venue page limit is checked against, so a stale record can hide an over-length paper.

Pure probe; writes only with --json_out. 0 GPU.

Negative control:
    python3 code/gate_build_record_matches_pdf.py --selftest_negative_control
perturbs each of the three fields in an in-memory copy and asserts the gate fails on
each one independently. Proves the assertion can stop something, not merely that it exists.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(HERE, "gate", "build_record.json")


def measure(pdf_path):
    with open(pdf_path, "rb") as f:
        b = f.read()
    out = {"pdf_bytes": len(b), "pdf_sha256": hashlib.sha256(b).hexdigest()}
    try:
        import pymupdf
        out["pdf_pages"] = pymupdf.open(pdf_path).page_count
    except Exception:
        try:
            import fitz
            out["pdf_pages"] = fitz.open(pdf_path).page_count
        except Exception as e:
            out["pdf_pages"] = None
            out["page_count_error"] = repr(e)
    return out


def check(rec, measured, pdf_path, verbose=True):
    fails = []
    for field in ("pdf_bytes", "pdf_sha256", "pdf_pages"):
        claimed = rec.get(field)
        actual = measured.get(field)
        if actual is None:
            fails.append(f"FAIL: cannot measure {field} on this host "
                         f"({measured.get('page_count_error','no reader')}); the build "
                         f"record's {field}={claimed!r} is therefore UNVERIFIED and must "
                         f"not be treated as certified.")
            continue
        ok = (claimed == actual)
        if verbose:
            print(f"  {field:<12} record={claimed!r:<70} actual={actual!r}  "
                  f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            extra = ""
            if field == "pdf_pages":
                extra = (" This is the field a venue page limit is checked against, so a "
                         "stale value can hide an over-length submission.")
            fails.append(f"FAIL: build_record.{field} = {claimed!r} but {os.path.basename(pdf_path)} "
                         f"is {actual!r}. The record does not describe the artefact beside "
                         f"it.{extra}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--selftest_negative_control", action="store_true")
    a = ap.parse_args()

    with open(a.record) as f:
        rec = json.load(f)
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(a.record)),
                            rec.get("pdf", "main.pdf"))
    if not os.path.exists(pdf_path):
        print(f"FAIL: build_record names pdf={rec.get('pdf')!r} but {pdf_path} does not exist.")
        return 2
    measured = measure(pdf_path)

    if a.selftest_negative_control:
        print("=" * 78)
        print("NEGATIVE CONTROL: perturbing each certified field on an IN-MEMORY copy")
        print("(gate/build_record.json is not modified)")
        print("=" * 78)
        rc = 0
        for field, mutate in [("pdf_bytes", lambda v: (v or 0) + 1),
                              ("pdf_sha256", lambda v: "0" * 64),
                              ("pdf_pages", lambda v: (v or 0) + 2)]:
            m = copy.deepcopy(rec)
            m[field] = mutate(measured[field])
            f = check(m, measured, pdf_path, verbose=False)
            hit = [x for x in f if field in x]
            print(f"NC perturb {field:<11} -> "
                  f"{'CAUGHT' if hit else 'NOT CAUGHT <-- GATE IS BLIND'}")
            for x in hit:
                print("     ", x[:140])
            rc |= 0 if hit else 1
        f0 = check(rec, measured, pdf_path, verbose=False)
        print(f"NC positive control (unperturbed record) -> {len(f0)} failure(s) "
              f"{'(record is stale on disk; see main run)' if f0 else 'OK'}")
        print()
        print("NEGATIVE CONTROL " + ("PASSED" if rc == 0 else "FAILED"))
        return rc

    print(f"build record: {a.record}")
    print(f"artefact    : {pdf_path}")
    fails = check(rec, measured, pdf_path)
    print()
    for x in fails:
        print("  " + x)
    verdict = "PASS" if not fails else "FAIL"
    print(f"GATE build_record_matches_pdf: {verdict}")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"schema_version": "1.0.0", "verdict": verdict,
                       "record": a.record, "pdf": pdf_path,
                       "claimed": {k: rec.get(k) for k in
                                   ("pdf_bytes", "pdf_sha256", "pdf_pages")},
                       "measured": measured, "failures": fails,
                       "gpu_used": "NONE"}, f, indent=1)
        print("wrote", a.json_out)
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
