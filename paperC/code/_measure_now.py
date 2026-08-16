#!/usr/bin/env python3
"""Measure current paperC/main.pdf main-text extent. Read-only probe.

Same definition as code/measure_page_budget.py:
  main_pages = 1-based PDF page carrying the "REFERENCES" heading
  extent     = ((ref_page - 1) * H + (ref_y - 85.6)) / H,  H = 646.7 pt
"""
import hashlib
import sys
from pathlib import Path

import pymupdf

BODY_TOP_PT = 85.6
BODY_H_PT = 646.7

pdf = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[1] / "main.pdf")
doc = pymupdf.open(pdf)
ref_page = ref_y = None
for i, page in enumerate(doc):
    for blk in page.get_text("dict")["blocks"]:
        for line in blk.get("lines", []):
            txt = "".join(s["text"] for s in line["spans"]).strip().upper()
            if txt == "REFERENCES":
                ref_page, ref_y = i + 1, line["bbox"][1]
                break
        if ref_page:
            break
    if ref_page:
        break
total = doc.page_count
doc.close()
if ref_page is None:
    print("REFERENCES heading NOT FOUND")
    sys.exit(2)
extent = ((ref_page - 1) * BODY_H_PT + (ref_y - BODY_TOP_PT)) / BODY_H_PT
print(f"pdf={pdf}")
print(f"total_pdf_pages={total}")
print(f"main_pages={ref_page}  ref_y={ref_y:.1f}")
print(f"extent={extent:.3f}")
print(f"pdf_sha256={hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]}")
