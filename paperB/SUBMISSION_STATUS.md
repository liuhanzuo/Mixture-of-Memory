# Submission status — Paper B

- Latest frozen manuscript: `review_history/v12_20260804_183842.pdf`
- Current working-PDF SHA-256: `8045e656d97168d7204b1347bacf9a5819a1120a326868f8142caa781760ba20`
- ⚠️ **Lines 5-7 below are STALE (2026-08-16).** They were written against an old `main.aux`. A clean rebuild of the *currently checked-in* sources (`latexmk -pdf -bibtex -norc -gg main.tex`, in-repo TeX Live 2026) gives **19 pages total, Conclusion on page 9, Table 4 on page 7, Appendix starting page 14** — before any of today's edits. The drift is pre-existing and unrelated to #192; **re-verify the body page count against ACL limits before submitting.**
- Counted body: 8 pages; Conclusion ends on page 8.
- Unnumbered Limitations and Ethical Considerations occupy page 9; References start page 10.
- Total PDF: 17 pages; Appendix starts page 12.
- **After the #192 A+ edit: 20 pages total, Conclusion still page 9, Table 4 still page 7, Appendix still page 14.** All pagination movement is inside the appendix (7 float labels shift by one page); the body is byte-for-byte unmoved. 0 LaTeX errors, 0 overfull boxes.
- Table 4 reports each trained endpoint at its own deepest retained checkpoint: keep8 121k, keep10 83.5k, keep12 124k, and keep14 / ShortGPT / frozen-front / random-init at 200k. It is an endpoint inventory, not a step-matched comparison. The true keep14 128k/153.5k/200k trajectory remains in Table 3.
- Table 4's keep8/keep10/keep12 rows are the single-protocol re-measurement (torch 2.13, eval batch 8, 8/8 shards merged, full per-task `n`). This replaced a keep12 ARC-Easy value that had been merged from only 6 of 8 shards (`n_scored`=1782 instead of 2376); the corrected value is `.694`.
- Table 4's MMLU column is same-source for all nine rows (the dual-interface letter snapshot of Table 5). The marginal-MMLU-CI appendix table now uses the same basis, so keep8/keep10/keep12 read `.2550`/`.2720`/`.2728` in both places.
- ⚠️ The remaining ten columns of Table 4 are **not** single-stack: the three shallow rows are torch 2.13, the other six are torch 2.7 (same batch size, same full sample counts). This is now disclosed in the caption with a measured bound (≤0.5 pt same-arch, ≤0.9 pt cross-arch). Closing it would require re-running base/full32/keep14/ShortGPT/frozen/random under the pinned protocol; frozen-front, random-init, and full32 have **no** H20 result on disk at all, so it cannot be closed from existing artifacts.
- Table 5 and all full32 endpoint references are labeled 200k.
- Working-PDF SHA-256 after the #192 A+ edit: `6f6395cf487bb6190e247631cac6a54d8f7f9b0039f216b2439529d1d4fce2ba` (line 4's hash is older still).
- Newly integrated completed result: strict-clean closed-book recomputation, with <=0.15-point shifts and unchanged ordering.
- Anonymous artifact checksum verification passes.
- Submission PDF: `main.pdf` / `final.pdf` (identical).
