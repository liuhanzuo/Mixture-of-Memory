# REPRO_SHA256_AUDIT_20260816.md — 6 of 9 recorded hashes do not match the files

**Verified 2026-08-16 by MAIN on wzc1. `REPRO_SHA256.txt` is NOT edited** — it is the record
of what was believed on 2026-08-04 17:14:38, and overwriting a hash manifest destroys exactly
the provenance it exists to provide. This file is the dated correction that supersedes it.

## Result

`sha256sum` over all nine pinned paths. All nine files are present on wzc1.

| verdict | mtime | path |
|---|---|---|
| MATCH | 2026-07-16 21:13 | `data/dolmino_now15b.npy` |
| MATCH | 2026-07-16 21:10 | `data/dolmino_now_val.npy` |
| **MISMATCH** | 2026-07-21 01:58 | `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` |
| **MISMATCH** | 2026-08-01 16:13 | `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt` |
| **MISMATCH** | 2026-07-25 06:05 | `outputs/olmo2_probe2_7B_keep14fresh2_fromscratch/step200000.pt` |
| **MISMATCH** | 2026-07-28 21:25 | `outputs/olmo2_probe2_7B_keep14fresh2_freezefront/step200000.pt` |
| MATCH | 2026-08-02 14:35 | `outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt` |
| **MISMATCH** | **2026-08-16 23:07** | `paperB/main.pdf` |
| **MISMATCH** | 2026-08-04 19:31 | `paperB/anonymous_artifact/SHA256SUMS.txt` |

**3 match, 6 mismatch.** Manifest written `2026-08-04 17:14:38`.

## Two different causes, and they must not be conflated

**(a) Legitimately stale — the file changed after the manifest.** Two rows, and both mtimes
are after 17:14:38:

* `paperB/main.pdf` was rebuilt **today at 23:07:51**, by the Table-4 work in flight. Note
  that `latexmk` here is **not byte-reproducible** — two runs over identical source produced
  different sha256 at identical byte length earlier this session — so a PDF hash in a manifest
  goes stale on any rebuild whether or not the source changed. A PDF hash is a claim about one
  build, not about the document.
* `paperB/anonymous_artifact/SHA256SUMS.txt` was written 2h17m after the manifest.

**(b) Wrong when recorded — the file has not changed since before the manifest.** Four
checkpoints, mtimes 07-21 through 08-01, none touched since:

* `keep14fresh2/step200000.pt`, `shortgpt16/step200000.pt`,
  `keep14fresh2_fromscratch/step200000.pt`, `keep14fresh2_freezefront/step200000.pt`.

For these the recorded value cannot be explained by later modification. Something was wrong at
recording time — a different arm hashed under this label, a truncated read, or a
transcription error. **Which of those, this audit does not establish.**

## Claims I checked and had to withdraw

* **"This is a submitted reproducibility manifest."** WRONG. `paperB/REPRO_SHA256.txt` is
  **untracked** (`git ls-files --error-unmatch` → *did not match any file known to git*), so it
  was never committed and never published. It is a local working file. My earlier
  characterisation of it as pinning a deletion candidate against a *submitted* claim overstated
  its status. It is still referenced by `paperB/data/README.md:34` and
  `paperB/review_history/v7_artifact_review_GPT56.md`.
* **"CephFS reads of large files are unstable."** REFUTED BY MY OWN TEST. Three consecutive
  `sha256sum` runs on the 48 GB checkpoint returned the identical `36413883…`, and a
  small-file control was likewise stable. The second value I briefly saw came from reading a
  background job's output file mid-write — my measurement error, not the filesystem's.
* **"All `.pt` files mismatch, all `.npy` match."** WRONG, and I nearly published it.
  `full32_dolmino/step25000.pt` **matches**. It is 4 of 5 checkpoints, not 5 of 5. The tidy
  file-type split does not exist; see
  `memory/state-direction-only-for-rows-you-computed.md` — the generalisation was one row
  ahead of the data, in the direction that made my story cleaner.

## What this does and does not affect

`paperB/data/README.md:34-37` already scopes the manifest correctly: it calls these
"current-file integrity hashes, not evidence that an identical checksum manifest existed when
the historical runs were launched." That scoping is accurate and needs no change. What it does
assert is that the values are **current**, and for 6 of 9 they are not.

**No paper number depends on these hashes.** They are integrity metadata, not measurements. No
result is retracted by this audit. The actionable consequences are:

1. Do not quote `REPRO_SHA256.txt` as evidence a file is unmodified without re-verifying.
2. The four checkpoint rows are **unusable as integrity baselines** — their recorded values
   never matched, so they cannot distinguish "file changed" from "hash was wrong".
3. If a hash manifest is wanted for a real submission, regenerate it, commit it, and **exclude
   the PDF** or state the build it belongs to.

## Reproduce

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
while read h p; do
  g=$(sha256sum "$p" 2>/dev/null | awk '{print $1}')
  [ "$h" = "$g" ] && echo "MATCH    $p" || echo "MISMATCH $p"
done < paperB/REPRO_SHA256.txt
```

~270 GB of reads; takes several minutes. `stat -c '%y %n'` on each path gives the mtimes that
separate cause (a) from cause (b).
