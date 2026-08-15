# paperC — LaTeX compile environment (read this before concluding "TeX is gone")

**Status 2026-08-15: the paper builds. TeX Live 2026 is installed and working.**

## The trap

Every standard probe for LaTeX says it is absent:

```
$ command -v pdflatex latexmk xelatex tectonic     # -> nothing
$ ls /usr/local/texlive /opt/texlive /root/texlive /usr/share/texlive
                                                   # -> all absent
```

This is **not** a wiped toolchain. TeX Live 2026 is installed **inside the repo, on the
project disk**:

```
./.texlive/2026/bin/x86_64-linux/       # 142 binaries: pdflatex, latexmk, xelatex, bibtex, biber
```

It is simply **not on `$PATH`**. Because it lives on wzc1 (the project disk) rather than in
`/root` or `/usr`, it is one of the few things that **does** survive a node restart — the
opposite of the conclusion the missing-`$PATH` symptom invites.

> A previous pass read `command -v` + the four absent system paths and concluded TeX Live had
> been "wiped by a node restart", citing `main.pdf`/`main.log` as proof it had once existed.
> The premise was wrong and the compile was available the whole time. `main.log` line 1 even
> names the engine that is still on disk. See `memory/persist-artifacts-on-wzc1-or-diskb.md`
> for the general rule; this is the counter-example worth remembering — **check for a
> repo-local install before declaring a tool gone.**

## How to build

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
export PATH="$PWD/.texlive/2026/bin/x86_64-linux:$PATH"
pdflatex --version     # -> pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)

cd paperC
latexmk -pdf -bibtex -interaction=nonstopmode main.tex
```

Verified 2026-08-15 (rc=0): **19 pages, 0 errors, 0 undefined references, 0 undefined
citations, 0 overfull hbox/vbox.** Machine-readable record: `paperC/gate/build_record.json`
(`pdf_sha256`, `pdf_pages`, per-pass rc, cite/input resolution).

## Where to build

| node | LaTeX | note |
|---|---|---|
| **LOCAL** | ✅ via `./.texlive` (repo-local, wzc1) | **build here** |
| `.212` (`28.89.18.212`) | ❌ nothing on `$PATH`, no system texlive tree | same wzc1 disk, so `./.texlive` is reachable there too if `$PATH` is exported — untested |
| `.73` / `.82` / `.104` | untested | different disk (zwfy6); would need its own checkout |

## Regenerating the generated table

`sections/tab_construct_nulls.tex` is **generated — never edit it by hand.** After any change
to `evidence/floor_winners_curse_calibration.json`:

```bash
PY=/opt/conda/envs/torch-base/bin/python3          # any python3; stdlib only, 0 GPU
$PY paperC/code/emit_tab_construct_nulls.py        # regenerate (refuses to write if self-tests fail)
$PY paperC/code/check_prose_vs_evidence.py         # prose/.tex vs evidence; rc=1 on any mismatch
$PY paperC/code/validate_tex_static.py --require-clean sections/tab_construct_nulls.tex
```

All three are CPU-only and take under a second. The emitter's caption embeds the evidence
file's `sha256`, so a stale table is visible in the PDF itself.

`validate_tex_static.py` is the **compile-independent** fallback: if a future node really does
lack `./.texlive`, it still checks environment nesting, brace/`$` balance, tabular column
arithmetic, row termination and `\input` resolution. It **cannot** prove the document
typesets — when it is all you have, say so explicitly rather than implying a build passed.
