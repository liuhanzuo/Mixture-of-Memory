# Build and submission files

## Compile

```bash
latexmk -pdf -bibtex -norc -gg main.tex
```

The review PDF uses the official unmodified COLM 2026 style in submission mode.
The final `main.pdf`, LaTeX sources, bibliography, figure assets, and tables are
kept in this paper directory.

## Anonymous source-package whitelist

Package only:

- `main.tex`
- `colm2026_conference.sty`, `colm2026_conference.bst`
- the bibliography file
- `sections/*.tex` that are reachable from `main.tex`
- final figure assets referenced by the paper
- `RESPONSIBLE_NLP_CHECKLIST.md` only for author-side form preparation (do not
  upload it as paper supplementary material unless requested)

Do not upload experiment notes, TODO files, raw results, absolute paths,
credentials, build logs, auxiliary files, or camera-ready author information.
