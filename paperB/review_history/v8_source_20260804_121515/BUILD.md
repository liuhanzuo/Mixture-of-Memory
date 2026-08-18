# Build and submission files

## Compile

```bash
latexmk -pdf -bibtex -norc -gg main.tex
```

The review PDF uses the official unmodified ACL style in review mode.
The final `main.pdf`, LaTeX sources, bibliography, figure assets, and tables are
kept in this paper directory.

## Anonymous source-package whitelist

Package only:

- `main.tex`
- `acl.sty`, `acl_natbib.bst`
- the bibliography file
- `sections/*.tex` that are reachable from `main.tex`
- final figure assets referenced by the paper
- `anonymous_artifact/` (compact evaluator/config/score snapshot; no weights or
  benchmark text)
- `RESPONSIBLE_NLP_CHECKLIST.md` only for author-side form preparation (do not
  upload it as paper supplementary material unless requested)

Do not upload experiment notes, TODO files, raw results, absolute paths,
credentials, build logs, auxiliary files, or camera-ready author information.
