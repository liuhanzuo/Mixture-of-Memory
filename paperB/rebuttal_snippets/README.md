# paperB/rebuttal_snippets/

Drop-in LaTeX 片段，若 Paper B rebuttal 需要用今晚 audit 得到的 Finding 2 单变量 letter-axis 证据，
直接从这里贴到 `paperB/sections/` 或 `paperB/sections/08_appendix.tex`。

## 内容

- **`tab_letter_headroom.tex`** — 10-arm 表：letter above-chance headroom + Wilson 95% CI + one-sided binomial。
  PPL 匹配 pair (Random-16L / keep14@67.5k) 都 chance-level 不显著；单调塌陷 base +35.5pp → chance。
  label: `tab:letter-headroom`。
- **`finding2_letter_axis_paragraph.tex`** — 单段解释，可直接插到 §4.2 (Finding 2) 后或
  §8 appendix。说明 chance-corrected 视角，`\ref{tab:letter-headroom}` 引用上表。

## 使用

```latex
% 在 04_experiments.tex 或 08_appendix.tex 加：
\input{sections/tab_letter_headroom}  % 若移到 sections/
\input{sections/finding2_letter_axis_paragraph}
```

或者只在 rebuttal PDF 引用，不改主 tex。

## 数据来源

`paperB/audit_20260805/finding2_letter_headroom.tsv` (commit `6a3b6bb`) + 
`paperB/audit_20260805/finding2_chance_correction.md` (commit `51c7349`) + 
`paperB/audit_20260805/tex_numbers_vs_disk.md` (commit `638fb04`)

原始每-item score：`paperB/anonymous_artifact/scores/mmlu/*_letter_acc.json`。
