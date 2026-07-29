# Paper B — Paired analysis: keep14@200k vs fully-random-init@200k

Per-task McNemar test + paired bootstrap 95% CI on accuracy difference.
Item-id paired, NaN excluded. CPU only.

| task | n | keep14 acc | random acc | diff (pp) | McNemar b/c | McNemar p | bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---|---:|---|
| mmlu | 14042 | 0.3191 | 0.2461 | 7.30 | 2955/1930 | 5.96e-49 | [6.35, 8.26] |
| lambada_openai | 5153 | 0.5773 | 0.4838 | 9.35 | 925/443 | 1.84e-39 | [7.94, 10.75] |
| boolq | 3270 | 0.6382 | 0.6138 | 2.45 | 730/650 | 3.34e-02 | [0.24, 4.65] |
| commonsense_qa | 1221 | 0.4988 | 0.4505 | 4.83 | 159/100 | 2.97e-04 | [2.29, 7.37] |
| social_iqa | 1954 | 0.4340 | 0.4156 | 1.84 | 166/130 | 4.17e-02 | [0.15, 3.53] |

## Interpretation
- diff > 0 means keep14 (inherited+train-all) beats fully-random-init.
- McNemar p < 0.05: significant discordant pairs (one arm right where other wrong).
- bootstrap CI excludes 0: significant accuracy difference.
- freeze-front has no per-example predictions, so is NOT in this paired analysis.
