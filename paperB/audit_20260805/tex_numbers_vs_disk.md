# paperB tex 数字 vs 磁盘 audit (2026-08-06 02:16, rebuttal-prep)

## 结论: 全部一致 ✓

### MMLU 数字 (letter/content_norm)
- keep14@200k letter=0.3184 ✓
- base letter=0.6054 ✓
- keep14@200k content_norm=0.3832 ✓ (tex line 40 "raises the paired snapshot to .3832")
- Random-16L content_norm=0.360 ✓ (disk 0.3598)
- Random-16L letter=0.247 ✓ (chance)

### Closed-book QA (PopQA contains / TriviaQA EM / NQ-open EM)
- base:      .257/.636/.205 ✓ (disk .2571/.6355/.2050)
- keep14:    .142/.294/.060 ✓ (disk .1415/.2940/.0598)
- ShortGPT:  .159/.330/.067 ✓ (disk .1585/.3301/.0668)
- full32:    .228/.572/.158 ✓ (disk .2280/.5715/.1582)

### 数据来源
paperB/anonymous_artifact/scores/closedbook/{base_full,keep14_step200k,shortgpt16_step200k,full32_step25000}/summary.json
paperB/anonymous_artifact/scores/closedbook/*_nqopen/summary.json
olmo2_mmlu_content_results/7B_{base,keep14_step200000,scratch16L_step200000}/summary.json

### rebuttal 意义
若 reviewer 挑具体数字, 我们可直接指向 anonymous_artifact/scores/ (已在发布 artifact 中含每-item score),
无数字漂移风险. Paper B tex 全部段落硬断言可信.
