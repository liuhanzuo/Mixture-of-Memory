# probe#2 Qwen3-8B prune-heal: in-domain vs held-out ppl (2026-07-13)
| dataset | full 36L | armB 14L(keep12+2fresh,step36000) | gap |
|---|---|---|---|
| slimpajama(in-domain) | 11.11 | 12.75 | +15% |
| wikitext(held-out) | 11.70 | 19.53 | +67% |

结论: slimpajama近似是in-domain假象; held-out(+67%)才是剪层真实通用能力损失. armB step36000仍训练中,追踪wiki gap趋势.

---
## ★修正(2026-07-13, 用户指正): 两benchmark都confound方向相反, 不下结论
- slimpajama: armB continue-train过 → armB占in-domain便宜(gap低估).
- wikitext: Qwen3-8B预训练大概率含wiki → full占in-domain便宜; armB在slimpajama CT后漂离wiki分布(de-adapt非丢通用能力) → gap高估.
- 两者都非中立. 不能从中途+单benchmark断言"接近"或"大损失". 正确做法: 等armB训完(200k)再评, 用多个中立held-out集.
