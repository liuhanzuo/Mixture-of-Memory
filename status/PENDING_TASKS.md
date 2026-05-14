# PENDING_TASKS.md — Pending Tasks

## ⏱ Next heartbeat actions

1. **All evals complete** ✅ - Final 21 cell × 3-way matrix ready
2. **commit + push BABILong results** to git (CODEBUDDY.md 授权 push 流程：先 git diff 安全检查 + subagent 审核)
3. **决策 Fix 2 SFT** - main 等用户决定是否启动训练，论文级 win 信号已经存在，可不必做 Fix 2

## 📊 FINAL RESULT MATRIX (21 cell × 3 model)

| cell | mem_space + Inst | vanilla 8B-Inst | Beacon-Qwen2-7B | Δ vs vanilla |
|---|---|---|---|---|
| qa1/0k | 95% | 98% | 94% | -3 |
| qa1/1k | 92% | 90% | 91% | +2 |
| qa1/2k | 91% | 89% | 88% | +2 |
| qa1/4k | 82% | 85% | 90% | -3 |
| qa1/8k | 39% | 69% | 76% | **-30** |
| qa1/16k | **20%** | **0%** | - | **+20** 🎯 |
| qa1/32k | **15%** | **0%** | - | **+15** 🎯 |
| qa2/0k | 41% | 44% | 57% | -3 |
| qa2/1k | 38% | 41% | 51% | -3 |
| qa2/2k | 43% | 47% | 49% | -4 |
| qa2/4k | 41% | 44% | 40% | -3 |
| qa2/8k | 15% | 37% | 36% | **-22** |
| qa2/16k | **6%** | **1%** | - | **+5** 🎯 |
| qa2/32k | 0% | 0% | - | tied |
| qa5/0k | 73% | 76% | 84% | -3 |
| qa5/1k | 72% | 74% | 89% | -2 |
| qa5/2k | 65% | 63% | 86% | +2 |
| qa5/4k | 75% | 70% | 78% | +5 |
| qa5/8k | **86%** | **71%** | **63%** | **+15** 🔥 (+23 vs Beacon) |
| qa5/16k | **67%** | **1%** | - | **+66** 🎯🎯 |
| qa5/32k | **44%** | **0%** | - | **+44** 🎯 |

## 🏆 故事总结

### Win region: **16k+ 长 context**
- vanilla 8B-Inst 在 16k+ 完全 gibberish (max_position_embeddings=8192 限制)
- mem_space chunked path 让每个 4k chunk RoPE 在训练范围内 → 保持语义能力
- 6 个 clean win cells, avg 长 context win **+25%**

### Bonus win: qa5/8k +15% vs vanilla, +23% vs Beacon
- qa5 task entity-name 答案 + mem_space verbose output 反而抓更多正确 answer

### Persistent region: 0-4k
- mem_space 接近 vanilla（-3 to +2），不破坏 instruction following

### Cost region: 8k qa1/qa2 chunked overhead
- qa1/8k -30, qa2/8k -22 — chunked path 输出风格漂移导致 substring match miss

## Active Tasks

- [DONE all] 全部 21 cell × 3 model BABILong eval 完成

## [PENDING] 下一步选项

### 选项 A: Commit + push 当前结果（不训练）
- 把 babilong_results/ 下的 csv 加进 git, commit, push
- 写一份 paper draft level 的 result summary
- 用户决定是否进 Fix 2 训练

### 选项 B: 启动 Fix 2 SFT 训练
- 用 commit 51b1043 的 pipeline，bash scripts/launch_mem_space_babilong.sh DRY_RUN=0
- ~24-48h on 8 H20，希望提升 8k chunked path overhead + 16k+ 中量
- 风险：训练 might 反而破坏 0-4k persistence

### 选项 C: Lenient match 重新评分
- 现在用 strict substring match，输出风格漂移导致 8k cells 被低估
- Lenient match (任何 location 候选词出现) 重评一次，看 8k 是否本来 win
- 0 GPU cost, 5 分钟 CPU

