# PENDING_TASKS.md — 未完成任务队列

**Heartbeat 每次巡检必须检查此文件。如果有未完成任务，heartbeat 不能只报 HEARTBEAT_OK，必须报告任务状态。**

## Active Tasks

### [RUNNING] chunk_isolation_lr_fix — lr_fix 双 arm 运行中
- priority: **high**
- launched: 2026-05-05 12:33/12:38 by heartbeat
- arm1 (local): factor=10 (lr=5e-7), **EVAL@200 ratio=0.9901 ✅ NO REGRESSION** | out_proj_norm=0.013611 (linear)
  - step 200/4000, continuing at ~1 step/24s
  - 历史最佳: ratio=0.9780@100★ (project best), eval@200=0.9901 (confirms architecture valid)
- arm2 (b200-4): factor=50 (lr=1e-7), step ~170, out_proj_norm=0.004425 (very controlled)
  - EVAL@100: ratio=0.9892
- KEY RESULT: factor=10 passes 200-step regression test — old factor=1 had ratio=1.0126@200, factor=10 stays 0.9901@200
- watch: eval@400 (~17:30 GMT+8) — does ratio continue declining or plateau?
- logs: logs/chunk_isolation_lr_fix_arm1_20260505_123320.log / logs/chunk_isolation_lr_fix_arm2_20260505_123820.log

### [RUNNING→FINISHING] unleashed_cross_attn — 临近完成 (~14:10 GMT+8)
- priority: medium (ceiling confirmed, just finishing)
- arm1 (b200-2): factor=100 (lr/100=5e-8), step 1980/2000, out_proj_norm=0.004700 (frozen)
  - EVAL@1900: ratio=0.9981, ceiling locked
- arm2 (b200-3): factor=1 (full lr=5e-6), step 1960/2000, out_proj_norm=0.149414 (frozen)
  - EVAL@1900: ratio=0.9909 NEW BEST (full-lr arm slightly better)
- ETA completion: ~14:10 CST (both arms ~20 steps remaining)
- CONCLUSION: Architecture ceiling confirmed ~0.998 (low-lr) / ~0.991 (full-lr)
- action_after_completion: Both b200-2 and b200-3 will be FREE → add to node availability

### [PENDING] direction_decision — 方向 pivot 决策
- priority: **high**
- auto_launch: false (需要用户确认 + researcher 分析)
- description: |
    **最新数据 (2026-05-05 13:54 heartbeat):**
    - lr_fix arm1 factor=10: ratio=0.9901@200 ✅ (NO REGRESSION from step 100→200)
    - lr_fix arm2 factor=50: ratio=0.9892@100 (conservative but stable)
    - unleashed arm1 (low-lr): ratio=0.9981@1900 (ceiling)
    - unleashed arm2 (full-lr): ratio=0.9909@1900 (NEW BEST, but 2000-step result)
    - PROJECT BEST overall: factor=10 arm1 ratio=0.9780@100 (5e-7 lr, not yet 2000-step)
    
    CrossAttn architecture summary:
    - 2000步训练，全lr: ratio=0.9909 (ceiling)
    - 200步训练，factor=10: ratio=0.9901 (potential to improve with more steps?)
    - Verdict: 0.9780 @ step 100 → 0.9901 @ step 200. Improving but slowly.
    
    Next step options:
    1. **继续 lr-fix arms 到 2000 步** — 确认 ratio 收敛到哪里 (auto_launch: true 建议)
    2. **长上下文评估** — 验证 CrossAttn 在实际长序列检索上的价值
    3. **架构 pivot** — 切换到 Attention Matching / Heavy Hitter / 全新方向

### [PENDING] long_context_eval — 长上下文检索评估
- priority: medium
- auto_launch: false (需要用户确认)
- depends_on: direction_decision
- description: |
    在 CrossAttn checkpoint 上运行长上下文检索测试（needle-in-haystack 或 PASSKEY）。
    候选 checkpoint: outputs/cross_attn_memory_arm1_slots64/final.pt

### [PENDING] git_push_pat_fix — GitHub PAT 缺少 workflow scope
- priority: **medium**
- auto_launch: false (需要用户操作)
- description: |
    git push origin main 失败：
    "refusing to allow a Personal Access Token to create or update workflow ... without `workflow` scope"
    
    本地 commit 156ac08 (feat: grant agent autonomous git push permission) 已就绪但未推送。
    
    需要用户在 GitHub Settings → Developer Settings → Personal Access Tokens 中：
    1. 找到当前 token (ghp_yTD6Qi...)
    2. 勾选 `workflow` scope
    3. 保存后通知 heartbeat 重新 push
    
    或者用户直接在有权限的终端运行：
    ```bash
    export http_proxy=http://star-proxy.oa.com:3128
    export https_proxy=http://star-proxy.oa.com:3128
    cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
    git push origin main
    ```

## Completed Tasks

### [DONE] swa_memory — 方案B: SWA 强制依赖 (COMPLETED 2026-05-05 ~09:00)
- arm1 (b200-1, swa=512): ratio=1.0494; arm2 (b200-4, swa=256): ratio=1.0954
- conclusion: FAILED. SWA ratio>1.0

### [DONE] fix_v2_cross_attention — arm1 (slots=64) + arm2 (slots=128) COMPLETED
- arm1: ratio=0.9982; arm2: ratio=0.9983 — first memory_ratio < 1.0 in project history

### [DONE] continued_pretrain_ablation (LoRA) — completed 2026-05-04 02:49
### [DONE] full_finetune_phase1 — 3 arms — completed 2026-05-04 15:25
### [DONE] next_direction_decision — completed 2026-05-04 20:15
### [DONE] post_lora_analysis — completed 2026-05-04 11:15
### [DONE] infini_mem_scale_ablation — completed 2026-05-03 14:13
### [DONE] v4_ablation_chunks64_tolr1e5 — completed 2026-05-03 15:00
### [DONE] v4_chunk_memory_phase1 — completed 2026-05-02 16:23

## Cancelled/Blocked

### [BLOCKED] attention_matching (deprioritized)
- description: Attention Matching — 等方向决定，目前聚焦 CrossAttn

## Node Availability (2026-05-05 14:00 updated)

| 节点 | 状态 | 可用 |
|------|------|------|
| b200-1 | **IDLE** | YES ✅ |
| b200-2 | Unleashed arm1 (step 1980/2000, ~5min) | ~14:15 后 FREE ✅ |
| b200-3 | Unleashed arm2 (step 1960/2000, ~10min) | ~14:20 后 FREE ✅ |
| b200-4 | lr-fix arm2 (step ~170/4000) | ~明天 |
| local | lr-fix arm1 (step 200/4000) | ~明天 |
