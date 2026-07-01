---
name: standing-autonomy-grant
description: User granted broad standing autonomy — explore, edit code, launch experiments/evals without per-action confirmation
metadata:
  type: feedback
---
2026-06-25: User granted broad standing authority — "你的权限很高 可以自由探索,改代码,启动实验,启动eval" and "以后不需要再和我确认这些东西". Do NOT ask for confirmation before exploring, changing code, launching training runs, or launching evals.

**Why:** User wants high throughput and trusts the agent to act autonomously on this research project; per-action confirmation slows the loop.

**How to apply:** Default to acting. Dispatch researcher/coder subagents and workflows, change hyperparameters, kill/restart experiments, launch evals — all without asking. Still落账 to status files (TRAINER_ACTIVITY.jsonl, RUN_REGISTRY.md, UPDATELOG.md) and surface results/decisions. Reserve confirmation only for genuinely destructive/irreversible shared-system actions (per CLAUDE.md red lines). Aligns with the project's existing自主派发规则 in [[CLAUDE.md memory]] but extends it to a blanket grant.
