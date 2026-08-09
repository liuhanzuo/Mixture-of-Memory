# STATUS: DEAD — Paper C v1 Frozen-Cap Proposal

Abandoned after 2026-08-06 review and repaired-evaluation rerun.

## Dead claims

- prune-and-graft is a novel construction;
- SQuAD EM demonstrates retained capability;
- forward-only probe predicts adaptation depth;
- keep14 “hero” beats meaningful controls.

## Evidence-based death

原始 SQuAD val 有 49.85% 相同拒答标签，常量函数击败全部剪枝臂。修复为
25% controlled refusal 后重跑 8 arms，所有 arm 仍不超过 25% constant floor。
因此 capability claim 在修复评测后仍失败。

P-C2 同样死亡：不同 probe 给出的 depth 横跨 0L–1L，knowledge-onset 被自身
depth sweep 证伪，拟合较好的 adaptation drift 又不是 forward-only。

## What remains useful

- controlled lesion checkpoints；
- evaluator failure case，已并入 A01；
- adaptation CKA raw evidence；
- reviewer/postmortem，防止旧 claim 复活。

Historical evidence only. Do not treat this directory as an active proposal.

