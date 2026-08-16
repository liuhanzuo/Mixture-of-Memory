---
name: direction-a-eval-fragility-established
description: "★2026-08-09 更正: B04/Direction A 的 GENERAL claim 已被 Qwen 跨家族复现判死 (ρ=+0.43 p=0.42); 只剩 OLMo-2-only 窄结论 (Spearman ±1.00 p=0.0028); novelty check 已做完=hold_in_backlog, 不是「只欠 novelty check」"
metadata:
  type: project
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**⚠️ 本条曾写成「Direction A 已 ESTABLISHED、距 paper 只欠 novelty check」——那是错的，已更正（2026-08-09 审计）。**

B04 `STATUS.json` 现在的 status 是 **`NARROWED_TO_OLMO_2_ONLY`**，`kill_history` 有**两**条（原记录只抄了第一条）：

| gate | verdict | 数字 |
|---|---|---|
| n=6 within-OLMo bs16 ladder | NOT_KILLED | Spearman(core6, median_margin)=**+1.00**、(core6, frac<0.005)=**−1.00**，均 exact p=**0.0028**（n=6 下限 1/360） |
| **Qwen 跨家族复现 n=6** | **GENERAL_CLAIM_KILLED** | 同指标 ρ=**+0.43 (p=0.42)** / **−0.49 (p=0.36)**；符号对但强度崩到不显著。per-rung margin 分布在 damage 上**非单调**（step20000 的 median 小于 step2000；scratch14L 大于任何 f12k2 rung） |

OLMo-2 六 rung 是 base_full / shortgpt16@200k / keep14@200k / keep12@124k / keep10@83.5k / keep8@121k；shortgpt16 天然嵌入 fragility 序（core6 0.6233、median margin 0.1165 都落在 base 与 keep14 之间）。

**Why:** OLMo-2 的 ±1.00 是真的（`evidence/B04_6rung_bs16_analysis.json` 在盘上），但**只在 OLMo-2 成立**。Qwen 上 aggregate core6 排序仍复现，**per-item margin compression 不复现** → 「damage → per-item margin 压缩」不是 family-general 性质。

**novelty check 已经做完了**（`novelty_check_2026_08_09` = **`hold_in_backlog`**），不是待办：最近 prior work 是 Tropeano et al. TMLR 2026 (arXiv:2606.24970)，用 ECE/Brier aggregate calibration + attention-only prune 无 heal，**测量族不重叠故未被 preempt**；但 OLMo-2-only 的 scope 对独立 paper 太窄。

**How to apply:**
- **不要**把 ±1.00 p=0.0028 当 general finding 引用。正确表述必须带 scope：「on the OLMo-2-7B keepN+shortgpt16 ladder」。
- 复活条件（`resurrection_conditions`）：(1) 第二个确认家族，**不能是 Qwen**（已失败）；(2) a-priori 的 mechanism-level 假设，解释为何该压缩 family-general。
- 推荐归宿（`recommended_home`）：折进 Paper B methods appendix 或 A01 null-cal spin-out，**不作独立 paper**。
- Provenance：`proposal/backlog/B04-eval-fragility-incubator/{DIRECTION_A_VERDICT.md, DIRECTION_A_QWEN_VERDICT.md, NOVELTY_CHECK.md, STATUS.json}` + `evidence/B04_{6rung,Qwen_6rung}_bs16_analysis.json`；脚本 `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_{5rung,qwen_6rung}.py`；per-item 在 `.73:/apdcephfs_zwfy6/.../olmo2_downstream_results/7B_*_bs16/per_example_*.jsonl`（每 rung n=17195 assert 过）。

**教训**：把一个方向记成「ESTABLISHED」之前，必须确认**所有已跑的 gate 都读过**，不只是支持性的那个。我当时只记了 NOT_KILLED，漏了同日的跨家族 KILLED——这正是 [[prior-work-differentiate-dont-abandon]] 的反向失误：不是过度悲观，而是**过度乐观地漏读判死证据**。

相关：[[cluster-two-disks-not-shared]]、[[same-harness-runs-bit-identical]]、[[prior-work-differentiate-dont-abandon]]。
