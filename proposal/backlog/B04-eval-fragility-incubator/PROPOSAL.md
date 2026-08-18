# B04 — Evaluation Fragility under Model Damage

> ## ⛔ SUPERSEDED NUMBERS — 2026-08-17 (MAIN, 0 GPU). READ BEFORE QUOTING ANY FIGURE BELOW.
>
> **The numeric table in "已成立" (§已成立, below) is SUPERSEDED and must not be quoted.**
> This banner discharges `STATUS.json.next_gate.prereg_G0_first_0_GPU` **step (c)** and
> `remaining_blockers_after_this_design[2]`, both of which require this file to be marked
> superseded **BEFORE any external write-up quotes a threshold against it**.
>
> **Authoritative replacements, in this order:**
> 1. `evidence/B04_wzc1_floor_analysis.json` — the **wzc1 sm_100 ladder**. This is the ladder
>    the kill gate's φ is defined against (comparator range **D = 0.021820**, σ̂ = 0.000541).
> 2. `evidence/B04_6rung_bs16_analysis.json` — the **zwfy6 ladder**. Cited by
>    `prereg_G0_first_0_GPU` as the reconciliation target.
> 3. `STATUS.json.kill_gate` — thresholds and the mandatory co-disclosures.
>
> ### The three ladders are DIFFERENT MEASUREMENTS, not one number computed three times
>
> This is the part `prereg_G0_first_0_GPU` step (c) does not spell out, and it matters:
> the rows below are **not** a miscopy of either JSON. They are a **third, earlier ladder**
> (`evidence/accnorm_margin_verified.md`, 2026-08-08, task #201) whose rungs sit at
> *different heal steps*. Re-verified by recomputation 2026-08-17:
>
> | median_margin | base | keep8 | ladder identity |
> |---|---|---|---|
> | **§已成立 below (SUPERSEDED)** | 0.124594 | 0.075801 | `accnorm_margin_verified.md`, 2026-08-08 |
> | `B04_6rung_bs16_analysis.json` | 0.131806 | 0.094933 | zwfy6; keep12 rung = step **124000** |
> | `B04_wzc1_floor_analysis.json` | 0.131678 | 0.094779 | wzc1 sm_100; keep12 rung = step **111500** ← **φ's comparator** |
>
> So "PROPOSAL.md disagrees on every rung" is true but under-describes the defect: the old
> row is not *wrong arithmetic*, it is **a different ladder** whose provenance this file never
> named. `B04_wzc1_floor_analysis.json.ladder_identity_warning` already records that quoting
> any Spearman(core6, heal_steps) **requires naming its ladder**; the same now applies here.
>
> ### The `p=.0167` line is a METRIC MIX-UP, not a stale value
>
> `prereg_G0_first_0_GPU` describes this as a disagreement "on the p (0.0167 vs 0.0028)".
> Recomputation 2026-08-17 (exact two-sided permutation over all 720 orderings, n=6) shows
> the cause is more specific — **`.0167` belongs to a different threshold than the line it is
> printed on.** On the very ladder §已成立 uses:
>
> | metric | ρ | exact p |
> |---|---|---|
> | median_margin | +1.000000 | 2/720 = **0.002778** |
> | frac<**0.005** | −0.942857 | 12/720 = **0.016667** |
> | frac<**0.010** | −1.000000 | 2/720 = **0.002778** |
>
> The line `Spearman(core6, frac<.005) = -.9429, p=.0167` is therefore **internally correct
> for that 2026-08-08 ladder**. But on the wzc1 and zwfy6 ladders **frac<0.005 reaches
> ρ = −1.000000, p = 0.002778**, and `−0.9429 / 0.0167` there belongs to **frac<0.010**.
> The `-.9429` is not a superseded measurement of the same quantity — on the current ladders
> it is *the wrong metric's* number. Cause: one rung inversion (ShortGPT-16 = 3.286% vs
> keep14 = 3.280%, a 0.006pp gap) that the later ladders do not reproduce.
>
> ### frac<0.005 must NOT be quoted as a headline at all
>
> Independently of which ladder: `kill_gate.primary_metric_choice_is_prereg_not_posthoc`
> **DEMOTED** frac(margin<0.005) pre-data because it **fails its own noise floor** —
> σ̂ = 0.004329, R = 3.88, **0/5** adjacent rung gaps clear 2σ̂. The PRIMARY metric is
> **median_margin** (σ̂ = 0.000541, R = 68.26, 4/5 gaps clear). Two of the four bullets in
> §已成立 are built on the demoted metric.
>
> ### What is NOT retracted
>
> The **direction and the rank structure** survive on all three ladders: median_margin falls
> monotonically with damage, near-tie density rises, ρ = +1.00 at p = 0.0028 for the primary
> metric. The §完成 gate list below is also **superseded but not by this banner** — clause 1
> was retired and clause 4 withdrawn (`kill_gate.clause_1_bs_ladder`,
> `clause_4_second_nuisance`); the live clause is **5**, and its φ is **UNDEFINED** until the
> G1 read-out is filled (`--readout-only` → rc=3 `READOUT_ABSENT`).
>
> Per `LIFECYCLE_SCHEMA.md` sec 0 the original text below is left **byte-for-byte unedited**
> as the dated 2026-08-08 record. Nothing below this banner is authoritative.

## 状态

**INCUBATOR。当前只证明 margin compression，不足以声称广义 fragility。**

## 已成立

正确 `acc_norm` 下，六个 damage rungs：

- median margin：`0.124594 → 0.075801`
- Spearman(core6, median margin)：`+1.00, p=.0028`
- near-tie `<.005`：`2.012% → 4.461%`
- Spearman(core6, frac<.005)：`-.9429, p=.0167`

即：damage 与 decision-margin compression/near-tie density 相关。

## 未成立

- flip rate 随 damage 单调；
- margin 中介 damage→flip；
- 跨 nuisance variable 的复现。

当前 bs8/bs16 仅 2/6 rung：

- base 0.081%
- ShortGPT 0.640%

不能做有效趋势检验。

## 完成 gate

1. 完成 6/6 bs ladder；
2. exact test 成立；
3. LOO margin model 优于 constant-rate null；
4. 第二种 nuisance（torch/GPU architecture）复现。

失败则作为 A01 appendix 的 negative result，不单独成篇。

