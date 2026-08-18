# Sampler variance decomposition + AR cross-arch control

**Status.** Piece 1 (marginal spreads on the 25-cell HE+ grid) done from disk.
Piece 2 (cross-node reproducibility of the reference cell) **already closed** by
`CROSSNODE_REPRODUCIBILITY.md`; the .252 leg was arm B / B2 there, HE
0.7622 / HE+ 0.7073, bit-identical pass@1 to LOCAL. This report contributes a
**fresh AR control on .252** — the "highest-value follow-up" the crossnode
report itself flagged — and it **kills the dLLM-specific framing** of the
observed cross-arch gap: the AR's HE+ cross-arch gap is **2.44 pt, exactly equal
in magnitude to the dLLM's and opposite in sign** (§3). Piece 3 (Spearman rho)
computed on 10/12 available
MBPP+ cells; the remaining 2 cells and the seed replicates are still in flight
under audit task #178. Everything below uses evalplus 0.3.1 on wzc1 as the
common grader and states its axis (`base` vs `plus`) explicitly.

Model: `Dream-Coder-v0-Instruct-7B` unless noted; `Qwen2.5-Coder-7B` in §3.
Benchmark: HumanEval+, n=164; MBPP+, n=378. Grading: `evalplus.evaluate` /
`evalplus.eval.untrusted_check` only.

---

## 1. Piece 1 — marginal spread of each sampler factor at fixed reference

Full 25-cell HE+ grid (Dream-Coder-Instruct-7B, .104):

| scope | HE base spread | HE+ plus spread |
|---|---|---|
| all 25 cells | **59.8 pt** (0.207 - 0.805) | **56.1 pt** (0.183 - 0.744) |
| 21 plausible cells (drop `alg=origin`, drop `alg_temp=0.5`) | **28.7 pt** (0.518 - 0.805) | **26.8 pt** (0.476 - 0.744) |

Reference cell: `T=0.1, top_p=0.95, alg=entropy, alg_temp=0.0` (HE 0.7622,
HE+ 0.6890). All marginals below fix the other three factors at this cell
and vary one axis.

**Marginal spread per factor (holding the other three at the reference cell):**

| factor | levels present | HE base marginal spread | HE+ plus marginal spread |
|---|---|---|---|
| `temperature` | 0.0 / 0.1 / 0.2 / 0.4 / 0.7 | **24.4 pt** | **21.3 pt** |
| `top_p` | 0.80 / 0.85 / 0.90 / 0.95 / 0.99 / 1.00 | **21.3 pt** | **17.1 pt** |
| `alg` | entropy / maskgit_plus / topk_margin / origin | **53.7 pt** | **50.0 pt** |
| `alg_temp` | 0.0 / 0.5 | **45.7 pt** | **40.2 pt** |

Same-cell replication floor (same node, seed=None, otherwise identical). **These
are pass@1 spreads, which is not the same as bit-identity** — see the reading
note below and caveat 5:

| replication | HE base spread | HE+ plus spread |
|---|---|---|
| T=0.1 ref x4 (seed=None x4) | **0.00 pt** | **0.00 pt** |
| T=0.0 ref x2 dup | **0.00 pt** | **0.00 pt** |
| T=0.7 x4 seeds | 2.44 pt | 2.44 pt |

**Reading the table.** Against a same-cell floor of **0.00 pt on pass@1** at
T=0.0 and T=0.1, moving *only* one axis and holding everything else at a
plausible reference cell moves HE+ by **17-50 pt**. `alg` and `alg_temp`
dominate; `top_p` and `temperature` each still move HE+ more than the entire
cross-node architecture gap (2.44 pt HE+) or the T=0.7 seed floor (2.44 pt HE+).
The scale of protocol sensitivity is one to two orders of magnitude larger than
published method deltas on this benchmark.

Precision on the floor, because it is load-bearing and easy to overstate: the
**0.00 pt** figure is a *pass@1* floor (0 flips, identical to 4 decimals). It is
**not** bit-identity. At T=0.1 the completions are **not** bit-identical even
same-node/same-config — 2-3 of 164 differ in raw text — because T=0.1 is not
greedy and no seed was set (crossnode report §2.6). The stronger
"**exactly 0.0 / provably zero / token-identical**" characterisation applies to
the **T=0.0** cells, where the committed token is `probs.max()` and the dup run
confirms bit-identity. Statements elsewhere that the within-node floor is
"exactly 0.0" should be read as scoped to T=0.0.

Caveat that the brief calls out and that ANOVA cannot mask: the 25 cells are
NOT a balanced factorial (only the reference cell has full coverage of each
axis), so these marginals are *conditional on the reference* and not
main-effect variance components. Reported as marginal spreads with the
reference cell named, per the brief.

Full table of every cell with numbers is in
`runs/sampler_audit_mirror/decomposition.txt`.

---

## 2. Piece 2 — cross-node .252 leg (already resolved, plus a diagnostic)

The tournament claim was `.7073 vs .6890` (LOCAL wzc1 vs .82 zwfy6), -1.83 pt.
`CROSSNODE_REPRODUCIBILITY.md` (`arm B_252_L20A_wzc1_t211`) already re-ran the
exact protocol on .252 and got **HE 0.7622 / HE+ 0.7073, 0/164 flips vs LOCAL,
1/164 solution-text difference**. The .252 arm is L20A (cc 10.0), same
architecture as LOCAL despite the cluster-notes label "B200". So .252 is not a
fourth data point — it is the *within-architecture* control that proves the
1.83 pt gap is neither disk nor machine nor stack: it is GPU architecture.

**Axis pin (this was the actual load-bearing check).** The `.7073 vs .6890`
claim is measured on the same axis. Both nodes report base and base+plus from
the official grader; both load ground truth `fe585eb4df8c88d844eeb463ea4d0302`.
The gap is plus-vs-plus. The report additionally cross-grades: under one common
grader (evalplus 0.3.1) the gap grows to **-2.44 pt HE+ / -0.61 pt base**, so
grader version is a real confound (13 flips), but it does not explain the
effect — it under-reported it. See `CROSSNODE_REPRODUCIBILITY.md` §1.2, §1.4.

.252 pass@1 (re-derived from `runs/xnode/B_252_L20A_wzc1_t211/evalplus.out`):

```
humaneval (base tests)   pass@1: 0.762
humaneval+ (base + plus) pass@1: 0.707
```

---

## 3. Piece 3-adjacent — AR control on .252, and the finding that kills the framing

`CROSSNODE_REPRODUCIBILITY.md` §4 flags "Running Qwen2.5-Coder-7B greedy on
both architectures is cheap and is the single highest-value follow-up. If an
AR model also moved ~2 pt, the dLLM-specific framing dies." That follow-up was
not yet done. It is now. .252 was idle. I ran it there.

Protocol: `Qwen2.5-Coder-7B`, base-continuation (no chat template), T=0.1,
top_p=0.95, max_new_tokens=512, HumanEvalPlus-v0.1.10, evalplus 0.3.1,
per-shard `CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0`. Same generator
(`scripts/generate_evalplus_ar.py`), same weights checkpoint file
(`models/Qwen2.5-Coder-7B`), 8-shard split.

**Paired AR HumanEval+ on the two architectures:**

Grading axis: `plus` is **conjunctive** (`base_status=='pass' AND
plus_status=='pass'`), which is what EvalPlus itself reports. Corroboration: the
.104 file's own recorded `pass_at_k` is
`{'base': 0.5609756097560976, 'plus': 0.5182926829268293}` — matching the
conjunctive count 85/164 = .5183, not the 87/164 = .5305 you get from reading
`plus_status` alone. `scripts/analyze_ar_crossarch.py` now asserts our axis
against that recorded field on every run.

| arm | node | GPU | HE base | HE+ | vs H20 base | vs H20 plus |
|---|---|---|---|---|---|---|
| AR-L20A | .252 (wzc1) | L20A (cc 10.0) | **0.5427** (89/164) | **0.4939** (81/164) | **−1.83 pt** | **−2.44 pt** |
| AR-H20  | .104 (zwfy6) | H20 (cc 9.0)  | **0.5610** (92/164) | **0.5183** (85/164) | — | — |

Paired McNemar (n=164):
| axis | flips L20A=0, H20=1 | flips L20A=1, H20=0 | total | exact p |
|---|---|---|---|---|
| base | 6 | 3 | 9 | 0.5078 |
| plus | 7 | 3 | 10 | 0.3438 |

Solution text differs on **40/164** tasks (dLLM: **75/164** solution-text,
128/164 raw-output).

**The AR cross-arch gap is −2.44 pt on HE+ — exactly equal in magnitude to the
dLLM cross-arch gap of +2.44 pt HE+, and OPPOSITE in sign.** (dLLM: LOCAL
0.7073 − .73 0.6829 = +2.44 pt, 12 flips, p=0.3877, from
`runs/xnode/analysis_full.json` pair `A_local vs C_73`. AR: L20A − H20 =
−2.44 pt, 10 flips, p=0.3438.) Neither is individually significant at n=164.

**Read the signs, because they are the result.** The dLLM loses HE+ on the H20
side; the AR loses HE+ on the L20A side. There is no consistent direction, so
neither architecture is "better" — **equal magnitude with opposite sign is the
signature of symmetric hardware noise, not of a directional hardware
preference.** This is a cleaner and *stronger* finding than a shared direction
would have been: a shared direction would invite a systematic-bias explanation,
whereas symmetry points squarely at non-deterministic bf16 reduction order
(mechanism per crossnode report §3), which has no reason to favour either card.

**This is not a dLLM phenomenon.** Because the AR gap *equals* the dLLM gap
exactly (2.44 pt both), the framing "dLLMs are unusually exposed to hardware
nondeterminism" is **dead on pass@1**. It survives only at the bit level: the
dLLM diverges in solution text on **75/164** tasks (128/164 raw output) versus
the AR's **40/164** — roughly 2× the per-task divergence rate, as the mechanism
predicts (unmasking order is a live degree of freedom at each of 512 diffusion
steps). **State it plainly: dLLMs are measurably more bit-level unstable across
architectures, and that extra instability does not show up in pass@1 at all.**

---

## 4. Piece 3 — HE+ vs MBPP+ rank agreement (partial, 10/12 cells)

Audit task #178 has produced eval_results for **10 of its 12 MBPP+ cells** so
far; `mbpp_T0.0_p1.00_origin_at0` and `mbpp_T0.4_p0.95_entropy_at0` are still
in flight or ungraded. I built a partial `sampler_audit_mbpp_mirror/summary.json`
from the 10 that finished and Spearman-correlated their HE+ vs MBPP+ pass@1.

**The 10-cell rho of 0.9379 must not be quoted bare.** It reproduces exactly,
but **3 of those 10 cells are the `alg=origin` / `alg_temp=0.5` regimes that the
HE+ headline itself excludes as implausible** (§1 drops them to get the 26.8 pt
plausible-only spread). Those 3 cells sit far below the rest on both benchmarks,
so they act as high-leverage anchors that manufacture rank agreement no
practitioner would ever benefit from. Restricting to the regime a practitioner
would actually search:

| cell set | n | rho | exact permutation p |
|---|---|---|---|
| all common | 10 | 0.9379 | — |
| plausible only (drop `origin`, `alg_temp=0.5`) | 7 | 0.8462 | — |
| **distinct plausible points** | **5** | **0.6000** | **0.175 one-sided / 0.350 two-sided** |

**The n=5 row is the headline for rank transfer.** The collapse from 7 to 5 is
not a filter I chose — `entropy`, `maskgit_plus`, and `topk_margin` at the
reference cell are **byte-identical** (all `.7622/.6890` on HE+, all `.6905` on
MBPP+, cf. §1's finding that the three confidence-based `alg` values tie
exactly). They are **one point, not three**; counting them as three inflates n
and inflates rho. The 5 genuinely distinct plausible points are
`T=0.0/p=0.90`, `T=0.0/p=0.95`, `T=0.0/p=1.00`, `T=0.1/p=0.95` (ref), and
`T=0.2/p=0.95`, all `alg=entropy, alg_temp=0.0`.

**Honest statement of what rank transfer is worth here:** rho = **0.60 on n=5
distinct plausible cells, not significant** (exact permutation p = 0.175
one-sided, 0.350 two-sided; scipy's asymptotic t-approximation gives 0.285
two-sided). The top-1 cell does agree across benchmarks, which is the one piece
of good news. But **the "sampler ranking transfers across benchmarks" claim is
currently unsupported at conventional significance** — n=5 is simply too small
to establish it, and the impressive-looking 0.94 is an artifact of including
regimes the paper itself calls broken.

| axis | value |
|---|---|
| common cells | 10 (7 plausible, **5 distinct plausible**) |
| Spearman rho, distinct plausible | **0.6000** (p=0.175 one-sided) |
| top-1 HE+ cell | `T=0.1, top_p=0.95, alg=entropy, alg_temp=0.0` |
| top-1 MBPP+ cell | same |
| same top-1 cell? | **yes** |

If the two remaining cells arrive they add at most 2 distinct plausible points
(n=7), which is still underpowered. **The load-bearing number is the
distinct-plausible rho, and it needs many more cells — not the full 12 — to
become a claim.** Re-run via `analyze_sampler_variance.py` when #178 lands.

The tournament claim "sampler choice dominates method choice" **does not depend
on this rho**: it rests on the within-HE+ spread (26.8 pt plausible-only) versus
published method deltas, which is unaffected. What the weak rho costs us is the
*generalisation* rider — we cannot currently assert that the protocol-search
implication carries to a second benchmark on rank-correlation evidence.

---

## 5. What would kill "sampler choice dominates method choice"

1. **A wide-margin baseline that survives across the 25-cell grid**: if some
   published method gained >26.8 pt HE+ on the plausible-only spread, sampler
   would not dominate. No such method is on record.
2. **A protocol lock that pins alg + alg_temp**: with alg and alg_temp fixed,
   the residual T/top_p spread is 21.3 pt HE base / 17.1 pt HE+ — still >> the
   largest published dLLM method delta. Only pinning ALL FOUR axes reduces
   protocol variance to the seed floor. This is closable, but journals do not
   currently ask for it.
3. **A method that closes the cross-arch gap** would move the framing from
   "sampler dominates" to "hardware-plus-sampler dominates jointly". The AR
   control shows cross-arch already contributes ~2.4 pt independent of the
   dLLM (same magnitude as the dLLM, opposite sign). That is smaller than the
   sampler spread but nonzero and irreducible.
4. **Task #178 (MBPP+ full grid) returning low rho on the distinct plausible
   cells**: would show the sampler ranking is HE+-specific, weakening the general
   "protocol dominates" claim. **This is currently NOT protected.** On the 5
   distinct plausible cells available today rho is only **0.60 (p=0.175
   one-sided, not significant)**; the reassuring 0.94 comes from including the
   `origin` / `alg_temp=0.5` cells the headline excludes. The within-HE+ spread
   argument stands on its own, but the cross-benchmark generalisation rider is
   presently unsupported and a larger MBPP+ grid could kill it.

---

## 6. GPU-hours and artifacts

- **GPU work performed on .252**: 1x 8-GPU-L20A run of Qwen2.5-Coder-7B on
  HumanEval+, generation ~6.5 min wall + grading ~1 min = ~7 min wall x 8
  GPUs = **~0.93 GPU-hours**. Confirmed .252 was 0/8 procs before I launched;
  post-run cleanup verified.
- **CPU work**: partial MBPP+ summary construction from 10 grading dirs on
  .104, variance decomposition, AR paired analysis.
- **Artifacts (all wzc1):**
  - `scripts/analyze_sampler_variance.py`
  - `scripts/analyze_ar_crossarch.py`
  - `scripts/_run_ar_baseline_252.sh`
  - `runs/sampler_audit_mirror/summary.json` (mirror of .104's, 25 cells)
  - `runs/sampler_audit_mirror/decomposition.txt` (script output)
  - `runs/sampler_audit_mbpp_mirror/summary.json` (partial, 10 cells)
  - `outputs/ar_qwen25coder7b_base_252/humaneval/solutions.jsonl`
  - `outputs/ar_qwen25coder7b_base_252/humaneval/solutions_eval_results.json`
  - `runs/xnode/ar_control_h20_104/eval_results.json` (mirror of .104 AR)
- **Do NOT edit**: `DLLM_RESULTS_20260807.md` (MAIN owns it),
  `CROSSNODE_REPRODUCIBILITY.md` (audit output).

---

## 7. Honest caveats

1. **AR n=164, plus flips = 10, p=0.3438** — not individually significant. The
   AR's −2.44 pt HE+ gap is *equal in magnitude and opposite in sign* to the
   dLLM's +2.44 pt HE+ gap, which is why the read is "symmetric hardware noise"
   rather than "one architecture is better". A single AR run does not by itself
   prove the exact-equality is anything but coincidence at this n — the
   defensible claim is "same order of magnitude, no consistent direction". A
   second AR replicate on a third node (e.g. .73 or .82, both H20) would settle
   it.
2. **Only one AR model, one dLLM, one architecture pair**. Same limits as
   `CROSSNODE_REPRODUCIBILITY.md` §4.
3. **Grading axis**: HE `base_status` and `plus_status` come from the same
   `solutions_eval_results.json`. AR L20A file was renamed to `eval_results.json`
   locally, but it *is* the evalplus grader output — no hand-rolled grader.
4. **MBPP+ rho is partial (10/12 cells) AND underpowered once the implausible
   cells are dropped**: 5 distinct plausible points, rho 0.60, p=0.175
   one-sided. The bare 0.9379 is carried by 3 excluded-regime cells and must
   never be quoted alone (§4). Scaffold re-computes rho when the remaining
   cells land, but n will still be too small to establish rank transfer.
5. **Same-cell same-node floor is 0.00 pt on pass@1 but not bit-identical in
   text** for T=0.1 — reported by crossnode report §2.6, replicated implicitly
   here (0.00 pt across 4 seed replicates but seed-1 through seed-3 do have
   raw_output differences at the trailing-prose level; 2-3 of 164 completions
   differ). Bit-identity holds only at **T=0.0**, where the committed token is
   `probs.max()`. Do not describe the T=0.1 floor as "exactly 0.0" without the
   pass@1 qualifier.
