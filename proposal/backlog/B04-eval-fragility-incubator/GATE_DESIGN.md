# B04 — Kill gate + next gate (written 2026-08-14, PRE-DATA, 0 GPU)

> **Why this file exists.** B04 sat at `ready_cpu` with the diagnosis "kill gate undefined".
> That diagnosis was half right. A four-clause completion gate *did* exist at
> `PROPOSAL.md:31-38` — it was simply never lifted into `STATUS.json`, so
> `ready_queue.py` could not see it. But three of its four clauses had **no numeric
> threshold**, and clause 1 had already been measured UNTESTABLE. So the honest fix is
> not "copy PROPOSAL.md across" — it is to keep the one clause that survived, retire the
> two that cannot be tested, and add the one clause that can actually kill B04.
>
> Everything below was computed at **0 GPU** from files already on wzc1.
> `STATUS.json.prereg_measured_constants_2026_08_14` holds the machine-readable copy.

---

## 0. What changed, and why each change

| PROPOSAL.md clause | Fate | Reason |
|---|---|---|
| 1. finish 6/6 bs ladder | **superseded** | `status/PAPERF_BS_LADDER_VERDICT.md` = `FLIP_RATE_MONOTONE: UNTESTABLE` at 2/6 rungs. Worse, `status/PAPERB_WITHIN_DISK_FLOOR_V3.md` shows same-driver same-arch re-runs are **bit-deterministic (0 flips)** — a bs8-vs-bs16 flip ladder has a structurally near-zero denominator. Cannot carry a claim. |
| 2. "exact test 成立" | **kept, thresholded** | Now: ρ must be exactly +1.0000 at exact two-sided p = 0.0028 (the n=6 floor, 2/720). Already PASSES. |
| 3. "LOO margin model beats constant-rate null" | **replaced by a floor test** | The LOO/constant-rate framing was never operationalised and needs a flip endpoint that clause 1 just showed is untestable. Replaced with: full damaged-ladder range ≥ 6·σ̂. Already PASSES at 40.3·σ̂. |
| 4. second nuisance (torch/GPU arch) | **withdrawn as a kill clause** | Crossing arch is exactly what `LIFECYCLE_SCHEMA.md §3` forbids for same-harness reproduction, and the within-arch term is *identically zero*. Keeping it as a kill clause would let a stack-version artefact kill a real finding. Demoted to a reported robustness check. |
| — | **NEW clause 5: budget discrimination** | The only clause that can still kill B04. See §3. |

---

## 1. The primary metric is changed, pre-data, and here is why that is legal

`matched_ladder_requirements[6]` says: *"reuse the already-designated PRIMARY metric
frac(margin<0.005); do not pick post hoc among the four thresholds."* Its **intent** is to
forbid threshold shopping after seeing a result. Honoured — by fixing the primary **now**,
before the new arms exist, and recording the reason.

The designated primary **does not clear its own noise floor**. σ̂ comes from the seed pair
(`keep14_s42` vs `keep14_s1234`, damage depth *and* heal step held exactly constant),
divided by `E[range of 2]/σ = 1.1284`:

| metric | σ̂ | ladder range | R = range/σ̂ | adjacent gaps > 2σ̂ |
|---|---|---|---|---|
| **median_margin** | 0.000541 | 0.036899 | **68.26** | **4/5** |
| frac<0.005 *(old primary)* | 0.004329 | 0.016807 | 3.88 | **0/5** |
| frac<0.001 | 0.002062 | 0.003780 | 1.83 | 0/5 |
| frac<0.010 | 0.004072 | 0.032277 | 7.93 | 1/5 |

The ρ = −1.00 on `frac<0.005` is carried by rank **order** over differences that are
mostly **not resolved** — 0 of 5 adjacent gaps clear 2σ̂. `median_margin` is resolved.
So: **`median_margin` is primary**; the three `frac` metrics are reported and explicitly
underpowered.

A ratio-of-two-ranges caveat applies to the k in `E[range of k]/σ`: 1.1284 at k=2, 1.6926
at k=3, 2.8472 at k=8. Using 1.0 understates the floor by 11.4% and **can flip the
boolean** (`memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md`).

---

## 2. Two ladders exist and they are not the same ladder

The zwfy6 ladder behind `evidence/B04_6rung_bs16_analysis.json` has **keep12 @ step124000**.
The wzc1 ladder used here has **keep12 @ step111500**. Consequences:

- `Spearman(core6, heal_steps)` = **+0.6669** (wzc1) vs **+0.8721** (zwfy6). Both real.
  **Always name the ladder when quoting either.**
- `PROPOSAL.md:9-15` disagrees with the evidence JSON on **every rung** of `median_margin`
  (base 0.124594 vs 0.131806; keep8 0.075801 vs 0.094933) and on the p (0.0167 vs 0.0028).
  The JSON side reproduces from disk; my independent wzc1 recompute of keep14@200k gives
  0.107890 against the JSON's 0.107934. **PROPOSAL.md's table must be marked superseded
  before any threshold is quoted against it** (G0 step (c)).
- The zwfy6 `_bs16` dirs are not mounted from this node. Per the two-disk rule that is
  *unverified from here*, **not** *gone*.

---

## 3. Clause 5 — the live kill clause

**The problem.** B04 attributes margin compression to **damage depth**. But on the wzc1
ladder, `Spearman(core6, heal_steps) = +0.6669` across the 5 damaged rungs (steps
200000/200000/111500/83500/121000). Heal budget is a live alternative explanation, and it
is *the exact defect that retired the Qwen leg*. The OLMo leg has only ever **disclosed**
this confound — never **tested** it.

**The test.** Measure the budget slope directly **at fixed damage**, then compare it to the
damage-driven range it would have to explain away:

```
beta_budget = OLS slope of median_margin on heal_step
              over the 5 fixed-damage points (keep_front=14, n_fresh=2, seed=1234)
phi         = |beta_budget| * 116500 / 0.021820
                                ^^^^^^   ^^^^^^^^
                    damaged-ladder        damaged-ladder
                    heal-step span        median_margin range
```

| φ | verdict | in absolute margin units |
|---|---|---|
| φ ≤ 0.30 | **PASS** → family ladder authorised | excursion ≤ 0.006546 (\|β\| ≤ 5.6189e-08 /step) |
| 0.30 < φ < 0.60 | **NARROWED** → joint damage+budget claim, spend still unauthorised | — |
| φ ≥ 0.60 | **KILL** → fold into Paper B methods appendix | excursion ≥ 0.013092 (\|β\| ≥ 1.1238e-07 /step) |

**Why 0.60.** At φ ≥ 0.60 the nuisance factor explains more of the observed range than the
factor of interest — precisely the standard `status_note_2026_08_10.disclosure` already
applied to retire the Qwen leg. **Why 0.30.** `6σ̂/0.021820 = 0.1486`, rounded up to ~2×;
a PASS then means the budget excursion is not merely resolvable-but-small but small *with a
2× margin over resolvability*.

### 3.1 This kill is reachable, not decorative

Adversarial pre-check, 0 GPU. The only fixed-damage budget ladder that exists anywhere in
this project is the Qwen `f12k2/14L` cell (steps 2000/20000/200000):

| step | core6 | median_margin |
|---|---|---|
| 2000 | 0.384821 | 0.107388 |
| 20000 | 0.446598 | 0.095181 |
| 200000 | 0.463169 | **0.133933** |

β = +1.3407e-07 /step → rescaled to 116500 steps → Δ = 0.015619 → **φ = 0.716**, *above the
kill line*. It is also **non-monotone** in budget. So the single most relevant empirical
precedent **predicts KILL**. This gate can fail.

### 3.2 Denominator guard (φ is a ratio)

- **(a)** Denominator = damaged range 0.021820. Admissible only if ≥ 6σ̂ = 0.0032435.
  Measured 6.73× the guard → PASS.
- **(b)** If the denominator were ≤ 0 or below 6σ̂, φ is **UNDEFINED** — not large, not
  small. Gate returns `DENOMINATOR_UNRESOLVED`, which **blocks the spend exactly as a KILL
  would**: a ratio against an unresolved range cannot license 244–2560 GPU-h.
- **(c)** σ̂ is itself a denominator in the floor check and is ill-defined at n=1. Requires
  ≥ 2 arms holding damage depth **and** heal step exactly constant, with the k-appropriate
  divisor.
- **(d)** If σ̂ = 0 — the *normal* outcome for same-driver re-runs — return
  `FLOOR_UNMEASURABLE`, which **does not pass**: it means the contrast is not a real
  nuisance contrast. The seed pair is safe because it varies init seed (σ̂ ≠ 0 on all four
  metrics).

---

## 4. Next gate G1 — 1.08 GPU-h, and it is the cheap version of a 2560 GPU-h question

**Single variable: heal step count.** `keep_front=14`, `n_fresh=2`, `seed=1234`, base model,
harness, shard count, batch size, `--save_per_example`, arch — all held fixed.

Arms (4 new + 1 already measured), all present on wzc1, verified 2026-08-14:

```
outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step25000.pt    48724473567 B
outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step50000.pt    48724473567 B
outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step100000.pt   48724474298 B
outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step128000.pt   48724474298 B
olmo2_downstream_results/keep14_s1234_step200000_sv181         ALREADY DONE, 0 GPU
```

**These 4 intermediate checkpoints are the find that makes this cheap.** No prior B04 audit
noticed them. They turn "is this damage or budget?" from a 244–2560 GPU-h re-heal question
into a **1.08 GPU-h eval** question — a 226×–2370× reduction.

### 4.1 Do G0 first (0 GPU, blocking)

1. Recompute the 6-rung wzc1 ladder margins. Either run
   `scripts/enrich_per_example_normscores.py` on the 6 dirs (idempotent, self-verifying,
   does not touch `summary.json`; HF cache present at `data/hf_datasets_cache/`), **or**
   transplant `norm_lens` by `item_id` read-only from the enriched donor —
   `norm_lens` is a property of the dataset, never of the model. I validated the transplant
   against the natively-enriched donor: identical on all four metrics
   (`median_margin` 0.10789000000000004 both ways).
2. Emit `evidence/B04_wzc1_floor_analysis.json` (σ̂, R, adjacent-gap counts, all 4 metrics).
3. **Reconcile `PROPOSAL.md` against the evidence JSON** (§2). No threshold may be quoted
   against two live number sets.

### 4.2 Hard aborts

- `PROTOCOL_VIOLATION` if any arm is not 8/8 shards, or pooled `n_scored != 17195`, or
  `n_nan != 0`. Asserting `n_scored` **per task** — not just `n_nan` — is required by
  `memory/same-harness-runs-bit-identical.md`.
- **Field symmetry, mandatory.** All 5 arms must carry `norm_scores`/`norm_lens` on all 6
  tasks or the driver must refuse to compute. An asymmetric-field paired comparison already
  produced a **56× artefact** once: a reported 34.7% flip rate whose true value was 0.62%
  (`status/PAPERF_ACCNORM_VERIFIED.md:43-67`), because one side had `norm_scores` and the
  other did not.

### 4.3 Free validation hook

`7B_keep14_step200000_wzc1_v2` and `keep14_s42_step200000_sv181` are **bit-identical**
(verified: 0/17195 `acc_norm_score` mismatches, 0/17195 `option_scores` mismatches, all six
`acc_norm` equal to 8 dp). The floor and the ladder therefore share a common point, so σ̂ is
directly commensurable with the ladder's own units — no cross-protocol rescaling. Any future
re-run of that rung that does *not* reproduce bit-identically is a **protocol break, not
noise**.

---

## 5. Architecture requirement: sm_100, and it is load-bearing

σ̂ and the entire comparison ladder are **sm_100** (LOCAL/`.212`, B200).
`paperB/SEEDVAR_KEEP14_VERDICT.md` line 3: *"Node: LOCAL (8×L20A, wzc1) only"* — and the
`L20A` name string is a display bug; the real hardware is B200/sm_100
(`memory/l20a-name-string-is-really-b200-sm100.md`; capability, SM count and HBM all say
B200). New arms **must** be sm_100, or the run-to-run term and the hardware term are
confounded and even a FAIL is uninterpretable (`LIFECYCLE_SCHEMA.md §3`). The zwfy6/sm_90
`_bs16` dirs cited at `DIRECTION_A_VERDICT.md:97` are **not admissible** as G1 comparators.

---

## 6. Cost, with measured anchors

| leg | cost | anchor |
|---|---|---|
| G0 | **0 GPU-h** | pure CPU on existing `per_example_*.jsonl` |
| **G1** | **1.08 GPU-h** | `logs/sv181_main.log:5-6` — core6 for `keep14_s42` ran 01:12:18→01:14:19 = **121 s** on 8 GPUs = 0.269 GPU-h/rung; × 4 rungs. Driver `scripts/_run_paperB_keep14_seedvar_local.sh:116-125`. Same harness *and same arch* as the comparator. |
| downstream matched family ladder @20k | 244 GPU-h | `logs/qwen3_armB_f12k2_20k.log` final step line **1.37 s/step** × 20000 × 8 / 3600 = 60.9 GPU-h/rung × 4 |
| downstream matched family ladder @200k | 2560 GPU-h | `logs/qwen3_armB_200k_local.log` final step line **1.44 s/step** × 200000 × 8 / 3600 = 640 GPU-h/rung × 4 |

Cross-check on the 200k anchor: that log's wall clock is 2026-07-13 20:15:57 → 2026-07-16
18:36:34 = 70.3 h × 8 = 562 GPU-h, but it **resumed at step 48000** (`[resume] continue @
step=48000`), i.e. 152k fresh steps → 1.33 s/step effective. Consistent with 1.44 s/step.

**The downstream spend is gated behind G1 and is not authorised.**

---

## 7. Read-out pre-registration

Primary read-out: `median_margin` at `ckpt_step ∈ {25000, 50000, 100000, 128000, 200000}` of
`olmo2_probe2_7B_keep14fresh2_seed1234`. Decision statistic: β_budget over **exactly those
5 points**, and φ.

That directory also contains step153500/165000/170000/175000/180000/185000/190000/195000/
199000/199500. **Those are not part of the read-out.** Extending n until the slope crosses a
threshold would be the paperC `--max_steps` error in a new costume — there the
pre-registered read-out was step121000, not the 200000 the flag mentions.

No metric substitution: a φ computed on any `frac` metric cannot overturn the primary.

---

## 8. Mandatory disclosure

Any quotation of `Spearman(core6, median_margin) = +1.00` must print beside it:
(i) `Spearman(core6, heal_steps)` **for the same ladder, ladder named** (+0.6669 wzc1 /
+0.8721 zwfy6); (ii) σ̂ and R; (iii) φ from clause 5.
Standing requirement from `STATUS.json.olmo_only_finding.budget_caveat_2026_08_10`.

---

## 9. Novelty is NOT the blocker

`novelty_check_2026_08_09.verdict = hold_in_backlog`, which `ready_queue.py`'s
`VERDICT_CLEARED` accepts; `NOVELTY_CHECK.md` is on disk. No prior work satisfies the
3-part `kill_definition` (depth-prune-**then-heal** ≥3 rungs **and** per-item acc_norm
margin distribution **and** co-variation with aggregate score). The closest, Tropeano 2026
TMLR, uses ECE/Brier aggregate calibration on attention-only pruning **without heal** — a
disjoint measurement family. `RELATED_WORK.md` absence blocks **promotion**, not GPU.

Per project rule: **a direction is killed only by its own experiment gate, never by "there
exists similar literature."**
