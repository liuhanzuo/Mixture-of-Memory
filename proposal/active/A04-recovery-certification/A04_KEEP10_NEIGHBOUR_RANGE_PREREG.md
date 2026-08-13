# A04 — PRE-REGISTRATION: a SECOND arm for the neighbour-range test (keep10+fresh2, 500-step triple)

**Written 2026-08-13 BEFORE any keep10 checkpoint was scored and BEFORE any number
existed.** Committed on its own so the git timestamp proves the readings were fixed in
advance. Nothing below may be edited after the first number lands; the verdict document
is a separate file.

---

## 1. What is under test, and why one arm was not enough

`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` (Leg A) measured, on **one arm** (keep8+fresh2),
the range of the NI margin across three checkpoints **500 steps apart**. Six ranges
resulted (2 clusters × 3 decision axes + demoted nq_open). **Exactly one crossed the
noise gate**: triviaqa in the clean cluster, **1.1202 pp = 1.70× `E[range | pure noise]`**.
`A04_GATE_DESIGN.md` §2.0.2 — the neighbour precondition on any reported accept — rests on
that single cell, and §2.5's proposed tolerance ("≈1.2 pp on triviaqa, ≈0.35 pp elsewhere")
is explicitly labelled a **one-arm estimate to be widened if a second arm is ever measured**.

This is that second arm. `outputs/olmo2_probe2_7B_keep10fresh2/` holds
**step89000 / step89500 / step90000** — a 500-step triple that has never been scored on any
axis, on a **third damage level** (keep_front=10, so 12 layers; keep8 = 10 layers,
keep12 = 14 layers). It is disjoint from the keep12 11-checkpoint trajectory concurrently
running on `.73` (different arm, different steps, different `output_name`).

## 2. The three questions, and what each possible answer means

### Q1 — per-axis range and gate verdict

For each of `popqa / triviaqa / nq_open / mmlu_content`, compute the NI margin at each of
the three steps, then `range = max − min`, and compare against the **item-noise range
floor**.

### Q2 — does keep8's one supra-noise range replicate?

keep8's single supra-noise range was **triviaqa, 1.1202 pp**. The pre-committed readings:

| keep10 triviaqa outcome | Reading — FIXED NOW |
|---|---|
| range **> gate** and ≥ ~0.5 pp | "adjacent-500-step margins can move a pp-scale amount" is **arm-independent**; §2.5's tolerance is corroborated on a second, differently-damaged arm and may be stated as a property of *these heal runs* rather than of keep8. |
| range **> gate** but < 0.5 pp | the phenomenon replicates in **kind** but not in **size**; the §2.5 tolerance stands as keep8-specific and the general statement must be "supra-noise but arm-dependent in magnitude". |
| range **≤ gate** (fails the gate) | **keep8's 1.1202 pp is an isolated cell.** The §2.0.2 precondition retains its logical force (one counterexample still makes single-checkpoint accepts unsafe) but **loses its claim to generality**, and §2.5's per-axis numbers must be labelled *keep8-only, not reproduced on keep10*. **This weakens our position and will be written exactly that way.** |

Note the asymmetry deliberately: **a non-replication cannot rescue the claim and will not be
spun as "consistent with noise, so no problem".** If keep10 fails the gate on every axis, the
headline sentence of the verdict document is *"the keep8 triviaqa range is a single cell that
did not reproduce on a second arm"*.

### Q3 — the same data restated in Heineman et al.'s unit

`arXiv:2508.13144` (**NeurIPS 2025 Spotlight**, OpenReview `sAFottNlra`; DBLP has CoRR only
— do not read DBLP as the venue) defines benchmark noise as the **relative standard
deviation of the accuracy over the final *n* intermediate checkpoints**:

```
Rel.Std.(m) = sqrt( Σ_i (m_i − m̄)² / (n−1) ) / m̄
```

Their Table 4 reports this for OLMo-2 1.5B/7B/13B/32B over the **final 30 checkpoints at
1000-step spacing**. The OLMo-2 **7B-4T** values relevant to us, as extracted in
`proposal/shared/literature/MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md` §P1
(`pdftotext -layout -f 23 -l 24` of v1): **TriviaQA 0.003, MMLU 0.023**. PopQA and NQ-open
are not in their suite, so those two axes get **no comparator and must be left blank** — not
filled with a nearby task.

The hypothesis this addresses: **are damaged (pruned-and-healing) arms noisier than intact
models of the same family and scale?** If yes on TriviaQA across *two* arms, that is a
publishable empirical point that Heineman cannot make (they have no injured models). If our
arms are comparable to theirs, then A04's contribution on this axis reduces to the
**normative** part (the equivalence-decision argument: in a non-inferiority test, neighbour
noise is a one-sided free option) with no accompanying empirical claim.

**HARD CONSTRAINT, imposed by the dispatch and binding on the verdict document:** this is
**n = 3 at 500-step spacing on a damaged arm** versus **n = 30 at 1000-step spacing on an
intact model under the OLMES/OLMo-2 protocol**. Different *n*, different spacing, different
harness, different metric conventions, different model condition. It is a **cross-protocol
hypothesis, never an equal-footing comparison**, it must be visually separated in the
document with the asymmetry stated in the same table caption, and **it may not be tabulated
as if the two numbers were measured together.** A ratio may be quoted only with "n=3 vs n=30,
different protocol" attached in the same sentence.

Note in advance: at n = 3 the sample SD is itself extremely noisy (the χ distribution with
2 df has a relative SD of ≈52 %), so a rel.std ratio of ~2× would be uninformative even
before the protocol mismatch. Only an order-of-magnitude gap would mean anything.

---

## 3. Definitions — fixed now, no post-hoc choice

### 3.1 "Crossing the gate" (`range_exceeds_item_noise`)

Verbatim the keep8 definition, via the **imported** `range_report` from
`a04_neighbour_variability.py` — not reimplemented:

```
E[range of k iid N(0, σ)] = (k=3) 3/√π · σ = 1.6925687506432689 · σ
expected_range_if_pure_noise_pp = 1.6925687506432689 × mean(per-cell bootstrap SE)
range_exceeds_item_noise        = (range_pp > expected_range_if_pure_noise_pp)
```

`σ` is the **mean over the three checkpoints of each cell's own paired-item bootstrap SE**,
where the SE comes from the *imported* `ni_rule` output exactly as keep8 computed it:
`SE = (diff_mean_pp − diff_lower95_one_sided_pp) / 1.6449`. Same `N_BOOT = 10000`, same base
`SEED = 0`, same tie convention `split`, same anchor. **This is the same σ pipeline as
`a04_neighbour_variability.json`; it is not re-derived and not swapped for a different
variance estimate.**

A range that does not exceed the floor **is not a measured gap** and will not be quoted as
one, per §2.3 of the keep8 verdict.

### 3.2 Margins are computed, never inferred

The margin is `diff_lower95_one_sided_pp + delta_pp` from the **imported** `ni_rule`, with
nulls from the **imported** `build_nulls(anchor)` and `Δ = 0.10 × residual(intact)` from the
imported guard. **No margin will be obtained by subtracting a recorded null from a recorded
accuracy** — that shortcut has produced three wrong numbers in this workstream today, the
worst underestimating an mmlu range by 3.0×.

### 3.3 rel.std, and which quantity it is taken over

`stdev(acc, ddof=1) / mean(acc)` over the **three raw accuracies** (not margins, not
deficits) — because that is what Heineman computes. Reported for all four axes; the
comparator column is populated only for triviaqa and mmlu. Also reported for keep8 cluster 2
and keep8 cluster 1 recomputed from the archived JSON, so the two arms are in one unit.

⚠️ **Their "MMLU" is standard letter-choice MMLU; our decision axis is `mmlu_content`
(content-continuation scoring, `content_norm_acc`).** Different interfaces on the same items
— `letter_acc` and `content_norm_acc` disagree on 40.1 % of items at the anchor
(`7B_base/summary.json:letter_vs_content_norm.agreement = 0.5994`). The document will name
this mismatch where the two are placed side by side. `letter_acc` is additionally recorded
as the interface-matched secondary so a reader can see both.

### 3.4 Which cluster is "clean"

Verified from the training log **before** scoring, not after: the keep10 triple must be
inside **one** process. `logs/olmo2_7B_keep10fresh2_resume200k_73.log` shows a **single**
`[resume] loading ckpt ... step86500.pt` banner at 03:57:09 (1 occurrence in the file), then
`saved ... step89000.pt` 08:44:52 (line 154), `step89500.pt` 09:42:19 (line 182),
`step90000.pt` 10:39:43 (line 210), with the process dying at 11:15 on a TCPStore error —
**after** all three saves. So: no seam, one loader, continuous data order.
This is the keep8-cluster-1 trap (§1.2 of the keep8 verdict) checked and cleared **in
advance**; had it failed, the triple would have been reported as seam-straddling rather than
silently used.

---

## 4. Protocol — must match, and is verified from the invocation not the artefact

| Field | Frozen value | How confirmed |
|---|---|---|
| closed-book batch size | **32** | driver echoes `DRIVER START ... cb_bs=32` + per-axis `START ... bs=32`; parsed and gated by the imported `protocol_asserted` |
| mmlu batch size | **16** | same |
| `add_bos` | **False** | in `summary.json:meta`, asserted `is False` — **never `is not True`**, which passes on `None` |
| `max_new_tokens` | **32** | `summary.json:meta`, generative axes |
| `chat_template` | **False** | structural: neither harness has a chat-template code path |
| shards | **8**, index set **exactly {0..7}** | `shard_integrity_report`: index *set*, exact merged n, 0 duplicate item_ids, 0 nan |
| anchor | vanilla `models/OLMo-2-1124-7B` = `{mmlu: 7B_base, cb: base_full, nq: base_full_nqopen}` | **imported** `ANCHOR`, never redeclared; `full32_step25000` forbidden (guard G2) |
| harness md5 | `eval_olmo2_closedbook_qa.py 2ed41993…`, `eval_olmo2_mmlu_content.py fe4a62db…` | verified identical to keep8/keep12 runs |

`summary.json:meta` records **neither** batch size **nor** chat_template
(`A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md`), which is why the driver log is the evidence.
Batch size is not free: bs32→bs48 flipped 12/14267 popqa and 10/3610 nq_open items
(`full32_rescore_v2_20260812.sensitivity_bs48_probe`).

## 5. Node, and the numpy caveat stated in advance

**All scoring and all bootstrap on `.82` (8×H20, zwfy6), `numpy 2.4.6`, torch 2.13.0.**
Verified 8×0 MiB / 0 % / no compute processes before launch; the driver refuses to start if
> 8000 MiB is held.

`Generator.multinomial` differs in **19/10000 rows** between numpy 2.4.6 (`.82`) and 2.5.1
(`.73`/`.104`/`.21`), max observed margin drift **0.005294 pp**, triviaqa only
(`memory/numpy-version-split-breaks-cross-node-bootstrap`). The **keep8 Leg A numbers this
work compares against were themselves published from `.82`/2.4.6**, so this comparison is
*within* one numpy — but the drift is stated because (a) a future re-run on `.73` will see
4th-decimal disagreement, and (b) it is 10.6× looser than the 5e-4 pp hard-fail in
`a04_keep14_trajectory_ni.py`. **It may not be invoked to explain away any move larger than
~0.006 pp.**

## 6. Naming, isolation, and what will not be touched

- `output_name` prefix: **`A04_7B_keep10f2_NBR_step{89000,89500,90000}`** (+ `_nqopen`).
  Verified **zero** existing dirs match `*NBR*` or collide with any keep10 name, and no
  collision with `.73`'s live `A04_7B_keep12f2_*`.
- `.73` (keep12 11-ckpt trajectory), `.104` (paperC Qwen3 heal), `LOCAL`/`.21`
  (SparseForge #246): **not touched, not inspected for free GPUs, not written to**.
- Bootstrap offsets: `arm_index` base **700**, guard `SEED+4700`, intervals `SEED+4900` —
  mechanically checked disjoint from every archived `bootstrap_offsets` block in
  `evidence/` (currently 203/300/301/400-408/500-503, guards 700/1700/2700/3700,
  intervals 900/1900/2400/2900/3900) by the imported-in-spirit `assert_seeds_disjoint`
  pattern, executed as code. **No archived number may be perturbed by this run.**
- `STATUS.json`: **append one new key** `keep10_neighbour_range_20260813`. **No existing key
  of the 39 may be modified.** Verified by comparing the pre/post key list and by asserting
  every prior key's value is unchanged.

## 7. What this run may NOT conclude, fixed in advance

- ⛔ **Not seed variance.** Three checkpoints of one optimisation are **not replicates**;
  their spread is heal progress + data order. No 7B `sd_run` exists or is reconstructible.
- ⛔ **Not a rung comparison.** keep10 (12 layers) vs keep8 (10 layers) vs keep12 (14) vs
  keep14 (16) are **different architectures** trained on **two different corpora** with
  **unequal step counts** (`STATUS.json:warning`). Only the **ranges** — a within-arm
  quantity — are compared across arms. **Absolute margins are never tabulated as a ladder.**
- ⛔ **Not harness noise.** Same-code re-runs on a fixed checkpoint are bit-identical
  (`full32_rescore_v2_20260812.correction_to_the_jitter_premise`). Item-sampling variability
  is a different thing and is what the gate quantifies.
- ⛔ **Not an equal-footing comparison with Heineman et al.** (§2, Q3). n=3 vs n=30,
  500 vs 1000-step spacing, damaged vs intact, our base protocol vs OLMES.
- ⛔ **No K1/K2/K3 clause** — those are defined over the pre-registered **1B** arm set.
- ⛔ **No accept may be claimed** for any keep10 checkpoint on the strength of a range;
  the range is a **checkpoint-selection** quantity.

## 8. Cost, declared before spending

3 checkpoints × 4 axes on 8 GPUs. keep8's comparable Leg B (3 ckpts, 4 axes, `.82`) took
1131 s + 447 s ≈ 0.44 GPU-h/ckpt; keep10 is a 12-layer model (vs shortgpt16's 16) so
expected **≈ 1.0–1.5 GPU-h total**, one node, eval only, zero training. `gpu_h_spent` will
be recorded as **driver wall-clock × 8**, and the analysis is CPU-only.
