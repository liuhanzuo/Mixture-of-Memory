# B03 — READ-OUT RULE (v2), FIRABILITY PRECHECK, COST, AND NOVELTY VERDICT

**Authored 2026-08-16. 0 GPU spent. No job launched, no job killed.**
All 24 H20 cards on `.73` / `.82` / `.104` were measured at **100 % utilisation**
(`nvidia-smi --query-gpu=utilization.gpu,memory.used`, read-only, 2026-08-16) — so nothing here
could have been run even if it had wanted a card.

**Scope note, and it is the most important sentence in this file.** The task that produced this
document was scoped as *"B03's read-out rule does not exist; write it."* **That premise is stale.**
A read-out rule **does** exist on disk: `GATE_PREREGISTRATION.md` (45,062 bytes, mtime
2026-08-15 10:48) plus `STATUS.json:next_gate_executable_20260814` (31,989 chars of JSON) and
`STATUS.json:kill_gate_executable_20260814`. The bare sentinel `STATUS.json:next_gate ==
"NOT_SPECIFIED"` is **still on disk and still says the rule is missing** — because the file is
append-only and the 2026-08-15 pass deliberately did not edit it. Reading the bare key and
concluding "no rule exists" is exactly the failure recorded in
`memory/read-what-the-consumer-reads-not-the-bare-key.md`, and `proposal/ready_queue.py:130`
already resolves the dated key `next_gate_executable_20260814` in preference to `next_gate`.

**So this document does NOT re-write the rule from nothing. It ADJUDICATES the existing rule,
and the adjudication found two defects that make the existing gate UNFIREABLE AS WRITTEN.**
Both are pre-data, both are arithmetic, and both are fixed below. That is a materially more
useful outcome than a second parallel rule would have been, and it is the honest one.

---

## 1. VERDICT UP FRONT

| # | Finding | Consequence |
|---|---|---|
| **D1** | The pre-registered primary test **cannot reach `α = 0.05` at any effect size**, because its own permutation scheme admits only `C(6,3) = 20` distinct statistics, giving a two-sided p floor of **`2/20 = 0.100000`** — not the `2/400 = 0.005000` the prereg claims. | Clause 1 was **UNDECIDABLE BY CONSTRUCTION**. Fixed in §2 (either `S ≥ 4`, or the sign-flip enumeration of §2.3). |
| **D2** | The prereg's `σ_run(K) ≤ 0.2903 pp` pass bound is applied to a `χ²` **upper bound** computed at `df = S−1 = 2`, where the inflation factor is **6.2847×**. That requires a measured `s ≤ 0.046191 pp` — **below all three measured priors in the repo** (0.05547 / 0.06558 / 0.07834 pp). | G1 would return `UNDERPOWERED` **on its most likely outcome**, burning 107.7 GPU-h to learn nothing. Fixed in §2.5. |
| **D3** | The prereg's own mandated self-check (`assert p is a multiple of 1/400`) **cannot detect D1**: under the degenerate scheme p is a multiple of `1/20 = 20/400`, which *is* a multiple of `1/400`. | A guard that passes silently while the test has lost 20× resolution. Replaced in §2.4. |
| **NOVELTY** | **SURVIVES** as a regime-boundary / negative-result question. **DEAD** as a method claim. No paper shows the same thing. | §5. Three venue calls the 2026-08-15 pass could not make **are now closed**; two of its records are **corrected**. |

**Firability, one line:** the gate as pre-registered on 2026-08-15 was **NOT firable**; with the
§2 amendments it **is** firable, and the honest expected outcome is stated in §3.4 — the
literature and our own priors both predict **KILL**.

---

## 2. THE READ-OUT RULE (v2). Amendments to `GATE_PREREGISTRATION.md`, pre-data.

Everything in `GATE_PREREGISTRATION.md` not listed below **stands unchanged and is re-affirmed**:
the quantity `R_know` / `R_ppl` / `D` (§1), the four measured construct nulls (§4.1), the primary
axis `mmlu_content` at `α = 0.05` (§5.1), Holm at `0.016667/0.025/0.05` for the three secondary
axes (§5.2), `Θ_int = 0.10 × residual = 1.0230 pp` (§6.1), `Θ_sep = 0.10` retention units (§6.2),
`c_2 = 1.1283791670955126` and `k = 2` for the range floor (§4.4), the three comparators (§7),
invariants I1–I7 (§8.1), and G0's four self-tests (§8.2). I re-verified each constant against its
cited artifact — see §4.

### 2.1 The exact quantity, and the artifact fields it is computed from

```
R_know(cell) = [ acc(cell, mmlu_content) - 0.28445022076627263 ] / 0.10234977923372737
R_ppl(cell)  = 10.6416 / ppl(cell)
D(regime,N)  = R_ppl(regime,N) - R_know(regime,N)
Psi          = [R_know(RD,3) - R_know(RD,0)] - [R_know(SP,3) - R_know(SP,0)]
```

* `acc(cell, mmlu_content)` = the per-run arm mean over the **frozen** 14042-item set, scored
  `chat_template=False`, base protocol, no BOS, likelihood-based MC, **content** interface
  (never `MMLU_letter` — banned, `GATE_PREREGISTRATION.md` §4.1).
  Field: the same `summary.json` / arm-mean field A03 used to produce
  `a03_sigma_run_n3.json:families.*.axes.mmlu_content.means_pct` (divide by 100).
* `0.28445022076627263` = the longest-option split-tie null;
  `0.10234977923372737 = 0.3868 − 0.28445022076627263` = the intact-1B residual.
  Both from `a03_1b_floor_nulls.json` (md5 `a97a73bf802737601a6057f767b70853`, **I re-hashed it
  and it matches**), as transcribed in A04 `STATUS.json:nulls_per_metric.MMLU_content`.
* `10.6416` = `ppl(intact_1B)` on `n_windows=4096`, `n_tokens=8,384,512` — A04
  `nulls_per_metric.in_domain_PPL`. PPL is **never** a capability axis; it enters only via `D`.
* `ppl(cell)` = the held-out NTP PPL from the same base-protocol harness
  (`exp(Σnll/Σtok)`, all 8 shards asserted non-empty and equal-length).

### 2.2 ⚠️ AMENDMENT 1 (D1) — the permutation scheme was degenerate. Fixed.

`GATE_PREREGISTRATION.md` §2.1 asserts two things that **cannot both be true**:

> (i) "`R_know(regime,0)` is *a pre-registered constant per regime*"
> (ii) "enumerate all `C(2S,S)²` joint assignments … p is a multiple of `1/400` at `S=3`"

If (i) holds, relabelling regime at the `N=0` level is a **no-op**: `Psi` depends on the `N=0`
terms only through two fixed constants. The enumeration then yields `C(6,3) = 20` distinct
values, not 400. **Measured by exhaustive enumeration on CPU** (`itertools.combinations`, exact
rational arithmetic, maximal-separation input):

| scheme | distinct statistics | two-sided p floor | reaches `α=0.05`? |
|---|---:|---:|---|
| (i) `N=0` as constants — **what the prereg's §1 and §2.1 actually specify** | **20** | **0.100000** | **NO** |
| (ii) `N=0` measured per-run and permuted — what §2.1's "400" assumes | 400 | 0.005000 | yes (attainable; verified by construction) |

**Under the scheme the prereg specifies, `p_exact(Psi) ≤ 0.05` is unreachable for any effect
size whatsoever.** Since `kill_if` requires `p_exact(Psi) > 0.05` **AND** `|Psi| < Θ_int` for
`C1_FAILS`, a huge true interaction would have landed in `UNDERPOWERED` **with certainty**, and
`Psi` could never have been declared significant. The 107.7 GPU-h of G1 plus the 359.1 GPU-h of
Tier 1 would have bought an outcome fixed in advance by combinatorics.

**Exact minimum `S` under scheme (i):** `2/C(2S,S) ≤ 0.05` first holds at `S = 4`
(`2/70 = 0.028571`). `S = 3 → 0.100000`, `S = 2 → 0.333333`, `S = 5 → 0.007937`.

**ADOPTED FIX — take (ii), not a larger `S`.** The `N=0` cells are **measured per-run at
`S = 3`**, not collapsed to a constant, and both levels are permuted, giving the full
`C(6,3)² = 400`. Rationale, decided now and not after seeing data:

1. It is the **cheaper** fix. `S = 4` everywhere costs `4·4 + 1 + 2·4 = 25` runs =
   **897.8 GPU-h** vs the prereg's 22 runs = **682.3 GPU-h** → **+215.5 GPU-h** for the same
   science. Scheme (ii) costs **0 extra GPU-h**: the `N=0` cells are already 3 runs each in the
   pre-registered Tier 1, they were merely being *averaged into a constant* at analysis time.
2. It is the **more honest** estimator. Treating a measured 3-run cell as a zero-variance
   constant discards its sampling error from `SE(Psi)`, which understates the uncertainty of
   the very contrast the gate turns on.
3. `R ≡ 1` at `N=0` (`GATE_PREREGISTRATION.md` §1) is retained **only** as the definition of the
   retention scale for *reporting*, and is **struck as an analysis-time identity**.

**Consequential edit, mandatory:** `GATE_PREREGISTRATION.md` §4.4's choice of `k = 2` (because
`D(regime,0) ≡ 0` "by construction") **survives**, because §4.4 concerns the range of `D` over
the *free* `N` levels `{1,3}` and `D(·,0)` is still definitionally the reference. `c_2 =
1.1283791670955126` is unchanged. I re-derived it: `2/√π` = `1.1283791670955126` exactly, and
`c_3/c_2 = 1.5000` exactly, confirming the prereg's "+50.0 %" claim.

### 2.3 The statistical test, stated executably

**PRIMARY.** Exhaustive enumeration, no RNG, therefore **bit-identical on every node regardless
of numpy version** (the repo has three: LOCAL 2.5.1, `.82` 2.4.6 — see
`memory/numpy-version-split-breaks-cross-node-bootstrap.md`).

```
for a3 in combinations(range(6), 3):          # regime labels at N=3
  for a0 in combinations(range(6), 3):        # regime labels at N=0
      Psi_perm[a3,a0] = (mean_{a3} - mean_{a0^c... }) ...   # per §2.1, both levels relabelled
p_exact = #{ |Psi_perm| >= |Psi_obs| } / 400
```

**SECONDARY, descriptive only, never decisive:** paired-item bootstrap CI, `N_BOOT = 10000`,
`seed = 20260815`, with `assert numpy.__version__ == "2.5.1"` → runs on LOCAL / `.73` / `.104`,
**forbidden on `.82`**. Unchanged from `GATE_PREREGISTRATION.md` §2.1.

**Interval, reported alongside and never instead:** `Psi_obs ± t_{.975,df}·SE(Psi)`,
`SE(Psi) = 2σ̂_run/√S`, `df` by the `F_{.975}(S−1,S−1) = 39.0000` pooling precondition of
`GATE_PREREGISTRATION.md` §2.3. That precondition and the 6-authority `t`/`F` self-test table are
re-affirmed unchanged.

### 2.4 ⚠️ AMENDMENT 2 (D3) — the self-check that could not fail. Replaced.

`GATE_PREREGISTRATION.md` §2.1 mandates `assert p is a multiple of 1/400 at S=3`. **Measured:**
under the degenerate scheme p is a multiple of `1/20`, and `k/20 = 20k/400` **is** a multiple of
`1/400` for every `k` — so the assert would have **passed silently** on the broken gate.
Replace it with an assert that discriminates:

```
assert len(set(Psi_perm.values())) > 20, \
    "permutation set collapsed to the C(6,3) degenerate scheme; N=0 is being treated as a constant"
assert n_enumerated == 400
assert Psi_perm[identity, identity] == Psi_obs      # 0 bits of difference
```

The **first** assert is the one that matters: it is the only one of the three that fails on D1.
(Cardinality is used, not `== 400` distinct values, because ties in `Psi_perm` are legitimate.)

### 2.5 ⚠️ AMENDMENT 3 (D2) — the power bound was set below every measured prior. Fixed.

`GATE_PREREGISTRATION.md` §10.2 makes G1 return `UNDERPOWERED` if `σ_run(K) > 0.2903 pp`
(pool-6) / `> 0.2773 pp` (corner-4), and §3.4/§8.3 say `S` is **not** raised afterwards. The
prereg does not say whether the bound is compared against the **point estimate** `s` or its
`χ²` **upper bound**. That ambiguity is decision-changing, and I measured how much:

At `S = 3`, `df = 2`, the `χ²` upper-bound inflation factor is
`√(2/χ²_{.025}(2)) = √(2/0.050636) = 6.2847` — computed with a hand-written regularized-incomplete-gamma
`χ²` routine (**no scipy on any of the five nodes**) whose **self-test against a textbook table
passed with worst relative error 9.48e-04** over 9 quantiles, and which **reproduces A03's own
published `chi2_ci95_pp` intervals to 6 decimal places** for all three families. So:

| interpretation | requirement on the measured `s` | vs measured priors |
|---|---|---|
| bound applies to **point estimate** `s` | `s ≤ 0.2903 pp` | all three priors PASS with 3.7–5.2× headroom |
| bound applies to **`χ²` upper** at `df=2` | `s ≤ 0.2903/6.2847 = **0.046191 pp**` | **all three priors FAIL** |

Measured priors, from `a03_sigma_run_n3.json` (md5 `5fb6cd4c3d693831e50d0817bda93ab8`, re-hashed,
matches): keep7 `S=4` `s = 0.055468 pp`; keep12 `S=3` `s = 0.078336 pp`; pooled `df=5`
`σ = 0.065580 pp`. **Every one exceeds 0.046191 pp.** So under the `χ²`-upper reading, G1's
`UNDERPOWERED` verdict was the *predetermined* outcome — a gate that spends 107.7 GPU-h to
re-derive a bound already violated by data on disk.

**ADOPTED FIX, pre-data.** The `σ_run(K)` pass condition is evaluated on the **point estimate**
`s` at `S = 3`, with the `χ²` interval **reported mandatorily alongside** and never used as the
threshold:

```
G1 PASSES power     iff  s_run(K, mmlu_content) <= 0.2903 pp        [pool-6; 0.2773 corner-4]
G1 returns UNDERPOWERED iff s_run(K) > 0.2903 pp
The chi2 95% interval on s MUST be printed with its df, per prereg s4's standing prohibition
   ("never quote a sigma_run point estimate without its d.o.f. and its chi2 interval").
```

Justification, and it is not a convenience: a `χ²` upper bound at `df = 2` is a 6.28×
multiplier. Requiring a **3-run** `s` to satisfy a bound *after* 6.28× inflation is requiring
`s` to be ~4.4× *smaller* than the best-measured prior in the repo — i.e. demanding that a
**reset** arm be far quieter than a **continue-train** arm, when §3.4 of the prereg argues the
opposite is likely. That is not a power criterion; it is an automatic failure.

**⚠️ I must flag this against myself, because it is the exact failure mode the task warns
about.** Amendment 3 *loosens* a threshold, and it does so *after* I computed that the measured
priors violate the tight reading. That is structurally the "choose the threshold so the number
you already know will not trip it" move. My defence, and the reader should weigh it rather than
accept it:

* The **number `0.2903 pp` is not changed.** It stays exactly as pre-registered. Only the
  *estimator it is compared against* is disambiguated, and the prereg never specified which.
* The disambiguation is forced in **one** direction by the prereg's own §3.3, which derives
  `0.2903 = Θ_int / 3.5235` from the MDE formula — and the MDE formula takes **`σ`**, a point
  estimate, not an upper confidence bound. **The prereg's own derivation therefore already
  implies the point-estimate reading.** Amendment 3 makes the text consistent with its own §3.3
  rather than choosing between two live options.
* The prereg **separately** reports the pessimistic `χ²`-upper MDE column throughout §3.3
  (1.81× headroom at the `χ²` upper vs 4.43× at `σ̂`), so the conservative case remains
  visible in the verdict; it is simply not the trip-wire.
* **Falsifiable consequence I accept now:** if G1 measures `s > 0.2903 pp`, B03 stops. I am not
  reserving the right to re-loosen.

### 2.6 Sample size, and how items are paired

* **Unit of replication = the RUN.** `S = 3` runs per cell, 6 cells + 1 `from_scratch` floor
  check at `S = 1` = **19 runs** for the full ladder (`GATE_PREREGISTRATION.md` §3.1's 18 + 1).
  Item-level significance at `n = 6` runs is not evidence about runs — B04's tombstone.
* **Items: frozen sets, never resampled.** `mmlu_content n = 14042`, `triviaqa n = 17944`,
  `popqa n = 14267`, `nq_open n = 3610`, verified by sha256 against A04's
  `g0_anchor_sha256_20260810` **before** scoring (invariant I5).
* **Pairing.** **Unpaired between regimes** — an `SP` run and an `RD` run share no data order,
  so no run-level pairing exists. **Paired within a regime across `N`** only through the shared
  frozen item set, which is what makes `R_know(RD,3) − R_know(RD,0)` a within-regime difference.
  The permutation permutes **regime labels within each `N` level** (§2.2 scheme (ii)); that is
  the only exchangeability the design supports.
* **`nq_open` is DEMOTED to descriptive-only** by inheritance: its item-level 95 % CI half-width
  (1.459–2.063 pp at `n = 3610`) already exceeds its own `Θ = 0.970 pp`. It may never carry a
  kill or proceed decision.

### 2.7 PASS / FAIL / INDETERMINATE — the full decision, as a boolean

```
C1_FAILS == (p_exact(Psi) > 0.05)  AND  (|Psi_hat| < 0.10 retention units = 1.0230 pp)
C2_FAILS == (p(D(3)-D(1)) > 0.05) AND  (|D(3)-D(1)| < 0.10 retention units)
                                   AND  (range(D over N in {1,3}) <= 1.1283791670955126 * sigma_D/sqrt(3))

KILL        == C1_FAILS AND C2_FAILS              # PRIMARY, the 存活条件 reading
KILL_strict == C1_FAILS OR  C2_FAILS              # ALSO REPORTED, each clause at alpha/2 = 0.025
PASS (survive) == NOT KILL, and NOT any INDETERMINATE verdict below
```

Both verdicts appear in the verdict document's **first** table. If they disagree the headline is
pre-committed verbatim: *"B03 survives its 存活条件 gate and fails its 关闭条件 gate; the two
sections of `PROPOSAL.md` are inconsistent and this result does not resolve which was
intended."* It may **not** be reported as a clean survival.

**THE INDETERMINATE BAND — explicit, and it is where most outcomes will land.**
A gate with no indeterminate band silently converts noise into a decision.

| condition | verdict | this is NOT |
|---|---|---|
| `p > 0.05` **and** `|effect| ≥ Θ` | **`UNDERPOWERED`** | not KILL — the kill needs **both** conjuncts |
| `p ≤ 0.05` **and** `|effect| < Θ` | **`SIGNIFICANT_BUT_BELOW_THRESHOLD`** | not a survival — `Θ` was set in advance to make this uninteresting |
| either range fails its §4.4 floor | **`UNRESOLVED_SUBNOISE`** on that clause | **no ratio of two ranges may be quoted** unless BOTH clear their own floors |
| G0 self-test (d) fails (optimizer 2→4-group shim fires) | **`NOT_EXPRESSIBLE`** | stop at **0 GPU**; the finding is a real protocol note |
| G1 `s_run(K) > 0.2903 pp` (pool-6) / `> 0.2773` (corner-4) | **`UNDERPOWERED`** | `S` is **NOT** raised post hoc |
| G1 `σ_R(ppl) > 0.03962` retention units | **`UNDERPOWERED_C2`** | clause 1 may still proceed; the verdict must say so |
| `len(set(Psi_perm)) <= 20` (§2.4 assert) | **`VOID`** | the analysis collapsed to the D1 scheme; re-run, do not report |

### 2.8 What is frozen by the commit that adds this file

`α = 0.05`; Holm `0.016667/0.025/0.05`; `α/2 = 0.025` for the strict-OR verdict;
`Θ_int = 1.0230 pp = 0.10` retention units; `Θ_sep = 0.10` retention units; `S = 3`;
`T_total = 8000` steps; `E = 8` epochs for the RD cell; `c_2 = 1.1283791670955126`, `k = 2`;
`F_{.975}(2,2) = 39.0000` pooling precondition; **`s_run(K) ≤ 0.2903 pp` compared on the POINT
ESTIMATE (§2.5)**; `σ_R(ppl) ≤ 0.03962`; **enumeration cardinality `= 400` with the
`>20`-distinct-values assert (§2.2, §2.4)**. If any is later changed, **the gate is VOID** and
must be re-run from the beginning.

---

## 3. FIRABILITY PRECHECK — computed against on-disk data, at smaller scale

**A gate that cannot fire is not a gate.** So I computed every statistic the gate needs that
existing data can supply, and report each against its threshold.

### 3.1 What exists on disk, and what it is a proxy for

**No B03 layer-reset run exists at any scale on either disk.** Verified 2026-08-16:
`ls -d outputs/*B03* outputs/*reset* outputs/*cyclic*` returns rc=2 on **wzc1**, and the same
listing under `/apdcephfs_zwfy6/.../outputs/` (via `.73`, which is where zwfy6 is mounted)
returns rc=2. So the pre-data ordering of `GATE_PREREGISTRATION.md` and of this file both hold.

What **does** exist is 7 runs of 1B OLMo-2 `keep_front + n_fresh` continue-training, scored on
the identical frozen item sets with the identical harness. Per `RELATED_WORK.md` §2.1 that
construction **is** LLF's mask with `L = K_f` — i.e. these are **`N = 1` reset arms** in B03's
own vocabulary (one reset event, then relearn). They are **not** `N = 3`, and they are all
**single-pass**, so they populate exactly **one** of the six cells. **They cannot produce `Psi`**,
which needs all four corners. That is stated so no one later reads §3.2 as a partial result.

### 3.2 `R_know` computed for every on-disk 1B arm — the retention scale is live

Using the frozen constants of §2.1 on `a03_sigma_run_n3.json:families.*.axes.mmlu_content.means_pct`:

| family (= `SP`, `N=1`-like) | seeds | acc range (%) | **mean `R_know`** | `s_R` (retention units) |
|---|---|---|---:|---:|
| `keep7` + 20k CPT @ step220000 | 0, 43, 44, 45 | 32.118 – 32.239 | **0.36651** | 0.005419 (`S=4`) |
| `keep12` @ step5000 (A04 Stage B) | 101, 102, 103 | 31.619 – 31.776 | **0.31781** | 0.007654 (`S=3`) |

**The read-out arithmetic executes end-to-end on real data.** `R_know ≈ 0.32–0.37` means these
arms sit at ~⅓ of the intact 1B's above-null MMLU-content knowledge — a large, non-degenerate
signal, comfortably off both the `R=0` null and the `R=1` ceiling. The retention scale is not
pathological, which was a live risk (A04's `Δ` is ill-defined when the intact residual ≤ 0;
here the residual is a healthy 10.235 pp).

### 3.3 The two power thresholds, evaluated NOW against measured priors

| statistic | prereg threshold | **measured on disk** | verdict |
|---|---|---|---|
| `σ_run(K)` on `mmlu_content`, **point estimate** (§2.5 fix) | `≤ 0.2903 pp` | pooled `df=5` **0.065580 pp**; keep7 `S=4` 0.055468; keep12 `S=3` 0.078336 | **PASSES**, 3.7–5.2× headroom |
| same, `χ²` **upper** at the run's own df | `≤ 0.2903 pp` | pooled `df=5` upper **0.160842** (passes); keep12 `df=2` upper **0.492324** (**fails**) | reading-dependent → **this is D2**, resolved in §2.5 |
| `σ_D` for clause 2 | `σ_D ≤ 0.04014` | `σ_R(know)` alone = **0.005419** (keep7) / **0.007654** (keep12) | knowledge component passes; **`σ_R(ppl)` is UNMEASURED** → `σ_D` cannot be evaluated |
| `p_exact` floor | must be `≤ 0.05` to be reachable | **0.100000** under the prereg's own scheme | **FAILS** → **this is D1**, fixed in §2.2 |

**The `σ_R(ppl)` gap is genuine and I could not close it at 0 GPU.** Computing `R_ppl` per run
needs a held-out-PPL number per 1B seed arm; `a03_sigma_run_n3.json` carries the four capability
axes but **not** per-seed PPL, and `A04:nulls_per_metric.in_domain_PPL` gives only the single
intact-1B value `10.6416`. Re-deriving per-seed PPL means running the eval harness on 7
checkpoints = GPU. So clause 2's power remains **UNVERIFIED**, and `UNDERPOWERED_C2` stays a live
outcome. I am not asserting it passes.

### 3.4 Expected outcome, stated honestly and in advance

**I expect this gate to return KILL, or `UNDERPOWERED` on clause 2.** The reasons are on the
record before any B03 number exists:

* **Six published negative priors** (§5.3) all point the same way: reinitialisation's benefit is
  confined to small-data / overfitting-prone regimes, and LM pretraining is neither.
* **Our own Paper B, read as a prior**, predicts *monotone worsening in `N`* — each reset pays
  another near-irreversible knowledge tax (keep14 @200k: PPL tax 1.428× while recovering only
  19.5 % of base above-chance MMLU). Monotone uniform degradation is precisely
  `PROPOSAL.md` §「关闭条件」 clause 1.
* **The measured spread is tiny relative to the threshold.** `σ_run = 0.0656 pp` against
  `Θ_int = 1.0230 pp` means the design can *resolve* a 10 %-of-residual interaction if one
  exists — so a null here would be a **real** null, not a noise floor. That is what makes the
  gate worth firing at all: it is capable of returning a *trustworthy* negative.

**That is the value proposition, and it should be stated plainly: B03's most likely deliverable
is a well-powered negative result with a located regime boundary — not a positive finding.**
Whether ~750 GPU-h is worth a trustworthy negative at `priority = low` is a resource decision,
not a scientific one, and §6 leaves it with the human.

---

## 4. GPU COST ESTIMATE, WITH ITS BASIS

### 4.1 The anchor — and a provenance correction the prereg needs

**BASIS: `2.02 s/step` median, `n = 36`, world_size = 8 (8×H20, `sm_90`).**

**I did not take this from A04's `STATUS.json`. I re-measured it from the raw log.** The log is
**not on wzc1** — `logs/olmo2_1B_keep7fresh2_1node.log` does not exist there — it is on **zwfy6**
at `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/olmo2_1B_keep7fresh2_1node.log`
(235,232 bytes, 1,421 lines, mtime 2026-07-16 22:20). Located by listing that directory over ssh
to `.73`; zwfy6 is **not mounted on LOCAL** (`/apdcephfs_zwfy6` does not exist here). **Every
citation of this log as a bare `logs/...` path is disk-ambiguous and should be qualified.**

Parsed on `.73` with CPU-only python:

* **36** lines match `[step N/200000] ... X s/step` — matching A04's `n=36` exactly.
* **median = 2.02 s/step**, mean 2.0461, min 2.02, max 2.74 (the max is the first logged step,
  amortising warm-up).
* **Independent Δ-elapsed cross-check**, per `memory/tqdm-elapsed-and-counter-have-different-origins.md`:
  step 20 → step 720 spans 1430 s wall over 700 iters = **2.0429 s/step**. Consistent with the
  median to 1.1 %, so 2.02 is a **compute** rate, not a ckpt-amortised artifact.
* Config line, verbatim from the log: `world_size=8 bs=16 gaccum=1 eff_bs=128 seq_len=2048
  lr_fresh=0.0001 lr_inh=2e-05 max_steps=200000` → `262,144 tok/step`. **Matches A04's anchor
  string field-for-field.**
* Bonus verification: the same log says `dataset rows=15491607` — the **canonical** 31.727 B-token
  corpus, confirming `STATUS.json:data_provenance_correction_20260815_MAIN` and **not** the
  7,570,911-row prefix. Any B03 cell must load the 15,491,607-row corpus or the
  single-pass-vs-repeated-data axis is confounded with corpus identity.

### 4.2 The arithmetic, re-derived from scratch

```
35.9111 GPU-h per run  =  2.02 s/step x 8000 steps x 8 GPU / 3600
                       =  4.489 h wall on ONE 8-GPU node
2.0972 B tokens per run =  8000 x 262,144
1.712375e-08 GPU-h/token;  0.0584 B tokens per GPU-h
```

Both derived quantities reproduce `GATE_PREREGISTRATION.md` §9.1 **exactly**.

| leg | runs | GPU-h | node-days (1×8-GPU) |
|---|---:|---:|---:|
| **G0** ckpt-surgery script + 4 self-tests + data check | 0 | **0** | 0 |
| **G1** `(SP,N=3)` at `S=3` | 3 | **107.7** | 0.56 |
| **Tier 1** 4 corner cells `S=3` (G1's 3 reused) | 9 new | **323.2** | 1.68 |
| **Tier 1b** `from_scratch` 1B floor check `S=1` | 1 | **35.9** | 0.19 |
| **Tier 2** two `N=1` cells `S=3` | 6 | **215.5** | 1.12 |
| **train total** | **19 new / 22 incl. reuse** | **682.3** | **3.55** |
| eval, 4 axes × 22 runs | — | ~66 (**order of magnitude only**) | — |
| **GRAND TOTAL** | | **~748 GPU-h** | **~3.9 node-days** |

`682.3` reproduces the prereg to < 1 GPU-h. **The §2.2 fix adds 0 GPU-h**; the rejected `S = 4`
alternative would have cost **897.8 GPU-h (+215.5)**.

### 4.3 Weaknesses of this estimate, named

* The anchor is a **continue-train** rate. Same trainer, same shape, same `world_size`, so it
  should transfer closely — **but a reset run's s/step has never been measured.** G1's first
  deliverable is a measured s/step.
* The **~66 GPU-h eval figure is scaled**, not re-derived, from A04's
  `cost.pilot_zero_eval_only_GPU_h = 3`. Quoted to order of magnitude only.
* **8 cards is the efficient unit.** A04's `scaling_note` measures 16 cards at only **1.36×**
  throughput for 2× the GPUs, so extra nodes buy **parallel cells, not faster cells**.
* Invariant I1 requires **all cells on `sm_90`** — `.73`/`.82`/`.104` only. Mixing in the
  `sm_100` B200s would confound the primary effect with hardware drift.

---

## 5. NOVELTY VERDICT

```
novelty_verdict: SURVIVES as a REGIME-BOUNDARY / NEGATIVE-RESULT question ONLY.
                 DEAD as a method claim -- LLF (ICLR 2022) owns the operator.
                 NO paper found shows the same thing. NOT preempted.
adjudicated:     2026-08-16, independently of the 2026-08-15 pass, 0 GPU.
```

**Positive control run first, as required.** DBLP `search/publ/api` returned
`conf/iclr/ZhouVLC22` for a known-good query, and `api2.openreview.net/notes/search` returned
HTTP 200 with a populated `notes` array. **Both controls passed, so the NOT-FOUND results in
§5.4 are meaningful.** Two transient failures are recorded honestly: DBLP intermittently served
an **HTTP error-500 HTML page** for some queries (retry with a differently-phrased query
succeeded — *not* evidence of absence), and one first-attempt parse failure was a transient that
succeeded on retry.

### 5.1 ★ The most important methodological finding: api2 is NOT down

`RELATED_WORK.md` §5.1 records `api2.openreview.net` as returning **HTTP 403
`ChallengeRequiredError` on every path**, and therefore left **three venue calls unverified**.
**That diagnosis was wrong, and I closed all three.** Measured 2026-08-16:

* `api2.openreview.net/notes/search?term=...&limit=N` → **HTTP 200**, returns `venue` + `venueid`.
* `api2.openreview.net/notes?forum=<id>` → **HTTP 403 `ChallengeRequiredError`**.

Exactly as the standing rule states: **a 403 on `notes?forum=` means "use the other endpoint",
not "OpenReview is unreachable."** The 2026-08-15 pass generalised one blocked path to the whole
API and downgraded three records for no reason.

### 5.2 Per-paper verified metadata

Venue authority by family: **OpenReview `venueid`** for ICLR/NeurIPS/ICML; **ACL Anthology +
DBLP** for the ACL family; DBLP as cross-check everywhere.

| Paper | Verified venue | Authority (this session, 2026-08-16) | Relation to B03 |
|---|---|---|---|
| **Zhou, Vani, Larochelle, Courville — LLF, *Fortuitous Forgetting in Connectionist Networks*** | **ICLR 2022** | DBLP **`conf/iclr/ZhouVLC22`** (venue `ICLR`, 2022). ⚠️ **ALSO** `journals/corr/abs-2202-00155` | ★ **Operator collision.** Its mask `M^l = 1[l<L]` **is** our `keep_front/n_fresh` construction. Kills the method claim. **Does NOT preempt** the surviving question: CNN image classification, no decoder-only transformer, no LM pretraining, no parametric-knowledge axis — and its own Table A8 reports LLF **losing** as data/baseline strengthen, which is the boundary B03 locates. |
| **Springer et al. — *Overtrained Language Models Are Harder to Fine-Tune*** | **ICML 2025** | DBLP **`conf/icml/SpringerGWKYMNR25`**. ⚠️ **ALSO** `journals/corr/abs-2503-19206` | ★ Owns the **timing** axis (*progressive sensitivity*) on **OLMo-1B / OLMo-2-7B** = our family and scale. Kills a B03 timing sweep as a contribution. Does not touch the data-regime × reset interaction. |
| **Chen et al. — *Improving Language Plasticity via Pretraining with Active Forgetting*** | **NeurIPS 2023** | DBLP **`conf/nips/ChenMRAS0A23`**; OpenReview `venue='NeurIPS 2023 poster'`. ⚠️ **ALSO** `journals/corr/abs-2307-01163` | Owns the **framing** (periodic reset during pretraining for plasticity) and the optimizer-moment-reset **hygiene**. Resets **embeddings**, not decoder blocks. |
| **Shin, Oh, Lee, Yun — DASH** | **NeurIPS 2024 poster** | OpenReview forum **`IdQuUYMA1t`**, `venueid = NeurIPS.cc/2024/Conference`. (DBLP served error-500 for this query — **not** absence.) | Published **negative prior**: App. C.1 says reset "cannot be a solution" under stationary data. Its protocol is not single-epoch streaming → prior only, **never** a cross-tabulated baseline. |
| **Sarfi et al. — SEAL** | **CVPR 2023** | DBLP **`conf/cvpr/SarfiKCKRMB23`**. ⚠️ ALSO `journals/corr/abs-2304-04858` | Negative prior: LLF features degrade transfer across all datasets explored. |
| **Muennighoff et al. — *Scaling Data-Constrained Language Models*** | **NeurIPS 2023** *and* **JMLR 2025** | DBLP **`conf/nips/MuennighoffRBST23`** **AND** **`journals/jmlr/MuennighoffRBSP25`** — **both confirmed present** | Owns B03's **axis 1**. Must be **cited, not claimed**. ⚠️ **Two records: cite one deliberately, never "NeurIPS/JMLR".** Also the source of `E = 8` (their ≤4-epoch benign band). |
| **Allen-Zhu & Li — *Physics of LMs 3.3, Knowledge Capacity Scaling Laws*** | **ICLR 2025** | DBLP **`conf/iclr/Allen-ZhuL25`**. ALSO `journals/corr/abs-2404-05405` | The yardstick for "parametric knowledge per parameter under repetition". |
| **FIRE — *Frobenius-Isometry Reinitialization…*** | ★ **ICLR 2026 Oral** — **UPGRADED, was `venueid` UNVERIFIED** | OpenReview forum **`CfZLxT3zIZ`**, `venue = 'ICLR 2026 Oral'`, **`venueid = ICLR.cc/2026/Conference`** — **KILLCHECK's 2026-08-06 record REPRODUCED**. (⚠️ a second note `P5deO9CrbA` carries `venue='CoRR 2026'` — the DBLP mirror, **not** the conference record.) | Makes "reinit-for-plasticity has never been done on LMs" **false** (GPT-0.1B / OpenWebText). Reinitialises **weight matrices**, not whole blocks. |
| **Han, Bordt, Zhang, Kakade — *Weight Decay Improves Language Model Plasticity*** | ★ **ICML 2026 regular** — **UPGRADED, was `venueid` UNVERIFIED** | OpenReview **`zMO9H4hLyR`**, `venue='ICML 2026 regular'`, **`venueid = ICML.cc/2026/Conference`**. ⚠️ Also 3 **workshop** notes (`colmweb.org/COLM/2026/Workshop/Sci-FM`, `ICLR.cc/2026/Workshop/Sci4DL`, `ICLR.cc/2026/Workshop/SPOT`) — **do not cite a workshop note as the main-track venue** | Same story skeleton (pretraining choice → adaptability), different knob (weight decay vs structural reset). |
| **Thangarasa et al. — SPDF** | **UAI 2023** — **UPGRADED, was "in-repo prior pass, not re-verified"** | DBLP **`conf/uai/ThangarasaGMLLD23`** (venue `UAI`, 2023), reproduced on two independent queries. ALSO `journals/corr/abs-2303-10464` | Kills the "no prune-regrow above 314M" scale-vacuum argument (1.3B GPT-3 XL). One-shot, unstructured, FLOPs-motivated. |
| **Alabdulmohsin, Maennel, Keysers — *Impact of Reinitialization on Generalization in CNNs*** | **arXiv-only** — and now with a *reason* | DBLP `journals/corr/abs-2109-00267`, **no `conf/` record**. ★ **NEW**: OpenReview shows `venue = 'Rejected by TMLR'` — **corroborates** arXiv-only rather than contradicting it | Strongest negative prior: §5 *"For large datasets … reinitialization does not seem to offer a benefit"*; decision tree splits on **"Training Set Size < 35K?"**. |
| **2606.06888 — *Data-Constrained LM Pretraining: Improved Regularization and Scaling Laws*** | ★ **HiLD @ ICML 2026 Workshop Poster** — **CORRECTED, `RELATED_WORK.md` said arXiv-only** | OpenReview **`W5k9IVRdp4`**, `venueid = **ICML.cc/2026/Workshop/HiLD**` | ⚠️ **A WORKSHOP, not main track** — `venueid` proves it. Concurrent (2026-06). Relevant: argues the additive Chinchilla form is misspecified under repeated data. |
| **LoRR — *Sample-efficient LLM Optimization with Reset Replay*** | **arXiv-only** | DBLP `journals/corr/abs-2508-06412` (2026-08-15 pass; **not** re-verified today) | 7B-scale negative: resetting `full_layers` "proves detrimental". Post-training, not pretraining. |
| **Wang et al. — CoMe, *Layer as Puzzle Pieces*** | **NeurIPS 2025 Poster** | Verified in A04's `RELATED_WORK.md` §C3 via OpenReview `venueid` + `Camera_Ready_Revision`. **Not** re-verified here | Layer-granularity prune-and-recover SOTA; the comparison a reviewer will demand. |

**Both fallacies actively avoided.** Six of the papers above carry **both** a `conf/` record and a
`journals/corr/` record — I treated the CoRR record as **NOT-FOUND**, never as
"is-a-preprint". And no `Withdrawn_Submission` was treated as "is-not-published". Semantic
Scholar was **not queried and not relied on**; its known HTTP 429 is not evidence of absence.

### 5.3 The six published negative priors — must be stated BEFORE any B03 result

1. `2109.00267` §5 — no reinit benefit on large datasets; tree splits at 35K samples.
2. **LLF (ICLR 2022) Table A8** — LLF loses to baseline (WRN-28-10 CIFAR-10 96.32→95.91;
   CIFAR-100 81.29→80.95).
3. **SEAL (CVPR 2023)** — LLF features degrade transfer across all datasets explored.
4. **DASH (NeurIPS 2024) App. C.1** — reset "cannot be a solution" under stationary data;
   LM pretraining is near-stationary.
5. **LoRR** — full-layer reset "detrimental" at Qwen2.5-7B class.
6. **OUR OWN Paper B keep14 @200k** — PPL tax **1.428×** while recovering only **19.5 %** of base
   above-chance MMLU (`status/PAPERB_KEEP14_200K_EVAL.md`, lines re-read 2026-08-16 and both
   numbers confirmed verbatim). Read as a prior: **monotone worsening in `N`**.

### 5.4 The surviving cell, and why it is not preempted

**Unoccupied:** the **crossing** of the data-constrained-pretraining axis (Muennighoff et al.)
with the reset/plasticity axis (LLF / Active Forgetting / DASH), measured on **parametric
knowledge** in a **decoder-only LM**. Neither literature measures the other's variable.

**Recency sweep, 2026-08-16, six OpenReview `notes/search` queries** on: layer-reinitialisation ×
repeated data/epochs; resetting layers × factual-knowledge forgetting; forget-and-relearn × LM
pretraining; cyclic layer reset × decoder-only × knowledge retention; later-layer forgetting ×
single-pass pretraining; reset × epochs interaction. **Result: no paper shows the same thing.**
Nearest returns were *"Mix Early, Forget Less"* (ICLR 2026 **Workshop** DATA-FM — data mixing, not
reset), *"How Do LLMs Acquire Factual Knowledge During Pretraining?"* (NeurIPS 2024 — acquisition
dynamics, no structural intervention), and *ReLearn* (ACL 2025 — unlearning, not pretraining).

Per the standing rule (`memory/prior-work-differentiate-dont-abandon.md`): the bar is
**完全相同 / 抄袭**, and work within 2–3 months is **concurrent**. **Nothing meets that bar.**
The audit's output is a **citation-obligation list** (11 must-not-claims, all still binding),
**not** a death certificate. **B03 may be killed only by its own `关闭条件` gate**, and that gate
has never been run.

### 5.5 Honest gaps in this novelty pass

1. **Zero `Camera_Ready_Revision` checks.** `notes?forum=<id>` is 403-gated
   (`ChallengeRequiredError`, challenge id `2026-08-15-7359500`), which is the only endpoint that
   exposes the invitation list. So **arXiv-vs-camera-ready diffs were NOT performed** — and this
   repo has observed camera-readies **deleting formulas** the arXiv version had. For FIRE and
   Weight-Decay-Plasticity the `venueid` is now solid but the **text** may differ from arXiv.
2. **No full-text PDF was read.** Every verbatim quotation is inherited from the `literature/`
   corpus (2026-08-06), whose extraction artifacts were stored in `/tmp` and are **gone**.
   Quotations are cited as the corpus's records, not as independently re-checked text.
3. **No forward-citation re-scan.** The corpus's 434-citation scan is now **10 days old**. My
   sweep was 6 OpenReview queries — a recency probe, not a substitute.
4. **Semantic Scholar not queried** (known 429). Its silence is not evidence of absence.
5. **LoRR and CoMe not re-verified today** — carried from the 2026-08-15 and A04 passes.
6. **DBLP intermittently served HTTP 500** for some phrasings (DASH, one SPDF attempt). Where a
   retry did not succeed I used OpenReview instead and said so; **no absence was inferred from a 500.**
7. **Cross-disk:** all cited repo files are on **wzc1**, except the cost anchor log, which is on
   **zwfy6** only (§4.1). The trainer source facts were read from the **wzc1** copy; the zwfy6
   checkout is a separate, often-lagging copy and was not diffed.

---

## 6. WHAT THIS DOES AND DOES NOT AUTHORISE

**It does NOT authorise a GPU.** B03 stays `lifecycle = ready_cpu`, `priority = low`,
`status = hold_gate_only`. Three things must all be true before a card is spent, unchanged from
`STATUS.json:gpu_policy`:

1. **G0 PASSES — and G0 is 0 GPU.** The reset operator **still does not exist as code**: no flag
   in `scripts/train_olmo2_arch_probe2.py` re-initialises the top `K` layers at a resume point.
   I re-verified the trainer facts the prereg cites: `_assert_fresh_init` at **line 140** checks
   `post_attention_layernorm` all-ones, `q_norm` all-ones, `q_proj.weight` std in `(0.01, 0.04)`,
   with the docstring stating **"OLMo-2 is POST-norm and has NO input_layernorm"** — so the
   prereg's insistence on `post_attention_layernorm` is correct. The `DistributedSampler(...,
   seed=args.seed)` fix is at **line 869**. The **2-group → 4-group optimizer shim** is at
   **line ~912** (`elif n_ckpt_groups == 2 and n_new_groups == 4`), so G0's self-test (d) is
   testing a real code path. If G0 fails (d), **B03 ends at 0 GPU** with a publishable protocol
   note.
2. **Explicit user authorisation** for ~748 GPU-h, or at minimum G1's 107.7 GPU-h. Not implied by
   this file.
3. **A free `sm_90` node.** Measured 2026-08-16: `.73`, `.82`, `.104` are **all at 100 % GPU
   utilisation on all 8 cards each** (96.4 / 78.5 / 78.8 GiB used per card). There is no capacity.

**Ordering of the amendments relative to data.** No B03 run exists on either disk (§3.1), so
none of §2's constants could have been informed by a B03 number. The git commit that adds this
file is the proof of ordering. **Nothing in §2 may be edited once the first B03 number lands.**

**Next action: G0, which is 0 GPU** — the checkpoint-surgery script and its four self-tests.
Amendment §2.2 additionally means Tier 1 must **retain per-run `N=0` scores** rather than
collapsing them to a per-regime constant; that is an analysis-side requirement with no extra
compute.
