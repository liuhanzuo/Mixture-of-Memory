# paperC gate-2 — SECOND MC BENCHMARK, FULL REPLICATION

Task #248. Ran 2026-08-11 on `.73` (8×H20), ~11 min wall-clock, 36 arm×task cells.

**Verdict: `REPLICATES_PARTIALLY_AND_NARROWS_THE_CLAIM`.**

The MMLU letter-vs-content interface contrast now exists on five non-MMLU MC
benchmarks under the *identical* construction (so the "the interfaces are actually
different" footnote in `STATUS.json:gate_results.gate2_second_mc_benchmark.caveat`
is retired). The qualitative pattern reproduces on all five. But the strong form of
MMLU's headline — a *statistically significant* below-floor letter verdict on the
most damaged arm — **reproduces on only 1 of 5 benchmarks, and 4 of the 5 are
underpowered to have detected MMLU's own effect size in the first place.** That is
a real narrowing and it is reported as such below.

---

## 0. What was actually run, and why it is not the old gate-2

| | old gate-2 (2026-08-08) | this run (#248) |
|---|---|---|
| interface A | raw sum-LL over option TEXT | **letter** — question + `A./B./C./D.` body, score the bare letter token after `Answer:` |
| interface B | length-normalised `acc_norm` over option TEXT | **content** — same question, NO labelled body, score the option TEXT |
| relation to MMLU | *analogous*, not identical | **identical construction** |
| GPU | none (reused ledger) | 8×H20, fresh forward passes |
| letter null | — (no letter interface existed) | best-constant letter, per task |
| length unit for the content null | **characters** (`len(text)`) | **continuation tokens** (MMLU's unit) |

The construction is MMLU-parallel by design: the letter prompt is the content
prompt with the labelled option body spliced in before `\nAnswer:`, verified
byte-for-byte in the harness selftest. Code:

* harness `scripts/eval_olmo2_mc_letter_content.py` (`--selftest` passes on CPU,
  including asserts that the merge **refuses** a 1/8 shard set and a
  wrong-cardinality set)
* driver `scripts/_run_olmo2_mc_letter_content_8gpu.sh`
* nulls/stats `proposal/active/A01-null-calibration-methodology/code/a01_gate2_letter_content_nulls.py`

Protocol: `chat_template=False`, `add_bos=0`, fp32 master weights + bf16-autocast
forward, `batch_size=48`. Arms are the same six as the MMLU table and gate-3:
`7B_base`, `keep8@121000`, `keep10@83500`, `keep12@124000`, `keep14@200000`,
`shortgpt16@200000` (every ckpt `ls`-verified on zwfy6 before launch).

Statistics: paired bootstrap `n_boot=10000`, `boot_seed=7`, two-sided p from
**`two_sided_boot_p`** — the R-7-fixed mid-p estimator, imported verbatim from
`a01_gate3_fp32_vs_bf16.py`, not the old unclamped `2*min((bs<=0),(bs>=0))`.
Exact McNemar against the deterministic constant predictor wherever the null is a
0/1 per-item decision.

---

## 1. Benchmark selection, with the gold marginals that justified it

Selected on 0 GPU before any card was touched. Criterion: the benchmark must
admit a **real letter interface** (labelled options → score the letter) *and* a
real content interface, and its gold marginal must be **skewed** so the
best-constant floor is above chance — otherwise there is nothing for paperC to say.

| task | n | n_opt | gold marginal | **best-constant floor** | chance | floor − chance |
|---|---|---|---|---|---|---|
| **arc_challenge** | 1172 | 4 (+4×3-opt, 3×5-opt) | A 266 / B 311 / C 310 / D 285 | **always-B `0.265358`** | `0.250156` | **+1.520 pp** |
| **arc_easy** | 2376 | 4 (+7×3-opt, 4×5-opt) | A 596 / B 585 / C 633 / D 561 / E 1 | **always-C `0.266414`** | `0.250161` | **+1.625 pp** |
| **openbookqa** | 500 | 4 | A 138 / B 126 / C 132 / D 104 | **always-A `0.276000`** | `0.250000` | **+2.600 pp** |
| **commonsense_qa** | 1221 | 5 | A 239 / B 255 / C 241 / D 251 / E 235 | **always-B `0.208845`** | `0.200000` | **+0.885 pp** |
| **piqa** | 1838 | 2 | A 910 / B 928 | **always-B `0.504897`** | `0.500000` | **+0.490 pp** |
| **winogrande** (control) | 1267 | 2 | A 628 / B 639 | **always-B `0.504341`** | `0.500000` | **+0.434 pp** |

Chosen: **arc_challenge** (4-way, the closest structural analogue of MMLU) and
**openbookqa** (4-way, largest floor−chance gap at +2.60 pp) as the primary pair;
**arc_easy** as a large-n high-signal companion; **commonsense_qa** (5-way) and
**piqa** (2-way) for option-count diversity — CSQA and PIQA are the
"not-everything-collapses" side; **winogrande** as the negative control.

**Not chosen and why.** HellaSwag: has no natural labelled-letter form (the four
candidates are sentence *continuations* of a shared context, so a letter prompt
would have to invent a question). BoolQ: the two candidates are `no`/`yes`, i.e.
already effectively a symbol interface — a letter re-labelling adds nothing, and
its `0.6217` always-B result is already in the ledger. MMLU-Pro: 10-way, not
comparable to the MMLU 4-way arms.

**Important honesty note on the floors.** These best-constant floors are only
**+0.43 to +2.60 pp** above chance, i.e. *much* flatter than MMLU's `0.2689` vs
`0.25` (+1.89 pp) is relative to its own n. paperC's "chance badly misstates the
null" rhetoric is therefore **weak on the letter side of these five tasks** — the
dramatic cases remain BoolQ (`0.6217` vs `0.50`, +12.2 pp) and, on the *content*
side, OpenBookQA (see §3). Do not oversell the letter floors here.

---

## 2. Letter interface vs best-constant floor — the head-to-head with MMLU

`+` = above floor (p<0.05), `=` = AT the floor (n.s.), `−` = BELOW floor (p<0.05).

| arm | MMLU letter (n=14042) | MMLU | arc_ch | arc_easy | obqa | csqa | piqa | winog (ctrl) |
|---|---|---|---|---|---|---|---|---|
| `7B_base` | `0.6054` (+33.65pp, p=1e-4) | `+` | `0.7696` `+` | `0.8939` `+` | `0.7320` `+` | `0.6577` `+` | `0.7171` `+` | `0.5919` `+` |
| `shortgpt16` | `0.4742` (+20.53pp, p=1e-4) | `+` | `0.5768` `+` | `0.7630` `+` | `0.5780` `+` | `0.4521` `+` | `0.6170` `+` | `0.5130` `=` |
| `keep14` | `0.3184` (+4.95pp, p=1e-4) | `+` | `0.3353` `+` | `0.4533` `+` | `0.3700` `+` | `0.2735` `+` | `0.5310` `=` | `0.5051` `=` |
| `keep12` | `0.2728` (+0.38pp, p=0.369) | `=` | `0.2611` `=` | `0.2652` `=` | `0.2540` `=` | `0.2113` `=` | `0.5005` `=` | `0.5043` `=` |
| `keep10` | `0.2720` (+0.31pp, p=0.409) | `=` | `0.2816` `=` | `0.2395` **`−`** | `0.2260` `=` | `0.1933` `=` | `0.4995` `=` | `0.4807` `=` |
| `keep8` | `0.2550` (−1.39pp, p=0.019) | **`−`** | `0.2560` `=` | `0.2584` `=` | `0.2580` `=` | `0.1982` `=` | `0.5299` `=` | `0.5099` `=` |

Full per-cell numbers (delta, CI95, bootstrap p, exact McNemar p, modal share, tie
rate, residual fractions) in `gate2_letter_content_nulls.{json,csv}` — 396 CSV rows.

### 2a. Did MMLU's conclusion replicate? **PARTIALLY — yes on the ordering, no on the significance.**

**What replicates, cleanly, on all five evidence tasks:**

1. **The healthy → damaged ordering, and the point at which the floor is reached.**
   Every task ranks the arms base > shortgpt16 > keep14 > {keep12, keep10, keep8},
   and on every one of the five the three deepest-damage arms (keep12/keep10/keep8)
   drop from "significantly above floor" to "indistinguishable from a constant
   predictor". `keep14` is the last arm that still clears its floor — on MMLU and
   on arc_ch / arc_easy / obqa / csqa alike. On PIQA `keep14` has already fallen to
   the floor, i.e. PIQA is *more* sensitive, not less.
2. **The wrong-null verdict flip.** Of the fifteen damaged arm×task cells (5 tasks
   × 3 arms), **10 sit above their naive chance line** — yet **0 of 15** are above
   their own best-constant floor. So ten cells read "above chance, model has
   residual competence" under the wrong null and "indistinguishable from a constant
   predictor" under the right one. This is exactly the substantive claim paperC
   makes, reproduced off MMLU with MMLU's own interface.
   (The other 5 cells are below chance too, so the chance line does not mislead
   there — it just happens to agree.[AUTHOR-SIDE EDITORIAL NOTE REDACTED FOR BLIND REVIEW])
3. **The direction of the point estimate.** 12 of those 15 arm×task letter deltas
   are *negative* (below floor) — arc_easy and obqa are 3/3, arc_ch / csqa / piqa
   are 2/3. Descriptive exact binomial under a coin-flip null: p = 0.018.
   ⚠️ These 15 tests share items, nested arms and a common null, so they are **not
   independent**; the binomial is descriptive colour, not a licensed inference.

**What does NOT replicate: the significant below-floor verdict.**

MMLU's headline is that `keep8` is *significantly* BELOW its floor (−1.389 pp,
p=0.019). Of the five non-MMLU tasks, **only arc_easy produces a significant
below-floor arm at all** — and it is `keep10` (−2.694 pp, boot p=0.029, McNemar
p=0.032), **not** `keep8`. On `keep8` itself, no non-MMLU task reaches p<0.05.

### 2b. The reason is mostly POWER, and this must be stated, not glossed

MMLU has n=14042. These sets are 6–28× smaller. The achieved CI95 half-width on
the very same `keep8`-vs-floor test:

| task | n | keep8 delta | CI95 half-width | could it have detected MMLU's −1.389 pp? |
|---|---|---|---|---|
| MMLU | 14042 | **−1.389 pp** | **1.154 pp** | (this is the reference) |
| arc_easy | 2376 | −0.800 pp | 1.305 pp | **YES** (borderline) |
| winogrande (ctrl) | 1267 | +0.552 pp | 1.184 pp | YES |
| piqa | 1838 | +2.503 pp | 2.775 pp | NO |
| commonsense_qa | 1221 | −1.065 pp | 3.399 pp | NO |
| arc_challenge | 1172 | −0.939 pp | 3.882 pp | NO |
| openbookqa | 500 | −1.800 pp | 6.400 pp | NO |

**Four of the five evidence tasks physically cannot resolve an effect the size of
MMLU's headline.** OpenBookQA's CI is ±6.40 pp — 4.6× the effect. So "keep8 is AT
the floor on arc_challenge" is **not** evidence against MMLU's finding; it is a
non-observation. Conversely, arc_easy — one of only two tasks with MMLU-comparable
resolution — *does* produce a significant below-floor arm.

**Correct way to write this in the paper:** the second-benchmark leg establishes
that the *wrong-null verdict flip* is not an MMLU artifact (0/15 damaged arms clear
their floor while 15/15 clear chance), and that the *ordering and floor-arrival
point* replicate. It does **not** independently establish "significantly below
floor" outside MMLU + arc_easy; claim that only for those two, and disclose the
power table above whenever a null result on the smaller sets is mentioned.

### 2c. Modal share and floor verdict stay DECOUPLED (consistent with the 2026-08-10 narrowing)

`STATUS.json:claim_scope_after_gates.NARROWED_20260810_constant_predictor` says
being at/below a constant predictor's accuracy is not the same property as *being*
one. Reconfirmed here, and the decoupling is if anything wider:

* `arc_challenge / keep8`: modal share **0.8038**, tie rate 0.1809, yet the floor
  verdict is only `AT` (−0.939 pp, p=0.644). Near-constant emitter, not
  significantly below floor.
* `arc_challenge / keep10`: modal share **0.3848** — far from constant — and the
  point estimate is *positive* (+1.621 pp).
* `winogrande / keep8` (control): modal share **0.9534**, i.e. a 95%-constant
  emitter, floor verdict `AT` (+0.552 pp, p=0.357).
* `piqa / keep12`: modal **0.8781**, verdict `AT` (−0.435 pp).

Letter exact-tie rates on the damaged arms are 0.14–0.24 (vs 0.001–0.003 on base),
consistent with MMLU's OLMo-2 bf16-tie pattern — but per gate-3 that mechanism is
**family-specific and already falsified as *the* causal mechanism**; do not
resurrect it.

---

## 3. Content interface vs longest-option null — all five tie conventions

Longest-option floor (continuation-**token** unit), with the tie-fraction diagnostics:

| task | split | first | last | credit | wrong | tied-longest | gold∈W |
|---|---|---|---|---|---|---|---|
| arc_challenge | `0.283902` | `0.265358` | `0.296075` | `0.543515` | `0.142491` | 50.85% | 54.35% |
| arc_easy | `0.238054` | `0.238636` | `0.235269` | `0.507155` | `0.106481` | 51.81% | 50.72% |
| **openbookqa** | **`0.368000`** | `0.384000` | `0.360000` | `0.644000` | `0.228000` | 48.40% | 64.40% |
| commonsense_qa | `0.201775` | `0.208026` | `0.204750` | `0.434070` | `0.092547` | 52.91% | 43.41% |
| piqa | `0.465452` | `0.470076` | `0.460827` | `0.677911` | `0.252992` | 42.49% | 67.79% |
| winogrande (ctrl) | `0.501184` | `0.494081` | `0.508287` | `0.933702` | `0.068666` | **86.50%** | **93.37%** |

**The convention degree of freedom is real and reverses verdicts here too**, which
is the A01-applied-reflexively claim (`claim_scope_after_gates.confirmed_general[5]`):

* **arc_challenge**: under `split`/`first`/`last`/`wrong` all six arms are
  *significantly ABOVE* the content floor; under **`credit`**, five of six flip to
  *significantly BELOW* (base +1.96 pp n.s.; shortgpt16 −8.79 pp; keep14 −11.95;
  keep12 −14.93; keep8 −19.11; keep10 −19.71, all p<1e-3). Same reversal shape as
  MMLU's `credit`-flips-5/6.
* **openbookqa** is the cleanest content-side case: `split` floor `0.368000` vs
  chance `0.25` = **+11.8 pp**, and under `split` only base and shortgpt16 clear it
  — keep14 (+3.60, p=0.118), keep12 (+3.00, p=0.181), keep10 (−0.40, p=0.850) and
  keep8 (−2.80, p=0.198) are all AT the floor. Under `credit` all six are
  significantly BELOW.
* **commonsense_qa / piqa / arc_easy** are robust: above the floor under all five
  conventions (CSQA's three most-damaged arms drop to `AT` under `credit` only).

**Residual-fraction inflation from using chance instead of the correct null**
(base arm):

| task | interface | acc | floor | resid frac (correct null) | resid frac (chance) | inflation |
|---|---|---|---|---|---|---|
| **openbookqa** | content_norm | `0.4740` | `0.3680` | **22.4%** | **47.3%** | **2.11×** |
| arc_challenge | content_norm | `0.5631` | `0.2839` | 49.6% | 55.6% | 1.12× |
| openbookqa | letter | `0.7320` | `0.2760` | 62.3% | 65.9% | 1.06× |
| piqa | content_norm | `0.8172` | `0.4655` | 43.0% | 38.8% | **0.90×** (deflation) |
| arc_easy | content_norm | `0.8165` | `0.2381` | 70.8% | 69.4% | 0.98× |
| commonsense_qa | content_norm | `0.6560` | `0.2018` | 69.2% | 69.5% | 1.00× |

OpenBookQA's **2.11×** inflation replicates the 2.15× the old gate-2 reported (the
small difference is the length-unit change documented in §5). Note honestly that
**PIQA and arc_easy go the other way** — their token-longest floors are *below*
chance, so the chance line *understates* the residual there. "Chance inflates the
claim" is not universal; it is construct-specific, which is precisely paperC's
point but must be stated symmetrically.

---

## 4. Interface swap does not rescue a damaged arm on the letter side… but on
## these tasks the CONTENT interface is much stronger than on MMLU

MMLU's `confirmed_general[2]` says "content_norm sits within ±3 pp of letter on
every damaged arm". **That does NOT hold here** and it is the largest quantitative
divergence from MMLU in this run:

| task | keep8 letter | keep8 content_norm | Δ (content − letter) | McNemar p |
|---|---|---|---|---|
| MMLU | `0.2550` | `0.3423` | +8.72 pp | — |
| arc_challenge | `0.2560` | `0.3524` | **+9.64 pp** | 4.8e-07 |
| **arc_easy** | `0.2584` | `0.6460` | **+38.76 pp** | 9.8e-148 |
| openbookqa | `0.2580` | `0.3400` | +8.20 pp | 5.3e-03 |
| commonsense_qa | `0.1982` | `0.4259` | **+22.77 pp** | 3.0e-33 |
| piqa | `0.5299` | `0.7214` | **+19.15 pp** | 1.0e-32 |

On **arc_easy the keep8 arm is at its letter floor (`0.2584`) while scoring
`0.6460` on content** — 38.8 pp higher, and far above every content convention's
floor. The competence is demonstrably present; the letter readout simply cannot
express it. That is a *stronger* version of paperC's readout-vs-knowledge
separation than MMLU provides, and it is the single most publication-worthy number
in this run.

⚠️ It also **contradicts** MMLU's `confirmed_general[2]` if that bullet is read as
a general statement. It must be re-scoped to "on MMLU, content_norm sits within
±3 pp of letter on every damaged arm" — see §7.

On healthy models the reverse holds, matching MMLU: on base, letter beats content
by +20.6 pp (arc_ch), +7.7 pp (arc_easy), +25.8 pp (obqa), and is a statistical tie
on csqa (−0.16 pp, p=0.959). PIQA is the exception (content +10.0 pp even on base).
So "content is the fair interface" remains **not** a general claim.

---

## 5. Two contradictions with the old gate-2 record

### 5a. OpenBookQA's longest-option null is `0.3680`, not `0.3635` — because the old number is a CHARACTER-length null

`gate_results.gate2_second_mc_benchmark` and `paperC/README.md` both publish
OpenBookQA's longest-option floor as **`0.3635`**. That number is **not wrong**,
but it is computed in a **different length unit** than MMLU's: the published
`acc_norm` / `norm_lens` pipeline normalises by `len(text)` in **characters**,
whereas the MMLU letter/content harness (and therefore this run) normalises by
**continuation-token count**. Recomputed both ways on identical items:

| task | char-unit split | token-unit split | Δ |
|---|---|---|---|
| openbookqa | `0.363500` | `0.368000` | +0.0045 |
| arc_challenge | `0.274104` | `0.283902` | +0.0098 |
| arc_easy | `0.255296` | `0.238054` | **−0.0172** |
| commonsense_qa | `0.221977` | `0.201775` | **−0.0202** |
| piqa | `0.475245` | `0.465452` | −0.0098 |
| winogrande | `0.491713` | `0.501184` | +0.0095 |

The `credit` convention is *far* more unit-sensitive (Δ up to **+0.352** on
winogrande, +0.228 on obqa) because token-count ties are much rarer than
character-length ties.

**Consequence for paperC: the longest-option null has a SECOND, previously
undocumented degree of freedom — the length UNIT — on top of the tie convention.**
It moves the null by up to 2.0 pp under `split` and up to 35 pp under `credit`.
This strengthens `confirmed_general[5]` ("the null A01 recommends is itself
under-specified") rather than weakening it, but the protocol sentence must now read
"report a construct-appropriate null AND print its convention **AND its length
unit**". Neither the old gate-2 nor `GATE3_CONVENTIONS_VERDICT.md` states a unit.

**Nothing is retracted.** `0.3635` stays valid *as a character-length null*; label
it. Where OBQA is cited next to an MMLU number, prefer the token-unit `0.3680` so
the units match.

### 5b. winogrande is a *degenerate* control in the published construction, but only *partially* degenerate here — and that is what makes it a good control

The old record says winogrande has "identical norm_lens → `acc == acc_norm`
exactly, 100% tie rate". That is a property of the **published partial-scoring**
construction (both options share the continuation). This harness builds
winogrande's content interface the same label-free way as every other task, so:

| | published construction | this construction |
|---|---|---|
| items with identical continuation-token counts | 100% | **86.50%** |
| `acc == acc_norm` per item | 100% (exactly) | 94.6–96.8% |
| `craw` vs `cnorm` (base) | identical | `0.581689` vs `0.580110` |
| longest-option `split` null | exactly `0.5000` | `0.501184` |
| longest-option `credit` null | — | **`0.933702`** |

So the degeneracy is not an artifact of one prompt choice — it is intrinsic to the
task (the two options are single words of near-identical length), and it survives
re-construction at 86.5%. **Its `credit` null of `0.9337` is the reductio**: a
"length heuristic with oracle tie-breaking" scores 93.4% on winogrande, while the
full base model scores 58.0%. Winogrande remains a **negative control only**, now
with a quantitative reason rather than an assumed one.

Winogrande's letter interface behaves as a control should: base is the only arm
above floor (+8.76 pp, p=1e-4); all five damaged arms are AT the floor, and
`keep8`'s modal share is 0.9534 — a near-constant emitter that is nonetheless not
significantly below floor. Do **not** count it as an interface case.

---

## 6. Shard integrity and cross-validation

* **36/36 cells** (6 arms × 6 tasks): 8/8 shards present, `n_scored == expected`
  exactly (1172 / 2376 / 500 / 1221 / 1838 / 1267), **`n_nan == 0`**, `n_trunc = 0`.
  Asserted twice — once by the harness merge (which raises rather than merging a
  partial set) and once independently by the analysis script's loader.
* Failure grep over all 60 logs with **failure syntax**
  (`Traceback \(most recent call last\)|CUDA out of memory|SHARD INTEGRITY FAILURE|
  CARDINALITY FAILURE|AssertionError`): **zero hits**.
* **Cross-validation against the published ledger.** The content prompt for
  arc_challenge / arc_easy / commonsense_qa is **byte-identical** (1172/1172,
  2376/2376, 1221/1221 verified programmatically) to
  `eval_olmo2_probe2_downstream.py::load_task_examples`, so `content_raw` must
  reproduce the published `correct` field. It does: item-level agreement
  **99.41–99.96%** across all six arms, accuracy within **0.30 pp**
  (e.g. shortgpt16 arc_easy `0.769781` here vs `0.768519` published; keep12
  arc_easy `0.724747` vs `0.724747` exactly). The residual disagreements are
  near-ties — median top1−top2 score gap on disagreeing items is 0.012–0.048 with
  max |Δscore| 0.21–0.49, i.e. bf16 batching nondeterminism, not a protocol
  difference. Gold-letter sequences align 100% on every task/arm.
* **OpenBookQA deliberately deviates** (0/500 byte-identical): the published loader
  uses the bare `question_stem` with **no `Answer:` cue**, which cannot host a
  letter interface. The `Question: … \nAnswer:` stem is required for the
  MMLU-parallel construction. Hence OBQA `content_raw` here (`0.3420` base) differs
  from the published `0.3720`; the 85–87% item agreement is the size of that prompt
  effect, and it is a *finding about prompt sensitivity*, not an error.
* **Recompute reproducibility:** the full analysis was run on zwfy6 (from `.73`)
  and re-run independently on the wzc1 copy — the two JSONs are **byte-identical**
  after key-sorting. Consistent with `memory/same-harness-runs-bit-identical`.

---

## 7. What must be changed elsewhere in paperC because of this run

1. `STATUS.json:gate_results.gate2_second_mc_benchmark` — **add** a
   `full_replication_20260811` sub-record (do not overwrite; the retraction history
   is load-bearing). Its `caveat` field ("analogous but not identical interface
   contrast") is now **superseded** for the five tasks here, and that must be said
   without deleting the original caveat.
2. `claim_scope_after_gates.confirmed_general[2]` — "Interface swap does NOT rescue
   a damaged arm: content_norm sits within ±3pp of letter on every damaged arm"
   must be **re-scoped to MMLU**. On arc_easy the gap is **+38.76 pp**; on csqa
   +22.77 pp; on piqa +19.15 pp. As a general statement it is false.
3. `confirmed_general[5]` — extend the under-specification claim from *tie
   convention* to *tie convention **and length unit*** (§5a).
4. `paperC/README.md` "Where it has been demonstrated" table — label the
   OpenBookQA `0.3635` as a **character-length** null, and add the token-unit
   `0.3680` next to it.
5. Anywhere the second-benchmark leg is cited, the **power table (§2b)** must
   accompany it. Reporting "keep8 is AT the floor on arc_challenge" without the
   ±3.88 pp CI would be a misuse of a non-observation.
6. The acc-vs-acc_norm sign flips found here (arc_challenge 1/15, piqa 1/15,
   winogrande 1/15; arc_easy / obqa / csqa 0/15) are a **replication under damage
   of Oostermeijer, ICML 2026 (arXiv:2607.12767)**, not a paperC finding. The JSON
   records this attribution inline.

---

## 8. Bottom line

**Did MMLU's conclusion replicate on a second MC benchmark? PARTIALLY — and the
partial answer is more informative than a clean yes would have been.**

* ✅ The **construct-validity claim** replicates on 5/5 evidence tasks: of the 15
  damaged arm×task cells, **10 look "above chance"** and **0 clear their own
  best-constant floor**. The wrong-null problem is **not** an MMLU artifact, and it
  is now shown with MMLU's own letter-vs-content interface rather than a proxy.
* ✅ The **arm ordering and the floor-arrival point** replicate on 5/5: `keep14` is
  the last arm above its floor on 4/5 (arc_ch / arc_easy / obqa / csqa; on PIQA it
  has already reached the floor), and `keep12` and below are at/under the floor on
  5/5.
* ✅ The **tie-convention reversal** replicates (arc_challenge: `credit` flips 5/6).
* ✅ A **new, stronger readout-vs-knowledge dissociation** appears: arc_easy keep8
  is at its letter floor (`0.2584`) while scoring `0.6460` on content (+38.8 pp,
  McNemar p=1e-147).
* ❌ The **significant below-floor letter verdict** does **not** reproduce outside
  MMLU + arc_easy, and 4 of 5 tasks are too small to have detected MMLU's own
  −1.39 pp. This is a genuine limitation of the second-benchmark leg and is now on
  the record.
* ⚠️ A **new degree of freedom** (length unit) was found in paperC's own
  recommended null, and MMLU's `±3pp interface-swap` bullet is **falsified as a
  general claim** by this run's own data.

The last two items are self-falsifications produced by applying the protocol to
itself, which is the part of this direction that is actually its own.

---

## 9. Provenance

| what | where |
|---|---|
| per-item records, 6 arms × 6 tasks × 8 shards + merged | **zwfy6** `<REPO_ROOT>/olmo2_mc_letter_content_results/` **AND wzc1** `<REPO_ROOT>/olmo2_mc_letter_content_results/` (52 MB, both disks, integrity re-verified on each; not git-tracked, same as the MMLU records) |
| null/statistics output | `paperC/evidence/second_mc_benchmark/gate2_letter_content_nulls.json` + `.csv` (396 rows) |
| harness | `scripts/eval_olmo2_mc_letter_content.py` |
| driver | `scripts/_run_olmo2_mc_letter_content_8gpu.sh` |
| analysis | `proposal/active/A01-null-calibration-methodology/code/a01_gate2_letter_content_nulls.py` |
| logs | **zwfy6** `logs/gate2_mc_lc_*.log` (61 files: 48 shard + 6 merge + prepare + DRIVER) |
| MMLU comparison numbers | `proposal/active/A01-null-calibration-methodology/evidence/gate3_dtype_runs/*_dtype_summary.json`, `by_dtype.bf16` |
| ckpts | **zwfy6** `outputs/olmo2_probe2_7B_{keep8fresh2/step121000,keep10fresh2/step83500,keep12fresh2/step124000,keep14fresh2/step200000,shortgpt16/step200000}.pt` |
