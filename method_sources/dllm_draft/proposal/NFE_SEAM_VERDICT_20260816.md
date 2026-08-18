# Adjudication — the dLLM cost-accounting seam ("nobody reports total forward passes")

**Written**: 2026-08-16 · **GPU used: 0** (literature + on-disk re-reading only; `/apdcephfs_zwfy6`
is not mounted on LOCAL, verified by `ls` → `No such file or directory`, so no per-item recount was
attempted) · **Inputs**: 5 source-reading passes + 15 adversarial votes (14 refute) · **This document
is the decision.**

The seam under adjudication, verbatim as posed:

> "published dLLM adaptive-length methods do not report an honest TOTAL forward-pass count for their
> own adaptive loop; the specific flaw is per-token normalisation hiding that adaptive methods change
> the denominator."

---

## 1. VERDICT

**SEAM DEAD.** Both of its clauses are false at source, each killed by a paper I re-verified myself
today, and one of the two killers is **not** concurrent.

- **Clause B ("the specific flaw is per-token normalisation hiding the denominator") is false for the
  flagship adaptive-length paper.** DAEDAL prints `N_token` — *"the average total tokens"*, an
  absolute count — in **every cell** of Tables 1/2/4/5, immediately beside the flattering `E_ratio`.
  HUMANEVAL: `N_token` = 64/128/256/512/1024/2048 for the six fixed rungs and **813** for DAEDAL.
  The denominator is *disclosed*, not hidden. RETRIEVED from `/tmp/2508.00819v2.txt` (arXiv v2 HTML,
  cached), read directly.
- **Clause A ("no honest TOTAL forward-pass count for their own adaptive loop") is falsified by
  VoidPadding**, whose Table 3(b) prints mean **NFE per task for its own loop and for two
  competitors' loops** — Daedal 228.82, ρ-[EOS] 139.70, VoidExpansion 172.10 (mean over four
  benchmarks; per-task HEval 254.66 / 191.52 / 73.06) — with the accounting rule stated verbatim:
  *"In all cases, NFE counts only decoding forward passes, while wall-clock time includes both canvas
  selection and decoding."* RETRIEVED from `/tmp/pre/t_2606.17999.txt`, read directly.
- **A non-concurrent absolute-total counterexample also exists**: ρ-EOS (2026-01, ~6.5 months old)
  prints `T_runtime` — *"the runtime spent on the evaluation (in seconds)"* — for all six fixed
  rungs **and** DAEDAL **and** both of its own variants, on 4 benchmarks × 9 columns. HUMANEVAL row:
  111 / 230 / 543 / 1474 / 4569 / 16046 | DAEDAL 1283 | Sym 580 | Asym 593. So even the
  concurrency defence does not save the universal.

**No claim of the form "the field does not report an absolute total for its adaptive loop" may be
made.** The reform on the compute axis is also already published as a protocol (CaRE, §5).

### The only thing that survives, stated so it cannot be inflated

Not a paper. A **methods paragraph**, and it is a *correction to* the seam rather than a version of it:

> "For canvas-**expanding** length controllers, a forward-pass count is not a sufficient cost unit,
> because attended work per forward grows with the canvas. On DreamOn-v0-7B / HumanEval+ (n=164),
> holding the 512-token generation ceiling fixed and sweeping only the initial canvas 8→128, true
> counted forward passes (a wrapper on `model.forward`) rise **3.444×** in the mean (172.3 → 593.4)
> while attended context rises **6.019×** (39,944.5 → 240,413.9 tokens). A protocol that standardises
> actual NFE alone therefore understates the compute growth of length adaptation by ~1.75× on this
> axis."

Both numbers are on wzc1 and re-derivable at 0 GPU (§4). This paragraph is worth **one Related-Work
sentence plus a diagnostic citation**, and it is a live objection to CaRE's own scoping (§7, item 2).
It is *not* a direction. Do not spend GPU on it.

---

## 2. The five candidates (+ the four papers that actually decided it)

Venue authority abbreviations: **AA** = ACL Anthology `.bib` fetched today (HTTP 200); **OR** =
`api2.openreview.net/notes/search` `venueid` + `invitations`; **DBLP** = `dblp.org/search/publ/api`;
**arXiv** = `export.arxiv.org/api/query?id_list=`.

| Paper | Verified venue + authority | Verbatim metric names | Absolute **total forward** count? | Length budget | Preempts seam |
|---|---|---|---|---|---|
| **SDAR** `2026.findings-acl.1110` | **Findings of ACL 2026**, pp. 22058–22075 — **AA** (`booktitle = "Findings of the Association for Computational Linguistics: ACL 2026"`); preprint `arXiv:2510.06303` — **arXiv** | "Effective Tokens Per Forward Pass (TPF)", "Tokens Generated Per Second (TGS)", "Realized Tokens Per Forward Pass", "Throughput (tokens/s)" | **NO — ratio only.** arXiv v3 defines `TPF = Total Generated Tokens / Total Forward Passes` (RETRIEVED, `/tmp/arxiv_2510.06303.txt`, "Total Forward Passes" ×2); neither numerator nor denominator is ever printed | **PINNED** (block L′∈{4..64}; static decode default) | **NO** — and it is not adaptive-*length* at all |
| **2510.18480** "How Efficient Are Diffusion Language Models?" | **preprint**, CoRR — **arXiv** (v1 2025-10-21, v2 **withdrawn**, v3 2025-11-10); DBLP type *Informal and Other Publications* | "Throughput", "tokens per second", "Arithmetic Intensity", "FLOPs/token" (defined, never computed) | **NO** | **PINNED by force** — forces AR/block baselines past their own `<eos>` to the target length | **NO** — owns the *problem statement*, prescribes nothing, evaluates zero adaptive-length methods |
| **Saber** `2026.acl-long.165` | **ACL 2026 Long** (Vol. 1), Peking Univ., editors Liakata/Moreira/Zhang/Jurgens, San Diego — **AA**; also `arXiv:2510.18165` — **arXiv** | "Step (avg generation steps per sample)", "Time (total generation time)" | **NO** — reports absolute *Steps* + *Time*, but never forwards; Algorithm 1 conditions the model **twice** per counted Step (lines 3 and 12) and Appendix C's overhead enumeration omits the extra forward | **PINNED at 256** | **PARTLY** — best-practice end of the distribution, still fails the forward-count test |
| **Focus-dLLM** `2026.acl-long.556` | **ACL 2026 Long** — **AA** | "Throughput (tokens/s)", "Throughput(16K)", "Gen. Len.", "Steps" | **NO** | **PINNED wherever throughput is reported** (§6.2: *"both the generation length and generation steps are fixed at 256"*); Steps == Gen. Len. in every Table-4 row | **NO** — fixed denominator, disclosed ⇒ out of scope, not a violator |
| **UNCODE** `2026.acl-long.311` ("Empirical Analysis of Decoding Biases in Masked Diffusion Models") | **ACL 2026 Long** — **AA** | "matched decoding budget", speedup ×, "Decoding Step", "Trivial Token Ratio" | **NO** — ratio-only for the adaptive-*step* axis it studies | **PINNED** (steps = sequence length) | **NO** |
| — **DAEDAL** `arXiv:2508.00819` | **ICLR 2026 Poster** — **OR** `venueid = ICLR.cc/2026/Conference`, Submission1382, `Camera_Ready_Revision` present (re-verified today) | `Acc`, `E_token`, `N_token` (*"average total tokens"*), `E_ratio` | **NO forwards** — but **YES an absolute total-token count** (`N_token`); `\bNFE\b`=0, `\bFLOPs?\b`=0, wall-clock=0 in full text | **FLOATING and disclosed** | **KILLS CLAUSE B** |
| — **ρ-EOS** `arXiv:2601.22527v2` | **preprint** — OR returns only the DBLP mirror (`dblp.org/journals/CORR/2026`); ~6.5 months old ⇒ **not concurrent** | `Acc`, `E_token`, `N_token`, `E_ratio`, **`T_runtime` (seconds)** | **NO forwards**, **YES an absolute total wall-clock** for its own loop *and* DAEDAL *and* 6 fixed rungs | FLOATING and disclosed | **KILLS the "no honest total" universal** |
| — **VoidPadding** `arXiv:2606.17999v2` | **arXiv 2026-06-16** (**arXiv**) + **COLM 2026 Efficient-Reasoning Workshop poster** (**OR** `venueid = colmweb.org/COLM/2026/Workshop/Efficient_Reasoning`, `odate` present) + **ACL ARR 2026 August Submission** (**OR**); DBLP CoRR 2026 *Informal* | **"NFE"** (mean, per task), "wall speedup ×" | **YES** — mean NFE for VoidExpansion **and** DAEDAL **and** ρ-EOS | FLOATING (adaptive canvas expansion) | **KILLS CLAUSE A** |
| — **CaRE** `arXiv:2607.24763v1` | **preprint only** — DBLP `CoRR 2026, Informal and Other Publications`; **OR search returns NO CaRE record** (see §7 item 1 — do not attach the ICML-workshop venueids to it) | "actual NFE", "nominal steps" | **YES** — Table 3: none 128 (1.0×), random 257 (2.0×), high_entropy/low_conf/conf_ent/agreement 437 (3.4×), running_confidence 513 (4.0×) | **PINNED** (`Generation budget: 128 tokens`; "adaptive length" ×0, "variable-length" ×0) | **Owns the reform on the step axis**; does not touch the length axis |
| — **SmartCrop** `arXiv:2603.06123` "DLMs Are Natively Length-Aware" | **ICML 2026 Rejected_Submission** — **OR** `venueid = ICML.cc/2026/Conference/Rejected_Submission`; **DBLP** CoRR 2026 *Informal* | "FLOPs Saved (%)", "Avg. Processed Length" | **NO** (relative FLOPs reduction) — but denominator disclosed (`L_p`, `Avg. Processed Length`, `L_c = L_p + L_new`), paired bootstrap on FLOPs, 5000 resamples | FLOATING and disclosed | **NO**, but shows the FLOPs axis is already in use on the length axis |

---

## 3. The crux: a RATIO vs an ABSOLUTE TOTAL

This is where the seam had to win, and it is where it loses. The honest statement of the distinction,
followed by the concession the evidence forces:

**Why a ratio genuinely does not close the accounting question.** `TPF = tokens / forwards`,
`FLOPs/token`, and `TPS = tokens / second` are all *quotients*. Publishing only a quotient makes the
total unrecoverable, and — decisively — **inverting a per-token normalisation is not de-normalising
it**. SDAR is the clean instance: it is the only paper in the set whose metric is *denominated in
forward passes*, its arXiv definition names both `Total Generated Tokens` and `Total Forward Passes`,
and it prints **neither** — only their quotient, with AR pinned at "TPF axiomatically 1", so the
number can only ever express a per-step advantage against a fixed-denominator baseline. Focus-dLLM's
`Throughput(16K)` and SmartCrop's `FLOPs Saved (%)` are likewise quotients.

**Why that nevertheless fails to kill any of these papers — the precise condition.** A ratio is
illegitimate only when **the denominator floats AND is undisclosed**. In the read set, *no paper
satisfies both*:

| Denominator | Disclosed | Papers | Ratio legitimate? |
|---|---|---|---|
| PINNED | yes, at the point of measurement | SDAR (block L′, static T), Focus-dLLM (*"generation length and generation steps are fixed at 256"*), Saber (256), UNCODE (steps = seq len), CaRE (gen = 128), 2510.18480 (forces baselines past `<eos>`) | **YES** — a fixed-denominator ratio is a faithful summary of a fixed total |
| FLOATING | yes, printed alongside the ratio | DAEDAL (`N_token`), ρ-EOS (`N_token` + `T_runtime`), SmartCrop (`Avg. Processed Length`), VoidPadding (mean `NFE`) | **YES** — the total is printed or recoverable |
| FLOATING | **no** | *none found* | — |

So the seam's mechanism has **no instance**. The strongest thing left is a narrower and much duller
observation: for the floating-denominator papers the disclosed total is in **tokens** (DAEDAL,
SmartCrop) or **seconds** (ρ-EOS), and only VoidPadding's is in **forwards**. That is a *unit*
complaint, not a *hiding* complaint — and §4 shows the unit complaint cuts against "count forwards"
too.

One asymmetry worth recording because it is the field's nearest brush with the seam and it is
*evidence for the field, not against it*: SDAR itself flags that its per-forward ratio can be gamed
by degeneration — *"the 1.7B model shows higher TPF on IFEval, which we attribute to SFT data
mismatch that induces repetitive degeneration and thus entropy collapse (inflated TPF)"*. It
diagnoses this as a model pathology, not a metric pathology, and does not generalise it. Cite as
"SDAR conjectures", never "SDAR shows" — the attribution is unsupported in the source (no repetition
rate, no length distribution).

---

## 4. Our counter-data: is it sound?

**It is the CORRECTED counted series, not the retracted one. Unambiguously.** Four independent
proofs, all re-verified today on wzc1:

1. The driver `proposal/archive/A05-structural-dllm-cost-frontier/code/a05_k1_dreamon_canvas.py` is a
   new harness whose docstring opens with **"FIX 1 — true NFE"** and whose `install_counters()`
   (line 135) wraps the model: `state["nfe"] += 1` (line 147), assigned to `model.forward` (line 157),
   `reset_counters()` per item (line 255), batch size 1 (line 277 `torch.tensor([initial], ...)`).
2. The same run passes **`output_history=False`** (line 282). The retracted quantity
   `len(output.history)` is therefore not merely unused but *unavailable* — it would be `None`.
   **The K1 cells physically cannot contain it.**
3. The retracted series is a **different pair of numbers**: 265.88 (HE+) / 135.65 (MBPP+), from the
   old driver, at canvas=8. `STATUS.json` records the pairing verbatim: *"r1 non-null 164/164 &
   378/378 with means 265.88/135.65 (= mean(len(history))) … True counted NFE 172.3/153.4. NOT a
   uniform inflation: HE+ 265.88→172.3 DOWN but MBPP+ 135.65→153.4 UP."*
4. Signature check: `nfe_median` equals the canvas **exactly** (8.0 / 32.0 / 128.0) in every cell,
   which is what a true forward count must do at `number_transfer_tokens=1`; `len(history)` (appended
   at 3 sites) would sit at ~3× that.

**And it still cannot carry the seam.** Five defects, four of them measured:

| # | Defect | Evidence (wzc1, re-read today) |
|---|---|---|
| 1 | **Forwards is the wrong unit** — the finding that also kills the reform framing | NFE mean grows **3.444×** (172.3→593.4) but `tokens_fed_effective_mean` grows **6.019×** (39,944.5→240,413.9) across HE+ canvas 8→128 |
| 2 | **The median item is not adaptive at all** | `nfe_median == canvas` exactly in all 5 graded cells ⇒ zero expand, zero delete on the median item |
| 3 | **The mean is a cap artefact** | `top15pct_share` = .9595 / .7056 / .5471 (HE+ c8/c32/c128) and .9555 / .6186 (MBPP+); 6–13% of items sit at DreamOn's ~2060–2180 forward cap. A05's own pre-registered condition **F5 fired** on exactly this |
| 4 | **No third MBPP+ point** | `cells_not_run`: `mbpp_c128` *"was killed after 30/378 items and is NOT graded"*. The 8→32→128 story is single-benchmark |
| 5 | **n = 1 per cell** | stochastic sampler (T=0.2, top_p=0.9, `alg=entropy`); per-`task_id` seeding gives paired comparability across canvases but no run-to-run band on the tail-dominated mean |

The five graded cells, for the record (`evidence/cells/*.json` for cost, `evidence/cells_corrected/`
for HE+ quality — they live in **different files** and must be joined; quoting `cells/*.json` for
both axes imports a retracted pass@1):

| cell | canvas | NFE mean | median | total | p90 | attended tok (mean) | gen tok (mean) | pass@1+ as-run | pass@1+ corrected |
|---|---|---|---|---|---|---|---|---|---|
| `he_c8` | 8 | 172.29 | 8 | 28,256 | 122 | 39,944.5 | 2.35 | .1280 | **.1341** |
| `he_c32` | 32 | 393.70 | 32 | 64,566 | 2,084 | 124,348.0 | 12.87 | .2134 | **.2561** |
| `he_c128` | 128 | 593.43 | 128 | 97,322 | 2,180 | 240,413.9 | 48.53 | .1707 | **.4817** |
| `mbpp_c8` | 8 | 153.44 | 8 | 58,000 | 40 | 23,367.4 | 1.57 | .0899 | (HE+ only regrade) |
| `mbpp_c32` | 32 | 466.01 | 32 | 176,152 | 2,084 | 101,202.1 | 11.43 | .3545 | (HE+ only regrade) |

Corrected HE+ pass@1(plus) is **monotone increasing**; "DreamOn degrades at large canvases" was
retracted as a harness artefact (`corrected_not_replacement`: generation byte-identical, no model
loaded, 0 GPU).

---

## 5. What we may claim, and what we may NOT

### MAY (diagnostic use — survives, and is the only live residue)

- **Auditing a third party's baseline configuration.** One sampler-config integer moved MBPP+ pass@1
  from .0899 → .3545 (**+26.6 pp**), and a post-processing stitch defect additionally understated
  HE+ by up to **31.1 pp** at c128 (parseability .2866 → .9634). Both are larger than most reported
  method gains. This is an *evaluation-practice / harness-validity* result and A05's closeout says so
  explicitly (`is_this_a05s_contribution: "NO"`).
- **Citing `2510.18480` as the established problem statement**, then noting precisely what it does not
  do: it prescribes nothing (`we recommend`/`we propose`/`should report`/`guideline`/`checklist` = 0
  hits), computes zero FLOPs/token values, evaluates zero adaptive-length methods, excludes quality by
  definition (footnote 1), files variable length under future work, and achieves comparability only by
  forcing baselines past their own `<eos>`.
- **Citing CaRE as the published reform on the step axis** and Saber as the best published
  step+time accounting — both with the caveats in §2.
- **The unit correction in §1**, as one methods sentence, reported as mean **and** median **and**
  total **and** tail share (never means alone — that reproduces the sin).

### MAY NOT

- **Any universal about the field's reporting.** "No adaptive-length paper reports a total forward
  count" is false (VoidPadding). "No adaptive-length paper reports an absolute total cost" is false
  (ρ-EOS `T_runtime`, non-concurrent). "The specific flaw is per-token normalisation" is false
  (DAEDAL `N_token`).
- **The canvas-sensitivity phenomenon claim.** DAEDAL's Table 1 already publishes a canvas sweep
  64→2048 on LLaDA-Instruct-8B (HUMANEVAL 18.9 → 47.6, +28.7 pp) and ρ-EOS replicates it
  independently. **Formally killed. Must not be resurrected**, including in disguised form ("cost of
  canvas", "quality is bought with canvas"). A05 is `ARCHIVED` with `POSTMORTEM.md`; its cost claim
  died on **4 of 5** pre-registered conditions (F1, F2, F3, F5); `novelty_still_owed: "YES"` was never
  cleared (Semantic Scholar HTTP 429 on every attempt).
- **"AR is dominated" or any cross-family cost claim from A05 assets.** F3 is fatal and on disk:
  Qwen2.5-Coder-7B on HE+ is `.5244` pass vs Scaffold `.177`, at `206.4` vs `13,980.0` tokens_fed
  (**~68× cheaper and +.35 more accurate**) — strict Pareto domination on the axis this repo
  designates cross-family-comparable.
- **Claiming the 2-forwards-per-Step reading of Saber as fact.** It is a reading of Algorithm 1
  (lines 3 and 12 condition on different sequences); Saber releases no code (0 `github` hits), so it
  cannot be recounted. Say "Algorithm 1 implies", not "Saber performs".

---

## 6. Kill gate — **not applicable**

The seam is dead on literature, not pending on measurement, so no gate is offered and **no GPU
should be requested for it**. Recorded for completeness, in case MAIN wants only the §1 methods
paragraph:

- **Cheapest first step: 0 GPU, already done in this document.** The two numbers (3.444× NFE vs
  6.019× attended tokens) are computed from `evidence/cells/*.json` on wzc1. Nothing further is
  needed to state the paragraph.
- **The version that would make it a measurement rather than a paragraph is NOT affordable and NOT
  worth it.** Filling `mbpp_c128` costs ≈ 8–9 GPU-h by analogy with `he_c128` (7.71 GPU-h billed,
  6.07 compute) and would buy only a third point on a second benchmark for a claim that is a
  correction to someone else's protocol. A05's own accounting: 5 cells = **14.39 compute / 20.66
  billed** GPU-h, against a 25 GPU-h cap. Under tonight's zero-GPU budget it is moot; under any
  budget it loses to anything with a live hypothesis.
- **Honest expected outcome if run anyway**: `mbpp_c128` NFE mean ≈ 600–700 with median exactly 128
  and ~50–60% of NFE mass in the top 15% of items — i.e. it would confirm the pattern and change no
  scientific conclusion. That is the definition of an experiment not worth running.

---

## 7. Unresolved — with the authority tried and its exact error

1. **CaRE's venue and identity — resolved *negatively*, and there is a trap here.** DBLP
   (`q=Compute-aware+Remasking+Evaluation`, JSON) returns exactly 1 hit: `CoRR | 2026 | Informal and
   Other Publications`. OpenReview search for the CaRE title returns **no CaRE record**; its top hits
   `HIMiqnTqLD` (`ICML.cc/2026/Workshop/SPIGM`) and `Bew2D82sWR` (`ICML.cc/2026/Workshop/AdaptFM`) are
   a **different paper** — *"Re-evaluating Confidence Remasking in Masked Diffusion Language Models"*,
   Frkovic / Jazbec / Zhang / Naesseth / Bogunovic / Nalisnick, which re-evaluates **WINO** (abstract
   fetched and read today). **Do not attach those venueids to CaRE.** CaRE's author list and any venue
   beyond CoRR remain UNVERIFIED; `api2.openreview.net/notes?id=` and `?forum=` both returned
   `count: None` with an empty note list for those IDs, so per-note fetch is not a usable authority
   here — only `/notes/search` returned content.
2. **CaRE has a broken forward reference, and it is the one that matters to us.** Its Appendix C says
   *"Because per-step forward passes dominate MDLM runtime, NFE differences translate to proportional
   latency differences on matched hardware; **Section A discusses limits of this approximation**."*
   I read Appendix A (offsets 40637–42600 of the extracted text): it is *"Ethical consideration and
   Future Work"* and never discusses that approximation. Sequence length appears once, in the
   statistical-caveats list, as one of the factors that *"also matter but produce smaller measured
   effects"*. **On our own data the sequence-length effect is a 1.75× discrepancy in the cost ratio,
   which is not smaller.** RETRIEVED and read; the inference that this is an unclaimed opening is
   mine (INFERRED).
3. **SDAR's camera-ready deletion of the TPF formula — half verified.** RETRIEVED myself: the formula
   `TPF = Total Generated Tokens / Total Forward Passes` **is** in `arXiv:2510.06303` (cached text,
   `"Total Forward Passes"` ×2). The claim that the ACL camera-ready **deleted** it (probe: `"Total"`
   = 0 hits in the 18-page Anthology PDF) I did **not** re-verify — I confirmed only the Anthology
   metadata (HTTP 200, Findings of ACL 2026, pp. 22058–22075). Mark as INFERRED-from-probe. The
   *motive* for any deletion is unknowable (ACL 2026 Findings reviews are not public) — **do not
   assert one.**
4. **Whether SDAR's or Saber's code logs an absolute forward counter.** SDAR cites an engine in a
   footnote (arXiv v3 names JetEngine; camera-ready text says Nano-vLLM + an unnamed industrial
   engine); Saber releases nothing (0 URLs, 0 `github` in full text). Not checked — 0-GPU checkable if
   anyone ever needs the "they had the number and did not print it" phrasing, which they should not.
5. **Per-item provenance is one disk deep.** Raw rows live only at
   `/apdcephfs_zwfy6/.../dllm_draft/runs/a05_k1/<cell>/metrics.rank*.jsonl`; `ls /apdcephfs_zwfy6` on
   LOCAL → `No such file or directory`. Aggregates + full order statistics survive on wzc1
   (`evidence/a05_closeout_cost_audit.json` carries mean/median/total/p90/max/top15pct_share), so
   every number in this document is re-derivable from wzc1 alone; a fresh per-item recount is not.
6. **Tooling traps that produced false readings and will again.** (a) Case-insensitive `NFE` matches
   *confidence* / *inference* — 51 spurious hits in Focus-dLLM, 45 in 2510.18480; always use
   `\bNFE\b` case-sensitive. (b) DBLP's API 500s on `+`-joined long queries
   (`q=Diffusion+Language+Models+Natively+Length+Aware` → HTTP 500 HTML) but succeeds with `%20`
   (`q=Natively%20Length-Aware` → 1 hit, CoRR 2026). (c) Anthology PDFs arrive truncated at 4096-byte
   multiples through the proxy; a `page_count == 0` with valid metadata is the tell — resume with
   `-C -` and check `%%EOF` against `Content-Length`.
