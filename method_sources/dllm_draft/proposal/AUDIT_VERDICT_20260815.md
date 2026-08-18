# AUDIT VERDICT — 7 dLLM proposals, adversarial panel

**Written**: 2026-08-16 (audit inputs dated 2026-08-15) · **GPU used: 0** · **GPU authorised: 0**
(all 40 cards on all 5 nodes occupied: LOCAL keep10, .21 keep14-distill, .73 keep12, .82 keep8,
.104 paperC-qwen) · **Inputs**: 7 proposals × (1 verification pass + 3 independent refuters with
distinct lenses: preemption / confound / decisive), plus 5 on-disk feasibility probes.
**Majority-kill rule**: `refuted_by >= 2 of 3`. **This document is the decision.**

Provenance convention used throughout: **RETRIEVED** = I fetched or opened it during this pass.
**INFERRED** = derived by me from something RETRIEVED. **CARRIED** = taken from the audit or an
existing on-disk document without independent re-verification (always labelled). **UNVERIFIED** =
no authority confirmed it; the authority tried is named.

---

## 1. Bottom line

**Zero of seven survive.** All seven were refuted 3/3 — not 2/3 — by panels using three
independent lenses, which means no proposal died of a single reviewer's taste. Five of the seven
died on a defect that costs nothing to check and that the author could have caught before writing:
**the pre-registered kill gate cannot fire** (P1, P2, P4, P6, P7), and in the two remaining cases
the gate fires *against* the proposal on day one (P3) or is closed at 0 GPU by tables in the
proposal's own target paper (P5). **The correct action tonight is not to dispatch a proposal.** The
single highest-value 0-GPU action is a ~60-minute consolidation: re-register P3's gate onto the one
non-preempted instance the audit actually found — **LR-DLLM ran DreamOn at initial mask length 1,
outside the 4–64 range DreamOn itself validated, and then drew a conclusion about the model** — and
record it as a reporting-standards note, *not* as a proposal. I verified that instance verbatim
from cached primary text (§3). It can start tonight at 0 GPU: all six papers P3 needs are already
on disk as full arXiv HTML, and the network is currently live (positive control below). Be warned
that even this residue is weak — LR-DLLM discloses the choice as deliberate ("to isolate its
variable-length adjustment capability") — so it is worth **one paragraph**, and MAIN should not let
it grow into a paper.

**Network state, established before any absence claim in this document**: `curl -s -m 45 -x
http://hy-proxy.woa.com:3128 "https://export.arxiv.org/api/query?id_list=2502.09992"` →
`http=200 size=3102`, `<title>Large Language Diffusion Models</title>`. RETRIEVED by me this pass.
Earlier agents recorded the same control degrading to `rc=28` mid-session; it is working now, so
the "not found" results in §7 are meaningful *as of this pass*.

---

## 2. Verdict table

| id | claim (short) | GPU | preemption | refuted_by/3 | gate fires? | verdict |
|---|---|---|---|---|---|---|
| **P1** | Adaptive-length dLLM papers don't report total forward passes (NFE) for their own loop; canvas quality is *bought* not *unlocked* | 0 | PARTLY_PREEMPTED | **3/3** | **NO** — both clauses arithmetically pre-determined non-firing | **DEAD** |
| **P2** | Adaptive length × confidence-parallel decoding: substitutes or complements? | 0 (step 1 needs a GPU trace) | PARTLY_PREEMPTED | **3/3** | **NO** — unguarded ratio, no stated test, conjunction excludes its own predicted split | **DEAD** |
| **P3** | The variable-length lane under-tunes its fixed-length baselines; headline margins shrink under re-basing | 0 | NOT_PREEMPTED | **3/3** | **YES — against itself, on day one** | **DEAD as written; one residue survives (§3)** |
| **P4** | Bolt a training-free contractor onto DreamOn; DreamOn cannot shrink an over-provisioned canvas | "0 GPU first step" — false, proposal itself concedes | PARTLY_PREEMPTED | **3/3** | **NO** — fires trivially; direction is inverted vs the target paper's own Table 2 | **DEAD** |
| **P5** | Decompose iLLaDA's gain over LLaDA into weights vs evaluation protocol | 8–12 GPU-h | PARTLY_PREEMPTED | **3/3** | YES, but closed at **0 GPU** by LLaDA's own Tables 10 + 6 | **DEAD** |
| **P6** | The continuous/flow lane (ELF) has no length-adaptation mechanism; probe terminal velocity | 8–16 GPU-h | UNRESOLVED | **3/3** | **NO** — the only novel clause references an EOS embedding ELF never mentions, against a gold length that is a constant 64 | **DEAD** |
| **P7** | Canvas sensitivity is concentrated in the gold-starved stratum; it is benchmark composition, not a model property | 0 | PARTLY_PREEMPTED | **3/3** | **NO — UNDEFINED, not underpowered** (comparison stratum n=0) | **DEAD** |

**Survivors: 0 / 7.**

---

## 3. Per surviving proposal

**None survives.** Section 3 is therefore not "the surviving proposal" but **the single residue
that is both non-preempted and dispatchable at 0 GPU**, stated at the size the evidence supports.
It is a re-scope of P3, and it must not be recorded as P3 passing.

### The residue: a third-party-configuration audit (n ≥ 1 existence criterion)

**The strongest surviving objection against it** — it is opposed, not unopposed. From the
`confound` lens, refuting P3: *"The audit's headline quantity — 'share of the gap that is canvas' —
is not a measurable property of the published papers. It is a free parameter set by the auditor's
choice of counterfactual baseline."* And independently, the tcodex review at
`/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/proposal/TCODEX_REVIEW2_20260815.md:45`
holds that **LR-DLLM's `initial_masks=1` is an explicit, disclosed stress test, not an undertuned
default**. I checked this objection against primary text and **it is substantially correct**: the
caption says the choice was made *"to isolate its variable-length adjustment capability"*. So the
residue cannot be framed as "we caught someone under-tuning a baseline". It can only be framed as
**"a third-party comparator was run outside the configuration range its own paper validated, and a
model-level conclusion was drawn from it"** — a reporting-standards observation.

**Exact first step** (0 GPU, ~60 CPU/reading minutes, LOCAL, no model load, no network required):
for each of the six papers, tabulate three fields — (a) the configuration range the paper validates
for its *own* method, (b) the configuration at which any *third-party* comparator was run, (c)
whether (b) lies inside (a) — then report the existence count. Do **not** compute "share of the gap
that is canvas"; that quantity is auditor-defined (objection above).

**Input files, with counts I verified this pass** (all under
`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/logs/_netprobe/`, all RETRIEVED by
me via `stat`):

| file | bytes | paper | usable? |
|---|---|---|---|
| `full_2602.07546.html` | 567,048 | LR-DLLM | yes — anchor case, verified verbatim below |
| `full_2508.00819.html` | 247,058 | DAEDAL **v2** | yes |
| `html_2508.00819v1.html` | 245,977 | DAEDAL **v1** | yes — **different md5 from v2; every DAEDAL number needs a version tag** |
| `full_2602.00476.html` | 373,310 | CAL | yes |
| `full_2601.22527.html` | 438,961 | ρ-EOS | yes |
| `p_2509.01025_v2.html` | 761,021 | FlexMDM | yes |
| `p_2602.01326_v1.html` | 258,357 | DreamOn | yes — **use v1** |
| `p_2602.01326_v2.html` | **7,715** | DreamOn v2 | **NO — arXiv shell, not full text** |

Correction to P3's own §5, which MAIN should fix in the frontier doc so the next agent does not
re-download: the claim *"Three of the six PDFs are already extracted to text on this machine"* is
wrong in **both** directions. Zero of the six are in `dllm_draft/sources/` or `dllm_draft/harvest/txt/`
(those hold unrelated ACL-2026 dLLM papers), but **all six are on disk as full arXiv HTML** in
`logs/_netprobe/`, which is better than claimed.

**The anchor instance, RETRIEVED by me from `full_2602.07546.html` (not carried from the audit)** —
LR-DLLM Table 3 caption, verbatim: *"For DreamOn, we set the initial mask length to 1 1 to isolate
its variable-length adjustment capability."* Its DreamCoder-7B rows: `Baseline 14.3 55.5 43.2 39.1`
vs `+DreamOn 21.6 73.5 36.1 37.9` (Random Span / Single-line / Multi-line / Mean) — i.e. a
**negative** headline margin, 37.9 vs 39.1. Against this, DreamOn's own Table 2 single-line row,
RETRIEVED by me from `p_2602.01326_v1.html`, verbatim: `+ DreamOn 88.7 90.6 91.0 91.6 92.1 90.8
91.6` at `Initial Mask Length` `4 8 16 32 64`, `Avg.`, `Oracle`. So DreamOn validates 4–64, scores
88.7 at its **worst** published canvas, and LR-DLLM ran it at 1 — outside that range — for 73.5.

**The pre-registered kill gate, to be committed BEFORE any measurement.** P3's original gate must
be discarded (it fires against P3 on day one — §5). Replacement, to be frozen verbatim:

> **KILL if**, across the six papers audited, **zero** third-party baselines or comparators were
> run at a configuration outside the range the comparator's own paper validates. Existence
> criterion, n ≥ 1; no margin-shrink quantity is computed, and no claim is made about the share of
> any gap. If n ≥ 1, the deliverable is capped at one paragraph plus one table, inside an existing
> document — it does not become a proposal.
>
> *Expected outcome*: **does not fire** — LR-DLLM/DreamOn at mask length 1 vs a validated 4–64 is
> already verified (above), so n ≥ 1. The gate is therefore weak by construction, and that is
> disclosed: its function is to cap the claim, not to discover it.

---

## 4. Per dead proposal — what killed it

### P1 — DEAD. Gate cannot fire; headline premise false; framing banned by an on-disk verdict.

Three independent kills.

1. **Both gate clauses were known-false before running** (§5).
2. **The novelty premise is false.** The non-concurrent killer is **ρ-EOS**, which publishes
   `T_runtime` — *"the runtime spent on the evaluation (in seconds)"* — an absolute total for its
   own adaptive loop **and** for DAEDAL's, across six fixed rungs. `arXiv:2601.22527`, published
   **2026-01-30** (RETRIEVED by me: `logs/_netprobe/abs_2601.22527.xml`, title *"ρ-EOS:
   Training-free Bidirectional Variable-Length Control for Masked Diffusion LLMs"*). **Venue: CoRR
   2026 preprint** — venueid `dblp.org/journals/CORR/2026`, forum `vBwjVjPF5B`, authority
   `api2.openreview.net/notes/search`, RETRIEVED by me from two independent cached responses
   (`ov_rhoeos.json`, `L_or_rhoeos.json`). At ~6.5 months it is **not concurrent**, so the
   concurrency defence does not rescue P1.
3. **P1's framing is explicitly forbidden by a verdict on disk dated 2026-08-16**:
   `dllm_draft/proposal/NFE_SEAM_VERDICT_20260816.md` declares "SEAM DEAD" and bans resurrection
   *"including in disguised form ('cost of canvas', 'quality is bought with canvas')"*. P1 claim 1
   reads *"large-canvas quality is bought, not unlocked."*

> **⚠️ CORRECTION MAIN MUST APPLY — do not lean on VoidPadding.** `NFE_SEAM_VERDICT_20260816.md`
> and the P1 feasibility probe both treat **VoidPadding** (`arXiv:2606.17999`) as the killer of the
> NFE clause. Its *content* is real (I confirmed the cached text: `/tmp/pre/t_2606.17999.txt:1119`
> *"NFE counts only decoding forward passes, while wall-clock time includes both canvas selection
> and decoding"*, with mean NFE 254.66 / 228.82 / 191.52 / 139.70 / 73.06 / 172.10 at lines
> 1229–1247). But the audit left its venue **UNVERIFIED**, and I resolved it this pass — and the
> answer weakens it as a killer on two counts:
> - **Date**: arXiv published **2026-06-16**, updated **2026-06-22**, comment *"Minor related-work
>   revisions; results unchanged"*. RETRIEVED by me (`export.arxiv.org/api/query?id_list=2606.17999`,
>   `http=200 size=2732`). That is **~2 months old = concurrent** under this repo's own standing
>   rule (2–3 months does not constitute preemption).
> - **Venue**: **not a main conference.** `api2.openreview.net/notes/search` (`http=200
>   size=65438`, RETRIEVED by me) returns three records for it: `CoRR 2026` /
>   `OpenReview.net/Public_Article`; **`COLM 2026 ER Workshop`** / venueid
>   `colmweb.org/COLM/2026/Workshop/Efficient_Reasoning`; and **`ACL ARR 2026 August Submission`** /
>   venueid `aclweb.org/ACL/ARR/2026/August/Submission` — i.e. a workshop poster plus an
>   under-review ARR submission.
>
> **Net effect: P1 still dies, but the citation must be ρ-EOS (2026-01, non-concurrent), not
> VoidPadding.** Any future document that kills a direction using VoidPadding as a
> preemption authority is making a concurrency error and a venue-tier error at once. Also:
> `/tmp/pre/t_2606.17999.txt` and `/tmp/2508.00819v2.txt` are the **only** copies of those two
> primary texts, and `/tmp` is wiped on reboot — copy to wzc1 before relying on them again.

Two further mechanical facts, useful because they will recur: **per-item NFE does not exist in any
file P1 names.** `evidence/cells/*.json` carry per-item *pass* but only aggregate
`cost_and_behaviour.{nfe_mean,nfe_median,nfe_total}`; per-item `process.nfe` lives only at
`/apdcephfs_zwfy6/.../dllm_draft/runs/a05_k1/<cell>/metrics.rank*.jsonl`, and **zwfy6 is not mounted
on LOCAL** (I re-confirmed: only `/apdcephfs_wzc1` and `/apdcephfs_wzc1_304376610` are present). So
P1's advertised "paired frontier" silently degrades to a 3-point marginal curve — a gap already
recorded at `TCODEX_REVIEW2_20260815.md:37` before this audit ran.

### P2 — DEAD. The "0 GPU first step" needs a GPU product, and the delta is false as written.

P2's step 1 asserts on *"a recorded logits trace"*. **No logits/confidence trace exists on either
disk** — every artifact records only post-hoc scalars (`process.nfe`, `generated_tokens`,
`raw_output`). Producing one requires a Dream-Coder-7B forward pass, so step 1 is not 0 GPU.
Additionally there is **no DAEDAL Alg. 1 implementation on disk to unit-test** (0 hits for `daedal`,
`conf_eos`, `tau_eos`, `E_factor` across all `.py`/`.sh` on both disks), and both the model
(LLaDA-8B-Instruct) and dataset (GSM8K) its GPU step names are **absent from both disks**.

Preemption: P2's stated delta — *"Fast-dLLM (ICLR'26) is fixed-length. Delta = the cross."* — is
false. **DPad** already ran the length × parallel factorial, explicitly tested
substitutes-vs-complements, and published **COMPLEMENTS**, on P2's own models. `arXiv:2508.14148`,
**ICLR 2026 Poster**, venueid `ICLR.cc/2026/Conference`, forum `0yOsSMU1eY` — authority
`api2.openreview.net/notes/search`, **RETRIEVED and re-verified by me this pass** (title *"DPad:
Efficient Diffusion Language Models with Suffix Dropout"*). Verbatim from DPad: *"suffix dropout and
parallel decoding address orthogonal bottlenecks and, when combined, yield near two orders of
magnitude improvement."* Separately, DAEDAL Fig. 5 already swept `tau_high × tau_low` on GSM8K and
reported flatness — so P2's quality-only gate is close to pre-answered NO.

### P3 — DEAD as written. Its own gate kills it on day one, in the direction it did not expect.

Clause 2 reads *"KILL if ... fewer than 2 of 6 papers report only a single baseline length."*
Measured: DAEDAL sweeps 6 lengths, ρ-EOS 6, DreamOn 5, CAL 4; FlexMDM cannot have one by design;
LR-DLLM sweeps 5 in its **Table 8** and re-bases against *"the best fixed-length choice selected in
hindsight"*. Count of single-baseline-length papers = **0–1 < 2 ⇒ KILL**. The author asserted the
opposite because they read LR-DLLM's Table 3 in isolation and never checked Table 8 of the same
paper. **This is the audit's headline finding about the repo's own process**: the gate was falsified
by a table the author had already downloaded.

### P4 — DEAD. The damage the gate needs does not exist; the target paper already ships the fix.

The gate needs DreamOn to *degrade* under an over-provisioned canvas. **It improves.** DreamOn's own
Table 2 single-line row is monotone increasing 88.7 → 92.1 over canvas 4 → 64, and its c=64 cell
(92.1) **exceeds its own Oracle** (91.6) — RETRIEVED by me above. So the measured "drop" is expected
to be negative and the gate fires trivially. Worse, DreamOn §3.4 **already ships a training-free
global contraction rule** ("Broadcasting Deletion as Length Predictor"), ablated in §5.3 at 0.6 pp
headroom — that is precisely the component P4 proposes to add. And **LR-DLLM already published the
method claim**: *"51.3% Pass@1 on HumanEval-Infilling under fully unknown lengths (+13.4% vs.
DreamOn)"*.

DreamOn's venue: **ICLR 2026 Poster**, venueid `ICLR.cc/2026/Conference`, forum `EQTPmqukiU` —
authority `api2.openreview.net/notes/search`, **CARRIED** from cached responses
(`logs/_netprobe/or_dreamon.json` and two siblings); I did not re-query it this pass.

**P4 also contains a fabricated citation.** P4 §4 asserts *"CAL/LR-DLLM both report DreamOn behaving
inconsistently across span regimes."* In CAL's full text the string `DreamOn` appears **exactly
once, in the bibliography**. CAL reports no DreamOn measurement at all. Flagging this explicitly
because it is the second fabricated-fact instance in this batch (see P6).

Two cost facts worth keeping: P4's harness (`generate_infilling.py`) has **no oracle-relative canvas
argument** — the oracle span is hardcoded `min(gold_span, max_new_tokens)` — so "oracle+128" is not
expressible without a patch; and P4's cost estimate is wrong by ~8× **in its own favour** (summing
on-disk per-item `elapsed_seconds` gives 0.6–0.7 GPU-h for two 1033-item arms, not the claimed 4–6).

### P5 — DEAD, and closed at 0 GPU rather than by argument.

The probed claim is verbatim-correct: iLLaDA Table 4 (Likelihood 77.2/60.2/74.3 vs Confidence
78.5/60.8/76.6) gives **+0.6 pp on ARC-C = 4.0% of the 14.9 pp headline delta**, far under P5's own
≥20% bar. Then the decisive move: **LLaDA's own tables close both remaining legs on the right
model.** Table 10 is a generation-length ablation (HumanEval 32.9/32.9/35.4 at L=256/512/1024,
caption *"The results are not sensitive to the length hyperparameter"*) — span **2.5 pp < the 3.3 pp
bar**, and the quoted 35.4 is the **maximum** of LLaDA's own sweep. Table 6 is a CFG ablation
(ARC-C 45.9 → 47.9, **+2.0 pp < the 3.0 pp bar**). So the revised gate dies at 0 GPU too. **8–12
GPU-h saved.** Note the falsified premise: P5 assumed quoted baselines *understate*; LLaDA reported
the best cell of its own sweep, so the bias runs the other way.

### P6 — DEAD. The only novel gate clause references a mechanism the target paper never mentions.

P6's descriptive premise is true and better-supported than claimed (ELF hard-codes `Sequence length
1024`, 0 hits for "padding"). But the write-up contains **two fabricated facts and one grep
artifact**: *"NFE appears 46 times"* (word-boundary count is **0**; the 45 substring hits are
"inference"); *"1024 steps vs 32 steps"* (0 hits; max steps in the paper is **64**); and *"EOS → 2
hits"*, where both hits are the letters `eos` inside **"videos"**. The last is fatal: clause (c) —
the only clause carrying P6's novelty — is defined against *"cosine distance ... to the decoder's
EOS embedding"*, and **ELF has zero word-boundary occurrences of EOS**. The clause has no referent.
Compounding it, clause (c) needs per-item gold length on ≥500 items, but ELF's only per-item
benchmarks (WMT14 De-En, XSum) fix **target length = 64 by construction** — a correlation against a
constant is undefined.

Note the `preemption` lens dissents on the premise (arguing ELF ships `mask_after_eos` +
`pad_token: eos`), which is **UNRESOLVED** between refuters. It does not change the disposition:
the gate is dead either way, and even a clean pass lands on a 105M OWT/perplexity model that P6's
own must-not-claim forbids comparing to this programme's 7–8B pass@1 results.

### P7 — DEAD. Gate UNDEFINED (not underpowered), and the group's own audit already refutes the headline.

The gate needs a `canvas >= 2*gold_len` stratum at HE+ canvas=8, i.e. items with gold ≤ 4 tokens.
**I recomputed this independently** from `logs/_netprobe/p7_goldlen_probe.json` joined to
`cells_corrected/a05_closeout_stitch_regrade.json` (both wzc1-resident; DreamOn tokenizer, no model
load):

```
HE+ n=164  gold tokens: min 43  median 174  max 558
items with gold <= 4 :  HE+ 0   MBPP+ 0
HE+ c8   corrected: starved n=164 pass=22 | middle n=0 | roomy(>=2x) n=0
HE+ c32  corrected: starved n=164 pass=42 | middle n=0 | roomy(>=2x) n=0
HE+ c128 corrected: starved n=127 pass=55 | middle n=32 pass=19 | roomy n=5 pass=5
```

The comparison arm is **n=0**, structurally, at the one canvas the gate names. A two-proportion
exact test on n=0 is not "failing to reach α=0.05" — it is **undefined**. P7 self-diagnosed a
*power* problem and predicted the 5 short-gold items would form the comparison group; those items
have gold lengths [43, 51, 55, 57, 57], each 5–7× larger than canvas 8. Dispatched as written, the
agent would have reported "gate fired on power" for a reason that factually did not happen.

Independently: my joins reproduce A05's own published aggregates exactly (HE+ gold ≥ 65 → **159/164**;
MBPP+ → **51/378**), which validates the derivation path.

**The headline is separately refuted by the group's own work.**
`/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/SPANLEN_STRATIFIED_AUDIT.md:172-179`
already ran this analysis *with* the AR difficulty control and found DreamOn's length sensitivity
**+0.7729 [+0.6990, +0.8462]** vs AR FIM control **+0.2103 [+0.1140, +0.3058]** — non-overlapping
CIs, *"~3.7x the AR FIM control's"* (RETRIEVED by me from that file). That is a **model** property,
which is the opposite of P7's "benchmark composition" claim.

---

## 5. Kill gates that CANNOT FIRE

This is the batch's dominant failure mode — **5 of 7** — and it is a recurring repo defect, so each
entry names the fix.

| id | why it cannot fire | fix |
|---|---|---|
| **P1 c1** | *"KILL if Spearman(nfe_mean, pass@1) over HE+ {8,32,128} is not +1.0"* — with **k=3** points both strictly monotone in the same order, ρ = +1.0 **by arithmetic identity**. Unfalsifiable. | Never pre-register a rank correlation on k=3. Use the per-item paired test (McNemar), which needs no NFE and is decisive. |
| **P1 c2** | *"KILL if incremental pp-per-NFE changes by more than 3x"* — measured **2.0498×**, and the proposal **states** the threshold *"is chosen so the measured 2.05x does not fire it; that is deliberate and disclosed."* | A threshold chosen from the measurement is not pre-registration. Set it from a power/effect-size argument before looking. |
| **P1, decisiveness** | Neither clause can adjudicate the headline claim; the proposal's own must-not-claim concedes the claim needs running other people's methods. | Re-register on the novelty claim: *"KILL if any published variable-length dLLM paper reports a total forward-pass or wall-clock count for its own adaptive loop."* Per this audit that **fires immediately** (ρ-EOS `T_runtime`). |
| **P2** | Ratio of two quality gains with **no floor guard on the denominator** (the sequential gain is 0.6 pp on HumanEval = **one item of 164**), so ">=80% of its gain" is *undefined*, not small; **no test is stated** for α=0.05 (point estimate? CI bound? TOST? — three readings, three verdicts); and the conjunction *"on GSM8K **and** HumanEval"* cannot resolve P2's **own** modal prediction of a split, which neither KILLs nor passes. | Add a pre-registered absolute minimum gain whose paired-bootstrap CI excludes 0, else INDETERMINATE; name the test; add the missing third branch; move the gate onto the **expansion/insertion counts**, which P2 itself calls the identifying measurement and which carry no gate at all. |
| **P3** | Fires **against** the proposal on day one: clause 2 requires ≥2 of 6 papers to report a single baseline length; measured 0–1. Clause 1 is also undefined on P3's own anchor — LR-DLLM's DreamOn margin is **negative** (37.9 vs 39.1), and a negative margin has no ">50% shrink". | Drop clause 2. Restrict shrink to **positive** margins. Replace with the existence criterion in §3. |
| **P4** | Needs pass@1 to **drop** ≥2.0 pp at oracle+128; the target paper's Table 2 is **monotone increasing** and its c=64 beats its own Oracle, so the gate fires trivially and for a published reason. Also oracle+128 sits **at/beyond L_max=128**, confounding with the expansion cap; and P4's own §6(d) concedes 2 pp *"sits inside the region where ceiling artefacts live"*. | Move to MultiLine/RandomSpan where DreamOn is genuinely non-monotone; stay strictly inside L_max; pre-register against the published c=4..64 curve; beat LR-DLLM's +13.4% rather than re-deriving "length control is imperfect". |
| **P6** | Clause (c)'s statistic is undefined — it references *"the decoder's EOS embedding"* and ELF has **0** word-boundary occurrences of EOS; and its gold-length axis is a **constant 64** in ELF's only per-item benchmarks (zero variance). Clause (a) already resolved FALSE, so there is no live 0-GPU branch. Note the proposal **peeked at the paper, then revised clause (b) and left (c) untouched**. | Re-state the statistic purely in ELF's own terms (per-slot `||x_theta(z_t,t) - z_t||` at the first flow step, frequency-normalised against a unigram control), on a task with variable per-item gold, and declare up front that the result is ELF-scoped. |
| **P7** | Comparison stratum **n=0** at the named canvas — UNDEFINED, not underpowered (§4, recomputed by me). | Amend **pre-data** to a populated cell. Verified populated: **HE+ c8 by A05's own ≥65 split** (5/5 = 1.000 vs 17/159 = 0.107, diff **89.3 pp**, Fisher exact **p = 2.833e-05** — I reproduce the audit's p to 3 s.f.), or **MBPP+ c32** (roomy 37/41 = .9024 vs starved 13/157 = .0828, diff **82.0 pp**). Freeze the choice **before** looking, or it is post-hoc benchmark-switching, which P7's own gate text forbids. |

> **⚠️ Second correction: one audit p-value does not reproduce.** For MBPP+ c32 roomy-vs-starved
> the audit reports `p=1.26e-13`; my exact two-sided Fisher on the same 2×2 `[[37,4],[13,144]]`
> gives **1.36e-24**. The 2×2 **counts** reproduce exactly, so the effect is not in doubt, but the
> p-value is. `scipy` is **not installed on any interpreter I tried** on LOCAL
> (`/opt/conda/envs/torch-base/bin/python`, `./.venv/bin/python`, `/usr/bin/python3.11` — all
> `ModuleNotFoundError: No module named 'scipy'`), so I could not adjudicate between my
> hand-rolled implementation and the audit's. **Do not quote either p-value until one is
> recomputed with a validated implementation.** Quote the counts.

---

## 6. What MAIN must NOT claim — the canvas / S4 finding

**Status: PREEMPTED as a general claim.** `S4_DISPOSITION.md` (2026-08-15) records G0 as **FIRED**
and S4's terminal state as **`caveat_citation_only` — "0 GPU authorised, now and in future."**

**The preempting papers, with verified venue + authority:**

- **DAEDAL**, `arXiv:2508.00819`, 2025-08-01 (v2 2025-08-18) — **ICLR 2026 Poster**, venue string
  `ICLR 2026 Poster`, venueid **`ICLR.cc/2026/Conference`**, forum `Ic2A2gCseC`. Authority:
  `api2.openreview.net/notes/search`. **RETRIEVED by me this pass** from four independent cached
  responses (`or_daedal.json`, `OR_daedal.json`, `or_daedal_v.json`, `daedal.json` — all four agree
  exactly). Shows the phenomenon on **full-program** MBPP and HumanEval with generation length swept
  and all else frozen: MBPP +16.6 pp, HumanEval +28.7 pp.
- **ρ-EOS**, `arXiv:2601.22527`, published 2026-01-30 — **CoRR 2026 preprint**, venueid
  `dblp.org/journals/CORR/2026`, forum `vBwjVjPF5B`. Authority: `api2.openreview.net/notes/search`
  (cached `ov_rhoeos.json`, `L_or_rhoeos.json`, agreeing), arXiv metadata `abs_2601.22527.xml`. Both
  **RETRIEVED by me this pass**. Replicates on two more LLaDA checkpoints.

Both predate S4's 2026-08-12 statement by **≥6.5 months**, so the 2–3-month concurrency clause does
not apply.

**Our +26.5 pp is a REPRODUCTION.** I re-derived it: `mbpp_c8.pass_at_1_plus = 0.08994708…` →
`mbpp_c32.pass_at_1_plus = 0.35449735…` = **+26.45 pp** (RETRIEVED from
`proposal/archive/A05-.../evidence/cells/`). It reproduces a published phenomenon on a third model.

### The line, stated so it cannot be blurred

| ❌ BARRED (preempted) | ✅ PERMITTED (diagnostic use) |
|---|---|
| "We discovered that dLLM code scores are a canvas-budget artefact." | "We independently reproduce, on DreamOn-v0-7B and the EvalPlus base/plus axes, the known canvas-budget sensitivity of mask-diffusion LMs on full-program code (DAEDAL, ρ-EOS)." |
| "A dLLM's reported code weakness is substantially an artefact." | "Our repo's published DreamOn operating point (`initial_masks=8`) sat in the crippled regime." |
| Any framing of canvas sensitivity as a **finding**, incl. disguised forms — *"cost of canvas"*, *"quality is bought with canvas"* (banned verbatim by `NFE_SEAM_VERDICT_20260816.md`). | Auditing a **named third party's** comparator configuration against the range that party's own paper validated — **and only with a live instance on the table** (§3 supplies exactly one: LR-DLLM/DreamOn at mask 1 vs a validated 4–64). |

**The diagnostic exception is narrow and conditional.** It is not a licence to re-run the canvas
sweep. It requires (i) a *specific* published comparator configuration, (ii) the comparator's own
validated range quoted from the comparator's own paper, and (iii) an acknowledgement that the choice
may have been a disclosed stress test rather than an error — which is exactly what LR-DLLM says it
was. Without a live instance the diagnostic framing collapses back into the barred column.

**Two standing guards survive S4's death and still bind.** (a) The stitch defect is an
**interaction** with canvas, not a second standalone 31 pp defect (17 arms re-graded, 2 move,
+0.61 pp — so no diffusion-vs-AR asymmetry may be claimed). (b) **Every "DreamOn reaches X" figure
in this repo is a LOWER BOUND**, because `he_c512`, `mbpp_c128` and `mbpp_c512` were never run.
Relatedly, **`mbpp_c128` does not exist as a scored cell** — the run was killed at 30/378 items and
never graded — so the "canvas 8/32/128 × both benchmarks" grid asserted in several places is
**HE+ only**; MBPP+ has 8/32. There are **5** cells on disk, not 6.

---

## 7. Unresolved, and which authority refused

| item | status | authority tried |
|---|---|---|
| **LR-DLLM** `arXiv:2602.07546` venue | **UNVERIFIED — NOT-FOUND, not "is a preprint"** | `api2.openreview.net/notes/search` (`http=200`, 58,984 B, RETRIEVED by me) returned **no matching record** — top hit an unrelated qMRI paper. arXiv comment field reads only *"diffusion language models"*. Its HTML contains a style-file line *"Keywords: Machine Learning, ICML"* which is **not** an acceptance record. **Semantic Scholar not attempted (HTTP 429 all day per prior passes).** LR-DLLM is load-bearing for the §3 residue, so this gap matters. |
| **CAL** `arXiv:2602.00476` venue | **UNVERIFIED (identity unresolved)** | `api2.openreview.net/notes/search` (`http=200`, 52,991 B, RETRIEVED by me) surfaced a same-topic **`ICLR 2026 DeLTa Workshop Poster`**, venueid `ICLR.cc/2026/Workshop/DeLTa`, titled *"Training-Free Length Discovery for Diffusion Language Model Infilling"* — plausibly the same work, **but I could not confirm it is**. Do not attach that venueid to CAL. |
| **VoidPadding** `arXiv:2606.17999` main-conference venue | **Resolved to workshop + ARR submission only** (§4) | `api2.openreview.net/notes/search` `http=200`, RETRIEVED by me: `COLM 2026 ER Workshop` + `ACL ARR 2026 August Submission` + `CoRR 2026`. No main-conference record. |
| **P6 / ELF premise** (does ELF ship `mask_after_eos` + `pad_token: eos`?) | **UNRESOLVED — refuters disagree** | `preemption` lens asserts yes (repo-level); `confound`/`decisive` lenses count 0 word-boundary EOS in the paper. Does not change P6's disposition. |
| **Dystruct** `arXiv:2605.09820` | **UNVERIFIED, unread — flagged as P6's highest-priority unchecked lead** | arXiv metadata only, from a cached search file. `export.arxiv.org` `search_query` endpoint returned **0 bytes on ~150 attempts** while `id_list` simultaneously returned 3,102 B. |
| **DreamOn** venue (ICLR 2026 Poster, forum `EQTPmqukiU`) | **CARRIED, not re-verified this pass** | Cached OpenReview responses agree; `S4_DISPOSITION.md` records an **OpenReview 403** on an earlier attempt. |
| **Semantic Scholar** | **REFUSED throughout — HTTP 429** | Not relied upon for any field in this document. |
| **`/apdcephfs_zwfy6`** | **Not mountable from LOCAL** — re-confirmed by me (`df` shows only `dop-fuse` on `/apdcephfs_wzc1/share_304376610`) | Any per-item NFE / raw-metrics work must run on `.73`/`.82`/`.104`, or the 896 KB must be `scp -O`'d. |
| **`scipy`** | **Absent from every LOCAL interpreter tried** | Blocks adjudication of the MBPP+ Fisher p-value (§5). |
