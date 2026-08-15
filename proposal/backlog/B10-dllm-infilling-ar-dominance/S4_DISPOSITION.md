# S4 — explicit disposition (2026-08-15)

**Status: G0 DISCHARGED. S4 is PREEMPTED as a general claim → caveat-citation only.**

---

## 0. Why this file exists

S4 is a sub-claim that was **orphaned twice**, and its terminal document never mentioned it.

| # | Fact | Verified how (2026-08-15) |
|---|---|---|
| 1 | **A05 died and handed S4 to B10.** `proposal/archive/A05-structural-dllm-cost-frontier/STATUS.json:finding_that_outlives_a05.owner_RESOLVED_20260812.owner` names `B10-dllm-infilling-ar-dominance, as pre-registered sub-claim S4`, and the same key's `novelty_still_owed` reads verbatim **`"YES -- S4-G0 is a hard blocker and was NOT done (Semantic Scholar HTTP 429 on every attempt)."`** | read the A05 archive STATUS.json |
| 2 | **B10 accepted ownership, and S4's own G0 is an un-discharged hard blocker.** `STATUS.json:subclaim_S4_inherited_from_A05.kill_gate.S4-G0_novelty` begins verbatim **`"HARD BLOCKER, NOT DONE (Semantic Scholar HTTP 429 on every attempt 2026-08-12)."`** | read B10's own STATUS.json |
| 3 | **B10's terminal deliverable is silent about S4.** `PROTOCOL_NOTE.md` (353 lines, md5 `91bdec5a90ad835d033d1698012d5007` — both re-verified here) mentions S4 **0 times** and `canvas` **0 times**: `grep -ciE "S4\|canvas" PROTOCOL_NOTE.md` → `0`. `grep -ci initial_mask` → `0`. (`DreamOn` does appear, 8×, but only as the model whose infilling arms Gate 1 scored.) | `md5sum`, `wc -l`, three greps |

Meanwhile `STATUS.json:protocol_note_20260815.kill_gate_fired_plus_residue_sealed`
asserts **"Nothing further is owed"**. That assertion is **correct about B10's own gate_1
and wrong about S4**: B10's kill gate fired legitimately (independently reproduced —
see §5), but the sub-claim B10 had *explicitly adopted* still carried a hard blocker
that was never discharged. A sub-claim with an open pre-registered blocker cannot be
retired by its owner's death going unmentioned.

This file discharges S4-G0 and states S4's terminal state. It adds **no new measurement**
and **spends no GPU** (0 GPU authorised, 0 GPU used; search was CPU + network only).

---

## 1. S4, quoted verbatim from `STATUS.json:subclaim_S4_inherited_from_A05`

> `"statement"`: "On full-program code generation, DreamOn-v0-7B's reported weakness is
> substantially an artefact of one sampler-config integer: initial_masks 8->32 (all else
> frozen) moves MBPP+ .0899->.3545 (+26.5 pp) and HE+ .1280->.2561. The MBPP+ leg runs on
> a code path with NO stitch, so it is uncontaminated by the HE+ post-processing defect."

### Its two guards, reproduced verbatim (both remain binding)

> `"must_not_claim_1"`: "NOT 'two independent artefacts each larger than typical method
> deltas'. The stitch defect's severity is CREATED by fixing the canvas: +0.61 pp at
> canvas=8, +4.27 pp at 32, +31.10 pp at 128. At the published operating point
> (initial_masks=8) only 1 of 164 raw outputs is multi-line, so the double-indent corrupts
> almost nothing and costs exactly 1 item (HumanEval/13). It is an INTERACTION, not a
> second standalone 31 pp defect."

> `"must_not_claim_2"`: "NOT 'published numbers across the repo are affected, including the
> AR control'. Measured NIL blast radius: 17 arms re-graded, 2 move (both already-known
> DreamOn HE+ arms, +0.61 pp). All others have exactly 0 items on the buggy branch. There
> is no shared stitch, hence no diffusion-vs-AR asymmetry to claim."

### The gate being discharged here, verbatim

> `"S4-G0_novelty"`: "HARD BLOCKER, NOT DONE (Semantic Scholar HTTP 429 on every attempt
> 2026-08-12). Must cover baseline-tuning literature, code-eval harness/post-processing
> sensitivity, dLLM length/canvas budget sensitivity. **KILL if an existing paper already
> shows a mask-diffusion LM's full-program code score is dominated by its initial-canvas
> budget -> S4 is a reproduction, citable as caveat only.**"

---

## 2. G0 search record

`S4-G0` was never done because one API rate-limited. **One API's 429 is not evidence of
absence**, so this pass used multiple independent routes and records which ones answered.

### 2.1 Authorities — reached vs refused

| Authority | Result | Detail |
|---|---|---|
| **arXiv API** (`export.arxiv.org/api/query`) | **REACHED** — primary route | HTTPS only; plain `http://` timed out (curl exit 28). ~20 structured queries + `id_list` metadata pulls. |
| **arXiv HTML full text** (`arxiv.org/html/<id>`) | **REACHED** | Used to read actual tables rather than trusting abstracts. 7 papers pulled in full. |
| **DBLP** (`dblp.org/search/publ/api`) | **REACHED (flaky)** | Long multi-word queries return `total 0` or HTTP 500; short 2–3 word queries work. Authoritative for the ACL-family hits below. |
| **ACL Anthology** (`aclanthology.org/<id>/`) | **REACHED for per-paper pages** | Its `/search/` endpoint is a JS Google-CSE shell and returns no results to `curl` — so DBLP was used to *find* candidates and Anthology per-paper pages to *verify* them. |
| **Semantic Scholar** (graph API) | **REFUSED — HTTP 429 on every attempt** | `/paper/search` and `/paper/arXiv:<id>` both. Backoff ladder 20 s / 45 s / 90 s / +30 s all 429; a 1-token control query (`query=test&limit=1`) also 429. **This is the same blocker that stalled G0 on 2026-08-12, reproduced.** No S2 field is used below. |
| **OpenReview** (`api.openreview.net`, `api2.openreview.net`) | **REFUSED — HTTP 403 `ChallengeRequiredError`** | Both API versions; `openreview.net/forum?id=…` returns 307 to the same challenge. **Consequence: no ICLR/NeurIPS/ICML `venueid` could be freshly verified this pass.** No candidate below is claimed as an OpenReview-family acceptance on this pass's own authority. |
| Web search / Google Scholar | not used | Not needed: the decisive hits were found on arXiv and their venues settled on DBLP + Anthology. Recorded so the route list is honest, not padded. |

### 2.2 Coverage against G0's own required scope

G0 demands three areas; the task brief expanded to five. All were queried:

- **(a) diffusion/mask-LM generation-length / canvas budget sensitivity on code** — ~10 queries (`"masked diffusion" + "length"`, `"generation length" + code`, `abs:"initial mask" + diffusion`, `abs:"canvas" + "diffusion language model"`, …). **This is where the decisive hits are.**
- **(b) baseline-tuning / baseline-fairness in code LM eval** — queried (`weak baselines`, `undertuned`, `"baselines" + "tuned" + "code generation"`, `"are we really making much progress"`). The genre exists and is well established **outside** code LMs (recsys 1907.06902, fluids 2407.07218, text classification 2204.03954). Nothing in it is about diffusion canvas.
- **(c) HumanEval+/MBPP+/EvalPlus harness & post-processing sensitivity** — queried. Nearest are already in B10's `RELATED_WORK.md` (SAFIM; `2505.18789`; `2605.07395`). **Not re-litigated here** — this leg was already surveyed and, per `RELATED_WORK.md` MUST-NOT-CLAIM 9, B10 is already barred from claiming FIM post-processing sensitivity as novel. It is also *not* the leg G0 turns on: G0's KILL clause is specifically about **initial-canvas budget dominating a full-program code score**.
- **(d) Dream / DreamOn / LLaDA / SEDD / Diffusion-LM sampler-config ablations** — queried by name and by mechanism. Yielded the decisive cluster.
- **(e) "reported weakness was a configuration artefact" in LM eval generally** — queried (`ti:"illusion"`, `abs:"artifact" + baseline + configuration`, `"reported" + "weakness" + "artifact"`). Found the adjacent `2606.29228`, `2509.01790`, and the recsys/fluids genre. Nothing showing S4's specific mechanism.

### 2.3 Candidates, with the per-candidate judgement

The judgement column answers exactly one question: **does it show that a mask-diffusion LM's
full-program code score is dominated by its initial-canvas / generation-length budget?**

#### PREEMPTING (does show it)

**P1 — DAEDAL. `arXiv:2508.00819` (v2), 2025-08-01.**
*Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models.*
Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin.
Venue: **CoRR / arXiv-only** — DBLP `CoRR 2025`, DOI `10.48550/ARXIV.2508.00819`; arXiv
Comments field carries only a code URL, **no acceptance note**. *(OpenReview unreachable this
pass, so an OpenReview-family acceptance cannot be excluded; treat as preprint and re-check
before citing a venue.)*

> **⚠️ CORRECTION 2026-08-15 — the venue line above is WRONG; the caveat it flagged has now been
> resolved against it. DAEDAL is `ICLR 2026 Poster`.** (Original assertion retained above as the
> record of what this pass concluded; do **not** cite it.)
>
> Authority: OpenReview api2 **`/notes/search`** (note `id = forum = Ic2A2gCseC`) —
> `"venue": "ICLR 2026 Poster"`, `"venueid": "ICLR.cc/2026/Conference"`, `Submission1382`, and
> `ICLR.cc/2026/Conference/Submission1382/-/Camera_Ready_Revision` present in `invitations[]`.
> Author list matches arXiv 6/6, so not a title collision. Decision published (`pdate`)
> **2026-01-26**. Cite as `@inproceedings{li2026beyond, booktitle={The Fourteenth International
> Conference on Learning Representations}, year={2026}, url={https://openreview.net/forum?id=Ic2A2gCseC}}`.
>
> **What this pass got right:** both stated observations were factually correct and re-confirmed —
> DBLP really does say `CoRR 2025`, and arXiv `comment` really is only the code URL with **no**
> acceptance note and **no** `journal_ref` (arXiv v2 is dated 2025-08-18, i.e. it predates the
> 2026-01-26 decision and was never refreshed). **Only OpenReview knows.** Refusing to assert
> absence was correct.
>
> **The one mistake, worth internalising:** "OpenReview unreachable" was inferred from a 403 on
> `api2 /notes?`. That path *is* challenge-gated (`ChallengeRequiredError`, reproduced verbatim
> 2026-08-15, incl. with browser UA), **but `api2 /notes/search?term=…&source=forum&limit=100` is
> NOT gated and returns 200.** Filter client-side on `content.title`. Also: **api v1**
> (`api.openreview.net`) returns only DBLP-mirror records and gives **0 title hits** for DAEDAL — a
> v1-only pass is exactly how one wrongly concludes "arXiv-only".
>
> **Effect on this disposition: the preemption gets STRONGER, not weaker** — it is now a
> peer-reviewed accepted ICLR 2026 result rather than an unreviewed preprint. The Table 1 numbers
> below were re-verified cell-by-cell against `arxiv.org/html/2508.00819v2` and are **correct as
> written (12/12)**; the swept columns sit under the paper's own column-group header
> **`Fixed-Length Denoising (Baseline)`**, confirming this is a sweep of the fixed-length
> **baseline** (DAEDAL itself is a single separate column at `L_init=64`), and the same design is
> repeated on a **second** checkpoint in Table 2 (`LLaDA-1.5-8B`: MBPP `20.6 30.2 39.2 38.6 39.8
> 39.6`, HUMANEVAL `18.3 22.0 37.8 45.1 49.4 50.0`).
>
> P2 (ρ-EOS, `2601.22527`) was also spot-checked and **exists**, with its MBPP baseline sweeps
> `21.0→36.7` (LLaDA-Instruct-8B) and `21.2→39.2` (LLaDA-1.5-8B) verified — but **its own venue
> was not re-run** through `/notes/search` and is therefore `NOT-FOUND`, **not**
> confirmed-preprint. Same applies to every other `CoRR`-from-DBLP venue in this file: DAEDAL
> proves a DBLP `CoRR` record **coexists** with an accepted ICLR-2026 record.
>
> Full evidence + verbatim quotes: `dllm_draft/proposal/VENUE_RESOLUTION_20260815.md`.

**Judgement: YES — this is a direct preemption, and it is ~12.5 months old, so the
concurrency clause does not apply.** Its **Table 1** is a fixed-length sweep of
LLaDA-Instruct-8B on exactly the two benchmarks S4 uses, at exactly the operating points
S4 contrasts, with all else frozen:

| Benchmark | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| MBPP (pass@1) | 20.8 | 28.0 | 37.4 | 38.2 | 37.4 | 38.8 |
| HUMANEVAL (pass@1) | 18.9 | 26.2 | 36.0 | 47.0 | 47.6 | 47.0 |

MBPP 20.8 → 37.4 across a 64→256 canvas change (**+16.6 pp**) and HumanEval 18.9 → 47.6
(**+28.7 pp**) are the same phenomenon, same direction, same order of magnitude as S4's
MBPP+ .0899→.3545 (+26.5 pp) / HE+ .1280→.2561. The paper's own framing states the point
S4 asserts: *"For the baseline models, performance is highly dependent on manually tuning
the generation length for each specific task"*, and the abstract's motivation is that
*"insufficient lengths cripple performance on complex tasks."*

**P2 — ρ-EOS. `arXiv:2601.22527` (v2), 2026-01-30.**
*ρ-EOS: Training-free Bidirectional Variable-Length Control for Masked Diffusion LLMs.*
Venue: **CoRR / arXiv-only** — DBLP `CoRR 2026`, DOI `10.48550/ARXIV.2601.22527`; arXiv
Comments = "11 pages,6 figures,6 tables", no acceptance note. *(Same OpenReview caveat.)*
**Judgement: YES — independent replication of P1's phenomenon.** Its Tables 1–2 repeat the
fixed-length baseline ladder on LLaDA-Instruct-8B and LLaDA-1.5-8B: MBPP **21.0 → 36.7**
(64→256) and **21.2 → 39.2** on LLaDA-1.5. Same claim, two more model instances. Published
**6.5 months before** S4 was written down, so also not concurrent.
*Note: this is the strongest available evidence on the question S4-G1 was going to ask
(does a SECOND mask-diffusion model move ≥10 pp?) — and the answer in the published
literature is already YES for LLaDA-8B-Instruct and LLaDA-1.5-8B. See §4.*

**P3 — dLLM-Var. `arXiv:2510.24605`, 2025-10-28.**
*Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way.*
Venue: **CoRR / arXiv-only** — DBLP `CoRR 2025`, DOI `10.48550/ARXIV.2510.24605`.
**Judgement: PARTIAL-YES.** Its Table 2 is a pure-diffusion generation-length ladder on
MBPP: 64→**38.40**, 128→**42.00**, 256→42.00, 512→43.20, 1024→41.80. The mechanism and the
non-monotone tail are S4's, but the MBPP swing here is only ~+4.8 pp — it corroborates the
*mechanism* without matching S4's *magnitude*. Counted as supporting, not as the decisive hit.

**P4 — Diffusion Language Models: An Experimental Analysis. `arXiv:2606.19475` (v2), 2026-06-17.**
Thomas Bertolani, Davide Bucciarelli, Leonardo Zini, Marcella Cornia, Lorenzo Baraldi.
Venue: **CoRR / arXiv-only** — DBLP `CoRR 2026`; no arXiv Comments field at all.
**Judgement: YES for the general claim, NO for S4's exact operating point.** It states the
generalised version of S4 as a headline finding: *"the behavior of DLMs is strongly
influenced by generation-time design choices"*, and isolates the responsible variable —
*"it is the generation length rather than the step count that drives performance
degradation"* — with MBPP/HumanEval curves for Dream and LLaDA that *"saturate or decline
beyond 256–512 tokens"*. It scales steps and length jointly (1:1) rather than freezing all
else, and never touches EvalPlus (`EvalPlus` count 0, `canvas` 0, `initial mask` 0), so it
does not reproduce S4's frozen-all-else contrast; but combined with P1/P2 it means S4's
*framing* — "a dLLM's reported code weakness is substantially a canvas-budget artefact" —
is published, not new.

#### ADJACENT (does NOT show it)

| Cite | Venue, verified by the right authority | Why it is not a preemption |
|---|---|---|
| **DreamOn**, `arXiv:2602.01326` | DBLP `CoRR 2026`. B10's `RELATED_WORK.md` §3.1 records ICLR 2026 Poster via OpenReview `venueid=ICLR.cc/2026/Conference`, forum `EQTPmqukiU` — **carried forward, NOT re-verified this pass** (OpenReview 403). | **The model S4 is about, and it sweeps initial mask length itself** (`initial mask` ×12, mask lengths 4–64, plus an oracle-length column). **But its sweep is on INFILLING, never full-program**: full text has `MBPP` 0, `MBPP+` 0, `EvalPlus` 0, `HumanEval+` 0. So it establishes canvas-sensitivity on infilling, which is the leg B10 already owned — **not** the full-program leg S4 claims. This is why S4 was not already dead. |
| **CAL**, `arXiv:2602.00476` | arXiv-only (already in `RELATED_WORK.md` §3.6) | +47.7 % pass@1 over fixed-length in **code infilling**, not full-program. Same boundary as DreamOn. |
| **LR-DLLM**, `arXiv:2602.07546` | arXiv-only | HumanEval-**Infilling** / McEval under unknown lengths; explicitly a DreamOn comparison. Infilling axis. |
| **Understanding Evaluation Illusion in dLLMs**, `arXiv:2606.29228` (v2) | DBLP `CoRR 2026` | Nearest in *spirit* ("reported rankings are a config artefact") and does ask *"How does generation length affect evaluation results?"* — but its dominant variable is the **prompt template** (×126), its length finding is on **GSM8K reasoning**, and its code tables (HumanEval/MBPP under LLaDA) vary **decoding method**, not canvas. `infilling` 0, `pass@1` 0, `EvalPlus` 0. |
| **CaRE / Re-evaluating Confidence Remasking**, `arXiv:2607.24763` / `arXiv:2606.12232` | DBLP `CoRR 2026` | Cost-unit and remasking-strategy rankings, not canvas budget. Already covered by `RELATED_WORK.md` MUST-NOT-CLAIM 10. |
| **ParallelBench**, `arXiv:2510.04767` (v2) | DBLP `CoRR 2025` | Parallel-decoding quality loss. Different knob. |
| **HLP**, `arXiv:2410.03103` | ⚠️ **EMNLP 2025 Main** — DBLP `conf/emnlp/DingDWSKW25`, DOI `10.18653/v1/2025.emnlp-main.1672`; **confirmed by fetching ACL Anthology page `2025.emnlp-main.1672`**, which returns Anthology ID `2025.emnlp-main.1672`, Volume "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing", Month November, Year 2025 | Trains a horizon-length predictor; "the length is the problem" prior art on **infilling**. **See §6 — B10's `RELATED_WORK.md` currently records this as "arXiv-only, and specifically NOT accepted", which is now falsified.** |
| **Flexible-length Text Infilling for Discrete Diffusion Models**, `arXiv:2506.13579` | **EMNLP 2025 Main** — DBLP `conf/emnlp/ZhangSTT25`, DOI `10.18653/v1/2025.emnlp-main.1597`; **confirmed on ACL Anthology `2025.emnlp-main.1597`** | Variable-length infilling method. Text, not full-program code. |
| **Template Infilling (TI)**, `arXiv:2510.13870` | **ACL 2026 Long** — DBLP DOI `10.18653/V1/2026.ACL-LONG.284` | Conditioning strategy (structural anchors), improves code among other tasks. Not a canvas-budget measurement. |
| **DIA**, `arXiv:2606.04535` | **ACL 2026 Long** — DBLP DOI `10.18653/V1/2026.ACL-LONG.1205` | Dynamic end-anchor estimation; GSM8K/MATH format compliance. Not code canvas. |
| **iLLaDA** `arXiv:2606.25331`; **DID** `arXiv:2603.23507`; **AR-vs-MDLM controlled comparison** `arXiv:2603.22075`; **PCD** `arXiv:2608.09424` | arXiv-only (DBLP `CoRR`) | Architecture / objective / training-side work. `2603.22075` is 50 M-token TinyStories, no code benchmark. |
| **Weak-baseline genre**: `1907.06902` (recsys), `2407.07218` (fluid PDEs), `2204.03954` (text classification), `2505.09364` (diffusion recsys) | Established literature | Establishes that "reported weakness is a baseline-tuning artefact" is a **known species of result**. None is about diffusion LMs or code. Relevant as the genre S4 belongs to, not as preemption. |
| `arXiv:2509.01790` *Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs* | arXiv-only | Prompt sensitivity, not canvas. |

**Count: 2 clean preemptions (P1 DAEDAL, P2 ρ-EOS) + 2 supporting (P3, P4) vs 15+ merely adjacent.**
No bibliographic field above is invented; anything not settled is marked with the authority that refused.

---

## 3. G0 verdict, against its own written condition

G0's condition: *"KILL if an existing paper already shows a mask-diffusion LM's full-program
code score is dominated by its initial-canvas budget -> S4 is a reproduction, citable as
caveat only."*

**That branch FIRES.**

- DAEDAL (`2508.00819`, 2025-08) shows it on LLaDA-Instruct-8B, on **MBPP and HumanEval**
  (full-program, pass@1), with generation length as the swept variable and everything else
  frozen: MBPP +16.6 pp and HumanEval +28.7 pp across the canvas ladder.
- ρ-EOS (`2601.22527`, 2026-01) replicates it on two more LLaDA checkpoints.
- Both predate S4's 2026-08-12 statement by **≥6.5 months**, so the "concurrent work within
  2–3 months does not preempt" clause does not rescue it.
- Per `RELATED_WORK.md`'s framing rule, the bar applied here was *"already shows the same
  thing"*, not *"is in the same area"* — which is why DreamOn, CAL, LR-DLLM, the
  Evaluation-Illusion paper and CaRE are all classed adjacent despite being close: each
  differs in the axis (infilling vs full-program) or the knob (prompt template, remasking,
  cost unit) that S4 turns on.

**⇒ S4 is a reproduction. It may be cited as a caveat only.** Concretely, the citable form is
the narrow, honest one: *"we independently reproduce, on DreamOn-v0-7B and the EvalPlus
base/plus axes, the known canvas-budget sensitivity of mask-diffusion LMs on full-program
code (DAEDAL, ρ-EOS), and note that our repo's published DreamOn operating point
(initial_masks=8) sat in the crippled regime."* Any stronger phrasing — "artefact we
discovered", "reported weakness is substantially an artefact" as a finding — is now barred.

---

## 4. S4's terminal state, and the reasoning

**Terminal state: `caveat_citation_only` — preempted, no longer a claim. Not revivable
as a general result. 0 GPU authorised, now and in future.**

Reasoning, in the order the constraints bind:

1. **S4 could never have become "live" on this pass regardless of G0.** G1 needs ~4–6 GPU-h
   and no GPU was authorised (all 40 cards across all 5 nodes were occupied by long
   training runs; the task was explicitly CPU/network only). So the only two reachable
   outcomes were *caveat-citation-only* (preempted) or
   *measured-but-ungeneralised-and-parked* (not preempted).
2. **G0 fired, so it is the first of those two.** §3.
3. **G1 is now moot, and — this is the point worth recording — the published literature
   has already answered it in the direction that would have killed S4's general claim
   anyway.** G1 asked: sweep `initial_masks` on a SECOND mask-diffusion model; KILL S4 as a
   general claim if that model does not move ≥10 pp. ρ-EOS and DAEDAL both show LLaDA-8B
   moving **+16 pp** on MBPP, i.e. the second model *does* move. But that cuts against S4
   rather than for it: if canvas sensitivity is a **general, published property of
   mask-diffusion LMs**, then DreamOn's version of it is not a finding about DreamOn or
   about how our repo invoked it — it is the known behaviour of the model class. So the
   generality result that G1 was designed to test is exactly what makes S4 a reproduction.
   The pre-registration's own note — *"THIS IS THE EXPECTED OUTCOME and S4 must not be
   rescued from it"* — is honoured: S4 collapses to a footnote, and is not rescued.
4. **S4's collapse does not disturb anything else.** It carried no GPU authorisation
   (`gpu_authorised_by_this_subclaim: "none"`), and B10's own kill gate had already fired
   independently on a different contrast.

### Guards that survive S4 (still binding on any future citation)

- `must_not_claim_1` and `must_not_claim_2`, verbatim in §1. Preemption does not void them:
  they bar overstatements S4 could still tempt someone into (the stitch defect is an
  INTERACTION with canvas, not a second standalone 31 pp defect; the blast radius is NIL,
   17 arms re-graded / 2 move / +0.61 pp, so no diffusion-vs-AR asymmetry may be claimed).
- **S4-G2's standing rule, restated as it stands**: *"DreamOn HE+ is monotone in canvas up
  to c128; he_c512/mbpp_c128/mbpp_c512 never ran, so its peak is UNKNOWN and >= .4817. All
  'DreamOn reaches X' statements must be lower bounds."* Every "DreamOn reaches X" figure
  anywhere in this repo is a **LOWER BOUND**, because **he_c512, mbpp_c128 and mbpp_c512
  were never run**. G0 firing does not relax this — a reproduction still has to report its
  own numbers honestly.
- B10's `gate_4` standing rule continues to bind every pass@1 from this surface (no absolute
  pass@1 as a capability measurement without a decontaminated companion).

### What would revive S4

Nothing revives it as originally stated — preemption is not undone by more measurement.
The only routes to a *different*, live claim:

1. **A distinguishing contrast that P1/P2/P4 do not run.** They sweep length on
   LLaDA-family models on MBPP/HumanEval. They do **not** run: (a) the EvalPlus **base vs
   plus** axis split; (b) a **gold-refill feasibility ceiling** on the axis graded; (c) an
   expansion/deletion model (DreamOn) on **full-program** rather than infilling — DreamOn's
   own paper never touches MBPP/EvalPlus. A claim built strictly on one of those, with
   DAEDAL and ρ-EOS cited as the established phenomenon, could be new. It would need its
   own PROPOSAL.md and kill gate — **it does not inherit S4's**.
2. **If and only if that new gate is written and clears**, the old G1 shape (~4–6 GPU-h:
   `initial_masks` sweep on a second mask-diffusion model) becomes meaningful again. Not
   before, and not under S4's name.

Both routes require a **new** proposal. S4 itself is closed.

---

## 5. What this file does NOT touch

B10's Gate-1 KILL stands and was **independently reproduced** before this file was written,
from the raw per-task records in `evidence/gate1_base/score_base/`: n=1033,
`qwen_fim` 966/1033 = 0.9351403679, `dreamon_oracle` 965/1033 = 0.9341723136,
delta = +0.0009680542, McNemar b=39 c=38 discordant=77 both_pass=927 both_fail=29,
exact two-sided p = 1.000000 — **bit-exact** against `gate_1_result`. Both KILL conditions
hold. `lifecycle="dead"` is correct and unchanged. **B10 must not re-enter `ready_gpu`**;
verified by running `proposal/ready_queue.py` before and after this edit (identical SUMMARY,
B10 in the `dead` bucket both times).

---

## 6. Correction owed to `RELATED_WORK.md` (recorded, not applied)

`RELATED_WORK.md` §3.6 records HLP (`arXiv:2410.03103`) as **"arXiv-only, and specifically
NOT accepted"** with an OpenReview `Rejected_Submission` venueid, and warns *"Do not cite as
an ICLR paper"*. The ICLR half is still right, but the paper **was subsequently accepted to
EMNLP 2025 Main**: DBLP `conf/emnlp/DingDWSKW25`, DOI `10.18653/v1/2025.emnlp-main.1672`,
confirmed by fetching ACL Anthology page `2025.emnlp-main.1672`
(Volume: Proceedings of EMNLP 2025; Month: November; Year: 2025). This is exactly the failure mode
`memory/venue-verify-acl-family-needs-anthology.md` warns about: an OpenReview check is
**not authoritative for the ACL family**, and a rejection at one venue is not a terminal
venue verdict.

Not applied here because `RELATED_WORK.md` is B10's provenance document and B10 is dead;
recorded so that any future citation of HLP uses the Anthology entry. Same note applies to
`Flexible-length Text Infilling` (`2506.13579` → EMNLP 2025 Main
`2025.emnlp-main.1597`), which `RELATED_WORK.md` §3.6 lists as arXiv-only and explicitly
flags as "not re-verified this session".

---

## 7. Provenance

- Written 2026-08-15. **0 GPU used** (search was CPU + network only; no model loaded, no
  node contacted for compute). No new measurement; every S4 number is quoted from
  `STATUS.json:subclaim_S4_inherited_from_A05` or A05's archived record, both of which
  predate this file.
- Network via `http_proxy=http://hy-proxy.woa.com:3128`.
- Machine-readable counterpart: `STATUS.json:s4_disposition_20260815` (append-only key; no
  pre-existing key was modified — verified by hashing all 35 keys before the edit and
  re-asserting byte-equality after).
- Inputs read: this directory's `STATUS.json`, `PROTOCOL_NOTE.md`, `RELATED_WORK.md`,
  `GATE1_BASE_AXIS_VERDICT.md`, `evidence/gate1_base/score_base/*.json`;
  `proposal/archive/A05-structural-dllm-cost-frontier/STATUS.json`;
  `proposal/LIFECYCLE_SCHEMA.md`; `proposal/ready_queue.py`; `proposal/append_status.py`.
