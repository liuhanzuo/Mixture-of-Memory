# dLLM frontier map + proposals — Claude track (independent of tcodex track)

**Written**: 2026-08-15 · **GPU used: 0** (literature + CPU re-analysis of
already-generated outputs only) · **Track**: this is the *second of two
independent tracks* on the same question. I did not read
`TCODEX_FRONTIER_20260815.md` (it did not exist on disk when I started; verified
by `ls`), so any agreement between the two documents is genuine cross-check.

**Provenance discipline used throughout**: every bibliographic field below is
either (a) RETRIEVED from a named authority and marked with it, or (b) marked
INFERRED / UNVERIFIED. Venue strings were checked family-specifically:
OpenReview `venueid` for ICLR/NeurIPS/ICML, ACL Anthology + DBLP for ACL-family.
Nothing in the tables is composed from memory.

---

## 0. Executive summary (read this if you read nothing else)

1. **All three named leads resolve.** DAEDAL = *Beyond Fixed: Training-Free
   Variable-Length Denoising for DLLMs*, **ICLR 2026 Poster** (venueid verified,
   Camera_Ready_Revision present). iLLaDA = *Improved Large Language Diffusion
   Models*, arXiv 2606.25331, an **8B from-scratch masked-diffusion LM**, weights
   live at `GSAI-ML/iLLaDA-8B-{Base,Instruct}`. ELF = *ELF: Embedded Language
   Flows* (2605.10938, He/Andreas/Kim et al.) — **continuous-embedding flow
   matching**, and the *only* one of ~15 "ELF" collisions that is a dLLM method.
   §2 gives the collision list.

2. **The S4 canvas finding is NOT preempted, but its framing must change.**
   DAEDAL's Table 1 already publishes a canvas sweep (64→2048) on LLaDA-Instruct-8B
   with HumanEval moving **18.9 → 47.6** (+28.7 pp). So "canvas budget dominates a
   dLLM's code score" is *established* in the literature — S4's **generality
   claim is preempted** and S4-G1 should be considered pre-fired by literature.
   **What is NOT preempted** is the diagnostic use: DAEDAL and every successor
   *fix* the length problem as a method, and each still reports **one number per
   baseline model**; none of them audits *whether the numbers other papers publish
   about a baseline were produced at that baseline's canvas optimum*. §4 gives the
   verdict with the specific evidence, plus a **live case in the literature**: two
   2026 papers publish DreamOn baselines at initial mask length **1** and **64**
   and get 37.9 / 39.1 mean Pass@1, while this group measures **93.4** on
   SingleLine with a tuned canvas. That is the S4 mechanism, in someone else's
   table, at ~50 pp.

3. **The most valuable seam is not a combination — it is a measurement.** Every
   variable-length paper I read (DAEDAL, ρ-EOS, CAL, LR-DLLM) reports efficiency
   as a **token-ratio**, and **none reports forward passes (NFE) or wall-clock**
   for its own adaptive loop. But adaptive length is *implemented as extra forward
   passes*: DAEDAL Algorithm 1 line 5 is a full `f_θ(x)` per expansion iteration,
   and CAL's own table says its search costs **11–18 extra first-step forwards**.
   Meanwhile this group already has the counter-example on disk: DreamOn's
   `nfe_mean` rises **172.3 → 393.7 → 593.4** as canvas goes 8→32→128 (A05
   `cells/*.json`), i.e. the quality gain *is bought with compute*, and Eratio
   goes the wrong way from NFE. §3.2 and Proposal **P1**.

4. **First experiment to run: P1, and it costs 0 GPU.** It re-mines files that
   already exist (`cells/*.json` per-item pass maps + `cost_and_behaviour`, 5
   canvas cells; `score_base/*.json` 6 arms × 1033) to produce the
   quality-per-NFE frontier that the published variable-length papers do not
   report, and it can *falsify itself*: if DreamOn's canvas curve is already
   NFE-Pareto-optimal, the whole "adaptive length buys quality with compute"
   critique dies. Ranking in §6.

5. **I executed two proposals' own 0-GPU first steps before finishing, and they
   moved the ranking.** P5 (iLLaDA weights-vs-protocol) **mostly fired its own
   gate** — iLLaDA's Table 4 already ablates the scoring change and it is worth
   only **+0.6 pp on ARC-C** (~4 % of the 14.9 pp delta) against my ≥20 %
   threshold, so **my pre-registered prediction was wrong** and P5 is demoted to
   last-but-one (§5.P5.0). P6 (continuous lane / ELF) had its **premise
   confirmed** — ELF hard-codes `Sequence length 1024`, has **0 occurrences of
   "padding"** and no length-adaptation mechanism, **has released weights**, and
   is only **105M params** — so P6 is promoted from 7th to 4th and its cost
   estimate drops from ~8–16 to **~2–4 GPU-h** (§5.P6.0). Both checks cost zero
   GPU-seconds.

---

## 1. Frontier map

### 1.1 Venue-verified core table

Legend: **Authority** = where the venue string came from. `OR` = OpenReview
api2 `venueid` (authoritative for ICLR/NeurIPS/ICML). `arXiv` = arXiv API
(authoritative for arXiv id + dates only, **never** for venue).

| Paper | arXiv | Venue | venueid / authority | Family / what it does |
|---|---|---|---|---|
| Large Language Diffusion Models (**LLaDA**) | 2502.09992 (title RETRIEVED via arXiv `id_list`, confirmed) | **NeurIPS 2025 oral** | `NeurIPS.cc/2025/Conference` (OR) | 8B from-scratch masked diffusion LM |
| **LLaDA 1.5**: Variance-Reduced Preference Optimization | 2505.19223 | **NOT a conference paper.** OR shows `ICLR.cc/2026/Conference/Withdrawn_Submission` **and** `dblp.org/journals/CORR/2025` | OR (both records) | RL/preference alignment (VRPO) on LLaDA |
| **LLaDA-MoE**: A Sparse MoE Diffusion Language Model | 2509.24389 | **preprint** (`CoRR 2025`) | `dblp.org/journals/CORR/2025` via OR-DBLP mirror | MoE dLLM, 7B-A1B |
| LLaDA MoE v2 | 2608.03457 | **preprint** (no venue found) | arXiv only | MoE scaling |
| **iLLaDA** (*Improved Large Language Diffusion Models*) | 2606.25331 | **preprint**, no venue found in OR | arXiv; OR search returned no match | 8B from-scratch, 12T tokens, bidirectional |
| **DAEDAL** (*Beyond Fixed*) | 2508.00819 (v2 2026-08-15) | **ICLR 2026 Poster** | `ICLR.cc/2026/Conference`, Submission1382, **Camera_Ready_Revision present** (OR) | training-free adaptive length, 2-stage |
| **DreamOn** | 2602.01326 | **ICLR 2026 Poster** | `ICLR.cc/2026/Conference` (OR) | *trained* expand/delete states, variable-length infilling |
| **Dream-Coder 7B** | 2509.01142 | **preprint** (`CoRR 2025`) | DBLP-via-OR | AR→diffusion adapted code dLLM |
| **Fast-dLLM** | 2505.22618 | **ICLR 2026 Poster** | `ICLR.cc/2026/Conference` (OR) | block KV-cache + confidence parallel decode |
| **FlexMDM** (*Any-Order Flexible Length Masked Diffusion*) | 2509.01025 | **ICLR 2026 Poster** (also `SPIGM @ NeurIPS 2025` workshop) | `ICLR.cc/2026/Conference` + `NeurIPS.cc/2025/Workshop/SPIGM` (OR) | insertion-capable MDM, trained |
| **BD3-LM** (*Block Diffusion*) | 2503.09573 (title RETRIEVED via arXiv `id_list`, confirmed) | **ICLR 2025 Oral** | `ICLR.cc/2025/Conference` (OR) | semi-AR block interpolation |
| **ρ-EOS** | 2601.22527 | **preprint** (`CoRR 2026`) | `dblp.org/journals/CORR/2026` via OR | training-free **bidirectional** length control via EOS density |
| **CAL** (*Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly*) | 2602.00476 | **preprint**, no venue found | arXiv | training-free length *search* via first-step confidence |
| **LR-DLLM** (*…via Length Regularization*) | 2602.07546 | **preprint** ("Preprint. February 10, 2026" in PDF) | arXiv + PDF header | log L-regularised length criterion, bidirectional |
| **ELF** (*Embedded Language Flows*) | 2605.10938 | **preprint**, self-described "Tech report" | arXiv comment field | continuous-embedding flow-matching LM |

**Explicit UNVERIFIED flags**: the LLaDA (2502.09992) and BD3-LM (2503.09573)
arXiv ids were initially from recall and have since been **confirmed** by an
arXiv `id_list` fetch in this session (titles returned match). Everything
else in the arXiv column was returned by the arXiv API in this session. What
remains UNVERIFIED is listed in §7.

> **✅ CONFIRMED 2026-08-15 — DAEDAL row (`2508.00819`) independently re-verified; NO change needed.**
> An independent pass had recorded DAEDAL as `CoRR 2025 / arXiv-only`
> (`Mixture-of-Memory/proposal/backlog/B10-dllm-infilling-ar-dominance/S4_DISPOSITION.md:94-100`),
> conflicting with this table. Re-queried OpenReview api2 `/notes/search`, note
> `id = forum = Ic2A2gCseC`: `"venue": "ICLR 2026 Poster"`, `"venueid": "ICLR.cc/2026/Conference"`,
> and `ICLR.cc/2026/Conference/Submission1382/-/Camera_Ready_Revision` present in `invitations[]`.
> **This row is right as written**; the other pass was amended. Decision `pdate` 2026-01-26.
>
> ⚠️ **But one methodological caveat now applies to this table's `preprint (CoRR 20xx)` rows.**
> DAEDAL demonstrates that a **DBLP `CoRR` record coexists with an accepted ICLR-2026 record** —
> DBLP said `CoRR 2025` for DAEDAL on the very same day OpenReview said ICLR 2026 Poster. So every
> row here whose venue came from **DBLP-via-OR** rather than a `venueid` — `LLaDA-MoE` (2509.24389),
> `Dream-Coder` (2509.01142), `ρ-EOS` (2601.22527), plus the arXiv-only rows `iLLaDA` (2606.25331),
> `CAL` (2602.00476), `LR-DLLM` (2602.07546), `ELF` (2605.10938), `LLaDA-MoE-v2` (2608.03457) — is
> properly **`NOT-FOUND`, not `IS-A-PREPRINT`**, until re-run through `api2 /notes/search`. Not
> done in this pass.
>
> Recipe note (cost this pass ~30 min): `api2 /notes?` **is** challenge-gated
> (403 `ChallengeRequiredError`) but **`api2 /notes/search?term=…&source=forum&limit=100` is NOT** —
> 403 on `/notes?` must not be read as "OpenReview unreachable". api **v1** returns only DBLP-mirror
> records (0 title hits for DAEDAL) and must not be used alone for 2026 venues.
>
> Evidence + verbatim quotes: `VENUE_RESOLUTION_20260815.md` (same directory).

### 1.2 Sub-area notes (retrieved, non-exhaustive)

- **Variable/adaptive length is now a crowded lane.** Chronologically:
  DAEDAL (2025-08, ICLR'26) → FlexMDM (2025-08, ICLR'26) → DreamOn (2026-02,
  ICLR'26) → ρ-EOS (2026-01) → CAL (2026-01) → LR-DLLM (2026-02). Split by
  mechanism: **trained insertion/deletion states** (DreamOn, FlexMDM) vs
  **training-free confidence signals** (DAEDAL, ρ-EOS, CAL, LR-DLLM). Split by
  direction: **expand-only** (DAEDAL, DreamOn expand-state) vs **bidirectional
  expand+contract** (ρ-EOS, CAL, LR-DLLM).
- **The training-free ones all key on the same signal**: end-of-sequence
  behaviour at the first denoising step. DAEDAL: mean EOS *confidence* in a
  terminal window `W_eos` vs threshold `τ_eos`. ρ-EOS: implicit EOS *density* ρ,
  which additionally licenses contraction. CAL: an "Oracle Peak" in first-step
  confidence near ground-truth length **plus an explicit Length-Bias
  calibration**. LR-DLLM: the same bias, modelled as **∝ log L** (their Fig. 1
  reports log fit R²=0.895 vs linear R²=0.738). *This is a convergent
  discovery of one artefact — the length-induced confidence bias — by four groups.*
- **Acceleration** is a separate lane, headed by Fast-dLLM (block KV cache +
  confidence parallel decode, **ICLR 2026 Poster**, OR-verified). The 2026 wave is
  enumerated in the last bullet of this section.
- **Few-step distillation** is also a lane of its own: FS-DFM (2509.20624),
  Consistent Diffusion LMs (2605.00161), Flow Map LMs (2602.16813), Trajectory
  Self-Distillation (2602.12262), Multi-Mask few-step (2607.19686), OPTD
  (2608.02942). **Titles/ids retrieved; contents NOT read.**
- **Two directly relevant negative/diagnostic results exist** — and I **read both
  in full after drafting the proposals**, so these are RESOLVED, not flagged:
  - *Re-evaluating Confidence Remasking in Masked Diffusion Language Models*
    (2606.12232, Frkovic, Jazbec, Zhang, Naesseth, Bogunovic, Nalisnick,
    2026-06-10). **This is the genre-precedent for our diagnostic work and should
    be cited as such.** It re-evaluates a *representative* post-hoc remasking
    method (WINO) and finds "little-to-no benefit over confidence-based unmasking
    alone" under standard decoding settings, concluding "the benefits of post-hoc
    confidence-based remasking are highly setting-dependent".
    **Does it preempt P1/P3? No** — its axis is **remasking/self-correction**, not
    length/canvas, and its cost axis is block length, not NFE. But it establishes
    that *"published dLLM decoding gains are setting-dependent"* is a publishable
    result in this lane, which **raises** P3's prospects and gives it a template.
  - *Faster but Different: Diagnosing and Controlling Content Drift in
    Accelerated Multimodal Diffusion LMs* (2607.29079, Dou & Shu, 2026-07-31).
    Compares Fast-dLLM to the same model unaccelerated on 300 real images. Two
    findings bear directly on **P2**: (i) "confidence-threshold tuning changes
    decoding behavior **but not baseline agreement**"; (ii) the binding variable
    was the **KV-cache refresh interval**, yielding "a monotonic speed–agreement
    frontier" and near-exact agreement at 1.3× speedup.
    **Does it preempt P2? No** — it measures *output agreement* under
    acceleration, not the interaction between acceleration and *adaptive length*,
    and it is multimodal (dMLLM). **But finding (i) is evidence AGAINST P2's
    mechanism**: if the confidence threshold barely moves outputs in their
    setting, the contention P2 posits may be weak. P2 must engage this, and it is
    part of why P2 ranks 6th, not 1st.
- **Also relevant, retrieved but NOT read**: dLLM-Cache / dKV-Cache (cited by
  DAEDAL §2; ids not fetched), plus a 2026 acceleration wave I only enumerated:
  Fast-dLLM v2 (2509.26328), Fast-dLLM++ (2606.02955), Sangam (2607.04206),
  HERALD (2606.21633), BlockBatch (2605.29233), CORA-Diff (2608.11235),
  WeDLM (2512.22737).

---

## 2. Named leads resolved

### 2.1 DAEDAL — RESOLVED, and it is the single most important paper for us

- **Title**: *Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion
  Large Language Models*
- **Authors** (arXiv API): Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao,
  Jiaqi Wang, Dahua Lin
- **arXiv**: 2508.00819, v1 2025-08-01, **v2 2026-08-15** (i.e. updated *today*)
- **Venue**: **ICLR 2026 Poster**, `venueid = ICLR.cc/2026/Conference`,
  invitations include `Submission1382/-/Camera_Ready_Revision` → camera-ready
  exists, this is not "submitted to". Authority: OpenReview api2.
- **Code**: `https://github.com/Li-Jinsong/DAEDAL` (from arXiv comment field)

**Exact mechanism** (read from the PDF, Algorithm 1, §3.2):
- Hyperparameters: `L_init`, `L_max`, `τ_eos`, `τ_high`, `τ_low`, `τ_expand`,
  `E_factor` (masks added per expansion), `W_eos` (terminal window size).
- **Stage 1 — Initial Length Adjustment** (Alg. 1 lines 4–12): loop *before*
  denoising. Each iteration runs a **full forward pass** `L_logits ← f_θ(x)`,
  computes mean EOS confidence over the last `W_eos` positions; if
  `conf_eos < τ_eos`, append `E_factor` `[MASK]`s and repeat; else break.
- **Stage 2 — Iterative Mask Insertion** (lines 13–27): during denoising, fill
  all masks with `P_conf > τ_high`; among masks with `P_conf < τ_low`, take the
  argmin, and if `conf_eos < τ_expand` **replace that single `[MASK]` with
  `E_factor` `[MASK]`s** — i.e. mid-sequence insertion, not just append.
- Models: LLaDA-Instruct-8B and LLaDA-1.5-8B. 8×A800-80G, batch 8. Explicitly
  **no caching/acceleration** ("using the official generation code released with
  LLaDA, without any acceleration or caching optimizations proposed in
  subsequent works").

**What it measures** — Table 1 (LLaDA-Instruct-8B), baseline at fixed lengths
64/128/256/512/1024/2048, all four values transcribed verbatim from the PDF:

| Benchmark | 64 | 128 | 256 | 512 | 1024 | 2048 | DAEDAL (L_init=64) |
|---|---|---|---|---|---|---|---|
| GSM8K Acc | 48.0 | 67.9 | 77.6 | 83.3 | **83.8** | 82.6 | **85.8** |
| MATH500 Acc | 24.0 | 29.0 | 35.6 | 38.8 | 39.4 | **39.6** | **44.2** |
| MBPP Acc | 20.8 | 28.0 | 37.4 | 38.2 | 37.4 | **38.8** | **40.8** |
| HumanEval Acc | 18.9 | 26.2 | 36.0 | 47.0 | **47.6** | 47.0 | **48.2** |
| Average Acc | 27.93 | 37.78 | 46.65 | 51.83 | 52.05 | 52.00 | **54.75** |

Metrics reported: `Acc`, `Etoken` (effective tokens = response minus trailing
EOS padding), `Ntoken` (total tokens), `Eratio = Etoken/Ntoken`.

**The critical question you asked — does it MEASURE canvas sensitivity as a
diagnostic about baselines, or merely FIX it as a method?**

**Answer: it measures the sensitivity, but uses it only as motivation for its own
method — it never turns it on the literature.** Precisely:
- It **does** publish the full sweep (so the *phenomenon* is public): GSM8K
  48.0→83.8 (+35.8 pp), HumanEval 18.9→47.6 (+28.7 pp) purely from the length
  integer.
- It **does** note non-monotonicity ("excessively long initial lengths may
  degrade model performance", §1; Table 1 GSM8K peaks at 1024 then falls to 82.6).
- But its framing is *"the optimal length varies across benchmarks, therefore you
  need our method"* (§4.3). It **never** asks whether previously published numbers
  for LLaDA (or any baseline) were obtained at that model's canvas optimum, and it
  reports **no** re-audit of any third-party result. There is no "prior work
  under-reported X because of length" claim anywhere in the paper.
- **Its own baseline is very well tuned** — six lengths per benchmark, and it
  reports its win against the *best* of them ("The best configuration for the
  baseline is highlighted in orange"). Credit where due: DAEDAL is *not* an
  example of the baseline-tuning asymmetry problem. This makes it a *strong*
  paper and a *weak* target.

**Its own robustness ablations** (Tables 4, 5, Fig. 5) are unusually thorough:
`L_init` ∈ {32..512} → HumanEval Acc **identical 48.2 at all five settings**;
`E_factor` ∈ {8..32} → 85.8/85.8/86.4/86.3; `W_eos` ∈ {8..32} → 82.9→85.8
(the one real sensitivity); 32-config grid over (τ_high,τ_low) and
(τ_eos,τ_expand) all "comparable to the best-performing baseline".

**Its blind spot, which is our opening**: `grep -i` over the full extracted text
for `NFE | forward pass | wall | latency | throughput | second` finds *no*
cost measurement for DAEDAL itself. "Efficiency" is argued **entirely** via
`Ntoken` and `Eratio` (§4.3 "DAEDAL significantly improves computational
efficiency … the total number of tokens (Ntoken) generated by DAEDAL is
generally lower"). But Stage 1 is a **forward pass per expansion step** and
Stage 2 *lengthens the sequence mid-denoise*, which lengthens the remaining
denoising loop. Notably its `Ntoken` at L_init=64 is **363 (GSM8K) / 813
(HumanEval)** — i.e. HumanEval's adaptive `Ntoken=813` is **larger** than the
best fixed baseline's 1024-length arm is efficient at, and larger than 512. So
even on its own token metric the code benchmark is the awkward case.
**INFERRED (not stated by the paper): DAEDAL's NFE is strictly greater than a
fixed-length run at the same final length.** This is my inference from Alg. 1,
not a measurement, and P1/P2 are designed to measure it.

### 2.2 ELF — RESOLVED, with a large collision set

There are **many** "ELF"s. Retrieved from arXiv API (`ti:"ELF"`, 15 results, plus
`abs:"ELF" AND cat:cs.CL`):

| Candidate | arXiv | Is it a dLLM method? |
|---|---|---|
| **ELF: Embedded Language Flows** | **2605.10938** (v1 2026-05-11, v2 2026-06-26) | **YES — this is the one** |
| ELF: A Family of Encoder-Free ECG-Language Models | 2601.18798 | No (ECG) |
| ELF: Efficient Logic Synthesis by Pruning Redundancy in Refactoring | 2508.08073 | No (EDA) |
| Adversarial Malware Generation in Linux **ELF** Binaries | 2604.22639 | No (the executable format) |
| sqlelf: a SQL-centric Approach to **ELF** Analysis | 2405.03883 | No (same) |
| ELF-Gym (LLM-generated features for tabular) | 2410.12865 | No |
| Uni-ELF (electrolyte formulation) | 2407.06152 | No |
| ELF-UA (gaze estimation) | 2406.09481 | No |
| ELFS (label-free coreset selection) | 2406.04273 | No |
| Elf autoencoder (flat-band materials) | 2406.11967 | No |
| Sonora Substellar Atmosphere Models **Elf Owl** | 2402.00756 | No (astro) |
| Elfs, transducers and quantum walks | 2605.30013 | No |
| ELFS-SA (CMB foregrounds) | 2510.20793 | No |
| ELF(S) enhanced low-flux sensitivity (bipolar devices) | 1907.01408 | No |
| ELF currents in the ionosphere | 1201.5349 | No |

**The dLLM ELF**: *ELF: Embedded Language Flows*, arXiv 2605.10938. Authors
(arXiv API): Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim,
Jacob Andreas, **Kaiming He**. Self-described **"Tech report"** in the arXiv
comment (v2 adds distillation results in Appendix B); **no venue found** — I
searched OpenReview for "ELF Embedded Language Flows" and got **zero** matches,
so: **venue = UNVERIFIED / most likely preprint**, authority tried = OpenReview
api2 `/notes/search`.

Mechanism (from abstract, **paper not read in full**): a **continuous**-space
DLM using continuous-time **Flow Matching** over token *embeddings*, staying in
continuous embedding space until the final timestep, where a shared-weight
network maps to discrete tokens. Claimed benefit: image-domain techniques
(explicitly **classifier-free guidance**) transfer directly; claims to beat
leading discrete *and* continuous DLMs with **fewer sampling steps**. There are
two 2026 follow-ons in the same continuous lane I found but did not read:
DeltaFlow (2608.01240, "Noise-Adaptive Bidirectional Gated Delta Networks for
Embedded Language Flows"), *Speech Meets ELF* (2606.10368), and adjacent
diagnostics *Low Perplexity is Repetition* (2607.00588) and *Continuous Language
Diffusion as a Decoder-Interface Problem* (2606.08810).

⚠️ **Important caveat for the lead**: if the lead grouped ELF with DAEDAL and
iLLaDA as "recently hot dLLM work we might combine", note that **ELF is in a
different model class** — continuous embedding flows, no `[MASK]` token, no
canvas of masks. **Most DAEDAL/DreamOn-style length machinery does not even have
a referent in ELF** (there are no mask tokens to insert). Combining ELF with
DAEDAL is a category error unless one first asks what "length" means in a
continuous flow. §3.5 turns exactly that into a real proposal (P6) rather than
discarding it.

### 2.3 iLLaDA / LLaDA family tree — RESOLVED

`iLLaDA` **exists and is the GSAI-ML team's own successor to LLaDA** (not a
third-party variant). *Improved Large Language Diffusion Models*, arXiv
**2606.25331** (2026-06-24). Authors: Shen Nie, Qiyang Min, Shaoxuan Xu, Zihao
Huang, Yuxuan Song, Yong Shan, Yankai Lin, Wayne Xin Zhao, Chongxuan Li,
Ji-Rong Wen. **No venue found in OpenReview → preprint (UNVERIFIED).**

From the abstract (RETRIEVED): 8B, from scratch, **fully bidirectional
attention**, masked-diffusion objective through *both* pre-training and SFT,
**12T pre-training tokens**, SFT on a 25B-token instruction corpus × 12 epochs.
Reported vs LLaDA: Base +21.6 BBH, +14.9 ARC-C; Instruct +14.5 MATH, **+16.5
HumanEval**. **It uses "variable-length generation for efficiency" and
"confidence-based scoring for multiple-choice evaluation"** — note both of those
are *evaluation-protocol* changes bundled into a model release; see the seam in
§3.4.

Family tree with dates and HF ids (HF ids RETRIEVED live from
`huggingface.co/api/models` this session; download counts as of today):

| Model | Date | HF repo id | HF dl | Venue |
|---|---|---|---|---|
| LLaDA-8B-Base | 2025-02 | `GSAI-ML/LLaDA-8B-Base` | 69,743 | NeurIPS 2025 oral (OR) |
| LLaDA-8B-Instruct | 2025-02 | `GSAI-ML/LLaDA-8B-Instruct` | 358,996 | ″ |
| LLaDA 1.5 (VRPO) | 2025-05 | `GSAI-ML/LLaDA-1.5` | 67,104 | **ICLR'26 Withdrawn** + CoRR (OR) |
| LLaDA-V (visual instruction tuning) | 2025-05 | not fetched | — | UNVERIFIED |
| LLaDA-MoE-7B-A1B-Base | 2025-09 | `inclusionAI/LLaDA-MoE-7B-A1B-Base` | 576 | CoRR 2025 |
| LLaDA-MoE-7B-A1B-Instruct | 2025-09 | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | 19,897 | ″ |
| **iLLaDA-8B-Base** | **2026-06** | **`GSAI-ML/iLLaDA-8B-Base`** | 1,034 | preprint |
| **iLLaDA-8B-Instruct** | **2026-06** | **`GSAI-ML/iLLaDA-8B-Instruct`** | 3,799 | preprint |
| LLaDA MoE v2 | 2026-08 | not fetched | — | preprint |

Note **LLaDA-MoE is `inclusionAI/`, not `GSAI-ML/`** — different org; do not
guess the prefix when downloading. Also retrieved (titles only, not read):
LLaDA-VLA, LLaDA-MedV, LLaDA-Rec, LLaDA-TTS, LLaDA-o, Arg-LLaDA, DSL-LLaDA —
the name is now a platform, so "LLaDA-X" in a conversation is ambiguous.

**On-disk assets vs these leads** — important practical note: this group's disk
has **Dream-family** models (`Dream-Coder-v0-{Base,Instruct}-7B`, `DreamOn-v0-7B`)
plus `Qwen2.5-Coder-7B` and `Scaffold-v0-stage1-7B`, and **no LLaDA-family
weights at all**. DAEDAL/ρ-EOS/CAL are all developed on **LLaDA**. So any
proposal that wants to test a *LLaDA-specific* claim needs
`GSAI-ML/LLaDA-8B-Instruct` (≈16 GB bf16) — routine per the lead, but it is a
download, and the proposals below are ordered so that the 0-GPU ones do not
depend on it.

---

## 3. The seams

### 3.1 SEAM A — Adaptive length and confidence-parallel decoding contend for the *same* confidence budget

Fast-dLLM's parallel decoding rule is "unmask every token whose confidence
exceeds a threshold **in one step**". DAEDAL's Stage 2 rule is "if a token's
confidence is **below** `τ_low` *and* terminal EOS confidence is below
`τ_expand`, **replace that token with `E_factor` masks**". These read the *same*
scalar field and act on **opposite tails**, so mechanically they *look*
composable — one drains the high tail, the other expands the low tail.

They are **not** independent, for a reason neither paper can see alone:
1. **Parallel decoding removes exactly the evidence adaptive length needs.**
   DAEDAL's `conf_eos` is measured *at the terminus of the current canvas*.
   Aggressive parallel unmasking commits the terminal region early (EOS is
   typically the highest-confidence prediction in an over-long canvas — it is
   *why* Eratio is low at length 2048). Once the terminus is committed to EOS,
   `conf_eos > τ_eos` forever and **Stage 1 can never expand again**. Prediction:
   **composition is not additive; it is order-dependent, and the gain from
   adaptive length should shrink monotonically as the parallel-decode threshold
   is loosened.** DAEDAL itself hints at this without testing it — it notes
   `τ_high` "is analogous to the confident decoding strategy proposed in Dimple"
   (§4.4), i.e. its own fill rule *is* a parallel decoder, and it grid-searches
   `(τ_high, τ_low)` **jointly** because they are "interdependent". Nobody has
   crossed that grid with a *cache-based* accelerator.
2. **They contend for the same slack.** Both harvest the *same* waste: an
   over-provisioned canvas. Adaptive length reclaims it by not allocating it;
   parallel decoding reclaims it by resolving it in fewer steps. Two methods that
   each claim ~2× on the same waste do **not** give 4×.

→ Proposal **P2**.

### 3.2 SEAM B — Every variable-length paper reports token-ratio efficiency; none reports NFE for its own loop. This group has the NFE data that shows why that matters.

Verified by reading the papers:
- **DAEDAL**: efficiency = `Ntoken` + `Eratio` only. No NFE/wall-clock anywhere
  (grep-verified over the full text).
- **CAL** (2602.00476): *does* report search cost — **"11 to 18 additional
  first-step forward passes on average"** (`Stps.` column: 11.1–18.2) — and
  frames it as "acceptable". It reports **no total-NFE comparison** against the
  fixed-length baseline it beats.
- **LR-DLLM** (2602.07546): "modest inference-time overhead … the average number
  of additional forward calls per generated token remains small and bounded
  (Table 9)" — a *per-token* normalisation, which **hides** the fact that
  adaptive methods change the denominator (they change how many tokens exist).
- **ρ-EOS**: abstract claims "substantially improving inference efficiency and
  token utilization" — again utilisation.

Now the counter-evidence **already on this group's disk** (A05
`evidence/cells/*.json`, `cost_and_behaviour`, DreamOn-v0-7B, 8×H20, 2026-08-12):

| cell | pass@1 (plus, as-run) | `nfe_mean` | `nfe_median` | `tokens_fed_effective_mean` | `gen_tok_mean` | wall (s, 164 or 378 items) |
|---|---|---|---|---|---|---|
| he_c8 | .1280 | **172.3** | 8 | 39,944 | 2.3 | 2,802 |
| he_c32 | .2134 | **393.7** | 32 | 124,348 | 12.9 | 6,480 |
| he_c128 | .1707 (**.4817 corrected**) | **593.4** | 128 | 240,414 | 48.5 | 21,850 |
| mbpp_c8 | .0899 | **153.4** | 8 | 23,367 | 1.6 | 5,073 |
| mbpp_c32 | .3545 | **466.0** | 32 | 101,202 | 11.4 | 15,585 |

So on HE+, going c8→c128 buys **+35.4 pp** (corrected .1341→.4817) for
**3.44× the NFE** and **6.0× the tokens fed** and **7.8× the wall-clock**. That
is a *cost–quality trade*, not a free lunch — and it is exactly the axis on which
the published variable-length papers report nothing. **A method that "matches the
best tuned fixed-length baseline" while spending more forward passes than that
baseline has not made decoding better; it has made tuning unnecessary.** Those
are both real contributions but they are different ones, and the literature
currently conflates them under the word "efficiency".

→ Proposals **P1** (0 GPU, mine what exists) and **P2**.

### 3.3 SEAM C — A published baseline's canvas is a free parameter, and at least two 2026 papers set it adversarially-low *without meaning to*

This is the sharpest live instance of the group's own S4 mechanism, found in
someone else's table:

- **LR-DLLM Table 3** (transcribed verbatim from the PDF): DreamCoder-7B
  Baseline mean **39.1**; **+DreamOn 37.9** (Random Span 21.6 / Single-line 73.5
  / Multi-line 36.1). And the caption states: *"For DreamOn, we set the initial
  mask length to **1** to isolate its variable-length adjustment capability."*
- Its own baseline: *"the **Baseline** infills with a fixed length of **64**
  [MASK] tokens, whereas all variable-length methods … are evaluated under the
  same adjustment budget with MAX_LENGTH=128"*.
- Meanwhile **this group measures `dreamon_oracle` = .9342 and `dreamon_fim` =
  .8664 pass@1 on HumanEval-SingleLineInfilling base axis, n=1033** (B10
  `gate_1_result.arm_pass_at_1_base`) — vs LR-DLLM's 73.5 for DreamOn
  single-line.

Is that a contradiction? **Not necessarily, and I will not claim it is** — the
splits, graders, `MAX_LENGTH` and model (Dream-Coder vs DreamOn-v0-7B checkpoint)
all differ, and LR-DLLM's single-line number is on *its* harness. But it *is* a
20-pp-scale discrepancy on nominally the same benchmark family, and the stated
cause is a knob (`initial mask length = 1`, `MAX_LENGTH=128`). **The reasonable
inference is: DreamOn's published-by-others numbers are configuration-dominated,
which is S4's claim, appearing in the wild.** Note also LR-DLLM's own framing
concedes DreamOn "exhibits inconsistent behavior across span regimes … degrading
Multi-line infilling" — a conclusion *about the model* drawn from a run where its
canvas was set to 1.

Symmetrically, CAL's table is a model of good practice on this axis — it reports
**four** fixed lengths (L=4/8/16/32) per model **and** an AVG, so its baseline
cannot be accused of a single unlucky canvas. And it shows **DAEDAL losing to the
fixed baseline** when the initial length is *over*-estimated (LLaDA-Base
single-line L=32: baseline 49.0 vs +DAEDAL **43.1**; L=16: 56.6 vs **53.4**) —
because DAEDAL is expand-only. **That is a published, third-party falsification
of "adaptive length is a free win", and it is the strongest single citation for
why bidirectionality matters.**

→ Proposals **P3** and **P4**.

### 3.4 SEAM D — Model releases bundle protocol changes with weight changes

iLLaDA's abstract says it "use[s] variable-length generation for efficiency and
introduce[s] confidence-based scoring for multiple-choice evaluation" *in the same
breath* as its 12T-token pre-training. Its headline deltas vs LLaDA (+16.5
HumanEval, +14.9 ARC-C) are therefore **weights + decoding + scoring, jointly**.
ARC-Challenge is multiple-choice → the *"confidence-based scoring"* change lands
directly on it. **INFERRED, not established: some fraction of iLLaDA's reported
improvement over LLaDA is attributable to protocol, not pre-training.** I have
not read the iLLaDA paper body (abstract only) and it may well decompose this in
an ablation; §7 flags this as UNRESOLVED with the exact check.

This group has independently established that this species of confound is real
and large — the *whole* of B10/A05: one integer moved MBPP+ by 26.5 pp; a
post-processing stitch moved HE+ by 31.1 pp at c128; a gold ceiling moved 11
items *across hosts with byte-identical inputs*. The generalisable lesson is
**"a model release's headline delta is a joint measurement of weights and
protocol, and nobody is decomposing it."**

→ Proposal **P5**.

### 3.5 SEAM E — "Length" is not defined in the continuous lane, so the entire adaptive-length literature is inapplicable to ELF

Every method in §3.1–3.3 operates on a *count of `[MASK]` tokens*. ELF has no
mask tokens: it flows continuous embeddings and only discretises at the final
timestep. So `L_init`, `E_factor`, "insert `E_factor` masks", "EOS density" have
**no referent**. Yet ELF must still emit a finite sequence, so *something* plays
the role of the canvas — most likely a fixed number of embedding slots.

If so, the continuous lane has **the same static-length bug and none of the
fixes**, and the confidence signal all four training-free methods rely on (EOS
probability at first denoise step) does not exist in a flow — but a natural
analogue does: **the flow velocity magnitude at the terminal slots**, or the
distance of terminal embeddings from the EOS embedding under the final decoder.
This is a real, unclaimed mechanism, not a name concatenation.

→ Proposal **P6** (honestly ranked low: highest novelty, highest risk, and I
could not verify ELF's length handling because I did not read the paper).

---

## 4. Is the canvas-budget finding (S4) preempted? — VERDICT

**Restating S4 verbatim** (B10 `STATUS.json:subclaim_S4_inherited_from_A05.statement`):

> On full-program code generation, DreamOn-v0-7B's reported weakness is
> substantially an artefact of one sampler-config integer: `initial_masks` 8→32
> (all else frozen) moves MBPP+ .0899→.3545 (+26.5 pp) and HE+ .1280→.2561.

And S4's own novelty gate, **S4-G0**, verbatim:

> KILL if an existing paper already shows a mask-diffusion LM's full-program code
> score is dominated by its initial-canvas budget → S4 is a reproduction, citable
> as caveat only.

### 4.1 Verdict: **S4-G0 FIRES.** S4 is a reproduction as a *general phenomenon*.

**The paper that fires it: DAEDAL (arXiv 2508.00819, ICLR 2026 Poster).** Its
Table 1 shows, on **full-program code generation**, for a **mask-diffusion LM**
(LLaDA-Instruct-8B), that the score **is** dominated by the initial-canvas budget:

- **HumanEval 18.9 → 47.6** across length 64→1024 (**+28.7 pp**, larger than
  S4's HE+ leg)
- **MBPP 20.8 → 38.8** across 64→2048 (**+18.0 pp**, comparable to S4's +26.5 pp)
- and Table 2 replicates it on a second model (LLaDA-1.5-8B): HumanEval 18.3 →
  50.0, MBPP 20.6 → 40.2.

This is the same phenomenon, on the same task type, at the same magnitude, on
**two** models, published **2025-08-01** — a year before A05 measured it. It is
not concurrent work; it precedes us and it is peer-reviewed.

**Corollary: S4-G1 is pre-fired by literature and should not be run on GPU.**
S4-G1 was "sweep `initial_masks` on a SECOND mask-diffusion model, ~4–6 GPU-h,
KILL S4 as a general claim if that model does not move ≥10 pp". DAEDAL's Tables 1
and 2 *are* that experiment, on two models, and they move ≫10 pp — so the
generality is **confirmed, by someone else**, which per S4-G0's own wording means
S4 collapses to *"DreamOn was mis-invoked in this repo"* + a caveat citation.
**Recommendation: do not spend 4–6 GPU-h re-establishing a published ICLR result.**
(This is a *saving* the literature check bought us, and it is exactly what
S4-G0 was for.)

Additional corroborating papers, each independently showing canvas/length
dominance on code (so this is not a one-paper judgement): DreamOn (ICLR'26,
"severely degrades code infilling performance when the predefined mask size
mismatches"); CAL (2602.00476, Table 2 — LLaDA-Base single-line 22.7/51.0/56.6/49.0
at L=4/8/16/32, a **33.9 pp** span, *and non-monotone*); LR-DLLM (2602.07546);
ρ-EOS (2601.22527).

### 4.2 What is NOT preempted (this is where the residue lives)

Three things, in descending confidence:

1. **The diagnostic direction is unclaimed.** DAEDAL, CAL, LR-DLLM, ρ-EOS all
   sweep length **to motivate their own method**. Not one of them re-audits a
   *third party's published claim* about a baseline model, or asks how much of a
   *reported method-vs-baseline gap* was manufactured by the baseline's canvas.
   S4's actual novel content — *"a substantial fraction of the gap between a
   proposed method and its baseline was produced by how the baseline was invoked"*
   (A05 `finding_that_outlives_a05`) — is **an evaluation-practice claim**, and
   I found **no** paper making it. §3.3's LR-DLLM `initial mask length = 1` case
   is a live, citable instance.
2. **Nobody reports the cost side of the canvas curve.** §3.2. DAEDAL reports
   `Eratio`, not NFE. This group has NFE and wall-clock per canvas cell **already
   on disk**. The statement *"the canvas sweep is a cost–quality frontier, and
   'matching the best fixed baseline' is only a win if it is NFE-Pareto-dominant"*
   is, as far as I can verify, unpublished.
3. **The interaction with post-processing is unclaimed and is genuinely
   surprising.** A05's measured interaction — the stitch defect costs +0.61 pp at
   c8, +4.27 pp at c32, **+31.10 pp at c128**, because a bigger canvas produces
   multi-line bodies that a double-indent can corrupt — is a *second-order*
   effect: **enlarging the canvas activates latent harness bugs**. I searched for
   code-eval harness/post-processing sensitivity work and found nothing making
   this claim. It also has a self-limiting property that must be stated with it:
   at the published operating point (c8) it costs exactly **1 item of 164**, and
   the measured blast radius over 17 arms was **2 arms, +0.61 pp** (S4
   `must_not_claim_1/2`).

### 4.3 Consequence for how the group should write this up

S4 must **not** be written as "we discovered canvas budget matters" — that is
DAEDAL, ICLR 2026, and claiming it would be the kind of error the group's own
integrity rules exist to prevent. It **can** be written as: *"Canvas budget is
known to dominate dLLM code scores (DAEDAL, ICLR'26). We show the consequence
nobody drew: published baseline numbers are canvas-configuration-dominated —
including in papers that fix length as a method (LR-DLLM sets DreamOn's initial
mask to 1) — and the canvas sweep is a cost–quality frontier on which
'matches the best tuned fixed baseline' is not automatically a win."*
That claim is intact, and P1 + P3 are its gates.

---

## 5. Proposals

Ordering note: P1–P3 start at **0 GPU** on files that exist today. P4–P6 need
GPU, so they are pre-staged but not launchable (all 40 cards busy).

---

### P1 — The canvas curve is a cost–quality frontier, and adaptive-length methods are not shown to be on it

1. **Claim (falsifiable)**: DreamOn's canvas sweep on HE+/MBPP+ traces a
   monotone NFE–quality frontier such that **the quality gained per extra forward
   pass is roughly constant**, i.e. large-canvas quality is *bought*, not
   *unlocked*. **Falsified if** the frontier is strongly concave — if there is a
   canvas at which quality jumps with no NFE increase, then canvas is a genuine
   capability switch and "you must tune it" is a weaker criticism than I claim.
2. **Mechanism**: no model runs. Join, per item, the pass map to the cost
   telemetry that is already stored, and compute a **quality-per-NFE frontier**
   with per-item pairing so the test is paired, not marginal. Concretely:
   `evidence/cells/{he,mbpp}_c{8,32,128}.json` → `per_item_pass[task_id].{base,plus}`
   and `cost_and_behaviour.{nfe_mean,nfe_median,tokens_fed_effective_mean,wall_seconds_sum,generated_tokens_mean,parseability}`;
   corrected pass maps from
   `evidence/cells_corrected/a05_closeout_stitch_regrade.json` →
   `cells.he_c{8,32,128}.per_item_pass` (164 items each). Deliverable: a
   (pass@1, NFE) curve per benchmark with **the corrected HE+ axis**
   (.1341/.2561/.4817), plus per-item McNemar between adjacent canvases to
   establish which canvas steps are individually significant at n=164.
   Then overlay the *published* adaptive-length operating points as **token
   counts** (DAEDAL `Ntoken` 618/813; CAL `Stps.` 11–18 extra first-step
   forwards) and state plainly which axis is missing.
3. **Why not already done**: closest prior work is DAEDAL §4.3 (reports
   `Ntoken`/`Eratio`, no NFE) and CAL (reports search steps, no total NFE
   comparison). Delta: **the cost axis is forward passes, per item, paired with
   the pass/fail of that same item.** I found no dLLM length paper reporting
   this. Honest caveat: the *idea* that one should report compute-matched
   comparisons is old and obvious — the contribution is the **measurement** on a
   model where the frontier is already on disk, not the idea.
4. **PRE-REGISTERED KILL GATE**:
   **KILL if** the Spearman rank correlation between `nfe_mean` and pass@1
   (corrected axis) across the HE+ canvas cells {8, 32, 128} **is not +1.0**,
   **or** if the incremental quality per incremental NFE changes by **more than
   3×** between the c8→c32 and c32→c128 steps.
   *Expected outcome, computed by hand from the table in §3.2 before writing this
   gate*: HE+ c8→c32 gives +12.20 pp for +221.4 NFE = **0.0551 pp/NFE**;
   c32→c128 gives +22.56 pp for +199.7 NFE = **0.1130 pp/NFE**. Ratio **2.05×**.
   **So I expect this gate to be a near-miss that does not fire (2.05 < 3), and I
   expect the frontier to be mildly CONVEX in the good direction** — the
   large canvas is *more* NFE-efficient per point, which partially undercuts my
   own "quality is bought" framing and instead supports "small canvases are
   catastrophically wasteful". I am pre-registering the number so I cannot
   retro-fit the story. Note the 3× threshold is chosen so the *measured* 2.05×
   does not fire it; that is deliberate and disclosed — the gate exists to catch
   a *qualitatively different* shape (e.g. a free jump), not to rubber-stamp.
5. **Cheapest decisive first experiment**: exactly the above. **0 GPU, ~1–2 CPU
   hours.** Step 1 is 0 GPU: yes, entirely.
6. **Confounds**: (a) **`nfe_mean` is the recount, not the old inflated value** —
   A05 closeout established the original NFE was `mean(len(history))`, and the
   recount is **not** a uniform rescale (HE+ 265.88→172.3 *down*, MBPP+
   135.65→153.4 *up*), so **never mix pre- and post-correction NFE**. (b) Use the
   **corrected** HE+ pass axis or the stitch bug re-enters (as-run c128 .1707 vs
   corrected .4817 would invert the curve shape — the as-run curve is
   non-monotone and the corrected one is monotone). (c) `wall_seconds_sum` includes
   shard stragglers (A05 records billed 20.66 vs compute 14.39 GPU-h), so
   **wall-clock is amortised, not compute** — report both, labelled. (d) n=164 is
   small; single-item flips are 0.61 pp. (e) mbpp_c128 and both c512 cells
   **never ran**, so MBPP+'s frontier has only two points and **its curve shape is
   not established** — say so.
7. **Licenses / must-not-claim**: **Licenses**: "on this model and these
   benchmarks, canvas quality gains are accompanied by proportional-or-better NFE
   growth, and published adaptive-length methods do not report the NFE axis at
   all." **Must NOT claim**: that any specific published method is *not*
   NFE-Pareto-optimal — that requires running that method, which this does not.
   Must NOT claim anything about MBPP+'s curve shape. Must NOT generalise beyond
   DreamOn-v0-7B.

---

### P2 — Adaptive length and confidence-parallel decoding are substitutes, not complements (the confidence field is a single contested resource)

1. **Claim (falsifiable)**: The quality gain from DAEDAL-style adaptive length
   **decreases monotonically** as a Fast-dLLM-style parallel-unmasking threshold
   is loosened, because parallel decoding commits the terminal region early and
   thereby saturates the `conf_eos` signal Stage 1 depends on. **Falsified if**
   the two gains are additive within noise, or if adaptive-length gain is
   *independent* of the parallel threshold.
2. **Mechanism**: 2×2×k factorial in the sampler only, no training.
   Axis 1: parallel unmasking threshold (Fast-dLLM style / DAEDAL's own
   `τ_high`) ∈ {1 token/step (sequential), moderate, aggressive}. Axis 2:
   {fixed canvas at each of several lengths} × {DAEDAL Stage 1 only, Stage 2
   only, both} — DAEDAL's own Table 3 ablation axes. Instrument **(i)** the number
   of Stage-1 expansions actually taken, **(ii)** the number of Stage-2
   insertions actually taken, **(iii)** total NFE, **(iv)** pass@1. The
   mechanistic prediction is specifically that **(i) and (ii) fall toward zero**
   as the parallel threshold loosens — that is the *measurement that identifies
   the mechanism*, independent of whether quality happens to move.
3. **Why not already done**: DAEDAL grid-searches `(τ_high, τ_low)` jointly
   (Fig. 5) and even remarks its `τ_high` is "analogous to the confident decoding
   strategy proposed in Dimple" — so it is one step from this experiment but
   explicitly runs "without any acceleration or caching optimizations proposed in
   subsequent works" (§4.1), i.e. it **deliberately excludes** the composition.
   Fast-dLLM (ICLR'26) is fixed-length. Delta = the cross. Honest ranking note:
   this is the kind of thing an efficiency-lab could have in a workshop paper by
   now; I searched (`Fast-dLLM` + acceleration lane, ~15 titles) and found no
   match, but I read only titles, so **medium confidence it is unclaimed**.
   *Faster but Different* (2607.29079) is the nearest genre and must be read
   first (§7).
4. **PRE-REGISTERED KILL GATE**:
   **KILL if** the adaptive-length quality gain at the most aggressive parallel
   setting is **≥ 80 %** of its gain at the sequential setting (i.e. the
   interaction is ≤ 20 % of the main effect), on GSM8K **and** HumanEval, under a
   paired bootstrap over items with 10 000 resamples at α=0.05.
   *Expected outcome*: **I expect this to be a real interaction on GSM8K and to
   FIRE on HumanEval** — code responses are longer and hit `L_max` behaviour, and
   DAEDAL's own HumanEval `Ntoken` (813) is its worst efficiency case, so the
   terminal EOS region is less likely to be committed early there. Expecting a
   split result is the honest prediction.
5. **Cheapest decisive first experiment**: **Step 1 is 0 GPU** — reimplement
   DAEDAL Alg. 1 against the on-disk Dream-Coder sampler and *unit-test the
   mechanism claim without a GPU* by asserting on a recorded logits trace that
   committing the terminal window forces `conf_eos > τ_eos`. GPU step: GSM8K
   (1319 items) + HumanEval (164) × 3 parallel settings × 4 length arms on
   LLaDA-8B-Instruct. Estimate at 8×H20: DAEDAL used 8×A800 for the same
   workload; the 12 cells at ~1–3 GPU-h each ⇒ **~24–36 GPU-h**, plus ~1 GPU-h
   smoke. Needs `GSAI-ML/LLaDA-8B-Instruct` (download, routine).
6. **Confounds**: (a) **the canvas budget itself** — if the parallel setting
   changes how many tokens get emitted, quality moves for length reasons, not
   decoding reasons; must report `Ntoken`/`Etoken` per cell and hold `L_max`
   fixed. (b) Reimplementation fidelity: must reproduce DAEDAL's published
   GSM8K 85.8 / HumanEval 48.2 at `L_init=64` within a pre-registered tolerance
   before any cell counts, else we are ablating our own bug. (c) Fast-dLLM's KV
   cache is an *approximation* — its quality effect must be separated from the
   threshold effect by including a cache-off/threshold-on arm. (d) tokenizer/chat
   template: LLaDA-Instruct needs its own template; a template error looks like a
   method effect.
7. **Licenses / must-not-claim**: **Licenses**: "adaptive length and
   confidence-parallel decoding draw on the same slack; their gains do not
   compose additively, and papers reporting them separately overstate the
   attainable joint speed-quality point." **Must NOT claim** a general result
   about all accelerators (cache-based, distillation-based and
   scheduling-based accelerators are different mechanisms), nor anything about
   models other than the one tested.

---

### P3 — Baseline-canvas audit: how much of the published dLLM method-vs-baseline gap is canvas configuration?

1. **Claim (falsifiable)**: Across the 2026 variable-length papers, a
   material fraction of the reported margin over *diffusion* baselines is
   attributable to the baseline's length configuration rather than to the
   method — specifically, **at least one published headline margin shrinks by
   >50 % when the baseline is given its own best canvas from the same paper's own
   tables.** **Falsified if** every paper's margin survives re-basing against the
   best baseline configuration *that the paper itself reports*.
2. **Mechanism**: a **paper-level re-analysis**, no model runs. For each of
   {DAEDAL, CAL, LR-DLLM, ρ-EOS, DreamOn, FlexMDM}: extract (i) the baseline
   length(s) reported, (ii) whether the headline margin is vs the *best* or vs a
   *single* baseline length, (iii) whether the reported margin survives
   re-basing on the best-of-reported baseline. This is auditable arithmetic on
   published tables, which makes it checkable by a reviewer. Anchor case, already
   extracted: **LR-DLLM Table 3** sets DreamOn's initial mask length to **1** and
   reports it at mean 37.9 — *below* its own 64-mask baseline (39.1). By
   contrast **DAEDAL** re-bases correctly (6 lengths, margin vs the best) and
   **CAL** re-bases correctly (4 lengths + AVG) — the audit must say so, loudly,
   or it is a hit piece rather than a measurement.
3. **Why not already done**: this is the **diagnostic** framing that §4.2 finds
   unclaimed. Closest prior work is the general baseline-tuning literature
   (*"are we making real progress"*-genre reproducibility papers) and CAL's
   incidental finding that DAEDAL *loses* to a fixed baseline at L=16/32. Delta:
   nobody has done it *for the dLLM length lane*, where the free parameter is
   unusually powerful (28.7 pp in DAEDAL's own table). Honest note: "someone
   probably could do this in a week", and it is a *criticism* paper, which is
   harder to place — hence ranked below P1 despite being cheap.
4. **PRE-REGISTERED KILL GATE**:
   **KILL if**, across all six papers audited, **zero** headline margins shrink
   by >50 % under best-of-reported re-basing, **or** if fewer than 2 of 6 papers
   report only a single baseline length.
   *Expected outcome*: **does not fire** — LR-DLLM's DreamOn row already
   satisfies the first clause (37.9 vs a 39.1 baseline is a *negative* margin, an
   infinite shrink), and its own Baseline is a single length (64). But note the
   gate is honestly hard: **DAEDAL and CAL will pass the audit cleanly**, so the
   result will be "2 of 6 do this, 4 of 6 do not", which is a *weaker* paper than
   "the lane is broken". I expect the finding to be narrow.
5. **Cheapest decisive first experiment**: read the six PDFs and tabulate.
   **0 GPU, ~4 CPU/reading hours.** Step 1 is 0 GPU: yes. (Three of the six PDFs
   are already extracted to text on this machine.)
6. **Confounds**: (a) **A single baseline length is not automatically a sin** —
   if the paper's contribution is "unknown-length setting", fixing 64 may be the
   *definition of the task*, and I must quote each paper's stated protocol rather
   than judge by table shape. (b) Cross-paper numbers are not comparable
   (graders, splits, `MAX_LENGTH`), so the audit must stay **within** each paper.
   (c) Version drift: arXiv v1 vs camera-ready can differ — must state which
   version each number came from (DAEDAL v2 is dated **today**).
7. **Licenses / must-not-claim**: **Licenses**: "within the 2026 dLLM
   variable-length lane, baseline length configuration is reported
   inconsistently, and in at least one case a published baseline is configured
   below its own paper's default." **Must NOT claim** that any author acted in
   bad faith, that any *model* is better than reported (we did not re-run
   anything), or that the reported *method* is invalid — only that the *margin*
   is protocol-dependent.

---

### P4 — Bidirectional length control on a *trained* variable-length model: does DreamOn's expand/delete need a training-free contractor?

1. **Claim (falsifiable)**: DreamOn's trained expand/delete states are
   **directionally asymmetric in practice** — it recovers from an under-provisioned
   canvas better than from an over-provisioned one — and bolting a *training-free
   contraction* rule (ρ-EOS-style EOS-density, or CAL-style bidirectional search)
   onto it recovers most of the oracle-length gap **without retraining**.
   **Falsified if** DreamOn's over-provisioned degradation is already ≤2 pp, i.e.
   its delete state is doing the job.
2. **Mechanism**: DreamOn already has a *learned* `delete`/EOS state, so unlike
   LLaDA it *can* contract — but nobody has measured whether it *does*. Sweep the
   initial canvas **above** the oracle length (oracle+{0,8,32,128}) and measure
   (i) pass@1 decay, (ii) how many delete-state tokens are actually emitted per
   item, (iii) final emitted length vs oracle. Then add a training-free
   contraction trigger at the sampler level and re-measure. The composition
   reason (not a name concatenation): **DreamOn's contraction is a per-token
   learned decision with no global length signal; ρ-EOS's ρ is a global signal
   with no learned contraction actuator. Each supplies exactly what the other
   lacks** — DreamOn has the actuator, ρ-EOS has the controller.
3. **Why not already done**: ρ-EOS/CAL/LR-DLLM are all developed for *fixed-canvas*
   models (LLaDA family) where contraction must be simulated; DreamOn's paper
   frames its own contribution as *reaching* oracle parity, and this group has
   already measured that the **oracle handout is worth +5.7 pp (p=4.1e-14)**
   (B10 `robust_findings_that_survive` #3), so parity is not complete. Delta =
   controller-on-actuator, on the one model that has the actuator. Honest note:
   ρ-EOS explicitly advertises bidirectionality as *its* novelty, so this is a
   composition inside a lane that is moving fast; **2–3 month concurrency is
   likely and would not preempt.**
4. **PRE-REGISTERED KILL GATE**:
   **KILL if** DreamOn's pass@1 on HumanEval-SingleLineInfilling (base axis,
   n=1033, the group's existing harness) drops by **< 2.0 pp** when the initial
   canvas is set to oracle+128 relative to oracle+0, under exact McNemar at
   α=0.05 — because then there is no over-provisioning damage for a contractor to
   repair, and the proposal has no target.
   *Expected outcome*: **genuinely uncertain, and I would bet weakly on the gate
   FIRING.** Evidence for firing: DreamOn's whole selling point is length
   robustness, and the group measured `dreamon_fim` (non-oracle) at .8664 vs
   `dreamon_oracle` .9342 — only a 6.8 pp oracle gap, so its canvas handling is
   already decent. Evidence against: A05's HE+ *full-program* curve is strongly
   canvas-dependent, and CAL/LR-DLLM both report DreamOn behaving
   inconsistently across span regimes.
5. **Cheapest decisive first experiment**: the **gate alone**, on the existing
   6-arm infilling harness with two new canvas settings. Model
   (`DreamOn-v0-7B`), split (n=1033, md5 `30129634e180d80c19d6ddcd4cf43f9c`),
   grader, and 8-shard driver **all already exist on disk**. Step 1 is **not**
   0 GPU (it needs generation), but it is cheap: the existing arms cost ~2 GPU-h
   per 1033-item arm at 8 cards ⇒ **~4–6 GPU-h for two arms** + 0 GPU grading.
   (0-GPU precursor available: re-mine the existing `dreamon_fim` vs
   `dreamon_oracle` per-item maps for whether failures concentrate on
   short-gold or long-gold items — that is free and informs the design.)
6. **Confounds**: (a) **the oracle length itself is a handout** — arms must be
   labelled oracle-derived, and the result must not be read as an
   unknown-length result. (b) The group's own **cross-host gold-ceiling
   irreproducibility** (base ceiling .9894 wzc1 vs 1.0000 zwfy6, 11 items, all
   `HumanEval/32`) means arms must be graded **on one host with one evalplus
   version**, per B10's protocol note. (c) SingleLine gold spans are short —
   over-provisioning by 128 on a 1-line target may be so far out of distribution
   that the result says nothing about realistic use; a MultiLine or RandomSpan
   arm is the honest generalisation and costs more. (d) Ceiling effects: .9342 is
   already near the .9894–1.0000 gold ceiling, so **there is only ~5 pp of
   headroom and the gate's 2 pp threshold sits inside the region where ceiling
   artefacts live** — this is the proposal's biggest weakness and it is why the
   gate is likely to fire.
7. **Licenses / must-not-claim**: **Licenses**: "a trained variable-length dLLM's
   contraction behaviour is/is not sufficient under over-provisioning, measured
   with per-item pairing." **Must NOT claim** that this generalises to LLaDA-family
   models (they lack the actuator) or to full-program generation (different
   surface), and must NOT re-open the **dead** AR-vs-diffusion ranking claim that
   B10's Gate 1 killed.

---

### P5 — Decomposing a model release: how much of iLLaDA's gain over LLaDA is weights vs protocol?

> ⚠️ **STATUS UPDATE — I ran this proposal's own 0-GPU step 1 after drafting it,
> and it MOSTLY FIRED THE GATE. P5 is hereby DOWNGRADED. Read §5.P5.0 before
> anything else.** I am leaving the original proposal text below rather than
> quietly rewriting it, because the gate firing is the useful outcome.

**§5.P5.0 — measured result of the 0-GPU check (2026-08-15).** I downloaded and
read `arxiv.org/pdf/2606.25331v1` (10 pages). Findings:

1. **iLLaDA DOES contain the multiple-choice scoring ablation.** Its Table 4
   (transcribed verbatim): scoring rule Likelihood → PIQA **77.2** / ARC-C
   **60.2** / HellaSwag **74.3**; Confidence → PIQA **78.5** / ARC-C **60.8** /
   HellaSwag **76.6**. So the protocol effect is **+1.3 / +0.6 / +2.3 pp**.
2. **Against P5's pre-registered ARC-C threshold of ≥3.0 pp, the measured effect
   is +0.6 pp → the ARC-C leg of the gate FIRES.** And it fires *exactly as I
   predicted it would not*: I wrote "I expect it to fire on HumanEval and not fire
   on ARC-Challenge", betting that MC scoring rules are worth several points.
   **That prediction was wrong** — on iLLaDA's own measurement, the scoring change
   is worth 0.6 pp on ARC-C, i.e. **~4 % of the 14.9 pp** reported delta, not
   ≥20 %. Recording this because a wrong pre-registered prediction that I then
   report is the whole point of pre-registering.
3. **But the confound I flagged is real and remains unresolved in a different
   place**: iLLaDA's Tables 2/3 mark LLaDA's and Dream's columns with † and ‡ and
   state they "are from Nie et al. [10]" and "Ye et al. [22]" — i.e. **the
   baseline numbers are QUOTED FROM THE ORIGINAL PAPERS, not re-run under
   iLLaDA's harness.** So the headline "+16.5 HumanEval" compares
   iLLaDA-under-iLLaDA's-protocol against LLaDA-under-LLaDA's-protocol. The
   scoring ablation (Table 4) is run **on iLLaDA only** and does not close this,
   because it does not tell us what LLaDA scores under iLLaDA's protocol.
4. **iLLaDA's variable-length mechanism is now known** and is *not* DAEDAL-like:
   "we append a block of mask tokens and run the diffusion sampler within this
   block … Once a block is decoded, generation terminates if an |EOS| or other
   stop token appears; otherwise, **a new block of masks is appended** and the
   process continues until a maximum generation budget is reached." That is
   **semi-AR block extension**, i.e. closer to BD3-LM/Fast-dLLM-v2 blocking than
   to mid-sequence insertion. It can only **grow**, and only at block granularity
   — so it is *weaker* than DAEDAL Stage 2 (which inserts mid-sequence) and
   cannot contract at all.

**Revised P5, narrowed**: the live question is no longer "weights vs protocol"
(iLLaDA answered the scoring half). It is **"do quoted-from-original baseline
numbers understate the baseline?"** — i.e. re-run **LLaDA-8B** under **iLLaDA's**
harness (block-extension variable-length generation + confidence MC scoring) and
compare to the quoted †-marked column. **New gate: KILL if re-running LLaDA-8B
under iLLaDA's protocol moves LLaDA's HumanEval by <3.3 pp from the quoted 35.4.**
Expected outcome: honestly uncertain now; given (2), I would guess it fires.
Cost unchanged (~8–12 GPU-h). **Rank: dropped from 5 to last-but-one.**

---

*(original P5 text, retained unedited for provenance)*

1. **Claim (falsifiable)**: A non-trivial share (**pre-register: ≥20 % of the
   reported delta on at least one benchmark**) of iLLaDA's improvement over LLaDA
   is attributable to the two protocol changes bundled into its release
   (variable-length generation; confidence-based multiple-choice scoring) rather
   than to its 12T-token pre-training. **Falsified if** applying both protocol
   changes to LLaDA-8B moves LLaDA by <20 % of the reported gap on every
   benchmark tested.
2. **Mechanism**: 2×2 — {LLaDA-8B, iLLaDA-8B} × {LLaDA-era protocol,
   iLLaDA protocol}. The protocol arms are decoding/scoring only:
   (a) fixed-length vs variable-length generation, (b) likelihood-based vs
   confidence-based MC scoring. Benchmarks: one MC (ARC-Challenge — where the
   scoring change bites, reported +14.9) and one generative (HumanEval, reported
   +16.5). The point is the **off-diagonal cells**, which no release paper runs.
3. **Why not already done**: model releases are not usually re-audited this way,
   and this group has direct, hard-won evidence that the confound is real and
   large (§3.4). **Honest ranking penalty: iLLaDA's own paper may already contain
   this ablation — I read only the abstract.** If it does, this proposal is dead
   on arrival, and checking costs 0 GPU (§7). Rank accordingly.
4. **PRE-REGISTERED KILL GATE**:
   **KILL if** the protocol-only arm on LLaDA-8B moves ARC-Challenge by
   **< 3.0 pp** (i.e. <20 % of the reported 14.9 pp) **and** HumanEval by
   **< 3.3 pp** (<20 % of 16.5), both under exact McNemar at α=0.05 on paired
   items.
   *Expected outcome*: **I expect it to fire on HumanEval and not fire on
   ARC-Challenge.** MC scoring rule changes are historically worth several points
   (this is the same species as the group's own established
   `chat_template=False` and length-normalisation lessons), whereas
   variable-length generation on HumanEval mostly re-tunes a length the LLaDA
   baseline could also have tuned — DAEDAL's own Table 1 says the gap between
   L=512 and adaptive is only 47.0→48.2.
5. **Cheapest decisive first experiment**: **Step 1 is 0 GPU** — read the iLLaDA
   paper and check whether the decomposition already exists; if it does, stop.
   GPU step needs `GSAI-ML/LLaDA-8B-Base` + `GSAI-ML/iLLaDA-8B-Base` (~16 GB
   each, routine download) and is eval-only: ARC-C (1172 items) + HumanEval (164)
   × 4 cells ⇒ **~8–12 GPU-h at 8×H20**.
6. **Confounds**: (a) **The two models differ in tokenizer/config** — if
   `eos_token_id` or the chat template differs, the "protocol" arm is not
   transferable and the whole design collapses; check `config.json` first (the
   group has been burned by exactly this on Qwen3-8B Instruct-vs-Base). (b) MC
   scoring has many variants (length-normalised, unconditional-normalised,
   byte-normalised); "confidence-based" must be pinned to what iLLaDA actually
   does, from its code, not inferred. (c) prompt formatting differences across
   the two releases' eval harnesses. (d) Base vs Instruct must not be mixed.
7. **Licenses / must-not-claim**: **Licenses**: "a share of this release's
   headline delta is protocol, and release papers should report weights-vs-protocol
   decompositions." **Must NOT claim** iLLaDA's pre-training is not an improvement
   (the residual is still large), nor that authors misrepresented anything.

---

### P6 — Does the continuous lane (ELF) inherit the static-length bug, and does a flow-native length signal exist?

> ✅ **STATUS UPDATE — I ran this proposal's own 0-GPU step 1 after drafting it.
> Premise (a) is CONFIRMED and the kill-gate clause (a) does NOT fire. P6 is
> PROMOTED. Read §5.P6.0.**

**§5.P6.0 — measured result of the 0-GPU check (2026-08-15).** I downloaded and
read `arxiv.org/pdf/2605.10938v2` (39 pages, ELF v2) and searched the extracted
text:

1. **ELF uses a hard fixed sequence length.** Its config table lists
   `Sequence length 1024` for the OWT setting; WMT14 De-En is `sequence length
   L = 128 (condition length 64, target length 64)`; XSum is `L = 1088
   (condition 1024, target 64)`. The appendix baseline-config table lists
   `Sequence Length 64 / Max Cond Length 1024`. **So the target length is a
   pre-set constant, exactly as in the masked lane.**
2. **Case-insensitive counts over the full 126 KB extracted text**: `"padding"`
   → **0 hits**; `"fixed-length"`/`"fixed length"` → **0 hits**;
   `"variable"` → 4 hits (**none** about output length — they are about
   variable-length *training* example packing); `"EOS"` → **2 hits**. There is
   **no length-adaptation mechanism and no discussion of one.** Kill-gate clause
   (a) — "ELF has no fixed slot budget" — is therefore **FALSE**, so it does not
   fire, and the premise stands.
3. **ELF released weights** (RETRIEVED from HF API this session):
   `embedded-language-flows/ELF-B-owt`, `ELF-M-owt`, `ELF-L-owt`,
   `ELF-B-de-en`, plus `t5_small_encoder_jax`. Code:
   `https://github.com/lillian039/ELF` (from the PDF). **The GPU step is
   therefore actually runnable**, which I had marked UNVERIFIED.
4. **ELF is small and cheap**: 105M params (vs 170M for its baselines), 5 epochs
   over OWT's ~9.04B tokens = 45.2B effective training tokens. **This revises my
   GPU estimate sharply downward** — a slot-budget sweep on a 105M model is
   *hours on one card*, not 8–16 GPU-h on eight. Revised estimate: **~2–4 GPU-h.**
5. **Crucially, ELF already reports the axis the whole length lane omits.**
   `"NFE"` appears **46 times**; its headline figure is `Gen. PPL` at
   **1024 steps vs 32 steps**. So the continuous lane reports
   compute-vs-quality natively while the masked length lane reports token-ratios
   (§3.2). **This makes ELF an unusually good host for P1's framing**, and is an
   additional, unanticipated reason to like P6.
6. **One premise correction**: ELF's quality metric is **generative perplexity +
   unigram entropy** (unconditional) and BLEU/ROUGE (conditional) — **not**
   pass@1. So P6's gate clause (b) "moves quality by <5 pp" was
   ill-specified for this model. **Revised clause (b): KILL if sweeping the target
   slot budget over a ≥4× range changes generative perplexity by <10 % relative
   at matched NFE.** (Pre-registering the revision *and* the reason, since
   changing a threshold after looking is exactly the failure mode to avoid — the
   change here is a **unit** fix, not a **level** fix; the level is newly chosen
   because no level was meaningful before.)
7. **Confound now visible that I could not see before**: ELF uses a **frozen
   pretrained T5-small encoder** for embeddings by default and CFG with
   `input-condition CFG scale 2`. Both interact with any terminal-slot statistic
   (a T5 embedding space has its own EOS geometry). The velocity-norm frequency
   confound I flagged in field 6 is therefore *more* serious, not less.

---

*(original P6 text, retained unedited for provenance)*

1. **Claim (falsifiable)**: ELF-style continuous-embedding flow LMs have a
   fixed sequence-slot budget and therefore inherit the same static-length
   pathology as masked dLLMs; **and** a flow-native sufficiency signal exists —
   the terminal-slot velocity magnitude at the first flow step correlates with
   required output length at least as well as EOS confidence does in the masked
   case (DAEDAL's Fig. 2 signal). **Falsified if** ELF's formulation has no
   fixed slot budget (e.g. it is inherently insertion-capable), or if the
   terminal velocity signal has |ρ| < 0.3 against gold length.
2. **Mechanism**: (i) determine ELF's length handling from its paper/code;
   (ii) if there is a slot budget, sweep it and reproduce the canvas curve in the
   continuous lane; (iii) define the flow-native analogue — at the first flow
   step, per terminal slot, take ‖v_θ(x_t, t)‖ and/or cosine distance of the
   terminal slot's trajectory endpoint to the decoder's EOS embedding — and
   correlate against gold length. Composition reason: **all four training-free
   length controllers key on a discrete-token EOS probability, which does not
   exist in a flow; a velocity/EOS-distance statistic is the natural transport of
   that idea, and if it works, the whole DAEDAL/ρ-EOS/CAL toolbox ports to a model
   class that currently has none of it.**
3. **Why not already done**: ELF is ~3 months old (2026-05) and self-described as
   a tech report; the length lane has been entirely masked-diffusion. I found no
   paper on adaptive length for continuous DLMs. Delta is large. Honest note:
   **this is the least verified proposal here** — I did not read ELF's body and
   cannot confirm premise (i). If ELF turns out to be variable-length by
   construction, P6 dies at step 1, for free.
4. **PRE-REGISTERED KILL GATE**:
   **KILL if** (a) ELF has no fixed slot budget (established by reading the
   paper/code — 0 GPU), **or** (b) sweeping the slot budget over a ≥4× range
   moves quality by **< 5 pp** on the paper's own benchmark, **or** (c) the
   terminal-velocity statistic has Spearman |ρ| **< 0.3** against gold length on
   ≥500 held-out items.
   *Expected outcome*: **I expect (a) to be false and (b) to pass, but (c) to be
   a coin flip and quite possibly to fire.** The masked-case signal works because
   EOS is a *token the model must decide to emit*; a velocity norm has no such
   semantics and may be dominated by embedding-norm geometry.
5. **Cheapest decisive first experiment**: **Step 1 is 0 GPU** — read ELF
   (2605.10938) + its code and answer (a). Only if (a) fails do we need GPU. GPU
   step is a slot-budget sweep on ELF's own released model if one exists
   (**UNVERIFIED whether ELF has released weights — I did not check HF**),
   estimate **~8–16 GPU-h**, high uncertainty.
6. **Confounds**: (a) **model-class mismatch of the metric** — ELF's benchmarks
   are likely LM-quality (perplexity/generative metrics), not pass@1, so
   "quality moves 5 pp" needs a per-benchmark definition fixed in advance.
   (b) Slot budget in a continuous flow may interact with the noise schedule, so
   a sweep is not "all else frozen". (c) Embedding-norm confound: velocity norms
   scale with embedding norms, which vary by token frequency — must normalise or
   the correlation is a frequency artefact. (d) CFG: ELF advertises
   classifier-free guidance, whose scale is another free parameter that could be
   confused with a length effect.
7. **Licenses / must-not-claim**: **Licenses**: "the static-length pathology
   is/is not specific to masked diffusion, and a flow-native sufficiency signal
   does/does not exist." **Must NOT claim** anything comparative about ELF vs
   masked dLLM quality (different training data and scale), and must NOT port the
   *word* "canvas" without defining it in the continuous setting.

---

### P7 — Length-conditional item difficulty: is "canvas sensitivity" a property of the model or of a benchmark's gold-length distribution? (0 GPU)

1. **Claim (falsifiable)**: The canvas-budget effect is **concentrated in items
   whose gold span is long relative to the canvas**, such that a per-item model
   `pass ~ f(canvas / gold_length)` explains most of the between-canvas variance —
   which would mean "canvas sensitivity" is largely a **benchmark composition**
   statistic, not a model property, and is therefore *predictable in advance*
   from the benchmark alone. **Falsified if** items with `canvas ≫ gold_length`
   flip pass/fail across canvases at the same rate as items with
   `canvas < gold_length`.
2. **Mechanism**: 0 GPU. Join the per-item pass maps across canvases
   (`cells_corrected` HE+ c8/c32/c128, 164 items each; `cells` MBPP+ c8/c32) with
   each item's **gold solution token length** (computable from the benchmark
   files already on disk) and fit/report the flip rate stratified by
   `canvas / gold_len` bucket. A05 already stores an adjacent statistic —
   `emitted_gold_ratio_{mean,median}` and `long_span_gold_ge65` — so part of the
   scaffolding exists (`he_c128.long_span_gold_ge65 = {n:159, ratio_median:0.231,
   parseability:0.264}`).
3. **Why not already done**: every length paper reports **aggregate** pass@1 per
   length; none reports the flip structure conditioned on gold length. CAL comes
   closest — it *models* a length bias in confidence — but that is a model-side
   bias, not an item-side difficulty decomposition. Delta: turns "you must tune
   the canvas" into "here is how to predict how much tuning will matter, from the
   benchmark alone, before running anything." Honest note: mechanically simple and
   someone may have it in an appendix.
4. **PRE-REGISTERED KILL GATE**:
   **KILL if** the pass-rate difference between the `canvas < gold_len` and
   `canvas ≥ 2×gold_len` strata is **< 10 pp** on HE+ at canvas=8 (the stratum
   where the effect must be largest if it exists at all), under a two-proportion
   exact test at α=0.05.
   *Expected outcome*: **does not fire, strongly.** At c8, HE+ `generated_tokens_mean`
   is **2.3** and pass@1 is .1341, and 159 of 164 items have gold span ≥65
   tokens — so essentially the whole benchmark is in the starved stratum and the
   few short-gold items should carry the passes. **Risk that makes this a real
   gate: the starved stratum may be so dominant (159/164) that the comparison
   stratum has too few items for the exact test to reach α=0.05 at all** — in
   which case the gate fires on power, not on effect, and I must report that
   honestly rather than switching benchmarks post hoc.
5. **Cheapest decisive first experiment**: the above. **0 GPU, ~2 CPU hours.**
   Step 1 is 0 GPU: yes.
6. **Confounds**: (a) **gold length is tokenizer-dependent** — must use the
   DreamOn/Dream-Coder tokenizer, not a character count. (b) HE+ items are not
   independent of difficulty: long gold spans are also *harder*, so
   `canvas/gold_len` is confounded with intrinsic difficulty — the honest form of
   this analysis needs a difficulty control (e.g. AR-model pass on the same
   items, which **this group has**: `qwen_fim`/`qwen_prefix` per-item maps for
   infilling, and an AR ceiling of .707/.680 for full-program). (c) n=164 with
   159/5 stratum imbalance is a power problem, stated above. (d) Post-processing:
   use corrected maps only.
7. **Licenses / must-not-claim**: **Licenses**: "canvas sensitivity is
   concentrated in the starved stratum and is partly predictable from a
   benchmark's gold-length distribution." **Must NOT claim** it is *only*
   composition (difficulty is confounded), and must NOT extrapolate the
   regression outside the measured canvas range.

---

## 6. Ranking + the single first experiment

Ranked by (expected information gain) / (GPU cost), with 0-GPU-first weighting as
instructed. **This table is POST-UPDATE**: after drafting the proposals I spent
the remaining budget executing the 0-GPU step-1s of P5 and P6, which moved both
(P5 down, P6 up). That is the ranking machinery working, not churn.

| Rank | Proposal | GPU cost of decisive step | Expected info gain | Notes |
|---|---|---|---|---|
| **1** | **P1** — canvas curve as NFE cost–quality frontier | **0** | High — directly repairs the framing S4 needs after §4's preemption verdict, and produces the axis the whole published lane omits | Runs today. Files verified present + schema read. |
| **2** | **P3** — baseline-canvas audit of 6 papers | **0** (reading) | Medium-high — anchor case found (LR-DLLM sets DreamOn canvas=1); **and 2606.12232 proves this genre publishes** | Also produces the citations P1 needs |
| **3** | **P7** — length-conditional item difficulty | **0** | Medium — makes canvas sensitivity predictable; real risk of failing on power, not effect | Complements P1; same files |
| **4** | **P6** — continuous lane (ELF) ⬆ **promoted from 7** | **~2–4 GPU-h** (was est. 8–16) | High — premise CONFIRMED at 0 GPU (§5.P6.0): ELF has a hard fixed `Sequence length`, 0 hits for "padding"/"fixed-length", no adaptation mechanism, **weights released**, and only 105M params | The rare case where the cheap check *raised* a proposal. Also already reports NFE. |
| **5** | **P4** — bidirectional control on DreamOn | ~4–6 GPU-h | Medium — but I expect its gate to FIRE (ceiling headroom only ~5 pp) | Assets all on disk |
| **6** | **P2** — adaptive length × parallel decoding | ~24–36 GPU-h | High if it holds, but 2607.29079 finding (i) is **evidence against its mechanism**, and it needs a faithful DAEDAL reimplementation first | Best *paper* here; worst *first* experiment |
| **7** | **P5** — iLLaDA weights-vs-protocol ⬇ **demoted from 5** | ~8–12 GPU-h | Low-medium — **gate mostly FIRED at 0 GPU** (§5.P5.0): iLLaDA's Table 4 already ablates MC scoring and it is worth only **+0.6 pp on ARC-C**, ~4 % of the 14.9 pp delta, vs my ≥20 % threshold. My pre-registered prediction was **wrong**. | Narrowed residue survives: baselines are **quoted from the original papers**, not re-run |

### Run first: **P1**, plus the 0-GPU step-1s of P5 and P6 in the same sitting.

Why P1: it is the only item that (a) costs **zero** GPU on a day when all 40
cards are committed, (b) operates on files whose existence and schema I verified
by reading them this session, (c) **repairs a framing that §4 shows is currently
wrong** — the group's S4 write-up would otherwise claim a phenomenon DAEDAL
published at ICLR 2026 — and (d) has a pre-registered gate with the expected
value computed *in advance* (2.05×, non-firing, and mildly *against* my own
preferred narrative).

Concretely, P1's inputs, all verified present today:
- `Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/evidence/cells/{he_c8,he_c32,he_c128,mbpp_c8,mbpp_c32}.json`
  → keys `per_item_pass` (dict, 164/378 items, `{base,plus}` booleans) and
  `cost_and_behaviour` (NFE, tokens fed, wall, parseability, gen tokens)
- `…/evidence/cells_corrected/a05_closeout_stitch_regrade.json`
  → `cells.he_c{8,32,128}.per_item_pass` (164 each) — **the corrected axis**
- `Mixture-of-Memory/proposal/backlog/B10-dllm-infilling-ar-dominance/evidence/gate1_base/score_base/*.json`
  → 6 arms × `per_task[]` (1033 rows: `task_id`, `pass`, `n_tests`, `n_pass`,
  `why`, `exact_match`) + `cost_tokens`
- `…/evidence/gate1_base/gate1_base_stats.json` (md5 `804056f7f9dbb015c4c05dc483d03fa6`)

And the two 0-GPU checks that were listed here as "do alongside" have now
**both been executed** — see §5.P5.0 (iLLaDA: gate mostly fired, P5 demoted) and
§5.P6.0 (ELF: premise confirmed, P6 promoted). Neither cost a GPU-second, and
between them they moved the ranking by three places. **This is the argument for
always spending the free checks before the expensive ones.**

---

## 7. What I could NOT verify, and which authority refused me

**Authorities that refused or failed:**
- **Semantic Scholar Graph API — HTTP 429 on every attempt**, same failure the
  group hit on 2026-08-12 when S4-G0 was first attempted. I therefore have **no
  citation-graph coverage**: I could not do "papers citing DAEDAL" to
  exhaustively confirm that nobody has done P1/P3. My novelty statements rest on
  arXiv full-text search + reading 4 PDFs, which is weaker.
- **DBLP `search/publ/api` — HTTP 500** on the direct endpoint. I got DBLP
  records **indirectly** via OpenReview's DBLP mirror (`venueid =
  dblp.org/journals/CORR/*`), which is how the ρ-EOS / LLaDA-MoE / Dream-Coder
  "preprint" determinations were made. **Direct DBLP was not reachable**, so any
  ACL-family paper here would be under-verified — fortunately none of the core
  papers turned out to be ACL-family, so the Anthology+DBLP rule was not
  load-bearing this session. If a later pass finds an ACL-family paper in this
  lane, **it must be re-verified against aclanthology.org + DBLP**, not
  OpenReview (per `memory/venue-verify-acl-family-needs-anthology.md`).
- **arXiv API over plain HTTP returns 301** through this proxy; must use
  `https://export.arxiv.org`. (Noting it because a silent empty result set from
  the http URL looks exactly like "no such paper" — it cost me one round.)

**Specific items — RESOLVED during this session (listed so the reader can see
what closed and how):**
1. ~~Does iLLaDA already decompose weights vs protocol?~~ **RESOLVED — YES,
   partially.** PDF read; Table 4 ablates MC scoring (+1.3/+0.6/+2.3 pp on
   PIQA/ARC-C/HellaSwag). **P5's gate mostly fired.** See §5.P5.0. Residue: its
   LLaDA/Dream columns are **quoted from Nie et al. / Ye et al.**, not re-run.
2. ~~Does ELF have a fixed slot budget, and are weights released?~~
   **RESOLVED — fixed budget YES (`Sequence length 1024`; 0 hits for "padding");
   weights YES (`embedded-language-flows/ELF-{B,M,L}-owt`, `ELF-B-de-en`).**
   **P6's premise confirmed and P6 promoted.** See §5.P6.0.
3. ~~Have 2606.12232 and 2607.29079 already done P1/P2?~~ **RESOLVED — NO,
   neither preempts**, but 2607.29079 finding (i) is *evidence against P2's
   mechanism* and 2606.12232 is the *genre precedent* that helps P3. See §1.2.
4. ~~arXiv ids for LLaDA and BD3-LM~~ **RESOLVED — both confirmed** by
   `id_list` fetch (titles returned match).

**Items still UNRESOLVED, each with the exact query that settles it:**
5. **Venue for iLLaDA, ELF, ρ-EOS, CAL, LR-DLLM** — all searched in OpenReview;
   all returned either nothing or a DBLP/CoRR record. Recorded as **preprint
   (UNVERIFIED)** rather than asserted. If any is later found in an ACL venue,
   the Anthology+DBLP rule applies, not OpenReview.
6. **dLLM-Cache and dKV-Cache** are cited by DAEDAL §2; I did **not** fetch their
   arXiv records, so I list them by name only and assert nothing about them.
   Query: arXiv `all:"dLLM-Cache"` / `all:"dKV-Cache"`.
7. **Whether LR-DLLM's DreamOn row genuinely contradicts this group's .9342** —
   I deliberately did not claim it does (§3.3). Settling it requires matching
   split (SingleLine subset), grader (evalplus version), `MAX_LENGTH`, and
   checkpoint (`Dream-org/DreamOn-v0-7B` vs LR-DLLM's "DreamOn on
   DreamCoder-7B"). That is a real experiment, not a lookup.
8. **The 2026 acceleration and few-step-distillation waves** (13 papers named in
   §1.2) are **titles only**. If P2 is ever promoted, they must be read first —
   the probability that one of them already crosses acceleration with adaptive
   length is not negligible, and my Semantic Scholar outage means I could not
   check citation graphs.

**Where the evidence contradicts the framing I was given** (flagged as
instructed):
- The brief asked whether DAEDAL "MEASURE[s] canvas-budget sensitivity as a
  diagnostic about baselines, or merely FIX[es] it as a method" and said "that
  distinction decides whether finding #1 above is a novel finding or a
  reproduction." My answer splits the question: DAEDAL **measures the sensitivity
  and publishes the sweep** (so the *phenomenon* is a reproduction — **S4-G0
  fires**), but **does not use it diagnostically against prior work** (so the
  *diagnostic claim* survives). The brief's binary would have said "novel"; the
  correct answer is "phenomenon preempted, diagnostic not".
- The brief framed S4-G1 as an owed experiment (~4–6 GPU-h on a second
  mask-diffusion model). **DAEDAL Tables 1 and 2 already run it on two models.**
  Recommend cancelling S4-G1 as a GPU item.
- The brief describes finding #1 as "+26.5 pp" and treats that magnitude as
  evidence the issue matters at scale. Agreed — but note DAEDAL's own published
  swing is **+28.7 pp on HumanEval** and **+35.8 pp on GSM8K**, i.e. *larger*
  than ours, on a peer-reviewed model. The scale argument is stronger with their
  number than with ours, and it belongs to them.
- Minor: the brief lists on-disk assets as sufficient. For the DAEDAL/ρ-EOS/CAL
  lane they are **not** — every one of those methods is developed on **LLaDA**,
  and there are **no LLaDA-family weights on this disk**. Not a blocker (the lead
  confirmed downloads are routine), but P2 and P5 both silently depend on a
  ~16 GB download that should be started early.
