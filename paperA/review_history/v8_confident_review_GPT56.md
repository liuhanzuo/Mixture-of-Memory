---
soundness: 3.5
excitement: 4.0
overall: 3.5
confidence: 4.5
reproducibility: 3.5
---

# ARR-style independent review — Paper A v8 (CoMem)

## Summary and overall assessment

This is a substantially more convincing and better calibrated version of the CoMem story. The core contribution is not merely “cache hidden states,” but a controlled reusable-context interface: store one residual per token at split depth $j$, retrieve a fixed bounded pack, and execute only the suffix $[j{:}L)$. The paper's strongest evidence is the matched $j{=}0$ vs. $j{=}12$ replay control: same Qwen3-8B, LoRA, examples, selected chunk IDs/order, pack tokens, mask, and timing harness; it reports 99.19 vs. 96.07 RULER-B on 1,500 paired examples and 931.9 vs. 664.4 ms Read latency (1.403x). The continuous-prefix oracle reaching 99.19 is a useful mechanistic attribution: the exposed quality loss is attributable to the independently written reusable interface, rather than lack of capacity in layers 12–35.

The confident narrative **does improve excitement**. It now cleanly distinguishes (i) the matched 1.403x *depth-only Read* trade-off from (ii) the 64.9x *store-ready, bounded-selection online-prefill operating point*, and it foregrounds the useful negative result that equal-latency quality is selector-dependent. This is materially more interesting than a generic “we are faster” claim. Importantly, most of the prose now carries the relevant boundary conditions rather than hiding them.

My remaining concerns are narrow but meaningful: the novelty superlatives still rely on an incomplete/fragile literature audit, one “first unified measurement” claim is broader than the evidence establishes, headline natural-task evidence is mixed, and the appendix has a genuine float-layout defect (a very large mostly blank p.16 caused by Table 19 being floated well after its introduction). I would support acceptance after these presentation/claim repairs, but the current scores reflect a **weak accept / borderline** paper rather than a fully settled top-tier systems result.

## Evidence anchors and evaluation

### 1. What is convincingly supported

1. **The matched depth-reuse claim is credible and well isolated.** Table 2 and Appendix Table 31 explicitly share the selected pack, order, examples, LoRA, and model; the source further states that all 168 mounted modules and per-example pack tokens are shared and only replay start differs. The observed 3.12-point loss has paired-bootstrap 95% CI [2.36, 3.93], with McNemar $p=8.79\times10^{-24}$ (83 full-only vs. one layer-12-only correct). The timing protocol uses three independent processes, 20 reads each, with process-level speedup 1.402–1.404. This is the paper’s central soundness strength.

2. **The paper correctly limits the 64.9x number.** Abstract, Introduction, Table 7, and Appendix A.1 say it is a 128k *store-ready* online-prefill comparison on one H20: dense 71.37 s / 50.0 GB versus CoMem 1.10 s / 18.7 GB, excluding one-time Write and external-store fetch. They explicitly say it combines bounded selection and depth reuse and is neither the incremental-depth effect nor end-to-end repeated-query speedup. Retaining this distinction makes the confident framing defensible.

3. **The formula/accounting is internally correct.** Eq. (1) derives residual/full-KV bytes per token as
   $$\frac{d}{2L n_{kv}d_{head}}=\frac{n_q}{2Ln_{kv}}.$$
   With the documented Qwen3-8B configuration ($L=36$, $n_q=32$, $n_{kv}=8$, $d=4096$, bf16), this is $32/(2\cdot36\cdot8)=1/18$; a 4096-wide bf16 residual is 8,192 B/token and full per-layer KV is 147,456 B/token. The stated approximately 1 GiB at 128k tokens is also arithmetically consistent (about 0.98 GiB). The nominal pack length is correctly given as $1+12\cdot512+512=6{,}657$.

4. **The main additional numerical claims are generally well scoped.** The 32-token overlap changes the displayed multikey score from 92.5 to 98.5 (gain 6.0; CI [3.0, 9.5]) while preserving stored residual count and Read pack size; the paper correctly says Write work/edit invalidation grow. Equal-latency Table 4 reports the correct sign convention (CoMem minus replay): BM25 is -11.56 with hierarchical CI [-18.67, -5.11], while BGE is -1.00 with CI [-10.67, 8.33]. The manuscript appropriately calls the latter unresolved.

5. **Baseline fairness is substantially improved.** The central claim uses a same-backbone/same-LoRA/same-pack endpoint. External baselines are consistently labelled descriptive or unmatched where backbone, prompt setup, training, position extension, retained state, or timing boundary differs. In particular: KV-Direct’s native-RoPE 64k/128k rows are called unextended stress references; MemoryLLM’s different chat-tuned Llama-3 backbone is not treated as matched; LLoCO’s three-task macro is not compared to the six-task LongBench macro; and SnapKV/PyramidKV are assigned their full-prefill boundary with no cross-table latency ratio. This is good ARR evidence hygiene.

6. **The paper does not conceal important weaknesses.** Limitations explicitly notes single-model/English/lexical-selector scope, a non-clean seed test (batch 8 vs. 3), lack of a same-backbone PIC/modular-KV implementation, storage/update costs, single-query medians rather than throughput/tails, synthetic-only overlap repair, incomplete contamination audits, and mutable undated GPT-4o judging. That candidness supports both soundness and reproducibility scores.

### 2. Novelty and “first” claims: what to keep vs. downgrade

The manuscript’s bibliography reaches July 2026 and includes nearby categories: reusable representations (ReadOnce; Embedding Recycling), hidden-state restoration (HCache; KV-Direct), PIC/chunk KV (CacheBlend, TurboRAG, RAGCache, Cache-Craft, EPIC/MEPIC/APE), and learned modular KV (KV Packet, Cartridges/Cartridges at Scale, SemPIC). The structural comparison in Table 1 is useful. **However, a recent-work cross-check identifies an omitted relevant paper, _LLMCache: Layer-Wise Caching Strategies for Accelerated Reuse in Transformers_ (arXiv:2512.16843, Dec. 2025): it reuses intermediate activations across semantically similar inputs and advertises support for arbitrary transformer layers.** Its problem setting and interface do not, from the available primary description, establish the same persistent-document / bounded-retrieval / query-conditioned suffix formulation or same-pack $j{=}0$ measurement; nevertheless, it makes the broad wording “first reusable-context system to tune split depth” unsafe unless the distinction is made explicit. Add it to Related Work/Table 1 and state the object/interface difference. The wording should track what this updated audit actually establishes.

| Claim location | Assessment | Recommended final wording |
|---|---|---|
| Abstract: “To our knowledge, CoMem is the first reusable-context system to tune split depth and isolate it with a matched $j{=}0$ endpoint…” | **Needs narrowing after the recent-work check.** LLMCache (arXiv:2512.16843) is an omitted layer-wise intermediate-activation reuse system that supports arbitrary layers. The distinctive conjunction—persistent document chunks, one residual/token, direct query-conditioned suffix execution, bounded retrieval, and an identical-evidence $j{=}0$ endpoint—still appears materially narrower. | Replace with: “**Among document-reuse systems we are aware of, CoMem jointly** treats split depth as a tunable serving variable **and evaluates it with** an identical-evidence $j{=}0$ endpoint.” Add LLMCache to the audit and explain why its semantically similar-input activation cache is not this exact setting. |
| Introduction: “CoMem is the first such formulation to our knowledge.” | Same judgment; redundant with the abstract. The surrounding paragraph is careful and cites the closest categories. | Keep only once, preferably in Introduction, in the softened form above; remove the duplicate abstract superlative or replace it with “This yields a controlled formulation.” |
| Introduction: “the first unified measurement of a single-residual reusable object across quality, per-query latency, persistent storage, one-time Write, and empirical break-even.” | **Needs downgrade.** It may be true in the authors’ intended niche, but “first unified measurement” is broader than the provided related-work comparison proves. Several cited systems report systems quantities, and the claim depends heavily on what counts as the same object/boundary. | Replace with: “**We provide a unified measurement** of this single-residual interface across …” Drop “first.” |
| Conclusion: “CoMem opens a new systems axis” / “establish transformer depth as a measurable and tunable dimension.” | **Supported as a framing claim**, provided it is read as the paper’s controlled abstraction, not a claim that no prior method varies compute depth. | Keep, but prefer “**makes transformer split depth an explicit, measured systems axis for reusable context**.” |
| Abstract/Introduction: “dominant tested [error] factor” and “most of the local gap can be repaired.” | **Supported only on the named 8k/16k paired synthetic multikey cohort.** The 2x2 changes 92.5/88.0 to 100/100; $w=32$ recovers 6.0 of the 7.5 local-context gap. | Keep the restriction inline: “the dominant factor **in this 8k/16k multikey diagnostic**” and “repairs most of the **tested local** gap.” |
| Abstract/Conclusion: 64.9x acceleration-style phrasing | **Supported only as an operating point**, not as general/end-to-end speedup. | Retain “store-ready 128k online-prefill operating point” every time the 64.9x number appears; do not shorten it to “64.9x faster serving.” |

This calibrated confidence is an improvement over cautious but vague positioning: the paper should state the real novelty directly, but it should not stake acceptance on an absolute literature priority claim that is expensive for reviewers to verify.

### 3. Soundness limits that still matter

- **Natural-task generalization is mixed and should not be rhetorically treated as validation of the headline trade-off.** In the matched rows, LongEval falls 97.2 to 69.0, LoCoMo 41.59 to 38.27, LongBench 12.31 to 12.15, and BABILong macro 56.14 to 50.43. The main text does report these numbers and says BABILong is mixed, but the sentence “these results show that the residual interface transfers beyond the synthetic headline” is too positive without immediately naming the substantial LongEval drop. Suggested replacement: “These scope checks show that the interface remains usable on several non-synthetic evaluations, but transfer is uneven—especially on LongEval—and they should not be read as a uniform quality claim.”

- **The oracle is strong attribution evidence, not an implementation remedy.** The continuous-prefix oracle re-runs lower layers over the selected packed sequence for every query and is explicitly non-reusable. The paper says this correctly in Table 32; preserve that qualification wherever “recovers the full gap” appears.

- **The core latency evidence is Read-phase rather than deployment throughput.** Table 2 correctly excludes selection, persistent I/O, Write, and decode; total decode medians are similar (about 2.76–2.86 s), and main serving numbers are single-query medians. This does not invalidate the contribution, but it supports the paper’s own limitation and prevents calling the system generally 1.403x faster. The title/abstract are okay; do not introduce a stronger serving claim in the final revision.

- **Reproducibility is good but not complete.** Configurations, hashes, sample counts, seeds, timing boundaries, uncertainty procedures, and artifact intentions are unusually specific. Still, the teacher top-64 probability mass was not logged; the flagship is one effective-batch-8 run; the other two runs alter effective batch; aggregate GPU accounting is incomplete; GPT-4o lacks a pinned snapshot; and the anonymous release supplies score-only exports rather than benchmark predictions/text. A 3.5/5 reproducibility score is appropriate: materially above average documentation, but not enough for fully independent end-to-end verification of every headline.

## Required format / manuscript-structure audit

- **Counted main-text density (new 7–8-page principle): Pass, substantively.** The counted material is PDF pp.1–7: title/abstract plus Sections 1–6. It contains seven full, dense pages of introduction, related work, method, equations, experimental protocol, six main tables/figures, mechanism analysis, and conclusion; p.7 is also materially filled (the conclusion occupies a full column, while the left column contains Tables 5–6 and the final experimental analysis). The next page is deliberately a `\clearpage`-separated, unnumbered Limitations/Ethics page, so Limitations, Ethics, References (pp.9–11), and Appendix are not being used to pad the count. Thus the submission both satisfies the ≤8-page cap and **substantively reaches the new ≥7-page floor**.
- **Density recommendation:** No filler prose is needed. If authors want to use any remaining main-text flexibility, prioritize moving one **claim-bearing experimental artifact** forward—preferably a compact matched natural-task scope table (the existing matched $j{=}0$/$j{=}12$ LongEval, LoCoMo, LongBench, BABILong values) or the equal-latency protocol’s compact cohort/boundary row—rather than expanding narrative. This would make the non-synthetic transfer limitations and selector-dependent conclusion visible at the point of claim, while retaining 7–8 dense pages.
- **Conclusion:** Pass. It is Section 6 on p.7 and does not overflow into the Limitations page.
- **Limitations and Ethics:** Pass. Both are unnumbered (`\section*{...}`), appear together on a distinct page (PDF p.8), and are separated from body floats by `\FloatBarrier` and `\clearpage`.
- **References:** Pass. They start after a dedicated page break on PDF p.9 and are followed by a clear page break before Appendix A. The visible order is conventional alphabetical bibliography order (e.g., Agarwal, Ahn, Bai, …; ending Yang, Zhang), with no apparent body/appendix float intrusion.

## Appendix float and typography audit (rendered page by page)

The appendix is legible overall: most tables use small/scriptsize, but the numerical columns and captions remain readable at normal PDF zoom; I did not see reversed table order. The authors have added several `\FloatBarrier`s and most cited tables land near the relevant discussion. However, the appendix still has one conspicuous layout failure and several weaker spacing issues.

| PDF page | Finding | Assessment / necessary repair |
|---|---|---|
| 12 | Table 7 precedes the A.1 prose that cites it, but it is adjacent and readable; page is otherwise well used. | Acceptable float behavior. |
| 13 | Table 8 appears directly after its A.1 callout; Table 9 is nearby. | Good. |
| 14 | Tables 10–14 are compact and readable; no reversed ordering. | Good, though captions are dense. |
| 15 | Tables 15–18 are logically ordered and readable. | Good. |
| **16** | **Major whitespace / far-from-citation problem:** A.3.1 heading appears at upper left, while Table 19 is floated in the right column and its preceding introduction is on p.13. The lower-left ~2/3 of the page is blank. This is visible both in the rendered PDF and page density (248 words). | **Must fix.** Do not leave a largely empty appendix page merely to float Table 19 several pages after its citation. Make Table 19 a non-floating `\captionof{table}`/minipage block immediately after A.3.1, or restructure the `table*`/single-column sequence with a barrier before A.3.1. Rebuild and verify p.16 has normal fill. |
| 17 | Table 20 is above A.3.2 and Tables 21–22 follow immediately. | Mostly good; Table 20 is close enough to its heading. |
| 18 | Tables 23–24 and A.4 prose lay out normally; Table 25 follows on p.19. | Good. |
| 19 | Table 25 is immediately followed by the contamination-boundary discussion. | Good. |
| 20–21 | Reproducibility tables 26–29 are readable, adjacent to their discussion, and use sensible compact type. | Good. |
| 22 | Several tables (30–33) plus text; dense but readable, no obvious float reversal. | Acceptable; avoid shrinking further. |
| 23 | Tables 34–35 are adjacent to Hy3 discussion. | Good. |
| 24–25 | Statistics Tables 36–37 appear near their relevant subsections; no large blank page. | Good. |

A small source-level issue worth fixing while addressing p.16: the appendix sequence inserts `\FloatBarrier` after `tab_overview` and after `tab_scaling`, but the cited single-column `tab_h2h` can still float to the next page/column in an aesthetically poor way. The final build should be inspected after changing float constraints, because adding more global barriers may create a different blank page.

## Score rationale

- **Soundness: 3.5 / 5.** The key causal/measurement comparison is unusually carefully paired, numerical accounting checks out, and limitations are frank. I do not give 4+ because headline real-task transfer is mixed, the central method is demonstrated chiefly on one model and lexical selection, the reusable repair is synthetic-only, and serving evaluation lacks concurrency/tail measurements.
- **Excitement: 4.0 / 5.** The self-confident revision improves the paper: split depth plus a true matched endpoint is a clean, reusable systems abstraction; the mechanism/selector boundary results make it more than a speed table. It could be broadly useful to document/repository/memory workloads, though the performance trade-off means the impact is not yet universally demonstrated.
- **Overall: 3.5 / 5.** Weak accept / borderline. The core claim is credible and interesting; acceptance should be conditioned on narrowing the literature-priority language and fixing the appendix layout. The paper would benefit from one more same-backbone nearest-interface comparison or stronger natural-task repeated-query evidence, but that is not a last-minute formatting fix.
- **Confidence: 4.5 / 5.** I inspected the allowed v8 PDF and source, checked claim wording, equations, tables, structure, citations, and rendered appendix pages. My uncertainty is principally about exhaustive priority relative to the broader very recent literature, not about what this manuscript reports.
- **Reproducibility: 3.5 / 5.** Excellent protocol disclosure for a paper of this kind, with hashes and detailed timing/uncertainty descriptions, but incomplete retention/pinning and mutable API judging prevent a higher score.

## Necessary final revisions (priority order)

1. **Fix Appendix p.16 float placement.** Put Table 19 immediately after A.3.1 (or otherwise use the empty lower-left page area); rebuild and visually inspect all appendix pages. This is the only clear current presentation defect.
2. **Update and narrow the novelty audit, without retreating to vagueness.** Add omitted LLMCache (arXiv:2512.16843) to Related Work/Table 1; keep one restricted structural claim (“Among document-reuse systems we are aware of…”), remove the duplicate absolute “first,” and change “first unified measurement” to “a unified measurement.”
3. **Make natural-task scope equally prominent in the confident narrative.** When claiming transfer beyond synthetic RULER, state the matched LongEval/LoCoMo/LongBench/BABILong values or explicitly say transfer is uneven. Do not let the 1.403x RULER trade-off imply uniform task-level retention.
4. **Maintain strict labels on every speed number.** “1.403x selected-pack Read” and “64.9x store-ready bounded-selection online-prefill” should remain inseparable labels in abstract/conclusion/captions. Do not call either an unqualified end-to-end serving speedup.
5. **One final consistency pass.** Ensure every external baseline statement retains its fairness qualifier (same backbone? same prompt? same context/position extension? same timing boundary?) and every “dominant/recovers” mechanism statement names its tested 8k/16k multikey cohort.
6. **Respect the 7–8-page density principle without padding.** The current counted body is already a substantive seven pages. If revising layout, move a compact claim-bearing experiment (matched natural-task scope or equal-latency cohort/boundary) into the body rather than adding explanatory prose; do not reduce the counted body below seven pages.
