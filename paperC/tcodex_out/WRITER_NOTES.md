# WRITER NOTES

## A. Numbers wanted but unavailable

- None required by the final prose remains as `\todo{}`.
- The manuscript does not print a step-121000 healed-Qwen3 result because it is not in the evidence pack and the paper was explicitly required to stand without it.
- No figure was produced because the brief prohibited figures. A human may still want a compact v1-versus-v2 re-sort figure if the restriction changes.

## B. Claims judged too weak or too confounded

- I did not claim a universal damaged-arm result. The manuscript uses the evidence-safe `14/15` MMLU-Pro statement and describes `qwen3_8b_base/k14` as a real but immaterial exception.
- I did not turn the two v1 below-floor MMLU-Pro cells into a heal-versus-no-heal claim. Under v2, neither is below its arm-conditional null.
- I did not fit or narrate a depth curve. The sampled ladder is non-monotone and is described as a cliff rather than a gradient.
- I did not treat pooled results as per-benchmark evidence.
- I did not claim that content is generally fairer, that letter is generally unreliable, that exact ties are a general mechanism, or that larger vocabularies monotonically produce more ties.
- I did not report the planned `H_heal`/`H_family` dichotomy as tested. Its comparator antecedent has dissolved under v2.
- I did not describe the healed and unhealed family contrast as causal because damage regime and family remain confounded.

## C. Source evolution and contradictions

1. **E1 changed while the paper was being written.**  
   The original user brief required limitation L1 to say that the OpenBookQA character-unit value was not recomputable. During this task, `paperC/tcodex_out/EVIDENCE_PACK.md` gained an August 14 addendum, backed by commit `165ccc9`, explicitly superseding its own Section J and closing E1 with `paperC/evidence/construct_nulls_length_unit.json`. Because the evidence pack is the authoritative and dynamically updated number source, the manuscript follows the addendum rather than the now-stale prompt text. `07_limitations.tex` records the smaller remaining provenance limitation instead.

2. **The evidence pack's older Section J remains internally stale.**  
   It says the character-unit values are unrecomputable; the addendum says to retain that text only for provenance and not use it. The manuscript uses the addendum.

3. **The gap audit contains historical and updated E1 statements in the same row.**  
   Its update says E1 is closed, while the preserved original paragraph says no suitable Python environment existed. The evidence-pack addendum resolves the conflict.

4. **The read-out v2 prose uses two denominator scopes for constant emitters.**  
   The full-dataset v1 examples always-A/E/J give `0.000`, `-2.111`, and `-3.807` pp, while the ten-letter implementation self-test conditions each letter on items where that letter is legal. The method section states these scopes separately.

5. **The evidence sources sometimes shorten `NO_ITEM_LEVEL_SIGNAL` to `NO_SIGNAL`.**  
   The manuscript uses the canonical full label in prose; appendix tables use the compact phrase “no signal” only as a display abbreviation.

6. **The formal G6 vocabulary and the Llama-2 intact row are not perfectly aligned.**  
   The evidence establishes absolute item-level signal but blocks relative recovery because the intact anchor is only `0.0545`. The manuscript says “anchor blocked” rather than forcing one contradictory label.

7. **The “15 damaged cells” subset can be miscounted if not defined.**  
   The prose calls it the “15 designated damaged cells” and reports the exact non-OLMo and OLMo counts rather than implying that every ladder arm is included.

## D. Human checks before submission

1. **Bibliographic venue strings and authors.**  
   The bibliography contains only the authorized evidence-pack citation set. Please verify the exact ICLR/ICML/NeurIPS BibTeX presentation desired by the venue, especially:
   - Cho et al., whose overlap audit used arXiv v4 because the ICLR camera-ready PDF could not be fetched.
   - Bean et al., whose full author list was taken from the arXiv metadata.
   - Ding et al., whose camera-ready title must remain **“Through”**, not arXiv's “with.”
   - The two distinct Zheng entries.

2. **Camera-ready novelty check.**  
   Retry the Cho et al. camera-ready diff before submission.

3. **Page budget and build.**  
   The source is organized so that the bibliography precedes the appendix, and the main text was written to target the nine-page budget. A human must run the official ICLR build, inspect float placement, and confirm how the ICLR 2026 page limit counts references.

4. **Template/anonymity.**  
   Confirm that the required anonymous author block and omission of `\iclrfinalcopy` match the submission portal's current instructions.

5. **Final evidence freeze.**  
   The authoritative evidence pack changed during drafting. Freeze and checksum the final pack, then re-run a numeral audit before submission.

6. **Artifact policy.**  
   Decide whether to include a reproducibility statement or anonymized code/evidence archive beyond the present paper. The source paths named in table captions are local project paths and may need an anonymous artifact mapping.

7. **Main-text density.**  
   The draft is intentionally dense to retain the four pillars in the main body. A human may move one or more main-body tables to the appendix if the final ICLR style or reference count requires more room, but the small-benchmark power table must remain adjacent to any small-set null-result discussion.

8. **No figures.**  
   The paper currently has no figures, as required. If that restriction is lifted, the most useful figure would be the 27-cell v1-to-v2 re-sort; any such figure must be generated from the existing evidence JSON rather than hand-entered.
