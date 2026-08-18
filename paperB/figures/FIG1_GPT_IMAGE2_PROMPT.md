# GPT Image 2 prompt for Paper B Figure 1

## Recommended usage

Upload `fig1_concept_preview.png` as a **layout reference**, then paste the
prompt below. Ask the model to rebuild the diagram rather than merely upscale
the reference. Generate at the highest available resolution in a wide
landscape aspect ratio close to **3:2 or 5:3**.

## Master prompt

```text
Use case: scientific-educational / infographic-diagram
Asset type: Figure 1 of a machine-learning research paper in an ACL-style two-column PDF

Primary request:
Recreate the attached reference as a polished, publication-quality scientific vector-style diagram. The figure explains a prune–re-grow–heal experiment on a 32-layer language model and the resulting dissociation between perplexity recovery and factual-knowledge recovery. Preserve the scientific meaning and all numerical values exactly, but improve the visual hierarchy, typography, alignment, spacing, and arrow routing. The final image must remain legible when reduced to one column of a two-column academic paper.

Canvas and overall composition:
- Wide landscape canvas, approximately 5:3 aspect ratio.
- Pure white background.
- Leave 6–8% clean outer padding on all sides.
- Use a strict left-to-right three-stage pipeline:
  1. original pretrained 32-layer model on the left;
  2. pruned-and-regrown 16-layer model in the center;
  3. two post-healing outcomes on the right.
- Use a clean invisible grid. Keep all text, arrows, stacks, and result boxes in separate lanes.
- No text may overlap any arrow, line, box, stack, or other label.
- Arrow lines must never pass through text. Put a small white buffer behind labels placed near arrows.

Stage 1 — original pretrained model:
- Place a tall, narrow vertical transformer stack at roughly 24% of the canvas width.
- The stack represents 32 pretrained decoder layers.
- Use a black or dark-slate outline, approximately 1.2–1.5 px at final resolution.
- Use 8 subtle horizontal group separators rather than 32 visually noisy lines.
- The bottom 14/32 of the stack, from 0 to 0.4375 of its height, is the retained region. Fill it with very pale periwinkle blue.
- Draw a prominent red dashed horizontal cut line at exactly 14/32 = 0.4375 of the stack height from the bottom.
- Everything above the cut line represents discarded pretrained layers. Fill this upper region with very light cool gray and a sparse diagonal hatch pattern.
- Add a small horizontal callout near this upper gray region reading exactly:
  "discarded pretrained layers"
- Below the stack, centered, write exactly:
  "Pretrained 32L"

Functional depth markers on the original stack:
- Reserve a dedicated label lane to the LEFT of the original stack.
- Each label uses a short horizontal leader arrow pointing to the correct depth. Leaders must not cross one another.
- At 0.13 of stack height from the bottom, draw a thin dashed blue marker and label it exactly:
  "Semantic features*"
  "~0.13L"
- Between 0.562 and 0.594 of stack height, draw a narrow amber/yellow horizontal band. This band must visibly lie ABOVE the red pruning cut, inside the discarded gray region. Label it exactly:
  "Knowledge decodability"
  "0.56–0.59L"
- At 0.944 of stack height, draw a thin dashed magenta-purple marker and label it exactly:
  "Next-token sharpening*"
  "~0.94L"
- The knowledge band is the most important depth marker. Make it visually salient but not fluorescent.

Transition from Stage 1 to Stage 2:
- Draw a red horizontal arrow from the cut line of the original stack toward the center stack.
- Place a compact two-line label above this arrow, with a white text background so the arrow never touches the letters.
- The exact text is:
  "Prune at 14/32"
  "+ 2 fresh layers"
- Do not use a scissors icon or any decorative symbol.

Stage 2 — compressed model:
- Place a second vertical stack near 50% of the canvas width.
- Its visual height should be approximately one half of the original stack, communicating 16 layers versus 32 layers.
- The lower 14/16 of this stack represents inherited pretrained decoder blocks. Fill it with pale periwinkle blue.
- The upper 2/16 represents newly initialized layers. Fill it with warm pale amber.
- Keep the same dark outline and subtle group separators as the original stack.
- Add two clean external labels, not overlapping the stack:
  "14 inherited"
  "2 fresh"
- Below this stack, centered, write exactly:
  "Compressed 16L"

Transition from Stage 2 to Stage 3:
- Draw a dark royal-blue horizontal arrow from the center model toward a branching point before the outcome boxes.
- Place the following exact label above the arrow:
  "Heal on Dolmino"
- The label must not touch or cover the arrow.
- At the right end, split the arrow cleanly into two branches, one pointing to the upper green result box and one pointing to the lower rose result box.
- Use square or gently rounded elbow routing. Both branches should have arrowheads and equal visual weight.

Stage 3 — outcomes:
- Place two equal-width rounded rectangular boxes in a neat vertical column on the right.
- Align their left and right edges.
- Give the boxes generous internal padding and no drop shadows.

Upper outcome box:
- Very pale mint-green fill, muted green-gray border.
- Exact text, preserving capitalization, punctuation, and numbers:
  "Perplexity recovers"
  "PPL 10.693 vs. 7.398"
  "1.445× tax"
- Make "Perplexity recovers" bold.

Lower outcome box:
- Very pale warm rose fill, muted rose-gray border.
- Exact text, preserving capitalization, punctuation, and numbers:
  "Knowledge lags"
  "MMLU recovery: 17.6%"
  "Fully random init: ≈0%"
- Make "Knowledge lags" bold.

Footnote:
- In small neutral-gray text at the lower left or lower center, write exactly:
  "* Qwen3 cross-model depth reference"
- Keep it separate from the stage labels and never let it collide with "Pretrained 32L" or "Compressed 16L".

Typography:
- Use a clean, professional sans-serif typeface similar to Inter, Helvetica, Arial, or Source Sans.
- Result-box headings should be semibold or bold.
- All other labels should use regular or medium weight.
- Use consistent capitalization exactly as specified above.
- Use true mathematical symbols: en dash in "0.56–0.59L", multiplication sign in "1.445× tax", approximation sign in "≈0%".
- Do not substitute malformed glyphs.
- Keep the smallest text comfortably legible after the figure is reduced to approximately 85 mm wide.

Color palette:
- inherited layers: very pale periwinkle blue, approximately #DDE4FF
- inherited-layer accent/text: deep muted blue, approximately #2949A8
- fresh layers: pale amber, approximately #F7D7A8
- fresh-layer accent/text: muted orange, approximately #C96A00
- discarded layers: very light cool gray, approximately #EEF0F3
- hatch lines: medium cool gray, approximately #9AA0A8
- pruning cut and prune arrow: deep muted red, approximately #B3262D
- semantic marker: muted cobalt blue, approximately #3157C8
- knowledge band: soft amber-yellow, approximately #F3D35A
- knowledge label: burnt orange, approximately #B85E00
- next-token marker: muted magenta-purple, approximately #C64F7B
- healing arrow: royal blue, approximately #173DBA
- PPL result fill: approximately #E6F7E6
- knowledge result fill: approximately #FBEAEA
- primary text: near-black, approximately #15171A

Style and rendering:
- Flat 2D vector infographic.
- Precise scientific diagram, not an artistic illustration.
- Crisp edges, uniform strokes, restrained academic palette, generous whitespace.
- No gradients, no glossy effects, no bevels, no 3D perspective, no shadows, no photorealism.
- No decorative neural-network nodes, brains, robots, chips, books, light bulbs, or icons.
- No title, figure number, caption, author name, institution, logo, watermark, or page background.
- Do not invent any values, labels, layers, arrows, tasks, or annotations.
- Do not paraphrase the specified text.
- Do not duplicate words or numbers.
- Do not place the amber knowledge band below the pruning cut.
- Do not make the compressed model the same height as the original model.
- Do not visually imply that the fully random-init control has zero perplexity recovery; the ≈0% value refers only to MMLU recovery.

Priority order:
1. Correct scientific structure and relative depth positions.
2. Exact numerical values and exact wording.
3. No overlaps and clear left-to-right reading order.
4. Legibility at paper-column scale.
5. Aesthetic polish.

If exact text rendering becomes unreliable, do NOT replace it with gibberish.
Instead, preserve clean empty label areas and result boxes with enough room for
manual text overlay.
```

## Follow-up correction prompt if the first output has text or overlap errors

```text
Keep the overall visual style and geometry, but correct the diagram with surgical precision:

1. Remove every text overlap and every arrow that crosses text.
2. Preserve the left-to-right order: Pretrained 32L → Compressed 16L → two outcomes.
3. Keep the red cut at 14/32 = 0.4375 of the original stack height.
4. Keep the amber knowledge band at 0.56–0.59L, above the cut and inside the discarded region.
5. Make the compressed stack exactly about half the visual height of the original stack, with 14/16 blue inherited layers and 2/16 amber fresh layers.
6. Replace all existing labels with the following exact text only:
   - Semantic features*
   - ~0.13L
   - Knowledge decodability
   - 0.56–0.59L
   - Next-token sharpening*
   - ~0.94L
   - discarded pretrained layers
   - Pretrained 32L
   - Prune at 14/32
   - + 2 fresh layers
   - 14 inherited
   - 2 fresh
   - Compressed 16L
   - Heal on Dolmino
   - Perplexity recovers
   - PPL 10.693 vs. 7.398
   - 1.445× tax
   - Knowledge lags
   - MMLU recovery: 17.6%
   - Fully random init: ≈0%
   - * Qwen3 cross-model depth reference
7. Do not add a title, caption, logo, icon, watermark, or any new number.
8. Maintain a pure white background and flat vector academic style.
```

## Text-safe fallback

If the model repeatedly corrupts text, ask it to generate only the geometry,
colors, boxes, and arrows, leaving all label regions blank. Add the text later
in LaTeX, Illustrator, Figma, or Inkscape. This normally gives the most reliable
camera-ready result.
