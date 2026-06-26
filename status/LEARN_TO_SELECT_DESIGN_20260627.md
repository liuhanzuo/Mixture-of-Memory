# Learn-to-Select + Token-Reforward — end-to-end training design (2026-06-27)

> Author: researcher subagent. Read-only, no GPU. All citations `file:line`.
> Brief: the deployable readout (token-reforward) is solved (oracle-token qa1
> 8k=50/16k=28). The frozen reader-attn selector is NOT (reader-attn-token =
> 12 = baseline). The whole remaining value is "can we TRAIN the model to select
> the right chunk so the deployable path goes 12 → toward 50." Design the
> concrete, implementable training scheme (mix=0).

---

## TL;DR — the honest verdict up front

| # | Answer | conf |
|---|---|---|
| **Recommended MVP** | Scheme **(c) supervised selection loss on T2 needle** + **eval-consistent token-reforward path baked into the T2 forward**. T2's queried needle is ALWAYS at chunk 0, offset 0 (`niah_chunked_dataset.py:226,253-257`) → the needle chunk index is known for free at train time → a clean supervised target for the selector. This is the only scheme with a real gradient toward "select the needle." | HIGH (feasibility) |
| **Will it work?** | **Probably partial at best.** Two pre-registered failure modes from history both apply: (1) "train a cross-chunk selector" is a death-listed direction — every trained scorer landed at random precision (`RUN_REGISTRY.md:1247-1252`, gist 0-10%); (2) T2-needle training overfits the synthetic code format and does NOT transfer to natural-language BABILong (`RUN_REGISTRY.md:109,207-214`: T2 needle-loss≈0 but held-out long qa1 collapsed). The new ingredient that gives a *coherent* gradient (the reader's OWN q·k is the selector, not a bolt-on head — `RUN_REGISTRY.md:1256-1258` reader-attn=55% vs gist random) is the only reason this is not a flat repeat. But the precision wall is a 512-chunk-scale recall problem, and T2-format transfer is unproven. | MED-LOW (that it lifts deployable qa1 8k >30) |
| **Architecture** | (a) selection = the reader's native q·k at one layer (`_fifo_select_keep_set_reader_attn`, `layer.py:1612`), already used by the eval probe. (b) readout = token-reforward window = the existing SWA-train path (`dolmino_train_step._make_swa_window`, `train_mem_space_dolmino_cpt.py:2411`), but with the window slots chosen by the selector instead of "last-W". (c) supervision = CE pushing the selector's per-chunk salience onto chunk 0 (the known needle). | HIGH |
| **Biggest risk** | T2 format overfit (history: `RUN_REGISTRY.md:214`). Mitigation = needle at a RANDOM chunk (not always chunk 0) + natural-language needle phrasing + measure on held-out BABILong, not T2 loss. Kill if BABILong qa1 8k ≤ 20 (= hidden-oracle ceiling, no selection gain) at step 1500. | — |

---

## 0. State of the evidence (what is pinned)

From the three status docs + RUN_REGISTRY:
- Clean NOLEAK FIFO baseline qa1 8k=12, 16k=8 (heartbeat; `RUN_REGISTRY.md` fifo_b25 table shows W0 qa1 8k=40 at chunk512 step3000 — NOTE this is a *different* run than the heartbeat's "12"; the heartbeat's judgement-chain anchor is the b25 NOLEAK eval, qa1 8k=12).
- hidden-oracle (perfect select, frozen hidden readout) = ~20. This is the readout ceiling of the hidden path.
- **oracle-token (perfect select + raw-token reforward) = 50/28** @8k/16k → readout SOLVED.
- **reader-attn-token (frozen reader q·k select + raw-token reforward) = 12** → selection is the wall.
- Reader-attn chunk recall (512-chunk scale, top-2): 72/78/58/40% @4k/8k/16k/32k
  (`RUN_REGISTRY.md:1290`). So even the BEST untrained selector misses the needle
  ~half the time at 16k+ — and token-reforward of a WRONG chunk = baseline.

The question this doc answers: **can training move the selector's precision up
enough that token-reforward (which we know reads at 50 given the right chunk)
fires on the right chunk?**

---

## 1. End-to-end training: the discreteness problem and the three schemes

The core difficulty (user's framing): which chunks to reforward is a **discrete
top-k choice**, non-differentiable. Three ways to get a training signal:

### (a) STE (straight-through) — front-end hard top-k, gradient straight through

**What it is.** Forward: select top-k chunks by reader q·k (hard argmax),
reforward their raw tokens. Backward: the selected chunks' token hiddens carry
gradient through the LM loss (they are LIVE tensors), but the *index choice* is
non-diff and gets no gradient — exactly the existing
`fifo_train_keep_set_mode flat_readerattn` contract
(`train_mem_space_dolmino_cpt.py:3294-3303`: "no_grad index selection, kept
chunks keep gradients").

**The user's exact objection is correct and decisive.** When the selector picks
the WRONG chunk, the LM loss flows into reading-the-wrong-chunk; there is **no
gradient telling the selector to have picked a different chunk**. The q/k
projections that *did* the selection are upstream of a no_grad argmax — the
loss cannot reach them through the selection path. So STE alone trains the
*readout* of whatever was selected, never the *selection rule*. The selector
only improves indirectly (the q/k drift because they also serve normal
attention), which is precisely the idiosyncratic, needle-uncorrelated drift the
gist runs showed (`RUN_REGISTRY.md:1250-1252`: scorer trained hard, precision
stayed 0%). **Verdict: STE for readout only; cannot teach selection. Insufficient alone.**

### (b) Soft-selection (differentiable) — soft-weight the reforward

**What it is.** Score all chunks, softmax → weights, and somehow soft-combine.
**The user's parenthetical is the killer: token reforward is a discrete
concatenation** (`run_babilong_mem_space.py:742-743`: `torch.cat([chunks[c] ...])`
→ a forward over raw int token-ids). There is no "soft token." You cannot
weight a token-id by 0.3. Options to make it soft all destroy the mechanism:
- soft-weight the *hidden* of each chunk → that is the FIFO hidden-prefix path,
  whose readout ceiling is ~20 (the thing we are escaping). Defeats the purpose.
- soft-weight the *attention logits* of reforwarded chunks (bias) → keeps tokens
  discrete but needs ALL chunks in the window (32-64 chunks × 512 = OOM /
  dilution, the original wall `RUN_REGISTRY.md` full_haystack=0%).

**Verdict: incompatible with token-reforward. Reject.** Soft selection only
works on the hidden path, which has the wrong readout ceiling.

### (c) ★ Auxiliary supervised selection loss (RECOMMENDED)

**What it is.** Train-time, the needle chunk index is KNOWN (T2 synthetic). Add a
loss that pushes the selector's per-chunk salience distribution onto the needle
chunk, while the main LM loss flows through token-reforward (scheme a) on the
selected chunks. Two coupled losses:
- `L_select` (NEW): CE / ranking loss on the per-chunk reader-attn salience
  vector `sal` (the `[C]` tensor computed at `layer.py:1662-1674`) with target =
  needle chunk index. This is a **direct, dense gradient into the q/k
  projections** telling them "make the needle chunk score highest."
- `L_lm` (existing): answer-digit loss on the token-reforward window built from
  the selected (or, during warmup, the gold) chunks.

**Why this is the only scheme with a coherent selection gradient.** Unlike the
death-listed gist scorer (a *separate* MLP whose output never fed the reader),
here the supervised object is the **reader's own q·k salience** — the same
quantity that does the selection at eval AND that already scores 55% precision
untrained (`RUN_REGISTRY.md:1256`). We are sharpening a signal that is already
the best one we have, not training a new bolt-on. The gradient path is real:
`sal[c]` at `layer.py:1672` is `einsum(q_proj(query), k_proj(chunk))`, both
differentiable; only the `torch.no_grad()` wrapper (`layer.py:1647`) currently
blocks it — for the supervised loss we recompute `sal` WITHOUT no_grad.

**Feasibility of the supervision signal (the user's key sub-question):**
- T2 needle is placed at **chunk 0, offset 0**, deterministically
  (`niah_chunked_dataset.py:226` `query_k=0`, `:253-257` embed at chunk0/offset0).
  So in `dolmino_train_step` the needle's document-absolute chunk index = 0,
  and after FIFO eviction at buffer depth `b`, its buffer-local index is known
  too. **The supervision target is free and exact.** No probe, no oracle hack —
  it falls out of the data generator.
- `answer_mask` already isolates the answer digits (`niah_chunked_dataset.py:287-294`),
  so `L_lm` is already the precise-readout objective.
- **Code change is moderate** (see §3): a new arg `--t2_select_loss_weight`, a
  grad-bearing recompute of `sal` (refactor the no_grad block of
  `_fifo_select_keep_set_reader_attn` into a `_reader_attn_salience(...)` helper
  that returns the differentiable `[C]` vector), and a CE term folded into the
  T2 backward in `dolmino_train_step`.

**Verdict: the only viable scheme. This is the MVP.**

### (d) Alternatives considered

- **Gumbel-softmax top-k.** A relaxed differentiable top-k over chunks. Adds
  noise/instability; the `TOKEN_REFORWARD_DESIGN` doc already rated it below STE
  (`:135`). And it still soft-mixes the SELECTION, not the tokens — to get a
  hard reforward window you straight-through anyway. No advantage over (c)'s
  direct supervision when the target is known. **Defer.**
- **REINFORCE / policy gradient on the discrete select** (reward = answer
  correct). High variance, slow, and we have a *supervised* target (the needle
  position is known), so PG throws away free information. **Reject for T2.** (Only
  relevant if we ever train on data where the needle position is unknown.)
- **Two-stage coarse→fine (HNST).** Orthogonal architecture change; `TOKEN_REFORWARD_DESIGN:149-153` defers it. **Defer.**

---

## 2. Token-reforward TRAINING path — how much code, can it borrow SWA-window?

**Short answer: it can almost entirely borrow the existing SWA-window train
path. The reforward machinery already exists; only the slot-selection changes.**

### What exists today
- `dolmino_train_step._make_swa_window(w)` (`train_mem_space_dolmino_cpt.py:2411-2426`)
  builds `window = cat([context_chunks[-w:], target])`, masks the prefix labels
  to -100, returns `(window, labels, prefix_len)`. This IS a token-reforward
  window — raw token ids of W context chunks concatenated with the target, run
  through a full forward. It is byte-identical in spirit to the eval
  oracle-token window (`run_babilong_mem_space.py:742-743`).
- The bank is frozen around this forward (`:2459-2461`) so re-presented chunks
  are not double-written. Exactly the eval contract.
- T2 currently does NOT use it: the T2 branch calls
  `dolmino_train_step(..., answer_mask=...)` with **no `swa_train_chunks`**
  (`train_mem_space_dolmino_cpt.py:3732-3735`) → T2 trains on the **FIFO hidden
  prefix** readout (`_forward_fifo`, `layer.py:1262`), i.e. the weak path. This
  is consistent with HIDDEN_VS_SWA Q2: the hidden path is what's being trained.

### The change for token-reforward training
`_make_swa_window` takes the LAST `w` chunks. We need it to take the
**selector-chosen** chunks. Two implementation options:

- **Option A (minimal, recommended for MVP):** since T2's needle is at chunk 0
  and the question is the target, build the window as `cat([chunk_0, target])`
  during a **teacher-forced warmup** (gold chunk known), then switch to
  `cat([selected_chunks, target])` once the selector is trained. The window
  builder is ~5 lines: replace the `range(len-w, len)` slice with an arbitrary
  index list. Reuse all the freezing/masking logic verbatim.
- **Option B (general):** add `select_chunks: Optional[List[int]]` param to
  `dolmino_train_step`; when provided, `_make_swa_window` uses those indices.
  The caller computes them from the selector (grad-bearing salience argmax).

**Change size: SMALL-MEDIUM.** It is NOT a big rewrite of the trainer. The
reforward forward, label masking, bank-freeze, answer_mask, and backward all
already exist. New code:
1. `_reader_attn_salience(...)` helper (grad-bearing version of the no_grad
   scorer block, `layer.py:1647-1674`) — ~40 LOC, new method on the layer.
2. `_make_swa_window` accepts an index list instead of "last-w" — ~5 LOC.
3. `L_select` CE term + the needle-chunk target plumbed from the T2 sample —
   ~30 LOC in `dolmino_train_step` + the T2 dispatch (`:3714-3738`).
4. New args (`--t2_select_loss_weight`, `--t2_select_layer`,
   `--t2_reforward_warmup_steps`) — ~15 LOC.

Total ~90-120 LOC, one layer.py method + the T2 step. **No architecture change,
no new module, no new dataset.** This is why (c)+SWA-borrow is the right MVP:
it is mostly wiring, and it directly tests the only open question.

### Train/eval consistency (critical — this is where past T2 runs died)
- **Eval** = the ALREADY-IMPLEMENTED `--swa_readerattn_token` path
  (`run_babilong_mem_space.py:709-731`, `_select_chunks_reader_attn:476`):
  reader q·k selects chunks, token-reforward window. This is the deployable
  number we are trying to lift.
- **Train** = scheme (c): same reader q·k salience (now SUPERVISED), same
  token-reforward window. **The selection layer and topk MUST match between
  train and eval** (train `--t2_select_layer 16` == eval default 16;
  `run_babilong_mem_space.py:1518`). If they diverge, we repeat the "trained one
  thing, evaluated another" trap.

---

## 3. ★ MVP training config (executable-level, mix=0)

**Goal:** train the reader's q·k to select the needle chunk (supervised on T2's
known needle position) while the LM loss reads via token-reforward, then measure
the DEPLOYABLE eval path (`--swa_readerattn_token`). Success = the deployable
qa1 8k climbs from 12 toward oracle 50; concrete success line **qa1 8k > 30**.

### Data (the overfit-prevention is here, not optional)
History lesson: T2-as-shipped overfit the synthetic format and did not transfer
(`RUN_REGISTRY.md:214`). The MVP MUST harden T2 before trusting it:
1. **Random needle chunk** — change `query_k`/placement so the queried needle is
   at a RANDOM context chunk, not always chunk 0 (`niah_chunked_dataset.py:226,253-257`).
   Otherwise the selector learns "always pick chunk 0" — a constant, not a
   content-addressed rule, and it will NOT transfer. This is a ~10 LOC dataset
   change and is **mandatory**. The supervision target becomes the random index
   (still known → still free).
2. **num_keys ≥ 3** (2 distractors) so selection is forced to discriminate
   (`niah_chunked_dataset.py:260-273` already supports this; matches the h1fix
   difficulty `RUN_REGISTRY.md:1241`).
3. Keep the natural-text background (already pg19). Do NOT over-tune the needle
   phrasing; the point is whether content-addressed selection transfers at all.

### Single highest-ROI run
```
# node: one 8×H20 (diskB .53 or .245.174) or B200 .53
PYBIN=.venv/bin/python ; mix=0
scripts/train_mem_space_dolmino_cpt.py
  --model_path models/Meta-Llama-3-8B
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach
  --chunk_size 512 --batch_size 1 --gradient_accumulation_steps 4
  --unfreeze_backbone --unfreeze_layers_from 16   # NOTE: 16 not 24 — the
        # selection layer (L16) q/k MUST be trainable for L_select to bite.
        # The default NOLEAK run unfreezes from 24 (layer.py select layer frozen!)
  --gradient_checkpointing
  --curriculum 0:3 --bptt_window 1                # dolmino stays small/fixed
  # ---- T2 (the learn-to-select carrier) ----
  --t2_recall_mix_fraction 0.5                    # heavy T2 (this is a probe)
  --t2_num_keys 3 --t2_gap_tokens 8192            # n_ctx=16 @chunk512: real dilution
  --t2_curriculum 0:16                            # grow needle distance to eval scale
  # ---- NEW selection-supervision + token-reforward (to implement) ----
  --t2_select_loss_weight 1.0                     # weight on L_select CE
  --t2_select_layer 16                            # MUST == eval select layer
  --t2_select_topk 4                              # MUST == eval topk
  --t2_reforward_train                            # T2 LM loss via token-reforward window
  --t2_reforward_warmup_steps 500                 # warmup: teacher-force gold chunk
  --babilong_mix_fraction 0                       # CLEAN
  --total_steps 3000 --save_interval 500 --lr 1e-4 --warmup_steps 100
  --eval_interval 0 --grad_clip 1.0 --proj_grad_clip 0.1 --dtype bfloat16 --seed 42
```

### Curriculum logic
- Warmup (0-500): `L_lm` reforwards the GOLD needle chunk (teacher-forced) so
  the readout head adapts to the reforward window WITHOUT being poisoned by a
  bad selector. `L_select` trains the selector against the known target the
  whole time. This decouples "learn to read the window" from "learn to select."
- Main (500-3000): `L_lm` reforwards the SELECTOR'S top-k (eval-consistent). If
  the selector is right, readout is reinforced; the supervised `L_select` keeps
  pulling it toward the needle.

### Eval (deployable, what actually decides)
After each saved ckpt, run the EXISTING deployable probe (no new eval code):
```
scripts/run_babilong_mem_space.py --swa_readerattn_token --swa_readerattn_topk 4
   --batch_size 1   (select_layer hard-coded 16, matches train)
tasks qa1 (then qa5); lengths 4k,8k,16k,32k; n=100; score_nested
```
Anchors in hand: deployable baseline 12, hidden-oracle 20, oracle-token 50.

### Pre-registered decision gates (step 1500 first checkpoint with a real eval)
- **qa1 8k > 30** → selection training WORKS; the deployable path is climbing
  toward oracle. Continue to 3000, sweep topk∈{4,6}, then write up. **Win.**
- **qa1 8k 20-30** → marginal: selection improved past the hidden ceiling but
  not to oracle. Diagnose with the reader-attn recall probe (`probe_reader_attn_topk.py`):
  did precision rise? If precision rose but qa1 didn't → readout/transfer issue;
  if precision flat → supervision didn't bite (check L16 unfrozen, L_select grad).
- **qa1 8k ≤ 20 (= hidden-oracle, no selection gain)** → **KILL.** Selection
  training failed exactly as the gist runs did. Do not throw more GPU at it.

### Diagnostics to log every step (so a kill is informative, not blind)
- `L_select` value + the selector's needle-chunk rank (is the needle in top-k on
  TRAIN data?). If `L_select`→0 but eval precision stays at the frozen 55%, that
  is the **smoking gun of T2-format overfit** (the historical failure) — selector
  memorized the T2 needle's surface form, learned nothing transferable.
- Held-out check at step 500: run the deployable probe on BABILong (NOT T2). The
  gap between "T2 needle rank≈0" and "BABILong qa1 flat" is the transfer failure
  we MUST watch for from the start.

---

## 4. Risks + kill standard

### A. T2 format overfit — the #1 risk, history-confirmed
`RUN_REGISTRY.md:109,207,214`: rawkv_methodA had **train t2_needle loss≈0** yet
held-out long BABILong qa1 collapsed to floor — explicitly diagnosed as
"过拟合 train needle 格式非泛化检索" (overfit the synthetic needle format, not
generalizable retrieval). The selector can learn "find the `MEMORIZE:` token
pattern" instead of "content-address the queried entity," and BABILong has no
`MEMORIZE:` marker. **Mitigations (all mandatory):** random needle chunk (not
chunk0); num_keys≥3; natural-ish phrasing; **judge ONLY on held-out BABILong,
never on T2 loss.** **Kill** if BABILong qa1 8k ≤ 20 at step 1500 regardless of
how good T2 looks.

### B. "Train a cross-chunk selector" is a death-listed direction
`RUN_REGISTRY.md:1247-1259`: every trained selector (gist, multiple lr, grad-sync
fixed) landed at random needle precision; final verdict H2 = "训练一个 cross-chunk
选择器这条路本身难" ("training a cross-chunk selector is itself hard"). **What is
genuinely different here** (and the only reason to try): we are NOT training a
new scorer — we supervise the reader's OWN q·k, which is already 55% precise
untrained (`RUN_REGISTRY.md:1256`), 8.8× random. We are sharpening the one signal
that demonstrably correlates with the needle. That is a materially different bet
than the dead gist head. But it is still the same *family* of risk, so confidence
the lift materializes is MED-LOW.

### C. Memory/speed of token-reforward training
- Reforward window @ n_ctx=16, topk=4: window = 5×512 = 2560 tokens through a
  full forward WITH gradient (vs the W0 single 512-chunk grad forward). ~5×
  activation memory on the grad forward. `gradient_checkpointing` is already on;
  the existing SWA-train path (`swa_train_chunks`) already does exactly this
  multi-chunk grad window, so it is a known-tractable regime on H20 — but topk=4
  at 2560 tokens is heavier than the current 0:3 curriculum. **Mitigation:** start
  topk=4, grad_accum=4, batch_size=1; if OOM, drop topk to 2 (matches the
  in-distribution k=2 the TOKEN_REFORWARD doc notes, `:98`). Speed: ~2-3× slower
  per T2 step than W0; at t2_frac=0.5 → ~1.5-2× overall slowdown. Tolerable for a
  3000-step probe.
- The extra grad-bearing salience recompute (one q_proj + C k_proj over the
  buffer) is cheap relative to the reforward forward.

### D. Selection layer must be trainable
The default NOLEAK run unfreezes from layer 24 (`launch_...NOLEAK...:34`), but the
selection layer is L16 — **frozen** under that config. `L_select` would have zero
gradient. The MVP MUST use `--unfreeze_layers_from 16` (config above). If a coder
copies the NOLEAK launch verbatim, the experiment is silently dead. This is the
single most likely implementation bug; flag it loudly.

### E. Curriculum/dilution-regime mismatch
HIDDEN_VS_SWA Q2 (`:78`): the model is trained at shallow buffer depth and
evaluated at 25-64 deep. The MVP raises T2 to n_ctx=16 (`--t2_curriculum 0:16`)
to put selection IN the diluted regime it must work in at eval — this directly
addresses the historical curriculum-depth gap. But 32k eval = 64 chunks is still
beyond n_ctx=16 training; expect 32k to lag (consistent with reader-attn recall
40% @32k). Judge primarily on 8k/16k.

### Kill standard (summary)
- **Hard kill:** BABILong qa1 8k ≤ 20 at the step-1500 ckpt (no gain over the
  hidden-oracle ceiling → selection training failed).
- **Hard kill:** `L_select`→0 on T2 but deployable BABILong precision flat at
  ~55% → confirmed format overfit, no transfer.
- **Soft continue:** qa1 8k in 20-30 AND reader-attn recall probe shows precision
  rose above 55% → readout/transfer tuning, not a clean win, escalate to user.

---

## 5. Honest bottom line + alternative if this fails

**Is "train to select" likely to work? Lean NO-to-PARTIAL, but it is the correct
next experiment because it is cheap-ish and decisively tests the last open
question.** The evidence is split:
- *For:* the readout is proven (oracle-token 50); the supervised object is the
  reader's own q·k which is already the best selector we have (55%, not a dead
  bolt-on); the training path is 90% existing code; the supervision signal is
  free and exact (T2 needle position known).
- *Against (heavier):* trained cross-chunk selection is death-listed
  (`RUN_REGISTRY.md` H2 verdict); T2→BABILong transfer is death-listed
  (`:214`); the precision wall is fundamentally a 512-chunk-scale recall problem
  that a 16-chunk curriculum may not close; reader-attn recall is already 40% @32k
  and supervision may sharpen TRAIN precision without moving the OOD eval.

**If the MVP kills (qa1 8k ≤ 20), the honest pivots are:**
1. **Accept the deployable ceiling and ship the strong-mid-range story.** Even
   the FROZEN reader-attn-token likely has a good 8k (the probe should be re-run
   to nail it — heartbeat shows reader-attn-token 8k=12, but that may be topk/layer
   under-tuned; sweep topk∈{2,4,6} and select_layer first, ZERO training, before
   concluding selection can't be lifted). A no-train topk/layer sweep is strictly
   cheaper than this training run and should arguably precede it.
2. **Multi-layer selection voting** (sum salience over L{12,16,20,24}) — still no
   training, may lift precision past single-layer 55% (the TOKEN_REFORWARD doc
   suggests this, `:76`). Cheaper than training.
3. **Concede the precise-long-range win lives on the selection-precision side and
   that frozen reader-attn is the practical ceiling**, and report the mechanism
   finding (token-reforward readout solved; selection precision is the
   information-theoretic wall at 512-chunk scale) as the honest scientific result.

**Recommendation order:** (i) FIRST do the zero-train topk/select_layer/multi-layer
sweep on the existing `--swa_readerattn_token` path (cheapest, may already lift 8k
above 30 without any training and would moot this whole training plan); (ii) ONLY
if that plateaus below 30, run the §3 supervised-selection MVP; (iii) kill per §4.

---

## Confidence
- §1 scheme analysis (STE can't teach selection / soft incompatible / (c) is the
  only coherent supervised path): **HIGH**.
- §2 (token-reforward training borrows the SWA-window path, small change):
  **HIGH** — the reforward machinery demonstrably exists (`:2411`, `:2681-2702`).
- §3 MVP config executable correctness: **HIGH** on the wiring; the
  `--unfreeze_layers_from 16` and random-needle-chunk points are the load-bearing
  non-obvious bits.
- That the MVP LIFTS deployable qa1 8k > 30: **MED-LOW** (two death-listed risks
  apply; the one novel ingredient is real but unproven at scale).
- Overall: **run the zero-train topk/layer/multi-layer sweep FIRST; gate the
  supervised-selection training on that sweep plateauing below 30.** Be ready to
  concede selection precision is the wall.

---

### Appendix — key citations
- T2 needle position known (chunk0/offset0, query_k=0): `src/memory/mem_space/niah_chunked_dataset.py:226,253-257`; answer_mask `:287-294`; num_keys/distractors `:260-273`; curriculum set_n_ctx `:131-137`.
- Reader-attn salience scorer (the selection signal, no_grad block to make grad-bearing): `src/memory/mem_space/layer.py:1612-1683` (sal compute `:1662-1674`, no_grad `:1647`).
- FIFO hidden readout (what T2 trains today, the weak path): `src/memory/mem_space/layer.py:1262-1342`.
- Existing STE keep-set train contract (readout-only gradient): `scripts/train_mem_space_dolmino_cpt.py:3294-3326`.
- SWA-window TRAIN path (the token-reforward train machinery to borrow): `scripts/train_mem_space_dolmino_cpt.py:2411-2426` (dolmino_train_step), `:2681-2702` (tbptt SWA window).
- T2 dispatch (currently NO swa_train_chunks → trains hidden path): `scripts/train_mem_space_dolmino_cpt.py:3714-3738`.
- route_aux grad-bearing score-hook + CE supervision (template for L_select): `scripts/train_mem_space_dolmino_cpt.py:792-816,819-831`.
- Eval deployable path (already implemented, the number to lift): `scripts/run_babilong_mem_space.py:476-594` (selector), `:709-731` (wiring), `:1047-1059,1517-1519` (CLI).
- Partial-unfreeze logic (the L16-must-be-trainable trap): `scripts/train_mem_space_dolmino_cpt.py:551-582`.
- NOLEAK launch (unfreezes from 24 — would freeze L16): `scripts/launch_mem_space_fifo_b25_chunk512_NOLEAK_diskB.sh:34`.
- History: trained-selector death (gist random precision): `status/RUN_REGISTRY.md:1240-1259`; reader-attn 55% vs gist random `:1256-1258`; reader-attn recall 72/78/58/40% `:1290`; T2→BABILong overfit/no-transfer `:109,207-214`.
