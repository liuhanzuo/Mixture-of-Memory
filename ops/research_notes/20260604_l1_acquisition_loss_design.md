# L1 Memory-Acquisition Loss — Design Report

**Date:** 2026-06-04
**Author:** general-purpose-9 (researcher)
**Premise (user, confirmed, do NOT relitigate):** In the current objective the only
load-bearing term is `lm_loss`. At `chunk_size=1024` with full local attention +
`bptt_window=2`, `lm_loss` is driven down **without using memory** — memory is off
the critical path, so the gradient toward "use memory well" is ≈0 and the model
takes the lazy solution (uniform routing, slots don't carry content). This is an
**objective** problem, not a hyperparameter problem. We need a loss that makes L1
(slot memory) *load-bearing*.

---

## 0. Evidence the premise is correct (grounding, not relitigating)

Three independent signals all say "memory contributes nothing the LM needs":

1. **Toy passcode (memory-is-the-only-way task):** with the legacy prepend path
   ON but **no routing supervision** (`toy_e2_aux_off`), after 800 steps
   `retrieval_exact_acc = 0.000` while `top1_sim` and `chunk1to2_overlap≈0.80`
   look "fine". The router even points back to the written slots, yet the answer
   is not recoverable → the **write path stores nothing decodable**, and pure LM
   loss never fixes it. (`logs/toy_e2_aux_off.log`)
2. **Toy with routing-supervision CE (`route_aux`) ON** (`toy_e2_aux_on`):
   `retrieval_exact_acc` rises 0.000 → ~0.19–0.25, `slot_norm_delta` grows
   1.10→1.74. The ONLY change was an explicit routing target. → direct
   supervision is what moves the gold metric. (`logs/toy_e2_aux_on.log`)
3. **8B `e5_route_aux` (real run, step 565):** `lm` falls to ~2.0–2.7 normally
   while `top1_sim_mean≈0.07` (uniform floor 1/128=0.0078, so only ~9× floor)
   and `route_aux≈3.0` is NOT dropping. `l3iso_noL3` (route_aux=0) is worse:
   `top1_sim≈0.02`, ~2.5× floor, `usage_ent≈0.96` (near-uniform load). LM loss
   trains fine in BOTH; routing quality is decoupled from it.

The GRAD_PROBE (`toy_e1_decoupled_on`) quantifies *why*: the selector weights'
gradient is **~95% from the aux losses, ~5% from lm_loss**
(`lm_Q_grad≈0.4–4.5` vs `aux_Q_grad≈14–30`). With `use_decoupled_read=True` the
H→L1 prepend attention is masked (`layer.py:160-161`, `_build_extended_attn_mask`
`mask_h_to_l1`), and the decoupled read attends over ALL slots with its own
softmax and **never touches `idx`/`scores`** (`selector.read`, `layer.py:1200`).
So the only LM→`scores` path is severed, and the surviving aux losses
(`load_balance` 0.01 + `entropy` 0.001) actively **push toward uniform**. Net:
the objective rewards uniform routing.

---

## 1. Inventory of existing-but-disabled candidates

| mechanism | weight in run | what it actually optimizes | "store" or "retrieve"? | why it can't make memory load-bearing |
|---|---|---|---|---|
| `route_aux` (train `_compute_route_aux`, script:558-574; folded in tbptt:1085-1090 / babilong:1185-1188) | **1.0 (E5) / 0** | CE pushing chunk *i*'s grad-bearing `scores2` onto the slot indices chunk *i-1* **wrote into** (`prev_idx1`) | **retrieve: addressing only** | Supervises *where to look*, with the target being "wherever you happened to write last chunk". It does NOT require the slot to **contain** the info, nor that reading it **helps the LM**. It can be satisfied by self-consistent but content-empty routing (toy: overlap 0.80 yet acc 0.00). Necessary scaffolding, not sufficient. |
| `l_recon` (recon_decoder.py; layer.py:1476-1479) | **0** | MSE(decode(`M_write`), stopgrad(L3 summary)) — a tiny 1-block cross-attn must rebuild the chunk's L3 summary from the slots just written | **store: encodability** | Forces *writes to be decodable into the L3 summary*, but (a) target is the **L3 summary**, an internal compression, not the **answer tokens** — a slot can reconstruct a generic summary without holding the needle; (b) it never couples to **retrieval at read time** or to the LM. Best of the three for "store", but optimizes a proxy target and is read-decoupled. |
| `l3_diversity` / `q_multi_diversity` (selector / l3_pool; layer.py:1462-1469) | **0** | pushes L3 summary tokens / post-projection routing queries apart (anti-collapse regularizers) | neither (anti-collapse) | Only prevents key/query **collapse**; says nothing about content correctness or usefulness. Pure regularizer. |

**Summary of the gap.** `route_aux` trains "取得对" (address right) but with a
weak self-referential target; `l_recon` trains "存得下" (store encodably) but
against a proxy target and decoupled from read; nothing trains **"取了有用" —
that retrieving the slot actually lowers the answer's CE.** That missing link is
exactly what makes memory optional. The fix must put memory **on the answer's
critical path**.

---

## 2. Proposed L1-acquisition losses (ranked)

### Ranking (essentiality × implementation cost) — do A first

1. **(A) memory-only LM auxiliary path** — directly makes memory load-bearing.
   `confidence: high`. Medium cost (~30 lines, one extra forward).
2. **(C) re-open `l_recon` + keep `route_aux`** — cheap bridge, run NOW in
   parallel as the safe baseline while A is implemented. `confidence: med`.
   Near-zero cost (weight flags only).
3. **(B) provenance retrieval supervision** — strongest *addressing* target but
   blocked on data plumbing + only helps BABILong. `confidence: low-med`. High
   cost. Defer.

---

### (A) Memory-only LM auxiliary path  ★ build this first

**Idea.** Add a second LM forward on the **target/answer chunk** in which the
live tokens are **denied local context** and can only lower their CE by reading
memory. Compute CE on the same answer labels; add it as `l_memlm`. Because the
*only* information route to the answer is the slot read, gradient is **forced**
through (routing → writeback → slot content → read), making all of them
load-bearing. This is the real-path generalization of the toy task (where chunks
are physically separate forwards), applied inside Dolmino/BABILong.

**Where local context is "removed" — the minimal implementation.**
The cleanest lever already exists: **SWA + `use_decoupled_read`**.

- `_build_extended_attn_mask(..., swa_window=W)` (layer.py:142-150) already builds
  a sliding-window-causal H×H block. Set `W` very small (e.g. 1 or a few tokens)
  for the aux path → each answer token sees almost no local history.
- With `use_decoupled_read=True`, the memory READ is the standalone
  `decoupled_read.read(hidden, slots, slots)` (layer.py:1200) gated by
  `inject_gate g` (layer.py:1216-1217) — it is **independent of the SWA mask**,
  so memory remains fully available while local attention is choked.
- Net: on the aux path, answer-token logits ≈ f(memory read). Lowering CE
  *requires* the slot it reads to contain the answer and the router to fetch it.

This is strictly better than "drop local KV" hacks because it reuses the audited
mask builder and the already-zero-init decoupled read (so adding the path is
behavior-preserving at init: `out_proj=0` ⇒ step-0 `l_memlm` = the no-memory CE,
no shock).

**Forward / gradient wiring (concrete落点).**

Add a config + arg `mem_lm_aux_weight` (default 0) and `mem_lm_swa_window`
(default 1). New step helper, called once per micro-step right after the normal
target forward, sharing the *same already-written memory bank* (do NOT reset):

In `scripts/train_mem_space_dolmino_cpt.py`:

- **New fn `_memlm_aux_forward(model, target_ids, device, swa_window)`** near
  `dolmino_train_step_tbptt` (script ~1118). Pseudocode:
  ```
  # memory bank currently holds the writes from the just-finished window
  # (call BEFORE _detach_banks of the final window, OR re-run context no_grad
  #  then this — see "graph" note). Toggle the layers to memory-only mode:
  for w in mem_layers: w._memlm_swa = swa_window   # new transient attr
  out = model(input_ids=target, labels=target, use_cache=False)
  for w in mem_layers: w._memlm_swa = None
  return out.loss   # this is l_memlm (CE on answer, memory-only route)
  ```
- **In `layer.py` forward**, read the transient override (≈3 lines, near the
  mask build at layer.py:1060): if `getattr(self,'_memlm_swa',None) is not None`,
  use `swa_window=self._memlm_swa` AND force the decoupled-read branch active
  (require `use_decoupled_read=True` for this experiment so the read path exists
  while H→L1 prepend is masked). Everything else unchanged.
- **In the loop** (script ~1642 for dolmino, ~1660 for babilong), after the main
  loss:
  ```
  if args.mem_lm_aux_weight > 0:
      l_memlm = _memlm_aux_forward(model, target_ids, device, args.mem_lm_swa_window)
      (args.mem_lm_aux_weight * l_memlm / scale).backward()   # or fold into window_loss
  ```

**Gradient flow.** `l_memlm` → answer logits → `next_hidden` (layer.py:1215-1217)
→ `g * decoupled_read_out` → `decoupled_read.read(hidden, read_slots, read_slots)`
where `read_slots = slot_to_hidden(slots)` (layer.py:1196-1199) → `slots` (the
bank) → the write graph from the context chunks (`O_mem_slot`, gates,
`hidden_to_slot`). Because the answer cannot be predicted from local context (SWA
choked), the **only** CE-reducing direction is "route to the slot that holds the
answer and have written it correctly" → forces routing + writeback to become
content-addressable. This is the load-bearing signal the LM loss alone never
gives.

**Two graph-handling options (pick by VRAM):**
- *Cheap (recommended first):* re-stream context chunks under `no_grad` to refill
  the bank (like `dolmino_train_step` script:975-978), `_detach_banks`, then run
  the memory-only target forward with grad. Gradient reaches the **read + slot
  content as it currently stands** but not back into the context-writes (write is
  detached). This already makes the *read/route* load-bearing and is a strong
  first cut. ~1 extra forward, negligible VRAM.
- *Full (later):* keep the context-write graph attached (don't detach) so
  `l_memlm` also back-props into the writes. Costs ~bptt_window× activations on
  the aux forward; gate behind a flag.

**Risk.** (1) Requires `use_decoupled_read=True` (E5 uses it; consistent). (2) If
SWA is *too* aggressive (W=0) the normal-LM token at position 0 has no context at
all and CE is huge/noisy — start W=1–8, treat as a curriculum knob. (3) Extra
forward ~1.5–2× step time on the aux micro — acceptable, gate by weight.
**Expected effect:** `top1_sim` rises well above floor AND BABILong ≥2k accuracy
moves off 0.0% (the gold check), because for the first time the answer's loss
depends on retrieval. `confidence: high` that it produces a real
memory-dependent gradient; `med-high` that it alone clears the BABILong gate.

---

### (B) Direct retrieval supervision from BABILong provenance

**Idea.** For BABILong needle tasks we know which supporting fact (sentence)
contains the answer; map it to the chunk that held it and supervise the router
(CE/contrastive) to select the slot(s) that chunk wrote.

**Data-pipeline reality check (blocking).** `BABILongTrainDataset`
(`babilong_dataset.py`) consumes only `sample["input"]`, `sample["question"]`,
`sample["target"]` via `get_formatted_input`; it does **not** surface the
RMT-team/babilong supporting-fact spans, and we tokenize the fully-formatted
prompt so we lose the char offsets of the needle. The HF `RMT-team/babilong`
rows do not ship per-sample `supporting_facts` offsets in the splits we load.
So **we currently cannot get provenance without re-deriving the needle position**
(string-search the answer/fact substring in `sample['input']`, map char→token→
chunk index). That is doable but non-trivial and BABILong-only.

**Verdict.** Strongest *addressing* target in principle, but: (a) needs new data
plumbing (substring→token→chunk provenance in the dataset + collate), (b) helps
only the 15% BABILong mix, not Dolmino, so it does not fix the general objective.
**Defer** until (A) is in. `confidence: low-med`. High cost. If pursued, the落点
is `babilong_dataset._tokenize_with_answer_mask` (add `needle_chunk_idx`) +
thread it as a `route_aux` target instead of `prev_idx1`.

---

### (C) Re-open `l_recon` (+ keep `route_aux`) — run NOW as the bridge

**Idea.** Zero-cost: flip `--l_recon_weight` (e.g. 0.5–1.0) so writes must be
decodable, and keep `--route_aux_weight 1.0` so routing is supervised. Together
they cover "store encodably" + "address consistently". Already fully implemented
(recon_decoder built when `l_recon_weight>0` and `use_l3_summary` on; collected
in `_collect_aux_loss` via key `"recon"`, script:590).

**Limit (why it's a bridge, not the answer).** `l_recon`'s target is the **L3
summary**, not the answer, and is **read-decoupled** — it never checks that
reading the slot lowers the LM loss. So it can raise "store" quality without
making memory load-bearing for prediction. It is the right thing to run in
parallel TODAY (free, and the toy showed `route_aux` already moves the gold
metric) while (A) is built. `confidence: med`. Near-zero cost.

---

## 3. Recommendation & sequencing

1. **Now (free, no code):** launch an arm = E5 config **+ `--l_recon_weight 1.0`**
   (keep `route_aux=1.0`). This is (C); tests whether store+address together move
   BABILong without new code. Auto-launchable.
2. **Build (A)** — the memory-only LM aux path — as the primary deliverable. It is
   the only candidate that puts L1 on the answer's critical path.落点 above:
   `config.py` + arg in train script (`mem_lm_aux_weight`, `mem_lm_swa_window`),
   ~3-line override in `layer.py` mask build, `_memlm_aux_forward` helper + one
   `.backward()` in the loop. Requires `use_decoupled_read=True`.
3. **Defer (B)** until A lands; it needs provenance plumbing and only helps the
   BABILong slice.

This matches the user's stated preference (A is the focus; B feasibility
assessed; C as transition).

---

## 4. Kill decision — `l3iso_noL3_local`

**Recommend KILL.** It is a no-L3 ablation control; at step 505 it behaves
exactly as expected (`top1_sim≈0.02`, ~2.5× floor; `usage_ent≈0.96` near-uniform)
— i.e. it confirms "pure LM loss + no routing supervision ⇒ uniform routing",
which we already know. Since the real problem is the objective (Section 0), letting
it run to 2000 yields no new information and occupies H20-1, which we want free to
launch the (C) bridge arm and then the (A) experiment. Keep `e5_route_aux_remote`
running (it's the route_aux validation and the cleanest A/B baseline). Free H20-1.
