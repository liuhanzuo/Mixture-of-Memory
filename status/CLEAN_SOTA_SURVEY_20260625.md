# CLEAN SOTA SURVEY — Slots vs FIFO, leakage-aware (2026-06-25)

> Author: researcher subagent. Read-only analysis (no GPU jobs run).
> Question answered: *"What was the best memory-SLOTS result historically, what does the FIFO-buffer approach reach now, and what is the right direction?"*
> **All scores classified CLEAN (`babilong_mix=0`) vs LEAKED (`babilong>0`), verified by grepping `Training complete` lines in `logs/`.** Only CLEAN scores are trusted as capability. For LEAKED runs, 0k-4k is VOID (memorized eval answers); 8k-32k is OOD-relatively-clean but still suspect (babilong SFT teaches QA-format/answer-extraction skill that can transfer).

---

## Leakage verification (direct from training logs)

| run | log line evidence | verdict |
|---|---|---|
| **P11 chunk512 deltarule** | `mem_space_p11_chunk512_deltarule_normreadout.log`: `babilong_mix=0.15` … `Training complete: … babilong=3034` | **LEAKED** — NOT a clean anchor |
| **FIFO b25 c512** | launch `scripts/launch_mem_space_fifo_b25_chunk512_diskB.sh` has NO `--babilong_mix_fraction` override → default 0.15. (b50 sibling log `mem_space_fifo_b50_chunk512.log`: `babilong_mix=0.15` … `babilong=1803`) | **LEAKED** |
| **FIFO b50/b100/c1024** | same family, default 0.15 (b50 log confirms babilong=1803) | **LEAKED** |
| **FIFO b25 NOLEAK** | `scripts/launch_mem_space_fifo_b25_chunk512_NOLEAK_diskB.sh:39` `--babilong_mix_fraction 0`; running on .7.53, ETA ~07:30 | **CLEAN — not yet evaluated** |
| **HARDOBJ lastchunk** (ctx7 N192, N128, …) | `HARDOBJ_lastchunk_ctx7_N192_diskA.log`: `babilong_mix=0.00` … `babilong=0` | **CLEAN** |
| **pg19 nctx7 distill** | `scripts/launch_distill_pg19_chunk512_nctx7.sh:54` `--babilong_mix_fraction 0.0` (pure pg19 data) | **CLEAN** |
| **pg19 nctx63 distill** | `distill_pg19_chunk512_nctx63.log`: `babilong_mix=0.00` … `babilong=0` (pg19_perbook data) | **CLEAN** |
| **self-study rawkv / swateacher** | `mem_space_selfstudy_rawkv_chunk512.log`: `babilong_mix=0.00` … `babilong=0` (pg19 data) | **CLEAN** |
| **distill_AB / MASS0p5** | `distill_chunk512_AB_dolmino.log`: `babilong_mix=0.00` … `babilong=0` | **CLEAN** |

Local-log census: **43 CLEAN vs 84 LEAKED** `Training complete` lines. The historical "SOTA" P11 and the entire FIFO 4-arm sweep are all LEAKED.

**Critical structural fact: there is currently ZERO clean FIFO BABILong data.** Every FIFO number on record (b25/b50/b100, c512/c1024) is leaked. The first clean FIFO point (NOLEAK b25) is still training. So the FIFO-vs-slots verdict below rests on (a) the leaked FIFO numbers minus (b) the clean same-mechanism analog (HARDOBJ last-chunk) and the historical clean prior.

---

## §1. Clean SOTA table — best SLOTS-family vs FIFO-family

Each cell: `value` + tag. **CLEAN** = trustworthy capability. **LEAKED** = void at 0k-4k, suspect at 8k-32k. **n/a** = no clean data exists.

### SLOTS family (MemorySpace 128-slot / gist / last-chunk), W0, qa5

| source | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|
| **pg19 nctx7 distill (clean SOTA)** | 75 ✅ | 73 ✅ | 51 ✅ | 29 ✅ | **19 ✅** | **16 ✅** | **9 ✅** |
| pg19 nctx63 (clean) | 61 ✅ | 73 ✅ | 47 ✅ | 27 ✅ | 14 ✅ | 12 ✅ | 9 ✅ |
| HARDOBJ last-chunk ctx7 (clean, = FIFO analog) | — | — | — | — | 13-15 ✅ | 11-14 ✅ | 8-9 ✅ |
| self-study rawkv step500 (clean) | — | — | — | — | 16-19 ✅ | 11 ✅ | 7-8 ✅ |
| ~~P11 step500~~ (74/89/81/60/48/45/44) | ❌LEAKED | ❌ | ❌ | ❌ | ⚠️48 | ⚠️45 | ⚠️44 |

**Best CLEAN slots = pg19 nctx7: qa5 8k=19, 16k=16, 32k=9.** This is the genuine project long-program ceiling. (Note P11's 48/45/44 at 8k-32k looked far better but is LEAKED — `babilong=3034` — so it is NOT a clean anchor and must not be used as one.)

### SLOTS family clean qa1 / qa2 (best clean = pg19 nctx63 step250, triple-seed confirmed)

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|
| qa1 (nctx63, clean) | 91 ✅ | 49 ✅ | 34 ✅ | 23 ✅ | 12 ✅ | 14 ✅ | 11 ✅ |
| qa2 (nctx63, clean) | 42 ✅ | 33 ✅ | 15 ✅ | 9 ✅ | 6 ✅ | 5 ✅ | 4 ✅ |

### FIFO family (方案B), W0

| task / source | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|
| **b25 c512 qa5** | 100 ❌ | 100 ❌ | 97 ❌ | 87 ❌ | 65 ⚠️ | 76 ⚠️ | 68 ⚠️ |
| **b25 c512 qa1** | 96 ❌ | 99 ❌ | 99 ❌ | 93 ❌ | 40 ⚠️ | 34 ⚠️ | 30 ⚠️ |
| **b25 c512 qa2** | 99 ❌ | 100 ❌ | 100 ❌ | 95 ❌ | 23 ⚠️ | 32 ⚠️ | 32 ⚠️ |
| b50 c1024 qa5 | 35 ❌ | 39 ❌ | 12 ❌ | 21 ❌ | 10 ⚠️ | 15 ⚠️ | 8 ⚠️ |
| **FIFO b25 NOLEAK qa5 (clean)** | — | — | — | — | **n/a** | **n/a** | **n/a** |

The only FIFO cells that aren't void-at-0k are the b25 8k-32k values (65/76/68), and those are exactly the cells under dispute.

### Anchors (external)

- **MemoryLLM teacher qa5** (the target to beat) = 47/50/45/39/39/38/**34** — confirmed in `status/SESSION_HANDOFF.md:73-74` and `RUN_REGISTRY.md:18`.
- **pg19 nctx7 clean SOTA qa5** = 75/73/51/29/19/16/9 — confirmed in `RUN_REGISTRY.md:871-872` and `HANDOFF_NEW_AGENT_20260625.md:74`. (The handoff's "75/73/51/29/19/16/9" is correct.)

---

## §2. Verdict — did FIFO beat slots on CLEAN long-docs?

**No clean FIFO measurement exists yet, so a direct head-to-head is not yet possible. But every available line of evidence says the leaked b25 "breakthrough" is ~85% leakage artifact, and clean FIFO will land at the slots/HARDOBJ level, NOT above it.**

Quantitative argument (all from verified-clean data):

1. **Same-mechanism clean analog.** HARDOBJ `--last_chunk_loss_only` (clean, `babilong=0`) is mechanistically almost identical to b25 (context chunks stream into memory no-grad, LM loss only on the last chunk). It reaches qa5 **8k=13-15 / 16k=11-14 / 32k=8-9** (`HARDOBJ_FINAL_REPORT.md:17-24`), flat across N96-N256. The leaked b25 with the same mechanism reaches **8k=65 / 16k=76 / 32k=68**.
   - **Leakage contribution: 8k +50 (≈4.5×), 16k +62 (≈5.5×), 32k +59 (≈7.5×).**
2. **Historical clean prior is airtight.** Across ~15+ verified-clean runs (pg19 nctx7/15/63, self-study, T2, HARDOBJ, distill_AB), **no clean run has ever exceeded qa5 8k≈19 or 32k≈9** (`HANDOFF_NEW_AGENT_20260625.md:72-74`). b25's 65/76/68 is 3-8× outside the entire clean distribution.
3. **Leakage fingerprint.** b25 qa5 0k=100/1k=100/2k=97/4k=87 are unprecedented saturation (clean record 0k≈80, P11 0k=74). The training `max_seq_len=2048` fully ingests the 0k/1k eval stories (memorized answers). Non-monotone hump (8k=65 < 16k=76 > 32k=68) is also unphysical for genuine long-program decay.

**Predicted clean FIFO (NOLEAK b25 W0, ETA 07:30): qa5 8k≈15-25, 16k≈12-18, 32k≈8-12 — i.e. the HARDOBJ/pg19 cluster.** If it lands there, FIFO does **not** beat the best clean slots result (pg19 nctx7 16k=16/32k=9); it roughly ties it. The handoff hypothesis ("clean FIFO ≈ HARDOBJ/pg19 level, b25 was ~85% leakage") is strongly supported and should be treated as the working conclusion until NOLEAK eval lands.

**Bottom line: FIFO's apparent long-doc win is a leakage mirage. On clean long-docs the FIFO architecture is not yet shown to beat the slots ceiling — both sit at 16k≈16, 32k≈9, and both are ~3-4× below MemoryLLM's 32k=34.**

---

## §3. The W0/W6 (pure-memory vs raw-token) gap — magnitude in CLEAN runs

The gap is large and reproduces across every clean run. It is the core unsolved problem.

| clean run | W0 (pure memory) qa5 8k/16k/32k | W6 / swa qa5 8k/16k/32k | gap @ 8k |
|---|---|---|---|
| HARDOBJ ctx (clean) | **8-15** (`HARDOBJ_FINAL_REPORT.md:8,42`) | swa6 = **50-60** | ≈ +40 (3-4×) |
| self-study step500 (clean) | swa0 16/11/7 | swa2 -/26/18 (`RUN_REGISTRY.md:130-132`) | 16k 11→26, 32k 7→18 (≈2×) |
| lr5e5 mass_coef2 (LEAKED ckpt, gap still indicative) | W0 16 | W6 **54** (`RUN_REGISTRY.md:69,73`) | +38 |

**Clean magnitude: W0 ≈ 8-19 vs W6/swa6 ≈ 50-60 at the 8k-16k mid-range — a 3-4× / +35-45pt gap.**

**What it implies (per the user's reframe in `HANDOFF_NEW_AGENT_20260625.md:90`):** For FIFO this gap should NOT exist. The buffer literally holds the last 25 chunks' hidden states; the W6 SWA window (last 6-7 chunks) is a strict subset of the buffer. If W0 reads the same chunks W6 sees, W0 should equal W6. The persistent gap therefore means **our hidden representation is lossy/unusable, not that SWA gets an "open-book" advantage.** Prime suspect = **position collapse** (`src/memory/mem_space/layer.py:1244`, confirmed: legacy path forces every buffer token to RoPE pos-0, lines 1308-1326 — "all prefix tokens at pos-0 (legacy, current)"), turning the buffer into a bag of position-less vectors while W6's raw tokens carry normal relative positions. Secondary suspect = staleness (hidden computed when the chunk was "current," never having seen the query).

Caveat: the older project framing (`RUN_REGISTRY.md:184-194`, "杠杆3" arm B) concluded the SWA gap was a *retrieval* (router picks wrong slot) problem, not readout — but that was on the **slots** architecture with a trained selector. For **FIFO** there is no router (full attention over the buffer), so the retrieval-vs-readout dichotomy doesn't apply; the FIFO gap is cleanly attributable to representation/position, which is exactly why FIFO is the better testbed for closing it.

---

## §4. Direction assessment — ranked by expected payoff / cost

The two cheapest, highest-information moves are already coded as **zero-training eval probes** (commit eddb4f1: `--fifo_pos_mode {packed,real}`, `--fifo_keep_set_mode flat_readerattn`, `--fifo_keep_all_buffer`). Both should be run on the **clean NOLEAK ckpt** the moment it lands.

| rank | direction | mechanism / evidence | cost | payoff | confidence |
|---|---|---|---|---|---|
| **1** | **Reader-attn keep-set** (SnapKV-on-chunks; keep-all, attend-few) | Directly attacks the proven dilution wall: Method A diagnosis showed full-haystack (16 chunks) = **0%** but isolated needle chunk = **97.5%** (`RUN_REGISTRY.md:1261-1267`). Reader-native q·k is the ONLY working selector: **55% needle precision = 8.8× random** vs all trained selectors at random (`RUN_REGISTRY.md:1254-1259`, :117). Zero-training probe P2/P3 exists. | LOW (zero-train probe on existing ckpt; then ~3000-step retrain if positive) | HIGH | **med-high** |
| **2** | **Position-fix retrain** (packed RoPE at train+eval) | Closes the W0/W6 gap's #1 suspect: `layer.py:1244` collapses buffer to pos-0. Zero-training probe P1 (`--fifo_pos_mode packed`) discriminates this directly. If P1 lifts clean W0 8k-32k → position is the driver → retrain with packed positions. | LOW to probe, MED to retrain | HIGH if P1 fires; LOW if not | **med** |
| **3** | **prediction-CE compression** (retry L3-recon with LM-CE loss, not MSE) | Reframes why all L3 token-recon aux were REJECTED (`v_prediction_not_reconstruction_2026-06-25.md`): wrong loss (reconstruction conflicts with prediction). Conceptually sound (Information Bottleneck: keep I(X;Y_pred), drop I(X;X)). But speculative, no probe, and the entire slots/compression line has a long negative-result history. | MED-HIGH (new loss + full retrain) | MED | **low** |
| **4** | **HNST tree** (hierarchical reader-native navigation) | Elaboration of #1: B-ary tree, 1-of-B per level navigation (avoids flat 1-of-64 selector death), leaf=raw hidden, internal=slot compression (`vN_HNST_tree_hidden_memory_2026-06-25.md`). Novel but complex; v2 needs training. Only worth it if flat reader-attn (#1) shows signal but hits the "lost early needle"/flat ceiling. | HIGH (v1 zero-train probe, but v2 trains L3+gist) | MED-HIGH if #1 plateaus | **low-med** |

**Reasoning for the ranking:** #1 and #2 dominate because both have (a) a concrete mechanistic target, (b) a zero-training probe already deployed, and (c) the only positive evidence in the project (reader-native attn 55% precision; the dilution 0%→97.5% jump). #1 edges out #2 because dilution has *direct* positive evidence (isolation recovers 97.5%) whereas position is a strong-but-unconfirmed suspect. #3 and #4 are deferred: #3 is a single untested hypothesis against a wall of compression negatives; #4 is strictly a follow-on to #1.

---

## §5. Concrete recommended next experiment (must be `--babilong_mix_fraction 0`)

**Step A (immediate, zero-GPU-training, decisive): Run the 5-probe matrix on the CLEAN NOLEAK b25 ckpt** (`outputs/mem_space_fifo_b25_chunk512_noleak/full_model.pt`, ETA 07:30), W0, n=100, qa1/qa2/qa5 × {8k,16k,32k}. Specifically the three that matter:
- **P1** `--fifo_pos_mode packed` → tests position-collapse (direction #2).
- **P2** `--fifo_keep_set_mode flat_readerattn --fifo_keep_topk 25 --fifo_keep_all_buffer` → tests reader-attn keep-set / dilution (direction #1).
- **P3** P1+P2 stacked → if W0 → W6 level, the gap is closed = top-tier result.

This is the cheapest possible discriminator: it tells us *which* of #1/#2 drives the W0/W6 gap, on a clean ckpt, with no training. Running it on the leaked b25 (as already queued) only gives relative deltas; the clean ckpt gives the real verdict.

**Step B (the actual training experiment, contingent on Step A):** Whichever probe lifts clean W0 long-docs the most, do its retrain — and it **must** set `--babilong_mix_fraction 0` (the default 0.15 is the leakage source; verify the launch script carries the flag, as `launch_mem_space_fifo_b25_chunk512_NOLEAK_diskB.sh:39` does).
- If P1 wins → **position-fix retrain**: apply packed RoPE positions at *training* time (not just eval), `--babilong_mix_fraction 0`, FIFO b25 c512, 3000 steps. Report only 8k-32k (clean discriminator).
- If P2 wins → **reader-attn keep-set retrain**: keep-all-buffer + reader-native top-k chunk selection baked into training, `--babilong_mix_fraction 0`.
- If P3 ≈ W6 → both, combined.

**Success criterion (clean, honest):** beat the clean slots SOTA on long-docs — qa5 16k > 16 and 32k > 9 — at W0, with `babilong=0` confirmed in the training log's `Training complete` line. Anything that merely reproduces 65/76/68 with leakage is not progress.

---

### Appendix — key file:line citations
- Leakage mechanism: `scripts/train_mem_space_dolmino_cpt.py` default `--babilong_mix_fraction=0.15`; `src/memory/mem_space/babilong_dataset.py:79` (no train/test split).
- Position collapse: `src/memory/mem_space/layer.py:1244`, modes at :1308-1326, :1338-1378.
- Clean SOTA qa5: `RUN_REGISTRY.md:871-872` (pg19 nctx7), :911 (nctx63 qa1).
- HARDOBJ clean baseline: `status/HARDOBJ_FINAL_REPORT.md:8,17-24,42`.
- Dilution diagnosis: `RUN_REGISTRY.md:1254-1267`.
- Leaked b25: `RUN_REGISTRY.md:1294-1303`, `SESSION_HANDOFF.md:16-40`.
- MemoryLLM anchor: `SESSION_HANDOFF.md:73-74`.
