# Pre-registration — separating the heal-vs-no-heal confound with a healed non-OLMo arm

**Written 2026-08-12, BEFORE any GPU was allocated.** Task: close the residual
to-do at `paperC/README.md:348-350` ("the heal-vs-no-heal confound is now the
single biggest open question in the direction ... it still needs healed non-OLMo
arms, i.e. real training").

The point of pre-registering is that the arm's identity (family, depth, budget)
and the read-out are fixed *before* the numbers exist, so the arm cannot be
re-chosen post-hoc to make the confound resolve in a convenient direction.

---

## 1. The defect being closed

`evidence/POWER_WALL_VERDICT.md` produced two findings that point opposite ways on
the same benchmark (MMLU-Pro, `n = 12032`, letter floor `always-A 0.116606`):

| cell | regime | Δ vs floor | boot p | verdict |
|---|---|---|---|---|
| `olmo2_7b/keep8` @ step121000 | prune → **heal 121k steps** | −0.116 pp, CI95 `[−0.698, +0.465]` | 0.7118 | AT floor; interval **excludes** −1.389 pp |
| `llama2_7b/k8` | eval-time truncation, **no heal** | −0.914 pp | 0.0168 | **BELOW** floor |
| `qwen3_8b_base/k8` | eval-time truncation, **no heal** | −0.881 pp | 0.0362 | **BELOW** floor |

Both are powered (CI95 half-width 0.58 / 0.74 / 0.82 pp against MMLU's own
−1.389 pp). So the disagreement is not a power artefact. Two explanations survive:

- **H_heal** — healing lifts a damaged arm back *up to* its floor; un-healed
  damage sits *below* it. Then "significantly below floor" is a property of the
  **un-healed** regime, and `keep8`'s AT-floor reading is the healed regime
  behaving differently, not a benchmark effect.
- **H_family** — OLMo-2 differs from Llama-2/Qwen3 for reasons unrelated to heal.

Every non-OLMo cell in the paper is un-healed and every OLMo-2 cell is healed, so
**regime and family are perfectly collinear across all 21 cells** and neither
hypothesis is currently identifiable. The paper already flags this in three
places (`README.md` lines 109-110, 264-266, 305-307). One healed non-OLMo arm
breaks the collinearity.

## 2. What is deliberately NOT being matched, and why

The naive move is "train Qwen3 keep8+fresh2 to 121k steps and compare to
`olmo2_7b/keep8`". That is wrong on the depth axis, and the reason is arithmetic:

| | OLMo-2-7B | Qwen3-8B |
|---|---|---|
| layers | **32** | **36** |
| `keep8` as absolute depth | 8/32 = **25.0%** | 8/36 = **22.2%** |
| depth-fraction-matched to OLMo-2 `keep8` (25.0%) | — | **9/36 = 25.0%** |

`keep8` therefore means two different things in the two families, and there is no
integer keep-depth for Qwen3 that is simultaneously absolute-matched and
fraction-matched.

**Decision: match the ABSOLUTE keep-depth (`keep_front = 8`), not the fraction.**
Justification, in order of weight:

1. **The comparison already in the paper is absolute-matched.** All 15 non-OLMo
   MMLU-Pro cells were produced by `load_truncated_any_family(..., keep_front_layers=N)`
   with `N ∈ {8,10,12,14}` — a literal front-N slice, ignoring each family's total
   depth (`scripts/eval_olmo2_probe2_ppl.py:175-228`, verified). `qwen3_8b_base/k8`,
   the specific cell whose verdict disagrees with `olmo2_7b/keep8`, is a **front-8**
   Qwen3. If I healed a front-9 Qwen3 instead, the healed arm would no longer be
   comparable to the un-healed Qwen3 cell it is supposed to be paired with, and I
   would have swapped the heal confound for a **depth** confound — which is exactly
   the failure mode this pre-registration exists to prevent.
2. **The contrast that identifies the confound is within-family.**
   `qwen3_8b_base/k8` (un-healed) vs `qwen3_8b/k8+heal` (healed) holds family,
   tokenizer, benchmark, floor, and keep-depth fixed and varies **only** heal. This
   is a cleaner identification of H_heal than any OLMo-2-vs-Qwen3 contrast, because
   it does not require the two families to be commensurable at all.
3. Fraction-matching would only matter if the claim were about relative depth. It
   is not; it is about whether heal moves an arm's position relative to its floor.

**Consequence I accept:** this arm does *not* let me say "Qwen3 at OLMo-2's
relative depth behaves like OLMo-2". It lets me say "healing a front-8 Qwen3
moves / does not move it off its floor". That is the question the confound poses.

## 3. The arm

| field | value | why |
|---|---|---|
| base | `models/Qwen3-8B-Base` | ⚠️ **NOT** `Qwen--Qwen3-8b`, which is Instruct — see §4 |
| `keep_front_layers` | **8** | absolute match to the `qwen3_8b_base/k8` cell (§2) |
| `n_fresh_layers` | **2** | matches all four OLMo-2 healed arms (`keep{8,10,12,14}fresh2`) |
| total depth | 10 layers, 3.175B params | |
| heal-step target | **121000** (primary read-out), milestones from 5k | equals `olmo2_7b/keep8`'s scored step exactly (§5) |
| `max_steps` (cosine horizon) | **200000** | see §5 — horizon, not stopping point |
| eff_bs | **128** = bs2 × accum8 × 8 ranks | matches OLMo-2 keep8 (`bs16×1×8`) and the historical Qwen3 armB |
| seq_len | 2048 | matches both |
| lr | fresh 1e-4 → 1e-5, inherited 2e-5 → 2e-6, warmup 150 | trainer defaults, matches OLMo-2 keepN launcher |
| data | `data/slimpajama_chunks_2048_qwen3.npy` | see §6 — the honest limitation |
| `--eval_interval` | **flag does not exist in this trainer** | see §7 |

## 4. Asset correction (verified, contradicts the task brief and paperB/TODOList)

I was told `models/Qwen--Qwen3-8b` "IS the Qwen3-8B base". **Measured, it is not.**

| dir | `eos_token_id` | `max_position_embeddings` | verdict |
|---|---|---|---|
| `Qwen3-8B-Base` | **151643** (`<\|endoftext\|>`) | 32768 | ✅ base |
| `Qwen--Qwen3-8b` | **151645** (`<\|im_end\|>`) | 40960 | ❌ **Instruct** |

This reproduces `paperC/README.md:184` and the warning already in
`scripts/_run_mmlu_pro_letter_content_8gpu.sh:134-137`; the criterion is
`eos_token_id` + ctx length, **not** the presence of a `chat_template` (both have
one, 4116 vs 4168 chars). `tokenizer.json` md5 differs between the two dirs
(`3f99a31…` vs `6423133…`) while `vocab.json` and `merges.txt` are **identical**,
so the *encoding* of ordinary text is the same — verified: both encode a test
sentence to the same ids. Only the special-token block differs.

**Therefore the existing `qwen3_minarch_*` arms are unusable for paperC for a
second, independent reason** beyond their short budgets: all of them record
`base_model_path: .../Qwen--Qwen3-8b` in `arch_meta.json` (verified on all 5), i.e.
they healed the **Instruct** model. Under paperC's `chat_template=False` protocol
(`memory/paper-eval-chat-false-mandatory`) an Instruct arm is not a valid base
arm, and it would not be paired with the `qwen3_8b_base/k8` cell that used
`Qwen3-8B-Base`. This is a stronger reason to train fresh than the budget
argument in the task brief.

## 5. Why `max_steps=200000` but read out at 121000

`max_steps` sets the **cosine horizon**, not the stopping point. OLMo-2 `keep8`
was launched with `max_steps=200000` and *scored* at step 121000, where its LR
was 8.09e-06 on the way down (verified in `logs/olmo2_7B_keep8fresh2.log`). To put
the Qwen3 arm at the same point on the same schedule shape, it must be launched
with the same horizon and read out at the same step. Launching with
`max_steps=121000` would make step 121000 the *end* of a cosine decay (LR ≈ min),
a different training state at the same step count. Milestones every 5000 steps are
retained so intermediate heal budgets are also scoreable — the confound question is
partly about *how much* heal is needed.

**121000 steps × 128 × 2048 = 31.72B tokens.**

## 6. Honest limitation: the corpus is NOT matched, and cannot be

OLMo-2's heal used **Dolmino** (`/dev/shm/dolmino_now15b.npy`, 31.7B OLMo-2
tokens). Qwen3 cannot consume it — the file holds OLMo-2 token ids (max id 100257,
vocab 100352) and Qwen3's vocab is 151936; the ids are not transferable. Producing
a Qwen3-tokenized Dolmino would require the raw Dolmino text, which is **not on
either disk** (verified: `data/dolmino_olmo2_shards/` and `data/dolmino_stage_now/`
contain only pre-tokenized `.npy`; no jsonl/parquet/gz source).

So the Qwen3 arm heals on **SlimPajama** (`slimpajama_chunks_2048_qwen3.npy`,
1127824 rows = 2.31B tokens). At eff_bs=128 that is **8814 steps per epoch**, so
121000 steps = **13.7 epochs** of a 2.31B-token corpus, against OLMo-2's 1.00
epoch of 31.7B.

This is a real, unremovable asymmetry and it is **the main threat to this arm's
interpretation**. Two mitigations, both taken:

- **Extend the corpus.** `data/slimpajama-6b/` raw parquet is on disk (14 GB, 48
  train shards) and the existing npy used only a subset. Re-tokenizing all 48
  shards with the **Base** tokenizer raises the corpus and cuts the repetition
  factor. This is CPU-only (0 GPU, 384 cores available) and runs concurrently with
  training, so it costs nothing on the critical path. If it completes before the
  arm reaches 121k, the run is restarted on the larger corpus; if not, the 13.7×
  repetition is reported as-is.
- **Report it.** Any verdict from this arm must carry "healed on 13.7 epochs of
  2.31B SlimPajama tokens vs OLMo-2's 1.0 epoch of 31.7B Dolmino tokens".

⚠️ Note the existing npy was built with the **Instruct** EOS (151645) as the
inter-document separator — verified: 100444 occurrences of 151645 and **zero** of
151643 in a 102M-token sample. For a Base arm the separator should be 151643.
The re-tokenization fixes this; the existing file is used only as the fallback,
with the mismatch declared.

## 7. `--eval_interval 0` is not applicable to this trainer

The instruction to set `--eval_interval 0` (CLAUDE.md: inline BABILong eval
desyncs DDP ranks and SIGABRTs on the 30-min NCCL watchdog) was checked against
the actual trainer: `scripts/train_qwen3_arch_probe2.py` contains **zero**
occurrences of `eval_interval`, `quick_eval`, or `babilong`. The flag does not
exist and passing it would be an argparse error. `scripts/train_olmo2_arch_probe2.py`
also has zero. **The hazard is structurally absent — there is no inline eval to
disable.** All eval is offline on checkpoints, which is what the rule wants.

## 8. Pre-registered read-out

Scored with the **unchanged** harness — `scripts/_run_mmlu_pro_letter_content_8gpu.sh`
plus `paperC/code/mmlu_pro_power_nulls.py` — at `MAXLEN=2048`, `add_bos 0`,
`desc_style none`, `chat_template=False`, 8 shards, asserting `n_trunc == 0` and
`n_nan == 0` per shard. The letter floor is `always-A 0.116606` and is asserted
bit-identical across all 21 existing cells, so the healed arm's floor is **fixed
in advance** and cannot be re-derived to taste.

The eval path already exists and needs no new code: `load_pruned_model` →
`build_pruned_shell` has a Qwen3 sibling (`scripts/eval_qwen3_probe2_ppl.py`,
`Qwen3Config`/`Qwen3ForCausalLM`, same builder contract). ⚠️ But
`scripts/eval_olmo2_mc_letter_content.py` imports `load_pruned_model` from the
**OLMo-2** module, which hardcodes `Olmo2Config`, so scoring a Qwen3 *pruned
checkpoint* needs a one-line family dispatch. That is a known, small, CPU-testable
change — **not** a research risk — and it is listed as a follow-up, not a blocker,
because the arm must exist before it can be scored.

Primary contrast, decided now:

| | comparison | reads on |
|---|---|---|
| **P1** | `qwen3_8b_base/k8` (un-healed, −0.881 pp, p=0.0362, **BELOW** floor) vs `qwen3_8b_base/k8+heal@121k` | H_heal, within-family, everything else held fixed |
| **P2** | `qwen3_8b_base/k8+heal@121k` vs `olmo2_7b/keep8@121000` (−0.116 pp, AT floor) | H_family, at matched heal steps and matched absolute depth |

Outcomes, committed in advance:

- **H_heal supported** if the healed Qwen3 arm moves UP to AT-floor (Δ CI95
  covering 0, as `olmo2_7b/keep8` does) while its un-healed twin is significantly
  below. Then "significantly below floor" is a property of the **un-healed**
  regime, `README.md`'s narrowing (b) is resolved in favour of regime, and the
  heal-vs-no-heal caveat can be dropped from the three places it appears.
- **H_family supported** if the healed Qwen3 arm stays significantly BELOW floor
  despite 121k heal steps. Then heal is not the separating variable and the
  difference is a family property — which would *strengthen* the existing
  per-family scope discipline rather than weaken it.
- **Neither / ambiguous** if the healed arm lands significantly ABOVE its floor
  (which `qwen3_8b_base/k14` already does un-healed, +0.233 pp p=0.0192, so this
  is a live possibility, not a strawman). Report as a third regime; do not force it
  into a binary.

**No outcome here rescues a retracted claim.** In particular this arm cannot
revive "letter is a family-general step function" (retracted 2026-08-10) or "k14
is the last arm above its floor" as a family-general ordering — those were killed
for reasons independent of heal.

## 9. What this arm will NOT resolve, even if it succeeds

Stated in advance so a successful run is not over-read:

1. **n = 1 family, 1 depth.** It de-confounds heal at (Qwen3, front-8). It does
   **not** establish that heal is the separating variable in Llama-2 or Llama-3, nor
   at k10/k12/k14. The full de-confounding is 15 healed arms; this is 1.
2. **The corpus stays unmatched** (§6). A skeptic can always attribute a
   difference to SlimPajama-vs-Dolmino or to 13.7-vs-1.0 epochs rather than to
   heal itself. Only a Qwen3-tokenized Dolmino would close that, and the raw text
   is not on either disk.
3. **Relative depth is untested** (§2). front-8-of-36 is not front-8-of-32.
4. **It says nothing about the null-calibration methodology claim**, which is
   paperC's actual contribution and does not depend on this at all. This closes a
   *scope* defect in one empirical leg.
5. **`n_fresh=2` + differential LR is a design choice, not a neutral "heal".** A
   different heal recipe (freeze-front, distill, from-scratch) could give a
   different answer; OLMo-2 has those control arms and Qwen3 will not.
6. ⚠️ **Differential LR must not be claimed for the OLMo-2 side of P2.** Verified
   in the logs: `olmo2_7B_keep8fresh2.log` and `keep14fresh2.log` show **only**
   `inh_decay` + `inh_nodecay` groups and **no fresh group** — the `module.`
   prefix-strip fix (`train_olmo2_arch_probe2.py:438-439`) postdates them, so those
   arms trained at a **uniform 2e-5**. `train_qwen3_arch_probe2.py:275-286` has the
   **same missing strip** and `build_param_groups` runs **after** DDP wrap
   (line 569 wrap, line 594 build), so the Qwen3 arm will also log only `inh_*`
   groups at a uniform 2e-5. **This is a bug-for-bug match, which is what
   comparability requires here** — but the paper must not describe either side as
   using differential LR. The `--lr 1e-4` on the command line is a **no-op** for
   both. (`keep14fresh2_seed1234`, launched after the fix, *does* show
   `fresh_decay 815.8M` — so it is NOT comparable to keep8/keep14 on this axis.)

## 10. Compute decision: 8 cards, not 16

CLAUDE.md prefers combining two same-disk nodes into 16-card DDP when ≤3
trainings are pending. All three H20s are same-disk (zwfy6) so 16-card is legal.
**I am launching on 8 cards anyway**, because the measured scaling for *this exact
trainer and arm* is bad:

| config | s/step | source |
|---|---|---|
| 1 node, 8 ranks, bs2×accum8, eff_bs 128 | **7.59-7.60** | `logs/qwen3_armB_200k_1node.log` steps 20040-36500 |
| 2 nodes, 16 ranks, bs2×accum4, eff_bs 128 | **6.91** | `logs/qwen3_armB_200k_node1.log` step 20040 |

Doubling the cards bought **1.10×**, not 2× — at eff_bs 128 held fixed, halving
accum to 4 leaves too little local work to hide the cross-node all-reduce
(plain DDP all-reduces the full 3.9B fp32 gradient every boundary; bond1 TCP, IB
disabled). The historical 2-node run also died at step 20040 on a TCPStore
heartbeat failure, i.e. it added a failure mode. 1.10× is not worth doubling the
blast radius on a multi-day run.

**Better use of the same 24 cards, given they are all mine:** 8 for the arm,
and the other 16 stay free for the offline MMLU-Pro scoring of milestones (which
is 8-GPU sharded and is the actual next bottleneck) plus the CPU re-tokenization.
Per CLAUDE.md the DDP-not-FSDP note applies: this is plain DDP
(`find_unused_parameters=False`, no sharding), so adding ranks does **not** reduce
per-rank memory — 16 cards would not have enabled a bigger local batch either.

## 11. Per-rank memory, computed before launch

fp32 master weights + fp32 AdamW (2 moments) ⇒ ≈ 4 × params bytes static:
3.175B params → **12.7 GB** static. Measured activation+workspace for the
14-layer `armB_f12k2` arm at bs2/seq2048/grad-ckpt-on was **96.0 GB maxmem** on a
97.8 GB H20 — that arm is 3.947B params, so this 3.175B / 10-layer arm has ~3.1 GB
more headroom. The measured `k8f2` point is exact and directly applicable:
**77.5 GB at bs2, world_size 6** (`logs/qwen3_k8f2_frontier.out`) — same depth,
same fresh count, same seq_len.

Chosen: **bs2 × accum8 × 8 = eff_bs 128**, `--gradient_checkpointing 1`.
Expected maxmem ≈ 78-80 GB / 97.8 GB = **80-82%**, satisfying CLAUDE.md's ≥80%
target. bs4 would exceed the card (the 2-node run needed accum4 at bs2 to stay
under). ⚠️ maxmem is `torch.cuda.max_memory_allocated()` (allocator, not
`nvidia-smi` reserved), so it under-reports total footprint; 80-82% allocated is
the right target and leaves the NCCL buffer room.

## 12. Node assignment and checkpoint volume

- Training: **`.104`** (8×H20, verified 0 MiB / 0% before launch).
- `.73` and `.82` stay free for scoring + the CPU re-tokenization.
- ⚠️ **Checkpoint volume is a real risk**: `qwen3_minarch_k8f2_frontier/step19000.pt`
  is **38 GB** (fp32 weights + fp32 AdamW state). At `save_every 500` over 121000
  steps that is 242 saves ≈ **9.2 TB**, and zwfy6 has **3.4 TB free**. Rotation is
  therefore mandatory: `--keep_last_n 3 --milestone_every 5000 --keep_milestones 8
  --keep_steps 121000`, bounding the arm at ≈ 11 × 38 GB ≈ **420 GB**.
- ⚠️ The zwfy6 copy of `train_qwen3_arch_probe2.py` was **older than wzc1's and had
  no rotation support at all** (verified: 42-line diff, missing the entire
  `ckpt_rotation` import and the `rotate_checkpoints` call, and missing
  `--seed`/`DistributedSampler(seed=...)`). Launching the stale copy would have
  filled the disk and silently ignored `--seed`. Both `train_qwen3_arch_probe2.py`
  and its dependency `train_semantic_bottleneck_1b.py` were `scp -O`'d and md5-verified
  identical before launch.

## 13. Falsifiability / kill conditions

Committed in advance, so the run is not kept alive by sunk cost:

- **kill** if loss diverges (NaN/Inf) or ppl > 100 after warmup (CLAUDE.md PPL
  ladder: >100 means the LM is broken, not mistuned).
- **kill** if the 5-check transplant sanity assert fails (it is a hard assert; a
  failure means the arm is not the architecture claimed).
- **kill** if measured s/step implies the 121k read-out cannot be reached in the
  time available; downgrade to the largest milestone that IS reachable and report
  the reduced budget honestly rather than reporting a short arm as "healed".
- **do not kill** for slow convergence or an unfavourable Δ — those are results.
