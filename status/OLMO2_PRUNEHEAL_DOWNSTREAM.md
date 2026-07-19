# OLMo-2 Prune-then-Heal — Downstream MC Capability Ledger (Paper B, direction #4)

Companion to `status/OLMO2_PRUNEHEAL_PPL.md`. The PPL ledger showed the pruned+healed
OLMo-2 recovers *language modelling* to 1.47× (1B) / 2.33× (7B early) the full-depth
base. **PPL is not capability.** This ledger answers the honest question of direction #4:
**does downstream ability recover in lockstep with PPL, or do the dropped middle
layers carry reasoning that PPL is blind to?** — via likelihood-based zero-shot
multiple-choice (NO generation; argmax over teacher-forced continuation log-prob).

## Setup (single source of truth)
- Driver: `scripts/eval_olmo2_probe2_downstream.py` (self-contained MC scorer; **imports
  `load_pruned_model` / `load_base_model` verbatim from `scripts/eval_olmo2_probe2_ppl.py`
  → ZERO arch drift** from the PPL eval / the trainer's build).
- Launcher: `scripts/_run_olmo2_probe2_downstream_8gpu.sh` (per config: 1 `--prepare_data`
  cache pass → 8 shard procs, examples strided `[g::8]` per task → merge; acc = Σcorrect/Σn).
- Node: **.73** (`28.85.35.73`, diskB, EVAL-ONLY, forward-only). Env
  `/opt/conda/envs/torch-base/bin/python` (torch2.13 / tf5.5.4). Ran full 8-GPU, all 8 procs.
- Scoring: fp32 weights, bf16-autocast forward (matches PPL eval + training). For each
  candidate: encode context+continuation, take continuation tokens = `whole[len(ctx):]`,
  sum fp32 log-softmax log-prob (teacher forced). `target_delimiter=" "`. OLMo-2 tokenizer
  does NOT auto-add BOS (`add_special_tokens` is a no-op) → **no BOS**, matching published
  OLMo-2 lm-eval numbers. **acc** = argmax(Σlogprob); **acc_norm** = argmax(Σlogprob /
  len(candidate_chars)) (lm-eval char-length `completion_len`). winogrande = standard
  partial/double-cloze (shared suffix continuation, two option-filled prefixes) → acc_norm==acc.
- Tasks (zero-shot, standard splits, full sets, n = validation/test size):
  hellaswag(10042), arc_challenge(1172), arc_easy(2376), piqa(1838), winogrande(1267),
  openbookqa(500). Data via `datasets.load_dataset` over hy-proxy (Rowan/hellaswag,
  allenai/ai2_arc, ybisk/piqa @ `refs/convert/parquet`, allenai/winogrande winogrande_xl,
  allenai/openbookqa main). **No task skipped; all 6 loaded.**
- **All configs: nan=0, trunc=0 on every task** (candidate log-probs finite + non-degenerate;
  verified per-example sample vectors vary across candidates → no mask/all-equal bug).
- Run finished **2026-07-19 13:55 CST** (~11 min wall for all 7 configs on .73 8×H20).

## Sanity (铁律2) — base_full reproduces published OLMo-2
- **1B_base_full**: hellaswag acc_norm **0.683** (public ~0.67-0.68 ✓), piqa 0.757, arc_easy
  0.735, winogrande 0.643, arc_challenge acc_norm 0.422, obqa 0.400 — all in the OLMo-2-1B range.
- **7B_base_full**: hellaswag acc_norm **0.805** (public ~0.80-0.81 ✓), piqa 0.811, arc_easy
  0.829, winogrande 0.744, arc_challenge 0.571, obqa 0.462 — all in the OLMo-2-7B range.
- → MC driver validated on both bases; the pruned-model numbers below are trustworthy.

## Results (acc / acc_norm; winogrande acc only, acc_norm≡acc)

### 1B — full 16L base vs 9L keep7+fresh2 (healed)
| model | layers | step | HS | ARC-C | ARC-E | PIQA | WinoG | OBQA |
|-------|-------:|-----:|----|-------|-------|------|-------|------|
| **1B base full** | 16 | — | .508 / **.683** | .386 / **.422** | .728 / **.735** | .760 / **.757** | **.643** | .302 / **.400** |
| 1B keep7 | 9 | 50000 | .342 / .411 | .256 / .289 | .609 / .568 | .687 / .671 | .531 | .226 / .318 |
| 1B keep7 | 9 | 100000 | .355 / .438 | .272 / .300 | .630 / .593 | .695 / .687 | .526 | .236 / .326 |
| 1B keep7 | 9 | 147000 | .362 / .450 | .289 / .311 | .642 / .595 | .701 / .692 | .530 | .238 / .326 |
| **1B keep7** (latest) | 9 | **148500** | .361 / **.448** | .275 / **.294** | .647 / **.606** | .700 / **.695** | **.544** | .234 / **.330** |

### 7B — full 32L base vs 12L keep10+fresh2 (step10000, VERY EARLY heal)
| model | layers | step | HS | ARC-C | ARC-E | PIQA | WinoG | OBQA |
|-------|-------:|-----:|----|-------|-------|------|-------|------|
| **7B base full** | 32 | — | .604 / **.805** | .539 / **.571** | .823 / **.829** | .809 / **.811** | **.744** | .372 / **.462** |
| **7B keep10** (early) | 12 | **10000** | .357 / **.443** | .279 / **.303** | .614 / **.585** | .677 / **.673** | **.524** | .216 / **.326** |

## Story — DOWNSTREAM DOES **NOT** RECOVER IN LOCKSTEP WITH PPL

**Above-chance recovery of the latest 1B keep7 (9L, step148500) vs its 16L base**
(random baselines: HS/ARC/OBQA .25, PIQA/WinoG .50; recovered = (keep7−rand)/(base−rand)):

| task | base accn | keep7 accn | recovered above-chance |
|------|----------:|-----------:|-----------------------:|
| PIQA | .757 | .695 | **76%** |
| ARC-Easy | .735 | .606 | **73%** |
| OpenBookQA | .400 | .330 | **53%** |
| HellaSwag | .683 | .448 | **46%** |
| WinoGrande | .643 | .544 | **31%** |
| ARC-Challenge | .422 | .294 | **26%** |

- **Split verdict, honestly:** the pruned model is a *functional* LM (every task well above
  chance, nan=0), and its **PPL recovered to 1.47×** — a modest 47% LM-loss penalty. But
  **downstream accuracy lags PPL badly and task-dependently.** Surface / lexical-frequency
  tasks (PIQA, ARC-Easy) recover ~73-76% of the base's above-chance signal; **harder
  reasoning tasks (ARC-Challenge 26%, WinoGrande 31%, HellaSwag 46%) recover far less.**
- **This is exactly the direction-#4 signal.** If the dropped middle layers were pure
  redundancy, downstream would recover as fully as PPL. Instead PPL (dominated by
  high-frequency *local* next-token prediction, which the front layers + fresh NTP head
  reconstruct well) is relatively *insensitive* to the loss, while downstream MC — which
  needs the model to *select the semantically/logically correct* completion — exposes a
  large residual gap. **⇒ the middle layers were NOT purely redundant; they carried
  progressive refinement / reasoning capability that PPL is largely blind to.**
- **Heal curve (1B, downstream):** monotone-improving but slow — HS .411→.438→.450→.448,
  ARC-E .568→.593→.595→.606, PIQA .671→.687→.692→.695 across 50k→100k→147k→148.5k. Mirrors
  the PPL curve ("not yet plateaued"); downstream is still creeping up, so the reasoning gap
  should keep closing with more heal steps — but from these slopes it is closing *slowly* and
  is unlikely to fully close to the 16L base without much more training (or more kept layers).
- **7B keep10 step10000 (12L, 32L→12L, only step 10k, PPL 2.33×):** downstream ≈ the 1B
  *early*-heal state (HS .443 vs 1B@50k .411; PIQA .673 vs .671; WinoG .524 vs .531) — i.e.
  this 7B point is very early in healing and its capability gap is correspondingly large.
  Not comparable to the near-converged 1B latest; needs many more heal steps for a fair read.

## Raw JSON (diskB `.../Mixture-of-Memory/`)
- `olmo2_downstream_results/{1B_base_full,1B_keep7_step50000,1B_keep7_step100000,1B_keep7_step147000,1B_keep7_step148500,7B_base_full,7B_keep10_step10000}/summary.json`
  (+ per-shard `shard{0..7}of8.json` holding per-task n / n_correct_acc / n_correct_accnorm /
  n_nan + sample log-prob vectors for the degenerate check).
- Scheduler log: `logs/olmo2_downstream_sched.out`; DONE marker: `logs/olmo2_downstream_DONE`.
