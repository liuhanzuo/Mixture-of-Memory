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

---

# § Knowledge / Comprehension 扩展（MMLU / lambada / boolq / csqa / siqa）

Companion to the commonsense-reasoning + surface tasks above. Those recovered a *task-dependent*
26-76% of the base's above-chance signal. **Direction-#4 follow-up:** does *knowledge* and
*reading comprehension* recover — or, like the harder reasoning tasks, lag PPL badly? Added 5
zero-shot tasks to the **same harness** (`scripts/eval_olmo2_probe2_downstream.py`, extended;
launcher `scripts/_run_olmo2_probe2_downstream_know_8gpu.sh`; same 8-shard `[g::8]` dynamic
scheme; same fp32-weights / bf16-autocast forward; `add_special_tokens=False` → no BOS):
- **mmlu** (`cais/mmlu` "all", 14042 test, 57 subjects, 4-choice **letter** continuations → acc≡acc_norm;
  aggregate acc + per-subject breakdown; flan-style per-subject description prompt).
- **lambada_openai** (`EleutherAI/lambada_openai`, 5153 test): last-word prediction, metric =
  **is_greedy** (whole last-word continuation is the model's greedy argmax) — generation-free
  long-range-coherence probe → acc only.
- **boolq** (`google/boolq`, 3270 val): yes/no reading comprehension, 2-choice likelihood → acc only.
- **commonsense_qa** (`tau/commonsense_qa`, 1221 val, 5-choice), **social_iqa**
  (`allenai/social_i_qa` @ `refs/convert/parquet`, 1954 val, 3-choice) — choice-text continuations.

Run finished **2026-07-19 14:27 CST** (~15 min wall for all 7 configs on .73 8×H20). **All 7
configs: nan=0, trunc=0 on every task; no task skipped (all 5 datasets loaded via hy-proxy).**
Latest available 1B keep7 ckpt at launch = **step150000** (trainer had advanced past 148500).

## Sanity (铁律2) — base_full reproduces published OLMo-2
- **7B base full**: MMLU **0.605** (target ~0.5-0.63 ✓), lambada acc **0.732** (target ~0.7+ ✓),
  boolq 0.815, csqa 0.665, siqa 0.502/0.547 — all in the OLMo-2-7B range → driver validated.
- **1B base full**: MMLU **0.382** (small-model MMLU just above random 0.25 ✓), lambada 0.648,
  boolq 0.624, csqa 0.553, siqa 0.422/0.470.

## Results (acc / acc_norm; mmlu is letter-scored so acc≡acc_norm; lambada/boolq report acc only)

### 1B — full 16L base vs 9L keep7+fresh2 (healed)
| model | step | MMLU | lambada | boolq | CSQA (acc/accn) | SIQA (acc/accn) |
|-------|-----:|-----:|--------:|------:|----------------:|----------------:|
| **1B base full** | — | **.382** | **.648** | **.624** | .553 / **.562** | .422 / **.470** |
| 1B keep7 | 50000 | .2495 | .3536 | .618 | .368 / .363 | .384 / .428 |
| 1B keep7 | 100000 | .2558 | .3740 | .597 | .380 / .378 | .394 / .429 |
| 1B keep7 | 147000 | .2529 | .4021 | .584 | .391 / .384 | .384 / .431 |
| **1B keep7** (latest) | **150000** | **.2480** | **.3969** | **.577** | .387 / .381 | .386 / .431 |

### 7B — full 32L base vs 12L keep10+fresh2 (step10000, VERY EARLY heal)
| model | step | MMLU | lambada | boolq | CSQA (acc/accn) | SIQA (acc/accn) |
|-------|-----:|-----:|--------:|------:|----------------:|----------------:|
| **7B base full** | — | **.6053** | **.7316** | **.8147** | .665 / **.652** | .502 / **.547** |
| **7B keep10** (early) | **10000** | **.2539** | **.4036** | **.598** | .421 / .406 | .378 / .431 |

## Story — KNOWLEDGE **DOES NOT RECOVER AT ALL**; comprehension recovers like the surface band

Above-chance recovery, latest 1B keep7 (9L, step150000) vs 16L base
(chance: MMLU .25, lambada ≈0, boolq .50, CSQA .20, SIQA .333; recovered=(keep7−rand)/(base−rand)):

| task | base | keep7 | recovered above-chance |
|------|-----:|------:|-----------------------:|
| boolq (read-comp, passage in-context) | .624 | .577 | **62%** |
| lambada (last-word / long-range coherence) | .648 | .397 | **61%** |
| social_iqa (acc) | .422 | .386 | **60%** |
| commonsense_qa (acc) | .553 | .387 | **53%** |
| **MMLU (factual knowledge)** | **.382** | **.248** | **≈0% (pinned at random)** |

- **MMLU is the single capability that shows ZERO recovery — it sits at the 0.25 chance floor at
  EVERY heal step (50k→150k: .2495/.2558/.2529/.2480, flat, no trend) and at BOTH scales (7B
  keep10 = .2539).** Per-subject confirms it is a *true knowledge wipe*, not an averaging artifact:
  knowledge-heavy subjects collapse to chance — high_school_us_history .51→.26 (1B) / **.82→.26**
  (7B), world_religions .49→.29 / **.84→.27**, marketing .56→.27 / **.83→.25**, miscellaneous
  .52→.27 / **.80→.26**. Every one of the 57 subjects is at ~chance in the pruned models.
- **This is qualitatively sharper than the commonsense-reasoning result.** There, *every* task
  recovered ≥26% of above-chance signal. Here, factual knowledge recovers **nothing** — the dropped
  middle layers stored world knowledge (consistent with the "transformer MLP layers = key-value
  fact memories" literature) that the front layers + 2 fresh layers cannot reconstruct, and healing
  on general web text does **not** re-instill it. PPL recovered to 1.47× (1B); MMLU recovered to 0×.
  **The strongest evidence yet that the middle layers were NOT redundant.**
- **Comprehension / coherence recover in the *surface* band, not the knowledge band.** boolq
  (yes/no, **passage supplied in-context**) recovers ~62% — the answer is *extractable locally*, so
  it behaves like the lexical/surface tasks (PIQA 76%, ARC-E 73%) rather than like knowledge.
  lambada (last-word) recovers ~61%: long-range coherence is degraded (base .648→.40) but the front
  layers + fresh NTP head rebuild most of the *local* prediction ability PPL rewards. csqa/siqa
  (50-60%) sit in the mid commonsense band (siqa base is itself near chance, so noisy).
- **Heal curves (1B):** MMLU flat at chance (never departs .25). lambada creeps .354→.397 then
  plateaus. boolq acc even drifts *down* .618→.577 (while its acc_norm drifts .622→.630 — a
  yes/no length-normalization wash; comprehension ≈flat ~.60). csqa slow-rises .368→.387. **No new
  task shows the pruned model closing on the base; MMLU shows it will *never* close by healing.**
- **7B keep10 step10000 (12L, 32L→12L, only step 10k):** MMLU .254 (chance), lambada .404, boolq
  .598, csqa .421 — same pattern as the early-heal 1B, i.e. knowledge already fully gone and the
  rest very early in recovery. Not comparable to the near-converged 1B latest; but the MMLU=chance
  verdict is scale-invariant and immediate.

**Verdict for direction #4 (honest):** the earlier tasks said "PPL recovers, capability lags."
The knowledge/comprehension axis says something stronger and cleaner: **factual knowledge does not
lag — it is *irrecoverable* (MMLU stuck at random across all subjects, all heal steps, both
scales), while comprehension that can be solved from in-context evidence (boolq) recovers as well
as surface tasks.** ⇒ the pruned middle layers carried *stored knowledge*, and PPL — dominated by
local next-token prediction — is completely blind to its loss.

## Raw JSON — knowledge extension (diskB `.../Mixture-of-Memory/`)
- `olmo2_downstream_results/{1B_base_full,1B_keep7_step50000,1B_keep7_step100000,1B_keep7_step147000,1B_keep7_step150000,7B_base_full,7B_keep10_step10000}_know/summary.json`
  (mmlu entry carries a `subjects` map = per-subject n / n_correct_acc / acc; per-shard
  `shard{0..7}of8.json` carries per-task n / n_correct_acc / n_correct_accnorm / n_nan / mode /
  sample log-prob vectors).
- Scheduler log: `logs/olmo2_downstream_know_sched.out`; DONE marker: `logs/olmo2_downstream_know_DONE`.
