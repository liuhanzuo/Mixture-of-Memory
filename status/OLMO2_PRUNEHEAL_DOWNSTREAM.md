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
| **7B keep14** (apex, converged) | 16 | **128000** | .480 / **.631** | .402 / **.426** | .729 / **.702** | .747 / **.747** | **.630** | .322 / **.402** |
| **7B keep14** (post-apex, +25.5k) | 16 | **153500** | .485 / **.643** | .412 / **.442** | .732 / **.705** | .746 / **.745** | **.633** | .328 / **.406** |

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
| **7B keep14** (apex, converged) | **128000** | **.3012** | **.5750** | **.6385** | .505 / .476 | .423 / .471 |
| **7B keep14** (post-apex, +25.5k) | **153500** | **.3124** | **.5698** | .6058 / .682 | .506 / .479 | .441 / .474 |

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

---

# § APEX POINT — 7B keep14+fresh2 (16L/32 = 50%), step128000 (fully-converged heal)

**Why this point (2026-07-19):** the earlier "MMLU pinned at random / knowledge irrecoverable"
verdict rested on two *under-healed / over-pruned* 7B/1B points — 7B keep10 (12L/37.5%) at
**step10000** (essentially no heal) and 1B keep7 (9L/56%). The obvious rebuttal was **"you didn't
keep enough layers, and you didn't heal long enough."** This is the apex data point that closes
that gap on the strong side: **more layers (16L/50%) AND fully-converged heal (step128000, trainer
long past this step, held-out PPL plateaued).** Same harness / same node .73 / same 8-shard `[g::8]`
/ same fp32-weights bf16-autocast forward. Driver auto-inferred `keep_front=14 n_fresh=2
num_hidden_layers=16 (179 tensors, strict)` from the ckpt state-dict — zero arch drift (driver md5
identical to wzc1). Ran 2026-07-19 ~15:32-15:42 CST (core ~5 min, knowledge ~4 min, 8×H20).

## 铁律2 self-audit (independent recompute from the 8 shard JSONs)
- **Core 6-task:** all 6 cells `Σn_correct/Σn` reproduce the merge summary **to 1e-9**; every task
  n_nan=0, n_trunc=0, full n (HS 10042 / ARC-C 1172 / ARC-E 2376 / PIQA 1838 / WinoG 1267 / OBQA 500).
- **Knowledge 5-task:** all 5 cells reproduce the merge summary **to 1e-9**; every task n_nan=0 and
  **full n** (mmlu 14042 / lambada 5153 / boolq 3270 / csqa 1221 / siqa 1954). boolq n_trunc=2 (the
  same 2 super-long passages truncated in every config — accepted, not a bug). No shard crashed
  despite non-fatal CUDACachingAllocator OOM *warnings* (allocator retried; all 8 shards wrote).
- ⇒ **Every cell verified; results trustworthy.**

## Result — knowledge **PARTIALLY recovers** at the apex; it does NOT stay pinned at floor
Above-chance recovery vs 7B base full (chance: HS/ARC/OBQA/MMLU .25, PIQA/WinoG/boolq .50,
lambada ≈0, csqa .20, siqa .333; recovered = (keep14−rand)/(base−rand); accn where reported):

| axis | task | 7B base | keep14 | recovered above-chance |
|------|------|--------:|-------:|-----------------------:|
| surface/reason | PIQA | .811 | .747 | **79%** |
| surface/reason | ARC-Easy | .829 | .702 | **78%** |
| surface/reason | OpenBookQA | .462 | .402 | **72%** |
| surface/reason | HellaSwag | .805 | .631 | **69%** |
| surface/reason | ARC-Challenge | .571 | .426 | **55%** |
| surface/reason | WinoGrande | .744 | .630 | **53%** |
| comprehension | lambada (last-word) | .732 | .575 | **79%** |
| comprehension | commonsense_qa (acc) | .665 | .505 | **65%** |
| comprehension | boolq (acc, passage in-ctx) | .815 | .639 | **44%** |
| comprehension | social_iqa (acc) | .502 | .423 | **53%** |
| **knowledge** | **MMLU** | **.605** | **.301** | **14%** |

- **The apex point does NOT reproduce the "0% / pinned-at-.25-floor" MMLU result.** MMLU = **.3012**
  on n=14042 is **~13 standard errors above the .25 chance floor** (SE≈.0039) — a *robust, real*
  above-chance signal. Per-subject confirms it is genuine differential recovery, not an averaging
  wash: knowledge-heavy subjects that sat at chance for keep10 (all ~.25-.27) **lift clearly off the
  floor** — world_religions **.27→.427**, us_foreign_policy → **.46**, virology **.404**, marketing
  **.25→.385**, high_school_us_history **.26→.309**, miscellaneous **.26→.364**. Only STEM/quantitative
  subjects stay at/below chance (professional_medicine .195, econometrics .219, college_chemistry .22).
- **HONEST verdict — the "heal-wasn't-enough" rebuttal is NOT fully closed; it is partly *vindicated*.**
  Keeping 16L (50%) instead of 12L, and healing to convergence instead of step10000, **does recover
  some factual knowledge** (MMLU .254→.301; and *all* core/comprehension tasks jump too: HS accn
  .443→.631, ARC-E .585→.702, PIQA .673→.747). So knowledge is **not strictly irrecoverable** — more
  depth + more heal buys real recovery.
- **But the core direction-#4 signal holds in its *ordering / magnitude* form.** Even at the apex
  (50% depth, fully converged), **knowledge is by far the weakest-recovering axis: MMLU recovers only
  14% of the base's above-chance signal, vs 53-79% for every reasoning/surface task and 44-79% for
  comprehension.** MMLU stays at .30 — less than half of the base's .605. The strict claim ("PPL
  recovers, MMLU stuck at random / 0%") was an artifact of under-healed/over-pruned points and must be
  softened; the robust claim survives: **the pruned middle layers carried stored world knowledge that
  heal recovers only a small fraction of, far less than it recovers reasoning/surface ability — PPL
  (recovered to ~1.5-2.3×) remains largely blind to this residual knowledge gap.**

## Raw JSON — apex point (diskB `.../Mixture-of-Memory/`)
- `olmo2_downstream_results/7B_keep14_step128000/summary.json` (core 6-task) and
  `olmo2_downstream_results/7B_keep14_step128000_know/summary.json` (knowledge; mmlu carries the
  57-subject `subjects` map) + per-shard `shard{0..7}of8.json`.
- Launchers: `scripts/_run_olmo2_probe2_downstream_keep14_8gpu.sh` (core) /
  `scripts/_run_olmo2_probe2_downstream_keep14_know_8gpu.sh` (knowledge).
- Scheduler logs: `logs/olmo2_downstream_keep14_sched.out` / `logs/olmo2_downstream_keep14_know_sched.out`;
  DONE markers: `logs/olmo2_downstream_keep14_DONE` / `logs/olmo2_downstream_keep14_know_DONE`.

---

# § POST-APEX TRAJECTORY — 7B keep14+fresh2 (16L/32), step153500 (+25.5k past apex step128000)

**Why (2026-07-20):** the trainer kept healing ~25.5k steps past the step128000 point where the
apex MC/knowledge was measured. **Core question:** after more heal, does capability *keep
recovering, hold flat, or roll back*? This de-risks the eventual step200000 eval and directly tests
the Paper-B claim "PPL plateaus early but knowledge keeps recovering (or stalls)." Same node .73 /
same 8-shard `[g::8]` / same fp32-weights bf16-autocast forward / base 口径 (add_bos False, no
chat_template, no generation). Ckpt cp-snapshotted off the wzc1 train dir (rotation-safe) → scp to
.73; driver loaded it `keep_front=14 n_fresh=2 num_hidden_layers=16 (179 tensors, strict)` — identical
arch signature to the apex ckpt. Ran 2026-07-20 04:06-04:12 CST (ppl ~5min, core ~3min, know ~4min).

## 铁律2 self-audit (independent recompute from the 8 shard JSONs)
- **PPL:** 8/8 shards, each n_tokens=1,048,064, none empty; recompute exp(Σnll/Σtok)=**10.693429** == summary to 1e-9.
- **Core 6-task:** 8/8 shards, empty_shards=0; every cell Σn_correct/Σn reproduces the merge to 1e-9; every task n_nan=0 n_trunc=0, full n (HS 10042 / ARC-C 1172 / ARC-E 2376 / PIQA 1838 / WinoG 1267 / OBQA 500).
- **Knowledge 5-task:** 8/8 shards, empty_shards=0; every cell reproduces the merge to 1e-9; every task n_nan=0, full n (mmlu 14042 / lambada 5153 / boolq 3270 / csqa 1221 / siqa 1954); boolq n_trunc=2 (same 2 super-long passages as apex, accepted).
- ⇒ **Every cell verified; results trustworthy.**

## Result — capability keeps creeping up, does NOT roll back; still knowledge-limited
Paired with `OLMO2_PRUNEHEAL_PPL.md` (same ckpt step153500): held-out PPL **10.827→10.693**
(1.463×→1.446×, slight further recovery). Downstream deltas apex(128000)→post-apex(153500):

| axis | task | apex 128000 | 153500 | Δ | dir |
|------|------|------------:|-------:|--:|:---:|
| knowledge | **MMLU** | .3012 | **.3124** | +.011 | up |
| surface/reason | HellaSwag accn | .631 | .643 | +.012 | up |
| surface/reason | ARC-Challenge accn | .426 | .442 | +.016 | up |
| surface/reason | ARC-Easy accn | .702 | .705 | +.003 | flat/up |
| surface/reason | PIQA accn | .747 | .745 | −.002 | flat |
| surface/reason | WinoGrande | .630 | .633 | +.003 | flat/up |
| surface/reason | OpenBookQA accn | .402 | .406 | +.004 | flat/up |
| comprehension | lambada (last-word) | .575 | .570 | −.005 | flat |
| comprehension | boolq (acc) | .639 | .606 | −.033 | down* |
| comprehension | commonsense_qa (acc) | .505 | .506 | +.001 | flat |
| comprehension | social_iqa (acc) | .423 | .441 | +.018 | up |

\*boolq: acc drifts down but acc_norm=.682 (yes/no length-normalization wash; the yes/no
answer-length imbalance makes raw acc noisy — comprehension itself ≈flat, cf. the same acc-vs-accn
split noted for 1B in the knowledge section).

- **MMLU above-chance recovery** rises **14.4%→17.6%** ((.3124−.25)/(.605−.25)) — a small but real
  further knowledge recovery, still by far the weakest axis (base .605, keep14 .312 = ~half).
- **Verdict:** +25.5k heal steps → **no regression / no overfitting collapse**; PPL and every
  reasoning/knowledge axis is **flat-to-slightly-up** (biggest gainers HS/ARC-C accn +.012/.016,
  MMLU +.011, siqa +.018). But the per-step yield is **tiny** vs the earlier 12L→16L / 10k→128k
  jump. ⇒ **supports "PPL plateaued early; knowledge keeps recovering but only marginally per extra
  step" — extending heal past the apex helps a little and does not hurt.** de-risks the step200000
  final eval (expect the same slow-creep, no rollback).

## Raw JSON — post-apex (diskB `.../Mixture-of-Memory/`)
- `olmo2_downstream_results/7B_keep14_step153500/summary.json` (core 6-task) and
  `olmo2_downstream_results/7B_keep14_step153500_know/summary.json` (knowledge; mmlu carries the
  57-subject `subjects` map) + per-shard `shard{0..7}of8.json`.
- Launchers: `scripts/_run_olmo2_probe2_downstream_keep14_s153500_8gpu.sh` /
  `scripts/_run_olmo2_probe2_downstream_keep14_s153500_know_8gpu.sh` (+ ppl
  `scripts/_run_olmo2_probe2_ppl_keep14_s153500.sh`, master `scripts/_run_olmo2_keep14_s153500_master.sh`).
- DONE markers: `logs/olmo2_downstream_keep14_s153500_DONE` / `logs/olmo2_downstream_keep14_s153500_know_DONE`;
  master `logs/keep14_s153500_ALL_DONE`.

---

# § CONTROL 2 — from-scratch 16L (random-init keep14/fresh2 shell, step200000)

**Why (2026-07-25):** the paired Paper-B control for "does *inheriting the pretrained front layers*
matter, or would training the same small 16L architecture *from scratch* on the same Dolmino data
reach the same place?" Same 16-layer arch as the healed keep14 (keep_front=14/n_fresh=2), **all weights
random-init**, trained to **step200000 — MORE steps than the healed keep14 (128k apex / 153.5k
post-apex)**. Paired with the held-out PPL row (`OLMO2_PRUNEHEAL_PPL.md`, from-scratch=11.498/1.554×,
worse than healed 10.693/1.446×). Ran on **LOCAL** (8×B200, wzc1, .venv); ckpt
`outputs/olmo2_probe2_7B_keep14fresh2_fromscratch/final.pt` loaded strict (keep_front=14 n_fresh=2
num_hidden_layers=16). Core 6-task **2026-07-25 10:25, all nan=0/trunc=0, full n**; knowledge 5-task
(mmlu/lambada/boolq/csqa/siqa) **DONE 2026-07-25 10:32, all nan=0, full n** (driver `_run_olmo2_downstream_scratch_know_wzc1.sh`).

## Core 6-task (acc / acc_norm; winogrande acc only)
| model | layers | step | HS | ARC-C | ARC-E | PIQA | WinoG | OBQA |
|-------|-------:|-----:|----|-------|-------|------|-------|------|
| **7B base full** | 32 | — | .604 / **.805** | .539 / **.571** | .823 / **.829** | .809 / **.811** | **.744** | .372 / **.462** |
| **7B keep14 healed** (post-apex) | 16 | 153500 | .485 / **.643** | .412 / **.442** | .732 / **.705** | .746 / **.745** | **.633** | .328 / **.406** |
| **7B from-scratch** (control 2) | 16 | **200000** | .447 / **.578** | .382 / **.414** | .732 / **.697** | .724 / **.733** | **.545** | .278 / **.384** |

## Story — CORE: from-scratch ≈ ties healed on SURFACE, lags on REASONING/coherence
Above-chance recovery (accn; chance HS/ARC/OBQA .25, PIQA/WinoG .50):

| task | base | healed 153.5k | from-scratch 200k | healed vs scratch |
|------|-----:|--------------:|------------------:|:-----------------:|
| ARC-Easy | .829 | 79% | **77%** | ≈tie |
| PIQA | .811 | 79% | **75%** | ≈tie |
| OpenBookQA | .462 | 74% | **63%** | healed +11pt |
| HellaSwag | .805 | 71% | **59%** | healed +12pt |
| ARC-Challenge | .571 | 60% | **51%** | healed +9pt |
| WinoGrande | .744 | 55% | **18%** | **healed +37pt** |

- **Honest core read:** on *surface / lexical* tasks (ARC-Easy, PIQA) that are **learnable from Dolmino
  from scratch**, from-scratch (200k steps) essentially **ties** the healed model — inheritance buys
  little there. The gap opens on tasks needing more than local co-occurrence: HellaSwag (+12pt),
  ARC-Challenge (+9pt), and especially **WinoGrande (+37pt)** — coreference/commonsense reasoning is
  where the inherited pretrained front layers help most. **This alone is a modest control** (surface
  ties, reasoning lags), consistent with the modest PPL gap (1.554× vs 1.446×).
## Knowledge 5-task — ★ THE DECISIVE CONTROL (confirmed 2026-07-25 10:32)
| model | step | MMLU | lambada | boolq (acc) | CSQA (acc) | SIQA (acc) |
|-------|-----:|-----:|--------:|------------:|-----------:|-----------:|
| **7B base full** | — | **.605** | .732 | .815 | .665 | .502 |
| **7B keep14 healed** (post-apex) | 153500 | **.312** | .570 | .606 | .506 | .441 |
| **7B from-scratch** (control 2) | **200000** | **.246** | .484 | .614 | .451 | .416 |

Above-chance recovery (chance MMLU .25, lambada ≈0, boolq .50, CSQA .20, SIQA .333; recovered=(x−rand)/(base−rand)):

| task | base | healed 153.5k | from-scratch 200k | verdict |
|------|-----:|--------------:|------------------:|:--------|
| **MMLU (stored world knowledge)** | .605 | **17.6%** | **≈0% (.246 = chance floor)** | **★ DECISIVE: only healed recovers** |
| lambada (long-range coherence) | .732 | 78% | 66% | healed +12pt |
| social_iqa | .502 | 64% | 49% | healed +15pt |
| commonsense_qa | .665 | 66% | 54% | healed +12pt |
| boolq (in-ctx reading comp) | .815 | 34% | **36%** | ≈tie (learnable from scratch) |

- **★ MMLU is the decisive control read, exactly as predicted.** from-scratch MMLU = **.2461** on
  n=14042 — statistically **at the .25 chance floor** (SE≈.0037; .2461 is ~1 SE *below* chance, i.e.
  indistinguishable from random). It recovers **0% of the base's above-chance signal** despite training
  **200k Dolmino steps — MORE than the healed keep14's 153.5k**. The healed keep14 (same 16L arch, same
  Dolmino data, FEWER steps) recovered MMLU to **.312 (17.6% above chance)** — **solely because it
  inherited the pretrained front 14 layers that store world knowledge.** from-scratch shows none of the
  healed model's per-subject lift (world_religions/us_foreign_policy/marketing/history all stay at chance).
- **The boolq tie is the perfect complement:** in-context reading comprehension (answer is *in the
  passage*) is **learnable from scratch** — from-scratch .614 ≈ healed .606 (both ~35% above chance;
  from-scratch even nominally higher). So the from-scratch model is **NOT globally broken** — it just
  **cannot acquire the stored world knowledge that lives in the inherited layers.** The deficit is
  knowledge-specific, not a general capacity failure.
- **★ Paper-B control 2 CLOSED — clean, sharp message:** *the healed model's recovered factual
  knowledge came from the inherited pretrained front layers, not from heal-training.* A from-scratch
  model of identical architecture, trained on the same data for MORE steps, stays pinned at the MMLU
  chance floor. Inheritance is **not merely a sample-efficiency shortcut** (the PPL gap is modest,
  1.554× vs 1.446×) — for stored world knowledge it is the **only** route (MMLU 0% vs 17.6% recovery).
  Reasoning/coherence (HS/ARC/WinoG/lambada/csqa/siqa) partly transfers from scratch but consistently
  lags healed (+9 to +37pt); only in-context comprehension (boolq) fully ties. **This is the strongest
  single piece of evidence that prune-then-heal ≫ train-small-from-scratch for the knowledge axis.**

## Raw JSON — control 2 (wzc1, LOCAL, shared FS)
- `olmo2_downstream_results/7B_scratch16L_step200000/summary.json` (core 6-task; done 10:25) +
  `olmo2_downstream_results/7B_scratch16L_step200000_know/summary.json` (knowledge; done 10:32 — MMLU .2461/lambada .4838/boolq .6138/csqa .4505/siqa .4156, all n_nan=0 full n).
- Launchers: `scripts/_run_olmo2_downstream_scratch_wzc1.sh` (core) /
  `scripts/_run_olmo2_downstream_scratch_know_wzc1.sh` (knowledge).
- DONE markers: `logs/olmo2_downstream_scratch_DONE` (core, landed 10:25) /
  `logs/olmo2_downstream_scratch_know_DONE` (knowledge, landed 10:32).
