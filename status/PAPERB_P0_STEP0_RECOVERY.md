# Paper B P0 — Step-0 (Pruned, UNHEALED) Recovery-Fraction Baseline (OLMo-2-7B)

**TL;DR.** Generated the missing **step-0 baseline** = layers pruned but **zero heal-training**, for
every depth-ladder rung `keep{8,10,12,14}+fresh2`, using the trainer's own exact recipe. Eval'd each
in the **identical base protocol / same harness** as the healed-run ledgers (no chat_template, no BOS,
fp32 weights + bf16-autocast, likelihood MC). **Result: step-0 is a broken LM** — held-out PPL
**167k–1.45M** (≈ 20,000–195,000× the vanilla base's 7.40) and **at the chance floor on every
downstream MC task** (core-6 and knowledge-5). Mean above-chance recovery of step-0 = **~0–3%** (noise
around zero). ⇒ This **empirically pins the recovery-fraction denominator**: since the heal curve
starts at ≈ chance, the "above-chance recovery %" numbers already reported for the healed checkpoints
(`OLMO2_PRUNEHEAL_DOWNSTREAM.md`) **are** the true recovery fractions (denominator = vanilla − step-0 ≈
vanilla − chance).

- **Node:** QCMem H20 .104 (`28.83.24.104:36000`), diskB, 8×H20, EVAL-ONLY forward. Env
  `/opt/conda/envs/torch-base/bin/python` (torch 2.13.0 / transformers 5.5.4). Ran 2026-08-01
  ~03:13–03:45 CST; GPUs returned idle.
- **Vanilla base (100% ref)** NOT re-run — reused existing summaries (`olmo2_ppl_results/7B_base_full`,
  `olmo2_downstream_results/7B_base_full{,_know}`; 铁律2-validated against published OLMo-2-7B).

---

## 1. Recipe (exact, read from `scripts/train_olmo2_arch_probe2.py`)

The prune-then-heal construction is **FRONT-keep, not ShortGPT block-influence**: keep the **front
`keep_front` decoder layers** (layers `0..keep-1`) + `embed_tokens` + final `model.norm` + `lm_head`
(untied) transplanted verbatim from vanilla **OLMo-2-1124-7B** (32L); **drop the top layers**; append
**`n_fresh=2` FRESH Olmo2-init tail layers** (correct `Olmo2ForCausalLM(cfg)` post_init random init).
`num_hidden_layers = keep_front + 2`. fp32 master weights. This is exactly what continue-training (the
"heal") starts from, so **step-0 = the genuine heal-curve origin**, not a hand-built pure-pruned shell.

- Confirmed against each healed run's `arch_meta.json`: keep8/10/12 all `n_fresh=2`, base
  `../models/OLMo-2-1124-7B`, seq_len 2048; keep14 = 16L/179-tensor signature (matches the healed
  keep14 ledger's "179 tensors, strict").
- **Generated via the trainer's own `--save_step0_and_exit`** path (byte-identical checkpoint format:
  `model_state` + arch meta + empty-AdamW optimizer_state; strict-loadable by the eval harness like any
  `step{N}.pt`). No training, no data, no DDP. One command per rung, `CUDA_VISIBLE_DEVICES=0`.
- **Code note:** diskB's `train_olmo2_arch_probe2.py` was an older revision lacking
  `--save_step0_and_exit`. Synced the newer wzc1 revision (md5 `81359a6…`); verified the *construction*
  functions are functionally identical (`build_olmo2_minimal` byte-identical; `transplant_front`
  differs only by an `n_fresh==0` guard — irrelevant here — and a log-string format). Eval scripts on
  .104 (`eval_olmo2_probe2_ppl.py`, `eval_olmo2_probe2_downstream.py`) were **left untouched** so the
  step-0 numbers use the exact evaluator that produced the base_full / healed references → zero drift.

**Step-0 artifacts (diskB `outputs/`):**

| rung | keep_front | n_fresh | num_hidden_layers | params | tensors | ckpt |
|------|-----------:|--------:|------------------:|-------:|--------:|------|
| keep8  | 8  | 2 | 10 | 2.846 B | 113 | `olmo2_probe2_7B_keep8_step0_pruned/step0.pt` (11 GB) |
| keep10 | 10 | 2 | 12 | 3.250 B | 135 | `olmo2_probe2_7B_keep10_step0_pruned/step0.pt` (13 GB) |
| keep12 | 12 | 2 | 14 | 3.656 B | 157 | `olmo2_probe2_7B_keep12_step0_pruned/step0.pt` (14 GB) |
| keep14 | 14 | 2 | 16 | 4.060 B | 179 | `olmo2_probe2_7B_keep14_step0_pruned/step0.pt` (16 GB) |

---

## 2. Held-out PPL (val `data/dolmino_now_val.npy`, 4096×2048, n_tokens 8,384,512; 8-shard token-weighted merge)

| model | layers | held-out PPL | avg_nll | × vanilla (7.398) |
|-------|-------:|-------------:|--------:|------------------:|
| **vanilla base full** | 32 | **7.3981** | 2.001 | 1.00× (denom) |
| keep8 step-0  | 10 | 1,446,538.6 | 14.185 | ~195,500× |
| keep10 step-0 | 12 |   887,878.4 | 13.697 | ~120,000× |
| keep12 step-0 | 14 | 1,296,702.9 | 14.075 | ~175,300× |
| keep14 step-0 | 16 |   167,371.3 | 12.028 |  ~22,600× |

- Step-0 PPL is **astronomical** at every depth (random fresh tail dominates the residual stream before
  `lm_head`). Not monotone in keep depth (keep14 lowest at 167k, keep8 highest at 1.45M) — this is just
  variance across unseeded random fresh-tail draws; the tail destroys the representation regardless.
- **PPL "retention/recovery" is not defined for step-0** (PPL is unbounded; the ratio is meaningless).
  The load-bearing fact is simply: **step-0 is a non-functional LM.** For context, the healed keep14
  recovers to 10.69 (1.446× vanilla) and from-scratch-16L to 11.50 (1.554×) — both from this ~167k
  floor.

---

## 3. Core-6 downstream MC (acc / **acc_norm**; WinoG acc≡acc_norm). All nan=0, trunc=0, full n.

| model | HS | ARC-C | ARC-E | PIQA | WinoG | OBQA |
|-------|----|-------|-------|------|-------|------|
| **vanilla base full** | .604 / **.805** | .539 / **.571** | .823 / **.829** | .809 / **.811** | **.744** | .372 / **.462** |
| keep8 step-0  | .256 / .266 | .218 / .272 | .258 / .243 | .542 / .522 | .483 | .160 / .288 |
| keep10 step-0 | .256 / .263 | .219 / .262 | .270 / .269 | .528 / .505 | .493 | .166 / .282 |
| keep12 step-0 | .259 / .267 | .213 / .259 | .259 / .255 | .548 / .527 | .503 | .154 / .286 |
| keep14 step-0 | .259 / .273 | .230 / .271 | .259 / .259 | .532 / .520 | .492 | .158 / .296 |
| *chance* | .25 | .25 | .25 | .50 | .50 | .25 |

Every rung sits **at chance** on all 6 core tasks.

## 4. Knowledge-5 downstream MC (acc / acc_norm; MMLU letter-scored acc≡accn). All nan=0, full n; boolq trunc=2 (same 2 long passages accepted in the healed runs).

| model | MMLU | lambada | boolq (acc/accn) | CSQA (acc/accn) | SIQA (acc/accn) |
|-------|-----:|--------:|-----------------:|----------------:|----------------:|
| **vanilla base full** | **.605** | **.732** | .815 / .753 | .665 / .652 | .502 / .547 |
| keep8 step-0  | .247 | .000 | .600 / .622 | .199 / .238 | .337 / .364 |
| keep10 step-0 | .261 | .000 | .580 / .622 | .166 / .223 | .328 / .361 |
| keep12 step-0 | .251 | .000 | .491 / .622 | .198 / .219 | .325 / .355 |
| keep14 step-0 | .254 | .000 | .582 / .622 | .170 / .215 | .335 / .365 |
| *chance* | .25 | ≈0 | .50 (maj≈.62) | .20 | .333 |

- **MMLU** = chance (.247–.261) at every rung → the pruned-unhealed model has **zero factual knowledge**
  (consistent with the healed-ledger story that knowledge lives in the dropped middle layers).
- **lambada = 0.000** (last-word greedy is-greedy metric) — a broken model never greedily completes the
  last word. Clean floor.
- **boolq accn is pinned at .6217 for all rungs = the yes/no majority-class rate** (boolq val is ~62%
  "yes"); raw acc wanders .49–.60. This is the same yes/no length-normalization / label-imbalance wash
  flagged in the healed ledgers — **not** real above-chance signal.

---

## 5. Recovery-fraction baseline — step-0 is the empirical 0% floor

Above-chance recovery of step-0 = `(step0 − chance)/(vanilla − chance)`, the **same formula** the healed
ledgers use. This is the honest "retention vs vanilla" metric (raw `step0/vanilla` ratios are inflated by
the high chance floors — e.g. MMLU raw .254/.605 = 42% is **all chance**, not retained knowledge).

| rung | mean above-chance recovery (10 tasks, ex-boolq) | range across tasks |
|------|-----------------------------------------------:|-------------------:|
| keep8  | **+2.8%** | [−7.0%, +17.9%] |
| keep10 | **+1.6%** | [−7.2%, +15.1%] |
| keep12 | **+2.9%** | [−4.7%, +17.0%] |
| keep14 | **+3.3%** | [−6.4%, +21.7%] |

keep14 per-task above-chance recovery: HS +4.1%, ARC-C +6.6%, ARC-E +1.6%, PIQA +6.3%, WinoG −3.4%,
OBQA +21.7%, MMLU +1.2%, lambada 0.0%, CSQA −6.4%, SIQA +1.3% (+ boolq +26% = majority artifact).

- Step-0 above-chance recovery is **essentially 0** (mean ~2–3%, individual tasks scatter −7% to +7%
  from sampling noise). The two apparent outliers are artifacts: **OBQA** (+22%) has a tiny denominator
  (chance .25, vanilla .462, n=500 → high variance; accn .296 is noise just above .25), and **boolq**
  (+26%) is the majority-class wash noted above.
- **⇒ Recovery-fraction denominator is empirically confirmed = vanilla − chance.** The heal curve starts
  at ≈ chance / broken-LM (this file) and the healed checkpoints' already-reported above-chance recovery
  IS the fraction of vanilla capability that heal-training restored. No re-computation of the healed
  numbers is needed; this file supplies the missing, previously-assumed **0% anchor**.

### Cross-reference (denominator now anchored on both ends)
| axis | step-0 (this file, ≈0%) | healed keep14 @153.5k (`OLMO2_PRUNEHEAL_DOWNSTREAM.md`) | vanilla |
|------|-------------------------|--------------------------------------------------------|---------|
| MMLU | .254 (chance) | .312 → 17.6% above-chance | .605 |
| HS accn | .273 (chance) | .643 → 71% | .805 |
| PIQA accn | .520 (chance) | .745 → 79% | .811 |
| held-out PPL | 167k (broken) | 10.69 (1.446×) | 7.398 |

---

## 6. Artifacts, harness, provenance

- **Step-0 ckpts:** `outputs/olmo2_probe2_7B_keep{8,10,12,14}_step0_pruned/step0.pt` (+ each dir's
  `arch_meta.json`-equivalent meta inside the ckpt; arm=`healing_front{N}+fresh2`, do_transplant=True).
- **PPL summaries:** `olmo2_ppl_results/7B_keep{8,10,12,14}_step0/summary.json` (+ per-shard `shard{0..7}of8.json`).
- **Downstream summaries:** `olmo2_downstream_results/7B_keep{N}_step0/summary.json` (core-6) and
  `…/7B_keep{N}_step0_know/summary.json` (knowledge-5), each with per-shard shard files.
- **Launcher (new):** `scripts/_run_olmo2_probe2_step0_recovery.sh` — Phase1 gen (all 4 rungs, exact
  recipe) → Phase2 PPL → Phase3 core-6 → Phase4 knowledge-5, all 8-GPU `[g::8]` sharded, base 口径.
- **Scheduler log:** `logs/olmo2_step0_recovery_sched.out`; DONE marker `logs/olmo2_step0_recovery_DONE`.
- **Reused (untouched) drivers:** `scripts/eval_olmo2_probe2_ppl.py`, `scripts/eval_olmo2_probe2_downstream.py`.

## 7. Caveats / pitfalls
- Step-0 = pruned **+ fresh random untrained tail** (the true training origin), NOT a pure ShortGPT
  keep_front-only shell. This is deliberate and correct for the recovery-fraction denominator, but means
  step-0 is a **broken LM** (random final 2 layers) rather than a "mildly degraded shallow model" — the
  numbers are chance/astronomical by construction, and that is the intended anchor.
- No `keep16` / `shortgpt16` run exists in `outputs/` (depth-ladder rungs are keep{8,10,12,14}); nothing
  to add.
- `train_olmo2_arch_probe2.py` on diskB was upgraded to the wzc1 revision to expose `--save_step0_and_exit`
  (construction logic verified equivalent). Eval drivers untouched.
- boolq / OBQA above-chance readouts are artifact-contaminated (majority-class / tiny-denominator) — do
  not over-read them; the aggregate "step-0 ≈ 0% above chance" is the robust claim.
