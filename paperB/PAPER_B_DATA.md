# Paper B — Consolidated Empirical Materials (prune-then-heal depth sweep, OLMo-2-1124 base)

> **Purpose**: single research-writing prep doc consolidating ALL empirical materials for Paper B.
> Every number is quoted from a file with its source path cited inline. Missing/uncertain numbers are
> marked `(TODO: verify from <where>)` rather than guessed. This doc invents no numbers.
>
> **Primary sources swept** (as of 2026-07-26):
> - `status/OLMO2_PRUNEHEAL_PPL.md` — held-out NTP perplexity ledger.
> - `status/OLMO2_PRUNEHEAL_DOWNSTREAM.md` — downstream MC + MMLU/knowledge ledger (core 6-task,
>   knowledge 5-task, APEX §, POST-APEX §, CONTROL 2 from-scratch §).
> - `status/RUN_REGISTRY.md` §"Paper B — OLMo-2 base 剪层-heal" (lines ~1930-1944) — arm configs.
> - `ops/research_notes/new_propositions_20260725.md` §5 (P2 two-depths RESULTS).
> - `results/knowledge_logit_lens_OLMo-2-1124-7B.json`, `results/knowledge_logit_lens_Qwen3-8b-local.json`,
>   `results/probe_linguistic_qwen3_8b.json` — logit-lens / linguistic probes.
> - Training logs `logs/olmo2_7B_keep14fresh2{,_fromscratch,_freezefront}.log`.
> - Checkpoints `outputs/olmo2_probe2_7B_keep14fresh2{,_fromscratch,_freezefront}/`.
> - `status/SESSION_HANDOFF.md` (Paper B snapshot + Paper B/C boundary context).

---

## 1. Thesis / one-liner + headline claims defensible TODAY

**One-liner.** Starting from the OLMo-2-1124 **base** model (pure pretrained, no instruct
contamination), truncate to the front `keep` layers, add `n_fresh=2` freshly-initialized tail layers,
and continue-train ("heal") on Dolmino (DCLM subset re-tokenized with the OLMo-2 tokenizer). Sweeping
`keep` gives an **available-checkpoint depth ladder**: how few front layers can be retained while
recovering base-LM quality, and where does the observed capability frontier lie? Because the rungs
are evaluated at different training steps, this is not a compute-matched depth law. The finding is a
**PPL-vs-capability dissociation**: language-modelling perplexity improves substantially, while
zero-shot MMLU is not comparably recovered within the observed training budgets. MMLU is treated as
a knowledge-sensitive closed-book measure, not as exhaustive coverage of factual competence.

**Headline claims Paper B can defend TODAY:**

1. **Prune-to-half-depth heals LM quality to a 42.8% PPL tax at 200k.** keep14
   (cut at 14/32 = 0.4375L; resulting depth 16/32 = 0.50L) reaches held-out NTP
   perplexity **10.561** vs the full 32L base **7.398** = **1.428×** at step200000.
   Earlier points are 10.826@128k and 10.693@153.5k; the final 46.5k steps buy
   another 0.132 PPL while MMLU moves only .3124→.3191.

2. **★ MMLU is not comparably recovered within the observed healing budget, and pretrained
   initialization matters at the evaluated operating points.** At keep14, MMLU = **.3012** (128k)
   → **.3124** (153.5k) → **.3191** (200k), or **14.4%→17.6%→19.5%** of the
   base's above-chance signal, versus roughly 44–79% for reported non-MMLU tasks.
   The **fully random-init control** uses the same 16L shape, corpus, batch, and 200k
   steps but loads no base weights and uses peak LR 1e-4, versus 2e-5 for all
   parameters in the executed inherited run. It reaches **MMLU .2461**, the .25
   chance floor. This is not an initialization-only ablation, and the measured
   signal is MMLU rather than factual knowledge in general. The matched-LR **freeze_front** arm
   reaches PPL **12.797** and MMLU **.2628** (**3.6%** recovery), versus PPL 10.561 and
   MMLU .3191 when all inherited blocks adapt. Reuse without adaptation does not reproduce
   the train-all result under this recipe.

3. **Selective rather than global failure.** At matched step200000 on BoolQ
   (yes/no, passage supplied in context), fully random-init raw accuracy is
   **.614** versus inherited keep14 **.638**; ARC-E and PIQA are also close. This shows the random-init model is not globally nonfunctional, while MMLU remains the
   clearest deficit under the evaluated protocol. It does not establish that all in-context capability
   is learnable from scratch or that closed-book knowledge is irrecoverable.

4. **Mechanistic correlate: knowledge becomes logit-lens-decodable deeper than the retained cuts.**
   On OLMo-2-7B, knowledge decodability onset is **L18 = 0.562L** and sat95 is
   **L19 = 0.594L** (`results/knowledge_logit_lens_OLMo-2-1124-7B.json`). A separate Qwen3
   linear-probe heuristic reaches 95% of the **final-layer** score at mean **~0.13L** across
   WiC/SST-2/RTE. Because final-layer scores can be below intermediate peaks (especially RTE), this
   is an exploratory cross-model reference, not an OLMo semantic boundary. The depth ordering is
   consistent with the behavioral result but does not causally explain it.

The frozen-front result supports an adaptation requirement under this recipe, but it does not
localize which inherited modules or parameter changes produce the gain.

**2026-07-29 additions.** Item-paired keep14@200k versus fully-random-init@200k
statistics are in `results/paperb_paired_analysis.json`: MMLU differs by +7.30pp
(McNemar p=5.96e-49; paired-bootstrap 95% CI [6.35,8.26]), and all five
knowledge/comprehension tasks have p<.05 with CIs excluding zero. The OLMo
same-model probe `results/probe_linguistic_olmo2_7b.json` gives semantic mean
sat95=0.073L and next-token sat95=1.000L; combined with MMLU sat95=0.594L, the
ordering is semantic < MMLU < next-token. The non-contiguous ShortGPT step-zero
sanity check (`olmo2_{ppl,downstream}_results/7B_shortgpt_step0*`) gives PPL
401.124, MMLU .2620, LAMBADA .00058, and HellaSwag acc_norm .391. It measures
immediate pruning damage only; healed cross-policy behavior is not yet claimed.
The keep12 PPL raw summary is
`olmo2_ppl_results/7B_keep12_step111500/summary.json` (PPL 11.56596; 4096
windows; 8,384,512 targets).

---

## 2. Method (reproducible recipe)

Source: `status/RUN_REGISTRY.md` L1932; `scripts/train_olmo2_arch_probe2.py`; arch_meta.json files.

- **Base model**: `models/OLMo-2-1124-7B` — pure pretrained BASE LM (no SFT/RL/instruct
  contamination). L=32 decoder layers, hidden_size 4096, vocab 100352, `tie_word_embeddings=false`
  (`outputs/olmo2_probe2_7B_keep14fresh2/arch_meta.json`). (1B variant
  `OLMo-2-0425-1B`, L=16, provides a small-scale replication point.)
- **Prune**: truncate to the front `keep` layers (`--keep_front_layers`).
- **Fresh tail**: append `n_fresh=2` freshly-initialized decoder layers (`--n_fresh_layers 2`).
  Transplant sanity checks at launch: copied front-K + embed/norm/lm_head tensors exact
  (`max|model−base|=0`), fresh `post_attn_ln`/`q_norm` all-ones, `fresh_q_std=0.02`
  (`RUN_REGISTRY.md` L1941).
- **Heal data**: dolmino-mix-1124 **DCLM subset**, re-tokenized with the OLMo-2 tokenizer →
  `data/dolmino_now15b.npy` (uint32 `[7,570,911, 2048]`, 15,505,225,728 packed tokens,
  62,020,902,912 bytes; staged to `/dev/shm/dolmino_now15b.npy`;
  `RUN_REGISTRY.md` L1943). Held-out val = `data/dolmino_now_val.npy`, shape `[4096, 2048]`
  (`OLMO2_PRUNEHEAL_PPL.md` L12).
- **Optimization (executed behavior, verified from logs)**: all arms share
  `seq_len 2048`, `max_steps 200000`, gradient checkpointing, fp32 master weights,
  `n_fresh 2`, corpus, and effective batch. Because parameter grouping was built
  after DDP wrapping, inherited distributed runs placed every trainable parameter
  in the inherited group: copied modules, fresh tail, and `lm_head` all used
  peak LR `2e-5` (minimum `2e-6`). In `from_scratch`, every parameter used peak
  LR `1e-4` (minimum `1e-5`). Thus learning rate is not matched. Effective batch
  is 128: H20 `bs4/ga4`, L20A `bs16/ga1`.
- **RED LINE**: no BABILong data is mixed into healing. The executed OLMo trainer
  reads only the single Dolmino `.npy` data source and has no
  `--babilong_mix_fraction` option.
- **Control-arm flags** (script-supported): `--freeze_front` freezes the kept front decoder blocks
  while fresh layers and non-block modules remain trainable; `--from_scratch` sets
  `transplant=False`, ignores all base weights, random-initializes the complete model, and trains all
  parameters at the fresh LR (`scripts/train_olmo2_arch_probe2.py` L286-306, L411-412, L514-521).
- **Launchers**: wzc1/L20A `scripts/run_olmo2_7B_keepN.sh`; diskB/H20 `scripts/run_olmo2_7B_keepN_diskB.sh`.
- **★ Eval 口径 (MANDATORY, per user MEMORY)**: OLMo-2 is a BASE LM → **base 口径 only**:
  raw 2048-tok NTP windows, NO chat_template, NO generation, NO BOS
  (`add_special_tokens=False` is a no-op for OLMo-2 tokenizer). Downstream = likelihood-based zero-shot
  MC (argmax over teacher-forced continuation log-prob, no generation). Compare only vs vanilla OLMo-2
  BASE (`OLMO2_PRUNEHEAL_DOWNSTREAM.md` L22-24, §Setup; MEMORY "Paper B OLMo-2 is BASE not chat").
- **Eval drivers** (zero arch drift — pruned-arch build copied verbatim from the trainer):
  `scripts/eval_olmo2_probe2_ppl.py` (+ `_run_olmo2_probe2_ppl_8gpu.sh`),
  `scripts/eval_olmo2_probe2_downstream.py` (+ `_run_olmo2_probe2_downstream_8gpu.sh` and knowledge
  variant). 8-GPU sharded `[g::8]`, token-weighted PPL merge `exp(Σnll/Σtok)`
  (`OLMO2_PRUNEHEAL_PPL.md` §Setup; `OLMO2_PRUNEHEAL_DOWNSTREAM.md` §Setup).

---

## 3. Results tables (with sources)

### 3(a) Depth-sweep held-out ppl / LM tax
Source: `status/OLMO2_PRUNEHEAL_PPL.md` Results table (L19-29).

| model | keep / arm | layers | frac depth | step | held-out PPL | tax vs full base |
|-------|-----------|-------:|-----------:|-----:|-------------:|-----------------:|
| OLMo-2-**7B** | **full base** (denom) | 32 | 1.00L | — | **7.398** | 1.00× |
| OLMo-2-7B | keep8+fresh2 | 10 | 0.3125L | 5000 (traj) | 22.331 | 3.019× |
| OLMo-2-7B | keep8+fresh2 | 10 | 0.3125L | 15000 (traj) | 17.868 | 2.416× |
| OLMo-2-7B | keep8+fresh2 | 10 | 0.3125L | 25000 (traj) | 16.426 | 2.220× |
| OLMo-2-7B | keep8+fresh2 | 10 | 0.3125L | 35000 (traj) | 15.612 | 2.110× |
| OLMo-2-7B | **keep8+fresh2** | 10 | 0.3125L | **44000 (main rung, under-healed 22%)** | **15.131** | **2.045×** |
| OLMo-2-7B | keep10+fresh2 | 12 | 0.375L | 10000 (early, NOT converged) | 17.239 | 2.330× |
| OLMo-2-7B | **keep12+fresh2** | 14 | 0.4375L | **111500 (55.75% healed)** | **11.566** | **1.563×** |
| OLMo-2-7B | **keep14+fresh2** | 16 | 0.50L | **128000 (apex)** | **10.826** | **1.463×** |
| OLMo-2-7B | keep14+fresh2 | 16 | 0.50L | 153500 (post-apex) | 10.693 | 1.445× |
| OLMo-2-7B | **from-scratch** (ctrl 2) | 16 | 0.50L arch | 200000 | 11.498 | 1.554× |
| OLMo-2-7B | **keep14+fresh2 FINAL** | 16 | 0.50L | **200000** | **10.561** | **1.428×** |
| OLMo-2-**1B** | full base (denom) | 16 | 1.00L | — | 10.642 | 1.00× |
| OLMo-2-1B | keep7+fresh2 | 9 | 0.56L | 147000 (latest, not plateaued) | 15.628 | 1.469× |

- **keep12 fills the 0.44L rung** (2026-07-27, 8-shard eval on .104 from wzc1 ckpt): held-out ppl 11.566
  at step 111500 (55.75% healed) — cleanly between keep14 apex 10.826 and fully random-init 11.498, with
  the same n=8384512 tokens / 4096 windows over `data/dolmino_now_val.npy` protocol
  (`olmo2_ppl_results/keep12_wzc1_step111500_ppl/summary.json` on .104 diskB).
- **keep8 trajectory + main rung filled** (2026-07-27, 5-point sweep on .104 diskB): under-healed 22%
  (44k/200k steps), PPL monotonically drops 22.33 → 15.13 across step 5k → 44k. It remains worse
  than from-scratch at 200k (11.50), but depth, initialization, and optimization stage differ, so the
  gap cannot be attributed to compute alone; see 3(b) for the available-checkpoint capability frontier.
- keep10 remains the only untrained-past-10k 7B point (step 10000 = 5% of 200k). New keep10 heal
  RESUMED on .82 (2026-07-27 05:27+08:00, resume from step10000.pt, target 200000, 6.79 s/step on H20
  → ETA in days; ckpts every 500 steps allow midway eval).
- **All three new points use the same protocol as keep14 apex** (same `data/dolmino_now_val.npy`, same
  bf16 autocast + fp32-master, same 2048-token windows, 8-shard sum-nll merge) — apples-to-apples with
  §POST-APEX numbers.

### 3(b) Downstream MC + MMLU/knowledge per arm (7B; core MC uses acc_norm, MMLU/WinoG/LAMBADA/BoolQ/CSQA/SIQA use raw acc)
Sources: `OLMO2_PRUNEHEAL_DOWNSTREAM.md` core table (L55-58), knowledge table (L145-148), §CONTROL2
(L352-354, L376-379).

| arm | step | MMLU | HS accn | ARC-C accn | ARC-E accn | PIQA accn | WinoG | OBQA accn | lambada | boolq acc | CSQA acc | SIQA acc |
|-----|-----:|-----:|--------:|-----------:|-----------:|----------:|------:|----------:|--------:|----------:|---------:|---------:|
| **7B base full (32L)** | — | **.605** | .805 | .571 | .829 | .811 | .744 | .462 | .732 | .815 | .665 | .502 |
| keep8 (10L) UNDER-healed | 44000 | **.2463** | .469 | .314 | .610 | .694 | **.519** | .332 | .433 | .588 | .442 | .400 |
| keep10 (12L) early | 10000 | .254 | .443 | .303 | .585 | .673 | .524 | .326 | .404 | .598 | .421 | .378 |
| **keep12 (14L) 55.75% healed** | 111500 | **.2726** | .596 | .407 | .688 | .736 | .615 | .376 | .530 | .610 | .508 | .415 |
| **keep14 (16L) apex** | 128000 | **.3012** | .631 | .426 | .702 | .747 | .630 | .402 | .575 | .639 | .505 | .423 |
| keep14 (16L) post-apex | 153500 | **.3124** | .643 | .442 | .705 | .745 | .633 | .406 | .570 | .606 | .506 | .441 |
| **keep14 (16L) final** | 200000 | **.3191** | .645 | .438 | .705 | .745 | .626 | .404 | .577 | .638 | .499 | .434 |
| **from-scratch (16L) ctrl** | 200000 | **.2461** | .578 | .414 | .697 | .733 | .545 | .384 | .484 | .614 | .450 | .416 |

**Above-chance MMLU recovery** ((x−.25)/(.6053−.25)): keep14 apex **14.4%**, post-apex
**17.6%**, final **19.5%**, **keep12 111500 6.4%**, **keep8 44000 ≈0% (chance floor .2463)**,
from-scratch **≈0%**
(`OLMO2_PRUNEHEAL_DOWNSTREAM.md` L243, L300, L385; new: `olmo2_downstream_results/7B_keep12_step111500_know/summary.json`,
`olmo2_downstream_results/7B_keep8_step44000_know/summary.json` on .104 diskB). Per-subject: keep14
lifts world_religions to .427, us_foreign_policy to .460, and marketing to .385 at the 128k apex.
The fully random-init model is lower on these subjects (.322/.340/.312), but subject-level effects
are heterogeneous rather than uniformly absent. Sample-weighted keep14-post group accuracy is
.295 STEM, .289 humanities, .329 social science, and .366 other; see the generated complete
57-subject appendix table.

**★ Capability cliff between keep8 and keep12 (2026-07-27 result).** MMLU (chance = .25):
keep8@44k **.2463** = chance, keep12@111.5k **.2726** = +2.63pp above chance, keep14 apex **.3012** =
+5.12pp above chance. WinoGrande: keep8 **.519** ≈ chance 50%, keep12 **.615**, keep14 **.630**.
At the available checkpoints, keep8 and early keep10 are at chance while keep12 and keep14 are above
chance. This brackets an **available-checkpoint knowledge-retention frontier**, not a converged
architectural threshold: keep8 was trained for only 22% of the budget, keep12 for 55.75%, and the
arms are not compute matched. A fully trained or matched-step keep8/10/12 sweep is required to
separate depth from optimization time.

**★ Within-keep8 test against uniform under-training (2026-07-27).** keep8 downstream trajectory
across step 10k → 25k → 44k (4.4× compute range, all trained on identical Dolmino and evaluated on
the same benchmark protocol). PPL was evaluated on a partially different 5k/15k/25k/35k/44k grid:

| Step | ppl | MMLU acc | HS accn | ARC-C accn | WinoG acc | lambada acc | boolq acc |
|-----:|----:|---------:|--------:|-----------:|----------:|------------:|----------:|
| 10000 | — (17.87@15k) | **.2542** | .3915 | .2654 | **.5209** | .3429 | .5713 |
| 25000 | 16.43 | **.2502** | .4390 | .3114 | **.5083** | .3827 | .6080 |
| 44000 | 15.13 | **.2463** | .4694 | .3140 | **.5185** | .4333 | .5881 |

**Dissociation is stark.** In-context / surface / reasoning heals monotonically over heal steps:
hellaswag_norm +7.79pp (.39 → .47), arc_c_norm +4.86pp (.27 → .31), lambada +9.04pp (.34 → .43),
and boolq +1.68pp (.57 → .60) from 10k→44k. On its separate 5k→44k grid,
held-out PPL falls 22.331→15.131 (-32.2%); PPL 17.869 is the 15k point and must
not be paired with the 10k downstream row. But **MMLU stays at chance floor**
(24.63-25.42%, max deviation 0.79pp from chance 25.00%) and **WinoGrande stays at chance**
(50.83-52.09%). Over the observed interval, 4.4× more steps lift several tasks but not MMLU or
WinoGrande. This strongly disfavors a **uniform-delay** explanation in which all capabilities improve
on the same schedule. It does not rule out later MMLU recovery after substantially more training and
therefore cannot by itself prove an architectural threshold. Sources on .104 diskB:
`olmo2_downstream_results/7B_keep8_step{10000,25000,44000}{,_know}/summary.json`.

**inherited keep14 vs fully random-init deltas** (accn above-chance recovery, §CONTROL2 L360-366, L383-389):
WinoGrande healed 55% vs scratch **18%** (+37pt); HellaSwag +12pt; ARC-C +9pt; OBQA +11pt; lambada
+12pt; siqa +15pt; csqa +12pt. Surface (ARC-E, PIQA) ≈tie; boolq ≈tie (in-context). MMLU: healed 17.6%
vs scratch 0% — the decisive gap.

Base-full sanity (铁律2, reproduces published OLMo-2): 7B HS accn **.805** (public ~.80-.81), MMLU
**.605**, lambada **.732** (`OLMO2_PRUNEHEAL_DOWNSTREAM.md` L37-38, L126-127) → drivers validated.

### 3(c) Continued-healing trajectory (keep14, final 200k best PPL/MMLU)
Source: `OLMO2_PRUNEHEAL_PPL.md` L35-36; `OLMO2_PRUNEHEAL_DOWNSTREAM.md` §POST-APEX table (L300-310).

- PPL: 128000 **10.826** → 153500 **10.693** (only −0.133, plateaued). No overfitting/regression with
  +25.5k extra heal steps.
- Downstream 128k→153.5k: MMLU .3012→.3124 (+.011), HS accn .631→.643, ARC-C .426→.442, siqa
  .423→.441 (all flat-to-slightly-up); boolq acc .639→.606 (down, but accn=.682 = yes/no
  length-norm wash, comprehension ≈flat; L307-314).
- **Verdict**: late healing continues to improve PPL modestly, while MMLU gains only
  0.67 points from 153.5k to 200k. This supports different observed recovery rates
  but does not rule out later recovery under longer or different training.
- Training log tails confirm DONE at step200000: keep14 healed `loss=2.3182 ppl=10.16` (train)
  (`logs/olmo2_7B_keep14fresh2.log` last line); from-scratch `loss=2.4138 ppl=11.18` (train)
  (`logs/olmo2_7B_keep14fresh2_fromscratch.log` last line).

### 3(d) Control arms
| control | status | held-out ppl | MMLU | verdict |
|---------|--------|-------------:|-----:|---------|
| **from_scratch** (16L, fully random init) | ✅ DONE step200000 (`final.pt` Jul 25; `logs/...fromscratch.log` ends step200000) | 11.498 / 1.554× | **.2461 = chance floor** | Under this budget, the evaluated random-init operating point reaches competitive PPL but not MMLU; raw BoolQ is similar. All modules are random and peak LR is 1e-4 rather than the inherited arm's 2e-5, so this is not an initialization-only ablation (`OLMO2_PRUNEHEAL_DOWNSTREAM.md` §CONTROL2; trainer code and logs) |
| **freeze_front** (front-14 frozen, fresh tail and non-block modules trainable) | ✅ DONE step200000 | **12.797 / 1.730×** | **.2628 / 3.6% recovery** | Learning-rate matched to train-all; reuse without inherited-prefix adaptation performs worse on PPL and retains little above-chance MMLU (`olmo2_{ppl,downstream}_results/7B_freezefront_step200000*/summary.json`; `n_trainable=1.227B`) |

- Frozen-front is not globally dominated by random init: it has worse PPL but higher MMLU,
  WinoGrande, and LAMBADA, producing a metric-dependent ordering across controls.

---

## 4. Mechanism — two depth markers (P2) and their correlation with the keep-N frontier

Sources: `ops/research_notes/new_propositions_20260725.md` §5 (P2 RESULTS, VERDICT SUPPORTS);
`results/knowledge_logit_lens_OLMo-2-1124-7B.json`; `results/knowledge_logit_lens_Qwen3-8b-local.json`;
`results/probe_linguistic_qwen3_8b.json`.

**Three saturation curves (fractional depth):**

| curve | Qwen3-8B (36L) | OLMo-2-7B (32L) | source |
|---|---|---|---|
| linguistic/semantic sat95 (WiC/SST2/RTE, linear probe) | **0.13L** | (Qwen3 cross-model ref 0.13L) | `probe_linguistic_qwen3_8b.json`; `new_propositions` §5 |
| **knowledge (MMLU logit-lens) sat95** | **0.694L** (L25) | **0.594L** (L19) | `knowledge_logit_lens_*.json` summaries |
| knowledge onset (chance+0.05) | 0.694L (L25) | **0.562L** (L18) | ibid |
| next-token (logit-lens) sat95 | **0.944L** (L34) | (Qwen3 ref 0.94L) | `probe_linguistic_qwen3_8b.json` |

- **Sharp step, not gradual ramp**: OLMo MMLU acc L17→L18→L19 = 0.251→0.326→0.544; below onset it sits
  at chance (~0.23–0.26) through the entire early/mid stack
  (`knowledge_logit_lens_OLMo-2-1124-7B.json` per_layer; `new_propositions` §5 L175). Qwen3 L24→L25 =
  0.236→0.621.
- **Gold-answer score keeps increasing after argmax accuracy plateaus**: OLMo gold-letter LL
  −8.52@L19 → −3.79@L32 while accuracy is nearly flat. This is not by itself a calibration
  measurement; it only shows that deeper layers increase the gold answer's score (§5 L176).

**How it relates to the available keep-N frontier** (`new_propositions` §5 verdict L186-192):
- The descriptive depth ordering holds in both models: MMLU sat95 (0.59–0.69L) is
  **≥2× deeper** than the exploratory semantic marker (~0.13L; 4.6×/5.3×). This is a separation of
  readout depths, not proof of functionally independent storage systems.
- **MMLU decodability onset (OLMo 0.562L) lies above the available keep-N frontier**: early keep10 at
  cut depth 0.3125L is at MMLU chance (.254), while keep14 at 0.4375L is partially above chance
  (.301 at 128k). The ordering is directionally consistent, but the ~0.12L gap is looser than a strict
  ±0.1L alignment and does not localize where knowledge is stored.
- **Honest framing (state this caveat)**: logit-lens *decodability* (0.56L) lags the layers that
  *install* knowledge (the compute cliff 0.31–0.44L precedes residual-stream readability 0.56L). Frame
  as "decodability lags installation," NOT "coincident boundaries" (§5 L190-192).
- **Payoff**: one figure ("three saturation curves, two depths") is a shared mechanistic section — the
  shallow semantic depth is where CoMem (Paper A) splits near-losslessly at j=12=0.33L; the deep
  knowledge-sensitive depth is where Paper B's observed keep-N frontier lies and where
  prune-heal loses MMLU signal under the measured budgets.

---

## 5. Figures needed

1. **Depth-sweep held-out PPL curve** — x = resulting depth `(keep+2)/32`, y = held-out PPL;
   points: keep8 (0.3125L), keep10 (0.375L, very early), keep12 (0.4375L), keep14 (0.50L,
   10.561/1.428× at 200k), and full base (1.00L, 7.398); overlay fully random-init (11.498) at 0.50L.
   All labels must show step because the inherited points are not compute matched.
2. **Capability-cliff bar chart** — above-chance recovery per task for final keep14@200k versus
   from-scratch@200k, grouped descriptively as surface / reasoning / comprehension /
   **knowledge-sensitive (MMLU)**, with inherited MMLU recovery at 19.5%. (data: §3b).
3. **Logit-lens 3-curve depth plot** — semantic (0.13L), knowledge MMLU (OLMo 0.594L / Qwen 0.694L),
   next-token (0.94L) vs fractional depth; annotate keep10/keep14 cliff positions + knowledge onset.
   (data: §4; JSON per_layer arrays).
4. **Training trajectory** — keep14 held-out PPL and MMLU at 128k/153.5k/200k;
   show that PPL continues improving modestly while MMLU rises much more slowly (data: §3c).

---

## 6. Related work to cite

- **Layer pruning / layer-drop depth studies**: ShortGPT (layer-importance drop); LaCo (Layer Collapse,
  merging); **Gromov et al. "The Unreasonable Ineffectiveness of the Deeper Layers"** (deep layers
  prunable with light healing) — our result *refines* theirs: PPL and MMLU recover at markedly different rates
  under the evaluated budgets, while the random-init operating point reaches competitive PPL but not
  the inherited model's MMLU signal. Its different learning rate prevents an initialization-only claim.
- **Depth up-scaling / layer addition**: SOLAR 10.7B (depth up-scaling) — inverse operation; our
  `n_fresh=2` tail append is a small-scale analogue.
- **Healing / continued pretraining after compression** (the "heal" step justification).
- **Knowledge localization in transformer MLPs**: ROME; Geva et al. "Transformer Feed-Forward Layers Are
  Key-Value Memories" / knowledge-neuron work motivates a depth hypothesis. These studies do not
  directly explain our MMLU result, and our logit lens is not a causal storage localization.
- **Logit-lens / stages of inference**: nostalgebraist logit-lens; "Stages of Inference" (2406.19384);
  DoLa / Layer-Fused Decoding (readout-from-layers) — context for the two-depths probe (§4;
  `new_propositions` §5 novelty note).

The current manuscript cites the corresponding entries collected in `paperB/paperB.bib`; repeat the bibliography audit if the related-work scope changes.

---

## 7. Open holes / what's still needed before submission

1. **Depth ladder remains non-compute-matched.** keep8 (44k), keep12 (111.5k), and keep14
   (128k/153.5k/200k) have PPL and downstream evaluations; keep10 remains very early. A clean
   architectural threshold requires fully trained or matched-step keep8/keep10/keep12/keep14 comparisons.
2. **Onset-cliff alignment is directional, not tight** (~0.12L gap). It must remain framed as
   "decodability lags installation" (§4), not as an exact shared boundary.
3. **External-method universality remains open.** Frozen-front now completes Paper B's internal
   train-all / freeze-front / random-init triad, but ShortGPT-, LaCo-, merging-, and
   distillation-based compression are not evaluated under the same protocol.

---

# 8. Status snapshot (2026-07-27, for the reviser) + FULL depth-ladder

This file is the single hand-off doc for revising Paper B. Everything above is sourced
inline. This section adds (a) the completed depth-ladder as of today, (b) live status of the
last control arm, (c) figure inventory + gpt-image2 prompts.

## 8.1 Depth-ladder — COMPLETE core (held-out ppl + downstream + knowledge), same protocol

n=8384512 tok / 4096 windows on `data/dolmino_now_val.npy`, bf16 autocast + fp32-master,
8-shard sum-nll merge. Downstream = likelihood-MC, 8-shard. All measured 2026-07-27.

| arm | keep frac | step | held-out ppl | MMLU acc | core-6 avg acc_norm | verdict |
|---|---|---|---|---|---|---|
| full base (denom) | 1.00L (32L) | — | 7.398 | .6053 | — | reference |
| keep14 apex | 0.50L | 128000 | **10.826** | **.3012** (+14.4%) | ~.62 | trajectory |
| keep14 post-apex | 0.50L | 153500 | 10.693 | .3124 (+17.6%) | ~.63 | trajectory |
| **keep14 final** | 0.50L | 200000 | **10.561** | **.3191** (+19.5%) | ~.63 | canonical final |
| **keep12** | 0.4375L | 111500 (55.75% healed) | **11.566** | **.2726** (+6.4%) | **.5696** | on-trajectory rung |
| keep10 | 0.375L | 10000 | 17.239 | .2540 (chance) | — | very early rung |
| **keep8** | 0.3125L | 44000 (22% healed) | **15.131** | **.2463** (chance) | **.4887** | **below cliff** |
| from_scratch (ctrl) | 0.50L arch | 200000 | 11.498 | .2461 (chance) | ~.55 | random-init control |

## 8.2 Capability frontier + within-arm test (2026-07-27)

- **Cliff between keep8 and keep12**: MMLU keep8 .2463 (chance) → keep12 .2726 (+2.6pp) →
  keep14 .3012 (+5.1pp). WinoGrande: keep8 .519 (≈chance) → keep12 .615 → keep14 .630.
- **Within-arm uniform-delay test (keep8 trajectory step 10k→25k→44k, 4.4× steps):**
  | step | ppl | MMLU | HS_norm | ARC-C_norm | WinoG | lambada |
  |---|---|---|---|---|---|---|
  | 10000 | — (17.87@15k) | .2542 | .3915 | .2654 | .5209 | .3429 |
  | 25000 | 16.43 | .2502 | .4390 | .3114 | .5083 | .3827 |
  | 44000 | 15.13 | .2463 | .4694 | .3140 | .5185 | .4333 |
  → On the separate PPL grid, 5k→44k gives 22.33→15.13 (−32.2%); on the downstream
  grid, 10k→44k gives HellaSwag +7.8pp and LAMBADA +9pp, but **MMLU stays at chance
  (24.6–25.4, max 0.8pp from chance) and WinoGrande remains near chance (50.8–52.1)**. This strongly disfavors uniform capability
  delay over the observed interval, but does not exclude later recovery or prove a hard architectural
  threshold before convergence.
  Source: `olmo2_downstream_results/7B_keep8_step{10000,25000,44000}{,_know}/summary.json` on .104.
- **keep12/keep8 held-out ppl**: `olmo2_ppl_results/{keep12_wzc1_step111500_ppl,keep8_step44000_ppl}/summary.json`.

## 8.3 Final control arm

- **freeze_front** (front-14 frozen; fresh tail and non-block modules trainable;
  n_trainable=1.227B) completed at **200000/200000**. Held-out PPL is **12.797**
  and MMLU is **.2628** (3.6% above-chance recovery). Full downstream and
  subject-level outputs are in `olmo2_downstream_results/7B_freezefront_step200000{,_know}/`.

## 8.4 Mechanism cross-ref (do NOT expand into CoMem/Qwen3 material)

Paper B's load-bearing mechanism evidence is OLMo MMLU logit-lens decodability at 0.56–0.59L,
which lies above the available cut frontier. The Qwen semantic 0.13L and next-token 0.94L values are
exploratory cross-model references, with 0.13L defined relative to final-layer probe scores.
Frame `from_scratch` conservatively: under this corpus and budget, a fully random-init model learns
useful in-context behavior but does not recover MMLU signal comparable to the inherited model. Do
not localize composition to the tail or knowledge to the trunk from this control, because every
module is random in `from_scratch`. The broader CoMem readout-depth /
adapter scale-law material lives in Paper A / prospective Paper D — do NOT import it here.

---

# 9. Figure inventory + generation prompts

Four **data figures** are already generated by matplotlib from the tables above (keep them
matplotlib — they need exact numbers, gpt-image2 is not suitable for precise data plots):

1. **fig_depth_ppl** — x = resulting depth `(keep+2)/32`, y = held-out PPL.
   Points: keep8 0.3125L/15.13, keep12 0.4375L/11.57, keep14 0.50L/**10.56@200k**,
   base 1.00L/7.40; overlay fully random-init at 0.50L/11.50. Data: §3(a) + §8.1.
2. **fig_capability_cliff** — grouped bar, above-chance recovery % per task family
   (surface / reasoning / comprehension / **knowledge=MMLU**), matched keep14@200k vs fully
   random-init@200k. MMLU is at chance for random initialization; this is a whole-initialization
   comparison, not a decoder-block ablation. Data: §3(b), §1 claim 2–3.
3. **fig_two_depths** — three saturation curves vs fractional depth (semantic 0.13L,
   knowledge 0.59–0.69L, next-token 0.94L), shade the keep-N cliff band. Data: §4.
4. **fig_keep8_falsification** (NEW — worth adding) — keep8 trajectory: x = heal step
   (10k/25k/44k), twin panel: left y = ppl (falling 17.9→15.1), right y = MMLU & WinoGrande
   (near chance). Visual: several measured tasks improve while MMLU does not over this interval. Data: §8.2.

## 9.1 gpt-image2 prompt — conceptual HERO figure (Figure 1)

Only the conceptual schematic is suitable for gpt-image2. Use this prompt; **manually fix any
garbled text labels afterward** (image models are unreliable on text — consider adding labels in
LaTeX/Illustrator over a text-free version, or accept and correct):

> PROMPT (gpt-image2):
> "A clean, minimal scientific diagram for a machine-learning paper, white background, flat vector
> style, muted academic color palette (slate blue, warm gray, one amber accent). Show a tall vertical
> stack of 32 thin horizontal rectangles representing the layers of a language model, numbered
> conceptually from bottom (input) to top (output). Divide the stack into three labeled depth zones
> with soft color bands: bottom ~13% 'semantic features' (light blue), middle band around 55–69% of
> the height 'MMLU decodability' (amber), and the very top ~6% 'next-token output' (gray). Draw a
> horizontal dashed red cutting line at roughly 44% of the height labeled 'prune here (keep front 44%)',
> with the region ABOVE the cut faded/greyed-out and marked 'discarded', and two small fresh green
> blocks re-attached just above the cut labeled '2 fresh layers (healed)'. To the right, two outcome
> callout boxes connected by thin arrows: one green check 'Perplexity recovers (10.561 vs 7.398;
> 1.428x at 200k)' and one red cross 'MMLU recovery lags (19.5% of above-chance signal at 200k)'.
> Elegant, publication-quality, lots of
> whitespace, no photorealism, no 3D, thin clean lines, sans-serif labels."

Fallback: the repo already has a precise TikZ version at `paperB/figures/fig1_concept.tex` — if
gpt-image2 text comes out garbled, keep the TikZ one (it is exact and compiles).
