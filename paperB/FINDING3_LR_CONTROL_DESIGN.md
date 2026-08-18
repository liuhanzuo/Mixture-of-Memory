# Finding 3 — matched-LR single-variable control (design note)

Scratch note for the ARR-reviewer "Finding 3 CONFOUNDED" audit. Path:line anchors
point at the exact claims/config so MAIN can log the resolution. **This note edits
nothing else** (no TODOList / .tex / versions / status writes).

Created 2026-08-04. Author context: control-design + launch subagent.

---

## 1. What Finding 3 claims, and the exact confound

**Finding 3 = "construction determines the recovery endpoint"**
(`paperB/sections/04_experiments.tex:54`). It rests on TWO sub-comparisons:

### (A) ShortGPT vs keep14 — NOT the confounded one
`paperB/sections/tab_construction_audit.tex:10-15` + `04_experiments.tex:57-70`.
Both models are 16L @ 200k and the caption states **"same peak LR"**
(`tab_construction_audit.tex:18-19`). ShortGPT beats keep14 by +15.5pp letter-MMLU.
Both sit in the same 2e-5 regime → the reviewer's LR flag does **not** apply here.

### (B) Same-shape operating points — THIS is the confounded comparison
`paperB/sections/tab_control.tex:10-13` and the paired table
`paperB/sections/tab_paired_operating_points.tex:10-12` + `04_experiments.tex:72-80`.

| Arm (tab_control) | Init | Trainable | Peak LR | PPL | MMLU-L |
|---|---|---|---|---|---|
| inherited, train all (**keep14**) | inherited front-14 + fresh tail-2 | all | **2e-5** | 10.561 | .3191 |
| inherited front frozen (**Frozen**) | inherited + fresh | fresh+norm+head | **2e-5** | 12.797 | .2628 |
| fully random init (**Random**) | random 16L | all | **1e-4** | 11.498 | .2461 |

Headline paired gap: **keep14 − Random = +7.14pp** letter-MMLU
(`tab_paired_operating_points.tex:10`, 95% CI [6.17,8.10], McNemar p=9.2e-47).

**THE CONFOUND (the two arms differ in ≥2 variables):**
- keep14 = **inherited** construction @ **uniform 2e-5**
- Random  = **fully random-init** construction @ **1e-4** (fresh bucket)

So keep14-vs-Random conflates **(i) initialization/construction** with
**(ii) learning rate** (2e-5 vs 1e-4). One cannot attribute the +7.14pp gap to
"construction determines the endpoint" without holding LR fixed.

**Why the LR differs — a code fact, not a design choice**
(`paperB/TODOList.md:170`): in the original keepN ladder, `build_param_groups`
saw DDP-wrapped names (`module.…`) and `_classify_param` failed to strip the
prefix, so the fresh tail + lm_head fell into the *inherited* 2e-5 bucket instead
of the intended fresh 1e-4. Net effect: **keep14 and Frozen endpoints are BOTH
uniform 2e-5.** The `from_scratch` (Random) arm is exempt because
`_classify_param` returns `"fresh"` first for `from_scratch`
(`scripts/train_olmo2_arch_probe2.py:314-315`), so all its params trained at
`--lr = 1e-4`. Hence Random is the ONE arm sitting at a different LR.

**The paper already hedges (B) in-text** (so this is a "confounded observation",
not a false claim): `tab_control.tex:16-19` ("Random init does not isolate
initialization because its learning rate also differs … neither comparison alone
identifies a causal factor"); `04_experiments.tex:77-79`; and
`app_tab_recovery` caption in `paperB/main.aux:1152` ("operating-point comparison
rather than an initialization-only … ablation"). The clean control below either
(a) lets Finding 3 make an initialization-causal statement, or (b) confirms the
current hedged "operating-point" wording is the right ceiling.

---

## 2. The minimal single-variable (init-only) control

The 2×2 that fully separates init from LR (`paperB/TODOList.md:530-535`):

| Init | Peak LR | Status |
|---|---|---|
| inherited | 2e-5 | DONE = keep14 (10.561 / .3191) |
| inherited | 1e-4 | not needed for THIS confound |
| **random** | **2e-5** | **← the missing cell; = task #127** |
| random | 1e-4 | DONE = Random (11.498 / .2461) |

To disentangle keep14(inherited,2e-5)-vs-Random(random,1e-4), the single needed
cell is **random-init @ uniform 2e-5**. Compared against keep14 @ uniform 2e-5,
the ONLY changed variable is initialization (inherited-prefix+fresh-tail vs fully
random), with corpus / shape (16L keep14+fresh2) / batch (eff_bs 128) / schedule
(cosine 200k, warmup 150) / trainable-set (all) / seed (42) all matched.

**This control already exists as task #127** and its `arch_meta.json` on .73
confirms it is exactly this cell:
`outputs/olmo2_p13_scratch16_lr2e5_uniform/arch_meta.json` →
`from_scratch:true, lr_fresh:2e-05, lr_inherited:2e-05, keep_front:14,
n_fresh:2, num_hidden_layers:16, n_params:4.06B, seed:42`.

#127 was **STOPPED EARLY at step ~26.9k / 200k (13.4%)** — a deliberate kill (not a
crash) to free .73+.82 for Paper A (`paperB/TODOList.md:522`); no held-out
PPL/MMLU endpoint was produced. So the DESIGN is correct and validated, but the
comparable **200k endpoint is missing**. Milestone checkpoints
step5000/10000/15000/20000/**25000**.pt survive on the shared FS (each carries
optimizer+RNG+step → clean resume).

**Design decision: RESUME #127 from step25000.pt**, not restart. Resuming
preserves the identical trajectory (same seed, same cosine horizon 200k → LR at
step 25k is unchanged) and salvages ~25k steps of prior compute. This is the
correct, non-misdesigned single-variable control.

### Matched-config table (control vs its keep14 reference)
| knob | keep14 (done) | this control (#127 resume) | matched? |
|---|---|---|---|
| shape | 16L (keep14+fresh2) | 16L (keep14+fresh2 shell) | ✓ |
| init | inherited front-14 + fresh-2 | fully random (from_scratch) | **← the 1 variable** |
| peak LR | uniform 2e-5 | uniform 2e-5 (`--lr 2e-5 --lr_inherited 2e-5`) | ✓ |
| trainable | all | all | ✓ |
| corpus | dolmino_now15b.npy | dolmino_now15b.npy | ✓ |
| eff_bs | 128 | 128 (2×8×8) | ✓ |
| schedule | cosine 200k, warmup 150 | cosine 200k, warmup 150 | ✓ |
| seed | 42 | 42 | ✓ |

---

## 3. Launch (node .73, 8×H20, single node)

- Launcher: `scripts/run_olmo2_p13_resume_1node_73.sh` (new; single-node
  `torchrun --standalone`, no IB needed).
- eff_bs held at 128 on 8 cards via **bs=2 × ga=8 × 8 GPU** (matches #127's
  16-card bs=2×ga=4×16 and keep14's bs=16×ga=1×8).
- `--resume_from …/olmo2_p13_scratch16_lr2e5_uniform/step25000.pt`,
  `--output_dir outputs/paperB_finding3_lr_control_randinit2e5/`,
  `--max_steps 200000`, `EVAL_INTERVAL` N/A (this trainer has no inline eval),
  `--save_every 5000 --gradient_checkpointing 1 --seed 42`.
- Memory: #127 ran bs=2 at maxmem 98.3 GB on H20 (≈100% of 97.8 GB) → bs=2 is the
  memory-safe maximum; bs≥4 would OOM AND break the eff_bs match. Do NOT raise bs.

## 4. Honesty / budget note (for MAIN's downgrade-vs-complete decision)
- The control is **correctly designed and now resumable**, NOT misdesigned.
- Full 200k endpoint from step25k = 175k steps. On 8×H20 ≈ 9–10 s/step (2× #127's
  16-card 4.68 s/step) → **~18–20 days / ~3.5k GPU·h**. This is why #127 was killed.
  合成 16 卡 (.73+.104/.82) would roughly halve wall-clock (~9–10 days).
- Milestone ckpts every 5000 steps → a matched-STEP directional readout (does
  random-init@2e-5 stay at chance MMLU like random-init@1e-4, or climb toward
  keep14?) is harvestable well before 200k. random-init@1e-4 sat at chance MMLU
  throughout; if @2e-5 does the same at 75–100k while keep14 is well above chance,
  that already shows **init (not LR) drives the gap** → Finding 3 stands.
- If MAIN/user judge the full 200k not worth it, the alternative is to keep the
  paper's existing hedged "operating-point comparison" wording (Finding 3 as a
  confounded observation), which the .tex already states.
