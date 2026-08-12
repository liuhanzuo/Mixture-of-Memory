# ⚠️ `pilot_one_stageB_S2_verdict.json` IS MISLABELED — read this before using it

Written by MAIN 2026-08-12 11:25 GMT+8, after nearly briefing a subagent off the
wrong baseline.

## The defect

`evidence/pilot_one_stageB_S2_verdict.json` (mtime 2026-08-12 10:24, **UNTRACKED
in git**) carries Stage **A**'s header on Stage **B**'s data.

| field | what it says | truth |
|---|---|---|
| `gate` | `"A04_pilot_one_stage_A_free_sd_run"` | it is Stage **B** |
| `one_directional_caveat.why_not` | "wrong arm: keep7+fresh2 … wrong budget: 20,000 steps … n=2 yields a range" | **all three are false for this file** |
| `provenance.seeds` | `["101","102"]` | ← this is the field that gives it away |
| `provenance.dir_template` | `A04_1B_stageB_keep12_seed{seed}_step5000` | keep12+fresh2 @ 5000 steps = exactly the arm and budget Pilot One was designed for |

The `gate` string and the caveat block were evidently copy-pasted from the real
Stage A file and never updated.

## Why this matters

There are now **two** files with the identical `gate` string
`"A04_pilot_one_stage_A_free_sd_run"` and **different numbers**:

| file | mtime | triviaqa sd_run | popqa | mmlu_content | nq_open |
|---|---|---|---|---|---|
| `pilot_one_stage_a_verdict.json` | 04:53 | 0.3231 | 0.2726 | 0.0252 | 0.0 |
| `pilot_one_stageB_S2_verdict.json` | 10:24 | 0.2877 | 0.4510 | 0.0554 | 0.2938 |

The 04:53 file is the genuine Stage A: A03 seeds 43/44, keep7+fresh2, 20k steps,
n=2 — the free check that structurally **could never** emit `K2_CLEARED`.

The 10:24 file is Stage B seeds 101/102 — the right arm, the right budget. Its
inherited "this can never clear K2" caveat therefore does **not** automatically
apply, because all three of the stated reasons (wrong arm / wrong budget / n=2)
are fixed by Stage B. Whether Stage B *can* clear K2 is a question for
`PILOT_ONE_PREREG.md`, not for a copy-pasted field.

## State of the three Stage B seeds

All three trained to `step5000.pt` on **2026-08-11** and sat unharvested:

- `outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed101/step5000.pt`
- `outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed102/step5000.pt`
- `outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed103/step5000.pt`

Seeds **101 and 102 have eval numbers** (in the 10:24 file). **Seed 103 does
not.** So the remaining work may be much smaller than a full 3-seed eval — check
whether `A04_1B_stageB_keep12_seed103_step5000` results already exist on zwfy6
before spending GPU.

## What the harvesting agent must do

1. Compute the **S=3** statistic per `PILOT_ONE_PREREG.md` — a real sd over three
   draws, **not** Stage A's `|m_a − m_b| / sqrt(2)`.
2. Re-verify seeds 101/102 result dirs against the integrity asserts (8/8 shards,
   exact item counts, no duplicate ids, `nan == 0`) rather than trusting the
   summary. The 10:24 file says its loaders were imported from A03's
   `code/recompute_cpt_trajectory_paired.py` and carry those asserts — confirm.
3. Emit a **correct** `gate` string and a **Stage-B-specific** caveat block.
4. State plainly that this file is superseded. **Do not delete it** — it is
   provenance for the 101/102 evals.

## Pre-registered decision inputs (unchanged, for convenience)

Δ per axis, pp: triviaqa `4.043134195274186`, popqa `1.3205298941613512`,
mmlu_content `1.0238926078906136`, nq_open `0.9695290858725762` (demoted, no
decision weight). Rule: K2 FIRES iff `bound_S > Δ` on ≥ 2 of the 3 decision axes.
`mmlu_content` is the mandated MMLU interface — the letter interface is **banned**
as a decision axis by A04 design §4.2.

Under the 10:24 file's own S=2 numbers, 0 of 3 decision axes exceed Δ.
