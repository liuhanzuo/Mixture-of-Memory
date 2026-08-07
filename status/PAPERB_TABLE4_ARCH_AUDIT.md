# Paper B Table 4 architecture audit — each rung matched to its measurement disk

**Date**: 2026-08-08 CST. **Author**: sub-agent, dispatched by MAIN for task #189.
**Scope**: audit which architecture (L20A/wzc1 or H20/zwfy6) produced every load-bearing
number the paper quotes for the depth ladder. **GPU cost**: 0.

Cross-references (owned by MAIN, do not edit): `PAPERB_CORE6_CROSSARCH_FLOOR.md`,
`PAPERB_TABLE4_BUDGET_DEFECT.md`.

---

## 🚨 Correction to `PAPERB_CORE6_CROSSARCH_FLOOR.md` §"Paper provenance, now pinned"

That doc states (lines 110–113):
> "Table 4's base row `0.7037` is the H20 number … the keep14 row (`.5938`, also H20 —
> `zwfy6` anchor) [is] at least consistent with each other in architecture. That is the
> good case."

**The keep14 attribution is wrong.** Recomputed core6:
- `wzc1:7B_keep14_step200000` = **.59376** → rounds to **`.5938`** (paper's value)
- `zwfy6:7B_keep14_step200000` = **.59532** → rounds to `.5953` (NOT the paper)

So base (`.7037`) is zwfy6/H20 but keep14 (`.5938`) is **wzc1/L20A**. The base and
keep14 rows are on **different architectures**. Table 4's core6 column mixes disks
across rungs — see full attribution below. The claim "good case where at least these
two agree in architecture" is false.

---

## Headline finding

**Table 4 (`tab_main_results.tex` + the previous `tab_policy_endpoint.tex` layout) mixes
architectures both across rungs and within rungs.** Concretely, matching every quoted
core6 value back to the exact `summary.json` on both disks:

| rung | Table 4 core6 | disk | recomputed core6 | actual step |
|---|---:|---|---:|---:|
| Base 32L | `.7037` | **zwfy6 / H20** (`7B_base_full`) | `.70368` | (base) |
| keep8+fresh2 | `.5238` | **wzc1 / L20A** (`7B_keep8_step121000`) | `.52377` | **121k** |
| keep10+fresh2 | `.5303` | **zwfy6 / H20** (`7B_keep10_step83500`) | `.53029` | **83.5k** |
| keep12+fresh2 | `.5669` | **zwfy6 / H20** (`7B_keep12_step124000`) | `.56694` | **124k** |
| keep14+fresh2 | `.5938` | **wzc1 / L20A** (`7B_keep14_step200000`) | `.59376` | 200k |
| ShortGPT-16 | `.6215` | **zwfy6 / H20** (`7B_shortgpt16_step200000`) | `.62149` | 200k |

Match is unique to 4 decimal places for all six rows. The zwfy6 keep14 measurement is
`.59532` (rounds to `.5953`, not the paper's `.5938`); the wzc1 base measurement is
`.70402` (`.7040`, not `.7037`). So each attribution above is unambiguous.

**Cross-arch mixing is confirmed for both core6 and MMLU.** Two rungs (keep8, keep14)
come from L20A and four rungs (base, keep10, keep12, ShortGPT-16) come from H20. Per
`PAPERB_CORE6_CROSSARCH_FLOOR.md` the cross-arch floor on core6 is 0.03–0.16 pp — small
compared to adjacent rung gaps of 2.7–3.7 pp, so ordering is safe, but the paper cannot
claim "measured on the same hardware" without qualification.

## The Table's core6 does NOT come from the same disk as its MMLU

Within a single row the paper's `MMLU-L` / `MMLU-C` columns and the core6 column can
disagree on architecture. MMLU per-example predictions have shard files (`shard0of8..7of8`)
only on **zwfy6**, and the merged `summary.json` on wzc1 is a byte-identical copy (verified
via md5). File mtimes also show zwfy6 preceded wzc1 by 11 minutes, so **all six rungs'
MMLU columns come from zwfy6/H20**, whereas core6 splits by rung as tabled above.

Consequence: for keep8 the paper reads `core6 .5238` from L20A but `MMLU .2535 / .3435 /
etc.` from H20 in the same row. The 0.03–0.16 pp cross-arch floor bleeds directionally
into within-row comparisons; the paper needs to either (a) state the disk per column
explicitly, or (b) re-run whichever side is missing so every row is single-disk.

## PPL

PPL is bf16-argmax-free (sum-NLL averages the jitter, see
`PAPERB_CORE6_CROSSARCH_FLOOR.md`). Both disks measured `PPL=7.3981` for base full32 to
three decimals (`wzc1:7.398071` / `zwfy6:7.398082`). Every paper-quoted PPL from
`tab_depth_ppl.tex` matches at least one candidate at ≤5e-4 tolerance:

| rung | Table PPL | wzc1 candidate | zwfy6 candidate | comment |
|---|---:|---:|---:|---|
| full base | 7.398 | `7B_full32_base_wzc1` 7.3981 | `7B_base_full` 7.3981 | disk-agnostic |
| keep8 | 13.333 | `7B_keep8_step121000` 13.3332 | `7B_keep8_step121000_v2` 13.3329 | both, prefer wzc1 (mtime earlier) |
| keep10 | 12.816 | *(no wzc1)* | `7B_keep10_step83500` 12.8160 | **zwfy6 only** |
| keep12 | 11.443 | *(no wzc1)* | `7B_keep12_step124000` 11.4426 | **zwfy6 only** |
| keep14 200k | 10.561 | `7B_keep14_step200000` 10.5613 | `7B_keep14_step200000` 10.5612 | both |
| ShortGPT-16 | 9.780 | `7B_shortgpt16_step200000_wzc1` 9.7800 | `7B_shortgpt16_step200000{,_v2}` 9.7803/9.7800 | both |
| Frozen | 12.797 | `7B_freezefront_step200000` 12.7973 | *(no zwfy6)* | **wzc1 only** |
| Random | 11.498 | `7B_scratch16L_step200000` 11.4983 | *(no zwfy6)* | **wzc1 only** |
| full32 25k CPT | 7.670 | `7B_full32_step25000` 7.6699 | *(no zwfy6)* | **wzc1 only** |

PPL is where **the budget defect (`PAPERB_TABLE4_BUDGET_DEFECT.md`) is fully verified**:
`keep10 12.816` and `keep12 11.443` in `tab_depth_ppl.tex` come from `step83500` and
`step124000` respectively, not from 200k — the same non-200k steps identified there for
core6.

## aux5_raw column

`paperB/P0_7_aggregate_audit.csv` records the source `core6_path` per rung. For rungs
whose `core6_path` begins with `paperB/data/raw/...` (base, keep10, keep12, ShortGPT-16,
freezefront, scratch16L), that directory holds a **zwfy6/H20 copy**:
`md5(paperB/data/raw/olmo2_downstream_results/7B_base_full/summary.json) = ef2fa75c...`,
which when parsed yields `.70368` — identical to the zwfy6 value for `7B_base_full`. For
rungs whose `core6_path` uses the plain `olmo2_downstream_results/...` root (keep8,
keep14), that resolves on **wzc1/L20A**.

The `aux5_raw` values in the audit CSV (base `.6637`, keep8 `.4289`, keep10 `.4491`,
keep12 `.4608`, keep14 `.4935`, ShortGPT-16 `.5596`) therefore inherit the same disk
attribution as the core6 column of that rung.

## Recomputation of Budget defect (extension of MAIN's finding)

| rung | Table 4 Budget claim | max step on disk (both sides) | ratio | attributed to |
|---|---:|---:|---:|---|
| Base 32L | — | (base) | — | zwfy6 |
| keep8 | (labelled 121k in older tables, 200k in `paperB/TODOList.md:267`) | **121,000** | 0.605 | wzc1 |
| keep10 | (83.5k in older; 200k in TODOList) | **83,500** | **0.418** | zwfy6 |
| keep12 | (124k in older; 200k in TODOList) | **124,000** | 0.620 | zwfy6 |
| keep14 | 200k | 200,000 | 1.000 | wzc1 |
| ShortGPT-16 | 200k | 200,000 | 1.000 | zwfy6 |

MAIN's summary of the defect stands: keep10 got 41.8% of the claimed 200k budget, keep8
got 60.5%. The `tab_depth_ppl.tex` currently checked into `paperB/sections/` does list the
true steps (121k / 83.5k / 124k) — so the defect is now confined to any narrative that
still says "compute-matched" or "same budget", plus the `paperB/TODOList.md:267` note.

## Sources

- wzc1: recomputed core6 over `olmo2_downstream_results/7B_*/summary.json` (16 candidates).
- zwfy6: same, over `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results/7B_*/summary.json` (60 candidates).
- PPL: `olmo2_ppl_results/*/summary.json` on both disks.
- MMLU per-example shard mtimes: `stat -c '%y'` on both disks; md5 confirms the merged summaries are copies.
- Contamination-audit provenance for the P0.7 CSV: `paperB/P0_7_aggregate_audit.csv:1-11`.
