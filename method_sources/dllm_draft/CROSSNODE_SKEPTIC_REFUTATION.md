# Track A skeptic: the "11/11 numbers re-derive exactly" pin is REFUTED

Adjudicating this claim:

> The cross-node claim's 11 load-bearing numbers all re-derive exactly, and the grading
> axis matches on both sides (both base and base+plus from official evalplus, identical
> ground-truth hash), so the track does not collapse into the known axis effect.

**Verdict: refuted as stated (9/11, not 11/11). The *underlying phenomenon* survives and
is in fact larger, but the specific 11-number pin is wrong on 2 numbers and the "grading
axis matches" sentence is false as an inference about grader comparability.**

Everything below was recomputed by me from the JSON, not read off a table.

---

## 1. Two of the eleven numbers are artifacts of a MIXED-GRADER comparison

The two sides were graded by **different evalplus versions**:
`stack_meta.json` records `evalplus 0.3.1` (wzc1 / arm A) vs `0.1.0.dev1` (zwfy6 / arm C).
The claim compares a 0.3.1 score against a 0.1.0.dev1 score.

I re-graded the *identical* .82 solution set under the single wzc1 grader:

| quantity | claim ("re-derived exactly") | my recompute, ONE grader (0.3.1) | status |
|---|---|---|---|
| HE base, LOCAL | .7622 (125/164) | .7622 (125/164) | holds |
| HE+ LOCAL | .7073 (116/164) | .7073 (116/164) | holds |
| **HE base, .82** | **.7622 (125/164)** | **.7561 (124/164)** | **WRONG** |
| **HE+ .82** | **.6890 (113/164)** | **.6829 (112/164)** | **WRONG** |
| HE+ delta | -1.83 pt | **-2.44 pt** | number changes |
| base flips | 14 (n01=7,n10=7) | **13 (n01=6,n10=7)** | changes |
| base McNemar p | 1.0000 | 1.0000 | holds (coincidence) |
| plus flips | 13 (n01=5,n10=8) | **12 (n01=4,n10=8)** | changes |
| plus McNemar p | 0.5811 | **0.3877** | changes |
| raw_output differs | 128/164 | 128/164 | holds |

So **9/11 hold, 2 are wrong, and 4 more move once the grader is held fixed.**

Provenance: `runs/xnode/crossgrade/sol_82.jsonl` is **md5-identical**
(`3f5acbbb61048b42dd4a0183bf7b4994`) to the zwfy6 `solutions.jsonl`, so this is the same
164 programs, same ground-truth pickle `fe585eb4df8c88d844eeb463ea4d0302`, differing only
in grader version. One base task and one plus task change verdict from the grader alone.

## 2. "The grading axis matches on both sides" is the wrong check

Both sides do report base and base+plus, and the GT hash *is* identical
(`fe585eb4df8c88d844eeb463ea4d0302` — I confirmed it in the wzc1 JSON and in the zwfy6
`evalplus.out`). The delta *is* plus-vs-plus. All true.

But identical GT hash does **not** imply comparable graders — the hash keys the test
*inputs*, not the harness (timeout policy, subprocess isolation, expected-output
handling). The proof is empirical: same programs, same hash, different score. So the
sentence "the grading axis matches ... so the track does not collapse into the known axis
effect" checks the wrong axis. The base/base+plus axis was never the threat; the
**grader-version axis** was, and the claim's own evidence list does not mention it.

Note the direction: the claim understates its effect. -1.83 pt is a partly-grader number;
the clean one-grader effect is **-2.44 pt plus / -0.61 pt base**. And the claim's
"base .7622 both sides" — used to argue the base axis is unaffected — is exactly the number
the grader manufactured. Under one grader the base scores are **not** equal.

## 3. The noise floor is NOT established at 0.00, and the effect is 4 tasks

The doc's own §2.6 admits raw text is not bit-identical within a node (2-3 of 164
completions differ on repeat runs, no seed set, T=0.1 is not greedy). I confirmed:
`A vs A2` 2/164 raw diffs, `B vs B2` 3/164, `A vs B` 3/164 — with 0 pass@1 flips.

That 0-flip observation is under-powered. Cross-arch, 128 raw diffs convert to 12 plus
flips = 0.094 flips per diverging task. Applying that rate to the 8 same-arch raw diffs
gives an expected 0.75 flips, so **P(observing 0 flips across all three same-arch pairs) =
0.47** even if the same-node conversion rate equals the cross-arch one. You would need
~12 same-arch pairs for a 0-flip result to bound the floor at p<0.05. Three pairs cannot.

Scale: -2.44 pt = **4 net tasks of 164**; -0.61 pt base = **1 net task**. Exact McNemar is
p=0.3877 (plus) / p=1.0000 (base) — i.e. not significant, which the doc concedes in Limit 4
but the headline framing ("far outside" the floor) does not.

## 4. Two doc statements I could not confirm, one of them backwards

- **Limit 5 says the 48-task subset was "deliberately enriched for cross-arch divergence"
  (33/48 diverge).** It is just the index prefix HumanEval/0..47 (verified: all four arms in
  `subset48_grades.json` are exactly `range(48)`). And it diverges *less* than the full set:
  **33/48 = 68.8% vs 128/164 = 78.0%**. The enrichment claim is false; harmless direction,
  but it is an unverified assertion in a doc whose thesis is verification.
- **§2.2 "C vs orig .82: 0/164 solution-text diffs, machine only".** This is true but
  trivially so — arm C's `solutions.jsonl` is **md5-identical** to the original .82 file
  (`3f5acbbb…`). I could not verify from the wzc1 side that C is a genuinely independent
  third-host generation rather than a copy; the `metrics.jsonl` md5 does differ
  (`d3f0c986…` local vs `1a44ed68…` on zwfy6), which is consistent with a fresh run whose
  outputs happened to match bit-for-bit, but "an independent third H20 host reproduces the
  second bit-for-bit" is the single most load-bearing sentence for stability and it rests
  on files that are byte-identical to their predecessor. Worth an explicit provenance note.

## 5. What DOES survive, and it is the stronger version

Recomputed under one grader (0.3.1), all n=164, base+plus axis stated:

| arm | GPU | HE base | HE+ |
|---|---|---|---|
| r2 (original LOCAL) | L20A | .7622 (125) | .7073 (116) |
| A (LOCAL rerun) | L20A | .7622 (125) | .7073 (116) |
| A2 (LOCAL dup) | L20A | .7622 (125) | .7073 (116) |
| B (.252) | L20A | .7622 (125) | .7073 (116) |
| C (.73) | H20 | .7561 (124) | .6829 (112) |
| .82 solutions re-graded | H20 | .7561 (124) | .6829 (112) |

- `r2 vs A`: 0 flips, **0/164 solution-text diffs**, 4/164 raw diffs — the original wzc1 cell
  reproduces on rerun.
- `A vs C`: base 13 flips (6/7) p=1.0000; plus 12 flips (4/8) p=0.3877; 75/164 solution-text
  diffs; 128/164 raw diffs.
- Confounds I independently verified from `metrics.jsonl`: **0/164 rank mismatches**,
  **0/164 `input_tokens` mismatches**, `(nfe, generated_tokens) == (512, 512)` for all 164 in
  both A and C, **0 generation errors** in every arm. These are real and correctly reported
  (they live in the `process` sub-dict, not top-level).
- Note an asymmetry nobody flagged: `final_parseable=False` on 3 tasks in A/r2
  (HumanEval/64, 83, 124) but **0** in C. Parse failure is not evenly distributed across
  architectures, which is a second channel from arch to score beyond token divergence.

**Zero of the 25 flips (13 base + 12 plus) occur on a task where the two programs are
textually identical.** So the flips are downstream of genuinely different generations, not
grader flakiness on identical input — the phenomenon is real. It is the *pin* that is wrong,
not the effect.

## 6. What would settle it

1. Restate the headline with the one-grader numbers (**-2.44 pt plus / -0.61 pt base, 12/13
   flips, p=0.3877/1.0000**) and retire .7622/.6890/-1.83/14/13/0.5811 as mixed-grader.
2. Grade every arm with a pinned evalplus version recorded in `stack_meta.json`, and treat
   grader version as a first-class axis alongside base/base+plus.
3. Run ~12 same-arch repeat pairs (cheap: one node, one config, different launches) to bound
   the within-arch floor with power, instead of inferring 0.00 from three pairs.
4. Document arm C's provenance explicitly (shard logs / timestamps) so byte-identity to the
   .82 set reads as reproduction rather than as copying.
5. Run the AR control (Qwen2.5-Coder-7B greedy on both archs) that the doc's Limit 2 already
   names. Without it, "diffusion LMs are unusually exposed" is unmeasured.

## Reproduction

```
/opt/conda/envs/dllm-env/bin/python   # evalplus 0.3.1
```
Files, all under `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/`:
`runs/dream_coder_instruct_heplus_r2/solutions_eval_results.json`,
`runs/xnode/crossgrade/{sol_local,sol_82}_eval_results.json`,
`runs/xnode/{A,A2,B,B2,C}_*/solutions_eval_results.json`,
`runs/xnode/{A,C}_*/metrics.jsonl`, `runs/xnode/subset48_grades.json`.
zwfy6 side: `.73:/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104/runs/sampler_audit/he_ref_T0.1_p0.95_entropy_at0/`
(`evalplus.out` reports 0.762 / 0.689 under its own 0.1.0.dev1).
