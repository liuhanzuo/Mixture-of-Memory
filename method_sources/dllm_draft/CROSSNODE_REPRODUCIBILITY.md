# Cross-node reproducibility of a fully-specified decoding protocol

**Status: PIN PASSED. Every number in the original claim re-derived, plus one
confound the original claim did not control for (grader version), which turned
out NOT to explain the effect. The factor was then isolated: it is GPU
architecture, not the software stack, not the disk/checkout, not the machine.**

Model: `Dream-Coder-v0-Instruct-7B` (masked diffusion LM). Benchmark: HumanEval+
n=164. Grading: official `evalplus.evaluate` / `evalplus.eval.untrusted_check`
only; no hand-rolled runner anywhere in this document.

---

## 1. The pin: is the original observation real?

The tournament's skeptic-2 audit claimed the same nominal protocol gives
different scores on two nodes. **Verified, independently, from the two run
directories, before any new GPU work.**

### 1.1 Are the two configs actually identical?

The wzc1 run does *not* record a `sampler` dict (its generator predates that
field), so "the configs match" cannot be read off the artifacts alone. It was
established by reading both launchers and both generator sources.

| item | LOCAL wzc1 `dream_coder_instruct_heplus_r2` | .82 zwfy6 `he_ref_T0.1_p0.95_entropy_at0` |
|---|---|---|
| launcher | `scripts/_run_baselines_r2_wzc1.sh` | `scripts/_run_sampler_sensitivity_audit_82.sh` |
| temperature | 0.1 (CLI) | 0.1 (CLI, recorded `sampler.temperature=0.1`) |
| top_p | 0.95 (CLI) | 0.95 (recorded) |
| alg | `"entropy"` **hardcoded** at generator line 240 | `entropy` (CLI, recorded) |
| alg_temp | `0.0` **hardcoded** at line 241 | `0.0` (CLI, recorded) |
| steps / max_new_tokens | 512 / 512 | 512 / 512 (recorded) |
| seed | not settable in that generator | `null` -> no `manual_seed` call |
| chat template | ON (no `--no-chat`) | ON |
| dtype | bf16 | bf16 |
| sharding | 8x, `CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0` | identical |

The two generator scripts differ (`c20699ee…`, 308 lines vs `0c8a2ffe…`, 337
lines). **`diff -u` shows the delta is purely additive**: it adds `--alg`,
`--alg-temp`, `--seed`, and the recorded `sampler` block, with defaults
`entropy` / `0.0` / `None` — i.e. exactly the values the wzc1 script hardcodes.
With `--seed` unset, the `torch.manual_seed` branch is not entered. So the
generation semantics are the same.

Byte-level identity of everything else, checked across the two disks:

| artifact | wzc1 md5 | zwfy6 md5 | same |
|---|---|---|---|
| `model-00001-of-00004.safetensors` | `78183b600e1d95c44e39071f834a5b3e` | `78183b600e1d95c44e39071f834a5b3e` | yes |
| `model-00002…` | `27b3d4e9600308956c0816cd5dc696ba` | same | yes |
| `model-00003…` | `2fc6e0c556da7709559ed8231bccad7f` | same | yes |
| `model-00004…` | `597ef90e5d5d425e193c9e8837ecf3aa` | same | yes |
| `config.json` | `06261896df2e289ae9427ebca88336da` | same | yes |
| `model.safetensors.index.json` | `c6d54e8f12cf9b10b112031f932ddf40` | same | yes |
| `generation_config.json` | `6a3a2023b1f72f5cf0044a3ae64d9156` | same | yes |
| `modeling_dream.py` | `893420e43895630d8c15b8fdd148d66c` | same | yes |
| `generation_utils.py` | `fe7dc873c3bb8e487d8451d356fe0de4` | same | yes |
| tokenizer_config / vocab / merges / added_tokens / special_tokens_map / tokenization_dream | all match | all match | yes |

The two nodes read prompts from **differently-named files** (`humaneval_plus.jsonl`,
md5 `7b9145…`, 89 KB vs `HumanEvalPlus-v0.1.10.jsonl`, md5 `fe585e…`, 7.7 MB) —
a real trap, since a prompt difference would trivially explain everything. It is
not one: same 164 task_ids, **0/164 prompt mismatches, 0/164 entry_point
mismatches**, concatenated-prompt md5 `1d8181b01008d4777f5e0a65bd66f488` on both.
The large file merely also carries the plus test inputs.

### 1.2 Grading axis — the check that could have killed the track

Both sides report **base and base+plus from the official grader**, and both
loaded the identical ground truth (`hash = fe585eb4df8c88d844eeb463ea4d0302` in
both `eval_results.json`). **The -1.8 pt is a plus-vs-plus comparison, not a
base-vs-plus axis artifact.** The track survives this check.

### 1.3 Re-derived numbers

Recomputed from `solutions_eval_results.json` / `eval_results.json`:

| quantity | audit claimed | **I re-derived** | verdict |
|---|---|---|---|
| HE base, LOCAL | .7622 | **.7622 (125/164)** | exact |
| HE base, .82 | .7622 | **.7622 (125/164)** | exact |
| HE+ , LOCAL | .7073 | **.7073 (116/164)** | exact |
| HE+ , .82 | .6890 | **.6890 (113/164)** | exact |
| HE+ delta | -1.8 pt | **-1.83 pt** | exact |
| base flips | 14 (n01=7, n10=7) | **14 (n01=7, n10=7)** | exact |
| base McNemar p | 1.0 | **1.0000** (exact binomial) | exact |
| plus flips | 13 (n01=5, n10=8) | **13 (n01=5, n10=8)** | exact |
| plus McNemar p | 0.5811 | **0.5811** (exact binomial) | exact |
| raw_output differs | 128/164 | **128/164** (36/164 identical) | exact |

**PIN PASSED — 11/11 numbers reproduce exactly.**

### 1.4 A confound the original claim did not control

The two nodes graded with **different evalplus versions**: wzc1 `0.3.1`, zwfy6
`0.1.0.dev1`. The audit compared scores produced by two different graders.
Cross-grading both solution sets under the single wzc1 grader (0.3.1):

| solution set | as originally reported | re-graded under evalplus 0.3.1 |
|---|---|---|
| LOCAL | .7622 / .7073 | **.7622 / .7073** (unchanged) |
| .82 | .7622 / .6890 | **.7561 / .6829** |

So the grader **was** doing work: the .82 base score of .7622 was partly a
grader artifact, and under a common grader the base scores are **not** equal
(.7622 vs .7561, 13 flips). The headline effect survives and in fact grows:
**HE+ -2.44 pt, HE base -0.61 pt** under one grader. Every number below uses
evalplus 0.3.1 on wzc1 for all arms, so the grader is never a variable.

---

## 2. Isolation: which factor?

New runs, identical pinned config, all graded centrally with evalplus 0.3.1.

| arm | node | GPU (cc) | disk / checkout | python / torch / transformers |
|---|---|---|---|---|
| A | LOCAL | L20A (10.0) | wzc1 | 3.11.15 / 2.11.0+cu128 / 4.51.3 |
| A2 | LOCAL | L20A (10.0) | wzc1 | same as A (determinism dup) |
| B | .252 | L20A (10.0) | wzc1 | 3.11.15 / 2.11.0+cu128 / 4.51.3 |
| B2 | .252 | L20A (10.0) | wzc1 | same as B (determinism dup) |
| C | .73 | H20 (9.0) | zwfy6 | 3.11.6 / 2.5.1+cu124 / 4.46.2 |
| E | .73 | H20 (9.0) | zwfy6 | 3.10 / **2.6.0+cu124** / **4.57.6** (48-task) |
| F | .73 | H20 (9.0) | zwfy6 | 2.5.1+cu124 / 4.46.2 (48-task, dup of C) |

Note `.252` is **L20A, cc 10.0 — the same architecture as LOCAL**, not B200 as
the cluster notes say. That makes A-vs-B a clean *pure machine* control:
different physical host, same arch, same stack, same disk.

### 2.1 Headline table (all n=164, evalplus 0.3.1, HE+ = base+plus axis)

| arm | HE base | HE+ | vs A base | vs A plus |
|---|---|---|---|---|
| A (LOCAL, L20A) | **.7622** (125) | **.7073** (116) | — | — |
| A2 (LOCAL dup) | **.7622** (125) | **.7073** (116) | 0.00 | 0.00 |
| B (.252, L20A) | **.7622** (125) | **.7073** (116) | 0.00 | 0.00 |
| B2 (.252 dup) | **.7622** (125) | **.7073** (116) | 0.00 | 0.00 |
| C (.73, H20) | **.7561** (124) | **.6829** (112) | **-0.61 pt** | **-2.44 pt** |
| original .82 H20 | **.7561** (124) | **.6829** (112) | -0.61 pt | -2.44 pt |

### 2.2 Paired statistics

| comparison | what varies | base flips | plus flips | plus exact p | solution text differs | raw_output differs |
|---|---|---|---|---|---|---|
| A vs A2 | nothing (same node) | **0** | **0** | 1.0 | **0/164** | 2/164 |
| B vs B2 | nothing (same node) | **0** | **0** | 1.0 | **0/164** | 3/164 |
| A vs B | machine only | **0** | **0** | 1.0 | 1/164 | 3/164 |
| A vs C | GPU arch + stack + disk | 13 (7/6) | 12 (8/4) | 0.3877 | **75/164** | **128/164** |
| B vs C | GPU arch + stack + disk | 13 (7/6) | 12 (8/4) | 0.3877 | 75/164 | 128/164 |
| C vs orig .82 | machine only (both H20) | **0** | **0** | 1.0 | **0/164** | — |

Two things to read off this. **(a) Within-architecture reproducibility is
perfect**: 0 flips, 0 solution-text differences, identical pass@1 to 4 decimals,
across two *different physical machines* and across repeat runs. So the
non-reproducibility is not machine-to-machine noise. **(b) C reproduces the
original .82 H20 cell exactly** (0 flips, 0 text differences) — an independent
third H20 host reproduces the second H20 host bit-for-bit, which is what makes
the cross-architecture gap a stable property rather than a one-off.

### 2.3 The decisive leg: GPU arch vs software

L20A (cc 10.0) cannot run torch 2.5.1, but H20 can run a *newer* stack. So the
software axis was varied on **fixed H20 hardware** (48-task subset, chosen
because 33/48 of those tasks diverge across architectures):

| comparison | what varies | raw_output differs | base flips | plus flips |
|---|---|---|---|---|
| F vs C (H20, both torch 2.5.1) | nothing | 2/48 | **0/48** | **0/48** |
| F vs E (H20, torch 2.5.1 vs **2.6.0**, tf 4.46.2 vs **4.57.6**) | software | 2/48 | **0/48** | **0/48** |
| E vs C (H20, torch 2.6.0 vs 2.5.1) | software | **0/48** | 0/48 | 0/48 |
| F vs A (H20 vs L20A) | **GPU arch** | **33/48** | **3/48** | **2/48** |

48-task pass counts: E 42/38, F 42/38, C 42/38, **A 43/40**.

**A two-minor-version torch jump plus an eleven-minor-version transformers jump
on the same GPU changes nothing (0 flips, and E vs C is bit-identical on all
48). Changing GPU architecture with the software held as close as the hardware
allows changes 33/48 generations and 3 base / 2 plus outcomes.** The variance is
**hardware-architecture-borne, not library-borne**.

### 2.4 Confounds explicitly ruled out

- **Grading axis**: both base and base+plus reported; effect present on both.
- **Grader version**: all arms graded once with evalplus 0.3.1; same GT hash.
- **Prompts**: 0/164 prompt or entry_point mismatches; identical concat md5.
- **Weights**: all four safetensors shards md5-identical across disks.
- **Remote code**: `modeling_dream.py` / `generation_utils.py` md5-identical.
- **Sharding**: **0/164 tasks assigned to a different rank** in A vs C.
- **Tokenisation**: **0/164 `input_tokens` mismatches**.
- **Step budget**: `nfe` and `generated_tokens` are `(512, 512)` for all 164 in
  both A and C — no truncation asymmetry.
- **Errors**: 0 generation errors in every arm.
- **Merge coverage**: every arm merged with `merge_evalplus_shards.py --expected`.
- **Disk/checkout**: A vs B are on the same disk and agree; C vs .82 are on the
  same disk and agree; the split is orthogonal to the disk.

### 2.5 Where the divergence starts

First divergent token index (model tokenizer), over diverging tasks:

| comparison | min | median | max | diverge at token 0 |
|---|---|---|---|---|
| A vs A2 (same node) | 433 | 473 | 473 | 0 |
| B vs B2 (same node) | 465 | 473 | 511 | 0 |
| A vs B (machine only) | 376 | 433 | 510 | 0 |
| **A vs C (cross-arch)** | **0** | **97** | 505 | **4** |

Within an architecture, the rare disagreements appear only **very late** (median
token 473 of 512) and, inspected directly, land in trailing explanatory prose
after the code block — e.g. HumanEval/62 `"the computed derivative"` vs
`"the calculated derivative"` at char 1880 of ~2100, HumanEval/88 `"has only one
element"` vs `"contains only one element"` at char 1541. That is why they cost
0 flips: the extracted program is unchanged (solution text differs 0/164 for
A vs A2 and B vs B2). Cross-architecture, divergence begins at **median token
97**, and in 4 tasks at **token 0** — inside the program, which is why it moves
pass@1.

### 2.6 Honest caveat on the within-node control

The brief anticipated an exactly-0.0 within-node floor (as the HE+ sampler audit
measured at T=0.1). At the **pass@1 level that holds**: 0 flips, identical to 4
decimals, 0/164 solution-text differences, in all three same-arch comparisons.
At the **raw-text level it does not**: 2-3 of 164 completions differ even on the
same node with the same config. T=0.1 is not greedy, and no seed was set, so a
non-deterministic reduction order suffices to flip a near-tie in trailing prose.
Reported rather than smoothed over. Consequence: the honest within-node floor
for this protocol is **0.00 pt on pass@1 but not bit-identical in text**, and
the cross-architecture effect (-2.44 pt, 12 flips, 128/164 text differences) is
far outside it.

---

## 3. What this means

A decoding protocol specified to the level journals and model cards actually
use — checkpoint hash, temperature, top_p, unmasking algorithm, alg_temp, step
budget, chat template, dtype, benchmark version, official grader — **is not
sufficient to reproduce pass@1 on HumanEval+ across GPU architectures.** The
gap is **2.44 pt HE+ / 0.61 pt base**, from an identical protocol on identical
weights, against a within-architecture floor of **0.00 pt** measured across two
machines and repeat runs.

For scale: 2.44 pt is comparable to or larger than several published method
gains on this benchmark (DiffuCoder +4.4, Order-Token Search +6.8 are bigger;
many reported deltas are smaller), on a benchmark routinely reported to three
decimals. It is smaller than the 26.8-59.8 pt sampler-protocol spread already
established at fixed NFE — this is a distinct, additional, and *irreducible*
term: the sampler spread can be closed by specifying hyperparameters, this one
cannot be closed by specifying anything at all in the protocol.

Mechanism, as far as it is established: at T=0.1 with confidence-based unmasking,
the committed token and the *order* of unmasking both depend on floating-point
comparisons between near-tied candidates. Different architectures produce
different bf16 reduction results, the confidence ranking permutes, and the
divergence compounds over 512 diffusion steps. This is consistent with the
already-verified finding that unmasking *order* alone moves HE+ by 3.0-4.9 pt
against a provably zero token-identity floor — the same lever, pulled here by
hardware rather than by top_p. Diffusion LMs should be expected to be more
exposed than autoregressive decoders, because order is a live degree of freedom
at every one of the 512 steps; **that comparison is not yet measured and is the
obvious next experiment.**

## 4. Limits — what would kill or shrink this

1. **One checkpoint, one benchmark, one architecture pair.** H20 (sm_90) vs
   L20A (sm_100), Dream-Coder-Instruct, HumanEval+. Whether the magnitude holds
   on MBPP+, other dLLMs, or A100/H100 is untested. A near-zero result on
   another pair would demote this to a quirk of these two.
2. **No AR control.** The claim "diffusion LMs are unusually exposed" is
   mechanistically motivated but *not* measured. Running Qwen2.5-Coder-7B
   greedy on both architectures is cheap and is the single highest-value
   follow-up. If an AR model also moved ~2 pt, the dLLM-specific framing dies
   and this becomes a generic bf16-reproducibility observation.
3. **Software axis tested only within H20's supported range.** torch 2.5.1 ->
   2.6.0 and transformers 4.46.2 -> 4.57.6 gave 0 flips, but torch 2.11 could
   not be tested on H20, so "software contributes nothing" is established over
   the reachable range, not universally.
4. **n=164, and the flips are not individually significant** (plus exact
   McNemar p=0.3877). The *reproducible* facts are the pass@1 gap and the
   128/164 text divergence, both of which replicate exactly on a third host;
   the claim is about reproducibility, not about one arm being better.
5. **Subset legs are n=48**, deliberately enriched for cross-arch divergence,
   so their pass counts are not comparable to the n=164 headline.
6. Anyone re-running the H20 arm must use `.venv_dream`; `/opt/conda/envs/
   torch-base` (transformers 5.5.4) breaks Dream's remote code.

## 5. Artifacts

- Driver: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/scripts/_run_crossnode_repro.sh` (md5 `a90629b5667660434fad3ee915347421`, byte-identical copy on zwfy6)
- Analysis: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/scripts/analyze_crossnode_repro.py`
- Per-pair report incl. per-task first-divergent-token: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/xnode/analysis_full.json`
- Subset grades: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/xnode/subset48_grades.json`
- Cross-grade of the original two solution sets under one grader: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/xnode/crossgrade/`
- Per-arm runs incl. `stack_meta.json`: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/xnode/{A,A2,B,B2,C,E,F}_*/`

### Process note

The first launch of arm A was accidentally started twice against one shard path.
`merge_evalplus_shards.py --expected 164` refused the merge (248 rows), so
**nothing was graded from it**; the directory was deleted and the arm re-run
once. Recorded because it is the third time in this project that the coverage
assertion caught a duplicated driver, and it is the reason no contaminated
number reached a table.
