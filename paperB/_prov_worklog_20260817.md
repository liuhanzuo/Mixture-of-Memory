# Table 4 provenance hunt — worklog (2026-08-17)

Target: files behind keep8/keep10/keep12 v2 rows in `paperB/sections/tab_downstream.tex`
(esp. keep12 ARC-Easy = .694 @ n=2376, replacing 6/8-shard .689 @ n=1782).

## Step 0: commit 6d15049
- `git show --stat 6d15049`: touches ONLY paperB/*.md + paperB/sections/*.tex. No outputs/ or evals/ path.
- `tab_downstream.tex` appears as `new file mode` -> file is NOT normally tracked; only landed in this commit.
- Commit message names the artifacts as "the clean _v2 batteries (torch 2.13, eval batch 8, 8/8 shards)".
  Keyword to search for: `_v2`, `_know`, `battery`.

## Step 1: why the original grep missed — results are at REPO ROOT, not outputs/
`ls -d olmo2_* results evals` at repo root (wzc1):
  olmo2_closedbook_results/ olmo2_downstream_results/ olmo2_mc_letter_content_results/
  olmo2_mmlu_content_results/ olmo2_ppl_results/ results/     (`evals/` does NOT exist)
=> The user's `grep ... outputs evals` never looked at these trees. Lead came from
   paperB/TODOList.md:292 which cites `.82:olmo2_downstream_results/...`.

## Step 2: the `_v2` dirs are zwfy6-only (confirmed both disks)
wzc1 `ls -d olmo2_*results/*_v2*` -> only full32_base_wzc1_v2 + keep14_step200000_wzc1_v2.
  NO keep8/keep10/keep12 _v2 on wzc1.
zwfy6 (via .73), `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`:
  olmo2_downstream_results/7B_keep8_step121000_v2{,_know}
  olmo2_downstream_results/7B_keep10_step83500_v2{,_know}
  olmo2_downstream_results/7B_keep12_step124000_v2{,_know}
  (+ matching _v2 in olmo2_ppl_results/, olmo2_closedbook_results/, olmo2_mmlu_content_results/)
Step numbers match Table 4's step column: 121k / 83.5k / 124k.

## Step 3: keep12 ARC-Easy .694 REPRODUCES
zwfy6:olmo2_downstream_results/7B_keep12_step124000_v2/summary.json
  arc_easy.acc_norm = 0.6936026936026936  -> .694  OK
  arc_easy.n_scored = 2376, n = 2376, n_nan = 0, n_trunc = 0
  top-level n_shards = 8, add_bos = false
  meta.ckpt = outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt, meta.ckpt_step = 124000
  All 8 shard{0..7}of8.json present on disk (mtime Aug 8 03:07), merged files Aug 8 13:52.
WHY THE ORIGINAL GREP FOR '0.6936' ON zwfy6 RETURNED ZERO: it was scoped to
`outputs evals`; the string lives in olmo2_downstream_results/ (repo-root sibling).
The value IS on disk. The absence was a search-scope artifact, not a missing file.

## Step 4: SELF-CORRECTION — my first verifier's 30 "MISMATCH" were my own bug
`fmt()` produced "0.694" and compared to printed ".694" -> string inequality.
All 30 were leading-zero artifacts. The n_scored/n_shards/add_bos/n_nan asserts in the
SAME run all passed, which is the self-contradiction that flagged it. Re-running with
a leading-zero-stripping comparison. DO NOT read the first run as failures.

## Step 5: REAL discrepancy surfaced — MMLU column
Table 4 prints keep8/keep10/keep12 MMLU = .2550 / .2720 / .2728.
`olmo2_mmlu_content_results/7B_keep{8,10,12}_..._v2/summary.json` letter_acc =
  keep8  0.25430850306224184 -> .2543   (printed .2550)  DIFFERS
  keep10 0.2706879361914257  -> .2707   (printed .2720)  DIFFERS
  keep12 0.27239709443099275 -> .2724   (printed .2728)  DIFFERS
So the MMLU column is NOT from the _v2 dual-interface dirs. TODOList:278 says the MMLU
column comes from `olmo2_mmlu_content_results/` and is "byte-identical on both disks";
commit msg says the same. -> next: check the NON-_v2 dual-interface dirs (both disks).

## Step 6: MMLU column source LOCATED (wzc1, NON-_v2 dual-interface dirs)
wzc1 olmo2_mmlu_content_results/<dir>/summary.json -> letter_acc:
  7B_keep8_step121000        0.2550206523 -> .2550  == printed  OK
  7B_keep10_step83500        0.2720410198 -> .2720  == printed  OK
  7B_keep12_step124000       0.2727531691 -> .2728  == printed  OK
all with n=n_valid=14042, n_nan=0, n_shards=8, add_bos=False, matching ckpt_step.
Decoy dirs that must NOT be used (they also exist and look plausible):
  7B_keep8_step121000_wzc1   .2546   |  7B_keep10_step83500_wzc1 .2717
  7B_keep12_step111500_wzc1  .2713 (also WRONG STEP: 111500, not 124000)
  and the zwfy6 `*_v2` mmlu dirs: .2543/.2707/.2724
=> Table 4's MMLU column is the NON-suffixed, NON-_v2 trio. Metric = `letter_acc`.
So Table 4's shallow rows draw from TWO different directory families:
  ten non-MMLU columns <- zwfy6 olmo2_downstream_results/*_v2{,_know}
  MMLU column          <- olmo2_mmlu_content_results/7B_keep{8,10,12}_step{121000,83500,124000}

## Step 7: ALL 33 CELLS REPRODUCE (corrected verifier, PY_RC=0)
Verifier /tmp/prov_verify.py, run on .73 with /opt/conda/envs/torch-base/bin/python.
33/33 cells match the printed Table 4 values at the printed precision, with per-cell
asserts n_scored==n==expected, n_nan==0, n_shards==8, add_bos==False, ckpt_step==row step.
Expected counts used: HS 10042 / ARC-C 1172 / ARC-E 2376 / PIQA 1838 / WinoG 1267 /
OBQA 500 / MMLU 14042 / LAMBADA 5153 / BoolQ 3270 / CSQA 1221 / SIQA 1954 -- all matched.
MMLU byte-identity across disks confirmed by sha256 (identical wzc1 vs zwfy6):
  keep8  97cf8f5f5c68c0323344f82140fdfcca2b812b120b0d01df0643c31e0df07a7f
  keep10 fabf8705dcc4a47993dcc94b20ccc7dcb238c56ff6dc09c866aa109c3872f4ee
  keep12 2e3ed46807d7629b3389830ce30b0ead92df5565c04a3ed1b8e35487ff889ebd
`find . -maxdepth 6 -type d -name '*keep{8,10,12}_step*_v2*'` on wzc1 -> EMPTY.
The ten non-MMLU columns are zwfy6-ONLY. This is a genuine reproducibility caveat.

## Step 8: SECOND self-correction, then the STRONGEST check
My shard probe read a nonexistent key (`tasks[t]["n_scored"]` in the per-shard JSON;
real schema is `{shard_index,num_shards,add_bos,meta,tasks:{T:{n,n_correct_acc,
n_correct_accnorm,n_nan,n_trunc}}}`) -> 33 bogus "SHARDJSON-SUM" failures while every
real check in the same run passed. Same self-contradiction signature as Step 4.

Rewrote as /tmp/prov_recompute.py: recompute each cell as
sum(n_correct_{acc|accnorm}) / sum(n) over shard0..7of8.json, IGNORING summary.json.
Result PY_RC=0: all 30 non-MMLU cells reproduce with 8/8 shard_index coverage,
n == expected, n_nan == 0, per-shard num_shards==8 / add_bos==False / ckpt_step==row step.
keep12 arc_easy recomputed 1648/2376 = 0.693603 -> .694 from shards alone.
Note (pre-existing, benign, not a defect in the cell): boolq carries n_trunc=2 in all
three arms; scored anyway, n_nan=0.

## Step 9: MMLU column recomputed from the 14,042 per-item records (PY_RC=0, wzc1)
The mmlu_content dirs keep only the merged per_example_mmlu.jsonl (shard*of8.json were
cleaned up), so recomputed as count(letter.correct)/count(lines):
  keep8   3581/14042 = 0.2550206523 -> .2550  == printed
  keep10  3820/14042 = 0.2720410198 -> .2720  == printed
  keep12  3830/14042 = 0.2727531691 -> .2728  == printed
n_items == unique item_id == 14042, zero nan, for all three.
=> Table 4 MMLU metric is the `letter` interface of the dual snapshot (NOT content_raw
   .3222/.3232/.3407 and NOT content_norm .3427/.3448/.3624), which matters because those
   also live in the same file and are much larger.

## Step 10: caption's "one set of 14,042 per-item records, single-source across all nine rows" HOLDS
Recomputed all NINE Table 4 MMLU cells from per_example_mmlu.jsonl and hashed each row's
(item_id, subject, gold, n_opt) stream: ONE distinct signature 17cf44e95acde711 for all nine.
9/9 cells reproduce (.6054 .5877 .2550 .2720 .2728 .3184 .4742 .2624 .2470), zero nan. PY_RC=0.
Dirs (wzc1 olmo2_mmlu_content_results/): 7B_base, 7B_full32_step25000,
7B_keep8_step121000, 7B_keep10_step83500, 7B_keep12_step124000, 7B_keep14_step200000,
7B_shortgpt16_step200000, 7B_freezefront_step200000, 7B_scratch16L_step200000.

## VERDICT SO FAR: provenance ESTABLISHED for all 33 re-measured cells.
Remaining work: record it where paperB records evidence paths.
