# The within-disk "floor" was harness drift, not runtime jitter — and it changes what cross-arch measures

**Date**: 2026-08-08 ~08:2x CST. **Verified by**: MAIN, from n=3 and n=2 same-arch same-disk repeats
committed to disk. **GPU cost**: ~40 min (variance-controls dispatch, agent `afe2a215`).
**This partially reverses `PAPERB_WITHIN_DISK_FLOOR.md` and revises `PAPERB_DAMAGE_SCALING_AUDIT.md`.**

## The measurement

Within-disk same-arch same-harness re-runs of pre-SFT eval batteries. Every pair below is on zwfy6
H20, same checkpoint, same code path:

| pair | Σ\|net flips\| across core6 |
|---|---:|
| keep14 v1 (Aug 8 01:31) vs v2 (Aug 8 08:02) | **0** (byte-identical) |
| keep8 v2 vs v3 | **0** (byte-identical) |
| shortgpt16 v2 vs v3 | **0** (byte-identical) |
| shortgpt16 v1 vs v3 | 20 |
| keep10 v1 vs v2 (from earlier tonight) | 18 |

Same-arch **same-harness-version** re-runs are **byte-identical** on all six core6 tasks. The 15–20
flip "floor" I documented in `PAPERB_WITHIN_DISK_FLOOR.md` was between **v1 (older harness) and v2+
(new harness added tonight)** — a **code-version boundary**, not runtime jitter.

## What partially reverses

- The claim in `PAPERB_WITHIN_DISK_FLOOR.md` that "core6 is reproducible only to ±0.2 pp under any
  re-run" and that this ~15-20 flip floor was intrinsic bf16/harness noise: **wrong**. Same-harness
  re-runs are bit-identical.
- The framing that cross-arch effects sit *inside* the noise floor: needs revision. Some of the
  cross-arch flip counts I measured earlier were **between old-harness zwfy6 and new-harness wzc1**,
  which conflated architecture with harness version.

## What survives, and what the actual cross-arch signal is

Recomputing cross-architecture with **matched (new) harness on both sides**:

| rung | wzc1 L20A (new) vs zwfy6 (new v2/v3) | per-task flip breakdown |
|---|---:|---|
| ShortGPT-16 | **7** | `hs +1, arc_c 0, arc_e +2, piqa −1, obqa −1, wino −2` |
| keep10 | **23** | `hs 0, arc_c +4, arc_e −6, piqa +2, obqa +2, wino +9` |
| keep8 | **29** | `hs +2, arc_c −4, arc_e −7, piqa +3, obqa −1, wino +12` |

So the honest cross-architecture picture at matched harness is:

1. **The floor now is 0**, not 18. Same-harness re-runs on the same GPU are deterministic to bit
   level.
2. **Cross-architecture flip counts range 7 to 29 across three rungs**, comfortably above the true
   floor.
3. The **winogrande task dominates**: it supplies +12, +9, and −2 of the three totals — the largest
   single component in all three rungs. If there is a genuine architecture effect on core6, it is
   plausibly a winogrande-specific one (winogrande option-scoring may involve near-tie likelihoods
   that bf16 reduction order can flip), and other tasks contribute little.
4. **The rungs are ordered ShortGPT-16 (7) < keep10 (23) < keep8 (29).** Suggestive of a damage
   effect, but n=3 with substantial per-task variability. Not enough to reinstate the damage-scaling
   claim; possibly enough to say "worth checking with per-task variance decomposition."

## What actually happened between v1 and v2+

**⚠️ MAIN correction (~08:4x CST): the "driver drift" mechanism named below and in the agent's
`PAPERB_WITHIN_DISK_FLOOR_V3.md` is NOT supported. The measurement (0 flips same-harness) is solid;
the cause is misattributed. See the CAUSE INVESTIGATION section at the end of this file.**

I don't yet know which specific harness change introduced the boundary. Candidates: the
`assert_8shards` guard (v1 predates it; if a v1 silently kept a stale shard from a previous run
mixed with fresh ones, that alone would produce ~20-item drift); a change in how per-item
scoring is invoked; a change in tokenization/BOS handling. **I have not diagnosed which.** Until
diagnosed, treat v1 batteries as suspect for exact reproduction and use v2/v3 wherever possible.

The keep12 partial-merge finding (`PAPERB_TABLE4_KEEP12_PARTIAL_MERGE.md`) is likely *one instance*
of the general v1-fragility mechanism, not a separate defect. Paper's Table 4 was quoting v1
numbers for every rung it named; whether any others carry the same class of silent defect below
the `n_scored`-check threshold I've applied is now the open question.

## Consequences for the paper writeup

1. **The damage-scaling story remains dead** (it died on other grounds: within-pruned monotonicity
   fails at n=5, and the pre-SFT-PPL axis mixes depth with unequal healing budgets — see
   `PAPERB_SFT_FIT_CONFOUNDED.md`). This revision does not resurrect it.
2. **Cross-arch effect is real but modest**: ~7–29 flips out of ~13–17k scored items, dominated by
   winogrande on the three pruned rungs measured. Do not lead with this.
3. **The `_v2`/`_v3` batteries are the paper-quality numbers.** Whenever a v1 and a v2 exist, use
   v2. Table 4 rewrite should re-source every rung to a v2 measurement, not a v1.
4. **Paper B needs an explicit protocol note**: "held-out evaluations use harness version X, seeded
   shard assignment, `assert_8shards` on merge." This is a paragraph in the appendix, not a table
   change.

## Provenance

- keep14 v2: `zwfy6:olmo2_downstream_results/7B_keep14_step200000_v2` (mtime Aug 8 08:02, size differs
  from v1 by 3 bytes only in JSON formatting; per-task counts byte-identical to v1)
- keep8 v3, shortgpt16 v3: `zwfy6:*_v3/` (produced by agent `afe2a215` this heartbeat)
- Prior contradicted claim: `PAPERB_WITHIN_DISK_FLOOR.md` (commit `af6d869`) — the section titled
  "the within-disk flip count (18) exceeds the cross-architecture flip count (17)" was true as
  written for keep10 v1 vs v2, but the 18-flip figure now reads as harness-version drift, not
  runtime jitter, and the "exceeds" comparison should be pulled from paper writeup.
- Related: `PAPERB_TABLE4_KEEP12_PARTIAL_MERGE.md` (a known v1-side bug consistent with this).

## Retraction accounting for tonight

Five framings now retracted or revised:
1. dLLM sampler-audit generalization to MBPP+ (killed by G1)
2. Cross-arch damage-scaling of flip count (killed by within-disk floor claim — **which itself is now revised as harness drift**, though the damage-scaling claim is not resurrected)
3. Linear ΔPPL vs pre-PPL fit (killed by keep8)
4. Monotone-saturating fallback (killed by keep10)
5. Any within-pruned SFT ordering (killed by n=5 spread within 1.6 pp)

Now plus this partial revision: **the intra-disk floor was itself an artifact of harness drift,
not a physical noise floor.** The cross-arch numbers with matched harness are still small and don't
reinstate any earlier claim; they just need to be re-cited carefully.

---

# CAUSE INVESTIGATION (MAIN, ~08:4x CST): "driver drift" is ruled out — and the real problem is worse

The variance-controls agent (`afe2a215`) closed the table at **4/4 rungs, 0 flips** — adding
`full32_base` on wzc1/L20A to the three zwfy6 rungs above. Its measurement is solid and its
mtime boundary is a perfect predictor:

| eval | mtime | side of Aug 2 20:12 | flips vs re-run |
|---|---|---|---:|
| shortgpt16 v1 | Aug 2 05:02 | before | 20 |
| keep12 v1 | Aug 2 10:34 | before | 437 (+ the 6/8 partial merge) |
| keep10 v1 | Aug 2 11:04 | before | 18 |
| keep14 v1 | Aug 8 01:31 | **after** | **0** |
| keep8 v2, sg16 v2, full32 v1 | Aug 8 02–08 | **after** | **0** |

**But the mechanism the agent named — "the old driver predates `--save_per_example`" — cannot
produce a flip, and I verified this five ways:**

1. **The driver diff is purely additive.** `8947078 → a4da5e8` on
   `scripts/eval_olmo2_probe2_downstream.py` is 135 insertions / 12 deletions, and every line is
   either the new `mmlu_pro` task, the `--save_per_example` side-record, or an
   `ex["gold"]`→local-`gold` refactor. **No change to the scoring, batching, or sharding math.**
2. `--batch_size 8 --num_shards 8` in both the v1-era launcher (`_run_olmo2_eval_shortgpt.sh:80`)
   and the v3 launcher (`_run_olmo2_within_disk_floor_v3.sh:136`).
3. `--max_len` defaults to 1024 (`:598`) and **neither** generation's launcher overrides it.
4. The driver never reads `LOCAL_RANK`/`RANK` — the v3 launcher sets them, the v1 launcher doesn't,
   and it makes no difference. `device = torch.device("cuda")` unconditionally (`:638`).
5. **No stale-shard mixing**: all 8 `shardNof8.json` mtimes inside each v1 dir cluster within ~3 s.

## The thing I found instead, which is the real finding

```
zwfy6:  git status --short scripts/eval_olmo2_probe2_downstream.py
        ?? scripts/eval_olmo2_probe2_downstream.py        <-- UNTRACKED
        git cat-file -p 2d98c5a:scripts/eval_olmo2_probe2_downstream.py
        fatal: path ... exists on disk, but not in '2d98c5a'
```

**The downstream eval driver is untracked in zwfy6's git checkout.** It was `scp`'d in at
**Aug 2 20:12** with no commit (alongside `_run_olmo2_mmlu_content.sh` at 20:12 and
`eval_olmo2_mmlu_content.py` at 20:22). The file is now byte-identical to wzc1's
(`md5 2bf40c0d379d37a51e412347cb012cd0`) and wzc1's copy **is** tracked and clean.

So: **the exact code that produced every v1 number — including four of Table 4's six rungs — was
never version-controlled and is now unrecoverable.** That is a provenance failure independent of
whether the numbers are right. It is also why the boundary lines up with an mtime instead of a
commit, and why I can't diff my way to the cause.

## Live candidates, none yet confirmed

- **(A) Dataset version drift.** The driver calls `load_dataset` **without pinning a revision** for
  hellaswag / ai2_arc / winogrande / openbookqa (only piqa pins
  `revision="refs/convert/parquet"`, `:161`). If the HF cache was rebuilt between Aug 2 and Aug 4,
  item content or order could change. Suggestive: zwfy6's `~/.cache/huggingface/datasets/`
  has `.lock` files dated **Aug 4 23:26** for winogrande and ai2_arc — the two tasks that dominate
  the cross-arch flip counts. **This would be the serious outcome**: it would mean core6 numbers
  are not comparable across time at all, not merely across harness versions.
- **(B) The pre-scp driver genuinely differed** in some scoring-relevant way. Unfalsifiable from
  git; only reachable by behavioral reconstruction.
- **(C) Base-model/tokenizer file drift** under `models/OLMo-2-1124-7B`.

Dispatched agent `a8604004` to bisect these (CPU per-task flip localization + two-disk dataset
fingerprint comparison first; a `force_redownload` differential run on .82/.104 only if that
doesn't settle it). The discriminator: if flips concentrate in winogrande/arc_easy → (A);
if spread evenly across all six tasks → (B).

## What this changes about the writeup

- **The 0-flip determinism result stands.** Same code + same data + same arch ⇒ bit-identical. That
  is worth stating as a protocol fact.
- **"Harness/driver drift" must NOT be asserted as the cause** until `a8604004` reports. Both this
  file's earlier sections and the agent's `PAPERB_WITHIN_DISK_FLOOR_V3.md` overreach on that point;
  the honest statement today is "a boundary at Aug 2 20:12 whose cause is not yet identified."
- **Paper B needs a provenance fix regardless of the outcome**: pin dataset revisions in the driver,
  and never run a paper number from an untracked script. The untracked-driver fact alone justifies
  re-sourcing Table 4 to post-boundary measurements.
- **Convenient**: wzc1 already has a **complete post-boundary ladder**, all six tasks at full
  `n_scored`, measured Aug 8 with the tracked driver — `full32_base .70402`, `shortgpt16 .62194`,
  `keep8@121000 .52377`, `keep10@83500 .53217`, `keep12@111500 .56941`. So a clean single-arch,
  single-driver, full-shard Table 4 is already largely on disk (keep14 is the gap: wzc1's copy is
  from Jul 28, pre-boundary).

