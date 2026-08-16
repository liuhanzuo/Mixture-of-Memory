# DISK_DECISION_20260816.md — what actually needs a human decision

> Written because I had been reporting "disk cleanup — awaiting your decision" for several
> heartbeats **without ever stating what the decision was**. That is the defect this file fixes.
> Everything below is measured today, 2026-08-16, and supersedes the pressure numbers in
> `DISK_CLEANUP_AUDIT_20260812.md` §8 (which decided KEEP *because* the disk was not under
> pressure — that premise has changed on wzc1 and not on zwfy6).

## Method note (reusable, saves hours)

On this `dop-fuse` mount a directory's own `st_size` **is the recursive byte total**. Verified
against an exhaustive `find` on two directories, byte-for-byte:

| dir | `stat -c %s` | `find` sum |
|---|---|---|
| `models/` | 17,411,334,693 (16.22 GiB) | 16.2 GiB |
| `data/` | 983,724,538,293 (916.16 GiB) | 916.2 GiB |

An independent `du -sh` of `pighzliu_code` returned **18T** vs `stat`'s **17.05 TiB** (+5.6%,
= block allocation vs apparent bytes). So `stat` is exact-apparent and `du` is exact-allocated.

**Consequence: never run `du` on this cluster again for sizing.** A `du --max-depth=1` over the
120 T disk was killed at 50 min having produced one line; the `stat` loop sized all 35 top-level
directories in under a second.

## 1. Measured pressure, both disks

| Disk | Total | Used | Free | Use% | 08-12 audit read | Δ |
|---|---|---|---|---|---|---|
| **wzc1** (LOCAL + `.212`) | 120 T | 110 T | **10 T** | **92%** | 109 T / 11 T / 91% | **−1 T free, +1pp** |
| **zwfy6** (`.73/.82/.104`) | 689 T | 667 T | **22 T** | **97%** | 650 T / 39 T / 95% | **−17 T free, +2pp** |

zwfy6 sampled 3× spaced 6 s: identical each time (667 T / 22 T / 97%), so this is not fuse lag.

## 2. Who owns the space — this REFUTES my own earlier claim

I had reported that ~96% of the shared space belongs to colleagues and our checkpoints are
"~1% of it". **That is wrong on wzc1.** Measured top-level:

| Disk | Our footprint | Our share of USED | Largest colleague |
|---|---|---|---|
| wzc1 | **17.05 TiB** | **15.5%** | `eachwang` 17.28 T (we are #2 of 35) |
| zwfy6 | **22.98 TiB** | **3.4%** | `hunyuan` **322.20 T** (48% of the disk alone) |

The "~1%" figure was wrong because I had only looked *inside the repo* (5.88 TiB). **11.2 TiB of
our wzc1 footprint sits outside it**, in sibling directories I was not measuring:

```
5.88 TiB  pighzliu_code/Mixture-of-Memory/     <- the repo (all I had been measuring)
4.69 TiB  pighzliu_code/out_llama/             <- 99 SparseForge sweep dirs, Jan-Apr
3.67 TiB  pighzliu_code/data/                  <- dolmino-raw 1.50 T, c4_* 1.82 T
1.32 TiB  pighzliu_code/outputs/               <- paper_v2 0.74 T, cast_repro 0.44 T
1.07 TiB  pighzliu_code/models/                <- Hy3 0.54 T, A13B 0.15 T
0.53 TiB  pighzliu_code/out_llama_tokenmatched_{slorb,noslorb}/  <- #246, recent
```

**The two disks therefore need opposite conclusions:**

- **wzc1** — our deletions matter. Freeing 5 T moves 92% → 87%.
- **zwfy6** — our deletions are nearly irrelevant. Deleting *everything we own* (22.98 T) moves
  97% → 94%. `hunyuan/` alone is 14× our footprint. **The 97% is not ours to fix**, and no
  deletion we make buys meaningful headroom there. If zwfy6 headroom is the real problem, the
  action is a quota conversation with the `hunyuan` owner, not cleanup by us.

## 3. Inside the repo (wzc1, 5.88 TiB)

`outputs/` is 4.86 TiB in **155 `.pt` files** (5,737 GiB measured exactly). Top runs:

| GiB | files | run | status |
|---|---|---|---|
| 726.1 | 16 | `olmo2_probe2_7B_keep14fresh2_seed1234` | trajectory arm, paper-cited |
| 545.0 | 15 | `olmo2_probe2_7B_keep10fresh2` | **LIVE — training right now on LOCAL** |
| 423.5 | 10 | `olmo2_probe2_7B_shortgpt16` | paperC damaged rung, paper-cited |
| 407.8 | 5 | `olmo2_probe2_7B_full32_dolmino` | control arm, paper-cited |
| 265.8 | 1 | `hyv3_probe2_keep36_fresh2` | deliberate sole survivor (see §5.3) |
| 228.1 | 10 | `olmo2_probe2_7B_keep14fresh2_distill` | **LIVE — training right now on `.212`** |

## 4. The free win is already taken, and it held

The 08-12 audit hardlinked 14 `final.pt` ↔ `step{max}.pt` pairs (structurally identical: the
trainer writes the terminal step twice). **Re-verified all 14 today by inode comparison — every
one is still hardlinked, both filenames still resolve, and no duplication has regrown.** So
there is no repeat of that win available; ~719 G on wzc1 stays collapsed.

## 5. THE ACTUAL DECISION — four items, all previously deferred

All four were filed `NEEDS-DECISION` on 08-12, then MAIN decided KEEP on all four **explicitly
because "the disk is not under pressure"**. wzc1 has since lost 1 T of the 11 T that justified
that. The four, largest first:

### 5.1 `pighzliu_code/out_llama/` — 4.69 TiB, wzc1 — **the whole decision, essentially**

99 dirs of SparseForge pruning sweeps, dated 2026-01-22 → 2026-04-27, none modified since
**2026-05-11** (3 months cold). By subject model: Llama2-7b 2,628 G / 39 dirs; Qwen3-8b 423 G / 7;
Llama2-13b 398 G / 4; opt-2.7b 308 G / 3; deepseek-moe 207 G / 2; gpt2 144 G / 13.

**Cited by name: only 3 of 99** (`20260413_201320` 80.9 G by 35 committed files including
`paperC`-adjacent B12 evidence and `SparseForge_Data` docs; `20260401_124938` 80.9 G by 6;
`20260404_000815` 88.4 G by 1 transcript only). Nothing is being written to it — the live
SparseForge arms write the *sibling* `out_llama_tokenmatched_*`.

**Why I will not delete it on my own authority:** task **#245 (reproduce ALPS+SLoRB as the
SparseForge matched control) is still `[pending]` on exactly the Llama-2-7B family** that holds
2.6 T of this. And the "only 3/99 cited" argument is the *same* loose inference the 08-12 audit
correctly rejected for `mem_space` — these dirs are addressed through `--out_dir`/globs, not
literal paths, so absence of a name-mention is not absence of use. Applying the strict standard
here means I cannot certify the other 96 as dead.

**This is the one question only the owner can answer:** which `out_llama/*` runs does the
SparseForge NeurIPS table + rebuttal still need? Answer that and the complement — plausibly
**3–4 TiB, i.e. wzc1 92% → ~89%** — becomes deletable. Note each dir's `args.json` +
`best_lm_eval.json` are kilobytes and can be retained as the permanent record even if the
80–170 G of weights go.

### 5.2 `distill_cache/512` — 946 G, **zwfy6** — decision is now moot

Regenerable-in-principle dolmino teacher cache, 74,032 `.npz`. Kept on 08-12 because
regeneration is a real GPU/CPU bill. **Per §2 this is now moot: 946 G is 4% of our zwfy6
footprint and 0.14% of that disk. Deleting it buys nothing measurable.** Recommend: drop from
the decision list entirely.

### 5.3 `outputs/hyv3_probe2_keep36_fresh2/step200.pt` — 265.8 GiB, wzc1 — **I recommend KEEP**

Sole surviving artifact of a 30B Hy-MT2 prune-heal frontier; a *prior* cleanup deliberately
deleted its siblings and kept only this one. Loss curves survive in logs, the weights do not
exist anywhere else, and the direction is dormant rather than refuted. Overriding a deliberate
prior survivor decision needs a scientific reason; "it is large" is not one. 266 G is also only
2.6% of wzc1 free space — it cannot solve the problem it would be sacrificed for.

### 5.4 `MemLong/` — 31 G wzc1 / 300 G zwfy6 — **too small to matter**

Never audited in depth. At 31 G on the disk that is actually tight, not worth a decision.

## 6. What I am NOT proposing to touch, and why

- `outputs/olmo2_probe2_7B_keep{8,10,12}fresh2` resume checkpoints — the three deprioritised
  Paper B arms. **They do not expire**; deleting them converts a pause into a 3–9 GPU-day loss.
- All five live training output dirs (LOCAL keep10, `.212` keep14-distill, `.73` keep12,
  `.82` keep8, `.104` paperC-k8f2).
- `outputs/olmo2_p05_arm{A_contig16,B_final14_fresh2}` — wzc1-only, B-P0.4's only gate-eval
  source; cross-disk transfer measured at 12 MB/s single-stream, so these are effectively
  irreplaceable in place.
- Anything cited by `paper*/`, `proposal/*/SOURCES.md`, or `RUN_REGISTRY.md`.

## 7. Bottom line

**One question needs you: §5.1, which `out_llama/*` SparseForge runs are still live for the
rebuttal.** That is worth 3–4 TiB on the disk that is actually tight.

The other three are mine to settle and I have: 5.2 moot, 5.3 keep, 5.4 too small.

**And the honest framing on zwfy6: its 97% is not our doing and not ours to fix.** I should not
have been implying a cleanup decision from us would relieve it.

---

## 8. Adversarial pass (17-agent workflow, 2026-08-16 21:0x) — 16 candidates, 1 survived

A parallel survey + adversarial-verification workflow re-derived the ownership census
independently and then tried to REFUTE each deletion candidate. Result: **1 SAFE, 8
conditional, 3 blocked** out of 16 considered. The refutations are the valuable part.

### It confirmed the §2 correction with a better method

It validated the recursive-`st_size` trick three ways rather than two, including **at
checkpoint scale**: `stat outputs/olmo2_probe2_7B_keep10fresh2` = 585144334902 B vs a
420-second `du -sb` = 585144334902 B, exact. And it caught something I did not: the sum of
wzc1 top-level `rbytes` falls **0.32% short of `df` Used** (380330344637 B), which matches to
within 4 MB the rstat propagation lag measured inside our own actively-written tree. Two
independent quantities agreeing to 4 MB out of 380 GB means `rbytes` is in `df`'s unit with no
replication multiplier — so our wzc1 total is properly a **range, 17.113–17.459 TiB**, reading
~2% low while writes are in flight. Conclusions are identical at either bound.

It also found **zwfy6's `hunyuan/` is 81 sub-users**, of which `rukizheng` alone holds
**79.58 TiB — 3.4× our entire zwfy6 footprint**. Our slice inside it is 0.0002 TiB.

### The only SAFE item

`cache/hmt_pg19_full` on zwfy6, **82.24 GiB verified present**. A dead HMT/pg19 tokenisation
cache; the only things that reference it are two launchers, one of which is already in
`legacy/`. Regenerating means re-downloading 6.58 GiB of parquet and re-tokenising ~11.5 GB.
**But per §2 this is on zwfy6, so it buys 0.01pp of Use%.** Correct to delete on tidiness
grounds, irrelevant to pressure.

### Refutations worth recording (these were proposed as "safe" and are not)

- **`data/dolmino-mix-1124-llama2/` (579.15 GiB)** — pinned by task **#245**, whose GATE0
  **passed on 2026-08-15** at 1.12 GPU-h (`aligned=224/224 misaligned=0`, rc=0, twice). I
  checked the apparent contradiction between `ALPS_SLORB_GATE0_FAILED.md` and
  `..._VERDICT.md`: the FAILED file carries a retraction banner, and both of its failure
  signals are the rank-local-counter and gated-postfix artifacts already in memory. **GATE0
  really passed, so this directory is a live dependency of a pending task.**
- **`data/dolmino_now15b.npy` (57.8 GiB)** — `paperB/REPRO_SHA256.txt:1` pins
  `4c1a2c89…41b` for exactly this path, and the same line appears in
  `sha256_repro_artifacts.txt`, `submission_source_v7`, and six review-history snapshots.
  Deleting it breaks a **submitted** reproducibility manifest. Verified the pin myself.
- **`models/` on either disk** — `models/Qwen3-8b-local` is the repo-relative default for
  every harness invoked without an override. *(Correction to the agent's own number: it
  reported "192 places across 126 files"; I measured **73 occurrences in 55 `.py`/`.sh`
  files**. Still load-bearing; the count was inflated ~2.3×.)* `bge-large-en-v1.5` is the
  only copy of the weights behind paperA's dense-retrieval leg.
- **`.venv` on zwfy6** — the interesting one: deleting it would **not** crash anything, it
  would **silently change benchmark inputs**, because `scripts/eval_ruler_cwe.py:48-95` wraps
  its `wonderwords` import in try/except and falls back to `nltk.corpus.words` and then to a
  hand-rolled synthesiser. A silent input change is worse than a crash.

### One claim I had to overturn

The verifier for `data/slimpajama_chunks_4096.npy.BROKEN_llama2tok_uint16overflow`
(11.95 GiB) wrote that deleting the wzc1 copy is fine because "a byte-identical twin survives
on zwfy6 — but ONLY on that condition, and the survey lane did not check it." **I checked:
there is no twin.** `stat` on the zwfy6 path returns `No such file or directory`. So this is
the only copy of a file whose own name records that it is corrupt. It is still probably
deletable — it is a known-broken uint16-overflow artifact — but the stated *reason* was wrong,
and it should be retired deliberately (with the overflow bug written down somewhere durable)
rather than as a no-cost duplicate.

**Net after the adversarial pass: still zero bytes deleted, and the one question in §7 is
unchanged.** What changed is that four "obvious wins" are now documented as live
dependencies, which is the outcome that saves a future agent from a bad deletion.

