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
