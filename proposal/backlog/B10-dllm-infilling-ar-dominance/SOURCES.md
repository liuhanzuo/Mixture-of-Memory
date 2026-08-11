# B10 SOURCES — evidence paths, corrected

**Compiled 2026-08-11. GPU used: none.** Every path below was `stat`-ed and
hashed on the disk named. Sizes in bytes; hashes are the first 16 hex chars of
sha256 (full hashes re-derivable with `sha256sum`).

> ⚠️ **TWO DISKS.** `wzc1` = LOCAL + `.21`. `zwfy6` = `.73` / `.82` / `.104`.
> They are *different physical disks with different checkouts*. A file is
> "missing" only after searching **both**. Most of B10's evidence is
> **zwfy6-only**, which is why a previous search declared it absent.

---

## 0. ★ SLATE's paths are STALE — read this first

`proposal/archive/revival-slate/SLATE.md` §`#3` cites its assets as
`dllm_draft/...` relative paths and says the six arms are "on zwfy6". Both are
too vague to act on, and one is actively misleading. Corrections:

| SLATE says | reality |
|---|---|
| `dllm_draft/` (implied inside the Mixture-of-Memory repo) | **does not exist there.** The real tree is one level **above** the repo: `pighzliu_code/dllm_draft/` |
| "six fully scored arms at n=1033 on zwfy6" (no path) | `zwfy6 pighzliu_code/dllm_draft_104/outputs/infilling_single_line/<arm>/` — a **different checkout** (`dllm_draft_104`, not `dllm_draft`) |
| `scripts/generate_evalplus_dreamon.py:133-134` | exists on **both** disks ✅ (this one is fine) |
| the infilling harness "lives at zwfy6 `dllm_draft_104/scripts/`" | ✅ correct, and it is **zwfy6-only** — absent from wzc1 |
| "DreamOn weights are absent from both mounted model dirs" | ❌ **WRONG.** Present at `zwfy6 dllm_draft/models/DreamOn-v0-7B/`. `dllm_draft_104/models` is a **symlink** → `dllm_draft/models`, which is probably why a `find` on the `_104` tree missed them. |

**Where MAIN looked and why it failed.** MAIN searched
`dllm_draft/runs/spanlen/` and correctly found no `SingleLine_*` arm dir. That
directory holds a *different, later* experiment (span-length stratification on
**RandomSpan** n=1640 / **MultiLine_sub** n=420). The headline SingleLine arms are
not under `runs/` at all — they are under `outputs/`, on the other checkout.
**Nothing is lost; nothing needs re-running.**

---

## 1. ★ PRIMARY EVIDENCE — the six scored arms (zwfy6 only)

Root: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104/outputs/infilling_single_line/`

| file | bytes | sha256[:16] |
|---|---|---|
| `qwen_fim/score.json` | 185707 | `bf607ea56c1596a3` |
| `qwen_fim/metrics.jsonl` | 562287 | `88410e66c1efd336` |
| `qwen_fim/solutions.jsonl` | 1058971 | `e1836814c2f8968b` |
| `qwen_prefix/score.json` | 185761 | `251baf7cf8257e2c` |
| `qwen_prefix/metrics.jsonl` | 1167461 | `d99fffbad52891cb` |
| `qwen_prefix/solutions.jsonl` | 1056005 | `9028aeb7b48e80e8` |
| `dream_fim/score.json` | 185655 | `1993ff707355a854` |
| `dream_fim/metrics.jsonl` | 570880 | `f664341ec7c3636b` |
| `dream_fim/solutions.jsonl` | 1058563 | `80579717d910344c` |
| `dream_prefix/score.json` | 185631 | `013470a2861d30f1` |
| `dream_prefix/metrics.jsonl` | 577196 | `8447933683dcd3c4` |
| `dream_prefix/solutions.jsonl` | 1051887 | `d4847df3298cbe81` |
| `dreamon_fim/score.json` | 185895 | `c0b2035a2f2172a2` |
| `dreamon_fim/metrics.jsonl` | 645248 | `ff831abefb73dd26` |
| `dreamon_fim/solutions.jsonl` | 1049065 | `7fc3919674caae0b` |
| `dreamon_oracle/score.json` | 185723 | `5ff3b6ff229528af` |
| `dreamon_oracle/metrics.jsonl` | 694850 | `7097543495eae6b7` |
| `dreamon_oracle/solutions.jsonl` | 1053939 | `590e8b33006e88d3` |

Each `score.json` carries `per_task[]` with 1033 rows
(`{task_id, pass, n_tests, n_pass, why, exact_match}`) — this is the raw record
pass@1 was recomputed from. Each `metrics.jsonl` has 1033 lines with
`cost.{tokens_fed, attended_context_sum, forward_passes}` and
`info.{initial_masks, middle_tokens, recovered, ...}`. Each arm dir also has a
`shards/` subdir with the 8 per-rank files (retained; the merge asserted
`n_shards == 8` and no duplicate ids, so the merged files are complete).

**Roll-up (do NOT score from this — use it only as a cross-check):**

| file | bytes | sha256[:16] |
|---|---|---|
| `dllm_draft_104/results/infilling/single_line_summary.json` | 6168 | `576cf5d3f046e57e` |

This is the *only* file in either tree containing all four headline values
together, and it is the one SLATE never names. It is **committed to git**
(`dllm_draft_104@eedd0075`), unlike the raw outputs.

---

## 2. Benchmark data (zwfy6 only)

Root: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft/data/humaneval_infilling/`

| file | lines | bytes | sha256[:16] |
|---|---|---|---|
| `HumanEval-SingleLineInfilling.jsonl` | **1033** | 1647941 | `6fffc71ec2f16743` |
| `HumanEval-RandomSpanInfilling.jsonl` | 1640 | 2203584 | `4f15be2d4a479b50` |
| `HumanEval-MultiLineInfilling.jsonl` | 5815 | 10487245 | `5b49f312f0a78142` |

This is what settles the "is n=1033 really SingleLine?" question: **yes**, 1033 is
SingleLine and 1640 is RandomSpan. Upstream: `loubnabnl/humaneval_infilling`
(HuggingFace), the official benchmark of Bavarian et al. 2022.

---

## 3. Harness (zwfy6 only — ABSENT from wzc1)

Root: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104/scripts/`

| file | bytes | sha256[:16] | role |
|---|---|---|---|
| `generate_infilling.py` | 16222 | `8408b06af45a4ed7` | all 6 arms; arm semantics in its docstring |
| `score_infilling.py` | 8176 | `9f1313d6b4d38589` | grading via `evalplus.eval.untrusted_check` + self-test |
| `_run_infilling_5arm_8gpu.sh` | 3532 | `d3f64de9e5400815` | 8-GPU launcher; pins split, ckpts, `--which plus` |

> ⚠️ `dllm_draft/DLLM_SALVAGE_ROADMAP_20260808.md` (wzc1) states
> "`scripts/generate_infilling.py` is **missing**" and lists "recover or rewrite
> the missing infilling generator" as a blocker. That is **true of wzc1 and false
> of zwfy6** — it was committed to the `dllm_draft_104` checkout only. The
> roadmap's blocker is a cross-disk artefact, not a real loss. **Do not rewrite
> the generator.**

Key facts read out of the harness (cited in `PROPOSAL.md` / `NUMBER_AUDIT.md`):
- `_run_infilling_5arm_8gpu.sh` maps `dream_fim|dream_prefix` →
  `models/Dream-Coder-v0-**Instruct**-7B` (**not** Base — the lineage confound);
- it scores with `--which plus` (the axis that carries the 0.8025 ceiling);
- `--temperature 0.0` (greedy), `--max-new-tokens 64`, `--initial-masks 4`;
- `generate_infilling.py:327-335` sets `dreamon_oracle`'s `initial_masks` to the
  true middle token count (the oracle handout), and gives `dream_fim` /
  `dream_prefix` the oracle `span_tokens`.

**Our DreamOn call site (on BOTH disks):**

| disk | path | bytes | sha256[:16] |
|---|---|---|---|
| wzc1 | `dllm_draft/scripts/generate_evalplus_dreamon.py` | 7085 | `579d1e0a9ec77558` |
| zwfy6 | `dllm_draft_104/scripts/generate_evalplus_dreamon.py` | 7085 | (same size) |

Lines 133-134 pass `mask_expansion=True, delete_eos_token=True` — the two inert
kwargs. Note this is the **from-scratch HumanEval+ harness**, *not* the infilling
harness; `generate_infilling.py` does **not** pass them. SLATE cites this file as
support for claim (b) without noting it belongs to a different experiment.

---

## 4. ★ DreamOn model files — PRESENT (contra SLATE)

Root: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft/models/DreamOn-v0-7B/`
(reachable as `dllm_draft_104/models/DreamOn-v0-7B/` — `models` is a symlink)

| file | bytes | sha256[:16] | why it matters |
|---|---|---|---|
| `generation_utils.py` | 24251 | `9ef97ad61d77cfbe` | **the file SLATE said it could not re-read.** Contains `DreamGenerationConfig` and the expand/delete sampler |
| `config.json` | 880 | `c02a98999d7d2249` | ships `expand_token_id: 151667`, `mask_token_id: 151666` → expansion live by default |
| `generation_config.json` | 431 | — | `expand_budget: null`, `delete_token_id: 151643` |
| `added_tokens.json` | 680 | — | `<\|expand\|>` = 151667, `<\|mask\|>` = 151666 |
| `README.md` | 4670 | — | the "Parameters" section that **never mentions** the two kwargs |
| `model-0000{1..4}-of-00004.safetensors` | ~15.2 GB total | — | weights (not needed for the static verification) |

Sibling checkpoints in the same `models/` dir, all with `config.json` read for the
lineage check: `Qwen2.5-Coder-7B`, `Dream-Coder-v0-Base-7B`,
`Dream-Coder-v0-Instruct-7B`.

---

## 5. Gold ceiling (wzc1 only)

| disk | path | bytes | sha256[:16] |
|---|---|---|---|
| wzc1 | `dllm_draft/runs/spanlen/gold_ceiling_SingleLine.json` | 269101 | `007baa0924f9e750` |

`overall.gold_ceiling_base = 0.989351403678606`,
`overall.gold_ceiling_plus = 0.8025169409486931`, `n_rows = 1033`, plus
`per_row[]` with `gold_plus_pass` per task — the field used to build the
829-item gold-feasible subset. **This file is wzc1-only**; to use it next to the
arms (which are zwfy6-only) it must be `scp -O`'d across. Verified transfer:
same sha256 `007baa0924f9e750...` on both ends.

This is the single most consequential file in B10: it is what shows the raw plus
numbers cannot be compared to the literature.

---

## 6. Internal narrative / retraction record (wzc1)

Root: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/`

| file | bytes | sha256[:16] | relevance |
|---|---|---|---|
| `DLLM_RESULTS_20260807.md` | 52634 | `fe3a26be7111fd43` | "Retraction 5" = the infilling experiment; also the "Length-elastic 被 DreamOn 吃掉" death record |
| `KSPAN_INFILLING_RESULTS.md` | 21170 | `a735c95f9b70fa63` | k-span ladder; "diffusion's home turf **WITHDRAWN**"; 26-28 pp decontamination drop |
| `KSPAN_NONORACLE_ARM.md` | ~25.5 KB | — | Retraction 7 amendment; non-oracle arm |
| `SPANLEN_STRATIFIED_AUDIT.md` | 16676 | `703e9a35737b3805` | the gold-ceiling table (§1); 23 buggy parent tasks |
| `DLLM_SALVAGE_ROADMAP_20260808.md` | 25803 | `cfe998282b0e2d42` | forward-looking decision record; "HumanEval k-span 'diffusion home turf' → **STOP**" |
| `TASK_SURFACE_LIT_GAPS.md` | 1625 | `14b6d9b568f92900` | **records DreamOn's Table-1 numbers incl. Qwen 92.6** — the internal note that refutes SLATE's "no AR control" premise |

> ⚠️ **Scope discipline.** `KSPAN_*` and `DLLM_SALVAGE_ROADMAP` withdraw
> "diffusion's home turf" for the **k-span multi-region** surface (n=415/408,
> nested k-ladder). B10 is the **SingleLine n=1033** surface — a *different
> experiment*. The withdrawal does not automatically transfer, but it is
> **directly relevant** because §4.5 of `KSPAN_INFILLING_RESULTS.md` shows
> HumanEval infilling scores are 26-28 pp surface memorisation, which applies to
> SingleLine too. Cited as a threat, not as a refutation.

Related runs (NOT B10's surface — do not mix):
`dllm_draft/runs/spanlen/` (RandomSpan n=1640, MultiLine_sub n=420, 8 arm dirs on
wzc1; zwfy6 `dllm_draft_104` additionally has `RandomSpan_dreamon_fim`),
`dllm_draft/runs/kspan_*` (16 dirs, n=415/408/236/165).

**Git provenance:** `dllm_draft_104@eedd0075` ("HumanEval-Infilling FIM eval, 6
arms … 8-GPU sharded", LiuHanzuo, 2026-08-07 13:35 +0800) added
`results/infilling/single_line_summary.json`, `scripts/generate_infilling.py`,
`scripts/score_infilling.py`, `scripts/_run_infilling_5arm_8gpu.sh` (846 insertions).
The raw `outputs/` are gitignored — hence hashes above.

---

## 7. External primary sources

| source | how verified | key content |
|---|---|---|
| **DreamOn** — "Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas" | **OpenReview `venueid = ICLR.cc/2026/Conference`, `venue = "ICLR 2026 Poster"`**, forum `EQTPmqukiU`. PDF footer reads "Published as a conference paper at ICLR 2026" | **Table 1: Qwen2.5-Coder-7B single-line 92.6, DreamCoder-7B+DreamOn 92.1** ⇒ the matched AR control ALREADY EXISTS. §4.2 baselines: Deepseek-Coder-6.7B, Seed-Coder-8B, Qwen2.5-Coder-7B, LLaDA-8B, Dream-7B, DiffuCoder-7B. §4.1: `Lmax=128` expansion cap, T=0.2, top_p=0.9, mask len 64 |
| DreamOn arXiv:2602.01326v1 | fetched via proxy; `pdftotext -layout`, 11 pages | same as above; authors Zirui Wu, Lin Zheng, Zhihui Xie, Jiacheng Ye, Jiahui Gao, Shansan Gong, Yansong Feng, Zhenguo Li, Wei Bi, Guorui Zhou, Lingpeng Kong |
| **A3** — "Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation", arXiv:2601.13228 | **OpenReview `venueid = ICLR.cc/2026/Conference`, "ICLR 2026 Poster"** | **the closest thesis-level collision**: AR reformulated for any-order generation "outperforms diffusion-based models"; includes story infilling |
| "Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly", arXiv:2602.00476 | arXiv abs page; **no OpenReview venue found ⇒ treat as preprint** | training-free CAL length search; +47.7 % pass@1 over fixed-length in code infilling — competes with DreamOn's premise, not with B10 |
| "From Interface to Inference…", arXiv:2607.26504 | arXiv abs page; no venue found | insertion-based / latent masked diffusion for any-order code inference |
| "Diffusion Language Models Are Natively Length-Aware", arXiv:2603.06123 | arXiv abs page; no venue found | zero-shot context cropping, **reports FLOPs** — the cost-accounting niche is being occupied |
| "Improving Variable-Length Generation in Diffusion LMs via Length Regularization", arXiv:2602.07546 | arXiv listing | variable-length diffusion |
| "Any-Order Flexible Length Masked Diffusion", arXiv:2509.01025v2 | arXiv listing | FlexMDM lineage |
| Bavarian et al., "Efficient Training of LM to Fill in the Middle", arXiv:2207.14255 | cited by DreamOn as the HumanEval-Infilling source | the FIM benchmark + method B10's `qwen_fim` arm relies on |
| Allal et al., SantaCoder / SantaCoder-FIM | cited in DreamOn §4.2 | second infilling surface, not on our disks |

**Venue-verification rule applied** (per project memory): OpenReview
`venueid` + camera-ready for ICLR/NeurIPS/ICML; aclanthology + DBLP for the ACL
family. S2/DBLP lag and misreport 2026 papers as preprints, so they were not
used as the authority. Two of the collisions are confirmed **peer-reviewed ICLR
2026 posters**, not preprints.

Local literature note that independently corroborates the novelty finding:
`dllm_draft/TASK_SURFACE_LIT_GAPS.md` already recorded
"DreamOn … **ALREADY did** HumanEval-Infilling single/multi-line … vs
Qwen2.5-Coder-7B … Qwen2.5-Coder-7B 92.6", and listed what is **NOT** covered:
"any compute/NFE accounting, RandomSpanInfilling, multi-span, non-Python."

---

## 8. Recompute scripts

Five short CPU-only scripts were written to `/tmp` on `.73` for this audit
(`audit_infill.py`, `cost_infill.py`, `mcnemar_infill.py`, `ceiling_norm.py`,
`dreamon_live_probe.py`). They are **transient and not committed** — each is
<60 lines and fully specified by the file+key pairs in `NUMBER_AUDIT.md`
§13. No B10 conclusion depends on a script that cannot be reconstructed from
this document.

Access recipe (read-only):

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
sshpass -f configs/password_h20_853573.txt ssh -o StrictHostKeyChecking=no \
  -o PreferredAuthentications=password root@28.85.35.73     # NOTE: omit -p
# python: /opt/conda/envs/torch-base/bin/python   (the .venv/bin/python on H20 is broken)
# for anything importing DreamOn's remote code: /apdcephfs_zwfy6/.../dllm_draft_104/.venv_dream/bin/python
```
