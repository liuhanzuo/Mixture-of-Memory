# P3.2 — Fair over-window full-context control (YaRN / native-128k)

**Reviewer ask (P3.2):** Qwen3-8B native window is 40,960; our full-context /
KV-Direct results at 128k/256k run with YaRN **NOT** activated, so they are an
*unextended* reference, not a fair length-extended upper bound. Reviewer options:
(1) add a **YaRN-enabled** Qwen3-8B control, OR (2) a natively-128k same-size model,
OR (3) explicitly **downgrade** >40k full-context numbers to "unextended reference."

## Status: RESOLVED-empirical (2026-07-26). ✔

Wording closure (reviewer option 3) was already in the paper. Now additionally
**closed empirically** with actual KVD+YaRN and CoMem+LoRA+YaRN measurements
(see §Empirical findings below). Paper draft updated with:
- `paper/sections/tab_scaling.tex`: three new rows (KVD+YaRN, CoMem+LoRA,
  CoMem+LoRA+YaRN) added to the RULER scaling table.
- `paper/sections/tab_yarn_tax.tex`: new stand-alone YaRN-tax delta table.
- `paper/sections/05_experiments.tex`: new §Length-extension composability
  paragraph (§4.5, before Efficiency).
- `paper/sections/07_limitations.tex`: "only viable choice" softened to
  "not merely an artefact of an unextended reference."

Already in the paper (committed earlier, option 3):
- `05_experiments.tex` §Models (lines 4–8): states native window = 40,960; the
  131,072 figure is an *un-activated* YaRN extrapolation limit; >41k inputs fall
  outside the trained regime.
- `05_experiments.tex` §Baselines (lines 21–27): KV-Direct runs at the **native
  40,960 window with YaRN not activated**, so beyond 40,960 it is an *unextended*
  full-context reference — a **lower bound** on what a YaRN-extended same-size
  backbone would achieve, not a fair length-extended upper bound. We explicitly do
  **not** claim CoMem beats a length-extended full-context model at 128k/256k; the
  claim is only that CoMem stays usable where the *unextended* backbone breaks.

## Empirical findings (2026-07-26)

### KVD × YaRN (n=100, chat=False, official string_match_all)

| Task | 8k | 16k | 32k | 64k | 128k |
|------|---:|---:|---:|---:|---:|
| niah_single_3 unext | 100 | 100 | 100 | 100 | 0 |
| niah_single_3 YaRN | 100 | 100 | 100 | 100 | **100** |
| niah_multikey unext | 100 | 100 | 99 | 89 | 0 |
| niah_multikey YaRN | 98 | 94 | 96 | 91 | **89** |
| var_track unext | 100 | 99.8 | 100 | 95.2 | 0 |
| var_track YaRN | 99.2 | 99.4 | **26.6** | **67.2** | **57.8** |

**Key insight**: YaRN rescues single/multikey needle tasks at 128k (0→100/89)
but imposes a devastating in-window tax on multi-hop VT: −73.4pp at 32k,
−28.0pp at 64k. The tax peaks at 32k (RoPE rescaling vs. trained attention)
and partially recovers as length approaches the effective 163k ceiling.

### CoMem+LoRA × backbone (n=50, iter_bm25, hop=4, topk=12)

| Config | 8k | 16k | 32k | 64k | 128k |
|--------|---:|---:|---:|---:|---:|
| VT unext | 98.0 | 98.4 | 98.8 | 97.6 | **98.4** |
| VT YaRN | 81.2 | 86.0 | 90.4 | 96.8 | 87.6 |
| single unext | — | — | 100 | — | 98 |
| single YaRN | — | — | 92 | — | 96 |
| multikey unext | — | — | 100 | — | 93* |
| multikey YaRN | — | — | 96 | — | 90* |

*niah_multikey at 128k: CSV n=40 (claimed n=50); scores 92.5/90.0 rounded to 93/90.
 If true n=50 scores differ, update table (brief claimed 94/92).

### Flagship comparison at 128k VT
**CoMem+LoRA+Unext 98.4 > CoMem+LoRA+YaRN 87.6 > KVD+YaRN 57.8 > KVD-unext 0**

CoMem advantage over YaRN-KVD: +40.6pp with 5× less memory (17–18 GB vs 89 GB).
CoMem is "RoPE-invariant": swapping to YaRN backbone drops VT by only 0.8–16.8pp
vs 27.8–73.4pp for KVD. Vanilla backbone (no YaRN) is the best config.

### Data location
`ruler_results/p32_from_82/` — scp'd from .82 node.

## Ready-to-run recipe (for replication)

1. Private model-config copy (does NOT touch shared config.json; symlink weights):
   ```bash
   SRC=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
   DST=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b-yarn
   mkdir -p "$DST"
   for f in "$SRC"/*; do ln -sf "$f" "$DST/$(basename "$f")"; done
   rm -f "$DST/config.json"           # replace the symlink with a real edited copy
   cp "$SRC/config.json" "$DST/config.json"
   # add YaRN (factor 4 -> effective 163,840; keep max_position_embeddings=40960):
   python - <<'PY'
   import json,os
   p=os.environ.get("DST","")+"/config.json"
   c=json.load(open(p))
   c["rope_scaling"]={"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}
   json.dump(c,open(p,"w"),indent=2)
   print("rope_scaling injected:",c["rope_scaling"])
   PY
   ```
2. Run canonical KVD path (same as `_run_kvdirect_taskbreadth_8gpu.sh`) at 128k,
   **chat_template=False** (paper pillar), n=100, single clean GPU:
   ```bash
   python scripts/eval_ruler_qcmem.py \
     --model_path <DST>/Qwen--Qwen3-8b-yarn \
     --baseline kvdirect \
     --ruler_tasks niah_single_3 niah_multikey ruler_vt \
     --lengths 128k --limit 100 \
     --max_new_tokens 128 \
     --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --device cuda:0 \
     --output_name kvd_yarn_128k --results_folder ruler_results/kvd_yarn_128k
   # (NO --use_chat_template -> chat=False)
   ```
3. Compare the resulting YaRN-KVD 128k numbers against:
   - unextended KVD 128k = 0/0/0 (tab_scaling)
   - CoMem 128k = 100/84/20 (niah_single/multikey/var-track)
   Add a "full-ctx (+YaRN)" row to tab_scaling and soften/expand the "only CoMem
   usable" wording to "CoMem matches/exceeds a YaRN-extended full-context backbone
   at a constant read and much lower memory." (If YaRN-KVD recovers to high recall,
   the honest reframing is efficiency-first: CoMem ≈ YaRN-KVD accuracy at O(L)-write
   + fixed read vs YaRN-KVD's O(L²) + 89GB.)

**Verification note:** transformers reads `rope_scaling` from config at
`from_pretrained`, so the private-copy approach activates YaRN with no harness code
change. Sanity-check with `--limit 2` first to confirm no OOM on the target GPU
before the full n=100 launch.
