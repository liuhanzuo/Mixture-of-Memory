# P3.2 — Fair over-window full-context control (YaRN / native-128k)

**Reviewer ask (P3.2):** Qwen3-8B native window is 40,960; our full-context /
KV-Direct results at 128k/256k run with YaRN **NOT** activated, so they are an
*unextended* reference, not a fair length-extended upper bound. Reviewer options:
(1) add a **YaRN-enabled** Qwen3-8B control, OR (2) a natively-128k same-size model,
OR (3) explicitly **downgrade** >40k full-context numbers to "unextended reference."

## Status: RESOLVED via option (3) — wording (reviewer-sanctioned). ✔

Already in the paper (committed):
- `05_experiments.tex` §Models (lines 4–8): states native window = 40,960; the
  131,072 figure is an *un-activated* YaRN extrapolation limit; >41k inputs fall
  outside the trained regime.
- `05_experiments.tex` §Baselines (lines 21–27): KV-Direct runs at the **native
  40,960 window with YaRN not activated**, so beyond 40,960 it is an *unextended*
  full-context reference — a **lower bound** on what a YaRN-extended same-size
  backbone would achieve, not a fair length-extended upper bound. We explicitly do
  **not** claim CoMem beats a length-extended full-context model at 128k/256k; the
  claim is only that CoMem stays usable where the *unextended* backbone breaks.

This is exactly reviewer option (3), which the reviewer listed as acceptable.

## Optional strengthener (empirical YaRN-KVD) — DEFERRED (no clean 80GB GPU)

**Why deferred (2026-07-26):** the only informative data point is **128k**
(unextended KVD collapses to 0 there — tab_scaling niah_single/multikey/var-track
128k = 0/0/0; at 64k unextended KVD already works 100/96/98, so a 64k YaRN control
is uninformative). A 128k full-context forward needs **~89 GB** (tab_eff). Current
hardware: `.82` fully saturated (8×100%), LOCAL busy (freeze_front #59), `.104` GPU5
free but **shared** (other jobs on-node) → 128k full-ctx OOMs there (confirmed: the
P3.1 pareto full-ctx 128k OOM'd on GPU5). No all-free 80GB+ GPU available.

**Ready-to-run recipe (execute when a clean full 80GB+ GPU frees):**

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
