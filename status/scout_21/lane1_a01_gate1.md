# lane1_a01_gate1 — A01 gate-1 "third model family's MC interface" scouting

## 0. HEADLINE FOR MAIN (read first)

1. **`.21` IS NO LONGER FREE.** A concurrent agent launched gate-1 itself at ~21:30 —
   8/8 GPUs, 126-151 GiB each, 34-72% util, PIDs 41778-41785:
   `scripts/eval_olmo2_mmlu_content.py --base_model ../models/Llama--Llama2-7b --any_family
   --output_name gate1_llama2_7b --num_shards 8 --shard_index {0..7} --batch_size 16 --add_bos 0`
   Do not double-book .21. (Dolmino download PID 25999 also still alive, CPU-only, 56 min elapsed.)
2. That run covers only the **INTACT Llama-2-7B base**. **No damaged third-family arm exists
   anywhere on either disk** (verified: `grep -l "llama\|mistral\|gemma\|pythia" outputs/*/arch_meta.json`
   → 0 hits; every prune-heal arch_meta is model_family ∈ {olmo2, qwen3, hunyuan_v1_moe, hy_v3}).
   So the in-flight run gives the **Obs-1 direction** (healthy → letter > content) in a third
   family, but **cannot test A01's load-bearing headline** ("letter degenerates BELOW the
   best-constant floor on DAMAGED models"), which is the claim that actually needs replicating.
3. **A validity trap I measured**: the content-interface null is **tokenizer-dependent** and must
   be recomputed per family. Reusing OLMo's `.2845` for a Llama-2 arm would be wrong by ~0.9 pp.
4. The remaining useful work — damaged third-family arms via **training-free layer truncation** —
   fits `.82` and needs ~15 lines of new code plus a KB-scale scp (code is not on zwfy6).

## 1. What gate-1 actually says

`proposal/active/A01-null-calibration-methodology/PROPOSAL.md:71-90`:

```
## 下一步 gate
### 必做
1. 第三个模型家族的 MC interface case。
2. 非 MMLU 的一个 MC benchmark。
3. OLMo full-fp32 forward：检验 bf16 exact tie 是否为因果机制。
4. C4 aggregation 预注册，不再选择性报告 10×。
### 成功条件
- 第三模型/第二 benchmark 保持"instrument validity before comparison"结论；
### Kill 条件
- 第三家族和第二 benchmark 均不复现 interface failure；
```

`STATUS.json`: `next_gate[0] = "third model family"`.

`claims/MMLU_INTERFACE_CASE.md:24-31` — it is also one of three **spin-out** preconditions:
```
## 独立成篇 gate
只有同时完成下列三项，才从 A01 拆为独立 paper：
1. full-fp32 forward 消除 ties，并恢复 letter validity；
2. 第三个模型家族复现；
3. 第二个 MC benchmark 复现。
```

**PASS/FAIL, honestly:**
- PASS = a third family shows the same instrument failure (letter interface on a damaged arm is
  not significantly above the best-constant floor `always-D = .2689`, while content stays above
  its own longest-option floor).
- FAIL alone does **not** kill A01: the kill condition is conjunctive —
  "第三家族 **和** 第二 benchmark **均**不复现". One FAIL narrows the claim to "family-specific
  mechanism", which the dossier already partly concedes (OLMo fails via bf16 ties, Qwen via letter
  prior collapse — `evidence/mmlu_interface_initial_dossier.md:147-173`).
- FAIL **does** kill the MMLU-interface **spin-out paper**, for which family-3 is necessary.
- Verdict: **narrowing gate for A01, kill gate for the spin-out.** More than decorative, not
  existential. And note A01's MC leg is only 1 of 4 constructs — the other three
  (generative prior, representation similarity, probe depth) are untouched by this gate.

## 2. Families A01 already covers

**Family 1 = OLMo-2-7B.** 18 dirs under
`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_mmlu_content_results/`,
**wzc1 only** (`ls -d /apdcephfs_zwfy6/.../olmo2_mmlu_content_results/*/ | wc -l` → **0**).
The 9 canonical arms hardcoded in `code/build_null_calibration_table.py:85-95`: `7B_base`,
`7B_full32_step25000`, `7B_keep8_step121000`, `7B_keep10_step83500`, `7B_keep12_step124000`,
`7B_keep14_step200000`, `7B_freezefront_step200000`, `7B_scratch16L_step200000`,
`7B_shortgpt16_step200000` (+ `7B_keep14_reheal_step67500` = the dossier's 10th). Remaining 8 dirs
are `_wzc1` / `_v2` / `p24_sft` re-run variants.

**Family 2 = Qwen3-8B.** 4 arms, **wzc1 only**:
`qwen3_mmlu_content_results/{qwen3_base, qwen3_f12k2_inherit_s2000, qwen3_f12k2_scratch_s2000,
qwen3_f12k2_inherit_s200k}/`. Verified `qwen3_base/summary.json`: n=14042, n_nan=0, n_shards=8,
letter .7294 / content_norm .5053, meta `model_family: qwen3`.

**Family 3 required.** The dossier already named the intended one (line 219):
"第三个家族（Llama-3-8B ckpt 在 .73）". The concurrent agent instead picked **Llama-2-7B**.

## 3. Third-family weights on disk

**wzc1** (= same disk as `.21`, so plain local `ls` is authoritative):

| path | size | complete? |
|---|---|---|
| `<repo>/models/Meta-Llama-3-8B/` | 4 safetensors 4.977+5.000+4.916+1.168 GB ≈ **16.06 GB** | yes (config+index+tokenizer.json) |
| `<parent>/models/Llama--Llama3-8b/` | same 4 shards ≈ **16.06 GB** | yes (2nd copy) |
| `<parent>/models/Llama--Llama2-7b/` | **40.4 GB** dir | yes — this is what the in-flight run uses |
| `<parent>/models/Llama-3.2-1B/` | 2.49 GB | yes |
| `<parent>/models/AST-official-LLaMA2-7B-2of4/` | 13.5 GB | sparse variant |

(`<repo>` = `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory`,
`<parent>` = `/apdcephfs_wzc1/share_304376610/pighzliu_code`.)

Also on wzc1 parent: OLMo-2-1124-7B (29.2 GB), OLMo-2-0425-1B, Qwen3-{0.6B,1.7B,4B,8B,30B-A3B,32B}-Base,
Hunyuan-A13B-Pretrain, Hy-MT2-30B-A3B, Qwen1.5-MoE-A2.7B, gpt2.
**No Mistral, no Gemma, no Pythia on either disk.**

**zwfy6** (via `.82` ssh): `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/` has
`Llama--Llama3-8b`, `Llama--Llama2-7b`, `Llama-3.2-1B{,-Instruct}`, `OLMo-2-{0425-1B,1124-7B}`,
`open_llama_3b_v2`. Repo `models/` has `Meta-Llama-3-8B` (4 shards, byte-identical sizes to wzc1),
`Llama--Llama2-7b`, `gpt2`, `opt-2.7b`, `bge-m3`, `Beacon-Qwen2-7B`.
⇒ **Llama-3-8B and Llama-2-7B are on BOTH disks. Zero cross-disk weight transfer needed.**
`cais___mmlu` HF cache also present on both (`data/hf_datasets_cache/cais___mmlu`).

**NO damaged third-family arm anywhere.** All `outputs/*/arch_meta.json` are olmo2 / qwen3 /
hunyuan_v1_moe / hy_v3. (`hunyuan_a13b_keep24_fresh2`, `hyv3_probe2_keep{12,24,30}_fresh2` are
arch_meta-only, no `.pt`; only `hyv3_probe2_keep36_fresh2/step200.pt` exists (285 GB) and it is a
step-200 MoE toy, not a healed arm.) This is the load-bearing gap.

## 4. Does a harness generalize? — YES, as of 21:24 today

`scripts/eval_olmo2_mmlu_content.py` was **OLMo-locked** until ~21:24 today. The lock was exactly
one line, in the imported helper `scripts/eval_olmo2_probe2_ppl.py:128`:
```python
model = Olmo2ForCausalLM.from_pretrained(
    base_path, torch_dtype=torch.float32, local_files_only=True)
```
The 792-line `scripts/eval_qwen3_mmlu_content.py` is a **port whose only substantive delta is the
import source** — its own docstring says so: *"The ONLY change is the import source of the
model-construction + tokenisation helpers ... Every protocol detail ... is family-agnostic and
UNCHANGED"*. I confirmed by diff: the deltas are the 3 imports, `Olmo2Config/Olmo2ForCausalLM` →
`Qwen3Config/Qwen3ForCausalLM` in `_selftest`, `cfg.head_dim=16`, docstrings. Scoring maths
identical.

**A concurrent agent has now added the generic path** (uncommitted; `git status` shows
` M scripts/eval_olmo2_mmlu_content.py` + ` M scripts/eval_olmo2_probe2_ppl.py`):
- new `load_base_model_any_family(base_path, device)` in `eval_olmo2_probe2_ppl.py` using
  `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32, local_files_only=True)`,
  returning `meta["mode"]="base_any_family"` + `architecture`. Leaves the OLMo path untouched.
- new `--any_family` flag in `eval_olmo2_mmlu_content.py`, with a guard:
  `--any_family is base-mode only; --ckpt implies OLMo-specific layer surgery`.

**Assessment of the pieces:**
- `encode_pair` (`eval_olmo2_probe2_downstream.py:311-327`) is **family-agnostic**: pure
  `tok.encode(..., add_special_tokens=False)` + trailing-space migration + optional BOS. I
  **empirically verified** prefix-consistency (`whole[:len(ctx_ids)] == ctx_ids`) on 9 real MMLU
  candidate pairs for Llama-3-8B, Llama-2-7B and OLMo-2-7B: **0 violations each**, letter
  `cont_len == 1` for all three (Llama-2 decodes `'A'`, Llama-3/OLMo `' A'` — SentencePiece vs BPE
  space handling; harmless since raw==norm for a 1-token continuation).
- `_safe_lp`, `mcnemar_exact_p`, `paired_bootstrap_diff`, `aggregate`, `merge`, `compare`,
  `load_mmlu_examples`, `score_examples` — all family-agnostic (no arch references).
- `load_pruned_model` / `build_pruned_shell` (`eval_olmo2_probe2_ppl.py:56-122`) are **OLMo-locked
  and do NOT generalize**: `Olmo2Config.from_pretrained` → mutate `num_hidden_layers` →
  `Olmo2ForCausalLM(cfg)` → `load_state_dict(strict=True)`. Hence the `--any_family`+`--ckpt` guard.
  This is precisely why **no damaged third-family arm can be loaded today**.

⇒ For an **intact** third-family base, gate-1 is now a **config change** (already in flight).
   For a **damaged** third-family arm, it **needs new code**.

## 5. Validity requirements — two real problems

**(a) Per-option scores are NOT full precision.** `_safe_lp` = `round(float(x), 6)`
(`eval_olmo2_probe2_downstream.py:74-76`), and `score_examples` stores every option score through it.
The gate-3 script explicitly calls this out (`a01_gate3_fp32_vs_bf16.py:36-41`): *"Scores are stored
at FULL float precision (repr), not rounded to 6 dp ... the smallest nonzero bf16 gap observed is
~2e-5, twenty times the rounding grid."* I sanity-checked that the rounding is adequate at
OLMo-scale margins: recomputing exact top1==top2 ties from the **rounded** archive
`7B_keep8_step121000/per_example_mmlu.jsonl` gives **4303/14042 = 0.30644**, matching the dossier's
30.6% and the concurrent agent's independent `.3064`. So 6 dp is fine for bf16-quantised OLMo
margins, but for an fp32 arm or a family with finer margins it is **not** safe — a gate-1 run that
wants to report tie rates should use the gate-3 full-precision path, not the archive path.

**(b) The gate does NOT compare against .2689/.2845 — the harness compares against .25.**
`eval_olmo2_mmlu_content.py:99` `CHANCE = 0.25`, and `aggregate()` emits
`above_chance = {letter: acc-0.25, ...}`; `compare()` computes
`recovery = (acc_a-0.25)/(acc_b-0.25)`. The construct-appropriate floors live only **downstream**:
- `code/build_null_calibration_table.py` derives always-D from the gold marginal and
  `longest_option_vector(..., conv="split")` for content (line 295: `longest = longest_convs["split"]`).
- `code/a01_gate3_fp32_vs_bf16.py:216` `const_floor = hits / n`, `:293` content vs `longest["split"]`.
So per-example jsonl is the right artifact and the floor must be recomputed downstream — **but**
`build_null_calibration_table.py` hardcodes `MMLU_DIR = "olmo2_mmlu_content_results"` (line 76) plus
a fixed 9-arm `MMLU_ARMS` list, and its only CLI args are `--n-perm/--seed/--n-boot/--out`.
**A third-family analysis therefore needs a new/parameterised analysis script, not just an eval run.**

**(c) ★ The content null is tokenizer-dependent — I measured it.** Recomputed from cais/mmlu
(n=14042) with each family's real tokenizer, split-tie convention:

| family | letter best-constant (always-D) | content longest-option split-tie null | tied-longest rate |
|---|---|---|---|
| OLMo-2-7B | 0.2689 | **0.2845** (matches A01 canonical) | 0.3422 |
| Qwen3-8B | 0.2689 | **0.2833** | 0.3244 |
| Llama-3-8B | 0.2689 | **0.2847** | 0.3429 |
| Llama-2-7B | 0.2689 | **0.2757** | 0.3532 |

Letter marginals A .2295 / B .2465 / C .2551 / D .2689 → always-D `0.2689` is **family-invariant**
(gold-label property). The content floor is **not**: Llama-2 is `0.2757`, **0.88 pp below** OLMo's
`.2845`. Using `.2845` for the in-flight Llama-2 arm would understate its content residual.
**Each family's content floor must be recomputed from its own `cont_tokens`.**

## 6. Node fit

| node | disk | Llama weights? | mmlu cache? | `--any_family` code? | free? |
|---|---|---|---|---|---|
| `.21` | wzc1 | yes | yes | yes (uncommitted) | **NO — 8/8 busy with gate1_llama2_7b** |
| `.82` | zwfy6 | yes (both L2-7B and L3-8B) | yes | **NO** (`grep -c any_family` → 0) | yes, 0 MiB × 8 |

`.82` needs the two uncommitted files scp'd over — `eval_olmo2_mmlu_content.py` (36 KB) +
`eval_olmo2_probe2_ppl.py`. KB-scale, seconds, **not** a cross-disk blocker.

Memory: .21 shows **126-151 GiB/card at `--batch_size 16`** for fp32 Llama-2-7B on 183 GiB L20A.
H20 is 97.8 GiB ⇒ bs16 will very likely OOM, and Llama-3-8B is worse (vocab 128256 vs 32000, so the
fp32 `log_softmax` tensor is ~4× larger). Recommend **BS=4** for Llama-3-8B on .82 (`unsure` on the
exact ceiling — I did not run a GPU probe, per read-only scope). Runtime reference: Qwen3-8B base
took **81-86 s/shard** for ~1755 items at bs8; even 5× slower is well under an hour per arm.

## 7. The missing piece and how to get it cheaply

Damaged third-family arms with **zero training**: load the intact HF model, truncate
`model.model.layers` to the front-N, and score. I **verified the mechanics on CPU**: a
`LlamaForCausalLM` built from Llama-3-8B's real config, with
`m.model.layers = torch.nn.ModuleList(list(m.model.layers)[:2])` and `config.num_hidden_layers=2`,
does a clean forward → logits `(1, 5, 128256)`. Llama has `layer_types = None` so there is no
sliding-window bookkeeping to fix (unlike Qwen3, where `_selftest` must reset `cfg.layer_types`).

This is the exact "step-0 / no-heal" damage condition A01 and Paper B already use as a control
(Paper B P0.2 "step-0 recovery-fraction eval battery"). It is a legitimate, training-free damage
operator and it is the ONLY way to get a damaged third-family arm without weeks of healing.

Cost: ~15 lines (`--truncate_layers N`, base-mode only, recorded in `meta`). This is **new code**,
so gate-1's headline leg is `READY_AFTER_SMALL_FIX`, not `READY_NOW`.

Caveat to state in any writeup: an untruncated-then-truncated arm is **not** compute-matched to
OLMo's healed keepN arms — it is "damage without repair". It tests the interface-degeneration claim
(letter falls below always-D while content holds above its own floor), which is the load-bearing
claim, but it is **not** a substitute for a healed cross-family ladder.
