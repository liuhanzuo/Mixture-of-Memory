# P1.2 — Content-depth probe: full protocol + robustness

**Date:** 2026-08-01  ·  **Node:** `.73` (28.85.35.73, port 36000, diskB, 8× H20, GPUs 0-6)
**Owner deliverable:** Paper A item **P1.2** — fully specify the "content-depth probe
method + robustness" that Paper A's depth-partition table (`paperA/sections/tab_depth.tex`,
"content-$j \approx 0.45L$, near scale-invariant") currently under-specifies.

This EXTENDS the existing per-layer logit-lens / truncation-downstream probe
(`scripts/probe_linguistic_layerwise.py`, `scripts/probe_truncated_downstream.py`,
`status/QCMEM_J_DETERMINATION.md`). New self-contained driver:
`scripts/probe_p1_2_content_depth.py` (reuses the old loader/extractors/verbalizers by
import, does **not** modify them). All BABILong/paper eval conventions preserved:
mechanism experiment, generation-based (native) readout uses `chat_template=False`.

**Per project rule: this record edits NO `.tex` file and does NOT touch `paperA/TODOList.md`.**
Numbers + protocol + provenance are handed to `main`, who folds them into the paper.

---

## 0. TL;DR verdict

- **The mechanism claim is ROBUST across 3 model families.** In every model, semantic content
  is *linearly decodable* in the mid-network (knee at ~0.17–0.53 $L$ depending on task),
  while the model's **own** readout pathway (logit-lens verbalizer) does not surface that
  content until **much deeper** (0.64–1.0 $L$). The mean "readout gap" (native knee − linear
  knee) is **+0.43 $L$ (Qwen), +0.59 $L$ (OLMo), +0.68 $L$ (Llama)** — i.e. *linearly
  decodable ≠ actually used by the model*, exactly the gap the self-distilled adapter closes.
- **The specific constant "content-$j \approx 0.45L$, near scale-invariant" is Qwen-family /
  task-averaged, NOT universal.** Rigorous 5-seed content-$j$ (mean over SST2+WiC+RTE):
  **Qwen3-8B 0.393 $L$**, **Llama-3-8B 0.269 $L$**, **OLMo-2-7B 0.285 $L$**. The paper's
  8B value (0.44 $L$ = L16) lies within the Qwen per-task CI but is above the cross-family
  values. **Recommendation:** soften the "≈0.45 $L$ universal / scale-invariant" wording to a
  Qwen-family, task-averaged figure; the family-invariant claim is the *ordering* (content
  mid-network ≪ model-readout depth), not the constant.
- Controls confirm the probe reads genuine content: SST2 probe peak **0.90** vs lexical-only
  0.71 vs position-only 0.56 ≈ majority 0.56; random-label peak 0.54 (selectivity +0.36).
- Two non-Qwen families replicated (**Llama-3-8B base** and **OLMo-2-1124-7B base**), exceeding
  the "≥1 non-Qwen" requirement.

---

## 1. Probe protocol (the six under-specified pieces, nailed down)

### 1.1 Data & labels
Three binary semantic tasks, all from HuggingFace **train** splits (labels certain):

| Task | Source (HF) | Signal | Feature | dim | majority |
|------|-------------|--------|---------|-----|----------|
| SST2 | `nyu-mll/glue` `sst2` | sentence sentiment | mean-pooled sentence hidden state | $H$ (4096) | 0.558 |
| WiC  | `aps/super_glue` `wic` | same word-sense across 2 sentences | target-word states of both, combined | $4H$ (16384) | 0.500 |
| RTE  | `nyu-mll/glue` `rte` | entailment of sentence pair | mean-pooled states of both, combined | $4H$ (16384) | 0.502 |

Pair combiner (reused `combine_pair`): `[a, b, a−b, a·b]` per layer → $4H$.

### 1.2 Sample count & split
- **Fixed labelled pool**, `pool_seed=0`, **stratified** subsample of the train split, target
  `n_pool=3000` (SST2/WiC = 3000; RTE train only has 2490 → whole split).
- Per-layer features are extracted **once** (single forward with `output_hidden_states`),
  stored fp16 on CPU. `inf` from fp16 overflow of massive-activation dims is mapped to 0 via
  `np.nan_to_num(posinf=0,neginf=0)` before scaling (guarded; no NaN reaches the fit).
- Each **seed** = a distinct **stratified 60/20/20 train/dev/test** partition of the same pool
  (`sklearn.train_test_split`, `stratify=y`). Because extraction is deterministic and the fit
  is deterministic given data, the split is the sole stochastic axis (+ random-label
  permutation), so the reported CI is the **partition-induced variance of knee98**.
  Typical sizes: SST2 n_train=1800/dev=600/test=600; RTE 1494/498/498.

### 1.3 Probe architecture / regularization / optimizer
- **L2-regularized logistic regression** (`sklearn.linear_model.LogisticRegression`,
  default `penalty=l2`, solver **lbfgs**, `max_iter=1000`) — a single affine readout of the
  **`StandardScaler`-normalised** pooled layer-$l$ hidden state.
- Inverse-reg strength **C selected on the dev split** from grid `{0.1, 1.0, 10.0}`; report
  **held-out TEST accuracy** (and balanced/macro-recall).
- Run independently for every hidden state $l \in \{0,\dots,L\}$ ($l{=}0$ = embedding output,
  $l{=}L$ = top layer), every seed, every task, every model.

### 1.4 knee98 — precise definition
For a task and a backbone with $L$ transformer layers, let $a(l)$ be the held-out TEST accuracy
of the layer-$l$ linear probe. Peak decodability $A = \max_{0\le l\le L} a(l)$. Then

$$\text{knee98} \;=\; \min\;\{\, l \in \{0,\dots,L\} \;:\; a(l) \ge 0.98\,A \,\}, \qquad
\text{fractional depth} = \text{knee98}/L.$$

"content-$j$" of a model = **mean of knee98 over the semantic task set**, CI over
{seeds × tasks}. (0.98 rather than 1.0: per-layer test accuracy is noisy near the peak;
$0.98A$ is inside the peak's sampling noise but robustly above the rising shoulder, so knee98
marks where the curve has essentially plateaued.)

### 1.5 Seeds & CI
**5 seeds** `{0,1,2,3,4}` (exceeds the ≥3 requirement). CI = **Student-t 95%** half-width
over the per-seed values (`scipy.stats.t.ppf(0.975, n−1)`). content-$j$ CI pools all
15 points (3 tasks × 5 seeds).

### 1.6 Controls (each a separate probe run)
- **lexical-only**: L2-logreg on binary uni+bi-gram `CountVectorizer` (≤50k feats) of the raw
  text — surface bag-of-words ceiling (model-independent).
- **position-only**: L2-logreg on `[len, √len]` token counts — length/position confound.
- **random-label**: TRAIN labels permuted, evaluated on REAL test labels →
  Hewitt–Liang **selectivity** = real_acc − random_acc.
- **class-balance**: majority baseline reported; plus a **balanced-train** probe (down-sampled
  to the minority count) and **macro-recall** (`balanced_accuracy_score`) per layer.

### 1.7 Native ("controlled") readout — decodable vs used
For each layer $l$, apply the model's **own** `final_norm + lm_head` to hidden$[l]$ at the last
prompt token and argmax over the verbalizer class tokens (SST2 ` negative`/` positive`,
RTE ` yes`/` no`, WiC ` yes`/` no`) — the model's native output pathway (`logit-lens`),
forward-only, `n_native=1000`, left-padded. This is the "more controlled readout" that a
plain linear probe is compared against.

---

## 2. Results — content-$j$ (linear-probe knee98), 5 seeds

**Aggregate (mean over 3 tasks × 5 seeds = 15 points):**

| Model | $L$ | content-$j$ (layer) | CI95 | content-$j$ ($/L$) | CI95 | std |
|-------|-----|--------------------|------|--------------------|------|-----|
| **Qwen3-8B**   | 36 | 14.13 | [11.50, 16.77] | **0.393** | [0.319, 0.466] | 0.132 |
| **Llama-3-8B** | 32 | 8.60  | [6.52, 10.68]  | **0.269** | [0.204, 0.334] | 0.117 |
| **OLMo-2-7B**  | 32 | 9.13  | [6.84, 11.42]  | **0.285** | [0.214, 0.357] | 0.129 |

**Per-task knee98 ($/L$, mean [CI95]) and peak TEST accuracy:**

| Task | Qwen knee | Qwen peak | Llama knee | Llama peak | OLMo knee | OLMo peak |
|------|-----------|-----------|------------|------------|-----------|-----------|
| SST2 | 0.389 [0.334,0.443] | 0.902 | 0.275 [0.224,0.326] | 0.897 | 0.281 [0.172,0.391] | 0.905 |
| WiC  | 0.261 [0.178,0.344] | 0.695 | 0.194 [−0.032,0.419] | 0.700 | 0.169 [0.093,0.244] | 0.696 |
| RTE  | 0.528 [0.401,0.654] | 0.629 | 0.338 [0.305,0.370] | 0.641 | 0.406 [0.272,0.541] | 0.636 |

Notes: (i) **SST2 is the cleanest content signal** (peak ~0.90, clear rise-then-plateau);
WiC/RTE peaks are modest (0.63–0.70), so their knee98 is noisier (wide CI; Llama-WiC CI even
dips below 0) and partly reflects "the weak signal that exists appears early" rather than a
strong content depth. (ii) Reconciliation with the paper: Qwen SST2 per-seed knee98 layers were
`[12,15,14,16,13]` → the paper's **8B content-$j$ = L16 (0.44 $L$)** sits at the top of the
Qwen SST2 CI and inside the Qwen layer CI [11.5,16.77]; the rigorous multi-seed, multi-task
mean is **L14 (0.393 $L$)**.

**Representative curve — Qwen3-8B SST2 (seed 0), linear probe $a(l)$:**
embedding l=0 already 0.747 (> majority 0.558) → rises to knee **l=12 (0.898)** → peak
**l=17 (0.908)** → **drops back to 0.847 at the top layer l=36** (top layers repurpose the
representation for next-token prediction, partly overwriting the clean linear sentiment — itself
evidence that the mid-layer linearly-decodable representation is *not* the one the model emits).

---

## 3. Controls — the probe reads genuine content, not artifacts

Means over 5 seeds (lexical/position are ~model-independent; values agree across the 3 models,
a sanity check — lexical is text-only, position is tokenizer-only):

| Task | probe peak | lexical-only | position-only | random-label peak | majority |
|------|-----------|--------------|---------------|-------------------|----------|
| SST2 | ~0.90 | 0.707 | 0.564 | 0.539 | 0.558 |
| WiC  | ~0.70 | 0.598 | 0.536 | 0.547 | 0.500 |
| RTE  | ~0.64 | 0.534 | 0.483 | 0.541 | 0.502 |

- **vs lexical-only:** SST2 0.90 ≫ 0.71, WiC 0.70 > 0.60, RTE 0.64 > 0.53 — the probe recovers
  content beyond bag-of-words (largest margin on SST2).
- **vs position-only:** 0.48–0.56 ≈ chance — sentence length carries no probe signal.
- **random-label (selectivity):** peak ≈ majority (0.54) → the probe cannot fit permuted labels;
  selectivity = peak − random ≈ **+0.36 (SST2)**, +0.15 (WiC), +0.09 (RTE). SST2 is a
  high-selectivity task; WiC/RTE are low-selectivity (genuinely hard, peak barely above chance).
- **class-balance:** all comparisons are against the per-task majority baseline above; a
  balanced-train probe + macro-recall are stored per-layer in each JSON (`balanced_train_acc`,
  `balanced_acc_metric`) — knee98 on the class-balanced probe is within CI of the standard knee.

---

## 4. Linear-decodable ≠ used-by-model (native/controlled readout)

Native logit-lens verbalizer readout, per model: knee98 ($/L$) of the model's OWN output
pathway, and its peak accuracy.

| Task | Qwen knee (peak) | Llama knee (peak) | OLMo knee (peak) |
|------|------------------|-------------------|------------------|
| SST2 | 0.639 (0.938) | 1.000 (0.752) | 0.750 (0.913) |
| WiC  | 0.889 (0.690) | **0.125 (0.529)†** | 0.875 (0.601) |
| RTE  | 0.944 (0.813) | 0.969 (0.530) | 1.000 (0.791) |

**Readout gap = native knee − linear knee** (per model, over well-behaved tasks):
- Qwen: SST2 +0.25, WiC +0.63, RTE +0.42 → **mean +0.43 $L$**
- OLMo: SST2 +0.47, WiC +0.71, RTE +0.59 → **mean +0.59 $L$**
- Llama: SST2 +0.73, RTE +0.63 → **mean +0.68 $L$** (WiC excluded, see †)

† **Llama-3-8B WiC native readout is degenerate** (peak 0.529 ≈ chance): its verbalizer never
works, so the "98%-of-peak" knee (0.125 $L$) is an artifact of a flat near-chance curve, not a
real early readout — excluded from the gap average. (This is exactly why we report *both* the
linear probe and the model's own readout: a base model's zero-shot verbalizer can be uninformative
even where the content is linearly present.)

**Illustration — Qwen3-8B SST2 native curve:** flat at chance (~0.45–0.56) for layers 0–21,
then a sharp jump at l≈24 to peak **0.938 at l=26**, knee **l=23 (0.639 $L$)**. So sentiment is
linearly readable at 0.90 by **l=12 (0.33 $L$)** but the model's own pathway is at chance until
**l≈23 (0.64 $L$)**. The ~0.3 $L$ separation between "linearly present" and "model-emitted" is
the core P1.2 evidence and reproduces in all three families.

---

## 5. Robustness verdict & recommendation for `main`

1. **Robust (keep):** the qualitative depth partition — content is linearly present in the
   mid-network **well before** the model's own readout surfaces it — holds across Qwen3-8B,
   Llama-3-8B, and OLMo-2-7B, with 5-seed CIs. This directly supports the "gap = adapter's job"
   thesis of `tab_depth`.
2. **Qualify (revise wording):** "content-$j \approx 0.45L$, **near scale-invariant**" is a
   **Qwen-family, task-averaged** figure. Cross-family, rigorous content-$j$ is **0.39 $L$
   (Qwen) / 0.27 $L$ (Llama) / 0.29 $L$ (OLMo)** — the paper's 0.44 $L$ (8B) is within the Qwen
   CI but is not reproduced by the two non-Qwen families. Suggested phrasing: *"content sits in
   a mid-network band (~0.27–0.44 $L$ across families; ~0.39 $L$ for Qwen3-8B under a 5-seed
   3-task probe), consistently far shallower than the model's own readout depth."* Do **not**
   claim a universal 0.45 $L$ constant.
3. **Method transparency (add to appendix):** report the probe protocol of §1 verbatim
   (data/labels/split/architecture/regularization/optimizer), the knee98 formula, the four
   controls, and the linear-vs-native distinction. State that SST2 is the high-selectivity anchor
   task and WiC/RTE are low-selectivity (so per-task knees vary and WiC/RTE CIs are wide).

---

## 6. Exact commands & provenance

**Isolated env (no shared-env mutation):**
```
# sklearn/scipy absent from torch-base and .venv -> installed to a project-local dir
pip install --target=.p1_2_pylibs "scikit-learn>=1.3"   # sklearn 1.9.0, scipy 1.18.0, joblib
export PYTHONPATH=.p1_2_pylibs                            # used for every run
# HF datasets fetched via proxy: http(s)_proxy=http://hy-proxy.woa.com:3128
# STRICT thread pinning (mandatory): OpenBLAS ignores OMP/MKL -> set explicitly
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
```

**Per-(model,task) run (9 total, one GPU each):**
```
CUDA_VISIBLE_DEVICES=$g /opt/conda/envs/torch-base/bin/python \
  scripts/probe_p1_2_content_depth.py --mode run \
  --model_path <MODEL> --task {SST2|WiC|RTE} \
  --device cuda:0 --dtype bf16 --max_len 128 --batch_size 32 \
  --n_pool 3000 --n_native 1000 --seeds 0,1,2,3,4 \
  --c_grid 0.1,1.0,10.0 --n_jobs 16 --results_dir results/p1_2 \
  --out results/p1_2/<model>_<task>.json
```
Driver (launches all jobs, thread-pinned): `scripts/_p1_2_relaunch.sh`.
Models: Qwen `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b`;
Llama `.../Mixture-of-Memory/models/Meta-Llama-3-8B`; OLMo `.../models/OLMo-2-1124-7B`.

**Aggregate:**
```
PYTHONPATH=.p1_2_pylibs python scripts/probe_p1_2_content_depth.py \
  --mode aggregate --results_dir results/p1_2 --out results/p1_2/p1_2_summary.json
```

**Sanity gate:** all 9 jobs completed, 5/5 seeds each, 0 crashes (a first launch thrashed on
BLAS oversubscription — 576 threads on 384 cores — and was killed & relaunched with the thread
pinning above; the 2 finished SST2 JSONs were preserved).

**Artifacts (LOCAL `results/p1_2/` on wzc1; also on `.73` diskB same relative path):**
- `{qwen3_8b,llama3_8b,olmo2_7b}_{SST2,WiC,RTE}.json` — per-(model,task): meta, per-layer
  curves (acc / balanced / random-label / selectivity / chosen C), per-seed knee98, controls,
  native readout curve.
- `p1_2_summary.json` — content-$j$ + per-task rollup used in §2–§4.
- Script: `scripts/probe_p1_2_content_depth.py` (new; reuses old probe modules by import).
