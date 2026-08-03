# Paper B — P2.5 Qwen3 cross-family protocol-complete supplementary eval (NOTES)

**Status:** harness built + statically verified (CPU). NO GPU run launched (40 卡全忙).
**Type:** PURE EVAL, no training. Ported from the OLMo-2 P0.6/P0.3 evaluators.
**Date:** 2026-08-03.

## 0. What P2.5 answers

The standard *letter* MMLU protocol gives the pruned+healed Qwen3-8B (f12k2 =
keep_front12 + fresh2, a 14/36-layer shell, healed to 200k) a chance-level MMLU
(P2.3: base full-36 = .7297 → f12k2 = .2495 ≈ chance .25). That low number
conflates "no knowledge" with "cannot bind answer content to a bare letter
symbol" (a readout / answer-symbol-binding lag). P2.5 ports the OLMo
protocol-complete evaluators to the Qwen cross-family point so the two families
are scored under a **byte-identical protocol**:

* **content-MMLU** (`eval_qwen3_mmlu_content.py`): three protocols in one run —
  `letter` (single letter likelihood, reproduces P2.3), `content_raw`
  (full-option-text sum-logprob), `content_norm` (length-normalised, **headline**).
  If content recovers while letter stays low → readout lag, not missing knowledge.
* **closed-book QA** (`eval_qwen3_closedbook_qa.py`): PopQA / TriviaQA / NQ-open,
  em / contains / f1, majority floor — the independent-knowledge cross-check.

**Cross-family rule:** only compare **base-normalised recovery / direction** vs
OLMo — never absolute PPL/acc (different family + SlimPajama vs Dolmino continued
pretraining). Above-chance recovery = (acc − 0.25) / (base_acc − 0.25).

## 1. Deliverables (new files only — no shared file touched)

| file | role |
|------|------|
| `scripts/eval_qwen3_mmlu_content.py` | dual/triple-protocol MMLU (letter+content_raw+content_norm), shard/merge/compare/selftest |
| `scripts/eval_qwen3_closedbook_qa.py` | closed-book PopQA/TriviaQA/NQ-open, em/contains/f1 + majority floor, shard/merge |
| `scripts/_run_qwen3_p25_8gpu.sh` | combined 8-shard→merge runner; base/pruned arms; 32-item sanity; MODE=mmlu\|closedbook\|both |
| `paperB/P2_5_qwen_protocol_NOTES.md` | this file |

**Port, not copy:** both python files import the *unified Qwen strict loader* —
`load_base_model` / `load_pruned_model` / `_log` from `eval_qwen3_probe2_ppl.py`
and `_safe_lp` / `encode_pair` from `eval_qwen3_probe2_downstream.py`. So the
shell rebuild (`cfg.num_hidden_layers = keep+fresh`, `cfg.layer_types` reset to
all `full_attention`, strict-load, fp32 master + bf16-autocast) and the
tokenisation are **byte-identical** to every other Paper B Qwen3 eval — zero arch
drift. The content-MMLU `letter` prompt + `encode_pair` are **byte-identical** to
the `mmlu` branch of `eval_qwen3_probe2_downstream.py`, so `letter_acc`
reproduces the P2.3 letter-MMLU numbers item-for-item.

## 2. Dependent checkpoint / model paths (verified on LOCAL wzc1 disk)

* **base (full 36L):** `models/Qwen3-8b-local` — 36 layers, hidden 4096, vocab
  151936, bos_id 151643, eos_id 151645. **It is a symlink to
  `../models/Qwen--Qwen3-8b`** (= `arch_meta.base_model_path`), i.e. the eval
  base is *literally the same weights* the f12k2 shell was carved from → maximal
  consistency. This is the base P2.3 used (`_run_qwen3_probe2_eval.sh` default).
* **f12k2 @ 200k healed ckpt:**
  `outputs/qwen3_minarch_armB_f12k2_200k/final.pt` (47 GB — includes optimizer
  state; the loader reads only `model_state`). `arch_meta.json`:
  arm=`healing_front12+fresh2`, keep_front=12, n_fresh=2, num_hidden_layers=14,
  hidden 4096, vocab 151936, tie=False, n_params=3.95 B. **Load with
  `--keep_front_layers 12 --n_fresh_layers 2`** (also read from ckpt meta; a CLI
  mismatch is a hard error).
  * NOTE: the TODOList wrote the path as `outputs/qwen3_probe2_8B_f12k2/`; that
    does not exist. The real f12k2@200k path is
    `outputs/qwen3_minarch_armB_f12k2_200k/`.

## 3. Qwen-vs-OLMo tokenizer / BOS / prompt alignment

| | OLMo-2-1124-7B | Qwen3-8B (this eval) |
|---|---|---|
| tokenizer `bos_token` | `<|endoftext|>` | **None** |
| tokenizer `bos_token_id` | resolves to a token | **None** |
| `add_bos_token` | None | **False** |
| base protocol | `add_special_tokens=False` (BOS suppressed) | `add_special_tokens=False` (no BOS exists) |

**Verified empirically:** for Qwen `tok.encode(x, add_special_tokens=True)` ==
`tok.encode(x, add_special_tokens=False)` (same length, no BOS ever added). So
`--add_bos 0` (default) on Qwen produces the **OLMo-equivalent no-leading-special-
token input**, and `--add_bos 1` is a **no-op** (guarded by `bos_id is not None`;
Qwen bos_id is None). Both families are therefore scored under the identical
"no BOS, chat_template=False, BASE LM, no SFT" protocol → cross-family
comparison is protocol-fair.

* pad/eos: Qwen `pad_token=<|endoftext|>` (151643), `eos=<|im_end|>` (151645).
  Content-MMLU (teacher-forced LL) does not generate → eos irrelevant. Closed-book
  generation uses `pad_token_id=pad_id` and greedy `do_sample=False,num_beams=1`;
  the prediction is the **first line** of the completion, so which eos fires
  (151645 or max_new_tokens) does not change the metric.
* prompts (base protocol, no chat template, zero-shot, no retrieval):
  * MMLU letter: `"<subject desc>\n\n" + question + "\n" + "A. ..\nB. ..\nC. ..\nD. .." + "\nAnswer:"`, candidate `(" "+letter, len=1)`.
  * MMLU content: same subject-desc + question + `"\nAnswer:"` (lettered body
    DROPPED), candidate `(" "+full_option_text, len=chars)`; `content_norm =
    raw_sum_logprob / #continuation_tokens` (headline).
  * closed-book: `"Question: " + question + "\nAnswer:"`, greedy 32 new tokens.

## 4. Protocol / sample-count checklist (aligned with OLMo P0.6 / P0.3)

* MMLU: `cais/mmlu` config `all`, split `test`, **n = 14,042**; mode = LL-based MC
  (teacher-forced sum-logprob argmax); an item is *valid* iff no candidate in
  either protocol is non-finite (letter & content always scored on the same item
  set → clean within-arm pairing).
* closed-book: PopQA `akariasai/PopQA` test **n = 14,267**; TriviaQA
  `mandarjoshi/trivia_qa` config `rc.nocontext` split `validation` **n = 17,944**;
  NQ-open `google-research-datasets/nq_open` split `validation` **n = 3,610**.
  em/contains/f1 with SQuAD-style normalisation, max over gold aliases;
  `majority_em` floor (most-frequent gold string).
* stats: exact McNemar in **log-space** (verified no OverflowError at
  n≈14k discordant pairs) + paired bootstrap 95% CI (n_boot=10000). `--compare`
  gives cross-arm paired McNemar + bootstrap + above-chance recovery.
* sharding: `examples[shard_index::num_shards]`, stable `item_id = shard_index +
  local_idx*num_shards`; merge sums counts / concatenates per-example rows
  (never mean-of-metric). Per-item option scores / predictions are saved.

## 5. Static verification done (CPU only, NO GPU)

* `py_compile` both python files → OK; `bash -n` runner → OK; `--help` on both → OK.
* `--selftest` (tiny random CPU Qwen3 + synthetic items): schema OK; `content_norm
  == content_raw / #cont_tokens` for every option; **independent recompute of a
  raw sum-logprob matches the harness** (−11.90384 == −11.90384); McNemar known
  cases (p(0,0)=1, p(10,0)=2·2⁻¹⁰, symmetry, p(k,k)=1); paired bootstrap on a
  dominating arm gives diff>0, CI≥0; aggregate/compare run end-to-end. ALL PASSED.
* closed-book metrics dry-run: `normalize_answer` (articles/punct/ws), `_f1`
  (bag-of-words, order-invariant), `score_prediction` (em/contains/f1 max-over-
  aliases), `build_prompt` → all correct.
* large-n McNemar: `mcnemar_exact_p(7000,6800)=0.0903`,
  `mcnemar_exact_p(6000,200)=0` (no overflow) — log-space path validated.

## 6. Launch commands (run on H20 `.104`, diskB; 8-shard→merge)

Env: `PY=/opt/conda/envs/torch-base/bin/python` (`.venv` broken on .104 — see
memory), node `.104`, `WD=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory`.
The runner defaults already match (`PY`, `BASE=models/Qwen3-8b-local`, NGPU=8).

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory

# ── STEP 0 (PRECONDITION): 32-item base sanity, single GPU, content-MMLU ──
SANITY=1 TAG=qwen3_base_full \
  bash scripts/_run_qwen3_p25_8gpu.sh
# → confirm base letter_acc is plausible (well above .25; full run must land ≈.7297).
#   f12k2 sanity too:
SANITY=1 TAG=qwen3_f12k2_step200k \
  CKPT=outputs/qwen3_minarch_armB_f12k2_200k/final.pt KEEP_FRONT=12 N_FRESH=2 \
  bash scripts/_run_qwen3_p25_8gpu.sh

# ── STEP 1: FULL base arm (both content-MMLU + closed-book), 8 GPUs ──
TAG=qwen3_base_full MODE=both \
  setsid nohup bash scripts/_run_qwen3_p25_8gpu.sh \
  > logs/qwen3_p25_base_full.out 2>&1 &

# ── STEP 2: FULL f12k2@200k arm (both), 8 GPUs ──
TAG=qwen3_f12k2_step200k MODE=both \
  CKPT=outputs/qwen3_minarch_armB_f12k2_200k/final.pt KEEP_FRONT=12 N_FRESH=2 \
  setsid nohup bash scripts/_run_qwen3_p25_8gpu.sh \
  > logs/qwen3_p25_f12k2_step200k.out 2>&1 &

# ── STEP 3: cross-arm paired compare (content_norm headline; also letter) ──
PY=/opt/conda/envs/torch-base/bin/python
$PY scripts/eval_qwen3_mmlu_content.py --compare \
  --file_a qwen3_mmlu_content_results/qwen3_f12k2_step200k \
  --file_b qwen3_mmlu_content_results/qwen3_base_full \
  --protocol content_norm --output_name f12k2_vs_base_contentnorm
$PY scripts/eval_qwen3_mmlu_content.py --compare \
  --file_a qwen3_mmlu_content_results/qwen3_f12k2_step200k \
  --file_b qwen3_mmlu_content_results/qwen3_base_full \
  --protocol letter --output_name f12k2_vs_base_letter
```

Results roots: `qwen3_mmlu_content_results/<TAG>/` and
`qwen3_closedbook_results/<TAG>/` (summary.json + per_example_*.jsonl each).

## 7. Alignment validation gate (before trusting content numbers)

1. base **full** letter_acc must ≈ **.7297** (P2.3 base MMLU).
2. f12k2 **full** letter_acc must ≈ **.2495** (P2.3 f12k2 MMLU, ≈chance).
   Both are byte-for-byte reproductions (same letter prompt + `encode_pair` +
   sum-logprob argmax as `eval_qwen3_probe2_downstream.py`). If they do NOT
   match, STOP — the harness/base is misaligned and the content numbers are moot.
3. Only after (1)&(2) match, read `content_norm_acc` and the within-arm
   `letter_vs_content_norm` (content-only-correct count, McNemar p, bootstrap CI)
   and the cross-arm above-chance recovery.

## 8. Open questions / caveats

* **Base-mode eos.** `models/Qwen3-8b-local` config eos = `<|im_end|>` (151645,
  chat-style) whereas `../models/Qwen3-8B-Base` eos = `<|endoftext|>` (151643).
  We deliberately use `Qwen3-8b-local` (= the training/P2.3 base, symlink to
  `Qwen--Qwen3-8b`). For teacher-forced MMLU this is irrelevant; for closed-book
  generation the first-line rule neutralises the eos choice. Documented, not a
  risk, but worth a note if closed-book generations look truncated.
* **Dataset cache.** PopQA/TriviaQA/NQ-open/cais-mmlu are shared with the OLMo
  P0.3/P0.6 runs and should already be cached on .104's `HF_DATASETS_CACHE`; the
  runner still does a one-time `--prepare_data` behind hy-proxy to guard a cold
  cache and avoid an 8-way download race.
* **47 GB ckpt CPU load.** `final.pt` carries optimizer state; the loader only
  extracts `model_state`. Peak host RAM to load the dict is ~47 GB (the OLMo
  keep14 48 GB ckpt loaded fine under the same pattern). Not a GPU concern.
* **`--content_desc`.** Default `full` keeps the MMLU subject description +
  question so the content prompt shares the letter prompt's framing (only the
  lettered body is dropped). `none` gives a bare `Question: …\nAnswer:` — a
  stricter ARC-style variant, available for an ablation if reviewers ask.
