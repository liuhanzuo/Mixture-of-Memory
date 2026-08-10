# COMem Public-Release Audit vs Paper A (CoMem/QCMem) actual experiment code

Read-only comparison of the public release `COMem` (`/tmp/COMem_inspect`) against the
working Paper A code (`src/memory/qcmem/`, `scripts/eval_*qcmem*.py`,
`scripts/train_qcmem_distill.py`, `scripts/bench_qcmem_vs_fullctx.py`, `paper/`).
Nothing was modified. Date: 2026-07-26.

Paper A protocol used for the verdict: Qwen3-8B, resume_j=12, chunk_size=512, sink=bos,
topk=12; selector per-task (variable_tracking→iter_bm25 iter_hop_topk=4, niah_*→bm25);
flagship LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`; **chat_template=False**;
n=100/cell (n=50 for RULER NIAH h2h); official scorers (RULER string_match_all,
BABILong compare_answers, LongBench F1, LoCoMo F1 + GPT-4o judge).

---

## (A) CONSISTENT — matches the protocol, DO NOT touch

- **Model core (`comem/model.py`) — byte-identical logic to `src/memory/qcmem/qcmem_model.py`** (see §C).
- **resume_j / split depth**: registry `qwen3-8b→12`, chunk_size=512 default, sink="bos" default. ✅
- **RULER selector routing** (`ruler.py:272-285` `_resolve_selector`): default `auto` →
  variable_tracking=`iter_bm25`, niah_*=`bm25`; `--iter_hop_topk` default **4**; `--topk` 12.
  Identical to our `scripts/eval_ruler_qcmem.py` (selector=auto, iter_hop_topk=4, topk=12). ✅
- **chat_template = NOT applied anywhere.** grep of the whole COMem tree for
  `apply_chat_template` / `enable_thinking` / `chat_template` → the only hit is
  `babilong.py:168 "chat_template": False` (passed to babilong's `get_formatted_input`, i.e.
  base-style formatting). All drivers tokenize with `tok.encode(text, add_special_tokens=True)`
  — no chat template, no thinking. This is exactly the paper's chat=False pillar. Our own eval
  scripts have a `--use_chat_template/--enable_thinking` option that **defaults to False**, so
  the paper ran chat=False; COMem hard-removes the option → SAFER, still faithful. ✅
- **Scorers**: RULER `_string_match_all_one` = official string_match_all recall (`ruler.py:247`);
  BABILong writes the nested CSV for official `babilong.metrics.compare_answers`
  (`babilong.py:225`, NOT regex); LongBench SQuAD token-F1/EM (`longbench.py:102`). ✅
- **Distill recipe (`train/distill.py`)**: teacher = j=0 read with `peft.disable_adapter()` under
  no_grad; student = LoRA on `layers[resume_j:]` only; bidirectional top-k KL
  (`teacher_topk=64`, `distill_lambda=0.6`); **defaults `lora_rank=32`, `n_ctx=7` (window=4k)** —
  which reproduces the flagship `..._j12_r32_4k` recipe. Matches `scripts/train_qcmem_distill.py`
  logic (our script's own defaults are r16/n_ctx3, but flagship was run at r32/n_ctx7 = COMem's
  defaults). ✅
- **Non-diverging n defaults**: longeval `--n`=50 (== ours 50), longbench `--n`=-1/all (== ours),
  locomo `--n`=-1/all (== ours). ✅
- **max_new_tokens defaults** match ours: babilong 20, longeval 16, locomo 48, ruler 48. ✅
- **Self-test gates** (`comem/selftest.py`, `train/distill.py:run_self_test`): fp32 tol 1e-4,
  j=0 packing == stock full forward + resume identity + KV-decode == recompute. Strong. ✅

---

## (B) MUST-FIX — COMem would produce DIFFERENT numbers/behavior than the paper

| # | file:line | current | should be | why |
|---|-----------|---------|-----------|-----|
| **B1** | `eval/ruler.py:324` | `--limit/--n default=500` + comment "paper default n=500/cell (aligns with official RULER)" | **default 50** (RULER NIAH h2h) / pass `--n 100` for scaling cells | Paper never ran n=500. Our `eval_ruler_qcmem.py:304` default = **50**. The comment is factually wrong ("official RULER" 500 ≠ what the paper ran). A blind `python -m eval.run --benchmark ruler` reproduces neither the h2h (n=50) nor the scaling (n=100) numbers, and takes 5-10× longer. |
| **B2** | `eval/babilong.py:125` | `--limit/--n default=500` + comment "paper default n=500/cell" | **default 100** | Our `eval_qcmem_babilong.py:947` default = **100** (babilong.metrics, n=100/cell). Wrong default + wrong comment. |
| **B3** | `eval/locomo.py` (whole scorer, `run_scoring`/`score_sample`) | F1 + EM + substring-acc ONLY; **no GPT-4o judge** | port the optional GPT-4o judge OR confirm the paper's LoCoMo table is F1/acc-only | Our `eval_qcmem_locomo.py:362-449` has `--use_llm_judge --judge_model gpt-4o` producing a `judge` column. **Decision needed**: the protocol I was given says "LoCoMo F1 + GPT-4o judge", but COMem's OWN paper text (`05_experiments.tex:31`) says "LoCoMo reports F1 and accuracy". If the canonical LoCoMo table has a judge column → this is a hard MUST-FIX (missing code); if F1/acc-only → already consistent (downgrade to A). |
| B4 (cosmetic, same as B1/B2) | `ruler.py:324`, `babilong.py:125` comments | "paper default n=500" | remove/correct | The misleading "paper default n=500" comments should be deleted regardless — they will send reproducers to the wrong n. |

Note (not a code divergence): RULER `--max_new_tokens` default 48 == our default 48. The P3
scaling tables that used 128 passed `--max_new_tokens 128` on the command line, not via the
default — so the code default is faithful; just document that scaling cells need `--max_new_tokens 128`.

---

## (C) DEEP SEMANTIC — comem/model.py vs src/memory/qcmem/qcmem_model.py

**Behaviorally FAITHFUL — the primitives are byte-for-byte identical.** Function-by-function:

- `write_chunk` / `write_chunks` (embed + `layers[0:j]`, chunk-local causal mask, RoPE `0:T`,
  batched-by-length no-padding grouping) — identical.
- `read_core` — identical, including: pack `[sink ; ctx… ; query]`, **fresh contiguous RoPE
  `0:H`**, causal mask, resume `layers[j:L]→norm→lm_head`; the `top_prepay_b>0` middle-band +
  query-local top-band branch; `logits_tail` slicing; and the **block-diagonal** ablation mask
  (`_make_block_diagonal_mask_and_rope`: sink global / chunk within-block / query sees sink+all
  chunks+itself, ⊆ causal) — identical construction.
- KV-decode fast path (`write_prefill` / `read_prefill` / `decode_step`, `_decode_attn_mask`,
  separate bottom/top `DynamicCache`, `resume_j==0` embedding shortcut) — identical.
- `resume_forward_ids`, `full_forward_logits`, `_layer_out_hidden` (MoE tuple-unwrap),
  `_run_layers` (+ grad_checkpoint), LoRA path (peft Linear deltas applied when CoMem calls the
  layers directly; `_common.load_backbone` hands over `base_model.model`) — identical.
- MoE sharded variant `comem/moe.py::CoMemMoE` (device-hopping subclass) mirrors our qcmem_hy3.

**Only real differences (refactor, not semantics):** COMem folds the high-level
`generate`/`generate_from_ids`/`encode`/`_decode_from_pack`/`_MODE_NO_RETRIEVAL`
(comem/kvdirect/hcache) into the model class, whereas our repo keeps that orchestration inside
the eval scripts (`qcmem_generate` in `eval_qcmem_babilong.py:583`). Verified equivalent:
sink=bos write, selector, pack order, KV vs recompute decode, and step-0 EOS suppression all
match. **One minor nuance:** COMem's `_bos_eos` stops on a SINGLE `eos_token_id`; our eval loop
stops on a SET of eos ids (Qwen EOS + end-of-turn). Low impact under chat=False + small
max_new_tokens, but worth noting if a Qwen3 run ever emits `<|im_end|>` first.

---

## (D) PAPER divergence — inventory only (do NOT merge)

Both papers now share the same section scheme (00_abstract, 01_introduction, 02_related,
03_motivation, 04_methodology, 05_experiments, 06_conclusion, 07_limitations). Table sets differ:

- **COMem has, we don't**: `tab_depth.tex` (depth-partition scaling) — \input in COMem
  `05_experiments.tex:281`.
- **We have, COMem doesn't**: `tab_yarn_tax.tex` (P3.1 YaRN-tax) \input at our `05_experiments.tex:131`;
  `tab_pareto.tex` (P3.2 Pareto) \input at our `05_experiments.tex:165`.
- **`tab_scale.tex`**: exists as a FILE in both, but COMem \inputs it (`05_experiments.tex:260`)
  while our working paper does NOT \input it (orphaned in ours).
- One-liner: **COMem's paper = depth-partition + scale tables (tab_depth, tab_scale); our working
  paper dropped those for the newer YaRN-tax + Pareto tables (tab_yarn_tax, tab_pareto) from
  P3.1/P3.2.** User must decide which paper is canonical before publishing.

---

## (E) SECRET / HYGIENE — verdict: **CLEAN**

- Full-tree grep for `apdcephfs`, `share_30xxxxxxx`, `/root/.`, IP literals (`28.*/29.*/30.*` and
  any `d.d.d.d`), `sk-…`, `eyJ…`, `wandb_v1_`, `hf_…`, `BEGIN … PRIVATE KEY`, `api_key`,
  `password`, `sshpass`, `*.woa.com/oa.com`, `ANTHROPIC/OPENAI` → **no matches** (the one prose
  hit in `05_experiments.tex` is "…per task…", benign).
- No absolute cluster paths, no internal IPs, no credentials, no wandb/HF keys anywhere.
- `.gitignore` is adequate: ignores `*.pt/*.bin/*.safetensors/*.ckpt`, `data/`, all
  `*_results/`, `outputs/`, `configs/password*`, `*.pem/*.key/.env`, `__pycache__`, `.venv`,
  LaTeX build artifacts. No weights or data can be committed accidentally.
- **Safe to push after the (B) fixes.**

---

## SYNTHESIS

### Prioritized fix-list (concrete one-line edits)
1. `eval/ruler.py:324` — `default=500` → `default=50`; delete the "paper default n=500" comment.
2. `eval/babilong.py:125` — `default=500` → `default=100`; delete the "paper default n=500" comment.
3. `eval/locomo.py` — **decide**: if the canonical LoCoMo table has a GPT-4o-judge column, port
   `--use_llm_judge`/`judge_model=gpt-4o` from `scripts/eval_qcmem_locomo.py`; else leave as-is
   (COMem is already consistent with its own paper's "F1 + accuracy").
4. (Optional) `README.md`/`run_cell.sh` examples already pass `--n 100`/`--n 50` — keep, and note
   RULER scaling cells need `--max_new_tokens 128`.
5. (Optional) align COMem `_bos_eos` to a multi-EOS stop set for Qwen3 robustness.

### Recommendation
COMem is an **intentional clean-room re-implementation that is behaviorally faithful** — the
CoMem model primitives are byte-identical to our `qcmem_model.py`, the distill recipe defaults
reproduce the flagship, the selector routing / sink / chunk / RoPE-reset all match, and
chat_template is correctly never applied. It has **NOT** diverged in semantics; only a handful of
**CLI defaults** (n=500 in ruler+babilong) and possibly the **LoCoMo judge** need aligning. No
deep code sync required.

**Push safety: YES after B1+B2 (secret-scan CLEAN).** Resolve B3 (judge yes/no) before claiming
the LoCoMo numbers are reproducible from the release.
