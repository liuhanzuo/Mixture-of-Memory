# Next Gates Readiness Report
Generated: 2026-08-08 (CPU-only prep work)

---

## Deliverable 1 — A01 gate-4: C4 aggregation pre-registration

**Status: READY_NOW (CPU-only, no GPU)**

### Assets verified on wzc1

| File | Size | Present |
|---|---:|---|
| `evidence/null_calibration_p1_nperm2000.json` | 25 KB | YES |
| `evidence/null_calibration_obs4_nperm2000.json` | 62 KB | YES |
| `code/build_null_calibration_table.py` | ~53 KB | YES |
| `proposal/shared/representation/repr_alignment_results.json` | present | YES |
| `results/p1_2/p1_2_summary.json` | present | YES |
| `olmo2_mmlu_content_results/7B_base/` | present | YES (all 9 arms) |
| `data/squad_val.jsonl` | 2000 lines | YES |
| `evidence_squad_label_prior/*.json` | present | YES |

### What the script does

`build_null_calibration_table.py` already enumerates all defensible C4 variants in its
`main()` function (the `variants` dict, lines 988–1005). The gate-4 script
`a01_gate4_c4_prereg.py` reuses the same `leg_probe()` helper and reports every variant
in a structured JSON with one pre-registered primary choice.

The variants are:
1. **Qwen+OLMo, native 3-task mean, pooled** (headline / pre-registered primary)
2. Qwen+OLMo, native 3-task mean, per-model then avg
3. All 3 models, native 3-task mean, per-model then avg
4. All 3 models, native = SST2 only (matched support)
5. Qwen+OLMo, native = SST2 only

Primary pre-registered = variant 1 because: (a) Llama's WiC/RTE native verbalizers sit at
chance (native_sst2 = 1.0, RTE = 1.0), making Llama's 3-task native aggregate unreliable
as a null; (b) pooled vs per-model-then-avg is symmetric for 2 equally-sized models; (c) this
is the variant already used in the existing `build_null_calibration_table.py` output —
pre-registering the EXISTING choice prevents any claim that a different choice was adopted
post-hoc.

**RESULT (computed and confirmed):**

| C4 variant | C4 frac | Full-table span | Gate |
|---|---:|---:|---|
| **V1 Qwen+OLMo, 3-task, pooled (PRIMARY)** | **0.7724** | **10.04×** | **PASS** |
| V2 Qwen+OLMo, 3-task, per-model avg | 0.7677 | 9.98× | FAIL |
| V3 All 3 models, 3-task, per-model avg | 0.7074 | 9.20× | FAIL |
| V4 All 3 models, SST2 only | 0.6852 | 8.91× | FAIL |
| V5 Qwen+OLMo, SST2 only | 0.5278 | 6.86× | FAIL |

Other three legs: C1=0.2094, C2=0.2436, C3=0.0769.
Spread within C4 variants alone: 1.46×.

**Key finding**: the 10× headline is marginal. It passes only under the pre-registered primary
variant (V1, span=10.04×); every alternative drops below 10×. The spread is 6.86–10.04×, 
i.e. "about 7–10×" is the honest range. This confirms the gate-4 purpose: the pre-registration 
removes the selective-reporting attack. The paper should report "spans ~7–10× depending on 
aggregation choice, with the pre-registered primary at 10.04×."

### Script created

`proposal/active/A01-null-calibration-methodology/code/a01_gate4_c4_prereg.py`

Launch command (CPU, ~30 sec, already run):
```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
python3 proposal/active/A01-null-calibration-methodology/code/a01_gate4_c4_prereg.py \
  --out proposal/active/A01-null-calibration-methodology/evidence/gate4_c4_prereg.json
```
Output JSON: `proposal/active/A01-null-calibration-methodology/evidence/gate4_c4_prereg.json` (created).

---

## Deliverable 2 — A03 next floor gate: axes measurability

**Status: PARTIAL — one axis (NQ-open) is READY_NOW; two axes (conflicting knowledge,
new injected facts) are BLOCKED_NO_DATA_ON_EITHER_DISK**

### Dataset search results

#### Old parametric knowledge axis (already floor-certified)
- MMLU: `data/hf_datasets_cache/cais___mmlu` — wzc1 YES, zwfy6 YES
- PopQA: `data/hf_datasets_cache/akariasai___pop_qa` — wzc1 YES, zwfy6 YES
- TriviaQA: `data/hf_datasets_cache/mandarjoshi___trivia_qa` — wzc1 YES, zwfy6 YES

#### Multi-evidence axis — partial data available
- **NQ-open** (`google-research-datasets/nq_open`, 3610 q validation): zwfy6 YES
  (`data/hf_datasets_cache/google-research-datasets___nq_open`, 6.7 MB), wzc1 NO.
  Harness already implements it (`eval_olmo2_closedbook_qa.py` lines 174–193, it is a supported
  task that can be passed as `--tasks nq_open`; `CLOSEDBOOK_TASKS` list only defaults to
  popqa+triviaqa but nq_open is fully implemented). Expected n=3610.
- HotpotQA fullwiki (multi-hop, closed-book): NOT in hf_datasets_cache on either disk.
  The `data/longbench_raw/data/hotpotqa.jsonl` IS on wzc1 (11 MB) but it is LongBench format
  (54k-char context passage per item) — it is an open-book/RAG eval, not a closed-book
  parametric test. Distinct from what A03 needs.
- 2WikiMultiHopQA, MuSiQue: data/longbench_raw versions exist on wzc1 (5.8 MB, 14 MB) but
  again in LongBench (with-context) format, not closed-book parametric format. NOT suitable.
- HotpotQA fullwiki closed-book is on HuggingFace as `hotpot_qa fullwiki` — fetchable via proxy
  with HF token in `configs/password_hf_token.txt`, ~550 MB download.

#### Conflicting/updated knowledge axis
- CounterFact: NOT on either disk. Fetchable from HuggingFace (`zchuhui/CounterFact` or
  direct construction from Wikipedia revision diffs). ~20 MB download.
- MQuAKE, zsRE, KnowEdit: NOT on either disk. Fetchable but need new harness code (not just
  a new jsonl, because the task requires context-injection or fact-editing, which the
  current closed-book harness does not do).
- **Blocker**: A03's "updated/conflicting knowledge" axis requires the model to answer a
  question where the "correct" answer contradicts the model's parametric knowledge. The
  current harness `eval_olmo2_closedbook_qa.py` just runs greedy QA with no injected
  context. Without injected context, CounterFact-style "updated" facts are not measurable
  closed-book (the model will emit its old parametric answer, which would be WRONG against
  the new gold). This axis requires either (a) a synthetic injection setup or (b) an
  open-book test with the new-fact context injected in-prompt. Both require new code.

#### New injected facts axis
- By definition, there are no standard closed-book benchmarks for "post-training new facts":
  those facts would have to be injected via CPT or in-context, which is what the 6-arm
  design itself tests. The floor gate for this axis would have to test whether the harness
  can detect ABOVE-floor performance on freshly injected facts.
- The only practical closed-book proxy is NQ-open-style questions about less-frequent entities
  (testing the tail of parametric knowledge). NOT a genuine "new injected facts" test.
- **Blocker**: no standard dataset for this axis without new synthetic construction or
  real CPT runs.

### Harness generalizability

`scripts/eval_olmo2_closedbook_qa.py` argument parsing (lines 351–376):
```python
p.add_argument("--tasks", type=str, default=",".join(CLOSEDBOOK_TASKS))
```
The tasks parameter accepts a comma-separated list. NQ-open is already fully implemented at
lines 174–193. A new axis that is "just another jsonl in the same format" would require adding
a new `load_task_examples` branch — roughly 15 lines of code per task. For axes whose data
can be expressed as `(question, [gold_aliases])` pairs, the integration is minimal.

### Floor gate protocol extracted from A03_1B_FLOOR_VERDICT.md

Must repeat for each new axis:
1. Best-constant null maximised over candidate answer strings (≥300 candidates)
2. Length-matched null for `contains` metric (match arm's mean prediction length)
3. Paired bootstrap n_boot=10000 on per-item difference vectors
4. Exact McNemar (when null is binary)
5. BH q=0.05 across the whole (arm × task × interface) family
6. 8-shard assertion: `ns==8` before any merge is trusted
7. Exact item-count assertion for each task before merge

`analyze_1b_knowledge_floor.py` implements items 1–5 already. Items 6–7 are in the
bash driver.

### Driver created

`scripts/_run_a03_axes_floor_82.sh` — covers NQ-open (the only new-axis data available
on zwfy6). The driver is adapted from `_run_a03_1b_floor_82.sh`.

**Note**: NQ-open is on zwfy6 only; the driver must run on a zwfy6 node (.82, .73, or .104).

**Note on analysis follow-up**: `analyze_1b_knowledge_floor.py` hardcodes only
`("popqa", 14267, ...)` and `("triviaqa", 17944, ...)` at lines 427-428. After the
NQ-open eval runs, a one-time extension of that script to also iterate over
`("nq_open", 3610, "em")` will be needed to compute the floor-calibrated residuals.
The data loading function `load_cb_arm` (line 216) and all statistical machinery
already work for any task — only the task list loop needs extending.

Launch command (target node: .82, 8× H20):
```bash
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
bash scripts/_run_a03_axes_floor_82.sh
```
Wall time estimate: ~15 min for 3 arms × 1 task on 8 H20 GPUs.

---

## Deliverable 3 — A02: canonical Read-LoRA identity

**Status: RESOLVED (CPU provenance work only)**

### Findings

| Candidate | Size | Disk | Evidence tying to flagship |
|---|---|---|---|
| `outputs/qcmem_distill_qwen_j12_r32_4k/final/` | 223 MB (adapter_config.json 1.4KB + adapter_model.safetensors 223MB + README.md 5.2KB) | **wzc1 YES, zwfy6 YES** | Multiple scripts, notes, provenance docs (see below) |
| `outputs/lora_best_ref/` | 0 bytes (empty directory) | wzc1: DOES NOT EXIST; zwfy6: empty dir (size=0, created May 29) | No weights; placeholder only |

**The canonical flagship Read-LoRA is `outputs/qcmem_distill_qwen_j12_r32_4k/final`.**

Adapter config confirms: r=32, lora_alpha=64, layers_to_transform=[12..35], target_modules all
attention+MLP projections, base_model = Qwen--Qwen3-8b. Matches "r32, layers 12..35" 
description in paperA/P0_16_E0_NOTES.md exactly.

Evidence tying this path to flagship Paper A rows (>10 independent citations):
- `paperA/P0_16_E0_NOTES.md`: "Flagship LoRA: outputs/qcmem_distill_qwen_j12_r32_4k/final"
- `paperA/P0_19_RULER_NOTES.md`: "flagship LoRA outputs/qcmem_distill_qwen_j12_r32_4k/final"
- `status/PAPERA_RESULTS_CONSOLIDATED.md` line ~270: "distilled LoRA = outputs/qcmem_distill_qwen_j12_r32_4k/final"
- `status/FLAGSHIP_TRAINING_COST.md`: "Target artifact: outputs/qcmem_distill_qwen_j12_r32_4k/final"
- `scripts/_ablation12_crosschunk_chatFALSE_driver.sh` line 36: `LORA=outputs/qcmem_distill_qwen_j12_r32_4k/final`
- `scripts/_run_p0_19_ruler_paired.sh` line 44: `LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"`
- `scripts/_infb_orchestrate_p21.sh` line 65: `--lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final`
- `status/P3_LORA_SEED_VARIANCE.md` line 15: "flagship) outputs/qcmem_distill_qwen_j12_r32_4k/final"
- `status/P0_3_MATCHED_N100.md` line 21: "LoRA adapter: outputs/qcmem_distill_qwen_j12_r32_4k/final"
- `status/COMEM_RELEASE_AUDIT.md`: "flagship LoRA outputs/qcmem_distill_qwen_j12_r32_4k/final"

`lora_best_ref` was created May 29, 2021 as an empty placeholder and never populated.
It has no relation to the actual LoRA.

### Write-LoRA (for A02 completeness)

| Candidate | Size | Disk | Status |
|---|---|---|---|
| `outputs/qcmem_writepath_distill_qwen_j12_r32/` | 556 MB (step500–2500) | **zwfy6 only** | Training configured for 4000 steps but stopped at step2500 (no `final/`). `distill_args.json` confirms it used `read_lora_path = outputs/qcmem_distill_qwen_j12_r32_4k/final` — cross-confirming the Read-LoRA identity. |
| `outputs/qcmem_writepath_distill_qwen_j12_r32_b200/` | present on wzc1 | wzc1 only | Different run from b200 node |

**Additional cross-confirmation**: The Write-LoRA `distill_args.json` on zwfy6 explicitly
records `"read_lora_path": "outputs/qcmem_distill_qwen_j12_r32_4k/final"` — this is a
third-party confirmation from the training pipeline itself that the flagship Read-LoRA is
`qcmem_distill_qwen_j12_r32_4k/final`.

### A02 overall status

READY_AFTER_SMALL_FIX: Read-LoRA identity is now confirmed. The remaining prep tasks:
1. Decide which Write-LoRA checkpoint to use (trivial eval of distill loss, CPU)
2. Choose a zwfy6 node for the phase-1 sweep (no cross-disk needed: both LoRAs + base model
   on zwfy6 once write-LoRA checkpoint is decided; base model at
   `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b`)

---

## Documentation bug found

**`A02/STATUS.json` calls lora_best_ref "512 bytes" — it is actually an empty directory
(0 bytes, 2 entries: . and ..), not a 512-byte file.** The A02 scout_21/lane4 also correctly
noted it is empty. The "512 bytes" figure is a `du` artifact (filesystem block overhead for
an empty directory inode), not file content. This is not dangerous but the wording in STATUS.json
should say "empty directory, 0 bytes of content" rather than "512 bytes".
