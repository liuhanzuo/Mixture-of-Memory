# A03 Arm 3 ("pruned + CPT") — Executable Spec
# Prepared: 2026-08-09 (read-only scout; no GPU job launched, no file edited)

---

## 0. Verdict: Option (a) — Arm 3 is a VALID, DISTINCT arm. CPT = more Dolmino steps beyond heal.

**Scientific justification from the PROPOSAL + KILL condition text:**

The PROPOSAL distinguishes two things:
- "heal" = the minimal repair pass to restore language-model fluency after pruning
  (the model needed 200k steps just to stop being incoherent)
- "CPT" = _continued_ pretraining BEYOND fluency, intended to test whether
  parametric factual knowledge (PopQA/TriviaQA/NQ-open entities) is recovered

The kill condition reads: **"CPT 能以更低总成本恢复全部目标能力"** — the comparison is
CPT (additional tokens) vs residual memory (inference-time overhead). This presupposes
that CPT has a measurable positive effect BEYOND what heal already provides. The valid
scientific question is: does Arm 2 (200k heal steps, TriviaQA EM = 9.59%) improve
further if we train for another N steps — and at what cost?

The fact that Arm 2 already used Dolmino for 200k steps does NOT collapse Arm 3 into
Arm 2. The learning curve did NOT plateau at step 200k (the 7B curves in the paper show
continued improvement; we do not have a 1B EM vs step curve yet). Arm 3 = **resume Arm 2
for an additional 20k steps on the same Dolmino data** and re-eval the four knowledge axes.

If the four-axis numbers do not move after 20k more steps, that IS a valid finding: it
means 200k Dolmino steps already saturated parametric knowledge recovery for this model
depth, and more CPT is futile — which is informative for the CPT vs memory cost comparison.

**Why NOT option (b) (different corpus):**
The PROPOSAL does not specify a corpus for CPT (it says "NTP/CPT" generically). A
Wikipedia-subset CPT would test domain-specific recovery but is NOT on disk, requires
~20 GB proxy download, and adds a confound (data distribution) on top of the CPT effect.
The scientific question ("how much recovery do we get?") is cleanly answered by Dolmino
continuation first. Wikipedia is a follow-up if Dolmino hits a ceiling.

**Why NOT option (c) (drop Arm 3):**
The kill condition for Arm 3 is testable and the result is informative either way.
The arm costs 8-9 GPU-hours on one H20 node. Dropping it would leave the CPT cost
comparison completely untested.

---

## 1. Arm 2 heal: verified configuration (evidence for Arm 3 starting point)

Source: `outputs/olmo2_probe2_1B_keep7fresh2_16card/arch_meta.json` on zwfy6 (.73/.82/.104):

```json
{
  "arm": "healing_front7+fresh2",
  "keep_front_layers": 7,
  "n_fresh_layers": 2,
  "num_hidden_layers": 9,
  "seq_len": 2048,
  "lr_fresh": 0.0001,
  "lr_inherited": 2e-05,
  "base_model_path": "/apdcephfs_zwfy6/.../models/OLMo-2-0425-1B"
}
```

Heal corpus: `data/dolmino_now15b.npy` (126.9 GB, verified on zwfy6).
Heal steps: 200k steps, eff_bs=128, seq_len=2048 → 26.2B tokens.
Final checkpoint: `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt` (11.3 GB,
zwfy6 only, verified on .73 and .82).

Known result (from GATE_NQOPEN_VERDICT.md / STATUS.json):
- TriviaQA EM = 0.0959 (9.59% vs intact 40.69%)
- NQ-open EM = 0.0285 (2.85% vs intact 10.25%)
- PopQA EM = 0.0394 (3.94% vs intact 15.50%)
- MMLU content_norm = 0.3244 (+3.99pp above null 0.2845)

---

## 2. Arm 3: what "CPT" means precisely

**Arm 3 = resume step200000.pt for 20k additional steps on the SAME Dolmino corpus,
uniform LR 2e-5 (no differential LR), output to a new output_dir.**

Scientific label: "+20k Dolmino steps from healed init"  
Token budget: 20k × 2048 × 8 = 327M additional tokens (1.25% extra over the 26.2B heal budget)  
Expected direction: either (i) TriviaQA/NQ-open EM continues increasing (→ CPT is still
recovering knowledge) or (ii) flat (→ Dolmino has saturated factual recovery at this
depth, motivating memory as the more efficient interface).

**Why uniform 2e-5 (not differential LR):**
The "fresh" layers (layer indices 7,8) are now fully trained at 200k steps; there is no
architecturally "new" component left. Using `--lr 2e-5 --lr_inherited 2e-5` makes all
parameters update at the same low rate, treating this as a standard continuation fine-tune.
This is cheaper (cosine re-scale, no warm-up needed for very short CPT), and the optimizer
state is faithfully resumed (4-group → 4-group, no remap needed — layer indices 7,8 are
"fresh", layers 0-6 are "inherited", `_classify_param` works correctly in HEAD because it
strips `module.` prefix).

---

## 3. Exact executable command (8×H20, zwfy6)

**Node**: `.73` (28.85.35.73, zwfy6) — Arm 2 checkpoint AND Dolmino data are on zwfy6.
**Python**: `/opt/conda/envs/torch-base/bin/python`
**Project root**: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`

### Pre-flight checks (MAIN must verify before launching)

```bash
# On .73:
ls -la /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt
# Expected: 12181310078 bytes (~11.3 GB)

ls -la /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/dolmino_now15b.npy
# Expected: 126907244672 bytes (~126.9 GB)

ls -la /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/../models/OLMo-2-0425-1B/
# Must exist (needed for config load on resume; transplant is SKIPPED on resume path)
```

### Launch command

```bash
setsid nohup \
  /opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node 8 \
  /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/train_olmo2_arch_probe2.py \
    --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B \
    --resume_from outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt \
    --keep_front_layers 7 \
    --n_fresh_layers 2 \
    --data_path data/dolmino_now15b.npy \
    --output_dir outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k \
    --max_steps 220000 \
    --lr 2e-5 \
    --min_lr 2e-6 \
    --lr_inherited 2e-5 \
    --min_lr_inherited 2e-6 \
    --seq_len 2048 \
    --batch_size 8 \
    --grad_accumulation_steps 1 \
    --warmup_steps 0 \
    --save_every 5000 \
    --gradient_checkpointing 1 \
    --seed 42 \
> /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/a03_arm3_cpt20k.log 2>&1 &
echo "PID: $!"
```

**Key flags:**
- `--max_steps 220000`: resume is at step 200000 → trains 20k more steps
- `--warmup_steps 0`: no warmup needed (resuming from trained weights + optimizer state)
- `--lr 2e-5 / --lr_inherited 2e-5`: uniform continuation LR (same as inherited rate)
- `--batch_size 8`: matches per-GPU batch of original 16-card run (same memory per card)

**Optimizer resume note:**
The step200000.pt was saved with 4 param groups (fresh_decay/fresh_nodecay/inh_decay/
inh_nodecay). HEAD `train_olmo2_arch_probe2.py` also builds 4 groups with correct
`_classify_param` (strips `module.` prefix; confirmed in HEAD at line 438-450). The
4→4 group resume path (`optimizer.load_state_dict(ckpt_optim)`) will succeed without
the compatibility remap; Adam moments are preserved.

**Memory estimate**: 1B model in fp32 = ~4 GB params + ~8 GB optimizer state = ~12 GB
static per card. With bs=8, seq_len=2048 and gradient_checkpointing: maxmem ≈ 65.8 GB
(measured from the 8-card 1-node log at the same bs=8). H20 has 97.8 GB — well within.

---

## 4. Wall time estimate

From `logs/olmo2_1B_keep7fresh2_16card_node0.log` (16-card, bs=8/GPU): **1.49s/step**
From `logs/olmo2_1B_keep7fresh2_1node.log` (8-card, bs=16/GPU): **2.02s/step**

The CPT run uses bs=8/GPU on 8 cards (same per-GPU batch as the 16-card run but only
8 cards). The dominant cost is the forward+backward on 8 GPUs. The 16-card gradient
all-reduce overhead is absent, so the per-step time should be ≤1.49s.

**Conservative estimate: 1.6s/step × 20k steps = 32,000s ≈ 9 hours on 8×H20.**

Breakpoints: step5k/10k/15k/20k auto-saved at save_every=5000. If step5k (≈2h) shows
no movement in the eval metrics → may cancel early and declare "Dolmino saturated".

---

## 5. Eval command after training (four certified axes)

Use the existing `_run_a03_axes_floor_82.sh`-style driver but add the new checkpoint.
Since `_run_a03_axes_floor_82.sh` hard-codes the three existing arms, use the driver as
a template and invoke `eval_olmo2_closedbook_qa.py` directly:

```bash
# On .73 (zwfy6), after training completes:
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
BASE=$W/../models/OLMo-2-0425-1B
CKPT=$W/outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k/step220000.pt

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy

# Run all 4 closed-book tasks in one shot
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY $W/scripts/eval_olmo2_closedbook_qa.py \
    --base_model "$BASE" \
    --ckpt "$CKPT" \
    --tasks popqa triviaqa nq_open mmlu_content \
    --num_shards 8 --shard_index $g \
    --batch_size 48 --add_bos 0 --max_new_tokens 32 \
    --output_name A03_1B_keep7_cpt20k \
    > $W/logs/a03_arm3_eval_shard${g}.log 2>&1 &
done
wait

# Merge and score
$PY $W/scripts/eval_olmo2_closedbook_qa.py \
  --merge --output_name A03_1B_keep7_cpt20k
```

Then run `analyze_1b_knowledge_floor.py` (after applying the 1-line nq_open patch from
`A03_6ARM_DESIGN.md §3`) to get BH-corrected floor-calibrated residuals for all 4 axes.

All paths on disk **zwfy6** (`.73`/`.82`/`.104`). Do NOT run on wzc1 nodes (checkpoint
and datasets are on zwfy6 only).

---

## 6. Summary table

| Item | Value |
|------|-------|
| Verdict | Option **(a)** — Arm 3 is valid; CPT = more Dolmino steps beyond heal |
| Starting point | zwfy6: `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt` (11.3 GB) |
| Data | zwfy6: `data/dolmino_now15b.npy` (126.9 GB) |
| Steps | 20k (global steps 200k → 220k) |
| LR | uniform 2e-5 (no warmup) |
| Batch | bs=8, eff_bs=64, seq_len=2048 |
| Output | zwfy6: `outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k/step220000.pt` |
| Wall time | ~9 hours on 8×H20 (conservative) |
| Node | `.73` (preferred; zwfy6, currently free) |
| Eval | `eval_olmo2_closedbook_qa.py` popqa+triviaqa+nq_open+mmlu, 4-axis floor analysis |
| Kill early? | Check step5k/10k EM: if flat vs Arm 2 → cancel at 10k, declare saturation |
