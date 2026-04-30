# NIAH Accuracy = 0.000 Root Cause Diagnosis
**Date:** 2026-04-28  
**Run:** swa_stage1_v3 / swa_stage1_v3_resumed  
**Script:** scripts/train_mem_space_pg19.py  

---

## TL;DR

**Root cause: off-by-one error in the NIAH accuracy metric (lines 672–681 of `train_mem_space_pg19.py`).** The code reads logits at the *label positions* instead of at *label positions minus 1*, which means it evaluates predictions for the tokens **after** the answer tokens — guaranteed to produce the wrong string, giving `niah_acc = 0.000` forever regardless of model quality or training progress.

The training LOSS is computed **correctly** (HuggingFace handles the causal shift internally). The model may actually be learning to retrieve the needle, but we cannot observe it through this broken metric.

---

## 1. NIAH accuracy code path (verbatim)

```python
# train_mem_space_pg19.py lines 671–681
answer_mask = last_lbl[0] != -100          # [seq_len] bool
if answer_mask.any():
    pred_tokens = out.logits[0].argmax(dim=-1)   # [seq_len]
    pred_ans = pred_tokens[answer_mask][:5]       # BUG: wrong positions
    pred_str = tokenizer.decode(pred_ans.tolist(), skip_special_tokens=True)
    expected_code = batch.get("code", "")
    if isinstance(expected_code, (list, tuple)):
        expected_code = expected_code[0]
    if expected_code and expected_code in pred_str:
        niah_correct += 1
    niah_total += 1
```

---

## 2. Why this is always zero

### 2.1 Causal LM logits are next-token predictions

`out.logits` from `LlamaForCausalLM.forward` has shape `[B, T, V]`.  
`out.logits[0, i, :]` is the model's probability distribution over **position i+1** (the next token given all tokens 0..i).  It does NOT predict the token at position `i` — it predicts what comes **after** `i`.

### 2.2 Label positions vs logit positions

From `NIAHIterableDataset._make_niah_sample` (niah_dataset.py, lines 269–278):

```python
question_start_idx = N_gap * chunk_size        # e.g. chunk at index N_gap
answer_start_in_chunk = len(question_ids)       # = 9 tokens for "The secret code for agent XXXXXX is "
answer_global_start = question_start_idx + answer_start_in_chunk
# answer_ids = tokenizer.encode(code) — for '12345' → [4513, 1774] (2 tokens)
for j in range(n_ans):
    labels[answer_global_start + j] = input_ids[answer_global_start + j]
```

After `.split(seq_len)`, the **last chunk** contains the question.  
Within `last_lbl[0]` (shape `[seq_len]`), the non-(-100) positions are:
- `answer_start_in_chunk + 0` = **9** (first answer digit token, e.g. 4513)
- `answer_start_in_chunk + 1` = **10** (second answer digit token, e.g. 1774)

So `answer_mask = (last_lbl[0] != -100)` is `True` at indices **9, 10**.

### 2.3 The bug

```python
pred_tokens = out.logits[0].argmax(dim=-1)  # [seq_len]
pred_ans = pred_tokens[answer_mask][:5]
```

This reads `pred_tokens[[9, 10]]` = `[logits[0,9].argmax(), logits[0,10].argmax()]`.

But:
- `logits[0, 9]` = prediction for **position 10** = the 2nd answer digit (not the 1st)
- `logits[0, 10]` = prediction for **position 11** = the padding token (not the 2nd answer digit)

**We wanted:**
- `logits[0, 8]` = prediction for position 9 = 1st answer digit  
- `logits[0, 9]` = prediction for position 10 = 2nd answer digit

The code reads one position too late. The predicted sequence is `[predicted_2nd_digit, predicted_padding]`, not `[predicted_1st_digit, predicted_2nd_digit]`.

### 2.4 Why this produces exactly 0.000

For a 5-digit code like `12345` (tokenized to 2 tokens `[4513, 1774]`):

1. `pred_ans` = decoded representations of positions **after** the code  
2. Concretely: the second digit token prediction + the EOS/padding prediction  
3. `pred_str` ≈ `"174"` or `""` (partial digit string, no full code)  
4. `expected_code in pred_str` → `"12345" in "174"` → **always False**  
5. `niah_correct` stays 0, `niah_total` increments  
6. `niah_acc = 0 / niah_total = 0.000` — **every single step**

Confirmed by tokenizer output:
```
code='12345' -> token_ids=[4513, 1774] (n=2)
question_ids: [791, 6367, 2082, 369, 8479, 19921, 13963, 374, 220] (n=9)
```
Answer at positions 9, 10; code reads logits at positions 9, 10; gets predictions for positions 10, 11.

---

## 3. What IS working correctly

### 3.1 NIAH loss (correct)

HuggingFace `LlamaForCausalLM.forward` handles the causal shift **internally**:
```python
shift_logits = logits[..., :-1, :].contiguous()   # positions 0..T-2
shift_labels = labels[..., 1:].contiguous()        # positions 1..T-1 (with -100 mask)
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

For our labels (non-(-100) at positions 9, 10):
- `shift_labels` non-(-100) at positions **8, 9** (i.e., labels[1:])
- `shift_logits[8]` = `logits[8]` = prediction for position 9 (first answer digit) ✅
- `shift_logits[9]` = `logits[9]` = prediction for position 10 (second answer digit) ✅

**The training loss correctly supervises the model to predict answer tokens.**

### 3.2 NIAH data pipeline (correct)

- `NIAHIterableDataset` with `niah_mix_fraction=1.0` always yields NIAH samples ✅
- `niah_collate_fn` correctly propagates `is_niah=True`, `code='12345'` string ✅
- The training loop enters the NIAH branch when `batch.get('is_niah', False) == True` ✅
- `answer_mask.any()` is `True` (label positions 9, 10 are non-(-100)) ✅
- `niah_total > 0` — with `niah_mix_fraction=0.10` and 600 steps, ~60 NIAH steps ✅

### 3.3 Memory bank writeback (functional after step 500)

- Warmup gate opens at step 500: `beta = sigmoid(gate_param) * (step/warmup) * gate_max`  
- After step 500: `beta = sigmoid(0) * 1.0 * 0.3 = 0.15` (non-zero writeback)  
- Haystack chunks processed with `no_grad` DO accumulate in the bank via writeback  
- The bank carries the accumulated hidden-state representation to the question chunk ✅

---

## 4. Root cause summary

| Issue | Location | Impact |
|---|---|---|
| **Off-by-one: `pred_tokens[answer_mask]` reads logits at label positions instead of label_positions-1** | `train_mem_space_pg19.py` lines 673-675 | `niah_acc = 0.000` always |
| _(secondary)_ No separate logging of NIAH loss vs pg19 loss | lines 707-715 | Cannot tell from logs if NIAH loss is decreasing |

---

## 5. Fix

Replace the accuracy computation block (lines 671–682) with:

```python
# NIAH accuracy: shift logit read by -1 to align with causal LM convention
# logits[i] predicts token i+1, so to predict answer at position p, read logits[p-1]
answer_positions = (last_lbl[0] != -100).nonzero(as_tuple=True)[0]
if len(answer_positions) > 0:
    pred_tokens = out.logits[0].argmax(dim=-1)  # [seq_len]
    ans_start = answer_positions[0].item()
    n_ans_toks = len(answer_positions)
    # Read logits[ans_start-1 : ans_start-1+n_ans_toks] — predictions FOR the answer tokens
    pred_slice_start = max(0, ans_start - 1)
    pred_ans = pred_tokens[pred_slice_start : pred_slice_start + 5]
    pred_str = tokenizer.decode(pred_ans.tolist(), skip_special_tokens=True)
    expected_code = batch.get("code", "")
    if isinstance(expected_code, (list, tuple)):
        expected_code = expected_code[0]
    if expected_code and expected_code in pred_str:
        niah_correct += 1
    niah_total += 1
```

**Key change:** `pred_slice_start = ans_start - 1` instead of directly indexing `answer_mask`.

---

## 6. Implications for current run

1. **The swa_stage1_v3_resumed training is NOT broken** — the loss signal for NIAH is correct; the model IS receiving gradient to learn needle retrieval.
2. **We have been blind to NIAH learning progress** — the metric has been broken from step 1.
3. **Fix the metric, re-run a diagnostic forward** — after applying the fix, even a short 100-step run will tell us whether the model has learned anything from the existing 13000+ steps of NIAH training.
4. **No need to restart training** — if the current checkpoint has learned NIAH (loss decreasing), we only need to fix the evaluation code. If NIAH loss was not decreasing (model converged to predict padding), we may need to adjust the training setup.

---

## 7. Recommended actions

1. **Immediate (coder):** Apply the fix to `train_mem_space_pg19.py` lines 671–682.  
   Also add separate `niah_loss` logging to distinguish NIAH CE from pg19 LM CE.

2. **Diagnostic (trainer):** Kill current run, apply fix, re-launch a short diagnostic run from the step-10000 checkpoint with `max_steps=200` to see if the corrected `niah_acc` is non-zero.

3. **If niah_acc still 0 after fix:** The model has not learned retrieval despite correct loss — investigate whether the memory bank is retaining needle information across the streaming chunks (may need to log slot similarity / writeback magnitude).
