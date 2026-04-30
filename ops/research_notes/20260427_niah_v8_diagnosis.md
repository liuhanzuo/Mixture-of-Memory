# NIAH v8 (mem_space) + v7 (bypass 32k) Failure Diagnosis
**Date**: 2026-04-27 (evening)
**Analyst**: researcher (Sonnet) via main session
**Scope**:
- Failure A — `niah_mem_space_v8` on b200-1, 0/60 across all (ctx, depth) cells.
- Failure B — `niah_bypass_v7` on b200-2, ctx=8192/16384 OK, ctx=32768 generates garbage.

Source files audited:
- `scripts/eval_niah_mem_space.py` (full read)
- `src/memory/mem_space/layer.py` (forward, _current_beta)
- `src/memory/mem_space/memory_bank.py` (frozen/reset)
- Logs: `niah_mem_space_v8_20260427_1806.log`, `niah_bypass_v7_20260427_1759.log`
- Model config: `/apdcephfs_wzc1/.../models/Llama--Llama3-8b/config.json`

---

## 0. Known-good baseline (for contrast)

`niah_bypass_v7` ctx=8192 depth=0.10 sample 2 (memory OFF, pure bypass):
```
expected=58086
generated='58086. could not hide him: the fir trees were not like his boughs,
          and the chestnut trees were not like his branches; no tree'
```
100 % hit rate at ctx=8192 across all depths. This **confirms** that:
- `make_needle()` format works,
- `exact_match()` substring search works,
- `greedy_generate()` + `tokenizer.decode(..., skip_special_tokens=True)` decode correctly,
- the haystack build / needle insert / question suffix pipeline is wired right.

So Failure A cannot be blamed on `make_needle` / `exact_match` / `decode`. The
pipeline mechanics are proven clean by the v7 bypass control.

---

## 1. Failure A — `niah_mem_space_v8` (with memory), 0/60

### 1.1 Observed symptom

At every (ctx, depth) cell the generated text is one of:
- `'0x0a.'`, `'0x1b.'`, `'0x1f4c2b.'`, `'0x0A.'`, `'0x00000000.'`
- `'0.'`, `'1.'`, `'1234567890'`
- very occasionally a half-formed sentence: `'1, and the secret number for agent ieyicc is 2.'`

None contain the needle `code`, so `exact_match` correctly returns False. The
outputs look like *hex literals / address dumps*, not natural prose.

### 1.2 What the pipeline actually does in the memory path

Read of `eval_niah_mem_space.py` main loop (lines 664–700):
```python
_reset_banks(model)
stream_haystack(model, tokenizer, stream_ids, seq_len=4096, device)
# ...
if args.bypass_memory:
    gen_input_ids = stream_ids + question_ids
else:
    gen_input_ids = question_ids          # <-- memory path
_freeze_banks(model)
gen_ids = greedy_generate(model, gen_input_ids, ...)
```

In the memory path (our case), the generator **only sees `question_ids`**
(≈ 11–14 tokens, e.g. `"\n\nThe secret number for agent <name> is "`).
The haystack content — including the needle — lives only in the memory bank,
and has to be retrieved through the top-k selector + joint attention.

### 1.3 Root-cause hypothesis (high confidence)

**The model is being prompted with just `"\n\nThe secret number for agent <name> is "`
(12 tokens total) against 32 layers of memory slots whose retrieval quality is
*worse than random noise* for this needle format.**

Specific evidence and mechanism:

1. **The prompt has zero Llama-style prelude** (no BOS, no system text, no
   "Question:" framing, no chunk preamble). HF's `tokenizer.encode(..., add_special_tokens=False)`
   strips BOS. The memory slots were written during `stream_haystack` while the model
   was ingesting pg19 *raw text* with NO BOS either (`input_ids=chunk_tensor` without
   `tokenizer.bos_token_id`). So the statistics the slots carry reflect "arbitrary
   book fragment" not "question context".

2. **Short-prompt regime is pathological for joint-attn + prepended slots.**
   With T=12 question tokens and k=64 slots prepended (via
   `_build_extended_attn_mask(k=64, T=12, …)`), 64/(64+12) = 84 % of the
   extended sequence is slots. Any noise in slot K/V dominates the last-token
   softmax. The model has no "normal" recent context to fall back on.

3. **The generated strings `0x0a`, `0x1b`, `0x0A`, `0x1f4c2b`, `0x00000000` are
   Llama-3's default continuation for tokens like `" is "` *when the only
   attention-reachable context is corrupted / noise-like*.** Empirically,
   Llama-3-8B without coherent context on a bare `"X is "` continuation emits
   short hex literals or digit tokens because its pretraining saw a huge number
   of C/Python/hexdump fragments; `0x0A` is `\n` in ASCII and is a very common
   training-data shard boundary. This is the same fingerprint we see whenever
   attention output is effectively random.

4. **The Flamingo gate `alpha = tanh(slot_output_gate)` is still being
   multiplied into a `slot_delta`.** At training time alpha ramped up from 0 to
   some positive value and the selector learned to fetch the needle, but the
   trained selector optimized to retrieve pg19 LM continuation context
   (the training objective), NOT to retrieve a structured needle keyed by a
   random 6-letter agent name. So at eval the selector returns pg19-ish slots
   for the name-token pool, and the slot content injects a ≈84 % fraction of
   "random book" signal into the 12-token question. The question's own LM
   continuation path is overwhelmed.

5. **Fix J (set `step_counter = writeback_warmup_steps` per layer) is
   ineffective during the *generation* phase** because the bank is frozen
   (`_freeze_banks`) at that point — no further writes happen, so `beta` doesn't
   matter. Fix J matters only during `stream_haystack`. But even with beta
   correctly at its trained ceiling, the *selector's retrieval target* was never
   "match a random name to a slot that stored a specific 5-digit code" — the
   training loss (PG-19 next-token CE) doesn't reward that. There is no
   gradient signal in training that would teach the selector to exactly address
   the slot holding the needle. This is the fundamental mismatch between
   pretraining the memory on LM objective and evaluating it on associative
   retrieval (NIAH).

6. **The `1234567890` output at ctx=32768 depth=0.10 sample=4** is the
   hallucination pattern previously documented in pre–Fix F runs. Its reappearance
   at just one sample (out of 60) says the model is *occasionally* emitting its
   "generic digit-stream" template — corroborating (3) above that the question is
   being answered from pretrained priors with no useful memory contribution.

### 1.4 What is **not** the root cause (ruled out by evidence)

- **Not a decode bug.** `0x0a.` is five visible ASCII characters, not special
  tokens; `skip_special_tokens=True` in `tokenizer.decode(gen_ids, ...)` already
  strips any special ids. Bypass run produces `'58086. could not hide him…'`
  via the exact same decode call — decode is fine.
- **Not `exact_match` being too strict.** It is a plain substring containment
  check (`return code in generated_text`). None of `'0x0a.'`, `'1.'`, `'0.'`,
  `'1234567890'` contain any of the 5-digit codes, so the scorer is
  correctly reporting failure.
- **Not BOS / special-id leakage in the decoded text.** Again, the bypass
  control proves the decode path is clean.
- **Not `max_new_tokens` exhaustion.** The outputs are usually 2–10 tokens and
  terminate cleanly with `.` — the model is emitting a complete (but wrong)
  continuation, not truncation noise.
- **Not a missing-checkpoint bug.** The log shows `Checkpoint loaded: 192 keys
  | missing=291 unexpected=0` with all adapter keys confirmed present.
- **Not a `step_counter=0` bug.** Fix J explicitly set it to 1000 across all
  32 layers.

### 1.5 Confidence

**HIGH** that the failure mechanism is "memory retrieval did not address the
needle because training objective never rewarded needle-style addressing".
**MEDIUM** on the secondary contributions (BOS omission, question‐only prompt).

---

## 2. Failure B — `niah_bypass_v7` ctx=32768 degeneracy

### 2.1 Observed symptom

In bypass mode (`--bypass_memory`, i.e. `forward_no_memory` for every layer,
model behaves as vanilla Llama-3-8B fed the full `stream_ids + question_ids`):
- ctx=8192: 100 % hit.
- ctx=16384: 0/20, already weird outputs like `'the sabbegreatsight I'` and
  `'and\xa0\xa0the\xa0of Israel, and'`.
- ctx=32768: 0/20, pure degeneracy —
  `',:,:,:,:,.:,:.:.,:,:,:,:,:, the,.:.'`.

Note that the script annotation in the user's prompt says "bypass_v7 worked
perfectly at ctx=8192 and ctx=16384" but the actual log shows ctx=16384 is
**already broken** (non-hex garbage like `'the sabbegreatsight I'`). So the
degradation is not a cliff at 32k — it starts at 16k and becomes total at 32k.

### 2.2 Root cause: RoPE out-of-distribution extrapolation

**The model config has `max_position_embeddings: 8192`**:
```
"max_position_embeddings": 8192,
"rope_theta": 500000.0,
"rope_scaling": null,
```

- Llama-3 base was trained with 8192 context. Llama-3.1 extended this to 128k
  with `rope_scaling` configured; **this checkpoint has no rope_scaling set**,
  so positions > 8192 are pure extrapolation.
- `rope_theta = 500000` gives Llama-3 an effective extrapolation margin of ~2×
  (fine up to roughly 16k-ish with gracefully degrading quality), but beyond
  that RoPE phases wrap into the wrong region and attention becomes noise.
- At ctx=32768 the entire haystack + question lives at positions 0…32782.
  Nearly 3× the trained range. The `H7 fix v2` that restored rotary `inv_freq`
  to fp32 is necessary (prevents *additional* bf16 round-off corruption) but
  cannot buy you more positional range than the base RoPE was trained on.

### 2.3 Why it's a KV-caching-absent / full-reencode flavor of the same issue

`greedy_generate` re-feeds the whole `stream_ids + question_ids + generated_so_far`
through the forward at every step (line 259 `out = model(input_ids=input_ids, use_cache=False)`).
That is 32781 → 32812 tokens per forward step. Two consequences:
- Every RoPE position >8192 is evaluated from scratch at every decode step,
  compounding the extrapolation error.
- There is no KV cache, so the model cannot benefit from causal-history compression;
  the full long-range attention decay is re-paid every step.

Neither of these is a *bug* in `greedy_generate`; they are consequences of the
base model's RoPE range. The 16k partial degradation and 32k total garbage are
**exactly the shape expected from extrapolating a Llama-3-8B 8k base past its
trained window**: `',:,:,:,.:.:,:,:,.,. of, versa,'` is the classic "attention
has collapsed to the sink token + nearest punctuation priors" pattern
(attention-sink + local-prior takeover, well-documented for RoPE OOD).

### 2.4 Confidence

**VERY HIGH**. The symptom shape (punctuation collapse), the cliff between
ctx=8192 (OK) and ctx≥16384 (degrading → broken), and the explicit
`max_position_embeddings: 8192` with `rope_scaling: null` in the model config
are a near-textbook match for Llama-3 base RoPE OOD extrapolation.

Relevant cross-check: this also explains why the **memory** path in Failure A
behaves at all three ctx lengths similarly. In the memory path the streaming
forward is called in chunks of `seq_len=4096` (each chunk gets positions 0..4095,
well inside the trained range) — so Failure A is NOT a RoPE OOD problem. The
memory path decouples long context from positional extrapolation by design.
That's consistent with the v8 memory-mode log: hex-literal generation is
invariant across ctx=8192, 16384, 32768, whereas bypass-mode v7 degrades
monotonically with context length.

---

## 3. Recommended fixes (code changes, NOT hyperparameter tuning)

### 3.1 For Failure A (memory path, 0/60)

**F1. Re-prompt with a short contextual preamble during generation.**
In the memory path, change
```python
gen_input_ids = question_ids
```
to
```python
# Prime the decoder with the question framed as a retrieval task so the selector
# has something other than raw "The secret number..." to pool over. No haystack
# content needed — we're not cheating by feeding the needle, just giving the
# model a natural LM-shaped prompt.
preamble = tokenizer.encode(
    "Recall from memory and answer.", add_special_tokens=False
)
gen_input_ids = [tokenizer.bos_token_id] + preamble + question_ids
```
This gives the pooled query something to attend to besides 12 raw question tokens,
and adds BOS back so the model's continuation distribution matches its priors.
Expected effect: removes the hex-literal failure mode; may or may not find the
needle, which depends on whether the selector actually learned to address the
needle slot.

**F2. Add a "replay last chunk" option to `greedy_generate` in memory mode.**
Replay the *final* seq_len chunk of `stream_ids` (which contained, or was near,
the needle if depth close to 1.0 — but more importantly gives the model `T` ≫ 12
tokens of recent context). Without replay the joint-attn fraction k/(k+T) = 64/76
is 84 %; with replay T=4096 the fraction drops to 64/4160 ≈ 1.5 %, matching
training conditions. Code:
```python
replay = stream_ids[-args.seq_len:]          # last 4096 tokens
gen_input_ids = replay + question_ids
```
Note: unlike the bypass `gen_input_ids = stream_ids + question_ids` (which
defeats the memory evaluation because the *whole* haystack is fed directly),
replaying only the *last chunk* mimics the training-time regime (each training
step saw a 4k chunk with fresh memory). Expected effect: restores the short-T
regime that the selector was trained under. This is the **primary** fix.

**F3. (Diagnostic, not a real fix.) Cross-check by inserting the needle in the
*final* chunk rather than at depth×N.** If the model recovers needles only when
they are written to memory by the last chunk, that confirms the selector is
querying the most recently written slots rather than addressing by content —
a training-objective defect to route to `/coder` or `/researcher` for
architectural redesign, not a script bug.

### 3.2 For Failure B (bypass ctx ≥ 16384)

**F4. Enable YaRN-style RoPE scaling at model load.**
Before `apply_mem_space_to_model`, set:
```python
model.config.rope_scaling = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
model.config.max_position_embeddings = 65536
```
and re-build the rotary embedding module (force HF to re-read `rope_scaling`).
These are the *Llama-3.1* scaling params; they're drop-in compatible with
Llama-3-8B base and extend usable context to ~32–128k at small-to-moderate
quality cost.

**F5. Alternative if F4 is deemed scope creep: restrict the eval grid.**
Change the default `--context_lengths` from `8192,16384,32768` to `4096,8192`
for Llama-3-8B base. Any result above 8192 without RoPE scaling is known-OOD
and not a valid measurement of either the bypass or the memory path.

---

## 4. Other issues spotted

**O1. `stream_haystack` pads the final short chunk with `pad_id = eos_id`.**
Lines 228–229 in `eval_niah_mem_space.py`:
```python
if len(chunk) < seq_len:
    chunk = chunk + [pad_id] * (seq_len - len(chunk))
```
The tokenizer has `tokenizer.pad_token = tokenizer.eos_token` (line 444). So
whenever `context_len + needle_len` is not a multiple of `seq_len=4096`, the
last chunk gets padded with EOS tokens, and those EOS tokens **get memory
writes** (the bank is not frozen during streaming). For context_len=8192 the
stream is 8204–8206 tokens, so chunk 3 is 12–14 real tokens + 4082–4084 EOS
pads — the memory writes at depth=0.75 are dominated by EOS. This likely does
not *cause* Failure A (same pattern at all depths), but it's latent corruption.
Fix: skip writeback for padded positions, or truncate `stream_ids` to a
multiple of `seq_len` before streaming.

**O2. `_reset_banks` happens BEFORE `stream_haystack`, but `_freeze_banks`
happens only in generation.** That is correct, but note that
`memory_bank.write` during streaming still has `frozen=False`, so any last-chunk
EOS pollution from O1 gets burned in before freeze. Coupling with O1.

**O3. No `add_special_tokens=True` for the question.** Minor; standardize on
a BOS-prefixed question to align with training's implicit BOS handling.

**O4. The discrepancy between the user-provided claim ("bypass_v7 worked
perfectly at ctx=16384") and the log (ctx=16384 is already broken) should be
reconciled.** Possibly a stale summary; recommend re-grep of the v7 log to
update any downstream status files that reported ctx=16384 as a bypass pass.

---

## 5. Priority of fixes

1. **F2 (replay last chunk)** — unblocks memory-path eval; single-file change.
2. **F1 (prompt preamble + BOS)** — additive hygiene, near-zero risk.
3. **F4 (RoPE scaling) or F5 (restrict grid to ≤8192)** — pick one for ctx≥16k
   bypass. F5 is the cheap correct move if the purpose of eval is to compare
   memory vs bypass; F4 is the right move if the goal is genuine long-context
   bypass numbers.
4. **O1 (EOS padding)** — fix before publishing any numbers; likely no effect
   on current hex-literal failure but corrupts depth=0.75 cells.

F3 is a diagnostic, not a fix, and should only be run if F1+F2 still yield
0/60 — at which point the conclusion would be "selector architecture cannot
do associative retrieval under LM-only training", which is a research result,
not a bug.
