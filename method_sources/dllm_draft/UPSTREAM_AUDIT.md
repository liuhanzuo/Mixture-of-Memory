# Upstream Dream-Coder / DreamOn Audit

Audit date: 2026-07-22

## Pinned source revisions

- DreamOn:
  - repository: `DreamLM/DreamOn`
  - commit: `8a0a54918412eda9402a327646f7f067f7160ec8`
  - commit date: 2026-02-03
- Dream-Coder:
  - repository: `DreamLM/Dream-Coder`
  - commit: `79d43878c55ba4e7474d5e0b6057d110b43acfcd`
  - commit date: 2025-11-17

The GPU server cannot reach public GitHub/Hugging Face directly. The repositories
were cloned in the local workspace and copied to the remote `vendor/` directory.

## Model revisions selected for reproducibility

- `Dream-org/Dream-Coder-v0-Instruct-7B`
  - revision: `5d9e88c723af9045f362748b5284bdf43d9c501e`
  - repository payload: approximately 14.19 GiB
- `Dream-org/Dream-Coder-v0-Base-7B`
  - revision: `2346ccd3be517d0d314152b988a3b9bafa7d6d63`
  - repository payload: approximately 14.19 GiB
- `Dream-org/DreamOn-v0-7B`
  - revision: `57e0e4115c97899e3e859b00eae26f54f9cf3872`
  - repository payload: approximately 14.19 GiB

## Important findings

### 1. Current model configs advertise 32,768 positions

At the pinned Hugging Face revisions, both Dream-Coder Instruct and DreamOn
configs use:

```json
"max_position_embeddings": 32768
```

The DreamOn repository README still says the context length is 2,048. The
README is therefore stale relative to the current model artifact. We will use
the pinned model config as the implementation source of truth and separately
test actual memory/quality at longer lengths.

This does not imply that the released training/evaluation recipe was validated
at 32K. DreamOn training uses `data.max_length=1024`; Dream-Coder SFT uses
`data.max_length=4000`.

### 2. Released DreamOn uses EOS as the delete action

The released DreamOn vocabulary adds `<|expand|>` but does not add a dedicated
delete token:

```text
mask   = 151666
expand = 151667
delete semantic action = EOS 151643
```

Evidence:

- `DreamOn-v0-7B/added_tokens.json` has `<|expand|>` but no delete token.
- `generation_config.json` sets `delete_token_id` to the EOS ID.
- `sft_expand_dataset.py` appends EOS targets as synthetic deletions.
- `eval/generator.py` removes EOS predictions made at masked middle positions.

In Scaffold-Coder documents, `[delete]` should be read as an abstract edit
action. Backend options are:

1. `dreamon-eos-delete`: map `[delete]` semantics to EOS for maximum checkpoint
   compatibility;
2. `dedicated-delete`: add a new atomic token to remove EOS overloading.

The upstream reproduction uses option 1. The structural model can ablate or
adopt option 2 after the baseline is reproduced.

### 3. Exact released merge behavior

For each infilling middle span:

1. sample one scalar `sampling_ratio ~ Uniform(0,1)`;
2. independently mask each input token with that probability;
3. always mask synthetic EOS/delete targets;
4. for adjacent masks, optionally:
   - make the first target `<|expand|>`;
   - hide the second position through `attention_mask=0`;
5. mix the configured scheduler with a static merge probability.

Released `dynamic_inverse` is:

```text
merge_prob * (1 - num_masked / region_length)
```

It is not literally `1 / num_masked`.

### 4. Released training command differs from some prose summaries

`vendor/DreamOn/run_dreamon.sh` currently uses:

- 8 GPUs with `torchrun`;
- FSDP FULL_SHARD;
- global batch size 128;
- micro batch size 8/GPU;
- maximum sequence length 1,024;
- learning rate `1e-5`;
- five epochs;
- Dream-Coder Base initialization;
- gradient checkpointing;
- FlashAttention 2;
- `ulysses_sequence_parallel_size=1`.

The codebase is based on verl `v0.3.0.post1`.

### 5. Released run selects a nonstandard time weight

The trainer supports:

- `original`: `1/t`;
- `linear`: `1-t`;
- otherwise: unit weight.

The released DreamOn launch script sets:

```text
diffusion.time_reweighting=linear
```

Therefore the exact released run does not use the trainer’s `1/t` branch. Our
baseline must reproduce the released command before changing the weighting.

### 6. Deletion downweighting

When `weight_eos=true`, ordinary masked-token losses are summed. Synthetic
EOS/delete losses are averaged over the number of EOS targets and then combined
with the ordinary sum. This gives the delete group approximately one ordinary
token’s aggregate weight.

### 7. Dynamic expansion is effectively batch-size one

The released generator rewrites tensors by insertion/deletion and explicitly
contains batch-size-one assumptions for relevant paths. Scaffold-Coder needs a
tree-aware single-example sampler first; length-bucketed batching is a later
optimization.

### 8. Dream-Coder has reserved embedding rows

The pinned Dream-Coder tokenizer currently has length 151,667 and ends at the
mask token ID 151,666, while the model config has:

```text
vocab_size = 152064
```

This leaves 397 in-range embedding/output rows available. Proposed tokens can be
assigned without resizing the 7B embedding matrices, provided allocation order
is fixed:

1. `<|expand|>` -> 151667, matching DreamOn;
2. optional dedicated delete token;
3. structural labels and rule-only holes.

The unused rows still require deliberate input-embedding and LM-head
initialization. Calling `resize_token_embeddings(len(tokenizer))` would shrink
the matrices and is therefore forbidden.

A full CPU checkpoint load confirmed:

- model parameters: 7,615,616,512;
- input/output matrix shapes: `[152064, 3584]`;
- input embedding rows 151667 onward are exactly zero in the Instruct
  checkpoint;
- corresponding LM-head rows are nonzero but appear untrained/reserved.

Both matrices must therefore be initialized for newly assigned tokens. Merely
adding tokenizer entries would leave structural tokens with zero input
embeddings.

### 9. Segmented tokenization is text-stable but ID-different

The tokenizer audit found that all tested rule/content segmentations decode
exactly to the intended Python text, but their token-ID sequences differ from
whole-text canonical tokenization. Examples include function headers,
indentation, nested `if`, and newline+indent boundaries.

Therefore training and inference must use the same segmentation policy. The
stronger claim `segment_ids == encode(full_rendered_text)` is false for ordinary
Dream tokenizer BPE boundaries.

## Environment implications

Official quickstarts request:

```text
Python compatible with torch 2.5.1
torch==2.5.1
transformers==4.46.2
```

The server default environment is Python 3.14 / torch 2.13.0 / transformers
5.5.4, so it is not used for reproduction. A separate shared Python 3.11 venv
is being installed at:

```text
/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft/.venv_dream
```

The full verl/FlashAttention training environment will be installed only after
the minimal inference smoke works.
