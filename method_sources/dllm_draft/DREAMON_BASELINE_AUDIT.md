# Matched DreamOn Baseline Audit

The released `SFTExpandDataset` was run on 100 canonical eval rows using the
pinned Dream-Coder Base tokenizer.

## Observed target distribution

```text
supervised targets: 4,510
delete/EOS targets: 3,171
expand targets:       239
```

Synthetic delete targets are approximately 70.3% of all supervised targets in
this sample. The released `weight_eos=true` loss is therefore essential; raw
target counts are not representative of aggregate loss mass.

## Padding behavior

Although canonical prompt+code examples average approximately 137 tokens, the
released collator pads every item to 1,024 and sets the padding attention mask
to one. Adjacent-mask merge positions are the main zeros.

Observed mean effective attention length:

```text
1,020.31 / 1,024
```

Therefore exact released DreamOn training performs nearly full 1,024-token
attention for every educational sample. Scaffold-Coder’s dynamic batch padding
is a major compute difference.

Required reporting:

1. **B3-exact** — released padding/augmentation behavior;
2. **B3-dynamic** — DreamOn augmentation with the same dynamic-padding policy
   used by Scaffold-Coder;
3. wall time and cumulative processed tokens in addition to NFE.

Without B3-dynamic, a throughput comparison would conflate structural
compression with a padding implementation difference.

## B3-dynamic implementation and statistics

`DreamOnDynamicDataset` preserves:

- line-middle FIM corruption;
- static/dynamic-inverse adjacent-mask merging;
- hidden second merge position and compressed position IDs;
- EOS/delete targets;
- up to 64 synthetic deletes;
- one reserved `<|expand|>` token.

It removes fixed 1,024 padding and uses the same dynamic batch collator as the
plain and Scaffold runs.

On 5,000 eval states:

| Metric | B3-exact released | B3-dynamic |
|---|---:|---:|
| mean effective attention length | 1,020.3 | 167.1 |
| mean stored sequence length | 1,024 | 169.8 |
| mean supervised masks | 45.1 on 100-row audit | 44.1 |
| mean expand targets | 2.39 on 100-row audit | 1.85 |
| mean delete targets | 31.71 on 100-row audit | 32.02 |

Thus dynamic padding reduces effective attention length by roughly 84% while
retaining a closely matched expand/delete target distribution.

## Expand-token tokenizer issue

For Dream-Coder Base:

```text
tokenizer length = 151,667
expand target ID = 151,667
model vocab size = 152,064
```

The released dataset writes the reserved expand ID directly but does not add
`<|expand|>` to the tokenizer. Training can still learn the output row because
the target is in range and the sampler rewrites expand immediately, but a saved
checkpoint tokenizer will not be self-contained unless the token is explicitly
added before release/evaluation.

Our exact smoke preserves the released behavior. Artifact-producing baseline
training must add and save `<|expand|>` deterministically after reproduction.

## GPU gate

Queued job:

```text
DREAMON-SFT-8GPU-SMOKE-001
```

It uses:

- the same normalized train/eval parquet as Scaffold-Coder;
- Dream-Coder Base;
- released FSDP trainer;
- released linear time weighting;
- EOS-delete weighting;
- two optimization steps and a checkpoint.

Additional queued dynamic control:

```text
DREAMON-DYNAMIC-SFT-8GPU-SMOKE-001
```
