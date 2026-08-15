# Paper B depth-ladder step200000 eval protocol

**Status**: PRE-DATA. Written 2026-08-15, before any of keep8 / keep10 / keep12 reached
step200000. **0 GPU spent on this document.** Its purpose is that when a ckpt appears, the
eval is launched from a script that already exists and has already been negative-tested,
instead of being written under time pressure with an idle node waiting.

**Artifacts**
- driver: `scripts/eval_paperb_ladder_200k.sh` (one arm per invocation, fully parameterised)
- assertions: `scripts/_ladder200k_assert.py` (CPU-only)
- dry-run + negative-test evidence: `paperB/evidence/ladder_200k_eval_dryrun.json`
- task board entries: `status/PENDING_TASKS.md` **#253** (keep12), **#254** (keep10, includes
  the cross-disk move), **#255** (keep8) — all `auto_launch: true` with the trigger condition
  stated per arm
- both scripts are on **both disks**, md5-verified identical
  (`649b602738666b4719a514e8d2b09b84`, `a5b8a6bea896eb90186174efbd120fea`)

---

## 1. The battery: `PPL + core6 + know5`

Three harnesses, run in this order, each 8-way GPU-sharded then merged. This is copied
from the two arms that already have a 200k row, so the new rows are directly comparable
to them:

| # | component | harness | tasks / data | batch size |
|---|---|---|---|---|
| 1 | held-out NTP PPL | `scripts/eval_olmo2_probe2_ppl.py` | `data/dolmino_now_val.npy` | 4 |
| 2 | core-6 downstream | `scripts/eval_olmo2_probe2_downstream.py` | see below | 8 |
| 3 | knowledge-5 | same harness, different `--tasks` | see below | 8 |

Source of the battery definition (these are the two drivers that produced the existing
200k rows):
- `scripts/_run_olmo2_eval_keep14_s200000_b200.sh` — keep14 train-all @200k
- `scripts/_run_olmo2_eval_freezefront_s200000.sh` — freeze_front @200k

### 1.1 core6 — the actual membership, with file:line

```
hellaswag, arc_challenge, arc_easy, piqa, winogrande, openbookqa
```

Two independent authorities agree:

| authority | file:line | content |
|---|---|---|
| harness default | `scripts/eval_olmo2_probe2_downstream.py:79` | `ALL_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]` |
| keep14@200k driver | `scripts/_run_olmo2_eval_keep14_s200000_b200.sh:42` | `CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"` |
| freeze_front@200k driver | `scripts/_run_olmo2_eval_freezefront_s200000.sh:43` | identical string |
| clean `_v2` ladder driver | `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh:54` (`CORE=`) | identical string |

Also confirmed against the data on disk: `olmo2_downstream_results/7B_keep14_step200000/summary.json`
holds exactly those six task keys.

**Note on ordering**: the harness constant (line 79), the docstring prose (line 34) and all
three drivers list the six tasks in the *same* order. Order is anyway irrelevant (the
harness iterates a dict and the aggregate is a macro-average), but keep the driver's string
byte-identical so nothing about the invocation differs from the existing rows.

**core6 aggregate convention** (not computed by the harness; computed downstream in
analysis): `acc_norm` for hellaswag / arc_challenge / arc_easy / piqa / openbookqa, and
`acc` for winogrande — because winogrande uses partial/double-cloze scoring where
`acc_norm == acc` by construction (`eval_olmo2_probe2_downstream.py:36-37`). This matches
`status/PAPERB_CORE6_CROSSARCH_FLOOR.md`'s recomputation.

### 1.2 know5 — the actual membership, with file:line

```
mmlu, lambada_openai, boolq, commonsense_qa, social_iqa
```

| authority | file:line | content |
|---|---|---|
| keep14@200k driver | `scripts/_run_olmo2_eval_keep14_s200000_b200.sh:58` | `KNOW_TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"` |
| freeze_front@200k driver | `scripts/_run_olmo2_eval_freezefront_s200000.sh:59` | identical string |
| clean `_v2` ladder driver | `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh:55` (`KNOW=`) | identical string |

⚠️ **know5 is NOT the harness's `KNOWLEDGE_TASKS` constant.** `eval_olmo2_probe2_downstream.py:91`
defines a **six**-element list that additionally contains `mmlu_pro`:

```python
KNOWLEDGE_TASKS = ["mmlu", "mmlu_pro", "lambada_openai", "boolq", "commonsense_qa", "social_iqa"]
```

`mmlu_pro` is **not** part of the Paper B know5 battery — no driver ever passed it, and no
`_know` results directory on either disk contains an `mmlu_pro` key (checked
`7B_keep14_step200000_know` and all three `_v2_know` dirs). The drivers always pass
`--tasks` explicitly, so the constant's default is never used. **Do not "fix" the driver to
use `KNOWLEDGE_TASKS`** — that would silently add a 7th task and break comparability.

### 1.3 Held-out PPL

- array: `data/dolmino_now_val.npy`, md5 **`f2ea48a2074a2f38fc3b4477fceecf11`**
  (verified byte-identical on wzc1 and zwfy6, 2026-08-15)
- expected merged totals: **`n_windows = 4096`, `n_tokens = 8384512`**
- `ppl = exp(sum_nll / sum_tokens)`, i.e. a token-weighted merge, not a mean of per-shard
  ppl (`eval_olmo2_probe2_ppl.py:271-272`)
- the driver pins the md5 and refuses to run against any other array.

### 1.4 Per-task expected item counts (pinned)

Asserted after every merge. Read 2026-08-15 from the clean `_v2` batteries on zwfy6; all
three arms agree exactly, so these are the protocol's item inventory rather than one run's
accident.

| core6 task | n | know5 task | n |
|---|---:|---|---:|
| hellaswag | 10042 | mmlu | 14042 |
| arc_challenge | 1172 | lambada_openai | 5153 |
| arc_easy | 2376 | boolq | 3270 |
| piqa | 1838 | commonsense_qa | 1221 |
| winogrande | 1267 | social_iqa | 1954 |
| openbookqa | 500 | | |

---

## 2. `chat_template = False` — and why it is not a flag

CLAUDE.md makes `chat_template=False` mandatory for the whole paper, and memory
`paper-eval-chat-false-mandatory` records that chat=True numbers are void. Memory
`paperb-olmo2-base-not-chat` adds that Paper B's OLMo-2 is a **BASE LM** (Dolmino continued
pretraining, no SFT), so it may only be scored on the base protocol — perplexity plus
likelihood-based MC — against vanilla OLMo-2 **BASE**.

**How this battery satisfies it**: neither `eval_olmo2_probe2_ppl.py` nor
`eval_olmo2_probe2_downstream.py` contains the string `chat_template` **at all** (grep
returns nothing in either file). They never call `apply_chat_template`; the MC harness
tokenises raw `(context, continuation)` string pairs and sums teacher-forced continuation
log-probs (`eval_olmo2_probe2_downstream.py:311-329` = `encode_pair`, called from
`score_task` at line 331). So chat=False is **structural, not
configurable** — there is no flag to pass and no way to turn it on.

Because "we passed the flag" is not available as evidence, the driver instead **asserts the
invariant**: preflight check P2 greps both harnesses for `chat_template|apply_chat_template`
and aborts if either ever gains such a code path. That converts an unwritten assumption
into a tripwire.

Related base-protocol invariants, all asserted:

| invariant | value | authority | asserted where |
|---|---|---|---|
| chat template | never applied | grep: absent from both harnesses | preflight P2 |
| BOS | **not** prepended (`add_bos = 0`) | `eval_olmo2_probe2_downstream.py:27-28` — OLMo-2's tokenizer does not auto-add BOS, and `add_special_tokens=False` matches how published OLMo-2 lm-eval numbers are made; harness default is `--add_bos 0` (line 614) | post-merge, `add_bos == false` in summary.json |
| dtype | fp32 master weights, bf16-autocast forward | `eval_olmo2_probe2_ppl.py:73` + `:104` (`build_pruned_shell(..., torch.float32)`); `eval_olmo2_probe2_downstream.py:21` | inherited; not configurable in the driver |
| few-shot | zero-shot | harness has no few-shot path | n/a |
| scoring | argmax over summed continuation log-prob (`acc`) and over log-prob / char-length (`acc_norm`) | `eval_olmo2_probe2_downstream.py:29-32` | n/a |
| target delimiter | `" "` prepended to every candidate | `eval_olmo2_probe2_downstream.py:25-26` | n/a |
| generation | none, except `lambada_openai` which is greedy last-word exact match | `GREEDY_TASKS` at line 93 | n/a |

---

## 3. Comparison denominator

Vanilla **OLMo-2-1124-7B BASE**, 32 layers, no continued pretraining, same battery.
On record and reusable — no need to re-measure:

| measurement | disk / arch | core6 | note |
|---|---|---:|---|
| `7B_base_full_bs8` | zwfy6 / H20 | .70365 | BS=8, torch 2.13.0 — **the protocol-matched one** |
| `7B_base_full` (a.k.a. `bs16`) | zwfy6 / H20 | .70368 | BS=16, off-protocol; the defect found in `PAPERB_FLIP_BOUNDARY_RESOLVED.md` |
| `7B_full32_base_wzc1` | wzc1 / B200 | .70402 | wrong architecture for this ladder |

Use `7B_base_full_bs8`. The three differ by ≤0.037 pp, which is instrument, not signal.

---

## 4. ★ Same-architecture comparability — where must keep10 be evaluated?

**This is a real risk, not a formality, and it points the opposite way from where keep10's
checkpoint lives.**

### 4.1 What the existing ladder numbers were measured on

The clean single-protocol batteries designated "the version of Table 4 to publish"
(`status/PAPERB_FLIP_BOUNDARY_RESOLVED.md`) are **all on zwfy6 / H20 / compute capability
9.0 / torch 2.13.0 / BS=8**:

| rung | directory (zwfy6) | core6 | arch |
|---|---|---:|---|
| base full32 | `7B_base_full_bs8` | .70365 | H20 cc9.0 |
| ShortGPT-16 | `7B_shortgpt16_step200000_v2` | .62247 | H20 cc9.0 |
| keep14@200000 | `7B_keep14_step200000_v2` | .59532 | H20 cc9.0 |
| keep12@124000 | `7B_keep12_step124000_v2` | .56888 | H20 cc9.0 |
| keep10@83500 | `7B_keep10_step83500_v2` | .52999 | H20 cc9.0 |
| keep8@121000 | `7B_keep8_step121000_v2` | .52328 | H20 cc9.0 |

(The *previously published* Table 4 mixed architectures across rungs — base/keep10/keep12/
ShortGPT from H20, keep8/keep14 from B200/L20A cc10.0 — see
`status/PAPERB_TABLE4_ARCH_AUDIT.md`. That mixing is the defect the `_v2` set fixes. We
must not re-introduce it.)

### 4.2 Size of the cross-architecture effect

Measured on **bit-identical weights** (`model_state` SHA-256 equal across disks) with the
same harness (`status/PAPERB_CORE6_CROSSARCH_FLOOR.md`, revised counts in
`PAPERB_HARNESS_DRIFT_REVISION.md`):

- keep14: 28 net-flipped items, core6 **+0.156 pp** (B200/L20A cc10.0 vs H20 cc9.0)
- vanilla base: core6 **+0.034 pp**
- matched-harness cross-arch flip counts: ShortGPT-16 = 7, keep10 = 23, keep8 = 29;
  winogrande dominates; signs vary per task, so it is symmetric noise rather than a
  directional hardware advantage
- PPL barely moves (1.4e-4 relative) because summed NLL averages the jitter instead of
  thresholding it

So the effect is **modest (≈0.03–0.16 pp on core6) but real, and it is entirely a
threshold/near-tie artifact of bf16 kernels**. Adjacent ladder rungs differ by 2.6–3.9 pp,
so ordering is not at risk. But the paper cannot claim a single measurement protocol while
mixing architectures, and per-item statistics (McNemar, paired bootstrap) are far more
exposed than the aggregates: batch-size experiments showed 107 exact flips behind only 16
net flips.

### 4.3 ⇒ Decision

> **All three arms — including keep10 — must be evaluated on H20 (`.73` / `.82` / `.104`,
> compute capability 9.0), with `/opt/conda/envs/torch-base/bin/python` (torch 2.13.0) and
> `--batch_size 8` for MC / `4` for PPL.**
>
> **keep10 must NOT be evaluated on the B200 it is training on**, even though its
> checkpoint is on wzc1 and the B200s are the fast, convenient, and possibly-idle nodes.

Grounds:
1. Every other rung of the publishable ladder is H20 cc9.0. A B200 keep10 row would be the
   *only* cross-arch row, i.e. exactly the defect `PAPERB_TABLE4_ARCH_AUDIT.md` reported.
2. keep10's own step83500 predecessor row is H20 (`7B_keep10_step83500_v2`), so an H20 200k
   row also gives a clean **within-arm** 83.5k → 200k trajectory. A B200 200k row would
   confound the trajectory with a hardware change.
3. The keep10 cross-arch flip count is 23 — the middle of the observed 7–29 range, not the
   small end. There is no basis for treating keep10 as the exception.
4. The cost of compliance is bounded and measured: moving one 39.0 GiB checkpoint across
   disks at 12–16 MB/s single-stream ≈ 42–53 min (measured 2026-08-15 with a 400 MiB
   `scp -O` probe). That is small next to the ~1 h eval and next to the 1.4 d of training
   that produced the checkpoint. Contrast the B-P0.4 case in CLAUDE.md, where moving two
   45.4 GiB checkpoints would have taken ~42 h and was correctly refused.

The driver enforces this with preflight **P4**, which reads `nvidia-smi --query-gpu=compute_cap`
and aborts unless every GPU reports `9.0`. Overriding it requires `SKIP_ARCH_GUARD=1`, which
prints a warning and, per this document, obliges you to record the deviation here. **A
B200-measured keep10 row must not enter the ladder table.**

### 4.4 Other protocol variables the driver pins

Four separate mechanisms have moved core6 without touching the model
(`PAPERB_FLIP_BOUNDARY_RESOLVED.md`), all silent by default:

| variable | effect | pinned to | asserted |
|---|---|---|---|
| torch version | 2.7.0 vs 2.13.0 ≈ 20 net flips | **2.13.x** | preflight P0 |
| eval batch size | bs8 vs bs16 = 107 exact / 16 net flips, +0.078 pp | **8** (MC), **4** (PPL) | driver default, recorded in the run JSON |
| partial shard merge | keep12 arc_easy 6/8 → +0.19 pp | **8/8 shards + full `n_scored`** | shard-file check + `n_scored` assertion |
| GPU architecture | 7–29 flips | **cc 9.0** | preflight P4 |

---

## 5. Per-arm disk / node / transfer plan

Live state as of 2026-08-15 ~17:45 +08:00 (read from the running processes, not from a ledger):

| arm | training node | training disk | ckpt dir | eval node | move needed? |
|---|---|---|---|---|---|
| **keep8** | `.82` (H20) | **zwfy6** | `outputs/olmo2_probe2_7B_keep8fresh2/` | any H20 (`.82` itself is natural) | **no** — already on zwfy6 |
| **keep12** | `.73` (H20) | **zwfy6** | `outputs/olmo2_probe2_7B_keep12fresh2/` | any H20 (`.73` itself is natural) | **no** — already on zwfy6 |
| **keep10** | **LOCAL (B200)** | **wzc1** | `outputs/olmo2_probe2_7B_keep10fresh2/` | **must be an H20** (§4.3) | **YES — `scp -O` step200000.pt wzc1 → zwfy6** |

Both disks currently hold *stale* copies of the other's keep-arm directories (e.g. wzc1 has
`keep12fresh2/step111500.pt` from July, zwfy6 has `keep10fresh2/` up to step90000 only), so
**`ls` the target disk before assuming a checkpoint is there.** CLAUDE.md records three
agents wasted on exactly this.

### keep10 transfer recipe (write it down now, run it then)

```bash
# on LOCAL (wzc1), once outputs/olmo2_probe2_7B_keep10fresh2/step200000.pt exists:
SRC=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep10fresh2/step200000.pt
DST=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep10fresh2/step200000.pt
md5sum "$SRC" | tee logs/keep10_step200k_src_md5.log          # ~1.5 min at ~500 MB/s
sshpass -f configs/password_h20_853573.txt scp -O \
  -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
  "$SRC" root@28.85.35.73:"$DST"                              # ~42-53 min for 39 GiB
# then on .73: md5sum "$DST" and compare byte-for-byte with the src log.
```

`scp -O` is mandatory (`.82`'s sftp subsystem is broken; plain `scp` reports
`subsystem request failed`). Do **not** omit `-p`-less form: the global `/etc/ssh/ssh_config`
already sets `Port 36000`, and writing `-p 22` gets `Permission denied`.

Checkpoint size note: keep10's saves are 39.0 GiB **because they carry optimizer state**
(the resume-enabling fields at `train_olmo2_arch_probe2.py:530-537`, inside the
`torch.save` block spanning lines 527-551). Only
`model_state` is needed for eval, but there is no strip tool in-repo and inventing one would
add a step whose correctness we would then have to prove. Move the whole file.

---

## 6. Preflight and completeness assertion checklist

### Preflight (all 8 run before any GPU work; each is fatal)

| # | check | why |
|---|---|---|
| P0 | `PYTHON_BIN` imports torch and it is **2.13.x** | version alone moves ~20 items |
| P1 | both harnesses + the assert helper exist under `PROJECT_ROOT` | the two-disk trap: wzc1-only files are invisible on zwfy6 |
| P2 | neither harness references `chat_template` / `apply_chat_template` | makes the mandatory chat=False protocol a tripwire rather than an assumption |
| P3 | `BASE_MODEL` dir exists; `VAL` exists **and** md5 == `f2ea48a2074a2f38fc3b4477fceecf11` | a different held-out array makes the PPL column non-comparable |
| P4 | every GPU reports `compute_cap == 9.0` (H20) | §4; overridable only via `SKIP_ARCH_GUARD=1` |
| P5 | ckpt exists; **`step == 200000`**; `keep_front_layers` and `n_fresh_layers` match the arm | stops evaluating step176500 under a 200k label |
| P6 | ckpt md5 recorded (`full` default; `head` = first 2 GiB; `none`) | provenance: proves later which bytes produced the numbers |
| P7 | none of the three output dirs already holds a `summary.json`; non-empty dirs without one are also refused | never overwrite an existing measurement; never let `--merge` mix stale and fresh shards |
| P8 | ≥5 GiB free on `PROJECT_ROOT` | the keep14 run was once killed mid-save on a full disk |

P5 reads `{step, keep_front_layers, n_fresh_layers, num_hidden_layers, has_optimizer,
train_args}` from the `.pt` via `torch.load(..., mmap=True)` — measured **5.6 s** on a
34 GiB file, no weights materialised, no GPU.

### Completeness, after each of the three merges

- all 8 `shard{i}of8.json` files present **before** the merge is even attempted
- `n_shards == 8` in the merged summary
- **per task: `n_scored == pinned expected count`** — not merely `n_nan == 0`
- per task: `n == expected` and `n_nan == 0` and `acc` / `acc_norm` numeric
- no task marked `skipped`
- `add_bos == false`
- PPL: `n_windows == 4096` and `n_tokens == 8384512` and `ppl > 0`

**Why `n_scored` and not just the shard-file count.** Empirically demonstrated in the
dry-run: copying 6 of the 8 real keep14 shard files into a scratch dir and running the
harness's own `--merge` produced a clean-looking summary — `n_nan = 0` on all six tasks,
plausible accuracies, no warning — with `arc_easy n = 1782`, which is byte-for-byte the
historic keep12 defect (1782 = 6×297). `eval_olmo2_probe2_downstream.py:489-570` (`merge`)
increments `n_skipped_shards` at line 507 but **never writes it into `summary.json`** (the
`summary` dict is built at lines 554-560 without it), so a partial
merge is invisible from the summary alone. (The PPL harness, by contrast, *does* refuse a
partial set — `eval_olmo2_probe2_ppl.py:297-308` raises unless `--allow_partial_merge`.) Our
assertion caught it with 13 defects.
And the converse case is real too: 8/8 shard files present with `n_nan = 3` still means 3
fewer items in the accuracy denominator; only the `n_scored` check catches that.

---

## 7. Launch order: who first, and on which node

Measured ETAs, from Δ(log timestamp)/Δ(step) over the last 60 and last 200 logged step
lines (**not** the `s/step` postfix field, and not a single sample; the two windows agree to
<1%). As of 2026-08-15 ~17:44 +08:00:

| arm | node | step | s/step | remaining | ETA to 200k |
|---|---|---:|---:|---:|---:|
| **keep10** | LOCAL (B200) | 109,000 | 1.336 | 91,000 | **1.41 d** |
| **keep12** | `.73` (H20) | 177,000 | 7.903 | 23,000 | **2.10 d** |
| **keep8** | `.82` (H20) | 145,860 | 5.852 | 54,140 | **3.67 d** |

**keep10 finishes training first but is evaluated last-ish**, because §4.3 requires H20 and
no H20 frees up until keep12 finishes at ~2.1 d. Resulting order:

1. **t ≈ 1.4 d — keep10 finishes training on LOCAL.** GPU-free work only: `scp -O` its
   `step200000.pt` to zwfy6 (~42–53 min) and verify md5. **Do not evaluate it on the B200.**
   LOCAL's 8 B200s become free here; per CLAUDE.md's priority rule they go to
   paperC / proposal, not to a Paper B eval that must not run there anyway.
2. **t ≈ 2.1 d — keep12 finishes on `.73`.** Launch keep12's eval on `.73` immediately
   (checkpoint is already local to zwfy6, zero transfer). When it completes (~1 h), launch
   keep10's eval on `.73` — by then its checkpoint has been sitting on zwfy6 for ~0.7 d.
3. **t ≈ 3.7 d — keep8 finishes on `.82`.** Launch keep8's eval on `.82` (zero transfer).

`.104` is running paperC Qwen3 and is not assumed available. If it frees up, keep10's eval
can go there instead of queueing behind keep12 — any H20 satisfies the protocol.

Wall clock per arm: 3 batteries × ~15–25 min per harness on 8×H20 ≈ **45–75 min**
(`_run_olmo2_p24_eval_ladder_prev2_73.sh` measured ~10 min/arm for a 5-harness battery on
H20 in one instance; budget the wider range).

### Exact launch commands

```bash
# ---- keep12, on .73 (zwfy6) ----
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
ARM=keep12 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  setsid nohup bash scripts/eval_paperb_ladder_200k.sh \
  > logs/ladder200k_eval_keep12.log 2>&1 &

# ---- keep8, on .82 (zwfy6) ----
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
ARM=keep8 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  setsid nohup bash scripts/eval_paperb_ladder_200k.sh \
  > logs/ladder200k_eval_keep8.log 2>&1 &

# ---- keep10, on ANY H20 (zwfy6), AFTER the scp -O of §5 ----
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
ARM=keep10 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  setsid nohup bash scripts/eval_paperb_ladder_200k.sh \
  > logs/ladder200k_eval_keep10.log 2>&1 &
```

Everything else is defaulted from `ARM`: ckpt path, `keep_front_layers`, output names
(`7B_<arm>_step200000` and `..._know`), batch sizes, shard count. Always dry-run first —
it costs seconds and no GPU:

```bash
DRY_RUN=1 ARM=keep12 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  bash scripts/eval_paperb_ladder_200k.sh
```

**Do not launch an eval on a node that is still training.** Each arm's eval starts only
after that arm's own training process has exited; `.73` / `.82` are fully occupied until
then (8×~78–96 GB per card).

---

## 8. What these rows will and will not settle

**Will settle**: the depth ladder at a genuinely equal **step** budget. All four arms
(keep8 / keep10 / keep12 / keep14) will be at 200,000 optimizer steps, and all six rungs of
the publishable table will be single-disk, single-arch, single-torch, single-batch-size,
full-shard.

Also resolved by these resumes, and worth stating explicitly: the **two-corpora defect**
(`status/PAPERB_TWO_CORPORA_DEFECT.md`) reported that keep8/10/12 trained on a 15,491,607-row
array while keep14 / ShortGPT / freeze_front trained on a 7,570,911-row **byte prefix** of
it. The three current resumes all log `dataset rows=15491607` with md5
`7df19b217e5b0670d58bf6e01e6559d0` — verified identical on LOCAL (`dolmino_now15b_wzc1.npy`),
`.73` and `.82` (`dolmino_now15b.npy`). So keep8/10/12 will be mutually comparable in
corpus, steps, and tokens. **keep14 / ShortGPT / freeze_front are still on the 7.57M-row
prefix and therefore still differ in epochs-over-corpus** (3.38 vs 1.65 at 200k). Reaching
step200000 does **not** fix that; only the shallow trio becomes internally clean.

**Will not settle**:
- **Resume fidelity.** All three resumes went through a warm restart of Adam moments (the
  ckpt's 2 param groups vs the current code's 4 → `AdamW.load_state_dict` raises and the
  trainer degrades to warm restart), and the dataloader position is not restored. keep10's
  resume log shows the 4-group split at a uniform `base_lr=2.00e-05`. So each arm's 200k is
  a spliced trajectory with a discontinuity at a different progress fraction, and keep14's
  200k is not. This is the cost `status/PAPERB_192_TABLE4_BUDGET_DECISION.md` §2.2 flagged
  when it recommended *against* resuming — it must be disclosed in the paper, not
  presented as "all arms at 200k, therefore matched".
- **Differential LR.** Do not claim it. The `_classify_param` prefix bug means the keepN
  ladder trained at a uniform 2e-5 (CLAUDE.md; `status/PAPERB_DIFFERENTIAL_LR_NEVER_ACTIVE.md`).
- Anything about seeds (n=1 per rung), model families, or causal attribution of depth.

**Do not overwrite the existing `_v2` rows.** The 121k / 83.5k / 124k measurements remain
valid and are the within-arm trajectory predecessors of the new 200k points. New output
names are `7B_<arm>_step200000{,_know}`, which collide with nothing on either disk (checked
2026-08-15), and preflight P7 refuses to run if they ever do.

---

## 9. Provenance of every claim in this document

| claim | source |
|---|---|
| battery = PPL + core6 + know5, and its flags | `scripts/_run_olmo2_eval_keep14_s200000_b200.sh`, `scripts/_run_olmo2_eval_freezefront_s200000.sh` |
| core6 membership | `scripts/eval_olmo2_probe2_downstream.py:79`; driver lines 42 (keep14) / 43 (freezefront) / 54 (`_v2`) |
| know5 membership, and that it excludes `mmlu_pro` | driver lines 58 (keep14) / 59 (freezefront) / 55 (`_v2`) vs harness `KNOWLEDGE_TASKS` at line 91; no `_know` summary on either disk has an `mmlu_pro` key |
| chat_template absent from both harnesses | `grep -n "chat_template" scripts/eval_olmo2_probe2_{ppl,downstream}.py` → no matches |
| add_bos convention | `scripts/eval_olmo2_probe2_downstream.py:27-28, 614`; `add_bos: false` in every summary.json |
| base LM protocol requirement | CLAUDE.md; memory `paperb-olmo2-base-not-chat`, `paper-eval-chat-false-mandatory` |
| clean single-protocol `_v2` table is all H20 cc9.0 / torch 2.13.0 / BS=8 | `status/PAPERB_FLIP_BOUNDARY_RESOLVED.md` |
| previously published Table 4 mixed architectures per rung | `status/PAPERB_TABLE4_ARCH_AUDIT.md` |
| cross-arch flip magnitude (28 flips / 0.156 pp; 7 / 23 / 29 matched) | `status/PAPERB_CORE6_CROSSARCH_FLOOR.md`, `status/PAPERB_HARNESS_DRIFT_REVISION.md` |
| torch-version and batch-size flip magnitudes | `status/PAPERB_FLIP_BOUNDARY_RESOLVED.md`, `status/PAPERB_BATCHSIZE_FLIP_CAUSE.md` |
| keep12 arc_easy 6/8 partial merge | `status/PAPERB_TABLE4_KEEP12_PARTIAL_MERGE.md`; reproduced in the dry-run |
| expected per-task counts | zwfy6 `olmo2_downstream_results/7B_{keep8_step121000,keep10_step83500,keep12_step124000}_v2{,_know}/summary.json`, read 2026-08-15 |
| val array md5 identical on both disks | `md5sum data/dolmino_now_val.npy` on LOCAL and `.73` → `f2ea48a...` both |
| ETAs | Δ(timestamp)/Δ(step) over `logs/olmo2_7B_keep10fresh2_resume200k_local_0815.log`, `zwfy6:logs/olmo2_7B_keep12fresh2_resume200k_73_0814.log`, `zwfy6:logs/olmo2_7B_keep8fresh2_resume200k_82_0814.log` |
| training corpora now identical | the three resume logs' `dataset rows=15491607` + md5 `7df19b...`; `logs/corpus_{wzc1,zwfy6}_md5.log` |
| resume warm-restart / uniform LR | `status/PAPERB_192_TABLE4_BUDGET_DECISION.md` §2.2; `status/PAPERB_DIFFERENTIAL_LR_NEVER_ACTIVE.md`; keep10 resume log `[optim] group ... base_lr=2.00e-05` ×4 |
| cross-disk transfer rate | measured 2026-08-15, `scp -O` of 400 MiB LOCAL→.73: 26.6 s to ceph (15.8 MB/s), 33.6 s to /tmp (12.5 MB/s) |
| ckpt sizes / which disk | `ls -la` on both disks, 2026-08-15 |
| assertions actually fire | `paperB/evidence/ladder_200k_eval_dryrun.json` — 13 negative controls, all rc=2 |

---

## MAIN independent verification (2026-08-15, after agent delivery)

I did not accept the load-bearing claims on the agent's word. The two that could
corrupt published numbers were re-derived from primary sources.

### 1. The silent 6/8 shard merge is REAL — verified at source

`scripts/eval_olmo2_probe2_downstream.py`:
- line **507**: `a["n_skipped_shards"] += 1` — the counter *is* incremented
- lines **553-559**: the `summary` dict carries only `output_name`, `n_shards`,
  `add_bos`, `meta`, `tasks` — **`n_skipped_shards` is never surfaced**

So a partial merge produces a summary with `n_nan=0`, plausible accuracies, and no
warning. The agent's reproduction (6 real keep14 shards → `arc_easy n=1782`) is
arithmetically exact: 2376/8 = 297 per shard, 6 × 297 = **1782**. And 1782 is not a
number the agent constructed — it appears independently in **four** pre-existing
records: `paperB/P0_7_AGGREGATE_AUDIT.md`, `status/PAPERB_192_TABLE4_BUDGET_DECISION.md`,
`status/PAPERB_P24_SFT_KEEP12_EVAL.md`. It is the documented historic keep12 defect.

**Consequence:** `n_nan == 0` is *not* sufficient as an integrity check on this
harness, and never was. Only a per-task `n_scored == expected` assertion catches it.

### 2. know5 is NOT the harness's `KNOWLEDGE_TASKS` constant — verified

`eval_olmo2_probe2_downstream.py:91`:
```python
KNOWLEDGE_TASKS = ["mmlu", "mmlu_pro", "lambada_openai", "boolq", "commonsense_qa", "social_iqa"]
```
That is **six** tasks. know5 as actually run is the five passed explicitly by every
driver: `mmlu, lambada_openai, boolq, commonsense_qa, social_iqa` — no `mmlu_pro`.
Also confirmed `ALL_TASKS` at line **79** is exactly the core6 six in the stated order.

**Consequence:** "simplifying" the driver to use `KNOWLEDGE_TASKS` would silently add
a 7th task and break comparability with every existing row. The explicit `--tasks`
list is load-bearing, not verbosity.

### 3. The architecture guard is executable, not advisory

`scripts/eval_paperb_ladder_200k.sh:165-171` reads
`nvidia-smi --query-gpu=compute_cap`, and `die`s unless every GPU reports the required
capability. `SKIP_ARCH_GUARD=1` exists as an escape but logs a warning that the run
will not be comparable. I checked this is real control flow because a prose-only
"should run on H20" would have been worthless the first time a B200 was free and idle.

### 4. What I did NOT verify

The 13 negative controls individually, the 12–16 MB/s transfer measurement, and the
0.03–0.16 pp cross-arch floor magnitudes. These are agent-measured and recorded as
such; the two claims above are the ones that gate whether the published table is
correct, so those are the ones I re-derived.
