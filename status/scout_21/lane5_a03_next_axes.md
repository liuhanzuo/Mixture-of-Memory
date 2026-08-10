# lane5_a03_next_axes — A03's next floor gate (three remaining knowledge axes)

STATUS: IN PROGRESS (written early, updated as evidence lands)

## Read so far
- status/A03_1B_FLOOR_VERDICT.md — verdict 1B_PILOT_VIABLE; only "old parametric knowledge"
  axis floor-certified (MMLU-content / PopQA / TriviaQA). Primary axis = TriviaQA EM.
- proposal/active/A03-parametric-vs-external-memory/{PROPOSAL.md,STATUS.json}
- scripts/_run_a03_1b_floor_82.sh (driver template, 114 lines)
- scripts/eval_olmo2_closedbook_qa.py (510 lines)

## Early finding: wzc1 ALSO has the old-axis HF caches
`/apdcephfs_wzc1/.../data/hf_datasets_cache/` contains `akariasai___pop_qa`,
`mandarjoshi___trivia_qa` (rc.nocontext), `cais___mmlu` — so the OLD axis is
reproducible on `.21` too, not only on `.82`.
(sizes + ckpt presence being measured next)

## VERIFIED EVIDENCE (paths + sizes actually observed)

### Nodes (measured 2026-08-08 ~21:27)
- `.82` (zwfy6, 8xH20): **all 8 GPUs 0%/0 MiB** — confirmed idle.
- `.21` (wzc1, 8xL20A): 7 of 8 idle. **GPU0 has 26318 MiB in use by PID 38067**
  = `eval_olmo2_mmlu_content.py --base_model ../models/Llama--Llama2-7b --any_family
  --output_name SMOKE_llama2_7b --limit 40` under `timeout 1500`, started 21:25.
  A 25-min-capped smoke, someone else's. Not mine to kill. It self-expires.
- `.21` interpreter `/opt/conda/envs/torch-base/bin/python`: torch 2.13.0, transformers 4.57.6,
  datasets 2.21.0, numpy 2.5.1, **NO scipy** -> `analyze_1b_knowledge_floor.py`'s exact
  McNemar (`from scipy.stats import binomtest`) would silently degrade to `nan` on .21.
  LOCAL and (implicitly) the CPU step have scipy 1.18.0 -> run the ANALYSIS on LOCAL, not .21.

### 1B checkpoints
- zwfy6: `/apdcephfs_zwfy6/.../outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`
  = 12181310078 B (**11.3 GiB**); siblings step50000/100000/150000/final.pt same size;
  `arch_meta.json` 826 B.
- zwfy6: `/apdcephfs_zwfy6/.../outputs/olmo2_probe2_1B_keep7fresh2/step500.pt`
  = 12181308233 B (**11.3 GiB**).
- wzc1: **BOTH DIRS ABSENT.** `ls outputs/olmo2_probe2_1B_keep7fresh2{,_16card}` ->
  "No such file or directory". No `olmo2_probe2_1B*` under wzc1 `outputs/` at all
  (only `sembott_1b_*`, `minarch_1b_*`, `e2e_1b_*`).
  => running on `.21` needs **~22.6 GiB cross-disk** (2 ckpts) ~ 10-30 min at 12-37 MB/s.
- Base model IS on wzc1: `../models/OLMo-2-0425-1B` **5.6G**, 16L / d2048 / ctx4096.

### Old-axis dataset caches (the ones the passed gate used)
- wzc1 `data/hf_datasets_cache/`: `akariasai___pop_qa` 5.4M, `mandarjoshi___trivia_qa` 120M,
  `cais___mmlu` 162M. Loaded offline OK on LOCAL just now: PopQA n=**14267** exact.
- zwfy6 has the same three **plus** `google-research-datasets___nq_open`.
  wzc1 is MISSING nq_open (harness supports it but A03 doesn't name it).

## 1. THE THREE REMAINING AXES, OPERATIONALISED

The passed gate's axis (old parametric knowledge) is **closed-book**: the harness prompt is
hardcoded `f"Question: {q}\nAnswer:"` (`eval_olmo2_closedbook_qa.py:212-213`), no context slot.
**All three remaining axes are evidence-supplied (open-book) axes** — they need a context in
the prompt. That is the single structural fact that decides the plan.

`eval_olmo2_closedbook_qa.py` does NOT generalise to them: `load_task_examples()` is a
hardcoded if/elif over `popqa`/`triviaqa`/`nq_open` calling `load_dataset(...)` (lines 134-209);
there is no `--val_path`, no context field, and `build_prompt` takes only a question.
Adding an axis there = editing that dispatch + the prompt = a code change.

**BUT there is a second, already-shipped harness that IS generic:**
`scripts/eval_paperC_squad_emf1.py` — same no-drift loaders (`load_pruned_model`/
`load_base_model` from `eval_olmo2_probe2_ppl`), same scorer (imports `normalize_answer`/
`score_prediction` **from `eval_olmo2_closedbook_qa`**, so metrics are byte-identical),
same base protocol (`--add_bos 0`, no chat template), same shard/merge, and crucially:

    p.add_argument("--val_path", type=str, default="data/squad_val.jsonl")
    p.add_argument("--max_ctx_len", type=int, default=1024)

with the loader
    prompt = (f"Context: {context}\n\nQuestion: {q}\n\nAnswer:" if context else ...)
    context = " ".join(memory_texts);  gold = [target_text]

So **any new open-book axis is "another jsonl in the same 4-field shape"**
(`input_text`, `target_text`, `memory_texts`, `relevant_indices`) and needs **zero harness
edits** — only a converter script. I VERIFIED this by writing HotpotQA into that shape and
loading it through the real, unmodified `load_squad()`: 7405/7405 rows loaded, prompt renders
`Context: Ed Wood: Edward Davis Wood Jr. ... Question: ... Answer:`, gold `['yes']`.

### Axis 2 — multi-evidence  → **HotpotQA distractor, on disk in 20 s**
- Literature standard = multi-hop. Already on BOTH disks as LongBench slices:
  `data/longbench_raw/data/{hotpotqa,2wikimqa,musique}.jsonl` (200 items each,
  11.5M/6.1M/14.1M) — **but they DON'T FIT**: LongBench packs them to
  `length` p50 = 10145 / 4218 / 11388 tokens, and OLMo-2-1B `max_position_embeddings=4096`.
  Only 10/200 hotpot and 1/200 musique are <=3500 tok. n=200 is also too small for the
  gate's BH family. **Use LongBench multihop only as an optional stress arm, not the gate.**
- **The right instantiation is the ORIGINAL HotpotQA distractor validation set**:
  7405 items, 2 gold paragraphs + 8 distractors, `supporting_facts` gives EXACT gold
  paragraph titles -> populates `relevant_indices` truthfully (which the A03 SQuAD builder
  had to fake, and `build_paperC_squad_eval.py:19-23` documents that as a defect).
  NOT on either disk. I pulled it to /tmp to measure: **27452575 B = 26 MiB, 3 s on LOCAL,
  20 s on .82** through `hy-proxy.woa.com:3128`. Negligible vs the CAST download.
  Token lengths under the OLMo-2 tokenizer, measured on 1200 items:
    * oracle (2 gold paras):  p50 229, p90 344, p99 482, max 640  -> **1200/1200 fit in 1024**
    * full (10 paras):        p50 1307, p90 1752, p99 2347, max 3119 -> 1166/1200 fit 2048,
                              **1200/1200 fit 3584**
  So a 2-condition design (oracle vs 10-paragraph distractor) fits 1B natively — and that
  contrast IS the multi-evidence measurement (does the model combine 2 facts when they are
  present vs buried).
- Floor I computed with the gate's own `best_constant_qa`: **EM best-constant = "no" 0.0315**
  overall, but **0.1567 on the `comparison` subtype (n=1487) vs 0.0035 on `bridge` (n=5918)**
  — because 458/7405 golds are literally yes/no. **This must be stratified by type or the
  comparison subset's yes/no prior will masquerade as multi-hop skill.** This is a real new
  methodological requirement for this axis, analogous to the length-matched `contains` finding.

### Axis 3 — updated / conflicting facts → **LongMemEval `knowledge-update`, already on disk**
- A03's PROPOSAL.md line 39 already names "LongMemEval update/temporal". It IS on disk:
  * wzc1 `data/longmemeval/longmemeval_s.json` **266M / 278025796 B**, 500 items
  * zwfy6 `data/dialogmem/longmemeval/longmemeval_s` 278025796 B (identical size)
    plus `longmemeval_oracle` 15388478 B (**only on zwfy6**)
- Type mix measured: `multi-session 133 / temporal-reasoning 133 / knowledge-update 78 /
  single-session-user 70 / single-session-assistant 56 / single-session-preference 30`.
  **`knowledge-update` n=78 is exactly the old/new-conflict construct** — and every one of
  its 78 items has **exactly 2 answer sessions** (the old value session + the update
  session), which is precisely the conflict pair.
- **Length problem is real**: oracle-only (answer sessions only, discarding the haystack)
  est. tokens p50 6780 / p90 8352 / max 10713 for `knowledge-update`; **0 of 78 fit in
  3500 tok**. multi-session p50 7948 max 21034; temporal p50 7613 max 27276.
  => On OLMo-2-1B (4096 ctx) this axis needs **turn-level extraction** (keep only the
  evidence turns rather than whole sessions), which is a genuine dataset-construction step,
  not a converter. And n=78 is thin for BH.
- **CounterFact is the cheaper, better-powered alternative for "conflicting facts"**:
  `azhx/counterfact` reachable via proxy, **1250790 B = 1.2 MiB**, downloaded and inspected:
  n=**2191**, fields `requested_rewrite{prompt,subject,target_true,target_new}`,
  `paraphrase_prompts`, **`neighborhood_prompts`** (the locality control), `attribute_prompts`.
  `target_true` vs `target_new` is a ready-made old-vs-new conflict pair, and the
  in-context-conflict framing (supply the counterfactual as context, ask the question)
  fits in ~200 tokens. zsRE via `zjunlp/KnowEdit` benchmark/ZsRE/ZsRE-test-all.json also
  reachable (HTTP 200). `henryzhongsc/MQuAKE` returns **401 (gated)** — do not plan on it.

### Axis 1 — new (post-cutoff) facts → **NOT on disk, and needs a design decision**
- Nothing resembling FreshQA / a post-cutoff set exists on either disk. Searched
  `data/`, `.hf_cache/`, `data/hf_datasets_cache/`, `/root/.cache/huggingface/hub` on wzc1
  and the zwfy6 equivalents for `*freshqa* *realtime* *templama* *temporal* *livebench*`:
  zero hits both disks.
- The tractable instantiation is **synthetic injection**: fabricated entity-attribute facts
  the model provably cannot know, supplied as context. This is *also* the arm that makes
  the axis meaningful for A03 (whose whole question is "where should NEW knowledge live"),
  and it makes the null trivially computable and contamination-proof.
  **But it is a dataset the project does not have and must generate** — that is a code task,
  not a launch. CounterFact's `target_new` (an attested-false value) is the closest
  on-hand proxy and can stand in for the pilot.

## 2. FLOOR-GATE PROTOCOL TO REPEAT (extracted from the verdict doc)

Confirmed present in `analyze_1b_knowledge_floor.py` and confirmed **reusable as a library**
(I imported it and ran the generic path on synthetic paperC-shaped rows — worked):

| requirement | function | line | reusable as-is? |
|---|---|---|---|
| best-constant null maximised over candidate answer strings (top-300 golds + empty + 2 refusals), NOT the harness's `majority_em` | `best_constant_qa` | 227 | **YES, axis-agnostic** (takes `rows` with `gold`) |
| length-matched verbose input-blind null for `contains` | `lengthmatched_contains_null` | 283 | **YES, axis-agnostic** |
| paired bootstrap n_boot=10000, multinomial, two-sided p floored 1/n_boot | `paired_bootstrap` | 123 | YES |
| exact-binomial McNemar on discordant items | `mcnemar_exact` | 136 | YES (**needs scipy** — absent on .21) |
| BH q=.05 across the whole cell family | `bh_reject` | 150 | YES |
| four-tuple cell reported/null/residual/frac + CI | `cell` | 316 | YES |
| self-test that the re-implemented scorer matches the harness | inline | 451-454 | YES |
| at-floor CONTROL arm (`step500`) so "at floor" is provably detectable | design | — | **must be kept** |

**What is NOT reusable**: `main()` is hardwired — `assert n_mmlu == 14042` (line 366),
`(("popqa", 14267, ...), ("triviaqa", 17944, ...))` (line 427), and it reads
`per_example_{popqa,triviaqa}.jsonl` from `olmo2_closedbook_results`. A new axis needs a
sibling `main` (or a `--axis name=dir/file:n` flag). ~60-80 lines, importing everything above.
This is the small fix; the statistics do not get re-derived.

Also load-bearing from the verdict: **MMLU-letter is BANNED at 1B** — irrelevant here since
all three remaining axes are generative, so `contains` length-matching is the live risk, and
for HotpotQA additionally the **yes/no comparison-subtype prior**.

## 3. NODE RECOMMENDATION: `.82`

- The 1B ckpts are zwfy6-only (measured above; wzc1 has neither dir). Running on `.21`
  costs a **22.6 GiB** cross-disk copy (2 x 11.3 GiB) — at the measured 12-37 MB/s that is
  **~17 min (4-stream) to ~31 min (single-stream)**, plus md5. Not fatal, but pure waste.
- `.82` has everything already: both ckpts, base model 5.6G, LongMemEval, the three
  harnesses, `analyze_1b_knowledge_floor.py`, scipy 1.18.0 + pyarrow 25.0.0, 86T free,
  8/8 GPUs at 0 MiB, and proxy egress (27 MiB in 20 s).
- `.21` is **worse on two counts**: (a) **no scipy** -> `mcnemar_exact` returns `nan`,
  silently dropping a required statistic; (b) GPU0 currently holds 26318 MiB
  (PID 38067, someone's `SMOKE_llama2_7b`, `timeout 1500`).
- Also `.82` git HEAD is `2d98c5a` — behind LOCAL — but **all five needed files are already
  present there** with recent mtimes (`_run_a03_1b_floor_82.sh` 21:08 today,
  `analyze_1b_knowledge_floor.py` 21:08 today), i.e. this lane needs no code sync
  except the NEW converter + NEW analyzer entry point.
- **A 1B x ~7400-item generative eval on 8 GPUs is ~1-3 min per arm** (the passed gate did
  14042 MMLU + 14267 + 17944 generative = 3 arms in **7 min total**, per
  `logs/a03_1b_floor_progress.log` on .82: DRIVER START 20:48 -> END 20:55, rc=0).
  So this gate is small. `.21` should take something bigger.

## 4. READINESS VERDICT PER AXIS

| axis | dataset | on disk? | harness | READY? |
|---|---|---|---|---|
| multi-evidence | HotpotQA distractor val (7405) | NO — 26 MiB, 3-20 s via proxy | `eval_paperC_squad_emf1.py --val_path` **unmodified** | **READY_AFTER_SMALL_FIX** (converter ~50 lines, validated) |
| updated/conflicting | CounterFact (2191) | NO — 1.2 MiB, downloaded+inspected | same, unmodified | READY_AFTER_SMALL_FIX (converter + conflict framing) |
| updated/conflicting (alt) | LongMemEval knowledge-update (78) | **YES both disks** | needs turn-extraction | BLOCKED_NEEDS_CODE (0/78 fit 4096 ctx; n=78 thin) |
| new facts | synthetic injection | **NO, nothing analogous** | same, unmodified | BLOCKED_MISSING_ASSET (must be generated) |

**Overall lane verdict: READY_AFTER_SMALL_FIX.** Two of three axes can be floor-gated
tonight on `.82` with ~2 new files and zero harness edits. The third (new facts) needs a
generated dataset first, so it should NOT hold up the other two.

## 5. WHAT THIS GATE DECIDES

It **CAN KILL** — but only partially, and it is important to be precise, because A03's
kill condition is conjunctive ("**所有** 知识指标均处于 floor"). The old-parametric axis
already passed, so **no result here can kill A03 as a whole any more.** What it can kill is
narrower and still decision-relevant:

* If the pruned+healed 1B arm is at/below its own floor on the multi-evidence axis while the
  intact arm is above it, that axis is **retired as an A03 interface at 1B** — exactly as
  MMLU-letter was retired. That prevents the 6-arm build from spending 6 x GPU-days
  producing an arm ranking that is really ranking degeneracy.
* If BOTH remaining measurable axes retire, A03's 6-arm design collapses to the single
  old-parametric axis, which materially weakens the "分工 depends on whether knowledge is
  old/new/updated/multi-evidence" thesis (PROPOSAL 核心假设 line 21) — that is a design
  kill for the paper's main claim even though the proposal survives.
* If they clear floor, it **licenses** the 6-arm build on 3 axes instead of 1.

So: **decorative for A03's life/death, load-bearing for A03's scope.** MAIN should not
oversell it as a kill gate.

## 6. RISKS / GOTCHAS I FOUND

1. **`eval_paperC_squad_emf1.py` does NOT merge per-example jsonl.** It writes
   `per_example_shard{i}of{N}.jsonl` (line 206-207) but `merge()` (line 108-131) only sums
   `n`/`em_hits`/`f1_sum` — there is **no `_merge_per_example`** (contrast
   `eval_olmo2_closedbook_qa.py:318,322`). The floor analysis NEEDS per-item vectors.
   => the driver must concatenate + sort by `item_id` itself, and assert the count.
   This is the one real defect; it is 6 lines.
2. **`merge()` does not assert shard count or n.** It globs whatever exists. A 5-of-8 merge
   would silently produce a wrong denominator (the exact failure mode the project already
   suffered). The driver MUST assert 8/8 and exact n before trusting summary.json.
3. **`contains` is not even reported** by this harness (only `em` + `f1`), so the
   length-matched-null finding must be re-derived from the per-example dump. Fine — the
   dump has `pred` and `gold`, and `lengthmatched_contains_null` is reusable.
4. **HotpotQA yes/no prior**: 458/7405 golds are yes/no; best-constant EM = 0.1567 on the
   1487 `comparison` items vs 0.0035 on 5918 `bridge`. Stratify or the floor is wrong.
5. `.21` has **no scipy** -> run any analysis on LOCAL or `.82`, never `.21`.
6. `analyze_1b_knowledge_floor.py`'s `main()` hardcodes 14042/14267/17944 — a new axis
   entering through `main()` will assert-fail. Use the library functions, not `main()`.

## 7. LAUNCH PLAN (node `.82`, 8 GPUs, est 25-40 min wall incl. the two new files)

Three steps; MAIN writes files 1+2 (or dispatches a coder), then runs step 3.
All paths absolute-from-`$W` where `W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`.

### Step 1 — fetch + convert (CPU on .82, ~2 min, proxy)
New file `$W/proposal/active/A03-parametric-vs-external-memory/code/build_a03_newaxes.py`
must: (a) read the two parquet files, (b) emit the paperC 4-field jsonl shape
(`input_text` / `target_text` / `memory_texts` / `relevant_indices`, plus
`question_type` + `item_key` for stratification), (c) print exact n, (d) be seeded.
Verified-correct core (I ran this against the real `load_squad()`):

```python
# hotpot: 3 conditions -> oracle (2 gold paras), full (10 paras), closed (no context)
c = x["context"]; titles=list(c["title"]); sents=list(c["sentences"])
gold = set(x["supporting_facts"]["title"])
paras = [f"{t}: {''.join(s)}".strip() for t,s in zip(titles,sents)]
keep  = [(t,p) for t,p in zip(titles,paras) if t in gold]   # or all, for `full`
idx = list(range(len(keep))); rng.shuffle(idx)              # rng = random.Random(20260808)
mem = [keep[i][1] for i in idx]
rel = [j for j,i in enumerate(idx) if keep[i][0] in gold]
row = {"input_text": "根据以下对话记录，回答问题：" + x["question"].strip(),
       "target_text": x["answer"].strip(), "memory_texts": mem,
       "relevant_indices": rel, "question_type": x["type"], "item_key": x["id"]}
```
(The CN prefix is required: `_clean_question` splits on the fullwidth colon `：`,
`eval_paperC_squad_emf1.py:47-53`. Without it the question is passed through unchanged,
which also works, but keeping the prefix matches the existing SQuAD rows exactly.)

For counterfact: `memory_texts = [f"{subject} {prompt_filled} {target_new}."]`
as the conflicting-context condition, `target_text = target_new.str` for the
"follow the context" reading and a second file with `target_text = target_true.str`
for the "stale/parametric" reading — A03's 必报成本 line 50 explicitly wants
`stale/conflict error`, and these two files give exactly that pair on identical inputs.
`neighborhood_prompts` gives the locality control (no-conflict items).

Commands:
```bash
sshpass -f /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_82250.txt \
 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.82.250.82 \
 'W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory; cd $W
  mkdir -p data/a03_newaxes logs
  export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=http://hy-proxy.woa.com:3128
  export no_proxy=localhost,127.0.0.1,.oa.com,.woa.com,.local
  curl -sSL -o data/a03_newaxes/hotpot_distractor_val.parquet \
    https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet
  curl -sSL -o data/a03_newaxes/counterfact_test.parquet \
    https://huggingface.co/datasets/azhx/counterfact/resolve/main/data/test-00000-of-00001-bacb83500fca49a9.parquet
  ls -la data/a03_newaxes/
  unset http_proxy https_proxy
  /opt/conda/envs/torch-base/bin/python \
    proposal/active/A03-parametric-vs-external-memory/code/build_a03_newaxes.py \
    --out_dir data/a03_newaxes --seed 20260808'
```
Expected: `hotpot_distractor_val.parquet` **27452575 B**, `counterfact_test.parquet`
**1250790 B**; hotpot jsonls **n=7405** each, counterfact **n=2191**.

### Step 2 — new driver `$W/scripts/_run_a03_newaxes_82.sh` (adapted from `_run_a03_1b_floor_82.sh`)
Keeps every invariant of the template: `--add_bos 0`, no chat template, offline env,
`PROG` log written at every phase, 8/8 shard assertion, **plus** the two things the
paperC harness lacks (exact-n assertion and per-example concatenation).

```bash
#!/usr/bin/env bash
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE_MODEL:-../models/OLMo-2-0425-1B}"
NGPU="${NGPU:-8}"; BS="${BS:-32}"
CKDIR16=outputs/olmo2_probe2_1B_keep7fresh2_16card
CKDIR_E=outputs/olmo2_probe2_1B_keep7fresh2
PROG=logs/a03_newaxes_progress.log
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs a03_newaxes_results
note() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" >> "$PROG"; }

run_axis() {   # $1=arm $2=ckarg $3=axis-tag $4=val_path $5=expected_n $6=max_ctx
  local NAME="$1_$3" RD="a03_newaxes_results/$1_$3"
  note "START arm=$1 axis=$3 n_exp=$5 ctx=$6"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_paperC_squad_emf1.py \
      --base_model "$BASE" $2 --val_path "$4" \
      --num_shards $NGPU --shard_index $g --batch_size $BS \
      --max_new_tokens 32 --max_ctx_len "$6" --add_bos 0 \
      --results_root a03_newaxes_results --output_name "$NAME" \
      > "logs/a03_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RD"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "shards arm=$1 axis=$3 -> $ns/$NGPU"
  [ "$ns" -ne "$NGPU" ] && { note "ABORT arm=$1 axis=$3 incomplete $ns/$NGPU"; return 9; }
  $PY scripts/eval_paperC_squad_emf1.py --merge --results_root a03_newaxes_results \
      --output_name "$NAME" >> "logs/a03_${NAME}_merge.log" 2>&1
  # the harness has NO per-example merge and NO n assertion -> do both here, hard.
  $PY - "$RD" "$NGPU" "$5" <<'EOF' || return 9
import glob,json,os,sys
rd,ng,exp=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
sh=sorted(glob.glob(os.path.join(rd,f"shard*of{ng}.json")))
assert len(sh)==ng, f"{rd}: {len(sh)}/{ng} shards"
s=json.load(open(os.path.join(rd,"summary.json")))
assert s["n_shards"]==ng, s["n_shards"]
assert s["n"]==exp, f"{rd}: merged n={s['n']} != expected {exp}"
pe=sorted(glob.glob(os.path.join(rd,f"per_example_shard*of{ng}.jsonl")))
assert len(pe)==ng, f"{rd}: {len(pe)}/{ng} per-example files"
rows=[json.loads(l) for p in pe for l in open(p) if l.strip()]
assert len(rows)==exp, f"{rd}: per-example {len(rows)} != {exp}"
ids=[r["item_id"] for r in rows]
assert len(set(ids))==exp, f"{rd}: duplicate item_id"
rows.sort(key=lambda r: r["item_id"])
with open(os.path.join(rd,"per_example.jsonl"),"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print(f"OK {os.path.basename(rd)} n={s['n']} EM={s['em']:.4f} F1={s['f1']:.4f}")
EOF
  note "DONE arm=$1 axis=$3"
}

note "DRIVER START on $(hostname) ngpu=$NGPU bs=$BS"
rc=0
for pair in "A03N_base|" \
            "A03N_keep7_step200k|--ckpt $CKDIR16/step200000.pt" \
            "A03N_keep7_step500|--ckpt $CKDIR_E/step500.pt"; do
  arm="${pair%%|*}"; ck="${pair#*|}"
  run_axis "$arm" "$ck" hpqa_oracle   data/a03_newaxes/hotpot_oracle.jsonl   7405 1024 || rc=1
  run_axis "$arm" "$ck" hpqa_full     data/a03_newaxes/hotpot_full.jsonl     7405 3584 || rc=1
  run_axis "$arm" "$ck" hpqa_closed   data/a03_newaxes/hotpot_closed.jsonl   7405  512 || rc=1
  run_axis "$arm" "$ck" cf_new        data/a03_newaxes/cf_target_new.jsonl   2191  512 || rc=1
  run_axis "$arm" "$ck" cf_true       data/a03_newaxes/cf_target_true.jsonl  2191  512 || rc=1
done
note "DRIVER END rc=$rc"
exit $rc
```
Launch:
```bash
sshpass -f /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_82250.txt \
 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.82.250.82 \
 'W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory; cd $W
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # MUST be all 0 MiB first
  : > logs/a03_newaxes_progress.log
  PROJECT_ROOT=$W PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
    setsid nohup bash scripts/_run_a03_newaxes_82.sh \
    > logs/a03_newaxes_sched.out 2>&1 < /dev/null &
  sleep 5; echo "pid=$!"; cat logs/a03_newaxes_progress.log'
```
Watch (MAIN's 20-min rule): `tail -5 logs/a03_newaxes_progress.log` — the template's cadence
was a `note` line every ~1 min; here it is one per (arm x axis) = 15 phases, so a >6 min gap
on the `hpqa_full` phase is the only expected quiet period.

### Step 3 — floor analysis (CPU, ~3 min, LOCAL or .82 — both have scipy 1.18.0)
Needs the new sibling entry point (do NOT call `analyze_1b_knowledge_floor.main()`, it
asserts n==14042). New file
`proposal/active/A03-parametric-vs-external-memory/code/analyze_newaxes_floor.py` that
`import`s `best_constant_qa`, `lengthmatched_contains_null`, `paired_bootstrap`,
`mcnemar_exact`, `bh_reject`, `cell`, `score_prediction` from the existing module
(all verified importable + working on paperC-shaped rows) and:
  * reads `a03_newaxes_results/<arm>_<axis>/per_example.jsonl`
  * asserts n + item_id alignment + identical `gold` across arms (as `main()` does at 444-449)
  * computes em + contains + contains_lenmatched cells per (arm x axis)
  * **stratifies HotpotQA by `question_type`** (comparison best-constant EM 0.1567 vs
    bridge 0.0035 — measured; unstratified is misleading)
  * BH q=.05 across the whole new family
  * writes `evidence/a03_newaxes_floor_nulls.json`

## 8. IDEMPOTENCY
Result dirs are keyed by `<arm>_<axis>` under a NEW root `a03_newaxes_results/`, so nothing
collides with the passed gate's `olmo2_closedbook_results/` or `olmo2_mmlu_content_results/`.
Re-running overwrites `shard{i}of8.json` + `per_example_shard{i}of8.jsonl` in place; the
assertion block recomputes `per_example.jsonl` from scratch each time, so a partial previous
run cannot contaminate a rerun. Safe to re-launch after a failure without cleanup. The two
parquet downloads are content-addressed by filename — re-curl is a no-op cost (26 MiB).

## 9. FINAL NODE RECHECK (21:52)
`.82` 8/8 at 0 MiB. `.21` **now also 8/8 at 0 MiB** — the `SMOKE_llama2_7b` job (PID 38067)
finished/expired, so `.21` is fully free too. That does NOT change the recommendation:
`.21` still lacks scipy and still lacks the 1B ckpts (22.6 GiB cross-disk), so **this lane
belongs on `.82`** and `.21` should take a wzc1-resident job.

## 10. SUMMARY FOR MAIN
- **Verdict: READY_AFTER_SMALL_FIX, node `.82`, 8 GPUs, ~25-40 min.**
- The small fix is **2 new files, 0 edits to existing harnesses**:
  a converter (`build_a03_newaxes.py`, core validated against the real loader) and an
  analyzer entry point (`analyze_newaxes_floor.py`, all statistics imported from the
  existing verified module).
- **Key discovery that makes this cheap**: `scripts/eval_paperC_squad_emf1.py` is already a
  generic open-book harness with `--val_path`, sharing loaders + scorer + base protocol with
  the passed gate. `eval_olmo2_closedbook_qa.py` is NOT generic (hardcoded task dispatch,
  question-only prompt) — do not try to extend it.
- **Key defect to guard**: that harness has no per-example merge and no n/shard assertion.
  The driver above supplies both, hard.
- **2 of 3 axes tonight** (multi-evidence via HotpotQA-distractor 7405;
  conflicting-facts via CounterFact 2191, both fetched in seconds via proxy).
  **New-facts axis is BLOCKED_MISSING_ASSET** — needs a generated synthetic-injection set,
  nothing on either disk; should not gate the other two.
- **It cannot kill A03** (kill condition is conjunctive and already survived); it can retire
  individual axes and thereby set A03's scope before the 6-arm build.
