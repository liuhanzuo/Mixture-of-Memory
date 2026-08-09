# Sources

## Internal — Paper B provenance and known defects

- `../../../paperB/sections/03_method.tex`
- `../../../paperB/sections/04_experiments.tex`
- `../../../paperB/sections/06_limitations.tex`
- `../../../status/PAPERB_TWO_CORPORA_DEFECT.md` — the two-corpora defect that motivates the
  `STATUS.json:warning`; re-verified 2026-08-09 from `ls -l` byte counts on both disks.
- `../../../status/PAPERB_DIFFERENTIAL_LR_NEVER_ACTIVE.md` — differential LR was never active in any
  historical arm; do not claim it.
- `../../../status/PAPERB_RESUME_WARM_RESTART_DEFECT.md` — why no A04 arm may resume.
- `../../../status/PAPERB_P12_SEED2.md` — §1.1/§1.2/§1.3: `--seed` postdates the original runs, and
  it controls fresh-tail init only, **not** data order.

## Internal — trainer and eval harness (A04 reuses these verbatim)

- `../../../scripts/train_olmo2_arch_probe2.py` — `--keep_front_layers`, `--n_fresh_layers`,
  `--random_trunk` (line 586), `--from_scratch` (line 584, mutually exclusive), `--seed` (637),
  `--milestone_every` (616). **Line 863 `DistributedSampler(ds, shuffle=True)` has no `seed=`** —
  the blocking defect in `STATUS.json:blocked_by`.
- `../../../scripts/train_olmo2_shortgpt.py`
- `../../../scripts/eval_olmo2_probe2_ppl.py` — `load_pruned_model` (71), `load_base_model` (125),
  `load_base_model_any_family` (143), `load_truncated_any_family` (175).
- `../../../scripts/eval_olmo2_mmlu_content.py`
- `../../../scripts/eval_olmo2_closedbook_qa.py` — popqa / triviaqa / nq_open.
- `../../../scripts/_run_a03_1b_floor_82.sh` — **the shard-completeness assertion pattern A04 must
  reuse** (asserts `n_shards==8`, `MMLU n==14042`, `n_valid+n_nan==exp`, `v["n"]==e` before merge).
- `../../../scripts/eval_olmo2_activation_patching.py`

## Internal — nulls and statistics conventions A04 inherits (cite, do not re-claim)

- `../A01-null-calibration-methodology/PROPOSAL.md` + `STATUS.json` — construct-appropriate
  best-constant nulls; MMLU letter null is **always-D 0.2689**, never 0.25; MMLU content null is
  longest-option split-tie **0.2845**.
- `../A03-parametric-vs-external-memory/evidence/a03_1b_floor_nulls.json` — **the 1B nulls A04 uses
  verbatim**: TriviaQA EM 0.0025635, PopQA EM 0.0229200, MMLU-content 0.2844502.
- `../A03-parametric-vs-external-memory/GATE_FOURAXES_VERDICT.md` — NQ-open EM null **0.0055**
  (canonical; 0.0053 must not be quoted), plus intact/pruned/barely-healed 1B residual table.
- `../../../status/A03_1B_FLOOR_VERDICT.md` — 1B knowledge axes are measurable (K3 provisionally
  cleared); MMLU-letter must be dropped at 1B; `contains` needs a length-matched null.
- `../../../status/scout_21/lane2_a01_gate2.md` — nulls if the gate is ever widened: BoolQ
  **0.6217 (always-B)** not 0.50; OpenBookQA longest-option **0.3635** not 0.25; winogrande is
  structurally degenerate (control only).
- `../../backlog/B04-eval-fragility-incubator/DIRECTION_A_QWEN_VERDICT.md` — **why A04's primary
  statistic is not a rank correlation over rungs**: the n=6 exact-permutation floor
  (2/720 = 0.002778) was hit twice on OLMo-2-7B and the claim still died on Qwen3-8B.

## Internal — measured cost and Pilot Zero evidence (on disk, verified 2026-08-09)

- `zwfy6:logs/olmo2_1B_keep7fresh2_1node.log` — 1B on 8×H20, median **2.02 s/step** (n=36),
  header `world_size=8 bs=16 gaccum=1 eff_bs=128 seq_len=2048`.
- `zwfy6:logs/olmo2_1B_keep7fresh2_16card_node0.log` — 1B on 16×H20, median **1.48 s/step**
  (n=10,000); wall 2026-07-16 22:40:29 → 2026-07-20 11:18:11 = **84.63 h / 200,000 steps**.
- `zwfy6:olmo2_ppl_results/1B_base_full/summary.json` — intact 1B PPL **10.6416**.
- `zwfy6:olmo2_ppl_results/1B_keep7_step{50000,100000,147000,200000}/summary.json` —
  **17.6194 / 16.1613 / 15.6285 / 15.4116**, all `n_shards=8`, `n_tokens=8384512`,
  `n_windows=4096`. The PPL trajectory Pilot Zero uses.
- `zwfy6:olmo2_closedbook_results/A03_1B_keep7_step200k/` and
  `zwfy6:olmo2_mmlu_content_results/A03_1B_keep7_step200k/` — per-example + per-shard jsonl for the
  capability side; Pilot Zero's disagreement analysis is pure CPU on these.
- `zwfy6:outputs/olmo2_probe2_1B_keep7fresh2_16card/step{50000,100000,150000,200000}.pt`, `final.pt`;
  `zwfy6:outputs/olmo2_probe2_1B_keep7fresh2/step500.pt`;
  `zwfy6:outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k/step{205000,210000,215000,220000}.pt`;
  `zwfy6:outputs/olmo2_probe2_1B_keep7f2_dolmino_arm4_peaklr20k/step{205000,210000,215000}.pt`.
- Base model `models/OLMo-2-0425-1B` — present on **both** disks;
  `num_hidden_layers=16, hidden_size=2048, vocab_size=100352, num_attention_heads=16`.
- `data/dolmino_now15b.npy` — **62,020,903,040 B on wzc1** (7,570,911 rows) vs
  **126,907,244,672 B on zwfy6** (15,491,607 rows), ratio **2.0462×**. Same name, different file.

## External primary sources — fetched and venue-verified 2026-08-09

Full per-paper overlap/gap/must-not-claim analysis in `RELATED_WORK.md`. Venue authority per repo
rule: OpenReview `venueid` for ICLR/NeurIPS/ICML/TMLR; ACL Anthology + DBLP for the ACL family.

| arXiv | Title | Verified venue | Verified by |
|---|---|---|---|
| 2606.14150 | Small LLMs: Pruning vs. Training from Scratch | preprint (CoRR 2026) | DBLP `journals/corr/abs-2606-14150` + OpenReview absence; S2 429-limited |
| 2607.00368 | Beyond Perplexity: Behavioral Evaluation Framework for Deployment-Memory Claims in LLM TTT | preprint (CoRR 2026) | DBLP record + S2 (`venue: ""`) |
| 2602.01997 | On the Limits of Layer Pruning for Generative Reasoning in LLMs | **ACL ARR 2026 May, UNDER REVIEW** — not published | DBLP CoRR 2026 + OpenReview note `LUBdOuX62N` whose own bibtex says `note={under review}` |
| 2508.13533 | Compressed Models are NOT Trust-equivalent to Their Large Counterparts | preprint (CoRR 2025) | DBLP record + OpenReview absence |
| 2601.22950 | Perplexity Cannot Always Tell Right from Wrong | **ICML 2026 Workshop CTB SUBMISSION** (not accepted) | OpenReview `venueid=ICML.cc/2026/Workshop/CTB/Submission` + DBLP |
| 2605.15491 | Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs | **ICML 2026 Workshop AdaptFM, Poster** | OpenReview note `U37zxXMEE0`, `venueid=ICML.cc/2026/Workshop/AdaptFM`. S2 says "arXiv.org" — the documented S2 lag |
| 2510.15304 | Layer as Puzzle Pieces (CoMe) | **NeurIPS 2025, Poster** | OpenReview note `enhFXzKii4` + `Submission22642/-/Camera_Ready_Revision` |
| 2606.03002 | Perplexity Can Miss SAE Feature Damage Under Quantization | **Under review for TMLR** | OpenReview `venueid=TMLR/Under_Review` + DBLP |
| 2403.17887 | The Unreasonable Ineffectiveness of the Deeper Layers | **ICLR 2025, Poster** | OpenReview `venueid=ICLR.cc/2025/Conference` + `Submission13737/-/Camera_Ready_Revision` |
| 2407.14679 | Compact Language Models via Pruning and Knowledge Distillation (Minitron) | **NeurIPS 2024, Poster** | OpenReview `venueid=NeurIPS.cc/2024/Conference` + `Submission15087/-/Camera_Ready_Revision` |
| 2402.02834 | Shortened LLaMA | **ICLR 2024 Workshop ME-FoMo, Poster** (title differs arXiv vs venue record) | OpenReview `venueid=ICLR.cc/2024/Workshop/ME-FoMo` + DBLP CoRR 2024 |
| 2604.14419 | Equifinality in Mixture of Experts | preprint — the only ML-adjacent TOST hit found; unrelated object (MoE routing topology at 76–84M) | arXiv metadata |
| 2408.11796 | LLM Pruning and Distillation in Practice (Minitron Approach) | **UNVERIFIED** — not checked | — |

Endpoints used (all through `hy-proxy.woa.com:3128`), with their failure modes recorded for the next
agent in `STATUS.json:ops_notes_for_the_next_agent`:

- `https://export.arxiv.org/api/query?id_list=<id>` — **needs a non-default `User-Agent`**, else
  returns an empty body.
- `https://api2.openreview.net/notes/search?term=<title>&content=all`
- `https://dblp.org/rec/journals/corr/abs-<id>.bib` — **`/search/publ/api` was HTTP 500 all
  session**; per-record `.bib` works.
- `https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>` — 429s after ~3 calls; cross-check
  only, never the venue authority.
- `https://aclanthology.org/search/` — client-side rendered, **not scrapable**.
