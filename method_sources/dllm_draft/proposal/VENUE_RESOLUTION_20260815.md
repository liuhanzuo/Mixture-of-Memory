# Venue resolution — `arXiv:2508.00819` (DAEDAL), 2026-08-15

## 1. The question

Two tracks run the same day disagreed about the venue of **one** paper:

> `arXiv:2508.00819` — *Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion
> Large Language Models* (Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang,
> Dahua Lin), a.k.a. **DAEDAL**.

| Track | File | Claim |
|---|---|---|
| **A** | `Mixture-of-Memory/proposal/backlog/B10-dllm-infilling-ar-dominance/S4_DISPOSITION.md:94-100` | **CoRR / arXiv-only**, DBLP `CoRR 2025`; explicitly noted OpenReview was unreachable (HTTP 403 `ChallengeRequiredError`) so an OpenReview-family acceptance *could not be excluded*; marked preprint-with-caveat |
| **B** | `dllm_draft/proposal/CLAUDE_FRONTIER_20260815.md:91` | **ICLR 2026 Poster**, `venueid = ICLR.cc/2026/Conference`, Submission1382, `Camera_Ready_Revision` present |

Per `memory/venue-verify-must-use-openreview-2026.md`, this is an **OpenReview-family** venue
claim, so OpenReview `venueid` + presence of `Camera_Ready_Revision` is the authority — **not**
Semantic Scholar, **not** DBLP (DBLP lags months behind OpenReview for 2026 conferences), and
**not** the arXiv `comment` field.

## 2. Authorities reached / refused

All queries from LOCAL (wzc1) with `http_proxy=https_proxy=http://hy-proxy.woa.com:3128`.

| Endpoint | Result |
|---|---|
| `api2.openreview.net/notes?...` (any `content.*` / `invitation` filter) | ❌ **HTTP 403** `{"name":"ChallengeRequiredError","message":"Challenge verification required (2026-08-15-5452855)","status":403,...}` — reproduces Track A's exact failure, incl. with a browser UA + `Referer` |
| `api2.openreview.net/notes/search?term=...` | ✅ **HTTP 200** — *this path is **not** challenge-gated.* **This is what settled it.** |
| `openreview.net/forum?id=Ic2A2gCseC` (HTML) | ⚠️ HTTP 307 redirect, 52 KB body, no venue string extractable (Next.js client-render) — not usable as authority |
| `api.openreview.net` (v1) `/notes/search` | ✅ 200, but returns only DBLP-mirror records; **0 title hits** for DAEDAL. *A v1-only pass is exactly how one would wrongly conclude "arXiv-only".* |
| `export.arxiv.org/api/query` (metadata) | ✅ 200 (needs **https**; plain `http://export.arxiv.org` → 301 with 0 bytes) |
| `arxiv.org/html/2508.00819v2` | ✅ 200, 247 058 B (full text) |

**Method note for future passes:** `api2` `/notes?` is challenge-gated but `api2` `/notes/search?`
is **not**. When `/notes?` returns 403, do **not** conclude "OpenReview unreachable" — retry
`/notes/search?term=<title words>&source=forum&limit=100` and filter client-side. The
`invitations[]` array on the returned note carries `Camera_Ready_Revision`.

## 3. Verdict — **Track B is RIGHT. DAEDAL is ICLR 2026 Poster.**

Verbatim from `https://api2.openreview.net/notes/search?term=Beyond%20Fixed%20Variable-Length%20Denoising%20Diffusion&source=forum&limit=100`, note `id = forum = Ic2A2gCseC`:

```json
"title":   "Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models",
"authors": ["Jinsong Li","Xiaoyi Dong","Yuhang Zang","Yuhang Cao","Jiaqi Wang","Dahua Lin"],
"venue":   "ICLR 2026 Poster",
"venueid": "ICLR.cc/2026/Conference",
"invitations": [
  "ICLR.cc/2026/Conference/-/Submission",
  "ICLR.cc/2026/Conference/-/Post_Submission",
  "ICLR.cc/2026/Conference/Submission1382/-/Full_Submission",
  "ICLR.cc/2026/Conference/Submission1382/-/Rebuttal_Revision",
  "ICLR.cc/2026/Conference/-/Edit",
  "ICLR.cc/2026/Conference/Submission1382/-/Camera_Ready_Revision"
]
```

`Camera_Ready_Revision` **present**; `Submission1382` matches Track B exactly. Author list matches
the arXiv record verbatim (6/6), so this is the same paper, not a title collision.

Official bibtex from the same note (**use this, not a DBLP `@article`**):

```bibtex
@inproceedings{li2026beyond,
  title={Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models},
  author={Jinsong Li and Xiaoyi Dong and Yuhang Zang and Yuhang Cao and Jiaqi Wang and Dahua Lin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=Ic2A2gCseC}
}
```

Note dates: `cdate` 2025-09-03 (submitted), `odate` 2025-10-08, `pdate` **2026-01-26** (decision
published). `paperhash = li|beyond_fixed_trainingfree_variablelength_denoising_for_diffusion_large_language_models`.

**Why Track A got it wrong, and why it was not negligent.** Track A's two *stated observations were
both factually correct*: (i) DBLP does say `CoRR 2025`, and (ii) arXiv `comment` is only
`"Code is available at https://github.com/Li-Jinsong/DAEDAL"` with **no** acceptance note and
**no** `journal_ref` — confirmed independently this pass. arXiv v2 is dated 2025-08-18, i.e. it
predates the 2026-01-26 decision and was never refreshed. So every non-OpenReview authority still
reads "preprint"; **only OpenReview knows**. Track A correctly refused to assert absence and
flagged the caveat — its error was concluding "unreachable" from the 403 on `/notes?` without
trying `/notes/search?`. Track A's disposition is amended, not faulted.

> ⚠️ Corollary: Track B's frontier table has **11 other rows whose "preprint (`CoRR 20xx`)" venue
> came from DBLP-via-OR** (`iLLaDA` 2606.25331, `LLaDA-MoE` 2509.24389, `Dream-Coder` 2509.01142,
> **`ρ-EOS` 2601.22527**, `CAL` 2602.00476, `LR-DLLM` 2602.07546, `ELF` 2605.10938,
> `LLaDA-MoE-v2` 2608.03457). DAEDAL shows a DBLP `CoRR` record **coexists** with an accepted
> ICLR-2026 record. Those rows are `NOT-FOUND`, not `IS-A-PREPRINT`, until each is re-run through
> `api2 /notes/search`. Not done in this pass — out of scope.

## 4. Numbers check — both tracks' DAEDAL table is **correct**, and the sweep IS a BASELINE

Source: `arxiv.org/html/2508.00819v2`, **Table 1**. Verbatim caption:

> *"Table 1: Main Results of DAEDAL on LLaDA-Instruct-8B. We compare the **baseline performance at
> various generation lengths (64 to 2048)** against DAEDAL. … **The best configuration for the
> baseline is highlighted in orange**."*

The swept columns sit under the verbatim column-group header **`Fixed-Length Denoising (Baseline)`**,
with a *separate single* `DAEDAL` column at `L_init=64`. Body text: *"…comparing the **Fixed-Length
Denoising b[aseline]**…"*; *"we report the **pass@1** metric"* for MBPP/HumanEval.

**→ The sweep is over the fixed-length LLaDA BASELINE, not over DAEDAL's own variants. The
preemption is the strong reading.** (DAEDAL's *own* initial-length ablation is a different table,
Table 4, and is not the sweep either track quoted.)

Table 1 — **LLaDA-Instruct-8B**, baseline columns 64/128/256/512/1024/2048, `Acc` rows:

| Benchmark | 64 | 128 | 256 | 512 | 1024 | 2048 | (DAEDAL @64) |
|---|---|---|---|---|---|---|---|
| MBPP | **20.8** | 28.0 | 37.4 | 38.2 | 37.4 | **38.8** | 40.8 |
| HUMANEVAL | **18.9** | 26.2 | 36.0 | 47.0 | **47.6** | 47.0 | 48.2 |

Both tracks' rows reproduce **byte-for-byte**, all 12 cells. The queried pair
"HumanEval 18.9 → 47.6 (64→1024)" and "MBPP 20.8 → 37.4 (…→256), or 38.8 (→2048)" both check out —
the `37.4` / `38.8` ambiguity is real and benign: MBPP is **non-monotone** (37.4 @256, 38.2 @512,
37.4 @1024, 38.8 @2048), so "→37.4" and "→38.8" are different endpoints of the same row, not a
discrepancy.

**Models: BOTH.** Table 1 = `LLaDA-Instruct-8B`; **Table 2 = `LLaDA-1.5-8B`** (same design). Table 2
`Acc`: MBPP `20.6 30.2 39.2 38.6 39.8 39.6` → DAEDAL `40.2`; HUMANEVAL `18.3 22.0 37.8 45.1 49.4
50.0` → DAEDAL `48.8`. (Strings `LLaDA-8B-Instruct` / `LLaDA-1.5` as written in our notes do not
appear in the paper; the paper's own names are `LLaDA-Instruct-8B` and `LLaDA-1.5-8B`.)

## 5. ρ-EOS spot-check (`arXiv:2601.22527`) — **EXISTS, and the sweep replicates**

arXiv API, `id_list=2601.22527` → `http://arxiv.org/abs/2601.22527v2`:

- Title: **`$ρ$-$\texttt{EOS}$: Training-free Bidirectional Variable-Length Control for Masked
  Diffusion LLMs`**
- Authors: Jingyi Yang, Yuxian Jiang, Jing Shao · updated 2026-02-07 · comment `"11 pages,6
  figures,6 tables"` · no `journal_ref`

Its Table 1 / Table 2 use the **same** column-group header `Fixed-Length Denoising (Baseline)` and
add `DAEDAL` + `ρ-EOS (Sym)` + `ρ-EOS (Asym)` columns. Track A's two cited MBPP sweeps both verify:

| Table (caption verbatim) | MBPP baseline `Acc` 64→2048 | Track A cited |
|---|---|---|
| **Table 1**, *"Main Results on **LLaDA-Instruct-8B** across Four Benchmarks"* | `21.0 28.8 **36.7** 38.7 37.5 38.8` | 21.0→36.7 ✅ |
| **Table 2**, *"Main Results on **LLaDA-1.5-8B** across Four Benchmarks"* | `21.2 30.4 **39.2** 38.8 39.4 39.4` | 21.2→39.2 ✅ |

So ρ-EOS is a **second, independent** replication of baseline canvas sensitivity on LLaDA
checkpoints (its HumanEval baselines move 17.1→48.2 and 17.6→49.4 respectively). Its own venue was
**not** re-verified this pass — see the §3 corollary; it remains `NOT-FOUND`, not confirmed-preprint.

## 6. Bottom line

- **Verdict: `arXiv:2508.00819` = ICLR 2026 Poster** (`ICLR.cc/2026/Conference`, Submission1382,
  `Camera_Ready_Revision` present). **Track B right, Track A amended.** Cite `@inproceedings`, ICLR
  2026 — a `CoRR 2025` cite for DAEDAL is now wrong.
- **The preemption is the strong reading, not the weak one**: the 64→2048 sweep is over the
  **fixed-length baseline**, on **two** LLaDA checkpoints, replicated by a second paper (ρ-EOS).
  And it is now a **peer-reviewed, accepted** result, which *raises* its preemptive weight over
  what Track A assumed.
- Neither track fabricated anything: 12/12 DAEDAL cells and 12/12 ρ-EOS cells verified against the
  papers' own HTML.

*Files patched with dated corrections (originals preserved, nothing deleted):*
`Mixture-of-Memory/proposal/backlog/B10-dllm-infilling-ar-dominance/S4_DISPOSITION.md` (Track A, the
wrong one) and this file. Track B's `CLAUDE_FRONTIER_20260815.md:91` needs **no** change —
confirmed as written.
