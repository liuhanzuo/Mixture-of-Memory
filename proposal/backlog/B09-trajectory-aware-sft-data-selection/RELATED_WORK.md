# B09 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU. Adjudication + venue-verification pass only — this file runs nothing
and authorises nothing.**

This closes the blocker `proposal/ready_queue.py` actually trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`) and discharges the task the
`RELATED_WORK_GAP_AUDIT_20260808.md:99` row assigns to B09:

> `| B09 trajectory-aware SFT selection | **较充分** | offline RL/imitation coreset；laminar/group
> constraints；verified decision credit；完整组合 collision | 主要继续补 SOURCES.md；发现直接
> collision 时再收窄。|`

B09 is the audit's **best-rated** proposal (`较充分` — reasonably sufficient), and the audit's
instruction is *not* "go do a novelty check": `NOVELTY.md` (2026-08-08, 16.7 KB) already did one and
already found a complete component-wise collision set. The assignment is (a) turn that into a
**named, venue-verified** collision table with first-hand authorities, (b) close the four
`SOURCES.md` "claims requiring further source audit" items, and (c) **narrow on contact** if a
direct combination collision turns up. One did — see §3.1 — so §5 narrows.

---

## 0. ⚠️ Read this before treating any experiment below as runnable

**B09's `next_gate` is `GATE -1`, and it means the candidate pool does not exist.**
`STATUS.json.blocker_2026_08_10` and `DATA_AUDIT_VERDICT_20260810.md` record that the
`|G| ≈ 10,000` agent trajectories / `|U| ≈ 100,000` derived SFT rows that `PROPOSAL.md` §1 treats as
a *visible existing asset* (its own line 168: `Candidate U | 100K 候选池 | 可见`) were searched for
on **both physical disks** and **are not there**. In this repository "trajectory" has always meant
*checkpoint-over-training-steps*, never agent rollout.

Therefore this document is a **literature boundary for a design**, not for results. Nothing here may
be read as "B09's pilot is ready". Every hypothesis H1–H5 in `PROPOSAL.md` §9 is downstream of
acquiring or generating a trajectory-structured pool. The one thing this file legitimately changes is
that after it, the remaining blocker is **data**, not **paperwork** — which is the honest state.

A second consequence, and it is a *positive* one: because no data exists, **narrowing costs nothing
right now.** Every narrowing in §5 is free if adopted before acquisition, and expensive if adopted
after a 100K-row pool has been built to the wrong spec.

---

## 1. What B09 claims RIGHT NOW

Read from `PROPOSAL.md` §"一句话主张" + `NOVELTY.md` §4.1, not reconstructed:

> When ~10K agent trajectories are expanded into ~100K SFT rows, the best 5K subset is not the Top-K
> of an independent per-row score. B09 studies the **two-level (parent, child) joint selection
> problem** that expansion creates, and asks whether **hierarchical grouping + set coverage under
> parent-multiplicity / benchmark / assistant-target-token constraints** dominates flat per-row
> quality ranking — with a **constraint-matched random** null strong enough to prove any gain is not
> just sibling de-duplication or benchmark quota.

Three properties of that claim are load-bearing for everything below:

1. It is a **problem formalisation + controlled empirical** claim, not an algorithmic-primitive
   claim. `NOVELTY.md` §5 already forbids 14 "first to…" formulations. This file does not re-open
   them and adds two more (§4).
2. Its **primary comparator is a null, not a method** (`PROPOSAL.md` §8: "主 null baseline 不是
   row-random"). That is unusual in this literature and is where the residual novelty actually lives.
3. Its unit of analysis is a **paired end-to-end SFT seed**, not a row (`PROPOSAL.md` §11: "不能把
   100K 派生 rows 当成独立样本").

---

## 2. Standing rules this adjudication obeys

**`memory/prior-work-differentiate-dont-abandon.md`** (user, 2026-08-07): the bar for preemption is
**完全相同 / 抄袭** — identical or plagiarised — **not overlap**. Work within 2–3 months is
**concurrent** and cannot preempt. When a close work exists, the required response is
differentiation or a follow-up that fixes a defect in it. **A direction dies from its own kill gate
(`PROPOSAL.md` §12), never from a literature count.** Nothing in this file kills B09; §0 and §5 are
about *scope* and *data*, and §6's verdict is `hold_in_backlog`, not `archive`.

**Venue verification by family**, per `memory/venue-verify-must-use-openreview-2026.md` and
`memory/venue-verify-acl-family-needs-anthology.md`:

| Family | Authority used, first-hand, this session |
|---|---|
| ICLR / NeurIPS / ICML / TMLR | OpenReview `venueid` (+ `Camera_Ready_Revision` invitation as the accept signal) |
| ACL / EMNLP / NAACL **including Findings** | ACL Anthology page (`Anthology ID` + `Venue` metadata block) **and** DBLP `conf/*` record |
| everything else | DBLP; `arXiv-only` = *I could not verify a peer-reviewed venue from this node* |

Endpoints reached on 2026-08-15 from this node: `export.arxiv.org/api/query` (requires a non-default
`User-Agent`), `dblp.org/search/publ/api` **and** `dblp.org/rec/<key>.bib`, `aclanthology.org/<id>/`,
`api2.openreview.net/notes/search`, `api.github.com`. Semantic Scholar was **not** used as a venue
authority (repo rule). Failures are in §7.

⚠️ **This pass found the ACL-family rule to be load-bearing exactly as the memory says.** Three of
B09's four closest collisions — ATLAS, MDS, CSO — are returned by DBLP's `journals/corr/` record and
by the arXiv API as **CoRR / preprint**, and are in fact **published Findings papers**. Anyone
auditing B09 through the CoRR record alone would under-rate its three sharpest collisions.

---

## 3. Named closest collisions (venue verified this session)

Ordered by threat. "Threat" = how much of B09's claim it removes, not topical similarity.

### 3.1 Weasel — **THE NEW DIRECT COLLISION, and the reason §5 narrows**

* **Cite**: Pesaran Zadeh, Fatemeh et al., *Weasel: Out-of-Domain Generalization for Web Agents via
  Importance-Diversity Data Selection*, `arXiv:2605.20291` (v2, 2026-05-19).
* **Venue, verified**: **ICML 2026** — OpenReview `venueid = ICML.cc/2026/Conference`,
  `venue = "ICML 2026 regular"`, forum `EXCUyr6hhZ`, `Submission26224/-/Camera_Ready_Revision`
  present. Also has an ICLR 2026 LLA workshop record (`venueid = ICLR.cc/2026/Workshop/LLA`, forum
  `ixNDssFCkd`) — the ICML main-conference record is the one to cite.
  **DBLP has it only as `journals/corr/abs-2605-20291` (CoRR 2026)** — a second instance of the
  S2/DBLP conference lag the repo rule warns about.
* **What it does**: selects a **fixed-budget subset of trajectory *steps*** for offline training of
  web agents, by maximising an objective combining **unary importance with pairwise diversity over
  states, websites, and interaction patterns**, solved **greedily**. Evaluated for **out-of-domain**
  generalisation (train AgentTrek/NNetNav → eval WebArena/WorkArena/MiniWob) across three student
  families (Qwen2.5-7B, Gemma3-4B, Qwen3-8B), reporting 9.7–12.5× training speedups.
* **Why this is the sharpest collision B09 has ever had.** It is not merely "trajectory-aware
  selection". Matched against `NOVELTY.md` §3's own component matrix, Weasel independently occupies
  **four** of the six columns *simultaneously* and at the **step** granularity B09 claims:
  * *fixed budget over steps* → B09's `|S| = 5000` count-controlled track;
  * *unary importance + pairwise diversity, greedy* → B09's `λ_q Σq_i + λ_p F_pool(S)` Stages 2/4/6;
  * *diversity over websites and interaction patterns* → B09's Stage 5 `F_meta` metadata coverage
    (benchmark / tool family / task family are the same construct under different names);
  * *strict OOD read-out across three students* → B09's §6 "Strict OOD / generalist" weight family
    **and** its §9 H3 prediction, on the same axis.
* **Precisely what Weasel does not do, checked against its own abstract, not assumed**:
  1. **No parent-multiplicity constraint.** It selects steps under a *global* budget with pairwise
     diversity. Nothing enforces `|S ∩ U_g| ≤ m_g`. Pairwise diversity *discourages* sibling
     redundancy but does not *bound* it, and B09's whole H1 is about the difference between a soft
     penalty and a hard partition cap — a difference which is only visible under B09's
     `m_g ∈ {1, 2, ∞}` ablation.
  2. **No target-token knapsack.** Its budget is *steps*, and its efficiency claim is *training
     speedup*. B09's `Σ c_i ≤ B` fixed-assistant-target-token track is the axis on which "the gain is
     just more tokens" is falsifiable, and it is B09's own kill criterion #5.
  3. **No decision-type taxonomy / criticality.** B09's Stage 1 eight-class taxonomy (PLAN,
     CRITICAL_OBSERVATION, …, FORMAT_ONLY) and its band-pass learnability are absent.
  4. **No constraint-matched random null.** This is the decisive one. Weasel's contribution is a
     *method that wins*; B09's contribution is *whether the method beats a null that already has the
     cap, the quota, the length bins and the token budget*. Nothing in Weasel's design answers that
     question, and per `RDS+`'s own finding (§3.5) the answer is frequently "no".
  5. **One task family (web/GUI agents), AXTree states.** B09 spans benchmark families with
     tool-semantics canonicalisation.
* **Verdict: overlapping, peer-reviewed, NOT preempting — but it forces a narrowing.** It is 2026-05,
  three months before this pass, so it is **not** concurrent and cannot be waved off on that basis.
  It **removes** "importance + diversity over agent trajectory steps under a fixed budget" from
  B09's claimable space entirely. What remains claimable is in §5, and it is narrower than
  `NOVELTY.md` §4.1's wording.

### 3.2 TopoCurate — whole-trajectory tool-use selection

* **Cite**: *TopoCurate: Modeling Interaction Topology for Tool-Use Agent Training*,
  `arXiv:2603.01714` (2026-03-02).
* **Venue**: **arXiv-only.** DBLP total = 1, `journals/corr/abs-2603-01714` (CoRR 2026). OpenReview
  search returned **no titled note**. Not verifiable as peer-reviewed from this node.
* **Overlap / gap**: as `NOVELTY.md` §1.1 records — semantic quotient topology over multi-trial
  rollouts, Reflective Recovery / Semantic Efficiency / Strategic Diversity, selects **whole
  trajectories**. It does not do child-row caps, within-trajectory decision selection, independent
  target-query relevance, or a token knapsack.
* **Threat: Critical for the framing, weaker than §3.1 for the mechanism.** It kills "first
  trajectory-aware agent data selection" (already forbidden). Being arXiv-only, it must be cited as
  a preprint and cannot be leaned on as settled.

### 3.3 MDS — multi-turn rows are not independent

* **Cite**: Li, Bo; Zhang, Shikun; Ye, Wei. *Data Selection for Multi-turn Dialogue Instruction
  Tuning*, `arXiv:2604.07892` (v3).
* **Venue, verified**: **Findings of ACL 2026** — Anthology `2026.findings-acl.130`
  (page HTTP 200, `Anthology ID: 2026.findings-acl.130`), DBLP `conf/acl/LiZY26`, pp. 2724–2739,
  DOI `10.18653/v1/2026.findings-acl.130`. The paper's own `journal_ref` also states
  "Findings of ACL 2026". **Not** the CoRR-only record DBLP's `journals/corr/` entry suggests.
* **Overlap / gap**: whole-dialogue global semantic coverage + local structural quality under a
  dialogue budget. No agent action/observation/state, no within-dialogue critical-turn selection, no
  child-row cap, no target-query relevance, no token knapsack.
* **Threat: Critical to one sentence.** B09 may **not** claim to have discovered that multi-turn data
  should not be scored as independent turns. MDS owns that, at a published ACL-family venue.

### 3.4 ATLAS and CSO — within-trajectory critical steps, and *verified* credit

* **ATLAS**: Chen, Zhixun; Li, Ming; Huang, Yuxuan; Du, Yali; Fang, Meng; Zhou, Tianyi.
  *ATLAS: Agent Tuning via Learning Critical Steps*, `arXiv:2503.02197`.
  **Venue, verified: Findings of ACL 2025** — Anthology `2025.findings-acl.1299` (HTTP 200), DBLP
  `conf/acl/ChenLH0F025`, pp. 25334–25349, DOI `10.18653/v1/2025.findings-acl.1299`.
  ⚠️ `SOURCES.md` cites this as `arXiv:2503.02197 / Findings of ACL 2025` with **no Anthology ID**;
  the ID is now pinned. Note the arXiv title capitalises `ATLaS` while the Anthology/DBLP record
  reads `ATLAS` — the exact kind of title-vs-metadata mismatch
  `memory/venue-verify-acl-family-needs-anthology.md` warns about. **Cite the Anthology spelling.**
* **CSO**: *Verified Critical Step Optimization for LLM Agents*, `arXiv:2602.03412`.
  **Venue, verified: Findings of ACL 2026** — Anthology `2026.findings-acl.1974` (HTTP 200), DBLP
  `conf/acl/LiZFLSLMY26`, pp. 39627–39639, DOI `10.18653/v1/2026.findings-acl.1974`.
  ⚠️ `SOURCES.md`/`NOVELTY.md` call it "arXiv:2602.03412, 2026" with no venue. It **is** published.
* **Overlap / gap**: ATLAS = critical-step loss masking with the full causal prefix retained, ~30 %
  of steps, benchmarked against full/random/PPL step selection. CSO = PRM-proposed candidate steps,
  expert alternative actions, continued rollout, **keeps only decisions that flip failure→success**,
  then targeted DPO.
* **Threat: Critical + High.** ATLAS is a mandatory *baseline*, not a motivation. CSO owns
  **outcome-flip verified decision credit** — so B09's `k_i` may only ever be called *criticality
  candidacy / heuristic decision importance / proxy decision credit*, and `NOVELTY.md` §6 item 2
  ("branch-verified decision credit") is now **a re-implementation of a published ACL-family method
  inside B09**, not a novel core. That materially weakens §6's escape route; see §5.3.

### 3.5 RDS+ — target-aware selection, **and the reason B09's null matters**

* **Cite**: Ivison, Hamish et al., *Large-Scale Data Selection for Instruction Tuning*,
  `arXiv:2503.01807` (v2).
* **Venue**: **arXiv-only.** DBLP total = 1, `journals/corr/abs-2503-01807` (CoRR 2025). No
  Anthology/OpenReview record found from this node.
* **Overlap / gap**: position-weighted hidden-state pooling, target-query representation similarity,
  round-robin target-aware selection. **Its headline finding is that many sophisticated selectors
  fail to beat random as the pool grows.**
* **Threat: High, and dual-signed.** It removes "target-query embedding relevance / query coverage
  is new". But it is also **B09's strongest scientific justification**: a literature in which
  complex selectors routinely lose to random at scale is exactly a literature that needs a
  constraint-matched null and paired-seed statistics. B09 should cite RDS+ *for* its design, not
  merely concede to it.

### 3.6 Target-aware submodular / budgeted coverage — three published owners

| Cite | Venue, verified | Owns |
|---|---|---|
| Agarwal et al., *DELIFT: Data Efficient Language Model Instruction Fine-Tuning*, `arXiv:2411.04425` | **ICLR 2025** — DBLP `conf/iclr/AgarwalK0D25`, OpenReview forum `Fty0wTcemV` | ICL-based pairwise utility + **facility location and facility-location mutual information**, incl. a *task-specific* objective that already joins target relevance with coverage |
| Renduchintala et al., *SMART: Submodular Data Mixture Strategy for Instruction Tuning*, `arXiv:2403.08370` | **Findings of ACL 2024** — DBLP `conf/acl/RenduchintalaBR24`, `2024.findings-acl.766`, pp. 12916–12934 | submodular **task selection + task-budget allocation + within-task instance selection** (facility location, log-determinant) |
| Chen et al., *MIG: … Maximizing Information Gain in Semantic Space*, `arXiv:2504.13835` | **Findings of ACL 2025** — DBLP `conf/acl/ChenLHMYC25`, `2025.findings-acl.515`, pp. 9902–9915 | quality-weighted label-graph information gain with **diminishing marginal returns** |

**Threat: High, collectively fatal to one whole layer of B09.** B09's Stages 3/4/5 —
`F_target`, `F_pool`, `F_meta`, and the `log(1 + n_h/τ_h)` diminishing-returns form — are, as a
*family of objectives*, fully owned here. ⚠️ `SOURCES.md` lists MIG and SMART without venues; both
are **published ACL-family papers**, and MIG's diminishing-marginal-coverage form is the closest
published analogue of B09's `F_meta`.

### 3.7 Baseline anchors that must be cited (not collisions, but non-negotiable)

All venues verified first-hand this session via DBLP `conf/*` records and/or arXiv camera-ready
comments:

| Cite | Venue, verified | Role in B09 |
|---|---|---|
| Xia et al., **LESS**, `arXiv:2402.04333` | **ICML 2024** — DBLP `conf/icml/XiaMGA024` | B09's §7 low-training upper bound; not novelty |
| Li et al., **IFD** (*From Quantity to Quality*), `arXiv:2308.12032` | **NAACL-HLT 2024** — DBLP `conf/naacl/LiZLCC0W0024` | difficulty signal; B09's band-pass is a *modification* |
| Li et al., **Superfiltering**, `arXiv:2402.00530` | **ACL 2024 Long** — DBLP `conf/acl/LiZHLZWCZ24`, `2024.acl-long.769`, pp. 14255–14273 | weak-proxy difficulty; forbids "small model scoring is new" |
| Liu et al., **DEITA**, `arXiv:2312.15685` | **ICLR 2024** — DBLP `conf/iclr/0131Z00H24` | complexity × quality then diversity filtering |
| Chen et al., **AlpaGasus**, `arXiv:2307.08701` | **ICLR 2024** — DBLP `conf/iclr/ChenLYWGYTS0HJ24` | frozen-judge quality filtering |
| Zhao et al., **Long Is More for Alignment**, `arXiv:2402.04833` | **ICML 2024** — DBLP `conf/icml/ZhaoACF24` | `Longest-Response-5K`; mandatory, and the reason the fixed-token track exists |
| Wang et al., **Data Whisperer**, `arXiv:2505.12212` | **ACL 2025 Main** — DBLP `conf/acl/0001JWWZLWLHHZ25` | training-free ICL selector |
| Chen et al., **Agent-FLAN**, `arXiv:2403.12881` | **Findings of ACL 2024** — DBLP `conf/acl/ChenLWZLLCZ24`, `2024.findings-acl.557`, pp. 9354–9366 | capability rebalancing; forbids "first to balance reasoning/tool/format" |
| Ye et al., **Reasoning vs Boilerplate Tokens**, `arXiv:2412.14780` | **Findings of ACL 2025** — DBLP `conf/acl/YeZZMLF25`, `2025.findings-acl.1078`, pp. 20939–20957 | forbids "first to down-weight boilerplate" — directly relevant to B09's `FORMAT_ONLY` class |
| **SWE-TRACE**, `arXiv:2604.14820` | **arXiv-only** (DBLP CoRR 2026) | token-efficient shortest-path SWE trajectory synthesis |
| Han, *Systematic Evaluation of Trajectory Data Curation for LoRA Fine-Tuning of Code Agents*, `arXiv:2607.17205` | **ICIC 2026** — DBLP `conf/icic/Han26`; arXiv `journal_ref`: LNCS vol. 16669, pp. 39–50, Springer 2027 | ⚠️ `SOURCES.md`/`NOVELTY.md` list this as a 2026 arXiv preprint. It is **published** (LNCS). Forbids "agent trajectory filtering is unstudied". |

### 3.8 The audit's "offline RL / imitation coreset" family — searched, and the finding is a *gap*

This is the one family `RELATED_WORK_GAP_AUDIT_20260808.md:99` names that `NOVELTY.md` never covered.
Searched via arXiv API this session (queries and totals verbatim in §7.2). Result:

| Cite | Venue, verified | What it does | Why it does not reach B09 |
|---|---|---|---|
| Hejna et al., **CUPID: Curating Data your Robot Loves with Influence Functions**, `arXiv:2506.19121` | **arXiv-only** by DBLP (CoRR 2025); the paper's own arXiv comment states "Accepted to CoRL 2025" — **self-reported, not independently verified from this node** | influence-function estimate of each **demonstration's** effect on closed-loop expected return; filters harmful demos, sub-selects new trajectories; <33 % of data matches SOTA on RoboMimic | unit is the **whole demonstration**, and the target is a **closed-loop robot policy return**. No parent→child expansion (a robot demo is not expanded into many supervised rows), no token budget, no benchmark quota, no set-coverage objective |
| Mandi et al. (DataMIL), *Selecting Data for Robot Imitation Learning with Datamodels*, `arXiv:2505.09603` | **arXiv-only** (DBLP CoRR 2025) | datamodel-based end-to-end selection optimising task success rather than human quality notions; surrogate loss to avoid rollouts | same: demonstration-level, policy-return objective, no hierarchical/laminar constraint, no token accounting |
| *Quality over Quantity: Demonstration Curation via Influence Functions for Data-Centric Robot Learning*, `arXiv:2603.09056` | **arXiv-only**; 2026-03 | influence-function demo curation | ditto |

**Finding, stated plainly: the audit-named collision family is thinner than the audit implied, and
the searches that came back empty are themselves the evidence.** `abs:"coreset" AND
abs:"imitation learning"` → **0 results**. `abs:"data selection" AND abs:"offline reinforcement
learning" AND abs:"trajectory"` → **1**, unrelated (test-time goal-related experience).
`abs:"data pruning" AND abs:"offline RL"` → **0**. `abs:"demonstration selection" AND
abs:"behavioral cloning"` → **0**. `abs:"submodular" AND abs:"matroid" AND abs:"data selection" AND
abs:"language model"` → **0**. `abs:"partition matroid" AND abs:"subset selection"` → **1**
(*Pareto Optimization for Subset Selection with Dynamic Partition Matroid Constraints*,
`arXiv:2012.08738`, 2020 — pure combinatorial optimisation, no LM, no trajectories).
`abs:"per-trajectory" AND abs:"cap" AND abs:"selection"` → **0**. `abs:"sibling" AND
abs:"redundancy" AND abs:"training data"` → **0**. `abs:"parent" AND abs:"child" AND abs:"coreset"`
→ **0**. `abs:"hierarchical" AND abs:"coreset" AND abs:"instruction"` → **0**.

So the correct statement is: **the imitation/offline-RL curation family exists but works at
demonstration granularity toward a policy-return objective, and the laminar/partition-constrained
coreset machinery exists in combinatorial optimisation with no LM-data instantiation found.** The
audit's worry that this family already holds "完整组合 collision" is **not confirmed**; the actual
combination collision came from a different direction entirely (§3.1, ICML 2026).

⚠️ Two honest caveats on that negative: (a) these are **abstract-field** searches, so a paper that
does group-constrained coreset selection without those words in its abstract would be missed;
(b) the classical facility-location-under-partition-matroid result is textbook, so B09 must never
claim the *algorithm*, only the *instantiation* — which is what `PROPOSAL.md` §"Classical
subset-selection background" already says.

### 3.9 Concurrent (≤ 3 months; cannot preempt, must be cited)

* *Agentic Instruction Data Selection: Let DataMaster Interpret Your Intent*, `arXiv:2608.10579`
  (**2026-08-11 — four days before this pass**). arXiv-only. An LLM agent that *composes selection
  strategies* from a natural-language intent, arguing no single metric generalises. Orthogonal to
  B09 (meta-selection of strategies vs. one pre-registered constrained objective) but its premise —
  "no single metric generalises" — is rhetorically adjacent to B09's H1 and should be cited.
* *ANCHOR: Branch-Point Data Generation for GUI Agents*, `arXiv:2602.07153`. arXiv-only. Identifies
  **branch points** in seed demos, generates variants, verifies, then applies **task-conditioned
  step-level filtering**. Relevant because it independently uses the branch structure B09 wanted for
  "branch-verified credit" — but for **generation**, not selection.
* *EDGE: Efficient Data Selection for LLM Agents via Guideline Effectiveness*, `arXiv:2502.12494`.
  arXiv-only. Guideline-Effectiveness scoring for multi-turn agent samples; 50–75 % data reduction.
  A per-sample score, so it is a §3.7-class anchor, but it is agent-specific and must be cited.

---

## 4. MUST-NOT-CLAIM (binding; extends `NOVELTY.md` §5, does not replace it)

`NOVELTY.md` §5's 14 prohibitions stand unchanged. This pass adds:

15. ❌ **"First to select agent trajectory *steps* under a fixed budget by combining importance with
    diversity over states/tasks/interaction patterns."** → **Weasel, ICML 2026** (§3.1). This is the
    single most important addition in this file.
16. ❌ **"First to show a fixed-budget step-level selector improves *out-of-domain* agent
    generalisation across multiple student families."** → Weasel does exactly this on three students.
    B09's H3 must be phrased as *ID-vs-OOD differential attribution under matched constraints*, not
    as an OOD-improvement claim.
17. ❌ **"Branch-verified / outcome-flip decision credit is a novel technical core."** → **CSO,
    Findings of ACL 2026** (§3.4). `NOVELTY.md` §6 item 2 offered this as one of two escape routes
    from "mere combination"; it is now a published method. Implementing it is a **baseline
    reproduction**, and any B09 write-up must say so.
18. ❌ **"Diminishing-marginal-returns coverage over metadata buckets is novel."** → MIG,
    Findings of ACL 2025 (§3.6).
19. ❌ Citing ATLAS, MDS, CSO, SMART, MIG, Superfiltering, Agent-FLAN, or the code-agent curation
    paper **as preprints**. All eight are published; IDs are in §3. This is now a factual error, not
    a stylistic one.
20. ❌ Any statement implying B09's pilot **can be run**. See §0 — the pool does not exist.

---

## 5. Safe residual claim, narrowed on contact with Weasel

`NOVELTY.md` §4.1's wording ("jointly select across parents and within-trajectory decisions under
target-token and coverage constraints") is **no longer safe as written**: strip the words "parent
cap" and "token", and Weasel occupies it. The narrowed version — **one falsifiable sentence**:

> **When a parent agent trajectory is expanded into many dependent SFT child rows, a selector that
> is given a hard parent-multiplicity bound `|S ∩ U_g| ≤ m_g` and a fixed assistant-target-token
> budget `Σ c_i ≤ B` does *not* outperform a random subset drawn under those *same* constraints,
> matched on benchmark distribution, decision-type mix and length bin — i.e. the measurable benefit
> of per-row quality/criticality/relevance scoring, once grouping and budget are equalised, is
> zero.**

That is the **null**, and B09's contribution is to be the study designed to reject it. Three
properties make it defensible after §3:

1. **The comparator is a constraint-matched null, not a method.** No work in §3 runs it. Weasel
   compares against standard fine-tuning and selection baselines; RDS+ compares selectors against
   *unconstrained* random. "Beat random" and "beat random that already has your cap and your quota"
   are different experiments, and RDS+'s own result says the gap between them is where selectors die.
2. **A hard partition bound is not a pairwise-diversity penalty.** Falsifiable and cheap:
   the `m_g ∈ {1, 2, ∞}` ablation at fixed everything-else. If `m_g = ∞` + diversity ≈ `m_g = 1`,
   B09's H1 is dead and Weasel's soft formulation was sufficient all along.
3. **The token track is the unit on which "the win is just more tokens" dies.** B09's own kill
   criterion #5. Weasel's budget is steps and its efficiency claim is wall-clock speedup, so this
   contrast is genuinely unoccupied.

### 5.1 The row-multiplicity stress test is now the *primary* candidate contribution

`NOVELTY.md` §6 offered five routes out of "mere combination". After §3.1 and §3.4:
route 2 (branch-verified credit) is **published** (CSO); routes 1/3/4 (joint two-level marginal gain,
group-aware target coverage, laminar+knapsack analysis) sit inside the objective family DELIFT /
SMART / MIG / Weasel already own, so they can only be *variants*. **Route 5 — the row-multiplicity
stress test — is the only one no work in §3 performs**, and §3.8's empty searches
(`"sibling" AND "redundancy" AND "training data"` → 0; `"parent" AND "child" AND "coreset"` → 0)
are consistent with that.

Its design is also the cheapest thing in B09 and the most diagnostic: **artificially vary how many
rows each trajectory is expanded into, holding the underlying trajectories fixed, and measure whether
a flat selector's chosen subset and downstream score move.** If they do, "flat Top-K is fooled by
multiplicity" becomes a *measured property of selectors* rather than an argument — and it is a
property of **existing published selectors**, which makes it a contribution about the field rather
than about B09's own method. **Recommendation: promote it from ablation to headline.**

### 5.2 What B09 is, honestly, after this pass

A **controlled empirical study + robust recipe** with one methodological instrument (the
multiplicity stress test) and one strong null. `NOVELTY.md` §8 already anticipated this
("若第 5 项没有实现, proposal 应定位为 controlled empirical study + robust selection recipe") — this
pass makes it the **expected** landing zone rather than the fallback, because the escape route
`NOVELTY.md` favoured (branch-verified credit) turned out to be published.

### 5.3 Consequence for the promotion criteria

`STATUS.json.promotion_criteria[4]` reads: *"the method contains a non-compositional core beyond
combining ATLAS, RDS+, and facility location."* That list must be **amended** — Weasel (ICML 2026)
and CSO (Findings ACL 2026) now belong in it, and CSO's inclusion removes the escape route the
criterion was written to permit. **Not edited here** (`STATUS.json` is append-only and this file is
not authorised to rewrite promotion criteria); recorded as a required amendment.

---

## 6. Verdict

```
verdict: hold_in_backlog
novelty_gate: NEEDS_NARROWING -> narrowed claim in section 5 is CLEARED for design work
gpu: NONE authorised. Blocked on GATE -1 (data acquisition), not on this file.
promotion: NOT eligible. Requires GATE -1 + the section 5 null to be rejected under
           paired seeds + promotion_criteria[4] amended per section 5.3.
```

* **No candidate is 完全相同 / 抄袭.** Weasel is the closest and it lacks the parent-multiplicity
  bound, the token knapsack, the decision taxonomy and — decisively — the constraint-matched null.
  Per the standing rule, `already_dead_should_archive` is **not** warranted.
* **Narrowing is mandatory and evidence-forced**, exactly as the audit's `发现直接 collision 时再收窄`
  instructs. §3.1 is that direct collision; §5 is the narrowing.
* **B09 is still the audit's best-positioned backlog proposal on the literature axis**, and now the
  worst-positioned on the data axis. Those are independent facts and both belong in the record.

---

## 7. Honest gaps in this adjudication

1. **TopoCurate and RDS+ — B09's two most-cited collisions — are `arXiv-only` from this node.**
   TopoCurate: DBLP CoRR only; OpenReview returned no titled note. RDS+: DBLP CoRR only. Both must
   be cited as preprints. This is *not* a claim that no venue exists — recent-conference lag in DBLP
   is documented in `memory/venue-verify-must-use-openreview-2026.md`, and Weasel is a live example
   in this very file (ICML 2026 on OpenReview, CoRR-only on DBLP). Re-verify before submission.
2. **CUPID's CoRL 2025 acceptance is self-reported** from its arXiv comment field. DBLP has CoRR
   only. Not independently verified; labelled as such in §3.8.
3. **DBLP was intermittent all session** — `curl: (56) Failure when receiving data from the peer` and
   30 s timeouts on roughly one call in four; `dblp.org/rec/<key>.bib` needed up to 3 retries.
   Every venue row above is from a call that **did** return; none is inferred from a failure.
   Semantic Scholar was not queried at all (repo rule: never a venue authority).
4. **No `.bib` entries are emitted.** Per `memory/tcodex-exec-no-dash-c-flag.md`, entries do not
   enter a bibliography until venue-verified by family. The IDs here are sufficient to generate them
   later; generating them now would put an unverified TopoCurate/RDS+ row into the bibliography.
5. **§3.8's negative result is abstract-field-scoped.** All searches were `abs:` / `all:` field
   queries against the arXiv API. A group-constrained coreset paper phrased differently (e.g. in the
   active-learning or dataset-distillation vocabularies) would be missed. Stated as *searched and
   not found*, never as *does not exist*.
6. **No paper's full text was read this session** — every characterisation above is from the
   abstract plus, where relevant, the venue metadata block. For the four Critical/High collisions
   (Weasel, TopoCurate, MDS, ATLAS, CSO) a **full-text differential read is required before any
   write-up**, because the differences claimed in §3 are about design details (is the cap hard or
   soft? is there a token budget anywhere?) that an abstract can hide. This is the largest
   remaining risk in this file and it is deliberately not papered over.
7. **Zero cross-disk verification.** `/apdcephfs_zwfy6` is not mounted on LOCAL. Every disk fact
   about the (non-)existence of the candidate pool is carried from `DATA_AUDIT_VERDICT_20260810.md`
   and `STATUS.json`, which did search both disks. Per
   `memory/two-disk-rule-applies-to-main-too.md`, I did not re-derive it here and am not claiming to.
8. **`SOURCES.md`'s four open audit items: 1 closed, 3 partially closed.**
   Item 1 (does an existing work combine parent caps + critical-step credit + target facility
   coverage + token-budgeted submodular selection?) → **effectively closed, and the answer is "the
   nearest is Weasel, which has 2 of the 4"**; see §3.1.
   Item 3 (venue status of recent 2026 arXiv works) → **closed for MDS, CSO, ATLAS, MIG, SMART,
   Superfiltering, Agent-FLAN, boilerplate-tokens, code-agent curation, DELIFT, Weasel; still open
   for TopoCurate, RDS+, SWE-TRACE**.
   Item 4 (imitation/offline-RL group-constrained coreset) → **searched; see §3.8; the family is
   demonstration-level and does not hold the combination**.
   Item 2 (official implementations + licence compatibility) → **NOT DONE.** No repository was
   cloned, no `LICENSE` file read. This remains a hard blocker for any reproduction and it is
   downstream of GATE -1 anyway.
