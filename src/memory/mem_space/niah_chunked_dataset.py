"""T2 chunked associative-recall (NIAH) IterableDataset for Memory-Space training.

This is the **chunked** counterpart of ``niah_dataset.NIAHIterableDataset``.

Motivation (see status/STAGING_REPORT_readout_pivot.md §5)
---------------------------------------------------------
The legacy ``NIAHIterableDataset`` yields a FLAT ``input_ids``/``labels`` sequence
that goes through a plain forward pass — so local causal attention can read the
needle directly and the memory bank is never forced to carry it. T2 fixes this by
emitting samples in the SAME format as ``DolminoCurriculumDataset``:

    {context_chunks: [n_ctx tensors, each chunk_size], target_ids, answer_mask}

These are fed to ``dolmino_train_step`` (the HARDER objective): the context chunks
stream into the memory bank under ``no_grad`` and are detached, then LM loss is
computed only on the target (question) chunk — but here, further restricted by
``answer_mask`` to ONLY the answer digit tokens. The needle therefore lives in a
detached context chunk that is NOT in the target's attention window, so the answer
can only be recovered by precisely addressing the memory slot that encoded it.
This applies maximal "precise readout" pressure (the bottleneck identified in the
staging report).

chunk_size is a first-class experiment variable (§5.4): smaller chunks mean the
needle occupies a larger fraction of each memory write, so the (key->value)
encoding is cleaner and readout approaches a table lookup. To compare chunk_sizes
fairly we hold the **needle->query token distance** (``gap_tokens``) constant and
derive ``n_ctx = round(gap_tokens / chunk_size)`` — so chunk128 uses 4x as many
chunks as chunk512 to span the same token gap, isolating the granularity effect
from the distance effect.

Key construction guarantees
---------------------------
* The queried needle sentence is placed ENTIRELY inside a single context chunk
  (``insert_at + len(needle_ids) <= chunk_size``); it never straddles a boundary.
* The queried needle is placed at the START (offset 0) of a context chunk. By
  default that is the FIRST context chunk, so the needle->query token distance
  equals exactly ``n_ctx * chunk_size`` for every chunk_size (clean fixed gap).
  When ``random_needle_chunk=True`` (learn-to-select training) it is a RANDOM
  context chunk (offset 0); the chosen index is returned as
  ``needle_chunk_index`` so a selection loss can supervise against it.
* ``answer_mask`` is True only on the answer's digit-token positions in the target
  chunk; everything else (question prefix + padding) is masked out of the loss.
"""
from __future__ import annotations

import random
import string
from typing import Any, Dict, Iterator, List

import numpy as np
import torch
import torch.utils.data


# --------------------------------------------------------------------------- #
# BABILong-qa5-SHAPED event generator (2026-07-02, noise-structure-match).
# Public bAbI qa5 grammar ONLY; every (agent, object, room, receiver) combo is
# freshly RNG-generated here. NEVER reads/copies any babilong-test sentence.
#
# Distributions calibrated against the REAL babilong qa5 16k test set (used for
# ANALYSIS ONLY, never trained on):
#   * give verbs {gave,handed,passed}; move/pickup/drop verbs from bAbI grammar.
#   * answer agents recur; queried giver recurs mean ~8x/doc (move/pickup fillers).
#   * RECENCY: 35% of docs have >1 give of the SAME (giver,object); answer is the
#     receiver of the temporally-LAST such give (100% in the test set).
# --------------------------------------------------------------------------- #
_QA5B_AGENTS = ["Bill", "Fred", "Jeff", "Mary"]
_QA5B_OBJS = ["apple", "football", "milk"]
_QA5B_GIVE = ["gave", "handed", "passed"]
_QA5B_MOVE = ["went to the", "journeyed to the", "travelled to the",
              "moved to the", "went back to the"]
_QA5B_GET = ["got", "grabbed", "picked up", "took"]
_QA5B_DROP = ["dropped", "discarded", "left", "put down"]
_QA5B_ROOMS = ["garden", "kitchen", "office", "bedroom",
               "hallway", "bathroom", "hall"]


def _qa5b_gen_events(rng: "random.Random", target_events: int):
    """Generate one BABILong-qa5-shaped instance.

    Returns (giver, obj, question, answer, placed) where ``placed`` is a list of
    ``(pos, sentence, is_needle)`` with ``pos`` a global reading fraction in
    [0,1) — sort by pos to get temporal (== reading) order. The needle is the
    temporally-LAST give of (giver -> obj); ``answer`` is its receiver. Earlier
    COMPETING gives of the SAME (giver,obj) to DIFFERENT receivers create the
    recency-reasoning pressure. Move/pickup fillers (many featuring the queried
    giver) create entity recurrence so exact name-match cannot locate the needle.
    """
    G = rng.choice(_QA5B_AGENTS)
    OBJ = rng.choice(_QA5B_OBJS)
    others = [a for a in _QA5B_AGENTS if a != G]
    # n competing gives of the SAME (G, OBJ). Measured on babilong qa5 (n=100/len):
    # ~6% of docs have >1 matching give -> mostly a SINGLE give (answer uniquely
    # findable), a small recency tail. Match that: 94% =1, 5% =2, 1% =3.
    n_comp = rng.choices([1, 2, 3], [0.94, 0.05, 0.01])[0]
    n_comp = max(1, min(n_comp, len(others)))
    receivers = rng.sample(others, n_comp)   # distinct; answer = LAST one
    answer = receivers[-1]

    # Global positions: answer give biased to the END (babilong median frac 0.83).
    answer_pos = rng.uniform(0.62, 0.96)
    comp_positions = sorted(rng.uniform(0.05, max(0.06, answer_pos - 0.03))
                            for _ in range(n_comp - 1)) + [answer_pos]

    placed: list = []
    for i, (r, p) in enumerate(zip(receivers, comp_positions)):
        s = f"{G} {rng.choice(_QA5B_GIVE)} the {OBJ} to {r}."
        placed.append((p, s, i == n_comp - 1))

    n_filler = max(0, target_events - n_comp)
    for _ in range(n_filler):
        roll = rng.random()
        if roll < 0.15:
            # hard-negative give: SAME giver diff object, OR diff giver same object.
            if rng.random() < 0.5:
                o2 = rng.choice([o for o in _QA5B_OBJS if o != OBJ])
                r2 = rng.choice(others)
                s = f"{G} {rng.choice(_QA5B_GIVE)} the {o2} to {r2}."
            else:
                g2 = rng.choice(others)
                r2 = rng.choice([a for a in _QA5B_AGENTS if a != g2])
                s = f"{g2} {rng.choice(_QA5B_GIVE)} the {OBJ} to {r2}."
        elif roll < 0.70:
            # movement (babilong's dominant filler); ~45% feature the queried giver.
            subj = G if rng.random() < 0.45 else rng.choice(_QA5B_AGENTS)
            s = f"{subj} {rng.choice(_QA5B_MOVE)} {rng.choice(_QA5B_ROOMS)}."
        else:
            subj = G if rng.random() < 0.40 else rng.choice(_QA5B_AGENTS)
            o = rng.choice(_QA5B_OBJS)
            if rng.random() < 0.6:
                s = f"{subj} {rng.choice(_QA5B_GET)} the {o} there."
            else:
                s = f"{subj} {rng.choice(_QA5B_DROP)} the {o}."
        placed.append((rng.random(), s, False))

    placed.sort(key=lambda t: t[0])
    question = f"Who did {G} give the {OBJ} to?"
    return G, OBJ, question, answer, placed


# --------------------------------------------------------------------------- #
# Level 8 (2026-06-25): MULTI-TEMPLATE BABILong-qa5. L6/L7 emit exactly ONE
# question template ("Who did {G} give the {OBJ} to?", answer ALWAYS a
# receiver/agent name). Real babilong qa5 mixes several question forms that query
# DIFFERENT slots of the give-event:
#   T1 "Who did {G} give the {OBJ} to?"  -> receiver  (agent name)
#   T2 "What did {G} give to {R}?"        -> object    (object word)
#   T3 "Who gave the {OBJ} to {R}?"       -> giver     (agent name)
#   T4 "Who received the {OBJ}?"          -> receiver  (agent name)
#   T5 "Who gave the {OBJ}?"              -> giver     (agent name)
# Each sample draws ONE template; the answer TYPE (agent name OR object word) and
# the queried slot vary accordingly. The single-template L6/L7 taught the model
# only "read the receiver name", which we diagnose as the root of the short-档
# trade-off + ceiling. L8 forces reading different memory slots.
#
# The NOISE STRUCTURE is IDENTICAL to L7 (continuous PG19 background, recency-
# competing gives on the QUERIED key, entity recurrence, end-biased needle) — only
# question/answer generation differs. All (agent, object, receiver, verb) combos
# are freshly RNG-generated from the PUBLIC bAbI grammar/word-list; NEVER reads or
# copies any babilong-test sentence/组合 (same red-line as L6/L7).
#
# _QA5B_TEMPLATE_WEIGHTS: (template_id, draw_weight). Weighted so BOTH agent- and
# object-answers appear and all 5 forms are represented.
_QA5B_TEMPLATE_WEIGHTS = [(1, 0.24), (2, 0.24), (3, 0.18), (4, 0.17), (5, 0.17)]


def _qa5b_gen_events_multi(rng: "random.Random", target_events: int):
    """Multi-template BABILong-qa5-shaped instance (difficulty level 8).

    Returns ``(question, answer, answer_type, placed)``:
      * ``question``    : the drawn template, filled with fresh entities.
      * ``answer``      : single-word answer (agent name or object word).
      * ``answer_type`` : "agent" or "object" (for diagnostics only).
      * ``placed``      : list of ``(pos, sentence, is_needle)`` — same format as
        ``_qa5b_gen_events`` (sort by pos for temporal == reading order).

    A random question TEMPLATE is chosen; the answer is the value of that
    template's queried slot at the temporally-LAST give-event matching the
    template's KNOWN slots (recency, as in real babilong qa5 — answer == last
    matching give 100% of the time). Earlier COMPETING gives share the known slots
    but differ in the queried slot, creating the recency-reasoning pressure. All
    FILLER gives are guaranteed NOT to match the key (``matches`` predicate), so
    they never form an alternative answer — the answer stays unique regardless of
    where fillers land.
    """
    G = rng.choice(_QA5B_AGENTS)
    OBJ = rng.choice(_QA5B_OBJS)
    others = [a for a in _QA5B_AGENTS if a != G]

    ttype = rng.choices([t for t, _ in _QA5B_TEMPLATE_WEIGHTS],
                        [w for _, w in _QA5B_TEMPLATE_WEIGHTS])[0]

    # number of recency competitors matching the key (same tail as L7: mostly 1).
    n_comp = max(1, min(rng.choices([1, 2, 3], [0.94, 0.05, 0.01])[0], 3))

    # Build the KEY-matching give-events (evs[-1] is the needle). Each is a
    # (giver, object, receiver) triple; the QUERIED slot varies across them so the
    # answer is uniquely the needle's queried slot (by recency). ``matches`` marks
    # a give as key-matching so fillers can be kept OFF the key.
    evs: list = []  # list of (giver, obj, receiver)
    if ttype == 1:
        # key=(giver G, object OBJ); vary receiver; answer = last receiver.
        nc = min(n_comp, len(others))
        recvs = rng.sample(others, nc)
        evs = [(G, OBJ, rc) for rc in recvs]
        answer, answer_type = recvs[-1], "agent"
        question = f"Who did {G} give the {OBJ} to?"
        matches = lambda gv, ob, rc: gv == G and ob == OBJ
    elif ttype == 2:
        # key=(giver G, receiver R); vary object; answer = last object.
        R = rng.choice(others)
        nc = min(n_comp, len(_QA5B_OBJS))
        objs = rng.sample(_QA5B_OBJS, nc)
        evs = [(G, ob, R) for ob in objs]
        answer, answer_type = objs[-1], "object"
        question = f"What did {G} give to {R}?"
        matches = lambda gv, ob, rc: gv == G and rc == R
    elif ttype == 3:
        # key=(object OBJ, receiver R); vary giver; answer = last giver.
        R = rng.choice(others)
        cand = [a for a in _QA5B_AGENTS if a != R]
        nc = min(n_comp, len(cand))
        givers = rng.sample(cand, nc)
        evs = [(gv, OBJ, R) for gv in givers]
        answer, answer_type = givers[-1], "agent"
        question = f"Who gave the {OBJ} to {R}?"
        matches = lambda gv, ob, rc: ob == OBJ and rc == R
    elif ttype == 4:
        # key=(object OBJ); vary (giver,receiver); answer = last receiver.
        nc = min(n_comp, len(_QA5B_AGENTS) - 1)
        recvs = rng.sample(_QA5B_AGENTS, nc)
        for rc in recvs:
            gv = rng.choice([a for a in _QA5B_AGENTS if a != rc])
            evs.append((gv, OBJ, rc))
        answer, answer_type = recvs[-1], "agent"
        question = f"Who received the {OBJ}?"
        matches = lambda gv, ob, rc: ob == OBJ
    else:  # ttype == 5
        # key=(object OBJ); vary giver; answer = last giver.
        nc = min(n_comp, len(_QA5B_AGENTS))
        givers = rng.sample(_QA5B_AGENTS, nc)
        for gv in givers:
            rc = rng.choice([a for a in _QA5B_AGENTS if a != gv])
            evs.append((gv, OBJ, rc))
        answer, answer_type = givers[-1], "agent"
        question = f"Who gave the {OBJ}?"
        matches = lambda gv, ob, rc: ob == OBJ

    # Global positions: needle (last matching give) biased to the END (babilong
    # median frac 0.83); competitors sprinkled before it (recency pressure).
    answer_pos = rng.uniform(0.62, 0.96)
    comp_positions = sorted(
        rng.uniform(0.05, max(0.06, answer_pos - 0.03)) for _ in range(len(evs) - 1)
    ) + [answer_pos]

    placed: list = []
    for i, ((gv, ob, rc), p) in enumerate(zip(evs, comp_positions)):
        s = f"{gv} {rng.choice(_QA5B_GIVE)} the {ob} to {rc}."
        placed.append((p, s, i == len(evs) - 1))

    # Entity recurrence: the needle's giver + receiver (which cover every agent
    # NAMED in the question and the agent ANSWER) recur across move/pickup fillers
    # so exact-name string-match cannot locate the needle.
    recur = list({evs[-1][0], evs[-1][2]})

    n_filler = max(0, target_events - len(evs))
    for _ in range(n_filler):
        roll = rng.random()
        if roll < 0.15:
            # hard-negative give guaranteed NOT to match the key (retry a few
            # draws; fall back to a movement if none is found — keeps the answer
            # unique without ever needing an on-key filler).
            made = False
            for _try in range(8):
                gv = rng.choice(_QA5B_AGENTS)
                rc = rng.choice([a for a in _QA5B_AGENTS if a != gv])
                ob = rng.choice(_QA5B_OBJS)
                if not matches(gv, ob, rc):
                    placed.append((rng.random(),
                                   f"{gv} {rng.choice(_QA5B_GIVE)} the {ob} to {rc}.",
                                   False))
                    made = True
                    break
            if made:
                continue
            roll = 0.5  # fall through to a movement filler
        if roll < 0.70:
            subj = rng.choice(recur) if rng.random() < 0.5 else rng.choice(_QA5B_AGENTS)
            s = f"{subj} {rng.choice(_QA5B_MOVE)} {rng.choice(_QA5B_ROOMS)}."
        else:
            subj = rng.choice(recur) if rng.random() < 0.45 else rng.choice(_QA5B_AGENTS)
            o = rng.choice(_QA5B_OBJS)
            if rng.random() < 0.6:
                s = f"{subj} {rng.choice(_QA5B_GET)} the {o} there."
            else:
                s = f"{subj} {rng.choice(_QA5B_DROP)} the {o}."
        placed.append((rng.random(), s, False))

    placed.sort(key=lambda t: t[0])
    return question, answer, answer_type, placed


class NIAHChunkedDataset(torch.utils.data.IterableDataset):
    """Infinite chunked associative-recall dataset (T2).

    Each yielded sample is a dict with the SAME keys as DolminoCurriculumDataset
    plus ``answer_mask``:

        ``context_chunks``: list of n_ctx LongTensors, each [chunk_size]
        ``target_ids``:     LongTensor [chunk_size]   (question + answer + pad)
        ``answer_mask``:    BoolTensor [chunk_size]    (True only on answer digits)
        ``is_t2``:          True
        ``code``:           str — the queried 5-digit code (for diagnostics)

    Args:
        background_data: [N_chunks, L] int array of pre-tokenised natural-text
            token IDs (e.g. ``data/pg19_chunks_llama3.npy``). Used to fill
            non-needle context chunks so the memory channel sees realistic text.
        chunk_size: Tokens per chunk. MUST match the training forward chunk_size
            and (downstream) the eval ``--chunk_size``.
        gap_tokens: Target needle->query token distance held constant across
            chunk_sizes for fair comparison. ``n_ctx = round(gap_tokens /
            chunk_size)`` (clamped to >= 1).
        tokenizer: HuggingFace tokenizer (encode / pad_token_id / eos_token_id).
        num_keys: Number of (name -> code) mappings written into the context.
            One of them is queried; the other ``num_keys - 1`` are distractors.
            Default 1 (single needle); >1 reserved for the T2b multi-key ablation.
        seed: Base RNG seed. Each DDP worker adds its worker_id so workers are
            de-correlated (mirrors NIAHIterableDataset).
        background_skip: Skip the first ``background_skip`` background chunks to
            avoid train/eval contamination.
    """

    def __init__(
        self,
        background_data: np.ndarray,
        chunk_size: int,
        gap_tokens: int,
        tokenizer: Any,
        num_keys: int = 1,
        seed: int = 42,
        background_skip: int = 0,
        random_needle_chunk: bool = False,
        hard_distractors: int = 0,
        hard_distractor_mode: str = "mention",
        gap_mix: "list | None" = None,
        gap_batch_size: int = 1,
    ) -> None:
        """random_needle_chunk (2026-06-27, learn-to-select):
            False (default) -> the queried needle is ALWAYS at context chunk 0
                (legacy behaviour; the produced samples are BYTE-IDENTICAL to the
                prior dataset because no extra RNG calls are made).
            True  -> the queried needle is placed at a RANDOM context chunk index
                in [0, n_ctx-1] (offset 0 within that chunk). This is MANDATORY for
                the supervised-selection training (status/LEARN_TO_SELECT_DESIGN):
                with the needle always at chunk 0 a selector trivially learns
                "always pick chunk 0", which does not transfer. The chosen index is
                returned in the sample as ``needle_chunk_index`` so the train loop
                can supervise the per-chunk salience against it.
        """
        super().__init__()
        if background_data.ndim != 2:
            raise ValueError(
                f"background_data must be 2-D [N_chunks, L], got shape {background_data.shape}"
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if gap_tokens < 1:
            raise ValueError(f"gap_tokens must be >= 1, got {gap_tokens}")
        if num_keys < 1:
            raise ValueError(f"num_keys must be >= 1, got {num_keys}")

        self.background_data = background_data
        self.chunk_size = chunk_size
        self.gap_tokens = gap_tokens
        self.tokenizer = tokenizer
        self.num_keys = num_keys
        self.seed = seed
        self.background_skip = background_skip
        self.random_needle_chunk = bool(random_needle_chunk)
        # hard_distractors (2026-06-29, fix-T2-too-easy): number of NON-needle
        # context chunks that ALSO mention the QUERIED agent name, but only in an
        # irrelevant sentence WITHOUT the code. With the legacy task the queried
        # name is globally unique, so the selector can win by pure exact-string
        # match of the 6-letter name — needle_rank hits 0 from step 1, no gradient
        # signal, and the learned salience never transfers to BABILong qa5 (where
        # the queried entity recurs across many irrelevant sentences). Collision
        # distractors force the selector to key on the "name + code-assignment
        # pattern" co-occurrence (semantic salience) rather than the name token
        # alone. 0 = off (byte-identical to the prior dataset).
        self.hard_distractors = max(0, int(hard_distractors))
        # hard_distractor_mode: "mention" (plain code-free sentence, weak) or
        # "format" (MEMORIZE-aligned hard negative, strong — selector must read
        # the code region not the MEMORIZE surface pattern). See _make_sample 4c.
        self.hard_distractor_mode = str(hard_distractor_mode)

        # n_ctx derived so the needle (at chunk 0 offset 0) -> query (target chunk
        # start) distance == n_ctx * chunk_size ~= gap_tokens, for any chunk_size.
        self.n_ctx = max(1, int(round(gap_tokens / chunk_size)))

        # gap_mix (2026-07-01, ProLong mixed-length recipe): a list of ALTERNATIVE
        # gap_tokens values. When non-empty, EACH sample draws one gap uniformly
        # and derives its OWN n_ctx = max(1, round(gap/chunk_size)); the fixed
        # gap_tokens/self.n_ctx are then only the fallback. This makes the
        # needle->query distance a MIXTURE across 2k/4k/8k/16k in one run so the
        # readout fix covers every length档 instead of over-fitting a single gap
        # (the H1 "capability translation" root cause, DIRECTION_C_RESULT §1).
        # Empty/None -> byte-identical to the prior single-gap dataset. REQUIRES
        # batch_size==1 (the collate rejects mixed-n_ctx batches); at batch_size 1
        # each micro-step is a fresh gap so grad-accum averages over the mixture.
        self.gap_mix: List[int] = [int(g) for g in (gap_mix or []) if int(g) >= 1]

        # gap_batch_size (2026-07-04, per-batch-gap for bs>1 with gap_mix): the
        # per-SAMPLE gap draw above forces batch_size==1, because two samples that
        # drew different gaps have different n_ctx and cannot be stacked. To use
        # the 183GB L20A headroom we instead draw a gap ONCE per group of
        # ``gap_batch_size`` consecutive samples: all samples in the group share
        # the gap (== same n_ctx) so a batch of that size stacks cleanly, while the
        # gap still VARIES across groups → mixed-length curriculum is preserved
        # (now per-batch instead of per-sample). Set == training --batch_size.
        # <=1 (default) → byte-identical to the per-sample draw (bs=1 behaviour).
        self.gap_batch_size: int = max(1, int(gap_batch_size))
        self._grp_gap: "int | None" = None   # current group's gap (runtime state)
        self._grp_left: int = 0              # samples remaining in current group

        # Signal-difficulty curriculum (2026-06-29, user idea): per-sample the
        # distractor STYLE is drawn from a difficulty mix, so we can start on an
        # easy signal and, once the selector masters it, slowly blend in a harder
        # one (instead of a single fixed difficulty). Levels (easy->hard):
        #   1  unique-name        : no name collision (legacy exact-match; trivial)
        #   2  mention            : non-needle chunks mention the name, no code
        #   3  format             : MEMORIZE-aligned hard-neg (no real code)
        #   4  multicode          : SAME name assigned a DIFFERENT code in a
        #                           distractor (selector must pick the queried one;
        #                           here we mark the queried needle as the valid one)
        #   5  paraphrase         : needle uses NATURAL-language assignment (no
        #                           MEMORIZE template) -> approaches BABILong qa5
        # ``difficulty_mix`` maps level->weight; a sample's style is drawn by
        # weight. Default {} means "use the static hard_distractors/mode flags"
        # (back-compat). set_difficulty_mix() adjusts it mid-training.
        self.difficulty_mix: Dict[int, float] = {}

        self._n_chunks = len(background_data)
        self._usable_start = (
            background_skip % self._n_chunks if self._n_chunks > 0 else 0
        )

        self._pad_id: int = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    # --------------------------------------------------------------------- #
    # Curriculum support: let the needle->query distance grow during training.
    # --------------------------------------------------------------------- #
    def set_n_ctx(self, n_ctx: int) -> None:
        """Override the number of context chunks (and hence the needle->query
        distance = n_ctx * chunk_size). Used by curriculum schedules so the T2
        recall distance grows alongside the dolmino context length, instead of
        staying fixed at construction time. Safe to call mid-training; the next
        sample built by ``_make_sample`` reads ``self.n_ctx`` fresh."""
        self.n_ctx = max(1, int(n_ctx))

    def set_difficulty_mix(self, mix: Dict[int, float]) -> None:
        """Set the per-sample difficulty-level mixture (level->weight). Safe to
        call mid-training; ``_make_sample`` reads it fresh and draws each sample's
        distractor style by weight. Empty dict reverts to the static
        hard_distractors/hard_distractor_mode flags. See difficulty levels in
        __init__. This is the 'stairs' mechanism: e.g. start {2:1.0}, then once
        the selector masters L2 ramp to {2:0.7, 3:0.3}, etc."""
        self.difficulty_mix = {int(k): float(v) for k, v in (mix or {}).items()
                               if float(v) > 0.0}

    def _draw_level(self, rng: random.Random) -> int:
        """Draw a difficulty level for this sample. Returns 0 if no mix is set
        (caller then falls back to the static hard_distractors flags)."""
        if not self.difficulty_mix:
            return 0
        levels = sorted(self.difficulty_mix)
        weights = [self.difficulty_mix[l] for l in levels]
        return rng.choices(levels, weights=weights, k=1)[0]

    # --------------------------------------------------------------------- #
    # IterableDataset protocol
    # --------------------------------------------------------------------- #
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        rng = random.Random(self.seed + worker_id)

        all_indices = list(range(self._usable_start, self._n_chunks))
        if not all_indices:
            all_indices = list(range(self._n_chunks))
        worker_indices = all_indices[worker_id::num_workers]
        if not worker_indices:
            worker_indices = all_indices

        pos = 0  # cursor into worker_indices (cycled)

        while True:
            sample, pos = self._make_sample(rng, worker_indices, pos)
            yield sample

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _get_bg_chunk(self, worker_indices: List[int], pos: int) -> tuple[list[int], int]:
        """Return (token_list of length chunk_size, new_pos) cycling bg chunks."""
        idx = worker_indices[pos % len(worker_indices)]
        new_pos = (pos + 1) % len(worker_indices)
        tokens = self.background_data[idx, : self.chunk_size].tolist()
        if len(tokens) < self.chunk_size:
            tokens = tokens + [self._pad_id] * (self.chunk_size - len(tokens))
        return tokens, new_pos

    def _embed_needle(self, bg_tokens: list[int], needle_ids: list[int],
                      insert_at: int) -> list[int]:
        """Overwrite ``len(needle_ids)`` background tokens at ``insert_at`` with
        the needle, keeping the chunk exactly chunk_size tokens."""
        chunk = (
            bg_tokens[:insert_at]
            + needle_ids
            + bg_tokens[insert_at + len(needle_ids):]
        )
        chunk = (chunk + [self._pad_id] * self.chunk_size)[: self.chunk_size]
        return chunk

    def _make_sample(
        self,
        rng: random.Random,
        worker_indices: List[int],
        pos: int,
    ) -> tuple[Dict[str, Any], int]:
        """Build one chunked associative-recall sample.

        Layout (n_ctx context chunks + 1 target chunk):
            [chunk_0 (queried needle at offset 0) + bg] [chunk_1 bg] ...
            [chunk_{n_ctx-1} bg] [target: question + answer + pad]

        Distractor needles (num_keys-1 of them) are scattered into the remaining
        context chunks at random offsets that keep each needle inside one chunk.
        """
        n_ctx = self.n_ctx
        cs = self.chunk_size

        # gap_mix (2026-07-01): draw a per-sample needle->query distance so one run
        # trains a MIXTURE of length档 (ProLong: single-distribution CPT translates
        # capability, mixing spreads the fix). Overrides self.n_ctx for THIS sample
        # only; set_n_ctx()/t2_curriculum still work when gap_mix is empty.
        if self.gap_mix:
            if self.gap_batch_size > 1:
                # Per-batch gap: draw once per group of gap_batch_size samples so
                # a whole batch shares n_ctx (stackable) while gap still varies
                # ACROSS batches (mixed-length preserved). Group state persists on
                # self across _make_sample calls within one worker's __iter__.
                if self._grp_left <= 0 or self._grp_gap is None:
                    self._grp_gap = rng.choice(self.gap_mix)
                    self._grp_left = self.gap_batch_size
                _g = self._grp_gap
                self._grp_left -= 1
            else:
                _g = rng.choice(self.gap_mix)
            n_ctx = max(1, int(round(_g / cs)))

        # Draw this sample's difficulty level ONCE (0 = use static flags). Levels
        # 1-3 only change the distractor STYLE (block 4c). Level 5 (paraphrase)
        # ALSO changes the needle/query/answer to natural language (no MEMORIZE
        # template) — the hardest, closest to BABILong qa5. Level 4 (multicode)
        # keeps the template but the queried needle is the only VALID assignment.
        _level = self._draw_level(rng)
        _paraphrase = (_level == 5)
        _qa5 = (_level == 6)  # qa5 give-event: "X gave the Y to Z", answer=single label

        # Level 7 (2026-07-02, noise-structure-match, THIS agent): a BABILong-qa5
        # -SHAPED sample. Unlike L6 (1 needle at chunk-0-offset-0 + <=6 clean
        # template distractors, one per chunk), L7 reproduces the REAL babilong qa5
        # difficulty structure measured against the 16k test set (analysis, never
        # trains on it):
        #   * MANY bAbI sentences (mean ~18/doc: give/move/pickup) scattered at
        #     RANDOM offsets INSIDE chunks, interspersed with continuous PG19 prose
        #     (NOT one-per-chunk at offset 0). Needle is embedded mid-prose.
        #   * RECENCY reasoning: 0-2 COMPETING gives of the SAME (giver,object) to
        #     DIFFERENT receivers appear EARLIER; the answer is the receiver of the
        #     LAST such give (babilong: 35% of docs have >1 matching give, answer==
        #     last 100% of the time). L6 has NO recency (needle uniquely findable).
        #   * ENTITY RECURRENCE: the queried giver recurs in many non-give sentences
        #     (moves/pickups) so exact-name string-match cannot locate the needle
        #     (babilong: queried giver recurs mean 8.2x/doc). L6's giver is unique.
        #   * NEEDLE POSITION biased to the end (babilong median frac 0.83).
        # All combos are freshly RNG-generated from the PUBLIC bAbI grammar; never
        # copies any babilong-test sentence/组合. Handled by a dedicated early-return
        # builder so L1-L6 stay byte-identical.
        if _level == 7:
            return self._make_qa5_babilong_sample(rng, worker_indices, pos, n_ctx, cs)

        # Level 8 (2026-06-25, THIS agent): MULTI-TEMPLATE babilong-qa5. Same
        # early-return dispatch as L7 so L1-L7 stay byte-identical (no extra RNG
        # consumed on the L1-L7 paths). Reuses L7's noise structure but the
        # question TEMPLATE (and hence the answer slot/type: agent OR object) is
        # drawn per-sample from the 5 real qa5 question forms. Handled by a
        # dedicated builder that mirrors _make_qa5_babilong_sample exactly except
        # for question/answer generation (via _qa5b_gen_events_multi).
        if _level == 8:
            return self._make_qa5_babilong_multi_sample(rng, worker_indices, pos, n_ctx, cs)

        # qa5 give-event vocab (bAbI public grammar; synthetic combos only, never
        # reads babilong test instances). Used when _level==6.
        _QA5_AGENTS = ["Fred", "Bill", "Jeff", "Mary", "Sandra", "Daniel",
                       "John", "Bob", "Susan", "Julie"]
        _QA5_OBJS = ["apple", "milk", "football", "bread", "juice", "book"]
        _QA5_VERBS = ["gave", "handed", "passed"]

        # 1. Sample num_keys distinct (name -> code) mappings.
        # codes are stored SPACE-SEPARATED ("8 0 4 0 2") so the Llama-3 BPE
        # tokenizer emits ONE token per digit (5 independent answer tokens),
        # rather than merging "80402" into ~2 BPE tokens. This gives a sharper
        # precise-readout signal: each of the 5 digits is a separate retrieval
        # target and the model cannot shortcut by predicting a later digit from
        # an earlier one within a merged token.
        names: list[str] = []
        codes: list[str] = []          # spaced form, e.g. "8 0 4 0 2"
        seen: set[str] = set()
        while len(names) < self.num_keys:
            nm = "".join(rng.choices(string.ascii_uppercase, k=6))
            if nm in seen:
                continue
            seen.add(nm)
            names.append(nm)
            codes.append(" ".join(rng.choices(string.digits, k=5)))

        query_k = 0  # the FIRST mapping is the queried one (placed at chunk0/offset0)

        # 2. Tokenise needles, question, answer.
        #    needle and query use the SAME spaced-code form so the written and
        #    read-out token shapes match exactly.
        needle_ids_list: list[list[int]] = []
        if _qa5:
            # qa5 give-event: needle = "G gave the OBJ to R."; question asks the
            # receiver; answer = single label word R. Distractors (block 4c) are
            # COMPETING give-events so the model must bind the correct sf, not
            # rely on entity frequency. codes[query_k] holds the answer label so
            # downstream code (answer_mask, "code" field) works unchanged.
            _g, _r = rng.sample(_QA5_AGENTS, 2)        # giver != receiver
            _obj = rng.choice(_QA5_OBJS)
            _verb = rng.choice(_QA5_VERBS)
            names = [_g]                                # queried "name" = giver
            codes = [_r]                                # "code" (answer) = receiver label
            # stash for block 4c competing give-events
            _qa5_ctx = {"g": _g, "r": _r, "obj": _obj}
            sent = f"{_g} {_verb} the {_obj} to {_r}."
            needle_ids_list.append(
                self.tokenizer.encode(" " + sent, add_special_tokens=False)
            )
            question_ids = self.tokenizer.encode(
                f" Who did {_g} give the {_obj} to?\nAnswer:",
                add_special_tokens=False,
            )
        elif _paraphrase:
            # L5 natural-language: assign the code in plain prose, query in prose,
            # NO MEMORIZE template. Forces semantic match, not surface-pattern.
            _para_assign = [
                "When agent {nm} checked in, the clerk wrote down the figures {cd} as the access number.",
                "It turned out that {nm}, after some delay, was finally given the sequence {cd} to use.",
                "The dossier noted, almost in passing, that {nm} relies on the digits {cd} for entry.",
            ]
            for nm, cd in zip(names, codes):
                tmpl = _para_assign[(hash(nm) % len(_para_assign))] if False else _para_assign[len(needle_ids_list) % len(_para_assign)]
                sent = tmpl.format(nm=nm, cd=cd)
                needle_ids_list.append(
                    self.tokenizer.encode(" " + sent, add_special_tokens=False)
                )
            question_ids = self.tokenizer.encode(
                f"The access number that agent {names[query_k]} uses is",
                add_special_tokens=False,
            )
        else:
            for nm, cd in zip(names, codes):
                sent = f"MEMORIZE: The secret code for agent {nm} is {cd}. END_MEMORIZE"
                needle_ids_list.append(
                    self.tokenizer.encode(" " + sent, add_special_tokens=False)
                )
            question_ids = self.tokenizer.encode(
                f"The secret code for agent {names[query_k]} is",
                add_special_tokens=False,
            )
        # Encode the answer as " 8 0 4 0 2" (leading space binds to the first
        # digit token, so each digit -> its own token; 5 answer tokens total).
        answer_ids: list[int] = self.tokenizer.encode(
            " " + codes[query_k], add_special_tokens=False
        )

        # 3. Build context chunks filled with background text.
        context_chunks: list[list[int]] = []
        for _ in range(n_ctx):
            bg, pos = self._get_bg_chunk(worker_indices, pos)
            context_chunks.append(bg)

        # 4a. Place the QUERIED needle. Legacy (random_needle_chunk=False): chunk 0,
        #     offset 0 -> fixed gap == n_ctx*chunk_size (NO rng call, so byte-
        #     identical to the prior dataset). Learn-to-select
        #     (random_needle_chunk=True): a RANDOM context chunk, offset 0, so the
        #     selector cannot shortcut to "always chunk 0". The chosen index is
        #     returned as ``needle_chunk_index`` for the supervised-selection loss.
        if self.random_needle_chunk and n_ctx > 1:
            needle_chunk_index = rng.randint(0, n_ctx - 1)
        else:
            needle_chunk_index = 0
        q_needle = needle_ids_list[query_k]
        if len(q_needle) > cs:
            q_needle = q_needle[:cs]  # safety (should never trigger for ~25-token needle)
        context_chunks[needle_chunk_index] = self._embed_needle(
            context_chunks[needle_chunk_index], q_needle, 0
        )

        # 4b. Scatter distractor needles into other context chunks (if num_keys>1).
        if self.num_keys > 1 and n_ctx > 1:
            other_chunks = [c for c in range(n_ctx) if c != needle_chunk_index]
            rng.shuffle(other_chunks)
            for ki in range(1, self.num_keys):
                d_needle = needle_ids_list[ki]
                if len(d_needle) > cs:
                    d_needle = d_needle[:cs]
                # pick a chunk (cycle through available ones)
                ci = other_chunks[(ki - 1) % len(other_chunks)]
                max_off = max(0, cs - len(d_needle))
                insert_at = rng.randint(0, max_off)
                context_chunks[ci] = self._embed_needle(
                    context_chunks[ci], d_needle, insert_at
                )

        # 4c. Collision distractors. The EFFECTIVE style/count is either the
        #     static flags (hard_distractors / hard_distractor_mode) OR, when a
        #     difficulty mix is active, derived from _level (drawn at top).
        #     Level -> (effective hard_distractors, effective mode):
        #       1 unique-name -> 0 collisions (trivial, exact-match)
        #       2 mention     -> fill, mention mode
        #       3,4 format    -> fill, format hard-neg (MEMORIZE-aligned)
        #       5 paraphrase  -> fill, paraphrase mode (natural-language hard-neg,
        #                        matches the L5 natural-language needle)
        _eff_hd = self.hard_distractors
        _eff_mode = self.hard_distractor_mode
        if _level > 0:
            if _level <= 1:
                _eff_hd = 0
            elif _level == 2:
                _eff_hd, _eff_mode = max(1, n_ctx - 1), "mention"
            elif _level == 5:
                _eff_hd, _eff_mode = max(1, n_ctx - 1), "paraphrase"
            elif _level == 6:
                _eff_hd, _eff_mode = max(1, n_ctx - 1), "qa5"
            else:  # 3,4 -> format hard-neg
                _eff_hd, _eff_mode = max(1, n_ctx - 1), "format"
        if _eff_hd > 0 and n_ctx > 1:
            avail = [c for c in range(n_ctx) if c != needle_chunk_index]
            rng.shuffle(avail)
            qname = names[query_k]
            if _eff_mode == "qa5":
                # qa5 competing give-events: same giver/object but DIFFERENT
                # receiver, plus unrelated give-events and movements. Forces the
                # model to bind the correct sf sentence (the real receiver) rather
                # than picking any nearby agent. This is the core "robust readout
                # in distraction" signal for direction-c.
                _g = _qa5_ctx["g"]; _r = _qa5_ctx["r"]; _obj = _qa5_ctx["obj"]
                _wrong = [a for a in _QA5_AGENTS if a not in (_g, _r)]
                rng.shuffle(_wrong)
                _o2 = rng.choice([o for o in _QA5_OBJS if o != _obj])
                _rooms = ["garden", "kitchen", "office", "bedroom", "hallway", "bathroom"]
                collide_templates = [
                    f"{_g} {rng.choice(_QA5_VERBS)} the {_o2} to {_wrong[0]}.",
                    f"{_wrong[1 % len(_wrong)]} {rng.choice(_QA5_VERBS)} the {_obj} to {_wrong[2 % len(_wrong)]}.",
                    f"{_g} went back to the {rng.choice(_rooms)}.",
                    f"{_wrong[0]} journeyed to the {rng.choice(_rooms)}.",
                ]
            elif _eff_mode == "paraphrase":
                # L5 natural-language hard negatives: mention the queried name in
                # prose that does NOT assign a concrete access number, matching the
                # L5 prose needle so the selector must read semantics, not surface.
                collide_templates = [
                    f"Agent {qname} arrived late and the access number was never written down.",
                    f"The clerk forgot to record any figures for agent {qname} that day.",
                    f"No access number was ever assigned to agent {qname} in this dossier.",
                    f"Agent {qname} was mentioned only in passing, with no digits given.",
                ]
            elif _eff_mode == "format":
                # FORMAT-ALIGNED hard negatives (2026-06-29): same MEMORIZE
                # opening + same queried name as the real needle, so the selector
                # CANNOT win by detecting the "MEMORIZE:/digit-string" surface
                # pattern — it must read the code REGION to tell that this entry
                # does not actually assign agent {qname} a concrete code. The
                # legacy "mention" mode (name in a plain sentence, no MEMORIZE) was
                # too easy: selector keyed on MEMORIZE presence (only 8% of steps
                # produced a non-zero select_ce). These keep MEMORIZE but make the
                # code slot non-committal / refer to a DIFFERENT agent.
                _others = [n for n in names if n != qname] or [qname]
                collide_templates = [
                    f"MEMORIZE: The secret code for agent {qname} is not recorded in this file. END_MEMORIZE",
                    f"MEMORIZE: The secret code for agent {qname} was revoked and left blank. END_MEMORIZE",
                    f"MEMORIZE: The secret code for agent {_others[0]} is unrelated to agent {qname}. END_MEMORIZE",
                    f"MEMORIZE: Agent {qname} has no secret code on record at this time. END_MEMORIZE",
                ]
            else:
                # legacy "mention" mode: name in a plain code-free sentence.
                collide_templates = [
                    f"Earlier that day agent {qname} left the secret facility without a word.",
                    f"Nobody had seen agent {qname} since the code review meeting ended.",
                    f"The report about agent {qname} made no mention of any secret code.",
                    f"Later, agent {qname} was reassigned and the old code was retired.",
                ]
            n_coll = min(_eff_hd, len(avail))
            for di in range(n_coll):
                ci = avail[di]
                sent = collide_templates[di % len(collide_templates)]
                coll_ids = self.tokenizer.encode(" " + sent, add_special_tokens=False)
                if len(coll_ids) > cs:
                    coll_ids = coll_ids[:cs]
                max_off = max(0, cs - len(coll_ids))
                insert_at = rng.randint(0, max_off)
                context_chunks[ci] = self._embed_needle(
                    context_chunks[ci], coll_ids, insert_at
                )

        # 5. Build target chunk: question + answer + padding to chunk_size.
        target_raw = question_ids + answer_ids
        if len(target_raw) < cs:
            target_tokens = target_raw + [self._pad_id] * (cs - len(target_raw))
        else:
            target_tokens = target_raw[:cs]

        # 6. answer_mask: True ONLY on the 5 digit-token positions.
        # The spaced answer " 6 7 0 0 8" tokenizes to [' ','6',' ','7',...] —
        # digit tokens interleaved with space tokens. We mask only the digit
        # tokens (skip the spaces) so the LM loss falls purely on the 5 retrieval
        # targets and is not diluted by trivial space-token prediction.
        answer_mask = [False] * cs
        ans_start = len(question_ids)
        for j, tid in enumerate(answer_ids):
            p = ans_start + j
            if p >= cs:
                break
            # Mask answer-content tokens: digits (code task) OR any non-empty
            # token (qa5 label words). For the digit task this is byte-identical
            # to the old .isdigit() check (code tokens are digits/spaces, and
            # spaces strip to "" so they stay unmasked).
            if self.tokenizer.decode([tid]).strip():
                answer_mask[p] = True

        sample = {
            "context_chunks": [
                torch.tensor(c, dtype=torch.long) for c in context_chunks
            ],
            "target_ids": torch.tensor(target_tokens, dtype=torch.long),
            "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
            "is_t2": True,
            "code": codes[query_k],
            # Document-absolute context-chunk index holding the queried needle.
            # 0 in the legacy (chunk-0) layout; a random index when
            # random_needle_chunk=True. Used by the supervised-selection loss
            # (status/LEARN_TO_SELECT_DESIGN) as the CE target for the per-chunk
            # reader-attn salience.
            "needle_chunk_index": int(needle_chunk_index),
        }
        return sample, pos

    # --------------------------------------------------------------------- #
    # BABILong-qa5-SHAPED builder (difficulty level 7).
    # --------------------------------------------------------------------- #
    def _embed_many(self, bg_tokens: list[int], items: list[tuple[int, list[int]]]) -> list[int]:
        """Overwrite bg tokens at multiple (offset, ids) sites, keeping chunk_size.
        Sites must be non-overlapping and sorted by offset (caller guarantees)."""
        chunk = list(bg_tokens)
        for off, ids in items:
            end = off + len(ids)
            if end > self.chunk_size:
                ids = ids[: self.chunk_size - off]
                end = off + len(ids)
            chunk[off:end] = ids
        chunk = (chunk + [self._pad_id] * self.chunk_size)[: self.chunk_size]
        return chunk

    def _make_qa5_babilong_sample(
        self,
        rng: random.Random,
        worker_indices: List[int],
        pos: int,
        n_ctx: int,
        cs: int,
    ) -> tuple[Dict[str, Any], int]:
        """Build a BABILong-qa5-SHAPED sample (difficulty level 7).

        Reproduces the REAL babilong qa5 noise structure (measured on the 16k
        test set, ANALYSIS ONLY): ~0.65 bAbI sentences/chunk scattered at random
        offsets INSIDE continuous-prose chunks, recency-competing gives, entity
        recurrence, end-biased needle. All combos freshly generated from the
        public grammar — never copies a babilong-test sentence.
        """
        # bAbI-sentence COUNT is ~FIXED across lengths in real babilong qa5 (median
        # 15, mean 18.5 events/doc AT EVERY LENGTH — only the PG19 padding grows).
        # Draw ~15 (not scaled by n_ctx); multiple events share a chunk when there
        # are fewer chunks than events (the builder groups per-chunk). Clamp only so
        # a degenerate 1-chunk doc still works.
        target_events = max(4, rng.randint(12, 18) if n_ctx >= 4 else n_ctx)
        giver, obj, question_str, answer, placed = _qa5b_gen_events(rng, target_events)

        # 1. Background context chunks (continuous prose).
        context_chunks: list[list[int]] = []
        for _ in range(n_ctx):
            bg, pos = self._get_bg_chunk(worker_indices, pos)
            context_chunks.append(bg)

        # 2. Map each placed sentence (global reading fraction) -> chunk index, in
        #    temporal (== reading) order. Group per chunk, then splice into evenly
        #    spread, non-overlapping, ORDERED slots inside that chunk (mid-prose).
        per_chunk: Dict[int, list] = {}
        needle_chunk_index = 0
        for gpos, sent, is_needle in placed:
            ci = min(n_ctx - 1, max(0, int(gpos * n_ctx)))
            per_chunk.setdefault(ci, []).append((sent, is_needle))
            if is_needle:
                needle_chunk_index = ci

        for ci, sents in per_chunk.items():
            k = len(sents)
            # tokenize (leading space so the sentence tokenizes naturally in prose)
            tok_sents = [
                self.tokenizer.encode(" " + s, add_special_tokens=False)[:cs]
                for s, _ in sents
            ]
            # Divide the chunk into k ordered slots; place sentence j at a random
            # offset within slot j so temporal order is preserved and sentences do
            # not overlap. Reserve tail room for the longest sentence.
            slot = max(1, cs // k)
            items: list[tuple[int, list[int]]] = []
            for j, ids in enumerate(tok_sents):
                lo = j * slot
                hi = min((j + 1) * slot, cs) - len(ids)
                hi = max(lo, hi)
                off = rng.randint(lo, hi) if hi > lo else lo
                off = min(off, max(0, cs - len(ids)))
                items.append((off, ids))
            items.sort(key=lambda t: t[0])
            # de-overlap: push each start past the previous end
            cleaned: list[tuple[int, list[int]]] = []
            cursor = 0
            for off, ids in items:
                off = max(off, cursor)
                if off + len(ids) > cs:
                    off = max(0, cs - len(ids))
                cleaned.append((off, ids))
                cursor = off + len(ids)
            context_chunks[ci] = self._embed_many(context_chunks[ci], cleaned)

        # 3. Target chunk: babilong-style question + single-word answer.
        question_ids = self.tokenizer.encode(
            f" {question_str}\nAnswer:", add_special_tokens=False
        )
        answer_ids = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        target_raw = question_ids + answer_ids
        if len(target_raw) < cs:
            target_tokens = target_raw + [self._pad_id] * (cs - len(target_raw))
        else:
            target_tokens = target_raw[:cs]

        # 4. answer_mask: True on the answer's content tokens (label word).
        answer_mask = [False] * cs
        ans_start = len(question_ids)
        for j, tid in enumerate(answer_ids):
            p = ans_start + j
            if p >= cs:
                break
            if self.tokenizer.decode([tid]).strip():
                answer_mask[p] = True

        sample = {
            "context_chunks": [
                torch.tensor(c, dtype=torch.long) for c in context_chunks
            ],
            "target_ids": torch.tensor(target_tokens, dtype=torch.long),
            "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
            "is_t2": True,
            "code": answer,
            "needle_chunk_index": int(needle_chunk_index),
        }
        return sample, pos

    # --------------------------------------------------------------------- #
    # MULTI-TEMPLATE BABILong-qa5-SHAPED builder (difficulty level 8).
    # --------------------------------------------------------------------- #
    def _make_qa5_babilong_multi_sample(
        self,
        rng: random.Random,
        worker_indices: List[int],
        pos: int,
        n_ctx: int,
        cs: int,
    ) -> tuple[Dict[str, Any], int]:
        """Build a MULTI-TEMPLATE BABILong-qa5-shaped sample (difficulty level 8).

        Identical NOISE STRUCTURE to ``_make_qa5_babilong_sample`` (level 7):
        continuous PG19 background, ~15 bAbI sentences scattered at random offsets
        mid-prose, recency-competing gives on the queried key, entity recurrence,
        end-biased needle. The ONLY difference is the question/answer: drawn from
        the 5 real qa5 question templates via ``_qa5b_gen_events_multi``, so the
        answer is sometimes a RECEIVER, sometimes a GIVER (agent names), and
        sometimes the OBJECT word — forcing the model to read whichever memory slot
        the template queries, not just "the receiver". All combos are freshly
        RNG-generated from the public bAbI grammar; never copies a babilong-test
        sentence.
        """
        target_events = max(4, rng.randint(12, 18) if n_ctx >= 4 else n_ctx)
        question_str, answer, answer_type, placed = _qa5b_gen_events_multi(
            rng, target_events
        )

        # 1. Background context chunks (continuous prose).
        context_chunks: list[list[int]] = []
        for _ in range(n_ctx):
            bg, pos = self._get_bg_chunk(worker_indices, pos)
            context_chunks.append(bg)

        # 2. Map each placed sentence (global reading fraction) -> chunk index, in
        #    temporal order; group per chunk; splice into ordered non-overlapping
        #    slots mid-prose (byte-for-byte the same placement logic as L7).
        per_chunk: Dict[int, list] = {}
        needle_chunk_index = 0
        for gpos, sent, is_needle in placed:
            ci = min(n_ctx - 1, max(0, int(gpos * n_ctx)))
            per_chunk.setdefault(ci, []).append((sent, is_needle))
            if is_needle:
                needle_chunk_index = ci

        for ci, sents in per_chunk.items():
            k = len(sents)
            tok_sents = [
                self.tokenizer.encode(" " + s, add_special_tokens=False)[:cs]
                for s, _ in sents
            ]
            slot = max(1, cs // k)
            items: list[tuple[int, list[int]]] = []
            for j, ids in enumerate(tok_sents):
                lo = j * slot
                hi = min((j + 1) * slot, cs) - len(ids)
                hi = max(lo, hi)
                off = rng.randint(lo, hi) if hi > lo else lo
                off = min(off, max(0, cs - len(ids)))
                items.append((off, ids))
            items.sort(key=lambda t: t[0])
            cleaned: list[tuple[int, list[int]]] = []
            cursor = 0
            for off, ids in items:
                off = max(off, cursor)
                if off + len(ids) > cs:
                    off = max(0, cs - len(ids))
                cleaned.append((off, ids))
                cursor = off + len(ids)
            context_chunks[ci] = self._embed_many(context_chunks[ci], cleaned)

        # 3. Target chunk: babilong-style question + single-word answer (agent name
        #    OR object word depending on the drawn template).
        question_ids = self.tokenizer.encode(
            f" {question_str}\nAnswer:", add_special_tokens=False
        )
        answer_ids = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        target_raw = question_ids + answer_ids
        if len(target_raw) < cs:
            target_tokens = target_raw + [self._pad_id] * (cs - len(target_raw))
        else:
            target_tokens = target_raw[:cs]

        # 4. answer_mask: True on the answer's content tokens (agent name or object
        #    word); spaces stay unmasked (same rule as L6/L7).
        answer_mask = [False] * cs
        ans_start = len(question_ids)
        for j, tid in enumerate(answer_ids):
            p = ans_start + j
            if p >= cs:
                break
            if self.tokenizer.decode([tid]).strip():
                answer_mask[p] = True

        sample = {
            "context_chunks": [
                torch.tensor(c, dtype=torch.long) for c in context_chunks
            ],
            "target_ids": torch.tensor(target_tokens, dtype=torch.long),
            "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
            "is_t2": True,
            "code": answer,
            "answer_type": answer_type,
            "needle_chunk_index": int(needle_chunk_index),
        }
        return sample, pos
# --------------------------------------------------------------------------- #


def niah_chunked_collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of NIAHChunkedDataset samples into batched tensors.

    All samples in a batch share the same ``n_ctx`` and ``chunk_size`` (T2 uses a
    fixed gap -> fixed n_ctx). Returns:

        ``context_chunks``: list of n_ctx tensors, each [B, chunk_size]
        ``target_ids``:     [B, chunk_size]
        ``answer_mask``:    [B, chunk_size]
        ``is_t2``:          True
    """
    if not batch:
        raise ValueError("niah_chunked_collate_fn received an empty batch")

    n_ctx = len(batch[0]["context_chunks"])
    for s in batch:
        if len(s["context_chunks"]) != n_ctx:
            raise ValueError(
                "niah_chunked_collate_fn: samples in a batch have different n_ctx "
                f"({len(s['context_chunks'])} vs {n_ctx}); T2 n_ctx must be constant."
            )

    context_chunks: List[torch.Tensor] = []
    for k in range(n_ctx):
        col = [s["context_chunks"][k] for s in batch]
        min_len = min(t.shape[0] for t in col)
        context_chunks.append(torch.stack([t[:min_len] for t in col], dim=0))

    tgt_col = [s["target_ids"] for s in batch]
    tgt_min = min(t.shape[0] for t in tgt_col)
    target_ids = torch.stack([t[:tgt_min] for t in tgt_col], dim=0)

    msk_col = [s["answer_mask"] for s in batch]
    answer_mask = torch.stack([t[:tgt_min] for t in msk_col], dim=0)

    # needle_chunk_index: [B] long tensor (CE target for the selection loss).
    # Absent in older samples -> default 0 (legacy chunk-0 layout).
    needle_chunk_index = torch.tensor(
        [int(s.get("needle_chunk_index", 0)) for s in batch], dtype=torch.long
    )

    return {
        "context_chunks": context_chunks,
        "target_ids": target_ids,
        "answer_mask": answer_mask,
        "is_t2": True,
        "needle_chunk_index": needle_chunk_index,
    }
