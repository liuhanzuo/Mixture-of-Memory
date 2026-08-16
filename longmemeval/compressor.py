"""Evidence compressors for the LongMemEval baseline.

A *compressor* is a separate concern from the reranker. Where the reranker
re-orders a candidate pool (precision over recall), the compressor takes the
retrieved rounds and produces a short, fixed-budget "notes" text that augments
(does NOT replace) the top raw evidence handed to the reader.

This implements the LongMemEval-V2 winning pattern — **raw evidence + notes** —
where a compact synopsis of the (possibly many) retrieved rounds is prepended
to a budget-limited set of raw evidence blocks. The reader then sees:

    MoM NOTES: <synopsis conditioned on the question>
    [Evidence 1 | ...]
    [Evidence 2 | ...]
    ...

so the headline facts survive a tight evidence-token budget even when the
relevant round would otherwise be truncated away.

Concrete compressors
---------------------
  * :class:`IdentityCompressor` — no-op. Produces an empty notes string, so the
    evidence list passed to the reader is unchanged (the baseline).
  * :class:`SelfNotesCompressor` — **self-notes**: the *reader's own* model
    writes the notes from the *same* retrieved evidence it will then read. This
    is what B08 leg-1's pre-registration requires
    (``notes_generator_must_be_the_reader_itself``), because a second model
    would confound "notes help" with "the second model is good". It shares the
    reader's already-loaded weights and persists ``{question_id: notes}`` so
    both notes arms consume byte-identical notes text.
  * :class:`MoMNotesCompressor` — loads the mem_space (MoM) model (reusing the
    BABILong loaders exactly like ``backends.MoMModelReranker``), streams the
    concatenated retrieved rounds through the memory bank, and *generates* a
    short notes string conditioned on the question. Because the mem_space model
    is a generative Llama-3 backbone with a memory bank, the genuine version is:
    stream the rounds into the bank, then ask it (in the last chunk) to
    summarize the facts relevant to the question, decoding a few dozen tokens
    via :func:`scripts.run_babilong_mem_space.generate_with_mem_space`.

No training is involved: this is inference-only over an existing checkpoint.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .backends import Evidence


#: The notes instruction, shared VERBATIM by ``MoMNotesCompressor`` and
#: ``SelfNotesCompressor``. Keeping one string means the notes *prompt* is not a
#: new uncontrolled variable when the notes *generator* changes
#: (``LEG1_IMPL_PLAN.md`` §2.3 point 5). Byte-identical to the pre-2026-08-16
#: literal inlined in ``MoMNotesCompressor._build_input_ids``.
_NOTES_INSTRUCTION = (
    "\n\nSummarize the facts from the conversation above that are "
    "relevant to answering this question: {question}\n"
    "Relevant facts:"
)


def _import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "torch is required to generate notes (import error: %r)" % (e,)
        )
    return torch


class Compressor(ABC):
    """Abstract evidence compressor.

    Given the question and the retrieved evidence, return a compact *notes*
    string (may be empty). The harness prepends a non-empty notes string as an
    extra, clearly-labeled evidence block in front of the raw evidence; an empty
    string means "no notes" (the evidence is passed through unchanged).
    """

    #: Provenance label rendered INTO the reader's prompt for the notes block
    #: (``Evidence.as_block`` puts ``session=`` in the prompt text, so this is
    #: model input, not a comment). Defaults to ``MoM`` so the historical
    #: ``mom_notes`` path stays byte-identical; ``SelfNotesCompressor``
    #: overrides it to ``SELF`` because with self-notes the generator IS the
    #: reader and a "MoM" label would misdescribe the arm's own provenance.
    label: str = "MoM"

    @abstractmethod
    def compress(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
    ) -> str:
        """Return a short notes string summarizing the evidence for the question."""


class IdentityCompressor(Compressor):
    """No-op compressor: returns an empty notes string.

    With empty notes the harness prepends nothing, so the reader sees the raw
    evidence unchanged. This is the baseline path (``--compressor none``).
    """

    def compress(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
    ) -> str:
        return ""


class MoMNotesCompressor(Compressor):
    """Generate question-conditioned notes with the mem_space (MoM) model.

    The model is loaded EXACTLY like :class:`backends.MoMModelReranker`: we reuse
    the BABILong mem_space loaders (``build_mem_space_config`` /
    ``load_mem_space_model``) so all the checkpoint handling (DDP-prefix
    stripping, RoPE fp32 snapshot, Fix J warmup) lives in one place.

    Notes generation (genuine, inference-only)
    ------------------------------------------
    1. Concatenate the retrieved rounds into a single context string (capped at
       ``max_context_tokens`` so streaming stays bounded).
    2. Append an instruction that asks the model to summarize the facts relevant
       to the question. The instruction + question land at the END of the input
       so they fall in the LAST chunk — mirroring how BABILong puts the question
       suffix after the haystack.
    3. Feed the whole thing to
       :func:`scripts.run_babilong_mem_space.generate_with_mem_space`, which
       resets the bank, streams all-but-last chunks into the memory bank
       (chunk-by-chunk, ``chunk_size``), freezes the bank, and autoregressively
       decodes ``max_new_tokens`` from the last chunk.

    The decoded text is the notes string. Generation is intentionally short
    (default ``max_new_tokens=128``) — the point is a compact synopsis, not a
    full answer.
    """

    def __init__(
        self,
        checkpoint: str,
        adapter_config: str,
        model_path: str = "models/Meta-Llama-3-8B",
        device: str = "cuda:0",
        chunk_size: int = 512,
        max_new_tokens: int = 128,
        max_context_tokens: int = 8192,
        dtype: str = "bfloat16",
    ):
        self.checkpoint = checkpoint
        self.adapter_config = adapter_config
        self.model_path = model_path
        self.device_str = device
        self.chunk_size = chunk_size
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.dtype_str = dtype

        # Heavy imports + model load happen once, at construction.
        import torch
        from transformers import AutoTokenizer

        # Reuse the BABILong mem_space loaders so checkpoint/config handling
        # stays in one place — identical to backends.MoMModelReranker.
        from scripts.run_babilong_mem_space import (
            build_mem_space_config,
            generate_with_mem_space,
            load_mem_space_model,
        )

        self._torch = torch
        self._generate_with_mem_space = generate_with_mem_space

        self._device = torch.device(device)
        self._dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        with open(adapter_config, "r") as f:
            adapter_cfg = json.load(f)
        mem_config = build_mem_space_config(adapter_cfg)
        # L3 token-recon head sizes pos_queries to chunk_size at train time;
        # mirror it here so the loaded ckpt's shapes line up.
        mem_config.l3_recon_max_positions = chunk_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = load_mem_space_model(
            model_path=model_path,
            checkpoint_path=checkpoint,
            mem_config=mem_config,
            device=self._device,
            dtype=self._dtype,
        )
        self.model.eval()
        self.loaded = True  # truthy marker so the CLI can assert non-degraded

    def _build_input_ids(self, question: str, evidence: List[Evidence]):
        """Tokenize <capped rounds context> + <instruction + question>.

        The rounds context is token-truncated (head-kept) to
        ``max_context_tokens`` so the instruction/question always survive at the
        end (last chunk). Returns a [1, T] LongTensor on the model device.
        """
        torch = self._torch

        context_text = "\n\n".join(ev.text for ev in evidence).strip()
        instruction = _NOTES_INSTRUCTION.format(question=question)

        instr_ids = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        ctx_ids = self.tokenizer(context_text, add_special_tokens=True)["input_ids"]

        # Reserve room for the instruction within the context cap so the
        # question always lands in the final chunk.
        ctx_cap = max(self.chunk_size, self.max_context_tokens - len(instr_ids))
        if len(ctx_ids) > ctx_cap:
            ctx_ids = ctx_ids[:ctx_cap]

        ids = ctx_ids + instr_ids
        if not ids:
            ids = self.tokenizer(" ", add_special_tokens=True)["input_ids"]
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def compress(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
    ) -> str:
        if not evidence:
            return ""
        input_ids = self._build_input_ids(question, evidence)
        notes = self._generate_with_mem_space(
            model=self.model,
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            max_new_tokens=self.max_new_tokens,
            device=self._device,
        )
        return (notes or "").strip()


class SelfNotesCompressor(Compressor):
    """Query-conditioned notes written by the READER'S OWN model (self-notes).

    Required by B08 leg-1's pre-registration
    (``STATUS.json.next_gate.notes_generator_must_be_the_reader_itself``,
    ``B08_LEG1_GATE_PREREG.md`` §4.1): *the same* ``Meta-Llama-3-8B`` that reads
    also writes the notes, from the *same* retrieved evidence. Using a second
    model (e.g. ``--compressor mom_notes``) would confound "notes help" with
    "the second model is good", which breaks the single-variable design.

    Design contract (each point maps to a specific hazard)
    -----------------------------------------------------
    1. **Shares the reader's already-loaded weights.** Construction takes a
       :class:`~longmemeval.reader.LocalHFReader` and uses ``reader.model`` /
       ``reader.tokenizer`` directly. No second ``from_pretrained``: "same
       model" is then a *fact*, not a claim, and resident weights do not double.
    2. **Generates from the SAME post-token-budget evidence list** the answer
       arms see. The harness calls ``compress()`` after ``_apply_token_budget``,
       so this holds by construction as long as nobody re-retrieves here.
    3. **Notes are generated ONCE per question and reused verbatim by BOTH notes
       arms** via ``notes_cache_path``. "Same notes text" is part of the frozen
       single variable; regenerating per arm would leak decoder nondeterminism
       into the contrast. The cache is also what makes the ``U`` metric
       computable at all, since ``U`` is defined against *that arm's own
       context*.
    4. **Greedy, matching the reader** (``do_sample=False, num_beams=1``),
       ``max_new_tokens`` default 128.
    5. **Instruction reused VERBATIM from** :class:`MoMNotesCompressor`
       (``_NOTES_INSTRUCTION``) so the notes prompt is not a new uncontrolled
       variable across compressors.

    The cache is a JSONL of ``{"question_id", "notes", "n_evidence",
    "evidence_round_ids", "source"}``. ``evidence_round_ids`` is recorded so a
    later audit can prove the notes were generated from the same evidence set
    the answer arm consumed. On a cache hit the persisted string is returned
    unchanged and no generation happens.
    """

    label = "SELF"

    def __init__(
        self,
        reader,
        max_new_tokens: int = 128,
        max_context_tokens: int = 8192,
        notes_cache_path: Optional[str] = None,
        allow_generate: bool = True,
    ):
        model = getattr(reader, "model", None)
        tokenizer = getattr(reader, "tokenizer", None)
        # The reader-identity constraint guards GENERATION: it exists so the
        # notes are written by the model that reads them. When
        # ``allow_generate=False`` this instance CANNOT generate (``compress``
        # raises on a cache miss), so every notes string it can return came
        # from a cache file whose provenance is recorded in that file. Requiring
        # 8B of weights to replay a JSONL would be theatre -- and it would make
        # the notes-only arm impossible to score on a CPU node.
        if allow_generate and (model is None or tokenizer is None):
            raise ValueError(
                "SelfNotesCompressor requires a reader exposing .model and "
                ".tokenizer (i.e. --reader local_hf / LocalHFReader) to GENERATE "
                f"notes. Got {type(reader).__name__}. The prereg requires the "
                "notes generator to BE the reader, so --compressor self_notes "
                "cannot generate with --reader stub or --reader openai. To "
                "replay an existing notes cache without a model, pass "
                "--notes_cache <file> --notes_cache_readonly."
            )
        if not allow_generate and not notes_cache_path:
            raise ValueError(
                "SelfNotesCompressor(allow_generate=False) needs a "
                "notes_cache_path: with generation disabled and no cache there "
                "is no source of notes at all, and the arm would silently "
                "degrade to the raw baseline."
            )
        self.reader = reader
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.notes_cache_path = notes_cache_path
        self.allow_generate = allow_generate

        self._cache: Dict[str, Dict[str, object]] = {}
        self._cache_hits = 0
        self._generated = 0
        if notes_cache_path:
            self._load_cache(notes_cache_path)

    # -- notes cache (the "same notes text across arms" guarantee) --------- #

    def _load_cache(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Fail LOUDLY on a malformed cache. A silently-skipped record
                # would make one arm regenerate its notes while the other used
                # the persisted string -- i.e. the frozen single variable would
                # quietly stop being frozen, with no error and no NaN.
                if "question_id" not in rec or "notes" not in rec:
                    raise ValueError(
                        f"{path}:{lineno}: notes-cache record is missing "
                        f"'question_id' and/or 'notes' (keys present: "
                        f"{sorted(rec)}). The notes cache is what makes both "
                        "notes arms see IDENTICAL notes text, so a malformed "
                        "record is a protocol failure, not a warning."
                    )
                qid = rec["question_id"]
                if not isinstance(qid, str):
                    raise ValueError(
                        f"{path}:{lineno}: question_id must be a str, got "
                        f"{type(qid).__name__} ({qid!r}). A non-str id silently "
                        "misses every dict lookup keyed on the string id, so "
                        "the arm would regenerate notes while reporting a cache."
                    )
                self._cache[qid] = rec

    def _append_cache(self, rec: Dict[str, object]) -> None:
        if not self.notes_cache_path:
            return
        parent = os.path.dirname(os.path.abspath(self.notes_cache_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.notes_cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "notes_cache_records": len(self._cache),
            "notes_cache_hits": self._cache_hits,
            "notes_generated": self._generated,
        }

    # -- generation -------------------------------------------------------- #

    def _build_prompt(self, question: str, evidence: List[Evidence]) -> str:
        """Capped rounds context + the VERBATIM MoMNotesCompressor instruction."""
        context_text = "\n\n".join(ev.text for ev in evidence).strip()
        instruction = _NOTES_INSTRUCTION.format(question=question)
        instr_ids = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        ctx_ids = self.tokenizer(context_text, add_special_tokens=False)["input_ids"]
        ctx_cap = max(256, self.max_context_tokens - len(instr_ids))
        if len(ctx_ids) > ctx_cap:
            context_text = self.tokenizer.decode(
                ctx_ids[:ctx_cap], skip_special_tokens=True
            )
        return context_text + instruction

    def compress(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
        question_id: Optional[str] = None,
    ) -> str:
        if question_id is not None and question_id in self._cache:
            self._cache_hits += 1
            return str(self._cache[question_id]["notes"])
        if not evidence:
            return ""
        if not self.allow_generate:
            raise RuntimeError(
                "SelfNotesCompressor(allow_generate=False) was asked to GENERATE "
                f"notes for question_id={question_id!r}: it is not in the notes "
                f"cache ({self.notes_cache_path!r}, {len(self._cache)} records). "
                "Both notes arms must read the SAME persisted notes; generating "
                "here would silently unfreeze the single variable."
            )

        torch = _import_torch()
        prompt = self._build_prompt(question, evidence)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            self.model.device
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = out[0][input_ids.shape[1]:]
        notes = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        self._generated += 1

        if question_id is not None:
            rec = {
                "question_id": question_id,
                "notes": notes,
                "n_evidence": len(evidence),
                "evidence_round_ids": [ev.round_id for ev in evidence],
                "source": "self_notes",
                "max_new_tokens": self.max_new_tokens,
            }
            self._cache[question_id] = rec
            self._append_cache(rec)
        return notes


def build_compressor(name: str, args=None, reader=None) -> Compressor:
    """Factory used by run_baseline. ``name`` in {none, mom_notes, self_notes}.

    ``reader`` is required by ``self_notes`` (the prereg's
    ``notes_generator_must_be_the_reader_itself``): the compressor shares the
    reader's already-loaded weights instead of loading a second model.
    """
    name = (name or "none").lower()
    if name == "none":
        return IdentityCompressor()
    if name == "self_notes":
        if reader is None:
            raise ValueError(
                "--compressor self_notes requires the reader (the notes "
                "generator MUST be the reader itself; see "
                "B08_LEG1_GATE_PREREG.md 4.1)"
            )
        return SelfNotesCompressor(
            reader=reader,
            max_new_tokens=getattr(args, "compressor_max_new_tokens", 128),
            notes_cache_path=getattr(args, "notes_cache", None),
            allow_generate=not getattr(args, "notes_cache_readonly", False),
        )
    if name == "mom_notes":
        if args is None:
            raise ValueError("mom_notes compressor requires CLI args (checkpoint/config)")
        if not args.compressor_checkpoint or not args.compressor_adapter_config:
            raise ValueError(
                "--compressor mom_notes requires --compressor_checkpoint and "
                "--compressor_adapter_config"
            )
        return MoMNotesCompressor(
            checkpoint=args.compressor_checkpoint,
            adapter_config=args.compressor_adapter_config,
            model_path=args.mom_model_path,
            device=args.compressor_device,
            chunk_size=args.mom_chunk_size,
            max_new_tokens=args.compressor_max_new_tokens,
        )
    raise ValueError(
        f"unknown compressor: {name} (use 'none', 'self_notes' or 'mom_notes')"
    )
