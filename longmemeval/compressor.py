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

from abc import ABC, abstractmethod
from typing import List

from .backends import Evidence


class Compressor(ABC):
    """Abstract evidence compressor.

    Given the question and the retrieved evidence, return a compact *notes*
    string (may be empty). The harness prepends a non-empty notes string as an
    extra, clearly-labeled evidence block in front of the raw evidence; an empty
    string means "no notes" (the evidence is passed through unchanged).
    """

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
        import json

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
        instruction = (
            "\n\nSummarize the facts from the conversation above that are "
            f"relevant to answering this question: {question}\n"
            "Relevant facts:"
        )

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


def build_compressor(name: str, args=None) -> Compressor:
    """Factory used by run_baseline. ``name`` in {none, mom_notes}."""
    name = (name or "none").lower()
    if name == "none":
        return IdentityCompressor()
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
    raise ValueError(f"unknown compressor: {name} (use 'none' or 'mom_notes')")
