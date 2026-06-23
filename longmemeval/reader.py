"""Reader modules: answer a question from retrieved evidence.

The reader is pluggable behind the :class:`Reader` interface so the retrieval
pipeline can be evaluated independently of the answering LLM:

  * :class:`StubReader`      - no LLM; deterministic, dependency-free. Used for
                               pipeline smoke tests and recall@k diagnostics.
  * :class:`OpenAIChatReader`- calls an OpenAI-compatible chat API (GPT-4o
                               style). API key + base URL come from env vars
                               (never hardcoded). Works with any
                               OpenAI-compatible endpoint (vLLM, etc).

Reader prompt follows the paper's Chain-of-Note / structured-JSON-reader
recommendation: evidence is presented as numbered structured blocks with
session ids + timestamps, and the model is asked to reason over notes before
answering. Temporal-reasoning questions benefit from the explicit dates.
"""

from __future__ import annotations

import json
import os
import textwrap
from abc import ABC, abstractmethod
from typing import List, Optional

from .backends import Evidence


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a long-term memory assistant. You are given a user's QUESTION and
    a set of EVIDENCE blocks retrieved from the user's past conversation
    sessions. Each evidence block is tagged with its session id and the date
    of that session.

    Instructions:
    1. Read the evidence and write brief notes on which blocks are relevant
       (Chain-of-Note). Pay attention to dates for time-sensitive questions
       and prefer the most recent information when facts were updated.
    2. If the evidence does not contain the answer, respond that you don't
       know rather than guessing (abstention).
    3. Output ONLY a JSON object: {"notes": "...", "answer": "..."}.
       The "answer" must be concise and directly answer the question.
    """
)


def build_prompt(question: str, question_date: str, evidence: List[Evidence], token_budget: int = 0) -> str:
    """Build the user message: question + date + structured evidence blocks.

    ``token_budget`` (>0) approximately caps the evidence section using a
    whitespace word-count proxy (no tokenizer dependency at the prompt layer;
    the harness applies a real token budget upstream).
    """
    blocks = []
    used = 0
    for idx, ev in enumerate(evidence, start=1):
        block = ev.as_block(idx)
        if token_budget and token_budget > 0:
            approx = len(block.split())
            if used + approx > token_budget and blocks:
                break
            used += approx
        blocks.append(block)
    evidence_text = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    date_line = f"QUESTION DATE: {question_date}\n" if question_date else ""
    return f"{date_line}QUESTION: {question}\n\nEVIDENCE:\n{evidence_text}"


class Reader(ABC):
    """Abstract reader interface."""

    @abstractmethod
    def answer(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
        token_budget: int = 0,
    ) -> str:
        """Return the hypothesis answer string."""


class StubReader(Reader):
    """Dependency-free reader for smoke tests / recall-only evaluation.

    It does not call any LLM. It returns the top evidence's assistant text
    (or a marker) so the end-to-end pipeline can be exercised without an API.
    Useful to validate retrieval recall@k independently of answer quality.
    """

    def answer(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
        token_budget: int = 0,
    ) -> str:
        if not evidence:
            return "I don't know."
        top = evidence[0]
        # Return a compact, deterministic summary of the top evidence.
        snippet = top.text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300]
        return f"[stub|session={top.session_id}|date={top.session_date}] {snippet}"


class OpenAIChatReader(Reader):
    """OpenAI-compatible chat reader (GPT-4o style).

    Configuration is entirely via environment variables (never hardcode keys):
        LONGMEMEVAL_READER_API_KEY  (or OPENAI_API_KEY)
        LONGMEMEVAL_READER_BASE_URL (or OPENAI_BASE_URL; optional)
        LONGMEMEVAL_READER_MODEL    (default "gpt-4o")

    Requires the ``openai`` package (>=1.0). If it is not installed or no key
    is set, construction raises with a clear message; the CLI surfaces this.
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.api_key = os.environ.get("LONGMEMEVAL_READER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("LONGMEMEVAL_READER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("LONGMEMEVAL_READER_MODEL", "gpt-4o")
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError(
                "OpenAIChatReader requires an API key in LONGMEMEVAL_READER_API_KEY "
                "or OPENAI_API_KEY (never hardcode it)."
            )
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "The 'openai' package is not installed in this venv. "
                "Install it or use --reader stub. (import error: %r)" % (e,)
            )
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def answer(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
        token_budget: int = 0,
    ) -> str:
        user_msg = build_prompt(question, question_date, evidence, token_budget)
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        content = resp.choices[0].message.content or ""
        # Try to parse the structured JSON; fall back to raw content.
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"]).strip()
        except Exception:
            pass
        return content.strip()


class LocalHFReader(Reader):
    """Local HuggingFace causal-LM reader (e.g. Llama-3-8B).

    Loads a local HF model with ``transformers`` and answers questions by
    feeding the retrieved evidence rounds into a chat-style QA prompt and
    greedily decoding a short hypothesis. No external API is needed; the model
    weights are read from a local path.

    The reader respects ``CUDA_VISIBLE_DEVICES`` (the caller pins a single GPU)
    and loads in bfloat16 on cuda. To stay within memory, the evidence text is
    truncated by the tokenizer so the full prompt is <= ``max_prompt_tokens``,
    leaving room for ``max_new_tokens`` of generation.
    """

    def __init__(
        self,
        model_path: str = "models/Meta-Llama-3-8B",
        max_new_tokens: int = 64,
        max_prompt_tokens: int = 7000,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "LocalHFReader requires torch + transformers in this venv "
                "(import error: %r)" % (e,)
            )
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

    def _truncate_evidence_block(self, evidence_text: str, reserve_tokens: int) -> str:
        """Token-truncate the evidence section so the whole prompt fits.

        ``reserve_tokens`` is the budget consumed by everything that is NOT the
        evidence text (instruction, question, scaffolding). We cap the evidence
        to ``max_prompt_tokens - reserve_tokens``.
        """
        cap = self.max_prompt_tokens - reserve_tokens
        if cap <= 0:
            cap = max(256, self.max_prompt_tokens // 2)
        ids = self.tokenizer(evidence_text, add_special_tokens=False)["input_ids"]
        if len(ids) <= cap:
            return evidence_text
        truncated = self.tokenizer.decode(ids[:cap], skip_special_tokens=True)
        return truncated + "\n[... evidence truncated ...]"

    def _build_prompt(self, question: str, question_date: str, evidence: List[Evidence]) -> str:
        # Build the evidence section from ALL retrieved rounds (not just top-1).
        blocks = [ev.as_block(i) for i, ev in enumerate(evidence, start=1)]
        evidence_text = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"

        date_line = f"Today's date is {question_date}.\n" if question_date else ""
        # Direct-completion QA prompt. Meta-Llama-3-8B is a *base* model, so a
        # concise instruction ending in "ANSWER:" elicits a short answer far
        # more reliably than a JSON / chain-of-note format (which the base model
        # tends to ramble past the short token budget).
        instruction = (
            "You are a long-term memory assistant. Using ONLY the evidence "
            "below from the user's past conversations, answer the question with "
            "a short, direct answer. Pay attention to dates and prefer the most "
            "recent information when facts were updated. If the evidence does "
            "not contain the answer, reply \"I don't know\".\n"
        )
        head = instruction
        tail = f"\n\n{date_line}QUESTION: {question}\nANSWER:"
        reserve = (
            len(self.tokenizer(head + "\n\nEVIDENCE:\n" + tail, add_special_tokens=False)["input_ids"])
            + self.max_new_tokens
            + 16
        )
        evidence_text = self._truncate_evidence_block(evidence_text, reserve)
        return f"{head}\nEVIDENCE:\n{evidence_text}{tail}"

    def _build_inputs(self, question: str, question_date: str, evidence: List[Evidence]):
        prompt = self._build_prompt(question, question_date, evidence)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return input_ids.to(self.model.device)

    def _extract_answer(self, text: str) -> str:
        text = text.strip()
        # Base model may keep generating extra turns; keep only the first line/
        # sentence-ish span and stop at obvious continuation markers.
        for marker in ("\nQUESTION:", "\nEVIDENCE:", "\nANSWER:", "\n\n"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def answer(
        self,
        question: str,
        question_date: str,
        evidence: List[Evidence],
        token_budget: int = 0,
    ) -> str:
        torch = self._torch
        input_ids = self._build_inputs(question, question_date, evidence)
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
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._extract_answer(text)


def build_reader(reader_type: str, model: Optional[str] = None) -> Reader:
    reader_type = (reader_type or "stub").lower()
    if reader_type == "stub":
        return StubReader()
    if reader_type in ("openai", "gpt4o", "openai-chat"):
        return OpenAIChatReader(model=model)
    if reader_type in ("local", "hf", "local_hf"):
        return LocalHFReader(model_path=model or "models/Meta-Llama-3-8B")
    raise ValueError(
        f"unknown reader type: {reader_type} (use 'stub', 'openai', or 'local')"
    )
