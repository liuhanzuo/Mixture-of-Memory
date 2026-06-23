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


def build_reader(reader_type: str, model: Optional[str] = None) -> Reader:
    reader_type = (reader_type or "stub").lower()
    if reader_type == "stub":
        return StubReader()
    if reader_type in ("openai", "gpt4o", "openai-chat"):
        return OpenAIChatReader(model=model)
    raise ValueError(f"unknown reader type: {reader_type} (use 'stub' or 'openai')")
