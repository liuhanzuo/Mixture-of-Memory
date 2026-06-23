"""Data loading for the official LongMemEval JSON format.

The official benchmark (https://github.com/xiaowu0162/LongMemEval) ships
three files: ``longmemeval_s``, ``longmemeval_m`` and
``longmemeval_oracle``. Each is a JSON *list* of question objects with the
following fields (per the paper / repo):

    question_id        : str   - unique id; "_abs" suffix marks abstention qs
    question_type      : str   - one of the 5 ability categories
    question           : str   - the user question
    answer             : str   - gold answer (string)
    question_date      : str   - timestamp when the question is asked
    haystack_sessions  : list  - list of sessions; each session is a list of
                                 turn dicts {"role": "user"/"assistant",
                                 "content": str, optional "has_answer": bool}
    haystack_dates     : list  - timestamp (str) per session, aligned with
                                 haystack_sessions
    haystack_session_ids : list - session id (str) per session (some dumps
                                 omit this; we synthesize stable ids)
    answer_session_ids : list  - the session ids that contain the evidence

This loader is tolerant of minor schema drift (missing session ids, turn
content under "text" instead of "content", etc.).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class Round:
    """A user+assistant exchange (the paper's ROUND granularity).

    The LongMemEval paper found round-level memory granularity beats
    session-level. A round bundles one user turn with the assistant
    response(s) that immediately follow it.
    """

    round_id: str
    session_id: str
    session_date: str
    user_text: str
    assistant_text: str
    turn_indices: Tuple[int, ...] = ()

    @property
    def text(self) -> str:
        parts = []
        if self.user_text:
            parts.append(f"User: {self.user_text}")
        if self.assistant_text:
            parts.append(f"Assistant: {self.assistant_text}")
        return "\n".join(parts)


@dataclass
class LongMemEvalExample:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_sessions: List[List[Dict[str, Any]]]
    haystack_dates: List[str]
    haystack_session_ids: List[str]
    answer_session_ids: List[str]
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_abstention(self) -> bool:
        # Abstention questions are marked with an "_abs" suffix in the id.
        return self.question_id.endswith("_abs") or self.question_type == "abstention"


def _turn_content(turn: Dict[str, Any]) -> str:
    for key in ("content", "text", "value", "message"):
        if key in turn and turn[key] is not None:
            return str(turn[key]).strip()
    return ""


def _turn_role(turn: Dict[str, Any]) -> str:
    role = turn.get("role") or turn.get("speaker") or turn.get("from") or ""
    role = str(role).lower()
    if role in ("human", "usr"):
        return "user"
    if role in ("ai", "bot", "gpt", "asst"):
        return "assistant"
    return role


def _parse_example(obj: Dict[str, Any]) -> LongMemEvalExample:
    qid = str(obj.get("question_id", obj.get("id", "unknown")))
    sessions = obj.get("haystack_sessions") or obj.get("sessions") or []
    dates = obj.get("haystack_dates") or obj.get("session_dates") or []

    # Session ids: some dumps omit them. Synthesize stable per-example ids.
    session_ids = (
        obj.get("haystack_session_ids")
        or obj.get("session_ids")
        or [f"{qid}_sess{i}" for i in range(len(sessions))]
    )

    # Align lengths defensively.
    if len(dates) < len(sessions):
        dates = list(dates) + [""] * (len(sessions) - len(dates))
    if len(session_ids) < len(sessions):
        session_ids = list(session_ids) + [
            f"{qid}_sess{i}" for i in range(len(session_ids), len(sessions))
        ]

    answer_session_ids = (
        obj.get("answer_session_ids")
        or obj.get("answer_session_id")
        or []
    )
    if isinstance(answer_session_ids, str):
        answer_session_ids = [answer_session_ids]

    return LongMemEvalExample(
        question_id=qid,
        question_type=str(obj.get("question_type", obj.get("type", ""))),
        question=str(obj.get("question", "")),
        answer=str(obj.get("answer", "")),
        question_date=str(obj.get("question_date", obj.get("date", ""))),
        haystack_sessions=[list(s) for s in sessions],
        haystack_dates=[str(d) for d in dates],
        haystack_session_ids=[str(s) for s in session_ids],
        answer_session_ids=[str(s) for s in answer_session_ids],
        raw=obj,
    )


def load_longmemeval(
    path: str,
    limit: Optional[int] = None,
) -> List[LongMemEvalExample]:
    """Load a LongMemEval JSON file into typed examples.

    Args:
        path: path to ``longmemeval_s.json`` (or _m / _oracle, or a JSONL).
        limit: optionally load only the first ``limit`` questions.

    Raises:
        FileNotFoundError: with a pointer to the download instructions if the
            file is absent (see longmemeval/README.md).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LongMemEval data not found at: {path}\n"
            "Download it from https://github.com/xiaowu0162/LongMemEval "
            "(HuggingFace dataset 'xiaowu0162/longmemeval'); see "
            "longmemeval/README.md for instructions. No data is bundled."
        )

    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            # JSONL fallback
            data = [json.loads(line) for line in f if line.strip()]

    if limit is not None:
        data = data[:limit]
    return [_parse_example(o) for o in data]


def iter_rounds(example: LongMemEvalExample) -> Iterator[Round]:
    """Split an example's haystack into ROUND-level units.

    A round = one user turn + the assistant turn(s) until the next user turn.
    This is the paper's recommended memory granularity.
    """
    for s_idx, (session, date, sid) in enumerate(
        zip(
            example.haystack_sessions,
            example.haystack_dates,
            example.haystack_session_ids,
        )
    ):
        round_idx = 0
        pending_user: Optional[str] = None
        pending_assist: List[str] = []
        pending_turns: List[int] = []

        def _flush(_round_idx: int):
            nonlocal pending_user, pending_assist, pending_turns
            if pending_user is None and not pending_assist:
                return None
            r = Round(
                round_id=f"{sid}_r{_round_idx}",
                session_id=sid,
                session_date=date,
                user_text=pending_user or "",
                assistant_text="\n".join(pending_assist),
                turn_indices=tuple(pending_turns),
            )
            pending_user = None
            pending_assist = []
            pending_turns = []
            return r

        for t_idx, turn in enumerate(session):
            role = _turn_role(turn)
            content = _turn_content(turn)
            if role == "user":
                # New user turn starts a new round; flush the previous one.
                flushed = _flush(round_idx)
                if flushed is not None:
                    yield flushed
                    round_idx += 1
                pending_user = content
                pending_turns = [t_idx]
            else:
                pending_assist.append(content)
                pending_turns.append(t_idx)

        flushed = _flush(round_idx)
        if flushed is not None:
            yield flushed
