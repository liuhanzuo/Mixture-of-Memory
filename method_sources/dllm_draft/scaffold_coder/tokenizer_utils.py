"""Tokenizer extension and embedding-row initialization utilities."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AddedToken, PreTrainedTokenizerBase

from .errors import RuntimeInvariantError
from .special_tokens import TOKEN_TEXT


INITIALIZATION_TEXT = {
    "[expand]": ("expand", "mask"),
    "[delete]": ("delete", "remove"),
    "[FUNC]": ("def", "function"),
    "[CLASS]": ("class",),
    "[FOR]": ("for",),
    "[WHILE]": ("while",),
    "[IF]": ("if",),
    "[ELIF]": ("elif", "else if"),
    "[ELSE]": ("else",),
    "[TRY]": ("try",),
    "[EXCEPT]": ("except",),
    "[FINALLY]": ("finally",),
    "[WITH]": ("with",),
    "[MATCH]": ("match",),
    "[HDR]": ("header", "condition", "arguments"),
    "[DOC]": ("docstring", "documentation"),
    "[BODY]": ("body", "block"),
    "[CLAUSES]": ("clauses", "else except finally"),
    "[STMT]": ("statement", "line"),
}

EDIT_SOURCE_TEXT = ("expand", "mask", "delete", "remove")


@dataclass(frozen=True, slots=True)
class TokenExtension:
    notation: str
    physical: str
    token_id: int
    existed_before: bool


def extend_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[TokenExtension, ...]:
    """Add tokens in deterministic DreamOn-compatible order."""

    before_vocab = tokenizer.get_vocab()
    additions = [
        AddedToken(
            physical,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=True,
        )
        for physical in TOKEN_TEXT.values()
        if physical not in before_vocab
    ]
    if additions:
        signature = inspect.signature(tokenizer.add_special_tokens)
        kwargs: dict[str, bool] = {}
        if "replace_additional_special_tokens" in signature.parameters:
            kwargs["replace_additional_special_tokens"] = False
        elif "replace_extra_special_tokens" in signature.parameters:
            kwargs["replace_extra_special_tokens"] = False
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additions},
            **kwargs,
        )

    result: list[TokenExtension] = []
    for notation, physical in TOKEN_TEXT.items():
        ids = tokenizer.encode(physical, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeInvariantError(
                f"special token {notation}={physical!r} is not atomic: {ids}"
            )
        result.append(
            TokenExtension(
                notation=notation,
                physical=physical,
                token_id=ids[0],
                existed_before=physical in before_vocab,
            )
        )
    return tuple(result)


def extend_dreamon_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[TokenExtension, ...]:
    """Add only DreamOn's expand token in the canonical reserved row."""

    physical = "<|expand|>"
    before_vocab = tokenizer.get_vocab()
    if physical not in before_vocab:
        token = AddedToken(
            physical,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=True,
        )
        signature = inspect.signature(tokenizer.add_special_tokens)
        kwargs: dict[str, bool] = {}
        if "replace_additional_special_tokens" in signature.parameters:
            kwargs["replace_additional_special_tokens"] = False
        elif "replace_extra_special_tokens" in signature.parameters:
            kwargs["replace_extra_special_tokens"] = False
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [token]},
            **kwargs,
        )
    ids = tokenizer.encode(physical, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeInvariantError(f"DreamOn expand token is not atomic: {ids}")
    return (
        TokenExtension(
            notation="[expand]",
            physical=physical,
            token_id=ids[0],
            existed_before=physical in before_vocab,
        ),
    )


@torch.no_grad()
def initialize_model_token_rows(
    model,
    tokenizer: PreTrainedTokenizerBase,
    extensions: Iterable[TokenExtension],
) -> dict[str, dict[str, object]]:
    """Initialize newly allocated input and output rows from related words.

    Existing tokens such as DreamOn's trained ``<|expand|>`` are preserved.
    The model matrices must already be large enough for every assigned ID.
    """

    input_weight = model.get_input_embeddings().weight
    output_layer = model.get_output_embeddings()
    if output_layer is None:
        raise RuntimeInvariantError("model does not expose output embeddings")
    output_weight = output_layer.weight

    report: dict[str, dict[str, object]] = {}
    for extension in extensions:
        token_id = extension.token_id
        if token_id >= input_weight.shape[0] or token_id >= output_weight.shape[0]:
            raise RuntimeInvariantError(
                f"token {extension.notation} id={token_id} exceeds model rows "
                f"input={input_weight.shape[0]} output={output_weight.shape[0]}"
            )
        if extension.existed_before:
            report[extension.notation] = {
                "token_id": token_id,
                "preserved": True,
                "source_ids": [],
            }
            continue

        source_ids = _source_ids(
            tokenizer, INITIALIZATION_TEXT[extension.notation]
        )
        input_value = input_weight[source_ids].mean(dim=0)
        output_value = output_weight[source_ids].mean(dim=0)
        input_weight[token_id].copy_(input_value)
        output_weight[token_id].copy_(output_value)
        report[extension.notation] = {
            "token_id": token_id,
            "preserved": False,
            "source_ids": source_ids,
            "source_text": list(INITIALIZATION_TEXT[extension.notation]),
        }
    return report


def validate_ids_within_model(
    extensions: Iterable[TokenExtension], vocab_size: int
) -> None:
    for extension in extensions:
        if extension.token_id >= vocab_size:
            raise RuntimeInvariantError(
                f"{extension.notation} id={extension.token_id} is outside "
                f"configured vocab_size={vocab_size}; model resizing would be required"
            )


def source_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    source_texts: Iterable[str],
) -> tuple[int, ...]:
    """Return ordinary vocabulary rows used to seed structural tokens."""

    return tuple(sorted(set(_source_ids(tokenizer, source_texts))))


def edit_source_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[int, ...]:
    """Rows that can leak literal edit words when edit tokens are frozen."""

    return source_token_ids(tokenizer, EDIT_SOURCE_TEXT)


def _source_ids(
    tokenizer: PreTrainedTokenizerBase, source_texts: Iterable[str]
) -> list[int]:
    source_ids: list[int] = []
    special_ids = set(tokenizer.all_special_ids)
    for text in source_texts:
        source_ids.extend(
            token_id
            for token_id in tokenizer.encode(text, add_special_tokens=False)
            if token_id not in special_ids
        )
    if not source_ids:
        raise RuntimeInvariantError(
            f"no ordinary source IDs for initialization texts {source_texts}"
        )
    return source_ids
