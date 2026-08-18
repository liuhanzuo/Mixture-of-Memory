"""Paper notation to atomic tokenizer-string mapping.

The paper/spec uses readable labels such as ``[FUNC]``. The tokenizer uses rare
Qwen-style special-token strings to avoid collisions with literal prompt text.
"""

from __future__ import annotations


TOKEN_TEXT = {
    # Keep DreamOn compatibility: when added to a Dream-Coder tokenizer,
    # <|expand|> receives the first free ID, 151667.
    "[expand]": "<|expand|>",
    "[delete]": "<|sc_delete|>",
    "[FUNC]": "<|sc_func|>",
    "[CLASS]": "<|sc_class|>",
    "[FOR]": "<|sc_for|>",
    "[WHILE]": "<|sc_while|>",
    "[IF]": "<|sc_if|>",
    "[ELIF]": "<|sc_elif|>",
    "[ELSE]": "<|sc_else|>",
    "[TRY]": "<|sc_try|>",
    "[EXCEPT]": "<|sc_except|>",
    "[FINALLY]": "<|sc_finally|>",
    "[WITH]": "<|sc_with|>",
    "[MATCH]": "<|sc_match|>",
    "[HDR]": "<|sc_hdr|>",
    "[DOC]": "<|sc_doc|>",
    "[BODY]": "<|sc_body|>",
    "[CLAUSES]": "<|sc_clauses|>",
    "[STMT]": "<|sc_stmt|>",
}

ALL_TOKEN_TEXTS = tuple(TOKEN_TEXT.values())
