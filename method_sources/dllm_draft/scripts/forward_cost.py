#!/usr/bin/env python3
"""Shared forward-pass cost instrumentation -- the SINGLE implementation.

Extracted verbatim from ``scripts/generate_evalplus_ar.py`` so that every arm
(diffusion and AR, on either physical disk) is measured by identical
instrumentation instead of a re-implementation that could silently drift.
``generate_evalplus_ar.py`` re-exports these names for backwards compatibility.

Why a separate module: importing the tracker from ``generate_evalplus_ar``
transitively pulls in ``generate_evalplus_dream`` -> ``scaffold_coder``, which
is not importable on every node. The cost axes must not depend on the scaffold
package being installed.

``tokens_fed`` and ``attended_context_sum`` are the two axes comparable across
model families. ``forward_passes`` (NFE) is NOT comparable: one diffusion step
re-feeds a whole canvas, one AR decode step feeds a single token.
"""

from __future__ import annotations

class ForwardCostTracker:
    """Measure per-forward-pass token cost via a top-level forward pre-hook.

    Records, for each ``forward`` invocation on the wrapped module:
      * ``new_tokens``: width of the ``input_ids`` (or ``inputs_embeds``) fed;
      * ``attended``:   cached prefix length + ``new_tokens``.
    """

    def __init__(self) -> None:
        self.new_tokens: list[int] = []
        self.attended: list[int] = []
        self.enabled = False

    def reset(self) -> None:
        self.new_tokens = []
        self.attended = []

    def hook(self, module, args, kwargs) -> None:  # noqa: ANN001
        if not self.enabled:
            return
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        embeds = kwargs.get("inputs_embeds")
        if ids is not None and hasattr(ids, "shape"):
            width = int(ids.shape[-1])
        elif embeds is not None and hasattr(embeds, "shape"):
            width = int(embeds.shape[1])
        else:
            return

        total = None
        cache_position = kwargs.get("cache_position")
        if cache_position is not None and getattr(cache_position, "numel", lambda: 0)():
            # Most reliable signal in transformers >= 4.40: absolute positions
            # of the tokens being fed on this pass.
            total = int(cache_position[-1].item()) + 1
        if total is None:
            past = kwargs.get("past_key_values")
            past_len = 0
            if past is not None:
                try:
                    past_len = int(past.get_seq_length())
                except Exception:
                    try:
                        past_len = int(past[0][0].shape[-2])
                    except Exception:
                        past_len = 0
            total = past_len + width

        self.new_tokens.append(width)
        self.attended.append(total)

    def summary(self) -> dict:
        return {
            "forward_passes": len(self.new_tokens),
            "tokens_fed": int(sum(self.new_tokens)),
            "attended_context_sum": int(sum(self.attended)),
            "per_pass_new_tokens": list(self.new_tokens),
            "per_pass_attended": list(self.attended),
        }


def analytic_cost(prompt_tokens: int, generated_tokens: int) -> dict:
    """Closed-form AR-with-KV-cache prediction, for cross-checking the hook."""
    passes = max(1, generated_tokens)
    decode_steps = max(0, generated_tokens - 1)
    return {
        "forward_passes": passes,
        "tokens_fed": prompt_tokens + decode_steps,
        "attended_context_sum": (
            prompt_tokens
            + decode_steps * prompt_tokens
            + decode_steps * (decode_steps + 1) // 2
        ),
    }


