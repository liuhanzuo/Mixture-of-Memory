#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OLMo-2-7B linguistic + next-token layer-wise probe (Paper B two-depths).

Thin wrapper around scripts/probe_linguistic_layerwise.py -- the model-agnostic
Tenney edge-probing + logit-lens driver that produced
``results/probe_linguistic_qwen3_8b.json``. Running OLMo-2 through the SAME code
guarantees an identical method and output schema, so the two-depths table gets
OLMo-2's semantic (WiC/SST2/RTE per-layer linear probe) sat95 depth and its
next-token logit-lens sat95 depth to sit beside the existing OLMo-2 MMLU
knowledge logit-lens (and to compare against Qwen's 0.13L / 0.944L).

The layerwise driver is already generic: the logit-lens locates the final norm
via ``getattr(model.model, "norm")`` and the head via
``model.get_output_embeddings()`` -- both valid for Olmo2ForCausalLM (untied
head), so no OLMo-specific code is needed here.

Defaults (overridable -- ANY flag accepted by probe_linguistic_layerwise.py is
passed straight through):
  --model_path  ../models/OLMo-2-1124-7B   (32 layers -> 33 hidden states)
  --out         results/probe_linguistic_olmo2_7b.json

Forward-only, single GPU. Example:
  CUDA_VISIBLE_DEVICES=7 python scripts/probe_linguistic_olmo2.py
  CUDA_VISIBLE_DEVICES=7 python scripts/probe_linguistic_olmo2.py --tasks WiC,SST2,RTE
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import probe_linguistic_layerwise as base  # noqa: E402

_PROOT = os.path.dirname(_HERE)  # project root (.../Mixture-of-Memory)
DEFAULT_MODEL = os.path.normpath(
    os.path.join(_PROOT, "..", "models", "OLMo-2-1124-7B"))
DEFAULT_OUT = os.path.join(_PROOT, "results", "probe_linguistic_olmo2_7b.json")


def _has(argv, flag):
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def main():
    argv = list(sys.argv[1:])
    if not _has(argv, "--model_path"):
        argv += ["--model_path", DEFAULT_MODEL]
    if not _has(argv, "--out"):
        argv += ["--out", DEFAULT_OUT]
    sys.argv = [sys.argv[0]] + argv
    base.main()


if __name__ == "__main__":
    main()
