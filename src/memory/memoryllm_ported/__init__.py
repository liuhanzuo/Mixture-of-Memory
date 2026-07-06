"""MemoryLLM ported to transformers 5.5.4 / torch 2.10 / sm_100 (L20A / B200).

Self-contained copy of the official MemoryLLM (YuWangX/memoryllm-8b-chat) modeling
file, adapted from its transformers 4.43-era assumptions so it loads and runs on the
local B200/L20A .venv. See the header of ``modeling_memoryllm.py`` for the exact list
of adaptations (all tagged ``# PORT:``).

Usage::

    import torch
    from transformers import AutoConfig, AutoTokenizer
    from src.memory.memoryllm_ported import MemoryLLM

    path = "/apdcephfs_wzc1/share_304376610/pighzliu_code/MemoryLLM-source"
    model = MemoryLLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(path)

The config is a plain ``LlamaConfig`` (model_type="llama") with the extra MemoryLLM
fields (num_blocks / num_tokens / lora_config / add_bos_embedding / ...), so
``AutoConfig.from_pretrained`` / ``LlamaConfig`` load it directly — no separate
configuration module is required.
"""

from .modeling_memoryllm import (  # noqa: F401
    MemoryLLM,
    LlamaForCausalLM,
    LlamaModel,
    LlamaConfig,
)

__all__ = ["MemoryLLM", "LlamaForCausalLM", "LlamaModel", "LlamaConfig"]
