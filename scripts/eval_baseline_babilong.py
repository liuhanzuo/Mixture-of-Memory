"""Unified BABILong-100 evaluator for 4 baselines.

Baselines:
    --baseline llama32_1b_instruct  : Llama-3.2-1B-Instruct (vanilla HF)
    --baseline memoryllm            : MemoryLLM-8B-chat (inject_memory + chunked)
    --baseline mplus                : M+ (inject_memory + chunked)
    --baseline beacon               : Activation Beacon Qwen2-7B-Instruct (trust_remote_code)

Outputs CSVs in <results_folder>/<output_name>/<task>_<length>_<suffix>.csv
matching the format of babilong/scripts/run_model_on_babilong.py so we can
re-use score_babilong_results.py.

Designed for transformers 4.46.3 on H20.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

# ---- path setup ----
H20_BABILONG = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/babilong-pkg"
H20_MEMORYLLM_SRC = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/MemoryLLM-source"
LOCAL_BABILONG = "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong"
LOCAL_MEMORYLLM_SRC = "/apdcephfs_wzc1/share_303098609/pighzliu_code/MemoryLLM-source"

for p in (H20_BABILONG, LOCAL_BABILONG):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import datasets  # noqa: E402
from transformers.generation import GenerationMixin  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402


def _set_proxy() -> None:
    os.environ.setdefault("http_proxy", "http://star-proxy.oa.com:3128")
    os.environ.setdefault("https_proxy", "http://star-proxy.oa.com:3128")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


# ---------------------------------------------------------------------------
# Baseline: Llama-3.2-1B-Instruct  (and any plain HF causal LM)
# ---------------------------------------------------------------------------

def run_plain_hf(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[plain_hf] Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = None
    for impl in ("flash_attention_2", "sdpa"):
        try:
            print(f"[plain_hf] Trying attn_implementation={impl}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=impl,
                device_map="cuda:0",
            )
            print(f"[plain_hf] Loaded with {impl}")
            break
        except (ValueError, ImportError, RuntimeError) as e:
            print(f"[plain_hf] {impl} failed: {e}")
    if model is None:
        raise RuntimeError("Could not load model with FA2 or SDPA")
    model.eval()

    _eval_loop(
        args,
        prepare_input=lambda ctx, q, examples, instr, post, use_chat: _hf_prepare_input(
            tokenizer, ctx, q, examples, instr, post, use_chat
        ),
        generate=lambda inputs: _hf_generate(model, tokenizer, inputs, args.max_new_tokens),
        per_sample_setup=None,
    )


def _hf_prepare_input(tokenizer, ctx, q, examples, instr, post, use_chat):
    text = get_formatted_input(ctx, q, examples, instr, post, template=DEFAULT_TEMPLATE)
    if use_chat:
        msgs = [{"role": "user", "content": text}]
        ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
        # transformers 5.x returns BatchEncoding; extract .input_ids tensor
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
    else:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    return ids.to("cuda:0")


def _hf_generate(model, tokenizer, input_ids, max_new):
    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            use_cache=False,
        )
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Baseline: MemoryLLM-8B-chat (inject_memory)
# ---------------------------------------------------------------------------

def run_memoryllm(args: argparse.Namespace) -> None:
    """MemoryLLM uses inject_memory(chunk) to load context, then generate(question)."""
    if H20_MEMORYLLM_SRC not in sys.path:
        sys.path.insert(0, H20_MEMORYLLM_SRC)
    if LOCAL_MEMORYLLM_SRC not in sys.path:
        sys.path.insert(0, LOCAL_MEMORYLLM_SRC)

    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from transformers.cache_utils import DynamicCache
    try:
        from modeling_memoryllm import MemoryLLM, LlamaModel as MemLlamaModel
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import MemoryLLM source from {H20_MEMORYLLM_SRC}: {e}"
        )

    # Shim 1: DynamicCache.from_legacy_cache (missing in older transformers)
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values):
            if past_key_values is None:
                return cls()
            return cls(ddp_cache_data=past_key_values)
        DynamicCache.from_legacy_cache = _from_legacy_cache

    # Shim 2: DynamicCache.to_legacy_cache (missing in older transformers)
    if not hasattr(DynamicCache, "to_legacy_cache"):
        def _to_legacy_cache(self):
            legacy = []
            for layer in getattr(self, "layers", []):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                legacy.append((keys, values))
            return tuple(legacy)
        DynamicCache.to_legacy_cache = _to_legacy_cache

    # Shim 3: MemoryLLM inherits prepare_inputs_for_generation from LlamaForCausalLM.
    # That method accesses cache_position[0] unconditionally when inputs_embeds is
    # passed, crashing when cache_position=None (older transformers won't pass it).
    # Patch MemoryLLM directly to synthesize cache_position when missing.
    if not getattr(MemoryLLM.prepare_inputs_for_generation,
                   "_memoryllm_cache_position_patch", False):
        _orig_memoryllm_prepare_inputs = MemoryLLM.prepare_inputs_for_generation

        def _memoryllm_prepare_inputs_with_cache_position(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            **kwargs,
        ):
            if cache_position is None:
                source = input_ids if input_ids is not None else inputs_embeds
                if source is not None:
                    past_seen = 0
                    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                        past_seen = past_key_values.get_seq_length()
                    elif past_key_values:
                        try:
                            past_seen = past_key_values[0][0].shape[-2]
                        except Exception:
                            past_seen = 0
                    cache_position = torch.arange(
                        past_seen,
                        past_seen + source.shape[1],
                        device=source.device,
                    )
            return _orig_memoryllm_prepare_inputs(
                self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

        _memoryllm_prepare_inputs_with_cache_position._memoryllm_cache_position_patch = True
        MemoryLLM.prepare_inputs_for_generation = _memoryllm_prepare_inputs_with_cache_position
        print("[memoryllm] Applied cache_position shim to MemoryLLM.prepare_inputs_for_generation")

    # Shim 4: inner LlamaModel stub (may not have prepare_inputs_for_generation)
    if not hasattr(MemLlamaModel, "prepare_inputs_for_generation"):
        def _memlm_prepare_inputs_for_generation(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            **kwargs,
        ):
            return {
                "input_ids": input_ids if input_ids is not None else kwargs.get("input_ids"),
                "past_key_values": (
                    past_key_values if past_key_values is not None else kwargs.get("past_key_values")
                ),
                "attention_mask": (
                    attention_mask if attention_mask is not None else kwargs.get("attention_mask")
                ),
            }
        MemLlamaModel.prepare_inputs_for_generation = _memlm_prepare_inputs_for_generation

    # Shim 5: MemoryLLM's forward() calls _update_causal_mask with:
    #   input_tensor = inputs_embeds (shape: [batch, input_len, d])
    #   cache_position = arange(0, input_len + num_tokens + bos) -- length = input_len + 257
    # The original _update_causal_mask builds causal_mask (seq_len, target_len) and does:
    #   causal_mask *= arange(target_len) > cache_position.reshape(-1,1)
    # This broadcasts (seq_len, target_len) against (cache_position_len, 1) -- fails if sizes differ.
    # Also, target_len must equal key_states.shape[-2] = input_len + 257 so attention add works.
    # Fix: keep seq_len = input_tensor.shape[1], but set target_len = max(cache_position) + 1,
    # and use cache_position[:seq_len] for the broadcast multiply.
    if not getattr(MemLlamaModel._update_causal_mask, "_memoryllm_causal_mask_patch", False):

        def _update_causal_mask_fixed(self, attention_mask, input_tensor, cache_position,
                                      past_key_values=None, output_attentions=False):
            import torch as _torch
            from transformers.cache_utils import StaticCache as _StaticCache
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter as _AMC
            seq_len = input_tensor.shape[1]
            past_key_values_obj = past_key_values
            past_seen_tokens = past_key_values_obj.get_seq_length() if past_key_values_obj is not None else 0
            using_static_cache = isinstance(past_key_values_obj, _StaticCache)
            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = _torch.finfo(dtype).min
            # SDPA short-circuit: return None when mask can be inferred
            if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
                if _AMC._ignore_causal_mask_sdpa(attention_mask, inputs_embeds=input_tensor,
                                                 past_key_values_length=past_seen_tokens,
                                                 is_training=self.training):
                    return None
            if using_static_cache:
                target_length = past_key_values_obj.get_max_length()
            else:
                if isinstance(attention_mask, _torch.Tensor):
                    target_length = attention_mask.shape[-1]
                elif cache_position is not None:
                    # Use full range of cache_position as target_length so the mask
                    # covers all key positions including prepended memory tokens.
                    target_length = int(cache_position[-1].item()) + 1
                else:
                    target_length = past_seen_tokens + seq_len + 1
            if attention_mask is not None and attention_mask.dim() == 4:
                if attention_mask.max() != 0:
                    raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0")
                return attention_mask
            causal_mask = _torch.full(
                (seq_len, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if seq_len != 1:
                causal_mask = _torch.triu(causal_mask, diagonal=1)
            if cache_position is not None:
                # Slice cache_position to seq_len for the broadcast multiply
                cp = cache_position[:seq_len]
                causal_mask *= _torch.arange(target_length, device=device) > cp.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            return causal_mask

        _update_causal_mask_fixed._memoryllm_causal_mask_patch = True
        MemLlamaModel._update_causal_mask = _update_causal_mask_fixed
        print("[memoryllm] Applied causal mask shim to LlamaModel._update_causal_mask")

    print(f"[memoryllm] Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Patch rope_theta if missing (transformers 4.46 may strip it from attached config)
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg_path = os.path.join(args.model_path, "config.json")
    raw = json.load(open(cfg_path))
    rope_theta = raw.get("rope_theta", None)
    if rope_theta is None and isinstance(raw.get("rope_scaling"), dict):
        rope_theta = raw["rope_scaling"].get("rope_theta", 500000.0)
    if rope_theta is None:
        rope_theta = 500000.0
    cfg.rope_theta = rope_theta
    print(f"[memoryllm] config.rope_theta = {rope_theta}")

    model = None
    for impl in ("sdpa",):
        try:
            model = MemoryLLM.from_pretrained(
                args.model_path,
                config=cfg,
                attn_implementation=impl,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            print(f"[memoryllm] Loaded with {impl}")
            break
        except (ValueError, ImportError, AttributeError, RuntimeError) as e:
            print(f"[memoryllm] {impl} failed: {e}")
    if model is None:
        raise RuntimeError("MemoryLLM: cannot load")

    model = model.to("cuda:0")
    model.eval()

    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    # Fix: the underlying LlamaModel inside MemoryLLM may be missing
    # `prepare_inputs_for_generation` if the override happened only on the wrapper.
    # Add a no-op shim if missing on whichever module .generate() walks.
    _patch_for_generate(model)

    initial_memory = None
    if hasattr(model, "memory") and isinstance(model.memory, torch.Tensor):
        initial_memory = model.memory.detach().clone()
        print(f"[memoryllm] initial_memory: {tuple(initial_memory.shape)} {initial_memory.dtype}")
    else:
        print("[memoryllm] WARN: no .memory attribute found; reset will be a no-op")

    chunk_size = getattr(model.config, "num_tokens", 1024)
    print(f"[memoryllm] chunk_size={chunk_size}")

    def _per_sample(_sample):
        if initial_memory is not None:
            model.memory.data.copy_(initial_memory)

    def _prepare(ctx, q, examples, instr, post, use_chat):
        # Inject the long context into memory first
        if ctx and ctx.strip():
            ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids
            ids = ids.to("cuda:0")
            total = ids.shape[1]
            if total >= 16:
                pos = 0
                # Disable use_cache during injection: KV cache must NOT accumulate
                # across chunks, otherwise past_key_values makes prefix_token_length=0
                _orig_use_cache = model.config.use_cache
                model.config.use_cache = False
                try:
                    while pos < total:
                        chk = ids[:, pos:pos + chunk_size]
                        if chk.shape[1] >= 16:
                            with torch.no_grad():
                                try:
                                    model.inject_memory(chk, update_memory=True)
                                except TypeError:
                                    model.inject_memory(chk)
                        pos += chunk_size
                finally:
                    model.config.use_cache = _orig_use_cache
        # Build the question prompt without context (it's in memory now)
        text = get_formatted_input("", q, examples, instr, post, template=DEFAULT_TEMPLATE)
        if use_chat:
            msgs = [{"role": "user", "content": text}]
            ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
            # transformers 5.x returns BatchEncoding; extract .input_ids tensor
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            # MemoryLLM trained without leading BOS — drop it if present
            if ids[0, 0].item() == tokenizer.bos_token_id and ids.shape[1] > 1:
                ids = ids[:, 1:]
        else:
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return ids.to("cuda:0")

    def _generate(input_ids):
        return _hf_generate(model, tokenizer, input_ids, args.max_new_tokens)

    _eval_loop(
        args,
        prepare_input=_prepare,
        generate=_generate,
        per_sample_setup=_per_sample,
    )


# ---------------------------------------------------------------------------
# Baseline: M+ (inject_memory)
# ---------------------------------------------------------------------------

def run_mplus(args: argparse.Namespace) -> None:
    """M+ uses inject_memory(chunk) to load context, then generate(question)."""
    if H20_MEMORYLLM_SRC not in sys.path:
        sys.path.insert(0, H20_MEMORYLLM_SRC)
    if LOCAL_MEMORYLLM_SRC not in sys.path:
        sys.path.insert(0, LOCAL_MEMORYLLM_SRC)

    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from transformers.cache_utils import DynamicCache
    try:
        from modeling_mplus import MPlus, LlamaModel as MPlusLlamaModel, LlamaAttention as MPlusLlamaAttention
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import MPlus source from {H20_MEMORYLLM_SRC}: {e}"
        )

    if not hasattr(MPlusLlamaModel, "prepare_inputs_for_generation"):
        def _mplus_prepare_inputs_for_generation(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            **kwargs,
        ):
            return {
                "input_ids": input_ids if input_ids is not None else kwargs.get("input_ids"),
                "past_key_values": (
                    past_key_values if past_key_values is not None else kwargs.get("past_key_values")
                ),
                "attention_mask": (
                    attention_mask if attention_mask is not None else kwargs.get("attention_mask")
                ),
            }

        MPlusLlamaModel.prepare_inputs_for_generation = _mplus_prepare_inputs_for_generation

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values):
            if past_key_values is None:
                return cls()
            return cls(ddp_cache_data=past_key_values)

        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not getattr(MPlusLlamaAttention.forward, "_mplus_tuple5_patch", False):
        _orig_mplus_attention_forward = MPlusLlamaAttention.forward

        def _mplus_attention_forward_tuple5(*f_args, **f_kwargs):
            out = _orig_mplus_attention_forward(*f_args, **f_kwargs)
            if isinstance(out, tuple) and len(out) == 4:
                return out + (None,)
            return out

        _mplus_attention_forward_tuple5._mplus_tuple5_patch = True
        MPlusLlamaAttention.forward = _mplus_attention_forward_tuple5

    if not getattr(MPlus.prepare_inputs_for_generation, "_mplus_cache_position_patch", False):
        _orig_mplus_prepare_inputs = MPlus.prepare_inputs_for_generation

        def _mplus_prepare_inputs_with_cache_position(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            **kwargs,
        ):
            if cache_position is None:
                source = input_ids if input_ids is not None else inputs_embeds
                if source is not None:
                    past_seen = 0
                    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                        past_seen = past_key_values.get_seq_length()
                    elif past_key_values:
                        past_seen = past_key_values[0][0].shape[-2]
                    cache_position = torch.arange(
                        past_seen,
                        past_seen + source.shape[1],
                        device=source.device,
                    )
            return _orig_mplus_prepare_inputs(
                self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

        _mplus_prepare_inputs_with_cache_position._mplus_cache_position_patch = True
        MPlus.prepare_inputs_for_generation = _mplus_prepare_inputs_with_cache_position

    if not hasattr(DynamicCache, "to_legacy_cache"):
        def _to_legacy_cache(self):
            legacy = []
            for layer in self.layers:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None:
                    legacy.append((None, None))
                else:
                    legacy.append((keys, values))
            return tuple(legacy)

        DynamicCache.to_legacy_cache = _to_legacy_cache

    print(f"[mplus] Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Patch rope_theta if missing (transformers 4.46 may strip it from attached config)
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg_path = os.path.join(args.model_path, "config.json")
    raw = json.load(open(cfg_path))
    rope_theta = raw.get("rope_theta", None)
    if rope_theta is None and isinstance(raw.get("rope_scaling"), dict):
        rope_theta = raw["rope_scaling"].get("rope_theta", 500000.0)
    if rope_theta is None:
        rope_theta = 500000.0
    cfg.rope_theta = rope_theta
    print(f"[mplus] config.rope_theta = {rope_theta}")

    model = None
    for impl in ("flash_attention_2", "eager", "sdpa"):
        try:
            model = MPlus.from_pretrained(
                args.model_path,
                config=cfg,
                attn_implementation=impl,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            print(f"[mplus] Loaded with {impl}")
            break
        except (ValueError, ImportError, AttributeError, RuntimeError) as e:
            print(f"[mplus] {impl} failed: {e}")
    if model is None:
        raise RuntimeError("MPlus: cannot load")

    model = model.to("cuda:0")
    model.eval()
    if hasattr(model, "put_ltm_to_numpy"):
        model.put_ltm_to_numpy()
        print("[mplus] Moved LTM state to CPU/numpy for inference")

    # ------------------------------------------------------------------
    # Re-initialize RoPE inv_freq buffers.
    #
    # Under transformers >= 5 + accelerate, `from_pretrained` constructs the
    # model under `init_empty_weights` (meta device).  `LlamaRotaryEmbedding`
    # registers `inv_freq` with `persistent=False`, so its values are not
    # stored in the checkpoint and the buffer stays as a meta tensor through
    # weight-loading.  When `.to("cuda:0")` later allocates real storage, it
    # contains uninitialised garbage (we observed values like 1.25e-38 / 0.0
    # instead of the expected 1.0, 0.81, ...).  Garbage inv_freq → garbage
    # cos/sin → NaN attention scores → logits collapse to argmax==token 0
    # ("!" with this tokenizer), which is exactly the degenerate "!!!!!!"
    # output seen on every BABILong sample (0% accuracy).
    #
    # Fix: walk every rotary_emb module and re-run its `rope_init_fn` to
    # repopulate inv_freq with the proper Llama3-scaled frequencies.
    _rope_fixed = 0
    for _mod in model.modules():
        if hasattr(_mod, "rope_init_fn") and hasattr(_mod, "inv_freq") and hasattr(_mod, "config"):
            try:
                _new_inv, _scaling = _mod.rope_init_fn(_mod.config, device=_mod.inv_freq.device)
                _mod.register_buffer("inv_freq", _new_inv.to(_mod.inv_freq.device), persistent=False)
                _mod.original_inv_freq = _mod.inv_freq
                _mod.attention_scaling = _scaling
                _rope_fixed += 1
            except Exception as _e:  # pragma: no cover - defensive
                print(f"[mplus] WARN: rope re-init on {type(_mod).__name__} failed: {_e}")
                break
    if _rope_fixed:
        print(f"[mplus] Re-initialized inv_freq on {_rope_fixed} rotary_emb modules")

    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    _patch_for_generate(model)

    initial_memory = None
    if hasattr(model, "memory") and isinstance(model.memory, torch.Tensor):
        initial_memory = model.memory.detach().clone()
        print(f"[mplus] initial_memory: {tuple(initial_memory.shape)} {initial_memory.dtype}")
    else:
        print("[mplus] WARN: no .memory attribute found; reset will be a no-op")

    chunk_size = getattr(model.config, "num_tokens", 1024)
    print(f"[mplus] chunk_size={chunk_size}")

    def _per_sample(_sample):
        if initial_memory is not None:
            with torch.no_grad():
                model.memory.copy_(initial_memory)

    def _prepare(ctx, q, examples, instr, post, use_chat):
        # Inject the long context into memory first
        if ctx and ctx.strip():
            ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids
            ids = ids.to("cuda:0")
            total = ids.shape[1]
            if total >= 16:
                pos = 0
                while pos < total:
                    chk = ids[:, pos:pos + chunk_size]
                    if chk.shape[1] >= 16:
                        with torch.no_grad():
                            try:
                                model.inject_memory(chk, update_memory=True)
                            except TypeError:
                                model.inject_memory(chk)
                    pos += chunk_size
        # MPlus-8B is a pretrained model; use the prompt format from its README.
        text = f"Question: {q.strip()} Answer:"
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return ids.to("cuda:0")

    def _generate(input_ids):
        eos_ids = [tokenizer.eos_token_id]
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos_ids.append(eot)
        attention_mask = torch.ones(
            input_ids.shape[0],
            model.num_tokens * (model.num_blocks - 1) + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                use_cache=False,
            )
        new_ids = out[0][input_ids.shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    _eval_loop(
        args,
        prepare_input=_prepare,
        generate=_generate,
        per_sample_setup=_per_sample,
    )


def _prepare_inputs_for_generation_stub(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    **kwargs,
):
    model_inputs = {"input_ids": input_ids}
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask
    model_inputs.update(kwargs)
    return model_inputs



def _patch_generate_class(target_cls: type | None) -> None:
    if target_cls is None:
        return
    if not hasattr(target_cls, "prepare_inputs_for_generation"):
        target_cls.prepare_inputs_for_generation = _prepare_inputs_for_generation_stub
    for attr_name, attr_value in GenerationMixin.__dict__.items():
        if attr_name.startswith("__"):
            continue
        if not callable(getattr(GenerationMixin, attr_name, None)):
            continue
        if not hasattr(target_cls, attr_name) or attr_name == "_expand_inputs_for_generation":
            setattr(target_cls, attr_name, attr_value)
    if not hasattr(target_cls, "_supports_cache_class"):
        target_cls._supports_cache_class = False



def _patch_generate_module_classes(module: ModuleType, *class_names: str) -> None:
    for class_name in class_names:
        _patch_generate_class(getattr(module, class_name, None))



def _patch_for_generate(model: Any) -> None:
    """Patch wrapper and common nested decoder classes for generate() compatibility."""
    if model is None:
        return

    candidates = []
    if isinstance(model, type):
        candidates.append(model)
    else:
        candidates.append(model.__class__)
        for attr in ("model", "base_model", "transformer", "decoder"):
            nested = getattr(model, attr, None)
            if nested is not None:
                candidates.append(nested if isinstance(nested, type) else nested.__class__)

    seen = set()
    for target_cls in candidates:
        if target_cls is None or id(target_cls) in seen:
            continue
        seen.add(id(target_cls))
        _patch_generate_class(target_cls)


# ---------------------------------------------------------------------------
# Baseline: Activation Beacon (Qwen2-7B-Instruct)
# ---------------------------------------------------------------------------

def run_beacon(args: argparse.Namespace) -> None:
    """Beacon model uses trust_remote_code; reset memory between samples."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[beacon] Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = None
    for impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=impl,
                device_map="cuda:0",
            )
            print(f"[beacon] Loaded with {impl}")
            break
        except Exception as e:
            print(f"[beacon] {impl} failed: {e}")

    if model is None:
        # Last resort: no explicit attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            print("[beacon] Loaded with default attn")
        except Exception as e:
            raise RuntimeError(f"Beacon: cannot load — {e}")
    model.eval()

    has_memory_reset = hasattr(model, "memory") and hasattr(model.memory, "reset")
    print(f"[beacon] has memory.reset = {has_memory_reset}")

    def _per_sample(_sample):
        if has_memory_reset:
            model.memory.reset()

    def _prepare(ctx, q, examples, instr, post, use_chat):
        # Beacon needs attention_mask alongside input_ids; return full BatchEncoding dict
        text = get_formatted_input(ctx, q, examples, instr, post, template=DEFAULT_TEMPLATE)
        if use_chat:
            msgs = [{"role": "user", "content": text}]
            # return_dict=True gives BatchEncoding with both input_ids and attention_mask
            inputs = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
        else:
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        return {k: v.to("cuda:0") for k, v in inputs.items()}

    def _generate(inputs):
        # inputs is a dict with input_ids (and attention_mask for chat mode)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else inputs
        eos_ids = [tokenizer.eos_token_id]
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos_ids.append(eot)
        with torch.no_grad():
            generate_kwargs = dict(inputs) if isinstance(inputs, dict) else {"input_ids": inputs}
            out = model.generate(
                **generate_kwargs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                top_p=1,
                temperature=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )
        result = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if has_memory_reset:
            try:
                model.memory.reset()
            except Exception:
                pass
        return result

    _eval_loop(
        args,
        prepare_input=_prepare,
        generate=_generate,
        per_sample_setup=_per_sample,
    )


# ---------------------------------------------------------------------------
# Common eval loop
# ---------------------------------------------------------------------------

def _eval_loop(args, prepare_input, generate, per_sample_setup):
    """Iterate over (task, length) -> samples; write a CSV per cell."""
    use_chat = args.use_chat_template
    suffix_parts = [
        "instruction_yes" if args.use_instruction else "instruction_no",
        "examples_yes" if args.use_examples else "examples_no",
        "post_prompt_yes" if args.use_post_prompt else "post_prompt_no",
        "chat_template_yes" if use_chat else "chat_template_no",
        "system_prompt_no",
    ]
    suffix = "_".join(suffix_parts)

    out_dir = Path(args.results_folder) / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tqdm(args.tasks, desc="tasks"):
        instr = DEFAULT_PROMPTS[task].get("instruction", "") if args.use_instruction else ""
        examples = DEFAULT_PROMPTS[task].get("examples", "") if args.use_examples else ""
        post = DEFAULT_PROMPTS[task].get("post_prompt", "") if args.use_post_prompt else ""

        for length in tqdm(args.lengths, desc=f"{task}", leave=False):
            outfile = out_dir / f"{task}_{length}_{suffix}.csv"
            if outfile.exists() and not args.overwrite:
                # resume: count existing rows
                try:
                    existing = pd.read_csv(outfile)
                    if len(existing) >= (args.limit or 100):
                        print(f"[eval_loop] {outfile.name} already done ({len(existing)} rows) — skip")
                        continue
                except Exception:
                    pass

            t0 = time.time()
            data = datasets.load_dataset(args.dataset_name, length, split=task)
            samples = list(data)
            if args.limit:
                samples = samples[: args.limit]

            rows = []
            for sample in tqdm(samples, desc=f"{task}/{length}", leave=False):
                if per_sample_setup is not None:
                    per_sample_setup(sample)
                target = sample["target"]
                ctx = sample["input"]
                q = sample["question"]
                try:
                    input_ids = prepare_input(ctx, q, examples, instr, post, use_chat)
                    output = generate(input_ids)
                except Exception as e:
                    output = f"<<ERROR: {type(e).__name__}: {str(e)[:200]}>>"
                    traceback.print_exc()
                    print(f"[eval_loop] sample failed: {output}")
                rows.append({
                    "target": target,
                    "output": output,
                    "question": q,
                    "correct": int(target.strip().lower() in output.strip().lower()) if "<<ERROR" not in output else "",
                })

                # Periodic checkpoint write
                if len(rows) % 25 == 0:
                    pd.DataFrame(rows).to_csv(outfile, index=False)

            pd.DataFrame(rows).to_csv(outfile, index=False)
            dt = time.time() - t0
            print(f"[eval_loop] saved {len(rows)} -> {outfile} ({dt:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["plain_hf", "memoryllm", "mplus", "beacon"], required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_name", type=str, required=True)
    p.add_argument(
        "--results_folder",
        type=str,
        default="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/babilong_results",
    )
    p.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    p.add_argument(
        "--tasks", type=str, nargs="+",
        default=["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"],
    )
    p.add_argument(
        "--lengths", type=str, nargs="+",
        default=["1k", "2k", "4k", "8k", "16k", "32k"],
    )
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--use_instruction", action="store_true")
    p.add_argument("--use_examples", action="store_true")
    p.add_argument("--use_post_prompt", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    _set_proxy()

    print("=" * 60)
    print(f"BABILong eval baseline={args.baseline}")
    print(f"  model_path  = {args.model_path}")
    print(f"  output_name = {args.output_name}")
    print(f"  tasks       = {args.tasks}")
    print(f"  lengths     = {args.lengths}")
    print(f"  results_dir = {args.results_folder}/{args.output_name}")
    print(f"  limit       = {args.limit}")
    print("=" * 60)

    if args.baseline == "plain_hf":
        run_plain_hf(args)
    elif args.baseline == "memoryllm":
        run_memoryllm(args)
    elif args.baseline == "mplus":
        run_mplus(args)
    elif args.baseline == "beacon":
        run_beacon(args)
    else:
        raise ValueError(args.baseline)

    print(f"[main] DONE: {args.output_name}")


if __name__ == "__main__":
    main()
