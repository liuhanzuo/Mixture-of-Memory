"""BABILong evaluation wrapper for the mem_space streaming memory architecture.

Evaluates a Llama-3-8B model patched with MemorySpaceLayer on BABILong tasks.
mem_space is stateful: stream chunks into the memory bank, reset per sample, and
generate from the last chunk with use_cache=False.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add babilong to path — relative to repo root
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import datasets  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM  # noqa: E402

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model, _reset_fifo_memory  # noqa: E402


# --------------------------------------------------------------------------- #
# BABILong dataset loading
# --------------------------------------------------------------------------- #


def _candidate_babilong_cache_dirs(user_cache_dir: str | None) -> list[Path]:
    roots = []
    if user_cache_dir:
        roots.append(Path(user_cache_dir).expanduser())
    for env in ("HF_DATASETS_CACHE", "HF_HOME"):
        if os.environ.get(env):
            root = Path(os.environ[env]).expanduser()
            roots.append(root if env == "HF_DATASETS_CACHE" else root / "datasets")
    roots += [Path(PROJECT_ROOT) / ".cache/huggingface/datasets", Path.home() / ".cache/huggingface/datasets"]
    seen, out = set(), []
    for root in roots:
        key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _load_babilong_from_arrow_cache(dataset_name: str, split_name: str, cache_dir: Path):
    root = cache_dir / dataset_name.replace("/", "___") / split_name
    arrow_roots = [p for p in root.glob("*/*") if p.is_dir() and any(p.glob("babilong-*.arrow"))]
    if not arrow_roots:
        return None
    arrow_root = max(arrow_roots, key=lambda p: p.stat().st_mtime)
    data = {
        p.stem.removeprefix("babilong-"): datasets.Dataset.from_file(str(p))
        for p in sorted(arrow_root.glob("babilong-*.arrow"))
    }
    if data:
        print(f"[mem_space-BABILong] Loaded {dataset_name}/{split_name} from local Arrow cache: {arrow_root}")
    return data or None


def load_babilong_dataset(dataset_name: str, split_name: str, cache_dir: str | None = None):
    last_error = None
    for candidate in _candidate_babilong_cache_dirs(cache_dir):
        try:
            data = datasets.load_dataset(dataset_name, split_name, cache_dir=str(candidate), download_mode="reuse_dataset_if_exists")
            print(f"[mem_space-BABILong] Loaded {dataset_name}/{split_name} with cache_dir={candidate}")
            return data
        except Exception as e:
            last_error = e
            data = _load_babilong_from_arrow_cache(dataset_name, split_name, candidate)
            if data is not None:
                return data
    try:
        return datasets.load_dataset(dataset_name, split_name, download_mode="reuse_dataset_if_exists")
    except Exception:
        raise last_error


# --------------------------------------------------------------------------- #
# Memory helpers (copied verbatim from eval_niah_mem_space.py:82-101)
# --------------------------------------------------------------------------- #


def _reset_banks(model: torch.nn.Module) -> None:
    """Wipe per-sample slot and summary state between BABILong samples."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
    else:
        mem_layers = getattr(root, "_mem_space_layers", None)
        if mem_layers:
            for w in mem_layers:
                w.memory_bank.reset()
    # Reset L3 summary state (cold start for new sample)
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._current_summary = None
        l3_pool._prev_chunk_h = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None
        # Batched-eval padding (2026-06-09): clear the previous-chunk token mask
        # consumed by the L3 pool. No-op for bsz=1 (mask never set).
        if hasattr(l3_pool, "_prev_chunk_token_mask"):
            l3_pool._prev_chunk_token_mask = None
        if hasattr(l3_pool, "_prev_summary"):
            l3_pool._prev_summary = None
    # Batched-eval padding: clear the per-chunk token mask stashed on the bank.
    if shared_bank is not None and hasattr(shared_bank, "_active_token_mask"):
        shared_bank._active_token_mask = None
    else:
        for w in getattr(root, "_mem_space_layers", []) or []:
            _b = getattr(w, "memory_bank", None)
            if _b is not None and hasattr(_b, "_active_token_mask"):
                _b._active_token_mask = None


def _reset_l2(model: torch.nn.Module) -> None:
    """Zero the L2 compressor's cross-chunk state (prev_latents).

    Called at every document boundary alongside ``_reset_banks``. No-op if the
    model was patched without ``use_l2``.
    """
    root = getattr(model, "module", model)
    comp = getattr(root, "_l2_compressor", None)
    if comp is not None:
        comp.reset()


def _freeze_banks(model: torch.nn.Module) -> None:
    """Freeze memory banks during greedy generation so writeback doesn't
    overwrite slots accumulated from the context."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = True
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = True


def _unfreeze_banks(model: torch.nn.Module) -> None:
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = False
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = False


# --------------------------------------------------------------------------- #
# Adapter config → MemorySpaceConfig translation
# --------------------------------------------------------------------------- #


# Map abbreviated field names found in adapter_config.json → MemorySpaceConfig
# fields. Anything not in this map and not a MemorySpaceConfig field is ignored.
_ADAPTER_CONFIG_FIELD_MAP = {
    "writeback_warmup_steps": "writeback_gate_warmup_steps",
}


def build_mem_space_config(adapter_cfg: dict) -> MemorySpaceConfig:
    """Construct a MemorySpaceConfig from an adapter_config.json dict.

    Handles two pieces of impedance mismatch:
      * `writeback_warmup_steps` (json) → `writeback_gate_warmup_steps` (dataclass)
      * `unfreeze_hidden_to_slot=True` (json) → `hidden_to_slot_frozen=False` (dataclass)

    Unknown keys (e.g. `max_train_steps`, `lr`) are silently dropped.
    """
    valid_fields = set(MemorySpaceConfig.__dataclass_fields__.keys())
    kwargs: dict = {}
    for k, v in adapter_cfg.items():
        # Rename if needed
        target = _ADAPTER_CONFIG_FIELD_MAP.get(k, k)
        if target == "unfreeze_hidden_to_slot":
            # Flip semantics: unfreeze=True means hidden_to_slot_frozen=False.
            kwargs["hidden_to_slot_frozen"] = not bool(v)
            continue
        if target in valid_fields:
            kwargs[target] = v
        # else: silently ignore (training-only keys like lr, max_train_steps)
    cfg = MemorySpaceConfig(**kwargs)
    # Dynamic (non-dataclass) config attrs that patch.py reads via getattr — must
    # be set explicitly so eval reconstructs the same modules as training.
    #   HNST v2 trainable tree-summary pool (2026-06-25): patch.py creates the
    #   TreeSummaryPool only when cfg.use_tree_summary is truthy.
    for _dyn, _default in (
        ("use_tree_summary", False),
        ("tree_summary_heads", 8),
        ("tree_summary_layers", 1),
        ("tree_summary_ffn_mult", 2),
        ("fifo_tree_readout", False),
        ("fifo_tree_readout_branch", 8),
        ("fifo_tree_readout_fine_chunks", 0),
    ):
        if _dyn in adapter_cfg:
            setattr(cfg, _dyn, adapter_cfg[_dyn])
        else:
            setattr(cfg, _dyn, _default)
    return cfg


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def load_mem_space_model(
    model_path: str,
    checkpoint_path: str,
    mem_config: MemorySpaceConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "sdpa",
):
    """Build base Llama + mem_space patch + load adapter ckpt."""
    print(f"[mem_space-BABILong] Loading base model from: {model_path}")
    # Backbone-agnostic load (2026-07-05): AutoModelForCausalLM picks the right
    # class from the checkpoint's config.model_type (LlamaForCausalLM for
    # model_type=='llama', Qwen3ForCausalLM for 'qwen3', etc.). The mem_space
    # FIFO flat readout path wraps whole DecoderLayers and is backbone-agnostic,
    # so the same patch works on any Llama-style decoder stack (Qwen3 keeps its
    # QK-norm because we call the wrapped layer's forward unchanged).
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)

    # H7 fix v2 pre-step: snapshot rotary inv_freq in fp32 BEFORE any
    # `.to(dtype=bf16)` corrupts them. See eval_niah_mem_space.py:502-525.
    _rope_snapshot: dict = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass

    # Patch all decoder layers with MemorySpaceLayer
    print(f"[mem_space-BABILong] Applying mem_space patch (num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}, shared_bank={mem_config.shared_memory_bank})")
    apply_mem_space_to_model(model, mem_config, layer_indices=None)

    # Move freshly-created mem_space modules to device/dtype
    model.to(device=device, dtype=dtype)

    # H7 fix v2 post-step: restore rotary buffers to fp32 on the right device.
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
        if _rope_snapshot:
            print(f"[mem_space-BABILong] H7 fix v2 applied: restored rotary buffers "
                  f"{sorted(_rope_snapshot.keys())} to float32")
    except AttributeError:
        print("[mem_space-BABILong] WARNING: rotary_emb not accessible — skipping H7 fix")

    # Load checkpoint
    # Zero-training sentinel (2026-07-05): "none"/"base"/"" → skip adapter load,
    # keep base backbone + freshly-initialised adapter params (inject_gate etc.).
    # Used to probe whether the FIFO hidden readout works WITHOUT any mem_space
    # training on a given backbone (e.g. Qwen3-8B vs Llama-3-8B).
    _zero_train = (checkpoint_path is None) or (
        str(checkpoint_path).strip().lower() in ("", "none", "base", "zero")
    )
    if _zero_train:
        print("[mem_space-BABILong] ZERO-TRAINING mode: skipping adapter "
              "checkpoint load (base backbone + fresh adapter init).")
    else:
        print(f"[mem_space-BABILong] Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Common state-dict layouts: raw OrderedDict / {model_state_dict: ...} / {state_dict: ...}.
    if not _zero_train:
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                # Assume the dict itself is the state_dict (this is what
                # eval_niah_mem_space.py:552 expects for the champion ckpt).
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip DDP "module." prefix if present.
        cleaned: dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[mem_space-BABILong] Loaded {len(cleaned)} keys | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print(f"[mem_space-BABILong] WARNING: first 5 unexpected keys: {list(unexpected)[:5]}")
        # Adapter-specific missing keys are real failures; base-model missing keys
        # are expected with strict=False (the base weights came from from_pretrained).
        adapter_missing = [
            k for k in missing
            if any(s in k for s in (
                "slot_output_gate", "gate_param", "Q_sel", "K_sel",
                "slot_to_hidden", "hidden_to_slot",
            ))
        ]
        if adapter_missing:
            print(f"[mem_space-BABILong] WARNING: {len(adapter_missing)} adapter keys NOT "
                  f"loaded — first 5: {adapter_missing[:5]}")

    # Fix J: force step_counter = warmup_steps so β/warmup_frac is fully ramped.
    from src.memory.mem_space.layer import MemorySpaceLayer as _MSL  # local import to avoid cycles
    _mem_layers = getattr(model, "_mem_space_layers", [])
    _warmup_target = mem_config.writeback_gate_warmup_steps if mem_config.writeback_gate_warmup_steps > 0 else 1
    for _w in _mem_layers:
        if isinstance(_w, _MSL):
            _w.step_counter = _warmup_target
    print(f"[mem_space-BABILong] Fix J: set step_counter={_warmup_target} on "
          f"{len(_mem_layers)} MemorySpaceLayer(s)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------- #
# Chunked generation
# --------------------------------------------------------------------------- #


def _set_memory_disabled(model, flag: bool) -> None:
    """Toggle MemorySpaceLayer._memory_disabled on every wrapped layer.

    When True, MemorySpaceLayer.forward() short-circuits to forward_no_memory
    (vanilla Llama path), bypassing all memory bank writes and reads. Used by
    --memory_disabled CLI flag as a falsification test: if BABILong scores stay
    high with memory disabled, the high scores are an artifact (in-context leak
    / few-shot prior) rather than evidence the memory bank is working.
    """
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._memory_disabled = flag


def _set_fifo_pos_mode(model, mode):
    """Set per-layer FIFO position-fix mode (None | 'packed' | 'real').

    None reverts to legacy all-pos-0 prefix; 'packed' uses kept-index based
    in-distribution positions; 'real' uses original chunk indices (may be OOD
    but exercises Llama-3 RoPE extrapolation).
    """
    root = getattr(model, "module", model)
    # Stash the outer model root on each layer so they can resolve rotary_emb
    # without a global handle (best-effort; falls back to wrapped_layer.self_attn).
    # List-wrapped (`[root]`) so nn.Module does NOT register the outer model as a
    # child submodule of the layer (a bare nn.Module attr would create a
    # model<->layer cycle that makes model.train() recurse infinitely).
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._fifo_pos_mode = mode
        w._fifo_rotary_root_ref = [root]


def _set_fifo_keep_set_mode(model, mode, topk=25, recency=2, tree_branch=8, tree_beam=2):
    """Set per-layer FIFO keep-set mode (None | 'flat_readerattn' | 'tree').

    None keeps ALL buffered chunks (legacy); 'flat_readerattn' uses reader q.k
    to score chunks and keeps the top-`topk` plus the last `recency` chunks;
    'tree' (HNST) runs a recursive beam descent over a B-ary max-pool tree on
    the buffer (branch=`tree_branch`, beam=`tree_beam`), surfacing top-`topk`
    leaves selected from ALL buffered chunks + last `recency`.
    """
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._fifo_keep_set_mode = mode
        w._fifo_keep_topk = int(topk)
        w._fifo_keep_recency = int(recency)
        w._fifo_tree_branch = int(tree_branch)
        w._fifo_tree_beam = int(tree_beam)


def _set_fifo_keep_all_buffer(model, flag: bool) -> None:
    """When True, suppress FIFO eviction so the buffer holds ALL past chunks
    (use only with a keep_set_mode → "keep-all-store, attend-few" probe)."""
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._fifo_keep_all_buffer = bool(flag)


def _set_fifo_buffer_chunks(model, n: int) -> None:
    """Eval-time override of the FIFO eviction buffer size (``fifo_buffer_chunks``).

    Lets a single clean checkpoint (trained with e.g. buffer=50) be evaluated as a
    SMALLER-buffer FIFO (e.g. 25) so early chunks are structurally EVICTED at long
    lengths -> the "amnesia" baseline for the HNST position-stratified test. No-op
    when n<=0 (keep the trained value). Ignored under --fifo_keep_all_buffer (that
    flag suppresses eviction entirely)."""
    if n is None or n <= 0:
        return
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._fifo_buffer_chunks = int(n)


def _find_subsequence_ids(haystack_ids, needle_ids):
    """Locate ``needle_ids`` as a contiguous subsequence of ``haystack_ids``
    (both 1-D list[int]). Returns the START token index of the match or None.

    Robust to whitespace-merge tokenisation at the answer boundary: try the full
    needle, then progressively trim leading tokens (space-prefix artifacts), then
    fall back to the longest trailing run that matches. Mirrors the RULER oracle's
    ``_find_subsequence`` (scripts/eval_ruler_mem_space.py:160) but returns only
    the start offset (the FIFO oracle needs the token position, not the span)."""
    H, N = haystack_ids, needle_ids
    nH, nN = len(H), len(N)
    if nN == 0 or nH == 0:
        return None

    def _scan(sub):
        ns = len(sub)
        if ns == 0 or ns > nH:
            return None
        # Search from the END so we find the LAST (most recent) occurrence — for
        # bAbI the answer-bearing supporting fact is the latest mention.
        for s in range(nH - ns, -1, -1):
            if H[s:s + ns] == sub:
                return s
        return None

    r = _scan(N)
    if r is not None:
        return r
    for drop in range(1, min(4, nN)):
        r = _scan(N[drop:])
        if r is not None:
            return r
    for keep in range(nN - 1, 0, -1):
        r = _scan(N[-keep:])
        if r is not None:
            return r
    return None


def _locate_needle_chunks(input_ids, target, tokenizer, chunk_size):
    """Return the set of 0-based DOCUMENT-ABSOLUTE chunk indices that contain the
    gold answer (``target``) in ``input_ids`` (a [1, L] LongTensor), or None if
    the answer cannot be located.

    chunk index = token_pos // chunk_size  (the same split used by
    generate_with_mem_space: ``tokens.split(chunk_size)``). The answer string can
    tokenise differently in isolation vs in-context (space-prefix merges), so we
    try a few encodings and take the union of any matches. Multi-token answers
    that straddle a chunk boundary contribute BOTH chunks (token span overlap).
    Returns a set so multi-chunk / multi-mention needles are all kept."""
    ids = input_ids[0].tolist()
    L = len(ids)
    tgt = (target or "").strip()
    if not tgt:
        return None
    cands = []
    for variant in (tgt, " " + tgt):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc:
            cands.append(enc)
    chunks = set()
    for needle_ids in cands:
        start = _find_subsequence_ids(ids, needle_ids)
        if start is None:
            continue
        end = min(L - 1, start + len(needle_ids) - 1)  # last token of the span
        for p in range(start, end + 1):
            chunks.add(p // chunk_size)
    return chunks or None


def _set_fifo_oracle_needle(model, needle_chunks) -> None:
    """Stash the per-sample needle absolute-chunk set on every FIFO layer AND
    reset the per-document FIFO buffer bookkeeping so document-absolute chunk
    indices restart at 0 for this sample.

    NOTE: ``_reset_banks`` (the slot/summary reset) does NOT touch ``_fifo_buf``;
    the FIFO oracle path needs a clean buffer + a 0-based absolute counter per
    sample for ``needle_token_pos // chunk_size`` to be meaningful, so this
    function also clears the buffer state. ``needle_chunks=None`` clears the
    oracle channel (layer falls back to keep-all)."""
    root = getattr(model, "module", model)
    nset = set(int(c) for c in needle_chunks) if needle_chunks else None
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._fifo_oracle_needle_chunks = nset
        # Fresh per-document FIFO state (the oracle indexes by absolute chunk idx).
        w._fifo_buf = []
        w._fifo_buf_abs_idx = []
        w._fifo_write_seq = 0


def _fifo_oracle_fallback_total(model):
    """Sum the per-layer oracle keep-all fallback counters → (fallback, evicted).
    fallback = needle unknown OR evicted; evicted = subset where needle was in the
    buffer's history but had been evicted. Counters are cumulative across the run."""
    root = getattr(model, "module", model)
    fb = ev = 0
    for w in getattr(root, "_mem_space_layers", []) or []:
        fb += int(getattr(w, "_fifo_oracle_fallback_count", 0))
        ev += int(getattr(w, "_fifo_oracle_evicted_count", 0))
    return fb, ev


@torch.no_grad()
def _select_chunks_reader_attn(
    model,
    last_chunk: torch.Tensor,
    n_chunks: int,
    device: torch.device,
    select_layer: int = 16,
    topk: int = 4,
):
    """DEPLOYABLE reader-attn chunk selection (2026-06-27, for --swa_readerattn_token).

    Returns a set of DOCUMENT-ABSOLUTE chunk indices (0-based, the same indexing
    ``generate_with_mem_space`` uses for ``tokens.split(chunk_size)``) chosen by
    the reader's OWN native q.k salience at ONE model-level selection layer — NO
    oracle, NO gold answer, NO trained selector. Returns ``None`` on any failure
    (caller then falls back to the plain last-chunk window).

    Must be called AFTER the streaming-ingestion loop has run (so every FIFO
    layer's ``_fifo_buf`` holds the per-chunk hidden snapshots of chunks
    ``0..n_chunks-2``) and AFTER ``_freeze_banks`` (banks frozen for generation).

    Mechanism
    ---------
      query   = the LAST (question) chunk's hidden at the input to ``select_layer``
                (== output of layer ``select_layer-1``), grabbed with one extra
                forward of the last chunk under ``output_hidden_states=True``.
                This matches the FIFO write convention: ``_fifo_buf`` stores each
                chunk's LAYER-INPUT hidden (``hidden_states`` arg of
                ``_forward_fifo``, layer.py:1554), i.e. the output of the previous
                layer. ``out.hidden_states[L]`` is exactly the input to layer L
                (hidden_states[0] = embeddings = input to layer 0).
      keys    = that same selection layer wrapper's ``_fifo_buf`` (one entry per
                streamed context chunk).
      scorer  = the layer's existing ``_fifo_select_keep_set_reader_attn``
                (layer.py:1612): q_proj/k_proj + RoPE + per-chunk amax salience,
                top-k. We pass ``recency=0`` (no recency floor — the last/question
                chunk is always appended by the token-window builder downstream).

    Buffer -> document mapping
    --------------------------
    The selector returns BUFFER-LOCAL indices into ``_fifo_buf``. In the deployable
    (non-oracle) FIFO path the abs-idx bookkeeping (``_fifo_buf_abs_idx``) is NOT
    maintained, so we map manually. After streaming chunks ``0..n_chunks-2``
    (= ``n_chunks-1`` writes) with FIFO eviction at ``fifo_buffer_chunks`` (eval
    default 25), the buffer holds only the MOST RECENT ``len(buf)`` of those
    chunks, contiguously. Hence:

        document_chunk_index(buffer_local_i) = ingested - len(buf) + i

    where ``ingested = n_chunks - 1`` (number of context chunks streamed; the last
    chunk is the question and is never streamed). This is exact for both the
    no-eviction case (len(buf) == ingested → offset 0) and the evicted case
    (len(buf) == fifo_buffer_chunks < ingested → offset > 0, drops the oldest).

    Side-effect safety: the extra last-chunk forward FIFO-WRITES the last chunk
    into every ``_fifo_buf`` (the write at layer.py:1554 is not frozen-gated). That
    would corrupt the readout window's prefix (the readout forward reads
    ``_fifo_buf`` too), so the CALLER snapshots+restores ``_fifo_buf`` around the
    whole selection (see generate_with_mem_space). This helper itself only READS
    the buffers; it does not mutate the model.
    """
    try:
        root = getattr(model, "module", model)
        mem_layers = getattr(root, "_mem_space_layers", None)
        if not mem_layers:
            return None
        L = int(select_layer)
        if L < 0 or L >= len(mem_layers):
            return None
        sel_wrapper = mem_layers[L]
        buf = getattr(sel_wrapper, "_fifo_buf", None)
        if not buf:
            return None  # nothing streamed (e.g. short doc) → fall back

        # ---- query hidden at the selection layer (input to layer L) ----
        # out.hidden_states is a tuple of length (num_layers + 1): index 0 is the
        # embedding output (= input to layer 0), index L is the input to layer L.
        cur = last_chunk.unsqueeze(0).to(device)            # [1, last_len]
        out = model(input_ids=cur, use_cache=False, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is None or L >= len(hs):
            return None
        q_hidden = hs[L]                                    # [1, last_len, d]

        # ---- RoPE cos/sin for the last-chunk positions 0..T-1 ----
        rot = None
        inner = getattr(root, "model", None)
        if inner is not None:
            rot = getattr(inner, "rotary_emb", None)
        if rot is None:
            rot = sel_wrapper._fifo_resolve_rotary_emb()
        if rot is None:
            return None
        Tq = q_hidden.shape[1]
        pos_ids = torch.arange(Tq, device=q_hidden.device).unsqueeze(0)  # [1, Tq]
        cos, sin = rot(q_hidden, pos_ids)

        # ---- score + top-k via the layer's existing reader-attn scorer ----
        # recency=0: no recency floor (last/question chunk is appended downstream).
        kept_local = sel_wrapper._fifo_select_keep_set_reader_attn(
            hidden_states=q_hidden,
            valid_chunks=list(buf),
            position_embeddings=(cos, sin),
            topk=int(topk),
            recency=0,
        )
        if kept_local is None:
            return None

        # ---- map buffer-local indices -> document-absolute chunk indices ----
        ingested = n_chunks - 1                 # context chunks streamed (no last)
        offset = ingested - len(buf)            # >=0; oldest evicted chunks dropped
        sel_abs = set()
        for i in kept_local:
            abs_idx = offset + int(i)
            if 0 <= abs_idx < ingested:         # exclude the last/question chunk
                sel_abs.add(abs_idx)
        return sel_abs or None
    except Exception:
        return None


def _select_chunks_tree(
    model,
    last_chunk: torch.Tensor,
    n_chunks: int,
    device: torch.device,
    select_layer: int = 16,
    topk: int = 4,
    branch: int = 4,
    beam: int = 2,
):
    """DEPLOYABLE HNST v2 trainable-tree chunk selection (2026-06-25).

    Drop-in parallel to ``_select_chunks_reader_attn`` but the selection is a
    beam descent over the TRAINABLE tree (``model._tree_pool`` learned leaf/node
    aggregation) using the trained reader's grad-free q.k salience at
    ``select_layer``. Returns a set of DOCUMENT-ABSOLUTE chunk indices, or None on
    failure (caller falls back to the plain last-chunk window).

    Same buffer→document mapping and same query-hidden convention as
    ``_select_chunks_reader_attn``: query = last-chunk hidden at the input to
    layer L; keys = that layer's ``_fifo_buf``. The heavy lifting is delegated to
    the layer's ``_fifo_select_keep_set_tree`` (which uses the tree pool when the
    layer's ``_fifo_tree_pool_ref`` is set), so train/eval navigation is identical.
    The caller snapshots+restores ``_fifo_buf`` around this (the extra last-chunk
    forward FIFO-writes it).
    """
    try:
        root = getattr(model, "module", model)
        mem_layers = getattr(root, "_mem_space_layers", None)
        if not mem_layers:
            return None
        L = int(select_layer)
        if L < 0 or L >= len(mem_layers):
            return None
        sel_wrapper = mem_layers[L]
        buf = getattr(sel_wrapper, "_fifo_buf", None)
        if not buf:
            return None
        cur = last_chunk.unsqueeze(0).to(device)
        out = model(input_ids=cur, use_cache=False, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is None or L >= len(hs):
            return None
        q_hidden = hs[L]
        rot = None
        inner = getattr(root, "model", None)
        if inner is not None:
            rot = getattr(inner, "rotary_emb", None)
        if rot is None:
            rot = sel_wrapper._fifo_resolve_rotary_emb()
        if rot is None:
            return None
        Tq = q_hidden.shape[1]
        pos_ids = torch.arange(Tq, device=q_hidden.device).unsqueeze(0)
        cos, sin = rot(q_hidden, pos_ids)
        kept_local = sel_wrapper._fifo_select_keep_set_tree(
            hidden_states=q_hidden,
            valid_chunks=list(buf),
            position_embeddings=(cos, sin),
            branch=int(branch),
            beam=int(beam),
            topk=int(topk),
            recency=0,
        )
        if kept_local is None:
            return None
        ingested = n_chunks - 1
        offset = ingested - len(buf)
        sel_abs = set()
        for i in kept_local:
            abs_idx = offset + int(i)
            if 0 <= abs_idx < ingested:
                sel_abs.add(abs_idx)
        return sel_abs or None
    except Exception:
        return None


def _get_stop_ids(tokenizer):
    """Token-id set of PURE generic English stopwords + punctuation (no qa/babi-
    specific words). Used by content-only BM25 (--swa_bm25_content_only): filter
    these out of the query before BM25 so high-IDF entity words drive selection.
    Validated by selector probe: content-only BM25 recall@4 0.52->0.67-0.72."""
    global _CONTENT_STOP_IDS
    if _CONTENT_STOP_IDS is not None:
        return _CONTENT_STOP_IDS
    words = ("the a an of to in on at is was were are be been being and or "
             "did do does done . , ? ! : ; \" ' who what where when how "
             "this that these those i you he she it they we my your his her "
             "their our s t").split()
    sid = set()
    for w in words:
        for variant in (w, " " + w):
            for t in tokenizer.encode(variant, add_special_tokens=False):
                sid.add(int(t))
    _CONTENT_STOP_IDS = sid
    return sid


def _bm25_scores(docs, query_ids, k1: float = 1.5, b: float = 0.75):
    """BM25 of ``query_ids`` (list[int]) against each candidate document's token
    IDs. Corpus == the candidate pool ``docs`` (list[list[int]]), so IDF is over
    that pool. Query terms are de-duplicated. Returns a python list[float] of
    length ``len(docs)`` (doc order preserved).

    Copied verbatim (formula + k1/b defaults) from
    ``scripts/e2_multiscorer_probe.py:score_S2_bm25`` so the deployable selection
    ranks chunks by the SAME signal whose long-doc recall@4 was measured at
    0.52-0.73. Pure CPU; no model forward (BM25 never touches the FIFO buffers,
    so — unlike the reader-attn selector — no last-chunk forward / buffer
    snapshot is needed)."""
    N = len(docs)
    if N <= 0:
        return None
    df = Counter()
    doc_tf = []
    doc_len = []
    for d in docs:
        c = Counter(d)
        doc_tf.append(c)
        doc_len.append(len(d))
        for t in c:
            df[t] += 1
    avgdl = (sum(doc_len) / N) if N > 0 else 0.0
    idf = {t: math.log((N - dft + 0.5) / (dft + 0.5) + 1.0) for t, dft in df.items()}
    qterms = set(int(t) for t in query_ids)
    scores = []
    for i in range(N):
        tf = doc_tf[i]
        dl = doc_len[i]
        s = 0.0
        for t in qterms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            it = idf.get(t, 0.0)
            if avgdl > 0:
                denom = f + k1 * (1.0 - b + b * dl / avgdl)
            else:
                denom = f + k1
            s += it * (f * (k1 + 1.0)) / denom
        scores.append(s)
    return scores


@torch.no_grad()
def _select_chunks_bm25_token(
    model,
    chunks,
    n_chunks: int,
    select_layer: int = 16,
    topk: int = 4,
    query_ids=None,
    content_only: bool = False,
    tokenizer=None,
):
    """DEPLOYABLE BM25 chunk selection (for --swa_bm25_token).

    Returns a set of DOCUMENT-ABSOLUTE chunk indices (0-based, the indexing
    ``generate_with_mem_space`` uses for ``tokens.split(chunk_size)``) chosen by
    pure lexical BM25 word-overlap between the question chunk and each candidate
    context chunk's RAW tokens — NO oracle, NO gold answer, NO trained selector,
    NO model forward. Returns ``None`` on any failure (caller then falls back to
    the plain last-chunk window).

    Drop-in parallel to ``_select_chunks_reader_attn``: it MUST be called from the
    SAME point (after the streaming loop has filled ``_fifo_buf`` and after
    ``_freeze_banks``) so the candidate pool == the reader-attn candidate pool,
    and the buffer->document mapping is byte-identical.

    Candidate pool / buffer->document mapping
    -----------------------------------------
    We read ``len(buf)`` at ``select_layer`` ONLY to recover the candidate pool
    that the reader-attn path would see (post-eviction; the buffer holds the most
    recent ``len(buf)`` of the ``n_chunks-1`` streamed context chunks). The
    document-absolute index of buffer-local 0 is

        offset = (n_chunks - 1) - len(buf)

    exactly as ``_select_chunks_reader_attn`` and ``e2_multiscorer_probe.
    _pool_from_buffer``. The BM25 corpus is then ``chunks[offset .. offset+C-1]``
    (the SAME chunks the reader-attn selector ranks), so recall numbers measured
    in the e2 probe transfer 1:1 to this selection.

    Query
    -----
    query token IDs = ``query_ids`` when provided (the caller threads the BARE
    question STRING's token IDs, ``tokenizer.encode(question, add_special_tokens=
    False)``), EXACTLY matching the e2 BM25 probe (``score_S2_bm25`` query =
    ``q_ids`` of the bare question), so the long-doc recall@4 measured at 0.52-0.73
    transfers 1:1. When ``query_ids`` is None we fall back to the LAST (question)
    chunk's raw tokens (``chunks[-1]`` = instruction + few-shot + post_prompt +
    "Question: ..."); that prefix dilutes the word-overlap signal, so the threaded
    pure-question query is preferred. The candidate-pool mapping is identical for
    both query sources — only the query text differs.
    """
    try:
        root = getattr(model, "module", model)
        mem_layers = getattr(root, "_mem_space_layers", None)
        if not mem_layers:
            return None
        L = int(select_layer)
        if L < 0 or L >= len(mem_layers):
            return None
        sel_wrapper = mem_layers[L]
        buf = getattr(sel_wrapper, "_fifo_buf", None)
        if not buf:
            return None  # nothing streamed (short doc) -> fall back

        C = len(buf)
        ingested = n_chunks - 1                  # context chunks streamed (no last)
        offset = ingested - C                    # >=0; oldest evicted chunks dropped
        if offset < 0 or C <= 0:
            return None
        # Candidate docs = the SAME pool the reader-attn selector ranks, in
        # document-absolute (== buffer) order.
        docs = [chunks[offset + i].tolist() for i in range(C)]
        # Query = bare-question token IDs threaded from the caller (matches the e2
        # probe's score_S2_bm25 query). Fall back to the whole last/question chunk
        # only when no pure-question query was provided.
        if query_ids is not None:
            q = list(query_ids)
        else:
            q = chunks[-1].tolist()          # question chunk raw tokens (fallback)
        # content-only: drop generic stopwords so high-IDF entity words drive BM25
        # (validated: recall@4 0.52->0.67-0.72). Needs tokenizer for the stop set.
        if content_only and tokenizer is not None:
            _stop = _get_stop_ids(tokenizer)
            q_filt = [t for t in q if int(t) not in _stop]
            if q_filt:                       # guard: don't empty the query
                q = q_filt
        scores = _bm25_scores(docs, q)
        if scores is None or len(scores) != C:
            return None
        keep_n = max(1, min(int(topk), C))
        order = sorted(range(C), key=lambda i: scores[i], reverse=True)
        sel_abs = set()
        for i in order[:keep_n]:
            abs_idx = offset + int(i)
            if 0 <= abs_idx < ingested:          # exclude the last/question chunk
                sel_abs.add(abs_idx)
        return sel_abs or None
    except Exception:
        return None


@torch.no_grad()
def generate_with_mem_space(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    chunk_size: int,
    max_new_tokens: int,
    device: torch.device,
    swa_eval_chunks: int = 0,
    oracle_token_chunks=None,
    needle_excluded_chunks=None,
    readerattn_token: bool = False,
    readerattn_select_layer: int = 16,
    readerattn_topk: int = 4,
    tree_token: bool = False,
    tree_select_layer: int = 16,
    tree_topk: int = 4,
    tree_branch: int = 4,
    tree_beam: int = 2,
    bm25_token: bool = False,
    bm25_select_layer: int = 16,
    bm25_topk: int = 4,
    bm25_query_ids=None,
    bm25_content_only: bool = False,
) -> str:
    """Streaming generation for a single BABILong sample.

    ORACLE-TOKEN-SWA probe (2026-06-27): when ``oracle_token_chunks`` is a
    non-empty set of document-absolute chunk indices, the final-forward window is
    built from the RAW TOKENS of those (oracle-selected needle) chunks + the last
    chunk — instead of the last (W+1) contiguous chunks. This re-forwards the
    needle's ORIGINAL TOKENS (not its stored hidden), so the needle content is
    re-contextualized against the query at every layer (live, query-conditional)
    with NO dilution. Decisive test of "live token re-forward >> frozen hidden
    snapshot": the FIFO/rawkv readout tops out ~20 even with perfect hidden
    isolation; if oracle-TOKEN re-forward scores far higher, the win is on the
    token-reforward side, not raw-hidden. All-but-needle chunks still stream into
    the bank exactly as W0 (no double counting beyond the selected needle chunks).

    Strategy (mirrors stream_haystack + F2 "last-chunk replay" trick from
    eval_niah_mem_space.py:858-901):

      1. Reset memory banks (fresh state for this sample).
      2. Stream all-but-last chunks through ``model(input_ids=...)`` so the
         memory bank accumulates context (no return value needed; mem_space
         writes the bank in-place during forward).
      3. Freeze the bank, then autoregressively generate from the last chunk.
         The last chunk is consumed in the FIRST forward call (we read logits
         at its last position); subsequent steps append one token at a time.
      4. Unfreeze the bank (for cleanliness; doesn't matter for inference but
         keeps the contract).

    We do NOT do the F2 last-chunk replay: BABILong's question_suffix is already
    embedded at the END of the formatted input (after the haystack); the last
    chunk already contains the question text + the right context, so logit
    quality at its tail is what we want to read.

    eval-only cross-chunk SWA (``swa_eval_chunks`` = W, D2a, 2026-06-09):
        Default W=0 reproduces the original behaviour bit-for-bit — the
        generation window is exactly the last chunk, and only the memory bank
        carries information about the earlier chunks. When W>0, the generation
        window becomes the concatenation of the last (W+1) chunks, so the final
        forward's self-attention can attend DIRECTLY to the previous W chunks'
        raw KV (sliding window), *in addition to* the memory readback. The bank
        streaming loop is unchanged (still ``chunks[:-1]``), i.e. those W chunks
        remain in the bank too — SWA is purely additive direct attention. This
        tests whether the no-cross-chunk-SWA eval systematically under-estimates
        the model's true long-context ability. Note that the combined window
        gives those tokens correct *relative* RoPE positions (within the window)
        instead of each chunk restarting at position 0.

        Short-doc fallback (2026-06-09): SWA only activates when there is at
        least one chunk the window does NOT cover, i.e. ``len(chunks) > W+1``.
        If the whole document fits in the window (``len(chunks) <= W+1``) we
        fall back to the W0 single-chunk path. Otherwise the earlier chunks
        would be counted twice (streamed into the bank AND directly attended in
        the window) and the forward window would balloon to (W+1)*chunk_size,
        well past the training-time single-chunk window — pure OOD damage on
        short docs (0k ~97% single-chunk, 1k ~100% two-chunk) with no remote
        chunk to actually benefit.

    Args:
        input_ids: [1, total_len] tensor on `device`.
        swa_eval_chunks: W >= 0. 0 = original (no cross-chunk SWA, default).

    Returns:
        Decoded text of `max_new_tokens` generated tokens (skip_special_tokens=True).
    """
    if device is None:
        device = next(model.parameters()).device

    _reset_banks(model)
    _reset_l2(model)
    # H-2 FIX (2026-07-02 code-health audit): ALWAYS clear the FIFO hidden buffer
    # at the per-sample (document) boundary. Previously this only fired for the
    # reader-attn/bm25/tree token probes, so the DEFAULT FIFO eval path carried a
    # previous unrelated document's chunk hidden into the next sample's readout
    # prefix — silently polluting SHORT lengths (doc chunk-count < fifo_buffer_chunks,
    # i.e. 0k-4k never self-clear via eviction) in a LENGTH-CORRELATED way. Different
    # documents must NEVER share the buffer. _reset_fifo_memory is a no-op on
    # non-FIFO ckpts, so this is safe/byte-identical there.
    _reset_fifo_memory(model)

    tokens = input_ids[0]  # [total_len]
    chunks = list(tokens.split(chunk_size))

    # Stream all-but-last chunks (memory accumulation only — no logit reads).
    # NOTE: unchanged by SWA — the bank always accumulates chunks[:-1] exactly
    # as before, so W>0 only ADDS direct attention, it never removes context
    # from the bank.
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            chunk_tensor = chunk.unsqueeze(0).to(device)  # [1, <=chunk_size]
            _ = model(input_ids=chunk_tensor, use_cache=False)

    # Freeze the bank — generation should not pollute the slots that hold the context.
    _freeze_banks(model)
    # READER-ATTN-TOKEN selection (2026-06-27, DEPLOYABLE): pick the chunk set by
    # the reader's own native q.k salience at one selection layer, then feed those
    # document-absolute indices into the EXISTING oracle-token window builder
    # below (so the token-reforward readout path is byte-identical — only the
    # chunk SOURCE differs: deployable selection vs the answer-cheating locator).
    # When this yields no selection (short doc / failure) oracle_token_chunks stays
    # falsy and we fall through to the plain last-chunk window unchanged.
    if readerattn_token:
        # The selection runs ONE extra forward of the last chunk, which would
        # FIFO-write the last chunk into every layer's _fifo_buf (write at
        # layer.py:1554 is not frozen-gated) and thereby change the readout
        # window's prefix. Snapshot + restore _fifo_buf around it so the
        # downstream readout sees the exact same buffer as the oracle-token path.
        _root = getattr(model, "module", model)
        _mem_layers = getattr(_root, "_mem_space_layers", []) or []
        _buf_snapshot = [list(getattr(w, "_fifo_buf", []) or []) for w in _mem_layers]
        try:
            _sel = _select_chunks_reader_attn(
                model=model,
                last_chunk=chunks[-1],
                n_chunks=len(chunks),
                device=device,
                select_layer=readerattn_select_layer,
                topk=readerattn_topk,
            )
        finally:
            for w, snap in zip(_mem_layers, _buf_snapshot):
                w._fifo_buf = snap
        if _sel:
            oracle_token_chunks = _sel
    # HNST v2 TREE-TOKEN selection (2026-06-25, DEPLOYABLE): pick the chunk set by
    # a beam descent over the TRAINED navigation tree (learned leaf/node pool +
    # reader q.k), then feed those doc-abs indices into the SAME token-reforward
    # window builder. Same last-chunk-forward buffer snapshot/restore as
    # readerattn. Mutually exclusive with readerattn_token / bm25_token / oracle.
    if tree_token:
        _root = getattr(model, "module", model)
        _mem_layers = getattr(_root, "_mem_space_layers", []) or []
        _buf_snapshot = [list(getattr(w, "_fifo_buf", []) or []) for w in _mem_layers]
        try:
            _sel = _select_chunks_tree(
                model=model,
                last_chunk=chunks[-1],
                n_chunks=len(chunks),
                device=device,
                select_layer=tree_select_layer,
                topk=tree_topk,
                branch=tree_branch,
                beam=tree_beam,
            )
        finally:
            for w, snap in zip(_mem_layers, _buf_snapshot):
                w._fifo_buf = snap
        if _sel:
            oracle_token_chunks = _sel
    # BM25-TOKEN selection (DEPLOYABLE): pick the chunk set by pure lexical BM25
    # word-overlap between the question chunk and each candidate chunk's raw
    # tokens, then feed those document-absolute indices into the SAME oracle-token
    # window builder below (token-reforward readout byte-identical — only the
    # chunk SOURCE differs: BM25 word-overlap vs reader-attn q.k vs the
    # answer-cheating locator). BM25 is pure-CPU and never forwards the model, so
    # — unlike the reader-attn path — there is NO last-chunk forward and thus NO
    # _fifo_buf mutation to snapshot/restore. It reads the SAME candidate pool
    # (same offset/len(buf) arithmetic), so the recall measured in the e2 probe
    # transfers 1:1. Mutually exclusive with readerattn_token (asserted in main).
    if bm25_token:
        _sel = _select_chunks_bm25_token(
            model=model,
            chunks=chunks,
            n_chunks=len(chunks),
            select_layer=bm25_select_layer,
            topk=bm25_topk,
            query_ids=bm25_query_ids,
            content_only=bm25_content_only,
            tokenizer=tokenizer,
        )
        print(f"[mem_space-BABILong] BM25-TOKEN selected chunks (doc-abs idx, "
              f"of {len(chunks)} total, last={len(chunks) - 1} is question): "
              f"{sorted(_sel) if _sel else _sel}", flush=True)
        if _sel:
            oracle_token_chunks = _sel
    try:
        if oracle_token_chunks:
            # ORACLE-TOKEN-SWA: window = raw tokens of selected needle chunks
            # (document-absolute idx in oracle_token_chunks, excluding the last
            # chunk which is always appended) + the last chunk. Re-forwards the
            # needle's ORIGINAL tokens so they re-attend the query at every layer.
            n_chunks = len(chunks)
            last_idx = n_chunks - 1
            sel = sorted(c for c in oracle_token_chunks
                         if 0 <= c < last_idx)  # needle chunks before the last
            pieces = [chunks[c] for c in sel] + [chunks[-1]]
            window = torch.cat(pieces, dim=0)   # [<= (len(sel)+1)*chunk_size]
            cur = window.unsqueeze(0).to(device)
        elif needle_excluded_chunks is None and swa_eval_chunks > 0 and len(chunks) > swa_eval_chunks + 1:
            # Cross-chunk SWA window: last (W+1) chunks concatenated.
            #
            # GUARD ``len(chunks) > W+1`` (not just ``> 1``): we only take this
            # path when there is at least one EARLIER chunk that the window does
            # NOT cover (chunk index < start). Those uncovered chunks reach the
            # final forward solely via the memory bank, while the last (W+1)
            # chunks get direct attention — the genuine remote-SWA benefit, no
            # double counting (each earlier chunk is either bank-only or
            # window-only, never both).
            #
            # When ``len(chunks) <= W+1`` the whole document already fits inside
            # the window, so concatenating "last (W+1) chunks" would re-attend
            # EVERY chunk that was also streamed into the bank (double counting)
            # AND blow the forward window up to (W+1)*chunk_size, far past the
            # training-time single-chunk window (OOD). For short docs (0k is
            # ~97% single-chunk, 1k ~100% two-chunk) that is pure OOD damage.
            # We therefore fall through to the W0 path below, which is
            # byte-identical to no-SWA: chunks[:-1] live only in the bank and
            # the generation window is exactly the last chunk.
            start = max(0, len(chunks) - (swa_eval_chunks + 1))
            window = torch.cat(list(chunks[start:]), dim=0)  # [<= (W+1)*chunk_size]
            cur = window.unsqueeze(0).to(device)
        elif needle_excluded_chunks is not None and swa_eval_chunks > 0 and len(chunks) > swa_eval_chunks + 1:
            # SWA-NEEDLE-EXCLUDED: same as the plain W window (last W+1 chunks,
            # same token count) but every needle chunk inside that window is
            # SWAPPED OUT for the nearest earlier NON-needle chunk. The last
            # chunk (the question) is always kept. This isolates the SWA gain
            # that is NOT explained by the needle physically landing in the
            # window: if score ~= W6 the gain is token-anchoring/OOD-repair; if
            # score ~= W0 the gain was purely "saw the needle".
            n_chunks = len(chunks)
            last_idx = n_chunks - 1
            needles = set(int(c) for c in needle_excluded_chunks)
            W = swa_eval_chunks
            # Target window = the W context chunks just before the last chunk,
            # i.e. indices [last_idx - W, last_idx). Replace any needle (or
            # already-picked) index by walking backwards to the nearest earlier
            # non-needle, non-duplicate context chunk.
            picked: list[int] = []
            used = set(needles)            # never include a needle chunk
            # candidate pointer starts just below the window and we also iterate
            # the natural window slots, swapping as needed.
            desired = list(range(max(0, last_idx - W), last_idx))
            probe = min(desired) - 1 if desired else last_idx - 1
            for slot in desired:
                if slot not in needles and slot not in picked:
                    picked.append(slot)
                    used.add(slot)
                else:
                    # walk backwards to nearest earlier non-needle unused chunk
                    while probe >= 0 and (probe in used or probe in picked):
                        probe -= 1
                    if probe >= 0:
                        picked.append(probe)
                        used.add(probe)
                        probe -= 1
                    # if we run out of earlier chunks, the window simply shrinks
                    # (still needle-free, still no double-count)
            picked = sorted(set(picked))
            pieces = [chunks[c] for c in picked] + [chunks[-1]]
            window = torch.cat(pieces, dim=0)
            cur = window.unsqueeze(0).to(device)
        else:
            # W=0, single chunk, OR SWA fallback (doc fits in window):
            # byte-identical to the original no-SWA path.
            cur = chunks[-1].unsqueeze(0).to(device)  # [1, last_chunk_len]
        generated_ids: list[int] = []
        for step in range(max_new_tokens):
            outputs = model(input_ids=cur, use_cache=False)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            if step == 0 and tokenizer.eos_token_id is not None:
                # Match H6 behaviour: suppress EOS as the very first generated
                # token so we don't return an empty answer.
                logits[:, tokenizer.eos_token_id] = float("-inf")
            next_tok = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            tok_id = int(next_tok.item())
            if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id and step > 0:
                break
            generated_ids.append(tok_id)
            cur = torch.cat([cur, next_tok], dim=-1)
    finally:
        _unfreeze_banks(model)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Batched (cell-internal) generation — opt-in via --batch_size > 1
# --------------------------------------------------------------------------- #


def _set_active_token_mask(model, mask) -> None:
    """Stash the current chunk's [B, T] token mask (1=real, 0=pad) on the
    memory bank(s) so MemorySpaceLayer's selector pooling can exclude pads.
    ``mask=None`` clears it (full/streaming chunks)."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank._active_token_mask = mask
        return
    for w in getattr(root, "_mem_space_layers", []) or []:
        _b = getattr(w, "memory_bank", None)
        if _b is not None:
            _b._active_token_mask = mask


def _set_prev_chunk_token_mask(model, mask) -> None:
    """Stash the PREVIOUS chunk's [B, T] token mask on the L3 pool so the
    recursive L3 summary reduces over real tokens only. ``mask=None`` clears."""
    root = getattr(model, "module", model)
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._prev_chunk_token_mask = mask


@torch.no_grad()
def generate_batch_with_mem_space(
    model,
    token_list,
    tokenizer,
    chunk_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    """Batched streaming generation for several BABILong samples at once.

    All samples in ``token_list`` MUST share the same number of chunks
    (``ceil(len/chunk_size)``) and that number MUST be >= 2 — the caller
    (``main``) buckets samples by chunk-count and routes single-chunk samples
    to the bsz=1 path. Under that contract every "streaming" chunk
    (``chunks[:-1]``) is EXACTLY ``chunk_size`` long for every sample (because
    ``Tensor.split`` only shortens the final chunk), so the streaming forwards
    are unpadded and byte-identical to the bsz=1 path. ONLY the final
    generation chunk varies in length and is RIGHT-padded; right-padding is
    free under causal self-attention (real tokens never attend to trailing
    pads), so the wrapped decoder needs no mask change — only the two
    non-causal pooling reductions (selector routing + the recursive L3 summary)
    receive an explicit token mask so they ignore pad positions.

    Args:
        token_list: list of 1-D LongTensors (variable length), same chunk-count.

    Returns:
        list[str]: decoded answer for each input, in the same order.
    """
    B = len(token_list)
    if B == 0:
        return []
    if device is None:
        device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    _reset_banks(model)
    _reset_l2(model)
    _reset_fifo_memory(model)  # H-2 FIX (2026-07-02): batch path also must clear FIFO buffer per-sample (no-op on non-FIFO)

    # Split each sample into chunks; verify the shared-chunk-count contract.
    per_sample_chunks = [list(t.split(chunk_size)) for t in token_list]
    n_chunks = len(per_sample_chunks[0])
    assert n_chunks >= 2, "batched path requires >=2 chunks; caller must bucket"
    for c in per_sample_chunks:
        assert len(c) == n_chunks, "batched samples must share chunk count"

    # ---- Stream all-but-last chunks (full chunk_size, unpadded) ----
    _set_active_token_mask(model, None)
    _set_prev_chunk_token_mask(model, None)
    for j in range(n_chunks - 1):
        stacked = torch.stack([per_sample_chunks[b][j] for b in range(B)], dim=0)
        assert stacked.shape[1] == chunk_size  # streaming chunks are always full
        stacked = stacked.to(device)
        _ = model(input_ids=stacked, use_cache=False)

    # ---- Build the (right-padded) generation chunk ----
    last_chunks = [per_sample_chunks[b][-1] for b in range(B)]
    last_lens = [int(c.shape[0]) for c in last_chunks]
    width = max(last_lens)
    cur = torch.full((B, width), pad_id, dtype=torch.long, device=device)
    for b in range(B):
        cur[b, : last_lens[b]] = last_chunks[b].to(device)
    cur_len = list(last_lens)                      # per-sample real length
    rows = torch.arange(B, device=device)

    def _mask_for(W: int) -> torch.Tensor:
        # [B, W] bool: True for positions < cur_len[b].
        ar = torch.arange(W, device=device).unsqueeze(0)       # [1, W]
        lens = torch.tensor(cur_len, device=device).unsqueeze(1)  # [B, 1]
        return ar < lens

    _freeze_banks(model)
    generated: list[list[int]] = [[] for _ in range(B)]
    finished = [False] * B
    prev_mask = None  # mask of the chunk currently held in l3_pool._prev_chunk_h
    try:
        for step in range(max_new_tokens):
            cur_mask = _mask_for(cur.shape[1])                 # [B, W]
            _set_active_token_mask(model, cur_mask)
            _set_prev_chunk_token_mask(model, prev_mask)

            outputs = model(input_ids=cur, use_cache=False)
            logits_all = outputs.logits                        # [B, W, V]
            # Read each row's logits at its OWN last real position.
            read_pos = torch.tensor(
                [cur_len[b] - 1 for b in range(B)], device=device
            )
            logits = logits_all[rows, read_pos, :]             # [B, V]
            if step == 0 and eos_id is not None:
                logits[:, eos_id] = float("-inf")
            next_tok = logits.argmax(dim=-1)                   # [B]

            # After this forward, l3_pool._prev_chunk_h == this cur's hidden;
            # remember the mask that matches it for the next step's L3 reduce.
            prev_mask = cur_mask

            # Place tokens / update lengths for unfinished samples.
            need_grow = False
            for b in range(B):
                if finished[b]:
                    continue
                tok = int(next_tok[b].item())
                if eos_id is not None and tok == eos_id and step > 0:
                    finished[b] = True
                    continue
                generated[b].append(tok)
                if cur_len[b] >= cur.shape[1]:
                    need_grow = True
            if all(finished):
                break
            # Grow the buffer by one pad column if any sample needs the slot.
            if need_grow:
                pad_col = torch.full((B, 1), pad_id, dtype=torch.long, device=device)
                cur = torch.cat([cur, pad_col], dim=1)
            for b in range(B):
                if finished[b]:
                    continue
                # Write the just-generated token at this sample's next position.
                cur[b, cur_len[b]] = generated[b][-1]
                cur_len[b] += 1
    finally:
        _unfreeze_banks(model)
        _set_active_token_mask(model, None)
        _set_prev_chunk_token_mask(model, None)

    return [
        tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated
    ]


# --------------------------------------------------------------------------- #
# CSV writing (robust against model outputs that contain newlines)
# --------------------------------------------------------------------------- #


def _sanitize_output(text: str) -> str:
    """Flatten embedded newlines/carriage returns in a model output before it
    is written to the result CSV.

    BABILong scoring (``compare_answers``) only checks whether the gold target
    label appears as a substring of the (lower-cased, sentence-truncated)
    output, so collapsing ``\\n``/``\\r`` to a single space is verdict-preserving
    (verified: 0 verdict changes over a polluted n=100 cell). Doing this at
    write time means a single physical CSV line == one record, which keeps the
    file readable by *any* downstream consumer (not just csv.DictReader, which
    already handles quoted multi-line fields, but also naive line-counters and
    quirky pandas paths). We additionally write with QUOTE_ALL as a second line
    of defence.
    """
    if not isinstance(text, str):
        return text
    return text.replace("\r", " ").replace("\n", " ")


def _write_results_csv(df: pd.DataFrame, outfile) -> None:
    """Write the (target, output, question) frame to ``outfile`` with embedded
    newlines flattened and every field quoted.

    Both safeguards are intentional and independent:
      * ``_sanitize_output`` guarantees one physical line per record.
      * ``quoting=csv.QUOTE_ALL`` guarantees correct parsing even if some other
        field (e.g. ``question``) ever grows a delimiter/newline.
    """
    safe = df.copy()
    if "output" in safe.columns:
        safe["output"] = safe["output"].map(_sanitize_output)
    safe.to_csv(outfile, index=False, quoting=csv.QUOTE_ALL)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text or "cuda oom" in text


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="BABILong evaluation for mem_space architecture")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base Llama-3-8B model directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to mem_space adapter .pt checkpoint. Pass "
                             "'none' / 'base' / '' to SKIP adapter loading and "
                             "run ZERO-TRAINING (base backbone + freshly-init "
                             "adapter params, e.g. inject_gate).")
    parser.add_argument("--adapter_config", type=str, required=True,
                        help="Path to adapter_config.json describing the MemorySpaceConfig")
    parser.add_argument("--results_folder", type=str, default="./babilong_results",
                        help="Folder to store BABILong eval results")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Subfolder name for this evaluation run")
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong",
                        help="HuggingFace dataset name")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["qa1", "qa2", "qa5"],
                        help="BABILong tasks to evaluate")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["0k", "1k", "2k", "4k", "8k", "16k"],
                        help="BABILong context lengths to evaluate")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Chunk size for memory accumulation (matches mem_space training seq_len)")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum new tokens to generate per sample")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum samples per task/length cell (default 100; -1 = all)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the (post-limit) sample set of each cell into "
                             "this many stride shards for sample-level parallelism. "
                             "1 (default) = no sharding, byte-identical to the "
                             "original single-process behaviour.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Which stride shard to evaluate (0-based; "
                             "requires --num_shards > 1). Shard i runs samples "
                             "[i::num_shards] so every shard gets an evenly "
                             "interleaved subset (no shard is all-hard samples).")
    parser.add_argument("--swa_oracle_token", action="store_true", default=False,
                        help="ORACLE-TOKEN-SWA probe (2026-06-27): final-forward "
                             "window = RAW TOKENS of the oracle-located needle "
                             "chunk(s) + last chunk, re-forwarded (live, "
                             "query-conditional, no dilution) instead of stored "
                             "hidden. Decisive 'live token >> frozen hidden "
                             "snapshot' test. Locates needle per-sample; bsz=1.")
    parser.add_argument("--swa_needle_excluded", action="store_true", default=False,
                        help="SWA-NEEDLE-EXCLUDED probe (2026-06-27): identical to a "
                             "plain --swa_eval_chunks W window, EXCEPT any needle chunk "
                             "(located per-sample via the gold answer) that would fall "
                             "inside the last (W+1) window is REPLACED by the nearest "
                             "earlier non-needle chunk, so the SWA window has the SAME "
                             "token count as plain W6 but provably does NOT contain the "
                             "needle. Decisive test of WHY W6>>W0: if score stays high "
                             "(near W6) the SWA gain is token-anchoring / OOD-repair "
                             "(needle-independent); if it falls back to W0 the gain was "
                             "purely from seeing the needle. Requires --swa_eval_chunks>0 "
                             "and --batch_size 1.")
    parser.add_argument("--swa_readerattn_token", action="store_true", default=False,
                        help="READER-ATTN-TOKEN probe (2026-06-27, DEPLOYABLE): "
                             "identical token-reforward readout as --swa_oracle_token "
                             "(same window builder + decode loop) but the selected "
                             "chunk set comes from a DEPLOYABLE reader-attn top-k "
                             "selection (the reader's own native q.k salience at one "
                             "model-level selection layer L16 over the streamed FIFO "
                             "hidden buffer) INSTEAD of the oracle needle-locator "
                             "(which cheats by reading the gold answer). Measures the "
                             "deployable upper bound = selection precision x "
                             "token-reforward readout. Requires --batch_size 1; "
                             "mutually exclusive with --swa_oracle_token.")
    parser.add_argument("--swa_readerattn_topk", type=int, default=4,
                        help="Top-k chunks to select for --swa_readerattn_token "
                             "(default 4). No recency floor (the last/question chunk "
                             "is always appended by the window builder).")
    parser.add_argument("--swa_readerattn_select_layer", type=int, default=16,
                        help="Which wrapped decoder layer's native q.k drives the "
                             "reader-attn chunk selection for --swa_readerattn_token "
                             "(default 16). Sweep target.")
    parser.add_argument("--swa_tree_token", action="store_true", default=False,
                        help="HNST v2 TREE-TOKEN probe (2026-06-25, DEPLOYABLE): "
                             "identical token-reforward readout as "
                             "--swa_readerattn_token but the chunk set comes from a "
                             "beam descent over the TRAINED navigation tree "
                             "(model._tree_pool learned aggregation + reader q.k). "
                             "Requires a checkpoint trained with --use_tree_summary "
                             "and --batch_size 1. Mutually exclusive with the other "
                             "--swa_*_token probes.")
    parser.add_argument("--swa_tree_topk", type=int, default=4,
                        help="Top-k leaf chunks surfaced by the tree descent for "
                             "--swa_tree_token (default 4).")
    parser.add_argument("--swa_tree_select_layer", type=int, default=16,
                        help="Selection layer for --swa_tree_token (default 16).")
    parser.add_argument("--swa_tree_branch", type=int, default=4,
                        help="B-ary branching factor of the navigation tree for "
                             "--swa_tree_token (match training --t2_tree_branch).")
    parser.add_argument("--swa_tree_beam", type=int, default=2,
                        help="Beam width per internal level for --swa_tree_token "
                             "(match training --t2_tree_beam).")
    parser.add_argument("--swa_bm25_token", action="store_true", default=False,
                        help="BM25-TOKEN probe (DEPLOYABLE): identical "
                             "token-reforward readout as --swa_oracle_token / "
                             "--swa_readerattn_token (same window builder + decode "
                             "loop) but the selected chunk set comes from pure "
                             "lexical BM25 word-overlap between the question chunk "
                             "and each candidate context chunk's RAW tokens "
                             "(zero-training, no model forward, no oracle). The "
                             "candidate pool is the SAME post-eviction FIFO buffer "
                             "the reader-attn path sees, so the long-doc recall@K "
                             "measured for BM25 (0.52-0.73) transfers 1:1. Tests "
                             "whether BM25-selected chunks beat the reader-attn "
                             "deployment end-to-end. Requires --batch_size 1; "
                             "mutually exclusive with --swa_oracle_token and "
                             "--swa_readerattn_token.")
    parser.add_argument("--swa_bm25_topk", type=int, default=4,
                        help="Top-k chunks to select for --swa_bm25_token "
                             "(default 4). No recency floor (the last/question "
                             "chunk is always appended by the window builder).")
    parser.add_argument("--swa_bm25_select_layer", type=int, default=16,
                        help="Which wrapped decoder layer's FIFO buffer defines the "
                             "candidate pool for --swa_bm25_token (default 16, "
                             "matching --swa_readerattn_select_layer so both paths "
                             "rank the identical pool).")
    parser.add_argument("--swa_bm25_content_only", action="store_true", default=False,
                        help="content-only BM25: filter generic stopwords out of the "
                             "query before BM25 so high-IDF entity words drive chunk "
                             "selection. Validated: recall@4 0.52->0.67-0.72 (qa5 16k). "
                             "Zero-train improvement over plain --swa_bm25_token.")
    # ----- Token-reforward MASTER SWITCH (2026-06-25 REFORWARD-GUARD) ----- #
    # The --swa_*_token probes (oracle / readerattn / tree / bm25) all drive the
    # SAME token-reforward readout window (generate_with_mem_space:1078, re-feeding
    # SELECTED chunks' RAW TOKENS through the whole model, ≈RAG). That is a
    # THEORETICAL UPPER BOUND / oracle ceiling (qa5 reforward 8k=52 vs hidden
    # ceiling 8k=28), NOT a deployable readout. It is HARD-DISABLED unless this flag
    # is passed; when passed, results dir gets a _UPPERBOUND suffix + cfg json is
    # tagged theoretical_upper_bound=true. Mirrors the train red-line block.
    # status/REFORWARD_AUDIT.md.  NOTE: --fifo_keep_set_mode (incl. oracle) and
    # --swa_eval_chunks are HIDDEN / contiguous-window paths, NOT reforward, and are
    # intentionally NOT guarded here.
    parser.add_argument("--allow_token_reforward", action="store_true", default=False,
                        help="Explicitly permit the token-reforward probes "
                             "(--swa_oracle_token / --swa_readerattn_token / "
                             "--swa_tree_token / --swa_bm25_token). OFF by default: "
                             "any of those aborts with a [REFORWARD-GUARD] "
                             "SystemExit. Reforward is a THEORETICAL UPPER BOUND "
                             "(≈RAG oracle), not deployable; when enabled the "
                             "results dir is suffixed _UPPERBOUND and tagged "
                             "theoretical_upper_bound=true.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Cell-internal sample batch size. 1 (default) = the "
                             "original byte-for-byte per-sample path. >1 batches "
                             "same-chunk-count samples through a single forward "
                             "(~1.4x/cell); single-chunk samples always use the "
                             "bsz=1 path. NOTE: batching is numerically correct "
                             "(B=1 batched == bsz=1 exactly) but >1 under bf16 + "
                             "hard top-k routing + greedy decode does NOT preserve "
                             "the exact BABILong score (qa2/2k drifted 27->21 over "
                             "n=100). Use >1 only for fast triage, not final numbers.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--swa_eval_chunks", type=int, default=0,
                        help="Eval-only cross-chunk sliding-window attention "
                             "(D2a). W=0 (default) = original behaviour, "
                             "bit-identical: the generation window is the last "
                             "chunk only and earlier chunks reach the final "
                             "forward solely via the memory bank. W>0 makes the "
                             "generation window the last (W+1) chunks "
                             "concatenated, so the final forward attends "
                             "DIRECTLY to the previous W chunks' raw KV (in "
                             "addition to memory readback). Bank streaming is "
                             "unchanged. Only supported on the bsz=1 path.")
    parser.add_argument("--use_slot_kv_cache", action="store_true", default=False,
                        help="Enable per-slot raw-KV cache readout during eval. "
                             "Streaming context chunks append raw hidden states "
                             "under the selected slot ids, and the generation "
                             "chunk retrieves raw KV from the currently selected "
                             "slots via the existing in-attention KV concat path. "
                             "Default off = existing W0/SWA behaviour unchanged.")
    parser.add_argument("--slot_kv_cache_layer", type=int, default=None,
                        help="Single decoder layer that owns per-slot raw-KV "
                             "cache write/read at eval. Default: value from "
                             "adapter_config.json if present, else 16. Only used "
                             "when --use_slot_kv_cache is set.")
    parser.add_argument("--slot_kv_select_mode", type=str, default="router",
                        choices=["router", "all", "recency"],
                        help="Which slot ids retrieve per-slot raw-KV at eval: "
                             "router uses current selector top-k (default), all "
                             "passes every slot id, recency uses the most recently "
                             "written top_k unique slot ids.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--rawkv_disable_col_bias", action="store_true",
                        help="Eval-time ablation: zero the trained gist col_bias "
                             "so the reader attends raw-KV via its OWN native q.k "
                             "only (no trained selection head). Tests whether the "
                             "reader's native attention is itself the retriever.")
    parser.add_argument("--rawkv_eval_topk", type=int, default=0,
                        help="Eval-time override of rawkv_readout_topk_chunks "
                             "(candidate restriction): keep only top-k chunks by "
                             "gist salience. With --rawkv_disable_col_bias, attend "
                             "them with no trained weight bias. 0 = no override.")
    parser.add_argument("--rawkv_keep_set_mode", type=str, default="",
                        choices=["", "gist", "reader_attn", "oracle"],
                        help="Kept-chunk selection mode for rawkv readout. "
                             "'reader_attn' = pick top-k chunks by the reader's "
                             "own native q.k salience (no trained scorer; HARD "
                             "isolation gathers only those into attention). "
                             "Empty = use adapter_config / default (gist).")
    parser.add_argument("--rawkv_grouped_readout", action="store_true",
                        help="(B) two-stage grouped-softmax readout: retrieved "
                             "sub-blocks compete as units (stage1) with internal "
                             "softmax (stage2), avoiding within-block dilution.")
    parser.add_argument("--rawkv_subblock_size", type=int, default=64,
                        help="Sub-block size for --rawkv_grouped_readout (=Landmark mem_freq).")
    parser.add_argument("--rawkv_stage1_select", action="store_true",
                        help="(B variant B) add per-sub-block reader-attn q.k "
                             "salience as stage-1 selection bias (concentrates "
                             "mass on needle sub-block).")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Wrap the formatted input in the tokenizer's chat template")
    parser.add_argument("--use_instruction", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['instruction']")
    parser.add_argument("--use_examples", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['examples']")
    parser.add_argument("--use_post_prompt", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['post_prompt']")
    parser.add_argument("--memory_disabled", action="store_true", default=False,
                        help="Falsification ablation: disable MemorySpaceLayer memory "
                             "(bypass to vanilla Llama forward) for the ENTIRE "
                             "per-sample inference, covering both the streaming "
                             "chunk ingestion AND the final generation. If scores "
                             "stay high with this flag, the result is an artifact "
                             "(in-context leak / few-shot prior), not evidence the "
                             "memory bank works.")
    # ------------------------------------------------------------------ #
    # FIFO eval-time probes (2026-06-25, H_POS + H_DIL falsification).
    # All default-OFF; off → byte-identical to legacy FIFO behaviour.
    # ------------------------------------------------------------------ #
    parser.add_argument("--fifo_pos_mode", type=str, default="none",
                        choices=["none", "packed", "real"],
                        help="H_POS probe: how to assign RoPE positions to the "
                             "FIFO buffer prefix. 'none' = legacy all-pos-0 "
                             "(current behaviour). 'packed' = kept-set index * "
                             "chunk + in-chunk-offset (in-distribution). "
                             "'real' = original chunk index * chunk + offset "
                             "(may be OOD; relies on Llama-3 RoPE theta=500000 "
                             "extrapolation).")
    parser.add_argument("--fifo_keep_set_mode", type=str, default="none",
                        choices=["none", "flat_readerattn", "oracle", "tree"],
                        help="H_DIL probe: how to select which buffered chunks "
                             "to KEEP in the prefix. 'none' = attend ALL "
                             "(legacy). 'flat_readerattn' = score each chunk "
                             "by reader q.k salience, keep top-K + last R. "
                             "'oracle' = PERFECT isolation: keep ONLY the "
                             "chunk(s) whose token span contains the gold answer "
                             "(needle) + last R recency floor (question). The "
                             "needle's absolute chunk index is located per-sample "
                             "by matching the target answer string in the token "
                             "stream (token pos // chunk_size), mirroring the "
                             "rawkv oracle's per-sample needle-chunk channel. "
                             "Requires --batch_size 1; needle evicted/unknown "
                             "falls back to keep-all (logged).")
    parser.add_argument("--fifo_keep_topk", type=int, default=25,
                        help="K for keep_set top-K (default 25).")
    parser.add_argument("--fifo_keep_recency", type=int, default=2,
                        help="R for keep_set recency floor (default 2 last "
                             "chunks always kept).")
    parser.add_argument("--fifo_keep_all_buffer", action="store_true",
                        default=False,
                        help="When set, suppress FIFO eviction (buffer holds "
                             "ALL past chunks). Use with keep_set_mode to test "
                             "'keep-all-store, attend-few'.")
    parser.add_argument("--fifo_tree_branch", type=int, default=8,
                        help="HNST (--fifo_keep_set_mode tree) branching factor B "
                             "of the max-pool navigation tree (default 8).")
    parser.add_argument("--fifo_tree_beam", type=int, default=2,
                        help="HNST beam width b kept at each internal tree level "
                             "(default 2; b>=2 lets a wrong high-level turn "
                             "recover).")
    parser.add_argument("--fifo_buffer_chunks_eval", type=int, default=0,
                        help="Eval-time override of the FIFO eviction buffer size "
                             "(0 = keep the trained fifo_buffer_chunks). Use to "
                             "evaluate a clean large-buffer ckpt as a SMALLER-buffer "
                             "FIFO so early chunks are EVICTED at long lengths (the "
                             "'amnesia' baseline for the HNST position test). "
                             "Ignored under --fifo_keep_all_buffer.")
    parser.add_argument("--record_needle_pos", action="store_true", default=False,
                        help="Position-stratified eval (HNST decisive test): for "
                             "every bsz=1 sample, locate the gold-answer needle "
                             "chunk(s) and write two extra CSV columns "
                             "'needle_chunks' (';'-joined 0-based doc-absolute "
                             "chunk indices) and 'n_chunks' (total chunks). Lets "
                             "the scorer bucket accuracy by needle position "
                             "(early/mid/late). Default off -> byte-identical CSV.")
    args = parser.parse_args()

    # ----- REFORWARD-GUARD hard block (2026-06-25) ----- #
    # Any --swa_*_token probe drives the token-reforward readout window (raw tokens
    # of SELECTED chunks re-fed through the whole model, ≈RAG). That is a THEORETICAL
    # UPPER BOUND, not a deployable readout, so it is DISABLED unless explicitly
    # opted-in via --allow_token_reforward. Mirrors the train --babilong_mix_fraction
    # red-line block. status/REFORWARD_AUDIT.md.
    _reforward_flags = [
        f for f, on in (
            ("--swa_oracle_token", args.swa_oracle_token),
            ("--swa_readerattn_token", args.swa_readerattn_token),
            ("--swa_tree_token", args.swa_tree_token),
            ("--swa_bm25_token", args.swa_bm25_token),
        ) if on
    ]
    if _reforward_flags and not getattr(args, "allow_token_reforward", False):
        raise SystemExit(
            "[REFORWARD-GUARD] token reforward is a THEORETICAL UPPER BOUND, not a "
            "deployable method. It is DISABLED by default. The flag(s) "
            f"{' '.join(_reforward_flags)} drive a token-reforward readout window "
            "(re-feeding selected chunks' RAW tokens through the whole model, ≈RAG "
            "oracle). If you intend to measure the oracle upper bound, pass "
            "--allow_token_reforward explicitly (results will be tagged "
            "THEORETICAL_UPPER_BOUND)."
        )
    _reforward_upperbound = bool(_reforward_flags)
    if _reforward_upperbound:
        print(
            "[REFORWARD-GUARD] token-reforward probe ENABLED via "
            "--allow_token_reforward: "
            f"{' '.join(_reforward_flags)}. Results are a THEORETICAL_UPPER_BOUND "
            "(≈RAG oracle), NOT a deployable capability; output dir suffixed "
            "_UPPERBOUND and cfg tagged theoretical_upper_bound=true.",
            file=sys.stderr,
        )
        if not args.output_name.endswith("_UPPERBOUND"):
            args.output_name = f"{args.output_name}_UPPERBOUND"

    if args.swa_eval_chunks < 0:
        parser.error("--swa_eval_chunks must be >= 0")
    if args.slot_kv_cache_layer is not None and args.slot_kv_cache_layer < 0:
        parser.error("--slot_kv_cache_layer must be >= 0")
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(
            f"--shard_index must be in [0, num_shards) = [0, {args.num_shards}); "
            f"got {args.shard_index}"
        )
    if args.swa_eval_chunks > 0 and args.batch_size > 1:
        parser.error(
            "--swa_eval_chunks > 0 is only supported on the bsz=1 path "
            "(use --batch_size 1). The batched generation path does not "
            "implement the cross-chunk SWA window."
        )

    if args.swa_oracle_token and args.batch_size > 1:
        parser.error(
            "--swa_oracle_token requires --batch_size 1 (per-sample needle "
            "location + raw-token re-forward is only wired on the bsz=1 path)."
        )

    if args.swa_readerattn_token and args.batch_size > 1:
        parser.error(
            "--swa_readerattn_token requires --batch_size 1 (per-sample "
            "reader-attn chunk selection + raw-token re-forward is only wired "
            "on the bsz=1 path)."
        )
    if args.swa_readerattn_token and args.swa_oracle_token:
        parser.error(
            "--swa_readerattn_token and --swa_oracle_token are mutually "
            "exclusive (both drive the token-reforward window from a different "
            "chunk source). Pick one."
        )

    if args.swa_bm25_token and args.batch_size > 1:
        parser.error(
            "--swa_bm25_token requires --batch_size 1 (per-sample BM25 chunk "
            "selection + raw-token re-forward is only wired on the bsz=1 path)."
        )
    if args.swa_bm25_token and (args.swa_oracle_token or args.swa_readerattn_token):
        parser.error(
            "--swa_bm25_token is mutually exclusive with --swa_oracle_token and "
            "--swa_readerattn_token (each drives the token-reforward window from a "
            "different chunk source). Pick one."
        )

    if args.swa_tree_token and args.batch_size > 1:
        parser.error(
            "--swa_tree_token requires --batch_size 1 (per-sample tree navigation "
            "+ raw-token re-forward is only wired on the bsz=1 path)."
        )
    if args.swa_tree_token and (args.swa_oracle_token or args.swa_readerattn_token
                                or args.swa_bm25_token):
        parser.error(
            "--swa_tree_token is mutually exclusive with the other --swa_*_token "
            "probes (each drives the token-reforward window from a different chunk "
            "source). Pick one."
        )

    print(f"[mem_space-BABILong] Configuration:")
    print(f"  Base model:      {args.model_path}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  Adapter config:  {args.adapter_config}")
    print(f"  Tasks:           {args.tasks}")
    print(f"  Lengths:         {args.lengths}")
    print(f"  Chunk size:      {args.chunk_size}")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print(f"  Limit/cell:      {args.limit}")
    if args.num_shards > 1:
        print(f"  Sharding:        shard {args.shard_index} of {args.num_shards} "
              f"(stride slice [{args.shard_index}::{args.num_shards}])")
    print(f"  SWA eval chunks: {args.swa_eval_chunks}")
    print(f"  Device:          {args.device}")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[mem_space-BABILong] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Load + parse adapter config
    with open(args.adapter_config, "r") as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    slot_kv_cache_layer = (
        int(args.slot_kv_cache_layer)
        if args.slot_kv_cache_layer is not None
        else int(adapter_cfg.get("slot_kv_cache_layer", 16))
    )
    mem_config.use_slot_kv_cache = bool(args.use_slot_kv_cache)
    mem_config.slot_kv_cache_layer = slot_kv_cache_layer
    mem_config.slot_kv_select_mode = args.slot_kv_select_mode
    if args.use_slot_kv_cache:
        print(f"[mem_space-BABILong] slot_kv_cache enabled at layer {slot_kv_cache_layer} "
              f"select_mode={args.slot_kv_select_mode}")
    # Eval-time ablation override (go/no-go): force pure reader attention over
    # raw-KV by zeroing the trained gist col_bias.
    if getattr(args, "rawkv_disable_col_bias", False):
        mem_config.rawkv_disable_col_bias = True
        # Keep ALL chunks so the reader attends the FULL historical raw-KV and
        # selection is purely its own native q.k (not gist-salience top-k). 0 =
        # keep_all in GistReadout.retrieve.
        mem_config.rawkv_readout_topk_chunks = 0
        print("[mem_space-BABILong] rawkv_disable_col_bias=True + topk_chunks=0 "
              "(keep all) -> reader native attention over FULL raw-KV (no trained "
              "selection head)")
    # Candidate-restriction override (2026-06-20): keep only top-k chunks by gist
    # salience BUT (with --rawkv_disable_col_bias) attend them with NO trained
    # weight bias — tests whether RESTRICTING candidates (vs keep_all=16) lets the
    # reader's native attention isolate the needle (climbing toward the 82-97%
    # isolated-injection ceiling). Applied AFTER the disable_col_bias block so it
    # overrides the keep_all there.
    if getattr(args, "rawkv_eval_topk", 0) and args.rawkv_eval_topk > 0:
        mem_config.rawkv_readout_topk_chunks = int(args.rawkv_eval_topk)
        print(f"[mem_space-BABILong] rawkv_eval_topk override -> "
              f"topk_chunks={args.rawkv_eval_topk} (candidate restriction)")
    if getattr(args, "rawkv_keep_set_mode", ""):
        mem_config.rawkv_keep_set_mode = args.rawkv_keep_set_mode
        print(f"[mem_space-BABILong] rawkv_keep_set_mode={args.rawkv_keep_set_mode} "
              f"(HARD isolation: gather only kept chunks into attention)")
    if getattr(args, "rawkv_grouped_readout", False):
        mem_config.rawkv_grouped_readout = True
        mem_config.rawkv_subblock_size = int(args.rawkv_subblock_size)
        mem_config.rawkv_stage1_select = bool(getattr(args, "rawkv_stage1_select", False))
        print(f"[mem_space-BABILong] rawkv_grouped_readout=True subblock="
              f"{args.rawkv_subblock_size} stage1_select={mem_config.rawkv_stage1_select} "
              f"(B: two-stage block-select x within-block softmax)")
    # L3 token-recon head builds pos_queries of shape [l3_recon_max_positions, d].
    # At train time this is set to chunk_size (train_mem_space_dolmino_cpt.py:1088),
    # but adapter_config.json carries no chunk_size, so the dataclass default (1024)
    # would mismatch a ckpt trained with a different chunk_size. Mirror training here.
    mem_config.l3_recon_max_positions = args.chunk_size
    if mem_config.use_slot_kv_cache:
        print(f"[mem_space-BABILong] MemorySpaceConfig: num_slots={mem_config.num_slots}, "
              f"top_k={mem_config.top_k}, selector_dim={mem_config.selector_dim}, "
              f"warmup_steps={mem_config.writeback_gate_warmup_steps}, "
              f"slot_init={mem_config.slot_init}, "
              f"shared_bank={mem_config.shared_memory_bank}, "
              f"hidden_to_slot_frozen={mem_config.hidden_to_slot_frozen}, "
              f"use_slot_kv_cache={mem_config.use_slot_kv_cache}, "
              f"slot_kv_cache_layer={mem_config.slot_kv_cache_layer}, "
              f"slot_kv_select_mode={mem_config.slot_kv_select_mode}")
    else:
        print(f"[mem_space-BABILong] MemorySpaceConfig: num_slots={mem_config.num_slots}, "
              f"top_k={mem_config.top_k}, selector_dim={mem_config.selector_dim}, "
              f"warmup_steps={mem_config.writeback_gate_warmup_steps}, "
              f"slot_init={mem_config.slot_init}, "
              f"shared_bank={mem_config.shared_memory_bank}, "
              f"hidden_to_slot_frozen={mem_config.hidden_to_slot_frozen}")

    # Build + load model
    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )

    # ------------------------------------------------------------------ #
    # Wire FIFO eval-time probes (no-op when all flags are at defaults).
    # ------------------------------------------------------------------ #
    _pos_mode_cli = getattr(args, "fifo_pos_mode", "none")
    if _pos_mode_cli and _pos_mode_cli != "none":
        _set_fifo_pos_mode(model, _pos_mode_cli)
        print(f"[mem_space-BABILong] FIFO probe: fifo_pos_mode={_pos_mode_cli} "
              f"(H_POS: RoPE positions for buffer prefix)")
    _ks_mode_cli = getattr(args, "fifo_keep_set_mode", "none")
    if _ks_mode_cli and _ks_mode_cli != "none":
        _set_fifo_keep_set_mode(
            model, _ks_mode_cli,
            topk=args.fifo_keep_topk, recency=args.fifo_keep_recency,
            tree_branch=args.fifo_tree_branch, tree_beam=args.fifo_tree_beam,
        )
        if _ks_mode_cli == "tree":
            print(f"[mem_space-BABILong] FIFO probe: fifo_keep_set_mode=tree "
                  f"branch={args.fifo_tree_branch} beam={args.fifo_tree_beam} "
                  f"topk={args.fifo_keep_topk} recency={args.fifo_keep_recency} "
                  f"(HNST tree beam descent)")
        else:
            print(f"[mem_space-BABILong] FIFO probe: fifo_keep_set_mode={_ks_mode_cli} "
                  f"topk={args.fifo_keep_topk} recency={args.fifo_keep_recency} (H_DIL)")
    # FIFO ORACLE keep-set (perfect isolation): needle chunk located per-sample
    # in the bsz=1 loop. Requires --batch_size 1 (the batched path has no
    # per-sample oracle wiring). When on we also reset the per-document FIFO
    # buffer per sample (see _set_fifo_oracle_needle).
    _fifo_oracle_on = (_ks_mode_cli == "oracle")
    if _fifo_oracle_on and args.batch_size > 1:
        parser.error(
            "--fifo_keep_set_mode oracle requires --batch_size 1 "
            "(per-sample needle location is only wired on the bsz=1 path)."
        )
    if getattr(args, "fifo_keep_all_buffer", False):
        _set_fifo_keep_all_buffer(model, True)
        print(f"[mem_space-BABILong] FIFO probe: fifo_keep_all_buffer=True "
              f"(eviction suppressed)")
    if getattr(args, "fifo_buffer_chunks_eval", 0) and not getattr(args, "fifo_keep_all_buffer", False):
        _set_fifo_buffer_chunks(model, args.fifo_buffer_chunks_eval)
        print(f"[mem_space-BABILong] FIFO probe: fifo_buffer_chunks_eval="
              f"{args.fifo_buffer_chunks_eval} (eviction buffer size override; "
              f"early chunks evicted at long lengths -> amnesia baseline)")
    if args.swa_readerattn_token:
        print(f"[mem_space-BABILong] READER-ATTN-TOKEN probe (deployable): "
              f"select_layer={args.swa_readerattn_select_layer} topk={args.swa_readerattn_topk} recency=0 "
              f"(token-reforward readout identical to --swa_oracle_token; "
              f"chunk source = reader-attn q.k selection, no oracle)")
    if args.swa_bm25_token:
        print(f"[mem_space-BABILong] BM25-TOKEN probe (deployable): "
              f"select_layer={args.swa_bm25_select_layer} topk={args.swa_bm25_topk} "
              f"(token-reforward readout identical to --swa_oracle_token; "
              f"chunk source = lexical BM25 word-overlap, no oracle, no forward)")

    # ------------------------------------------------------------------ #
    # BABILong eval loop (mirrors run_babilong_h6.py:406-512)
    # ------------------------------------------------------------------ #
    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue

        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if args.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if args.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if args.use_post_prompt else "",
            "template":    DEFAULT_TEMPLATE,
            "chat_template": args.use_chat_template,
            "system_prompt": "",
        }
        prompt_name = "_".join(
            [f"{k}_yes" if prompt_cfg[k] else f"{k}_no"
             for k in prompt_cfg if k != "template"]
        )

        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            print(f"\n[mem_space-BABILong] task={task}, length={split_name}")

            try:
                data = load_babilong_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load dataset {args.dataset_name}/{split_name}/{task}: {e}")
                continue

            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            # Sharded runs write to a per-shard CSV so concurrent shards never
            # clobber each other; num_shards==1 keeps the original filename
            # exactly (byte-identical, no suffix). The scorer
            # (score_nested_babilong.py) globs ``{task}_{length}_*.csv`` so it
            # transparently picks up either the single full cell or the set of
            # shard files and merges them.
            sharded = args.num_shards > 1
            shard_tag = (
                f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
            )
            outfile = outdir / f"{task}_{split_name}_{prompt_name}{shard_tag}.csv"
            cfg_file = outdir / f"{task}_{split_name}_{prompt_name}{shard_tag}.json"

            json.dump(
                {
                    "prompt": prompt_cfg,
                    "generate_kwargs": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "num_beams": 1,
                    },
                    # REFORWARD-GUARD tag (2026-06-25): true when this cell used a
                    # token-reforward probe (--swa_*_token). Those numbers are a
                    # THEORETICAL UPPER BOUND (≈RAG oracle), not a deployable
                    # capability. status/REFORWARD_AUDIT.md.
                    "theoretical_upper_bound": bool(_reforward_upperbound),
                    "model": {
                        "model_path":      args.model_path,
                        "checkpoint":      args.checkpoint,
                        "adapter_config":  args.adapter_config,
                        "chunk_size":      args.chunk_size,
                        "swa_eval_chunks": args.swa_eval_chunks,
                        **({
                            "use_slot_kv_cache": mem_config.use_slot_kv_cache,
                            "slot_kv_cache_layer": mem_config.slot_kv_cache_layer,
                            "slot_kv_select_mode": mem_config.slot_kv_select_mode,
                        } if mem_config.use_slot_kv_cache else {}),
                        "num_slots":       mem_config.num_slots,
                        "top_k":           mem_config.top_k,
                        "shared_memory_bank": mem_config.shared_memory_bank,
                    },
                },
                open(cfg_file, "w"),
                indent=4,
            )

            if getattr(args, "record_needle_pos", False):
                df = pd.DataFrame({"target": [], "output": [], "question": [],
                                   "needle_chunks": [], "n_chunks": []})
            else:
                df = pd.DataFrame({"target": [], "output": [], "question": []})

            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)

            # Sample-level sharding: take a stride slice of the post-limit index
            # range so this process only evaluates its shard. Stride slicing
            # ([i::N]) interleaves samples across shards, so no single shard gets
            # a contiguous (potentially all-hard) block. num_shards==1 yields
            # exactly list(range(num_samples)) — byte-identical to the original.
            sample_indices = list(range(num_samples))[args.shard_index::args.num_shards]
            if sharded:
                print(f"[mem_space-BABILong] shard {args.shard_index}/{args.num_shards}: "
                      f"{len(sample_indices)} of {num_samples} samples "
                      f"(indices {sample_indices[:3]}{'...' if len(sample_indices) > 3 else ''})")

            def _encode_sample(idx):
                sample = task_data[idx]
                input_text = get_formatted_input(
                    sample["input"],
                    sample["question"],
                    prompt_cfg["examples"],
                    prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"],
                )
                if args.use_chat_template:
                    messages = [{"role": "user", "content": input_text}]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                return sample["target"], sample["question"], ids

            if args.batch_size <= 1:
                # ---- bsz=1 path: byte-for-byte the original per-sample loop ----
                for idx in tqdm(sample_indices, desc=f"{task}/{split_name}", leave=False):
                    target, question, input_ids = _encode_sample(idx)
                    input_ids = input_ids.to(device)
                    # CROSS-DOC SAFETY: with --fifo_keep_all_buffer eviction is
                    # OFF, so the FIFO buffer would otherwise ACCUMULATE chunks
                    # across documents (generate_with_mem_space's _reset_banks
                    # does NOT clear _fifo_buf). The oracle path already resets via
                    # _set_fifo_oracle_needle; for every other keep-all-buffer mode
                    # (tree / flat_readerattn) we must clear the buffer per sample
                    # so each document builds its tree from a clean slate.
                    if getattr(args, "fifo_keep_all_buffer", False) and not _fifo_oracle_on:
                        _reset_fifo_memory(model)
                    # FIFO ORACLE keep-set: locate the needle (gold answer) chunk
                    # for THIS sample and stash it on the FIFO layers (mirrors the
                    # rawkv oracle's per-sample needle-chunk channel). Also resets
                    # the per-document FIFO buffer so absolute chunk indices are
                    # valid. No-op unless --fifo_keep_set_mode oracle.
                    if _fifo_oracle_on:
                        _needle = _locate_needle_chunks(
                            input_ids, target, tokenizer, args.chunk_size
                        )
                        _set_fifo_oracle_needle(model, _needle)
                    # POSITION-STRATIFIED recording (HNST decisive test): locate
                    # the needle chunk(s) so the scorer can bucket by position.
                    _rec_needle = None
                    _rec_nchunks = None
                    if getattr(args, "record_needle_pos", False):
                        import math as _m
                        _rec_nchunks = max(1, _m.ceil(input_ids.shape[1] / args.chunk_size))
                        _rec_needle = _locate_needle_chunks(
                            input_ids, target, tokenizer, args.chunk_size
                        )
                    # ORACLE-TOKEN-SWA: locate needle chunks to re-forward their
                    # RAW TOKENS (independent of the FIFO-hidden oracle path).
                    _oracle_tok = None
                    if args.swa_oracle_token:
                        _oracle_tok = _locate_needle_chunks(
                            input_ids, target, tokenizer, args.chunk_size
                        )
                    _needle_excl = None
                    if args.swa_needle_excluded:
                        # locate needle so the SWA window can EXCLUDE it; if it
                        # can't be located, pass empty set (window = plain SWA)
                        _nx = _locate_needle_chunks(
                            input_ids, target, tokenizer, args.chunk_size
                        )
                        _needle_excl = _nx if _nx is not None else set()
                    # BM25-TOKEN query: encode the BARE question STRING (no
                    # special tokens, no instruction/few-shot/post_prompt) so the
                    # deployable BM25 chunk selection ranks by the SAME query the
                    # e2 probe's score_S2_bm25 used (q_ids = pure question). Only
                    # built when the flag is on; otherwise stays None and the
                    # selector falls back to chunks[-1] (byte-identical to before).
                    _bm25_q_ids = None
                    if args.swa_bm25_token:
                        _bm25_q_ids = tokenizer.encode(
                            (question or "").strip(), add_special_tokens=False
                        )
                    try:
                        if args.memory_disabled:
                            _set_memory_disabled(model, True)
                        try:
                            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                                output = generate_with_mem_space(
                                    model=model,
                                    input_ids=input_ids,
                                    tokenizer=tokenizer,
                                    chunk_size=args.chunk_size,
                                    max_new_tokens=args.max_new_tokens,
                                    device=device,
                                    swa_eval_chunks=args.swa_eval_chunks,
                                    oracle_token_chunks=_oracle_tok,
                                    needle_excluded_chunks=_needle_excl,
                                    readerattn_token=args.swa_readerattn_token,
                                    readerattn_select_layer=args.swa_readerattn_select_layer,
                                    readerattn_topk=args.swa_readerattn_topk,
                                    tree_token=args.swa_tree_token,
                                    tree_select_layer=args.swa_tree_select_layer,
                                    tree_topk=args.swa_tree_topk,
                                    tree_branch=args.swa_tree_branch,
                                    tree_beam=args.swa_tree_beam,
                                    bm25_token=args.swa_bm25_token,
                                    bm25_select_layer=args.swa_bm25_select_layer,
                                    bm25_topk=args.swa_bm25_topk,
                                    bm25_query_ids=_bm25_q_ids,
                                    bm25_content_only=args.swa_bm25_content_only,
                                )
                        finally:
                            if args.memory_disabled:
                                _set_memory_disabled(model, False)
                    except RuntimeError as e:
                        if not _is_cuda_oom(e):
                            raise
                        output = "[OOM]"
                        print(f"[OOM] sample_idx={idx} task={task} length={split_name}: {e}", flush=True)
                        _reset_banks(model)
                        _reset_l2(model)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if getattr(args, "record_needle_pos", False):
                        _nc_str = (";".join(str(int(c)) for c in sorted(_rec_needle))
                                   if _rec_needle else "")
                        df.loc[len(df)] = [target, output, question,
                                           _nc_str, int(_rec_nchunks or 0)]
                    else:
                        df.loc[len(df)] = [target, output, question]
                    if len(df) % 10 == 0 or idx == sample_indices[-1]:
                        _write_results_csv(df, outfile)
            else:
                # ---- batched path: bucket by chunk-count, then batch ----
                # Encode everything first so we can group by chunk count. Each
                # row keeps its original index so the CSV order is preserved.
                import math as _math
                rows = []  # (orig_idx, target, question, tokens_1d, n_chunks)
                for idx in sample_indices:
                    target, question, input_ids = _encode_sample(idx)
                    toks = input_ids[0]
                    n_chunks = max(1, _math.ceil(toks.shape[0] / args.chunk_size))
                    rows.append((idx, target, question, toks, n_chunks))

                results: dict = {}  # orig_idx -> output text

                # Single-chunk samples: must use the bsz=1 cold-start path.
                singles = [r for r in rows if r[4] <= 1]
                multis = [r for r in rows if r[4] > 1]
                for (idx, target, question, toks, _nc) in tqdm(
                    singles, desc=f"{task}/{split_name}/single", leave=False
                ):
                    if args.memory_disabled:
                        _set_memory_disabled(model, True)
                    try:
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            out = generate_with_mem_space(
                                model=model,
                                input_ids=toks.unsqueeze(0).to(device),
                                tokenizer=tokenizer,
                                chunk_size=args.chunk_size,
                                max_new_tokens=args.max_new_tokens,
                                device=device,
                                swa_eval_chunks=args.swa_eval_chunks,
                            )
                    finally:
                        if args.memory_disabled:
                            _set_memory_disabled(model, False)
                    results[idx] = out

                # Multi-chunk samples: group by exact chunk count, then split
                # into batches of <= batch_size.
                from collections import defaultdict as _dd
                by_nc = _dd(list)
                for r in multis:
                    by_nc[r[4]].append(r)
                for nc, group in by_nc.items():
                    # Sort the group by total token length so each <=batch_size
                    # slice has similar last-chunk lengths → minimal right-pad
                    # (less wasted compute, and padded rows stay numerically
                    # closer to the unpadded last chunk).
                    group = sorted(group, key=lambda r: int(r[3].shape[0]))
                    for s in tqdm(
                        range(0, len(group), args.batch_size),
                        desc=f"{task}/{split_name}/nc{nc}", leave=False,
                    ):
                        batch = group[s:s + args.batch_size]
                        tok_list = [b[3] for b in batch]
                        if args.memory_disabled:
                            _set_memory_disabled(model, True)
                        try:
                            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                                outs = generate_batch_with_mem_space(
                                    model=model,
                                    token_list=tok_list,
                                    tokenizer=tokenizer,
                                    chunk_size=args.chunk_size,
                                    max_new_tokens=args.max_new_tokens,
                                    device=device,
                                )
                        finally:
                            if args.memory_disabled:
                                _set_memory_disabled(model, False)
                        for b, o in zip(batch, outs):
                            results[b[0]] = o

                # Reassemble in original order.
                for (idx, target, question, _toks, _nc) in rows:
                    df.loc[len(df)] = [target, results[idx], question]
                _write_results_csv(df, outfile)

            _write_results_csv(df, outfile)
            print(f"[mem_space-BABILong] Saved {len(df)} results to {outfile}")

    print("\n[mem_space-BABILong] Evaluation complete!")
    if _fifo_oracle_on:
        _fb, _ev = _fifo_oracle_fallback_total(model)
        print(f"[mem_space-BABILong] FIFO oracle keep-all fallbacks (cumulative "
              f"across all forwards/layers): {_fb} (of which needle-evicted: "
              f"{_ev}). Non-zero = some chunks could not isolate the needle "
              f"(e.g. needle in an evicted chunk at long lengths) and kept all.")


if __name__ == "__main__":
    main()
