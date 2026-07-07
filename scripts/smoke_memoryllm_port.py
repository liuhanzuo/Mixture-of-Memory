#!/usr/bin/env python
"""Smoke test for the ported MemoryLLM (src/memory/memoryllm_ported).

Goal (Pyramid task,阶段 1): confirm the port loads and runs end-to-end on the
local B200/L20A .venv (torch 2.10 + transformers 5.5.4, sm_100):

  (a) MemoryLLM.from_pretrained (absolute path, local_files_only) does not crash;
  (b) the fixed memory pool has the expected shape [L, num_blocks*num_tokens, d]
      and is printed;
  (c) inject_memory writes a short context into the pool (delta_memory shape
      printed) and a subsequent forward / generate produces finite, non-degenerate
      logits (not NaN, not all-zero, not a single collapsed argmax).

GPU policy: all GPUs may be busy with other evals. We pick the first *free* GPU
(lowest used memory) reported by nvidia-smi. If CUDA is unavailable (e.g. run
with CUDA_VISIBLE_DEVICES=""), we fall back to a tiny CPU forward and tag the
result "CPU verified". Loading is bf16 on GPU (~20 GB for 8B + 1.67B pool).

Usage:
    # syntax / import check only (no weights):
    python scripts/smoke_memoryllm_port.py --dry_run
    # full load + inject + generate on the first free GPU:
    python scripts/smoke_memoryllm_port.py
    # force CPU tiny forward:
    CUDA_VISIBLE_DEVICES="" python scripts/smoke_memoryllm_port.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_MEMORYLLM_PATH = "/apdcephfs_wzc1/share_304376610/pighzliu_code/MemoryLLM-source"


def _pick_free_gpu() -> int | None:
    """Return the index of the GPU with the most free memory, or None if no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return 0  # CUDA present but nvidia-smi unavailable; default to 0
    best_idx, best_used = 0, None
    for line in out.strip().splitlines():
        try:
            idx_s, used_s = line.split(",")
            idx, used = int(idx_s.strip()), int(used_s.strip())
        except Exception:
            continue
        if best_used is None or used < best_used:
            best_idx, best_used = idx, used
    return best_idx


def main() -> int:
    ap = argparse.ArgumentParser(description="MemoryLLM port smoke test")
    ap.add_argument("--model_path", type=str, default=DEFAULT_MEMORYLLM_PATH)
    ap.add_argument("--dry_run", action="store_true",
                    help="Only import the port + build tokenizer paths; skip weights.")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--cpu_ctx_tokens", type=int, default=24,
                    help="Context length used on the CPU fallback (>16 required).")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from src.memory.memoryllm_ported import MemoryLLM

    print("=" * 72)
    print("MemoryLLM port smoke test")
    print("=" * 72)
    print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"  model_path={args.model_path}")
    print(f"  local_files_only=True")

    if args.dry_run:
        # Import-only sanity: confirm the class + config are resolvable offline.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        print(f"  [dry_run] config: model_type={cfg.model_type} "
              f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
              f"num_blocks={getattr(cfg, 'num_blocks', '?')} "
              f"num_tokens={getattr(cfg, 'num_tokens', '?')}")
        print("  [dry_run] import + config OK — skipping weight load.")
        return 0

    gpu_idx = _pick_free_gpu()
    if gpu_idx is not None:
        device = torch.device(f"cuda:{gpu_idx}")
        dtype = torch.bfloat16
        where = f"GPU cuda:{gpu_idx}"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        where = "CPU (CUDA unavailable)"
    print(f"  device={where}  dtype={dtype}")

    # ---- (a) load ----------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = MemoryLLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    print("  (a) load: OK")

    # ---- (b) memory pool shape --------------------------------------------
    mem = model.memory
    L = model.L
    nb, nt, d = model.num_blocks, model.num_tokens, model.d
    print(f"  (b) memory pool: shape={tuple(mem.shape)} dtype={mem.dtype} "
          f"(L={L}, num_blocks={nb}, num_tokens={nt}, d={d}) "
          f"-> expected [{L}, {nb*nt}, {d}]")
    print(f"      memory params = {mem.numel()/1e9:.4f} B, "
          f"initialized flag = {int(model.initialized)}, "
          f"add_bos_embedding={model.add_bos_embedding}, "
          f"add_decoder_lora={model.add_decoder_lora}")
    shape_ok = tuple(mem.shape) == (L, nb * nt, d)
    mem_finite = bool(torch.isfinite(mem.float()).all().item())
    print(f"      shape_ok={shape_ok}  memory_all_finite={mem_finite}")

    # ---- (c) inject a short context + forward -----------------------------
    # README hard minimum: inject context must be > 16 tokens.
    ctx = ("Last week, John had a wonderful picnic with David. During their "
           "conversation, David mentioned multiple times that he likes eating "
           "apples. Though he did not mention any other fruits, John says he can "
           "infer that David also likes bananas.")
    ctx_ids = tok(ctx, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)
    if device.type == "cpu":
        # keep the CPU forward tiny — trim to the requested short context.
        ctx_ids = ctx_ids[:, : max(17, args.cpu_ctx_tokens)]
    print(f"  (c) inject_memory: context {ctx_ids.shape[1]} tokens")

    delta = model.inject_memory(ctx_ids, update_memory=True)
    print(f"      delta_memory shape={tuple(delta.shape)} "
          f"(expected [1, {L}, {nt}, {d}]); "
          f"initialized flag now = {int(model.initialized)}")
    delta_finite = bool(torch.isfinite(delta.float()).all().item())
    print(f"      delta_all_finite={delta_finite}")

    # A plain forward now prepends the (updated) memory pool at every layer.
    q = "What fruits does David like?"
    q_ids = tok(q, return_tensors="pt",
                add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=q_ids, return_dict=True)
    logits = out.logits  # [1, T_q, V]
    last = logits[0, -1].float()
    logits_finite = bool(torch.isfinite(last).all().item())
    nonzero = bool((last.abs().sum() > 0).item())
    # degeneracy guard: a healthy LM's next-token distribution is not a spike
    # concentrated on a single id with ~1.0 mass.
    probs = torch.softmax(last, dim=-1)
    top_p, top_id = probs.max(dim=-1)
    top1 = tok.decode([int(top_id)])
    print(f"      forward logits: shape={tuple(logits.shape)} "
          f"finite={logits_finite} nonzero={nonzero}")
    print(f"      argmax next-token id={int(top_id)} p={float(top_p):.4f} "
          f"decoded={top1!r}")

    # ---- short greedy generation from memory (manual loop) -----------------
    # NOTE: we do NOT use model.generate(): under transformers 5.5.4 its new
    # _prefill flow calls prepare_inputs_for_generation(cache_position=None),
    # which the ported (tf-4.43-era) prepare_inputs_for_generation does not
    # handle. That is a generate()-API plumbing gap, orthogonal to the memory
    # mechanism, and fixing it would mean editing the (read-only) port file.
    # A manual greedy loop of full forwards re-reads the injected memory pool at
    # every step and is the cleaner smoke of the read path. O(n^2) but n is tiny.
    gen_text = None
    if device.type != "cpu":
        cur = q_ids
        gen_ids = []
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                lg = model(input_ids=cur, return_dict=True).logits[0, -1]
                nxt = int(lg.float().argmax().item())
                gen_ids.append(nxt)
                if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                    break
                cur = torch.cat(
                    [cur, torch.tensor([[nxt]], device=device)], dim=1)
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        print(f"      greedy generate({args.max_new_tokens} new): {gen_text!r}")

    ok = (shape_ok and mem_finite and delta_finite and logits_finite
          and nonzero and float(top_p) < 0.999)
    print("-" * 72)
    tag = where + (" verified" if device.type == "cpu" else "")
    print(f"SMOKE: {'PASS' if ok else 'FAIL'}  [{tag}]")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
