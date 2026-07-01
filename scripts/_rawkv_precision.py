"""Needle-retrieval-precision check for the parallel raw-KV channel (2026-06-18).

The 3-arm RULER result (raw-KV OFF=21 vs topk64=22 vs topk256=22 @4k, n=100)
is at noise — a NULL. But null is CONFOUNDED with the already-falsified EV-prefix
readout interface, so it does NOT prove raw-KV retrieval fails. Before we can even
attribute the null to the readout interface (vs a retrieval-quality failure), we
must rule out the separate failure mode: "the needle token is never even retrieved
into the top-k". This script answers exactly that, with NO bearing on the readout.

Mechanism (mirrors the production W0 streaming path bit-for-bit):
  1. Build a niah_single_1 sample (same RNG/builder as the harness) and tokenize.
  2. Locate the gold needle token span [p0, p0+S) inside the full prompt ids via
     the harness's own _find_subsequence.
  3. Stream chunks[:-1] so the per-sequence raw-KV store fills IN TOKEN ORDER
     (append_rawkv concatenates every real token of every streamed chunk), hence
     store-entry-index m == global token offset m for the streamed region.
  4. At question-time (the first generation forward) the rawkv_layer scores the
     store with the question's routing key and takes top-k. We monkeypatch
     retrieve_rawkv to stash the selected store indices (_top_i).
  5. PRECISION = does {top-k store indices} intersect [p0, p0+S)? i.e. is the
     gold needle among what got retrieved? Report hit-rate over N samples + the
     rank at which the first needle token appears (0 = top scored).

If precision is HIGH but RULER is null  -> bottleneck is the EV-prefix readout
  interface (raw-KV is being retrieved but the frozen decoder can't exploit it
  through this injection path). Architecturally-correct next test: true
  in-attention K/V concat at layer 16 (landmark §4b).
If precision is LOW -> a distinct retrieval-quality failure (routing-key inner
  product can't find the needle) that must be fixed before any readout test.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
    _reset_banks,
    _reset_l2,
    _freeze_banks,
    _unfreeze_banks,
)
from scripts.eval_ruler_mem_space import (  # noqa: E402
    _build_sample,
    _LENGTH_TOKENS,
)
from src.memory.mem_space.memory_bank import MemoryBank  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adapter_config", required=True)
    p.add_argument("--chunk_size", type=int, default=1024)
    p.add_argument("--rawkv_layer", type=int, default=16)
    p.add_argument("--rawkv_topk", type=int, default=64)
    p.add_argument("--length", type=str, default="4k")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size
    mem_config.use_rawkv_retrieval = True
    mem_config.rawkv_layer = args.rawkv_layer
    mem_config.rawkv_topk = args.rawkv_topk
    print(f"[prec] RAW-KV: layer={args.rawkv_layer} topk={args.rawkv_topk} "
          f"chunk={args.chunk_size}")

    model = load_mem_space_model(
        model_path=args.model_path, checkpoint_path=args.checkpoint,
        mem_config=mem_config, device=device, dtype=dtype,
    )

    # --- Monkeypatch retrieve_rawkv to stash the selected store indices. ---
    _orig_retrieve = MemoryBank.retrieve_rawkv

    def _retrieve_capture(self, query_key, topk):
        ret = _orig_retrieve(self, query_key, topk)
        # Re-derive the selected indices the same way the real method does, so
        # we observe EXACTLY what was injected (no behavioural change to ret).
        if ret is not None and self.rawkv_key is not None:
            B, M, _d = self.rawkv_hidden.shape
            if query_key.shape[0] == B and self.rawkv_key.shape[1] == M and M > 0:
                with torch.no_grad():
                    _qk = query_key.to(dtype=torch.float32)
                    _aff = torch.einsum("bqs,bms->bqm", _qk, self.rawkv_key)
                    _score = _aff.max(dim=1).values
                    R = min(int(topk), M)
                    _top_s, _top_i = torch.topk(_score, k=R, dim=1)
                    self._last_rawkv_top_i = _top_i.detach().cpu()  # [B, R]
                    self._last_rawkv_store_M = M
        return ret

    MemoryBank.retrieve_rawkv = _retrieve_capture

    def _get_bank(m):
        root = getattr(m, "module", m)
        return getattr(root, "_mem_space_shared_bank", None)

    target_tokens = _LENGTH_TOKENS[args.length]
    base_seed = args.seed + (hash(("niah_single_1", args.length)) % 100000)

    n_hit = 0
    n_found_span = 0
    n_eval = 0
    first_ranks = []  # rank (0-based) of the first needle token among top-k
    needle_in_last_chunk = 0

    for i in range(args.num_samples):
        rng = random.Random(base_seed * 1000 + i)
        prompt, answers, gold_needle = _build_sample(
            "niah_single_1", target_tokens, tokenizer, rng, None)
        # Char-offset localization (robust to needle-boundary tokenization, which
        # broke the token-subsequence _find_subsequence -> spurious 1-tok match).
        # The scored answer value (answers[0], e.g. the 7-digit magic number) is
        # the token we must retrieve; find its char span, then map to tokens via
        # the fast tokenizer's offset mapping.
        enc = tokenizer(prompt, add_special_tokens=True,
                        return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        offsets = enc["offset_mapping"][0].tolist()  # [(c0,c1), ...] per token
        full = ids[0].tolist()

        if gold_needle is None or not answers:
            continue
        val = str(answers[0])
        # The gold needle sentence appears verbatim once; locate the VALUE inside
        # it (search the gold-needle char span first, then fall back to last
        # occurrence in the prompt — value strings can recur, the needle is the
        # one embedded in the gold sentence).
        gpos = prompt.find(gold_needle)
        if gpos >= 0:
            cval = prompt.find(val, gpos)
        else:
            cval = prompt.rfind(val)
        if cval < 0:
            continue
        c0, c1 = cval, cval + len(val)
        needle_tok = [t for t, (a, b) in enumerate(offsets)
                      if b > a and a < c1 and b > c0]  # tokens overlapping value
        if not needle_tok:
            continue
        n_found_span += 1
        p0, S = needle_tok[0], len(needle_tok)
        needle_set = set(needle_tok)

        # Where does the streamed region end? chunks[:-1] are streamed into the
        # store; the last chunk is the question window (NOT in the store).
        total_len = len(full)
        n_chunks = (total_len + args.chunk_size - 1) // args.chunk_size
        streamed_len = (n_chunks - 1) * args.chunk_size  # tokens in chunks[:-1]
        if p0 >= streamed_len:
            # Needle fell into the last (question) chunk -> never written to the
            # store; retrieval cannot contain it by construction. Track + skip.
            needle_in_last_chunk += 1
            continue

        # --- Stream exactly like generate_with_mem_space, capture retrieval. ---
        _reset_banks(model)
        _reset_l2(model)
        bank = _get_bank(model)
        if hasattr(bank, "_last_rawkv_top_i"):
            del bank._last_rawkv_top_i

        tokens = ids[0]
        chunks = list(tokens.split(args.chunk_size))
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
            if len(chunks) > 1:
                for chunk in chunks[:-1]:
                    _ = model(input_ids=chunk.unsqueeze(0).to(device),
                              use_cache=False)
            _freeze_banks(model)
            try:
                cur = chunks[-1].unsqueeze(0).to(device)
                _ = model(input_ids=cur, use_cache=False)  # question forward
            finally:
                _unfreeze_banks(model)

        top_i = getattr(bank, "_last_rawkv_top_i", None)
        if top_i is None:
            continue
        n_eval += 1
        retrieved = top_i[0].tolist()  # store indices, == global token offsets
        # rank of the first retrieved index that is inside the needle span
        first_rank = None
        for r, m in enumerate(retrieved):
            if m in needle_set:
                first_rank = r
                break
        if first_rank is not None:
            n_hit += 1
            first_ranks.append(first_rank)
        print(f"[prec] sample {i}: needle@[{p0},{p0+S}) store_M="
              f"{getattr(bank,'_last_rawkv_store_M','?')} "
              f"hit={'Y' if first_rank is not None else 'N'} "
              f"first_rank={first_rank}", flush=True)

    print("\n[prec] ===== NEEDLE-RETRIEVAL-PRECISION SUMMARY =====")
    print(f"[prec] length={args.length} topk={args.rawkv_topk} layer={args.rawkv_layer}")
    print(f"[prec] needle span located in prompt: {n_found_span}/{args.num_samples}")
    print(f"[prec] needle fell in last(question) chunk (un-storable): "
          f"{needle_in_last_chunk}")
    print(f"[prec] evaluated (needle in store): {n_eval}")
    if n_eval:
        print(f"[prec] NEEDLE RETRIEVED in top-{args.rawkv_topk}: "
              f"{n_hit}/{n_eval} = {100.0*n_hit/n_eval:.1f}%")
    if first_ranks:
        fr = sorted(first_ranks)
        print(f"[prec] first-needle-token rank among top-k: "
              f"min={fr[0]} median={fr[len(fr)//2]} max={fr[-1]}")


if __name__ == "__main__":
    main()
