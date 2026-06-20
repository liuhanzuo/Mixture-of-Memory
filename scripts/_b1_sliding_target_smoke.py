"""B1 sliding-target-loss CPU/GPU smoke (2026-06-20, landmark-repro).

Verifies the --sliding_target_loss change is NOT a dead path:
  (1) the repartition logic produces VARIED (context_len, target) splits over
      many draws (not stuck at last-chunk);
  (2) on a tiny multi-layer rawkv_readout model, running dolmino_train_step with
      an EARLY-chunk target (j < n_ctx) still drives NON-ZERO gradient into the
      readout-layer params (the reader's k/v_proj at the readout layers) AND the
      shared GistReadout projections — i.e. the reader trains to consume cross-
      chunk memory at that distance, which is the whole point of B1.

Tiny random Llama (8 layers), readout at layers [4,5,6,7]. Single process.
"""
from __future__ import annotations
import sys, random
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402
from src.memory.mem_space.config import MemorySpaceConfig  # noqa: E402
from src.memory.mem_space.layer import MemorySpaceLayer  # noqa: E402
from src.memory.mem_space.patch import apply_mem_space_to_model  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def build():
    torch.manual_seed(0)
    MemorySpaceLayer._instance_counter = 0
    cfg = LlamaConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(cfg)
    mem_cfg = MemorySpaceConfig(
        num_slots=32, top_k=8, slot_dim=None, swa_window=0,
        use_rawkv_readout=True,
        rawkv_readout_layer=4,
        rawkv_readout_layers=[4, 5, 6, 7],
        rawkv_gist_dim=64,
        rawkv_readout_topk_chunks=2,
        rawkv_readout_temp=1.0,
        rawkv_gist_pool="max",
    )
    apply_mem_space_to_model(model, mem_cfg, layer_indices=None)
    return model.to(DEVICE).to(DTYPE), mem_cfg


def repartition(all_chunks):
    """Mirror the train-loop sliding-target logic exactly."""
    if len(all_chunks) >= 2:
        j = random.randint(1, len(all_chunks) - 1)
        return all_chunks[:j], all_chunks[j], j
    return all_chunks[:-1], all_chunks[-1], len(all_chunks) - 1


def main():
    print(f"[B1smoke] device={DEVICE}")

    # ---- (1) repartition varies ----
    random.seed(0)
    n_ctx_plus1 = 6  # 5 context + 1 target
    js = [repartition(list(range(n_ctx_plus1)))[2] for _ in range(200)]
    uniq = sorted(set(js))
    print(f"[B1smoke] repartition split points over 200 draws: {uniq} "
          f"(min={min(js)} max={max(js)})")
    assert len(uniq) >= 3 and min(js) >= 1 and max(js) <= n_ctx_plus1 - 1, \
        "repartition not varying across the chunk range"
    early = sum(1 for j in js if j < n_ctx_plus1 - 1)
    print(f"[B1smoke] {early}/200 draws target an EARLY (non-last) chunk "
          f"({100*early/200:.0f}%) — these are the new cross-distance training signals")
    assert early > 0, "sliding never picks an early target — would be a no-op"

    # ---- (2) early-target step drives readout-path gradient ----
    from scripts.train_mem_space_dolmino_cpt import dolmino_train_step
    model, mem_cfg = build()
    model.train()
    bank = getattr(model, "_mem_space_shared_bank", None)
    if bank is not None:
        bank.reset(1)
    T = 24
    all_chunks = [torch.randint(0, 512, (T,), device=DEVICE) for _ in range(6)]
    # force an EARLY target (j=2): chunks[0:2] -> memory, chunk 2 -> grad target
    ctx, tgt, j = all_chunks[:2], all_chunks[2], 2
    print(f"[B1smoke] forcing early target j={j}: {len(ctx)} context chunks -> chunk {j}")
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    lm_loss, aux_loss, *_ = dolmino_train_step(
        model, ctx, tgt, DEVICE, grad_accum=1,
    )
    print(f"[B1smoke] early-target step: lm_loss={float(lm_loss):.4f} "
          f"finite={torch.isfinite(lm_loss).item()}")
    assert torch.isfinite(lm_loss).item(), "loss not finite"

    # collect grad on readout-layer reader k/v_proj + gist proj
    ro_layers = set(mem_cfg.rawkv_readout_layers)
    gist_grad = 0.0
    readout_kv_grad = 0.0
    n_readout_params = 0
    root = getattr(model, "module", model)
    for ml in (getattr(root, "_mem_space_layers", None) or []):
        li = getattr(ml, "_layer_idx", None)
        if li in ro_layers:
            attn = getattr(getattr(ml, "wrapped_layer", ml), "self_attn", None)
            for nm in ("k_proj", "v_proj"):
                proj = getattr(attn, nm, None)
                if proj is not None and proj.weight.grad is not None:
                    readout_kv_grad += float(proj.weight.grad.norm())
                    n_readout_params += 1
        gist = getattr(ml, "gist_readout", None)
        if gist is not None:
            for pn, p in gist.named_parameters():
                if p.grad is not None:
                    gist_grad += float(p.grad.norm())
    print(f"[B1smoke] readout-layer k/v_proj grad-norm sum={readout_kv_grad:.4e} "
          f"over {n_readout_params} proj tensors")
    print(f"[B1smoke] GistReadout proj grad-norm sum={gist_grad:.4e}")
    assert n_readout_params > 0, "no readout-layer k/v_proj found — wiring wrong"
    assert readout_kv_grad > 0, "readout-layer reader got ZERO grad on early target = dead path"
    assert gist_grad > 0, "gist scorer got ZERO grad = dead selection"

    print("\n[B1smoke] VERDICT: PASS — sliding repartition varies + early-chunk "
          "target drives non-zero readout & gist gradient (not a dead path).")


if __name__ == "__main__":
    main()
