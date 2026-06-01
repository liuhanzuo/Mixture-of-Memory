#!/usr/bin/env python3
"""Toy-task memory-bootstrap diagnostic — decisively separate "memory training
loop is broken" from "Dolmino CPT objective is too weak".

WHY THIS SCRIPT EXISTS
----------------------
On Dolmino CPT the memory routing collapses to uniform (top1_sim ≈ 1/N). The
upgraded hypothesis is a *content-addressable memory bootstrap failure*:

    random routing → random writes → slots become generic averages →
    inject gate (alpha) closes → weak gradient → routing learns even less →
    death spiral.

To tell whether the **loop itself** is broken vs. the **CPT objective is too
weak**, we build the smallest possible task where memory is the ONLY way to
solve it, and watch whether the loop can bootstrap:

    * If even this toy task cannot be learned  → the training loop is broken
      (read/write/route/gradient plumbing). Fix the loop.
    * If the toy task IS learned but Dolmino is not → the loop works; the
      next-token-prediction objective on generic web text simply does not
      *reward* successful retrieval strongly enough. Add an aux retrieval
      objective.

TOY TASK (synthetic passcode retrieval)
---------------------------------------
Each sample is a forced 2-chunk key-value memory probe:

    chunk 1 (context, WRITE):  "The passcode is 7392."      (random 4 digits)
    chunk 2 (target,  READ) :  "The passcode is" → 7392

The answer digits appear ONLY in chunk 1. Chunks are fed to the model in
separate forward passes (streamed), so the vanilla attention path in chunk 2
literally cannot see the digits — the model MUST carry the fact through the
memory bank to solve it. LM loss is computed ONLY on the answer tokens in
chunk 2 (the "The passcode is" prompt portion is label-masked to -100), so the
loss directly rewards a successful retrieval.

We reuse the EXACT memory machinery from
``scripts/train_mem_space_dolmino_cpt.py`` (same ``build_model`` patch flow,
same ``_reset_banks`` / ``_detach_banks``, the same MemorySpaceLayer forward
and its QUERY_DIAG / WRITEBACK_DIAG prints). The cross-chunk credit assignment
mirrors ``dolmino_train_step_tbptt`` with ``bptt_window=2`` (the whole 2-chunk
sample lives in ONE autograd graph, so the chunk-1 writer receives gradient
that says "you wrote a slot that helped chunk 2's loss").

THREE DIAGNOSTIC SIGNALS (printed as TOY_DIAG lines)
----------------------------------------------------
1. selected-slot norm delta (chunk-1 write):
   norm of the chunk-1-selected slots AFTER the write minus their value at
   init. If ≈ 0, the writer is not actually depositing the fact (it is just
   drifting toward the generic mean). Healthy = clearly positive & growing.
2. chunk-2 routing concentration:
   * top1_sim (vs the uniform floor 1/num_slots) — is routing non-uniform?
   * chunk1→chunk2 slot overlap — does chunk 2 route BACK to the very slots
     chunk 1 wrote? This is the crux of content addressing.
3. retrieval accuracy (THE gold standard):
   exact-match accuracy of chunk-2's greedy argmax on the answer digits over a
   fixed eval batch. If this climbs above chance, the loop bootstraps.

Plus inject_gate alpha (mean) — does it rise from its ~0.12/0.46 init, i.e. is
the model learning to actually *use* the retrieved memory?

INTERPRETATION
--------------
    toy acc stays at chance + alpha flat + slot delta ≈ 0  → LOOP IS BROKEN.
    toy acc climbs (loop bootstraps) but Dolmino still uniform → CPT OBJECTIVE
        IS TOO WEAK → add an auxiliary retrieval objective.

--force_gate_alpha (bootstrap-breaker probe)
--------------------------------------------
A chicken-and-egg sub-hypothesis: alpha (inject gate) starts low, so the memory
read barely influences the output, so the LM loss barely rewards routing, so
alpha never has a reason to open. ``--force_gate_alpha 0.5`` clamps the inject
gate to a fixed value for the first ``--force_gate_steps`` steps (then releases
it to learn freely), forcing strong memory exposure to test whether that breaks
the death spiral. This is implemented fully (NOT a stub): the inject gate is a
``nn.Linear(d_model, 1)`` whose output passes through a sigmoid; we set its
weight to 0 and its bias to ``logit(force_gate_alpha)`` so the gate emits the
forced constant, then restore the original weights after the forced window.

This is a DIAGNOSTIC tool, not an architecture version — no versions/ file.
Single-GPU by design (the toy forward is cheap; simplicity over scale).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Reuse the canonical patch / bank / step machinery — do NOT reimplement the
# memory forward.
import train_mem_space_dolmino_cpt as T  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy memory-bootstrap diagnostic")
    p.add_argument("--model_path", type=str, default="models/Meta-Llama-3-8B")
    p.add_argument("--output_dir", type=str, default="outputs/toy_memory_bootstrap")
    p.add_argument("--total_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_slots", type=int, default=128)
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--routing_pool_mode", type=str, default="multi_query",
                   choices=["max_pool", "chunk_query", "multi_query", "slot_query"])
    p.add_argument("--selector_temperature", type=float, default=20.0)
    p.add_argument("--l_recon_weight", type=float, default=0.0,
                   help="P1/v12 summary-reconstruction aux loss weight. >0 "
                        "enables the MemoryReconDecoder (requires L3 summary, "
                        "which the toy always sets). 0 = disabled (default).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory",
                   help="Set to '' to disable wandb.")
    p.add_argument("--diag_interval", type=int, default=10,
                   help="Print a TOY_DIAG line every N steps.")
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--eval_batch_size", type=int, default=16,
                   help="Fixed eval batch used for the retrieval-acc gold metric.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    # bootstrap-breaker probe (fully implemented)
    p.add_argument("--force_gate_alpha", type=float, default=None,
                   help="If set (e.g. 0.5), clamp the inject gate to this value "
                        "for the first --force_gate_steps steps, then release. "
                        "Tests whether forced memory exposure breaks the "
                        "bootstrap death spiral.")
    p.add_argument("--force_gate_steps", type=int, default=200,
                   help="Number of initial steps to hold the forced gate value.")
    return p.parse_args()


def build_base_args(args: argparse.Namespace) -> argparse.Namespace:
    """Construct the train-script arg namespace (with all defaults) so that
    ``T.build_model`` sees exactly the fields it expects, then override the
    toy-relevant knobs. Using the train parser keeps us in sync with its
    defaults automatically."""
    _old_argv = sys.argv
    sys.argv = ["toy", "--output_dir", args.output_dir]
    try:
        base = T.parse_args()
    finally:
        sys.argv = _old_argv

    base.model_path = args.model_path
    base.num_slots = args.num_slots
    base.top_k = args.top_k
    base.selector_temperature = args.selector_temperature
    base.routing_pool_mode = args.routing_pool_mode
    base.l_recon_weight = args.l_recon_weight
    base.attn_impl = args.attn_impl
    base.dtype = args.dtype
    base.seed = args.seed
    base.init_checkpoint = None          # fresh, untrained memory — we WANT to
    base.init_adapter_config = None      # test bootstrap from scratch.
    base.gradient_checkpointing = False  # toy seqs are tiny; not needed.
    base.use_l3_summary = True           # required for multi_query sub-queries.
    base.writeback_warmup_steps = 0      # let writeback act from step 0.
    return base


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


class ToyPasscodeData:
    """Generates batches of the 2-chunk passcode retrieval task.

    chunk 1 (context, write):  "The passcode is <NNNN>."
    chunk 2 (target,  read) :  "The passcode is <NNNN>"   (loss on <NNNN> only)
    """

    PROMPT = "The passcode is"

    def __init__(self, tokenizer, rng: random.Random, device: torch.device):
        self.tok = tokenizer
        self.rng = rng
        self.device = device
        self.pad_id = tokenizer.pad_token_id
        # token length of the prompt-only prefix (used to mask labels). The
        # target string is PROMPT + " " + passcode, so the first len(prompt
        # tokens) positions are the prompt and get label -100.
        self._prompt_ids = tokenizer.encode(self.PROMPT, add_special_tokens=True)
        self._prompt_len = len(self._prompt_ids)

    def _random_passcode(self) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(4))

    def _pad_stack(self, seqs: List[List[int]], pad_id: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def make_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        ctx_seqs: List[List[int]] = []
        tgt_seqs: List[List[int]] = []
        lbl_seqs: List[List[int]] = []
        for _ in range(batch_size):
            pc = self._random_passcode()
            ctx_text = f"{self.PROMPT} {pc}."
            tgt_text = f"{self.PROMPT} {pc}"
            ctx_ids = self.tok.encode(ctx_text, add_special_tokens=True)
            tgt_ids = self.tok.encode(tgt_text, add_special_tokens=True)
            labels = list(tgt_ids)
            # Mask the prompt prefix; only the answer (passcode) tokens count.
            for j in range(min(self._prompt_len, len(labels))):
                labels[j] = -100
            ctx_seqs.append(ctx_ids)
            tgt_seqs.append(tgt_ids)
            lbl_seqs.append(labels)

        ctx = self._pad_stack(ctx_seqs, self.pad_id).to(self.device)
        tgt = self._pad_stack(tgt_seqs, self.pad_id).to(self.device)
        # pad label positions get -100 (no loss / not counted as answer).
        lbl = self._pad_stack(lbl_seqs, -100).to(self.device)
        return {"ctx": ctx, "tgt": tgt, "labels": lbl}


# --------------------------------------------------------------------------- #
# force-gate plumbing
# --------------------------------------------------------------------------- #


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class ForceGate:
    """Clamp every MemorySpaceLayer's inject_gate to emit a fixed alpha.

    The inject gate computes ``g = sigmoid(inject_gate(hidden))``. Setting the
    Linear weight to 0 and the bias to ``logit(alpha)`` makes the gate emit the
    constant ``alpha`` for all tokens, bypassing the chicken-and-egg deadlock.
    ``release()`` restores the original learned weights so the gate can train.
    """

    def __init__(self, mem_layers, alpha: float):
        self.mem_layers = mem_layers
        self.alpha = alpha
        self._saved: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._engaged = False

    def engage(self) -> None:
        if self._engaged:
            return
        self._saved = []
        for w in self.mem_layers:
            ig = w.inject_gate
            self._saved.append((ig.weight.detach().clone(), ig.bias.detach().clone()))
            with torch.no_grad():
                ig.weight.zero_()
                ig.bias.fill_(_logit(self.alpha))
        self._engaged = True

    def reassert(self) -> None:
        """Re-apply the clamp after an optimizer step (the step may have nudged
        the gate params). Cheap; called once per training step while forced."""
        if not self._engaged:
            return
        with torch.no_grad():
            for w in self.mem_layers:
                w.inject_gate.weight.zero_()
                w.inject_gate.bias.fill_(_logit(self.alpha))

    def release(self) -> None:
        if not self._engaged:
            return
        with torch.no_grad():
            for (sw, sb), w in zip(self._saved, self.mem_layers):
                w.inject_gate.weight.copy_(sw)
                w.inject_gate.bias.copy_(sb)
        self._engaged = False


# --------------------------------------------------------------------------- #
# Training step (toy 2-chunk windowed BPTT, mirrors dolmino_train_step_tbptt)
# --------------------------------------------------------------------------- #


def toy_train_step(model, ctx: torch.Tensor, tgt: torch.Tensor,
                   labels: torch.Tensor, device: torch.device):
    """One toy step with bptt_window=2 (whole sample = one autograd graph).

    chunk 1 forward (grad ON, no loss) writes the fact into memory; chunk 2
    forward reads it and computes loss ONLY on the answer tokens. We keep both
    forwards connected in one graph, backward once, then detach — exactly the
    cross-chunk credit-assignment ``dolmino_train_step_tbptt`` provides at
    ``bptt_window=2``, so the chunk-1 writer gets the "you helped chunk 2"
    gradient.
    """
    T._reset_banks(model)

    # chunk 1: write the fact (gradient-bearing, no LM loss on context).
    model(input_ids=ctx, use_cache=False)

    # chunk 2: read + loss on answer tokens only (labels mask the prompt).
    out = model(input_ids=tgt, labels=labels, use_cache=False)
    lm_loss = out.loss
    aux_loss = T._collect_aux_loss(model, device)

    if lm_loss is None or not torch.isfinite(lm_loss + aux_loss):
        return None, lm_loss, aux_loss

    total = lm_loss + aux_loss
    total.backward()
    T._detach_banks(model)
    return total, lm_loss, aux_loss


# --------------------------------------------------------------------------- #
# Diagnostics (no-grad; runs on a fixed eval batch)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def toy_diag(model, mem_layers, batch: Dict[str, torch.Tensor],
             device: torch.device, num_slots: int,
             forced_alpha) -> Dict[str, float]:
    ctx = batch["ctx"]
    tgt = batch["tgt"]
    labels = batch["labels"]
    bank = getattr(model, "_mem_space_shared_bank", None)
    mem0 = mem_layers[0]
    B = ctx.shape[0]

    # ---- pass A: frozen write (measures selected-slot norm at INIT) ---- #
    if bank is not None:
        bank.frozen = True
    T._reset_banks(model)
    model(input_ids=ctx, use_cache=False)
    idx_init = mem0.last_idx          # [B, k]
    init_sel_norm = float("nan")
    if bank is not None and bank.slots is not None and idx_init is not None:
        sd = bank.slots.shape[-1]
        gi = idx_init.unsqueeze(-1).expand(-1, -1, sd)
        init_sel = bank.slots.gather(1, gi)            # [B, k, d]
        init_sel_norm = init_sel.float().norm(dim=-1).mean().item()
    if bank is not None:
        bank.frozen = False

    # ---- pass B: real write (measures selected-slot norm AFTER write) ---- #
    T._reset_banks(model)
    model(input_ids=ctx, use_cache=False)
    idx1 = mem0.last_idx              # [B, k] chunk-1 selected slots
    post_sel_norm = float("nan")
    if bank is not None and bank.slots is not None and idx1 is not None:
        sd = bank.slots.shape[-1]
        gi = idx1.unsqueeze(-1).expand(-1, -1, sd)
        post_sel = bank.slots.gather(1, gi)
        post_sel_norm = post_sel.float().norm(dim=-1).mean().item()
    slot_norm_delta = (post_sel_norm - init_sel_norm
                       if not (math.isnan(post_sel_norm) or math.isnan(init_sel_norm))
                       else float("nan"))

    # ---- chunk 2 forward: routing + retrieval acc (memory from pass B) ---- #
    # Capture the inject-gate alpha via a pre-hook on layer 0 (the layer
    # computes the gate inline via F.linear, so we recompute it from the
    # captured hidden states the same way layer.py does).
    captured: Dict[str, torch.Tensor] = {}

    def _pre_hook(_mod, _args):
        captured["h"] = _args[0].detach()

    handle = mem0.register_forward_pre_hook(_pre_hook)
    try:
        out2 = model(input_ids=tgt, use_cache=False)
    finally:
        handle.remove()

    idx2 = mem0.last_idx              # [B, k] chunk-2 selected slots
    top1_sim = float(getattr(mem0, "_last_top1_sim", 0.0))

    # chunk1 → chunk2 slot overlap (did chunk 2 route back to the written slots?)
    overlap = float("nan")
    if idx1 is not None and idx2 is not None:
        ov = 0.0
        for b in range(B):
            s1 = set(idx1[b].tolist())
            s2 = set(idx2[b].tolist())
            ov += len(s1 & s2) / max(1, len(s2))
        overlap = ov / B

    # inject-gate alpha (mean), reconstructed exactly as layer.py does.
    alpha_mean = float("nan")
    if "h" in captured:
        ig = mem0.inject_gate
        hsf = captured["h"].float()
        glog = F.linear(hsf, ig.weight.float(), ig.bias.float())
        alpha_mean = torch.sigmoid(glog).mean().item()

    # retrieval accuracy on the answer tokens (teacher-forced argmax).
    logits = out2.logits                          # [B, T, V]
    pred = logits[:, :-1, :].argmax(dim=-1)        # token predicted at pos+1
    gold = tgt[:, 1:]
    ans_mask = (labels[:, 1:] != -100)             # answer-token positions
    tok_correct = (pred == gold) & ans_mask
    n_ans = ans_mask.sum(dim=1).clamp(min=1)
    sample_exact = (tok_correct.sum(dim=1) == ans_mask.sum(dim=1)) & (ans_mask.sum(dim=1) > 0)
    exact_acc = sample_exact.float().mean().item()
    tok_acc = (tok_correct.sum().float() / ans_mask.sum().clamp(min=1).float()).item()

    return {
        "slot_init_sel_norm": init_sel_norm,
        "slot_post_sel_norm": post_sel_norm,
        "slot_norm_delta": slot_norm_delta,
        "top1_sim": top1_sim,
        "uniform_floor": 1.0 / num_slots,
        "chunk1to2_overlap": overlap,
        "alpha_mean": alpha_mean,
        "retrieval_exact_acc": exact_acc,
        "retrieval_tok_acc": tok_acc,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    base = build_base_args(args)

    # wandb (optional)
    use_wandb = _WANDB_AVAILABLE and bool(args.wandb_project)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or os.path.basename(args.output_dir),
            config={**vars(args)},
            dir=args.output_dir,
        )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model (reuse the canonical patch flow)
    print(f"[toy] building model from {args.model_path} "
          f"(routing={args.routing_pool_mode}, num_slots={args.num_slots}, "
          f"top_k={args.top_k}, temp={args.selector_temperature})", flush=True)
    model = T.build_model(base, device, dtype)
    T._freeze_backbone(model)
    mem_layers = model._mem_space_layers
    n_trainable = sum(p.numel() for p in T._mem_space_params(model))
    print(f"[toy] mem_space trainable params: {n_trainable/1e6:.2f}M "
          f"across {len(mem_layers)} layers", flush=True)

    # data
    data_rng = random.Random(args.seed)
    data = ToyPasscodeData(tokenizer, data_rng, device)
    # fixed eval batch (seeded separately so it is stable across the run)
    eval_rng = random.Random(args.seed + 9999)
    eval_data = ToyPasscodeData(tokenizer, eval_rng, device)
    eval_batch = eval_data.make_batch(args.eval_batch_size)

    # one-time label-mask confirmation
    _lb0 = eval_batch["labels"][0]
    _tg0 = eval_batch["tgt"][0]
    _nonmask = (_lb0 != -100).nonzero(as_tuple=True)[0].tolist()
    print(f"[toy] LABEL_MASK_CHECK sample0 tgt_tokens={_tg0.tolist()}", flush=True)
    print(f"[toy] LABEL_MASK_CHECK sample0 answer(non -100) positions={_nonmask} "
          f"-> ids={[_tg0[i].item() for i in _nonmask]} "
          f"decoded={tokenizer.decode([_tg0[i].item() for i in _nonmask])!r}", flush=True)

    # optimizer
    trainable = T._mem_space_params(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0,
                                  betas=(0.9, 0.95))

    # force-gate probe
    force_gate = None
    if args.force_gate_alpha is not None:
        force_gate = ForceGate(mem_layers, args.force_gate_alpha)
        force_gate.engage()
        print(f"[toy] FORCE_GATE engaged: alpha={args.force_gate_alpha} "
              f"for first {args.force_gate_steps} steps", flush=True)

    model.train()
    t0 = time.time()
    n_nonfinite = 0

    for step in range(args.total_steps):
        # release the forced gate once the window closes.
        if force_gate is not None and force_gate._engaged and step >= args.force_gate_steps:
            force_gate.release()
            print(f"[toy] FORCE_GATE released at step {step}", flush=True)

        batch = data.make_batch(args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        total, lm_loss, aux_loss = toy_train_step(
            model, batch["ctx"], batch["tgt"], batch["labels"], device
        )
        if total is None:
            n_nonfinite += 1
        else:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
        T._step_counters_inc(model)

        # keep the gate clamped during the forced window (optimizer may nudge).
        if force_gate is not None and force_gate._engaged:
            force_gate.reassert()

        if step % args.log_interval == 0:
            lm_v = lm_loss.item() if lm_loss is not None else float("nan")
            aux_v = aux_loss.item() if aux_loss is not None else float("nan")
            # P1/v12: surface the recon aux component (layer-0 singleton) so the
            # smoke test can confirm it is finite and trending down.
            recon_v = float("nan")
            _rc = mem_layers[0].last_aux_losses.get("recon")
            if _rc is not None:
                recon_v = _rc.item()
            sps = (step + 1) / max(1e-9, time.time() - t0)
            print(f"[toy step {step}/{args.total_steps}] lm={lm_v:.4f} "
                  f"aux={aux_v:.4f} recon={recon_v:.4f} nf={n_nonfinite} "
                  f"speed={sps:.2f} it/s",
                  flush=True)
            if use_wandb:
                wandb.log({"train/lm_loss": lm_v, "train/aux_loss": aux_v,
                           "train/recon_loss": recon_v,
                           "train/n_nonfinite": n_nonfinite}, step=step)

        if step % args.diag_interval == 0 or step == args.total_steps - 1:
            model.eval()
            d = toy_diag(model, mem_layers, eval_batch, device,
                         args.num_slots, args.force_gate_alpha)
            model.train()
            if force_gate is not None and force_gate._engaged:
                force_gate.reassert()
            print(
                f"[TOY_DIAG step={step}]"
                f" retrieval_exact_acc={d['retrieval_exact_acc']:.3f}"
                f" retrieval_tok_acc={d['retrieval_tok_acc']:.3f}"
                f" top1_sim={d['top1_sim']:.6f}(floor={d['uniform_floor']:.6f})"
                f" chunk1to2_overlap={d['chunk1to2_overlap']:.3f}"
                f" alpha_mean={d['alpha_mean']:.4f}"
                f" slot_norm_delta={d['slot_norm_delta']:.4f}"
                f" (init={d['slot_init_sel_norm']:.4f}->post={d['slot_post_sel_norm']:.4f})",
                flush=True,
            )
            if use_wandb:
                wandb.log({f"diag/{k}": v for k, v in d.items()}, step=step)

    # final diag already covered by step == total_steps-1 above.
    print(f"[toy] done: {args.total_steps} steps, non-finite={n_nonfinite}, "
          f"time={(time.time()-t0)/60:.1f} min", flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
