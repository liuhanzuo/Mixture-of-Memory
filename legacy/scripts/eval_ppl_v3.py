"""PPL evaluation for RMT v3 (LoRA + RMT memory).
Replicates the exact forward pass from train_rmt_v3.py for accurate eval.
Usage: python scripts/eval_ppl_v3.py <checkpoint_dir> [--heldout <data_path>] [--max_docs 50] [--skip_base]
"""
import torch, json, sys, math, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.memory.rmt.rmt_module import RMTMemory, build_rmt_attention_mask, build_rmt_position_ids
import torch.nn as nn


def load_model(ckpt, device):
    """Load LoRA + RMT model, matching train_rmt_v3.py's approach."""
    cfg_path = os.path.join(os.path.dirname(ckpt.rstrip("/")), "rmt_config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ckpt, "rmt_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    num_mem = cfg["num_memory_tokens"]
    seg_len = cfg["segment_length"]
    max_seg = cfg["max_segments"]
    bottleneck_dim = cfg.get("bottleneck_dim", 32)
    num_mem_heads = cfg.get("num_memory_heads", 8)
    extractor_version = cfg.get("extractor_version", 2)
    use_reconstruction = cfg.get("use_reconstruction", False)
    base_model_path = cfg.get("model_path", "../models/Qwen--Qwen3-8b")

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    # Load base model + LoRA (same as training)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    base_model.eval()
    lora_model = PeftModel.from_pretrained(base_model, ckpt)
    lora_model.eval()

    # Get internal references (same as training)
    inner_model = lora_model.get_base_model()
    backbone = inner_model.model  # Qwen3Model

    # Load RMT memory module
    hidden_dim = lora_model.config.hidden_size
    rmt_mem = RMTMemory(
        hidden_dim, num_mem, num_mem_heads, max_seg + 1,
        bottleneck_dim=bottleneck_dim,
        extractor_version=extractor_version,
        use_reconstruction=use_reconstruction,
    ).to(device=device, dtype=torch.bfloat16)
    rmt_mem.load_state_dict(torch.load(os.path.join(ckpt, "rmt_memory.pt"), map_location=device, weights_only=False))
    rmt_mem.eval()

    print(f"Config: mem={num_mem}, seg_len={seg_len}, max_seg={max_seg}, bottleneck={bottleneck_dim}")
    return tok, lora_model, inner_model, backbone, rmt_mem, cfg


def eval_doc_rmt(tok, inner_model, backbone, rmt_mem, tokens, device, cfg):
    """Eval a single document with RMT + LoRA, replicating training forward."""
    num_mem = cfg["num_memory_tokens"]
    seg_len = cfg["segment_length"]
    max_seg = cfg["max_segments"]

    max_tokens = seg_len * max_seg
    tokens = tokens[:max_tokens]
    num_segments = (len(tokens) + seg_len - 1) // seg_len

    input_ids = torch.tensor([tokens], device=device)
    B = 1

    total_loss = 0.0
    total_tokens = 0

    old_memory = None
    with torch.no_grad():
        for seg_idx in range(num_segments):
            start = seg_idx * seg_len
            end = min(start + seg_len, len(tokens))
            seg_ids = input_ids[:, start:end]

            # Get memory
            if old_memory is None:
                mem = rmt_mem.get_initial_memory(seg_idx, B, device, torch.bfloat16)
            else:
                mem = old_memory

            # Embed with memory
            token_embeds = inner_model.get_input_embeddings()(seg_ids)
            inputs_embeds = torch.cat([mem, token_embeds], dim=1)

            # Attention mask
            attn_mask = build_rmt_attention_mask(seg_len, num_mem, device)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

            # Position IDs
            position_ids = build_rmt_position_ids(seg_len, num_mem, seg_idx, device).unsqueeze(0).expand(B, -1)

            # Labels
            seg_len_actual = seg_ids.shape[1]
            mem_labels = torch.full((B, num_mem), -100, device=device, dtype=torch.long)
            full_labels = torch.cat([mem_labels, seg_ids], dim=1)

            # Forward (same as training)
            outputs = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask={"full_attention": attn_mask},
                position_ids=position_ids,
            )
            hidden = outputs.last_hidden_state
            logits = inner_model.lm_head(hidden)

            # Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss.item() * (seg_len_actual - 1)
            total_tokens += seg_len_actual - 1

            # Extract memory for next segment
            seg_hidden = hidden[:, num_mem:, :]
            mem_result = rmt_mem.extract_memory(seg_hidden, old_memory)
            # V3 returns (new_memory, recon_loss), V2 returns new_memory
            if isinstance(mem_result, tuple):
                old_memory = mem_result[0]
            else:
                old_memory = mem_result

    return total_loss, total_tokens


def eval_doc_base(model, tokens, device, max_len):
    """Eval a single document with base model (no RMT, no LoRA)."""
    tokens = tokens[:max_len]
    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()
    n = len(tokens) - 1
    return loss * n, n


def eval_doc_lora(lora_model, tokens, device, max_len):
    """Eval a single document with LoRA only (no RMT)."""
    tokens = tokens[:max_len]
    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        outputs = lora_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()
    n = len(tokens) - 1
    return loss * n, n


def run_eval(tok, data_path, max_docs, device, cfg, eval_fn, label=""):
    """Run eval on a dataset with given eval function."""
    seg_len = cfg["segment_length"]
    max_seg = cfg["max_segments"]

    docs = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            docs.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0
    skipped = 0

    for i, doc in enumerate(docs):
        tokens = tok.encode(doc, add_special_tokens=False)
        if len(tokens) < seg_len:
            skipped += 1
            continue
        loss, n_tok = eval_fn(tokens)
        total_loss += loss
        total_tokens += n_tok
        if (i + 1) % 10 == 0:
            avg_ppl = math.exp(total_loss / total_tokens)
            print(f"  [{label} {i+1}/{min(max_docs, len(docs))}] ppl={avg_ppl:.4f}")

    if total_tokens == 0:
        return None, 0, 0
    ppl = math.exp(total_loss / total_tokens)
    print(f"  {label}: PPL={ppl:.4f} (docs={len(docs)-skipped}, skipped={skipped})")
    return ppl, len(docs) - skipped, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint dir (e.g. outputs/rmt_v3_.../final)")
    parser.add_argument("--heldout", default=None, help="Held-out data path")
    parser.add_argument("--train_data", default="data/rmt_train_mixed.jsonl", help="Training data")
    parser.add_argument("--max_docs", type=int, default=50, help="Max docs to eval")
    parser.add_argument("--skip_base", action="store_true", help="Skip base model eval")
    args = parser.parse_args()

    device = "cuda"
    print("=" * 60)
    print("RMT v3 (LoRA) PPL Evaluation")
    print("=" * 60)

    tok, lora_model, inner_model, backbone, rmt_mem, cfg = load_model(args.checkpoint, device)

    seg_len = cfg["segment_length"]
    max_seg = cfg["max_segments"]
    max_len = seg_len * max_seg

    results = []

    # 1. Training set
    print(f"\n--- Training Set ({args.train_data}, {args.max_docs} docs) ---")

    rmt_ppl, nd, nt = run_eval(tok, args.train_data, args.max_docs, device, cfg,
        lambda tokens: eval_doc_rmt(tok, inner_model, backbone, rmt_mem, tokens, device, cfg),
        "RMT+LoRA")
    results.append(("Train-RMT+LoRA", rmt_ppl, nd, nt))

    if not args.skip_base:
        # Need to load base model separately for clean comparison
        from transformers import AutoModelForCausalLM as AM
        base_model = AM.from_pretrained(
            cfg.get("model_path", "../models/Qwen--Qwen3-8b"),
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        base_model.eval()

        base_ppl, _, _ = run_eval(tok, args.train_data, args.max_docs, device, cfg,
            lambda tokens: eval_doc_base(base_model, tokens, device, max_len),
            "Base")
        results.append(("Train-Base", base_ppl, nd, nt))

        lora_ppl, _, _ = run_eval(tok, args.train_data, args.max_docs, device, cfg,
            lambda tokens: eval_doc_lora(lora_model, tokens, device, max_len),
            "LoRA-only")
        results.append(("Train-LoRA-only", lora_ppl, nd, nt))

        del base_model
        torch.cuda.empty_cache()

    # 2. Held-out set
    if args.heldout:
        print(f"\n--- Held-Out Set ({args.heldout}, {args.max_docs} docs) ---")

        rmt_ppl_h, nd_h, nt_h = run_eval(tok, args.heldout, args.max_docs, device, cfg,
            lambda tokens: eval_doc_rmt(tok, inner_model, backbone, rmt_mem, tokens, device, cfg),
            "RMT+LoRA")
        results.append(("Heldout-RMT+LoRA", rmt_ppl_h, nd_h, nt_h))

        if not args.skip_base:
            base_model = AM.from_pretrained(
                cfg.get("model_path", "../models/Qwen--Qwen3-8b"),
                trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}
            )
            base_model.eval()

            base_ppl_h, _, _ = run_eval(tok, args.heldout, args.max_docs, device, cfg,
                lambda tokens: eval_doc_base(base_model, tokens, device, max_len),
                "Base")
            results.append(("Heldout-Base", base_ppl_h, nd_h, nt_h))

            del base_model
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ppl, nd, nt in results:
        print(f"  {name:20s}: PPL={ppl:.4f}" if ppl else f"  {name:20s}: N/A")

    # Compute diffs
    def find(name):
        for r in results:
            if r[0] == name and r[1]:
                return r[1]
        return None

    rmt_train = find("Train-RMT+LoRA")
    base_train = find("Train-Base")
    lora_train = find("Train-LoRA-only")
    rmt_held = find("Heldout-RMT+LoRA")
    base_held = find("Heldout-Base")

    if rmt_train and base_train:
        print(f"\n  Train RMT+LoRA vs Base: {rmt_train - base_train:+.4f} ({(rmt_train/base_train-1)*100:+.2f}%)")
    if lora_train and base_train:
        print(f"  Train LoRA-only vs Base: {lora_train - base_train:+.4f} ({(lora_train/base_train-1)*100:+.2f}%)")
    if rmt_held and base_held:
        print(f"  Heldout RMT+LoRA vs Base: {rmt_held - base_held:+.4f} ({(rmt_held/base_held-1)*100:+.2f}%)")


if __name__ == "__main__":
    main()
