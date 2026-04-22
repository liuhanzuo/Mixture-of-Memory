"""Comprehensive PPL evaluation for RMT v2.
Usage: python scripts/eval_ppl_v2.py <checkpoint_dir> [--heldout <data_path>] [--max_docs 50] [--skip_base]
"""
import torch, json, sys, math, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import RMTMemory, RMTModel

def load_rmt(ckpt, device):
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

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    rmt_mem = RMTMemory(hidden_dim, num_mem, num_mem_heads, max_seg + 1, bottleneck_dim=bottleneck_dim).to(device=device, dtype=torch.bfloat16)
    rmt_mem.load_state_dict(torch.load(os.path.join(ckpt, "rmt_memory.pt"), map_location=device, weights_only=False))
    rmt_mem.eval()
    rmt_model = RMTModel(model, rmt_mem, seg_len).to(device=device, dtype=torch.bfloat16)
    rmt_model.eval()

    print(f"Config: mem={num_mem}, seg_len={seg_len}, max_seg={max_seg}, bottleneck={bottleneck_dim}")
    return tok, model, rmt_model, seg_len, max_seg

def eval_ppl(tok, model, rmt_model, data_path, max_docs, seg_len, max_seg, device, use_rmt=True):
    docs = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            docs.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0
    skipped = 0
    losses_per_doc = []

    eval_model = rmt_model if use_rmt else model

    with torch.no_grad():
        for i, doc in enumerate(docs):
            tokens = tok.encode(doc, add_special_tokens=False)
            max_tokens = seg_len * max_seg
            if len(tokens) < seg_len:
                skipped += 1
                continue
            tokens = tokens[:max_tokens]
            input_ids = torch.tensor([tokens], device=device)

            if use_rmt:
                loss_val = rmt_model(input_ids=input_ids, labels=input_ids, training=False)
            else:
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss_val = outputs.loss

            n = len(tokens) - 1
            doc_loss = loss_val.item()
            total_loss += doc_loss * n
            total_tokens += n
            losses_per_doc.append(doc_loss)

            if (i + 1) % 10 == 0:
                avg_so_far = math.exp(total_loss / total_tokens)
                print(f"  [{i+1}/{min(max_docs, len(docs))}] avg_ppl={avg_so_far:.4f} last_loss={doc_loss:.4f}")

    if total_tokens == 0:
        print("  No valid docs (all too short)!")
        return None, 0, 0

    ppl = math.exp(total_loss / total_tokens)
    avg_loss = total_loss / total_tokens
    min_loss = min(losses_per_doc)
    max_loss = max(losses_per_doc)
    print(f"  Docs: {len(losses_per_doc)}, Skipped: {skipped}, Tokens: {total_tokens}")
    return ppl, len(losses_per_doc), total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint directory (e.g. outputs/rmt_v2_.../final)")
    parser.add_argument("--heldout", default=None, help="Held-out data path for generalization test")
    parser.add_argument("--train_data", default="data/rmt_train_wikitext.jsonl", help="Training data path")
    parser.add_argument("--max_docs", type=int, default=50, help="Max docs to eval")
    parser.add_argument("--skip_base", action="store_true", help="Skip base model eval")
    args = parser.parse_args()

    device = "cuda"
    print("=" * 60)
    print("RMT v2 Comprehensive PPL Evaluation")
    print("=" * 60)

    tok, model, rmt_model, seg_len, max_seg = load_rmt(args.checkpoint, device)

    results = []

    # 1. Training set eval
    print(f"\n--- Training Set PPL ({args.train_data}, {args.max_docs} docs) ---")
    rmt_ppl, n_docs, n_tok = eval_ppl(tok, model, rmt_model, args.train_data, args.max_docs, seg_len, max_seg, device, use_rmt=True)
    results.append(("Train-RMT", rmt_ppl, n_docs, n_tok))

    if not args.skip_base:
        base_ppl, _, _ = eval_ppl(tok, model, rmt_model, args.train_data, args.max_docs, seg_len, max_seg, device, use_rmt=False)
        results.append(("Train-Base", base_ppl, n_docs, n_tok))

    # 2. Held-out eval
    if args.heldout:
        print(f"\n--- Held-Out Set PPL ({args.heldout}, {args.max_docs} docs) ---")
        rmt_ppl_h, n_docs_h, n_tok_h = eval_ppl(tok, model, rmt_model, args.heldout, args.max_docs, seg_len, max_seg, device, use_rmt=True)
        results.append(("Heldout-RMT", rmt_ppl_h, n_docs_h, n_tok_h))

        if not args.skip_base:
            base_ppl_h, _, _ = eval_ppl(tok, model, rmt_model, args.heldout, args.max_docs, seg_len, max_seg, device, use_rmt=False)
            results.append(("Heldout-Base", base_ppl_h, n_docs_h, n_tok_h))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ppl, nd, nt in results:
        print(f"  {name:20s}: PPL={ppl:.4f} ({nd} docs, {nt} tokens)")

    if len(results) >= 2:
        r = results[0]  # Train-RMT
        b = results[1]  # Train-Base
        if r[1] and b[1]:
            diff = r[1] - b[1]
            pct = (r[1] / b[1] - 1) * 100
            print(f"\n  Train PPL diff: {diff:+.4f} ({pct:+.2f}%)")

    if len(results) >= 4:
        rh = results[2]  # Heldout-RMT
        bh = results[3]  # Heldout-Base
        if rh[1] and bh[1]:
            diff_h = rh[1] - bh[1]
            pct_h = (rh[1] / bh[1] - 1) * 100
            print(f"  Heldout PPL diff: {diff_h:+.4f} ({pct_h:+.2f}%)")

if __name__ == "__main__":
    main()
