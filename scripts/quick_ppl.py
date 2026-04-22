"""Quick PPL evaluation for RMT v1."""
import torch, json, sys, math, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import RMTMemory, RMTModel

def main():
    device = "cuda"
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "outputs/rmt_v1_20260415_200307/final"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/rmt_train_wikitext.jsonl"
    max_docs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Load config
    cfg_path = os.path.join(ckpt, "rmt_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    num_mem = cfg["num_memory_tokens"]
    seg_len = cfg["segment_length"]
    max_seg = cfg["max_segments"]
    print(f"Config: mem={num_mem}, seg_len={seg_len}, max_seg={max_seg}")

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    print("Model loaded")

    hidden_dim = model.config.hidden_size
    bottleneck_dim = cfg.get("bottleneck_dim", 32)
    rmt_mem = RMTMemory(hidden_dim, num_mem, cfg.get("num_memory_heads", 8), max_seg + 1, bottleneck_dim=bottleneck_dim).to(device=device, dtype=torch.bfloat16)
    rmt_mem.load_state_dict(torch.load(os.path.join(ckpt, "rmt_memory.pt"), map_location=device, weights_only=False))
    rmt_mem.eval()
    rmt_model = RMTModel(model, rmt_mem, seg_len).to(device=device, dtype=torch.bfloat16)
    rmt_model.eval()
    print("RMT loaded, starting eval...")

    # Load docs
    docs = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            docs.append(json.loads(line)["text"][:seg_len * max_seg])

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, doc in enumerate(docs):
            tokens = tok.encode(doc, add_special_tokens=False)
            if len(tokens) < seg_len:
                continue
            tokens = tokens[:seg_len * max_seg]
            input_ids = torch.tensor([tokens], device=device)
            loss_val = rmt_model(input_ids=input_ids, labels=input_ids, training=False)
            n = len(tokens) - 1
            total_loss += loss_val.item() * n
            total_tokens += n
            print(f"  Doc {i}: loss={loss_val.item():.4f}, tokens={n}")

    ppl = math.exp(total_loss / total_tokens)
    print(f"\n=== RMT PPL: {ppl:.4f} ({total_tokens} tokens, {len(docs)} docs) ===")

    # Also compute base model PPL for comparison
    print("\nComputing base model PPL (no RMT)...")
    total_loss_base = 0.0
    total_tokens_base = 0
    with torch.no_grad():
        for i, doc in enumerate(docs):
            tokens = tok.encode(doc, add_special_tokens=False)
            if len(tokens) < seg_len:
                continue
            tokens = tokens[:seg_len * max_seg]
            input_ids = torch.tensor([tokens], device=device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            n = len(tokens) - 1
            total_loss_base += outputs.loss.item() * n
            total_tokens_base += n
            print(f"  Doc {i}: loss={outputs.loss.item():.4f}, tokens={n}")

    ppl_base = math.exp(total_loss_base / total_tokens_base)
    print(f"\n=== Base PPL: {ppl_base:.4f} ({total_tokens_base} tokens) ===")
    print(f"\n=== PPL diff: {ppl - ppl_base:+.4f} ({(ppl/ppl_base - 1)*100:+.2f}%) ===")

if __name__ == "__main__":
    main()
