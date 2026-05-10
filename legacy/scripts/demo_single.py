"""Run a single model's eval on one GPU for demo comparison. Called per-model."""
import os, sys, math, json, textwrap
import torch
import numpy as np
from transformers import LlamaTokenizer

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["vanilla", "selective_128", "selective_256", "full_write_128", "full_write_256"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/root/Mixture-of-Memory/models/Llama--Llama2-7b/")
    parser.add_argument("--data_path", type=str, default="/root/Mixture-of-Memory/data/pg19_chunks.npy")
    parser.add_argument("--chunk_idx", type=int, default=40042)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # Load data
    data = np.load(args.data_path, mmap_mode="r")
    chunk = data[args.chunk_idx].astype(np.int64)
    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
    labels = torch.tensor(chunk[1:], dtype=torch.long)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

    # Config map
    CONFIGS = {
        "selective_128": (128, 8, 256),
        "selective_256": (256, 16, 256),
        "full_write_128": (128, 8, 256),
        "full_write_256": (256, 16, 256),
    }

    if args.model_type == "vanilla":
        from transformers import LlamaForCausalLM
        print(f"[{args.model_type}] Loading vanilla Llama2-7B on GPU {args.gpu}...", flush=True)
        model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": device})
        model.eval()
    else:
        sys.path.insert(0, "/root/Mixture-of-Memory")
        from src.memory.sparse_memory.model import SparseMemoryLlamaForCausalLM
        from safetensors.torch import load_file as st_load
        import glob

        slots, topk, sw = CONFIGS[args.model_type]
        print(f"[{args.model_type}] Loading sparse model (slots={slots}, top_k={topk}, sw={sw}) on GPU {args.gpu}...", flush=True)
        model = SparseMemoryLlamaForCausalLM(
            base_model=args.model_path,
            memory_slots=slots, top_k=topk, sliding_window=sw,
            torch_dtype=torch.bfloat16,
        )
        # Load fine-tuned weights
        ckpt_path = args.checkpoint
        st_files = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
        ckpt_sd = {}
        for f in st_files:
            ckpt_sd.update(st_load(f, device="cpu"))
        model_sd = model.state_dict()
        remapped = {}
        for k, v in ckpt_sd.items():
            if k.startswith("model.") and not k.startswith("model.model."):
                new_key = "model.model." + k[len("model."):]
            else:
                continue
            if "memory_bank.memory" in new_key or "memory_bank.write_index" in new_key:
                continue
            if new_key in model_sd:
                remapped[new_key] = v
        model.load_state_dict(remapped, strict=False)
        model = model.to(device)
        model.eval()
        print(f"[{args.model_type}] Loaded {len(remapped)} params from checkpoint", flush=True)

    # Eval loss on full 4096 seq
    print(f"[{args.model_type}] Running forward pass...", flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids.unsqueeze(0).to(device), labels=labels.unsqueeze(0).to(device))
    loss = out.loss.item()
    ppl = math.exp(loss)

    # Generate from first 512 tokens
    print(f"[{args.model_type}] Generating 128 tokens from 512 input...", flush=True)
    with torch.no_grad():
        gen_ids = model.generate(input_ids[:512].unsqueeze(0).to(device), max_new_tokens=128, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # Only show the GENERATED part (after input)
    input_text = tokenizer.decode(chunk[:512], skip_special_tokens=True)
    new_text = gen_text[len(input_text):]

    result = {
        "model_type": args.model_type,
        "loss": round(loss, 4),
        "ppl": round(ppl, 2),
        "generated_new_tokens": new_text.strip(),
    }

    # Print report
    print(f"\n{'='*80}", flush=True)
    print(f"MODEL: {args.model_type}", flush=True)
    print(f"Loss: {loss:.4f}  |  PPL: {ppl:.2f}", flush=True)
    print(f"--- Generated (new tokens only, 512 input -> 128 new) ---", flush=True)
    print(new_text.strip(), flush=True)
    print(f"{'='*80}\n", flush=True)

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[{args.model_type}] Saved to {args.output}", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"[{args.model_type}] Done.", flush=True)

if __name__ == "__main__":
    main()
