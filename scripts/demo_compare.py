"""Sample one pg19 chunk, run through vanilla + 4 trained models, print comparison."""
import os, sys, json, argparse, logging, math, textwrap
import torch
import numpy as np
from transformers import LlamaTokenizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Dataset ──
def load_chunk(npy_path, idx=40042):
    data = np.load(npy_path, mmap_mode="r")
    chunk = data[idx].astype(np.int64)
    return chunk

# ── Vanilla Llama2-7B ──
def eval_vanilla(model_path, input_ids, labels, device):
    from transformers import LlamaForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids.unsqueeze(0).to(device), labels=labels.unsqueeze(0).to(device))
    loss = out.loss.item()
    # Generate continuation
    gen_ids = model.generate(input_ids[:512].unsqueeze(0).to(device), max_new_tokens=128, do_sample=False, temperature=1.0)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return {"loss": loss, "ppl": math.exp(loss), "generated": gen_text, "tokenizer": tokenizer}

# ── SparseMemory model ──
def eval_sparse(checkpoint_path, model_path, input_ids, labels, device,
                memory_slots=128, top_k=8, sliding_window=256):
    sys.path.insert(0, "/root/Mixture-of-Memory")
    from transformers import LlamaTokenizer
    from src.memory.sparse_memory.model import SparseMemoryLlamaForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = SparseMemoryLlamaForCausalLM(
        base_model=model_path,
        memory_slots=memory_slots,
        top_k=top_k,
        sliding_window=sliding_window,
        torch_dtype=torch.bfloat16,
    )
    # Load fine-tuned weights
    from safetensors.torch import load_file as st_load
    import glob
    st_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
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

    with torch.no_grad():
        out = model(input_ids=input_ids.unsqueeze(0).to(device), labels=labels.unsqueeze(0).to(device))
    loss = out.loss.item()
    gen_ids = model.generate(input_ids[:512].unsqueeze(0).to(device), max_new_tokens=128, do_sample=False, temperature=1.0)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return {"loss": loss, "ppl": math.exp(loss), "generated": gen_text, "tokenizer": tokenizer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/root/Mixture-of-Memory/models/Llama--Llama2-7b/")
    parser.add_argument("--chunk_idx", type=int, default=40042)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # Load one chunk
    chunk = load_chunk(args.data_path, args.chunk_idx)
    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
    labels = torch.tensor(chunk[1:], dtype=torch.long)

    # Show input text preview (first 500 tokens)
    tmp_tok = LlamaTokenizer.from_pretrained(args.model_path)
    input_text = tmp_tok.decode(chunk[:500], skip_special_tokens=True)
    del tmp_tok

    print("=" * 80)
    print(f"INPUT CHUNK (idx={args.chunk_idx}, first 500 tokens):")
    print("-" * 80)
    print(input_text[:1000])
    print("...")
    print("-" * 80)

    # 1. Vanilla
    print("\n" + "=" * 80)
    print("1) VANILLA Llama2-7B (no memory, no fine-tuning)")
    print("=" * 80)
    r = eval_vanilla(args.model_path, input_ids, labels, device)
    print(f"  Loss: {r['loss']:.4f}  |  PPL: {r['ppl']:.2f}")
    print(f"  Generated (512 input → 128 new tokens):")
    print(textwrap.indent(r['generated'][:600], "  "))
    if len(r['generated']) > 600:
        print("  ...")

    
    # 2-5. Trained models
    experiments = [
        ("selective_128", 128, 8, 256, "/root/Mixture-of-Memory/outputs/slp_selective_128/final"),
        ("selective_256", 256, 16, 256, "/root/Mixture-of-Memory/outputs/slp_selective_256/final"),
        ("full_write_128", 128, 8, 256, "/root/Mixture-of-Memory/outputs/slp_full_write_128/final"),
        ("full_write_256", 256, 16, 256, "/root/Mixture-of-Memory/outputs/slp_full_write_256/final"),
    ]

    for i, (name, slots, topk, sw, ckpt) in enumerate(experiments, 2):
        print(f"\n{'=' * 80}")
        print(f"{i}) {name} (slots={slots}, top_k={topk}, sliding_window={sw})")
        print("=" * 80)
        r = eval_sparse(ckpt, args.model_path, input_ids, labels, device,
                       memory_slots=slots, top_k=topk, sliding_window=sw)
        print(f"  Loss: {r['loss']:.4f}  |  PPL: {r['ppl']:.2f}")
        print(f"  Generated (512 input → 128 new tokens):")
        print(textwrap.indent(r['generated'][:600], "  "))
        if len(r['generated']) > 600:
            print("  ...")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Input: pg19 chunk {args.chunk_idx}, seq_len={len(chunk)}")
    print()

if __name__ == "__main__":
    main()
