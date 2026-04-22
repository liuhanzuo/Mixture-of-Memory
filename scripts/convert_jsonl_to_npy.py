"""Convert .jsonl ({"text": ...}) to pre-tokenized .npy [N, seq_len] uint16."""
import argparse, json, numpy as np, sys
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--max_docs", type=int, default=100000)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizing {args.input} -> {args.output} (seq_len={args.seq_len})")

    chunks = []
    doc_count = 0
    with open(args.input) as f:
        for line in f:
            doc_count += 1
            if doc_count > args.max_docs:
                break
            text = json.loads(line)["text"]
            ids = tok.encode(text, add_special_tokens=True)
            # Split into fixed-length chunks
            for i in range(0, len(ids), args.seq_len):
                chunk = ids[i:i+args.seq_len]
                if len(chunk) < args.seq_len:
                    chunk = chunk + [tok.pad_token_id or 0] * (args.seq_len - len(chunk))
                chunks.append(chunk)
            if doc_count % 1000 == 0:
                print(f"  docs={doc_count}, chunks={len(chunks)}", flush=True)

    arr = np.array(chunks, dtype=np.uint32)
    np.save(args.output, arr)
    print(f"Done: {arr.shape[0]} chunks, shape={arr.shape}, saved to {args.output}")

if __name__ == "__main__":
    main()
