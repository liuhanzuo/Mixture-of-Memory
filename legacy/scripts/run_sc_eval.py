#!/usr/bin/env python3
"""Run Selective Context evaluation - self-contained, uses local gpt2."""
import torch, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transformers
from src.memory.selective_context import SelectiveContext

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "openai-community--gpt2")

def main():
    print(f"Loading GPT-2 from {MODEL_PATH}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, use_safetensors=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    prompts = {
        "medium": [
            "The history of artificial intelligence dates back to ancient times when philosophers first began to ponder the nature of thought and consciousness. " * 20,
            "In computer science, algorithms form the foundation of efficient problem-solving. Various sorting algorithms have been developed over the decades. " * 20,
            "The concept of deep learning has transformed natural language processing. Transformers architecture introduced attention mechanisms. " * 20,
        ],
        "long": [
            "The evolution of language models represents a fascinating journey in artificial intelligence. Early systems relied on statistical methods and n-gram models. " * 40,
            "Attention mechanisms revolutionized natural language processing by allowing models to focus on relevant parts of input sequences. " * 40,
            "The development of large language models has accelerated dramatically in recent years, with models growing from millions to billions of parameters. " * 40,
        ],
    }

    results = {}
    for ctx_type, texts in prompts.items():
        print(f"\n{'='*50}\n{ctx_type} context\n{'='*50}")
        for ratio in [0.3, 0.5, 0.7]:
            compressor = SelectiveContext(compression_ratio=ratio, method="importance", window_size=64)
            orig_ppls, comp_ppls, comp_ratios = [], [], []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
                with torch.no_grad():
                    orig_loss = model(**inputs, labels=inputs['input_ids']).loss.item()
                orig_ppl = torch.exp(torch.tensor(orig_loss)).item()

                c_ids, c_mask = compressor.compress(inputs['input_ids'], inputs['attention_mask'])
                if c_ids.shape[1] < 2:
                    continue
                c_inputs = {'input_ids': c_ids, 'attention_mask': c_mask}
                with torch.no_grad():
                    comp_loss = model(**c_inputs, labels=c_inputs['input_ids']).loss.item()
                comp_ppl = torch.exp(torch.tensor(comp_loss)).item()
                actual_ratio = c_ids.shape[1] / inputs['input_ids'].shape[1]

                orig_ppls.append(orig_ppl)
                comp_ppls.append(comp_ppl)
                comp_ratios.append(actual_ratio)

            avg_o = sum(orig_ppls)/len(orig_ppls)
            avg_c = sum(comp_ppls)/len(comp_ppls)
            avg_r = sum(comp_ratios)/len(comp_ratios)
            pct = (avg_c - avg_o) / avg_o * 100
            print(f"  ratio={ratio}: orig_ppl={avg_o:.2f}  comp_ppl={avg_c:.2f}  change={pct:+.1f}%  actual_ratio={avg_r:.3f}")
            results[f"{ctx_type}_r{ratio}"] = {
                "orig_ppl": round(avg_o, 2),
                "comp_ppl": round(avg_c, 2),
                "ppl_change_pct": round(pct, 1),
                "actual_ratio": round(avg_r, 3),
            }

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/selective_context_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/selective_context_eval.json")

    # Print summary comparison
    print("\n" + "="*50)
    print("COMPARISON WITH SPARSE MEMORY")
    print("="*50)
    print("Sparse memory: +20% PPL regression (PPL 41.24 -> ~49.6)")
    print("Selective Context results:")
    for k, v in results.items():
        print(f"  {k}: {v['ppl_change_pct']:+.1f}% PPL change at ratio {v['actual_ratio']}")

if __name__ == "__main__":
    main()
