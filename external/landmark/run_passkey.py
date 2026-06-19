# Parametrized passkey-retrieval eval for Landmark Attention reproduction (Phase 1 S0).
# Adapted from epfml/landmark-attention llama/run_test.py: paths/knobs are read from env
# instead of being hardcoded, results are written to CSV, and only the requested models run.
#
# Env vars:
#   LM_BASE   : path to original LLaMA-1-7B (HF format) base weights  (for "base" arm; optional)
#   LM_TUNED  : path to recovered landmark-tuned ckpt                 (for "mem" arm)
#   LM_CACHE  : hf cache dir (default ./hf-cache)
#   LM_MODELS : comma list subset of {base,mem}  (default "base,mem")
#   LM_TOPK   : retrieval top_k blocks (default 5)
#   LM_NTESTS : tests per length (default 50)
#   LM_NVALUES: comma list of n_garbage char counts (default repo set)
#   LM_OUT    : output CSV path (default ./passkey_results.csv)
#   LM_BASE_DEVICE / LM_MEM_DEVICE : cuda devices (default cuda:0 / cuda:0)
import os
import random
import re
import csv
import torch

llama_weights_7b_base = os.environ.get("LM_BASE", "")
llama_weights_7b_tuned = os.environ.get("LM_TUNED", "")
cache_path = os.environ.get("LM_CACHE", "./hf-cache/")
use_flash = False  # high-level path; flash only needed for cpu-offload inference
top_k = int(os.environ.get("LM_TOPK", "5"))
dtype = torch.bfloat16
models = [m for m in os.environ.get("LM_MODELS", "base,mem").split(",") if m]
num_tests = int(os.environ.get("LM_NTESTS", "50"))
base_device = os.environ.get("LM_BASE_DEVICE", "cuda:0")
mem_device = os.environ.get("LM_MEM_DEVICE", "cuda:0")
out_csv = os.environ.get("LM_OUT", "./passkey_results.csv")
seed = int(os.environ.get("LM_SEED", "1234"))
random.seed(seed)

if "LM_NVALUES" in os.environ:
    n_values = [int(x) for x in os.environ["LM_NVALUES"].split(",")]
else:
    n_values = [0, 100, 500, 1000, 5000, 8000, 10000, 12000, 14000, 18000, 20000, 25000, 38000]


def make_llama_base_pipe():
    from transformers import pipeline
    from transformers.models.llama import LlamaForCausalLM
    import transformers
    m = LlamaForCausalLM.from_pretrained(llama_weights_7b_base, cache_dir=cache_path, torch_dtype=dtype)
    m = m.to(base_device)
    tok = transformers.AutoTokenizer.from_pretrained(
        llama_weights_7b_base, cache_dir=cache_path, model_max_length=2048,
        padding_side="right", use_fast=False)
    return pipeline("text-generation", model=m, tokenizer=tok, device=m.device)


def make_llama_mem_pipe():
    from llama_mem import LlamaForCausalLM
    import transformers
    from transformers import pipeline
    model = LlamaForCausalLM.from_pretrained(llama_weights_7b_tuned, cache_dir=cache_path, torch_dtype=dtype)
    model.to(mem_device)
    tok = transformers.AutoTokenizer.from_pretrained(
        llama_weights_7b_tuned, cache_dir=cache_path,
        model_max_length=model.config.train_context_length,
        padding_side="right", use_fast=False)
    mem_id = tok.convert_tokens_to_ids("<landmark>")
    model.set_mem_id(mem_id)
    return pipeline("text-generation", model=model, tokenizer=tok, device=model.device,
                    offload_cache_to_cpu=use_flash, use_flash=use_flash, cache_top_k=top_k)


pipes = {}
if "base" in models:
    if not llama_weights_7b_base:
        raise SystemExit("LM_BASE unset but 'base' requested")
    print("Loading base pipe...", flush=True)
    pipes["base"] = make_llama_base_pipe()
if "mem" in models:
    if not llama_weights_7b_tuned:
        raise SystemExit("LM_TUNED unset but 'mem' requested")
    print("Loading mem (landmark) pipe...", flush=True)
    pipes["mem"] = make_llama_mem_pipe()


def generate_prompt(n_garbage):
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix
    task_description = ("There is an important info hidden inside a lot of irrelevant text. "
                        "Find it and memorize them. I will quiz you about the important "
                        "information there.")
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 2000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [task_description, garbage_prefix, information_line, garbage_suffix, final_question]
    return "\n".join(lines), pass_key


def test_model(prompt_text, pass_key, model_name):
    response = pipes[model_name](prompt_text, num_return_sequences=1, max_new_tokens=10)[0]["generated_text"][len(prompt_text):]
    assert f"The pass key is {pass_key}" in prompt_text
    try:
        out = int(re.search(r'\d+', response).group())
    except Exception:
        out = response[:20]
    return out


accuracies = {x: [] for x in models}
rows = []
for n in n_values:
    correct_count = {x: 0 for x in models}
    ntok = {x: 0 for x in models}
    for i in range(num_tests):
        prompt_text, pass_key = generate_prompt(n)
        for model_name in models:
            num_tokens = len(pipes[model_name].tokenizer.encode(prompt_text))
            ntok[model_name] = num_tokens
            model_output = test_model(prompt_text, pass_key, model_name)
            ok = (pass_key == model_output)
            if ok:
                correct_count[model_name] += 1
            print(f"n={n} test {i+1}/{num_tests} [{model_name}] tok={num_tokens} "
                  f"expect={pass_key} got={model_output} {'OK' if ok else 'FAIL'}", flush=True)
    for model in models:
        acc = (correct_count[model] / num_tests) * 100
        accuracies[model].append(acc)
        rows.append({"model": model, "n_garbage": n, "num_tokens": ntok[model],
                     "num_tests": num_tests, "correct": correct_count[model], "accuracy_pct": acc})
        print(f"==> Accuracy {model} n={n} (tok~{ntok[model]}): {acc}%", flush=True)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "n_garbage", "num_tokens", "num_tests", "correct", "accuracy_pct"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("\n=== SUMMARY ===", flush=True)
for model in models:
    print(model, dict(zip(n_values, accuracies[model])), flush=True)
print("CSV written to", out_csv, flush=True)
