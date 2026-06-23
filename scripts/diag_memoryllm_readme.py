"""Minimal README-faithful MemoryLLM test on the pinned env, with faulthandler
to locate the SIGFPE. Follows the HF README verbatim.
"""
from __future__ import annotations
import faulthandler, os, sys, json
faulthandler.enable()
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORYLLM_SRC = PROJECT_ROOT.parent / "MemoryLLM-source"
for _p in (str(MEMORYLLM_SRC), str(PROJECT_ROOT)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer, AutoConfig
from modeling_memoryllm import MemoryLLM

SNAP = os.environ["MLLM_SNAP"]
DEVICE = "cuda:0"

tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
config = AutoConfig.from_pretrained(SNAP, local_files_only=True)
raw = json.load(open(os.path.join(SNAP, "config.json")))
if "rope_theta" not in raw and isinstance(raw.get("rope_scaling"), dict):
    config.rope_theta = raw["rope_scaling"].get("rope_theta", 500000.0)
else:
    config.rope_theta = raw.get("rope_theta", 500000.0)
print("[cfg] rope_scaling=", getattr(config, "rope_scaling", None), "rope_theta=", config.rope_theta, flush=True)

model = MemoryLLM.from_pretrained(SNAP, attn_implementation="sdpa",
                                  config=config, torch_dtype=torch.bfloat16, local_files_only=True)
model = model.to(DEVICE)
model.eval()
print("[ok] loaded; initialized=", int(model.initialized.item()), flush=True)

ctx = ("Last week, John had a wonderful picnic with David. During their conversation, "
       "David mentioned multiple times that he likes eating apples. Though he didn't mention "
       "any other fruits, John says he can infer that David also like bananas.")
ctx_ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
print("[inject] ctx tokens=", ctx_ids.shape[1], flush=True)
with torch.no_grad():
    model.inject_memory(ctx_ids, update_memory=True)
print("[inject] done; initialized=", int(model.initialized.item()), flush=True)

messages = [{"role": "user", "content": "What fruits does David like?"}]
inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[:, 1:]
terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
print("[gen] input shape=", inputs.shape, flush=True)

# (A) single forward -> argmax (isolates forward math from generation loop)
model.config.use_cache = False
with torch.no_grad():
    logits = model(input_ids=inputs.to(DEVICE), use_cache=False, return_dict=True).logits
nxt = int(logits[0, -1].argmax().item())
print("[forward] last-logit argmax token=", nxt, repr(tok.decode([nxt])),
      "logit max/min=", float(logits[0,-1].max()), float(logits[0,-1].min()), flush=True)

# (B) generate with use_cache=False (exactly the runner config)
with torch.no_grad():
    out = model.generate(input_ids=inputs.to(DEVICE), max_new_tokens=20,
                         eos_token_id=terminators, do_sample=False, num_beams=1,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=False)
print("[gen use_cache=False] raw decode:", repr(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)), flush=True)
print("[done]", flush=True)
