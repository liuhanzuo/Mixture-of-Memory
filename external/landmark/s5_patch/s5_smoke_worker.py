"""S5 smoke worker. Runs INSIDE a copied llama/ package dir (cwd on sys.path).
Builds a tiny LlamaForCausalLM (landmark variant), inserts physical <landmark>
tokens every (mem_freq+1), runs ONE forward on a fixed input with a fixed seed,
and writes the resulting logits + the S5 readout-branch counters to <out>.pt.

Usage:  python s5_smoke_worker.py <single_layer_mem|none> <out.pt>
CPU-only (caller sets CUDA_VISIBLE_DEVICES="").
"""
import sys, os
import torch

sys.path.insert(0, os.getcwd())
import llama_mem as LM
from llama_landmark_config import LlamaLandmarkConfig

arg = sys.argv[1]
out = sys.argv[2]
single_layer_mem = None if arg.lower() == "none" else int(arg)

# Tiny faithful config. 4 layers so a single target layer is unambiguous.
# Tiny faithful config. 32 layers (matches the real LLaMA-1-7B depth) so a
# single target layer (e.g. L16) is an unambiguous, representative test.
MEM_FREQ = 7          # block size = 8
MEM_ID = 300
VOCAB = 320
cfg_kwargs = dict(
    vocab_size=VOCAB,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=32,
    num_attention_heads=4,
    max_position_embeddings=512,
    mem_id=MEM_ID,
    mem_freq=MEM_FREQ,
    train_context_length=64,
)
# Only pass single_layer_mem if the config supports it (anchor build won't).
try:
    cfg = LlamaLandmarkConfig(single_layer_mem=single_layer_mem, **cfg_kwargs)
    supports_s5 = hasattr(cfg, "single_layer_mem")
except TypeError:
    cfg = LlamaLandmarkConfig(**cfg_kwargs)
    supports_s5 = False

torch.manual_seed(0)
model = LM.LlamaForCausalLM(cfg)
model.eval().float()

# Build a fixed input: 4 blocks of (7 normal tokens + 1 landmark) = 32 tokens.
torch.manual_seed(123)
n_blocks = 4
block = MEM_FREQ + 1
ids = []
for b in range(n_blocks):
    ids.extend(torch.randint(0, 290, (MEM_FREQ,)).tolist())  # normal tokens < mem_id
    ids.append(MEM_ID)                                       # landmark token
input_ids = torch.tensor([ids], dtype=torch.long)
attn = torch.ones_like(input_ids)

if hasattr(LM, "_s5_reset_counters"):
    LM._s5_reset_counters()

with torch.no_grad():
    out_obj = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
logits = out_obj.logits.detach().float()

counters = {}
if hasattr(LM, "_S5_READOUT_COUNTS"):
    counters = {
        "grouped_calls": LM._S5_READOUT_COUNTS["grouped"],
        "plain_calls": LM._S5_READOUT_COUNTS["plain"],
        "grouped_layers": sorted(LM._S5_GROUPED_LAYERS),
        "plain_layers": sorted(LM._S5_PLAIN_LAYERS),
    }

torch.save({"logits": logits, "single_layer_mem": single_layer_mem,
            "supports_s5": supports_s5, "counters": counters}, out)
print(f"[worker] arg={arg} supports_s5={supports_s5} logits_shape={tuple(logits.shape)} "
      f"sum={logits.double().sum().item():.6f} counters={counters}")
