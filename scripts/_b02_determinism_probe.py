"""B02 pre-data gate: prove the FIXED sample set is reproducible across
independent processes (the property T21 lacked). Builds RULER vt samples with
the current crc32 base_seed and prints sha256 of the tokenized prompt.
CPU only -- no model weights loaded."""
import hashlib, json, os, random, sys, zlib
sys.path.insert(0, os.getcwd())
from transformers import AutoTokenizer
import scripts.eval_ruler_mem_space as ruler

MODEL = sys.argv[1]
LENGTH = sys.argv[2]
N = int(sys.argv[3])
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
TASK = "variable_tracking"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_seed = SEED + (zlib.crc32(f"{TASK}\x00{LENGTH}".encode()) % 100000)
vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)
out = {"pid": os.getpid(), "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
       "base_seed": base_seed, "length": LENGTH, "seed": SEED, "items": []}
for i in range(N):
    rng = random.Random(base_seed * 1000 + i)
    prompt, answers, gold = ruler._build_sample(TASK, ruler._LENGTH_TOKENS[LENGTH], tok, rng, vt_icl)
    ids = tok(prompt, return_tensors="pt").input_ids
    out["items"].append({
        "i": i,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:16],
        "n_tok": int(ids.shape[1]),
        "target": " | ".join(answers),
    })
print(json.dumps(out))
