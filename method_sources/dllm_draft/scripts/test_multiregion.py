import torch, time
from transformers import AutoModel, AutoTokenizer
mp="models/Dream-Coder-v0-Instruct-7B"
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
m=AutoModel.from_pretrained(mp,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
MASK=m.config.mask_token_id
print("mask_token_id",MASK)

# multi-region infilling: TWO holes, mutually dependent (var name must match)
tmpl_a = "def f(xs):\n    total = 0\n    for "
tmpl_b = " in xs:\n        total += "
tmpl_c = "\n    return total\n"
ids_a=tok(tmpl_a,add_special_tokens=False).input_ids
ids_b=tok(tmpl_b,add_special_tokens=False).input_ids
ids_c=tok(tmpl_c,add_special_tokens=False).input_ids
H=3
canvas = ids_a + [MASK]*H + ids_b + [MASK]*H + ids_c
x=torch.tensor([canvas],device="cuda")
print("canvas len",x.shape, "n_masks",(x==MASK).sum().item())
t0=time.perf_counter()
out=m.diffusion_generate(x, attention_mask=torch.ones_like(x), max_new_tokens=1,
    steps=8, temperature=0.0, top_p=0.95, alg="entropy", alg_temp=0.,
    output_history=False, return_dict_in_generate=True)
print("elapsed %.1fs"%(time.perf_counter()-t0))
seq=out.sequences[0].tolist()
print("n_masks_left",sum(1 for t in seq if t==MASK))
print("---FILLED---")
print(tok.decode(seq,skip_special_tokens=False))
