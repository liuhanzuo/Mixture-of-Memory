import torch, time
from transformers import AutoModel, AutoTokenizer
mp="models/Dream-Coder-v0-Instruct-7B"
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
m=AutoModel.from_pretrained(mp,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
MASK=m.config.mask_token_id

# A buggy solution. Bug is LOCAL: `>` should be `<`. Remask only that span, keep everything else.
buggy = '''def smallest(xs):
    best = xs[0]
    for v in xs:
        if v > best:
            best = v
    return best
'''
ids = tok(buggy, add_special_tokens=False).input_ids
dec = [tok.decode([t]) for t in ids]
print("TOKENS:", list(enumerate(dec)))
# find the ' >' token
tgt = [i for i,d in enumerate(dec) if d.strip()=='>']
print("remask idx", tgt)
canvas = list(ids)
for i in tgt: canvas[i]=MASK
x=torch.tensor([canvas],device="cuda")
out=m.diffusion_generate(x, attention_mask=torch.ones_like(x), max_new_tokens=1,
    steps=4, temperature=0.0, alg="entropy", alg_temp=0., return_dict_in_generate=True)
seq=out.sequences[0].tolist()
print("--- AFTER TARGETED REMASK (steps=4, 1 hole) ---")
print(tok.decode(seq[:len(ids)],skip_special_tokens=False))
# also: what does the model think the masked token should be, top-5?
with torch.no_grad():
    lg = m(x, "full", None).logits
    lg = torch.cat([lg[:,:1], lg[:,:-1]],dim=1)
p = lg[0,tgt[0]].softmax(-1).topk(5)
print("top5 at hole:", [(tok.decode([i]),round(v.item(),3)) for v,i in zip(p.values,p.indices)])
