import torch,time
from transformers import AutoModel, AutoTokenizer
mp="models/Dream-Coder-v0-Instruct-7B"
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
m=AutoModel.from_pretrained(mp,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
MASK=m.config.mask_token_id
code=open("/tmp/sample.py").read() if False else '''def solve(nums, target):
    seen = {}
    for i, v in enumerate(nums):
        if target - v in seen:
            return [seen[target - v], i]
        seen[v] = i
    return []
'''*3
ids=tok(code,add_special_tokens=False).input_ids
n=len(ids); print("n_tokens",n)
# STRIDED LOO: mask every k-th position per row -> n/stride... use stride s so masked positions are >=s apart
for stride in (1,8,32):
    rows=[]; meta=[]
    if stride==1:
        for i in range(n):
            r=list(ids); r[i]=MASK; rows.append(r); meta.append([i])
    else:
        for off in range(stride):
            pos=list(range(off,n,stride))
            r=list(ids)
            for i in pos: r[i]=MASK
            rows.append(r); meta.append(pos)
    X=torch.tensor(rows,device="cuda")
    torch.cuda.synchronize(); t0=time.perf_counter()
    outs=[]
    B=32
    with torch.no_grad():
        for b in range(0,len(rows),B):
            lg=m(X[b:b+B],"full",None).logits
            lg=torch.cat([lg[:,:1],lg[:,:-1]],dim=1)
            outs.append(lg.softmax(-1).cpu())
    torch.cuda.synchronize()
    P=torch.cat(outs)
    conf={}
    for r,pos in enumerate(meta):
        for i in pos: conf[i]=P[r,i,ids[i]].item()
    lowest=sorted(conf,key=lambda i:conf[i])[:6]
    print(f"stride={stride:3d} rows={len(rows):4d} time={time.perf_counter()-t0:6.2f}s  lowest6={[(i,repr(tok.decode([ids[i]])),round(conf[i],3)) for i in lowest]}")
