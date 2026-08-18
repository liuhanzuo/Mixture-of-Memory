import torch
from transformers import AutoModel, AutoTokenizer
mp="models/Dream-Coder-v0-Instruct-7B"
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
m=AutoModel.from_pretrained(mp,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
MASK=m.config.mask_token_id

def scan(code,label):
    ids=tok(code,add_special_tokens=False).input_ids
    x=torch.tensor([ids],device="cuda")
    with torch.no_grad(): lg=m(x,"full",None).logits
    lg=torch.cat([lg[:,:1],lg[:,:-1]],dim=1)
    pr=lg[0].softmax(-1)
    # (a) VISIBLE self-confidence: p(observed token | full visible seq incl itself)
    vis=[pr[i,t].item() for i,t in enumerate(ids)]
    # (b) LEAVE-ONE-OUT: mask position i, ask p(observed token)
    loo=[]
    for i in range(len(ids)):
        xx=x.clone(); xx[0,i]=MASK
        with torch.no_grad(): l2=m(xx,"full",None).logits
        l2=torch.cat([l2[:,:1],l2[:,:-1]],dim=1)
        loo.append(l2[0,i].softmax(-1)[ids[i]].item())
    order_v=sorted(range(len(ids)),key=lambda i:vis[i])
    order_l=sorted(range(len(ids)),key=lambda i:loo[i])
    print(f"\n=== {label}")
    print(" visible-conf 5 lowest:", [(i,tok.decode([ids[i]]),round(vis[i],3)) for i in order_v[:5]])
    print(" LOO-conf     5 lowest:", [(i,tok.decode([ids[i]]),round(loo[i],3)) for i in order_l[:5]])
    return ids,vis,loo,order_v,order_l

buggy='''def smallest(xs):
    best = xs[0]
    for v in xs:
        if v > best:
            best = v
    return best
'''
ids,vis,loo,ov,ol=scan(buggy,"BUGGY (bug at idx 20 ' >')")
print(" rank of true bug idx20:  visible=",ov.index(20)," LOO=",ol.index(20)," / n=",len(ids))
good=buggy.replace(" v > best"," v < best")
scan(good,"CORRECT (control)")
