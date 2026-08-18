import json,ast,torch,random
from transformers import AutoModel, AutoTokenizer
from evalplus.data import get_human_eval_plus
mp="models/Dream-Coder-v0-Instruct-7B"
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
m=AutoModel.from_pretrained(mp,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
MASK=m.config.mask_token_id
probs=get_human_eval_plus()
res=json.load(open('runs/dream_coder_instruct_heplus_r2/solutions_eval_results.json'))['eval']
MUT={'>':'<','<':'>','+':'-','-':'+','0':'1','1':'0','max':'min','min':'max','and':'or','or':'and','==':'!=','!=':'=='}
passing=sorted([(t,(v[0] if isinstance(v,list) else v)['solution']) for t,v in res.items()
   if (v[0] if isinstance(v,list) else v)['base_status']=='pass'])
random.seed(0)
def loo(ids,stride=8):
    n=len(ids);rows=[];meta=[]
    for off in range(stride):
        pos=list(range(off,n,stride)); r=list(ids)
        for j in pos: r[j]=MASK
        rows.append(r);meta.append(pos)
    with torch.no_grad(): lg=m(torch.tensor(rows,device="cuda"),"full",None).logits
    lg=torch.cat([lg[:,:1],lg[:,:-1]],dim=1);P=lg.softmax(-1)
    c={}
    for r,pos in enumerate(meta):
        for j in pos: c[j]=P[r,j,ids[j]].item()
    return c
n_done=0; collateral=[]
for tid,sol in passing:
    ids=tok(sol,add_special_tokens=False).input_ids
    cand=[(i,tok.decode([t])) for i,t in enumerate(ids) if tok.decode([t]).strip() in MUT]
    if not cand: continue
    i,d=random.choice(cand)
    nt=tok(d.replace(d.strip(),MUT[d.strip()]),add_special_tokens=False).input_ids
    if len(nt)!=1: continue
    mids=list(ids); mids[i]=nt[0]
    c=loo(mids); order=sorted(c,key=lambda j:c[j])
    # for the 3 non-bug positions in top-4: is the ORIGINAL token recoverable? (LOO conf in CLEAN code)
    cc=loo(ids)
    nb=[j for j in order[:4] if j!=i][:3]
    collateral.append([(repr(tok.decode([mids[j]])), round(cc.get(j,0),3)) for j in nb])
    n_done+=1
    if n_done>=12: break
print("For each case: top-4 LOO positions OTHER than the bug, with their LOO-confidence IN CLEAN CODE")
print("(low clean-conf => irreducibly unpredictable-but-CORRECT => masking it inflicts collateral damage)")
allc=[]
for row in collateral:
    print("  ",row); allc+= [v for _,v in row]
import statistics
print("n=%d non-bug top4 positions; median clean LOO conf=%.3f ; frac<0.5 = %.2f"%(
   len(allc),statistics.median(allc), sum(1 for v in allc if v<0.5)/len(allc)))
