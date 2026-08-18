import json,ast,torch,random
from transformers import AutoModel, AutoTokenizer
from evalplus.eval import untrusted_check
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth
mp="models/Dream-Coder-v0-Instruct-7B"
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
m=AutoModel.from_pretrained(mp,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
MASK=m.config.mask_token_id
probs=get_human_eval_plus(); gt=get_groundtruth(probs,get_human_eval_plus_hash(),[])
res=json.load(open('runs/dream_coder_instruct_heplus_r2/solutions_eval_results.json'))['eval']
def ok(tid,sol):
    p=probs[tid]
    try:
        st,_=untrusted_check("humaneval",p['prompt']+"\n"+sol,p['base_input'],p['entry_point'],
            expected=gt[tid]['base'],atol=p['atol'],ref_time=gt[tid]['base_time'],
            fast_check=False,min_time_limit=1.0,gt_time_limit_factor=4.0)
        return st=="pass"
    except Exception: return False
MUT={'>':'<','<':'>','>=':'<=','<=':'>=','==':'!=','!=':'==','+':'-','-':'+','0':'1','1':'0','and':'or','or':'and','max':'min','min':'max'}
passing=[(t,(v[0] if isinstance(v,list) else v)['solution']) for t,v in res.items()
         if (v[0] if isinstance(v,list) else v)['base_status']=='pass' and (v[0] if isinstance(v,list) else v)['plus_status']=='pass']
passing.sort(); random.seed(0)
cases=[]
for tid,sol in passing:
    ids=tok(sol,add_special_tokens=False).input_ids
    cand=[(i,tok.decode([t])) for i,t in enumerate(ids) if tok.decode([t]).strip() in MUT]
    if not cand: continue
    i,d=random.choice(cand)
    new=MUT[d.strip()]
    rep=d.replace(d.strip(),new)
    nt=tok(rep,add_special_tokens=False).input_ids
    if len(nt)!=1: continue
    mids=list(ids); mids[i]=nt[0]
    ms=tok.decode(mids,skip_special_tokens=True)
    try: ast.parse(ms)
    except Exception: continue
    if ok(tid,ms): continue   # mutation must actually break it
    cases.append((tid,ids,mids,i))
    if len(cases)>=30: break
print("n_mutation_cases",len(cases),flush=True)
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
def rd(ids,holes,steps):
    c=list(ids)
    for j in holes: c[j]=MASK
    x=torch.tensor([c],device="cuda")
    o=m.diffusion_generate(x,attention_mask=torch.ones_like(x),max_new_tokens=1,steps=steps,
        temperature=0.0,alg="entropy",alg_temp=0.,return_dict_in_generate=True)
    return tok.decode(o.sequences[0].tolist()[:len(ids)],skip_special_tokens=True)
top1=top5=0; rep_loo=rep_rand=rep_oracle=0
for tid,ids,mids,bug in cases:
    c=loo(mids); order=sorted(c,key=lambda j:c[j])
    r=order.index(bug)
    top1+= r==0; top5+= r<5
    for name,h in [('loo',order[:1]),('rand',order[:4]),('oracle',[bug])]:
        s=rd(mids,h,steps=max(2,len(h)))
        if ok(tid,s):
            if name=='loo': rep_loo+=1
            elif name=='rand': rep_rand+=1
            else: rep_oracle+=1
    print(tid,"n",len(mids),"bugrank",r,f"top1={top1} top5={top5} repK1={rep_loo} repK4={rep_rand} rep_oracle={rep_oracle}",flush=True)
n=len(cases)
print(f"FINAL n={n} loc_top1={top1}/{n} loc_top5={top5}/{n} repair_K1={rep_loo}/{n} repair_K4={rep_rand}/{n} repair_oracle={rep_oracle}/{n}")
