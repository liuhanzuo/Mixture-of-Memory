import json,ast,torch,random,difflib,os
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
def check(tid,sol):
    p=probs[tid]
    try:
        st,_=untrusted_check("humaneval",p['prompt']+"\n"+sol,p['base_input'],p['entry_point'],
            expected=gt[tid]['base'],atol=p['atol'],ref_time=gt[tid]['base_time'],
            fast_check=False,min_time_limit=1.0,gt_time_limit_factor=4.0)
        return st=="pass"
    except Exception: return False
fails=[]
for tid,v in res.items():
    r=v[0] if isinstance(v,list) else v
    if r['base_status']=='pass' and r['plus_status']=='pass': continue
    s=r['solution']
    try: ast.parse(s)
    except Exception: continue
    fails.append((tid,s))
fails.sort(); fails=fails[:24]
def rediffuse(ids,holes,steps):
    c=list(ids)
    for i in holes: c[i]=MASK
    x=torch.tensor([c],device="cuda")
    o=m.diffusion_generate(x,attention_mask=torch.ones_like(x),max_new_tokens=1,steps=steps,
        temperature=0.1,top_p=0.95,alg="entropy",alg_temp=0.,return_dict_in_generate=True)
    return tok.decode(o.sequences[0].tolist()[:len(ids)],skip_special_tokens=True)
def oracle_holes(ids,tid):
    # ORACLE: token-level diff vs canonical_solution -> mask the differing positions
    ref=probs[tid]['canonical_solution']
    rids=tok(ref,add_special_tokens=False).input_ids
    sm=difflib.SequenceMatcher(a=ids,b=rids)
    h=[]
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op!='equal': h.extend(range(i1,i2))
    return h
random.seed(0); tally={}
for K in (16,48):
    for name in ('loo','rand'): tally[(K,name)]=0
tally[('oracle','oracle')]=0; tally[('oracle','full')]=0
for tid,sol in fails:
    ids=tok(sol,add_special_tokens=False).input_ids; n=len(ids)
    # LOO confs
    rows=[];meta=[]
    for off in range(8):
        pos=list(range(off,n,8)); r=list(ids)
        for i in pos: r[i]=MASK
        rows.append(r);meta.append(pos)
    with torch.no_grad(): lg=m(torch.tensor(rows,device="cuda"),"full",None).logits
    lg=torch.cat([lg[:,:1],lg[:,:-1]],dim=1);P=lg.softmax(-1)
    conf={}
    for r,pos in enumerate(meta):
        for i in pos: conf[i]=P[r,i,ids[i]].item()
    for K in (16,48):
        for name,h in [('loo',sorted(conf,key=lambda i:conf[i])[:K]),('rand',random.sample(range(n),min(K,n)))]:
            new=rediffuse(ids,h,steps=max(8,K//2))
            if check(tid,new): tally[(K,name)]+=1
    oh=oracle_holes(ids,tid)
    if oh:
        new=rediffuse(ids,oh,steps=max(8,len(oh)//2)); 
        if check(tid,new): tally[('oracle','oracle')]+=1
    new=rediffuse(ids,list(range(n)),steps=max(16,n//2))
    if check(tid,new): tally[('oracle','full')]+=1
    print(tid,n,"n_oracle_holes",len(oh),dict(tally),flush=True)
print("FINAL n=%d"%len(fails),dict(tally))
