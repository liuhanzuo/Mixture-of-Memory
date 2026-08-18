import json,ast,torch,random,sys,os
from transformers import AutoModel, AutoTokenizer
from evalplus.eval import untrusted_check
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth
mp="models/Dream-Coder-v0-Instruct-7B"
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
m=AutoModel.from_pretrained(mp,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda").eval()
MASK=m.config.mask_token_id
probs=get_human_eval_plus(); h=get_human_eval_plus_hash()
gt=get_groundtruth(probs,h,[])
res=json.load(open('runs/dream_coder_instruct_heplus_r2/solutions_eval_results.json'))['eval']
def check(tid,sol):
    p=probs[tid]; code=p['prompt']+"\n"+sol
    st,det=untrusted_check("humaneval",code,p['base_input'],p['entry_point'],
        expected=gt[tid]['base'],atol=p['atol'],ref_time=gt[tid]['base_time'],
        fast_check=False,min_time_limit=1.0,gt_time_limit_factor=4.0)
    return st
fails=[]
for tid,v in res.items():
    r=v[0] if isinstance(v,list) else v
    if r['base_status']=='pass' and r['plus_status']=='pass': continue
    s=r['solution']
    try: ast.parse(s)
    except Exception: continue
    fails.append((tid,s))
fails.sort(); fails=fails[:24]
print("n_repairable_candidates",len(fails),flush=True)
def loo_conf(ids,stride=8):
    n=len(ids); conf={}
    rows=[];meta=[]
    for off in range(stride):
        pos=list(range(off,n,stride)); r=list(ids)
        for i in pos: r[i]=MASK
        rows.append(r);meta.append(pos)
    X=torch.tensor(rows,device="cuda")
    with torch.no_grad():
        lg=m(X,"full",None).logits
    lg=torch.cat([lg[:,:1],lg[:,:-1]],dim=1); P=lg.softmax(-1)
    for r,pos in enumerate(meta):
        for i in pos: conf[i]=P[r,i,ids[i]].item()
    return conf
def rediffuse(ids,holes,steps=16):
    c=list(ids)
    for i in holes: c[i]=MASK
    x=torch.tensor([c],device="cuda")
    out=m.diffusion_generate(x,attention_mask=torch.ones_like(x),max_new_tokens=1,steps=steps,
        temperature=0.1,top_p=0.95,alg="entropy",alg_temp=0.,return_dict_in_generate=True)
    return tok.decode(out.sequences[0].tolist()[:len(ids)],skip_special_tokens=True)
K=int(os.environ.get("K","8")); random.seed(0)
tally={'loo':0,'rand':0,'tail':0}
for tid,sol in fails:
    ids=tok(sol,add_special_tokens=False).input_ids
    n=len(ids)
    if n<12 or n>900: continue
    conf=loo_conf(ids)
    loo_h=sorted(conf,key=lambda i:conf[i])[:K]
    rand_h=random.sample(range(n),min(K,n))
    tail_h=list(range(max(0,n-K),n))
    for name,h in [('loo',loo_h),('rand',rand_h),('tail',tail_h)]:
        new=rediffuse(ids,h)
        try: ast.parse(new)
        except Exception: continue
        if check(tid,new)=="pass": tally[name]+=1
    print(tid,n,tally,flush=True)
print("FINAL K=%d n=%d"%(K,len(fails)),tally)
