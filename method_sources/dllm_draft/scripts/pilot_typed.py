import json,ast,io,tokenize,torch,random
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
# TYPED POOL: char offsets of Python OP tokens + keywords in {and,or,not,if,else} + NUMBER literals.
# These are the "semantically load-bearing, low-aleatoric" cells a typed-AST runtime knows about.
SEMKW={'and','or','not','in','is'}
def typed_char_spans(src):
    spans=[]
    try:
        lines=src.splitlines(keepends=True)
        offs=[0]
        for l in lines: offs.append(offs[-1]+len(l))
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type==tokenize.OP or (t.type==tokenize.NAME and t.string in SEMKW) or t.type==tokenize.NUMBER:
                a=offs[t.start[0]-1]+t.start[1]; b=offs[t.end[0]-1]+t.end[1]
                spans.append((a,b))
    except Exception: pass
    return spans
def typed_token_idx(src,ids):
    spans=typed_char_spans(src); out=set()
    pos=0
    for i,t in enumerate(ids):
        s=tok.decode([t]); a=pos; b=pos+len(s); pos=b
        for (x,y) in spans:
            if a<y and x<b: out.add(i); break
    return out
MUT={'>':'<','<':'>','+':'-','-':'+','0':'1','1':'0','max':'min','min':'max','and':'or','or':'and','==':'!=','!=':'=='}
passing=sorted([(t,(v[0] if isinstance(v,list) else v)['solution']) for t,v in res.items()
   if (v[0] if isinstance(v,list) else v)['base_status']=='pass'])
random.seed(0); cases=[]
for tid,sol in passing:
    ids=tok(sol,add_special_tokens=False).input_ids
    cand=[(i,tok.decode([t])) for i,t in enumerate(ids) if tok.decode([t]).strip() in MUT]
    if not cand: continue
    i,d=random.choice(cand)
    nt=tok(d.replace(d.strip(),MUT[d.strip()]),add_special_tokens=False).input_ids
    if len(nt)!=1: continue
    mids=list(ids); mids[i]=nt[0]; ms=tok.decode(mids,skip_special_tokens=True)
    try: ast.parse(ms)
    except Exception: continue
    if ok(tid,ms): continue
    cases.append((tid,mids,ms,i))
    if len(cases)>=30: break
print("n",len(cases),flush=True)
def loo(ids,stride=8):
    n=len(ids);rows=[];meta=[]
    for off in range(stride):
        pos=list(range(off,n,stride)); r=list(ids)
        for j in pos: r[j]=MASK
        rows.append(r);meta.append(pos)
    with torch.no_grad(): lg=m(torch.tensor(rows,device="cuda"),"full",None).logits
    lg=torch.cat([lg[:,:1],lg[:,:-1]],dim=1);P=lg.softmax(-1)
    return {j:P[r,j,ids[j]].item() for r,pos in enumerate(meta) for j in pos}
def rd(ids,holes,steps):
    c=list(ids)
    for j in holes: c[j]=MASK
    x=torch.tensor([c],device="cuda")
    o=m.diffusion_generate(x,attention_mask=torch.ones_like(x),max_new_tokens=1,steps=steps,
        temperature=0.0,alg="entropy",alg_temp=0.,return_dict_in_generate=True)
    return tok.decode(o.sequences[0].tolist()[:len(ids)],skip_special_tokens=True)
T={'plainK4':0,'typedK4':0,'typedK8':0,'plainK8':0}
tl1=tt1=0
for tid,mids,ms,bug in cases:
    c=loo(mids); pool=typed_token_idx(ms,mids)
    plain=sorted(c,key=lambda j:c[j])
    typed=sorted([j for j in c if j in pool],key=lambda j:c[j])
    tl1+= (plain[0]==bug); tt1+= (bool(typed) and typed[0]==bug)
    for nm,h in [('plainK4',plain[:4]),('typedK4',typed[:4]),('plainK8',plain[:8]),('typedK8',typed[:8])]:
        if not h: continue
        if ok(tid,rd(mids,h,steps=max(2,len(h)))): T[nm]+=1
    print(tid,"|pool|",len(pool),"/",len(mids),"top1 plain",tl1,"typed",tt1,dict(T),flush=True)
n=len(cases)
print(f"FINAL n={n} top1_plain={tl1} top1_typed={tt1} "+" ".join(f"{k}={v}/{n}" for k,v in T.items()))
