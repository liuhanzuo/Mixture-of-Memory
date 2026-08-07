"""RDS+ embeddings over the FULL 100K pool, sharded across 8 GPUs.
RDS+ (arXiv:2503.01807): position-weighted mean-pool of a frozen LM's last hidden states.
One forward pass per candidate, zero training."""
import json, torch, os, sys, numpy as np, time
from transformers import AutoTokenizer, AutoModel
MODEL=sys.argv[1]; OUT=sys.argv[2]; SHARD=int(sys.argv[3]); NSHARD=int(sys.argv[4])
ML=int(os.environ.get('MAXLEN','1024')); BS=int(os.environ.get('BS','32'))
rows=[json.loads(l) for l in open('pool.jsonl')]
idx=list(range(SHARD,len(rows),NSHARD))          # strided shard, [i::NSHARD]
rows=[rows[i] for i in idx]
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side='right'
m=AutoModel.from_pretrained(MODEL,torch_dtype=torch.bfloat16).cuda().eval()
txt=[r['prompt']+'\n'+r['resp'] for r in rows]
# sort by length for padding efficiency, remember inverse permutation
order=np.argsort([len(t) for t in txt]); inv=np.argsort(order)
txt=[txt[i] for i in order]
out=[]; t0=time.time()
for i in range(0,len(txt),BS):
    b=txt[i:i+BS]
    t=tok(b,return_tensors='pt',padding=True,truncation=True,max_length=ML)
    t={k:v.cuda() for k,v in t.items()}
    with torch.no_grad(): h=m(**t).last_hidden_state
    mask=t['attention_mask'].to(h.dtype)
    # position weight i/sum(i): later tokens weighted more (corrects causal-mask asymmetry)
    pos=torch.arange(1,h.shape[1]+1,device=h.device,dtype=h.dtype).unsqueeze(0)*mask
    w=(pos/pos.sum(1,keepdim=True).clamp(min=1e-6)).unsqueeze(-1)
    v=(h*w).sum(1)
    v=torch.nn.functional.normalize(v.float(),dim=-1)
    out.append(v.cpu().numpy().astype(np.float32))
    if (i//BS)%200==0: print(f'{i}/{len(txt)} {time.time()-t0:.0f}s',flush=True)
E=np.concatenate(out)[inv]
np.save(OUT,E)
json.dump(dict(model=MODEL,shard=SHARD,nshard=NSHARD,maxlen=ML,
               rows=[{k:r[k] for k in ('traj','env','step')} for r in rows]),
          open(OUT+'.meta.json','w'))
print('saved',E.shape,f'{time.time()-t0:.0f}s')
