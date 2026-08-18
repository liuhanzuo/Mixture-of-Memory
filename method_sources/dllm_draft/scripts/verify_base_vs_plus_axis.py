"""Re-grade R5 qwen_fim with BASE tests only (the pilot's grading protocol) vs PLUS
(the R5 protocol). This isolates the base-vs-plus leg of the .7638 vs .950 gap.
CPU only. Uses local evalplus 0.3.1."""
import json, re, collections, os, sys
os.environ.setdefault('HF_DATASETS_OFFLINE','1')
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval import untrusted_check

probs=get_human_eval_plus(); gt=get_groundtruth(probs,get_human_eval_plus_hash(),[])
sols=[json.loads(l) for l in open('/tmp/r5/qwen_fim.solutions.jsonl')]
print('solutions', len(sols))

rows=[json.loads(l) for l in open('/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/data/infilling/HumanEval-SingleLineInfilling.jsonl')]
per=collections.defaultdict(dict)
for r in rows:
    mm=re.match(r'SingleLineInfilling/(HumanEval/\d+)/L(\d+)$', r['task_id'])
    per[mm.group(1)][int(mm.group(2))]=r
def pick(base,K):
    spans=per[base]; bytrue={}
    for r in spans.values(): bytrue.setdefault(r['prompt'].count('\n'), r)
    tl=sorted(bytrue); picked=[]; last=-10
    for x in tl:
        if x>=last+2: picked.append(x); last=x
        if len(picked)==K: break
    return (picked, bytrue) if len(picked)==K else (None,None)

order=sorted(per, key=lambda s:int(s.split('/')[1]))
target={}   # task_id (SingleLine...) -> base
for b in order:
    p,bytrue=pick(b,1)
    if p is None: continue
    target[bytrue[p[0]]['task_id']]=b
print('k=1 target rows', len(target))

byid={s['task_id']:s for s in sols}
def grade(tid,src,which):
    p=probs[tid]
    inp = p['base_input'] if which=='base' else p['base_input']+p['plus_input']
    exp = gt[tid]['base'] if which=='base' else gt[tid]['base']+gt[tid]['plus']
    tl  = gt[tid]['base_time'] if which=='base' else gt[tid]['base_time']+gt[tid]['plus_time']
    try:
        st,_=untrusted_check("humaneval",src,inp,p['entry_point'],expected=exp,atol=p['atol'],
            ref_time=tl,fast_check=False,min_time_limit=1.0,gt_time_limit_factor=4.0)
        return st=="pass"
    except Exception as e:
        return False

elig=[t for t in target if t in byid]
elig.sort(key=lambda t:int(t.split('/')[2]))
print('gradeable', len(elig))
res={}
for i,t in enumerate(elig):
    s=byid[t]; b=target[t]
    res[t]=(grade(b,s['solution'],'base'), grade(b,s['solution'],'plus'))
    if (i+1)%40==0: print('  ...%d'%(i+1), flush=True)

def rep(label, subset):
    bb=sum(1 for t in subset if res[t][0])/len(subset)
    pp=sum(1 for t in subset if res[t][1])/len(subset)
    print('  %-36s BASE-only=%.4f   PLUS=%.4f   n=%d   (base-plus gap %+0.4f)'%(label,bb,pp,len(subset),bb-pp))

print()
print('=== R5 qwen_fim on the k=1 pilot hole, base-only vs plus grading ===')
rep('ALL 164 eligible', elig)
rep('first 40 (AR pilot cap)', elig[:40])
rep('first 60 (diffusion pilot cap)', elig[:60])
