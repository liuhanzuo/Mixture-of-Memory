#!/usr/bin/env python
"""B11 K3 precheck -- claim (b): is first-period truncation's effect SIGN-DEPENDENT?

Two-armed, per-item, 0 GPU. Arm CANON = babilong.metrics.compare_answers as shipped.
Arm NOTRUNC = identical except metrics.py:27 (output.split('.')[0]) removed. Uniqueness
requirement retained in both, so choice-lists still score 0 (no chance inflation).

EFFECT OF TRUNCATION on an item = canon - notrunc.
  +1 = truncation RESCUED a correct answer   (canon=1, notrunc=0)
  -1 = truncation DESTROYED a correct answer (canon=0, notrunc=1)
   0 = no effect
Reported STRATIFIED, with n per stratum, because the aggregate hides the mechanism.
"""
import csv, glob, json, os, sys
from pathlib import Path
sys.path.insert(0,'/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg')
from babilong.metrics import TASK_LABELS, compare_answers

W = Path(os.environ.get("A02_W","/tmp/b11W"))
ARMS = {"A0":"a02_dvr_babilong_j0_top12","A2":"a02_rtax_babilong_A2_j6",
        "A3":"a02_rtax_babilong_A3_j9","A4":"a02_babilong_c2_j12_readlora",
        "A5":"a02_rtax_babilong_A5_j18"}
NSHARD=8

def score_notrunc(target,output,question,labels):
    out=output.lower(); tgt=target.lower()
    labs={l.lower() for l in labels}
    inq={l for l in labs if l in question.lower()}
    ino={l for l in labs if l in out}-inq
    if "," in tgt and len(tgt)>3:
        subs=tgt.split(",")
        return all(t in ino for t in subs) and len(ino)==len(subs)
    return tgt in ino and len(ino)==1

def is_list_format(raw):
    low=raw.strip().lower()
    if low.startswith(("choices","options")): return True
    h=low[:60]
    return ("a." in h and "b." in h) or ("a)" in h and "b)" in h)

def load_indexed(sub,task,length):
    out={}
    files=glob.glob(str(W/"babilong_results"/sub/f"{task}_{length}_*shard*of{NSHARD}.csv"))
    assert len(files)==NSHARD, f"SHARD_INCOMPLETE {len(files)}/{NSHARD} {sub}/{task}/{length}"
    for f in files:
        s=int(Path(f).stem.split("shard")[1].split("of")[0])
        for r,row in enumerate(csv.DictReader(open(f))): out[s+r*NSHARD]=row
    assert len(out)==100, f"N={len(out)}"
    return out

recs=[]
for task in ("qa1","qa2","qa5"):
    for length in ("16k","32k"):
        per={a:load_indexed(s,task,length) for a,s in ARMS.items()}
        idx=sorted(set.intersection(*[set(d) for d in per.values()]))
        assert len(idx)==100, len(idx)
        for a,d in per.items():
            for i in idx:
                raw,q,tg=d[i]["output"],d[i]["question"],d[i]["target"]
                c=int(bool(compare_answers(tg,raw,q,TASK_LABELS[task])))
                nt=int(bool(score_notrunc(tg,raw,q,TASK_LABELS[task])))
                # does the output contain a period at all (i.e. can line 27 do anything)?
                lo=raw.lower()
                pre=lo.split('.')[0]
                recs.append(dict(task=task,length=length,cell=f"{task}|{length}",arm=a,item=i,
                                 canon=c, notrunc=nt, effect=c-nt,
                                 is_list=int(is_list_format(raw)),
                                 has_period=int('.' in lo),
                                 truncation_active=int(pre!=lo)))
print(f"per-item records: {len(recs)}   (= 5 arms x 6 cells x 100 items = {5*6*100})")
assert len(recs)==3000
json.dump(recs, open('/tmp/b11out/b11_peritem.json','w'))

def tab(rows,label):
    n=len(rows); resc=sum(1 for r in rows if r['effect']==1); dest=sum(1 for r in rows if r['effect']==-1)
    net=100.0*(resc-dest)/n if n else float('nan')
    ca=100.0*sum(r['canon'] for r in rows)/n if n else float('nan')
    nta=100.0*sum(r['notrunc'] for r in rows)/n if n else float('nan')
    print(f"  {label:42s} n={n:5d}  rescued={resc:4d}  destroyed={dest:4d}  "
          f"net_effect_of_trunc={net:+7.2f}pp   canon={ca:5.2f}  notrunc={nta:5.2f}")
    return dict(stratum=label,n=n,rescued=resc,destroyed=dest,net_pp=round(net,4),
                canon_acc=round(ca,4),notrunc_acc=round(nta,4))

out={}
print()
print("="*112); print("S0  AGGREGATE (what a single number would say)"); print("="*112)
out['aggregate']=[tab(recs,"ALL")]

print()
print("="*112); print("S1  STRATIFIED BY OUTPUT FORMAT HABIT (is_list): the mechanism stratum"); print("="*112)
out['by_is_list']=[tab([r for r in recs if r['is_list']==1],"is_list=1 (choice-list habit)"),
                   tab([r for r in recs if r['is_list']==0],"is_list=0 (non-list)")]

print()
print("="*112); print("S2  STRATIFIED BY TASK (qa1/qa2 = high list-rate, qa5 = low)"); print("="*112)
out['by_task']=[tab([r for r in recs if r['task']==t],f"task={t}") for t in ("qa1","qa2","qa5")]

print()
print("="*112); print("S3  is_list x task (does the sign follow FORMAT or follow TASK?)"); print("="*112)
o=[]
for t in ("qa1","qa2","qa5"):
    for L in (1,0):
        o.append(tab([r for r in recs if r['task']==t and r['is_list']==L],f"task={t}, is_list={L}"))
out['by_task_x_islist']=o

print()
print("="*112); print("S4  STRATIFIED BY ARM (the arms whose ordering is at stake)"); print("="*112)
out['by_arm']=[tab([r for r in recs if r['arm']==a],f"arm={a}") for a in ARMS]

print()
print("="*112); print("S5  BY ARM x is_list  (A4 = high list rate, A5 = low)"); print("="*112)
o=[]
for a in ARMS:
    for L in (1,0):
        o.append(tab([r for r in recs if r['arm']==a and r['is_list']==L],f"arm={a}, is_list={L}"))
out['by_arm_x_islist']=o

print()
print("="*112); print("S6  PER CELL (the 6 cells; sign per cell)"); print("="*112)
o=[]
for t in ("qa1","qa2","qa5"):
    for l in ("16k","32k"):
        o.append(tab([r for r in recs if r['cell']==f"{t}|{l}"],f"cell={t}|{l}"))
out['by_cell']=o

print()
print("="*112); print("S7  list-format RATE per arm x cell (the covariate that drives the sign)"); print("="*112)
print(f"  {'cell':12s}" + "".join(f"{a:>9s}" for a in ARMS))
lf={}
for t in ("qa1","qa2","qa5"):
    for l in ("16k","32k"):
        row={a:100.0*sum(r['is_list'] for r in recs if r['cell']==f'{t}|{l}' and r['arm']==a)/100 for a in ARMS}
        lf[f"{t}|{l}"]=row
        print(f"  {t+'|'+l:12s}" + "".join(f"{row[a]:9.1f}" for a in ARMS))
out['list_format_rate_pct']=lf

print()
print("="*112); print("VERDICT"); print("="*112)
agg=out['aggregate'][0]; l1=out['by_is_list'][0]; l0=out['by_is_list'][1]
print(f"  aggregate net effect of truncation      : {agg['net_pp']:+.2f} pp  (n={agg['n']})")
print(f"  within is_list=1                        : {l1['net_pp']:+.2f} pp  (n={l1['n']})")
print(f"  within is_list=0                        : {l0['net_pp']:+.2f} pp  (n={l0['n']})")
signflip = (l1['net_pp']<0) != (l0['net_pp']<0)
print(f"  SIGN DIFFERS BETWEEN THE TWO STRATA     : {signflip}")
print(f"  aggregate hides it (agg sign == is_list=1 sign, opposite of is_list=0): "
      f"{(agg['net_pp']<0)==(l1['net_pp']<0)}")
json.dump(out, open('/tmp/b11out/b11_stratified.json','w'), indent=1)
print("\nwrote /tmp/b11out/b11_stratified.json and /tmp/b11out/b11_peritem.json")
