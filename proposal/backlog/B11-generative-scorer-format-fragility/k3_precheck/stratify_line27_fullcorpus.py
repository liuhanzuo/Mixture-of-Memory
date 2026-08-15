#!/usr/bin/env python
"""FULL-CORPUS (deterministic, no sampling) stratified effect of metrics.py:27.

Supersedes the record's 500-random-CSV / 46241-item figure with an exhaustive pass over
EVERY qa*.csv under babilong_results on BOTH disks. 0 GPU.
"""
import csv, glob, json, os, sys, hashlib
sys.path.insert(0,'/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg')
from babilong.metrics import TASK_LABELS, compare_answers

def score_notrunc(target,output,question,labels):
    out=output.lower(); tgt=target.lower()
    labs={l.lower() for l in labels}
    inq={l for l in labs if l in question.lower()}
    ino={l for l in labs if l in out}-inq
    if "," in tgt and len(tgt)>3:
        subs=tgt.split(",")
        return all(t in ino for t in subs) and len(ino)==len(subs)
    return tgt in ino and len(ino)==1

def is_list_format(raw):          # verbatim from analyze_a02_truncation_ablation.py
    low=raw.strip().lower()
    if low.startswith(("choices","options")): return True
    h=low[:60]
    return ("a." in h and "b." in h) or ("a)" in h and "b)" in h)

ROOTS = {"wzc1":"/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/babilong_results",
         "zwfy6":"/tmp/b11corpus/zwfy6"}
files=[]
for disk,root in ROOTS.items():
    for f in sorted(glob.glob(os.path.join(root,"*","qa*.csv"))):
        files.append((disk,f))
print("CSV files found:", len(files), " (by disk:", {d:sum(1 for x,_ in files if x==d) for d in ROOTS},")")

# dedupe identical files ACROSS disks by content hash -- the two disks share history
seen={}; dupes=0
uniq=[]
for disk,f in files:
    h=hashlib.md5(open(f,'rb').read()).hexdigest()
    if h in seen: dupes+=1; continue
    seen[h]=f; uniq.append((disk,f,h))
print(f"content-identical duplicates removed: {dupes}   unique CSVs kept: {len(uniq)}")

n=harm=help_=0; skipped_files=0; skipped_rows=0
S={}   # (is_list) -> [n, harm, help, canon, notrunc]
def bump(key,c,nt):
    a=S.setdefault(key,[0,0,0,0,0])
    a[0]+=1; a[3]+=c; a[4]+=nt
    if c==1 and nt==0: a[2]+=1     # truncation HELPED (rescued)
    if c==0 and nt==1: a[1]+=1     # truncation HARMED (destroyed)
task_of=lambda p: os.path.basename(p).split("_")[0]
for disk,f,h in uniq:
    t=task_of(f)
    if t not in TASK_LABELS: skipped_files+=1; continue
    try: rows=list(csv.DictReader(open(f)))
    except Exception: skipped_files+=1; continue
    if not rows or not {"output","target","question"} <= set(rows[0]): skipped_files+=1; continue
    for r in rows:
        raw,q,tg = r.get("output"),r.get("question"),r.get("target")
        if raw is None or q is None or tg is None: skipped_rows+=1; continue
        c=int(bool(compare_answers(tg,raw,q,TASK_LABELS[t])))
        nt=int(bool(score_notrunc(tg,raw,q,TASK_LABELS[t])))
        n+=1
        if c==0 and nt==1: harm+=1
        if c==1 and nt==0: help_+=1
        bump(is_list_format(raw),c,nt)
        bump(("task",t),c,nt)
print(f"files skipped (unknown task / unparseable / wrong schema): {skipped_files}   rows skipped: {skipped_rows}")
print()
def show(label,a):
    N,H,P,C,NT=a
    net=100.0*(P-H)/N if N else float('nan')
    print(f"  {label:34s} n={N:6d}  destroyed(HARM)={H:5d} ({100.0*H/N:5.2f}%)  rescued(HELP)={P:5d} ({100.0*P/N:5.2f}%)  "
          f"net_of_KEEPING_trunc={net:+7.2f}pp  canon={100.0*C/N:6.2f}  notrunc={100.0*NT/N:6.2f}")
    return dict(stratum=label,n=N,destroyed=H,rescued=P,net_pp=round(net,4),
                canon_acc=round(100.0*C/N,4),notrunc_acc=round(100.0*NT/N,4))
print("="*130); print("FULL CORPUS, deterministic (every qa*.csv on both disks, content-deduped)"); print("="*130)
out={}
out['aggregate']=show("ALL",[n,harm,help_,S[True][3]+S[False][3],S[True][4]+S[False][4]])
print()
print("  STRATIFIED BY OUTPUT FORMAT (is_list_format, verbatim from the A02 script):")
out['LIST']=show("is_list=1 (LIST format)",S[True])
out['NONLIST']=show("is_list=0 (non-LIST)",S[False])
print()
print("  STRATIFIED BY TASK:")
out['by_task']=[show(f"task={k[1]}",v) for k,v in sorted((kk,vv) for kk,vv in S.items() if isinstance(kk,tuple))]
print()
print("="*130)
sf=(out['LIST']['net_pp']<0)!=(out['NONLIST']['net_pp']<0)
print(f"  SIGN DIFFERS BETWEEN LIST AND NON-LIST : {sf}")
print(f"  LIST     net = {out['LIST']['net_pp']:+.2f} pp (n={out['LIST']['n']})")
print(f"  NON-LIST net = {out['NONLIST']['net_pp']:+.2f} pp (n={out['NONLIST']['n']})")
print(f"  AGGREGATE net= {out['aggregate']['net_pp']:+.2f} pp (n={out['aggregate']['n']}) <- the number that hides it")
out['sign_differs_between_strata']=sf
out['n_csv_files_scanned']=len(uniq); out['n_csv_files_found']=len(files)
out['duplicates_removed']=dupes; out['files_skipped']=skipped_files
json.dump(out,open('/tmp/b11out/b11_fullcorpus.json','w'),indent=1)
print("\nwrote /tmp/b11out/b11_fullcorpus.json")
