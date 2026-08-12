import ast,glob,json,re,textwrap
from collections import Counter
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft"
def extract_python(t):
    f=re.findall(r"```(?:python)?\s*\n?(.*?)```",t,flags=re.DOTALL|re.IGNORECASE)
    if f: t=max(f,key=len)
    else:
        u=re.search(r"```(?:python)?\s*\n?(.*)$",t,flags=re.DOTALL|re.IGNORECASE)
        if u: t=u.group(1)
    t=t.strip()
    s=[m.start() for m in re.finditer(r"(?m)^(?:async\s+def|def|from|import|@)\s*",t)]
    if s: t=t[min(s):]
    return t.rstrip()+("\n" if t else "")
def combine(p,g):
    e=extract_python(g)
    if re.search(r"(?m)^(?:async\s+def|def)\s+",e): return e
    return p.rstrip()+"\n"+textwrap.indent(e.strip() or "pass","    ")+"\n"
prompts={}
for l in open(f"{R}/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"):
    if l.strip():
        d=json.loads(l); prompts[d["task_id"]]=d["prompt"]
rows=[]
for f in sorted(glob.glob(f"{R}/runs/a05_k1/he_c128/metrics.rank*.jsonl")):
    for l in open(f):
        if l.strip(): rows.append(json.loads(l))
c=Counter(); lines=Counter()
for r in rows:
    s=combine(prompts[r["task_id"]], r.get("raw_output") or "")
    try: ast.parse(s)
    except SyntaxError as e:
        c[e.msg]+=1; lines[e.lineno]+=1
    except Exception as e: c[type(e).__name__]+=1
print("total unparseable:",sum(c.values()))
for k,v in c.most_common(): print(f"  {v:4d}  {k}")
indenty=sum(v for k,v in c.items() if "indent" in k.lower())
print("indentation-related:",indenty,"of",sum(c.values()))
print("min err line:",min(lines),"max:",max(lines))
