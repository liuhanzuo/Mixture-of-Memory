import ast, glob, json, re, textwrap
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft"

def extract_python(text):
    fences=re.findall(r"```(?:python)?\s*\n?(.*?)```",text,flags=re.DOTALL|re.IGNORECASE)
    if fences: text=max(fences,key=len)
    else:
        u=re.search(r"```(?:python)?\s*\n?(.*)$",text,flags=re.DOTALL|re.IGNORECASE)
        if u: text=u.group(1)
    text=text.strip()
    st=[m.start() for m in re.finditer(r"(?m)^(?:async\s+def|def|from|import|@)\s*",text)]
    if st: text=text[min(st):]
    return text.rstrip()+("\n" if text else "")

def combine_AS_RUN(prompt,generated):
    e=extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+",e): return e
    body=e.strip() or "pass"
    return prompt.rstrip()+"\n"+textwrap.indent(body,"    ")+"\n"

def combine_FIXED(prompt,generated):
    """Dedent BEFORE extraction, so relative indentation survives.

    The archived stitch called extract_python() first; its .strip() removes the
    leading whitespace of the FIRST line only, which flattens line 1 to column 0
    while leaving lines 2..n at their original depth. textwrap.indent() then adds
    4 spaces uniformly -> line 1 at 4, line 2 at 8 -> 'unexpected indent'.
    A dedent applied AFTER extract_python is a no-op (common prefix is already 0).
    """
    e = extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+", e):
        return e
    body = textwrap.dedent(generated.replace("\t", "    ")).strip("\n").rstrip()
    if not body.strip():
        body = "pass"
    return prompt.rstrip()+"\n"+textwrap.indent(body,"    ")+"\n"

def parseable(t):
    try: ast.parse(t); return True
    except Exception: return False

prompts={}
for l in open(f"{R}/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"):
    if l.strip():
        d=json.loads(l); prompts[d["task_id"]]=d["prompt"]

out={}
for cell in ["he_c8","he_c32","he_c128"]:
    rows=[]
    for f in sorted(glob.glob(f"{R}/runs/a05_k1/{cell}/metrics.rank*.jsonl")):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    assert len(rows)==164 and len({r["task_id"] for r in rows})==164, cell
    a_ok=b_ok=fixed_by=genuine=changed=0
    for r in rows:
        raw=r.get("raw_output") or ""; p=prompts[r["task_id"]]
        A=combine_AS_RUN(p,raw); B=combine_FIXED(p,raw)
        pa,pb=parseable(A),parseable(B)
        a_ok+=pa; b_ok+=pb
        if A!=B: changed+=1
        if not pa:
            if pb: fixed_by+=1
            else: genuine+=1
    out[cell]=dict(n=164,as_run=a_ok,fixed=b_ok,fixed_by_dedent=fixed_by,genuine=genuine,text_changed=changed)
    print(f"{cell}: as_run={a_ok}/164  FIXED={b_ok}/164  rescued={fixed_by}  still_broken={genuine}  text_changed_on={changed}")
print()
print(json.dumps(out,indent=1))
