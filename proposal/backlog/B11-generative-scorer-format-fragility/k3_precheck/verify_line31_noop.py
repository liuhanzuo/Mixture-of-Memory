import sys, importlib, os
sys.path.insert(0, '/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg')
import babilong.metrics as M
print("module file :", M.__file__)
import hashlib
print("module md5  :", hashlib.md5(open(M.__file__,'rb').read()).hexdigest())
print("python      :", sys.version.split()[0])
print()

print("="*72)
print("PART 1 -- is line 31 EXECUTED? (line-level trace)")
print("="*72)
import inspect
src, start = inspect.getsourcelines(M.preprocess_output)
print("preprocess_output source starts at file line", start)
target_file = os.path.abspath(M.__file__)
hits = []
def tracer(frame, event, arg):
    if event == 'call':
        if os.path.abspath(frame.f_code.co_filename) == target_file and frame.f_code.co_name == 'preprocess_output':
            return tracer
        return None
    if event == 'line' and os.path.abspath(frame.f_code.co_filename) == target_file:
        hits.append(frame.f_lineno)
    return tracer

for probe in ["kitchen Question: Where is Mary? Answer: garden",
              "kitchen. Question: Where is Mary?",
              "plain kitchen"]:
    hits.clear()
    sys.settrace(tracer)
    M.preprocess_output(probe)
    sys.settrace(None)
    print(f"  input={probe!r:50s} lines executed = {hits}")
print("  -> line 31 IS executed on every input. It is NOT dead in the control-flow sense.")
print()

print("="*72)
print("PART 2 -- is line 31 a provable NO-OP? (value-level)")
print("="*72)
# instrument: capture the value entering and leaving line 31
def stepwise(output):
    v0 = output
    v1 = v0.lower()
    v2 = v1.split('.')[0]
    v3 = v2.split('<context>')[0]
    v4 = v3.split('<example>')[0]
    v5 = v4.split('Question')[0]      # line 31
    return v0,v1,v2,v3,v4,v5
for probe in ["kitchen Question: Where is Mary? Answer: garden",
              "QUESTION here",
              "Question at the very start"]:
    v = stepwise(probe)
    assert v[5] == M.preprocess_output(probe), "stepwise must reproduce the real function"
    print(f"  in ={probe!r}")
    print(f"    value entering line 31 = {v[4]!r}")
    print(f"    value leaving  line 31 = {v[5]!r}")
    print(f"    line 31 changed the value? {v[4] != v[5]}")
print()

print("="*72)
print("PART 3 -- WHY it can never change the value: str.lower() cannot emit 'Q'")
print("="*72)
bad = [cp for cp in range(0x110000) if 'Q' in chr(cp).lower()]
print("  codepoints whose .lower() contains ASCII 'Q':", bad)
bad2 = [cp for cp in range(0x110000) if any(u in chr(cp).lower() for u in 'QUESTION')]
print("  codepoints whose .lower() contains ANY of 'QUESTION':", bad2)
multi = [(hex(cp), chr(cp).lower()) for cp in range(0x110000)
         if len(chr(cp).lower()) > 1 and any('A' <= ch <= 'Z' for ch in chr(cp).lower())]
print("  multi-char lowerings that still contain an ASCII uppercase:", multi)
noniedem = [cp for cp in range(0x110000) if chr(cp).lower().lower() != chr(cp).lower()]
print("  codepoints where .lower() is NOT idempotent:", noniedem)
print("  Greek final-sigma context rule check: 'ODOS'.lower() ->", repr('ΟΔΟΣ'.lower()))
print()

print("="*72)
print("PART 4 -- CONTROL: lines 29/30 DO fire (so the defect is specific to line 31)")
print("="*72)
for probe in ["kitchen <CONTEXT> blah", "kitchen <EXAMPLE> blah",
              "kitchen <context> blah", "kitchen Question: blah"]:
    print(f"  preprocess_output({probe!r:28s}) = {M.preprocess_output(probe)!r}")
print()

print("="*72)
print("PART 5 -- the ONE-CHARACTER FIX, executed side by side")
print("="*72)
def fixed(output):
    output = output.lower()
    output = output.split('.')[0]
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    output = output.split('question')[0]   # <-- lowercase
    return output
for probe in ["kitchen Question: Where is Mary? Answer: garden",
              "the football is in the kitchen Question: Where is Mary?"]:
    print(f"  in       : {probe!r}")
    print(f"    current: {M.preprocess_output(probe)!r}")
    print(f"    fixed  : {fixed(probe)!r}")
print()

print("="*72)
print("PART 6 -- only in-repo caller of preprocess_output")
print("="*72)
import re
txt = open(M.__file__).read()
for i,l in enumerate(txt.splitlines(),1):
    if 'preprocess_output' in l:
        print(f"  metrics.py:{i}: {l.strip()}")
