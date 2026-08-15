import sys, hashlib
sys.path.insert(0,'/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg')
from babilong.metrics import TASK_LABELS, compare_answers, preprocess_output
import babilong.metrics as M
print("scorer md5:", hashlib.md5(open(M.__file__,'rb').read()).hexdigest())
print("python    :", sys.version.split()[0]); print()

def notrunc(target,output,question,labels):
    out=output.lower(); tgt=target.lower()
    labs={l.lower() for l in labels}
    inq={l for l in labs if l in question.lower()}
    ino={l for l in labs if l in out}-inq
    return tgt in ino and len(ino)==1

L=TASK_LABELS['qa1']; Q="Where is John?"
print("Q =",repr(Q),"  (contains no task label:",
      not any(l in Q.lower() for l in [x.lower() for x in L]),")")
print()

print("### DEMO SET 1 -- point 1, the dead guard")
cases=[("kitchen Question: Where is Mary? Answer: garden","intended 'kitchen ' -- guard should fire"),
       ("kitchen <CONTEXT> blah","control: line 29 DOES fire"),
       ("kitchen <EXAMPLE> blah","control: line 30 DOES fire")]
for s,why in cases:
    print(f"  preprocess_output({s!r})")
    print(f"    -> {preprocess_output(s)!r}      # {why}")
print()

print("### DEMO SET 2 -- point 1, the SCORE consequence")
def fixed_preprocess(o):
    o=o.lower(); o=o.split('.')[0]; o=o.split('<context>')[0]
    o=o.split('<example>')[0]; o=o.split('question')[0]; return o
def compare_fixed(target,output,question,labels):
    out=fixed_preprocess(output); tgt=target.lower()
    labs={l.lower() for l in labels}
    inq={l for l in labs if l in question.lower()}
    ino={l for l in labs if l in out}-inq
    return tgt in ino and len(ino)==1
s="kitchen Question: where is Mary? Answer: garden"
print(f"  raw = {s!r}   target='kitchen'")
print(f"    preprocess_output       -> {preprocess_output(s)!r}")
print(f"    compare_answers (as-is) -> {compare_answers('kitchen',s,Q,L)}")
print(f"    with 'Question'->'question' fix:")
print(f"    preprocess (fixed)      -> {fixed_preprocess(s)!r}")
print(f"    compare (fixed)         -> {compare_fixed('kitchen',s,Q,L)}")
assert compare_answers('kitchen',s,Q,L) is False and compare_fixed('kitchen',s,Q,L) is True
print("    VERIFIED: dead guard causes a FALSE NEGATIVE; fix is not score-neutral.")
print()

print("### DEMO SET 3 -- point 2, truncation DESTROYS correct answers")
for s,label in [("The answer is A. kitchen","enumerated / letter-then-answer"),
                ("John moved several times. He is in the kitchen","reason-then-answer")]:
    c=compare_answers("kitchen",s,Q,L); nt=notrunc("kitchen",s,Q,L)
    print(f"  raw = {s!r}   ({label})")
    print(f"    preprocess_output -> {preprocess_output(s)!r}")
    print(f"    canonical={c}   notrunc={nt}   <- truncation is what killed it: {c is False and nt is True}")
    assert c is False and nt is True, "DEMO FAILED to exhibit the mechanism"
print()

print("### DEMO SET 4 -- point 2, truncation RESCUES correct answers")
s="kitchen. Question: Where is Mary? Answer: garden"
c=compare_answers("kitchen",s,Q,L); nt=notrunc("kitchen",s,Q,L)
print(f"  raw = {s!r}")
print(f"    preprocess_output -> {preprocess_output(s)!r}")
print(f"    canonical={c}   notrunc={nt}   <- truncation is what saved it: {c is True and nt is False}")
assert c is True and nt is False
print()

print("### DEMO SET 5 -- point 2, truncation MANUFACTURES a correct answer (false positive)")
s="kitchen is wrong. the answer is garden"
c=compare_answers("kitchen",s,Q,L); nt=notrunc("kitchen",s,Q,L)
print(f"  raw = {s!r}   target='kitchen', model actually answered 'garden'")
print(f"    preprocess_output -> {preprocess_output(s)!r}")
print(f"    canonical={c}   notrunc={nt}   <- false positive created by line 27: {c is True and nt is False}")
assert c is True and nt is False
print()

print("### NEGATIVE CONTROL -- the string previously in our record that does NOT show the mechanism")
s="Choices: A. In the kitchen B. In the garden. The answer is kitchen."
c=compare_answers("kitchen",s,Q,L); nt=notrunc("kitchen",s,Q,L)
print(f"  raw = {s!r}")
print(f"    preprocess_output -> {preprocess_output(s)!r}")
print(f"    canonical={c}   notrunc={nt}")
print(f"    -> BOTH False. It dies on the UNIQUENESS requirement (both 'kitchen' and 'garden'")
print(f"       survive without truncation), NOT on truncation. Correctly EXCLUDED from the issue.")
assert c is False and nt is False
print()
print("ALL DEMOS EXECUTED AND ASSERTED. Every demo exhibits the mechanism it is presented as showing.")
