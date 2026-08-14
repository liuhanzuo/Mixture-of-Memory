"""Prove the piqa OVERRIDE reproduces the archive's doc_hash AND prompt_hash.

check_piqa_source_matches_archive.py showed the raw parquet rows equal the
archived docs. That is necessary but not sufficient: what lm_eval actually scores
is the doc after `process_docs` and the rendered prompt after `doc_to_text`. This
script drives the override THROUGH lm_eval's own task pipeline and recomputes
lm_eval's own hashes (lm_eval.utils.hash_string, as used at
evaluator.py:578/586/587), then compares them to the archived samples file.

If these match for all 1838 docs, the substituted piqa cell is scoring the same
items with the same prompts as the archive, so its accuracy is comparable rather
than merely present. Runs CPU-only -- no model is loaded.
"""
import json
import sys

from lm_eval import utils
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import handle_non_serializable, hash_string

ARCHIVE = ("/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/cast_eval_spec_union9/"
           "dense_ref/lm_eval_out/__apdcephfs_wzc1__share_304376610__pighzliu_code__"
           "models__Llama--Llama2-7b/samples_piqa_2026-08-11T11-58-45.812255.jsonl")
YAML = ("/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/baselines/"
        "cast_repro/union9_taskoverride/piqa.yaml")

arch = {}
with open(ARCHIVE) as f:
    for line in f:
        r = json.loads(line)
        arch[r["doc_id"]] = r
print("archive piqa docs: %d" % len(arch))

cfg = utils.load_yaml_config(YAML)
task = list(get_task_dict([cfg], TaskManager()).values())[0]
docs = list(task.eval_docs)
print("override piqa docs: %d  (dataset_path=%s)" % (len(docs), task.config.dataset_path))

if len(docs) != len(arch):
    sys.exit("FAIL: doc count %d != archive %d" % (len(docs), len(arch)))

bad_doc, bad_prompt, bad_target = [], [], []
for i, d in enumerate(docs):
    a = arch[i]
    # Reproduced EXACTLY from lm_eval/evaluator.py:578-587 (0.4.8), including the
    # indent=2 / ensure_ascii=False kwargs -- any deviation changes the digest and
    # would make this check silently vacuous.
    dh = hash_string(json.dumps(d, indent=2, default=handle_non_serializable, ensure_ascii=False))
    ph = hash_string(task.doc_to_text(d))
    th = hash_string(str(task.doc_to_target(d)))
    if dh != a["doc_hash"]:
        bad_doc.append(i)
    if ph != a["prompt_hash"]:
        bad_prompt.append(i)
    if th != a["target_hash"]:
        bad_target.append(i)

print("doc_hash    mismatches: %d" % len(bad_doc))
print("prompt_hash mismatches: %d" % len(bad_prompt))
print("target_hash mismatches: %d" % len(bad_target))
for name, bad in (("doc", bad_doc), ("prompt", bad_prompt), ("target", bad_target)):
    if bad:
        i = bad[0]
        print("  first %s mismatch at doc %d" % (name, i))
        print("    archive doc_hash=%s prompt_hash=%s" % (arch[i]["doc_hash"], arch[i]["prompt_hash"]))
        print("    rendered prompt=%r" % task.doc_to_text(docs[i])[:200])

if bad_doc or bad_prompt or bad_target:
    print("VERDICT: piqa override is NOT hash-identical to the archive -> NOT admissible")
    sys.exit(1)
print("VERDICT: piqa override is hash-identical to the archive for all %d docs" % len(docs))
