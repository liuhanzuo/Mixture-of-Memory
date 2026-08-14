"""Check whether an alternative piqa source reproduces the ARCHIVED piqa docs exactly.

The union-9 archive (2026-08-11) loaded piqa from the script-built cache
    /root/.cache/huggingface/datasets/ybisk___piqa/plain_text/1.1.0/6c611c1a...
That cache is GONE from every node (LOCAL/.212 caches wiped by the 08-13 restart;
.73 has only a DIFFERENT parquet-built cache `default/0.0.0/142c5123...`; .82/.104
have no piqa at all). And `datasets 5.0.1` refuses `piqa.py` outright
("Dataset scripts are no longer supported"), so the archive's load path is not
reconstructible at all.

This script asks the only question that matters: does some available source
yield the SAME 1838 docs in the SAME order, so that a re-scored piqa cell is
comparable to the archived one? Compared field-by-field, not by accuracy.
"""
import json
import sys

ARCHIVE = ("/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/cast_eval_spec_union9/"
           "dense_ref/lm_eval_out/__apdcephfs_wzc1__share_304376610__pighzliu_code__"
           "models__Llama--Llama2-7b/samples_piqa_2026-08-11T11-58-45.812255.jsonl")

arch = {}
with open(ARCHIVE) as f:
    for line in f:
        r = json.loads(line)
        arch[r["doc_id"]] = r
print("archive piqa docs:", len(arch))

import datasets

CANDIDATES = [
    ("hub ybisk/piqa parquet (Mixture-of-Memory/.hf_cache)",
     dict(path="parquet",
          data_files={"validation": "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/"
                                    ".hf_cache/hub/datasets--ybisk--piqa/snapshots/"
                                    "142c51238b3ca2bc61e9a075913871b8b600e8e1/plain_text/validation/0000.parquet"},
          split="validation")),
    ("local arrow baber___piqa",
     dict(path="arrow",
          data_files={"validation": "/apdcephfs_wzc1/share_304376610/pighzliu_code/data/hf_datasets/"
                                    "baber___piqa/default/0.0.0/"
                                    "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1/piqa-validation.arrow"},
          split="validation")),
]

for name, kw in CANDIDATES:
    print("\n=== candidate:", name)
    try:
        ds = datasets.load_dataset(**kw)
    except Exception as e:
        print("  LOAD FAIL %s: %s" % (type(e).__name__, str(e)[:200]))
        continue
    print("  loaded n=%d cols=%s" % (len(ds), ds.column_names))
    if len(ds) != len(arch):
        print("  LENGTH MISMATCH: %d vs archive %d" % (len(ds), len(arch)))
    n_cmp = min(len(ds), len(arch))
    diffs = []
    for i in range(n_cmp):
        a = arch[i]["doc"]
        b = ds[i]
        for k in ("goal", "sol1", "sol2", "label"):
            if a.get(k) != b.get(k):
                diffs.append((i, k, repr(a.get(k))[:80], repr(b.get(k))[:80]))
    print("  field diffs: %d" % len(diffs))
    for d in diffs[:5]:
        print("    doc %d field %s archive=%s cand=%s" % d)
    if not diffs and len(ds) == len(arch):
        print("  >>> IDENTICAL to archive piqa docs (all %d, all fields, same order)" % n_cmp)
