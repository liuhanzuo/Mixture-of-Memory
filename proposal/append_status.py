#!/usr/bin/env python3
"""Byte-prefix-preserving append to a STATUS.json.

The schema's rule (LIFECYCLE_SCHEMA.md sec 0) is that the ONLY allowed byte change is
turning the closing '}' into ',' and adding new keys. This helper enforces that
literally: it asserts the original text is an exact byte prefix of the result,
so no existing key -- including A03-style prose in `status` -- can be touched
even by accident (e.g. by a json.dump that reindents).

Usage: append_status.py <status.json> <new_keys.json> <indent>
"""
import json, sys

path, addpath, indent = sys.argv[1], sys.argv[2], int(sys.argv[3])
orig = open(path, encoding="utf-8").read()
add = json.load(open(addpath, encoding="utf-8"))

before = json.loads(orig)
for k in add:
    if k in before:
        sys.exit(f"REFUSING: key {k!r} already exists -- append-only forbids overwrite")

body = orig.rstrip()
assert body.endswith("}"), "file does not end with }"
body = body[:-1].rstrip()
if body.endswith(","):
    body = body[:-1]

pad = " " * indent
chunks = []
for k, v in add.items():
    txt = json.dumps(v, ensure_ascii=False, indent=indent)
    txt = ("\n" + txt).replace("\n", "\n" + pad)[1:]  # re-indent nested lines
    chunks.append(f"{pad}{json.dumps(k, ensure_ascii=False)}: {txt}")
new = body + ",\n" + ",\n".join(chunks) + "\n}\n"

# --- the guarantees ---------------------------------------------------------
after = json.loads(new)                       # 1. still valid JSON
assert new.startswith(body), "prefix broken"  # 2. original bytes preserved
for k, v in before.items():
    assert k in after and after[k] == v, f"existing key {k!r} changed"
assert list(after) == list(before) + list(add), "key order changed"

open(path, "w", encoding="utf-8").write(new)
print(f"OK {path}: {len(before)} keys -> {len(after)} keys; added {list(add)}")
