#!/usr/bin/env bash
# Build the ISOLATED S5 package dir on diskB: a fresh PRISTINE-ANCHOR (99631a8)
# checkout of the landmark llama/ package in a physically separate tree, then
# apply the S5 single_layer_mem patch onto it (never touching S4b's live file),
# then re-run the CPU regression smoke FROM that isolated tree.
# CPU-only. No GPU.
set -uo pipefail
R=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
NESTED=$R/external/landmark-attention          # nested git repo (HEAD d963e50 = anchor 99631a8)
S5TREE=$R/external/landmark_s5_tree            # <-- isolated S5 package dir
PATCHDIR=$R/external/landmark/s5_patch
PY=$R/external/landmark_venv/bin/python

echo "===== [0] sanity: nested anchor HEAD md5 ====="
( cd "$NESTED" && git show HEAD:llama/llama_mem.py | md5sum )   # expect 99631a8...

rm -rf "$S5TREE"
mkdir -p "$S5TREE/llama"

echo "===== [1] materialize PRISTINE anchor llama/ package into isolated tree ====="
# Pull every file of the llama/ package from the nested repo HEAD (anchor),
# NOT from the working tree (which is S4b 8ef7994). git archive = clean HEAD.
( cd "$NESTED" && git archive HEAD llama/ | tar -x -C "$S5TREE" )
echo "anchor llama_mem.py md5 in isolated tree (pre-patch):"
md5sum "$S5TREE/llama/llama_mem.py"

echo "===== [2] apply S5 patch onto the isolated anchor tree ====="
bash "$PATCHDIR/apply_s5.sh" "$S5TREE/llama"
RC=$?
if [ $RC -ne 0 ]; then echo "APPLY FAILED rc=$RC"; exit $RC; fi
echo "post-patch single_layer_mem refs:"
grep -c single_layer_mem "$S5TREE/llama/llama_mem.py" "$S5TREE/llama/llama_landmark_config.py"

# Stage a copy of the PRISTINE anchor (separate dir) to compare against.
ANCHORTREE=$R/external/_s5_anchor_ref
rm -rf "$ANCHORTREE"; mkdir -p "$ANCHORTREE"
( cd "$NESTED" && git archive HEAD llama/ | tar -x -C "$ANCHORTREE" )

# Drop the smoke worker into both trees.
cp "$PATCHDIR/s5_smoke_worker.py" "$S5TREE/llama/s5_smoke_worker.py"
cp "$PATCHDIR/s5_smoke_worker.py" "$ANCHORTREE/llama/s5_smoke_worker.py"

export CUDA_VISIBLE_DEVICES=""
export LM_S5_DEBUG_COUNTER=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

echo "===== [3] ANCHOR forward (pristine, from _s5_anchor_ref) ====="
( cd "$ANCHORTREE/llama" && "$PY" s5_smoke_worker.py none "$R/external/_s5_anchor_ref/anchor_none.pt" ) || { echo "ANCHOR RUN FAILED"; exit 21; }

echo "===== [4] S5 isolated-tree forward, single_layer_mem=None (regression) ====="
( cd "$S5TREE/llama" && "$PY" s5_smoke_worker.py none "$S5TREE/s5_none.pt" ) || { echo "S5-none RUN FAILED"; exit 22; }

echo "===== [5] S5 isolated-tree forward, single_layer_mem=16 (isolate L16) ====="
( cd "$S5TREE/llama" && "$PY" s5_smoke_worker.py 16 "$S5TREE/s5_l16.pt" ) || { echo "S5-l16 RUN FAILED"; exit 23; }

echo "===== [6] COMPARE (from isolated tree) ====="
"$PY" - "$ANCHORTREE/anchor_none.pt" "$S5TREE/s5_none.pt" "$S5TREE/s5_l16.pt" <<'PYEOF'
import sys, torch
a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2]); c = torch.load(sys.argv[3])
la, lb, lc = a["logits"], b["logits"], c["logits"]
mad_reg = (la - lb).abs().max().item()
mad_l16 = (la - lc).abs().max().item()
print(f"[regression] pristine-anchor(none) vs ISOLATED-S5(none) max-abs-diff = {mad_reg:.3e}")
print(f"[isolate]    pristine-anchor(none) vs ISOLATED-S5(L16)  max-abs-diff = {mad_l16:.3e}  (>0 expected)")
print(f"[counters anchor] {a['counters']}")
print(f"[counters S5 none] {b['counters']}")
print(f"[counters S5 L16 ] {c['counters']}")
ok = True
if mad_reg > 1e-6:
    print("FAIL: regression diff too large"); ok = False
cn = c["counters"]
if cn:
    if cn.get("grouped_layers") != [16]:
        print(f"FAIL: expected only [16] grouped, got {cn.get('grouped_layers')}"); ok = False
    others = [i for i in range(32) if i != 16]
    if cn.get("plain_layers") != others:
        print(f"FAIL: expected plain = all-but-16, got {cn.get('plain_layers')}"); ok = False
if mad_l16 <= 1e-6:
    print("WARN: L16 gating produced no change vs anchor (unexpected)")
print("ISOLATED_SMOKE_RESULT:", "PASS" if ok else "FAIL")
PYEOF
