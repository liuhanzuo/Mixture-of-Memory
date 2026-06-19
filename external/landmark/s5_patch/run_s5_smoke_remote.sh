#!/usr/bin/env bash
# S5 CPU-only smoke on diskB, fully isolated from the shared llama/ package
# (so it cannot collide with landmark-s5's S4b). Builds two throwaway package
# copies under a unique staging dir, runs the worker on each, compares logits.
set -uo pipefail
R=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
SRC=$R/external/landmark-attention/llama
STAGE=$R/external/_s5_smoke
PY=$R/external/landmark_venv/bin/python

rm -rf "$STAGE/pkg_anchor" "$STAGE/pkg_s5" "$STAGE"/*.pt
mkdir -p "$STAGE/pkg_anchor" "$STAGE/pkg_s5"

# pkg_anchor = pristine clone of the shared llama package (the anchor).
cp -r "$SRC"/* "$STAGE/pkg_anchor/" 2>/dev/null
# pkg_s5 = same, then overlay my modified files (staged separately, NOT in llama/).
cp -r "$SRC"/* "$STAGE/pkg_s5/" 2>/dev/null
cp "$STAGE/_incoming/llama_mem.py"            "$STAGE/pkg_s5/llama_mem.py"
cp "$STAGE/_incoming/llama_landmark_config.py" "$STAGE/pkg_s5/llama_landmark_config.py"
cp "$STAGE/_incoming/s5_smoke_worker.py"      "$STAGE/pkg_anchor/s5_smoke_worker.py"
cp "$STAGE/_incoming/s5_smoke_worker.py"      "$STAGE/pkg_s5/s5_smoke_worker.py"

export CUDA_VISIBLE_DEVICES=""
export LM_S5_DEBUG_COUNTER=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

echo "===== [1/3] ANCHOR forward (single_layer_mem unsupported) ====="
( cd "$STAGE/pkg_anchor" && "$PY" s5_smoke_worker.py none "$STAGE/anchor_none.pt" ) || { echo "ANCHOR RUN FAILED"; exit 11; }

echo "===== [2/3] S5 forward, single_layer_mem=None (regression proof) ====="
( cd "$STAGE/pkg_s5" && "$PY" s5_smoke_worker.py none "$STAGE/s5_none.pt" ) || { echo "S5-none RUN FAILED"; exit 12; }

echo "===== [3/3] S5 forward, single_layer_mem=1 (only L1 grouped) ====="
( cd "$STAGE/pkg_s5" && "$PY" s5_smoke_worker.py 1 "$STAGE/s5_l1.pt" ) || { echo "S5-l1 RUN FAILED"; exit 13; }

echo "===== COMPARE ====="
"$PY" - "$STAGE" <<'PYEOF'
import sys, torch
stage = sys.argv[1]
a = torch.load(f"{stage}/anchor_none.pt")
b = torch.load(f"{stage}/s5_none.pt")
c = torch.load(f"{stage}/s5_l1.pt")

la, lb, lc = a["logits"], b["logits"], c["logits"]
mad_reg = (la - lb).abs().max().item()
print(f"[regression] anchor(none) vs S5(none) max-abs-diff = {mad_reg:.3e}")
print(f"[regression] shapes a={tuple(la.shape)} b={tuple(lb.shape)} c={tuple(lc.shape)}")
print(f"[counters anchor] {a['counters']}  supports_s5={a['supports_s5']}")
print(f"[counters S5 none] {b['counters']}  supports_s5={b['supports_s5']}")
print(f"[counters S5 L1  ] {c['counters']}  supports_s5={c['supports_s5']}")

mad_l1 = (la - lc).abs().max().item()
print(f"[effect] anchor(none) vs S5(L1) max-abs-diff = {mad_l1:.3e}  (should be >0: gating changes output)")

ok = True
if mad_reg > 1e-6:
    print("FAIL: regression diff too large"); ok = False
cn = c["counters"]
if cn:
    if cn.get("grouped_layers") != [1]:
        print(f"FAIL: expected only layer [1] grouped, got {cn.get('grouped_layers')}"); ok = False
    if cn.get("plain_layers") != [0,2,3]:
        print(f"FAIL: expected plain layers [0,2,3], got {cn.get('plain_layers')}"); ok = False
if mad_l1 <= 1e-6:
    print("WARN: L1 gating produced no change vs anchor (unexpected)")
print("SMOKE_RESULT:", "PASS" if ok else "FAIL")
PYEOF
