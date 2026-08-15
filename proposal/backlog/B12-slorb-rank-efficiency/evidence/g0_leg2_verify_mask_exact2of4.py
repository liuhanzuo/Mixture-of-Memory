"""Verify the EXACT-2:4 property of the mask rung A is built on, all 224 tensors, CPU."""
import json, sys, torch
sys.path.insert(0, "baselines/cast_repro/tools")
from emit_slorb_ladder import nm_2_4_hard, PROJECTIONS, AUX_SUFFIXES

CK = ("/apdcephfs_wzc1/share_304376610/pighzliu_code/out_llama/"
      "models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/"
      "model_best_lm_eval.pt")
blob = torch.load(CK, map_location="cpu", weights_only=False, mmap=True)
sd = blob["model_state_dict"]

n_t = mask_elems = mask_ones = mask_viol = 0
wm_zeros = wm_elems = wm_viol = 0
tail_cols = 0
for key in sd:
    if any(key.endswith(s) for s in AUX_SUFFIXES):
        continue
    if not (key.endswith(".weight") and any(f".{p}." in key for p in PROJECTIONS)):
        continue
    base = key[: -len("weight")]
    W = sd[key].float()
    m = nm_2_4_hard(sd[base + "mask"].float())
    out_d, in_d = m.shape
    tail_cols += in_d - (in_d // 4) * 4
    n_t += 1
    mask_elems += m.numel(); mask_ones += int((m != 0).sum())
    g = (m != 0).reshape(out_d, in_d // 4, 4).sum(-1)
    mask_viol += int((g != 2).sum())
    Wm = W * m
    wm_elems += Wm.numel(); wm_zeros += int((Wm == 0).sum())
    gz = (Wm != 0).reshape(out_d, in_d // 4, 4).sum(-1)
    wm_viol += int((gz != 2).sum())

r = {
  "n_in_scope_tensors": n_t,
  "mask_elements": mask_elems, "mask_ones": mask_ones,
  "mask_ones_fraction": mask_ones / mask_elems,
  "mask_exact_2of4_violations": mask_viol,
  "columns_outside_any_group_of_4": tail_cols,
  "W_times_mask_elements": wm_elems,
  "W_times_mask_zeros": wm_zeros,
  "W_times_mask_zero_fraction": wm_zeros / wm_elems,
  "W_times_mask_exact_2of4_violations": wm_viol,
}
print(json.dumps(r, indent=2))
json.dump(r, open("/tmp/b12_mask_verify.json", "w"), indent=2)
