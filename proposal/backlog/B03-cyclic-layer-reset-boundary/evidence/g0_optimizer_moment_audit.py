import json, os, sys, torch, hashlib, time
sys.path.insert(0, os.path.abspath("scripts"))
import train_olmo2_arch_probe2 as T

KF, NF = 7, 2
MP = "/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-0425-1B"
IN  = "outputs/b03_g0_fixture_1B_keep7fresh2/step2.pt"
OUT = "outputs/b03_g0_fixture_1B_keep7fresh2/step2_reset.pt"

def lid(n):
    if not n.startswith("model.layers."): return None
    try: return int(n.split(".")[2])
    except Exception: return None
reset_ids = set(range(KF, KF+NF))

shell,_,_ = T.build_olmo2_minimal(MP, KF, NF, torch.float32, transplant=False, is_main=False)
buckets = {"fresh_decay":[], "fresh_nodecay":[], "inh_decay":[], "inh_nodecay":[]}
for nm, pp in shell.named_parameters():
    cls = T._classify_param(nm, KF, False, random_trunk=False)
    pre = "fresh" if cls=="fresh" else "inh"
    buckets[f"{pre}_decay" if pp.ndim>=2 else f"{pre}_nodecay"].append(nm)
order = [nm for b in ("fresh_decay","fresh_nodecay","inh_decay","inh_nodecay") for nm in buckets[b] if buckets[b]]
del shell

res = {"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu_used": False}
for tag, path in (("input", IN), ("surgical", OUT)):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["optimizer_state"]["state"]
    r_zero = r_nonzero = n_zero = n_nonzero = 0
    r_step, n_step = set(), set()
    for i, nm in enumerate(order):
        e = st.get(i, st.get(str(i)))
        if e is None: continue
        z = bool(torch.all(e["exp_avg"] == 0).item() and torch.all(e["exp_avg_sq"] == 0).item())
        s = e.get("step")
        s = int(s.item()) if isinstance(s, torch.Tensor) else int(s)
        if lid(nm) is not None and lid(nm) in reset_ids:
            r_step.add(s); r_zero += z; r_nonzero += (not z)
        else:
            n_step.add(s); n_zero += z; n_nonzero += (not z)
    res[tag] = {
      "path": os.path.abspath(path),
      "bytes": os.path.getsize(path),
      "n_param_groups": len(ck["optimizer_state"]["param_groups"]),
      "reset_layer_params_with_both_moments_zero": r_zero,
      "reset_layer_params_with_a_nonzero_moment": r_nonzero,
      "nonreset_params_with_both_moments_zero": n_zero,
      "nonreset_params_with_a_nonzero_moment": n_nonzero,
      "reset_layer_step_values": sorted(r_step),
      "nonreset_step_values": sorted(n_step),
      "ckpt_step_field": ck.get("step"),
      "has_b03_reset_provenance": "b03_reset_provenance" in ck,
    }
    del ck
i, o = res["input"], res["surgical"]
res["assertions"] = {
  "input_reset_moments_were_NOT_already_zero": i["reset_layer_params_with_a_nonzero_moment"] > 0,
  "surgical_all_reset_moments_zero": o["reset_layer_params_with_both_moments_zero"] == 22
                                     and o["reset_layer_params_with_a_nonzero_moment"] == 0,
  "surgical_nonreset_moments_untouched": o["nonreset_params_with_a_nonzero_moment"]
                                          == i["nonreset_params_with_a_nonzero_moment"],
  "surgical_reset_step_is_zero": o["reset_layer_step_values"] == [0],
  "surgical_nonreset_step_preserved": o["nonreset_step_values"] == i["nonreset_step_values"],
  "group_count_preserved_on_disk": i["n_param_groups"] == o["n_param_groups"] == 4,
}
res["all_pass"] = all(res["assertions"].values())
print(json.dumps(res, indent=2))
