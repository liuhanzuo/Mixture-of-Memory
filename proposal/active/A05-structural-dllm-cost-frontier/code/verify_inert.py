"""Verify (b): are mask_expansion / delete_eos_token really inert? CPU only, no model weights."""
import sys, importlib.util, glob, json
CKPT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft/models/DreamOn-v0-7B"
sys.path.insert(0, CKPT)
spec = None
for cand in glob.glob(f"{CKPT}/*generation_utils*.py") + glob.glob(f"{CKPT}/*configuration*.py"):
    print("found module file:", cand)
# import the generation config class directly from the checkpoint's remote code
import importlib.util as iu
p = f"{CKPT}/generation_utils.py"
spec = iu.spec_from_file_location("dreamon_genutils", p)
m = iu.module_from_spec(spec)
sys.modules["dreamon_genutils"] = m
spec.loader.exec_module(m)
cls = None
for name in dir(m):
    if "GenerationConfig" in name:
        print("class:", name)
        cls = getattr(m, name)
cfg = cls()
print("has mask_expansion attr BEFORE:", hasattr(cfg, "mask_expansion"))
print("has delete_eos_token attr BEFORE:", hasattr(cfg, "delete_eos_token"))
unused = cfg.update(**{"mask_expansion": True, "delete_eos_token": True})
print("update() returned as UNUSED:", unused)
print("has mask_expansion attr AFTER :", hasattr(cfg, "mask_expansion"))
print("has delete_eos_token attr AFTER :", hasattr(cfg, "delete_eos_token"))
# sanity: a real parameter IS consumed
unused2 = cfg.update(**{"temperature": 0.2})
print("update(temperature=0.2) unused:", unused2, "-> cfg.temperature =", getattr(cfg,"temperature",None))
