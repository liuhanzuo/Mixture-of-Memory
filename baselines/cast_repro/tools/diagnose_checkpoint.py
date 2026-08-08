#!/usr/bin/env python3
"""Run the audit section-5 diagnostic on a CAST checkpoint.

    python baselines/cast_repro/tools/diagnose_checkpoint.py \
        --ckpt outputs/cast_repro_ddp/prefinal.pt

Reports masked-vs-kept magnitude statistics per projection type and a verdict
against the broken run's numbers (ratio 0.294 / 21.5% below 1e-4).

Must be run on a PRE-finalization checkpoint: after finalize() the masked
entries are exactly zero by construction, so the metric says nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cast.diagnostics import exactness_report, magnitude_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="models/Llama--Llama2-7b")
    ap.add_argument("--project-root", default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    ap.add_argument("--scale-groups", type=int, default=2)
    ap.add_argument("--max-modules", type=int, default=None,
                    help="sample only the first N modules (faster)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--after-finalize", action="store_true",
                    help="also/instead report exact-N:M statistics")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = Path(args.project_root)
    from transformers import LlamaForCausalLM

    from cast import convert_llama_to_cast

    print(f"building skeleton from {args.model} (meta weights, then load ckpt)")
    model = LlamaForCausalLM.from_pretrained(
        str(root / args.model), torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    convert_llama_to_cast(model, scale_groups=args.scale_groups)

    print(f"loading {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob)
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  WARNING {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model.to(args.device)

    out = {"ckpt": args.ckpt, "step": blob.get("step")}
    if not args.after_finalize:
        rep = magnitude_report(model, max_modules=args.max_modules)
        out["magnitude"] = rep
        print("\n=== masked vs kept magnitude ===")
        print(json.dumps(rep["summary"], indent=2))
        print("\n=== ratio by projection (broken run: q .210 v .128 k .342 o .344 "
              "gate .346 up .345 down .344) ===")
        print(json.dumps(rep["by_projection"], indent=2))
    else:
        rep = exactness_report(model)
        out["exactness"] = rep
        print("\n=== exact N:M check ===")
        print(json.dumps(rep, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
