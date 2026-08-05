#!/usr/bin/env python3
"""CPU self-test for --random_trunk in scripts/train_olmo2_arch_probe2.py.

Why this exists (Paper C confound): the A4 (keep front j + graft K fresh) vs A3
(--from_scratch, same depth) contrast is confounded, because --from_scratch also
random-inits model.embed_tokens / model.norm / lm_head. A ~1.6M-token SQuAD SFT
cannot learn a 100352-row vocab embedding plus an output map, so "A4 > A3" could
be fully explained by A3 having no usable tokeniser interface -- nothing to do
with inheriting trunk weights. --random_trunk is the matched control: trunk
random, readout inherited.

Test strategy: build a TINY but *structurally faithful* OLMo-2 base on disk
(hidden_size=128, intermediate=256, 4 heads, vocab=256, but the REAL default
depth num_hidden_layers=16 so the production A4 path `--keep_front_layers 14
--n_fresh_layers 2` is exercised verbatim, and the real Olmo2ForCausalLM +
Olmo2Config classes / the real _init_weights are used). ~2.9M params -> a couple
of seconds on CPU. Base weights are then offset by +1.0 so no random init can
ever coincide with them, making every torch.equal assertion meaningful.

Run:
  /opt/conda/envs/torch-base/bin/python scripts/selftest_random_trunk.py
Exit code 0 == all assertions pass. No GPU, no training, no repo writes
(the tiny base lives in a tempdir that is removed on exit).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(PROJECT_ROOT, "scripts")
for _p in (PROJECT_ROOT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_olmo2_arch_probe2 as T  # noqa: E402

TRAINER = os.path.join(SCRIPTS, "train_olmo2_arch_probe2.py")

KEEP, FRESH = 14, 2           # the production A4 default
TOTAL = KEEP + FRESH          # 16
SEED = 42
NONLAYER = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")


def _ok(msg):
    print(f"  [ok] {msg}")


def make_tiny_base(path):
    """Save a tiny-but-real 16-layer Olmo2ForCausalLM as the pretrained base.

    Every base tensor gets +1.0 so it can never collide with a fresh random init
    (RMSNorm weights become 2.0, projections become ~1.0+N(0,0.02)); that makes
    "transplanted == base" and "random != base" both decidable by torch.equal."""
    cfg = Olmo2Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=TOTAL, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, rope_theta=500000.0, rms_norm_eps=1e-6,
        initializer_range=0.02, tie_word_embeddings=False, torch_dtype="float32",
    )
    torch.manual_seed(1234)               # deliberately NOT the build seed
    base = Olmo2ForCausalLM(cfg).to(torch.float32)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(1.0)
    base.save_pretrained(path, safe_serialization=True)
    sd = base.state_dict()
    n_layer_keys = sum(1 for k in sd if k.startswith("model.layers.0."))
    assert n_layer_keys == T.N_TENSORS_PER_LAYER, (
        f"tiny base has {n_layer_keys} tensors in layer 0, expected "
        f"{T.N_TENSORS_PER_LAYER} -> the tiny base is not layout-faithful"
    )
    assert set(k for k in sd if not k.startswith("model.layers.")) == set(NONLAYER)
    print(f"[base] tiny OLMo-2 base saved to {path}: "
          f"{cfg.num_hidden_layers}L hidden={cfg.hidden_size} vocab={cfg.vocab_size} "
          f"params={sum(p.numel() for p in base.parameters())/1e6:.2f}M "
          f"({T.N_TENSORS_PER_LAYER} tensors/layer, {T.N_NONLAYER_KEYS} non-layer keys)")
    del base
    return {k: v.clone() for k, v in sd.items()}


def build(base_path, mode):
    """Reproduce EXACTLY what main() does for each arm (same seed, same flags,
    same do_transplant expression: `do_transplant = not from_scratch`)."""
    T.set_seed(SEED)
    random_trunk = mode == "random_trunk"
    from_scratch = mode == "from_scratch"
    model, cfg, sanity = T.build_olmo2_minimal(
        base_path, KEEP, FRESH, torch.float32,
        transplant=(not from_scratch), is_main=False, random_trunk=random_trunk,
    )
    return model, cfg, sanity


def fake_args(mode):
    return argparse.Namespace(
        keep_front_layers=KEEP, n_fresh_layers=FRESH,
        from_scratch=(mode == "from_scratch"),
        random_trunk=(mode == "random_trunk"),
        freeze_front=False,
        lr=1e-4, min_lr=1e-5, lr_inherited=2e-5, min_lr_inherited=2e-6,
        weight_decay=0.1,
    )


class _DDPish(torch.nn.Module):
    """Wrap so named_parameters() are 'module.'-prefixed, as after DDP wrap --
    guards the 'module.' strip in _classify_param (a previously-fixed bug)."""

    def __init__(self, m):
        super().__init__()
        self.module = m


def main():
    tmp = tempfile.mkdtemp(prefix="selftest_random_trunk_")
    base_path = os.path.join(tmp, "tiny_olmo2_base")
    try:
        base_sd = make_tiny_base(base_path)

        models, cfgs, sanities = {}, {}, {}
        for mode in ("keep_front", "random_trunk", "from_scratch"):
            models[mode], cfgs[mode], sanities[mode] = build(base_path, mode)
            print(f"[build] mode={mode:<13} num_hidden_layers="
                  f"{cfgs[mode].num_hidden_layers} sanity={sanities[mode]}")

        def layer_keys(lid):
            return [k for k in base_sd if k.startswith(f"model.layers.{lid}.")]

        def sd(mode):
            return models[mode].state_dict()

        # ---- 1. --random_trunk: readout bit-identical to base, trunk not ------
        print("\n[1] --random_trunk: embed/norm/lm_head == base (elementwise), "
              "trunk != base")
        rt = sd("random_trunk")
        for k in NONLAYER:
            assert torch.equal(rt[k], base_sd[k]), (
                f"random_trunk {k} != base elementwise "
                f"(max|d|={(rt[k]-base_sd[k]).abs().max().item():.3e})")
            _ok(f"{k} torch.equal(base) -> True  (shape {tuple(rt[k].shape)})")
        for lid in range(TOTAL):
            for k in layer_keys(lid):
                assert not torch.equal(rt[k], base_sd[k]), (
                    f"random_trunk {k} EQUALS base -> trunk was not randomised")
        _ok(f"all {TOTAL} layers ({TOTAL * T.N_TENSORS_PER_LAYER} tensors) "
            f"differ from base, incl. the front {KEEP}")
        assert sanities["random_trunk"]["random_trunk"] is True
        assert sanities["random_trunk"]["n_copied"] == T.N_NONLAYER_KEYS
        assert sanities["random_trunk"]["transplant_max_abs_diff"] == 0.0
        _ok("sanity dict: random_trunk=True n_copied=3 max_abs_diff=0.0")

        # ---- 2. --from_scratch: readout also random (flags really differ) -----
        print("\n[2] --from_scratch: embed/norm/lm_head ALSO != base "
              "(proves the two flags differ)")
        fs = sd("from_scratch")
        for k in NONLAYER:
            assert not torch.equal(fs[k], base_sd[k]), (
                f"from_scratch {k} equals base -> from_scratch inherited it?!")
            _ok(f"{k} torch.equal(base) -> False")
        for lid in range(TOTAL):
            for k in layer_keys(lid):
                assert not torch.equal(fs[k], base_sd[k])
        _ok(f"all {TOTAL} layers differ from base too")
        assert sanities["from_scratch"] == {"transplanted": False}
        _ok("sanity dict: transplanted=False (no base load at all)")

        # ---- 3. default A4 path: front KEEP == base, tail FRESH != base -------
        print(f"\n[3] --keep_front_layers {KEEP} --n_fresh_layers {FRESH} "
              f"(default A4): front {KEEP} == base, tail {FRESH} != base")
        kf = sd("keep_front")
        for k in NONLAYER:
            assert torch.equal(kf[k], base_sd[k]), f"keep_front {k} != base"
        _ok("embed/norm/lm_head torch.equal(base) -> True")
        for lid in range(KEEP):
            for k in layer_keys(lid):
                assert torch.equal(kf[k], base_sd[k]), (
                    f"keep_front layer {lid} tensor {k} != base")
        _ok(f"layers 0..{KEEP - 1} torch.equal(base) -> True "
            f"({KEEP * T.N_TENSORS_PER_LAYER} tensors)")
        for lid in range(KEEP, TOTAL):
            for k in layer_keys(lid):
                assert not torch.equal(kf[k], base_sd[k]), (
                    f"keep_front FRESH layer {lid} tensor {k} EQUALS base")
        _ok(f"layers {KEEP}..{TOTAL - 1} differ from base (fresh tail)")

        # ---- 4. identical shape/params/depth across the three modes ----------
        print("\n[4] identical depth / param count / tensor set across modes "
              "(arms are comparable)")
        ref_n = sum(p.numel() for p in models["keep_front"].parameters())
        ref_keys = sorted(sd("keep_front").keys())
        for mode in models:
            n = sum(p.numel() for p in models[mode].parameters())
            assert cfgs[mode].num_hidden_layers == TOTAL, (
                f"{mode} num_hidden_layers={cfgs[mode].num_hidden_layers} != {TOTAL}")
            assert n == ref_n, f"{mode} n_params={n} != {ref_n}"
            assert sorted(sd(mode).keys()) == ref_keys, f"{mode} tensor set differs"
            shapes_match = all(sd(mode)[k].shape == sd("keep_front")[k].shape
                               for k in ref_keys)
            assert shapes_match, f"{mode} has a shape mismatch"
            _ok(f"{mode:<13} num_hidden_layers={cfgs[mode].num_hidden_layers} "
                f"n_params={n} n_tensors={len(ref_keys)} shapes=match")

        # ---- 5. same init function: random_trunk == from_scratch on the trunk -
        print("\n[5] random_trunk uses the SAME fresh init as the fresh blocks "
              "(same seed -> bit-identical)")
        for lid in range(KEEP, TOTAL):
            for k in layer_keys(lid):
                assert torch.equal(rt[k], kf[k]), (
                    f"random_trunk vs keep_front differ on FRESH layer {lid} ({k}) "
                    f"-> not the same init path")
        _ok(f"random_trunk layers {KEEP}..{TOTAL - 1} == keep_front's fresh tail, "
            f"bit-identical")
        for lid in range(TOTAL):
            for k in layer_keys(lid):
                assert torch.equal(rt[k], fs[k]), (
                    f"random_trunk vs from_scratch differ on layer {lid} ({k})")
        _ok(f"random_trunk trunk == from_scratch trunk for all {TOTAL} layers "
            f"-> random_trunk == from_scratch + inherited readout, exactly")
        _ok("=> the trunk is drawn from Olmo2ForCausalLM(cfg) post_init, the very "
            "same init the fresh graft blocks use (no separate distribution)")

        # ---- 6. param groups -------------------------------------------------
        print("\n[6] build_param_groups: group names + params per mode "
              "(DDP-style 'module.' prefixed names)")
        for mode in ("keep_front", "random_trunk", "from_scratch"):
            wrapped = _DDPish(models[mode])
            args = fake_args(mode)
            groups = T.build_param_groups(wrapped, args, is_main=False)
            named = {}
            for name, pp in wrapped.named_parameters():
                cls = T._classify_param(name, args.keep_front_layers,
                                        args.from_scratch,
                                        random_trunk=args.random_trunk)
                named.setdefault(cls, 0)
                named[cls] += pp.numel()
            lrs = {}
            for g in groups:
                lrs[g["base_lr"]] = lrs.get(g["base_lr"], 0) + sum(
                    p.numel() for p in g["params"])
            print(f"  mode={mode}")
            print(f"    class totals : " + ", ".join(
                f"{c}={v}" for c, v in sorted(named.items())))
            print(f"    groups       : " + ", ".join(
                f"{len(g['params'])}t/{sum(p.numel() for p in g['params'])}p"
                f"@lr={g['base_lr']:.0e},wd={g['weight_decay']}" for g in groups))
            trunk = sum(pp.numel() for n, pp in wrapped.named_parameters()
                        if ".model.layers." in n)
            readout = sum(pp.numel() for n, pp in wrapped.named_parameters()
                          if ".model.layers." not in n)
            assert trunk + readout == ref_n
            if mode == "random_trunk":
                assert named.get("fresh", 0) == trunk, (
                    f"random_trunk fresh={named.get('fresh')} != trunk={trunk}")
                assert named.get("inherited", 0) == readout, (
                    f"random_trunk inherited={named.get('inherited')} != "
                    f"readout={readout}")
                _ok(f"random_trunk: trunk {trunk}p -> 'fresh' @lr={args.lr:.0e}; "
                    f"embed+norm+lm_head {readout}p -> 'inherited' "
                    f"@lr={args.lr_inherited:.0e}")
            elif mode == "from_scratch":
                assert named.get("inherited", 0) == 0 and named["fresh"] == ref_n
                _ok(f"from_scratch: single 'fresh' bucket, all {ref_n}p "
                    f"@lr={args.lr:.0e} (unchanged)")
            else:
                front = sum(pp.numel() for n, pp in wrapped.named_parameters()
                            if ".model.layers." in n
                            and int(n.split(".model.layers.")[1].split(".")[0]) < KEEP)
                embed_norm = sum(pp.numel() for n, pp in wrapped.named_parameters()
                                 if n.endswith("model.embed_tokens.weight")
                                 or n.endswith("model.norm.weight"))
                assert named["inherited"] == front + embed_norm, (
                    f"keep_front inherited={named['inherited']} != "
                    f"front+embed+norm={front + embed_norm}")
                _ok(f"keep_front: front{KEEP}+embed+norm {front + embed_norm}p -> "
                    f"'inherited'; fresh tail+lm_head {ref_n - front - embed_norm}p "
                    f"-> 'fresh' (unchanged)")

        # ---- 7. mutual exclusion / CLI wiring --------------------------------
        print("\n[7] CLI: --random_trunk is mutually exclusive with "
              "--from_scratch / --freeze_front")
        for extra in (["--from_scratch"], ["--freeze_front"]):
            cp = subprocess.run(
                [sys.executable, TRAINER, "--data_path", "/nonexistent.npy",
                 "--output_dir", os.path.join(tmp, "out"), "--random_trunk"] + extra,
                capture_output=True, text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            assert cp.returncode == 2, (
                f"--random_trunk {extra[0]} exited {cp.returncode}, expected 2 "
                f"(argparse error)\nstdout={cp.stdout[-800:]}\nstderr={cp.stderr[-800:]}")
            assert "mutually exclusive" in cp.stderr or "would freeze RANDOM" in cp.stderr
            _ok(f"--random_trunk {extra[0]} -> exit 2 with explanation: "
                f"{cp.stderr.strip().splitlines()[-1][:90]}")
        cp = subprocess.run(
            [sys.executable, TRAINER, "--data_path", "/nonexistent.npy",
             "--output_dir", os.path.join(tmp, "out"), "--model_path", base_path,
             "--random_trunk", "--dry_run_build"],
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        assert cp.returncode == 0, (
            f"--random_trunk --dry_run_build failed ({cp.returncode})\n{cp.stderr[-1500:]}")
        assert "randtrunk" in cp.stderr, f"arm name missing:\n{cp.stderr[-800:]}"
        _ok("--random_trunk --dry_run_build exits 0, arm="
            + [l.split("arm=")[1].split()[0] for l in cp.stderr.splitlines()
               if "arm=" in l][0])

        # ---- 8. arch_meta / ckpt meta records random_trunk -------------------
        print("\n[8] arch_meta.json + ckpt meta record random_trunk")
        src = open(TRAINER).read()
        assert src.count('"random_trunk": bool(') == 2, (
            "expected random_trunk recorded in BOTH _save's ckpt meta and "
            "arch_meta.json")
        _ok('"random_trunk": bool(...) present in ckpt meta and arch_meta.json')

        print("\n=== ALL RANDOM_TRUNK SELFTESTS PASSED ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
