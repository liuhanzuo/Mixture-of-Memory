#!/usr/bin/env python3
"""PRIMARY data path: re-tokenize Dolmino-Mix-1124 with the LLaMA-2 tokenizer.

WHY THIS IS NEEDED.  CAST trains LLaMA models on Dolmino-Mix-1124 (Sec. VI-A
"Data").  On disk we have:

  data/dolmino-mix-1124-llama3/  469B tokens but tokenizer=Llama3-8B,
                                 vocab_size=128000  -> WRONG TOKENIZER.
                                 LLaMA-2 has vocab 32000; the id spaces are
                                 unrelated, so these tokens are unusable.
  data/dolmino-flan-heavy/       LLaMA-2 tokenized (vocab 32000) BUT only
                                 499.5M train tokens and a FLAN-heavy custom
                                 mixture (38.9% FLAN) that is not the Dolmino
                                 default mix. Too small for 7.5B and not the
                                 paper's distribution.
  data/c4_llama/                 21.7B LLaMA-2 tokens of C4 -> the FALLBACK.

So a paper-setting run needs Dolmino re-tokenized with the LLaMA-2 tokenizer.
The raw Dolmino download is NOT on disk (data/dolmino-mix-1124-raw/ does not
exist), so this script has to download first.

    # step 1: download the raw subsets (needs the HTTP proxy, see below)
    python prepare_dolmino_llama2.py --stage download --raw-dir data/dolmino-mix-1124-raw

    # step 2: tokenize to a flat uint32 .bin (LLaMA-2 vocab 32000 fits in uint16
    # but the shared scripts/download_and_tokenize_dolmino.py hardcodes uint32;
    # train_cast_llama.py --data-dtype=auto reads dtype from metadata.json, so
    # either dtype is fine at load time. uint32 doubles disk but keeps the
    # already-validated tokenize pipeline unchanged.)
    python prepare_dolmino_llama2.py --stage tokenize \
        --raw-dir data/dolmino-mix-1124-raw \
        --out-dir data/dolmino-mix-1124-llama2 \
        --target-tokens 9000000000 --workers 64

COST ESTIMATE (do not run casually):
  * 7500 steps x 256 x 4096 = 7.86B tokens needed; target >= 8B, 9B for slack.
  * Raw text for ~9B LLaMA-2 tokens is ~35-40 GB compressed.
    Download at ~50 MB/s through the proxy: ~15-20 min, but Dolmino subset
    files are large and the proxy is the bottleneck -- budget 1-3 h.
  * Tokenization: the existing LLaMA-3 run in this repo did 469B tokens in
    3306 s wall with the fast tokenizer (metadata.json "elapsed_seconds"),
    i.e. ~142M tokens/s aggregate on this box. 9B tokens => ~1-2 min of pure
    tokenizer time; realistically 20-40 min including I/O and the JSON parse.
  * Output size: 9B x 2 bytes = 18 GB.
  TOTAL: ~2-4 h wall, dominated by the download.

The paper's mixture: Sec. VI-A only says "Dolmino-Mix-1124"; it does not give
subset weights. Using the dataset's natural proportions is the neutral choice
and must be recorded as an implementation choice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROXY = {
    "http_proxy": "http://hy-proxy.woa.com:3128",
    "https_proxy": "http://hy-proxy.woa.com:3128",
    "all_proxy": "http://hy-proxy.woa.com:3128",
    "no_proxy": "mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local",
}

# Natural Dolmino-Mix-1124 subsets. Weighted by the dataset's own proportions
# unless --ratios is given. [implementation_choice: the paper gives no weights]
DEFAULT_SUBSETS = ("dclm", "flan", "math", "pes2o", "wiki", "stackexchange")


def stage_download(args) -> int:
    os.environ.update(PROXY)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed in this interpreter", file=sys.stderr)
        return 2

    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    patterns = [f"data/{s}/*" for s in args.subsets]
    print(f"downloading allenai/dolmino-mix-1124 subsets={list(args.subsets)} -> {raw}")
    print(f"allow_patterns={patterns}")
    t0 = time.time()
    snapshot_download(
        repo_id="allenai/dolmino-mix-1124",
        repo_type="dataset",
        local_dir=str(raw),
        allow_patterns=patterns,
        max_workers=args.workers,
    )
    print(f"done in {time.time()-t0:.0f}s")
    return 0


def stage_tokenize(args) -> int:
    """Tokenize raw jsonl/jsonl.gz into one flat uint32 .bin + a metadata sidecar.

    NB: the underlying scripts/download_and_tokenize_dolmino.py hardcodes uint32.
    train_cast_llama.py --data-dtype=auto reads dtype from metadata.json, so the
    downstream loader adapts automatically. Do not claim uint16 in wrapper output.

    Reuses the repo's existing, already-validated pipeline rather than a new
    implementation: scripts/download_and_tokenize_dolmino.py supports
    --local_input_dir and a Llama-2 tokenizer path. This wrapper just pins the
    arguments so the output is unambiguous.
    """
    root = Path(args.project_root)
    script = root / "scripts" / "download_and_tokenize_dolmino.py"
    if not script.exists():
        print(f"ERROR: {script} not found", file=sys.stderr)
        return 2
    tok = root / args.tokenizer
    if not tok.exists():
        print(f"ERROR: tokenizer {tok} not found", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        str(script),
        "--tokenizer_path", str(tok),
        "--local_input_dir", str(Path(args.raw_dir).resolve()),
        "--output_dir", str(Path(args.out_dir).resolve()),
        "--output_format", "numpy",
        "--max_seq_len", "4096",
        "--num_workers", str(args.workers),
    ]
    print("RUN:", " ".join(cmd))
    if args.dry_run:
        print("\n--dry-run: not executing.")
        print(f"Expected output: {args.out_dir}/train.bin  (~{args.target_tokens*4/2**30:.0f} GiB, uint32)")
        return 0
    import subprocess

    return subprocess.call(cmd, env={**os.environ, **PROXY})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["download", "tokenize", "plan"])
    ap.add_argument("--raw-dir", default="data/dolmino-mix-1124-raw")
    ap.add_argument("--out-dir", default="data/dolmino-mix-1124-llama2")
    ap.add_argument("--tokenizer", default="models/Llama--Llama2-7b")
    ap.add_argument("--project-root", default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    ap.add_argument("--subsets", nargs="*", default=list(DEFAULT_SUBSETS))
    ap.add_argument("--target-tokens", type=int, default=9_000_000_000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.stage == "plan":
        need = 7500 * 256 * 4096
        print(json.dumps({
            "tokens_needed_7500_steps": need,
            "target_tokens": args.target_tokens,
            "slack": args.target_tokens - need,
            "output_bytes_uint32": args.target_tokens * 4,
            "on_disk_alternatives": {
                "data/dolmino-mix-1124-llama3": "469B tokens but LLaMA-3 vocab 128000 -> UNUSABLE",
                "data/dolmino-flan-heavy": "LLaMA-2 vocab but only 499.5M tokens, FLAN-heavy custom mix",
                "data/c4_llama": "21.7B LLaMA-2 tokens of C4 -> FALLBACK, not paper setting",
            },
            "estimated_wall_hours": "2-4 (download dominated)",
        }, indent=2))
        return 0
    if args.stage == "download":
        return stage_download(args)
    return stage_tokenize(args)


if __name__ == "__main__":
    raise SystemExit(main())
