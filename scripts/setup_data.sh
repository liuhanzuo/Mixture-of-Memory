#!/usr/bin/env bash
# Stub: bring up the large data assets required for training / evaluation.
#
# The big tokenized corpora are NOT version-controlled (they are 10s-100s of GB).
# Two recommended ways to obtain them:
#
#   (A) Re-tokenize from source (slow but reproducible):
#         python -m llmshearing.data.tokenize_all_files ...
#       See data/README.md for the exact command set used originally.
#
#   (B) Copy from your cluster's shared storage, e.g.:
#         rsync -avh /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/ ./data/
#
# Expected layout after this script:
#   data/armt_pg19_real_tokenized_full/      (~82 GB)
#   data/slimpajama-6b/                      (~14 GB)
#   data/pg19_train.jsonl                    (~11 GB)
#   data/slimpajama_chunks_4096.npy          (~12 GB)
#   data/mag_train_*.jsonl                   (~1 GB total)
#
# Smoke-test data (small, can be regenerated quickly) is shipped in-repo:
#   data/armt_pg19_real_tokenized_smoke/     (~44 MB)
#   data/armt_smoke_tokenized_pg19_like/     (~17 MB)
#   data/mag_eval_generated.jsonl            (~2 MB)
set -euo pipefail

cat <<'EOF'
[setup_data] This is a placeholder script.

Big training corpora are NOT distributed via git. Please either:
  (A) follow data/README.md to regenerate them from raw sources, or
  (B) copy from a shared storage location, e.g.:
      rsync -avh <REMOTE>:/path/to/Mixture-of-Memory/data/ ./data/

If you only want to run smoke tests, the in-repo data under data/*_smoke* is enough.
EOF
