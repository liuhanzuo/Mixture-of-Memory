#!/usr/bin/env bash
# Clone external repositories that are NOT vendored into this repo
# (because they are large, have their own .git, or are upstream-mirrored).
#
# After running this, your tree will look like:
#   ./MemLong/                                       (from Bui1dMySea/MemLong)
#   ./third_party/HMT-pytorch/                       (from OswaldHe/HMT-pytorch)
#   ./third_party/recurrent-memory-transformer/      (from booydar/recurrent-memory-transformer)
#   ./third_party/associative-recurrent-memory-transformer/  (from RodkinIvan/associative-recurrent-memory-transformer)
#   ./third_party/babilong-pkg/                       (from booydar/babilong, pinned scorer)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

clone_if_missing() {
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ] || [ -f "$dest/.git" ]; then
    echo "[third_party] $dest already exists, skipping"
    return
  fi
  echo "[third_party] Cloning $url -> $dest"
  mkdir -p "$(dirname "$dest")"
  git clone --depth 1 "$url" "$dest"
}

clone_pinned_if_missing() {
  local url="$1" dest="$2" revision="$3"
  if [ -d "$dest/.git" ] || [ -f "$dest/.git" ]; then
    echo "[third_party] $dest already exists, skipping pinned checkout"
    echo "[third_party] expected revision: $revision"
    return
  fi
  echo "[third_party] Cloning $url -> $dest at $revision"
  mkdir -p "$(dirname "$dest")"
  git clone --no-checkout "$url" "$dest"
  git -C "$dest" checkout --detach "$revision"
}

clone_if_missing https://github.com/Bui1dMySea/MemLong.git                              MemLong
clone_if_missing https://github.com/OswaldHe/HMT-pytorch.git                            third_party/HMT-pytorch
clone_if_missing https://github.com/booydar/recurrent-memory-transformer.git            third_party/recurrent-memory-transformer
clone_if_missing https://github.com/RodkinIvan/associative-recurrent-memory-transformer.git third_party/associative-recurrent-memory-transformer
clone_pinned_if_missing https://github.com/booydar/babilong.git                         third_party/babilong-pkg f09a184b43316a751d5059e13de7c557b6daca86

echo "[third_party] Done."
echo "NOTE: MemLong's own python env (memlong_env/) is NOT cloned — create your own venv if needed."
