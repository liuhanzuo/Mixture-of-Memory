"""Vendored SnapKV (FasterDecoding/SnapKV).

Upstream commit ``e216ddc84c5bd210378cbdbbba12ba02102aa640`` (see
``src/baselines/PROVENANCE.md``). The FasterDecoding ``SnapKVCluster`` in
``snapkv_utils_fasterdecoding.py`` is the *canonical* SnapKV algorithm but is
NOT GQA-aware, so on Qwen3 (32 q-heads / 8 kv-heads) the actually-driven cluster
is the GQA-aware ``SnapKVCluster`` vendored from Zefan-Cai in
``src/baselines/pyramidkv/pyramidkv_utils.py`` (byte-identical observation-window
scoring, plus a kv-head-granular gather so the compressed cache matches Qwen3's
``DynamicLayer`` layout). The reference ``llama_hijack_4_37_reference.py`` targets
transformers 4.37 + Llama and is kept only to document the exact upstream call
site that ``src/baselines/qwen3_kvcompress.py`` reproduces for transformers 5.14.
"""

from importlib import import_module as _imp

# Canonical FasterDecoding classes/helpers (reference; kept importable so the
# provenance is testable). These are NOT the GQA path used on Qwen3.
_fd = _imp("src.baselines.snapkv.snapkv_utils_fasterdecoding")
SnapKVClusterFasterDecoding = _fd.SnapKVCluster
init_snapkv_fasterdecoding = _fd.init_snapkv

__all__ = ["SnapKVClusterFasterDecoding", "init_snapkv_fasterdecoding"]
