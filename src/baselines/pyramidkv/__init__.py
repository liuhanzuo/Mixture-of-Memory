"""Vendored PyramidKV + GQA-aware SnapKV (Zefan-Cai/PyramidKV).

Upstream commit ``94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0`` (see
``src/baselines/PROVENANCE.md``). ``pyramidkv_utils.py`` is vendored verbatim; it
provides the GQA-aware ``SnapKVCluster`` / ``PyramidKVCluster`` (observation-window
softmax -> avgpool -> top-k over the past, keep the recent window; PyramidKV adds
the per-layer pyramidal budget schedule) together with the ``init_snapkv`` /
``init_pyramidkv`` factories. These clusters return the compressed K/V at kv-head
granularity (8 heads on Qwen3), exactly the ``DynamicLayer`` storage layout, so
``src/baselines/qwen3_kvcompress.py`` can write them straight back into the cache.
The reference ``monkeypatch_reference.py`` targets transformers 4.37 + Llama /
Mistral and is kept only to document the upstream integration.
"""

from .pyramidkv_utils import (  # noqa: F401
    SnapKVCluster,
    PyramidKVCluster,
    init_snapkv,
    init_pyramidkv,
)

__all__ = ["SnapKVCluster", "PyramidKVCluster", "init_snapkv", "init_pyramidkv"]
