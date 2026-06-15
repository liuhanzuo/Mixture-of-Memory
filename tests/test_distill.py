"""Unit tests for v21 self-study distillation (logits-KL + hidden-cosine).

Covers:
  1. bf16<->int16 npz round-trip + (doc_idx, group_pos) naming round-trip via
     load_distill_npz (the cache format produced by build_distill_cache.py).
  2. hidden_states off-by-one (decoder layer L -> hidden_states[L+1])固化 as an
     explicit assertion mirroring the builder's logic.
  3. distill_logits_kl: == 0 when student logits match teacher on the support;
     bidirectional KL is non-negative; gradient flows to student only.
  4. distill_hidden_cosine: ~0 when aligned; positive otherwise; teacher stopgrad.

Run: .venv/bin/python -m pytest tests/test_distill.py -q
"""
import os
import sys
import tempfile

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import the helpers from the training script (module-level, no side effects
# beyond imports + logging.basicConfig).
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "train_mod", os.path.join(_ROOT, "scripts", "train_mem_space_dolmino_cpt.py"))
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)

load_distill_npz = train_mod.load_distill_npz
distill_logits_kl = train_mod.distill_logits_kl
distill_hidden_cosine = train_mod.distill_hidden_cosine
_bf16_from_int16 = train_mod._bf16_from_int16
assert_distill_cache_consistent = train_mod.assert_distill_cache_consistent


def _write_fake_cache(out_dir, doc_idx, group_pos, A=8, topk=64, n_sel=3, D=16,
                      layers=(12, 20, 28)):
    """Mimic build_distill_cache.py's npz format exactly (bf16-as-int16)."""
    logit_idx = np.random.randint(0, 1000, size=(A, topk)).astype(np.int32)
    logit_val = torch.randn(A, topk).to(torch.bfloat16)
    hidden = torch.randn(A, n_sel, D).to(torch.bfloat16)
    answer_mask = np.ones((A,), dtype=bool)
    path = os.path.join(out_dir, f"{doc_idx}_{group_pos}.npz")
    np.savez(
        path,
        logit_idx=logit_idx,
        logit_val=logit_val.view(torch.int16).cpu().numpy(),
        hidden=hidden.view(torch.int16).cpu().numpy(),
        answer_mask=answer_mask,
        meta_doc_idx=np.int64(doc_idx),
        meta_group_pos=np.int64(group_pos),
        meta_n_ctx=np.int64(3),
        meta_chunk_size=np.int64(A),
        meta_layers=np.array(layers, dtype=np.int32),
    )
    return path, logit_idx, logit_val, hidden, layers


def test_npz_roundtrip_and_naming():
    with tempfile.TemporaryDirectory() as d:
        doc_idx, group_pos = 4242, 3
        path, logit_idx, logit_val, hidden, layers = _write_fake_cache(
            d, doc_idx, group_pos)
        # naming round-trip: file is named exactly {doc_idx}_{group_pos}.npz
        assert os.path.basename(path) == f"{doc_idx}_{group_pos}.npz"
        cache = load_distill_npz(path, torch.device("cpu"))
        # int32 idx round-trips losslessly
        assert torch.equal(cache["logit_idx"],
                           torch.from_numpy(logit_idx).long())
        # bf16 values round-trip bit-exactly (no precision loss in store/load)
        assert torch.equal(cache["logit_val"], logit_val)
        assert torch.equal(cache["hidden"], hidden)
        assert cache["layers"].tolist() == list(layers)
        assert cache["answer_mask"].all()
    print("test_npz_roundtrip_and_naming OK")


def test_bf16_int16_view_roundtrip():
    x = torch.randn(100).to(torch.bfloat16)
    as_i16 = x.view(torch.int16).cpu().numpy()
    back = _bf16_from_int16(as_i16)
    assert torch.equal(x, back), "bf16 bit-pattern round-trip must be exact"
    print("test_bf16_int16_view_roundtrip OK")


def test_hidden_states_off_by_one():
    """固化 decoder-layer L -> hidden_states[L+1] (index 0 = embeddings).

    Mirrors build_distill_cache.py: for n_decoder_layers layers, hidden_states
    is a tuple of length n_decoder_layers+1; layer L's OUTPUT is at index L+1.
    """
    n_decoder_layers = 32
    # Simulate HF's hidden_states tuple: index 0 = embed, 1..32 = layer outputs.
    hs = tuple(torch.full((1, 4, 8), float(i)) for i in range(n_decoder_layers + 1))
    assert len(hs) == n_decoder_layers + 1
    for L in (12, 20, 28):
        layer_out = hs[L + 1]
        # value encodes the tuple index, so verify we picked L+1 not L.
        assert layer_out[0, 0, 0].item() == float(L + 1), (
            f"decoder layer {L} must come from hidden_states[{L+1}]")
    print("test_hidden_states_off_by_one OK")


def test_distill_logits_kl_zero_when_matched():
    A, topk, V = 6, 64, 2000
    torch.manual_seed(0)
    teacher_idx = torch.stack(
        [torch.randperm(V)[:topk] for _ in range(A)], dim=0)  # [A, topk] unique
    teacher_val = torch.randn(A, topk)
    # Build student logits so that on the teacher support they EXACTLY equal
    # teacher_val (everything else very negative). Then q == p on support -> KL=0.
    student = torch.full((A, V), -1e4)
    for i in range(A):
        student[i, teacher_idx[i]] = teacher_val[i]
    loss = distill_logits_kl(student, teacher_idx, teacher_val, lam=0.6)
    assert loss.item() < 1e-4, f"matched KL should be ~0, got {loss.item()}"
    print(f"test_distill_logits_kl_zero_when_matched OK (loss={loss.item():.2e})")


def test_distill_logits_kl_nonneg_and_grad():
    A, topk, V = 4, 64, 1000
    torch.manual_seed(1)
    teacher_idx = torch.stack(
        [torch.randperm(V)[:topk] for _ in range(A)], dim=0)
    teacher_val = torch.randn(A, topk)
    student = torch.randn(A, V, requires_grad=True)
    loss = distill_logits_kl(student, teacher_idx, teacher_val, lam=0.6)
    # Each KL term is non-negative; convex combo is non-negative.
    assert loss.item() >= -1e-6, f"bidirectional KL must be >=0, got {loss}"
    assert loss.item() > 0, "mismatched dists should give strictly positive KL"
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    # teacher is a constant tensor (no grad requested) -> gradient only on student.
    assert not teacher_val.requires_grad
    print(f"test_distill_logits_kl_nonneg_and_grad OK (loss={loss.item():.4f})")


def test_distill_hidden_cosine_zero_when_aligned():
    A, n_sel, D = 5, 3, 32
    torch.manual_seed(2)
    teacher = torch.randn(A, n_sel, D)
    # student = positive scalar multiple of teacher -> same direction -> cos=1.
    student = (teacher.clone() * 2.5).requires_grad_(True)
    loss = distill_hidden_cosine(student, teacher)
    assert loss.item() < 1e-4, f"aligned cosine loss should be ~0, got {loss.item()}"
    print(f"test_distill_hidden_cosine_zero_when_aligned OK (loss={loss.item():.2e})")


def test_distill_hidden_cosine_positive_and_stopgrad():
    A, n_sel, D = 5, 3, 32
    torch.manual_seed(3)
    teacher = torch.randn(A, n_sel, D, requires_grad=True)  # even if grad asked
    student = torch.randn(A, n_sel, D, requires_grad=True)
    loss = distill_hidden_cosine(student, teacher)
    assert loss.item() > 0, "misaligned hidden -> positive 1-cos"
    # summed over n_sel layers, each in [0,2], so bounded by 2*n_sel.
    assert loss.item() <= 2.0 * n_sel + 1e-4
    loss.backward()
    # teacher is detached inside -> receives no gradient.
    assert teacher.grad is None, "teacher hidden must be stopgrad (no gradient)"
    assert student.grad is not None and torch.isfinite(student.grad).all()
    print(f"test_distill_hidden_cosine_positive_and_stopgrad OK (loss={loss.item():.4f})")


def _write_fake_meta(out_dir, n_ctx=3, chunk_size=512, layers=(12, 20, 28)):
    import json
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "n_ctx": int(n_ctx),
            "chunk_size": int(chunk_size),
            "distill_layers": [int(x) for x in layers],
            "model_path": "models/Meta-Llama-3-8B",
            "topk": 64,
            "group_len": (int(n_ctx) + 1) * int(chunk_size),
        }, f)


def test_cache_consistency_guard():
    """Mismatched n_ctx/chunk_size/layers -> RuntimeError; match -> ok; missing
    meta.json -> RuntimeError (must build cache first)."""
    with tempfile.TemporaryDirectory() as d:
        # (a) missing meta.json -> raise
        raised = False
        try:
            assert_distill_cache_consistent(d, 512, 3, [12, 20, 28])
        except RuntimeError as e:
            raised = True
            assert "meta.json not found" in str(e)
        assert raised, "missing meta.json must raise"

        # write a matching meta.json
        _write_fake_meta(d, n_ctx=3, chunk_size=512, layers=(12, 20, 28))

        # (b) matching config -> no raise, returns meta
        meta = assert_distill_cache_consistent(d, 512, 3, [12, 20, 28])
        assert int(meta["n_ctx"]) == 3 and int(meta["chunk_size"]) == 512

        # (c) chunk_size mismatch -> raise
        for kwargs in (
            dict(train_chunk_size=256, train_n_ctx=3, train_layers=[12, 20, 28]),
            dict(train_chunk_size=512, train_n_ctx=7, train_layers=[12, 20, 28]),
            dict(train_chunk_size=512, train_n_ctx=3, train_layers=[12, 20, 24]),
            dict(train_chunk_size=512, train_n_ctx=3, train_layers=[12, 20]),
        ):
            raised = False
            try:
                assert_distill_cache_consistent(d, **kwargs)
            except RuntimeError as e:
                raised = True
                assert "MISMATCH" in str(e)
            assert raised, f"config {kwargs} must raise mismatch"
    print("test_cache_consistency_guard OK")


if __name__ == "__main__":
    test_npz_roundtrip_and_naming()
    test_bf16_int16_view_roundtrip()
    test_hidden_states_off_by_one()
    test_distill_logits_kl_zero_when_matched()
    test_distill_logits_kl_nonneg_and_grad()
    test_distill_hidden_cosine_zero_when_aligned()
    test_distill_hidden_cosine_positive_and_stopgrad()
    test_cache_consistency_guard()
    print("\nALL DISTILL UNIT TESTS PASSED")
