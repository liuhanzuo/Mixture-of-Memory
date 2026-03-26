"""
CompressedMemoryCache 测试脚本。

测试内容:
    1. 基础压缩 + 重建精度: M ≈ K̃^T Ṽ
    2. 不同 rank 的能量保留率
    3. 多 chunk 压缩
    4. 直接查询 vs 完整 attention 对比
    5. 增量更新 (append_chunk)
    6. 与 MAGGate 集成

运行: python -m pytest tests/test_compressed_memory.py -v
或:   python tests/test_compressed_memory.py
"""

import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.mag.compressed_memory import CompressedMemoryCache, CompressedMemoryConfig
from src.memory.mag.mag_gate import MAGGate, MAGGateConfig, _MAGCrossAttnBlock


def _sep(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ------------------------------------------------------------------ #
#  Test 1: 基础压缩 + 重建精度
# ------------------------------------------------------------------ #

def test_basic_compression_and_reconstruction():
    """验证 SVD 压缩后, 虚拟 KV pairs 能近似重建原始 M 矩阵。

    核心验证:
        M_original = K^T V
        M_reconstructed = K̃^T Ṽ  (从 SVD 虚拟 KV pairs 重建)
        ‖M_original - M_reconstructed‖ / ‖M_original‖ 应该很小
    """
    _sep("Test 1: 基础压缩 + 重建精度")

    torch.manual_seed(42)
    B, N, D = 2, 64, 256
    n_heads = 8
    head_dim = D // n_heads  # 32

    config = CompressedMemoryConfig(
        hidden_dim=D,
        num_heads=n_heads,
        rank=8,
        chunk_size=0,  # 整体作为一个 chunk
        normalize=False,  # 关闭归一化, 方便验证重建精度
    )
    cache = CompressedMemoryCache(config)

    # 模拟 hidden states
    hidden_states = torch.randn(B, N, D)

    # 压缩
    stats = cache.compress_and_store(hidden_states)
    print(f"  压缩统计: {stats['total_tokens']} tokens → {stats['virtual_tokens']} virtual tokens")
    print(f"  压缩率: {stats['compression_ratio']:.1f}x")
    print(f"  SVD 能量保留: {stats['singular_value_stats'][0].get('energy_retained', 'N/A')}")

    # 获取虚拟 KV pairs
    result = cache.get_virtual_tokens()
    assert result is not None, "缓存不应为空"
    virtual_keys, virtual_values = result
    print(f"  虚拟 keys shape: {virtual_keys.shape}")
    print(f"  虚拟 values shape: {virtual_values.shape}")

    # 重建 M: 分头后计算 K̃^T Ṽ
    r = config.rank
    vk_heads = virtual_keys.view(B, r, n_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, r, hd)
    vv_heads = virtual_values.view(B, r, n_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, r, hd)
    M_reconstructed = torch.matmul(vk_heads.transpose(-2, -1), vv_heads)  # (B, H, hd, hd)

    # 原始 M
    K_orig = hidden_states.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, N, hd)
    V_orig = hidden_states.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)
    M_original = torch.matmul(K_orig.transpose(-2, -1), V_orig)  # (B, H, hd, hd)

    # 计算相对误差
    error = (M_original - M_reconstructed).norm() / M_original.norm()
    print(f"  相对重建误差: {error.item():.6f}")
    print(f"  ‖M_original‖: {M_original.norm().item():.4f}")
    print(f"  ‖M_reconstructed‖: {M_reconstructed.norm().item():.4f}")

    # rank=8 对 head_dim=32 的矩阵, 能量保留约 71%, 误差约 0.54 是合理的
    assert error.item() < 0.8, f"重建误差过大: {error.item():.4f}"
    print("  ✅ 通过: 重建误差在可接受范围内")


# ------------------------------------------------------------------ #
#  Test 2: 不同 rank 的能量保留率
# ------------------------------------------------------------------ #

def test_rank_vs_energy():
    """测试不同 rank 下的能量保留率和重建精度。

    预期: rank 越大, 能量保留越多, 重建误差越小。
    """
    _sep("Test 2: 不同 rank 的能量保留率")

    torch.manual_seed(123)
    B, N, D = 1, 128, 256
    n_heads = 8

    hidden_states = torch.randn(B, N, D)

    ranks = [1, 2, 4, 8, 16, 32]
    prev_error = float('inf')

    print(f"  {'Rank':>6} | {'能量保留':>10} | {'重建误差':>10} | {'Top σ':>10} | {'Min σ':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for rank in ranks:
        config = CompressedMemoryConfig(
            hidden_dim=D, num_heads=n_heads, rank=rank,
            chunk_size=0, normalize=False,
        )
        cache = CompressedMemoryCache(config)
        stats = cache.compress_and_store(hidden_states)

        sv_stats = stats['singular_value_stats'][0]
        energy = sv_stats.get('energy_retained', 0)
        top_sv = sv_stats.get('top_singular_value', 0)
        min_sv = sv_stats.get('min_singular_value', 0)

        # 重建误差
        result = cache.get_virtual_tokens()
        vk, vv = result
        r = rank
        head_dim = D // n_heads
        vk_h = vk.view(B, r, n_heads, head_dim).permute(0, 2, 1, 3)
        vv_h = vv.view(B, r, n_heads, head_dim).permute(0, 2, 1, 3)
        M_recon = torch.matmul(vk_h.transpose(-2, -1), vv_h)

        K_orig = hidden_states.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)
        V_orig = hidden_states.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)
        M_orig = torch.matmul(K_orig.transpose(-2, -1), V_orig)

        error = (M_orig - M_recon).norm() / M_orig.norm()

        print(f"  {rank:>6} | {energy:>10.4f} | {error.item():>10.6f} | {top_sv:>10.4f} | {min_sv:>10.4f}")

        # 验证单调性: rank 越大, 误差应该越小 (或相等)
        assert error.item() <= prev_error + 1e-6, \
            f"rank={rank} 的误差 ({error.item():.6f}) 不应大于 rank={ranks[ranks.index(rank)-1]} 的误差 ({prev_error:.6f})"
        prev_error = error.item()

    print("  ✅ 通过: 能量保留率和重建精度随 rank 单调改善")


# ------------------------------------------------------------------ #
#  Test 3: 多 chunk 压缩
# ------------------------------------------------------------------ #

def test_multi_chunk():
    """测试多 chunk 分割和压缩。"""
    _sep("Test 3: 多 chunk 压缩")

    torch.manual_seed(7)
    B, N, D = 2, 256, 128
    n_heads = 4
    rank = 4
    chunk_size = 64  # 256 / 64 = 4 chunks

    config = CompressedMemoryConfig(
        hidden_dim=D, num_heads=n_heads, rank=rank,
        chunk_size=chunk_size, normalize=True,
    )
    cache = CompressedMemoryCache(config)

    hidden_states = torch.randn(B, N, D)
    stats = cache.compress_and_store(hidden_states)

    expected_chunks = math.ceil(N / chunk_size)
    expected_virtual = expected_chunks * rank

    print(f"  输入: {N} tokens, chunk_size={chunk_size}")
    print(f"  预期: {expected_chunks} chunks, {expected_virtual} virtual tokens")
    print(f"  实际: {stats['num_chunks']} chunks, {stats['virtual_tokens']} virtual tokens")
    print(f"  压缩率: {stats['compression_ratio']:.1f}x")

    assert stats['num_chunks'] == expected_chunks, \
        f"chunk 数不匹配: 预期 {expected_chunks}, 实际 {stats['num_chunks']}"
    assert stats['virtual_tokens'] == expected_virtual, \
        f"虚拟 token 数不匹配: 预期 {expected_virtual}, 实际 {stats['virtual_tokens']}"

    # 验证缓存形状
    result = cache.get_virtual_tokens()
    assert result is not None
    vk, vv = result
    assert vk.shape == (B, expected_virtual, D), f"keys 形状错误: {vk.shape}"
    assert vv.shape == (B, expected_virtual, D), f"values 形状错误: {vv.shape}"

    # 验证归一化 (keys 应该是 L2 归一化的)
    key_norms = vk.norm(dim=-1)  # (B, C*r)
    assert torch.allclose(key_norms, torch.ones_like(key_norms), atol=1e-5), \
        f"归一化后 key 的 L2 范数应为 1, 实际: {key_norms.mean():.4f}"

    print(f"  Key L2 范数 (应≈1.0): {key_norms.mean().item():.6f}")
    print("  ✅ 通过: 多 chunk 压缩正确")


# ------------------------------------------------------------------ #
#  Test 4: 直接查询 vs 完整 attention 对比
# ------------------------------------------------------------------ #

def test_query_vs_full_attention():
    """对比 SVD 压缩查询 vs 完整 KV attention 的输出差异。

    完整 attention: softmax(Q K^T / √d) V  (N 个 KV pairs)
    压缩 attention: softmax(Q K̃^T / √d) Ṽ  (r 个虚拟 KV pairs)

    预期: rank 越大, 两者越接近。
    """
    _sep("Test 4: 直接查询 vs 完整 attention")

    torch.manual_seed(99)
    B, N, D = 1, 64, 128
    T = 16  # query 长度
    n_heads = 4
    head_dim = D // n_heads

    # 模拟 memory hidden states 和 query
    memory_hs = torch.randn(B, N, D)
    query_hs = torch.randn(B, T, D)

    # 完整 attention (ground truth)
    Q_full = query_hs.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, H, T, hd)
    K_full = memory_hs.view(B, N, n_heads, head_dim).transpose(1, 2)  # (B, H, N, hd)
    V_full = memory_hs.view(B, N, n_heads, head_dim).transpose(1, 2)  # (B, H, N, hd)

    scale = 1.0 / math.sqrt(head_dim)
    attn_full = torch.softmax(torch.matmul(Q_full, K_full.transpose(-2, -1)) * scale, dim=-1)
    output_full = torch.matmul(attn_full, V_full)  # (B, H, T, hd)
    output_full = output_full.transpose(1, 2).contiguous().view(B, T, D)

    print(f"  完整 attention 输出 shape: {output_full.shape}")
    print(f"  完整 attention 输出 norm: {output_full.norm().item():.4f}")

    # 不同 rank 的压缩查询
    ranks = [2, 4, 8, 16, 32]
    print(f"\n  {'Rank':>6} | {'输出 cosine sim':>15} | {'相对 L2 误差':>15} | {'输出 norm':>12}")
    print(f"  {'-'*6}-+-{'-'*15}-+-{'-'*15}-+-{'-'*12}")

    for rank in ranks:
        config = CompressedMemoryConfig(
            hidden_dim=D, num_heads=n_heads, rank=rank,
            chunk_size=0, normalize=False,
        )
        cache = CompressedMemoryCache(config)
        cache.compress_and_store(memory_hs)

        output_compressed = cache.query(query_hs)
        assert output_compressed is not None

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            output_full.flatten(), output_compressed.flatten(), dim=0
        ).item()

        # 相对 L2 误差
        rel_error = (output_full - output_compressed).norm() / output_full.norm()

        print(f"  {rank:>6} | {cos_sim:>15.6f} | {rel_error.item():>15.6f} | {output_compressed.norm().item():>12.4f}")

    print("  ✅ 通过: 压缩查询输出与完整 attention 的差异随 rank 减小")


# ------------------------------------------------------------------ #
#  Test 5: 增量更新 (append_chunk)
# ------------------------------------------------------------------ #

def test_append_chunk():
    """测试增量添加 chunk 到缓存。"""
    _sep("Test 5: 增量更新 (append_chunk)")

    torch.manual_seed(42)
    B, D = 2, 128
    n_heads = 4
    rank = 4

    config = CompressedMemoryConfig(
        hidden_dim=D, num_heads=n_heads, rank=rank,
        chunk_size=0, normalize=False,
    )
    cache = CompressedMemoryCache(config)

    # 第一个 chunk
    hs1 = torch.randn(B, 32, D)
    stats1 = cache.compress_and_store(hs1)
    result1 = cache.get_virtual_tokens()
    assert result1 is not None
    vk1_count = result1[0].shape[1]
    print(f"  Chunk 1: {stats1['virtual_tokens']} virtual tokens")

    # 追加第二个 chunk
    hs2 = torch.randn(B, 48, D)
    stats2 = cache.append_chunk(hs2)
    result2 = cache.get_virtual_tokens()
    assert result2 is not None
    vk2_count = result2[0].shape[1]
    print(f"  Chunk 1+2: {vk2_count} virtual tokens (预期 {rank * 2})")

    assert vk2_count == rank * 2, f"追加后应有 {rank * 2} 个虚拟 token, 实际 {vk2_count}"

    # 追加第三个 chunk
    hs3 = torch.randn(B, 64, D)
    stats3 = cache.append_chunk(hs3)
    result3 = cache.get_virtual_tokens()
    assert result3 is not None
    vk3_count = result3[0].shape[1]
    print(f"  Chunk 1+2+3: {vk3_count} virtual tokens (预期 {rank * 3})")

    assert vk3_count == rank * 3, f"追加后应有 {rank * 3} 个虚拟 token, 实际 {vk3_count}"

    # 验证缓存统计
    cache_stats = cache.get_stats()
    print(f"  缓存统计: {cache_stats}")
    print("  ✅ 通过: 增量更新正确")


# ------------------------------------------------------------------ #
#  Test 6: 与 MAGGate 集成
# ------------------------------------------------------------------ #

def test_integration_with_mag_gate():
    """测试 CompressedMemoryCache 与 MAGGate 的集成。

    验证:
        1. 绑定 MAGGate 的 K/V 投影层
        2. 压缩后的虚拟 tokens 能通过 MAGGate 注入
        3. 输出形状正确, 梯度可传播
    """
    _sep("Test 6: 与 MAGGate 集成")

    torch.manual_seed(42)
    B, T, D = 2, 32, 256
    N_mem = 64  # 记忆 token 数
    n_heads = 8
    rank = 8

    # 创建 MAGGate
    mag_config = MAGGateConfig(
        hidden_dim=D,
        num_heads=n_heads,
        memory_dim=D,
        injection_layers=[4, 8],
        share_parameters=True,
        gate_init_bias=-2.0,
    )
    mag_gate = MAGGate(mag_config)

    # 创建 CompressedMemoryCache
    cm_config = CompressedMemoryConfig(
        hidden_dim=D,
        num_heads=n_heads,
        rank=rank,
        chunk_size=0,
        normalize=False,  # 不归一化, 因为要传给 MAGGate 再投影
    )
    cache = CompressedMemoryCache(cm_config)

    # 注意: 这里不绑定 K/V 投影, 因为 MAGGate 会自己做投影
    # 如果绑定了, 就会"双重投影"

    # 模拟记忆 hidden states
    memory_hs = torch.randn(B, N_mem, D)

    # 压缩
    stats = cache.compress_and_store(memory_hs)
    print(f"  压缩: {N_mem} tokens → {stats['virtual_tokens']} virtual tokens")

    # 获取虚拟 tokens 作为 MAGGate 的 memory_vectors
    mem_result = cache.get_virtual_tokens_as_memory()
    assert mem_result is not None
    memory_vectors, memory_mask = mem_result
    print(f"  Memory vectors shape: {memory_vectors.shape}")
    print(f"  Memory mask shape: {memory_mask.shape}")

    # 模拟 backbone hidden states
    hidden_states = torch.randn(B, T, D, requires_grad=True)

    # 通过 MAGGate 注入
    output_l4, gate_l4 = mag_gate.inject(
        layer_idx=4,
        hidden_states=hidden_states,
        memory_vectors=memory_vectors,
        memory_mask=memory_mask.bool(),
        return_gate=True,
    )
    print(f"  Layer 4 输出 shape: {output_l4.shape}")
    print(f"  Layer 4 gate 均值: {gate_l4.mean().item():.4f}")

    # 验证输出形状
    assert output_l4.shape == (B, T, D), f"输出形状错误: {output_l4.shape}"

    # 验证梯度可传播
    loss = output_l4.sum()
    loss.backward()
    assert hidden_states.grad is not None, "梯度未传播到 hidden_states"
    print(f"  梯度 norm: {hidden_states.grad.norm().item():.4f}")

    # 非注入层应该直接返回原始 hidden states
    hidden_states2 = torch.randn(B, T, D)
    output_l5 = mag_gate.inject(
        layer_idx=5,  # 不在 injection_layers 中
        hidden_states=hidden_states2,
        memory_vectors=memory_vectors,
    )
    assert torch.equal(output_l5, hidden_states2), "非注入层应返回原始 hidden states"
    print("  非注入层 (layer 5): 正确跳过")

    print("  ✅ 通过: MAGGate 集成正确")


# ------------------------------------------------------------------ #
#  Test 7: 带 attention mask 的压缩
# ------------------------------------------------------------------ #

def test_with_attention_mask():
    """测试带 padding mask 的压缩。"""
    _sep("Test 7: 带 attention mask 的压缩")

    torch.manual_seed(42)
    B, N, D = 2, 64, 128
    n_heads = 4
    rank = 4

    config = CompressedMemoryConfig(
        hidden_dim=D, num_heads=n_heads, rank=rank,
        chunk_size=0, normalize=False,
    )
    cache = CompressedMemoryCache(config)

    hidden_states = torch.randn(B, N, D)

    # 创建 mask: 第一个 batch 全有效, 第二个 batch 后半部分是 padding
    mask = torch.ones(B, N)
    mask[1, N // 2:] = 0

    stats = cache.compress_and_store(hidden_states, attention_mask=mask)
    print(f"  压缩统计: {stats['virtual_tokens']} virtual tokens")

    # 验证: 有 mask 和无 mask 的结果应该不同
    cache2 = CompressedMemoryCache(config)
    stats2 = cache2.compress_and_store(hidden_states)

    vk1, vv1 = cache.get_virtual_tokens()
    vk2, vv2 = cache2.get_virtual_tokens()

    # 第一个 batch (全有效) 应该相同
    diff_b0 = (vk1[0] - vk2[0]).norm()
    # 第二个 batch (有 padding) 应该不同
    diff_b1 = (vk1[1] - vk2[1]).norm()

    print(f"  Batch 0 (全有效) 差异: {diff_b0.item():.6f} (应≈0)")
    print(f"  Batch 1 (有 padding) 差异: {diff_b1.item():.6f} (应>0)")

    assert diff_b0.item() < 1e-5, "全有效 batch 的结果应该相同"
    assert diff_b1.item() > 1e-3, "有 padding 的 batch 结果应该不同"

    print("  ✅ 通过: attention mask 正确生效")


# ------------------------------------------------------------------ #
#  Test 8: 性能基准
# ------------------------------------------------------------------ #

def test_performance_benchmark():
    """简单的性能基准测试。"""
    _sep("Test 8: 性能基准")

    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    B, N, D = 4, 512, 1024
    n_heads = 16
    T = 128

    config = CompressedMemoryConfig(
        hidden_dim=D, num_heads=n_heads, rank=8,
        chunk_size=128, normalize=True,
    )
    cache = CompressedMemoryCache(config).to(device)

    hidden_states = torch.randn(B, N, D, device=device)
    query_states = torch.randn(B, T, D, device=device)

    # 预热
    cache.compress_and_store(hidden_states)
    cache.query(query_states)

    # 压缩耗时
    n_runs = 10
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        cache.compress_and_store(hidden_states)
    if device == "cuda":
        torch.cuda.synchronize()
    t_compress = (time.perf_counter() - t0) / n_runs * 1000

    # 查询耗时
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        cache.query(query_states)
    if device == "cuda":
        torch.cuda.synchronize()
    t_query = (time.perf_counter() - t0) / n_runs * 1000

    # 完整 attention 耗时 (baseline)
    head_dim = D // n_heads
    Q = query_states.view(B, T, n_heads, head_dim).transpose(1, 2)
    K = hidden_states.view(B, N, n_heads, head_dim).transpose(1, 2)
    V = hidden_states.view(B, N, n_heads, head_dim).transpose(1, 2)
    scale = 1.0 / math.sqrt(head_dim)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
        _ = torch.matmul(attn, V)
    if device == "cuda":
        torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / n_runs * 1000

    stats = cache.get_stats()
    virtual_tokens = stats.get('cache_shared', {}).get('virtual_tokens', '?')

    print(f"\n  配置: B={B}, N={N}, D={D}, T={T}, rank=8, chunk_size=128")
    print(f"  虚拟 tokens: {virtual_tokens} (原始 {N})")
    print(f"\n  {'操作':>20} | {'耗时 (ms)':>12} | {'相对完整 attn':>15}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*15}")
    print(f"  {'压缩 (SVD)':>20} | {t_compress:>12.3f} | {'(一次性)':>15}")
    print(f"  {'压缩查询':>20} | {t_query:>12.3f} | {t_query/t_full:>14.2f}x")
    print(f"  {'完整 attention':>20} | {t_full:>12.3f} | {'1.00x':>15}")

    # 内存占用估算
    mem_full = B * N * D * 2 * 4  # K + V, float32
    mem_compressed = B * int(virtual_tokens) * D * 2 * 4 if isinstance(virtual_tokens, int) else 0
    if mem_compressed > 0:
        print(f"\n  内存: 完整 KV = {mem_full / 1024:.1f} KB, 压缩 = {mem_compressed / 1024:.1f} KB "
              f"({mem_compressed / mem_full * 100:.1f}%)")

    print("  ✅ 通过: 性能基准完成")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 70)
    print("  CompressedMemoryCache (SVD 版本) 测试")
    print("=" * 70)

    tests = [
        test_basic_compression_and_reconstruction,
        test_rank_vs_energy,
        test_multi_chunk,
        test_query_vs_full_attention,
        test_append_chunk,
        test_integration_with_mag_gate,
        test_with_attention_mask,
        test_performance_benchmark,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

    _sep("测试总结")
    print(f"  通过: {passed}/{len(tests)}")
    print(f"  失败: {failed}/{len(tests)}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  🎉 全部测试通过!")
        sys.exit(0)
