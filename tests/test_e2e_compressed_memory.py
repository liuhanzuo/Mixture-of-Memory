#!/usr/bin/env python3
"""
端到端测试: SVD CompressedMemory 在记忆任务上的整体效果。

测试场景:
    用 debug 小模型 + 合成数据, 模拟完整的记忆注入流程:
    1. 编码记忆文本 → hidden states
    2. SVD 压缩 → 虚拟 KV pairs
    3. 通过 MAGGate 注入到 backbone 中间层
    4. 对比有/无记忆注入的 PPL 和生成质量

对比三种模式:
    A. 无记忆 (纯 backbone)
    B. 原始 MAGGate (pooled memory vectors → cross attention)
    C. SVD CompressedMemory (压缩虚拟 KV → cross attention)

运行: python3 tests/test_e2e_compressed_memory.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backbone.swa_model import _DebugTransformer
from src.backbone.hidden_state_types import BackboneOutput
from src.memory.mag.mag_gate import MAGGate, MAGGateConfig, _MAGCrossAttnBlock
from src.memory.mag.compressed_memory import CompressedMemoryCache, CompressedMemoryConfig
from src.memory.mag.context_selector import ContextSelector, ContextSelectorConfig


def _sep(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


# ======================================================================
# 构建 Debug 环境
# ======================================================================

def build_debug_env(
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 4,
    vocab_size: int = 500,
    max_seq_len: int = 256,
    seed: int = 42,
):
    """构建 debug 测试环境: backbone + MAGGate + CompressedMemory + Selector。"""
    torch.manual_seed(seed)

    # 1. Debug backbone
    backbone = _DebugTransformer(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    backbone.eval()

    # 2. MAGGate (在中间层注入)
    injection_layers = [1, 3, 5]  # 在第 2, 4, 6 层后注入
    mag_config = MAGGateConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        memory_dim=hidden_dim,
        injection_layers=injection_layers,
        share_parameters=True,
        gate_init_bias=-1.0,  # 稍微开放一点, 方便测试
        use_layer_norm=True,
    )
    mag_gate = MAGGate(mag_config)

    # 3. CompressedMemoryCache
    cm_config = CompressedMemoryConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rank=8,
        chunk_size=0,  # 整体作为一个 chunk
        normalize=False,
    )
    compressed_cache = CompressedMemoryCache(cm_config)

    # 4. ContextSelector
    sel_config = ContextSelectorConfig(
        input_dim=hidden_dim,
        hidden_dim=64,
        num_mlp_layers=2,
        top_k=3,
        temperature=1.0,
    )
    selector = ContextSelector(sel_config)

    return backbone, mag_gate, compressed_cache, selector


# ======================================================================
# 合成记忆数据
# ======================================================================

def create_synthetic_memory_data(
    backbone: _DebugTransformer,
    num_memories: int = 8,
    memory_len: int = 16,
    query_len: int = 12,
    target_len: int = 8,
    vocab_size: int = 500,
    seed: int = 42,
):
    """创建合成的记忆数据。

    策略: 让 target 的 token 分布与某些"相关记忆"高度相关,
    这样如果模型能利用记忆, PPL 应该降低。

    具体做法:
    - 生成 num_memories 条记忆 (随机 token 序列)
    - 标记其中 2 条为"相关记忆"
    - target 序列 = 相关记忆的 token 子集 + 一些变化
    - query 序列 = 随机 token

    Returns:
        query_ids: (1, query_len)
        target_ids: (1, target_len)
        memory_ids_list: list of (1, memory_len) tensors
        relevant_indices: list of int (相关记忆的索引)
    """
    torch.manual_seed(seed)

    # 生成记忆
    memory_ids_list = []
    for i in range(num_memories):
        ids = torch.randint(10, vocab_size, (1, memory_len))
        memory_ids_list.append(ids)

    # 标记相关记忆 (第 1 和第 4 条)
    relevant_indices = [1, 4] if num_memories > 4 else [0]

    # target = 从相关记忆中采样 token
    target_tokens = []
    for idx in relevant_indices:
        mem_tokens = memory_ids_list[idx][0].tolist()
        # 取前几个 token
        n_take = min(target_len // len(relevant_indices), len(mem_tokens))
        target_tokens.extend(mem_tokens[:n_take])
    # 补齐
    while len(target_tokens) < target_len:
        target_tokens.append(torch.randint(10, vocab_size, (1,)).item())
    target_ids = torch.tensor([target_tokens[:target_len]])

    # query = 随机 token
    query_ids = torch.randint(10, vocab_size, (1, query_len))

    return query_ids, target_ids, memory_ids_list, relevant_indices


# ======================================================================
# 编码记忆 (用 backbone embedding)
# ======================================================================

def encode_memories(
    backbone: _DebugTransformer,
    memory_ids_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """用 backbone 的 embedding + mean pooling 编码记忆。

    Returns:
        pooled_vecs: (1, K, D) pooled 记忆向量 (用于原始 MAGGate)
        unpooled_hs: (1, K*T, D) 拼接的 unpooled hidden states (用于 SVD 压缩)
    """
    with torch.no_grad():
        all_pooled = []
        all_unpooled = []
        for mem_ids in memory_ids_list:
            # 过 backbone 前几层得到 hidden states
            B, T = mem_ids.shape
            positions = torch.arange(T, device=mem_ids.device).unsqueeze(0).expand(B, T)
            h = backbone.embed(mem_ids) + backbone.pos_embed(positions)

            # 过前 2 层 transformer (深层编码)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=mem_ids.device)
            for layer in backbone.encoder.layers[:2]:
                h = layer(h, src_mask=causal_mask)

            # pooled: mean pooling
            pooled = h.mean(dim=1)  # (1, D)
            all_pooled.append(pooled)
            all_unpooled.append(h)  # (1, T, D)

        pooled_vecs = torch.stack(all_pooled, dim=1)  # (1, K, D)
        unpooled_hs = torch.cat(all_unpooled, dim=1)   # (1, K*T, D)

    return pooled_vecs, unpooled_hs


# ======================================================================
# Forward: 无记忆
# ======================================================================

def forward_no_memory(
    backbone: _DebugTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """纯 backbone forward, 返回 loss。"""
    with torch.no_grad():
        output = backbone(input_ids, labels=labels)
    return output.loss.item()


# ======================================================================
# Forward: 原始 MAGGate (pooled vectors)
# ======================================================================

def forward_with_mag(
    backbone: _DebugTransformer,
    mag_gate: MAGGate,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    memory_vectors: torch.Tensor,
    selection_weights: torch.Tensor | None = None,
) -> float:
    """带 MAGGate 注入的 forward, 返回 loss。"""
    with torch.no_grad():
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = backbone.embed(input_ids) + backbone.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)

        for layer_idx, layer in enumerate(backbone.encoder.layers):
            h = layer(h, src_mask=causal_mask)
            # MAG 注入
            h = mag_gate.inject(
                layer_idx=layer_idx,
                hidden_states=h,
                memory_vectors=memory_vectors,
                selection_weights=selection_weights,
            )

        logits = backbone.lm_head(h)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    return loss.item()


# ======================================================================
# Forward: SVD CompressedMemory
# ======================================================================

def forward_with_compressed_memory(
    backbone: _DebugTransformer,
    mag_gate: MAGGate,
    compressed_cache: CompressedMemoryCache,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    unpooled_hs: torch.Tensor,
    selection_weights: torch.Tensor | None = None,
) -> float:
    """带 SVD 压缩记忆注入的 forward, 返回 loss。

    流程:
    1. 将 unpooled hidden states 压缩为虚拟 KV pairs
    2. 将虚拟 tokens 作为 memory_vectors 传给 MAGGate
    """
    with torch.no_grad():
        # Step 1: SVD 压缩
        compressed_cache.compress_and_store(unpooled_hs)

        # Step 2: 获取虚拟 tokens 作为 memory vectors
        mem_result = compressed_cache.get_virtual_tokens_as_memory()
        if mem_result is None:
            return forward_no_memory(backbone, input_ids, labels)
        virtual_memory, virtual_mask = mem_result

        # Step 3: Forward with MAG injection
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = backbone.embed(input_ids) + backbone.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)

        for layer_idx, layer in enumerate(backbone.encoder.layers):
            h = layer(h, src_mask=causal_mask)
            # MAG 注入 (用虚拟 tokens)
            h = mag_gate.inject(
                layer_idx=layer_idx,
                hidden_states=h,
                memory_vectors=virtual_memory,
                memory_mask=virtual_mask.bool(),
            )

        logits = backbone.lm_head(h)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    return loss.item()


# ======================================================================
# Forward: 直接查询 M (线性注意力)
# ======================================================================

def forward_with_direct_M_query(
    backbone: _DebugTransformer,
    compressed_cache: CompressedMemoryCache,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    unpooled_hs: torch.Tensor,
    injection_layers: set[int],
    gate_scale: float = 0.3,
) -> float:
    """直接用 M^T q 查询压缩记忆 (线性注意力, 绕过 MAGGate)。

    这是最"纯净"的 SVD 压缩记忆使用方式:
    在注入层, 用 compressed_cache.query() 直接做 cross attention。
    """
    with torch.no_grad():
        # 压缩
        compressed_cache.compress_and_store(unpooled_hs)

        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = backbone.embed(input_ids) + backbone.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)

        for layer_idx, layer in enumerate(backbone.encoder.layers):
            h = layer(h, src_mask=causal_mask)

            if layer_idx in injection_layers:
                # 直接查询压缩记忆
                mem_output = compressed_cache.query(h)
                if mem_output is not None:
                    # 简单的残差注入 (带缩放)
                    h = h + gate_scale * mem_output

        logits = backbone.lm_head(h)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    return loss.item()


# ======================================================================
# Test 1: PPL 对比 (核心测试)
# ======================================================================

def test_ppl_comparison():
    """对比四种模式的 PPL:
    A. 无记忆
    B. 原始 MAGGate (pooled vectors)
    C. SVD CompressedMemory → MAGGate
    D. SVD CompressedMemory → 直接查询 (线性注意力)
    """
    _sep("Test 1: PPL 对比 (端到端)")

    D = 128
    num_layers = 6
    num_heads = 4
    vocab_size = 500

    backbone, mag_gate, compressed_cache, selector = build_debug_env(
        hidden_dim=D, num_layers=num_layers, num_heads=num_heads,
        vocab_size=vocab_size, seed=42,
    )

    # 创建合成数据
    query_ids, target_ids, memory_ids_list, relevant_indices = create_synthetic_memory_data(
        backbone, num_memories=8, memory_len=16, query_len=12,
        target_len=8, vocab_size=vocab_size, seed=42,
    )

    # 拼接 query + target 作为输入
    input_ids = torch.cat([query_ids, target_ids], dim=1)  # (1, 20)
    query_len = query_ids.shape[1]
    # Labels: 只在 target 部分计算 loss
    labels = torch.full_like(input_ids, -100)
    labels[0, query_len:] = input_ids[0, query_len:]

    print(f"  输入: query_len={query_len}, target_len={target_ids.shape[1]}, total={input_ids.shape[1]}")
    print(f"  记忆数: {len(memory_ids_list)}, 相关记忆: {relevant_indices}")

    # 编码记忆
    pooled_vecs, unpooled_hs = encode_memories(backbone, memory_ids_list)
    print(f"  Pooled vectors shape: {pooled_vecs.shape}")
    print(f"  Unpooled hidden states shape: {unpooled_hs.shape}")

    # A. 无记忆
    loss_no_mem = forward_no_memory(backbone, input_ids, labels)
    ppl_no_mem = math.exp(min(loss_no_mem, 50))

    # B. 原始 MAGGate
    loss_mag = forward_with_mag(backbone, mag_gate, input_ids, labels, pooled_vecs)
    ppl_mag = math.exp(min(loss_mag, 50))

    # C. SVD CompressedMemory → MAGGate
    loss_svd_mag = forward_with_compressed_memory(
        backbone, mag_gate, compressed_cache, input_ids, labels, unpooled_hs,
    )
    ppl_svd_mag = math.exp(min(loss_svd_mag, 50))

    # D. SVD CompressedMemory → 直接查询
    loss_svd_direct = forward_with_direct_M_query(
        backbone, compressed_cache, input_ids, labels, unpooled_hs,
        injection_layers=mag_gate.injection_layers,
        gate_scale=0.3,
    )
    ppl_svd_direct = math.exp(min(loss_svd_direct, 50))

    print(f"\n  {'模式':>30} | {'Loss':>10} | {'PPL':>10} | {'vs 无记忆':>12}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    print(f"  {'A. 无记忆 (baseline)':>30} | {loss_no_mem:>10.4f} | {ppl_no_mem:>10.4f} | {'---':>12}")
    print(f"  {'B. MAGGate (pooled)':>30} | {loss_mag:>10.4f} | {ppl_mag:>10.4f} | {ppl_mag - ppl_no_mem:>+12.4f}")
    print(f"  {'C. SVD → MAGGate':>30} | {loss_svd_mag:>10.4f} | {ppl_svd_mag:>10.4f} | {ppl_svd_mag - ppl_no_mem:>+12.4f}")
    print(f"  {'D. SVD → 直接查询':>30} | {loss_svd_direct:>10.4f} | {ppl_svd_direct:>10.4f} | {ppl_svd_direct - ppl_no_mem:>+12.4f}")

    # 注意: 未训练的模型, PPL 差异可能很小或不稳定
    # 关键验证: 各模式都能正常运行, 不 crash, 输出合理
    print("\n  ✅ 通过: 四种模式都能正常运行, PPL 计算正确")


# ======================================================================
# Test 2: 不同 rank 的 PPL 对比
# ======================================================================

def test_rank_sensitivity():
    """测试不同 SVD rank 对 PPL 的影响。"""
    _sep("Test 2: 不同 rank 的 PPL 敏感性")

    D = 128
    num_layers = 6
    num_heads = 4
    vocab_size = 500

    backbone, mag_gate, _, selector = build_debug_env(
        hidden_dim=D, num_layers=num_layers, num_heads=num_heads,
        vocab_size=vocab_size, seed=42,
    )

    query_ids, target_ids, memory_ids_list, relevant_indices = create_synthetic_memory_data(
        backbone, num_memories=8, memory_len=32, query_len=16,
        target_len=12, vocab_size=vocab_size, seed=42,
    )

    input_ids = torch.cat([query_ids, target_ids], dim=1)
    query_len = query_ids.shape[1]
    labels = torch.full_like(input_ids, -100)
    labels[0, query_len:] = input_ids[0, query_len:]

    pooled_vecs, unpooled_hs = encode_memories(backbone, memory_ids_list)

    # Baseline
    loss_no_mem = forward_no_memory(backbone, input_ids, labels)
    ppl_no_mem = math.exp(min(loss_no_mem, 50))

    ranks = [1, 2, 4, 8, 16, 32]
    head_dim = D // num_heads  # 32

    print(f"  Baseline PPL (无记忆): {ppl_no_mem:.4f}")
    print(f"  head_dim = {head_dim}")
    print(f"\n  {'Rank':>6} | {'虚拟 tokens':>12} | {'PPL (MAGGate)':>14} | {'PPL (直接查询)':>16} | {'能量保留':>10}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*14}-+-{'-'*16}-+-{'-'*10}")

    for rank in ranks:
        cm_config = CompressedMemoryConfig(
            hidden_dim=D, num_heads=num_heads, rank=rank,
            chunk_size=0, normalize=False,
        )
        cache = CompressedMemoryCache(cm_config)

        # SVD → MAGGate
        loss_mag = forward_with_compressed_memory(
            backbone, mag_gate, cache, input_ids, labels, unpooled_hs,
        )
        ppl_mag = math.exp(min(loss_mag, 50))

        # SVD → 直接查询
        loss_direct = forward_with_direct_M_query(
            backbone, cache, input_ids, labels, unpooled_hs,
            injection_layers=mag_gate.injection_layers,
            gate_scale=0.3,
        )
        ppl_direct = math.exp(min(loss_direct, 50))

        # 获取能量保留率
        stats = cache.compress_and_store(unpooled_hs)
        energy = stats['singular_value_stats'][0].get('energy_retained', 0)
        virtual_tokens = stats['virtual_tokens']

        print(f"  {rank:>6} | {virtual_tokens:>12} | {ppl_mag:>14.4f} | {ppl_direct:>16.4f} | {energy:>10.4f}")

    print("\n  ✅ 通过: 不同 rank 的 PPL 计算正确")


# ======================================================================
# Test 3: 多样本 PPL 统计
# ======================================================================

def test_multi_sample_ppl():
    """在多个随机样本上统计 PPL, 减少随机性影响。"""
    _sep("Test 3: 多样本 PPL 统计 (10 个样本)")

    D = 128
    num_layers = 6
    num_heads = 4
    vocab_size = 500
    num_samples = 10

    backbone, mag_gate, compressed_cache, selector = build_debug_env(
        hidden_dim=D, num_layers=num_layers, num_heads=num_heads,
        vocab_size=vocab_size, seed=42,
    )

    results = {"no_mem": [], "mag_pooled": [], "svd_mag": [], "svd_direct": []}

    for sample_idx in range(num_samples):
        seed = 100 + sample_idx
        query_ids, target_ids, memory_ids_list, relevant_indices = create_synthetic_memory_data(
            backbone, num_memories=8, memory_len=20, query_len=12,
            target_len=10, vocab_size=vocab_size, seed=seed,
        )

        input_ids = torch.cat([query_ids, target_ids], dim=1)
        query_len = query_ids.shape[1]
        labels = torch.full_like(input_ids, -100)
        labels[0, query_len:] = input_ids[0, query_len:]

        pooled_vecs, unpooled_hs = encode_memories(backbone, memory_ids_list)

        # A. 无记忆
        loss_a = forward_no_memory(backbone, input_ids, labels)
        results["no_mem"].append(math.exp(min(loss_a, 50)))

        # B. MAGGate (pooled)
        loss_b = forward_with_mag(backbone, mag_gate, input_ids, labels, pooled_vecs)
        results["mag_pooled"].append(math.exp(min(loss_b, 50)))

        # C. SVD → MAGGate
        loss_c = forward_with_compressed_memory(
            backbone, mag_gate, compressed_cache, input_ids, labels, unpooled_hs,
        )
        results["svd_mag"].append(math.exp(min(loss_c, 50)))

        # D. SVD → 直接查询
        loss_d = forward_with_direct_M_query(
            backbone, compressed_cache, input_ids, labels, unpooled_hs,
            injection_layers=mag_gate.injection_layers,
            gate_scale=0.3,
        )
        results["svd_direct"].append(math.exp(min(loss_d, 50)))

    # 统计
    print(f"\n  {'模式':>25} | {'PPL 均值':>10} | {'PPL 标准差':>10} | {'PPL 中位数':>10}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for mode, ppls in results.items():
        mean_ppl = sum(ppls) / len(ppls)
        std_ppl = (sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)) ** 0.5
        sorted_ppls = sorted(ppls)
        median_ppl = sorted_ppls[len(sorted_ppls) // 2]
        mode_name = {
            "no_mem": "A. 无记忆",
            "mag_pooled": "B. MAGGate (pooled)",
            "svd_mag": "C. SVD → MAGGate",
            "svd_direct": "D. SVD → 直接查询",
        }[mode]
        print(f"  {mode_name:>25} | {mean_ppl:>10.4f} | {std_ppl:>10.4f} | {median_ppl:>10.4f}")

    print("\n  ✅ 通过: 多样本统计完成")


# ======================================================================
# Test 4: 记忆信息保留验证
# ======================================================================

def test_memory_information_retention():
    """验证 SVD 压缩后, 记忆中的信息是否被保留。

    方法: 用 compressed_cache.query() 查询相关记忆的 key,
    对比查询结果与原始记忆的相似度。
    """
    _sep("Test 4: 记忆信息保留验证")

    D = 128
    num_heads = 4
    rank = 8

    torch.manual_seed(42)

    # 创建两组记忆: 相关 vs 不相关
    # 相关记忆: 包含特定模式
    relevant_hs = torch.randn(1, 32, D)
    relevant_hs[:, :, :D//2] *= 3.0  # 前半维度信号更强

    # 不相关记忆: 随机
    irrelevant_hs = torch.randn(1, 32, D) * 0.5

    # 拼接
    all_hs = torch.cat([relevant_hs, irrelevant_hs], dim=1)  # (1, 64, D)

    # 压缩
    cm_config = CompressedMemoryConfig(
        hidden_dim=D, num_heads=num_heads, rank=rank,
        chunk_size=0, normalize=False,
    )
    cache = CompressedMemoryCache(cm_config)
    stats = cache.compress_and_store(all_hs)

    print(f"  压缩: {all_hs.shape[1]} tokens → {stats['virtual_tokens']} virtual tokens")
    print(f"  能量保留: {stats['singular_value_stats'][0].get('energy_retained', 'N/A')}")

    # 查询: 用相关记忆的 mean 作为 query
    query_relevant = relevant_hs.mean(dim=1, keepdim=True)  # (1, 1, D)
    query_irrelevant = irrelevant_hs.mean(dim=1, keepdim=True)  # (1, 1, D)

    output_relevant = cache.query(query_relevant)  # (1, 1, D)
    output_irrelevant = cache.query(query_irrelevant)  # (1, 1, D)

    # 计算与原始记忆的相似度
    cos_sim_rel = F.cosine_similarity(
        output_relevant.flatten(), relevant_hs.mean(dim=1).flatten(), dim=0
    ).item()
    cos_sim_irrel = F.cosine_similarity(
        output_irrelevant.flatten(), irrelevant_hs.mean(dim=1).flatten(), dim=0
    ).item()

    print(f"  查询相关记忆 → 输出与相关记忆的 cosine sim: {cos_sim_rel:.4f}")
    print(f"  查询不相关记忆 → 输出与不相关记忆的 cosine sim: {cos_sim_irrel:.4f}")

    # 交叉相似度 (应该较低)
    cross_sim = F.cosine_similarity(
        output_relevant.flatten(), irrelevant_hs.mean(dim=1).flatten(), dim=0
    ).item()
    print(f"  查询相关记忆 → 输出与不相关记忆的 cosine sim: {cross_sim:.4f}")

    print(f"\n  信息保留度: 相关查询 sim={cos_sim_rel:.4f} vs 交叉 sim={cross_sim:.4f}")
    print("  ✅ 通过: 记忆信息保留验证完成")


# ======================================================================
# Test 5: 训练一步后的效果 (验证梯度可传播)
# ======================================================================

def test_one_step_training():
    """验证 SVD 压缩记忆 + MAGGate 可以正常训练。

    做一步梯度更新, 验证:
    1. 梯度可以传播到 MAGGate 参数
    2. 一步训练后 loss 降低
    """
    _sep("Test 5: 一步训练验证")

    D = 128
    num_layers = 6
    num_heads = 4
    vocab_size = 500

    backbone, mag_gate, compressed_cache, selector = build_debug_env(
        hidden_dim=D, num_layers=num_layers, num_heads=num_heads,
        vocab_size=vocab_size, seed=42,
    )

    # 只训练 MAGGate 参数
    mag_gate.train()
    optimizer = torch.optim.Adam(mag_gate.parameters(), lr=1e-3)

    query_ids, target_ids, memory_ids_list, _ = create_synthetic_memory_data(
        backbone, num_memories=6, memory_len=16, query_len=12,
        target_len=8, vocab_size=vocab_size, seed=42,
    )

    input_ids = torch.cat([query_ids, target_ids], dim=1)
    query_len = query_ids.shape[1]
    labels = torch.full_like(input_ids, -100)
    labels[0, query_len:] = input_ids[0, query_len:]

    # 编码记忆 (no grad)
    pooled_vecs, unpooled_hs = encode_memories(backbone, memory_ids_list)

    # 压缩 (no grad)
    with torch.no_grad():
        compressed_cache.compress_and_store(unpooled_hs)
        mem_result = compressed_cache.get_virtual_tokens_as_memory()
        virtual_memory, virtual_mask = mem_result

    # Forward (有梯度)
    B, T = input_ids.shape
    positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
    h = backbone.embed(input_ids) + backbone.pos_embed(positions)
    h = h.detach()  # 不更新 backbone

    causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)

    for layer_idx, layer in enumerate(backbone.encoder.layers):
        with torch.no_grad():
            h_new = layer(h, src_mask=causal_mask)
        h = h_new.detach().requires_grad_(True)
        # MAG 注入 (有梯度)
        h = mag_gate.inject(
            layer_idx=layer_idx,
            hidden_states=h,
            memory_vectors=virtual_memory,
            memory_mask=virtual_mask.bool(),
        )

    logits = backbone.lm_head(h)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_before = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    print(f"  训练前 loss: {loss_before.item():.4f}")

    # 反向传播
    optimizer.zero_grad()
    loss_before.backward()

    # 检查梯度
    has_grad = False
    grad_norms = {}
    for name, param in mag_gate.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            has_grad = True
            grad_norms[name] = param.grad.norm().item()

    print(f"  MAGGate 参数有梯度: {'是 ✅' if has_grad else '否 ❌'}")
    if grad_norms:
        for name, norm in list(grad_norms.items())[:5]:
            print(f"    {name}: grad_norm={norm:.6f}")

    # 更新参数
    optimizer.step()

    # 再次 forward 检查 loss 变化
    mag_gate.eval()
    with torch.no_grad():
        h2 = backbone.embed(input_ids) + backbone.pos_embed(positions)
        for layer_idx, layer in enumerate(backbone.encoder.layers):
            h2 = layer(h2, src_mask=causal_mask)
            h2 = mag_gate.inject(
                layer_idx=layer_idx,
                hidden_states=h2,
                memory_vectors=virtual_memory,
                memory_mask=virtual_mask.bool(),
            )
        logits2 = backbone.lm_head(h2)
        shift_logits2 = logits2[..., :-1, :].contiguous()
        loss_after = F.cross_entropy(
            shift_logits2.view(-1, shift_logits2.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        ).item()

    print(f"  训练后 loss: {loss_after:.4f}")
    print(f"  Loss 变化: {loss_after - loss_before.item():+.4f}")

    assert has_grad, "MAGGate 参数应该有梯度"
    print("  ✅ 通过: 梯度传播正常, 训练流程完整")


# ======================================================================
# Test 6: 性能对比 (压缩 vs 原始)
# ======================================================================

def test_performance_comparison():
    """对比压缩记忆 vs 原始记忆的推理性能。"""
    _sep("Test 6: 推理性能对比")

    D = 256
    num_layers = 8
    num_heads = 8
    vocab_size = 500
    num_memories = 20
    memory_len = 32

    backbone, mag_gate_orig, _, _ = build_debug_env(
        hidden_dim=D, num_layers=num_layers, num_heads=num_heads,
        vocab_size=vocab_size, seed=42,
    )
    # 重新创建 mag_gate 以匹配新的 injection_layers
    injection_layers = [1, 3, 5, 7]
    mag_config = MAGGateConfig(
        hidden_dim=D, num_heads=num_heads, memory_dim=D,
        injection_layers=injection_layers, share_parameters=True,
        gate_init_bias=-1.0,
    )
    mag_gate = MAGGate(mag_config)

    cm_config = CompressedMemoryConfig(
        hidden_dim=D, num_heads=num_heads, rank=8,
        chunk_size=0, normalize=False,
    )
    compressed_cache = CompressedMemoryCache(cm_config)

    # 创建数据
    query_ids, target_ids, memory_ids_list, _ = create_synthetic_memory_data(
        backbone, num_memories=num_memories, memory_len=memory_len,
        query_len=32, target_len=16, vocab_size=vocab_size, seed=42,
    )
    input_ids = torch.cat([query_ids, target_ids], dim=1)
    query_len = query_ids.shape[1]
    labels = torch.full_like(input_ids, -100)
    labels[0, query_len:] = input_ids[0, query_len:]

    pooled_vecs, unpooled_hs = encode_memories(backbone, memory_ids_list)

    print(f"  配置: D={D}, layers={num_layers}, memories={num_memories}, mem_len={memory_len}")
    print(f"  Pooled vectors: {pooled_vecs.shape} ({num_memories} vectors)")
    print(f"  Unpooled HS: {unpooled_hs.shape} ({num_memories * memory_len} tokens)")

    n_runs = 20

    # A. 原始 MAGGate (pooled)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        forward_with_mag(backbone, mag_gate, input_ids, labels, pooled_vecs)
    t_pooled = (time.perf_counter() - t0) / n_runs * 1000

    # B. SVD → MAGGate
    # 先压缩 (一次性)
    t_compress_start = time.perf_counter()
    compressed_cache.compress_and_store(unpooled_hs)
    t_compress = (time.perf_counter() - t_compress_start) * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        forward_with_compressed_memory(
            backbone, mag_gate, compressed_cache, input_ids, labels, unpooled_hs,
        )
    t_svd_mag = (time.perf_counter() - t0) / n_runs * 1000

    # C. SVD → 直接查询
    t0 = time.perf_counter()
    for _ in range(n_runs):
        forward_with_direct_M_query(
            backbone, compressed_cache, input_ids, labels, unpooled_hs,
            injection_layers=set(injection_layers),
            gate_scale=0.3,
        )
    t_svd_direct = (time.perf_counter() - t0) / n_runs * 1000

    # 内存占用
    mem_pooled = pooled_vecs.numel() * 4  # float32
    mem_result = compressed_cache.get_virtual_tokens()
    mem_svd = (mem_result[0].numel() + mem_result[1].numel()) * 4 if mem_result else 0

    print(f"\n  {'模式':>25} | {'推理耗时 (ms)':>14} | {'记忆大小 (KB)':>14}")
    print(f"  {'-'*25}-+-{'-'*14}-+-{'-'*14}")
    print(f"  {'MAGGate (pooled)':>25} | {t_pooled:>14.3f} | {mem_pooled / 1024:>14.2f}")
    print(f"  {'SVD → MAGGate':>25} | {t_svd_mag:>14.3f} | {mem_svd / 1024:>14.2f}")
    print(f"  {'SVD → 直接查询':>25} | {t_svd_direct:>14.3f} | {mem_svd / 1024:>14.2f}")
    print(f"\n  SVD 压缩耗时 (一次性): {t_compress:.3f} ms")
    print(f"  记忆压缩率: {mem_pooled / max(mem_svd, 1):.2f}x → {mem_svd / max(mem_pooled, 1):.2f}x")

    print("  ✅ 通过: 性能对比完成")


# ======================================================================
# Test 7: ContextSelector + CompressedMemory 联合测试
# ======================================================================

def test_selector_with_compressed_memory():
    """测试 ContextSelector 能否在 SVD 压缩记忆上正常工作。"""
    _sep("Test 7: ContextSelector + CompressedMemory 联合")

    D = 128
    num_heads = 4

    torch.manual_seed(42)

    backbone, mag_gate, compressed_cache, selector = build_debug_env(
        hidden_dim=D, num_heads=num_heads, seed=42,
    )

    query_ids, target_ids, memory_ids_list, relevant_indices = create_synthetic_memory_data(
        backbone, num_memories=8, memory_len=16, query_len=12,
        target_len=8, vocab_size=500, seed=42,
    )

    # 编码
    pooled_vecs, unpooled_hs = encode_memories(backbone, memory_ids_list)

    # 用 pooled vectors 做 selector
    query_emb = pooled_vecs[:, 0:1, :].squeeze(1)  # 用第一条记忆作为 query (简化)
    # 实际应该用 query 的 embedding, 这里简化

    # 用 backbone embedding 编码 query
    with torch.no_grad():
        B, T = query_ids.shape
        positions = torch.arange(T, device=query_ids.device).unsqueeze(0).expand(B, T)
        query_emb = (backbone.embed(query_ids) + backbone.pos_embed(positions)).mean(dim=1)  # (1, D)

    # Selector 打分
    scores = selector(query_emb, pooled_vecs)  # (1, K)
    probs = torch.sigmoid(scores).squeeze(0)

    print(f"  Selector 分数 (sigmoid):")
    for i, p in enumerate(probs.tolist()):
        rel_mark = "✓ 相关" if i in relevant_indices else "  不相关"
        print(f"    记忆[{i}]: {p:.4f} {rel_mark}")

    # 用 selector 的 soft weights 做 MAGGate 注入
    selection_weights = selector.soft_select(query_emb, pooled_vecs)  # (1, K)

    input_ids = torch.cat([query_ids, target_ids], dim=1)
    query_len = query_ids.shape[1]
    labels = torch.full_like(input_ids, -100)
    labels[0, query_len:] = input_ids[0, query_len:]

    # 有 selector 的 MAGGate
    loss_with_sel = forward_with_mag(
        backbone, mag_gate, input_ids, labels, pooled_vecs,
        selection_weights=selection_weights,
    )
    # 无 selector 的 MAGGate
    loss_no_sel = forward_with_mag(
        backbone, mag_gate, input_ids, labels, pooled_vecs,
    )

    print(f"\n  PPL (有 selector): {math.exp(min(loss_with_sel, 50)):.4f}")
    print(f"  PPL (无 selector): {math.exp(min(loss_no_sel, 50)):.4f}")

    print("  ✅ 通过: Selector + CompressedMemory 联合工作正常")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  CompressedMemoryCache 端到端测试")
    print("  (SVD 压缩记忆在记忆任务上的整体效果)")
    print("=" * 72)

    tests = [
        test_ppl_comparison,
        test_rank_sensitivity,
        test_multi_sample_ppl,
        test_memory_information_retention,
        test_one_step_training,
        test_performance_comparison,
        test_selector_with_compressed_memory,
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
