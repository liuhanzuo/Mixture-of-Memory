"""
CompressedMemoryCache: 基于 SVD 的压缩记忆缓存。

核心思想:
    对每个记忆 chunk, 计算关联记忆矩阵 M_j = K_j^T V_j ∈ ℝ^(d_k × d_v),
    然后对 M_j 做 rank-r SVD 分解:
        M_j ≈ U_r Σ_r V_r^T
    将其转化为 r 个"虚拟 KV pairs":
        K̃_j = U_r √Σ_r ∈ ℝ^(r × d_k)
        Ṽ_j = V_r √Σ_r ∈ ℝ^(r × d_v)
    这样 K̃_j^T Ṽ_j ≈ M_j, 且可以直接拼进标准 cross attention。

优势:
    - 每个 chunk 从 N 个 token 压缩到 r 个虚拟 token (r ≪ N)
    - 保留了 chunk 内所有 token 的 KV 交互信息 (关联记忆)
    - 完全兼容标准 attention 机制, 可复用 MAGGate 的 cross attention
    - 存储代价: O(C × r × d) vs 完整 KV 的 O(C × N × d)

与现有模块的关系:
    - MemoryEncoder: 提供 unpooled hidden states (需新增方法)
    - MAGGate: 复用其 cross attention + gate 机制, 只是 memory_vectors
              从 (B, K, D) 的 pooled 向量变成 (B, C*r, D) 的虚拟 KV pairs
    - ContextSelector: 可在 chunk 级别做选择 (选哪些 chunk 参与)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CompressedMemoryConfig:
    """CompressedMemoryCache 配置。

    Attributes:
        hidden_dim: backbone 隐藏维度.
        num_heads: multi-head attention 头数 (用于分头计算 M).
        rank: SVD 保留的秩 (每个 chunk 压缩为 rank 个虚拟 token).
              推荐 4~16, 越大信息保留越多但存储越大.
        chunk_size: 每个记忆 chunk 的 token 数.
                    设为 0 表示整条记忆作为一个 chunk.
        normalize: 是否对虚拟 KV pairs 做 L2 归一化.
        use_learnable_compress: 是否使用可学习的压缩投影 (替代固定 SVD).
                                如果为 True, 用一个线性层将 M 投影到低秩空间.
        store_per_layer: 是否为每个注入层存储独立的压缩记忆.
                         如果为 False, 所有层共享同一组虚拟 KV pairs.
        max_chunks: 最大 chunk 数量 (超过则丢弃最旧的).
    """
    hidden_dim: int = 2048
    num_heads: int = 8
    rank: int = 8
    chunk_size: int = 0
    normalize: bool = True
    use_learnable_compress: bool = False
    store_per_layer: bool = False
    max_chunks: int = 64


class CompressedMemoryCache(nn.Module):
    """基于 SVD 的压缩记忆缓存。

    将记忆文本的 hidden states 压缩为虚拟 KV pairs, 可直接用于
    标准 cross attention (复用 MAGGate)。

    工作流程:
        1. 编码阶段: 接收 MemoryEncoder 输出的 unpooled hidden states
        2. 压缩阶段: 计算 M = K^T V, 做 SVD, 得到虚拟 KV pairs
        3. 查询阶段: 将虚拟 KV pairs 传给 MAGGate 做 cross attention

    Usage::

        cache = CompressedMemoryCache(config)
        cache.set_backbone_projections(mag_gate)  # 复用 MAGGate 的 K/V 投影

        # 编码 + 压缩
        hidden_states = memory_encoder.encode_texts_deep_unpooled(texts)  # (B, N, D)
        cache.compress_and_store(hidden_states)

        # 查询: 获取虚拟 KV pairs, 传给 MAGGate
        virtual_kv = cache.get_virtual_tokens()  # (B, C*r, D)
        output = mag_gate.inject(layer_idx, h, virtual_kv)
    """

    def __init__(self, config: CompressedMemoryConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = CompressedMemoryConfig()
        elif isinstance(config, dict):
            config = CompressedMemoryConfig(**{
                k: v for k, v in config.items()
                if k in CompressedMemoryConfig.__dataclass_fields__
            })
        self.config = config

        D = config.hidden_dim
        n_heads = config.num_heads
        head_dim = D // n_heads
        assert D % n_heads == 0, f"hidden_dim({D}) 必须能被 num_heads({n_heads}) 整除"

        self.head_dim = head_dim
        self.n_heads = n_heads

        # 可学习的压缩投影 (可选, 替代固定 SVD)
        self._compress_k: nn.Linear | None = None
        self._compress_v: nn.Linear | None = None
        if config.use_learnable_compress:
            # 将 head_dim 维的 M 矩阵投影到 rank 维
            # M ∈ ℝ^(head_dim × head_dim) → 低秩近似
            self._compress_k = nn.Linear(head_dim, config.rank, bias=False)
            self._compress_v = nn.Linear(head_dim, config.rank, bias=False)
            # 小初始化, 初期接近 SVD 行为
            nn.init.orthogonal_(self._compress_k.weight)
            nn.init.orthogonal_(self._compress_v.weight)

        # 缓存: 存储压缩后的虚拟 KV pairs
        # 格式: {layer_idx_or_'shared': (virtual_keys, virtual_values, chunk_mask)}
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # 用于从 backbone/MAGGate 借用的 K/V 投影 (不拥有参数)
        self._k_proj: nn.Linear | None = None
        self._v_proj: nn.Linear | None = None

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[CompressedMemoryCache] 初始化: hidden_dim={D}, "
            f"num_heads={n_heads}, rank={config.rank}, "
            f"chunk_size={config.chunk_size}, "
            f"learnable={config.use_learnable_compress}, "
            f"per_layer={config.store_per_layer}, "
            f"total_params={num_params:,}"
        )

    def set_backbone_projections(
        self,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> None:
        """绑定 K/V 投影层 (通常来自 MAGGate 的 _MAGCrossAttnBlock)。

        这些投影层的参数不由本模块管理, 只是引用。
        用于将 hidden states 投影到 K/V 空间后再计算 M = K^T V。

        Args:
            k_proj: Key 投影层, (D, D) 或 (M_dim, D).
            v_proj: Value 投影层, (D, D) 或 (M_dim, D).
        """
        self._k_proj = k_proj
        self._v_proj = v_proj
        logger.info("[CompressedMemoryCache] 已绑定 K/V 投影层")

    def clear_cache(self) -> None:
        """清空所有缓存的压缩记忆。"""
        self._cache.clear()

    # ------------------------------------------------------------------ #
    #  核心: 压缩 + 存储
    # ------------------------------------------------------------------ #

    def compress_and_store(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> dict[str, Any]:
        """将 hidden states 压缩为虚拟 KV pairs 并存入缓存。

        Args:
            hidden_states: (B, N, D) 来自 MemoryEncoder 的 unpooled 输出.
                          N 是总 token 数 (可能包含多条记忆拼接).
            attention_mask: (B, N) token 级别的 mask (1=有效, 0=padding).
            layer_idx: 如果 store_per_layer=True, 指定存储到哪一层.
                      如果为 None 且 store_per_layer=True, 则存到 'shared'.

        Returns:
            stats: 压缩统计信息 (chunk 数, 压缩率, 奇异值分布等).
        """
        B, N, D = hidden_states.shape
        r = self.config.rank
        chunk_size = self.config.chunk_size if self.config.chunk_size > 0 else N

        # 确定缓存 key
        cache_key = str(layer_idx) if (self.config.store_per_layer and layer_idx is not None) else "shared"

        # Step 1: 投影到 K/V 空间
        with torch.no_grad():
            if self._k_proj is not None and self._v_proj is not None:
                K_all = self._k_proj(hidden_states)  # (B, N, D)
                V_all = self._v_proj(hidden_states)  # (B, N, D)
            else:
                # 没有绑定投影层, 直接用 hidden states 作为 K 和 V
                K_all = hidden_states
                V_all = hidden_states

        # Step 2: 分 chunk
        num_chunks = math.ceil(N / chunk_size)
        if num_chunks > self.config.max_chunks:
            logger.warning(
                f"[CompressedMemoryCache] chunk 数 {num_chunks} 超过上限 "
                f"{self.config.max_chunks}, 截断到最近的 {self.config.max_chunks} 个"
            )
            # 保留最后 max_chunks 个 chunk (最近的记忆)
            start_chunk = num_chunks - self.config.max_chunks
            start_token = start_chunk * chunk_size
            K_all = K_all[:, start_token:]
            V_all = V_all[:, start_token:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, start_token:]
            N = K_all.shape[1]
            num_chunks = self.config.max_chunks

        # Step 3: 对每个 chunk 计算 M = K^T V, 然后 SVD 压缩
        all_virtual_keys = []
        all_virtual_values = []
        all_chunk_masks = []
        singular_value_stats = []

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)

            K_chunk = K_all[:, start:end]  # (B, n_c, D)
            V_chunk = V_all[:, start:end]  # (B, n_c, D)

            # 如果有 mask, 将 padding token 的 K/V 置零
            if attention_mask is not None:
                chunk_mask = attention_mask[:, start:end]  # (B, n_c)
                mask_3d = chunk_mask.unsqueeze(-1).to(K_chunk.dtype)  # (B, n_c, 1)
                K_chunk = K_chunk * mask_3d
                V_chunk = V_chunk * mask_3d

            # 分头计算: (B, n_heads, n_c, head_dim)
            n_c = end - start
            K_heads = K_chunk.view(B, n_c, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            V_heads = V_chunk.view(B, n_c, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            # M_h = K_h^T V_h ∈ ℝ^(B, n_heads, head_dim, head_dim)
            M_heads = torch.matmul(K_heads.transpose(-2, -1), V_heads)  # (B, n_heads, head_dim, head_dim)

            # SVD 压缩
            virtual_k, virtual_v, sv_stats = self._svd_compress(M_heads, r)
            # virtual_k: (B, n_heads, r, head_dim)
            # virtual_v: (B, n_heads, r, head_dim)

            singular_value_stats.append(sv_stats)

            # 合并头: (B, n_heads, r, head_dim) → (B, r, D)
            virtual_k = virtual_k.permute(0, 2, 1, 3).contiguous().view(B, r, D)
            virtual_v = virtual_v.permute(0, 2, 1, 3).contiguous().view(B, r, D)

            # 可选: L2 归一化
            if self.config.normalize:
                virtual_k = nn.functional.normalize(virtual_k, dim=-1)
                # 注意: 不对 V 做归一化, 保留数值信息

            all_virtual_keys.append(virtual_k)
            all_virtual_values.append(virtual_v)
            # chunk mask: 全 1 (每个虚拟 token 都有效)
            all_chunk_masks.append(torch.ones(B, r, device=hidden_states.device))

        # Step 4: 拼接所有 chunk 的虚拟 KV pairs
        virtual_keys = torch.cat(all_virtual_keys, dim=1)    # (B, C*r, D)
        virtual_values = torch.cat(all_virtual_values, dim=1)  # (B, C*r, D)
        chunk_mask = torch.cat(all_chunk_masks, dim=1)         # (B, C*r)

        # 存入缓存
        self._cache[cache_key] = (virtual_keys, virtual_values, chunk_mask)

        # 统计信息
        compression_ratio = N / (num_chunks * r) if num_chunks * r > 0 else float('inf')
        stats = {
            "num_chunks": num_chunks,
            "total_tokens": N,
            "virtual_tokens": num_chunks * r,
            "compression_ratio": compression_ratio,
            "rank": r,
            "cache_key": cache_key,
            "singular_value_stats": singular_value_stats,
        }

        logger.debug(
            f"[CompressedMemoryCache] 压缩完成: {N} tokens → "
            f"{num_chunks * r} virtual tokens (压缩率 {compression_ratio:.1f}x)"
        )

        return stats

    def _svd_compress(
        self,
        M: torch.Tensor,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """对记忆矩阵 M 做 rank-r SVD 分解, 得到虚拟 KV pairs。

        数学:
            M ≈ U_r Σ_r V_r^T
            virtual_key = U_r √Σ_r    ∈ ℝ^(r × d_k)
            virtual_value = V_r √Σ_r  ∈ ℝ^(r × d_v)
            这样 virtual_key^T @ virtual_value = U_r √Σ_r √Σ_r V_r^T = U_r Σ_r V_r^T ≈ M

        Args:
            M: (B, n_heads, d_k, d_v) 记忆矩阵.
            rank: 保留的秩.

        Returns:
            virtual_keys: (B, n_heads, r, d_k)
            virtual_values: (B, n_heads, r, d_v)
            stats: 奇异值统计信息.
        """
        B, H, dk, dv = M.shape
        r = min(rank, dk, dv)

        if self.config.use_learnable_compress and self._compress_k is not None:
            # 可学习压缩: 用线性层替代 SVD
            # M ∈ (B, H, dk, dv)
            # virtual_key = M @ W_k^T ∈ (B, H, dk, r) → 转置为 (B, H, r, dk)
            # virtual_value = M^T @ W_v^T ∈ (B, H, dv, r) → 转置为 (B, H, r, dv)
            W_k = self._compress_k.weight  # (r, head_dim)
            W_v = self._compress_v.weight  # (r, head_dim)

            # virtual_key: 从 M 的行空间提取 r 个方向
            # M @ W_k^T: (B, H, dk, dv) @ (dv, r) → (B, H, dk, r)
            virtual_keys = torch.matmul(M, W_k.t()).transpose(-2, -1)  # (B, H, r, dk)

            # virtual_value: 从 M 的列空间提取 r 个方向
            # M^T @ W_v^T: (B, H, dv, dk) @ (dk, r) → (B, H, dv, r)
            virtual_values = torch.matmul(M.transpose(-2, -1), W_v.t()).transpose(-2, -1)  # (B, H, r, dv)

            stats = {"method": "learnable", "rank": r}
            return virtual_keys, virtual_values, stats

        # 固定 SVD 压缩
        # 为了数值稳定性, 在 float32 下做 SVD
        M_f32 = M.float()

        try:
            # torch.linalg.svd: M = U @ diag(S) @ Vh
            # U: (B, H, dk, min(dk,dv))
            # S: (B, H, min(dk,dv))
            # Vh: (B, H, min(dk,dv), dv)
            U, S, Vh = torch.linalg.svd(M_f32, full_matrices=False)
        except RuntimeError as e:
            # SVD 不收敛时的 fallback: 用随机投影
            logger.warning(f"[CompressedMemoryCache] SVD 失败 ({e}), 使用随机投影 fallback")
            virtual_keys = torch.randn(B, H, r, dk, device=M.device, dtype=M.dtype) * 0.01
            virtual_values = torch.randn(B, H, r, dv, device=M.device, dtype=M.dtype) * 0.01
            stats = {"method": "random_fallback", "rank": r}
            return virtual_keys, virtual_values, stats

        # 取 top-r
        U_r = U[:, :, :, :r]      # (B, H, dk, r)
        S_r = S[:, :, :r]          # (B, H, r)
        Vh_r = Vh[:, :, :r, :]     # (B, H, r, dv)

        # √Σ
        sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))  # (B, H, r)

        # virtual_key = U_r @ diag(√Σ) → (B, H, dk, r) @ (B, H, r, r) → (B, H, dk, r)
        # 简化: 直接逐元素乘
        # U_r: (B, H, dk, r), sqrt_S: (B, H, r) → (B, H, 1, r)
        virtual_keys = (U_r * sqrt_S.unsqueeze(-2)).transpose(-2, -1)  # (B, H, r, dk)

        # virtual_value = diag(√Σ) @ Vh_r → (B, H, r, dv)
        virtual_values = Vh_r * sqrt_S.unsqueeze(-1)  # (B, H, r, dv)

        # 转回原始 dtype
        virtual_keys = virtual_keys.to(dtype=M.dtype)
        virtual_values = virtual_values.to(dtype=M.dtype)

        # 统计: 奇异值分布 (用于监控压缩质量)
        with torch.no_grad():
            total_energy = (S_f32 := S.float()).pow(2).sum(dim=-1)  # (B, H)
            retained_energy = S_f32[:, :, :r].pow(2).sum(dim=-1)   # (B, H)
            energy_ratio = (retained_energy / total_energy.clamp(min=1e-12)).mean().item()

        stats = {
            "method": "svd",
            "rank": r,
            "energy_retained": energy_ratio,
            "top_singular_value": S_r[:, :, 0].mean().item(),
            "min_singular_value": S_r[:, :, -1].mean().item(),
        }

        return virtual_keys, virtual_values, stats

    # ------------------------------------------------------------------ #
    #  查询接口
    # ------------------------------------------------------------------ #

    def get_virtual_tokens(
        self,
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """获取缓存的虚拟 KV pairs。

        返回的 virtual_keys 和 virtual_values 可以直接作为
        MAGGate cross attention 的 memory_vectors 使用。

        但注意: MAGGate 的 _MAGCrossAttnBlock 会对 memory_vectors
        再做一次 k_proj/v_proj 投影。而我们的虚拟 KV pairs 已经
        在 K/V 空间了。所以有两种使用方式:

        方式 A (推荐): 绕过 MAGGate 的 k_proj/v_proj, 直接用虚拟 KV
        方式 B: 不绑定 k_proj/v_proj, 让 hidden states 直接作为 K/V,
               然后 MAGGate 的投影层会学习适配

        Args:
            layer_idx: 如果 store_per_layer=True, 指定获取哪一层的缓存.

        Returns:
            (virtual_keys, virtual_values): 各 (B, C*r, D), 或 None (缓存为空).
        """
        cache_key = str(layer_idx) if (self.config.store_per_layer and layer_idx is not None) else "shared"

        if cache_key not in self._cache:
            return None

        virtual_keys, virtual_values, _ = self._cache[cache_key]
        return virtual_keys, virtual_values

    def get_virtual_tokens_as_memory(
        self,
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """获取虚拟 tokens, 格式兼容 MAGGate.inject() 的 memory_vectors 参数。

        将 virtual_keys 和 virtual_values 合并为一个 tensor,
        以便直接传给 MAGGate。

        注意: 这种方式下, MAGGate 的 k_proj/v_proj 会再次投影虚拟 tokens。
        如果你在 compress_and_store 时已经用了 MAGGate 的 k_proj/v_proj,
        那么这里会产生"双重投影"。建议使用 get_virtual_tokens() 配合
        自定义的 cross attention 逻辑。

        Args:
            layer_idx: 层索引.

        Returns:
            (memory_vectors, memory_mask):
                memory_vectors: (B, C*r, D) 虚拟记忆向量 (keys 和 values 的均值).
                memory_mask: (B, C*r) 有效性 mask.
        """
        cache_key = str(layer_idx) if (self.config.store_per_layer and layer_idx is not None) else "shared"

        if cache_key not in self._cache:
            return None

        virtual_keys, virtual_values, chunk_mask = self._cache[cache_key]

        # 简单策略: 用 (key + value) / 2 作为 memory_vectors
        # MAGGate 会再用自己的 k_proj/v_proj 投影
        memory_vectors = (virtual_keys + virtual_values) / 2.0

        return memory_vectors, chunk_mask

    # ------------------------------------------------------------------ #
    #  直接查询 (绕过 MAGGate, 自己做 cross attention)
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_states: torch.Tensor,
        layer_idx: int | None = None,
        scale: float | None = None,
    ) -> torch.Tensor | None:
        """直接用 query 查询压缩记忆, 绕过 MAGGate。

        这是最"纯净"的使用方式: 虚拟 KV pairs 已经在 K/V 空间,
        直接做 multi-head cross attention, 不需要额外投影。

        公式:
            attn = softmax(Q @ K̃^T / √d) @ Ṽ

        Args:
            query_states: (B, T, D) 来自 backbone 的 hidden states.
                         注意: 这里的 Q 需要已经经过 q_proj 投影,
                         或者你需要自己做投影。
            layer_idx: 层索引.
            scale: attention 缩放因子, 默认 1/√head_dim.

        Returns:
            output: (B, T, D) cross attention 输出, 或 None (缓存为空).
        """
        result = self.get_virtual_tokens(layer_idx)
        if result is None:
            return None

        virtual_keys, virtual_values = result
        B, T, D = query_states.shape
        r_total = virtual_keys.shape[1]  # C * r

        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        # 分头
        Q = query_states.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        K = virtual_keys.view(B, r_total, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, C*r, hd)
        V = virtual_values.view(B, r_total, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, C*r, hd)

        # Attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, C*r)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, V)  # (B, H, T, hd)
        output = output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        return output

    # ------------------------------------------------------------------ #
    #  增量更新
    # ------------------------------------------------------------------ #

    def append_chunk(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> dict[str, Any]:
        """增量添加一个新的记忆 chunk 到缓存。

        与 compress_and_store 不同, 这个方法不会清空现有缓存,
        而是将新 chunk 的虚拟 KV pairs 追加到已有缓存后面。

        Args:
            hidden_states: (B, N_new, D) 新 chunk 的 hidden states.
            attention_mask: (B, N_new) mask.
            layer_idx: 层索引.

        Returns:
            stats: 压缩统计信息.
        """
        cache_key = str(layer_idx) if (self.config.store_per_layer and layer_idx is not None) else "shared"

        # 先压缩新 chunk
        # 临时保存旧缓存
        old_cache = self._cache.get(cache_key)

        # 压缩新数据 (会覆盖缓存)
        stats = self.compress_and_store(hidden_states, attention_mask, layer_idx)

        if old_cache is not None:
            old_keys, old_values, old_mask = old_cache
            new_keys, new_values, new_mask = self._cache[cache_key]

            # 拼接
            combined_keys = torch.cat([old_keys, new_keys], dim=1)
            combined_values = torch.cat([old_values, new_values], dim=1)
            combined_mask = torch.cat([old_mask, new_mask], dim=1)

            # 检查是否超过 max_chunks 限制
            max_virtual_tokens = self.config.max_chunks * self.config.rank
            if combined_keys.shape[1] > max_virtual_tokens:
                # 截断最旧的
                excess = combined_keys.shape[1] - max_virtual_tokens
                combined_keys = combined_keys[:, excess:]
                combined_values = combined_values[:, excess:]
                combined_mask = combined_mask[:, excess:]
                logger.debug(
                    f"[CompressedMemoryCache] 缓存溢出, 丢弃最旧的 "
                    f"{excess} 个虚拟 token"
                )

            self._cache[cache_key] = (combined_keys, combined_values, combined_mask)
            stats["total_virtual_tokens"] = combined_keys.shape[1]

        return stats

    # ------------------------------------------------------------------ #
    #  Per-Layer Multi-Chunk SVD 压缩 (核心新功能)
    # ------------------------------------------------------------------ #

    def compress_and_store_per_layer(
        self,
        per_layer_hidden_states: dict[int, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """对每层的 hidden states 独立做 multi-chunk SVD 压缩并存入缓存。

        这是 Per-Layer Multi-Chunk SVD 方案的核心方法:
        - 接收 MemoryEncoder.encode_texts_deep_all_layers() 的输出
        - 对每个目标层独立做 chunk 切分 + M=K^TV + SVD 压缩
        - 每层的虚拟 KV pairs 独立存储, 推理时第 l 层用第 l 层的 KV

        数学:
            对每层 l, 每个 chunk c:
                M_{c,h}^{(l)} = Σ_i k_{h,i}^{(l)} v_{h,i}^{(l)T}
                SVD: M ≈ U_r Σ_r V_r^T
                K̃_{c}^{(l)} = U_r √Σ_r, Ṽ_{c}^{(l)} = V_r √Σ_r
            推理时拼接所有 chunk:
                K̃_all^{(l)} = [K̃_1^{(l)}; K̃_2^{(l)}; ...; K̃_C^{(l)}]

        Args:
            per_layer_hidden_states: {layer_idx: (B, N, D)} 每层的 unpooled hidden states.
                                    来自 MemoryEncoder.encode_texts_deep_all_layers().
            attention_mask: (B, N) token 级别的 mask (1=有效, 0=padding).
                           所有层共享同一个 mask.

        Returns:
            stats: 压缩统计信息 (每层的 chunk 数, 压缩率等).
        """
        if not per_layer_hidden_states:
            return {"error": "empty per_layer_hidden_states"}

        all_stats = {}
        total_layers = 0
        total_virtual_tokens = 0

        # 确保 per-layer 存储模式开启
        original_store_per_layer = self.config.store_per_layer
        self.config.store_per_layer = True

        for layer_idx, hidden_states in per_layer_hidden_states.items():
            # 对每层调用已有的 compress_and_store, 使用 per-layer 存储
            layer_stats = self.compress_and_store(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
            )
            all_stats[f"layer_{layer_idx}"] = layer_stats
            total_layers += 1
            total_virtual_tokens += layer_stats.get("virtual_tokens", 0)

        # 恢复原始配置 (保持向后兼容)
        self.config.store_per_layer = original_store_per_layer

        all_stats["summary"] = {
            "total_layers": total_layers,
            "total_virtual_tokens_all_layers": total_virtual_tokens,
            "layers": sorted(per_layer_hidden_states.keys()),
        }

        logger.debug(
            f"[CompressedMemoryCache] Per-layer 压缩完成: "
            f"{total_layers} 层, 每层 {total_virtual_tokens // max(total_layers, 1)} 虚拟 tokens"
        )

        return all_stats

    def get_virtual_tokens_for_layer(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """获取指定层的虚拟 KV pairs (Per-Layer 模式专用)。

        与 get_virtual_tokens 的区别:
        - get_virtual_tokens: 通用接口, 根据 store_per_layer 配置决定行为
        - get_virtual_tokens_for_layer: 专门用于 per-layer 模式, 直接按层索引查询

        Args:
            layer_idx: backbone 层索引.

        Returns:
            (virtual_keys, virtual_values): 各 (B, C*r, D), 或 None (该层无缓存).
        """
        cache_key = str(layer_idx)
        if cache_key not in self._cache:
            return None
        virtual_keys, virtual_values, _ = self._cache[cache_key]
        return virtual_keys, virtual_values

    def get_virtual_tokens_as_memory_for_layer(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """获取指定层的虚拟 tokens, 格式兼容 MAGGate.inject() 的 memory_vectors 参数。

        将 virtual_keys 和 virtual_values 合并为一个 tensor,
        以便直接传给 MAGGate。

        Args:
            layer_idx: backbone 层索引.

        Returns:
            (memory_vectors, memory_mask):
                memory_vectors: (B, C*r, D) 虚拟记忆向量.
                memory_mask: (B, C*r) 有效性 mask.
            或 None (该层无缓存).
        """
        cache_key = str(layer_idx)
        if cache_key not in self._cache:
            return None
        virtual_keys, virtual_values, chunk_mask = self._cache[cache_key]
        # 用 (key + value) / 2 作为 memory_vectors
        memory_vectors = (virtual_keys + virtual_values) / 2.0
        return memory_vectors, chunk_mask

    def has_per_layer_cache(self) -> bool:
        """检查是否有 per-layer 缓存 (即缓存 key 是数字而非 'shared')。"""
        return any(k.isdigit() for k in self._cache.keys())

    def get_cached_layers(self) -> list[int]:
        """返回所有有缓存的层索引列表。"""
        return sorted(int(k) for k in self._cache.keys() if k.isdigit())

    # ------------------------------------------------------------------ #
    #  工具方法
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """返回缓存统计信息。"""
        stats = {
            "num_cache_entries": len(self._cache),
            "config": {
                "rank": self.config.rank,
                "chunk_size": self.config.chunk_size,
                "max_chunks": self.config.max_chunks,
                "normalize": self.config.normalize,
                "learnable": self.config.use_learnable_compress,
                "per_layer": self.config.store_per_layer,
            },
        }
        for key, (vk, vv, mask) in self._cache.items():
            stats[f"cache_{key}"] = {
                "virtual_tokens": vk.shape[1],
                "shape": list(vk.shape),
                "device": str(vk.device),
                "dtype": str(vk.dtype),
            }
        return stats

    def __repr__(self) -> str:
        cached = {k: v[0].shape[1] for k, v in self._cache.items()}
        return (
            f"CompressedMemoryCache(rank={self.config.rank}, "
            f"heads={self.n_heads}, head_dim={self.head_dim}, "
            f"cached={cached})"
        )
