#!/usr/bin/env python3
"""
benchmark.py — 对比 Dense vs 2:4 Sparse 推理性能。

测试方案：
  1. 加载同一个导出的 2:4 稀疏模型
  2. 先用 dense mode（普通 nn.Linear）跑 benchmark
  3. 再将权重转为 semi-structured 2:4 格式跑 benchmark
  4. 对比两者的吞吐量和延迟

同时验证两种模式的输出是否一致（数值对齐）。

用法：
    python3 deploy_sparse_24/benchmark.py \
        --model_dir deploy_sparse_24/exported_model \
        [--batch_sizes 1,4,8,16] \
        [--seq_lens 128,256,512,1024] \
        [--num_runs 20]
"""

import argparse
import os
import sys
import time
import json
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deploy_sparse_24.inference import (
    load_model,
    check_sparse_support,
    apply_sparse_semi_structured,
)


@torch.no_grad()
def profile_prefill(
    model: nn.Module,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Dict[str, float]:
    """
    精确测量 prefill（一次前向传播）的延迟。
    
    使用 CUDA events 计时（比 time.perf_counter 更准确）。
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(num_warmup):
        _ = model(input_ids)
    torch.cuda.synchronize()

    # 使用 CUDA events 精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_ms = []
    for _ in range(num_runs):
        start_event.record()
        _ = model(input_ids)
        end_event.record()
        torch.cuda.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    std_ms = (sum((x - avg_ms) ** 2 for x in latencies_ms) / len(latencies_ms)) ** 0.5

    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / (avg_ms / 1000.0)

    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "tokens_per_sec": tokens_per_sec,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def verify_numerical_alignment(
    model_dense: nn.Module,
    model_sparse: nn.Module,
    vocab_size: int,
    device: str,
    batch_size: int = 2,
    seq_len: int = 64,
) -> Dict[str, float]:
    """
    验证 dense 和 sparse 模型的输出是否数值对齐。
    
    由于 semi-structured 使用不同的 GEMM kernel，可能存在微小的浮点差异，
    但 logits 的相对误差应该很小。
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    out_dense = model_dense(input_ids)
    logits_dense = out_dense.logits if hasattr(out_dense, 'logits') else out_dense[0]

    out_sparse = model_sparse(input_ids)
    logits_sparse = out_sparse.logits if hasattr(out_sparse, 'logits') else out_sparse[0]

    # 计算各种误差度量
    abs_diff = (logits_dense.float() - logits_sparse.float()).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # 相对误差（避免除零）
    denom = logits_dense.float().abs().clamp(min=1e-8)
    rel_diff = abs_diff / denom
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # cosine similarity
    flat_dense = logits_dense.float().reshape(-1)
    flat_sparse = logits_sparse.float().reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_dense.unsqueeze(0), flat_sparse.unsqueeze(0)
    ).item()

    # top-1 token accuracy（最关键的指标：生成的 token 是否一致）
    topk_dense = logits_dense[:, -1, :].argmax(dim=-1)
    topk_sparse = logits_sparse[:, -1, :].argmax(dim=-1)
    top1_match = (topk_dense == topk_sparse).float().mean().item()

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "cosine_similarity": cos_sim,
        "top1_token_match": top1_match,
    }


def measure_gpu_memory() -> Dict[str, float]:
    """获取当前 GPU 显存使用情况。"""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser(description="Dense vs 2:4 Sparse 推理性能对比")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="导出的 2:4 稀疏 HuggingFace 模型目录")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--batch_sizes", type=str, default="1,4,8,16,32,64",
                        help="逗号分隔的 batch size 列表")
    parser.add_argument("--seq_lens", type=str, default="128,256,512,1024,2048",
                        help="逗号分隔的序列长度列表")
    parser.add_argument("--num_warmup", type=int, default=5,
                        help="预热次数")
    parser.add_argument("--num_runs", type=int, default=20,
                        help="每次测试的运行次数")
    parser.add_argument("--skip_verify", action="store_true",
                        help="跳过数值对齐验证")
    parser.add_argument("--save_results", type=str, default=None,
                        help="保存结果到 JSON 文件")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    if not check_sparse_support():
        print("❌ 当前环境不支持 2:4 稀疏加速，无法进行对比测试")
        sys.exit(1)

    # ============================================================
    # Step 1: 加载模型（Dense 模式）
    # ============================================================
    print("=" * 70)
    print("Step 1: 加载模型 (Dense 模式)")
    print("=" * 70)
    model_dense, tokenizer = load_model(args.model_dir, args.device, dtype)
    vocab_size = model_dense.config.vocab_size
    
    dense_mem = measure_gpu_memory()
    print(f"[显存] Dense 模型: allocated={dense_mem.get('allocated_mb', 0):.0f}MB")

    # ============================================================
    # Step 2: Dense Benchmark
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: Dense Benchmark (Prefill)")
    print("=" * 70)
    dense_results = {}
    for bs in batch_sizes:
        for sl in seq_lens:
            key = f"bs{bs}_sl{sl}"
            print(f"\n  测试: batch_size={bs}, seq_len={sl}")
            try:
                r = profile_prefill(model_dense, vocab_size, bs, sl,
                                    args.device, args.num_warmup, args.num_runs)
                dense_results[key] = r
                print(f"    延迟: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f}ms)")
                print(f"    吞吐: {r['tokens_per_sec']:.0f} tok/s")
            except torch.cuda.OutOfMemoryError:
                print(f"    ⚠️ OOM, 跳过")
                torch.cuda.empty_cache()

    # ============================================================
    # Step 3: 转换为 2:4 Semi-Structured
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: 将权重转换为 2:4 Semi-Structured 格式")
    print("=" * 70)
    
    # 为了数值验证，先保存一份 dense 的 state dict
    if not args.skip_verify:
        # 加载一个新的 dense 模型用于验证（因为 apply_sparse_semi_structured 会修改原模型）
        print("[INFO] 重新加载模型用于数值对齐验证...")
        model_verify_dense, _ = load_model(args.model_dir, args.device, dtype)
    
    converted = apply_sparse_semi_structured(model_dense, verbose=False)
    model_sparse = model_dense  # 同一个模型，权重已就地替换

    sparse_mem = measure_gpu_memory()
    print(f"[显存] Sparse 模型: allocated={sparse_mem.get('allocated_mb', 0):.0f}MB")
    
    mem_saved = dense_mem.get('allocated_mb', 0) - sparse_mem.get('allocated_mb', 0)
    if mem_saved > 0:
        print(f"[显存] 节省: {mem_saved:.0f}MB ({mem_saved / dense_mem.get('allocated_mb', 1) * 100:.1f}%)")

    # ============================================================
    # Step 4: 数值对齐验证
    # ============================================================
    if not args.skip_verify:
        print("\n" + "=" * 70)
        print("Step 4: 数值对齐验证 (Dense vs Sparse)")
        print("=" * 70)
        
        align = verify_numerical_alignment(
            model_verify_dense, model_sparse, vocab_size, args.device
        )
        print(f"  最大绝对误差: {align['max_abs_diff']:.6e}")
        print(f"  平均绝对误差: {align['mean_abs_diff']:.6e}")
        print(f"  最大相对误差: {align['max_rel_diff']:.6e}")
        print(f"  平均相对误差: {align['mean_rel_diff']:.6e}")
        print(f"  Cosine 相似度: {align['cosine_similarity']:.8f}")
        print(f"  Top-1 Token 匹配率: {align['top1_token_match']:.2%}")

        if align['cosine_similarity'] > 0.9999:
            print("  ✅ 数值对齐：优秀（CUTLASS 后端预期效果）")
        elif align['cosine_similarity'] > 0.999:
            print("  ✅ 数值对齐：良好")
        elif align['cosine_similarity'] > 0.99:
            print("  ⚠️ 数值对齐：一般（可能影响生成质量）")
        elif align['cosine_similarity'] > 0.95:
            print("  ⚠️ 数值对齐：较差（cuSPARSELt 后端可能重新选择非零位置）")
            print("  提示: cuSPARSELt 会重新做 2:4 选择，可能与训练时的稀疏模式不同")
            print("  建议: 使用 CUTLASS 后端（需要 A100/H100 等 SM 80-86 GPU）")
        else:
            print("  ❌ 数值对齐：差（可能有 bug 或后端兼容性问题）")
            print("  提示: 如果使用 cuSPARSELt 后端，它会重新选择非零位置，导致结果偏差大")

        # 释放验证模型
        del model_verify_dense
        torch.cuda.empty_cache()

    # ============================================================
    # Step 5: Sparse Benchmark
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 5: 2:4 Sparse Benchmark (Prefill)")
    print("=" * 70)
    sparse_results = {}
    for bs in batch_sizes:
        for sl in seq_lens:
            key = f"bs{bs}_sl{sl}"
            print(f"\n  测试: batch_size={bs}, seq_len={sl}")
            try:
                r = profile_prefill(model_sparse, vocab_size, bs, sl,
                                    args.device, args.num_warmup, args.num_runs)
                sparse_results[key] = r
                print(f"    延迟: {r['avg_ms']:.2f}ms (±{r['std_ms']:.2f}ms)")
                print(f"    吞吐: {r['tokens_per_sec']:.0f} tok/s")
            except torch.cuda.OutOfMemoryError:
                print(f"    ⚠️ OOM, 跳过")
                torch.cuda.empty_cache()

    # ============================================================
    # Step 6: 汇总对比
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 6: 性能对比汇总")
    print("=" * 70)

    print(f"\n{'Config':<20} {'Dense (ms)':<15} {'Sparse (ms)':<15} {'Speedup':<10} {'Dense tok/s':<15} {'Sparse tok/s':<15}")
    print("-" * 90)

    comparison = {}
    for key in sorted(set(list(dense_results.keys()) + list(sparse_results.keys()))):
        d = dense_results.get(key)
        s = sparse_results.get(key)
        if d and s:
            speedup = d['avg_ms'] / s['avg_ms']
            print(f"{key:<20} {d['avg_ms']:<15.2f} {s['avg_ms']:<15.2f} {speedup:<10.2f}x {d['tokens_per_sec']:<15.0f} {s['tokens_per_sec']:<15.0f}")
            comparison[key] = {
                "dense_ms": d['avg_ms'],
                "sparse_ms": s['avg_ms'],
                "speedup": speedup,
                "dense_tok_s": d['tokens_per_sec'],
                "sparse_tok_s": s['tokens_per_sec'],
            }

    if comparison:
        avg_speedup = sum(v['speedup'] for v in comparison.values()) / len(comparison)
        print(f"\n平均加速比: {avg_speedup:.2f}x")
        
        if avg_speedup > 1.3:
            print("✅ 2:4 稀疏加速效果显著！")
        elif avg_speedup > 1.1:
            print("✅ 2:4 稀疏加速效果明显")
        elif avg_speedup > 1.0:
            print("⚠️ 2:4 稀疏加速效果较小（可能是 batch size / seq_len 太小，GEMM 未被充分利用）")
        else:
            print("⚠️ 未观察到加速（可能的原因：访存瓶颈、batch 太小、overhead）")

    # ============================================================
    # 保存结果
    # ============================================================
    if args.save_results:
        results = {
            "gpu": torch.cuda.get_device_name(0),
            "dtype": args.dtype,
            "model_dir": args.model_dir,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "comparison": comparison,
            "memory": {
                "dense_mb": dense_mem.get('allocated_mb', 0),
                "sparse_mb": sparse_mem.get('allocated_mb', 0),
            },
        }
        if not args.skip_verify:
            results["numerical_alignment"] = align

        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n结果已保存: {args.save_results}")


if __name__ == "__main__":
    main()
