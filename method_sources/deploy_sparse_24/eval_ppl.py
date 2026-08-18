#!/usr/bin/env python3
"""
eval_ppl.py — 评估 2:4 稀疏模型在 WikiText-2 上的 PPL（困惑度）。

对比模式：
  1. 原始 base model (dense) 的 PPL
  2. 2:4 稀疏模型（dense 推理）的 PPL  
  3. 2:4 稀疏模型（semi-structured 加速推理）的 PPL  ← 应与 2 相同

用法：
    python3 deploy_sparse_24/eval_ppl.py \
        --model_dir deploy_sparse_24/exported_model \
        --base_model models/Qwen--Qwen3-1.7B \
        [--block_size 1024]
"""

import argparse
import os
import sys
import time
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deploy_sparse_24.inference import (
    load_model,
    check_sparse_support,
    apply_sparse_semi_structured,
)


@torch.no_grad()
def eval_wikitext2_ppl(
    model: nn.Module,
    tokenizer,
    device: str = "cuda:0",
    block_size: int = 1024,
    max_samples: int = -1,
) -> float:
    """
    在 WikiText-2 test set 上计算 PPL。
    
    Args:
        model: 模型
        tokenizer: tokenizer
        device: 设备
        block_size: 每个样本的 token 数
        max_samples: 最大样本数（-1 表示全部）
    
    Returns:
        PPL 值
    """
    from datasets import load_dataset

    # 加载 WikiText-2
    wikitext_path = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext")
    if os.path.exists(wikitext_path):
        print(f"  使用本地 WikiText-2: {wikitext_path}")
        testdata = load_dataset(wikitext_path, split="test")
    else:
        print("  从 HuggingFace 下载 WikiText-2...")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # 编码整个测试集
    text = "\n\n".join(testdata["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    nsamples = seq_len // block_size
    if max_samples > 0:
        nsamples = min(nsamples, max_samples)

    print(f"  总 tokens: {seq_len}, block_size: {block_size}, 样本数: {nsamples}")

    model.eval()
    nlls = []

    for i in range(nsamples):
        start = i * block_size
        end = start + block_size
        batch = input_ids[:, start:end]

        # 用同一个 batch 做 label（shift 由 loss 函数内部处理）
        outputs = model(batch, labels=batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        nlls.append(loss.float().item())

        if (i + 1) % 50 == 0:
            partial_ppl = math.exp(sum(nlls) / len(nlls))
            print(f"  [{i+1}/{nsamples}] partial PPL: {partial_ppl:.2f}")

    avg_nll = sum(nlls) / len(nlls)
    ppl = math.exp(avg_nll)
    return ppl


def main():
    parser = argparse.ArgumentParser(description="评估 2:4 稀疏模型的 WikiText-2 PPL")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="导出的 2:4 稀疏模型目录")
    parser.add_argument("--base_model", type=str, default=None,
                        help="原始 base model 路径（用于对比 PPL）")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最大评估样本数（-1=全部）")
    parser.add_argument("--verify_sparse_accel", action="store_true",
                        help="同时验证 semi-structured 加速后的 PPL 是否一致")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    results = {}

    # ============================================================
    # 1. 评估 base model PPL（如果提供）
    # ============================================================
    if args.base_model:
        print("=" * 60)
        print(f"评估 Base Model PPL: {args.base_model}")
        print("=" * 60)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(args.device)
        base_model.eval()

        base_ppl = eval_wikitext2_ppl(base_model, base_tokenizer, args.device,
                                       args.block_size, args.max_samples)
        print(f"\n📊 Base Model PPL: {base_ppl:.2f}")
        results["base_ppl"] = base_ppl

        del base_model
        torch.cuda.empty_cache()

    # ============================================================
    # 2. 评估 2:4 稀疏模型 PPL（Dense 推理）
    # ============================================================
    print("\n" + "=" * 60)
    print(f"评估 2:4 Sparse Model PPL (Dense 推理): {args.model_dir}")
    print("=" * 60)

    from transformers import AutoTokenizer
    model, tokenizer = load_model(args.model_dir, args.device, dtype)

    sparse_ppl = eval_wikitext2_ppl(model, tokenizer, args.device,
                                     args.block_size, args.max_samples)
    print(f"\n📊 2:4 Sparse Model PPL (Dense 推理): {sparse_ppl:.2f}")
    results["sparse_ppl_dense_inference"] = sparse_ppl

    # ============================================================
    # 3. 验证 semi-structured 加速后的 PPL
    # ============================================================
    if args.verify_sparse_accel and check_sparse_support():
        print("\n" + "=" * 60)
        print("评估 2:4 Sparse Model PPL (Semi-Structured 加速推理)")
        print("=" * 60)

        converted = apply_sparse_semi_structured(model, verbose=False)
        if converted > 0:
            accel_ppl = eval_wikitext2_ppl(model, tokenizer, args.device,
                                            args.block_size, args.max_samples)
            print(f"\n📊 2:4 Sparse Model PPL (加速推理): {accel_ppl:.2f}")
            results["sparse_ppl_accel_inference"] = accel_ppl

            diff = abs(accel_ppl - sparse_ppl)
            print(f"\n  Dense 推理 vs 加速推理 PPL 差异: {diff:.4f}")
            if diff < 0.1:
                print("  ✅ PPL 一致，semi-structured 转换正确")
            else:
                print("  ⚠️ PPL 有差异，请检查转换是否正确")
        else:
            print("  ⚠️ 没有层被转换，无法验证")

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)

    if "base_ppl" in results:
        print(f"  Base Model PPL:           {results['base_ppl']:.2f}")
    print(f"  2:4 Sparse PPL (Dense):   {results['sparse_ppl_dense_inference']:.2f}")
    if "sparse_ppl_accel_inference" in results:
        print(f"  2:4 Sparse PPL (加速):   {results['sparse_ppl_accel_inference']:.2f}")

    if "base_ppl" in results:
        ppl_increase = results['sparse_ppl_dense_inference'] - results['base_ppl']
        ppl_ratio = results['sparse_ppl_dense_inference'] / results['base_ppl']
        print(f"\n  PPL 增加: +{ppl_increase:.2f} ({ppl_ratio:.4f}x)")
        if ppl_ratio < 1.05:
            print("  ✅ 质量损失很小（<5%）")
        elif ppl_ratio < 1.10:
            print("  ✅ 质量损失可接受（<10%）")
        elif ppl_ratio < 1.20:
            print("  ⚠️ 质量损失较大（10-20%）")
        else:
            print("  ❌ 质量损失过大（>20%），可能需要更多 retrain")

    # 清理
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
