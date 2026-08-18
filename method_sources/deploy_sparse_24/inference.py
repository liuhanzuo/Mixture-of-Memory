#!/usr/bin/env python3
"""
inference.py — 使用 NVIDIA 2:4 Semi-Structured Sparsity 硬件加速进行推理。

核心思路：
  1. 加载已导出的 2:4 稀疏 HuggingFace 模型
  2. 使用 torch.sparse.to_sparse_semi_structured() 将权重转换为 CUTLASS 2:4 压缩格式
  3. 这样 nn.Linear 的 forward 会自动使用 2:4 稀疏 GEMM kernel，实现约 1.3-1.9x 加速

支持的 GPU：SM80+ (A100, H100, H800, L20A, ...)

用法：
    python3 deploy_sparse_24/inference.py \
        --model_dir deploy_sparse_24/exported_model \
        --prompt "Hello, my name is" \
        --max_new_tokens 128 \
        [--no_sparse_accel]  # 关闭加速用于对比
"""

import argparse
import os
import sys
import time
from typing import Optional, List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def check_sparse_support() -> bool:
    """检查当前环境是否支持 2:4 semi-structured sparsity 加速。"""
    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，无法使用 2:4 稀疏加速")
        return False

    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        print(f"[WARN] GPU compute capability {capability} < 8.0，不支持 2:4 稀疏加速")
        print(f"  需要 SM80+ (A100, H100, H800, L20A, ...)")
        return False

    try:
        from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
        return True
    except ImportError:
        print("[WARN] PyTorch 版本不支持 to_sparse_semi_structured")
        print(f"  当前版本: {torch.__version__}, 需要 >= 2.1.0")
        return False


def apply_sparse_semi_structured(model: nn.Module, verbose: bool = True) -> int:
    """
    将模型中所有满足 2:4 稀疏结构的 nn.Linear / Conv1D 层的权重转换为
    CUTLASS semi-structured 压缩格式。
    
    转换后，这些层的 forward 会自动使用 2:4 稀疏 GEMM kernel。
    
    注意：GPT-2 使用 Conv1D（权重形状 [in, out]），需要特殊处理。
    Conv1D 的 forward: x @ weight + bias，权重形状为 [in_features, out_features]
    而 semi-structured 要求权重形状为 [out_features, in_features]（与 nn.Linear 一致）
    因此对 Conv1D 层，先转置 → 转为 semi-structured → 替换为 nn.Linear。
    
    Args:
        model: PyTorch 模型
        verbose: 是否打印详细信息
    
    Returns:
        成功转换的层数
    """
    from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
    
    # 尝试导入 Conv1D（GPT-2 特有）
    try:
        from transformers.pytorch_utils import Conv1D
        has_conv1d = True
    except ImportError:
        has_conv1d = False
    
    # 选择 2:4 稀疏后端
    # 
    # 两种后端对比:
    # - CUTLASS 后端: 保留原有稀疏模式（只做压缩编码），数值完全对齐，overhead 较小
    #   SM 80 (A100) 上性能最优，但可能不支持某些 GPU (SM 89, SM 90 的某些变体)
    # - cuSPARSELt 后端: 使用 NVIDIA cuSPARSELt 库，支持范围更广
    #   但在某些 GPU 上 (如 H20) 可能有较大的 dispatch overhead，导致性能不如预期
    #
    # 策略: 优先用 CUTLASS（数值正确 + 性能好），如果不支持则用 cuSPARSELt
    capability = torch.cuda.get_device_capability()
    sm_version = capability[0] * 10 + capability[1]  # 如 80, 86, 89, 90
    gpu_name = torch.cuda.get_device_name(0)
    
    _use_cutlass = False
    try:
        # 先尝试 CUTLASS 后端
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
        # 使用合理大小的测试矩阵（满足所有后端的 alignment 要求）
        _test_w = torch.zeros(64, 64, dtype=torch.float16, device='cuda')
        for i in range(64):
            for j in range(0, 64, 4):
                _test_w[i, j] = 1.0
                _test_w[i, j+1] = 2.0
        _test_sparse = to_sparse_semi_structured(_test_w)
        # 做一次实际的矩阵乘法，确保 GEMM kernel 可以正常运行
        _test_input = torch.randn(1, 64, dtype=torch.float16, device='cuda')
        _test_output = torch.nn.functional.linear(_test_input, _test_sparse)
        _use_cutlass = True
        del _test_w, _test_sparse, _test_input, _test_output
        print(f"[sparse] 使用 CUTLASS 后端 (SM {sm_version}, {gpu_name})")
    except Exception as e:
        _use_cutlass = False
        cutlass_error = str(e)
        if hasattr(torch, '_cslt_sparse_mm'):
            SparseSemiStructuredTensor._FORCE_CUTLASS = False
            print(f"[sparse] CUTLASS 不支持当前 GPU (SM {sm_version}, {gpu_name})")
            print(f"[sparse]   CUTLASS 错误: {cutlass_error[:120]}")
            print(f"[sparse] 回退到 cuSPARSELt 后端")
            print(f"[sparse] ⚠️ cuSPARSELt 在部分 GPU（如 H20）上有较大 dispatch overhead，")
            print(f"[sparse]   可能导致小 batch/seq 下 Sparse 比 Dense 更慢。")
            print(f"[sparse]   建议: 使用 A100/H100（非 H20）以获得最佳 2:4 稀疏加速效果。")
        else:
            print(f"[sparse] ❌ CUTLASS 和 cuSPARSELt 均不可用 (SM {sm_version})")
            print(f"[sparse] CUTLASS 错误: {cutlass_error}")
            return 0

    # cuSPARSELt 后端兼容性有限，某些层（如 lm_head、embed）在 HuggingFace generate 中
    # 会触发 expand/slice 等操作，cuSPARSELt 不支持这些操作会导致 NotImplementedError。
    # 因此在非 CUTLASS 模式下，需要跳过这些特殊层。
    # 即使在 CUTLASS 模式下，lm_head 通常也不需要稀疏加速（只在最后一步调用一次）。
    SKIP_LAYER_PATTERNS = [
        'lm_head',        # 最终输出层，generate 时有 slice/expand 操作
        'embed',          # embedding 层
        'wte', 'wpe',     # GPT-2 的 word/position embedding
    ]
    
    converted = 0
    skipped = 0
    errors = 0
    
    # 收集需要替换的 Conv1D 层（不能在遍历时修改）
    conv1d_replacements = []

    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = has_conv1d and isinstance(module, Conv1D) and not is_linear
        
        if not is_linear and not is_conv1d:
            continue

        # 跳过特殊层（lm_head、embedding 等）
        should_skip = False
        for pattern in SKIP_LAYER_PATTERNS:
            if pattern in name:
                should_skip = True
                break
        if should_skip:
            if verbose:
                print(f"  [skip] {name}: 特殊层，不适用 2:4 稀疏转换")
            skipped += 1
            continue
        
        w = module.weight.data
        if w.dim() != 2:
            skipped += 1
            continue

        if is_conv1d:
            # Conv1D weight: [in_features, out_features]
            # 需要转置为 [out_features, in_features] 来检查和转换
            w_check = w.t().contiguous()
        else:
            # nn.Linear weight: [out_features, in_features]
            w_check = w

        out_dim, in_dim = w_check.shape
        
        # 2:4 semi-structured 要求 in_features 能被 4 整除
        # 且对于 CUTLASS 后端，通常还需要 out_features 能被 8 整除
        if in_dim % 4 != 0:
            if verbose:
                print(f"  [skip] {name}: in_features={in_dim} 不能被4整除")
            skipped += 1
            continue

        # 检查权重是否已经满足 2:4 结构
        M = 4
        in_full = (in_dim // M) * M
        core = w_check[:, :in_full]
        grouped = core.view(out_dim, in_full // M, M)
        zeros_per_group = (grouped == 0).sum(dim=-1)
        
        # 严格要求每组恰好 2 个零
        violation_ratio = (zeros_per_group != 2).float().mean().item()
        if violation_ratio > 0.01:  # 允许 1% 的容差（边界元素）
            if verbose:
                print(f"  [skip] {name}: 2:4 违反率 {violation_ratio:.2%}，不是有效的 2:4 稀疏")
            skipped += 1
            continue

        # 转换为 semi-structured 格式
        try:
            sparse_w = to_sparse_semi_structured(w_check)
            
            # 对 cuSPARSELt 后端，验证基本的矩阵乘法是否能正常工作
            if not _use_cutlass:
                try:
                    _verify_input = torch.randn(1, in_dim, dtype=w_check.dtype, device=w_check.device)
                    _verify_output = torch.nn.functional.linear(_verify_input, sparse_w)
                    del _verify_input, _verify_output
                except Exception as verify_e:
                    if verbose:
                        print(f"  [✗] {name}: cuSPARSELt 矩阵乘法验证失败 - {verify_e}")
                    errors += 1
                    continue
            
            if is_conv1d:
                # Conv1D → 替换为 nn.Linear（因为 semi-structured 只能用于 Linear 的 [out, in] 格式）
                conv1d_replacements.append((name, module, sparse_w))
            else:
                module.weight = nn.Parameter(sparse_w, requires_grad=False)
            
            converted += 1
            layer_type = "Conv1D→Linear" if is_conv1d else "Linear"
            if verbose:
                print(f"  [✓] {name}: {list(w.shape)} -> semi-structured ({layer_type})")
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [✗] {name}: 转换失败 - {e}")

    # 替换 Conv1D 层为 nn.Linear（需要在遍历结束后进行）
    for name, old_module, sparse_w in conv1d_replacements:
        # name 格式如 "transformer.h.0.attn.c_attn"
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name
        
        # 创建新的 nn.Linear 层
        out_features, in_features = sparse_w.shape
        new_linear = nn.Linear(in_features, out_features, bias=old_module.bias is not None,
                               device=sparse_w.device, dtype=sparse_w.dtype)
        new_linear.weight = nn.Parameter(sparse_w, requires_grad=False)
        if old_module.bias is not None:
            new_linear.bias = nn.Parameter(old_module.bias.data.clone(), requires_grad=False)
        
        setattr(parent, attr_name, new_linear)
        if verbose:
            print(f"  [replace] {name}: Conv1D → nn.Linear({in_features}, {out_features})")

    print(f"\n[sparse] 2:4 Semi-Structured 转换完成:")
    print(f"  成功: {converted} 层")
    print(f"  跳过: {skipped} 层")
    if errors:
        print(f"  失败: {errors} 层")

    return converted


def load_model(model_dir: str, device: str = "cuda:0", 
               dtype: torch.dtype = torch.float16) -> tuple:
    """
    加载 HuggingFace 模型和 tokenizer。
    
    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] 加载模型: {model_dir}")
    print(f"[load] 设备: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[load] 模型参数量: {num_params / 1e6:.1f}M")

    return model, tokenizer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda:0",
) -> str:
    """生成文本。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


@torch.no_grad()
def benchmark_throughput(
    model,
    tokenizer,
    device: str = "cuda:0",
    batch_sizes: List[int] = [1, 4, 8],
    seq_len: int = 512,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """
    基准测试推理吞吐量。
    
    测量 prefill（prompt 编码）的速度，因为 2:4 稀疏主要加速的是 GEMM 操作，
    prefill 阶段的 GEMM 比例远高于 decode 阶段。
    
    Returns:
        {batch_size: {"tokens_per_sec": float, "latency_ms": float}}
    """
    vocab_size = model.config.vocab_size
    results = {}

    for bs in batch_sizes:
        # 生成随机 input_ids
        input_ids = torch.randint(0, vocab_size, (bs, seq_len), device=device)

        # Warmup
        for _ in range(num_warmup):
            _ = model(input_ids)

        torch.cuda.synchronize()

        # 计时
        latencies = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        avg_latency = sum(latencies) / len(latencies)
        total_tokens = bs * seq_len
        tokens_per_sec = total_tokens / avg_latency

        results[bs] = {
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": avg_latency * 1000,
            "batch_size": bs,
            "seq_len": seq_len,
        }

        print(f"  bs={bs}, seq_len={seq_len}: "
              f"{tokens_per_sec:.0f} tok/s, "
              f"latency={avg_latency * 1000:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="2:4 稀疏加速推理")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="导出的 HuggingFace 模型目录")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="推理提示词")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="计算精度")
    parser.add_argument("--no_sparse_accel", action="store_true",
                        help="禁用 2:4 稀疏加速（用于对比测试）")
    parser.add_argument("--benchmark", action="store_true",
                        help="运行吞吐量基准测试")
    parser.add_argument("--benchmark_batch_sizes", type=str, default="1,4,8",
                        help="基准测试的 batch size 列表")
    parser.add_argument("--benchmark_seq_len", type=int, default=512,
                        help="基准测试的序列长度")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="打印详细信息")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # 1. 加载模型
    model, tokenizer = load_model(args.model_dir, args.device, dtype)

    # 2. 应用 2:4 稀疏加速
    if not args.no_sparse_accel:
        if check_sparse_support():
            print("\n[sparse] 正在将权重转换为 2:4 Semi-Structured 格式...")
            converted = apply_sparse_semi_structured(model, verbose=args.verbose)
            if converted > 0:
                print(f"\n✅ {converted} 层已启用 2:4 稀疏硬件加速")
            else:
                print("\n⚠️  没有层被转换，可能权重不满足 2:4 结构")
        else:
            print("\n⚠️  当前环境不支持 2:4 稀疏加速，将使用 dense 推理")
    else:
        print("\n[INFO] 2:4 稀疏加速已禁用（--no_sparse_accel）")

    # 3. 文本生成
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    generated = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    t1 = time.perf_counter()

    print(f"\nGenerated ({t1-t0:.2f}s):")
    print(generated)

    # 4. 基准测试
    if args.benchmark:
        batch_sizes = [int(x) for x in args.benchmark_batch_sizes.split(",")]
        print(f"\n{'='*60}")
        print(f"Throughput Benchmark (prefill)")
        print(f"{'='*60}")
        results = benchmark_throughput(
            model, tokenizer, args.device,
            batch_sizes=batch_sizes,
            seq_len=args.benchmark_seq_len,
        )


if __name__ == "__main__":
    main()
