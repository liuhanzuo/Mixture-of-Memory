#!/usr/bin/env python3
"""
convert.py — 从 AST 训练 checkpoint 中提取 2:4 稀疏权重，导出为标准 HuggingFace 模型。

功能：
1. 加载训练时的 checkpoint（包含 soft mask 和权重）
2. 对每一层的权重应用 2:4 hard mask（每4个元素保留绝对值最大的2个）
3. 将被 mask 的权重置零
4. 保存为标准 HuggingFace 格式（可直接用 AutoModelForCausalLM.from_pretrained 加载）

用法：
    python3 deploy_sparse_24/convert.py \
        --checkpoint /path/to/model.pt \
        --base_model models/Qwen--Qwen3-1.7B \
        --output_dir deploy_sparse_24/exported_model \
        [--model_type qwen]
"""

import argparse
import os
import sys
import json
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

# 添加上层目录到 path，以便导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def apply_nm_24_mask(weight: torch.Tensor, N: int = 2, M: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 2D 权重矩阵应用 2:4 结构化稀疏 mask。
    
    每 M 个连续元素中，保留绝对值最大的 N 个，其余置零。
    
    Args:
        weight: (out_features, in_features) 权重矩阵
        N: 每组保留的元素数量
        M: 每组的总元素数量
    
    Returns:
        (pruned_weight, binary_mask) 二元组
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    out_dim, in_dim = weight.shape
    in_full = (in_dim // M) * M

    # 默认全部保留
    mask = torch.ones_like(weight)
    if in_full == 0:
        return weight, mask

    # 对可被 M 整除的部分应用 N:M 稀疏
    core = weight[:, :in_full].abs().float()
    groups = in_full // M
    grouped = core.view(out_dim, groups, M)

    # 每组中选 top-N
    topi = torch.topk(grouped, k=N, dim=-1, largest=True).indices
    group_mask = torch.zeros_like(grouped)
    group_mask.scatter_(-1, topi, 1.0)

    mask[:, :in_full] = group_mask.view(out_dim, in_full)
    pruned_weight = weight * mask.to(dtype=weight.dtype)

    return pruned_weight, mask


def apply_mask_from_checkpoint(weight: torch.Tensor, soft_mask: torch.Tensor, 
                                N: int = 2, M: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 checkpoint 中的 soft mask 来生成 hard mask 并应用。
    
    与 sparse_modeling.py 中 _hard_mask_from_soft 的 nm_2_4 分支一致：
    按 soft_mask 值的排序（而非权重绝对值）选择 top-N。
    
    Args:
        weight: 权重矩阵
        soft_mask: soft mask（0-1 连续值）
        N, M: N:M 稀疏参数
    
    Returns:
        (pruned_weight, binary_mask)
    """
    assert weight.dim() == 2 and soft_mask.dim() == 2
    out_dim, in_dim = weight.shape
    in_full = (in_dim // M) * M

    mask = torch.ones_like(weight)
    if in_full == 0:
        return weight, mask

    core = soft_mask[:, :in_full].float()
    groups = in_full // M
    grouped = core.view(out_dim, groups, M)

    topi = torch.topk(grouped, k=N, dim=-1, largest=True).indices
    group_mask = torch.zeros_like(grouped)
    group_mask.scatter_(-1, topi, 1.0)

    mask[:, :in_full] = group_mask.view(out_dim, in_full)
    pruned_weight = weight * mask.to(dtype=weight.dtype)

    return pruned_weight, mask


def load_checkpoint(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    加载 checkpoint，返回 (state_dict, args_dict)。
    支持多种 checkpoint 格式。
    """
    print(f"[convert] 加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    args_dict = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # 移除 "student." 前缀（来自蒸馏模式的 Distill_Model 包装）
    if any(k.startswith("student.") for k in state_dict.keys()):
        print("[convert] 检测到 'student.' 前缀，正在移除...")
        state_dict = {
            k[len("student."):] if k.startswith("student.") else k: v
            for k, v in state_dict.items()
        }

    # 检测并移除 wrapper 类的 "model." 前缀
    # QwenSparse/LLaMASparse 等训练 wrapper 有一个 self.model 属性指向 HF 模型，
    # 这会在 state_dict key 中添加一个额外的 "model." 前缀。
    # 例如: model.model.layers.0.self_attn.q_proj.weight -> model.layers.0.self_attn.q_proj.weight
    # 检测方法: 如果同时存在 "model.model." 开头的 key（表示 wrapper.model -> HFModel.model），
    # 说明有 wrapper 前缀需要移除。
    has_wrapper_prefix = any(k.startswith("model.model.") for k in state_dict.keys())
    has_wrapper_lm_head = any(k.startswith("model.lm_head.") for k in state_dict.keys())
    if has_wrapper_prefix or has_wrapper_lm_head:
        # 进一步验证：wrapper 前缀下的 key 应该包含 HF 模型的标准结构
        # 移除第一个 "model." 前缀
        print("[convert] 检测到 wrapper 类的 'model.' 前缀（QwenSparse/LLaMASparse），正在移除...")
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k[len("model."):]
                new_sd[new_key] = v
            else:
                new_sd[k] = v
        state_dict = new_sd
        # 打印示例 key 帮助调试
        sample_keys = list(state_dict.keys())[:5]
        print(f"[convert] 前缀移除后示例 keys: {sample_keys}")

    return state_dict, args_dict


def convert_state_dict_to_sparse_24(
    state_dict: Dict[str, torch.Tensor],
    use_soft_mask: bool = True,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    将 state_dict 中的所有 Linear 权重转换为 2:4 稀疏格式。
    
    如果 use_soft_mask=True 且 checkpoint 中包含对应的 .mask，则使用 soft mask 排序；
    否则使用权重绝对值排序。
    
    Args:
        state_dict: 模型 state dict
        use_soft_mask: 是否优先使用 soft mask 排序
    
    Returns:
        (clean_state_dict, sparsity_report):
        - clean_state_dict: 不含 mask/训练辅助张量的 state dict
        - sparsity_report: 每层的稀疏度统计
    """
    # 找出所有 mask key
    mask_keys = {k for k in state_dict if k.endswith(".mask")}
    # 找出所有 weight key（2D）
    weight_keys = {k for k in state_dict if k.endswith(".weight") and state_dict[k].dim() == 2}

    # 训练辅助张量，不需要导出
    training_patterns = [
        ".mask", ".hessian_diag", ".frozen_mask_flags", ".grad_ema",
        ".importance_ema", ".scaler_row",
        ".SLoRB_Weight", ".x_proj",  # SLoRB 低秩补偿的辅助权重
    ]

    report = {}
    converted = 0
    total_params = 0
    total_zeros = 0

    clean_sd = {}
    for key, tensor in state_dict.items():
        # 跳过训练辅助张量
        if any(key.endswith(pat) for pat in training_patterns):
            continue

        if key in weight_keys:
            # 尝试查找对应的 mask
            mask_key = key.replace(".weight", ".mask")
            if use_soft_mask and mask_key in mask_keys:
                soft_mask = state_dict[mask_key]
                if soft_mask.shape == tensor.shape:
                    pruned_w, binary_mask = apply_mask_from_checkpoint(tensor, soft_mask)
                else:
                    print(f"[convert] 警告: {mask_key} 形状不匹配 ({soft_mask.shape} vs {tensor.shape})，使用权重绝对值")
                    pruned_w, binary_mask = apply_nm_24_mask(tensor)
            else:
                # 没有 mask，使用权重绝对值排序
                pruned_w, binary_mask = apply_nm_24_mask(tensor)

            clean_sd[key] = pruned_w

            # 统计
            numel = tensor.numel()
            zeros = (pruned_w == 0).sum().item()
            sparsity = zeros / numel if numel > 0 else 0
            report[key] = {"shape": list(tensor.shape), "sparsity": sparsity}
            total_params += numel
            total_zeros += zeros
            converted += 1
        else:
            # 非 weight 或 1D 张量（如 bias, layernorm），直接保留
            clean_sd[key] = tensor

    global_sparsity = total_zeros / total_params if total_params > 0 else 0
    print(f"\n[convert] 转换完成:")
    print(f"  转换层数: {converted}")
    print(f"  全局稀疏度: {global_sparsity * 100:.2f}%")
    print(f"  总参数: {total_params:,}")
    print(f"  零参数: {total_zeros:,}")

    report["__global__"] = {
        "total_layers": converted,
        "total_params": total_params,
        "total_zeros": total_zeros,
        "global_sparsity": global_sparsity,
    }

    return clean_sd, report


def verify_24_structure(weight: torch.Tensor, M: int = 4) -> Tuple[bool, float]:
    """
    验证 2D 权重是否满足 2:4 结构化稀疏约束。
    
    Returns:
        (is_valid, violation_ratio): 
        - is_valid: 所有可覆盖区域是否满足 2:4
        - violation_ratio: 违反约束的组的比例
    """
    if weight.dim() != 2:
        return True, 0.0
    out_dim, in_dim = weight.shape
    in_full = (in_dim // M) * M
    if in_full == 0:
        return True, 0.0

    core = weight[:, :in_full]
    grouped = core.view(out_dim, in_full // M, M)
    zeros_per_group = (grouped == 0).sum(dim=-1)  # (out_dim, groups)
    # 2:4 要求每组恰好 2 个零
    violations = (zeros_per_group != 2).sum().item()
    total_groups = zeros_per_group.numel()
    violation_ratio = violations / total_groups if total_groups > 0 else 0

    return violation_ratio == 0, violation_ratio


# GPT-2 Conv1D 层的权重后缀列表
# GPT-2 HuggingFace 模型内部使用 Conv1D，权重形状为 [in_features, out_features]
# 而训练代码（model.py）将其转置为 nn.Linear 的 [out_features, in_features]
# 导出时需要转置回去
GPT2_CONV1D_WEIGHT_SUFFIXES = [
    'attn.c_attn.weight',
    'attn.c_proj.weight', 
    'mlp.c_fc.weight',
    'mlp.c_proj.weight',
]


def export_to_hf(
    clean_state_dict: Dict[str, torch.Tensor],
    base_model_path: str,
    output_dir: str,
    model_type: str = "qwen",
):
    """
    将稀疏化后的 state dict 保存为标准 HuggingFace 模型目录。
    
    步骤：
    1. 从 base_model 复制 config / tokenizer 等元数据
    2. 对 GPT-2 的 Conv1D 权重做转置（训练时是 [out,in]，HF 期望 [in,out]）
    3. 保存稀疏化的权重
    """
    from transformers import AutoConfig, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # 复制 config
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    config.save_pretrained(output_dir)

    # 复制 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        print(f"[export] Tokenizer 已保存到 {output_dir}")
    except Exception as e:
        print(f"[export] 警告: Tokenizer 保存失败: {e}")

    # GPT-2 特殊处理：将 Conv1D 层的权重从 [out, in] 转置回 [in, out]
    if model_type == "gpt2":
        transposed_count = 0
        for key in list(clean_state_dict.keys()):
            if any(key.endswith(suffix) for suffix in GPT2_CONV1D_WEIGHT_SUFFIXES):
                w = clean_state_dict[key]
                if w.dim() == 2:
                    clean_state_dict[key] = w.t().contiguous()
                    transposed_count += 1
        print(f"[export] GPT-2 Conv1D 权重转置: {transposed_count} 层")

    # 保存权重（使用 safetensors 格式）
    try:
        from safetensors.torch import save_file as safetensors_save
        weights_path = os.path.join(output_dir, "model.safetensors")
        # safetensors 要求所有 tensor 在 CPU 且是 contiguous
        save_dict = {k: v.contiguous().cpu() for k, v in clean_state_dict.items()}
        safetensors_save(save_dict, weights_path)
        print(f"[export] 权重已保存（safetensors）: {weights_path}")
    except ImportError:
        weights_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(clean_state_dict, weights_path)
        print(f"[export] 权重已保存（pytorch）: {weights_path}")

    print(f"\n[export] 模型已导出到: {output_dir}")
    print(f"  可以直接使用 AutoModelForCausalLM.from_pretrained('{output_dir}') 加载")


def main():
    parser = argparse.ArgumentParser(description="将 AST 训练的 2:4 稀疏模型导出为 HuggingFace 格式")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="AST 训练 checkpoint 路径 (model.pt)")
    parser.add_argument("--base_model", type=str, required=True,
                        help="基础模型路径 (用于复制 config 和 tokenizer)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="导出目录")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=["qwen", "llama", "opt", "mistral", "gpt2"],
                        help="模型类型")
    parser.add_argument("--use_weight_magnitude", action="store_true",
                        help="使用权重绝对值（而非 soft mask）来决定保留哪些元素")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="验证导出的权重是否满足 2:4 结构约束")
    parser.add_argument("--save_report", action="store_true", default=True,
                        help="保存稀疏度统计报告")
    args = parser.parse_args()

    # 1. 加载 checkpoint
    state_dict, args_dict = load_checkpoint(args.checkpoint)

    # 自动检测模型类型（如果 checkpoint 包含 args）
    if args_dict:
        ckpt_model = args_dict.get('student_model', '')
        if 'gpt2' in ckpt_model.lower():
            detected_type = 'gpt2'
        elif 'qwen' in ckpt_model.lower():
            detected_type = 'qwen'
        elif 'llama' in ckpt_model.lower():
            detected_type = 'llama'
        elif 'opt' in ckpt_model.lower():
            detected_type = 'opt'
        elif 'mistral' in ckpt_model.lower():
            detected_type = 'mistral'
        else:
            detected_type = None
        
        if detected_type and detected_type != args.model_type:
            print(f"[convert] 自动检测到模型类型: {detected_type}（覆盖命令行参数 {args.model_type}）")
            args.model_type = detected_type

    # 打印 checkpoint 信息
    total_keys = len(state_dict)
    mask_keys = sum(1 for k in state_dict if ".mask" in k)
    weight_keys = sum(1 for k in state_dict if k.endswith(".weight"))
    print(f"[convert] Checkpoint 信息:")
    print(f"  总 keys: {total_keys}")
    print(f"  weight keys: {weight_keys}")
    print(f"  mask keys: {mask_keys}")
    print(f"  模型类型: {args.model_type}")
    if args_dict:
        print(f"  训练 hard_mask_type: {args_dict.get('hard_mask_type', 'N/A')}")
        print(f"  训练 sparsity_ratio: {args_dict.get('sparsity_ratio', 'N/A')}")

    # 2. 转换为 2:4 稀疏
    use_soft_mask = not args.use_weight_magnitude
    clean_sd, report = convert_state_dict_to_sparse_24(state_dict, use_soft_mask=use_soft_mask)

    # 3. 验证 2:4 结构
    if args.verify:
        print("\n[verify] 验证 2:4 结构化稀疏...")
        violations = 0
        checked = 0
        for key, tensor in clean_sd.items():
            if tensor.dim() == 2 and key.endswith(".weight"):
                is_valid, vr = verify_24_structure(tensor)
                checked += 1
                if not is_valid:
                    violations += 1
                    print(f"  ❌ {key}: violation_ratio={vr:.4f}")
        if violations == 0:
            print(f"  ✅ 所有 {checked} 层满足 2:4 结构约束")
        else:
            print(f"  ⚠️  {violations}/{checked} 层存在 2:4 违反")

    # 4. 导出为 HuggingFace 格式
    export_to_hf(clean_sd, args.base_model, args.output_dir, args.model_type)

    # 5. 保存报告
    if args.save_report:
        report_path = os.path.join(args.output_dir, "sparsity_report.json")
        # 转换为可序列化格式
        serializable_report = {}
        for k, v in report.items():
            serializable_report[k] = {kk: (vv if not isinstance(vv, float) else round(vv, 6)) for kk, vv in v.items()}
        with open(report_path, "w") as f:
            json.dump(serializable_report, f, indent=2)
        print(f"[convert] 稀疏度报告已保存: {report_path}")


if __name__ == "__main__":
    main()
