#!/usr/bin/env python3
"""
MAC (Memory-Augmented Context) 评估脚本。

评估维度:
1. Selector 准确率: Hit@K, Precision@K, NDCG@K
2. PPL 对比: 有 prefix 注入 vs 无 prefix 的困惑度差异
3. 生成质量对比: 有/无记忆的生成文本差异

与 eval_mag.py 的关键区别:
- 不需要 MAGGate, 不需要分段 forward + 逐层注入
- 直接用 backbone(inputs_embeds=cat[prefix, prompt]) 做标准前向
- 生成时使用 model.generate(inputs_embeds=...) , 无需手动自回归循环

用法:
    python scripts/eval_mac.py \\
        --model_path ../models/Qwen--Qwen3-8b/ \\
        --mac_weights_dir outputs/mac_trained \\
        --data_path data/mag_train_generated.jsonl \\
        --num_eval_samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.memory.mag.memory_encoder import MemoryEncoder, MemoryEncoderConfig
from src.memory.mag.context_selector import ContextSelector, ContextSelectorConfig
from src.memory.mag.prefix_projector import PrefixProjector, PrefixProjectorConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("eval_mac")


# ======================================================================
# 参数解析
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAC 评估脚本")

    parser.add_argument("--model_path", type=str, required=True, help="Backbone 模型路径")
    parser.add_argument("--mac_weights_dir", type=str, required=True,
                        help="MAC 训练权重目录 (含 context_selector.pt, prefix_projector.pt, mac_config.json)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--data_path", type=str, required=True, help="评估数据路径 (JSONL)")
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--num_generate_samples", type=int, default=5, help="生成对比样本数")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--deep_encode_layers", type=int, default=8)
    parser.add_argument("--sliding_window", type=int, default=0,
                        help="加载模型时启用 SWA (需与训练时一致)")

    parser.add_argument("--output_file", type=str, default="", help="评估结果 JSON 输出路径")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ======================================================================
# 模型加载
# ======================================================================

def load_backbone(model_path: str, device: str, dtype_str: str, sliding_window: int = 0):
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.float32)
    logger.info(f"加载 backbone: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if sliding_window > 0:
        if hasattr(config, "sliding_window"):
            config.sliding_window = sliding_window
            logger.info(f"★ SWA 已启用: sliding_window={sliding_window}")
        if hasattr(config, "max_window_layers"):
            config.max_window_layers = config.num_hidden_layers
            logger.info(f"  max_window_layers={config.max_window_layers}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    logger.info(f"backbone 加载完成: hidden_dim={hidden_dim}, num_layers={num_layers}")
    return model, tokenizer, hidden_dim, num_layers


def load_mac_weights(
    weights_dir: str, hidden_dim: int, device: str, dtype_str: str,
) -> tuple[ContextSelector, PrefixProjector, dict]:
    """加载训练好的 MAC 权重。"""
    weights_path = Path(weights_dir)

    config_path = weights_path / "mac_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 MAC 配置文件: {config_path}")

    with open(config_path, "r") as f:
        mac_config = json.load(f)

    logger.info(f"MAC 配置: {json.dumps(mac_config, indent=2, ensure_ascii=False)}")

    # ContextSelector
    sel_cfg = ContextSelectorConfig(
        input_dim=mac_config.get("hidden_dim", hidden_dim),
        hidden_dim=mac_config.get("selector_hidden_dim", 256),
        top_k=mac_config.get("selector_top_k", 5),
    )
    selector = ContextSelector(sel_cfg).to(device)
    sel_ckpt = weights_path / "context_selector.pt"
    if sel_ckpt.exists():
        selector.load_state_dict(torch.load(sel_ckpt, map_location=device))
        logger.info(f"Selector 权重已加载: {sel_ckpt} ({sel_ckpt.stat().st_size / 1024:.1f} KB)")

    # PrefixProjector
    proj_cfg = PrefixProjectorConfig(
        hidden_dim=mac_config.get("hidden_dim", hidden_dim),
        tokens_per_memory=mac_config.get("tokens_per_memory", 4),
        num_mlp_layers=mac_config.get("projector_layers", 2),
        mlp_expansion=mac_config.get("projector_expansion", 1.5),
        use_gating=mac_config.get("use_gating", True),
        init_scale=mac_config.get("projector_init_scale", 0.01),
    )
    projector = PrefixProjector(proj_cfg).to(device)
    proj_ckpt = weights_path / "prefix_projector.pt"
    if proj_ckpt.exists():
        projector.load_state_dict(torch.load(proj_ckpt, map_location=device))
        logger.info(f"PrefixProjector 权重已加载: {proj_ckpt} ({proj_ckpt.stat().st_size / 1024:.1f} KB)")

    return selector, projector, mac_config


# ======================================================================
# 数据加载
# ======================================================================

def load_eval_data(data_path: str, max_samples: int = 100) -> list[dict]:
    """加载评估数据 (JSONL 格式)。"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input_text" in obj and "memory_texts" in obj:
                data.append(obj)
                if len(data) >= max_samples:
                    break

    logger.info(f"加载了 {len(data)} 条评估样本")
    return data


# ======================================================================
# 评估 1: Selector 准确率
# ======================================================================

def eval_selector(
    selector: ContextSelector,
    encoder: MemoryEncoder,
    eval_data: list[dict],
    device: str,
) -> dict:
    """评估 ContextSelector 的选择准确率。"""
    logger.info("=" * 60)
    logger.info("评估 1: ContextSelector 准确率")
    logger.info("=" * 60)

    selector.eval()
    top_k = selector.config.top_k

    hit_count = 0
    precision_sum = 0.0
    ndcg_sum = 0.0
    relevant_score_sum = 0.0
    irrelevant_score_sum = 0.0
    relevant_count = 0
    irrelevant_count = 0

    valid_samples = 0
    for sample in eval_data:
        if "relevant_indices" not in sample or not sample["relevant_indices"]:
            continue

        with torch.no_grad():
            query_emb = encoder.encode_texts_deep([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"]).unsqueeze(0)
            selected_indices, selected_scores = selector.select(query_emb, memory_embs, top_k=top_k)

        relevant_set = set(sample["relevant_indices"])
        selected_set = set(selected_indices[0].cpu().tolist())

        # Hit@K
        if relevant_set & selected_set:
            hit_count += 1

        # Precision@K
        tp = len(relevant_set & selected_set)
        precision_sum += tp / top_k

        # NDCG@K
        dcg = 0.0
        for rank, idx in enumerate(selected_indices[0].cpu().tolist()):
            if idx in relevant_set:
                dcg += 1.0 / math.log2(rank + 2)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), top_k)))
        ndcg_sum += dcg / max(ideal_dcg, 1e-9)

        # 相关/不相关记忆的平均分
        all_scores = torch.sigmoid(selector(query_emb, memory_embs))[0]
        for i, score in enumerate(all_scores.cpu().tolist()):
            if i in relevant_set:
                relevant_score_sum += score
                relevant_count += 1
            else:
                irrelevant_score_sum += score
                irrelevant_count += 1

        valid_samples += 1

    n = max(valid_samples, 1)
    hit_at_k = hit_count / n
    precision_at_k = precision_sum / n
    ndcg_at_k = ndcg_sum / n
    avg_relevant = relevant_score_sum / max(relevant_count, 1)
    avg_irrelevant = irrelevant_score_sum / max(irrelevant_count, 1)
    gap = avg_relevant - avg_irrelevant

    logger.info(f"  样本数: {valid_samples}")
    logger.info(f"  Hit@{top_k}:       {hit_at_k:.4f}")
    logger.info(f"  Precision@{top_k}: {precision_at_k:.4f}")
    logger.info(f"  NDCG@{top_k}:      {ndcg_at_k:.4f}")
    logger.info(f"  相关记忆平均分:    {avg_relevant:.4f}")
    logger.info(f"  不相关记忆平均分:  {avg_irrelevant:.4f}")
    logger.info(f"  区分度 (gap):      {gap:.4f}")

    if gap > 0.1:
        logger.info(f"  ✅ Selector 区分度良好, 能有效区分相关/不相关记忆")
    else:
        logger.info(f"  ⚠️ Selector 区分度不足, 可能需要更多训练")

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "ndcg_at_k": ndcg_at_k,
        "avg_relevant_score": avg_relevant,
        "avg_irrelevant_score": avg_irrelevant,
        "gap": gap,
    }


# ======================================================================
# 评估 2: PPL 对比 (有 prefix vs 无 prefix)
# ======================================================================

def eval_ppl(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    projector: PrefixProjector,
    eval_data: list[dict],
    args: argparse.Namespace,
) -> dict:
    """评估有 prefix 和无 prefix 时的 PPL 差异。"""
    logger.info("=" * 60)
    logger.info("评估 2: PPL 对比 (有记忆 vs 无记忆)")
    logger.info("=" * 60)

    model.eval()
    selector.eval()
    projector.eval()

    total_loss_with = 0.0
    total_loss_without = 0.0
    total_tokens = 0
    valid_samples = 0

    target_samples = [s for s in eval_data if s.get("target_text")]
    logger.info(f"  有 target_text 的样本: {len(target_samples)}")

    for sample in target_samples:
        with torch.no_grad():
            # 编码
            query_emb = encoder.encode_texts_deep([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"]).unsqueeze(0)

            # Selector 打分
            selection_weights = selector.soft_select(query_emb, memory_embs)

            # PrefixProjector 生成 prefix tokens
            prefix_tokens = projector(memory_embs, selection_weights)

            # Tokenize
            query_enc = tokenizer(
                sample["input_text"], add_special_tokens=True,
                truncation=True, max_length=args.max_seq_len // 2,
            )
            target_enc = tokenizer(
                sample["target_text"], add_special_tokens=False,
                truncation=True, max_length=args.max_seq_len // 2,
            )
            query_ids = query_enc["input_ids"]
            target_ids = target_enc["input_ids"]
            combined_ids = query_ids + target_ids
            if len(combined_ids) > args.max_seq_len:
                combined_ids = combined_ids[:args.max_seq_len]

            input_ids = torch.tensor([combined_ids], device=args.device)
            query_len = len(query_ids)

            # 获取 prompt embedding
            if hasattr(model.model, "embed_tokens"):
                prompt_embeds = model.model.embed_tokens(input_ids)
            else:
                prompt_embeds = model.get_input_embeddings()(input_ids)

            prefix_len = prefix_tokens.shape[1]

            # ---- 有记忆 PPL ---- #
            full_embeds = torch.cat([prefix_tokens, prompt_embeds], dim=1)
            total_len = full_embeds.shape[1]
            labels_with = torch.full((1, total_len), -100, dtype=torch.long, device=args.device)
            target_start = prefix_len + query_len
            target_end = total_len
            labels_with[0, target_start:target_end] = input_ids[0, query_len:query_len + (target_end - target_start)]

            out_with = model(inputs_embeds=full_embeds, labels=None)
            shift_logits_w = out_with.logits[..., :-1, :].contiguous()
            shift_labels_w = labels_with[..., 1:].contiguous()
            loss_with = F.cross_entropy(
                shift_logits_w.view(-1, shift_logits_w.size(-1)),
                shift_labels_w.view(-1), ignore_index=-100,
            )

            # ---- 无记忆 PPL ---- #
            labels_without = torch.full((1, prompt_embeds.shape[1]), -100, dtype=torch.long, device=args.device)
            labels_without[0, query_len:] = input_ids[0, query_len:]

            out_without = model(inputs_embeds=prompt_embeds, labels=None)
            shift_logits_wo = out_without.logits[..., :-1, :].contiguous()
            shift_labels_wo = labels_without[..., 1:].contiguous()
            loss_without = F.cross_entropy(
                shift_logits_wo.view(-1, shift_logits_wo.size(-1)),
                shift_labels_wo.view(-1), ignore_index=-100,
            )

            # 有效 token 数
            n_tokens = (shift_labels_w.view(-1) != -100).sum().item()
            if n_tokens == 0:
                continue

            total_loss_with += loss_with.item() * n_tokens
            total_loss_without += loss_without.item() * n_tokens
            total_tokens += n_tokens
            valid_samples += 1

    if total_tokens == 0:
        logger.warning("  无有效样本")
        return {}

    avg_loss_with = total_loss_with / total_tokens
    avg_loss_without = total_loss_without / total_tokens
    ppl_with = math.exp(avg_loss_with)
    ppl_without = math.exp(avg_loss_without)
    ppl_drop = ppl_without - ppl_with
    ppl_drop_pct = ppl_drop / ppl_without * 100

    logger.info(f"  评估样本数: {valid_samples}, 总 token: {total_tokens}")
    logger.info(f"  PPL (有记忆):   {ppl_with:.4f}  (avg_loss={avg_loss_with:.4f})")
    logger.info(f"  PPL (无记忆):   {ppl_without:.4f}  (avg_loss={avg_loss_without:.4f})")
    logger.info(f"  PPL 降低:       {ppl_drop:.4f} ({ppl_drop_pct:.2f}%)")

    if ppl_drop > 0.5:
        logger.info(f"  ✅ 有记忆注入时 PPL 显著降低, MAC 有效!")
    elif ppl_drop > 0:
        logger.info(f"  🟡 PPL 有所降低, 但幅度不大")
    else:
        logger.info(f"  ❌ PPL 未降低, MAC 可能未学到有效注入")

    return {
        "ppl_with_memory": ppl_with,
        "ppl_without_memory": ppl_without,
        "ppl_drop": ppl_drop,
        "ppl_drop_pct": ppl_drop_pct,
        "avg_loss_with": avg_loss_with,
        "avg_loss_without": avg_loss_without,
        "num_samples": valid_samples,
        "total_tokens": total_tokens,
    }


# ======================================================================
# 评估 3: 生成质量对比
# ======================================================================

def eval_generation(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    projector: PrefixProjector,
    eval_data: list[dict],
    args: argparse.Namespace,
) -> dict:
    """对比有/无 prefix 的生成结果。"""
    logger.info("=" * 60)
    logger.info("评估 3: 生成质量对比")
    logger.info("=" * 60)

    model.eval()
    selector.eval()
    projector.eval()

    num_samples = min(args.num_generate_samples, len(eval_data))
    diff_count = 0
    overlap_with = []
    overlap_without = []

    for i in range(num_samples):
        sample = eval_data[i]
        logger.info(f"\n--- 样本 {i+1}/{num_samples} ---")
        logger.info(f"  Query: {sample['input_text'][:150]}...")
        if sample.get("target_text"):
            logger.info(f"  Ground Truth: {sample['target_text'][:150]}...")

        logger.info(f"  记忆数: {len(sample['memory_texts'])}")
        if sample.get("relevant_indices"):
            logger.info(f"  相关记忆索引: {sample['relevant_indices']}")

        # 打印前 3 条记忆
        for j, mem in enumerate(sample["memory_texts"][:3]):
            is_relevant = j in sample.get("relevant_indices", [])
            marker = "✓" if is_relevant else "✗"
            logger.info(f"  记忆[{j}] ({marker}): {mem[:100]}...")

        with torch.no_grad():
            # 编码
            query_emb = encoder.encode_texts_deep([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"]).unsqueeze(0)

            # Selector 打分 + PrefixProjector
            selection_weights = selector.soft_select(query_emb, memory_embs)
            prefix_tokens = projector(memory_embs, selection_weights)

            # Tokenize query
            input_ids = tokenizer(
                sample["input_text"], return_tensors="pt",
                truncation=True, max_length=args.max_seq_len,
            )["input_ids"].to(args.device)

            # ---- 生成 (无记忆) ---- #
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 1.0,
            }
            out_no_mem = model.generate(input_ids=input_ids, **gen_kwargs)
            text_no_mem = tokenizer.decode(out_no_mem[0][input_ids.shape[1]:], skip_special_tokens=True)

            # ---- 生成 (有记忆) ---- #
            # 拼接 prefix + prompt embedding
            if hasattr(model.model, "embed_tokens"):
                prompt_embeds = model.model.embed_tokens(input_ids)
            else:
                prompt_embeds = model.get_input_embeddings()(input_ids)

            full_embeds = torch.cat([prefix_tokens, prompt_embeds], dim=1)
            full_attn = torch.ones((1, full_embeds.shape[1]), dtype=torch.long, device=args.device)

            out_with_mem = model.generate(
                inputs_embeds=full_embeds, attention_mask=full_attn, **gen_kwargs
            )
            # generate 输出包含 full_embeds 对应的 token, 截取新生成的部分
            text_with_mem = tokenizer.decode(out_with_mem[0][full_embeds.shape[1]:], skip_special_tokens=True)

        logger.info(f"  生成 (无记忆):  {text_no_mem[:200]}...")
        logger.info(f"  生成 (有记忆): {text_with_mem[:200]}...")

        # 词重叠率
        gt = sample.get("target_text", "")
        if gt:
            gt_words = set(gt.split())
            no_mem_words = set(text_no_mem.split())
            with_mem_words = set(text_with_mem.split())

            ov_no = len(gt_words & no_mem_words) / max(len(gt_words), 1)
            ov_with = len(gt_words & with_mem_words) / max(len(gt_words), 1)
            overlap_without.append(ov_no)
            overlap_with.append(ov_with)
            logger.info(f"  词重叠 (无记忆 vs GT): {ov_no:.4f}")
            logger.info(f"  词重叠 (有记忆 vs GT): {ov_with:.4f}")

        # 差异判断
        is_diff = text_no_mem.strip() != text_with_mem.strip()
        diff_count += int(is_diff)
        logger.info(f"  两种生成是否不同: {'是 ✅' if is_diff else '否 ❌'}")

    logger.info(f"\n  生成结果汇总:")
    logger.info(f"  两种生成有差异的比例: {diff_count / max(num_samples, 1) * 100:.1f}%")
    if overlap_without:
        logger.info(f"  平均词重叠 (无记忆): {np.mean(overlap_without):.4f}")
        logger.info(f"  平均词重叠 (有记忆): {np.mean(overlap_with):.4f}")

    return {
        "diff_ratio": diff_count / max(num_samples, 1),
        "avg_overlap_without": float(np.mean(overlap_without)) if overlap_without else 0,
        "avg_overlap_with": float(np.mean(overlap_with)) if overlap_with else 0,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    t_start = time.time()

    # 加载 backbone
    model, tokenizer, hidden_dim, num_layers = load_backbone(
        args.model_path, args.device, args.dtype,
        sliding_window=args.sliding_window,
    )

    # 加载 MAC 权重
    selector, projector, mac_config = load_mac_weights(
        args.mac_weights_dir, hidden_dim, args.device, args.dtype,
    )

    # 初始化 MemoryEncoder
    enc_cfg = MemoryEncoderConfig(
        max_memory_tokens=mac_config.get("max_memory_tokens", 64),
        pooling="mean",
        deep_encode_layers=args.deep_encode_layers,
    )
    encoder = MemoryEncoder(enc_cfg)
    encoder.set_backbone(
        backbone_model=model, tokenizer=tokenizer,
        hidden_dim=hidden_dim, device=args.device,
        dtype=getattr(torch, args.dtype, torch.float32),
    )

    # 加载评估数据
    eval_data = load_eval_data(args.data_path, args.num_eval_samples)
    if not eval_data:
        logger.error("评估数据为空")
        return

    results = {}

    # 评估 1: Selector
    results["selector"] = eval_selector(selector, encoder, eval_data, args.device)

    # 评估 2: PPL
    results["ppl"] = eval_ppl(model, tokenizer, encoder, selector, projector, eval_data, args)

    # 评估 3: 生成质量
    results["generation"] = eval_generation(
        model, tokenizer, encoder, selector, projector, eval_data, args
    )

    total_time = time.time() - t_start

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("评估汇总报告")
    logger.info("=" * 60)
    logger.info(f"  Selector Hit@{selector.config.top_k}: {results['selector'].get('hit_at_k', 0):.4f}")
    logger.info(f"  Selector 区分度:    {results['selector'].get('gap', 0):.4f}")
    if results["ppl"]:
        logger.info(f"  PPL (有记忆):       {results['ppl'].get('ppl_with_memory', 0):.4f}")
        logger.info(f"  PPL (无记忆):       {results['ppl'].get('ppl_without_memory', 0):.4f}")
        logger.info(f"  PPL 降低:           {results['ppl'].get('ppl_drop_pct', 0):.2f}%")
    logger.info(f"  生成差异比例:       {results['generation'].get('diff_ratio', 0) * 100:.1f}%")
    logger.info(f"  总耗时: {total_time:.1f}s")

    # 综合判断
    logger.info("")
    logger.info("=" * 60)
    logger.info("综合判断")
    logger.info("=" * 60)

    positives = []
    warnings = []

    if results["selector"].get("gap", 0) > 0.1:
        positives.append(f"Selector 区分度良好 (gap={results['selector']['gap']:.4f})")
    else:
        warnings.append(f"Selector 区分度不足 (gap={results['selector'].get('gap', 0):.4f})")

    if results["ppl"] and results["ppl"].get("ppl_drop", 0) > 0.5:
        positives.append(f"PPL 显著降低 ({results['ppl']['ppl_drop']:.4f})")
    elif results["ppl"] and results["ppl"].get("ppl_drop", 0) > 0:
        positives.append(f"PPL 有所降低 ({results['ppl'].get('ppl_drop', 0):.4f})")
    else:
        warnings.append("PPL 未改善")

    if results["generation"].get("diff_ratio", 0) > 0.5:
        positives.append(f"生成有差异 ({results['generation']['diff_ratio']*100:.0f}%)")

    if positives:
        logger.info(f"  ✅ 积极信号:")
        for p in positives:
            logger.info(f"    - {p}")

    if warnings:
        logger.info(f"  ⚠️ 需关注:")
        for w in warnings:
            logger.info(f"    - {w}")

    if len(positives) >= 2 and not warnings:
        logger.info(f"\n  MAC 训练效果良好! 可以集成到 MemoryAgent 使用。")
    elif positives:
        logger.info(f"\n  MAC 有一定效果, 但仍有改进空间。")
    else:
        logger.info(f"\n  MAC 效果不理想, 建议调整训练参数。")

    # 保存结果
    if args.output_file:
        results["total_time_s"] = total_time
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n评估结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
