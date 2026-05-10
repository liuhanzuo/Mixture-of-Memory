#!/usr/bin/env python3
"""
benchmark_mac.py — MAC 标准 Benchmark 评测脚本。

在 **测试集** 上对比三种方案:
  1. No Memory:  backbone 直接生成 (无任何记忆)
  2. RAG Concat: 将 top-k 记忆文本拼接到 prompt 前 (传统 RAG 方式)
  3. MAC Prefix: ContextSelector + PrefixProjector 注入 soft prefix tokens

评测维度:
  ① PPL (困惑度) — 模型对 target 的预测能力
  ② 字符级 F1 — 生成文本与 ground truth 的字符重叠
  ③ 记忆召回率 — Selector 是否选中了相关记忆
  ④ 按 dialogue / session 分组统计 — 观察不同对话深度的效果变化

Usage:
    python scripts/benchmark_mac.py \\
        --model_path ../models/Qwen--Qwen3-8b \\
        --mac_weights_dir outputs/mac_train_full_20260324_202042 \\
        --test_data data/raw/long_dialogue_test.jsonl \\
        --output_file outputs/metrics/benchmark_mac_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 项目根目录
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# 延迟导入
MemoryEncoder = None
MemoryEncoderConfig = None
ContextSelector = None
ContextSelectorConfig = None
PrefixProjector = None
PrefixProjectorConfig = None


def _lazy_import():
    global MemoryEncoder, MemoryEncoderConfig
    global ContextSelector, ContextSelectorConfig
    global PrefixProjector, PrefixProjectorConfig
    if MemoryEncoder is None:
        from src.memory.mag.memory_encoder import MemoryEncoder as _ME, MemoryEncoderConfig as _MEC
        MemoryEncoder = _ME
        MemoryEncoderConfig = _MEC
    if ContextSelector is None:
        from src.memory.mag.context_selector import ContextSelector as _CS, ContextSelectorConfig as _CSC
        ContextSelector = _CS
        ContextSelectorConfig = _CSC
    if PrefixProjector is None:
        from src.memory.mag.prefix_projector import PrefixProjector as _PP, PrefixProjectorConfig as _PPC
        PrefixProjector = _PP
        PrefixProjectorConfig = _PPC


# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark_mac")


# ============================================================
# 参数
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="MAC Benchmark Evaluation")
    p.add_argument("--model_path", type=str, required=True, help="Backbone 模型路径")
    p.add_argument("--mac_weights_dir", type=str, required=True, help="MAC 训练权重目录")
    p.add_argument("--test_data", type=str, required=True, help="测试数据 (.jsonl)")
    p.add_argument("--output_file", type=str, default="outputs/metrics/benchmark_mac_result.json")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--deep_encode_layers", type=int, default=8)
    p.add_argument("--sliding_window", type=int, default=0)
    p.add_argument("--num_generate_samples", type=int, default=20, help="生成对比样本数")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================
# 字符级 F1
# ============================================================
def char_f1(prediction: str, ground_truth: str) -> dict:
    """计算中文友好的字符级 F1。"""
    pred_chars = list(prediction.replace(" ", ""))
    gt_chars = list(ground_truth.replace(" ", ""))

    if not pred_chars or not gt_chars:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    from collections import Counter
    pred_counter = Counter(pred_chars)
    gt_counter = Counter(gt_chars)

    common = sum((pred_counter & gt_counter).values())
    precision = common / len(pred_chars) if pred_chars else 0
    recall = common / len(gt_chars) if gt_chars else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def char_overlap(prediction: str, ground_truth: str) -> float:
    """字符集合重叠率 (recall)。"""
    pred_chars = set(prediction.replace(" ", ""))
    gt_chars = set(ground_truth.replace(" ", ""))
    if not gt_chars:
        return 0.0
    return len(gt_chars & pred_chars) / len(gt_chars)


# ============================================================
# 加载模型
# ============================================================
def load_backbone(model_path: str, device: str, dtype_str: str, sliding_window: int):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[dtype_str]

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if sliding_window > 0 and hasattr(config, "sliding_window"):
        config.sliding_window = sliding_window
        logger.info(f"★ SWA 已启用: sliding_window={sliding_window}")

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
    logger.info(f"Backbone 加载完成: hidden_dim={hidden_dim}, num_layers={num_layers}")
    return model, tokenizer, hidden_dim, num_layers


def load_mac_weights(weights_dir: str, hidden_dim: int, device: str):
    _lazy_import()
    weights_path = Path(weights_dir)
    with open(weights_path / "mac_config.json") as f:
        mac_config = json.load(f)

    sel_cfg = ContextSelectorConfig(
        input_dim=mac_config.get("hidden_dim", hidden_dim),
        hidden_dim=mac_config.get("selector_hidden_dim", 256),
        top_k=mac_config.get("selector_top_k", 5),
    )
    selector = ContextSelector(sel_cfg).to(device)
    sel_ckpt = weights_path / "context_selector.pt"
    if sel_ckpt.exists():
        selector.load_state_dict(torch.load(sel_ckpt, map_location=device))

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

    return selector, projector, mac_config


# ============================================================
# 核心评估: PPL 三方对比
# ============================================================
def eval_ppl_comparison(
    model, tokenizer, encoder, selector, projector, test_data, args
) -> dict:
    """PPL 对比: No Memory vs RAG Concat vs MAC Prefix。"""
    logger.info("=" * 60)
    logger.info("评估 1: PPL 三方对比 (No Memory / RAG Concat / MAC Prefix)")
    logger.info("=" * 60)

    model.eval()
    selector.eval()
    projector.eval()

    results_by_dialogue = defaultdict(lambda: defaultdict(list))

    total_loss = {"no_mem": 0.0, "rag": 0.0, "mac": 0.0}
    total_tokens = 0
    valid_samples = 0

    for i, sample in enumerate(test_data):
        if not sample.get("target_text"):
            continue

        memory_texts = sample["memory_texts"]
        relevant_indices = set(sample.get("relevant_indices", []))
        dialogue_id = sample.get("dialogue_id", "unknown")
        session_idx = sample.get("session_idx", 0)

        with torch.no_grad():
            # ---- 编码记忆 & Selector ---- #
            memory_embs = encoder.encode_texts_deep(memory_texts)
            query_emb = encoder.encode_texts_deep([sample["input_text"]])[0]
            top_indices, top_scores = selector.select(
                query_emb.unsqueeze(0), memory_embs.unsqueeze(0)
            )
            selection_weights = selector.soft_select(
                query_emb.unsqueeze(0), memory_embs.unsqueeze(0)
            )
            prefix_tokens = projector(memory_embs.unsqueeze(0), selection_weights)

            # ---- Tokenize ---- #
            query_enc = tokenizer(
                sample["input_text"], add_special_tokens=False,
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

            if hasattr(model.model, "embed_tokens"):
                prompt_embeds = model.model.embed_tokens(input_ids)
            else:
                prompt_embeds = model.get_input_embeddings()(input_ids)

            # ---- RAG: 拼接 top-k 记忆文本到 prompt 前 ---- #
            topk_idx = top_indices[0].cpu().tolist()
            rag_context = "\n".join([memory_texts[idx] for idx in topk_idx])
            rag_text = rag_context + "\n" + sample["input_text"]
            rag_enc = tokenizer(
                rag_text, add_special_tokens=False,
                truncation=True, max_length=args.max_seq_len // 2,
            )
            rag_combined = rag_enc["input_ids"] + target_ids
            if len(rag_combined) > args.max_seq_len:
                rag_combined = rag_combined[:args.max_seq_len]
            rag_input_ids = torch.tensor([rag_combined], device=args.device)
            rag_query_len = len(rag_enc["input_ids"])

            # ---- 计算三种 PPL ---- #
            def compute_loss(embeds, labels_ids, prefix_len):
                total_len = embeds.shape[1]
                labels = torch.full((1, total_len), -100, dtype=torch.long, device=args.device)
                target_start = prefix_len
                target_end = total_len
                n_target = target_end - target_start
                labels[0, target_start:target_end] = labels_ids[0, -n_target:]

                out = model(inputs_embeds=embeds, labels=None)
                shift_logits = out.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), ignore_index=-100,
                )
                n_tokens = (shift_labels.view(-1) != -100).sum().item()
                return loss.item(), n_tokens

            # (a) No Memory
            loss_no, n_no = compute_loss(prompt_embeds, input_ids, query_len)

            # (b) RAG Concat
            if hasattr(model.model, "embed_tokens"):
                rag_embeds = model.model.embed_tokens(rag_input_ids)
            else:
                rag_embeds = model.get_input_embeddings()(rag_input_ids)
            loss_rag, n_rag = compute_loss(rag_embeds, rag_input_ids, rag_query_len)

            # (c) MAC Prefix
            prefix_tokens_cast = prefix_tokens.to(dtype=prompt_embeds.dtype)
            mac_embeds = torch.cat([prefix_tokens_cast, prompt_embeds], dim=1)
            prefix_len = prefix_tokens_cast.shape[1]
            loss_mac, n_mac = compute_loss(mac_embeds, input_ids, prefix_len + query_len)

        if n_no == 0:
            continue

        total_loss["no_mem"] += loss_no * n_no
        total_loss["rag"] += loss_rag * n_rag
        total_loss["mac"] += loss_mac * n_mac
        total_tokens += n_no
        valid_samples += 1

        # 按 dialogue 分组
        results_by_dialogue[dialogue_id][session_idx].append({
            "loss_no_mem": loss_no,
            "loss_rag": loss_rag,
            "loss_mac": loss_mac,
            "n_tokens": n_no,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  进度: {i+1}/{len(test_data)}")

    if total_tokens == 0:
        logger.warning("无有效样本")
        return {}

    # 汇总
    ppl = {}
    for key in ["no_mem", "rag", "mac"]:
        avg_loss = total_loss[key] / total_tokens
        ppl[key] = {"avg_loss": avg_loss, "ppl": math.exp(min(avg_loss, 20))}

    # 按 dialogue 汇总
    dialogue_summary = {}
    for did, sessions in results_by_dialogue.items():
        d_ppl = {"no_mem": 0, "rag": 0, "mac": 0}
        d_tokens = 0
        for sid, items in sessions.items():
            for item in items:
                n = item["n_tokens"]
                d_ppl["no_mem"] += item["loss_no_mem"] * n
                d_ppl["rag"] += item["loss_rag"] * n
                d_ppl["mac"] += item["loss_mac"] * n
                d_tokens += n
        if d_tokens > 0:
            dialogue_summary[did] = {
                "ppl_no_mem": math.exp(min(d_ppl["no_mem"] / d_tokens, 20)),
                "ppl_rag": math.exp(min(d_ppl["rag"] / d_tokens, 20)),
                "ppl_mac": math.exp(min(d_ppl["mac"] / d_tokens, 20)),
                "n_tokens": d_tokens,
            }

    logger.info(f"  有效样本: {valid_samples}, 总 tokens: {total_tokens}")
    logger.info(f"  PPL (No Memory):   {ppl['no_mem']['ppl']:.4f} (loss={ppl['no_mem']['avg_loss']:.4f})")
    logger.info(f"  PPL (RAG Concat):  {ppl['rag']['ppl']:.4f} (loss={ppl['rag']['avg_loss']:.4f})")
    logger.info(f"  PPL (MAC Prefix):  {ppl['mac']['ppl']:.4f} (loss={ppl['mac']['avg_loss']:.4f})")

    mac_drop = (ppl["no_mem"]["ppl"] - ppl["mac"]["ppl"]) / ppl["no_mem"]["ppl"] * 100
    rag_drop = (ppl["no_mem"]["ppl"] - ppl["rag"]["ppl"]) / ppl["no_mem"]["ppl"] * 100
    logger.info(f"  MAC PPL 降低:  {mac_drop:.2f}%")
    logger.info(f"  RAG PPL 降低:  {rag_drop:.2f}%")

    if mac_drop > rag_drop:
        logger.info(f"  ✅ MAC 优于 RAG! (MAC 多降 {mac_drop - rag_drop:.2f}%)")
    else:
        logger.info(f"  ⚠️ RAG 优于 MAC (差 {rag_drop - mac_drop:.2f}%)")

    return {
        "ppl_no_mem": ppl["no_mem"]["ppl"],
        "ppl_rag": ppl["rag"]["ppl"],
        "ppl_mac": ppl["mac"]["ppl"],
        "mac_ppl_drop_pct": mac_drop,
        "rag_ppl_drop_pct": rag_drop,
        "num_samples": valid_samples,
        "total_tokens": total_tokens,
        "dialogue_breakdown": dialogue_summary,
    }


# ============================================================
# 核心评估: Selector 准确率 (测试集)
# ============================================================
def eval_selector(model, tokenizer, encoder, selector, test_data, args) -> dict:
    """在测试集上评估 Selector 准确率。"""
    logger.info("=" * 60)
    logger.info("评估 2: Selector 准确率 (测试集)")
    logger.info("=" * 60)

    selector.eval()
    hit_at_k = []
    precision_at_k = []
    ndcg_at_k = []
    gaps = []

    for sample in test_data:
        memory_texts = sample["memory_texts"]
        relevant_indices = set(sample.get("relevant_indices", []))
        if not relevant_indices:
            continue

        with torch.no_grad():
            memory_embs = encoder.encode_texts_deep(memory_texts)
            query_emb = encoder.encode_texts_deep([sample["input_text"]])[0]
            top_indices, top_scores = selector.select(
                query_emb.unsqueeze(0), memory_embs.unsqueeze(0)
            )
            scores = selector(
                query_emb.unsqueeze(0), memory_embs.unsqueeze(0)
            )  # (B, K) raw scores for gap computation

        top_k_set = set(top_indices[0].cpu().tolist())
        k = len(top_k_set)

        # Hit@K
        hit = 1.0 if top_k_set & relevant_indices else 0.0
        hit_at_k.append(hit)

        # Precision@K
        prec = len(top_k_set & relevant_indices) / k if k > 0 else 0
        precision_at_k.append(prec)

        # NDCG@K
        dcg = sum(1.0 / math.log2(rank + 2)
                   for rank, idx in enumerate(top_indices[0].cpu().tolist())
                   if idx in relevant_indices)
        ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(relevant_indices), k)))
        ndcg = dcg / ideal if ideal > 0 else 0
        ndcg_at_k.append(ndcg)

        # 区分度
        all_scores = torch.sigmoid(scores[0]).cpu().numpy()
        rel_scores = [all_scores[i] for i in range(len(all_scores)) if i in relevant_indices]
        irrel_scores = [all_scores[i] for i in range(len(all_scores)) if i not in relevant_indices]
        if rel_scores and irrel_scores:
            gaps.append(np.mean(rel_scores) - np.mean(irrel_scores))

    results = {
        "num_samples": len(hit_at_k),
        "hit_at_k": float(np.mean(hit_at_k)),
        "precision_at_k": float(np.mean(precision_at_k)),
        "ndcg_at_k": float(np.mean(ndcg_at_k)),
        "score_gap": float(np.mean(gaps)) if gaps else 0,
    }

    logger.info(f"  样本数: {results['num_samples']}")
    logger.info(f"  Hit@K:       {results['hit_at_k']:.4f}")
    logger.info(f"  Precision@K: {results['precision_at_k']:.4f}")
    logger.info(f"  NDCG@K:      {results['ndcg_at_k']:.4f}")
    logger.info(f"  区分度:       {results['score_gap']:.4f}")

    return results


# ============================================================
# 核心评估: 生成质量三方对比
# ============================================================
def eval_generation(
    model, tokenizer, encoder, selector, projector, test_data, args
) -> dict:
    """生成质量三方对比: No Memory / RAG Concat / MAC Prefix。"""
    logger.info("=" * 60)
    logger.info("评估 3: 生成质量对比 (No Memory / RAG Concat / MAC Prefix)")
    logger.info("=" * 60)

    model.eval()
    selector.eval()
    projector.eval()

    num_samples = min(args.num_generate_samples, len(test_data))
    # 从不同 dialogue 中均匀采样
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(test_data))[:num_samples].tolist()

    metrics = {"no_mem": [], "rag": [], "mac": []}
    examples = []

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.2,
    }

    for idx_i, sample_idx in enumerate(indices):
        sample = test_data[sample_idx]
        gt = sample.get("target_text", "")
        if not gt:
            continue

        memory_texts = sample["memory_texts"]
        dialogue_id = sample.get("dialogue_id", "unknown")
        session_idx = sample.get("session_idx", 0)

        logger.info(f"\n--- 样本 {idx_i+1}/{num_samples} [{dialogue_id} S{session_idx}] ---")
        logger.info(f"  Query: {sample['input_text'][:100]}...")
        logger.info(f"  GT:    {gt[:100]}...")

        with torch.no_grad():
            memory_embs = encoder.encode_texts_deep(memory_texts)
            query_emb = encoder.encode_texts_deep([sample["input_text"]])[0]
            top_indices, top_scores = selector.select(
                query_emb.unsqueeze(0), memory_embs.unsqueeze(0)
            )
            selection_weights = selector.soft_select(
                query_emb.unsqueeze(0), memory_embs.unsqueeze(0)
            )
            prefix_tokens = projector(memory_embs.unsqueeze(0), selection_weights)

            input_ids = tokenizer(
                sample["input_text"], return_tensors="pt",
                truncation=True, max_length=args.max_seq_len,
            )["input_ids"].to(args.device)

            # (a) No Memory
            out_no = model.generate(input_ids=input_ids, **gen_kwargs)
            text_no = tokenizer.decode(out_no[0][input_ids.shape[1]:], skip_special_tokens=True)

            # (b) RAG Concat
            topk_idx = top_indices[0].cpu().tolist()
            rag_context = "\n".join([memory_texts[i] for i in topk_idx])
            rag_text = rag_context + "\n" + sample["input_text"]
            rag_ids = tokenizer(
                rag_text, return_tensors="pt",
                truncation=True, max_length=args.max_seq_len,
            )["input_ids"].to(args.device)
            out_rag = model.generate(input_ids=rag_ids, **gen_kwargs)
            text_rag = tokenizer.decode(out_rag[0][rag_ids.shape[1]:], skip_special_tokens=True)

            # (c) MAC Prefix
            if hasattr(model.model, "embed_tokens"):
                prompt_embeds = model.model.embed_tokens(input_ids)
            else:
                prompt_embeds = model.get_input_embeddings()(input_ids)
            prefix_tokens_cast = prefix_tokens.to(dtype=prompt_embeds.dtype)
            mac_embeds = torch.cat([prefix_tokens_cast, prompt_embeds], dim=1)
            mac_attn = torch.ones((1, mac_embeds.shape[1]), dtype=torch.long, device=args.device)
            out_mac = model.generate(
                inputs_embeds=mac_embeds, attention_mask=mac_attn, **gen_kwargs
            )
            gen_ids = out_mac[0]
            if gen_ids.shape[0] > mac_embeds.shape[1]:
                gen_ids = gen_ids[mac_embeds.shape[1]:]
            text_mac = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 计算指标
        f1_no = char_f1(text_no, gt)
        f1_rag = char_f1(text_rag, gt)
        f1_mac = char_f1(text_mac, gt)

        ov_no = char_overlap(text_no, gt)
        ov_rag = char_overlap(text_rag, gt)
        ov_mac = char_overlap(text_mac, gt)

        metrics["no_mem"].append({"f1": f1_no["f1"], "overlap": ov_no})
        metrics["rag"].append({"f1": f1_rag["f1"], "overlap": ov_rag})
        metrics["mac"].append({"f1": f1_mac["f1"], "overlap": ov_mac})

        logger.info(f"  [No Mem]  F1={f1_no['f1']:.4f}  Overlap={ov_no:.4f}  | {text_no[:80]}...")
        logger.info(f"  [RAG]     F1={f1_rag['f1']:.4f}  Overlap={ov_rag:.4f}  | {text_rag[:80]}...")
        logger.info(f"  [MAC]     F1={f1_mac['f1']:.4f}  Overlap={ov_mac:.4f}  | {text_mac[:80]}...")

        examples.append({
            "dialogue_id": dialogue_id,
            "session_idx": session_idx,
            "query": sample["input_text"][:200],
            "gt": gt[:200],
            "text_no_mem": text_no[:200],
            "text_rag": text_rag[:200],
            "text_mac": text_mac[:200],
            "f1_no_mem": f1_no["f1"],
            "f1_rag": f1_rag["f1"],
            "f1_mac": f1_mac["f1"],
        })

    # 汇总
    summary = {}
    for key in ["no_mem", "rag", "mac"]:
        f1_vals = [m["f1"] for m in metrics[key]]
        ov_vals = [m["overlap"] for m in metrics[key]]
        summary[key] = {
            "avg_f1": float(np.mean(f1_vals)) if f1_vals else 0,
            "avg_overlap": float(np.mean(ov_vals)) if ov_vals else 0,
        }

    logger.info(f"\n  === 生成质量汇总 ({num_samples} 样本) ===")
    logger.info(f"  [No Memory]  avg_F1={summary['no_mem']['avg_f1']:.4f}  avg_Overlap={summary['no_mem']['avg_overlap']:.4f}")
    logger.info(f"  [RAG Concat] avg_F1={summary['rag']['avg_f1']:.4f}  avg_Overlap={summary['rag']['avg_overlap']:.4f}")
    logger.info(f"  [MAC Prefix] avg_F1={summary['mac']['avg_f1']:.4f}  avg_Overlap={summary['mac']['avg_overlap']:.4f}")

    # 判断胜负
    mac_wins_f1 = sum(1 for i in range(len(metrics["mac"])) if metrics["mac"][i]["f1"] > metrics["no_mem"][i]["f1"])
    rag_wins_f1 = sum(1 for i in range(len(metrics["rag"])) if metrics["rag"][i]["f1"] > metrics["no_mem"][i]["f1"])
    total = len(metrics["mac"])
    logger.info(f"  MAC 胜 No-Mem 比例 (F1): {mac_wins_f1}/{total} ({mac_wins_f1/total*100:.1f}%)")
    logger.info(f"  RAG 胜 No-Mem 比例 (F1): {rag_wins_f1}/{total} ({rag_wins_f1/total*100:.1f}%)")

    mac_wins_rag = sum(1 for i in range(len(metrics["mac"])) if metrics["mac"][i]["f1"] > metrics["rag"][i]["f1"])
    logger.info(f"  MAC 胜 RAG 比例 (F1):    {mac_wins_rag}/{total} ({mac_wins_rag/total*100:.1f}%)")

    return {
        "summary": summary,
        "mac_win_rate_vs_no_mem": mac_wins_f1 / total if total else 0,
        "rag_win_rate_vs_no_mem": rag_wins_f1 / total if total else 0,
        "mac_win_rate_vs_rag": mac_wins_rag / total if total else 0,
        "examples": examples[:5],  # 只保留前5个示例
    }


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    _lazy_import()

    t_start = time.time()

    # ---- 加载 backbone ---- #
    logger.info(f"加载 backbone: {args.model_path}")
    model, tokenizer, hidden_dim, num_layers = load_backbone(
        args.model_path, args.device, args.dtype, args.sliding_window
    )

    # ---- 加载 MAC 权重 ---- #
    logger.info(f"加载 MAC 权重: {args.mac_weights_dir}")
    selector, projector, mac_config = load_mac_weights(
        args.mac_weights_dir, hidden_dim, args.device
    )

    # ---- 初始化 MemoryEncoder ---- #
    deep_layers = args.deep_encode_layers
    enc_cfg = MemoryEncoderConfig(
        max_memory_tokens=mac_config.get("max_memory_tokens", 64),
        pooling="mean",
        deep_encode_layers=deep_layers,
    )
    encoder = MemoryEncoder(enc_cfg)
    encoder.set_backbone(
        backbone_model=model, tokenizer=tokenizer,
        hidden_dim=hidden_dim, device=args.device,
        dtype=getattr(torch, args.dtype, torch.float32),
    )

    # ---- 加载测试数据 ---- #
    logger.info(f"加载测试数据: {args.test_data}")
    test_data = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
    logger.info(f"测试样本数: {len(test_data)}")

    # ---- 运行评估 ---- #
    results = {}

    # 评估 1: PPL 三方对比
    results["ppl"] = eval_ppl_comparison(
        model, tokenizer, encoder, selector, projector, test_data, args
    )

    # 评估 2: Selector 准确率
    results["selector"] = eval_selector(
        model, tokenizer, encoder, selector, test_data, args
    )

    # 评估 3: 生成质量对比
    results["generation"] = eval_generation(
        model, tokenizer, encoder, selector, projector, test_data, args
    )

    total_time = time.time() - t_start

    # ---- 汇总报告 ---- #
    logger.info("\n" + "=" * 60)
    logger.info("📊 Benchmark 汇总报告")
    logger.info("=" * 60)

    logger.info(f"\n  1. PPL 对比:")
    logger.info(f"     No Memory:   {results['ppl'].get('ppl_no_mem', 'N/A'):.4f}")
    logger.info(f"     RAG Concat:  {results['ppl'].get('ppl_rag', 'N/A'):.4f}")
    logger.info(f"     MAC Prefix:  {results['ppl'].get('ppl_mac', 'N/A'):.4f}")
    logger.info(f"     MAC 降低:    {results['ppl'].get('mac_ppl_drop_pct', 0):.2f}%")
    logger.info(f"     RAG 降低:    {results['ppl'].get('rag_ppl_drop_pct', 0):.2f}%")

    logger.info(f"\n  2. Selector (测试集):")
    logger.info(f"     Hit@K:       {results['selector'].get('hit_at_k', 0):.4f}")
    logger.info(f"     Precision@K: {results['selector'].get('precision_at_k', 0):.4f}")
    logger.info(f"     NDCG@K:      {results['selector'].get('ndcg_at_k', 0):.4f}")
    logger.info(f"     区分度:       {results['selector'].get('score_gap', 0):.4f}")

    gen = results.get("generation", {})
    gen_sum = gen.get("summary", {})
    logger.info(f"\n  3. 生成质量 (字符级 F1):")
    logger.info(f"     No Memory:   {gen_sum.get('no_mem', {}).get('avg_f1', 0):.4f}")
    logger.info(f"     RAG Concat:  {gen_sum.get('rag', {}).get('avg_f1', 0):.4f}")
    logger.info(f"     MAC Prefix:  {gen_sum.get('mac', {}).get('avg_f1', 0):.4f}")
    logger.info(f"     MAC 胜 No-Mem: {gen.get('mac_win_rate_vs_no_mem', 0)*100:.1f}%")
    logger.info(f"     MAC 胜 RAG:    {gen.get('mac_win_rate_vs_rag', 0)*100:.1f}%")

    logger.info(f"\n  总耗时: {total_time:.1f}s")

    # ---- 保存 ---- #
    results["meta"] = {
        "model_path": args.model_path,
        "mac_weights_dir": args.mac_weights_dir,
        "test_data": args.test_data,
        "num_test_samples": len(test_data),
        "total_time_s": total_time,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
