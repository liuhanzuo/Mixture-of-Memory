#!/usr/bin/env python3
"""
MAC (Memory-Augmented Context) 训练脚本: 训练 ContextSelector + PrefixProjector。

与 MAG 的关键区别:
- MAG: 在 backbone 中间层做 CrossAttn + Gate 残差注入 → 侵入 backbone → 输出崩溃
- MAC: 在输入端拼接 soft prefix tokens → 零侵入 → backbone 语言能力完全保持

训练流程:
1. 加载 backbone (Qwen3) 并冻结其参数
2. 初始化 MemoryEncoder (共享 backbone embedding, 不训练)
3. 初始化 ContextSelector (可训练, 复用已有权重或从头训练)
4. 初始化 PrefixProjector (可训练)
5. 训练:
   - 编码记忆 → ContextSelector 打分 → PrefixProjector 生成 prefix tokens
   - 将 prefix tokens 拼接到 prompt embedding 前面
   - backbone 正常前向推理, 计算 LM loss
   - 梯度只更新 ContextSelector + PrefixProjector

支持单卡和多卡 DDP 训练:
    # 单卡
    python scripts/train_mac.py \\
        --model_path ../models/Qwen--Qwen3-8b \\
        --output_dir outputs/mac_trained

    # 多卡 DDP (单机 8 卡)
    torchrun --nproc_per_node=8 scripts/train_mac.py \\
        --model_path ../models/Qwen--Qwen3-8b \\
        --output_dir outputs/mac_trained
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# 将项目根目录加入 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.memory.mag.memory_encoder import MemoryEncoder, MemoryEncoderConfig
from src.memory.mag.context_selector import ContextSelector, ContextSelectorConfig
from src.memory.mag.prefix_projector import PrefixProjector, PrefixProjectorConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_mac")


# ======================================================================
# 分布式训练工具函数 (与 train_mag.py 相同)
# ======================================================================

def setup_distributed() -> tuple[int, int, int]:
    """初始化分布式训练环境。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        if rank == 0:
            logger.info(f"分布式训练已初始化: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return rank, local_rank, world_size
    else:
        logger.info("未检测到分布式环境变量, 使用单卡模式")
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    return rank == 0


def dist_barrier():
    if dist.is_initialized():
        dist.barrier()


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


class MACTrainDataset(Dataset):
    """MAC 训练数据集封装。"""
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


# ======================================================================
# 参数解析
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAC (Memory-Augmented Context) 训练脚本")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="../models/Qwen--Qwen3-8b",
                        help="Backbone 模型路径")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # MAC 架构配置
    parser.add_argument("--tokens_per_memory", type=int, default=4,
                        help="每条记忆映射为多少个 soft token (推荐 2~8)")
    parser.add_argument("--projector_layers", type=int, default=2,
                        help="PrefixProjector MLP 层数")
    parser.add_argument("--projector_expansion", type=float, default=1.5,
                        help="PrefixProjector MLP 扩展倍数")
    parser.add_argument("--projector_init_scale", type=float, default=0.01,
                        help="PrefixProjector 输出层初始化缩放 (小值=训练初期接近零)")
    parser.add_argument("--use_gating", action="store_true", default=True,
                        help="PrefixProjector 是否使用记忆重要性 gate")
    parser.add_argument("--no_gating", action="store_true", default=False,
                        help="禁用 PrefixProjector gating")
    parser.add_argument("--selector_hidden_dim", type=int, default=256,
                        help="ContextSelector MLP 隐藏层维度")
    parser.add_argument("--selector_top_k", type=int, default=5,
                        help="选出的 top-k 记忆数")
    parser.add_argument("--max_memory_tokens", type=int, default=64,
                        help="每条记忆最大 token 数")
    parser.add_argument("--deep_encode_layers", type=int, default=8,
                        help="记忆深层编码使用的 backbone 层数")

    # 可选: 从 MAG 已训练的 ContextSelector 加载权重
    parser.add_argument("--pretrained_selector", type=str, default="",
                        help="预训练的 ContextSelector 权重路径 (可从 MAG 训练中复用)")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="outputs/mac_trained")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练序列最大长度")
    parser.add_argument("--sliding_window", type=int, default=0,
                        help="启用 SWA 并设置窗口大小 (0=不启用, 建议 4096)")
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)

    # Anti-Teacher-Forcing (MAC 架构更安全, 但仍保留作为选项)
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Label Smoothing (MAC 下可以设较小值, 因为不侵入 backbone)")
    parser.add_argument("--kl_beta", type=float, default=0.1,
                        help="KL 约束系数 (MAC 下可以设较小值)")
    parser.add_argument("--kl_temperature", type=float, default=2.0)

    # 分布式
    parser.add_argument("--local_rank", type=int, default=-1)

    # 数据配置
    parser.add_argument("--data_source", type=str, default="locomo",
                        choices=["synthetic", "msc", "locomo", "jsonl"],
                        help="训练数据源")
    parser.add_argument("--data_path", type=str, default="",
                        help="数据文件路径")
    parser.add_argument("--msc_subset", type=str, default="session_1")
    parser.add_argument("--num_synthetic_samples", type=int, default=1000)
    parser.add_argument("--max_real_samples", type=int, default=10000)
    parser.add_argument("--num_memories_per_sample", type=int, default=10)
    parser.add_argument("--num_hard_negatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ======================================================================
# 模型加载
# ======================================================================

def load_backbone(model_path: str, device: str, dtype_str: str, sliding_window: int = 0):
    """加载 backbone 模型和 tokenizer。"""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.float32)
    logger.info(f"加载模型: {model_path}")
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

    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    logger.info(f"模型加载完成: hidden_dim={hidden_dim}, num_layers={num_layers}")
    return model, tokenizer, hidden_dim, num_layers


# ======================================================================
# 数据加载 (复用 train_mag.py 的数据加载函数)
# ======================================================================

def load_training_data(args: argparse.Namespace) -> list[dict[str, Any]]:
    """加载训练数据, 复用 train_mag.py 的数据加载逻辑。"""
    # 导入 train_mag 中的数据加载函数
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_mag import (
        load_msc_dataset,
        load_locomo_dataset,
        conversations_to_mag_samples,
        load_jsonl_dataset,
        generate_synthetic_data,
    )

    if args.data_source == "synthetic":
        logger.info("使用合成数据")
        return generate_synthetic_data(
            tokenizer=None, num_samples=args.num_synthetic_samples,
            num_memories=args.num_memories_per_sample,
            max_seq_len=args.max_seq_len, seed=args.seed,
        )
    elif args.data_source == "msc":
        conversations = load_msc_dataset(
            subset=args.msc_subset, max_samples=args.max_real_samples,
            data_path=args.data_path,
        )
        return conversations_to_mag_samples(
            conversations, num_memories_per_sample=args.num_memories_per_sample,
            num_hard_negatives=args.num_hard_negatives, seed=args.seed,
        )
    elif args.data_source == "locomo":
        if not args.data_path:
            raise ValueError("data_source=locomo 时需要指定 --data_path")
        conversations = load_locomo_dataset(
            data_path=args.data_path, max_samples=args.max_real_samples,
        )
        return conversations_to_mag_samples(
            conversations, num_memories_per_sample=args.num_memories_per_sample,
            num_hard_negatives=args.num_hard_negatives, seed=args.seed,
        )
    elif args.data_source == "jsonl":
        if not args.data_path:
            raise ValueError("data_source=jsonl 时需要指定 --data_path")
        data = load_jsonl_dataset(args.data_path, max_samples=args.max_real_samples)
        if data and "input_text" in data[0] and "memory_texts" in data[0]:
            return data[:args.max_real_samples]
        else:
            return conversations_to_mag_samples(
                data, num_memories_per_sample=args.num_memories_per_sample,
                num_hard_negatives=args.num_hard_negatives, seed=args.seed,
            )
    else:
        raise ValueError(f"未知的 data_source: {args.data_source}")


# ======================================================================
# 训练 Phase 1: 预训练 ContextSelector (如果没有预训练权重)
# ======================================================================

def train_phase1_selector(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: nn.Module,
    projector: nn.Module,
    train_data: list[dict[str, Any]],
    args: argparse.Namespace,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Phase 1: 使用 counterfactual 信号预训练 ContextSelector。

    通过比较"有某条记忆"vs"无某条记忆"时 prefix 对 LM loss 的影响,
    训练 selector 学会识别有用记忆。
    """
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("Phase 1: 预训练 ContextSelector")
        logger.info("=" * 60)

    selector_module = selector.module if isinstance(selector, DDP) else selector
    projector_module = projector.module if isinstance(projector, DDP) else projector

    optimizer = optim.AdamW(selector_module.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)

    dataset = MACTrainDataset(train_data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed) if world_size > 1 else None

    log_interval = args.log_every if args.log_every > 0 else max(len(train_data) // 20, 10)

    selector.train()
    projector.eval()  # projector 在 Phase 1 不训练
    model.eval()

    epoch_loss = 0.0
    epoch_steps = 0

    indices = list(sampler) if sampler is not None else list(range(len(dataset)))

    for step_i, data_idx in enumerate(indices):
        sample = dataset[data_idx]

        if "relevant_indices" not in sample or not sample["relevant_indices"]:
            continue

        with torch.no_grad():
            query_emb = encoder.encode_texts_deep([sample["input_text"]])
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"])
            memory_embs = memory_embs.unsqueeze(0)

        # Selector 打分 (有梯度)
        scores = selector_module(query_emb.detach(), memory_embs.detach())

        # 构造 target
        K = len(sample["memory_texts"])
        target = torch.zeros(1, K, device=scores.device)
        for idx in sample["relevant_indices"]:
            if idx < K:
                target[0, idx] = 1.0

        loss = selector_module.compute_loss(scores, target)

        loss.backward()

        if (step_i + 1) % args.grad_accumulation_steps == 0:
            if world_size > 1:
                for p in selector_module.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(selector_module.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_steps += 1

        if is_main_process(rank) and (step_i + 1) % log_interval == 0:
            avg_loss = epoch_loss / epoch_steps
            logger.info(f"  [P1] [{step_i+1}/{len(indices)}] loss={avg_loss:.4f}")

    if is_main_process(rank):
        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Phase 1 完成: avg_loss={avg_loss:.4f}, steps={epoch_steps}")


# ======================================================================
# 训练 Phase 2: 联合训练 Selector + PrefixProjector
# ======================================================================

def train_phase2_joint(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: nn.Module,
    projector: nn.Module,
    train_data: list[dict[str, Any]],
    args: argparse.Namespace,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Phase 2: 联合训练 Selector + PrefixProjector (端到端 LM loss)。

    核心优势 (对比 MAG 的 Phase 2):
    - 不需要分段 forward + 手动逐层处理
    - 直接用 backbone(inputs_embeds=cat[prefix, prompt]) 做标准前向
    - backbone 完全不修改, 梯度只流过 prefix_tokens → PrefixProjector + Selector
    """
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("Phase 2: 联合训练 Selector + PrefixProjector (端到端)")
        logger.info("=" * 60)

    selector_module = selector.module if isinstance(selector, DDP) else selector
    projector_module = projector.module if isinstance(projector, DDP) else projector

    # 训练 selector + projector 参数
    trainable_params = list(selector.parameters()) + list(projector.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = MACTrainDataset(train_data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed) if world_size > 1 else None
    steps_per_epoch = len(dataset) // max(world_size, 1)

    total_steps = steps_per_epoch * args.num_epochs // args.grad_accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps + 1,
        pct_start=warmup_steps / max(total_steps, 1), anneal_strategy="cos",
    )

    log_interval = args.log_every if args.log_every > 0 else max(steps_per_epoch // 20, 10)
    if is_main_process(rank):
        logger.info(f"数据量: {len(train_data)} (每卡约 {steps_per_epoch}), 每 {log_interval} 步打印")

    selector.train()
    projector.train()
    model.eval()

    num_steps = 0
    prev_epoch_loss = None

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        indices = list(sampler) if sampler is not None else list(range(len(dataset)))

        for step_i, data_idx in enumerate(indices):
            sample = dataset[data_idx]

            # ---- Step 1: 编码记忆 (无梯度) ---- #
            with torch.no_grad():
                query_emb = encoder.encode_texts_deep([sample["input_text"]])
                memory_embs = encoder.encode_texts_deep(sample["memory_texts"])
                memory_embs = memory_embs.unsqueeze(0)  # (1, K, D)

            # ---- Step 2: Selector 打分 (有梯度) ---- #
            selection_weights = selector_module.soft_select(
                query_emb.detach(), memory_embs.detach()
            )  # (1, K)

            # ---- Step 3: PrefixProjector 生成 soft prefix tokens (有梯度) ---- #
            prefix_tokens = projector_module(
                memory_embs.detach(), selection_weights
            )  # (1, prefix_len, D)

            # ---- Step 4: 构造 [prefix | query | target] 序列 ---- #
            has_target = "target_text" in sample and sample["target_text"]

            if has_target:
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
            else:
                encoded = tokenizer(
                    sample["input_text"], return_tensors="pt",
                    truncation=True, max_length=args.max_seq_len,
                )
                input_ids = encoded["input_ids"].to(args.device)
                query_len = input_ids.shape[1]

            # 获取 prompt embedding
            with torch.no_grad():
                if hasattr(model.model, "embed_tokens"):
                    prompt_embeds = model.model.embed_tokens(input_ids)
                else:
                    prompt_embeds = model.get_input_embeddings()(input_ids)

            # ★ 核心: 拼接 [prefix_tokens | prompt_embeds]
            # prefix_tokens 有梯度 → 梯度流回 PrefixProjector + Selector
            # prompt_embeds 无梯度 → backbone embedding 不被修改
            full_embeds = torch.cat([prefix_tokens, prompt_embeds], dim=1)  # (1, P+T, D)

            prefix_len = prefix_tokens.shape[1]

            # 构造 labels: prefix 部分为 -100, query 部分为 -100, target 部分有标签
            total_len = full_embeds.shape[1]
            labels = torch.full((1, total_len), -100, dtype=torch.long, device=args.device)
            if has_target:
                # target 部分的标签 (偏移 prefix_len)
                target_start = prefix_len + query_len
                target_end = total_len
                labels[0, target_start:target_end] = input_ids[0, query_len:query_len + (target_end - target_start)]
            else:
                # 无 target: 对 prompt 部分 (不含 prefix) 做自回归
                labels[0, prefix_len:] = input_ids[0, :]

            # 构造 attention_mask: 全部有效
            attention_mask = torch.ones((1, total_len), dtype=torch.long, device=args.device)

            # ★ position_ids: prefix 和 prompt 共享一个连续的位置编码
            position_ids = torch.arange(total_len, device=args.device).unsqueeze(0)

            # ---- Step 5: backbone 前向 (完全不修改 backbone!) ---- #
            # 关键: 只传 inputs_embeds, 不传 input_ids
            # backbone 看到的就是一个稍长的 token 序列, 它自己通过 attention 从 prefix 中提取信息
            outputs = model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=None,  # 手动计算 loss
            )
            logits = outputs.logits  # (1, P+T, vocab_size)

            # ---- Step 6: 计算 loss ---- #
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=args.label_smoothing,
            )

            # ★ KL 约束: 有记忆 vs 无记忆输出分布不能差太远
            kl_loss = torch.tensor(0.0, device=lm_loss.device)
            if args.kl_beta > 0 and has_target:
                with torch.no_grad():
                    # 无记忆 baseline: 只用 prompt_embeds 前向
                    out_no_mem = model(
                        inputs_embeds=prompt_embeds,
                        attention_mask=torch.ones((1, prompt_embeds.shape[1]), dtype=torch.long, device=args.device),
                        labels=None,
                    )
                    logits_no_mem = out_no_mem.logits

                # 对齐: 有记忆的 logits 从 prefix_len 开始对应原始 prompt
                logits_with_mem = logits[:, prefix_len:, :]
                # 两者长度应该相同
                min_len = min(logits_with_mem.shape[1], logits_no_mem.shape[1])
                if min_len > 1:
                    T = args.kl_temperature
                    log_p_with = F.log_softmax(logits_with_mem[:, :min_len-1, :].contiguous().view(-1, logits.size(-1)) / T, dim=-1)
                    p_without = F.softmax(logits_no_mem[:, :min_len-1, :].contiguous().view(-1, logits.size(-1)) / T, dim=-1)
                    kl_loss = F.kl_div(log_p_with, p_without.detach(), reduction="batchmean") * (T * T)

            # 辅助 loss: Selector 打分辅助
            aux_loss = torch.tensor(0.0, device=lm_loss.device)
            if "relevant_indices" in sample and sample["relevant_indices"]:
                K = len(sample["memory_texts"])
                sel_target = torch.zeros(1, K, device=lm_loss.device)
                for idx in sample["relevant_indices"]:
                    if idx < K:
                        sel_target[0, idx] = 1.0
                raw_scores = selector_module(query_emb.detach(), memory_embs.detach())
                aux_loss = F.binary_cross_entropy_with_logits(raw_scores, sel_target)

            # 总 loss
            loss = 1.0 * lm_loss + args.kl_beta * kl_loss + 0.1 * aux_loss

            # nan guard
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(rank) and step_i < 10:
                    logger.warning(f"  [P2] Step {step_i}: loss={loss.item():.4f} 跳过 (nan/inf)")
                optimizer.zero_grad()
                continue

            # 梯度累积
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            if (step_i + 1) % args.grad_accumulation_steps == 0:
                if world_size > 1:
                    for param in trainable_params:
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                num_steps += 1

            epoch_loss += loss.item() * args.grad_accumulation_steps
            epoch_lm_loss += lm_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_aux_loss += aux_loss.item()
            epoch_steps += 1

            if is_main_process(rank) and (step_i + 1) % log_interval == 0:
                avg = epoch_loss / epoch_steps
                avg_lm = epoch_lm_loss / epoch_steps
                avg_kl = epoch_kl_loss / epoch_steps
                avg_aux = epoch_aux_loss / epoch_steps
                elapsed = time.time() - t_epoch_start
                eta = elapsed / (step_i + 1) * (len(indices) - step_i - 1)
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(
                    f"  [P2] Epoch {epoch+1} [{step_i+1}/{len(indices)}] "
                    f"loss={avg:.4f} (lm={avg_lm:.4f} kl={avg_kl:.4f} aux={avg_aux:.4f}) "
                    f"lr={lr_now:.2e} ETA={eta:.0f}s"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

        if world_size > 1:
            loss_tensor = torch.tensor(avg_epoch_loss, device=args.device)
            reduce_mean(loss_tensor, world_size)
            avg_epoch_loss = loss_tensor.item()

        epoch_time = time.time() - t_epoch_start
        delta_str = ""
        if prev_epoch_loss is not None:
            delta = avg_epoch_loss - prev_epoch_loss
            delta_str = f"  Δ={delta:+.4f}"
        prev_epoch_loss = avg_epoch_loss

        if is_main_process(rank):
            logger.info(f"Epoch {epoch+1} 完成: avg_loss={avg_epoch_loss:.4f}{delta_str} 耗时={epoch_time:.1f}s")

    if is_main_process(rank):
        logger.info(f"Phase 2 完成: total_steps={num_steps}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # 分布式初始化
    rank, local_rank, world_size = setup_distributed()

    if world_size > 1:
        args.device = f"cuda:{local_rank}"
    elif args.device == "cuda" and torch.cuda.is_available():
        args.device = "cuda:0"

    if not is_main_process(rank):
        logging.getLogger("train_mac").setLevel(logging.WARNING)
        logging.getLogger("src.memory.mag").setLevel(logging.WARNING)

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    dist_barrier()

    # 1. 加载 backbone (节点级串行加载)
    if world_size > 1:
        if local_rank == 0:
            logger.info(f"[节点串行加载] local_rank=0 开始加载 backbone...")
            model, tokenizer, hidden_dim, num_layers = load_backbone(
                args.model_path, args.device, args.dtype, sliding_window=args.sliding_window,
            )
        dist.barrier()
        if local_rank != 0:
            model, tokenizer, hidden_dim, num_layers = load_backbone(
                args.model_path, args.device, args.dtype, sliding_window=args.sliding_window,
            )
        dist.barrier()
    else:
        model, tokenizer, hidden_dim, num_layers = load_backbone(
            args.model_path, args.device, args.dtype, sliding_window=args.sliding_window,
        )

    # 2. MemoryEncoder
    enc_cfg = MemoryEncoderConfig(
        max_memory_tokens=args.max_memory_tokens,
        pooling="mean",
        deep_encode_layers=args.deep_encode_layers,
    )
    encoder = MemoryEncoder(enc_cfg)
    encoder.set_backbone(
        backbone_model=model, tokenizer=tokenizer,
        hidden_dim=hidden_dim, device=args.device,
        dtype=getattr(torch, args.dtype, torch.float32),
    )

    # 3. ContextSelector
    sel_cfg = ContextSelectorConfig(
        input_dim=hidden_dim,
        hidden_dim=args.selector_hidden_dim,
        top_k=args.selector_top_k,
    )
    selector = ContextSelector(sel_cfg).to(args.device)

    # 可选: 从 MAG 训练中加载预训练的 selector 权重
    if args.pretrained_selector and Path(args.pretrained_selector).exists():
        selector.load_state_dict(torch.load(args.pretrained_selector, map_location=args.device))
        if is_main_process(rank):
            logger.info(f"已加载预训练 Selector: {args.pretrained_selector}")

    # 4. PrefixProjector (★ 新模块, 替代 MAGGate)
    use_gating = args.use_gating and not args.no_gating
    proj_cfg = PrefixProjectorConfig(
        hidden_dim=hidden_dim,
        tokens_per_memory=args.tokens_per_memory,
        num_mlp_layers=args.projector_layers,
        mlp_expansion=args.projector_expansion,
        dropout=0.1,
        use_layer_norm=True,
        use_gating=use_gating,
        init_scale=args.projector_init_scale,
    )
    projector = PrefixProjector(proj_cfg).to(args.device)

    # DDP 包装
    if world_size > 1:
        selector = DDP(selector, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=False)
        projector = DDP(projector, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)
        if is_main_process(rank):
            logger.info(f"DDP 包装完成: selector + projector (world_size={world_size})")

    # 打印参数量
    sel_mod = selector.module if isinstance(selector, DDP) else selector
    proj_mod = projector.module if isinstance(projector, DDP) else projector
    total_trainable = sum(p.numel() for p in sel_mod.parameters()) + \
                      sum(p.numel() for p in proj_mod.parameters())
    if is_main_process(rank):
        logger.info(f"可训练参数量: {total_trainable:,}")
        logger.info(f"  Selector: {sum(p.numel() for p in sel_mod.parameters()):,}")
        logger.info(f"  PrefixProjector: {sum(p.numel() for p in proj_mod.parameters()):,}")
        logger.info(f"  tokens_per_memory: {args.tokens_per_memory}")

    # 5. 加载训练数据
    train_data = load_training_data(args)
    if not train_data:
        if is_main_process(rank):
            logger.error("训练数据为空")
        cleanup_distributed()
        return

    if is_main_process(rank):
        logger.info(f"训练数据: {len(train_data)} 条")

    t0 = time.time()

    # Phase 1: 预训练 Selector (如果没有预训练权重)
    if not args.pretrained_selector:
        train_phase1_selector(
            model, tokenizer, encoder, selector, projector, train_data, args,
            rank=rank, world_size=world_size,
        )
        dist_barrier()

    # Phase 2: 联合训练 Selector + PrefixProjector
    train_phase2_joint(
        model, tokenizer, encoder, selector, projector, train_data, args,
        rank=rank, world_size=world_size,
    )
    dist_barrier()

    total_time = time.time() - t0

    # 保存
    if is_main_process(rank):
        save_path = Path(args.output_dir)
        torch.save(sel_mod.state_dict(), save_path / "context_selector.pt")
        torch.save(proj_mod.state_dict(), save_path / "prefix_projector.pt")

        config_info = {
            "architecture": "MAC",  # ★ 标识这是 MAC 而非 MAG
            "model_path": args.model_path,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "tokens_per_memory": args.tokens_per_memory,
            "projector_layers": args.projector_layers,
            "projector_expansion": args.projector_expansion,
            "projector_init_scale": args.projector_init_scale,
            "use_gating": use_gating,
            "selector_hidden_dim": args.selector_hidden_dim,
            "selector_top_k": args.selector_top_k,
            "max_memory_tokens": args.max_memory_tokens,
            "deep_encode_layers": args.deep_encode_layers,
            "total_trainable_params": total_trainable,
            "training_time_s": total_time,
            "world_size": world_size,
        }
        with open(save_path / "mac_config.json", "w") as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)

        logger.info(f"训练完成! 总耗时: {total_time:.1f}s")
        logger.info(f"权重已保存到: {args.output_dir}")
        logger.info(f"  context_selector.pt: {(save_path / 'context_selector.pt').stat().st_size / 1024:.1f} KB")
        logger.info(f"  prefix_projector.pt: {(save_path / 'prefix_projector.pt').stat().st_size / 1024:.1f} KB")

    cleanup_distributed()


if __name__ == "__main__":
    main()
