#!/usr/bin/env python3
"""
MAG 训练脚本: 联合训练 ContextSelector + MAGGate。

训练流程:
1. 加载 backbone (Qwen3) 并冻结其参数
2. 初始化 MemoryEncoder (共享 backbone embedding, 不训练)
3. 初始化 ContextSelector (可训练)
4. 初始化 MAGGate (可训练)
5. 使用 MoM benchmark 数据作为训练数据:
   - Phase 1: 收集 counterfactual ΔLoss 数据, 预训练 Scorer
   - Phase 2: 联合训练 Scorer + Gate, 端到端优化 LM loss
6. 保存训练好的 Scorer + Gate 权重

支持单卡和多卡 DDP 训练:

    # 单卡
    python scripts/train_mag.py \\
        --model_path ../models/Qwen--Qwen3-8b \\
        --output_dir outputs/mag_trained \\
        --num_epochs 3 --lr 1e-4

    # 多卡 DDP (单机 4 卡)
    torchrun --nproc_per_node=4 scripts/train_mag.py \\
        --model_path ../models/Qwen--Qwen3-8b \\
        --output_dir outputs/mag_trained \\
        --mag_injection_layers 6 12 18 23 \\
        --num_epochs 3 --lr 1e-4

    # 多机多卡 (2 机各 4 卡)
    torchrun --nnodes=2 --nproc_per_node=4 \\
        --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:29500 \\
        scripts/train_mag.py \\
        --model_path ../models/Qwen--Qwen3-8b \\
        --output_dir outputs/mag_trained \\
        --num_epochs 3 --lr 1e-4
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
from src.memory.mag.mag_gate import MAGGate, MAGGateConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_mag")


# ======================================================================
# 分布式训练工具函数
# ======================================================================

def setup_distributed() -> tuple[int, int, int]:
    """初始化分布式训练环境。

    自动检测 torchrun 设置的环境变量, 如果不存在则回退到单卡模式。

    Returns:
        (rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])

        # 设置当前 GPU
        torch.cuda.set_device(local_rank)

        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )

        if rank == 0:
            logger.info(f"分布式训练已初始化: rank={rank}, local_rank={local_rank}, "
                        f"world_size={world_size}")
        return rank, local_rank, world_size
    else:
        # 单卡模式
        logger.info("未检测到分布式环境变量, 使用单卡模式")
        return 0, 0, 1


def cleanup_distributed():
    """清理分布式训练环境。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """是否为主进程 (rank 0)。"""
    return rank == 0


def get_rank() -> int:
    """获取当前进程的 global rank。"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数。"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def dist_barrier():
    """分布式同步屏障。"""
    if dist.is_initialized():
        dist.barrier()


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """跨进程求均值 (in-place)。"""
    if world_size <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


class MAGTrainDataset(Dataset):
    """MAG 训练数据的 Dataset 封装, 用于配合 DistributedSampler。"""

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAG (Memory-Augmented Generation) 训练脚本")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="../models/Qwen--Qwen3-1.7b",
                        help="Backbone 模型路径")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # MAG 架构配置
    parser.add_argument("--mag_num_heads", type=int, default=8, help="CrossAttention 头数")
    parser.add_argument("--mag_injection_layers", type=int, nargs="+", default=None,
                        help="注入的 Transformer 层索引 (如 6 12 18 23)")
    parser.add_argument("--mag_share_parameters", action="store_true", default=True,
                        help="注入层之间是否共享参数")
    parser.add_argument("--mag_gate_init_bias", type=float, default=-2.0,
                        help="Gate 偏置初始值 (负值使初始 gate 接近 0)")
    parser.add_argument("--selector_hidden_dim", type=int, default=256,
                        help="Selector MLP 隐藏层维度")
    parser.add_argument("--selector_top_k", type=int, default=5, help="选出的 top-k 记忆数")
    parser.add_argument("--max_memory_tokens", type=int, default=64, help="每条记忆最大 token 数")
    parser.add_argument("--deep_encode_layers", type=int, default=8,
                        help="记忆深层编码使用的 backbone 层数 (0=仅用 embedding, 推荐 8~12)")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="outputs/mag_trained")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练序列最大长度")
    parser.add_argument("--sliding_window", type=int, default=0,
                        help="启用 SWA 并设置窗口大小 (0=不启用, 建议 4096). "
                             "启用后训练序列应 >> window_size 才能让 MAG 学到有意义的注入")
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50,
                        help="每隔多少步打印一次日志 (0=自动根据数据量调整)")
    parser.add_argument("--save_every", type=int, default=500)

    # ★ Anti-Teacher-Forcing 配置 (解决 lm_loss → 0 的问题)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label Smoothing 系数 (0=不使用, 推荐 0.1). "
                             "从 loss 端阻止模型过度自信, 即使 CrossAttn 能从记忆中读到答案, "
                             "也不会被鼓励把概率推到 1.0")
    parser.add_argument("--kl_beta", type=float, default=0.5,
                        help="KL 约束系数 (0=不使用, 推荐 0.1~1.0). "
                             "限制有记忆/无记忆输出分布的差距, 确保记忆只起辅助作用而非主导生成")
    parser.add_argument("--kl_temperature", type=float, default=2.0,
                        help="KL 约束的温度参数 (推荐 1.0~4.0). "
                             "较高温度使分布更平滑, 聚焦于整体分布形状而非极端值")
    parser.add_argument("--detach_value", action="store_true", default=True,
                        help="对 CrossAttention 的 V 分支做 stop-gradient. "
                             "Q/K 仍有梯度 (学习'去哪查'), V 无梯度 (阻止'抄什么'), "
                             "从根源上解耦 MAG 注入答案的路径")
    parser.add_argument("--no_detach_value", action="store_true", default=False,
                        help="禁用 detach_value (调试用)")
    parser.add_argument("--scheduled_sampling", action="store_true", default=False,
                        help="启用 Scheduled Sampling (训练后期逐步用模型自回归 token 替代 ground truth). "
                             "缓解训练/推理分布不匹配 (exposure bias)")
    parser.add_argument("--ss_start_epoch", type=int, default=1,
                        help="Scheduled Sampling 生效的起始 epoch (0-based, 推荐从第 2 个 epoch 开始)")
    parser.add_argument("--ss_max_ratio", type=float, default=0.5,
                        help="Scheduled Sampling 的最大替换比例 (0.0~1.0, 推荐 0.3~0.5). "
                             "最终训练时最多有 ss_max_ratio 比例的 token 用模型自己的生成替代 ground truth")

    # 分布式训练配置 (torchrun 自动设置 LOCAL_RANK)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="由 torchrun 自动设置, 无需手动指定")

    # 数据配置
    parser.add_argument("--data_source", type=str, default="msc",
                        choices=["synthetic", "msc", "locomo", "jsonl"],
                        help="训练数据源: synthetic=合成, msc=Multi-Session Chat, "
                             "locomo=LoCoMo, jsonl=自定义JSONL")
    parser.add_argument("--data_path", type=str, default="",
                        help="JSONL/LoCoMo 数据文件路径 (data_source=jsonl/locomo 时必需)")
    parser.add_argument("--msc_subset", type=str, default="session_1",
                        help="MSC 子集名称")
    parser.add_argument("--num_synthetic_samples", type=int, default=1000,
                        help="合成训练样本数量 (仅 data_source=synthetic)")
    parser.add_argument("--max_real_samples", type=int, default=10000,
                        help="真实数据最大样本数")
    parser.add_argument("--num_memories_per_sample", type=int, default=10,
                        help="每个训练样本的候选记忆条数")
    parser.add_argument("--num_hard_negatives", type=int, default=3,
                        help="每个样本的硬负例数 (来自同一 session 的非相关轮)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_backbone(model_path: str, device: str, dtype_str: str, sliding_window: int = 0) -> tuple[nn.Module, Any, int, int]:
    """加载 backbone 模型和 tokenizer。

    Args:
        model_path: 模型路径.
        device: 设备.
        dtype_str: 数据类型.
        sliding_window: SWA 窗口大小 (0=不启用).
                       启用后, backbone 的 self-attention 只能看到窗口内的 token,
                       超出窗口的信息必须通过 MAG 注入才能获得。

    Returns:
        (model, tokenizer, hidden_dim, num_layers)
    """
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.float32)

    logger.info(f"加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # ★ 关键: 设置 SWA (Sliding Window Attention)
    # 启用后 backbone 的每层 self-attention 只看窗口内的 token,
    # 窗口外的信息必须通过 MAG 注入 → MAG 学到的是"真需求"而非"锦上添花"
    if sliding_window > 0:
        if hasattr(config, "sliding_window"):
            config.sliding_window = sliding_window
            logger.info(f"★ SWA 已启用: sliding_window={sliding_window}")
        else:
            logger.warning(f"模型 config 不支持 sliding_window 属性, SWA 未生效")
        # Qwen2/Qwen3: max_window_layers 控制前多少层使用 SWA
        if hasattr(config, "max_window_layers"):
            config.max_window_layers = config.num_hidden_layers  # 所有层都用 SWA
            logger.info(f"  max_window_layers={config.max_window_layers} (全部层启用 SWA)")
    else:
        logger.info("SWA 未启用 (sliding_window=0), backbone 使用 full attention")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 冻结 backbone 参数
    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    logger.info(f"模型加载完成: hidden_dim={hidden_dim}, num_layers={num_layers}")

    return model, tokenizer, hidden_dim, num_layers


# ======================================================================
# 真实数据加载与转换
# ======================================================================

def load_msc_dataset(subset: str = "session_1", max_samples: int = 2000,
                     data_path: str = "") -> list[dict]:
    """从 HuggingFace 或本地加载 Multi-Session Chat 数据集。

    加载策略 (按优先级):
        1. 如果指定了 data_path，从本地 JSONL/JSON 加载
        2. 尝试从 HuggingFace Hub 加载 (不使用 trust_remote_code)
        3. 尝试 ParlAI 格式的 MSC 本地缓存

    Returns:
        对话列表, 每条: {"messages": [{"role": ..., "content": ...}, ...], "personas": [...]}
    """
    import json as _json
    from pathlib import Path as _Path

    # --- 策略 1: 本地文件优先 ---
    if data_path:
        local_path = _Path(data_path)
        if local_path.exists():
            logger.info(f"从本地文件加载 MSC: {data_path}")
            return _load_msc_from_local(local_path, max_samples)
        else:
            logger.warning(f"指定的 data_path 不存在: {data_path}, 尝试在线加载")

    # --- 策略 2: HuggingFace Hub ---
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("请先安装 datasets: pip install datasets")

    logger.info(f"正在从 HuggingFace 加载 MSC ({subset})...")

    # 尝试多种数据集名称和参数组合
    load_attempts = [
        # 标准名称, 不带 trust_remote_code
        {"path": "facebook/msc", "name": subset, "split": "train"},
        # 社区镜像
        {"path": "parlai/msc", "name": subset, "split": "train"},
        # 不指定 name (有些版本 subset 是 config)
        {"path": "facebook/msc", "split": "train"},
    ]

    dataset = None
    last_error = None
    for attempt in load_attempts:
        try:
            logger.info(f"  尝试: {attempt}")
            dataset = load_dataset(**attempt)
            logger.info(f"  成功加载: {attempt['path']}")
            break
        except Exception as e:
            last_error = e
            logger.debug(f"  失败: {e}")
            continue

    if dataset is None:
        logger.error(f"MSC 在线加载失败 (最后错误: {last_error})")
        logger.info("="*60)
        logger.info("MSC 数据集需要认证或无法在线访问。请使用以下方式之一:")
        logger.info("  方式 1: 手动下载 MSC 数据, 以 JSONL 格式保存后指定 --data_path")
        logger.info("  方式 2: 设置 HF_TOKEN 环境变量 (huggingface-cli login)")
        logger.info("  方式 3: 使用 --data_source jsonl --data_path <your_data.jsonl>")
        logger.info("  方式 4: 使用 --data_source synthetic 先跑通流程 (仅 debug)")
        logger.info("="*60)
        logger.info("正在回退到使用 DailyDialog 公开数据集...")
        return _load_dailydialog_fallback(max_samples)

    conversations = []
    for i, example in enumerate(dataset):
        if len(conversations) >= max_samples:
            break

        dialog = example.get("dialog", example.get("dialogue", []))
        if not dialog or len(dialog) < 4:
            continue

        messages = []
        for j, utt in enumerate(dialog):
            if isinstance(utt, str):
                role = "user" if j % 2 == 0 else "assistant"
                messages.append({"role": role, "content": utt})
            elif isinstance(utt, dict):
                messages.append({
                    "role": utt.get("role", "user" if j % 2 == 0 else "assistant"),
                    "content": utt.get("text", utt.get("content", "")),
                })

        personas = example.get("personas", example.get("persona", []))
        if personas and isinstance(personas[0], list):
            personas = personas[0]

        conversations.append({"messages": messages, "personas": personas or []})

    logger.info(f"MSC 加载完成: {len(conversations)} 条对话")
    return conversations


def _load_msc_from_local(path, max_samples: int = 2000) -> list[dict]:
    """从本地 JSON/JSONL 文件加载 MSC 格式对话。"""
    import json as _json

    conversations = []
    suffix = str(path).lower()

    if suffix.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                conv = _parse_msc_example(obj)
                if conv:
                    conversations.append(conv)
                if len(conversations) >= max_samples:
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())
        if isinstance(raw, list):
            for obj in raw:
                conv = _parse_msc_example(obj)
                if conv:
                    conversations.append(conv)
                if len(conversations) >= max_samples:
                    break

    logger.info(f"从本地加载 MSC: {len(conversations)} 条对话")
    return conversations


def _parse_msc_example(example: dict) -> dict | None:
    """解析单个 MSC example 为统一格式。"""
    dialog = example.get("dialog", example.get("dialogue", []))
    if not dialog or len(dialog) < 4:
        return None

    messages = []
    for j, utt in enumerate(dialog):
        if isinstance(utt, str):
            role = "user" if j % 2 == 0 else "assistant"
            messages.append({"role": role, "content": utt})
        elif isinstance(utt, dict):
            messages.append({
                "role": utt.get("role", "user" if j % 2 == 0 else "assistant"),
                "content": utt.get("text", utt.get("content", "")),
            })

    personas = example.get("personas", example.get("persona", []))
    if personas and isinstance(personas[0], list):
        personas = personas[0]

    return {"messages": messages, "personas": personas or []}


def _load_dailydialog_fallback(max_samples: int = 2000) -> list[dict]:
    """加载 DailyDialog 作为 MSC 不可用时的公开回退数据集。

    DailyDialog 是完全公开的多轮对话数据集, 无需认证。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("请先安装 datasets: pip install datasets")

    logger.info("正在加载 DailyDialog (公开数据集, 无需认证)...")
    try:
        dataset = load_dataset("daily_dialog", split="train")
    except Exception:
        try:
            dataset = load_dataset("roskoN/dailydialog", split="train")
        except Exception as e2:
            logger.error(f"DailyDialog 也无法加载: {e2}")
            logger.error("请手动准备数据: --data_source jsonl --data_path <your_data.jsonl>")
            raise RuntimeError(
                "无法加载任何在线数据集。请使用 --data_source jsonl 并指定本地数据文件, "
                "或使用 --data_source synthetic 进行 debug。"
            )

    conversations = []
    for i, example in enumerate(dataset):
        if len(conversations) >= max_samples:
            break

        dialog = example.get("dialog", example.get("dialogue", []))
        if not dialog or len(dialog) < 4:
            continue

        messages = []
        for j, utt in enumerate(dialog):
            if isinstance(utt, str):
                content = utt.strip()
                if not content:
                    continue
                role = "user" if j % 2 == 0 else "assistant"
                messages.append({"role": role, "content": content})

        if len(messages) >= 4:
            conversations.append({"messages": messages, "personas": []})

    logger.info(f"DailyDialog 加载完成: {len(conversations)} 条对话")
    return conversations


def load_locomo_dataset(data_path: str, max_samples: int = 2000) -> list[dict]:
    """从本地 JSON 加载 LoCoMo 数据集。

    LoCoMo 格式:
    - 顶层是 list, 每个元素是一个长期对话
    - conversation 是 dict, 包含 speaker_a, speaker_b, session_N, session_N_date_time
    - 每个 session_N 是 turn 列表, 每个 turn 有 speaker, dia_id, text
    - 还有 qa (QA标注), observation, session_summary 等

    转换策略:
    - 将每个对话的所有 session 合并成一个超长对话
    - speaker_a -> user, speaker_b -> assistant
    - 这样 conversations_to_mag_samples 可以从中提取跨 session 的记忆
    """
    import json as _json
    from pathlib import Path as _Path
    import re as _re

    path = _Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo 数据文件不存在: {data_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = _json.load(f)

    if isinstance(raw, dict):
        raw = list(raw.values())

    conversations = []
    for conv in raw:
        conv_data = conv.get("conversation", {})
        if not isinstance(conv_data, dict):
            continue

        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")

        # 提取所有 session (session_1, session_2, ...) 并按编号排序
        session_keys = sorted(
            [k for k in conv_data.keys() if _re.match(r"^session_\d+$", k)],
            key=lambda k: int(k.split("_")[1])
        )

        if not session_keys:
            continue

        # 合并所有 session 的 turn 到一个消息列表
        all_messages = []
        for sk in session_keys:
            session_turns = conv_data[sk]
            if not isinstance(session_turns, list):
                continue
            # 可选: 在 session 间加入分隔符帮助模型区分时间
            date_key = f"{sk}_date_time"
            if date_key in conv_data and all_messages:
                # 加一条分隔, 模拟时间跳跃
                all_messages.append({
                    "role": "user",
                    "content": f"[新会话, 时间: {conv_data[date_key]}]"
                })
            for turn in session_turns:
                if not isinstance(turn, dict) or "text" not in turn:
                    continue
                speaker = turn.get("speaker", "")
                text = turn["text"].strip()
                if not text:
                    continue
                # speaker_a -> user, speaker_b -> assistant
                if speaker == speaker_a:
                    role = "user"
                elif speaker == speaker_b:
                    role = "assistant"
                else:
                    role = "user"  # fallback
                all_messages.append({"role": role, "content": text})

        if len(all_messages) < 6:
            continue

        # 提取 persona 信息 (从 observation 字段中提取)
        # observation 结构: {"session_N_observation": {"SpeakerA": [[text,id],...], ...}}
        personas = []
        obs_raw = conv.get("observation", {})
        if isinstance(obs_raw, dict):
            for obs_key, speaker_dict in obs_raw.items():
                if not isinstance(speaker_dict, dict):
                    continue
                for speaker_name, obs_list in speaker_dict.items():
                    if not isinstance(obs_list, list):
                        continue
                    for item in obs_list:
                        if isinstance(item, list) and len(item) >= 1:
                            text = item[0].strip() if isinstance(item[0], str) else str(item[0])
                            if text:
                                personas.append(text)
                        elif isinstance(item, str) and item.strip():
                            personas.append(item.strip())
                    if len(personas) >= 30:  # 限制总量
                        break

        conversations.append({"messages": all_messages, "personas": personas})

        if len(conversations) >= max_samples:
            break

    total_turns = sum(len(c["messages"]) for c in conversations)
    logger.info(f"LoCoMo 加载完成: {len(conversations)} 条对话, 共 {total_turns} 轮, "
                f"平均 {total_turns / max(len(conversations), 1):.0f} 轮/对话")
    return conversations


def load_jsonl_dataset(data_path: str, max_samples: int = 2000) -> list[dict]:
    """从 JSONL 文件加载数据。

    每行格式: {"messages": [{"role": ..., "content": ...}, ...], "personas": [...]}
    或: {"input_text": ..., "memory_texts": [...], "relevant_indices": [...]}
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {data_path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(_json.loads(line))
            if len(data) >= max_samples:
                break

    logger.info(f"JSONL 加载完成: {len(data)} 条记录")
    return data


def conversations_to_mag_samples(
    conversations: list[dict],
    num_memories_per_sample: int = 10,
    num_hard_negatives: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """将多轮对话转换为 MAG 训练样本。

    核心策略:
    - 对于每个对话, 取第 t 轮的 user 发言作为 query
    - 对应的 assistant 回复作为 target_text (Phase 2 的预测目标)
    - 前 t-1 轮的对话作为**相关记忆** (历史上下文)
    - 同 session 中其他发言作为**硬负例** (同域但不相关)
    - 其他对话的发言作为**随机负例** (跨域噪声)
    - Persona 条目 (如果有) 始终作为相关记忆

    重要: target_text 是 Phase 2 训练的关键!
    没有 target_text, Phase 2 退化为自回归预测 user 发言, backbone 本身就能做到,
    gate 没有动力打开。有了 target_text (assistant 回复), 模型必须利用记忆中的历史
    上下文才能生成合理回复, gate 就有动力学习何时以及如何注入记忆。

    Args:
        conversations: [{"messages": [...], "personas": [...]}, ...]
        num_memories_per_sample: 每个样本的候选记忆总数
        num_hard_negatives: 硬负例数量
        seed: 随机种子

    Returns:
        训练样本列表, 每个包含:
        - input_text: query (user 发言)
        - target_text: 对应的 assistant 回复 (用于 Phase 2 LM loss)
        - memory_texts: 候选记忆列表
        - relevant_indices: 相关记忆的索引
    """
    import random
    random.seed(seed)

    # 预先收集所有 user 发言, 用作跨对话随机负例
    all_user_utterances: list[str] = []
    for conv in conversations:
        for msg in conv["messages"]:
            if msg["role"] == "user" and len(msg["content"].strip()) >= 10:
                all_user_utterances.append(msg["content"].strip())

    if not all_user_utterances:
        logger.warning("真实数据中无有效 user 发言, 无法生成训练样本")
        return []

    samples: list[dict[str, Any]] = []

    for conv_idx, conv in enumerate(conversations):
        messages = conv["messages"]
        personas = conv.get("personas", [])

        # 收集该对话中所有 (user, assistant) 轮次对
        turn_pairs: list[tuple[str, str]] = []  # [(user_text, assistant_text), ...]
        user_turns: list[str] = []
        assistant_turns: list[str] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            content = msg["content"].strip()
            if msg["role"] == "user" and content and len(content) >= 5:
                user_text = content
                user_turns.append(user_text)
                # 找对应的 assistant 回复
                assistant_text = None
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    a_content = messages[i + 1]["content"].strip()
                    if a_content and len(a_content) >= 5:
                        assistant_text = a_content
                        assistant_turns.append(assistant_text)
                        i += 1  # 跳过 assistant 消息
                turn_pairs.append((user_text, assistant_text))
            elif msg["role"] == "assistant" and content and len(content) >= 5:
                assistant_turns.append(content)
            i += 1

        if len(turn_pairs) < 3:
            continue  # 至少需要 3 轮对话才有意义

        # 对每个 t >= 2 的轮构建一个训练样本
        for t in range(2, len(turn_pairs)):
            query = turn_pairs[t][0]
            target = turn_pairs[t][1]  # assistant 回复, 可能为 None

            # --- 相关记忆 ---
            # 前面的 user 发言 (历史上下文)
            relevant_memories = [tp[0] for tp in turn_pairs[:t]]
            # 加入 persona (如果有)
            if personas:
                for p in personas:
                    if isinstance(p, str) and p.strip():
                        relevant_memories.append(p.strip())

            # 限制相关记忆数量
            max_relevant = min(len(relevant_memories), num_memories_per_sample // 2)
            if max_relevant < len(relevant_memories):
                relevant_memories = random.sample(relevant_memories, max_relevant)

            # --- 硬负例 (同 session 的 assistant 回复) ---
            hard_negs = []
            if assistant_turns:
                # 排除当前轮的 assistant 回复 (如果有)
                candidates = [a for a in assistant_turns if a != target]
                if candidates:
                    n_hard = min(num_hard_negatives, len(candidates))
                    hard_negs = random.sample(candidates, n_hard)

            # --- 随机负例 (来自其他对话) ---
            remaining_slots = num_memories_per_sample - len(relevant_memories) - len(hard_negs)
            random_negs = []
            if remaining_slots > 0:
                # 过滤掉当前对话的发言
                other_utterances = [
                    u for u in all_user_utterances
                    if u not in user_turns and u != query
                ]
                if other_utterances:
                    n_random = min(remaining_slots, len(other_utterances))
                    random_negs = random.sample(other_utterances, n_random)

            # 组装记忆列表
            all_memories = relevant_memories + hard_negs + random_negs
            relevant_indices = list(range(len(relevant_memories)))

            # 打乱顺序
            indices = list(range(len(all_memories)))
            random.shuffle(indices)
            shuffled_memories = [all_memories[i] for i in indices]
            shuffled_relevant = [indices.index(r) for r in relevant_indices]

            sample = {
                "input_text": query,
                "memory_texts": shuffled_memories,
                "relevant_indices": shuffled_relevant,
            }
            # 如果有 assistant 回复, 加入 target_text 用于 Phase 2
            if target:
                sample["target_text"] = target

            samples.append(sample)

    # 全局打乱
    random.shuffle(samples)

    n_with_target = sum(1 for s in samples if "target_text" in s)
    logger.info(
        f"从 {len(conversations)} 条对话生成了 {len(samples)} 个 MAG 训练样本 "
        f"(平均每样本 {sum(len(s['memory_texts']) for s in samples) / max(len(samples), 1):.1f} 条记忆, "
        f"平均 {sum(len(s['relevant_indices']) for s in samples) / max(len(samples), 1):.1f} 条相关, "
        f"{n_with_target} 条含 target_text)"
    )
    return samples


def load_training_data(args: argparse.Namespace) -> list[dict[str, Any]]:
    """根据 data_source 参数加载训练数据。"""
    if args.data_source == "synthetic":
        logger.info("使用合成数据 (仅用于 debug/smoke test)")
        return generate_synthetic_data(
            tokenizer=None,  # 合成数据不需要 tokenizer
            num_samples=args.num_synthetic_samples,
            num_memories=args.num_memories_per_sample,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
        )

    elif args.data_source == "msc":
        conversations = load_msc_dataset(
            subset=args.msc_subset,
            max_samples=args.max_real_samples,
            data_path=args.data_path,
        )
        return conversations_to_mag_samples(
            conversations,
            num_memories_per_sample=args.num_memories_per_sample,
            num_hard_negatives=args.num_hard_negatives,
            seed=args.seed,
        )

    elif args.data_source == "locomo":
        if not args.data_path:
            raise ValueError("data_source=locomo 时需要指定 --data_path")
        conversations = load_locomo_dataset(
            data_path=args.data_path,
            max_samples=args.max_real_samples,
        )
        return conversations_to_mag_samples(
            conversations,
            num_memories_per_sample=args.num_memories_per_sample,
            num_hard_negatives=args.num_hard_negatives,
            seed=args.seed,
        )

    elif args.data_source == "jsonl":
        if not args.data_path:
            raise ValueError("data_source=jsonl 时需要指定 --data_path")
        data = load_jsonl_dataset(args.data_path, max_samples=args.max_real_samples)
        # 检测格式: 如果已经是 MAG 格式 (有 input_text + memory_texts), 直接用
        if data and "input_text" in data[0] and "memory_texts" in data[0]:
            logger.info(f"JSONL 数据已是 MAG 格式, 直接使用 {len(data)} 条样本")
            return data
        # 否则当作对话格式处理
        conversations = [
            d for d in data
            if "messages" in d and len(d.get("messages", [])) >= 4
        ]
        if not conversations:
            raise ValueError("JSONL 文件格式不正确: 需要 'messages' 字段或 'input_text'+'memory_texts' 字段")
        return conversations_to_mag_samples(
            conversations,
            num_memories_per_sample=args.num_memories_per_sample,
            num_hard_negatives=args.num_hard_negatives,
            seed=args.seed,
        )

    else:
        raise ValueError(f"未知 data_source: {args.data_source}")


# ======================================================================
# 合成数据生成 (仅用于 debug)
# ======================================================================

def generate_synthetic_data(
    tokenizer: Any,
    num_samples: int,
    num_memories: int,
    max_seq_len: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """生成合成训练数据。

    每个样本包含:
    - input_text: 一段对话文本 (query)
    - memory_texts: 一组候选记忆文本
    - relevant_indices: 与 query 相关的记忆索引 (ground truth)

    Returns:
        训练样本列表.
    """
    import random
    random.seed(seed)

    # 预定义话题和记忆模板
    topics = [
        ("NLP", "自然语言处理", ["BERT", "GPT", "Transformer", "预训练", "微调"]),
        ("CV", "计算机视觉", ["ResNet", "ViT", "目标检测", "图像分类", "分割"]),
        ("推荐系统", "推荐算法", ["协同过滤", "深度推荐", "CTR预测", "召回", "排序"]),
        ("强化学习", "RL", ["PPO", "DQN", "策略梯度", "奖励函数", "环境"]),
        ("数据库", "SQL", ["索引", "查询优化", "事务", "分布式", "缓存"]),
    ]

    memory_templates = [
        "用户正在研究{topic}领域的{keyword}",
        "用户的项目涉及{keyword}技术",
        "用户偏好使用{keyword}方法",
        "上次讨论了{topic}的{keyword}问题",
        "用户提到了{keyword}的最新进展",
    ]

    query_templates = [
        "关于{keyword}，你能告诉我最新进展吗？",
        "我之前说过关于{keyword}的什么？",
        "{keyword}的原理是什么？",
        "帮我回忆一下我们讨论过的{topic}",
        "我的研究方向是什么来着？",
    ]

    noise_memories = [
        "用户喜欢喝咖啡",
        "今天天气不错",
        "用户使用 macOS 系统",
        "上次对话是两天前",
        "用户说下周要开会",
        "用户提到了周末计划",
        "最近在看一本小说",
        "用户的时区是 UTC+8",
    ]

    samples: list[dict[str, Any]] = []

    for _ in range(num_samples):
        # 随机选择一个话题
        topic_name, topic_desc, keywords = random.choice(topics)
        keyword = random.choice(keywords)

        # 生成 query
        query = random.choice(query_templates).format(
            topic=topic_name, keyword=keyword
        )

        # 生成记忆列表 (部分相关, 部分不相关)
        num_relevant = random.randint(1, min(3, num_memories))
        memories: list[str] = []
        relevant: list[int] = []

        # 相关记忆
        for i in range(num_relevant):
            kw = keywords[i % len(keywords)]
            mem = random.choice(memory_templates).format(
                topic=topic_name, keyword=kw
            )
            memories.append(mem)
            relevant.append(i)

        # 不相关记忆 (噪声)
        for _ in range(num_memories - num_relevant):
            # 有一定概率来自其他话题
            if random.random() < 0.3:
                other_topic, _, other_kws = random.choice([t for t in topics if t[0] != topic_name])
                mem = random.choice(memory_templates).format(
                    topic=other_topic, keyword=random.choice(other_kws)
                )
            else:
                mem = random.choice(noise_memories)
            memories.append(mem)

        # 打乱记忆顺序
        indices = list(range(len(memories)))
        random.shuffle(indices)
        shuffled_memories = [memories[i] for i in indices]
        shuffled_relevant = [indices.index(r) for r in relevant]

        samples.append({
            "input_text": query,
            "memory_texts": shuffled_memories,
            "relevant_indices": shuffled_relevant,
        })

    logger.info(f"生成了 {len(samples)} 个合成训练样本")
    return samples


def _compute_log_interval(total_steps: int, max_logs_per_epoch: int = 20) -> int:
    """根据数据量自动计算合理的日志打印间隔。

    原则: 每个 epoch 最多打印 max_logs_per_epoch 条日志。
    """
    if total_steps <= max_logs_per_epoch:
        return 1
    return max(total_steps // max_logs_per_epoch, 1)


def train_phase1_scorer(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    train_data: list[dict[str, Any]],
    args: argparse.Namespace,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Phase 1: 预训练 Scorer (使用合成数据的 ground truth relevance)。

    DDP 模式下, selector 已被 DDP 包装, 每卡处理不同的数据子集,
    梯度通过 all-reduce 自动同步。
    """
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("Phase 1: 预训练 ContextSelector (Scorer)")
        logger.info("=" * 60)

    # 获取底层 module (DDP 包装后需要通过 .module 访问)
    selector_module = selector.module if isinstance(selector, DDP) else selector

    optimizer = optim.AdamW(
        selector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 构建 Dataset + DistributedSampler
    dataset = MAGTrainDataset(train_data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed) if world_size > 1 else None
    # 每卡看到的数据量
    steps_per_epoch = len(dataset) // world_size if world_size > 1 else len(dataset)

    # 学习率调度: warmup + cosine decay
    total_steps = steps_per_epoch * args.num_epochs // args.grad_accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps + 1,  # +1 避免越界
        pct_start=warmup_steps / max(total_steps, 1),
        anneal_strategy="cos",
    )

    # 日志间隔: 自动或手动
    log_interval = args.log_every if args.log_every > 0 else _compute_log_interval(steps_per_epoch)
    if is_main_process(rank):
        logger.info(f"数据量: {len(train_data)} (每卡约 {steps_per_epoch}), 每 {log_interval} 步打印一次")
        if world_size > 1:
            logger.info(f"DDP 模式: {world_size} 卡并行")

    selector.train()
    num_steps = 0
    best_loss = float("inf")
    patience = 0
    max_patience = 2  # 连续 2 个 epoch 不下降则 early stop

    for epoch in range(args.num_epochs):
        # DDP: 设置 epoch 使每轮打乱顺序不同
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        # 迭代数据: DDP 模式下用 sampler, 单卡直接遍历
        if sampler is not None:
            indices = list(sampler)
        else:
            indices = list(range(len(dataset)))

        for step_i, data_idx in enumerate(indices):
            sample = dataset[data_idx]

            # 编码 query (浅层 embedding 即可, 用于 selector 打分)
            query_emb = encoder.encode_texts([sample["input_text"]])  # (1, D)

            # 编码记忆 (深层编码: 过 backbone 前 N 层, 获得高质量语义向量)
            memory_embs = encoder.encode_texts_deep(sample["memory_texts"])  # (K, D)
            memory_embs = memory_embs.unsqueeze(0)  # (1, K, D)

            # 构建 target: 相关=1.0, 不相关=0.0
            K = len(sample["memory_texts"])
            target = torch.zeros(1, K, device=query_emb.device)
            for idx in sample["relevant_indices"]:
                target[0, idx] = 1.0

            # Forward (通过 DDP 包装的 selector)
            scores = selector(query_emb, memory_embs)  # (1, K)

            # 直接用 BCE loss: sigmoid(scores) vs target(0/1)
            loss = F.binary_cross_entropy_with_logits(scores, target)

            # 梯度累积
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            if (step_i + 1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                num_steps += 1

            epoch_loss += loss.item() * args.grad_accumulation_steps
            epoch_steps += 1

            if is_main_process(rank) and (step_i + 1) % log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - t_epoch_start
                eta = elapsed / (step_i + 1) * (len(indices) - step_i - 1)
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(
                    f"  [P1] Epoch {epoch+1} [{step_i+1}/{len(indices)}] "
                    f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                    f"ETA={eta:.0f}s"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

        # DDP: 跨卡同步 loss 以获得全局平均
        if world_size > 1:
            loss_tensor = torch.tensor(avg_epoch_loss, device=args.device)
            reduce_mean(loss_tensor, world_size)
            avg_epoch_loss = loss_tensor.item()

        epoch_time = time.time() - t_epoch_start
        if is_main_process(rank):
            logger.info(
                f"Epoch {epoch+1} 完成: avg_loss={avg_epoch_loss:.4f}  "
                f"耗时={epoch_time:.1f}s"
            )

        # Early stopping
        if avg_epoch_loss < best_loss - 1e-4:
            best_loss = avg_epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                if is_main_process(rank):
                    logger.info(f"Phase 1 Early Stop: 连续 {max_patience} 个 epoch 无改善")
                break

    if is_main_process(rank):
        logger.info(f"Phase 1 完成: total_steps={num_steps}, best_loss={best_loss:.4f}")


def _get_backbone_layers(model: nn.Module):
    """获取 backbone 的 transformer 层列表、final norm、lm_head 和 rotary_emb。"""
    # Qwen3 / LLaMA 结构: model.model.layers[i], model.model.norm, model.lm_head
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        rotary_emb = getattr(model.model, "rotary_emb", None)
        return model.model.layers, model.model.norm, model.lm_head, rotary_emb
    raise ValueError("不支持的模型结构，请检查 backbone 的层结构")


def _compute_position_embeddings(model: nn.Module, h: torch.Tensor, position_ids: torch.Tensor, rotary_emb=None):
    """计算 Qwen3/LLaMA 的 rotary position embeddings (cos, sin)。
    
    新版 HuggingFace transformers 中, DecoderLayer.forward 不再直接接受 position_ids,
    而是需要预计算好的 position_embeddings=(cos, sin) 元组。
    """
    if rotary_emb is not None:
        # 标准路径: 使用 model.model.rotary_emb
        position_embeddings = rotary_emb(h, position_ids)
        return position_embeddings  # (cos, sin)
    
    # Fallback: 尝试从 model.model 获取
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        position_embeddings = model.model.rotary_emb(h, position_ids)
        return position_embeddings
    
    # 最后 fallback: 返回 None, 让层自己处理 (旧版 transformers)
    return None


def train_phase2_joint(
    model: nn.Module,
    tokenizer: Any,
    encoder: MemoryEncoder,
    selector: ContextSelector,
    mag_gate: MAGGate,
    train_data: list[dict[str, Any]],
    args: argparse.Namespace,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Phase 2: 联合训练 Selector + MAGGate (端到端 LM loss)。

    核心策略: 分段 forward + 正确的梯度链路。
    DDP 模式下, mag_gate 已被 DDP 包装, 但 Phase 2 使用手动逐层 forward,
    因此 DDP 包装只用于自动梯度同步, 实际 inject 调用底层 module。
    
    梯度流路径:
      LM Loss → logits → lm_head → final_norm → backbone 后续层
               → MAG inject (gate params 有梯度)
                 → CrossAttn(h, memory_vecs, selection_weights)
                   → selection_weights ← selector.soft_select (selector params 有梯度)

    关键点:
    1. memory_embs 不需要梯度 (冻结的 backbone embedding 生成)
    2. selection_weights 需要梯度 (通过 selector 参数)
    3. MAGGate 的 inject 输出需要梯度 (通过 gate 参数)
    4. backbone 层冻结但仍参与前向传播 (只有 MAG/Selector 参数更新)
    """
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("Phase 2: 联合训练 Selector + MAGGate (端到端)")
        logger.info("=" * 60)

    # 获取底层 module (DDP 包装后)
    selector_module = selector.module if isinstance(selector, DDP) else selector
    mag_gate_module = mag_gate.module if isinstance(mag_gate, DDP) else mag_gate

    # 获取 backbone 内部结构
    backbone_layers, final_norm, lm_head, rotary_emb = _get_backbone_layers(model)
    num_backbone_layers = len(backbone_layers)

    # 确定注入层
    sorted_injection = sorted(mag_gate_module.injection_layers)
    first_injection_layer = sorted_injection[0] if sorted_injection else 0
    last_injection_layer = sorted_injection[-1] if sorted_injection else num_backbone_layers - 1
    if is_main_process(rank):
        logger.info(f"注入层: {sorted_injection}")
        logger.info(
            f"backbone 共 {num_backbone_layers} 层, "
            f"从第 {first_injection_layer} 层开始带梯度 forward"
        )

    # 只训练 selector + gate 的参数
    trainable_params = list(selector.parameters()) + list(mag_gate.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,  # 使用完整学习率
        weight_decay=args.weight_decay,
    )

    # 构建 Dataset + DistributedSampler
    dataset = MAGTrainDataset(train_data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed) if world_size > 1 else None
    steps_per_epoch = len(dataset) // world_size if world_size > 1 else len(dataset)

    # 学习率调度
    total_steps = steps_per_epoch * args.num_epochs // args.grad_accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps + 1,
        pct_start=warmup_steps / max(total_steps, 1),
        anneal_strategy="cos",
    )

    # 日志间隔
    log_interval = args.log_every if args.log_every > 0 else _compute_log_interval(steps_per_epoch)
    if is_main_process(rank):
        logger.info(f"数据量: {len(train_data)} (每卡约 {steps_per_epoch}), 每 {log_interval} 步打印一次")

    # ★ Anti-Teacher-Forcing: 解析参数
    use_detach_value = args.detach_value and not args.no_detach_value
    use_label_smoothing = args.label_smoothing > 0
    use_kl_constraint = args.kl_beta > 0
    use_scheduled_sampling = args.scheduled_sampling

    if is_main_process(rank):
        logger.info(f"★ Anti-Teacher-Forcing 配置:")
        logger.info(f"  Label Smoothing: {args.label_smoothing}")
        logger.info(f"  KL 约束: beta={args.kl_beta}, T={args.kl_temperature}")
        logger.info(f"  Detach V: {use_detach_value}")
        logger.info(f"  Scheduled Sampling: {use_scheduled_sampling} "
                    f"(start_epoch={args.ss_start_epoch}, max_ratio={args.ss_max_ratio})")

    selector.train()
    mag_gate.train()
    num_steps = 0
    prev_epoch_loss = None

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        # 迭代数据
        if sampler is not None:
            indices = list(sampler)
        else:
            indices = list(range(len(dataset)))

        for step_i, data_idx in enumerate(indices):
            sample = dataset[data_idx]

            # 编码 query (无梯度, backbone embedding 冻结)
            with torch.no_grad():
                query_emb = encoder.encode_texts([sample["input_text"]])  # (1, D)

                # 编码记忆 (深层编码, 无梯度)
                memory_embs = encoder.encode_texts_deep(sample["memory_texts"])  # (K, D)
                memory_embs = memory_embs.unsqueeze(0)  # (1, K, D)

            # ★ 关键: selection_weights 需要有梯度 (通过 selector 参数)
            selection_weights = selector_module.soft_select(
                query_emb.detach(), memory_embs.detach()
            )  # (1, K), 有梯度通过 selector 参数

            # ---- Tokenize: 构造 [query | target] 序列 ---- #
            # Phase 2 的关键改进: 如果有 target_text (assistant 回复), 则:
            #   input_ids = [query_tokens, target_tokens]
            #   labels    = [-100...-100, target_tokens]  (只在 target 部分计算 loss)
            # 这样 backbone 无法仅靠 query 预测 target, 必须利用 MAG 注入的记忆
            has_target = "target_text" in sample and sample["target_text"]

            if has_target:
                # 分别 tokenize query 和 target
                query_enc = tokenizer(
                    sample["input_text"],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=args.max_seq_len // 2,
                )
                target_enc = tokenizer(
                    sample["target_text"],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_seq_len // 2,
                )

                # 拼接 input_ids
                query_ids = query_enc["input_ids"]
                target_ids = target_enc["input_ids"]
                combined_ids = query_ids + target_ids
                # 截断到 max_seq_len
                if len(combined_ids) > args.max_seq_len:
                    combined_ids = combined_ids[:args.max_seq_len]
                    target_ids = combined_ids[len(query_ids):]

                input_ids = torch.tensor([combined_ids], device=args.device)

                # 构造 labels: query 部分为 -100, target 部分为原始 token id
                labels = torch.full_like(input_ids, -100)
                query_len = len(query_ids)
                labels[0, query_len:] = input_ids[0, query_len:]

                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                # fallback: 无 target_text, 退化为自回归
                encoded = tokenizer(
                    sample["input_text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq_len,
                )
                input_ids = encoded["input_ids"].to(args.device)
                labels = input_ids.clone()
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=args.device, dtype=torch.bool)
                else:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # ★ Scheduled Sampling: 训练后期逐步在 target 部分引入 token corruption
            # 原理: 将 target 部分的部分 token (在 input_ids 中) 随机替换为其他 token,
            #        但 labels 保持原始 ground truth 不变。这样即使 CrossAttention 能
            #        从 V 中读到答案, 它看到的 input context 已经被"污染"了一部分,
            #        模型必须学会在不完美输入下也能正确预测, 缓解 exposure bias。
            # 替换策略: 80% 随机词, 10% 保持原词, 10% mask token (类似 BERT)
            if use_scheduled_sampling and has_target and epoch >= args.ss_start_epoch:
                # 替换比例: 从 0 线性增长到 ss_max_ratio
                progress = (epoch - args.ss_start_epoch) / max(args.num_epochs - args.ss_start_epoch, 1)
                ss_ratio = min(args.ss_max_ratio * progress, args.ss_max_ratio)

                if ss_ratio > 0 and query_len < input_ids.shape[1]:
                    target_length = input_ids.shape[1] - query_len
                    num_corrupt = max(1, int(target_length * ss_ratio))
                    # 在 target 部分随机选择要替换的位置
                    corrupt_positions = torch.randperm(target_length)[:num_corrupt] + query_len
                    vocab_size = tokenizer.vocab_size or 32000
                    for pos in corrupt_positions:
                        r = torch.rand(1).item()
                        if r < 0.8:
                            # 80%: 替换为随机 token
                            input_ids[0, pos] = torch.randint(0, vocab_size, (1,), device=args.device)
                        elif r < 0.9:
                            # 10%: 保持原词不变 (do nothing)
                            pass
                        else:
                            # 10%: 替换为 pad token (类似 mask)
                            input_ids[0, pos] = tokenizer.pad_token_id or 0

            # ---- 分段 Forward ---- #
            # Step 1: backbone embedding + 注入层之前的层 (no_grad)
            with torch.no_grad():
                if hasattr(model.model, "embed_tokens"):
                    h = model.model.embed_tokens(input_ids)
                else:
                    h = model.get_input_embeddings()(input_ids)

                position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
                position_embeddings = _compute_position_embeddings(
                    model, h, position_ids, rotary_emb
                )

                layer_kwargs = {"attention_mask": attention_mask}
                if position_embeddings is not None:
                    layer_kwargs["position_embeddings"] = position_embeddings
                else:
                    layer_kwargs["position_ids"] = position_ids

                # Forward 到第一个注入层之前 (纯 no_grad)
                for layer_idx in range(first_injection_layer):
                    layer_output = backbone_layers[layer_idx](h, **layer_kwargs)
                    h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # Step 2: 从第一个注入层开始, 带梯度 forward
            h = h.detach()

            # position_embeddings 不需要梯度 (只是 cos/sin 位置编码)
            with torch.no_grad():
                position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
                position_embeddings = _compute_position_embeddings(
                    model, h, position_ids, rotary_emb
                )

            layer_kwargs_grad = {"attention_mask": attention_mask}
            if position_embeddings is not None:
                layer_kwargs_grad["position_embeddings"] = position_embeddings
            else:
                layer_kwargs_grad["position_ids"] = position_ids

            # 收集 gate 值用于 gate activation regularization
            gate_values_for_reg: list[torch.Tensor] = []

            # 保存 inject 前的 hidden states (用于 contrastive loss)
            h_before_inject = h.detach().clone()

            # 从第一个注入层到最后一层, 逐层 forward + 注入
            for layer_idx in range(first_injection_layer, num_backbone_layers):
                # Backbone layer forward (backbone 冻结, 但张量流有梯度)
                layer_output = backbone_layers[layer_idx](h, **layer_kwargs_grad)
                h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

                # ★ MAG 注入 (返回 gate 激活值用于 regularization)
                # detach_value=True: V 分支不回传梯度, 阻止 lm_loss 驱动"抄答案"
                if layer_idx in mag_gate_module.injection_layers:
                    h, gate_val = mag_gate_module.inject(
                        layer_idx, h, memory_embs.detach(),
                        selection_weights=selection_weights,
                        return_gate=True,
                        detach_value=use_detach_value,
                    )
                    if gate_val is not None:
                        gate_values_for_reg.append(gate_val)

            # Final norm + lm_head
            h = final_norm(h)
            logits = lm_head(h)

            # ★ 计算 LM loss (Label Smoothing 防止过度自信)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=args.label_smoothing,  # ★ Label Smoothing: 阻止 lm_loss → 0
            )

            # ★ 核心改进 1: KL 约束 (有记忆 vs 无记忆输出分布不能差太远)
            # 原理: 如同知识蒸馏, "有记忆的你"不能和"无记忆的你"差太远,
            #       强制记忆只起辅助作用而非主导生成
            # 公式: kl_loss = KL(softmax(logits_with_mem / T) || softmax(logits_no_mem / T))
            kl_loss = torch.tensor(0.0, device=lm_loss.device)
            gap_loss = torch.tensor(0.0, device=lm_loss.device)
            lm_loss_no_mem = torch.tensor(0.0, device=lm_loss.device)
            if has_target:
                with torch.no_grad():
                    # 无记忆 baseline: 从 inject 之前的 hidden state 开始
                    h_no_mem = h_before_inject.clone()
                    for layer_idx_c in range(first_injection_layer, num_backbone_layers):
                        layer_output_c = backbone_layers[layer_idx_c](h_no_mem, **layer_kwargs_grad)
                        h_no_mem = layer_output_c[0] if isinstance(layer_output_c, tuple) else layer_output_c
                    h_no_mem = final_norm(h_no_mem)
                    logits_no_mem = lm_head(h_no_mem)

                    # 计算无记忆 lm_loss (用于对比, 无梯度)
                    shift_logits_nm = logits_no_mem[..., :-1, :].contiguous()
                    lm_loss_no_mem = F.cross_entropy(
                        shift_logits_nm.view(-1, shift_logits_nm.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                # DPO 风格 Gap Loss (保留, 鼓励记忆有帮助)
                beta = 2.0
                gap = lm_loss_no_mem.detach() - lm_loss  # 正值 = 好
                gap_loss = -F.logsigmoid(beta * gap)

                # ★ KL 约束: 限制记忆注入对输出分布的影响幅度
                if use_kl_constraint:
                    T = args.kl_temperature
                    # 只在 target 部分 (有效 label 位置) 计算 KL
                    valid_mask = (shift_labels.view(-1) != -100)
                    if valid_mask.any():
                        # logits_with_mem (有梯度) vs logits_no_mem (无梯度, 作为 target 分布)
                        log_p_with = F.log_softmax(shift_logits.view(-1, shift_logits.size(-1))[valid_mask] / T, dim=-1)
                        p_without = F.softmax(shift_logits_nm.view(-1, shift_logits_nm.size(-1))[valid_mask] / T, dim=-1)
                        # KL(P_with_mem || P_no_mem): 有记忆输出不能偏离无记忆输出太远
                        kl_loss = F.kl_div(log_p_with, p_without.detach(), reduction="batchmean") * (T * T)

            # ★ 辅助 loss: 鼓励 selector 给相关记忆更高权重
            aux_loss = torch.tensor(0.0, device=lm_loss.device)
            if "relevant_indices" in sample and sample["relevant_indices"]:
                K = len(sample["memory_texts"])
                sel_target = torch.zeros(1, K, device=lm_loss.device)
                for idx in sample["relevant_indices"]:
                    if idx < K:
                        sel_target[0, idx] = 1.0
                raw_scores = selector_module(query_emb.detach(), memory_embs.detach())
                aux_loss = F.binary_cross_entropy_with_logits(raw_scores, sel_target)

            # ★ 辅助统计: gate 激活值 (仅用于日志, 不参与 loss)
            gate_reg_loss = torch.tensor(0.0, device=lm_loss.device)  # 保留变量, 不参与 loss
            gate_mean_val = 0.0
            if gate_values_for_reg:
                all_gate_means = [g.mean() for g in gate_values_for_reg]
                avg_gate = torch.stack(all_gate_means).mean()
                gate_mean_val = avg_gate.item()

            # ★ 总 loss 组合:
            # gap_loss:  DPO 风格, 鼓励记忆注入降低 lm_loss
            # lm_loss:   直接最小化 (带 Label Smoothing, 防止趋于 0)
            # kl_loss:   KL 约束, 记忆只能辅助不能主导
            # aux_loss:  selector 打分辅助
            loss = 1.0 * gap_loss + 0.3 * lm_loss + args.kl_beta * kl_loss + 0.1 * aux_loss

            # nan guard
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(rank) and step_i < 10:
                    logger.warning(
                        f"  [P2] Step {step_i}: loss={loss.item():.4f} 跳过 (nan/inf), "
                        f"lm={lm_loss.item():.4f} gap={gap_loss.item():.4f} "
                        f"aux={aux_loss.item():.4f} g_reg={gate_reg_loss.item():.4f}"
                    )
                optimizer.zero_grad()
                continue

            # 梯度累积
            loss = loss / args.grad_accumulation_steps
            loss.backward()

            if (step_i + 1) % args.grad_accumulation_steps == 0:
                # DDP 模式下手动同步梯度 (因为没走标准 DDP forward)
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
            epoch_steps += 1

            if is_main_process(rank) and (step_i + 1) % log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - t_epoch_start
                eta = elapsed / (step_i + 1) * (len(indices) - step_i - 1)
                lr_now = optimizer.param_groups[0]['lr']

                # 计算 gate 统计信息 (诊断用)
                gate_info = ""
                try:
                    if mag_gate_module.config.share_parameters and mag_gate_module._shared_block is not None:
                        gate_bias = mag_gate_module._shared_block.gate_proj.bias.data.mean().item()
                        gate_info = f"  gate_mean={gate_mean_val:.4f} bias={gate_bias:.3f}"
                except Exception:
                    gate_info = f"  gate_mean={gate_mean_val:.4f}"

                logger.info(
                    f"  [P2] Epoch {epoch+1} [{step_i+1}/{len(indices)}] "
                    f"loss={avg_loss:.4f} (lm={lm_loss.item():.4f} gap={gap_loss.item():.4f} "
                    f"kl={kl_loss.item():.4f} aux={aux_loss.item():.4f} "
                    f"lm_nm={lm_loss_no_mem.item():.4f} Δ={lm_loss_no_mem.item()-lm_loss.item():.4f})"
                    f"  lr={lr_now:.2e}{gate_info}  ETA={eta:.0f}s"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

        # DDP: 跨卡同步 loss
        if world_size > 1:
            loss_tensor = torch.tensor(avg_epoch_loss, device=args.device)
            reduce_mean(loss_tensor, world_size)
            avg_epoch_loss = loss_tensor.item()

        epoch_time = time.time() - t_epoch_start

        # 计算 loss 变化
        delta_str = ""
        if prev_epoch_loss is not None:
            delta = avg_epoch_loss - prev_epoch_loss
            delta_str = f"  Δ={delta:+.4f}"
        prev_epoch_loss = avg_epoch_loss

        if is_main_process(rank):
            logger.info(
                f"Epoch {epoch+1} 完成: avg_loss={avg_epoch_loss:.4f}{delta_str}  "
                f"耗时={epoch_time:.1f}s"
            )

    if is_main_process(rank):
        logger.info(f"Phase 2 完成: total_steps={num_steps}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ====== 分布式初始化 ======
    rank, local_rank, world_size = setup_distributed()

    # DDP 模式下, device 绑定到 local_rank 对应的 GPU
    if world_size > 1:
        args.device = f"cuda:{local_rank}"
    elif args.device == "cuda" and torch.cuda.is_available():
        args.device = "cuda:0"

    # 非主进程降低日志级别
    if not is_main_process(rank):
        logging.getLogger("train_mag").setLevel(logging.WARNING)
        logging.getLogger("src.memory.mag.memory_encoder").setLevel(logging.WARNING)
        logging.getLogger("src.memory.mag.context_selector").setLevel(logging.WARNING)
        logging.getLogger("src.memory.mag.mag_gate").setLevel(logging.WARNING)

    # 创建输出目录 (只 rank 0)
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    dist_barrier()  # 等待 rank 0 创建目录

    # 1. 加载 backbone
    # ⚠️ CephFS I/O 瓶颈: 多节点同时读取大模型权重会导致严重 I/O 竞争和 NCCL 超时
    # 解决: 节点级串行加载 — 每个节点仅 local_rank==0 先加载, 其他进程等待后从 Page Cache 读取
    if world_size > 1:
        if local_rank == 0:
            logger.info(f"[节点串行加载] local_rank=0 开始加载 backbone (world_size={world_size})...")
            model, tokenizer, hidden_dim, num_layers = load_backbone(
                args.model_path, args.device, args.dtype,
                sliding_window=args.sliding_window,
            )
            logger.info(f"[节点串行加载] local_rank=0 加载完成, 通知同节点其他进程")
        # 同节点 barrier: 等待 local_rank==0 加载完成, 权重文件进入 OS Page Cache
        dist.barrier()
        if local_rank != 0:
            logger.info(f"[节点串行加载] local_rank={local_rank} 从 Page Cache 加载 backbone...")
            model, tokenizer, hidden_dim, num_layers = load_backbone(
                args.model_path, args.device, args.dtype,
                sliding_window=args.sliding_window,
            )
        dist.barrier()  # 所有节点加载完成
    else:
        if is_main_process(rank):
            logger.info(f"加载 backbone (单卡模式)...")
        model, tokenizer, hidden_dim, num_layers = load_backbone(
            args.model_path, args.device, args.dtype,
            sliding_window=args.sliding_window,
        )

    # 2. 初始化 MemoryEncoder (共享 backbone embedding, 不训练)
    enc_cfg = MemoryEncoderConfig(
        max_memory_tokens=args.max_memory_tokens,
        pooling="mean",
        deep_encode_layers=args.deep_encode_layers,
    )
    encoder = MemoryEncoder(enc_cfg)
    encoder.set_backbone(
        backbone_model=model,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        device=args.device,
        dtype=getattr(torch, args.dtype, torch.float32),
    )

    # 3. 初始化 ContextSelector
    sel_cfg = ContextSelectorConfig(
        input_dim=hidden_dim,
        hidden_dim=args.selector_hidden_dim,
        top_k=args.selector_top_k,
    )
    selector = ContextSelector(sel_cfg).to(args.device)

    # 4. 初始化 MAGGate
    injection_layers = args.mag_injection_layers
    if injection_layers is None:
        # 默认: 均匀选择 4 个层
        injection_layers = [
            num_layers // 4,
            num_layers // 2,
            num_layers * 3 // 4,
            num_layers - 1,
        ]

    gate_cfg = MAGGateConfig(
        hidden_dim=hidden_dim,
        num_heads=args.mag_num_heads,
        memory_dim=hidden_dim,
        injection_layers=injection_layers,
        share_parameters=args.mag_share_parameters,
        gate_init_bias=args.mag_gate_init_bias,
    )
    mag_gate = MAGGate(gate_cfg).to(args.device)

    # ====== DDP 包装可训练模块 ======
    # 注意: backbone 冻结不做 DDP, 只包装 selector 和 mag_gate
    if world_size > 1:
        # Phase 1 对 selector 做 DDP (标准 forward 调用)
        selector = DDP(
            selector,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        # Phase 2 对 mag_gate 做 DDP
        # 注意: Phase 2 使用手动逐层 forward + inject,
        # DDP 包装主要用于保持参数同步 (初始化时 broadcast),
        # 梯度同步在 Phase 2 中通过手动 all_reduce 完成
        mag_gate = DDP(
            mag_gate,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # inject 不走标准 forward
        )
        if is_main_process(rank):
            logger.info(f"DDP 包装完成: selector + mag_gate (world_size={world_size})")

    # 打印参数量 (只 rank 0)
    selector_module = selector.module if isinstance(selector, DDP) else selector
    mag_gate_module = mag_gate.module if isinstance(mag_gate, DDP) else mag_gate
    total_trainable = sum(p.numel() for p in selector_module.parameters()) + \
                      sum(p.numel() for p in mag_gate_module.parameters())
    if is_main_process(rank):
        logger.info(f"可训练参数量: {total_trainable:,}")
        logger.info(f"  Selector: {sum(p.numel() for p in selector_module.parameters()):,}")
        logger.info(f"  MAGGate: {sum(p.numel() for p in mag_gate_module.parameters()):,}")
        logger.info(f"注入层: {injection_layers}")

    # 5. 加载训练数据 (所有进程都加载全量数据, DistributedSampler 负责分片)
    train_data = load_training_data(args)
    if not train_data:
        if is_main_process(rank):
            logger.error("训练数据为空, 请检查数据源配置")
        cleanup_distributed()
        return

    if is_main_process(rank):
        logger.info(f"训练数据: {len(train_data)} 条, world_size={world_size}, "
                    f"每卡约 {len(train_data) // max(world_size, 1)} 条/epoch")

    t0 = time.time()

    # Phase 1: 预训练 Scorer
    train_phase1_scorer(
        model, tokenizer, encoder, selector, train_data, args,
        rank=rank, world_size=world_size,
    )
    dist_barrier()

    # Phase 2: 联合训练 Selector + Gate
    train_phase2_joint(
        model, tokenizer, encoder, selector, mag_gate, train_data, args,
        rank=rank, world_size=world_size,
    )
    dist_barrier()

    total_time = time.time() - t0

    # 6. 保存 (只 rank 0)
    if is_main_process(rank):
        save_path = Path(args.output_dir)

        # 保存底层 module 的 state_dict (去掉 DDP 的 "module." 前缀)
        torch.save(selector_module.state_dict(), save_path / "context_selector.pt")
        torch.save(mag_gate_module.state_dict(), save_path / "mag_gate.pt")

        # 保存配置
        config_info = {
            "model_path": args.model_path,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "injection_layers": injection_layers,
            "mag_num_heads": args.mag_num_heads,
            "mag_share_parameters": args.mag_share_parameters,
            "mag_gate_init_bias": args.mag_gate_init_bias,
            "selector_hidden_dim": args.selector_hidden_dim,
            "selector_top_k": args.selector_top_k,
            "max_memory_tokens": args.max_memory_tokens,
            "total_trainable_params": total_trainable,
            "training_time_s": total_time,
            "world_size": world_size,
        }
        with open(save_path / "mag_config.json", "w") as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)

        logger.info(f"训练完成! 总耗时: {total_time:.1f}s")
        logger.info(f"权重已保存到: {args.output_dir}")
        logger.info(f"  context_selector.pt: {(save_path / 'context_selector.pt').stat().st_size / 1024:.1f} KB")
        logger.info(f"  mag_gate.pt: {(save_path / 'mag_gate.pt').stat().st_size / 1024:.1f} KB")

    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    main()
