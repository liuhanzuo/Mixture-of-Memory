"""
L1 门控网络训练器。

训练目标: 学习最优门控参数，使得 L1 记忆读取结果与隐藏状态的融合
能最小化下游 language modeling loss。

训练策略:
1. 冻结 backbone 参数
2. 冻结 L1 投影层 (可选，也可联合训练)
3. 只训练门控网络参数 (gate_proj, output_proj, learned_gate)

数据管线:
- 方式 A (推荐): 先用 backbone 对训练语料做 forward pass，缓存隐藏状态到磁盘
- 方式 B (在线):  每个 batch 实时跑 backbone forward，提取隐藏状态
- 方式 C (合成):  随机张量，仅用于调试

损失函数:
- 主损失: 门控融合后的隐藏状态经 LM head 的 cross-entropy loss
- 辅助损失 (可选): 门控稀疏性正则化，鼓励门控信号稀疏
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.memory.l1.assoc_memory import AssociativeMemoryL1, L1Config
from src.memory.l1.gating import L1Gate

logger = logging.getLogger(__name__)


# ======================================================================
# 训练配置
# ======================================================================

@dataclass
class GateTrainConfig:
    """门控训练配置。"""

    # 数据
    data_mode: str = "backbone"  # "backbone" | "cached" | "synthetic"
    data_path: str = ""          # 训练语料路径 (jsonl/txt)
    cache_dir: str = "data/cache/gate_hidden_states"  # 缓存目录
    max_seq_len: int = 512       # 最大序列长度
    num_samples: int = 1000      # 合成模式的样本数

    # 优化器
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    warmup_steps: int = 100

    # 训练
    batch_size: int = 8
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 正则化
    gate_sparsity_weight: float = 0.01  # 门控稀疏性正则化权重

    # 联合训练 L1 投影层
    train_projections: bool = False

    # backbone 隐藏层选取 (从哪一层提取隐藏状态)
    hidden_layer_index: int = -1  # -1 表示最后一层

    # 保存
    save_dir: str = "outputs/runs/gate_training"
    save_every: int = 200

    # 日志
    log_every: int = 10


# ======================================================================
# 数据集
# ======================================================================

class SyntheticGateDataset(Dataset):
    """合成门控训练数据集 (仅用于 debug/smoke test)。

    生成随机隐藏状态序列。实际训练应使用 BackboneHiddenStateDataset
    或 CachedHiddenStateDataset。
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 128,
        hidden_dim: int = 256,
        vocab_size: int = 1000,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hidden_states = torch.randn(self.seq_len, self.hidden_dim)
        target_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            "hidden_states": hidden_states,
            "target_ids": target_ids,
        }


class CachedHiddenStateDataset(Dataset):
    """从磁盘加载预缓存的隐藏状态。

    文件格式: 每个样本为一个 .pt 文件，包含:
        {"hidden_states": Tensor(seq_len, hidden_dim), "target_ids": Tensor(seq_len,)}
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"缓存目录不存在: {self.cache_dir}")
        self.files = sorted(self.cache_dir.glob("sample_*.pt"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"缓存目录中无 sample_*.pt 文件: {self.cache_dir}")
        logger.info(f"[CachedHiddenStateDataset] 加载 {len(self.files)} 个缓存样本")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        return {
            "hidden_states": data["hidden_states"],
            "target_ids": data["target_ids"],
        }


class TextFileDataset(Dataset):
    """从文本/jsonl 文件加载原始文本，用于在线 backbone forward。

    支持格式:
    - .jsonl: 每行一个 JSON，取 "text" 字段
    - .txt:   每行一段文本
    """

    def __init__(self, data_path: str | Path, max_samples: int | None = None):
        self.data_path = Path(data_path)
        self.texts: list[str] = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        if self.data_path.suffix == ".jsonl":
            with open(self.data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    obj = json.loads(line.strip())
                    text = obj.get("text", obj.get("content", ""))
                    if text.strip():
                        self.texts.append(text)
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    if line.strip():
                        self.texts.append(line.strip())

        logger.info(f"[TextFileDataset] 加载 {len(self.texts)} 条文本")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


# ======================================================================
# 隐藏状态缓存工具
# ======================================================================

def cache_hidden_states_from_backbone(
    backbone: Any,
    data_path: str | Path,
    cache_dir: str | Path,
    max_seq_len: int = 512,
    max_samples: int = 1000,
    batch_size: int = 4,
    hidden_layer_index: int = -1,
) -> int:
    """用冻结 backbone 对训练语料做 forward pass，缓存隐藏状态到磁盘。

    Args:
        backbone: BackboneModel 实例 (SWABackbone 或 FullAttentionBackbone)
        data_path: 训练语料文件路径
        cache_dir: 缓存输出目录
        max_seq_len: 最大序列长度
        max_samples: 最大缓存样本数
        batch_size: batch 大小
        hidden_layer_index: 提取哪一层的隐藏状态 (-1 = 最后一层)

    Returns:
        缓存的样本数
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = backbone.get_tokenizer()
    if tokenizer is None:
        raise ValueError(
            "Backbone 没有 tokenizer，无法缓存隐藏状态。"
            "请使用非 debug 模式的 backbone。"
        )

    # 加载文本数据
    text_dataset = TextFileDataset(data_path, max_samples=max_samples)

    device = backbone.get_device()
    hf_model = backbone.get_hf_model()

    # 冻结 backbone
    hf_model.eval()
    for param in hf_model.parameters():
        param.requires_grad = False

    sample_idx = 0
    total = min(len(text_dataset), max_samples)

    logger.info(
        f"[CacheHiddenStates] 开始缓存: {total} 样本, "
        f"seq_len={max_seq_len}, layer={hidden_layer_index}"
    )

    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="缓存隐藏状态"):
            batch_texts = [text_dataset[j] for j in range(i, min(i + batch_size, total))]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # 提取指定层隐藏状态
            if outputs.hidden_states:
                hidden = outputs.hidden_states[hidden_layer_index]
            else:
                hidden = outputs.last_hidden_state

            # 构造 target_ids (shifted input_ids)
            target_ids = input_ids.clone()
            target_ids[:, :-1] = input_ids[:, 1:]
            target_ids[:, -1] = -100  # 最后一个位置忽略

            # 逐样本保存到磁盘
            for b in range(hidden.shape[0]):
                # 只保留非 padding 部分
                mask = attention_mask[b].bool()
                h = hidden[b][mask].cpu()
                t = target_ids[b][mask].cpu()

                save_path = cache_dir / f"sample_{sample_idx:06d}.pt"
                torch.save({"hidden_states": h, "target_ids": t}, save_path)
                sample_idx += 1

    logger.info(f"[CacheHiddenStates] 缓存完成: {sample_idx} 样本 -> {cache_dir}")
    return sample_idx


# ======================================================================
# L1 门控训练器
# ======================================================================

class L1GateTrainer:
    """L1 门控网络训练器。

    训练流程:
    1. 初始化 L1 记忆模块 (含门控)
    2. 获取 LM head (复用 backbone 或创建独立)
    3. 冻结非训练参数
    4. 迭代训练门控参数

    支持三种数据模式:
    - "backbone": 在线从 backbone 提取隐藏状态 (需要 GPU 显存较大)
    - "cached":   从磁盘加载预缓存的隐藏状态 (推荐，需先运行缓存)
    - "synthetic": 随机张量 (仅用于 debug)
    """

    def __init__(
        self,
        l1_config: L1Config,
        train_config: GateTrainConfig,
        hidden_dim: int = 256,
        vocab_size: int = 1000,
        backbone: Any = None,
    ):
        self.l1_config = l1_config
        self.train_config = train_config
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.backbone = backbone

        # 初始化 L1 模块
        self.l1 = AssociativeMemoryL1(l1_config)
        self.l1.set_hidden_dim(hidden_dim)

        # LM head: 优先复用 backbone 的 LM head
        self.lm_head: nn.Module | None = None
        self._lm_head_is_shared = False
        self._init_lm_head()

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_lm_head(self) -> None:
        """初始化 LM head，优先复用 backbone 的权重。"""
        if self.backbone is not None and not self.backbone.is_debug():
            try:
                hf_model = self.backbone.get_hf_model()
                # 尝试获取 HF 模型的 lm_head
                if hasattr(hf_model, "lm_head"):
                    self.lm_head = hf_model.lm_head
                    self._lm_head_is_shared = True
                    # 冻结共享的 LM head
                    for param in self.lm_head.parameters():
                        param.requires_grad = False
                    logger.info(
                        "[GateTrainer] 复用 backbone LM head "
                        f"(frozen, out_features={self.lm_head.out_features})"
                    )
                    self.vocab_size = self.lm_head.out_features
                    return
            except Exception as e:
                logger.warning(f"[GateTrainer] 复用 backbone LM head 失败: {e}")

        # Fallback: 创建独立的 LM head (随机初始化)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self._lm_head_is_shared = False
        logger.info(
            f"[GateTrainer] 创建独立 LM head (hidden_dim={self.hidden_dim}, "
            f"vocab_size={self.vocab_size})"
        )

    def _get_trainable_params(self) -> list[nn.Parameter]:
        """获取需要训练的参数。"""
        params: list[nn.Parameter] = []

        # 门控参数 (核心训练目标)
        if self.l1.gate is not None:
            params.extend(self.l1.gate.parameters())

        # 可选: L1 投影层参数
        if self.train_config.train_projections:
            if self.l1.proj_k is not None:
                params.extend(self.l1.proj_k.parameters())
            if self.l1.proj_v is not None:
                params.extend(self.l1.proj_v.parameters())
            if self.l1.proj_q is not None:
                params.extend(self.l1.proj_q.parameters())

        # 仅当 LM head 非共享时才训练
        if not self._lm_head_is_shared and self.lm_head is not None:
            params.extend(self.lm_head.parameters())

        return params

    def _build_dataset(self) -> Dataset:
        """根据配置构建训练数据集。"""
        mode = self.train_config.data_mode

        if mode == "cached":
            cache_dir = self.train_config.cache_dir
            if not Path(cache_dir).exists() or not list(Path(cache_dir).glob("sample_*.pt")):
                # 缓存不存在，尝试从 backbone 生成
                if self.backbone is not None and self.train_config.data_path:
                    logger.info("[GateTrainer] 缓存不存在，正在从 backbone 生成...")
                    cache_hidden_states_from_backbone(
                        backbone=self.backbone,
                        data_path=self.train_config.data_path,
                        cache_dir=cache_dir,
                        max_seq_len=self.train_config.max_seq_len,
                        max_samples=self.train_config.num_samples,
                        hidden_layer_index=self.train_config.hidden_layer_index,
                    )
                else:
                    raise FileNotFoundError(
                        f"缓存目录 {cache_dir} 不存在且无法自动生成。"
                        "请先运行缓存脚本或提供 backbone + data_path。"
                    )
            return CachedHiddenStateDataset(cache_dir)

        elif mode == "backbone":
            if self.backbone is None:
                raise ValueError(
                    "data_mode='backbone' 需要提供 backbone 实例。"
                    "请在构造 L1GateTrainer 时传入 backbone 参数。"
                )
            return self._build_online_dataset()

        else:  # synthetic
            logger.warning("[GateTrainer] 使用合成数据集训练 (仅用于 debug)。")
            return SyntheticGateDataset(
                num_samples=self.train_config.num_samples,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
            )

    def _build_online_dataset(self) -> Dataset:
        """在线从 backbone 提取隐藏状态构建数据集。

        将所有隐藏状态缓存到内存中（适合中小规模数据）。
        大规模数据请用 cache_hidden_states_from_backbone 缓存到磁盘。
        """
        tokenizer = self.backbone.get_tokenizer()
        if tokenizer is None:
            raise ValueError("Backbone 没有 tokenizer，请使用非 debug 模式。")

        data_path = self.train_config.data_path
        if not data_path or not Path(data_path).exists():
            logger.warning(
                f"[GateTrainer] 数据路径 '{data_path}' 不存在，"
                "回退到合成数据集。"
            )
            return SyntheticGateDataset(
                num_samples=self.train_config.num_samples,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
            )

        text_dataset = TextFileDataset(data_path, max_samples=self.train_config.num_samples)
        device = self.backbone.get_device()
        hf_model = self.backbone.get_hf_model()
        hf_model.eval()

        samples: list[dict[str, torch.Tensor]] = []
        max_seq_len = self.train_config.max_seq_len
        layer_idx = self.train_config.hidden_layer_index
        batch_size = min(self.train_config.batch_size, 4)  # 控制显存

        logger.info(
            f"[GateTrainer] 在线提取隐藏状态: {len(text_dataset)} 文本, "
            f"seq_len={max_seq_len}, layer={layer_idx}"
        )

        with torch.no_grad():
            for i in tqdm(
                range(0, len(text_dataset), batch_size),
                desc="提取隐藏状态",
            ):
                batch_texts = [
                    text_dataset[j]
                    for j in range(i, min(i + batch_size, len(text_dataset)))
                ]
                encodings = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                outputs = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                if outputs.hidden_states:
                    hidden = outputs.hidden_states[layer_idx]
                else:
                    hidden = outputs.last_hidden_state

                target_ids = input_ids.clone()
                target_ids[:, :-1] = input_ids[:, 1:]
                target_ids[:, -1] = -100

                for b in range(hidden.shape[0]):
                    mask = attention_mask[b].bool()
                    samples.append({
                        "hidden_states": hidden[b][mask].cpu(),
                        "target_ids": target_ids[b][mask].cpu(),
                    })

        logger.info(f"[GateTrainer] 提取完成: {len(samples)} 样本")
        return _InMemoryDataset(samples)

    def _compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """计算训练损失。

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            target_ids: (batch, seq_len)

        Returns:
            total_loss, metrics_dict
        """
        # L1 前向: 写入 + 读取 + 门控融合
        gated_output = self.l1(hidden_states, update=True)

        # LM head
        assert self.lm_head is not None
        logits = self.lm_head(gated_output)  # (B, S, vocab)

        # Cross-entropy loss
        ce_loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_ids.view(-1),
            ignore_index=-100,
        )

        # 门控稀疏性正则化
        sparsity_loss = torch.tensor(0.0, device=self.device)
        if self.l1.gate is not None and self.train_config.gate_sparsity_weight > 0:
            gate_vals = self.l1.gate._compute_gate(hidden_states)
            if isinstance(gate_vals, torch.Tensor):
                # 鼓励门控值接近 0 或 1 (L1 正则)
                sparsity_loss = gate_vals.mean()

        total_loss = ce_loss + self.train_config.gate_sparsity_weight * sparsity_loss

        metrics = {
            "ce_loss": ce_loss.item(),
            "sparsity_loss": sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    def _collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """将变长样本 pad 到同一长度。"""
        max_len = max(b["hidden_states"].shape[0] for b in batch)
        hidden_dim = batch[0]["hidden_states"].shape[-1]

        padded_hidden = torch.zeros(len(batch), max_len, hidden_dim)
        padded_targets = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, b in enumerate(batch):
            seq_len = b["hidden_states"].shape[0]
            padded_hidden[i, :seq_len] = b["hidden_states"]
            padded_targets[i, :seq_len] = b["target_ids"]

        return {
            "hidden_states": padded_hidden,
            "target_ids": padded_targets,
        }

    def train(
        self,
        dataset: Dataset | None = None,
    ) -> dict[str, list[float]]:
        """执行门控训练。

        Args:
            dataset: 训练数据集。若为 None，根据配置自动构建。

        Returns:
            训练指标历史记录。
        """
        if dataset is None:
            dataset = self._build_dataset()

        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

        # 移至设备
        self.l1.to(self.device)
        if self.lm_head is not None:
            self.lm_head.to(self.device)

        # 优化器
        trainable_params = self._get_trainable_params()
        if not trainable_params:
            logger.error("[GateTrainer] 无可训练参数！请检查 L1 配置。")
            return {"ce_loss": [], "sparsity_loss": [], "total_loss": []}

        optimizer = optim.AdamW(
            trainable_params,
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )

        # 学习率预热调度器
        total_steps = self.train_config.max_steps
        warmup_steps = self.train_config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 训练循环
        history: dict[str, list[float]] = {
            "ce_loss": [],
            "sparsity_loss": [],
            "total_loss": [],
        }

        self.l1.train()
        if self.lm_head is not None and not self._lm_head_is_shared:
            self.lm_head.train()

        global_step = 0
        save_dir = Path(self.train_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n_params = sum(p.numel() for p in trainable_params)
        logger.info(f"[GateTrainer] 开始训练: {total_steps} 步")
        logger.info(f"[GateTrainer] 可训练参数数: {n_params:,}")
        logger.info(f"[GateTrainer] 数据模式: {self.train_config.data_mode}")
        logger.info(f"[GateTrainer] LM head 共享: {self._lm_head_is_shared}")
        logger.info(f"[GateTrainer] 设备: {self.device}")

        pbar = tqdm(total=total_steps, desc="训练门控")
        while global_step < total_steps:
            for batch in dataloader:
                if global_step >= total_steps:
                    break

                hidden_states = batch["hidden_states"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                # 每个 batch 开始前重置 L1 记忆
                self.l1.reset()

                try:
                    loss, metrics = self._compute_loss(hidden_states, target_ids)
                    loss = loss / self.train_config.grad_accumulation_steps
                    loss.backward()
                except RuntimeError as e:
                    logger.warning(f"[GateTrainer] step={global_step} 训练错误: {e}")
                    optimizer.zero_grad()
                    global_step += 1
                    pbar.update(1)
                    continue

                if (global_step + 1) % self.train_config.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        self.train_config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # 记录
                for k, v in metrics.items():
                    history[k].append(v)

                if global_step % self.train_config.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        f"[GateTrainer] step={global_step} "
                        f"ce_loss={metrics['ce_loss']:.4f} "
                        f"sparsity={metrics['sparsity_loss']:.4f} "
                        f"total={metrics['total_loss']:.4f} "
                        f"lr={lr_now:.2e}"
                    )

                # 保存检查点
                if (global_step + 1) % self.train_config.save_every == 0:
                    ckpt_path = save_dir / f"gate_step{global_step + 1}.pt"
                    self._save_checkpoint(ckpt_path, global_step)

                global_step += 1
                pbar.update(1)

        pbar.close()

        # 最终保存
        final_path = save_dir / "gate_final.pt"
        self._save_checkpoint(final_path, global_step)
        logger.info(f"[GateTrainer] 训练完成。最终模型保存至 {final_path}")

        return history

    def _save_checkpoint(self, path: Path, step: int) -> None:
        """保存检查点。"""
        state: dict[str, Any] = {
            "step": step,
            "l1_state_dict": self.l1.state_dict(),
            "l1_config": self.l1_config,
            "train_config_data_mode": self.train_config.data_mode,
        }
        if not self._lm_head_is_shared and self.lm_head is not None:
            state["lm_head_state_dict"] = self.lm_head.state_dict()
        torch.save(state, path)
        logger.info(f"[GateTrainer] 检查点保存至 {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """加载检查点。

        Returns:
            恢复的 step 数。
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.l1.load_state_dict(ckpt["l1_state_dict"])
        if "lm_head_state_dict" in ckpt and not self._lm_head_is_shared:
            assert self.lm_head is not None
            self.lm_head.load_state_dict(ckpt["lm_head_state_dict"])
        step = ckpt.get("step", 0)
        logger.info(f"[GateTrainer] 从 {path} 恢复，step={step}")
        return step


# ======================================================================
# 辅助类
# ======================================================================

class _InMemoryDataset(Dataset):
    """将内存中的样本列表包装为 Dataset。"""

    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
