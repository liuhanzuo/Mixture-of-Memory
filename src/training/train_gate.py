"""
L1 门控网络训练器。

训练目标: 学习最优门控参数，使得 L1 记忆读取结果与隐藏状态的融合
能最小化下游 language modeling loss。

训练策略:
1. 冻结 backbone 参数
2. 冻结 L1 投影层 (可选，也可联合训练)
3. 只训练门控网络参数 (gate_proj, output_proj, learned_gate)

损失函数:
- 主损失: 门控融合后的隐藏状态经 LM head 的 cross-entropy loss
- 辅助损失 (可选): 门控稀疏性正则化，鼓励门控信号稀疏
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.memory.l1.assoc_memory import AssociativeMemoryL1, L1Config
from src.memory.l1.gating import L1Gate

logger = logging.getLogger(__name__)


@dataclass
class GateTrainConfig:
    """门控训练配置。"""

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

    # 保存
    save_dir: str = "outputs/runs/gate_training"
    save_every: int = 200

    # 日志
    log_every: int = 10


class SyntheticGateDataset(Dataset):
    """合成门控训练数据集。

    生成随机隐藏状态序列，用于训练门控网络。
    实际使用时应替换为从 backbone 中提取的真实隐藏状态。

    TODO: 替换为真实数据管线:
    1. 用冻结的 backbone 对训练语料做 forward pass
    2. 保存中间层隐藏状态
    3. 用这些隐藏状态训练门控
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
        # 合成目标 token ids
        target_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            "hidden_states": hidden_states,
            "target_ids": target_ids,
        }


class L1GateTrainer:
    """L1 门控网络训练器。

    训练流程:
    1. 初始化 L1 记忆模块 (含门控)
    2. 创建简化的 LM head 用于计算损失
    3. 冻结非训练参数
    4. 迭代训练门控参数
    """

    def __init__(
        self,
        l1_config: L1Config,
        train_config: GateTrainConfig,
        hidden_dim: int = 256,
        vocab_size: int = 1000,
    ):
        self.l1_config = l1_config
        self.train_config = train_config
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # 初始化 L1 模块
        self.l1 = AssociativeMemoryL1(l1_config)
        self.l1.set_hidden_dim(hidden_dim)

        # 简化的 LM head (用于计算损失)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_trainable_params(self) -> list[nn.Parameter]:
        """获取需要训练的参数。"""
        params: list[nn.Parameter] = []

        # 门控参数
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

        # LM head 参数 (也需要训练以适配门控输出)
        params.extend(self.lm_head.parameters())

        return params

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
        # L1 前向: 写入 + 读取 + 门控
        gated_output = self.l1(hidden_states, update=True)

        # LM head
        logits = self.lm_head(gated_output)  # (B, S, vocab)

        # Cross-entropy loss
        ce_loss = nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_ids.view(-1),
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

    def train(
        self,
        dataset: Dataset | None = None,
    ) -> dict[str, list[float]]:
        """执行门控训练。

        Args:
            dataset: 训练数据集。若为 None，使用合成数据。

        Returns:
            训练指标历史记录。
        """
        if dataset is None:
            dataset = SyntheticGateDataset(
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
            )
            logger.warning("[GateTrainer] 使用合成数据集训练。请替换为真实数据。")

        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # 移至设备
        self.l1.to(self.device)
        self.lm_head.to(self.device)

        # 优化器
        trainable_params = self._get_trainable_params()
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )

        # 训练循环
        history: dict[str, list[float]] = {
            "ce_loss": [],
            "sparsity_loss": [],
            "total_loss": [],
        }

        self.l1.train()
        self.lm_head.train()
        global_step = 0

        save_dir = Path(self.train_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[GateTrainer] 开始训练，共 {self.train_config.max_steps} 步")
        logger.info(f"[GateTrainer] 可训练参数数: {sum(p.numel() for p in trainable_params)}")

        while global_step < self.train_config.max_steps:
            for batch in dataloader:
                if global_step >= self.train_config.max_steps:
                    break

                hidden_states = batch["hidden_states"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                # 每个 batch 开始前重置 L1 记忆
                self.l1.reset()

                loss, metrics = self._compute_loss(hidden_states, target_ids)
                loss = loss / self.train_config.grad_accumulation_steps
                loss.backward()

                if (global_step + 1) % self.train_config.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        self.train_config.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                # 记录
                for k, v in metrics.items():
                    history[k].append(v)

                if global_step % self.train_config.log_every == 0:
                    logger.info(
                        f"[GateTrainer] step={global_step} "
                        f"ce_loss={metrics['ce_loss']:.4f} "
                        f"sparsity={metrics['sparsity_loss']:.4f} "
                        f"total={metrics['total_loss']:.4f}"
                    )

                # 保存检查点
                if (global_step + 1) % self.train_config.save_every == 0:
                    ckpt_path = save_dir / f"gate_step{global_step + 1}.pt"
                    self._save_checkpoint(ckpt_path, global_step)

                global_step += 1

        # 最终保存
        final_path = save_dir / "gate_final.pt"
        self._save_checkpoint(final_path, global_step)
        logger.info(f"[GateTrainer] 训练完成。最终模型保存至 {final_path}")

        return history

    def _save_checkpoint(self, path: Path, step: int) -> None:
        """保存检查点。"""
        state = {
            "step": step,
            "l1_state_dict": self.l1.state_dict(),
            "lm_head_state_dict": self.lm_head.state_dict(),
            "l1_config": self.l1_config,
        }
        torch.save(state, path)
        logger.info(f"[GateTrainer] 检查点保存至 {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """加载检查点。

        Returns:
            恢复的 step 数。
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.l1.load_state_dict(ckpt["l1_state_dict"])
        self.lm_head.load_state_dict(ckpt["lm_head_state_dict"])
        step = ckpt.get("step", 0)
        logger.info(f"[GateTrainer] 从 {path} 恢复，step={step}")
        return step
