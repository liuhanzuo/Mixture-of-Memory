"""
Memory-Augmented Trainer: 核心训练循环。

编排所有模块完成 block-wise 记忆增强训练:
  1. 冻结 backbone 前向传播，获取 hidden states
  2. 按 block 切分，逐 block:
     a. BlockEvaluator 打分
     b. AnchorSelector 选 top-k
     c. RetrospectiveGather 聚合
     d. MemoryWriter 决策写入
     e. MOM 更新
  3. MemoryReadout 读取记忆
  4. FusionHead 融合
  5. 通过 lm_head 计算 logits
  6. 计算组合损失并反向传播
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anchor.evaluator import BlockEvaluator
from src.anchor.selector import AnchorSelector
from src.backbone.lm_wrapper import FrozenLMWrapper
from src.fusion.fusion_head import FusionHead
from src.gather.block_buffer import BlockBuffer
from src.gather.retrospective_attn import RetrospectiveGather
from src.memory.mom import MixtureOfMemory
from src.memory.readout import MemoryReadout
from src.memory.retention import RetentionScheduler
from src.memory.update import MemoryWriter
from src.training.losses import MemoryAugmentedLoss
from src.training.stages import TrainingStageManager
from src.training.utils import (
    clip_grad_norm,
    collect_trainable_params,
    count_trainable_params,
    get_optimizer,
    get_scheduler,
    load_checkpoint,
    log_param_summary,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


class MemoryTrainer:
    """Memory-augmented 系统的训练器。

    管理完整的训练/验证循环，包括:
    - 模块初始化与参数冻结
    - Block-wise 前向传播
    - 记忆更新与读取
    - 损失计算与梯度更新
    - 日志记录与检查点保存

    Args:
        cfg: OmegaConf 配置对象。
        backbone: 冻结的 LM 包装器。
        device: 训练设备。
    """

    def __init__(
        self,
        cfg: Any,
        backbone: FrozenLMWrapper,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.cfg = cfg
        self.backbone = backbone
        self.device = device

        # ---- 从配置中提取参数 ----
        hidden_dim = backbone.hidden_dim
        mem_cfg = cfg.memory
        anchor_cfg = cfg.anchor
        gather_cfg = cfg.gather
        write_cfg = cfg.write
        readout_cfg = cfg.readout
        fusion_cfg = cfg.fusion
        train_cfg = cfg.training
        loss_cfg = cfg.loss

        # ---- 创建外部模块 ----
        self.block_buffer = BlockBuffer(
            block_size=cfg.block.block_size,
        )

        self.evaluator = BlockEvaluator(
            hidden_dim=hidden_dim,
            scorer_type=anchor_cfg.evaluator_type,
            num_layers=anchor_cfg.evaluator_layers,
        ).to(device)

        self.selector = AnchorSelector(
            top_k=anchor_cfg.top_k,
        ).to(device)

        self.gather = RetrospectiveGather(
            hidden_dim=hidden_dim,
            gather_dim=gather_cfg.gather_dim,
        ).to(device)

        self.mom = MixtureOfMemory(
            num_memories=mem_cfg.num_memories,
            key_dim=mem_cfg.mem_dim,
            value_dim=mem_cfg.mem_dim,
        ).to(device)

        self.writer = MemoryWriter(
            input_dim=hidden_dim,
            key_dim=mem_cfg.mem_dim,
            value_dim=mem_cfg.mem_dim,
            num_memories=mem_cfg.num_memories,
            hidden_dim=write_cfg.head_hidden,
        ).to(device)

        self.retention = RetentionScheduler(
            num_memories=mem_cfg.num_memories,
            mode="learned",
            default_retentions=list(mem_cfg.init_retention),
            memory_names=self.mom.memory_names,
        ).to(device)

        self.readout = MemoryReadout(
            hidden_dim=hidden_dim,
            key_dim=mem_cfg.mem_dim,
            value_dim=mem_cfg.mem_dim,
            num_memories=mem_cfg.num_memories,
        ).to(device)

        self.fusion = FusionHead(
            hidden_dim=hidden_dim,
            memory_dim=mem_cfg.mem_dim,
            gate_init_bias=fusion_cfg.gate_init_bias,
        ).to(device)

        # ---- 损失函数 ----
        self.loss_fn = MemoryAugmentedLoss(
            lm_weight=loss_cfg.lm_weight,
            utility_weight=loss_cfg.utility_weight,
            utility_loss_type="mse",
        )

        # ---- 训练配置 ----
        self.block_size = cfg.block.block_size
        self.max_steps = train_cfg.max_steps
        self.eval_every = train_cfg.eval_every
        self.save_every = train_cfg.save_every
        self.log_every = train_cfg.log_every
        self.max_grad_norm = train_cfg.max_grad_norm
        self.gradient_accumulation = train_cfg.get("gradient_accumulation", 1)

        # ---- 收集可训练模块 ----
        self.trainable_modules: Dict[str, nn.Module] = {
            "evaluator": self.evaluator,
            "gather": self.gather,
            "writer": self.writer,
            "retention": self.retention,
            "readout": self.readout,
            "fusion": self.fusion,
        }

        # ---- 配置训练阶段 ----
        stage_manager = TrainingStageManager(stage=train_cfg.stage)
        trainable_params = stage_manager.configure_stage(
            backbone=self.backbone,
            trainable_modules=self.trainable_modules,
        )

        # ---- 优化器 & 调度器 ----
        self.optimizer = get_optimizer(
            params=trainable_params,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            max_steps=train_cfg.max_steps,
        )

        # ---- 打印参数统计 ----
        logger.info("参数统计:")
        log_param_summary({"backbone": self.backbone, **self.trainable_modules})

        # ---- 训练状态 ----
        self.global_step = 0
        self.best_val_loss = float("inf")

        # ---- 检查点路径 ----
        self.checkpoint_dir = Path(cfg.paths.checkpoint_dir)
        self.output_dir = Path(cfg.paths.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 核心前向传播: block-wise 记忆增强
    # ================================================================

    def memory_augmented_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """完整的 memory-augmented 前向传播。

        流程:
          1. Backbone 前向 → hidden states
          2. 按 block 切分 hidden states
          3. 逐 block:
             - evaluator 打分
             - selector 选 anchor
             - gather 聚合
             - writer 决策
             - MOM 更新
          4. 用更新后的 MOM 对整个序列做 readout
          5. Fusion → fused hidden
          6. lm_head → logits
          7. 计算损失

        Args:
            input_ids: [B, T]
            attention_mask: [B, T]
            labels: [B, T]

        Returns:
            result_dict: 包含 logits, loss, loss_dict, stats
        """
        B, T = input_ids.shape

        # ---- Step 1: Backbone 前向传播 ----
        with torch.no_grad():
            backbone_out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = backbone_out.hidden_states  # [B, T, D]

        # ---- Step 2: 初始化记忆 ----
        self.mom.reset(batch_size=B, device=self.device)

        # ---- Step 3: Block-wise 处理 ----
        block_size = self.block_size
        num_blocks = (T + block_size - 1) // block_size

        all_evaluator_scores = []
        all_anchor_indices = []
        all_write_stats = []

        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, T)
            block_hidden = hidden_states[:, start:end, :]  # [B, L, D]

            if block_hidden.shape[1] < 2:
                # block 太短，跳过
                continue

            # 3a. Evaluator 打分
            scores = self.evaluator(block_hidden)  # [B, L]
            all_evaluator_scores.append((start, scores))

            # 3b. 选择 top-k anchor
            selection = self.selector(scores, block_hidden)
            # selection.indices: [B, K], selection.hidden_states: [B, K, D]
            all_anchor_indices.append((start, selection.indices))

            # 3c. Retrospective gather
            gathered, attn_weights = self.gather(
                block_hidden=block_hidden,
                anchor_hidden=selection.hidden_states,
            )  # gathered: [B, K, D]

            # 3d. 逐 anchor 写入 MOM
            for k in range(gathered.shape[1]):
                z_k = gathered[:, k, :]  # [B, D]

                # Writer 决策
                decision = self.writer(z_k)

                # 混合 retention
                lam = self.retention(
                    batch_size=B,
                    writer_lam=decision.lam,
                    blend_ratio=0.5,
                )

                # 更新 MOM
                self.mom.update(
                    key=decision.key,
                    value=decision.value,
                    alpha=decision.alpha,
                    rho=decision.rho,
                    lam=lam,
                )

            # 收集写入统计
            if gathered.shape[1] > 0:
                last_decision = self.writer(gathered[:, -1, :])
                all_write_stats.append(
                    self.writer.get_decision_stats(last_decision)
                )

        # ---- Step 4: Memory readout (对整个序列) ----
        memory_states = [
            mem.state for mem in self.mom.memories
        ]  # list of [B, D_k, D_v]

        readout_seq = self.readout.forward_sequence(
            hidden_seq=hidden_states,
            memory_states=memory_states,
        )  # [B, T, D_v]

        # ---- Step 5: Fusion ----
        fused_hidden = self.fusion(
            hidden_states=hidden_states,
            memory_readout=readout_seq,
        )  # [B, T, D]

        # ---- Step 6: lm_head → logits ----
        logits = self.backbone.get_logits_from_hidden(fused_hidden)  # [B, T, V]

        # ---- Step 7: 计算损失 ----
        total_loss = None
        loss_dict = {}

        if labels is not None:
            total_loss, loss_dict = self.loss_fn(
                logits=logits,
                labels=labels,
            )

        # ---- 收集统计信息 ----
        stats = self._collect_stats(
            all_evaluator_scores, all_anchor_indices, all_write_stats
        )

        return {
            "logits": logits,
            "fused_hidden": fused_hidden,
            "loss": total_loss,
            "loss_dict": loss_dict,
            "stats": stats,
        }

    # ================================================================
    # 训练循环
    # ================================================================

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """主训练循环。

        Args:
            train_loader: 训练数据加载器。
            val_loader: 可选的验证数据加载器。

        Returns:
            history: 训练历史记录。
        """
        logger.info("=" * 60)
        logger.info(f"开始训练: max_steps={self.max_steps}")
        logger.info("=" * 60)

        # 切换训练模式
        self._set_train_mode()

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        accumulated_loss = 0.0
        accumulated_steps = 0
        epoch = 0

        progress = tqdm(total=self.max_steps, desc="训练中")
        progress.update(self.global_step)

        while self.global_step < self.max_steps:
            epoch += 1
            for batch in train_loader:
                if self.global_step >= self.max_steps:
                    break

                # ---- 前向传播 ----
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels")
                if labels is not None:
                    labels = labels.to(self.device)

                result = self.memory_augmented_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = result["loss"]
                if loss is None:
                    continue

                # 梯度累积
                scaled_loss = loss / self.gradient_accumulation
                scaled_loss.backward()

                accumulated_loss += loss.item()
                accumulated_steps += 1

                # ---- 梯度更新 ----
                if accumulated_steps % self.gradient_accumulation == 0:
                    # 梯度裁剪
                    all_params = []
                    for module in self.trainable_modules.values():
                        all_params.extend(
                            p for p in module.parameters() if p.requires_grad
                        )
                    grad_norm = clip_grad_norm(all_params, self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    avg_loss = accumulated_loss / self.gradient_accumulation
                    accumulated_loss = 0.0

                    history["train_loss"].append(avg_loss)
                    history["lr"].append(self.scheduler.get_last_lr()[0])

                    # ---- 日志 ----
                    if self.global_step % self.log_every == 0:
                        self._log_training_step(
                            avg_loss, grad_norm, result["stats"],
                            result.get("loss_dict", {})
                        )

                    # ---- 验证 ----
                    if val_loader is not None and self.global_step % self.eval_every == 0:
                        val_loss = self.evaluate(val_loader)
                        history["val_loss"].append(val_loss)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_best()
                        self._set_train_mode()

                    # ---- 保存检查点 ----
                    if self.global_step % self.save_every == 0:
                        self._save_checkpoint()

                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    )

        progress.close()
        logger.info(f"训练完成: {self.global_step} steps, best_val_loss={self.best_val_loss:.4f}")

        # 保存最终检查点
        self._save_checkpoint(tag="final")

        return history

    # ================================================================
    # 验证循环
    # ================================================================

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> float:
        """验证循环。

        Args:
            val_loader: 验证数据加载器。

        Returns:
            avg_loss: 平均验证损失。
        """
        self._set_eval_mode()

        total_loss = 0.0
        total_batches = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)

            result = self.memory_augmented_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            if result["loss"] is not None:
                total_loss += result["loss"].item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        logger.info(
            f"[Eval] step={self.global_step}, "
            f"val_loss={avg_loss:.4f}, "
            f"best_val_loss={self.best_val_loss:.4f}"
        )
        return avg_loss

    # ================================================================
    # 辅助方法
    # ================================================================

    def _set_train_mode(self) -> None:
        """设置所有可训练模块为训练模式。"""
        for module in self.trainable_modules.values():
            module.train()
        self.backbone.eval()  # backbone 始终 eval

    def _set_eval_mode(self) -> None:
        """设置所有模块为评估模式。"""
        for module in self.trainable_modules.values():
            module.eval()
        self.backbone.eval()

    def _collect_stats(
        self,
        evaluator_scores_list: List[Tuple[int, torch.Tensor]],
        anchor_indices_list: List[Tuple[int, torch.Tensor]],
        write_stats_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """收集训练统计信息。"""
        stats: Dict[str, float] = {}

        # Evaluator 统计
        if evaluator_scores_list:
            all_scores = torch.cat(
                [s for _, s in evaluator_scores_list], dim=1
            )
            stats["eval_score_mean"] = all_scores.mean().item()
            stats["eval_score_std"] = all_scores.std().item()

        # Anchor 统计
        if anchor_indices_list:
            total_anchors = sum(
                idx.shape[1] for _, idx in anchor_indices_list
            )
            num_blocks = len(anchor_indices_list)
            stats["avg_anchors_per_block"] = total_anchors / max(num_blocks, 1)

        # Write 统计
        if write_stats_list:
            # 取最后一个 block 的统计作为代表
            for key, val in write_stats_list[-1].items():
                stats[f"write_{key}"] = val

        # Memory 统计
        mem_stats = self.mom.get_memory_stats()
        for mem_name, mem_stat in mem_stats.items():
            for stat_key, stat_val in mem_stat.items():
                stats[f"mem_{mem_name}_{stat_key}"] = stat_val

        # Retention 统计
        ret_stats = self.retention.get_stats()
        stats.update(ret_stats)

        return stats

    def _log_training_step(
        self,
        loss: float,
        grad_norm: float,
        stats: Dict[str, float],
        loss_dict: Dict[str, float],
    ) -> None:
        """记录训练步骤日志。"""
        lr = self.scheduler.get_last_lr()[0]
        msg_parts = [
            f"[Train] step={self.global_step}",
            f"loss={loss:.4f}",
            f"lr={lr:.2e}",
            f"grad_norm={grad_norm:.4f}",
        ]

        # 损失分量
        for k, v in loss_dict.items():
            if k != "total_loss":
                msg_parts.append(f"{k}={v:.4f}")

        # 关键统计
        key_stats = [
            "avg_anchors_per_block",
            "write_alpha_mean",
            "write_route_entropy",
        ]
        for k in key_stats:
            if k in stats:
                msg_parts.append(f"{k}={stats[k]:.3f}")

        logger.info("  ".join(msg_parts))

    def _save_checkpoint(self, tag: Optional[str] = None) -> None:
        """保存检查点。"""
        if tag is None:
            tag = f"step_{self.global_step}"
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        save_checkpoint(
            path=path,
            step=self.global_step,
            modules=self.trainable_modules,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics={"best_val_loss": self.best_val_loss},
        )

    def _save_best(self) -> None:
        """保存最佳验证模型。"""
        path = self.checkpoint_dir / "checkpoint_best.pt"
        save_checkpoint(
            path=path,
            step=self.global_step,
            modules=self.trainable_modules,
            metrics={"best_val_loss": self.best_val_loss},
        )
        logger.info(f"  保存最佳模型 (val_loss={self.best_val_loss:.4f})")

    def load_from_checkpoint(self, path: str) -> None:
        """从检查点恢复训练状态。

        Args:
            path: 检查点文件路径。
        """
        checkpoint = load_checkpoint(
            path=path,
            modules=self.trainable_modules,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.global_step = checkpoint.get("step", 0)
        metrics = checkpoint.get("metrics", {})
        self.best_val_loss = metrics.get("best_val_loss", float("inf"))
        logger.info(
            f"从检查点恢复: step={self.global_step}, "
            f"best_val_loss={self.best_val_loss:.4f}"
        )
