"""
合成数据集评估脚本。

功能:
  1. 加载训练好的 memory-augmented 模型
  2. 在合成测试集上进行自回归生成
  3. 计算 exact match / update accuracy / temporal accuracy
  4. 按任务类型输出详细报告

可作为独立脚本运行，也可被 trainer 调用。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backbone.lm_wrapper import FrozenLMWrapper
from src.common.config import load_config
from src.common.logging import setup_logging
from src.common.seed import set_seed
from src.data.synthetic_dataset import build_synthetic_from_config
from src.eval.metrics import MemoryMetrics, extract_answer_from_generation

logger = logging.getLogger(__name__)


class SyntheticEvaluator:
    """合成数据集评估器。

    封装了从加载模型到输出指标的完整评估流程。

    Args:
        cfg: OmegaConf 配置。
        device: 评估设备。
    """

    def __init__(
        self,
        cfg: Any,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.metrics = MemoryMetrics(normalize=True)

    @torch.no_grad()
    def evaluate_with_trainer(
        self,
        trainer: Any,
        test_loader: DataLoader,
        tokenizer: Any,
        max_new_tokens: int = 16,
    ) -> Dict[str, Any]:
        """使用 trainer 中的模型进行评估。

        采用两阶段评估:
        1. Teacher forcing: 将完整序列（含答案）输入模型，看 loss
        2. 自回归生成: 只提供问题前的 context，让模型生成答案

        Args:
            trainer: MemoryTrainer 实例。
            test_loader: 测试数据加载器。
            tokenizer: HuggingFace tokenizer。
            max_new_tokens: 生成最大 token 数。

        Returns:
            metrics_dict: 评估指标字典。
        """
        self.metrics.reset()

        # ---- Teacher forcing 评估（计算 loss） ----
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(test_loader, desc="评估中 (teacher forcing)"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)

            result = trainer.memory_augmented_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if result["loss"] is not None:
                total_loss += result["loss"].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Teacher forcing 平均损失: {avg_loss:.4f}")

        # ---- 自回归生成评估（计算 accuracy） ----
        for batch in tqdm(test_loader, desc="评估中 (生成)"):
            answers = batch["answer"]
            task_types = batch["task_type"]
            texts = batch["text"]

            for i, text in enumerate(texts):
                # 找到 "Question:" 的位置，截断输入
                question_idx = text.rfind("Question:")
                if question_idx == -1:
                    continue

                # 用 "Answer:" 之前的文本作为 prompt
                answer_idx = text.rfind("Answer:")
                if answer_idx == -1:
                    prompt = text[:question_idx + len("Question:") + 100]
                else:
                    prompt = text[:answer_idx + len("Answer:")]

                # Tokenize prompt
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.get("block", {}).get("block_size", 512) * 4,
                )
                prompt_ids = encoded["input_ids"].to(self.device)

                # 简单的自回归生成（使用 backbone only）
                # v0: 不做完整的 memory-augmented generation，因为需要
                # 逐 token 更新逻辑比较复杂。先用 backbone 做基线评估。
                generated = self._simple_generate(
                    trainer, prompt_ids, max_new_tokens, tokenizer
                )
                pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                pred_answer = extract_answer_from_generation(pred_text)

                self.metrics.add_result(
                    prediction=pred_answer,
                    answer=answers[i],
                    task_type=task_types[i],
                )

        # ---- 汇总指标 ----
        metrics = self.metrics.compute()
        metrics["avg_loss"] = avg_loss
        self.metrics.log_metrics(metrics)

        return metrics

    @torch.no_grad()
    def _simple_generate(
        self,
        trainer: Any,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        tokenizer: Any,
    ) -> torch.Tensor:
        """简单的自回归生成（使用 backbone + memory fusion）。

        v0 简化版本:
        1. 对 prompt 做完整的 memory-augmented forward（建立记忆）
        2. 逐 token 生成时，使用已建立的 MOM 做 readout + fusion

        Args:
            trainer: MemoryTrainer 实例。
            prompt_ids: [1, T] prompt token ids。
            max_new_tokens: 最大生成 token 数。
            tokenizer: tokenizer 用于 eos 检测。

        Returns:
            generated: [1, T + new_tokens] 完整序列。
        """
        B = prompt_ids.shape[0]

        # Step 1: 对 prompt 做 memory-augmented forward 以建立记忆
        result = trainer.memory_augmented_forward(
            input_ids=prompt_ids,
        )

        # 获取当前 MOM 状态（已通过 forward 建立）
        memory_states = [mem.state for mem in trainer.mom.memories]

        # Step 2: 逐 token 生成
        generated = prompt_ids.clone()
        eos_id = getattr(tokenizer, "eos_token_id", None)

        for _ in range(max_new_tokens):
            # 获取最后一个 token 的 backbone hidden state
            with torch.no_grad():
                backbone_out = trainer.backbone(input_ids=generated)
            last_hidden = backbone_out.hidden_states[:, -1:, :]  # [B, 1, D]

            # Memory readout
            readout = trainer.readout.forward_sequence(
                hidden_seq=last_hidden,
                memory_states=memory_states,
            )  # [B, 1, D_v]

            # Fusion
            fused = trainer.fusion(
                hidden_states=last_hidden,
                memory_readout=readout,
            )  # [B, 1, D]

            # lm_head → logits
            logits = trainer.backbone.get_logits_from_hidden(fused)  # [B, 1, V]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)

            # EOS 检测
            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated

    @torch.no_grad()
    def evaluate_backbone_only(
        self,
        backbone: FrozenLMWrapper,
        test_loader: DataLoader,
        tokenizer: Any,
    ) -> Dict[str, Any]:
        """仅用 backbone（无记忆）评估，作为基线。

        Args:
            backbone: 冻结的 LM 包装器。
            test_loader: 测试数据加载器。
            tokenizer: tokenizer。

        Returns:
            metrics_dict: 评估指标。
        """
        self.metrics.reset()
        backbone.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(test_loader, desc="Backbone-only 评估"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)

            output = backbone(input_ids=input_ids, attention_mask=attention_mask)
            if labels is not None:
                # 简单的 CE loss
                from torch.nn import functional as F
                shift_logits = output.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"avg_loss": avg_loss, "method": "backbone_only"}
        logger.info(f"Backbone-only 平均损失: {avg_loss:.4f}")
        return metrics


def save_evaluation_results(
    metrics: Dict[str, Any],
    output_path: str,
    predictions: Optional[List[Dict[str, str]]] = None,
) -> None:
    """保存评估结果到 JSON 文件。

    Args:
        metrics: 指标字典。
        output_path: 输出路径。
        predictions: 可选的预测列表。
    """
    output = {
        "metrics": metrics,
    }
    if predictions:
        output["predictions"] = predictions

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"评估结果已保存: {path}")


# ============================================================
# CLI 入口
# ============================================================

def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="合成数据集评估")
    parser.add_argument(
        "--config", type=str, default="configs/synthetic.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/eval_synthetic.json",
        help="结果输出路径",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device(args.device)

    # 加载 backbone
    backbone = FrozenLMWrapper(
        model_name=cfg.model.name,
        freeze=True,
    ).to(device)
    tokenizer = backbone.tokenizer

    # 构建测试数据集
    _, _, test_ds = build_synthetic_from_config(cfg, tokenizer=tokenizer)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    # 加载 trainer 并恢复检查点
    from src.training.trainer import MemoryTrainer
    trainer = MemoryTrainer(cfg=cfg, backbone=backbone, device=device)
    trainer.load_from_checkpoint(args.checkpoint)

    # 评估
    evaluator = SyntheticEvaluator(cfg=cfg, device=device)
    metrics = evaluator.evaluate_with_trainer(
        trainer=trainer,
        test_loader=test_loader,
        tokenizer=tokenizer,
    )

    # 保存结果
    save_evaluation_results(metrics, args.output)


if __name__ == "__main__":
    main()
