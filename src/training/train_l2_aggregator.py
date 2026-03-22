"""
L2 聚合器训练器。

训练目标: 微调 LLM 后端，使其能从对话消息中准确提取结构化记忆对象。

训练策略:
1. 加载基础模型 (如 Qwen3-1.7B) + tokenizer
2. 应用 LoRA (peft) 进行参数高效微调
3. 构造 (messages → expected L2 objects JSON) 的训练数据对
4. 使用 causal LM 损失训练
5. 保存 LoRA adapter 权重

数据格式 (JSONL):
每行: {"messages": [...], "objects": [...]}
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.memory.l2.types import ChatMessage, L2MemoryObject

logger = logging.getLogger(__name__)


# ======================================================================
# 训练配置
# ======================================================================

@dataclass
class L2AggregatorTrainConfig:
    """L2 聚合器训练配置。"""

    # 数据
    train_data_path: str = "data/processed/l2_train.jsonl"
    val_data_path: str = "data/processed/l2_val.jsonl"

    # 训练参数
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 模型
    base_model: str = "/apdcephfs/pig_data/Adaptive-Sparse-Trainer/models/Qwen--Qwen3-1.7b"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 保存
    save_dir: str = "outputs/runs/l2_aggregator_training"
    save_every_epoch: bool = True

    # 日志
    log_every: int = 10


# ======================================================================
# Prompt 模板
# ======================================================================

AGGREGATOR_SYSTEM_PROMPT = """You are a memory extraction assistant. Given a conversation, extract structured memory objects.

Each memory object should have:
- object_type: one of "topic", "preference", "task", "state", "entity"
- summary_text: a concise description
- confidence: a float between 0 and 1

Output as a JSON list of objects."""

AGGREGATOR_INPUT_TEMPLATE = """## Conversation:
{conversation}

## Extract memory objects (JSON list):"""


def format_messages_for_prompt(messages: list[dict[str, str]]) -> str:
    """将消息列表格式化为训练 prompt 的对话部分。"""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def format_objects_as_target(objects: list[dict[str, Any]]) -> str:
    """将期望的 L2 对象格式化为训练目标。"""
    simplified = []
    for obj in objects:
        simplified.append({
            "object_type": obj.get("object_type", "topic"),
            "summary_text": obj.get("summary_text", ""),
            "confidence": obj.get("confidence", 0.8),
        })
    return json.dumps(simplified, ensure_ascii=False, indent=2)


# ======================================================================
# 数据集
# ======================================================================

class L2AggregatorDataset(Dataset):
    """L2 聚合器训练数据集。

    数据格式 (JSONL):
    {
        "messages": [{"role": "user", "content": "..."}, ...],
        "objects": [{"object_type": "topic", "summary_text": "...", "confidence": 0.8}, ...]
    }
    """

    def __init__(self, data_path: str, max_seq_len: int = 2048):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.samples: list[dict[str, Any]] = []

        if self.data_path.exists():
            self._load_data()
        else:
            logger.warning(f"[L2 Dataset] 数据文件不存在: {data_path}，使用合成数据")
            self._generate_synthetic()

    def _load_data(self) -> None:
        """从 JSONL 文件加载数据。"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"[L2 Dataset] 跳过无效行: {e}")
        logger.info(f"[L2 Dataset] 加载了 {len(self.samples)} 条训练样本")

    def _generate_synthetic(self) -> None:
        """生成合成训练数据。"""
        synthetic_samples = [
            {
                "messages": [
                    {"role": "user", "content": "I prefer using Python for data science."},
                    {"role": "assistant", "content": "Python is great for data science!"},
                    {"role": "user", "content": "I'm currently working on a memory system for LLM agents."},
                ],
                "objects": [
                    {"object_type": "preference", "summary_text": "using Python for data science", "confidence": 0.9},
                    {"object_type": "task", "summary_text": "working on a memory system for LLM agents", "confidence": 0.85},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "Can you help me understand transformer attention mechanisms?"},
                    {"role": "assistant", "content": "Sure! Transformer attention uses query, key, value..."},
                    {"role": "user", "content": "I always prefer detailed technical explanations with examples."},
                ],
                "objects": [
                    {"object_type": "topic", "summary_text": "transformer attention mechanisms", "confidence": 0.8},
                    {"object_type": "preference", "summary_text": "detailed technical explanations with examples", "confidence": 0.9},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "I'm researching sparse training for large language models."},
                    {"role": "assistant", "content": "Sparse training is an active area..."},
                    {"role": "user", "content": "Specifically, I'm looking at channel pruning and SLoRB."},
                ],
                "objects": [
                    {"object_type": "topic", "summary_text": "sparse training for large language models", "confidence": 0.9},
                    {"object_type": "state", "summary_text": "researching channel pruning and SLoRB", "confidence": 0.85},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "我最近在研究大语言模型的稀疏化训练。"},
                    {"role": "assistant", "content": "稀疏化训练是一个很有前景的方向。"},
                    {"role": "user", "content": "帮我记一下，我的项目叫MoM-Agent。"},
                ],
                "objects": [
                    {"object_type": "topic", "summary_text": "大语言模型的稀疏化训练", "confidence": 0.85},
                    {"object_type": "task", "summary_text": "项目MoM-Agent", "confidence": 0.9},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "My name is Alice and I work at DeepMind."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "I'm exploring RLHF techniques for code generation."},
                ],
                "objects": [
                    {"object_type": "entity", "summary_text": "User: Alice, works at DeepMind", "confidence": 0.95},
                    {"object_type": "state", "summary_text": "exploring RLHF techniques for code generation", "confidence": 0.85},
                ],
            },
        ]
        # 复制合成样本以获得更多训练数据
        for _ in range(30):
            self.samples.extend(synthetic_samples)
        logger.info(f"[L2 Dataset] 生成了 {len(self.samples)} 条合成训练样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        sample = self.samples[idx]
        conversation = format_messages_for_prompt(sample["messages"])
        input_text = AGGREGATOR_INPUT_TEMPLATE.format(conversation=conversation)
        target_text = format_objects_as_target(sample["objects"])
        return {
            "input_text": f"{AGGREGATOR_SYSTEM_PROMPT}\n\n{input_text}",
            "target_text": target_text,
            "full_text": f"{AGGREGATOR_SYSTEM_PROMPT}\n\n{input_text}\n{target_text}",
        }


# ======================================================================
# L2 聚合器训练器
# ======================================================================

class L2AggregatorTrainer:
    """L2 聚合器 LoRA 微调训练器。

    训练流程:
    1. 加载基础模型和 tokenizer
    2. 应用 LoRA 适配器 (仅训练少量参数)
    3. Tokenize 数据集 (input 部分不计算 loss，只对 target 部分计算)
    4. 自定义训练循环: warmup + cosine decay、梯度累积、检查点
    5. 保存 LoRA adapter 权重

    推理时可用 peft 加载 adapter 还原完整模型。
    """

    def __init__(self, config: L2AggregatorTrainConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------ #
    #  模型加载
    # ------------------------------------------------------------------ #

    def _load_model_and_tokenizer(self) -> None:
        """加载基础模型并应用 LoRA。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"[L2 Trainer] 加载基础模型: {self.config.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None,  # 手动管理设备
        )

        if self.config.use_lora:
            self._apply_lora()
        else:
            logger.info("[L2 Trainer] 全参数微调模式")

        self.model.to(self.device)

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"[L2 Trainer] 模型加载完成: "
            f"总参数={total_params:,}, 可训练={trainable_params:,} "
            f"({trainable_params / total_params:.2%})"
        )

    def _apply_lora(self) -> None:
        """应用 LoRA 适配器。"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info(
                f"[L2 Trainer] LoRA 已应用: rank={self.config.lora_rank}, "
                f"alpha={self.config.lora_alpha}, "
                f"target_modules={self.config.lora_target_modules}"
            )
            self.model.print_trainable_parameters()

        except ImportError:
            logger.error(
                "[L2 Trainer] peft 未安装！请运行: pip install peft>=0.7.0\n"
                "回退到全参数微调。"
            )

    # ------------------------------------------------------------------ #
    #  数据准备
    # ------------------------------------------------------------------ #

    def prepare_data(self) -> tuple[L2AggregatorDataset, L2AggregatorDataset | None]:
        """准备训练和验证数据集。"""
        train_dataset = L2AggregatorDataset(
            self.config.train_data_path,
            self.config.max_seq_len,
        )

        val_dataset = None
        if Path(self.config.val_data_path).exists():
            val_dataset = L2AggregatorDataset(
                self.config.val_data_path,
                self.config.max_seq_len,
            )

        return train_dataset, val_dataset

    def _tokenize_sample(self, sample: dict[str, str]) -> dict[str, torch.Tensor]:
        """Tokenize 单个样本。

        策略: 对 full_text (input + target) 整体 tokenize，
        然后将 input 部分的 labels 设为 -100，仅对 target 部分计算 loss。
        """
        assert self.tokenizer is not None

        input_text = sample["input_text"]
        target_text = sample["target_text"]
        full_text = sample["full_text"]

        # Tokenize input 部分 (用于确定 input 长度)
        input_ids_only = self.tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        input_len = len(input_ids_only)

        # Tokenize 完整文本
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)       # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)

        # 构造 labels: input 部分 = -100, target 部分 = input_ids
        labels = input_ids.clone()
        # 将 input 部分 (不含 target) 的 labels 设为 -100
        labels[:min(input_len, labels.shape[0])] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _collate_fn(
        self, batch: list[dict[str, str]]
    ) -> dict[str, torch.Tensor]:
        """Tokenize + pad 一个 batch。"""
        tokenized = [self._tokenize_sample(sample) for sample in batch]

        max_len = max(t["input_ids"].shape[0] for t in tokenized)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for t in tokenized:
            seq_len = t["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # Left padding (与 generation 一致)
            batch_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), t["input_ids"]])
            )
            batch_attention_mask.append(
                torch.cat([torch.zeros(pad_len, dtype=torch.long), t["attention_mask"]])
            )
            batch_labels.append(
                torch.cat([torch.full((pad_len,), -100, dtype=torch.long), t["labels"]])
            )

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }

    # ------------------------------------------------------------------ #
    #  训练
    # ------------------------------------------------------------------ #

    def train(self) -> dict[str, Any]:
        """执行 LoRA 微调训练。

        Returns:
            训练结果摘要（含 loss 历史）。
        """
        # 加载模型
        self._load_model_and_tokenizer()

        # 准备数据
        logger.info("[L2 Trainer] 准备训练数据...")
        train_dataset, val_dataset = self.prepare_data()
        logger.info(f"[L2 Trainer] 训练集大小: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"[L2 Trainer] 验证集大小: {len(val_dataset)}")

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

        # 优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # 学习率调度器 (线性 warmup + cosine decay)
        steps_per_epoch = len(dataloader) // self.config.grad_accumulation_steps
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 训练循环
        logger.info(f"[L2 Trainer] 开始训练: {self.config.epochs} epochs, {total_steps} 步")
        logger.info(f"[L2 Trainer] Warmup 步数: {warmup_steps}")
        logger.info(f"[L2 Trainer] 设备: {self.device}")

        history: dict[str, list[float]] = {"train_loss": [], "lr": []}
        global_step = 0
        best_val_loss = float("inf")

        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            optimizer.zero_grad()

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                leave=True,
            )

            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.config.grad_accumulation_steps
                    loss.backward()
                except RuntimeError as e:
                    logger.warning(
                        f"[L2 Trainer] epoch={epoch+1} batch={batch_idx} 训练错误: {e}"
                    )
                    optimizer.zero_grad()
                    continue

                if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                raw_loss = loss.item() * self.config.grad_accumulation_steps
                epoch_loss += raw_loss
                epoch_steps += 1
                history["train_loss"].append(raw_loss)
                history["lr"].append(scheduler.get_last_lr()[0])

                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "avg_loss": f"{epoch_loss / epoch_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # 日志
                if global_step % self.config.log_every == 0 and global_step > 0:
                    logger.info(
                        f"[L2 Trainer] step={global_step} "
                        f"loss={raw_loss:.4f} "
                        f"avg_loss={epoch_loss / epoch_steps:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

            pbar.close()
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                f"[L2 Trainer] Epoch {epoch + 1} 完成: "
                f"avg_loss={avg_epoch_loss:.4f}, steps={epoch_steps}"
            )

            # Epoch 结束验证
            if val_dataset:
                val_loss = self._validate(val_dataset)
                logger.info(f"[L2 Trainer] Epoch {epoch + 1} val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(self.save_dir / "best_adapter", epoch, global_step)
                    logger.info(f"[L2 Trainer] 最佳模型已更新 (val_loss={val_loss:.4f})")

            # 按 epoch 保存
            if self.config.save_every_epoch:
                self._save_checkpoint(
                    self.save_dir / f"epoch_{epoch + 1}",
                    epoch,
                    global_step,
                )

        # 最终保存
        self._save_checkpoint(self.save_dir / "final_adapter", self.config.epochs, global_step)

        # 保存训练配置
        config_path = self.save_dir / "train_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2, ensure_ascii=False)

        results = {
            "status": "completed",
            "epochs": self.config.epochs,
            "total_steps": global_step,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
            "best_val_loss": best_val_loss if val_dataset else None,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "base_model": self.config.base_model,
            "use_lora": self.config.use_lora,
            "save_dir": str(self.save_dir),
        }

        logger.info(f"[L2 Trainer] 训练完成! 最终结果: {results}")
        return results

    # ------------------------------------------------------------------ #
    #  验证
    # ------------------------------------------------------------------ #

    def _validate(self, val_dataset: L2AggregatorDataset) -> float:
        """在验证集上计算平均 loss。"""
        self.model.eval()

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_loss += outputs.loss.item()
                    total_steps += 1
                except RuntimeError as e:
                    logger.warning(f"[L2 Trainer] 验证错误: {e}")
                    continue

        self.model.train()
        return total_loss / max(total_steps, 1)

    # ------------------------------------------------------------------ #
    #  评估
    # ------------------------------------------------------------------ #

    def evaluate(self, dataset: L2AggregatorDataset) -> dict[str, float]:
        """评估聚合器性能。

        用训练后的模型对 dataset 中的样本做推理，
        然后与 ground truth 对比计算 precision / recall / F1。
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("[L2 Trainer] 模型未加载，无法评估。")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        self.model.eval()

        total_precision = 0.0
        total_recall = 0.0
        total_count = 0

        for i in range(min(len(dataset), 50)):  # 限制评估样本数
            sample = dataset[i]
            input_text = sample["input_text"]
            target_text = sample["target_text"]

            # 生成
            encoding = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len,
            )
            input_ids = encoding["input_ids"].to(self.device)

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=1.0,
                    )
                generated = self.tokenizer.decode(
                    outputs[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()
            except Exception as e:
                logger.warning(f"[L2 Trainer] 评估样本 {i} 生成失败: {e}")
                continue

            # 解析生成结果和 ground truth
            pred_types = self._extract_object_types(generated)
            gt_types = self._extract_object_types(target_text)

            if gt_types:
                hits = len(pred_types & gt_types)
                precision = hits / len(pred_types) if pred_types else 0.0
                recall = hits / len(gt_types)
                total_precision += precision
                total_recall += recall
                total_count += 1

        if total_count == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        avg_precision = total_precision / total_count
        avg_recall = total_recall / total_count
        f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )

        metrics = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1,
            "num_evaluated": total_count,
        }
        logger.info(f"[L2 Trainer] 评估结果: {metrics}")
        return metrics

    @staticmethod
    def _extract_object_types(text: str) -> set[str]:
        """从 JSON 文本中提取 object_type 集合。"""
        types = set()
        try:
            objects = json.loads(text)
            if isinstance(objects, list):
                for obj in objects:
                    if isinstance(obj, dict) and "object_type" in obj:
                        types.add(f"{obj['object_type']}:{obj.get('summary_text', '')[:30]}")
        except (json.JSONDecodeError, TypeError):
            # 尝试从非 JSON 文本中正则提取
            import re
            for match in re.findall(r'"object_type"\s*:\s*"([^"]+)"', text):
                types.add(match)
        return types

    # ------------------------------------------------------------------ #
    #  检查点
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, save_path: Path, epoch: int, step: int) -> None:
        """保存 LoRA adapter 或完整模型。"""
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            if self.config.use_lora and hasattr(self.model, "save_pretrained"):
                # 保存 LoRA adapter
                self.model.save_pretrained(str(save_path))
                logger.info(f"[L2 Trainer] LoRA adapter 已保存至 {save_path}")
            else:
                # 保存完整模型
                self.model.save_pretrained(str(save_path))
                logger.info(f"[L2 Trainer] 模型已保存至 {save_path}")

            # 保存 tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(save_path))

            # 保存元信息
            meta = {"epoch": epoch, "step": step, "base_model": self.config.base_model}
            with open(save_path / "training_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        except Exception as e:
            logger.error(f"[L2 Trainer] 保存检查点失败: {e}", exc_info=True)

    def load_checkpoint(self, adapter_path: str | Path) -> None:
        """加载已保存的 LoRA adapter。"""
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter 路径不存在: {adapter_path}")

        if self.model is None:
            self._load_model_and_tokenizer()

        try:
            from peft import PeftModel
            # 重新加载 base model 并注入 adapter
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.to(self.device)
            logger.info(f"[L2 Trainer] LoRA adapter 已从 {adapter_path} 加载")

        except ImportError:
            logger.error("[L2 Trainer] peft 未安装，无法加载 LoRA adapter。")
        except Exception as e:
            logger.error(f"[L2 Trainer] 加载 adapter 失败: {e}", exc_info=True)
