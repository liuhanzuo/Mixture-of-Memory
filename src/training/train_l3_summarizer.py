"""
L3 总结器训练器。

训练目标: 微调 LLM 后端，使其能从 L2 记忆对象中生成高质量的
长期语义/画像记忆条目。

训练策略:
1. 加载基础模型 (如 Qwen3-1.7B) + tokenizer
2. 应用 LoRA (peft) 进行参数高效微调
3. 构造 (L2 objects → expected L3 profile entries JSON) 的训练数据对
4. 使用 causal LM 损失训练 (仅对 target 部分计算 loss)
5. 保存 LoRA adapter 权重

数据格式 (JSONL):
每行: {"l2_objects": [...], "profile_entries": [...]}
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.memory.l3.summarizer import L3ProfileEntry

logger = logging.getLogger(__name__)


# ======================================================================
# 训练配置
# ======================================================================

@dataclass
class L3SummarizerTrainConfig:
    """L3 总结器训练配置。"""

    # 数据
    train_data_path: str = "data/processed/l3_train.jsonl"
    val_data_path: str = "data/processed/l3_val.jsonl"

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
    save_dir: str = "outputs/runs/l3_summarizer_training"
    save_every_epoch: bool = True

    # 日志
    log_every: int = 10


# ======================================================================
# Prompt 模板
# ======================================================================

SUMMARIZER_SYSTEM_PROMPT = """You are a user profile generation assistant. Given a set of episodic memory objects from a conversation, generate long-term profile entries that capture the user's stable traits, interests, and preferences.

Each profile entry should have:
- key: a short identifier (e.g., "research_interest", "preferred_style")
- value: a natural language description
- confidence: a float between 0 and 1
- category: one of "research_interest", "preference", "long_term_project", "identity", "factual"

Output as a JSON list."""

SUMMARIZER_INPUT_TEMPLATE = """## L2 Memory Objects:
{objects_text}

## Generate profile entries (JSON list):"""


def format_l2_objects_for_prompt(objects: list[dict[str, Any]]) -> str:
    """将 L2 对象列表格式化为 prompt 中的描述。"""
    lines = []
    for i, obj in enumerate(objects):
        obj_type = obj.get("object_type", "unknown")
        summary = obj.get("summary_text", "")
        confidence = obj.get("confidence", 0.5)
        lines.append(f"{i + 1}. [{obj_type}] {summary} (conf={confidence:.2f})")
    return "\n".join(lines)


def format_profile_entries_as_target(entries: list[dict[str, Any]]) -> str:
    """将期望的 L3 条目格式化为训练目标。"""
    simplified = []
    for entry in entries:
        simplified.append({
            "key": entry.get("key", ""),
            "value": entry.get("value", ""),
            "confidence": entry.get("confidence", 0.8),
            "category": entry.get("category", "factual"),
        })
    return json.dumps(simplified, ensure_ascii=False, indent=2)


# ======================================================================
# 数据集
# ======================================================================

class L3SummarizerDataset(Dataset):
    """L3 总结器训练数据集。

    数据格式 (JSONL):
    {
        "l2_objects": [{"object_type": "topic", "summary_text": "...", "confidence": 0.8}, ...],
        "profile_entries": [{"key": "...", "value": "...", "confidence": 0.9, "category": "..."}, ...]
    }
    """

    def __init__(self, data_path: str, max_seq_len: int = 2048):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.samples: list[dict[str, Any]] = []

        if self.data_path.exists():
            self._load_data()
        else:
            logger.warning(f"[L3 Dataset] 数据文件不存在: {data_path}，使用合成数据")
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
                        logger.warning(f"[L3 Dataset] 跳过无效行: {e}")
        logger.info(f"[L3 Dataset] 加载了 {len(self.samples)} 条训练样本")

    def _generate_synthetic(self) -> None:
        """生成合成训练数据。"""
        synthetic_samples = [
            {
                "l2_objects": [
                    {"object_type": "topic", "summary_text": "sparse training for LLMs", "confidence": 0.9},
                    {"object_type": "topic", "summary_text": "channel pruning techniques", "confidence": 0.85},
                    {"object_type": "state", "summary_text": "researching model compression", "confidence": 0.8},
                ],
                "profile_entries": [
                    {
                        "key": "research_interest",
                        "value": "The user is actively researching sparse training and model compression for large language models, with a focus on channel pruning.",
                        "confidence": 0.9,
                        "category": "research_interest",
                    },
                ],
            },
            {
                "l2_objects": [
                    {"object_type": "preference", "summary_text": "detailed technical explanations", "confidence": 0.9},
                    {"object_type": "preference", "summary_text": "structured responses with examples", "confidence": 0.85},
                    {"object_type": "preference", "summary_text": "using Python for implementations", "confidence": 0.8},
                ],
                "profile_entries": [
                    {
                        "key": "preferred_response_style",
                        "value": "The user prefers detailed, technical, and structured explanations with concrete examples.",
                        "confidence": 0.9,
                        "category": "preference",
                    },
                    {
                        "key": "preferred_language",
                        "value": "The user prefers Python for implementations and code examples.",
                        "confidence": 0.85,
                        "category": "preference",
                    },
                ],
            },
            {
                "l2_objects": [
                    {"object_type": "task", "summary_text": "building a hierarchical memory system for agents", "confidence": 0.9},
                    {"object_type": "task", "summary_text": "implementing associative matrix memory", "confidence": 0.85},
                    {"object_type": "topic", "summary_text": "agent memory architecture", "confidence": 0.8},
                ],
                "profile_entries": [
                    {
                        "key": "long_term_project",
                        "value": "The user's long-term project is building a hierarchical memory system (MoM) for LLM agents, featuring associative matrix memory for online context compensation.",
                        "confidence": 0.9,
                        "category": "long_term_project",
                    },
                ],
            },
            {
                "l2_objects": [
                    {"object_type": "entity", "summary_text": "User: Alice, works at DeepMind", "confidence": 0.95},
                    {"object_type": "state", "summary_text": "exploring RLHF techniques", "confidence": 0.8},
                    {"object_type": "topic", "summary_text": "reinforcement learning from human feedback", "confidence": 0.85},
                ],
                "profile_entries": [
                    {
                        "key": "identity",
                        "value": "The user is Alice, working at DeepMind.",
                        "confidence": 0.95,
                        "category": "identity",
                    },
                    {
                        "key": "research_interest",
                        "value": "The user is exploring RLHF (Reinforcement Learning from Human Feedback) techniques.",
                        "confidence": 0.85,
                        "category": "research_interest",
                    },
                ],
            },
            {
                "l2_objects": [
                    {"object_type": "topic", "summary_text": "FSDP distributed training", "confidence": 0.8},
                    {"object_type": "topic", "summary_text": "NCCL communication optimization", "confidence": 0.75},
                    {"object_type": "state", "summary_text": "debugging multi-GPU training pipeline", "confidence": 0.85},
                ],
                "profile_entries": [
                    {
                        "key": "research_interest",
                        "value": "The user is working on distributed training infrastructure, particularly FSDP and NCCL optimization for multi-GPU setups.",
                        "confidence": 0.8,
                        "category": "research_interest",
                    },
                ],
            },
        ]
        # 复制合成样本以获得更多训练数据
        for _ in range(30):
            self.samples.extend(synthetic_samples)
        logger.info(f"[L3 Dataset] 生成了 {len(self.samples)} 条合成训练样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        sample = self.samples[idx]
        objects_text = format_l2_objects_for_prompt(sample["l2_objects"])
        input_text = f"{SUMMARIZER_SYSTEM_PROMPT}\n\n{SUMMARIZER_INPUT_TEMPLATE.format(objects_text=objects_text)}"
        target_text = format_profile_entries_as_target(sample["profile_entries"])
        return {
            "input_text": input_text,
            "target_text": target_text,
            "full_text": f"{input_text}\n{target_text}",
        }


# ======================================================================
# L3 总结器训练器
# ======================================================================

class L3SummarizerTrainer:
    """L3 总结器 LoRA 微调训练器。

    训练流程:
    1. 加载基础模型和 tokenizer
    2. 应用 LoRA 适配器 (仅训练少量参数)
    3. Tokenize 数据集 (input 部分不计算 loss，只对 target 部分计算)
    4. 自定义训练循环: warmup + cosine decay、梯度累积、检查点
    5. 保存 LoRA adapter 权重

    推理时可用 peft 加载 adapter 还原完整模型。
    """

    def __init__(self, config: L3SummarizerTrainConfig):
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

        logger.info(f"[L3 Trainer] 加载基础模型: {self.config.base_model}")

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
            logger.info("[L3 Trainer] 全参数微调模式")

        self.model.to(self.device)

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"[L3 Trainer] 模型加载完成: "
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
                f"[L3 Trainer] LoRA 已应用: rank={self.config.lora_rank}, "
                f"alpha={self.config.lora_alpha}, "
                f"target_modules={self.config.lora_target_modules}"
            )
            self.model.print_trainable_parameters()

        except ImportError:
            logger.error(
                "[L3 Trainer] peft 未安装！请运行: pip install peft>=0.7.0\n"
                "回退到全参数微调。"
            )

    # ------------------------------------------------------------------ #
    #  数据准备
    # ------------------------------------------------------------------ #

    def prepare_data(self) -> tuple[L3SummarizerDataset, L3SummarizerDataset | None]:
        """准备训练和验证数据集。"""
        train_dataset = L3SummarizerDataset(
            self.config.train_data_path,
            self.config.max_seq_len,
        )

        val_dataset = None
        if Path(self.config.val_data_path).exists():
            val_dataset = L3SummarizerDataset(
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

        input_ids = encoding["input_ids"].squeeze(0)           # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)

        # 构造 labels: input 部分 = -100, target 部分 = input_ids
        labels = input_ids.clone()
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
        logger.info("[L3 Trainer] 准备训练数据...")
        train_dataset, val_dataset = self.prepare_data()
        logger.info(f"[L3 Trainer] 训练集大小: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"[L3 Trainer] 验证集大小: {len(val_dataset)}")

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
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 训练循环
        logger.info(f"[L3 Trainer] 开始训练: {self.config.epochs} epochs, {total_steps} 步")
        logger.info(f"[L3 Trainer] Warmup 步数: {warmup_steps}")
        logger.info(f"[L3 Trainer] 设备: {self.device}")

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
                        f"[L3 Trainer] epoch={epoch+1} batch={batch_idx} 训练错误: {e}"
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
                        f"[L3 Trainer] step={global_step} "
                        f"loss={raw_loss:.4f} "
                        f"avg_loss={epoch_loss / epoch_steps:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

            pbar.close()
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                f"[L3 Trainer] Epoch {epoch + 1} 完成: "
                f"avg_loss={avg_epoch_loss:.4f}, steps={epoch_steps}"
            )

            # Epoch 结束验证
            if val_dataset:
                val_loss = self._validate(val_dataset)
                logger.info(f"[L3 Trainer] Epoch {epoch + 1} val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(self.save_dir / "best_adapter", epoch, global_step)
                    logger.info(f"[L3 Trainer] 最佳模型已更新 (val_loss={val_loss:.4f})")

            # 按 epoch 保存
            if self.config.save_every_epoch:
                self._save_checkpoint(
                    self.save_dir / f"epoch_{epoch + 1}",
                    epoch,
                    global_step,
                )

        # 最终保存
        self._save_checkpoint(self.save_dir / "final_adapter", self.config.epochs, global_step)

        # 保存训练配置和历史
        config_path = self.save_dir / "train_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2, ensure_ascii=False)

        history_path = self.save_dir / "train_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

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

        logger.info(f"[L3 Trainer] 训练完成! 最终结果: {results}")
        return results

    # ------------------------------------------------------------------ #
    #  验证
    # ------------------------------------------------------------------ #

    def _validate(self, val_dataset: L3SummarizerDataset) -> float:
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
                    logger.warning(f"[L3 Trainer] 验证错误: {e}")
                    continue

        self.model.train()
        return total_loss / max(total_steps, 1)

    # ------------------------------------------------------------------ #
    #  评估
    # ------------------------------------------------------------------ #

    def evaluate(self, dataset: L3SummarizerDataset) -> dict[str, float]:
        """评估总结器性能。

        用训练后的模型对 dataset 中的样本做推理，
        然后与 ground truth 对比计算 coverage / precision / consistency。
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("[L3 Trainer] 模型未加载，无法评估。")
            return {"coverage": 0.0, "precision": 0.0, "consistency": 0.0}

        self.model.eval()

        total_coverage = 0.0
        total_precision = 0.0
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
                logger.warning(f"[L3 Trainer] 评估样本 {i} 生成失败: {e}")
                continue

            # 解析生成结果和 ground truth
            pred_keys = self._extract_profile_keys(generated)
            gt_keys = self._extract_profile_keys(target_text)

            if gt_keys:
                hits = len(pred_keys & gt_keys)
                coverage = hits / len(gt_keys)
                precision = hits / len(pred_keys) if pred_keys else 0.0
                total_coverage += coverage
                total_precision += precision
                total_count += 1

        if total_count == 0:
            return {"coverage": 0.0, "precision": 0.0, "consistency": 0.0}

        avg_coverage = total_coverage / total_count
        avg_precision = total_precision / total_count
        consistency = (
            2 * avg_coverage * avg_precision / (avg_coverage + avg_precision)
            if (avg_coverage + avg_precision) > 0
            else 0.0
        )

        metrics = {
            "coverage": avg_coverage,
            "precision": avg_precision,
            "consistency": consistency,
            "num_evaluated": total_count,
        }
        logger.info(f"[L3 Trainer] 评估结果: {metrics}")
        return metrics

    @staticmethod
    def _extract_profile_keys(text: str) -> set[str]:
        """从 JSON 文本中提取 profile entry key + category 集合。"""
        keys = set()
        try:
            entries = json.loads(text)
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and "key" in entry:
                        keys.add(f"{entry.get('category', 'unknown')}:{entry['key']}")
        except (json.JSONDecodeError, TypeError):
            # 尝试从非 JSON 文本中正则提取
            import re
            for match in re.findall(r'"key"\s*:\s*"([^"]+)"', text):
                keys.add(match)
        return keys

    # ------------------------------------------------------------------ #
    #  推理: 从 L2 对象生成 profile entries
    # ------------------------------------------------------------------ #

    def generate_profile_from_l2(
        self,
        l2_objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """使用训练后的模型从 L2 对象生成 profile entries。

        如果模型未加载，退回到规则方法。
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("[L3 Trainer] 模型未加载，退回到规则方法。")
            return self._fallback_rule_based(l2_objects)

        self.model.eval()

        objects_text = format_l2_objects_for_prompt(l2_objects)
        input_text = f"{SUMMARIZER_SYSTEM_PROMPT}\n\n{SUMMARIZER_INPUT_TEMPLATE.format(objects_text=objects_text)}"

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

            # 解析 JSON
            entries = json.loads(generated)
            if isinstance(entries, list):
                return entries
            else:
                logger.warning(f"[L3 Trainer] 生成输出不是列表: {type(entries)}")
                return self._fallback_rule_based(l2_objects)

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[L3 Trainer] 生成或解析失败: {e}, 退回到规则方法")
            return self._fallback_rule_based(l2_objects)

    @staticmethod
    def _fallback_rule_based(l2_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """退回到规则方法生成 profile entries。"""
        from src.memory.l3.summarizer import RuleBasedSummarizer
        from src.memory.l2.types import L2MemoryObject

        mem_objects = []
        for obj in l2_objects:
            mem_objects.append(L2MemoryObject(
                object_id=obj.get("object_id", "stub"),
                object_type=obj.get("object_type", "topic"),
                summary_text=obj.get("summary_text", ""),
                confidence=obj.get("confidence", 0.5),
                source_turn_ids=obj.get("source_turn_ids", []),
            ))

        summarizer = RuleBasedSummarizer()
        entries = summarizer.summarize(mem_objects)
        return [e.to_dict() for e in entries]

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
                logger.info(f"[L3 Trainer] LoRA adapter 已保存至 {save_path}")
            else:
                # 保存完整模型
                self.model.save_pretrained(str(save_path))
                logger.info(f"[L3 Trainer] 模型已保存至 {save_path}")

            # 保存 tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(save_path))

            # 保存元信息
            meta = {"epoch": epoch, "step": step, "base_model": self.config.base_model}
            with open(save_path / "training_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        except Exception as e:
            logger.error(f"[L3 Trainer] 保存检查点失败: {e}", exc_info=True)

    def load_checkpoint(self, adapter_path: str | Path) -> None:
        """加载已保存的 LoRA adapter。"""
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter 路径不存在: {adapter_path}")

        if self.model is None:
            self._load_model_and_tokenizer()

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            # 重新加载 base model 并注入 adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.to(self.device)
            logger.info(f"[L3 Trainer] LoRA adapter 已从 {adapter_path} 加载")

        except ImportError:
            logger.error("[L3 Trainer] peft 未安装，无法加载 LoRA adapter。")
        except Exception as e:
            logger.error(f"[L3 Trainer] 加载 adapter 失败: {e}", exc_info=True)
