"""FSDP SFT trainer integrating the hierarchical Scaffold-Coder collator."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
from torch import optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoModel, PreTrainedModel

from src.trainer.fsdp_sft_expand_trainer import (
    FSDPSFTTrainer as UpstreamFSDPSFTTrainer,
    convert_to_regular_types,
)
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.torch_functional import get_cosine_schedule_with_warmup

from scaffold_coder.corruption import (
    HierarchicalBandConfig,
    RungMixtureConfig,
)
from scaffold_coder.loss import (
    shifted_masked_forward_kl,
    shifted_weighted_masked_ce,
)
from scaffold_coder.canvas import ROLE_ID
from scaffold_coder.roles import MaskRole
from scaffold_coder.sft_dataset import (
    EpochAwareDistributedSampler,
    LengthBucketDistributedSampler,
    DreamOnDynamicDataset,
    PlainMaskedSFTDataset,
    RungMixtureSFTDataset,
    ScheduleOnlySFTDataset,
    ScaffoldBatchCollator,
    ScaffoldSFTDataset,
)
from scaffold_coder.tokenizer_utils import (
    extend_dreamon_tokenizer,
    extend_tokenizer,
    initialize_model_token_rows,
    validate_ids_within_model,
)
from scaffold_coder.token_row_training import (
    configure_token_row_only_training,
    lexical_teacher_target_mask,
    selected_token_target_mask,
)

logger = logging.getLogger(__name__)


class ScaffoldFSDPSFTTrainer(UpstreamFSDPSFTTrainer):
    def _build_dataloader(self) -> None:
        config = self.config
        train_file = _single_path(config.data.train_files)
        val_file = _single_path(config.data.val_files)
        mode = str(config.scaffold.mode)
        if mode == "hierarchical":
            self.token_extensions = extend_tokenizer(self.tokenizer)
            band = HierarchicalBandConfig(
                **convert_to_regular_types(config.scaffold.band)
            )
            self.train_dataset = ScaffoldSFTDataset(
                train_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=True,
                seed=config.trainer.seed,
                band_config=band,
                desync_sigma=float(config.scaffold.desync_sigma),
            )
            self.val_dataset = ScaffoldSFTDataset(
                val_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=False,
                seed=config.trainer.seed,
                band_config=band,
                desync_sigma=float(config.scaffold.desync_sigma),
            )
        elif mode == "rung_mixture":
            self.token_extensions = extend_tokenizer(self.tokenizer)
            mixture = RungMixtureConfig(
                root_probability=float(
                    config.scaffold.rung.root_probability
                ),
                body_probability=float(
                    config.scaffold.rung.body_probability
                ),
                leaf_probability=float(
                    config.scaffold.rung.leaf_probability
                ),
                leaf_u_min=float(config.scaffold.rung.leaf_u_min),
                leaf_u_max=float(config.scaffold.rung.leaf_u_max),
                token_merge_base_probability=float(
                    config.scaffold.rung.token_merge_base_probability
                ),
                line_merge_probability=float(
                    config.scaffold.rung.line_merge_probability
                ),
                static_merge_mix_probability=float(
                    config.scaffold.rung.static_merge_mix_probability
                ),
                max_token_delete=int(
                    config.scaffold.rung.max_token_delete
                ),
                max_line_delete=int(
                    config.scaffold.rung.max_line_delete
                ),
                weighting=str(config.scaffold.rung.weighting),
                maximum_weight=float(
                    config.scaffold.rung.maximum_weight
                ),
                normalize_within_sample=bool(
                    config.scaffold.rung.normalize_within_sample
                ),
            )
            self.train_dataset = RungMixtureSFTDataset(
                train_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=True,
                seed=config.trainer.seed,
                mixture_config=mixture,
            )
            self.val_dataset = RungMixtureSFTDataset(
                val_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=False,
                seed=config.trainer.seed,
                mixture_config=mixture,
            )
        elif mode == "plain":
            self.token_extensions = ()
            self.train_dataset = PlainMaskedSFTDataset(
                train_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=True,
                seed=config.trainer.seed,
                maximum_weight=config.scaffold.band.maximum_weight,
                all_mask_probability=float(
                    config.scaffold.plain_all_mask_probability
                ),
                high_noise_probability=float(
                    config.scaffold.plain_high_noise_probability
                ),
                high_noise_min_t=float(
                    config.scaffold.plain_high_noise_min_t
                ),
            )
            self.val_dataset = PlainMaskedSFTDataset(
                val_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=False,
                seed=config.trainer.seed,
                maximum_weight=config.scaffold.band.maximum_weight,
                all_mask_probability=0.0,
                high_noise_probability=0.0,
                high_noise_min_t=float(
                    config.scaffold.plain_high_noise_min_t
                ),
            )
        elif mode == "schedule_only":
            self.token_extensions = ()
            band = HierarchicalBandConfig(
                **convert_to_regular_types(config.scaffold.band)
            )
            self.train_dataset = ScheduleOnlySFTDataset(
                train_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=True,
                seed=config.trainer.seed,
                band_config=band,
            )
            self.val_dataset = ScheduleOnlySFTDataset(
                val_file,
                self.tokenizer,
                max_length=config.data.max_length,
                training=False,
                seed=config.trainer.seed,
                band_config=band,
            )
        elif mode == "dreamon_dynamic":
            self.token_extensions = extend_dreamon_tokenizer(self.tokenizer)
            expand_id = self.token_extensions[0].token_id
            kwargs = {
                "expand_token_id": expand_id,
                "max_length": config.data.max_length,
                "seed": config.trainer.seed,
                "merge_probability": config.scaffold.band.token_merge_base_probability,
                "static_merge_mix_probability": config.scaffold.band.static_merge_mix_probability,
                "max_delete": config.scaffold.dreamon_max_delete,
                "maximum_weight": config.scaffold.band.maximum_weight,
            }
            self.train_dataset = DreamOnDynamicDataset(
                train_file,
                self.tokenizer,
                training=True,
                **kwargs,
            )
            self.val_dataset = DreamOnDynamicDataset(
                val_file,
                self.tokenizer,
                training=False,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown scaffold.mode={mode!r}")

        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(
                f"Scaffold dataset rank={rank}/{world_size} "
                f"train={len(self.train_dataset)} val={len(self.val_dataset)}"
            )

        collator = ScaffoldBatchCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=config.data.max_length,
            pad_to_max_length=config.data.pad_to_max_length,
        )
        workers = int(config.data.num_workers)
        loader_kwargs = {
            "num_workers": workers,
            "pin_memory": True,
            "drop_last": True,
            "collate_fn": collator,
            "persistent_workers": workers > 0,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = int(config.data.prefetch_factor)

        if bool(config.data.get("bucket_by_length", False)):
            lengths = getattr(
                self.train_dataset,
                "approximate_lengths",
                None,
            )
            if lengths is None:
                raise ValueError(
                    "bucket_by_length requires dataset length metadata"
                )
            self.train_sampler = LengthBucketDistributedSampler(
                self.train_dataset,
                lengths=lengths,
                batch_size_per_rank=int(config.data.train_batch_size),
                bucket_multiplier=int(
                    config.data.get("bucket_multiplier", 50)
                ),
                shuffle=True,
                num_replicas=world_size,
                rank=rank,
                drop_last=True,
                seed=config.trainer.seed,
            )
        else:
            self.train_sampler = EpochAwareDistributedSampler(
                self.train_dataset,
                shuffle=True,
                num_replicas=world_size,
                rank=rank,
                drop_last=True,
                seed=config.trainer.seed,
            )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            **loader_kwargs,
        )
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
            seed=config.trainer.seed,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            **loader_kwargs,
        )

    def _build_model_optimizer(self, checkpoint_path=None) -> None:
        if checkpoint_path:
            local_model_path = checkpoint_path
        elif self.resume_training and self.resume_checkpoint_path:
            local_model_path = self.resume_checkpoint_path
        else:
            local_model_path = copy_local_path_from_hdfs(
                src=self.config.model.partial_pretrain, verbose=True
            )

        log_gpu_memory_usage("Before model allocation", logger=logger)
        trust_remote_code = self.config.model.trust_remote_code
        model_config = AutoConfig.from_pretrained(
            local_model_path, trust_remote_code=trust_remote_code
        )
        if self.token_extensions:
            validate_ids_within_model(
                self.token_extensions, model_config.vocab_size
            )

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings
        )
        with init_context():
            self.model: PreTrainedModel = AutoModel.from_pretrained(
                local_model_path,
                config=model_config,
                torch_dtype=torch.float32,
                attn_implementation=self.config.model.attn_implementation,
                trust_remote_code=trust_remote_code,
            )
            if self.token_extensions and torch.distributed.get_rank() == 0:
                initialization = initialize_model_token_rows(
                    self.model, self.tokenizer, self.token_extensions
                )
                print(
                    "Scaffold token initialization:\n"
                    + json.dumps(initialization, indent=2)
                )

            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=self.model)

            if bool(self.config.model.get("token_row_only", False)):
                if not self.token_extensions:
                    raise ValueError(
                        "token_row_only requires Scaffold token extensions"
                    )
                if self.config.model.get("lora_rank", 0) > 0:
                    raise ValueError(
                        "token_row_only cannot be combined with LoRA"
                    )
                requested_notations = self.config.model.get(
                    "token_row_notations",
                    None,
                )
                if requested_notations:
                    requested = set(
                        convert_to_regular_types(requested_notations)
                    )
                    selected_extensions = tuple(
                        item
                        for item in self.token_extensions
                        if item.notation in requested
                    )
                    missing = requested - {
                        item.notation for item in selected_extensions
                    }
                    if missing:
                        raise ValueError(
                            f"unknown token-row notations: {sorted(missing)}"
                        )
                else:
                    selected_extensions = self.token_extensions
                self.token_row_target_ids = tuple(
                    item.token_id for item in selected_extensions
                )
                self.model, token_row_report = (
                    configure_token_row_only_training(
                        self.model,
                        token_ids=self.token_row_target_ids,
                    )
                )
                token_row_report["selected_notations"] = [
                    item.notation for item in selected_extensions
                ]
                if torch.distributed.get_rank() == 0:
                    print(
                        "Token-row-only training:\n"
                        + json.dumps(token_row_report, indent=2)
                    )
            elif self.config.model.get("lora_rank", 0) > 0:
                from peft import LoraConfig, TaskType, get_peft_model

                self.model.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(
                        self.config.model.target_modules
                    ),
                    "bias": "none",
                }
                modules_to_save = self.config.model.get(
                    "modules_to_save",
                    None,
                )
                if modules_to_save:
                    lora_config["modules_to_save"] = (
                        convert_to_regular_types(modules_to_save)
                    )
                self.model = get_peft_model(
                    self.model, LoraConfig(**lora_config)
                )
        teacher_kl_weight = float(
            self.config.model.get("teacher_kl_weight", 0.0)
        )
        if teacher_kl_weight < 0:
            raise ValueError("teacher_kl_weight must be non-negative")
        self.teacher_model = None
        self.teacher_kl_weight = teacher_kl_weight
        self.teacher_kl_temperature = float(
            self.config.model.get("teacher_kl_temperature", 1.0)
        )
        self.teacher_kl_topk = int(
            self.config.model.get("teacher_kl_topk", 256)
        )
        self.teacher_sharding = str(
            self.config.model.get("teacher_sharding", "replicated")
        )
        if self.teacher_kl_temperature <= 0:
            raise ValueError("teacher_kl_temperature must be positive")
        if self.teacher_kl_topk <= 0:
            raise ValueError("teacher_kl_topk must be positive")
        if self.teacher_sharding not in {"replicated", "full_shard"}:
            raise ValueError(
                "teacher_sharding must be replicated or full_shard"
            )
        if teacher_kl_weight > 0:
            if bool(self.config.model.get("token_row_only", False)):
                raise ValueError(
                    "teacher KL is not supported with token-row-only training"
                )
            requested_roles = convert_to_regular_types(
                self.config.model.get(
                    "teacher_kl_roles",
                    ["TOKEN_STMT", "TOKEN_HDR", "TOKEN_DOC"],
                )
            )
            try:
                self.teacher_kl_role_ids = tuple(
                    ROLE_ID[MaskRole(str(role))]
                    for role in requested_roles
                )
            except ValueError as exc:
                raise ValueError(
                    f"unknown teacher_kl_roles={requested_roles}"
                ) from exc
            teacher_model_path = (
                copy_local_path_from_hdfs(
                    src=self.config.model.teacher_partial_pretrain,
                    verbose=True,
                )
                if self.config.model.get(
                    "teacher_partial_pretrain", None
                )
                else local_model_path
            )
            teacher_config = AutoConfig.from_pretrained(
                teacher_model_path,
                trust_remote_code=trust_remote_code,
            )
            teacher_init_context = (
                get_init_weight_context_manager(
                    use_meta_tensor=not teacher_config.tie_word_embeddings
                )
                if self.teacher_sharding == "full_shard"
                else nullcontext
            )
            with teacher_init_context():
                self.teacher_model = AutoModel.from_pretrained(
                    teacher_model_path,
                    config=teacher_config,
                    torch_dtype=torch.float32,
                    attn_implementation=(
                        self.config.model.attn_implementation
                    ),
                    trust_remote_code=trust_remote_code,
                )
                if (
                    self.token_extensions
                    and (
                        self.teacher_sharding == "replicated"
                        or torch.distributed.get_rank() == 0
                    )
                ):
                    teacher_extensions = self.token_extensions
                    if Path(teacher_model_path).resolve() != Path(
                        local_model_path
                    ).resolve():
                        teacher_extensions = tuple(
                            replace(item, existed_before=False)
                            for item in self.token_extensions
                        )
                    initialize_model_token_rows(
                        self.teacher_model,
                        self.tokenizer,
                        teacher_extensions,
                    )
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        self.model.config.scaffold_mode = str(self.config.scaffold.mode)
        if self.token_extensions:
            token_ids = {
                item.notation: item.token_id for item in self.token_extensions
            }
            self.model.config.scaffold_token_ids = token_ids
            self.model.config.scaffold_spec_version = "v0"
            self.model.config.expand_token_id = token_ids["[expand]"]
            self.model.config.delete_token_id = token_ids.get(
                "[delete]", self.tokenizer.eos_token_id
            )

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=(
                self.config.model.get("lora_rank", 0) > 0
                or bool(self.config.model.get("token_row_only", False))
            ),
        )
        cpu_offload = (
            CPUOffload(
                offload_params=self.config.model.fsdp_config.offload_params
            )
            if self.config.model.fsdp_config.cpu_offload
            else None
        )
        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=bool(
                self.config.model.get("token_row_only", False)
            ),
        )
        self.teacher_fsdp_model = None
        if self.teacher_model is not None:
            if self.teacher_sharding == "full_shard":
                self.teacher_fsdp_model = FSDP(
                    module=self.teacher_model,
                    auto_wrap_policy=get_fsdp_wrap_policy(
                        self.teacher_model,
                        config=self.config.model.fsdp_config.wrap_policy,
                        is_lora=False,
                    ),
                    param_init_fn=init_fn,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=mixed_precision,
                    device_mesh=self.device_mesh,
                    sync_module_states=True,
                    device_id=torch.cuda.current_device(),
                    cpu_offload=cpu_offload,
                    use_orig_params=True,
                )
                self.teacher_fsdp_model.eval()
            else:
                self.teacher_model.to(
                    device=torch.cuda.current_device(),
                    dtype=torch.bfloat16,
                )
                self.teacher_model.eval()
        trainable_parameters = [
            parameter
            for parameter in self.fsdp_model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("optimizer has no trainable parameters")
        self.optimizer = optim.AdamW(
            trainable_parameters,
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = (
            self.steps_per_epoch * self.config.trainer.total_epochs
        )
        num_warmup_steps = int(
            self.total_steps * self.config.optim.warmup_steps_ratio
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        if self.device_mesh.get_rank() == 0:
            print(
                f"steps/epoch={self.steps_per_epoch} "
                f"epochs={self.config.trainer.total_epochs} "
                f"total_steps={self.total_steps}"
            )

    def _compute_loss_and_backward(self, batch, do_backward=True):
        use_sp = (
            self.use_remove_padding
            and self.config.ulysses_sequence_parallel_size > 1
        )
        if use_sp:
            raise NotImplementedError(
                "Scaffold sequence parallelism is not enabled in v0"
            )

        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        attention_mask = batch["attention_mask"].cuda().bool()
        position_ids = batch["position_ids"].cuda()
        loss_mask = batch["loss_mask"].cuda().bool()
        loss_weights = batch["loss_weights"].cuda().float()
        role_ids = batch["role_ids"].cuda()
        if bool(self.config.model.get("token_row_only", False)):
            selected_targets = selected_token_target_mask(
                labels,
                token_ids=self.token_row_target_ids,
            )
            loss_mask &= selected_targets
            loss_weights *= selected_targets
            if not loss_mask.any():
                raise ValueError(
                    "token-row micro-batch has no selected target"
                )

        context = nullcontext()
        with context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pairwise_attention = torch.logical_and(
                    attention_mask.unsqueeze(1).unsqueeze(-2),
                    attention_mask.unsqueeze(1).unsqueeze(-1),
                )
                output = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=pairwise_attention,
                    position_ids=position_ids,
                    use_cache=False,
                )
                loss, _ = shifted_weighted_masked_ce(
                    output.logits,
                    labels,
                    loss_mask,
                    loss_weights,
                )
                teacher = self.teacher_fsdp_model or self.teacher_model
                if teacher is not None:
                    lexical_mask = loss_mask & lexical_teacher_target_mask(
                        labels,
                        role_ids,
                        lexical_role_ids=self.teacher_kl_role_ids,
                        excluded_token_ids=tuple(
                            item.token_id for item in self.token_extensions
                        ),
                    )
                    if lexical_mask.any():
                        with torch.no_grad():
                            teacher_output = teacher(
                                input_ids=input_ids,
                                attention_mask=pairwise_attention,
                                position_ids=position_ids,
                                use_cache=False,
                            )
                        kl_loss, _ = shifted_masked_forward_kl(
                            output.logits,
                            teacher_output.logits,
                            lexical_mask,
                            loss_weights,
                            temperature=self.teacher_kl_temperature,
                            topk=self.teacher_kl_topk,
                        )
                        loss = loss + self.teacher_kl_weight * kl_loss
        if do_backward:
            loss.backward()
        return loss

    def training_step(self, batch):
        """Run one optimizer step with optional low-overhead telemetry."""

        step_index = getattr(self, "_scaffold_step_index", 0)
        profile_every = int(
            self.config.trainer.get("profile_every_steps", 0)
        )
        profile = profile_every > 0 and step_index % profile_every == 0
        if profile:
            local_examples = int(batch["input_ids"].shape[0])
            local_tokens = int(batch["attention_mask"].sum().item())
            local_supervised = int(batch["loss_mask"].sum().item())
            local_padded_tokens = int(batch["input_ids"].numel())
            local_sequence_length = int(batch["input_ids"].shape[1])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()

        metrics = super().training_step(batch)
        self._scaffold_step_index = step_index + 1
        if not profile:
            return metrics

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        device = torch.device("cuda", torch.cuda.current_device())
        elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
        count_tensor = torch.tensor(
            [
                local_examples,
                local_tokens,
                local_supervised,
                local_padded_tokens,
            ],
            device=device,
            dtype=torch.float64,
        )
        sequence_tensor = torch.tensor(
            local_sequence_length,
            device=device,
            dtype=torch.float64,
        )
        memory_tensor = torch.tensor(
            [
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
            ],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(
            elapsed_tensor, op=torch.distributed.ReduceOp.MAX
        )
        torch.distributed.all_reduce(
            count_tensor, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            memory_tensor, op=torch.distributed.ReduceOp.MAX
        )
        torch.distributed.all_reduce(
            sequence_tensor, op=torch.distributed.ReduceOp.MAX
        )

        step_seconds = float(elapsed_tensor.item())
        (
            global_examples,
            global_tokens,
            global_supervised,
            global_padded_tokens,
        ) = (
            float(value) for value in count_tensor.tolist()
        )
        profile_metrics = {
            "train/step_seconds": step_seconds,
            "train/examples_per_second": global_examples / step_seconds,
            "train/nonpadding_tokens_per_second": (
                global_tokens / step_seconds
            ),
            "train/padded_tokens_per_second": (
                global_padded_tokens / step_seconds
            ),
            "train/padding_fraction": (
                1.0 - global_tokens / global_padded_tokens
            ),
            "train/supervised_tokens_per_second": (
                global_supervised / step_seconds
            ),
            "train/maximum_sequence_length": float(
                sequence_tensor.item()
            ),
            "train/peak_allocated_gib": (
                float(memory_tensor[0].item()) / 2**30
            ),
            "train/peak_reserved_gib": (
                float(memory_tensor[1].item()) / 2**30
            ),
        }
        metrics.update(profile_metrics)
        self._write_profile_record(
            step_index=step_index,
            local_batch_size=local_examples,
            metrics=profile_metrics,
        )
        return metrics

    def _write_profile_record(
        self,
        *,
        step_index: int,
        local_batch_size: int,
        metrics: dict[str, float],
    ) -> None:
        if self.device_mesh.get_rank() != 0:
            return
        path_value = self.config.trainer.get("metrics_jsonl", None)
        if not path_value:
            return
        path = Path(str(path_value))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_step": step_index,
            "world_size": torch.distributed.get_world_size(),
            "global_batch_size": (
                local_batch_size * torch.distributed.get_world_size()
            ),
            "micro_batch_size_per_gpu": int(
                self.config.data.micro_batch_size_per_gpu
            ),
            **metrics,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def save_checkpoint(self, step):
        if bool(self.config.trainer.get("skip_checkpoint", False)):
            if self.device_mesh.get_rank() == 0:
                print(f"Skipping checkpoint at step {step} (benchmark mode)")
            return
        super().save_checkpoint(step)
        if self.device_mesh.get_rank() == 0:
            path = Path(self.config.trainer.default_local_dir) / f"global_step_{step}"
            source = Path(str(self.config.model.partial_pretrain))
            for name in (
                "configuration_dream.py",
                "generation_utils.py",
                "modeling_dream.py",
                "tokenization_dream.py",
            ):
                source_file = source / name
                if source_file.exists():
                    shutil.copy2(source_file, path / name)
            manifest = {
                item.notation: {
                    "physical": item.physical,
                    "token_id": item.token_id,
                    "existed_before": item.existed_before,
                }
                for item in self.token_extensions
            }
            (path / "scaffold_tokens.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )


def _single_path(value) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("v0 trainer expects one parquet path per split")
        return str(value[0])
    return str(value)


def _command_output(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def _write_resolved_training_config(
    config,
    *,
    rank: int,
    world_size: int,
) -> None:
    if rank != 0:
        return
    root = Path(__file__).resolve().parents[2]
    output = Path(str(config.trainer.default_local_dir))
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "rank": rank,
        "world_size": world_size,
        "argv": sys.argv,
        "config": convert_to_regular_types(config),
        "git": {
            "commit": _command_output(["git", "rev-parse", "HEAD"], root),
            "status": _command_output(
                [
                    "git",
                    "status",
                    "--short",
                    "--",
                    ".",
                    ":(exclude)ops/queue.tsv",
                    ":(exclude)ops/history.tsv",
                ],
                root,
            ),
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    path = output / "resolved_training_config.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@hydra.main(
    config_path="config",
    config_name="scaffold_sft",
    version_base=None,
)
def main(config):
    _, rank, world_size = initialize_global_process_group()
    rank_seed = int(config.trainer.seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    trainer = ScaffoldFSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
    )
    _write_resolved_training_config(
        config,
        rank=rank,
        world_size=world_size,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
