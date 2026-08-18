"""On-the-fly hierarchical SFT dataset and dynamic-padding collator."""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DistributedSampler, get_worker_info
from transformers import PreTrainedTokenizerBase

from .canvas import ROLE_ID, TokenRegistry
from .corruption import (
    GlobalBandSampler,
    HierarchicalBandConfig,
    HierarchicalBandSchedule,
    RungMixtureConfig,
    RungMixtureSampler,
    SampledTrainingState,
)
from .desync import DesyncConfig, DesynchronizedGlobalSampler
from .ir import (
    Body,
    ForStatement,
    FunctionDefinition,
    IfStatement,
    Module,
    WhileStatement,
)
from .renderer import SpanKind, render_with_source_map
from .serialization import module_from_dict
from .roles import MaskRole


class ScaffoldSFTDataset(Dataset):
    def __init__(
        self,
        parquet_file: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 1024,
        training: bool = True,
        seed: int = 1,
        band_config: HierarchicalBandConfig | None = None,
        desync_sigma: float = 0.0,
        max_resample_attempts: int = 8,
    ) -> None:
        self.parquet_file = str(parquet_file)
        self.dataframe = pd.read_parquet(
            self.parquet_file,
            columns=["seq_id", "prompt", "ir_json"],
        )
        self.tokenizer = tokenizer
        self.registry = TokenRegistry.build(tokenizer)
        base_sampler = GlobalBandSampler(self.registry, band_config)
        self.sampler = (
            DesynchronizedGlobalSampler(
                self.registry,
                base_sampler=base_sampler,
                config=DesyncConfig(sigma_d=desync_sigma),
            )
            if desync_sigma > 0
            else base_sampler
        )
        self.max_length = max_length
        self.training = training
        self.seed = seed
        self.max_resample_attempts = max_resample_attempts
        self._epoch = mp.Value("q", 0)

    def __len__(self) -> int:
        return len(self.dataframe)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        seq_id = int(row["seq_id"])
        module = module_from_dict(json.loads(row["ir_json"]))
        worker = get_worker_info()
        if self.training:
            worker_seed = (
                torch.initial_seed() if worker is not None else self.seed
            )
            initial_seed = (
                seq_id * 1_000_003
                + worker_seed
                + index * 97
                + self.epoch * 1_000_000_007
            ) % (2**63)
            explicit_t = None
        else:
            initial_seed = (seq_id * 1_000_003 + self.seed) % (2**63)
            # Deterministic validation still covers the entire hierarchy.
            explicit_t = ((seq_id % 997) + 0.5) / 997

        sampled: SampledTrainingState | None = None
        last_length = 0
        last_supervised = 0
        last_weight_mass = 0.0
        for attempt in range(self.max_resample_attempts):
            sampled = self.sampler.sample(
                module,
                str(row["prompt"]),
                seed=initial_seed + attempt * 104_729,
                t=explicit_t,
            )
            last_length = len(sampled.state.input_ids)
            last_supervised = sum(sampled.state.loss_mask)
            last_weight_mass = sum(sampled.loss_weights)
            if (
                last_length <= self.max_length
                and last_supervised > 0
                and last_weight_mass > 0
            ):
                break
        else:
            raise ValueError(
                f"sample seq_id={seq_id} has no valid supervised state after "
                f"{self.max_resample_attempts} attempts "
                f"(length={last_length}, max_length={self.max_length}, "
                f"supervised={last_supervised}, "
                f"weight_mass={last_weight_mass})"
            )

        tensors = sampled.to_tensors()
        tensors["seq_id"] = torch.tensor(seq_id, dtype=torch.long)
        tensors["length"] = torch.tensor(
            len(sampled.state.input_ids), dtype=torch.long
        )
        return tensors


class RungMixtureSFTDataset(Dataset):
    """Stage-wise structural SFT without the rejected depth-band schedule."""

    def __init__(
        self,
        parquet_file: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 1024,
        training: bool = True,
        seed: int = 1,
        mixture_config: RungMixtureConfig | None = None,
        max_resample_attempts: int = 8,
    ) -> None:
        self.dataframe = pd.read_parquet(
            str(parquet_file),
            columns=[
                "seq_id",
                "prompt",
                "ir_json",
                "total_tokens",
            ],
        )
        self.tokenizer = tokenizer
        self.registry = TokenRegistry.build(tokenizer)
        self.sampler = RungMixtureSampler(
            self.registry,
            mixture_config,
        )
        self.max_length = max_length
        self.training = training
        self.seed = seed
        self.max_resample_attempts = max_resample_attempts
        self._epoch = mp.Value("q", 0)

    @property
    def approximate_lengths(self) -> list[int]:
        return self.dataframe["total_tokens"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        seq_id = int(row["seq_id"])
        module = module_from_dict(json.loads(row["ir_json"]))
        worker = get_worker_info()
        worker_seed = (
            torch.initial_seed()
            if self.training and worker is not None
            else self.seed
        )
        initial_seed = (
            seq_id * 1_000_003
            + worker_seed
            + index * 97
            + (self.epoch if self.training else 0) * 1_000_000_007
        ) % (2**63)

        sampled: SampledTrainingState | None = None
        for attempt in range(self.max_resample_attempts):
            sampled = self.sampler.sample(
                module,
                str(row["prompt"]),
                seed=initial_seed + attempt * 104_729,
            )
            if (
                len(sampled.state.input_ids) <= self.max_length
                and any(sampled.state.loss_mask)
                and sum(sampled.loss_weights) > 0
            ):
                break
        else:
            raise ValueError(
                f"rung-mixture sample seq_id={seq_id} has no valid state "
                f"after {self.max_resample_attempts} attempts"
            )

        tensors = sampled.to_tensors()
        tensors["seq_id"] = torch.tensor(seq_id, dtype=torch.long)
        tensors["length"] = torch.tensor(
            len(sampled.state.input_ids),
            dtype=torch.long,
        )
        return tensors


class PlainMaskedSFTDataset(Dataset):
    """Matched plain SFT control: uniform response masking, no scaffold tokens."""

    def __init__(
        self,
        parquet_file: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 1024,
        training: bool = True,
        seed: int = 1,
        maximum_weight: float = 20.0,
        all_mask_probability: float = 0.0,
        high_noise_probability: float = 0.0,
        high_noise_min_t: float = 0.8,
    ) -> None:
        if not 0 <= all_mask_probability <= 1:
            raise ValueError("all_mask_probability must be in [0,1]")
        if not 0 <= high_noise_probability <= 1:
            raise ValueError("high_noise_probability must be in [0,1]")
        if all_mask_probability + high_noise_probability > 1:
            raise ValueError(
                "all_mask_probability + high_noise_probability must be <= 1"
            )
        if not 0 <= high_noise_min_t < 1:
            raise ValueError("high_noise_min_t must be in [0,1)")
        self.dataframe = pd.read_parquet(
            str(parquet_file),
            columns=["seq_id", "prompt", "response", "total_tokens"],
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training
        self.seed = seed
        self.maximum_weight = maximum_weight
        self.all_mask_probability = all_mask_probability
        self.high_noise_probability = high_noise_probability
        self.high_noise_min_t = high_noise_min_t
        self._epoch = mp.Value("q", 0)

    @property
    def approximate_lengths(self) -> list[int]:
        return self.dataframe["total_tokens"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        seq_id = int(row["seq_id"])
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["prompt"])}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False
        )
        response_ids = self.tokenizer.encode(
            str(row["response"]), add_special_tokens=False
        ) + [self.tokenizer.eos_token_id]
        total_length = len(prompt_ids) + len(response_ids)
        if total_length > self.max_length:
            raise ValueError(
                f"plain sample seq_id={seq_id} length={total_length} "
                f"exceeds max_length={self.max_length}"
            )

        worker = get_worker_info()
        if self.training:
            worker_seed = (
                torch.initial_seed() if worker is not None else self.seed
            )
            sample_seed = (
                seq_id * 1_000_003
                + worker_seed
                + index * 97
                + self.epoch * 1_000_000_007
            ) % (2**63)
            generator = torch.Generator().manual_seed(sample_seed)
            mixture_draw = float(
                torch.rand((), generator=generator).item()
            )
            if mixture_draw < self.all_mask_probability:
                t = 1.0
            elif mixture_draw < (
                self.all_mask_probability + self.high_noise_probability
            ):
                high_draw = float(
                    torch.rand((), generator=generator).item()
                )
                t = self.high_noise_min_t + (
                    1 - self.high_noise_min_t
                ) * high_draw
            else:
                t = float(torch.rand((), generator=generator).item())
        else:
            t = ((seq_id % 997) + 0.5) / 997
            generator = torch.Generator().manual_seed(
                (seq_id * 1_000_003 + self.seed) % (2**63)
            )
        t = max(t, 1e-6)
        response_mask = (
            torch.ones(len(response_ids), dtype=torch.bool)
            if t >= 1
            else torch.rand(len(response_ids), generator=generator) < t
        )
        if not response_mask.any():
            response_mask[
                int(torch.randint(len(response_ids), (), generator=generator))
            ] = True

        response_tensor = torch.tensor(response_ids, dtype=torch.long)
        masked_response = response_tensor.clone()
        masked_response[response_mask] = self.tokenizer.mask_token_id
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        input_ids = torch.cat([prompt_tensor, masked_response])
        labels = torch.cat([prompt_tensor, response_tensor])
        prompt_false = torch.zeros(len(prompt_ids), dtype=torch.bool)
        loss_mask = torch.cat([prompt_false, response_mask])
        supervised = int(response_mask.sum())
        base_weight = min(self.maximum_weight, 1.0 / t)
        response_weights = torch.zeros(len(response_ids), dtype=torch.float32)
        response_weights[response_mask] = base_weight / supervised
        loss_weights = torch.cat(
            [
                torch.zeros(len(prompt_ids), dtype=torch.float32),
                response_weights,
            ]
        )
        role_ids = torch.cat(
            [
                torch.full(
                    (len(prompt_ids),),
                    ROLE_ID[MaskRole.RULE],
                    dtype=torch.long,
                ),
                torch.full(
                    (len(response_ids),),
                    ROLE_ID[MaskRole.TOKEN_STMT],
                    dtype=torch.long,
                ),
            ]
        )
        attention = torch.ones(total_length, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "loss_weights": loss_weights,
            "role_ids": role_ids,
            "eligible": loss_mask.clone(),
            "attention_mask": attention,
            "position_ids": torch.arange(total_length, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.float32),
            "local_u": torch.tensor(t, dtype=torch.float32),
            "seq_id": torch.tensor(seq_id, dtype=torch.long),
            "length": torch.tensor(total_length, dtype=torch.long),
        }


class ScheduleOnlySFTDataset(Dataset):
    """Ordinary Python tokens with depth-banded masking and no meta-tokens."""

    def __init__(
        self,
        parquet_file: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 1024,
        training: bool = True,
        seed: int = 1,
        band_config: HierarchicalBandConfig | None = None,
        max_resample_attempts: int = 8,
    ) -> None:
        self.dataframe = pd.read_parquet(
            str(parquet_file),
            columns=["seq_id", "prompt", "ir_json", "total_tokens"],
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training
        self.seed = seed
        self.band_config = band_config or HierarchicalBandConfig()
        self.band_config.validate()
        self.max_resample_attempts = max_resample_attempts
        self._epoch = mp.Value("q", 0)

    @property
    def approximate_lengths(self) -> list[int]:
        return self.dataframe["total_tokens"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        seq_id = int(row["seq_id"])
        module = module_from_dict(json.loads(row["ir_json"]))
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["prompt"])}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        )
        response_ids, response_roles, depths, structural = (
            _segmented_schedule_only_response(
                module,
                self.tokenizer,
            )
        )
        response_ids.append(self.tokenizer.eos_token_id)
        response_roles.append(MaskRole.RULE)
        depths.append(0)
        structural.append(True)
        total_length = len(prompt_ids) + len(response_ids)
        if total_length > self.max_length:
            raise ValueError(
                f"schedule-only sample seq_id={seq_id} "
                f"length={total_length} exceeds {self.max_length}"
            )

        max_depth = max(depths, default=0)
        schedule = HierarchicalBandSchedule.build(
            max_depth,
            self.band_config,
        )
        worker = get_worker_info()
        worker_seed = (
            torch.initial_seed() if worker is not None else self.seed
        )
        initial_seed = (
            seq_id * 1_000_003
            + worker_seed
            + index * 97
            + (self.epoch if self.training else 0) * 1_000_000_007
        ) % (2**63)

        selected_t = 0.0
        mask = torch.zeros(len(response_ids), dtype=torch.bool)
        probabilities: list[float] = []
        bases: list[float] = []
        for attempt in range(self.max_resample_attempts):
            generator = torch.Generator().manual_seed(
                initial_seed + attempt * 104_729
            )
            if self.training:
                selected_t = float(torch.rand((), generator=generator))
            else:
                selected_t = ((seq_id % 997) + 0.5) / 997
            probabilities = []
            bases = []
            for depth, is_structural in zip(
                depths,
                structural,
                strict=True,
            ):
                band = (
                    schedule.structural_bands[depth]
                    if is_structural
                    else schedule.content_bands[depth]
                )
                u = band.clock(selected_t)
                probabilities.append(u)
                bases.append(
                    min(
                        self.band_config.maximum_weight,
                        1.0 / max(u, 1e-6),
                    )
                )
            mask = torch.rand(
                len(response_ids),
                generator=generator,
            ) < torch.tensor(probabilities)
            if mask.any():
                break
        if not mask.any():
            forced = max(
                range(len(response_ids)),
                key=lambda position: probabilities[position],
            )
            mask[forced] = True
            bases[forced] = self.band_config.maximum_weight

        response_tensor = torch.tensor(response_ids, dtype=torch.long)
        masked_response = response_tensor.clone()
        masked_response[mask] = self.tokenizer.mask_token_id
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        prompt_false = torch.zeros(len(prompt_ids), dtype=torch.bool)
        loss_mask = torch.cat([prompt_false, mask])
        response_weights = torch.zeros(
            len(response_ids),
            dtype=torch.float32,
        )
        response_weights[mask] = torch.tensor(
            [base for base, selected in zip(bases, mask, strict=True) if selected],
            dtype=torch.float32,
        )
        if self.band_config.normalize_within_sample:
            mass = float(response_weights.sum())
            selected_bases = [
                base
                for base, selected in zip(bases, mask, strict=True)
                if selected
            ]
            desired = (
                sum(selected_bases) / len(selected_bases)
                if selected_bases
                else 1.0
            )
            if mass > 0:
                response_weights *= desired / mass

        role_ids = torch.cat(
            [
                torch.full(
                    (len(prompt_ids),),
                    ROLE_ID[MaskRole.RULE],
                    dtype=torch.long,
                ),
                torch.tensor(
                    [ROLE_ID[role] for role in response_roles],
                    dtype=torch.long,
                ),
            ]
        )
        input_ids = torch.cat([prompt_tensor, masked_response])
        labels = torch.cat([prompt_tensor, response_tensor])
        loss_weights = torch.cat(
            [
                torch.zeros(len(prompt_ids), dtype=torch.float32),
                response_weights,
            ]
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "loss_weights": loss_weights,
            "role_ids": role_ids,
            "eligible": loss_mask.clone(),
            "attention_mask": torch.ones(total_length, dtype=torch.long),
            "position_ids": torch.arange(total_length, dtype=torch.long),
            "t": torch.tensor(selected_t, dtype=torch.float32),
            "local_u": torch.tensor(
                max(
                    (
                        probability
                        for probability, selected in zip(
                            probabilities,
                            mask,
                            strict=True,
                        )
                        if selected
                    ),
                    default=1e-6,
                ),
                dtype=torch.float32,
            ),
            "seq_id": torch.tensor(seq_id, dtype=torch.long),
            "length": torch.tensor(total_length, dtype=torch.long),
        }


class DreamOnDynamicDataset(Dataset):
    """DreamOn expand/EOS-delete control with dynamic, non-1024 padding."""

    def __init__(
        self,
        parquet_file: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        expand_token_id: int,
        max_length: int = 1024,
        training: bool = True,
        seed: int = 1,
        merge_probability: float = 0.5,
        static_merge_mix_probability: float = 0.5,
        max_delete: int = 64,
        maximum_weight: float = 20.0,
    ) -> None:
        self.dataframe = pd.read_parquet(
            str(parquet_file),
            columns=["seq_id", "prompt", "response"],
        )
        self.tokenizer = tokenizer
        self.expand_token_id = expand_token_id
        self.max_length = max_length
        self.training = training
        self.seed = seed
        self.merge_probability = merge_probability
        self.static_merge_mix_probability = static_merge_mix_probability
        self.max_delete = max_delete
        self.maximum_weight = maximum_weight
        self._epoch = mp.Value("q", 0)

    def __len__(self) -> int:
        return len(self.dataframe)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        seq_id = int(row["seq_id"])
        worker = get_worker_info()
        worker_seed = (
            torch.initial_seed() if worker is not None else self.seed
        )
        sample_seed = (
            seq_id * 1_000_003
            + worker_seed
            + index * 97
            + (self.epoch if self.training else 0) * 1_000_000_007
        ) % (2**63)
        generator = torch.Generator().manual_seed(sample_seed)

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["prompt"])}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False
        )
        prefix, middle, suffix = _choose_middle_lines(
            str(row["response"]), generator
        )
        prefix_ids = self.tokenizer.encode(
            prefix, add_special_tokens=False
        )
        middle_ids = self.tokenizer.encode(
            middle, add_special_tokens=False
        )
        suffix_ids = self.tokenizer.encode(
            suffix, add_special_tokens=False
        ) + [self.tokenizer.eos_token_id]

        available = (
            self.max_length
            - len(prompt_ids)
            - len(prefix_ids)
            - len(middle_ids)
            - len(suffix_ids)
        )
        max_delete = max(0, min(self.max_delete, available))
        delete_count = (
            int(torch.randint(max_delete + 1, (), generator=generator))
            if max_delete
            else 0
        )
        clean_middle = torch.tensor(
            middle_ids + [self.tokenizer.eos_token_id] * delete_count,
            dtype=torch.long,
        )
        if self.training:
            t = max(float(torch.rand((), generator=generator)), 1e-6)
        else:
            t = ((seq_id % 997) + 0.5) / 997
        mask_flags = torch.rand(
            len(clean_middle), generator=generator
        ) < t
        if delete_count:
            mask_flags[-delete_count:] = True
        if not mask_flags.any():
            mask_flags[
                int(torch.randint(len(clean_middle), (), generator=generator))
            ] = True

        masked_middle = clean_middle.clone()
        masked_middle[mask_flags] = self.tokenizer.mask_token_id
        labels_middle = clean_middle.clone()
        attention_middle = torch.ones(len(clean_middle), dtype=torch.long)
        num_masked = int(mask_flags.sum())
        if (
            float(torch.rand((), generator=generator))
            < self.static_merge_mix_probability
        ):
            merge_probability = self.merge_probability
        else:
            merge_probability = self.merge_probability * (
                1 - num_masked / max(1, len(clean_middle))
            )
        merge_draws = torch.rand(
            max(0, len(clean_middle) - 1), generator=generator
        )
        for position in range(len(clean_middle) - 1):
            if clean_middle[position] == self.tokenizer.eos_token_id:
                break
            if (
                masked_middle[position] == self.tokenizer.mask_token_id
                and masked_middle[position + 1]
                == self.tokenizer.mask_token_id
                and merge_draws[position] < merge_probability
            ):
                labels_middle[position] = self.expand_token_id
                attention_middle[position + 1] = 0

        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long)
        suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long)
        input_ids = torch.cat(
            [prompt_tensor, prefix_tensor, masked_middle, suffix_tensor]
        )
        labels = torch.cat(
            [prompt_tensor, prefix_tensor, labels_middle, suffix_tensor]
        )
        attention = torch.cat(
            [
                torch.ones(len(prompt_ids) + len(prefix_ids), dtype=torch.long),
                attention_middle,
                torch.ones(len(suffix_ids), dtype=torch.long),
            ]
        )
        loss_mask = (
            input_ids == self.tokenizer.mask_token_id
        ) & attention.bool()
        base_weight = min(self.maximum_weight, 1.0 / max(t, 1e-6))
        weights = torch.zeros(len(input_ids), dtype=torch.float32)
        delete_positions = (
            labels == self.tokenizer.eos_token_id
        ) & loss_mask
        ordinary_positions = loss_mask & ~delete_positions
        delete_supervised = int(delete_positions.sum())
        weights[ordinary_positions] = base_weight
        if delete_supervised:
            weights[delete_positions] = base_weight / delete_supervised
        current_mass = float(weights.sum())
        if current_mass:
            weights *= base_weight / current_mass

        role_ids = torch.full(
            (len(input_ids),),
            ROLE_ID[MaskRole.TOKEN_STMT],
            dtype=torch.long,
        )
        role_ids[: len(prompt_ids)] = ROLE_ID[MaskRole.RULE]
        position_ids = attention.cumsum(dim=0) - 1
        position_ids.masked_fill_(attention == 0, 1)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "loss_weights": weights,
            "role_ids": role_ids,
            "eligible": loss_mask.clone(),
            "attention_mask": attention,
            "position_ids": position_ids,
            "t": torch.tensor(t, dtype=torch.float32),
            "local_u": torch.tensor(t, dtype=torch.float32),
            "seq_id": torch.tensor(seq_id, dtype=torch.long),
            "length": torch.tensor(len(input_ids), dtype=torch.long),
        }


def _ir_node_depth_map(module: Module) -> dict[str, int]:
    mapping = {module.node_id: 0}

    def visit_body(body: Body) -> None:
        mapping[body.body_id] = body.depth
        for line in body.lines:
            mapping[line.node_id] = line.depth
            if isinstance(line, FunctionDefinition):
                visit_body(line.body)
            elif isinstance(line, IfStatement):
                visit_body(line.body)
                for clause in line.elif_clauses:
                    mapping[clause.node_id] = clause.depth
                    visit_body(clause.body)
                if line.else_body:
                    visit_body(line.else_body)
            elif isinstance(line, (ForStatement, WhileStatement)):
                visit_body(line.body)
                if line.else_body:
                    visit_body(line.else_body)

    visit_body(module.body)
    return mapping


def _segmented_schedule_only_response(
    module: Module,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[MaskRole], list[int], list[bool]]:
    rendered = render_with_source_map(module)
    depth_map = _ir_node_depth_map(module)
    token_ids: list[int] = []
    roles: list[MaskRole] = []
    depths: list[int] = []
    structural: list[bool] = []
    for span in rendered.spans:
        ids = tokenizer.encode(
            rendered.text[span.start : span.end],
            add_special_tokens=False,
        )
        token_ids.extend(ids)
        roles.extend([span.role] * len(ids))
        depths.extend([depth_map.get(span.node_id, 0)] * len(ids))
        structural.extend([span.kind is SpanKind.RULE] * len(ids))
    decoded = tokenizer.decode(token_ids)
    if decoded != rendered.text:
        raise ValueError("segmented schedule-only tokenization is not reversible")
    return token_ids, roles, depths, structural


class EpochAwareDistributedSampler(DistributedSampler):
    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class LengthBucketDistributedSampler(DistributedSampler):
    """Shuffle length-sorted global batches, then split each across ranks."""

    def __init__(
        self,
        dataset,
        *,
        lengths: list[int],
        batch_size_per_rank: int,
        bucket_multiplier: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(dataset, **kwargs)
        if len(lengths) != len(dataset):
            raise ValueError("length metadata must match dataset size")
        if batch_size_per_rank <= 0:
            raise ValueError("batch_size_per_rank must be positive")
        if bucket_multiplier <= 0:
            raise ValueError("bucket_multiplier must be positive")
        if not self.drop_last:
            raise ValueError("length bucketing currently requires drop_last")
        self.lengths = lengths
        self.batch_size_per_rank = batch_size_per_rank
        self.bucket_multiplier = bucket_multiplier
        global_batch = self.batch_size_per_rank * self.num_replicas
        self.complete_global_batches = len(dataset) // global_batch
        self.num_samples = (
            self.complete_global_batches * self.batch_size_per_rank
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shuffled = torch.randperm(
            len(self.dataset),
            generator=generator,
        ).tolist()[: self.total_size]
        global_batch = self.batch_size_per_rank * self.num_replicas
        bucket_size = global_batch * self.bucket_multiplier
        global_batches: list[list[int]] = []
        for start in range(0, len(shuffled), bucket_size):
            bucket = sorted(
                shuffled[start : start + bucket_size],
                key=self.lengths.__getitem__,
            )
            for offset in range(0, len(bucket), global_batch):
                batch = bucket[offset : offset + global_batch]
                if len(batch) == global_batch:
                    global_batches.append(batch)
        order = torch.randperm(
            len(global_batches),
            generator=generator,
        ).tolist()
        rank_start = self.rank * self.batch_size_per_rank
        rank_end = rank_start + self.batch_size_per_rank
        rank_indices: list[int] = []
        for batch_index in order:
            rank_indices.extend(
                global_batches[batch_index][rank_start:rank_end]
            )
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("length-bucket sampler produced wrong rank size")
        return iter(rank_indices)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


@dataclass(frozen=True, slots=True)
class ScaffoldBatchCollator:
    pad_token_id: int
    max_length: int = 1024
    pad_to_max_length: bool = False

    def __call__(
        self, items: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if not items:
            raise ValueError("cannot collate an empty batch")
        observed = max(int(item["length"]) for item in items)
        target_length = self.max_length if self.pad_to_max_length else observed
        if observed > self.max_length:
            raise ValueError(
                f"batch sequence length {observed} exceeds {self.max_length}"
            )

        batch: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "labels": [],
            "loss_mask": [],
            "loss_weights": [],
            "role_ids": [],
            "eligible": [],
            "attention_mask": [],
            "position_ids": [],
        }
        scalar_keys = ("t", "local_u", "seq_id", "length")
        scalars: dict[str, list[torch.Tensor]] = {
            key: [] for key in scalar_keys
        }

        for item in items:
            length = int(item["length"])
            padding = target_length - length
            batch["input_ids"].append(
                _pad(item["input_ids"], padding, self.pad_token_id)
            )
            batch["labels"].append(
                _pad(item["labels"], padding, self.pad_token_id)
            )
            batch["loss_mask"].append(
                _pad(item["loss_mask"], padding, False)
            )
            batch["loss_weights"].append(
                _pad(item["loss_weights"], padding, 0.0)
            )
            batch["role_ids"].append(
                _pad(item["role_ids"], padding, ROLE_ID[MaskRole.RULE])
            )
            batch["eligible"].append(
                _pad(item["eligible"], padding, False)
            )
            source_attention = item.get(
                "attention_mask",
                torch.ones(length, dtype=torch.long),
            )
            attention = _pad(source_attention, padding, 0)
            position = attention.cumsum(dim=0) - 1
            position.masked_fill_(attention == 0, 1)
            batch["attention_mask"].append(attention)
            batch["position_ids"].append(position)
            for key in scalar_keys:
                scalars[key].append(item[key])

        result = {
            key: torch.stack(value, dim=0) for key, value in batch.items()
        }
        result.update(
            {key: torch.stack(value, dim=0) for key, value in scalars.items()}
        )
        return result


def _pad(
    tensor: torch.Tensor, amount: int, value: int | float | bool
) -> torch.Tensor:
    if amount == 0:
        return tensor
    padding = torch.full(
        (amount,),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=0)


def _choose_middle_lines(
    response: str, generator: torch.Generator
) -> tuple[str, str, str]:
    lines = response.splitlines(keepends=True)
    if len(lines) < 3:
        return "", response, ""
    start = int(torch.randint(1, len(lines) - 1, (), generator=generator))
    end = int(
        torch.randint(start + 1, len(lines) + 1, (), generator=generator)
    )
    return "".join(lines[:start]), "".join(lines[start:end]), "".join(lines[end:])
