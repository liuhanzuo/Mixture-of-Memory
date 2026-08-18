from __future__ import annotations

from pathlib import Path
import unittest

import torch
from torch.utils.data import DistributedSampler
from transformers import AutoTokenizer

from scaffold_coder.canvas import ROLE_ID
from scaffold_coder.roles import MaskRole
from scaffold_coder.sft_dataset import (
    DreamOnDynamicDataset,
    LengthBucketDistributedSampler,
    PlainMaskedSFTDataset,
    RungMixtureSFTDataset,
    ScheduleOnlySFTDataset,
    ScaffoldBatchCollator,
    ScaffoldSFTDataset,
)
from scaffold_coder.tokenizer_utils import extend_dreamon_tokenizer
from scaffold_coder.token_row_training import (
    lexical_teacher_target_mask,
    selected_token_target_mask,
    summarize_trainable_token_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Dream-Coder-v0-Instruct-7B"
EVAL_PATH = ROOT / "data" / "scaffold_edu_v0" / "eval_data.parquet"


@unittest.skipUnless(
    MODEL_PATH.exists() and EVAL_PATH.exists(),
    "model tokenizer or normalized eval data is unavailable",
)
class ScaffoldSFTDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.tokenizer = tokenizer
        cls.dataset = ScaffoldSFTDataset(
            EVAL_PATH,
            tokenizer,
            training=False,
            max_length=1024,
        )

    def test_item_is_model_ready_and_deterministic(self) -> None:
        first = self.dataset[0]
        second = self.dataset[0]
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertEqual(first["input_ids"].ndim, 1)
        self.assertEqual(first["input_ids"].shape, first["labels"].shape)
        self.assertTrue(first["loss_mask"].any())
        self.assertTrue(torch.isfinite(first["loss_weights"]).all())
        self.assertLessEqual(int(first["length"]), 1024)

    def test_dynamic_padding_batch(self) -> None:
        items = [self.dataset[index] for index in range(4)]
        collator = ScaffoldBatchCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=1024,
            pad_to_max_length=False,
        )
        batch = collator(items)
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertEqual(
            batch["input_ids"].shape, batch["attention_mask"].shape
        )
        self.assertEqual(batch["input_ids"].shape, batch["loss_mask"].shape)
        self.assertTrue((batch["length"] <= batch["input_ids"].shape[1]).all())
        for row, length in enumerate(batch["length"].tolist()):
            self.assertTrue(
                (batch["attention_mask"][row, :length] == 1).all()
            )
            self.assertTrue(
                (batch["attention_mask"][row, length:] == 0).all()
            )

    def test_first_hundred_eval_items_fit(self) -> None:
        for index in range(100):
            item = self.dataset[index]
            self.assertLessEqual(int(item["length"]), 1024)
            self.assertTrue(item["loss_mask"].any())

    def test_training_epoch_changes_sample_but_validation_does_not(self) -> None:
        train = ScaffoldSFTDataset(
            EVAL_PATH,
            self.tokenizer,
            training=True,
            max_length=1024,
            seed=11,
        )
        train.set_epoch(0)
        first = train[0]
        train.set_epoch(1)
        second = train[0]
        self.assertFalse(torch.equal(first["input_ids"], second["input_ids"]))

        self.dataset.set_epoch(0)
        val_first = self.dataset[0]
        self.dataset.set_epoch(3)
        val_second = self.dataset[0]
        self.assertTrue(
            torch.equal(val_first["input_ids"], val_second["input_ids"])
        )

    def test_desynchronized_training_always_has_positive_supervised_mass(self) -> None:
        soft = ScaffoldSFTDataset(
            EVAL_PATH,
            self.tokenizer,
            training=True,
            max_length=1024,
            seed=19,
            desync_sigma=0.2,
        )
        for epoch in range(3):
            soft.set_epoch(epoch)
            for index in range(100):
                item = soft[index]
                self.assertTrue(item["loss_mask"].any())
                self.assertGreater(float(item["loss_weights"].sum()), 0.0)

        soft_eval = ScaffoldSFTDataset(
            EVAL_PATH,
            self.tokenizer,
            training=False,
            max_length=1024,
            seed=19,
            desync_sigma=0.2,
        )
        for index in range(100):
            item = soft_eval[index]
            self.assertTrue(item["loss_mask"].any())
            self.assertGreater(float(item["loss_weights"].sum()), 0.0)

    def test_plain_control_is_deterministic_and_uniform_response_only(self) -> None:
        plain = PlainMaskedSFTDataset(
            EVAL_PATH,
            self.tokenizer,
            training=False,
            max_length=1024,
            seed=3,
        )
        first = plain[0]
        second = plain[0]
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(first["loss_mask"].any())
        supervised_roles = first["role_ids"][first["loss_mask"]]
        self.assertTrue(
            (
                supervised_roles
                == ROLE_ID[MaskRole.TOKEN_STMT]
            ).all()
        )
        expected_sum = min(20.0, 1.0 / float(first["t"]))
        self.assertAlmostEqual(
            float(first["loss_weights"].sum()), expected_sum, places=4
        )

    def test_rung_mixture_is_deterministic_and_predicts_structure(self) -> None:
        mixture = RungMixtureSFTDataset(
            EVAL_PATH,
            self.tokenizer,
            training=False,
            max_length=1024,
            seed=29,
        )
        first = mixture[0]
        second = mixture[0]
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(first["loss_mask"].any())
        self.assertGreater(float(first["loss_weights"].sum()), 0.0)
        scaffold_ids = {
            item.token_id for item in self.dataset.registry.extensions
        }
        saw_structural_target = False
        for index in range(100):
            item = mixture[index]
            targets = set(item["labels"][item["loss_mask"]].tolist())
            saw_structural_target |= bool(scaffold_ids.intersection(targets))
            self.assertLessEqual(int(item["length"]), 1024)
        self.assertTrue(saw_structural_target)

    def test_plain_all_mask_mixture_exposes_prompt_only_generation_state(self) -> None:
        plain = PlainMaskedSFTDataset(
            EVAL_PATH,
            self.tokenizer,
            training=True,
            max_length=1024,
            seed=13,
            all_mask_probability=1.0,
        )
        item = plain[0]
        response_positions = (
            item["role_ids"] == ROLE_ID[MaskRole.TOKEN_STMT]
        )
        self.assertEqual(float(item["t"]), 1.0)
        self.assertTrue(item["loss_mask"][response_positions].all())
        self.assertTrue(
            (
                item["input_ids"][response_positions]
                == self.tokenizer.mask_token_id
            ).all()
        )
        self.assertAlmostEqual(
            float(item["loss_weights"].sum()),
            1.0,
            places=5,
        )

    def test_schedule_only_uses_ordinary_tokens_and_positive_mass(self) -> None:
        schedule_only = ScheduleOnlySFTDataset(
            EVAL_PATH,
            self.tokenizer,
            training=False,
            max_length=1024,
            seed=23,
        )
        scaffold_ids = {
            item.token_id for item in self.dataset.registry.extensions
        }
        for index in range(100):
            item = schedule_only[index]
            self.assertTrue(item["loss_mask"].any())
            self.assertGreater(float(item["loss_weights"].sum()), 0.0)
            targets = item["labels"][item["loss_mask"]].tolist()
            self.assertFalse(scaffold_ids.intersection(targets))
            self.assertLessEqual(int(item["length"]), 1024)

    def test_dynamic_dreamon_control_preserves_internal_merge_mask(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "models" / "Dream-Coder-v0-Base-7B",
            trust_remote_code=True,
            local_files_only=True,
        )
        expand_id = extend_dreamon_tokenizer(tokenizer)[0].token_id
        dataset = DreamOnDynamicDataset(
            EVAL_PATH,
            tokenizer,
            expand_token_id=expand_id,
            training=False,
            max_length=1024,
            seed=5,
        )
        found_expand = False
        found_delete = False
        found_hidden_merge = False
        items = []
        for index in range(100):
            item = dataset[index]
            items.append(item)
            found_expand |= bool(
                ((item["labels"] == expand_id) & item["loss_mask"]).any()
            )
            found_delete |= bool(
                (
                    (item["labels"] == tokenizer.eos_token_id)
                    & item["loss_mask"]
                ).any()
            )
            found_hidden_merge |= bool((item["attention_mask"] == 0).any())
            self.assertLess(int(item["length"]), 1024)
        self.assertTrue(found_expand)
        self.assertTrue(found_delete)
        self.assertTrue(found_hidden_merge)

        batch = ScaffoldBatchCollator(
            tokenizer.pad_token_id, max_length=1024
        )(items[:8])
        self.assertLess(batch["input_ids"].shape[1], 1024)
        self.assertTrue((batch["attention_mask"] == 0).any())


class LengthBucketSamplerTests(unittest.TestCase):
    def test_distributed_bucket_sampler_is_deterministic_and_disjoint(self) -> None:
        dataset = list(range(1000))
        lengths = list(range(1000))
        samplers = [
            LengthBucketDistributedSampler(
                dataset,
                lengths=lengths,
                batch_size_per_rank=4,
                bucket_multiplier=10,
                num_replicas=2,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=7,
            )
            for rank in range(2)
        ]
        rank_indices = [list(iter(sampler)) for sampler in samplers]
        self.assertFalse(set(rank_indices[0]).intersection(rank_indices[1]))
        self.assertEqual(
            len(set(rank_indices[0]) | set(rank_indices[1])),
            samplers[0].total_size,
        )
        self.assertEqual(
            rank_indices[0],
            list(iter(samplers[0])),
        )
        bucket_ranges = [
            max(batch) - min(batch)
            for indices in rank_indices
            for start in range(0, len(indices), 4)
            if len(batch := indices[start : start + 4]) == 4
        ]
        random_ranges = []
        for rank in range(2):
            random_sampler = DistributedSampler(
                dataset,
                num_replicas=2,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=7,
            )
            indices = list(iter(random_sampler))
            random_ranges.extend(
                max(batch) - min(batch)
                for start in range(0, len(indices), 4)
                if len(batch := indices[start : start + 4]) == 4
            )
        self.assertLess(
            sum(bucket_ranges) / len(bucket_ranges),
            sum(random_ranges) / len(random_ranges),
        )
        samplers[0].set_epoch(1)
        self.assertNotEqual(rank_indices[0], list(iter(samplers[0])))


class TokenRowGradientTests(unittest.TestCase):
    def test_validates_only_compact_trainable_token_deltas(self) -> None:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Module()
                self.embed_tokens.trainable_tokens_delta = torch.nn.Parameter(
                    torch.zeros(2, 5)
                )
                self.lm_head = torch.nn.Module()
                self.lm_head.trainable_tokens_delta = torch.nn.Parameter(
                    torch.zeros(2, 5)
                )
                self.frozen = torch.nn.Parameter(
                    torch.zeros(10, 5),
                    requires_grad=False,
                )

        report = summarize_trainable_token_parameters(
            FakeModel(),
            token_ids=(1, 4),
        )
        self.assertEqual(report["trainable_parameters"], 20)
        self.assertEqual(len(report["matched_parameters"]), 2)

    def test_selects_only_adapted_targets(self) -> None:
        labels = torch.tensor([[1, 2, 4], [4, 3, 1]])
        selected = selected_token_target_mask(
            labels,
            token_ids=(1, 4),
        )
        self.assertTrue(
            torch.equal(
                selected,
                torch.tensor(
                    [[True, False, True], [True, False, True]]
                ),
            )
        )

    def test_teacher_mask_excludes_edit_targets_in_lexical_roles(self) -> None:
        labels = torch.tensor([[10, 20, 30, 40]])
        roles = torch.tensor([[4, 4, 5, 2]])
        selected = lexical_teacher_target_mask(
            labels,
            roles,
            lexical_role_ids=(4, 5, 6),
            excluded_token_ids=(20, 30),
        )
        self.assertTrue(
            torch.equal(
                selected,
                torch.tensor([[True, False, False, False]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
