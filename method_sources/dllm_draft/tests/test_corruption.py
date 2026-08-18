from __future__ import annotations

from pathlib import Path
import unittest

from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.corruption import (
    GlobalBandSampler,
    HierarchicalBandConfig,
    HierarchicalBandSchedule,
    Rung,
    RungMixtureConfig,
    RungMixtureSampler,
)
from scaffold_coder.parser import parse_source
from scaffold_coder.roles import DELETE


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "Dream-Coder-v0-Instruct-7B"


@unittest.skipUnless(MODEL_PATH.exists(), "Dream tokenizer is not available")
class CorruptionSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.registry = TokenRegistry.build(tokenizer)
        cls.module = parse_source(
            "def f(xs):\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        total += x\n"
            "    return total\n"
        )

    def test_each_rung_can_be_forced(self) -> None:
        configs = [
            (Rung.ROOT_PLAN, (1.0, 0.0, 0.0)),
            (Rung.BODY_PLAN, (0.0, 1.0, 0.0)),
            (Rung.LEAF_INFILL, (0.0, 0.0, 1.0)),
        ]
        for expected, probabilities in configs:
            config = RungMixtureConfig(
                root_probability=probabilities[0],
                body_probability=probabilities[1],
                leaf_probability=probabilities[2],
                max_token_delete=2,
                max_line_delete=1,
            )
            sampled = RungMixtureSampler(self.registry, config).sample(
                self.module, "Write f.", seed=7
            )
            self.assertEqual(sampled.rung, expected)
            sampled.state.validate(self.registry)
            self.assertEqual(
                len(sampled.loss_weights), len(sampled.state.input_ids)
            )
            self.assertTrue(any(sampled.state.loss_mask))

    def test_sampling_is_seed_deterministic(self) -> None:
        sampler = RungMixtureSampler(self.registry)
        first = sampler.sample(self.module, "Write f.", seed=123)
        second = sampler.sample(self.module, "Write f.", seed=123)
        self.assertEqual(first, second)

    def test_delete_group_is_downweighted(self) -> None:
        config = RungMixtureConfig(
            root_probability=1.0,
            body_probability=0.0,
            leaf_probability=0.0,
            line_merge_probability=0.0,
            max_line_delete=4,
            weighting="unit",
            normalize_within_sample=False,
        )
        sampler = RungMixtureSampler(self.registry, config)
        found = None
        for seed in range(50):
            sampled = sampler.sample(self.module, "Write f.", seed=seed)
            delete_id = self.registry.special_id(DELETE)
            indices = [
                index
                for index, (target, supervised) in enumerate(
                    zip(
                        sampled.state.labels,
                        sampled.state.loss_mask,
                        strict=True,
                    )
                )
                if supervised and target == delete_id
            ]
            if len(indices) >= 2:
                found = (sampled, indices)
                break
        self.assertIsNotNone(found)
        sampled, indices = found
        self.assertTrue(
            all(
                sampled.loss_weights[index] == 1 / len(indices)
                for index in indices
            )
        )

    def test_content_bands_reproduce_deep_first_example(self) -> None:
        schedule = HierarchicalBandSchedule.build(
            3, HierarchicalBandConfig()
        )
        expected = {
            3: (0.00, 0.30),
            2: (0.05, 0.35),
            1: (0.10, 0.40),
            0: (0.15, 0.45),
        }
        for depth, (start, end) in expected.items():
            self.assertAlmostEqual(schedule.content_bands[depth].start, start)
            self.assertAlmostEqual(schedule.content_bands[depth].end, end)

    def test_global_band_sampler_visits_all_phases(self) -> None:
        sampler = GlobalBandSampler(self.registry)
        cases = [
            (0.20, Rung.LEAF_INFILL),
            (0.60, {Rung.BODY_PLAN, Rung.ROOT_PLAN}),
            (0.98, Rung.ROOT_PLAN),
        ]
        for t, expected in cases:
            sampled = sampler.sample(
                self.module, "Write f.", seed=5, t=t
            )
            if isinstance(expected, set):
                self.assertIn(sampled.rung, expected)
            else:
                self.assertEqual(sampled.rung, expected)
            sampled.state.validate(self.registry)
            self.assertTrue(any(sampled.state.loss_mask))
            self.assertTrue(all(weight >= 0 for weight in sampled.loss_weights))
            self.assertEqual(sampled.global_t_proxy, t)

    def test_global_band_sampling_is_deterministic(self) -> None:
        sampler = GlobalBandSampler(self.registry)
        first = sampler.sample(self.module, "Write f.", seed=99, t=0.63)
        second = sampler.sample(self.module, "Write f.", seed=99, t=0.63)
        self.assertEqual(first, second)

    def test_global_band_uses_per_position_depth_weights(self) -> None:
        sampler = GlobalBandSampler(self.registry)
        selected = None
        for seed in range(100):
            sampled = sampler.sample(
                self.module, "Write f.", seed=seed, t=0.20
            )
            if (
                sampled.metadata["base_weight_min"]
                < sampled.metadata["base_weight_max"]
            ):
                selected = sampled
                break
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(
            sum(selected.loss_weights),
            selected.metadata["base_weight"],
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
