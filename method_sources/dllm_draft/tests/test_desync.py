from __future__ import annotations

from pathlib import Path
import unittest

from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.desync import (
    DesyncConfig,
    DesynchronizedGlobalSampler,
)
from scaffold_coder.parser import parse_source
from scaffold_coder.corruption import Rung


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Dream-Coder-v0-Instruct-7B"


@unittest.skipUnless(MODEL_PATH.exists(), "Dream tokenizer is unavailable")
class DesynchronizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.registry = TokenRegistry.build(tokenizer)
        cls.module = parse_source(
            "def helper(x):\n"
            "    return x + 1\n"
            "\n"
            "def solve(xs):\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        total += helper(x)\n"
            "    return total\n"
        )

    def test_desynchronized_state_is_deterministic_and_mixed(self) -> None:
        sampler = DesynchronizedGlobalSampler(
            self.registry, config=DesyncConfig(sigma_d=0.35)
        )
        selected = None
        for seed in range(100):
            sampled = sampler.sample(
                self.module, "Write helper and solve.", seed=seed, t=0.55
            )
            rungs = {item["rung"] for item in sampled.metadata["subtrees"]}
            if len(rungs) > 1:
                selected = (seed, sampled)
                break
        self.assertIsNotNone(selected)
        seed, first = selected
        second = sampler.sample(
            self.module, "Write helper and solve.", seed=seed, t=0.55
        )
        self.assertEqual(first, second)
        self.assertEqual(first.rung, Rung.MIXED)
        first.state.validate(self.registry)
        self.assertEqual(len(first.metadata["subtrees"]), 2)
        self.assertTrue(any(first.state.loss_mask))
        self.assertEqual(len(first.loss_weights), len(first.state.input_ids))

    def test_zero_desync_matches_base_single_clock(self) -> None:
        sampler = DesynchronizedGlobalSampler(
            self.registry, config=DesyncConfig(sigma_d=0.0)
        )
        sampled = sampler.sample(
            self.module, "Write helper and solve.", seed=3, t=0.2
        )
        self.assertNotEqual(sampled.rung, Rung.MIXED)
        self.assertEqual(sampled.metadata["desync_offsets"], [0.0])


if __name__ == "__main__":
    unittest.main()

