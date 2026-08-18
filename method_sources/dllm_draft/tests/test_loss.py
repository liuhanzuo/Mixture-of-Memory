from __future__ import annotations

import unittest

import torch

from scaffold_coder.loss import (
    shifted_masked_forward_kl,
    shifted_weighted_masked_ce,
)


class WeightedLossTests(unittest.TestCase):
    def test_weighted_mean_matches_manual_computation(self) -> None:
        logits = torch.tensor(
            [
                [
                    [4.0, 1.0, 0.0],
                    [0.0, 3.0, 1.0],
                    [1.0, 0.0, 2.0],
                ]
            ]
        )
        labels = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[False, True, True]])
        weights = torch.tensor([[0.0, 0.25, 0.75]])
        loss, metrics = shifted_weighted_masked_ce(
            logits, labels, mask, weights
        )
        shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        raw = torch.nn.functional.cross_entropy(
            shifted.view(-1, 3), labels.view(-1), reduction="none"
        ).view_as(labels)
        expected = (raw * weights).sum() / weights.sum()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(int(metrics["supervised_tokens"]), 2)
        self.assertTrue(torch.allclose(metrics["weight_sum"], torch.tensor(1.0)))

    def test_empty_supervision_fails(self) -> None:
        with self.assertRaises(ValueError):
            shifted_weighted_masked_ce(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, dtype=torch.long),
                torch.zeros(1, 2, dtype=torch.bool),
                torch.zeros(1, 2),
            )

    def test_teacher_kl_is_zero_for_identical_logits(self) -> None:
        logits = torch.randn(2, 4, 7)
        mask = torch.tensor(
            [[False, True, True, False], [True, False, True, False]]
        )
        weights = mask.float()
        loss, metrics = shifted_masked_forward_kl(
            logits,
            logits.clone(),
            mask,
            weights,
            temperature=2.0,
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-6))
        self.assertEqual(int(metrics["anchored_tokens"]), 4)

    def test_teacher_kl_backpropagates_only_to_student(self) -> None:
        student = torch.randn(1, 3, 5, requires_grad=True)
        teacher = torch.randn(1, 3, 5, requires_grad=True)
        mask = torch.tensor([[False, True, True]])
        weights = mask.float()
        loss, _ = shifted_masked_forward_kl(
            student,
            teacher.detach(),
            mask,
            weights,
        )
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_topk_teacher_kl_is_zero_for_identical_logits(self) -> None:
        logits = torch.randn(1, 4, 11)
        mask = torch.tensor([[False, True, True, False]])
        loss, _ = shifted_masked_forward_kl(
            logits,
            logits.clone(),
            mask,
            mask.float(),
            topk=3,
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
