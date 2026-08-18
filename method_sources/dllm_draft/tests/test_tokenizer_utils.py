from __future__ import annotations

import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scaffold_coder.tokenizer_utils import (
    edit_source_token_ids,
    TokenExtension,
    extend_dreamon_tokenizer,
    initialize_model_token_rows,
    validate_ids_within_model,
)


class _EmbeddingModel:
    def __init__(self) -> None:
        self.input = torch.nn.Embedding(8, 3)
        self.output = torch.nn.Linear(3, 8, bias=False)
        with torch.no_grad():
            self.input.weight.copy_(torch.arange(24).reshape(8, 3))
            self.output.weight.copy_(torch.arange(24, 48).reshape(8, 3))

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output


class _Tokenizer:
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "expand": [1],
            "mask": [2],
            "delete": [3],
            "remove": [4],
        }
        return mapping[text]


class TokenizerUtilityTests(unittest.TestCase):
    def test_edit_source_ids_are_ordinary_and_unique(self) -> None:
        tokenizer = _Tokenizer()
        ids = edit_source_token_ids(tokenizer)
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids).intersection(tokenizer.all_special_ids))

    def test_reserved_rows_are_initialized_in_both_matrices(self) -> None:
        model = _EmbeddingModel()
        tokenizer = _Tokenizer()
        before_existing_in = model.input.weight[5].clone()
        before_existing_out = model.output.weight[5].clone()
        extensions = [
            TokenExtension("[expand]", "<|expand|>", 5, True),
            TokenExtension("[delete]", "<|sc_delete|>", 6, False),
        ]
        report = initialize_model_token_rows(model, tokenizer, extensions)
        self.assertTrue(torch.equal(model.input.weight[5], before_existing_in))
        self.assertTrue(torch.equal(model.output.weight[5], before_existing_out))
        self.assertTrue(
            torch.equal(
                model.input.weight[6],
                model.input.weight[[3, 4]].mean(dim=0),
            )
        )
        self.assertTrue(
            torch.equal(
                model.output.weight[6],
                model.output.weight[[3, 4]].mean(dim=0),
            )
        )
        self.assertTrue(report["[expand]"]["preserved"])
        self.assertFalse(report["[delete]"]["preserved"])

    def test_ids_must_fit_reserved_rows(self) -> None:
        validate_ids_within_model(
            [TokenExtension("[FUNC]", "<|sc_func|>", 7, False)],
            vocab_size=8,
        )
        with self.assertRaises(Exception):
            validate_ids_within_model(
                [TokenExtension("[FUNC]", "<|sc_func|>", 8, False)],
                vocab_size=8,
            )

    def test_dreamon_extension_uses_reserved_151667(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "Dream-Coder-v0-Base-7B"
        )
        if not model_path.exists():
            self.skipTest("Dream-Coder Base tokenizer unavailable")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        extensions = extend_dreamon_tokenizer(tokenizer)
        self.assertEqual(len(extensions), 1)
        self.assertEqual(extensions[0].token_id, 151667)
        self.assertEqual(len(tokenizer), 151668)


if __name__ == "__main__":
    unittest.main()
