from types import SimpleNamespace

import pytest
import torch

from scripts.eval_qcmem_babilong import qcmem_generate


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 151645

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(map(str, token_ids))


class _FakeQCMem:
    device = torch.device("cpu")

    def __init__(self, stop_id):
        self.model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
        self.stop_id = stop_id
        self.read_calls = 0

    @staticmethod
    def write_chunk(token_ids):
        length = len(token_ids)
        return torch.zeros((1, length, 1))

    def read(self, sink_hj, selected_hj, query_hj):
        logits = torch.full((1, 1, 151646), -100.0)
        if self.read_calls == 0:
            # Both official EOS tokens are preferred at step 0 and must be
            # masked, leaving token 999 as the selected first token.
            logits[0, 0, 151643] = 10.0
            logits[0, 0, 151645] = 9.0
            logits[0, 0, 999] = 8.0
        elif self.read_calls == 1:
            logits[0, 0, self.stop_id] = 10.0
            logits[0, 0, 1000] = 9.0
        else:
            # Reaching this branch means the EOS token failed to stop decode.
            logits[0, 0, 1001] = 10.0
        self.read_calls += 1
        return logits


@pytest.mark.parametrize("stop_id", [151645, 151643])
def test_qcmem_generate_honors_every_generation_config_eos(stop_id):
    qc = _FakeQCMem(stop_id)
    prediction = qcmem_generate(
        qc=qc,
        tokenizer=_FakeTokenizer(),
        input_ids=torch.tensor([[10, 11]]),
        chunk_size=1,
        max_new_tokens=4,
        selector="recency",
        topk=1,
        sink_tokens="bos",
    )

    assert prediction == "999"
    assert qc.read_calls == 2
