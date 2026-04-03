import torch

from rlm.clients.transformers_local import TransformersClient


class FakeBatch(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "templated prompt"

    def __call__(self, texts, return_tensors="pt"):
        return FakeBatch(torch.tensor([[10, 11, 12]]))

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "<think>hidden</think>{\"action_type\":\"STOP\",\"answer\":\"done\"}"


class FakeModel:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        return torch.tensor([[10, 11, 12, 20, 21]])


def test_transformers_client_completion_tracks_usage():
    client = TransformersClient(
        model_name="fake-transformer",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        enable_thinking=False,
    )

    response = client.completion("Return JSON")

    assert response == '{"action_type":"STOP","answer":"done"}'
    summary = client.get_usage_summary()
    assert summary.model_usage_summaries["fake-transformer"].total_calls == 1
    assert summary.model_usage_summaries["fake-transformer"].total_input_tokens == 3
    assert summary.model_usage_summaries["fake-transformer"].total_output_tokens == 2
