import unittest

import torch

from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel


class Qwen3_5GatedDeltaNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.chunk_gated_delta_rule = lambda *args, **kwargs: None


class DummyLinearBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class DummyModel(torch.nn.Module):

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner


class TestLinearAttentionProvider(unittest.TestCase):

    def test_resolve_qwen35_provider(self):
        sp = SequenceParallel()
        model = DummyModel(Qwen3_5GatedDeltaNet())

        provider = sp._resolve_linear_attention_provider(model)

        self.assertIsNotNone(provider)
        self.assertEqual(provider.name, 'qwen35')

    def test_non_qwen_model_is_ignored(self):
        sp = SequenceParallel()
        model = DummyModel(DummyLinearBlock())

        provider = sp._resolve_linear_attention_provider(model)

        self.assertIsNone(provider)


if __name__ == '__main__':
    unittest.main()
