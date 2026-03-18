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


class TestLinearAttentionModelPatch(unittest.TestCase):

    def test_resolve_qwen35_model_patch(self):
        sp = SequenceParallel()
        model = DummyModel(Qwen3_5GatedDeltaNet())

        model_patch = sp._resolve_linear_attention_model_patch(model)

        self.assertIsNotNone(model_patch)
        self.assertEqual(model_patch.name, 'qwen35')

    def test_non_qwen_model_is_ignored(self):
        sp = SequenceParallel()
        model = DummyModel(DummyLinearBlock())

        model_patch = sp._resolve_linear_attention_model_patch(model)

        self.assertIsNone(model_patch)


if __name__ == '__main__':
    unittest.main()
