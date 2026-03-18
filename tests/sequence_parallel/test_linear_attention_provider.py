import unittest

import torch

from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel


class Qwen3_5GatedDeltaNet(torch.nn.Module):

    def __init__(self, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.chunk_gated_delta_rule = lambda *args, **kwargs: None
        self.recurrent_gated_delta_rule = None
        self.causal_conv1d_fn = None

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class DummyLinearBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class DummyModel(torch.nn.Module):

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner


class DummyDecoderLayer(torch.nn.Module):

    def __init__(self, layer_type: str, layer_idx: int):
        super().__init__()
        self.layer_type = layer_type
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = object()
        if layer_type == 'linear_attention':
            self.linear_attn = Qwen3_5GatedDeltaNet(layer_idx=layer_idx)
        else:
            self.self_attn = DummyLinearBlock()


class DummyQwenTextModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            DummyDecoderLayer('linear_attention', 0),
            DummyDecoderLayer('full_attention', 1),
        ])
        self.is_gradient_checkpointing = True


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

    def test_qwen35_only_disables_linear_attention_layer_gc(self):
        sp = SequenceParallel()
        sp.world_size = 2
        sp.sp_world_size = 2
        model = DummyQwenTextModel()

        sp._activate_linear_attention_model_patch(model, model)

        self.assertEqual(sp.linear_attention_model_patch_name, 'qwen35')
        self.assertEqual(
            sp.extra_kwargs['linear_attention_model_patch_disabled_gradient_checkpointing_layers'],
            [0],
        )
        self.assertFalse(model.layers[0].gradient_checkpointing)
        self.assertIsNone(model.layers[0]._gradient_checkpointing_func)
        self.assertTrue(model.layers[1].gradient_checkpointing)
        self.assertIsNotNone(model.layers[1]._gradient_checkpointing_func)


if __name__ == '__main__':
    unittest.main()
