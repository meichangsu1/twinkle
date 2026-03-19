import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from twinkle.model.transformers.strategy.linear_attention.qwen35 import Qwen35LinearAttentionSPModelPatch
from twinkle.model.transformers.strategy.linear_attention.qwen35_strict import Qwen35StrictFullSeqHelper
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

    def test_qwen35_head_parallel_keeps_gc_enabled(self):
        sp = SequenceParallel()
        sp.world_size = 2
        sp.sp_world_size = 2
        model = DummyQwenTextModel()

        with mock.patch.dict('os.environ', {'QWEN35_SP_LINEAR_HEAD_PARALLEL': '1'}, clear=False):
            sp._activate_linear_attention_model_patch(model, model)

        self.assertEqual(sp.linear_attention_model_patch_name, 'qwen35')
        self.assertEqual(sp.extra_kwargs['linear_attention_model_patch_impl'], 'head_parallel')
        self.assertEqual(
            sp.extra_kwargs['linear_attention_model_patch_disabled_gradient_checkpointing_layers'],
            [],
        )
        self.assertTrue(model.layers[0].gradient_checkpointing)
        self.assertIsNotNone(model.layers[0]._gradient_checkpointing_func)

    def test_qwen35_strict_helper_is_opt_in_and_rejects_fsdp(self):
        with mock.patch.dict('os.environ', {'QWEN35_SP_LINEAR_STRICT': '1'}):
            helper = Qwen35StrictFullSeqHelper()
        self.assertTrue(helper.enabled)

        sequence_parallel = SimpleNamespace(device_mesh=SimpleNamespace(fsdp_world_size=2))
        with self.assertRaisesRegex(RuntimeError, 'only supported without FSDP sharding'):
            helper.validate_runtime(sequence_parallel)

    def test_qwen35_impl_flags_are_mutually_exclusive(self):
        with mock.patch.dict(
            'os.environ',
            {
                'QWEN35_SP_LINEAR_HEAD_PARALLEL': '1',
                'QWEN35_SP_LINEAR_STRICT': '1',
            },
            clear=False,
        ):
            patch = Qwen35LinearAttentionSPModelPatch()
            with self.assertRaisesRegex(RuntimeError, 'mutually exclusive'):
                patch._resolve_impl_name()


if __name__ == '__main__':
    unittest.main()
