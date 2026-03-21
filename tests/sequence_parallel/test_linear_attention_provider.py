import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel


class DummyPatchableModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bound_sequence_parallel = None
        self.bound_impl_name = None

    def twinkle_bind_sequence_parallel(self, sequence_parallel, *, impl_name=None):
        self.bound_sequence_parallel = sequence_parallel
        self.bound_impl_name = impl_name


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2)
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.patchable = DummyPatchableModule()
        self.other = torch.nn.Linear(2, 2)


class TestSequenceParallelRuntimeBinding(unittest.TestCase):

    def test_bind_sequence_parallel_modules_binds_patchable_layers(self):
        sp = SequenceParallel()
        model = DummyModel()

        sp._bind_sequence_parallel_modules(model)

        self.assertIs(model.patchable.bound_sequence_parallel, sp)
        self.assertIsNone(model.patchable.bound_impl_name)

    def test_bind_sequence_parallel_modules_ignores_non_patchable_layers(self):
        sp = SequenceParallel()
        model = DummyModel()

        sp._bind_sequence_parallel_modules(model)

        self.assertFalse(hasattr(model.other, 'bound_sequence_parallel'))

    def test_prepare_uses_generic_runtime_binding(self):
        sp = SequenceParallel()
        model = DummyModel()
        tokenizer = object()
        global_inited = SequenceParallel._global_inited
        SequenceParallel._global_inited = True
        try:
            with mock.patch.object(sp, '_prepare_forward_hook') as prepare_forward_hook, mock.patch.object(
                    SequenceParallel, '_is_moe_model', return_value=False):
                sp.prepare(sp_size=2, model=model, tokenizer=tokenizer, device_mesh=None)
        finally:
            SequenceParallel._global_inited = global_inited

        prepare_forward_hook.assert_called_once()
        self.assertIs(model.patchable.bound_sequence_parallel, sp)
        self.assertEqual(sp.world_size, 2)
        self.assertIs(sp.tokenizer, tokenizer)

    def test_should_build_causal_mask_defaults_true(self):
        sp = SequenceParallel()

        self.assertTrue(sp._should_build_causal_mask())


if __name__ == '__main__':
    unittest.main()
