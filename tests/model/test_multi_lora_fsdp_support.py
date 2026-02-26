# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
import torch
from peft import LoraConfig
from unittest.mock import patch

import twinkle
from twinkle.model.multi_lora import LoraTenant, MultiLora
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel

twinkle.initialize(mode='local')


class _FakeHFModel:

    def gradient_checkpointing_enable(self):
        return None


class _FakeModelCls:

    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeHFModel()


class _FakeMultiLora:

    def __init__(self, *args, **kwargs):
        pass

    def patch(self, model):
        return model

    def save_initial_weights(self):
        return None


class _FakeDeviceMesh:

    def __init__(self, fsdp_world_size: int):
        self.fsdp_world_size = fsdp_world_size


class _FakeStrategy:

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def wrap_model(self, model):
        return self._wrapped


class _FakeDTensor:

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.clone()
        self.device_mesh = 'mesh'
        self.placements = ('replicate', )
        self.device = self.tensor.device
        self.dtype = self.tensor.dtype

    def full_tensor(self):
        return self.tensor.clone()

    def to_local(self):
        return self.tensor.clone()

    def copy_(self, value):
        if isinstance(value, _FakeDTensor):
            self.tensor = value.tensor.clone()
        else:
            self.tensor = value.clone()
        return self


class _FakeParameter:

    def __init__(self, tensor: torch.Tensor):
        self.data = _FakeDTensor(tensor)


class _FakeModule:

    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())


class TestMultiLoraTransformersModelStrategy(unittest.TestCase):

    @patch('twinkle.model.transformers.multi_lora_transformers.HubOperation.download_model', return_value='dummy')
    @patch('twinkle.model.transformers.multi_lora_transformers.MultiLora', _FakeMultiLora)
    @patch('twinkle.model.transformers.multi_lora_transformers.logger')
    @patch('twinkle.model.transformers.multi_lora_transformers.AccelerateStrategy')
    @patch('twinkle.model.transformers.multi_lora_transformers.NativeFSDPStrategy')
    def test_force_native_fsdp_when_fsdp_world_size_gt_1(self, native_cls, accelerate_cls, logger_mock, _download):
        native_strategy = _FakeStrategy(wrapped=('native_wrapped_model', None))
        native_cls.return_value = native_strategy
        accelerate_cls.return_value = _FakeStrategy(wrapped='accelerate_wrapped_model')

        model = MultiLoraTransformersModel(
            model_cls=_FakeModelCls,
            model_id='dummy-model',
            device_mesh=_FakeDeviceMesh(fsdp_world_size=2),
            strategy='accelerate',
            fsdp_config={'reshard_after_forward': False},
        )

        native_cls.assert_called_once()
        accelerate_cls.assert_not_called()
        logger_mock.warning.assert_called_once()
        self.assertIs(model.strategy, native_strategy)
        self.assertEqual(model.model, 'native_wrapped_model')

    @patch('twinkle.model.transformers.multi_lora_transformers.HubOperation.download_model', return_value='dummy')
    @patch('twinkle.model.transformers.multi_lora_transformers.MultiLora', _FakeMultiLora)
    @patch('twinkle.model.transformers.multi_lora_transformers.AccelerateStrategy')
    @patch('twinkle.model.transformers.multi_lora_transformers.NativeFSDPStrategy')
    def test_keep_non_fsdp_behavior_for_non_fsdp_mesh(self, native_cls, accelerate_cls, _download):
        accelerate_strategy = _FakeStrategy(wrapped='accelerate_wrapped_model')
        accelerate_cls.return_value = accelerate_strategy
        native_cls.return_value = _FakeStrategy(wrapped=('native_wrapped_model', None))

        model = MultiLoraTransformersModel(
            model_cls=_FakeModelCls,
            model_id='dummy-model',
            device_mesh=_FakeDeviceMesh(fsdp_world_size=1),
            strategy='native_fsdp',
        )

        accelerate_cls.assert_called_once()
        native_cls.assert_not_called()
        self.assertIs(model.strategy, accelerate_strategy)
        self.assertEqual(model.model, 'accelerate_wrapped_model')


class TestMultiLoraDTensorWriteback(unittest.TestCase):

    def _build_multi_lora(self):
        max_r = 4
        tenant_config = LoraConfig(r=2, lora_alpha=8, target_modules='all-linear')
        lora = MultiLora(max_loras=1, max_r=max_r)
        lora_tenant = LoraTenant(
            index=0,
            adapter_name='lora_0',
            config=LoraConfig(r=max_r, lora_alpha=32, target_modules='all-linear'),
            tenant_adapter_name='tenant_a',
            tenant_config=tenant_config,
        )
        lora.loras = [lora_tenant]
        return lora

    @patch(
        'torch.distributed.tensor.distribute_tensor',
        side_effect=lambda tensor, *_args, **_kwargs: _FakeDTensor(tensor),
    )
    def test_set_state_dict_updates_dtensor_backed_param(self, _distribute):
        lora = self._build_multi_lora()
        param = _FakeParameter(torch.zeros(4, 3))
        lora.module = _FakeModule({'layer.lora_A.lora_0.weight': param})

        source = torch.arange(6, dtype=torch.float32).view(2, 3)
        lora.set_state_dict('tenant_a', {'layer.lora_A.weight': source})

        updated = param.data.full_tensor()
        self.assertTrue(torch.equal(updated[:2, :], source))
        self.assertTrue(torch.equal(updated[2:, :], torch.zeros(2, 3)))

    @patch(
        'torch.distributed.tensor.distribute_tensor',
        side_effect=lambda tensor, *_args, **_kwargs: _FakeDTensor(tensor),
    )
    def test_release_lora_restores_initial_and_zeros_b(self, _distribute):
        lora = self._build_multi_lora()
        param_a = _FakeParameter(torch.zeros(4, 3))
        param_b = _FakeParameter(torch.ones(3, 4))
        lora.module = _FakeModule({
            'layer.lora_A.lora_0.weight': param_a,
            'layer.lora_B.lora_0.weight': param_b,
        })
        lora.loras[0].lora_A_weights['layer.lora_A.lora_0.weight'] = torch.full((4, 3), 2.0)

        lora.release_lora('tenant_a')

        self.assertIsNone(lora.loras[0].tenant_adapter_name)
        self.assertIsNone(lora.loras[0].tenant_config)
        self.assertTrue(torch.equal(param_a.data.full_tensor(), torch.full((4, 3), 2.0)))
        self.assertTrue(torch.equal(param_b.data.full_tensor(), torch.zeros(3, 4)))


if __name__ == '__main__':
    unittest.main()
