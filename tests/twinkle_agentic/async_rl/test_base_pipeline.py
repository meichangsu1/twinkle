from types import SimpleNamespace

import pytest

from twinkle_agentic.async_rl import (
    BaseRLPipeline,
    BaseRLPipelineConfig,
    ComponentResult,
    PartitionStatus,
    PromptLoader,
    LoraContext,
    TransferQueueDataPlane,
)

from .fakes import FakeTransferQueueClient


def make_sample(i=0):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'q{i}'}],
        'input_ids': [10 + i, 20 + i],
        'labels': [-100, 20 + i],
        'attention_mask': [1, 1],
        'group_id': f'g{i}',
        'generation_idx': 0,
        'logprobs': [-0.1],
    }


class EchoRollout:

    def __init__(self):
        self.calls = []

    def __call__(self, trajectories, **kwargs):
        self.calls.append(kwargs)
        out = []
        for trajectory in trajectories:
            copied = dict(trajectory)
            copied['messages'] = list(copied.get('messages', [])) + [{'role': 'assistant', 'content': 'ok'}]
            out.append(copied)
        return out


class FakeMultiLoraModel:

    def __init__(self):
        self.forward_backward_calls = []
        self.step_calls = []
        self.save_calls = []

    def forward_backward(self, **kwargs):
        self.forward_backward_calls.append(kwargs)

    def clip_grad_and_step(self, **kwargs):
        self.step_calls.append(kwargs)

    def save(self, **kwargs):
        self.save_calls.append(kwargs)
        return SimpleNamespace(twinkle_path=f"/tmp/{kwargs['adapter_name']}-v{len(self.save_calls)}")


class FakeGRPOPipeline(BaseRLPipeline):

    def __init__(
        self,
        *,
        config,
        model,
        rollout,
        reward_registry=None,
        data_plane=None,
        receive_weights_fn=None,
    ):
        self.test_model = model
        self.test_rollout = rollout
        self.test_reward_registry = reward_registry or {}
        self.test_data_plane = data_plane or TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        self.test_receive_weights_fn = receive_weights_fn
        super().__init__(config=config)

    def build_model(self):
        return self.test_model

    def build_rollout(self):
        return self.test_rollout

    def build_reward_registry(self):
        return self.test_reward_registry

    def build_data_plane(self):
        return self.test_data_plane

    def build_receive_weights_fn(self):
        return self.test_receive_weights_fn


def test_base_pipeline_runs_one_multilora_grpo_partition():
    model = FakeMultiLoraModel()
    rollout = EchoRollout()
    received = []
    pipeline = FakeGRPOPipeline(
        config=BaseRLPipelineConfig(
            tenant_id='tenant',
            training_run_id='run',
            base_model_id='base',
            adapter_name='lora_a',
            reward_type='constant',
            max_train_steps=1,
        ),
        model=model,
        rollout=rollout,
        reward_registry={'constant': lambda trajectories, **_: [1.0 for _ in trajectories]},
        data_plane=TransferQueueDataPlane(tq_client=FakeTransferQueueClient()),
        receive_weights_fn=lambda context: received.append(context),
    )

    history = pipeline.run([make_sample(0)])

    assert history[-1]['train'] is not None
    assert model.forward_backward_calls[0]['adapter_name'] == 'lora_a'
    assert model.forward_backward_calls[0]['advantages'] == [0.0]
    assert model.forward_backward_calls[0]['old_logps'] == [[-0.1]]
    assert model.step_calls[0]['adapter_name'] == 'lora_a'
    assert model.save_calls[0]['adapter_name'] == 'lora_a'
    assert model.save_calls[0]['is_sampler'] is True
    assert received[0].policy_version == 1
    assert received[0].adapter_path == '/tmp/lora_a-v1'
    assert pipeline.data_plane.list_partitions(pipeline.context) == []


def test_base_pipeline_uses_latest_adapter_path_for_next_rollout():
    model = FakeMultiLoraModel()
    rollout = EchoRollout()
    pipeline = FakeGRPOPipeline(
        config=BaseRLPipelineConfig(
            tenant_id='tenant',
            training_run_id='run',
            base_model_id='base',
            adapter_name='lora_a',
            reward_type='constant',
            max_train_steps=1,
        ),
        model=model,
        rollout=rollout,
        reward_registry={'constant': lambda trajectories, **_: [1.0 for _ in trajectories]},
        data_plane=TransferQueueDataPlane(tq_client=FakeTransferQueueClient()),
    )

    pipeline.run([make_sample(0)], max_steps=1)
    pipeline.run([make_sample(1)], max_steps=1)

    assert rollout.calls[0]['adapter_name'] == 'lora_a'
    assert 'adapter_path' not in rollout.calls[0]
    assert rollout.calls[1]['adapter_path'] == '/tmp/lora_a-v1'


def test_base_pipeline_runs_two_lora_contexts_in_one_pipeline():
    context_a = LoraContext(
        tenant_id='tenant_a',
        training_run_id='run_a',
        base_model_id='base',
        adapter_name='lora_a',
        reward_type='constant',
    )
    context_b = LoraContext(
        tenant_id='tenant_b',
        training_run_id='run_b',
        base_model_id='base',
        adapter_name='lora_b',
        reward_type='constant',
    )
    model = FakeMultiLoraModel()
    rollout = EchoRollout()
    received = []
    pipeline = FakeGRPOPipeline(
        config=BaseRLPipelineConfig(
            lora_contexts=[context_a, context_b],
            reward_type='constant',
            default_rollout_batch_size=1,
            max_train_steps=2,
        ),
        model=model,
        rollout=rollout,
        reward_registry={'constant': lambda trajectories, **_: [1.0 for _ in trajectories]},
        data_plane=TransferQueueDataPlane(tq_client=FakeTransferQueueClient()),
        receive_weights_fn=lambda context: received.append(context),
    )

    pipeline.submit_prompt_groups([make_sample(0)], context=context_a)
    pipeline.submit_prompt_groups([make_sample(1)], context=context_b)
    history = pipeline.run_until_idle(max_steps=2)

    trained = [step['train'] for step in history if step['train'] is not None]
    assert len(trained) == 2
    assert {call['adapter_name'] for call in model.forward_backward_calls} == {'lora_a', 'lora_b'}
    assert {call['adapter_name'] for call in model.step_calls} == {'lora_a', 'lora_b'}
    assert {call['adapter_name'] for call in model.save_calls} == {'lora_a', 'lora_b'}
    assert {context.adapter_name for context in received} == {'lora_a', 'lora_b'}
    assert pipeline.data_plane.list_partitions(context_a) == []
    assert pipeline.data_plane.list_partitions(context_b) == []


def test_base_pipeline_feeds_prompts_from_prompt_loaders():
    context_a = LoraContext(
        tenant_id='tenant_a',
        training_run_id='run_a',
        base_model_id='base',
        adapter_name='lora_a',
        reward_type='constant',
    )
    context_b = LoraContext(
        tenant_id='tenant_b',
        training_run_id='run_b',
        base_model_id='base',
        adapter_name='lora_b',
        reward_type='constant',
    )
    model = FakeMultiLoraModel()
    rollout = EchoRollout()
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

    class FeederPipeline(FakeGRPOPipeline):

        def build_prompt_loaders(self):
            return [
                PromptLoader(context=context_a, dataloader=[[make_sample(0)]], rollouter=self.rollouter),
                PromptLoader(context=context_b, dataloader=[[make_sample(1)]], rollouter=self.rollouter),
            ]

    pipeline = FeederPipeline(
        config=BaseRLPipelineConfig(
            lora_contexts=[context_a, context_b],
            reward_type='constant',
            default_rollout_batch_size=1,
            max_train_steps=2,
        ),
        model=model,
        rollout=rollout,
        reward_registry={'constant': lambda trajectories, **_: [1.0 for _ in trajectories]},
        data_plane=data_plane,
    )

    history = pipeline.run_until_idle(max_steps=2)

    trained = [step['train'] for step in history if step['train'] is not None]
    assert len(trained) == 2
    assert {call['adapter_name'] for call in model.forward_backward_calls} == {'lora_a', 'lora_b'}
    assert all(loader.exhausted for loader in pipeline.prompt_loaders)


def test_component_only_pipeline_is_not_supported_by_orchestrated_runner():

    class DummyComponent:

        def __init__(self):
            self.calls = 0

        def step(self):
            if self.calls > 0:
                return None
            self.calls += 1
            return ComponentResult(component='dummy', kind='dummy', count=1)

        def is_idle(self):
            return self.calls > 0

        def shutdown(self):
            return None

    class DummyAlgorithmPipeline(BaseRLPipeline):

        def __init__(self, *args, component, data_plane, **kwargs):
            self._dummy_component = component
            self._dummy_data_plane = data_plane
            super().__init__(*args, **kwargs)

        def build_model(self):
            return object()

        def build_data_plane(self):
            return self._dummy_data_plane

        def create_roles(self):
            assert self.config.algorithm == 'dummy'
            self.components = [self._dummy_component]

    component = DummyComponent()
    pipeline = DummyAlgorithmPipeline(
        config=BaseRLPipelineConfig(
            tenant_id='tenant',
            training_run_id='run',
            base_model_id='base',
            adapter_name='lora_a',
            algorithm='dummy',
            max_train_steps=1,
        ),
        data_plane=TransferQueueDataPlane(tq_client=FakeTransferQueueClient()),
        component=component,
    )

    with pytest.raises(NotImplementedError):
        pipeline.run_until_idle(max_steps=1)

    assert component.calls == 0
    assert not hasattr(pipeline, 'rollouter')
