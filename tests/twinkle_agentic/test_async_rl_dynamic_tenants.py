from __future__ import annotations

import asyncio
from types import SimpleNamespace

from twinkle_agentic.async_rl.context_manager import ContextStatus, LoraContextManager
from twinkle_agentic.async_rl.runtime import AsyncRLRuntime
from twinkle_agentic.async_rl.types import LoraContext
from twinkle_agentic.async_rl.workers import RolloutWorker
from twinkle_agentic.async_rl.scheduler import SchedulerConfig


class _RemoteMethod:

    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _LocalActor:

    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return _RemoteMethod(getattr(self._target, name))


def test_context_can_be_reserved_activated_drained_and_removed():
    manager = LoraContextManager()
    context = LoraContext('tenant', 'run', 'model', 'adapter')

    manager.reserve_context(context)
    assert manager.context_status(context) is ContextStatus.ADDING
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is None

    manager.activate_context(context, adapter_path='initial')
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    assert admission is not None

    manager.request_context_drain(context)
    assert manager.context_status(context) is ContextStatus.DRAINING
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is None
    assert not manager.context_is_drained(context)

    manager.on_partition_training_started(admission)
    manager.on_partition_trained(admission, adapter_path='next')
    manager.on_partition_cleared(admission)
    assert manager.context_is_drained(context)
    assert manager.context_adapter_paths(context) == ['initial', 'next']

    manager.unregister_context(context)
    assert manager.list_context_snapshots() == []


def test_context_level_max_steps_finishes_without_stopping_other_contexts():
    manager = LoraContextManager()
    limited = LoraContext('tenant', 'limited', 'model', 'limited')
    unlimited = LoraContext('tenant', 'unlimited', 'model', 'unlimited')
    manager.reserve_context(limited, max_steps=1)
    manager.activate_context(limited)
    manager.register_context(unlimited)

    admission = manager.request_rollout_partition(limited, target_groups=1, num_generations=2)
    manager.on_partition_training_started(admission)
    manager.on_partition_trained(admission, adapter_path='limited-v1')
    manager.on_partition_cleared(admission)

    assert manager.context_status(limited) is ContextStatus.FINISHED
    assert manager.request_rollout_partition(limited, target_groups=1, num_generations=2) is None
    assert manager.request_rollout_partition(unlimited, target_groups=1, num_generations=2) is not None


def test_persistent_rollout_worker_accepts_context_after_start():
    context = LoraContext('tenant', 'run', 'model', 'adapter')
    manager = LoraContextManager()
    manager.register_context(context)

    class DataPlane:

        async def prepare_rollout_partition(self, admission, prompts, sampling_params):
            return type('Prepared', (), {
                'groups': (),
                'sampling_params': sampling_params,
            })()

    class Sampler:

        def __init__(self, loop):
            self.loop = loop
            self.submitted = asyncio.Event()

        def sample(self, groups, sampling_params, allow_partial):
            self.loop.call_soon_threadsafe(self.submitted.set)

    async def run():
        sampler = Sampler(asyncio.get_running_loop())
        worker = RolloutWorker(
            context_manager=_LocalActor(manager),
            data_plane=DataPlane(),
            sampler=sampler,
            prompt_batches={},
            rollout_config={},
            scheduler=SchedulerConfig(),
            persistent=True,
            idle_delay_s=0.001,
        )
        await worker.start()
        await asyncio.sleep(0.01)
        assert (await worker.get_service_state())['running']

        await worker.register_context(
            context,
            [[{'input_ids': [1]}]],
            {
                'context': context,
                'batch_size': 1,
                'num_generations': 2,
                'sampling_params': {},
            },
        )
        await asyncio.wait_for(sampler.submitted.wait(), timeout=1)
        await worker.unregister_context(context)
        assert (await worker.get_service_state())['running']
        await worker.stop()

    asyncio.run(run())


def test_runtime_rolls_back_partial_registration():
    manager = LoraContextManager()

    class Model:

        def __init__(self):
            self.removed = []

        def add_adapter_to_model(self, adapter_name):
            self.adapter_name = adapter_name

        def save(self, *args, **kwargs):
            return '/tmp/initial'

        def remove_adapter(self, adapter_name):
            self.removed.append(adapter_name)

    class Sampler:

        def __init__(self):
            self.rewards = {}

        def register_reward(self, key, reward):
            self.rewards[key] = reward

        def unregister_reward(self, key):
            self.rewards.pop(key, None)

        def unload_lora_paths(self, paths):
            pass

    class RolloutRegistry:

        def __init__(self):
            self.keys = set()

        def register_context(self, context, prompt_source, rollout_config):
            self.keys.add(context.key)

        def unregister_context(self, context):
            self.keys.discard(context.key if isinstance(context, str) else context.key)

    class TrainerRegistry:

        def register_context(self, *args, **kwargs):
            raise RuntimeError('trainer registration failed')

        def unregister_context(self, *args, **kwargs):
            pass

    class Runtime(AsyncRLRuntime):

        def _prepare_tenant(self, item, context):
            return {
                'prompt_source': [],
                'rollout_config': {'context': context},
                'train_config': SimpleNamespace(mini_batch_size=2),
                'reward': object(),
                'evaluation_config': None,
                'evaluation_reward': None,
            }

        def _add_model_adapter(self, item, context):
            self.model.add_adapter_to_model(context.adapter_name)

        def _configure_model(self, item, context):
            pass

    model = Model()
    sampler = Sampler()
    rollout = RolloutRegistry()
    pipeline = SimpleNamespace(
        context_manager=_LocalActor(manager),
        model=model,
        sampler=sampler,
        rollout_worker=_LocalActor(rollout),
        advantage_worker=object(),
        trainer_worker=_LocalActor(TrainerRegistry()),
    )
    runtime = Runtime(
        pipeline,
        {
            'runtime': {
                'model_id': 'model',
                'output_dir': '/tmp',
            },
        },
    )

    async def run():
        try:
            await runtime.add_tenant({
                'tenant_id': 'tenant',
                'training_run_id': 'run',
                'adapter_name': 'adapter',
                'train': {},
            })
        except RuntimeError as exc:
            assert 'trainer registration failed' in str(exc)
        else:
            raise AssertionError('expected dynamic registration to fail')

    asyncio.run(run())
    assert manager.list_context_snapshots() == []
    assert rollout.keys == set()
    assert sampler.rewards == {}
    assert model.removed == ['adapter']
