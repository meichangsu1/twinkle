# Copyright (c) ModelScope Contributors. All rights reserved.
"""E2E integration test for TransferQueueDataPlane with gsm8k-like data.

Runs the full BaseRLPipeline lifecycle without GPU:
  rollout -> reward -> advantage -> train -> weight sync -> clear

Uses FakeTransferQueueClient and mock model/rollout to verify the complete
data flow through TransferQueueDataPlane.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from twinkle_agentic.async_rl import (
    BaseRLPipeline,
    BaseRLPipelineConfig,
    PartitionStatus,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)

from .fakes import FakeTransferQueueClient


def make_gsm8k_sample(i: int) -> Dict[str, Any]:
    return {
        'sample_id': f'gsm8k_{i}',
        'messages': [
            {'role': 'user', 'content': f'Question: Janet has {i + 3} apples and gives away {i + 1}. How many does she have left?'},
        ],
        'group_id': f'group_{i}',
        'generation_idx': 0,
        'old_logps': [-0.1, -0.2, -0.3],
        'user_data': {'answer': str(2)},
    }


def make_gsm8k_reward_fn():
    def gsm8k_reward(trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            has_boxed = any('\\boxed' in str(m.get('content', '')) for m in messages if m.get('role') == 'assistant')
            rewards.append(1.0 if has_boxed else 0.0)
        return rewards
    return gsm8k_reward


class FakeMultiLoraModel:
    def __init__(self):
        self.forward_backward_calls: list = []
        self.step_calls: list = []
        self.save_calls: list = []

    def forward_backward(self, **kwargs):
        self.forward_backward_calls.append(kwargs)

    def clip_grad_and_step(self, **kwargs):
        self.step_calls.append(kwargs)

    def save(self, **kwargs):
        self.save_calls.append(kwargs)
        return SimpleNamespace(twinkle_path=f"/tmp/{kwargs['adapter_name']}-v{len(self.save_calls)}")


class EchoRolloutWithBoxed:
    """Mock rollout that appends an assistant message with \\boxed{} answer."""

    def __init__(self):
        self.calls = []

    def __call__(self, trajectories, **kwargs):
        self.calls.append(kwargs)
        out = []
        for traj in trajectories:
            copied = dict(traj)
            copied['messages'] = list(copied.get('messages', [])) + [
                {'role': 'assistant', 'content': 'Let me think... The answer is \\boxed{2}.'},
            ]
            out.append(copied)
        return out


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


class TestE2EGSM8KIntegration:
    """End-to-end test: BaseRLPipeline + TransferQueueDataPlane with gsm8k-like data."""

    def test_full_pipeline_runs_multiple_partitions(self):
        model = FakeMultiLoraModel()
        rollout = EchoRolloutWithBoxed()
        received = []

        data_plane = TransferQueueDataPlane(
            tq_client=FakeTransferQueueClient(),
            tq_config=TransferQueueRuntimeConfig(max_rows=1000),
        )

        config = BaseRLPipelineConfig(
            tenant_id='test_tenant',
            training_run_id='gsm8k_grpo',
            base_model_id='Qwen/Qwen3.5-0.8B',
            adapter_name='gsm8k_lora',
            reward_type='gsm8k',
            loss_type='grpo',
            algorithm='grpo',
            max_staleness=0,
            target_groups_per_partition=1,
            max_train_partitions=3,
        )

        pipeline = FakeGRPOPipeline(
            config=config,
            model=model,
            rollout=rollout,
            reward_registry={'gsm8k': make_gsm8k_reward_fn()},
            data_plane=data_plane,
            receive_weights_fn=lambda ctx: received.append(ctx),
        )

        samples = [make_gsm8k_sample(i) for i in range(3)]
        history = pipeline.run(samples, max_steps=3)

        trained = sum(1 for step in history if step['train'] is not None)
        assert trained == 3, f'Expected 3 trained partitions, got {trained}'

        assert len(model.forward_backward_calls) == 3
        assert all(call['adapter_name'] == 'gsm8k_lora' for call in model.forward_backward_calls)
        assert all(call['old_logps'] == [[-0.1, -0.2, -0.3]] for call in model.forward_backward_calls)

        assert len(model.save_calls) == 3
        assert all(call['adapter_name'] == 'gsm8k_lora' for call in model.save_calls)

        assert len(received) == 3
        assert received[-1].policy_version == 3
        assert received[-1].adapter_revision == '/tmp/gsm8k_lora-v3'

        partitions = data_plane.list_partitions(pipeline.context)
        cleared = [p for p in partitions if p.status == PartitionStatus.CLEARED]
        assert len(cleared) == 3

    def test_pipeline_context_metadata_matches_gsm8k_config(self):
        model = FakeMultiLoraModel()
        rollout = EchoRolloutWithBoxed()
        data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

        config = BaseRLPipelineConfig(
            tenant_id='math_dept',
            training_run_id='gsm8k_run_001',
            base_model_id='Qwen/Qwen3.5-0.8B',
            adapter_name='math_lora',
            reward_type='gsm8k',
            loss_type='grpo',
            algorithm='grpo',
            max_train_partitions=1,
        )

        pipeline = FakeGRPOPipeline(
            config=config,
            model=model,
            rollout=rollout,
            reward_registry={'gsm8k': make_gsm8k_reward_fn()},
            data_plane=data_plane,
        )

        ctx = pipeline.context
        assert ctx.tenant_id == 'math_dept'
        assert ctx.training_run_id == 'gsm8k_run_001'
        assert ctx.base_model_id == 'Qwen/Qwen3.5-0.8B'
        assert ctx.adapter_name == 'math_lora'
        assert ctx.reward_type == 'gsm8k'
        assert ctx.loss_type == 'grpo'
        assert ctx.algorithm == 'grpo'

        namespace = ctx.partition_id(0)
        assert namespace == 'math_dept/gsm8k_run_001/math_lora/train_0'

        pipeline.run([make_gsm8k_sample(0)], max_steps=1)

        partitions = data_plane.list_partitions(pipeline.context)
        assert len(partitions) == 1
        assert partitions[0].context.tenant_id == 'math_dept'
        assert partitions[0].context.adapter_name == 'math_lora'

    def test_pipeline_weight_sync_increments_policy_version(self):
        """Verify policy_version increments after each train_k per design doc step 20-21."""
        model = FakeMultiLoraModel()
        rollout = EchoRolloutWithBoxed()
        received = []
        data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

        pipeline = FakeGRPOPipeline(
            config=BaseRLPipelineConfig(
                tenant_id='tenant',
                training_run_id='run',
                base_model_id='Qwen/Qwen3.5-0.8B',
                adapter_name='lora',
                reward_type='gsm8k',
                max_train_partitions=3,
            ),
            model=model,
            rollout=rollout,
            reward_registry={'gsm8k': make_gsm8k_reward_fn()},
            data_plane=data_plane,
            receive_weights_fn=lambda ctx: received.append(ctx),
        )

        samples = [make_gsm8k_sample(i) for i in range(3)]
        pipeline.run(samples, max_steps=3)

        assert len(received) == 3
        assert received[0].policy_version == 1
        assert received[1].policy_version == 2
        assert received[2].policy_version == 3
        assert received[0].adapter_revision == '/tmp/lora-v1'
        assert received[1].adapter_revision == '/tmp/lora-v2'
        assert received[2].adapter_revision == '/tmp/lora-v3'

    def test_pipeline_partition_lifecycle_is_complete(self):
        """Verify OPEN -> ROLLOUT_DONE -> REWARD_DONE -> TRAIN_READY -> TRAINING -> TRAIN_DONE -> CLEARED."""
        model = FakeMultiLoraModel()
        rollout = EchoRolloutWithBoxed()
        data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

        pipeline = FakeGRPOPipeline(
            config=BaseRLPipelineConfig(
                tenant_id='tenant',
                training_run_id='run',
                base_model_id='Qwen/Qwen3.5-0.8B',
                adapter_name='lora',
                reward_type='gsm8k',
                max_train_partitions=1,
            ),
            model=model,
            rollout=rollout,
            reward_registry={'gsm8k': make_gsm8k_reward_fn()},
            data_plane=data_plane,
        )

        pipeline.run([make_gsm8k_sample(0)], max_steps=1)

        partitions = data_plane.list_partitions(pipeline.context)
        assert len(partitions) == 1
        assert partitions[0].status == PartitionStatus.CLEARED

    def test_pipeline_no_residual_tq_data_after_clear(self):
        """Verify TQ has no residual data after partition clear per design doc step 22."""
        fake = FakeTransferQueueClient()
        model = FakeMultiLoraModel()
        rollout = EchoRolloutWithBoxed()
        data_plane = TransferQueueDataPlane(tq_client=fake)

        pipeline = FakeGRPOPipeline(
            config=BaseRLPipelineConfig(
                tenant_id='tenant',
                training_run_id='run',
                base_model_id='Qwen/Qwen3.5-0.8B',
                adapter_name='lora',
                reward_type='gsm8k',
                max_train_partitions=1,
            ),
            model=model,
            rollout=rollout,
            reward_registry={'gsm8k': make_gsm8k_reward_fn()},
            data_plane=data_plane,
        )

        pipeline.run([make_gsm8k_sample(0)], max_steps=1)

        for partition_id, fields in fake.fields.items():
            assert len(fields) == 0, f'partition {partition_id} has residual field data'
        for partition_id, tags in fake.tags.items():
            assert len(tags) == 0, f'partition {partition_id} has residual tag data'
