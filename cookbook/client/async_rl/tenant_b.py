# Copyright (c) ModelScope Contributors. All rights reserved.
"""Register a second async-RL tenant on the shared runtime."""

from __future__ import annotations

import json
import os

from twinkle_client.async_rl import AsyncRLClient
from twinkle_client.http import set_api_key, set_base_url
from twinkle_client.types import (
    AsyncRLDatasetConfig,
    AsyncRLRewardConfig,
    AsyncRLRolloutConfig,
    AsyncRLTenantConfig,
    AsyncRLTenantStatus,
    AsyncRLTrainConfig,
)


def main() -> None:
    dataset_id = os.environ.get('TENANT_B_DATASET_ID')
    if not dataset_id:
        raise RuntimeError('Set TENANT_B_DATASET_ID to tenant B training dataset path or hub ID')

    set_base_url(os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000'))
    set_api_key(os.environ.get('TWINKLE_SERVER_TOKEN', 'dev-token'))
    client = AsyncRLClient()

    config = AsyncRLTenantConfig(
        tenant_id='tenant-b',
        training_run_id='gsm8k-run-002',
        adapter_name='gsm8k-lora-b',
        dataset=AsyncRLDatasetConfig(
            dataset_id=dataset_id,
            subset_name='main',
            split='train',
            data_num=128,
            max_length=2048,
            processor='GSM8KProcessor',
            system_prompt=(
                'You are a helpful math assistant. Solve the problem with minimal but '
                'correct reasoning and put your final answer within \\boxed{}.'
            ),
        ),
        rollout=AsyncRLRolloutConfig(
            batch_size=8,
            num_generations=2,
            max_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.0,
        ),
        train=AsyncRLTrainConfig(
            mini_batch_size=2,
            micro_batch_size=1,
            dynamic_batching=False,
            max_steps=10,
        ),
        reward=AsyncRLRewardConfig(
            class_path='twinkle.reward.GSM8KAccuracyReward',
            kwargs={},
        ),
    )

    tenant = client.add_tenant(config)
    print(json.dumps(tenant.model_dump(mode='json'), ensure_ascii=False, indent=2))

    active = client.wait_for_status(
        tenant.context_id,
        {AsyncRLTenantStatus.ACTIVE},
        poll_interval=1.0,
        timeout=600,
    )
    print(json.dumps(active.model_dump(mode='json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
