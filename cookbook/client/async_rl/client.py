# Copyright (c) ModelScope Contributors. All rights reserved.
"""Manage dynamic tenants in the async-RL service.

Examples:
    export TWINKLE_SERVER_URL=http://127.0.0.1:8000
    export TWINKLE_SERVER_TOKEN=dev-token
    export TENANT_DATASET_ID=/path/to/gsm8k.parquet

    python cookbook/client/async_rl/client.py add
    python cookbook/client/async_rl/client.py list
    python cookbook/client/async_rl/client.py get CONTEXT_ID
    python cookbook/client/async_rl/client.py remove CONTEXT_ID
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

from twinkle_client.async_rl import AsyncRLClient
from twinkle_client.http import set_api_key, set_base_url
from twinkle_client.types import (
    AsyncRLDatasetConfig,
    AsyncRLRewardConfig,
    AsyncRLRolloutConfig,
    AsyncRLTenantConfig,
    AsyncRLTenantInfo,
    AsyncRLTenantStatus,
    AsyncRLTrainConfig,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--server-url',
        default=os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000'),
        help='Twinkle Server root URL; do not append /api/v1/async-rl.',
    )
    parser.add_argument(
        '--token',
        default=os.environ.get('TWINKLE_SERVER_TOKEN', 'dev-token'),
        help='Bearer token used as the async-RL tenant owner.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    add = subparsers.add_parser('add', help='Add a tenant and wait until it is ACTIVE.')
    add.add_argument('--tenant-id', default='tenant-a')
    add.add_argument('--training-run-id', default='gsm8k-run-001')
    add.add_argument('--adapter-name', default='gsm8k-lora')
    add.add_argument('--dataset-id', default=os.environ.get('TENANT_DATASET_ID'))
    add.add_argument('--subset-name', default='main')
    add.add_argument('--split', default='train')
    add.add_argument('--data-num', type=int, default=128)
    add.add_argument('--max-length', type=int, default=4096)
    add.add_argument('--rollout-batch-size', type=int, default=16)
    add.add_argument('--num-generations', type=int, default=4)
    add.add_argument('--max-tokens', type=int, default=2048)
    add.add_argument('--mini-batch-size', type=int, default=4)
    add.add_argument('--micro-batch-size', type=int, default=1)
    add.add_argument('--max-steps', type=int, default=10)
    add.add_argument('--timeout', type=float, default=600.0)
    add.add_argument('--no-wait', action='store_true')

    subparsers.add_parser('list', help='List tenants owned by the current token.')

    get = subparsers.add_parser('get', help='Get one tenant.')
    get.add_argument('context_id')

    remove = subparsers.add_parser('remove', help='Drain and remove one tenant.')
    remove.add_argument('context_id')
    remove.add_argument('--timeout', type=float, default=600.0)
    remove.add_argument('--no-wait', action='store_true')
    return parser


def _tenant_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> AsyncRLTenantConfig:
    if not args.dataset_id:
        parser.error('add requires --dataset-id or the TENANT_DATASET_ID environment variable')
    return AsyncRLTenantConfig(
        tenant_id=args.tenant_id,
        training_run_id=args.training_run_id,
        adapter_name=args.adapter_name,
        dataset=AsyncRLDatasetConfig(
            dataset_id=args.dataset_id,
            subset_name=args.subset_name,
            split=args.split,
            data_num=args.data_num,
            max_length=args.max_length,
            processor='GSM8KProcessor',
            system_prompt=(
                'You are a helpful math assistant. Solve the problem with minimal but '
                'correct reasoning and put your final answer within \\boxed{}.'
            ),
        ),
        rollout=AsyncRLRolloutConfig(
            batch_size=args.rollout_batch_size,
            num_generations=args.num_generations,
            max_tokens=args.max_tokens,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.0,
        ),
        train=AsyncRLTrainConfig(
            mini_batch_size=args.mini_batch_size,
            micro_batch_size=args.micro_batch_size,
            dynamic_batching=False,
            max_steps=args.max_steps,
        ),
        reward=AsyncRLRewardConfig(
            class_path='twinkle.reward.GSM8KAccuracyReward',
            kwargs={},
        ),
    )


def _print(value: AsyncRLTenantInfo | Sequence[AsyncRLTenantInfo]) -> None:
    if isinstance(value, AsyncRLTenantInfo):
        payload: Any = value.model_dump(mode='json')
    else:
        payload = [tenant.model_dump(mode='json') for tenant in value]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    set_base_url(args.server_url)
    set_api_key(args.token)
    client = AsyncRLClient()

    if args.command == 'add':
        tenant = client.add_tenant(_tenant_config(args, parser))
        _print(tenant)
        if not args.no_wait:
            _print(
                client.wait_for_status(
                    tenant.context_id,
                    {AsyncRLTenantStatus.ACTIVE},
                    timeout=args.timeout,
                ))
        return

    if args.command == 'list':
        _print(client.list_tenants())
        return

    if args.command == 'get':
        _print(client.get_tenant(args.context_id))
        return

    tenant = client.remove_tenant(args.context_id)
    _print(tenant)
    if not args.no_wait:
        _print(
            client.wait_for_status(
                tenant.context_id,
                {AsyncRLTenantStatus.REMOVED},
                timeout=args.timeout,
            ))


if __name__ == '__main__':
    main()
