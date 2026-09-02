#!/usr/bin/env python3
"""Save last-token logits from a running four-layer Twinkle server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig

from twinkle_client import init_twinkle_client
from twinkle_client.model import MultiLoraTransformersModel


DEFAULT_INPUT_IDS = [0, 128803, 2788, 6573, 70979, 36005, 320, 128804, 128821]
TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('no_ep', 'ep_loop', 'ep_gmm'))
    parser.add_argument('--server-url', default='http://127.0.0.1:8000')
    parser.add_argument('--server-token', default='EMPTY_TOKEN')
    parser.add_argument('--served-model', default='deepseek-v4-0731-local')
    parser.add_argument('--output-dir', type=Path, default=Path('output/dsv4_ep_diag'))
    parser.add_argument('--input-ids', default=','.join(str(item) for item in DEFAULT_INPUT_IDS))
    parser.add_argument(
        '--capacity-wait-seconds',
        type=float,
        default=90.0,
        help='Wait this long for a previous diagnostic adapter to expire (default: 90).',
    )
    parser.add_argument(
        '--capacity-poll-seconds',
        type=float,
        default=5.0,
        help='Seconds between LoRA-capacity checks (default: 5).',
    )
    return parser.parse_args()


def _extract_logits(result: Any) -> torch.Tensor:
    if hasattr(result, 'model_dump'):
        result = result.model_dump()
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        result = result[0]
    if not isinstance(result, dict) or result.get('logits') is None:
        raise RuntimeError(f'forward_only did not return logits; result type={type(result).__name__}')

    logits = torch.as_tensor(result['logits'], dtype=torch.float32)
    original_shape = tuple(logits.shape)
    while logits.ndim > 3:
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits[0, -1]
    elif logits.ndim == 2:
        logits = logits[-1]
    elif logits.ndim != 1:
        raise RuntimeError(f'Unsupported logits shape: {original_shape}')
    if logits.numel() < 1000:
        raise RuntimeError(f'Last-token logits look too small: shape={tuple(logits.shape)}')
    return logits.contiguous()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _wait_for_free_lora(client: Any, timeout: float, poll_interval: float) -> Any:
    """Wait for the prior probe's session-bound adapter to expire."""
    if timeout < 0:
        raise ValueError('--capacity-wait-seconds must be non-negative')
    if poll_interval <= 0:
        raise ValueError('--capacity-poll-seconds must be positive')

    deadline = time.monotonic() + timeout
    while True:
        capacity = client.get_capacity_info()
        if capacity.free_loras >= 1:
            return capacity

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if capacity.max_loras == 0:
                raise RuntimeError(
                    'Timed out waiting for ModelManagement capacity: '
                    'max_loras=0, used_loras=0. No model replica registered its '
                    'capacity; inspect the ModelManagement startup and replica '
                    'registration logs.')
            raise RuntimeError(
                'Timed out waiting for a free LoRA slot: '
                f'max_loras={capacity.max_loras}, used_loras={capacity.used_loras}, '
                f'free_loras={capacity.free_loras}. The previous diagnostic adapter '
                'did not expire; inspect the ModelManagement cleanup logs or restart '
                'the diagnostic server.')

        sleep_seconds = min(poll_interval, remaining)
        print(
            'Waiting for a free LoRA slot: '
            f'used={capacity.used_loras}/{capacity.max_loras}, '
            f'remaining={remaining:.1f}s',
            flush=True,
        )
        time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    input_ids = [int(item.strip()) for item in args.input_ids.split(',') if item.strip()]
    if not input_ids:
        raise SystemExit('--input-ids must contain at least one token ID')

    client = init_twinkle_client(
        base_url=args.server_url,
        api_key=args.server_token,
        session_heartbeat_interval=10,
    )
    try:
        # ModelManagement registers its capacity lazily on the first model
        # request.  ``MultiLoraTransformersModel.__init__`` calls /create,
        # ensuring get_capacity_info() reports 0/max_loras instead of 0/0.
        model = MultiLoraTransformersModel(model_id=args.served_model)
        _wait_for_free_lora(
            client,
            timeout=args.capacity_wait_seconds,
            poll_interval=args.capacity_poll_seconds,
        )

        model.add_adapter_to_model(
            f'dsv4_diag_{args.mode}',
            LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=None,
                target_parameters=TARGET_PARAMETERS,
                bias='none',
            ),
            gradient_accumulation_steps=1,
        )
        model.set_processor('InputProcessor', padding_side='left', padding_free=False)

        raw_input = {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'position_ids': list(range(len(input_ids))),
        }
        response = model.forward_only(
            inputs=[raw_input],
            disable_lora=True,
            return_logits=True,
        )
        last_logits = _extract_logits(response.result)
    finally:
        client.close()

    finite = torch.isfinite(last_logits)
    top_values, top_indices = torch.topk(last_logits, k=min(20, last_logits.numel()))
    report = {
        'mode': args.mode,
        'server_url': args.server_url,
        'served_model': args.served_model,
        'input_ids': input_ids,
        'last_logits_shape': list(last_logits.shape),
        'dtype_saved': str(last_logits.dtype),
        'sha256': _tensor_sha256(last_logits),
        'finite': bool(finite.all().item()),
        'nan_count': int(torch.isnan(last_logits).sum().item()),
        'inf_count': int(torch.isinf(last_logits).sum().item()),
        'sum': last_logits.sum().item(),
        'abs_sum': last_logits.abs().sum().item(),
        'min': last_logits.min().item(),
        'max': last_logits.max().item(),
        'top_token_ids': top_indices.tolist(),
        'top_logits': top_values.tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / f'{args.mode}_last_logits.pt'
    json_path = args.output_dir / f'{args.mode}_last_logits.json'
    torch.save(
        {
            'mode': args.mode,
            'input_ids': input_ids,
            'last_logits': last_logits,
        },
        tensor_path,
    )
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Logits saved to: {tensor_path.resolve()}')
    print(f'Report saved to: {json_path.resolve()}')


if __name__ == '__main__':
    main()
