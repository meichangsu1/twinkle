# Copyright (c) ModelScope Contributors. All rights reserved.
"""Benchmark vLLM LoRA switching overhead.

This benchmark keeps the dataset, prompt count, generation count, and sampler
call granularity fixed while changing only the adapter path schedule:

1. frequent_switch_serial / frequent_switch_concurrent
   Prompt groups cycle through adapters one by one: A, B, A, B...
2. infrequent_switch_serial / infrequent_switch_concurrent
   Prompt groups use adapters in blocks: A, A, ..., B, B, ...

The goal is to measure whether frequent LoRA path switching hurts sampler
throughput compared with less frequent switching over the same adapters.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
SCRIPT_ROOT = REPO_ROOT / 'scripts' / 'async_rl'
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_sampler_batching import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    build_prompt_groups,
    build_sampler,
    count_output_tokens,
    expand_prompt_group,
    make_sampling_params,
    reset_prefix_cache_if_needed,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _parse_adapter_values(raw_values: list[str], *, field_name: str) -> list[str]:
    values: list[str] = []
    for raw_value in raw_values:
        values.extend(item.strip() for item in raw_value.split(',') if item.strip())
    assert values, f'{field_name} must contain at least one value'
    return values


def _adapter_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    paths = _parse_adapter_values(args.adapter_paths, field_name='adapter-paths')
    if args.adapter_names:
        names = _parse_adapter_values(args.adapter_names, field_name='adapter-names')
        assert len(names) == len(paths), (
            f'adapter-names length must match adapter-paths length, got {len(names)} != {len(paths)}')
    else:
        names = [f'benchmark_lora_{idx}' for idx in range(len(paths))]
    return [{'adapter_name': name, 'adapter_path': path} for name, path in zip(names, paths)]


def _sample_group(
    sampler: Any,
    prompt_group: dict[str, Any],
    prompt_idx: int,
    adapter: dict[str, str],
    sampling_params: Any,
    num_generations: int,
) -> tuple[float, int, int, str]:
    inputs = expand_prompt_group(prompt_group, prompt_idx, num_generations)
    start = time.perf_counter()
    responses = sampler.sample(
        inputs,
        sampling_params,
        adapter_name=adapter['adapter_name'],
        adapter_path=adapter['adapter_path'],
        use_base_model=False,
    )
    latency_s = time.perf_counter() - start
    return latency_s, len(responses), count_output_tokens(responses), adapter['adapter_name']


def _adapter_for_group(
    adapters: list[dict[str, str]],
    prompt_idx: int,
    *,
    block_size: int,
) -> dict[str, str]:
    adapter_idx = (prompt_idx // block_size) % len(adapters)
    return adapters[adapter_idx]


def _count_switches(adapter_names: list[str]) -> int:
    if not adapter_names:
        return 0
    return sum(1 for prev, cur in zip(adapter_names, adapter_names[1:]) if prev != cur)


def run_serial(
    sampler: Any,
    prompt_groups: list[dict[str, Any]],
    adapters: list[dict[str, str]],
    sampling_params: Any,
    args: argparse.Namespace,
    *,
    block_size: int,
) -> dict[str, Any]:
    call_latencies = []
    output_tokens = 0
    response_count = 0
    adapter_counts: dict[str, int] = {}
    adapter_sequence: list[str] = []
    start = time.perf_counter()
    for prompt_idx, prompt_group in enumerate(prompt_groups):
        adapter = _adapter_for_group(adapters, prompt_idx, block_size=block_size)
        latency_s, responses, tokens, adapter_name = _sample_group(
            sampler,
            prompt_group,
            prompt_idx,
            adapter,
            sampling_params,
            args.num_generations,
        )
        call_latencies.append(latency_s)
        response_count += responses
        output_tokens += tokens
        adapter_counts[adapter_name] = adapter_counts.get(adapter_name, 0) + 1
        adapter_sequence.append(adapter_name)
    latency_s = time.perf_counter() - start
    return {
        'sample_calls': len(prompt_groups),
        'prompt_groups': len(prompt_groups),
        'request_count': len(prompt_groups) * args.num_generations,
        'response_count': response_count,
        'output_tokens': output_tokens,
        'latency_s': latency_s,
        'call_latencies_s': call_latencies,
        'adapter_counts': adapter_counts,
        'adapter_switches': _count_switches(adapter_sequence),
    }


def run_concurrent(
    sampler: Any,
    prompt_groups: list[dict[str, Any]],
    adapters: list[dict[str, str]],
    sampling_params: Any,
    args: argparse.Namespace,
    *,
    block_size: int,
) -> dict[str, Any]:
    call_latencies = []
    output_tokens = 0
    response_count = 0
    adapter_counts: dict[str, int] = {}
    scheduled_adapters: list[str] = []
    worker_count = min(args.concurrency, len(prompt_groups))
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for prompt_idx, prompt_group in enumerate(prompt_groups):
            adapter = _adapter_for_group(adapters, prompt_idx, block_size=block_size)
            scheduled_adapters.append(adapter['adapter_name'])
            futures.append(executor.submit(
                _sample_group,
                sampler,
                prompt_group,
                prompt_idx,
                adapter,
                sampling_params,
                args.num_generations,
            ))
        for future in concurrent.futures.as_completed(futures):
            latency_s, responses, tokens, adapter_name = future.result()
            call_latencies.append(latency_s)
            response_count += responses
            output_tokens += tokens
            adapter_counts[adapter_name] = adapter_counts.get(adapter_name, 0) + 1
    latency_s = time.perf_counter() - start
    return {
        'sample_calls': len(prompt_groups),
        'prompt_groups': len(prompt_groups),
        'request_count': len(prompt_groups) * args.num_generations,
        'response_count': response_count,
        'output_tokens': output_tokens,
        'latency_s': latency_s,
        'call_latencies_s': call_latencies,
        'adapter_counts': adapter_counts,
        'adapter_switches': _count_switches(scheduled_adapters),
        'concurrency': worker_count,
    }


def run_method(
    method: str,
    sampler: Any,
    prompt_groups: list[dict[str, Any]],
    adapters: list[dict[str, str]],
    sampling_params: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method == 'frequent_switch_serial':
        return run_serial(sampler, prompt_groups, adapters, sampling_params, args, block_size=1)
    if method == 'infrequent_switch_serial':
        return run_serial(
            sampler,
            prompt_groups,
            adapters,
            sampling_params,
            args,
            block_size=args.infrequent_switch_block_size,
        )
    if method == 'frequent_switch_concurrent':
        return run_concurrent(sampler, prompt_groups, adapters, sampling_params, args, block_size=1)
    if method == 'infrequent_switch_concurrent':
        return run_concurrent(
            sampler,
            prompt_groups,
            adapters,
            sampling_params,
            args,
            block_size=args.infrequent_switch_block_size,
        )
    raise ValueError(f'unsupported method: {method}')


def _percentile(values: list[float], q: float) -> float:
    assert values, 'percentile requires non-empty values'
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return sorted(values)[index]


def summarize_events(events: list[dict[str, Any]], args: argparse.Namespace, adapters: list[dict[str, str]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get('phase') == 'measure':
            by_method.setdefault(event['method'], []).append(event)

    summary: dict[str, Any] = {
        'config': {
            'model_id': args.model_id,
            'adapter_paths': [adapter['adapter_path'] for adapter in adapters],
            'adapter_names': [adapter['adapter_name'] for adapter in adapters],
            'batch_size': args.batch_size,
            'num_generations': args.num_generations,
            'request_count_per_round': args.batch_size * args.num_generations,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'logprobs': args.logprobs,
            'mode': args.mode,
            'gpus': args.gpus,
            'tp_size': args.tp_size,
            'concurrency': args.concurrency,
            'max_loras': args.max_loras,
            'infrequent_switch_block_size': args.infrequent_switch_block_size,
        },
        'methods': {},
    }
    for method, method_events in sorted(by_method.items()):
        latencies = [float(event['latency_s']) for event in method_events]
        output_tokens = sum(int(event['output_tokens']) for event in method_events)
        request_count = sum(int(event['request_count']) for event in method_events)
        adapter_switches = [int(event.get('adapter_switches', 0)) for event in method_events]
        call_latencies = [
            float(latency)
            for event in method_events
            for latency in event.get('call_latencies_s', [])
        ]
        total_latency = sum(latencies)
        summary['methods'][method] = {
            'rounds': len(method_events),
            'latency_s_mean': statistics.mean(latencies),
            'latency_s_min': min(latencies),
            'latency_s_max': max(latencies),
            'request_per_sec': request_count / total_latency if total_latency > 0 else 0.0,
            'output_tokens_per_sec': output_tokens / total_latency if total_latency > 0 else 0.0,
            'request_count': request_count,
            'output_tokens': output_tokens,
            'adapter_switches_mean': statistics.mean(adapter_switches),
            'adapter_switches_per_round': adapter_switches,
            'call_latency_s_mean': statistics.mean(call_latencies) if call_latencies else None,
            'call_latency_s_p95': _percentile(call_latencies, 0.95) if call_latencies else None,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--dataset-id', default='ms://modelscope/gsm8k')
    parser.add_argument('--subset-name', default='main')
    parser.add_argument('--split', default='train')
    parser.add_argument('--prompt', default='', help='Use a repeated synthetic prompt instead of a dataset.')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of prompt groups per measured round.')
    parser.add_argument('--num-generations', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup-rounds', type=int, default=1)
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['frequent_switch_concurrent', 'infrequent_switch_concurrent'],
        choices=[
            'frequent_switch_serial',
            'infrequent_switch_serial',
            'frequent_switch_concurrent',
            'infrequent_switch_concurrent',
        ],
    )
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--infrequent-switch-block-size', type=int, default=16,
                        help='Prompt groups per adapter before switching in infrequent_switch methods.')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--logprobs', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=256)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--template', default='Qwen3_5Template')
    parser.add_argument('--enable-thinking', action='store_true')
    parser.add_argument('--system', default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument('--adapter-paths', nargs='+', required=True,
                        help='LoRA adapter paths. Accepts space-separated values or comma-separated values.')
    parser.add_argument('--adapter-names', nargs='*', default=[],
                        help='Optional adapter names matching --adapter-paths.')
    parser.add_argument('--max-loras', type=int, default=8)
    parser.add_argument('--max-lora-rank', type=int, default=16)
    parser.add_argument('--mode', choices=('ray', 'local'), default='ray')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable-prefix-caching', action='store_true')
    parser.add_argument('--no-reset-prefix-cache', action='store_true')
    parser.add_argument('--output-dir', default='outputs/async_rl_experiments/lora_switching_benchmark')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.batch_size > 0, f'batch-size must be positive, got {args.batch_size}'
    assert args.num_generations > 0, f'num-generations must be positive, got {args.num_generations}'
    assert args.rounds > 0, f'rounds must be positive, got {args.rounds}'
    assert args.warmup_rounds >= 0, f'warmup-rounds must be non-negative, got {args.warmup_rounds}'
    assert args.concurrency > 0, f'concurrency must be positive, got {args.concurrency}'
    assert args.infrequent_switch_block_size > 0, (
        f'infrequent-switch-block-size must be positive, got {args.infrequent_switch_block_size}')

    adapters = _adapter_plan(args)
    assert len(adapters) >= 2, 'LoRA switching benchmark requires at least two adapter paths'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / 'lora_switching_events.jsonl'
    summary_path = output_dir / 'summary.json'
    events_path.unlink(missing_ok=True)

    prompt_groups = build_prompt_groups(args)
    sampler = build_sampler(args)
    sampling_params = make_sampling_params(args)
    reset_prefix_cache = not args.no_reset_prefix_cache

    config_event = {
        'phase': 'config',
        'model_id': args.model_id,
        'adapter_paths': [adapter['adapter_path'] for adapter in adapters],
        'adapter_names': [adapter['adapter_name'] for adapter in adapters],
        'batch_size': args.batch_size,
        'num_generations': args.num_generations,
        'request_count_per_round': args.batch_size * args.num_generations,
        'methods': args.methods,
        'infrequent_switch_block_size': args.infrequent_switch_block_size,
    }
    write_jsonl(events_path, config_event)

    events: list[dict[str, Any]] = []
    for phase, rounds in [('warmup', args.warmup_rounds), ('measure', args.rounds)]:
        for round_idx in range(rounds):
            for method in args.methods:
                reset_prefix_cache_if_needed(sampler, reset_prefix_cache)
                result = run_method(method, sampler, prompt_groups, adapters, sampling_params, args)
                event = {
                    'phase': phase,
                    'round_idx': round_idx,
                    'method': method,
                    **result,
                    'requests_per_sec': result['request_count'] / result['latency_s']
                    if result['latency_s'] > 0 else 0.0,
                    'output_tokens_per_sec': result['output_tokens'] / result['latency_s']
                    if result['latency_s'] > 0 else 0.0,
                }
                write_jsonl(events_path, event)
                logger.info(
                    'lora switching %s round=%s method=%s latency=%.3fs requests=%s req/s=%.3f tok/s=%.3f adapters=%s',
                    phase,
                    round_idx,
                    method,
                    event['latency_s'],
                    event['request_count'],
                    event['requests_per_sec'],
                    event['output_tokens_per_sec'],
                    event['adapter_counts'],
                )
                if phase == 'measure':
                    events.append(event)

    summary = summarize_events(events, args, adapters)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info('Wrote LoRA switching events to %s', events_path)
    logger.info('Wrote LoRA switching summary to %s', summary_path)


if __name__ == '__main__':
    main()


# Example:
# python scripts/async_rl/benchmark_lora_switching.py \
#   --model-id /nas/disk1/Qwen3.5-4B \
#   --dataset-id /model/ljl/project/data/gsm8k/train-00000-of-00001.parquet \
#   --split train \
#   --batch-size 32 \
#   --num-generations 4 \
#   --concurrency 32 \
#   --infrequent-switch-block-size 16 \
#   --adapter-paths \
#     output/gsm8k_split_multi_lora/lora_sync/async-grpo-final-gsm8k_split_tenant_a-tenant_a_gsm8k_lora \
#     output/gsm8k_split_multi_lora/lora_sync/async-grpo-final-gsm8k_split_tenant_b-tenant_b_gsm8k_lora \
#   --adapter-names tenant_a_gsm8k_lora tenant_b_gsm8k_lora \
#   --max-loras 8 \
#   --max-lora-rank 16 \
#   --max-tokens 1024 \
#   --temperature 1.0 \
#   --top-p 0.95 \
#   --logprobs 1 \
#   --gpus 1 \
#   --tp-size 1 \
#   --mode ray \
#   --output-dir outputs/async_rl_experiments/lora_switching_qwen35_4b
