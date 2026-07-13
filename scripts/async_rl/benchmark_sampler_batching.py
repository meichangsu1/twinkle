# Copyright (c) ModelScope Contributors. All rights reserved.
"""Benchmark vLLM sampler call granularity.

This script compares the two rollout submission shapes that matter for the
async-vs-sync GRPO discussion:

1. batched: one sampler.sample(batch_size * num_generations)
2. concurrent_single: batch_size concurrent sampler.sample(num_generations)
3. serial_single: batch_size serial sampler.sample(num_generations)

The prompt expansion mirrors ServerSingleTurnRollout: each prompt group is
expanded into ``num_generations`` independent requests before calling
``sampler.sample``.
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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    'You are a helpful math assistant. Solve the problem step by step '
    'and put your final answer within \\boxed{}.'
)


def _resolve_local_dataset_file(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / f'{split}.parquet',
        path / f'{split}.jsonl',
        path / f'{split}.json',
        path / f'{split}-00000-of-00001.parquet',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f'cannot find local dataset file for split={split!r} under {path}')


def load_gsm8k_rows(dataset_id: str, *, subset_name: str | None, split: str) -> list[dict[str, Any]]:
    from datasets import DatasetDict
    from datasets import load_dataset

    dataset_ref = dataset_id.removeprefix('hf://')
    path = Path(dataset_ref.removeprefix('ms://'))
    if path.exists():
        data_file = _resolve_local_dataset_file(path, split)
        suffix = data_file.suffix.lower()
        if suffix == '.parquet':
            dataset = load_dataset('parquet', data_files={split: str(data_file)}, split=split)
        elif suffix in {'.json', '.jsonl'}:
            dataset = load_dataset('json', data_files={split: str(data_file)}, split=split)
        else:
            raise ValueError(f'unsupported local GSM8K dataset file: {data_file}')
    else:
        from twinkle.hub import HubOperation

        dataset = HubOperation.load_dataset(dataset_id, subset_name or 'main', split)

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split] if split in dataset else dataset['train']
    if hasattr(dataset, 'to_hf_dataset'):
        dataset = dataset.to_hf_dataset()

    rows = list(dataset)
    missing = [field for field in ('question', 'answer') if rows and field not in rows[0]]
    assert not missing, f'GSM8K rows must contain question/answer fields, missing={missing}'
    return rows


def build_prompt_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    from twinkle.preprocessor import GSM8KProcessor

    if args.prompt:
        rows = [{'question': args.prompt, 'answer': '0'} for _ in range(args.batch_size)]
    else:
        rows = load_gsm8k_rows(args.dataset_id, subset_name=args.subset_name, split=args.split)
        assert len(rows) >= args.batch_size, (
            f'dataset must contain at least batch_size={args.batch_size} rows, got {len(rows)}')
        rows = rows[:args.batch_size]
    processor = GSM8KProcessor(system=args.system)
    return [processor.preprocess(row) for row in rows]


def expand_prompt_group(prompt_group: dict[str, Any], prompt_idx: int, num_generations: int) -> list[dict[str, Any]]:
    group_id = prompt_group.get('group_id') or prompt_group.get('sample_id') or f'prompt_{prompt_idx}'
    expanded = []
    for generation_idx in range(num_generations):
        item = dict(prompt_group)
        item['group_id'] = group_id
        item['generation_idx'] = generation_idx
        expanded.append(item)
    return expanded


def expand_prompt_groups(prompt_groups: list[dict[str, Any]], num_generations: int) -> list[dict[str, Any]]:
    expanded = []
    for prompt_idx, prompt_group in enumerate(prompt_groups):
        expanded.extend(expand_prompt_group(prompt_group, prompt_idx, num_generations))
    return expanded


def build_sampler(args: argparse.Namespace) -> Any:
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh, Platform
    from twinkle.sampler import vLLMSampler

    assert args.gpus >= args.tp_size, f'gpus must be >= tp_size, got {args.gpus} < {args.tp_size}'
    assert args.gpus % args.tp_size == 0, f'gpus must be divisible by tp_size, got {args.gpus} % {args.tp_size}'
    device_type = Platform.device_prefix()
    device_groups = [
        DeviceGroup(
            name='sampler',
            ranks=list(range(args.gpus)),
            device_type=device_type,
            gpus_per_worker=args.tp_size,
        )
    ]
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=args.gpus,
        dp_size=max(1, args.gpus // args.tp_size),
        tp_size=args.tp_size,
    )
    twinkle.initialize(
        mode=args.mode,
        nproc_per_node=args.gpus,
        groups=device_groups,
        seed=args.seed,
        lazy_collect=False,
    )
    engine_args = {
        'tensor_parallel_size': args.tp_size,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
        'seed': args.seed,
        'max_num_seqs': args.max_num_seqs,
        'enable_prefix_caching': args.enable_prefix_caching,
    }
    if args.adapter_path:
        engine_args.update({
            'enable_lora': True,
            'max_loras': args.max_loras,
            'max_lora_rank': args.max_lora_rank,
        })
    sampler_kwargs = {'remote_group': 'sampler'} if args.mode == 'ray' else {}
    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args=engine_args,
        device_mesh=sampler_mesh,
        **sampler_kwargs,
    )
    sampler.set_template(
        args.template,
        model_id=args.model_id,
        max_length=args.max_length,
        truncation_strategy='delete',
        enable_thinking=args.enable_thinking,
    )
    return sampler


def make_sampling_params(args: argparse.Namespace) -> Any:
    from twinkle.data_format import SamplingParams

    return SamplingParams(
        max_tokens=args.max_tokens,
        num_samples=1,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.logprobs,
    )


def count_output_tokens(responses: list[Any]) -> int:
    total = 0
    for response in responses:
        for sequence in response.sequences:
            total += len(sequence.tokens)
    return total


def reset_prefix_cache_if_needed(sampler: Any, enabled: bool) -> None:
    if not enabled:
        return
    reset_prefix_cache = getattr(sampler, 'reset_prefix_cache', None)
    if reset_prefix_cache is not None:
        reset_prefix_cache()


def call_sampler(
    sampler: Any,
    inputs: list[dict[str, Any]],
    sampling_params: Any,
    sample_kwargs: dict[str, Any],
) -> tuple[float, int, int]:
    start = time.perf_counter()
    responses = sampler.sample(inputs, sampling_params, **sample_kwargs)
    latency_s = time.perf_counter() - start
    return latency_s, len(responses), count_output_tokens(responses)


def run_batched(
    sampler: Any,
    prompt_groups: list[dict[str, Any]],
    sampling_params: Any,
    sample_kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    expanded = expand_prompt_groups(prompt_groups, args.num_generations)
    latency_s, response_count, output_tokens = call_sampler(sampler, expanded, sampling_params, sample_kwargs)
    return {
        'method': 'batched',
        'sample_calls': 1,
        'prompt_groups': len(prompt_groups),
        'request_count': len(expanded),
        'response_count': response_count,
        'output_tokens': output_tokens,
        'latency_s': latency_s,
        'call_latencies_s': [latency_s],
    }


def run_serial_single(
    sampler: Any,
    prompt_groups: list[dict[str, Any]],
    sampling_params: Any,
    sample_kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    call_latencies = []
    output_tokens = 0
    response_count = 0
    start = time.perf_counter()
    for prompt_idx, prompt_group in enumerate(prompt_groups):
        expanded = expand_prompt_group(prompt_group, prompt_idx, args.num_generations)
        latency_s, responses, tokens = call_sampler(sampler, expanded, sampling_params, sample_kwargs)
        call_latencies.append(latency_s)
        response_count += responses
        output_tokens += tokens
    latency_s = time.perf_counter() - start
    return {
        'method': 'serial_single',
        'sample_calls': len(prompt_groups),
        'prompt_groups': len(prompt_groups),
        'request_count': len(prompt_groups) * args.num_generations,
        'response_count': response_count,
        'output_tokens': output_tokens,
        'latency_s': latency_s,
        'call_latencies_s': call_latencies,
    }


def run_concurrent_single(
    sampler: Any,
    prompt_groups: list[dict[str, Any]],
    sampling_params: Any,
    sample_kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    call_latencies = []
    output_tokens = 0
    response_count = 0
    worker_count = min(args.concurrency, len(prompt_groups))
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for prompt_idx, prompt_group in enumerate(prompt_groups):
            expanded = expand_prompt_group(prompt_group, prompt_idx, args.num_generations)
            futures.append(executor.submit(call_sampler, sampler, expanded, sampling_params, sample_kwargs))
        for future in concurrent.futures.as_completed(futures):
            latency_s, responses, tokens = future.result()
            call_latencies.append(latency_s)
            response_count += responses
            output_tokens += tokens
    latency_s = time.perf_counter() - start
    return {
        'method': 'concurrent_single',
        'sample_calls': len(prompt_groups),
        'prompt_groups': len(prompt_groups),
        'request_count': len(prompt_groups) * args.num_generations,
        'response_count': response_count,
        'output_tokens': output_tokens,
        'latency_s': latency_s,
        'call_latencies_s': call_latencies,
        'concurrency': worker_count,
    }


def summarize_events(events: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get('phase') != 'measure':
            continue
        by_method.setdefault(event['method'], []).append(event)

    summary = {
        'config': {
            'model_id': args.model_id,
            'adapter_path': args.adapter_path or None,
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
        },
        'methods': {},
    }
    for method, method_events in sorted(by_method.items()):
        latencies = [float(event['latency_s']) for event in method_events]
        output_tokens = sum(int(event['output_tokens']) for event in method_events)
        request_count = sum(int(event['request_count']) for event in method_events)
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
        }
    return summary


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(payload, ensure_ascii=False) + '\n')


def run_method(
    method: str,
    sampler: Any,
    prompt_groups: list[dict[str, Any]],
    sampling_params: Any,
    sample_kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method == 'batched':
        return run_batched(sampler, prompt_groups, sampling_params, sample_kwargs, args)
    if method == 'concurrent_single':
        return run_concurrent_single(sampler, prompt_groups, sampling_params, sample_kwargs, args)
    if method == 'serial_single':
        return run_serial_single(sampler, prompt_groups, sampling_params, sample_kwargs, args)
    raise ValueError(f'unsupported method: {method}')


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
    parser.add_argument('--methods', nargs='+', default=['batched', 'concurrent_single'],
                        choices=['batched', 'concurrent_single', 'serial_single'])
    parser.add_argument('--concurrency', type=int, default=16)
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
    parser.add_argument('--adapter-path', default='')
    parser.add_argument('--adapter-name', default='benchmark_lora')
    parser.add_argument('--max-loras', type=int, default=1)
    parser.add_argument('--max-lora-rank', type=int, default=16)
    parser.add_argument('--mode', choices=('ray', 'local'), default='ray')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable-prefix-caching', action='store_true')
    parser.add_argument('--no-reset-prefix-cache', action='store_true')
    parser.add_argument('--output-dir', default='outputs/async_rl_experiments/sampler_batching_benchmark')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.batch_size > 0, f'batch-size must be positive, got {args.batch_size}'
    assert args.num_generations > 0, f'num-generations must be positive, got {args.num_generations}'
    assert args.rounds > 0, f'rounds must be positive, got {args.rounds}'
    assert args.warmup_rounds >= 0, f'warmup-rounds must be non-negative, got {args.warmup_rounds}'
    assert args.concurrency > 0, f'concurrency must be positive, got {args.concurrency}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / 'sampler_batching_events.jsonl'
    summary_path = output_dir / 'summary.json'
    events_path.unlink(missing_ok=True)

    prompt_groups = build_prompt_groups(args)
    sampler = build_sampler(args)
    sampling_params = make_sampling_params(args)
    sample_kwargs = {
        'adapter_name': args.adapter_name if args.adapter_path else '',
        'adapter_path': args.adapter_path or None,
        'use_base_model': not bool(args.adapter_path),
    }
    reset_prefix_cache = not args.no_reset_prefix_cache

    events: list[dict[str, Any]] = []
    config_event = {
        'phase': 'config',
        'model_id': args.model_id,
        'adapter_path': args.adapter_path or None,
        'batch_size': args.batch_size,
        'num_generations': args.num_generations,
        'request_count_per_round': args.batch_size * args.num_generations,
        'methods': args.methods,
    }
    write_jsonl(events_path, config_event)

    for phase, rounds in [('warmup', args.warmup_rounds), ('measure', args.rounds)]:
        for round_idx in range(rounds):
            for method in args.methods:
                reset_prefix_cache_if_needed(sampler, reset_prefix_cache)
                result = run_method(method, sampler, prompt_groups, sampling_params, sample_kwargs, args)
                event = {
                    'phase': phase,
                    'round_idx': round_idx,
                    **result,
                    'requests_per_sec': result['request_count'] / result['latency_s']
                    if result['latency_s'] > 0 else 0.0,
                    'output_tokens_per_sec': result['output_tokens'] / result['latency_s']
                    if result['latency_s'] > 0 else 0.0,
                }
                write_jsonl(events_path, event)
                logger.info(
                    'sampler batching %s round=%s method=%s latency=%.3fs requests=%s req/s=%.3f tok/s=%.3f',
                    phase,
                    round_idx,
                    method,
                    event['latency_s'],
                    event['request_count'],
                    event['requests_per_sec'],
                    event['output_tokens_per_sec'],
                )
                if phase == 'measure':
                    events.append(event)

    summary = summarize_events(events, args)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info('Wrote sampler batching events to %s', events_path)
    logger.info('Wrote sampler batching summary to %s', summary_path)


if __name__ == '__main__':
    main()


# python scripts/async_rl/benchmark_sampler_batching.py \
#   --model-id /nas/disk1/Qwen3.5-4B \
#   --dataset-id /model/ljl/project/data/gsm8k/train-00000-of-00001.parquet \
#   --split train \
#   --batch-size 8 \
#   --num-generations 8 \
#   --concurrency 8 \
#   --max-tokens 1024 \
#   --temperature 1.0 \
#   --top-p 0.95 \
#   --logprobs 1 \
#   --gpus 1 \
#   --tp-size 1 \
#   --mode ray \
#   --output-dir outputs/async_rl_experiments/sampler_batching_qwen35_4b



# python scripts/async_rl/benchmark_sampler_batching.py \
#   --model-id /nas/disk1/Qwen3.5-4B \
#   --dataset-id /model/ljl/project/data/gsm8k/train-00000-of-00001.parquet \
#   --split train \
#   --batch-size 8 \
#   --num-generations 8 \
#   --concurrency 8 \
#   --adapter-path output/gsm8k_split_multi_lora/lora_sync/async-grpo-final-gsm8k_split_tenant_a-tenant_a_gsm8k_lora \
#   --adapter-name tenant_a_gsm8k_lora \
#   --max-lora-rank 16 \
#   --max-tokens 1024 \
#   --temperature 1.0 \ 
#   --top-p 0.95 \
#   --logprobs 1 \
#   --gpus 1 \
#   --tp-size 1 \
#   --mode ray