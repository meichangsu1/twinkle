# Copyright (c) ModelScope Contributors. All rights reserved.
"""Evaluate a base model or one LoRA adapter on Maxwell-Jia/AIME_2024.

By default this script evaluates the base sampler output. When ``--adapter-path``
is provided, vLLM LoRA loading is enabled and the same evaluation runs against
that adapter. It uses the same AIMEProcessor and DAPOMathAccuracyReward as the
async/sync multi-LoRA experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger
from twinkle.data_format import SamplingParams
from twinkle.preprocessor import AIMEProcessor
from twinkle.reward import DAPOMathAccuracyReward
from twinkle.sampler import vLLMSampler

logger = get_logger()


def load_aime_rows(dataset_id: str, *, subset_name: str | None, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset_ref = dataset_id.removeprefix('hf://')
    path = Path(dataset_ref)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == '.parquet':
            dataset = load_dataset('parquet', data_files={split: str(path)}, split=split)
        elif suffix in {'.json', '.jsonl'}:
            dataset = load_dataset('json', data_files={split: str(path)}, split=split)
        else:
            raise ValueError(f'unsupported local AIME dataset file: {dataset_id}')
    else:
        kwargs = {'split': split}
        if subset_name and subset_name.lower() not in {'none', 'null'}:
            kwargs['name'] = subset_name
        dataset = load_dataset(dataset_ref, **kwargs)
    rows = list(dataset)
    missing = [field for field in ('ID', 'Problem', 'Answer') if rows and field not in rows[0]]
    assert not missing, f'AIME rows must contain ID/Problem/Answer fields, missing={missing}'
    return rows


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    assert batch_size > 0, f'batch_size must be positive, got {batch_size}'
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def last_assistant_content(trajectory: dict[str, Any]) -> str:
    for message in reversed(trajectory.get('messages') or []):
        if message.get('role') == 'assistant':
            return str(message.get('content') or '')
    return ''


def build_sampler(args) -> vLLMSampler:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', default='/nas/disk1/Qwen3.5-4B')
    parser.add_argument('--dataset-id', default='hf://Maxwell-Jia/AIME_2024')
    parser.add_argument('--subset-name', default='')
    parser.add_argument('--split', default='train')
    parser.add_argument('--data-num', type=int, default=0, help='0 means use all prompts.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--template', default='Qwen3_5Template')
    parser.add_argument('--enable-thinking', action='store_true')
    parser.add_argument('--adapter-path', default='', help='Optional LoRA adapter path. Empty means base model eval.')
    parser.add_argument('--adapter-name', default='aime_eval_lora')
    parser.add_argument('--max-loras', type=int, default=1)
    parser.add_argument('--max-lora-rank', type=int, default=16)
    parser.add_argument('--mode', choices=('ray', 'local'), default='ray')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/async_rl_experiments/base_aime_eval')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / 'predictions.jsonl'
    summary_path = output_dir / 'summary.json'

    rows = load_aime_rows(args.dataset_id, subset_name=args.subset_name, split=args.split)
    if args.data_num > 0:
        rows = rows[:args.data_num]
    assert rows, 'AIME eval rows must not be empty'

    processor = AIMEProcessor()
    trajectories = [processor.preprocess(row) for row in rows]
    sampler = build_sampler(args)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    adapter_path = args.adapter_path or None
    eval_target = f'adapter:{args.adapter_name}' if adapter_path else 'base'
    sample_kwargs = {
        'adapter_name': args.adapter_name if adapter_path else '',
        'adapter_path': adapter_path,
        'use_base_model': adapter_path is None,
    }
    reward_fn = DAPOMathAccuracyReward()

    start = time.time()
    prompt_count = 0
    sample_count = 0
    correct_count = 0.0
    completion_lengths: list[int] = []
    prompt_batches = batched(trajectories, args.batch_size)

    with predictions_path.open('w', encoding='utf-8') as file:
        for batch_index, batch in enumerate(prompt_batches):
            responses = sampler.sample(batch, sampling_params, **sample_kwargs)
            prompt_count += len(batch)
            input_data = []
            row_refs = []
            token_lengths = []
            for row, response in zip(rows[prompt_count - len(batch):prompt_count], responses):
                for sequence in response.sequences:
                    trajectory = sequence.new_input_feature
                    input_data.append(trajectory)
                    row_refs.append(row)
                    token_length = len(sequence.tokens)
                    token_lengths.append(token_length)
                    completion_lengths.append(token_length)
            rewards = reward_fn(input_data)
            for row, trajectory, token_length, reward in zip(row_refs, input_data, token_lengths, rewards):
                completion = last_assistant_content(trajectory)
                predicted = DAPOMathAccuracyReward.extract_answer(completion)
                expected = str(row['Answer']).strip()
                payload = {
                    'id': row['ID'],
                    'adapter_path': adapter_path,
                    'expected': expected,
                    'predicted': predicted,
                    'correct': float(reward),
                    'completion_length': token_length,
                    'completion': completion,
                }
                file.write(json.dumps(payload, ensure_ascii=False) + '\n')
            sample_count += len(rewards)
            correct_count += sum(float(value) for value in rewards)
            logger.info(
                'AIME %s eval batch %s/%s: prompt_count=%s sample_count=%s accuracy=%.4f',
                eval_target,
                batch_index + 1,
                len(prompt_batches),
                prompt_count,
                sample_count,
                correct_count / sample_count if sample_count else 0.0,
            )

    elapsed_s = time.time() - start
    summary = {
        'model_id': args.model_id,
        'eval_target': eval_target,
        'adapter_name': args.adapter_name if adapter_path else '',
        'adapter_path': adapter_path,
        'dataset_id': args.dataset_id,
        'split': args.split,
        'eval/accuracy': correct_count / sample_count if sample_count else 0.0,
        'eval/sample_count': sample_count,
        'eval/prompt_count': prompt_count,
        'eval/completion_length': (
            sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0.0
        ),
        'eval/max_completion_length': max(completion_lengths) if completion_lengths else 0,
        'eval/max_tokens': args.max_tokens,
        'eval/num_samples': args.num_samples,
        'eval/latency_s': elapsed_s,
        'predictions_path': str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    logger.info('AIME base eval summary: %s', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
# python cookbook/rl/eval_aime2024_base.py \
#   --model-id /nas/disk1/Qwen3.5-4B \
#   --dataset-id hf://Maxwell-Jia/AIME_2024 \
#   --adapter-path output/xxx/final-tenant_b_dapo_math_lora \
#   --adapter-name tenant_b_dapo_math_lora \
#   --max-lora-rank 16 \
#   --gpus 1 \
#   --tp-size 1 \
#   --batch-size 8 \
#   --max-tokens 4096 \
#   --output-dir outputs/async_rl_experiments/qwen35_4b_lora_aime2024