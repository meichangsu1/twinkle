# Copyright (c) ModelScope Contributors. All rights reserved.
"""Evaluate a Qwen3 base model or LoRA with the AReaL GSM8K protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger
from twinkle.data_format import SamplingParams
from twinkle.hub import HubOperation
from twinkle.preprocessor import AReaLGSM8KProcessor
from twinkle.reward import MathVerifyAccuracyReward
from twinkle.sampler import vLLMSampler

logger = get_logger()


def load_gsm8k_rows(dataset_id: str, *, subset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import DatasetDict, load_dataset

    path = Path(dataset_id)
    if path.is_file():
        if path.suffix == '.parquet':
            dataset = load_dataset('parquet', data_files={split: str(path)}, split=split)
        elif path.suffix in {'.json', '.jsonl'}:
            dataset = load_dataset('json', data_files={split: str(path)}, split=split)
        else:
            raise ValueError(f'GSM8K dataset must be a parquet, json, or jsonl file: {path}')
    else:
        dataset = HubOperation.load_dataset(dataset_id, subset_name, split)

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    if hasattr(dataset, 'to_hf_dataset'):
        dataset = dataset.to_hf_dataset()

    rows = list(dataset)
    if not rows:
        raise ValueError('GSM8K evaluation dataset is empty')
    missing = {'question', 'answer'} - rows[0].keys()
    if missing:
        raise ValueError(f'GSM8K rows are missing required fields: {sorted(missing)}')
    return rows


def last_assistant_content(trajectory: dict[str, Any]) -> str:
    for message in reversed(trajectory.get('messages') or []):
        if message.get('role') == 'assistant':
            return str(message.get('content') or '')
    return ''


def ground_truth(trajectory: dict[str, Any]) -> str:
    for key, value in trajectory.get('user_data') or []:
        if key == 'ground_truth':
            return str(value)
    return ''


def build_sampler(args: argparse.Namespace) -> vLLMSampler:
    if args.gpus < args.tp_size or args.gpus % args.tp_size:
        raise ValueError('gpus must be a positive multiple of tp-size')

    device_group = DeviceGroup(
        name='sampler',
        ranks=list(range(args.gpus)),
        device_type=Platform.device_prefix(),
        gpus_per_worker=args.tp_size,
    )
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=args.gpus,
        dp_size=args.gpus // args.tp_size,
        tp_size=args.tp_size,
    )
    twinkle.initialize(
        mode=args.mode,
        nproc_per_node=args.gpus,
        groups=[device_group],
        seed=args.seed,
        lazy_collect=False,
    )
    engine_args = {
        'tensor_parallel_size': args.tp_size,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
        'max_num_seqs': args.max_num_seqs,
        'enforce_eager': args.enforce_eager,
        'seed': args.seed,
    }
    if args.adapter_path:
        engine_args.update({
            'enable_lora': True,
            'max_loras': 1,
            'max_lora_rank': args.max_lora_rank,
        })

    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args=engine_args,
        device_mesh=sampler_mesh,
        **({'remote_group': 'sampler'} if args.mode == 'ray' else {}),
    )
    sampler.set_template(
        'Template',
        model_id=args.model_id,
        max_length=args.max_length,
        truncation_strategy='delete',
        enable_thinking=args.enable_thinking,
    )
    return sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--subset-name', default='main')
    parser.add_argument('--split', default='test')
    parser.add_argument('--data-num', type=int, default=0, help='0 evaluates the complete split')
    parser.add_argument('--adapter-path', default='')
    parser.add_argument('--adapter-name', default='gsm8k_areal_pk_lora')
    parser.add_argument('--max-lora-rank', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=2048)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--enforce-eager', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mode', choices=('ray', 'local'), default='ray')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', default='outputs/gsm8k_areal_eval')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_gsm8k_rows(args.dataset_id, subset_name=args.subset_name, split=args.split)
    if args.data_num > 0:
        rows = rows[:args.data_num]

    processor = AReaLGSM8KProcessor()
    prompts = [processor.preprocess(row) for row in rows]
    sampler = build_sampler(args)
    reward_fn = MathVerifyAccuracyReward()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=0,
    )
    adapter_path = args.adapter_path or None
    sample_kwargs = {
        'adapter_name': args.adapter_name if adapter_path else '',
        'adapter_path': adapter_path,
        'use_base_model': adapter_path is None,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / 'predictions.jsonl'
    started = time.perf_counter()
    correct = 0.0
    sample_count = 0
    completion_lengths: list[int] = []

    with predictions_path.open('w', encoding='utf-8') as stream:
        for offset in range(0, len(prompts), args.batch_size):
            batch = prompts[offset:offset + args.batch_size]
            responses = sampler.sample(batch, sampling_params, **sample_kwargs)
            evaluated = []
            sources = []
            lengths = []
            for source, response in zip(batch, responses):
                for sequence in response.sequences:
                    evaluated.append(sequence.new_input_feature)
                    sources.append(source)
                    lengths.append(len(sequence.tokens))

            rewards = reward_fn(evaluated)
            for source, trajectory, length, reward in zip(sources, evaluated, lengths, rewards):
                stream.write(json.dumps(
                    {
                        'adapter_path': adapter_path,
                        'ground_truth': ground_truth(source),
                        'correct': float(reward),
                        'completion_length': length,
                        'completion': last_assistant_content(trajectory),
                    },
                    ensure_ascii=False,
                ) + '\n')
            correct += sum(float(reward) for reward in rewards)
            sample_count += len(rewards)
            completion_lengths.extend(lengths)
            logger.info(
                'GSM8K AReaL eval: prompts=%s/%s samples=%s accuracy=%.4f',
                min(offset + len(batch), len(prompts)),
                len(prompts),
                sample_count,
                correct / sample_count,
            )

    summary = {
        'model_id': args.model_id,
        'adapter_name': args.adapter_name if adapter_path else '',
        'adapter_path': adapter_path,
        'dataset_id': args.dataset_id,
        'split': args.split,
        'protocol': {
            'template': 'Template',
            'enable_thinking': args.enable_thinking,
            'processor': 'AReaLGSM8KProcessor',
            'reward': 'MathVerifyAccuracyReward',
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens,
        },
        'eval/accuracy': correct / sample_count,
        'eval/sample_count': sample_count,
        'eval/prompt_count': len(prompts),
        'eval/completion_length': sum(completion_lengths) / len(completion_lengths),
        'eval/max_completion_length': max(completion_lengths),
        'eval/latency_s': time.perf_counter() - started,
        'predictions_path': str(predictions_path),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
