#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


DAPO_MATH_COLUMNS = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sample two equal, disjoint DAPO-Math datasets for multi-LoRA experiments.',
    )
    parser.add_argument('--dataset-id', default='hf://BytedTsinghua-SIA/DAPO-Math-17k')
    parser.add_argument('--subset-name', default=None)
    parser.add_argument('--split', default='train')
    parser.add_argument('--samples-per-dataset', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('data/dapo_math_split_multi_lora'))
    args = parser.parse_args()

    if args.samples_per_dataset <= 0:
        raise ValueError(
            f'--samples-per-dataset must be positive, got {args.samples_per_dataset}')

    source = _load_dataset(args.dataset_id, subset_name=args.subset_name, split=args.split)
    _require_dapo_math_columns(source, name=f'{args.dataset_id}:{args.split}')

    total_samples = args.samples_per_dataset * 2
    if len(source) < total_samples:
        raise ValueError(
            f'DAPO-Math source only has {len(source)} rows, '
            f'cannot build two datasets with {args.samples_per_dataset} rows each')

    selected = source.shuffle(seed=args.seed) if not args.no_shuffle else source
    selected = selected.select(range(total_samples)).select_columns(DAPO_MATH_COLUMNS)
    _validate_dapo_math_rows(selected)

    split_at = args.samples_per_dataset
    tenant_a = selected.select(range(split_at))
    tenant_b = selected.select(range(split_at, total_samples))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tenant_a_path = output_dir / 'tenant_a_train.parquet'
    tenant_b_path = output_dir / 'tenant_b_train.parquet'
    tenant_a.to_parquet(tenant_a_path.as_posix())
    tenant_b.to_parquet(tenant_b_path.as_posix())

    manifest: dict[str, Any] = {
        'dataset_id': args.dataset_id,
        'subset_name': args.subset_name,
        'split': args.split,
        'seed': args.seed,
        'shuffle': not args.no_shuffle,
        'samples_per_dataset': args.samples_per_dataset,
        'total_samples': total_samples,
        'tenant_a_train': tenant_a_path.as_posix(),
        'tenant_b_train': tenant_b_path.as_posix(),
    }
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _load_dataset(dataset_id: str, *, subset_name: str | None, split: str) -> Dataset:
    dataset_ref = dataset_id.removeprefix('hf://')
    local_path = Path(dataset_ref)
    if local_path.exists():
        if local_path.suffix.lower() != '.parquet':
            raise ValueError(f'Local DAPO-Math source must be a parquet file, got {local_path}')
        dataset = load_dataset('parquet', data_files={split: local_path.as_posix()}, split=split)
    elif dataset_id.startswith('ms://'):
        from twinkle.hub import HubOperation

        dataset = HubOperation.load_dataset(dataset_id, subset_name or 'default', split)
    else:
        dataset = load_dataset(dataset_ref, subset_name, split=split)

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found in {dataset_id!r}. Available: {list(dataset.keys())}")
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(f'Expected datasets.Dataset for {dataset_id}:{split}, got {type(dataset)!r}')
    return dataset


def _require_dapo_math_columns(dataset: Dataset, *, name: str) -> None:
    columns = set(dataset.column_names)
    missing = sorted(set(DAPO_MATH_COLUMNS) - columns)
    if missing:
        raise ValueError(f'{name} must contain DAPO-Math columns {DAPO_MATH_COLUMNS}, missing {missing}')


def _validate_dapo_math_rows(dataset: Dataset) -> None:
    for index, row in enumerate(dataset):
        if not isinstance(row['data_source'], str):
            raise TypeError(f'DAPO-Math row {index} data_source must be a string')
        if not isinstance(row['ability'], str):
            raise TypeError(f'DAPO-Math row {index} ability must be a string')

        prompt = row['prompt']
        if not isinstance(prompt, list) or not prompt:
            raise TypeError(f'DAPO-Math row {index} prompt must be a non-empty message list')
        for message_index, message in enumerate(prompt):
            if not isinstance(message, dict):
                raise TypeError(
                    f'DAPO-Math row {index} prompt message {message_index} must be a dict')
            if not isinstance(message.get('role'), str) or not isinstance(message.get('content'), str):
                raise TypeError(
                    f'DAPO-Math row {index} prompt message {message_index} '
                    'must contain string role/content')

        reward_model = row['reward_model']
        if not isinstance(reward_model, dict):
            raise TypeError(f'DAPO-Math row {index} reward_model must be a dict')
        if not isinstance(reward_model.get('ground_truth'), str):
            raise TypeError(f'DAPO-Math row {index} reward_model.ground_truth must be a string')
        if not isinstance(row['extra_info'], dict):
            raise TypeError(f'DAPO-Math row {index} extra_info must be a dict')


if __name__ == '__main__':
    main()
    
    
#  PYTHONPATH=src python scripts/async_rl/split_dapo_math_for_multilora.py \
#   --dataset-id hf://BytedTsinghua-SIA/DAPO-Math-17k \
#   --samples-per-dataset 2000 \
#   --seed 42 \
#   --output-dir data/dapo_math_split_multi_lora   
# PYTHONPATH=src python scripts/async_rl/split_dapo_math_for_multilora.py \
#   --dataset-id data/dapo-math-17k.parquet \
#   --samples-per-dataset 2000 \
#   --seed 42 \
#   --output-dir data/dapo_math_split_multi_lora