#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


REQUIRED_GSM8K_COLUMNS = {'question', 'answer'}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Split GSM8K train rows into two equal local parquet files for multi-LoRA experiments.',
    )
    parser.add_argument('--dataset-id', default='ms://modelscope/gsm8k')
    parser.add_argument('--subset-name', default='main')
    parser.add_argument('--split', default='train')
    parser.add_argument('--eval-split', default='test')
    parser.add_argument('--total-prompts', type=int, default=80)
    parser.add_argument('--eval-data-num', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('data/gsm8k_split_multi_lora'))
    args = parser.parse_args()

    if args.total_prompts <= 0:
        raise ValueError(f'--total-prompts must be positive, got {args.total_prompts}')
    if args.total_prompts % 2 != 0:
        raise ValueError(f'--total-prompts must be even for equal tenant splits, got {args.total_prompts}')
    if args.eval_data_num < 0:
        raise ValueError(f'--eval-data-num must be non-negative, got {args.eval_data_num}')

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _load_dataset(args.dataset_id, subset_name=args.subset_name, split=args.split)
    train = _require_gsm8k_schema(train, name=f'{args.dataset_id}:{args.split}')
    if len(train) < args.total_prompts:
        raise ValueError(
            f'GSM8K split source only has {len(train)} rows, cannot take {args.total_prompts} prompts')

    selected = train.shuffle(seed=args.seed) if not args.no_shuffle else train
    selected = selected.select(range(args.total_prompts))
    selected = _select_gsm8k_columns(selected)
    half = args.total_prompts // 2
    tenant_a = selected.select(range(0, half))
    tenant_b = selected.select(range(half, args.total_prompts))

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
        'total_prompts': args.total_prompts,
        'tenant_prompt_count': half,
        'tenant_a_train': tenant_a_path.as_posix(),
        'tenant_b_train': tenant_b_path.as_posix(),
    }

    if not args.skip_eval:
        eval_rows = _load_dataset(args.dataset_id, subset_name=args.subset_name, split=args.eval_split)
        eval_rows = _require_gsm8k_schema(eval_rows, name=f'{args.dataset_id}:{args.eval_split}')
        eval_limit = min(args.eval_data_num, len(eval_rows)) if args.eval_data_num > 0 else len(eval_rows)
        eval_rows = _select_gsm8k_columns(eval_rows.select(range(eval_limit)))
        eval_path = output_dir / 'gsm8k_test.parquet'
        eval_rows.to_parquet(eval_path.as_posix())
        manifest.update({
            'eval_split': args.eval_split,
            'eval_prompt_count': eval_limit,
            'eval': eval_path.as_posix(),
        })

    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                             encoding='utf-8')

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _load_dataset(dataset_id: str, *, subset_name: str | None, split: str) -> Dataset:
    dataset_ref = dataset_id.removeprefix('hf://')
    local_path = Path(dataset_ref.removeprefix('ms://'))
    if local_path.exists():
        dataset = _load_local_dataset(local_path, split=split)
    elif dataset_id.startswith('ms://'):
        from twinkle.hub import HubOperation

        dataset = HubOperation.load_dataset(dataset_id, subset_name or 'main', split)
    else:
        dataset = load_dataset(dataset_ref, subset_name, split=split)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found in {dataset_id!r}. Available: {list(dataset.keys())}")
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(f'Expected datasets.Dataset for {dataset_id}:{split}, got {type(dataset)!r}')
    return dataset


def _load_local_dataset(path: Path, *, split: str) -> Dataset:
    if path.is_dir():
        for suffix in ('parquet', 'jsonl', 'json'):
            candidate = path / f'{split}.{suffix}'
            if candidate.exists():
                return _load_local_dataset(candidate, split=split)
        raise FileNotFoundError(f'No {split}.parquet/jsonl/json found in {path}')

    suffix = path.suffix.lower()
    if suffix == '.parquet':
        return load_dataset('parquet', data_files={split: path.as_posix()}, split=split)
    if suffix in {'.json', '.jsonl'}:
        return load_dataset('json', data_files={split: path.as_posix()}, split=split)
    raise ValueError(f'Unsupported local GSM8K file type: {path}')


def _require_gsm8k_schema(dataset: Dataset, *, name: str) -> Dataset:
    columns = set(dataset.column_names)
    missing = sorted(REQUIRED_GSM8K_COLUMNS - columns)
    if missing:
        raise ValueError(f'{name} must contain GSM8K columns {sorted(REQUIRED_GSM8K_COLUMNS)}, missing {missing}')
    return dataset


def _select_gsm8k_columns(dataset: Dataset) -> Dataset:
    return dataset.select_columns(['question', 'answer'])


if __name__ == '__main__':
    main()
