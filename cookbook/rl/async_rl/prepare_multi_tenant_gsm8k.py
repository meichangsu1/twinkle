"""Create deterministic GSM8K JSONL files for two RL tenants.

Examples:
    python cookbook/rl/async_rl/prepare_multi_tenant_gsm8k.py

    python cookbook/rl/async_rl/prepare_multi_tenant_gsm8k.py \
        --source /data/gsm8k/train.jsonl \
        --output-dir /data/gsm8k_multi_tenant
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def _local_dataset(path: Path, split: str) -> Dataset:
    if path.is_file():
        suffix = path.suffix.lower().lstrip('.')
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(suffix, suffix)
        return load_dataset(file_type, data_files=str(path), split='train')

    if (path / 'state.json').exists() or (path / 'dataset_dict.json').exists():
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        return loaded

    candidates = [
        item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in {'.json', '.jsonl', '.csv', '.parquet'}
    ]
    preferred = [item for item in candidates if split.lower() in item.stem.lower()]
    selected = preferred or candidates
    if not selected:
        raise ValueError(f'no supported dataset files found in {path}')
    suffix = selected[0].suffix.lower().lstrip('.')
    if any(item.suffix.lower().lstrip('.') != suffix for item in selected):
        raise ValueError(f'dataset files in {path} must use one file type')
    file_type = {'jsonl': 'json'}.get(suffix, suffix)
    return load_dataset(file_type, data_files=[str(item) for item in selected], split='train')


def load_source(source: str, subset: str, split: str) -> Dataset:
    local_path = Path(source).expanduser()
    if local_path.exists():
        dataset = _local_dataset(local_path, split)
    else:
        from twinkle.hub import HubOperation

        dataset = HubOperation.load_dataset(source, subset, split)
        if hasattr(dataset, 'to_hf_dataset'):
            dataset = dataset.to_hf_dataset()
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(f'expected a Hugging Face Dataset, got {type(dataset).__name__}')
    missing = {'question', 'answer'} - set(dataset.column_names)
    if missing:
        raise ValueError(f'GSM8K dataset is missing required columns: {sorted(missing)}')
    return dataset


def prepare_tenant_datasets(
    dataset: Dataset,
    *,
    samples_per_tenant: int,
    seed: int,
    shared: bool,
) -> tuple[Dataset, Dataset]:
    required = samples_per_tenant if shared else samples_per_tenant * 2
    if len(dataset) < required:
        raise ValueError(f'dataset has {len(dataset)} rows, but {required} rows are required')
    shuffled = dataset.shuffle(seed=seed)
    tenant_a = shuffled.select(range(samples_per_tenant))
    tenant_b_start = 0 if shared else samples_per_tenant
    tenant_b = shuffled.select(range(tenant_b_start, tenant_b_start + samples_per_tenant))
    return tenant_a, tenant_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='ms://modelscope/gsm8k')
    parser.add_argument('--subset', default='main')
    parser.add_argument('--split', default='train')
    parser.add_argument('--samples-per-tenant', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', default='data/gsm8k_multi_tenant')
    parser.add_argument(
        '--shared',
        action='store_true',
        help='write the same shuffled rows for both tenants instead of disjoint shards',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_tenant <= 0:
        raise ValueError('--samples-per-tenant must be positive')

    dataset = load_source(args.source, args.subset, args.split)
    tenant_a, tenant_b = prepare_tenant_datasets(
        dataset,
        samples_per_tenant=args.samples_per_tenant,
        seed=args.seed,
        shared=args.shared,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tenant_a_path = output_dir / 'tenant_a_train.jsonl'
    tenant_b_path = output_dir / 'tenant_b_train.jsonl'
    tenant_a.to_json(str(tenant_a_path), force_ascii=False)
    tenant_b.to_json(str(tenant_b_path), force_ascii=False)

    print(f'tenant_a: {len(tenant_a)} rows -> {tenant_a_path}')
    print(f'tenant_b: {len(tenant_b)} rows -> {tenant_b_path}')
    print(f'export TENANT_A_DATASET_ID={tenant_a_path}')
    print(f'export TENANT_B_DATASET_ID={tenant_b_path}')


if __name__ == '__main__':
    main()
