#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

QUERY_FIELDS = [
    'timestamp',
    'index',
    'utilization.gpu',
    'memory.used',
    'power.draw',
]


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect GPU utilization samples with nvidia-smi.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval-s', type=float, default=1.0)
    parser.add_argument('--duration-s', type=float, default=None)
    parser.add_argument(
        '--gpu-role',
        action='append',
        default=[],
        help='Optional GPU role mapping, for example --gpu-role 0=trainer --gpu-role 1=sampler.',
    )
    args = parser.parse_args()

    role_by_index = _parse_gpu_roles(args.gpu_role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'elapsed_s',
                'timestamp',
                'gpu_index',
                'role',
                'gpu_util',
                'memory_used_mb',
                'power_w',
            ],
        )
        writer.writeheader()
        while args.duration_s is None or time.time() - start < args.duration_s:
            for sample in _sample_gpus():
                gpu_index = sample['index']
                writer.writerow({
                    'elapsed_s': f'{time.time() - start:.3f}',
                    'timestamp': sample['timestamp'],
                    'gpu_index': gpu_index,
                    'role': role_by_index.get(gpu_index, ''),
                    'gpu_util': sample['utilization.gpu'],
                    'memory_used_mb': sample['memory.used'],
                    'power_w': sample['power.draw'],
                })
            file.flush()
            time.sleep(max(args.interval_s, 0.1))


def _sample_gpus() -> list[dict[str, str]]:
    command = [
        'nvidia-smi',
        f'--query-gpu={",".join(QUERY_FIELDS)}',
        '--format=csv,noheader,nounits',
    ]
    output = subprocess.check_output(command, text=True)
    samples = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(',')]
        if len(values) != len(QUERY_FIELDS):
            continue
        samples.append(dict(zip(QUERY_FIELDS, values)))
    return samples


def _parse_gpu_roles(items: list[str]) -> dict[str, str]:
    roles = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'--gpu-role must be INDEX=ROLE, got {item!r}')
        index, role = item.split('=', 1)
        roles[index.strip()] = role.strip()
    return roles


if __name__ == '__main__':
    main()
