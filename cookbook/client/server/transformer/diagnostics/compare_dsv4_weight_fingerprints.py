#!/usr/bin/env python3
"""Compare rank-local parameter fingerprints from two DSV4 CLI probe runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METADATA_FIELDS = (
    'global_shape',
    'global_stride',
    'local_shape',
    'local_stride',
    'dtype',
    'local_numel',
    'local_is_contiguous',
    'local_is_meta',
    'sample_count',
    'is_dtensor',
    'placements',
    'device_mesh_shape',
    'device_mesh_dim_names',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_dir', type=Path)
    parser.add_argument('candidate_dir', type=Path)
    parser.add_argument('--mode', default='ep_loop', choices=('no_ep', 'ep_loop', 'ep_gmm'))
    parser.add_argument('--max-mismatches', type=int, default=50)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def load_rank_reports(directory: Path, mode: str) -> dict[int, dict[str, Any]]:
    pattern = f'cli_{mode}_weight_fingerprints_rank*.json'
    reports = {}
    for path in sorted(directory.glob(pattern)):
        payload = json.loads(path.read_text(encoding='utf-8'))
        rank = int(payload['rank'])
        if rank in reports:
            raise RuntimeError(f'Duplicate rank {rank} in {directory}')
        payload['_path'] = str(path)
        reports[rank] = payload
    if not reports:
        raise FileNotFoundError(f'No files matching {directory / pattern}')
    return reports


def tensor_map(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    # Reports written before buffers were added contain only ``parameters``.
    records = report.get('tensors')
    if records is None:
        records = [dict(record, kind='parameter') for record in report['parameters']]
    return {(record.get('kind', 'parameter'), record['name']): record for record in records}


def unravel_index(index: int, shape: list[int]) -> list[int]:
    coordinate = []
    remaining = index
    for size in reversed(shape):
        coordinate.append(remaining % size)
        remaining //= size
    return list(reversed(coordinate))


def sample_positions(numel: int, count: int) -> list[int]:
    if numel <= 0 or count <= 0:
        return []
    count = min(numel, count)
    if count == 1:
        return [0]
    return [index * (numel - 1) // (count - 1) for index in range(count)]


def describe_value_mismatch(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_values = reference['sample_values']
    candidate_values = candidate['sample_values']
    if len(reference_values) != len(candidate_values):
        return {
            'reference_sample_count': len(reference_values),
            'candidate_sample_count': len(candidate_values),
        }

    differences = [abs(float(left) - float(right)) for left, right in zip(reference_values, candidate_values)]
    differing = [index for index, difference in enumerate(differences) if difference != 0.0]
    result = {
        'different_sample_count': len(differing),
        'sample_count': len(differences),
        'max_abs_sample_diff': max(differences, default=0.0),
        'mean_abs_sample_diff': sum(differences) / len(differences) if differences else 0.0,
    }
    if differing:
        sample_index = differing[0]
        flat_positions = sample_positions(reference['local_numel'], reference['sample_count'])
        flat_index = flat_positions[sample_index]
        result['first_difference'] = {
            'sample_index': sample_index,
            'local_flat_index': flat_index,
            'local_coordinate': unravel_index(flat_index, reference['local_shape']),
            'reference': reference_values[sample_index],
            'candidate': candidate_values[sample_index],
        }
    return result


def main() -> None:
    args = parse_args()
    if args.max_mismatches <= 0:
        raise SystemExit('--max-mismatches must be positive')

    reference_reports = load_rank_reports(args.reference_dir, args.mode)
    candidate_reports = load_rank_reports(args.candidate_dir, args.mode)
    reference_ranks = set(reference_reports)
    candidate_ranks = set(candidate_reports)
    if reference_ranks != candidate_ranks:
        raise RuntimeError(
            f'Rank sets differ: reference={sorted(reference_ranks)}, candidate={sorted(candidate_ranks)}')

    metadata_mismatches = []
    value_mismatches = []
    missing_parameters = []
    unexpected_parameters = []
    compared_tensors = 0
    exact_tensors = 0
    full_hash_comparisons = 0

    for rank in sorted(reference_ranks):
        reference_report = reference_reports[rank]
        candidate_report = candidate_reports[rank]
        if reference_report['sample_algorithm'] != candidate_report['sample_algorithm']:
            raise RuntimeError(f'Rank {rank}: sample algorithms differ')

        reference_tensors = tensor_map(reference_report)
        candidate_tensors = tensor_map(candidate_report)
        reference_names = set(reference_tensors)
        candidate_names = set(candidate_tensors)
        for kind, name in sorted(reference_names - candidate_names):
            missing_parameters.append({'rank': rank, 'kind': kind, 'name': name})
        for kind, name in sorted(candidate_names - reference_names):
            unexpected_parameters.append({'rank': rank, 'kind': kind, 'name': name})

        for kind, name in sorted(reference_names & candidate_names):
            compared_tensors += 1
            reference = reference_tensors[(kind, name)]
            candidate = candidate_tensors[(kind, name)]
            changed_metadata = {
                field: {'reference': reference.get(field), 'candidate': candidate.get(field)}
                for field in METADATA_FIELDS
                if reference.get(field) != candidate.get(field)
            }
            if changed_metadata:
                metadata_mismatches.append({
                    'rank': rank,
                    'kind': kind,
                    'name': name,
                    'fields': changed_metadata,
                })
                continue
            reference_full_sha = reference.get('full_sha256')
            candidate_full_sha = candidate.get('full_sha256')
            if bool(reference_full_sha) != bool(candidate_full_sha):
                metadata_mismatches.append({
                    'rank': rank,
                    'kind': kind,
                    'name': name,
                    'fields': {
                        'full_sha256_available': {
                            'reference': bool(reference_full_sha),
                            'candidate': bool(candidate_full_sha),
                        }
                    },
                })
                continue
            if reference_full_sha:
                full_hash_comparisons += 1
                values_match = reference_full_sha == candidate_full_sha
                comparison_basis = 'full_sha256'
            else:
                values_match = reference['sample_sha256'] == candidate['sample_sha256']
                comparison_basis = 'sample_sha256'
            if values_match:
                exact_tensors += 1
                continue
            mismatch = {
                'rank': rank,
                'kind': kind,
                'name': name,
                'comparison_basis': comparison_basis,
                'reference_sha256': reference['sample_sha256'],
                'candidate_sha256': candidate['sample_sha256'],
                'reference_full_sha256': reference_full_sha,
                'candidate_full_sha256': candidate_full_sha,
            }
            mismatch.update(describe_value_mismatch(reference, candidate))
            value_mismatches.append(mismatch)

    total_mismatches = (
        len(metadata_mismatches)
        + len(value_mismatches)
        + len(missing_parameters)
        + len(unexpected_parameters)
    )
    report = {
        'mode': args.mode,
        'reference_dir': str(args.reference_dir),
        'candidate_dir': str(args.candidate_dir),
        'reference_memory_efficient_init': reference_reports[min(reference_ranks)]['memory_efficient_init'],
        'candidate_memory_efficient_init': candidate_reports[min(candidate_ranks)]['memory_efficient_init'],
        'ranks': sorted(reference_ranks),
        'compared_tensors': compared_tensors,
        'full_hash_comparisons': full_hash_comparisons,
        'exact_tensors': exact_tensors,
        'exact_tensor_fraction': exact_tensors / compared_tensors if compared_tensors else 0.0,
        'metadata_mismatch_count': len(metadata_mismatches),
        'value_mismatch_count': len(value_mismatches),
        'missing_parameter_count': len(missing_parameters),
        'unexpected_parameter_count': len(unexpected_parameters),
        'passed': total_mismatches == 0,
        'metadata_mismatches': metadata_mismatches[:args.max_mismatches],
        'value_mismatches': value_mismatches[:args.max_mismatches],
        'missing_parameters': missing_parameters[:args.max_mismatches],
        'unexpected_parameters': unexpected_parameters[:args.max_mismatches],
        'mismatch_lists_truncated': any(
            len(items) > args.max_mismatches
            for items in (
                metadata_mismatches,
                value_mismatches,
                missing_parameters,
                unexpected_parameters,
            )
        ),
    }
    output = args.output or args.candidate_dir / f'compare_{args.mode}_weight_fingerprints.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Report saved to: {output.resolve()}')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
