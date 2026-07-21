#!/usr/bin/env python3
"""Split a ZIP into Base64 text chunks and restore it losslessly."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO


MANIFEST_NAME = 'manifest.json'
READ_SIZE = 3 * 1024 * 1024


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f'output path is not a directory: {path}')
        if any(path.iterdir()):
            raise ValueError(f'output directory is not empty: {path}')
    else:
        path.mkdir(parents=True)


def _open_chunk(output_dir: Path, index: int) -> BinaryIO:
    return (output_dir / f'part-{index:06d}.txt').open('wb')


def encode(input_path: Path, output_dir: Path, chunk_size: int) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if chunk_size <= 0 or chunk_size % 4:
        raise ValueError('chunk size must be a positive multiple of 4 Base64 characters')

    _prepare_output_directory(output_dir)
    digest = hashlib.sha256()
    input_size = 0
    encoded_size = 0
    chunk_index = 0
    chunk_written = 0
    chunk_stream: BinaryIO | None = None

    try:
        with input_path.open('rb') as source:
            while raw := source.read(READ_SIZE):
                digest.update(raw)
                input_size += len(raw)
                encoded = base64.b64encode(raw)
                encoded_size += len(encoded)
                offset = 0
                while offset < len(encoded):
                    if chunk_stream is None:
                        chunk_index += 1
                        chunk_stream = _open_chunk(output_dir, chunk_index)
                        chunk_written = 0
                    write_size = min(chunk_size - chunk_written, len(encoded) - offset)
                    chunk_stream.write(encoded[offset:offset + write_size])
                    chunk_written += write_size
                    offset += write_size
                    if chunk_written == chunk_size:
                        chunk_stream.close()
                        chunk_stream = None
    finally:
        if chunk_stream is not None:
            chunk_stream.close()

    manifest: dict[str, object] = {
        'format': 'base64-chunks-v1',
        'filename': input_path.name,
        'input_size': input_size,
        'encoded_size': encoded_size,
        'sha256': digest.hexdigest(),
        'chunk_size': chunk_size,
        'chunk_count': chunk_index,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='ascii')
    return manifest


def _load_manifest(input_dir: Path) -> dict[str, object]:
    manifest_path = input_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f'missing manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='ascii'))
    if manifest.get('format') != 'base64-chunks-v1':
        raise ValueError(f'unsupported chunk format: {manifest.get("format")!r}')
    return manifest


def decode(input_dir: Path, output_path: Path | None) -> Path:
    manifest = _load_manifest(input_dir)
    if output_path is None:
        output_path = Path(str(manifest['filename']))
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_count = int(manifest['chunk_count'])
    digest = hashlib.sha256()
    restored_size = 0
    temporary_path = output_path.with_name(f'.{output_path.name}.partial')
    if temporary_path.exists():
        raise FileExistsError(temporary_path)

    try:
        with temporary_path.open('wb') as target:
            for index in range(1, chunk_count + 1):
                chunk_path = input_dir / f'part-{index:06d}.txt'
                if not chunk_path.is_file():
                    raise FileNotFoundError(f'missing chunk: {chunk_path}')
                encoded = chunk_path.read_bytes()
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except ValueError as exc:
                    raise ValueError(f'invalid Base64 data in {chunk_path}') from exc
                target.write(raw)
                digest.update(raw)
                restored_size += len(raw)

        expected_size = int(manifest['input_size'])
        expected_digest = str(manifest['sha256'])
        if restored_size != expected_size:
            raise ValueError(f'restored size mismatch: expected {expected_size}, got {restored_size}')
        if digest.hexdigest() != expected_digest:
            raise ValueError(f'SHA-256 mismatch: expected {expected_digest}, got {digest.hexdigest()}')
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    encode_parser = subparsers.add_parser('encode', help='encode a ZIP as Base64 text chunks')
    encode_parser.add_argument('input', type=Path)
    encode_parser.add_argument('--output-dir', type=Path, required=True)
    encode_parser.add_argument(
        '--chunk-size-mib',
        type=int,
        default=10,
        help='maximum size of each text chunk in MiB (default: 10)',
    )

    decode_parser = subparsers.add_parser('decode', help='restore a ZIP from Base64 text chunks')
    decode_parser.add_argument('input_dir', type=Path)
    decode_parser.add_argument('--output', type=Path)

    args = parser.parse_args()
    if args.command == 'encode':
        manifest = encode(args.input, args.output_dir, args.chunk_size_mib * 1024 * 1024)
        print(json.dumps(manifest, indent=2))
    else:
        output_path = decode(args.input_dir, args.output)
        print(f'restored {output_path}')


if __name__ == '__main__':
    main()
