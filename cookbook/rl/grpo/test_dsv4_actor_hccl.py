# Copyright (c) ModelScope Contributors. All rights reserved.
"""Probe the 32-rank actor HCCL group without loading a model or starting Ray.

Launch this file with torchrun on both actor nodes using the same rendezvous
address. The all_gather_object call mirrors the num_tokens gather in
TransformersModel.clip_grad_norm().
"""

import argparse
from datetime import timedelta
import os
import socket
import time
import traceback

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--tensor-mib', type=int, default=1)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--backend', choices=('hccl', 'gloo'), default='hccl')
    args = parser.parse_args()
    if min(args.rounds, args.tensor_mib, args.timeout_seconds) <= 0:
        parser.error('--rounds, --tensor-mib and --timeout-seconds must be positive')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    host = socket.gethostname()
    stage = 'device setup'

    try:
        if args.backend == 'hccl':
            import torch_npu  # noqa: F401 -- registers HCCL and the NPU device

            if local_rank >= torch.npu.device_count():
                raise RuntimeError(
                    f'LOCAL_RANK={local_rank}, but only {torch.npu.device_count()} NPUs are visible')
            torch.npu.set_device(local_rank)
            device = torch.device('npu', local_rank)
        else:
            device = torch.device('cpu')

        print(
            f'JOIN rank={rank}/{world_size} host={host} local_rank={local_rank} '
            f'device={device} visible={os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unset")}',
            flush=True,
        )
        timeout = timedelta(seconds=args.timeout_seconds)
        stage = 'init_process_group'
        dist.init_process_group(args.backend, init_method='env://', timeout=timeout)
        stage = 'new_group(all ranks)'
        group = dist.new_group(ranks=list(range(world_size)), timeout=timeout)

        elements = args.tensor_mib * 1024 * 1024 // 4
        payload = torch.full((elements,), rank, dtype=torch.int32, device=device)
        gathered_tensors = [torch.empty_like(payload) for _ in range(world_size)]
        expected_markers = torch.arange(world_size, device=device, dtype=torch.int32)
        for iteration in range(args.rounds):
            started = time.monotonic()
            stage = f'round {iteration}: all_gather_object(num_tokens)'
            gathered_objects = [None] * world_size
            dist.all_gather_object(gathered_objects, [rank + 1], group=group)
            if gathered_objects != [[peer + 1] for peer in range(world_size)]:
                raise AssertionError(f'incorrect object gather: {gathered_objects}')

            stage = f'round {iteration}: tensor all_gather'
            dist.all_gather(gathered_tensors, payload, group=group)
            markers = torch.stack([tensor[0] for tensor in gathered_tensors])
            if not torch.equal(markers, expected_markers):
                raise AssertionError(f'incorrect tensor gather: {markers.cpu().tolist()}')

            stage = f'round {iteration}: all_reduce'
            checksum = torch.tensor([rank], dtype=torch.int32, device=device)
            dist.all_reduce(checksum, group=group)
            if checksum.item() != world_size * (world_size - 1) // 2:
                raise AssertionError(f'incorrect all_reduce: {checksum.item()}')

            if rank == 0:
                print(f'PASS round={iteration + 1}/{args.rounds} seconds={time.monotonic() - started:.3f}',
                      flush=True)

        if rank == 0:
            print(f'PASS all ranks: world_size={world_size}, backend={args.backend}', flush=True)
        dist.destroy_process_group(group)
        dist.destroy_process_group()
    except BaseException:
        print(f'FAIL rank={rank} host={host} local_rank={local_rank} stage={stage}', flush=True)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
