import os
import pickle
from datetime import timedelta

import torch
import torch.distributed as dist

from twinkle.utils.framework import Torch
from twinkle.utils.platforms import Platform, ensure_hccl_socket_env


def log(stage: str) -> None:
    rank = int(os.environ.get('RANK', '-1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    world_size = int(os.environ.get('WORLD_SIZE', '-1'))
    print(f'[rank{rank}/local{local_rank}/world{world_size}] {stage}', flush=True)


def init_process_group() -> str:
    backend = Platform.get_platform().device_backend()
    master_port = int(os.environ['MASTER_PORT'])

    Torch.set_device()
    if backend == 'hccl':
        ensure_hccl_socket_env(master_port)
        log(
            'hccl_env '
            f"HCCL_IF_BASE_PORT={os.environ.get('HCCL_IF_BASE_PORT')} "
            f"HCCL_HOST_SOCKET_PORT_RANGE={os.environ.get('HCCL_HOST_SOCKET_PORT_RANGE')} "
            f"HCCL_NPU_SOCKET_PORT_RANGE={os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE')}"
        )

    log(f'before_init_process_group backend={backend}')
    dist.init_process_group(backend=backend, init_method='env://', timeout=timedelta(seconds=300))
    default_pg = dist.distributed_c10d._get_default_group()
    if getattr(default_pg, 'bound_device_id', None) is not None:
        default_pg.bound_device_id = None
    log('after_init_process_group')
    return backend


def current_device() -> torch.device:
    return torch.device(Torch.get_device())


def test_builtin_object_collective() -> None:
    rank = dist.get_rank()
    obj = [{'ok': rank}] if rank == 0 else [None]
    log('builtin.before_broadcast_object_list')
    dist.broadcast_object_list(obj, src=0)
    log(f'builtin.after_broadcast_object_list obj={obj}')


def test_manual_object_collective(device: torch.device) -> None:
    rank = dist.get_rank()
    if rank == 0:
        payload_bytes = pickle.dumps({'ok': rank}, protocol=pickle.HIGHEST_PROTOCOL)
        size_tensor = torch.tensor([len(payload_bytes)], dtype=torch.long, device=device)
    else:
        payload_bytes = None
        size_tensor = torch.empty(1, dtype=torch.long, device=device)

    log('manual.before_size_broadcast')
    dist.broadcast(size_tensor, src=0)
    payload_size = int(size_tensor.item())
    log(f'manual.after_size_broadcast size={payload_size}')

    if rank == 0:
        payload_tensor = torch.tensor(list(payload_bytes), dtype=torch.uint8, device=device)
    else:
        payload_tensor = torch.empty(payload_size, dtype=torch.uint8, device=device)

    log('manual.before_payload_broadcast')
    dist.broadcast(payload_tensor, src=0)
    log('manual.after_payload_broadcast')

    if rank != 0:
        decoded = pickle.loads(bytes(payload_tensor.cpu().tolist()))
        log(f'manual.after_deserialize decoded={decoded}')


def test_tensor_collective(device: torch.device) -> None:
    rank = dist.get_rank()
    tensor = torch.ones(1, device=device) if rank == 0 else torch.zeros(1, device=device)
    log(f'tensor.before_broadcast value={tensor.item()}')
    dist.broadcast(tensor, src=0)
    log(f'tensor.after_broadcast value={tensor.item()}')


def main() -> None:
    backend = init_process_group()
    device = current_device()
    log(f'device={device} backend={backend}')

    if os.environ.get('TEST_BUILTIN_OBJECT', '0') == '1':
        test_builtin_object_collective()
        log('builtin.before_barrier')
        dist.barrier()
        log('builtin.after_barrier')

    test_manual_object_collective(device)
    log('manual.before_barrier')
    dist.barrier()
    log('manual.after_barrier')

    test_tensor_collective(device)
    log('tensor.before_barrier')
    dist.barrier()
    log('tensor.after_barrier')


if __name__ == '__main__':
    main()
