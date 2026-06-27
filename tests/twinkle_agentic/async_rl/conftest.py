# Copyright (c) ModelScope Contributors. All rights reserved.
"""conftest for real TransferQueue tests on NPU.

Sets up Ray once for the entire test session to avoid init/shutdown overhead.
"""
import os
import pytest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'


@pytest.fixture(scope='session', autouse=True)
def ray_session():
    import ray
    if not ray.is_initialized():
        ray.init(
            namespace='TQDataPlaneTest',
            ignore_reinit_error=True,
            num_gpus=0,
            include_dashboard=False,
        )
    yield
    ray.shutdown()
