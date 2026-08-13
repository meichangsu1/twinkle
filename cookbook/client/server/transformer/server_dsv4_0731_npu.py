# Copyright (c) ModelScope Contributors. All rights reserved.
"""Launch the two-NPU DeepSeek-V4-0731 Multi-LoRA server."""
import os

os.environ.setdefault('TWINKLE_TRUST_REMOTE_CODE', '1')

from twinkle.server import launch_server  # noqa: E402

file_dir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_dir, 'server_config_dsv4_0731_npu.yaml')

launch_server(config_path=config_path)
