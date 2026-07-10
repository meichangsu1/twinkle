"""Synchronous GSM8K + DAPO GRPO baseline using MultiLoraTransformersModel.

This script is the two-LoRA sync counterpart for
``cookbook/rl/async_multi_lora_grpo.py``:

  for each LoRA: rollout with latest adapter_path
  barrier
  for each LoRA: reward/advantage -> train -> save adapter

It intentionally does not use TransferQueue. The goal is to keep the model
class and sampler weight-sync path aligned with async RL while preserving a
stage-barrier synchronous training loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import MultiLoraTransformersModel
from twinkle.preprocessor.llm import AIMEProcessor, DAPOMathProcessor, GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import DAPOMathAccuracyReward, GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

logger = get_logger()


def _optional_step_limit(value: str | None, *, default: int) -> int | None:
    if value is None:
        return default
    parsed = int(value)
    return None if parsed <= 0 else parsed


# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
SAMPLER_TP = int(os.environ.get('SAMPLER_TP', '1'))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

MODEL_TP_SIZE = int(os.environ.get('MODEL_TP_SIZE', '1'))
MODEL_EP_SIZE = int(os.environ.get('MODEL_EP_SIZE', '1'))
MODEL_PP_SIZE = int(os.environ.get('MODEL_PP_SIZE', '1'))
MODEL_DP_SIZE = int(os.environ.get('MODEL_DP_SIZE', str(MODEL_GPUS // (MODEL_TP_SIZE * MODEL_EP_SIZE * MODEL_PP_SIZE))))
SEQUENCE_PARALLEL = bool(int(os.environ.get('SEQUENCE_PARALLEL', '0')))
MIXED_PRECISION = os.environ.get('MIXED_PRECISION', 'bf16')

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = _optional_step_limit(os.environ.get('MAX_STEPS'), default=1000)
LR_SCHEDULER_T_MAX = int(os.environ.get('LR_SCHEDULER_T_MAX', str(MAX_STEPS or 1000)))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
GSM8K_ADAPTER_NAME = os.environ.get('GSM8K_ADAPTER_NAME', os.environ.get('ADAPTER_NAME', 'tenant_a_gsm8k_lora'))
DAPO_ADAPTER_NAME = os.environ.get('DAPO_ADAPTER_NAME', 'tenant_b_dapo_math_lora')
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', LORA_RANK * 2))
LORA_DROPOUT = float(os.environ.get('LORA_DROPOUT', 0.05))
MODEL_MAX_R = int(os.environ.get('MODEL_MAX_R', LORA_RANK))
MAX_LORAS = int(os.environ.get('MAX_LORAS', 2))
MODEL_MAX_LENGTH = int(os.environ.get('MODEL_MAX_LENGTH', 4096))
SAMPLER_MAX_MODEL_LEN = int(os.environ.get('SAMPLER_MAX_MODEL_LEN', 8192))
SAMPLER_MAX_LORA_RANK = int(os.environ.get('SAMPLER_MAX_LORA_RANK', LORA_RANK))
DATA_NUM = int(os.environ.get('DATA_NUM', '0') or 0)
GSM8K_DATA_NUM = int(os.environ.get('GSM8K_DATA_NUM', str(DATA_NUM)) or 0)
DAPO_DATA_NUM = int(os.environ.get('DAPO_DATA_NUM', str(DATA_NUM)) or 0)
DAPO_DATASET_ID = os.environ.get('DAPO_DATASET_ID', 'data/dapo-math-17k.parquet')
EVAL_AT_END = bool(int(os.environ.get('EVAL_AT_END', '1')))
EVAL_EVERY_PARTITIONS = int(os.environ.get('EVAL_EVERY_PARTITIONS', '0'))
EVAL_DATA_NUM = int(os.environ.get('EVAL_DATA_NUM', '128') or 0)
EVAL_BATCH_SIZE = int(os.environ.get('EVAL_BATCH_SIZE', str(BATCH_SIZE)))
EVAL_MAX_NEW_TOKENS = int(os.environ.get('EVAL_MAX_NEW_TOKENS', str(MAX_NEW_TOKENS)))
EVAL_TEMPERATURE = float(os.environ.get('EVAL_TEMPERATURE', '0.0'))
EVAL_TOP_P = float(os.environ.get('EVAL_TOP_P', '1.0'))
EVAL_NUM_SAMPLES = int(os.environ.get('EVAL_NUM_SAMPLES', '1'))
GSM8K_EVAL_DATA_NUM = int(os.environ.get('GSM8K_EVAL_DATA_NUM', str(EVAL_DATA_NUM)) or 0)
GSM8K_EVAL_SPLIT = os.environ.get('GSM8K_EVAL_SPLIT', 'test')
DAPO_EVAL_DATASET_ID = os.environ.get('DAPO_EVAL_DATASET_ID', 'data/aime-2024.parquet')
DAPO_EVAL_DATA_NUM = int(os.environ.get('DAPO_EVAL_DATA_NUM', str(EVAL_DATA_NUM)) or 0)
DAPO_EVAL_SPLIT = os.environ.get('DAPO_EVAL_SPLIT', 'train')
DAPO_EVAL_FORMAT = os.environ.get('DAPO_EVAL_FORMAT', 'aime').strip().lower()
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', 1.0))
NORM_TYPE = int(os.environ.get('NORM_TYPE', 2))
SAVE_OPTIMIZER = bool(int(os.environ.get('SAVE_OPTIMIZER', '0')))
IS_SAMPLER_CHECKPOINT = bool(int(os.environ.get('IS_SAMPLER_CHECKPOINT', '0')))
SYNC_INITIAL_ADAPTER = bool(int(os.environ.get('SYNC_INITIAL_ADAPTER', '0')))

RUN_ID = os.environ.get('RUN_ID') or (
    f'sync_multilora_short_math_seed{os.environ["TWINKLE_SEED"]}'
    if os.environ.get('TWINKLE_SEED') else 'sync_multilora_short_math')
MODE = os.environ.get('MODE', 'sync_multilora_gsm8k_dapo')
METRICS_JSONL = Path(
    os.environ.get('METRICS_JSONL')
    or Path('outputs/async_rl_experiments') / RUN_ID / 'metrics.jsonl')
ADAPTER_CHECKPOINT_DIR = os.environ.get(
    'ADAPTER_CHECKPOINT_DIR',
    str(Path('output') / 'short_math_grpo_multilora' / RUN_ID / 'lora_sync'),
)
TRAINING_RUN_ID = os.environ.get('TRAINING_RUN_ID', 'gsm8k_dapo_sync_multilora')
TENANT_ID = os.environ.get('TENANT_ID', 'tenant_a')
DAPO_TENANT_ID = os.environ.get('DAPO_TENANT_ID', 'tenant_b')

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')


@dataclass
class LoraRunContext:
    adapter_name: str
    tenant_id: str
    training_run_id: str
    dataset_name: str
    dataset_factory: Callable[[], Dataset]
    eval_dataset_name: str | None
    eval_dataset_factory: Callable[[], Dataset] | None
    eval_data_num: int
    accuracy_reward: Reward
    brevity_reward: Reward
    policy_version: int = 0
    latest_adapter_path: str | None = None
    optimizer_steps: int = 0
    partition_index: int = 0
    exhausted: bool = False

    @property
    def context_key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'


class JSONLMetricsWriter:

    def __init__(self, path: Path, *, run_id: str, mode: str):
        self.path = path
        self.run_id = run_id
        self.mode = mode
        self.start_time = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w', encoding='utf-8')

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def write_metadata(self, contexts: List[LoraRunContext]) -> None:
        self.write_event(
            event='run_metadata',
            phase='run',
            elapsed_s=0.0,
            metrics={
                'source': 'cookbook/rl/short_math_grpo_multilora_jsonl.py',
                'reference_source': 'cookbook/rl/short_math_grpo_jsonl.py',
                'model_class': 'MultiLoraTransformersModel',
                'weight_sync': 'adapter_path_after_train',
                'model_id': MODEL_ID,
                'model_gpus': MODEL_GPUS,
                'sampler_gpus': SAMPLER_GPUS,
                'sampler_tp': SAMPLER_TP,
                'num_generations': NUM_GENERATIONS,
                'max_new_tokens': MAX_NEW_TOKENS,
                'learning_rate': LEARNING_RATE,
                'max_steps': MAX_STEPS,
                'batch_size': BATCH_SIZE,
                'mini_batch_size': MINI_BATCH_SIZE,
                'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
                'adapter_names': [context.adapter_name for context in contexts],
                'datasets': {
                    context.adapter_name: context.dataset_name
                    for context in contexts
                },
                'eval_datasets': {
                    context.adapter_name: context.eval_dataset_name
                    for context in contexts
                },
                'eval_data_nums': {
                    context.adapter_name: context.eval_data_num or None
                    for context in contexts
                },
                'lora_rank': LORA_RANK,
                'model_max_r': MODEL_MAX_R,
                'max_loras': MAX_LORAS,
                'gsm8k_data_num': GSM8K_DATA_NUM or None,
                'dapo_data_num': DAPO_DATA_NUM or None,
                'dapo_dataset_id': DAPO_DATASET_ID,
                'eval_at_end': EVAL_AT_END,
                'eval_every_partitions': EVAL_EVERY_PARTITIONS,
                'eval_data_num': EVAL_DATA_NUM or None,
                'eval_batch_size': EVAL_BATCH_SIZE,
                'eval_max_new_tokens': EVAL_MAX_NEW_TOKENS,
                'eval_temperature': EVAL_TEMPERATURE,
                'eval_top_p': EVAL_TOP_P,
                'eval_num_samples': EVAL_NUM_SAMPLES,
                'gsm8k_eval_data_num': GSM8K_EVAL_DATA_NUM or None,
                'gsm8k_eval_split': GSM8K_EVAL_SPLIT,
                'dapo_eval_dataset_id': DAPO_EVAL_DATASET_ID or None,
                'dapo_eval_data_num': DAPO_EVAL_DATA_NUM or None,
                'dapo_eval_split': DAPO_EVAL_SPLIT,
                'dapo_eval_format': DAPO_EVAL_FORMAT,
                'sync_initial_adapter': SYNC_INITIAL_ADAPTER,
                'adapter_checkpoint_dir': ADAPTER_CHECKPOINT_DIR,
            },
        )

    def write_train_metrics(
        self,
        *,
        context: LoraRunContext,
        optimizer_step: int,
        metrics: dict[str, Any],
        policy_version: int,
        partition_id: str,
    ) -> None:
        event_metrics = dict(metrics)
        event_metrics['optimizer_step'] = optimizer_step
        event_metrics['step'] = optimizer_step
        event_metrics['max_steps'] = MAX_STEPS
        self.write_event(
            event='train_batch_done',
            phase='train',
            elapsed_s=_elapsed_s(event_metrics, fallback=time.time() - self.start_time),
            policy_version=policy_version,
            partition_id=partition_id,
            context=context,
            metrics=event_metrics,
        )

    def write_completed(self, *, contexts: List[LoraRunContext]) -> None:
        self.write_event(
            event='run_completed',
            phase='run',
            elapsed_s=time.time() - self.start_time,
            metrics={
                'optim_step': sum(context.optimizer_steps for context in contexts),
                'max_policy_version': max(context.policy_version for context in contexts),
                'per_context': {
                    context.context_key: {
                        'adapter_name': context.adapter_name,
                        'dataset': context.dataset_name,
                        'optimizer_steps': context.optimizer_steps,
                        'policy_version': context.policy_version,
                    }
                    for context in contexts
                },
            },
        )

    def write_event(
        self,
        *,
        event: str,
        phase: str,
        elapsed_s: float,
        metrics: dict[str, Any],
        policy_version: int | None = None,
        partition_id: str | None = None,
        context: LoraRunContext | None = None,
    ) -> None:
        payload = {
            'ts': time.time(),
            'elapsed_s': elapsed_s,
            'run_id': self.run_id,
            'mode': self.mode,
            'seed': _optional_int(os.environ.get('TWINKLE_SEED') or os.environ.get('SEED')),
            'event': event,
            'phase': phase,
            'context_key': context.context_key if context is not None else None,
            'adapter_name': context.adapter_name if context is not None else None,
            'partition_id': partition_id,
            'policy_version': policy_version,
            'metrics': _json_safe(metrics),
        }
        self.file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
        self.file.flush()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _elapsed_s(metrics: dict[str, Any], *, fallback: float) -> float:
    value = metrics.get('total time elapse')
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith(' seconds'):
            number = _coerce_number(stripped[:-8].strip())
            return number if number is not None else fallback
        if stripped.endswith(' minutes'):
            number = _coerce_number(stripped[:-8].strip())
            return number * 60.0 if number is not None else fallback
        number = _coerce_number(stripped)
        return number if number is not None else fallback
    return fallback


def _safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


SCRIPT_INSTANCE_ID = f'{_safe_name(RUN_ID)}-{os.getpid()}'


def _adapter_path_from_save_result(save_result: Any) -> str | None:
    if isinstance(save_result, str):
        return save_result
    path = getattr(save_result, 'twinkle_path', None)
    if path is not None:
        return path
    if isinstance(save_result, dict):
        return save_result.get('twinkle_path') or save_result.get('path')
    return None


def _step_limit_reached(context: LoraRunContext) -> bool:
    return MAX_STEPS is not None and context.optimizer_steps >= MAX_STEPS


def _default_config_path() -> str | None:
    value = os.environ.get('SYNC_LORA_CONFIG')
    if value:
        return value
    path = Path(__file__).with_suffix('.yaml')
    return path.as_posix() if path.exists() else None


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        from omegaconf import OmegaConf
    except ImportError:
        import yaml

        with Path(path).open(encoding='utf-8') as file:
            loaded = yaml.safe_load(file) or {}
    else:
        cfg = OmegaConf.load(path)
        loaded = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError(f'YAML config must load as a dict, got {type(loaded)!r}')
    return loaded


def _cfg_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key)
    return default if value is None else int(value)


def _cfg_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key)
    return default if value is None else bool(value)


def _processor_from_config(processor_cfg: dict[str, Any], *, default_cls: str) -> Any:
    processor_cls = str(processor_cfg.get('cls') or default_cls).split('.')[-1]
    if processor_cls == 'GSM8KProcessor':
        return GSM8KProcessor(
            system=processor_cfg.get('system', SYSTEM_PROMPT),
            add_assistant=bool(processor_cfg.get('add_assistant', False)),
        )
    if processor_cls == 'DAPOMathProcessor':
        return DAPOMathProcessor()
    if processor_cls == 'AIMEProcessor':
        return AIMEProcessor()
    raise ValueError(f'Unsupported processor cls={processor_cls!r}')


def _dataset_format(dataset_cfg: dict[str, Any], dataset_id: str) -> str:
    suffix = Path(dataset_id).suffix.lower().lstrip('.')
    if suffix in {'json', 'jsonl'}:
        return suffix
    configured = dataset_cfg.get('format') or dataset_cfg.get('file_type')
    if configured:
        return str(configured).lower().lstrip('.')
    if suffix:
        return suffix
    return ''


def _read_local_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        rows = []
        with path.open(encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open(encoding='utf-8') as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('data', 'rows', 'train'):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f'JSON dataset must be a row list or contain data/rows/train list: {path}')


def _local_json_dataset_path(dataset_id: str, dataset_cfg: dict[str, Any]) -> Path | None:
    path = Path(dataset_id)
    dataset_format = _dataset_format(dataset_cfg, dataset_id)
    if path.is_file() and dataset_format in {'json', 'jsonl'}:
        return path
    if path.is_dir() and dataset_format not in {'parquet', 'csv', 'arrow'}:
        split = str(dataset_cfg.get('split', 'train'))
        suffixes = [dataset_format] if dataset_format in {'json', 'jsonl'} else []
        suffixes.extend(['jsonl', 'json'])
        for suffix in suffixes:
            candidate = path / f'{split}.{suffix}'
            if candidate.exists():
                return candidate
    return None


def create_dataset_from_config(dataset_cfg: dict[str, Any], *, default_processor_cls: str) -> Dataset:
    dataset_id = dataset_cfg.get('dataset_id')
    if not dataset_id:
        raise ValueError(f'dataset config missing dataset_id: {dataset_cfg!r}')
    data_num = _cfg_int(dataset_cfg, 'data_num', 0)
    data_slice = range(data_num) if data_num else None
    dataset = Dataset()
    local_json_path = _local_json_dataset_path(str(dataset_id), dataset_cfg)
    if local_json_path is not None:
        dataset.add_dataset(
            DatasetMeta(
                subset_name=str(dataset_cfg.get('subset_name', 'default')),
                split=str(dataset_cfg.get('split', 'train')),
                data_slice=data_slice,
                data=_read_local_json_rows(local_json_path),
            ))
    else:
        dataset.add_dataset(
            DatasetMeta(
                str(dataset_id),
                subset_name=str(dataset_cfg.get('subset_name', 'default')),
                split=str(dataset_cfg.get('split', 'train')),
                data_slice=data_slice,
            ))
    dataset.set_template(
        str(dataset_cfg.get('template_cls', 'Qwen3_5Template')),
        model_id=MODEL_ID,
        max_length=_cfg_int(dataset_cfg, 'max_length', MODEL_MAX_LENGTH),
        truncation_strategy=str(dataset_cfg.get('truncation_strategy', 'delete')),
        enable_thinking=_cfg_bool(dataset_cfg, 'enable_thinking', False),
    )
    dataset.map(_processor_from_config(_cfg_dict(dataset_cfg.get('processor')), default_cls=default_processor_cls))
    dataset.encode(add_generation_prompt=True)
    return dataset


def _reward_pair(reward_type: str) -> tuple[Reward, Reward]:
    normalized = reward_type.lower().replace('-', '_')
    if normalized == 'gsm8k':
        return GSM8KAccuracyReward(), GSM8KBrevityReward()
    if normalized in {'dapo', 'dapo_math', 'aime', 'math'}:
        return DAPOMathAccuracyReward(), GSM8KBrevityReward()
    raise ValueError(f'Unsupported reward_type={reward_type!r}')


def _default_eval_dataset_cfg(reward_type: str) -> dict[str, Any] | None:
    normalized = reward_type.lower().replace('-', '_')
    if normalized == 'gsm8k':
        return {
            'name': f'gsm8k/{GSM8K_EVAL_SPLIT}',
            'dataset_id': 'ms://modelscope/gsm8k',
            'subset_name': 'main',
            'split': GSM8K_EVAL_SPLIT,
            'data_num': GSM8K_EVAL_DATA_NUM,
            'max_length': MODEL_MAX_LENGTH,
            'processor': {
                'cls': 'GSM8KProcessor',
                'system': SYSTEM_PROMPT,
            },
        }
    if normalized in {'dapo', 'dapo_math', 'aime', 'math'} and DAPO_EVAL_DATASET_ID:
        processor_cls = 'AIMEProcessor' if DAPO_EVAL_FORMAT == 'aime' else 'DAPOMathProcessor'
        return {
            'name': f'{DAPO_EVAL_DATASET_ID}:{DAPO_EVAL_SPLIT}:{DAPO_EVAL_FORMAT}',
            'dataset_id': DAPO_EVAL_DATASET_ID,
            'split': DAPO_EVAL_SPLIT,
            'data_num': DAPO_EVAL_DATA_NUM,
            'max_length': MODEL_MAX_LENGTH,
            'processor': {
                'cls': processor_cls,
            },
        }
    return None


def _default_processor_for_reward(reward_type: str) -> str:
    normalized = reward_type.lower().replace('-', '_')
    if normalized == 'gsm8k':
        return 'GSM8KProcessor'
    if normalized in {'dapo', 'dapo_math', 'aime', 'math'}:
        return 'DAPOMathProcessor'
    raise ValueError(f'Unsupported reward_type={reward_type!r}')


class GSM8KBrevityReward(Reward):
    """Reward shorter completions that contain a valid answer."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
                or re.search(r'(?im)^\s*Answer\s*:\s*\S+', completion)
            )

            if not has_answer:
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= 300:
                    rewards.append(1.0)
                else:
                    rewards.append(max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


def create_gsm8k_dataset():
    data_slice = range(GSM8K_DATA_NUM) if GSM8K_DATA_NUM else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            'ms://modelscope/gsm8k',
            subset_name='main',
            split='train',
            data_slice=data_slice,
        ))
    dataset.set_template(
        'Qwen3_5Template',
        model_id=MODEL_ID,
        max_length=MODEL_MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def create_gsm8k_eval_dataset():
    data_slice = range(GSM8K_EVAL_DATA_NUM) if GSM8K_EVAL_DATA_NUM else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            'ms://modelscope/gsm8k',
            subset_name='main',
            split=GSM8K_EVAL_SPLIT,
            data_slice=data_slice,
        ))
    dataset.set_template(
        'Qwen3_5Template',
        model_id=MODEL_ID,
        max_length=MODEL_MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def create_dapo_math_dataset():
    data_slice = range(DAPO_DATA_NUM) if DAPO_DATA_NUM else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            DAPO_DATASET_ID,
            split='train',
            data_slice=data_slice,
        ))
    dataset.set_template(
        'Qwen3_5Template',
        model_id=MODEL_ID,
        max_length=MODEL_MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.map(DAPOMathProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def create_dapo_math_eval_dataset():
    if not DAPO_EVAL_DATASET_ID:
        raise ValueError('DAPO_EVAL_DATASET_ID must be set to enable DAPO validation')
    data_slice = range(DAPO_EVAL_DATA_NUM) if DAPO_EVAL_DATA_NUM else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            DAPO_EVAL_DATASET_ID,
            split=DAPO_EVAL_SPLIT,
            data_slice=data_slice,
        ))
    dataset.set_template(
        'Qwen3_5Template',
        model_id=MODEL_ID,
        max_length=MODEL_MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    if DAPO_EVAL_FORMAT == 'dapo':
        dataset.map(DAPOMathProcessor())
    elif DAPO_EVAL_FORMAT == 'aime':
        dataset.map(AIMEProcessor())
    else:
        raise ValueError(f'Unsupported DAPO_EVAL_FORMAT={DAPO_EVAL_FORMAT!r}; expected dapo or aime')
    dataset.encode(add_generation_prompt=True)
    return dataset


def _dataset_factory_from_config(dataset_cfg: dict[str, Any], *, default_processor_cls: str) -> Callable[[], Dataset]:
    return lambda: create_dataset_from_config(dataset_cfg, default_processor_cls=default_processor_cls)


def _lora_context_from_config(context_cfg: dict[str, Any]) -> LoraRunContext:
    reward_type = str(context_cfg.get('reward_type', 'gsm8k'))
    dataset_cfg = _cfg_dict(context_cfg.get('dataset'))
    if not dataset_cfg:
        raise ValueError(f'lora context {context_cfg.get("adapter_name")!r} missing dataset config')
    default_processor_cls = _default_processor_for_reward(reward_type)
    eval_dataset_cfg = _cfg_dict(
        context_cfg.get('eval_dataset')
        or context_cfg.get('validation_dataset')
        or _default_eval_dataset_cfg(reward_type)
    )
    accuracy_reward, brevity_reward = _reward_pair(reward_type)
    adapter_name = str(context_cfg['adapter_name'])
    dataset_name = str(dataset_cfg.get('name') or context_cfg.get('dataset_name') or reward_type)
    eval_dataset_name = None
    eval_dataset_factory = None
    eval_data_num = 0
    if eval_dataset_cfg:
        eval_dataset_name = str(
            eval_dataset_cfg.get('name')
            or f'{eval_dataset_cfg.get("dataset_id")}:{eval_dataset_cfg.get("split", "train")}')
        eval_processor_cfg = _cfg_dict(eval_dataset_cfg.get('processor'))
        eval_default_processor = str(eval_processor_cfg.get('cls') or default_processor_cls)
        eval_data_num = _cfg_int(eval_dataset_cfg, 'data_num', 0)
        eval_dataset_factory = _dataset_factory_from_config(
            eval_dataset_cfg,
            default_processor_cls=eval_default_processor,
        )

    return LoraRunContext(
        adapter_name=adapter_name,
        tenant_id=str(context_cfg.get('tenant_id', TENANT_ID)),
        training_run_id=str(context_cfg.get('training_run_id', TRAINING_RUN_ID)),
        dataset_name=dataset_name,
        dataset_factory=_dataset_factory_from_config(dataset_cfg, default_processor_cls=default_processor_cls),
        eval_dataset_name=eval_dataset_name,
        eval_dataset_factory=eval_dataset_factory,
        eval_data_num=eval_data_num,
        accuracy_reward=accuracy_reward,
        brevity_reward=brevity_reward,
    )


def build_lora_contexts(config: dict[str, Any] | None = None) -> List[LoraRunContext]:
    config = config or {}
    configured_contexts = config.get('lora_contexts')
    if configured_contexts is None and config.get('lora_context') is not None:
        configured_contexts = [config['lora_context']]
    if configured_contexts is not None:
        if not isinstance(configured_contexts, list) or not configured_contexts:
            raise ValueError('YAML config lora_contexts must be a non-empty list')
        return [_lora_context_from_config(_cfg_dict(context_cfg)) for context_cfg in configured_contexts]

    return [
        LoraRunContext(
            adapter_name=GSM8K_ADAPTER_NAME,
            tenant_id=TENANT_ID,
            training_run_id=f'{TRAINING_RUN_ID}_gsm8k',
            dataset_name='gsm8k',
            dataset_factory=create_gsm8k_dataset,
            eval_dataset_name=f'gsm8k/{GSM8K_EVAL_SPLIT}',
            eval_dataset_factory=create_gsm8k_eval_dataset,
            eval_data_num=GSM8K_EVAL_DATA_NUM,
            accuracy_reward=GSM8KAccuracyReward(),
            brevity_reward=GSM8KBrevityReward(),
        ),
        LoraRunContext(
            adapter_name=DAPO_ADAPTER_NAME,
            tenant_id=DAPO_TENANT_ID,
            training_run_id=f'{TRAINING_RUN_ID}_dapo_math',
            dataset_name='dapo_math',
            dataset_factory=create_dapo_math_dataset,
            eval_dataset_name=f'{DAPO_EVAL_DATASET_ID or "none"}:{DAPO_EVAL_SPLIT}:{DAPO_EVAL_FORMAT}',
            eval_dataset_factory=create_dapo_math_eval_dataset if DAPO_EVAL_DATASET_ID else None,
            eval_data_num=DAPO_EVAL_DATA_NUM,
            accuracy_reward=DAPOMathAccuracyReward(),
            brevity_reward=GSM8KBrevityReward(),
        ),
    ]


def compute_rewards(
    context: LoraRunContext,
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_rewards = context.accuracy_reward(trajectories)
    brevity_rewards = context.brevity_reward(trajectories)
    total_rewards = [a + b for a, b in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, brevity_rewards, accuracy_rewards


def reward_metrics_for_slice(
    *,
    completion_lengths: List[int],
    total_rewards: List[float],
    brevity_rewards: List[float],
    accuracy_rewards: List[float],
) -> dict[str, Any]:
    metric = CompletionRewardMetric()
    metric.accumulate(
        completion_lengths=completion_lengths,
        rewards={
            'total': total_rewards,
            'brevity': brevity_rewards,
            'accuracy': accuracy_rewards,
        },
    )
    return metric.calculate()


def build_eval_batches(contexts: List[LoraRunContext]) -> dict[str, list[list[Any]]]:
    if not EVAL_AT_END and EVAL_EVERY_PARTITIONS <= 0:
        return {}
    if EVAL_BATCH_SIZE <= 0:
        raise ValueError(f'EVAL_BATCH_SIZE must be positive, got {EVAL_BATCH_SIZE}')
    batches_by_adapter: dict[str, list[list[Any]]] = {}
    for context in contexts:
        if context.eval_dataset_factory is None:
            logger.info('Validation disabled for adapter=%s dataset=%s: no eval dataset configured',
                        context.adapter_name, context.dataset_name)
            continue
        dataset = context.eval_dataset_factory()
        dataset_len = len(dataset)
        max_prompts = min(context.eval_data_num, dataset_len) if context.eval_data_num > 0 else dataset_len
        batches = []
        batch = []
        for index in range(max_prompts):
            batch.append(dataset[index])
            if len(batch) == EVAL_BATCH_SIZE:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        batches_by_adapter[context.adapter_name] = batches
        logger.info(
            'Loaded validation batches for adapter=%s eval_dataset=%s dataset_len=%s max_prompts=%s batches=%s',
            context.adapter_name,
            context.eval_dataset_name,
            dataset_len,
            max_prompts,
            len(batches),
        )
    return batches_by_adapter


def run_validation(
    *,
    metrics_writer: JSONLMetricsWriter,
    sampler: vLLMSampler,
    context: LoraRunContext,
    eval_batches: list[list[Any]],
    sampling_params: SamplingParams,
    adapter_path: str,
    partition_id: str,
) -> None:
    if not eval_batches:
        return

    eval_start = time.time()
    sample_kwargs: dict[str, Any] = {
        'adapter_name': context.adapter_name,
        'adapter_path': adapter_path,
    }

    prompt_count = 0
    sample_count = 0
    completion_lengths: List[int] = []
    accuracy_rewards: List[float] = []

    metrics_writer.write_event(
        event='eval_started',
        phase='eval',
        elapsed_s=eval_start - metrics_writer.start_time,
        policy_version=context.policy_version,
        partition_id=partition_id,
        context=context,
        metrics={
            'dataset': context.dataset_name,
            'eval_dataset': context.eval_dataset_name,
            'adapter_path': adapter_path,
            'eval_batch_count': len(eval_batches),
        },
    )

    sampler.reset_prefix_cache()
    for batch in eval_batches:
        prompt_count += len(batch)
        sample_responses = sampler.sample(batch, sampling_params, **sample_kwargs)
        input_data: List[Dict[str, Any]] = []
        batch_completion_lengths: List[int] = []
        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                input_data.append(sequence.new_input_feature)
                batch_completion_lengths.append(len(sequence.tokens))
        batch_accuracy = context.accuracy_reward(input_data)
        sample_count += len(input_data)
        completion_lengths.extend(batch_completion_lengths)
        accuracy_rewards.extend(batch_accuracy)

    eval_metrics = {
        'eval/accuracy': sum(accuracy_rewards) / len(accuracy_rewards) if accuracy_rewards else 0.0,
        'eval/sample_count': sample_count,
        'eval/prompt_count': prompt_count,
        'eval/completion_length': sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0.0,
        'dataset': context.dataset_name,
        'eval_dataset': context.eval_dataset_name,
        'optimizer_step': context.optimizer_steps,
        'step': context.optimizer_steps,
        'policy_version': context.policy_version,
        'eval_num_samples': EVAL_NUM_SAMPLES,
        'eval_latency_s': time.time() - eval_start,
    }
    logger.info('[%s Eval step %s version %s] %s', context.adapter_name, context.optimizer_steps,
                context.policy_version, eval_metrics)
    metrics_writer.write_event(
        event='eval_done',
        phase='eval',
        elapsed_s=time.time() - metrics_writer.start_time,
        policy_version=context.policy_version,
        partition_id=partition_id,
        context=context,
        metrics=eval_metrics,
    )


def save_adapter_snapshot(
    model: MultiLoraTransformersModel,
    context: LoraRunContext,
    *,
    policy_version: int,
) -> str | None:
    name = f'sync-multilora-{_safe_name(RUN_ID)}-{_safe_name(context.adapter_name)}-v{policy_version}'
    save_result = model.save(
        name,
        output_dir=ADAPTER_CHECKPOINT_DIR,
        adapter_name=context.adapter_name,
        save_optimizer=SAVE_OPTIMIZER,
        is_sampler=IS_SAMPLER_CHECKPOINT,
    )
    return _adapter_path_from_save_result(save_result)


def build_model(model_mesh: DeviceMesh, contexts: List[LoraRunContext]) -> MultiLoraTransformersModel:
    if MAX_LORAS < len(contexts):
        raise ValueError(f'MAX_LORAS={MAX_LORAS} is smaller than context count {len(contexts)}')
    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )
    model = MultiLoraTransformersModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
        mixed_precision=MIXED_PRECISION,
        max_loras=MAX_LORAS,
        max_r=MODEL_MAX_R,
        max_length=MODEL_MAX_LENGTH,
        target_modules='all-linear',
    )
    for context in contexts:
        model.add_adapter_to_model(
            context.adapter_name,
            lora_config,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )
        model.set_optimizer('AdamW', lr=LEARNING_RATE, adapter_name=context.adapter_name)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=LR_SCHEDULER_T_MAX, eta_min=0, adapter_name=context.adapter_name)
        model.set_loss('GRPOLoss', epsilon=0.2, adapter_name=context.adapter_name)
        model.add_metric('GRPOMetric', adapter_name=context.adapter_name, epsilon=0.2)
        model.set_processor(InputProcessor, adapter_name=context.adapter_name, padding_free=True)
        model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, adapter_name=context.adapter_name)
    return model


def build_sampler(sampler_mesh: DeviceMesh) -> vLLMSampler:
    engine_args = {
        'tensor_parallel_size': SAMPLER_TP,
        'gpu_memory_utilization': 0.8,
        'max_model_len': SAMPLER_MAX_MODEL_LEN,
        'max_loras': MAX_LORAS,
        'max_lora_rank': SAMPLER_MAX_LORA_RANK,
        'enable_lora': True,
    }
    sampler_seed = _optional_int(os.environ.get('SAMPLER_SEED'))
    if sampler_seed is not None:
        engine_args['seed'] = sampler_seed
    if bool(int(os.environ.get('ENABLE_TOWER_CONNECTOR_LORA', '0'))):
        engine_args['enable_tower_connector_lora'] = True
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args=engine_args,
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)
    return sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=_default_config_path(),
        help='Optional YAML file with lora_contexts/dataset/eval_dataset definitions.',
    )
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    contexts = build_lora_contexts(config)
    logger.info('Loaded sync multi-LoRA config: %s', args.config or '<defaults>')
    logger.info('Configured LoRA contexts: %s', [
        {
            'adapter_name': context.adapter_name,
            'dataset': context.dataset_name,
            'eval_dataset': context.eval_dataset_name,
            'eval_data_num': context.eval_data_num or None,
        }
        for context in contexts
    ])
    metrics_writer = JSONLMetricsWriter(METRICS_JSONL, run_id=RUN_ID, mode=MODE)
    metrics_writer.write_metadata(contexts)
    try:
        _main(metrics_writer, contexts)
    finally:
        metrics_writer.close()
        print(METRICS_JSONL)


def _main(metrics_writer: JSONLMetricsWriter, contexts: List[LoraRunContext]):
    device_type = Platform.device_prefix()
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type=device_type),
        DeviceGroup(
            name='sampler',
            ranks=list(range(MODEL_GPUS, NUM_GPUS)),
            device_type=device_type,
            gpus_per_worker=SAMPLER_TP,
        ),
    ]

    model_mesh = DeviceMesh.from_sizes(
        world_size=MODEL_GPUS,
        dp_size=MODEL_DP_SIZE,
        tp_size=MODEL_TP_SIZE,
        ep_size=MODEL_EP_SIZE,
        pp_size=MODEL_PP_SIZE,
        sequence_parallel=SEQUENCE_PARALLEL,
    )
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=SAMPLER_GPUS,
        dp_size=max(1, SAMPLER_GPUS // SAMPLER_TP),
        tp_size=SAMPLER_TP,
    )
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    model = build_model(model_mesh, contexts)
    sampler = build_sampler(sampler_mesh)

    global_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloaders = {
        context.adapter_name: iter(
            DataLoader(
                dataset=context.dataset_factory,
                batch_size=global_batch_size,
                min_batch_size=global_batch_size,
                device_mesh=model_mesh,
                remote_group='model',
                instance_id=f'{SCRIPT_INSTANCE_ID}-{_safe_name(context.adapter_name)}-train-',
            ))
        for context in contexts
    }

    eval_batches_by_adapter = build_eval_batches(contexts)
    advantage_fn = GRPOAdvantage()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)
    eval_sampling_params = SamplingParams(
        max_tokens=EVAL_MAX_NEW_TOKENS,
        num_samples=EVAL_NUM_SAMPLES,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
    )

    if SYNC_INITIAL_ADAPTER:
        for context in contexts:
            context.latest_adapter_path = save_adapter_snapshot(model, context, policy_version=context.policy_version)

    logger.info('Starting GSM8K + DAPO GRPO training (MultiLoraTransformersModel sync JSONL baseline)')
    logger.info(get_device_placement())

    while any(not context.exhausted and not _step_limit_reached(context) for context in contexts):
        active_contexts = [
            context for context in contexts if not context.exhausted and not _step_limit_reached(context)
        ]
        rollout_batches = []

        for context in active_contexts:
            try:
                batch = next(dataloaders[context.adapter_name])
            except StopIteration:
                context.exhausted = True
                logger.info('Dataloader exhausted for adapter=%s dataset=%s', context.adapter_name, context.dataset_name)
                continue

            partition_id = f'{context.context_key}/train_{context.partition_index}'
            context.partition_index += 1
            expand_prompts = []
            for prompt in batch:
                expand_prompts.extend([prompt] * NUM_GENERATIONS)

            rollout_start = time.time()
            metrics_writer.write_event(
                event='rollout_started',
                phase='rollout',
                elapsed_s=rollout_start - metrics_writer.start_time,
                policy_version=context.policy_version,
                partition_id=partition_id,
                context=context,
                metrics={
                    'dataset': context.dataset_name,
                    'prompt_count': len(batch),
                    'sample_count': len(expand_prompts),
                    'adapter_path': context.latest_adapter_path,
                },
            )

            sampler.reset_prefix_cache()
            sample_kwargs: dict[str, Any] = {'adapter_name': context.adapter_name}
            if context.latest_adapter_path is not None:
                sample_kwargs['adapter_path'] = context.latest_adapter_path
            sample_responses = sampler.sample(
                expand_prompts,
                sampling_params,
                **sample_kwargs,
            )

            all_input_data: List[Dict[str, Any]] = []
            all_old_logps: List[List[float]] = []
            all_completion_lengths: List[int] = []

            for sample_response in sample_responses:
                for sequence in sample_response.sequences:
                    all_input_data.append(sequence.new_input_feature)
                    all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                    all_completion_lengths.append(len(sequence.tokens))

            total_rewards, brevity_rewards, accuracy_rewards = compute_rewards(context, all_input_data)
            rollout_latency_s = time.time() - rollout_start
            metrics_writer.write_event(
                event='rollout_done',
                phase='rollout',
                elapsed_s=time.time() - metrics_writer.start_time,
                policy_version=context.policy_version,
                partition_id=partition_id,
                context=context,
                metrics={
                    'dataset': context.dataset_name,
                    'prompt_count': len(batch),
                    'sample_count': len(all_input_data),
                    'rollout_policy_version': context.policy_version,
                    'policy_version_gap': 0.0,
                    'rollout_latency_s': rollout_latency_s,
                },
            )
            rollout_batches.append({
                'context': context,
                'partition_id': partition_id,
                'batch': batch,
                'inputs': all_input_data,
                'old_logps': all_old_logps,
                'completion_lengths': all_completion_lengths,
                'total_rewards': total_rewards,
                'brevity_rewards': brevity_rewards,
                'accuracy_rewards': accuracy_rewards,
            })

        if not rollout_batches:
            break

        for rollout_batch in rollout_batches:
            context = rollout_batch['context']
            partition_id = rollout_batch['partition_id']
            all_input_data = rollout_batch['inputs']
            all_old_logps = rollout_batch['old_logps']
            all_completion_lengths = rollout_batch['completion_lengths']
            total_rewards = rollout_batch['total_rewards']
            brevity_rewards = rollout_batch['brevity_rewards']
            accuracy_rewards = rollout_batch['accuracy_rewards']
            advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

            total_completions = len(all_input_data)
            trained_samples = 0
            for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
                if _step_limit_reached(context):
                    break
                mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
                mb_inputs = all_input_data[mb_start:mb_end]
                mb_old_logps = all_old_logps[mb_start:mb_end]
                mb_advantages = advantages[mb_start:mb_end]
                mb_completion_lengths = all_completion_lengths[mb_start:mb_end]
                mb_total_rewards = total_rewards[mb_start:mb_end]
                mb_brevity_rewards = brevity_rewards[mb_start:mb_end]
                mb_accuracy_rewards = accuracy_rewards[mb_start:mb_end]

                train_start = time.time()
                model.forward_backward(
                    inputs=mb_inputs,
                    old_logps=mb_old_logps,
                    advantages=mb_advantages,
                    adapter_name=context.adapter_name,
                )
                model.clip_grad_and_step(
                    adapter_name=context.adapter_name,
                    max_grad_norm=MAX_GRAD_NORM,
                    norm_type=NORM_TYPE,
                )
                context.optimizer_steps += 1
                trained_samples += len(mb_inputs)

                log_dict = reward_metrics_for_slice(
                    completion_lengths=mb_completion_lengths,
                    total_rewards=mb_total_rewards,
                    brevity_rewards=mb_brevity_rewards,
                    accuracy_rewards=mb_accuracy_rewards,
                )
                log_dict.update(model.calculate_metric(is_training=True, adapter_name=context.adapter_name))
                log_dict.update({
                    'dataset': context.dataset_name,
                    'optimizer_step': context.optimizer_steps,
                    'global_optimizer_step': sum(item.optimizer_steps for item in contexts),
                    'sample_count': len(mb_inputs),
                    'prompt_count': len(mb_inputs) / NUM_GENERATIONS,
                    'outer_prompt_count': len(rollout_batch['batch']),
                    'outer_sample_count': total_completions,
                    'num_generations': NUM_GENERATIONS,
                    'rollout_policy_version': context.policy_version,
                    'policy_version_gap': 0.0,
                    'train_batch_latency_s': time.time() - train_start,
                })
                logger.info('[%s Step %s/%s] %s', context.adapter_name, context.optimizer_steps,
                            MAX_STEPS or 'all', log_dict)
                metrics_writer.write_train_metrics(
                    context=context,
                    optimizer_step=context.optimizer_steps,
                    metrics=log_dict,
                    policy_version=context.policy_version,
                    partition_id=partition_id,
                )

                if SAVE_STEPS > 0 and context.optimizer_steps % SAVE_STEPS == 0:
                    model.save(
                        f'math-grpo-multilora-{_safe_name(context.adapter_name)}-checkpoint-{context.optimizer_steps}',
                        output_dir=ADAPTER_CHECKPOINT_DIR,
                        adapter_name=context.adapter_name,
                        save_optimizer=SAVE_OPTIMIZER,
                        is_sampler=IS_SAMPLER_CHECKPOINT,
                    )

            sync_start = time.time()
            context.policy_version += 1
            context.latest_adapter_path = save_adapter_snapshot(
                model,
                context,
                policy_version=context.policy_version,
            )
            metrics_writer.write_event(
                event='weight_sync_done',
                phase='train',
                elapsed_s=time.time() - metrics_writer.start_time,
                policy_version=context.policy_version,
                partition_id=partition_id,
                context=context,
                metrics={
                    'dataset': context.dataset_name,
                    'adapter_path': context.latest_adapter_path,
                    'optimizer_step': context.optimizer_steps,
                    'sample_count': trained_samples,
                    'weight_sync_latency_s': time.time() - sync_start,
                },
            )
            metrics_writer.write_event(
                event='partition_train_done',
                phase='train',
                elapsed_s=time.time() - metrics_writer.start_time,
                policy_version=context.policy_version,
                partition_id=partition_id,
                context=context,
                metrics={
                    'dataset': context.dataset_name,
                    'optimizer_step': context.optimizer_steps,
                    'prompt_count': len(rollout_batch['batch']),
                    'sample_count': trained_samples,
                },
            )
            eval_batches = eval_batches_by_adapter.get(context.adapter_name) or []
            if EVAL_EVERY_PARTITIONS > 0 and eval_batches and context.policy_version % EVAL_EVERY_PARTITIONS == 0:
                if context.latest_adapter_path is None:
                    raise RuntimeError(f'No adapter path available for validation: {context.adapter_name}')
                run_validation(
                    metrics_writer=metrics_writer,
                    sampler=sampler,
                    context=context,
                    eval_batches=eval_batches,
                    sampling_params=eval_sampling_params,
                    adapter_path=context.latest_adapter_path,
                    partition_id=partition_id,
                )

    logger.info('Training completed. per_context=%s', {
        context.adapter_name: {
            'dataset': context.dataset_name,
            'optimizer_steps': context.optimizer_steps,
            'policy_version': context.policy_version,
        }
        for context in contexts
    })
    for context in contexts:
        final_save_result = model.save(
            f'math-grpo-multilora-final-{_safe_name(context.adapter_name)}',
            output_dir=ADAPTER_CHECKPOINT_DIR,
            adapter_name=context.adapter_name,
            save_optimizer=SAVE_OPTIMIZER,
            is_sampler=IS_SAMPLER_CHECKPOINT,
        )
        final_adapter_path = _adapter_path_from_save_result(final_save_result)
        if final_adapter_path is None:
            raise RuntimeError(f'Final adapter save did not return a sampler adapter path: {context.adapter_name}')
        context.latest_adapter_path = final_adapter_path
        eval_batches = eval_batches_by_adapter.get(context.adapter_name) or []
        if EVAL_AT_END and eval_batches:
            run_validation(
                metrics_writer=metrics_writer,
                sampler=sampler,
                context=context,
                eval_batches=eval_batches,
                sampling_params=eval_sampling_params,
                adapter_path=final_adapter_path,
                partition_id=f'{context.context_key}/final_eval',
            )
    metrics_writer.write_completed(contexts=contexts)


if __name__ == '__main__':
    main()

# Example:
# RUN_ID=sync_multilora_gsm8k_dapo_seed42 \
# MODE=sync_multilora_gsm8k_dapo \
# TWINKLE_SEED=42 \
# SAMPLER_SEED=42 \
# MODEL_ID=ms://Qwen/Qwen3.5-0.8B \
# MODEL_GPUS=2 \
# SAMPLER_GPUS=1 \
# SAMPLER_TP=1 \
# EVAL_AT_END=1 \
# EVAL_EVERY_PARTITIONS=0 \
# EVAL_BATCH_SIZE=16 \
# EVAL_MAX_NEW_TOKENS=1024 \
# NUM_GENERATIONS=8 \
# MAX_NEW_TOKENS=1024 \
# LR=5e-5 \
# MAX_STEPS=100 \
# BATCH_SIZE=4 \
# MINI_BATCH_SIZE=8 \
# GRADIENT_ACCUMULATION_STEPS=1 \
# LORA_RANK=16 \
# MODEL_MAX_R=16 \
# MAX_LORAS=2 \
# SAMPLER_MAX_LORA_RANK=16 \
# python cookbook/rl/short_math_grpo_multilora_jsonl.py \
#   --config cookbook/rl/short_math_grpo_multilora_jsonl.yaml
