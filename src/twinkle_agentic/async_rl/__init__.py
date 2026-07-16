"""Native TransferQueue building blocks for async multi-LoRA RL."""

from .context_manager import ContextStatus, LoraContextManager, PartitionStatus
from .data_plane import TQDataPlane
from .metrics import CompositeMetricsRecorder, JSONLMetricsRecorder, MetricsBuffer
from .pipeline import AsyncMultiLoraGRPOConfig, AsyncMultiLoraGRPOPipeline, create_cpu_actor
from .scheduler import ContextSchedulePolicy, ContextScheduler, ScheduleCandidate, SchedulerConfig
from .types import (ClaimedBatch, LoraContext, MetricEvent, PartitionAdmission, PreparedPartition, PromptGroup,
                    RolloutPolicy)
from .workers import AdvantageWorker, RolloutWorker, TrainerWorker

__all__ = [
    'AdvantageWorker',
    'AsyncMultiLoraGRPOConfig',
    'AsyncMultiLoraGRPOPipeline',
    'ClaimedBatch',
    'CompositeMetricsRecorder',
    'ContextSchedulePolicy',
    'ContextScheduler',
    'ContextStatus',
    'ContextGRPOGroupNSampler',
    'JSONLMetricsRecorder',
    'LoraContext',
    'LoraContextManager',
    'MetricEvent',
    'PartitionAdmission',
    'MetricsBuffer',
    'PartitionStatus',
    'PreparedPartition',
    'PromptGroup',
    'RolloutPolicy',
    'RolloutWorker',
    'ScheduleCandidate',
    'SchedulerConfig',
    'TQDataPlane',
    'TrainerWorker',
    'create_cpu_actor',
    'VLLMSamplerTQ',
]


def __getattr__(name: str):
    if name == 'ContextGRPOGroupNSampler':
        from .group_sampler import ContextGRPOGroupNSampler
        return ContextGRPOGroupNSampler
    if name == 'VLLMSamplerTQ':
        from .vllm_sampler_tq import VLLMSamplerTQ
        return VLLMSamplerTQ
    raise AttributeError(name)
