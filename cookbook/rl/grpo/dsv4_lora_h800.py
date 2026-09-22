# Copyright (c) ModelScope Contributors. All rights reserved.
"""H800, single-tenant DeepSeek-V4 GRPO; online synchronization never uses files.

See dsv4_lora_h800_read_me.md for resource placement and acceptance tests.
"""
import json
import os
import time
import torch
from pathlib import Path
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.model import MultiLoraTransformersModel
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.sampler import vLLMSampler
from twinkle.utils.platforms import ensure_npu_backend

TENANT = 'tenant_a'


def required_path(name):
    path = Path(os.environ[name]).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'{name}: {path}')
    return str(path)


def build_workers(model_cls=MultiLoraTransformersModel,
                  sampler_cls=vLLMSampler,
                  extra_engine_args=None,
                  *,
                  backend='nvidia'):
    """The two base paths must refer to the same model revision/layer subset."""
    from transformers import AutoConfig
    if backend not in ('nvidia', 'ascend'):
        raise ValueError('backend must be nvidia or ascend')
    is_npu = backend == 'ascend'
    if is_npu:
        ensure_npu_backend()
        if not torch.npu.is_available():
            raise RuntimeError('Ascend example requires an available torch.npu runtime')
    actor_path = required_path('ACTOR_MODEL')
    rollout_path = required_path('ROLLOUT_MODEL')
    actor_devices = int(os.environ.get('ACTOR_NPUS' if is_npu else 'ACTOR_GPUS', '4' if is_npu else '8'))
    rollout_tp = int(os.environ.get('ROLLOUT_TP', '4'))
    per_node = int(os.environ.get('NPUS_PER_NODE' if is_npu else 'GPUS_PER_NODE', '16' if is_npu else '8'))
    if min(actor_devices, rollout_tp, per_node) <= 0:
        raise ValueError('Actor, rollout and per-node device counts must be positive')
    # Default: put rollout on the next node. A reduced-model single-node test
    # can start rollout immediately after the actor devices if TP fits that node.
    start = int(os.environ.get('ROLLOUT_START_RANK', str(((actor_devices + per_node - 1) // per_node) * per_node)))
    if start < actor_devices or start // per_node != (start + rollout_tp - 1) // per_node:
        raise ValueError('Use disjoint actor/rollout devices and keep rollout TP on one node')
    ep = int(os.environ.get('ACTOR_EP', str(actor_devices)))
    if ep <= 0 or actor_devices % ep:
        raise ValueError('ACTOR_EP must be positive and divide the actor device count')
    rank, alpha = int(os.environ.get('LORA_R', '8')), int(os.environ.get('LORA_ALPHA', '32'))
    precision = os.environ.get('ACTOR_PRECISION', 'bf16')
    if precision not in ('bf16', 'fp16'):
        raise ValueError('ACTOR_PRECISION must be bf16 or fp16')
    config = AutoConfig.from_pretrained(actor_path, local_files_only=True, trust_remote_code=True)
    if config.n_routed_experts % ep:
        raise ValueError('ACTOR_EP must divide the model routed expert count')
    config.use_cache = False
    device_type, platform = ('npu', 'NPU') if is_npu else ('cuda', 'GPU')
    actor_mesh = DeviceMesh.from_sizes(fsdp_size=actor_devices, dp_size=1, ep_size=ep, device_type=device_type)
    rollout_mesh = DeviceMesh.from_sizes(world_size=rollout_tp, dp_size=1, tp_size=rollout_tp, device_type=device_type)
    twinkle.initialize(
        mode='ray',
        nproc_per_node=per_node,
        lazy_collect=False,
        groups=[
            DeviceGroup(name='actor', ranks=list(range(actor_devices)), device_type=platform),
            DeviceGroup(
                name='rollout',
                ranks=list(range(start, start + rollout_tp)),
                device_type=platform,
                gpus_per_worker=rollout_tp),
        ])
    # One small, inactive PEFT placeholder per layer establishes the container;
    # tenant target_modules=[] excludes these slots from training/export.
    placeholder = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0, target_modules=['q_a_proj'])
    model = model_cls(
        model_id=actor_path,
        config=config,
        remote_group='actor',
        device_mesh=actor_mesh,
        dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
        strategy='native_fsdp',
        mixed_precision=precision,
        memory_efficient_init=True,
        max_loras=1,
        max_r=rank,
        max_length=8192,
        lora_config=placeholder,
        fsdp_config={'expert_parallel': {
            'enabled': ep > 1,
            'router_dtype': 'fp32'
        }})
    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0,
        target_modules=[],
        target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'])
    model.add_adapter_to_model(TENANT, lora, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=float(os.environ.get('LR', '1e-5')), adapter_name=TENANT)
    model.set_loss('GRPOLoss', beta=0.0, epsilon=0.2, adapter_name=TENANT)
    model.set_processor(InputProcessor, adapter_name=TENANT)
    model.set_template('DeepseekV4Template', model_id=actor_path, adapter_name=TENANT)
    weight_adapter = dict(
        class_path='twinkle.sampler.vllm_sampler.dsv4_lora.DeepSeekV4NvidiaLoraAdapter', options={})
    if is_npu:
        weight_adapter = dict(
            class_path='twinkle.sampler.vllm_sampler.dsv4_lora_ascend.DeepSeekV4AscendQuaRotLoraAdapter',
            options=dict(training_base_model=actor_path, recipe='w8a8_dynamic_quarot_v1'))
    engine_args = dict(
        tensor_parallel_size=rollout_tp,
        enable_lora=True,
        weight_adapter=weight_adapter,
        lora_dtype='bfloat16' if precision == 'bf16' else 'float16',
        max_loras=1,
        max_lora_rank=rank,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_dtype='auto' if is_npu else 'fp8',
        max_model_len=int(os.environ.get('MAX_MODEL_LEN', '4096')),
        max_num_seqs=int(os.environ.get('MAX_NUM_SEQS', '4')),
        gpu_memory_utilization=float(os.environ.get('GPU_MEMORY_UTILIZATION', '0.85')))
    if is_npu:
        engine_args.update(block_size=128, max_num_batched_tokens=int(os.environ.get('MAX_NUM_BATCHED_TOKENS', '4096')))
    engine_args.update(extra_engine_args or {})
    sampler = sampler_cls(
        model_id=rollout_path, remote_group='rollout', device_mesh=rollout_mesh, engine_args=engine_args)
    sampler.set_template('DeepseekV4Template', model_id=actor_path)
    return model, sampler, CheckpointEngineManager(model, sampler, platform=platform)


def local_gsm8k():
    from datasets import load_dataset
    path = Path(required_path('GSM8K_PATH'))
    files = sorted(path.glob('train*.parquet')) if path.is_dir() else [path]
    if not files or any(p.suffix != '.parquet' for p in files):
        raise ValueError('GSM8K_PATH must be a train Parquet file or a directory containing train*.parquet')
    return load_dataset('parquet', data_files=[str(p) for p in files], split='train')


def local_dapo():
    from datasets import load_dataset
    path = Path(required_path('DAPO_PATH'))
    if path.is_dir():
        path = path / 'dapo-math-17k.parquet'
    if not path.is_file() or path.suffix != '.parquet':
        raise ValueError('DAPO_PATH must be dapo-math-17k.parquet or its containing directory')
    dataset = load_dataset('parquet', data_files=str(path), split='train')
    required = {'prompt', 'reward_model'}
    if not required.issubset(dataset.column_names):
        raise ValueError(f'DAPO dataset is missing columns: {sorted(required - set(dataset.column_names))}')
    return dataset


def actual_logprobs(sequence):
    if sequence.logprobs is None or len(sequence.logprobs) != len(sequence.tokens):
        raise RuntimeError('Missing actual rollout token logprobs')
    return [dict(entries)[token] for token, entries in zip(sequence.tokens, sequence.logprobs)]


def main(worker_builder=None):
    dataset_kind = os.environ.get('DATASET_KIND', 'gsm8k').lower()
    if dataset_kind == 'dapo':
        from .dsv4_dapo import DAPOMathAccuracyReward, DAPOMathProcessor
        dataset = local_dapo()
        processor, reward_fn = DAPOMathProcessor(), DAPOMathAccuracyReward()
    elif dataset_kind == 'gsm8k':
        dataset = local_gsm8k()
        processor, reward_fn = GSM8KProcessor(), GSM8KAccuracyReward()
    else:
        raise ValueError('DATASET_KIND must be gsm8k or dapo')
    max_rows = int(os.environ.get('DATASET_MAX_ROWS', '0'))
    if max_rows < 0:
        raise ValueError('DATASET_MAX_ROWS must be nonnegative')
    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    steps, batch, generations = (int(os.environ.get(k, default))
                                 for k, default in [('STEPS', '3'), ('BATCH_SIZE', '4'), ('NUM_GENERATIONS', '4')])
    if min(steps, batch) <= 0 or generations < 2 or len(dataset) < steps * batch:
        raise ValueError('Require positive steps/batch, >=2 generations and enough unrepeated dataset rows')
    output = Path(os.environ.get('REPORT_DIR', './dsv4_grpo_reports')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model, sampler, manager = (worker_builder or build_workers)()
    advantage_fn = GRPOAdvantage()
    if (batch * generations) % model.device_mesh.data_world_size:
        raise ValueError('BATCH_SIZE * NUM_GENERATIONS must be divisible by actor data_world_size')
    params = SamplingParams(
        max_tokens=int(os.environ.get('MAX_NEW_TOKENS', '512')),
        num_samples=generations,
        logprobs=1,
        temperature=1.0,
        top_p=1.0,
        top_k=-1)
    for step in range(steps):
        sync_start = time.monotonic()
        manager.sync_weights(
            merge_and_sync=False,
            lora_only=True,
            adapter_name=TENANT)
        report = dict(training_step=step, sync_seconds=time.monotonic() - sync_start)
        prompts = [processor.preprocess(dataset[i]) for i in range(step * batch, (step + 1) * batch)]
        responses = sampler.sample(prompts, params)
        if len(responses) != batch:
            raise RuntimeError('Missing rollout responses')
        features, old_logps, reward_inputs, samples = [], [], [], []
        for prompt, response in zip(prompts, responses):
            if len(response.sequences) != generations:
                raise RuntimeError('Unexpected number of rollout sequences')
            for sequence in response.sequences:
                features.append(sequence.new_input_feature)  # actual token IDs, never re-encode decoded text
                old_logps.append(actual_logprobs(sequence))
                reward_inputs.append(
                    dict(messages=[{
                        'role': 'assistant',
                        'content': sequence.decoded
                    }], user_data=prompt['user_data']))
                samples.append(
                    dict(
                        tokens=sequence.tokens,
                        logprobs=old_logps[-1],
                        text=sequence.decoded,
                        training_step=step,
                        prompt_tokens=response.prompt_token_ids))
        rewards = reward_fn(reward_inputs)
        advantages = advantage_fn(rewards, num_generations=generations, scale='group').tolist()
        result = model.forward_backward(
            inputs=features, old_logps=old_logps, advantages=advantages, micro_batch_size=1, adapter_name=TENANT)
        model.clip_grad_and_step(adapter_name=TENANT)
        report.update(rewards=rewards, advantages=advantages, samples=samples)
        (output / f'round_{step}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(
            f'round={step} mean_reward={sum(rewards)/len(rewards):.4f} '
            f'nonzero_advantages={sum(a != 0 for a in advantages)} result={result}',
            flush=True)
    # Synchronize the last optimizer update too, without adding a fourth training round.
    sync_start = time.monotonic()
    manager.sync_weights(
        merge_and_sync=False,
        lora_only=True,
        adapter_name=TENANT)
    final = dict(training_step=steps, sync_seconds=time.monotonic() - sync_start)
    (output / 'final_sync.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    if os.environ.get('FINAL_CHECKPOINT_DIR'):
        model.save('dsv4-grpo-final', output_dir=os.environ['FINAL_CHECKPOINT_DIR'], adapter_name=TENANT)
    print(f'Reports: {output}', flush=True)


if __name__ == '__main__':
    main()
