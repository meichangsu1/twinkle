#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real GRPO training: GSM8K or synthetic multi-turn prefix-cache / affinity trials.

Run from the repository root; see README_grpo_sticky_session.md. Each trial
runs in a fresh process with a fresh trainer, sampler pool and optimizer.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import logging
import math
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import json

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_sticky_session import prompt_key, request_metrics

GROUPS = {'A': (False, False), 'B': (False, True), 'D': (True, True)}
STAGES = ('sync_s', 'cache_reset_s', 'sample_s', 'reward_advantage_s', 'train_s')
logger = logging.getLogger(__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def expand_groups(prompts, generations, affinity):
    """IDs belong to this sampling batch, never to input features or engine IDs."""
    expanded, sessions = [], []
    for prompt in prompts:
        sid = uuid.uuid4().hex if affinity else None
        expanded.extend([prompt] * generations)
        sessions.extend([sid] * generations)
    return expanded, sessions if affinity else None


def observe_engine(engine, observations, instance_id):
    """Observe each native request separately, including identical prompt copies."""
    original = engine.generate

    async def generate(*args, **kwargs):
        started = time.perf_counter()
        final = None
        async for output in original(*args, **kwargs):
            final = output
            yield output
        if final is not None:
            observations.append({
                **request_metrics(final),
                'instance_id': instance_id,
                'engine_request_id': final.request_id,
                'input_hash': prompt_key(final.prompt_token_ids),
                'input_tokens': len(final.prompt_token_ids),
                'output_tokens': sum(len(output.token_ids) for output in final.outputs),
                'engine_call_s': time.perf_counter() - started,
            })

    engine.generate = generate


def observed_classes():
    """Benchmark instrumentation only; production sampler and router stay unchanged."""
    from twinkle import remote_class, remote_function
    from twinkle.model import TransformersModel
    from twinkle.sampler import vLLMSampler

    @remote_class()
    class TimedModel(TransformersModel):

        @remote_function(collect='first', lazy_collect=False)
        def synchronize(self):
            from twinkle.utils.framework import Torch
            Torch.synchronize()

    @remote_class(max_concurrency=24)
    class ObservedSampler(vLLMSampler):

        def __init__(self, model_id, engine_args=None, device_mesh=None, **kwargs):
            super().__init__(model_id, engine_args=engine_args, device_mesh=device_mesh, **kwargs)
            import ray
            self._benchmark_rows = []
            self._benchmark_instance = ray.get_runtime_context().get_actor_id()
            self._benchmark_instance = str(self._benchmark_instance)
            observe_engine(self.engine.engine, self._benchmark_rows, self._benchmark_instance)

        async def _sample_single(self, *args, **kwargs):
            # Benchmark-only correlation: identical prompt copies still have distinct
            # native engine requests. Keep routing IDs out of the engine API.
            kwargs['request_id'] = kwargs.get('request_id') or uuid.uuid4().hex
            response = await super()._sample_single(*args, **kwargs)
            response._benchmark_request_id = kwargs['request_id']
            return response

        @remote_function(collect='flatten', lazy_collect=False)
        def take_observations(self):
            # Run on the engine loop after sample completes, without a cross-thread race.
            async def take():
                rows = self._benchmark_rows[:]
                self._benchmark_rows.clear()
                return rows
            return self._run_in_loop(take())

        @remote_function(collect='flatten', lazy_collect=False)
        def benchmark_info(self):
            import os

            import ray
            context = ray.get_runtime_context()
            return [{
                'instance_id': self._benchmark_instance,
                'node_id': context.get_node_id(),
                'accelerator_ids': context.get_accelerator_ids(),
                'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
                'ascend_visible_devices': os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                'enable_prefix_caching': bool(self.engine.engine.vllm_config.cache_config.enable_prefix_caching),
                'logprobs_mode': self.engine.engine.vllm_config.model_config.logprobs_mode,
            }]

    return TimedModel, ObservedSampler


def prepare_inputs(config):
    """Freeze encoded real dataset inputs (including reward labels) before any trial."""
    if config.get('task') == 'synthetic':
        from scripts.synthetic_grpo_workload import prepare_inputs as prepare_synthetic
        return prepare_synthetic(config)
    from datasets import load_dataset

    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor.llm import GSM8KProcessor

    if config['input_plan']:
        inputs = read_inputs(config['input_plan'], config)
    else:
        directory = Path(config['dataset_id']).expanduser().resolve()
        data = load_dataset(str(directory), split='train')
        dataset = Dataset(DatasetMeta(data=data))
        dataset.map(GSM8KProcessor())
        dataset.set_template('Template', model_id=config['model'], max_length=config['max_prompt_tokens'],
                             truncation_strategy='delete', enable_thinking=False)
        dataset.encode(add_generation_prompt=True)
        count = config['steps'] * config['batch_size']
        if len(dataset) < count:
            raise ValueError(f'Need {count} prompts after length filtering, found {len(dataset)}; reduce --steps')
        # Keep this order fixed across seeds as well as experimental groups.
        inputs = [dataset[index] for index in range(count)]

    def serializable(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(f'Cannot serialize {type(value).__name__}')

    with Path(config['plan_file']).open('x') as stream:
        for feature in inputs:
            stream.write(json.dumps(feature, sort_keys=True, default=serializable, allow_nan=False) + '\n')


def read_inputs(path, config):
    data = Path(path).read_bytes()
    if config.get('plan_hash') and hashlib.sha256(data).hexdigest() != config['plan_hash']:
        raise ValueError('Input plan hash changed since experiment preparation')
    inputs = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    expected = config['steps'] * config['batch_size']
    if len(inputs) != expected:
        raise ValueError(f'Input plan must have exactly {expected} prompts; found {len(inputs)}')
    for index, feature in enumerate(inputs):
        tokens = feature['input_ids']
        if not tokens or len(tokens) > config['max_prompt_tokens']:
            raise ValueError(f'Invalid prompt length at input plan row {index}')
        if not dict(feature.get('user_data') or []).get('ground_truth'):
            raise ValueError(f'Missing reward ground truth at input plan row {index}')
    return inputs


def summarize_observations(rows, prompts, generations):
    expected = Counter({})
    for prompt in prompts:
        expected[prompt_key(prompt['input_ids'])] += generations
    if Counter(row['input_hash'] for row in rows) != expected:
        raise RuntimeError('Engine observations do not match the submitted prompt multiset')
    if any(row['output_tokens'] < 1 for row in rows):
        raise RuntimeError('Engine returned an empty completion')
    known = [row for row in rows if row['cached_tokens'] is not None]
    total_input = sum(row['input_tokens'] for row in rows)
    replicas = defaultdict(set)
    per_instance = {}
    for row in rows:
        replicas[row['input_hash']].add(row['instance_id'])
        instance = per_instance.setdefault(row['instance_id'], {'requests': 0, 'input_tokens': 0, 'output_tokens': 0})
        instance['requests'] += 1
        instance['input_tokens'] += row['input_tokens']
        instance['output_tokens'] += row['output_tokens']
    # Duplicate dataset rows cannot be attributed to individual routing groups by hash.
    unique = [key for key, count in expected.items() if count == generations]
    return {
        'input_tokens': total_input,
        'output_tokens': sum(row['output_tokens'] for row in rows),
        'cached_tokens': sum(row['cached_tokens'] for row in known) if len(known) == len(rows) else None,
        'cache_metric_coverage': len(known) / len(rows),
        'cache_hit_fraction': (sum(row['cached_tokens'] for row in known) / total_input
                               if len(known) == len(rows) else None),
        'group_replica_counts': [len(replicas[key]) for key in unique],
        'ambiguous_prompt_groups': len(prompts) - len(unique),
        'per_instance': per_instance,
    }


def train_batches(config, inputs, model, sampler, checkpoint, sampling_params, reward_fn, advantage_fn,
                  steps_stream, requests_stream, router_status):
    """Real update loop, injectable so lifecycle and update order can be verified on CPU."""
    if config.get('task') == 'synthetic':
        from scripts.synthetic_grpo_workload import train_batches as train_synthetic
        return train_synthetic(config, inputs, model, sampler, checkpoint, sampling_params, advantage_fn,
                               steps_stream, requests_stream, router_status)
    affinity, _ = GROUPS[config['group']]
    generations = config['num_generations']
    records = []
    for step in range(config['steps']):
        begin = time.perf_counter()
        row = {'step': step, 'measured': step >= config['warmup_steps']}

        def timed(name, action, row=row):
            started = time.perf_counter()
            result = action()
            row[name] = time.perf_counter() - started
            return result

        # All variants keep identical weight/cache invalidation semantics.
        timed('sync_s', lambda: checkpoint.sync_weights(merge_and_sync=False))
        timed('cache_reset_s', sampler.reset_prefix_cache)
        start = step * config['batch_size']
        # Some templates mutate features when appending completions. Freeze the replay plan.
        prompts = copy.deepcopy(inputs[start:start + config['batch_size']])
        expanded, sessions = expand_groups(prompts, generations, affinity)

        def sample(expanded=expanded, sessions=sessions):
            try:
                return sampler.sample(expanded, sampling_params, session_ids=sessions)
            finally:
                if sessions:
                    sampler.release_sessions(list(dict.fromkeys(sessions)))

        responses = timed('sample_s', sample)
        observations = sampler.take_observations()
        for observation in observations:
            requests_stream.write(json.dumps({'step': step, **observation}, allow_nan=False) + '\n')
        requests_stream.flush()
        row.update(summarize_observations(observations, prompts, generations))
        status = router_status()
        row['router_after'] = status
        if status['active_leases'] or status['session_count']:
            raise RuntimeError(f'Routing state leaked after sampling: {status}')
        if affinity and any(count != 1 for count in row['group_replica_counts']):
            raise RuntimeError('A GRPO group was routed to more than one instance with affinity enabled')

        def score(responses=responses, expanded=expanded, row=row):
            if len(responses) != len(expanded) or any(len(response.sequences) != 1 for response in responses):
                raise RuntimeError('Expected exactly one completion per expanded input')
            sequences = [response.sequences[0] for response in responses]
            features, old_logps = [], []
            for sequence in sequences:
                if (not sequence.tokens or not sequence.logprobs
                        or len(sequence.logprobs) != len(sequence.tokens)):
                    raise RuntimeError('Sampled tokens and old logprobs are empty or misaligned')
                features.append(sequence.new_input_feature)
                old_logps.append([values[0][1] for values in sequence.logprobs])
            rewards, accuracy = reward_fn(features)
            advantages = advantage_fn(rewards, num_generations=generations, scale='group').tolist()
            row['reward_mean'] = statistics.mean(rewards)
            row['accuracy'] = statistics.mean(accuracy)
            row['completion_tokens_mean'] = statistics.mean(len(sequence.tokens) for sequence in sequences)
            row['length_stop_fraction'] = (sum(sequence.stop_reason == 'length' for sequence in sequences)
                                           / len(sequences))
            row['nonzero_advantage_fraction'] = sum(abs(value) > 1e-8 for value in advantages) / len(advantages)
            if sum(len(sequence.tokens) for sequence in sequences) != row['output_tokens']:
                raise RuntimeError('Engine and sampler output token counts disagree')
            return features, old_logps, advantages

        features, old_logps, advantages = timed('reward_advantage_s', score)

        def update(features=features, old_logps=old_logps, advantages=advantages, row=row):
            count = 0
            for offset in range(0, len(features), config['mini_batch_size']):
                end = offset + config['mini_batch_size']
                model.forward_backward(inputs=features[offset:end], old_logps=old_logps[offset:end],
                                       advantages=advantages[offset:end], micro_batch_size=config['micro_batch_size'])
                model.clip_grad_and_step()
                count += 1
            # A Ray return with None does not by itself guarantee device kernels finished.
            model.synchronize()
            row['model_metrics'] = model.calculate_metric(is_training=True)
            return count

        row['optimizer_updates'] = timed('train_s', update)
        row['total_s'] = time.perf_counter() - begin
        row['trajectories'] = len(expanded)
        row['overhead_s'] = row['total_s'] - sum(row[name] for name in STAGES)
        row['prompt_hashes'] = [prompt_key(prompt['input_ids']) for prompt in prompts]
        steps_stream.write(json.dumps(row, allow_nan=False) + '\n')
        steps_stream.flush()
        records.append(row)
        logger.info('%s seed=%s batch=%s/%s total=%.3fs sample=%.3fs train=%.3fs reward=%.3f cache=%s',
                    config['group'], config['seed'], step + 1, config['steps'], row['total_s'], row['sample_s'],
                    row['train_s'], row['reward_mean'], row['cache_hit_fraction'])
    return records


def summarize_trial(config, records):
    measured = [row for row in records if row['measured']]
    if not measured:
        raise ValueError('No measured training batches')
    sums = {name: sum(row[name] for row in measured) for name in (*STAGES, 'overhead_s', 'total_s')}
    trajectories = sum(row['trajectories'] for row in measured)
    inputs = sum(row['input_tokens'] for row in measured)
    outputs = sum(row['output_tokens'] for row in measured)
    cached = [row['cached_tokens'] for row in measured]
    summary = {
        'status': 'complete', 'group': config['group'], 'seed': config['seed'],
        'plan_hash': config['plan_hash'], 'config': config,
        'measured_batches': len(measured), 'optimizer_updates': sum(row['optimizer_updates'] for row in measured),
        'all_batches': len(records), 'all_optimizer_updates': sum(row['optimizer_updates'] for row in records),
        'warmup_totals_s': {name: sum(row[name] for row in records if not row['measured'])
                            for name in (*STAGES, 'overhead_s', 'total_s')},
        'stage_totals_s': sums, 'trajectories': trajectories,
        'sample_trajectories_per_s': trajectories / sums['sample_s'],
        'train_trajectories_per_s': trajectories / sums['total_s'],
        'output_tokens_per_sample_s': outputs / sums['sample_s'],
        'input_tokens': inputs, 'output_tokens': outputs,
        'cache_hit_fraction': sum(cached) / inputs if all(value is not None for value in cached) else None,
        'cache_metric_coverage': sum(row['cache_metric_coverage'] * row['trajectories']
                                     for row in measured) / trajectories,
        **{name: statistics.mean(row[name] for row in measured) for name in
           ('reward_mean', 'accuracy', 'completion_tokens_mean', 'length_stop_fraction', 'nonzero_advantage_fraction')},
        'mean_replicas_per_prompt': statistics.mean(
            count for row in measured for count in row['group_replica_counts'])
        if any(row['group_replica_counts'] for row in measured) else None,
    }
    if config.get('task') == 'synthetic':
        from scripts.synthetic_grpo_workload import summarize_multiturn
        summary.update(summarize_multiturn(measured))
    return summary


def run_training(config):
    startup_begin = time.perf_counter()
    if config['device'] == 'npu':
        from twinkle.utils.platforms.npu import ensure_npu_backend
        ensure_npu_backend()

    import ray
    import torch
    from peft import LoraConfig

    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.advantage import GRPOAdvantage
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from twinkle.data_format import SamplingParams
    from twinkle.processor import InputProcessor

    platform = 'NPU' if config['device'] == 'npu' else 'GPU'
    accelerator = torch.npu if platform == 'NPU' else torch.cuda
    needed = config['model_devices'] + config['replicas']
    if not accelerator.is_available() or accelerator.device_count() < needed:
        raise RuntimeError(f'Need {needed} visible {platform} devices ({config["model_devices"]} train + '
                           f'{config["replicas"]} sample), found {accelerator.device_count()}')
    inputs = read_inputs(config['plan_file'], config)
    _, cache = GROUPS[config['group']]
    output = Path(config['trial_dir'])
    ray.init(namespace=f'grpo-sticky-{uuid.uuid4().hex}',
             resources={'NPU': accelerator.device_count()} if platform == 'NPU' else None)
    try:
        groups = [DeviceGroup(name='model', ranks=list(range(config['model_devices'])), device_type=platform),
                  DeviceGroup(name='sampler', ranks=list(range(config['model_devices'], needed)), device_type=platform)]
        twinkle.initialize(mode='ray', nproc_per_node=needed, groups=groups, lazy_collect=False,
                           extra_torch_kwargs={'seed': config['seed']})
        model_cls, sampler_cls = observed_classes()
        model = model_cls(model_id=config['model'], remote_group='model', strategy='accelerate',
                          device_mesh=DeviceMesh.from_sizes(world_size=config['model_devices'],
                                                            dp_size=config['model_devices'],
                                                            device_type=config['device']),
                          mixed_precision='bf16', torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        # Text attention/MLP and Qwen3.5 linear attention only; avoid vision adapters.
        targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                   'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj']
        model.add_adapter_to_model('default', LoraConfig(target_modules=targets, r=config['lora_rank'],
                                                       lora_alpha=2 * config['lora_rank'], lora_dropout=0.0),
                                   gradient_accumulation_steps=1)
        model.set_optimizer('AdamW', lr=config['learning_rate'])
        updates = config['steps'] * config['batch_size'] * config['num_generations'] // config['mini_batch_size']
        model.set_lr_scheduler('CosineAnnealingLR', T_max=updates, eta_min=0)
        model.set_loss('GRPOLoss', epsilon=0.2)
        model.set_processor(InputProcessor)
        model.set_template('Template', model_id=config['model'], max_length=config['max_model_len'],
                           enable_thinking=False)
        engine_args = {
            'tensor_parallel_size': 1, 'enable_prefix_caching': cache,
            'gpu_memory_utilization': config['gpu_memory_utilization'],
            'max_model_len': config['max_model_len'], 'max_num_seqs': config['max_num_seqs'],
            'enable_lora': True, 'max_lora_rank': config['lora_rank'], 'seed': config['seed'],
            'disable_log_stats': False, 'enforce_eager': config['enforce_eager'],
            # Match the trainer's unwarped model probabilities, independent of sampling top-p/temperature.
            'logprobs_mode': 'raw_logprobs',
        }
        if config['language_model_only']:
            engine_args['language_model_only'] = True
        sampler = sampler_cls(model_id=config['model'], engine_args=engine_args, remote_group='sampler',
                              device_mesh=DeviceMesh.from_sizes(world_size=config['replicas'],
                                                                dp_size=config['replicas'],
                                                                device_type=config['device']),
                              routing_config={'max_sessions': config['batch_size'] * (
                                  config['num_generations'] if config.get('task') == 'synthetic' else 1)})
        sampler.set_template('Template', model_id=config['model'], max_length=config['max_model_len'],
                             enable_thinking=False)
        info = sampler.benchmark_info()
        if len(info) != config['replicas'] or any(item['enable_prefix_caching'] != cache for item in info):
            raise RuntimeError(f'Unexpected sampler pool / prefix cache configuration: {info}')
        if any(item['logprobs_mode'] != 'raw_logprobs' for item in info):
            raise RuntimeError(f'Engine must return raw model logprobs for GRPO training: {info}')
        write_json(output / 'instances.json', info)
        checkpoint = CheckpointEngineManager(model=model, sampler=sampler, platform=platform)
        # Do not give all n=1 copies the same per-request seed: that collapses GRPO diversity.
        sampling_params = SamplingParams(num_samples=1, max_tokens=config['max_tokens'], logprobs=1,
                                         temperature=config['temperature'], top_p=config['top_p'])
        reward_fn = None
        if config.get('task') != 'synthetic':
            from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
            accuracy_fn, format_fn = GSM8KAccuracyReward(), GSM8KFormatReward()

            def reward_fn(features):
                accuracy, formats = accuracy_fn(features), format_fn(features)
                return [a + f for a, f in zip(accuracy, formats, strict=True)], accuracy

        def router_status():
            return ray.get(sampler._request_router.get_status.remote())

        model.synchronize()
        startup_s = time.perf_counter() - startup_begin
        training_begin = time.perf_counter()
        with ((output / 'steps.jsonl').open('x') as steps_stream,
              (output / 'requests.jsonl').open('x') as requests_stream):
            records = train_batches(config, inputs, model, sampler, checkpoint, sampling_params, reward_fn,
                                    GRPOAdvantage(), steps_stream, requests_stream, router_status)
        summary = summarize_trial(config, records)
        summary['startup_s'] = startup_s
        summary['training_wall_s_including_warmup'] = time.perf_counter() - training_begin
        checkpoint_begin = time.perf_counter()
        if config['save_checkpoint']:
            model.save('final', output_dir=str(output / 'checkpoint'))
        summary['checkpoint_save_s'] = time.perf_counter() - checkpoint_begin if config['save_checkpoint'] else 0.0
        write_json(output / 'summary.json', summary)
    finally:
        ray.shutdown()


def paired_comparisons(trials):
    completed = {(trial['seed'], trial['group']): trial for trial in trials if trial['status'] == 'complete'}
    result = []
    for seed in sorted({key[0] for key in completed}):
        for baseline, candidate in (('A', 'B'), ('B', 'D'), ('A', 'D')):
            if (seed, baseline) not in completed or (seed, candidate) not in completed:
                continue
            before, after = completed[seed, baseline], completed[seed, candidate]
            if (before['plan_hash'], before['trajectories'], before['optimizer_updates']) != (
                    after['plan_hash'], after['trajectories'], after['optimizer_updates']):
                raise ValueError('Paired trials have different inputs or training budgets')
            result.append({
                'seed': seed, 'comparison': f'{candidate}/{baseline}',
                'sample_throughput_ratio': after['sample_trajectories_per_s'] / before['sample_trajectories_per_s'],
                'train_throughput_ratio': after['train_trajectories_per_s'] / before['train_trajectories_per_s'],
                'total_time_reduction': 1 - after['stage_totals_s']['total_s'] / before['stage_totals_s']['total_s'],
                'output_tokens_ratio': after['output_tokens'] / before['output_tokens'],
                'accuracy_delta': after['accuracy'] - before['accuracy'],
            })
    return result


def launch_process(flag, config_file, log_file):
    with Path(log_file).open('x') as stream:
        result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), flag, str(config_file)],
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = '\n'.join(Path(log_file).read_text(errors='replace').splitlines()[-150:])
        raise RuntimeError(f'Child process exited {result.returncode}; inspect {log_file}\n{tail}')


def run_experiment(config):
    output = Path(config['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {**config, 'plan_file': str(output / 'inputs.jsonl')}
    packages = {}
    for name in ('twinkle-kit', 'torch', 'torch-npu', 'vllm', 'vllm-ascend', 'ray', 'transformers', 'peft'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    benchmark = ('synthetic_multiturn_grpo_affinity' if config.get('task') == 'synthetic'
                 else 'real_grpo_group_affinity')
    manifest = {'benchmark': benchmark, 'status': 'running', 'config': config,
                'packages': packages, 'trials': [], 'comparisons': []}
    write_json(output / 'config.json', config)
    write_json(output / 'summary.json', manifest)
    try:
        logger.info('Preparing fixed %s inputs; log: %s', config.get('task', 'gsm8k'), output / 'prepare.log')
        prepare_begin = time.perf_counter()
        launch_process('--prepare-config', output / 'config.json', output / 'prepare.log')
        manifest['prepare_wall_s'] = time.perf_counter() - prepare_begin
        config['plan_hash'] = hashlib.sha256(Path(config['plan_file']).read_bytes()).hexdigest()
        write_json(output / 'config.json', config)
        for repeat, seed in enumerate(config['seeds']):
            # Rotate order to avoid always measuring D last on a warming machine.
            offset = repeat % len(config['groups'])
            groups = config['groups'][offset:] + config['groups'][:offset]
            for group in groups:
                trial_dir = output / f'{group}-seed-{seed}'
                trial_dir.mkdir()
                trial = {**config, 'group': group, 'seed': seed, 'trial_dir': str(trial_dir)}
                write_json(trial_dir / 'config.json', trial)
                logger.info('Training group=%s seed=%s; log: %s', group, seed, trial_dir / 'run.log')
                try:
                    process_begin = time.perf_counter()
                    launch_process('--trial-config', trial_dir / 'config.json', trial_dir / 'run.log')
                except Exception as error:
                    failure = {'status': 'failed', 'group': group, 'seed': seed, 'error': str(error)}
                    write_json(trial_dir / 'error.json', failure)
                    manifest['trials'].append(failure)
                    raise
                trial_summary = json.loads((trial_dir / 'summary.json').read_text())
                trial_summary['process_wall_s'] = time.perf_counter() - process_begin
                manifest['trials'].append(trial_summary)
                manifest['comparisons'] = paired_comparisons(manifest['trials'])
                write_json(output / 'summary.json', manifest)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        write_json(output / 'summary.json', manifest)
    logger.info('Finished: %s', output / 'summary.json')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-config', help=argparse.SUPPRESS)
    parser.add_argument('--trial-config', help=argparse.SUPPRESS)
    parser.add_argument('--model')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', choices=['cuda', 'npu'], default='cuda')
    parser.add_argument('--groups', choices=list(GROUPS), nargs='+', default=['A', 'B', 'D'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--model-devices', type=int, default=1)
    parser.add_argument('--replicas', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50, help='Complete rollout batches, NOT optimizer updates')
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16, help='Distinct prompt groups per rollout batch')
    parser.add_argument('--num-generations', type=int, default=8)
    parser.add_argument('--mini-batch-size', type=int, default=32)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--lora-rank', type=int, choices=[8, 16, 32, 64], default=16)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.7)
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--language-model-only', action='store_true')
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--task', choices=['gsm8k', 'synthetic'], default='gsm8k')
    parser.add_argument('--prompt-tokens', type=int, default=3072,
                        help='Synthetic initial prompt target (including the chat template)')
    parser.add_argument('--turns', type=int, default=4, help='Fixed generation turns per synthetic episode')
    parser.add_argument('--feedback-tokens', type=int, default=256,
                        help='Per-episode synthetic observation payload tokens, excluding the next query')
    parser.add_argument('--feedback-delay-ms', type=float, default=100,
                        help='Synthetic waits are frozen uniformly between zero and this value')
    parser.add_argument('--episode-concurrency', type=int, default=8)
    parser.add_argument('--data-seed', type=int, default=2026, help='Synthetic plan seed, independent of training seeds')
    parser.add_argument('--dataset-id', help='Local directory containing GSM8K train/test Parquet files')
    parser.add_argument('--input-plan', help='Encoded input JSONL from a previous run; exact row count must match')
    args = parser.parse_args(argv)
    if args.prepare_config or args.trial_config:
        if args.prepare_config and args.trial_config:
            parser.error('Only one child operation is allowed')
        return args
    if not args.model or not args.output_dir:
        parser.error('--model and --output-dir are required')
    if args.task == 'gsm8k' and not args.dataset_id and not args.input_plan:
        parser.error('--dataset-id is required unless --input-plan is provided')
    if args.task == 'synthetic':
        if args.dataset_id:
            parser.error('--task synthetic generates its own inputs; omit --dataset-id')
        if args.prompt_tokens < 512 or args.prompt_tokens > args.max_prompt_tokens:
            parser.error('Synthetic --prompt-tokens must be >= 512 and <= --max-prompt-tokens')
        if args.turns < 2 or args.episode_concurrency < 1:
            parser.error('Synthetic experiments need at least two turns and positive episode concurrency')
        if args.feedback_tokens < 0:
            parser.error('--feedback-tokens must be nonnegative')
        if not math.isfinite(args.feedback_delay_ms) or args.feedback_delay_ms < 0:
            parser.error('--feedback-delay-ms must be finite and nonnegative')
        # Reserve room for feedback and chat separators as well as every completion.
        if (args.prompt_tokens + args.turns * args.max_tokens
                + (args.turns - 1) * (args.feedback_tokens + 256) > args.max_model_len):
            parser.error('--max-model-len must fit the synthetic prompt, all outputs and feedback '
                         '(reserve feedback payload + 256 tokens per feedback turn)')
    positive = ('model_devices', 'replicas', 'steps', 'batch_size', 'num_generations', 'mini_batch_size',
                'micro_batch_size', 'max_prompt_tokens', 'max_tokens', 'max_model_len', 'max_num_seqs')
    if any(getattr(args, name) < 1 for name in positive):
        parser.error('Device, batch, token and step counts must be positive')
    if args.replicas < 2 or args.num_generations < 2:
        parser.error('Need at least 2 sampler replicas and 2 generations per group')
    if not 0 <= args.warmup_steps < args.steps:
        parser.error('--warmup-steps must be >= 0 and less than --steps')
    if args.max_prompt_tokens + args.max_tokens > args.max_model_len:
        parser.error('--max-model-len must fit max prompt plus max output')
    total = args.batch_size * args.num_generations
    if (total % args.mini_batch_size or args.mini_batch_size % args.num_generations
            or args.mini_batch_size % args.model_devices or args.mini_batch_size < args.model_devices):
        parser.error('Mini-batches must divide the rollout batch and contain whole GRPO groups and trainer DP shards')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and > 0 for GRPO diversity')
    if not 0 < args.top_p <= 1 or not 0 < args.gpu_memory_utilization < 1:
        parser.error('Invalid top-p or device memory utilization')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be finite and > 0')
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Groups and seeds must be unique')
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    if args.prepare_config:
        prepare_inputs(json.loads(Path(args.prepare_config).read_text()))
    elif args.trial_config:
        run_training(json.loads(Path(args.trial_config).read_text()))
    else:
        run_experiment(vars(args))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
