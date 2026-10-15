#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fixed-prompt, multi-turn replay through Twinkle's continuous-work dispatcher.

GPU: python scripts/benchmark_sticky_session.py --model /path/to/model --output-dir /tmp/sticky-gpu
NPU: python scripts/benchmark_sticky_session.py --device npu --model /path/to/model --output-dir /tmp/sticky-npu
CPU: python scripts/benchmark_sticky_session.py --backend mock --episodes 8 --turns 3 \
         --prompt-tokens 64 --append-tokens 16 --max-tokens 4 --concurrency 4 \
         --repeats 1 --output-dir /tmp/sticky-smoke

Mock cache/timing data are synthetic and cannot establish accelerator performance gains.
This benchmark fixes weights and prompts; it does not run tools, training, or partial rollout.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import logging
import math
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import json

# (session affinity, prefix cache)
GROUPS = {'A': (False, False), 'B': (False, True), 'C': (True, False), 'D': (True, True)}
logger = logging.getLogger(__name__)


def local_ray_options(config):
    options = {'num_cpus': max(2, config['replicas'])}
    if config['backend'] == 'vllm' and config['device'] == 'npu':
        from twinkle.utils.platforms.npu import ensure_npu_backend
        ensure_npu_backend()
        import torch

        count = torch.npu.device_count() if torch.npu.is_available() else 0
        if not count:
            raise RuntimeError('No visible NPU; check torch_npu, CANN and ASCEND_RT_VISIBLE_DEVICES')
        options['resources'] = {'NPU': count}
    return options


def worker_options(config):
    """Reserve whole devices per replica; TP runs inside that replica on one node."""
    options = {'num_cpus': 1, 'num_gpus': 0, 'max_concurrency': max(config['concurrency']) + 2}
    if config['backend'] == 'vllm':
        if config['device'] == 'npu':
            options['resources'] = {'NPU': config['tensor_parallel_size']}
            visible_flag = 'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'
        else:
            options['num_gpus'] = config['tensor_parallel_size']
            visible_flag = 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'
        # Unlike Twinkle's placement-group workers, these raw actors let Ray pin devices.
        options['runtime_env'] = {'env_vars': {visible_flag: '0'}}
    return options


def worker_device_info(config):
    import ray

    context = ray.get_runtime_context()
    info = {'device': 'cpu', 'device_ids': [], 'node_id': context.get_node_id()}
    if config['backend'] == 'vllm':
        resource = 'NPU' if config['device'] == 'npu' else 'GPU'
        device_ids = [str(value) for value in context.get_accelerator_ids().get(resource, [])]
        if len(device_ids) != config['tensor_parallel_size']:
            raise RuntimeError(f'Expected {config["tensor_parallel_size"]} {resource} devices, got {device_ids}')
        env = 'ASCEND_RT_VISIBLE_DEVICES' if resource == 'NPU' else 'CUDA_VISIBLE_DEVICES'
        # Pin before importing the sampler/torch; never inherit the driver's full visible pool.
        os.environ[env] = ','.join(device_ids)
        info.update(device=config['device'], device_ids=device_ids, visible_device_env=env)
    return info


def prompt_key(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def build_workload(encode, episodes, turns, prompt_tokens, append_tokens, seed):
    """Unique early prefixes, then strictly append-only, frozen token inputs."""
    body = encode(' Document context: cache routing latency worker tokens observation. ')
    if not body:
        raise ValueError('Tokenizer produced an empty document')

    def fill(prefix, length):
        if len(prefix) > length:
            raise ValueError('Token budget is too small for the episode/turn marker')
        return prefix + (body * math.ceil((length - len(prefix)) / len(body)))[:length - len(prefix)]

    workload = []
    for episode in range(episodes):
        tokens = fill(encode(f'Episode {episode}, seed {seed}.\n'), prompt_tokens)
        prompts = [tokens]
        for turn in range(1, turns):
            suffix = fill(encode(f'\nObservation {turn}. '), append_tokens)
            tokens = tokens + suffix
            prompts.append(tokens)
        workload.append({'episode_id': str(episode), 'prompts': prompts})
    keys = [prompt_key(tokens) for episode in workload for tokens in episode['prompts']]
    if len(set(keys)) != len(keys):
        raise ValueError('Replay prompts must be unique for unambiguous request metric correlation')
    return workload


def _number(obj, name):
    value = getattr(obj, name, None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _difference(obj, end, start):
    end, start = _number(obj, end), _number(obj, start)
    if end is not None and start is not None and end >= start > 0:
        return end - start
    return None


def request_metrics(output):
    """Read supported RequestOutput metrics; absent fields stay null, never fabricated zeros.

    Newer vLLM has core-monotonic *_ts fields and a frontend-wall-clock arrival_time.
    Do not subtract timestamps from those two clock domains.
    """
    metrics = getattr(output, 'metrics', None)
    ttft = _number(metrics, 'first_token_latency')
    if ttft is None:
        ttft = _difference(metrics, 'first_token_time', 'arrival_time')
    prefill = _number(metrics, 'prefill_time')
    if prefill is None:
        prefill = _difference(metrics, 'first_token_ts', 'scheduled_ts')
    if prefill is None:
        prefill = _difference(metrics, 'first_token_time', 'first_scheduled_time')
    queue = _number(metrics, 'queued_time')
    if queue is None:
        queue = _difference(metrics, 'scheduled_ts', 'queued_ts')
    if queue is None:
        queue = _difference(metrics, 'first_scheduled_time', 'arrival_time')
    cached = _number(output, 'num_cached_tokens')
    return {
        'cached_tokens': int(cached) if cached is not None else None,
        'engine_ttft_s': ttft,
        'engine_prefill_s': prefill,
        'engine_queue_s': queue,
    }


class BenchmarkWorker:
    """Benchmark-only observer around a local vLLMSampler on each Ray replica."""

    def __init__(self, config, prefix_cache):
        import ray

        self.instance_id = str(ray.get_runtime_context().get_actor_id())
        self.device_info = worker_device_info(config)
        self.config = config
        self.prefix_cache = prefix_cache
        self.cache = []
        self.observations = {}
        self.lock = threading.Lock()
        self.active = 0
        self.sampler = None
        if config['backend'] == 'mock':
            return

        if config['device'] == 'npu':
            from twinkle.utils.platforms.npu import ensure_npu_backend
            ensure_npu_backend()

        import twinkle.infra as infra
        from twinkle.sampler.vllm_sampler import vLLMSampler

        # Ray owns the replica process; the sampler inside it directly owns its engine.
        infra.initialize(mode='local', lazy_collect=False)
        self.sampler = vLLMSampler(config['model'], engine_args={
            'tensor_parallel_size': config['tensor_parallel_size'],
            'gpu_memory_utilization': config['gpu_memory_utilization'],
            'max_model_len': config['max_model_len'],
            'enable_prefix_caching': prefix_cache,
            'disable_log_stats': False,
            'seed': config['seed'],
        })
        self.sampler.set_template('Template', model_id=config['model'])
        engine = self.sampler.engine.engine
        actual = engine.vllm_config.cache_config.enable_prefix_caching
        if bool(actual) != prefix_cache:
            raise RuntimeError(f'Prefix-cache control did not reach vLLM: expected {prefix_cache}, got {actual}')
        original_generate = engine.generate

        async def observed_generate(*args, **kwargs):
            started = time.perf_counter()
            first_output_s = None
            final = None
            async for output in original_generate(*args, **kwargs):
                final = output
                if first_output_s is None and any(value.token_ids for value in output.outputs):
                    first_output_s = time.perf_counter() - started
                yield output
            if final is not None:
                observation = {
                    **request_metrics(final),
                    'engine_request_id': final.request_id,
                    'first_output_s': first_output_s,
                    'engine_call_s': time.perf_counter() - started,
                }
                with self.lock:
                    self.observations[prompt_key(final.prompt_token_ids)] = observation

        engine.generate = observed_generate

    def workload(self):
        encode = ((lambda text: list(text.encode())) if self.sampler is None else
                  lambda text: self.sampler.template.tokenizer.encode(text, add_special_tokens=False))
        return build_workload(encode, self.config['episodes'], self.config['turns'],
                              self.config['prompt_tokens'], self.config['append_tokens'], self.config['seed'])

    def info(self):
        return {'instance_id': self.instance_id, 'prefix_cache': self.prefix_cache,
                'backend': self.config['backend'], **self.device_info}

    async def reset(self):
        if self.active:
            raise RuntimeError('Cannot reset a benchmark replica with requests running')
        if self.sampler is not None:
            await asyncio.to_thread(self.sampler.reset_prefix_cache)
        self.cache.clear()
        with self.lock:
            self.observations.clear()

    async def close(self):
        if self.active:
            raise RuntimeError('Cannot close a benchmark replica with requests running')
        if self.sampler is not None:
            await asyncio.to_thread(self.sampler.shutdown)

    async def _twinkle_async_sample(self, inputs):
        self.active += 1
        active_at_start = self.active
        try:
            if self.sampler is not None:
                results = await asyncio.to_thread(self._real_sample, inputs)
                for result in results:
                    result['worker_active_at_start'] = active_at_start
                return results
            results = []
            for feature in inputs:
                tokens = feature['input_ids']
                cached = 0
                if self.prefix_cache:
                    for previous in self.cache:
                        matched = 0
                        for left, right in zip(previous, tokens, strict=False):
                            if left != right:
                                break
                            matched += 1
                        cached = max(cached, matched // 16 * 16)
                # Synthetic cost only checks harness plumbing; not a vLLM cost model.
                started = time.perf_counter()
                await asyncio.sleep(0.001 + (len(tokens) - cached) / 100000)
                if self.prefix_cache:
                    self.cache.append(list(tokens))
                elapsed = time.perf_counter() - started
                results.append({
                    'instance_id': self.instance_id, 'input_tokens': len(tokens),
                    'output_tokens': self.config['max_tokens'], 'cached_tokens': cached,
                    'engine_prefill_s': None, 'engine_queue_s': None, 'engine_ttft_s': None,
                    'first_output_s': None, 'engine_call_s': elapsed,
                    'metrics_source': 'synthetic_mock',
                    'worker_active_at_start': active_at_start,
                })
            return results
        finally:
            self.active -= 1

    def _real_sample(self, inputs):
        from twinkle.data_format import SamplingParams

        started = time.perf_counter()
        responses = self.sampler.sample(inputs, sampling_params=SamplingParams(
            temperature=0.0, max_tokens=self.config['max_tokens'], seed=self.config['seed'],
        ))
        elapsed = time.perf_counter() - started
        results = []
        for response in responses:
            with self.lock:
                observation = self.observations.pop(prompt_key(response.prompt_token_ids))
            results.append({
                **observation, 'instance_id': self.instance_id,
                'input_tokens': len(response.prompt_token_ids),
                'output_tokens': sum(len(seq.tokens) for seq in response.sequences),
                'worker_sample_s': elapsed, 'metrics_source': 'vllm_request_output',
            })
        return results


class RoutedSampler:
    """Use the production dispatcher, router, leases and ordering without a trainer."""

    def __init__(self, workers, max_sessions, timeout):
        from twinkle.infra.ray_routing import get_handle_router

        self._actors = workers
        self._routing_config = {'max_sessions': max_sessions}
        self.timeout = timeout
        get_handle_router(self, workers)

    def sample(self, inputs, session_ids=None):
        from twinkle.infra import _run_continous_work
        from twinkle.infra._ray import RayHelper

        return _run_continous_work(self, 'sample', RayHelper.execute_all_async, self._actors,
                                  (inputs,), {}, len(inputs), self.timeout, session_ids=session_ids)

    def release_sessions(self, ids):
        from twinkle.infra.ray_routing import release_handle_sessions

        release_handle_sessions(self, ids)

    def status(self):
        import ray

        return ray.get(self._request_router.get_status.remote())


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(rows, wall_s, status):
    success = [row for row in rows if row.get('error') is None]
    known = [row for row in success if row.get('cached_tokens') is not None]
    continuation = [row for row in success if row['turn'] > 0]

    def hit_fraction(values):
        if not values or any(row.get('cached_tokens') is None for row in values):
            return None
        return sum(row['cached_tokens'] for row in values) / sum(row['input_tokens'] for row in values)

    by_episode = {}
    by_instance = {}
    for row in success:
        by_episode.setdefault(row['episode_id'], []).append(row)
        instance = by_instance.setdefault(row['instance_id'], {'requests': 0, 'input_tokens': 0,
                                                               'output_tokens': 0, 'engine_call_s': 0.0,
                                                               'max_active_at_start': 0})
        for key in ['input_tokens', 'output_tokens', 'engine_call_s']:
            instance[key] += row[key]
        instance['requests'] += 1
        instance['max_active_at_start'] = max(instance['max_active_at_start'], row.get('worker_active_at_start', 0))
    transitions = switches = 0
    episode_times = []
    for values in by_episode.values():
        values.sort(key=lambda row: row['turn'])
        for previous, current in zip(values, values[1:], strict=False):
            transitions += 1
            switches += previous['instance_id'] != current['instance_id']
        if values[-1].get('episode_finished'):
            episode_times.append(values[-1]['finished_s'] - values[0]['started_s'])
    completed = sum(bool(values[-1].get('episode_finished')) for values in by_episode.values())
    result = {
        'wall_s': wall_s, 'requests': len(rows), 'errors': len(rows) - len(success),
        'completed_episodes': completed, 'episodes_per_s': completed / wall_s,
        'output_tokens_per_s': sum(row['output_tokens'] for row in success) / wall_s,
        'cache_metric_coverage': len(known) / len(success) if success else 0,
        'cache_hit_fraction': hit_fraction(success),
        'continuation_cache_hit_fraction': hit_fraction(continuation),
        'worker_switches': switches, 'worker_transitions': transitions,
        'worker_switch_fraction': switches / transitions if transitions else None,
        'episode_latency_p50_s': percentile(episode_times, 0.5),
        'episode_latency_p95_s': percentile(episode_times, 0.95),
        'per_instance': by_instance, 'router_after': status,
    }
    for name in ['latency_s', 'engine_ttft_s', 'engine_prefill_s', 'engine_queue_s', 'first_output_s']:
        values = [row[name] for row in success if row.get(name) is not None]
        result[name] = {'count': len(values), 'p50': percentile(values, 0.5), 'p95': percentile(values, 0.95)}
    return result


def run_trial(sampler, workload, group, repeat, concurrency, output):
    affinity, cache = GROUPS[group]
    previous_status = sampler.status()
    started = time.perf_counter()
    rows = []

    def run_episode(episode):
        episode_rows = []
        episode_id = episode['episode_id']
        session_ids = [f'{group}:{repeat}:{concurrency}:{episode_id}'] if affinity else None
        try:
            for turn, tokens in enumerate(episode['prompts']):
                began = time.perf_counter()
                row = {'group': group, 'repeat': repeat, 'concurrency': concurrency,
                       'session_affinity': affinity, 'prefix_cache': cache, 'episode_id': episode_id,
                       'turn': turn, 'input_hash': prompt_key(tokens), 'started_s': began - started}
                try:
                    result = sampler.sample([{'input_ids': tokens}], session_ids=session_ids)[0]
                    row.update(result)
                    row['episode_finished'] = turn == len(episode['prompts']) - 1
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                finished = time.perf_counter()
                row.update(finished_s=finished - started, latency_s=finished - began)
                episode_rows.append(row)
                if row.get('error'):
                    break
        finally:
            if session_ids is not None:
                sampler.release_sessions(session_ids)
        return episode_rows

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_episode, episode) for episode in workload]
        for future in as_completed(futures):
            episode_rows = future.result()
            rows.extend(episode_rows)
            for row in episode_rows:
                output.write(json.dumps(row, allow_nan=False) + '\n')
    wall_s = time.perf_counter() - started
    output.flush()
    result = summarize(rows, wall_s, sampler.status())
    result['capacity_fallbacks_delta'] = (result['router_after']['capacity_fallbacks']
                                         - previous_status['capacity_fallbacks'])
    return {'group': group, 'repeat': repeat, 'concurrency': concurrency,
            'active_episode_limit': min(concurrency, len(workload)),
            'session_affinity': affinity, 'prefix_cache': cache, **result}


def comparisons(trials):
    """Paired differences, retaining each repeat rather than hiding variability."""
    result = []
    for concurrency in sorted({trial['concurrency'] for trial in trials}):
        matched = {(trial['group'], trial['repeat']): trial for trial in trials
                   if trial['concurrency'] == concurrency}
        pairs = []
        for repeat in sorted({trial['repeat'] for trial in trials}):
            baseline, sticky = matched.get(('B', repeat)), matched.get(('D', repeat))
            if (baseline is None or sticky is None or baseline['errors'] or sticky['errors']
                    or not baseline['completed_episodes'] or not sticky['completed_episodes']):
                continue
            pairs.append({'repeat': repeat,
                          'episodes_per_s_ratio': sticky['episodes_per_s'] / baseline['episodes_per_s'],
                          'episode_p95_ratio': sticky['episode_latency_p95_s'] / baseline['episode_latency_p95_s']})
        if pairs:
            result.append({'concurrency': concurrency, 'pairs': pairs,
                           'median_episodes_per_s_ratio': statistics.median(p['episodes_per_s_ratio'] for p in pairs),
                           'median_episode_p95_ratio': statistics.median(p['episode_p95_ratio'] for p in pairs)})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=['vllm', 'mock'], default='vllm')
    parser.add_argument('--device', choices=['cuda', 'npu'], default='cuda',
                        help='Accelerator for vllm; mock always runs on CPU.')
    parser.add_argument('--model', help='Cached model path or a Twinkle model ID; required for vllm.')
    parser.add_argument('--replicas', type=int, default=2)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.7,
                        help='vLLM device memory fraction; applies to both GPU and NPU.')
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--turns', type=int, default=6)
    parser.add_argument('--prompt-tokens', type=int, default=2048)
    parser.add_argument('--append-tokens', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--concurrency', nargs='+', type=int, default=[8])
    parser.add_argument('--groups', nargs='+', choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-sessions', type=int, default=10000)
    parser.add_argument('--warmup-requests', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--ray-address')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory; existing results are not overwritten.')
    args = parser.parse_args(argv)
    for name in ['tensor_parallel_size', 'max_model_len', 'episodes', 'turns', 'prompt_tokens',
                 'append_tokens', 'max_tokens', 'repeats', 'max_sessions']:
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.replicas < 2 or any(n < 1 for n in args.concurrency):
        parser.error('Use at least two replicas and positive concurrency values')
    if args.warmup_requests < 0 or args.timeout <= 0 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error('Invalid warmup count, timeout, or GPU memory utilization')
    if args.prompt_tokens + (args.turns - 1) * args.append_tokens + args.max_tokens > args.max_model_len:
        parser.error('Replay input plus output budget exceeds --max-model-len')
    if args.backend == 'vllm' and not args.model:
        parser.error('--model is required for the vllm backend')
    if len(set(args.groups)) != len(args.groups) or len(set(args.concurrency)) != len(args.concurrency):
        parser.error('Duplicate groups or concurrency values would make comparisons ambiguous')
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import ray

    if ray.is_initialized():
        raise RuntimeError('Run this benchmark in a dedicated Python process')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['output_dir'] = str(args.output_dir)
    packages = {}
    for name in ['ray', 'vllm', 'vllm-ascend', 'torch', 'torch-npu', 'twinkle']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {'config': config, 'packages': packages, 'workload_kind': 'frozen_append_only_token_replay',
                'synthetic': args.backend == 'mock', 'trials': [], 'replica_pools': [],
                'started_unix_s': time.time()}
    if args.backend == 'mock':
        manifest['notice'] = 'Synthetic cache/timing data: validates the harness, not accelerator performance.'
    (args.output_dir / 'config.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        ray.init(address=args.ray_address, namespace=f'sticky-benchmark-{time.time_ns()}',
                 include_dashboard=False, log_to_driver=False,
                 **(local_ray_options(config) if not args.ray_address else {}))
        resources = ray.available_resources()
        resource = 'NPU' if args.device == 'npu' else 'GPU'
        needed_devices = args.replicas * args.tensor_parallel_size if args.backend == 'vllm' else 0
        if resources.get(resource, 0) < needed_devices or resources.get('CPU', 0) < args.replicas:
            raise RuntimeError(f'Need {args.replicas} CPUs and {needed_devices} {resource}s; available: {resources}')
        workload = None
        with (args.output_dir / 'requests.jsonl').open('w') as output:
            for cache in sorted({GROUPS[group][1] for group in args.groups}):
                workers = []
                sampler = None
                try:
                    worker_cls = ray.remote(BenchmarkWorker).options(**worker_options(config))
                    workers = [worker_cls.remote(config, cache) for _ in range(args.replicas)]
                    infos = ray.get([worker.info.remote() for worker in workers], timeout=args.timeout)
                    manifest['replica_pools'].append(infos)
                    if workload is None:
                        workload = ray.get(workers[0].workload.remote(), timeout=args.timeout)
                        with (args.output_dir / 'workload.jsonl').open('w') as dataset:
                            for episode in workload:
                                dataset.write(json.dumps(episode) + '\n')
                        manifest['workload_hash'] = hashlib.sha256(json.dumps(workload).encode()).hexdigest()
                    for _ in range(args.warmup_requests):
                        ray.get([worker._twinkle_async_sample.remote(
                            [{'input_ids': workload[0]['prompts'][-1]}]) for worker in workers], timeout=args.timeout)
                    sampler = RoutedSampler(workers, args.max_sessions, args.timeout)
                    groups = [group for group in args.groups if GROUPS[group][1] == cache]
                    for concurrency in args.concurrency:
                        for repeat in range(args.repeats):
                            order = groups.copy()
                            random.Random(args.seed + repeat).shuffle(order)
                            for group in order:
                                ray.get([worker.reset.remote() for worker in workers], timeout=args.timeout)
                                logger.info('%s: group=%s concurrency=%s repeat=%s',
                                            args.backend, group, concurrency, repeat)
                                trial = run_trial(sampler, workload, group, repeat, concurrency, output)
                                manifest['trials'].append(trial)
                                status = trial['router_after']
                                if trial['errors'] or status['active_leases'] or status['session_count']:
                                    raise RuntimeError('Trial failed or leaked routing state; inspect requests.jsonl')
                                if (GROUPS[group][0] and args.max_sessions >= min(concurrency, args.episodes)
                                        and trial['worker_switches']):
                                    raise RuntimeError('Sticky routing changed replicas while bindings remain valid')
                finally:
                    try:
                        if workers:
                            ray.get([worker.close.remote() for worker in workers], timeout=args.timeout)
                    except Exception as exc:
                        manifest.setdefault('cleanup_errors', []).append(f'{type(exc).__name__}: {exc}')
                    finally:
                        for worker in workers:
                            ray.kill(worker)
                        if sampler is not None:
                            ray.kill(sampler._request_router)
    except Exception as exc:
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['comparisons_D_over_B'] = comparisons(manifest['trials'])
        (args.output_dir / 'summary.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        ray.shutdown()
    logger.info('Results: %s', args.output_dir / 'summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
