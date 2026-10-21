# Copyright (c) ModelScope Contributors. All rights reserved.
"""Synthetic multi-turn workload for benchmark_grpo_sticky_session.py.

Only the task and feedback schedule are synthetic. Sampling, routing, token
accounting, GRPO advantages, backward, optimizer steps and weight sync are real.
The fixed-turn driver deliberately continues after a per-turn length stop;
native stop reasons are recorded unchanged. No external environment is needed.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)
SYSTEM = ('Read the lookup table and answer each query using the original table. '
          'Reply with exactly one letter: A, B, C or D. Do not explain or call tools.')


def plan_for(feature, config):
    plan = json.loads(dict(feature['user_data'])['synthetic_plan'])
    turns, generations = config['turns'], config['num_generations']
    if (len(plan['answers']) != turns or any(answer not in 'ABCD' or len(answer) != 1
                                           for answer in plan['answers'])
            or len(plan['feedback']) != turns - 1 or any(not text for text in plan['feedback'])
            or len(plan['waits_s']) != generations
            or any(len(waits) != turns - 1 for waits in plan['waits_s'])
            or len(plan['observations']) != generations
            or any(len(observations) != turns - 1 for observations in plan['observations'])
            or any(not math.isfinite(wait) or wait < 0 for waits in plan['waits_s'] for wait in waits)):
        raise ValueError('Synthetic input plan does not match the turn/generation configuration')
    if not config['prompt_tokens'] - 16 <= len(feature['input_ids']) <= config['prompt_tokens']:
        raise ValueError('Synthetic input plan prompt length does not match --prompt-tokens')
    return plan


def build_input(template, config, index):
    """Fit a text lookup task to the tokenizer, retaining the query and its answer."""
    from twinkle_agentic.utils.token_utils import _to_plain

    rng = random.Random(f'{config["data_seed"]}:{index}')
    target = config['prompt_tokens']
    records = [(f'k{i:05d}', rng.choice('ABCD')) for i in range(max(32, target // 2))]
    # These early records fit even at the minimum prompt budget. The rest are
    # deterministic distractors; an answer is always present in the actual text.
    queries = [rng.randrange(min(len(records), max(4, target // 64))) for _ in range(config['turns'])]
    body_ids = template.tokenizer.encode(''.join(f'{key}={value}\n' for key, value in records),
                                         add_special_tokens=False)

    def encode(length):
        body = template.tokenizer.decode(body_ids[:length], skip_special_tokens=True)
        key = records[queries[0]][0]
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': f'Case {index:08d}\nLookup table:\n{body}\n'
                                                f'Query: {key}. Reply with one letter only.'}]
        feature = _to_plain(template.encode({'messages': messages}, add_generation_prompt=True))
        return feature, body

    low, high = 0, len(body_ids)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate, body = encode(middle)
        if len(candidate['input_ids']) <= target:
            best = candidate, body
            low = middle + 1
        else:
            high = middle - 1
    if best is None or len(best[0]['input_ids']) < target - 16:
        raise ValueError('Could not fit the synthetic prompt within 16 tokens of its target')
    feature, body = best
    if any(f'{records[q][0]}={records[q][1]}' not in body for q in queries):
        raise ValueError('Prompt budget is too small to include every queried record')
    answers = [records[q][1] for q in queries]
    plan = {
        'answers': answers,
        'feedback': [f'Local simulator checkpoint {turn + 1}. Next query: {records[q][0]}. '
                     'Use the original table. Reply with one letter only.'
                     for turn, q in enumerate(queries[1:])],
        'waits_s': [[rng.uniform(0, config['feedback_delay_ms'] / 1000)
                     for _ in range(config['turns'] - 1)] for _ in range(config['num_generations'])],
    }
    # Candidate-specific observations create growing private histories while the
    # GRPO group still starts from the same task. Otherwise a long shared prompt
    # may already be cached on both replicas and hide the cross-turn effect.
    plan['observations'] = []
    for candidate in range(config['num_generations']):
        observations = []
        for turn in range(config['turns'] - 1):
            noise = ' '.join(f'{rng.getrandbits(32):08x}' for _ in range(config['feedback_tokens']))
            noise_ids = template.tokenizer.encode(noise, add_special_tokens=False)[:config['feedback_tokens']]
            text = template.tokenizer.decode(noise_ids, skip_special_tokens=True)
            observations.append(f'Observation {index}/{candidate}/{turn} (ignore for lookup): {text}')
        plan['observations'].append(observations)
    feature['user_data'] = [('ground_truth', json.dumps(answers)), ('synthetic_plan', json.dumps(plan))]
    return feature


def prepare_inputs(config):
    from scripts.benchmark_grpo_sticky_session import read_inputs

    if config['input_plan']:
        inputs = read_inputs(config['input_plan'], config)
    else:
        from twinkle.template import Template
        template = Template(config['model'], max_length=None, enable_thinking=False)
        inputs = [build_input(template, config, i) for i in range(config['steps'] * config['batch_size'])]
    with Path(config['plan_file']).open('x') as stream:
        for feature in inputs:
            plan_for(feature, config)
            stream.write(json.dumps(feature, sort_keys=True, allow_nan=False,
                                    default=lambda value: value.tolist()) + '\n')
    logger.info('Frozen %s synthetic prompts; token lengths min=%s max=%s', len(inputs),
                min(len(row['input_ids']) for row in inputs), max(len(row['input_ids']) for row in inputs))


def answer_score(text, target):
    match = re.fullmatch(r'\s*(?:Answer:\s*)?([ABCD])\s*[.!]?\s*', text, flags=re.IGNORECASE)
    valid = match is not None
    correct = bool(valid and match.group(1).upper() == target)
    return float(correct) + 0.1 * valid, float(correct)


def run_episode(index, feature, config, sampler, template, sampling_params):
    from scripts.benchmark_grpo_sticky_session import prompt_key
    from twinkle_agentic.rollout.ledger import TurnLedger

    plan = plan_for(feature, config)
    ledger = TurnLedger(template, label=f'synthetic episode {index}')
    # Use the frozen IDs directly; never re-encode a sampled history.
    ledger.adopt(copy.deepcopy(feature))
    session = uuid.uuid4().hex if config['group'] == 'D' else None
    requests = []
    waits = plan['waits_s'][index % config['num_generations']]
    observations = plan['observations'][index % config['num_generations']]
    for turn in range(config['turns']):
        pif = ledger.input_feature
        if len(pif['input_ids']) + config['max_tokens'] > config['max_model_len']:
            raise ValueError('Synthetic episode exceeds context budget; increase --max-model-len')
        input_hash = prompt_key(pif['input_ids'])
        responses = sampler.sample([pif], sampling_params=sampling_params,
                                   _session_ids=[session] if session else None)
        if len(responses) != 1 or len(responses[0].sequences) != 1:
            raise RuntimeError('Expected one sampled sequence for each synthetic turn')
        response = responses[0]
        sequence = response.sequences[0]
        if (not sequence.tokens or not sequence.logprobs or len(sequence.tokens) != len(sequence.logprobs)
                or sequence.stop_reason not in ('stop', 'length')):
            raise RuntimeError('Synthetic turn returned empty, failed or misaligned generation')
        ledger.record(sequence)
        requests.append({
            'episode': index, 'turn': turn + 1, 'engine_request_id': response._benchmark_request_id,
            'session_id': session,
            'input_hash': input_hash, 'sampled_tokens': len(sequence.tokens),
            'decoded': template.tokenizer.decode(sequence.tokens, skip_special_tokens=True),
            'stop_reason': sequence.stop_reason, 'target': plan['answers'][turn],
            'feedback_wait_s': waits[turn - 1] if turn else 0.0,
        })
        if turn + 1 < config['turns']:
            time.sleep(waits[turn])
            feedback = observations[turn] + '\n' + plan['feedback'][turn]
            if not ledger.observe([{'role': 'user', 'content': feedback}]):
                raise ValueError('Synthetic feedback exceeds context budget; increase --max-model-len')
    result = ledger.merge(feature, turns=ledger.turns)
    return result, requests


def summarize_requests(observations, episodes, config):
    """Join native requests by ID, including identical copies and continuations."""
    expected = {row['engine_request_id']: row for _, requests in episodes for row in requests}
    actual_ids = [row['engine_request_id'] for row in observations]
    if len(expected) != len(episodes) * config['turns'] or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError('Duplicate or missing synthetic native request IDs')
    if set(actual_ids) != set(expected):
        raise RuntimeError('Engine observations do not match synthetic episode requests')
    rows, instances = [], {}
    for observation in observations:
        meta = expected[observation['engine_request_id']]
        if observation['input_hash'] != meta['input_hash'] or observation['output_tokens'] != meta['sampled_tokens']:
            raise RuntimeError('Engine and episode token accounting disagree')
        row = {**observation, **meta}
        rows.append(row)
        instance = instances.setdefault(row['instance_id'], {'requests': 0, 'input_tokens': 0, 'output_tokens': 0})
        for name in instance:
            instance[name] += 1 if name == 'requests' else row[name]
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row['episode'], []).append(row)
    groups = {}
    switches, transitions, episode_counts = 0, 0, []
    for index, requests in sorted(by_episode.items()):
        requests.sort(key=lambda row: row['turn'])
        targets = [row['instance_id'] for row in requests]
        episode_counts.append(len(set(targets)))
        switches += sum(a != b for a, b in zip(targets, targets[1:]))
        transitions += len(targets) - 1
        groups.setdefault(index // config['num_generations'], set()).update(targets)
    if config['group'] == 'D' and any(count != 1 for count in episode_counts):
        raise RuntimeError('An affinity-enabled episode switched sampler instances')

    def counts(selected):
        known = [row for row in selected if row['cached_tokens'] is not None]
        return (sum(row['input_tokens'] for row in selected),
                sum(row['cached_tokens'] for row in known) if len(known) == len(selected) else None)

    total, cached = counts(rows)
    continuation_input, continuation_cached = counts([row for row in rows if row['turn'] > 1])
    return rows, {
        'input_tokens': total, 'output_tokens': sum(row['output_tokens'] for row in rows),
        'cached_tokens': cached, 'cache_hit_fraction': cached / total if cached is not None else None,
        'cache_metric_coverage': sum(row['cached_tokens'] is not None for row in rows) / len(rows),
        'per_instance': instances, 'group_replica_counts': [len(targets) for targets in groups.values()],
        'episode_replica_counts': episode_counts, 'replica_switches': switches, 'replica_transitions': transitions,
        'continuation_input_tokens': continuation_input, 'continuation_cached_tokens': continuation_cached,
        'initial_input_tokens': sum(row['input_tokens'] for row in rows if row['turn'] == 1),
        'requests': len(rows), 'turns_mean': config['turns'],
        'feedback_wait_s_sum': sum(row['feedback_wait_s'] for row in rows),
    }


def train_batches(config, inputs, model, sampler, checkpoint, sampling_params, advantage_fn,
                  steps_stream, requests_stream, router_status):
    from scripts.benchmark_grpo_sticky_session import (
        STAGES, check_routing_idle, prompt_key, reset_sampling_batch,
    )
    from twinkle.template import Template

    template = Template(config['model'], max_length=config['max_model_len'],
                        truncation_strategy='raise', enable_thinking=False)
    records = []
    for step in range(config['steps']):
        begin = time.perf_counter()
        row = {'step': step, 'measured': step >= config['warmup_steps']}

        def timed(name, action):
            start = time.perf_counter()
            result = action()
            row[name] = time.perf_counter() - start
            return result

        timed('sync_s', lambda: checkpoint.sync_weights(merge_and_sync=False))
        timed('cache_reset_s', lambda: reset_sampling_batch(sampler))
        offset = step * config['batch_size']
        prompts = inputs[offset:offset + config['batch_size']]
        expanded = [prompt for prompt in prompts for _ in range(config['num_generations'])]

        def sample():
            # Each thread drives an entire episode. There is no turn-wide barrier:
            # one episode may generate while another is waiting on local feedback.
            with ThreadPoolExecutor(max_workers=config['episode_concurrency']) as pool:
                return list(pool.map(lambda pair: run_episode(*pair, config, sampler, template, sampling_params),
                                     enumerate(expanded)))

        episodes = timed('sample_s', sample)
        observations = sampler.take_observations()
        requests, stats = summarize_requests(observations, episodes, config)
        for request in requests:
            requests_stream.write(json.dumps({'step': step, **request}, allow_nan=False) + '\n')
        requests_stream.flush()
        row.update(stats)
        status = router_status()
        row['router_after'] = status
        check_routing_idle(status)

        def score():
            features, logps, rewards, accuracies = [], [], [], []
            for feature, turns in episodes:
                features.append(feature)
                logps.append([values[0][1] for values in feature['logprobs']])
                scores = [answer_score(turn['decoded'], turn['target']) for turn in turns]
                rewards.append(statistics.mean(value[0] for value in scores))
                accuracies.append(statistics.mean(value[1] for value in scores))
            advantages = advantage_fn(rewards, num_generations=config['num_generations'], scale='group').tolist()
            row.update(reward_mean=statistics.mean(rewards), accuracy=statistics.mean(accuracies),
                       completion_tokens_mean=statistics.mean(len(logp) for logp in logps),
                       length_stop_fraction=sum(request['stop_reason'] == 'length'
                                                for request in requests) / len(requests),
                       nonzero_advantage_fraction=sum(abs(value) > 1e-8 for value in advantages) / len(advantages))
            if sum(len(logp) for logp in logps) != row['output_tokens']:
                raise RuntimeError('Multi-turn trainable tokens and native outputs disagree')
            if not row['nonzero_advantage_fraction']:
                logger.warning('Batch %s has zero GRPO advantages; all group rewards match', step)
            return features, logps, advantages

        features, old_logps, advantages = timed('reward_advantage_s', score)

        def update():
            count = 0
            for start in range(0, len(features), config['mini_batch_size']):
                end = start + config['mini_batch_size']
                model.forward_backward(inputs=features[start:end], old_logps=old_logps[start:end],
                                       advantages=advantages[start:end], micro_batch_size=config['micro_batch_size'])
                model.clip_grad_and_step()
                count += 1
            model.synchronize()
            row['model_metrics'] = model.calculate_metric(is_training=True)
            return count

        row['optimizer_updates'] = timed('train_s', update)
        row['trajectories'] = len(episodes)
        row['prompt_hashes'] = [prompt_key(prompt['input_ids']) for prompt in prompts]
        row['total_s'] = time.perf_counter() - begin
        row['overhead_s'] = row['total_s'] - sum(row[name] for name in STAGES)
        steps_stream.write(json.dumps(row, allow_nan=False) + '\n')
        steps_stream.flush()
        records.append(row)
        logger.info('%s seed=%s batch=%s/%s total=%.3fs rollout=%.3fs train=%.3fs cache=%s switches=%s/%s',
                    config['group'], config['seed'], step + 1, config['steps'], row['total_s'], row['sample_s'],
                    row['train_s'], row['cache_hit_fraction'], row['replica_switches'], row['replica_transitions'])
    return records


def summarize_multiturn(records):
    continuations = sum(row['continuation_input_tokens'] for row in records)
    cached = [row['continuation_cached_tokens'] for row in records]
    transitions = sum(row['replica_transitions'] for row in records)
    return {
        'requests': sum(row['requests'] for row in records),
        'turns_mean': statistics.mean(row['turns_mean'] for row in records),
        'mean_replicas_per_episode': statistics.mean(
            count for row in records for count in row['episode_replica_counts']),
        'replica_switch_rate': sum(row['replica_switches'] for row in records) / transitions,
        'continuation_cache_hit_fraction': sum(cached) / continuations
        if all(value is not None for value in cached) else None,
        'feedback_wait_s_sum': sum(row['feedback_wait_s_sum'] for row in records),
        'initial_prompt_tokens_mean': statistics.mean(
            row['initial_input_tokens'] / len(row['episode_replica_counts']) for row in records),
    }
