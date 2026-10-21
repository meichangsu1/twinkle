# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import asyncio
import json
import pickle
import threading
from io import StringIO
from types import SimpleNamespace

import pytest

from scripts.benchmark_grpo_sticky_session import (
    check_routing_idle, cw_status, parse_args, prompt_key, summarize_trial, train_batches,
)
from scripts.synthetic_grpo_workload import (
    answer_score, build_input, plan_for, prepare_inputs, run_episode, summarize_requests,
)
from tests.twinkle_agentic.test_multi_turn_rollout import FakeTemplate, FakeTokenizer
from tests.infra.test_grpo_sticky_session_benchmark import planned_worker_ids
from twinkle.data_format import SampleResponse, SampledSequence, SamplingParams


def config(**overrides):
    return {**dict(task='synthetic', model='no-download', group='D', seed=42, data_seed=2026,
                steps=2, warmup_steps=1, batch_size=2, num_generations=2, mini_batch_size=2,
                micro_batch_size=1, prompt_tokens=512, max_prompt_tokens=512, max_tokens=8,
                max_model_len=4096, turns=3, feedback_delay_ms=0, episode_concurrency=2,
                feedback_tokens=64, input_plan=None, plan_hash=None), **overrides}


@pytest.fixture
def template():
    return FakeTemplate(FakeTokenizer())


def test_plan_is_reproducible_varies_by_case_and_preserves_all_queries(template):
    options = config(feedback_delay_ms=123)
    first = build_input(template, options, 0)
    assert first == build_input(template, options, 0)
    assert first != build_input(template, options, 1)
    assert 496 <= len(first['input_ids']) <= 512
    plan = plan_for(first, options)
    assert all(0 <= wait <= 0.123 for waits in plan['waits_s'] for wait in waits)
    assert plan['waits_s'][0] != plan['waits_s'][1]
    assert plan['observations'][0] != plan['observations'][1]
    changed = copy.deepcopy(first)
    plan['feedback'].pop()
    changed['user_data'][1] = ('synthetic_plan', json.dumps(plan))
    with pytest.raises(ValueError, match='configuration'):
        plan_for(changed, options)


def test_preparation_roundtrips_frozen_plan_without_loading_a_dataset(template, tmp_path, monkeypatch):
    import twinkle.template
    monkeypatch.setattr(twinkle.template, 'Template', lambda *args, **kwargs: template)
    first, second = tmp_path / 'first.jsonl', tmp_path / 'second.jsonl'
    prepare_inputs({**config(), 'plan_file': str(first)})
    prepare_inputs({**config(), 'input_plan': str(first), 'plan_file': str(second)})
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_text().splitlines()) == 4


def test_preparation_with_actual_hf_tokenizer_and_template_serializes_tensor_fields(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, PreTrainedTokenizerFast

    directory = tmp_path / 'local-tokenizer'
    directory.mkdir()
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ['Read the table. A B C D k00001=A\nCase 00001 Next query'] * 4,
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=['[UNK]', '<|im_start|>', '<|im_end|>']))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', eos_token='<|im_end|>')
    fast.chat_template = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] "
                          "+ '<|im_end|>\\n' }}{% endfor %}"
                          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    fast.save_pretrained(directory)
    GPT2Config(vocab_size=len(fast), n_embd=16, n_layer=1, n_head=2).save_pretrained(directory)
    path = tmp_path / 'inputs.jsonl'
    options = config(model=str(directory), steps=1, batch_size=1, plan_file=str(path))
    prepare_inputs(options)
    feature = json.loads(path.read_text())
    assert isinstance(feature['token_type_ids'], list)
    assert 496 <= len(feature['input_ids']) <= 512
    plan_for(feature, options)


@pytest.mark.parametrize('text,target,expected', [
    ('A', 'A', (1.1, 1)), ('Answer: b.', 'B', (1.1, 1)),
    ('D', 'A', (0.1, 0)), ('The answer is A because...', 'A', (0, 0)), ('', 'A', (0, 0)),
])
def test_rewards_are_based_on_actual_answer(text, target, expected):
    assert answer_score(text, target) == expected


class Sampler:
    """Synthetic generation with production CW bindings; no accelerator timing claims."""

    def __init__(self, template, options, failure=None):
        self.template, self.options, self.failure = template, options, failure
        self.rows, self.calls = [], []
        self.lock = threading.Lock()
        self.local = threading.local()
        self.episodes = 0

    def reset_prefix_cache(self):
        check_routing_idle(cw_status(self))
        assert cw_status(self)['session_count'] == 0

    def sample(self, inputs, sampling_params=None, _session_ids=None):
        feature = inputs[0]
        turn = sum(message['role'] == 'assistant' for message in feature['messages']) + 1
        with self.lock:
            if turn == 1:
                self.local.correct = self.episodes % 2
                self.episodes += 1
            sid = _session_ids[0] if _session_ids else None
            worker = planned_worker_ids(self, _session_ids, len(inputs))[0]
            self.calls.append((threading.get_ident(), turn, sid))
        if self.failure == 'sample' and turn == 2:
            raise RuntimeError('sample failure')
        target = json.loads(dict(feature['user_data'])['synthetic_plan'])['answers'][turn - 1]
        text = target if self.local.correct else 'ABCD'[('ABCD'.index(target) + 1) % 4]
        # A real length-limited turn need not contain an end-of-turn token.
        tokens = self.template.tokenizer.encode(text + ('<|im_end|>' if turn != 2 else ''))
        result = SampleResponse([SampledSequence(
            stop_reason='length' if turn == 2 else 'stop', tokens=tokens,
            logprobs=[[(token, -0.5)] for token in tokens],
            new_input_feature=self.template.concat_input_feature(feature, tokens))])
        with self.lock:
            request_id = f'request-{len(self.rows)}'
            self.rows.append({'engine_request_id': request_id, 'input_hash': prompt_key(feature['input_ids']),
                              'input_tokens': len(feature['input_ids']), 'output_tokens': len(tokens),
                              'instance_id': worker,
                              'cached_tokens': (0 if self.options['group'] == 'A'
                                                else len(feature['input_ids']) // 2)})
        result._benchmark_request_id = request_id
        return [result]

    def take_observations(self):
        result, self.rows = self.rows, []
        return result


@pytest.mark.parametrize('group', ['A', 'B', 'D'])
@pytest.mark.parametrize('failure', [None, 'sample', 'train'])
def test_real_backward_after_multiturn_rollout_and_cleanup(template, monkeypatch, group, failure):
    import torch
    import twinkle.template
    from twinkle.advantage import GRPOAdvantage

    options = {**config(), 'group': group}
    features = [build_input(template, options, i) for i in range(4)]
    frozen = copy.deepcopy(features)
    sampler = Sampler(template, options, failure)
    monkeypatch.setattr(twinkle.template, 'Template', lambda *args, **kwargs: template)
    syncs = []

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.optimizer = torch.optim.SGD([self.weight], lr=0.1)
            self.updates = 0

        def forward_backward(self, inputs, old_logps, advantages, micro_batch_size):
            check_routing_idle(cw_status(sampler))
            if failure == 'train':
                raise RuntimeError('train failure')
            signs = []
            for feature, logps in zip(inputs, old_logps, strict=True):
                assert sum(label != -100 for label in feature['labels']) == len(logps) == 5
                assert sum(feature['completion_mask']) == 5
                assert len(feature['input_ids']) == len(feature['labels'])
                target = json.loads(dict(feature['user_data'])['ground_truth'])[-1]
                signs.append(1 if feature['messages'][-1]['content'] == target else -1)
            loss = -(torch.nn.functional.logsigmoid(self.weight * torch.tensor(signs))
                     * torch.tensor(advantages)).mean()
            loss.backward()

        def clip_grad_and_step(self):
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1

        def synchronize(self):
            pass

        def calculate_metric(self, **kwargs):
            return {'weight': self.weight.item()}

    model = Model()
    steps, requests = StringIO(), StringIO()
    call = lambda: train_batches(options, features, model, sampler,
                                 SimpleNamespace(sync_weights=lambda **kwargs: syncs.append(True)),
                                 SamplingParams(max_tokens=8), None, GRPOAdvantage(), steps, requests,
                                 lambda: cw_status(sampler))
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            call()
    else:
        records = call()
        assert model.weight.item() > 0 and model.updates == 4
        assert len(syncs) == 2
        assert all(row['requests'] == 12 and row['turns_mean'] == 3 for row in records)
        assert all(row['nonzero_advantage_fraction'] == 1 for row in records)
        summary = summarize_trial(options, records)
        assert summary['requests'] == 12 and summary['trajectories'] == 4
        assert summary['initial_prompt_tokens_mean'] == 512
        assert summary['length_stop_fraction'] == pytest.approx(1 / 3)
        if group == 'D':
            assert summary['mean_replicas_per_episode'] == 1 and summary['replica_switch_rate'] == 0
            # Distinct episode IDs, stable across all of an episode's turns.
            ids = [sid for _, turn, sid in sampler.calls if turn == 1]
            assert len(set(ids)) == 8
            assert all(sum(sid == candidate for _, _, sid in sampler.calls) == 3 for candidate in ids)
        else:
            assert all(sid is None for _, _, sid in sampler.calls)
        assert len(requests.getvalue().splitlines()) == 24
    check_routing_idle(cw_status(sampler))
    assert cw_status(sampler)['session_count'] <= (4 if group == 'D' else 0)
    assert features == frozen


def test_observation_correlation_rejects_missing_requests_and_preserves_missing_cache(template):
    options = config()
    feature = build_input(template, options, 0)
    sampler = Sampler(template, options)
    episodes = [run_episode(0, feature, options, sampler, template, SamplingParams(max_tokens=8))]
    rows = sampler.take_observations()
    rows[1]['cached_tokens'] = None
    _, stats = summarize_requests(rows, episodes, options)
    assert stats['cached_tokens'] is stats['cache_hit_fraction'] is stats['continuation_cached_tokens'] is None
    assert stats['cache_metric_coverage'] == pytest.approx(2 / 3)
    with pytest.raises(RuntimeError, match='match'):
        summarize_requests(rows[:-1], episodes, options)
    rows[0]['input_hash'] = 'wrong'
    with pytest.raises(RuntimeError, match='accounting'):
        summarize_requests(rows, episodes, options)


def test_synthetic_cli_does_not_require_a_dataset():
    args = parse_args(['--task', 'synthetic', '--model', 'local', '--output-dir', 'unused', '--max-tokens', '32'])
    assert args.dataset_id is None and args.turns == 4
    for flags in (['--turns', '1'], ['--prompt-tokens', '8192'], ['--episode-concurrency', '0'],
                  ['--feedback-delay-ms', 'nan'], ['--max-model-len', '4096'], ['--dataset-id', '/data/gsm8k']):
        with pytest.raises(SystemExit):
            parse_args(['--task', 'synthetic', '--model', 'local', '--output-dir', 'unused', *flags])


def test_benchmark_request_correlation_survives_serialization(monkeypatch):
    from scripts.benchmark_grpo_sticky_session import observed_classes
    from twinkle.sampler import vLLMSampler
    seen = []

    async def sample_single(self, *args, request_id=None, **kwargs):
        seen.append(request_id)
        return SampleResponse([])

    monkeypatch.setattr(vLLMSampler, '_sample_single', sample_single)
    _, observed = observed_classes()
    sampler = object.__new__(observed)
    first = asyncio.run(sampler._sample_single({}, request_id=None))
    second = asyncio.run(sampler._sample_single({}))
    assert seen[0] != seen[1] and all(isinstance(value, str) and value for value in seen)
    assert pickle.loads(pickle.dumps(first))._benchmark_request_id == seen[0]
    assert second._benchmark_request_id == seen[1]
