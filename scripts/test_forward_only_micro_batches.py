"""Compare real forward_only results through the public model/client API.

Server (must run this branch with the Transformers backend):
    python scripts/test_forward_only_micro_batches.py --model-id Qwen/Qwen3-0.6B
Local model directory, without a server:
    python scripts/test_forward_only_micro_batches.py --local --model-id /path/to/model --check-logits

The server test creates a temporary adapter and removes it when finished. No
backward or optimizer step is performed. Use --padding-free only with a model
and attention backend that support packed sequences. aux_loss is not compared
against the unsplit result: its contract is weighted micro-batch accumulation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
import time
import uuid

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--base-url', default='http://localhost:9000')
    parser.add_argument('--api-key', default=os.environ.get('TWINKLE_API_KEY', 'EMPTY_API_KEY'))
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--samples', type=int, default=8,
                        help='Use enough samples per DP rank to execute multiple micro-batches.')
    parser.add_argument('--micro-batch-size', type=int, default=2)
    parser.add_argument('--padding-free', action='store_true')
    parser.add_argument('--check-logits', action='store_true', help='Also return full logits; increases memory/HTTP payload.')
    parser.add_argument('--rtol', type=float, default=1e-4, help='For BF16/FP16, consider 0.01.')
    parser.add_argument('--atol', type=float, default=1e-5, help='For BF16/FP16, consider 0.01.')
    args = parser.parse_args()
    if args.samples < 3 or not 0 < args.micro_batch_size < args.samples:
        parser.error('require samples >= 3 and 0 < micro-batch-size < samples')
    return args


def build_inputs(count):
    # Deliberately unsorted lengths and distinct token patterns expose reordering.
    lengths = [7, 13, 5, 11, 9, 6, 12, 8]
    inputs = []
    for index in range(count):
        length = lengths[index % len(lengths)]
        tokens = [1 + (index + offset) % 6 for offset in range(length)]
        inputs.append({'input_ids': tokens, 'labels': [-100, *tokens[1:]],
                       'position_ids': list(range(length))})
    return inputs


def assert_same_outputs(expected, actual, lengths, *, check_logits, rtol, atol):
    for field in ('logps', 'logits') if check_logits else ('logps',):
        if expected.get(field) is None or actual.get(field) is None:
            raise AssertionError(f'missing required output: {field}')
        reference = torch.as_tensor(expected[field]).detach().cpu().float()
        result = torch.as_tensor(actual[field]).detach().cpu().float()
        if reference.ndim != result.ndim or result.shape[0] != len(lengths) or reference.shape[0] != len(lengths):
            raise AssertionError(f'{field}: invalid batch shape {reference.shape} vs {result.shape}')
        if reference.shape[2:] != result.shape[2:]:
            raise AssertionError(f'{field}: feature dimensions differ')
        max_error = 0.0
        for index, length in enumerate(lengths):
            if reference.shape[1] < length or result.shape[1] < length:
                raise AssertionError(f'{field}: missing tokens for sample {index}')
            expected_sample = reference[index, :length]
            actual_sample = result[index, :length]
            if not torch.isfinite(actual_sample).all():
                raise AssertionError(f'{field}: non-finite output for sample {index}')
            torch.testing.assert_close(actual_sample, expected_sample, rtol=rtol, atol=atol,
                                       msg=lambda message: f'{field}, sample {index}: {message}')
            max_error = max(max_error, (actual_sample - expected_sample).abs().max().item())
        print(f'  {field}: samples={len(lengths)}, max_abs_error={max_error:.6g}', flush=True)
    if not check_logits and actual.get('logits') is not None:
        raise AssertionError('return_logits=False unexpectedly returned logits')


def main():
    args = parse_args()
    adapter_name = None
    hook = None
    forward_calls = []
    if args.local:
        from twinkle.model import TransformersModel
        model = TransformersModel(model_id=args.model_id, mixed_precision='no')
        model.set_loss('CrossEntropyLoss')
        model.set_processor('InputProcessor', padding_free=args.padding_free, padding_side='right')
        model._lazy_wrap_model()
        hook = model.model.register_forward_pre_hook(lambda module, inputs: forward_calls.append(1))
    else:
        from peft import LoraConfig
        from twinkle import init_twinkle_client
        from twinkle_client.model import MultiLoraTransformersModel
        init_twinkle_client(base_url=args.base_url, api_key=args.api_key)
        model = MultiLoraTransformersModel(model_id=args.model_id)
        adapter_name = f'e2e-forward-only-{uuid.uuid4().hex[:10]}'
        model.add_adapter_to_model(adapter_name, LoraConfig(target_modules='all-linear'))

    try:
        if not args.local:
            model.set_processor('InputProcessor', padding_free=args.padding_free, padding_side='right')
            model.set_loss('CrossEntropyLoss')
        inputs = build_inputs(args.samples)
        lengths = [len(sample['input_ids']) for sample in inputs]
        # A token budget that is valid even for a single longest sample.
        budget = max(lengths) * args.micro_batch_size
        options = {'return_logits': args.check_logits}

        def forward(label, config):
            start = time.monotonic()
            previous_calls = len(forward_calls)
            response = model.forward_only(inputs=deepcopy(inputs), **options, **config)
            output = response if args.local else response.result
            if not isinstance(output, dict):
                raise AssertionError(f'{label}: expected an output dict, got {type(output).__name__}')
            calls = len(forward_calls) - previous_calls
            if args.local:
                if (config and calls <= 1) or (not config and calls != 1):
                    raise AssertionError(f'{label}: unexpected model forward count {calls}')
            suffix = f', model_forwards={calls}' if args.local else ''
            print(f'{label}: {time.monotonic() - start:.2f}s{suffix}', flush=True)
            return output

        baseline = forward('unsplit', {})
        configs = [('static', {'micro_batch_size': args.micro_batch_size})]
        configs.extend((algorithm, {
            'micro_batch_size': args.micro_batch_size, 'dynamic_batching': True,
            'max_tokens_per_micro_batch': budget, 'packing_algorithm': algorithm,
        }) for algorithm in ('ffd', 'kk'))
        for label, config in configs:
            output = forward(label, config)
            assert_same_outputs(baseline, output, lengths, check_logits=args.check_logits,
                                rtol=args.rtol, atol=args.atol)
        output = forward('unsplit-after-micro-batching', {})
        assert_same_outputs(baseline, output, lengths, check_logits=args.check_logits,
                            rtol=args.rtol, atol=args.atol)
        print('ALL FORWARD_ONLY MICRO-BATCH CHECKS PASSED', flush=True)
    finally:
        if hook is not None:
            hook.remove()
        if adapter_name is not None:
            model.remove_adapter(adapter_name)


if __name__ == '__main__':
    main()
