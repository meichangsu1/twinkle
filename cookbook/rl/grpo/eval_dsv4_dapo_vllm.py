# Copyright (c) ModelScope Contributors. All rights reserved.
"""Evaluate DAPO-Math answers through a running vLLM chat-completions API.

Example:
    python -m cookbook.rl.grpo.eval_dsv4_dapo_vllm \
        --dataset /highcode/shared_data/DAPO-Math-17k \
        --base-url http://127.0.0.1:8000/v1 --model deepseek-v4 \
        --limit 100 --output-dir /highcode/shared_data/dsv4_logs/dapo_eval_100

The score uses the same DAPOMathAccuracyReward as dsv4_lora_h800.py.
"""

import argparse
import importlib.util
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ANSWER_LINE = re.compile(r'^\s*Answer:\s*\S+', re.IGNORECASE | re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=os.environ.get('DAPO_PATH'),
                        help='DAPO Parquet file or its directory (also DAPO_PATH)')
    parser.add_argument('--base-url', default=os.environ.get('VLLM_BASE_URL', 'http://127.0.0.1:8000/v1'),
                        help='vLLM OpenAI-compatible base URL')
    parser.add_argument('--model', default=os.environ.get('VLLM_MODEL', 'deepseek-v4'),
                        help='Served model name, or a loaded LoRA name')
    parser.add_argument('--output-dir', default='./dsv4_dapo_eval',
                        help='Parent directory; each run creates a new subdirectory')
    parser.add_argument('--limit', type=int, default=100, help='First N rows; 0 evaluates the entire file')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--workers', type=int, default=4, help='Concurrent HTTP requests')
    parser.add_argument('--timeout-seconds', type=float, default=900.0)
    parser.add_argument('--retries', type=int, default=2, help='Retries for 429, 5xx, and network errors')
    parser.add_argument('--enable-thinking', action='store_true',
                        help='Enable DeepSeek-V4 thinking; disabled by default to match GRPO')
    args = parser.parse_args()
    if not args.dataset:
        parser.error('--dataset or DAPO_PATH is required')
    if args.limit < 0 or args.max_tokens <= 0 or args.workers <= 0 or args.timeout_seconds <= 0 or args.retries < 0:
        parser.error('limit/retries must be nonnegative; max-tokens/workers/timeout-seconds must be positive')
    if not 0 <= args.temperature <= 2:
        parser.error('temperature must be between 0 and 2')
    return args


def resolve_dataset(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path /= 'dapo-math-17k.parquet'
    if not path.is_file() or path.suffix != '.parquet':
        raise FileNotFoundError(f'DAPO Parquet not found: {path}')
    return path


def iter_rows(path, limit):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    required = {'prompt', 'reward_model'}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f'DAPO Parquet is missing columns: {sorted(missing)}')
    columns = ['prompt', 'reward_model']
    if 'extra_info' in parquet.schema_arrow.names:
        columns.append('extra_info')
    index = 0
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            if limit and index >= limit:
                return
            messages = row['prompt']
            reward_model = row['reward_model']
            if not isinstance(messages, list) or not messages or not isinstance(reward_model, dict):
                raise ValueError(f'Invalid DAPO row {index}: prompt or reward_model is malformed')
            if not reward_model.get('ground_truth'):
                raise ValueError(f'Invalid DAPO row {index}: missing ground_truth')
            for message in messages:
                if message.get('role') not in ('system', 'developer', 'user', 'assistant') or not isinstance(
                        message.get('content'), str):
                    raise ValueError(f'Invalid DAPO row {index}: malformed message')
            yield index, {
                'messages': [{'role': item['role'], 'content': item['content']} for item in messages],
                'ground_truth': str(reward_model['ground_truth']),
                'example_id': (row.get('extra_info') or {}).get('index'),
            }
            index += 1


def chat_completion(url, model, messages, max_tokens, temperature, enable_thinking, timeout, retries, api_key):
    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'top_p': 1.0,
        'max_tokens': max_tokens,
        'chat_template_kwargs': {'enable_thinking': enable_thinking},
    }
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            choice = data['choices'][0]
            message = choice['message']
            content = message.get('content') or ''
            if not isinstance(content, str):
                raise ValueError('vLLM returned a non-text message content')
            return {
                'response': content,
                'reasoning': message.get('reasoning') or message.get('reasoning_content'),
                'finish_reason': choice.get('finish_reason'),
                'usage': data.get('usage') or {},
                'error': None,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode('utf-8', errors='replace')
            error = f'HTTP {exc.code}: {detail}'
            retryable = exc.code == 429 or 500 <= exc.code < 600
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            retryable = True
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            error = f'Invalid vLLM response: {exc}'
            retryable = False
        if not retryable or attempt == retries:
            return {'response': None, 'reasoning': None, 'finish_reason': None, 'usage': {}, 'error': error}
        time.sleep(min(2**attempt, 8))
    raise AssertionError('unreachable')


def evaluate_one(item, *, url, args, api_key):
    index, row = item
    result = chat_completion(
        url, args.model, row['messages'], args.max_tokens, args.temperature,
        args.enable_thinking, args.timeout_seconds, args.retries, api_key)
    return {
        'row_index': index,
        'example_id': row['example_id'],
        'ground_truth': row['ground_truth'],
        'prompt': row['messages'],
        **result,
    }


def main():
    args = parse_args()
    dataset = resolve_dataset(args.dataset)
    # Import the training scorer only after parsing arguments: this script does
    # not instantiate Twinkle models, Ray, or any NPU runtime.
    import pyarrow.parquet  # noqa: F401
    from cookbook.rl.grpo.dsv4_dapo import DAPOMathAccuracyReward

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run_', dir=output_root))
    url = args.base_url.rstrip('/')
    if not url.endswith('/v1'):
        url += '/v1'
    url += '/chat/completions'
    api_key = os.environ.get('VLLM_API_KEY')

    reward_fn = DAPOMathAccuracyReward()
    scored = correct = formatted = capped = failed = 0
    total_completion_tokens = 0
    completion_token_reports = 0
    start = time.monotonic()
    predictions = output / 'predictions.jsonl'
    rows = iter_rows(dataset, args.limit)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, predictions.open('w', encoding='utf-8') as sink:
        while True:
            chunk = []
            for _ in range(args.workers * 4):
                try:
                    chunk.append(next(rows))
                except StopIteration:
                    break
            if not chunk:
                break
            for result in pool.map(lambda item: evaluate_one(item, url=url, args=args, api_key=api_key), chunk):
                if result['error'] is None:
                    reward_input = {
                        'messages': [{'role': 'assistant', 'content': result['response']}],
                        'user_data': [('ground_truth', result['ground_truth'])],
                    }
                    try:
                        result['correct'] = bool(reward_fn([reward_input])[0])
                    except Exception as exc:
                        result['error'] = f'Scoring failed: {type(exc).__name__}: {exc}'
                if result['error'] is None:
                    scored += 1
                    correct += result['correct']
                    formatted += bool(ANSWER_LINE.search(result['response']))
                    capped += result['finish_reason'] == 'length'
                    completion_tokens = result['usage'].get('completion_tokens')
                    if isinstance(completion_tokens, int):
                        total_completion_tokens += completion_tokens
                        completion_token_reports += 1
                else:
                    failed += 1
                sink.write(json.dumps(result, ensure_ascii=False) + '\n')
            sink.flush()
            print(f'evaluated={scored + failed} correct={correct} failed={failed}', flush=True)

    summary = {
        'dataset': str(dataset),
        'model': args.model,
        'endpoint': url,
        'enable_thinking': args.enable_thinking,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'limit': args.limit,
        'attempted': scored + failed,
        'scored': scored,
        'correct': correct,
        'failed': failed,
        'accuracy': correct / scored if scored and not failed else None,
        'accuracy_on_scored': correct / scored if scored else None,
        'answer_format_rate': formatted / scored if scored else None,
        'length_cap_rate': capped / scored if scored else None,
        'mean_completion_tokens': total_completion_tokens / completion_token_reports if completion_token_reports else None,
        'math_verify_available': importlib.util.find_spec('math_verify') is not None,
        'elapsed_seconds': time.monotonic() - start,
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f'Files: {output}', flush=True)
    if failed or not scored:
        print('Evaluation has failed requests/scoring or no scored rows; accuracy is unset.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
