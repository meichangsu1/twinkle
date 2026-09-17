"""CPU orchestration tests for the examples; not GPU/NPU hardware validation."""
import json
import types

import pytest
import torch

from cookbook.rl.grpo import dsv4_lora_h800 as example
from cookbook.rl.grpo import dsv4_lora_sync_audit as audit


@pytest.fixture
def setup_example(monkeypatch, tmp_path):
    for key in ('ACTOR_GPUS', 'ACTOR_NPUS', 'GPUS_PER_NODE', 'NPUS_PER_NODE',
                'ACTOR_EP', 'ROLLOUT_TP', 'ROLLOUT_START_RANK', 'ACTOR_PRECISION',
                'LORA_R', 'LORA_ALPHA', 'MAX_MODEL_LEN', 'MAX_NUM_SEQS',
                'MAX_NUM_BATCHED_TOKENS', 'GPU_MEMORY_UTILIZATION'):
        monkeypatch.delenv(key, raising=False)
    for key in ('ACTOR_MODEL', 'ROLLOUT_MODEL'):
        path = tmp_path / key
        path.mkdir()
        monkeypatch.setenv(key, str(path))
    monkeypatch.setattr(example, 'ensure_npu_backend', lambda: None)
    monkeypatch.setattr(torch, 'npu', types.SimpleNamespace(is_available=lambda: True), raising=False)
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained',
                        lambda *a, **kw: types.SimpleNamespace(use_cache=True, n_routed_experts=256))
    calls = {}
    monkeypatch.setattr(example.twinkle, 'initialize', lambda **kw: calls.update(initialize=kw))
    monkeypatch.setattr(example, 'CheckpointEngineManager',
                        lambda *a, **kw: calls.update(manager=kw) or 'manager')

    class Actor:
        def __init__(self, **kwargs):
            calls['actor'] = kwargs

        def add_adapter_to_model(self, name, config, **kwargs):
            calls['tenant'] = (name, config)

        def __getattr__(self, name):
            assert name.startswith('set_')
            return lambda *a, **kw: None

    class Sampler:
        def __init__(self, **kwargs):
            calls['sampler'] = kwargs

        def set_template(self, *args, **kwargs):
            calls['template'] = kwargs

    return calls, Actor, Sampler


@pytest.mark.parametrize('backend', ['nvidia', 'ascend'])
@pytest.mark.parametrize('precision', ['bf16', 'fp16'])
def test_example_backend_configuration(setup_example, monkeypatch, backend, precision):
    calls, actor, sampler = setup_example
    monkeypatch.setenv('ACTOR_PRECISION', precision)
    if backend == 'nvidia':
        monkeypatch.setenv('GPUS_PER_NODE', '4')
        monkeypatch.setenv('ACTOR_GPUS', '2')
        monkeypatch.setenv('ROLLOUT_START_RANK', '2')
        monkeypatch.setenv('ROLLOUT_TP', '2')
        devices, per_node, tp, start, platform, device = 2, 4, 2, 2, 'GPU', 'cuda'
    else:
        # A prior GPU run's environment must not override the NPU allocation.
        monkeypatch.setenv('GPUS_PER_NODE', '4')
        monkeypatch.setenv('ACTOR_GPUS', '99')
        devices, per_node, tp, start, platform, device = 4, 16, 4, 16, 'NPU', 'npu'
    _, _, manager = example.build_workers(actor, sampler, {'worker_extension_cls': 'test.Worker'}, backend=backend)
    assert manager == 'manager' and calls['manager'] == {'platform': platform}
    assert calls['initialize']['nproc_per_node'] == per_node
    groups = calls['initialize']['groups']
    assert groups[0].ranks == list(range(devices))
    assert groups[1].ranks == list(range(start, start + tp))
    assert all(group.device_type == platform for group in groups)
    assert str(calls['actor']['device_mesh'].device_type) == device
    assert str(calls['sampler']['device_mesh'].device_type) == device
    expected_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    assert calls['actor']['dtype'] == expected_dtype
    assert calls['actor']['memory_efficient_init']
    assert calls['actor']['lora_config'].target_modules == {'q_a_proj'}
    _, config = calls['tenant']
    assert not config.target_modules
    assert config.target_parameters == ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']
    engine = calls['sampler']['engine_args']
    assert engine['worker_extension_cls'] == 'test.Worker'
    assert engine['tensor_parallel_size'] == tp
    assert engine['lora_dtype'] == ('bfloat16' if precision == 'bf16' else 'float16')
    assert not engine['enable_prefix_caching']
    adapter = engine['weight_adapter']
    if backend == 'ascend':
        assert adapter['class_path'].endswith('DeepSeekV4AscendQuaRotLoraAdapter')
        assert adapter['options'] == dict(training_base_model=calls['actor']['model_id'],
                                          recipe='w8a8_dynamic_quarot_v1')
        assert engine['kv_cache_dtype'] == 'auto' and engine['block_size'] == 128
        assert engine['max_num_batched_tokens'] == 4096
    else:
        assert adapter['class_path'].endswith('DeepSeekV4NvidiaLoraAdapter')
        assert not adapter['options'] and engine['kv_cache_dtype'] == 'fp8'


@pytest.mark.parametrize('key,value', [('ACTOR_NPUS', '0'), ('ROLLOUT_TP', '0'),
                                      ('NPUS_PER_NODE', '0'), ('ROLLOUT_START_RANK', '2'),
                                      ('ROLLOUT_START_RANK', '15'), ('ACTOR_EP', '3')])
def test_bad_npu_topology_fails_before_ray(setup_example, monkeypatch, key, value):
    calls, actor, sampler = setup_example
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        example.build_workers(actor, sampler, backend='ascend')
    assert 'initialize' not in calls


def test_npu_requires_available_runtime(setup_example, monkeypatch):
    calls, actor, sampler = setup_example
    monkeypatch.setattr(torch.npu, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='available torch.npu'):
        example.build_workers(actor, sampler, backend='ascend')
    assert 'initialize' not in calls


def test_ep_must_divide_experts_before_ray(setup_example, monkeypatch):
    calls, actor, sampler = setup_example
    monkeypatch.setenv('ACTOR_NPUS', '3')
    with pytest.raises(ValueError, match='routed expert count'):
        example.build_workers(actor, sampler, backend='ascend')
    assert 'initialize' not in calls


@pytest.mark.parametrize('backend', ['nvidia', 'ascend'])
def test_independent_oracle_selects_backend_and_rotation(backend):
    command = audit.oracle_command('converter.py', 'source', 'output', 'bf16-base', backend, 'quantized-base')
    assert command[1:9] == ['converter.py', 'source', 'output', '--backend', backend, '--format', '2d', '--base-model']
    assert command[9] == 'bf16-base'
    assert command[10:] == (['--quarot-model', 'quantized-base'] if backend == 'ascend' else [])
    with pytest.raises(ValueError, match='actual QuaRot'):
        audit.oracle_command('converter.py', 'source', 'output', 'bf16-base', 'ascend')


@pytest.mark.parametrize('device', ['cuda', 'npu'])
def test_audit_memory_uses_worker_device(monkeypatch, device):
    calls = []
    api = types.SimpleNamespace(synchronize=lambda: calls.append(device), memory_allocated=lambda: 10,
                                memory_reserved=lambda: 20, max_memory_allocated=lambda: 30)
    monkeypatch.setattr(torch, device, api, raising=False)
    worker = types.SimpleNamespace(rank=2, device=types.SimpleNamespace(type=device))
    report = audit.AuditWorker.audit_memory(worker)
    assert calls == [device]
    assert report['device_type'] == device and report['rank'] == 2
    assert report['allocated'] == 10 and report['reserved'] == 20 and report['peak_allocated'] == 30


def test_npu_entrypoints_select_ascend(monkeypatch):
    from cookbook.rl.grpo import dsv4_lora_npu, dsv4_lora_sync_audit_npu
    calls = []
    monkeypatch.setattr(dsv4_lora_npu, 'run_grpo', lambda **kw: calls.append(kw))
    dsv4_lora_npu.main()
    assert calls.pop() == {'worker_builder': dsv4_lora_npu.build_workers}
    monkeypatch.setattr(dsv4_lora_npu, '_build_workers', lambda *a, **kw: calls.append(kw))
    dsv4_lora_npu.build_workers()
    assert calls.pop() == {'backend': 'ascend'}
    monkeypatch.setattr(dsv4_lora_sync_audit_npu, 'run_audit', lambda **kw: calls.append(kw))
    dsv4_lora_sync_audit_npu.main()
    assert calls.pop() == {'backend': 'ascend'}


@pytest.mark.parametrize('backend', ['nvidia', 'ascend'])
def test_audit_loop_passes_backend_and_records_results(monkeypatch, tmp_path, backend):
    monkeypatch.setenv('AUDIT_DIR', str(tmp_path / 'results'))
    monkeypatch.setenv('ACTOR_MODEL', 'bf16-base')
    monkeypatch.setenv('ROLLOUT_MODEL', 'quantized-base')
    monkeypatch.setenv('SYNC_REPEATS', '20')
    monkeypatch.setenv('MAX_GROWTH_MIB', '64')
    converter = tmp_path / 'converter.py'
    converter.touch()
    converter.with_name('diagnose_dsv4_quarot.py').touch()
    monkeypatch.setenv('OFFLINE_LORA_CONVERTER', str(converter))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained',
                        lambda *a, **kw: types.SimpleNamespace(encode=lambda text: [1, 2]))
    state, events = {}, []

    def fixture(value):
        state['value'] = value

    def oracle(path, converter, base, **kwargs):
        assert kwargs == {'backend': backend, 'rollout_path': 'quantized-base'}
        assert base == 'bf16-base'
        state[path] = state['value']

    def sync(**kwargs):
        assert kwargs == dict(merge_and_sync=False, lora_only=True, adapter_name='tenant_a')
        events.append('sync')

    actor = types.SimpleNamespace(fixture=fixture, write_file_oracle=oracle)
    sampler = types.SimpleNamespace(
        sample=lambda *a: [types.SimpleNamespace(prompt_logprobs=[None, state['value']])],
        memory=lambda: [dict(rank=0, allocated=10)],
        file_oracle_score=lambda path, tokens: [None, state[path]])

    def build(*args, **kwargs):
        assert kwargs == {'backend': backend}
        return actor, sampler, types.SimpleNamespace(sync_weights=sync)

    monkeypatch.setattr(audit, 'build_workers', build)
    audit.main(backend=backend)
    assert events == ['sync'] * 21
    report = json.loads((tmp_path / 'results' / 'summary.json').read_text())
    assert report['passed'] and report['backend'] == backend
    assert report['atol'] == report['rtol'] == 1e-4
    assert len(list((tmp_path / 'results').glob('sync_*.json'))) == 20


def test_missing_ascend_oracle_dependency_fails_before_loading(monkeypatch, tmp_path):
    converter = tmp_path / 'converter.py'
    converter.touch()
    monkeypatch.setenv('OFFLINE_LORA_CONVERTER', str(converter))
    monkeypatch.setenv('AUDIT_DIR', str(tmp_path / 'results'))
    monkeypatch.setenv('SYNC_REPEATS', '20')
    monkeypatch.setattr(audit, 'build_workers', lambda *a, **kw: pytest.fail('Do not load models'))
    with pytest.raises(FileNotFoundError, match='diagnose_dsv4_quarot.py'):
        audit.main(backend='ascend')
