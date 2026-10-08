"""CPU conversion/worker-hook tests, NOT Ascend transport or runtime acceptance."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from twinkle.sampler.vllm_sampler.dsv4_lora_ascend import (
    QUAROT_RECIPE, DeepSeekV4AscendQuaRotLoraAdapter, transform_quarot_pair)
from twinkle.sampler.vllm_sampler.dsv4_lora import NORMAL_MODULE_RULES, PREFIX
from twinkle.sampler.vllm_sampler.weight_sync import create_weight_adapter

EXPECTED_NORMAL_QUAROT_RULES = {
    'self_attn.compressor.indexer.weights_proj': 'attention_input',
    'self_attn.compressor.indexer.scorer.weights_proj': 'attention_input',
    'self_attn.compressor.indexer.q_b_proj': 'identity',
    'self_attn.compressor.indexer.kv_proj': 'attention_input',
    'self_attn.compressor.indexer.gate_proj': 'attention_input',
    'self_attn.compressor.kv_proj': 'attention_input',
    'self_attn.compressor.gate_proj': 'attention_input',
    'self_attn.q_a_proj': 'attention_input',
    'self_attn.q_b_proj': 'identity',
    'self_attn.kv_proj': 'attention_input',
    'self_attn.o_a_proj': 'identity',
    'self_attn.o_b_proj': 'hidden_output',
    'mlp.shared_experts.gate_proj': 'ffn_input',
    'mlp.shared_experts.down_proj': 'hidden_output',
    'mlp.shared_experts.up_proj': 'ffn_input',
}


def test_ascend_rules_preserve_independent_mapping_and_rotation_recipe():
    from cookbook.rl.grpo.convert_twinkle_dsv4_lora_for_vllm import NORMAL_MODULE_MAPPING
    assert {suffix: rule.quarot for suffix, rule in NORMAL_MODULE_RULES.items()} == EXPECTED_NORMAL_QUAROT_RULES
    for suffix, rule in NORMAL_MODULE_RULES.items():
        mapped = NORMAL_MODULE_MAPPING[suffix]
        if mapped.startswith('attn.'):
            expected = 'self_attn.' + mapped.removeprefix('attn.')
        else:
            expected = 'mlp.' + mapped.removeprefix('ffn.')
            parent, projection = expected.rsplit('.', 1)
            expected = parent + '.' + {'w1': 'gate_proj', 'w2': 'down_proj', 'w3': 'up_proj'}[projection]
        assert rule.ascend == expected


@pytest.fixture
def recipe(tmp_path):
    source, quant = tmp_path / 'training', tmp_path / 'rollout'
    source.mkdir()
    (quant / 'optional').mkdir(parents=True)
    dims = dict(model_type='deepseek_v4', num_hidden_layers=2, n_routed_experts=3,
                hidden_size=8, moe_intermediate_size=4)
    for path in (source, quant):
        (path / 'config.json').write_text(json.dumps(dims))
    rng = torch.Generator().manual_seed(17)
    rotation = torch.linalg.qr(torch.randn(8, 8, generator=rng))[0].contiguous()
    gammas = {layer: torch.rand(8, generator=rng) + 0.5 for layer in range(2)}
    attn_gammas = {layer: torch.rand(8, generator=rng) + 0.5 for layer in range(2)}
    save_file({'global_rotation': rotation}, str(quant / 'optional/quarot.safetensors'))
    source_weights = {f'model.layers.{layer}.post_attention_layernorm.weight': gamma
                      for layer, gamma in gammas.items()}
    source_weights.update({f'model.layers.{layer}.input_layernorm.weight': gamma
                           for layer, gamma in attn_gammas.items()})
    save_file(source_weights, str(source / 'model.safetensors'))
    quant_weights = {f'layers.{layer}.ffn_norm.weight': torch.ones(8) for layer in range(2)}
    quant_weights.update({f'layers.{layer}.attn_norm.weight': torch.ones(8) for layer in range(2)})
    description = dict(optional=dict(quarot=dict(rotation_map=dict(global_rotation='optional/quarot.safetensors'))))
    for layer in range(2):
        for expert in range(3):
            for proj in ('w1', 'w2', 'w3'):
                key = f'layers.{layer}.ffn.experts.{expert}.{proj}.weight'
                quant_weights[key] = torch.zeros((8, 4) if proj == 'w2' else (4, 8), dtype=torch.int8)
                description[key] = 'W8A8_DYNAMIC'
    save_file(quant_weights, str(quant / 'model.safetensors'))
    (quant / 'quant_model_description.json').write_text(json.dumps(description))
    target = dict(device='npu:0', model_path=str(quant), model_config=dims, enable_lora=True,
                  max_lora_rank=8, lora_dtype='torch.bfloat16', pp_size=1, ep_enabled=False, tp_size=8)
    options = dict(training_base_model=str(source), recipe=QUAROT_RECIPE)
    peft = dict(r=2, lora_alpha=8, use_rslora=False, target_modules=[],
                target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'])
    return types.SimpleNamespace(source=source, quant=quant, target=target, options=options,
                                 peft=peft, rotation=rotation, gammas=gammas,
                                 attn_gammas=attn_gammas, description=description)


def adapter_config(recipe):
    return dict(class_path='twinkle.sampler.vllm_sampler.dsv4_lora_ascend.DeepSeekV4AscendQuaRotLoraAdapter',
                options=recipe.options)


def weights(dtype=torch.bfloat16):
    rng = torch.Generator().manual_seed(3)
    result = []
    for layer in range(2):
        for proj, ins, outs in [('gate_up_proj', 8, 8), ('down_proj', 4, 8)]:
            name = f'model.layers.{layer}.mlp.experts.{proj}'
            result.extend([(f'{name}.lora_A.weight', torch.randn(3, 2, ins, generator=rng).to(dtype)),
                           (f'{name}.lora_B.weight', torch.randn(3, outs, 2, generator=rng).to(dtype))])
    return result


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_ascend_matches_independent_formula_and_repeated_updates(recipe, dtype):
    recipe.target['lora_dtype'] = str(dtype)
    adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
    source = weights(dtype)
    original = {name: tensor.clone() for name, tensor in source}
    expected = {}
    for index in range(0, len(source), 2):
        name, a = source[index]
        _, b = source[index + 1]
        layer = int(name.split('.')[2])
        if 'gate_up_proj' in name:
            a = ((a.float() * recipe.gammas[layer]) @ recipe.rotation).to(dtype)
            projections = [('w1', b[:, :4]), ('w3', b[:, 4:])]
        else:
            b = (recipe.rotation.T @ b.float()).to(dtype)
            projections = [('w2', b)]
        for expert in range(3):
            for proj, bp in projections:
                prefix = f'base_model.model.model.layers.{layer}.mlp.experts.{expert}.{proj}'
                expected[f'{prefix}.lora_A.weight'] = a[expert]
                expected[f'{prefix}.lora_B.weight'] = bp[expert]
    # Reordering, repeated A -> B -> A updates and actual-rank/scaling preservation.
    for factor in (1, 2, 1):
        incoming = [(name, value * factor if '.lora_B.' in name else value) for name, value in reversed(source)]
        actual, config = adapter.process(incoming, recipe.peft)
        actual = dict(actual)
        assert len(actual) == 36 and actual.keys() == expected.keys()
        for name, value in actual.items():
            assert value.dtype == dtype and value.device.type == 'cpu'
            assert torch.equal(value, expected[name] * (factor if '.lora_B.' in name else 1))
        assert config['r'] == 2 and config['lora_alpha'] == 8 and not config['use_rslora']
        assert config['target_parameters'] is None and config['target_modules'] == ['experts']
        assert not hasattr(adapter, 'pending')
        assert all(torch.equal(value, original[name]) for name, value in source)


@pytest.mark.parametrize('suffix,rule', NORMAL_MODULE_RULES.items())
def test_ordinary_quarot_coordinate_rules(recipe, suffix, rule):
    # Algebraic conversion check; real W8A8 forward parity remains untested.
    recipe.peft['target_modules'] = [suffix]
    recipe.peft['target_parameters'] = []
    a_in = 8 if rule.quarot in ('attention_input', 'ffn_input') else 4
    b_out = 8 if rule.quarot == 'hidden_output' else 4
    a = torch.arange(2 * a_in, dtype=torch.float32).reshape(2, a_in).to(torch.bfloat16) / 10
    b = torch.arange(b_out * 2, dtype=torch.float32).reshape(b_out, 2).to(torch.bfloat16) / 10
    adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
    source_name = f'model.layers.0.{suffix}'
    converted, config = adapter.process([
        (f'{source_name}.lora_B.weight', b), (f'{source_name}.lora_A.weight', a)], recipe.peft)
    converted = dict(converted)
    target_suffix = rule.ascend
    target_name = f'{PREFIX}.0.{target_suffix}'
    if rule.quarot == 'attention_input':
        expected_a = ((a.float() * recipe.attn_gammas[0]) @ recipe.rotation).to(a.dtype)
    elif rule.quarot == 'ffn_input':
        expected_a = ((a.float() * recipe.gammas[0]) @ recipe.rotation).to(a.dtype)
    else:
        expected_a = a
    expected_b = (recipe.rotation.T @ b.float()).to(b.dtype) if rule.quarot == 'hidden_output' else b
    assert torch.equal(converted[f'{target_name}.lora_A.weight'], expected_a)
    assert torch.equal(converted[f'{target_name}.lora_B.weight'], expected_b)
    runtime = target_suffix.rsplit('.', 1)[-1]
    assert config['target_modules'] == [runtime]


def test_attention_requires_norm_fusion(recipe):
    recipe.peft['target_modules'] = ['q_a_proj']
    recipe.peft['target_parameters'] = []
    quant_weights = load_file(str(recipe.quant / 'model.safetensors'))
    quant_weights['layers.0.attn_norm.weight'][0] = 2
    save_file(quant_weights, str(recipe.quant / 'model.safetensors'))
    adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
    name = 'model.layers.0.self_attn.q_a_proj'
    with pytest.raises(ValueError, match='attention norm'):
        adapter.process([(f'{name}.lora_A.weight', torch.ones(2, 8, dtype=torch.bfloat16)),
                         (f'{name}.lora_B.weight', torch.ones(4, 2, dtype=torch.bfloat16))], recipe.peft)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_parity_with_validated_offline_quarot_converter(recipe, tmp_path, dtype):
    converter = Path(__file__).resolve().parents[2] / 'output/convert_twinkle_dsv4_lora_for_vllm.py'
    if not converter.is_file() or not converter.with_name('diagnose_dsv4_quarot.py').is_file():
        pytest.skip('Optional local offline oracle unavailable; independent formula tests still run')
    source_dir = tmp_path / 'adapter'
    source_dir.mkdir()
    (source_dir / 'adapter_config.json').write_text(json.dumps(recipe.peft))
    save_file({('base_model.model.base_model.model.' + name.replace('.gate_up_proj', '.base_layer')
                .replace('.down_proj', '')): value for name, value in weights(dtype)},
              str(source_dir / 'adapter_model.safetensors'))
    subprocess.run([sys.executable, str(converter), str(source_dir), str(tmp_path / 'converted'),
                    '--backend', 'ascend', '--format', '2d', '--base-model', str(recipe.source),
                    '--quarot-model', str(recipe.quant)], check=True, capture_output=True, timeout=60)
    oracle = load_file(str(tmp_path / 'converted/adapter_model.safetensors'))
    recipe.target['lora_dtype'] = str(dtype)
    adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
    actual, config = adapter.process(weights(dtype), recipe.peft)
    actual = dict(actual)
    assert actual.keys() == oracle.keys()
    assert all(actual[name].dtype == oracle[name].dtype and torch.equal(actual[name], oracle[name]) for name in actual)
    oracle_config = json.loads((tmp_path / 'converted/adapter_config.json').read_text())
    for name in ('r', 'lora_alpha', 'use_rslora', 'target_modules', 'target_parameters'):
        assert config[name] == oracle_config[name]


@pytest.mark.parametrize('fault', ['recipe', 'extra_option', 'cuda', 'pp', 'ep', 'enable_lora', 'model_path',
                                   'dimensions', 'rotation_missing', 'rotation_nonfinite', 'rotation_shape',
                                   'not_orthogonal', 'gamma_nonfinite', 'gamma_shape', 'not_norm_fused',
                                   'static_quant', 'missing_norm', 'path_escape'])
def test_invalid_recipe_rejected_at_initialization(recipe, fault, tmp_path):
    if fault == 'recipe':
        recipe.options['recipe'] = 'guess'
    elif fault == 'extra_option':
        recipe.options['quantized_base'] = '/some/different/base'
    elif fault == 'cuda':
        recipe.target['device'] = 'cuda:0'
    elif fault == 'pp':
        recipe.target['pp_size'] = 2
    elif fault == 'ep':
        recipe.target['ep_enabled'] = True
    elif fault == 'enable_lora':
        recipe.target['enable_lora'] = False
    elif fault == 'model_path':
        recipe.target['model_path'] = None
    elif fault == 'dimensions':
        (recipe.source / 'config.json').write_text(json.dumps(recipe.target['model_config'] | {'hidden_size': 16}))
    elif fault.startswith('rotation_') or fault == 'not_orthogonal':
        if fault == 'rotation_missing':
            recipe.description.pop('optional')
        else:
            q = recipe.rotation.clone()
            if fault == 'rotation_nonfinite':
                q[0, 0] = float('nan')
            elif fault == 'rotation_shape':
                q = q[:4]
            else:
                q *= 2
            save_file({'global_rotation': q}, str(recipe.quant / 'optional/quarot.safetensors'))
    elif fault.startswith('gamma_') or fault == 'missing_norm':
        values = load_file(str(recipe.source / 'model.safetensors'))
        key = 'model.layers.0.post_attention_layernorm.weight'
        if fault == 'gamma_nonfinite':
            values[key][0] = float('inf')
        elif fault == 'gamma_shape':
            values[key] = values[key][:4]
        else:
            del values[key]
        save_file(values, str(recipe.source / 'model.safetensors'))
    elif fault == 'not_norm_fused':
        values = load_file(str(recipe.quant / 'model.safetensors'))
        values['layers.0.ffn_norm.weight'][0] = 2
        save_file(values, str(recipe.quant / 'model.safetensors'))
    elif fault == 'static_quant':
        recipe.description['layers.1.ffn.experts.2.w3.weight'] = 'W8A8'
    elif fault == 'path_escape':
        save_file({'global_rotation': recipe.rotation}, str(tmp_path / 'outside.safetensors'))
        recipe.description['optional']['quarot']['rotation_map']['global_rotation'] = '../outside.safetensors'
    (recipe.quant / 'quant_model_description.json').write_text(json.dumps(recipe.description))
    with pytest.raises((ValueError, FileNotFoundError)):
        create_weight_adapter(adapter_config(recipe), recipe.target)


def test_only_constants_read_from_index_and_cached_in_worker(recipe, monkeypatch):
    import twinkle.sampler.vllm_sampler.dsv4_lora_ascend as module
    import twinkle.patch.vllm_lora_weights as patch
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    for directory in (recipe.source, recipe.quant):
        mapping = {key: 'model.safetensors' for key in load_file(str(directory / 'model.safetensors'))}
        (directory / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
    original_open = module.safe_open
    reads = []

    class HeaderReader:
        def __init__(self, *args, **kwargs):
            self.handle = original_open(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.__exit__(*args)
        def get_slice(self, key):
            return self.handle.get_slice(key)
        def get_tensor(self, key):
            reads.append(key)
            assert key == 'global_rotation' or 'norm.weight' in key
            return self.handle.get_tensor(key)

    monkeypatch.setattr(module, 'safe_open', HeaderReader)
    monkeypatch.setattr(patch, 'TensorLoRARequest', types.SimpleNamespace)
    worker = object.__new__(TwinkleWorkerExtension)
    worker.device = 'npu:0'  # No torch_npu/hardware needed for CPU conversion tests.
    worker.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model=str(recipe.quant),
                                          hf_config=types.SimpleNamespace(**recipe.target['model_config'])),
        lora_config=types.SimpleNamespace(max_lora_rank=8, lora_dtype=torch.bfloat16),
        parallel_config=types.SimpleNamespace(pipeline_parallel_size=1, enable_expert_parallel=False,
                                             tensor_parallel_size=8))
    installed = []
    worker.add_lora = installed.append
    for _ in range(3):
        worker._load_weights(weights(), recipe.peft, False, lora_only=True, weight_adapter=adapter_config(recipe))
    assert len(reads) == 5  # Q + two original gammas + two quantized norms, once only.
    assert len(installed) == 3 and all(request.load_inplace for request in installed)
    assert all('.mlp.experts.' in key for key in installed[-1].lora_tensors)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'nonfinite', 'overflow', 'dtype', 'unknown_module'])
def test_update_failure_does_not_install_and_preserves_constants(recipe, monkeypatch, fault):
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
    rotation, gammas = adapter.rotation, adapter.gammas
    source = weights()
    config = dict(recipe.peft)
    if fault == 'missing':
        source.pop()
    elif fault == 'duplicate':
        source.append(source[0])
    elif fault == 'nonfinite':
        source[0][1][0, 0, 0] = float('nan')
    elif fault == 'overflow':
        adapter.gammas[0] = torch.full((8,), torch.finfo(torch.float32).max)
    elif fault == 'dtype':
        source[0] = source[0][0], source[0][1].float()
    else:
        config['target_modules'] = ['q_a_proj']
        name = 'model.layers.0.self_attn.unknown_proj'
        source.extend([(f'{name}.lora_A.weight', torch.ones(2, 8, dtype=torch.bfloat16)),
                       (f'{name}.lora_B.weight', torch.ones(4, 2, dtype=torch.bfloat16))])
    worker = object.__new__(TwinkleWorkerExtension)
    worker._get_weight_adapter = lambda _: adapter
    worker.add_lora = lambda _: pytest.fail('Conversion failure must not reach the loader')
    with pytest.raises(ValueError):
        worker._load_weights(source, config, False, lora_only=True, weight_adapter=adapter_config(recipe))
    assert not hasattr(adapter, 'pending')
    assert adapter.rotation is rotation and adapter.gammas is gammas


@pytest.mark.parametrize('projection', ['gate_up_proj', 'down_proj'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_numeric_batches_and_unchanged_projection_half(recipe, projection, dtype):
    a = torch.randn(19, 2, 8 if projection == 'gate_up_proj' else 4).to(dtype)
    b = torch.randn(19, 8, 2).to(dtype)
    new_a, new_b = transform_quarot_pair(a, b, projection, recipe.gammas[0], recipe.rotation)
    if projection == 'gate_up_proj':
        assert torch.equal(new_a, ((a.float() * recipe.gammas[0]) @ recipe.rotation).to(dtype))
        assert new_b is b
    else:
        assert new_a is a
        assert torch.equal(new_b, (recipe.rotation.T @ b.float()).to(dtype))


def test_outer_autocast_does_not_change_fp32_recipe(recipe):
    adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
    source = weights()
    expected, _ = adapter.process(source, recipe.peft)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
        actual, _ = adapter.process(source, recipe.peft)
    expected = dict(expected)
    assert all(torch.equal(value, expected[name]) for name, value in actual)


def test_ascend_preserves_rslora_scaling(recipe):
    config = recipe.peft | {'use_rslora': True}
    adapter = create_weight_adapter(adapter_config(recipe), recipe.target)
    standard, _ = adapter.process(weights(), recipe.peft)
    actual, result_config = adapter.process(weights(), config)
    assert result_config['use_rslora'] is True
    assert result_config['r'] == 2 and result_config['lora_alpha'] == 8
    standard = dict(standard)
    assert all(torch.equal(value, standard[name]) for name, value in actual)
