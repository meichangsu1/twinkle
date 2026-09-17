# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in H800/Ascend acceptance harness; files exist ONLY as a test oracle.

Run as a module so Ray/vLLM subprocesses can import the diagnostic subclasses.
No training quality claim is made: this uses two deterministic, nonzero LoRAs.
"""
import json
import os
import time
import torch
from pathlib import Path

from twinkle import remote_class, remote_function
from twinkle.data_format import SamplingParams
from twinkle.model import MultiLoraTransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.sampler.vllm_sampler.vllm_worker_extension import VLLM_LORA_INT_ID, TwinkleWorkerExtension
from .dsv4_lora_h800 import TENANT, build_workers


def oracle_command(converter, source, destination, base_path, backend, rollout_path=None):
    """Use the independent offline converter, including its Ascend rotation."""
    import sys
    if backend not in ('nvidia', 'ascend'):
        raise ValueError('backend must be nvidia or ascend')
    command = [sys.executable, str(converter), str(source), str(destination),
               '--backend', backend, '--format', '2d', '--base-model', str(base_path)]
    if backend == 'ascend':
        if not rollout_path:
            raise ValueError('Ascend file oracle requires the actual QuaRot rollout base')
        command.extend(['--quarot-model', str(rollout_path)])
    return command


@remote_class()
class AuditActor(MultiLoraTransformersModel):

    @remote_function(dispatch='all', lazy_collect=False)
    def fixture(self, value):
        """Only modify LoRA B, never base weights; deterministic v0/v1/v0."""
        with torch.no_grad():
            for name, parameter in self.multi_adapter.target_parameter_manager.named_slot_parameters(TENANT):
                if '.lora_B.' in name:
                    local = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
                    local.fill_(value)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def write_file_oracle(self, path, converter, base_path, backend='nvidia', rollout_path=None):
        """The independent offline converter is used ONLY by this acceptance test."""
        import subprocess
        from safetensors.torch import save_file

        from twinkle import Platform
        # Use the unchanged checkpoint save export as the independent source,
        # not the new synchronization export/processor under test.
        weights = self._get_adapter_state_dict_for_save(TENANT)
        config = self.get_peft_config_dict(TENANT)
        if Platform.is_master():
            source = Path(path + '-source')
            source.mkdir(parents=True, exist_ok=False)
            save_file({name: tensor.contiguous() for name, tensor in weights.items()},
                      str(source / 'adapter_model.safetensors'))
            # PEFT's own serialization handles sets in target_modules.
            from peft import LoraConfig
            LoraConfig(**config).save_pretrained(source)
            # Do not import/call the new rollout converter to generate its own oracle.
            subprocess.run(oracle_command(converter, source, path, base_path, backend, rollout_path), check=True)


class AuditWorker(TwinkleWorkerExtension):

    def install_file_oracle(self, path):
        from vllm.lora.request import LoRARequest
        if VLLM_LORA_INT_ID in self.list_loras():
            self.remove_lora(VLLM_LORA_INT_ID)
        if not self.add_lora(LoRARequest(TENANT, VLLM_LORA_INT_ID, path)):
            raise RuntimeError('File oracle did not install')
        return self.rank

    def audit_memory(self):
        import resource
        device_type = self.device.type
        if device_type == 'cuda':
            api = torch.cuda
        elif device_type in ('npu', 'privateuseone'):
            api = torch.npu
        else:
            raise RuntimeError(f'Unsupported audit device: {device_type}')
        api.synchronize()
        return dict(
            rank=self.rank,
            device_type=device_type,
            allocated=api.memory_allocated(),
            reserved=api.memory_reserved(),
            peak_allocated=api.max_memory_allocated(),
            host_peak_kib=resource.getrusage(0).ru_maxrss)


@remote_class()
class AuditSampler(vLLMSampler):

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def memory(self):
        return self._run_in_loop(self.engine.engine.collective_rpc('audit_memory'))

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def file_oracle_score(self, path, tokens):

        async def run():
            from vllm.lora.request import LoRARequest
            await self.engine.engine.collective_rpc('install_file_oracle', kwargs={'path': path})
            await self.engine.reset_prefix_cache()
            # Diagnostic-only file oracle. The harness makes no further
            # sampling calls until the next checkpoint sync succeeds.
            self.engine._synced_lora_request = LoRARequest(TENANT, VLLM_LORA_INT_ID, path)
            result = await self.engine.sample(
                prompt=tokens,
                sampling_params=SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=1),
                lora_request=LoRARequest(TENANT, VLLM_LORA_INT_ID, path))
            self.engine.invalidate_synced_lora()
            return result.prompt_logprobs

        return self._run_in_loop(run())


def close(a, b):
    if len(a) != len(b) or any((x is None) != (y is None) for x, y in zip(a, b)):
        raise AssertionError('Prompt logprob alignment mismatch')
    a, b = torch.tensor([v for v in a if v is not None]), torch.tensor([v for v in b if v is not None])
    if not a.numel():
        raise AssertionError('No prompt logprobs returned')
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    return (a - b).abs().max().item()


def main(backend='nvidia'):
    if backend not in ('nvidia', 'ascend'):
        raise ValueError('backend must be nvidia or ascend')
    count = int(os.environ.get('SYNC_REPEATS', '20'))
    if count < 20:
        raise ValueError('Acceptance requires at least 20 synchronizations')
    output = Path(os.environ['AUDIT_DIR']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    converter = Path(os.environ['OFFLINE_LORA_CONVERTER']).resolve()
    if not converter.is_file():
        raise FileNotFoundError(converter)
    if backend == 'ascend' and not converter.with_name('diagnose_dsv4_quarot.py').is_file():
        raise FileNotFoundError(
            'Place diagnose_dsv4_quarot.py beside OFFLINE_LORA_CONVERTER for the Ascend file oracle')
    model, sampler, manager = build_workers(
        AuditActor, AuditSampler, {'worker_extension_cls': 'cookbook.rl.grpo.dsv4_lora_sync_audit.AuditWorker'},
        backend=backend)
    # Tokenize once, then use exactly these IDs for every memory/file comparison.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ['ACTOR_MODEL'], local_files_only=True, trust_remote_code=True)
    tokens = tokenizer.encode('你是谁？我是由ModelScope开发的人工智能语言模型，名为twinkle model。')
    paths = {}
    for label, value in [('v0', 0.01), ('v1', 0.02)]:
        model.fixture(value)
        paths[label] = str(output / label)
        model.write_file_oracle(paths[label], str(converter), os.environ['ACTOR_MODEL'],
                                backend=backend, rollout_path=os.environ['ROLLOUT_MODEL'])
    records = []
    for version in range(count):
        label = 'v1' if version == 1 else 'v0'
        model.fixture(0.02 if label == 'v1' else 0.01)
        sync_start = time.monotonic()
        manager.sync_weights(
            merge_and_sync=False,
            lora_only=True,
            adapter_name=TENANT)
        sync_seconds = time.monotonic() - sync_start
        memory_response = sampler.sample([{
            'input_ids': tokens,
            'labels': [-100] * len(tokens)
        }], SamplingParams(max_tokens=0, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=1))[0]
        memory = sampler.memory()
        file_scores = sampler.file_oracle_score(paths[label], tokens)
        error = close(memory_response.prompt_logprobs, file_scores)
        record = dict(
            backend=backend,
            version=version,
            fixture=label,
            max_abs_diff=error,
            memory=memory,
            memory_logprobs=memory_response.prompt_logprobs,
            file_logprobs=file_scores,
            sync_seconds=sync_seconds)
        records.append(record)
        (output / f'sync_{version:03d}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(f'version={version} fixture={label} file/memory max_abs_diff={error}', flush=True)
    close(records[0]['memory_logprobs'], records[2]['memory_logprobs'])
    if records[0]['memory_logprobs'] == records[1]['memory_logprobs']:
        raise AssertionError('v0/v1 fixtures did not change scores; acceptance is inconclusive')
    growth_limit = int(os.environ.get('MAX_GROWTH_MIB', '64')) << 20
    for before, after in zip(records[2]['memory'], records[-1]['memory']):
        if before['rank'] != after['rank'] or after['allocated'] - before['allocated'] > growth_limit:
            raise AssertionError(f'Device live allocation grew after warmup: {before} -> {after}')
    # Restore a real online policy and leave no diagnostic file-loaded policy live.
    manager.sync_weights(
        merge_and_sync=False,
        lora_only=True,
        adapter_name=TENANT)
    (output / 'summary.json').write_text(
        json.dumps(
            dict(
                passed=True,
                backend=backend,
                repeats=count,
                tokens=tokens,
                atol=1e-4,
                rtol=1e-4,
                scope='Real transport and runtime equivalence; synthetic LoRA, not training quality'),
            indent=2))


if __name__ == '__main__':
    main()
